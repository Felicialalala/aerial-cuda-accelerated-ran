#!/usr/bin/env python3
"""Gym-like client for Stage-B online training bridge."""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from training.gnnrl.online_protocol import (
    HEADER_STRUCT,
    MSG_CLOSE_REQ,
    MSG_CLOSE_RSP,
    MSG_ERROR_RSP,
    MSG_RESET_REQ,
    MSG_RESET_RSP,
    MSG_STEP_REQ,
    MSG_STEP_RSP,
    pack_header,
    pack_reset_req,
    pack_step_req,
    unpack_error_payload,
    unpack_header,
    unpack_state_header,
)


_I32 = np.dtype("<i4")
_I16 = np.dtype("<i2")
_F32 = np.dtype("<f4")
_U8 = np.dtype("u1")


@dataclass(frozen=True)
class AerialEnvConfig:
    socket_path: str
    connect_timeout_s: float = 20.0


class AerialEnvClient:
    def __init__(self, cfg: AerialEnvConfig):
        self.cfg = cfg
        self._sock: Optional[socket.socket] = None
        self._dims = None
        self._step_idx = 0
        self._closed = False

    @property
    def dims(self):
        return self._dims

    def connect(self) -> None:
        if self._sock is not None:
            return

        deadline = time.time() + max(0.1, self.cfg.connect_timeout_s)
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            if not os.path.exists(self.cfg.socket_path):
                time.sleep(0.05)
                continue
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.connect(self.cfg.socket_path)
                self._sock = s
                return
            except OSError as e:
                last_err = e
                s.close()
                time.sleep(0.05)

        raise RuntimeError(f"failed to connect socket={self.cfg.socket_path}, err={last_err}")

    def _recv_exact(self, n: int) -> bytes:
        assert self._sock is not None
        data = bytearray()
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise RuntimeError("socket closed while receiving")
            data.extend(chunk)
        return bytes(data)

    def _send(self, msg_type: int, payload: bytes) -> None:
        assert self._sock is not None
        self._sock.sendall(pack_header(msg_type, len(payload)) + payload)

    def _recv(self, expected_type: int) -> bytes:
        blob = self._recv_exact(HEADER_STRUCT.size)
        msg_type, payload_len = unpack_header(blob)
        payload = self._recv_exact(payload_len)
        if msg_type == MSG_ERROR_RSP:
            code, msg = unpack_error_payload(payload)
            raise RuntimeError(f"bridge error code={code}, msg={msg}")
        if msg_type != expected_type:
            raise RuntimeError(f"unexpected message type={msg_type}, expected={expected_type}")
        return payload

    def _parse_state_payload(self, payload: bytes) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        h, off = unpack_state_header(payload)
        dims = h.dims

        def take(dtype: np.dtype, count: int, shape):
            nonlocal off
            nbytes = int(dtype.itemsize) * int(count)
            if off + nbytes > len(payload):
                raise RuntimeError("state payload truncated")
            arr = np.frombuffer(payload, dtype=dtype, count=count, offset=off).copy().reshape(shape)
            off += nbytes
            return arr

        obs_cell = take(_F32, dims.n_cell * 5, (dims.n_cell, 5))
        obs_ue = take(_F32, dims.n_active_ue * 8, (dims.n_active_ue, 8))
        obs_edge_index = take(_I16, dims.n_edges * 2, (dims.n_edges, 2))
        obs_edge_attr = take(_F32, dims.n_edges * 2, (dims.n_edges, 2))
        action_mask_ue = take(_U8, dims.n_active_ue, (dims.n_active_ue,))
        action_mask_cell_ue = take(_U8, dims.n_cell * dims.n_active_ue, (dims.n_cell, dims.n_active_ue))
        action_mask_prg_cell = take(_U8, dims.n_cell * dims.n_prg, (dims.n_cell, dims.n_prg))

        if off != len(payload):
            raise RuntimeError(f"state payload has trailing bytes: {len(payload) - off}")

        self._dims = dims
        obs = {
            "obs_cell_features": obs_cell,
            "obs_ue_features": obs_ue,
            "obs_edge_index": obs_edge_index.astype(np.int64, copy=False),
            "obs_edge_attr": obs_edge_attr,
            "action_mask_ue": action_mask_ue.astype(np.bool_, copy=False),
            "action_mask_cell_ue": action_mask_cell_ue.astype(np.bool_, copy=False),
            "action_mask_prg_cell": action_mask_prg_cell.astype(np.bool_, copy=False),
        }
        info = {
            "tti": h.tti,
            "reward_terms": h.reward_terms,
            "n_cell": dims.n_cell,
            "n_active_ue": dims.n_active_ue,
            "n_sched_ue": dims.n_sched_ue,
            "n_tot_cell": dims.n_tot_cell,
            "n_prg": dims.n_prg,
            "alloc_type": dims.alloc_type,
        }
        return obs, h.reward_scalar, h.done, info

    def reset(self, seed: int = 0, episode_horizon: int = 400) -> Dict[str, np.ndarray]:
        self.connect()
        self._step_idx = 0
        payload = pack_reset_req(seed=seed, episode_horizon=episode_horizon, flags=0)
        self._send(MSG_RESET_REQ, payload)
        rsp = self._recv(MSG_RESET_RSP)
        obs, _reward, done, _info = self._parse_state_payload(rsp)
        if done:
            raise RuntimeError("env returned done=true on reset")
        return obs

    def step(self, action_ue_select: np.ndarray, action_prg_alloc: np.ndarray):
        if self._dims is None:
            raise RuntimeError("reset must be called before step")

        ue = np.asarray(action_ue_select, dtype=np.int32).reshape(-1)
        prg = np.asarray(action_prg_alloc, dtype=np.int16).reshape(-1)
        if ue.size != self._dims.n_sched_ue:
            raise ValueError(f"action_ue_select size mismatch: got={ue.size}, expected={self._dims.n_sched_ue}")
        if prg.size != self._dims.action_alloc_len:
            raise ValueError(f"action_prg_alloc size mismatch: got={prg.size}, expected={self._dims.action_alloc_len}")

        payload = pack_step_req(
            step_idx=self._step_idx,
            action_ue=ue.astype(_I32, copy=False).tobytes(order="C"),
            action_prg=prg.astype(_I16, copy=False).tobytes(order="C"),
            ue_len=int(ue.size),
            prg_len=int(prg.size),
        )
        self._send(MSG_STEP_REQ, payload)
        rsp = self._recv(MSG_STEP_RSP)
        obs, reward, done, info = self._parse_state_payload(rsp)
        self._step_idx += 1
        return obs, float(reward), bool(done), info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._sock is not None:
            try:
                self._send(MSG_CLOSE_REQ, b"")
                _ = self._recv(MSG_CLOSE_RSP)
            except Exception:
                pass
            try:
                self._sock.close()
            finally:
                self._sock = None


__all__ = ["AerialEnvConfig", "AerialEnvClient"]
