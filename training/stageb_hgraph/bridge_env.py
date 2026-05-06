#!/usr/bin/env python3
"""Minimal online bridge client for the Stage-B sparse-graph stack."""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .online_protocol import (
    HEADER_STRUCT,
    MSG_CLOSE_REQ,
    MSG_CLOSE_RSP,
    MSG_ERROR_RSP,
    MSG_RESET_REQ,
    MSG_RESET_RSP,
    MSG_STEP_REQ,
    MSG_STEP_RSP,
    STATE_FLAG_HAS_ACCEPTED_PRE_PDSCH_UE_DIAGNOSTICS,
    STATE_FLAG_HAS_ACTUAL_UE_DIAGNOSTICS,
    STATE_FLAG_HAS_NATIVE_REJECT_DIAGNOSTICS,
    STATE_FLAG_HAS_NATIVE_SLOT_LAYOUT_DIAGNOSTICS,
    STATE_FLAG_HAS_POST_EQ_SINR,
    STATE_FLAG_HAS_TEACHER_ACTION,
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
class BridgeEnvConfig:
    socket_path: str
    connect_timeout_s: float = 20.0


class BridgeEnvClient:
    def __init__(self, cfg: BridgeEnvConfig):
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
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.cfg.socket_path)
                self._sock = sock
                return
            except OSError as exc:
                last_err = exc
                sock.close()
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
        header, off = unpack_state_header(payload)
        dims = header.dims

        def take(dtype: np.dtype, count: int, shape):
            nonlocal off
            nbytes = int(dtype.itemsize) * int(count)
            if off + nbytes > len(payload):
                raise RuntimeError("state payload truncated")
            arr = np.frombuffer(payload, dtype=dtype, count=count, offset=off).copy().reshape(shape)
            off += nbytes
            return arr

        obs_cell = take(_F32, dims.n_cell * dims.cell_feat_dim, (dims.n_cell, dims.cell_feat_dim))
        obs_ue = take(_F32, dims.n_active_ue * dims.ue_feat_dim, (dims.n_active_ue, dims.ue_feat_dim))
        obs_prg = take(_F32, dims.n_cell * dims.n_prg * dims.prg_feat_dim, (dims.n_cell, dims.n_prg, dims.prg_feat_dim))
        obs_edge_index = take(_I16, dims.n_edges * 2, (dims.n_edges, 2))
        obs_edge_attr = take(_F32, dims.n_edges * dims.edge_feat_dim, (dims.n_edges, dims.edge_feat_dim))
        action_mask_ue = take(_U8, dims.n_active_ue, (dims.n_active_ue,))
        action_mask_cell_ue = take(_U8, dims.n_cell * dims.n_active_ue, (dims.n_cell, dims.n_active_ue))
        action_mask_prg_cell = take(_U8, dims.n_cell * dims.n_prg, (dims.n_cell, dims.n_prg))

        post_eq_sinr = None
        if header.state_flags & STATE_FLAG_HAS_POST_EQ_SINR:
            if dims.post_eq_layer_dim <= 0:
                raise RuntimeError("state payload flags mark post_eq_sinr present, but post_eq_layer_dim <= 0")
            count = dims.n_active_ue * dims.n_prg * dims.post_eq_layer_dim
            post_eq_sinr = take(_F32, count, (dims.n_active_ue, dims.n_prg, dims.post_eq_layer_dim))

        teacher_action_ue = None
        teacher_action_prg = None
        if header.state_flags & STATE_FLAG_HAS_TEACHER_ACTION:
            teacher_action_ue = take(_I32, dims.n_sched_ue, (dims.n_sched_ue,))
            teacher_action_prg = take(_I16, dims.action_alloc_len, (dims.action_alloc_len,))

        accepted_pre_pdsch_ue_prg_count = None
        if header.state_flags & STATE_FLAG_HAS_ACCEPTED_PRE_PDSCH_UE_DIAGNOSTICS:
            accepted_pre_pdsch_ue_prg_count = take(_I32, dims.n_active_ue, (dims.n_active_ue,))

        native_reject_wrong_cell_slot_count = None
        native_reject_empty_slot_count = None
        native_reject_prg_mask_count = None
        native_reject_oob_slot_count = None
        native_reject_duplicate_ue_count = None
        if header.state_flags & STATE_FLAG_HAS_NATIVE_REJECT_DIAGNOSTICS:
            native_reject_wrong_cell_slot_count = take(_I32, dims.n_cell, (dims.n_cell,))
            native_reject_empty_slot_count = take(_I32, dims.n_cell, (dims.n_cell,))
            native_reject_prg_mask_count = take(_I32, dims.n_cell, (dims.n_cell,))
            native_reject_oob_slot_count = take(_I32, dims.n_cell, (dims.n_cell,))
            native_reject_duplicate_ue_count = take(_I32, dims.n_cell, (dims.n_cell,))

        native_slot_count_per_cell = None
        native_cell_slot_start = None
        native_slot_to_cell = None
        native_slot_valid = None
        if header.state_flags & STATE_FLAG_HAS_NATIVE_SLOT_LAYOUT_DIAGNOSTICS:
            native_slot_count_per_cell = take(_I32, dims.n_cell, (dims.n_cell,))
            native_cell_slot_start = take(_I32, dims.n_cell, (dims.n_cell,))
            native_slot_to_cell = take(_I32, dims.n_sched_ue, (dims.n_sched_ue,))
            native_slot_valid = take(_I32, dims.n_sched_ue, (dims.n_sched_ue,))

        actual_ue_prg_count = None
        actual_ue_served_bytes = None
        actual_ue_goodput_bytes = None
        actual_ue_tb_tx_count = None
        actual_ue_tb_err_count = None
        if header.state_flags & STATE_FLAG_HAS_ACTUAL_UE_DIAGNOSTICS:
            actual_ue_prg_count = take(_I32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_served_bytes = take(_F32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_goodput_bytes = take(_F32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_tb_tx_count = take(_I32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_tb_err_count = take(_I32, dims.n_active_ue, (dims.n_active_ue,))

        if off != len(payload):
            raise RuntimeError(f"state payload parse mismatch: parsed={off} total={len(payload)}")

        self._dims = dims
        obs = {
            "obs_cell_features": obs_cell,
            "obs_ue_features": obs_ue,
            "obs_prg_features": obs_prg,
            "obs_edge_index": obs_edge_index.astype(np.int64, copy=False),
            "obs_edge_attr": obs_edge_attr,
            "action_mask_ue": action_mask_ue.astype(np.bool_, copy=False),
            "action_mask_cell_ue": action_mask_cell_ue.astype(np.bool_, copy=False),
            "action_mask_prg_cell": action_mask_prg_cell.astype(np.bool_, copy=False),
            "n_sched_ue": np.int32(dims.n_sched_ue),
        }
        if post_eq_sinr is not None:
            obs["post_eq_sinr"] = post_eq_sinr
        if teacher_action_ue is not None and teacher_action_prg is not None:
            obs["teacher_action_ue_select"] = teacher_action_ue.astype(np.int32, copy=False)
            obs["teacher_action_prg_alloc"] = teacher_action_prg.astype(np.int16, copy=False)
        info = {
            "tti": header.tti,
            "reward_terms": header.reward_terms,
            "n_cell": dims.n_cell,
            "n_active_ue": dims.n_active_ue,
            "n_sched_ue": dims.n_sched_ue,
            "n_tot_cell": dims.n_tot_cell,
            "n_prg": dims.n_prg,
            "alloc_type": dims.alloc_type,
            "action_alloc_len": dims.action_alloc_len,
            "cell_feat_dim": dims.cell_feat_dim,
            "ue_feat_dim": dims.ue_feat_dim,
            "edge_feat_dim": dims.edge_feat_dim,
            "prg_feat_dim": dims.prg_feat_dim,
            "post_eq_layer_dim": int(dims.post_eq_layer_dim),
            "teacher_available": bool(header.state_flags & STATE_FLAG_HAS_TEACHER_ACTION),
            "accepted_pre_pdsch_ue_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_ACCEPTED_PRE_PDSCH_UE_DIAGNOSTICS
            ),
            "native_reject_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_NATIVE_REJECT_DIAGNOSTICS
            ),
            "native_slot_layout_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_NATIVE_SLOT_LAYOUT_DIAGNOSTICS
            ),
            "actual_ue_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_ACTUAL_UE_DIAGNOSTICS
            ),
        }
        if accepted_pre_pdsch_ue_prg_count is not None:
            info["accepted_pre_pdsch_ue_prg_count"] = accepted_pre_pdsch_ue_prg_count.astype(
                np.int32,
                copy=False,
            )
        if native_reject_wrong_cell_slot_count is not None:
            info.update(
                {
                    "native_reject_wrong_cell_slot_count": native_reject_wrong_cell_slot_count.astype(
                        np.int32,
                        copy=False,
                    ),
                    "native_reject_empty_slot_count": native_reject_empty_slot_count.astype(
                        np.int32,
                        copy=False,
                    ),
                    "native_reject_prg_mask_count": native_reject_prg_mask_count.astype(
                        np.int32,
                        copy=False,
                    ),
                    "native_reject_oob_slot_count": native_reject_oob_slot_count.astype(
                        np.int32,
                        copy=False,
                    ),
                    "native_reject_duplicate_ue_count": native_reject_duplicate_ue_count.astype(
                        np.int32,
                        copy=False,
                    ),
                }
            )
        if native_slot_count_per_cell is not None:
            info.update(
                {
                    "native_slot_count_per_cell": native_slot_count_per_cell.astype(np.int32, copy=False),
                    "native_cell_slot_start": native_cell_slot_start.astype(np.int32, copy=False),
                    "native_slot_to_cell": native_slot_to_cell.astype(np.int32, copy=False),
                    "native_slot_valid": native_slot_valid.astype(np.int32, copy=False),
                }
            )
        if actual_ue_prg_count is not None:
            info.update(
                {
                    "actual_ue_prg_count": actual_ue_prg_count.astype(np.int32, copy=False),
                    "actual_ue_served_bytes": actual_ue_served_bytes.astype(np.float32, copy=False),
                    "actual_ue_goodput_bytes": actual_ue_goodput_bytes.astype(np.float32, copy=False),
                    "actual_ue_tb_tx_count": actual_ue_tb_tx_count.astype(np.int32, copy=False),
                    "actual_ue_tb_err_count": actual_ue_tb_err_count.astype(np.int32, copy=False),
                }
            )
        return obs, float(header.reward_scalar), bool(header.done), info

    def reset(self, seed: int = 0, episode_horizon: int = 400):
        self.connect()
        self._step_idx = 0
        self._send(MSG_RESET_REQ, pack_reset_req(seed=seed, episode_horizon=episode_horizon, flags=0))
        rsp = self._recv(MSG_RESET_RSP)
        obs, reward, done, info = self._parse_state_payload(rsp)
        if done:
            raise RuntimeError("env returned done=true on reset")
        return obs, reward, info

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
        return obs, reward, done, info

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


__all__ = ["BridgeEnvClient", "BridgeEnvConfig"]
