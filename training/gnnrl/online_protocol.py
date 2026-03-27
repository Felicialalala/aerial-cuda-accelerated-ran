#!/usr/bin/env python3
"""Protocol definition for Stage-B online training bridge."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

MAGIC = 0x524C4E4F  # "ONLR"
VERSION = 1

MSG_RESET_REQ = 1
MSG_RESET_RSP = 2
MSG_STEP_REQ = 3
MSG_STEP_RSP = 4
MSG_CLOSE_REQ = 5
MSG_CLOSE_RSP = 6
MSG_ERROR_RSP = 7

HEADER_STRUCT = struct.Struct("<IHHI")
RESET_REQ_STRUCT = struct.Struct("<iiI")
STEP_REQ_HEAD_STRUCT = struct.Struct("<iII")
STATE_HEAD_STRUCT = struct.Struct("<iB3xfffffIIIIIIII")
ERROR_HEAD_STRUCT = struct.Struct("<iI")


@dataclass(frozen=True)
class EnvDims:
    n_cell: int
    n_active_ue: int
    n_sched_ue: int
    n_tot_cell: int
    n_prg: int
    n_edges: int
    alloc_type: int
    action_alloc_len: int


@dataclass(frozen=True)
class StateHeader:
    tti: int
    done: bool
    reward_scalar: float
    reward_terms: Tuple[float, float, float, float]
    dims: EnvDims


def pack_header(msg_type: int, payload_len: int) -> bytes:
    return HEADER_STRUCT.pack(MAGIC, VERSION, int(msg_type), int(payload_len))


def unpack_header(blob: bytes) -> Tuple[int, int]:
    magic, version, msg_type, payload_len = HEADER_STRUCT.unpack(blob)
    if magic != MAGIC:
        raise ValueError(f"invalid magic: {magic:#x}")
    if version != VERSION:
        raise ValueError(f"unsupported version: {version}")
    return int(msg_type), int(payload_len)


def pack_reset_req(seed: int, episode_horizon: int, flags: int = 0) -> bytes:
    return RESET_REQ_STRUCT.pack(int(seed), int(episode_horizon), int(flags))


def pack_step_req(step_idx: int, action_ue: bytes, action_prg: bytes, ue_len: int, prg_len: int) -> bytes:
    return STEP_REQ_HEAD_STRUCT.pack(int(step_idx), int(ue_len), int(prg_len)) + action_ue + action_prg


def unpack_state_header(payload: bytes) -> Tuple[StateHeader, int]:
    if len(payload) < STATE_HEAD_STRUCT.size:
        raise ValueError("state payload too small")
    vals = STATE_HEAD_STRUCT.unpack_from(payload, 0)
    (
        tti,
        done_u8,
        reward_scalar,
        r0,
        r1,
        r2,
        r3,
        n_cell,
        n_active_ue,
        n_sched_ue,
        n_tot_cell,
        n_prg,
        n_edges,
        alloc_type,
        action_alloc_len,
    ) = vals
    dims = EnvDims(
        n_cell=int(n_cell),
        n_active_ue=int(n_active_ue),
        n_sched_ue=int(n_sched_ue),
        n_tot_cell=int(n_tot_cell),
        n_prg=int(n_prg),
        n_edges=int(n_edges),
        alloc_type=int(alloc_type),
        action_alloc_len=int(action_alloc_len),
    )
    h = StateHeader(
        tti=int(tti),
        done=bool(done_u8),
        reward_scalar=float(reward_scalar),
        reward_terms=(float(r0), float(r1), float(r2), float(r3)),
        dims=dims,
    )
    return h, STATE_HEAD_STRUCT.size


def unpack_error_payload(payload: bytes) -> Tuple[int, str]:
    if len(payload) < ERROR_HEAD_STRUCT.size:
        return -1, "unknown error"
    code, msg_len = ERROR_HEAD_STRUCT.unpack_from(payload, 0)
    msg_off = ERROR_HEAD_STRUCT.size
    msg_end = min(len(payload), msg_off + int(msg_len))
    msg = payload[msg_off:msg_end].decode("utf-8", errors="replace")
    return int(code), msg
