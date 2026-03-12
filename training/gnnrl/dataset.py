#!/usr/bin/env python3
"""Replay dataset reader for Stage-B GNN+RL training."""

from __future__ import annotations

import bisect
import json
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100


@dataclass(frozen=True)
class ReplayDims:
    n_cell: int
    n_active_ue: int
    n_sched_ue: int
    n_tot_cell: int
    n_prg: int
    alloc_type: int
    n_edges: int
    action_alloc_len: int


@dataclass(frozen=True)
class ReplayFeatureDims:
    cell: int
    ue: int
    edge: int
    reward_terms: int


@dataclass(frozen=True)
class ReplaySpec:
    replay_dir: str
    records_path: str
    record_count: int
    record_bytes: int
    version: int
    has_action_mask_cell_ue: bool
    dims: ReplayDims
    feature_dims: ReplayFeatureDims


@dataclass
class ReplaySample:
    tti: torch.Tensor
    done: torch.Tensor
    reward_scalar: torch.Tensor
    reward_terms: torch.Tensor
    obs_cell_features: torch.Tensor
    obs_ue_features: torch.Tensor
    obs_edge_index: torch.Tensor
    obs_edge_attr: torch.Tensor
    action_ue_select: torch.Tensor
    action_prg_alloc: torch.Tensor
    action_mask_ue: torch.Tensor
    action_mask_cell_ue: torch.Tensor
    action_mask_prg_cell: torch.Tensor
    next_cell_features: torch.Tensor
    next_ue_features: torch.Tensor
    next_edge_attr: torch.Tensor
    target_ue_class: torch.Tensor
    target_prg_class: torch.Tensor


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_spec(replay_dir: str) -> ReplaySpec:
    replay_dir = os.path.abspath(replay_dir)
    meta = _load_json(os.path.join(replay_dir, "rl_replay_meta.json"))
    schema = _load_json(os.path.join(replay_dir, "rl_replay_schema.json"))

    dims = meta["dims"]
    feat = meta["feature_dims"]
    records_file = meta.get("records_file", "rl_replay_records.bin")
    records_path = os.path.join(replay_dir, records_file)

    if not os.path.isfile(records_path):
        raise FileNotFoundError(f"records file not found: {records_path}")

    layout = schema.get("record_layout", [])
    has_action_mask_cell_ue = any(entry.get("name") == "action_mask_cell_ue" for entry in layout)

    return ReplaySpec(
        replay_dir=replay_dir,
        records_path=records_path,
        record_count=int(meta["record_count"]),
        record_bytes=int(meta["record_bytes"]),
        version=int(meta.get("version", 1)),
        has_action_mask_cell_ue=bool(meta.get("has_action_mask_cell_ue", has_action_mask_cell_ue)),
        dims=ReplayDims(
            n_cell=int(dims["n_cell"]),
            n_active_ue=int(dims["n_active_ue"]),
            n_sched_ue=int(dims["n_sched_ue"]),
            n_tot_cell=int(dims["n_tot_cell"]),
            n_prg=int(dims["n_prg"]),
            alloc_type=int(dims["alloc_type"]),
            n_edges=int(dims["n_edges"]),
            action_alloc_len=int(dims["action_alloc_len"]),
        ),
        feature_dims=ReplayFeatureDims(
            cell=int(feat["cell"]),
            ue=int(feat["ue"]),
            edge=int(feat["edge"]),
            reward_terms=int(feat["reward_terms"]),
        ),
    )


class ReplayBinaryDataset(Dataset):
    """Random-access dataset over one or more replay shards."""

    def __init__(self, replay_dirs: Sequence[str]):
        if not replay_dirs:
            raise ValueError("replay_dirs is empty")

        self._specs: List[ReplaySpec] = [_parse_spec(p) for p in replay_dirs]
        self.spec = self._specs[0]
        self.dims = self.spec.dims
        self.feature_dims = self.spec.feature_dims

        for s in self._specs[1:]:
            if (
                s.dims != self.dims
                or s.feature_dims != self.feature_dims
                or s.record_bytes != self.spec.record_bytes
                or s.has_action_mask_cell_ue != self.spec.has_action_mask_cell_ue
            ):
                raise ValueError(
                    "all replay dirs must share identical dims/feature_dims/record_bytes; "
                    f"mismatch: {self.spec.replay_dir} vs {s.replay_dir}"
                )

        if self.dims.alloc_type != 0:
            raise NotImplementedError(
                f"M1 training currently supports alloc_type=0 only, got alloc_type={self.dims.alloc_type}"
            )

        self._prefix_counts: List[int] = []
        total = 0
        for s in self._specs:
            total += s.record_count
            self._prefix_counts.append(total)
        self._total = total

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int) -> Tuple[ReplaySpec, int]:
        if idx < 0:
            idx += self._total
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        shard_id = bisect.bisect_right(self._prefix_counts, idx)
        prev = 0 if shard_id == 0 else self._prefix_counts[shard_id - 1]
        local_idx = idx - prev
        return self._specs[shard_id], local_idx

    @staticmethod
    def _from_buffer(blob: bytes, offset: int, dtype: torch.dtype, count: int, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, int]:
        t = torch.frombuffer(blob, dtype=dtype, count=count, offset=offset)
        item_bytes = torch.empty((), dtype=dtype).element_size()
        offset += item_bytes * count
        return t.clone().reshape(shape), offset

    @staticmethod
    def _to_ue_target(action_ue_select: torch.Tensor, n_active_ue: int) -> torch.Tensor:
        tgt = action_ue_select.to(torch.int64)
        tgt = torch.where(tgt < 0, torch.full_like(tgt, n_active_ue), tgt)
        out_of_range = (tgt < 0) | (tgt > n_active_ue)
        tgt = torch.where(out_of_range, torch.full_like(tgt, IGNORE_INDEX), tgt)
        return tgt

    @staticmethod
    def _to_prg_target(action_prg_alloc: torch.Tensor, n_sched_ue: int) -> torch.Tensor:
        tgt = action_prg_alloc.to(torch.int64)
        tgt = torch.where(tgt < 0, torch.full_like(tgt, n_sched_ue), tgt)
        out_of_range = (tgt < 0) | (tgt > n_sched_ue)
        tgt = torch.where(out_of_range, torch.full_like(tgt, IGNORE_INDEX), tgt)
        return tgt

    @staticmethod
    def _infer_cell_ue_mask(
        action_mask_ue: torch.Tensor,
        n_cell: int,
        n_active_ue: int,
    ) -> torch.Tensor:
        cell_ue = torch.zeros((n_cell, n_active_ue), dtype=torch.uint8)

        if n_cell > 0 and (n_active_ue % n_cell) == 0:
            ue_per_cell = n_active_ue // n_cell
            for ue_idx in range(n_active_ue):
                c_idx = ue_idx // ue_per_cell
                cell_ue[c_idx, ue_idx] = 1
        else:
            cell_ue.fill_(1)

        if n_cell > 0:
            cell_ue &= action_mask_ue.view(1, n_active_ue)
        return cell_ue

    @staticmethod
    def _enforce_action_mask_consistency(
        action_ue_select: torch.Tensor,
        action_mask_ue: torch.Tensor,
        action_mask_cell_ue: torch.Tensor,
        n_cell: int,
        n_sched_ue: int,
        n_active_ue: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slots_per_cell = max(1, n_sched_ue // max(1, n_cell))
        for slot_idx in range(n_sched_ue):
            ue_idx = int(action_ue_select[slot_idx].item())
            if ue_idx < 0 or ue_idx >= n_active_ue:
                continue
            action_mask_ue[ue_idx] = 1
            c_idx = min(slot_idx // slots_per_cell, n_cell - 1)
            action_mask_cell_ue[c_idx, ue_idx] = 1

        return action_mask_ue, action_mask_cell_ue

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec, local_idx = self._locate(idx)
        with open(spec.records_path, "rb") as f:
            f.seek(local_idx * spec.record_bytes)
            blob = f.read(spec.record_bytes)

        if len(blob) != spec.record_bytes:
            raise RuntimeError(
                f"short read from {spec.records_path}: got={len(blob)} expected={spec.record_bytes} idx={local_idx}"
            )
        # torch.frombuffer warns on read-only bytes; use a writable copy once per record.
        blob = bytearray(blob)

        d = spec.dims
        fd = spec.feature_dims
        off = 0

        tti = struct.unpack_from("<i", blob, off)[0]
        off += 4

        done_u8 = struct.unpack_from("<B", blob, off)[0]
        off += 4  # done + 3-byte padding

        reward_scalar = struct.unpack_from("<f", blob, off)[0]
        off += 4

        reward_terms, off = self._from_buffer(blob, off, torch.float32, fd.reward_terms, (fd.reward_terms,))
        obs_cell_features, off = self._from_buffer(
            blob, off, torch.float32, d.n_cell * fd.cell, (d.n_cell, fd.cell)
        )
        obs_ue_features, off = self._from_buffer(
            blob, off, torch.float32, d.n_active_ue * fd.ue, (d.n_active_ue, fd.ue)
        )
        obs_edge_index, off = self._from_buffer(blob, off, torch.int16, d.n_edges * 2, (d.n_edges, 2))
        obs_edge_attr, off = self._from_buffer(
            blob, off, torch.float32, d.n_edges * fd.edge, (d.n_edges, fd.edge)
        )
        action_ue_select, off = self._from_buffer(blob, off, torch.int32, d.n_sched_ue, (d.n_sched_ue,))
        action_prg_alloc, off = self._from_buffer(blob, off, torch.int16, d.action_alloc_len, (d.action_alloc_len,))
        action_mask_ue, off = self._from_buffer(blob, off, torch.uint8, d.n_active_ue, (d.n_active_ue,))
        if spec.has_action_mask_cell_ue:
            action_mask_cell_ue, off = self._from_buffer(
                blob, off, torch.uint8, d.n_cell * d.n_active_ue, (d.n_cell, d.n_active_ue)
            )
            action_mask_ue, action_mask_cell_ue = self._enforce_action_mask_consistency(
                action_ue_select=action_ue_select,
                action_mask_ue=action_mask_ue,
                action_mask_cell_ue=action_mask_cell_ue,
                n_cell=d.n_cell,
                n_sched_ue=d.n_sched_ue,
                n_active_ue=d.n_active_ue,
            )
        else:
            action_mask_cell_ue = self._infer_cell_ue_mask(
                action_mask_ue=action_mask_ue,
                n_cell=d.n_cell,
                n_active_ue=d.n_active_ue,
            )
            action_mask_ue, action_mask_cell_ue = self._enforce_action_mask_consistency(
                action_ue_select=action_ue_select,
                action_mask_ue=action_mask_ue,
                action_mask_cell_ue=action_mask_cell_ue,
                n_cell=d.n_cell,
                n_sched_ue=d.n_sched_ue,
                n_active_ue=d.n_active_ue,
            )
        action_mask_prg_cell, off = self._from_buffer(
            blob, off, torch.uint8, d.n_cell * d.n_prg, (d.n_cell, d.n_prg)
        )
        next_cell_features, off = self._from_buffer(
            blob, off, torch.float32, d.n_cell * fd.cell, (d.n_cell, fd.cell)
        )
        next_ue_features, off = self._from_buffer(
            blob, off, torch.float32, d.n_active_ue * fd.ue, (d.n_active_ue, fd.ue)
        )
        next_edge_attr, off = self._from_buffer(
            blob, off, torch.float32, d.n_edges * fd.edge, (d.n_edges, fd.edge)
        )

        if off != spec.record_bytes:
            raise RuntimeError(
                f"record parse mismatch for {spec.records_path}: parsed={off} expected={spec.record_bytes}"
            )

        target_ue_class = self._to_ue_target(action_ue_select, d.n_active_ue)
        target_prg_class = self._to_prg_target(action_prg_alloc, d.n_sched_ue)

        return {
            "tti": torch.tensor(tti, dtype=torch.int32),
            "done": torch.tensor(done_u8 != 0, dtype=torch.bool),
            "reward_scalar": torch.tensor(reward_scalar, dtype=torch.float32),
            "reward_terms": reward_terms,
            "obs_cell_features": obs_cell_features,
            "obs_ue_features": obs_ue_features,
            "obs_edge_index": obs_edge_index.to(torch.int64),
            "obs_edge_attr": obs_edge_attr,
            "action_ue_select": action_ue_select.to(torch.int64),
            "action_prg_alloc": action_prg_alloc.to(torch.int64),
            "action_mask_ue": action_mask_ue.to(torch.bool),
            "action_mask_cell_ue": action_mask_cell_ue.to(torch.bool),
            "action_mask_prg_cell": action_mask_prg_cell.to(torch.bool),
            "next_cell_features": next_cell_features,
            "next_ue_features": next_ue_features,
            "next_edge_attr": next_edge_attr,
            "target_ue_class": target_ue_class,
            "target_prg_class": target_prg_class,
        }


def load_dataset(replay_dirs: Sequence[str]) -> ReplayBinaryDataset:
    return ReplayBinaryDataset(replay_dirs)
