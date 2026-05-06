from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from .types import GraphFeatureSnapshot, NamedFeatureBlock, StaticMeta

_CELL_INSTANT_TB_ERR_IDX = 4

_UE_BUFFER_IDX = 0
_UE_AVG_RATE_IDX = 1
_UE_WB_SINR_IDX = 2
_UE_CQI_IDX = 3
_UE_RI_IDX = 4
_UE_TB_ERR_LAST_IDX = 5
_UE_NEW_DATA_IDX = 6
_UE_STALE_SLOTS_IDX = 7
_UE_HOL_DELAY_IDX = 8
_UE_TTL_SLACK_IDX = 9
_UE_RECENT_SCHED_RATIO_IDX = 10
_UE_RECENT_DEFICIT_IDX = 11

_PRG_TOP1_SINR_IDX = 0
_PRG_TOP2_GAP_IDX = 1
_PRG_PREV_ASSIGNED_IDX = 2
_PRG_REUSE_RATIO_IDX = 3
_PRG_NEIGHBOR_MAX_TOP1_IDX = 4
_PRG_NEIGHBOR_MEAN_TOP1_IDX = 5
_PRG_SAME_PRG_CONFLICT_IDX = 6
_PRG_ICI_PROXY_IDX = 7

_CELL_FEATURE_NAMES = (
    "cell_load_bytes",
    "active_ue_count",
    "mean_wb_sinr_lin",
    "mean_avg_rate_mbps",
    "tb_err_rate",
)

_UE_FEATURE_NAMES = (
    "buffer_bytes",
    "avg_rate_mbps",
    "wb_sinr_lin",
    "cqi",
    "ri",
    "tb_err_last",
    "new_data_flag",
    "stale_slots",
    "hol_delay_ms",
    "ttl_slack_ms",
    "recent_scheduled_ratio",
    "recent_goodput_deficit_norm",
)

_PRG_FEATURE_NAMES = (
    "top1_sinr_db",
    "top2_gap_db",
    "prev_prg_assigned",
    "prev_prg_reuse_ratio",
    "neighbor_max_top1_sinr_db",
    "neighbor_mean_top1_sinr_db",
    "same_prg_conflict_ratio",
    "ici_proxy",
)


@dataclass(frozen=True)
class SnapshotBuilderConfig:
    tb_err_ema_alpha: float = 0.10
    default_topology: str = "3cell"


def _to_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _infer_topology(n_cell: int, default_topology: str) -> str:
    if n_cell == 3:
        return "3cell"
    if n_cell == 7:
        return "7cell"
    return default_topology


def _build_default_cell_adjacency(n_cell: int) -> torch.Tensor:
    if n_cell <= 1:
        return torch.zeros((0, 2), dtype=torch.long)
    pairs = []
    for src in range(n_cell):
        for dst in range(n_cell):
            if src == dst:
                continue
            pairs.append((src, dst))
    return torch.tensor(pairs, dtype=torch.long)


def _build_default_cell_edge_attr(cell_adjacency: torch.Tensor) -> torch.Tensor:
    edge_count = int(cell_adjacency.shape[0])
    return torch.zeros((edge_count, 2), dtype=torch.float32)


class FeatureSnapshotBuilder:
    """Build graph snapshots while preserving raw online-bridge node features."""

    def __init__(self, cfg: SnapshotBuilderConfig | None = None):
        self.cfg = cfg or SnapshotBuilderConfig()
        self._cell_tb_err_ema: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._cell_tb_err_ema = None

    def build(self, obs: Mapping[str, Any]) -> GraphFeatureSnapshot:
        obs_cell = _to_tensor(obs["obs_cell_features"], dtype=torch.float32)
        obs_ue = _to_tensor(obs["obs_ue_features"], dtype=torch.float32)
        obs_prg = _to_tensor(obs["obs_prg_features"], dtype=torch.float32)
        action_mask_cell_ue = _to_tensor(obs["action_mask_cell_ue"], dtype=torch.bool)
        action_mask_prg_cell = _to_tensor(obs["action_mask_prg_cell"], dtype=torch.bool)

        if obs_cell.dim() != 2 or obs_ue.dim() != 2 or obs_prg.dim() != 3:
            raise ValueError(
                "expected obs_cell [C,Fc], obs_ue [U,Fu], obs_prg [C,P,Fp], "
                f"got {tuple(obs_cell.shape)}, {tuple(obs_ue.shape)}, {tuple(obs_prg.shape)}"
            )

        n_cell, _ = obs_cell.shape
        n_ue, _ = obs_ue.shape
        prg_cell_dim, n_prg, _ = obs_prg.shape
        raw_n_sched_ue = obs.get("n_sched_ue")
        n_sched_ue = int(raw_n_sched_ue) if raw_n_sched_ue is not None else int(n_ue)
        if prg_cell_dim != n_cell:
            raise ValueError(f"obs_prg cell dim mismatch: {prg_cell_dim} vs n_cell={n_cell}")
        if action_mask_cell_ue.shape != (n_cell, n_ue):
            raise ValueError(
                "action_mask_cell_ue mismatch: "
                f"got={tuple(action_mask_cell_ue.shape)} expected={(n_cell, n_ue)}"
            )
        if action_mask_prg_cell.shape != (n_cell, n_prg):
            raise ValueError(
                "action_mask_prg_cell mismatch: "
                f"got={tuple(action_mask_prg_cell.shape)} expected={(n_cell, n_prg)}"
            )
        raw_action_mask_ue = obs.get("action_mask_ue")
        action_mask_ue = (
            torch.ones((n_ue,), dtype=torch.bool)
            if raw_action_mask_ue is None
            else _to_tensor(raw_action_mask_ue, dtype=torch.bool)
        )
        if action_mask_ue.shape != (n_ue,):
            raise ValueError(
                "action_mask_ue mismatch: "
                f"got={tuple(action_mask_ue.shape)} expected={(n_ue,)}"
            )

        instant_tb_err = obs_cell[:, _CELL_INSTANT_TB_ERR_IDX]
        if self._cell_tb_err_ema is None or self._cell_tb_err_ema.shape != instant_tb_err.shape:
            self._cell_tb_err_ema = instant_tb_err.clone()
        else:
            alpha = float(self.cfg.tb_err_ema_alpha)
            self._cell_tb_err_ema = (1.0 - alpha) * self._cell_tb_err_ema + alpha * instant_tb_err

        cell_values = obs_cell[:, : len(_CELL_FEATURE_NAMES)].contiguous()
        ue_values = obs_ue[:, : len(_UE_FEATURE_NAMES)].contiguous()
        prg_values = obs_prg[:, :, : len(_PRG_FEATURE_NAMES)].reshape(
            n_cell * n_prg,
            len(_PRG_FEATURE_NAMES),
        ).contiguous()

        ue_serving_cell = self._infer_ue_serving_cell(action_mask_cell_ue)
        prg_owner_cell = torch.arange(n_cell, dtype=torch.long).repeat_interleave(n_prg)
        prg_local_index = torch.arange(n_prg, dtype=torch.long).repeat(n_cell)

        raw_edge_index = obs.get("obs_edge_index")
        if raw_edge_index is None:
            cell_adjacency = _build_default_cell_adjacency(n_cell)
        else:
            cell_adjacency = _to_tensor(raw_edge_index, dtype=torch.long)
            if cell_adjacency.dim() != 2 or cell_adjacency.shape[1] != 2:
                raise ValueError(
                    "obs_edge_index must be [E,2], "
                    f"got shape={tuple(cell_adjacency.shape)}"
                )
        raw_edge_attr = obs.get("obs_edge_attr")
        if raw_edge_attr is None:
            cell_edge_attr = _build_default_cell_edge_attr(cell_adjacency)
        else:
            cell_edge_attr = _to_tensor(raw_edge_attr, dtype=torch.float32)
            if cell_edge_attr.dim() != 2:
                raise ValueError(
                    f"obs_edge_attr must be [E,F], got shape={tuple(cell_edge_attr.shape)}"
                )
            if cell_edge_attr.shape[0] != cell_adjacency.shape[0]:
                raise ValueError(
                    "obs_edge_attr edge count mismatch: "
                    f"obs_edge_attr={tuple(cell_edge_attr.shape)} obs_edge_index={tuple(cell_adjacency.shape)}"
                )
        post_eq_sinr = obs.get("post_eq_sinr")
        post_eq_tensor = None
        candidate_mode = "shared_cell_pool"
        if post_eq_sinr is not None:
            post_eq_tensor = _to_tensor(post_eq_sinr, dtype=torch.float32)
            candidate_mode = "post_eq_topk"
        teacher_action_ue = None
        raw_teacher_action_ue = obs.get("teacher_action_ue_select")
        if raw_teacher_action_ue is not None:
            teacher_action_ue = _to_tensor(raw_teacher_action_ue, dtype=torch.long)
        teacher_action_prg = None
        raw_teacher_action_prg = obs.get("teacher_action_prg_alloc")
        if raw_teacher_action_prg is not None:
            teacher_action_prg = _to_tensor(raw_teacher_action_prg, dtype=torch.long)

        static_meta = StaticMeta(
            n_cell=int(n_cell),
            n_ue=int(n_ue),
            n_sched_ue=int(n_sched_ue),
            n_prg=int(n_prg),
            topology=_infer_topology(int(n_cell), self.cfg.default_topology),
            action_mask_cell_ue=action_mask_cell_ue,
            action_mask_prg_cell=action_mask_prg_cell,
            cell_adjacency=cell_adjacency,
            cell_edge_attr=cell_edge_attr,
            ue_serving_cell=ue_serving_cell,
            prg_owner_cell=prg_owner_cell,
            prg_local_index=prg_local_index,
            action_mask_ue=action_mask_ue,
            post_eq_sinr=post_eq_tensor,
            teacher_action_ue=teacher_action_ue,
            teacher_action_prg=teacher_action_prg,
            candidate_mode=candidate_mode,
        )

        return GraphFeatureSnapshot(
            cell_features=NamedFeatureBlock(values=cell_values, names=_CELL_FEATURE_NAMES),
            ue_features=NamedFeatureBlock(values=ue_values, names=_UE_FEATURE_NAMES),
            prg_features=NamedFeatureBlock(values=prg_values, names=_PRG_FEATURE_NAMES),
            static_meta=static_meta,
        )

    @staticmethod
    def _infer_ue_serving_cell(action_mask_cell_ue: torch.Tensor) -> torch.Tensor:
        n_cell, n_ue = action_mask_cell_ue.shape
        serving_cell = torch.full((n_ue,), -1, dtype=torch.long)
        for ue_idx in range(n_ue):
            owner = torch.nonzero(action_mask_cell_ue[:, ue_idx], as_tuple=False).flatten()
            if owner.numel() > 0:
                serving_cell[ue_idx] = int(owner[0].item())
            elif n_cell > 0:
                serving_cell[ue_idx] = 0
        return serving_cell
