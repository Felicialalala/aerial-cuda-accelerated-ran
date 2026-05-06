#!/usr/bin/env python3
"""Online PPO fine-tuning for the Stage-B sparse entity graph policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.gnnrl.masks import (
    apply_prg_action_mask,
    apply_ue_action_mask,
    build_slot_selection_mask,
    sanitize_targets,
)
from training.gnnrl.slot_layout import build_type0_slot_layout
from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from training.stageb_hgraph.graph_ops import (
    decode_joint_actions_to_type0,
    demand_active_ue_mask_from_features,
    joint_targets_from_type0_actions,
    type0_action_preflight_reject_counts,
)
from training.stageb_hgraph.model import HGraphModelConfig, StageBHGraphPolicy
from training.stageb_hgraph.online_imitation import OnlineEpisodeRunner, _pick_device, _set_seed
from training.stageb_hgraph.reward_utils import (
    compute_stageb_blankaware_reward,
    compute_stageb_v1_reward,
    extract_reward_metrics,
)
from training.stageb_hgraph.teacher import HeuristicTeacherConfig, build_joint_teacher
from training.stageb_hgraph.types import SparseEntityGraph


@dataclass
class RolloutStep:
    graph: SparseEntityGraph
    ue_action_class: torch.Tensor
    prg_action_class: torch.Tensor
    old_logp: torch.Tensor
    old_value: float
    next_value: float
    reward: float
    reward_raw_env: float
    done: float
    entropy: float
    blank_ratio: float
    post_decode_drop_ratio: float
    final_blank_ratio: float
    cap_drop_count: float
    fallback_success_count: float
    final_blank_count: float
    action_fill_ratio: float
    action_unique_ue_count: float
    candidate_unique_ue_count: float
    demand_active_ue_count: float
    candidate_coverage_ratio: float
    unserved_demand_ue_count: float
    planned_unserved_demand_ue_count: float
    actual_unserved_backlog_bytes: float
    accepted_pre_pdsch_diagnostics_available: float
    actual_diagnostics_available: float
    native_reject_diagnostics_available: float
    native_reject_wrong_cell_slot_count: float
    native_reject_empty_slot_count: float
    native_reject_prg_mask_count: float
    native_reject_oob_slot_count: float
    native_reject_duplicate_ue_count: float
    python_preflight_wrong_cell_slot_count: float
    python_preflight_empty_slot_count: float
    python_preflight_prg_mask_count: float
    python_preflight_oob_slot_count: float
    python_preflight_duplicate_ue_count: float
    native_slot_layout_available: float
    native_slot_count_abs_delta_sum: float
    native_cell_start_abs_delta_sum: float
    native_slot_to_cell_mismatch_count: float
    planned_accepted_pre_pdsch_prg_gap_count: float
    planned_accepted_pre_pdsch_prg_gap_ratio: float
    accepted_pre_pdsch_actual_prg_gap_count: float
    accepted_pre_pdsch_actual_prg_gap_ratio: float
    planned_native_prg_gap_count: float
    planned_native_prg_gap_ratio: float
    no_service_streak_p90: float
    no_service_streak_max: float
    recent100_service_rate_p10: float
    recent100_service_rate_p50: float
    recent100_unserved_backlog_p90_bytes: float
    recent100_debt_score_p90_bytes: float
    recent100_debt_score_max_bytes: float
    backlog_p90_bytes: float
    backlog_max_bytes: float
    goodput_mbps: float
    tb_err_rate: float
    expiry_drop_rate: float
    prg_utilization_ratio: float
    prg_reuse_ratio: float
    packet_completed_count: float
    packet_effective_service_rate_mbps: float
    conflict_mean: float
    ici_mean: float
    urgent_backlog_penalty: float
    backlog_debt_coef: float
    packet_rate_bonus: float
    packet_completion_bonus: float
    aged_backlog_penalty: float
    aged_backlog_coef: float
    active_prg_goodput_efficiency: float
    harmful_packing_penalty: float
    coverage_repair_count: float


_NATIVE_REJECT_INFO_KEYS = (
    "native_reject_wrong_cell_slot_count",
    "native_reject_empty_slot_count",
    "native_reject_prg_mask_count",
    "native_reject_oob_slot_count",
    "native_reject_duplicate_ue_count",
)
_PREFLIGHT_REJECT_KEYS = (
    "wrong_cell_slot",
    "empty_slot",
    "prg_mask",
    "oob_slot",
    "duplicate_ue",
)


class GraphValueHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        in_dim = hidden_dim * 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.net(state_vec).squeeze(-1)


@dataclass
class RunningNorm:
    dim: int
    eps: float = 1.0e-6

    def __post_init__(self):
        self.count = 0
        self.mean = torch.zeros(self.dim, dtype=torch.float32)
        self.m2 = torch.zeros(self.dim, dtype=torch.float32)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().view(-1, self.dim).cpu()
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean = self.mean + delta / float(self.count)
            delta2 = row - self.mean
            self.m2 = self.m2 + delta * delta2

    def std(self) -> torch.Tensor:
        if self.count < 2:
            return torch.ones_like(self.mean)
        var = self.m2 / float(self.count)
        return torch.sqrt(var.clamp_min(self.eps))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std().to(x.device)

    def state_dict(self) -> Dict[str, object]:
        return {
            "count": int(self.count),
            "mean": self.mean.tolist(),
            "std": self.std().tolist(),
        }


@dataclass
class RunningScalarNorm:
    eps: float = 1.0e-6

    def __post_init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        dx = x - self.mean
        self.mean += dx / float(self.count)
        dx2 = x - self.mean
        self.m2 += dx * dx2

    def std(self) -> float:
        if self.count < 2:
            return 1.0
        var = self.m2 / float(self.count)
        return float(max(var, self.eps) ** 0.5)

    def normalize(self, x: float) -> float:
        return float((x - self.mean) / self.std())

    def state_dict(self) -> Dict[str, float]:
        return {
            "count": float(self.count),
            "mean": float(self.mean),
            "std": float(self.std()),
        }


class PerUeDiagnosticsTracker:
    def __init__(self, n_ue: int, window: int = 100):
        self.n_ue = int(n_ue)
        self.window = max(1, int(window))
        self.ptr = 0
        self.filled = 0
        self.demand_hist = np.zeros((self.window, self.n_ue), dtype=np.int16)
        self.served_hist = np.zeros((self.window, self.n_ue), dtype=np.int16)
        self.goodput_served_hist = np.zeros((self.window, self.n_ue), dtype=np.int16)
        self.demand_backlog_hist = np.zeros((self.window, self.n_ue), dtype=np.float32)
        self.unserved_backlog_hist = np.zeros((self.window, self.n_ue), dtype=np.float32)
        self.effective_prg_hist = np.zeros((self.window, self.n_ue), dtype=np.float32)
        self.goodput_bytes_hist = np.zeros((self.window, self.n_ue), dtype=np.float32)
        self.demand_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.served_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.goodput_served_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.demand_backlog_sum = np.zeros((self.n_ue,), dtype=np.float32)
        self.unserved_backlog_sum = np.zeros((self.n_ue,), dtype=np.float32)
        self.effective_prg_sum = np.zeros((self.n_ue,), dtype=np.float32)
        self.goodput_bytes_sum = np.zeros((self.n_ue,), dtype=np.float32)
        self.no_service_streak = np.zeros((self.n_ue,), dtype=np.int32)
        self.no_goodput_streak = np.zeros((self.n_ue,), dtype=np.int32)
        self.last_demand = np.zeros((self.n_ue,), dtype=bool)
        self.last_served = np.zeros((self.n_ue,), dtype=bool)
        self.last_goodput_served = np.zeros((self.n_ue,), dtype=bool)
        self.last_selected_slot_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.last_assigned_prg_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.last_backlog_bytes = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_hol_delay_ms = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_ttl_slack_ms = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_recent_scheduled_ratio = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_recent_goodput_deficit_norm = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_planned_served = np.zeros((self.n_ue,), dtype=bool)
        self.last_planned_assigned_prg_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.last_accepted_pre_pdsch_available = False
        self.last_accepted_pre_pdsch_assigned_prg_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.last_actual_available = False
        self.last_actual_served = np.zeros((self.n_ue,), dtype=bool)
        self.last_actual_assigned_prg_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.last_actual_served_bytes = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_actual_goodput_bytes = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_actual_tb_tx_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.last_actual_tb_err_count = np.zeros((self.n_ue,), dtype=np.int32)

    def update(
        self,
        *,
        demand_mask: np.ndarray,
        planned_served_mask: np.ndarray,
        planned_selected_slot_count: np.ndarray | None = None,
        planned_assigned_prg_count: np.ndarray,
        backlog_bytes: np.ndarray,
        hol_delay_ms: np.ndarray | None = None,
        ttl_slack_ms: np.ndarray | None = None,
        recent_scheduled_ratio: np.ndarray | None = None,
        recent_goodput_deficit_norm: np.ndarray | None = None,
        accepted_pre_pdsch_available: bool = False,
        accepted_pre_pdsch_assigned_prg_count: np.ndarray | None = None,
        actual_available: bool = False,
        actual_assigned_prg_count: np.ndarray | None = None,
        actual_served_bytes: np.ndarray | None = None,
        actual_goodput_bytes: np.ndarray | None = None,
        actual_tb_tx_count: np.ndarray | None = None,
        actual_tb_err_count: np.ndarray | None = None,
    ) -> None:
        demand_mask = np.asarray(demand_mask, dtype=bool).reshape(self.n_ue)
        planned_served_mask = np.asarray(planned_served_mask, dtype=bool).reshape(self.n_ue)
        planned_selected_slot_count = (
            np.zeros((self.n_ue,), dtype=np.int32)
            if planned_selected_slot_count is None
            else np.asarray(planned_selected_slot_count, dtype=np.int32).reshape(self.n_ue)
        )
        planned_assigned_prg_count = np.asarray(planned_assigned_prg_count, dtype=np.int32).reshape(self.n_ue)
        backlog_bytes = np.asarray(backlog_bytes, dtype=np.float32).reshape(self.n_ue)
        hol_delay_ms = (
            np.zeros((self.n_ue,), dtype=np.float32)
            if hol_delay_ms is None
            else np.asarray(hol_delay_ms, dtype=np.float32).reshape(self.n_ue)
        )
        ttl_slack_ms = (
            np.zeros((self.n_ue,), dtype=np.float32)
            if ttl_slack_ms is None
            else np.asarray(ttl_slack_ms, dtype=np.float32).reshape(self.n_ue)
        )
        recent_scheduled_ratio = (
            np.zeros((self.n_ue,), dtype=np.float32)
            if recent_scheduled_ratio is None
            else np.asarray(recent_scheduled_ratio, dtype=np.float32).reshape(self.n_ue)
        )
        recent_goodput_deficit_norm = (
            np.zeros((self.n_ue,), dtype=np.float32)
            if recent_goodput_deficit_norm is None
            else np.asarray(recent_goodput_deficit_norm, dtype=np.float32).reshape(self.n_ue)
        )
        accepted_pre_pdsch_ok = bool(
            accepted_pre_pdsch_available and accepted_pre_pdsch_assigned_prg_count is not None
        )
        accepted_pre_pdsch_assigned_prg_count = (
            np.asarray(accepted_pre_pdsch_assigned_prg_count, dtype=np.int32).reshape(self.n_ue)
            if accepted_pre_pdsch_ok
            else np.zeros((self.n_ue,), dtype=np.int32)
        )

        actual_ok = bool(
            actual_available
            and actual_assigned_prg_count is not None
            and actual_served_bytes is not None
            and actual_goodput_bytes is not None
            and actual_tb_tx_count is not None
            and actual_tb_err_count is not None
        )
        if actual_ok:
            actual_assigned_prg_count = np.asarray(actual_assigned_prg_count, dtype=np.int32).reshape(self.n_ue)
            actual_served_bytes = np.asarray(actual_served_bytes, dtype=np.float32).reshape(self.n_ue)
            actual_goodput_bytes = np.asarray(actual_goodput_bytes, dtype=np.float32).reshape(self.n_ue)
            actual_tb_tx_count = np.asarray(actual_tb_tx_count, dtype=np.int32).reshape(self.n_ue)
            actual_tb_err_count = np.asarray(actual_tb_err_count, dtype=np.int32).reshape(self.n_ue)
            actual_served_mask = (
                (actual_assigned_prg_count > 0)
                | (actual_served_bytes > 0.0)
                | (actual_tb_tx_count > 0)
            )
            goodput_served_mask = actual_goodput_bytes > 0.0
            served_mask = actual_served_mask
            assigned_prg_count = actual_assigned_prg_count
        else:
            actual_assigned_prg_count = np.zeros((self.n_ue,), dtype=np.int32)
            actual_served_bytes = np.zeros((self.n_ue,), dtype=np.float32)
            actual_goodput_bytes = np.zeros((self.n_ue,), dtype=np.float32)
            actual_tb_tx_count = np.zeros((self.n_ue,), dtype=np.int32)
            actual_tb_err_count = np.zeros((self.n_ue,), dtype=np.int32)
            actual_served_mask = np.zeros((self.n_ue,), dtype=bool)
            goodput_served_mask = planned_served_mask
            served_mask = planned_served_mask
            assigned_prg_count = planned_assigned_prg_count

        served_for_window = demand_mask & served_mask
        goodput_served_for_window = demand_mask & goodput_served_mask
        old_demand = self.demand_hist[self.ptr].astype(np.int32, copy=False)
        old_served = self.served_hist[self.ptr].astype(np.int32, copy=False)
        old_goodput_served = self.goodput_served_hist[self.ptr].astype(np.int32, copy=False)
        old_demand_backlog = self.demand_backlog_hist[self.ptr]
        old_unserved_backlog = self.unserved_backlog_hist[self.ptr]
        old_effective_prg = self.effective_prg_hist[self.ptr]
        old_goodput_bytes = self.goodput_bytes_hist[self.ptr]
        new_demand = demand_mask.astype(np.int32, copy=False)
        new_served = served_for_window.astype(np.int32, copy=False)
        new_goodput_served = goodput_served_for_window.astype(np.int32, copy=False)
        new_demand_backlog = np.where(demand_mask, backlog_bytes, 0.0).astype(np.float32, copy=False)
        new_unserved_backlog = np.where(demand_mask & ~goodput_served_mask, backlog_bytes, 0.0).astype(
            np.float32,
            copy=False,
        )
        new_effective_prg = assigned_prg_count.astype(np.float32, copy=False)
        new_goodput_bytes = actual_goodput_bytes.astype(np.float32, copy=False) if actual_ok else np.zeros((self.n_ue,), dtype=np.float32)
        self.demand_count += new_demand - old_demand
        self.served_count += new_served - old_served
        self.goodput_served_count += new_goodput_served - old_goodput_served
        self.demand_backlog_sum += new_demand_backlog - old_demand_backlog
        self.unserved_backlog_sum += new_unserved_backlog - old_unserved_backlog
        self.effective_prg_sum += new_effective_prg - old_effective_prg
        self.goodput_bytes_sum += new_goodput_bytes - old_goodput_bytes
        self.demand_hist[self.ptr] = new_demand.astype(np.int16, copy=False)
        self.served_hist[self.ptr] = new_served.astype(np.int16, copy=False)
        self.goodput_served_hist[self.ptr] = new_goodput_served.astype(np.int16, copy=False)
        self.demand_backlog_hist[self.ptr] = new_demand_backlog
        self.unserved_backlog_hist[self.ptr] = new_unserved_backlog
        self.effective_prg_hist[self.ptr] = new_effective_prg
        self.goodput_bytes_hist[self.ptr] = new_goodput_bytes
        self.ptr = (self.ptr + 1) % self.window
        self.filled = min(self.window, self.filled + 1)

        self.no_service_streak = np.where(
            demand_mask,
            np.where(served_mask, 0, self.no_service_streak + 1),
            0,
        ).astype(np.int32, copy=False)
        self.no_goodput_streak = np.where(
            demand_mask,
            np.where(goodput_served_mask, 0, self.no_goodput_streak + 1),
            0,
        ).astype(np.int32, copy=False)
        self.last_demand = demand_mask
        self.last_served = served_mask
        self.last_goodput_served = goodput_served_mask
        self.last_selected_slot_count = planned_selected_slot_count
        self.last_assigned_prg_count = assigned_prg_count
        self.last_backlog_bytes = backlog_bytes
        self.last_hol_delay_ms = hol_delay_ms
        self.last_ttl_slack_ms = ttl_slack_ms
        self.last_recent_scheduled_ratio = recent_scheduled_ratio
        self.last_recent_goodput_deficit_norm = recent_goodput_deficit_norm
        self.last_planned_served = planned_served_mask
        self.last_planned_assigned_prg_count = planned_assigned_prg_count
        self.last_accepted_pre_pdsch_available = accepted_pre_pdsch_ok
        self.last_accepted_pre_pdsch_assigned_prg_count = accepted_pre_pdsch_assigned_prg_count
        self.last_actual_available = actual_ok
        self.last_actual_served = actual_served_mask
        self.last_actual_assigned_prg_count = actual_assigned_prg_count
        self.last_actual_served_bytes = actual_served_bytes
        self.last_actual_goodput_bytes = actual_goodput_bytes
        self.last_actual_tb_tx_count = actual_tb_tx_count
        self.last_actual_tb_err_count = actual_tb_err_count

    def recent100_service_rate(self) -> np.ndarray:
        rate = np.ones((self.n_ue,), dtype=np.float32)
        has_demand_window = self.demand_count > 0
        rate[has_demand_window] = (
            self.served_count[has_demand_window].astype(np.float32)
            / self.demand_count[has_demand_window].astype(np.float32)
        )
        return rate

    def recent100_goodput_service_rate(self) -> np.ndarray:
        rate = np.ones((self.n_ue,), dtype=np.float32)
        has_demand_window = self.demand_count > 0
        rate[has_demand_window] = (
            self.goodput_served_count[has_demand_window].astype(np.float32)
            / self.demand_count[has_demand_window].astype(np.float32)
        )
        return rate

    def recent100_demand_backlog_mean(self) -> np.ndarray:
        mean = np.zeros((self.n_ue,), dtype=np.float32)
        has_demand_window = self.demand_count > 0
        mean[has_demand_window] = (
            self.demand_backlog_sum[has_demand_window]
            / self.demand_count[has_demand_window].astype(np.float32)
        )
        return mean

    def recent100_unserved_backlog_mean(self) -> np.ndarray:
        mean = np.zeros((self.n_ue,), dtype=np.float32)
        has_demand_window = self.demand_count > 0
        mean[has_demand_window] = (
            self.unserved_backlog_sum[has_demand_window]
            / self.demand_count[has_demand_window].astype(np.float32)
        )
        return mean

    def recent100_effective_prg_mean(self) -> np.ndarray:
        denom = float(max(1, self.filled))
        return (self.effective_prg_sum / denom).astype(np.float32, copy=False)

    def recent100_goodput_bytes_mean(self) -> np.ndarray:
        denom = float(max(1, self.filled))
        return (self.goodput_bytes_sum / denom).astype(np.float32, copy=False)

    def recent100_debt_score(self) -> np.ndarray:
        rates = self.recent100_goodput_service_rate()
        unserved_backlog_mean = self.recent100_unserved_backlog_mean()
        window = float(max(1, self.filled))
        streak_boost = 1.0 + (self.no_goodput_streak.astype(np.float32) / window)
        current_unfair_backlog = self.last_backlog_bytes * (1.0 - rates)
        return ((unserved_backlog_mean + current_unfair_backlog) * streak_boost).astype(np.float32, copy=False)

    def summary(self) -> Dict[str, float]:
        demand_mask = self.last_demand
        demand_count = float(demand_mask.sum())
        planned_prg_total = float(self.last_planned_assigned_prg_count.sum())
        accepted_pre_pdsch_prg_total = (
            float(self.last_accepted_pre_pdsch_assigned_prg_count.sum())
            if self.last_accepted_pre_pdsch_available
            else planned_prg_total
        )
        actual_prg_total = (
            float(self.last_actual_assigned_prg_count.sum())
            if self.last_actual_available
            else planned_prg_total
        )
        planned_accepted_pre_pdsch_prg_gap_count = (
            max(0.0, planned_prg_total - accepted_pre_pdsch_prg_total)
            if self.last_accepted_pre_pdsch_available
            else 0.0
        )
        planned_accepted_pre_pdsch_prg_gap_ratio = (
            planned_accepted_pre_pdsch_prg_gap_count / max(planned_prg_total, 1.0)
        )
        accepted_pre_pdsch_actual_prg_gap_count = (
            max(0.0, accepted_pre_pdsch_prg_total - actual_prg_total)
            if self.last_accepted_pre_pdsch_available
            else 0.0
        )
        accepted_pre_pdsch_actual_prg_gap_ratio = (
            accepted_pre_pdsch_actual_prg_gap_count / max(accepted_pre_pdsch_prg_total, 1.0)
        )
        planned_native_prg_gap_count = max(0.0, planned_prg_total - actual_prg_total)
        planned_native_prg_gap_ratio = planned_native_prg_gap_count / max(planned_prg_total, 1.0)
        if demand_count <= 0:
            return {
                "unserved_demand_ue_count": 0.0,
                "planned_unserved_demand_ue_count": 0.0,
                "actual_unserved_demand_ue_count": 0.0,
                "actual_unserved_backlog_bytes": 0.0,
                "accepted_pre_pdsch_diagnostics_available": float(self.last_accepted_pre_pdsch_available),
                "actual_diagnostics_available": float(self.last_actual_available),
                "planned_accepted_pre_pdsch_prg_gap_count": float(planned_accepted_pre_pdsch_prg_gap_count),
                "planned_accepted_pre_pdsch_prg_gap_ratio": float(planned_accepted_pre_pdsch_prg_gap_ratio),
                "accepted_pre_pdsch_actual_prg_gap_count": float(accepted_pre_pdsch_actual_prg_gap_count),
                "accepted_pre_pdsch_actual_prg_gap_ratio": float(accepted_pre_pdsch_actual_prg_gap_ratio),
                "planned_native_prg_gap_count": float(planned_native_prg_gap_count),
                "planned_native_prg_gap_ratio": float(planned_native_prg_gap_ratio),
                "no_service_streak_p90": 0.0,
                "no_service_streak_max": 0.0,
                "recent100_service_rate_p10": 1.0,
                "recent100_service_rate_p50": 1.0,
                "recent100_unserved_backlog_p90_bytes": 0.0,
                "recent100_debt_score_p90_bytes": 0.0,
                "recent100_debt_score_max_bytes": 0.0,
                "backlog_p90_bytes": 0.0,
                "backlog_max_bytes": 0.0,
            }
        streak = self.no_service_streak[demand_mask].astype(np.float32, copy=False)
        rates = self.recent100_service_rate()[demand_mask].astype(np.float32, copy=False)
        unserved_backlog_mean = self.recent100_unserved_backlog_mean()[demand_mask].astype(np.float32, copy=False)
        debt_scores = self.recent100_debt_score()[demand_mask].astype(np.float32, copy=False)
        backlog = self.last_backlog_bytes[demand_mask].astype(np.float32, copy=False)
        effective_unserved = np.logical_and(demand_mask, ~self.last_served)
        planned_unserved = np.logical_and(demand_mask, ~self.last_planned_served)
        actual_unserved = np.logical_and(demand_mask, ~self.last_actual_served) if self.last_actual_available else effective_unserved
        return {
            "unserved_demand_ue_count": float(effective_unserved.sum()),
            "planned_unserved_demand_ue_count": float(planned_unserved.sum()),
            "actual_unserved_demand_ue_count": float(actual_unserved.sum()),
            "actual_unserved_backlog_bytes": float(self.last_backlog_bytes[actual_unserved].sum()),
            "accepted_pre_pdsch_diagnostics_available": float(self.last_accepted_pre_pdsch_available),
            "actual_diagnostics_available": float(self.last_actual_available),
            "planned_accepted_pre_pdsch_prg_gap_count": float(planned_accepted_pre_pdsch_prg_gap_count),
            "planned_accepted_pre_pdsch_prg_gap_ratio": float(planned_accepted_pre_pdsch_prg_gap_ratio),
            "accepted_pre_pdsch_actual_prg_gap_count": float(accepted_pre_pdsch_actual_prg_gap_count),
            "accepted_pre_pdsch_actual_prg_gap_ratio": float(accepted_pre_pdsch_actual_prg_gap_ratio),
            "planned_native_prg_gap_count": float(planned_native_prg_gap_count),
            "planned_native_prg_gap_ratio": float(planned_native_prg_gap_ratio),
            "no_service_streak_p90": float(np.percentile(streak, 90)),
            "no_service_streak_max": float(np.max(streak) if streak.size > 0 else 0.0),
            "recent100_service_rate_p10": float(np.percentile(rates, 10)),
            "recent100_service_rate_p50": float(np.percentile(rates, 50)),
            "recent100_unserved_backlog_p90_bytes": float(np.percentile(unserved_backlog_mean, 90)),
            "recent100_debt_score_p90_bytes": float(np.percentile(debt_scores, 90)),
            "recent100_debt_score_max_bytes": float(np.max(debt_scores) if debt_scores.size > 0 else 0.0),
            "backlog_p90_bytes": float(np.percentile(backlog, 90)),
            "backlog_max_bytes": float(np.max(backlog) if backlog.size > 0 else 0.0),
        }

    def rows(self, *, ue_serving_cell: np.ndarray) -> List[Dict[str, float]]:
        rates = self.recent100_service_rate()
        goodput_rates = self.recent100_goodput_service_rate()
        demand_backlog_mean = self.recent100_demand_backlog_mean()
        unserved_backlog_mean = self.recent100_unserved_backlog_mean()
        effective_prg_mean = self.recent100_effective_prg_mean()
        goodput_bytes_mean = self.recent100_goodput_bytes_mean()
        debt_scores = self.recent100_debt_score()
        rows: List[Dict[str, float]] = []
        for ue_idx in range(self.n_ue):
            rows.append(
                {
                    "ue_idx": float(ue_idx),
                    "serving_cell": float(int(ue_serving_cell[ue_idx])),
                    "demand_active": float(bool(self.last_demand[ue_idx])),
                    "served_this_tti": float(bool(self.last_served[ue_idx])),
                    "assigned_prg_count": float(int(self.last_assigned_prg_count[ue_idx])),
                    "planned_selected_slot_count": float(int(self.last_selected_slot_count[ue_idx])),
                    "planned_selected_slot_this_tti": float(self.last_selected_slot_count[ue_idx] > 0),
                    "planned_served_this_tti": float(bool(self.last_planned_served[ue_idx])),
                    "planned_assigned_prg_count": float(int(self.last_planned_assigned_prg_count[ue_idx])),
                    "accepted_pre_pdsch_diagnostics_available": float(
                        self.last_accepted_pre_pdsch_available
                    ),
                    "accepted_pre_pdsch_served_this_tti": float(
                        bool(self.last_accepted_pre_pdsch_assigned_prg_count[ue_idx] > 0)
                    ),
                    "accepted_pre_pdsch_assigned_prg_count": float(
                        int(self.last_accepted_pre_pdsch_assigned_prg_count[ue_idx])
                    ),
                    "actual_diagnostics_available": float(self.last_actual_available),
                    "actual_served_this_tti": float(bool(self.last_actual_served[ue_idx])),
                    "actual_goodput_served_this_tti": float(bool(self.last_goodput_served[ue_idx])),
                    "actual_assigned_prg_count": float(int(self.last_actual_assigned_prg_count[ue_idx])),
                    "actual_served_bytes": float(self.last_actual_served_bytes[ue_idx]),
                    "actual_goodput_bytes": float(self.last_actual_goodput_bytes[ue_idx]),
                    "actual_tb_tx_count": float(int(self.last_actual_tb_tx_count[ue_idx])),
                    "actual_tb_err_count": float(int(self.last_actual_tb_err_count[ue_idx])),
                    "backlog_bytes": float(self.last_backlog_bytes[ue_idx]),
                    "hol_delay_ms": float(self.last_hol_delay_ms[ue_idx]),
                    "ttl_slack_ms": float(self.last_ttl_slack_ms[ue_idx]),
                    "recent_scheduled_ratio_feat": float(self.last_recent_scheduled_ratio[ue_idx]),
                    "recent_goodput_deficit_norm_feat": float(self.last_recent_goodput_deficit_norm[ue_idx]),
                    "no_service_streak": float(int(self.no_service_streak[ue_idx])),
                    "no_goodput_streak": float(int(self.no_goodput_streak[ue_idx])),
                    "planned_accepted_pre_pdsch_prg_delta": float(
                        int(self.last_planned_assigned_prg_count[ue_idx])
                        - int(self.last_accepted_pre_pdsch_assigned_prg_count[ue_idx])
                    ),
                    "accepted_pre_pdsch_actual_prg_delta": float(
                        int(self.last_accepted_pre_pdsch_assigned_prg_count[ue_idx])
                        - int(self.last_actual_assigned_prg_count[ue_idx])
                    ),
                    "planned_but_pre_pdsch_dropped_this_tti": float(
                        bool(
                            self.last_accepted_pre_pdsch_available
                            and self.last_planned_assigned_prg_count[ue_idx] > 0
                            and self.last_accepted_pre_pdsch_assigned_prg_count[ue_idx] <= 0
                        )
                    ),
                    "accepted_pre_pdsch_but_final_dropped_this_tti": float(
                        bool(
                            self.last_accepted_pre_pdsch_available
                            and self.last_actual_available
                            and self.last_accepted_pre_pdsch_assigned_prg_count[ue_idx] > 0
                            and self.last_actual_assigned_prg_count[ue_idx] <= 0
                        )
                    ),
                    "planned_native_prg_delta": float(
                        int(self.last_planned_assigned_prg_count[ue_idx])
                        - int(self.last_actual_assigned_prg_count[ue_idx])
                    ),
                    "planned_but_native_dropped_this_tti": float(
                        bool(
                            self.last_actual_available
                            and self.last_planned_assigned_prg_count[ue_idx] > 0
                            and self.last_actual_assigned_prg_count[ue_idx] <= 0
                        )
                    ),
                    "recent100_demand_tti": float(int(self.demand_count[ue_idx])),
                    "recent100_served_tti": float(int(self.served_count[ue_idx])),
                    "recent100_goodput_served_tti": float(int(self.goodput_served_count[ue_idx])),
                    "recent100_service_rate": float(rates[ue_idx]),
                    "recent100_goodput_service_rate": float(goodput_rates[ue_idx]),
                    "recent100_backlog_demand_mean_bytes": float(demand_backlog_mean[ue_idx]),
                    "recent100_unserved_backlog_mean_bytes": float(unserved_backlog_mean[ue_idx]),
                    "recent100_effective_prg_mean": float(effective_prg_mean[ue_idx]),
                    "recent100_goodput_bytes_mean": float(goodput_bytes_mean[ue_idx]),
                    "recent100_debt_score_bytes": float(debt_scores[ue_idx]),
                }
            )
        return rows


def _restore_running_norm(payload: Dict[str, object], dim: int) -> RunningNorm:
    norm = RunningNorm(dim=dim)
    count = int(payload.get("count", 0))
    if count <= 0:
        return norm
    mean = torch.tensor(payload.get("mean", []), dtype=torch.float32).view(-1)
    std = torch.tensor(payload.get("std", []), dtype=torch.float32).view(-1)
    if mean.numel() != dim or std.numel() != dim:
        return norm
    norm.count = count
    norm.mean = mean.clone()
    norm.m2 = torch.zeros(dim, dtype=torch.float32) if count < 2 else std.square() * float(count)
    return norm


def _restore_running_scalar_norm(payload: Dict[str, float]) -> RunningScalarNorm:
    norm = RunningScalarNorm()
    count = int(payload.get("count", 0.0))
    if count <= 0:
        return norm
    std = float(payload.get("std", 1.0))
    norm.count = count
    norm.mean = float(payload.get("mean", 0.0))
    norm.m2 = 0.0 if count < 2 else (std * std) * float(count)
    return norm


def _resolve_prg_candidate_count(args: argparse.Namespace) -> int:
    if int(getattr(args, "max_prg_candidates", 0)) > 0:
        return int(args.max_prg_candidates)
    return max(1, int(args.ue_prg_topk)) + max(0, int(args.ue_prg_diversity_extra))


def _resolve_teacher_aux_coef(args: argparse.Namespace, iter_idx: int) -> float:
    base = float(getattr(args, "teacher_aux_coef", 0.0))
    if base <= 0.0 or str(args.teacher_mode) != "pfq_passthrough":
        return 0.0
    decay_iters = max(0, int(getattr(args, "teacher_aux_decay_iters", 0)))
    if decay_iters <= 0:
        return base
    progress = max(0.0, 1.0 - (max(0, int(iter_idx) - 1) / float(decay_iters)))
    return base * progress


def _graph_to_device(graph: SparseEntityGraph, device: torch.device) -> SparseEntityGraph:
    snap = graph.snapshot
    meta = snap.static_meta
    moved_snapshot = replace(
        snap,
        cell_features=replace(snap.cell_features, values=snap.cell_features.values.to(device=device)),
        ue_features=replace(snap.ue_features, values=snap.ue_features.values.to(device=device)),
        prg_features=replace(snap.prg_features, values=snap.prg_features.values.to(device=device)),
        static_meta=replace(
            meta,
            action_mask_cell_ue=meta.action_mask_cell_ue.to(device=device),
            action_mask_prg_cell=meta.action_mask_prg_cell.to(device=device),
            cell_adjacency=meta.cell_adjacency.to(device=device),
            cell_edge_attr=meta.cell_edge_attr.to(device=device) if meta.cell_edge_attr is not None else None,
            ue_serving_cell=meta.ue_serving_cell.to(device=device),
            prg_owner_cell=meta.prg_owner_cell.to(device=device),
            prg_local_index=meta.prg_local_index.to(device=device),
            action_mask_ue=meta.action_mask_ue.to(device=device) if meta.action_mask_ue is not None else None,
            post_eq_sinr=meta.post_eq_sinr.to(device=device) if meta.post_eq_sinr is not None else None,
            teacher_action_ue=meta.teacher_action_ue.to(device=device) if meta.teacher_action_ue is not None else None,
            teacher_action_prg=meta.teacher_action_prg.to(device=device) if meta.teacher_action_prg is not None else None,
        ),
    )
    moved_edges = {
        name: replace(
            edge,
            index=edge.index.to(device=device),
            attr=edge.attr.to(device=device) if edge.attr is not None else None,
        )
        for name, edge in graph.edges.items()
    }
    return SparseEntityGraph(
        snapshot=moved_snapshot,
        edges=moved_edges,
        cell_embeddings=graph.cell_embeddings.to(device=device) if graph.cell_embeddings is not None else None,
        ue_embeddings=graph.ue_embeddings.to(device=device) if graph.ue_embeddings is not None else None,
        prg_embeddings=graph.prg_embeddings.to(device=device) if graph.prg_embeddings is not None else None,
        stats=dict(graph.stats),
    )


def _pool_actor_state(actor_out: Dict[str, torch.Tensor]) -> torch.Tensor:
    cell = actor_out["cell_embeddings"].mean(dim=0)
    ue = actor_out["ue_embeddings"].mean(dim=0)
    prg = actor_out["prg_embeddings"].mean(dim=0)
    return torch.cat([cell, ue, prg], dim=-1)


def _pool_actor_state_batch(actor_out: Dict[str, torch.Tensor]) -> torch.Tensor:
    cell = actor_out["cell_embeddings"].mean(dim=1)
    ue = actor_out["ue_embeddings"].mean(dim=1)
    prg = actor_out["prg_embeddings"].mean(dim=1)
    return torch.cat([cell, ue, prg], dim=-1)


def _prg_feature_mean(graph: SparseEntityGraph, feature_idx: int) -> float:
    prg = graph.snapshot.prg_features.values
    if prg.shape[1] <= feature_idx:
        return 0.0
    return float(torch.clamp(prg[:, feature_idx], 0.0, 1.0).mean().item())


def _evaluate_graph(
    actor: StageBHGraphPolicy,
    critic: GraphValueHead,
    graph: SparseEntityGraph,
    *,
    state_norm: RunningNorm | None = None,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    actor_out = actor.forward_graph(graph)
    state_vec = _pool_actor_state(actor_out)
    if state_norm is not None:
        state_vec = state_norm.normalize(state_vec)
    value = critic(state_vec)
    return actor_out, value


def _require_compatible_graph_batch(graphs: Sequence[SparseEntityGraph]) -> None:
    if not graphs:
        raise ValueError("graph batch must not be empty")
    ref = graphs[0].snapshot.static_meta
    for idx, graph in enumerate(graphs[1:], start=1):
        meta = graph.snapshot.static_meta
        if (
            meta.n_cell != ref.n_cell
            or meta.n_ue != ref.n_ue
            or meta.n_sched_ue != ref.n_sched_ue
            or meta.n_prg != ref.n_prg
        ):
            raise ValueError(
                "all graphs in a minibatch must share the same entity dimensions: "
                f"ref={(ref.n_cell, ref.n_ue, ref.n_sched_ue, ref.n_prg)} "
                f"graph[{idx}]={(meta.n_cell, meta.n_ue, meta.n_sched_ue, meta.n_prg)}"
            )


def _stack_graph_masks(
    graphs: Sequence[SparseEntityGraph],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref = graphs[0].snapshot.static_meta
    action_mask_ue = torch.stack(
        [
            (
                graph.snapshot.static_meta.action_mask_ue
                if graph.snapshot.static_meta.action_mask_ue is not None
                else torch.ones((ref.n_ue,), dtype=torch.bool, device=device)
            ).to(device=device, dtype=torch.bool)
            for graph in graphs
        ],
        dim=0,
    )
    action_mask_cell_ue = torch.stack(
        [graph.snapshot.static_meta.action_mask_cell_ue.to(device=device, dtype=torch.bool) for graph in graphs],
        dim=0,
    )
    action_mask_prg_cell = torch.stack(
        [graph.snapshot.static_meta.action_mask_prg_cell.to(device=device, dtype=torch.bool) for graph in graphs],
        dim=0,
    )
    return action_mask_ue, action_mask_cell_ue, action_mask_prg_cell


def _build_batched_edges(
    graphs: Sequence[SparseEntityGraph],
    device: torch.device,
) -> Dict[str, object]:
    ref = graphs[0].snapshot.static_meta
    n_cell = int(ref.n_cell)
    n_ue = int(ref.n_ue)
    n_prg_tokens = int(ref.n_cell * ref.n_prg)

    def _relation(name: str, src_stride: int, dst_stride: int):
        ref_edge = graphs[0].edges[name]
        indices = []
        attrs = []
        for batch_idx, graph in enumerate(graphs):
            edge = graph.edges[name]
            idx = edge.index.to(device=device, dtype=torch.long).clone()
            idx[0] += batch_idx * src_stride
            idx[1] += batch_idx * dst_stride
            indices.append(idx)
            if edge.attr is not None:
                attrs.append(edge.attr.to(device=device))
        merged_index = (
            torch.cat(indices, dim=1) if indices else torch.zeros((2, 0), dtype=torch.long, device=device)
        )
        merged_attr = torch.cat(attrs, dim=0) if attrs else None
        return type(ref_edge)(
            name=ref_edge.name,
            src_type=ref_edge.src_type,
            dst_type=ref_edge.dst_type,
            index=merged_index,
            attr=merged_attr,
        )

    return {
        "E_CC": _relation("E_CC", n_cell, n_cell),
        "E_CU": _relation("E_CU", n_cell, n_ue),
        "E_CP": _relation("E_CP", n_cell, n_prg_tokens),
        "E_UP": _relation("E_UP", n_ue, n_prg_tokens),
        "E_PP": _relation("E_PP", n_prg_tokens, n_prg_tokens),
    }


def _forward_joint_head_batched(
    actor: StageBHGraphPolicy,
    *,
    cell_x: torch.Tensor,
    ue_x: torch.Tensor,
    prg_x: torch.Tensor,
    graphs: Sequence[SparseEntityGraph],
    batched_up_index: torch.Tensor | None = None,
    batched_up_attr: torch.Tensor | None = None,
    slot_ue_class_for_prg: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    device = cell_x.device
    action_out = actor.action_head.forward_batched(
        cell_x,
        ue_x,
        prg_x,
        graphs,
        batched_up_index=batched_up_index,
        batched_up_attr=batched_up_attr,
        slot_ue_class=slot_ue_class_for_prg,
    )
    action_mask_ue, action_mask_cell_ue, action_mask_prg_cell = _stack_graph_masks(graphs, device)
    return {
        **action_out,
        "cell_embeddings": cell_x,
        "ue_embeddings": ue_x,
        "prg_embeddings": prg_x,
        "action_mask_ue": action_mask_ue,
        "action_mask_cell_ue": action_mask_cell_ue,
        "action_mask_prg_cell": action_mask_prg_cell,
    }


def _evaluate_graph_batch(
    actor: StageBHGraphPolicy,
    critic: GraphValueHead,
    graphs: Sequence[SparseEntityGraph],
    *,
    state_norm: RunningNorm | None = None,
    slot_ue_class_for_prg: torch.Tensor | None = None,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    _require_compatible_graph_batch(graphs)
    device = next(actor.parameters()).device
    ref = graphs[0].snapshot.static_meta
    batch_size = len(graphs)

    cell_x = actor.cell_residual(
        actor.cell_encoder(
            torch.stack([graph.snapshot.cell_features.values.to(device=device) for graph in graphs], dim=0)
        )
    )
    ue_x = actor.ue_residual(
        actor.ue_encoder(
            torch.stack([graph.snapshot.ue_features.values.to(device=device) for graph in graphs], dim=0)
        )
    )
    prg_x = actor.prg_residual(
        actor.prg_encoder(
            torch.stack([graph.snapshot.prg_features.values.to(device=device) for graph in graphs], dim=0)
        )
    )

    flat_cell_x = cell_x.reshape(batch_size * ref.n_cell, -1)
    flat_ue_x = ue_x.reshape(batch_size * ref.n_ue, -1)
    flat_prg_x = prg_x.reshape(batch_size * ref.n_cell * ref.n_prg, -1)
    batched_graph = SparseEntityGraph(
        snapshot=graphs[0].snapshot,
        edges=_build_batched_edges(graphs, device=device),
    )
    for layer in actor.message_layers:
        flat_cell_x, flat_ue_x, flat_prg_x = layer(flat_cell_x, flat_ue_x, flat_prg_x, batched_graph)

    cell_x = flat_cell_x.reshape(batch_size, ref.n_cell, -1)
    ue_x = flat_ue_x.reshape(batch_size, ref.n_ue, -1)
    prg_x = flat_prg_x.reshape(batch_size, ref.n_cell * ref.n_prg, -1)
    actor_out = _forward_joint_head_batched(
        actor,
        cell_x=cell_x,
        ue_x=ue_x,
        prg_x=prg_x,
        graphs=graphs,
        batched_up_index=batched_graph.edges["E_UP"].index,
        batched_up_attr=batched_graph.edges["E_UP"].attr,
        slot_ue_class_for_prg=slot_ue_class_for_prg,
    )
    state_vec = _pool_actor_state_batch(actor_out).detach()
    if state_norm is not None:
        state_vec = state_norm.normalize(state_vec)
    value = critic(state_vec)
    return actor_out, value


def _graph_masks(graph: SparseEntityGraph, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    meta = graph.snapshot.static_meta
    action_mask_ue = (
        meta.action_mask_ue
        if meta.action_mask_ue is not None
        else torch.ones((meta.n_ue,), dtype=torch.bool)
    ).to(device=device, dtype=torch.bool).unsqueeze(0)
    action_mask_cell_ue = meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).unsqueeze(0)
    action_mask_prg_cell = meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).unsqueeze(0)
    return action_mask_ue, action_mask_cell_ue, action_mask_prg_cell


def _masked_ue_logits(
    actor_out: Dict[str, torch.Tensor],
    graph: SparseEntityGraph,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = actor_out["ue_logits"].device
    action_mask_ue, action_mask_cell_ue, _ = _graph_masks(graph, device)
    ue_logits, ue_valid = apply_ue_action_mask(
        actor_out["ue_logits"].unsqueeze(0),
        action_mask_ue,
        n_cell=graph.snapshot.static_meta.n_cell,
        action_mask_cell_ue=action_mask_cell_ue,
    )
    return ue_logits.squeeze(0), ue_valid.squeeze(0)


def _reduce_multi_categorical(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_class = logits.shape[-1]
    logp_all = F.log_softmax(logits, dim=-1)
    probs = logp_all.exp()

    valid = target != -100
    safe_target = target.clamp(min=0, max=n_class - 1)
    chosen_logp = logp_all.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * logp_all).sum(dim=-1)

    bsz = logits.shape[0]
    valid_f = valid.reshape(bsz, -1).float()
    chosen_logp = chosen_logp.reshape(bsz, -1)
    entropy = entropy.reshape(bsz, -1)

    denom = valid_f.sum(dim=-1).clamp_min(1.0)
    mean_logp = (chosen_logp * valid_f).sum(dim=-1) / denom
    mean_entropy = (entropy * valid_f).sum(dim=-1) / denom
    return mean_logp, mean_entropy


def _masked_prg_logits(
    actor_out: Dict[str, torch.Tensor],
    graph: SparseEntityGraph,
    *,
    ue_action_class: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = actor_out["ue_logits"].device
    meta = graph.snapshot.static_meta
    action_mask_ue = (
        meta.action_mask_ue
        if meta.action_mask_ue is not None
        else torch.ones((meta.n_ue,), dtype=torch.bool)
    ).to(device=device, dtype=torch.bool).unsqueeze(0)
    action_mask_cell_ue = meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).unsqueeze(0)
    action_mask_prg_cell = meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).reshape(
        1,
        meta.n_cell,
        meta.n_prg,
    )
    slot_prior_ue = (
        ue_action_class
        if ue_action_class is not None
        else actor_out["slot_prior_ue_class"]
    ).to(device=device, dtype=torch.long).unsqueeze(0)
    selected_slot_mask = build_slot_selection_mask(
        slot_prior_ue,
        action_mask_ue,
        n_cell=meta.n_cell,
        action_mask_cell_ue=action_mask_cell_ue,
    )
    prg_logits, prg_valid = apply_prg_action_mask(
        actor_out["prg_logits"].unsqueeze(0),
        action_mask_prg_cell,
        n_cell=meta.n_cell,
        action_mask_cell_ue=action_mask_cell_ue,
        action_mask_ue=action_mask_ue,
        selected_slot_mask=selected_slot_mask,
    )
    prg_logits = prg_logits.squeeze(0)
    prg_valid = prg_valid.squeeze(0)
    return prg_logits, prg_valid


def _masked_prg_logits_batch(
    actor_out: Dict[str, torch.Tensor],
    graphs: Sequence[SparseEntityGraph],
    *,
    ue_action_class: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = actor_out["ue_logits"].device
    batch_size = len(graphs)
    ref = graphs[0].snapshot.static_meta
    slot_prior_ue = (
        ue_action_class
        if ue_action_class is not None
        else actor_out["slot_prior_ue_class"]
    ).to(device=device, dtype=torch.long)
    action_mask_ue = torch.stack(
        [
            (
                graph.snapshot.static_meta.action_mask_ue
                if graph.snapshot.static_meta.action_mask_ue is not None
                else torch.ones((ref.n_ue,), dtype=torch.bool)
            ).to(device=device, dtype=torch.bool)
            for graph in graphs
        ],
        dim=0,
    )
    action_mask_cell_ue = torch.stack(
        [graph.snapshot.static_meta.action_mask_cell_ue.to(device=device, dtype=torch.bool) for graph in graphs],
        dim=0,
    )
    action_mask_prg_cell = torch.stack(
        [
            graph.snapshot.static_meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).reshape(
                ref.n_cell,
                ref.n_prg,
            )
            for graph in graphs
        ],
        dim=0,
    )
    selected_slot_mask = build_slot_selection_mask(
        slot_prior_ue,
        action_mask_ue,
        n_cell=ref.n_cell,
        action_mask_cell_ue=action_mask_cell_ue,
    )
    prg_logits, prg_valid = apply_prg_action_mask(
        actor_out["prg_logits"].reshape(batch_size, ref.n_cell, ref.n_prg, -1),
        action_mask_prg_cell,
        n_cell=ref.n_cell,
        action_mask_cell_ue=action_mask_cell_ue,
        action_mask_ue=action_mask_ue,
        selected_slot_mask=selected_slot_mask,
    )
    return prg_logits, prg_valid


def _repair_ue_action_for_coverage(
    ue_class: torch.Tensor,
    graph: SparseEntityGraph,
    actor_out: Dict[str, torch.Tensor],
    *,
    max_repairs_per_cell: int = 0,
    min_backlog_bytes: float = 1.0,
) -> tuple[torch.Tensor, float]:
    device = ue_class.device
    meta = graph.snapshot.static_meta
    n_ue = int(meta.n_ue)
    max_repairs = int(max_repairs_per_cell)
    if n_ue <= 0 or max_repairs <= 0:
        return ue_class, 0.0

    action_mask_ue = (
        meta.action_mask_ue
        if meta.action_mask_ue is not None
        else torch.ones((n_ue,), dtype=torch.bool)
    ).to(device=device, dtype=torch.bool)
    action_mask_cell_ue = meta.action_mask_cell_ue.to(device=device, dtype=torch.bool)
    demand_mask = demand_active_ue_mask_from_features(
        graph.snapshot.ue_features.values.to(device=device),
        action_mask_ue=action_mask_ue,
    )
    slot_to_cell = actor_out["slot_to_cell"].to(device=device, dtype=torch.long)
    slot_valid_mask = actor_out["slot_valid_mask"].to(device=device, dtype=torch.bool)
    ue_features = graph.snapshot.ue_features.values.to(device=device)
    repaired = ue_class.to(device=device, dtype=torch.long).clone()
    repair_count = 0

    def _feature(idx: int) -> torch.Tensor:
        if ue_features.shape[1] <= idx:
            return torch.zeros((n_ue,), dtype=torch.float32, device=device)
        return ue_features[:, idx].float()

    backlog = torch.clamp_min(_feature(0), 0.0).detach().cpu()
    stale = torch.clamp_min(_feature(7), 0.0).detach().cpu()
    hol_delay = torch.clamp_min(_feature(8), 0.0).detach().cpu()
    ttl_slack = _feature(9).detach().cpu()
    recent_deficit = torch.clamp_min(_feature(11), 0.0).detach().cpu()

    for cell_idx in range(int(meta.n_cell)):
        cell_slots = torch.nonzero(slot_valid_mask & (slot_to_cell == cell_idx), as_tuple=False).flatten()
        if cell_slots.numel() == 0:
            continue

        selected: set[int] = set()
        replace_slots: list[int] = []
        for slot_id in cell_slots.detach().cpu().tolist():
            ue_idx = int(repaired[slot_id].item())
            valid = (
                0 <= ue_idx < n_ue
                and bool(action_mask_ue[ue_idx].item())
                and bool(action_mask_cell_ue[cell_idx, ue_idx].item())
            )
            if not valid or ue_idx in selected:
                replace_slots.append(slot_id)
            else:
                selected.add(ue_idx)

        if not replace_slots:
            continue
        candidate_mask = action_mask_cell_ue[cell_idx] & action_mask_ue & demand_mask
        if min_backlog_bytes > 0.0:
            candidate_mask = candidate_mask & (_feature(0) >= float(min_backlog_bytes))
        for ue_idx in selected:
            candidate_mask[ue_idx] = False
        candidate_ids = torch.nonzero(candidate_mask, as_tuple=False).flatten().detach().cpu().tolist()
        if not candidate_ids:
            continue

        def _priority(ue_idx: int) -> tuple[float, float, float, float, float]:
            ttl = float(ttl_slack[ue_idx].item())
            ttl_priority = -ttl if ttl >= 0.0 else -1.0e9
            return (
                float(backlog[ue_idx].item()),
                float(recent_deficit[ue_idx].item()),
                float(stale[ue_idx].item()),
                float(hol_delay[ue_idx].item()),
                ttl_priority,
            )

        candidate_ids.sort(key=_priority, reverse=True)
        for slot_id, ue_idx in zip(replace_slots[:max_repairs], candidate_ids):
            repaired[slot_id] = int(ue_idx)
            repair_count += 1

    return repaired, float(repair_count)


def _prg_class_from_type0_alloc(graph: SparseEntityGraph, action_prg_alloc: torch.Tensor) -> torch.Tensor:
    meta = graph.snapshot.static_meta
    device = action_prg_alloc.device
    alloc = torch.as_tensor(action_prg_alloc, dtype=torch.long, device=device).reshape(int(meta.n_prg * meta.n_cell))
    owner_cell = meta.prg_owner_cell.to(device=device, dtype=torch.long)
    local_prg = meta.prg_local_index.to(device=device, dtype=torch.long)
    linear_prg = local_prg * int(meta.n_cell) + owner_cell
    prg_class = alloc.gather(0, linear_prg)
    blank_class = torch.full_like(prg_class, int(meta.n_sched_ue))
    prg_class = torch.where((prg_class >= 0) & (prg_class < int(meta.n_sched_ue)), prg_class, blank_class)
    return prg_class.reshape(int(meta.n_cell), int(meta.n_prg))


def _action_logp_entropy(
    actor_out: Dict[str, torch.Tensor],
    graph: SparseEntityGraph,
    ue_action_class: torch.Tensor,
    prg_action_class: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = actor_out["ue_logits"].device
    ue_logits, ue_valid = _masked_ue_logits(actor_out, graph)
    ue_target = ue_action_class.to(device=device, dtype=torch.long).unsqueeze(0)
    ue_target, _ = sanitize_targets(ue_target, ue_valid.unsqueeze(0), ignore_index=-100)
    ue_logp, ue_entropy = _reduce_multi_categorical(ue_logits.unsqueeze(0), ue_target)

    prg_logits, prg_valid = _masked_prg_logits(actor_out, graph, ue_action_class=ue_action_class)
    prg_logits = prg_logits.unsqueeze(0)
    prg_target = prg_action_class.to(device=device, dtype=torch.long).unsqueeze(0)
    prg_target, _ = sanitize_targets(prg_target, prg_valid.unsqueeze(0), ignore_index=-100)
    prg_logp, prg_entropy = _reduce_multi_categorical(prg_logits, prg_target)
    return (ue_logp + prg_logp).squeeze(0), (0.5 * (ue_entropy + prg_entropy)).squeeze(0)


def _action_logp_entropy_batch(
    actor_out: Dict[str, torch.Tensor],
    graphs: Sequence[SparseEntityGraph],
    ue_action_class: torch.Tensor,
    prg_action_class: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = actor_out["ue_logits"].device
    ref = graphs[0].snapshot.static_meta
    action_mask_ue = torch.stack(
        [
            (
                graph.snapshot.static_meta.action_mask_ue
                if graph.snapshot.static_meta.action_mask_ue is not None
                else torch.ones((ref.n_ue,), dtype=torch.bool)
            ).to(device=device, dtype=torch.bool)
            for graph in graphs
        ],
        dim=0,
    )
    action_mask_cell_ue = torch.stack(
        [graph.snapshot.static_meta.action_mask_cell_ue.to(device=device, dtype=torch.bool) for graph in graphs],
        dim=0,
    )
    ue_logits, ue_valid = apply_ue_action_mask(
        actor_out["ue_logits"].reshape(len(graphs), ref.n_sched_ue, ref.n_ue + 1),
        action_mask_ue,
        n_cell=ref.n_cell,
        action_mask_cell_ue=action_mask_cell_ue,
    )
    ue_target = ue_action_class.to(device=device, dtype=torch.long)
    ue_target, _ = sanitize_targets(ue_target, ue_valid, ignore_index=-100)
    ue_logp, ue_entropy = _reduce_multi_categorical(ue_logits, ue_target)

    prg_logits, prg_valid = _masked_prg_logits_batch(actor_out, graphs, ue_action_class=ue_action_class)
    prg_target = prg_action_class.to(device=device, dtype=torch.long)
    prg_target, _ = sanitize_targets(prg_target, prg_valid, ignore_index=-100)
    prg_logp, prg_entropy = _reduce_multi_categorical(prg_logits, prg_target)
    return ue_logp + prg_logp, 0.5 * (ue_entropy + prg_entropy)


def _sample_joint_action(
    actor: StageBHGraphPolicy,
    actor_out: Dict[str, torch.Tensor],
    graph: SparseEntityGraph,
    *,
    slot_repair_max_per_cell: int = 0,
    slot_repair_min_backlog_bytes: float = 1.0,
) -> Dict[str, torch.Tensor | float]:
    device = actor_out["ue_logits"].device
    ue_logits_masked, _ue_valid = _masked_ue_logits(actor_out, graph)
    ue_dist = Categorical(logits=ue_logits_masked)
    ue_class_raw = ue_dist.sample()
    ue_class, coverage_repair_count = _repair_ue_action_for_coverage(
        ue_class_raw,
        graph,
        actor_out,
        max_repairs_per_cell=int(slot_repair_max_per_cell),
        min_backlog_bytes=float(slot_repair_min_backlog_bytes),
    )
    ue_logp = ue_dist.log_prob(ue_class).mean()
    ue_entropy = ue_dist.entropy().mean()

    prg_actor_out = actor.forward_graph(graph, slot_ue_class_for_prg=ue_class)
    prg_logits_masked, prg_valid = _masked_prg_logits(prg_actor_out, graph, ue_action_class=ue_class)
    prg_dist = Categorical(logits=prg_logits_masked)
    prg_class = prg_dist.sample()
    prg_logp = prg_dist.log_prob(prg_class).mean()
    prg_entropy = prg_dist.entropy().mean()

    decoded = decode_joint_actions_to_type0(
        graph,
        ue_class.detach().cpu(),
        prg_class.detach().cpu().to(torch.long),
    )
    action_logp = ue_logp + prg_logp
    action_entropy = 0.5 * (ue_entropy + prg_entropy)
    blank_ratio = float((prg_class == graph.snapshot.static_meta.n_sched_ue).float().mean().item())
    action_fill_ratio = float(decoded.prg_chosen_mask.float().mean().item())
    final_blank_ratio = float(1.0 - action_fill_ratio)
    post_decode_drop_ratio = float(max(0.0, 1.0 - blank_ratio - action_fill_ratio))
    cap_drop_count = float(decoded.cap_drop_count)
    fallback_success_count = float(decoded.fallback_success_count)
    final_blank_count = float(decoded.final_blank_count)
    chosen = decoded.chosen_ue_per_prg[decoded.chosen_ue_per_prg >= 0]
    action_unique_ue_count = float(torch.unique(chosen).numel()) if chosen.numel() > 0 else 0.0
    e_up = graph.edges["E_UP"].index
    candidate_unique_ue_count = float(torch.unique(e_up[0]).numel()) if e_up.numel() > 0 else 0.0
    demand_active_ue_mask = demand_active_ue_mask_from_features(
        graph.snapshot.ue_features.values,
        action_mask_ue=graph.snapshot.static_meta.action_mask_ue,
    )
    demand_active_ue_count = float(demand_active_ue_mask.sum().item())
    candidate_coverage_ratio = candidate_unique_ue_count / max(demand_active_ue_count, 1.0)

    return {
        "ue_action_class": ue_class.detach().cpu().to(torch.long),
        "prg_action_class": prg_class.detach().cpu().to(torch.long),
        "action_ue_select": decoded.action_ue_select.cpu().numpy(),
        "action_prg_alloc": decoded.action_prg_alloc.cpu().numpy(),
        "chosen_ue_per_prg": decoded.chosen_ue_per_prg.cpu(),
        "prg_chosen_mask": decoded.prg_chosen_mask,
        "blank_ratio": blank_ratio,
        "post_decode_drop_ratio": post_decode_drop_ratio,
        "final_blank_ratio": final_blank_ratio,
        "cap_drop_count": cap_drop_count,
        "fallback_success_count": fallback_success_count,
        "final_blank_count": final_blank_count,
        "action_fill_ratio": action_fill_ratio,
        "action_unique_ue_count": action_unique_ue_count,
        "candidate_unique_ue_count": candidate_unique_ue_count,
        "demand_active_ue_count": demand_active_ue_count,
        "candidate_coverage_ratio": candidate_coverage_ratio,
        "coverage_repair_count": coverage_repair_count,
        "logp": action_logp.detach().cpu().to(torch.float32),
        "entropy": float(action_entropy.item()),
    }


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_len = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype)
    for t in range(t_len - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def _load_checkpoint_compatible(model: nn.Module, ckpt_path: str) -> tuple[int, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("actor_state") or ckpt
    target_state = model.state_dict()
    compatible = {}
    for k, v in state.items():
        if k in target_state and target_state[k].shape == v.shape:
            compatible[k] = v
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(target_state)


def _run_online_imitation_warm_start(
    *,
    args: argparse.Namespace,
    actor: StageBHGraphPolicy,
    snapshot_builder: FeatureSnapshotBuilder,
    edge_generator: SparseEdgeGenerator,
    device: torch.device,
) -> Dict[str, float]:
    max_prg_candidates = _resolve_prg_candidate_count(args)
    teacher = build_joint_teacher(
        str(args.teacher_mode),
        heuristic_cfg=HeuristicTeacherConfig(max_candidates=max_prg_candidates),
        max_candidates=max_prg_candidates,
    )
    imitation_lr = float(args.imitation_lr) if float(args.imitation_lr) > 0.0 else (
        float(args.actor_lr) if float(args.actor_lr) > 0.0 else float(args.lr)
    )
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=imitation_lr,
        weight_decay=float(args.weight_decay),
    )
    runner = OnlineEpisodeRunner(args)
    max_steps = int(args.imitation_max_steps) if int(args.imitation_max_steps) > 0 else int(args.episode_horizon)
    log_every = max(int(args.rollout_log_every_steps), 0)
    loss_trace: List[float] = []
    acc_trace: List[float] = []
    reward_trace: List[float] = []
    goodput_trace: List[float] = []
    tb_err_trace: List[float] = []
    blank_trace: List[float] = []
    ue_loss_weight = 1.0

    print(
        f"[online-ppo] imitation_warm_start episodes={int(args.imitation_episodes)} "
        f"teacher_mode={str(args.teacher_mode)} max_steps={max_steps} "
        f"log_every={log_every if log_every > 0 else 'off'} "
        f"ue_loss_weight={ue_loss_weight:.2f} slot_prior=learned_joint"
    )

    try:
        for episode_idx in range(1, int(args.imitation_episodes) + 1):
            obs, _reward0, _info, _episode_seed, _topology_seed, resumed_stream = runner.begin_rollout(episode_idx)
            if not resumed_stream:
                snapshot_builder.reset()
            step_idx = 0
            episode_loss_trace: List[float] = []
            episode_acc_trace: List[float] = []
            episode_reward_trace: List[float] = []
            episode_goodput_trace: List[float] = []
            episode_tb_err_trace: List[float] = []
            episode_blank_trace: List[float] = []
            episode_t0 = time.perf_counter()
            while True:
                graph = _graph_to_device(edge_generator.build_graph(snapshot_builder.build(obs)), device)
                meta = graph.snapshot.static_meta
                teacher_out = teacher.label_graph(graph)
                teacher_target_ue = teacher_out["target_ue_class"].to(device=device, dtype=torch.long)
                teacher_target_prg = teacher_out["target_prg_class"].to(device=device, dtype=torch.long)

                actor.train(True)
                out = actor.forward_graph(graph)
                ue_logits_raw = out["ue_logits"].unsqueeze(0)
                prg_logits_raw = out["prg_logits"].unsqueeze(0)

                action_mask_ue = (
                    meta.action_mask_ue
                    if meta.action_mask_ue is not None
                    else torch.ones((meta.n_ue,), dtype=torch.bool, device=device)
                ).to(device=device, dtype=torch.bool).unsqueeze(0)
                action_mask_cell_ue = meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).unsqueeze(0)

                ue_logits, ue_valid = apply_ue_action_mask(
                    ue_logits_raw,
                    action_mask_ue,
                    n_cell=meta.n_cell,
                    action_mask_cell_ue=action_mask_cell_ue,
                )
                ue_target, _ue_bad = sanitize_targets(teacher_target_ue.unsqueeze(0), ue_valid)
                selected_slot_mask = build_slot_selection_mask(
                    teacher_target_ue.unsqueeze(0),
                    action_mask_ue,
                    n_cell=meta.n_cell,
                    action_mask_cell_ue=action_mask_cell_ue,
                )
                prg_logits, prg_valid = apply_prg_action_mask(
                    prg_logits_raw,
                    meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).reshape(1, meta.n_cell, meta.n_prg),
                    n_cell=meta.n_cell,
                    action_mask_cell_ue=action_mask_cell_ue,
                    action_mask_ue=action_mask_ue,
                    selected_slot_mask=selected_slot_mask,
                )
                prg_target, _prg_bad = sanitize_targets(teacher_target_prg.unsqueeze(0), prg_valid)

                ue_loss = F.cross_entropy(
                    ue_logits.reshape(-1, ue_logits.shape[-1]),
                    ue_target.reshape(-1),
                    ignore_index=-100,
                )
                prg_loss = F.cross_entropy(
                    prg_logits.reshape(-1, prg_logits.shape[-1]),
                    prg_target.reshape(-1),
                    ignore_index=-100,
                )
                loss = prg_loss + ue_loss_weight * ue_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(args.max_grad_norm))
                optimizer.step()

                with torch.no_grad():
                    ue_acc = float(((ue_logits.argmax(dim=-1) == ue_target) & (ue_target != -100)).float().sum().item())
                    ue_den = max(1, int((ue_target != -100).sum().item()))
                    ue_acc = ue_acc / float(ue_den)
                    prg_acc = float(((prg_logits.argmax(dim=-1) == prg_target) & (prg_target != -100)).float().sum().item())
                    prg_den = max(1, int((prg_target != -100).sum().item()))
                    prg_acc = prg_acc / float(prg_den)
                    acc_trace.append(0.5 * (ue_acc + prg_acc))
                loss_trace.append(float(loss.item()))

                assert runner.env is not None
                next_obs, _reward_raw, done, _next_info = runner.env.step(
                    teacher_out["action_ue_select"].detach().cpu().numpy(),
                    teacher_out["action_prg_alloc"].detach().cpu().numpy(),
                )
                runner.record_transition(next_obs, float(_reward_raw), _next_info)
                reward_metrics = extract_reward_metrics(_next_info)
                reward_raw = float(_reward_raw)
                goodput_mbps = float(reward_metrics.get("goodput_mbps", 0.0))
                tb_err_rate = float(reward_metrics.get("tb_err_rate", 0.0))
                teacher_blank_ratio = float(teacher_out["teacher_blank_ratio"].item())
                step_idx += 1
                joint_acc = 0.5 * (ue_acc + prg_acc)
                reward_trace.append(reward_raw)
                goodput_trace.append(goodput_mbps)
                tb_err_trace.append(tb_err_rate)
                blank_trace.append(teacher_blank_ratio)
                episode_loss_trace.append(float(loss.item()))
                episode_acc_trace.append(joint_acc)
                episode_reward_trace.append(reward_raw)
                episode_goodput_trace.append(goodput_mbps)
                episode_tb_err_trace.append(tb_err_rate)
                episode_blank_trace.append(teacher_blank_ratio)
                if log_every > 0 and ((step_idx % log_every) == 0 or done or step_idx >= max_steps):
                    print(
                    f"[imitation ep {episode_idx:04d}] "
                    f"tti={step_idx:04d}/{max_steps} "
                    f"loss={float(np.mean(episode_loss_trace[-min(len(episode_loss_trace), log_every):])):.4f} "
                    f"ue_acc={ue_acc:.3f} prg_acc={prg_acc:.3f} joint={joint_acc:.3f} "
                        f"reward={float(np.mean(episode_reward_trace[-min(len(episode_reward_trace), log_every):])):.4f} "
                        f"goodput={float(np.mean(episode_goodput_trace[-min(len(episode_goodput_trace), log_every):])):.4f} "
                        f"tb_err={float(np.mean(episode_tb_err_trace[-min(len(episode_tb_err_trace), log_every):])):.4f} "
                        f"blank={float(np.mean(episode_blank_trace[-min(len(episode_blank_trace), log_every):])):.4f}"
                    )
                if done or step_idx >= max_steps:
                    print(
                        f"[imitation ep {episode_idx:04d}] "
                        f"summary steps={step_idx} "
                        f"loss={float(np.mean(episode_loss_trace)):.4f} "
                        f"joint={float(np.mean(episode_acc_trace)):.3f} "
                        f"reward={float(np.mean(episode_reward_trace)):.4f} "
                        f"goodput={float(np.mean(episode_goodput_trace)):.4f} "
                        f"tb_err={float(np.mean(episode_tb_err_trace)):.4f} "
                        f"blank={float(np.mean(episode_blank_trace)):.4f} "
                        f"time={time.perf_counter() - episode_t0:.1f}s"
                    )
                    break
                obs = next_obs
    finally:
        runner.close_episode(graceful=False)

    return {
        "episodes": float(int(args.imitation_episodes)),
        "steps": float(len(loss_trace)),
        "mean_loss": float(np.mean(loss_trace)) if loss_trace else 0.0,
        "mean_acc": float(np.mean(acc_trace)) if acc_trace else 0.0,
        "mean_reward": float(np.mean(reward_trace)) if reward_trace else 0.0,
        "mean_goodput": float(np.mean(goodput_trace)) if goodput_trace else 0.0,
        "mean_tb_err": float(np.mean(tb_err_trace)) if tb_err_trace else 0.0,
        "mean_teacher_blank_ratio": float(np.mean(blank_trace)) if blank_trace else 0.0,
        "lr": float(imitation_lr),
    }


def _collect_rollout(
    *,
    runner: OnlineEpisodeRunner,
    snapshot_builder: FeatureSnapshotBuilder,
    edge_generator: SparseEdgeGenerator,
    actor: StageBHGraphPolicy,
    critic: GraphValueHead,
    device: torch.device,
    rollout_steps: int,
    reward_mode: str,
    reward_cfg: Dict[str, float],
    episode_idx: int,
    rollout_log_every_steps: int,
    state_norm: RunningNorm | None = None,
    reward_norm: RunningScalarNorm | None = None,
    normalize_reward: bool = True,
    per_ue_log_dir: Path | None = None,
    native_reject_log_dir: Path | None = None,
    per_ue_tracker: PerUeDiagnosticsTracker | None = None,
    slot_repair_max_per_cell: int = 0,
    slot_repair_min_backlog_bytes: float = 1.0,
) -> tuple[List[RolloutStep], Dict[str, float], PerUeDiagnosticsTracker]:
    actor.eval()
    critic.eval()

    obs, _reward0, info, episode_seed, topology_seed, resumed_stream = runner.begin_rollout(episode_idx)
    if not resumed_stream:
        snapshot_builder.reset()
    graph = _graph_to_device(edge_generator.build_graph(snapshot_builder.build(obs)), device)
    cached_actor_out = None
    cached_value = None
    steps: List[RolloutStep] = []
    if per_ue_tracker is None or int(per_ue_tracker.n_ue) != int(graph.snapshot.static_meta.n_ue):
        per_ue_tracker = PerUeDiagnosticsTracker(graph.snapshot.static_meta.n_ue)
    n_cell = int(graph.snapshot.static_meta.n_cell)
    native_reject_cell_sum = {
        key: np.zeros((n_cell,), dtype=np.float64) for key in _NATIVE_REJECT_INFO_KEYS
    }
    native_reject_cell_last = {
        key: np.zeros((n_cell,), dtype=np.float64) for key in _NATIVE_REJECT_INFO_KEYS
    }
    native_reject_seen_count = 0
    python_preflight_cell_sum = {
        key: np.zeros((n_cell,), dtype=np.float64) for key in _PREFLIGHT_REJECT_KEYS
    }
    python_preflight_cell_last = {
        key: np.zeros((n_cell,), dtype=np.float64) for key in _PREFLIGHT_REJECT_KEYS
    }
    native_layout_seen_count = 0
    native_layout_python_slot_count_sum = np.zeros((n_cell,), dtype=np.float64)
    native_layout_native_slot_count_sum = np.zeros((n_cell,), dtype=np.float64)
    native_layout_slot_count_abs_delta_sum = np.zeros((n_cell,), dtype=np.float64)
    native_layout_python_cell_start_sum = np.zeros((n_cell,), dtype=np.float64)
    native_layout_native_cell_start_sum = np.zeros((n_cell,), dtype=np.float64)
    native_layout_cell_start_abs_delta_sum = np.zeros((n_cell,), dtype=np.float64)
    native_layout_python_slot_count_last = np.zeros((n_cell,), dtype=np.float64)
    native_layout_native_slot_count_last = np.zeros((n_cell,), dtype=np.float64)
    native_layout_slot_count_abs_delta_last = np.zeros((n_cell,), dtype=np.float64)
    native_layout_python_cell_start_last = np.zeros((n_cell,), dtype=np.float64)
    native_layout_native_cell_start_last = np.zeros((n_cell,), dtype=np.float64)
    native_layout_cell_start_abs_delta_last = np.zeros((n_cell,), dtype=np.float64)
    native_layout_slot_to_cell_mismatch_total = 0.0
    native_layout_slot_to_cell_mismatch_latest = 0.0

    for step_idx in range(rollout_steps):
        with torch.no_grad():
            if cached_actor_out is None or cached_value is None:
                actor_out, value = _evaluate_graph(actor, critic, graph, state_norm=state_norm)
                if state_norm is not None:
                    state_norm.update(_pool_actor_state(actor_out))
            else:
                actor_out, value = cached_actor_out, cached_value
                cached_actor_out = None
                cached_value = None
            sampled = _sample_joint_action(
                actor,
                actor_out,
                graph,
                slot_repair_max_per_cell=int(slot_repair_max_per_cell),
                slot_repair_min_backlog_bytes=float(slot_repair_min_backlog_bytes),
            )

        python_preflight_counts = {
            key: np.zeros((n_cell,), dtype=np.float64) for key in _PREFLIGHT_REJECT_KEYS
        }
        try:
            preflight_t = type0_action_preflight_reject_counts(
                graph,
                torch.as_tensor(sampled["action_ue_select"]),
                torch.as_tensor(sampled["action_prg_alloc"]),
            )
            for key in _PREFLIGHT_REJECT_KEYS:
                python_preflight_counts[key] = (
                    preflight_t[key].detach().cpu().numpy().astype(np.float64, copy=False).reshape(n_cell)
                )
        except (TypeError, ValueError, KeyError) as exc:
            print(f"[online-ppo] WARN preflight reject diagnostic failed: {exc}")
        for key in _PREFLIGHT_REJECT_KEYS:
            python_preflight_cell_sum[key] += python_preflight_counts[key]
            python_preflight_cell_last[key] = python_preflight_counts[key]

        meta = graph.snapshot.static_meta
        pre_action_mask_ue = (
            meta.action_mask_ue
            if meta.action_mask_ue is not None
            else torch.ones((meta.n_ue,), dtype=torch.bool, device=graph.snapshot.ue_features.values.device)
        ).to(device=graph.snapshot.ue_features.values.device, dtype=torch.bool)
        pre_action_layout = build_type0_slot_layout(
            meta.action_mask_cell_ue.to(device=pre_action_mask_ue.device, dtype=torch.bool).unsqueeze(0),
            n_sched_ue=int(meta.n_sched_ue),
            action_mask_ue=pre_action_mask_ue.unsqueeze(0),
        )
        python_slot_count = (
            pre_action_layout.slot_counts.squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)
        )
        python_cell_start = (
            pre_action_layout.cell_slot_start.squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)
        )
        python_slot_to_cell = (
            pre_action_layout.slot_to_cell.squeeze(0).detach().cpu().numpy().astype(np.int32, copy=False)
        )
        python_slot_valid = (
            pre_action_layout.slot_valid_mask.squeeze(0).detach().cpu().numpy().astype(np.bool_, copy=False)
        )

        ue_features_cpu = graph.snapshot.ue_features.values.detach().cpu()
        action_mask_ue_cpu = (
            graph.snapshot.static_meta.action_mask_ue.detach().cpu()
            if graph.snapshot.static_meta.action_mask_ue is not None
            else None
        )
        demand_mask_cpu = demand_active_ue_mask_from_features(
            ue_features_cpu,
            action_mask_ue=action_mask_ue_cpu,
        ).numpy()
        backlog_bytes_cpu = torch.clamp_min(ue_features_cpu[:, 0], 0.0).numpy()
        hol_delay_ms_cpu = ue_features_cpu[:, 8].numpy() if ue_features_cpu.shape[1] > 8 else None
        ttl_slack_ms_cpu = ue_features_cpu[:, 9].numpy() if ue_features_cpu.shape[1] > 9 else None
        recent_scheduled_ratio_cpu = ue_features_cpu[:, 10].numpy() if ue_features_cpu.shape[1] > 10 else None
        recent_goodput_deficit_norm_cpu = ue_features_cpu[:, 11].numpy() if ue_features_cpu.shape[1] > 11 else None
        chosen_ue_per_prg = sampled["chosen_ue_per_prg"].numpy()
        action_ue_select_cpu = np.asarray(sampled["action_ue_select"], dtype=np.int32).reshape(-1)
        planned_selected_slot_count_cpu = np.zeros((graph.snapshot.static_meta.n_ue,), dtype=np.int32)
        selected_valid = action_ue_select_cpu[
            (action_ue_select_cpu >= 0) & (action_ue_select_cpu < int(graph.snapshot.static_meta.n_ue))
        ]
        if selected_valid.size > 0:
            selected_bincount = np.bincount(
                selected_valid.astype(np.int64, copy=False),
                minlength=int(graph.snapshot.static_meta.n_ue),
            ).astype(np.int32, copy=False)
            planned_selected_slot_count_cpu[: selected_bincount.shape[0]] = selected_bincount
        planned_served_mask_cpu = np.zeros((graph.snapshot.static_meta.n_ue,), dtype=bool)
        planned_assigned_prg_count_cpu = np.zeros((graph.snapshot.static_meta.n_ue,), dtype=np.int32)
        chosen_valid = chosen_ue_per_prg[chosen_ue_per_prg >= 0]
        if chosen_valid.size > 0:
            bincount = np.bincount(
                chosen_valid.astype(np.int64, copy=False),
                minlength=int(graph.snapshot.static_meta.n_ue),
            ).astype(np.int32, copy=False)
            planned_assigned_prg_count_cpu[: bincount.shape[0]] = bincount
            planned_served_mask_cpu = planned_assigned_prg_count_cpu > 0

        assert runner.env is not None
        next_obs, raw_env_reward, done, next_info = runner.env.step(
            sampled["action_ue_select"],
            sampled["action_prg_alloc"],
        )
        runner.record_transition(next_obs, float(raw_env_reward), next_info)
        reward_metrics = extract_reward_metrics(next_info)
        accepted_pre_pdsch_available = bool(
            next_info.get("accepted_pre_pdsch_ue_diagnostics_available", False)
        )
        native_reject_available = bool(next_info.get("native_reject_diagnostics_available", False))
        native_reject_counts = {
            key: np.zeros((n_cell,), dtype=np.float64) for key in _NATIVE_REJECT_INFO_KEYS
        }
        if native_reject_available:
            try:
                for key in _NATIVE_REJECT_INFO_KEYS:
                    native_reject_counts[key] = np.asarray(
                        next_info.get(key),
                        dtype=np.float64,
                    ).reshape(n_cell)
            except (TypeError, ValueError):
                native_reject_available = False
                native_reject_counts = {
                    key: np.zeros((n_cell,), dtype=np.float64) for key in _NATIVE_REJECT_INFO_KEYS
                }
        if native_reject_available:
            native_reject_seen_count += 1
            for key in _NATIVE_REJECT_INFO_KEYS:
                native_reject_cell_sum[key] += native_reject_counts[key]
                native_reject_cell_last[key] = native_reject_counts[key]
        native_slot_layout_available = bool(next_info.get("native_slot_layout_diagnostics_available", False))
        native_slot_count_abs_delta = np.zeros((n_cell,), dtype=np.float64)
        native_cell_start_abs_delta = np.zeros((n_cell,), dtype=np.float64)
        native_slot_to_cell_mismatch_count = 0.0
        if native_slot_layout_available:
            try:
                native_slot_count = np.asarray(
                    next_info.get("native_slot_count_per_cell"), dtype=np.float64
                ).reshape(n_cell)
                native_cell_start = np.asarray(next_info.get("native_cell_slot_start"), dtype=np.float64).reshape(n_cell)
                native_slot_to_cell = np.asarray(next_info.get("native_slot_to_cell"), dtype=np.int32).reshape(
                    int(meta.n_sched_ue)
                )
                native_slot_valid = np.asarray(next_info.get("native_slot_valid"), dtype=np.bool_).reshape(
                    int(meta.n_sched_ue)
                )
                native_slot_count_abs_delta = np.abs(native_slot_count - python_slot_count)
                native_cell_start_abs_delta = np.abs(native_cell_start - python_cell_start)
                slot_compare_mask = native_slot_valid | python_slot_valid
                native_slot_to_cell_mismatch_count = float(
                    np.sum(slot_compare_mask & (native_slot_to_cell != python_slot_to_cell))
                    + np.sum(native_slot_valid != python_slot_valid)
                )
                native_layout_seen_count += 1
                native_layout_python_slot_count_sum += python_slot_count
                native_layout_native_slot_count_sum += native_slot_count
                native_layout_slot_count_abs_delta_sum += native_slot_count_abs_delta
                native_layout_python_cell_start_sum += python_cell_start
                native_layout_native_cell_start_sum += native_cell_start
                native_layout_cell_start_abs_delta_sum += native_cell_start_abs_delta
                native_layout_python_slot_count_last = python_slot_count
                native_layout_native_slot_count_last = native_slot_count
                native_layout_slot_count_abs_delta_last = native_slot_count_abs_delta
                native_layout_python_cell_start_last = python_cell_start
                native_layout_native_cell_start_last = native_cell_start
                native_layout_cell_start_abs_delta_last = native_cell_start_abs_delta
                native_layout_slot_to_cell_mismatch_total += native_slot_to_cell_mismatch_count
                native_layout_slot_to_cell_mismatch_latest = native_slot_to_cell_mismatch_count
            except (TypeError, ValueError):
                native_slot_layout_available = False
        actual_available = bool(next_info.get("actual_ue_diagnostics_available", False))
        per_ue_tracker.update(
            demand_mask=demand_mask_cpu,
            planned_served_mask=planned_served_mask_cpu,
            planned_selected_slot_count=planned_selected_slot_count_cpu,
            planned_assigned_prg_count=planned_assigned_prg_count_cpu,
            backlog_bytes=backlog_bytes_cpu,
            hol_delay_ms=hol_delay_ms_cpu,
            ttl_slack_ms=ttl_slack_ms_cpu,
            recent_scheduled_ratio=recent_scheduled_ratio_cpu,
            recent_goodput_deficit_norm=recent_goodput_deficit_norm_cpu,
            accepted_pre_pdsch_available=accepted_pre_pdsch_available,
            accepted_pre_pdsch_assigned_prg_count=next_info.get("accepted_pre_pdsch_ue_prg_count"),
            actual_available=actual_available,
            actual_assigned_prg_count=next_info.get("actual_ue_prg_count"),
            actual_served_bytes=next_info.get("actual_ue_served_bytes"),
            actual_goodput_bytes=next_info.get("actual_ue_goodput_bytes"),
            actual_tb_tx_count=next_info.get("actual_ue_tb_tx_count"),
            actual_tb_err_count=next_info.get("actual_ue_tb_err_count"),
        )
        per_ue_summary = per_ue_tracker.summary()
        packet_completed_count = float(reward_metrics.get("packet_completed_count", 0.0))
        packet_effective_service_rate_mbps = float(
            reward_metrics.get("packet_effective_service_rate_mbps", 0.0)
        )
        packet_rate_bonus = 0.0
        packet_completion_bonus = 0.0
        aged_backlog_penalty = 0.0
        aged_backlog_coef = 0.0
        if reward_mode == "stageb_v1":
            shaped = compute_stageb_v1_reward(
                raw_env_reward=float(raw_env_reward),
                reward_metrics=reward_metrics,
                snapshot=graph.snapshot,
                goodput_weight=float(reward_cfg["goodput_weight"]),
                tb_err_weight=float(reward_cfg["tb_err_weight"]),
                expiry_weight=float(reward_cfg["expiry_weight"]),
                conflict_weight=float(reward_cfg["conflict_weight"]),
                ici_weight=float(reward_cfg["ici_weight"]),
                urgent_backlog_weight=float(reward_cfg["urgent_backlog_weight"]),
                urgent_ttl_ms=float(reward_cfg["urgent_ttl_ms"]),
                selected_prg_mask=sampled["prg_chosen_mask"],
            )
            reward_raw = float(shaped["reward_shaped"])
            conflict_mean = float(shaped["reward_conflict_mean"])
            ici_mean = float(shaped["reward_ici_mean"])
            urgent_backlog_penalty = float(shaped["reward_urgent_backlog_penalty"])
            backlog_debt_coef = 0.0
            active_prg_goodput_efficiency = 0.0
            harmful_packing_penalty = 0.0
        elif reward_mode == "stageb_blankaware":
            shaped = compute_stageb_blankaware_reward(
                raw_env_reward=float(raw_env_reward),
                reward_metrics=reward_metrics,
                goodput_weight=float(reward_cfg["goodput_weight"]),
                tb_err_weight=float(reward_cfg["tb_err_weight"]),
                expiry_weight=float(reward_cfg["expiry_weight"]),
                conflict_weight=float(reward_cfg["conflict_weight"]),
                ici_weight=float(reward_cfg["ici_weight"]),
                urgent_backlog_weight=float(reward_cfg["urgent_backlog_weight"]),
                actual_unserved_backlog_bytes=float(per_ue_summary["actual_unserved_backlog_bytes"]),
                actual_unserved_demand_ue_count=float(per_ue_summary["actual_unserved_demand_ue_count"]),
                backlog_debt_weight=float(reward_cfg.get("backlog_debt_weight", 0.0)),
                packet_rate_weight=float(reward_cfg.get("packet_rate_weight", 0.0)),
                packet_completion_weight=float(reward_cfg.get("packet_completion_weight", 0.0)),
                packet_rate_scale_mbps=float(reward_cfg.get("packet_rate_scale_mbps", 10.0)),
                packet_completion_scale=float(reward_cfg.get("packet_completion_scale", 36.0)),
                aged_backlog_weight=float(reward_cfg.get("aged_backlog_weight", 0.0)),
                aged_backlog_bytes=float(per_ue_summary["recent100_debt_score_p90_bytes"]),
                aged_backlog_streak=float(per_ue_summary["no_service_streak_p90"]),
            )
            reward_raw = float(shaped["reward_shaped"])
            conflict_mean = _prg_feature_mean(graph, 6)
            ici_mean = _prg_feature_mean(graph, 7)
            urgent_backlog_penalty = float(shaped["reward_backlog_debt_penalty"])
            backlog_debt_coef = float(shaped["reward_backlog_debt_coef"])
            packet_rate_bonus = float(shaped["reward_packet_rate_bonus"])
            packet_completion_bonus = float(shaped["reward_packet_completion_bonus"])
            aged_backlog_penalty = float(shaped["reward_aged_backlog_penalty"])
            aged_backlog_coef = float(shaped["reward_aged_backlog_coef"])
            active_prg_goodput_efficiency = float(shaped["reward_active_prg_goodput_efficiency"])
            harmful_packing_penalty = float(shaped["reward_harmful_packing_penalty"])
        else:
            reward_raw = float(raw_env_reward)
            conflict_mean = _prg_feature_mean(graph, 6)
            ici_mean = _prg_feature_mean(graph, 7)
            urgent_backlog_penalty = 0.0
            backlog_debt_coef = 0.0
            active_prg_goodput_efficiency = 0.0
            harmful_packing_penalty = 0.0
        if reward_norm is not None:
            reward_norm.update(float(reward_raw))
            reward = reward_norm.normalize(float(reward_raw)) if normalize_reward else float(reward_raw)
        else:
            reward = float(reward_raw)

        next_value = 0.0
        next_graph = None
        next_actor_out = None
        next_value_t = None
        if not done and (step_idx + 1) < rollout_steps:
            next_graph = _graph_to_device(edge_generator.build_graph(snapshot_builder.build(next_obs)), device)
            with torch.no_grad():
                next_actor_out, next_value_t = _evaluate_graph(actor, critic, next_graph, state_norm=state_norm)
                if state_norm is not None:
                    state_norm.update(_pool_actor_state(next_actor_out))
                next_value = float(next_value_t.item())

        steps.append(
            RolloutStep(
                graph=graph,
                ue_action_class=sampled["ue_action_class"],
                prg_action_class=sampled["prg_action_class"],
                old_logp=sampled["logp"],
                old_value=float(value.item()),
                next_value=next_value,
                reward=reward,
                reward_raw_env=float(raw_env_reward),
                done=1.0 if done else 0.0,
                entropy=float(sampled["entropy"]),
                blank_ratio=float(sampled["blank_ratio"]),
                post_decode_drop_ratio=float(sampled["post_decode_drop_ratio"]),
                final_blank_ratio=float(sampled["final_blank_ratio"]),
                cap_drop_count=float(sampled["cap_drop_count"]),
                fallback_success_count=float(sampled["fallback_success_count"]),
                final_blank_count=float(sampled["final_blank_count"]),
                action_fill_ratio=float(sampled["action_fill_ratio"]),
                action_unique_ue_count=float(sampled["action_unique_ue_count"]),
                candidate_unique_ue_count=float(sampled["candidate_unique_ue_count"]),
                demand_active_ue_count=float(sampled["demand_active_ue_count"]),
                candidate_coverage_ratio=float(sampled["candidate_coverage_ratio"]),
                unserved_demand_ue_count=float(per_ue_summary["unserved_demand_ue_count"]),
                planned_unserved_demand_ue_count=float(per_ue_summary["planned_unserved_demand_ue_count"]),
                actual_unserved_backlog_bytes=float(per_ue_summary["actual_unserved_backlog_bytes"]),
                accepted_pre_pdsch_diagnostics_available=float(
                    per_ue_summary["accepted_pre_pdsch_diagnostics_available"]
                ),
                actual_diagnostics_available=float(per_ue_summary["actual_diagnostics_available"]),
                native_reject_diagnostics_available=float(native_reject_available),
                native_reject_wrong_cell_slot_count=float(
                    native_reject_counts["native_reject_wrong_cell_slot_count"].sum()
                ),
                native_reject_empty_slot_count=float(
                    native_reject_counts["native_reject_empty_slot_count"].sum()
                ),
                native_reject_prg_mask_count=float(
                    native_reject_counts["native_reject_prg_mask_count"].sum()
                ),
                native_reject_oob_slot_count=float(
                    native_reject_counts["native_reject_oob_slot_count"].sum()
                ),
                native_reject_duplicate_ue_count=float(
                    native_reject_counts["native_reject_duplicate_ue_count"].sum()
                ),
                python_preflight_wrong_cell_slot_count=float(
                    python_preflight_counts["wrong_cell_slot"].sum()
                ),
                python_preflight_empty_slot_count=float(
                    python_preflight_counts["empty_slot"].sum()
                ),
                python_preflight_prg_mask_count=float(
                    python_preflight_counts["prg_mask"].sum()
                ),
                python_preflight_oob_slot_count=float(
                    python_preflight_counts["oob_slot"].sum()
                ),
                python_preflight_duplicate_ue_count=float(
                    python_preflight_counts["duplicate_ue"].sum()
                ),
                native_slot_layout_available=float(native_slot_layout_available),
                native_slot_count_abs_delta_sum=float(native_slot_count_abs_delta.sum()),
                native_cell_start_abs_delta_sum=float(native_cell_start_abs_delta.sum()),
                native_slot_to_cell_mismatch_count=float(native_slot_to_cell_mismatch_count),
                planned_accepted_pre_pdsch_prg_gap_count=float(
                    per_ue_summary["planned_accepted_pre_pdsch_prg_gap_count"]
                ),
                planned_accepted_pre_pdsch_prg_gap_ratio=float(
                    per_ue_summary["planned_accepted_pre_pdsch_prg_gap_ratio"]
                ),
                accepted_pre_pdsch_actual_prg_gap_count=float(
                    per_ue_summary["accepted_pre_pdsch_actual_prg_gap_count"]
                ),
                accepted_pre_pdsch_actual_prg_gap_ratio=float(
                    per_ue_summary["accepted_pre_pdsch_actual_prg_gap_ratio"]
                ),
                planned_native_prg_gap_count=float(per_ue_summary["planned_native_prg_gap_count"]),
                planned_native_prg_gap_ratio=float(per_ue_summary["planned_native_prg_gap_ratio"]),
                no_service_streak_p90=float(per_ue_summary["no_service_streak_p90"]),
                no_service_streak_max=float(per_ue_summary["no_service_streak_max"]),
                recent100_service_rate_p10=float(per_ue_summary["recent100_service_rate_p10"]),
                recent100_service_rate_p50=float(per_ue_summary["recent100_service_rate_p50"]),
                recent100_unserved_backlog_p90_bytes=float(
                    per_ue_summary["recent100_unserved_backlog_p90_bytes"]
                ),
                recent100_debt_score_p90_bytes=float(per_ue_summary["recent100_debt_score_p90_bytes"]),
                recent100_debt_score_max_bytes=float(per_ue_summary["recent100_debt_score_max_bytes"]),
                backlog_p90_bytes=float(per_ue_summary["backlog_p90_bytes"]),
                backlog_max_bytes=float(per_ue_summary["backlog_max_bytes"]),
                goodput_mbps=float(reward_metrics["goodput_mbps"]),
                tb_err_rate=float(reward_metrics["tb_err_rate"]),
                expiry_drop_rate=float(reward_metrics["expiry_drop_rate"]),
                prg_utilization_ratio=float(reward_metrics["prg_utilization_ratio"]),
                prg_reuse_ratio=float(reward_metrics["prg_reuse_ratio"]),
                packet_completed_count=packet_completed_count,
                packet_effective_service_rate_mbps=packet_effective_service_rate_mbps,
                conflict_mean=conflict_mean,
                ici_mean=ici_mean,
                urgent_backlog_penalty=urgent_backlog_penalty,
                backlog_debt_coef=backlog_debt_coef,
                packet_rate_bonus=packet_rate_bonus,
                packet_completion_bonus=packet_completion_bonus,
                aged_backlog_penalty=aged_backlog_penalty,
                aged_backlog_coef=aged_backlog_coef,
                active_prg_goodput_efficiency=active_prg_goodput_efficiency,
                harmful_packing_penalty=harmful_packing_penalty,
                coverage_repair_count=float(sampled["coverage_repair_count"]),
            )
        )

        log_every = int(rollout_log_every_steps)
        step_count = step_idx + 1
        if log_every > 0 and ((step_count % log_every) == 0 or step_count == rollout_steps or done):
            tail = steps[-min(log_every, len(steps)) :]
            print(
                f"[rollout iter {episode_idx:04d}] "
                f"tti={step_count:04d}/{rollout_steps:04d} "
                f"reward={float(np.mean([s.reward for s in tail])):.4f} "
                f"goodput={float(np.mean([s.goodput_mbps for s in tail])):.4f} "
                f"tb_err={float(np.mean([s.tb_err_rate for s in tail])):.4f} "
                f"blank={float(np.mean([s.blank_ratio for s in tail])):.4f} "
                f"drop={float(np.mean([s.post_decode_drop_ratio for s in tail])):.4f} "
                f"fill={float(np.mean([s.action_fill_ratio for s in tail])):.4f} "
                f"capDrop={float(np.mean([s.cap_drop_count for s in tail])):.2f} "
                f"fbOK={float(np.mean([s.fallback_success_count for s in tail])):.2f} "
                f"blankCnt={float(np.mean([s.final_blank_count for s in tail])):.2f} "
                f"uniq_ue={float(np.mean([s.action_unique_ue_count for s in tail])):.2f} "
                f"cand_ue={float(np.mean([s.candidate_unique_ue_count for s in tail])):.2f} "
                f"demand_ue={float(np.mean([s.demand_active_ue_count for s in tail])):.2f} "
                f"cand_cov={float(np.mean([s.candidate_coverage_ratio for s in tail])):.3f} "
                f"unsrv_ue={float(np.mean([s.unserved_demand_ue_count for s in tail])):.2f} "
                f"debtKB={float(np.mean([s.actual_unserved_backlog_bytes for s in tail]))/1024.0:.1f} "
                f"preGap={float(np.mean([s.planned_accepted_pre_pdsch_prg_gap_ratio for s in tail])):.3f} "
                f"postGap={float(np.mean([s.accepted_pre_pdsch_actual_prg_gap_ratio for s in tail])):.3f} "
                f"gap={float(np.mean([s.planned_native_prg_gap_ratio for s in tail])):.3f} "
                f"pyWrong={float(np.mean([s.python_preflight_wrong_cell_slot_count for s in tail])):.1f} "
                f"rejWrong={float(np.mean([s.native_reject_wrong_cell_slot_count for s in tail])):.1f} "
                f"slotCntΔ={float(np.mean([s.native_slot_count_abs_delta_sum for s in tail])):.1f} "
                f"slotMapΔ={float(np.mean([s.native_slot_to_cell_mismatch_count for s in tail])):.1f} "
                f"rejEmpty={float(np.mean([s.native_reject_empty_slot_count for s in tail])):.1f} "
                f"rejDup={float(np.mean([s.native_reject_duplicate_ue_count for s in tail])):.1f} "
                f"repair={float(np.mean([s.coverage_repair_count for s in tail])):.2f} "
                f"streak90={float(np.mean([s.no_service_streak_p90 for s in tail])):.1f} "
                f"streakMax={float(np.mean([s.no_service_streak_max for s in tail])):.1f} "
                f"srv100_p10={float(np.mean([s.recent100_service_rate_p10 for s in tail])):.3f} "
                f"debt100_p90={float(np.mean([s.recent100_debt_score_p90_bytes for s in tail]))/1024.0:.1f}KB "
                f"bklg_p90={float(np.mean([s.backlog_p90_bytes for s in tail]))/1024.0:.1f}KB"
            )

        if done:
            break
        obs, info = next_obs, next_info
        graph = next_graph if next_graph is not None else graph
        cached_actor_out = next_actor_out
        cached_value = next_value_t

    summary = {
        "episode_seed": float(episode_seed),
        "topology_seed": float(topology_seed),
        "rollout_steps": float(len(steps)),
        "rollout_post_eq_enabled_mean": (
            float(np.mean([1.0 if s.graph.snapshot.static_meta.post_eq_sinr is not None else 0.0 for s in steps]))
            if steps
            else 0.0
        ),
        "rollout_post_eq_layer_dim_mean": (
            float(
                np.mean(
                    [
                        (
                            int(s.graph.snapshot.static_meta.post_eq_sinr.shape[-1])
                            if s.graph.snapshot.static_meta.post_eq_sinr is not None
                            and s.graph.snapshot.static_meta.post_eq_sinr.dim() == 3
                            else (1 if s.graph.snapshot.static_meta.post_eq_sinr is not None else 0)
                        )
                        for s in steps
                    ]
                )
            )
            if steps
            else 0.0
        ),
        "rollout_reward_mean": float(np.mean([s.reward for s in steps])) if steps else 0.0,
        "rollout_reward_raw_env_mean": float(np.mean([s.reward_raw_env for s in steps])) if steps else 0.0,
        "rollout_blank_ratio_mean": float(np.mean([s.blank_ratio for s in steps])) if steps else 0.0,
        "rollout_post_decode_drop_ratio_mean": float(np.mean([s.post_decode_drop_ratio for s in steps])) if steps else 0.0,
        "rollout_final_blank_ratio_mean": float(np.mean([s.final_blank_ratio for s in steps])) if steps else 0.0,
        "rollout_cap_drop_count_mean": float(np.mean([s.cap_drop_count for s in steps])) if steps else 0.0,
        "rollout_fallback_success_count_mean": (
            float(np.mean([s.fallback_success_count for s in steps])) if steps else 0.0
        ),
        "rollout_final_blank_count_mean": float(np.mean([s.final_blank_count for s in steps])) if steps else 0.0,
        "rollout_action_fill_ratio_mean": float(np.mean([s.action_fill_ratio for s in steps])) if steps else 0.0,
        "rollout_action_unique_ue_mean": float(np.mean([s.action_unique_ue_count for s in steps])) if steps else 0.0,
        "rollout_candidate_unique_ue_mean": (
            float(np.mean([s.candidate_unique_ue_count for s in steps])) if steps else 0.0
        ),
        "rollout_demand_active_ue_mean": float(np.mean([s.demand_active_ue_count for s in steps])) if steps else 0.0,
        "rollout_candidate_coverage_ratio_mean": (
            float(np.mean([s.candidate_coverage_ratio for s in steps])) if steps else 0.0
        ),
        "rollout_unserved_demand_ue_mean": (
            float(np.mean([s.unserved_demand_ue_count for s in steps])) if steps else 0.0
        ),
        "rollout_planned_unserved_demand_ue_mean": (
            float(np.mean([s.planned_unserved_demand_ue_count for s in steps])) if steps else 0.0
        ),
        "rollout_actual_unserved_backlog_bytes_mean": (
            float(np.mean([s.actual_unserved_backlog_bytes for s in steps])) if steps else 0.0
        ),
        "rollout_accepted_pre_pdsch_ue_diagnostics_available_mean": (
            float(np.mean([s.accepted_pre_pdsch_diagnostics_available for s in steps])) if steps else 0.0
        ),
        "rollout_actual_ue_diagnostics_available_mean": (
            float(np.mean([s.actual_diagnostics_available for s in steps])) if steps else 0.0
        ),
        "rollout_native_reject_diagnostics_available_mean": (
            float(np.mean([s.native_reject_diagnostics_available for s in steps])) if steps else 0.0
        ),
        "rollout_native_reject_wrong_cell_slot_count_mean": (
            float(np.mean([s.native_reject_wrong_cell_slot_count for s in steps])) if steps else 0.0
        ),
        "rollout_native_reject_empty_slot_count_mean": (
            float(np.mean([s.native_reject_empty_slot_count for s in steps])) if steps else 0.0
        ),
        "rollout_native_reject_prg_mask_count_mean": (
            float(np.mean([s.native_reject_prg_mask_count for s in steps])) if steps else 0.0
        ),
        "rollout_native_reject_oob_slot_count_mean": (
            float(np.mean([s.native_reject_oob_slot_count for s in steps])) if steps else 0.0
        ),
        "rollout_native_reject_duplicate_ue_count_mean": (
            float(np.mean([s.native_reject_duplicate_ue_count for s in steps])) if steps else 0.0
        ),
        "rollout_python_preflight_wrong_cell_slot_count_mean": (
            float(np.mean([s.python_preflight_wrong_cell_slot_count for s in steps])) if steps else 0.0
        ),
        "rollout_python_preflight_empty_slot_count_mean": (
            float(np.mean([s.python_preflight_empty_slot_count for s in steps])) if steps else 0.0
        ),
        "rollout_python_preflight_prg_mask_count_mean": (
            float(np.mean([s.python_preflight_prg_mask_count for s in steps])) if steps else 0.0
        ),
        "rollout_python_preflight_oob_slot_count_mean": (
            float(np.mean([s.python_preflight_oob_slot_count for s in steps])) if steps else 0.0
        ),
        "rollout_python_preflight_duplicate_ue_count_mean": (
            float(np.mean([s.python_preflight_duplicate_ue_count for s in steps])) if steps else 0.0
        ),
        "rollout_native_slot_layout_available_mean": (
            float(np.mean([s.native_slot_layout_available for s in steps])) if steps else 0.0
        ),
        "rollout_native_slot_count_abs_delta_sum_mean": (
            float(np.mean([s.native_slot_count_abs_delta_sum for s in steps])) if steps else 0.0
        ),
        "rollout_native_cell_start_abs_delta_sum_mean": (
            float(np.mean([s.native_cell_start_abs_delta_sum for s in steps])) if steps else 0.0
        ),
        "rollout_native_slot_to_cell_mismatch_count_mean": (
            float(np.mean([s.native_slot_to_cell_mismatch_count for s in steps])) if steps else 0.0
        ),
        "rollout_planned_accepted_pre_pdsch_prg_gap_count_mean": (
            float(np.mean([s.planned_accepted_pre_pdsch_prg_gap_count for s in steps])) if steps else 0.0
        ),
        "rollout_planned_accepted_pre_pdsch_prg_gap_ratio_mean": (
            float(np.mean([s.planned_accepted_pre_pdsch_prg_gap_ratio for s in steps])) if steps else 0.0
        ),
        "rollout_accepted_pre_pdsch_actual_prg_gap_count_mean": (
            float(np.mean([s.accepted_pre_pdsch_actual_prg_gap_count for s in steps])) if steps else 0.0
        ),
        "rollout_accepted_pre_pdsch_actual_prg_gap_ratio_mean": (
            float(np.mean([s.accepted_pre_pdsch_actual_prg_gap_ratio for s in steps])) if steps else 0.0
        ),
        "rollout_planned_native_prg_gap_count_mean": (
            float(np.mean([s.planned_native_prg_gap_count for s in steps])) if steps else 0.0
        ),
        "rollout_planned_native_prg_gap_ratio_mean": (
            float(np.mean([s.planned_native_prg_gap_ratio for s in steps])) if steps else 0.0
        ),
        "rollout_no_service_streak_p90_mean": (
            float(np.mean([s.no_service_streak_p90 for s in steps])) if steps else 0.0
        ),
        "rollout_no_service_streak_max_mean": (
            float(np.mean([s.no_service_streak_max for s in steps])) if steps else 0.0
        ),
        "rollout_recent100_service_rate_p10_mean": (
            float(np.mean([s.recent100_service_rate_p10 for s in steps])) if steps else 0.0
        ),
        "rollout_recent100_service_rate_p50_mean": (
            float(np.mean([s.recent100_service_rate_p50 for s in steps])) if steps else 0.0
        ),
        "rollout_recent100_unserved_backlog_p90_bytes_mean": (
            float(np.mean([s.recent100_unserved_backlog_p90_bytes for s in steps])) if steps else 0.0
        ),
        "rollout_recent100_debt_score_p90_bytes_mean": (
            float(np.mean([s.recent100_debt_score_p90_bytes for s in steps])) if steps else 0.0
        ),
        "rollout_recent100_debt_score_max_bytes_mean": (
            float(np.mean([s.recent100_debt_score_max_bytes for s in steps])) if steps else 0.0
        ),
        "rollout_backlog_p90_bytes_mean": (
            float(np.mean([s.backlog_p90_bytes for s in steps])) if steps else 0.0
        ),
        "rollout_backlog_max_bytes_mean": (
            float(np.mean([s.backlog_max_bytes for s in steps])) if steps else 0.0
        ),
        "rollout_goodput_mbps_mean": float(np.mean([s.goodput_mbps for s in steps])) if steps else 0.0,
        "rollout_tb_err_rate_mean": float(np.mean([s.tb_err_rate for s in steps])) if steps else 0.0,
        "rollout_expiry_drop_rate_mean": float(np.mean([s.expiry_drop_rate for s in steps])) if steps else 0.0,
        "rollout_prg_utilization_ratio_mean": (
            float(np.mean([s.prg_utilization_ratio for s in steps])) if steps else 0.0
        ),
        "rollout_prg_reuse_ratio_mean": float(np.mean([s.prg_reuse_ratio for s in steps])) if steps else 0.0,
        "rollout_packet_completed_count_mean": (
            float(np.mean([s.packet_completed_count for s in steps])) if steps else 0.0
        ),
        "rollout_packet_effective_service_rate_mbps_mean": (
            float(np.mean([s.packet_effective_service_rate_mbps for s in steps])) if steps else 0.0
        ),
        "rollout_conflict_mean": float(np.mean([s.conflict_mean for s in steps])) if steps else 0.0,
        "rollout_ici_mean": float(np.mean([s.ici_mean for s in steps])) if steps else 0.0,
        "rollout_active_prg_goodput_efficiency_mean": (
            float(np.mean([s.active_prg_goodput_efficiency for s in steps])) if steps else 0.0
        ),
        "rollout_harmful_packing_penalty_mean": (
            float(np.mean([s.harmful_packing_penalty for s in steps])) if steps else 0.0
        ),
        "rollout_backlog_debt_penalty_mean": (
            float(np.mean([s.urgent_backlog_penalty for s in steps])) if steps else 0.0
        ),
        "rollout_backlog_debt_coef_mean": (
            float(np.mean([s.backlog_debt_coef for s in steps])) if steps else 0.0
        ),
        "rollout_backlog_debt_weighted_penalty_mean": (
            float(np.mean([s.urgent_backlog_penalty * s.backlog_debt_coef for s in steps])) if steps else 0.0
        ),
        "rollout_packet_rate_bonus_mean": (
            float(np.mean([s.packet_rate_bonus for s in steps])) if steps else 0.0
        ),
        "rollout_packet_completion_bonus_mean": (
            float(np.mean([s.packet_completion_bonus for s in steps])) if steps else 0.0
        ),
        "rollout_aged_backlog_penalty_mean": (
            float(np.mean([s.aged_backlog_penalty for s in steps])) if steps else 0.0
        ),
        "rollout_aged_backlog_coef_mean": (
            float(np.mean([s.aged_backlog_coef for s in steps])) if steps else 0.0
        ),
        "rollout_aged_backlog_weighted_penalty_mean": (
            float(np.mean([s.aged_backlog_penalty * s.aged_backlog_coef for s in steps])) if steps else 0.0
        ),
        "rollout_coverage_repair_count_mean": (
            float(np.mean([s.coverage_repair_count for s in steps])) if steps else 0.0
        ),
        "rollout_entropy_mean": float(np.mean([s.entropy for s in steps])) if steps else 0.0,
    }
    if per_ue_log_dir is not None and steps:
        per_ue_log_dir.mkdir(parents=True, exist_ok=True)
        per_ue_rows = per_ue_tracker.rows(
            ue_serving_cell=graph.snapshot.static_meta.ue_serving_cell.detach().cpu().numpy(),
        )
        _write_csv(per_ue_rows, per_ue_log_dir / f"iter_{episode_idx:04d}_per_ue.csv")
        _write_csv(per_ue_rows, per_ue_log_dir / "per_ue_latest.csv")
    if native_reject_log_dir is not None and steps:
        native_reject_log_dir.mkdir(parents=True, exist_ok=True)
        denom = float(max(1, native_reject_seen_count))
        layout_denom = float(max(1, native_layout_seen_count))
        native_reject_rows: List[Dict[str, float]] = []
        for cell_idx in range(n_cell):
            row: Dict[str, float] = {
                "cell_idx": float(cell_idx),
                "diagnostic_tti_count": float(native_reject_seen_count),
            }
            total_all = 0.0
            mean_all = 0.0
            latest_all = 0.0
            for key in _NATIVE_REJECT_INFO_KEYS:
                short = key.removeprefix("native_reject_").removesuffix("_count")
                total = float(native_reject_cell_sum[key][cell_idx])
                mean = total / denom
                latest = float(native_reject_cell_last[key][cell_idx])
                row[f"{short}_total"] = total
                row[f"{short}_mean"] = mean
                row[f"{short}_latest"] = latest
                total_all += total
                mean_all += mean
                latest_all += latest
            for key in _PREFLIGHT_REJECT_KEYS:
                total = float(python_preflight_cell_sum[key][cell_idx])
                mean = total / float(max(1, len(steps)))
                latest = float(python_preflight_cell_last[key][cell_idx])
                row[f"python_preflight_{key}_total"] = total
                row[f"python_preflight_{key}_mean"] = mean
                row[f"python_preflight_{key}_latest"] = latest
                native_key = f"native_reject_{key}_count"
                if native_key in _NATIVE_REJECT_INFO_KEYS:
                    row[f"native_minus_python_{key}_mean"] = (
                        float(native_reject_cell_sum[native_key][cell_idx]) / denom - mean
                    )
            row["native_slot_layout_tti_count"] = float(native_layout_seen_count)
            row["python_slot_count_mean"] = float(native_layout_python_slot_count_sum[cell_idx] / layout_denom)
            row["native_slot_count_mean"] = float(native_layout_native_slot_count_sum[cell_idx] / layout_denom)
            row["slot_count_abs_delta_mean"] = float(native_layout_slot_count_abs_delta_sum[cell_idx] / layout_denom)
            row["python_slot_count_latest"] = float(native_layout_python_slot_count_last[cell_idx])
            row["native_slot_count_latest"] = float(native_layout_native_slot_count_last[cell_idx])
            row["slot_count_abs_delta_latest"] = float(native_layout_slot_count_abs_delta_last[cell_idx])
            row["python_cell_start_mean"] = float(native_layout_python_cell_start_sum[cell_idx] / layout_denom)
            row["native_cell_start_mean"] = float(native_layout_native_cell_start_sum[cell_idx] / layout_denom)
            row["cell_start_abs_delta_mean"] = float(native_layout_cell_start_abs_delta_sum[cell_idx] / layout_denom)
            row["python_cell_start_latest"] = float(native_layout_python_cell_start_last[cell_idx])
            row["native_cell_start_latest"] = float(native_layout_native_cell_start_last[cell_idx])
            row["cell_start_abs_delta_latest"] = float(native_layout_cell_start_abs_delta_last[cell_idx])
            row["slot_to_cell_mismatch_mean"] = float(native_layout_slot_to_cell_mismatch_total / layout_denom)
            row["slot_to_cell_mismatch_latest"] = float(native_layout_slot_to_cell_mismatch_latest)
            row["all_reject_total"] = total_all
            row["all_reject_mean"] = mean_all
            row["all_reject_latest"] = latest_all
            native_reject_rows.append(row)
        _write_csv(native_reject_rows, native_reject_log_dir / f"iter_{episode_idx:04d}_native_reject.csv")
        _write_csv(native_reject_rows, native_reject_log_dir / "native_reject_latest.csv")
    return steps, summary, per_ue_tracker


def _ppo_update(
    *,
    actor: StageBHGraphPolicy,
    critic: GraphValueHead,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    rollout: Sequence[RolloutStep],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    update_epochs: int,
    minibatch_size: int,
    iter_idx: int,
    update_log_every_minibatches: int,
    target_kl: float,
    state_norm: RunningNorm | None = None,
    max_grad_norm: float = 5.0,
    value_loss_type: str = "mse",
    value_huber_beta: float = 10.0,
    teacher_aux_coef: float = 0.0,
) -> Dict[str, float]:
    actor.train(True)
    critic.train(True)

    old_logp = torch.stack([s.old_logp.to(torch.float32) for s in rollout], dim=0)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_teacher_aux_loss = 0.0
    update_count = 0
    n_steps = len(rollout)
    train_device = next(actor.parameters()).device
    old_logp = old_logp.to(device=train_device)
    advantages = advantages.to(device=train_device)
    returns = returns.to(device=train_device)

    update_t0 = time.perf_counter()
    minibatches_per_epoch = max(1, math.ceil(n_steps / max(1, int(minibatch_size))))
    min_updates_before_early_stop = max(4, n_steps // max(1, int(minibatch_size)))
    stopped_by_target_kl = 0.0
    target_kl_value = float(target_kl)
    stop_early = False
    completed_epochs = 0

    for epoch_idx in range(update_epochs):
        perm = torch.randperm(n_steps, device=train_device)
        mb_idx = 0
        for start in range(0, n_steps, minibatch_size):
            mb_idx += 1
            idx_tensor = perm[start : start + minibatch_size]
            idx = idx_tensor.tolist()
            adv_mb = advantages[idx_tensor]
            ret_mb = returns[idx_tensor]
            old_logp_mb = old_logp[idx_tensor]
            graphs_mb = [rollout[sample_idx].graph for sample_idx in idx]
            ue_action_mb = torch.stack([rollout[sample_idx].ue_action_class for sample_idx in idx], dim=0).to(
                device=train_device, dtype=torch.long
            )
            prg_action_mb = torch.stack([rollout[sample_idx].prg_action_class for sample_idx in idx], dim=0).to(
                device=train_device, dtype=torch.long
            )

            actor_out, value_pred = _evaluate_graph_batch(
                actor,
                critic,
                graphs_mb,
                state_norm=state_norm,
                slot_ue_class_for_prg=ue_action_mb,
            )
            new_logp, entropy_vec = _action_logp_entropy_batch(
                actor_out,
                graphs_mb,
                ue_action_mb,
                prg_action_mb,
            )
            entropy = entropy_vec.mean()

            ratio = torch.exp(new_logp - old_logp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            if value_loss_type == "huber":
                value_loss = F.smooth_l1_loss(value_pred, ret_mb, beta=value_huber_beta)
            else:
                value_loss = F.mse_loss(value_pred, ret_mb)
            teacher_aux_loss = torch.zeros((), dtype=torch.float32, device=train_device)
            if teacher_aux_coef > 0.0:
                prg_logits_masked, prg_valid = _masked_prg_logits_batch(
                    actor_out,
                    graphs_mb,
                    ue_action_class=ue_action_mb,
                )
                teacher_targets: List[torch.Tensor] = []
                for batch_idx, graph in enumerate(graphs_mb):
                    meta = graph.snapshot.static_meta
                    if meta.teacher_action_ue is None or meta.teacher_action_prg is None:
                        teacher_targets.append(
                            torch.full(
                                (meta.n_cell, meta.n_prg),
                                -100,
                                dtype=torch.long,
                                device=train_device,
                            )
                        )
                        continue
                    _teacher_ue, teacher_prg = joint_targets_from_type0_actions(
                        action_ue_select=meta.teacher_action_ue.to(device=train_device, dtype=torch.long),
                        action_prg_alloc=meta.teacher_action_prg.to(device=train_device, dtype=torch.long),
                        n_ue=meta.n_ue,
                        n_sched_ue=meta.n_sched_ue,
                        n_cell=meta.n_cell,
                        n_prg=meta.n_prg,
                    )
                    teacher_targets.append(teacher_prg.to(device=train_device, dtype=torch.long))
                teacher_target = torch.stack(teacher_targets, dim=0)
                teacher_target, _ = sanitize_targets(teacher_target, prg_valid, ignore_index=-100)
                if bool((teacher_target != -100).any().item()):
                    teacher_aux_loss = F.cross_entropy(
                        prg_logits_masked.reshape(-1, prg_logits_masked.shape[-1]),
                        teacher_target.reshape(-1),
                        ignore_index=-100,
                    )
            actor_loss = policy_loss - entropy_coef * entropy + teacher_aux_coef * teacher_aux_loss
            critic_loss = value_coef * value_loss

            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=max_grad_norm)
            actor_optimizer.step()

            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=max_grad_norm)
            critic_optimizer.step()

            approx_kl = (old_logp_mb - new_logp).mean()
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_approx_kl += float(approx_kl.item())
            total_teacher_aux_loss += float(teacher_aux_loss.item())
            update_count += 1

            if (
                target_kl_value > 0.0
                and update_count >= min_updates_before_early_stop
                and float(approx_kl.item()) > (1.5 * target_kl_value)
            ):
                stopped_by_target_kl = 1.0
                stop_early = True
                break
        completed_epochs = epoch_idx + 1
        if stop_early:
            break

    denom = max(update_count, 1)
    summary = {
        "policy_loss": total_policy_loss / denom,
        "value_loss": total_value_loss / denom,
        "entropy_mean": total_entropy / denom,
        "approx_kl_mean": total_approx_kl / denom,
        "teacher_aux_loss": total_teacher_aux_loss / denom,
        "teacher_aux_coef": float(teacher_aux_coef),
        "num_updates": float(update_count),
        "stopped_by_target_kl": float(stopped_by_target_kl),
        "epochs_completed": float(completed_epochs),
    }
    update_elapsed_s = time.perf_counter() - update_t0
    print(
        f"[update iter {iter_idx:04d}] "
        f"epochs={completed_epochs}/{update_epochs} "
        f"minibatches={update_count}/{max(1, update_epochs * minibatches_per_epoch)} "
        f"policy={summary['policy_loss']:.4f} "
        f"value={summary['value_loss']:.4f} "
        f"entropy={summary['entropy_mean']:.4f} "
        f"kl={summary['approx_kl_mean']:.4f} "
        f"teacher_aux={summary['teacher_aux_loss']:.4f}*{summary['teacher_aux_coef']:.3f} "
        f"time={update_elapsed_s:.1f}s "
        f"stop_kl={int(stopped_by_target_kl)}"
    )
    return summary


def _write_csv(rows: Sequence[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(
    *,
    path: Path,
    args: argparse.Namespace,
    device: torch.device,
    reward_cfg: Dict[str, float],
    loaded_compatible: int,
    loaded_total: int,
    metrics_csv: Path,
    actor_last: Path,
    actor_best: Path,
    actor_absolute_best: Path,
    actor_threshold_best: Path,
    train_last: Path,
    train_best: Path,
    train_absolute_best: Path,
    train_threshold_best: Path,
    imitation_actor_init: Path | None,
    absolute_best_iter: int,
    absolute_best_score: float,
    threshold_best_iter: int,
    threshold_best_score: float,
    history: Sequence[Dict[str, float]],
) -> None:
    summary = {
        "iterations": int(args.iterations),
        "rollout_steps": int(args.rollout_steps),
        "device": str(device),
        "reward_mode": str(args.reward_mode),
        "reward_cfg": reward_cfg,
        "init_checkpoint": str(args.init_checkpoint),
        "ue_prg_topk": int(args.ue_prg_topk),
        "ue_prg_diversity_extra": int(args.ue_prg_diversity_extra),
        "candidate_mode": str(args.candidate_mode),
        "max_prg_candidates": int(args.max_prg_candidates),
        "slot_repair_max_per_cell": int(args.slot_repair_max_per_cell),
        "slot_repair_min_backlog_bytes": float(args.slot_repair_min_backlog_bytes),
        "imitation_episodes": int(args.imitation_episodes),
        "teacher_mode": str(args.teacher_mode),
        "teacher_aux_coef": float(args.teacher_aux_coef),
        "teacher_aux_decay_iters": int(args.teacher_aux_decay_iters),
        "normalize_state": int(args.normalize_state),
        "normalize_reward": int(args.normalize_reward),
        "normalize_adv": int(args.normalize_adv),
        "value_loss": str(args.value_loss),
        "loaded_compatible_params": int(loaded_compatible),
        "loaded_total_params": int(loaded_total),
        "metrics_csv": str(metrics_csv),
        "actor_last": str(actor_last),
        "actor_best": str(actor_best),
        "actor_absolute_best": str(actor_absolute_best),
        "actor_threshold_best": str(actor_threshold_best),
        "actor_imitation_init": str(imitation_actor_init) if imitation_actor_init is not None else "",
        "train_last": str(train_last),
        "train_best": str(train_best),
        "train_absolute_best": str(train_absolute_best),
        "train_threshold_best": str(train_threshold_best),
        "best_metric": str(args.best_metric),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "absolute_best_iter": int(absolute_best_iter),
        "absolute_best_score": float(absolute_best_score),
        "threshold_best_iter": int(threshold_best_iter),
        "threshold_best_score": float(threshold_best_score),
        "best_iter": int(threshold_best_iter),
        "best_score": float(threshold_best_score),
        "completed_iterations": len(history),
        "history_tail": list(history[-5:]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage-B sparse-graph online PPO fine-tuning")
    p.add_argument("--sim-bin", required=True, help="Path to multiCellSchedulerUeSelection binary")
    p.add_argument("--sim-args", default="-d 1 -b 2 -x 0 -f 0 -g 100 -r 3000", help="CLI args for the simulation binary")
    p.add_argument("--sim-cwd", default="", help="Optional cwd for launching the simulation")
    p.add_argument("--sim-env", action="append", default=[], help="Extra simulator env vars, format KEY=VALUE")
    p.add_argument("--sim-log-mode", choices=["file", "inherit"], default="file")
    p.add_argument("--socket-path", default="/tmp/cumac_stageb_hgraph_ppo.sock")
    p.add_argument("--connect-timeout-s", type=float, default=20.0)
    p.add_argument("--sim-wait-timeout", type=float, default=10.0)
    p.add_argument("--online-persistent", type=int, choices=[0, 1], default=1)
    p.add_argument("--episode-horizon", type=int, default=400)
    p.add_argument("--rollout-steps", type=int, default=400)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--topology-seed", type=int, default=42)
    p.add_argument("--topology-seed-mode", choices=["auto", "fixed", "sequential", "list_cycle"], default="auto")
    p.add_argument("--seed-list", default="", help="Comma-separated topology seeds for list_cycle mode")
    p.add_argument("--out-dir", default="training/stageb_hgraph/runs/online_ppo")
    p.add_argument("--ue-prg-topk", type=int, default=4)
    p.add_argument("--ue-prg-diversity-extra", type=int, default=0)
    p.add_argument("--candidate-mode", choices=["all_demand_ues", "topk_diversity"], default="all_demand_ues")
    p.add_argument("--max-prg-candidates", type=int, default=0, help="0 means fallback to topk+diversity width")
    p.add_argument(
        "--slot-repair-max-per-cell",
        type=int,
        default=0,
        help="0 disables sampled-UE coverage repair; positive values cap repair replacements per cell",
    )
    p.add_argument(
        "--slot-repair-min-backlog-bytes",
        type=float,
        default=1.0,
        help="minimum backlog for UEs inserted by slot repair",
    )
    p.add_argument("--tb-err-ema-alpha", type=float, default=0.10)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--message-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--actor-lr", type=float, default=0.0, help="0 means fallback to --lr")
    p.add_argument("--critic-lr", type=float, default=0.0, help="0 means fallback to max(--lr, 3e-4)")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--target-kl", type=float, default=0.05, help="0 disables PPO update early-stop on KL")
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--normalize-state", type=int, choices=[0, 1], default=1)
    p.add_argument("--normalize-reward", type=int, choices=[0, 1], default=1)
    p.add_argument("--normalize-adv", type=int, choices=[0, 1], default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--value-loss", choices=["mse", "huber"], default="huber")
    p.add_argument("--value-huber-beta", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--print-every-iters", type=int, default=1)
    p.add_argument("--rollout-log-every-steps", type=int, default=500)
    p.add_argument("--update-log-every-minibatches", type=int, default=2)
    p.add_argument("--init-checkpoint", default="", help="Optional imitation/actor checkpoint to warm-start from")
    p.add_argument("--imitation-episodes", type=int, default=0, help="Optional online imitation warm-start episodes before PPO")
    p.add_argument("--imitation-max-steps", type=int, default=0, help="0 means use episode horizon during imitation warm-start")
    p.add_argument("--imitation-lr", type=float, default=0.0, help="0 means fallback to actor lr during imitation warm-start")
    p.add_argument("--teacher-mode", choices=["heuristic_graph", "pfq_passthrough"], default="pfq_passthrough")
    p.add_argument("--teacher-aux-coef", type=float, default=0.20)
    p.add_argument("--teacher-aux-decay-iters", type=int, default=80)
    p.add_argument("--reward-mode", choices=["raw_env", "stageb_v1", "stageb_blankaware"], default="stageb_v1")
    p.add_argument("--best-metric", choices=["reward", "goodput"], default="goodput")
    p.add_argument("--early-stop-patience", type=int, default=60, help="0 disables early stop")
    p.add_argument("--early-stop-min-delta", type=float, default=1.0)
    p.add_argument("--early-stop-warmup-iters", type=int, default=50)
    p.add_argument("--goodput-weight", type=float, default=1.0)
    p.add_argument("--tb-err-weight", type=float, default=8.0)
    p.add_argument("--expiry-weight", type=float, default=6.0)
    p.add_argument("--conflict-weight", type=float, default=1.0)
    p.add_argument("--ici-weight", type=float, default=1.0)
    p.add_argument("--urgent-backlog-weight", type=float, default=0.5)
    p.add_argument("--urgent-ttl-ms", type=float, default=20.0)
    p.add_argument(
        "--backlog-debt-weight",
        type=float,
        default=0.0,
        help="coefficient for actual-unserved backlog debt in stageb_blankaware; 0 disables it",
    )
    p.add_argument(
        "--packet-rate-weight",
        type=float,
        default=0.0,
        help="coefficient for per-TTI packet effective service rate bonus in stageb_blankaware; 0 disables it",
    )
    p.add_argument(
        "--packet-completion-weight",
        type=float,
        default=0.0,
        help="coefficient for per-TTI packet completion-count bonus in stageb_blankaware; 0 disables it",
    )
    p.add_argument(
        "--packet-rate-scale-mbps",
        type=float,
        default=10.0,
        help="log scale for packet-rate reward bonus",
    )
    p.add_argument(
        "--packet-completion-scale",
        type=float,
        default=36.0,
        help="log scale for packet-completion reward bonus",
    )
    p.add_argument(
        "--aged-backlog-weight",
        type=float,
        default=0.0,
        help="coefficient for recent100 aged-backlog debt penalty in stageb_blankaware; 0 disables it",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.sim_cwd == "":
        args.sim_cwd = None
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be >= 1")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.imitation_episodes < 0:
        raise ValueError("--imitation-episodes must be >= 0")
    if args.imitation_max_steps < 0:
        raise ValueError("--imitation-max-steps must be >= 0")
    if args.ue_prg_diversity_extra < 0:
        raise ValueError("--ue-prg-diversity-extra must be >= 0")
    if args.max_prg_candidates < 0:
        raise ValueError("--max-prg-candidates must be >= 0")
    if args.teacher_aux_decay_iters < 0:
        raise ValueError("--teacher-aux-decay-iters must be >= 0")

    _set_seed(int(args.seed))
    device = _pick_device(args.device)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "online_ppo_metrics.csv"
    summary_json = out_dir / "online_ppo_summary.json"
    per_ue_log_dir = out_dir / "per_ue_diagnostics"
    native_reject_log_dir = out_dir / "native_reject_diagnostics"
    actor_last = out_dir / "ppo_actor_last.pt"
    actor_best = out_dir / "ppo_actor_best.pt"
    actor_absolute_best = out_dir / "ppo_actor_absolute_best.pt"
    actor_threshold_best = out_dir / "ppo_actor_threshold_best.pt"
    train_last = out_dir / "ppo_train_last.pt"
    train_best = out_dir / "ppo_train_best.pt"
    train_absolute_best = out_dir / "ppo_train_absolute_best.pt"
    train_threshold_best = out_dir / "ppo_train_threshold_best.pt"
    imitation_actor_init = out_dir / "ppo_actor_imitation_init.pt"

    snapshot_builder = FeatureSnapshotBuilder(
        SnapshotBuilderConfig(tb_err_ema_alpha=float(args.tb_err_ema_alpha))
    )
    max_prg_candidates = _resolve_prg_candidate_count(args)
    edge_generator = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            ue_prg_topk=int(args.ue_prg_topk),
            ue_prg_diversity_extra=int(args.ue_prg_diversity_extra),
            candidate_mode=str(args.candidate_mode),
        )
    )
    actor = StageBHGraphPolicy(
        HGraphModelConfig(
            hidden_dim=int(args.hidden_dim),
            message_layers=int(args.message_layers),
            dropout=float(args.dropout),
            max_prg_candidates=max_prg_candidates,
        )
    ).to(device)
    critic = GraphValueHead(hidden_dim=int(args.hidden_dim)).to(device)
    actor_lr = float(args.actor_lr) if float(args.actor_lr) > 0.0 else float(args.lr)
    critic_lr = float(args.critic_lr) if float(args.critic_lr) > 0.0 else max(float(args.lr), 3.0e-4)

    loaded_compatible = 0
    loaded_total = 0
    if args.init_checkpoint:
        loaded_compatible, loaded_total = _load_checkpoint_compatible(actor, args.init_checkpoint)
    warm_start_metrics = None
    if int(args.imitation_episodes) > 0:
        warm_start_metrics = _run_online_imitation_warm_start(
            args=args,
            actor=actor,
            snapshot_builder=snapshot_builder,
            edge_generator=edge_generator,
            device=device,
        )
        torch.save(
            {
                "actor_state": actor.state_dict(),
                "actor_config": actor.model_config_dict(),
                "args": vars(args),
                "teacher_mode": str(args.teacher_mode),
                "warm_start_metrics": warm_start_metrics,
                "source": "online_ppo_imitation_warm_start",
            },
            imitation_actor_init,
        )
        with open(out_dir / "imitation_warm_start_summary.json", "w", encoding="utf-8") as f:
            json.dump(warm_start_metrics, f, ensure_ascii=True, indent=2)

    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=actor_lr,
        weight_decay=float(args.weight_decay),
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=critic_lr,
        weight_decay=float(args.weight_decay),
    )
    state_norm = RunningNorm(dim=int(args.hidden_dim) * 3) if int(args.normalize_state) == 1 else None
    reward_norm = RunningScalarNorm() if int(args.normalize_reward) == 1 else None

    reward_cfg = {
        "goodput_weight": float(args.goodput_weight),
        "tb_err_weight": float(args.tb_err_weight),
        "expiry_weight": float(args.expiry_weight),
        "conflict_weight": float(args.conflict_weight),
        "ici_weight": float(args.ici_weight),
        "urgent_backlog_weight": float(args.urgent_backlog_weight),
        "urgent_ttl_ms": float(args.urgent_ttl_ms),
        "backlog_debt_weight": float(args.backlog_debt_weight),
        "packet_rate_weight": float(args.packet_rate_weight),
        "packet_completion_weight": float(args.packet_completion_weight),
        "packet_rate_scale_mbps": float(args.packet_rate_scale_mbps),
        "packet_completion_scale": float(args.packet_completion_scale),
        "aged_backlog_weight": float(args.aged_backlog_weight),
    }

    runner = OnlineEpisodeRunner(args)
    per_ue_tracker: PerUeDiagnosticsTracker | None = None
    history: List[Dict[str, float]] = []
    absolute_best_score = -1.0e30
    absolute_best_iter = 0
    threshold_best_score = -1.0e30
    threshold_best_iter = 0
    no_improve_iters = 0

    try:
        for iter_idx in range(1, int(args.iterations) + 1):
            iter_t0 = time.perf_counter()
            collect_t0 = time.perf_counter()
            rollout, rollout_summary, per_ue_tracker = _collect_rollout(
                runner=runner,
                snapshot_builder=snapshot_builder,
                edge_generator=edge_generator,
                actor=actor,
                critic=critic,
                device=device,
                rollout_steps=int(args.rollout_steps),
                reward_mode=str(args.reward_mode),
                reward_cfg=reward_cfg,
                episode_idx=iter_idx,
                rollout_log_every_steps=int(args.rollout_log_every_steps),
                state_norm=state_norm,
                reward_norm=reward_norm,
                normalize_reward=int(args.normalize_reward) == 1,
                per_ue_log_dir=per_ue_log_dir,
                native_reject_log_dir=native_reject_log_dir,
                per_ue_tracker=per_ue_tracker,
                slot_repair_max_per_cell=int(args.slot_repair_max_per_cell),
                slot_repair_min_backlog_bytes=float(args.slot_repair_min_backlog_bytes),
            )
            collect_time_s = time.perf_counter() - collect_t0

            rewards = torch.tensor([s.reward for s in rollout], dtype=torch.float32)
            values = torch.tensor([s.old_value for s in rollout], dtype=torch.float32)
            next_values = torch.tensor([s.next_value for s in rollout], dtype=torch.float32)
            dones = torch.tensor([s.done for s in rollout], dtype=torch.float32)
            advantages, returns = _compute_gae(
                rewards=rewards,
                values=values,
                next_values=next_values,
                dones=dones,
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
            )
            if int(args.normalize_adv) == 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

            update_t0 = time.perf_counter()
            update_metrics = _ppo_update(
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                rollout=rollout,
                advantages=advantages,
                returns=returns,
                clip_eps=float(args.clip_eps),
                value_coef=float(args.value_coef),
                entropy_coef=float(args.entropy_coef),
                update_epochs=int(args.update_epochs),
                minibatch_size=int(args.minibatch_size),
                iter_idx=iter_idx,
                update_log_every_minibatches=int(args.update_log_every_minibatches),
                target_kl=float(args.target_kl),
                state_norm=state_norm,
                max_grad_norm=float(args.max_grad_norm),
                value_loss_type=str(args.value_loss),
                value_huber_beta=float(args.value_huber_beta),
                teacher_aux_coef=_resolve_teacher_aux_coef(args, iter_idx),
            )
            update_time_s = time.perf_counter() - update_t0
            iter_time_s = time.perf_counter() - iter_t0

            row = {
                "iter": float(iter_idx),
                **rollout_summary,
                **update_metrics,
                "collect_time_s": float(collect_time_s),
                "update_time_s": float(update_time_s),
                "iter_time_s": float(iter_time_s),
                "adv_mean": float(advantages.mean().item()),
                "adv_std": float(advantages.std(unbiased=False).item()),
                "return_mean": float(returns.mean().item()),
            }
            history.append(row)

            actor_checkpoint = {
                "iter": iter_idx,
                "actor_state": actor.state_dict(),
                "actor_config": actor.model_config_dict(),
                "reward_mode": args.reward_mode,
                "reward_cfg": reward_cfg,
                "args": vars(args),
                "metrics": row,
                "warm_start_metrics": warm_start_metrics,
            }
            train_checkpoint = {
                **actor_checkpoint,
                "critic_state": critic.state_dict(),
                "actor_optimizer_state": actor_optimizer.state_dict(),
                "critic_optimizer_state": critic_optimizer.state_dict(),
                "state_norm": state_norm.state_dict() if state_norm is not None else {},
                "reward_norm": reward_norm.state_dict() if reward_norm is not None else {},
            }
            torch.save(actor_checkpoint, actor_last)
            torch.save(train_checkpoint, train_last)

            if str(args.best_metric) == "goodput":
                score = float(row["rollout_goodput_mbps_mean"])
            else:
                score = float(row["rollout_reward_mean"])

            if score > absolute_best_score:
                absolute_best_score = score
                absolute_best_iter = iter_idx
                torch.save(actor_checkpoint, actor_absolute_best)
                torch.save(train_checkpoint, train_absolute_best)

            threshold_improved = score > (threshold_best_score + float(args.early_stop_min_delta))
            if threshold_improved:
                threshold_best_score = score
                threshold_best_iter = iter_idx
                torch.save(actor_checkpoint, actor_threshold_best)
                torch.save(train_checkpoint, train_threshold_best)
                torch.save(actor_checkpoint, actor_best)
                torch.save(train_checkpoint, train_best)
                no_improve_iters = 0
            else:
                no_improve_iters += 1

            _write_csv(history, metrics_csv)
            _write_summary(
                path=summary_json,
                args=args,
                device=device,
                reward_cfg=reward_cfg,
                loaded_compatible=loaded_compatible,
                loaded_total=loaded_total,
                metrics_csv=metrics_csv,
                actor_last=actor_last,
                actor_best=actor_best,
                actor_absolute_best=actor_absolute_best,
                actor_threshold_best=actor_threshold_best,
                imitation_actor_init=imitation_actor_init if warm_start_metrics is not None else None,
                train_last=train_last,
                train_best=train_best,
                train_absolute_best=train_absolute_best,
                train_threshold_best=train_threshold_best,
                absolute_best_iter=absolute_best_iter,
                absolute_best_score=absolute_best_score,
                threshold_best_iter=threshold_best_iter,
                threshold_best_score=threshold_best_score,
                history=history,
            )

            if int(args.print_every_iters) > 0 and ((iter_idx % int(args.print_every_iters)) == 0):
                print(
                    f"[ppo iter {iter_idx:04d}] reward={row['rollout_reward_mean']:.4f} "
                    f"goodput={row['rollout_goodput_mbps_mean']:.4f} "
                    f"tb_err={row['rollout_tb_err_rate_mean']:.4f} "
                    f"blank={row['rollout_blank_ratio_mean']:.4f} "
                    f"drop={row['rollout_post_decode_drop_ratio_mean']:.4f} "
                    f"fill={row['rollout_action_fill_ratio_mean']:.4f} "
                    f"capDrop={row['rollout_cap_drop_count_mean']:.2f} "
                    f"fbOK={row['rollout_fallback_success_count_mean']:.2f} "
                    f"blankCnt={row['rollout_final_blank_count_mean']:.2f} "
                    f"uniq_ue={row['rollout_action_unique_ue_mean']:.2f} "
                    f"cand_ue={row['rollout_candidate_unique_ue_mean']:.2f} "
                    f"demand_ue={row['rollout_demand_active_ue_mean']:.2f} "
                    f"cand_cov={row['rollout_candidate_coverage_ratio_mean']:.3f} "
                    f"unsrv_ue={row['rollout_unserved_demand_ue_mean']:.2f} "
                    f"streak90={row['rollout_no_service_streak_p90_mean']:.1f} "
                    f"streakMax={row['rollout_no_service_streak_max_mean']:.1f} "
                    f"srv100_p10={row['rollout_recent100_service_rate_p10_mean']:.3f} "
                    f"debt100_p90={row['rollout_recent100_debt_score_p90_bytes_mean']/1024.0:.1f}KB "
                    f"bklg_p90={row['rollout_backlog_p90_bytes_mean']/1024.0:.1f}KB "
                    f"preDiag={row['rollout_accepted_pre_pdsch_ue_diagnostics_available_mean']:.0f} "
                    f"preGap={row['rollout_planned_accepted_pre_pdsch_prg_gap_ratio_mean']:.3f} "
                    f"postGap={row['rollout_accepted_pre_pdsch_actual_prg_gap_ratio_mean']:.3f} "
                    f"gap={row['rollout_planned_native_prg_gap_ratio_mean']:.3f} "
                    f"pyWrong={row['rollout_python_preflight_wrong_cell_slot_count_mean']:.1f} "
                    f"rejWrong={row['rollout_native_reject_wrong_cell_slot_count_mean']:.1f} "
                    f"slotCntΔ={row['rollout_native_slot_count_abs_delta_sum_mean']:.1f} "
                    f"slotMapΔ={row['rollout_native_slot_to_cell_mismatch_count_mean']:.1f} "
                    f"rejEmpty={row['rollout_native_reject_empty_slot_count_mean']:.1f} "
                    f"rejDup={row['rollout_native_reject_duplicate_ue_count_mean']:.1f} "
                    f"posteqL={row['rollout_post_eq_layer_dim_mean']:.0f} "
                    f"time={row['iter_time_s']:.1f}s "
                    f"(collect={row['collect_time_s']:.1f}s update={row['update_time_s']:.1f}s) "
                    f"policy={row['policy_loss']:.4f} value={row['value_loss']:.4f} "
                    f"entropy={row['entropy_mean']:.4f} kl={row['approx_kl_mean']:.4f} "
                    f"teacher_aux={row['teacher_aux_loss']:.4f}*{row['teacher_aux_coef']:.3f}"
                )

            if (
                int(args.early_stop_patience) > 0
                and iter_idx >= int(args.early_stop_warmup_iters)
                and no_improve_iters >= int(args.early_stop_patience)
            ):
                print(
                    f"[online-ppo] early_stop iter={iter_idx:04d} "
                    f"threshold_best_iter={threshold_best_iter:04d} threshold_best_{args.best_metric}={threshold_best_score:.4f} "
                    f"absolute_best_iter={absolute_best_iter:04d} absolute_best_{args.best_metric}={absolute_best_score:.4f} "
                    f"patience={int(args.early_stop_patience)} min_delta={float(args.early_stop_min_delta):.4f}"
                )
                break
    finally:
        runner.close_episode(graceful=False)

    _write_csv(history, metrics_csv)
    _write_summary(
        path=summary_json,
        args=args,
        device=device,
        reward_cfg=reward_cfg,
        loaded_compatible=loaded_compatible,
        loaded_total=loaded_total,
        metrics_csv=metrics_csv,
        actor_last=actor_last,
        actor_best=actor_best,
        actor_absolute_best=actor_absolute_best,
        actor_threshold_best=actor_threshold_best,
        imitation_actor_init=imitation_actor_init if warm_start_metrics is not None else None,
        train_last=train_last,
        train_best=train_best,
        train_absolute_best=train_absolute_best,
        train_threshold_best=train_threshold_best,
        absolute_best_iter=absolute_best_iter,
        absolute_best_score=absolute_best_score,
        threshold_best_iter=threshold_best_iter,
        threshold_best_score=threshold_best_score,
        history=history,
    )

    print(f"[online-ppo] wrote metrics: {metrics_csv}")
    print(f"[online-ppo] wrote summary: {summary_json}")
    print(f"[online-ppo] last checkpoint: {actor_last}")
    print(f"[online-ppo] best checkpoint (threshold): {actor_best}")
    print(f"[online-ppo] threshold best checkpoint: {actor_threshold_best}")
    print(f"[online-ppo] absolute best checkpoint: {actor_absolute_best}")
    print(f"[online-ppo] threshold best train checkpoint: {train_threshold_best}")
    print(f"[online-ppo] absolute best train checkpoint: {train_absolute_best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
