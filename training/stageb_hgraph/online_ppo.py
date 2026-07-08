#!/usr/bin/env python3
"""Online PPO fine-tuning for the Stage-B sparse entity graph policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


def _configure_torch_threads_from_env() -> None:
    num_threads_raw = os.environ.get("CUMAC_TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    interop_raw = os.environ.get("CUMAC_TORCH_NUM_INTEROP_THREADS")
    if num_threads_raw:
        try:
            torch.set_num_threads(max(1, int(num_threads_raw)))
        except (RuntimeError, ValueError):
            pass
    if interop_raw:
        try:
            torch.set_num_interop_threads(max(1, int(interop_raw)))
        except (RuntimeError, ValueError):
            pass


_configure_torch_threads_from_env()

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
from training.stageb_hgraph.edge_generator import (
    CANDIDATE_RISK_FEATURE_DIM,
    EdgeGeneratorConfig,
    SparseEdgeGenerator,
)
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from training.stageb_hgraph.graph_ops import (
    _candidate_packet_completion_scores,
    _estimate_ue_prg_demand_cap,
    _candidate_seq_fail_risk_features,
    _candidate_seq_state_delta,
    apply_candidate_blank_budget,
    build_prg_candidate_valid_mask,
    candidate_prg_decode_goodput_targets,
    candidate_prg_pairwise_rank_targets,
    candidate_prg_packet_tail_quota_scores,
    candidate_prg_rescue_priority_scores,
    candidate_prg_service_protection_mask,
    candidate_prg_success_targets,
    candidate_prg_soft_targets_from_type0_actions,
    candidate_prg_tail_bias_scores,
    decode_candidate_actions_to_type0,
    decode_joint_actions_to_type0,
    demand_active_ue_mask_from_features,
    joint_candidate_targets_from_type0_actions,
    joint_targets_from_type0_actions,
    pack_prg_candidates,
    select_budgeted_candidate_actions,
    select_and_decode_budgeted_candidate_actions,
    type0_action_preflight_reject_counts,
)
from training.stageb_hgraph.model import HGraphModelConfig, StageBHGraphPolicy
from training.stageb_hgraph.online_imitation import OnlineEpisodeRunner, _pick_device, _set_seed
from training.stageb_hgraph.reward_utils import (
    compute_stageb_blankaware_reward,
    compute_stageb_tailcredit_expiry_reward,
    compute_stageb_tailcredit_reward,
    compute_stageb_tailguard_simple_reward,
    compute_stageb_v1_reward,
    extract_reward_metrics,
)
from training.stageb_hgraph.teacher import HeuristicTeacherConfig, build_joint_teacher
from training.stageb_hgraph.types import NamedFeatureBlock, SparseEntityGraph


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
    safe_fill_count: float
    missed_safe_opportunity_count: float
    blank_budget_repair_count: float
    final_blank_count: float
    post_decode_policy_blank_count: float
    post_decode_no_valid_candidate_count: float
    post_decode_demand_cap_count: float
    post_decode_no_slot_count: float
    post_decode_other_drop_count: float
    action_fill_ratio: float
    action_unique_ue_count: float
    candidate_unique_ue_count: float
    demand_active_ue_count: float
    candidate_coverage_ratio: float
    candidate_budgeted_changed_count: float
    candidate_budgeted_no_headroom_count: float
    candidate_budgeted_active_ue_count: float
    candidate_budgeted_tail_bias_coef: float
    candidate_budgeted_tail_bias_candidate_mean: float
    candidate_budgeted_tail_bias_selected_mean: float
    candidate_budgeted_tail_bias_active_count: float
    candidate_budgeted_rescue_bias_coef: float
    candidate_budgeted_rescue_bias_candidate_mean: float
    candidate_budgeted_rescue_bias_selected_mean: float
    candidate_budgeted_rescue_bias_active_count: float
    candidate_budgeted_completion_bonus: float
    candidate_budgeted_completion_candidate_mean: float
    candidate_budgeted_completion_selected_mean: float
    candidate_budgeted_completion_active_count: float
    candidate_budgeted_tail_quota_enabled: float
    candidate_budgeted_tail_quota_candidate_mean: float
    candidate_budgeted_tail_quota_active_ue_count: float
    candidate_budgeted_tail_quota_selected_count: float
    candidate_budgeted_tail_quota_selected_mean: float
    candidate_budgeted_tail_quota_served_ue_count: float
    unserved_demand_ue_count: float
    planned_unserved_demand_ue_count: float
    actual_unserved_backlog_bytes: float
    accepted_pre_pdsch_diagnostics_available: float
    actual_diagnostics_available: float
    actual_packet_diagnostics_available: float
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
    ue_macro_rate_proxy_mbps: float
    ue_p10_rate_proxy_mbps: float
    ue_min_rate_proxy_mbps: float
    ue_rate_proxy_active_count: float
    ue_macro_packet_rate_mbps: float
    ue_p10_packet_rate_mbps: float
    ue_min_packet_rate_mbps: float
    ue_packet_rate_active_count: float
    ue_packet_delivered_active_count: float
    ue_macro_packet_delay_ms: float
    ue_p90_packet_delay_ms: float
    recent100_unserved_backlog_p90_bytes: float
    recent100_debt_score_p90_bytes: float
    recent100_debt_score_max_bytes: float
    backlog_p90_bytes: float
    backlog_max_bytes: float
    goodput_mbps: float
    tb_err_rate: float
    expiry_drop_rate: float
    expiry_reward_rate: float
    prg_utilization_ratio: float
    prg_reuse_ratio: float
    packet_completed_count: float
    packet_system_time_ms: float
    packet_delay_mean_ms: float
    packet_effective_service_rate_mbps: float
    packet_effective_service_rate_per_packet_mean_mbps: float
    conflict_mean: float
    ici_mean: float
    prg_expected_goodput_proxy_mbps_mean: float
    prg_low_sinr_hol_ttl_weight_mean: float
    urgent_backlog_penalty: float
    backlog_debt_coef: float
    packet_rate_bonus: float
    packet_per_packet_rate_bonus: float
    packet_per_packet_rate_coef: float
    packet_completion_bonus: float
    packet_delay_penalty: float
    packet_delay_coef: float
    ue_macro_rate_proxy_bonus: float
    ue_macro_rate_proxy_coef: float
    ue_p10_rate_proxy_gap_penalty: float
    ue_p10_rate_proxy_gap_coef: float
    ue_macro_packet_rate_bonus: float
    ue_macro_packet_rate_coef: float
    ue_p10_packet_rate_gap_penalty: float
    ue_p10_packet_rate_gap_coef: float
    aged_backlog_penalty: float
    aged_backlog_coef: float
    active_prg_goodput_efficiency: float
    harmful_packing_penalty: float
    util_floor_gate: float
    util_floor_gap: float
    util_floor_penalty: float
    util_floor_coef: float
    missed_safe_opportunity_penalty: float
    missed_safe_opportunity_coef: float
    tb_err_gap: float
    tb_err_over_target_coef: float
    tb_err_over_target_penalty: float
    tb_err_goodput_gate_coef: float
    tb_err_goodput_gate_penalty: float
    tail_potential_before: float
    tail_potential_after: float
    tail_potential_delta: float
    tail_potential_topk_mean: float
    tail_potential_max: float
    expiry_potential_before: float
    expiry_potential_after: float
    expiry_potential_delta: float
    expiry_potential_topk_mean: float
    expiry_potential_max: float
    rescue_credit: float
    rescue_miss: float
    rescue_risk_topk_mean: float
    rescue_served_ratio: float
    coverage_repair_count: float
    candidate_decode_goodput_target: torch.Tensor | None = None
    candidate_decode_goodput_mask: torch.Tensor | None = None


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


class ExpiryRewardWindowTracker:
    def __init__(self, *, window_steps: int = 0, offered_bytes_per_tti: float = 0.0):
        self.window_steps = max(0, int(window_steps))
        self.offered_bytes_per_tti = max(0.0, float(offered_bytes_per_tti))
        maxlen = max(1, self.window_steps)
        self.expired_bytes = deque(maxlen=maxlen)
        self.delivered_bytes = deque(maxlen=maxlen)

    @property
    def enabled(self) -> bool:
        return self.window_steps > 0

    def update(self, reward_metrics: Dict[str, float]) -> float:
        raw_rate = max(0.0, min(1.0, float(reward_metrics.get("expiry_drop_rate", 0.0))))
        if not self.enabled:
            return raw_rate

        expired_bytes = max(0.0, float(reward_metrics.get("expired_bytes", 0.0)))
        delivered_bytes = max(0.0, float(reward_metrics.get("packet_delivered_bits", 0.0))) / 8.0
        self.expired_bytes.append(expired_bytes)
        self.delivered_bytes.append(delivered_bytes)

        expired_sum = float(sum(self.expired_bytes))
        if self.offered_bytes_per_tti > 0.0:
            denom = self.offered_bytes_per_tti * float(len(self.expired_bytes))
        else:
            denom = expired_sum + float(sum(self.delivered_bytes))
        if denom <= 1.0e-9:
            return 0.0
        return max(0.0, min(1.0, expired_sum / denom))


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
        self.packet_delivered_packets_hist = np.zeros((self.window, self.n_ue), dtype=np.float32)
        self.packet_delivered_bits_hist = np.zeros((self.window, self.n_ue), dtype=np.float32)
        self.packet_system_time_ms_hist = np.zeros((self.window, self.n_ue), dtype=np.float32)
        self.demand_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.served_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.goodput_served_count = np.zeros((self.n_ue,), dtype=np.int32)
        self.demand_backlog_sum = np.zeros((self.n_ue,), dtype=np.float32)
        self.unserved_backlog_sum = np.zeros((self.n_ue,), dtype=np.float32)
        self.effective_prg_sum = np.zeros((self.n_ue,), dtype=np.float32)
        self.goodput_bytes_sum = np.zeros((self.n_ue,), dtype=np.float32)
        self.packet_delivered_packets_sum = np.zeros((self.n_ue,), dtype=np.float32)
        self.packet_delivered_bits_sum = np.zeros((self.n_ue,), dtype=np.float32)
        self.packet_system_time_ms_sum = np.zeros((self.n_ue,), dtype=np.float32)
        self.no_service_streak = np.zeros((self.n_ue,), dtype=np.int32)
        self.no_goodput_streak = np.zeros((self.n_ue,), dtype=np.int32)
        self.have_prev_actual_packet = False
        self.prev_actual_packet_delivered_packets = np.zeros((self.n_ue,), dtype=np.float32)
        self.prev_actual_packet_delivered_bits = np.zeros((self.n_ue,), dtype=np.float32)
        self.prev_actual_packet_system_time_ms = np.zeros((self.n_ue,), dtype=np.float32)
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
        self.last_actual_packet_available = False
        self.last_actual_packet_delivered_packets = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_actual_packet_pending_packets = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_actual_packet_delivered_bits = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_actual_packet_system_time_ms = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_actual_packet_service_rate_mbps = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_actual_packet_service_rate_per_packet_mean_mbps = np.zeros(
            (self.n_ue,),
            dtype=np.float32,
        )
        self.last_actual_packet_delta_delivered_packets = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_actual_packet_delta_delivered_bits = np.zeros((self.n_ue,), dtype=np.float32)
        self.last_actual_packet_delta_system_time_ms = np.zeros((self.n_ue,), dtype=np.float32)

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
        actual_packet_available: bool = False,
        actual_packet_delivered_packets: np.ndarray | None = None,
        actual_packet_pending_packets: np.ndarray | None = None,
        actual_packet_delivered_bits: np.ndarray | None = None,
        actual_packet_system_time_ms: np.ndarray | None = None,
        actual_packet_service_rate_mbps: np.ndarray | None = None,
        actual_packet_service_rate_per_packet_mean_mbps: np.ndarray | None = None,
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

        actual_packet_ok = bool(
            actual_packet_available
            and actual_packet_delivered_packets is not None
            and actual_packet_pending_packets is not None
            and actual_packet_delivered_bits is not None
            and actual_packet_system_time_ms is not None
            and actual_packet_service_rate_mbps is not None
            and actual_packet_service_rate_per_packet_mean_mbps is not None
        )
        if actual_packet_ok:
            actual_packet_delivered_packets = np.asarray(
                actual_packet_delivered_packets,
                dtype=np.float32,
            ).reshape(self.n_ue)
            actual_packet_pending_packets = np.asarray(
                actual_packet_pending_packets,
                dtype=np.float32,
            ).reshape(self.n_ue)
            actual_packet_delivered_bits = np.asarray(
                actual_packet_delivered_bits,
                dtype=np.float32,
            ).reshape(self.n_ue)
            actual_packet_system_time_ms = np.asarray(
                actual_packet_system_time_ms,
                dtype=np.float32,
            ).reshape(self.n_ue)
            actual_packet_service_rate_mbps = np.asarray(
                actual_packet_service_rate_mbps,
                dtype=np.float32,
            ).reshape(self.n_ue)
            actual_packet_service_rate_per_packet_mean_mbps = np.asarray(
                actual_packet_service_rate_per_packet_mean_mbps,
                dtype=np.float32,
            ).reshape(self.n_ue)
            base_packets = self.prev_actual_packet_delivered_packets if self.have_prev_actual_packet else 0.0
            base_bits = self.prev_actual_packet_delivered_bits if self.have_prev_actual_packet else 0.0
            base_time = self.prev_actual_packet_system_time_ms if self.have_prev_actual_packet else 0.0
            packet_delta_delivered_packets = np.maximum(
                actual_packet_delivered_packets - base_packets,
                0.0,
            ).astype(np.float32, copy=False)
            packet_delta_delivered_bits = np.maximum(
                actual_packet_delivered_bits - base_bits,
                0.0,
            ).astype(np.float32, copy=False)
            packet_delta_system_time_ms = np.maximum(
                actual_packet_system_time_ms - base_time,
                0.0,
            ).astype(np.float32, copy=False)
            self.prev_actual_packet_delivered_packets = actual_packet_delivered_packets.copy()
            self.prev_actual_packet_delivered_bits = actual_packet_delivered_bits.copy()
            self.prev_actual_packet_system_time_ms = actual_packet_system_time_ms.copy()
            self.have_prev_actual_packet = True
        else:
            actual_packet_delivered_packets = np.zeros((self.n_ue,), dtype=np.float32)
            actual_packet_pending_packets = np.zeros((self.n_ue,), dtype=np.float32)
            actual_packet_delivered_bits = np.zeros((self.n_ue,), dtype=np.float32)
            actual_packet_system_time_ms = np.zeros((self.n_ue,), dtype=np.float32)
            actual_packet_service_rate_mbps = np.zeros((self.n_ue,), dtype=np.float32)
            actual_packet_service_rate_per_packet_mean_mbps = np.zeros((self.n_ue,), dtype=np.float32)
            packet_delta_delivered_packets = np.zeros((self.n_ue,), dtype=np.float32)
            packet_delta_delivered_bits = np.zeros((self.n_ue,), dtype=np.float32)
            packet_delta_system_time_ms = np.zeros((self.n_ue,), dtype=np.float32)

        served_for_window = demand_mask & served_mask
        goodput_served_for_window = demand_mask & goodput_served_mask
        old_demand = self.demand_hist[self.ptr].astype(np.int32, copy=False)
        old_served = self.served_hist[self.ptr].astype(np.int32, copy=False)
        old_goodput_served = self.goodput_served_hist[self.ptr].astype(np.int32, copy=False)
        old_demand_backlog = self.demand_backlog_hist[self.ptr]
        old_unserved_backlog = self.unserved_backlog_hist[self.ptr]
        old_effective_prg = self.effective_prg_hist[self.ptr]
        old_goodput_bytes = self.goodput_bytes_hist[self.ptr]
        old_packet_delivered_packets = self.packet_delivered_packets_hist[self.ptr]
        old_packet_delivered_bits = self.packet_delivered_bits_hist[self.ptr]
        old_packet_system_time_ms = self.packet_system_time_ms_hist[self.ptr]
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
        new_packet_delivered_packets = packet_delta_delivered_packets.astype(np.float32, copy=False)
        new_packet_delivered_bits = packet_delta_delivered_bits.astype(np.float32, copy=False)
        new_packet_system_time_ms = packet_delta_system_time_ms.astype(np.float32, copy=False)
        self.demand_count += new_demand - old_demand
        self.served_count += new_served - old_served
        self.goodput_served_count += new_goodput_served - old_goodput_served
        self.demand_backlog_sum += new_demand_backlog - old_demand_backlog
        self.unserved_backlog_sum += new_unserved_backlog - old_unserved_backlog
        self.effective_prg_sum += new_effective_prg - old_effective_prg
        self.goodput_bytes_sum += new_goodput_bytes - old_goodput_bytes
        self.packet_delivered_packets_sum += new_packet_delivered_packets - old_packet_delivered_packets
        self.packet_delivered_bits_sum += new_packet_delivered_bits - old_packet_delivered_bits
        self.packet_system_time_ms_sum += new_packet_system_time_ms - old_packet_system_time_ms
        self.demand_hist[self.ptr] = new_demand.astype(np.int16, copy=False)
        self.served_hist[self.ptr] = new_served.astype(np.int16, copy=False)
        self.goodput_served_hist[self.ptr] = new_goodput_served.astype(np.int16, copy=False)
        self.demand_backlog_hist[self.ptr] = new_demand_backlog
        self.unserved_backlog_hist[self.ptr] = new_unserved_backlog
        self.effective_prg_hist[self.ptr] = new_effective_prg
        self.goodput_bytes_hist[self.ptr] = new_goodput_bytes
        self.packet_delivered_packets_hist[self.ptr] = new_packet_delivered_packets
        self.packet_delivered_bits_hist[self.ptr] = new_packet_delivered_bits
        self.packet_system_time_ms_hist[self.ptr] = new_packet_system_time_ms
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
        self.last_actual_packet_available = actual_packet_ok
        self.last_actual_packet_delivered_packets = actual_packet_delivered_packets
        self.last_actual_packet_pending_packets = actual_packet_pending_packets
        self.last_actual_packet_delivered_bits = actual_packet_delivered_bits
        self.last_actual_packet_system_time_ms = actual_packet_system_time_ms
        self.last_actual_packet_service_rate_mbps = actual_packet_service_rate_mbps
        self.last_actual_packet_service_rate_per_packet_mean_mbps = (
            actual_packet_service_rate_per_packet_mean_mbps
        )
        self.last_actual_packet_delta_delivered_packets = packet_delta_delivered_packets
        self.last_actual_packet_delta_delivered_bits = packet_delta_delivered_bits
        self.last_actual_packet_delta_system_time_ms = packet_delta_system_time_ms

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

    def recent100_packet_rate_mbps(self) -> np.ndarray:
        rate = np.zeros((self.n_ue,), dtype=np.float32)
        has_time = self.packet_system_time_ms_sum > 1.0e-6
        rate[has_time] = (
            self.packet_delivered_bits_sum[has_time]
            / self.packet_system_time_ms_sum[has_time]
            / 1000.0
        )
        return rate.astype(np.float32, copy=False)

    def recent100_packet_delay_ms(self) -> np.ndarray:
        delay = np.zeros((self.n_ue,), dtype=np.float32)
        has_packets = self.packet_delivered_packets_sum > 0.0
        delay[has_packets] = (
            self.packet_system_time_ms_sum[has_packets]
            / self.packet_delivered_packets_sum[has_packets]
        )
        return delay.astype(np.float32, copy=False)

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
                "actual_packet_diagnostics_available": float(self.last_actual_packet_available),
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
                "ue_macro_rate_proxy_mbps": 0.0,
                "ue_p10_rate_proxy_mbps": 0.0,
                "ue_min_rate_proxy_mbps": 0.0,
                "ue_rate_proxy_active_count": 0.0,
                "ue_macro_packet_rate_mbps": 0.0,
                "ue_p10_packet_rate_mbps": 0.0,
                "ue_min_packet_rate_mbps": 0.0,
                "ue_packet_rate_active_count": 0.0,
                "ue_packet_delivered_active_count": 0.0,
                "ue_macro_packet_delay_ms": 0.0,
                "ue_p90_packet_delay_ms": 0.0,
                "recent100_unserved_backlog_p90_bytes": 0.0,
                "recent100_debt_score_p90_bytes": 0.0,
                "recent100_debt_score_max_bytes": 0.0,
                "backlog_p90_bytes": 0.0,
                "backlog_max_bytes": 0.0,
            }
        streak = self.no_service_streak[demand_mask].astype(np.float32, copy=False)
        rates = self.recent100_service_rate()[demand_mask].astype(np.float32, copy=False)
        ue_rate_proxy_mbps = (
            self.recent100_goodput_bytes_mean()[demand_mask].astype(np.float32, copy=False) * 0.016
        )
        ue_packet_rate_mbps = self.recent100_packet_rate_mbps()[demand_mask].astype(
            np.float32,
            copy=False,
        )
        delivered_window = self.packet_delivered_packets_sum[demand_mask].astype(np.float32, copy=False)
        ue_packet_delay_ms = self.recent100_packet_delay_ms()[demand_mask].astype(
            np.float32,
            copy=False,
        )
        delivered_mask = delivered_window > 0.0
        ue_packet_delay_delivered = ue_packet_delay_ms[delivered_mask]
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
            "actual_packet_diagnostics_available": float(self.last_actual_packet_available),
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
            "ue_macro_rate_proxy_mbps": float(np.mean(ue_rate_proxy_mbps) if ue_rate_proxy_mbps.size > 0 else 0.0),
            "ue_p10_rate_proxy_mbps": float(
                np.percentile(ue_rate_proxy_mbps, 10) if ue_rate_proxy_mbps.size > 0 else 0.0
            ),
            "ue_min_rate_proxy_mbps": float(np.min(ue_rate_proxy_mbps) if ue_rate_proxy_mbps.size > 0 else 0.0),
            "ue_rate_proxy_active_count": float(ue_rate_proxy_mbps.size),
            "ue_macro_packet_rate_mbps": float(
                np.mean(ue_packet_rate_mbps) if ue_packet_rate_mbps.size > 0 else 0.0
            ),
            "ue_p10_packet_rate_mbps": float(
                np.percentile(ue_packet_rate_mbps, 10) if ue_packet_rate_mbps.size > 0 else 0.0
            ),
            "ue_min_packet_rate_mbps": float(
                np.min(ue_packet_rate_mbps) if ue_packet_rate_mbps.size > 0 else 0.0
            ),
            "ue_packet_rate_active_count": float(ue_packet_rate_mbps.size),
            "ue_packet_delivered_active_count": float(delivered_mask.sum()),
            "ue_macro_packet_delay_ms": float(
                np.mean(ue_packet_delay_delivered) if ue_packet_delay_delivered.size > 0 else 0.0
            ),
            "ue_p90_packet_delay_ms": float(
                np.percentile(ue_packet_delay_delivered, 90)
                if ue_packet_delay_delivered.size > 0
                else 0.0
            ),
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
        packet_rate_mbps = self.recent100_packet_rate_mbps()
        packet_delay_ms = self.recent100_packet_delay_ms()
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
                    "actual_packet_diagnostics_available": float(self.last_actual_packet_available),
                    "actual_packet_delivered_packets": float(
                        self.last_actual_packet_delivered_packets[ue_idx]
                    ),
                    "actual_packet_pending_packets": float(
                        self.last_actual_packet_pending_packets[ue_idx]
                    ),
                    "actual_packet_delivered_bits": float(
                        self.last_actual_packet_delivered_bits[ue_idx]
                    ),
                    "actual_packet_system_time_ms": float(
                        self.last_actual_packet_system_time_ms[ue_idx]
                    ),
                    "actual_packet_service_rate_mbps": float(
                        self.last_actual_packet_service_rate_mbps[ue_idx]
                    ),
                    "actual_packet_service_rate_per_packet_mean_mbps": float(
                        self.last_actual_packet_service_rate_per_packet_mean_mbps[ue_idx]
                    ),
                    "actual_packet_delta_delivered_packets": float(
                        self.last_actual_packet_delta_delivered_packets[ue_idx]
                    ),
                    "actual_packet_delta_delivered_bits": float(
                        self.last_actual_packet_delta_delivered_bits[ue_idx]
                    ),
                    "actual_packet_delta_system_time_ms": float(
                        self.last_actual_packet_delta_system_time_ms[ue_idx]
                    ),
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
                    "recent100_packet_delivered_packets": float(
                        self.packet_delivered_packets_sum[ue_idx]
                    ),
                    "recent100_packet_delivered_bits": float(self.packet_delivered_bits_sum[ue_idx]),
                    "recent100_packet_system_time_ms": float(self.packet_system_time_ms_sum[ue_idx]),
                    "recent100_packet_rate_mbps": float(packet_rate_mbps[ue_idx]),
                    "recent100_packet_delay_ms": float(packet_delay_ms[ue_idx]),
                    "recent100_debt_score_bytes": float(debt_scores[ue_idx]),
                }
            )
        return rows


_PACKET_TAIL_FEATURE_NAMES = (
    "recent_packet_rate_gap_norm",
    "recent_packet_delay_norm",
    "recent_packet_no_delivery_flag",
)


def _append_packet_tail_features(
    graph: SparseEntityGraph,
    tracker: PerUeDiagnosticsTracker | None,
    *,
    enabled: bool,
    rate_target_mbps: float,
    delay_scale_ms: float,
) -> SparseEntityGraph:
    if not enabled:
        return graph
    ue_block = graph.snapshot.ue_features
    ue_values = ue_block.values
    n_ue = int(graph.snapshot.static_meta.n_ue)
    if int(ue_values.shape[0]) != n_ue:
        return graph

    base_values = ue_values
    base_names = tuple(ue_block.names)
    if len(base_names) >= len(_PACKET_TAIL_FEATURE_NAMES) and base_names[-3:] == _PACKET_TAIL_FEATURE_NAMES:
        base_values = base_values[:, :-3]
        base_names = base_names[:-3]

    device = base_values.device
    dtype = base_values.dtype
    zeros = torch.zeros((n_ue,), dtype=dtype, device=device)
    if tracker is None or int(tracker.n_ue) != n_ue:
        extra = torch.stack((zeros, zeros, zeros), dim=-1)
    else:
        rate = torch.as_tensor(tracker.recent100_packet_rate_mbps(), dtype=dtype, device=device)
        delay = torch.as_tensor(tracker.recent100_packet_delay_ms(), dtype=dtype, device=device)
        delivered = torch.as_tensor(
            tracker.packet_delivered_packets_sum > 0.0,
            dtype=torch.bool,
            device=device,
        )
        demand = torch.as_tensor(tracker.last_demand, dtype=torch.bool, device=device)
        history_scale = min(1.0, float(max(0, int(tracker.filled))) / 20.0)
        target = max(float(rate_target_mbps), 1.0e-6)
        delay_scale = max(float(delay_scale_ms), 1.0e-6)
        rate_gap = torch.clamp((target - rate) / target, min=0.0, max=2.0)
        rate_gap = torch.where(delivered & demand, rate_gap, torch.zeros_like(rate_gap))
        cold_gap = torch.full_like(rate_gap, float(history_scale))
        rate_gap = torch.where((~delivered) & demand, cold_gap, rate_gap)
        delay_norm = torch.clamp(delay / delay_scale, min=0.0, max=2.0)
        delay_norm = torch.where(delivered & demand, delay_norm, torch.zeros_like(delay_norm))
        no_delivery = ((~delivered) & demand).to(dtype=dtype) * float(history_scale)
        extra = torch.stack(
            (
                rate_gap.to(dtype=dtype),
                delay_norm.to(dtype=dtype),
                no_delivery.to(dtype=dtype),
            ),
            dim=-1,
        )

    new_ue = NamedFeatureBlock(
        values=torch.cat((base_values, extra), dim=-1).contiguous(),
        names=base_names + _PACKET_TAIL_FEATURE_NAMES,
    )
    new_snapshot = replace(graph.snapshot, ue_features=new_ue)
    return replace(graph, snapshot=new_snapshot)


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
    min_coef = max(0.0, float(getattr(args, "teacher_aux_min_coef", 0.0)))
    return max(min_coef, base * progress)


def _tail_risk_potential_from_arrays(
    *,
    demand_mask: np.ndarray,
    backlog_bytes: np.ndarray,
    hol_delay_ms: np.ndarray | None,
    ttl_slack_ms: np.ndarray | None,
    service_rate: np.ndarray | None,
    goodput_deficit_norm: np.ndarray | None,
    streak: np.ndarray | None,
    service_target: float,
    ttl_guard_ms: float,
    topk: int,
) -> Dict[str, float]:
    demand = np.asarray(demand_mask, dtype=bool).reshape(-1)
    if demand.size == 0 or not bool(np.any(demand)):
        return {
            "tail_potential": 0.0,
            "tail_topk_mean": 0.0,
            "tail_max": 0.0,
            "tail_mean": 0.0,
            "tail_active_count": 0.0,
        }

    n_ue = int(demand.size)

    def _vec(values, default: float = 0.0) -> np.ndarray:
        if values is None:
            return np.full((n_ue,), default, dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size < n_ue:
            arr = np.pad(arr, (0, n_ue - arr.size), constant_values=default)
        return arr[:n_ue]

    backlog = np.maximum(_vec(backlog_bytes), 0.0)
    hol = np.maximum(_vec(hol_delay_ms), 0.0)
    ttl = _vec(ttl_slack_ms, default=-1.0)
    if service_rate is None:
        service = np.ones((n_ue,), dtype=np.float32)
    else:
        service = np.clip(_vec(service_rate, default=1.0), 0.0, 1.0)
    deficit = np.clip(_vec(goodput_deficit_norm), 0.0, 1.5)
    streak_v = np.maximum(_vec(streak), 0.0)

    target = max(0.05, float(service_target))
    ttl_guard = max(1.0, float(ttl_guard_ms))
    backlog_score = np.clip(np.log1p(backlog / 3000.0) / 6.0, 0.0, 1.5)
    hol_score = np.clip(hol / 200.0, 0.0, 1.5)
    ttl_score = np.where(ttl >= 0.0, np.clip((ttl_guard - ttl) / ttl_guard, 0.0, 1.5), 0.0)
    service_gap = np.clip((target - service) / target, 0.0, 1.5)
    streak_score = np.clip(streak_v / 10.0, 0.0, 1.5)
    risk = (
        0.26 * backlog_score
        + 0.22 * ttl_score
        + 0.20 * service_gap
        + 0.14 * deficit
        + 0.10 * hol_score
        + 0.08 * streak_score
    )
    active_risk = risk[demand].astype(np.float32, copy=False)
    if active_risk.size == 0:
        return {
            "tail_potential": 0.0,
            "tail_topk_mean": 0.0,
            "tail_max": 0.0,
            "tail_mean": 0.0,
            "tail_active_count": 0.0,
        }
    k = max(1, min(int(topk), int(active_risk.size)))
    top = np.partition(active_risk, active_risk.size - k)[active_risk.size - k :]
    topk_mean = float(np.mean(top))
    tail_max = float(np.max(active_risk))
    return {
        "tail_potential": float(0.65 * topk_mean + 0.35 * tail_max),
        "tail_topk_mean": topk_mean,
        "tail_max": tail_max,
        "tail_mean": float(np.mean(active_risk)),
        "tail_active_count": float(active_risk.size),
    }


def _expiry_risk_potential_from_arrays(
    *,
    demand_mask: np.ndarray,
    backlog_bytes: np.ndarray,
    hol_delay_ms: np.ndarray | None,
    ttl_slack_ms: np.ndarray | None,
    service_rate: np.ndarray | None,
    goodput_deficit_norm: np.ndarray | None,
    streak: np.ndarray | None,
    ttl_guard_ms: float,
    topk: int,
) -> Dict[str, float]:
    demand = np.asarray(demand_mask, dtype=bool).reshape(-1)
    if demand.size == 0 or not bool(np.any(demand)):
        return {
            "expiry_potential": 0.0,
            "expiry_topk_mean": 0.0,
            "expiry_max": 0.0,
            "expiry_mean": 0.0,
            "expiry_active_count": 0.0,
        }

    n_ue = int(demand.size)

    def _vec(values, default: float = 0.0) -> np.ndarray:
        if values is None:
            return np.full((n_ue,), default, dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size < n_ue:
            arr = np.pad(arr, (0, n_ue - arr.size), constant_values=default)
        return arr[:n_ue]

    backlog = np.maximum(_vec(backlog_bytes), 0.0)
    hol = np.maximum(_vec(hol_delay_ms), 0.0)
    ttl = _vec(ttl_slack_ms, default=-1.0)
    if service_rate is None:
        service = np.ones((n_ue,), dtype=np.float32)
    else:
        service = np.clip(_vec(service_rate, default=1.0), 0.0, 1.0)
    deficit = np.clip(_vec(goodput_deficit_norm), 0.0, 1.5)
    streak_v = np.maximum(_vec(streak), 0.0)

    ttl_guard = max(1.0, float(ttl_guard_ms))
    ttl_known = ttl >= 0.0
    near_drop = np.where(ttl_known, np.clip((ttl_guard - ttl) / ttl_guard, 0.0, 2.0), 0.0)
    critical_guard = max(1.0, 0.5 * ttl_guard)
    critical = np.where(ttl_known, np.clip((critical_guard - ttl) / critical_guard, 0.0, 2.0), 0.0)
    backlog_score = np.clip(np.log1p(backlog / 3000.0) / 6.0, 0.0, 1.5)
    hol_score = np.clip(hol / 200.0, 0.0, 1.5)
    service_gap = np.clip(1.0 - service, 0.0, 1.5)
    streak_score = np.clip(streak_v / 10.0, 0.0, 1.5)
    risk = (
        0.48 * near_drop
        + 0.20 * critical
        + 0.12 * backlog_score
        + 0.08 * hol_score
        + 0.06 * service_gap
        + 0.04 * deficit
        + 0.02 * streak_score
    )
    active_risk = risk[demand].astype(np.float32, copy=False)
    if active_risk.size == 0:
        return {
            "expiry_potential": 0.0,
            "expiry_topk_mean": 0.0,
            "expiry_max": 0.0,
            "expiry_mean": 0.0,
            "expiry_active_count": 0.0,
        }
    k = max(1, min(int(topk), int(active_risk.size)))
    top = np.partition(active_risk, active_risk.size - k)[active_risk.size - k :]
    topk_mean = float(np.mean(top))
    expiry_max = float(np.max(active_risk))
    return {
        "expiry_potential": float(0.75 * topk_mean + 0.25 * expiry_max),
        "expiry_topk_mean": topk_mean,
        "expiry_max": expiry_max,
        "expiry_mean": float(np.mean(active_risk)),
        "expiry_active_count": float(active_risk.size),
    }


def _ue_rescue_scores_from_arrays(
    *,
    demand_mask: np.ndarray,
    backlog_bytes: np.ndarray,
    hol_delay_ms: np.ndarray | None,
    ttl_slack_ms: np.ndarray | None,
    service_rate: np.ndarray | None,
    goodput_deficit_norm: np.ndarray | None,
    streak: np.ndarray | None,
    service_target: float,
    ttl_guard_ms: float,
    max_score: float = 1.5,
) -> np.ndarray:
    demand = np.asarray(demand_mask, dtype=bool).reshape(-1)
    n_ue = int(demand.size)
    if n_ue <= 0:
        return np.zeros((0,), dtype=np.float32)

    def _vec(values, default: float = 0.0) -> np.ndarray:
        if values is None:
            return np.full((n_ue,), default, dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size < n_ue:
            arr = np.pad(arr, (0, n_ue - arr.size), constant_values=default)
        return arr[:n_ue]

    backlog = np.maximum(_vec(backlog_bytes), 0.0)
    hol = np.maximum(_vec(hol_delay_ms), 0.0)
    ttl = _vec(ttl_slack_ms, default=-1.0)
    service = np.ones((n_ue,), dtype=np.float32) if service_rate is None else np.clip(_vec(service_rate, 1.0), 0.0, 1.0)
    deficit = np.clip(_vec(goodput_deficit_norm), 0.0, 1.5)
    streak_v = np.maximum(_vec(streak), 0.0)

    target = max(0.05, float(service_target))
    ttl_guard = max(1.0, float(ttl_guard_ms))
    critical_guard = max(1.0, 0.5 * ttl_guard)
    ttl_known = ttl >= 0.0
    ttl_score = np.where(ttl_known, np.clip((ttl_guard - ttl) / ttl_guard, 0.0, 2.0), 0.0)
    critical_score = np.where(ttl_known, np.clip((critical_guard - ttl) / critical_guard, 0.0, 2.0), 0.0)
    service_gap = np.clip((target - service) / target, 0.0, 1.5)
    backlog_score = np.clip(np.log1p(backlog / 3000.0) / 6.0, 0.0, 1.5)
    hol_score = np.clip(hol / 200.0, 0.0, 1.5)
    streak_score = np.clip(streak_v / 10.0, 0.0, 1.5)
    rescue = (
        0.34 * ttl_score
        + 0.16 * critical_score
        + 0.18 * deficit
        + 0.14 * service_gap
        + 0.09 * backlog_score
        + 0.05 * hol_score
        + 0.04 * streak_score
    )
    rescue = np.clip(rescue, 0.0, max(0.0, float(max_score))).astype(np.float32, copy=False)
    return np.where(demand, rescue, 0.0).astype(np.float32, copy=False)


def _rescue_credit_from_scores(
    *,
    rescue_score: np.ndarray,
    actual_goodput_bytes: object,
    topk: int,
    goodput_bytes_scale: float,
) -> Dict[str, float]:
    risk = np.asarray(rescue_score, dtype=np.float32).reshape(-1)
    if risk.size == 0:
        return {
            "rescue_credit": 0.0,
            "rescue_miss": 0.0,
            "rescue_risk_topk_mean": 0.0,
            "rescue_served_ratio": 0.0,
        }
    try:
        goodput = np.asarray(actual_goodput_bytes, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        goodput = np.zeros((risk.size,), dtype=np.float32)
    if goodput.size < risk.size:
        goodput = np.pad(goodput, (0, risk.size - goodput.size), constant_values=0.0)
    goodput = goodput[: risk.size]

    active = risk > 0.0
    if not bool(np.any(active)):
        return {
            "rescue_credit": 0.0,
            "rescue_miss": 0.0,
            "rescue_risk_topk_mean": 0.0,
            "rescue_served_ratio": 0.0,
        }
    active_idx = np.flatnonzero(active)
    active_risk = risk[active_idx]
    k = max(1, min(int(topk), int(active_risk.size)))
    local_top = np.argpartition(active_risk, active_risk.size - k)[active_risk.size - k :]
    top_idx = active_idx[local_top]
    top_risk = risk[top_idx]
    served_fraction = np.clip(goodput[top_idx] / max(1.0, float(goodput_bytes_scale)), 0.0, 1.0)
    return {
        "rescue_credit": float(np.mean(top_risk * served_fraction)),
        "rescue_miss": float(np.mean(top_risk * (1.0 - served_fraction))),
        "rescue_risk_topk_mean": float(np.mean(top_risk)),
        "rescue_served_ratio": float(np.mean(served_fraction > 0.0)),
    }


def _soft_target_cross_entropy(logits: torch.Tensor, target_prob: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != target_prob.shape:
        raise ValueError(f"soft target shape mismatch: logits={tuple(logits.shape)} target={tuple(target_prob.shape)}")
    logp = torch.log_softmax(logits, dim=-1)
    per_item = -(target_prob.to(dtype=logp.dtype) * logp).sum(dim=-1)
    row_mask = valid_mask.to(dtype=torch.bool).any(dim=-1) & torch.isfinite(per_item)
    if not bool(row_mask.any().item()):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return per_item[row_mask].mean()


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


def _prg_feature_raw_mean(graph: SparseEntityGraph, feature_idx: int) -> float:
    prg = graph.snapshot.prg_features.values
    if prg.shape[1] <= feature_idx:
        return 0.0
    return float(prg[:, feature_idx].mean().item())


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


def _budgeted_candidate_logp_entropy(
    graph: SparseEntityGraph,
    packed,
    candidate_valid: torch.Tensor,
    prg_logits: torch.Tensor,
    prg_action_class: torch.Tensor,
    *,
    mode: str = "sample",
    demand_cap_slack_bytes: float = 3000.0,
    hard_cap: bool = True,
    usage_penalty: float = 0.35,
    overcap_penalty: float = 8.0,
    first_service_bonus: float = 0.0,
    tail_bias_coef: float = 0.0,
    tail_bias_service_target: float = 0.75,
    tail_bias_ttl_guard_ms: float = 20.0,
    tail_bias_max: float = 1.5,
    rescue_bias_coef: float = 0.0,
    rescue_bias_service_target: float = 0.80,
    rescue_bias_ttl_guard_ms: float = 60.0,
    rescue_bias_max: float = 1.5,
    completion_bonus: float = 0.0,
    completion_progress_weight: float = 0.25,
    completion_max_score: float = 1.5,
    tail_quota_enable: bool = False,
    tail_quota_min_score: float = 0.55,
    tail_quota_topk_ues: int = 6,
    tail_quota_prgs_per_ue: int = 1,
    tail_quota_bonus: float = 4.0,
    tail_quota_order_bonus: float = 2.0,
    tail_quota_max_score: float = 4.0,
    tail_quota_risk_max: float = 0.70,
    seq_state_weights: torch.Tensor | None = None,
    seq_state_coef: float = 1.0,
    seq_fail_risk_feature_offset: int = -1,
    logp_floor: float = -30.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of a PRG action under the sequential budgeted selector.

    This mirrors ``select_budgeted_candidate_actions``.  The PRG visitation
    order is treated as part of the discrete policy; gradients flow through the
    per-step candidate logits, while the order/headroom masks are held fixed for
    the current forward pass.
    """
    select_mode = str(mode).lower()
    if select_mode not in ("sample", "greedy"):
        raise ValueError(f"budgeted mode must be sample or greedy, got {mode!r}")

    device = prg_logits.device
    logits = torch.as_tensor(prg_logits, dtype=torch.float32, device=device)
    expected = (int(packed.num_prg_tokens), int(packed.max_candidates) + 1)
    if tuple(logits.shape) != expected:
        raise ValueError(f"prg_logits mismatch: got={tuple(logits.shape)} expected={expected}")
    valid = candidate_valid.to(device=device, dtype=torch.bool)
    if tuple(valid.shape) != expected:
        raise ValueError(f"candidate_valid mismatch: got={tuple(valid.shape)} expected={expected}")
    target = prg_action_class.to(device=device, dtype=torch.long).reshape(-1)
    if tuple(target.shape) != (int(packed.num_prg_tokens),):
        raise ValueError(
            f"prg_action_class mismatch: got={tuple(target.shape)} expected={(int(packed.num_prg_tokens),)}"
        )

    blank_class = int(packed.blank_class_index)
    if packed.max_candidates <= 0 or packed.num_prg_tokens == 0:
        if target.numel() == 0:
            zero = logits.sum() * 0.0
            return zero, zero
        forced_logp = torch.where(
            target == blank_class,
            torch.zeros_like(target, dtype=logits.dtype, device=device),
            torch.full_like(target, float(logp_floor), dtype=logits.dtype, device=device),
        )
        return forced_logp.mean(), logits.sum() * 0.0

    valid_main = valid[:, :blank_class]
    seq_weights = None
    if seq_state_weights is not None and float(seq_state_coef) != 0.0:
        candidate_seq_weights = torch.as_tensor(seq_state_weights, dtype=logits.dtype, device=device)
        if candidate_seq_weights.dim() == 3 and tuple(candidate_seq_weights.shape[:2]) == (
            int(packed.num_prg_tokens),
            int(packed.max_candidates),
        ):
            seq_weights = candidate_seq_weights
    seq_fail_risk = _candidate_seq_fail_risk_features(
        packed,
        device=device,
        dtype=logits.dtype,
        fail_risk_feature_offset=int(seq_fail_risk_feature_offset),
    )
    safe_ue_ids = packed.ue_ids.to(device=device, dtype=torch.long).clamp(
        min=0,
        max=max(0, int(graph.snapshot.static_meta.n_ue) - 1),
    )
    ue_prg_cap = _estimate_ue_prg_demand_cap(
        graph,
        slack_bytes=float(demand_cap_slack_bytes),
    ).to(device=device, dtype=torch.long)
    ue_prg_used = torch.zeros_like(ue_prg_cap)

    tail_bias = candidate_prg_tail_bias_scores(
        graph,
        packed,
        valid,
        service_target=float(tail_bias_service_target),
        ttl_guard_ms=float(tail_bias_ttl_guard_ms),
        max_score=float(tail_bias_max),
    ).to(device=device, dtype=logits.dtype)
    tail_adjustment = max(0.0, float(tail_bias_coef)) * tail_bias
    success_target_for_rescue, _success_mask = candidate_prg_success_targets(graph, packed, valid)
    rescue_bias, _rescue_mask = candidate_prg_rescue_priority_scores(
        graph,
        packed,
        valid,
        success_target=success_target_for_rescue,
        service_target=float(rescue_bias_service_target),
        ttl_guard_ms=float(rescue_bias_ttl_guard_ms),
        max_score=float(rescue_bias_max),
    )
    rescue_bias = rescue_bias.to(device=device, dtype=logits.dtype)
    rescue_adjustment = max(0.0, float(rescue_bias_coef)) * rescue_bias
    completion_coef = max(0.0, float(completion_bonus))
    zero_used = torch.zeros_like(safe_ue_ids, dtype=logits.dtype, device=device)
    completion_score = _candidate_packet_completion_scores(
        graph,
        safe_ue_ids,
        zero_used,
        progress_weight=float(completion_progress_weight),
        max_score=float(completion_max_score),
    ).to(device=device, dtype=logits.dtype)
    completion_adjustment = completion_coef * completion_score
    tail_quota_on = (
        bool(tail_quota_enable)
        and select_mode == "greedy"
        and int(tail_quota_topk_ues) > 0
        and int(tail_quota_prgs_per_ue) > 0
        and float(tail_quota_bonus) > 0.0
    )
    if tail_quota_on:
        quota_choice, _quota_metrics = select_budgeted_candidate_actions(
            graph,
            packed,
            valid,
            logits.detach(),
            mode="greedy",
            demand_cap_slack_bytes=float(demand_cap_slack_bytes),
            hard_cap=bool(hard_cap),
            usage_penalty=float(usage_penalty),
            overcap_penalty=float(overcap_penalty),
            first_service_bonus=float(first_service_bonus),
            tail_bias_coef=float(tail_bias_coef),
            tail_bias_service_target=float(tail_bias_service_target),
            tail_bias_ttl_guard_ms=float(tail_bias_ttl_guard_ms),
            tail_bias_max=float(tail_bias_max),
            rescue_bias_coef=float(rescue_bias_coef),
            rescue_bias_service_target=float(rescue_bias_service_target),
            rescue_bias_ttl_guard_ms=float(rescue_bias_ttl_guard_ms),
            rescue_bias_max=float(rescue_bias_max),
            completion_bonus=float(completion_bonus),
            completion_progress_weight=float(completion_progress_weight),
            completion_max_score=float(completion_max_score),
            tail_quota_enable=True,
            tail_quota_min_score=float(tail_quota_min_score),
            tail_quota_topk_ues=int(tail_quota_topk_ues),
            tail_quota_prgs_per_ue=int(tail_quota_prgs_per_ue),
            tail_quota_bonus=float(tail_quota_bonus),
            tail_quota_order_bonus=float(tail_quota_order_bonus),
            tail_quota_max_score=float(tail_quota_max_score),
            tail_quota_risk_max=float(tail_quota_risk_max),
            seq_state_weights=seq_weights,
            seq_state_coef=float(seq_state_coef),
            seq_fail_risk_feature_offset=int(seq_fail_risk_feature_offset),
        )
        match = target == quota_choice.to(device=device, dtype=torch.long)
        return torch.where(match, torch.zeros_like(target, dtype=logits.dtype), logits.new_full(target.shape, float(logp_floor))).mean(), logits.sum() * 0.0
    best_main_logit = (
        logits[:, :blank_class] + tail_adjustment + rescue_adjustment + completion_adjustment
    ).masked_fill(~valid_main, -1.0e9).max(dim=-1).values
    order = torch.argsort(best_main_logit.detach(), descending=True, stable=True)
    logp_terms: List[torch.Tensor] = []
    entropy_terms: List[torch.Tensor] = []
    floor = logits.new_tensor(float(logp_floor))

    for prg_idx in order.tolist():
        selected = int(target[prg_idx].item())
        main_scores = logits[prg_idx, :blank_class]
        prg_valid = valid_main[prg_idx]
        if not bool(prg_valid.any().item()):
            logp_terms.append(torch.zeros((), dtype=logits.dtype, device=device) if selected == blank_class else floor)
            entropy_terms.append(torch.zeros((), dtype=logits.dtype, device=device))
            continue

        ue_ids = safe_ue_ids[prg_idx]
        used = ue_prg_used.gather(0, ue_ids).to(dtype=torch.float32)
        cap = ue_prg_cap.gather(0, ue_ids)
        remaining = cap - ue_prg_used.gather(0, ue_ids)
        in_headroom = remaining > 0
        selectable = prg_valid & in_headroom
        seq_delta = _candidate_seq_state_delta(
            seq_weights,
            prg_idx=int(prg_idx),
            used=used,
            cap=cap,
            remaining=remaining,
            fail_risk=seq_fail_risk[prg_idx],
            coef=float(seq_state_coef),
        )
        main_scores = main_scores + seq_delta

        if completion_coef > 0.0:
            completion_score_prg = _candidate_packet_completion_scores(
                graph,
                ue_ids,
                used,
                progress_weight=float(completion_progress_weight),
                max_score=float(completion_max_score),
            ).to(device=device, dtype=logits.dtype)
        else:
            completion_score_prg = torch.zeros_like(main_scores)
        completion_adjustment_prg = completion_coef * completion_score_prg

        if bool(selectable.any().item()):
            adjusted = (
                main_scores
                + tail_adjustment[prg_idx]
                + rescue_adjustment[prg_idx]
                + completion_adjustment_prg
                - float(usage_penalty) * used
            )
            if float(first_service_bonus) != 0.0:
                adjusted = adjusted + torch.where(
                    (used <= 0.0) & (cap > 0),
                    torch.full_like(adjusted, float(first_service_bonus)),
                    torch.zeros_like(adjusted),
                )
            adjusted = adjusted.masked_fill(~selectable, -1.0e9)
        elif bool(hard_cap):
            adjusted = torch.full_like(main_scores, -1.0e9)
        else:
            adjusted = (
                main_scores
                + tail_adjustment[prg_idx]
                + rescue_adjustment[prg_idx]
                + completion_adjustment_prg
                - float(usage_penalty) * used
            )
            adjusted = adjusted - torch.where(
                in_headroom,
                torch.zeros_like(adjusted),
                torch.full_like(adjusted, float(overcap_penalty)),
            )
            adjusted = adjusted.masked_fill(~prg_valid, -1.0e9)

        selectable_now = adjusted > -1.0e8
        has_candidate = bool(selectable_now.any().item())
        selected_is_main = 0 <= selected < blank_class
        selected_valid = selected_is_main and bool(selectable_now[selected].item())
        if not has_candidate:
            logp_terms.append(torch.zeros((), dtype=logits.dtype, device=device) if selected == blank_class else floor)
            entropy_terms.append(torch.zeros((), dtype=logits.dtype, device=device))
        elif select_mode == "greedy":
            greedy_idx = int(adjusted.argmax().item())
            logp_terms.append(torch.zeros((), dtype=logits.dtype, device=device) if selected == greedy_idx else floor)
            entropy_terms.append(torch.zeros((), dtype=logits.dtype, device=device))
        else:
            logp_all = F.log_softmax(adjusted, dim=0)
            probs = logp_all.exp()
            logp_terms.append(logp_all[selected] if selected_valid else floor)
            entropy_terms.append(-(probs[selectable_now] * logp_all[selectable_now]).sum())

        if selected_valid:
            ue_idx = int(ue_ids[selected].item())
            if 0 <= ue_idx < int(ue_prg_used.numel()):
                ue_prg_used[ue_idx] += 1

    if not logp_terms:
        zero = logits.sum() * 0.0
        return zero, zero
    return torch.stack(logp_terms).mean(), torch.stack(entropy_terms).mean()


def _budgeted_candidate_logp_entropy_batch(
    graphs: Sequence[SparseEntityGraph],
    prg_logits: torch.Tensor,
    prg_valid: torch.Tensor,
    ue_action_class: torch.Tensor,
    prg_action_class: torch.Tensor,
    *,
    mode: str = "sample",
    demand_cap_slack_bytes: float = 3000.0,
    hard_cap: bool = True,
    usage_penalty: float = 0.35,
    overcap_penalty: float = 8.0,
    first_service_bonus: float = 0.0,
    tail_bias_coef: float = 0.0,
    tail_bias_service_target: float = 0.75,
    tail_bias_ttl_guard_ms: float = 20.0,
    tail_bias_max: float = 1.5,
    rescue_bias_coef: float = 0.0,
    rescue_bias_service_target: float = 0.80,
    rescue_bias_ttl_guard_ms: float = 60.0,
    rescue_bias_max: float = 1.5,
    completion_bonus: float = 0.0,
    completion_progress_weight: float = 0.25,
    completion_max_score: float = 1.5,
    tail_quota_enable: bool = False,
    tail_quota_min_score: float = 0.55,
    tail_quota_topk_ues: int = 6,
    tail_quota_prgs_per_ue: int = 1,
    tail_quota_bonus: float = 4.0,
    tail_quota_order_bonus: float = 2.0,
    tail_quota_max_score: float = 4.0,
    tail_quota_risk_max: float = 0.70,
    seq_state_weights: torch.Tensor | None = None,
    seq_state_coef: float = 1.0,
    seq_fail_risk_feature_offset: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    logps: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    max_candidates = max(0, int(prg_logits.shape[-1]) - 1)
    for batch_idx, graph in enumerate(graphs):
        packed = pack_prg_candidates(graph, max_candidates=max_candidates, fixed_width=True)
        valid, _selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(
            graph,
            packed,
            ue_action_class[batch_idx].to(dtype=torch.long),
        )
        valid = valid.to(device=prg_valid.device, dtype=torch.bool) & prg_valid[batch_idx].to(dtype=torch.bool)
        seq_weights = None
        if seq_state_weights is not None:
            seq_weights = seq_state_weights[batch_idx]
        logp, entropy = _budgeted_candidate_logp_entropy(
            graph,
            packed,
            valid,
            prg_logits[batch_idx],
            prg_action_class[batch_idx],
            mode=mode,
            demand_cap_slack_bytes=float(demand_cap_slack_bytes),
            hard_cap=bool(int(hard_cap)),
            usage_penalty=float(usage_penalty),
            overcap_penalty=float(overcap_penalty),
            first_service_bonus=float(first_service_bonus),
            tail_bias_coef=float(tail_bias_coef),
            tail_bias_service_target=float(tail_bias_service_target),
            tail_bias_ttl_guard_ms=float(tail_bias_ttl_guard_ms),
            tail_bias_max=float(tail_bias_max),
            rescue_bias_coef=float(rescue_bias_coef),
            rescue_bias_service_target=float(rescue_bias_service_target),
            rescue_bias_ttl_guard_ms=float(rescue_bias_ttl_guard_ms),
            rescue_bias_max=float(rescue_bias_max),
            completion_bonus=float(completion_bonus),
            completion_progress_weight=float(completion_progress_weight),
            completion_max_score=float(completion_max_score),
            tail_quota_enable=bool(tail_quota_enable),
            tail_quota_min_score=float(tail_quota_min_score),
            tail_quota_topk_ues=int(tail_quota_topk_ues),
            tail_quota_prgs_per_ue=int(tail_quota_prgs_per_ue),
            tail_quota_bonus=float(tail_quota_bonus),
            tail_quota_order_bonus=float(tail_quota_order_bonus),
            tail_quota_max_score=float(tail_quota_max_score),
            tail_quota_risk_max=float(tail_quota_risk_max),
            seq_state_weights=seq_weights,
            seq_state_coef=float(seq_state_coef),
            seq_fail_risk_feature_offset=int(seq_fail_risk_feature_offset),
        )
        logps.append(logp)
        entropies.append(entropy)
    return torch.stack(logps, dim=0), torch.stack(entropies, dim=0)


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


def _masked_candidate_prg_logits(
    actor_out: Dict[str, torch.Tensor],
    graph: SparseEntityGraph,
    *,
    ue_action_class: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "candidate_prg_logits" not in actor_out:
        raise KeyError("candidate_prg_logits missing from actor output")
    logits = actor_out["candidate_prg_logits"]
    device = logits.device
    if logits.dim() != 2:
        raise ValueError(f"candidate_prg_logits must be [P,K+1], got {tuple(logits.shape)}")
    max_candidates = max(0, int(logits.shape[-1]) - 1)
    packed = pack_prg_candidates(graph, max_candidates=max_candidates, fixed_width=True)
    valid, _selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(
        graph,
        packed,
        ue_action_class.to(device=device, dtype=torch.long),
    )
    valid = valid.to(device=device, dtype=torch.bool)
    return logits.masked_fill(~valid, -1.0e9), valid


def _masked_candidate_prg_logits_batch(
    actor_out: Dict[str, torch.Tensor],
    graphs: Sequence[SparseEntityGraph],
    *,
    ue_action_class: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "candidate_prg_logits" not in actor_out:
        raise KeyError("candidate_prg_logits missing from actor output")
    logits = actor_out["candidate_prg_logits"]
    device = logits.device
    if logits.dim() != 3:
        raise ValueError(f"candidate_prg_logits must be [B,P,K+1], got {tuple(logits.shape)}")
    batch_size = len(graphs)
    if int(logits.shape[0]) != batch_size:
        raise ValueError(f"candidate batch mismatch: logits={tuple(logits.shape)} graphs={batch_size}")
    max_candidates = max(0, int(logits.shape[-1]) - 1)
    valid_rows: List[torch.Tensor] = []
    for batch_idx, graph in enumerate(graphs):
        packed = pack_prg_candidates(graph, max_candidates=max_candidates, fixed_width=True)
        valid, _selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(
            graph,
            packed,
            ue_action_class[batch_idx].to(device=device, dtype=torch.long),
        )
        valid_rows.append(valid.to(device=device, dtype=torch.bool))
    valid_mask = torch.stack(valid_rows, dim=0)
    return logits.masked_fill(~valid_mask, -1.0e9), valid_mask


def _apply_candidate_direct_service_bias(
    logits: torch.Tensor,
    graph: SparseEntityGraph,
    packed,
    valid: torch.Tensor,
    *,
    tail_bias_coef: float = 0.0,
    tail_bias_service_target: float = 0.75,
    tail_bias_ttl_guard_ms: float = 20.0,
    tail_bias_max: float = 1.5,
    rescue_bias_coef: float = 0.0,
    rescue_bias_service_target: float = 0.80,
    rescue_bias_ttl_guard_ms: float = 60.0,
    rescue_bias_max: float = 1.5,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor | None, torch.Tensor | None]:
    device = logits.device
    valid = valid.to(device=device, dtype=torch.bool)
    blank_class = int(packed.blank_class_index)
    valid_main = valid[:, :blank_class] if blank_class > 0 else valid[:, :0]
    tail_coef = max(0.0, float(tail_bias_coef))
    rescue_coef = max(0.0, float(rescue_bias_coef))
    metrics = {
        "candidate_budgeted_changed_count": 0.0,
        "candidate_budgeted_no_headroom_count": 0.0,
        "candidate_budgeted_active_ue_count": 0.0,
        "candidate_budgeted_tail_bias_coef": float(tail_coef),
        "candidate_budgeted_tail_bias_candidate_mean": 0.0,
        "candidate_budgeted_tail_bias_selected_mean": 0.0,
        "candidate_budgeted_tail_bias_active_count": 0.0,
        "candidate_budgeted_rescue_bias_coef": float(rescue_coef),
        "candidate_budgeted_rescue_bias_candidate_mean": 0.0,
        "candidate_budgeted_rescue_bias_selected_mean": 0.0,
        "candidate_budgeted_rescue_bias_active_count": 0.0,
    }
    if blank_class <= 0 or (tail_coef <= 0.0 and rescue_coef <= 0.0):
        return logits.masked_fill(~valid, -1.0e9), metrics, None, None

    adjusted = logits.clone()
    tail_scores = None
    rescue_scores = None
    if tail_coef > 0.0:
        tail_scores = candidate_prg_tail_bias_scores(
            graph,
            packed,
            valid,
            service_target=float(tail_bias_service_target),
            ttl_guard_ms=float(tail_bias_ttl_guard_ms),
            max_score=float(tail_bias_max),
        ).to(device=device, dtype=adjusted.dtype)
        adjusted[:, :blank_class] = adjusted[:, :blank_class] + tail_coef * tail_scores
        valid_tail = tail_scores[valid_main]
        metrics["candidate_budgeted_tail_bias_candidate_mean"] = (
            float(valid_tail.mean().item()) if valid_tail.numel() > 0 else 0.0
        )
        metrics["candidate_budgeted_tail_bias_active_count"] = float(((tail_scores > 0.5) & valid_main).sum().item())

    if rescue_coef > 0.0:
        success_target, _success_mask = candidate_prg_success_targets(graph, packed, valid)
        rescue_scores, _rescue_mask = candidate_prg_rescue_priority_scores(
            graph,
            packed,
            valid,
            success_target=success_target,
            service_target=float(rescue_bias_service_target),
            ttl_guard_ms=float(rescue_bias_ttl_guard_ms),
            max_score=float(rescue_bias_max),
        )
        rescue_scores = rescue_scores.to(device=device, dtype=adjusted.dtype)
        adjusted[:, :blank_class] = adjusted[:, :blank_class] + rescue_coef * rescue_scores
        valid_rescue = rescue_scores[valid_main]
        metrics["candidate_budgeted_rescue_bias_candidate_mean"] = (
            float(valid_rescue.mean().item()) if valid_rescue.numel() > 0 else 0.0
        )
        metrics["candidate_budgeted_rescue_bias_active_count"] = float(
            ((rescue_scores > 0.5) & valid_main).sum().item()
        )

    return adjusted.masked_fill(~valid, -1.0e9), metrics, tail_scores, rescue_scores


def _candidate_direct_bias_selected_mean(scores: torch.Tensor | None, prg_class: torch.Tensor, blank_class: int) -> float:
    if scores is None or int(blank_class) <= 0:
        return 0.0
    selected = prg_class.to(device=scores.device, dtype=torch.long).reshape(-1)
    selected_main = (selected >= 0) & (selected < int(blank_class))
    if not bool(selected_main.any().item()):
        return 0.0
    row = torch.arange(selected.numel(), device=scores.device, dtype=torch.long)[selected_main]
    col = selected[selected_main]
    return float(scores[row, col].mean().item())


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
    *,
    candidate_decode_demand_slack_bytes: float = 3000.0,
    candidate_budgeted_select_mode: str = "off",
    candidate_budgeted_hard_cap: int = 1,
    candidate_budgeted_usage_penalty: float = 0.35,
    candidate_budgeted_overcap_penalty: float = 8.0,
    candidate_budgeted_first_service_bonus: float = 0.0,
    candidate_budgeted_tail_bias_coef: float = 0.0,
    candidate_budgeted_tail_bias_service_target: float = 0.75,
    candidate_budgeted_tail_bias_ttl_guard_ms: float = 20.0,
    candidate_budgeted_tail_bias_max: float = 1.5,
    candidate_budgeted_rescue_bias_coef: float = 0.0,
    candidate_budgeted_rescue_bias_service_target: float = 0.80,
    candidate_budgeted_rescue_bias_ttl_guard_ms: float = 60.0,
    candidate_budgeted_rescue_bias_max: float = 1.5,
    candidate_budgeted_completion_bonus: float = 0.0,
    candidate_budgeted_completion_progress_weight: float = 0.25,
    candidate_budgeted_completion_max_score: float = 1.5,
    candidate_budgeted_tail_quota_enable: int = 0,
    candidate_budgeted_tail_quota_min_score: float = 0.55,
    candidate_budgeted_tail_quota_topk_ues: int = 6,
    candidate_budgeted_tail_quota_prgs_per_ue: int = 1,
    candidate_budgeted_tail_quota_bonus: float = 4.0,
    candidate_budgeted_tail_quota_order_bonus: float = 2.0,
    candidate_budgeted_tail_quota_max_score: float = 4.0,
    candidate_budgeted_tail_quota_risk_max: float = 0.70,
    candidate_seq_state_coef: float = 1.0,
    candidate_seq_fail_risk_feature_offset: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = actor_out["ue_logits"].device
    ue_logits, ue_valid = _masked_ue_logits(actor_out, graph)
    ue_target = ue_action_class.to(device=device, dtype=torch.long).unsqueeze(0)
    ue_target, _ = sanitize_targets(ue_target, ue_valid.unsqueeze(0), ignore_index=-100)
    ue_logp, ue_entropy = _reduce_multi_categorical(ue_logits.unsqueeze(0), ue_target)

    if "candidate_prg_logits" in actor_out:
        prg_logits, prg_valid = _masked_candidate_prg_logits(
            actor_out,
            graph,
            ue_action_class=ue_action_class,
        )
        budget_mode = str(candidate_budgeted_select_mode).lower()
        if budget_mode != "off":
            packed = pack_prg_candidates(
                graph,
                max_candidates=max(0, int(prg_logits.shape[-1]) - 1),
                fixed_width=True,
            )
            valid, _selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(
                graph,
                packed,
                ue_action_class.to(device=device, dtype=torch.long),
            )
            valid = valid.to(device=device, dtype=torch.bool) & prg_valid.to(device=device, dtype=torch.bool)
            prg_logp, prg_entropy = _budgeted_candidate_logp_entropy(
                graph,
                packed,
                valid,
                prg_logits,
                prg_action_class.to(device=device, dtype=torch.long),
                mode=("sample" if budget_mode == "sample" else "greedy"),
                demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
                hard_cap=bool(int(candidate_budgeted_hard_cap)),
                usage_penalty=float(candidate_budgeted_usage_penalty),
                overcap_penalty=float(candidate_budgeted_overcap_penalty),
                first_service_bonus=float(candidate_budgeted_first_service_bonus),
                tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                tail_bias_max=float(candidate_budgeted_tail_bias_max),
                rescue_bias_coef=float(candidate_budgeted_rescue_bias_coef),
                rescue_bias_service_target=float(candidate_budgeted_rescue_bias_service_target),
                rescue_bias_ttl_guard_ms=float(candidate_budgeted_rescue_bias_ttl_guard_ms),
                rescue_bias_max=float(candidate_budgeted_rescue_bias_max),
                completion_bonus=float(candidate_budgeted_completion_bonus),
                completion_progress_weight=float(candidate_budgeted_completion_progress_weight),
                completion_max_score=float(candidate_budgeted_completion_max_score),
                tail_quota_enable=bool(int(candidate_budgeted_tail_quota_enable)),
                tail_quota_min_score=float(candidate_budgeted_tail_quota_min_score),
                tail_quota_topk_ues=int(candidate_budgeted_tail_quota_topk_ues),
                tail_quota_prgs_per_ue=int(candidate_budgeted_tail_quota_prgs_per_ue),
                tail_quota_bonus=float(candidate_budgeted_tail_quota_bonus),
                tail_quota_order_bonus=float(candidate_budgeted_tail_quota_order_bonus),
                tail_quota_max_score=float(candidate_budgeted_tail_quota_max_score),
                tail_quota_risk_max=float(candidate_budgeted_tail_quota_risk_max),
                seq_state_weights=actor_out.get("candidate_seq_state_weights"),
                seq_state_coef=float(candidate_seq_state_coef),
                seq_fail_risk_feature_offset=int(candidate_seq_fail_risk_feature_offset),
            )
            return (ue_logp.squeeze(0) + prg_logp), (0.5 * (ue_entropy.squeeze(0) + prg_entropy))
    else:
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
    *,
    candidate_decode_demand_slack_bytes: float = 3000.0,
    candidate_budgeted_select_mode: str = "off",
    candidate_budgeted_hard_cap: int = 1,
    candidate_budgeted_usage_penalty: float = 0.35,
    candidate_budgeted_overcap_penalty: float = 8.0,
    candidate_budgeted_first_service_bonus: float = 0.0,
    candidate_budgeted_tail_bias_coef: float = 0.0,
    candidate_budgeted_tail_bias_service_target: float = 0.75,
    candidate_budgeted_tail_bias_ttl_guard_ms: float = 20.0,
    candidate_budgeted_tail_bias_max: float = 1.5,
    candidate_budgeted_rescue_bias_coef: float = 0.0,
    candidate_budgeted_rescue_bias_service_target: float = 0.80,
    candidate_budgeted_rescue_bias_ttl_guard_ms: float = 60.0,
    candidate_budgeted_rescue_bias_max: float = 1.5,
    candidate_budgeted_completion_bonus: float = 0.0,
    candidate_budgeted_completion_progress_weight: float = 0.25,
    candidate_budgeted_completion_max_score: float = 1.5,
    candidate_budgeted_tail_quota_enable: int = 0,
    candidate_budgeted_tail_quota_min_score: float = 0.55,
    candidate_budgeted_tail_quota_topk_ues: int = 6,
    candidate_budgeted_tail_quota_prgs_per_ue: int = 1,
    candidate_budgeted_tail_quota_bonus: float = 4.0,
    candidate_budgeted_tail_quota_order_bonus: float = 2.0,
    candidate_budgeted_tail_quota_max_score: float = 4.0,
    candidate_budgeted_tail_quota_risk_max: float = 0.70,
    candidate_seq_state_coef: float = 1.0,
    candidate_seq_fail_risk_feature_offset: int = -1,
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

    if "candidate_prg_logits" in actor_out:
        prg_logits, prg_valid = _masked_candidate_prg_logits_batch(
            actor_out,
            graphs,
            ue_action_class=ue_action_class,
        )
        budget_mode = str(candidate_budgeted_select_mode).lower()
        if budget_mode != "off":
            prg_logp, prg_entropy = _budgeted_candidate_logp_entropy_batch(
                graphs,
                prg_logits,
                prg_valid,
                ue_action_class.to(device=device, dtype=torch.long),
                prg_action_class.to(device=device, dtype=torch.long),
                mode=("sample" if budget_mode == "sample" else "greedy"),
                demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
                hard_cap=bool(int(candidate_budgeted_hard_cap)),
                usage_penalty=float(candidate_budgeted_usage_penalty),
                overcap_penalty=float(candidate_budgeted_overcap_penalty),
                first_service_bonus=float(candidate_budgeted_first_service_bonus),
                tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                tail_bias_max=float(candidate_budgeted_tail_bias_max),
                rescue_bias_coef=float(candidate_budgeted_rescue_bias_coef),
                rescue_bias_service_target=float(candidate_budgeted_rescue_bias_service_target),
                rescue_bias_ttl_guard_ms=float(candidate_budgeted_rescue_bias_ttl_guard_ms),
                rescue_bias_max=float(candidate_budgeted_rescue_bias_max),
                seq_state_weights=actor_out.get("candidate_seq_state_weights"),
                tail_quota_enable=bool(int(candidate_budgeted_tail_quota_enable)),
                tail_quota_min_score=float(candidate_budgeted_tail_quota_min_score),
                tail_quota_topk_ues=int(candidate_budgeted_tail_quota_topk_ues),
                tail_quota_prgs_per_ue=int(candidate_budgeted_tail_quota_prgs_per_ue),
                tail_quota_bonus=float(candidate_budgeted_tail_quota_bonus),
                tail_quota_order_bonus=float(candidate_budgeted_tail_quota_order_bonus),
                tail_quota_max_score=float(candidate_budgeted_tail_quota_max_score),
                tail_quota_risk_max=float(candidate_budgeted_tail_quota_risk_max),
                seq_state_coef=float(candidate_seq_state_coef),
                seq_fail_risk_feature_offset=int(candidate_seq_fail_risk_feature_offset),
            )
            return ue_logp + prg_logp, 0.5 * (ue_entropy + prg_entropy)
        if float(candidate_budgeted_tail_bias_coef) > 0.0 or float(candidate_budgeted_rescue_bias_coef) > 0.0:
            biased_rows: List[torch.Tensor] = []
            max_candidates = max(0, int(prg_logits.shape[-1]) - 1)
            for batch_idx, graph in enumerate(graphs):
                packed = pack_prg_candidates(graph, max_candidates=max_candidates, fixed_width=True)
                biased, _metrics, _tail_scores, _rescue_scores = _apply_candidate_direct_service_bias(
                    prg_logits[batch_idx],
                    graph,
                    packed,
                    prg_valid[batch_idx],
                    tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                    tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                    tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                    tail_bias_max=float(candidate_budgeted_tail_bias_max),
                    rescue_bias_coef=float(candidate_budgeted_rescue_bias_coef),
                    rescue_bias_service_target=float(candidate_budgeted_rescue_bias_service_target),
                    rescue_bias_ttl_guard_ms=float(candidate_budgeted_rescue_bias_ttl_guard_ms),
                    rescue_bias_max=float(candidate_budgeted_rescue_bias_max),
                )
                biased_rows.append(biased)
            prg_logits = torch.stack(biased_rows, dim=0)
    else:
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
    safe_fill_max_per_cell: int = 0,
    safe_fill_min_quality_db: float = -2.0,
    safe_fill_max_quality_gap_db: float = 8.0,
    safe_fill_max_prg_risk: float = 0.70,
    safe_fill_min_backlog_bytes: float = 1.0,
    safe_fill_max_ue_tb_err: float = 0.45,
    candidate_blank_budget_max_per_cell: int = -1,
    candidate_blank_budget_min_decision_logit: float = -1.0e9,
    candidate_decode_demand_slack_bytes: float = 3000.0,
    candidate_marginal_headroom_slack_bytes: float = -1.0,
    candidate_argmax_decode_headroom_rank_mode: str = "stable",
    candidate_budgeted_select_mode: str = "off",
    candidate_budgeted_hard_cap: int = 1,
    candidate_budgeted_usage_penalty: float = 0.35,
    candidate_budgeted_overcap_penalty: float = 8.0,
    candidate_budgeted_first_service_bonus: float = 0.0,
    candidate_budgeted_tail_bias_coef: float = 0.0,
    candidate_budgeted_tail_bias_service_target: float = 0.75,
    candidate_budgeted_tail_bias_ttl_guard_ms: float = 20.0,
    candidate_budgeted_tail_bias_max: float = 1.5,
    candidate_budgeted_rescue_bias_coef: float = 0.0,
    candidate_budgeted_rescue_bias_service_target: float = 0.80,
    candidate_budgeted_rescue_bias_ttl_guard_ms: float = 60.0,
    candidate_budgeted_rescue_bias_max: float = 1.5,
    candidate_budgeted_completion_bonus: float = 0.0,
    candidate_budgeted_completion_progress_weight: float = 0.25,
    candidate_budgeted_completion_max_score: float = 1.5,
    candidate_budgeted_tail_quota_enable: int = 0,
    candidate_budgeted_tail_quota_min_score: float = 0.55,
    candidate_budgeted_tail_quota_topk_ues: int = 6,
    candidate_budgeted_tail_quota_prgs_per_ue: int = 1,
    candidate_budgeted_tail_quota_bonus: float = 4.0,
    candidate_budgeted_tail_quota_order_bonus: float = 2.0,
    candidate_budgeted_tail_quota_max_score: float = 4.0,
    candidate_budgeted_tail_quota_risk_max: float = 0.70,
    candidate_seq_state_coef: float = 1.0,
    candidate_seq_fail_risk_feature_offset: int = -1,
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
    use_candidate_prg = "candidate_prg_logits" in prg_actor_out
    packed = None
    budgeted_metrics: Dict[str, float] = {}
    tail_scores: torch.Tensor | None = None
    rescue_scores: torch.Tensor | None = None
    budget_mode = str(candidate_budgeted_select_mode).lower()
    if use_candidate_prg:
        prg_logits_masked, prg_valid = _masked_candidate_prg_logits(
            prg_actor_out,
            graph,
            ue_action_class=ue_class,
        )
        packed = pack_prg_candidates(
            graph,
            max_candidates=max(0, int(prg_logits_masked.shape[-1]) - 1),
            fixed_width=True,
        )
        if budget_mode == "off" and (
            float(candidate_budgeted_tail_bias_coef) > 0.0 or float(candidate_budgeted_rescue_bias_coef) > 0.0
        ):
            prg_logits_masked, budgeted_metrics, tail_scores, rescue_scores = _apply_candidate_direct_service_bias(
                prg_logits_masked,
                graph,
                packed,
                prg_valid,
                tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                tail_bias_max=float(candidate_budgeted_tail_bias_max),
                rescue_bias_coef=float(candidate_budgeted_rescue_bias_coef),
                rescue_bias_service_target=float(candidate_budgeted_rescue_bias_service_target),
                rescue_bias_ttl_guard_ms=float(candidate_budgeted_rescue_bias_ttl_guard_ms),
                rescue_bias_max=float(candidate_budgeted_rescue_bias_max),
            )
    else:
        prg_logits_masked, prg_valid = _masked_prg_logits(prg_actor_out, graph, ue_action_class=ue_class)
    del prg_valid
    prg_dist = Categorical(logits=prg_logits_masked)
    prg_class = prg_dist.sample()
    prg_entropy = prg_dist.entropy().mean()
    budgeted_prg_logp: torch.Tensor | None = None
    budgeted_prg_entropy: torch.Tensor | None = None
    blank_budget_repair_count = 0
    headroom_rank_mode = str(candidate_argmax_decode_headroom_rank_mode).lower()

    if use_candidate_prg:
        assert packed is not None
        blank_class_for_rank = int(packed.blank_class_index)
        candidate_rank_scores = prg_logits_masked[:, :blank_class_for_rank].detach()
        decoded = None
        if budget_mode != "off":
            valid, _selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(
                graph,
                packed,
                ue_class,
            )
            decoded, prg_class, budgeted_metrics = select_and_decode_budgeted_candidate_actions(
                graph,
                packed,
                valid,
                ue_class.detach().cpu(),
                prg_logits_masked,
                mode=("sample" if budget_mode == "sample" else "greedy"),
                demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
                marginal_headroom_slack_bytes=float(candidate_marginal_headroom_slack_bytes),
                hard_cap=bool(int(candidate_budgeted_hard_cap)),
                usage_penalty=float(candidate_budgeted_usage_penalty),
                overcap_penalty=float(candidate_budgeted_overcap_penalty),
                first_service_bonus=float(candidate_budgeted_first_service_bonus),
                tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                tail_bias_max=float(candidate_budgeted_tail_bias_max),
                rescue_bias_coef=float(candidate_budgeted_rescue_bias_coef),
                rescue_bias_service_target=float(candidate_budgeted_rescue_bias_service_target),
                rescue_bias_ttl_guard_ms=float(candidate_budgeted_rescue_bias_ttl_guard_ms),
                rescue_bias_max=float(candidate_budgeted_rescue_bias_max),
                completion_bonus=float(candidate_budgeted_completion_bonus),
                completion_progress_weight=float(candidate_budgeted_completion_progress_weight),
                completion_max_score=float(candidate_budgeted_completion_max_score),
                tail_quota_enable=bool(int(candidate_budgeted_tail_quota_enable)),
                tail_quota_min_score=float(candidate_budgeted_tail_quota_min_score),
                tail_quota_topk_ues=int(candidate_budgeted_tail_quota_topk_ues),
                tail_quota_prgs_per_ue=int(candidate_budgeted_tail_quota_prgs_per_ue),
                tail_quota_bonus=float(candidate_budgeted_tail_quota_bonus),
                tail_quota_order_bonus=float(candidate_budgeted_tail_quota_order_bonus),
                tail_quota_max_score=float(candidate_budgeted_tail_quota_max_score),
                tail_quota_risk_max=float(candidate_budgeted_tail_quota_risk_max),
                seq_state_weights=prg_actor_out.get("candidate_seq_state_weights"),
                seq_state_coef=float(candidate_seq_state_coef),
                seq_fail_risk_feature_offset=int(candidate_seq_fail_risk_feature_offset),
                safe_fill_max_per_cell=int(safe_fill_max_per_cell),
                safe_fill_min_quality_db=float(safe_fill_min_quality_db),
                safe_fill_max_quality_gap_db=float(safe_fill_max_quality_gap_db),
                safe_fill_max_prg_risk=float(safe_fill_max_prg_risk),
                safe_fill_min_backlog_bytes=float(safe_fill_min_backlog_bytes),
                safe_fill_max_ue_tb_err=float(safe_fill_max_ue_tb_err),
                headroom_rank_mode=headroom_rank_mode,
                candidate_rank_scores=candidate_rank_scores,
            )
            budgeted_prg_logp, budgeted_prg_entropy = _budgeted_candidate_logp_entropy(
                graph,
                packed,
                valid.to(device=device, dtype=torch.bool),
                prg_logits_masked,
                prg_class,
                mode=("sample" if budget_mode == "sample" else "greedy"),
                demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
                hard_cap=bool(int(candidate_budgeted_hard_cap)),
                usage_penalty=float(candidate_budgeted_usage_penalty),
                overcap_penalty=float(candidate_budgeted_overcap_penalty),
                first_service_bonus=float(candidate_budgeted_first_service_bonus),
                tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                tail_bias_max=float(candidate_budgeted_tail_bias_max),
                rescue_bias_coef=float(candidate_budgeted_rescue_bias_coef),
                rescue_bias_service_target=float(candidate_budgeted_rescue_bias_service_target),
                rescue_bias_ttl_guard_ms=float(candidate_budgeted_rescue_bias_ttl_guard_ms),
                rescue_bias_max=float(candidate_budgeted_rescue_bias_max),
                completion_bonus=float(candidate_budgeted_completion_bonus),
                completion_progress_weight=float(candidate_budgeted_completion_progress_weight),
                completion_max_score=float(candidate_budgeted_completion_max_score),
                seq_state_weights=prg_actor_out.get("candidate_seq_state_weights"),
                seq_state_coef=float(candidate_seq_state_coef),
                seq_fail_risk_feature_offset=int(candidate_seq_fail_risk_feature_offset),
            )
        else:
            blank_class_for_metrics = int(packed.blank_class_index)
            if tail_scores is not None:
                budgeted_metrics["candidate_budgeted_tail_bias_selected_mean"] = _candidate_direct_bias_selected_mean(
                    tail_scores,
                    prg_class,
                    blank_class_for_metrics,
                )
            if rescue_scores is not None:
                budgeted_metrics["candidate_budgeted_rescue_bias_selected_mean"] = _candidate_direct_bias_selected_mean(
                    rescue_scores,
                    prg_class,
                    blank_class_for_metrics,
                )
            prg_class, blank_budget_repair_count = apply_candidate_blank_budget(
                graph,
                packed,
                prg_logits_masked,
                prg_class,
                max_blank_per_cell=int(candidate_blank_budget_max_per_cell),
                min_decision_logit=float(candidate_blank_budget_min_decision_logit),
            )
        if decoded is None:
            decoded = decode_candidate_actions_to_type0(
                graph,
                packed,
                ue_class.detach().cpu(),
                prg_class.detach().cpu().to(torch.long),
                safe_fill_max_per_cell=int(safe_fill_max_per_cell),
                safe_fill_min_quality_db=float(safe_fill_min_quality_db),
                safe_fill_max_quality_gap_db=float(safe_fill_max_quality_gap_db),
                safe_fill_max_prg_risk=float(safe_fill_max_prg_risk),
                safe_fill_min_backlog_bytes=float(safe_fill_min_backlog_bytes),
                safe_fill_max_ue_tb_err=float(safe_fill_max_ue_tb_err),
                demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
                marginal_headroom_slack_bytes=float(candidate_marginal_headroom_slack_bytes),
                headroom_rank_mode=headroom_rank_mode,
                candidate_rank_scores=candidate_rank_scores,
            )
        blank_class = int(packed.blank_class_index)
    else:
        decoded = decode_joint_actions_to_type0(
            graph,
            ue_class.detach().cpu(),
            prg_class.detach().cpu().to(torch.long),
        )
        blank_class = int(graph.snapshot.static_meta.n_sched_ue)
    if budgeted_prg_logp is not None and budgeted_prg_entropy is not None:
        prg_logp = budgeted_prg_logp
        prg_entropy = budgeted_prg_entropy
    else:
        prg_logp = prg_dist.log_prob(prg_class).mean()
    action_logp = ue_logp + prg_logp
    action_entropy = 0.5 * (ue_entropy + prg_entropy)
    blank_ratio = float((prg_class == blank_class).float().mean().item())
    action_fill_ratio = float(decoded.prg_chosen_mask.float().mean().item())
    final_blank_ratio = float(1.0 - action_fill_ratio)
    post_decode_drop_ratio = float(max(0.0, 1.0 - blank_ratio - action_fill_ratio))
    cap_drop_count = float(decoded.cap_drop_count)
    fallback_success_count = float(decoded.fallback_success_count)
    safe_fill_count = float(decoded.safe_fill_count)
    missed_safe_opportunity_count = float(decoded.missed_safe_opportunity_count)
    final_blank_count = float(decoded.final_blank_count)
    post_decode_policy_blank_count = float(decoded.post_decode_policy_blank_count)
    post_decode_no_valid_candidate_count = float(decoded.post_decode_no_valid_candidate_count)
    post_decode_demand_cap_count = float(decoded.post_decode_demand_cap_count)
    post_decode_no_slot_count = float(decoded.post_decode_no_slot_count)
    post_decode_other_drop_count = float(decoded.post_decode_other_drop_count)
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
        "safe_fill_count": safe_fill_count,
        "missed_safe_opportunity_count": missed_safe_opportunity_count,
        "blank_budget_repair_count": float(blank_budget_repair_count),
        "final_blank_count": final_blank_count,
        "post_decode_policy_blank_count": post_decode_policy_blank_count,
        "post_decode_no_valid_candidate_count": post_decode_no_valid_candidate_count,
        "post_decode_demand_cap_count": post_decode_demand_cap_count,
        "post_decode_no_slot_count": post_decode_no_slot_count,
        "post_decode_other_drop_count": post_decode_other_drop_count,
        "action_fill_ratio": action_fill_ratio,
        "action_unique_ue_count": action_unique_ue_count,
        "candidate_unique_ue_count": candidate_unique_ue_count,
        "demand_active_ue_count": demand_active_ue_count,
        "candidate_coverage_ratio": candidate_coverage_ratio,
        "coverage_repair_count": coverage_repair_count,
        "candidate_budgeted_changed_count": float(budgeted_metrics.get("candidate_budgeted_changed_count", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_no_headroom_count": float(budgeted_metrics.get("candidate_budgeted_no_headroom_count", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_active_ue_count": float(budgeted_metrics.get("candidate_budgeted_active_ue_count", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_tail_bias_coef": float(budgeted_metrics.get("candidate_budgeted_tail_bias_coef", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_tail_bias_candidate_mean": float(budgeted_metrics.get("candidate_budgeted_tail_bias_candidate_mean", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_tail_bias_selected_mean": float(budgeted_metrics.get("candidate_budgeted_tail_bias_selected_mean", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_tail_bias_active_count": float(budgeted_metrics.get("candidate_budgeted_tail_bias_active_count", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_rescue_bias_coef": float(budgeted_metrics.get("candidate_budgeted_rescue_bias_coef", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_rescue_bias_candidate_mean": float(budgeted_metrics.get("candidate_budgeted_rescue_bias_candidate_mean", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_rescue_bias_selected_mean": float(budgeted_metrics.get("candidate_budgeted_rescue_bias_selected_mean", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_rescue_bias_active_count": float(budgeted_metrics.get("candidate_budgeted_rescue_bias_active_count", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_completion_bonus": float(budgeted_metrics.get("candidate_budgeted_completion_bonus", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_completion_candidate_mean": float(budgeted_metrics.get("candidate_budgeted_completion_candidate_mean", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_completion_selected_mean": float(budgeted_metrics.get("candidate_budgeted_completion_selected_mean", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_completion_active_count": float(budgeted_metrics.get("candidate_budgeted_completion_active_count", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_tail_quota_enabled": float(budgeted_metrics.get("candidate_budgeted_tail_quota_enabled", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_tail_quota_candidate_mean": float(budgeted_metrics.get("candidate_budgeted_tail_quota_candidate_mean", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_tail_quota_active_ue_count": float(budgeted_metrics.get("candidate_budgeted_tail_quota_active_ue_count", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_tail_quota_selected_count": float(budgeted_metrics.get("candidate_budgeted_tail_quota_selected_count", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_tail_quota_selected_mean": float(budgeted_metrics.get("candidate_budgeted_tail_quota_selected_mean", 0.0)) if use_candidate_prg else 0.0,
        "candidate_budgeted_tail_quota_served_ue_count": float(budgeted_metrics.get("candidate_budgeted_tail_quota_served_ue_count", 0.0)) if use_candidate_prg else 0.0,
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


def _configure_actor_trainable_params(
    actor: StageBHGraphPolicy,
    *,
    candidate_seq_train_only: bool = False,
) -> tuple[List[nn.Parameter], int, int]:
    """Return PPO-trainable actor params after applying optional seq-only freeze."""
    total_count = sum(p.numel() for p in actor.parameters())
    if not bool(candidate_seq_train_only):
        for param in actor.parameters():
            param.requires_grad_(True)
        params = [p for p in actor.parameters() if p.requires_grad]
        return params, sum(p.numel() for p in params), total_count

    for param in actor.parameters():
        param.requires_grad_(False)
    seq_head = getattr(getattr(actor, "action_head", None), "candidate_seq_state_score", None)
    if seq_head is None:
        raise ValueError("--candidate-seq-train-only requires --candidate-seq-state-dim > 0")
    for param in seq_head.parameters():
        param.requires_grad_(True)
    params = [p for p in actor.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in params)
    if trainable_count <= 0:
        raise ValueError("--candidate-seq-train-only found no trainable sequential-state parameters")
    return params, trainable_count, total_count


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
                out = actor.forward_graph(graph, slot_ue_class_for_prg=teacher_target_ue)
                ue_logits_raw = out["ue_logits"].unsqueeze(0)

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
                prg_soft_target = None

                if "candidate_prg_logits" in out:
                    packed = pack_prg_candidates(
                        graph,
                        max_candidates=max(0, int(out["candidate_prg_logits"].shape[-1]) - 1),
                        fixed_width=True,
                    )
                    teacher_target_ue, teacher_target_prg = joint_candidate_targets_from_type0_actions(
                        graph,
                        packed,
                        action_ue_select=teacher_out["action_ue_select"].to(device=device, dtype=torch.long),
                        action_prg_alloc=teacher_out["action_prg_alloc"].to(device=device, dtype=torch.long),
                    )
                    teacher_target_ue = teacher_target_ue.to(device=device, dtype=torch.long)
                    teacher_target_prg = teacher_target_prg.to(device=device, dtype=torch.long)
                    ue_target, _ue_bad = sanitize_targets(teacher_target_ue.unsqueeze(0), ue_valid)
                    prg_logits, prg_valid = _masked_candidate_prg_logits(
                        out,
                        graph,
                        ue_action_class=teacher_target_ue,
                    )
                    prg_logits = prg_logits.unsqueeze(0)
                    prg_valid = prg_valid.unsqueeze(0)
                    if int(args.teacher_soft_target) == 1:
                        prg_soft_target = candidate_prg_soft_targets_from_type0_actions(
                            graph,
                            packed,
                            prg_valid.squeeze(0),
                            action_ue_select=teacher_out["action_ue_select"].to(device=device, dtype=torch.long),
                            action_prg_alloc=teacher_out["action_prg_alloc"].to(device=device, dtype=torch.long),
                            temperature=float(args.teacher_soft_temperature),
                            teacher_hit_bonus=float(args.teacher_soft_hit_bonus),
                            teacher_blank_bonus=float(args.teacher_soft_blank_bonus),
                            blank_safe_penalty=float(args.teacher_soft_blank_safe_penalty),
                            blank_no_safe_bonus=float(args.teacher_soft_blank_no_safe_bonus),
                            min_quality_db=float(args.safe_fill_min_quality_db),
                            max_quality_gap_db=float(args.safe_fill_max_quality_gap_db),
                            max_prg_risk=float(args.safe_fill_max_prg_risk),
                            min_backlog_bytes=float(args.safe_fill_min_backlog_bytes),
                            max_ue_tb_err=float(args.safe_fill_max_ue_tb_err),
                        ).unsqueeze(0)
                else:
                    prg_logits_raw = out["prg_logits"].unsqueeze(0)
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
                if prg_soft_target is not None:
                    prg_loss = _soft_target_cross_entropy(prg_logits, prg_soft_target, prg_valid)
                else:
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
    rollout_warmup_steps: int,
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
    expiry_reward_tracker: ExpiryRewardWindowTracker | None = None,
    slot_repair_max_per_cell: int = 0,
    slot_repair_min_backlog_bytes: float = 1.0,
) -> tuple[List[RolloutStep], Dict[str, float], PerUeDiagnosticsTracker, ExpiryRewardWindowTracker]:
    actor.eval()
    critic.eval()

    obs, _reward0, info, episode_seed, topology_seed, resumed_stream = runner.begin_rollout(episode_idx)
    if not resumed_stream:
        snapshot_builder.reset()
    graph = _graph_to_device(edge_generator.build_graph(snapshot_builder.build(obs)), device)
    cached_actor_out = None
    cached_value = None
    steps: List[RolloutStep] = []
    warmup_steps_requested = max(0, int(rollout_warmup_steps))
    target_rollout_steps = max(1, int(rollout_steps))
    total_env_steps = warmup_steps_requested + target_rollout_steps
    warmup_steps_completed = 0
    if per_ue_tracker is None or int(per_ue_tracker.n_ue) != int(graph.snapshot.static_meta.n_ue):
        per_ue_tracker = PerUeDiagnosticsTracker(graph.snapshot.static_meta.n_ue)
    if expiry_reward_tracker is None:
        expiry_reward_tracker = ExpiryRewardWindowTracker(
            window_steps=int(reward_cfg.get("expiry_reward_window_steps", 0)),
            offered_bytes_per_tti=float(reward_cfg.get("expiry_reward_offered_bytes_per_tti", 0.0)),
        )
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

    for step_idx in range(total_env_steps):
        is_warmup_step = step_idx < warmup_steps_requested
        graph = _append_packet_tail_features(
            graph,
            per_ue_tracker,
            enabled=bool(int(getattr(runner.args, "candidate_packet_tail_feature_enable", 0))),
            rate_target_mbps=float(getattr(runner.args, "candidate_packet_tail_rate_target_mbps", 2.0)),
            delay_scale_ms=float(getattr(runner.args, "candidate_packet_tail_delay_scale_ms", 10.0)),
        )
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
                safe_fill_max_per_cell=int(getattr(runner.args, "safe_fill_max_per_cell", 0)),
                safe_fill_min_quality_db=float(getattr(runner.args, "safe_fill_min_quality_db", -2.0)),
                safe_fill_max_quality_gap_db=float(getattr(runner.args, "safe_fill_max_quality_gap_db", 8.0)),
                safe_fill_max_prg_risk=float(getattr(runner.args, "safe_fill_max_prg_risk", 0.70)),
                safe_fill_min_backlog_bytes=float(getattr(runner.args, "safe_fill_min_backlog_bytes", 1.0)),
                safe_fill_max_ue_tb_err=float(getattr(runner.args, "safe_fill_max_ue_tb_err", 0.45)),
                candidate_blank_budget_max_per_cell=int(
                    getattr(runner.args, "candidate_blank_budget_max_per_cell", -1)
                ),
                candidate_blank_budget_min_decision_logit=float(
                    getattr(runner.args, "candidate_blank_budget_min_decision_logit", -1.0e9)
                ),
                candidate_decode_demand_slack_bytes=float(
                    getattr(runner.args, "candidate_decode_demand_slack_bytes", 3000.0)
                ),
                candidate_marginal_headroom_slack_bytes=float(
                    getattr(runner.args, "candidate_marginal_headroom_slack_bytes", -1.0)
                ),
                candidate_argmax_decode_headroom_rank_mode=str(
                    getattr(runner.args, "candidate_argmax_decode_headroom_rank_mode", "stable")
                ),
                candidate_budgeted_select_mode=str(getattr(runner.args, "candidate_budgeted_select_mode", "off")),
                candidate_budgeted_hard_cap=int(getattr(runner.args, "candidate_budgeted_hard_cap", 1)),
                candidate_budgeted_usage_penalty=float(
                    getattr(runner.args, "candidate_budgeted_usage_penalty", 0.35)
                ),
                candidate_budgeted_overcap_penalty=float(
                    getattr(runner.args, "candidate_budgeted_overcap_penalty", 8.0)
                ),
                candidate_budgeted_first_service_bonus=float(
                    getattr(runner.args, "candidate_budgeted_first_service_bonus", 0.0)
                ),
                candidate_budgeted_tail_bias_coef=float(
                    getattr(runner.args, "candidate_budgeted_tail_bias_coef", 0.0)
                ),
                candidate_budgeted_tail_bias_service_target=float(
                    getattr(runner.args, "candidate_budgeted_tail_bias_service_target", 0.75)
                ),
                candidate_budgeted_tail_bias_ttl_guard_ms=float(
                    getattr(runner.args, "candidate_budgeted_tail_bias_ttl_guard_ms", 20.0)
                ),
                candidate_budgeted_tail_bias_max=float(
                    getattr(runner.args, "candidate_budgeted_tail_bias_max", 1.5)
                ),
                candidate_budgeted_rescue_bias_coef=float(
                    getattr(runner.args, "candidate_budgeted_rescue_bias_coef", 0.0)
                ),
                candidate_budgeted_rescue_bias_service_target=float(
                    getattr(runner.args, "candidate_budgeted_rescue_bias_service_target", 0.80)
                ),
                candidate_budgeted_rescue_bias_ttl_guard_ms=float(
                    getattr(runner.args, "candidate_budgeted_rescue_bias_ttl_guard_ms", 60.0)
                ),
                candidate_budgeted_rescue_bias_max=float(
                    getattr(runner.args, "candidate_budgeted_rescue_bias_max", 1.5)
                ),
                candidate_budgeted_completion_bonus=float(
                    getattr(runner.args, "candidate_budgeted_completion_bonus", 0.0)
                ),
                candidate_budgeted_completion_progress_weight=float(
                    getattr(runner.args, "candidate_budgeted_completion_progress_weight", 0.25)
                ),
                candidate_budgeted_completion_max_score=float(
                    getattr(runner.args, "candidate_budgeted_completion_max_score", 1.5)
                ),
                candidate_budgeted_tail_quota_enable=int(
                    getattr(runner.args, "candidate_budgeted_tail_quota_enable", 0)
                ),
                candidate_budgeted_tail_quota_min_score=float(
                    getattr(runner.args, "candidate_budgeted_tail_quota_min_score", 0.55)
                ),
                candidate_budgeted_tail_quota_topk_ues=int(
                    getattr(runner.args, "candidate_budgeted_tail_quota_topk_ues", 6)
                ),
                candidate_budgeted_tail_quota_prgs_per_ue=int(
                    getattr(runner.args, "candidate_budgeted_tail_quota_prgs_per_ue", 1)
                ),
                candidate_budgeted_tail_quota_bonus=float(
                    getattr(runner.args, "candidate_budgeted_tail_quota_bonus", 4.0)
                ),
                candidate_budgeted_tail_quota_order_bonus=float(
                    getattr(runner.args, "candidate_budgeted_tail_quota_order_bonus", 2.0)
                ),
                candidate_budgeted_tail_quota_max_score=float(
                    getattr(runner.args, "candidate_budgeted_tail_quota_max_score", 4.0)
                ),
                candidate_budgeted_tail_quota_risk_max=float(
                    getattr(runner.args, "candidate_budgeted_tail_quota_risk_max", 0.70)
                ),
                candidate_seq_state_coef=float(getattr(runner.args, "candidate_seq_state_coef", 1.0)),
                candidate_seq_fail_risk_feature_offset=int(
                    getattr(runner.args, "candidate_seq_fail_risk_feature_offset", -1)
                ),
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
        visible_service_rate_cpu = (
            np.clip(1.0 - recent_goodput_deficit_norm_cpu, 0.0, 1.0)
            if recent_goodput_deficit_norm_cpu is not None
            else recent_scheduled_ratio_cpu
        )
        tail_before_stats = _tail_risk_potential_from_arrays(
            demand_mask=demand_mask_cpu,
            backlog_bytes=backlog_bytes_cpu,
            hol_delay_ms=hol_delay_ms_cpu,
            ttl_slack_ms=ttl_slack_ms_cpu,
            service_rate=visible_service_rate_cpu,
            goodput_deficit_norm=recent_goodput_deficit_norm_cpu,
            streak=per_ue_tracker.no_goodput_streak,
            service_target=float(reward_cfg.get("tail_service_target", reward_cfg.get("service_rate_target", 0.75))),
            ttl_guard_ms=float(reward_cfg.get("tail_ttl_guard_ms", 20.0)),
            topk=int(reward_cfg.get("tail_potential_topk", 8)),
        )
        expiry_before_stats = _expiry_risk_potential_from_arrays(
            demand_mask=demand_mask_cpu,
            backlog_bytes=backlog_bytes_cpu,
            hol_delay_ms=hol_delay_ms_cpu,
            ttl_slack_ms=ttl_slack_ms_cpu,
            service_rate=visible_service_rate_cpu,
            goodput_deficit_norm=recent_goodput_deficit_norm_cpu,
            streak=per_ue_tracker.no_goodput_streak,
            ttl_guard_ms=float(reward_cfg.get("expiry_ttl_guard_ms", 40.0)),
            topk=int(reward_cfg.get("expiry_potential_topk", 8)),
        )
        rescue_before_score = _ue_rescue_scores_from_arrays(
            demand_mask=demand_mask_cpu,
            backlog_bytes=backlog_bytes_cpu,
            hol_delay_ms=hol_delay_ms_cpu,
            ttl_slack_ms=ttl_slack_ms_cpu,
            service_rate=visible_service_rate_cpu,
            goodput_deficit_norm=recent_goodput_deficit_norm_cpu,
            streak=per_ue_tracker.no_goodput_streak,
            service_target=float(reward_cfg.get("rescue_service_target", 0.80)),
            ttl_guard_ms=float(reward_cfg.get("rescue_ttl_guard_ms", reward_cfg.get("expiry_ttl_guard_ms", 60.0))),
            max_score=float(reward_cfg.get("rescue_max_score", 1.5)),
        )
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
        expiry_reward_rate = expiry_reward_tracker.update(reward_metrics)
        reward_metrics_for_reward = dict(reward_metrics)
        reward_metrics_for_reward["expiry_drop_rate"] = float(expiry_reward_rate)
        rescue_credit_stats = _rescue_credit_from_scores(
            rescue_score=rescue_before_score,
            actual_goodput_bytes=next_info.get("actual_ue_goodput_bytes"),
            topk=int(reward_cfg.get("rescue_topk", 8)),
            goodput_bytes_scale=float(reward_cfg.get("rescue_goodput_bytes_scale", 3000.0)),
        )
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
        actual_packet_available = bool(next_info.get("actual_ue_packet_diagnostics_available", False))
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
            actual_packet_available=actual_packet_available,
            actual_packet_delivered_packets=next_info.get("actual_ue_packet_delivered_packets"),
            actual_packet_pending_packets=next_info.get("actual_ue_packet_pending_packets"),
            actual_packet_delivered_bits=next_info.get("actual_ue_packet_delivered_bits"),
            actual_packet_system_time_ms=next_info.get("actual_ue_packet_system_time_ms"),
            actual_packet_service_rate_mbps=next_info.get("actual_ue_packet_service_rate_mbps"),
            actual_packet_service_rate_per_packet_mean_mbps=next_info.get(
                "actual_ue_packet_service_rate_per_packet_mean_mbps"
            ),
        )
        per_ue_summary = per_ue_tracker.summary()
        tail_after_stats = _tail_risk_potential_from_arrays(
            demand_mask=per_ue_tracker.last_demand,
            backlog_bytes=per_ue_tracker.last_backlog_bytes,
            hol_delay_ms=per_ue_tracker.last_hol_delay_ms,
            ttl_slack_ms=per_ue_tracker.last_ttl_slack_ms,
            service_rate=per_ue_tracker.recent100_goodput_service_rate(),
            goodput_deficit_norm=per_ue_tracker.last_recent_goodput_deficit_norm,
            streak=per_ue_tracker.no_goodput_streak,
            service_target=float(reward_cfg.get("tail_service_target", reward_cfg.get("service_rate_target", 0.75))),
            ttl_guard_ms=float(reward_cfg.get("tail_ttl_guard_ms", 20.0)),
            topk=int(reward_cfg.get("tail_potential_topk", 8)),
        )
        expiry_after_stats = _expiry_risk_potential_from_arrays(
            demand_mask=per_ue_tracker.last_demand,
            backlog_bytes=per_ue_tracker.last_backlog_bytes,
            hol_delay_ms=per_ue_tracker.last_hol_delay_ms,
            ttl_slack_ms=per_ue_tracker.last_ttl_slack_ms,
            service_rate=per_ue_tracker.recent100_goodput_service_rate(),
            goodput_deficit_norm=per_ue_tracker.last_recent_goodput_deficit_norm,
            streak=per_ue_tracker.no_goodput_streak,
            ttl_guard_ms=float(reward_cfg.get("expiry_ttl_guard_ms", 40.0)),
            topk=int(reward_cfg.get("expiry_potential_topk", 8)),
        )
        candidate_decode_goodput_target = None
        candidate_decode_goodput_mask = None
        if (
            str(getattr(runner.args, "action_head_mode", "")) == "candidate"
            and float(getattr(runner.args, "candidate_decode_goodput_coef", 0.0)) > 0.0
            and actual_available
        ):
            try:
                max_candidates = _resolve_prg_candidate_count(runner.args)
                packed = pack_prg_candidates(graph, max_candidates=max_candidates, fixed_width=True)
                valid, _selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(
                    graph,
                    packed,
                    sampled["ue_action_class"].to(device=packed.ue_ids.device, dtype=torch.long),
                )
                candidate_decode_goodput_target, candidate_decode_goodput_mask = (
                    candidate_prg_decode_goodput_targets(
                        graph,
                        packed,
                        valid,
                        sampled_prg_action_class=sampled["prg_action_class"],
                        decoded_chosen_ue_per_prg=sampled["chosen_ue_per_prg"],
                        actual_ue_goodput_bytes=next_info.get("actual_ue_goodput_bytes"),
                        actual_ue_prg_count=next_info.get("actual_ue_prg_count"),
                        actual_ue_tb_tx_count=next_info.get("actual_ue_tb_tx_count"),
                        actual_ue_tb_err_count=next_info.get("actual_ue_tb_err_count"),
                        target_bytes=float(getattr(runner.args, "candidate_decode_goodput_target_bytes", 3000.0)),
                        max_target=float(getattr(runner.args, "candidate_decode_goodput_max_target", 1.0)),
                    )
                )
                candidate_decode_goodput_target = candidate_decode_goodput_target.detach().cpu()
                candidate_decode_goodput_mask = candidate_decode_goodput_mask.detach().cpu()
            except (TypeError, ValueError, KeyError) as exc:
                print(f"[online-ppo] WARN decode-goodput target build failed: {exc}")
        packet_completed_count = float(reward_metrics.get("packet_completed_count", 0.0))
        packet_system_time_ms = float(reward_metrics.get("packet_system_time_ms", 0.0))
        packet_delay_mean_ms = float(reward_metrics.get("packet_delay_mean_ms", 0.0))
        packet_effective_service_rate_mbps = float(
            reward_metrics.get("packet_effective_service_rate_mbps", 0.0)
        )
        packet_effective_service_rate_per_packet_mean_mbps = float(
            reward_metrics.get("packet_effective_service_rate_per_packet_mean_mbps", 0.0)
        )
        packet_rate_bonus = 0.0
        packet_per_packet_rate_bonus = 0.0
        packet_per_packet_rate_coef = 0.0
        packet_completion_bonus = 0.0
        packet_delay_penalty = 0.0
        packet_delay_coef = 0.0
        ue_macro_rate_proxy_bonus = 0.0
        ue_macro_rate_proxy_coef = 0.0
        ue_p10_rate_proxy_gap_penalty = 0.0
        ue_p10_rate_proxy_gap_coef = 0.0
        ue_macro_packet_rate_bonus = 0.0
        ue_macro_packet_rate_coef = 0.0
        ue_p10_packet_rate_gap_penalty = 0.0
        ue_p10_packet_rate_gap_coef = 0.0
        aged_backlog_penalty = 0.0
        aged_backlog_coef = 0.0
        util_floor_gate = 0.0
        util_floor_gap = 0.0
        util_floor_penalty = 0.0
        util_floor_coef = 0.0
        missed_safe_opportunity_penalty = 0.0
        missed_safe_opportunity_coef = 0.0
        tb_err_gap = 0.0
        tb_err_over_target_coef = 0.0
        tb_err_over_target_penalty = 0.0
        tb_err_goodput_gate_coef = 0.0
        tb_err_goodput_gate_penalty = 0.0
        tail_potential_before = float(tail_before_stats["tail_potential"])
        tail_potential_after = float(tail_after_stats["tail_potential"])
        tail_potential_delta = tail_potential_before - tail_potential_after
        tail_potential_topk_mean = float(tail_after_stats["tail_topk_mean"])
        tail_potential_max = float(tail_after_stats["tail_max"])
        expiry_potential_before = float(expiry_before_stats["expiry_potential"])
        expiry_potential_after = float(expiry_after_stats["expiry_potential"])
        expiry_potential_delta = expiry_potential_before - expiry_potential_after
        expiry_potential_topk_mean = float(expiry_after_stats["expiry_topk_mean"])
        expiry_potential_max = float(expiry_after_stats["expiry_max"])
        rescue_credit = float(rescue_credit_stats["rescue_credit"])
        rescue_miss = float(rescue_credit_stats["rescue_miss"])
        rescue_risk_topk_mean = float(rescue_credit_stats["rescue_risk_topk_mean"])
        rescue_served_ratio = float(rescue_credit_stats["rescue_served_ratio"])
        if reward_mode == "stageb_v1":
            shaped = compute_stageb_v1_reward(
                raw_env_reward=float(raw_env_reward),
                reward_metrics=reward_metrics_for_reward,
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
                reward_metrics=reward_metrics_for_reward,
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
                packet_per_packet_rate_weight=float(reward_cfg.get("packet_per_packet_rate_weight", 0.0)),
                packet_completion_weight=float(reward_cfg.get("packet_completion_weight", 0.0)),
                packet_delay_weight=float(reward_cfg.get("packet_delay_weight", 0.0)),
                packet_rate_scale_mbps=float(reward_cfg.get("packet_rate_scale_mbps", 10.0)),
                packet_per_packet_rate_scale_mbps=float(
                    reward_cfg.get("packet_per_packet_rate_scale_mbps", 24.0)
                ),
                packet_completion_scale=float(reward_cfg.get("packet_completion_scale", 36.0)),
                packet_delay_scale_ms=float(reward_cfg.get("packet_delay_scale_ms", 10.0)),
                ue_macro_rate_proxy_weight=float(reward_cfg.get("ue_macro_rate_proxy_weight", 0.0)),
                ue_p10_rate_proxy_gap_weight=float(
                    reward_cfg.get("ue_p10_rate_proxy_gap_weight", 0.0)
                ),
                ue_rate_proxy_scale_mbps=float(reward_cfg.get("ue_rate_proxy_scale_mbps", 8.0)),
                ue_p10_rate_proxy_target_mbps=float(
                    reward_cfg.get("ue_p10_rate_proxy_target_mbps", 0.0)
                ),
                ue_macro_packet_rate_weight=float(
                    reward_cfg.get("ue_macro_packet_rate_weight", 0.0)
                ),
                ue_p10_packet_rate_gap_weight=float(
                    reward_cfg.get("ue_p10_packet_rate_gap_weight", 0.0)
                ),
                ue_packet_rate_scale_mbps=float(
                    reward_cfg.get("ue_packet_rate_scale_mbps", 8.0)
                ),
                ue_p10_packet_rate_target_mbps=float(
                    reward_cfg.get("ue_p10_packet_rate_target_mbps", 0.0)
                ),
                ue_macro_rate_proxy_mbps=float(per_ue_summary["ue_macro_rate_proxy_mbps"]),
                ue_p10_rate_proxy_mbps=float(per_ue_summary["ue_p10_rate_proxy_mbps"]),
                ue_macro_packet_rate_mbps=float(per_ue_summary["ue_macro_packet_rate_mbps"]),
                ue_p10_packet_rate_mbps=float(per_ue_summary["ue_p10_packet_rate_mbps"]),
                aged_backlog_weight=float(reward_cfg.get("aged_backlog_weight", 0.0)),
                aged_backlog_bytes=float(per_ue_summary["recent100_debt_score_p90_bytes"]),
                aged_backlog_streak=float(per_ue_summary["no_service_streak_p90"]),
                unserved_ue_weight=float(reward_cfg.get("unserved_ue_weight", 0.0)),
                total_ue_count=float(per_ue_tracker.n_ue),
                service_gap_weight=float(reward_cfg.get("service_gap_weight", 0.0)),
                service_rate_p10=float(per_ue_summary["recent100_service_rate_p10"]),
                service_rate_target=float(reward_cfg.get("service_rate_target", 0.0)),
                util_floor_weight=float(reward_cfg.get("util_floor_weight", 0.0)),
                util_floor_target=float(reward_cfg.get("util_floor_target", 0.0)),
                util_floor_tb_err_guard=float(reward_cfg.get("util_floor_tb_err_guard", 1.0)),
                util_floor_tb_err_softness=float(reward_cfg.get("util_floor_tb_err_softness", 0.0)),
                missed_safe_opportunity_count=float(sampled.get("missed_safe_opportunity_count", 0.0)),
                missed_safe_opportunity_weight=float(reward_cfg.get("missed_safe_opportunity_weight", 0.0)),
                missed_safe_opportunity_scale=float(reward_cfg.get("missed_safe_opportunity_scale", 51.0)),
                prg_efficiency_weight=float(reward_cfg.get("prg_efficiency_weight", -1.0)),
                harmful_packing_weight=float(reward_cfg.get("harmful_packing_weight", -1.0)),
                tb_err_target=float(reward_cfg.get("tb_err_target", -1.0)),
                tb_err_over_target_weight=float(reward_cfg.get("tb_err_over_target_weight", 0.0)),
                tb_err_goodput_gate_weight=float(reward_cfg.get("tb_err_goodput_gate_weight", 0.0)),
                tail_potential_before=tail_potential_before,
                tail_potential_after=tail_potential_after,
                tail_delta_weight=float(reward_cfg.get("tail_delta_weight", 0.0)),
                tail_pressure_weight=float(reward_cfg.get("tail_pressure_weight", 0.0)),
                tail_delta_clip=float(reward_cfg.get("tail_delta_clip", 1.0)),
                tail_potential_discount=float(reward_cfg.get("tail_potential_discount", 1.0)),
                expiry_potential_before=expiry_potential_before,
                expiry_potential_after=expiry_potential_after,
                expiry_delta_weight=float(reward_cfg.get("expiry_delta_weight", 0.0)),
                expiry_pressure_weight=float(reward_cfg.get("expiry_pressure_weight", 0.0)),
                expiry_delta_clip=float(reward_cfg.get("expiry_delta_clip", 1.0)),
                expiry_potential_discount=float(reward_cfg.get("expiry_potential_discount", 1.0)),
                rescue_credit=rescue_credit,
                rescue_miss=rescue_miss,
                rescue_credit_weight=float(reward_cfg.get("rescue_credit_weight", 0.0)),
                rescue_miss_weight=float(reward_cfg.get("rescue_miss_weight", 0.0)),
            )
            reward_raw = float(shaped["reward_shaped"])
            conflict_mean = _prg_feature_mean(graph, 6)
            ici_mean = _prg_feature_mean(graph, 7)
            urgent_backlog_penalty = float(shaped["reward_backlog_debt_penalty"])
            backlog_debt_coef = float(shaped["reward_backlog_debt_coef"])
            packet_rate_bonus = float(shaped["reward_packet_rate_bonus"])
            packet_per_packet_rate_bonus = float(shaped["reward_packet_per_packet_rate_bonus"])
            packet_per_packet_rate_coef = float(shaped["reward_packet_per_packet_rate_coef"])
            packet_completion_bonus = float(shaped["reward_packet_completion_bonus"])
            packet_delay_penalty = float(shaped["reward_packet_delay_penalty"])
            packet_delay_coef = float(shaped["reward_packet_delay_coef"])
            ue_macro_rate_proxy_bonus = float(shaped["reward_ue_macro_rate_proxy_bonus"])
            ue_macro_rate_proxy_coef = float(shaped["reward_ue_macro_rate_proxy_coef"])
            ue_p10_rate_proxy_gap_penalty = float(shaped["reward_ue_p10_rate_proxy_gap_penalty"])
            ue_p10_rate_proxy_gap_coef = float(shaped["reward_ue_p10_rate_proxy_gap_coef"])
            ue_macro_packet_rate_bonus = float(shaped["reward_ue_macro_packet_rate_bonus"])
            ue_macro_packet_rate_coef = float(shaped["reward_ue_macro_packet_rate_coef"])
            ue_p10_packet_rate_gap_penalty = float(shaped["reward_ue_p10_packet_rate_gap_penalty"])
            ue_p10_packet_rate_gap_coef = float(shaped["reward_ue_p10_packet_rate_gap_coef"])
            aged_backlog_penalty = float(shaped["reward_aged_backlog_penalty"])
            aged_backlog_coef = float(shaped["reward_aged_backlog_coef"])
            active_prg_goodput_efficiency = float(shaped["reward_active_prg_goodput_efficiency"])
            harmful_packing_penalty = float(shaped["reward_harmful_packing_penalty"])
            util_floor_gate = float(shaped["reward_util_floor_gate"])
            util_floor_gap = float(shaped["reward_util_floor_gap"])
            util_floor_penalty = float(shaped["reward_util_floor_penalty"])
            util_floor_coef = float(shaped["reward_util_floor_coef"])
            missed_safe_opportunity_penalty = float(shaped["reward_missed_safe_opportunity_penalty"])
            missed_safe_opportunity_coef = float(shaped["reward_missed_safe_opportunity_coef"])
            tb_err_gap = float(shaped["reward_tb_err_gap"])
            tb_err_over_target_coef = float(shaped["reward_tb_err_over_target_coef"])
            tb_err_over_target_penalty = float(shaped["reward_tb_err_over_target_penalty"])
            tb_err_goodput_gate_coef = float(shaped["reward_tb_err_goodput_gate_coef"])
            tb_err_goodput_gate_penalty = float(shaped["reward_tb_err_goodput_gate_penalty"])
            tail_potential_delta = float(shaped["reward_tail_delta"])
            expiry_potential_delta = float(shaped["reward_expiry_delta"])
        elif reward_mode == "stageb_tailguard_simple":
            shaped = compute_stageb_tailguard_simple_reward(
                raw_env_reward=float(raw_env_reward),
                reward_metrics=reward_metrics_for_reward,
                goodput_weight=float(reward_cfg["goodput_weight"]),
                tb_err_weight=float(reward_cfg["tb_err_weight"]),
                expiry_weight=float(reward_cfg["expiry_weight"]),
                actual_unserved_backlog_bytes=float(per_ue_summary["actual_unserved_backlog_bytes"]),
                actual_unserved_demand_ue_count=float(per_ue_summary["actual_unserved_demand_ue_count"]),
                aged_backlog_bytes=float(per_ue_summary["recent100_debt_score_p90_bytes"]),
                aged_backlog_streak=float(per_ue_summary["no_service_streak_p90"]),
                total_ue_count=float(per_ue_tracker.n_ue),
                service_gap_weight=float(reward_cfg.get("service_gap_weight", 1.25)),
                service_rate_p10=float(per_ue_summary["recent100_service_rate_p10"]),
                service_rate_target=float(reward_cfg.get("service_rate_target", 0.75)),
            )
            reward_raw = float(shaped["reward_shaped"])
            conflict_mean = _prg_feature_mean(graph, 6)
            ici_mean = _prg_feature_mean(graph, 7)
            urgent_backlog_penalty = float(shaped["reward_tail_pressure"])
            backlog_debt_coef = float(shaped["reward_tail_coef"])
            packet_rate_bonus = float(shaped["reward_goodput_norm"])
            packet_completion_bonus = float(shaped["reward_reliability_penalty"])
            aged_backlog_penalty = float(shaped["reward_historical_debt_pressure"])
            aged_backlog_coef = float(shaped["reward_tail_coef"])
            active_prg_goodput_efficiency = 0.0
            harmful_packing_penalty = 0.0
            util_floor_gate = 0.0
            util_floor_gap = 0.0
            util_floor_penalty = 0.0
            util_floor_coef = 0.0
        elif reward_mode == "stageb_tailcredit_v4":
            shaped = compute_stageb_tailcredit_reward(
                raw_env_reward=float(raw_env_reward),
                reward_metrics=reward_metrics_for_reward,
                goodput_weight=float(reward_cfg["goodput_weight"]),
                tb_err_weight=float(reward_cfg["tb_err_weight"]),
                expiry_weight=float(reward_cfg["expiry_weight"]),
                tail_potential_before=tail_potential_before,
                tail_potential_after=tail_potential_after,
                tail_delta_weight=float(reward_cfg.get("tail_delta_weight", 1.0)),
                tail_pressure_weight=float(reward_cfg.get("tail_pressure_weight", 0.15)),
                tail_delta_clip=float(reward_cfg.get("tail_delta_clip", 1.0)),
                tail_potential_discount=float(reward_cfg.get("tail_potential_discount", 1.0)),
            )
            reward_raw = float(shaped["reward_shaped"])
            conflict_mean = _prg_feature_mean(graph, 6)
            ici_mean = _prg_feature_mean(graph, 7)
            urgent_backlog_penalty = float(shaped["reward_tail_potential_after"])
            backlog_debt_coef = float(shaped["reward_tail_pressure_coef"])
            packet_rate_bonus = float(shaped["reward_goodput_norm"])
            packet_completion_bonus = float(shaped["reward_reliability_penalty"])
            aged_backlog_penalty = float(shaped["reward_tail_delta"])
            aged_backlog_coef = float(shaped["reward_tail_delta_coef"])
            tail_potential_delta = float(shaped["reward_tail_delta"])
            active_prg_goodput_efficiency = 0.0
            harmful_packing_penalty = 0.0
            util_floor_gate = 0.0
            util_floor_gap = 0.0
            util_floor_penalty = 0.0
            util_floor_coef = 0.0
        elif reward_mode in ("stageb_tailcredit_expiry_v5", "stageb_tailcredit_expiry_rescue_v6"):
            shaped = compute_stageb_tailcredit_expiry_reward(
                raw_env_reward=float(raw_env_reward),
                reward_metrics=reward_metrics_for_reward,
                goodput_weight=float(reward_cfg["goodput_weight"]),
                tb_err_weight=float(reward_cfg["tb_err_weight"]),
                expiry_weight=float(reward_cfg["expiry_weight"]),
                tail_potential_before=tail_potential_before,
                tail_potential_after=tail_potential_after,
                tail_delta_weight=float(reward_cfg.get("tail_delta_weight", 1.0)),
                tail_pressure_weight=float(reward_cfg.get("tail_pressure_weight", 0.15)),
                tail_delta_clip=float(reward_cfg.get("tail_delta_clip", 1.0)),
                tail_potential_discount=float(reward_cfg.get("tail_potential_discount", 1.0)),
                expiry_potential_before=expiry_potential_before,
                expiry_potential_after=expiry_potential_after,
                expiry_delta_weight=float(reward_cfg.get("expiry_delta_weight", 1.0)),
                expiry_pressure_weight=float(reward_cfg.get("expiry_pressure_weight", 0.25)),
                expiry_delta_clip=float(reward_cfg.get("expiry_delta_clip", 1.0)),
                expiry_potential_discount=float(reward_cfg.get("expiry_potential_discount", 1.0)),
                rescue_credit=rescue_credit if reward_mode == "stageb_tailcredit_expiry_rescue_v6" else 0.0,
                rescue_miss=rescue_miss if reward_mode == "stageb_tailcredit_expiry_rescue_v6" else 0.0,
                rescue_credit_weight=(
                    float(reward_cfg.get("rescue_credit_weight", 0.0))
                    if reward_mode == "stageb_tailcredit_expiry_rescue_v6"
                    else 0.0
                ),
                rescue_miss_weight=(
                    float(reward_cfg.get("rescue_miss_weight", 0.0))
                    if reward_mode == "stageb_tailcredit_expiry_rescue_v6"
                    else 0.0
                ),
            )
            reward_raw = float(shaped["reward_shaped"])
            conflict_mean = _prg_feature_mean(graph, 6)
            ici_mean = _prg_feature_mean(graph, 7)
            urgent_backlog_penalty = float(shaped["reward_expiry_potential_after"])
            backlog_debt_coef = float(shaped["reward_expiry_pressure_coef"])
            packet_rate_bonus = float(shaped["reward_goodput_norm"])
            packet_completion_bonus = float(shaped["reward_reliability_penalty"])
            aged_backlog_penalty = float(shaped["reward_tail_delta"])
            aged_backlog_coef = float(shaped["reward_tail_delta_coef"])
            tail_potential_delta = float(shaped["reward_tail_delta"])
            expiry_potential_delta = float(shaped["reward_expiry_delta"])
            active_prg_goodput_efficiency = 0.0
            harmful_packing_penalty = 0.0
            util_floor_gate = 0.0
            util_floor_gap = 0.0
            util_floor_penalty = 0.0
            util_floor_coef = 0.0
        else:
            reward_raw = float(raw_env_reward)
            conflict_mean = _prg_feature_mean(graph, 6)
            ici_mean = _prg_feature_mean(graph, 7)
            urgent_backlog_penalty = 0.0
            backlog_debt_coef = 0.0
            active_prg_goodput_efficiency = 0.0
            harmful_packing_penalty = 0.0
        prg_expected_goodput_proxy_mbps_mean = _prg_feature_raw_mean(graph, 10)
        prg_low_sinr_hol_ttl_weight_mean = _prg_feature_raw_mean(graph, 11)
        if reward_norm is not None:
            if not is_warmup_step:
                reward_norm.update(float(reward_raw))
            reward = reward_norm.normalize(float(reward_raw)) if normalize_reward else float(reward_raw)
        else:
            reward = float(reward_raw)

        next_value = 0.0
        next_graph = None
        next_actor_out = None
        next_value_t = None
        if not done and (step_idx + 1) < total_env_steps:
            next_graph = _graph_to_device(edge_generator.build_graph(snapshot_builder.build(next_obs)), device)
            next_graph = _append_packet_tail_features(
                next_graph,
                per_ue_tracker,
                enabled=bool(int(getattr(runner.args, "candidate_packet_tail_feature_enable", 0))),
                rate_target_mbps=float(getattr(runner.args, "candidate_packet_tail_rate_target_mbps", 2.0)),
                delay_scale_ms=float(getattr(runner.args, "candidate_packet_tail_delay_scale_ms", 10.0)),
            )
            with torch.no_grad():
                next_actor_out, next_value_t = _evaluate_graph(actor, critic, next_graph, state_norm=state_norm)
                if state_norm is not None:
                    state_norm.update(_pool_actor_state(next_actor_out))
                next_value = float(next_value_t.item())

        step_record = RolloutStep(
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
                safe_fill_count=float(sampled.get("safe_fill_count", 0.0)),
                missed_safe_opportunity_count=float(sampled.get("missed_safe_opportunity_count", 0.0)),
                blank_budget_repair_count=float(sampled.get("blank_budget_repair_count", 0.0)),
                final_blank_count=float(sampled["final_blank_count"]),
                post_decode_policy_blank_count=float(sampled.get("post_decode_policy_blank_count", 0.0)),
                post_decode_no_valid_candidate_count=float(
                    sampled.get("post_decode_no_valid_candidate_count", 0.0)
                ),
                post_decode_demand_cap_count=float(sampled.get("post_decode_demand_cap_count", 0.0)),
                post_decode_no_slot_count=float(sampled.get("post_decode_no_slot_count", 0.0)),
                post_decode_other_drop_count=float(sampled.get("post_decode_other_drop_count", 0.0)),
                action_fill_ratio=float(sampled["action_fill_ratio"]),
                action_unique_ue_count=float(sampled["action_unique_ue_count"]),
                candidate_unique_ue_count=float(sampled["candidate_unique_ue_count"]),
                demand_active_ue_count=float(sampled["demand_active_ue_count"]),
                candidate_coverage_ratio=float(sampled["candidate_coverage_ratio"]),
                candidate_budgeted_changed_count=float(sampled.get("candidate_budgeted_changed_count", 0.0)),
                candidate_budgeted_no_headroom_count=float(sampled.get("candidate_budgeted_no_headroom_count", 0.0)),
                candidate_budgeted_active_ue_count=float(sampled.get("candidate_budgeted_active_ue_count", 0.0)),
                candidate_budgeted_tail_bias_coef=float(sampled.get("candidate_budgeted_tail_bias_coef", 0.0)),
                candidate_budgeted_tail_bias_candidate_mean=float(
                    sampled.get("candidate_budgeted_tail_bias_candidate_mean", 0.0)
                ),
                candidate_budgeted_tail_bias_selected_mean=float(
                    sampled.get("candidate_budgeted_tail_bias_selected_mean", 0.0)
                ),
                candidate_budgeted_tail_bias_active_count=float(
                    sampled.get("candidate_budgeted_tail_bias_active_count", 0.0)
                ),
                candidate_budgeted_rescue_bias_coef=float(
                    sampled.get("candidate_budgeted_rescue_bias_coef", 0.0)
                ),
                candidate_budgeted_rescue_bias_candidate_mean=float(
                    sampled.get("candidate_budgeted_rescue_bias_candidate_mean", 0.0)
                ),
                candidate_budgeted_rescue_bias_selected_mean=float(
                    sampled.get("candidate_budgeted_rescue_bias_selected_mean", 0.0)
                ),
                candidate_budgeted_rescue_bias_active_count=float(
                    sampled.get("candidate_budgeted_rescue_bias_active_count", 0.0)
                ),
                candidate_budgeted_completion_bonus=float(
                    sampled.get("candidate_budgeted_completion_bonus", 0.0)
                ),
                candidate_budgeted_completion_candidate_mean=float(
                    sampled.get("candidate_budgeted_completion_candidate_mean", 0.0)
                ),
                candidate_budgeted_completion_selected_mean=float(
                    sampled.get("candidate_budgeted_completion_selected_mean", 0.0)
                ),
                candidate_budgeted_completion_active_count=float(
                    sampled.get("candidate_budgeted_completion_active_count", 0.0)
                ),
                candidate_budgeted_tail_quota_enabled=float(
                    sampled.get("candidate_budgeted_tail_quota_enabled", 0.0)
                ),
                candidate_budgeted_tail_quota_candidate_mean=float(
                    sampled.get("candidate_budgeted_tail_quota_candidate_mean", 0.0)
                ),
                candidate_budgeted_tail_quota_active_ue_count=float(
                    sampled.get("candidate_budgeted_tail_quota_active_ue_count", 0.0)
                ),
                candidate_budgeted_tail_quota_selected_count=float(
                    sampled.get("candidate_budgeted_tail_quota_selected_count", 0.0)
                ),
                candidate_budgeted_tail_quota_selected_mean=float(
                    sampled.get("candidate_budgeted_tail_quota_selected_mean", 0.0)
                ),
                candidate_budgeted_tail_quota_served_ue_count=float(
                    sampled.get("candidate_budgeted_tail_quota_served_ue_count", 0.0)
                ),
                unserved_demand_ue_count=float(per_ue_summary["unserved_demand_ue_count"]),
                planned_unserved_demand_ue_count=float(per_ue_summary["planned_unserved_demand_ue_count"]),
                actual_unserved_backlog_bytes=float(per_ue_summary["actual_unserved_backlog_bytes"]),
                accepted_pre_pdsch_diagnostics_available=float(
                    per_ue_summary["accepted_pre_pdsch_diagnostics_available"]
                ),
                actual_diagnostics_available=float(per_ue_summary["actual_diagnostics_available"]),
                actual_packet_diagnostics_available=float(
                    per_ue_summary["actual_packet_diagnostics_available"]
                ),
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
                ue_macro_rate_proxy_mbps=float(per_ue_summary["ue_macro_rate_proxy_mbps"]),
                ue_p10_rate_proxy_mbps=float(per_ue_summary["ue_p10_rate_proxy_mbps"]),
                ue_min_rate_proxy_mbps=float(per_ue_summary["ue_min_rate_proxy_mbps"]),
                ue_rate_proxy_active_count=float(per_ue_summary["ue_rate_proxy_active_count"]),
                ue_macro_packet_rate_mbps=float(per_ue_summary["ue_macro_packet_rate_mbps"]),
                ue_p10_packet_rate_mbps=float(per_ue_summary["ue_p10_packet_rate_mbps"]),
                ue_min_packet_rate_mbps=float(per_ue_summary["ue_min_packet_rate_mbps"]),
                ue_packet_rate_active_count=float(per_ue_summary["ue_packet_rate_active_count"]),
                ue_packet_delivered_active_count=float(
                    per_ue_summary["ue_packet_delivered_active_count"]
                ),
                ue_macro_packet_delay_ms=float(per_ue_summary["ue_macro_packet_delay_ms"]),
                ue_p90_packet_delay_ms=float(per_ue_summary["ue_p90_packet_delay_ms"]),
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
                expiry_reward_rate=float(expiry_reward_rate),
                prg_utilization_ratio=float(reward_metrics["prg_utilization_ratio"]),
                prg_reuse_ratio=float(reward_metrics["prg_reuse_ratio"]),
                packet_completed_count=packet_completed_count,
                packet_system_time_ms=packet_system_time_ms,
                packet_delay_mean_ms=packet_delay_mean_ms,
                packet_effective_service_rate_mbps=packet_effective_service_rate_mbps,
                packet_effective_service_rate_per_packet_mean_mbps=(
                    packet_effective_service_rate_per_packet_mean_mbps
                ),
                conflict_mean=conflict_mean,
                ici_mean=ici_mean,
                prg_expected_goodput_proxy_mbps_mean=prg_expected_goodput_proxy_mbps_mean,
                prg_low_sinr_hol_ttl_weight_mean=prg_low_sinr_hol_ttl_weight_mean,
                urgent_backlog_penalty=urgent_backlog_penalty,
                backlog_debt_coef=backlog_debt_coef,
                packet_rate_bonus=packet_rate_bonus,
                packet_per_packet_rate_bonus=packet_per_packet_rate_bonus,
                packet_per_packet_rate_coef=packet_per_packet_rate_coef,
                packet_completion_bonus=packet_completion_bonus,
                packet_delay_penalty=packet_delay_penalty,
                packet_delay_coef=packet_delay_coef,
                ue_macro_rate_proxy_bonus=ue_macro_rate_proxy_bonus,
                ue_macro_rate_proxy_coef=ue_macro_rate_proxy_coef,
                ue_p10_rate_proxy_gap_penalty=ue_p10_rate_proxy_gap_penalty,
                ue_p10_rate_proxy_gap_coef=ue_p10_rate_proxy_gap_coef,
                ue_macro_packet_rate_bonus=ue_macro_packet_rate_bonus,
                ue_macro_packet_rate_coef=ue_macro_packet_rate_coef,
                ue_p10_packet_rate_gap_penalty=ue_p10_packet_rate_gap_penalty,
                ue_p10_packet_rate_gap_coef=ue_p10_packet_rate_gap_coef,
                aged_backlog_penalty=aged_backlog_penalty,
                aged_backlog_coef=aged_backlog_coef,
                active_prg_goodput_efficiency=active_prg_goodput_efficiency,
                harmful_packing_penalty=harmful_packing_penalty,
                util_floor_gate=util_floor_gate,
                util_floor_gap=util_floor_gap,
                util_floor_penalty=util_floor_penalty,
                util_floor_coef=util_floor_coef,
                missed_safe_opportunity_penalty=missed_safe_opportunity_penalty,
                missed_safe_opportunity_coef=missed_safe_opportunity_coef,
                tb_err_gap=tb_err_gap,
                tb_err_over_target_coef=tb_err_over_target_coef,
                tb_err_over_target_penalty=tb_err_over_target_penalty,
                tb_err_goodput_gate_coef=tb_err_goodput_gate_coef,
                tb_err_goodput_gate_penalty=tb_err_goodput_gate_penalty,
                tail_potential_before=tail_potential_before,
                tail_potential_after=tail_potential_after,
                tail_potential_delta=tail_potential_delta,
                tail_potential_topk_mean=tail_potential_topk_mean,
                tail_potential_max=tail_potential_max,
                expiry_potential_before=expiry_potential_before,
                expiry_potential_after=expiry_potential_after,
                expiry_potential_delta=expiry_potential_delta,
                expiry_potential_topk_mean=expiry_potential_topk_mean,
                expiry_potential_max=expiry_potential_max,
                rescue_credit=rescue_credit,
                rescue_miss=rescue_miss,
                rescue_risk_topk_mean=rescue_risk_topk_mean,
                rescue_served_ratio=rescue_served_ratio,
                coverage_repair_count=float(sampled["coverage_repair_count"]),
                candidate_decode_goodput_target=candidate_decode_goodput_target,
                candidate_decode_goodput_mask=candidate_decode_goodput_mask,
            )
        if is_warmup_step:
            warmup_steps_completed += 1
        else:
            steps.append(step_record)

        log_every = int(rollout_log_every_steps)
        step_count = len(steps)
        if log_every > 0 and step_count > 0 and ((step_count % log_every) == 0 or step_count == target_rollout_steps or done):
            tail = steps[-min(log_every, len(steps)) :]
            print(
                f"[rollout iter {episode_idx:04d}] "
                f"tti={step_count:04d}/{target_rollout_steps:04d} "
                f"warmup={warmup_steps_completed:04d}/{warmup_steps_requested:04d} "
                f"reward={float(np.mean([s.reward for s in tail])):.4f} "
                f"goodput={float(np.mean([s.goodput_mbps for s in tail])):.4f} "
                f"tb_err={float(np.mean([s.tb_err_rate for s in tail])):.4f} "
                f"blank={float(np.mean([s.blank_ratio for s in tail])):.4f} "
                f"drop={float(np.mean([s.post_decode_drop_ratio for s in tail])):.4f} "
                f"fill={float(np.mean([s.action_fill_ratio for s in tail])):.4f} "
                f"capDrop={float(np.mean([s.cap_drop_count for s in tail])):.2f} "
                f"fbOK={float(np.mean([s.fallback_success_count for s in tail])):.2f} "
                f"safeFill={float(np.mean([s.safe_fill_count for s in tail])):.2f} "
                f"blankBudget={float(np.mean([s.blank_budget_repair_count for s in tail])):.2f} "
                f"blankCnt={float(np.mean([s.final_blank_count for s in tail])):.2f} "
                f"dropCap={float(np.mean([s.post_decode_demand_cap_count for s in tail])):.2f} "
                f"dropSlot={float(np.mean([s.post_decode_no_slot_count for s in tail])):.2f} "
                f"dropOth={float(np.mean([s.post_decode_other_drop_count for s in tail])):.2f} "
                f"uniq_ue={float(np.mean([s.action_unique_ue_count for s in tail])):.2f} "
                f"cand_ue={float(np.mean([s.candidate_unique_ue_count for s in tail])):.2f} "
                f"demand_ue={float(np.mean([s.demand_active_ue_count for s in tail])):.2f} "
                f"cand_cov={float(np.mean([s.candidate_coverage_ratio for s in tail])):.3f} "
                f"budgChg={float(np.mean([s.candidate_budgeted_changed_count for s in tail])):.1f} "
                f"budgNoHr={float(np.mean([s.candidate_budgeted_no_headroom_count for s in tail])):.1f} "
                f"tailBiasSel={float(np.mean([s.candidate_budgeted_tail_bias_selected_mean for s in tail])):.3f} "
                f"rescueBiasSel={float(np.mean([s.candidate_budgeted_rescue_bias_selected_mean for s in tail])):.3f} "
                f"unsrv_ue={float(np.mean([s.unserved_demand_ue_count for s in tail])):.2f} "
                f"debtKB={float(np.mean([s.actual_unserved_backlog_bytes for s in tail]))/1024.0:.1f} "
                f"tailPot={float(np.mean([s.tail_potential_after for s in tail])):.3f} "
                f"tailDelta={float(np.mean([s.tail_potential_delta for s in tail])):.3f} "
                f"resc={float(np.mean([s.rescue_credit for s in tail])):.3f} "
                f"miss={float(np.mean([s.rescue_miss for s in tail])):.3f} "
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

    decode_goodput_mask_total = 0.0
    decode_goodput_target_sum = 0.0
    decode_goodput_positive_total = 0.0
    decode_goodput_mask_per_step: List[float] = []
    for step in steps:
        if step.candidate_decode_goodput_target is None or step.candidate_decode_goodput_mask is None:
            decode_goodput_mask_per_step.append(0.0)
            continue
        mask = step.candidate_decode_goodput_mask.to(dtype=torch.bool)
        target = step.candidate_decode_goodput_target.to(dtype=torch.float32)
        mask_count = float(mask.to(dtype=torch.float32).sum().item())
        decode_goodput_mask_per_step.append(mask_count)
        if mask_count <= 0.0:
            continue
        decode_goodput_mask_total += mask_count
        decode_goodput_target_sum += float((target * mask.to(dtype=torch.float32)).sum().item())
        decode_goodput_positive_total += float(((target > 0.0) & mask).to(dtype=torch.float32).sum().item())

    summary = {
        "episode_seed": float(episode_seed),
        "topology_seed": float(topology_seed),
        "rollout_steps": float(len(steps)),
        "rollout_warmup_steps": float(warmup_steps_completed),
        "rollout_warmup_steps_requested": float(warmup_steps_requested),
        "rollout_total_env_steps": float(warmup_steps_completed + len(steps)),
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
        "rollout_safe_fill_count_mean": float(np.mean([s.safe_fill_count for s in steps])) if steps else 0.0,
        "rollout_missed_safe_opportunity_count_mean": (
            float(np.mean([s.missed_safe_opportunity_count for s in steps])) if steps else 0.0
        ),
        "rollout_blank_budget_repair_count_mean": (
            float(np.mean([s.blank_budget_repair_count for s in steps])) if steps else 0.0
        ),
        "rollout_final_blank_count_mean": float(np.mean([s.final_blank_count for s in steps])) if steps else 0.0,
        "rollout_post_decode_policy_blank_count_mean": (
            float(np.mean([s.post_decode_policy_blank_count for s in steps])) if steps else 0.0
        ),
        "rollout_post_decode_no_valid_candidate_count_mean": (
            float(np.mean([s.post_decode_no_valid_candidate_count for s in steps])) if steps else 0.0
        ),
        "rollout_post_decode_demand_cap_count_mean": (
            float(np.mean([s.post_decode_demand_cap_count for s in steps])) if steps else 0.0
        ),
        "rollout_post_decode_no_slot_count_mean": (
            float(np.mean([s.post_decode_no_slot_count for s in steps])) if steps else 0.0
        ),
        "rollout_post_decode_other_drop_count_mean": (
            float(np.mean([s.post_decode_other_drop_count for s in steps])) if steps else 0.0
        ),
        "rollout_action_fill_ratio_mean": float(np.mean([s.action_fill_ratio for s in steps])) if steps else 0.0,
        "rollout_action_unique_ue_mean": float(np.mean([s.action_unique_ue_count for s in steps])) if steps else 0.0,
        "rollout_candidate_unique_ue_mean": (
            float(np.mean([s.candidate_unique_ue_count for s in steps])) if steps else 0.0
        ),
        "rollout_demand_active_ue_mean": float(np.mean([s.demand_active_ue_count for s in steps])) if steps else 0.0,
        "rollout_candidate_coverage_ratio_mean": (
            float(np.mean([s.candidate_coverage_ratio for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_changed_count_mean": (
            float(np.mean([s.candidate_budgeted_changed_count for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_no_headroom_count_mean": (
            float(np.mean([s.candidate_budgeted_no_headroom_count for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_active_ue_count_mean": (
            float(np.mean([s.candidate_budgeted_active_ue_count for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_tail_bias_coef_mean": (
            float(np.mean([s.candidate_budgeted_tail_bias_coef for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_tail_bias_candidate_mean": (
            float(np.mean([s.candidate_budgeted_tail_bias_candidate_mean for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_tail_bias_selected_mean": (
            float(np.mean([s.candidate_budgeted_tail_bias_selected_mean for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_tail_bias_active_count_mean": (
            float(np.mean([s.candidate_budgeted_tail_bias_active_count for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_rescue_bias_coef_mean": (
            float(np.mean([s.candidate_budgeted_rescue_bias_coef for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_rescue_bias_candidate_mean": (
            float(np.mean([s.candidate_budgeted_rescue_bias_candidate_mean for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_rescue_bias_selected_mean": (
            float(np.mean([s.candidate_budgeted_rescue_bias_selected_mean for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_rescue_bias_active_count_mean": (
            float(np.mean([s.candidate_budgeted_rescue_bias_active_count for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_completion_bonus_mean": (
            float(np.mean([s.candidate_budgeted_completion_bonus for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_completion_candidate_mean": (
            float(np.mean([s.candidate_budgeted_completion_candidate_mean for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_completion_selected_mean": (
            float(np.mean([s.candidate_budgeted_completion_selected_mean for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_completion_active_count_mean": (
            float(np.mean([s.candidate_budgeted_completion_active_count for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_tail_quota_enabled_mean": (
            float(np.mean([s.candidate_budgeted_tail_quota_enabled for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_tail_quota_candidate_mean": (
            float(np.mean([s.candidate_budgeted_tail_quota_candidate_mean for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_tail_quota_active_ue_count_mean": (
            float(np.mean([s.candidate_budgeted_tail_quota_active_ue_count for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_tail_quota_selected_count_mean": (
            float(np.mean([s.candidate_budgeted_tail_quota_selected_count for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_tail_quota_selected_mean": (
            float(np.mean([s.candidate_budgeted_tail_quota_selected_mean for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_budgeted_tail_quota_served_ue_count_mean": (
            float(np.mean([s.candidate_budgeted_tail_quota_served_ue_count for s in steps])) if steps else 0.0
        ),
        "rollout_rescue_credit_mean": float(np.mean([s.rescue_credit for s in steps])) if steps else 0.0,
        "rollout_rescue_miss_mean": float(np.mean([s.rescue_miss for s in steps])) if steps else 0.0,
        "rollout_rescue_risk_topk_mean": float(np.mean([s.rescue_risk_topk_mean for s in steps])) if steps else 0.0,
        "rollout_rescue_served_ratio_mean": float(np.mean([s.rescue_served_ratio for s in steps])) if steps else 0.0,
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
        "rollout_actual_ue_packet_diagnostics_available_mean": (
            float(np.mean([s.actual_packet_diagnostics_available for s in steps])) if steps else 0.0
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
        "rollout_ue_macro_rate_proxy_mbps_mean": (
            float(np.mean([s.ue_macro_rate_proxy_mbps for s in steps])) if steps else 0.0
        ),
        "rollout_ue_p10_rate_proxy_mbps_mean": (
            float(np.mean([s.ue_p10_rate_proxy_mbps for s in steps])) if steps else 0.0
        ),
        "rollout_ue_min_rate_proxy_mbps_mean": (
            float(np.mean([s.ue_min_rate_proxy_mbps for s in steps])) if steps else 0.0
        ),
        "rollout_ue_rate_proxy_active_count_mean": (
            float(np.mean([s.ue_rate_proxy_active_count for s in steps])) if steps else 0.0
        ),
        "rollout_ue_macro_packet_rate_mbps_mean": (
            float(np.mean([s.ue_macro_packet_rate_mbps for s in steps])) if steps else 0.0
        ),
        "rollout_ue_p10_packet_rate_mbps_mean": (
            float(np.mean([s.ue_p10_packet_rate_mbps for s in steps])) if steps else 0.0
        ),
        "rollout_ue_min_packet_rate_mbps_mean": (
            float(np.mean([s.ue_min_packet_rate_mbps for s in steps])) if steps else 0.0
        ),
        "rollout_ue_packet_rate_active_count_mean": (
            float(np.mean([s.ue_packet_rate_active_count for s in steps])) if steps else 0.0
        ),
        "rollout_ue_packet_delivered_active_count_mean": (
            float(np.mean([s.ue_packet_delivered_active_count for s in steps])) if steps else 0.0
        ),
        "rollout_ue_macro_packet_delay_ms_mean": (
            float(np.mean([s.ue_macro_packet_delay_ms for s in steps])) if steps else 0.0
        ),
        "rollout_ue_p90_packet_delay_ms_mean": (
            float(np.mean([s.ue_p90_packet_delay_ms for s in steps])) if steps else 0.0
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
        "rollout_expiry_reward_rate_mean": (
            float(np.mean([s.expiry_reward_rate for s in steps])) if steps else 0.0
        ),
        "rollout_prg_utilization_ratio_mean": (
            float(np.mean([s.prg_utilization_ratio for s in steps])) if steps else 0.0
        ),
        "rollout_prg_reuse_ratio_mean": float(np.mean([s.prg_reuse_ratio for s in steps])) if steps else 0.0,
        "rollout_packet_completed_count_mean": (
            float(np.mean([s.packet_completed_count for s in steps])) if steps else 0.0
        ),
        "rollout_packet_system_time_ms_mean": (
            float(np.mean([s.packet_system_time_ms for s in steps])) if steps else 0.0
        ),
        "rollout_packet_system_time_ms_sum": (
            float(np.sum([s.packet_system_time_ms for s in steps])) if steps else 0.0
        ),
        "rollout_packet_completed_count_sum": (
            float(np.sum([s.packet_completed_count for s in steps])) if steps else 0.0
        ),
        "rollout_packet_delay_mean_ms_mean": (
            float(np.mean([s.packet_delay_mean_ms for s in steps])) if steps else 0.0
        ),
        "rollout_packet_delay_weighted_mean_ms": (
            float(np.sum([s.packet_system_time_ms for s in steps]))
            / max(1.0, float(np.sum([s.packet_completed_count for s in steps])))
            if steps
            else 0.0
        ),
        "rollout_packet_effective_service_rate_mbps_mean": (
            float(np.mean([s.packet_effective_service_rate_mbps for s in steps])) if steps else 0.0
        ),
        "rollout_packet_effective_service_rate_per_packet_mean_mbps_mean": (
            float(np.mean([s.packet_effective_service_rate_per_packet_mean_mbps for s in steps]))
            if steps
            else 0.0
        ),
        "rollout_conflict_mean": float(np.mean([s.conflict_mean for s in steps])) if steps else 0.0,
        "rollout_ici_mean": float(np.mean([s.ici_mean for s in steps])) if steps else 0.0,
        "rollout_prg_expected_goodput_proxy_mbps_mean": (
            float(np.mean([s.prg_expected_goodput_proxy_mbps_mean for s in steps])) if steps else 0.0
        ),
        "rollout_prg_low_sinr_hol_ttl_weight_mean": (
            float(np.mean([s.prg_low_sinr_hol_ttl_weight_mean for s in steps])) if steps else 0.0
        ),
        "rollout_active_prg_goodput_efficiency_mean": (
            float(np.mean([s.active_prg_goodput_efficiency for s in steps])) if steps else 0.0
        ),
        "rollout_harmful_packing_penalty_mean": (
            float(np.mean([s.harmful_packing_penalty for s in steps])) if steps else 0.0
        ),
        "rollout_util_floor_gate_mean": (
            float(np.mean([s.util_floor_gate for s in steps])) if steps else 0.0
        ),
        "rollout_util_floor_gap_mean": (
            float(np.mean([s.util_floor_gap for s in steps])) if steps else 0.0
        ),
        "rollout_util_floor_penalty_mean": (
            float(np.mean([s.util_floor_penalty for s in steps])) if steps else 0.0
        ),
        "rollout_util_floor_coef_mean": (
            float(np.mean([s.util_floor_coef for s in steps])) if steps else 0.0
        ),
        "rollout_missed_safe_opportunity_penalty_mean": (
            float(np.mean([s.missed_safe_opportunity_penalty for s in steps])) if steps else 0.0
        ),
        "rollout_missed_safe_opportunity_coef_mean": (
            float(np.mean([s.missed_safe_opportunity_coef for s in steps])) if steps else 0.0
        ),
        "rollout_tb_err_gap_mean": (
            float(np.mean([s.tb_err_gap for s in steps])) if steps else 0.0
        ),
        "rollout_tb_err_over_target_coef_mean": (
            float(np.mean([s.tb_err_over_target_coef for s in steps])) if steps else 0.0
        ),
        "rollout_tb_err_over_target_weighted_penalty_mean": (
            float(np.mean([s.tb_err_over_target_penalty * s.tb_err_over_target_coef for s in steps]))
            if steps
            else 0.0
        ),
        "rollout_tb_err_goodput_gate_coef_mean": (
            float(np.mean([s.tb_err_goodput_gate_coef for s in steps])) if steps else 0.0
        ),
        "rollout_tb_err_goodput_gate_penalty_mean": (
            float(np.mean([s.tb_err_goodput_gate_penalty for s in steps])) if steps else 0.0
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
        "rollout_packet_per_packet_rate_bonus_mean": (
            float(np.mean([s.packet_per_packet_rate_bonus for s in steps])) if steps else 0.0
        ),
        "rollout_packet_per_packet_rate_coef_mean": (
            float(np.mean([s.packet_per_packet_rate_coef for s in steps])) if steps else 0.0
        ),
        "rollout_packet_per_packet_rate_weighted_bonus_mean": (
            float(np.mean([s.packet_per_packet_rate_bonus * s.packet_per_packet_rate_coef for s in steps]))
            if steps
            else 0.0
        ),
        "rollout_packet_completion_bonus_mean": (
            float(np.mean([s.packet_completion_bonus for s in steps])) if steps else 0.0
        ),
        "rollout_packet_delay_penalty_mean": (
            float(np.mean([s.packet_delay_penalty for s in steps])) if steps else 0.0
        ),
        "rollout_packet_delay_coef_mean": (
            float(np.mean([s.packet_delay_coef for s in steps])) if steps else 0.0
        ),
        "rollout_packet_delay_weighted_penalty_mean": (
            float(np.mean([s.packet_delay_penalty * s.packet_delay_coef for s in steps])) if steps else 0.0
        ),
        "rollout_ue_macro_rate_proxy_bonus_mean": (
            float(np.mean([s.ue_macro_rate_proxy_bonus for s in steps])) if steps else 0.0
        ),
        "rollout_ue_macro_rate_proxy_coef_mean": (
            float(np.mean([s.ue_macro_rate_proxy_coef for s in steps])) if steps else 0.0
        ),
        "rollout_ue_macro_rate_proxy_weighted_bonus_mean": (
            float(np.mean([s.ue_macro_rate_proxy_bonus * s.ue_macro_rate_proxy_coef for s in steps]))
            if steps
            else 0.0
        ),
        "rollout_ue_p10_rate_proxy_gap_penalty_mean": (
            float(np.mean([s.ue_p10_rate_proxy_gap_penalty for s in steps])) if steps else 0.0
        ),
        "rollout_ue_p10_rate_proxy_gap_coef_mean": (
            float(np.mean([s.ue_p10_rate_proxy_gap_coef for s in steps])) if steps else 0.0
        ),
        "rollout_ue_p10_rate_proxy_weighted_gap_penalty_mean": (
            float(
                np.mean(
                    [s.ue_p10_rate_proxy_gap_penalty * s.ue_p10_rate_proxy_gap_coef for s in steps]
                )
            )
            if steps
            else 0.0
        ),
        "rollout_ue_macro_packet_rate_bonus_mean": (
            float(np.mean([s.ue_macro_packet_rate_bonus for s in steps])) if steps else 0.0
        ),
        "rollout_ue_macro_packet_rate_coef_mean": (
            float(np.mean([s.ue_macro_packet_rate_coef for s in steps])) if steps else 0.0
        ),
        "rollout_ue_macro_packet_rate_weighted_bonus_mean": (
            float(np.mean([s.ue_macro_packet_rate_bonus * s.ue_macro_packet_rate_coef for s in steps]))
            if steps
            else 0.0
        ),
        "rollout_ue_p10_packet_rate_gap_penalty_mean": (
            float(np.mean([s.ue_p10_packet_rate_gap_penalty for s in steps])) if steps else 0.0
        ),
        "rollout_ue_p10_packet_rate_gap_coef_mean": (
            float(np.mean([s.ue_p10_packet_rate_gap_coef for s in steps])) if steps else 0.0
        ),
        "rollout_ue_p10_packet_rate_weighted_gap_penalty_mean": (
            float(
                np.mean(
                    [s.ue_p10_packet_rate_gap_penalty * s.ue_p10_packet_rate_gap_coef for s in steps]
                )
            )
            if steps
            else 0.0
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
        "rollout_tail_potential_before_mean": (
            float(np.mean([s.tail_potential_before for s in steps])) if steps else 0.0
        ),
        "rollout_tail_potential_after_mean": (
            float(np.mean([s.tail_potential_after for s in steps])) if steps else 0.0
        ),
        "rollout_tail_potential_delta_mean": (
            float(np.mean([s.tail_potential_delta for s in steps])) if steps else 0.0
        ),
        "rollout_tail_potential_topk_mean": (
            float(np.mean([s.tail_potential_topk_mean for s in steps])) if steps else 0.0
        ),
        "rollout_tail_potential_max_mean": (
            float(np.mean([s.tail_potential_max for s in steps])) if steps else 0.0
        ),
        "rollout_expiry_potential_before_mean": (
            float(np.mean([s.expiry_potential_before for s in steps])) if steps else 0.0
        ),
        "rollout_expiry_potential_after_mean": (
            float(np.mean([s.expiry_potential_after for s in steps])) if steps else 0.0
        ),
        "rollout_expiry_potential_delta_mean": (
            float(np.mean([s.expiry_potential_delta for s in steps])) if steps else 0.0
        ),
        "rollout_expiry_potential_topk_mean": (
            float(np.mean([s.expiry_potential_topk_mean for s in steps])) if steps else 0.0
        ),
        "rollout_expiry_potential_max_mean": (
            float(np.mean([s.expiry_potential_max for s in steps])) if steps else 0.0
        ),
        "rollout_coverage_repair_count_mean": (
            float(np.mean([s.coverage_repair_count for s in steps])) if steps else 0.0
        ),
        "rollout_candidate_decode_goodput_mask_count_mean": (
            float(np.mean(decode_goodput_mask_per_step)) if decode_goodput_mask_per_step else 0.0
        ),
        "rollout_candidate_decode_goodput_target_mean": (
            decode_goodput_target_sum / max(decode_goodput_mask_total, 1.0)
        ),
        "rollout_candidate_decode_goodput_positive_ratio": (
            decode_goodput_positive_total / max(decode_goodput_mask_total, 1.0)
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
    return steps, summary, per_ue_tracker, expiry_reward_tracker


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
    target_kl_min_updates: int = 4,
    state_norm: RunningNorm | None = None,
    max_grad_norm: float = 5.0,
    value_loss_type: str = "mse",
    value_huber_beta: float = 10.0,
    teacher_aux_coef: float = 0.0,
    teacher_soft_target: int = 1,
    teacher_soft_temperature: float = 0.85,
    teacher_soft_hit_bonus: float = 2.0,
    teacher_soft_blank_bonus: float = 0.35,
    teacher_soft_blank_safe_penalty: float = 1.25,
    teacher_soft_blank_no_safe_bonus: float = 1.0,
    candidate_success_aux_coef: float = 0.0,
    candidate_success_calib_coef: float = 0.0,
    candidate_success_max_calib_coef: float = 0.0,
    candidate_success_target_eps: float = 0.05,
    candidate_pairwise_rank_coef: float = 0.0,
    candidate_pairwise_rank_margin: float = 0.25,
    candidate_pairwise_rank_min_gap: float = 0.10,
    candidate_pairwise_rank_success_weight: float = 0.70,
    candidate_pairwise_rank_service_weight: float = 0.30,
    candidate_pairwise_rank_risk_weight: float = 0.15,
    candidate_pairwise_rank_fail_risk_weight: float = 0.0,
    candidate_pairwise_rank_fail_risk_feature_offset: int = -1,
    candidate_decode_goodput_coef: float = 0.0,
    candidate_decode_goodput_pos_weight: float = 2.0,
    candidate_argmax_decode_coef: float = 0.0,
    candidate_argmax_decode_pos_weight: float = 2.0,
    candidate_argmax_decode_headroom_rank_mode: str = "stable",
    candidate_blank_target_coef: float = 0.0,
    candidate_blank_target_threshold: float = 0.45,
    candidate_blank_target_margin: float = 0.5,
    candidate_blank_target_deadband: float = 0.05,
    candidate_blank_target_pos_weight: float = 1.0,
    candidate_blank_target_fill_weight: float = 1.0,
    candidate_blank_target_positive_only: int = 0,
    candidate_blank_target_service_guard: int = 0,
    candidate_blank_target_service_min_success: float = 0.70,
    candidate_blank_target_service_min_debt: float = 0.25,
    candidate_blank_target_service_min_hol_ms: float = 1.0,
    candidate_blank_target_service_max_scheduled_ratio: float = 0.65,
    candidate_blank_target_service_fill_weight: float = 0.0,
    candidate_blank_target_service_fill_margin: float = 0.15,
    candidate_blank_target_service_fill_min_success: float = 0.995,
    safe_fill_max_per_cell: int = 0,
    safe_fill_min_quality_db: float = -2.0,
    safe_fill_max_quality_gap_db: float = 8.0,
    safe_fill_max_prg_risk: float = 0.70,
    safe_fill_min_backlog_bytes: float = 1.0,
    safe_fill_max_ue_tb_err: float = 0.45,
    candidate_blank_budget_max_per_cell: int = -1,
    candidate_blank_budget_min_decision_logit: float = -1.0e9,
    candidate_decode_demand_slack_bytes: float = 3000.0,
    candidate_marginal_headroom_slack_bytes: float = -1.0,
    candidate_budgeted_select_mode: str = "off",
    candidate_budgeted_hard_cap: int = 1,
    candidate_budgeted_usage_penalty: float = 0.35,
    candidate_budgeted_overcap_penalty: float = 8.0,
    candidate_budgeted_first_service_bonus: float = 0.0,
    candidate_budgeted_tail_bias_coef: float = 0.0,
    candidate_budgeted_tail_bias_service_target: float = 0.75,
    candidate_budgeted_tail_bias_ttl_guard_ms: float = 20.0,
    candidate_budgeted_tail_bias_max: float = 1.5,
    candidate_budgeted_rescue_bias_coef: float = 0.0,
    candidate_budgeted_rescue_bias_service_target: float = 0.80,
    candidate_budgeted_rescue_bias_ttl_guard_ms: float = 60.0,
    candidate_budgeted_rescue_bias_max: float = 1.5,
    candidate_budgeted_completion_bonus: float = 0.0,
    candidate_budgeted_completion_progress_weight: float = 0.25,
    candidate_budgeted_completion_max_score: float = 1.5,
    candidate_budgeted_tail_quota_enable: int = 0,
    candidate_budgeted_tail_quota_min_score: float = 0.55,
    candidate_budgeted_tail_quota_topk_ues: int = 6,
    candidate_budgeted_tail_quota_prgs_per_ue: int = 1,
    candidate_budgeted_tail_quota_bonus: float = 4.0,
    candidate_budgeted_tail_quota_order_bonus: float = 2.0,
    candidate_budgeted_tail_quota_max_score: float = 4.0,
    candidate_budgeted_tail_quota_risk_max: float = 0.70,
    candidate_seq_state_coef: float = 1.0,
    candidate_seq_fail_risk_feature_offset: int = -1,
    candidate_rescue_aux_coef: float = 0.0,
    candidate_rescue_pos_weight: float = 2.0,
    candidate_rescue_service_target: float = 0.80,
    candidate_rescue_ttl_guard_ms: float = 60.0,
    candidate_rescue_max_target: float = 1.0,
    candidate_pairwise_rank_rescue_weight: float = 0.0,
) -> Dict[str, float]:
    actor.train(True)
    critic.train(True)

    old_logp = torch.stack([s.old_logp.to(torch.float32) for s in rollout], dim=0)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_teacher_aux_loss = 0.0
    total_candidate_success_aux_loss = 0.0
    total_candidate_success_calib_loss = 0.0
    total_candidate_success_max_calib_loss = 0.0
    total_candidate_pairwise_rank_loss = 0.0
    total_candidate_pairwise_rank_pair_ratio = 0.0
    total_candidate_rescue_aux_loss = 0.0
    total_candidate_rescue_target_mean = 0.0
    total_candidate_rescue_positive_ratio = 0.0
    total_candidate_rescue_prob_mae = 0.0
    total_candidate_rescue_mask_ratio = 0.0
    total_candidate_decode_goodput_loss = 0.0
    total_candidate_decode_goodput_target_mean = 0.0
    total_candidate_decode_goodput_positive_ratio = 0.0
    total_candidate_decode_goodput_prob_mean = 0.0
    total_candidate_decode_goodput_prob_mae = 0.0
    total_candidate_decode_goodput_mask_ratio = 0.0
    total_candidate_decode_goodput_logit_mean = 0.0
    total_candidate_argmax_decode_loss = 0.0
    total_candidate_argmax_decode_target_mean = 0.0
    total_candidate_argmax_decode_positive_ratio = 0.0
    total_candidate_argmax_decode_keep_ratio = 0.0
    total_candidate_argmax_decode_prob_mae = 0.0
    total_candidate_argmax_decode_mask_ratio = 0.0
    total_candidate_blank_target_loss = 0.0
    total_candidate_blank_target_ratio = 0.0
    total_candidate_blank_target_raw_ratio = 0.0
    total_candidate_blank_target_service_block_ratio = 0.0
    total_candidate_blank_target_service_fill_ratio = 0.0
    total_candidate_blank_target_weighted_ratio = 0.0
    total_candidate_shadow_blank_ratio = 0.0
    total_candidate_success_max_logit = 0.0
    total_candidate_success_max_target = 0.0
    total_candidate_success_max_target_p10 = 0.0
    total_candidate_success_max_target_p50 = 0.0
    total_candidate_success_max_target_p90 = 0.0
    total_candidate_success_max_logit_p10 = 0.0
    total_candidate_success_max_logit_p50 = 0.0
    total_candidate_success_max_logit_p90 = 0.0
    total_candidate_success_prob_mae = 0.0
    total_candidate_success_prob_brier = 0.0
    total_candidate_success_prob_mean = 0.0
    total_candidate_success_target_mean = 0.0
    total_candidate_blank_decision_logit_mean = 0.0
    total_candidate_blank_decision_logit_p10 = 0.0
    total_candidate_blank_decision_logit_p50 = 0.0
    total_candidate_blank_decision_logit_p90 = 0.0
    update_count = 0
    n_steps = len(rollout)
    train_device = next(actor.parameters()).device
    old_logp = old_logp.to(device=train_device)
    advantages = advantages.to(device=train_device)
    returns = returns.to(device=train_device)

    update_t0 = time.perf_counter()
    minibatches_per_epoch = max(1, math.ceil(n_steps / max(1, int(minibatch_size))))
    min_updates_before_early_stop = max(1, int(target_kl_min_updates))
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
                candidate_decode_demand_slack_bytes=float(candidate_decode_demand_slack_bytes),
                candidate_budgeted_select_mode=str(candidate_budgeted_select_mode),
                candidate_budgeted_hard_cap=int(candidate_budgeted_hard_cap),
                candidate_budgeted_usage_penalty=float(candidate_budgeted_usage_penalty),
                candidate_budgeted_overcap_penalty=float(candidate_budgeted_overcap_penalty),
                candidate_budgeted_first_service_bonus=float(candidate_budgeted_first_service_bonus),
                candidate_budgeted_tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                candidate_budgeted_tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                candidate_budgeted_tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                candidate_budgeted_tail_bias_max=float(candidate_budgeted_tail_bias_max),
                candidate_budgeted_rescue_bias_coef=float(candidate_budgeted_rescue_bias_coef),
                candidate_budgeted_rescue_bias_service_target=float(candidate_budgeted_rescue_bias_service_target),
                candidate_budgeted_rescue_bias_ttl_guard_ms=float(candidate_budgeted_rescue_bias_ttl_guard_ms),
                candidate_budgeted_rescue_bias_max=float(candidate_budgeted_rescue_bias_max),
                candidate_budgeted_completion_bonus=float(candidate_budgeted_completion_bonus),
                candidate_budgeted_completion_progress_weight=float(candidate_budgeted_completion_progress_weight),
                candidate_budgeted_completion_max_score=float(candidate_budgeted_completion_max_score),
                candidate_budgeted_tail_quota_enable=int(candidate_budgeted_tail_quota_enable),
                candidate_budgeted_tail_quota_min_score=float(candidate_budgeted_tail_quota_min_score),
                candidate_budgeted_tail_quota_topk_ues=int(candidate_budgeted_tail_quota_topk_ues),
                candidate_budgeted_tail_quota_prgs_per_ue=int(candidate_budgeted_tail_quota_prgs_per_ue),
                candidate_budgeted_tail_quota_bonus=float(candidate_budgeted_tail_quota_bonus),
                candidate_budgeted_tail_quota_order_bonus=float(candidate_budgeted_tail_quota_order_bonus),
                candidate_budgeted_tail_quota_max_score=float(candidate_budgeted_tail_quota_max_score),
                candidate_budgeted_tail_quota_risk_max=float(candidate_budgeted_tail_quota_risk_max),
                candidate_seq_state_coef=float(candidate_seq_state_coef),
                candidate_seq_fail_risk_feature_offset=int(candidate_seq_fail_risk_feature_offset),
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
                if "candidate_prg_logits" in actor_out:
                    prg_logits_masked, prg_valid = _masked_candidate_prg_logits_batch(
                        actor_out,
                        graphs_mb,
                        ue_action_class=ue_action_mb,
                    )
                else:
                    prg_logits_masked, prg_valid = _masked_prg_logits_batch(
                        actor_out,
                        graphs_mb,
                        ue_action_class=ue_action_mb,
                    )
                if "candidate_prg_logits" in actor_out and int(teacher_soft_target) == 1:
                    soft_targets: List[torch.Tensor] = []
                    for batch_idx, graph in enumerate(graphs_mb):
                        meta = graph.snapshot.static_meta
                        if meta.teacher_action_ue is None or meta.teacher_action_prg is None:
                            blank = torch.zeros_like(prg_logits_masked[batch_idx])
                            blank[:, -1] = 1.0
                            soft_targets.append(blank)
                            continue
                        packed = pack_prg_candidates(
                            graph,
                            max_candidates=max(0, int(prg_logits_masked.shape[-1]) - 1),
                            fixed_width=True,
                        )
                        soft_targets.append(
                            candidate_prg_soft_targets_from_type0_actions(
                                graph,
                                packed,
                                prg_valid[batch_idx],
                                action_ue_select=meta.teacher_action_ue.to(device=train_device, dtype=torch.long),
                                action_prg_alloc=meta.teacher_action_prg.to(device=train_device, dtype=torch.long),
                                temperature=float(teacher_soft_temperature),
                                teacher_hit_bonus=float(teacher_soft_hit_bonus),
                                teacher_blank_bonus=float(teacher_soft_blank_bonus),
                                blank_safe_penalty=float(teacher_soft_blank_safe_penalty),
                                blank_no_safe_bonus=float(teacher_soft_blank_no_safe_bonus),
                                min_quality_db=float(safe_fill_min_quality_db),
                                max_quality_gap_db=float(safe_fill_max_quality_gap_db),
                                max_prg_risk=float(safe_fill_max_prg_risk),
                                min_backlog_bytes=float(safe_fill_min_backlog_bytes),
                                max_ue_tb_err=float(safe_fill_max_ue_tb_err),
                            )
                        )
                    teacher_aux_loss = _soft_target_cross_entropy(
                        prg_logits_masked,
                        torch.stack(soft_targets, dim=0),
                        prg_valid,
                    )
                else:
                    teacher_targets: List[torch.Tensor] = []
                    for batch_idx, graph in enumerate(graphs_mb):
                        meta = graph.snapshot.static_meta
                        if meta.teacher_action_ue is None or meta.teacher_action_prg is None:
                            target_shape = (
                                (int(graph.snapshot.n_prg_tokens),)
                                if "candidate_prg_logits" in actor_out
                                else (meta.n_cell, meta.n_prg)
                            )
                            teacher_targets.append(
                                torch.full(
                                    target_shape,
                                    -100,
                                    dtype=torch.long,
                                    device=train_device,
                                )
                            )
                            continue
                        if "candidate_prg_logits" in actor_out:
                            packed = pack_prg_candidates(
                                graph,
                                max_candidates=max(0, int(prg_logits_masked.shape[-1]) - 1),
                                fixed_width=True,
                            )
                            _teacher_ue, teacher_prg = joint_candidate_targets_from_type0_actions(
                                graph,
                                packed,
                                action_ue_select=meta.teacher_action_ue.to(device=train_device, dtype=torch.long),
                                action_prg_alloc=meta.teacher_action_prg.to(device=train_device, dtype=torch.long),
                            )
                        else:
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
            candidate_success_aux_loss = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_calib_loss = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_max_calib_loss = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_pairwise_rank_loss = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_pairwise_rank_pair_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_rescue_aux_loss = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_rescue_target_mean = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_rescue_positive_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_rescue_prob_mae = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_rescue_mask_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_decode_goodput_loss = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_decode_goodput_target_mean = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_decode_goodput_positive_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_decode_goodput_prob_mean = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_decode_goodput_prob_mae = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_decode_goodput_mask_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_decode_goodput_logit_mean = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_argmax_decode_loss = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_argmax_decode_target_mean = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_argmax_decode_positive_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_argmax_decode_keep_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_argmax_decode_prob_mae = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_argmax_decode_mask_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_blank_target_loss = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_blank_target_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_blank_target_raw_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_blank_target_service_block_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_blank_target_service_fill_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_blank_target_weighted_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_shadow_blank_ratio = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_max_logit = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_max_target = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_max_target_p10 = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_max_target_p50 = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_max_target_p90 = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_max_logit_p10 = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_max_logit_p50 = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_max_logit_p90 = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_prob_mae = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_prob_brier = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_prob_mean = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_success_target_mean = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_blank_decision_logit_mean = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_blank_decision_logit_p10 = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_blank_decision_logit_p50 = torch.zeros((), dtype=torch.float32, device=train_device)
            candidate_blank_decision_logit_p90 = torch.zeros((), dtype=torch.float32, device=train_device)
            need_candidate_calib = (
                candidate_success_aux_coef > 0.0
                or candidate_success_calib_coef > 0.0
                or candidate_success_max_calib_coef > 0.0
                or candidate_pairwise_rank_coef > 0.0
                or candidate_rescue_aux_coef > 0.0
                or candidate_decode_goodput_coef > 0.0
                or candidate_argmax_decode_coef > 0.0
                or candidate_blank_target_coef > 0.0
            )
            if need_candidate_calib and "candidate_success_logits" in actor_out:
                success_logits = actor_out["candidate_success_logits"]
                if success_logits.dim() != 3:
                    raise ValueError(f"candidate_success_logits must be [B,P,K], got {tuple(success_logits.shape)}")
                max_candidates = int(success_logits.shape[-1])
                success_targets: List[torch.Tensor] = []
                success_masks: List[torch.Tensor] = []
                rank_targets: List[torch.Tensor] = []
                rescue_targets: List[torch.Tensor] = []
                service_protection_masks: List[torch.Tensor] = []
                for batch_idx, graph in enumerate(graphs_mb):
                    packed = pack_prg_candidates(graph, max_candidates=max_candidates, fixed_width=True)
                    valid, _selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(
                        graph,
                        packed,
                        ue_action_mb[batch_idx].to(dtype=torch.long),
                    )
                    target, mask = candidate_prg_success_targets(graph, packed, valid)
                    if candidate_rescue_aux_coef > 0.0:
                        rescue_target, _rescue_mask = candidate_prg_rescue_priority_scores(
                            graph,
                            packed,
                            valid,
                            success_target=target,
                            service_target=float(candidate_rescue_service_target),
                            ttl_guard_ms=float(candidate_rescue_ttl_guard_ms),
                            max_score=float(candidate_rescue_max_target),
                        )
                        rescue_targets.append(rescue_target.to(device=train_device, dtype=success_logits.dtype))
                    if candidate_pairwise_rank_coef > 0.0:
                        rank_target, _rank_mask = candidate_prg_pairwise_rank_targets(
                            graph,
                            packed,
                            valid,
                            success_target=target,
                            success_weight=float(candidate_pairwise_rank_success_weight),
                            service_weight=float(candidate_pairwise_rank_service_weight),
                            risk_weight=float(candidate_pairwise_rank_risk_weight),
                            fail_risk_weight=float(candidate_pairwise_rank_fail_risk_weight),
                            fail_risk_feature_offset=int(candidate_pairwise_rank_fail_risk_feature_offset),
                            rescue_weight=float(candidate_pairwise_rank_rescue_weight),
                            rescue_service_target=float(candidate_rescue_service_target),
                            rescue_ttl_guard_ms=float(candidate_rescue_ttl_guard_ms),
                        )
                        rank_targets.append(rank_target.to(device=train_device, dtype=success_logits.dtype))
                    if int(candidate_blank_target_service_guard) == 1:
                        service_protected = candidate_prg_service_protection_mask(
                            graph,
                            packed,
                            valid,
                            success_target=target,
                            min_success_target=float(candidate_blank_target_service_min_success),
                            min_backlog_bytes=float(safe_fill_min_backlog_bytes),
                            max_ue_tb_err=float(safe_fill_max_ue_tb_err),
                            min_quality_db=float(safe_fill_min_quality_db),
                            max_quality_gap_db=float(safe_fill_max_quality_gap_db),
                            max_prg_risk=float(safe_fill_max_prg_risk),
                            min_debt=float(candidate_blank_target_service_min_debt),
                            min_hol_delay_ms=float(candidate_blank_target_service_min_hol_ms),
                            max_recent_scheduled_ratio=float(
                                candidate_blank_target_service_max_scheduled_ratio
                            ),
                        )
                    else:
                        service_protected = torch.zeros(
                            (int(packed.num_prg_tokens),),
                            dtype=torch.bool,
                            device=target.device,
                        )
                    success_targets.append(target.to(device=train_device, dtype=success_logits.dtype))
                    success_masks.append(mask.to(device=train_device, dtype=torch.bool))
                    service_protection_masks.append(service_protected.to(device=train_device, dtype=torch.bool))
                success_target = torch.stack(success_targets, dim=0)
                success_mask = torch.stack(success_masks, dim=0)
                rank_target = torch.stack(rank_targets, dim=0) if rank_targets else None
                rescue_target = torch.stack(rescue_targets, dim=0) if rescue_targets else None
                service_protection_mask = torch.stack(service_protection_masks, dim=0)
                success_weight = success_mask.to(dtype=success_logits.dtype)
                success_weight_sum = success_weight.sum().clamp_min(1.0)
                valid_prg = success_mask.any(dim=-1)
                valid_prg_weight = valid_prg.to(dtype=success_logits.dtype)
                valid_prg_weight_sum = valid_prg_weight.sum().clamp_min(1.0)
                max_success_logit = success_logits.masked_fill(~success_mask, -1.0e9).max(dim=-1).values
                max_success_target = success_target.masked_fill(~success_mask, -1.0).max(dim=-1).values
                candidate_success_max_logit = (max_success_logit * valid_prg_weight).sum() / valid_prg_weight_sum
                candidate_success_max_target = (max_success_target * valid_prg_weight).sum() / valid_prg_weight_sum
                valid_max_targets = max_success_target[valid_prg].to(dtype=torch.float32)
                valid_max_logits = max_success_logit[valid_prg].to(dtype=torch.float32)
                if valid_max_targets.numel() > 0:
                    candidate_success_max_target_p10 = torch.quantile(valid_max_targets, 0.10)
                    candidate_success_max_target_p50 = torch.quantile(valid_max_targets, 0.50)
                    candidate_success_max_target_p90 = torch.quantile(valid_max_targets, 0.90)
                if valid_max_logits.numel() > 0:
                    candidate_success_max_logit_p10 = torch.quantile(valid_max_logits, 0.10)
                    candidate_success_max_logit_p50 = torch.quantile(valid_max_logits, 0.50)
                    candidate_success_max_logit_p90 = torch.quantile(valid_max_logits, 0.90)

                success_prob = torch.sigmoid(success_logits)
                prob_error = success_prob - success_target
                candidate_success_prob_mae = (
                    prob_error.abs() * success_weight
                ).sum() / success_weight_sum
                candidate_success_prob_brier = (
                    prob_error.square() * success_weight
                ).sum() / success_weight_sum
                candidate_success_prob_mean = (success_prob * success_weight).sum() / success_weight_sum
                candidate_success_target_mean = (success_target * success_weight).sum() / success_weight_sum

                if candidate_success_aux_coef > 0.0:
                    success_loss = F.binary_cross_entropy_with_logits(
                        success_logits,
                        success_target,
                        reduction="none",
                    )
                    candidate_success_aux_loss = (success_loss * success_weight).sum() / success_weight_sum

                if candidate_success_calib_coef > 0.0:
                    eps = min(max(float(candidate_success_target_eps), 1.0e-4), 0.49)
                    target_logit = torch.logit(success_target.clamp(min=eps, max=1.0 - eps))
                    success_calib = F.smooth_l1_loss(
                        success_logits,
                        target_logit,
                        beta=1.0,
                        reduction="none",
                    )
                    candidate_success_calib_loss = (
                        success_calib * success_weight
                    ).sum() / success_weight_sum

                if candidate_success_max_calib_coef > 0.0:
                    eps = min(max(float(candidate_success_target_eps), 1.0e-4), 0.49)
                    max_target_logit = torch.logit(max_success_target.clamp(min=eps, max=1.0 - eps))
                    max_calib_loss = F.smooth_l1_loss(
                        max_success_logit,
                        max_target_logit,
                        beta=1.0,
                        reduction="none",
                    )
                    candidate_success_max_calib_loss = (
                        max_calib_loss * valid_prg_weight
                    ).sum() / valid_prg_weight_sum

                if candidate_rescue_aux_coef > 0.0 and rescue_target is not None:
                    prg_logits_masked, prg_valid = _masked_candidate_prg_logits_batch(
                        actor_out,
                        graphs_mb,
                        ue_action_class=ue_action_mb,
                    )
                    main_logits = prg_logits_masked[:, :, :-1]
                    main_valid = prg_valid[:, :, :-1] & success_mask
                    rescue_target_clamped = rescue_target.clamp(min=0.0, max=1.0)
                    rescue_mask_weight = main_valid.to(dtype=success_logits.dtype)
                    rescue_weight = (
                        rescue_mask_weight
                        * (1.0 + max(0.0, float(candidate_rescue_pos_weight)) * rescue_target_clamped)
                    )
                    rescue_weight_sum = rescue_weight.sum().clamp_min(1.0)
                    if bool(main_valid.any().item()):
                        rescue_loss = F.binary_cross_entropy_with_logits(
                            main_logits,
                            rescue_target_clamped,
                            reduction="none",
                        )
                        candidate_rescue_aux_loss = (rescue_loss * rescue_weight).sum() / rescue_weight_sum
                        rescue_prob = torch.sigmoid(main_logits)
                        rescue_mask_sum = rescue_mask_weight.sum().clamp_min(1.0)
                        candidate_rescue_target_mean = (
                            rescue_target_clamped * rescue_mask_weight
                        ).sum() / rescue_mask_sum
                        candidate_rescue_positive_ratio = (
                            ((rescue_target_clamped > 0.2) & main_valid).to(dtype=success_logits.dtype).sum()
                            / rescue_mask_sum
                        )
                        candidate_rescue_prob_mae = (
                            (rescue_prob - rescue_target_clamped).abs() * rescue_mask_weight
                        ).sum() / rescue_mask_sum
                        candidate_rescue_mask_ratio = main_valid.to(dtype=torch.float32).mean()

                if candidate_pairwise_rank_coef > 0.0 and rank_target is not None:
                    prg_logits_masked, prg_valid = _masked_candidate_prg_logits_batch(
                        actor_out,
                        graphs_mb,
                        ue_action_class=ue_action_mb,
                    )
                    main_logits = prg_logits_masked[:, :, :-1]
                    main_valid = prg_valid[:, :, :-1] & success_mask
                    rank_gap = rank_target.unsqueeze(-1) - rank_target.unsqueeze(-2)
                    pair_valid = (
                        main_valid.unsqueeze(-1)
                        & main_valid.unsqueeze(-2)
                        & (rank_gap >= float(candidate_pairwise_rank_min_gap))
                    )
                    if bool(pair_valid.any().item()):
                        score_diff = main_logits.unsqueeze(-1) - main_logits.unsqueeze(-2)
                        pair_weight = pair_valid.to(dtype=success_logits.dtype) * rank_gap.clamp_min(0.0).clamp_max(3.0)
                        pair_weight_sum = pair_weight.sum().clamp_min(1.0)
                        rank_loss = F.softplus(float(candidate_pairwise_rank_margin) - score_diff)
                        candidate_pairwise_rank_loss = (rank_loss * pair_weight).sum() / pair_weight_sum
                        comparable = (
                            main_valid.unsqueeze(-1)
                            & main_valid.unsqueeze(-2)
                            & (rank_gap.abs() > 0.0)
                        )
                        candidate_pairwise_rank_pair_ratio = (
                            pair_valid.to(dtype=torch.float32).sum()
                            / comparable.to(dtype=torch.float32).sum().clamp_min(1.0)
                        )

                if candidate_decode_goodput_coef > 0.0:
                    decode_targets: List[torch.Tensor] = []
                    decode_masks: List[torch.Tensor] = []
                    target_shape = (int(success_logits.shape[1]), int(success_logits.shape[2]))
                    for sample_idx in idx:
                        step = rollout[sample_idx]
                        if (
                            step.candidate_decode_goodput_target is None
                            or step.candidate_decode_goodput_mask is None
                            or tuple(step.candidate_decode_goodput_target.shape) != target_shape
                            or tuple(step.candidate_decode_goodput_mask.shape) != target_shape
                        ):
                            decode_targets.append(torch.zeros(target_shape, dtype=success_logits.dtype, device=train_device))
                            decode_masks.append(torch.zeros(target_shape, dtype=torch.bool, device=train_device))
                            continue
                        decode_targets.append(
                            step.candidate_decode_goodput_target.to(device=train_device, dtype=success_logits.dtype)
                        )
                        decode_masks.append(
                            step.candidate_decode_goodput_mask.to(device=train_device, dtype=torch.bool)
                        )
                    decode_target = torch.stack(decode_targets, dim=0).clamp(min=0.0, max=1.0)
                    decode_mask = torch.stack(decode_masks, dim=0) & success_mask
                    prg_logits_masked, prg_valid = _masked_candidate_prg_logits_batch(
                        actor_out,
                        graphs_mb,
                        ue_action_class=ue_action_mb,
                    )
                    main_logits = prg_logits_masked[:, :, :-1]
                    main_valid = prg_valid[:, :, :-1] & decode_mask
                    decode_mask_weight = main_valid.to(dtype=success_logits.dtype)
                    decode_weight_sum = decode_mask_weight.sum().clamp_min(1.0)
                    if bool(main_valid.any().item()):
                        target_weight = (
                            decode_mask_weight
                            * (1.0 + max(0.0, float(candidate_decode_goodput_pos_weight)) * decode_target)
                        )
                        target_weight_sum = target_weight.sum().clamp_min(1.0)
                        decode_loss = F.binary_cross_entropy_with_logits(
                            main_logits,
                            decode_target,
                            reduction="none",
                        )
                        candidate_decode_goodput_loss = (decode_loss * target_weight).sum() / target_weight_sum
                        decode_prob = torch.sigmoid(main_logits)
                        candidate_decode_goodput_target_mean = (
                            decode_target * decode_mask_weight
                        ).sum() / decode_weight_sum
                        candidate_decode_goodput_positive_ratio = (
                            ((decode_target > 0.0) & main_valid).to(dtype=success_logits.dtype).sum()
                            / decode_weight_sum
                        )
                        candidate_decode_goodput_prob_mean = (
                            decode_prob * decode_mask_weight
                        ).sum() / decode_weight_sum
                        candidate_decode_goodput_prob_mae = (
                            (decode_prob - decode_target).abs() * decode_mask_weight
                        ).sum() / decode_weight_sum
                        candidate_decode_goodput_mask_ratio = main_valid.to(dtype=torch.float32).mean()
                        candidate_decode_goodput_logit_mean = (
                            main_logits * decode_mask_weight
                        ).sum() / decode_weight_sum

                if candidate_argmax_decode_coef > 0.0:
                    prg_logits_masked, prg_valid = _masked_candidate_prg_logits_batch(
                        actor_out,
                        graphs_mb,
                        ue_action_class=ue_action_mb,
                    )
                    main_logits = prg_logits_masked[:, :, :-1]
                    argmax_targets: List[torch.Tensor] = []
                    argmax_masks: List[torch.Tensor] = []
                    argmax_kept_masks: List[torch.Tensor] = []
                    row_idx = torch.arange(int(success_logits.shape[1]), dtype=torch.long, device=train_device)
                    rank_mode = str(candidate_argmax_decode_headroom_rank_mode).lower()
                    if rank_mode not in ("stable", "utility", "logit", "phy"):
                        rank_mode = "stable"
                    for batch_idx, graph in enumerate(graphs_mb):
                        packed = pack_prg_candidates(graph, max_candidates=max_candidates, fixed_width=True)
                        target = torch.zeros(
                            (int(success_logits.shape[1]), int(success_logits.shape[2])),
                            dtype=success_logits.dtype,
                            device=train_device,
                        )
                        mask = torch.zeros_like(target, dtype=torch.bool)
                        kept_mask = torch.zeros_like(target, dtype=torch.bool)
                        candidate_valid, _selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(
                            graph,
                            packed,
                            ue_action_mb[batch_idx].detach().cpu(),
                        )
                        decoded, choice, _deploy_metrics = select_and_decode_budgeted_candidate_actions(
                            graph,
                            packed,
                            candidate_valid,
                            ue_action_mb[batch_idx].detach().cpu(),
                            prg_logits_masked[batch_idx].detach(),
                            mode="greedy",
                            demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
                            marginal_headroom_slack_bytes=float(candidate_marginal_headroom_slack_bytes),
                            hard_cap=bool(int(candidate_budgeted_hard_cap)),
                            usage_penalty=float(candidate_budgeted_usage_penalty),
                            overcap_penalty=float(candidate_budgeted_overcap_penalty),
                            first_service_bonus=float(candidate_budgeted_first_service_bonus),
                            tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                            tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                            tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                            tail_bias_max=float(candidate_budgeted_tail_bias_max),
                            rescue_bias_coef=float(candidate_budgeted_rescue_bias_coef),
                            rescue_bias_service_target=float(candidate_budgeted_rescue_bias_service_target),
                            rescue_bias_ttl_guard_ms=float(candidate_budgeted_rescue_bias_ttl_guard_ms),
                            rescue_bias_max=float(candidate_budgeted_rescue_bias_max),
                            completion_bonus=float(candidate_budgeted_completion_bonus),
                            completion_progress_weight=float(candidate_budgeted_completion_progress_weight),
                            completion_max_score=float(candidate_budgeted_completion_max_score),
                            tail_quota_enable=bool(int(candidate_budgeted_tail_quota_enable)),
                            tail_quota_min_score=float(candidate_budgeted_tail_quota_min_score),
                            tail_quota_topk_ues=int(candidate_budgeted_tail_quota_topk_ues),
                            tail_quota_prgs_per_ue=int(candidate_budgeted_tail_quota_prgs_per_ue),
                            tail_quota_bonus=float(candidate_budgeted_tail_quota_bonus),
                            tail_quota_order_bonus=float(candidate_budgeted_tail_quota_order_bonus),
                            tail_quota_max_score=float(candidate_budgeted_tail_quota_max_score),
                            tail_quota_risk_max=float(candidate_budgeted_tail_quota_risk_max),
                            seq_state_weights=(
                                actor_out.get("candidate_seq_state_weights")[batch_idx].detach()
                                if actor_out.get("candidate_seq_state_weights") is not None
                                else None
                            ),
                            seq_state_coef=float(candidate_seq_state_coef),
                            seq_fail_risk_feature_offset=int(candidate_seq_fail_risk_feature_offset),
                            safe_fill_max_per_cell=int(safe_fill_max_per_cell),
                            safe_fill_min_quality_db=float(safe_fill_min_quality_db),
                            safe_fill_max_quality_gap_db=float(safe_fill_max_quality_gap_db),
                            safe_fill_max_prg_risk=float(safe_fill_max_prg_risk),
                            safe_fill_min_backlog_bytes=float(safe_fill_min_backlog_bytes),
                            safe_fill_max_ue_tb_err=float(safe_fill_max_ue_tb_err),
                            headroom_rank_mode=rank_mode,
                            candidate_rank_scores=prg_logits_masked[batch_idx, :, :max_candidates].detach(),
                        )
                        choice = choice.to(device=train_device, dtype=torch.long)
                        choice_is_main = (choice >= 0) & (choice < max_candidates)
                        if bool(choice_is_main.any().item()):
                            choice_safe = choice.clamp(min=0, max=max(0, max_candidates - 1))
                            valid_choice = choice_is_main & prg_valid[batch_idx, :, :-1].gather(
                                1,
                                choice_safe.unsqueeze(-1),
                            ).squeeze(-1)
                            if bool(valid_choice.any().item()):
                                mask[row_idx[valid_choice], choice_safe[valid_choice]] = True
                                chosen_ue = packed.ue_ids.to(device=train_device, dtype=torch.long).gather(
                                    1,
                                    choice_safe.unsqueeze(-1),
                                ).squeeze(-1)
                                decoded_ue = decoded.chosen_ue_per_prg.to(device=train_device, dtype=torch.long)
                                kept = valid_choice & (decoded_ue == chosen_ue)
                                if bool(kept.any().item()):
                                    kept_mask[row_idx[kept], choice_safe[kept]] = True
                                    target[row_idx[kept], choice_safe[kept]] = success_target[
                                        batch_idx,
                                        row_idx[kept],
                                        choice_safe[kept],
                                    ]
                        argmax_targets.append(target)
                        argmax_masks.append(mask)
                        argmax_kept_masks.append(kept_mask)
                    argmax_target = torch.stack(argmax_targets, dim=0).clamp(min=0.0, max=1.0)
                    argmax_mask = torch.stack(argmax_masks, dim=0) & prg_valid[:, :, :-1] & success_mask
                    argmax_kept_mask = torch.stack(argmax_kept_masks, dim=0) & argmax_mask
                    argmax_mask_weight = argmax_mask.to(dtype=success_logits.dtype)
                    argmax_weight_sum = argmax_mask_weight.sum().clamp_min(1.0)
                    if bool(argmax_mask.any().item()):
                        argmax_weight = (
                            argmax_mask_weight
                            * (1.0 + max(0.0, float(candidate_argmax_decode_pos_weight)) * argmax_target)
                        )
                        argmax_weight_sum = argmax_weight.sum().clamp_min(1.0)
                        argmax_loss = F.binary_cross_entropy_with_logits(
                            main_logits,
                            argmax_target,
                            reduction="none",
                        )
                        candidate_argmax_decode_loss = (argmax_loss * argmax_weight).sum() / argmax_weight_sum
                        argmax_prob = torch.sigmoid(main_logits)
                        candidate_argmax_decode_target_mean = (
                            argmax_target * argmax_mask_weight
                        ).sum() / argmax_mask_weight.sum().clamp_min(1.0)
                        candidate_argmax_decode_positive_ratio = (
                            ((argmax_target > 0.0) & argmax_mask).to(dtype=success_logits.dtype).sum()
                            / argmax_mask_weight.sum().clamp_min(1.0)
                        )
                        candidate_argmax_decode_keep_ratio = (
                            argmax_kept_mask.to(dtype=success_logits.dtype).sum()
                            / argmax_mask_weight.sum().clamp_min(1.0)
                        )
                        candidate_argmax_decode_prob_mae = (
                            (argmax_prob - argmax_target).abs() * argmax_mask_weight
                        ).sum() / argmax_mask_weight.sum().clamp_min(1.0)
                        candidate_argmax_decode_mask_ratio = argmax_mask.to(dtype=torch.float32).mean()

                if candidate_blank_target_coef > 0.0:
                    prg_logits_masked, prg_valid = _masked_candidate_prg_logits_batch(
                        actor_out,
                        graphs_mb,
                        ue_action_class=ue_action_mb,
                    )
                    main_valid = prg_valid[:, :, :-1]
                    has_main_valid = main_valid.any(dim=-1)
                    max_main_logit = prg_logits_masked[:, :, :-1].max(dim=-1).values
                    blank_logit = prg_logits_masked[:, :, -1]
                    blank_class = int(prg_logits_masked.shape[-1]) - 1
                    shadow_choice = prg_logits_masked.argmax(dim=-1)
                    candidate_shadow_blank_ratio = (shadow_choice == blank_class).to(dtype=torch.float32).mean()

                    threshold = float(candidate_blank_target_threshold)
                    deadband = max(0.0, float(candidate_blank_target_deadband))
                    raw_blank_target = max_success_target < threshold
                    blank_target = raw_blank_target & ~service_protection_mask
                    confident = (max_success_target - threshold).abs() >= deadband
                    raw_decision_valid = valid_prg & has_main_valid & confident
                    pos_weight = max(0.0, float(candidate_blank_target_pos_weight))
                    fill_weight = max(0.0, float(candidate_blank_target_fill_weight))
                    decision_logit = blank_logit - max_main_logit
                    margin = float(candidate_blank_target_margin)
                    raw_decision_weight = raw_decision_valid.to(dtype=success_logits.dtype)
                    raw_decision_weight_sum = raw_decision_weight.sum().clamp_min(1.0)
                    service_fill_anchor = torch.zeros_like(raw_decision_valid)
                    if int(candidate_blank_target_positive_only) == 1:
                        fill_anchor_weight_value = max(0.0, float(candidate_blank_target_service_fill_weight))
                        fill_anchor_min_success = float(candidate_blank_target_service_fill_min_success)
                        service_fill_anchor = raw_decision_valid & (~blank_target) & (
                            service_protection_mask | (max_success_target >= fill_anchor_min_success)
                        )
                        blank_decision_weight = (
                            raw_decision_weight
                            * blank_target.to(dtype=success_logits.dtype)
                            * pos_weight
                        )
                        fill_anchor_weight = (
                            service_fill_anchor.to(dtype=success_logits.dtype)
                            * fill_anchor_weight_value
                        )
                        blank_loss = F.softplus(margin - decision_logit)
                        fill_loss = F.softplus(
                            float(candidate_blank_target_service_fill_margin) + decision_logit
                        )
                        decision_weight = blank_decision_weight + fill_anchor_weight
                        decision_weight_sum = decision_weight.sum().clamp_min(1.0)
                        candidate_blank_target_loss = (
                            blank_loss * blank_decision_weight + fill_loss * fill_anchor_weight
                        ).sum() / decision_weight_sum
                    else:
                        target_weight = torch.where(
                            blank_target,
                            torch.full_like(max_success_target, pos_weight),
                            torch.full_like(max_success_target, fill_weight),
                        )
                        decision_weight = raw_decision_weight * target_weight
                        decision_weight_sum = decision_weight.sum().clamp_min(1.0)
                        signed_target = torch.where(
                            blank_target,
                            torch.ones_like(decision_logit),
                            -torch.ones_like(decision_logit),
                        )
                        blank_loss = F.softplus(margin - signed_target * decision_logit)
                        candidate_blank_target_loss = (blank_loss * decision_weight).sum() / decision_weight_sum
                    candidate_blank_target_ratio = (
                        blank_target.to(dtype=success_logits.dtype) * raw_decision_weight
                    ).sum() / raw_decision_weight_sum
                    candidate_blank_target_raw_ratio = (
                        raw_blank_target.to(dtype=success_logits.dtype) * raw_decision_weight
                    ).sum() / raw_decision_weight_sum
                    candidate_blank_target_service_block_ratio = (
                        (raw_blank_target & service_protection_mask).to(dtype=success_logits.dtype)
                        * raw_decision_weight
                    ).sum() / raw_decision_weight_sum
                    candidate_blank_target_service_fill_ratio = (
                        service_fill_anchor.to(dtype=success_logits.dtype) * raw_decision_weight
                    ).sum() / raw_decision_weight_sum
                    candidate_blank_target_weighted_ratio = (
                        blank_target.to(dtype=success_logits.dtype) * decision_weight
                    ).sum() / decision_weight_sum
                    argmax_decision_valid = valid_prg & has_main_valid
                    decision_values = decision_logit[argmax_decision_valid].to(dtype=torch.float32)
                    if decision_values.numel() > 0:
                        candidate_blank_decision_logit_mean = decision_values.mean()
                        candidate_blank_decision_logit_p10 = torch.quantile(decision_values, 0.10)
                        candidate_blank_decision_logit_p50 = torch.quantile(decision_values, 0.50)
                        candidate_blank_decision_logit_p90 = torch.quantile(decision_values, 0.90)

            actor_loss = (
                policy_loss
                - entropy_coef * entropy
                + teacher_aux_coef * teacher_aux_loss
                + candidate_success_aux_coef * candidate_success_aux_loss
                + candidate_success_calib_coef * candidate_success_calib_loss
                + candidate_success_max_calib_coef * candidate_success_max_calib_loss
                + candidate_pairwise_rank_coef * candidate_pairwise_rank_loss
                + candidate_rescue_aux_coef * candidate_rescue_aux_loss
                + candidate_decode_goodput_coef * candidate_decode_goodput_loss
                + candidate_argmax_decode_coef * candidate_argmax_decode_loss
                + candidate_blank_target_coef * candidate_blank_target_loss
            )
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
            total_candidate_success_aux_loss += float(candidate_success_aux_loss.item())
            total_candidate_success_calib_loss += float(candidate_success_calib_loss.item())
            total_candidate_success_max_calib_loss += float(candidate_success_max_calib_loss.item())
            total_candidate_pairwise_rank_loss += float(candidate_pairwise_rank_loss.item())
            total_candidate_pairwise_rank_pair_ratio += float(candidate_pairwise_rank_pair_ratio.item())
            total_candidate_rescue_aux_loss += float(candidate_rescue_aux_loss.item())
            total_candidate_rescue_target_mean += float(candidate_rescue_target_mean.item())
            total_candidate_rescue_positive_ratio += float(candidate_rescue_positive_ratio.item())
            total_candidate_rescue_prob_mae += float(candidate_rescue_prob_mae.item())
            total_candidate_rescue_mask_ratio += float(candidate_rescue_mask_ratio.item())
            total_candidate_decode_goodput_loss += float(candidate_decode_goodput_loss.item())
            total_candidate_decode_goodput_target_mean += float(candidate_decode_goodput_target_mean.item())
            total_candidate_decode_goodput_positive_ratio += float(candidate_decode_goodput_positive_ratio.item())
            total_candidate_decode_goodput_prob_mean += float(candidate_decode_goodput_prob_mean.item())
            total_candidate_decode_goodput_prob_mae += float(candidate_decode_goodput_prob_mae.item())
            total_candidate_decode_goodput_mask_ratio += float(candidate_decode_goodput_mask_ratio.item())
            total_candidate_decode_goodput_logit_mean += float(candidate_decode_goodput_logit_mean.item())
            total_candidate_argmax_decode_loss += float(candidate_argmax_decode_loss.item())
            total_candidate_argmax_decode_target_mean += float(candidate_argmax_decode_target_mean.item())
            total_candidate_argmax_decode_positive_ratio += float(candidate_argmax_decode_positive_ratio.item())
            total_candidate_argmax_decode_keep_ratio += float(candidate_argmax_decode_keep_ratio.item())
            total_candidate_argmax_decode_prob_mae += float(candidate_argmax_decode_prob_mae.item())
            total_candidate_argmax_decode_mask_ratio += float(candidate_argmax_decode_mask_ratio.item())
            total_candidate_blank_target_loss += float(candidate_blank_target_loss.item())
            total_candidate_blank_target_ratio += float(candidate_blank_target_ratio.item())
            total_candidate_blank_target_raw_ratio += float(candidate_blank_target_raw_ratio.item())
            total_candidate_blank_target_service_block_ratio += float(
                candidate_blank_target_service_block_ratio.item()
            )
            total_candidate_blank_target_service_fill_ratio += float(
                candidate_blank_target_service_fill_ratio.item()
            )
            total_candidate_blank_target_weighted_ratio += float(candidate_blank_target_weighted_ratio.item())
            total_candidate_shadow_blank_ratio += float(candidate_shadow_blank_ratio.item())
            total_candidate_success_max_logit += float(candidate_success_max_logit.item())
            total_candidate_success_max_target += float(candidate_success_max_target.item())
            total_candidate_success_max_target_p10 += float(candidate_success_max_target_p10.item())
            total_candidate_success_max_target_p50 += float(candidate_success_max_target_p50.item())
            total_candidate_success_max_target_p90 += float(candidate_success_max_target_p90.item())
            total_candidate_success_max_logit_p10 += float(candidate_success_max_logit_p10.item())
            total_candidate_success_max_logit_p50 += float(candidate_success_max_logit_p50.item())
            total_candidate_success_max_logit_p90 += float(candidate_success_max_logit_p90.item())
            total_candidate_success_prob_mae += float(candidate_success_prob_mae.item())
            total_candidate_success_prob_brier += float(candidate_success_prob_brier.item())
            total_candidate_success_prob_mean += float(candidate_success_prob_mean.item())
            total_candidate_success_target_mean += float(candidate_success_target_mean.item())
            total_candidate_blank_decision_logit_mean += float(candidate_blank_decision_logit_mean.item())
            total_candidate_blank_decision_logit_p10 += float(candidate_blank_decision_logit_p10.item())
            total_candidate_blank_decision_logit_p50 += float(candidate_blank_decision_logit_p50.item())
            total_candidate_blank_decision_logit_p90 += float(candidate_blank_decision_logit_p90.item())
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
        "candidate_success_aux_loss": total_candidate_success_aux_loss / denom,
        "candidate_success_aux_coef": float(candidate_success_aux_coef),
        "candidate_success_calib_loss": total_candidate_success_calib_loss / denom,
        "candidate_success_calib_coef": float(candidate_success_calib_coef),
        "candidate_success_max_calib_loss": total_candidate_success_max_calib_loss / denom,
        "candidate_success_max_calib_coef": float(candidate_success_max_calib_coef),
        "candidate_pairwise_rank_loss": total_candidate_pairwise_rank_loss / denom,
        "candidate_pairwise_rank_coef": float(candidate_pairwise_rank_coef),
        "candidate_pairwise_rank_pair_ratio": total_candidate_pairwise_rank_pair_ratio / denom,
        "candidate_pairwise_rank_margin": float(candidate_pairwise_rank_margin),
        "candidate_pairwise_rank_min_gap": float(candidate_pairwise_rank_min_gap),
        "candidate_pairwise_rank_success_weight": float(candidate_pairwise_rank_success_weight),
        "candidate_pairwise_rank_service_weight": float(candidate_pairwise_rank_service_weight),
        "candidate_pairwise_rank_risk_weight": float(candidate_pairwise_rank_risk_weight),
        "candidate_pairwise_rank_fail_risk_weight": float(candidate_pairwise_rank_fail_risk_weight),
        "candidate_pairwise_rank_rescue_weight": float(candidate_pairwise_rank_rescue_weight),
        "candidate_rescue_aux_loss": total_candidate_rescue_aux_loss / denom,
        "candidate_rescue_aux_coef": float(candidate_rescue_aux_coef),
        "candidate_rescue_pos_weight": float(candidate_rescue_pos_weight),
        "candidate_rescue_target_mean": total_candidate_rescue_target_mean / denom,
        "candidate_rescue_positive_ratio": total_candidate_rescue_positive_ratio / denom,
        "candidate_rescue_prob_mae": total_candidate_rescue_prob_mae / denom,
        "candidate_rescue_mask_ratio": total_candidate_rescue_mask_ratio / denom,
        "candidate_rescue_service_target": float(candidate_rescue_service_target),
        "candidate_rescue_ttl_guard_ms": float(candidate_rescue_ttl_guard_ms),
        "candidate_rescue_max_target": float(candidate_rescue_max_target),
        "candidate_decode_goodput_loss": total_candidate_decode_goodput_loss / denom,
        "candidate_decode_goodput_coef": float(candidate_decode_goodput_coef),
        "candidate_decode_goodput_pos_weight": float(candidate_decode_goodput_pos_weight),
        "candidate_decode_goodput_target_mean": total_candidate_decode_goodput_target_mean / denom,
        "candidate_decode_goodput_positive_ratio": total_candidate_decode_goodput_positive_ratio / denom,
        "candidate_decode_goodput_prob_mean": total_candidate_decode_goodput_prob_mean / denom,
        "candidate_decode_goodput_prob_mae": total_candidate_decode_goodput_prob_mae / denom,
        "candidate_decode_goodput_mask_ratio": total_candidate_decode_goodput_mask_ratio / denom,
        "candidate_decode_goodput_logit_mean": total_candidate_decode_goodput_logit_mean / denom,
        "candidate_argmax_decode_loss": total_candidate_argmax_decode_loss / denom,
        "candidate_budgeted_greedy_decode_loss": total_candidate_argmax_decode_loss / denom,
        "candidate_argmax_decode_coef": float(candidate_argmax_decode_coef),
        "candidate_argmax_decode_pos_weight": float(candidate_argmax_decode_pos_weight),
        "candidate_argmax_decode_target_mean": total_candidate_argmax_decode_target_mean / denom,
        "candidate_argmax_decode_positive_ratio": total_candidate_argmax_decode_positive_ratio / denom,
        "candidate_argmax_decode_keep_ratio": total_candidate_argmax_decode_keep_ratio / denom,
        "candidate_argmax_decode_prob_mae": total_candidate_argmax_decode_prob_mae / denom,
        "candidate_argmax_decode_mask_ratio": total_candidate_argmax_decode_mask_ratio / denom,
        "candidate_blank_target_loss": total_candidate_blank_target_loss / denom,
        "candidate_blank_target_coef": float(candidate_blank_target_coef),
        "candidate_blank_target_ratio": total_candidate_blank_target_ratio / denom,
        "candidate_blank_target_raw_ratio": total_candidate_blank_target_raw_ratio / denom,
        "candidate_blank_target_service_block_ratio": total_candidate_blank_target_service_block_ratio / denom,
        "candidate_blank_target_service_fill_ratio": total_candidate_blank_target_service_fill_ratio / denom,
        "candidate_blank_target_weighted_ratio": total_candidate_blank_target_weighted_ratio / denom,
        "candidate_blank_target_pos_weight": float(candidate_blank_target_pos_weight),
        "candidate_blank_target_fill_weight": float(candidate_blank_target_fill_weight),
        "candidate_blank_target_positive_only": float(candidate_blank_target_positive_only),
        "candidate_blank_target_service_guard": float(candidate_blank_target_service_guard),
        "candidate_blank_target_service_min_success": float(candidate_blank_target_service_min_success),
        "candidate_blank_target_service_min_debt": float(candidate_blank_target_service_min_debt),
        "candidate_blank_target_service_min_hol_ms": float(candidate_blank_target_service_min_hol_ms),
        "candidate_blank_target_service_max_scheduled_ratio": float(
            candidate_blank_target_service_max_scheduled_ratio
        ),
        "candidate_blank_target_service_fill_weight": float(candidate_blank_target_service_fill_weight),
        "candidate_blank_target_service_fill_margin": float(candidate_blank_target_service_fill_margin),
        "candidate_blank_target_service_fill_min_success": float(candidate_blank_target_service_fill_min_success),
        "candidate_shadow_blank_ratio": total_candidate_shadow_blank_ratio / denom,
        "candidate_success_max_logit_mean": total_candidate_success_max_logit / denom,
        "candidate_success_max_target_mean": total_candidate_success_max_target / denom,
        "candidate_success_max_target_p10": total_candidate_success_max_target_p10 / denom,
        "candidate_success_max_target_p50": total_candidate_success_max_target_p50 / denom,
        "candidate_success_max_target_p90": total_candidate_success_max_target_p90 / denom,
        "candidate_success_max_logit_p10": total_candidate_success_max_logit_p10 / denom,
        "candidate_success_max_logit_p50": total_candidate_success_max_logit_p50 / denom,
        "candidate_success_max_logit_p90": total_candidate_success_max_logit_p90 / denom,
        "candidate_success_prob_mae": total_candidate_success_prob_mae / denom,
        "candidate_success_prob_brier": total_candidate_success_prob_brier / denom,
        "candidate_success_prob_mean": total_candidate_success_prob_mean / denom,
        "candidate_success_target_mean": total_candidate_success_target_mean / denom,
        "candidate_blank_decision_logit_mean": total_candidate_blank_decision_logit_mean / denom,
        "candidate_blank_decision_logit_p10": total_candidate_blank_decision_logit_p10 / denom,
        "candidate_blank_decision_logit_p50": total_candidate_blank_decision_logit_p50 / denom,
        "candidate_blank_decision_logit_p90": total_candidate_blank_decision_logit_p90 / denom,
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
        f"success_aux={summary['candidate_success_aux_loss']:.4f}*{summary['candidate_success_aux_coef']:.3f} "
        f"succ_calib={summary['candidate_success_calib_loss']:.4f}*{summary['candidate_success_calib_coef']:.3f} "
        f"max_calib={summary['candidate_success_max_calib_loss']:.4f}*{summary['candidate_success_max_calib_coef']:.3f} "
        f"rank={summary['candidate_pairwise_rank_loss']:.4f}*{summary['candidate_pairwise_rank_coef']:.3f} "
        f"rescue={summary['candidate_rescue_aux_loss']:.4f}*{summary['candidate_rescue_aux_coef']:.3f} "
        f"rescueT={summary['candidate_rescue_target_mean']:.3f}/"
        f"{summary['candidate_rescue_positive_ratio']:.3f} "
        f"decode_gp={summary['candidate_decode_goodput_loss']:.4f}*{summary['candidate_decode_goodput_coef']:.3f} "
        f"decodeT={summary['candidate_decode_goodput_target_mean']:.3f}/"
        f"{summary['candidate_decode_goodput_positive_ratio']:.3f} "
        f"deploy_dec={summary['candidate_budgeted_greedy_decode_loss']:.4f}*"
        f"{summary['candidate_argmax_decode_coef']:.3f} "
        f"deployKeep={summary['candidate_argmax_decode_keep_ratio']:.3f} "
        f"blank_tgt={summary['candidate_blank_target_loss']:.4f}*{summary['candidate_blank_target_coef']:.3f} "
        f"blankT={summary['candidate_blank_target_ratio']:.3f}/{summary['candidate_blank_target_weighted_ratio']:.3f} "
        f"raw/block={summary['candidate_blank_target_raw_ratio']:.3f}/"
        f"{summary['candidate_blank_target_service_block_ratio']:.3f}/"
        f"{summary['candidate_blank_target_service_fill_ratio']:.3f} "
        f"shadowBlank={summary['candidate_shadow_blank_ratio']:.3f} "
        f"maxT[p10,p50,p90]={summary['candidate_success_max_target_p10']:.3f},"
        f"{summary['candidate_success_max_target_p50']:.3f},{summary['candidate_success_max_target_p90']:.3f} "
        f"succMAE={summary['candidate_success_prob_mae']:.3f} "
        f"blankD[p50,p90]={summary['candidate_blank_decision_logit_p50']:.2f},"
        f"{summary['candidate_blank_decision_logit_p90']:.2f} "
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


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw or "").split(","):
        item = part.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def _row_float(row: Dict[str, float], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _multi_seed_rate_window_metrics(
    history: Sequence[Dict[str, float]],
    args: argparse.Namespace,
) -> Dict[str, float]:
    requested_seeds = sorted(set(_parse_int_list(getattr(args, "seed_list", ""))))
    window_iters = int(getattr(args, "best_multi_seed_window_iters", 0))
    if window_iters <= 0:
        window_iters = max(1, len(requested_seeds))
    rows = list(history[-window_iters:])
    observed_seeds = [
        int(round(_row_float(row, "topology_seed", -1.0)))
        for row in rows
        if _row_float(row, "topology_seed", -1.0) >= 0.0
    ]
    observed_set = sorted(set(observed_seeds))
    if requested_seeds:
        covered_set = sorted(set(observed_set).intersection(requested_seeds))
    else:
        covered_set = observed_set
    min_covered = int(getattr(args, "best_multi_seed_min_covered_seeds", 0))
    if min_covered <= 0:
        min_covered = max(1, len(requested_seeds)) if requested_seeds else 1
    coverage_pass = len(rows) >= window_iters and len(covered_set) >= min_covered

    def values(key: str, default: float = 0.0) -> List[float]:
        return [_row_float(row, key, default) for row in rows]

    goodput = values("rollout_goodput_mbps_mean")
    tb_err = values("rollout_tb_err_rate_mean")
    expiry = [
        _row_float(row, "rollout_expiry_reward_rate_mean", _row_float(row, "rollout_expiry_drop_rate_mean"))
        for row in rows
    ]
    packet_rate = values("rollout_packet_effective_service_rate_mbps_mean")
    packet_per_rate = values("rollout_packet_effective_service_rate_per_packet_mean_mbps_mean")
    packet_delay = [
        _row_float(row, "rollout_packet_delay_weighted_mean_ms", _row_float(row, "rollout_packet_delay_mean_ms_mean"))
        for row in rows
    ]
    ue_macro_rate_proxy = values("rollout_ue_macro_rate_proxy_mbps_mean")
    ue_p10_rate_proxy = values("rollout_ue_p10_rate_proxy_mbps_mean")
    ue_macro_packet_rate = values("rollout_ue_macro_packet_rate_mbps_mean")
    ue_p10_packet_rate = values("rollout_ue_p10_packet_rate_mbps_mean")
    service_p10 = values("rollout_recent100_service_rate_p10_mean", 1.0)
    unserved = values("rollout_unserved_demand_ue_mean")
    backlog = values("rollout_backlog_p90_bytes_mean")

    mean_goodput = float(np.mean(goodput)) if goodput else 0.0
    mean_packet_rate = float(np.mean(packet_rate)) if packet_rate else 0.0
    mean_packet_per_rate = float(np.mean(packet_per_rate)) if packet_per_rate else 0.0
    mean_ue_macro_rate_proxy = float(np.mean(ue_macro_rate_proxy)) if ue_macro_rate_proxy else 0.0
    mean_ue_p10_rate_proxy = float(np.mean(ue_p10_rate_proxy)) if ue_p10_rate_proxy else 0.0
    mean_ue_macro_packet_rate = float(np.mean(ue_macro_packet_rate)) if ue_macro_packet_rate else 0.0
    mean_ue_p10_packet_rate = float(np.mean(ue_p10_packet_rate)) if ue_p10_packet_rate else 0.0
    max_packet_delay = float(np.max(packet_delay)) if packet_delay else 0.0
    max_tb_err = float(np.max(tb_err)) if tb_err else 0.0
    max_expiry = float(np.max(expiry)) if expiry else 0.0
    max_backlog = float(np.max(backlog)) if backlog else 0.0
    min_packet_rate = float(np.min(packet_rate)) if packet_rate else 0.0
    min_packet_per_rate = float(np.min(packet_per_rate)) if packet_per_rate else 0.0
    min_ue_macro_rate_proxy = float(np.min(ue_macro_rate_proxy)) if ue_macro_rate_proxy else 0.0
    min_ue_p10_rate_proxy = float(np.min(ue_p10_rate_proxy)) if ue_p10_rate_proxy else 0.0
    min_ue_macro_packet_rate = float(np.min(ue_macro_packet_rate)) if ue_macro_packet_rate else 0.0
    min_ue_p10_packet_rate = float(np.min(ue_p10_packet_rate)) if ue_p10_packet_rate else 0.0
    min_service_p10 = float(np.min(service_p10)) if service_p10 else 1.0
    max_unserved = float(np.max(unserved)) if unserved else 0.0
    mean_tb_err = float(np.mean(tb_err)) if tb_err else 0.0
    mean_packet_delay = float(np.mean(packet_delay)) if packet_delay else 0.0

    tb_gap = max(0.0, max_tb_err - float(args.best_tb_err_target))
    packet_rate_bonus = float(args.best_packet_rate_bonus_mbps) * math.log1p(
        max(0.0, mean_packet_rate) / max(1.0e-6, float(args.packet_rate_scale_mbps))
    )
    packet_per_rate_bonus = float(args.best_packet_per_packet_rate_bonus_mbps) * math.log1p(
        max(0.0, mean_packet_per_rate) / max(1.0e-6, float(args.packet_per_packet_rate_scale_mbps))
    )
    packet_delay_penalty = float(args.best_packet_delay_penalty_mbps) * math.log1p(
        max(0.0, max_packet_delay) / max(1.0e-6, float(args.packet_delay_scale_ms))
    )
    ue_macro_rate_proxy_bonus = float(args.best_ue_macro_rate_proxy_bonus_mbps) * math.log1p(
        max(0.0, mean_ue_macro_rate_proxy) / max(1.0e-6, float(args.ue_rate_proxy_scale_mbps))
    )
    ue_p10_rate_proxy_gap_penalty = float(args.best_ue_p10_rate_proxy_gap_penalty_mbps) * math.log1p(
        max(0.0, float(args.best_ue_p10_rate_proxy_target_mbps) - min_ue_p10_rate_proxy)
        / max(1.0e-6, float(args.ue_rate_proxy_scale_mbps))
    )
    ue_macro_packet_rate_bonus = float(args.best_ue_macro_packet_rate_bonus_mbps) * math.log1p(
        max(0.0, mean_ue_macro_packet_rate) / max(1.0e-6, float(args.ue_packet_rate_scale_mbps))
    )
    ue_p10_packet_rate_gap_penalty = float(args.best_ue_p10_packet_rate_gap_penalty_mbps) * math.log1p(
        max(0.0, float(args.best_ue_p10_packet_rate_target_mbps) - min_ue_p10_packet_rate)
        / max(1.0e-6, float(args.ue_packet_rate_scale_mbps))
    )
    score = (
        mean_goodput
        + packet_rate_bonus
        + packet_per_rate_bonus
        + ue_macro_rate_proxy_bonus
        + ue_macro_packet_rate_bonus
        - packet_delay_penalty
        - ue_p10_rate_proxy_gap_penalty
        - ue_p10_packet_rate_gap_penalty
        - float(args.best_tb_err_penalty_mbps) * tb_gap
        - float(args.best_expiry_penalty_mbps) * max_expiry
        - float(args.best_backlog_p90_penalty_mbps) * max_backlog
    )

    hard_tb_err_pass = 1.0
    hard_expiry_pass = 1.0
    hard_service_pass = 1.0
    hard_unserved_pass = 1.0
    hard_backlog_pass = 1.0
    hard_packet_rate_pass = 1.0
    hard_packet_per_packet_rate_pass = 1.0
    hard_packet_delay_pass = 1.0
    hard_ue_macro_rate_proxy_pass = 1.0
    hard_ue_p10_rate_proxy_pass = 1.0
    hard_ue_macro_packet_rate_pass = 1.0
    hard_ue_p10_packet_rate_pass = 1.0
    if float(args.best_hard_tb_err_max) >= 0.0:
        hard_tb_err_pass = float(max_tb_err <= float(args.best_hard_tb_err_max))
    if float(args.best_hard_expiry_rate_max) >= 0.0:
        hard_expiry_pass = float(max_expiry <= float(args.best_hard_expiry_rate_max))
    if float(args.best_hard_service_p10_min) >= 0.0:
        hard_service_pass = float(min_service_p10 >= float(args.best_hard_service_p10_min))
    if float(args.best_hard_unserved_ue_max) >= 0.0:
        hard_unserved_pass = float(max_unserved <= float(args.best_hard_unserved_ue_max))
    if float(args.best_hard_backlog_p90_max) >= 0.0:
        hard_backlog_pass = float(max_backlog <= float(args.best_hard_backlog_p90_max))
    if float(args.best_hard_packet_rate_min_mbps) >= 0.0:
        hard_packet_rate_pass = float(min_packet_rate >= float(args.best_hard_packet_rate_min_mbps))
    if float(args.best_hard_packet_per_packet_rate_min_mbps) >= 0.0:
        hard_packet_per_packet_rate_pass = float(
            min_packet_per_rate >= float(args.best_hard_packet_per_packet_rate_min_mbps)
        )
    if float(args.best_hard_packet_delay_max_ms) >= 0.0:
        hard_packet_delay_pass = float(max_packet_delay <= float(args.best_hard_packet_delay_max_ms))
    if float(args.best_hard_ue_macro_rate_proxy_min_mbps) >= 0.0:
        hard_ue_macro_rate_proxy_pass = float(
            min_ue_macro_rate_proxy >= float(args.best_hard_ue_macro_rate_proxy_min_mbps)
        )
    if float(args.best_hard_ue_p10_rate_proxy_min_mbps) >= 0.0:
        hard_ue_p10_rate_proxy_pass = float(
            min_ue_p10_rate_proxy >= float(args.best_hard_ue_p10_rate_proxy_min_mbps)
        )
    if float(args.best_hard_ue_macro_packet_rate_min_mbps) >= 0.0:
        hard_ue_macro_packet_rate_pass = float(
            min_ue_macro_packet_rate >= float(args.best_hard_ue_macro_packet_rate_min_mbps)
        )
    if float(args.best_hard_ue_p10_packet_rate_min_mbps) >= 0.0:
        hard_ue_p10_packet_rate_pass = float(
            min_ue_p10_packet_rate >= float(args.best_hard_ue_p10_packet_rate_min_mbps)
        )
    fail_count = float(
        int(not bool(coverage_pass))
        + int(not bool(hard_tb_err_pass))
        + int(not bool(hard_expiry_pass))
        + int(not bool(hard_service_pass))
        + int(not bool(hard_unserved_pass))
        + int(not bool(hard_backlog_pass))
        + int(not bool(hard_packet_rate_pass))
        + int(not bool(hard_packet_per_packet_rate_pass))
        + int(not bool(hard_packet_delay_pass))
        + int(not bool(hard_ue_macro_rate_proxy_pass))
        + int(not bool(hard_ue_p10_rate_proxy_pass))
        + int(not bool(hard_ue_macro_packet_rate_pass))
        + int(not bool(hard_ue_p10_packet_rate_pass))
    )
    return {
        "best_multi_seed_window_iters": float(window_iters),
        "best_multi_seed_window_count": float(len(rows)),
        "best_multi_seed_min_covered_seeds": float(min_covered),
        "best_multi_seed_covered_seed_count": float(len(covered_set)),
        "best_multi_seed_coverage_pass": float(bool(coverage_pass)),
        "best_multi_seed_mean_goodput_mbps": float(mean_goodput),
        "best_multi_seed_mean_tb_err_rate": float(mean_tb_err),
        "best_multi_seed_max_tb_err_rate": float(max_tb_err),
        "best_multi_seed_max_expiry_rate": float(max_expiry),
        "best_multi_seed_mean_packet_rate_mbps": float(mean_packet_rate),
        "best_multi_seed_min_packet_rate_mbps": float(min_packet_rate),
        "best_multi_seed_mean_packet_per_packet_rate_mbps": float(mean_packet_per_rate),
        "best_multi_seed_min_packet_per_packet_rate_mbps": float(min_packet_per_rate),
        "best_multi_seed_mean_packet_delay_ms": float(mean_packet_delay),
        "best_multi_seed_max_packet_delay_ms": float(max_packet_delay),
        "best_multi_seed_mean_ue_macro_rate_proxy_mbps": float(mean_ue_macro_rate_proxy),
        "best_multi_seed_min_ue_macro_rate_proxy_mbps": float(min_ue_macro_rate_proxy),
        "best_multi_seed_mean_ue_p10_rate_proxy_mbps": float(mean_ue_p10_rate_proxy),
        "best_multi_seed_min_ue_p10_rate_proxy_mbps": float(min_ue_p10_rate_proxy),
        "best_multi_seed_mean_ue_macro_packet_rate_mbps": float(mean_ue_macro_packet_rate),
        "best_multi_seed_min_ue_macro_packet_rate_mbps": float(min_ue_macro_packet_rate),
        "best_multi_seed_mean_ue_p10_packet_rate_mbps": float(mean_ue_p10_packet_rate),
        "best_multi_seed_min_ue_p10_packet_rate_mbps": float(min_ue_p10_packet_rate),
        "best_multi_seed_min_service_p10": float(min_service_p10),
        "best_multi_seed_max_unserved_ue": float(max_unserved),
        "best_multi_seed_max_backlog_p90_bytes": float(max_backlog),
        "best_multi_seed_packet_rate_bonus_mbps_equiv": float(packet_rate_bonus),
        "best_multi_seed_packet_per_packet_rate_bonus_mbps_equiv": float(packet_per_rate_bonus),
        "best_multi_seed_packet_delay_penalty_mbps_equiv": float(packet_delay_penalty),
        "best_multi_seed_ue_macro_rate_proxy_bonus_mbps_equiv": float(ue_macro_rate_proxy_bonus),
        "best_multi_seed_ue_p10_rate_proxy_gap_penalty_mbps_equiv": float(
            ue_p10_rate_proxy_gap_penalty
        ),
        "best_multi_seed_ue_macro_packet_rate_bonus_mbps_equiv": float(
            ue_macro_packet_rate_bonus
        ),
        "best_multi_seed_ue_p10_packet_rate_gap_penalty_mbps_equiv": float(
            ue_p10_packet_rate_gap_penalty
        ),
        "best_multi_seed_score": float(score),
        "best_multi_seed_hard_tb_err_pass": float(hard_tb_err_pass),
        "best_multi_seed_hard_expiry_pass": float(hard_expiry_pass),
        "best_multi_seed_hard_service_pass": float(hard_service_pass),
        "best_multi_seed_hard_unserved_pass": float(hard_unserved_pass),
        "best_multi_seed_hard_backlog_pass": float(hard_backlog_pass),
        "best_multi_seed_hard_packet_rate_pass": float(hard_packet_rate_pass),
        "best_multi_seed_hard_packet_per_packet_rate_pass": float(hard_packet_per_packet_rate_pass),
        "best_multi_seed_hard_packet_delay_pass": float(hard_packet_delay_pass),
        "best_multi_seed_hard_ue_macro_rate_proxy_pass": float(hard_ue_macro_rate_proxy_pass),
        "best_multi_seed_hard_ue_p10_rate_proxy_pass": float(hard_ue_p10_rate_proxy_pass),
        "best_multi_seed_hard_ue_macro_packet_rate_pass": float(hard_ue_macro_packet_rate_pass),
        "best_multi_seed_hard_ue_p10_packet_rate_pass": float(hard_ue_p10_packet_rate_pass),
        "best_multi_seed_hard_gate_fail_count": float(fail_count),
        "best_multi_seed_hard_gate_pass": float(fail_count == 0.0),
    }


def _write_summary(
    *,
    path: Path,
    args: argparse.Namespace,
    device: torch.device,
    reward_cfg: Dict[str, float],
    loaded_compatible: int,
    loaded_total: int,
    actor_trainable_count: int,
    actor_total_count: int,
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
        "rollout_warmup_steps": int(args.rollout_warmup_steps),
        "device": str(device),
        "reward_mode": str(args.reward_mode),
        "reward_cfg": reward_cfg,
        "init_checkpoint": str(args.init_checkpoint),
        "ue_prg_topk": int(args.ue_prg_topk),
        "ue_prg_diversity_extra": int(args.ue_prg_diversity_extra),
        "candidate_mode": str(args.candidate_mode),
        "max_prg_candidates": int(args.max_prg_candidates),
        "action_head_mode": str(args.action_head_mode),
        "candidate_blank_logit_bias": float(args.candidate_blank_logit_bias),
        "candidate_success_logit_scale": float(args.candidate_success_logit_scale),
        "candidate_blank_success_scale": float(args.candidate_blank_success_scale),
        "include_pfq_score_anchor": int(args.include_pfq_score_anchor),
        "include_candidate_risk_features": int(args.include_candidate_risk_features),
        "candidate_risk_feature_dim": int(
            CANDIDATE_RISK_FEATURE_DIM if int(args.include_candidate_risk_features) else 0
        ),
        "write_diagnostics": int(args.write_diagnostics),
        "candidate_pfq_anchor_logit_coef": float(args.candidate_pfq_anchor_logit_coef),
        "candidate_success_aux_coef": float(args.candidate_success_aux_coef),
        "candidate_success_calib_coef": float(args.candidate_success_calib_coef),
        "candidate_success_max_calib_coef": float(args.candidate_success_max_calib_coef),
        "candidate_success_target_eps": float(args.candidate_success_target_eps),
        "candidate_pairwise_rank_coef": float(args.candidate_pairwise_rank_coef),
        "candidate_pairwise_rank_margin": float(args.candidate_pairwise_rank_margin),
        "candidate_pairwise_rank_min_gap": float(args.candidate_pairwise_rank_min_gap),
        "candidate_pairwise_rank_success_weight": float(args.candidate_pairwise_rank_success_weight),
        "candidate_pairwise_rank_service_weight": float(args.candidate_pairwise_rank_service_weight),
        "candidate_pairwise_rank_risk_weight": float(args.candidate_pairwise_rank_risk_weight),
        "candidate_pairwise_rank_fail_risk_weight": float(args.candidate_pairwise_rank_fail_risk_weight),
        "candidate_pairwise_rank_rescue_weight": float(args.candidate_pairwise_rank_rescue_weight),
        "candidate_rescue_aux_coef": float(args.candidate_rescue_aux_coef),
        "candidate_rescue_pos_weight": float(args.candidate_rescue_pos_weight),
        "candidate_rescue_service_target": float(args.candidate_rescue_service_target),
        "candidate_rescue_ttl_guard_ms": float(args.candidate_rescue_ttl_guard_ms),
        "candidate_rescue_max_target": float(args.candidate_rescue_max_target),
        "candidate_decode_goodput_coef": float(args.candidate_decode_goodput_coef),
        "candidate_decode_goodput_target_bytes": float(args.candidate_decode_goodput_target_bytes),
        "candidate_decode_goodput_max_target": float(args.candidate_decode_goodput_max_target),
        "candidate_decode_goodput_pos_weight": float(args.candidate_decode_goodput_pos_weight),
        "candidate_argmax_decode_coef": float(args.candidate_argmax_decode_coef),
        "candidate_argmax_decode_pos_weight": float(args.candidate_argmax_decode_pos_weight),
        "candidate_argmax_decode_headroom_rank_mode": str(args.candidate_argmax_decode_headroom_rank_mode),
        "candidate_blank_target_coef": float(args.candidate_blank_target_coef),
        "candidate_blank_target_threshold": float(args.candidate_blank_target_threshold),
        "candidate_blank_target_margin": float(args.candidate_blank_target_margin),
        "candidate_blank_target_deadband": float(args.candidate_blank_target_deadband),
        "candidate_blank_target_pos_weight": float(args.candidate_blank_target_pos_weight),
        "candidate_blank_target_fill_weight": float(args.candidate_blank_target_fill_weight),
        "candidate_blank_target_positive_only": int(args.candidate_blank_target_positive_only),
        "candidate_blank_target_service_guard": int(args.candidate_blank_target_service_guard),
        "candidate_blank_target_service_min_success": float(args.candidate_blank_target_service_min_success),
        "candidate_blank_target_service_min_debt": float(args.candidate_blank_target_service_min_debt),
        "candidate_blank_target_service_min_hol_ms": float(args.candidate_blank_target_service_min_hol_ms),
        "candidate_blank_target_service_max_scheduled_ratio": float(
            args.candidate_blank_target_service_max_scheduled_ratio
        ),
        "candidate_blank_target_service_fill_weight": float(args.candidate_blank_target_service_fill_weight),
        "candidate_blank_target_service_fill_margin": float(args.candidate_blank_target_service_fill_margin),
        "candidate_blank_target_service_fill_min_success": float(
            args.candidate_blank_target_service_fill_min_success
        ),
        "safe_fill_max_per_cell": int(args.safe_fill_max_per_cell),
        "safe_fill_min_quality_db": float(args.safe_fill_min_quality_db),
        "safe_fill_max_quality_gap_db": float(args.safe_fill_max_quality_gap_db),
        "safe_fill_max_prg_risk": float(args.safe_fill_max_prg_risk),
        "safe_fill_min_backlog_bytes": float(args.safe_fill_min_backlog_bytes),
        "safe_fill_max_ue_tb_err": float(args.safe_fill_max_ue_tb_err),
        "candidate_blank_budget_max_per_cell": int(args.candidate_blank_budget_max_per_cell),
        "candidate_blank_budget_min_decision_logit": float(args.candidate_blank_budget_min_decision_logit),
        "candidate_decode_demand_slack_bytes": float(args.candidate_decode_demand_slack_bytes),
        "candidate_marginal_headroom_slack_bytes": float(args.candidate_marginal_headroom_slack_bytes),
        "candidate_seq_state_dim": int(args.candidate_seq_state_dim),
        "candidate_seq_state_coef": float(args.candidate_seq_state_coef),
        "candidate_seq_fail_risk_feature_offset": int(args.candidate_seq_fail_risk_feature_offset),
        "candidate_seq_train_only": int(args.candidate_seq_train_only),
        "actor_trainable_param_count": int(actor_trainable_count),
        "actor_total_param_count": int(actor_total_count),
        "candidate_budgeted_select_mode": str(args.candidate_budgeted_select_mode),
        "candidate_budgeted_hard_cap": int(args.candidate_budgeted_hard_cap),
        "candidate_budgeted_usage_penalty": float(args.candidate_budgeted_usage_penalty),
        "candidate_budgeted_overcap_penalty": float(args.candidate_budgeted_overcap_penalty),
        "candidate_budgeted_first_service_bonus": float(args.candidate_budgeted_first_service_bonus),
        "candidate_budgeted_tail_bias_coef": float(args.candidate_budgeted_tail_bias_coef),
        "candidate_budgeted_tail_bias_service_target": float(args.candidate_budgeted_tail_bias_service_target),
        "candidate_budgeted_tail_bias_ttl_guard_ms": float(args.candidate_budgeted_tail_bias_ttl_guard_ms),
        "candidate_budgeted_tail_bias_max": float(args.candidate_budgeted_tail_bias_max),
        "candidate_budgeted_rescue_bias_coef": float(args.candidate_budgeted_rescue_bias_coef),
        "candidate_budgeted_rescue_bias_service_target": float(args.candidate_budgeted_rescue_bias_service_target),
        "candidate_budgeted_rescue_bias_ttl_guard_ms": float(args.candidate_budgeted_rescue_bias_ttl_guard_ms),
        "candidate_budgeted_rescue_bias_max": float(args.candidate_budgeted_rescue_bias_max),
        "candidate_budgeted_completion_bonus": float(args.candidate_budgeted_completion_bonus),
        "candidate_budgeted_completion_progress_weight": float(args.candidate_budgeted_completion_progress_weight),
        "candidate_budgeted_completion_max_score": float(args.candidate_budgeted_completion_max_score),
        "candidate_budgeted_tail_quota_enable": int(args.candidate_budgeted_tail_quota_enable),
        "candidate_budgeted_tail_quota_min_score": float(args.candidate_budgeted_tail_quota_min_score),
        "candidate_budgeted_tail_quota_topk_ues": int(args.candidate_budgeted_tail_quota_topk_ues),
        "candidate_budgeted_tail_quota_prgs_per_ue": int(args.candidate_budgeted_tail_quota_prgs_per_ue),
        "candidate_budgeted_tail_quota_bonus": float(args.candidate_budgeted_tail_quota_bonus),
        "candidate_budgeted_tail_quota_order_bonus": float(args.candidate_budgeted_tail_quota_order_bonus),
        "candidate_budgeted_tail_quota_max_score": float(args.candidate_budgeted_tail_quota_max_score),
        "candidate_budgeted_tail_quota_risk_max": float(args.candidate_budgeted_tail_quota_risk_max),
        "slot_repair_max_per_cell": int(args.slot_repair_max_per_cell),
        "slot_repair_min_backlog_bytes": float(args.slot_repair_min_backlog_bytes),
        "imitation_episodes": int(args.imitation_episodes),
        "teacher_mode": str(args.teacher_mode),
        "teacher_aux_coef": float(args.teacher_aux_coef),
        "teacher_aux_min_coef": float(args.teacher_aux_min_coef),
        "teacher_aux_decay_iters": int(args.teacher_aux_decay_iters),
        "teacher_soft_target": int(args.teacher_soft_target),
        "teacher_soft_temperature": float(args.teacher_soft_temperature),
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
        "candidate_checkpoint_interval": int(args.candidate_checkpoint_interval),
        "candidate_checkpoint_require_gate": int(args.candidate_checkpoint_require_gate),
        "candidate_checkpoint_dir": str(Path(args.out_dir).resolve() / "candidate_checkpoints"),
        "best_metric": str(args.best_metric),
        "best_min_iter": int(args.best_min_iter),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "best_tb_err_target": float(args.best_tb_err_target),
        "best_tb_err_penalty_mbps": float(args.best_tb_err_penalty_mbps),
        "best_expiry_penalty_mbps": float(args.best_expiry_penalty_mbps),
        "best_packet_rate_bonus_mbps": float(args.best_packet_rate_bonus_mbps),
        "best_packet_per_packet_rate_bonus_mbps": float(
            args.best_packet_per_packet_rate_bonus_mbps
        ),
        "best_packet_delay_penalty_mbps": float(args.best_packet_delay_penalty_mbps),
        "best_ue_macro_rate_proxy_bonus_mbps": float(args.best_ue_macro_rate_proxy_bonus_mbps),
        "best_ue_p10_rate_proxy_gap_penalty_mbps": float(
            args.best_ue_p10_rate_proxy_gap_penalty_mbps
        ),
        "best_ue_p10_rate_proxy_target_mbps": float(args.best_ue_p10_rate_proxy_target_mbps),
        "best_ue_macro_packet_rate_bonus_mbps": float(
            args.best_ue_macro_packet_rate_bonus_mbps
        ),
        "best_ue_p10_packet_rate_gap_penalty_mbps": float(
            args.best_ue_p10_packet_rate_gap_penalty_mbps
        ),
        "best_ue_p10_packet_rate_target_mbps": float(
            args.best_ue_p10_packet_rate_target_mbps
        ),
        "best_backlog_p90_penalty_mbps": float(args.best_backlog_p90_penalty_mbps),
        "best_hard_tb_err_max": float(args.best_hard_tb_err_max),
        "best_hard_expiry_rate_max": float(args.best_hard_expiry_rate_max),
        "best_hard_service_p10_min": float(args.best_hard_service_p10_min),
        "best_hard_unserved_ue_max": float(args.best_hard_unserved_ue_max),
        "best_hard_backlog_p90_max": float(args.best_hard_backlog_p90_max),
        "best_hard_packet_rate_min_mbps": float(args.best_hard_packet_rate_min_mbps),
        "best_hard_packet_per_packet_rate_min_mbps": float(
            args.best_hard_packet_per_packet_rate_min_mbps
        ),
        "best_hard_packet_delay_max_ms": float(args.best_hard_packet_delay_max_ms),
        "best_hard_ue_macro_rate_proxy_min_mbps": float(
            args.best_hard_ue_macro_rate_proxy_min_mbps
        ),
        "best_hard_ue_p10_rate_proxy_min_mbps": float(args.best_hard_ue_p10_rate_proxy_min_mbps),
        "best_hard_ue_macro_packet_rate_min_mbps": float(
            args.best_hard_ue_macro_packet_rate_min_mbps
        ),
        "best_hard_ue_p10_packet_rate_min_mbps": float(
            args.best_hard_ue_p10_packet_rate_min_mbps
        ),
        "best_multi_seed_window_iters": int(args.best_multi_seed_window_iters),
        "best_multi_seed_min_covered_seeds": int(args.best_multi_seed_min_covered_seeds),
        "ue_macro_rate_proxy_weight": float(args.ue_macro_rate_proxy_weight),
        "ue_p10_rate_proxy_gap_weight": float(args.ue_p10_rate_proxy_gap_weight),
        "ue_rate_proxy_scale_mbps": float(args.ue_rate_proxy_scale_mbps),
        "ue_p10_rate_proxy_target_mbps": float(args.ue_p10_rate_proxy_target_mbps),
        "ue_macro_packet_rate_weight": float(args.ue_macro_packet_rate_weight),
        "ue_p10_packet_rate_gap_weight": float(args.ue_p10_packet_rate_gap_weight),
        "ue_packet_rate_scale_mbps": float(args.ue_packet_rate_scale_mbps),
        "ue_p10_packet_rate_target_mbps": float(args.ue_p10_packet_rate_target_mbps),
        "tail_delta_weight": float(args.tail_delta_weight),
        "tail_pressure_weight": float(args.tail_pressure_weight),
        "tail_delta_clip": float(args.tail_delta_clip),
        "tail_potential_discount": float(args.tail_potential_discount),
        "tail_potential_topk": int(args.tail_potential_topk),
        "tail_service_target": float(args.tail_service_target),
        "tail_ttl_guard_ms": float(args.tail_ttl_guard_ms),
        "expiry_delta_weight": float(args.expiry_delta_weight),
        "expiry_pressure_weight": float(args.expiry_pressure_weight),
        "expiry_delta_clip": float(args.expiry_delta_clip),
        "expiry_potential_discount": float(args.expiry_potential_discount),
        "expiry_potential_topk": int(args.expiry_potential_topk),
        "expiry_ttl_guard_ms": float(args.expiry_ttl_guard_ms),
        "expiry_reward_window_steps": int(args.expiry_reward_window_steps),
        "expiry_reward_offered_bytes_per_tti": float(args.expiry_reward_offered_bytes_per_tti),
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
        "--action-head-mode",
        choices=["slot", "candidate"],
        default="slot",
        help="slot keeps legacy PRG->slot logits; candidate scores PRG->candidate UE choices before Type-0 decode",
    )
    p.add_argument(
        "--candidate-blank-logit-bias",
        type=float,
        default=0.0,
        help="fixed additive bias on the candidate-head blank class; negative values encourage filling valid PRG candidates",
    )
    p.add_argument(
        "--candidate-success-logit-scale",
        type=float,
        default=1.0,
        help="scale for candidate success logits; <=0 keeps the legacy single-score candidate head behavior",
    )
    p.add_argument(
        "--candidate-seq-state-dim",
        type=int,
        default=0,
        help="dynamic intra-TTI candidate state feature dimension for the sequential candidate scorer; 0 disables it",
    )
    p.add_argument(
        "--candidate-seq-state-coef",
        type=float,
        default=1.0,
        help="deployment/training multiplier for learned sequential candidate state logit deltas",
    )
    p.add_argument(
        "--candidate-seq-fail-risk-feature-offset",
        type=int,
        default=-1,
        help="candidate edge feature offset for fail-risk in sequential state features; -1 infers from edge width",
    )
    p.add_argument(
        "--candidate-blank-success-scale",
        type=float,
        default=1.0,
        help="scale for explicit blank/success gate; positive values make blank more likely when max candidate success is low",
    )
    p.add_argument(
        "--include-pfq-score-anchor",
        type=int,
        choices=[0, 1],
        default=1,
        help="include PFQ-like score as the sixth UE-PRG edge feature; set 0 for strict no-anchor runs",
    )
    p.add_argument(
        "--include-candidate-risk-features",
        type=int,
        choices=[0, 1],
        default=0,
        help="append candidate fail/interference-delay risk features to UE-PRG edge attributes",
    )
    p.add_argument(
        "--candidate-pfq-anchor-logit-coef",
        type=float,
        default=1.0,
        help="additive coefficient for the UE-PRG PFQ anchor feature in candidate-head logits; 0 disables",
    )
    p.add_argument(
        "--candidate-success-aux-coef",
        type=float,
        default=0.0,
        help="auxiliary BCE weight for candidate success_score soft physical-success targets",
    )
    p.add_argument(
        "--candidate-success-calib-coef",
        type=float,
        default=0.0,
        help="auxiliary SmoothL1 weight aligning every candidate success logit to logit(success target)",
    )
    p.add_argument(
        "--candidate-success-max-calib-coef",
        type=float,
        default=0.0,
        help="auxiliary SmoothL1 weight aligning max success logits to logit(max success target)",
    )
    p.add_argument(
        "--candidate-success-target-eps",
        type=float,
        default=0.05,
        help="clamp epsilon when converting soft success targets to logits for calibration",
    )
    p.add_argument("--candidate-pairwise-rank-coef", type=float, default=0.0)
    p.add_argument("--candidate-pairwise-rank-margin", type=float, default=0.25)
    p.add_argument("--candidate-pairwise-rank-min-gap", type=float, default=0.10)
    p.add_argument("--candidate-pairwise-rank-success-weight", type=float, default=0.70)
    p.add_argument("--candidate-pairwise-rank-service-weight", type=float, default=0.30)
    p.add_argument("--candidate-pairwise-rank-risk-weight", type=float, default=0.15)
    p.add_argument(
        "--candidate-pairwise-rank-fail-risk-weight",
        type=float,
        default=0.0,
        help="extra utility penalty on explicit candidate_fail_risk edge feature for pairwise ranking; 0 disables",
    )
    p.add_argument("--candidate-pairwise-rank-rescue-weight", type=float, default=0.0)
    p.add_argument("--candidate-rescue-aux-coef", type=float, default=0.0)
    p.add_argument("--candidate-rescue-pos-weight", type=float, default=2.0)
    p.add_argument("--candidate-rescue-service-target", type=float, default=0.80)
    p.add_argument("--candidate-rescue-ttl-guard-ms", type=float, default=60.0)
    p.add_argument("--candidate-rescue-max-target", type=float, default=1.0)
    p.add_argument(
        "--candidate-decode-goodput-coef",
        type=float,
        default=0.0,
        help="auxiliary BCE weight for sampled candidate scorer logits using post-decode/post-PHY goodput targets",
    )
    p.add_argument(
        "--candidate-decode-goodput-target-bytes",
        type=float,
        default=3000.0,
        help="per-PRG goodput bytes mapped to target 1.0 for decode-goodput supervision",
    )
    p.add_argument(
        "--candidate-decode-goodput-max-target",
        type=float,
        default=1.0,
        help="maximum normalized target for decode-goodput supervision",
    )
    p.add_argument(
        "--candidate-decode-goodput-pos-weight",
        type=float,
        default=2.0,
        help="extra weight proportional to positive decode-goodput targets",
    )
    p.add_argument(
        "--candidate-argmax-decode-coef",
        type=float,
        default=0.0,
        help="auxiliary BCE weight for current scorer-argmax candidates after Type-0 decode shadowing",
    )
    p.add_argument(
        "--candidate-argmax-decode-pos-weight",
        type=float,
        default=2.0,
        help="extra weight proportional to positive argmax-decode shadow targets",
    )
    p.add_argument(
        "--candidate-argmax-decode-headroom-rank-mode",
        choices=["stable", "utility", "logit", "phy"],
        default="stable",
        help="headroom ranking mode used when shadow-decoding current scorer argmax during training",
    )
    p.add_argument(
        "--candidate-blank-target-coef",
        type=float,
        default=0.0,
        help="auxiliary margin weight for explicit blank-vs-best-candidate deployment argmax target",
    )
    p.add_argument(
        "--candidate-blank-target-threshold",
        type=float,
        default=0.45,
        help="PRG is blank-targeted when max candidate success target is below this threshold",
    )
    p.add_argument(
        "--candidate-blank-target-margin",
        type=float,
        default=0.5,
        help="desired logit margin between blank and best candidate for explicit blank/fill target",
    )
    p.add_argument(
        "--candidate-blank-target-deadband",
        type=float,
        default=0.05,
        help="ignore PRGs whose max success target is within this distance of the blank threshold",
    )
    p.add_argument(
        "--candidate-blank-target-pos-weight",
        type=float,
        default=1.0,
        help="extra weight for positive blank-target PRGs in explicit blank margin loss",
    )
    p.add_argument(
        "--candidate-blank-target-fill-weight",
        type=float,
        default=1.0,
        help="weight for non-blank/fill-target PRGs in explicit blank margin loss",
    )
    p.add_argument(
        "--candidate-blank-target-positive-only",
        type=int,
        default=0,
        choices=[0, 1],
        help="when 1, apply explicit blank margin only to positive blank-target PRGs",
    )
    p.add_argument(
        "--candidate-blank-target-service-guard",
        type=int,
        default=0,
        choices=[0, 1],
        help="when 1, suppress blank targets on PRGs with plausible high-service-pressure candidates",
    )
    p.add_argument(
        "--candidate-blank-target-service-min-success",
        type=float,
        default=0.70,
        help="minimum candidate success target for service guard to block blank supervision",
    )
    p.add_argument(
        "--candidate-blank-target-service-min-debt",
        type=float,
        default=0.25,
        help="minimum recent-goodput deficit for service guard to block blank supervision",
    )
    p.add_argument(
        "--candidate-blank-target-service-min-hol-ms",
        type=float,
        default=1.0,
        help="minimum HOL delay for service guard to block blank supervision",
    )
    p.add_argument(
        "--candidate-blank-target-service-max-scheduled-ratio",
        type=float,
        default=0.65,
        help="maximum recent scheduled ratio for service guard to block blank supervision",
    )
    p.add_argument(
        "--candidate-blank-target-service-fill-weight",
        type=float,
        default=0.0,
        help="small fill-anchor weight for service-protected or very high-success PRGs when positive-only blank target is enabled",
    )
    p.add_argument(
        "--candidate-blank-target-service-fill-margin",
        type=float,
        default=0.15,
        help="desired candidate-over-blank margin for the local service fill anchor",
    )
    p.add_argument(
        "--candidate-blank-target-service-fill-min-success",
        type=float,
        default=0.995,
        help="success target above which a non-blank PRG receives local fill-anchor protection",
    )
    p.add_argument("--safe-fill-max-per-cell", type=int, default=0, help="guarded blank-PRG fill cap per cell; 0 disables")
    p.add_argument("--safe-fill-min-quality-db", type=float, default=-2.0)
    p.add_argument("--safe-fill-max-quality-gap-db", type=float, default=8.0)
    p.add_argument("--safe-fill-max-prg-risk", type=float, default=0.70)
    p.add_argument("--safe-fill-min-backlog-bytes", type=float, default=1.0)
    p.add_argument("--safe-fill-max-ue-tb-err", type=float, default=0.45)
    p.add_argument(
        "--candidate-blank-budget-max-per-cell",
        type=int,
        default=-1,
        help="-1 disables; otherwise cap candidate-head blank decisions per cell before Type-0 decode",
    )
    p.add_argument(
        "--candidate-blank-budget-min-decision-logit",
        type=float,
        default=-1.0e9,
        help="only blank decisions above this blank-minus-best-candidate logit can consume the blank budget",
    )
    p.add_argument(
        "--candidate-decode-demand-slack-bytes",
        type=float,
        default=3000.0,
        help="extra per-UE backlog slack used by candidate->Type-0 demand cap before leaving PRGs blank",
    )
    p.add_argument(
        "--candidate-marginal-headroom-slack-bytes",
        type=float,
        default=-1.0,
        help="-1 disables; otherwise soft per-UE marginal-demand slack used before excess PRGs are considered",
    )
    p.add_argument(
        "--candidate-budgeted-select-mode",
        choices=["off", "greedy", "sample"],
        default="off",
        help="headroom-aware candidate action selection before Type-0 decode; off keeps independent PRG choices",
    )
    p.add_argument(
        "--candidate-budgeted-hard-cap",
        type=int,
        choices=[0, 1],
        default=1,
        help="when enabled, candidates whose UE has no remaining estimated PRG headroom are masked during budgeted selection",
    )
    p.add_argument("--candidate-budgeted-usage-penalty", type=float, default=0.35)
    p.add_argument("--candidate-budgeted-overcap-penalty", type=float, default=8.0)
    p.add_argument("--candidate-budgeted-first-service-bonus", type=float, default=0.0)
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
    p.add_argument(
        "--target-kl-min-updates",
        type=int,
        default=4,
        help="minimum optimizer minibatches before target-KL early stop can trigger",
    )
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
    p.add_argument(
        "--rollout-warmup-steps",
        type=int,
        default=0,
        help="drive the environment for N steps before collecting PPO trajectory/best metrics",
    )
    p.add_argument("--update-log-every-minibatches", type=int, default=2)
    p.add_argument(
        "--write-diagnostics",
        type=int,
        choices=[0, 1],
        default=1,
        help="write per-iteration per-UE/native-reject diagnostic CSVs; disable for faster long PPO runs",
    )
    p.add_argument("--init-checkpoint", default="", help="Optional imitation/actor checkpoint to warm-start from")
    p.add_argument("--imitation-episodes", type=int, default=0, help="Optional online imitation warm-start episodes before PPO")
    p.add_argument("--imitation-max-steps", type=int, default=0, help="0 means use episode horizon during imitation warm-start")
    p.add_argument("--imitation-lr", type=float, default=0.0, help="0 means fallback to actor lr during imitation warm-start")
    p.add_argument("--teacher-mode", choices=["none", "heuristic_graph", "pfq_passthrough"], default="pfq_passthrough")
    p.add_argument("--teacher-aux-coef", type=float, default=0.20)
    p.add_argument("--teacher-aux-min-coef", type=float, default=0.0)
    p.add_argument("--teacher-aux-decay-iters", type=int, default=80)
    p.add_argument(
        "--teacher-soft-target",
        type=int,
        choices=[0, 1],
        default=1,
        help="candidate PRG teacher auxiliary uses soft performance-aware targets when enabled",
    )
    p.add_argument("--teacher-soft-temperature", type=float, default=0.85)
    p.add_argument("--teacher-soft-hit-bonus", type=float, default=2.0)
    p.add_argument("--teacher-soft-blank-bonus", type=float, default=0.35)
    p.add_argument("--teacher-soft-blank-safe-penalty", type=float, default=1.25)
    p.add_argument("--teacher-soft-blank-no-safe-bonus", type=float, default=1.0)
    p.add_argument(
        "--reward-mode",
        choices=[
            "raw_env",
            "stageb_v1",
            "stageb_blankaware",
            "stageb_tailguard_simple",
            "stageb_tailcredit_v4",
            "stageb_tailcredit_expiry_v5",
            "stageb_tailcredit_expiry_rescue_v6",
        ],
        default="stageb_v1",
    )
    p.add_argument(
        "--best-metric",
        choices=[
            "reward",
            "goodput",
            "service_guarded_goodput",
            "latency_guarded_goodput",
            "packet_latency_guarded_goodput",
            "reliability_guarded_goodput",
            "rate_guarded_goodput",
            "rate_tail_guarded_goodput",
            "multi_seed_rate_guarded_goodput",
        ],
        default="goodput",
    )
    p.add_argument("--early-stop-patience", type=int, default=60, help="0 disables early stop")
    p.add_argument("--early-stop-min-delta", type=float, default=1.0)
    p.add_argument("--early-stop-warmup-iters", type=int, default=50)
    p.add_argument("--best-min-iter", type=int, default=0, help="minimum iteration eligible for best checkpoints")
    p.add_argument(
        "--candidate-checkpoint-interval",
        type=int,
        default=0,
        help="save actor/train candidate checkpoints every N iterations after --best-min-iter; 0 disables",
    )
    p.add_argument(
        "--candidate-checkpoint-require-gate",
        type=int,
        choices=[0, 1],
        default=1,
        help="when saving periodic candidate checkpoints, require checkpoint hard gates to pass",
    )
    p.add_argument(
        "--best-service-gap-penalty-mbps",
        type=float,
        default=100.0,
        help="score penalty per unit service p10 gap for --best-metric service_guarded_goodput",
    )
    p.add_argument(
        "--best-unserved-ue-target",
        type=float,
        default=4.3,
        help="unserved demand UE target for --best-metric service_guarded_goodput",
    )
    p.add_argument(
        "--best-unserved-penalty-mbps",
        type=float,
        default=3.0,
        help="score penalty per unserved demand UE above target for --best-metric service_guarded_goodput",
    )
    p.add_argument("--best-tb-err-target", type=float, default=0.115)
    p.add_argument("--best-tb-err-penalty-mbps", type=float, default=500.0)
    p.add_argument("--best-expiry-penalty-mbps", type=float, default=1200.0)
    p.add_argument("--best-packet-rate-bonus-mbps", type=float, default=12.0)
    p.add_argument("--best-packet-per-packet-rate-bonus-mbps", type=float, default=0.0)
    p.add_argument(
        "--best-packet-delay-penalty-mbps",
        type=float,
        default=60.0,
        help="score penalty multiplier for log1p(packet delay / --packet-delay-scale-ms)",
    )
    p.add_argument(
        "--best-ue-macro-rate-proxy-bonus-mbps",
        type=float,
        default=0.0,
        help="score bonus multiplier for log1p(UE macro rate proxy / --ue-rate-proxy-scale-mbps)",
    )
    p.add_argument(
        "--best-ue-p10-rate-proxy-gap-penalty-mbps",
        type=float,
        default=0.0,
        help="score penalty multiplier for UE p10 rate proxy below --best-ue-p10-rate-proxy-target-mbps",
    )
    p.add_argument(
        "--best-ue-p10-rate-proxy-target-mbps",
        type=float,
        default=0.0,
        help="target for --best-ue-p10-rate-proxy-gap-penalty-mbps; 0 disables the gap unless set",
    )
    p.add_argument(
        "--best-ue-macro-packet-rate-bonus-mbps",
        type=float,
        default=0.0,
        help="score bonus multiplier for log1p(true UE macro packet rate / --ue-packet-rate-scale-mbps)",
    )
    p.add_argument(
        "--best-ue-p10-packet-rate-gap-penalty-mbps",
        type=float,
        default=0.0,
        help="score penalty multiplier for true UE p10 packet rate below --best-ue-p10-packet-rate-target-mbps",
    )
    p.add_argument(
        "--best-ue-p10-packet-rate-target-mbps",
        type=float,
        default=0.0,
        help="target for --best-ue-p10-packet-rate-gap-penalty-mbps; 0 disables the gap unless set",
    )
    p.add_argument("--best-backlog-p90-penalty-mbps", type=float, default=0.0)
    p.add_argument("--best-hard-tb-err-max", type=float, default=-1.0, help="<0 disables hard TB-error gate")
    p.add_argument(
        "--best-hard-expiry-rate-max",
        type=float,
        default=-1.0,
        help="<0 disables hard expiry gate; uses rollout_expiry_reward_rate_mean when present",
    )
    p.add_argument("--best-hard-service-p10-min", type=float, default=-1.0, help="<0 disables hard service-p10 gate")
    p.add_argument("--best-hard-unserved-ue-max", type=float, default=-1.0, help="<0 disables hard unserved-UE gate")
    p.add_argument("--best-hard-backlog-p90-max", type=float, default=-1.0, help="<0 disables hard backlog-p90 gate")
    p.add_argument(
        "--best-hard-packet-rate-min-mbps",
        type=float,
        default=-1.0,
        help="<0 disables hard aggregate packet service-rate gate",
    )
    p.add_argument(
        "--best-hard-packet-per-packet-rate-min-mbps",
        type=float,
        default=-1.0,
        help="<0 disables hard per-completed-packet service-rate gate",
    )
    p.add_argument(
        "--best-hard-packet-delay-max-ms",
        type=float,
        default=-1.0,
        help="<0 disables hard weighted packet-delay gate",
    )
    p.add_argument(
        "--best-hard-ue-macro-rate-proxy-min-mbps",
        type=float,
        default=-1.0,
        help="<0 disables hard UE macro rate proxy gate",
    )
    p.add_argument(
        "--best-hard-ue-p10-rate-proxy-min-mbps",
        type=float,
        default=-1.0,
        help="<0 disables hard UE p10 rate proxy gate",
    )
    p.add_argument(
        "--best-hard-ue-macro-packet-rate-min-mbps",
        type=float,
        default=-1.0,
        help="<0 disables hard true UE macro packet-rate gate",
    )
    p.add_argument(
        "--best-hard-ue-p10-packet-rate-min-mbps",
        type=float,
        default=-1.0,
        help="<0 disables hard true UE p10 packet-rate gate",
    )
    p.add_argument(
        "--best-multi-seed-window-iters",
        type=int,
        default=0,
        help="rolling iteration window for multi_seed_rate_guarded_goodput; 0 means one seed-list cycle",
    )
    p.add_argument(
        "--best-multi-seed-min-covered-seeds",
        type=int,
        default=0,
        help="minimum distinct topology seeds in the rolling best window; 0 means all seeds in --seed-list",
    )
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
        "--packet-per-packet-rate-weight",
        type=float,
        default=0.0,
        help="coefficient for completed-packet mean effective service rate bonus in stageb_blankaware; 0 disables it",
    )
    p.add_argument(
        "--packet-completion-weight",
        type=float,
        default=0.0,
        help="coefficient for per-TTI packet completion-count bonus in stageb_blankaware; 0 disables it",
    )
    p.add_argument(
        "--packet-delay-weight",
        type=float,
        default=0.0,
        help="coefficient for successful-packet mean delay penalty in stageb_blankaware; 0 disables it",
    )
    p.add_argument(
        "--packet-rate-scale-mbps",
        type=float,
        default=10.0,
        help="log scale for packet-rate reward bonus",
    )
    p.add_argument(
        "--packet-per-packet-rate-scale-mbps",
        type=float,
        default=24.0,
        help="log scale for completed-packet mean-rate reward bonus",
    )
    p.add_argument(
        "--packet-completion-scale",
        type=float,
        default=36.0,
        help="log scale for packet-completion reward bonus",
    )
    p.add_argument(
        "--packet-delay-scale-ms",
        type=float,
        default=10.0,
        help="log scale for successful-packet mean delay penalty",
    )
    p.add_argument(
        "--ue-macro-rate-proxy-weight",
        type=float,
        default=0.0,
        help="coefficient for demand-active UE macro goodput-rate proxy bonus in stageb_blankaware",
    )
    p.add_argument(
        "--ue-p10-rate-proxy-gap-weight",
        type=float,
        default=0.0,
        help="coefficient for UE p10 goodput-rate proxy target gap penalty in stageb_blankaware",
    )
    p.add_argument(
        "--ue-rate-proxy-scale-mbps",
        type=float,
        default=8.0,
        help="log scale for UE rate proxy reward and best-score terms",
    )
    p.add_argument(
        "--ue-p10-rate-proxy-target-mbps",
        type=float,
        default=0.0,
        help="UE p10 rate proxy target used by --ue-p10-rate-proxy-gap-weight",
    )
    p.add_argument(
        "--ue-macro-packet-rate-weight",
        type=float,
        default=0.0,
        help="coefficient for true demand-active UE macro packet-rate bonus in stageb_blankaware",
    )
    p.add_argument(
        "--ue-p10-packet-rate-gap-weight",
        type=float,
        default=0.0,
        help="coefficient for true UE p10 packet-rate target gap penalty in stageb_blankaware",
    )
    p.add_argument(
        "--ue-packet-rate-scale-mbps",
        type=float,
        default=8.0,
        help="log scale for true UE packet-rate reward and best-score terms",
    )
    p.add_argument(
        "--ue-p10-packet-rate-target-mbps",
        type=float,
        default=0.0,
        help="true UE p10 packet-rate target used by --ue-p10-packet-rate-gap-weight",
    )
    p.add_argument(
        "--aged-backlog-weight",
        type=float,
        default=0.0,
        help="coefficient for recent100 aged-backlog debt penalty in stageb_blankaware; 0 disables it",
    )
    p.add_argument(
        "--unserved-ue-weight",
        type=float,
        default=0.0,
        help="coefficient for normalized actual-unserved demand UE count in stageb_blankaware; 0 disables it",
    )
    p.add_argument(
        "--service-gap-weight",
        type=float,
        default=0.0,
        help="coefficient for max(0, service-rate-target - recent100 service p10) in stageb_blankaware",
    )
    p.add_argument(
        "--service-rate-target",
        type=float,
        default=0.0,
        help="recent100 service-rate p10 target used by --service-gap-weight",
    )
    p.add_argument(
        "--util-floor-weight",
        type=float,
        default=0.0,
        help="coefficient for BLER-gated under-utilization penalty in stageb_blankaware; 0 disables it",
    )
    p.add_argument(
        "--util-floor-target",
        type=float,
        default=0.0,
        help="target PRG utilization ratio for --util-floor-weight",
    )
    p.add_argument(
        "--util-floor-tb-err-guard",
        type=float,
        default=1.0,
        help="full utilization-floor penalty applies while TB error is at or below this value",
    )
    p.add_argument(
        "--util-floor-tb-err-softness",
        type=float,
        default=0.0,
        help="optional linear TB-error guard softness; gate fades to zero at guard+softness",
    )
    p.add_argument(
        "--missed-safe-opportunity-weight",
        type=float,
        default=0.0,
        help="coefficient for guarded final-blank PRGs that still had hard-headroom-safe candidates",
    )
    p.add_argument(
        "--missed-safe-opportunity-scale",
        type=float,
        default=51.0,
        help="log1p normalization scale for --missed-safe-opportunity-weight",
    )
    p.add_argument(
        "--prg-efficiency-weight",
        type=float,
        default=-1.0,
        help="explicit coefficient for active PRG goodput efficiency in stageb_blankaware; <0 uses ICI-derived legacy coefficient",
    )
    p.add_argument(
        "--harmful-packing-weight",
        type=float,
        default=-1.0,
        help="explicit coefficient for throughput-minus-goodput harmful packing penalty in stageb_blankaware; <0 uses conflict-derived legacy coefficient",
    )
    p.add_argument(
        "--tb-err-target",
        type=float,
        default=-1.0,
        help="target TB error rate for optional over-target reward penalties; <0 disables",
    )
    p.add_argument(
        "--tb-err-over-target-weight",
        type=float,
        default=0.0,
        help="coefficient for max(0, tb_err - --tb-err-target) in stageb_blankaware",
    )
    p.add_argument(
        "--tb-err-goodput-gate-weight",
        type=float,
        default=0.0,
        help="goodput-equivalent gate penalty multiplier for TB error above --tb-err-target",
    )
    p.add_argument("--tail-delta-weight", type=float, default=0.0)
    p.add_argument("--tail-pressure-weight", type=float, default=0.0)
    p.add_argument("--tail-delta-clip", type=float, default=1.0)
    p.add_argument("--tail-potential-discount", type=float, default=1.0)
    p.add_argument("--tail-potential-topk", type=int, default=8)
    p.add_argument("--tail-service-target", type=float, default=0.75)
    p.add_argument("--tail-ttl-guard-ms", type=float, default=20.0)
    p.add_argument("--expiry-delta-weight", type=float, default=0.0)
    p.add_argument("--expiry-pressure-weight", type=float, default=0.0)
    p.add_argument("--expiry-delta-clip", type=float, default=1.0)
    p.add_argument("--expiry-potential-discount", type=float, default=1.0)
    p.add_argument("--expiry-potential-topk", type=int, default=8)
    p.add_argument("--expiry-ttl-guard-ms", type=float, default=40.0)
    p.add_argument(
        "--expiry-reward-window-steps",
        type=int,
        default=0,
        help="if >0, train reward uses recent expired bytes over this TTI window instead of cumulative expiry_drop_rate",
    )
    p.add_argument(
        "--expiry-reward-offered-bytes-per-tti",
        type=float,
        default=0.0,
        help="normalizer for windowed expiry reward; if <=0, uses delivered+expired bytes in the window",
    )
    p.add_argument("--rescue-credit-weight", type=float, default=0.0)
    p.add_argument("--rescue-miss-weight", type=float, default=0.0)
    p.add_argument("--rescue-topk", type=int, default=8)
    p.add_argument("--rescue-service-target", type=float, default=0.80)
    p.add_argument("--rescue-ttl-guard-ms", type=float, default=60.0)
    p.add_argument("--rescue-max-score", type=float, default=1.5)
    p.add_argument("--rescue-goodput-bytes-scale", type=float, default=3000.0)
    p.add_argument("--candidate-budgeted-tail-bias-coef", type=float, default=0.0)
    p.add_argument("--candidate-budgeted-tail-bias-service-target", type=float, default=0.75)
    p.add_argument("--candidate-budgeted-tail-bias-ttl-guard-ms", type=float, default=20.0)
    p.add_argument("--candidate-budgeted-tail-bias-max", type=float, default=1.5)
    p.add_argument("--candidate-packet-tail-feature-enable", type=int, choices=[0, 1], default=0)
    p.add_argument("--candidate-packet-tail-rate-target-mbps", type=float, default=2.0)
    p.add_argument("--candidate-packet-tail-delay-scale-ms", type=float, default=10.0)
    p.add_argument("--candidate-budgeted-rescue-bias-coef", type=float, default=0.0)
    p.add_argument("--candidate-budgeted-rescue-bias-service-target", type=float, default=0.80)
    p.add_argument("--candidate-budgeted-rescue-bias-ttl-guard-ms", type=float, default=60.0)
    p.add_argument("--candidate-budgeted-rescue-bias-max", type=float, default=1.5)
    p.add_argument("--candidate-budgeted-completion-bonus", type=float, default=0.0)
    p.add_argument("--candidate-budgeted-completion-progress-weight", type=float, default=0.25)
    p.add_argument("--candidate-budgeted-completion-max-score", type=float, default=1.5)
    p.add_argument("--candidate-budgeted-tail-quota-enable", type=int, choices=[0, 1], default=0)
    p.add_argument("--candidate-budgeted-tail-quota-min-score", type=float, default=0.55)
    p.add_argument("--candidate-budgeted-tail-quota-topk-ues", type=int, default=6)
    p.add_argument("--candidate-budgeted-tail-quota-prgs-per-ue", type=int, default=1)
    p.add_argument("--candidate-budgeted-tail-quota-bonus", type=float, default=4.0)
    p.add_argument("--candidate-budgeted-tail-quota-order-bonus", type=float, default=2.0)
    p.add_argument("--candidate-budgeted-tail-quota-max-score", type=float, default=4.0)
    p.add_argument("--candidate-budgeted-tail-quota-risk-max", type=float, default=0.70)
    p.add_argument(
        "--candidate-seq-train-only",
        type=int,
        choices=[0, 1],
        default=0,
        help="freeze the actor except candidate_seq_state_score during PPO actor updates",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.sim_cwd == "":
        args.sim_cwd = None
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be >= 1")
    if int(args.rollout_warmup_steps) < 0:
        raise ValueError("--rollout-warmup-steps must be >= 0")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.best_min_iter < 0:
        raise ValueError("--best-min-iter must be >= 0")
    if int(args.best_multi_seed_window_iters) < 0:
        raise ValueError("--best-multi-seed-window-iters must be >= 0")
    if int(args.best_multi_seed_min_covered_seeds) < 0:
        raise ValueError("--best-multi-seed-min-covered-seeds must be >= 0")
    if args.best_ue_macro_rate_proxy_bonus_mbps < 0.0:
        raise ValueError("--best-ue-macro-rate-proxy-bonus-mbps must be >= 0")
    if args.best_ue_p10_rate_proxy_gap_penalty_mbps < 0.0:
        raise ValueError("--best-ue-p10-rate-proxy-gap-penalty-mbps must be >= 0")
    if args.best_ue_p10_rate_proxy_target_mbps < 0.0:
        raise ValueError("--best-ue-p10-rate-proxy-target-mbps must be >= 0")
    if args.best_ue_macro_packet_rate_bonus_mbps < 0.0:
        raise ValueError("--best-ue-macro-packet-rate-bonus-mbps must be >= 0")
    if args.best_ue_p10_packet_rate_gap_penalty_mbps < 0.0:
        raise ValueError("--best-ue-p10-packet-rate-gap-penalty-mbps must be >= 0")
    if args.best_ue_p10_packet_rate_target_mbps < 0.0:
        raise ValueError("--best-ue-p10-packet-rate-target-mbps must be >= 0")
    if int(args.candidate_checkpoint_interval) < 0:
        raise ValueError("--candidate-checkpoint-interval must be >= 0")
    if args.imitation_episodes < 0:
        raise ValueError("--imitation-episodes must be >= 0")
    if args.imitation_max_steps < 0:
        raise ValueError("--imitation-max-steps must be >= 0")
    if args.ue_prg_diversity_extra < 0:
        raise ValueError("--ue-prg-diversity-extra must be >= 0")
    if args.max_prg_candidates < 0:
        raise ValueError("--max-prg-candidates must be >= 0")
    if int(args.include_pfq_score_anchor) not in (0, 1):
        raise ValueError("--include-pfq-score-anchor must be 0 or 1")
    if int(args.include_candidate_risk_features) not in (0, 1):
        raise ValueError("--include-candidate-risk-features must be 0 or 1")
    if int(args.write_diagnostics) not in (0, 1):
        raise ValueError("--write-diagnostics must be 0 or 1")
    if args.candidate_seq_state_dim < 0:
        raise ValueError("--candidate-seq-state-dim must be >= 0")
    if args.candidate_seq_state_coef < 0.0:
        raise ValueError("--candidate-seq-state-coef must be >= 0")
    if args.candidate_seq_fail_risk_feature_offset < -1:
        raise ValueError("--candidate-seq-fail-risk-feature-offset must be >= -1")
    if int(args.candidate_seq_train_only) not in (0, 1):
        raise ValueError("--candidate-seq-train-only must be 0 or 1")
    if int(args.candidate_seq_train_only) == 1 and int(args.candidate_seq_state_dim) <= 0:
        raise ValueError("--candidate-seq-train-only requires --candidate-seq-state-dim > 0")
    if args.safe_fill_max_per_cell < 0:
        raise ValueError("--safe-fill-max-per-cell must be >= 0")
    if args.candidate_blank_budget_max_per_cell < -1:
        raise ValueError("--candidate-blank-budget-max-per-cell must be >= -1")
    if args.candidate_decode_demand_slack_bytes < 0.0:
        raise ValueError("--candidate-decode-demand-slack-bytes must be >= 0")
    if args.candidate_marginal_headroom_slack_bytes < -1.0:
        raise ValueError("--candidate-marginal-headroom-slack-bytes must be >= -1")
    if args.candidate_budgeted_usage_penalty < 0.0:
        raise ValueError("--candidate-budgeted-usage-penalty must be >= 0")
    if args.candidate_budgeted_overcap_penalty < 0.0:
        raise ValueError("--candidate-budgeted-overcap-penalty must be >= 0")
    if args.candidate_budgeted_tail_bias_coef < 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-coef must be >= 0")
    if args.candidate_budgeted_tail_bias_service_target <= 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-service-target must be > 0")
    if args.candidate_budgeted_tail_bias_ttl_guard_ms <= 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-ttl-guard-ms must be > 0")
    if args.candidate_budgeted_tail_bias_max < 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-max must be >= 0")
    if args.candidate_packet_tail_rate_target_mbps <= 0.0:
        raise ValueError("--candidate-packet-tail-rate-target-mbps must be > 0")
    if args.candidate_packet_tail_delay_scale_ms <= 0.0:
        raise ValueError("--candidate-packet-tail-delay-scale-ms must be > 0")
    if args.candidate_budgeted_rescue_bias_coef < 0.0:
        raise ValueError("--candidate-budgeted-rescue-bias-coef must be >= 0")
    if args.candidate_budgeted_rescue_bias_service_target <= 0.0:
        raise ValueError("--candidate-budgeted-rescue-bias-service-target must be > 0")
    if args.candidate_budgeted_rescue_bias_ttl_guard_ms <= 0.0:
        raise ValueError("--candidate-budgeted-rescue-bias-ttl-guard-ms must be > 0")
    if args.candidate_budgeted_rescue_bias_max < 0.0:
        raise ValueError("--candidate-budgeted-rescue-bias-max must be >= 0")
    if args.candidate_budgeted_completion_bonus < 0.0:
        raise ValueError("--candidate-budgeted-completion-bonus must be >= 0")
    if args.candidate_budgeted_completion_progress_weight < 0.0:
        raise ValueError("--candidate-budgeted-completion-progress-weight must be >= 0")
    if args.candidate_budgeted_completion_max_score < 0.0:
        raise ValueError("--candidate-budgeted-completion-max-score must be >= 0")
    if int(args.candidate_budgeted_tail_quota_enable) not in (0, 1):
        raise ValueError("--candidate-budgeted-tail-quota-enable must be 0 or 1")
    if args.candidate_budgeted_tail_quota_min_score < 0.0:
        raise ValueError("--candidate-budgeted-tail-quota-min-score must be >= 0")
    if args.candidate_budgeted_tail_quota_topk_ues < 0:
        raise ValueError("--candidate-budgeted-tail-quota-topk-ues must be >= 0")
    if args.candidate_budgeted_tail_quota_prgs_per_ue < 0:
        raise ValueError("--candidate-budgeted-tail-quota-prgs-per-ue must be >= 0")
    if args.candidate_budgeted_tail_quota_bonus < 0.0:
        raise ValueError("--candidate-budgeted-tail-quota-bonus must be >= 0")
    if args.candidate_budgeted_tail_quota_order_bonus < 0.0:
        raise ValueError("--candidate-budgeted-tail-quota-order-bonus must be >= 0")
    if args.candidate_budgeted_tail_quota_max_score < 0.0:
        raise ValueError("--candidate-budgeted-tail-quota-max-score must be >= 0")
    if args.candidate_budgeted_tail_quota_risk_max < -1.0:
        raise ValueError("--candidate-budgeted-tail-quota-risk-max must be >= -1")
    if args.candidate_pfq_anchor_logit_coef < 0.0:
        raise ValueError("--candidate-pfq-anchor-logit-coef must be >= 0")
    if args.teacher_aux_decay_iters < 0:
        raise ValueError("--teacher-aux-decay-iters must be >= 0")
    if args.candidate_success_aux_coef < 0.0:
        raise ValueError("--candidate-success-aux-coef must be >= 0")
    if args.candidate_success_calib_coef < 0.0:
        raise ValueError("--candidate-success-calib-coef must be >= 0")
    if args.candidate_success_max_calib_coef < 0.0:
        raise ValueError("--candidate-success-max-calib-coef must be >= 0")
    if args.candidate_pairwise_rank_coef < 0.0:
        raise ValueError("--candidate-pairwise-rank-coef must be >= 0")
    if args.candidate_pairwise_rank_margin < 0.0:
        raise ValueError("--candidate-pairwise-rank-margin must be >= 0")
    if args.candidate_pairwise_rank_min_gap < 0.0:
        raise ValueError("--candidate-pairwise-rank-min-gap must be >= 0")
    if args.candidate_pairwise_rank_success_weight < 0.0:
        raise ValueError("--candidate-pairwise-rank-success-weight must be >= 0")
    if args.candidate_pairwise_rank_service_weight < 0.0:
        raise ValueError("--candidate-pairwise-rank-service-weight must be >= 0")
    if args.candidate_pairwise_rank_risk_weight < 0.0:
        raise ValueError("--candidate-pairwise-rank-risk-weight must be >= 0")
    if args.candidate_pairwise_rank_fail_risk_weight < 0.0:
        raise ValueError("--candidate-pairwise-rank-fail-risk-weight must be >= 0")
    if args.candidate_pairwise_rank_rescue_weight < 0.0:
        raise ValueError("--candidate-pairwise-rank-rescue-weight must be >= 0")
    if args.candidate_rescue_aux_coef < 0.0:
        raise ValueError("--candidate-rescue-aux-coef must be >= 0")
    if args.candidate_rescue_pos_weight < 0.0:
        raise ValueError("--candidate-rescue-pos-weight must be >= 0")
    if args.candidate_rescue_service_target <= 0.0:
        raise ValueError("--candidate-rescue-service-target must be > 0")
    if args.candidate_rescue_ttl_guard_ms <= 0.0:
        raise ValueError("--candidate-rescue-ttl-guard-ms must be > 0")
    if not (0.0 < float(args.candidate_rescue_max_target) <= 1.5):
        raise ValueError("--candidate-rescue-max-target must be in (0, 1.5]")
    if args.candidate_decode_goodput_coef < 0.0:
        raise ValueError("--candidate-decode-goodput-coef must be >= 0")
    if args.candidate_decode_goodput_target_bytes <= 0.0:
        raise ValueError("--candidate-decode-goodput-target-bytes must be > 0")
    if not (0.0 < float(args.candidate_decode_goodput_max_target) <= 1.0):
        raise ValueError("--candidate-decode-goodput-max-target must be in (0, 1]")
    if args.candidate_decode_goodput_pos_weight < 0.0:
        raise ValueError("--candidate-decode-goodput-pos-weight must be >= 0")
    if args.candidate_argmax_decode_coef < 0.0:
        raise ValueError("--candidate-argmax-decode-coef must be >= 0")
    if args.candidate_argmax_decode_pos_weight < 0.0:
        raise ValueError("--candidate-argmax-decode-pos-weight must be >= 0")
    if args.candidate_blank_target_coef < 0.0:
        raise ValueError("--candidate-blank-target-coef must be >= 0")
    if args.tail_delta_weight < 0.0:
        raise ValueError("--tail-delta-weight must be >= 0")
    if args.tail_pressure_weight < 0.0:
        raise ValueError("--tail-pressure-weight must be >= 0")
    if args.tail_delta_clip < 0.0:
        raise ValueError("--tail-delta-clip must be >= 0")
    if args.tail_potential_discount < 0.0:
        raise ValueError("--tail-potential-discount must be >= 0")
    if args.tail_potential_topk < 1:
        raise ValueError("--tail-potential-topk must be >= 1")
    if args.tail_service_target <= 0.0:
        raise ValueError("--tail-service-target must be > 0")
    if args.tail_ttl_guard_ms <= 0.0:
        raise ValueError("--tail-ttl-guard-ms must be > 0")
    if args.expiry_delta_weight < 0.0:
        raise ValueError("--expiry-delta-weight must be >= 0")
    if args.expiry_pressure_weight < 0.0:
        raise ValueError("--expiry-pressure-weight must be >= 0")
    if args.expiry_delta_clip < 0.0:
        raise ValueError("--expiry-delta-clip must be >= 0")
    if args.expiry_potential_discount < 0.0:
        raise ValueError("--expiry-potential-discount must be >= 0")
    if args.expiry_potential_topk < 1:
        raise ValueError("--expiry-potential-topk must be >= 1")
    if args.expiry_ttl_guard_ms <= 0.0:
        raise ValueError("--expiry-ttl-guard-ms must be > 0")
    if args.expiry_reward_window_steps < 0:
        raise ValueError("--expiry-reward-window-steps must be >= 0")
    if args.expiry_reward_offered_bytes_per_tti < 0.0:
        raise ValueError("--expiry-reward-offered-bytes-per-tti must be >= 0")
    if args.rescue_credit_weight < 0.0:
        raise ValueError("--rescue-credit-weight must be >= 0")
    if args.rescue_miss_weight < 0.0:
        raise ValueError("--rescue-miss-weight must be >= 0")
    if args.rescue_topk < 1:
        raise ValueError("--rescue-topk must be >= 1")
    if args.rescue_service_target <= 0.0:
        raise ValueError("--rescue-service-target must be > 0")
    if args.rescue_ttl_guard_ms <= 0.0:
        raise ValueError("--rescue-ttl-guard-ms must be > 0")
    if args.rescue_max_score < 0.0:
        raise ValueError("--rescue-max-score must be >= 0")
    if args.rescue_goodput_bytes_scale <= 0.0:
        raise ValueError("--rescue-goodput-bytes-scale must be > 0")
    if not (0.0 < float(args.candidate_success_target_eps) < 0.5):
        raise ValueError("--candidate-success-target-eps must be in (0, 0.5)")
    if not (0.0 <= float(args.candidate_blank_target_threshold) <= 1.0):
        raise ValueError("--candidate-blank-target-threshold must be in [0, 1]")
    if args.candidate_blank_target_deadband < 0.0:
        raise ValueError("--candidate-blank-target-deadband must be >= 0")
    if args.candidate_blank_target_pos_weight < 0.0:
        raise ValueError("--candidate-blank-target-pos-weight must be >= 0")
    if args.candidate_blank_target_fill_weight < 0.0:
        raise ValueError("--candidate-blank-target-fill-weight must be >= 0")
    if not (0.0 <= float(args.candidate_blank_target_service_min_success) <= 1.0):
        raise ValueError("--candidate-blank-target-service-min-success must be in [0, 1]")
    if args.candidate_blank_target_service_min_debt < 0.0:
        raise ValueError("--candidate-blank-target-service-min-debt must be >= 0")
    if args.candidate_blank_target_service_min_hol_ms < 0.0:
        raise ValueError("--candidate-blank-target-service-min-hol-ms must be >= 0")
    if not (0.0 <= float(args.candidate_blank_target_service_max_scheduled_ratio) <= 1.0):
        raise ValueError("--candidate-blank-target-service-max-scheduled-ratio must be in [0, 1]")
    if args.candidate_blank_target_service_fill_weight < 0.0:
        raise ValueError("--candidate-blank-target-service-fill-weight must be >= 0")
    if args.candidate_blank_target_service_fill_margin < 0.0:
        raise ValueError("--candidate-blank-target-service-fill-margin must be >= 0")
    if not (0.0 <= float(args.candidate_blank_target_service_fill_min_success) <= 1.0):
        raise ValueError("--candidate-blank-target-service-fill-min-success must be in [0, 1]")
    if args.missed_safe_opportunity_weight < 0.0:
        raise ValueError("--missed-safe-opportunity-weight must be >= 0")
    if args.missed_safe_opportunity_scale <= 0.0:
        raise ValueError("--missed-safe-opportunity-scale must be > 0")
    if args.ue_macro_rate_proxy_weight < 0.0:
        raise ValueError("--ue-macro-rate-proxy-weight must be >= 0")
    if args.ue_p10_rate_proxy_gap_weight < 0.0:
        raise ValueError("--ue-p10-rate-proxy-gap-weight must be >= 0")
    if args.ue_rate_proxy_scale_mbps <= 0.0:
        raise ValueError("--ue-rate-proxy-scale-mbps must be > 0")
    if args.ue_p10_rate_proxy_target_mbps < 0.0:
        raise ValueError("--ue-p10-rate-proxy-target-mbps must be >= 0")
    if args.ue_macro_packet_rate_weight < 0.0:
        raise ValueError("--ue-macro-packet-rate-weight must be >= 0")
    if args.ue_p10_packet_rate_gap_weight < 0.0:
        raise ValueError("--ue-p10-packet-rate-gap-weight must be >= 0")
    if args.ue_packet_rate_scale_mbps <= 0.0:
        raise ValueError("--ue-packet-rate-scale-mbps must be > 0")
    if args.ue_p10_packet_rate_target_mbps < 0.0:
        raise ValueError("--ue-p10-packet-rate-target-mbps must be >= 0")
    if args.prg_efficiency_weight < 0.0 and abs(float(args.prg_efficiency_weight) + 1.0) > 1.0e-9:
        raise ValueError("--prg-efficiency-weight must be >= 0, or -1 to use the legacy coefficient")
    if args.harmful_packing_weight < 0.0 and abs(float(args.harmful_packing_weight) + 1.0) > 1.0e-9:
        raise ValueError("--harmful-packing-weight must be >= 0, or -1 to use the legacy coefficient")

    _set_seed(int(args.seed))
    device = _pick_device(args.device)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "online_ppo_metrics.csv"
    summary_json = out_dir / "online_ppo_summary.json"
    write_diagnostics = bool(int(args.write_diagnostics))
    per_ue_log_dir = out_dir / "per_ue_diagnostics" if write_diagnostics else None
    native_reject_log_dir = out_dir / "native_reject_diagnostics" if write_diagnostics else None
    actor_last = out_dir / "ppo_actor_last.pt"
    actor_best = out_dir / "ppo_actor_best.pt"
    actor_absolute_best = out_dir / "ppo_actor_absolute_best.pt"
    actor_threshold_best = out_dir / "ppo_actor_threshold_best.pt"
    train_last = out_dir / "ppo_train_last.pt"
    train_best = out_dir / "ppo_train_best.pt"
    train_absolute_best = out_dir / "ppo_train_absolute_best.pt"
    train_threshold_best = out_dir / "ppo_train_threshold_best.pt"
    imitation_actor_init = out_dir / "ppo_actor_imitation_init.pt"
    candidate_checkpoint_dir = out_dir / "candidate_checkpoints"
    if int(args.candidate_checkpoint_interval) > 0:
        candidate_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    snapshot_builder = FeatureSnapshotBuilder(
        SnapshotBuilderConfig(tb_err_ema_alpha=float(args.tb_err_ema_alpha))
    )
    max_prg_candidates = _resolve_prg_candidate_count(args)
    edge_generator = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            ue_prg_topk=int(args.ue_prg_topk),
            ue_prg_diversity_extra=int(args.ue_prg_diversity_extra),
            candidate_mode=str(args.candidate_mode),
            include_pfq_score_anchor=bool(int(args.include_pfq_score_anchor)),
            include_candidate_risk_features=bool(int(args.include_candidate_risk_features)),
            candidate_risk_feature_dim=CANDIDATE_RISK_FEATURE_DIM
            if int(args.include_candidate_risk_features)
            else 0,
        )
    )
    up_edge_attr_dim = 5 + (
        CANDIDATE_RISK_FEATURE_DIM if int(args.include_candidate_risk_features) else 0
    )
    if int(args.include_pfq_score_anchor):
        up_edge_attr_dim += 1
    actor = StageBHGraphPolicy(
        HGraphModelConfig(
            hidden_dim=int(args.hidden_dim),
            message_layers=int(args.message_layers),
            dropout=float(args.dropout),
            up_edge_attr_dim=up_edge_attr_dim,
            max_prg_candidates=max_prg_candidates,
            action_head_mode=str(args.action_head_mode),
            candidate_blank_logit_bias=float(args.candidate_blank_logit_bias),
            candidate_success_logit_scale=float(args.candidate_success_logit_scale),
            candidate_blank_success_scale=float(args.candidate_blank_success_scale),
            candidate_pfq_anchor_logit_coef=float(args.candidate_pfq_anchor_logit_coef),
            has_pfq_score_anchor=int(args.include_pfq_score_anchor),
            candidate_seq_state_dim=int(args.candidate_seq_state_dim),
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

    actor_trainable_params, actor_trainable_count, actor_total_count = _configure_actor_trainable_params(
        actor,
        candidate_seq_train_only=bool(int(args.candidate_seq_train_only)),
    )
    print(
        f"[online-ppo] actor_trainable_params={actor_trainable_count}/{actor_total_count} "
        f"candidate_seq_train_only={int(args.candidate_seq_train_only)}"
    )
    actor_optimizer = torch.optim.AdamW(
        actor_trainable_params,
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
        "packet_per_packet_rate_weight": float(args.packet_per_packet_rate_weight),
        "packet_completion_weight": float(args.packet_completion_weight),
        "packet_delay_weight": float(args.packet_delay_weight),
        "packet_rate_scale_mbps": float(args.packet_rate_scale_mbps),
        "packet_per_packet_rate_scale_mbps": float(args.packet_per_packet_rate_scale_mbps),
        "packet_completion_scale": float(args.packet_completion_scale),
        "packet_delay_scale_ms": float(args.packet_delay_scale_ms),
        "ue_macro_rate_proxy_weight": float(args.ue_macro_rate_proxy_weight),
        "ue_p10_rate_proxy_gap_weight": float(args.ue_p10_rate_proxy_gap_weight),
        "ue_rate_proxy_scale_mbps": float(args.ue_rate_proxy_scale_mbps),
        "ue_p10_rate_proxy_target_mbps": float(args.ue_p10_rate_proxy_target_mbps),
        "ue_macro_packet_rate_weight": float(args.ue_macro_packet_rate_weight),
        "ue_p10_packet_rate_gap_weight": float(args.ue_p10_packet_rate_gap_weight),
        "ue_packet_rate_scale_mbps": float(args.ue_packet_rate_scale_mbps),
        "ue_p10_packet_rate_target_mbps": float(args.ue_p10_packet_rate_target_mbps),
        "aged_backlog_weight": float(args.aged_backlog_weight),
        "unserved_ue_weight": float(args.unserved_ue_weight),
        "service_gap_weight": float(args.service_gap_weight),
        "service_rate_target": float(args.service_rate_target),
        "util_floor_weight": float(args.util_floor_weight),
        "util_floor_target": float(args.util_floor_target),
        "util_floor_tb_err_guard": float(args.util_floor_tb_err_guard),
        "util_floor_tb_err_softness": float(args.util_floor_tb_err_softness),
        "missed_safe_opportunity_weight": float(args.missed_safe_opportunity_weight),
        "missed_safe_opportunity_scale": float(args.missed_safe_opportunity_scale),
        "prg_efficiency_weight": float(args.prg_efficiency_weight),
        "harmful_packing_weight": float(args.harmful_packing_weight),
        "tb_err_target": float(args.tb_err_target),
        "tb_err_over_target_weight": float(args.tb_err_over_target_weight),
        "tb_err_goodput_gate_weight": float(args.tb_err_goodput_gate_weight),
        "tail_delta_weight": float(args.tail_delta_weight),
        "tail_pressure_weight": float(args.tail_pressure_weight),
        "tail_delta_clip": float(args.tail_delta_clip),
        "tail_potential_discount": float(args.tail_potential_discount),
        "tail_potential_topk": float(args.tail_potential_topk),
        "tail_service_target": float(args.tail_service_target),
        "tail_ttl_guard_ms": float(args.tail_ttl_guard_ms),
        "expiry_delta_weight": float(args.expiry_delta_weight),
        "expiry_pressure_weight": float(args.expiry_pressure_weight),
        "expiry_delta_clip": float(args.expiry_delta_clip),
        "expiry_potential_discount": float(args.expiry_potential_discount),
        "expiry_potential_topk": float(args.expiry_potential_topk),
        "expiry_ttl_guard_ms": float(args.expiry_ttl_guard_ms),
        "expiry_reward_window_steps": int(args.expiry_reward_window_steps),
        "expiry_reward_offered_bytes_per_tti": float(args.expiry_reward_offered_bytes_per_tti),
        "rescue_credit_weight": float(args.rescue_credit_weight),
        "rescue_miss_weight": float(args.rescue_miss_weight),
        "rescue_topk": float(args.rescue_topk),
        "rescue_service_target": float(args.rescue_service_target),
        "rescue_ttl_guard_ms": float(args.rescue_ttl_guard_ms),
        "rescue_max_score": float(args.rescue_max_score),
        "rescue_goodput_bytes_scale": float(args.rescue_goodput_bytes_scale),
    }

    runner = OnlineEpisodeRunner(args)
    per_ue_tracker: PerUeDiagnosticsTracker | None = None
    expiry_reward_tracker: ExpiryRewardWindowTracker | None = None
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
            rollout, rollout_summary, per_ue_tracker, expiry_reward_tracker = _collect_rollout(
                runner=runner,
                snapshot_builder=snapshot_builder,
                edge_generator=edge_generator,
                actor=actor,
                critic=critic,
                device=device,
                rollout_steps=int(args.rollout_steps),
                rollout_warmup_steps=int(args.rollout_warmup_steps),
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
                expiry_reward_tracker=expiry_reward_tracker,
                slot_repair_max_per_cell=int(args.slot_repair_max_per_cell),
                slot_repair_min_backlog_bytes=float(args.slot_repair_min_backlog_bytes),
            )
            collect_time_s = time.perf_counter() - collect_t0

            if not rollout:
                raise RuntimeError(
                    "rollout produced no PPO steps; reduce --rollout-warmup-steps "
                    "or increase --episode-horizon"
                )

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
                target_kl_min_updates=int(args.target_kl_min_updates),
                state_norm=state_norm,
                max_grad_norm=float(args.max_grad_norm),
                value_loss_type=str(args.value_loss),
                value_huber_beta=float(args.value_huber_beta),
                teacher_aux_coef=_resolve_teacher_aux_coef(args, iter_idx),
                teacher_soft_target=int(args.teacher_soft_target),
                teacher_soft_temperature=float(args.teacher_soft_temperature),
                teacher_soft_hit_bonus=float(args.teacher_soft_hit_bonus),
                teacher_soft_blank_bonus=float(args.teacher_soft_blank_bonus),
                teacher_soft_blank_safe_penalty=float(args.teacher_soft_blank_safe_penalty),
                teacher_soft_blank_no_safe_bonus=float(args.teacher_soft_blank_no_safe_bonus),
                candidate_success_aux_coef=float(args.candidate_success_aux_coef),
                candidate_success_calib_coef=float(args.candidate_success_calib_coef),
                candidate_success_max_calib_coef=float(args.candidate_success_max_calib_coef),
                candidate_success_target_eps=float(args.candidate_success_target_eps),
                candidate_pairwise_rank_coef=float(args.candidate_pairwise_rank_coef),
                candidate_pairwise_rank_margin=float(args.candidate_pairwise_rank_margin),
                candidate_pairwise_rank_min_gap=float(args.candidate_pairwise_rank_min_gap),
                candidate_pairwise_rank_success_weight=float(args.candidate_pairwise_rank_success_weight),
                candidate_pairwise_rank_service_weight=float(args.candidate_pairwise_rank_service_weight),
                candidate_pairwise_rank_risk_weight=float(args.candidate_pairwise_rank_risk_weight),
                candidate_pairwise_rank_fail_risk_weight=float(args.candidate_pairwise_rank_fail_risk_weight),
                candidate_pairwise_rank_fail_risk_feature_offset=(
                    5 + int(args.include_pfq_score_anchor)
                    if int(args.include_candidate_risk_features)
                    else -1
                ),
                candidate_pairwise_rank_rescue_weight=float(args.candidate_pairwise_rank_rescue_weight),
                candidate_rescue_aux_coef=float(args.candidate_rescue_aux_coef),
                candidate_rescue_pos_weight=float(args.candidate_rescue_pos_weight),
                candidate_rescue_service_target=float(args.candidate_rescue_service_target),
                candidate_rescue_ttl_guard_ms=float(args.candidate_rescue_ttl_guard_ms),
                candidate_rescue_max_target=float(args.candidate_rescue_max_target),
                candidate_decode_goodput_coef=float(args.candidate_decode_goodput_coef),
                candidate_decode_goodput_pos_weight=float(args.candidate_decode_goodput_pos_weight),
                candidate_argmax_decode_coef=float(args.candidate_argmax_decode_coef),
                candidate_argmax_decode_pos_weight=float(args.candidate_argmax_decode_pos_weight),
                candidate_argmax_decode_headroom_rank_mode=str(args.candidate_argmax_decode_headroom_rank_mode),
                candidate_blank_target_coef=float(args.candidate_blank_target_coef),
                candidate_blank_target_threshold=float(args.candidate_blank_target_threshold),
                candidate_blank_target_margin=float(args.candidate_blank_target_margin),
                candidate_blank_target_deadband=float(args.candidate_blank_target_deadband),
                candidate_blank_target_pos_weight=float(args.candidate_blank_target_pos_weight),
                candidate_blank_target_fill_weight=float(args.candidate_blank_target_fill_weight),
                candidate_blank_target_positive_only=int(args.candidate_blank_target_positive_only),
                candidate_blank_target_service_guard=int(args.candidate_blank_target_service_guard),
                candidate_blank_target_service_min_success=float(args.candidate_blank_target_service_min_success),
                candidate_blank_target_service_min_debt=float(args.candidate_blank_target_service_min_debt),
                candidate_blank_target_service_min_hol_ms=float(args.candidate_blank_target_service_min_hol_ms),
                candidate_blank_target_service_max_scheduled_ratio=float(
                    args.candidate_blank_target_service_max_scheduled_ratio
                ),
                candidate_blank_target_service_fill_weight=float(args.candidate_blank_target_service_fill_weight),
                candidate_blank_target_service_fill_margin=float(args.candidate_blank_target_service_fill_margin),
                candidate_blank_target_service_fill_min_success=float(
                    args.candidate_blank_target_service_fill_min_success
                ),
                safe_fill_max_per_cell=int(args.safe_fill_max_per_cell),
                safe_fill_min_quality_db=float(args.safe_fill_min_quality_db),
                safe_fill_max_quality_gap_db=float(args.safe_fill_max_quality_gap_db),
                safe_fill_max_prg_risk=float(args.safe_fill_max_prg_risk),
                safe_fill_min_backlog_bytes=float(args.safe_fill_min_backlog_bytes),
                safe_fill_max_ue_tb_err=float(args.safe_fill_max_ue_tb_err),
                candidate_blank_budget_max_per_cell=int(args.candidate_blank_budget_max_per_cell),
                candidate_blank_budget_min_decision_logit=float(args.candidate_blank_budget_min_decision_logit),
                candidate_decode_demand_slack_bytes=float(args.candidate_decode_demand_slack_bytes),
                candidate_marginal_headroom_slack_bytes=float(args.candidate_marginal_headroom_slack_bytes),
                candidate_budgeted_select_mode=str(args.candidate_budgeted_select_mode),
                candidate_budgeted_hard_cap=int(args.candidate_budgeted_hard_cap),
                candidate_budgeted_usage_penalty=float(args.candidate_budgeted_usage_penalty),
                candidate_budgeted_overcap_penalty=float(args.candidate_budgeted_overcap_penalty),
                candidate_budgeted_first_service_bonus=float(args.candidate_budgeted_first_service_bonus),
                candidate_budgeted_tail_bias_coef=float(args.candidate_budgeted_tail_bias_coef),
                candidate_budgeted_tail_bias_service_target=float(args.candidate_budgeted_tail_bias_service_target),
                candidate_budgeted_tail_bias_ttl_guard_ms=float(args.candidate_budgeted_tail_bias_ttl_guard_ms),
                candidate_budgeted_tail_bias_max=float(args.candidate_budgeted_tail_bias_max),
                candidate_budgeted_rescue_bias_coef=float(args.candidate_budgeted_rescue_bias_coef),
                candidate_budgeted_rescue_bias_service_target=float(args.candidate_budgeted_rescue_bias_service_target),
                candidate_budgeted_rescue_bias_ttl_guard_ms=float(args.candidate_budgeted_rescue_bias_ttl_guard_ms),
                candidate_budgeted_rescue_bias_max=float(args.candidate_budgeted_rescue_bias_max),
                candidate_budgeted_completion_bonus=float(args.candidate_budgeted_completion_bonus),
                candidate_budgeted_completion_progress_weight=float(args.candidate_budgeted_completion_progress_weight),
                candidate_budgeted_completion_max_score=float(args.candidate_budgeted_completion_max_score),
                candidate_budgeted_tail_quota_enable=int(args.candidate_budgeted_tail_quota_enable),
                candidate_budgeted_tail_quota_min_score=float(args.candidate_budgeted_tail_quota_min_score),
                candidate_budgeted_tail_quota_topk_ues=int(args.candidate_budgeted_tail_quota_topk_ues),
                candidate_budgeted_tail_quota_prgs_per_ue=int(args.candidate_budgeted_tail_quota_prgs_per_ue),
                candidate_budgeted_tail_quota_bonus=float(args.candidate_budgeted_tail_quota_bonus),
                candidate_budgeted_tail_quota_order_bonus=float(args.candidate_budgeted_tail_quota_order_bonus),
                candidate_budgeted_tail_quota_max_score=float(args.candidate_budgeted_tail_quota_max_score),
                candidate_budgeted_tail_quota_risk_max=float(args.candidate_budgeted_tail_quota_risk_max),
                candidate_seq_state_coef=float(args.candidate_seq_state_coef),
                candidate_seq_fail_risk_feature_offset=int(args.candidate_seq_fail_risk_feature_offset),
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

            best_packet_rate_bonus = 0.0
            best_packet_per_packet_rate_bonus = 0.0
            best_packet_delay_penalty = 0.0
            best_ue_macro_rate_proxy_bonus = 0.0
            best_ue_p10_rate_proxy_gap_penalty = 0.0
            best_ue_macro_packet_rate_bonus = 0.0
            best_ue_p10_packet_rate_gap_penalty = 0.0
            if str(args.best_metric) == "goodput":
                score = float(row["rollout_goodput_mbps_mean"])
            elif str(args.best_metric) == "service_guarded_goodput":
                service_target = max(0.0, float(args.service_rate_target))
                service_gap = (
                    max(0.0, service_target - float(row["rollout_recent100_service_rate_p10_mean"]))
                    if service_target > 0.0
                    else 0.0
                )
                unserved_gap = max(
                    0.0,
                    float(row["rollout_unserved_demand_ue_mean"]) - float(args.best_unserved_ue_target),
                )
                score = (
                    float(row["rollout_goodput_mbps_mean"])
                    - float(args.best_service_gap_penalty_mbps) * service_gap
                    - float(args.best_unserved_penalty_mbps) * unserved_gap
                )
            elif str(args.best_metric) == "latency_guarded_goodput":
                service_target = max(0.0, float(args.service_rate_target))
                service_gap = (
                    max(0.0, service_target - float(row["rollout_recent100_service_rate_p10_mean"]))
                    if service_target > 0.0
                    else 0.0
                )
                unserved_gap = max(
                    0.0,
                    float(row["rollout_unserved_demand_ue_mean"]) - float(args.best_unserved_ue_target),
                )
                tb_gap = max(0.0, float(row["rollout_tb_err_rate_mean"]) - float(args.best_tb_err_target))
                packet_rate_bonus = float(args.best_packet_rate_bonus_mbps) * math.log1p(
                    max(0.0, float(row["rollout_packet_effective_service_rate_mbps_mean"]))
                    / max(1.0e-6, float(args.packet_rate_scale_mbps))
                )
                best_packet_rate_bonus = packet_rate_bonus
                score = (
                    float(row["rollout_goodput_mbps_mean"])
                    + packet_rate_bonus
                    - float(args.best_service_gap_penalty_mbps) * service_gap
                    - float(args.best_unserved_penalty_mbps) * unserved_gap
                    - float(args.best_tb_err_penalty_mbps) * tb_gap
                    - float(args.best_expiry_penalty_mbps)
                    * float(row.get("rollout_expiry_reward_rate_mean", row["rollout_expiry_drop_rate_mean"]))
                    - float(args.best_backlog_p90_penalty_mbps) * float(row["rollout_backlog_p90_bytes_mean"])
                )
            elif str(args.best_metric) == "packet_latency_guarded_goodput":
                packet_rate_bonus = float(args.best_packet_rate_bonus_mbps) * math.log1p(
                    max(0.0, float(row["rollout_packet_effective_service_rate_mbps_mean"]))
                    / max(1.0e-6, float(args.packet_rate_scale_mbps))
                )
                best_packet_rate_bonus = packet_rate_bonus
                packet_delay_ms = float(
                    row.get(
                        "rollout_packet_delay_weighted_mean_ms",
                        row.get("rollout_packet_delay_mean_ms_mean", 0.0),
                    )
                )
                packet_delay_penalty = float(args.best_packet_delay_penalty_mbps) * math.log1p(
                    max(0.0, packet_delay_ms) / max(1.0e-6, float(args.packet_delay_scale_ms))
                )
                best_packet_delay_penalty = packet_delay_penalty
                score = (
                    float(row["rollout_goodput_mbps_mean"])
                    + packet_rate_bonus
                    - packet_delay_penalty
                )
            elif str(args.best_metric) == "reliability_guarded_goodput":
                tb_gap = max(0.0, float(row["rollout_tb_err_rate_mean"]) - float(args.best_tb_err_target))
                score = (
                    float(row["rollout_goodput_mbps_mean"])
                    - float(args.best_tb_err_penalty_mbps) * tb_gap
                    - float(args.best_expiry_penalty_mbps)
                    * float(row.get("rollout_expiry_reward_rate_mean", row["rollout_expiry_drop_rate_mean"]))
                    - float(args.best_backlog_p90_penalty_mbps) * float(row["rollout_backlog_p90_bytes_mean"])
                )
            elif str(args.best_metric) in (
                "rate_guarded_goodput",
                "rate_tail_guarded_goodput",
                "multi_seed_rate_guarded_goodput",
            ):
                tb_gap = max(0.0, float(row["rollout_tb_err_rate_mean"]) - float(args.best_tb_err_target))
                packet_rate_bonus = float(args.best_packet_rate_bonus_mbps) * math.log1p(
                    max(0.0, float(row["rollout_packet_effective_service_rate_mbps_mean"]))
                    / max(1.0e-6, float(args.packet_rate_scale_mbps))
                )
                packet_per_packet_rate_bonus = float(
                    args.best_packet_per_packet_rate_bonus_mbps
                ) * math.log1p(
                    max(
                        0.0,
                        float(
                            row.get(
                                "rollout_packet_effective_service_rate_per_packet_mean_mbps_mean",
                                0.0,
                            )
                        ),
                    )
                    / max(1.0e-6, float(args.packet_per_packet_rate_scale_mbps))
                )
                packet_delay_ms = float(
                    row.get(
                        "rollout_packet_delay_weighted_mean_ms",
                        row.get("rollout_packet_delay_mean_ms_mean", 0.0),
                    )
                )
                packet_delay_penalty = float(args.best_packet_delay_penalty_mbps) * math.log1p(
                    max(0.0, packet_delay_ms) / max(1.0e-6, float(args.packet_delay_scale_ms))
                )
                ue_macro_rate_proxy_bonus = float(
                    args.best_ue_macro_rate_proxy_bonus_mbps
                ) * math.log1p(
                    max(0.0, float(row.get("rollout_ue_macro_rate_proxy_mbps_mean", 0.0)))
                    / max(1.0e-6, float(args.ue_rate_proxy_scale_mbps))
                )
                ue_p10_rate_proxy_gap_penalty = float(
                    args.best_ue_p10_rate_proxy_gap_penalty_mbps
                ) * math.log1p(
                    max(
                        0.0,
                        float(args.best_ue_p10_rate_proxy_target_mbps)
                        - float(row.get("rollout_ue_p10_rate_proxy_mbps_mean", 0.0)),
                    )
                    / max(1.0e-6, float(args.ue_rate_proxy_scale_mbps))
                )
                ue_macro_packet_rate_bonus = float(
                    args.best_ue_macro_packet_rate_bonus_mbps
                ) * math.log1p(
                    max(0.0, float(row.get("rollout_ue_macro_packet_rate_mbps_mean", 0.0)))
                    / max(1.0e-6, float(args.ue_packet_rate_scale_mbps))
                )
                ue_p10_packet_rate_gap_penalty = float(
                    args.best_ue_p10_packet_rate_gap_penalty_mbps
                ) * math.log1p(
                    max(
                        0.0,
                        float(args.best_ue_p10_packet_rate_target_mbps)
                        - float(row.get("rollout_ue_p10_packet_rate_mbps_mean", 0.0)),
                    )
                    / max(1.0e-6, float(args.ue_packet_rate_scale_mbps))
                )
                best_packet_rate_bonus = packet_rate_bonus
                best_packet_per_packet_rate_bonus = packet_per_packet_rate_bonus
                best_packet_delay_penalty = packet_delay_penalty
                best_ue_macro_rate_proxy_bonus = ue_macro_rate_proxy_bonus
                best_ue_p10_rate_proxy_gap_penalty = ue_p10_rate_proxy_gap_penalty
                best_ue_macro_packet_rate_bonus = ue_macro_packet_rate_bonus
                best_ue_p10_packet_rate_gap_penalty = ue_p10_packet_rate_gap_penalty
                score = (
                    float(row["rollout_goodput_mbps_mean"])
                    + packet_rate_bonus
                    + packet_per_packet_rate_bonus
                    + ue_macro_rate_proxy_bonus
                    + ue_macro_packet_rate_bonus
                    - packet_delay_penalty
                    - ue_p10_rate_proxy_gap_penalty
                    - ue_p10_packet_rate_gap_penalty
                    - float(args.best_tb_err_penalty_mbps) * tb_gap
                    - float(args.best_expiry_penalty_mbps)
                    * float(row.get("rollout_expiry_reward_rate_mean", row["rollout_expiry_drop_rate_mean"]))
                    - float(args.best_backlog_p90_penalty_mbps) * float(row["rollout_backlog_p90_bytes_mean"])
                )
            else:
                score = float(row["rollout_reward_mean"])

            hard_tb_err_pass = 1.0
            hard_expiry_pass = 1.0
            hard_service_pass = 1.0
            hard_unserved_pass = 1.0
            hard_backlog_pass = 1.0
            hard_packet_rate_pass = 1.0
            hard_packet_per_packet_rate_pass = 1.0
            hard_packet_delay_pass = 1.0
            hard_ue_macro_rate_proxy_pass = 1.0
            hard_ue_p10_rate_proxy_pass = 1.0
            hard_ue_macro_packet_rate_pass = 1.0
            hard_ue_p10_packet_rate_pass = 1.0
            if float(args.best_hard_tb_err_max) >= 0.0:
                hard_tb_err_pass = float(row["rollout_tb_err_rate_mean"]) <= float(args.best_hard_tb_err_max)
            if float(args.best_hard_expiry_rate_max) >= 0.0:
                expiry_for_gate = float(
                    row.get("rollout_expiry_reward_rate_mean", row["rollout_expiry_drop_rate_mean"])
                )
                hard_expiry_pass = expiry_for_gate <= float(args.best_hard_expiry_rate_max)
            if float(args.best_hard_service_p10_min) >= 0.0:
                hard_service_pass = (
                    float(row["rollout_recent100_service_rate_p10_mean"])
                    >= float(args.best_hard_service_p10_min)
                )
            if float(args.best_hard_unserved_ue_max) >= 0.0:
                hard_unserved_pass = (
                    float(row["rollout_unserved_demand_ue_mean"])
                    <= float(args.best_hard_unserved_ue_max)
                )
            if float(args.best_hard_backlog_p90_max) >= 0.0:
                hard_backlog_pass = (
                    float(row["rollout_backlog_p90_bytes_mean"])
                    <= float(args.best_hard_backlog_p90_max)
                )
            if float(args.best_hard_packet_rate_min_mbps) >= 0.0:
                hard_packet_rate_pass = (
                    float(row["rollout_packet_effective_service_rate_mbps_mean"])
                    >= float(args.best_hard_packet_rate_min_mbps)
                )
            if float(args.best_hard_packet_per_packet_rate_min_mbps) >= 0.0:
                hard_packet_per_packet_rate_pass = (
                    float(
                        row.get(
                            "rollout_packet_effective_service_rate_per_packet_mean_mbps_mean",
                            0.0,
                        )
                    )
                    >= float(args.best_hard_packet_per_packet_rate_min_mbps)
                )
            if float(args.best_hard_packet_delay_max_ms) >= 0.0:
                hard_packet_delay_pass = (
                    float(
                        row.get(
                            "rollout_packet_delay_weighted_mean_ms",
                            row.get("rollout_packet_delay_mean_ms_mean", 0.0),
                        )
                    )
                    <= float(args.best_hard_packet_delay_max_ms)
                )
            if float(args.best_hard_ue_macro_rate_proxy_min_mbps) >= 0.0:
                hard_ue_macro_rate_proxy_pass = (
                    float(row.get("rollout_ue_macro_rate_proxy_mbps_mean", 0.0))
                    >= float(args.best_hard_ue_macro_rate_proxy_min_mbps)
                )
            if float(args.best_hard_ue_p10_rate_proxy_min_mbps) >= 0.0:
                hard_ue_p10_rate_proxy_pass = (
                    float(row.get("rollout_ue_p10_rate_proxy_mbps_mean", 0.0))
                    >= float(args.best_hard_ue_p10_rate_proxy_min_mbps)
                )
            if float(args.best_hard_ue_macro_packet_rate_min_mbps) >= 0.0:
                hard_ue_macro_packet_rate_pass = (
                    float(row.get("rollout_ue_macro_packet_rate_mbps_mean", 0.0))
                    >= float(args.best_hard_ue_macro_packet_rate_min_mbps)
                )
            if float(args.best_hard_ue_p10_packet_rate_min_mbps) >= 0.0:
                hard_ue_p10_packet_rate_pass = (
                    float(row.get("rollout_ue_p10_packet_rate_mbps_mean", 0.0))
                    >= float(args.best_hard_ue_p10_packet_rate_min_mbps)
                )
            hard_gate_fail_count = float(
                int(not hard_tb_err_pass)
                + int(not hard_expiry_pass)
                + int(not hard_service_pass)
                + int(not hard_unserved_pass)
                + int(not hard_backlog_pass)
                + int(not hard_packet_rate_pass)
                + int(not hard_packet_per_packet_rate_pass)
                + int(not hard_packet_delay_pass)
                + int(not hard_ue_macro_rate_proxy_pass)
                + int(not hard_ue_p10_rate_proxy_pass)
                + int(not hard_ue_macro_packet_rate_pass)
                + int(not hard_ue_p10_packet_rate_pass)
            )
            hard_gate_pass = hard_gate_fail_count == 0.0
            if str(args.best_metric) == "multi_seed_rate_guarded_goodput":
                row["best_single_hard_gate_fail_count"] = float(hard_gate_fail_count)
                row["best_single_hard_gate_pass"] = float(hard_gate_pass)
                multi_seed_metrics = _multi_seed_rate_window_metrics(history, args)
                row.update(multi_seed_metrics)
                score = float(multi_seed_metrics["best_multi_seed_score"])
                best_packet_rate_bonus = float(
                    multi_seed_metrics["best_multi_seed_packet_rate_bonus_mbps_equiv"]
                )
                best_packet_per_packet_rate_bonus = float(
                    multi_seed_metrics["best_multi_seed_packet_per_packet_rate_bonus_mbps_equiv"]
                )
                best_packet_delay_penalty = float(
                    multi_seed_metrics["best_multi_seed_packet_delay_penalty_mbps_equiv"]
                )
                best_ue_macro_rate_proxy_bonus = float(
                    multi_seed_metrics["best_multi_seed_ue_macro_rate_proxy_bonus_mbps_equiv"]
                )
                best_ue_p10_rate_proxy_gap_penalty = float(
                    multi_seed_metrics["best_multi_seed_ue_p10_rate_proxy_gap_penalty_mbps_equiv"]
                )
                best_ue_macro_packet_rate_bonus = float(
                    multi_seed_metrics["best_multi_seed_ue_macro_packet_rate_bonus_mbps_equiv"]
                )
                best_ue_p10_packet_rate_gap_penalty = float(
                    multi_seed_metrics[
                        "best_multi_seed_ue_p10_packet_rate_gap_penalty_mbps_equiv"
                    ]
                )
                hard_tb_err_pass = float(multi_seed_metrics["best_multi_seed_hard_tb_err_pass"])
                hard_expiry_pass = float(multi_seed_metrics["best_multi_seed_hard_expiry_pass"])
                hard_service_pass = float(multi_seed_metrics["best_multi_seed_hard_service_pass"])
                hard_unserved_pass = float(multi_seed_metrics["best_multi_seed_hard_unserved_pass"])
                hard_backlog_pass = float(multi_seed_metrics["best_multi_seed_hard_backlog_pass"])
                hard_packet_rate_pass = float(
                    multi_seed_metrics["best_multi_seed_hard_packet_rate_pass"]
                )
                hard_packet_per_packet_rate_pass = float(
                    multi_seed_metrics["best_multi_seed_hard_packet_per_packet_rate_pass"]
                )
                hard_packet_delay_pass = float(
                    multi_seed_metrics["best_multi_seed_hard_packet_delay_pass"]
                )
                hard_ue_macro_rate_proxy_pass = float(
                    multi_seed_metrics["best_multi_seed_hard_ue_macro_rate_proxy_pass"]
                )
                hard_ue_p10_rate_proxy_pass = float(
                    multi_seed_metrics["best_multi_seed_hard_ue_p10_rate_proxy_pass"]
                )
                hard_ue_macro_packet_rate_pass = float(
                    multi_seed_metrics["best_multi_seed_hard_ue_macro_packet_rate_pass"]
                )
                hard_ue_p10_packet_rate_pass = float(
                    multi_seed_metrics["best_multi_seed_hard_ue_p10_packet_rate_pass"]
                )
                hard_gate_fail_count = float(multi_seed_metrics["best_multi_seed_hard_gate_fail_count"])
                hard_gate_pass = bool(multi_seed_metrics["best_multi_seed_hard_gate_pass"])
            best_eligible = (iter_idx >= int(args.best_min_iter)) and hard_gate_pass
            row["best_score"] = float(score)
            row["best_packet_rate_bonus_mbps_equiv"] = float(best_packet_rate_bonus)
            row["best_packet_per_packet_rate_bonus_mbps_equiv"] = float(
                best_packet_per_packet_rate_bonus
            )
            row["best_packet_delay_penalty_mbps_equiv"] = float(best_packet_delay_penalty)
            row["best_ue_macro_rate_proxy_bonus_mbps_equiv"] = float(
                best_ue_macro_rate_proxy_bonus
            )
            row["best_ue_p10_rate_proxy_gap_penalty_mbps_equiv"] = float(
                best_ue_p10_rate_proxy_gap_penalty
            )
            row["best_ue_macro_packet_rate_bonus_mbps_equiv"] = float(
                best_ue_macro_packet_rate_bonus
            )
            row["best_ue_p10_packet_rate_gap_penalty_mbps_equiv"] = float(
                best_ue_p10_packet_rate_gap_penalty
            )
            row["best_hard_tb_err_pass"] = float(hard_tb_err_pass)
            row["best_hard_expiry_pass"] = float(hard_expiry_pass)
            row["best_hard_service_pass"] = float(hard_service_pass)
            row["best_hard_unserved_pass"] = float(hard_unserved_pass)
            row["best_hard_backlog_pass"] = float(hard_backlog_pass)
            row["best_hard_packet_rate_pass"] = float(hard_packet_rate_pass)
            row["best_hard_packet_per_packet_rate_pass"] = float(
                hard_packet_per_packet_rate_pass
            )
            row["best_hard_packet_delay_pass"] = float(hard_packet_delay_pass)
            row["best_hard_ue_macro_rate_proxy_pass"] = float(hard_ue_macro_rate_proxy_pass)
            row["best_hard_ue_p10_rate_proxy_pass"] = float(hard_ue_p10_rate_proxy_pass)
            row["best_hard_ue_macro_packet_rate_pass"] = float(hard_ue_macro_packet_rate_pass)
            row["best_hard_ue_p10_packet_rate_pass"] = float(hard_ue_p10_packet_rate_pass)
            row["best_hard_gate_fail_count"] = float(hard_gate_fail_count)
            row["best_hard_gate_pass"] = float(hard_gate_pass)
            row["best_eligible"] = float(best_eligible)
            row["candidate_checkpoint_saved"] = 0.0
            row["candidate_actor_checkpoint"] = ""
            row["candidate_train_checkpoint"] = ""
            if best_eligible:
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
            else:
                no_improve_iters = 0

            candidate_interval = int(args.candidate_checkpoint_interval)
            candidate_due = (
                candidate_interval > 0
                and iter_idx >= int(args.best_min_iter)
                and (iter_idx % candidate_interval) == 0
            )
            if candidate_due and (
                int(args.candidate_checkpoint_require_gate) == 0 or hard_gate_pass
            ):
                candidate_actor = candidate_checkpoint_dir / f"ppo_actor_iter_{iter_idx:04d}.pt"
                candidate_train = candidate_checkpoint_dir / f"ppo_train_iter_{iter_idx:04d}.pt"
                row["candidate_checkpoint_saved"] = 1.0
                row["candidate_actor_checkpoint"] = str(candidate_actor)
                row["candidate_train_checkpoint"] = str(candidate_train)
                torch.save(actor_checkpoint, candidate_actor)
                torch.save(train_checkpoint, candidate_train)

            _write_csv(history, metrics_csv)
            _write_summary(
                path=summary_json,
                args=args,
                device=device,
                reward_cfg=reward_cfg,
                loaded_compatible=loaded_compatible,
                loaded_total=loaded_total,
                actor_trainable_count=actor_trainable_count,
                actor_total_count=actor_total_count,
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
                    f"exp={row['rollout_expiry_drop_rate_mean']:.4f} "
                    f"expR={row.get('rollout_expiry_reward_rate_mean', row['rollout_expiry_drop_rate_mean']):.4f} "
                    f"blank={row['rollout_blank_ratio_mean']:.4f} "
                    f"drop={row['rollout_post_decode_drop_ratio_mean']:.4f} "
                    f"fill={row['rollout_action_fill_ratio_mean']:.4f} "
                    f"capDrop={row['rollout_cap_drop_count_mean']:.2f} "
                    f"fbOK={row['rollout_fallback_success_count_mean']:.2f} "
                    f"safeFill={row['rollout_safe_fill_count_mean']:.2f} "
                    f"blankBudget={row['rollout_blank_budget_repair_count_mean']:.2f} "
                    f"blankCnt={row['rollout_final_blank_count_mean']:.2f} "
                    f"dropCap={row['rollout_post_decode_demand_cap_count_mean']:.2f} "
                    f"dropSlot={row['rollout_post_decode_no_slot_count_mean']:.2f} "
                    f"dropOth={row['rollout_post_decode_other_drop_count_mean']:.2f} "
                    f"uniq_ue={row['rollout_action_unique_ue_mean']:.2f} "
                    f"cand_ue={row['rollout_candidate_unique_ue_mean']:.2f} "
                    f"demand_ue={row['rollout_demand_active_ue_mean']:.2f} "
                    f"cand_cov={row['rollout_candidate_coverage_ratio_mean']:.3f} "
                    f"budgChg={row.get('rollout_candidate_budgeted_changed_count_mean', 0.0):.1f} "
                    f"budgNoHr={row.get('rollout_candidate_budgeted_no_headroom_count_mean', 0.0):.1f} "
                    f"tailBiasSel={row.get('rollout_candidate_budgeted_tail_bias_selected_mean', 0.0):.3f} "
                    f"rescueBiasSel={row.get('rollout_candidate_budgeted_rescue_bias_selected_mean', 0.0):.3f} "
                    f"unsrv_ue={row['rollout_unserved_demand_ue_mean']:.2f} "
                    f"tailPot={row.get('rollout_tail_potential_after_mean', 0.0):.3f} "
                    f"tailDelta={row.get('rollout_tail_potential_delta_mean', 0.0):.3f} "
                    f"expPot={row.get('rollout_expiry_potential_after_mean', 0.0):.3f} "
                    f"expDelta={row.get('rollout_expiry_potential_delta_mean', 0.0):.3f} "
                    f"resc={row.get('rollout_rescue_credit_mean', 0.0):.3f} "
                    f"miss={row.get('rollout_rescue_miss_mean', 0.0):.3f} "
                    f"streak90={row['rollout_no_service_streak_p90_mean']:.1f} "
                    f"streakMax={row['rollout_no_service_streak_max_mean']:.1f} "
                    f"srv100_p10={row['rollout_recent100_service_rate_p10_mean']:.3f} "
                    f"ueRateM={row.get('rollout_ue_macro_rate_proxy_mbps_mean', 0.0):.2f} "
                    f"ueRateP10={row.get('rollout_ue_p10_rate_proxy_mbps_mean', 0.0):.2f} "
                    f"uePktM={row.get('rollout_ue_macro_packet_rate_mbps_mean', 0.0):.2f} "
                    f"uePktP10={row.get('rollout_ue_p10_packet_rate_mbps_mean', 0.0):.2f} "
                    f"debt100_p90={row['rollout_recent100_debt_score_p90_bytes_mean']/1024.0:.1f}KB "
                    f"bklg_p90={row['rollout_backlog_p90_bytes_mean']/1024.0:.1f}KB "
                    f"pktDelay={row.get('rollout_packet_delay_weighted_mean_ms', 0.0):.2f}ms "
                    f"pktRate={row.get('rollout_packet_effective_service_rate_mbps_mean', 0.0):.3f} "
                    f"pktRatePkt={row.get('rollout_packet_effective_service_rate_per_packet_mean_mbps_mean', 0.0):.3f} "
                    f"pktDone={row.get('rollout_packet_completed_count_mean', 0.0):.2f} "
                    f"tbGap={row.get('rollout_tb_err_gap_mean', 0.0):.4f} "
                    f"gatePen={row.get('rollout_tb_err_goodput_gate_penalty_mean', 0.0):.3f} "
                    f"bestGate={row.get('best_hard_gate_pass', 1.0):.0f}/"
                    f"{row.get('best_hard_gate_fail_count', 0.0):.0f} "
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
                    f"teacher_aux={row['teacher_aux_loss']:.4f}*{row['teacher_aux_coef']:.3f} "
                    f"success_aux={row['candidate_success_aux_loss']:.4f}*{row['candidate_success_aux_coef']:.3f} "
                    f"succ_calib={row['candidate_success_calib_loss']:.4f}*{row['candidate_success_calib_coef']:.3f} "
                    f"max_calib={row['candidate_success_max_calib_loss']:.4f}*{row['candidate_success_max_calib_coef']:.3f} "
                    f"rank={row['candidate_pairwise_rank_loss']:.4f}*{row['candidate_pairwise_rank_coef']:.3f} "
                    f"rescue={row.get('candidate_rescue_aux_loss', 0.0):.4f}*"
                    f"{row.get('candidate_rescue_aux_coef', 0.0):.3f} "
                    f"decode_gp={row['candidate_decode_goodput_loss']:.4f}*{row['candidate_decode_goodput_coef']:.3f} "
                    f"decodeT={row['candidate_decode_goodput_target_mean']:.3f}/"
                    f"{row['candidate_decode_goodput_positive_ratio']:.3f} "
                    f"deploy_dec={row.get('candidate_budgeted_greedy_decode_loss', row['candidate_argmax_decode_loss']):.4f}*"
                    f"{row['candidate_argmax_decode_coef']:.3f} "
                    f"deployKeep={row['candidate_argmax_decode_keep_ratio']:.3f} "
                    f"blank_tgt={row['candidate_blank_target_loss']:.4f}*{row['candidate_blank_target_coef']:.3f} "
                    f"blankT={row['candidate_blank_target_ratio']:.3f}/{row['candidate_blank_target_weighted_ratio']:.3f} "
                    f"raw/block={row.get('candidate_blank_target_raw_ratio', 0.0):.3f}/"
                    f"{row.get('candidate_blank_target_service_block_ratio', 0.0):.3f}/"
                    f"{row.get('candidate_blank_target_service_fill_ratio', 0.0):.3f} "
                    f"shadowBlank={row['candidate_shadow_blank_ratio']:.3f} "
                    f"maxT[p10,p50,p90]={row['candidate_success_max_target_p10']:.3f},"
                    f"{row['candidate_success_max_target_p50']:.3f},{row['candidate_success_max_target_p90']:.3f} "
                    f"succMAE={row['candidate_success_prob_mae']:.3f} "
                    f"blankD[p50,p90]={row['candidate_blank_decision_logit_p50']:.2f},"
                    f"{row['candidate_blank_decision_logit_p90']:.2f}"
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
        actor_trainable_count=actor_trainable_count,
        actor_total_count=actor_total_count,
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
