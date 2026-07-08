from __future__ import annotations

import math
from typing import Dict, Mapping

import torch

from .types import GraphFeatureSnapshot

_UE_BUFFER_IDX = 0
_UE_TTL_SLACK_IDX = 9
_PRG_CONFLICT_IDX = 6
_PRG_ICI_IDX = 7

REWARD_TERM_IDX_THROUGHPUT_MBPS = 0
REWARD_TERM_IDX_TOTAL_BUFFER_MB = 1
REWARD_TERM_IDX_TB_ERR_RATE = 2
REWARD_TERM_IDX_FAIRNESS_JAIN = 3
REWARD_TERM_IDX_GOODPUT_MBPS = 4
REWARD_TERM_IDX_SCHED_WB_SINR_DB = 5
REWARD_TERM_IDX_PRG_UTILIZATION_RATIO = 6
REWARD_TERM_IDX_GOODPUT_SPECTRAL_EFFICIENCY_BPSHZ = 7
REWARD_TERM_IDX_PRG_REUSE_RATIO = 8
REWARD_TERM_IDX_EXPIRED_BYTES = 9
REWARD_TERM_IDX_EXPIRED_PACKETS = 10
REWARD_TERM_IDX_EXPIRY_DROP_RATE = 11
REWARD_TERM_IDX_PACKET_COMPLETED_COUNT = 12
REWARD_TERM_IDX_PACKET_DELIVERED_BITS = 13
REWARD_TERM_IDX_PACKET_SYSTEM_TIME_MS = 14
REWARD_TERM_IDX_PACKET_EFFECTIVE_SERVICE_RATE_MBPS = 15
REWARD_TERM_IDX_PACKET_EFFECTIVE_SERVICE_RATE_PER_PACKET_MEAN_MBPS = 16


def extract_reward_metrics(info: Mapping) -> Dict[str, float]:
    reward_terms = info.get("reward_terms", (0.0,) * 17)
    try:
        throughput_mbps = float(reward_terms[REWARD_TERM_IDX_THROUGHPUT_MBPS])
        total_buffer_mb = float(reward_terms[REWARD_TERM_IDX_TOTAL_BUFFER_MB])
        tb_err_rate = float(reward_terms[REWARD_TERM_IDX_TB_ERR_RATE])
        fairness_jain = float(reward_terms[REWARD_TERM_IDX_FAIRNESS_JAIN])
        goodput_mbps = float(reward_terms[REWARD_TERM_IDX_GOODPUT_MBPS]) if len(reward_terms) >= 5 else throughput_mbps
        sched_wb_sinr_db = float(reward_terms[REWARD_TERM_IDX_SCHED_WB_SINR_DB]) if len(reward_terms) >= 6 else 0.0
        prg_utilization_ratio = float(reward_terms[REWARD_TERM_IDX_PRG_UTILIZATION_RATIO]) if len(reward_terms) >= 7 else 0.0
        goodput_spectral_efficiency_bpshz = (
            float(reward_terms[REWARD_TERM_IDX_GOODPUT_SPECTRAL_EFFICIENCY_BPSHZ]) if len(reward_terms) >= 8 else 0.0
        )
        prg_reuse_ratio = float(reward_terms[REWARD_TERM_IDX_PRG_REUSE_RATIO]) if len(reward_terms) >= 9 else 0.0
        expired_bytes = float(reward_terms[REWARD_TERM_IDX_EXPIRED_BYTES]) if len(reward_terms) >= 10 else 0.0
        expired_packets = float(reward_terms[REWARD_TERM_IDX_EXPIRED_PACKETS]) if len(reward_terms) >= 11 else 0.0
        expiry_drop_rate = float(reward_terms[REWARD_TERM_IDX_EXPIRY_DROP_RATE]) if len(reward_terms) >= 12 else 0.0
        packet_completed_count = (
            float(reward_terms[REWARD_TERM_IDX_PACKET_COMPLETED_COUNT]) if len(reward_terms) >= 13 else 0.0
        )
        packet_delivered_bits = (
            float(reward_terms[REWARD_TERM_IDX_PACKET_DELIVERED_BITS]) if len(reward_terms) >= 14 else 0.0
        )
        packet_system_time_ms = (
            float(reward_terms[REWARD_TERM_IDX_PACKET_SYSTEM_TIME_MS]) if len(reward_terms) >= 15 else 0.0
        )
        packet_effective_service_rate_mbps = (
            float(reward_terms[REWARD_TERM_IDX_PACKET_EFFECTIVE_SERVICE_RATE_MBPS])
            if len(reward_terms) >= 16
            else 0.0
        )
        packet_effective_service_rate_per_packet_mean_mbps = (
            float(reward_terms[REWARD_TERM_IDX_PACKET_EFFECTIVE_SERVICE_RATE_PER_PACKET_MEAN_MBPS])
            if len(reward_terms) >= 17
            else 0.0
        )
        packet_delay_mean_ms = packet_system_time_ms / max(packet_completed_count, 1.0)
    except Exception:
        throughput_mbps = 0.0
        total_buffer_mb = 0.0
        tb_err_rate = 0.0
        fairness_jain = 0.0
        goodput_mbps = 0.0
        sched_wb_sinr_db = 0.0
        prg_utilization_ratio = 0.0
        goodput_spectral_efficiency_bpshz = 0.0
        prg_reuse_ratio = 0.0
        expired_bytes = 0.0
        expired_packets = 0.0
        expiry_drop_rate = 0.0
        packet_completed_count = 0.0
        packet_delivered_bits = 0.0
        packet_system_time_ms = 0.0
        packet_effective_service_rate_mbps = 0.0
        packet_effective_service_rate_per_packet_mean_mbps = 0.0
        packet_delay_mean_ms = 0.0

    return {
        "goodput_mbps": goodput_mbps,
        "throughput_mbps": throughput_mbps,
        "total_buffer_mb": total_buffer_mb,
        "tb_err_rate": tb_err_rate,
        "fairness_jain": fairness_jain,
        "sched_wb_sinr_db": sched_wb_sinr_db,
        "expired_bytes": expired_bytes,
        "expired_packets": expired_packets,
        "expiry_drop_rate": expiry_drop_rate,
        "prg_utilization_ratio": prg_utilization_ratio,
        "goodput_spectral_efficiency_bpshz": goodput_spectral_efficiency_bpshz,
        "prg_reuse_ratio": prg_reuse_ratio,
        "packet_completed_count": packet_completed_count,
        "packet_delivered_bits": packet_delivered_bits,
        "packet_system_time_ms": packet_system_time_ms,
        "packet_delay_mean_ms": packet_delay_mean_ms,
        "packet_effective_service_rate_mbps": packet_effective_service_rate_mbps,
        "packet_effective_service_rate_per_packet_mean_mbps": packet_effective_service_rate_per_packet_mean_mbps,
    }


def compute_stageb_v1_reward(
    *,
    raw_env_reward: float,
    reward_metrics: Mapping[str, float],
    snapshot: GraphFeatureSnapshot,
    goodput_weight: float = 1.0,
    tb_err_weight: float = 8.0,
    expiry_weight: float = 6.0,
    conflict_weight: float = 1.0,
    ici_weight: float = 1.0,
    urgent_backlog_weight: float = 0.5,
    urgent_ttl_ms: float = 20.0,
    selected_prg_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    ue = snapshot.ue_features.values
    prg = snapshot.prg_features.values

    ttl_slack_ms = ue[:, _UE_TTL_SLACK_IDX]
    buffer_bytes = torch.clamp_min(ue[:, _UE_BUFFER_IDX], 0.0)
    urgent_mask = (ttl_slack_ms >= 0.0) & (ttl_slack_ms < float(urgent_ttl_ms))
    urgent_backlog_term = torch.log1p(buffer_bytes / 3000.0)
    urgent_backlog_penalty = float((urgent_mask.float() * urgent_backlog_term).mean().item())

    prg_view = prg
    if selected_prg_mask is not None:
        mask = torch.as_tensor(selected_prg_mask, dtype=torch.bool)
        if mask.shape != (prg.shape[0],):
            raise ValueError(
                f"selected_prg_mask shape mismatch: got={tuple(mask.shape)} expected={(prg.shape[0],)}"
            )
        if bool(mask.any().item()):
            prg_view = prg[mask]

    conflict_mean = float(torch.clamp(prg_view[:, _PRG_CONFLICT_IDX], 0.0, 1.0).mean().item())
    ici_mean = float(torch.clamp(prg_view[:, _PRG_ICI_IDX], 0.0, 1.0).mean().item())

    shaped_reward = (
        goodput_weight * float(reward_metrics.get("goodput_mbps", raw_env_reward))
        - tb_err_weight * float(reward_metrics.get("tb_err_rate", 0.0))
        - expiry_weight * float(reward_metrics.get("expiry_drop_rate", 0.0))
        - conflict_weight * conflict_mean
        - ici_weight * ici_mean
        - urgent_backlog_weight * urgent_backlog_penalty
    )
    return {
        "reward_shaped": float(shaped_reward),
        "reward_raw_env": float(raw_env_reward),
        "reward_conflict_mean": conflict_mean,
        "reward_ici_mean": ici_mean,
        "reward_urgent_backlog_penalty": urgent_backlog_penalty,
    }


def _compute_active_prg_goodput_efficiency(
    goodput_spectral_efficiency_bpshz: float,
    prg_utilization_ratio: float,
) -> float:
    k_util_floor = 0.35
    k_efficiency_cap = 24.0
    effective_util = max(k_util_floor, min(1.0, max(0.0, float(prg_utilization_ratio))))
    return min(k_efficiency_cap, max(0.0, float(goodput_spectral_efficiency_bpshz)) / effective_util)


def _compute_harmful_packing_penalty(
    throughput_mbps: float,
    goodput_mbps: float,
    prg_utilization_ratio: float,
    prg_reuse_ratio: float,
) -> float:
    if throughput_mbps <= 1.0e-6:
        tx_waste_ratio = 0.0
    else:
        wasted_mbps = max(0.0, float(throughput_mbps) - float(goodput_mbps))
        tx_waste_ratio = min(1.0, wasted_mbps / float(throughput_mbps))
    dense_packing_pressure = min(1.0, max(0.0, float(prg_utilization_ratio))) * (
        0.25 + 0.75 * min(1.0, max(0.0, float(prg_reuse_ratio)))
    )
    return tx_waste_ratio * dense_packing_pressure


def compute_stageb_blankaware_reward(
    *,
    raw_env_reward: float,
    reward_metrics: Mapping[str, float],
    goodput_weight: float = 1.0,
    tb_err_weight: float = 8.0,
    expiry_weight: float = 6.0,
    conflict_weight: float = 1.0,
    ici_weight: float = 1.0,
    urgent_backlog_weight: float = 0.5,
    actual_unserved_backlog_bytes: float = 0.0,
    actual_unserved_demand_ue_count: float = 0.0,
    backlog_debt_weight: float = 0.0,
    packet_rate_weight: float = 0.0,
    packet_per_packet_rate_weight: float = 0.0,
    packet_completion_weight: float = 0.0,
    packet_delay_weight: float = 0.0,
    ue_macro_rate_proxy_weight: float = 0.0,
    ue_p10_rate_proxy_gap_weight: float = 0.0,
    ue_macro_packet_rate_weight: float = 0.0,
    ue_p10_packet_rate_gap_weight: float = 0.0,
    packet_rate_scale_mbps: float = 10.0,
    packet_per_packet_rate_scale_mbps: float = 24.0,
    packet_completion_scale: float = 36.0,
    packet_delay_scale_ms: float = 10.0,
    ue_rate_proxy_scale_mbps: float = 8.0,
    ue_p10_rate_proxy_target_mbps: float = 0.0,
    ue_packet_rate_scale_mbps: float = 8.0,
    ue_p10_packet_rate_target_mbps: float = 0.0,
    ue_macro_rate_proxy_mbps: float = 0.0,
    ue_p10_rate_proxy_mbps: float = 0.0,
    ue_macro_packet_rate_mbps: float = 0.0,
    ue_p10_packet_rate_mbps: float = 0.0,
    aged_backlog_weight: float = 0.0,
    aged_backlog_bytes: float = 0.0,
    aged_backlog_streak: float = 0.0,
    unserved_ue_weight: float = 0.0,
    total_ue_count: float = 1.0,
    service_gap_weight: float = 0.0,
    service_rate_p10: float = 1.0,
    service_rate_target: float = 0.0,
    util_floor_weight: float = 0.0,
    util_floor_target: float = 0.0,
    util_floor_tb_err_guard: float = 1.0,
    util_floor_tb_err_softness: float = 0.0,
    missed_safe_opportunity_count: float = 0.0,
    missed_safe_opportunity_weight: float = 0.0,
    missed_safe_opportunity_scale: float = 51.0,
    prg_efficiency_weight: float = -1.0,
    harmful_packing_weight: float = -1.0,
    tb_err_target: float = -1.0,
    tb_err_over_target_weight: float = 0.0,
    tb_err_goodput_gate_weight: float = 0.0,
    tail_potential_before: float = 0.0,
    tail_potential_after: float = 0.0,
    tail_delta_weight: float = 0.0,
    tail_pressure_weight: float = 0.0,
    tail_delta_clip: float = 1.0,
    tail_potential_discount: float = 1.0,
    expiry_potential_before: float = 0.0,
    expiry_potential_after: float = 0.0,
    expiry_delta_weight: float = 0.0,
    expiry_pressure_weight: float = 0.0,
    expiry_delta_clip: float = 1.0,
    expiry_potential_discount: float = 1.0,
    rescue_credit: float = 0.0,
    rescue_miss: float = 0.0,
    rescue_credit_weight: float = 0.0,
    rescue_miss_weight: float = 0.0,
) -> Dict[str, float]:
    prg_utilization_ratio = float(reward_metrics.get("prg_utilization_ratio", 0.0))
    tb_err_rate = float(reward_metrics.get("tb_err_rate", 0.0))
    active_prg_goodput_efficiency = _compute_active_prg_goodput_efficiency(
        float(reward_metrics.get("goodput_spectral_efficiency_bpshz", 0.0)),
        prg_utilization_ratio,
    )
    harmful_packing_penalty = _compute_harmful_packing_penalty(
        float(reward_metrics.get("throughput_mbps", 0.0)),
        float(reward_metrics.get("goodput_mbps", 0.0)),
        prg_utilization_ratio,
        float(reward_metrics.get("prg_reuse_ratio", 0.0)),
    )
    # Preserve current behavior under the existing CLI defaults while letting
    # HGraph PPO reward_cfg actually control the blank-aware objective.
    goodput_coef = 0.05 * (float(goodput_weight) / 1.0)
    default_efficiency_coef = 0.25 * (float(ici_weight) / 1.0)
    tb_err_coef = 2.0 * (float(tb_err_weight) / 8.0)
    expiry_coef = 4.0 * (float(expiry_weight) / 6.0)
    default_packing_coef = 10.0 * (float(conflict_weight) / 1.0)
    efficiency_coef = (
        max(0.0, float(prg_efficiency_weight))
        if float(prg_efficiency_weight) >= 0.0
        else default_efficiency_coef
    )
    packing_coef = (
        max(0.0, float(harmful_packing_weight))
        if float(harmful_packing_weight) >= 0.0
        else default_packing_coef
    )
    fairness_coef = 0.25 * (float(urgent_backlog_weight) / 0.5)
    backlog_debt_coef = max(0.0, float(backlog_debt_weight))
    backlog_debt_penalty = math.log1p(max(0.0, float(actual_unserved_backlog_bytes)) / 3000.0)
    packet_rate_coef = max(0.0, float(packet_rate_weight))
    packet_per_packet_rate_coef = max(0.0, float(packet_per_packet_rate_weight))
    packet_completion_coef = max(0.0, float(packet_completion_weight))
    packet_delay_coef = max(0.0, float(packet_delay_weight))
    ue_macro_rate_proxy_coef = max(0.0, float(ue_macro_rate_proxy_weight))
    ue_p10_rate_proxy_gap_coef = max(0.0, float(ue_p10_rate_proxy_gap_weight))
    ue_macro_packet_rate_coef = max(0.0, float(ue_macro_packet_rate_weight))
    ue_p10_packet_rate_gap_coef = max(0.0, float(ue_p10_packet_rate_gap_weight))
    aged_backlog_coef = max(0.0, float(aged_backlog_weight))
    unserved_ue_coef = max(0.0, float(unserved_ue_weight))
    service_gap_coef = max(0.0, float(service_gap_weight))
    util_floor_coef = max(0.0, float(util_floor_weight))
    missed_safe_coef = max(0.0, float(missed_safe_opportunity_weight))
    packet_rate_bonus = math.log1p(
        max(0.0, float(reward_metrics.get("packet_effective_service_rate_mbps", 0.0)))
        / max(1.0e-6, float(packet_rate_scale_mbps))
    )
    packet_per_packet_rate_bonus = math.log1p(
        max(
            0.0,
            float(reward_metrics.get("packet_effective_service_rate_per_packet_mean_mbps", 0.0)),
        )
        / max(1.0e-6, float(packet_per_packet_rate_scale_mbps))
    )
    packet_completion_bonus = math.log1p(
        max(0.0, float(reward_metrics.get("packet_completed_count", 0.0)))
        / max(1.0e-6, float(packet_completion_scale))
    )
    packet_delay_mean_ms = max(0.0, float(reward_metrics.get("packet_delay_mean_ms", 0.0)))
    packet_delay_penalty = math.log1p(packet_delay_mean_ms / max(1.0e-6, float(packet_delay_scale_ms)))
    ue_rate_proxy_scale = max(1.0e-6, float(ue_rate_proxy_scale_mbps))
    ue_macro_rate_proxy_bonus = math.log1p(
        max(0.0, float(ue_macro_rate_proxy_mbps)) / ue_rate_proxy_scale
    )
    ue_p10_rate_proxy_gap = max(
        0.0,
        float(ue_p10_rate_proxy_target_mbps) - float(ue_p10_rate_proxy_mbps),
    )
    ue_p10_rate_proxy_gap_penalty = math.log1p(ue_p10_rate_proxy_gap / ue_rate_proxy_scale)
    ue_packet_rate_scale = max(1.0e-6, float(ue_packet_rate_scale_mbps))
    ue_macro_packet_rate_bonus = math.log1p(
        max(0.0, float(ue_macro_packet_rate_mbps)) / ue_packet_rate_scale
    )
    ue_p10_packet_rate_gap = max(
        0.0,
        float(ue_p10_packet_rate_target_mbps) - float(ue_p10_packet_rate_mbps),
    )
    ue_p10_packet_rate_gap_penalty = math.log1p(ue_p10_packet_rate_gap / ue_packet_rate_scale)
    aged_backlog_penalty = math.log1p(max(0.0, float(aged_backlog_bytes)) / 3000.0)
    aged_backlog_penalty *= 1.0 + min(5.0, max(0.0, float(aged_backlog_streak)) / 20.0)
    unserved_ue_penalty = max(0.0, float(actual_unserved_demand_ue_count)) / max(1.0, float(total_ue_count))
    service_gap_penalty = max(0.0, float(service_rate_target) - float(service_rate_p10))
    util_floor_gap = max(0.0, float(util_floor_target) - prg_utilization_ratio)
    if util_floor_gap <= 0.0 or util_floor_coef <= 0.0:
        util_floor_gate = 0.0
    else:
        tb_guard = float(util_floor_tb_err_guard)
        tb_softness = max(0.0, float(util_floor_tb_err_softness))
        if tb_softness <= 0.0:
            util_floor_gate = 1.0 if tb_err_rate <= tb_guard else 0.0
        else:
            util_floor_gate = min(1.0, max(0.0, (tb_guard + tb_softness - tb_err_rate) / tb_softness))
    util_floor_penalty = util_floor_gate * util_floor_gap
    missed_safe_penalty = math.log1p(max(0.0, float(missed_safe_opportunity_count))) / math.log1p(
        max(1.0, float(missed_safe_opportunity_scale))
    )
    tb_err_target_value = float(tb_err_target)
    tb_err_gap = max(0.0, tb_err_rate - tb_err_target_value) if tb_err_target_value >= 0.0 else 0.0
    tb_err_over_target_coef = max(0.0, float(tb_err_over_target_weight))
    tb_err_goodput_gate_coef = max(0.0, float(tb_err_goodput_gate_weight))
    tb_err_goodput_gate_penalty = (
        goodput_coef
        * max(0.0, float(reward_metrics.get("goodput_mbps", raw_env_reward)))
        * tb_err_gap
        * tb_err_goodput_gate_coef
    )
    tail_before = max(0.0, float(tail_potential_before))
    tail_after = max(0.0, float(tail_potential_after))
    tail_raw_delta = tail_before - max(0.0, float(tail_potential_discount)) * tail_after
    tail_clip = max(0.0, float(tail_delta_clip))
    tail_delta = max(-tail_clip, min(tail_clip, tail_raw_delta)) if tail_clip > 0.0 else tail_raw_delta
    tail_delta_coef = max(0.0, float(tail_delta_weight))
    tail_pressure_coef = max(0.0, float(tail_pressure_weight))

    expiry_before = max(0.0, float(expiry_potential_before))
    expiry_after = max(0.0, float(expiry_potential_after))
    expiry_raw_delta = expiry_before - max(0.0, float(expiry_potential_discount)) * expiry_after
    expiry_clip = max(0.0, float(expiry_delta_clip))
    expiry_delta = (
        max(-expiry_clip, min(expiry_clip, expiry_raw_delta)) if expiry_clip > 0.0 else expiry_raw_delta
    )
    expiry_delta_coef = max(0.0, float(expiry_delta_weight))
    expiry_pressure_coef = max(0.0, float(expiry_pressure_weight))
    rescue_credit_coef = max(0.0, float(rescue_credit_weight))
    rescue_miss_coef = max(0.0, float(rescue_miss_weight))
    shaped_reward = (
        goodput_coef * float(reward_metrics.get("goodput_mbps", raw_env_reward))
        + efficiency_coef * active_prg_goodput_efficiency
        - tb_err_coef * tb_err_rate
        - expiry_coef * float(reward_metrics.get("expiry_drop_rate", 0.0))
        - packing_coef * harmful_packing_penalty
        + fairness_coef * float(reward_metrics.get("fairness_jain", 0.0))
        - backlog_debt_coef * backlog_debt_penalty
        + packet_rate_coef * packet_rate_bonus
        + packet_per_packet_rate_coef * packet_per_packet_rate_bonus
        + packet_completion_coef * packet_completion_bonus
        - packet_delay_coef * packet_delay_penalty
        + ue_macro_rate_proxy_coef * ue_macro_rate_proxy_bonus
        - ue_p10_rate_proxy_gap_coef * ue_p10_rate_proxy_gap_penalty
        + ue_macro_packet_rate_coef * ue_macro_packet_rate_bonus
        - ue_p10_packet_rate_gap_coef * ue_p10_packet_rate_gap_penalty
        - aged_backlog_coef * aged_backlog_penalty
        - unserved_ue_coef * unserved_ue_penalty
        - service_gap_coef * service_gap_penalty
        - util_floor_coef * util_floor_penalty
        - missed_safe_coef * missed_safe_penalty
        - tb_err_over_target_coef * tb_err_gap
        - tb_err_goodput_gate_penalty
        + tail_delta_coef * tail_delta
        - tail_pressure_coef * tail_after
        + expiry_delta_coef * expiry_delta
        - expiry_pressure_coef * expiry_after
        + rescue_credit_coef * max(0.0, float(rescue_credit))
        - rescue_miss_coef * max(0.0, float(rescue_miss))
    )
    return {
        "reward_shaped": float(shaped_reward),
        "reward_raw_env": float(raw_env_reward),
        "reward_active_prg_goodput_efficiency": float(active_prg_goodput_efficiency),
        "reward_harmful_packing_penalty": float(harmful_packing_penalty),
        "reward_goodput_coef": float(goodput_coef),
        "reward_efficiency_coef": float(efficiency_coef),
        "reward_tb_err_coef": float(tb_err_coef),
        "reward_expiry_coef": float(expiry_coef),
        "reward_packing_coef": float(packing_coef),
        "reward_prg_efficiency_weight": float(prg_efficiency_weight),
        "reward_harmful_packing_weight": float(harmful_packing_weight),
        "reward_fairness_coef": float(fairness_coef),
        "reward_backlog_debt_coef": float(backlog_debt_coef),
        "reward_backlog_debt_penalty": float(backlog_debt_penalty),
        "reward_packet_rate_coef": float(packet_rate_coef),
        "reward_packet_rate_bonus": float(packet_rate_bonus),
        "reward_packet_per_packet_rate_coef": float(packet_per_packet_rate_coef),
        "reward_packet_per_packet_rate_bonus": float(packet_per_packet_rate_bonus),
        "reward_packet_completion_coef": float(packet_completion_coef),
        "reward_packet_completion_bonus": float(packet_completion_bonus),
        "reward_packet_delay_coef": float(packet_delay_coef),
        "reward_packet_delay_mean_ms": float(packet_delay_mean_ms),
        "reward_packet_delay_penalty": float(packet_delay_penalty),
        "reward_ue_macro_rate_proxy_coef": float(ue_macro_rate_proxy_coef),
        "reward_ue_macro_rate_proxy_mbps": float(ue_macro_rate_proxy_mbps),
        "reward_ue_macro_rate_proxy_bonus": float(ue_macro_rate_proxy_bonus),
        "reward_ue_p10_rate_proxy_gap_coef": float(ue_p10_rate_proxy_gap_coef),
        "reward_ue_p10_rate_proxy_mbps": float(ue_p10_rate_proxy_mbps),
        "reward_ue_p10_rate_proxy_target_mbps": float(ue_p10_rate_proxy_target_mbps),
        "reward_ue_p10_rate_proxy_gap_mbps": float(ue_p10_rate_proxy_gap),
        "reward_ue_p10_rate_proxy_gap_penalty": float(ue_p10_rate_proxy_gap_penalty),
        "reward_ue_macro_packet_rate_coef": float(ue_macro_packet_rate_coef),
        "reward_ue_macro_packet_rate_mbps": float(ue_macro_packet_rate_mbps),
        "reward_ue_macro_packet_rate_bonus": float(ue_macro_packet_rate_bonus),
        "reward_ue_p10_packet_rate_gap_coef": float(ue_p10_packet_rate_gap_coef),
        "reward_ue_p10_packet_rate_mbps": float(ue_p10_packet_rate_mbps),
        "reward_ue_p10_packet_rate_target_mbps": float(ue_p10_packet_rate_target_mbps),
        "reward_ue_p10_packet_rate_gap_mbps": float(ue_p10_packet_rate_gap),
        "reward_ue_p10_packet_rate_gap_penalty": float(ue_p10_packet_rate_gap_penalty),
        "reward_aged_backlog_coef": float(aged_backlog_coef),
        "reward_aged_backlog_penalty": float(aged_backlog_penalty),
        "reward_unserved_ue_coef": float(unserved_ue_coef),
        "reward_unserved_ue_penalty": float(unserved_ue_penalty),
        "reward_service_gap_coef": float(service_gap_coef),
        "reward_service_gap_penalty": float(service_gap_penalty),
        "reward_service_rate_p10": float(service_rate_p10),
        "reward_service_rate_target": float(service_rate_target),
        "reward_util_floor_coef": float(util_floor_coef),
        "reward_util_floor_target": float(util_floor_target),
        "reward_util_floor_tb_err_guard": float(util_floor_tb_err_guard),
        "reward_util_floor_tb_err_softness": float(util_floor_tb_err_softness),
        "reward_util_floor_gate": float(util_floor_gate),
        "reward_util_floor_gap": float(util_floor_gap),
        "reward_util_floor_penalty": float(util_floor_penalty),
        "reward_missed_safe_opportunity_coef": float(missed_safe_coef),
        "reward_missed_safe_opportunity_count": float(missed_safe_opportunity_count),
        "reward_missed_safe_opportunity_penalty": float(missed_safe_penalty),
        "reward_tb_err_target": float(tb_err_target_value),
        "reward_tb_err_gap": float(tb_err_gap),
        "reward_tb_err_over_target_coef": float(tb_err_over_target_coef),
        "reward_tb_err_over_target_penalty": float(tb_err_gap),
        "reward_tb_err_goodput_gate_coef": float(tb_err_goodput_gate_coef),
        "reward_tb_err_goodput_gate_penalty": float(tb_err_goodput_gate_penalty),
        "reward_tail_potential_before": float(tail_before),
        "reward_tail_potential_after": float(tail_after),
        "reward_tail_delta_raw": float(tail_raw_delta),
        "reward_tail_delta": float(tail_delta),
        "reward_tail_delta_coef": float(tail_delta_coef),
        "reward_tail_pressure_coef": float(tail_pressure_coef),
        "reward_tail_delta_clip": float(tail_clip),
        "reward_tail_potential_discount": float(max(0.0, float(tail_potential_discount))),
        "reward_expiry_potential_before": float(expiry_before),
        "reward_expiry_potential_after": float(expiry_after),
        "reward_expiry_delta_raw": float(expiry_raw_delta),
        "reward_expiry_delta": float(expiry_delta),
        "reward_expiry_delta_coef": float(expiry_delta_coef),
        "reward_expiry_pressure_coef": float(expiry_pressure_coef),
        "reward_expiry_delta_clip": float(expiry_clip),
        "reward_expiry_potential_discount": float(max(0.0, float(expiry_potential_discount))),
        "reward_rescue_credit": float(max(0.0, float(rescue_credit))),
        "reward_rescue_miss": float(max(0.0, float(rescue_miss))),
        "reward_rescue_credit_coef": float(rescue_credit_coef),
        "reward_rescue_miss_coef": float(rescue_miss_coef),
        "reward_packet_completed_count": float(reward_metrics.get("packet_completed_count", 0.0)),
        "reward_packet_effective_service_rate_mbps": float(
            reward_metrics.get("packet_effective_service_rate_mbps", 0.0)
        ),
        "reward_packet_effective_service_rate_per_packet_mean_mbps": float(
            reward_metrics.get("packet_effective_service_rate_per_packet_mean_mbps", 0.0)
        ),
        "reward_actual_unserved_demand_ue_count": float(actual_unserved_demand_ue_count),
    }


def compute_stageb_tailguard_simple_reward(
    *,
    raw_env_reward: float,
    reward_metrics: Mapping[str, float],
    goodput_weight: float = 1.0,
    tb_err_weight: float = 1.0,
    expiry_weight: float = 6.0,
    actual_unserved_backlog_bytes: float = 0.0,
    actual_unserved_demand_ue_count: float = 0.0,
    aged_backlog_bytes: float = 0.0,
    aged_backlog_streak: float = 0.0,
    total_ue_count: float = 1.0,
    service_gap_weight: float = 1.25,
    service_rate_p10: float = 1.0,
    service_rate_target: float = 0.75,
) -> Dict[str, float]:
    """Three-signal reward: goodput, reliability, and a max-pooled tail guard."""

    goodput_mbps = float(reward_metrics.get("goodput_mbps", raw_env_reward))
    tb_err_rate = max(0.0, float(reward_metrics.get("tb_err_rate", 0.0)))
    expiry_drop_rate = max(0.0, float(reward_metrics.get("expiry_drop_rate", 0.0)))

    goodput_norm = min(1.25, max(0.0, goodput_mbps) / 3000.0)
    reliability_penalty = tb_err_rate + max(0.0, float(expiry_weight)) * expiry_drop_rate

    def _debt_pressure(value_bytes: float) -> float:
        return min(1.5, math.log1p(max(0.0, float(value_bytes)) / 3000.0) / 6.0)

    target = max(0.05, float(service_rate_target))
    service_gap_pressure = max(0.0, target - float(service_rate_p10)) / target
    current_debt_pressure = _debt_pressure(actual_unserved_backlog_bytes)
    historical_debt_pressure = _debt_pressure(aged_backlog_bytes)
    streak_pressure = min(1.5, max(0.0, float(aged_backlog_streak)) / 10.0)
    unserved_pressure = min(
        1.5,
        4.0 * max(0.0, float(actual_unserved_demand_ue_count)) / max(1.0, float(total_ue_count)),
    )
    tail_pressure = max(
        service_gap_pressure,
        current_debt_pressure,
        historical_debt_pressure,
        streak_pressure,
        unserved_pressure,
    )

    shaped_reward = (
        max(0.0, float(goodput_weight)) * goodput_norm
        - max(0.0, float(tb_err_weight)) * reliability_penalty
        - max(0.0, float(service_gap_weight)) * tail_pressure
    )
    return {
        "reward_shaped": float(shaped_reward),
        "reward_raw_env": float(raw_env_reward),
        "reward_goodput_norm": float(goodput_norm),
        "reward_reliability_penalty": float(reliability_penalty),
        "reward_tail_pressure": float(tail_pressure),
        "reward_service_gap_pressure": float(service_gap_pressure),
        "reward_current_debt_pressure": float(current_debt_pressure),
        "reward_historical_debt_pressure": float(historical_debt_pressure),
        "reward_streak_pressure": float(streak_pressure),
        "reward_unserved_pressure": float(unserved_pressure),
        "reward_goodput_coef": float(max(0.0, float(goodput_weight))),
        "reward_reliability_coef": float(max(0.0, float(tb_err_weight))),
        "reward_tail_coef": float(max(0.0, float(service_gap_weight))),
        "reward_expiry_multiplier": float(max(0.0, float(expiry_weight))),
        "reward_service_rate_p10": float(service_rate_p10),
        "reward_service_rate_target": float(service_rate_target),
        "reward_actual_unserved_demand_ue_count": float(actual_unserved_demand_ue_count),
    }


def compute_stageb_tailcredit_reward(
    *,
    raw_env_reward: float,
    reward_metrics: Mapping[str, float],
    goodput_weight: float = 1.0,
    tb_err_weight: float = 1.0,
    expiry_weight: float = 6.0,
    tail_potential_before: float = 0.0,
    tail_potential_after: float = 0.0,
    tail_delta_weight: float = 1.0,
    tail_pressure_weight: float = 0.15,
    tail_delta_clip: float = 1.0,
    tail_potential_discount: float = 1.0,
) -> Dict[str, float]:
    """Goodput/reliability reward plus local tail-risk potential shaping.

    The tail term is a soft credit-assignment signal: serving high-risk UEs is
    rewarded when it reduces the tail potential from the pre-action state to
    the post-action state. A small residual pressure term keeps sustained tail
    risk visible without turning the reward into a hard scheduling rule.
    """

    goodput_mbps = float(reward_metrics.get("goodput_mbps", raw_env_reward))
    tb_err_rate = max(0.0, float(reward_metrics.get("tb_err_rate", 0.0)))
    expiry_drop_rate = max(0.0, float(reward_metrics.get("expiry_drop_rate", 0.0)))

    goodput_norm = min(1.25, max(0.0, goodput_mbps) / 3000.0)
    reliability_penalty = tb_err_rate + max(0.0, float(expiry_weight)) * expiry_drop_rate

    before = max(0.0, float(tail_potential_before))
    after = max(0.0, float(tail_potential_after))
    raw_delta = before - max(0.0, float(tail_potential_discount)) * after
    clip = max(0.0, float(tail_delta_clip))
    tail_delta = max(-clip, min(clip, raw_delta)) if clip > 0.0 else raw_delta

    shaped_reward = (
        max(0.0, float(goodput_weight)) * goodput_norm
        - max(0.0, float(tb_err_weight)) * reliability_penalty
        + max(0.0, float(tail_delta_weight)) * tail_delta
        - max(0.0, float(tail_pressure_weight)) * after
    )
    return {
        "reward_shaped": float(shaped_reward),
        "reward_raw_env": float(raw_env_reward),
        "reward_goodput_norm": float(goodput_norm),
        "reward_reliability_penalty": float(reliability_penalty),
        "reward_tail_potential_before": float(before),
        "reward_tail_potential_after": float(after),
        "reward_tail_delta_raw": float(raw_delta),
        "reward_tail_delta": float(tail_delta),
        "reward_goodput_coef": float(max(0.0, float(goodput_weight))),
        "reward_reliability_coef": float(max(0.0, float(tb_err_weight))),
        "reward_tail_delta_coef": float(max(0.0, float(tail_delta_weight))),
        "reward_tail_pressure_coef": float(max(0.0, float(tail_pressure_weight))),
        "reward_tail_delta_clip": float(clip),
        "reward_tail_potential_discount": float(max(0.0, float(tail_potential_discount))),
        "reward_expiry_multiplier": float(max(0.0, float(expiry_weight))),
    }


def compute_stageb_tailcredit_expiry_reward(
    *,
    raw_env_reward: float,
    reward_metrics: Mapping[str, float],
    goodput_weight: float = 1.0,
    tb_err_weight: float = 1.0,
    expiry_weight: float = 6.0,
    tail_potential_before: float = 0.0,
    tail_potential_after: float = 0.0,
    tail_delta_weight: float = 1.0,
    tail_pressure_weight: float = 0.15,
    tail_delta_clip: float = 1.0,
    tail_potential_discount: float = 1.0,
    expiry_potential_before: float = 0.0,
    expiry_potential_after: float = 0.0,
    expiry_delta_weight: float = 1.0,
    expiry_pressure_weight: float = 0.25,
    expiry_delta_clip: float = 1.0,
    expiry_potential_discount: float = 1.0,
    rescue_credit: float = 0.0,
    rescue_miss: float = 0.0,
    rescue_credit_weight: float = 0.0,
    rescue_miss_weight: float = 0.0,
) -> Dict[str, float]:
    """v5/v6 reward: tail/expiry potential deltas plus optional rescue credit."""

    goodput_mbps = float(reward_metrics.get("goodput_mbps", raw_env_reward))
    tb_err_rate = max(0.0, float(reward_metrics.get("tb_err_rate", 0.0)))
    expiry_drop_rate = max(0.0, float(reward_metrics.get("expiry_drop_rate", 0.0)))

    goodput_norm = min(1.25, max(0.0, goodput_mbps) / 3000.0)
    reliability_penalty = tb_err_rate + max(0.0, float(expiry_weight)) * expiry_drop_rate

    tail_before = max(0.0, float(tail_potential_before))
    tail_after = max(0.0, float(tail_potential_after))
    tail_raw_delta = tail_before - max(0.0, float(tail_potential_discount)) * tail_after
    tail_clip = max(0.0, float(tail_delta_clip))
    tail_delta = max(-tail_clip, min(tail_clip, tail_raw_delta)) if tail_clip > 0.0 else tail_raw_delta

    expiry_before = max(0.0, float(expiry_potential_before))
    expiry_after = max(0.0, float(expiry_potential_after))
    expiry_raw_delta = expiry_before - max(0.0, float(expiry_potential_discount)) * expiry_after
    expiry_clip = max(0.0, float(expiry_delta_clip))
    expiry_delta = (
        max(-expiry_clip, min(expiry_clip, expiry_raw_delta)) if expiry_clip > 0.0 else expiry_raw_delta
    )

    shaped_reward = (
        max(0.0, float(goodput_weight)) * goodput_norm
        - max(0.0, float(tb_err_weight)) * reliability_penalty
        + max(0.0, float(tail_delta_weight)) * tail_delta
        - max(0.0, float(tail_pressure_weight)) * tail_after
        + max(0.0, float(expiry_delta_weight)) * expiry_delta
        - max(0.0, float(expiry_pressure_weight)) * expiry_after
        + max(0.0, float(rescue_credit_weight)) * max(0.0, float(rescue_credit))
        - max(0.0, float(rescue_miss_weight)) * max(0.0, float(rescue_miss))
    )
    return {
        "reward_shaped": float(shaped_reward),
        "reward_raw_env": float(raw_env_reward),
        "reward_goodput_norm": float(goodput_norm),
        "reward_reliability_penalty": float(reliability_penalty),
        "reward_tail_potential_before": float(tail_before),
        "reward_tail_potential_after": float(tail_after),
        "reward_tail_delta_raw": float(tail_raw_delta),
        "reward_tail_delta": float(tail_delta),
        "reward_expiry_potential_before": float(expiry_before),
        "reward_expiry_potential_after": float(expiry_after),
        "reward_expiry_delta_raw": float(expiry_raw_delta),
        "reward_expiry_delta": float(expiry_delta),
        "reward_goodput_coef": float(max(0.0, float(goodput_weight))),
        "reward_reliability_coef": float(max(0.0, float(tb_err_weight))),
        "reward_tail_delta_coef": float(max(0.0, float(tail_delta_weight))),
        "reward_tail_pressure_coef": float(max(0.0, float(tail_pressure_weight))),
        "reward_tail_delta_clip": float(tail_clip),
        "reward_tail_potential_discount": float(max(0.0, float(tail_potential_discount))),
        "reward_expiry_delta_coef": float(max(0.0, float(expiry_delta_weight))),
        "reward_expiry_pressure_coef": float(max(0.0, float(expiry_pressure_weight))),
        "reward_expiry_delta_clip": float(expiry_clip),
        "reward_expiry_potential_discount": float(max(0.0, float(expiry_potential_discount))),
        "reward_expiry_multiplier": float(max(0.0, float(expiry_weight))),
        "reward_rescue_credit": float(max(0.0, float(rescue_credit))),
        "reward_rescue_miss": float(max(0.0, float(rescue_miss))),
        "reward_rescue_credit_coef": float(max(0.0, float(rescue_credit_weight))),
        "reward_rescue_miss_coef": float(max(0.0, float(rescue_miss_weight))),
    }
