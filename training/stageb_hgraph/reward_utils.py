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
    packet_completion_weight: float = 0.0,
    packet_rate_scale_mbps: float = 10.0,
    packet_completion_scale: float = 36.0,
    aged_backlog_weight: float = 0.0,
    aged_backlog_bytes: float = 0.0,
    aged_backlog_streak: float = 0.0,
) -> Dict[str, float]:
    active_prg_goodput_efficiency = _compute_active_prg_goodput_efficiency(
        float(reward_metrics.get("goodput_spectral_efficiency_bpshz", 0.0)),
        float(reward_metrics.get("prg_utilization_ratio", 0.0)),
    )
    harmful_packing_penalty = _compute_harmful_packing_penalty(
        float(reward_metrics.get("throughput_mbps", 0.0)),
        float(reward_metrics.get("goodput_mbps", 0.0)),
        float(reward_metrics.get("prg_utilization_ratio", 0.0)),
        float(reward_metrics.get("prg_reuse_ratio", 0.0)),
    )
    # Preserve current behavior under the existing CLI defaults while letting
    # HGraph PPO reward_cfg actually control the blank-aware objective.
    goodput_coef = 0.05 * (float(goodput_weight) / 1.0)
    efficiency_coef = 0.25 * (float(ici_weight) / 1.0)
    tb_err_coef = 2.0 * (float(tb_err_weight) / 8.0)
    expiry_coef = 4.0 * (float(expiry_weight) / 6.0)
    packing_coef = 10.0 * (float(conflict_weight) / 1.0)
    fairness_coef = 0.25 * (float(urgent_backlog_weight) / 0.5)
    backlog_debt_coef = max(0.0, float(backlog_debt_weight))
    backlog_debt_penalty = math.log1p(max(0.0, float(actual_unserved_backlog_bytes)) / 3000.0)
    packet_rate_coef = max(0.0, float(packet_rate_weight))
    packet_completion_coef = max(0.0, float(packet_completion_weight))
    aged_backlog_coef = max(0.0, float(aged_backlog_weight))
    packet_rate_bonus = math.log1p(
        max(0.0, float(reward_metrics.get("packet_effective_service_rate_mbps", 0.0)))
        / max(1.0e-6, float(packet_rate_scale_mbps))
    )
    packet_completion_bonus = math.log1p(
        max(0.0, float(reward_metrics.get("packet_completed_count", 0.0)))
        / max(1.0e-6, float(packet_completion_scale))
    )
    aged_backlog_penalty = math.log1p(max(0.0, float(aged_backlog_bytes)) / 3000.0)
    aged_backlog_penalty *= 1.0 + min(5.0, max(0.0, float(aged_backlog_streak)) / 20.0)
    shaped_reward = (
        goodput_coef * float(reward_metrics.get("goodput_mbps", raw_env_reward))
        + efficiency_coef * active_prg_goodput_efficiency
        - tb_err_coef * float(reward_metrics.get("tb_err_rate", 0.0))
        - expiry_coef * float(reward_metrics.get("expiry_drop_rate", 0.0))
        - packing_coef * harmful_packing_penalty
        + fairness_coef * float(reward_metrics.get("fairness_jain", 0.0))
        - backlog_debt_coef * backlog_debt_penalty
        + packet_rate_coef * packet_rate_bonus
        + packet_completion_coef * packet_completion_bonus
        - aged_backlog_coef * aged_backlog_penalty
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
        "reward_fairness_coef": float(fairness_coef),
        "reward_backlog_debt_coef": float(backlog_debt_coef),
        "reward_backlog_debt_penalty": float(backlog_debt_penalty),
        "reward_packet_rate_coef": float(packet_rate_coef),
        "reward_packet_rate_bonus": float(packet_rate_bonus),
        "reward_packet_completion_coef": float(packet_completion_coef),
        "reward_packet_completion_bonus": float(packet_completion_bonus),
        "reward_aged_backlog_coef": float(aged_backlog_coef),
        "reward_aged_backlog_penalty": float(aged_backlog_penalty),
        "reward_packet_completed_count": float(reward_metrics.get("packet_completed_count", 0.0)),
        "reward_packet_effective_service_rate_mbps": float(
            reward_metrics.get("packet_effective_service_rate_mbps", 0.0)
        ),
        "reward_actual_unserved_demand_ue_count": float(actual_unserved_demand_ue_count),
    }
