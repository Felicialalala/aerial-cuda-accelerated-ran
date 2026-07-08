from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Dict, List

import torch

from .mask_checker import MaskCheckReport
from .types import SparseEntityGraph


@dataclass(frozen=True)
class GraphSanityStats:
    n_cell: int
    n_ue: int
    n_prg_tokens: int
    candidate_mode: str
    post_eq_layer_dim: int
    prg_feat_dim: int
    available_prg_ratio: float
    prg_prev_assigned_ratio: float
    n_edges_cc: int
    n_edges_cu: int
    n_edges_cp: int
    n_edges_up: int
    n_edges_pp: int
    per_cell_assoc_ue_counts: Dict[int, int]
    per_cell_legal_ue_counts: Dict[int, int]
    per_cell_candidate_ue_counts: Dict[int, int]
    per_cell_candidate_coverage_ratio: Dict[int, float]
    per_prg_candidate_edge_min: int
    per_prg_candidate_edge_mean: float
    per_prg_candidate_edge_max: int
    conflict_edge_count: int
    prg_without_candidates: int
    prg_without_legal_actions: int
    illegal_cross_cell_up_edges: int
    unschedulable_up_edges: int
    prg_top1_sinr_db_mean: float
    prg_top1_sinr_db_p10: float
    prg_top1_sinr_db_p50: float
    prg_top1_sinr_db_p90: float
    prg_top2_gap_db_mean: float
    prg_top2_gap_db_p50: float
    prg_top2_gap_db_p90: float
    prg_neighbor_max_top1_sinr_db_mean: float
    prg_neighbor_max_top1_sinr_db_p50: float
    prg_neighbor_max_top1_sinr_db_p90: float
    prg_same_prg_conflict_ratio_mean: float
    prg_same_prg_conflict_ratio_p90: float
    prg_same_prg_conflict_ratio_max: float
    prg_ici_proxy_mean: float
    prg_ici_proxy_p90: float
    prg_ici_proxy_max: float
    prg_post_eq_p10_sinr_db_mean: float
    prg_post_eq_p10_sinr_db_p10: float
    prg_post_eq_p10_sinr_db_p50: float
    prg_post_eq_mean_sinr_db_mean: float
    prg_post_eq_mean_sinr_db_p10: float
    prg_post_eq_mean_sinr_db_p50: float
    prg_backlog_weighted_expected_goodput_mbps_mean: float
    prg_backlog_weighted_expected_goodput_mbps_p90: float
    prg_backlog_weighted_expected_goodput_mbps_max: float
    prg_low_sinr_hol_ttl_weight_mean: float
    prg_low_sinr_hol_ttl_weight_p90: float
    prg_low_sinr_hol_ttl_weight_max: float

    def to_dict(self) -> Dict[str, float | int | str | Dict[int, int]]:
        return asdict(self)


class GraphSanityLogger:
    @staticmethod
    def _quantile(x: torch.Tensor, q: float) -> float:
        if x.numel() == 0:
            return 0.0
        return float(torch.quantile(x, q).item())

    def _summarize_prg_features(self, prg_features: torch.Tensor) -> Dict[str, float]:
        availability = torch.ones((prg_features.shape[0],), dtype=prg_features.dtype, device=prg_features.device)
        stats_view = prg_features
        feat_dim = int(stats_view.shape[1]) if stats_view.dim() == 2 else 0

        top1 = stats_view[:, 0]
        top2_gap = stats_view[:, 1]
        neighbor_max = stats_view[:, 4]
        conflict = stats_view[:, 6]
        ici = stats_view[:, 7]
        prev_assigned = stats_view[:, 2]

        out = {
            "available_prg_ratio": float(availability.mean().item()) if availability.numel() > 0 else 0.0,
            "prg_prev_assigned_ratio": float(prev_assigned.mean().item()) if prev_assigned.numel() > 0 else 0.0,
            "prg_top1_sinr_db_mean": float(top1.mean().item()) if top1.numel() > 0 else 0.0,
            "prg_top1_sinr_db_p10": self._quantile(top1, 0.10),
            "prg_top1_sinr_db_p50": self._quantile(top1, 0.50),
            "prg_top1_sinr_db_p90": self._quantile(top1, 0.90),
            "prg_top2_gap_db_mean": float(top2_gap.mean().item()) if top2_gap.numel() > 0 else 0.0,
            "prg_top2_gap_db_p50": self._quantile(top2_gap, 0.50),
            "prg_top2_gap_db_p90": self._quantile(top2_gap, 0.90),
            "prg_neighbor_max_top1_sinr_db_mean": float(neighbor_max.mean().item()) if neighbor_max.numel() > 0 else 0.0,
            "prg_neighbor_max_top1_sinr_db_p50": self._quantile(neighbor_max, 0.50),
            "prg_neighbor_max_top1_sinr_db_p90": self._quantile(neighbor_max, 0.90),
            "prg_same_prg_conflict_ratio_mean": float(conflict.mean().item()) if conflict.numel() > 0 else 0.0,
            "prg_same_prg_conflict_ratio_p90": self._quantile(conflict, 0.90),
            "prg_same_prg_conflict_ratio_max": float(conflict.max().item()) if conflict.numel() > 0 else 0.0,
            "prg_ici_proxy_mean": float(ici.mean().item()) if ici.numel() > 0 else 0.0,
            "prg_ici_proxy_p90": self._quantile(ici, 0.90),
            "prg_ici_proxy_max": float(ici.max().item()) if ici.numel() > 0 else 0.0,
        }
        if feat_dim > 8:
            p10 = stats_view[:, 8]
            out.update(
                {
                    "prg_post_eq_p10_sinr_db_mean": float(p10.mean().item()) if p10.numel() > 0 else 0.0,
                    "prg_post_eq_p10_sinr_db_p10": self._quantile(p10, 0.10),
                    "prg_post_eq_p10_sinr_db_p50": self._quantile(p10, 0.50),
                }
            )
        if feat_dim > 9:
            mean_sinr = stats_view[:, 9]
            out.update(
                {
                    "prg_post_eq_mean_sinr_db_mean": float(mean_sinr.mean().item()) if mean_sinr.numel() > 0 else 0.0,
                    "prg_post_eq_mean_sinr_db_p10": self._quantile(mean_sinr, 0.10),
                    "prg_post_eq_mean_sinr_db_p50": self._quantile(mean_sinr, 0.50),
                }
            )
        if feat_dim > 10:
            bw_goodput = stats_view[:, 10]
            out.update(
                {
                    "prg_backlog_weighted_expected_goodput_mbps_mean": (
                        float(bw_goodput.mean().item()) if bw_goodput.numel() > 0 else 0.0
                    ),
                    "prg_backlog_weighted_expected_goodput_mbps_p90": self._quantile(bw_goodput, 0.90),
                    "prg_backlog_weighted_expected_goodput_mbps_max": (
                        float(bw_goodput.max().item()) if bw_goodput.numel() > 0 else 0.0
                    ),
                }
            )
        if feat_dim > 11:
            tail = stats_view[:, 11]
            out.update(
                {
                    "prg_low_sinr_hol_ttl_weight_mean": float(tail.mean().item()) if tail.numel() > 0 else 0.0,
                    "prg_low_sinr_hol_ttl_weight_p90": self._quantile(tail, 0.90),
                    "prg_low_sinr_hol_ttl_weight_max": float(tail.max().item()) if tail.numel() > 0 else 0.0,
                }
            )
        return out

    def summarize(self, graph: SparseEntityGraph, mask_report: MaskCheckReport) -> GraphSanityStats:
        meta = graph.snapshot.static_meta
        e_up = graph.edges["E_UP"].index
        action_mask_ue = meta.action_mask_ue
        per_cell_candidate_set: Dict[int, set[int]] = {c_idx: set() for c_idx in range(meta.n_cell)}
        per_cell_assoc_counts: Dict[int, int] = {}
        per_cell_legal_counts: Dict[int, int] = {}
        per_cell_coverage: Dict[int, float] = {}
        per_prg_counts = torch.zeros((meta.n_cell * meta.n_prg,), dtype=torch.long)

        for c_idx in range(meta.n_cell):
            assoc_mask = meta.action_mask_cell_ue[c_idx]
            legal_mask = assoc_mask if action_mask_ue is None else (assoc_mask & action_mask_ue)
            per_cell_assoc_counts[c_idx] = int(assoc_mask.to(torch.int64).sum().item())
            per_cell_legal_counts[c_idx] = int(legal_mask.to(torch.int64).sum().item())

        for edge_idx in range(e_up.shape[1]):
            ue_idx = int(e_up[0, edge_idx].item())
            prg_token = int(e_up[1, edge_idx].item())
            owner_cell = int(meta.prg_owner_cell[prg_token].item())
            per_cell_candidate_set[owner_cell].add(ue_idx)
            per_prg_counts[prg_token] += 1

        per_prg_list: List[int] = per_prg_counts.tolist()
        min_count = min(per_prg_list) if per_prg_list else 0
        mean_count = mean(per_prg_list) if per_prg_list else 0.0
        max_count = max(per_prg_list) if per_prg_list else 0
        per_cell_candidate_counts = {c_idx: len(ue_ids) for c_idx, ue_ids in per_cell_candidate_set.items()}
        for c_idx in range(meta.n_cell):
            denom = max(1, per_cell_legal_counts[c_idx])
            per_cell_coverage[c_idx] = float(per_cell_candidate_counts[c_idx]) / float(denom)
        prg_stats = self._summarize_prg_features(graph.snapshot.prg_features.values)

        return GraphSanityStats(
            n_cell=meta.n_cell,
            n_ue=meta.n_ue,
            n_prg_tokens=meta.n_cell * meta.n_prg,
            candidate_mode=meta.candidate_mode,
            post_eq_layer_dim=(
                int(meta.post_eq_sinr.shape[-1])
                if meta.post_eq_sinr is not None and meta.post_eq_sinr.dim() == 3
                else (1 if meta.post_eq_sinr is not None else 0)
            ),
            prg_feat_dim=int(graph.snapshot.prg_features.values.shape[1]),
            available_prg_ratio=prg_stats["available_prg_ratio"],
            prg_prev_assigned_ratio=prg_stats["prg_prev_assigned_ratio"],
            n_edges_cc=graph.edges["E_CC"].num_edges,
            n_edges_cu=graph.edges["E_CU"].num_edges,
            n_edges_cp=graph.edges["E_CP"].num_edges,
            n_edges_up=graph.edges["E_UP"].num_edges,
            n_edges_pp=graph.edges["E_PP"].num_edges,
            per_cell_assoc_ue_counts=per_cell_assoc_counts,
            per_cell_legal_ue_counts=per_cell_legal_counts,
            per_cell_candidate_ue_counts=per_cell_candidate_counts,
            per_cell_candidate_coverage_ratio=per_cell_coverage,
            per_prg_candidate_edge_min=min_count,
            per_prg_candidate_edge_mean=float(mean_count),
            per_prg_candidate_edge_max=max_count,
            conflict_edge_count=graph.edges["E_PP"].num_edges,
            prg_without_candidates=mask_report.prg_without_candidates,
            prg_without_legal_actions=mask_report.prg_without_legal_actions,
            illegal_cross_cell_up_edges=mask_report.illegal_cross_cell_up_edges,
            unschedulable_up_edges=mask_report.unschedulable_up_edges,
            prg_top1_sinr_db_mean=prg_stats["prg_top1_sinr_db_mean"],
            prg_top1_sinr_db_p10=prg_stats["prg_top1_sinr_db_p10"],
            prg_top1_sinr_db_p50=prg_stats["prg_top1_sinr_db_p50"],
            prg_top1_sinr_db_p90=prg_stats["prg_top1_sinr_db_p90"],
            prg_top2_gap_db_mean=prg_stats["prg_top2_gap_db_mean"],
            prg_top2_gap_db_p50=prg_stats["prg_top2_gap_db_p50"],
            prg_top2_gap_db_p90=prg_stats["prg_top2_gap_db_p90"],
            prg_neighbor_max_top1_sinr_db_mean=prg_stats["prg_neighbor_max_top1_sinr_db_mean"],
            prg_neighbor_max_top1_sinr_db_p50=prg_stats["prg_neighbor_max_top1_sinr_db_p50"],
            prg_neighbor_max_top1_sinr_db_p90=prg_stats["prg_neighbor_max_top1_sinr_db_p90"],
            prg_same_prg_conflict_ratio_mean=prg_stats["prg_same_prg_conflict_ratio_mean"],
            prg_same_prg_conflict_ratio_p90=prg_stats["prg_same_prg_conflict_ratio_p90"],
            prg_same_prg_conflict_ratio_max=prg_stats["prg_same_prg_conflict_ratio_max"],
            prg_ici_proxy_mean=prg_stats["prg_ici_proxy_mean"],
            prg_ici_proxy_p90=prg_stats["prg_ici_proxy_p90"],
            prg_ici_proxy_max=prg_stats["prg_ici_proxy_max"],
            prg_post_eq_p10_sinr_db_mean=prg_stats.get("prg_post_eq_p10_sinr_db_mean", 0.0),
            prg_post_eq_p10_sinr_db_p10=prg_stats.get("prg_post_eq_p10_sinr_db_p10", 0.0),
            prg_post_eq_p10_sinr_db_p50=prg_stats.get("prg_post_eq_p10_sinr_db_p50", 0.0),
            prg_post_eq_mean_sinr_db_mean=prg_stats.get("prg_post_eq_mean_sinr_db_mean", 0.0),
            prg_post_eq_mean_sinr_db_p10=prg_stats.get("prg_post_eq_mean_sinr_db_p10", 0.0),
            prg_post_eq_mean_sinr_db_p50=prg_stats.get("prg_post_eq_mean_sinr_db_p50", 0.0),
            prg_backlog_weighted_expected_goodput_mbps_mean=prg_stats.get(
                "prg_backlog_weighted_expected_goodput_mbps_mean", 0.0
            ),
            prg_backlog_weighted_expected_goodput_mbps_p90=prg_stats.get(
                "prg_backlog_weighted_expected_goodput_mbps_p90", 0.0
            ),
            prg_backlog_weighted_expected_goodput_mbps_max=prg_stats.get(
                "prg_backlog_weighted_expected_goodput_mbps_max", 0.0
            ),
            prg_low_sinr_hol_ttl_weight_mean=prg_stats.get("prg_low_sinr_hol_ttl_weight_mean", 0.0),
            prg_low_sinr_hol_ttl_weight_p90=prg_stats.get("prg_low_sinr_hol_ttl_weight_p90", 0.0),
            prg_low_sinr_hol_ttl_weight_max=prg_stats.get("prg_low_sinr_hol_ttl_weight_max", 0.0),
        )
