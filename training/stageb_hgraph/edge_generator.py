from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence

import torch

from .graph_ops import demand_active_ue_mask_from_features
from .types import EdgeSet, GraphFeatureSnapshot, SparseEntityGraph


CANDIDATE_RISK_FEATURE_DIM = 24


@dataclass(frozen=True)
class EdgeGeneratorConfig:
    ue_prg_topk: int = 4
    ue_prg_diversity_extra: int = 0
    candidate_mode: str = "all_demand_ues"
    include_pfq_score_anchor: bool = True
    include_candidate_risk_features: bool = False
    candidate_risk_feature_dim: int = CANDIDATE_RISK_FEATURE_DIM
    include_bidirectional_prg_conflict: bool = True
    shared_pool_size_factor: int = 3
    shared_pool_min_size: int = 6
    post_eq_diversity_pool_size_factor: int = 3
    post_eq_diversity_pool_min_size: int = 6


class SparseEdgeGenerator:
    def __init__(self, cfg: EdgeGeneratorConfig | None = None):
        self.cfg = cfg or EdgeGeneratorConfig()

    def build_graph(self, snapshot: GraphFeatureSnapshot) -> SparseEntityGraph:
        edges = self.build_edges(snapshot)
        return SparseEntityGraph(snapshot=snapshot, edges=edges)

    def build_edges(self, snapshot: GraphFeatureSnapshot) -> Dict[str, EdgeSet]:
        meta = snapshot.static_meta
        edges = {
            "E_CC": self._build_cell_cell_edges(meta.cell_adjacency, meta.cell_edge_attr),
            "E_CU": self._build_cell_ue_edges(meta.action_mask_cell_ue),
            "E_CP": self._build_cell_prg_edges(meta.n_cell, meta.n_prg),
            "E_UP": self._build_ue_prg_edges(snapshot),
            "E_PP": self._build_prg_prg_edges(meta.cell_adjacency, meta.n_prg),
        }
        return edges

    @staticmethod
    def _build_cell_cell_edges(
        cell_adjacency: torch.Tensor,
        cell_edge_attr: torch.Tensor | None,
    ) -> EdgeSet:
        if cell_adjacency.numel() == 0:
            index = torch.zeros((2, 0), dtype=torch.long)
        else:
            index = cell_adjacency.t().contiguous().to(dtype=torch.long)
        return EdgeSet(
            name="E_CC",
            src_type="cell",
            dst_type="cell",
            index=index,
            attr=cell_edge_attr,
        )

    @staticmethod
    def _build_cell_ue_edges(action_mask_cell_ue: torch.Tensor) -> EdgeSet:
        pairs = torch.nonzero(action_mask_cell_ue, as_tuple=False)
        index = torch.zeros((2, 0), dtype=torch.long) if pairs.numel() == 0 else pairs.t().contiguous().to(torch.long)
        return EdgeSet(name="E_CU", src_type="cell", dst_type="ue", index=index)

    @staticmethod
    def _build_cell_prg_edges(n_cell: int, n_prg: int) -> EdgeSet:
        src: List[int] = []
        dst: List[int] = []
        for c_idx in range(n_cell):
            for prg_idx in range(n_prg):
                src.append(c_idx)
                dst.append(c_idx * n_prg + prg_idx)
        index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        return EdgeSet(name="E_CP", src_type="cell", dst_type="prg", index=index)

    def _build_ue_prg_edges(self, snapshot: GraphFeatureSnapshot) -> EdgeSet:
        meta = snapshot.static_meta
        if meta.post_eq_sinr is not None:
            return self._build_ue_prg_edges_post_eq(snapshot)
        return self._build_ue_prg_edges_shared_pool(snapshot)

    def _build_ue_prg_edges_shared_pool(self, snapshot: GraphFeatureSnapshot) -> EdgeSet:
        meta = snapshot.static_meta
        ue = snapshot.ue_features.values
        prg = snapshot.prg_features.values.reshape(meta.n_cell, meta.n_prg, -1)
        src: List[int] = []
        dst: List[int] = []
        edge_attr: List[List[float]] = []

        shared_scores = self._shared_pool_scores(ue)
        urgency_scores = self._urgency_scores(ue)
        debt_scores = self._debt_scores(ue)
        wb_sinr_db = 10.0 * torch.log10(torch.clamp_min(ue[:, 2], 1.0e-9))
        pfq_anchor = (
            self._pfq_anchor_scores(wb_sinr_db, ue)
            if self.cfg.include_pfq_score_anchor
            else torch.zeros_like(wb_sinr_db)
        )
        for c_idx in range(meta.n_cell):
            legal_ues = self._schedulable_ues(meta, ue, c_idx)
            if legal_ues.numel() == 0:
                continue
            ranked_local = legal_ues[torch.argsort(shared_scores[legal_ues], descending=True)]
            best_wb_db = float(wb_sinr_db[ranked_local[0]].item())
            if self.cfg.candidate_mode == "all_demand_ues":
                ranked_pool = ranked_local
                pool_size = int(ranked_pool.numel())
            else:
                pool_size = min(
                    int(legal_ues.numel()),
                    max(
                        int(self.cfg.ue_prg_topk),
                        int(self.cfg.shared_pool_min_size),
                        int(self.cfg.ue_prg_topk) * int(self.cfg.shared_pool_size_factor),
                    ),
                )
                ranked_pool = ranked_local[:pool_size]
            pool_size = min(
                int(ranked_pool.numel()),
                max(1, pool_size),
            )
            stride = max(1, pool_size // max(1, min(pool_size, int(self.cfg.ue_prg_topk))))
            for prg_idx in range(meta.n_prg):
                if not bool(meta.action_mask_prg_cell[c_idx, prg_idx].item()):
                    continue
                prg_token = c_idx * meta.n_prg + prg_idx
                prg_risk = self._prg_risk_proxy(prg[c_idx, prg_idx])
                if self.cfg.candidate_mode == "all_demand_ues":
                    local_pick = [int(ue_idx) for ue_idx in ranked_pool.tolist()]
                else:
                    offset = (prg_idx * stride) % max(1, pool_size)
                    local_pick = []
                    for shift in range(pool_size):
                        ranked_idx = int((offset + shift) % pool_size)
                        ue_idx = int(ranked_pool[ranked_idx].item())
                        if ue_idx in local_pick:
                            continue
                        local_pick.append(ue_idx)
                        if len(local_pick) >= int(self.cfg.ue_prg_topk):
                            break
                local_pick_tensor = torch.tensor(local_pick, dtype=torch.long, device=ue.device)
                relative_features = self._relative_candidate_features(
                    local_quality_db=wb_sinr_db[local_pick_tensor],
                    local_layer_min_quality_db=wb_sinr_db[local_pick_tensor],
                    local_layer_spread_db=torch.zeros_like(wb_sinr_db[local_pick_tensor]),
                    local_debt=debt_scores[local_pick_tensor],
                    local_tb_err=torch.clamp_min(ue[local_pick_tensor, 5], 0.0)
                    if ue.shape[1] > 5
                    else torch.zeros_like(wb_sinr_db[local_pick_tensor]),
                    local_backlog_bytes=torch.clamp_min(ue[local_pick_tensor, 0], 0.0)
                    if ue.shape[1] > 0
                    else torch.zeros_like(wb_sinr_db[local_pick_tensor]),
                    local_recent_sched_ratio=torch.clamp(ue[local_pick_tensor, 10], 0.0, 1.0)
                    if ue.shape[1] > 10
                    else torch.ones_like(wb_sinr_db[local_pick_tensor]),
                    prg_risk=float(prg_risk.item()),
                )
                for feature_pos, ue_idx in enumerate(local_pick):
                    src.append(int(ue_idx))
                    dst.append(prg_token)
                    edge_attr.append(
                        self._ue_prg_edge_features(
                            quality_db=float(wb_sinr_db[ue_idx].item()),
                            best_quality_db=best_wb_db,
                            urgency=float(urgency_scores[ue_idx].item()),
                            debt=float(debt_scores[ue_idx].item()),
                            prg_risk=float(prg_risk.item()),
                            ue_tb_err=float(ue[ue_idx, 5].item()) if ue.shape[1] > 5 else 0.0,
                            backlog_bytes=float(ue[ue_idx, 0].item()) if ue.shape[1] > 0 else 0.0,
                            recent_sched_ratio=float(ue[ue_idx, 10].item()) if ue.shape[1] > 10 else 1.0,
                            hol_delay_ms=float(ue[ue_idx, 8].item()) if ue.shape[1] > 8 else 0.0,
                            ttl_slack_ms=float(ue[ue_idx, 9].item()) if ue.shape[1] > 9 else -1.0,
                            recent_deficit=float(ue[ue_idx, 11].item()) if ue.shape[1] > 11 else 0.0,
                            relative_features=relative_features[feature_pos].tolist(),
                            pfq_anchor=float(pfq_anchor[ue_idx].item()),
                        )
                    )
        index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else None
        return EdgeSet(name="E_UP", src_type="ue", dst_type="prg", index=index, attr=attr)

    def _build_ue_prg_edges_post_eq(self, snapshot: GraphFeatureSnapshot) -> EdgeSet:
        meta = snapshot.static_meta
        post_eq = meta.post_eq_sinr
        if post_eq is None:
            raise RuntimeError("post_eq_sinr unexpectedly missing")
        layer_min_db = None
        layer_spread_db = None
        if post_eq.dim() == 3:
            post_eq_layer_db = 10.0 * torch.log10(torch.clamp_min(post_eq, 1.0e-9))
            layer_min_db = post_eq_layer_db.min(dim=-1).values
            layer_spread_db = post_eq_layer_db.max(dim=-1).values - layer_min_db
            post_eq = post_eq.mean(dim=-1)
        if post_eq.shape != (meta.n_ue, meta.n_prg):
            raise ValueError(
                f"post_eq_sinr must be [U,P] or [U,P,L], got shape={tuple(meta.post_eq_sinr.shape)}"
            )
        post_eq_db = 10.0 * torch.log10(torch.clamp_min(post_eq, 1.0e-9))
        if layer_min_db is None:
            layer_min_db = post_eq_db
        if layer_spread_db is None:
            layer_spread_db = torch.zeros_like(post_eq_db)

        ue = snapshot.ue_features.values
        prg = snapshot.prg_features.values.reshape(meta.n_cell, meta.n_prg, -1)
        src: List[int] = []
        dst: List[int] = []
        edge_attr: List[List[float]] = []

        urgency_scores = self._urgency_scores(ue)
        debt_scores = self._debt_scores(ue)
        for c_idx in range(meta.n_cell):
            legal_ues = self._schedulable_ues(meta, ue, c_idx)
            if legal_ues.numel() == 0:
                continue
            legal_urgency = urgency_scores[legal_ues]
            legal_debt = debt_scores[legal_ues]
            legal_tb_err = torch.clamp_min(ue[legal_ues, 5], 0.0)
            for prg_idx in range(meta.n_prg):
                if not bool(meta.action_mask_prg_cell[c_idx, prg_idx].item()):
                    continue
                prg_token = c_idx * meta.n_prg + prg_idx
                prg_risk = float(self._prg_risk_proxy(prg[c_idx, prg_idx]).item())
                local_quality_db = post_eq_db[legal_ues, prg_idx]
                local_layer_min_db = layer_min_db[legal_ues, prg_idx]
                local_layer_spread_db = layer_spread_db[legal_ues, prg_idx]
                local_pfq_anchor = (
                    self._pfq_anchor_scores(local_quality_db, ue[legal_ues])
                    if self.cfg.include_pfq_score_anchor
                    else torch.zeros_like(local_quality_db)
                )
                best_quality_db = float(local_quality_db.max().item())
                local_candidate_score = self._candidate_scores(
                    local_quality_db=local_quality_db,
                    local_layer_min_quality_db=local_layer_min_db,
                    local_layer_spread_db=local_layer_spread_db,
                    local_urgency=legal_urgency,
                    local_debt=legal_debt,
                    local_tb_err=legal_tb_err,
                )
                if self.cfg.candidate_mode == "all_demand_ues":
                    picked = torch.argsort(local_candidate_score, descending=True)
                else:
                    picked = self._select_post_eq_candidates(
                        local_quality_db=local_quality_db,
                        local_layer_min_quality_db=local_layer_min_db,
                        local_layer_spread_db=local_layer_spread_db,
                        local_urgency=legal_urgency,
                        local_debt=legal_debt,
                        local_tb_err=legal_tb_err,
                        prg_idx=prg_idx,
                    )
                if picked.numel() == 0:
                    continue
                local_backlog_bytes = (
                    torch.clamp_min(ue[legal_ues, 0], 0.0)
                    if ue.shape[1] > 0
                    else torch.zeros_like(local_quality_db)
                )
                local_recent_sched_ratio = (
                    torch.clamp(ue[legal_ues, 10], 0.0, 1.0)
                    if ue.shape[1] > 10
                    else torch.ones_like(local_quality_db)
                )
                relative_features = self._relative_candidate_features(
                    local_quality_db=local_quality_db[picked],
                    local_layer_min_quality_db=local_layer_min_db[picked],
                    local_layer_spread_db=local_layer_spread_db[picked],
                    local_debt=legal_debt[picked],
                    local_tb_err=legal_tb_err[picked],
                    local_backlog_bytes=local_backlog_bytes[picked],
                    local_recent_sched_ratio=local_recent_sched_ratio[picked],
                    prg_risk=prg_risk,
                )

                for feature_pos, local_rank in enumerate(picked.tolist()):
                    ue_idx = int(legal_ues[local_rank].item())
                    src.append(ue_idx)
                    dst.append(prg_token)
                    quality_db = float(local_quality_db[local_rank].item())
                    edge_attr.append(
                        self._ue_prg_edge_features(
                            quality_db=quality_db,
                            best_quality_db=best_quality_db,
                            urgency=float(legal_urgency[local_rank].item()),
                            debt=float(legal_debt[local_rank].item()),
                            prg_risk=prg_risk,
                            ue_tb_err=float(legal_tb_err[local_rank].item()),
                            backlog_bytes=float(ue[ue_idx, 0].item()) if ue.shape[1] > 0 else 0.0,
                            recent_sched_ratio=float(ue[ue_idx, 10].item()) if ue.shape[1] > 10 else 1.0,
                            hol_delay_ms=float(ue[ue_idx, 8].item()) if ue.shape[1] > 8 else 0.0,
                            ttl_slack_ms=float(ue[ue_idx, 9].item()) if ue.shape[1] > 9 else -1.0,
                            recent_deficit=float(ue[ue_idx, 11].item()) if ue.shape[1] > 11 else 0.0,
                            layer_min_quality_db=float(local_layer_min_db[local_rank].item()),
                            layer_spread_db=float(local_layer_spread_db[local_rank].item()),
                            relative_features=relative_features[feature_pos].tolist(),
                            pfq_anchor=float(local_pfq_anchor[local_rank].item()),
                        )
                    )
        index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else None
        return EdgeSet(name="E_UP", src_type="ue", dst_type="prg", index=index, attr=attr)

    def _select_post_eq_candidates(
        self,
        *,
        local_quality_db: torch.Tensor,
        local_layer_min_quality_db: torch.Tensor | None = None,
        local_layer_spread_db: torch.Tensor | None = None,
        local_urgency: torch.Tensor,
        local_debt: torch.Tensor,
        local_tb_err: torch.Tensor,
        prg_idx: int,
    ) -> torch.Tensor:
        candidate_count = int(local_quality_db.numel())
        topk = min(int(self.cfg.ue_prg_topk), candidate_count)
        if topk <= 0:
            return torch.zeros((0,), dtype=torch.long, device=local_quality_db.device)

        rank_score = self._candidate_scores(
            local_quality_db=local_quality_db,
            local_layer_min_quality_db=local_layer_min_quality_db,
            local_layer_spread_db=local_layer_spread_db,
            local_urgency=local_urgency,
            local_debt=local_debt,
            local_tb_err=local_tb_err,
        )
        candidate_rank = torch.argsort(rank_score, descending=True)
        core = candidate_rank[:topk]

        extra_limit = max(0, int(self.cfg.ue_prg_diversity_extra))
        remaining = candidate_rank[topk:]
        if extra_limit <= 0 or remaining.numel() == 0:
            return core

        extra_count = min(extra_limit, int(remaining.numel()))
        remaining_by_diversity = remaining[torch.argsort(rank_score[remaining], descending=True)]
        pool_size = min(
            int(remaining_by_diversity.numel()),
            max(
                extra_count,
                int(self.cfg.post_eq_diversity_pool_min_size),
                int(self.cfg.ue_prg_topk) * int(self.cfg.post_eq_diversity_pool_size_factor),
            ),
        )
        ranked_pool = remaining_by_diversity[:pool_size]
        if ranked_pool.numel() == 0:
            return core

        stride = max(1, pool_size // max(1, extra_count))
        offset = (prg_idx * stride) % max(1, pool_size)
        picked_extra: List[int] = []
        for shift in range(pool_size):
            if len(picked_extra) >= extra_count:
                break
            pool_pos = int((offset + shift) % pool_size)
            local_rank = int(ranked_pool[pool_pos].item())
            if local_rank in picked_extra:
                continue
            picked_extra.append(local_rank)
        if not picked_extra:
            return core
        extra = torch.tensor(picked_extra, dtype=torch.long, device=local_quality_db.device)
        return torch.cat([core, extra], dim=0)

    @staticmethod
    def _schedulable_ues(meta, ue_features: torch.Tensor, c_idx: int) -> torch.Tensor:
        legal_mask = meta.action_mask_cell_ue[c_idx]
        legal_mask = legal_mask & demand_active_ue_mask_from_features(
            ue_features,
            action_mask_ue=meta.action_mask_ue,
        )
        return torch.nonzero(legal_mask, as_tuple=False).flatten()

    def _build_prg_prg_edges(self, cell_adjacency: torch.Tensor, n_prg: int) -> EdgeSet:
        src: List[int] = []
        dst: List[int] = []
        for c_src, c_dst in cell_adjacency.tolist():
            for prg_idx in range(n_prg):
                src.append(c_src * n_prg + prg_idx)
                dst.append(c_dst * n_prg + prg_idx)
        index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        return EdgeSet(name="E_PP", src_type="prg", dst_type="prg", index=index)

    @staticmethod
    def _urgency_scores(ue_features: torch.Tensor) -> torch.Tensor:
        buffer_norm = torch.log1p(torch.clamp_min(ue_features[:, 0], 0.0))
        hol_norm = ue_features[:, 8] / (torch.max(ue_features[:, 8]).clamp_min(1.0))
        ttl_term = torch.where(
            ue_features[:, 9] >= 0.0,
            1.0 / (1.0 + torch.clamp_min(ue_features[:, 9], 0.0)),
            torch.zeros_like(ue_features[:, 9]),
        )
        return 0.45 * buffer_norm + 0.30 * hol_norm + 0.25 * ttl_term

    @staticmethod
    def _debt_scores(ue_features: torch.Tensor) -> torch.Tensor:
        return 0.6 * torch.clamp_min(ue_features[:, 11], 0.0) + 0.4 * (
            1.0 - torch.clamp(ue_features[:, 10], 0.0, 1.0)
        )

    @staticmethod
    def _normalize_score(score: torch.Tensor) -> torch.Tensor:
        if score.numel() <= 1:
            return torch.zeros_like(score)
        return (score - score.mean()) / score.std(unbiased=False).clamp_min(1.0)

    @staticmethod
    def _prg_risk_proxy(prg_feature: torch.Tensor) -> torch.Tensor:
        same_prg_conflict = torch.clamp(prg_feature[..., 6], 0.0, 1.0)
        ici_proxy = torch.clamp(prg_feature[..., 7], 0.0, 1.0)
        return torch.clamp(0.5 * same_prg_conflict + 0.5 * ici_proxy, 0.0, 1.0)

    def _candidate_risk_features(
        self,
        *,
        quality_db: float,
        layer_min_quality_db: float | None = None,
        layer_spread_db: float = 0.0,
        urgency: float,
        debt: float,
        prg_risk: float,
        ue_tb_err: float,
        backlog_bytes: float = 0.0,
        recent_sched_ratio: float = 1.0,
        hol_delay_ms: float = 0.0,
        ttl_slack_ms: float = -1.0,
        recent_deficit: float = 0.0,
        relative_features: Sequence[float] | None = None,
    ) -> List[float]:
        sinr_arg = max(-60.0, min(60.0, (float(quality_db) - 6.0) / 3.0))
        sinr_risk = 1.0 / (1.0 + math.exp(sinr_arg))
        layer_min_db = float(quality_db) if layer_min_quality_db is None else float(layer_min_quality_db)
        layer_arg = max(-60.0, min(60.0, (layer_min_db - 3.0) / 3.0))
        low_layer_risk = 1.0 / (1.0 + math.exp(layer_arg))
        layer_spread_norm = min(1.0, max(0.0, float(layer_spread_db) / 12.0))
        tb_risk = min(1.0, max(0.0, float(ue_tb_err)))
        prg_risk_clamped = min(1.0, max(0.0, float(prg_risk)))
        candidate_fail_risk = min(
            1.0,
            max(
                0.0,
                0.35 * sinr_risk
                + 0.25 * low_layer_risk
                + 0.20 * tb_risk
                + 0.15 * prg_risk_clamped
                + 0.05 * layer_spread_norm,
            ),
        )
        delay_pressure = min(1.0, max(0.0, 0.65 * float(urgency) + 0.35 * float(debt)))
        success_proxy = 1.0 - candidate_fail_risk
        demand_prg_need = min(1.5, max(0.0, math.log1p(max(0.0, float(backlog_bytes)) / 3000.0) / 4.0))
        service_gap = min(1.0, max(0.0, 1.0 - float(recent_sched_ratio)))
        marginal_need_proxy = min(
            1.0,
            max(0.0, 0.55 * demand_prg_need + 0.30 * float(debt) + 0.15 * service_gap),
        )
        expected_decode_value = success_proxy * marginal_need_proxy
        interference_cost_proxy = prg_risk_clamped * success_proxy * max(0.0, 1.0 - marginal_need_proxy)
        direct_context_features = self._candidate_direct_context_features(
            quality_db=quality_db,
            backlog_bytes=backlog_bytes,
            recent_sched_ratio=recent_sched_ratio,
            hol_delay_ms=hol_delay_ms,
            ttl_slack_ms=ttl_slack_ms,
            recent_deficit=recent_deficit,
        )
        features = [
            candidate_fail_risk,
            low_layer_risk * prg_risk_clamped,
            candidate_fail_risk * delay_pressure,
            layer_spread_norm,
            low_layer_risk * delay_pressure,
            expected_decode_value,
            marginal_need_proxy,
            interference_cost_proxy,
        ]
        if relative_features is not None:
            features.extend(float(value) for value in relative_features)
        features.extend(direct_context_features)
        dim = max(0, int(self.cfg.candidate_risk_feature_dim))
        if dim <= len(features):
            return features[:dim]
        return features + [0.0] * (dim - len(features))

    @staticmethod
    def _candidate_direct_context_features(
        *,
        quality_db: float,
        backlog_bytes: float,
        recent_sched_ratio: float,
        hol_delay_ms: float,
        ttl_slack_ms: float,
        recent_deficit: float,
    ) -> List[float]:
        hol = max(0.0, float(hol_delay_ms))
        ttl = float(ttl_slack_ms)
        ttl_known = 1.0 if ttl >= 0.0 else 0.0
        ttl_nonnegative = max(0.0, ttl)
        hol_norm = min(1.5, hol / 200.0)
        ttl_near_40 = (
            min(1.5, max(0.0, (40.0 - ttl_nonnegative) / 40.0))
            if ttl_known > 0.0
            else 0.0
        )
        ttl_critical_20 = (
            min(1.5, max(0.0, (20.0 - ttl_nonnegative) / 20.0))
            if ttl_known > 0.0
            else 0.0
        )
        if ttl_known > 0.0:
            age_to_deadline = min(1.0, max(0.0, hol / max(1.0, hol + ttl_nonnegative)))
        else:
            age_to_deadline = min(1.0, hol_norm / (1.0 + hol_norm))

        quality_lin = math.pow(10.0, max(-60.0, min(60.0, float(quality_db))) / 10.0)
        spectral_eff = math.log2(1.0 + max(1.0e-9, quality_lin))
        bytes_scale = min(4.0, max(0.5, spectral_eff / 2.0))
        est_bytes_per_prg = min(12000.0, max(1200.0, 3000.0 * bytes_scale))
        backlog = max(0.0, float(backlog_bytes))
        marginal_need_prg = (backlog + 3000.0) / max(1.0, est_bytes_per_prg)
        decode_need_prg = (backlog + 9000.0) / max(1.0, est_bytes_per_prg)
        after_one_margin = (backlog + 3000.0 - est_bytes_per_prg) / max(1.0, est_bytes_per_prg)

        return [
            hol_norm,
            ttl_known,
            ttl_near_40,
            ttl_critical_20,
            age_to_deadline,
            min(1.0, max(0.0, 1.0 - float(recent_sched_ratio))),
            min(1.5, max(0.0, float(recent_deficit))),
            min(1.5, max(0.0, decode_need_prg / 4.0)),
            min(1.5, max(0.0, marginal_need_prg / 4.0)),
            min(1.0, max(-1.0, after_one_margin)),
        ]

    @staticmethod
    def _candidate_risk_tensors(
        *,
        local_quality_db: torch.Tensor,
        local_layer_min_quality_db: torch.Tensor | None = None,
        local_layer_spread_db: torch.Tensor | None = None,
        local_debt: torch.Tensor,
        local_tb_err: torch.Tensor,
        local_backlog_bytes: torch.Tensor,
        local_recent_sched_ratio: torch.Tensor,
        prg_risk: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer_min_db = local_quality_db if local_layer_min_quality_db is None else local_layer_min_quality_db
        layer_spread = torch.zeros_like(local_quality_db) if local_layer_spread_db is None else local_layer_spread_db
        sinr_risk = torch.sigmoid(-((local_quality_db - 6.0) / 3.0))
        low_layer_risk = torch.sigmoid(-((layer_min_db - 3.0) / 3.0))
        layer_spread_norm = torch.clamp(layer_spread / 12.0, 0.0, 1.0)
        tb_risk = torch.clamp(local_tb_err, 0.0, 1.0)
        prg_risk_tensor = torch.full_like(local_quality_db, float(min(1.0, max(0.0, prg_risk))))
        candidate_fail_risk = torch.clamp(
            0.35 * sinr_risk
            + 0.25 * low_layer_risk
            + 0.20 * tb_risk
            + 0.15 * prg_risk_tensor
            + 0.05 * layer_spread_norm,
            0.0,
            1.0,
        )
        demand_prg_need = torch.clamp(torch.log1p(torch.clamp_min(local_backlog_bytes, 0.0) / 3000.0) / 4.0, 0.0, 1.5)
        service_gap = torch.clamp(1.0 - local_recent_sched_ratio, 0.0, 1.0)
        marginal_need_proxy = torch.clamp(
            0.55 * demand_prg_need + 0.30 * local_debt + 0.15 * service_gap,
            0.0,
            1.0,
        )
        expected_decode_value = (1.0 - candidate_fail_risk) * marginal_need_proxy
        return candidate_fail_risk, expected_decode_value, marginal_need_proxy

    @staticmethod
    def _rank_percentile(values: torch.Tensor, *, descending: bool) -> torch.Tensor:
        count = int(values.numel())
        if count <= 1:
            return torch.ones_like(values, dtype=torch.float32)
        order = torch.argsort(values, descending=descending)
        ranks = torch.empty((count,), dtype=torch.float32, device=values.device)
        ranks[order] = torch.arange(count, dtype=torch.float32, device=values.device)
        return 1.0 - ranks / float(count - 1)

    def _relative_candidate_features(
        self,
        *,
        local_quality_db: torch.Tensor,
        local_layer_min_quality_db: torch.Tensor | None = None,
        local_layer_spread_db: torch.Tensor | None = None,
        local_debt: torch.Tensor,
        local_tb_err: torch.Tensor,
        local_backlog_bytes: torch.Tensor,
        local_recent_sched_ratio: torch.Tensor,
        prg_risk: float,
    ) -> torch.Tensor:
        count = int(local_quality_db.numel())
        if count <= 0:
            return torch.zeros((0, 6), dtype=torch.float32, device=local_quality_db.device)

        candidate_fail_risk, expected_decode_value, _ = self._candidate_risk_tensors(
            local_quality_db=local_quality_db,
            local_layer_min_quality_db=local_layer_min_quality_db,
            local_layer_spread_db=local_layer_spread_db,
            local_debt=local_debt,
            local_tb_err=local_tb_err,
            local_backlog_bytes=local_backlog_bytes,
            local_recent_sched_ratio=local_recent_sched_ratio,
            prg_risk=prg_risk,
        )
        sinr_vs_mean = torch.tanh((local_quality_db - local_quality_db.mean()) / 6.0)
        if count > 1:
            top2_quality = torch.topk(local_quality_db, k=2).values
            best_quality = top2_quality[0]
            second_quality = top2_quality[1]
            competitor_quality = torch.where(
                torch.isclose(local_quality_db, best_quality),
                second_quality.expand_as(local_quality_db),
                best_quality.expand_as(local_quality_db),
            )
            sinr_vs_nearest_best = torch.tanh((local_quality_db - competitor_quality) / 6.0)
        else:
            sinr_vs_nearest_best = torch.zeros_like(local_quality_db)
        risk_vs_mean = torch.clamp(candidate_fail_risk.mean() - candidate_fail_risk, -1.0, 1.0)
        risk_rank = self._rank_percentile(candidate_fail_risk, descending=False)
        goodput_vs_mean = torch.tanh((expected_decode_value - expected_decode_value.mean()) / 0.25)
        goodput_rank = self._rank_percentile(expected_decode_value, descending=True)
        return torch.stack(
            [
                sinr_vs_mean,
                sinr_vs_nearest_best,
                risk_vs_mean,
                risk_rank,
                goodput_vs_mean,
                goodput_rank,
            ],
            dim=-1,
        ).to(dtype=torch.float32)

    def _candidate_scores(
        self,
        *,
        local_quality_db: torch.Tensor,
        local_layer_min_quality_db: torch.Tensor | None = None,
        local_layer_spread_db: torch.Tensor | None = None,
        local_urgency: torch.Tensor,
        local_debt: torch.Tensor,
        local_tb_err: torch.Tensor,
    ) -> torch.Tensor:
        quality_norm = self._normalize_score(local_quality_db)
        layer_min_norm = self._normalize_score(
            local_quality_db if local_layer_min_quality_db is None else local_layer_min_quality_db
        )
        layer_spread_norm = self._normalize_score(
            torch.zeros_like(local_quality_db) if local_layer_spread_db is None else local_layer_spread_db
        )
        urgency_norm = self._normalize_score(local_urgency)
        debt_norm = self._normalize_score(local_debt)
        tb_err_norm = self._normalize_score(torch.clamp_min(local_tb_err, 0.0))
        return (
            0.42 * quality_norm
            + 0.20 * layer_min_norm
            + 0.22 * urgency_norm
            + 0.20 * debt_norm
            - 0.10 * tb_err_norm
            - 0.08 * layer_spread_norm
        )

    def _shared_pool_scores(self, ue_features: torch.Tensor) -> torch.Tensor:
        urgency = self._urgency_scores(ue_features)
        feasibility = torch.clamp_min(ue_features[:, 2], 0.0) - 0.25 * torch.clamp_min(ue_features[:, 5], 0.0)
        debt = self._debt_scores(ue_features)
        tb_err = torch.clamp_min(ue_features[:, 5], 0.0)
        return 0.35 * urgency + 0.35 * feasibility + 0.25 * debt - 0.10 * tb_err

    def _ue_prg_edge_features(
        self,
        *,
        quality_db: float,
        best_quality_db: float,
        urgency: float,
        debt: float,
        prg_risk: float,
        ue_tb_err: float,
        backlog_bytes: float = 0.0,
        recent_sched_ratio: float = 1.0,
        hol_delay_ms: float = 0.0,
        ttl_slack_ms: float = -1.0,
        recent_deficit: float = 0.0,
        layer_min_quality_db: float | None = None,
        layer_spread_db: float = 0.0,
        pfq_anchor: float,
        relative_features: Sequence[float] | None = None,
    ) -> List[float]:
        features = [
            quality_db,
            float(max(0.0, best_quality_db - quality_db)),
            urgency,
            debt,
            prg_risk,
        ]
        if self.cfg.include_pfq_score_anchor:
            features.append(pfq_anchor)
        if self.cfg.include_candidate_risk_features:
            features.extend(
                self._candidate_risk_features(
                    quality_db=quality_db,
                    layer_min_quality_db=layer_min_quality_db,
                    layer_spread_db=layer_spread_db,
                    urgency=urgency,
                    debt=debt,
                    prg_risk=prg_risk,
                    ue_tb_err=ue_tb_err,
                    backlog_bytes=backlog_bytes,
                    recent_sched_ratio=recent_sched_ratio,
                    hol_delay_ms=hol_delay_ms,
                    ttl_slack_ms=ttl_slack_ms,
                    recent_deficit=recent_deficit,
                    relative_features=relative_features,
                )
            )
        return features

    @staticmethod
    def _pfq_anchor_scores(quality_db: torch.Tensor, ue_features: torch.Tensor) -> torch.Tensor:
        quality_lin = torch.pow(
            torch.full_like(quality_db, 10.0),
            quality_db / 10.0,
        )
        # PFQ uses instantaneous data rate over long-term average rate, multiplied
        # by the queue-aware buffer weight. Constants mirror the native defaults.
        est_rate_mbps = 5.76 * torch.log2(1.0 + torch.clamp_min(quality_lin, 1.0e-9))
        avg_rate_mbps = torch.clamp_min(ue_features[:, 1], 0.1)
        buffer_bytes = torch.clamp_min(ue_features[:, 0], 0.0)
        queue_weight = 1.0 + 0.25 * torch.log2(1.0 + buffer_bytes / 3000.0)
        pfq_score = torch.clamp_min(est_rate_mbps / avg_rate_mbps, 0.0) * queue_weight
        return torch.log1p(pfq_score)
