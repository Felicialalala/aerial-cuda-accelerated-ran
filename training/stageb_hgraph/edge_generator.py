from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .graph_ops import demand_active_ue_mask_from_features
from .types import EdgeSet, GraphFeatureSnapshot, SparseEntityGraph


@dataclass(frozen=True)
class EdgeGeneratorConfig:
    ue_prg_topk: int = 4
    ue_prg_diversity_extra: int = 0
    candidate_mode: str = "all_demand_ues"
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
                prg_conflict = prg[c_idx, prg_idx, 4]
                prg_ici = prg[c_idx, prg_idx, 5]
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
                for ue_idx in local_pick:
                    src.append(int(ue_idx))
                    dst.append(prg_token)
                    edge_attr.append(
                        [
                            float(wb_sinr_db[ue_idx].item()),
                            float(max(0.0, best_wb_db - float(wb_sinr_db[ue_idx].item()))),
                            float(urgency_scores[ue_idx].item()),
                            float(debt_scores[ue_idx].item()),
                            float(prg_conflict + prg_ici),
                        ]
                    )
        index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else None
        return EdgeSet(name="E_UP", src_type="ue", dst_type="prg", index=index, attr=attr)

    def _build_ue_prg_edges_post_eq(self, snapshot: GraphFeatureSnapshot) -> EdgeSet:
        meta = snapshot.static_meta
        post_eq = meta.post_eq_sinr
        if post_eq is None:
            raise RuntimeError("post_eq_sinr unexpectedly missing")
        if post_eq.dim() == 3:
            post_eq = post_eq.mean(dim=-1)
        if post_eq.shape != (meta.n_ue, meta.n_prg):
            raise ValueError(
                f"post_eq_sinr must be [U,P] or [U,P,L], got shape={tuple(meta.post_eq_sinr.shape)}"
            )
        post_eq_db = 10.0 * torch.log10(torch.clamp_min(post_eq, 1.0e-9))

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
            for prg_idx in range(meta.n_prg):
                if not bool(meta.action_mask_prg_cell[c_idx, prg_idx].item()):
                    continue
                prg_token = c_idx * meta.n_prg + prg_idx
                prg_risk = float(prg[c_idx, prg_idx, 4] + prg[c_idx, prg_idx, 5])
                local_quality_db = post_eq_db[legal_ues, prg_idx]
                best_quality_db = float(local_quality_db.max().item())
                if self.cfg.candidate_mode == "all_demand_ues":
                    picked = torch.argsort(local_quality_db, descending=True)
                else:
                    picked = self._select_post_eq_candidates(
                        local_quality_db=local_quality_db,
                        local_urgency=legal_urgency,
                        local_debt=legal_debt,
                        prg_idx=prg_idx,
                    )

                for local_rank in picked.tolist():
                    ue_idx = int(legal_ues[local_rank].item())
                    src.append(ue_idx)
                    dst.append(prg_token)
                    quality_db = float(local_quality_db[local_rank].item())
                    edge_attr.append(
                        [
                            quality_db,
                            float(max(0.0, best_quality_db - quality_db)),
                            float(legal_urgency[local_rank].item()),
                            float(legal_debt[local_rank].item()),
                            prg_risk,
                        ]
                    )
        index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else None
        return EdgeSet(name="E_UP", src_type="ue", dst_type="prg", index=index, attr=attr)

    def _select_post_eq_candidates(
        self,
        *,
        local_quality_db: torch.Tensor,
        local_urgency: torch.Tensor,
        local_debt: torch.Tensor,
        prg_idx: int,
    ) -> torch.Tensor:
        candidate_count = int(local_quality_db.numel())
        topk = min(int(self.cfg.ue_prg_topk), candidate_count)
        if topk <= 0:
            return torch.zeros((0,), dtype=torch.long, device=local_quality_db.device)

        quality_rank = torch.argsort(local_quality_db, descending=True)
        core = quality_rank[:topk]

        extra_limit = max(0, int(self.cfg.ue_prg_diversity_extra))
        remaining = quality_rank[topk:]
        if extra_limit <= 0 or remaining.numel() == 0:
            return core

        extra_count = min(extra_limit, int(remaining.numel()))
        quality_norm = self._normalize_score(local_quality_db)
        urgency_norm = self._normalize_score(local_urgency)
        debt_norm = self._normalize_score(local_debt)
        diversity_score = 0.50 * quality_norm + 0.30 * urgency_norm + 0.20 * debt_norm
        remaining_by_diversity = remaining[torch.argsort(diversity_score[remaining], descending=True)]
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

    def _shared_pool_scores(self, ue_features: torch.Tensor) -> torch.Tensor:
        urgency = self._urgency_scores(ue_features)
        feasibility = torch.clamp_min(ue_features[:, 2], 0.0) - 0.25 * torch.clamp_min(ue_features[:, 5], 0.0)
        debt = self._debt_scores(ue_features)
        return 0.40 * urgency + 0.35 * feasibility + 0.25 * debt
