from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import EdgeSet, GraphFeatureSnapshot, SparseEntityGraph


@dataclass(frozen=True)
class MaskCheckReport:
    total_prg_tokens: int
    prg_without_candidates: int
    prg_without_legal_actions: int
    illegal_cross_cell_up_edges: int
    unschedulable_up_edges: int
    unavailable_prg_with_candidates: int
    missing_cell_prg_edges: int
    cell_ue_edge_mismatch: int
    ok: bool


class GraphMaskChecker:
    def check(self, graph: SparseEntityGraph | GraphFeatureSnapshot, edges: dict[str, EdgeSet] | None = None) -> MaskCheckReport:
        if isinstance(graph, SparseEntityGraph):
            snapshot = graph.snapshot
            edge_map = graph.edges
        else:
            snapshot = graph
            if edges is None:
                raise ValueError("edges are required when checking a raw snapshot")
            edge_map = edges

        meta = snapshot.static_meta
        e_cu = edge_map["E_CU"].index
        e_cp = edge_map["E_CP"].index
        e_up = edge_map["E_UP"].index

        expected_cu = int(meta.action_mask_cell_ue.to(torch.int64).sum().item())
        actual_cu = int(e_cu.shape[1])
        cell_ue_edge_mismatch = abs(expected_cu - actual_cu)

        expected_cp = meta.n_cell * meta.n_prg
        missing_cell_prg_edges = max(0, expected_cp - int(e_cp.shape[1]))

        prg_candidate_counts = torch.zeros((meta.n_cell * meta.n_prg,), dtype=torch.long)
        illegal_cross_cell_up_edges = 0
        unschedulable_up_edges = 0
        unavailable_prg_with_candidates = 0
        for edge_idx in range(e_up.shape[1]):
            ue_idx = int(e_up[0, edge_idx].item())
            prg_token = int(e_up[1, edge_idx].item())
            owner_cell = int(meta.prg_owner_cell[prg_token].item())
            prg_candidate_counts[prg_token] += 1
            if not bool(meta.action_mask_cell_ue[owner_cell, ue_idx].item()):
                illegal_cross_cell_up_edges += 1
            if meta.action_mask_ue is not None and not bool(meta.action_mask_ue[ue_idx].item()):
                unschedulable_up_edges += 1
            prg_local_idx = int(meta.prg_local_index[prg_token].item())
            if not bool(meta.action_mask_prg_cell[owner_cell, prg_local_idx].item()):
                unavailable_prg_with_candidates += 1

        prg_without_candidates = int((prg_candidate_counts == 0).sum().item())
        if meta.blank_action_always_allowed:
            prg_without_legal_actions = 0
        else:
            prg_without_legal_actions = prg_without_candidates

        ok = (
            cell_ue_edge_mismatch == 0
            and missing_cell_prg_edges == 0
            and illegal_cross_cell_up_edges == 0
            and unschedulable_up_edges == 0
            and unavailable_prg_with_candidates == 0
            and prg_without_legal_actions == 0
        )
        return MaskCheckReport(
            total_prg_tokens=meta.n_cell * meta.n_prg,
            prg_without_candidates=prg_without_candidates,
            prg_without_legal_actions=prg_without_legal_actions,
            illegal_cross_cell_up_edges=illegal_cross_cell_up_edges,
            unschedulable_up_edges=unschedulable_up_edges,
            unavailable_prg_with_candidates=unavailable_prg_with_candidates,
            missing_cell_prg_edges=missing_cell_prg_edges,
            cell_ue_edge_mismatch=cell_ue_edge_mismatch,
            ok=ok,
        )
