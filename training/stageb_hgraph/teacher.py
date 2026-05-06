from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import torch

from .graph_ops import (
    joint_targets_from_type0_actions,
    decode_prg_choices_to_type0,
    pack_prg_candidates,
)
from .types import SparseEntityGraph


@dataclass(frozen=True)
class HeuristicTeacherConfig:
    blank_bias: float = 0.10
    candidate_score_weight: float = 0.30
    local_value_weight: float = 0.25
    risk_penalty_weight: float = 0.35
    debt_weight: float = 0.15
    prev_assigned_bonus: float = 0.05
    coverage_target_per_cell: int = 4
    coverage_bonus: float = 0.30
    repeat_penalty_weight: float = 0.20
    share_cap_penalty_weight: float = 0.30
    max_candidates: int = 0


def _ue_urgency(ue: torch.Tensor) -> torch.Tensor:
    buffer_term = torch.log1p(torch.clamp_min(ue[:, 0], 0.0))
    hol_term = ue[:, 8] / torch.clamp(ue[:, 8].max(), min=1.0)
    ttl_term = torch.where(
        ue[:, 9] >= 0.0,
        1.0 / (1.0 + torch.clamp_min(ue[:, 9], 0.0)),
        torch.zeros_like(ue[:, 9]),
    )
    return 0.45 * buffer_term + 0.30 * hol_term + 0.25 * ttl_term


def _ue_feasibility(ue: torch.Tensor) -> torch.Tensor:
    sinr_term = torch.tanh(torch.clamp_min(ue[:, 2], 0.0) / 10.0)
    tb_err_term = torch.clamp(ue[:, 5], 0.0, 1.0)
    return sinr_term - 0.35 * tb_err_term


def _ue_service_debt(ue: torch.Tensor) -> torch.Tensor:
    avg_rate = torch.log1p(torch.clamp_min(ue[:, 1], 0.0))
    sched_ratio = torch.clamp(ue[:, 10], 0.0, 1.0)
    deficit = torch.clamp_min(ue[:, 11], 0.0)
    return 0.20 * avg_rate + 0.35 * (1.0 - sched_ratio) + 0.45 * deficit


def _prg_local_value(prg: torch.Tensor) -> torch.Tensor:
    top1 = torch.tanh(prg[:, 0] / 20.0)
    gap = torch.tanh(torch.clamp_min(prg[:, 1], 0.0) / 10.0)
    return 0.65 * top1 + 0.35 * gap


def _prg_conflict_risk(prg: torch.Tensor) -> torch.Tensor:
    neigh = torch.sigmoid((prg[:, 4] - prg[:, 0]) / 5.0)
    same_prg = torch.clamp(prg[:, 6], 0.0, 1.0)
    ici = torch.clamp(prg[:, 7], 0.0, 1.0)
    return 0.20 * neigh + 0.45 * same_prg + 0.35 * ici


class GraphHeuristicTeacher:
    def __init__(self, cfg: HeuristicTeacherConfig | None = None):
        self.cfg = cfg or HeuristicTeacherConfig()

    def label_graph(
        self,
        graph: SparseEntityGraph,
    ) -> Dict[str, torch.Tensor]:
        snapshot = graph.snapshot
        prg = snapshot.prg_features.values
        ue = snapshot.ue_features.values
        packed = pack_prg_candidates(
            graph,
            max_candidates=(int(self.cfg.max_candidates) if int(self.cfg.max_candidates) > 0 else None),
            fixed_width=int(self.cfg.max_candidates) > 0,
        )
        device = packed.ue_ids.device

        n_prg_tokens = snapshot.n_prg_tokens
        blank_idx = packed.blank_class_index
        target = torch.full((n_prg_tokens,), blank_idx, dtype=torch.long, device=device)
        chosen_scores = torch.zeros((n_prg_tokens,), dtype=torch.float32, device=device)

        urgency = _ue_urgency(ue)
        feasibility = _ue_feasibility(ue)
        debt = _ue_service_debt(ue)
        ue_score = urgency + feasibility + self.cfg.debt_weight * debt

        local_value = _prg_local_value(prg)
        conflict_risk = _prg_conflict_risk(prg)
        availability = graph.snapshot.static_meta.action_mask_prg_cell.reshape(-1).to(device=device, dtype=torch.bool)
        prev_assigned = torch.clamp(prg[:, 2], 0.0, 1.0)
        owner_cell = graph.snapshot.static_meta.prg_owner_cell.to(torch.long)
        n_cell = int(graph.snapshot.static_meta.n_cell)

        for cell_idx in range(n_cell):
            cell_tokens = torch.nonzero((owner_cell == cell_idx) & availability, as_tuple=False).flatten()
            if cell_tokens.numel() == 0:
                continue

            order_score = local_value[cell_tokens] - 0.5 * conflict_risk[cell_tokens]
            cell_tokens = cell_tokens[torch.argsort(order_score, descending=True)]

            unique_candidates: set[int] = set()
            for prg_token in cell_tokens.tolist():
                for local_idx in range(packed.max_candidates):
                    if not bool(packed.mask[prg_token, local_idx].item()):
                        continue
                    unique_candidates.add(int(packed.ue_ids[prg_token, local_idx].item()))

            coverage_target = min(int(self.cfg.coverage_target_per_cell), len(unique_candidates))
            soft_share_cap = max(1, math.ceil(len(cell_tokens) / max(1, coverage_target)))
            ue_usage: dict[int, int] = {}

            for prg_token in cell_tokens.tolist():
                blank_score = (
                    self.cfg.blank_bias
                    + self.cfg.risk_penalty_weight * conflict_risk[prg_token]
                    - 0.10 * local_value[prg_token]
                )
                best_idx = blank_idx
                best_score = float(blank_score.item())

                covered_ues = sum(1 for count in ue_usage.values() if count > 0)
                for local_idx in range(packed.max_candidates):
                    if not bool(packed.mask[prg_token, local_idx].item()):
                        continue
                    ue_idx = int(packed.ue_ids[prg_token, local_idx].item())
                    usage_count = int(ue_usage.get(ue_idx, 0))

                    score = float(ue_score[ue_idx].item())
                    if packed.edge_attr is not None:
                        post_eq_db = float(packed.edge_attr[prg_token, local_idx, 0].item())
                        delta_to_best = float(packed.edge_attr[prg_token, local_idx, 1].item())
                        candidate_quality = math.tanh(post_eq_db / 20.0) - 0.20 * math.tanh(delta_to_best / 10.0)
                        score += self.cfg.candidate_score_weight * candidate_quality
                    score += self.cfg.local_value_weight * float(local_value[prg_token].item())
                    score += self.cfg.prev_assigned_bonus * float(prev_assigned[prg_token].item())
                    score -= self.cfg.risk_penalty_weight * float(conflict_risk[prg_token].item())

                    if usage_count == 0 and covered_ues < coverage_target:
                        score += self.cfg.coverage_bonus * float(coverage_target - covered_ues)
                    else:
                        score -= self.cfg.repeat_penalty_weight * float(usage_count)

                    if usage_count >= soft_share_cap:
                        score -= self.cfg.share_cap_penalty_weight * float(usage_count - soft_share_cap + 1)

                    if score > best_score:
                        best_score = score
                        best_idx = local_idx

                target[prg_token] = int(best_idx)
                chosen_scores[prg_token] = float(best_score)
                if best_idx != blank_idx and packed.max_candidates > 0:
                    ue_idx = int(packed.ue_ids[prg_token, best_idx].item())
                    ue_usage[ue_idx] = int(ue_usage.get(ue_idx, 0)) + 1

        decoded = decode_prg_choices_to_type0(
            graph,
            packed,
            target,
            action_mask_ue=graph.snapshot.static_meta.action_mask_ue
            if graph.snapshot.static_meta.action_mask_ue is not None
            else torch.ones((graph.snapshot.static_meta.n_ue,), dtype=torch.bool),
            n_sched_ue=graph.snapshot.static_meta.n_sched_ue,
        )
        meta = graph.snapshot.static_meta
        target_ue_class, target_prg_class = joint_targets_from_type0_actions(
            action_ue_select=decoded.action_ue_select,
            action_prg_alloc=decoded.action_prg_alloc,
            n_ue=meta.n_ue,
            n_sched_ue=meta.n_sched_ue,
            n_cell=meta.n_cell,
            n_prg=meta.n_prg,
        )
        return {
            "target_ue_class": target_ue_class,
            "target_prg_class": target_prg_class,
            "action_ue_select": decoded.action_ue_select.to(torch.int32),
            "action_prg_alloc": decoded.action_prg_alloc.to(torch.int16),
            "teacher_scores": chosen_scores,
            "teacher_blank_ratio": (decoded.action_prg_alloc.to(torch.long) < 0).float().mean(),
        }


class PfqPassthroughTeacher:
    def __init__(self, *, max_candidates: int = 0):
        self.max_candidates = int(max_candidates)

    def label_graph(self, graph: SparseEntityGraph) -> Dict[str, torch.Tensor]:
        meta = graph.snapshot.static_meta
        if meta.teacher_action_ue is None or meta.teacher_action_prg is None:
            raise RuntimeError("PFQ passthrough teacher requested, but teacher actions are missing from the bridge state")
        target_ue_class, target_prg_class = joint_targets_from_type0_actions(
            action_ue_select=meta.teacher_action_ue,
            action_prg_alloc=meta.teacher_action_prg,
            n_ue=meta.n_ue,
            n_sched_ue=meta.n_sched_ue,
            n_cell=meta.n_cell,
            n_prg=meta.n_prg,
        )
        blank_ratio = (meta.teacher_action_prg.to(torch.long) < 0).float().mean()
        return {
            "target_ue_class": target_ue_class,
            "target_prg_class": target_prg_class,
            "action_ue_select": meta.teacher_action_ue.to(torch.int32),
            "action_prg_alloc": meta.teacher_action_prg.to(torch.int16),
            "teacher_blank_ratio": blank_ratio,
        }


def build_joint_teacher(
    mode: str,
    *,
    heuristic_cfg: HeuristicTeacherConfig | None = None,
    max_candidates: int = 0,
):
    normalized = mode.strip().lower()
    if normalized == "heuristic_graph":
        return GraphHeuristicTeacher(heuristic_cfg or HeuristicTeacherConfig(max_candidates=int(max_candidates)))
    if normalized == "pfq_passthrough":
        return PfqPassthroughTeacher(max_candidates=int(max_candidates))
    raise ValueError(f"unsupported teacher mode: {mode}")
