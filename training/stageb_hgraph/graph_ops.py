from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from training.gnnrl.action_modes import build_type0_all_ue_action
from training.gnnrl.masks import build_slot_selection_mask
from training.gnnrl.slot_layout import build_type0_slot_layout

from .types import SparseEntityGraph

_UE_BUFFER_IDX = 0
_UE_AVG_RATE_IDX = 1
_UE_WB_SINR_IDX = 2
_UE_TB_ERR_LAST_IDX = 5
_UE_NEW_DATA_IDX = 6
_UE_STALE_SLOTS_IDX = 7
_UE_HOL_DELAY_IDX = 8
_UE_TTL_SLACK_IDX = 9
_UE_RECENT_SCHED_RATIO_IDX = 10
_UE_RECENT_DEFICIT_IDX = 11
_UE_PACKET_RATE_GAP_IDX = 12
_UE_PACKET_DELAY_NORM_IDX = 13
_UE_PACKET_NO_DELIVERY_IDX = 14


@dataclass(frozen=True)
class PackedPrgCandidates:
    ue_ids: torch.Tensor
    mask: torch.Tensor
    counts: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.ue_ids.dim() != 2:
            raise ValueError(f"ue_ids must be [P,K], got shape={tuple(self.ue_ids.shape)}")
        if self.mask.shape != self.ue_ids.shape:
            raise ValueError(
                f"candidate mask mismatch: ue_ids={tuple(self.ue_ids.shape)} mask={tuple(self.mask.shape)}"
            )
        if self.counts.shape != (self.ue_ids.shape[0],):
            raise ValueError(
                f"candidate counts mismatch: got={tuple(self.counts.shape)} expected={(self.ue_ids.shape[0],)}"
            )
        if self.edge_attr is not None and self.edge_attr.shape[:2] != self.ue_ids.shape:
            raise ValueError(
                "edge_attr leading dims mismatch: "
                f"edge_attr={tuple(self.edge_attr.shape)} ue_ids={tuple(self.ue_ids.shape)}"
            )

    @property
    def num_prg_tokens(self) -> int:
        return int(self.ue_ids.shape[0])

    @property
    def max_candidates(self) -> int:
        return int(self.ue_ids.shape[1])

    @property
    def blank_class_index(self) -> int:
        return self.max_candidates


@dataclass(frozen=True)
class DecodedType0Action:
    action_ue_select: torch.Tensor
    action_prg_alloc: torch.Tensor
    chosen_ue_per_prg: torch.Tensor
    prg_chosen_mask: torch.Tensor
    cap_drop_count: int = 0
    fallback_success_count: int = 0
    safe_fill_count: int = 0
    final_blank_count: int = 0
    post_decode_policy_blank_count: int = 0
    post_decode_no_valid_candidate_count: int = 0
    post_decode_demand_cap_count: int = 0
    post_decode_no_slot_count: int = 0
    post_decode_other_drop_count: int = 0
    missed_safe_opportunity_count: int = 0


_TYPE0_PREFLIGHT_KEYS = (
    "wrong_cell_slot",
    "empty_slot",
    "prg_mask",
    "oob_slot",
    "duplicate_ue",
)


def type0_action_preflight_reject_counts(
    graph: SparseEntityGraph,
    action_ue_select: torch.Tensor,
    action_prg_alloc: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Predict Type-0 native reject counters before sending an action to C++."""
    meta = graph.snapshot.static_meta
    device = graph.snapshot.ue_features.values.device
    ue_action = torch.as_tensor(action_ue_select, dtype=torch.long, device=device).reshape(-1)
    prg_action = torch.as_tensor(action_prg_alloc, dtype=torch.long, device=device).reshape(-1)
    if ue_action.shape != (int(meta.n_sched_ue),):
        raise ValueError(
            f"action_ue_select mismatch: got={tuple(ue_action.shape)} expected={(int(meta.n_sched_ue),)}"
        )
    if prg_action.shape != (int(meta.n_cell) * int(meta.n_prg),):
        raise ValueError(
            "action_prg_alloc mismatch: "
            f"got={tuple(prg_action.shape)} expected={(int(meta.n_cell) * int(meta.n_prg),)}"
        )

    action_mask_ue = _default_action_mask_ue(graph, device)
    action_mask_cell_ue = meta.action_mask_cell_ue.to(device=device, dtype=torch.bool)
    action_mask_prg_cell = meta.action_mask_prg_cell.to(device=device, dtype=torch.bool)
    layout = build_type0_slot_layout(
        action_mask_cell_ue.unsqueeze(0),
        n_sched_ue=int(meta.n_sched_ue),
        action_mask_ue=action_mask_ue.unsqueeze(0),
    )
    slot_to_cell = layout.slot_to_cell.squeeze(0).to(device=device, dtype=torch.long)
    slot_valid_mask = layout.slot_valid_mask.squeeze(0).to(device=device, dtype=torch.bool)

    counts = {
        key: torch.zeros((int(meta.n_cell),), dtype=torch.long, device=device) for key in _TYPE0_PREFLIGHT_KEYS
    }
    accepted_slot = torch.zeros((int(meta.n_sched_ue),), dtype=torch.bool, device=device)
    used_ue = torch.zeros((int(meta.n_ue),), dtype=torch.bool, device=device)
    for slot_idx in range(min(int(meta.n_sched_ue), int(ue_action.shape[0]))):
        if not bool(slot_valid_mask[slot_idx].item()):
            continue
        ue_idx = int(ue_action[slot_idx].item())
        if ue_idx < 0 or ue_idx >= int(meta.n_ue):
            continue
        if not bool(action_mask_ue[ue_idx].item()):
            continue
        cell_idx = int(slot_to_cell[slot_idx].item())
        if not bool(action_mask_cell_ue[cell_idx, ue_idx].item()):
            continue
        if bool(used_ue[ue_idx].item()):
            counts["duplicate_ue"][cell_idx] += 1
            continue
        used_ue[ue_idx] = True
        accepted_slot[slot_idx] = True

    for linear_idx in range(min(int(meta.n_cell) * int(meta.n_prg), int(prg_action.shape[0]))):
        slot_idx = int(prg_action[linear_idx].item())
        if slot_idx < 0:
            continue
        cell_idx = linear_idx % int(meta.n_cell)
        prg_idx = linear_idx // int(meta.n_cell)
        if not bool(action_mask_prg_cell[cell_idx, prg_idx].item()):
            counts["prg_mask"][cell_idx] += 1
            continue
        if slot_idx >= int(meta.n_sched_ue):
            counts["oob_slot"][cell_idx] += 1
            continue
        if not bool(slot_valid_mask[slot_idx].item()) or int(slot_to_cell[slot_idx].item()) != cell_idx:
            counts["wrong_cell_slot"][cell_idx] += 1
            continue
        if not bool(accepted_slot[slot_idx].item()):
            counts["empty_slot"][cell_idx] += 1
            continue

    return counts


def _finalize_decoded_type0_action(
    graph: SparseEntityGraph,
    action_ue_select: torch.Tensor,
    action_prg_alloc: torch.Tensor,
    *,
    cap_drop_count: int = 0,
    fallback_success_count: int = 0,
    safe_fill_count: int = 0,
    post_decode_policy_blank_count: int = 0,
    post_decode_no_valid_candidate_count: int = 0,
    post_decode_demand_cap_count: int = 0,
    post_decode_no_slot_count: int = 0,
    post_decode_other_drop_count: int = 0,
    missed_safe_opportunity_count: int = 0,
) -> DecodedType0Action:
    """Validate decoded Type-0 PRG allocations in native allocSol order."""
    meta = graph.snapshot.static_meta
    device = action_prg_alloc.device
    action_ue_select = torch.as_tensor(action_ue_select, dtype=torch.int32, device=device).reshape(
        int(meta.n_sched_ue)
    )
    alloc = torch.as_tensor(action_prg_alloc, dtype=torch.int16, device=device).reshape(
        int(meta.n_cell) * int(meta.n_prg)
    )

    action_mask_ue = _default_action_mask_ue(graph, device)
    layout = build_type0_slot_layout(
        meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).unsqueeze(0),
        n_sched_ue=int(meta.n_sched_ue),
        action_mask_ue=action_mask_ue.unsqueeze(0),
    )
    slot_to_cell = layout.slot_to_cell.squeeze(0).to(device=device, dtype=torch.long)
    slot_valid_mask = layout.slot_valid_mask.squeeze(0).to(device=device, dtype=torch.bool)

    owner_cell = meta.prg_owner_cell.to(device=device, dtype=torch.long)
    local_prg = meta.prg_local_index.to(device=device, dtype=torch.long)
    native_linear = local_prg * int(meta.n_cell) + owner_cell
    slot = alloc.to(dtype=torch.long).gather(0, native_linear)

    slot_in_range = (slot >= 0) & (slot < int(meta.n_sched_ue))
    safe_slot = slot.clamp(min=0, max=max(0, int(meta.n_sched_ue) - 1))
    if slot_valid_mask.numel() > 0:
        slot_native_ok = slot_in_range & slot_valid_mask.gather(0, safe_slot)
        slot_native_ok = slot_native_ok & (slot_to_cell.gather(0, safe_slot) == owner_cell)
    else:
        slot_native_ok = torch.zeros_like(slot_in_range)

    chosen_ue = action_ue_select.to(dtype=torch.long).gather(0, safe_slot)
    ue_ok = (chosen_ue >= 0) & (chosen_ue < int(meta.n_ue))
    prg_chosen_mask = slot_native_ok & ue_ok

    assigned_prg = slot >= 0
    if bool((~prg_chosen_mask & assigned_prg).any().item()):
        alloc = alloc.clone()
        alloc[native_linear[~prg_chosen_mask & assigned_prg]] = -1

    chosen_ue_per_prg = torch.full(
        (int(meta.n_cell) * int(meta.n_prg),),
        -1,
        dtype=torch.long,
        device=device,
    )
    chosen_ue_per_prg[prg_chosen_mask] = chosen_ue[prg_chosen_mask]

    return DecodedType0Action(
        action_ue_select=action_ue_select,
        action_prg_alloc=alloc.to(dtype=torch.int16),
        chosen_ue_per_prg=chosen_ue_per_prg,
        prg_chosen_mask=prg_chosen_mask,
        cap_drop_count=int(cap_drop_count),
        fallback_success_count=int(fallback_success_count),
        safe_fill_count=int(safe_fill_count),
        final_blank_count=int((~prg_chosen_mask).sum().item()),
        post_decode_policy_blank_count=int(post_decode_policy_blank_count),
        post_decode_no_valid_candidate_count=int(post_decode_no_valid_candidate_count),
        post_decode_demand_cap_count=int(post_decode_demand_cap_count),
        post_decode_no_slot_count=int(post_decode_no_slot_count),
        post_decode_other_drop_count=int(post_decode_other_drop_count),
        missed_safe_opportunity_count=int(missed_safe_opportunity_count),
    )


def _default_action_mask_ue(graph: SparseEntityGraph, device: torch.device) -> torch.Tensor:
    meta = graph.snapshot.static_meta
    if meta.action_mask_ue is None:
        return torch.ones((meta.n_ue,), dtype=torch.bool, device=device)
    return meta.action_mask_ue.to(device=device, dtype=torch.bool)


def demand_active_ue_mask_from_features(
    ue_features: torch.Tensor,
    *,
    action_mask_ue: torch.Tensor | None = None,
    buffer_threshold_bytes: float = 1.0,
    new_data_threshold: float = 0.5,
) -> torch.Tensor:
    if ue_features.dim() != 2:
        raise ValueError(f"ue_features must be [U,F], got shape={tuple(ue_features.shape)}")
    if ue_features.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.bool, device=ue_features.device)

    buffer_bytes = torch.clamp_min(ue_features[:, _UE_BUFFER_IDX], 0.0)
    new_data_flag = (
        ue_features[:, _UE_NEW_DATA_IDX]
        if ue_features.shape[1] > _UE_NEW_DATA_IDX
        else torch.zeros_like(buffer_bytes)
    )
    demand_mask = (buffer_bytes >= float(buffer_threshold_bytes)) | (new_data_flag > float(new_data_threshold))
    if action_mask_ue is not None:
        demand_mask = demand_mask & action_mask_ue.to(device=ue_features.device, dtype=torch.bool)
    return demand_mask


def _estimate_ue_bytes_per_prg(
    graph: SparseEntityGraph,
    *,
    base_bytes_per_prg: float = 3000.0,
    min_bytes_per_prg: float = 1200.0,
    max_bytes_per_prg: float = 12000.0,
) -> torch.Tensor:
    meta = graph.snapshot.static_meta
    ue = graph.snapshot.ue_features.values
    if ue.numel() == 0:
        return torch.zeros((meta.n_ue,), dtype=torch.float32, device=ue.device)

    if meta.post_eq_sinr is not None:
        post_eq = meta.post_eq_sinr
        if post_eq.dim() == 3:
            post_eq = post_eq.mean(dim=-1)
        quality_lin = torch.clamp_min(post_eq, 1.0e-9).amax(dim=1)
    else:
        quality_lin = torch.clamp_min(ue[:, _UE_WB_SINR_IDX], 1.0e-9)

    spectral_eff = torch.log2(1.0 + quality_lin)
    scale = torch.clamp(spectral_eff / 2.0, min=0.5, max=4.0)
    est = float(base_bytes_per_prg) * scale
    return torch.clamp(est, min=float(min_bytes_per_prg), max=float(max_bytes_per_prg))


def _estimate_ue_prg_demand_cap(
    graph: SparseEntityGraph,
    *,
    bytes_per_prg_floor: float = 3000.0,
    slack_bytes: float = 3000.0,
) -> torch.Tensor:
    """
    Estimate a first-version per-UE PRG upper bound from current MAC backlog.

    This mirrors the intent of PFQ's demand-aware cap: once a UE has already
    received enough PRGs to plausibly cover its current demand (plus a small
    slack), later PRGs should be allowed to fall through to other UEs instead of
    continuing to accumulate on the same UE.
    """
    meta = graph.snapshot.static_meta
    ue = graph.snapshot.ue_features.values
    if ue.numel() == 0:
        return torch.zeros((meta.n_ue,), dtype=torch.long, device=ue.device)

    safe_bytes_per_prg = max(float(bytes_per_prg_floor), 1.0)
    safe_slack_bytes = max(float(slack_bytes), 0.0)
    buffer_bytes = torch.clamp_min(ue[:, _UE_BUFFER_IDX], 0.0)
    new_data_flag = (
        ue[:, _UE_NEW_DATA_IDX]
        if ue.shape[1] > _UE_NEW_DATA_IDX
        else torch.zeros_like(buffer_bytes)
    )
    demand_mask = demand_active_ue_mask_from_features(
        ue,
        action_mask_ue=meta.action_mask_ue,
    )
    est_bytes_per_prg = _estimate_ue_bytes_per_prg(graph, base_bytes_per_prg=safe_bytes_per_prg)
    demand_bytes = buffer_bytes + safe_slack_bytes
    demand_bytes = torch.where((new_data_flag > 0.5) & (demand_bytes <= 0.0), est_bytes_per_prg, demand_bytes)
    cap = torch.ceil(demand_bytes / est_bytes_per_prg.clamp_min(1.0)).to(torch.long)
    cap = torch.where((new_data_flag > 0.5) & (cap <= 0), torch.ones_like(cap), cap)
    cap = torch.where(demand_mask, cap, torch.zeros_like(cap))

    return torch.clamp(cap, min=0, max=int(meta.n_prg))


def _candidate_packet_completion_scores(
    graph: SparseEntityGraph,
    ue_ids: torch.Tensor,
    used_prg: torch.Tensor,
    *,
    packet_size_bytes: float = 3000.0,
    progress_weight: float = 0.25,
    max_score: float = 1.5,
) -> torch.Tensor:
    """Score the next PRG by whether it progresses/completes packet-sized demand."""
    meta = graph.snapshot.static_meta
    ue = graph.snapshot.ue_features.values
    device = ue_ids.device
    dtype = used_prg.dtype if used_prg.is_floating_point() else torch.float32
    if ue.numel() == 0 or ue_ids.numel() == 0:
        return torch.zeros_like(used_prg, dtype=dtype, device=device)

    safe_packet_bytes = max(float(packet_size_bytes), 1.0)
    buffer_bytes = torch.clamp_min(ue[:, _UE_BUFFER_IDX].to(device=device, dtype=dtype), 0.0)
    new_data_flag = (
        ue[:, _UE_NEW_DATA_IDX].to(device=device, dtype=dtype)
        if ue.shape[1] > _UE_NEW_DATA_IDX
        else torch.zeros_like(buffer_bytes)
    )
    demand_bytes = torch.where(
        (new_data_flag > 0.5) & (buffer_bytes <= 0.0),
        torch.full_like(buffer_bytes, safe_packet_bytes),
        buffer_bytes,
    )
    bytes_per_prg = _estimate_ue_bytes_per_prg(graph).to(device=device, dtype=dtype).clamp_min(1.0)
    safe_ue = ue_ids.to(device=device, dtype=torch.long).clamp(min=0, max=max(0, int(meta.n_ue) - 1))
    cand_demand = demand_bytes[safe_ue]
    cand_bpp = bytes_per_prg[safe_ue]
    used = used_prg.to(device=device, dtype=dtype).clamp_min(0.0)
    before = torch.minimum(used * cand_bpp, cand_demand)
    after = torch.minimum((used + 1.0) * cand_bpp, cand_demand)
    marginal_bytes = torch.clamp_min(after - before, 0.0)
    packets_before = torch.floor(before / safe_packet_bytes)
    packets_after = torch.floor(after / safe_packet_bytes)
    completion_gain = torch.clamp(packets_after - packets_before, min=0.0, max=1.0)
    progress_gain = torch.clamp(marginal_bytes / safe_packet_bytes, min=0.0, max=1.0)
    score = completion_gain + max(0.0, float(progress_weight)) * progress_gain
    return torch.clamp(score, min=0.0, max=max(0.0, float(max_score)))


def _pick_joint_fallback_slot(
    *,
    owner_cell: int,
    action_ue_select: torch.Tensor,
    cell_slot_start: torch.Tensor,
    slot_counts: torch.Tensor,
    slot_valid_mask: torch.Tensor,
    ue_prg_assigned: torch.Tensor,
    ue_prg_demand_cap: torch.Tensor,
    demand_active_mask: torch.Tensor,
    ue_backlog_bytes: torch.Tensor,
    exclude_slot: int = -1,
) -> int:
    start = int(cell_slot_start[owner_cell].item())
    count = int(slot_counts[owner_cell].item())
    best_slot = -1
    best_priority = (-1, -1, -1, -1, -1.0)
    for rel_idx in range(count):
        slot_id = start + rel_idx
        if slot_id == exclude_slot:
            continue
        if slot_id < 0 or slot_id >= int(action_ue_select.shape[0]):
            continue
        if not bool(slot_valid_mask[slot_id].item()):
            continue
        ue_idx = int(action_ue_select[slot_id].item())
        if ue_idx < 0 or ue_idx >= int(ue_prg_demand_cap.shape[0]):
            continue
        headroom = int(ue_prg_demand_cap[ue_idx].item()) - int(ue_prg_assigned[ue_idx].item())
        if headroom <= 0:
            continue
        demand_active = bool(demand_active_mask[ue_idx].item())
        backlog_bytes = float(ue_backlog_bytes[ue_idx].item())
        underserved = int(ue_prg_assigned[ue_idx].item()) <= 0
        priority = (
            1 if (underserved and demand_active and backlog_bytes > 0.0) else 0,
            1 if (underserved and demand_active) else 0,
            1 if (demand_active and backlog_bytes > 0.0) else 0,
            headroom,
            backlog_bytes,
        )
        if priority > best_priority:
            best_priority = priority
            best_slot = slot_id
    return best_slot


def _stable_group_rank(group_ids: torch.Tensor) -> torch.Tensor:
    if group_ids.dim() != 1:
        raise ValueError(f"group_ids must be rank-1, got shape={tuple(group_ids.shape)}")
    if group_ids.numel() == 0:
        return torch.zeros_like(group_ids)
    order = torch.argsort(group_ids, stable=True)
    sorted_ids = group_ids[order]
    _, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    starts = torch.repeat_interleave(counts.cumsum(dim=0) - counts, counts)
    sorted_rank = torch.arange(sorted_ids.shape[0], device=group_ids.device, dtype=torch.long) - starts
    rank = torch.empty_like(sorted_rank)
    rank[order] = sorted_rank
    return rank


def _stable_group_rank_by_score(group_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    if group_ids.dim() != 1 or scores.dim() != 1:
        raise ValueError(
            f"group_ids/scores must be rank-1, got {tuple(group_ids.shape)} and {tuple(scores.shape)}"
        )
    if group_ids.shape != scores.shape:
        raise ValueError(f"group_ids/scores shape mismatch: {tuple(group_ids.shape)} vs {tuple(scores.shape)}")
    if group_ids.numel() == 0:
        return torch.zeros_like(group_ids)

    rank = torch.empty_like(group_ids)
    for group_id in torch.unique(group_ids, sorted=True).tolist():
        idx = torch.nonzero(group_ids == int(group_id), as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        order = torch.argsort(scores[idx], descending=True, stable=True)
        local_rank = torch.empty((idx.numel(),), dtype=torch.long, device=group_ids.device)
        local_rank[order] = torch.arange(idx.numel(), dtype=torch.long, device=group_ids.device)
        rank[idx] = local_rank
    return rank


def _candidate_priority_order(choice_idx: torch.Tensor, max_candidates: int) -> torch.Tensor:
    if max_candidates <= 0:
        return torch.zeros((choice_idx.shape[0], 0), dtype=torch.long, device=choice_idx.device)
    base = torch.arange(max_candidates, device=choice_idx.device, dtype=torch.long).unsqueeze(0)
    priority = (base + 1).expand(choice_idx.shape[0], -1).clone()
    valid_primary = (choice_idx >= 0) & (choice_idx < max_candidates)
    if bool(valid_primary.any().item()):
        chosen = choice_idx.clamp(min=0, max=max_candidates - 1).unsqueeze(-1)
        priority = torch.where(valid_primary.unsqueeze(-1) & (base == chosen), torch.zeros_like(priority), priority)
    return torch.argsort(priority, dim=-1, stable=True)


def pack_prg_candidates_from_edge_index(
    edge_index: torch.Tensor,
    *,
    n_prg_tokens: int,
    edge_attr: Optional[torch.Tensor] = None,
    max_candidates: int | None = None,
    fixed_width: bool = False,
) -> PackedPrgCandidates:
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be [2,E], got shape={tuple(edge_index.shape)}")
    device = edge_index.device
    edge_count = int(edge_index.shape[1])
    counts = torch.zeros((n_prg_tokens,), dtype=torch.long, device=device)
    if edge_count > 0:
        dst = edge_index[1].to(device=device, dtype=torch.long)
        counts.scatter_add_(0, dst, torch.ones_like(dst))

    max_observed = int(counts.max().item()) if counts.numel() > 0 else 0
    max_k = max_observed
    if max_candidates is not None:
        limit = max(0, int(max_candidates))
        max_k = limit if fixed_width else min(max_k, limit)

    ue_ids = torch.full((n_prg_tokens, max_k), -1, dtype=torch.long, device=device)
    mask = torch.zeros((n_prg_tokens, max_k), dtype=torch.bool, device=device)
    out_edge_attr = None
    if edge_attr is not None:
        out_edge_attr = torch.zeros(
            (n_prg_tokens, max_k, edge_attr.shape[1]),
            dtype=edge_attr.dtype,
            device=edge_attr.device,
        )

    if edge_count > 0 and max_k > 0:
        src = edge_index[0].to(device=device, dtype=torch.long)
        dst = edge_index[1].to(device=device, dtype=torch.long)
        order = torch.argsort(dst, stable=True)
        sorted_src = src.index_select(0, order)
        sorted_dst = dst.index_select(0, order)
        _, run_counts = torch.unique_consecutive(sorted_dst, return_counts=True)
        starts = torch.repeat_interleave(run_counts.cumsum(dim=0) - run_counts, run_counts)
        local_idx = torch.arange(edge_count, device=device, dtype=torch.long) - starts
        keep = local_idx < max_k
        if bool(keep.any().item()):
            linear = sorted_dst[keep] * max_k + local_idx[keep]
            ue_ids.view(-1)[linear] = sorted_src[keep]
            mask.view(-1)[linear] = True
            if out_edge_attr is not None:
                kept_attr = edge_attr.index_select(0, order[keep])
                out_edge_attr.view(n_prg_tokens * max_k, -1)[linear] = kept_attr

    return PackedPrgCandidates(
        ue_ids=ue_ids,
        mask=mask,
        counts=torch.clamp(counts, max=max_k),
        edge_attr=out_edge_attr,
    )


def pack_prg_candidates(
    graph: SparseEntityGraph,
    *,
    max_candidates: int | None = None,
    fixed_width: bool = False,
) -> PackedPrgCandidates:
    e_up = graph.edges["E_UP"]
    return pack_prg_candidates_from_edge_index(
        e_up.index,
        n_prg_tokens=graph.snapshot.n_prg_tokens,
        edge_attr=e_up.attr,
        max_candidates=max_candidates,
        fixed_width=fixed_width,
    )


def prg_choice_to_ue_ids(choice_idx: torch.Tensor, packed: PackedPrgCandidates) -> torch.Tensor:
    if choice_idx.dim() != 1 or choice_idx.shape[0] != packed.num_prg_tokens:
        raise ValueError(
            f"choice_idx must be [P], got shape={tuple(choice_idx.shape)} for P={packed.num_prg_tokens}"
        )
    out = torch.full((packed.num_prg_tokens,), -1, dtype=torch.long, device=choice_idx.device)
    if packed.max_candidates <= 0:
        return out
    valid = (choice_idx >= 0) & (choice_idx < packed.max_candidates)
    if not bool(valid.any().item()):
        return out
    probe = choice_idx.clamp(min=0, max=packed.max_candidates - 1)
    gathered_mask = packed.mask.to(choice_idx.device).gather(1, probe.unsqueeze(-1)).squeeze(-1)
    valid = valid & gathered_mask
    selected = packed.ue_ids.to(choice_idx.device).gather(1, probe.unsqueeze(-1)).squeeze(-1)
    out[valid] = selected[valid]
    return out


def candidate_prg_tail_bias_scores(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    *,
    service_target: float = 0.75,
    ttl_guard_ms: float = 20.0,
    max_score: float = 1.5,
) -> torch.Tensor:
    """Return soft per-candidate tail-risk scores used as optional logit bias."""
    device = packed.ue_ids.device
    valid = candidate_valid.to(device=device, dtype=torch.bool)
    valid_main = valid[:, : packed.max_candidates] if packed.max_candidates > 0 else valid[:, :0]
    if packed.max_candidates <= 0:
        return torch.empty((packed.num_prg_tokens, 0), dtype=torch.float32, device=device)

    ue = graph.snapshot.ue_features.values.to(device=device, dtype=torch.float32)
    safe_ue_ids = packed.ue_ids.to(device=device, dtype=torch.long).clamp(
        min=0,
        max=max(0, int(graph.snapshot.static_meta.n_ue) - 1),
    )
    flat_ue = safe_ue_ids.reshape(-1)
    backlog = torch.clamp_min(ue[:, _UE_BUFFER_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    hol_delay = torch.clamp_min(ue[:, _UE_HOL_DELAY_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    ttl_slack = ue[:, _UE_TTL_SLACK_IDX].gather(0, flat_ue).reshape_as(safe_ue_ids)
    recent_sched = torch.clamp(ue[:, _UE_RECENT_SCHED_RATIO_IDX], 0.0, 1.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    recent_deficit = torch.clamp_min(ue[:, _UE_RECENT_DEFICIT_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    tb_err_last = torch.clamp_min(ue[:, _UE_TB_ERR_LAST_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    packet_rate_gap = (
        torch.clamp(ue[:, _UE_PACKET_RATE_GAP_IDX], 0.0, 2.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
        if ue.shape[1] > _UE_PACKET_RATE_GAP_IDX
        else torch.zeros_like(backlog)
    )
    packet_delay_norm = (
        torch.clamp(ue[:, _UE_PACKET_DELAY_NORM_IDX], 0.0, 2.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
        if ue.shape[1] > _UE_PACKET_DELAY_NORM_IDX
        else torch.zeros_like(backlog)
    )
    packet_no_delivery = (
        torch.clamp(ue[:, _UE_PACKET_NO_DELIVERY_IDX], 0.0, 1.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
        if ue.shape[1] > _UE_PACKET_NO_DELIVERY_IDX
        else torch.zeros_like(backlog)
    )

    if packed.edge_attr is None:
        edge_attr = torch.zeros((packed.num_prg_tokens, packed.max_candidates, 5), dtype=torch.float32, device=device)
    else:
        edge_attr = packed.edge_attr.to(device=device, dtype=torch.float32)
        if edge_attr.shape[-1] < 5:
            pad = torch.zeros((*edge_attr.shape[:2], 5 - int(edge_attr.shape[-1])), dtype=edge_attr.dtype, device=device)
            edge_attr = torch.cat([edge_attr, pad], dim=-1)

    quality_gap = torch.clamp_min(edge_attr[..., 1], 0.0)
    debt_edge = torch.clamp_min(edge_attr[..., 3], 0.0)
    prg_risk = torch.clamp(edge_attr[..., 4], 0.0, 1.0)

    target = max(0.05, float(service_target))
    ttl_guard = max(1.0, float(ttl_guard_ms))
    backlog_score = torch.clamp(torch.log1p(backlog / 3000.0) / 6.0, 0.0, 1.5)
    hol_score = torch.clamp(hol_delay / 200.0, 0.0, 1.5)
    ttl_score = torch.where(
        ttl_slack >= 0.0,
        torch.clamp((ttl_guard - ttl_slack) / ttl_guard, 0.0, 1.5),
        torch.zeros_like(ttl_slack),
    )
    service_gap = torch.clamp((target - recent_sched) / target, 0.0, 1.5)
    quality_gap_penalty = torch.clamp(quality_gap / 12.0, 0.0, 1.0)

    score = (
        0.18 * backlog_score
        + 0.16 * ttl_score
        + 0.14 * service_gap
        + 0.12 * torch.clamp(recent_deficit, 0.0, 1.5)
        + 0.09 * torch.clamp(debt_edge, 0.0, 1.5)
        + 0.06 * hol_score
        + 0.25 * packet_rate_gap
        + 0.12 * packet_delay_norm
        + 0.08 * packet_no_delivery
        - 0.10 * prg_risk
        - 0.05 * quality_gap_penalty
        - 0.05 * tb_err_last
    )
    score = torch.clamp(score, min=0.0, max=max(0.0, float(max_score)))
    return torch.where(valid_main, score, torch.zeros_like(score))


def candidate_prg_rescue_priority_scores(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    *,
    success_target: Optional[torch.Tensor] = None,
    service_target: float = 0.80,
    ttl_guard_ms: float = 60.0,
    max_score: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-candidate soft rescue priorities for near-drop/starved UEs.

    Unlike the broader tail score, this target is deliberately UE-facing:
    it highlights candidates whose UE has near-expiry backlog or poor recent
    goodput service, while still down-weighting clearly risky PRGs.
    """
    device = packed.ue_ids.device
    valid = candidate_valid.to(device=device, dtype=torch.bool)
    valid_main = valid[:, : packed.max_candidates] if packed.max_candidates > 0 else valid[:, :0]
    if packed.max_candidates <= 0:
        empty = torch.empty((packed.num_prg_tokens, 0), dtype=torch.float32, device=device)
        return empty, valid_main

    ue = graph.snapshot.ue_features.values.to(device=device, dtype=torch.float32)
    safe_ue_ids = packed.ue_ids.to(device=device, dtype=torch.long).clamp(
        min=0,
        max=max(0, int(graph.snapshot.static_meta.n_ue) - 1),
    )
    flat_ue = safe_ue_ids.reshape(-1)
    backlog = torch.clamp_min(ue[:, _UE_BUFFER_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    hol_delay = torch.clamp_min(ue[:, _UE_HOL_DELAY_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    ttl_slack = ue[:, _UE_TTL_SLACK_IDX].gather(0, flat_ue).reshape_as(safe_ue_ids)
    recent_sched = torch.clamp(ue[:, _UE_RECENT_SCHED_RATIO_IDX], 0.0, 1.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    recent_deficit = torch.clamp_min(ue[:, _UE_RECENT_DEFICIT_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    stale_slots = torch.clamp_min(ue[:, _UE_STALE_SLOTS_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    tb_err_last = torch.clamp_min(ue[:, _UE_TB_ERR_LAST_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    packet_rate_gap = (
        torch.clamp(ue[:, _UE_PACKET_RATE_GAP_IDX], 0.0, 2.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
        if ue.shape[1] > _UE_PACKET_RATE_GAP_IDX
        else torch.zeros_like(backlog)
    )
    packet_delay_norm = (
        torch.clamp(ue[:, _UE_PACKET_DELAY_NORM_IDX], 0.0, 2.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
        if ue.shape[1] > _UE_PACKET_DELAY_NORM_IDX
        else torch.zeros_like(backlog)
    )
    packet_no_delivery = (
        torch.clamp(ue[:, _UE_PACKET_NO_DELIVERY_IDX], 0.0, 1.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
        if ue.shape[1] > _UE_PACKET_NO_DELIVERY_IDX
        else torch.zeros_like(backlog)
    )

    if packed.edge_attr is None:
        edge_attr = torch.zeros((packed.num_prg_tokens, packed.max_candidates, 5), dtype=torch.float32, device=device)
    else:
        edge_attr = packed.edge_attr.to(device=device, dtype=torch.float32)
        if edge_attr.shape[-1] < 5:
            pad = torch.zeros((*edge_attr.shape[:2], 5 - int(edge_attr.shape[-1])), dtype=edge_attr.dtype, device=device)
            edge_attr = torch.cat([edge_attr, pad], dim=-1)

    quality_gap = torch.clamp_min(edge_attr[..., 1], 0.0)
    prg_risk = torch.clamp(edge_attr[..., 4], 0.0, 1.0)

    target = max(0.05, float(service_target))
    ttl_guard = max(1.0, float(ttl_guard_ms))
    critical_guard = max(1.0, 0.5 * ttl_guard)
    ttl_score = torch.where(
        ttl_slack >= 0.0,
        torch.clamp((ttl_guard - ttl_slack) / ttl_guard, 0.0, 2.0),
        torch.zeros_like(ttl_slack),
    )
    critical_score = torch.where(
        ttl_slack >= 0.0,
        torch.clamp((critical_guard - ttl_slack) / critical_guard, 0.0, 2.0),
        torch.zeros_like(ttl_slack),
    )
    service_gap = torch.clamp((target - recent_sched) / target, 0.0, 1.5)
    backlog_score = torch.clamp(torch.log1p(backlog / 3000.0) / 6.0, 0.0, 1.5)
    hol_score = torch.clamp(hol_delay / 200.0, 0.0, 1.5)
    stale_score = torch.clamp(stale_slots / 10.0, 0.0, 1.5)
    rescue = (
        0.24 * ttl_score
        + 0.12 * critical_score
        + 0.13 * torch.clamp(recent_deficit, 0.0, 1.5)
        + 0.10 * service_gap
        + 0.07 * backlog_score
        + 0.04 * hol_score
        + 0.03 * stale_score
        + 0.20 * packet_rate_gap
        + 0.11 * packet_delay_norm
        + 0.08 * packet_no_delivery
    )

    if success_target is not None:
        success = success_target.to(device=device, dtype=torch.float32)
        if success.shape != valid_main.shape:
            raise ValueError(
                "success_target shape mismatch: "
                f"got={tuple(success.shape)} expected={tuple(valid_main.shape)}"
            )
        rescue = rescue * (0.30 + 0.70 * torch.clamp(success, 0.0, 1.0))

    rescue = rescue - 0.10 * torch.clamp(quality_gap / 12.0, 0.0, 1.0) - 0.12 * prg_risk - 0.08 * tb_err_last
    rescue = torch.clamp(rescue, min=0.0, max=max(0.0, float(max_score)))
    return torch.where(valid_main, rescue, torch.zeros_like(rescue)), valid_main


def candidate_prg_packet_tail_quota_scores(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    *,
    max_score: float = 4.0,
) -> torch.Tensor:
    """Return per-candidate urgency scores for explicit packet-tail quota pass."""
    device = packed.ue_ids.device
    valid = candidate_valid.to(device=device, dtype=torch.bool)
    valid_main = valid[:, : packed.max_candidates] if packed.max_candidates > 0 else valid[:, :0]
    if packed.max_candidates <= 0:
        return torch.empty((packed.num_prg_tokens, 0), dtype=torch.float32, device=device)

    ue = graph.snapshot.ue_features.values.to(device=device, dtype=torch.float32)
    safe_ue_ids = packed.ue_ids.to(device=device, dtype=torch.long).clamp(
        min=0,
        max=max(0, int(graph.snapshot.static_meta.n_ue) - 1),
    )
    flat_ue = safe_ue_ids.reshape(-1)

    backlog = torch.clamp_min(ue[:, _UE_BUFFER_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    hol_delay = torch.clamp_min(ue[:, _UE_HOL_DELAY_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    ttl_slack = ue[:, _UE_TTL_SLACK_IDX].gather(0, flat_ue).reshape_as(safe_ue_ids)
    recent_deficit = torch.clamp_min(ue[:, _UE_RECENT_DEFICIT_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    stale_slots = torch.clamp_min(ue[:, _UE_STALE_SLOTS_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    tb_err_last = torch.clamp_min(ue[:, _UE_TB_ERR_LAST_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    packet_rate_gap = (
        torch.clamp(ue[:, _UE_PACKET_RATE_GAP_IDX], 0.0, 2.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
        if ue.shape[1] > _UE_PACKET_RATE_GAP_IDX
        else torch.zeros_like(backlog)
    )
    packet_delay_norm = (
        torch.clamp(ue[:, _UE_PACKET_DELAY_NORM_IDX], 0.0, 2.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
        if ue.shape[1] > _UE_PACKET_DELAY_NORM_IDX
        else torch.zeros_like(backlog)
    )
    packet_no_delivery = (
        torch.clamp(ue[:, _UE_PACKET_NO_DELIVERY_IDX], 0.0, 1.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
        if ue.shape[1] > _UE_PACKET_NO_DELIVERY_IDX
        else torch.zeros_like(backlog)
    )

    if packed.edge_attr is None:
        edge_attr = torch.zeros((packed.num_prg_tokens, packed.max_candidates, 5), dtype=torch.float32, device=device)
    else:
        edge_attr = packed.edge_attr.to(device=device, dtype=torch.float32)
        if edge_attr.shape[-1] < 5:
            pad = torch.zeros((*edge_attr.shape[:2], 5 - int(edge_attr.shape[-1])), dtype=edge_attr.dtype, device=device)
            edge_attr = torch.cat([edge_attr, pad], dim=-1)

    quality_gap = torch.clamp_min(edge_attr[..., 1], 0.0)
    prg_risk = torch.clamp(edge_attr[..., 4], 0.0, 1.0)
    backlog_score = torch.clamp(torch.log1p(backlog / 3000.0) / 6.0, 0.0, 1.5)
    hol_score = torch.clamp(hol_delay / 120.0, 0.0, 2.0)
    stale_score = torch.clamp(stale_slots / 10.0, 0.0, 1.5)
    ttl_score = torch.where(
        ttl_slack >= 0.0,
        torch.clamp((220.0 - ttl_slack) / 220.0, 0.0, 1.5),
        torch.zeros_like(ttl_slack),
    )

    score = (
        0.55 * packet_rate_gap
        + 0.25 * packet_delay_norm
        + 0.18 * packet_no_delivery
        + 0.12 * torch.clamp(recent_deficit, 0.0, 1.5)
        + 0.08 * hol_score
        + 0.06 * stale_score
        + 0.05 * ttl_score
        + 0.04 * backlog_score
        - 0.10 * prg_risk
        - 0.05 * torch.clamp(quality_gap / 12.0, 0.0, 1.0)
        - 0.04 * tb_err_last
    )
    score = torch.clamp(score, min=0.0, max=max(0.0, float(max_score)))
    return torch.where(valid_main, score, torch.zeros_like(score))


def _assign_owner_cell_slots(
    *,
    new_cells: torch.Tensor,
    new_ues: torch.Tensor,
    slot_to_cell: torch.Tensor,
    slot_valid_mask: torch.Tensor,
    action_ue_select: torch.Tensor,
    pair_slot_lookup: torch.Tensor,
    slot_prg_load: torch.Tensor,
) -> None:
    if new_cells.numel() == 0:
        return
    if new_cells.shape != new_ues.shape:
        raise ValueError(
            f"new_cells/new_ues shape mismatch: {tuple(new_cells.shape)} vs {tuple(new_ues.shape)}"
        )

    current_ue = action_ue_select.to(torch.long)
    for cell_idx in torch.unique(new_cells, sorted=True).tolist():
        cell_mask = new_cells == int(cell_idx)
        cell_ues = new_ues[cell_mask]
        if cell_ues.numel() == 0:
            continue

        same_cell = slot_valid_mask & (slot_to_cell == int(cell_idx))
        free_slots = torch.nonzero(same_cell & (current_ue < 0), as_tuple=False).flatten()
        replaceable_slots = torch.nonzero(
            same_cell
            & (current_ue >= 0)
            & (slot_prg_load <= 0),
            as_tuple=False,
        ).flatten()
        candidate_slots = torch.cat([free_slots, replaceable_slots], dim=0)
        take = min(int(cell_ues.numel()), int(candidate_slots.numel()))
        for local_idx in range(take):
            slot_id = int(candidate_slots[local_idx].item())
            ue_idx = int(cell_ues[local_idx].item())
            prev_ue = int(current_ue[slot_id].item())
            if prev_ue >= 0:
                pair_slot_lookup[int(cell_idx), prev_ue] = -1
            pair_slot_lookup[int(cell_idx), ue_idx] = slot_id
            action_ue_select[slot_id] = int(ue_idx)
            current_ue[slot_id] = int(ue_idx)


def build_prg_candidate_valid_mask(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    ue_action_class: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    meta = graph.snapshot.static_meta
    device = packed.ue_ids.device
    ue_class = torch.as_tensor(ue_action_class, dtype=torch.long, device=device).reshape(-1)
    if ue_class.shape != (meta.n_sched_ue,):
        raise ValueError(
            f"ue_action_class mismatch: got={tuple(ue_class.shape)} expected={(meta.n_sched_ue,)}"
        )

    action_mask_ue = _default_action_mask_ue(graph, device)
    selected_slot_mask = build_slot_selection_mask(
        ue_class.unsqueeze(0),
        action_mask_ue.unsqueeze(0),
        n_cell=meta.n_cell,
        action_mask_cell_ue=meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).unsqueeze(0),
    ).squeeze(0)

    selected_ue_mask = torch.zeros((meta.n_ue,), dtype=torch.bool, device=device)
    valid_slots = selected_slot_mask & (ue_class >= 0) & (ue_class < meta.n_ue)
    if bool(valid_slots.any().item()):
        selected_ue_mask[ue_class[valid_slots]] = True

    if packed.max_candidates > 0:
        safe_ue_ids = packed.ue_ids.clamp(min=0, max=max(0, meta.n_ue - 1))
        owner_cell = meta.prg_owner_cell.to(device=device, dtype=torch.long)
        owner_cell_ue_mask = meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).index_select(0, owner_cell)
        owner_cell_ue_mask = owner_cell_ue_mask.gather(1, safe_ue_ids)
        ue_enabled = action_mask_ue.gather(0, safe_ue_ids.reshape(-1)).reshape_as(safe_ue_ids)
        valid_main = packed.mask & owner_cell_ue_mask & ue_enabled
    else:
        valid_main = packed.mask
    prg_available = meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).reshape(-1)
    valid_main = valid_main & prg_available.unsqueeze(-1)
    blank_valid = torch.ones((packed.num_prg_tokens, 1), dtype=torch.bool, device=device)
    valid = torch.cat([valid_main, blank_valid], dim=-1)
    return valid, selected_slot_mask, selected_ue_mask


def candidate_prg_targets_from_type0_actions(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    *,
    action_ue_select: torch.Tensor,
    action_prg_alloc: torch.Tensor,
) -> torch.Tensor:
    meta = graph.snapshot.static_meta
    device = packed.ue_ids.device
    ue = torch.as_tensor(action_ue_select, dtype=torch.long, device=device).reshape(-1)
    prg_alloc = torch.as_tensor(action_prg_alloc, dtype=torch.long, device=device).reshape(-1)
    if ue.shape != (meta.n_sched_ue,):
        raise ValueError(
            f"action_ue_select mismatch: got={tuple(ue.shape)} expected={(meta.n_sched_ue,)}"
        )
    if prg_alloc.shape != (meta.n_prg * meta.n_cell,):
        raise ValueError(
            f"action_prg_alloc mismatch: got={tuple(prg_alloc.shape)} expected={(meta.n_prg * meta.n_cell,)}"
        )

    target = torch.full((packed.num_prg_tokens,), packed.blank_class_index, dtype=torch.long, device=device)
    if packed.max_candidates <= 0 or packed.num_prg_tokens == 0:
        return target

    owner_cell = meta.prg_owner_cell.to(device=device, dtype=torch.long)
    local_prg = meta.prg_local_index.to(device=device, dtype=torch.long)
    linear_prg = local_prg * int(meta.n_cell) + owner_cell
    slot_id = prg_alloc.gather(0, linear_prg)
    slot_valid = (slot_id >= 0) & (slot_id < int(meta.n_sched_ue))
    if not bool(slot_valid.any().item()):
        return target

    safe_slot = slot_id.clamp(min=0, max=max(0, int(meta.n_sched_ue) - 1))
    chosen_ue = ue.gather(0, safe_slot)
    ue_valid = slot_valid & (chosen_ue >= 0) & (chosen_ue < int(meta.n_ue))
    safe_ue_ids = packed.ue_ids.clamp(min=0, max=max(0, int(meta.n_ue) - 1))
    matches = packed.mask & ue_valid.unsqueeze(-1) & (safe_ue_ids == chosen_ue.unsqueeze(-1))
    has_match = matches.any(dim=-1)
    if bool(has_match.any().item()):
        target[has_match] = matches.to(dtype=torch.long).argmax(dim=-1)[has_match]
    return target


def _masked_zscore(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32, device=mask.device)
    mask_f = mask.to(dtype=values.dtype)
    count = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * mask_f).sum(dim=-1, keepdim=True) / count
    centered = torch.where(mask, values - mean, torch.zeros_like(values))
    var = (centered * centered * mask_f).sum(dim=-1, keepdim=True) / count
    return torch.where(mask, centered / torch.sqrt(var.clamp_min(1.0e-4)), torch.zeros_like(values))


def candidate_prg_soft_targets_from_type0_actions(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    *,
    action_ue_select: torch.Tensor,
    action_prg_alloc: torch.Tensor,
    temperature: float = 0.85,
    teacher_hit_bonus: float = 2.0,
    teacher_blank_bonus: float = 0.35,
    blank_safe_penalty: float = 1.25,
    blank_no_safe_bonus: float = 1.0,
    min_quality_db: float = -3.0,
    max_quality_gap_db: float = 10.0,
    max_prg_risk: float = 0.80,
    min_backlog_bytes: float = 1.0,
    max_ue_tb_err: float = 0.60,
) -> torch.Tensor:
    """Build PRG-level soft teacher targets over candidate UE classes plus blank.

    The target is a performance-oriented distribution, not a utilization-only
    label. PFQ's exact PRG choice is used as a positive prior, but other safe
    high-quality/backlogged candidates retain probability mass; blank stays
    attractive only when no candidate clears the quality/risk/backlog guard.
    """
    device = packed.ue_ids.device
    meta = graph.snapshot.static_meta
    valid = candidate_valid.to(device=device, dtype=torch.bool)
    valid_main = valid[:, : packed.max_candidates] if packed.max_candidates > 0 else valid[:, :0]
    blank_valid = valid[:, packed.blank_class_index] if valid.shape[-1] > packed.blank_class_index else torch.ones(
        (packed.num_prg_tokens,), dtype=torch.bool, device=device
    )

    if packed.max_candidates <= 0:
        target = torch.zeros((packed.num_prg_tokens, 1), dtype=torch.float32, device=device)
        target[:, 0] = 1.0
        return target

    if packed.edge_attr is None:
        edge_attr = torch.zeros((packed.num_prg_tokens, packed.max_candidates, 5), dtype=torch.float32, device=device)
    else:
        edge_attr = packed.edge_attr.to(device=device, dtype=torch.float32)
        if edge_attr.shape[-1] < 5:
            pad = torch.zeros((*edge_attr.shape[:2], 5 - int(edge_attr.shape[-1])), dtype=edge_attr.dtype, device=device)
            edge_attr = torch.cat([edge_attr, pad], dim=-1)

    quality_db = edge_attr[..., 0]
    quality_gap = edge_attr[..., 1]
    urgency = edge_attr[..., 2]
    debt = edge_attr[..., 3]
    prg_risk = torch.clamp(edge_attr[..., 4], 0.0, 1.0)

    ue = graph.snapshot.ue_features.values.to(device=device, dtype=torch.float32)
    safe_ue_ids = packed.ue_ids.to(device=device, dtype=torch.long).clamp(min=0, max=max(0, int(meta.n_ue) - 1))
    backlog = torch.clamp_min(ue[:, _UE_BUFFER_IDX], 0.0).gather(0, safe_ue_ids.reshape(-1)).reshape_as(safe_ue_ids)
    tb_err_last = torch.clamp_min(ue[:, _UE_TB_ERR_LAST_IDX], 0.0).gather(0, safe_ue_ids.reshape(-1)).reshape_as(safe_ue_ids)

    safe_main = (
        valid_main
        & (backlog >= float(min_backlog_bytes))
        & (tb_err_last <= float(max_ue_tb_err))
        & (quality_db >= float(min_quality_db))
        & (quality_gap <= float(max_quality_gap_db))
        & (prg_risk <= float(max_prg_risk))
    )
    safe_any = safe_main.any(dim=-1)

    success_proxy = torch.sigmoid((quality_db - 0.35 * quality_gap - 7.5 * prg_risk - 6.0 * tb_err_last) / 4.0)
    main_score = (
        0.55 * _masked_zscore(quality_db, valid_main)
        - 0.25 * _masked_zscore(quality_gap, valid_main)
        + 0.25 * _masked_zscore(urgency, valid_main)
        + 0.20 * _masked_zscore(debt, valid_main)
        + 0.35 * _masked_zscore(success_proxy, valid_main)
        - 0.35 * prg_risk
        - 0.15 * tb_err_last
    )

    hard_target = candidate_prg_targets_from_type0_actions(
        graph,
        packed,
        action_ue_select=action_ue_select,
        action_prg_alloc=action_prg_alloc,
    ).to(device=device, dtype=torch.long)
    target_is_main = (hard_target >= 0) & (hard_target < packed.blank_class_index)
    if bool(target_is_main.any().item()):
        main_score[target_is_main, hard_target[target_is_main]] += float(teacher_hit_bonus)

    target_is_blank = hard_target == packed.blank_class_index
    blank_score = torch.full((packed.num_prg_tokens, 1), -0.15, dtype=torch.float32, device=device)
    blank_score += torch.where(
        safe_any.unsqueeze(-1),
        torch.full_like(blank_score, -float(blank_safe_penalty)),
        torch.full_like(blank_score, float(blank_no_safe_bonus)),
    )
    blank_score += target_is_blank.to(dtype=torch.float32).unsqueeze(-1) * float(teacher_blank_bonus)
    blank_score = torch.where(blank_valid.unsqueeze(-1), blank_score, torch.full_like(blank_score, -1.0e9))

    logits = torch.cat([main_score, blank_score], dim=-1)
    logits[:, : packed.max_candidates] = logits[:, : packed.max_candidates].masked_fill(~valid_main, -1.0e9)
    logits = logits / max(float(temperature), 1.0e-3)
    return torch.softmax(logits, dim=-1)


def candidate_prg_success_targets(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return soft physical-success targets for candidate UE choices.

    Shape is ``[num_prg_tokens, max_candidates]``. Targets intentionally avoid
    direct utilization pressure: they estimate whether a candidate PRG->UE
    assignment is likely to succeed if chosen, using post-eq quality, gap to
    the best local candidate, PRG interference risk, and recent UE TB errors.
    """
    device = packed.ue_ids.device
    valid = candidate_valid.to(device=device, dtype=torch.bool)
    valid_main = valid[:, : packed.max_candidates] if packed.max_candidates > 0 else valid[:, :0]
    if packed.max_candidates <= 0:
        empty = torch.empty((packed.num_prg_tokens, 0), dtype=torch.float32, device=device)
        return empty, valid_main

    if packed.edge_attr is None:
        edge_attr = torch.zeros((packed.num_prg_tokens, packed.max_candidates, 5), dtype=torch.float32, device=device)
    else:
        edge_attr = packed.edge_attr.to(device=device, dtype=torch.float32)
        if edge_attr.shape[-1] < 5:
            pad = torch.zeros((*edge_attr.shape[:2], 5 - int(edge_attr.shape[-1])), dtype=edge_attr.dtype, device=device)
            edge_attr = torch.cat([edge_attr, pad], dim=-1)

    quality_db = edge_attr[..., 0]
    quality_gap = edge_attr[..., 1]
    prg_risk = torch.clamp(edge_attr[..., 4], 0.0, 1.0)
    ue = graph.snapshot.ue_features.values.to(device=device, dtype=torch.float32)
    safe_ue_ids = packed.ue_ids.to(device=device, dtype=torch.long).clamp(
        min=0,
        max=max(0, int(graph.snapshot.static_meta.n_ue) - 1),
    )
    tb_err_last = torch.clamp_min(ue[:, _UE_TB_ERR_LAST_IDX], 0.0).gather(0, safe_ue_ids.reshape(-1)).reshape_as(safe_ue_ids)

    success = torch.sigmoid((quality_db - 0.35 * quality_gap - 7.5 * prg_risk - 6.0 * tb_err_last) / 4.0)
    return torch.where(valid_main, success, torch.zeros_like(success)), valid_main


def candidate_prg_decode_goodput_targets(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    *,
    sampled_prg_action_class: torch.Tensor,
    decoded_chosen_ue_per_prg: torch.Tensor,
    actual_ue_goodput_bytes,
    actual_ue_prg_count=None,
    actual_ue_tb_tx_count=None,
    actual_ue_tb_err_count=None,
    target_bytes: float = 3000.0,
    max_target: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bandit-style decode/PHY goodput targets for sampled candidates.

    Only the sampled main-candidate class is supervised. Unchosen alternatives
    are left masked because the online simulator only reveals the deployed
    action's post-Type-0/post-PHY outcome.
    """
    device = packed.ue_ids.device
    valid = candidate_valid.to(device=device, dtype=torch.bool)
    valid_main = valid[:, : packed.max_candidates] if packed.max_candidates > 0 else valid[:, :0]
    target = torch.zeros((packed.num_prg_tokens, packed.max_candidates), dtype=torch.float32, device=device)
    mask = torch.zeros_like(target, dtype=torch.bool)
    if packed.max_candidates <= 0 or packed.num_prg_tokens == 0:
        return target, mask

    meta = graph.snapshot.static_meta
    n_ue = int(meta.n_ue)
    sampled = torch.as_tensor(sampled_prg_action_class, dtype=torch.long, device=device).reshape(-1)
    if sampled.shape != (packed.num_prg_tokens,):
        raise ValueError(
            "sampled_prg_action_class mismatch: "
            f"got={tuple(sampled.shape)} expected={(packed.num_prg_tokens,)}"
        )
    decoded = torch.as_tensor(decoded_chosen_ue_per_prg, dtype=torch.long, device=device).reshape(-1)
    if decoded.shape != (packed.num_prg_tokens,):
        raise ValueError(
            "decoded_chosen_ue_per_prg mismatch: "
            f"got={tuple(decoded.shape)} expected={(packed.num_prg_tokens,)}"
        )

    sample_is_main = (sampled >= 0) & (sampled < packed.max_candidates)
    if not bool(sample_is_main.any().item()):
        return target, mask
    row_idx = torch.arange(packed.num_prg_tokens, dtype=torch.long, device=device)
    sampled_safe = sampled.clamp(min=0, max=packed.max_candidates - 1)
    sampled_valid = sample_is_main & valid_main[row_idx, sampled_safe]
    if not bool(sampled_valid.any().item()):
        return target, mask

    def _ue_vector(values, *, dtype: torch.dtype, default: float = 0.0) -> torch.Tensor:
        if values is None:
            return torch.full((n_ue,), default, dtype=dtype, device=device)
        out = torch.as_tensor(values, dtype=dtype, device=device).reshape(-1)
        if int(out.numel()) < n_ue:
            pad = torch.full((n_ue - int(out.numel()),), default, dtype=dtype, device=device)
            out = torch.cat([out, pad], dim=0)
        return out[:n_ue]

    goodput = torch.clamp_min(_ue_vector(actual_ue_goodput_bytes, dtype=torch.float32), 0.0)
    if actual_ue_prg_count is None:
        decoded_valid = (decoded >= 0) & (decoded < n_ue)
        actual_prg_count = torch.zeros((n_ue,), dtype=torch.float32, device=device)
        if bool(decoded_valid.any().item()):
            actual_prg_count.scatter_add_(0, decoded[decoded_valid], torch.ones_like(decoded[decoded_valid], dtype=torch.float32))
    else:
        actual_prg_count = torch.clamp_min(_ue_vector(actual_ue_prg_count, dtype=torch.float32), 0.0)

    tb_tx = _ue_vector(actual_ue_tb_tx_count, dtype=torch.float32, default=0.0)
    tb_err = _ue_vector(actual_ue_tb_err_count, dtype=torch.float32, default=0.0)
    has_tx_success = (tb_tx <= 0.0) | (tb_err < tb_tx)

    sampled_ue = packed.ue_ids.to(device=device, dtype=torch.long)[row_idx, sampled_safe].clamp(
        min=0,
        max=max(0, n_ue - 1),
    )
    preserved_by_decode = decoded == sampled_ue
    observed = sampled_valid & preserved_by_decode
    mask[row_idx[sampled_valid], sampled_safe[sampled_valid]] = True
    if not bool(observed.any().item()):
        return target, mask

    per_prg_goodput = goodput.gather(0, sampled_ue) / actual_prg_count.gather(0, sampled_ue).clamp_min(1.0)
    per_prg_goodput = torch.where(has_tx_success.gather(0, sampled_ue), per_prg_goodput, torch.zeros_like(per_prg_goodput))
    denom = max(float(target_bytes), 1.0e-6)
    normalized = torch.clamp(per_prg_goodput / denom, min=0.0, max=max(0.0, float(max_target)))
    target[row_idx[observed], sampled_safe[observed]] = normalized[observed]
    return target, mask


def candidate_prg_pairwise_rank_targets(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    *,
    success_target: Optional[torch.Tensor] = None,
    success_weight: float = 0.70,
    service_weight: float = 0.30,
    risk_weight: float = 0.15,
    fail_risk_weight: float = 0.0,
    fail_risk_feature_offset: int = -1,
    rescue_weight: float = 0.0,
    rescue_service_target: float = 0.80,
    rescue_ttl_guard_ms: float = 60.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-PRG candidate utility targets for pairwise ranking.

    The utility is deliberately decode-facing: success probability is the main
    term, while backlog/HOL/debt/recent service shape ties among safe candidates.
    The returned utility is z-scored within each PRG so pairwise margins are
    comparable across channel snapshots.
    """
    device = packed.ue_ids.device
    valid = candidate_valid.to(device=device, dtype=torch.bool)
    valid_main = valid[:, : packed.max_candidates] if packed.max_candidates > 0 else valid[:, :0]
    if packed.max_candidates <= 0:
        empty = torch.empty((packed.num_prg_tokens, 0), dtype=torch.float32, device=device)
        return empty, valid_main

    if success_target is None:
        success_target, _mask = candidate_prg_success_targets(graph, packed, valid)
    else:
        success_target = success_target.to(device=device, dtype=torch.float32)
        if success_target.shape != valid_main.shape:
            raise ValueError(
                "success_target shape mismatch: "
                f"got={tuple(success_target.shape)} expected={tuple(valid_main.shape)}"
            )

    if packed.edge_attr is None:
        edge_attr = torch.zeros((packed.num_prg_tokens, packed.max_candidates, 5), dtype=torch.float32, device=device)
    else:
        edge_attr = packed.edge_attr.to(device=device, dtype=torch.float32)
        if edge_attr.shape[-1] < 5:
            pad = torch.zeros((*edge_attr.shape[:2], 5 - int(edge_attr.shape[-1])), dtype=edge_attr.dtype, device=device)
            edge_attr = torch.cat([edge_attr, pad], dim=-1)

    urgency = edge_attr[..., 2]
    debt_edge = edge_attr[..., 3]
    prg_risk = torch.clamp(edge_attr[..., 4], 0.0, 1.0)
    ue = graph.snapshot.ue_features.values.to(device=device, dtype=torch.float32)
    safe_ue_ids = packed.ue_ids.to(device=device, dtype=torch.long).clamp(
        min=0,
        max=max(0, int(graph.snapshot.static_meta.n_ue) - 1),
    )
    flat_ue = safe_ue_ids.reshape(-1)
    backlog = torch.clamp_min(ue[:, _UE_BUFFER_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    hol_delay = torch.clamp_min(ue[:, _UE_HOL_DELAY_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    recent_sched = torch.clamp(ue[:, _UE_RECENT_SCHED_RATIO_IDX], 0.0, 1.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    debt_ue = torch.clamp_min(ue[:, _UE_RECENT_DEFICIT_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    tb_err_last = torch.clamp_min(ue[:, _UE_TB_ERR_LAST_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)

    backlog_score = torch.log1p(backlog / 3000.0)
    hol_score = torch.log1p(hol_delay)
    service_raw = (
        0.30 * _masked_zscore(urgency, valid_main)
        + 0.25 * _masked_zscore(debt_edge + debt_ue, valid_main)
        + 0.20 * _masked_zscore(backlog_score, valid_main)
        + 0.15 * _masked_zscore(1.0 - recent_sched, valid_main)
        + 0.10 * _masked_zscore(hol_score, valid_main)
    )
    utility = (
        float(success_weight) * success_target
        + float(service_weight) * service_raw
        - float(risk_weight) * prg_risk
        - 0.10 * tb_err_last
    )
    if float(fail_risk_weight) > 0.0 and 0 <= int(fail_risk_feature_offset) < int(edge_attr.shape[-1]):
        candidate_fail_risk = torch.clamp(edge_attr[..., int(fail_risk_feature_offset)], 0.0, 1.0)
        utility = utility - float(fail_risk_weight) * candidate_fail_risk
    if float(rescue_weight) > 0.0:
        rescue_target, _rescue_mask = candidate_prg_rescue_priority_scores(
            graph,
            packed,
            valid,
            success_target=success_target,
            service_target=float(rescue_service_target),
            ttl_guard_ms=float(rescue_ttl_guard_ms),
        )
        utility = utility + float(rescue_weight) * _masked_zscore(rescue_target, valid_main)
    utility = _masked_zscore(utility, valid_main)
    return torch.where(valid_main, utility, torch.zeros_like(utility)), valid_main


def candidate_prg_service_protection_mask(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    *,
    success_target: Optional[torch.Tensor] = None,
    min_success_target: float = 0.70,
    min_backlog_bytes: float = 1.0,
    max_ue_tb_err: float = 0.45,
    min_quality_db: float = -2.0,
    max_quality_gap_db: float = 8.0,
    max_prg_risk: float = 0.70,
    min_debt: float = 0.25,
    min_hol_delay_ms: float = 1.0,
    max_recent_scheduled_ratio: float = 0.65,
) -> torch.Tensor:
    """Return PRGs where blank supervision should yield to service pressure.

    The mask is intentionally stricter than "has backlog": a PRG is protected
    only when at least one valid candidate has plausible physical success and
    represents a UE with service debt, HOL delay, or low recent scheduling.
    """
    device = packed.ue_ids.device
    valid = candidate_valid.to(device=device, dtype=torch.bool)
    valid_main = valid[:, : packed.max_candidates] if packed.max_candidates > 0 else valid[:, :0]
    if packed.max_candidates <= 0 or packed.num_prg_tokens == 0:
        return torch.zeros((packed.num_prg_tokens,), dtype=torch.bool, device=device)

    if packed.edge_attr is None:
        edge_attr = torch.zeros((packed.num_prg_tokens, packed.max_candidates, 5), dtype=torch.float32, device=device)
    else:
        edge_attr = packed.edge_attr.to(device=device, dtype=torch.float32)
        if edge_attr.shape[-1] < 5:
            pad = torch.zeros((*edge_attr.shape[:2], 5 - int(edge_attr.shape[-1])), dtype=edge_attr.dtype, device=device)
            edge_attr = torch.cat([edge_attr, pad], dim=-1)

    quality_db = edge_attr[..., 0]
    quality_gap = edge_attr[..., 1]
    prg_risk = torch.clamp(edge_attr[..., 4], 0.0, 1.0)

    ue = graph.snapshot.ue_features.values.to(device=device, dtype=torch.float32)
    safe_ue_ids = packed.ue_ids.to(device=device, dtype=torch.long).clamp(
        min=0,
        max=max(0, int(graph.snapshot.static_meta.n_ue) - 1),
    )
    flat_ue = safe_ue_ids.reshape(-1)
    backlog = torch.clamp_min(ue[:, _UE_BUFFER_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    tb_err_last = torch.clamp_min(ue[:, _UE_TB_ERR_LAST_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    hol_delay = torch.clamp_min(ue[:, _UE_HOL_DELAY_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    recent_sched = torch.clamp(ue[:, _UE_RECENT_SCHED_RATIO_IDX], 0.0, 1.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    debt = torch.clamp_min(ue[:, _UE_RECENT_DEFICIT_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)

    if success_target is None:
        success_target, _mask = candidate_prg_success_targets(graph, packed, valid)
    else:
        success_target = success_target.to(device=device, dtype=torch.float32)
        if success_target.shape != valid_main.shape:
            raise ValueError(
                "success_target shape mismatch: "
                f"got={tuple(success_target.shape)} expected={tuple(valid_main.shape)}"
            )

    service_pressure = (
        (debt >= float(min_debt))
        | (hol_delay >= float(min_hol_delay_ms))
        | (recent_sched <= float(max_recent_scheduled_ratio))
    )
    physically_plausible = (
        (success_target >= float(min_success_target))
        & (tb_err_last <= float(max_ue_tb_err))
        & (quality_db >= float(min_quality_db))
        & (quality_gap <= float(max_quality_gap_db))
        & (prg_risk <= float(max_prg_risk))
    )
    protected = (
        valid_main
        & (backlog >= float(min_backlog_bytes))
        & service_pressure
        & physically_plausible
    )
    return protected.any(dim=-1)


def guarded_safe_fill_candidate_choices(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    prg_choice: torch.Tensor,
    *,
    max_per_cell: int = 0,
    min_quality_db: float = -2.0,
    max_quality_gap_db: float = 8.0,
    max_prg_risk: float = 0.70,
    min_backlog_bytes: float = 1.0,
    max_ue_tb_err: float = 0.45,
) -> tuple[torch.Tensor, int]:
    """Fill only blank PRGs whose best candidate clears conservative guards."""
    limit = int(max_per_cell)
    if limit <= 0 or packed.max_candidates <= 0 or packed.edge_attr is None:
        return prg_choice, 0

    meta = graph.snapshot.static_meta
    device = packed.ue_ids.device
    choice = torch.as_tensor(prg_choice, dtype=torch.long, device=device).reshape(-1).clone()
    valid_main = candidate_valid.to(device=device, dtype=torch.bool)[:, : packed.max_candidates]
    edge_attr = packed.edge_attr.to(device=device, dtype=torch.float32)
    if edge_attr.shape[-1] < 5:
        return choice, 0

    quality_db = edge_attr[..., 0]
    quality_gap = edge_attr[..., 1]
    urgency = edge_attr[..., 2]
    debt = edge_attr[..., 3]
    prg_risk = torch.clamp(edge_attr[..., 4], 0.0, 1.0)

    ue = graph.snapshot.ue_features.values.to(device=device, dtype=torch.float32)
    safe_ue_ids = packed.ue_ids.to(device=device, dtype=torch.long).clamp(min=0, max=max(0, int(meta.n_ue) - 1))
    backlog = torch.clamp_min(ue[:, _UE_BUFFER_IDX], 0.0).gather(0, safe_ue_ids.reshape(-1)).reshape_as(safe_ue_ids)
    tb_err_last = torch.clamp_min(ue[:, _UE_TB_ERR_LAST_IDX], 0.0).gather(0, safe_ue_ids.reshape(-1)).reshape_as(safe_ue_ids)

    safe = (
        valid_main
        & (choice == packed.blank_class_index).unsqueeze(-1)
        & (backlog >= float(min_backlog_bytes))
        & (tb_err_last <= float(max_ue_tb_err))
        & (quality_db >= float(min_quality_db))
        & (quality_gap <= float(max_quality_gap_db))
        & (prg_risk <= float(max_prg_risk))
    )
    if not bool(safe.any().item()):
        return choice, 0

    score = quality_db - 0.45 * quality_gap + 0.10 * urgency + 0.08 * debt - 7.5 * prg_risk - 4.0 * tb_err_last
    score = score.masked_fill(~safe, -1.0e9)
    best_score, best_idx = score.max(dim=-1)
    fillable = best_score > -1.0e8
    owner_cell = meta.prg_owner_cell.to(device=device, dtype=torch.long)
    filled = 0
    for cell_idx in range(int(meta.n_cell)):
        cell_candidates = torch.nonzero(fillable & (owner_cell == cell_idx), as_tuple=False).flatten()
        if cell_candidates.numel() == 0:
            continue
        order = torch.argsort(best_score[cell_candidates], descending=True, stable=True)
        take = cell_candidates[order[:limit]]
        choice[take] = best_idx[take]
        filled += int(take.numel())
    return choice, filled


def count_missed_safe_candidate_opportunities(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    prg_chosen_mask: torch.Tensor,
    ue_prg_assigned: torch.Tensor,
    ue_prg_cap: torch.Tensor,
    *,
    min_quality_db: float = -2.0,
    max_quality_gap_db: float = 8.0,
    max_prg_risk: float = 0.70,
    min_backlog_bytes: float = 1.0,
    max_ue_tb_err: float = 0.45,
) -> int:
    """Count final blank PRGs that still had a guarded, headroom-safe candidate."""
    if packed.max_candidates <= 0 or packed.edge_attr is None:
        return 0
    device = packed.ue_ids.device
    final_blank = ~torch.as_tensor(prg_chosen_mask, dtype=torch.bool, device=device).reshape(-1)
    if not bool(final_blank.any().item()):
        return 0
    valid_main = candidate_valid.to(device=device, dtype=torch.bool)[:, : packed.max_candidates]
    edge_attr = packed.edge_attr.to(device=device, dtype=torch.float32)
    if edge_attr.shape[-1] < 5:
        return 0

    meta = graph.snapshot.static_meta
    quality_db = edge_attr[..., 0]
    quality_gap = edge_attr[..., 1]
    prg_risk = torch.clamp(edge_attr[..., 4], 0.0, 1.0)
    ue = graph.snapshot.ue_features.values.to(device=device, dtype=torch.float32)
    safe_ue_ids = packed.ue_ids.to(device=device, dtype=torch.long).clamp(min=0, max=max(0, int(meta.n_ue) - 1))
    flat_ue = safe_ue_ids.reshape(-1)
    backlog = torch.clamp_min(ue[:, _UE_BUFFER_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    tb_err_last = torch.clamp_min(ue[:, _UE_TB_ERR_LAST_IDX], 0.0).gather(0, flat_ue).reshape_as(safe_ue_ids)
    hard_remaining = torch.clamp_min(ue_prg_cap - ue_prg_assigned, 0)
    hard_remaining = hard_remaining.gather(0, flat_ue).reshape_as(safe_ue_ids)

    safe = (
        final_blank.unsqueeze(-1)
        & valid_main
        & (hard_remaining > 0)
        & (backlog >= float(min_backlog_bytes))
        & (tb_err_last <= float(max_ue_tb_err))
        & (quality_db >= float(min_quality_db))
        & (quality_gap <= float(max_quality_gap_db))
        & (prg_risk <= float(max_prg_risk))
    )
    return int(safe.any(dim=-1).sum().item())


def apply_candidate_blank_budget(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    prg_logits: torch.Tensor,
    prg_choice: torch.Tensor,
    *,
    max_blank_per_cell: int = -1,
    min_decision_logit: float = -1.0e9,
) -> tuple[torch.Tensor, int]:
    """Cap deployment blank choices per cell before Type-0 candidate decode."""
    budget = int(max_blank_per_cell)
    if budget < 0 or packed.max_candidates <= 0 or packed.num_prg_tokens == 0:
        return prg_choice, 0

    device = prg_logits.device
    logits = torch.as_tensor(prg_logits, dtype=torch.float32, device=device)
    if logits.shape != (packed.num_prg_tokens, packed.max_candidates + 1):
        raise ValueError(
            "prg_logits mismatch: "
            f"got={tuple(logits.shape)} expected={(packed.num_prg_tokens, packed.max_candidates + 1)}"
        )
    choice = torch.as_tensor(prg_choice, dtype=torch.long, device=device).reshape(-1).clone()
    if choice.shape != (packed.num_prg_tokens,):
        raise ValueError(
            f"prg_choice mismatch: got={tuple(choice.shape)} expected={(packed.num_prg_tokens,)}"
        )

    blank_class = int(packed.blank_class_index)
    main_logits = logits[:, :blank_class]
    best_main_logit, best_main_idx = main_logits.max(dim=-1)
    has_valid_main = best_main_logit > -1.0e8
    blank_logit = logits[:, blank_class]
    decision_logit = blank_logit - best_main_logit
    owner_cell = graph.snapshot.static_meta.prg_owner_cell.to(device=device, dtype=torch.long)
    blank_mask = (choice == blank_class) & has_valid_main
    if not bool(blank_mask.any().item()):
        return choice, 0

    repaired = 0
    keep_threshold = float(min_decision_logit)
    for cell_idx in range(int(graph.snapshot.static_meta.n_cell)):
        cell_blanks = torch.nonzero(blank_mask & (owner_cell == cell_idx), as_tuple=False).flatten()
        if cell_blanks.numel() == 0:
            continue
        strong_blank = cell_blanks[decision_logit[cell_blanks] >= keep_threshold]
        if strong_blank.numel() > 0 and budget > 0:
            order = torch.argsort(decision_logit[strong_blank], descending=True, stable=True)
            keep = strong_blank[order[:budget]]
            keep_mask = torch.zeros((packed.num_prg_tokens,), dtype=torch.bool, device=device)
            keep_mask[keep] = True
        else:
            keep_mask = torch.zeros((packed.num_prg_tokens,), dtype=torch.bool, device=device)
        replace = cell_blanks[~keep_mask[cell_blanks]]
        if replace.numel() == 0:
            continue
        choice[replace] = best_main_idx[replace].to(dtype=torch.long)
        repaired += int(replace.numel())
    return choice, repaired


def _candidate_seq_fail_risk_features(
    packed: PackedPrgCandidates,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fail_risk_feature_offset: int = -1,
) -> torch.Tensor:
    if packed.edge_attr is None:
        return torch.zeros((packed.num_prg_tokens, packed.max_candidates), dtype=dtype, device=device)
    edge_attr = packed.edge_attr.to(device=device, dtype=dtype)
    if edge_attr.dim() != 3:
        return torch.zeros((packed.num_prg_tokens, packed.max_candidates), dtype=dtype, device=device)
    attr_dim = int(edge_attr.shape[-1])
    offset = int(fail_risk_feature_offset)
    if offset < 0:
        if attr_dim >= 14:
            offset = 6
        elif attr_dim >= 13:
            offset = 5
    if offset < 0 or offset >= attr_dim:
        return torch.zeros((packed.num_prg_tokens, packed.max_candidates), dtype=dtype, device=device)
    return torch.clamp(edge_attr[..., offset], min=0.0, max=1.0)


def _candidate_seq_state_delta(
    seq_state_weights: Optional[torch.Tensor],
    *,
    prg_idx: int,
    used: torch.Tensor,
    cap: torch.Tensor,
    remaining: torch.Tensor,
    fail_risk: torch.Tensor,
    coef: float,
) -> torch.Tensor:
    if seq_state_weights is None or float(coef) == 0.0:
        return torch.zeros_like(used, dtype=torch.float32)
    weights = seq_state_weights[int(prg_idx)]
    if weights.numel() == 0 or int(weights.shape[-1]) <= 0:
        return torch.zeros_like(used, dtype=torch.float32)

    used_f = used.to(dtype=weights.dtype)
    cap_raw_f = cap.to(dtype=weights.dtype)
    cap_f = cap_raw_f.clamp_min(1.0)
    remaining_f = remaining.to(dtype=weights.dtype)
    fail_risk_f = fail_risk.to(device=weights.device, dtype=weights.dtype)
    state = torch.stack(
        [
            used_f / cap_f,
            remaining_f.clamp_min(0.0) / cap_f,
            (remaining_f - 1.0).clamp_min(-1.0) / cap_f,
            (remaining_f <= 0.0).to(dtype=weights.dtype),
            ((used_f <= 0.0) & (cap_raw_f > 0.0)).to(dtype=weights.dtype),
            cap_f.clamp(max=8.0) / 8.0,
            fail_risk_f.clamp(min=0.0, max=1.0),
            torch.ones_like(used_f, dtype=weights.dtype),
        ],
        dim=-1,
    )
    dim = int(weights.shape[-1])
    if dim < int(state.shape[-1]):
        state = state[..., :dim]
    elif dim > int(state.shape[-1]):
        pad = torch.zeros((*state.shape[:-1], dim - int(state.shape[-1])), dtype=state.dtype, device=state.device)
        state = torch.cat((state, pad), dim=-1)
    return float(coef) * (weights * state).sum(dim=-1)


def select_budgeted_candidate_actions(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    prg_logits: torch.Tensor,
    *,
    mode: str = "greedy",
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
    seq_state_weights: Optional[torch.Tensor] = None,
    seq_state_coef: float = 1.0,
    seq_fail_risk_feature_offset: int = -1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select candidate PRG actions with explicit per-UE remaining-headroom state.

    The policy still provides per-PRG scorer logits, but this selector turns
    deployment into a budget-aware pass: PRGs are visited from high to low
    scorer confidence, and each candidate is adjusted by how many PRGs its UE
    has already consumed relative to the estimated demand/headroom cap.  A
    policy blank is only deployed when no valid candidate remains; this keeps
    the selector aligned with the blank-budget deployment path used by the
    candidate scorer experiments.
    """
    select_mode = str(mode).lower()
    if select_mode not in ("greedy", "sample"):
        raise ValueError(f"mode must be greedy or sample, got {mode!r}")

    device = packed.ue_ids.device
    logits = torch.as_tensor(prg_logits, dtype=torch.float32, device=device)
    expected = (packed.num_prg_tokens, packed.max_candidates + 1)
    if tuple(logits.shape) != expected:
        raise ValueError(f"prg_logits mismatch: got={tuple(logits.shape)} expected={expected}")
    valid = candidate_valid.to(device=device, dtype=torch.bool)
    if tuple(valid.shape) != expected:
        raise ValueError(f"candidate_valid mismatch: got={tuple(valid.shape)} expected={expected}")

    blank_class = int(packed.blank_class_index)
    choice = torch.full((packed.num_prg_tokens,), blank_class, dtype=torch.long, device=device)
    if packed.max_candidates <= 0 or packed.num_prg_tokens == 0:
        return choice, {
            "candidate_budgeted_changed_count": 0.0,
            "candidate_budgeted_selected_main_count": 0.0,
            "candidate_budgeted_blank_count": float(packed.num_prg_tokens),
            "candidate_budgeted_no_headroom_count": 0.0,
            "candidate_budgeted_overcap_candidate_count": 0.0,
            "candidate_budgeted_used_prg_sum": 0.0,
            "candidate_budgeted_active_ue_count": 0.0,
            "candidate_budgeted_tail_bias_coef": float(max(0.0, float(tail_bias_coef))),
            "candidate_budgeted_tail_bias_candidate_mean": 0.0,
            "candidate_budgeted_tail_bias_selected_mean": 0.0,
            "candidate_budgeted_tail_bias_active_count": 0.0,
            "candidate_budgeted_rescue_bias_coef": float(max(0.0, float(rescue_bias_coef))),
            "candidate_budgeted_rescue_bias_candidate_mean": 0.0,
            "candidate_budgeted_rescue_bias_selected_mean": 0.0,
            "candidate_budgeted_rescue_bias_active_count": 0.0,
            "candidate_budgeted_completion_bonus": float(max(0.0, float(completion_bonus))),
            "candidate_budgeted_completion_candidate_mean": 0.0,
            "candidate_budgeted_completion_selected_mean": 0.0,
            "candidate_budgeted_completion_active_count": 0.0,
            "candidate_budgeted_tail_quota_enabled": float(bool(tail_quota_enable)),
            "candidate_budgeted_tail_quota_candidate_mean": 0.0,
            "candidate_budgeted_tail_quota_active_ue_count": 0.0,
            "candidate_budgeted_tail_quota_selected_count": 0.0,
            "candidate_budgeted_tail_quota_selected_mean": 0.0,
            "candidate_budgeted_tail_quota_served_ue_count": 0.0,
        }

    valid_main = valid[:, :blank_class]
    seq_weights = None
    if seq_state_weights is not None and float(seq_state_coef) != 0.0:
        candidate_seq_weights = torch.as_tensor(seq_state_weights, dtype=logits.dtype, device=device)
        if candidate_seq_weights.dim() == 3 and tuple(candidate_seq_weights.shape[:2]) == (
            packed.num_prg_tokens,
            packed.max_candidates,
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
    )
    tail_coef = max(0.0, float(tail_bias_coef))
    tail_adjustment = tail_coef * tail_bias
    success_target_for_rescue, _success_mask = candidate_prg_success_targets(graph, packed, valid)
    rescue_bias, _rescue_valid = candidate_prg_rescue_priority_scores(
        graph,
        packed,
        valid,
        success_target=success_target_for_rescue,
        service_target=float(rescue_bias_service_target),
        ttl_guard_ms=float(rescue_bias_ttl_guard_ms),
        max_score=float(rescue_bias_max),
    )
    rescue_coef = max(0.0, float(rescue_bias_coef))
    rescue_adjustment = rescue_coef * rescue_bias
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
    raw_choice = logits.masked_fill(~valid, -1.0e9).argmax(dim=-1)
    best_main_logit = (
        logits[:, :blank_class] + tail_adjustment + rescue_adjustment + completion_adjustment
    ).masked_fill(~valid_main, -1.0e9).max(dim=-1).values
    order = torch.argsort(best_main_logit, descending=True, stable=True)
    tail_quota_on = (
        bool(tail_quota_enable)
        and int(tail_quota_topk_ues) > 0
        and int(tail_quota_prgs_per_ue) > 0
        and float(tail_quota_bonus) > 0.0
    )
    tail_quota_score = (
        candidate_prg_packet_tail_quota_scores(
            graph,
            packed,
            valid,
            max_score=float(tail_quota_max_score),
        ).to(device=device, dtype=logits.dtype)
        if tail_quota_on
        else torch.zeros_like(tail_bias, dtype=logits.dtype, device=device)
    )
    if packed.edge_attr is not None and int(packed.edge_attr.shape[-1]) > 4:
        tail_quota_risk = torch.clamp(packed.edge_attr.to(device=device, dtype=logits.dtype)[..., 4], 0.0, 1.0)
    else:
        tail_quota_risk = torch.zeros_like(tail_quota_score)

    changed_count = 0
    selected_main_count = 0
    no_headroom_count = 0
    blocked_overcap_count = 0
    selected_tail_bias_sum = 0.0
    selected_rescue_bias_sum = 0.0
    selected_completion_sum = 0.0
    selected_tail_quota_sum = 0.0
    selected_tail_quota_count = 0
    tail_quota_active_ue_count = 0
    tail_quota_served_mask = torch.zeros_like(ue_prg_cap, dtype=torch.bool)
    assigned = torch.zeros((packed.num_prg_tokens,), dtype=torch.bool, device=device)

    if tail_quota_on:
        flat_valid = valid_main.reshape(-1)
        flat_score = tail_quota_score.reshape(-1)
        flat_ue = safe_ue_ids.reshape(-1)
        quota_mask = flat_valid & (flat_score >= float(tail_quota_min_score))
        tail_quota_remaining = torch.zeros_like(ue_prg_cap, dtype=torch.long)
        if bool(quota_mask.any().item()):
            quota_ue = flat_ue[quota_mask]
            quota_val = flat_score[quota_mask]
            score_sum = torch.zeros_like(ue_prg_cap, dtype=logits.dtype)
            score_count = torch.zeros_like(ue_prg_cap, dtype=logits.dtype)
            score_sum.scatter_add_(0, quota_ue, quota_val)
            score_count.scatter_add_(0, quota_ue, torch.ones_like(quota_val))
            ue_quota_mean = score_sum / score_count.clamp_min(1.0)
            active = (score_count > 0.0) & (ue_prg_cap > 0) & (ue_quota_mean >= float(tail_quota_min_score))
            active_count = int(active.sum().item())
            if active_count > 0:
                topk = min(int(tail_quota_topk_ues), active_count)
                top_values, top_idx = torch.topk(ue_quota_mean.masked_fill(~active, -1.0e9), k=topk)
                active_top = top_values >= float(tail_quota_min_score)
                if bool(active_top.any().item()):
                    selected_ues = top_idx[active_top]
                    tail_quota_remaining[selected_ues] = int(tail_quota_prgs_per_ue)
                    tail_quota_active_ue_count = int(selected_ues.numel())

        if int(tail_quota_remaining.sum().item()) > 0:
            quota_best = (
                logits[:, :blank_class]
                + tail_adjustment
                + rescue_adjustment
                + float(tail_quota_order_bonus) * tail_quota_score
            ).masked_fill(~valid_main, -1.0e9).max(dim=-1).values
            quota_order = torch.argsort(quota_best, descending=True, stable=True)
            risk_max = float(tail_quota_risk_max)
            for prg_idx in quota_order.tolist():
                if int(tail_quota_remaining.sum().item()) <= 0:
                    break
                prg_valid = valid_main[prg_idx]
                if not bool(prg_valid.any().item()):
                    continue
                ue_ids = safe_ue_ids[prg_idx]
                used = ue_prg_used.gather(0, ue_ids).to(dtype=torch.float32)
                cap = ue_prg_cap.gather(0, ue_ids)
                remaining = cap - ue_prg_used.gather(0, ue_ids)
                quota_remaining = tail_quota_remaining.gather(0, ue_ids) > 0
                selectable = prg_valid & quota_remaining & (remaining > 0)
                if risk_max >= 0.0:
                    selectable = selectable & (tail_quota_risk[prg_idx] <= risk_max)
                if not bool(selectable.any().item()):
                    continue

                main_scores = logits[prg_idx, :blank_class].clone()
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
                adjusted = (
                    main_scores
                    + tail_adjustment[prg_idx]
                    + rescue_adjustment[prg_idx]
                    + completion_coef * completion_score_prg
                    + float(tail_quota_bonus) * tail_quota_score[prg_idx]
                    - float(usage_penalty) * used
                ).masked_fill(~selectable, -1.0e9)
                if not bool((adjusted > -1.0e8).any().item()):
                    continue
                selected = int(adjusted.argmax().item())
                choice[prg_idx] = int(selected)
                assigned[prg_idx] = True
                if int(selected) != int(raw_choice[prg_idx].item()):
                    changed_count += 1
                ue_idx = int(ue_ids[selected].item())
                if 0 <= ue_idx < int(ue_prg_used.numel()):
                    ue_prg_used[ue_idx] += 1
                    tail_quota_remaining[ue_idx] = torch.clamp_min(tail_quota_remaining[ue_idx] - 1, 0)
                    tail_quota_served_mask[ue_idx] = True
                selected_main_count += 1
                selected_tail_bias_sum += float(tail_bias[prg_idx, selected].item())
                selected_rescue_bias_sum += float(rescue_bias[prg_idx, selected].item())
                selected_completion_sum += float(completion_score_prg[selected].item())
                selected_tail_quota_sum += float(tail_quota_score[prg_idx, selected].item())
                selected_tail_quota_count += 1

    for prg_idx in order.tolist():
        if bool(assigned[prg_idx].item()):
            continue
        main_scores = logits[prg_idx, :blank_class].clone()
        prg_valid = valid_main[prg_idx]
        if not bool(prg_valid.any().item()):
            choice[prg_idx] = blank_class
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
        if bool((prg_valid & ~in_headroom).any().item()):
            blocked_overcap_count += int((prg_valid & ~in_headroom).sum().item())

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
            no_headroom_count += 1
            adjusted = torch.full_like(main_scores, -1.0e9)
        else:
            no_headroom_count += 1
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

        has_candidate = bool((adjusted > -1.0e8).any().item())
        if not has_candidate:
            selected = blank_class
        elif select_mode == "sample":
            probs = torch.softmax(adjusted, dim=0)
            if not bool(torch.isfinite(probs).all().item()) or float(probs.sum().item()) <= 0.0:
                selected = int(adjusted.argmax().item())
            else:
                selected = int(torch.multinomial(probs, num_samples=1).item())
        else:
            selected = int(adjusted.argmax().item())
        choice[prg_idx] = int(selected)
        if int(selected) != int(raw_choice[prg_idx].item()):
            changed_count += 1
        if 0 <= selected < blank_class:
            ue_idx = int(ue_ids[selected].item())
            if 0 <= ue_idx < int(ue_prg_used.numel()):
                ue_prg_used[ue_idx] += 1
            selected_main_count += 1
            selected_tail_bias_sum += float(tail_bias[prg_idx, selected].item())
            selected_rescue_bias_sum += float(rescue_bias[prg_idx, selected].item())
            selected_completion_sum += float(completion_score_prg[selected].item())

    valid_tail = tail_bias[valid_main]
    candidate_tail_mean = float(valid_tail.mean().item()) if valid_tail.numel() > 0 else 0.0
    valid_rescue = rescue_bias[valid_main]
    candidate_rescue_mean = float(valid_rescue.mean().item()) if valid_rescue.numel() > 0 else 0.0
    valid_completion = completion_score[valid_main]
    candidate_completion_mean = float(valid_completion.mean().item()) if valid_completion.numel() > 0 else 0.0
    valid_tail_quota = tail_quota_score[valid_main]
    candidate_tail_quota_mean = float(valid_tail_quota.mean().item()) if valid_tail_quota.numel() > 0 else 0.0

    return choice, {
        "candidate_budgeted_changed_count": float(changed_count),
        "candidate_budgeted_selected_main_count": float(selected_main_count),
        "candidate_budgeted_blank_count": float((choice == blank_class).sum().item()),
        "candidate_budgeted_no_headroom_count": float(no_headroom_count),
        "candidate_budgeted_overcap_candidate_count": float(blocked_overcap_count),
        "candidate_budgeted_used_prg_sum": float(ue_prg_used.sum().item()),
        "candidate_budgeted_active_ue_count": float((ue_prg_used > 0).sum().item()),
        "candidate_budgeted_tail_bias_coef": float(tail_coef),
        "candidate_budgeted_tail_bias_candidate_mean": float(candidate_tail_mean),
        "candidate_budgeted_tail_bias_selected_mean": (
            float(selected_tail_bias_sum) / max(float(selected_main_count), 1.0)
        ),
        "candidate_budgeted_tail_bias_active_count": float(((tail_bias > 0.5) & valid_main).sum().item()),
        "candidate_budgeted_rescue_bias_coef": float(rescue_coef),
        "candidate_budgeted_rescue_bias_candidate_mean": float(candidate_rescue_mean),
        "candidate_budgeted_rescue_bias_selected_mean": (
            float(selected_rescue_bias_sum) / max(float(selected_main_count), 1.0)
        ),
        "candidate_budgeted_rescue_bias_active_count": float(((rescue_bias > 0.5) & valid_main).sum().item()),
        "candidate_budgeted_completion_bonus": float(completion_coef),
        "candidate_budgeted_completion_candidate_mean": float(candidate_completion_mean),
        "candidate_budgeted_completion_selected_mean": (
            float(selected_completion_sum) / max(float(selected_main_count), 1.0)
        ),
        "candidate_budgeted_completion_active_count": float(((completion_score > 0.5) & valid_main).sum().item()),
        "candidate_budgeted_tail_quota_enabled": float(tail_quota_on),
        "candidate_budgeted_tail_quota_candidate_mean": float(candidate_tail_quota_mean),
        "candidate_budgeted_tail_quota_active_ue_count": float(tail_quota_active_ue_count),
        "candidate_budgeted_tail_quota_selected_count": float(selected_tail_quota_count),
        "candidate_budgeted_tail_quota_selected_mean": (
            float(selected_tail_quota_sum) / max(float(selected_tail_quota_count), 1.0)
        ),
        "candidate_budgeted_tail_quota_served_ue_count": float(tail_quota_served_mask.sum().item()),
        "candidate_budgeted_seq_state_coef": float(seq_state_coef) if seq_weights is not None else 0.0,
    }


def select_and_decode_budgeted_candidate_actions(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    ue_action_class: torch.Tensor,
    prg_logits: torch.Tensor,
    *,
    mode: str = "greedy",
    demand_cap_slack_bytes: float = 3000.0,
    hard_cap: bool = True,
    usage_penalty: float = 0.35,
    overcap_penalty: float = 8.0,
    first_service_bonus: float = 0.0,
    safe_fill_max_per_cell: int = 0,
    safe_fill_min_quality_db: float = -2.0,
    safe_fill_max_quality_gap_db: float = 8.0,
    safe_fill_max_prg_risk: float = 0.70,
    safe_fill_min_backlog_bytes: float = 1.0,
    safe_fill_max_ue_tb_err: float = 0.45,
    headroom_rank_mode: str = "stable",
    marginal_headroom_slack_bytes: float = -1.0,
    candidate_rank_scores: Optional[torch.Tensor] = None,
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
    seq_state_weights: Optional[torch.Tensor] = None,
    seq_state_coef: float = 1.0,
    seq_fail_risk_feature_offset: int = -1,
) -> tuple[DecodedType0Action, torch.Tensor, dict[str, float]]:
    """Canonical candidate action operator for budgeted deployment.

    It is the single Python entry point for candidate scorer deployment:
    first select PRG candidates with the budgeted hard/soft cap, then decode
    that exact choice tensor into native Type-0 actions under the same demand
    cap slack and headroom ranking knobs.
    """
    prg_choice, metrics = select_budgeted_candidate_actions(
        graph,
        packed,
        candidate_valid,
        prg_logits,
        mode=mode,
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
        tail_quota_enable=bool(tail_quota_enable),
        tail_quota_min_score=float(tail_quota_min_score),
        tail_quota_topk_ues=int(tail_quota_topk_ues),
        tail_quota_prgs_per_ue=int(tail_quota_prgs_per_ue),
        tail_quota_bonus=float(tail_quota_bonus),
        tail_quota_order_bonus=float(tail_quota_order_bonus),
        tail_quota_max_score=float(tail_quota_max_score),
        tail_quota_risk_max=float(tail_quota_risk_max),
        seq_state_weights=seq_state_weights,
        seq_state_coef=float(seq_state_coef),
        seq_fail_risk_feature_offset=int(seq_fail_risk_feature_offset),
    )
    decoded = decode_candidate_actions_to_type0(
        graph,
        packed,
        ue_action_class,
        prg_choice,
        safe_fill_max_per_cell=int(safe_fill_max_per_cell),
        safe_fill_min_quality_db=float(safe_fill_min_quality_db),
        safe_fill_max_quality_gap_db=float(safe_fill_max_quality_gap_db),
        safe_fill_max_prg_risk=float(safe_fill_max_prg_risk),
        safe_fill_min_backlog_bytes=float(safe_fill_min_backlog_bytes),
        safe_fill_max_ue_tb_err=float(safe_fill_max_ue_tb_err),
        demand_cap_slack_bytes=float(demand_cap_slack_bytes),
        marginal_headroom_slack_bytes=float(marginal_headroom_slack_bytes),
        headroom_rank_mode=str(headroom_rank_mode),
        candidate_rank_scores=candidate_rank_scores,
    )
    metrics = dict(metrics)
    metrics.update(
        {
            "candidate_budgeted_post_decode_policy_blank_count": float(decoded.post_decode_policy_blank_count),
            "candidate_budgeted_post_decode_no_valid_candidate_count": float(
                decoded.post_decode_no_valid_candidate_count
            ),
            "candidate_budgeted_post_decode_demand_cap_count": float(decoded.post_decode_demand_cap_count),
            "candidate_budgeted_post_decode_no_slot_count": float(decoded.post_decode_no_slot_count),
            "candidate_budgeted_post_decode_other_drop_count": float(decoded.post_decode_other_drop_count),
            "candidate_budgeted_final_blank_count": float(decoded.final_blank_count),
        }
    )
    return decoded, prg_choice, metrics


def _candidate_decode_headroom_scores(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    *,
    mode: str = "utility",
    candidate_rank_scores: Optional[torch.Tensor] = None,
    logit_weight: float = 0.20,
) -> torch.Tensor:
    valid = candidate_valid.to(device=packed.ue_ids.device, dtype=torch.bool)
    valid_main = valid[:, : packed.max_candidates] if packed.max_candidates > 0 else valid[:, :0]
    score_mode = str(mode).lower()
    if score_mode == "phy":
        score, valid_main = candidate_prg_success_targets(graph, packed, candidate_valid)
    elif score_mode == "logit":
        if candidate_rank_scores is None:
            return torch.zeros((packed.num_prg_tokens, packed.max_candidates), dtype=torch.float32, device=packed.ue_ids.device)
        score = torch.as_tensor(candidate_rank_scores, dtype=torch.float32, device=packed.ue_ids.device)
        expected_shape = (packed.num_prg_tokens, packed.max_candidates)
        if score.shape != expected_shape:
            raise ValueError(f"candidate_rank_scores mismatch: got={tuple(score.shape)} expected={expected_shape}")
    else:
        score, valid_main = candidate_prg_pairwise_rank_targets(graph, packed, candidate_valid)
        if candidate_rank_scores is not None and packed.max_candidates > 0:
            logits = torch.as_tensor(candidate_rank_scores, dtype=torch.float32, device=score.device)
            expected_shape = (packed.num_prg_tokens, packed.max_candidates)
            if logits.shape != expected_shape:
                raise ValueError(f"candidate_rank_scores mismatch: got={tuple(logits.shape)} expected={expected_shape}")
            score = score + float(logit_weight) * _masked_zscore(logits, valid_main)
    if candidate_rank_scores is not None and packed.max_candidates > 0:
        expected_shape = (packed.num_prg_tokens, packed.max_candidates)
        if tuple(torch.as_tensor(candidate_rank_scores).shape) != expected_shape:
            raise ValueError(
                f"candidate_rank_scores mismatch: got={tuple(torch.as_tensor(candidate_rank_scores).shape)} "
                f"expected={expected_shape}"
            )
    return score.masked_fill(~valid_main, -1.0e9)


def decode_candidate_actions_to_type0(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    ue_action_class: torch.Tensor,
    prg_choice: torch.Tensor,
    *,
    safe_fill_max_per_cell: int = 0,
    safe_fill_min_quality_db: float = -2.0,
    safe_fill_max_quality_gap_db: float = 8.0,
    safe_fill_max_prg_risk: float = 0.70,
    safe_fill_min_backlog_bytes: float = 1.0,
    safe_fill_max_ue_tb_err: float = 0.45,
    demand_cap_slack_bytes: float = 3000.0,
    marginal_headroom_slack_bytes: float = -1.0,
    headroom_rank_mode: str = "stable",
    candidate_rank_scores: Optional[torch.Tensor] = None,
) -> DecodedType0Action:
    meta = graph.snapshot.static_meta
    device = packed.ue_ids.device
    ue_class = torch.as_tensor(ue_action_class, dtype=torch.long, device=device).reshape(-1)
    prg_choice = torch.as_tensor(prg_choice, dtype=torch.long, device=device).reshape(-1)
    if ue_class.shape != (meta.n_sched_ue,):
        raise ValueError(
            f"ue_action_class mismatch: got={tuple(ue_class.shape)} expected={(meta.n_sched_ue,)}"
        )
    if prg_choice.shape != (packed.num_prg_tokens,):
        raise ValueError(
            f"prg_choice mismatch: got={tuple(prg_choice.shape)} expected={(packed.num_prg_tokens,)}"
        )
    rank_mode = str(headroom_rank_mode).lower()
    if rank_mode not in ("stable", "utility", "logit", "phy"):
        raise ValueError(f"headroom_rank_mode must be stable, utility, logit, or phy, got {headroom_rank_mode!r}")

    action_mask_ue = _default_action_mask_ue(graph, device)
    layout = build_type0_slot_layout(
        meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).unsqueeze(0),
        n_sched_ue=int(meta.n_sched_ue),
        action_mask_ue=action_mask_ue.unsqueeze(0),
    )
    slot_to_cell = layout.slot_to_cell.squeeze(0)
    slot_valid_mask = layout.slot_valid_mask.squeeze(0).to(dtype=torch.bool)
    candidate_valid, selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(graph, packed, ue_class)

    action_ue_select = torch.where(
        (ue_class >= 0) & (ue_class < meta.n_ue),
        ue_class.to(torch.int32),
        torch.full((meta.n_sched_ue,), -1, dtype=torch.int32, device=device),
    )
    action_ue_select = torch.where(
        selected_slot_mask,
        action_ue_select,
        torch.full_like(action_ue_select, -1),
    )

    action_prg_alloc = torch.full((meta.n_prg * meta.n_cell,), -1, dtype=torch.int16, device=device)
    chosen_ue_per_prg = torch.full((packed.num_prg_tokens,), -1, dtype=torch.long, device=device)
    prg_chosen_mask = torch.zeros((packed.num_prg_tokens,), dtype=torch.bool, device=device)
    safe_fill_count = 0
    if packed.max_candidates <= 0 or packed.num_prg_tokens == 0:
        return DecodedType0Action(
            action_ue_select=action_ue_select,
            action_prg_alloc=action_prg_alloc,
            chosen_ue_per_prg=chosen_ue_per_prg,
            prg_chosen_mask=prg_chosen_mask,
            final_blank_count=int((~prg_chosen_mask).sum().item()),
        )

    owner_cell = meta.prg_owner_cell.to(device=device, dtype=torch.long)
    local_prg = meta.prg_local_index.to(device=device, dtype=torch.long)
    safe_ue_ids = packed.ue_ids.clamp(min=0, max=max(0, int(meta.n_ue) - 1))
    pair_slot_lookup = torch.full((int(meta.n_cell), int(meta.n_ue)), -1, dtype=torch.long, device=device)
    slot_ids = torch.arange(int(meta.n_sched_ue), device=device, dtype=torch.long)
    selected_slot_ids = action_ue_select.to(torch.long)
    slot_assignable = (
        selected_slot_mask
        & slot_valid_mask
        & (selected_slot_ids >= 0)
        & (selected_slot_ids < int(meta.n_ue))
    )
    if bool(slot_assignable.any().item()):
        pair_slot_lookup[slot_to_cell[slot_assignable], selected_slot_ids[slot_assignable]] = slot_ids[slot_assignable]

    candidate_valid_main = candidate_valid[:, : packed.max_candidates]
    if int(safe_fill_max_per_cell) > 0:
        prg_choice, safe_fill_count = guarded_safe_fill_candidate_choices(
            graph,
            packed,
            candidate_valid,
            prg_choice,
            max_per_cell=int(safe_fill_max_per_cell),
            min_quality_db=float(safe_fill_min_quality_db),
            max_quality_gap_db=float(safe_fill_max_quality_gap_db),
            max_prg_risk=float(safe_fill_max_prg_risk),
            min_backlog_bytes=float(safe_fill_min_backlog_bytes),
            max_ue_tb_err=float(safe_fill_max_ue_tb_err),
        )
    ordered_score = None
    decode_score = None
    if rank_mode in ("utility", "logit", "phy"):
        decode_score = _candidate_decode_headroom_scores(
            graph,
            packed,
            candidate_valid,
            mode=rank_mode,
            candidate_rank_scores=candidate_rank_scores,
        )
    if decode_score is None:
        choice_order = _candidate_priority_order(prg_choice, packed.max_candidates)
    else:
        choice_order = torch.argsort(decode_score, dim=-1, descending=True, stable=True)
    ordered_valid = candidate_valid_main.gather(1, choice_order)
    ordered_ue = safe_ue_ids.gather(1, choice_order)
    if decode_score is not None:
        ordered_score = decode_score.gather(1, choice_order)
    active = (prg_choice >= 0) & (prg_choice < packed.blank_class_index)
    policy_blank_count = int((prg_choice == packed.blank_class_index).sum().item())
    no_valid_candidate_count = 0
    demand_cap_count = 0
    no_slot_count = 0
    other_drop_count = 0
    reason_code = torch.zeros((packed.num_prg_tokens,), dtype=torch.int8, device=device)
    ue_prg_cap = _estimate_ue_prg_demand_cap(
        graph,
        slack_bytes=float(demand_cap_slack_bytes),
    ).to(device=device, dtype=torch.long)
    if float(marginal_headroom_slack_bytes) >= 0.0:
        ue_prg_soft_cap = _estimate_ue_prg_demand_cap(
            graph,
            slack_bytes=float(marginal_headroom_slack_bytes),
        ).to(device=device, dtype=torch.long)
        min_soft_cap = torch.where(
            ue_prg_cap > 0,
            torch.ones_like(ue_prg_cap),
            torch.zeros_like(ue_prg_cap),
        )
        ue_prg_soft_cap = torch.minimum(torch.maximum(ue_prg_soft_cap, min_soft_cap), ue_prg_cap)
    else:
        ue_prg_soft_cap = ue_prg_cap
    ue_prg_assigned = torch.zeros_like(ue_prg_cap)
    assigned = torch.zeros((packed.num_prg_tokens,), dtype=torch.bool, device=device)
    slot_prg_load = torch.zeros((int(meta.n_sched_ue),), dtype=torch.long, device=device)

    for headroom_phase in ("first", "soft", "excess"):
        for rank_idx in range(packed.max_candidates):
            proposal_mask = (~assigned) & active & ordered_valid[:, rank_idx]
            if headroom_phase == "first":
                proposal_mask = proposal_mask & (ue_prg_assigned.gather(0, ordered_ue[:, rank_idx]) <= 0)
            elif headroom_phase == "soft":
                proposal_mask = proposal_mask & (
                    ue_prg_assigned.gather(0, ordered_ue[:, rank_idx]) < ue_prg_soft_cap.gather(0, ordered_ue[:, rank_idx])
                )
            if not bool(proposal_mask.any().item()):
                continue
            proposal_idx = torch.nonzero(proposal_mask, as_tuple=False).flatten()
            proposal_ue = ordered_ue[proposal_idx, rank_idx]
            proposal_cell = owner_cell.gather(0, proposal_idx)
            if ordered_score is not None:
                ue_rank = _stable_group_rank_by_score(proposal_ue, ordered_score[proposal_idx, rank_idx])
            else:
                ue_rank = _stable_group_rank(proposal_ue)
            if headroom_phase == "first":
                headroom = torch.where(
                    ue_prg_cap > ue_prg_assigned,
                    torch.ones_like(ue_prg_cap),
                    torch.zeros_like(ue_prg_cap),
                )
            elif headroom_phase == "soft":
                headroom = torch.clamp_min(ue_prg_soft_cap - ue_prg_assigned, 0)
            else:
                headroom = torch.clamp_min(ue_prg_cap - ue_prg_assigned, 0)
            accepted = ue_rank < headroom.gather(0, proposal_ue)
            if bool((~accepted).any().item()):
                rejected_idx = proposal_idx[~accepted]
                update_mask = reason_code[rejected_idx] == 0
                if headroom_phase == "excess" and bool(update_mask.any().item()):
                    reason_code[rejected_idx[update_mask]] = 3
            if not bool(accepted.any().item()):
                continue
            proposal_idx = proposal_idx[accepted]
            proposal_ue = proposal_ue[accepted]
            proposal_cell = proposal_cell[accepted]

            existing_slot = pair_slot_lookup[proposal_cell, proposal_ue]
            new_pair_mask = existing_slot < 0
            if bool(new_pair_mask.any().item()):
                new_ue = proposal_ue[new_pair_mask]
                new_cell = proposal_cell[new_pair_mask]
                pair_id = new_cell * int(meta.n_ue) + new_ue
                first_pair = _stable_group_rank(pair_id) == 0
                if bool(first_pair.any().item()):
                    _assign_owner_cell_slots(
                        new_cells=new_cell[first_pair],
                        new_ues=new_ue[first_pair],
                        slot_to_cell=slot_to_cell,
                        slot_valid_mask=slot_valid_mask,
                        action_ue_select=action_ue_select,
                        pair_slot_lookup=pair_slot_lookup,
                        slot_prg_load=slot_prg_load,
                    )

            proposal_slot = pair_slot_lookup[proposal_cell, proposal_ue]
            has_slot = proposal_slot >= 0
            if bool((~has_slot).any().item()):
                no_slot_idx = proposal_idx[~has_slot]
                update_mask = reason_code[no_slot_idx] == 0
                if bool(update_mask.any().item()):
                    reason_code[no_slot_idx[update_mask]] = 4
            if not bool(has_slot.any().item()):
                continue
            proposal_idx = proposal_idx[has_slot]
            proposal_ue = proposal_ue[has_slot]
            proposal_cell = proposal_cell[has_slot]
            proposal_slot = proposal_slot[has_slot]

            linear_prg = local_prg.gather(0, proposal_idx) * int(meta.n_cell) + proposal_cell
            action_prg_alloc[linear_prg] = proposal_slot.to(torch.int16)
            chosen_ue_per_prg[proposal_idx] = proposal_ue
            prg_chosen_mask[proposal_idx] = True
            assigned[proposal_idx] = True
            slot_prg_load.scatter_add_(0, proposal_slot, torch.ones_like(proposal_slot))
            increment = torch.zeros_like(ue_prg_assigned)
            increment.scatter_add_(0, proposal_ue, torch.ones_like(proposal_ue))
            ue_prg_assigned = ue_prg_assigned + increment

    unassigned_active = active & (~assigned)
    if bool(unassigned_active.any().item()):
        has_any_valid = ordered_valid.any(dim=-1)
        no_valid_mask = unassigned_active & (~has_any_valid)
        no_valid_candidate_count = int(no_valid_mask.sum().item())
        reason_code[no_valid_mask] = 2
        demand_cap_count = int((unassigned_active & (reason_code == 3)).sum().item())
        no_slot_count = int((unassigned_active & (reason_code == 4)).sum().item())
        other_mask = unassigned_active & (reason_code == 0)
        other_drop_count = int(other_mask.sum().item())
    missed_safe_opportunity_count = count_missed_safe_candidate_opportunities(
        graph,
        packed,
        candidate_valid,
        prg_chosen_mask,
        ue_prg_assigned,
        ue_prg_cap,
        min_quality_db=float(safe_fill_min_quality_db),
        max_quality_gap_db=float(safe_fill_max_quality_gap_db),
        max_prg_risk=float(safe_fill_max_prg_risk),
        min_backlog_bytes=float(safe_fill_min_backlog_bytes),
        max_ue_tb_err=float(safe_fill_max_ue_tb_err),
    )

    return _finalize_decoded_type0_action(
        graph,
        action_ue_select,
        action_prg_alloc,
        safe_fill_count=int(safe_fill_count),
        post_decode_policy_blank_count=int(policy_blank_count),
        post_decode_no_valid_candidate_count=int(no_valid_candidate_count),
        post_decode_demand_cap_count=int(demand_cap_count),
        post_decode_no_slot_count=int(no_slot_count),
        post_decode_other_drop_count=int(other_drop_count),
        missed_safe_opportunity_count=int(missed_safe_opportunity_count),
    )


def decode_prg_choices_to_type0(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    choice_idx: torch.Tensor,
    *,
    action_mask_ue: torch.Tensor,
    n_sched_ue: int,
) -> DecodedType0Action:
    meta = graph.snapshot.static_meta
    device = packed.ue_ids.device
    action_mask_cell_ue = meta.action_mask_cell_ue.to(device=device, dtype=torch.bool)
    action_mask_prg_cell = meta.action_mask_prg_cell.to(device=device, dtype=torch.bool)
    action_mask_ue = torch.as_tensor(action_mask_ue, dtype=torch.bool, device=device)
    if action_mask_ue.shape != (meta.n_ue,):
        raise ValueError(
            f"action_mask_ue mismatch: got={tuple(action_mask_ue.shape)} expected={(meta.n_ue,)}"
        )

    layout = build_type0_slot_layout(
        action_mask_cell_ue.unsqueeze(0),
        n_sched_ue=int(n_sched_ue),
        action_mask_ue=action_mask_ue.unsqueeze(0),
    )
    slot_valid_mask = layout.slot_valid_mask.squeeze(0)
    slot_to_cell = layout.slot_to_cell.squeeze(0)
    cell_slot_start = layout.cell_slot_start.squeeze(0)
    slot_counts = layout.slot_counts.squeeze(0)

    action_ue_select = torch.full((int(n_sched_ue),), -1, dtype=torch.int32, device=device)
    action_prg_alloc = torch.full((meta.n_prg * meta.n_cell,), -1, dtype=torch.int16, device=device)
    chosen_ue_per_prg = torch.full((graph.snapshot.n_prg_tokens,), -1, dtype=torch.long, device=device)
    prg_chosen_mask = torch.zeros((graph.snapshot.n_prg_tokens,), dtype=torch.bool, device=device)

    if packed.max_candidates <= 0 or graph.snapshot.n_prg_tokens == 0:
        return DecodedType0Action(
            action_ue_select=action_ue_select,
            action_prg_alloc=action_prg_alloc,
            chosen_ue_per_prg=chosen_ue_per_prg,
            prg_chosen_mask=prg_chosen_mask,
            final_blank_count=int((~prg_chosen_mask).sum().item()),
        )

    owner_cell = meta.prg_owner_cell.to(device=device, dtype=torch.long)
    local_prg = meta.prg_local_index.to(device=device, dtype=torch.long)
    prg_available = action_mask_prg_cell.reshape(-1)
    safe_ue_ids = packed.ue_ids.clamp(min=0, max=max(0, int(meta.n_ue) - 1))
    owner_cell_ue_mask = action_mask_cell_ue.index_select(0, owner_cell).gather(1, safe_ue_ids)
    ue_enabled = action_mask_ue.gather(0, safe_ue_ids.reshape(-1)).reshape_as(safe_ue_ids)
    candidate_valid = packed.mask & owner_cell_ue_mask & ue_enabled & prg_available.unsqueeze(-1)
    choice_order = _candidate_priority_order(choice_idx.to(torch.long), packed.max_candidates)
    ordered_valid = candidate_valid.gather(1, choice_order)
    ordered_ue = safe_ue_ids.gather(1, choice_order)
    remaining_cap = _estimate_ue_prg_demand_cap(graph).to(device=device, dtype=torch.long)
    remaining_slots = slot_counts.to(device=device, dtype=torch.long).clone()
    next_slot = cell_slot_start.to(device=device, dtype=torch.long).clone()
    pair_slot_lookup = torch.full((int(meta.n_cell), int(meta.n_ue)), -1, dtype=torch.long, device=device)
    assigned = torch.zeros((graph.snapshot.n_prg_tokens,), dtype=torch.bool, device=device)

    for rank_idx in range(packed.max_candidates):
        proposal_mask = (~assigned) & prg_available & ordered_valid[:, rank_idx]
        if not bool(proposal_mask.any().item()):
            continue
        proposal_idx = torch.nonzero(proposal_mask, as_tuple=False).flatten()
        proposal_ue = ordered_ue[proposal_idx, rank_idx]
        proposal_cell = owner_cell.gather(0, proposal_idx)
        ue_rank = _stable_group_rank(proposal_ue)
        cap_keep = ue_rank < remaining_cap.gather(0, proposal_ue)
        if not bool(cap_keep.any().item()):
            continue
        proposal_idx = proposal_idx[cap_keep]
        proposal_ue = proposal_ue[cap_keep]
        proposal_cell = proposal_cell[cap_keep]

        existing_slot = pair_slot_lookup[proposal_cell, proposal_ue]
        new_pair_mask = existing_slot < 0
        if bool(new_pair_mask.any().item()):
            new_idx = proposal_idx[new_pair_mask]
            new_ue = proposal_ue[new_pair_mask]
            new_cell = proposal_cell[new_pair_mask]
            pair_id = new_cell * int(meta.n_ue) + new_ue
            first_pair = _stable_group_rank(pair_id) == 0
            if bool(first_pair.any().item()):
                first_ue = new_ue[first_pair]
                first_cell = new_cell[first_pair]
                cell_rank = _stable_group_rank(first_cell)
                slot_keep = cell_rank < remaining_slots.gather(0, first_cell)
                if bool(slot_keep.any().item()):
                    kept_ue = first_ue[slot_keep]
                    kept_cell = first_cell[slot_keep]
                    kept_rank = cell_rank[slot_keep]
                    assigned_slot = next_slot.gather(0, kept_cell) + kept_rank
                    pair_slot_lookup[kept_cell, kept_ue] = assigned_slot
                    action_ue_select[assigned_slot] = kept_ue.to(torch.int32)
                    used_slots = torch.bincount(kept_cell, minlength=int(meta.n_cell))
                    next_slot = next_slot + used_slots
                    remaining_slots = torch.clamp_min(remaining_slots - used_slots, 0)

        slot_id = pair_slot_lookup[proposal_cell, proposal_ue]
        accepted = slot_id >= 0
        if not bool(accepted.any().item()):
            continue
        proposal_idx = proposal_idx[accepted]
        proposal_ue = proposal_ue[accepted]
        proposal_cell = proposal_cell[accepted]
        slot_id = slot_id[accepted]
        linear_prg = local_prg.gather(0, proposal_idx) * int(meta.n_cell) + proposal_cell
        action_prg_alloc[linear_prg] = slot_id.to(torch.int16)
        chosen_ue_per_prg[proposal_idx] = proposal_ue
        prg_chosen_mask[proposal_idx] = True
        assigned[proposal_idx] = True
        decrement = torch.zeros_like(remaining_cap)
        decrement.scatter_add_(0, proposal_ue, torch.ones_like(proposal_ue))
        remaining_cap = torch.clamp_min(remaining_cap - decrement, 0)

    return _finalize_decoded_type0_action(graph, action_ue_select, action_prg_alloc)


def build_default_joint_slot_targets(graph: SparseEntityGraph) -> torch.Tensor:
    meta = graph.snapshot.static_meta
    action_mask_ue = meta.action_mask_ue
    if action_mask_ue is None:
        action_mask_ue = torch.ones((meta.n_ue,), dtype=torch.bool)
    return build_type0_all_ue_action(
        meta.action_mask_cell_ue,
        n_sched_ue=meta.n_sched_ue,
        action_mask_ue=action_mask_ue,
    ).squeeze(0)


def joint_targets_from_type0_actions(
    *,
    action_ue_select: torch.Tensor,
    action_prg_alloc: torch.Tensor,
    n_ue: int,
    n_sched_ue: int,
    n_cell: int,
    n_prg: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ue = torch.as_tensor(action_ue_select, dtype=torch.long).reshape(-1)
    prg = torch.as_tensor(action_prg_alloc, dtype=torch.long).reshape(-1)
    if ue.shape != (n_sched_ue,):
        raise ValueError(
            f"action_ue_select mismatch: got={tuple(ue.shape)} expected={(n_sched_ue,)}"
        )
    if prg.shape != (n_prg * n_cell,):
        raise ValueError(
            f"action_prg_alloc mismatch: got={tuple(prg.shape)} expected={(n_prg * n_cell,)}"
        )
    ue_target = torch.where(ue >= 0, ue, torch.full_like(ue, n_ue))
    ue_target = torch.where(
        (ue_target >= 0) & (ue_target <= n_ue),
        ue_target,
        torch.full_like(ue_target, n_ue),
    )
    prg_target = prg.view(n_prg, n_cell).transpose(0, 1).contiguous()
    prg_target = torch.where(prg_target >= 0, prg_target, torch.full_like(prg_target, n_sched_ue))
    prg_target = torch.where(
        (prg_target >= 0) & (prg_target <= n_sched_ue),
        prg_target,
        torch.full_like(prg_target, n_sched_ue),
    )
    return ue_target, prg_target


def joint_candidate_targets_from_type0_actions(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    *,
    action_ue_select: torch.Tensor,
    action_prg_alloc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    meta = graph.snapshot.static_meta
    ue_target, _unused_slot_prg_target = joint_targets_from_type0_actions(
        action_ue_select=action_ue_select,
        action_prg_alloc=action_prg_alloc,
        n_ue=meta.n_ue,
        n_sched_ue=meta.n_sched_ue,
        n_cell=meta.n_cell,
        n_prg=meta.n_prg,
    )
    prg_target = candidate_prg_targets_from_type0_actions(
        graph,
        packed,
        action_ue_select=action_ue_select,
        action_prg_alloc=action_prg_alloc,
    )
    return ue_target, prg_target


def decode_joint_actions_to_type0(
    graph: SparseEntityGraph,
    ue_class: torch.Tensor,
    prg_class: torch.Tensor,
) -> DecodedType0Action:
    meta = graph.snapshot.static_meta
    device = graph.snapshot.ue_features.values.device
    ue = torch.as_tensor(ue_class, dtype=torch.long, device=device).reshape(-1)
    prg = torch.as_tensor(prg_class, dtype=torch.long, device=device)
    if ue.shape != (meta.n_sched_ue,):
        raise ValueError(
            f"ue_class mismatch: got={tuple(ue.shape)} expected={(meta.n_sched_ue,)}"
        )
    if prg.shape != (meta.n_cell, meta.n_prg):
        raise ValueError(
            f"prg_class mismatch: got={tuple(prg.shape)} expected={(meta.n_cell, meta.n_prg)}"
        )

    action_ue_select = torch.where(
        (ue >= 0) & (ue < meta.n_ue),
        ue.to(torch.int32),
        torch.full((meta.n_sched_ue,), -1, dtype=torch.int32, device=device),
    )
    action_mask_ue = meta.action_mask_ue
    layout = build_type0_slot_layout(
        meta.action_mask_cell_ue.unsqueeze(0),
        n_sched_ue=int(meta.n_sched_ue),
        action_mask_ue=action_mask_ue.unsqueeze(0) if action_mask_ue is not None else None,
    )
    slot_valid_mask = layout.slot_valid_mask.squeeze(0)
    slot_to_cell = layout.slot_to_cell.squeeze(0)
    cell_slot_start = layout.cell_slot_start.squeeze(0)
    slot_counts = layout.slot_counts.squeeze(0)
    selected_slot_mask = build_slot_selection_mask(
        ue.unsqueeze(0),
        (
            action_mask_ue.unsqueeze(0)
            if action_mask_ue is not None
            else torch.ones((1, meta.n_ue), dtype=torch.bool, device=ue.device)
        ),
        n_cell=int(meta.n_cell),
        action_mask_cell_ue=meta.action_mask_cell_ue.unsqueeze(0),
    ).squeeze(0)
    action_ue_select = torch.where(
        selected_slot_mask,
        action_ue_select,
        torch.full_like(action_ue_select, -1),
    )
    prg_valid = (prg >= 0) & (prg < meta.n_sched_ue)
    action_prg_alloc = torch.where(
        prg_valid,
        prg.to(torch.int16),
        torch.full((meta.n_cell, meta.n_prg), -1, dtype=torch.int16, device=device),
    ).transpose(0, 1).reshape(-1)

    chosen_ue_per_prg = torch.full((meta.n_cell * meta.n_prg,), -1, dtype=torch.long, device=device)
    prg_chosen_mask = prg_valid.reshape(-1)
    ue_prg_demand_cap = _estimate_ue_prg_demand_cap(graph)
    demand_active_mask = demand_active_ue_mask_from_features(
        graph.snapshot.ue_features.values,
        action_mask_ue=meta.action_mask_ue,
    )
    ue_backlog_bytes = torch.clamp_min(graph.snapshot.ue_features.values[:, _UE_BUFFER_IDX], 0.0)
    ue_prg_assigned = torch.zeros((meta.n_ue,), dtype=torch.long, device=device)
    cap_drop_count = 0
    fallback_success_count = 0
    if prg_valid.any():
        safe_slots = prg.clamp(min=0, max=max(0, meta.n_sched_ue - 1))
        chosen_ues = action_ue_select.to(torch.long).gather(0, safe_slots.reshape(-1))
        chosen_ues = torch.where(prg_chosen_mask, chosen_ues, torch.full_like(chosen_ues, -1))
        capped_chosen_ues = torch.full_like(chosen_ues, -1)
        for prg_token in range(meta.n_cell * meta.n_prg):
            owner_cell = int(meta.prg_owner_cell[prg_token].item())
            local_prg = int(meta.prg_local_index[prg_token].item())
            ue_idx = int(chosen_ues[prg_token].item())
            if ue_idx < 0 or ue_idx >= meta.n_ue:
                prg_chosen_mask[prg_token] = False
                action_prg_alloc[local_prg * meta.n_cell + owner_cell] = -1
                continue
            if ue_prg_assigned[ue_idx] >= ue_prg_demand_cap[ue_idx]:
                old_slot = int(action_prg_alloc[local_prg * meta.n_cell + owner_cell].item())
                fallback_slot = _pick_joint_fallback_slot(
                    owner_cell=owner_cell,
                    action_ue_select=action_ue_select,
                    cell_slot_start=cell_slot_start,
                    slot_counts=slot_counts,
                    slot_valid_mask=slot_valid_mask,
                    ue_prg_assigned=ue_prg_assigned,
                    ue_prg_demand_cap=ue_prg_demand_cap,
                    demand_active_mask=demand_active_mask,
                    ue_backlog_bytes=ue_backlog_bytes,
                    exclude_slot=old_slot,
                )
                if fallback_slot < 0:
                    cap_drop_count += 1
                    action_prg_alloc[local_prg * meta.n_cell + owner_cell] = -1
                    prg_chosen_mask[prg_token] = False
                    continue
                else:
                    action_prg_alloc[local_prg * meta.n_cell + owner_cell] = int(fallback_slot)
                    ue_idx = int(action_ue_select[fallback_slot].item())
                    if ue_idx < 0 or ue_idx >= meta.n_ue:
                        prg_chosen_mask[prg_token] = False
                        action_prg_alloc[local_prg * meta.n_cell + owner_cell] = -1
                        continue
                    fallback_success_count += 1
            capped_chosen_ues[prg_token] = ue_idx
            ue_prg_assigned[ue_idx] += 1
        chosen_ue_per_prg = capped_chosen_ues

    decoded = _finalize_decoded_type0_action(graph, action_ue_select, action_prg_alloc)
    return DecodedType0Action(
        action_ue_select=decoded.action_ue_select,
        action_prg_alloc=decoded.action_prg_alloc,
        chosen_ue_per_prg=decoded.chosen_ue_per_prg,
        prg_chosen_mask=decoded.prg_chosen_mask,
        cap_drop_count=int(cap_drop_count),
        fallback_success_count=int(fallback_success_count),
        final_blank_count=int((~decoded.prg_chosen_mask).sum().item()),
    )
