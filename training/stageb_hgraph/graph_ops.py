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
    final_blank_count: int = 0


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
        final_blank_count=int((~prg_chosen_mask).sum().item()),
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


def decode_candidate_actions_to_type0(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    ue_action_class: torch.Tensor,
    prg_choice: torch.Tensor,
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
    choice_order = _candidate_priority_order(prg_choice, packed.max_candidates)
    ordered_valid = candidate_valid_main.gather(1, choice_order)
    ordered_ue = safe_ue_ids.gather(1, choice_order)
    active = (prg_choice >= 0) & (prg_choice < packed.blank_class_index)
    ue_prg_cap = _estimate_ue_prg_demand_cap(graph).to(device=device, dtype=torch.long)
    ue_prg_assigned = torch.zeros_like(ue_prg_cap)
    assigned = torch.zeros((packed.num_prg_tokens,), dtype=torch.bool, device=device)
    slot_prg_load = torch.zeros((int(meta.n_sched_ue),), dtype=torch.long, device=device)

    for prefer_first_service in (True, False):
        for rank_idx in range(packed.max_candidates):
            proposal_mask = (~assigned) & active & ordered_valid[:, rank_idx]
            if prefer_first_service:
                proposal_mask = proposal_mask & (ue_prg_assigned.gather(0, ordered_ue[:, rank_idx]) <= 0)
            if not bool(proposal_mask.any().item()):
                continue
            proposal_idx = torch.nonzero(proposal_mask, as_tuple=False).flatten()
            proposal_ue = ordered_ue[proposal_idx, rank_idx]
            proposal_cell = owner_cell.gather(0, proposal_idx)
            ue_rank = _stable_group_rank(proposal_ue)
            headroom = torch.clamp_min(ue_prg_cap - ue_prg_assigned, 0)
            accepted = ue_rank < headroom.gather(0, proposal_ue)
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

    return _finalize_decoded_type0_action(graph, action_ue_select, action_prg_alloc)


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
