#!/usr/bin/env python3
"""Type-0 scheduler slot layout helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Type0SlotLayout:
    slot_to_cell: torch.Tensor
    slot_local_idx: torch.Tensor
    slot_valid_mask: torch.Tensor
    slot_counts: torch.Tensor
    cell_slot_start: torch.Tensor


def build_schedulable_cell_ue_mask(
    action_mask_cell_ue: torch.Tensor,
    action_mask_ue: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns [B,C,U] bool mask for currently schedulable cell/UE pairs."""
    if action_mask_cell_ue.dim() == 2:
        action_mask_cell_ue = action_mask_cell_ue.unsqueeze(0)
    if action_mask_cell_ue.dim() != 3:
        raise ValueError(
            "action_mask_cell_ue must have shape [B,C,U] or [C,U], "
            f"got shape={tuple(action_mask_cell_ue.shape)}"
        )

    sched_mask = action_mask_cell_ue.to(dtype=torch.bool)
    if action_mask_ue is None:
        return sched_mask

    if action_mask_ue.dim() == 1:
        action_mask_ue = action_mask_ue.unsqueeze(0)
    if action_mask_ue.dim() != 2:
        raise ValueError(
            "action_mask_ue must have shape [B,U] or [U], "
            f"got shape={tuple(action_mask_ue.shape)}"
        )
    if action_mask_ue.shape[0] != sched_mask.shape[0] or action_mask_ue.shape[1] != sched_mask.shape[2]:
        raise ValueError(
            f"action_mask_ue shape mismatch: got={tuple(action_mask_ue.shape)} "
            f"expected=({sched_mask.shape[0]},{sched_mask.shape[2]})"
        )

    return sched_mask & action_mask_ue.to(dtype=torch.bool).unsqueeze(1)


def build_type0_slot_layout(
    action_mask_cell_ue: torch.Tensor,
    n_sched_ue: int,
    action_mask_ue: torch.Tensor | None = None,
) -> Type0SlotLayout:
    """
    Build a per-sample contiguous slot layout for Type-0 scheduling.

    Slots are assigned to cells in cell-index order. Each cell receives as many
    slots as it has currently schedulable UEs (`action_mask_cell_ue &
    action_mask_ue` when provided), clipped by the remaining global slot
    budget. Any trailing slots beyond that live-UE count remain invalid.
    """
    if n_sched_ue < 0:
        raise ValueError(f"n_sched_ue must be non-negative, got {n_sched_ue}")

    sched_mask = build_schedulable_cell_ue_mask(action_mask_cell_ue, action_mask_ue=action_mask_ue)
    device = sched_mask.device
    raw_counts = sched_mask.sum(dim=-1, dtype=torch.long)
    if raw_counts.shape[1] == 0:
        raise ValueError("n_cell must be positive")

    clipped_ends = torch.cumsum(raw_counts, dim=-1).clamp(max=n_sched_ue)
    zero = torch.zeros((raw_counts.shape[0], 1), dtype=torch.long, device=device)
    cell_slot_start = torch.cat([zero, clipped_ends[:, :-1]], dim=-1)
    slot_counts = clipped_ends - cell_slot_start

    slot_ids = torch.arange(n_sched_ue, device=device, dtype=torch.long).view(1, 1, -1)
    slot_membership = (slot_ids >= cell_slot_start.unsqueeze(-1)) & (
        slot_ids < (cell_slot_start + slot_counts).unsqueeze(-1)
    )
    slot_valid_mask = slot_membership.any(dim=1)
    slot_to_cell = slot_membership.to(dtype=torch.long).argmax(dim=1)
    slot_start = cell_slot_start.gather(1, slot_to_cell)
    slot_local_idx = torch.where(
        slot_valid_mask,
        slot_ids.squeeze(1) - slot_start,
        torch.zeros_like(slot_to_cell),
    )

    return Type0SlotLayout(
        slot_to_cell=slot_to_cell,
        slot_local_idx=slot_local_idx,
        slot_valid_mask=slot_valid_mask,
        slot_counts=slot_counts,
        cell_slot_start=cell_slot_start,
    )
