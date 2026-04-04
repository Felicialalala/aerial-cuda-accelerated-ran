#!/usr/bin/env python3
"""Action mask utilities for Stage-B BC/PPO training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from training.gnnrl.slot_layout import build_schedulable_cell_ue_mask, build_type0_slot_layout


def _slot_to_cell(n_cell: int, n_sched_ue: int, device: torch.device) -> torch.Tensor:
    slots_per_cell = n_sched_ue // n_cell
    return torch.arange(n_sched_ue, device=device, dtype=torch.long) // slots_per_cell


def _ue_to_cell(n_cell: int, n_active_ue: int, device: torch.device) -> torch.Tensor:
    ue_per_cell = n_active_ue // n_cell
    return torch.arange(n_active_ue, device=device, dtype=torch.long) // ue_per_cell


def build_slot_ue_cell_compat_mask(n_cell: int, n_sched_ue: int, n_active_ue: int, device: torch.device) -> torch.Tensor:
    """Returns [S, U] boolean mask for same-cell slot/UE mapping."""
    s2c = _slot_to_cell(n_cell, n_sched_ue, device)
    u2c = _ue_to_cell(n_cell, n_active_ue, device)
    return s2c.unsqueeze(1) == u2c.unsqueeze(0)


def build_type0_slot_valid_mask(
    action_mask_cell_ue: torch.Tensor,
    n_sched_ue: int,
    action_mask_ue: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns [B, S] boolean mask for actually populated Type-0 slots."""
    return build_type0_slot_layout(
        action_mask_cell_ue,
        n_sched_ue=n_sched_ue,
        action_mask_ue=action_mask_ue,
    ).slot_valid_mask


def build_slot_selection_mask(
    ue_class: torch.Tensor,
    action_mask_ue: torch.Tensor,
    n_cell: int,
    action_mask_cell_ue: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns [B, S] bool mask for slots that survive hard UE selection semantics:
    valid slot, live UE, cell-compatible UE, and first occurrence of that UE.
    """
    if ue_class.dim() == 1:
        ue_class = ue_class.unsqueeze(0)
    if action_mask_ue.dim() == 1:
        action_mask_ue = action_mask_ue.unsqueeze(0)
    if ue_class.dim() != 2 or action_mask_ue.dim() != 2:
        raise ValueError(
            f"expected ue_class [B,S] and action_mask_ue [B,U], got {tuple(ue_class.shape)} and {tuple(action_mask_ue.shape)}"
        )

    bsz, n_sched_ue = ue_class.shape
    if action_mask_ue.shape[0] != bsz:
        raise ValueError(
            f"batch mismatch between ue_class and action_mask_ue: {tuple(ue_class.shape)} vs {tuple(action_mask_ue.shape)}"
        )
    n_active_ue = action_mask_ue.shape[1]

    if action_mask_cell_ue is not None:
        layout = build_type0_slot_layout(
            action_mask_cell_ue,
            n_sched_ue=n_sched_ue,
            action_mask_ue=action_mask_ue,
        )
        slot_cells = layout.slot_to_cell
        slot_valid = layout.slot_valid_mask.bool()
        compat = build_schedulable_cell_ue_mask(action_mask_cell_ue, action_mask_ue=action_mask_ue).gather(
            1,
            slot_cells.unsqueeze(-1).expand(-1, -1, n_active_ue),
        )
    else:
        slot_cells = _slot_to_cell(n_cell, n_sched_ue, ue_class.device).unsqueeze(0).expand(bsz, -1)
        slot_valid = torch.ones((bsz, n_sched_ue), dtype=torch.bool, device=ue_class.device)
        compat = build_slot_ue_cell_compat_mask(n_cell, n_sched_ue, n_active_ue, ue_class.device)
        compat = compat.unsqueeze(0).expand(bsz, -1, -1)
        compat = compat & action_mask_ue.bool().unsqueeze(1)

    ue_idx = ue_class.clamp(min=0, max=max(0, n_active_ue - 1))
    in_range = (ue_class >= 0) & (ue_class < n_active_ue)
    compat_selected = compat.gather(-1, ue_idx.unsqueeze(-1)).squeeze(-1)
    slot_selected = torch.zeros((bsz, n_sched_ue), dtype=torch.bool, device=ue_class.device)
    used_ue = torch.zeros((bsz, n_active_ue), dtype=torch.bool, device=ue_class.device)

    for slot_idx in range(n_sched_ue):
        ue_now = ue_idx[:, slot_idx]
        keep = in_range[:, slot_idx] & slot_valid[:, slot_idx] & compat_selected[:, slot_idx]
        keep = keep & (~used_ue.gather(1, ue_now.unsqueeze(-1)).squeeze(-1))
        slot_selected[:, slot_idx] = keep
        used_ue.scatter_(1, ue_now.unsqueeze(-1), used_ue.gather(1, ue_now.unsqueeze(-1)) | keep.unsqueeze(-1))

    return slot_selected


def apply_ue_action_mask(
    ue_logits: torch.Tensor,
    action_mask_ue: torch.Tensor,
    n_cell: int,
    action_mask_cell_ue: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        ue_logits: [B, S, U+1]
        action_mask_ue: [B, U]
        action_mask_cell_ue: optional [B, C, U], preferred over index-based cell assumption
    Returns:
        masked_logits: [B, S, U+1]
        valid_mask: [B, S, U+1]
    """
    bsz, n_sched_ue, ue_plus_no = ue_logits.shape
    n_active_ue = ue_plus_no - 1

    if action_mask_cell_ue is not None:
        if action_mask_cell_ue.dim() != 3:
            raise ValueError(f"action_mask_cell_ue must be [B,C,U], got shape={tuple(action_mask_cell_ue.shape)}")
        if action_mask_cell_ue.shape[0] != bsz or action_mask_cell_ue.shape[1] != n_cell or action_mask_cell_ue.shape[2] != n_active_ue:
            raise ValueError(
                f"action_mask_cell_ue shape mismatch: got={tuple(action_mask_cell_ue.shape)} "
                f"expected=({bsz},{n_cell},{n_active_ue})"
            )
        layout = build_type0_slot_layout(
            action_mask_cell_ue,
            n_sched_ue=n_sched_ue,
            action_mask_ue=action_mask_ue,
        )
        slot_cells = layout.slot_to_cell
        compat = build_schedulable_cell_ue_mask(action_mask_cell_ue, action_mask_ue=action_mask_ue).gather(
            1,
            slot_cells.unsqueeze(-1).expand(-1, -1, n_active_ue),
        )
        slot_valid = layout.slot_valid_mask.bool().unsqueeze(-1)
    else:
        compat = build_slot_ue_cell_compat_mask(n_cell, n_sched_ue, n_active_ue, ue_logits.device)
        compat = compat.unsqueeze(0).expand(bsz, -1, -1)
        slot_valid = torch.ones((bsz, n_sched_ue, 1), dtype=torch.bool, device=ue_logits.device)

    ue_alive = action_mask_ue.bool().unsqueeze(1).expand(-1, n_sched_ue, -1)
    valid_main = compat & ue_alive & slot_valid
    valid_no = torch.ones((bsz, n_sched_ue, 1), dtype=torch.bool, device=ue_logits.device)
    valid = torch.cat([valid_main, valid_no], dim=-1)

    masked = ue_logits.masked_fill(~valid, -1.0e9)
    return masked, valid


def apply_prg_action_mask(
    prg_logits: torch.Tensor,
    action_mask_prg_cell: torch.Tensor,
    n_cell: int,
    slot_valid_mask: Optional[torch.Tensor] = None,
    action_mask_cell_ue: Optional[torch.Tensor] = None,
    action_mask_ue: Optional[torch.Tensor] = None,
    selected_slot_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        prg_logits: [B, C, P, S+1]
        action_mask_prg_cell: [B, C, P]
        slot_valid_mask: optional [B, S], used to mask out empty Type-0 slots
    Returns:
        masked_logits: [B, C, P, S+1]
        valid_mask: [B, C, P, S+1]
    """
    bsz, cdim, n_prg, sched_plus_no = prg_logits.shape
    if cdim != n_cell:
        raise ValueError(f"prg_logits cell dim mismatch: got={cdim}, expected={n_cell}")
    n_sched_ue = sched_plus_no - 1

    if slot_valid_mask is not None:
        if slot_valid_mask.dim() == 1:
            slot_valid_mask = slot_valid_mask.unsqueeze(0)
        if slot_valid_mask.dim() != 2 or slot_valid_mask.shape[0] != bsz or slot_valid_mask.shape[1] != n_sched_ue:
            raise ValueError(
                f"slot_valid_mask shape mismatch: got={tuple(slot_valid_mask.shape)} "
                f"expected=({bsz},{n_sched_ue})"
            )

    derived_slot_valid_mask = None
    if action_mask_cell_ue is not None:
        if action_mask_cell_ue.dim() != 3:
            raise ValueError(f"action_mask_cell_ue must be [B,C,U], got shape={tuple(action_mask_cell_ue.shape)}")
        layout = build_type0_slot_layout(
            action_mask_cell_ue,
            n_sched_ue=n_sched_ue,
            action_mask_ue=action_mask_ue,
        )
        slot_cells = layout.slot_to_cell
        derived_slot_valid_mask = layout.slot_valid_mask
    else:
        slot_cells = _slot_to_cell(n_cell, n_sched_ue, prg_logits.device).unsqueeze(0).expand(bsz, -1)

    if derived_slot_valid_mask is not None:
        if slot_valid_mask is None:
            slot_valid_mask = derived_slot_valid_mask
        else:
            slot_valid_mask = slot_valid_mask.bool() & derived_slot_valid_mask.bool()
    if selected_slot_mask is not None:
        if selected_slot_mask.dim() == 1:
            selected_slot_mask = selected_slot_mask.unsqueeze(0)
        if selected_slot_mask.dim() != 2 or selected_slot_mask.shape[0] != bsz or selected_slot_mask.shape[1] != n_sched_ue:
            raise ValueError(
                f"selected_slot_mask shape mismatch: got={tuple(selected_slot_mask.shape)} "
                f"expected=({bsz},{n_sched_ue})"
            )
        if slot_valid_mask is None:
            slot_valid_mask = selected_slot_mask.bool()
        else:
            slot_valid_mask = slot_valid_mask.bool() & selected_slot_mask.bool()

    slot_local = torch.arange(n_cell, device=prg_logits.device, dtype=torch.long).view(1, n_cell, 1, 1) == slot_cells.view(
        bsz, 1, 1, n_sched_ue
    )

    prg_ok = action_mask_prg_cell.bool().unsqueeze(-1)
    valid_main = slot_local & prg_ok
    if slot_valid_mask is not None:
        valid_main = valid_main & slot_valid_mask.bool().unsqueeze(1).unsqueeze(1)
    valid_no = torch.ones((bsz, n_cell, n_prg, 1), dtype=torch.bool, device=prg_logits.device)
    valid = torch.cat([valid_main, valid_no], dim=-1)

    masked = prg_logits.masked_fill(~valid, -1.0e9)
    return masked, valid


def sanitize_targets(target: torch.Tensor, valid_mask: torch.Tensor, ignore_index: int = -100) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace illegal target classes with ignore_index."""
    out = target.clone().to(torch.long)
    n_class = valid_mask.shape[-1]
    bad = (out < 0) | (out >= n_class)

    probe = out.clamp(min=0, max=n_class - 1).unsqueeze(-1)
    legal = valid_mask.gather(-1, probe).squeeze(-1)
    bad = bad | (~legal)

    out[bad] = ignore_index
    return out, bad


def classification_accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> float:
    pred = logits.argmax(dim=-1)
    valid = target != ignore_index
    denom = int(valid.sum().item())
    if denom == 0:
        return 0.0
    correct = ((pred == target) & valid).sum().item()
    return float(correct) / float(denom)


def predicted_legal_ratio(logits: torch.Tensor, valid_mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    legal = valid_mask.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
    return float(legal.float().mean().item())
