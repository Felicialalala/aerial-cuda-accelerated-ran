#!/usr/bin/env python3
"""Action mask utilities for Stage-B BC/PPO training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


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
        slot_cells = _slot_to_cell(n_cell, n_sched_ue, ue_logits.device)
        compat = action_mask_cell_ue.bool()[:, slot_cells, :]
    else:
        compat = build_slot_ue_cell_compat_mask(n_cell, n_sched_ue, n_active_ue, ue_logits.device)
        compat = compat.unsqueeze(0).expand(bsz, -1, -1)

    ue_alive = action_mask_ue.bool().unsqueeze(1).expand(-1, n_sched_ue, -1)
    valid_main = compat & ue_alive
    valid_no = torch.ones((bsz, n_sched_ue, 1), dtype=torch.bool, device=ue_logits.device)
    valid = torch.cat([valid_main, valid_no], dim=-1)

    masked = ue_logits.masked_fill(~valid, -1.0e9)
    return masked, valid


def apply_prg_action_mask(prg_logits: torch.Tensor, action_mask_prg_cell: torch.Tensor, n_cell: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        prg_logits: [B, C, P, S+1]
        action_mask_prg_cell: [B, C, P]
    Returns:
        masked_logits: [B, C, P, S+1]
        valid_mask: [B, C, P, S+1]
    """
    bsz, cdim, n_prg, sched_plus_no = prg_logits.shape
    if cdim != n_cell:
        raise ValueError(f"prg_logits cell dim mismatch: got={cdim}, expected={n_cell}")
    n_sched_ue = sched_plus_no - 1

    slot_cells = _slot_to_cell(n_cell, n_sched_ue, prg_logits.device)
    slot_local = torch.arange(n_cell, device=prg_logits.device, dtype=torch.long).unsqueeze(1) == slot_cells.unsqueeze(0)
    slot_local = slot_local.unsqueeze(0).unsqueeze(2).expand(bsz, -1, n_prg, -1)

    prg_ok = action_mask_prg_cell.bool().unsqueeze(-1)
    valid_main = slot_local & prg_ok
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
