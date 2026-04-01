#!/usr/bin/env python3
"""Action-mode helpers for Stage-B GNN+RL training/inference."""

from __future__ import annotations

from typing import Iterable

import torch


ACTION_MODE_JOINT = "joint"
ACTION_MODE_PRG_ONLY_TYPE0 = "prg_only_type0"
ACTION_MODES = (
    ACTION_MODE_JOINT,
    ACTION_MODE_PRG_ONLY_TYPE0,
)


def normalize_action_mode(value: str | None) -> str:
    mode = (value or ACTION_MODE_JOINT).strip().lower()
    if mode not in ACTION_MODES:
        valid = ", ".join(ACTION_MODES)
        raise ValueError(f"unsupported action_mode={value!r}, expected one of: {valid}")
    return mode


def is_prg_only_type0(action_mode: str | None) -> bool:
    return normalize_action_mode(action_mode) == ACTION_MODE_PRG_ONLY_TYPE0


def build_type0_all_ue_action(
    action_mask_cell_ue: torch.Tensor,
    n_sched_ue: int,
) -> torch.Tensor:
    """
    Build deterministic Type-0 slot fill that mirrors the native baseline:
    each cell's scheduler slots are filled by its associated active UEs in
    ascending UE index order, with remaining slots kept as -1.

    Args:
        action_mask_cell_ue: [B, C, U] or [C, U] association mask.
        n_sched_ue: total scheduler slots.
    Returns:
        [B, S] tensor of UE ids, with -1 for empty slots.
    """
    if action_mask_cell_ue.dim() == 2:
        action_mask_cell_ue = action_mask_cell_ue.unsqueeze(0)
    if action_mask_cell_ue.dim() != 3:
        raise ValueError(
            "action_mask_cell_ue must have shape [B,C,U] or [C,U], "
            f"got shape={tuple(action_mask_cell_ue.shape)}"
        )

    bsz, n_cell, _n_active_ue = action_mask_cell_ue.shape
    if n_cell <= 0:
        raise ValueError("n_cell must be positive")
    if n_sched_ue % n_cell != 0:
        raise ValueError(f"n_sched_ue={n_sched_ue} must be divisible by n_cell={n_cell}")

    slots_per_cell = n_sched_ue // n_cell
    out = torch.full((bsz, n_sched_ue), -1, dtype=torch.long, device=action_mask_cell_ue.device)
    assoc = action_mask_cell_ue.to(dtype=torch.bool)

    for b_idx in range(bsz):
        for c_idx in range(n_cell):
            ue_ids = torch.nonzero(assoc[b_idx, c_idx], as_tuple=False).flatten()
            take = min(slots_per_cell, int(ue_ids.numel()))
            if take <= 0:
                continue
            start = c_idx * slots_per_cell
            out[b_idx, start : start + take] = ue_ids[:take]
    return out


def action_mode_choices() -> Iterable[str]:
    return ACTION_MODES
