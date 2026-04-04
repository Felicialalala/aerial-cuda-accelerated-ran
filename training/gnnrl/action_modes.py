#!/usr/bin/env python3
"""Action-mode helpers for Stage-B GNN+RL training/inference."""

from __future__ import annotations

from typing import Iterable

import torch

from training.gnnrl.slot_layout import build_schedulable_cell_ue_mask, build_type0_slot_layout


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
    action_mask_ue: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build deterministic Type-0 slot fill over currently schedulable UEs:
    each cell's live scheduler slots are filled by its associated/live UEs in
    ascending UE index order, with remaining slots kept as -1.

    Args:
        action_mask_cell_ue: [B, C, U] or [C, U] association mask.
        n_sched_ue: total scheduler slots.
        action_mask_ue: optional [B, U] or [U] live/schedulable UE mask.
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

    layout = build_type0_slot_layout(
        action_mask_cell_ue,
        n_sched_ue=n_sched_ue,
        action_mask_ue=action_mask_ue,
    )
    out = torch.full((bsz, n_sched_ue), -1, dtype=torch.long, device=action_mask_cell_ue.device)
    sched_mask = build_schedulable_cell_ue_mask(action_mask_cell_ue, action_mask_ue=action_mask_ue)

    for b_idx in range(bsz):
        for c_idx in range(n_cell):
            ue_ids = torch.nonzero(sched_mask[b_idx, c_idx], as_tuple=False).flatten()
            take = min(int(layout.slot_counts[b_idx, c_idx].item()), int(ue_ids.numel()))
            if take <= 0:
                continue
            start = int(layout.cell_slot_start[b_idx, c_idx].item())
            out[b_idx, start : start + take] = ue_ids[:take]
    return out


def action_mode_choices() -> Iterable[str]:
    return ACTION_MODES
