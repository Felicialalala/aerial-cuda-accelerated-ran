#!/usr/bin/env python3
"""Lightweight GNN policy network for Stage-B offline BC training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.gnnrl.action_modes import ACTION_MODE_JOINT, normalize_action_mode
from training.gnnrl.slot_layout import build_schedulable_cell_ue_mask, build_type0_slot_layout


@dataclass(frozen=True)
class ModelConfig:
    n_cell: int
    n_active_ue: int
    n_sched_ue: int
    n_prg: int
    cell_feat_dim: int
    ue_feat_dim: int
    edge_feat_dim: int
    prg_feat_dim: int = 0
    hidden_dim: int = 128
    num_cell_msg_layers: int = 2
    max_prg: int = 512
    action_mode: str = ACTION_MODE_JOINT
    max_slot_local_pos: int = 0


class CellMessageLayer(nn.Module):
    """Single message-passing block on directed cell graph."""

    def __init__(self, hidden_dim: int, edge_feat_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, cell_x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # cell_x: [B, C, H], edge_index: [B, E, 2], edge_attr: [B, E, Fe]
        bsz, n_cell, hid = cell_x.shape

        src = edge_index[..., 0].clamp(min=0, max=n_cell - 1)
        dst = edge_index[..., 1].clamp(min=0, max=n_cell - 1)

        src_x = cell_x.gather(1, src.unsqueeze(-1).expand(-1, -1, hid))
        msg = self.msg(torch.cat([src_x, edge_attr], dim=-1))

        agg = torch.zeros((bsz, n_cell, hid), dtype=cell_x.dtype, device=cell_x.device)
        dst_idx = dst.unsqueeze(-1).expand(-1, -1, hid)
        agg.scatter_add_(1, dst_idx, msg)

        cnt = torch.zeros((bsz, n_cell, 1), dtype=cell_x.dtype, device=cell_x.device)
        one = torch.ones((bsz, dst.shape[1], 1), dtype=cell_x.dtype, device=cell_x.device)
        cnt.scatter_add_(1, dst.unsqueeze(-1), one)
        agg = agg / cnt.clamp_min(1.0)

        out = self.upd(torch.cat([cell_x, agg], dim=-1))
        return self.norm(cell_x + out)


class StageBGnnPolicy(nn.Module):
    """Two-head policy: UE selection logits and PRG allocation logits."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.action_mode = normalize_action_mode(cfg.action_mode)

        if cfg.n_prg > cfg.max_prg:
            raise ValueError(f"n_prg={cfg.n_prg} exceeds max_prg={cfg.max_prg}")

        hid = cfg.hidden_dim
        local_pos_size = max(1, cfg.max_slot_local_pos or self.slots_per_cell)

        self.cell_in = nn.Sequential(
            nn.Linear(cfg.cell_feat_dim, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
        )
        self.ue_in = nn.Sequential(
            nn.Linear(cfg.ue_feat_dim, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
        )
        self.ue_cell_fuse = nn.Sequential(
            nn.Linear(2 * hid, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
        )

        self.cell_layers = nn.ModuleList([CellMessageLayer(hid, cfg.edge_feat_dim) for _ in range(cfg.num_cell_msg_layers)])

        self.slot_query = nn.Linear(hid, hid)
        self.ue_key = nn.Linear(hid, hid)
        self.slot_key = nn.Linear(hid, hid)
        self.ue_null_head = nn.Linear(hid, 1)
        # Break symmetry among scheduler slots inside the same cell.
        self.slot_pos_emb = nn.Embedding(cfg.n_sched_ue, hid)
        self.slot_local_pos_emb = nn.Embedding(local_pos_size, hid)
        self.slot_fuse = nn.Sequential(
            nn.Linear(3 * hid, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
        )
        self.slot_ue_fuse = nn.Sequential(
            nn.Linear(2 * hid, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
        )

        self.prg_in = (
            nn.Sequential(
                nn.Linear(cfg.prg_feat_dim, hid),
                nn.ReLU(),
                nn.LayerNorm(hid),
            )
            if cfg.prg_feat_dim > 0
            else None
        )
        self.prg_pos_emb = nn.Embedding(cfg.max_prg, hid)
        self.prg_ctx = nn.Sequential(
            nn.Linear(3 * hid, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
        )
        self.prg_query = nn.Linear(hid, hid)
        self.prg_slot_key = nn.Linear(hid, hid)
        self.prg_null_head = nn.Linear(hid, 1)

        self.register_buffer(
            "default_slot_to_cell",
            (torch.arange(cfg.n_sched_ue, dtype=torch.long) // self.slots_per_cell).clamp(max=max(0, cfg.n_cell - 1)),
            persistent=False,
        )
        self.register_buffer(
            "slot_global_idx",
            torch.arange(cfg.n_sched_ue, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "default_slot_local_idx",
            torch.arange(cfg.n_sched_ue, dtype=torch.long) % self.slots_per_cell,
            persistent=False,
        )
        if cfg.n_active_ue % cfg.n_cell == 0:
            ue_to_cell = torch.arange(cfg.n_active_ue, dtype=torch.long) // (cfg.n_active_ue // cfg.n_cell)
        else:
            ue_to_cell = torch.empty((0,), dtype=torch.long)
        self.register_buffer("ue_to_cell", ue_to_cell, persistent=False)

    @property
    def slots_per_cell(self) -> int:
        return max(1, self.cfg.n_sched_ue // max(1, self.cfg.n_cell))

    def forward(
        self,
        obs_cell_features: torch.Tensor,
        obs_ue_features: torch.Tensor,
        obs_edge_index: torch.Tensor,
        obs_edge_attr: torch.Tensor,
        obs_prg_features: torch.Tensor | None = None,
        action_mask_ue: torch.Tensor | None = None,
        action_mask_cell_ue: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            ue_logits: [B, S, U+1]  (last class is NO_UE)
            prg_logits: [B, C, P, S+1] (last class is NO_TX)
        """
        cfg = self.cfg
        hid = cfg.hidden_dim
        bsz = obs_cell_features.shape[0]

        cell_h = self.cell_in(obs_cell_features)
        for layer in self.cell_layers:
            cell_h = layer(cell_h, obs_edge_index, obs_edge_attr)

        ue_h = self.ue_in(obs_ue_features)
        if action_mask_cell_ue is not None:
            ue_cell_weights = action_mask_cell_ue.to(dtype=cell_h.dtype).transpose(1, 2)
            ue_cell_weights = ue_cell_weights / ue_cell_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            ue_cell = torch.einsum("buc,bch->buh", ue_cell_weights, cell_h)
        else:
            if self.ue_to_cell.numel() != cfg.n_active_ue:
                raise ValueError("action_mask_cell_ue is required when n_active_ue is not evenly divisible by n_cell")
            ue_cell = cell_h[:, self.ue_to_cell, :]
        ue_h = self.ue_cell_fuse(torch.cat([ue_h, ue_cell], dim=-1))

        sched_cell_ue_mask = None
        if action_mask_cell_ue is not None:
            sched_cell_ue_mask = build_schedulable_cell_ue_mask(action_mask_cell_ue, action_mask_ue=action_mask_ue)
            layout = build_type0_slot_layout(
                action_mask_cell_ue,
                n_sched_ue=cfg.n_sched_ue,
                action_mask_ue=action_mask_ue,
            )
            slot_to_cell = layout.slot_to_cell
            slot_local_idx = layout.slot_local_idx
            slot_valid_mask = layout.slot_valid_mask
        else:
            slot_to_cell = self.default_slot_to_cell.unsqueeze(0).expand(bsz, -1)
            slot_local_idx = self.default_slot_local_idx.unsqueeze(0).expand(bsz, -1)
            slot_valid_mask = torch.ones((bsz, cfg.n_sched_ue), dtype=torch.bool, device=obs_cell_features.device)

        slot_cell_ctx = cell_h.gather(1, slot_to_cell.unsqueeze(-1).expand(-1, -1, hid))
        slot_pos = self.slot_pos_emb(self.slot_global_idx).unsqueeze(0).expand(bsz, -1, -1)
        slot_local_idx = slot_local_idx.clamp(max=self.slot_local_pos_emb.num_embeddings - 1)
        slot_local_pos = self.slot_local_pos_emb(slot_local_idx)
        slot_ctx = self.slot_fuse(torch.cat([slot_cell_ctx, slot_pos, slot_local_pos], dim=-1))

        slot_q = self.slot_query(slot_ctx)
        ue_k = self.ue_key(ue_h)
        ue_logits_main = torch.einsum("bsh,buh->bsu", slot_q, ue_k) / math.sqrt(float(hid))
        ue_no = self.ue_null_head(slot_ctx)
        ue_logits = torch.cat([ue_logits_main, ue_no], dim=-1)

        slot_ue_logits = ue_logits_main
        if sched_cell_ue_mask is not None:
            slot_cell_ue_mask = sched_cell_ue_mask.gather(
                1,
                slot_to_cell.unsqueeze(-1).expand(-1, -1, cfg.n_active_ue),
            )
            slot_cell_ue_mask = slot_cell_ue_mask & slot_valid_mask.unsqueeze(-1)
            slot_ue_logits = slot_ue_logits.masked_fill(~slot_cell_ue_mask, -1.0e9)
        slot_ue_prob = torch.softmax(slot_ue_logits, dim=-1)
        if slot_valid_mask is not None:
            slot_ue_prob = torch.where(slot_valid_mask.unsqueeze(-1), slot_ue_prob, torch.zeros_like(slot_ue_prob))
        slot_ue_ctx_weights = slot_ue_prob
        if sched_cell_ue_mask is not None:
            hard_slot_ue = F.one_hot(slot_ue_logits.argmax(dim=-1), num_classes=cfg.n_active_ue).to(slot_ue_prob.dtype)
            slot_ue_ctx_weights = hard_slot_ue + (slot_ue_prob - slot_ue_prob.detach())
            slot_ue_ctx_weights = torch.where(
                slot_valid_mask.unsqueeze(-1),
                slot_ue_ctx_weights,
                torch.zeros_like(slot_ue_ctx_weights),
            )
        slot_ue_soft_ctx = torch.einsum("bsu,buh->bsh", slot_ue_ctx_weights, ue_h)
        slot_prg_ctx = self.slot_ue_fuse(torch.cat([slot_ctx, slot_ue_soft_ctx], dim=-1))

        prg_ids = torch.arange(cfg.n_prg, dtype=torch.long, device=obs_cell_features.device)
        prg_emb = self.prg_pos_emb(prg_ids).unsqueeze(0).expand(bsz, -1, -1)
        if self.prg_in is not None and obs_prg_features is not None:
            prg_feat_h = self.prg_in(obs_prg_features)
        else:
            prg_feat_h = torch.zeros((bsz, cfg.n_cell, cfg.n_prg, hid), dtype=cell_h.dtype, device=cell_h.device)

        cell_expand = cell_h.unsqueeze(2).expand(-1, -1, cfg.n_prg, -1)
        prg_expand = prg_emb.unsqueeze(1).expand(-1, cfg.n_cell, -1, -1)
        prg_ctx = self.prg_ctx(torch.cat([cell_expand, prg_expand, prg_feat_h], dim=-1))

        prg_q = self.prg_query(prg_ctx)
        slot_k = self.prg_slot_key(slot_prg_ctx)
        prg_logits_main = torch.einsum("bcph,bsh->bcps", prg_q, slot_k) / math.sqrt(float(hid))
        prg_no = self.prg_null_head(prg_ctx)
        prg_logits = torch.cat([prg_logits_main, prg_no], dim=-1)

        return {
            "ue_logits": ue_logits,
            "prg_logits": prg_logits,
        }

    def model_config_dict(self) -> Dict[str, int]:
        return {
            "n_cell": self.cfg.n_cell,
            "n_active_ue": self.cfg.n_active_ue,
            "n_sched_ue": self.cfg.n_sched_ue,
            "n_prg": self.cfg.n_prg,
            "cell_feat_dim": self.cfg.cell_feat_dim,
            "ue_feat_dim": self.cfg.ue_feat_dim,
            "edge_feat_dim": self.cfg.edge_feat_dim,
            "prg_feat_dim": self.cfg.prg_feat_dim,
            "hidden_dim": self.cfg.hidden_dim,
            "num_cell_msg_layers": self.cfg.num_cell_msg_layers,
            "max_prg": self.cfg.max_prg,
            "action_mode": self.action_mode,
            "max_slot_local_pos": self.cfg.max_slot_local_pos,
        }


def build_model_from_config(cfg_dict: Dict[str, int]) -> StageBGnnPolicy:
    cfg_dict = dict(cfg_dict)
    cfg_dict.setdefault("prg_feat_dim", 0)
    return StageBGnnPolicy(ModelConfig(**cfg_dict))
