#!/usr/bin/env python3
"""Lightweight GNN policy network for Stage-B offline BC training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_cell: int
    n_active_ue: int
    n_sched_ue: int
    n_prg: int
    cell_feat_dim: int
    ue_feat_dim: int
    edge_feat_dim: int
    hidden_dim: int = 128
    num_cell_msg_layers: int = 2
    max_prg: int = 512


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

        if cfg.n_sched_ue % cfg.n_cell != 0:
            raise ValueError(f"n_sched_ue={cfg.n_sched_ue} must be divisible by n_cell={cfg.n_cell}")
        if cfg.n_active_ue % cfg.n_cell != 0:
            raise ValueError(f"n_active_ue={cfg.n_active_ue} must be divisible by n_cell={cfg.n_cell}")
        if cfg.n_prg > cfg.max_prg:
            raise ValueError(f"n_prg={cfg.n_prg} exceeds max_prg={cfg.max_prg}")

        hid = cfg.hidden_dim

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
        self.slot_local_pos_emb = nn.Embedding(self.slots_per_cell, hid)
        self.slot_fuse = nn.Sequential(
            nn.Linear(3 * hid, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
        )

        self.prg_pos_emb = nn.Embedding(cfg.max_prg, hid)
        self.prg_ctx = nn.Sequential(
            nn.Linear(2 * hid, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
        )
        self.prg_query = nn.Linear(hid, hid)
        self.prg_null_head = nn.Linear(hid, 1)

        self.register_buffer(
            "slot_to_cell",
            torch.arange(cfg.n_sched_ue, dtype=torch.long) // (cfg.n_sched_ue // cfg.n_cell),
            persistent=False,
        )
        self.register_buffer(
            "slot_global_idx",
            torch.arange(cfg.n_sched_ue, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "slot_local_idx",
            torch.arange(cfg.n_sched_ue, dtype=torch.long) % (cfg.n_sched_ue // cfg.n_cell),
            persistent=False,
        )
        self.register_buffer(
            "ue_to_cell",
            torch.arange(cfg.n_active_ue, dtype=torch.long) // (cfg.n_active_ue // cfg.n_cell),
            persistent=False,
        )

    @property
    def slots_per_cell(self) -> int:
        return self.cfg.n_sched_ue // self.cfg.n_cell

    def forward(
        self,
        obs_cell_features: torch.Tensor,
        obs_ue_features: torch.Tensor,
        obs_edge_index: torch.Tensor,
        obs_edge_attr: torch.Tensor,
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
        ue_cell = cell_h[:, self.ue_to_cell, :]
        ue_h = self.ue_cell_fuse(torch.cat([ue_h, ue_cell], dim=-1))

        slot_cell_ctx = cell_h[:, self.slot_to_cell, :]
        slot_pos = self.slot_pos_emb(self.slot_global_idx).unsqueeze(0).expand(bsz, -1, -1)
        slot_local_pos = self.slot_local_pos_emb(self.slot_local_idx).unsqueeze(0).expand(bsz, -1, -1)
        slot_ctx = self.slot_fuse(torch.cat([slot_cell_ctx, slot_pos, slot_local_pos], dim=-1))

        slot_q = self.slot_query(slot_ctx)
        ue_k = self.ue_key(ue_h)
        ue_logits_main = torch.einsum("bsh,buh->bsu", slot_q, ue_k) / math.sqrt(float(hid))
        ue_no = self.ue_null_head(slot_ctx)
        ue_logits = torch.cat([ue_logits_main, ue_no], dim=-1)

        prg_ids = torch.arange(cfg.n_prg, dtype=torch.long, device=obs_cell_features.device)
        prg_emb = self.prg_pos_emb(prg_ids).unsqueeze(0).expand(bsz, -1, -1)

        cell_expand = cell_h.unsqueeze(2).expand(-1, -1, cfg.n_prg, -1)
        prg_expand = prg_emb.unsqueeze(1).expand(-1, cfg.n_cell, -1, -1)
        prg_ctx = self.prg_ctx(torch.cat([cell_expand, prg_expand], dim=-1))

        prg_q = self.prg_query(prg_ctx)
        slot_k = self.slot_key(slot_ctx)
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
            "hidden_dim": self.cfg.hidden_dim,
            "num_cell_msg_layers": self.cfg.num_cell_msg_layers,
            "max_prg": self.cfg.max_prg,
        }


def build_model_from_config(cfg_dict: Dict[str, int]) -> StageBGnnPolicy:
    return StageBGnnPolicy(ModelConfig(**cfg_dict))
