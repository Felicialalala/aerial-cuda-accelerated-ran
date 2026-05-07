#!/usr/bin/env python3
"""Export a fixed-size Stage-B HGraph actor to the C++ TRT runtime ONNX ABI.

This exporter targets the current `GnnRlPolicyRuntime` ABI:

Logits-mode inputs:
  obs_cell_features  [1, 3, 5]
  obs_ue_features    [1, 36, 12]
  obs_prg_features   [1, 3, 17, 8]
  obs_edge_index     [1, 6, 2]
  obs_edge_attr      [1, 6, 2]
  action_mask_ue     [1, 36]
  action_mask_cell_ue[1, 3, 36]

Logits-mode outputs:
  ue_logits          [1, 36, 37]
  prg_logits         [1, 3, 17, 37]

Action-mode adds `obs_post_eq_sinr [1, 36, 17]` and emits deterministic
`action_ue_select [1, 36]` plus native-order `action_prg_alloc [1, 51]`.
The C++ runtime then only finalizes demand caps/fallback and writes Type-0
native scheduler buffers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.gnnrl.slot_layout import build_schedulable_cell_ue_mask, build_type0_slot_layout
from training.stageb_hgraph.model import StageBHGraphPolicy, build_model_from_config


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "training/stageb_hgraph/runs/"
    "online_ppo_seed42_svd_rawbridge_joint_v33_capSlack3000_simpleExecutor_ablation/"
    "ppo_actor_absolute_best.pt"
)
DEFAULT_OUT = REPO_ROOT / "training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_argmax.onnx"
DEFAULT_ACTION_OUT = REPO_ROOT / "training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx"


def _fixed_cell_edge_index(n_cell: int) -> torch.Tensor:
    pairs: list[tuple[int, int]] = []
    for src in range(n_cell):
        for dst in range(n_cell):
            if src != dst:
                pairs.append((src, dst))
    return torch.tensor(pairs, dtype=torch.long)


def _fixed_cell_prg_index(n_cell: int, n_prg: int) -> torch.Tensor:
    src: list[int] = []
    dst: list[int] = []
    for c_idx in range(n_cell):
        for prg_idx in range(n_prg):
            src.append(c_idx)
            dst.append(c_idx * n_prg + prg_idx)
    return torch.tensor([src, dst], dtype=torch.long)


def _mean_aggregation_matrix(dst_index: torch.Tensor, n_dst: int) -> torch.Tensor:
    out = torch.zeros((n_dst, int(dst_index.numel())), dtype=torch.float32)
    counts = torch.zeros((n_dst,), dtype=torch.float32)
    for edge_idx, dst in enumerate(dst_index.tolist()):
        counts[int(dst)] += 1.0
    for edge_idx, dst in enumerate(dst_index.tolist()):
        out[int(dst), edge_idx] = 1.0 / max(float(counts[int(dst)]), 1.0)
    return out


def _default_action_mask_cell_ue(n_cell: int, n_ue: int) -> torch.Tensor:
    mask = torch.zeros((n_cell, n_ue), dtype=torch.float32)
    ue_per_cell = n_ue // n_cell
    for c_idx in range(n_cell):
        start = c_idx * ue_per_cell
        end = n_ue if c_idx == n_cell - 1 else (c_idx + 1) * ue_per_cell
        mask[c_idx, start:end] = 1.0
    return mask


class FixedHGraphOnnxWrapper(nn.Module):
    """Tensor-only HGraph wrapper for the fixed 3cell/36UE/17PRG deployment path."""

    def __init__(
        self,
        actor: StageBHGraphPolicy,
        *,
        n_cell: int = 3,
        n_ue: int = 36,
        n_prg: int = 17,
        n_sched_ue: int = 36,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.n_cell = int(n_cell)
        self.n_ue = int(n_ue)
        self.n_prg = int(n_prg)
        self.n_sched_ue = int(n_sched_ue)

        cell_edges = _fixed_cell_edge_index(self.n_cell)
        self.register_buffer("cell_edge_src", cell_edges[:, 0], persistent=False)
        self.register_buffer("cell_edge_dst", cell_edges[:, 1], persistent=False)
        self.register_buffer(
            "cell_edge_to_dst_mean",
            _mean_aggregation_matrix(cell_edges[:, 1], self.n_cell),
            persistent=False,
        )
        self.register_buffer("edge_index_cp", _fixed_cell_prg_index(self.n_cell, self.n_prg), persistent=False)
        prg_owner = torch.arange(self.n_cell, dtype=torch.long).repeat_interleave(self.n_prg)
        prg_local = torch.arange(self.n_prg, dtype=torch.long).repeat(self.n_cell)
        self.register_buffer("prg_owner_cell", prg_owner, persistent=False)
        self.register_buffer("prg_local_index", prg_local, persistent=False)

        prg_edge_src: list[int] = []
        prg_edge_dst: list[int] = []
        for src_cell, dst_cell in cell_edges.tolist():
            for prg_idx in range(self.n_prg):
                prg_edge_src.append(int(src_cell) * self.n_prg + prg_idx)
                prg_edge_dst.append(int(dst_cell) * self.n_prg + prg_idx)
        self.register_buffer("prg_edge_src", torch.tensor(prg_edge_src, dtype=torch.long), persistent=False)
        self.register_buffer("prg_edge_dst", torch.tensor(prg_edge_dst, dtype=torch.long), persistent=False)
        self.register_buffer(
            "prg_edge_to_dst_mean",
            _mean_aggregation_matrix(torch.tensor(prg_edge_dst, dtype=torch.long), self.n_cell * self.n_prg),
            persistent=False,
        )

    @staticmethod
    def _relation_dense_masked_no_attr(rel: nn.Module, src_x: torch.Tensor, mask_src_dst: torch.Tensor) -> torch.Tensor:
        weights = mask_src_dst.to(dtype=src_x.dtype)
        src_msg = rel.msg_proj(rel.src_proj(src_x))
        counts = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        return torch.matmul(weights.transpose(0, 1), src_msg) / counts

    def _relation_cell_cell(
        self,
        rel: nn.Module,
        cell_x: torch.Tensor,
        obs_edge_index: torch.Tensor,
        obs_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        del obs_edge_index
        src_idx = self.cell_edge_src.to(device=cell_x.device, dtype=torch.long)
        msg = rel.src_proj(cell_x[src_idx])
        if rel.edge_proj is not None:
            msg = msg + rel.edge_proj(obs_edge_attr[0].to(dtype=cell_x.dtype))
        msg = rel.msg_proj(msg)
        mean_mat = self.cell_edge_to_dst_mean.to(device=cell_x.device, dtype=cell_x.dtype)
        return torch.matmul(mean_mat, msg)

    def _relation_cell_prg(self, rel: nn.Module, cell_x: torch.Tensor) -> torch.Tensor:
        owner = self.prg_owner_cell.to(device=cell_x.device)
        msg = rel.msg_proj(rel.src_proj(cell_x[owner]))
        return msg

    def _relation_prg_prg(self, rel: nn.Module, prg_x: torch.Tensor, obs_edge_index: torch.Tensor) -> torch.Tensor:
        del obs_edge_index
        src_idx = self.prg_edge_src.to(device=prg_x.device, dtype=torch.long)
        msg = rel.msg_proj(rel.src_proj(prg_x[src_idx]))
        mean_mat = self.prg_edge_to_dst_mean.to(device=prg_x.device, dtype=prg_x.dtype)
        return torch.matmul(mean_mat, msg)

    def _relation_ue_prg(
        self,
        rel: nn.Module,
        ue_x: torch.Tensor,
        ue_features: torch.Tensor,
        prg_features: torch.Tensor,
        action_mask_ue: torch.Tensor,
        action_mask_cell_ue: torch.Tensor,
        obs_post_eq_sinr: torch.Tensor | None,
    ) -> torch.Tensor:
        active_ue = action_mask_ue[0].to(dtype=torch.bool)
        demand = (torch.clamp_min(ue_features[:, 0], 0.0) >= 1.0) | (ue_features[:, 6] > 0.5)
        legal_cell_ue = action_mask_cell_ue[0].to(dtype=torch.bool) & (active_ue & demand).unsqueeze(0)

        wb_sinr_db = 10.0 * torch.log10(torch.clamp_min(ue_features[:, 2], 1.0e-9))
        if obs_post_eq_sinr is None:
            quality_db = wb_sinr_db.view(1, 1, self.n_ue).expand(self.n_cell, self.n_prg, self.n_ue)
            masked_quality = torch.where(
                legal_cell_ue.view(self.n_cell, 1, self.n_ue),
                quality_db,
                torch.full(
                    (self.n_cell, self.n_prg, self.n_ue),
                    -1.0e9,
                    dtype=ue_features.dtype,
                    device=ue_features.device,
                ),
            )
        else:
            post_eq_db = 10.0 * torch.log10(torch.clamp_min(obs_post_eq_sinr[0], 1.0e-9))
            quality_db = post_eq_db.transpose(0, 1).view(1, self.n_prg, self.n_ue).expand(
                self.n_cell,
                self.n_prg,
                self.n_ue,
            )
            masked_quality = torch.where(
                legal_cell_ue.view(self.n_cell, 1, self.n_ue),
                quality_db,
                torch.full(
                    (self.n_cell, self.n_prg, self.n_ue),
                    -1.0e9,
                    dtype=ue_features.dtype,
                    device=ue_features.device,
                ),
            )
        best_quality = masked_quality.max(dim=2).values
        best_quality = torch.where(best_quality > -1.0e8, best_quality, torch.zeros_like(best_quality))
        quality_gap = torch.clamp_min(best_quality.unsqueeze(-1) - quality_db, 0.0)

        buffer_norm = torch.log1p(torch.clamp_min(ue_features[:, 0], 0.0))
        hol_norm = ue_features[:, 8] / torch.max(ue_features[:, 8]).clamp_min(1.0)
        ttl_term = torch.where(
            ue_features[:, 9] >= 0.0,
            1.0 / (1.0 + torch.clamp_min(ue_features[:, 9], 0.0)),
            torch.zeros_like(ue_features[:, 9]),
        )
        urgency = 0.45 * buffer_norm + 0.30 * hol_norm + 0.25 * ttl_term
        debt = 0.6 * torch.clamp_min(ue_features[:, 11], 0.0) + 0.4 * (
            1.0 - torch.clamp(ue_features[:, 10], 0.0, 1.0)
        )

        prg_by_cell = prg_features.reshape(self.n_cell, self.n_prg, -1)
        prg_risk = prg_by_cell[:, :, 4] + prg_by_cell[:, :, 5]

        edge_attr = torch.stack(
            [
                quality_db,
                quality_gap,
                urgency.view(1, 1, self.n_ue).expand(self.n_cell, self.n_prg, self.n_ue),
                debt.view(1, 1, self.n_ue).expand(self.n_cell, self.n_prg, self.n_ue),
                prg_risk.unsqueeze(-1).expand(self.n_cell, self.n_prg, self.n_ue),
            ],
            dim=-1,
        )

        msg = rel.src_proj(ue_x).view(1, 1, self.n_ue, -1)
        if rel.edge_proj is not None:
            msg = msg + rel.edge_proj(edge_attr.to(dtype=ue_x.dtype))
        msg = rel.msg_proj(msg)
        weights = legal_cell_ue.to(dtype=ue_x.dtype).view(self.n_cell, 1, self.n_ue, 1)
        summed = (msg * weights).sum(dim=2)
        counts = weights.sum(dim=2).clamp_min(1.0)
        return (summed / counts).reshape(self.n_cell * self.n_prg, -1)

    def _encode(
        self,
        obs_cell_features: torch.Tensor,
        obs_ue_features: torch.Tensor,
        obs_prg_features: torch.Tensor,
        obs_edge_index: torch.Tensor,
        obs_edge_attr: torch.Tensor,
        action_mask_ue: torch.Tensor,
        action_mask_cell_ue: torch.Tensor,
        obs_post_eq_sinr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cell_features = obs_cell_features[0]
        ue_features = obs_ue_features[0]
        prg_features = obs_prg_features[0].reshape(self.n_cell * self.n_prg, -1)

        cell_x = self.actor.cell_residual(self.actor.cell_encoder(cell_features))
        ue_x = self.actor.ue_residual(self.actor.ue_encoder(ue_features))
        prg_x = self.actor.prg_residual(self.actor.prg_encoder(prg_features))

        for layer in self.actor.message_layers:
            agg_cell = self._relation_cell_cell(layer.rel_cc, cell_x, obs_edge_index, obs_edge_attr)
            agg_ue = self._relation_dense_masked_no_attr(
                layer.rel_cu,
                cell_x,
                action_mask_cell_ue[0].to(dtype=torch.bool) & action_mask_ue[0].to(dtype=torch.bool).unsqueeze(0),
            )
            agg_prg_from_cell = self._relation_cell_prg(layer.rel_cp, cell_x)
            agg_prg_from_ue = self._relation_ue_prg(
                layer.rel_up,
                ue_x,
                ue_features,
                prg_features,
                action_mask_ue,
                action_mask_cell_ue,
                obs_post_eq_sinr,
            )
            agg_prg_from_prg = self._relation_prg_prg(layer.rel_pp, prg_x, obs_edge_index)

            cell_x = layer.update_cell(cell_x, agg_cell)
            ue_x = layer.update_ue(ue_x, agg_ue)
            prg_x = layer.update_prg(prg_x, agg_prg_from_cell, agg_prg_from_ue, agg_prg_from_prg)
        return cell_x, ue_x, prg_x

    def _selected_slot_mask(
        self,
        ue_class: torch.Tensor,
        action_mask_ue: torch.Tensor,
        action_mask_cell_ue: torch.Tensor,
        slot_valid_mask: torch.Tensor,
        slot_to_cell: torch.Tensor,
    ) -> torch.Tensor:
        action_mask_ue_b = action_mask_ue.to(dtype=torch.bool)
        action_mask_cell_ue_b = action_mask_cell_ue.to(dtype=torch.bool)
        ue_idx = ue_class.clamp(min=0, max=max(0, self.n_ue - 1))
        in_range = (ue_class >= 0) & (ue_class < self.n_ue)
        compat = action_mask_cell_ue_b.gather(
            1,
            slot_to_cell.unsqueeze(-1).expand(-1, -1, self.n_ue),
        )
        compat = compat & action_mask_ue_b.unsqueeze(1)
        compat_selected = compat.gather(-1, ue_idx.unsqueeze(-1)).squeeze(-1)
        selected = torch.zeros_like(slot_valid_mask, dtype=torch.bool)
        used_ue = torch.zeros((1, self.n_ue), dtype=torch.bool, device=ue_class.device)
        for slot_idx in range(self.n_sched_ue):
            ue_now = ue_idx[:, slot_idx]
            keep = in_range[:, slot_idx] & slot_valid_mask[:, slot_idx] & compat_selected[:, slot_idx]
            keep = keep & (~used_ue.gather(1, ue_now.unsqueeze(-1)).squeeze(-1))
            selected[:, slot_idx] = keep
            used_ue.scatter_(1, ue_now.unsqueeze(-1), used_ue.gather(1, ue_now.unsqueeze(-1)) | keep.unsqueeze(-1))
        return selected

    def _mask_prg_logits(
        self,
        prg_logits: torch.Tensor,
        selected_slot_mask: torch.Tensor,
        slot_valid_mask: torch.Tensor,
        slot_to_cell: torch.Tensor,
    ) -> torch.Tensor:
        slot_ids = torch.arange(self.n_sched_ue, dtype=torch.long, device=prg_logits.device)
        slot_cell_ok = torch.arange(self.n_cell, dtype=torch.long, device=prg_logits.device).view(
            1,
            self.n_cell,
            1,
            1,
        ) == slot_to_cell.view(1, 1, 1, self.n_sched_ue)
        slot_ok = selected_slot_mask.bool() & slot_valid_mask.bool()
        valid_main = slot_cell_ok.expand(1, self.n_cell, self.n_prg, self.n_sched_ue)
        valid_main = valid_main & slot_ok.view(1, 1, 1, self.n_sched_ue)
        valid_no = torch.ones((1, self.n_cell, self.n_prg, 1), dtype=torch.bool, device=prg_logits.device)
        valid = torch.cat([valid_main, valid_no], dim=-1)
        del slot_ids
        return prg_logits.masked_fill(~valid, -1.0e9)

    def _actions_from_logits(
        self,
        ue_logits: torch.Tensor,
        prg_logits: torch.Tensor,
        action_mask_ue: torch.Tensor,
        action_mask_cell_ue: torch.Tensor,
        slot_valid_mask: torch.Tensor,
        slot_to_cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ue_class = ue_logits.argmax(dim=-1)
        selected_slot_mask = self._selected_slot_mask(
            ue_class,
            action_mask_ue,
            action_mask_cell_ue,
            slot_valid_mask,
            slot_to_cell,
        )
        action_ue_select = torch.where(
            selected_slot_mask,
            ue_class.clamp(min=0, max=max(0, self.n_ue - 1)),
            torch.full_like(ue_class, -1),
        )
        prg_logits_masked = self._mask_prg_logits(prg_logits, selected_slot_mask, slot_valid_mask, slot_to_cell)
        prg_class = prg_logits_masked.argmax(dim=-1)
        prg_valid = (prg_class >= 0) & (prg_class < self.n_sched_ue)
        action_prg_by_cell = torch.where(
            prg_valid,
            prg_class.to(dtype=torch.long),
            torch.full((1, self.n_cell, self.n_prg), -1, dtype=torch.long, device=prg_class.device),
        )
        action_prg_native = action_prg_by_cell.squeeze(0).transpose(0, 1).reshape(1, self.n_cell * self.n_prg)
        return action_ue_select.to(dtype=torch.float32), action_prg_native.to(dtype=torch.float32)

    def _action_head(
        self,
        cell_x: torch.Tensor,
        ue_x: torch.Tensor,
        prg_x: torch.Tensor,
        action_mask_ue: torch.Tensor,
        action_mask_cell_ue: torch.Tensor,
        slot_ue_class: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        head = self.actor.action_head
        batch_size = 1
        hid = int(cell_x.shape[-1])
        dtype = cell_x.dtype
        device = cell_x.device

        cell_x_b = cell_x.unsqueeze(0)
        ue_x_b = ue_x.unsqueeze(0)
        prg_x_b = prg_x.unsqueeze(0)
        action_mask_cell_ue_b = action_mask_cell_ue.to(dtype=torch.bool)
        action_mask_ue_b = action_mask_ue.to(dtype=torch.bool)

        layout = build_type0_slot_layout(
            action_mask_cell_ue_b,
            n_sched_ue=self.n_sched_ue,
            action_mask_ue=action_mask_ue_b,
        )
        slot_to_cell = layout.slot_to_cell
        slot_local_idx = layout.slot_local_idx
        slot_valid_mask = layout.slot_valid_mask

        slot_global_idx = torch.arange(self.n_sched_ue, device=device, dtype=torch.long)
        slot_cell_ctx = cell_x_b.gather(1, slot_to_cell.unsqueeze(-1).expand(-1, -1, hid))
        slot_pos = head.slot_pos_emb(slot_global_idx.clamp(max=head.slot_pos_emb.num_embeddings - 1)).unsqueeze(0)
        slot_pos = slot_pos.expand(batch_size, -1, -1)
        slot_local_pos = head.slot_local_pos_emb(
            slot_local_idx.clamp(max=head.slot_local_pos_emb.num_embeddings - 1)
        )
        slot_ctx = head.slot_fuse(torch.cat([slot_cell_ctx, slot_pos, slot_local_pos], dim=-1))

        slot_q = head.slot_query(slot_ctx)
        ue_k = head.ue_key(ue_x_b)
        ue_logits_main = torch.einsum("bsh,buh->bsu", slot_q, ue_k) / math.sqrt(float(hid))

        sched_cell_ue_mask = build_schedulable_cell_ue_mask(
            action_mask_cell_ue_b,
            action_mask_ue=action_mask_ue_b,
        )
        slot_cell_ue_mask = sched_cell_ue_mask.gather(
            1,
            slot_to_cell.unsqueeze(-1).expand(-1, -1, self.n_ue),
        )
        slot_cell_ue_mask = slot_cell_ue_mask & slot_valid_mask.unsqueeze(-1)
        ue_logits_main = ue_logits_main.masked_fill(~slot_cell_ue_mask, -1.0e9)
        ue_no = head.ue_null_head(slot_ctx)
        ue_logits = torch.cat([ue_logits_main, ue_no], dim=-1)

        if slot_ue_class is None:
            slot_prior_ue_class = ue_logits_main.argmax(dim=-1)
            slot_prior_ue_class = torch.where(
                slot_valid_mask,
                slot_prior_ue_class,
                torch.full_like(slot_prior_ue_class, -1),
            )
        else:
            slot_prior_ue_class = slot_ue_class.to(device=device, dtype=torch.long)

        safe_slot_ue = slot_prior_ue_class.clamp(min=0, max=max(0, self.n_ue - 1))
        explicit_valid = (
            slot_valid_mask
            & (slot_prior_ue_class >= 0)
            & (slot_prior_ue_class < self.n_ue)
            & slot_cell_ue_mask.gather(-1, safe_slot_ue.unsqueeze(-1)).squeeze(-1)
        )
        gathered_slot_ue = ue_x_b.gather(1, safe_slot_ue.unsqueeze(-1).expand(-1, -1, hid))
        slot_ue_soft_ctx = torch.where(
            explicit_valid.unsqueeze(-1),
            gathered_slot_ue,
            torch.zeros_like(gathered_slot_ue),
        )
        slot_prg_ctx = head.slot_ue_fuse(torch.cat([slot_ctx, slot_ue_soft_ctx], dim=-1))

        prg_by_cell = prg_x_b.reshape(batch_size, self.n_cell, self.n_prg, hid)
        cell_expand = cell_x_b.unsqueeze(2).expand(-1, -1, self.n_prg, -1)
        prg_local_idx = self.prg_local_index.to(device=device, dtype=torch.long).reshape(1, self.n_cell, self.n_prg)
        prg_pos = head.prg_pos_emb(prg_local_idx.clamp(max=head.prg_pos_emb.num_embeddings - 1))
        prg_ctx = head.prg_ctx(torch.cat([prg_by_cell, cell_expand, prg_pos], dim=-1))
        prg_q = head.prg_query(prg_ctx)
        slot_k = head.prg_slot_key(slot_prg_ctx)
        prg_logits_main = torch.einsum("bcph,bsh->bcps", prg_q, slot_k) / math.sqrt(float(hid))
        prg_no = head.prg_null_head(prg_ctx)
        prg_logits = torch.cat([prg_logits_main, prg_no], dim=-1)
        return ue_logits, prg_logits, slot_valid_mask, slot_to_cell

    def forward(
        self,
        obs_cell_features: torch.Tensor,
        obs_ue_features: torch.Tensor,
        obs_prg_features: torch.Tensor,
        obs_edge_index: torch.Tensor,
        obs_edge_attr: torch.Tensor,
        action_mask_ue: torch.Tensor,
        action_mask_cell_ue: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cell_x, ue_x, prg_x = self._encode(
            obs_cell_features,
            obs_ue_features,
            obs_prg_features,
            obs_edge_index,
            obs_edge_attr,
            action_mask_ue,
            action_mask_cell_ue,
        )
        ue_logits, _, slot_valid_mask, _slot_to_cell = self._action_head(
            cell_x,
            ue_x,
            prg_x,
            action_mask_ue,
            action_mask_cell_ue,
            slot_ue_class=None,
        )
        slot_ue_class = ue_logits[:, :, : self.n_ue].argmax(dim=-1)
        slot_ue_class = torch.where(slot_valid_mask, slot_ue_class, torch.full_like(slot_ue_class, -1))
        _, prg_logits, _slot_valid_mask_2, _slot_to_cell_2 = self._action_head(
            cell_x,
            ue_x,
            prg_x,
            action_mask_ue,
            action_mask_cell_ue,
            slot_ue_class=slot_ue_class,
        )
        edge_index_keepalive = obs_edge_index.to(dtype=ue_logits.dtype).sum() * 0.0
        return ue_logits + edge_index_keepalive, prg_logits + edge_index_keepalive


class FixedHGraphActionOnnxWrapper(FixedHGraphOnnxWrapper):
    """Fixed-shape wrapper that emits final-path deterministic action candidates."""

    def forward(
        self,
        obs_cell_features: torch.Tensor,
        obs_ue_features: torch.Tensor,
        obs_prg_features: torch.Tensor,
        obs_edge_index: torch.Tensor,
        obs_edge_attr: torch.Tensor,
        action_mask_ue: torch.Tensor,
        action_mask_cell_ue: torch.Tensor,
        obs_post_eq_sinr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cell_x, ue_x, prg_x = self._encode(
            obs_cell_features,
            obs_ue_features,
            obs_prg_features,
            obs_edge_index,
            obs_edge_attr,
            action_mask_ue,
            action_mask_cell_ue,
            obs_post_eq_sinr,
        )
        ue_logits, _, slot_valid_mask, slot_to_cell = self._action_head(
            cell_x,
            ue_x,
            prg_x,
            action_mask_ue,
            action_mask_cell_ue,
            slot_ue_class=None,
        )
        ue_class = ue_logits.argmax(dim=-1)
        _, prg_logits, _, _ = self._action_head(
            cell_x,
            ue_x,
            prg_x,
            action_mask_ue,
            action_mask_cell_ue,
            slot_ue_class=ue_class,
        )
        action_ue_select, action_prg_alloc = self._actions_from_logits(
            ue_logits,
            prg_logits,
            action_mask_ue,
            action_mask_cell_ue,
            slot_valid_mask,
            slot_to_cell,
        )
        edge_index_keepalive = obs_edge_index.to(dtype=action_ue_select.dtype).sum() * 0.0
        return action_ue_select + edge_index_keepalive, action_prg_alloc + edge_index_keepalive


def load_actor(checkpoint_path: Path, device: torch.device) -> tuple[StageBHGraphPolicy, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "actor_state" not in checkpoint or "actor_config" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not look like a Stage-B HGraph checkpoint")
    actor = build_model_from_config(checkpoint["actor_config"])
    actor.load_state_dict(checkpoint["actor_state"])
    actor.to(device=device)
    actor.eval()
    return actor, checkpoint


def make_dummy_inputs(
    *,
    n_cell: int,
    n_ue: int,
    n_prg: int,
    cell_feat_dim: int,
    ue_feat_dim: int,
    prg_feat_dim: int,
    device: torch.device,
    include_post_eq: bool = False,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    obs_cell = torch.rand((1, n_cell, cell_feat_dim), generator=generator, dtype=torch.float32).to(device)
    obs_ue = torch.rand((1, n_ue, ue_feat_dim), generator=generator, dtype=torch.float32).to(device)
    obs_prg = torch.rand((1, n_cell, n_prg, prg_feat_dim), generator=generator, dtype=torch.float32).to(device)

    obs_ue[:, :, 0] = obs_ue[:, :, 0] * 50000.0 + 1000.0
    obs_ue[:, :, 2] = obs_ue[:, :, 2] * 20.0 + 1.0e-3
    obs_ue[:, :, 6] = 1.0
    obs_ue[:, :, 8] = obs_ue[:, :, 8] * 200.0
    obs_ue[:, :, 9] = obs_ue[:, :, 9] * 200.0

    edge_index = _fixed_cell_edge_index(n_cell).view(1, n_cell * (n_cell - 1), 2).to(device)
    edge_attr = torch.rand((1, n_cell * (n_cell - 1), 2), generator=generator, dtype=torch.float32).to(device)
    action_mask_ue = torch.ones((1, n_ue), dtype=torch.float32, device=device)
    action_mask_cell_ue = _default_action_mask_cell_ue(n_cell, n_ue).view(1, n_cell, n_ue).to(device)
    if include_post_eq:
        obs_post_eq = torch.rand((1, n_ue, n_prg), generator=generator, dtype=torch.float32).to(device)
        obs_post_eq = obs_post_eq * 20.0 + 1.0e-3
        return obs_cell, obs_ue, obs_prg, edge_index, edge_attr, action_mask_ue, action_mask_cell_ue, obs_post_eq
    return obs_cell, obs_ue, obs_prg, edge_index, edge_attr, action_mask_ue, action_mask_cell_ue


def export_onnx(
    wrapper: FixedHGraphOnnxWrapper,
    dummy_inputs: tuple[torch.Tensor, ...],
    out_path: Path,
    *,
    opset: int,
    output_mode: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    input_names = [
        "obs_cell_features",
        "obs_ue_features",
        "obs_prg_features",
        "obs_edge_index",
        "obs_edge_attr",
        "action_mask_ue",
        "action_mask_cell_ue",
    ]
    output_names = ["ue_logits", "prg_logits"]
    if output_mode == "action":
        input_names.append("obs_post_eq_sinr")
        output_names = ["action_ue_select", "action_prg_alloc"]
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=False,
            dynamic_axes=None,
            dynamo=False,
        )


def check_onnx(
    onnx_path: Path,
    wrapper: FixedHGraphOnnxWrapper,
    dummy_inputs: tuple[torch.Tensor, ...],
    *,
    output_mode: str,
) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    result: dict[str, Any] = {"onnx_checker": "ok"}

    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - depends on local env
        result["onnxruntime"] = f"skipped: {type(exc).__name__}: {exc}"
        return result

    wrapper.eval()
    with torch.no_grad():
        torch_out = wrapper(*dummy_inputs)
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_names = [
        "obs_cell_features",
        "obs_ue_features",
        "obs_prg_features",
        "obs_edge_index",
        "obs_edge_attr",
        "action_mask_ue",
        "action_mask_cell_ue",
    ]
    output_names = ["ue_logits", "prg_logits"]
    if output_mode == "action":
        input_names.append("obs_post_eq_sinr")
        output_names = ["action_ue_select", "action_prg_alloc"]
    feed = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(input_names, dummy_inputs)
    }
    ort_out = session.run(output_names, feed)
    max_abs_diff = {}
    for name, torch_tensor, ort_array in zip(output_names, torch_out, ort_out):
        max_abs_diff[name] = float(
            torch.max(torch.abs(torch_tensor.detach().cpu() - torch.from_numpy(ort_array))).item()
        )
    result["onnxruntime"] = "ok"
    result["max_abs_diff"] = max_abs_diff
    return result


def write_metadata(
    metadata_path: Path,
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    out_path: Path,
    n_cell: int,
    n_ue: int,
    n_prg: int,
    n_sched_ue: int,
    opset: int,
    output_mode: str,
    use_post_eq_input: bool,
    check_result: dict[str, Any] | None,
) -> None:
    output_shapes = (
        {
            "action_ue_select": [1, n_sched_ue],
            "action_prg_alloc": [1, n_cell * n_prg],
        }
        if output_mode == "action"
        else {
            "ue_logits": [1, n_sched_ue, n_ue + 1],
            "prg_logits": [1, n_cell, n_prg, n_sched_ue + 1],
        }
    )
    input_shapes = {
        "obs_cell_features": [1, n_cell, 5],
        "obs_ue_features": [1, n_ue, 12],
        "obs_prg_features": [1, n_cell, n_prg, 8],
        "obs_edge_index": [1, n_cell * (n_cell - 1), 2],
        "obs_edge_attr": [1, n_cell * (n_cell - 1), 2],
        "action_mask_ue": [1, n_ue],
        "action_mask_cell_ue": [1, n_cell, n_ue],
    }
    if use_post_eq_input:
        input_shapes["obs_post_eq_sinr"] = [1, n_ue, n_prg]
    metrics = checkpoint.get("metrics")
    metadata = {
        "onnx_path": str(out_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_iter": checkpoint.get("iter"),
        "actor_config": checkpoint.get("actor_config"),
        "checkpoint_metrics": {
            key: metrics.get(key)
            for key in (
                "rollout_reward_mean",
                "rollout_post_eq_enabled_mean",
                "rollout_goodput_mbps_mean",
                "rollout_queue_delay_ms_mean",
            )
            if isinstance(metrics, dict) and key in metrics
        },
        "fixed_shape": {
            "n_cell": n_cell,
            "n_ue": n_ue,
            "n_prg": n_prg,
            "n_sched_ue": n_sched_ue,
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
        },
        "output_mode": output_mode,
        "decode_path": (
            "masked_argmax_to_action_ue_select_and_pre_cap_action_prg_alloc"
            if output_mode == "action"
            else "masked_argmax_for_internal_ue_conditioning"
        ),
        "use_post_eq_input": use_post_eq_input,
        "opset": opset,
        "check": check_result,
        "runtime_note": (
            "Action-mode ABI: C++ passes raw obs_post_eq_sinr, consumes action_ue_select/action_prg_alloc, "
            "then applies Type-0 legality, demand-cap, and fallback finalization."
            if output_mode == "action"
            else (
                "Logits-mode ABI: C++ consumes ue_logits/prg_logits and performs masked decode in "
                "GnnRlPolicyRuntime. Strict v33 post-eq parity requires using action mode or another "
                "export that includes obs_post_eq_sinr."
            )
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Stage-B HGraph actor checkpoint")
    parser.add_argument("--out", type=Path, default=None, help="Output ONNX path")
    parser.add_argument("--metadata-out", type=Path, default=None, help="Optional metadata JSON path")
    parser.add_argument("--output-mode", choices=["logits", "action"], default="logits")
    parser.add_argument(
        "--use-post-eq-input",
        action="store_true",
        help="Add obs_post_eq_sinr [1,U,P] input; currently required for output-mode=action.",
    )
    parser.add_argument("--n-cell", type=int, default=3)
    parser.add_argument("--n-ue", type=int, default=36)
    parser.add_argument("--n-prg", type=int, default=17)
    parser.add_argument("--n-sched-ue", type=int, default=36)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--check", action="store_true", help="Run ONNX checker and onnxruntime consistency if available")
    parser.add_argument("--device", default="cpu", help="Export device, usually cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if args.output_mode == "action" and not args.use_post_eq_input:
        args.use_post_eq_input = True
    out_path = (
        args.out.expanduser().resolve()
        if args.out is not None
        else (DEFAULT_ACTION_OUT if args.output_mode == "action" else DEFAULT_OUT).resolve()
    )
    metadata_path = (
        args.metadata_out.expanduser().resolve()
        if args.metadata_out is not None
        else out_path.with_suffix(out_path.suffix + ".metadata.json")
    )
    device = torch.device(args.device)

    actor, checkpoint = load_actor(checkpoint_path, device)
    wrapper_cls = FixedHGraphActionOnnxWrapper if args.output_mode == "action" else FixedHGraphOnnxWrapper
    wrapper = wrapper_cls(
        actor,
        n_cell=args.n_cell,
        n_ue=args.n_ue,
        n_prg=args.n_prg,
        n_sched_ue=args.n_sched_ue,
    ).to(device=device)
    wrapper.eval()

    actor_config = checkpoint["actor_config"]
    dummy_inputs = make_dummy_inputs(
        n_cell=args.n_cell,
        n_ue=args.n_ue,
        n_prg=args.n_prg,
        cell_feat_dim=int(actor_config["cell_feat_dim"]),
        ue_feat_dim=int(actor_config["ue_feat_dim"]),
        prg_feat_dim=int(actor_config["prg_feat_dim"]),
        device=device,
        include_post_eq=bool(args.use_post_eq_input),
    )
    with torch.no_grad():
        ue_logits, prg_logits = wrapper(*dummy_inputs)
    if args.output_mode == "action":
        expected_ue = (1, args.n_sched_ue)
        expected_prg = (1, args.n_cell * args.n_prg)
    else:
        expected_ue = (1, args.n_sched_ue, args.n_ue + 1)
        expected_prg = (1, args.n_cell, args.n_prg, args.n_sched_ue + 1)
    if tuple(ue_logits.shape) != expected_ue or tuple(prg_logits.shape) != expected_prg:
        raise RuntimeError(
            f"unexpected wrapper output shapes: ue={tuple(ue_logits.shape)} expected={expected_ue}, "
            f"prg={tuple(prg_logits.shape)} expected={expected_prg}"
        )

    export_onnx(wrapper, dummy_inputs, out_path, opset=args.opset, output_mode=str(args.output_mode))
    check_result = (
        check_onnx(out_path, wrapper, dummy_inputs, output_mode=str(args.output_mode)) if args.check else None
    )
    write_metadata(
        metadata_path,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        out_path=out_path,
        n_cell=args.n_cell,
        n_ue=args.n_ue,
        n_prg=args.n_prg,
        n_sched_ue=args.n_sched_ue,
        opset=args.opset,
        output_mode=str(args.output_mode),
        use_post_eq_input=bool(args.use_post_eq_input),
        check_result=check_result,
    )
    print(f"[export] ONNX: {out_path}")
    print(f"[export] metadata: {metadata_path}")
    if check_result is not None:
        print(f"[export] check: {json.dumps(check_result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
