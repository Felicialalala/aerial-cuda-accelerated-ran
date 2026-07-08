from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from training.gnnrl.slot_layout import build_schedulable_cell_ue_mask, build_type0_slot_layout

from .graph_ops import pack_prg_candidates
from .template_encoders import CellTemplateEncoder, PrgTemplateEncoder, UeTemplateEncoder
from .types import SparseEntityGraph


@dataclass(frozen=True)
class HGraphModelConfig:
    cell_feat_dim: int = 5
    ue_feat_dim: int = 12
    prg_feat_dim: int = 12
    hidden_dim: int = 128
    message_layers: int = 2
    cc_edge_attr_dim: int = 2
    up_edge_attr_dim: int = 6
    dropout: float = 0.0
    max_slot_global_pos: int = 256
    max_slot_local_pos: int = 128
    max_prg_local_pos: int = 128
    max_prg_candidates: int = 4
    action_head_mode: str = "slot"
    candidate_blank_logit_bias: float = 0.0
    candidate_success_logit_scale: float = 0.0
    candidate_blank_success_scale: float = 0.0
    candidate_pfq_anchor_logit_coef: float = 1.0
    has_pfq_score_anchor: int = 1
    candidate_seq_state_dim: int = 0


def _scatter_mean(messages: torch.Tensor, dst_index: torch.Tensor, n_dst: int) -> torch.Tensor:
    hidden_dim = int(messages.shape[-1])
    out = torch.zeros((n_dst, hidden_dim), dtype=messages.dtype, device=messages.device)
    if messages.numel() == 0:
        return out
    dst = dst_index.to(torch.long)
    out.index_add_(0, dst, messages)
    counts = torch.zeros((n_dst,), dtype=messages.dtype, device=messages.device)
    counts.index_add_(0, dst, torch.ones((dst.shape[0],), dtype=messages.dtype, device=messages.device))
    out = out / counts.clamp_min(1.0).unsqueeze(-1)
    return out


def _fit_feature_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    target_dim = int(dim)
    current_dim = int(x.shape[-1])
    if current_dim == target_dim:
        return x
    if current_dim > target_dim:
        return x[..., :target_dim]
    pad_shape = (*x.shape[:-1], target_dim - current_dim)
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat((x, pad), dim=-1)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _RelationMessage(nn.Module):
    def __init__(self, hidden_dim: int, edge_attr_dim: int = 0):
        super().__init__()
        self.edge_attr_dim = int(edge_attr_dim)
        self.src_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_attr_dim, hidden_dim) if edge_attr_dim > 0 else None
        self.msg_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        src_x: torch.Tensor,
        edge_index: torch.Tensor,
        n_dst: int,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros((n_dst, src_x.shape[-1]), dtype=src_x.dtype, device=src_x.device)
        src_idx = edge_index[0].to(device=src_x.device, dtype=torch.long)
        dst_idx = edge_index[1].to(device=src_x.device, dtype=torch.long)
        msg = self.src_proj(src_x[src_idx])
        if self.edge_proj is not None and edge_attr is not None:
            msg = msg + self.edge_proj(edge_attr.to(device=src_x.device, dtype=src_x.dtype))
        msg = self.msg_proj(msg)
        return _scatter_mean(msg, dst_idx, n_dst)


class _NodeUpdate(nn.Module):
    def __init__(self, hidden_dim: int, num_inputs: int, dropout: float = 0.0):
        super().__init__()
        in_dim = hidden_dim * (1 + num_inputs)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, self_x: torch.Tensor, *aggs: torch.Tensor) -> torch.Tensor:
        merged = torch.cat((self_x, *aggs), dim=-1)
        return self.norm(self_x + self.net(merged))


class HeteroMessageLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cc_edge_attr_dim: int = 2,
        up_edge_attr_dim: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rel_cc = _RelationMessage(hidden_dim, edge_attr_dim=cc_edge_attr_dim)
        self.rel_cu = _RelationMessage(hidden_dim)
        self.rel_cp = _RelationMessage(hidden_dim)
        self.rel_up = _RelationMessage(hidden_dim, edge_attr_dim=up_edge_attr_dim)
        self.rel_pp = _RelationMessage(hidden_dim)

        self.update_cell = _NodeUpdate(hidden_dim, num_inputs=1, dropout=dropout)
        self.update_ue = _NodeUpdate(hidden_dim, num_inputs=1, dropout=dropout)
        self.update_prg = _NodeUpdate(hidden_dim, num_inputs=3, dropout=dropout)

    def forward(
        self,
        cell_x: torch.Tensor,
        ue_x: torch.Tensor,
        prg_x: torch.Tensor,
        graph: SparseEntityGraph,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e_cc = graph.edges["E_CC"]
        e_cu = graph.edges["E_CU"]
        e_cp = graph.edges["E_CP"]
        e_up = graph.edges["E_UP"]
        e_pp = graph.edges["E_PP"]

        agg_cell = self.rel_cc(cell_x, e_cc.index, n_dst=cell_x.shape[0], edge_attr=e_cc.attr)
        agg_ue = self.rel_cu(cell_x, e_cu.index, n_dst=ue_x.shape[0], edge_attr=e_cu.attr)
        agg_prg_from_cell = self.rel_cp(cell_x, e_cp.index, n_dst=prg_x.shape[0], edge_attr=e_cp.attr)
        agg_prg_from_ue = self.rel_up(ue_x, e_up.index, n_dst=prg_x.shape[0], edge_attr=e_up.attr)
        agg_prg_from_prg = self.rel_pp(prg_x, e_pp.index, n_dst=prg_x.shape[0], edge_attr=e_pp.attr)

        cell_x = self.update_cell(cell_x, agg_cell)
        ue_x = self.update_ue(ue_x, agg_ue)
        prg_x = self.update_prg(prg_x, agg_prg_from_cell, agg_prg_from_ue, agg_prg_from_prg)
        return cell_x, ue_x, prg_x


class JointType0ActionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        up_edge_attr_dim: int = 6,
        max_slot_global_pos: int = 256,
        max_slot_local_pos: int = 128,
        max_prg_local_pos: int = 128,
        max_prg_candidates: int = 4,
        action_head_mode: str = "slot",
        candidate_blank_logit_bias: float = 0.0,
        candidate_success_logit_scale: float = 1.0,
        candidate_blank_success_scale: float = 1.0,
        candidate_pfq_anchor_logit_coef: float = 1.0,
        has_pfq_score_anchor: int = 1,
        candidate_seq_state_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        action_head_mode = str(action_head_mode)
        if action_head_mode not in {"slot", "candidate"}:
            raise ValueError(f"action_head_mode must be 'slot' or 'candidate', got {action_head_mode!r}")
        self.action_head_mode = action_head_mode
        self.max_prg_candidates = int(max_prg_candidates)
        self.up_edge_attr_dim = int(up_edge_attr_dim)
        self.candidate_blank_logit_bias = float(candidate_blank_logit_bias)
        self.candidate_success_logit_scale = float(candidate_success_logit_scale)
        self.candidate_blank_success_scale = float(candidate_blank_success_scale)
        self.candidate_pfq_anchor_logit_coef = float(candidate_pfq_anchor_logit_coef)
        self.has_pfq_score_anchor = bool(int(has_pfq_score_anchor))
        self.candidate_seq_state_dim = max(0, int(candidate_seq_state_dim))
        self.slot_pos_emb = nn.Embedding(max_slot_global_pos, hidden_dim)
        self.slot_local_pos_emb = nn.Embedding(max_slot_local_pos, hidden_dim)
        self.prg_pos_emb = nn.Embedding(max_prg_local_pos, hidden_dim)

        self.slot_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_query = nn.Linear(hidden_dim, hidden_dim)
        self.ue_key = nn.Linear(hidden_dim, hidden_dim)
        self.ue_feedback_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ue_null_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.slot_ue_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prg_ctx = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prg_query = nn.Linear(hidden_dim, hidden_dim)
        self.prg_slot_key = nn.Linear(hidden_dim, hidden_dim)
        self.prg_null_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.candidate_edge_proj = (
            nn.Linear(max(1, int(up_edge_attr_dim)), hidden_dim) if action_head_mode == "candidate" else None
        )
        self.candidate_pair_score = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if action_head_mode == "candidate"
            else None
        )
        self.candidate_success_score = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if action_head_mode == "candidate"
            else None
        )
        self.candidate_seq_state_score = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.candidate_seq_state_dim),
            )
            if action_head_mode == "candidate" and self.candidate_seq_state_dim > 0
            else None
        )
        self.candidate_null_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if action_head_mode == "candidate"
            else None
        )
        if self.candidate_success_score is not None:
            final = self.candidate_success_score[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.constant_(final.bias, 1.5)
        if self.candidate_seq_state_score is not None:
            final = self.candidate_seq_state_score[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def _candidate_logits_batched(
        self,
        prg_ctx: torch.Tensor,
        ue_x: torch.Tensor,
        graphs: Sequence[SparseEntityGraph],
    ) -> Dict[str, torch.Tensor]:
        if self.candidate_pair_score is None or self.candidate_success_score is None or self.candidate_null_head is None:
            return {}
        if prg_ctx.dim() != 4:
            raise ValueError(f"candidate head expects prg_ctx [B,C,P,H], got {tuple(prg_ctx.shape)}")

        batch_size, n_cell, n_prg, hid = prg_ctx.shape
        n_prg_tokens = n_cell * n_prg
        max_candidates = max(0, int(self.max_prg_candidates))
        device = prg_ctx.device
        dtype = prg_ctx.dtype
        prg_token_ctx = prg_ctx.reshape(batch_size, n_prg_tokens, hid)

        packed = [
            pack_prg_candidates(graph, max_candidates=max_candidates, fixed_width=True)
            for graph in graphs
        ]
        candidate_ue_ids = torch.stack(
            [item.ue_ids.to(device=device, dtype=torch.long) for item in packed],
            dim=0,
        )
        candidate_mask = torch.stack(
            [item.mask.to(device=device, dtype=torch.bool) for item in packed],
            dim=0,
        )

        if max_candidates > 0:
            safe_ue_ids = candidate_ue_ids.clamp(min=0, max=max(0, int(ue_x.shape[1]) - 1))
            gathered_ue = ue_x.gather(
                1,
                safe_ue_ids.reshape(batch_size, n_prg_tokens * max_candidates)
                .unsqueeze(-1)
                .expand(-1, -1, hid),
            ).reshape(batch_size, n_prg_tokens, max_candidates, hid)

            attr_dim = max(1, int(self.up_edge_attr_dim))
            edge_attr = torch.zeros(
                (batch_size, n_prg_tokens, max_candidates, attr_dim),
                dtype=dtype,
                device=device,
            )
            for batch_idx, item in enumerate(packed):
                if item.edge_attr is None:
                    continue
                raw_attr = item.edge_attr.to(device=device, dtype=dtype)
                copy_dim = min(attr_dim, int(raw_attr.shape[-1]))
                if copy_dim > 0:
                    edge_attr[batch_idx, :, :, :copy_dim] = raw_attr[:, :, :copy_dim]
            pfq_anchor_logits = (
                edge_attr[..., 5] * float(self.candidate_pfq_anchor_logit_coef)
                if self.has_pfq_score_anchor and attr_dim >= 6
                else torch.zeros((batch_size, n_prg_tokens, max_candidates), dtype=dtype, device=device)
            )
            edge_ctx = self.candidate_edge_proj(edge_attr) if self.candidate_edge_proj is not None else torch.zeros_like(gathered_ue)
            prg_expand = prg_token_ctx.unsqueeze(2).expand(-1, -1, max_candidates, -1)
            pair_ctx = torch.cat([prg_expand, gathered_ue, edge_ctx], dim=-1)
            candidate_service_value = self.candidate_pair_score(pair_ctx).squeeze(-1)
            candidate_success_logits = self.candidate_success_score(pair_ctx).squeeze(-1)
            if self.candidate_seq_state_score is not None:
                candidate_seq_state_weights = self.candidate_seq_state_score(pair_ctx)
            else:
                candidate_seq_state_weights = torch.empty(
                    (batch_size, n_prg_tokens, max_candidates, 0),
                    dtype=dtype,
                    device=device,
                )
            success_scale = float(self.candidate_success_logit_scale)
            if success_scale > 0.0:
                candidate_logits_main = candidate_service_value + F.logsigmoid(
                    candidate_success_logits * max(success_scale, 1.0e-3)
                )
            else:
                candidate_logits_main = candidate_service_value
            candidate_logits_main = candidate_logits_main + pfq_anchor_logits
            candidate_logits_main = candidate_logits_main.masked_fill(~candidate_mask, -1.0e9)
            candidate_service_value = candidate_service_value.masked_fill(~candidate_mask, 0.0)
            candidate_success_logits = candidate_success_logits.masked_fill(~candidate_mask, -1.0e9)
            candidate_seq_state_weights = candidate_seq_state_weights.masked_fill(
                ~candidate_mask.unsqueeze(-1),
                0.0,
            )
            pfq_anchor_logits = pfq_anchor_logits.masked_fill(~candidate_mask, 0.0)
        else:
            candidate_logits_main = torch.empty(
                (batch_size, n_prg_tokens, 0),
                dtype=dtype,
                device=device,
            )
            candidate_service_value = candidate_logits_main
            candidate_success_logits = candidate_logits_main
            pfq_anchor_logits = candidate_logits_main
            candidate_seq_state_weights = torch.empty(
                (batch_size, n_prg_tokens, 0, self.candidate_seq_state_dim),
                dtype=dtype,
                device=device,
            )

        success_scale = float(self.candidate_success_logit_scale)
        blank_success_scale = float(self.candidate_blank_success_scale)
        if max_candidates > 0 and success_scale > 0.0 and blank_success_scale != 0.0:
            has_valid = candidate_mask.any(dim=-1, keepdim=True)
            max_success = candidate_success_logits.max(dim=-1, keepdim=True).values
            blank_gate = torch.where(
                has_valid,
                torch.clamp(-max_success * max(success_scale, 1.0e-3), min=-8.0, max=8.0),
                torch.zeros((batch_size, n_prg_tokens, 1), dtype=dtype, device=device),
            )
        else:
            blank_gate = torch.zeros((batch_size, n_prg_tokens, 1), dtype=dtype, device=device)
        candidate_no = (
            self.candidate_null_head(prg_token_ctx)
            + float(self.candidate_blank_logit_bias)
            + float(self.candidate_blank_success_scale) * blank_gate
        )
        candidate_logits = torch.cat([candidate_logits_main, candidate_no], dim=-1)
        return {
            "candidate_prg_logits": candidate_logits,
            "candidate_success_logits": candidate_success_logits,
            "candidate_service_value_logits": candidate_service_value,
            "candidate_pfq_anchor_logits": pfq_anchor_logits,
            "candidate_seq_state_weights": candidate_seq_state_weights,
            "candidate_blank_success_logprob": blank_gate,
            "candidate_blank_success_margin": blank_gate,
            "candidate_ue_ids": candidate_ue_ids,
            "candidate_mask": candidate_mask,
        }

    def forward_batched(
        self,
        cell_x: torch.Tensor,
        ue_x: torch.Tensor,
        prg_x: torch.Tensor,
        graphs: Sequence[SparseEntityGraph],
        *,
        batched_up_index: torch.Tensor | None = None,
        batched_up_attr: torch.Tensor | None = None,
        slot_ue_class: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if cell_x.dim() != 3 or ue_x.dim() != 3 or prg_x.dim() != 3:
            raise ValueError(
                "forward_batched expects [B,N,H] tensors, got "
                f"cell={tuple(cell_x.shape)} ue={tuple(ue_x.shape)} prg={tuple(prg_x.shape)}"
            )
        if len(graphs) != int(cell_x.shape[0]):
            raise ValueError(f"graph batch mismatch: tensors batch={cell_x.shape[0]} graphs={len(graphs)}")

        ref = graphs[0].snapshot.static_meta
        batch_size = int(cell_x.shape[0])
        n_cell = int(ref.n_cell)
        n_ue = int(ref.n_ue)
        n_sched_ue = int(ref.n_sched_ue)
        n_prg = int(ref.n_prg)
        n_prg_tokens = n_cell * n_prg
        device = prg_x.device
        dtype = prg_x.dtype
        hid = int(prg_x.shape[-1])

        action_mask_cell_ue = torch.stack(
            [graph.snapshot.static_meta.action_mask_cell_ue.to(device=device, dtype=torch.bool) for graph in graphs],
            dim=0,
        )
        action_mask_ue = torch.stack(
            [
                (
                    graph.snapshot.static_meta.action_mask_ue
                    if graph.snapshot.static_meta.action_mask_ue is not None
                    else torch.ones((n_ue,), dtype=torch.bool)
                ).to(device=device, dtype=torch.bool)
                for graph in graphs
            ],
            dim=0,
        )
        layout = build_type0_slot_layout(
            action_mask_cell_ue,
            n_sched_ue=n_sched_ue,
            action_mask_ue=action_mask_ue,
        )
        slot_to_cell = layout.slot_to_cell
        slot_local_idx = layout.slot_local_idx
        slot_valid_mask = layout.slot_valid_mask

        prg_owner_cell = torch.stack(
            [graph.snapshot.static_meta.prg_owner_cell.to(device=device, dtype=torch.long) for graph in graphs],
            dim=0,
        )
        prg_local_idx = torch.stack(
            [graph.snapshot.static_meta.prg_local_index.to(device=device, dtype=torch.long) for graph in graphs],
            dim=0,
        )
        slot_global_idx = torch.arange(n_sched_ue, device=device, dtype=torch.long)
        slot_cell_ctx = cell_x.gather(1, slot_to_cell.unsqueeze(-1).expand(-1, -1, hid))
        slot_pos = self.slot_pos_emb(slot_global_idx.clamp(max=self.slot_pos_emb.num_embeddings - 1)).unsqueeze(0)
        slot_pos = slot_pos.expand(batch_size, -1, -1)
        slot_local_pos = self.slot_local_pos_emb(
            slot_local_idx.clamp(max=self.slot_local_pos_emb.num_embeddings - 1)
        )
        slot_ctx = self.slot_fuse(torch.cat([slot_cell_ctx, slot_pos, slot_local_pos], dim=-1))

        slot_q = self.slot_query(slot_ctx)
        ue_k = self.ue_key(ue_x)
        ue_logits_main = torch.einsum("bsh,buh->bsu", slot_q, ue_k) / math.sqrt(float(hid))

        sched_cell_ue_mask = build_schedulable_cell_ue_mask(
            action_mask_cell_ue,
            action_mask_ue=action_mask_ue,
        )
        slot_cell_ue_mask = sched_cell_ue_mask.gather(
            1,
            slot_to_cell.unsqueeze(-1).expand(-1, -1, n_ue),
        )
        slot_cell_ue_mask = slot_cell_ue_mask & slot_valid_mask.unsqueeze(-1)
        ue_logits_main = ue_logits_main.masked_fill(~slot_cell_ue_mask, -1.0e9)
        ue_no = self.ue_null_head(slot_ctx)
        ue_logits = torch.cat([ue_logits_main, ue_no], dim=-1)

        slot_prior_ue_class = ue_logits_main.argmax(dim=-1)
        slot_prior_ue_class = torch.where(slot_valid_mask, slot_prior_ue_class, torch.full_like(slot_prior_ue_class, -1))
        if slot_ue_class is None:
            slot_ue_prob = torch.softmax(ue_logits_main, dim=-1)
            slot_ue_prob = torch.where(slot_valid_mask.unsqueeze(-1), slot_ue_prob, torch.zeros_like(slot_ue_prob))
            hard_slot_ue = F.one_hot(
                slot_prior_ue_class.clamp(min=0, max=max(0, n_ue - 1)),
                num_classes=n_ue,
            ).to(dtype)
            slot_ue_ctx_weights = hard_slot_ue + (slot_ue_prob - slot_ue_prob.detach())
            slot_ue_ctx_weights = torch.where(
                slot_valid_mask.unsqueeze(-1),
                slot_ue_ctx_weights,
                torch.zeros_like(hard_slot_ue),
            )
        else:
            explicit_slot_ue = slot_ue_class.to(device=device, dtype=torch.long)
            if explicit_slot_ue.shape != (batch_size, n_sched_ue):
                raise ValueError(
                    "slot_ue_class shape mismatch: "
                    f"got={tuple(explicit_slot_ue.shape)} expected={(batch_size, n_sched_ue)}"
                )
            safe_slot_ue = explicit_slot_ue.clamp(min=0, max=max(0, n_ue - 1))
            explicit_valid = (
                slot_valid_mask
                & (explicit_slot_ue >= 0)
                & (explicit_slot_ue < n_ue)
                & slot_cell_ue_mask.gather(-1, safe_slot_ue.unsqueeze(-1)).squeeze(-1)
            )
            hard_slot_ue = F.one_hot(safe_slot_ue, num_classes=n_ue).to(dtype)
            slot_ue_ctx_weights = torch.where(
                explicit_valid.unsqueeze(-1),
                hard_slot_ue,
                torch.zeros_like(hard_slot_ue),
            )
        slot_ue_soft_ctx = torch.einsum("bsu,buh->bsh", slot_ue_ctx_weights, ue_x)
        slot_prg_ctx = self.slot_ue_fuse(torch.cat([slot_ctx, slot_ue_soft_ctx], dim=-1))

        prg_by_cell = prg_x.reshape(batch_size, n_cell, n_prg, hid)
        cell_expand = cell_x.unsqueeze(2).expand(-1, -1, n_prg, -1)
        prg_pos = self.prg_pos_emb(
            prg_local_idx.reshape(batch_size, n_cell, n_prg).clamp(max=self.prg_pos_emb.num_embeddings - 1)
        )
        prg_ctx = self.prg_ctx(torch.cat([prg_by_cell, cell_expand, prg_pos], dim=-1))
        prg_q = self.prg_query(prg_ctx)
        slot_k = self.prg_slot_key(slot_prg_ctx)
        prg_logits_main = torch.einsum("bcph,bsh->bcps", prg_q, slot_k) / math.sqrt(float(hid))
        prg_no = self.prg_null_head(prg_ctx)
        prg_logits = torch.cat([prg_logits_main, prg_no], dim=-1)
        candidate_out = self._candidate_logits_batched(prg_ctx, ue_x, graphs)
        return {
            "ue_logits": ue_logits,
            "prg_logits": prg_logits,
            "slot_valid_mask": slot_valid_mask,
            "slot_to_cell": slot_to_cell,
            "slot_local_idx": slot_local_idx,
            "slot_prior_ue_class": slot_prior_ue_class,
            **candidate_out,
        }

    def forward(
        self,
        cell_x: torch.Tensor,
        ue_x: torch.Tensor,
        prg_x: torch.Tensor,
        graph: SparseEntityGraph,
        *,
        slot_ue_class: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batched_slot_ue_class = None
        if slot_ue_class is not None:
            batched_slot_ue_class = slot_ue_class
            if batched_slot_ue_class.dim() == 1:
                batched_slot_ue_class = batched_slot_ue_class.unsqueeze(0)
        out = self.forward_batched(
            cell_x.unsqueeze(0),
            ue_x.unsqueeze(0),
            prg_x.unsqueeze(0),
            [graph],
            slot_ue_class=batched_slot_ue_class,
        )
        out_single = {
            "ue_logits": out["ue_logits"].squeeze(0),
            "prg_logits": out["prg_logits"].squeeze(0),
            "slot_valid_mask": out["slot_valid_mask"].squeeze(0),
            "slot_to_cell": out["slot_to_cell"].squeeze(0),
            "slot_local_idx": out["slot_local_idx"].squeeze(0),
            "slot_prior_ue_class": out["slot_prior_ue_class"].squeeze(0),
        }
        if "candidate_prg_logits" in out:
            out_single["candidate_prg_logits"] = out["candidate_prg_logits"].squeeze(0)
            out_single["candidate_success_logits"] = out["candidate_success_logits"].squeeze(0)
            out_single["candidate_service_value_logits"] = out["candidate_service_value_logits"].squeeze(0)
            out_single["candidate_pfq_anchor_logits"] = out["candidate_pfq_anchor_logits"].squeeze(0)
            out_single["candidate_seq_state_weights"] = out["candidate_seq_state_weights"].squeeze(0)
            out_single["candidate_blank_success_logprob"] = out["candidate_blank_success_logprob"].squeeze(0)
            out_single["candidate_blank_success_margin"] = out["candidate_blank_success_margin"].squeeze(0)
            out_single["candidate_ue_ids"] = out["candidate_ue_ids"].squeeze(0)
            out_single["candidate_mask"] = out["candidate_mask"].squeeze(0)
        return out_single


class StageBHGraphPolicy(nn.Module):
    def __init__(self, cfg: HGraphModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or HGraphModelConfig()
        hid = int(self.cfg.hidden_dim)

        self.cell_encoder = CellTemplateEncoder(
            in_dim=self.cfg.cell_feat_dim,
            hidden_dim=hid,
            out_dim=hid,
        )
        self.ue_encoder = UeTemplateEncoder(
            in_dim=self.cfg.ue_feat_dim,
            hidden_dim=max(hid, 96),
            out_dim=hid,
        )
        self.prg_encoder = PrgTemplateEncoder(
            in_dim=self.cfg.prg_feat_dim,
            hidden_dim=hid,
            out_dim=hid,
        )

        self.cell_residual = _ResidualBlock(hid, hid * 2, dropout=self.cfg.dropout)
        self.ue_residual = _ResidualBlock(hid, hid * 2, dropout=self.cfg.dropout)
        self.prg_residual = _ResidualBlock(hid, hid * 2, dropout=self.cfg.dropout)

        self.message_layers = nn.ModuleList(
            [
                HeteroMessageLayer(
                    hidden_dim=hid,
                    cc_edge_attr_dim=self.cfg.cc_edge_attr_dim,
                    up_edge_attr_dim=self.cfg.up_edge_attr_dim,
                    dropout=self.cfg.dropout,
                )
                for _ in range(max(1, int(self.cfg.message_layers)))
            ]
        )
        self.action_head = JointType0ActionHead(
            hidden_dim=hid,
            up_edge_attr_dim=int(self.cfg.up_edge_attr_dim),
            max_slot_global_pos=int(self.cfg.max_slot_global_pos),
            max_slot_local_pos=int(self.cfg.max_slot_local_pos),
            max_prg_local_pos=int(self.cfg.max_prg_local_pos),
            max_prg_candidates=int(self.cfg.max_prg_candidates),
            action_head_mode=str(self.cfg.action_head_mode),
            candidate_blank_logit_bias=float(self.cfg.candidate_blank_logit_bias),
            candidate_success_logit_scale=float(self.cfg.candidate_success_logit_scale),
            candidate_blank_success_scale=float(self.cfg.candidate_blank_success_scale),
            candidate_pfq_anchor_logit_coef=float(self.cfg.candidate_pfq_anchor_logit_coef),
            has_pfq_score_anchor=int(self.cfg.has_pfq_score_anchor),
            candidate_seq_state_dim=int(self.cfg.candidate_seq_state_dim),
            dropout=float(self.cfg.dropout),
        )

    def encode_graph(self, graph: SparseEntityGraph) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        cell_features = _fit_feature_dim(graph.snapshot.cell_features.values.to(device=device), self.cfg.cell_feat_dim)
        ue_features = _fit_feature_dim(graph.snapshot.ue_features.values.to(device=device), self.cfg.ue_feat_dim)
        prg_features = _fit_feature_dim(graph.snapshot.prg_features.values.to(device=device), self.cfg.prg_feat_dim)
        cell_x = self.cell_residual(self.cell_encoder(cell_features))
        ue_x = self.ue_residual(self.ue_encoder(ue_features))
        prg_x = self.prg_residual(self.prg_encoder(prg_features))
        for layer in self.message_layers:
            cell_x, ue_x, prg_x = layer(cell_x, ue_x, prg_x, graph)
        return {
            "cell_embeddings": cell_x,
            "ue_embeddings": ue_x,
            "prg_embeddings": prg_x,
        }

    def forward_graph(
        self,
        graph: SparseEntityGraph,
        *,
        slot_ue_class_for_prg: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        embeddings = self.encode_graph(graph)
        action_out = self.action_head(
            embeddings["cell_embeddings"],
            embeddings["ue_embeddings"],
            embeddings["prg_embeddings"],
            graph,
            slot_ue_class=slot_ue_class_for_prg,
        )
        return {
            **embeddings,
            **action_out,
        }

    def forward(
        self,
        graph: SparseEntityGraph,
        *,
        slot_ue_class_for_prg: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_graph(graph, slot_ue_class_for_prg=slot_ue_class_for_prg)

    def model_config_dict(self) -> Dict[str, int | float]:
        return {
            "cell_feat_dim": self.cfg.cell_feat_dim,
            "ue_feat_dim": self.cfg.ue_feat_dim,
            "prg_feat_dim": self.cfg.prg_feat_dim,
            "hidden_dim": self.cfg.hidden_dim,
            "message_layers": self.cfg.message_layers,
            "cc_edge_attr_dim": self.cfg.cc_edge_attr_dim,
            "up_edge_attr_dim": self.cfg.up_edge_attr_dim,
            "dropout": self.cfg.dropout,
            "max_slot_global_pos": self.cfg.max_slot_global_pos,
            "max_slot_local_pos": self.cfg.max_slot_local_pos,
            "max_prg_local_pos": self.cfg.max_prg_local_pos,
            "max_prg_candidates": self.cfg.max_prg_candidates,
            "action_head_mode": self.cfg.action_head_mode,
            "candidate_blank_logit_bias": self.cfg.candidate_blank_logit_bias,
            "candidate_success_logit_scale": self.cfg.candidate_success_logit_scale,
            "candidate_blank_success_scale": self.cfg.candidate_blank_success_scale,
            "candidate_pfq_anchor_logit_coef": self.cfg.candidate_pfq_anchor_logit_coef,
            "has_pfq_score_anchor": self.cfg.has_pfq_score_anchor,
            "candidate_seq_state_dim": self.cfg.candidate_seq_state_dim,
        }


def build_model_from_config(cfg_dict: Dict[str, int | float]) -> StageBHGraphPolicy:
    cfg = dict(cfg_dict)
    if "candidate_pfq_anchor_logit_coef" not in cfg:
        cfg["candidate_pfq_anchor_logit_coef"] = 0.0
    if "has_pfq_score_anchor" not in cfg:
        edge_dim = int(cfg.get("up_edge_attr_dim", 6))
        cfg["has_pfq_score_anchor"] = int(edge_dim in (6, 9, 11, 14, 20, 30))
    if "candidate_seq_state_dim" not in cfg:
        cfg["candidate_seq_state_dim"] = 0
    return StageBHGraphPolicy(HGraphModelConfig(**cfg))
