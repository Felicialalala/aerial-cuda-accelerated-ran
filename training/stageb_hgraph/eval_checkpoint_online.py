#!/usr/bin/env python3
"""Evaluate a fixed Stage-B HGraph checkpoint through the online bridge."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys

    _script_dir = __file__.rsplit("/", 1)[0]
    _repo_root = _script_dir.rsplit("/", 2)[0]
    sys.path = [p for p in sys.path if p not in ("", _script_dir)]
    sys.path.insert(0, _repo_root)

import argparse
import json
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.distributions import Categorical


def _configure_torch_threads_from_env() -> None:
    num_threads_raw = os.environ.get("CUMAC_TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    interop_raw = os.environ.get("CUMAC_TORCH_NUM_INTEROP_THREADS")
    if num_threads_raw:
        try:
            torch.set_num_threads(max(1, int(num_threads_raw)))
        except (RuntimeError, ValueError):
            pass
    if interop_raw:
        try:
            torch.set_num_interop_threads(max(1, int(interop_raw)))
        except (RuntimeError, ValueError):
            pass


_configure_torch_threads_from_env()

from training.stageb_hgraph.bridge_env import BridgeEnvClient, BridgeEnvConfig
from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from training.gnnrl.masks import apply_prg_action_mask, apply_ue_action_mask, build_slot_selection_mask
from training.stageb_hgraph.graph_ops import (
    apply_candidate_blank_budget,
    build_prg_candidate_valid_mask,
    decode_candidate_actions_to_type0,
    decode_joint_actions_to_type0,
    pack_prg_candidates,
    select_and_decode_budgeted_candidate_actions,
)
from training.stageb_hgraph.model import HGraphModelConfig, StageBHGraphPolicy, build_model_from_config
from training.stageb_hgraph.online_imitation import _pick_device, _set_seed
from training.stageb_hgraph.online_ppo import (
    _apply_candidate_direct_service_bias,
    _candidate_direct_bias_selected_mean,
    _graph_to_device,
    _repair_ue_action_for_coverage,
)
from training.stageb_hgraph.reward_utils import extract_reward_metrics


def _load_actor_checkpoint(path: Path, device: torch.device) -> tuple[StageBHGraphPolicy, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    actor_cfg = ckpt.get("actor_config")
    if isinstance(actor_cfg, dict):
        actor = build_model_from_config(actor_cfg)
    else:
        actor = StageBHGraphPolicy(HGraphModelConfig())

    state = ckpt.get("actor_state")
    if state is None:
        state = ckpt.get("model_state", ckpt)
    target_state = actor.state_dict()
    compatible = {k: v for k, v in state.items() if k in target_state and target_state[k].shape == v.shape}
    actor.load_state_dict(compatible, strict=False)
    actor.to(device)
    actor.eval()
    return actor, ckpt


def _checkpoint_arg(ckpt: Dict[str, Any], name: str, fallback: Any) -> Any:
    args = ckpt.get("args")
    if isinstance(args, dict) and args.get(name) is not None:
        return args[name]
    return fallback


def _select_joint_action(
    actor: StageBHGraphPolicy,
    graph,
    *,
    decode_mode: str,
    slot_repair_max_per_cell: int = 0,
    slot_repair_min_backlog_bytes: float = 1.0,
    safe_fill_max_per_cell: int = 0,
    safe_fill_min_quality_db: float = -2.0,
    safe_fill_max_quality_gap_db: float = 8.0,
    safe_fill_max_prg_risk: float = 0.70,
    safe_fill_min_backlog_bytes: float = 1.0,
    safe_fill_max_ue_tb_err: float = 0.45,
    candidate_blank_budget_max_per_cell: int = -1,
    candidate_blank_budget_min_decision_logit: float = -1.0e9,
    candidate_decode_demand_slack_bytes: float = 3000.0,
    candidate_marginal_headroom_slack_bytes: float = -1.0,
    candidate_headroom_rank_mode: str = "stable",
    candidate_budgeted_select_mode: str = "off",
    candidate_budgeted_hard_cap: int = 1,
    candidate_budgeted_usage_penalty: float = 0.35,
    candidate_budgeted_overcap_penalty: float = 8.0,
    candidate_budgeted_first_service_bonus: float = 0.0,
    candidate_budgeted_tail_bias_coef: float = 0.0,
    candidate_budgeted_tail_bias_service_target: float = 0.75,
    candidate_budgeted_tail_bias_ttl_guard_ms: float = 20.0,
    candidate_budgeted_tail_bias_max: float = 1.5,
    candidate_seq_state_coef: float = 1.0,
    candidate_seq_fail_risk_feature_offset: int = -1,
) -> Dict[str, Any]:
    with torch.inference_mode():
        actor_out = actor.forward_graph(graph)

    device = actor_out["ue_logits"].device
    meta = graph.snapshot.static_meta
    action_mask_ue = (
        meta.action_mask_ue
        if meta.action_mask_ue is not None
        else torch.ones((meta.n_ue,), dtype=torch.bool, device=device)
    ).to(device=device, dtype=torch.bool).unsqueeze(0)
    action_mask_cell_ue = meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).unsqueeze(0)
    ue_logits, _ue_valid = apply_ue_action_mask(
        actor_out["ue_logits"].unsqueeze(0),
        action_mask_ue,
        n_cell=meta.n_cell,
        action_mask_cell_ue=action_mask_cell_ue,
    )
    ue_logits = ue_logits.squeeze(0)
    if decode_mode == "sample":
        ue_class_raw = Categorical(logits=ue_logits).sample()
    else:
        ue_class_raw = ue_logits.argmax(dim=-1)
    ue_class, coverage_repair_count = _repair_ue_action_for_coverage(
        ue_class_raw,
        graph,
        actor_out,
        max_repairs_per_cell=int(slot_repair_max_per_cell),
        min_backlog_bytes=float(slot_repair_min_backlog_bytes),
    )
    with torch.inference_mode():
        prg_actor_out = actor.forward_graph(graph, slot_ue_class_for_prg=ue_class)
    use_candidate_prg = "candidate_prg_logits" in prg_actor_out
    budgeted_metrics: Dict[str, float] = {}
    tail_scores: torch.Tensor | None = None
    budget_mode = str(candidate_budgeted_select_mode).lower()
    if use_candidate_prg:
        packed = pack_prg_candidates(
            graph,
            max_candidates=max(0, int(prg_actor_out["candidate_prg_logits"].shape[-1]) - 1),
            fixed_width=True,
        )
        candidate_valid, _selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(
            graph,
            packed,
            ue_class,
        )
        prg_logits = prg_actor_out["candidate_prg_logits"].masked_fill(
            ~candidate_valid.to(device=device, dtype=torch.bool),
            -1.0e9,
        )
        blank_class = int(packed.blank_class_index)
        if budget_mode == "off" and float(candidate_budgeted_tail_bias_coef) > 0.0:
            prg_logits, budgeted_metrics, tail_scores, _rescue_scores = _apply_candidate_direct_service_bias(
                prg_logits,
                graph,
                packed,
                candidate_valid,
                tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                tail_bias_max=float(candidate_budgeted_tail_bias_max),
            )
    else:
        selected_slot_mask = build_slot_selection_mask(
            ue_class.unsqueeze(0),
            action_mask_ue,
            n_cell=meta.n_cell,
            action_mask_cell_ue=action_mask_cell_ue,
        )
        prg_logits, _prg_valid = apply_prg_action_mask(
            prg_actor_out["prg_logits"].unsqueeze(0),
            meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).reshape(1, meta.n_cell, meta.n_prg),
            n_cell=meta.n_cell,
            action_mask_cell_ue=action_mask_cell_ue,
            action_mask_ue=action_mask_ue,
            selected_slot_mask=selected_slot_mask,
        )
        prg_logits = prg_logits.squeeze(0)
        blank_class = int(meta.n_sched_ue)
    if decode_mode == "sample":
        prg_class = Categorical(logits=prg_logits).sample()
    else:
        prg_class = prg_logits.argmax(dim=-1)

    blank_budget_repair_count = 0
    decoded = None
    if use_candidate_prg:
        if budget_mode != "off":
            decoded, prg_class, budgeted_metrics = select_and_decode_budgeted_candidate_actions(
                graph,
                packed,
                candidate_valid,
                ue_class.detach().cpu(),
                prg_logits,
                mode=("sample" if budget_mode == "sample" else "greedy"),
                demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
                marginal_headroom_slack_bytes=float(candidate_marginal_headroom_slack_bytes),
                hard_cap=bool(int(candidate_budgeted_hard_cap)),
                usage_penalty=float(candidate_budgeted_usage_penalty),
                overcap_penalty=float(candidate_budgeted_overcap_penalty),
                first_service_bonus=float(candidate_budgeted_first_service_bonus),
                tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                tail_bias_max=float(candidate_budgeted_tail_bias_max),
                seq_state_weights=prg_actor_out.get("candidate_seq_state_weights"),
                seq_state_coef=float(candidate_seq_state_coef),
                seq_fail_risk_feature_offset=int(candidate_seq_fail_risk_feature_offset),
                safe_fill_max_per_cell=int(safe_fill_max_per_cell),
                safe_fill_min_quality_db=float(safe_fill_min_quality_db),
                safe_fill_max_quality_gap_db=float(safe_fill_max_quality_gap_db),
                safe_fill_max_prg_risk=float(safe_fill_max_prg_risk),
                safe_fill_min_backlog_bytes=float(safe_fill_min_backlog_bytes),
                safe_fill_max_ue_tb_err=float(safe_fill_max_ue_tb_err),
                headroom_rank_mode=str(candidate_headroom_rank_mode),
                candidate_rank_scores=prg_logits[:, :blank_class].detach(),
            )
        else:
            if tail_scores is not None:
                budgeted_metrics["candidate_budgeted_tail_bias_selected_mean"] = _candidate_direct_bias_selected_mean(
                    tail_scores,
                    prg_class,
                    int(blank_class),
                )
            prg_class, blank_budget_repair_count = apply_candidate_blank_budget(
                graph,
                packed,
                prg_logits,
                prg_class,
                max_blank_per_cell=int(candidate_blank_budget_max_per_cell),
                min_decision_logit=float(candidate_blank_budget_min_decision_logit),
            )
        if decoded is None:
            decoded = decode_candidate_actions_to_type0(
                graph,
                packed,
                ue_class.detach().cpu(),
                prg_class.detach().cpu(),
                safe_fill_max_per_cell=int(safe_fill_max_per_cell),
                safe_fill_min_quality_db=float(safe_fill_min_quality_db),
                safe_fill_max_quality_gap_db=float(safe_fill_max_quality_gap_db),
                safe_fill_max_prg_risk=float(safe_fill_max_prg_risk),
                safe_fill_min_backlog_bytes=float(safe_fill_min_backlog_bytes),
                safe_fill_max_ue_tb_err=float(safe_fill_max_ue_tb_err),
                demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
                marginal_headroom_slack_bytes=float(candidate_marginal_headroom_slack_bytes),
                headroom_rank_mode=str(candidate_headroom_rank_mode),
                candidate_rank_scores=prg_logits[:, :blank_class].detach(),
            )
    else:
        decoded = decode_joint_actions_to_type0(
            graph,
            ue_class.detach().cpu(),
            prg_class.detach().cpu(),
        )
    return {
        "action_ue_select": decoded.action_ue_select.cpu().numpy(),
        "action_prg_alloc": decoded.action_prg_alloc.cpu().numpy(),
        "blank_ratio": float((prg_class == blank_class).float().mean().item()),
        "fill_ratio": float(decoded.prg_chosen_mask.float().mean().item()),
        "cap_drop_count": float(decoded.cap_drop_count),
        "fallback_success_count": float(decoded.fallback_success_count),
        "safe_fill_count": float(decoded.safe_fill_count),
        "missed_safe_opportunity_count": float(decoded.missed_safe_opportunity_count),
        "blank_budget_repair_count": float(blank_budget_repair_count),
        "final_blank_count": float(decoded.final_blank_count),
        "post_decode_policy_blank_count": float(decoded.post_decode_policy_blank_count),
        "post_decode_no_valid_candidate_count": float(decoded.post_decode_no_valid_candidate_count),
        "post_decode_demand_cap_count": float(decoded.post_decode_demand_cap_count),
        "post_decode_no_slot_count": float(decoded.post_decode_no_slot_count),
        "post_decode_other_drop_count": float(decoded.post_decode_other_drop_count),
        "coverage_repair_count": float(coverage_repair_count),
        "budgeted_changed_count": float(budgeted_metrics.get("candidate_budgeted_changed_count", 0.0)),
        "budgeted_no_headroom_count": float(budgeted_metrics.get("candidate_budgeted_no_headroom_count", 0.0)),
        "budgeted_active_ue_count": float(budgeted_metrics.get("candidate_budgeted_active_ue_count", 0.0)),
        "budgeted_tail_bias_coef": float(budgeted_metrics.get("candidate_budgeted_tail_bias_coef", 0.0)),
        "budgeted_tail_bias_candidate_mean": float(
            budgeted_metrics.get("candidate_budgeted_tail_bias_candidate_mean", 0.0)
        ),
        "budgeted_tail_bias_selected_mean": float(
            budgeted_metrics.get("candidate_budgeted_tail_bias_selected_mean", 0.0)
        ),
        "budgeted_tail_bias_active_count": float(
            budgeted_metrics.get("candidate_budgeted_tail_bias_active_count", 0.0)
        ),
        "uniq_ue": float(torch.unique(decoded.chosen_ue_per_prg[decoded.chosen_ue_per_prg >= 0]).numel())
        if bool((decoded.chosen_ue_per_prg >= 0).any().item())
        else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a fixed HGraph checkpoint against the online bridge")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--socket-path", required=True)
    p.add_argument("--connect-timeout-s", type=float, default=20.0)
    p.add_argument("--episode-horizon", type=int, default=4096)
    p.add_argument("--max-steps", type=int, default=0, help="0 means run until env signals done")
    p.add_argument("--ue-prg-topk", type=int, default=None, help="Default: checkpoint args, else 6")
    p.add_argument("--ue-prg-diversity-extra", type=int, default=None, help="Default: checkpoint args, else 0")
    p.add_argument("--tb-err-ema-alpha", type=float, default=None, help="Default: checkpoint args, else 0.10")
    p.add_argument("--decode-mode", choices=["argmax", "sample"], default="argmax")
    p.add_argument("--slot-repair-max-per-cell", type=int, default=None, help="Default: checkpoint args, else 0")
    p.add_argument("--slot-repair-min-backlog-bytes", type=float, default=None, help="Default: checkpoint args, else 1.0")
    p.add_argument("--safe-fill-max-per-cell", type=int, default=None, help="Default: checkpoint args, else 0")
    p.add_argument("--safe-fill-min-quality-db", type=float, default=None, help="Default: checkpoint args, else -2.0")
    p.add_argument("--safe-fill-max-quality-gap-db", type=float, default=None, help="Default: checkpoint args, else 8.0")
    p.add_argument("--safe-fill-max-prg-risk", type=float, default=None, help="Default: checkpoint args, else 0.70")
    p.add_argument("--safe-fill-min-backlog-bytes", type=float, default=None, help="Default: checkpoint args, else 1.0")
    p.add_argument("--safe-fill-max-ue-tb-err", type=float, default=None, help="Default: checkpoint args, else 0.45")
    p.add_argument(
        "--candidate-blank-budget-max-per-cell",
        type=int,
        default=None,
        help="Default: checkpoint args, else -1. -1 disables deployment blank-budget repair.",
    )
    p.add_argument(
        "--candidate-blank-budget-min-decision-logit",
        type=float,
        default=None,
        help="Default: checkpoint args, else -1e9. Blank-minus-best-candidate threshold for budget slots.",
    )
    p.add_argument(
        "--candidate-decode-demand-slack-bytes",
        type=float,
        default=None,
        help="Default: checkpoint args, else 3000. Candidate decode demand-cap slack before leaving PRGs blank.",
    )
    p.add_argument(
        "--candidate-marginal-headroom-slack-bytes",
        type=float,
        default=None,
        help="Default: checkpoint args, else -1. Soft marginal-demand slack before excess PRGs are considered.",
    )
    p.add_argument(
        "--candidate-headroom-rank-mode",
        choices=["stable", "utility", "logit", "phy"],
        default=None,
        help="Default: env CUMAC_HGRAPH_CANDIDATE_HEADROOM_RANK_MODE, checkpoint args, else stable.",
    )
    p.add_argument(
        "--candidate-budgeted-select-mode",
        choices=["off", "greedy", "sample"],
        default=None,
        help="Default: checkpoint args, else off. Budget-aware PRG candidate selection using per-UE remaining headroom.",
    )
    p.add_argument("--candidate-budgeted-hard-cap", type=int, choices=[0, 1], default=None)
    p.add_argument("--candidate-budgeted-usage-penalty", type=float, default=None)
    p.add_argument("--candidate-budgeted-overcap-penalty", type=float, default=None)
    p.add_argument("--candidate-budgeted-first-service-bonus", type=float, default=None)
    p.add_argument("--candidate-budgeted-tail-bias-coef", type=float, default=None)
    p.add_argument("--candidate-budgeted-tail-bias-service-target", type=float, default=None)
    p.add_argument("--candidate-budgeted-tail-bias-ttl-guard-ms", type=float, default=None)
    p.add_argument("--candidate-budgeted-tail-bias-max", type=float, default=None)
    p.add_argument("--candidate-seq-state-coef", type=float, default=None)
    p.add_argument("--candidate-seq-fail-risk-feature-offset", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42, help="Bridge reset seed")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--print-every-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=1, help="1 writes eval_steps.jsonl; 0 writes summary only")
    p.add_argument(
        "--stats-warmup-steps",
        type=int,
        default=0,
        help="Ignore the first N evaluator steps in stats_mean_* summary fields; the environment still runs them.",
    )
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _set_seed(int(args.sample_seed))
    device = _pick_device(args.device)

    ckpt_path = Path(args.checkpoint).resolve()
    actor, ckpt = _load_actor_checkpoint(ckpt_path, device)
    ue_prg_topk = int(args.ue_prg_topk if args.ue_prg_topk is not None else _checkpoint_arg(ckpt, "ue_prg_topk", 6))
    ue_prg_diversity_extra = int(
        args.ue_prg_diversity_extra
        if args.ue_prg_diversity_extra is not None
        else _checkpoint_arg(ckpt, "ue_prg_diversity_extra", 0)
    )
    tb_err_ema_alpha = float(
        args.tb_err_ema_alpha if args.tb_err_ema_alpha is not None else _checkpoint_arg(ckpt, "tb_err_ema_alpha", 0.10)
    )
    slot_repair_max_per_cell = int(
        args.slot_repair_max_per_cell
        if args.slot_repair_max_per_cell is not None
        else _checkpoint_arg(ckpt, "slot_repair_max_per_cell", 0)
    )
    slot_repair_min_backlog_bytes = float(
        args.slot_repair_min_backlog_bytes
        if args.slot_repair_min_backlog_bytes is not None
        else _checkpoint_arg(ckpt, "slot_repair_min_backlog_bytes", 1.0)
    )
    safe_fill_max_per_cell = int(
        args.safe_fill_max_per_cell
        if args.safe_fill_max_per_cell is not None
        else _checkpoint_arg(ckpt, "safe_fill_max_per_cell", 0)
    )
    safe_fill_min_quality_db = float(
        args.safe_fill_min_quality_db
        if args.safe_fill_min_quality_db is not None
        else _checkpoint_arg(ckpt, "safe_fill_min_quality_db", -2.0)
    )
    safe_fill_max_quality_gap_db = float(
        args.safe_fill_max_quality_gap_db
        if args.safe_fill_max_quality_gap_db is not None
        else _checkpoint_arg(ckpt, "safe_fill_max_quality_gap_db", 8.0)
    )
    safe_fill_max_prg_risk = float(
        args.safe_fill_max_prg_risk
        if args.safe_fill_max_prg_risk is not None
        else _checkpoint_arg(ckpt, "safe_fill_max_prg_risk", 0.70)
    )
    safe_fill_min_backlog_bytes = float(
        args.safe_fill_min_backlog_bytes
        if args.safe_fill_min_backlog_bytes is not None
        else _checkpoint_arg(ckpt, "safe_fill_min_backlog_bytes", 1.0)
    )
    safe_fill_max_ue_tb_err = float(
        args.safe_fill_max_ue_tb_err
        if args.safe_fill_max_ue_tb_err is not None
        else _checkpoint_arg(ckpt, "safe_fill_max_ue_tb_err", 0.45)
    )
    candidate_blank_budget_max_per_cell = int(
        args.candidate_blank_budget_max_per_cell
        if args.candidate_blank_budget_max_per_cell is not None
        else _checkpoint_arg(ckpt, "candidate_blank_budget_max_per_cell", -1)
    )
    candidate_blank_budget_min_decision_logit = float(
        args.candidate_blank_budget_min_decision_logit
        if args.candidate_blank_budget_min_decision_logit is not None
        else _checkpoint_arg(ckpt, "candidate_blank_budget_min_decision_logit", -1.0e9)
    )
    candidate_decode_demand_slack_bytes = float(
        args.candidate_decode_demand_slack_bytes
        if args.candidate_decode_demand_slack_bytes is not None
        else _checkpoint_arg(ckpt, "candidate_decode_demand_slack_bytes", 3000.0)
    )
    candidate_marginal_headroom_slack_bytes = float(
        args.candidate_marginal_headroom_slack_bytes
        if args.candidate_marginal_headroom_slack_bytes is not None
        else _checkpoint_arg(ckpt, "candidate_marginal_headroom_slack_bytes", -1.0)
    )
    candidate_headroom_rank_mode = str(
        args.candidate_headroom_rank_mode
        if args.candidate_headroom_rank_mode is not None
        else os.environ.get(
            "CUMAC_HGRAPH_CANDIDATE_HEADROOM_RANK_MODE",
            _checkpoint_arg(ckpt, "candidate_headroom_rank_mode", "stable"),
        )
    ).lower()
    candidate_budgeted_select_mode = str(
        args.candidate_budgeted_select_mode
        if args.candidate_budgeted_select_mode is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_select_mode", "off")
    ).lower()
    candidate_budgeted_hard_cap = int(
        args.candidate_budgeted_hard_cap
        if args.candidate_budgeted_hard_cap is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_hard_cap", 1)
    )
    candidate_budgeted_usage_penalty = float(
        args.candidate_budgeted_usage_penalty
        if args.candidate_budgeted_usage_penalty is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_usage_penalty", 0.35)
    )
    candidate_budgeted_overcap_penalty = float(
        args.candidate_budgeted_overcap_penalty
        if args.candidate_budgeted_overcap_penalty is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_overcap_penalty", 8.0)
    )
    candidate_budgeted_first_service_bonus = float(
        args.candidate_budgeted_first_service_bonus
        if args.candidate_budgeted_first_service_bonus is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_first_service_bonus", 0.0)
    )
    candidate_budgeted_tail_bias_coef = float(
        args.candidate_budgeted_tail_bias_coef
        if args.candidate_budgeted_tail_bias_coef is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_tail_bias_coef", 0.0)
    )
    candidate_budgeted_tail_bias_service_target = float(
        args.candidate_budgeted_tail_bias_service_target
        if args.candidate_budgeted_tail_bias_service_target is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_tail_bias_service_target", 0.75)
    )
    candidate_budgeted_tail_bias_ttl_guard_ms = float(
        args.candidate_budgeted_tail_bias_ttl_guard_ms
        if args.candidate_budgeted_tail_bias_ttl_guard_ms is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_tail_bias_ttl_guard_ms", 20.0)
    )
    candidate_budgeted_tail_bias_max = float(
        args.candidate_budgeted_tail_bias_max
        if args.candidate_budgeted_tail_bias_max is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_tail_bias_max", 1.5)
    )
    candidate_seq_state_coef = float(
        args.candidate_seq_state_coef
        if args.candidate_seq_state_coef is not None
        else _checkpoint_arg(ckpt, "candidate_seq_state_coef", 1.0)
    )
    candidate_seq_fail_risk_feature_offset = int(
        args.candidate_seq_fail_risk_feature_offset
        if args.candidate_seq_fail_risk_feature_offset is not None
        else _checkpoint_arg(ckpt, "candidate_seq_fail_risk_feature_offset", -1)
    )
    if ue_prg_topk <= 0:
        raise ValueError("--ue-prg-topk must be > 0")
    if ue_prg_diversity_extra < 0:
        raise ValueError("--ue-prg-diversity-extra must be >= 0")
    if slot_repair_max_per_cell < 0:
        raise ValueError("--slot-repair-max-per-cell must be >= 0")
    if safe_fill_max_per_cell < 0:
        raise ValueError("--safe-fill-max-per-cell must be >= 0")
    if candidate_blank_budget_max_per_cell < -1:
        raise ValueError("--candidate-blank-budget-max-per-cell must be >= -1")
    if candidate_decode_demand_slack_bytes < 0.0:
        raise ValueError("--candidate-decode-demand-slack-bytes must be >= 0")
    if candidate_marginal_headroom_slack_bytes < -1.0:
        raise ValueError("--candidate-marginal-headroom-slack-bytes must be >= -1")
    if candidate_headroom_rank_mode not in ("stable", "utility", "logit", "phy"):
        raise ValueError("--candidate-headroom-rank-mode must be stable, utility, logit, or phy")
    if candidate_budgeted_select_mode not in ("off", "greedy", "sample"):
        raise ValueError("--candidate-budgeted-select-mode must be off, greedy, or sample")
    if candidate_budgeted_hard_cap not in (0, 1):
        raise ValueError("--candidate-budgeted-hard-cap must be 0 or 1")
    if candidate_budgeted_usage_penalty < 0.0:
        raise ValueError("--candidate-budgeted-usage-penalty must be >= 0")
    if candidate_budgeted_overcap_penalty < 0.0:
        raise ValueError("--candidate-budgeted-overcap-penalty must be >= 0")
    if candidate_budgeted_tail_bias_coef < 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-coef must be >= 0")
    if candidate_budgeted_tail_bias_service_target <= 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-service-target must be > 0")
    if candidate_budgeted_tail_bias_ttl_guard_ms <= 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-ttl-guard-ms must be > 0")
    if candidate_budgeted_tail_bias_max < 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-max must be >= 0")
    if int(args.save_steps) not in (0, 1):
        raise ValueError("--save-steps must be 0 or 1")
    if int(args.stats_warmup_steps) < 0:
        raise ValueError("--stats-warmup-steps must be >= 0")

    out_dir = None
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_builder = FeatureSnapshotBuilder(
        SnapshotBuilderConfig(tb_err_ema_alpha=float(tb_err_ema_alpha))
    )
    up_edge_attr_dim = int(getattr(actor.cfg, "up_edge_attr_dim", 6))
    ckpt_anchor = _checkpoint_arg(ckpt, "include_pfq_score_anchor", None)
    if ckpt_anchor is not None:
        inferred_pfq_anchor = bool(int(ckpt_anchor))
    else:
        inferred_pfq_anchor = bool(
            int(
                getattr(
                    actor.cfg,
                    "has_pfq_score_anchor",
                    int(up_edge_attr_dim in (6, 9, 11, 14, 20, 30)),
                )
            )
        )
    inferred_risk_feature_dim = max(0, up_edge_attr_dim - 5 - int(inferred_pfq_anchor))
    inferred_candidate_risk_features = inferred_risk_feature_dim > 0
    include_candidate_risk_features = bool(
        int(_checkpoint_arg(ckpt, "include_candidate_risk_features", int(inferred_candidate_risk_features)))
    )
    candidate_risk_feature_dim = inferred_risk_feature_dim if include_candidate_risk_features else 0
    include_pfq_score_anchor = bool(
        int(_checkpoint_arg(ckpt, "include_pfq_score_anchor", int(inferred_pfq_anchor)))
    )
    edge_generator = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            ue_prg_topk=int(ue_prg_topk),
            ue_prg_diversity_extra=int(ue_prg_diversity_extra),
            include_pfq_score_anchor=include_pfq_score_anchor,
            include_candidate_risk_features=include_candidate_risk_features,
            candidate_risk_feature_dim=candidate_risk_feature_dim,
        )
    )
    env = BridgeEnvClient(
        BridgeEnvConfig(
            socket_path=str(args.socket_path),
            connect_timeout_s=float(args.connect_timeout_s),
        )
    )

    obs, _reward0, info = env.reset(seed=int(args.seed), episode_horizon=int(args.episode_horizon))
    step_idx = 0
    rows = []
    tail_rows = deque(maxlen=max(1, int(args.print_every_steps) if int(args.print_every_steps) > 0 else 1))
    sums = {
        "goodput_mbps": 0.0,
        "tb_err_rate": 0.0,
        "expiry_drop_rate": 0.0,
        "prg_utilization_ratio": 0.0,
        "prg_reuse_ratio": 0.0,
        "packet_completed_count": 0.0,
        "packet_delivered_bits": 0.0,
        "packet_system_time_ms": 0.0,
        "packet_delay_mean_ms": 0.0,
        "packet_effective_service_rate_mbps": 0.0,
        "packet_effective_service_rate_per_packet_mean_mbps": 0.0,
        "blank_ratio": 0.0,
        "fill_ratio": 0.0,
        "cap_drop_count": 0.0,
        "fallback_success_count": 0.0,
        "safe_fill_count": 0.0,
        "missed_safe_opportunity_count": 0.0,
        "blank_budget_repair_count": 0.0,
        "final_blank_count": 0.0,
        "post_decode_policy_blank_count": 0.0,
        "post_decode_no_valid_candidate_count": 0.0,
        "post_decode_demand_cap_count": 0.0,
        "post_decode_no_slot_count": 0.0,
        "post_decode_other_drop_count": 0.0,
        "uniq_ue": 0.0,
        "coverage_repair_count": 0.0,
        "budgeted_changed_count": 0.0,
        "budgeted_no_headroom_count": 0.0,
        "budgeted_active_ue_count": 0.0,
        "budgeted_tail_bias_candidate_mean": 0.0,
        "budgeted_tail_bias_selected_mean": 0.0,
        "budgeted_tail_bias_active_count": 0.0,
    }
    stats_sums = {key: 0.0 for key in sums}
    stats_count = 0
    try:
        while True:
            graph = _graph_to_device(edge_generator.build_graph(snapshot_builder.build(obs)), device)
            selected = _select_joint_action(
                actor,
                graph,
                decode_mode=str(args.decode_mode),
                slot_repair_max_per_cell=int(slot_repair_max_per_cell),
                slot_repair_min_backlog_bytes=float(slot_repair_min_backlog_bytes),
                safe_fill_max_per_cell=int(safe_fill_max_per_cell),
                safe_fill_min_quality_db=float(safe_fill_min_quality_db),
                safe_fill_max_quality_gap_db=float(safe_fill_max_quality_gap_db),
                safe_fill_max_prg_risk=float(safe_fill_max_prg_risk),
                safe_fill_min_backlog_bytes=float(safe_fill_min_backlog_bytes),
                safe_fill_max_ue_tb_err=float(safe_fill_max_ue_tb_err),
                candidate_blank_budget_max_per_cell=int(candidate_blank_budget_max_per_cell),
                candidate_blank_budget_min_decision_logit=float(candidate_blank_budget_min_decision_logit),
                candidate_decode_demand_slack_bytes=float(candidate_decode_demand_slack_bytes),
                candidate_marginal_headroom_slack_bytes=float(candidate_marginal_headroom_slack_bytes),
                candidate_headroom_rank_mode=str(candidate_headroom_rank_mode),
                candidate_budgeted_select_mode=str(candidate_budgeted_select_mode),
                candidate_budgeted_hard_cap=int(candidate_budgeted_hard_cap),
                candidate_budgeted_usage_penalty=float(candidate_budgeted_usage_penalty),
                candidate_budgeted_overcap_penalty=float(candidate_budgeted_overcap_penalty),
                candidate_budgeted_first_service_bonus=float(candidate_budgeted_first_service_bonus),
                candidate_budgeted_tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                candidate_budgeted_tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                candidate_budgeted_tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                candidate_budgeted_tail_bias_max=float(candidate_budgeted_tail_bias_max),
                candidate_seq_state_coef=float(candidate_seq_state_coef),
                candidate_seq_fail_risk_feature_offset=int(candidate_seq_fail_risk_feature_offset),
            )
            next_obs, raw_reward, done, next_info = env.step(
                selected["action_ue_select"],
                selected["action_prg_alloc"],
            )
            reward_metrics = extract_reward_metrics(next_info)
            row = {
                "step": step_idx,
                "tti": int(next_info.get("tti", step_idx)),
                "reward_raw_env": float(raw_reward),
                "goodput_mbps": float(reward_metrics.get("goodput_mbps", 0.0)),
                "tb_err_rate": float(reward_metrics.get("tb_err_rate", 0.0)),
                "expiry_drop_rate": float(reward_metrics.get("expiry_drop_rate", 0.0)),
                "prg_utilization_ratio": float(reward_metrics.get("prg_utilization_ratio", 0.0)),
                "prg_reuse_ratio": float(reward_metrics.get("prg_reuse_ratio", 0.0)),
                "packet_completed_count": float(reward_metrics.get("packet_completed_count", 0.0)),
                "packet_delivered_bits": float(reward_metrics.get("packet_delivered_bits", 0.0)),
                "packet_system_time_ms": float(reward_metrics.get("packet_system_time_ms", 0.0)),
                "packet_delay_mean_ms": float(reward_metrics.get("packet_delay_mean_ms", 0.0)),
                "packet_effective_service_rate_mbps": float(
                    reward_metrics.get("packet_effective_service_rate_mbps", 0.0)
                ),
                "packet_effective_service_rate_per_packet_mean_mbps": float(
                    reward_metrics.get("packet_effective_service_rate_per_packet_mean_mbps", 0.0)
                ),
                "blank_ratio": float(selected["blank_ratio"]),
                "fill_ratio": float(selected["fill_ratio"]),
                "cap_drop_count": float(selected["cap_drop_count"]),
                "fallback_success_count": float(selected["fallback_success_count"]),
                "safe_fill_count": float(selected["safe_fill_count"]),
                "missed_safe_opportunity_count": float(selected["missed_safe_opportunity_count"]),
                "blank_budget_repair_count": float(selected["blank_budget_repair_count"]),
                "final_blank_count": float(selected["final_blank_count"]),
                "post_decode_policy_blank_count": float(selected["post_decode_policy_blank_count"]),
                "post_decode_no_valid_candidate_count": float(selected["post_decode_no_valid_candidate_count"]),
                "post_decode_demand_cap_count": float(selected["post_decode_demand_cap_count"]),
                "post_decode_no_slot_count": float(selected["post_decode_no_slot_count"]),
                "post_decode_other_drop_count": float(selected["post_decode_other_drop_count"]),
                "uniq_ue": float(selected["uniq_ue"]),
                "coverage_repair_count": float(selected["coverage_repair_count"]),
                "budgeted_changed_count": float(selected["budgeted_changed_count"]),
                "budgeted_no_headroom_count": float(selected["budgeted_no_headroom_count"]),
                "budgeted_active_ue_count": float(selected["budgeted_active_ue_count"]),
                "budgeted_tail_bias_candidate_mean": float(selected["budgeted_tail_bias_candidate_mean"]),
                "budgeted_tail_bias_selected_mean": float(selected["budgeted_tail_bias_selected_mean"]),
                "budgeted_tail_bias_active_count": float(selected["budgeted_tail_bias_active_count"]),
            }
            if int(args.save_steps) == 1:
                rows.append(row)
            tail_rows.append(row)
            for key in sums:
                sums[key] += float(row[key])
            if step_idx >= int(args.stats_warmup_steps):
                for key in stats_sums:
                    stats_sums[key] += float(row[key])
                stats_count += 1
            step_idx += 1

            if int(args.print_every_steps) > 0 and (
                (step_idx % int(args.print_every_steps)) == 0 or done or (args.max_steps > 0 and step_idx >= args.max_steps)
            ):
                tail = list(tail_rows)
                print(
                    f"[hgraph-eval] tti={step_idx:04d} "
                    f"goodput={float(np.mean([r['goodput_mbps'] for r in tail])):.4f} "
                    f"tb_err={float(np.mean([r['tb_err_rate'] for r in tail])):.4f} "
                    f"blank={float(np.mean([r['blank_ratio'] for r in tail])):.4f} "
                    f"fill={float(np.mean([r['fill_ratio'] for r in tail])):.4f} "
                    f"capDrop={float(np.mean([r['cap_drop_count'] for r in tail])):.2f} "
                    f"fbOK={float(np.mean([r['fallback_success_count'] for r in tail])):.2f} "
                    f"safeFill={float(np.mean([r['safe_fill_count'] for r in tail])):.2f} "
                    f"missedSafe={float(np.mean([r['missed_safe_opportunity_count'] for r in tail])):.2f} "
                    f"blankBudget={float(np.mean([r['blank_budget_repair_count'] for r in tail])):.2f} "
                    f"blankCnt={float(np.mean([r['final_blank_count'] for r in tail])):.2f} "
                    f"dropCap={float(np.mean([r['post_decode_demand_cap_count'] for r in tail])):.2f} "
                    f"dropSlot={float(np.mean([r['post_decode_no_slot_count'] for r in tail])):.2f} "
                    f"dropOth={float(np.mean([r['post_decode_other_drop_count'] for r in tail])):.2f} "
                    f"budgChg={float(np.mean([r['budgeted_changed_count'] for r in tail])):.2f} "
                    f"budgNoHr={float(np.mean([r['budgeted_no_headroom_count'] for r in tail])):.2f} "
                    f"tailBiasSel={float(np.mean([r['budgeted_tail_bias_selected_mean'] for r in tail])):.3f} "
                    f"uniq_ue={float(np.mean([r['uniq_ue'] for r in tail])):.2f}"
                )

            if done or (int(args.max_steps) > 0 and step_idx >= int(args.max_steps)):
                break
            obs, info = next_obs, next_info

    finally:
        env.close()

    denom = max(step_idx, 1)
    def _stats_mean(key: str):
        return None if stats_count <= 0 else float(stats_sums[key] / float(stats_count))

    def _stats_sum(key: str):
        return None if stats_count <= 0 else float(stats_sums[key])

    mean_packet_delay_weighted = (
        float(sums["packet_system_time_ms"] / max(sums["packet_completed_count"], 1.0))
        if sums["packet_completed_count"] > 0.0
        else 0.0
    )
    stats_packet_delay_weighted = (
        None
        if stats_count <= 0 or stats_sums["packet_completed_count"] <= 0.0
        else float(stats_sums["packet_system_time_ms"] / stats_sums["packet_completed_count"])
    )
    mean_packet_rate_weighted = (
        float((sums["packet_delivered_bits"] / 1000.0) / max(sums["packet_system_time_ms"], 1.0e-6))
        if sums["packet_system_time_ms"] > 0.0
        else 0.0
    )
    stats_packet_rate_weighted = (
        None
        if stats_count <= 0 or stats_sums["packet_system_time_ms"] <= 0.0
        else float((stats_sums["packet_delivered_bits"] / 1000.0) / stats_sums["packet_system_time_ms"])
    )

    summary = {
        "checkpoint": str(ckpt_path),
        "decode_mode": str(args.decode_mode),
        "ue_prg_topk": int(ue_prg_topk),
        "ue_prg_diversity_extra": int(ue_prg_diversity_extra),
        "include_pfq_score_anchor": int(include_pfq_score_anchor),
        "include_candidate_risk_features": int(include_candidate_risk_features),
        "candidate_risk_feature_dim": int(candidate_risk_feature_dim),
        "tb_err_ema_alpha": float(tb_err_ema_alpha),
        "slot_repair_max_per_cell": int(slot_repair_max_per_cell),
        "slot_repair_min_backlog_bytes": float(slot_repair_min_backlog_bytes),
        "safe_fill_max_per_cell": int(safe_fill_max_per_cell),
        "safe_fill_min_quality_db": float(safe_fill_min_quality_db),
        "safe_fill_max_quality_gap_db": float(safe_fill_max_quality_gap_db),
        "safe_fill_max_prg_risk": float(safe_fill_max_prg_risk),
        "safe_fill_min_backlog_bytes": float(safe_fill_min_backlog_bytes),
        "safe_fill_max_ue_tb_err": float(safe_fill_max_ue_tb_err),
        "candidate_blank_budget_max_per_cell": int(candidate_blank_budget_max_per_cell),
        "candidate_blank_budget_min_decision_logit": float(candidate_blank_budget_min_decision_logit),
        "candidate_decode_demand_slack_bytes": float(candidate_decode_demand_slack_bytes),
        "candidate_marginal_headroom_slack_bytes": float(candidate_marginal_headroom_slack_bytes),
        "candidate_headroom_rank_mode": str(candidate_headroom_rank_mode),
        "candidate_budgeted_select_mode": str(candidate_budgeted_select_mode),
        "candidate_budgeted_hard_cap": int(candidate_budgeted_hard_cap),
        "candidate_budgeted_usage_penalty": float(candidate_budgeted_usage_penalty),
        "candidate_budgeted_overcap_penalty": float(candidate_budgeted_overcap_penalty),
        "candidate_budgeted_first_service_bonus": float(candidate_budgeted_first_service_bonus),
        "candidate_budgeted_tail_bias_coef": float(candidate_budgeted_tail_bias_coef),
        "candidate_budgeted_tail_bias_service_target": float(candidate_budgeted_tail_bias_service_target),
        "candidate_budgeted_tail_bias_ttl_guard_ms": float(candidate_budgeted_tail_bias_ttl_guard_ms),
        "candidate_budgeted_tail_bias_max": float(candidate_budgeted_tail_bias_max),
        "candidate_seq_state_coef": float(candidate_seq_state_coef),
        "candidate_seq_fail_risk_feature_offset": int(candidate_seq_fail_risk_feature_offset),
        "save_steps": int(args.save_steps),
        "completed_steps": int(step_idx),
        "stats_warmup_steps": int(args.stats_warmup_steps),
        "stats_count": int(stats_count),
        "mean_goodput_mbps": float(sums["goodput_mbps"] / denom),
        "mean_tb_err_rate": float(sums["tb_err_rate"] / denom),
        "mean_expiry_drop_rate": float(sums["expiry_drop_rate"] / denom),
        "mean_prg_utilization_ratio": float(sums["prg_utilization_ratio"] / denom),
        "mean_prg_reuse_ratio": float(sums["prg_reuse_ratio"] / denom),
        "mean_packet_completed_count": float(sums["packet_completed_count"] / denom),
        "mean_packet_delivered_bits": float(sums["packet_delivered_bits"] / denom),
        "mean_packet_system_time_ms": float(sums["packet_system_time_ms"] / denom),
        "mean_packet_delay_mean_ms": float(sums["packet_delay_mean_ms"] / denom),
        "mean_weighted_packet_delay_mean_ms": mean_packet_delay_weighted,
        "mean_packet_effective_service_rate_mbps": float(
            sums["packet_effective_service_rate_mbps"] / denom
        ),
        "mean_weighted_packet_effective_service_rate_mbps": mean_packet_rate_weighted,
        "mean_packet_effective_service_rate_per_packet_mean_mbps": float(
            sums["packet_effective_service_rate_per_packet_mean_mbps"] / denom
        ),
        "mean_blank_ratio": float(sums["blank_ratio"] / denom),
        "mean_fill_ratio": float(sums["fill_ratio"] / denom),
        "mean_cap_drop_count": float(sums["cap_drop_count"] / denom),
        "mean_fallback_success_count": float(sums["fallback_success_count"] / denom),
        "mean_safe_fill_count": float(sums["safe_fill_count"] / denom),
        "mean_missed_safe_opportunity_count": float(sums["missed_safe_opportunity_count"] / denom),
        "mean_blank_budget_repair_count": float(sums["blank_budget_repair_count"] / denom),
        "mean_final_blank_count": float(sums["final_blank_count"] / denom),
        "mean_post_decode_policy_blank_count": float(sums["post_decode_policy_blank_count"] / denom),
        "mean_post_decode_no_valid_candidate_count": float(sums["post_decode_no_valid_candidate_count"] / denom),
        "mean_post_decode_demand_cap_count": float(sums["post_decode_demand_cap_count"] / denom),
        "mean_post_decode_no_slot_count": float(sums["post_decode_no_slot_count"] / denom),
        "mean_post_decode_other_drop_count": float(sums["post_decode_other_drop_count"] / denom),
        "mean_uniq_ue": float(sums["uniq_ue"] / denom),
        "mean_coverage_repair_count": float(sums["coverage_repair_count"] / denom),
        "mean_budgeted_changed_count": float(sums["budgeted_changed_count"] / denom),
        "mean_budgeted_no_headroom_count": float(sums["budgeted_no_headroom_count"] / denom),
        "mean_budgeted_active_ue_count": float(sums["budgeted_active_ue_count"] / denom),
        "mean_budgeted_tail_bias_candidate_mean": float(sums["budgeted_tail_bias_candidate_mean"] / denom),
        "mean_budgeted_tail_bias_selected_mean": float(sums["budgeted_tail_bias_selected_mean"] / denom),
        "mean_budgeted_tail_bias_active_count": float(sums["budgeted_tail_bias_active_count"] / denom),
        "stats_mean_goodput_mbps": _stats_mean("goodput_mbps"),
        "stats_mean_tb_err_rate": _stats_mean("tb_err_rate"),
        "stats_mean_expiry_drop_rate": _stats_mean("expiry_drop_rate"),
        "stats_mean_prg_utilization_ratio": _stats_mean("prg_utilization_ratio"),
        "stats_mean_prg_reuse_ratio": _stats_mean("prg_reuse_ratio"),
        "stats_sum_packet_completed_count": _stats_sum("packet_completed_count"),
        "stats_sum_packet_delivered_bits": _stats_sum("packet_delivered_bits"),
        "stats_sum_packet_system_time_ms": _stats_sum("packet_system_time_ms"),
        "stats_mean_packet_completed_count": _stats_mean("packet_completed_count"),
        "stats_mean_packet_delivered_bits": _stats_mean("packet_delivered_bits"),
        "stats_mean_packet_system_time_ms": _stats_mean("packet_system_time_ms"),
        "stats_mean_packet_delay_mean_ms": _stats_mean("packet_delay_mean_ms"),
        "stats_weighted_packet_delay_mean_ms": stats_packet_delay_weighted,
        "stats_mean_packet_effective_service_rate_mbps": _stats_mean(
            "packet_effective_service_rate_mbps"
        ),
        "stats_weighted_packet_effective_service_rate_mbps": stats_packet_rate_weighted,
        "stats_mean_packet_effective_service_rate_per_packet_mean_mbps": _stats_mean(
            "packet_effective_service_rate_per_packet_mean_mbps"
        ),
        "stats_mean_blank_ratio": _stats_mean("blank_ratio"),
        "stats_mean_fill_ratio": _stats_mean("fill_ratio"),
        "stats_mean_cap_drop_count": _stats_mean("cap_drop_count"),
        "stats_mean_fallback_success_count": _stats_mean("fallback_success_count"),
        "stats_mean_safe_fill_count": _stats_mean("safe_fill_count"),
        "stats_mean_missed_safe_opportunity_count": _stats_mean("missed_safe_opportunity_count"),
        "stats_mean_blank_budget_repair_count": _stats_mean("blank_budget_repair_count"),
        "stats_mean_final_blank_count": _stats_mean("final_blank_count"),
        "stats_mean_post_decode_policy_blank_count": _stats_mean("post_decode_policy_blank_count"),
        "stats_mean_post_decode_no_valid_candidate_count": _stats_mean("post_decode_no_valid_candidate_count"),
        "stats_mean_post_decode_demand_cap_count": _stats_mean("post_decode_demand_cap_count"),
        "stats_mean_post_decode_no_slot_count": _stats_mean("post_decode_no_slot_count"),
        "stats_mean_post_decode_other_drop_count": _stats_mean("post_decode_other_drop_count"),
        "stats_mean_uniq_ue": _stats_mean("uniq_ue"),
        "stats_mean_coverage_repair_count": _stats_mean("coverage_repair_count"),
        "stats_mean_budgeted_changed_count": _stats_mean("budgeted_changed_count"),
        "stats_mean_budgeted_no_headroom_count": _stats_mean("budgeted_no_headroom_count"),
        "stats_mean_budgeted_active_ue_count": _stats_mean("budgeted_active_ue_count"),
        "stats_mean_budgeted_tail_bias_candidate_mean": _stats_mean("budgeted_tail_bias_candidate_mean"),
        "stats_mean_budgeted_tail_bias_selected_mean": _stats_mean("budgeted_tail_bias_selected_mean"),
        "stats_mean_budgeted_tail_bias_active_count": _stats_mean("budgeted_tail_bias_active_count"),
        "actor_config": ckpt.get("actor_config", {}),
    }

    if out_dir is not None:
        (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if int(args.save_steps) == 1:
            with open(out_dir / "eval_steps.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
