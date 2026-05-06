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
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.distributions import Categorical

from training.stageb_hgraph.bridge_env import BridgeEnvClient, BridgeEnvConfig
from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from training.gnnrl.masks import apply_prg_action_mask, apply_ue_action_mask, build_slot_selection_mask
from training.stageb_hgraph.graph_ops import decode_joint_actions_to_type0
from training.stageb_hgraph.model import HGraphModelConfig, StageBHGraphPolicy, build_model_from_config
from training.stageb_hgraph.online_imitation import _pick_device, _set_seed
from training.stageb_hgraph.online_ppo import _graph_to_device, _repair_ue_action_for_coverage
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
    selected_slot_mask = build_slot_selection_mask(
        ue_class.unsqueeze(0),
        action_mask_ue,
        n_cell=meta.n_cell,
        action_mask_cell_ue=action_mask_cell_ue,
    )
    prg_logits, prg_valid = apply_prg_action_mask(
        prg_actor_out["prg_logits"].unsqueeze(0),
        meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).reshape(1, meta.n_cell, meta.n_prg),
        n_cell=meta.n_cell,
        action_mask_cell_ue=action_mask_cell_ue,
        action_mask_ue=action_mask_ue,
        selected_slot_mask=selected_slot_mask,
    )
    prg_logits = prg_logits.squeeze(0)
    if decode_mode == "sample":
        prg_class = Categorical(logits=prg_logits).sample()
    else:
        prg_class = prg_logits.argmax(dim=-1)

    decoded = decode_joint_actions_to_type0(
        graph,
        ue_class.detach().cpu(),
        prg_class.detach().cpu(),
    )
    return {
        "action_ue_select": decoded.action_ue_select.cpu().numpy(),
        "action_prg_alloc": decoded.action_prg_alloc.cpu().numpy(),
        "blank_ratio": float((prg_class == meta.n_sched_ue).float().mean().item()),
        "fill_ratio": float(decoded.prg_chosen_mask.float().mean().item()),
        "cap_drop_count": float(decoded.cap_drop_count),
        "fallback_success_count": float(decoded.fallback_success_count),
        "final_blank_count": float(decoded.final_blank_count),
        "coverage_repair_count": float(coverage_repair_count),
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
    if ue_prg_topk <= 0:
        raise ValueError("--ue-prg-topk must be > 0")
    if ue_prg_diversity_extra < 0:
        raise ValueError("--ue-prg-diversity-extra must be >= 0")
    if slot_repair_max_per_cell < 0:
        raise ValueError("--slot-repair-max-per-cell must be >= 0")
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
    edge_generator = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            ue_prg_topk=int(ue_prg_topk),
            ue_prg_diversity_extra=int(ue_prg_diversity_extra),
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
        "blank_ratio": 0.0,
        "fill_ratio": 0.0,
        "cap_drop_count": 0.0,
        "fallback_success_count": 0.0,
        "final_blank_count": 0.0,
        "uniq_ue": 0.0,
        "coverage_repair_count": 0.0,
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
                "blank_ratio": float(selected["blank_ratio"]),
                "fill_ratio": float(selected["fill_ratio"]),
                "cap_drop_count": float(selected["cap_drop_count"]),
                "fallback_success_count": float(selected["fallback_success_count"]),
                "final_blank_count": float(selected["final_blank_count"]),
                "uniq_ue": float(selected["uniq_ue"]),
                "coverage_repair_count": float(selected["coverage_repair_count"]),
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
                    f"blankCnt={float(np.mean([r['final_blank_count'] for r in tail])):.2f} "
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

    summary = {
        "checkpoint": str(ckpt_path),
        "decode_mode": str(args.decode_mode),
        "ue_prg_topk": int(ue_prg_topk),
        "ue_prg_diversity_extra": int(ue_prg_diversity_extra),
        "tb_err_ema_alpha": float(tb_err_ema_alpha),
        "slot_repair_max_per_cell": int(slot_repair_max_per_cell),
        "slot_repair_min_backlog_bytes": float(slot_repair_min_backlog_bytes),
        "save_steps": int(args.save_steps),
        "completed_steps": int(step_idx),
        "stats_warmup_steps": int(args.stats_warmup_steps),
        "stats_count": int(stats_count),
        "mean_goodput_mbps": float(sums["goodput_mbps"] / denom),
        "mean_tb_err_rate": float(sums["tb_err_rate"] / denom),
        "mean_blank_ratio": float(sums["blank_ratio"] / denom),
        "mean_fill_ratio": float(sums["fill_ratio"] / denom),
        "mean_cap_drop_count": float(sums["cap_drop_count"] / denom),
        "mean_fallback_success_count": float(sums["fallback_success_count"] / denom),
        "mean_final_blank_count": float(sums["final_blank_count"] / denom),
        "mean_uniq_ue": float(sums["uniq_ue"] / denom),
        "mean_coverage_repair_count": float(sums["coverage_repair_count"] / denom),
        "stats_mean_goodput_mbps": _stats_mean("goodput_mbps"),
        "stats_mean_tb_err_rate": _stats_mean("tb_err_rate"),
        "stats_mean_expiry_drop_rate": _stats_mean("expiry_drop_rate"),
        "stats_mean_prg_utilization_ratio": _stats_mean("prg_utilization_ratio"),
        "stats_mean_prg_reuse_ratio": _stats_mean("prg_reuse_ratio"),
        "stats_mean_blank_ratio": _stats_mean("blank_ratio"),
        "stats_mean_fill_ratio": _stats_mean("fill_ratio"),
        "stats_mean_cap_drop_count": _stats_mean("cap_drop_count"),
        "stats_mean_fallback_success_count": _stats_mean("fallback_success_count"),
        "stats_mean_final_blank_count": _stats_mean("final_blank_count"),
        "stats_mean_uniq_ue": _stats_mean("uniq_ue"),
        "stats_mean_coverage_repair_count": _stats_mean("coverage_repair_count"),
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
