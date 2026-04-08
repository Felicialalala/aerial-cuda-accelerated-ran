#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training curves for a Stage-B online PPO run directory.")
    p.add_argument("--run-dir", required=True, help="Checkpoint/run directory containing online_ppo_summary.json")
    p.add_argument("--smooth-window", type=int, default=9, help="Moving-average window for curves")
    p.add_argument("--output-png", default="", help="Output PNG path (default: <run-dir>/online_ppo_curves_key_metrics.png)")
    p.add_argument("--output-svg", default="", help="Optional output SVG path")
    p.add_argument("--title", default="", help="Optional figure title")
    return p.parse_args()


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if not values:
        return []
    window = max(1, int(window))
    if window <= 1:
        return list(values)
    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + float(v))
    out: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx + 1 - window)
        count = idx + 1 - start
        out.append((prefix[idx + 1] - prefix[start]) / float(count))
    return out


def to_float_list(rows: Sequence[Dict[str, str]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        raw = row.get(key, "")
        if raw in ("", None):
            values.append(float("nan"))
        else:
            values.append(float(raw))
    return values


def to_int_list(rows: Sequence[Dict[str, str]], key: str, fallback_start: int = 1) -> List[int]:
    values: List[int] = []
    for idx, row in enumerate(rows):
        raw = row.get(key, "")
        values.append(int(raw) if raw not in ("", None) else fallback_start + idx)
    return values


def finite_pairs(xs: Sequence[float], ys: Sequence[float]) -> Tuple[List[float], List[float]]:
    fx: List[float] = []
    fy: List[float] = []
    for x, y in zip(xs, ys):
        if y == y:
            fx.append(float(x))
            fy.append(float(y))
    return fx, fy


def get_best_iter(summary: Dict, key: str) -> Optional[int]:
    value = summary.get(key)
    return int(value) if value not in (None, "") else None


def maybe_load_candidate_eval(candidate_eval_dir: Path) -> Optional[Tuple[List[int], List[float], List[float]]]:
    summary_path = candidate_eval_dir / "candidate_eval_summary.json"
    if not summary_path.exists():
        return None
    payload = load_json(summary_path)
    rows = payload.get("candidates", [])
    if not isinstance(rows, list):
        return None
    iters: List[int] = []
    goodputs: List[float] = []
    expiries: List[float] = []
    for row in rows:
        iter_value = row.get("iter")
        goodput = row.get("mean_cluster_goodput_mbps")
        expiry = row.get("mean_expiry_drop_rate")
        if iter_value in (None, "") or goodput in (None, ""):
            continue
        iters.append(int(iter_value))
        goodputs.append(float(goodput))
        expiries.append(float(expiry) if expiry not in (None, "") else float("nan"))
    if not iters:
        return None
    return iters, goodputs, expiries


def add_curve(ax, x: Sequence[float], y: Sequence[float], label: str, color: str, smooth_window: int) -> None:
    fx, fy = finite_pairs(x, y)
    if not fx:
        return
    ax.plot(fx, fy, linewidth=1.0, alpha=0.30, color=color, label=f"{label} raw")
    if smooth_window > 1:
        ax.plot(fx, moving_average(fy, smooth_window), linewidth=2.0, color=color, label=f"{label} ma{smooth_window}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary_path = run_dir / "online_ppo_summary.json"
    iter_csv_path = run_dir / "online_ppo_metrics.csv"
    episode_csv_path = run_dir / "online_ppo_episode_metrics.csv"
    candidate_csv_path = run_dir / "online_ppo_candidate_checkpoints.csv"
    candidate_eval_dir = run_dir / "candidate_main_eval"

    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary file: {summary_path}")
    if not iter_csv_path.exists():
        raise FileNotFoundError(f"missing iter metrics csv: {iter_csv_path}")
    if not episode_csv_path.exists():
        raise FileNotFoundError(f"missing episode metrics csv: {episode_csv_path}")

    summary = load_json(summary_path)
    iter_rows = load_csv_rows(iter_csv_path)
    episode_rows = load_csv_rows(episode_csv_path)
    candidate_rows = load_csv_rows(candidate_csv_path) if candidate_csv_path.exists() else []

    iter_x = to_int_list(iter_rows, "iter", fallback_start=1)
    episode_x = to_int_list(episode_rows, "episode", fallback_start=1)

    best_rollout_iter = get_best_iter(summary, "best_rollout_iter")
    best_objective_iter = get_best_iter(summary, "best_objective_iter")
    best_checkpoint_iter = get_best_iter(summary, "best_checkpoint_iter")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib unavailable: {e}") from e

    fig, axes = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)
    ax = axes.flatten()

    # 1. Iteration objective and normalized reward.
    add_curve(ax[0], iter_x, to_float_list(iter_rows, "objective"), "objective", "tab:blue", int(args.smooth_window))
    add_curve(ax[0], iter_x, to_float_list(iter_rows, "rollout_reward_mean"), "reward", "tab:orange", int(args.smooth_window))
    if best_objective_iter is not None:
        ax[0].axvline(best_objective_iter, color="tab:blue", linestyle="--", linewidth=1.2, alpha=0.8, label=f"best objective iter={best_objective_iter}")
    ax[0].set_title("Objective and Reward")
    ax[0].set_xlabel("iteration")
    ax[0].grid(True, alpha=0.25)
    ax[0].legend(fontsize=8)

    # 2. Rollout throughput / goodput.
    add_curve(ax[1], iter_x, to_float_list(iter_rows, "rollout_goodput_mbps_mean"), "goodput", "tab:green", int(args.smooth_window))
    add_curve(ax[1], iter_x, to_float_list(iter_rows, "rollout_throughput_mbps_mean"), "throughput", "tab:purple", int(args.smooth_window))
    if best_rollout_iter is not None:
        ax[1].axvline(best_rollout_iter, color="tab:red", linestyle="--", linewidth=1.2, alpha=0.8, label=f"best rollout iter={best_rollout_iter}")
    ax[1].set_title("Rollout Throughput and Goodput")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("Mbps")
    ax[1].grid(True, alpha=0.25)
    ax[1].legend(fontsize=8)

    # 3. Rollout expiry / BLER.
    add_curve(ax[2], iter_x, to_float_list(iter_rows, "rollout_expiry_drop_rate_mean"), "expiry", "tab:orange", int(args.smooth_window))
    add_curve(ax[2], iter_x, to_float_list(iter_rows, "rollout_tb_err_rate_mean"), "tb_bler", "tab:red", int(args.smooth_window))
    ax[2].set_title("Rollout Expiry and TB BLER")
    ax[2].set_xlabel("iteration")
    ax[2].grid(True, alpha=0.25)
    ax[2].legend(fontsize=8)

    # 4. Rollout buffer.
    add_curve(ax[3], iter_x, to_float_list(iter_rows, "rollout_total_buffer_mb_mean"), "total_buffer", "tab:brown", int(args.smooth_window))
    ax[3].set_title("Rollout Total Buffer")
    ax[3].set_xlabel("iteration")
    ax[3].set_ylabel("MB")
    ax[3].grid(True, alpha=0.25)
    ax[3].legend(fontsize=8)

    # 5. Entropy and KL.
    add_curve(ax[4], iter_x, to_float_list(iter_rows, "entropy"), "entropy", "tab:cyan", int(args.smooth_window))
    add_curve(ax[4], iter_x, to_float_list(iter_rows, "approx_kl"), "approx_kl", "tab:pink", int(args.smooth_window))
    target_kl = summary.get("args", {}).get("target_kl")
    if target_kl not in (None, ""):
        ax[4].axhline(float(target_kl), color="tab:red", linestyle="--", linewidth=1.2, alpha=0.8, label=f"target_kl={float(target_kl):.3f}")
    ax[4].set_title("Entropy and KL")
    ax[4].set_xlabel("iteration")
    ax[4].grid(True, alpha=0.25)
    ax[4].legend(fontsize=8)

    # 6. PRG behavior.
    add_curve(ax[5], iter_x, to_float_list(iter_rows, "rollout_prg_utilization_ratio_mean"), "prg_util", "tab:olive", int(args.smooth_window))
    add_curve(ax[5], iter_x, to_float_list(iter_rows, "rollout_prg_reuse_ratio_mean"), "prg_reuse", "tab:gray", int(args.smooth_window))
    ax[5].set_title("PRG Utilization and Reuse")
    ax[5].set_xlabel("iteration")
    ax[5].grid(True, alpha=0.25)
    ax[5].legend(fontsize=8)

    # 7. Episode metrics.
    add_curve(ax[6], episode_x, to_float_list(episode_rows, "episode_goodput_mbps_mean"), "episode_goodput", "tab:green", int(args.smooth_window))
    add_curve(ax[6], episode_x, to_float_list(episode_rows, "episode_throughput_mbps_mean"), "episode_throughput", "tab:purple", int(args.smooth_window))
    ax[6].set_title("Episode Throughput and Goodput")
    ax[6].set_xlabel("episode")
    ax[6].set_ylabel("Mbps")
    ax[6].grid(True, alpha=0.25)
    ax[6].legend(fontsize=8)

    # 8. Episode expiry / BLER.
    add_curve(ax[7], episode_x, to_float_list(episode_rows, "episode_expiry_drop_rate_mean"), "episode_expiry", "tab:orange", int(args.smooth_window))
    add_curve(ax[7], episode_x, to_float_list(episode_rows, "episode_tb_err_rate_mean"), "episode_tb_bler", "tab:red", int(args.smooth_window))
    ax[7].set_title("Episode Expiry and TB BLER")
    ax[7].set_xlabel("episode")
    ax[7].grid(True, alpha=0.25)
    ax[7].legend(fontsize=8)

    # 9. Candidate checkpoints and optional main eval.
    cand_x = to_int_list(candidate_rows, "candidate_iter", fallback_start=1) if candidate_rows else []
    cand_goodput = to_float_list(candidate_rows, "rollout_goodput_mbps_mean") if candidate_rows else []
    cand_expiry = to_float_list(candidate_rows, "rollout_expiry_drop_rate_mean") if candidate_rows else []
    if cand_x:
        ax[8].plot(cand_x, cand_goodput, marker="o", linewidth=1.4, color="tab:green", label="candidate rollout goodput")
        ax8b = ax[8].twinx()
        ax8b.plot(cand_x, cand_expiry, marker="s", linewidth=1.2, color="tab:orange", alpha=0.85, label="candidate rollout expiry")
        candidate_eval = maybe_load_candidate_eval(candidate_eval_dir)
        if candidate_eval is not None:
            eval_x, eval_goodput, eval_expiry = candidate_eval
            ax[8].plot(eval_x, eval_goodput, marker="^", linewidth=1.4, color="tab:blue", label="candidate main-eval goodput")
            ax8b.plot(eval_x, eval_expiry, marker="v", linewidth=1.2, color="tab:red", alpha=0.85, label="candidate main-eval expiry")
        if best_checkpoint_iter is not None:
            ax[8].axvline(best_checkpoint_iter, color="tab:red", linestyle="--", linewidth=1.2, alpha=0.8, label=f"best checkpoint iter={best_checkpoint_iter}")
        ax[8].set_title("Candidate Checkpoints")
        ax[8].set_xlabel("iteration")
        ax[8].set_ylabel("goodput (Mbps)")
        ax8b.set_ylabel("expiry drop rate")
        ax[8].grid(True, alpha=0.25)
        handles1, labels1 = ax[8].get_legend_handles_labels()
        handles2, labels2 = ax8b.get_legend_handles_labels()
        ax[8].legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="best")
    else:
        ax[8].axis("off")

    env_dims = summary.get("env_dims", {})
    title = args.title.strip() if args.title else ""
    if not title:
        title = (
            f"Stage-B Online PPO Curves: {run_dir.name}\n"
            f"cells={env_dims.get('n_cell', '?')} ue={env_dims.get('n_active_ue', '?')} prg={env_dims.get('n_prg', '?')} "
            f"best_rollout_iter={best_rollout_iter} best_checkpoint_iter={best_checkpoint_iter}"
        )
    fig.suptitle(title, fontsize=14)

    output_png = Path(args.output_png).resolve() if args.output_png else (run_dir / "online_ppo_curves_key_metrics.png")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160, bbox_inches="tight")

    if args.output_svg:
        output_svg = Path(args.output_svg).resolve()
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_svg, bbox_inches="tight")

    plt.close(fig)
    print(f"curves written: {output_png}")
    if args.output_svg:
        print(f"curves written: {output_svg}")


if __name__ == "__main__":
    main()
