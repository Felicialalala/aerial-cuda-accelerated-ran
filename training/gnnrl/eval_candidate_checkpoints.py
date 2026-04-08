#!/usr/bin/env python3
"""Export periodic online-PPO candidate checkpoints and pick deployment best via main experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(token) for token in text.replace(",", " ").split()]


def _read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def _safe_tag(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:96] if len(text) > 96 else text


def _candidate_train_sort_key(row: Dict) -> Tuple[float, float, float]:
    return (
        float(row.get("rollout_goodput_mbps_mean", 0.0)),
        -float(row.get("rollout_expiry_drop_rate_mean", 1.0e30)),
        -float(row.get("rollout_tb_err_rate_mean", 1.0e30)),
    )


def _resolve_repo_artifact_path(path_text: str, repo_root: Path, fallback: Optional[Path] = None) -> Path:
    candidates: List[Path] = []
    if path_text:
        raw_path = Path(path_text)
        candidates.append(raw_path)
        if raw_path.is_absolute():
            for prefix in (Path("/opt/nvidia/cuBB"),):
                try:
                    candidates.append(repo_root / raw_path.relative_to(prefix))
                except ValueError:
                    continue
        else:
            candidates.append(repo_root / raw_path)
    if fallback is not None:
        candidates.append(fallback)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve() if candidates else repo_root.resolve()


def _eval_rank_key(row: Dict, metric: str, tiebreak: str, tiebreak2: str) -> Tuple[float, float, float]:
    return (
        float(row.get(metric, -1.0e30)),
        -float(row.get(tiebreak, 1.0e30)),
        -float(row.get(tiebreak2, 1.0e30)),
    )


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return 0.0 if not vals else sum(vals) / float(len(vals))


def _discover_candidates(train_out_dir: Path, repo_root: Path, limit: int) -> List[Dict]:
    summary_path = train_out_dir / "online_ppo_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    candidate_csv = train_out_dir / "online_ppo_candidate_checkpoints.csv"

    periodic_rows: List[Dict] = []
    if candidate_csv.exists():
        with open(candidate_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("candidate_path"):
                    continue
                periodic_rows.append(
                    {
                        "candidate_id": f"iter{int(float(row.get('candidate_iter', row.get('iter', 0)))):04d}",
                        "iter": int(float(row.get("candidate_iter", row.get("iter", 0)))),
                        "checkpoint_path": str(
                            _resolve_repo_artifact_path(row["candidate_path"], repo_root=repo_root).resolve()
                        ),
                        "candidate_roles": "periodic",
                        "checkpoint_source": row.get("candidate_reason", "periodic_interval"),
                        "rollout_goodput_mbps_mean": float(row.get("rollout_goodput_mbps_mean", 0.0)),
                        "rollout_expiry_drop_rate_mean": float(row.get("rollout_expiry_drop_rate_mean", 0.0)),
                        "rollout_tb_err_rate_mean": float(row.get("rollout_tb_err_rate_mean", 0.0)),
                        "rollout_prg_utilization_ratio_mean": float(row.get("rollout_prg_utilization_ratio_mean", 0.0)),
                        "rollout_prg_reuse_ratio_mean": float(row.get("rollout_prg_reuse_ratio_mean", 0.0)),
                    }
                )

    periodic_by_iter = {int(row["iter"]): row for row in periodic_rows}
    best_iter = int(summary.get("best_checkpoint_iter", -1))
    best_actor_path = _resolve_repo_artifact_path(
        summary.get("best_actor_checkpoint", ""),
        repo_root=repo_root,
        fallback=train_out_dir / "ppo_actor_best.pt",
    )
    if best_actor_path.exists():
        if best_iter in periodic_by_iter:
            periodic_by_iter[best_iter]["candidate_roles"] = "periodic,trainer_best"
        else:
            periodic_rows.append(
                {
                    "candidate_id": f"iter{best_iter:04d}_trainer_best" if best_iter >= 0 else "trainer_best",
                    "iter": best_iter,
                    "checkpoint_path": str(best_actor_path),
                    "candidate_roles": "trainer_best",
                    "checkpoint_source": "trainer_best",
                    "rollout_goodput_mbps_mean": float(summary.get("best_rollout_goodput_mbps_mean", 0.0)),
                    "rollout_expiry_drop_rate_mean": float(summary.get("best_rollout_expiry_drop_rate_mean", 0.0)),
                    "rollout_tb_err_rate_mean": float(summary.get("best_rollout_tb_err_rate_mean", 0.0)),
                    "rollout_prg_utilization_ratio_mean": 0.0,
                    "rollout_prg_reuse_ratio_mean": 0.0,
                }
            )

    dedup: Dict[str, Dict] = {}
    for row in periodic_rows:
        dedup[str(Path(row["checkpoint_path"]).resolve())] = row
    candidates = sorted(dedup.values(), key=_candidate_train_sort_key, reverse=True)
    if limit > 0:
        trainer_best_rows = [row for row in candidates if "trainer_best" in row.get("candidate_roles", "")]
        limited = candidates[:limit]
        if trainer_best_rows and trainer_best_rows[0] not in limited:
            limited = limited[:-1] + trainer_best_rows[:1]
        candidates = sorted(limited, key=_candidate_train_sort_key, reverse=True)
    return candidates


def _find_new_output_dir(output_root: Path, tag: str, before: Sequence[Path]) -> Optional[Path]:
    pattern = f"stageB_main_experiment_{tag}_*"
    before_set = {path.resolve() for path in before}
    after = sorted(output_root.glob(pattern), key=lambda p: p.stat().st_mtime)
    new_dirs = [path.resolve() for path in after if path.resolve() not in before_set]
    if new_dirs:
        return new_dirs[-1]
    return after[-1].resolve() if after else None


def _collect_scenario_kpis(run_dir: Path) -> List[Tuple[str, Dict]]:
    results: List[Tuple[str, Dict]] = []
    for summary_path in sorted(run_dir.glob("*/kpi_summary.json")):
        scenario_name = summary_path.parent.name
        try:
            payload = _read_json(summary_path)
            results.append((scenario_name, dict(payload.get("global_kpi", {}))))
        except Exception:
            continue
    return results


def _run_cmd(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(proc.returncode), " ".join(cmd)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Evaluate periodic online PPO checkpoints via main experiment")
    p.add_argument("--train-out-dir", required=True)
    p.add_argument("--run-script", default=str(repo_root / "cuMAC/scripts/run_stageB_main_experiment.sh"))
    p.add_argument("--export-script", default=str(repo_root / "training/gnnrl/export_onnx.py"))
    p.add_argument("--output-root", default=str(repo_root / "output"))
    p.add_argument("--build-method", default="skip")
    p.add_argument("--topology-scenario", required=True)
    p.add_argument("--total-ue-count", type=int, required=True)
    p.add_argument("--prbs-per-group", type=int, required=True)
    p.add_argument("--baseline-scheduler", default="pfq")
    p.add_argument("--fading-mode", type=int, default=0)
    p.add_argument("--cdl-profiles", default="NA")
    p.add_argument("--cdl-delay-spreads", default="0")
    p.add_argument("--tti", type=int, required=True)
    p.add_argument("--packet-size-bytes", type=int, required=True)
    p.add_argument("--traffic-arrival-rate", type=float, required=True)
    p.add_argument("--packet-ttl-tti", type=int, default=0)
    p.add_argument("--packet-ttl-ms", type=float, default=0.0)
    p.add_argument("--topology-seed", type=int, required=True)
    p.add_argument("--progress-tti", type=int, default=1000)
    p.add_argument("--kpi-tti-log", type=int, default=0)
    p.add_argument("--compare-tti", type=int, default=0)
    p.add_argument("--compact-output", type=int, choices=[0, 1], default=1)
    p.add_argument("--exec-mode", default="both")
    p.add_argument("--gnnrl-action-mode", default="joint")
    p.add_argument("--gnnrl-model-decode-mode", choices=["sample", "argmax"], default="sample")
    p.add_argument("--gnnrl-model-sample-seeds", default="42")
    p.add_argument("--gnnrl-model-no-ue-bias", default="0")
    p.add_argument("--gnnrl-model-no-prg-bias", default="0")
    p.add_argument("--gnnrl-model-min-sched-ratio", default="0")
    p.add_argument("--gnnrl-model-min-prg-ratio", default="0")
    p.add_argument("--candidate-limit", type=int, default=0, help="Evaluate top-N training candidates; 0 means all")
    p.add_argument("--rank-metric", default="cluster_goodput_mbps")
    p.add_argument("--tiebreak-metric", default="expiry_drop_rate")
    p.add_argument("--tiebreak2-metric", default="global_tb_bler")
    p.add_argument("--promote-best", type=int, choices=[0, 1], default=1)
    p.add_argument("--tag-prefix", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    train_out_dir = Path(args.train_out_dir).resolve()
    output_root = Path(args.output_root).resolve()
    run_script = Path(args.run_script).resolve()
    export_script = Path(args.export_script).resolve()
    eval_dir = train_out_dir / "candidate_main_eval"
    onnx_dir = eval_dir / "onnx"
    runs_csv_path = eval_dir / "candidate_eval_runs.csv"
    metrics_csv_path = eval_dir / "candidate_eval_metrics.csv"
    summary_path = eval_dir / "candidate_eval_summary.json"
    best_eval_ckpt_path = train_out_dir / "ppo_actor_best_eval.pt"
    best_eval_onnx_path = train_out_dir / "model_best_eval.onnx"

    eval_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    candidates = _discover_candidates(train_out_dir, repo_root=repo_root, limit=int(args.candidate_limit))
    if not candidates:
        raise RuntimeError(f"no candidate checkpoints found under {train_out_dir}")

    if args.gnnrl_model_decode_mode == "sample":
        sample_seeds = _parse_int_list(args.gnnrl_model_sample_seeds)
        if not sample_seeds:
            raise ValueError("--gnnrl-model-sample-seeds must be non-empty in sample mode")
    else:
        sample_seeds = [None]

    print(f"[candidate-eval] candidates={len(candidates)} decode_mode={args.gnnrl_model_decode_mode}")
    run_rows: List[Dict] = []

    for candidate in candidates:
        ckpt_path = Path(candidate["checkpoint_path"]).resolve()
        if not ckpt_path.exists():
            print(f"[candidate-eval] skip missing checkpoint: {ckpt_path}")
            continue
        candidate_id = _safe_tag(candidate["candidate_id"])
        onnx_path = onnx_dir / f"{candidate_id}.onnx"
        export_cmd = [
            "python3",
            str(export_script),
            "--checkpoint",
            str(ckpt_path),
            "--out",
            str(onnx_path),
            "--opset",
            "18",
        ]
        export_rc, export_cmd_str = _run_cmd(export_cmd, repo_root)
        if export_rc != 0:
            run_rows.append(
                {
                    "candidate_id": candidate_id,
                    "iter": candidate["iter"],
                    "checkpoint_path": str(ckpt_path),
                    "onnx_path": str(onnx_path),
                    "eval_seed": "",
                    "status": "export_failed",
                    "return_code": export_rc,
                    "command": export_cmd_str,
                }
            )
            continue

        for sample_seed in sample_seeds:
            tag_seed = f"s{sample_seed}" if sample_seed is not None else "det"
            tag_parts = [args.tag_prefix or _safe_tag(train_out_dir.name), candidate_id, args.gnnrl_model_decode_mode, tag_seed]
            run_tag = _safe_tag("_".join(part for part in tag_parts if part))
            before = list(output_root.glob(f"stageB_main_experiment_{run_tag}_*"))
            cmd = [
                str(run_script),
                "--topology-scenario",
                args.topology_scenario,
                "--total-ue-count",
                str(args.total_ue_count),
                "--build-method",
                args.build_method,
                "--prbs-per-group",
                str(args.prbs_per_group),
                "--baseline-scheduler",
                args.baseline_scheduler,
                "--custom-ue-prg",
                "1",
                "--custom-policy",
                "gnnrl_model",
                "--gnnrl-action-mode",
                args.gnnrl_action_mode,
                "--gnnrl-model-decode-mode",
                args.gnnrl_model_decode_mode,
                "--gnnrl-model-no-ue-bias",
                str(args.gnnrl_model_no_ue_bias),
                "--gnnrl-model-no-prg-bias",
                str(args.gnnrl_model_no_prg_bias),
                "--gnnrl-model-min-sched-ratio",
                str(args.gnnrl_model_min_sched_ratio),
                "--gnnrl-model-min-prg-ratio",
                str(args.gnnrl_model_min_prg_ratio),
                "--model-path",
                str(onnx_path),
                "--fading-mode",
                str(args.fading_mode),
                "--cdl-profiles",
                args.cdl_profiles,
                "--cdl-delay-spreads",
                args.cdl_delay_spreads,
                "--tti",
                str(args.tti),
                "--packet-size-bytes",
                str(args.packet_size_bytes),
                "--traffic-arrival-rate",
                str(args.traffic_arrival_rate),
                "--packet-ttl-tti",
                str(args.packet_ttl_tti),
                "--packet-ttl-ms",
                str(args.packet_ttl_ms),
                "--topology-seed",
                str(args.topology_seed),
                "--progress-tti",
                str(args.progress_tti),
                "--kpi-tti-log",
                str(args.kpi_tti_log),
                "--compare-tti",
                str(args.compare_tti),
                "--compact-output",
                str(args.compact_output),
                "--exec-mode",
                args.exec_mode,
                "--tag",
                run_tag,
            ]
            if sample_seed is not None:
                cmd.extend(["--gnnrl-model-sample-seed", str(sample_seed)])

            rc, cmd_str = _run_cmd(cmd, repo_root)
            run_dir = _find_new_output_dir(output_root, run_tag, before)
            base_row = {
                "candidate_id": candidate_id,
                "iter": candidate["iter"],
                "candidate_roles": candidate.get("candidate_roles", ""),
                "checkpoint_path": str(ckpt_path),
                "onnx_path": str(onnx_path),
                "eval_seed": "" if sample_seed is None else int(sample_seed),
                "decode_mode": args.gnnrl_model_decode_mode,
                "run_tag": run_tag,
                "run_dir": "" if run_dir is None else str(run_dir),
                "return_code": rc,
                "command": cmd_str,
                "status": "ok" if rc == 0 else "run_failed",
                "train_rollout_goodput_mbps_mean": float(candidate.get("rollout_goodput_mbps_mean", 0.0)),
                "train_rollout_expiry_drop_rate_mean": float(candidate.get("rollout_expiry_drop_rate_mean", 0.0)),
                "train_rollout_tb_err_rate_mean": float(candidate.get("rollout_tb_err_rate_mean", 0.0)),
                "train_rollout_prg_utilization_ratio_mean": float(candidate.get("rollout_prg_utilization_ratio_mean", 0.0)),
                "train_rollout_prg_reuse_ratio_mean": float(candidate.get("rollout_prg_reuse_ratio_mean", 0.0)),
            }
            if rc != 0 or run_dir is None:
                run_rows.append(base_row)
                continue

            scenario_kpis = _collect_scenario_kpis(run_dir)
            if not scenario_kpis:
                base_row["status"] = "kpi_missing"
                run_rows.append(base_row)
                continue

            metrics_keys = set()
            for _scenario_name, kpi in scenario_kpis:
                metrics_keys.update(kpi.keys())
            metric_means: Dict[str, float] = {}
            for key in metrics_keys:
                values = [float(kpi.get(key, 0.0)) for _scenario_name, kpi in scenario_kpis if key in kpi]
                if values:
                    metric_means[key] = _mean(values)
            base_row["scenario_count"] = len(scenario_kpis)
            base_row["scenario_names"] = ",".join(name for name, _kpi in scenario_kpis)
            base_row.update(metric_means)
            run_rows.append(base_row)
            print(
                f"[candidate-eval] {candidate_id} seed={tag_seed} "
                f"goodput={metric_means.get('cluster_goodput_mbps', 0.0):.2f} "
                f"expR={metric_means.get('expiry_drop_rate', 0.0):.4f} "
                f"bler={metric_means.get('global_tb_bler', 0.0):.4f}"
            )

    _write_csv(run_rows, runs_csv_path)

    agg_rows: List[Dict] = []
    grouped: Dict[str, List[Dict]] = {}
    for row in run_rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)

    for candidate_id, rows in grouped.items():
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        agg = {
            "candidate_id": candidate_id,
            "iter": rows[0].get("iter", -1),
            "candidate_roles": rows[0].get("candidate_roles", ""),
            "checkpoint_path": rows[0].get("checkpoint_path", ""),
            "onnx_path": rows[0].get("onnx_path", ""),
            "decode_mode": rows[0].get("decode_mode", ""),
            "num_runs": len(rows),
            "num_successful_runs": len(ok_rows),
            "status": "ok" if ok_rows else "failed",
            "train_rollout_goodput_mbps_mean": rows[0].get("train_rollout_goodput_mbps_mean", 0.0),
            "train_rollout_expiry_drop_rate_mean": rows[0].get("train_rollout_expiry_drop_rate_mean", 0.0),
            "train_rollout_tb_err_rate_mean": rows[0].get("train_rollout_tb_err_rate_mean", 0.0),
            "train_rollout_prg_utilization_ratio_mean": rows[0].get("train_rollout_prg_utilization_ratio_mean", 0.0),
            "train_rollout_prg_reuse_ratio_mean": rows[0].get("train_rollout_prg_reuse_ratio_mean", 0.0),
            "successful_run_dirs": ",".join(str(row.get("run_dir", "")) for row in ok_rows),
        }
        if ok_rows:
            metric_keys = set()
            for row in ok_rows:
                for key in row.keys():
                    if key.startswith("train_") or key in {
                        "candidate_id",
                        "iter",
                        "candidate_roles",
                        "checkpoint_path",
                        "onnx_path",
                        "decode_mode",
                        "eval_seed",
                        "run_tag",
                        "run_dir",
                        "return_code",
                        "command",
                        "status",
                        "scenario_count",
                        "scenario_names",
                    }:
                        continue
                    try:
                        float(row[key])
                    except (TypeError, ValueError):
                        continue
                    metric_keys.add(key)
            for key in sorted(metric_keys):
                values = [float(row[key]) for row in ok_rows if key in row]
                agg[f"mean_{key}"] = _mean(values)
                agg[f"min_{key}"] = min(values)
                agg[f"max_{key}"] = max(values)
        agg_rows.append(agg)

    agg_rows = sorted(
        agg_rows,
        key=lambda row: _eval_rank_key(
            row,
            f"mean_{args.rank_metric}",
            f"mean_{args.tiebreak_metric}",
            f"mean_{args.tiebreak2_metric}",
        ),
        reverse=True,
    )
    _write_csv(agg_rows, metrics_csv_path)

    best_row: Optional[Dict] = None
    ok_agg_rows = [row for row in agg_rows if row.get("status") == "ok"]
    if ok_agg_rows:
        best_row = ok_agg_rows[0]
        if int(args.promote_best) == 1:
            shutil.copy2(best_row["checkpoint_path"], best_eval_ckpt_path)
            shutil.copy2(best_row["onnx_path"], best_eval_onnx_path)

    summary = {
        "status": "ok" if best_row is not None else "failed",
        "train_out_dir": str(train_out_dir),
        "candidate_count": len(candidates),
        "evaluated_run_count": len(run_rows),
        "successful_candidate_count": len(ok_agg_rows),
        "rank_metric": args.rank_metric,
        "tiebreak_metric": args.tiebreak_metric,
        "tiebreak2_metric": args.tiebreak2_metric,
        "runs_csv": str(runs_csv_path),
        "metrics_csv": str(metrics_csv_path),
        "promoted_checkpoint": str(best_eval_ckpt_path) if best_row is not None and int(args.promote_best) == 1 else "",
        "promoted_onnx": str(best_eval_onnx_path) if best_row is not None and int(args.promote_best) == 1 else "",
        "best_candidate": best_row,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    if best_row is not None:
        print(
            f"[candidate-eval] best={best_row['candidate_id']} "
            f"goodput={best_row.get('mean_cluster_goodput_mbps', 0.0):.2f} "
            f"expR={best_row.get('mean_expiry_drop_rate', 0.0):.4f} "
            f"bler={best_row.get('mean_global_tb_bler', 0.0):.4f}"
        )
        if int(args.promote_best) == 1:
            print(f"[candidate-eval] promoted checkpoint: {best_eval_ckpt_path}")
            print(f"[candidate-eval] promoted onnx: {best_eval_onnx_path}")
        return 0

    print("[candidate-eval] no successful candidate evaluation")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
