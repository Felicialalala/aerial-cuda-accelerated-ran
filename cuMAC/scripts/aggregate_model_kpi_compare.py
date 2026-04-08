#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DEFAULT_METRICS = [
    {"name": "traffic.served_mbps_est", "unit": "Mbps", "direction": "higher_better"},
    {"name": "traffic.goodput_mbps", "unit": "Mbps", "direction": "higher_better"},
    {"name": "traffic.expired_bytes", "unit": "bytes", "direction": "lower_better"},
    {"name": "traffic.expired_packets", "unit": None, "direction": "lower_better"},
    {"name": "traffic.expiry_drop_rate", "unit": "%", "direction": "lower_better"},
    {"name": "global_kpi.cluster_sum_throughput_mbps", "unit": "Mbps", "direction": "higher_better"},
    {"name": "global_kpi.cluster_goodput_mbps", "unit": "Mbps", "direction": "higher_better"},
    {"name": "global_kpi.cluster_spectral_efficiency_bps_per_hz", "unit": "bps/Hz", "direction": "higher_better"},
    {"name": "global_kpi.average_ue_throughput_mbps", "unit": "Mbps", "direction": "higher_better"},
    {"name": "global_kpi.average_ue_goodput_mbps", "unit": "Mbps", "direction": "higher_better"},
    {"name": "global_kpi.ue_throughput_jain", "unit": None, "direction": "higher_better"},
    {"name": "global_kpi.ue_goodput_jain", "unit": None, "direction": "higher_better"},
    {"name": "global_kpi.ue_throughput_p5_mbps", "unit": "Mbps", "direction": "higher_better"},
    {"name": "global_kpi.ue_goodput_p5_mbps", "unit": "Mbps", "direction": "higher_better"},
    {"name": "global_kpi.cell_edge_spectral_efficiency_p5_bps_per_hz", "unit": "bps/Hz", "direction": "higher_better"},
    {"name": "global_kpi.ue_throughput_p10_mbps", "unit": "Mbps", "direction": "higher_better"},
    {"name": "global_kpi.ue_goodput_p10_mbps", "unit": "Mbps", "direction": "higher_better"},
    {"name": "global_kpi.global_tb_bler", "unit": "%", "direction": "lower_better"},
    {"name": "global_kpi.global_tx_success_rate", "unit": "%", "direction": "higher_better"},
    {"name": "global_kpi.residual_buffer_ratio", "unit": "%", "direction": "lower_better"},
    {"name": "global_kpi.non_residual_buffer_ratio", "unit": "%", "direction": "higher_better"},
    {"name": "global_kpi.served_buffer_ratio", "unit": "%", "direction": "higher_better"},
    {"name": "global_kpi.backlog_free_ue_ratio", "unit": "%", "direction": "higher_better"},
    {"name": "global_kpi.prg_utilization_ratio", "unit": "%", "direction": "higher_better"},
    {"name": "traffic.queue_delay_est_ms", "unit": "ms", "direction": "lower_better"},
    {"name": "traffic.packet_delay_served_pkt_count", "unit": None, "direction": "higher_better"},
    {"name": "traffic.packet_delay_pending_pkt_count", "unit": None, "direction": "lower_better"},
    {"name": "traffic.packet_delay_mean_ms", "unit": "ms", "direction": "lower_better"},
    {"name": "traffic.packet_delay_p50_ms", "unit": "ms", "direction": "lower_better"},
    {"name": "traffic.packet_delay_p90_ms", "unit": "ms", "direction": "lower_better"},
    {"name": "traffic.packet_delay_p95_ms", "unit": "ms", "direction": "lower_better"},
    {"name": "global_kpi.queue_delay_p50_ms", "unit": "ms", "direction": "lower_better"},
    {"name": "global_kpi.queue_delay_p90_ms", "unit": "ms", "direction": "lower_better"},
    {"name": "global_kpi.queue_delay_p95_ms", "unit": "ms", "direction": "lower_better"},
    {"name": "global_kpi.ue_fraction_queue_delay_gt_10s", "unit": "%", "direction": "lower_better"},
    {"name": "global_kpi.ue_fraction_queue_delay_gt_20s", "unit": "%", "direction": "lower_better"},
    {"name": "global_kpi.scheduled_ratio_mean", "unit": "%", "direction": "higher_better"},
    {"name": "global_kpi.scheduled_ratio_jain", "unit": None, "direction": "higher_better"},
    {"name": "global_kpi.scheduled_ratio_p5", "unit": "%", "direction": "higher_better"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate multi-seed Stage-B model KPI runs and compare with RR/PFQ means.")
    p.add_argument("--manifest", required=True, help="CSV manifest produced by run_stageB_model_multi_seed_compare.sh")
    p.add_argument("--output-dir", required=True, help="Directory to write aggregated outputs")
    p.add_argument("--model-label", default="", help="Optional model label for output columns/files")
    p.add_argument(
        "--baseline-mean-root",
        default="",
        help="Optional root or scenario directory containing rr_vs_*_compare_mean.csv outputs",
    )
    p.add_argument(
        "--baseline-mean-csv",
        default="",
        help="Optional single rr_vs_*_compare_mean.csv path used for all scenarios",
    )
    return p.parse_args()


def normalize_label(value: str, default: str = "model") -> str:
    raw = (value or "").strip()
    if not raw:
        raw = default
    safe = "".join(ch if ch.isalnum() else "_" for ch in raw)
    safe = safe.strip("_")
    return safe or default


def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coerce_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> Optional[float]:
    return None if not values else float(sum(values) / float(len(values)))


def _std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.pstdev(values))


def _min(values: Sequence[float]) -> Optional[float]:
    return None if not values else float(min(values))


def _max(values: Sequence[float]) -> Optional[float]:
    return None if not values else float(max(values))


def safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or abs(den) <= 1.0e-12:
        return None
    return float(num / den)


def format_value(v: Optional[float], unit: Optional[str] = None) -> str:
    if v is None:
        return "N/A"
    if unit == "%":
        return f"{v * 100.0:.3f}%"
    if unit == "ms":
        return f"{v:.3f} ms"
    if unit == "Mbps":
        return f"{v:.6f} Mbps"
    if unit == "bps/Hz":
        return f"{v:.6f} bps/Hz"
    if unit == "bytes":
        return f"{int(round(v))}"
    return f"{v:.10f}"


def compare_winner(
    left_value: Optional[float],
    right_value: Optional[float],
    direction: str,
    left_label: str,
    right_label: str,
) -> str:
    if left_value is None or right_value is None:
        return "unknown"
    if abs(right_value - left_value) <= 1.0e-12:
        return "tie"
    if direction == "higher_better":
        return right_label if right_value > left_value else left_label
    return right_label if right_value < left_value else left_label


def dotted_get(data: Dict, dotted_name: str):
    cur = data
    for key in dotted_name.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_baseline_csv(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    baseline_label = ""
    for field in fieldnames:
        if field.endswith("_mean") and field not in {"rr_mean"}:
            baseline_label = field[: -len("_mean")]
            break
    baseline_label = normalize_label(baseline_label, default="pf")
    return {
        "path": str(path),
        "rows": rows,
        "fieldnames": fieldnames,
        "baseline_label": baseline_label,
    }


def resolve_baseline_csv_for_scenario(
    scenario: str,
    baseline_mean_root: Optional[Path],
    baseline_mean_csv: Optional[Path],
) -> Optional[Path]:
    if baseline_mean_csv is not None:
        return baseline_mean_csv
    if baseline_mean_root is None:
        return None
    if baseline_mean_root.is_file() and baseline_mean_root.suffix.lower() == ".csv":
        return baseline_mean_root
    candidates: List[Path] = []
    scenario_dir = baseline_mean_root / scenario
    if scenario_dir.is_dir():
        candidates.extend(sorted(scenario_dir.glob("rr_vs_*_compare_mean.csv")))
    if baseline_mean_root.is_dir():
        candidates.extend(sorted(baseline_mean_root.glob("rr_vs_*_compare_mean.csv")))
    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates[0] if unique_candidates else None


def infer_model_label(rows: Sequence[Dict[str, str]], preferred: str) -> str:
    if preferred:
        return normalize_label(preferred)
    for row in rows:
        label = row.get("model_label", "")
        if label:
            return normalize_label(label)
    for row in rows:
        model_path = row.get("model_path", "")
        if model_path:
            return normalize_label(Path(model_path).stem)
    return "model"


def build_metric_specs(
    first_summary: Dict,
    baseline_rows: Optional[Sequence[Dict[str, str]]],
) -> List[Dict]:
    if baseline_rows:
        specs: List[Dict] = []
        seen = set()
        for row in baseline_rows:
            metric_name = str(row.get("metric", "")).strip()
            if not metric_name or metric_name in seen:
                continue
            specs.append(
                {
                    "name": metric_name,
                    "unit": row.get("unit") or None,
                    "direction": row.get("direction") or "higher_better",
                    "note": row.get("note", "") or "",
                }
            )
            seen.add(metric_name)
        return specs

    metric_definitions = first_summary.get("metric_definitions", {}) if isinstance(first_summary, dict) else {}
    return [
        {
            "name": spec["name"],
            "unit": spec.get("unit"),
            "direction": spec.get("direction", "higher_better"),
            "note": metric_definitions.get(spec["name"], ""),
        }
        for spec in DEFAULT_METRICS
    ]


def aggregate_model_metrics(
    metric_specs: Sequence[Dict],
    summaries_by_seed: Dict[str, Dict],
    model_label: str,
    requested_seeds: Sequence[str],
) -> List[Dict]:
    rows: List[Dict] = []
    ordered_seeds = sorted({str(seed) for seed in requested_seeds}, key=lambda s: int(s))
    for spec in metric_specs:
        values_by_seed: Dict[str, float] = {}
        for seed in ordered_seeds:
            summary = summaries_by_seed.get(seed)
            if not summary:
                continue
            value = dotted_get(summary, spec["name"])
            parsed = coerce_float(value)
            if parsed is not None:
                values_by_seed[seed] = parsed
        values = [values_by_seed[seed] for seed in ordered_seeds if seed in values_by_seed]
        rows.append(
            {
                "metric": spec["name"],
                f"{model_label}_mean": _mean(values),
                f"{model_label}_std": _std(values),
                f"{model_label}_min": _min(values),
                f"{model_label}_max": _max(values),
                "direction": spec.get("direction", "higher_better"),
                "unit": spec.get("unit"),
                "seed_count": len(values_by_seed),
                "seeds": "|".join(seed for seed in ordered_seeds if seed in values_by_seed),
                "note": spec.get("note", ""),
                f"{model_label}_values_by_seed": values_by_seed,
            }
        )
    return rows


def build_compare_rows(
    model_rows: Sequence[Dict],
    baseline_rows: Sequence[Dict[str, str]],
    baseline_label: str,
    model_label: str,
) -> List[Dict]:
    baseline_index = {str(row.get("metric", "")): row for row in baseline_rows}
    compare_rows: List[Dict] = []
    for model_row in model_rows:
        metric_name = str(model_row["metric"])
        baseline_row = baseline_index.get(metric_name, {})
        rr_mean = coerce_float(baseline_row.get("rr_mean"))
        baseline_mean = coerce_float(baseline_row.get(f"{baseline_label}_mean"))
        model_mean = coerce_float(model_row.get(f"{model_label}_mean"))
        direction = str(baseline_row.get("direction") or model_row.get("direction") or "higher_better")
        unit = baseline_row.get("unit") or model_row.get("unit")
        note = baseline_row.get("note") or model_row.get("note", "")
        compare_rows.append(
            {
                "metric": metric_name,
                "rr_mean": rr_mean,
                f"{baseline_label}_mean": baseline_mean,
                f"{model_label}_mean": model_mean,
                f"{model_label}_std": model_row.get(f"{model_label}_std"),
                f"{model_label}_min": model_row.get(f"{model_label}_min"),
                f"{model_label}_max": model_row.get(f"{model_label}_max"),
                f"{model_label}_minus_rr_mean": None
                if rr_mean is None or model_mean is None
                else float(model_mean - rr_mean),
                f"{model_label}_over_rr_mean_ratio": safe_ratio(model_mean, rr_mean),
                f"{model_label}_minus_{baseline_label}_mean": None
                if baseline_mean is None or model_mean is None
                else float(model_mean - baseline_mean),
                f"{model_label}_over_{baseline_label}_mean_ratio": safe_ratio(model_mean, baseline_mean),
                "direction": direction,
                f"winner_rr_vs_{baseline_label}_mean": compare_winner(
                    rr_mean,
                    baseline_mean,
                    direction,
                    "rr",
                    baseline_label,
                ),
                f"winner_{model_label}_vs_rr_mean": compare_winner(
                    rr_mean,
                    model_mean,
                    direction,
                    "rr",
                    model_label,
                ),
                f"winner_{model_label}_vs_{baseline_label}_mean": compare_winner(
                    baseline_mean,
                    model_mean,
                    direction,
                    baseline_label,
                    model_label,
                ),
                "unit": unit,
                "seed_count": model_row.get("seed_count"),
                "seeds": model_row.get("seeds"),
                "note": note,
            }
        )
    return compare_rows


def write_model_csv(out_dir: Path, model_label: str, rows: Sequence[Dict]) -> Path:
    path = out_dir / f"{model_label}_kpi_mean.csv"
    fieldnames = [
        "metric",
        f"{model_label}_mean",
        f"{model_label}_std",
        f"{model_label}_min",
        f"{model_label}_max",
        "direction",
        "unit",
        "seed_count",
        "seeds",
        "note",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_model_json(
    out_dir: Path,
    scenario: str,
    source_rows: Sequence[Dict[str, str]],
    model_label: str,
    rows: Sequence[Dict],
) -> Path:
    path = out_dir / f"{model_label}_kpi_mean.json"
    payload = {
        "scenario": scenario,
        "model_label": model_label,
        "seed_count": len(source_rows),
        "seeds": [int(row["seed"]) for row in sorted(source_rows, key=lambda r: int(r["seed"]))],
        "source_runs": list(source_rows),
        "aggregated_metrics": list(rows),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_model_text(
    out_dir: Path,
    scenario: str,
    source_rows: Sequence[Dict[str, str]],
    model_label: str,
    rows: Sequence[Dict],
) -> Path:
    path = out_dir / f"{model_label}_kpi_mean.txt"
    lines = [
        f"scenario: {scenario}",
        f"model_label: {model_label}",
        f"seed_count: {len(source_rows)}",
        "seeds: " + ", ".join(row["seed"] for row in sorted(source_rows, key=lambda r: int(r["seed"]))),
        "",
        "[Source KPI Summaries]",
    ]
    for row in sorted(source_rows, key=lambda r: int(r["seed"])):
        lines.append(f"seed={row['seed']} kpi_summary_json={row['kpi_summary_json']}")
    lines.extend(["", "[Aggregated Model Mean Metrics]"])
    for row in rows:
        lines.append(
            f"{row['metric']}: "
            f"{model_label}_mean={format_value(row.get(f'{model_label}_mean'), row.get('unit'))} "
            f"{model_label}_std={format_value(row.get(f'{model_label}_std'), row.get('unit'))} "
            f"{model_label}_min={format_value(row.get(f'{model_label}_min'), row.get('unit'))} "
            f"{model_label}_max={format_value(row.get(f'{model_label}_max'), row.get('unit'))}"
            + (f" | {row['note']}" if row.get("note") else "")
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_compare_csv(
    out_dir: Path,
    baseline_label: str,
    model_label: str,
    rows: Sequence[Dict],
) -> Path:
    path = out_dir / f"rr_vs_{baseline_label}_vs_{model_label}_compare_mean.csv"
    fieldnames = [
        "metric",
        "rr_mean",
        f"{baseline_label}_mean",
        f"{model_label}_mean",
        f"{model_label}_std",
        f"{model_label}_min",
        f"{model_label}_max",
        f"{model_label}_minus_rr_mean",
        f"{model_label}_over_rr_mean_ratio",
        f"{model_label}_minus_{baseline_label}_mean",
        f"{model_label}_over_{baseline_label}_mean_ratio",
        "direction",
        f"winner_rr_vs_{baseline_label}_mean",
        f"winner_{model_label}_vs_rr_mean",
        f"winner_{model_label}_vs_{baseline_label}_mean",
        "unit",
        "seed_count",
        "seeds",
        "note",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_compare_json(
    out_dir: Path,
    scenario: str,
    source_rows: Sequence[Dict[str, str]],
    baseline_csv_path: Path,
    baseline_label: str,
    model_label: str,
    rows: Sequence[Dict],
) -> Path:
    path = out_dir / f"rr_vs_{baseline_label}_vs_{model_label}_compare_mean.json"
    payload = {
        "scenario": scenario,
        "model_label": model_label,
        "baseline_label": baseline_label,
        "baseline_mean_csv": str(baseline_csv_path),
        "seed_count": len(source_rows),
        "seeds": [int(row["seed"]) for row in sorted(source_rows, key=lambda r: int(r["seed"]))],
        "source_runs": list(source_rows),
        "compare_metrics": list(rows),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_compare_text(
    out_dir: Path,
    scenario: str,
    source_rows: Sequence[Dict[str, str]],
    baseline_csv_path: Path,
    baseline_label: str,
    model_label: str,
    rows: Sequence[Dict],
) -> Path:
    path = out_dir / f"rr_vs_{baseline_label}_vs_{model_label}_compare_mean.txt"
    lines = [
        f"scenario: {scenario}",
        f"baseline_mean_csv: {baseline_csv_path}",
        f"baseline_label: {baseline_label}",
        f"model_label: {model_label}",
        f"seed_count: {len(source_rows)}",
        "seeds: " + ", ".join(row["seed"] for row in sorted(source_rows, key=lambda r: int(r["seed"]))),
        "",
        "[Source KPI Summaries]",
    ]
    for row in sorted(source_rows, key=lambda r: int(r["seed"])):
        lines.append(f"seed={row['seed']} kpi_summary_json={row['kpi_summary_json']}")
    lines.extend(["", "[RR/PFQ/Model Mean Compare]"])
    for row in rows:
        lines.append(
            f"{row['metric']}: "
            f"rr_mean={format_value(row.get('rr_mean'), row.get('unit'))} "
            f"{baseline_label}_mean={format_value(row.get(f'{baseline_label}_mean'), row.get('unit'))} "
            f"{model_label}_mean={format_value(row.get(f'{model_label}_mean'), row.get('unit'))} "
            f"{model_label}_minus_rr_mean={format_value(row.get(f'{model_label}_minus_rr_mean'), row.get('unit'))} "
            f"{model_label}_minus_{baseline_label}_mean={format_value(row.get(f'{model_label}_minus_{baseline_label}_mean'), row.get('unit'))} "
            f"winner_{model_label}_vs_rr_mean={row.get(f'winner_{model_label}_vs_rr_mean', 'unknown')} "
            f"winner_{model_label}_vs_{baseline_label}_mean={row.get(f'winner_{model_label}_vs_{baseline_label}_mean', 'unknown')}"
            + (f" | {row['note']}" if row.get("note") else "")
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_mean_root = None
    if args.baseline_mean_root:
        baseline_mean_root = Path(args.baseline_mean_root).resolve()
    baseline_mean_csv = None
    if args.baseline_mean_csv:
        baseline_mean_csv = Path(args.baseline_mean_csv).resolve()

    manifest_rows = load_manifest_rows(manifest_path)
    if not manifest_rows:
        raise RuntimeError(f"manifest is empty: {manifest_path}")

    model_label = infer_model_label(manifest_rows, args.model_label)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[str(row["scenario"])].append(row)

    aggregate_manifest_rows: List[Dict[str, str]] = []
    for scenario in sorted(grouped.keys()):
        rows = sorted(grouped[scenario], key=lambda r: int(r["seed"]))
        summaries_by_seed = {
            str(row["seed"]): load_json(Path(row["kpi_summary_json"]).resolve())
            for row in rows
        }
        requested_seeds = [str(row["seed"]) for row in rows]

        baseline_csv_path = resolve_baseline_csv_for_scenario(
            scenario,
            baseline_mean_root,
            baseline_mean_csv,
        )
        baseline_payload = load_baseline_csv(baseline_csv_path) if baseline_csv_path else None
        baseline_rows = baseline_payload["rows"] if baseline_payload else None
        baseline_label = baseline_payload["baseline_label"] if baseline_payload else ""

        first_summary = summaries_by_seed[requested_seeds[0]]
        metric_specs = build_metric_specs(first_summary, baseline_rows)
        model_rows = aggregate_model_metrics(metric_specs, summaries_by_seed, model_label, requested_seeds)

        scenario_out_dir = out_root / scenario
        scenario_out_dir.mkdir(parents=True, exist_ok=True)
        model_csv = write_model_csv(scenario_out_dir, model_label, model_rows)
        model_json = write_model_json(scenario_out_dir, scenario, rows, model_label, model_rows)
        model_txt = write_model_text(scenario_out_dir, scenario, rows, model_label, model_rows)

        compare_csv = None
        compare_json = None
        compare_txt = None
        if baseline_payload is not None:
            compare_rows = build_compare_rows(model_rows, baseline_payload["rows"], baseline_label, model_label)
            compare_csv = write_compare_csv(scenario_out_dir, baseline_label, model_label, compare_rows)
            compare_json = write_compare_json(
                scenario_out_dir,
                scenario,
                rows,
                baseline_csv_path,
                baseline_label,
                model_label,
                compare_rows,
            )
            compare_txt = write_compare_text(
                scenario_out_dir,
                scenario,
                rows,
                baseline_csv_path,
                baseline_label,
                model_label,
                compare_rows,
            )

        aggregate_manifest_rows.append(
            {
                "scenario": scenario,
                "seed_count": str(len(rows)),
                "seeds": "|".join(row["seed"] for row in rows),
                "aggregate_dir": str(scenario_out_dir),
                "model_mean_csv": str(model_csv),
                "model_mean_json": str(model_json),
                "model_mean_txt": str(model_txt),
                "baseline_mean_csv": "" if baseline_csv_path is None else str(baseline_csv_path),
                "compare_csv": "" if compare_csv is None else str(compare_csv),
                "compare_json": "" if compare_json is None else str(compare_json),
                "compare_txt": "" if compare_txt is None else str(compare_txt),
            }
        )

    aggregate_manifest_path = out_root / "aggregate_manifest.csv"
    with open(aggregate_manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "seed_count",
                "seeds",
                "aggregate_dir",
                "model_mean_csv",
                "model_mean_json",
                "model_mean_txt",
                "baseline_mean_csv",
                "compare_csv",
                "compare_json",
                "compare_txt",
            ],
        )
        writer.writeheader()
        for row in aggregate_manifest_rows:
            writer.writerow(row)

    aggregate_summary_path = out_root / "aggregate_summary.txt"
    aggregate_summary_path.write_text(
        "\n".join(
            [
                f"manifest={manifest_path}",
                f"model_label={model_label}",
                f"scenario_count={len(aggregate_manifest_rows)}",
                f"aggregate_manifest={aggregate_manifest_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Aggregated model KPI manifest: {aggregate_manifest_path}")
    print(f"Aggregated model KPI summary: {aggregate_summary_path}")


if __name__ == "__main__":
    main()
