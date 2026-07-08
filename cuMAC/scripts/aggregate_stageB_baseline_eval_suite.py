#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_ROOT = Path(__file__).resolve().parents[2]


METRICS = [
    ("traffic.served_mbps_est", "Mbps", "higher_better"),
    ("traffic.goodput_mbps", "Mbps", "higher_better"),
    ("traffic.packet_effective_service_rate_mbps", "Mbps", "higher_better"),
    ("traffic.packet_effective_service_rate_per_packet_mean_mbps", "Mbps", "higher_better"),
    ("traffic.ue_macro_packet_effective_service_rate_mbps", "Mbps", "higher_better"),
    ("traffic.ue_macro_packet_effective_service_rate_per_packet_mean_mbps", "Mbps", "higher_better"),
    ("traffic.packet_delay_mean_ms", "ms", "lower_better"),
    ("traffic.packet_delay_p50_ms", "ms", "lower_better"),
    ("traffic.packet_delay_p90_ms", "ms", "lower_better"),
    ("traffic.packet_delay_p95_ms", "ms", "lower_better"),
    ("traffic.packet_delay_max_ms", "ms", "lower_better"),
    ("traffic.ue_macro_packet_delay_mean_ms", "ms", "lower_better"),
    ("global_kpi.average_ue_goodput_mbps", "Mbps", "higher_better"),
    ("global_kpi.ue_goodput_p5_mbps", "Mbps", "higher_better"),
    ("global_kpi.ue_goodput_p10_mbps", "Mbps", "higher_better"),
    ("global_kpi.ue_goodput_jain", None, "higher_better"),
    ("global_kpi.global_tb_bler", "%", "lower_better"),
    ("traffic.expiry_drop_rate", "%", "lower_better"),
    ("global_kpi.prg_utilization_ratio", "%", "higher_better"),
]


UNIFORM_PAIRS = [
    ("pf", "rr"),
    ("pfq", "rrq"),
    ("rr", "rrq"),
    ("rr", "pfq"),
    ("pf", "rrq"),
    ("pf", "pfq"),
]

BOUNDARY_PAIRS = [("pfq", "rrq")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate the Stage-B baseline eval suite into four-baseline tables.")
    p.add_argument("--suite-dir", required=True, help="Output root produced by run_stageB_baseline_eval_suite.sh")
    p.add_argument("--scenario", default="RAYLEIGH", help="Scenario subdir/name to aggregate (default: RAYLEIGH)")
    return p.parse_args()


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remap_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    old_prefix = "/opt/nvidia/cuBB"
    if path_text.startswith(old_prefix):
        mapped = SCRIPT_ROOT / path_text[len(old_prefix) :].lstrip("/")
        if mapped.exists():
            return mapped
    return path


def summary_path(run_dir_text: str) -> Path:
    path = remap_existing_path(run_dir_text)
    if path.is_dir():
        return path / "kpi_summary.json"
    return path


def nested_get(data: Dict, dotted_key: str) -> Optional[float]:
    cur = data
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if cur is None or cur == "":
        return None
    return float(cur)


def mean(values: Sequence[float]) -> Optional[float]:
    return None if not values else float(sum(values) / len(values))


def stdev(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.pstdev(values))


def fmt(value: Optional[float], unit: Optional[str]) -> str:
    if value is None:
        return "N/A"
    if unit == "%":
        return f"{100.0 * value:.3f}%"
    if unit == "ms":
        return f"{value:.3f} ms"
    if unit == "Mbps":
        return f"{value:.6f} Mbps"
    return f"{value:.6f}"


def read_leg_runs(suite_dir: Path, leg: str, scenario: str) -> Dict[str, Dict[int, Dict]]:
    manifest = suite_dir / leg / "scenario_seed_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing suite leg manifest: {manifest}")

    runs: Dict[str, Dict[int, Dict]] = {}
    for row in load_csv_rows(manifest):
        if row.get("scenario") != scenario:
            continue
        seed = int(row["seed"])
        for baseline_key, dir_key in (
            (row["reference_baseline"], "reference_dir"),
            (row["other_baseline"], "other_dir"),
        ):
            baseline = baseline_key.strip().lower()
            path = summary_path(row[dir_key])
            if not path.exists():
                raise FileNotFoundError(f"Missing kpi_summary.json for {leg} seed={seed} {baseline}: {path}")
            runs.setdefault(baseline, {})[seed] = load_json(path)
    return runs


def merge_runs(*run_sets: Dict[str, Dict[int, Dict]]) -> Dict[str, Dict[int, Dict]]:
    merged: Dict[str, Dict[int, Dict]] = {}
    for run_set in run_sets:
        for baseline, by_seed in run_set.items():
            merged.setdefault(baseline, {}).update(by_seed)
    return merged


def collect_metric_values(runs: Dict[str, Dict[int, Dict]], metric: str) -> Dict[str, Dict[int, float]]:
    values: Dict[str, Dict[int, float]] = {}
    for baseline, by_seed in runs.items():
        for seed, summary in by_seed.items():
            value = nested_get(summary, metric)
            if value is not None:
                values.setdefault(baseline, {})[seed] = value
    return values


def build_metric_rows(
    scenario: str,
    runs: Dict[str, Dict[int, Dict]],
    baselines: Sequence[str],
    pairs: Sequence[Tuple[str, str]],
) -> List[Dict]:
    rows: List[Dict] = []
    for metric, unit, direction in METRICS:
        values = collect_metric_values(runs, metric)
        row: Dict = {
            "scenario": scenario,
            "metric": metric,
            "unit": unit,
            "direction": direction,
        }
        seed_sets = []
        for baseline in baselines:
            baseline_values = values.get(baseline, {})
            seed_sets.append(set(baseline_values.keys()))
            arr = [baseline_values[s] for s in sorted(baseline_values)]
            row[f"{baseline}_mean"] = mean(arr)
            row[f"{baseline}_std"] = stdev(arr)
            row[f"{baseline}_seed_count"] = len(arr)
            row[f"{baseline}_seeds"] = "|".join(str(s) for s in sorted(baseline_values))
        row["common_seed_count"] = len(set.intersection(*seed_sets)) if seed_sets else 0
        row["common_seeds"] = "|".join(str(s) for s in sorted(set.intersection(*seed_sets))) if seed_sets else ""
        for lhs, rhs in pairs:
            lhs_values = values.get(lhs, {})
            rhs_values = values.get(rhs, {})
            common = sorted(set(lhs_values) & set(rhs_values))
            deltas = [lhs_values[s] - rhs_values[s] for s in common]
            row[f"{lhs}_minus_{rhs}_mean"] = mean(deltas)
            row[f"{lhs}_minus_{rhs}_std"] = stdev(deltas)
            row[f"{lhs}_minus_{rhs}_seed_count"] = len(deltas)
        rows.append(row)
    return rows


def write_metric_csv(path: Path, rows: List[Dict], baselines: Sequence[str], pairs: Sequence[Tuple[str, str]]) -> None:
    fieldnames = ["scenario", "metric", "unit", "direction", "common_seed_count", "common_seeds"]
    for baseline in baselines:
        fieldnames.extend([f"{baseline}_mean", f"{baseline}_std", f"{baseline}_seed_count", f"{baseline}_seeds"])
    for lhs, rhs in pairs:
        fieldnames.extend([f"{lhs}_minus_{rhs}_mean", f"{lhs}_minus_{rhs}_std", f"{lhs}_minus_{rhs}_seed_count"])
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_metric_json(path: Path, rows: List[Dict], baselines: Sequence[str], pairs: Sequence[Tuple[str, str]]) -> None:
    payload = {
        "baselines": list(baselines),
        "pairs": [{"lhs": lhs, "rhs": rhs, "delta": f"{lhs}_minus_{rhs}"} for lhs, rhs in pairs],
        "metrics": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_metric_txt(path: Path, title: str, rows: List[Dict], baselines: Sequence[str], pairs: Sequence[Tuple[str, str]]) -> None:
    lines = [title, ""]
    for row in rows:
        unit = row.get("unit")
        parts = [row["metric"]]
        for baseline in baselines:
            parts.append(f"{baseline}={fmt(row.get(f'{baseline}_mean'), unit)}")
        for lhs, rhs in pairs:
            parts.append(f"{lhs}_minus_{rhs}={fmt(row.get(f'{lhs}_minus_{rhs}_mean'), unit)}")
        lines.append(" | ".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def packet_delay_sample_rows(suite_dir: Path, legs: Iterable[str], scenario: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    for leg in legs:
        runs = read_leg_runs(suite_dir, leg, scenario)
        for baseline, by_seed in runs.items():
            for seed, summary in sorted(by_seed.items()):
                key = (scenario, leg, seed, baseline)
                if key in seen:
                    continue
                seen.add(key)
                traffic = summary.get("traffic", {}) if isinstance(summary.get("traffic"), dict) else {}
                csv_path = traffic.get("packet_delay_samples_csv")
                rows.append(
                    {
                        "scenario": scenario,
                        "leg": leg,
                        "seed": str(seed),
                        "baseline": baseline,
                        "csv": "" if csv_path is None else str(csv_path),
                        "rows": "" if traffic.get("packet_delay_samples_rows") is None else str(traffic.get("packet_delay_samples_rows")),
                        "exists": str(bool(csv_path and remap_existing_path(str(csv_path)).exists())).lower(),
                    }
                )
    return rows


def write_sample_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = ["scenario", "leg", "seed", "baseline", "csv", "rows", "exists"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    scenario = args.scenario
    out_dir = suite_dir / "suite_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    uniform_rrq_pfq = read_leg_runs(suite_dir, "uniform_rrq_pfq", scenario)
    uniform_rr_pf = read_leg_runs(suite_dir, "uniform_rr_pf", scenario)
    uniform_runs = merge_runs(uniform_rr_pf, uniform_rrq_pfq)
    uniform_baselines = ["rr", "pf", "rrq", "pfq"]
    uniform_rows = build_metric_rows(scenario, uniform_runs, uniform_baselines, UNIFORM_PAIRS)
    write_metric_csv(out_dir / "uniform_four_baseline_metrics.csv", uniform_rows, uniform_baselines, UNIFORM_PAIRS)
    write_metric_json(out_dir / "uniform_four_baseline_metrics.json", uniform_rows, uniform_baselines, UNIFORM_PAIRS)
    write_metric_txt(
        out_dir / "uniform_four_baseline_metrics.txt",
        "Uniform RR/PF/RRQ/PFQ seed-mean metrics",
        uniform_rows,
        uniform_baselines,
        UNIFORM_PAIRS,
    )

    boundary_runs = read_leg_runs(suite_dir, "boundary_rrq_pfq", scenario)
    boundary_baselines = ["rrq", "pfq"]
    boundary_rows = build_metric_rows(scenario, boundary_runs, boundary_baselines, BOUNDARY_PAIRS)
    write_metric_csv(out_dir / "boundary_rrq_pfq_metrics.csv", boundary_rows, boundary_baselines, BOUNDARY_PAIRS)
    write_metric_json(out_dir / "boundary_rrq_pfq_metrics.json", boundary_rows, boundary_baselines, BOUNDARY_PAIRS)
    write_metric_txt(
        out_dir / "boundary_rrq_pfq_metrics.txt",
        "Boundary RRQ/PFQ seed-mean metrics",
        boundary_rows,
        boundary_baselines,
        BOUNDARY_PAIRS,
    )

    samples = packet_delay_sample_rows(
        suite_dir,
        ["uniform_rrq_pfq", "boundary_rrq_pfq", "uniform_rr_pf"],
        scenario,
    )
    write_sample_manifest(out_dir / "packet_delay_samples_manifest_all.csv", samples)

    summary = [
        f"suite_dir={suite_dir}",
        f"scenario={scenario}",
        f"uniform_four_baseline_metrics={out_dir / 'uniform_four_baseline_metrics.csv'}",
        f"boundary_rrq_pfq_metrics={out_dir / 'boundary_rrq_pfq_metrics.csv'}",
        f"packet_delay_samples_manifest_all={out_dir / 'packet_delay_samples_manifest_all.csv'}",
    ]
    (out_dir / "suite_aggregate_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Stage-B baseline suite aggregate written: {out_dir}")


if __name__ == "__main__":
    main()
