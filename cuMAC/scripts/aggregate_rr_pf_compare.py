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
from typing import Dict, Iterable, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate multi-seed Stage-B RR/RRQ vs PF/PFQ compare outputs.")
    p.add_argument("--manifest", required=True, help="CSV manifest produced by run_stageB_rr_pf_multi_seed_compare.sh")
    p.add_argument("--output-dir", required=True, help="Directory to write aggregated scenario compare outputs")
    p.add_argument(
        "--reference-baseline",
        default="",
        help="Optional label for the first/reference baseline (default: infer from manifest / compare JSON)",
    )
    p.add_argument(
        "--compare-baseline",
        default="",
        help="Optional label for the second baseline (default: infer from manifest / compare JSON)",
    )
    return p.parse_args()


def normalize_compare_baseline(value: str) -> str:
    baseline = (value or "pf").strip().lower()
    if not baseline:
        return "pf"
    safe = "".join(ch if ch.isalnum() else "_" for ch in baseline)
    return safe or "pf"


def compare_display_name(compare_baseline: str) -> str:
    return compare_baseline.upper()


def comparison_stem(reference_baseline: str, compare_baseline: str) -> str:
    return f"{reference_baseline}_vs_{compare_baseline}_compare"


def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


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


def winner_by_direction(
    reference_mean: Optional[float],
    other_mean: Optional[float],
    direction: str,
    reference_baseline: str,
    compare_baseline: str,
) -> str:
    if reference_mean is None or other_mean is None:
        return "unknown"
    if abs(other_mean - reference_mean) <= 1.0e-12:
        return "tie"
    if direction == "higher_better":
        return compare_baseline if other_mean > reference_mean else reference_baseline
    return compare_baseline if other_mean < reference_mean else reference_baseline


def aggregate_metric_rows(
    metric_rows_by_seed: Dict[str, List[Dict]],
    reference_baseline: str,
    compare_baseline: str,
) -> List[Dict]:
    ordered_metric_names: List[str] = []
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for seed in sorted(metric_rows_by_seed.keys(), key=lambda s: int(s)):
        for row in metric_rows_by_seed[seed]:
            name = str(row.get("name", ""))
            if not name:
                continue
            if name not in grouped:
                ordered_metric_names.append(name)
            grouped[name].append({"seed": seed, **row})

    aggregated: List[Dict] = []
    for metric_name in ordered_metric_names:
        rows = grouped[metric_name]
        reference_values = {
            str(row["seed"]): float(row.get("reference", row.get("rr")))
            for row in rows
            if row.get("reference", row.get("rr")) is not None and row.get("reference", row.get("rr")) != ""
        }
        other_values = {
            str(row["seed"]): float(row.get("compare", row.get("pf")))
            for row in rows
            if row.get("compare", row.get("pf")) is not None and row.get("compare", row.get("pf")) != ""
        }
        reference_list = list(reference_values.values())
        other_list = list(other_values.values())
        reference_mean = _mean(reference_list)
        other_mean = _mean(other_list)
        direction = str(rows[0].get("direction", "higher_better"))
        unit = rows[0].get("unit")
        note = rows[0].get("note", "")
        aggregated.append(
            {
                "metric": metric_name,
                f"{reference_baseline}_mean": reference_mean,
                f"{reference_baseline}_std": _std(reference_list),
                f"{reference_baseline}_min": _min(reference_list),
                f"{reference_baseline}_max": _max(reference_list),
                f"{compare_baseline}_mean": other_mean,
                f"{compare_baseline}_std": _std(other_list),
                f"{compare_baseline}_min": _min(other_list),
                f"{compare_baseline}_max": _max(other_list),
                f"{compare_baseline}_minus_{reference_baseline}_mean": None
                if reference_mean is None or other_mean is None
                else float(other_mean - reference_mean),
                f"{compare_baseline}_over_{reference_baseline}_mean_ratio": safe_ratio(other_mean, reference_mean),
                "direction": direction,
                "winner_by_mean": winner_by_direction(
                    reference_mean, other_mean, direction, reference_baseline, compare_baseline
                ),
                "unit": unit,
                "seed_count": len(sorted(set(reference_values) | set(other_values))),
                "seeds": "|".join(sorted(set(reference_values) | set(other_values), key=lambda s: int(s))),
                "note": note,
                f"{reference_baseline}_values_by_seed": reference_values,
                f"{compare_baseline}_values_by_seed": other_values,
            }
        )
    return aggregated


def write_csv(out_dir: Path, reference_baseline: str, compare_baseline: str, aggregated_metrics: List[Dict]) -> Path:
    path = out_dir / f"{comparison_stem(reference_baseline, compare_baseline)}_mean.csv"
    fieldnames = [
        "metric",
        f"{reference_baseline}_mean",
        f"{reference_baseline}_std",
        f"{reference_baseline}_min",
        f"{reference_baseline}_max",
        f"{compare_baseline}_mean",
        f"{compare_baseline}_std",
        f"{compare_baseline}_min",
        f"{compare_baseline}_max",
        f"{compare_baseline}_minus_{reference_baseline}_mean",
        f"{compare_baseline}_over_{reference_baseline}_mean_ratio",
        "direction",
        "winner_by_mean",
        "unit",
        "seed_count",
        "seeds",
        "note",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_metrics:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_json(
    out_dir: Path,
    scenario: str,
    reference_baseline: str,
    compare_baseline: str,
    source_rows: List[Dict[str, str]],
    aggregated_metrics: List[Dict],
) -> Path:
    path = out_dir / f"{comparison_stem(reference_baseline, compare_baseline)}_mean.json"
    payload = {
        "scenario": scenario,
        "reference_baseline": reference_baseline,
        "reference_display_name": compare_display_name(reference_baseline),
        "compare_baseline": compare_baseline,
        "compare_display_name": compare_display_name(compare_baseline),
        "seed_count": len(source_rows),
        "seeds": [int(row["seed"]) for row in sorted(source_rows, key=lambda r: int(r["seed"]))],
        "source_runs": source_rows,
        "aggregated_metrics": aggregated_metrics,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_text(
    out_dir: Path,
    scenario: str,
    reference_baseline: str,
    compare_baseline: str,
    source_rows: List[Dict[str, str]],
    aggregated_metrics: List[Dict],
) -> Path:
    path = out_dir / f"{comparison_stem(reference_baseline, compare_baseline)}_mean.txt"
    reference_display_name = compare_display_name(reference_baseline)
    display_name = compare_display_name(compare_baseline)
    lines = [
        f"scenario: {scenario}",
        f"reference_baseline: {reference_baseline}",
        f"compare_baseline: {compare_baseline}",
        f"seed_count: {len(source_rows)}",
        "seeds: " + ", ".join(row["seed"] for row in sorted(source_rows, key=lambda r: int(r["seed"]))),
        "",
        "[Source Compare JSON]",
    ]
    for row in sorted(source_rows, key=lambda r: int(r["seed"])):
        lines.append(f"seed={row['seed']} compare_json={row['compare_json']}")

    lines.extend(
        [
            "",
            f"[{reference_display_name} vs {display_name} Mean Metrics]",
        ]
    )
    for row in aggregated_metrics:
        lines.append(
            f"{row['metric']}: "
            f"{reference_baseline}_mean={format_value(row.get(f'{reference_baseline}_mean'), row.get('unit'))} "
            f"{reference_baseline}_std={format_value(row.get(f'{reference_baseline}_std'), row.get('unit'))} "
            f"{compare_baseline}_mean={format_value(row.get(f'{compare_baseline}_mean'), row.get('unit'))} "
            f"{compare_baseline}_std={format_value(row.get(f'{compare_baseline}_std'), row.get('unit'))} "
            f"{compare_baseline}_minus_{reference_baseline}_mean={format_value(row.get(f'{compare_baseline}_minus_{reference_baseline}_mean'), row.get('unit'))} "
            f"{compare_baseline}_over_{reference_baseline}_mean_ratio={format_value(row.get(f'{compare_baseline}_over_{reference_baseline}_mean_ratio'))} "
            f"winner_by_mean={row['winner_by_mean']}"
            + (f" | {row['note']}" if row.get("note") else "")
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def infer_reference_baseline(rows: Sequence[Dict[str, str]], preferred: str) -> str:
    if preferred:
        return normalize_compare_baseline(preferred)
    for row in rows:
        baseline = row.get("reference_baseline", "")
        if baseline:
            return normalize_compare_baseline(baseline)
    for row in rows:
        payload = load_json(Path(row["compare_json"]))
        baseline = payload.get("reference_baseline", "")
        if baseline:
            return normalize_compare_baseline(str(baseline))
    return "rr"


def infer_compare_baseline(rows: Sequence[Dict[str, str]], preferred: str) -> str:
    if preferred:
        return normalize_compare_baseline(preferred)
    for row in rows:
        baseline = row.get("other_baseline", "")
        if baseline:
            return normalize_compare_baseline(baseline)
    for row in rows:
        payload = load_json(Path(row["compare_json"]))
        baseline = payload.get("compare_baseline", "")
        if baseline:
            return normalize_compare_baseline(str(baseline))
    return "pf"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest_rows(manifest_path)
    if not manifest_rows:
        raise RuntimeError(f"manifest is empty: {manifest_path}")

    reference_baseline = infer_reference_baseline(manifest_rows, args.reference_baseline)
    compare_baseline = infer_compare_baseline(manifest_rows, args.compare_baseline)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[str(row["scenario"])].append(row)

    aggregate_manifest_rows: List[Dict[str, str]] = []
    for scenario in sorted(grouped.keys()):
        rows = sorted(grouped[scenario], key=lambda r: int(r["seed"]))
        metric_rows_by_seed: Dict[str, List[Dict]] = {}
        for row in rows:
            compare_json_path = Path(row["compare_json"]).resolve()
            payload = load_json(compare_json_path)
            metric_rows = payload.get("rr_vs_pf_metrics") or payload.get("rr_vs_compare_metrics") or []
            metric_rows_by_seed[str(row["seed"])] = list(metric_rows)

        aggregated_metrics = aggregate_metric_rows(metric_rows_by_seed, reference_baseline, compare_baseline)
        scenario_out_dir = out_root / scenario
        scenario_out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = write_csv(scenario_out_dir, reference_baseline, compare_baseline, aggregated_metrics)
        json_path = write_json(scenario_out_dir, scenario, reference_baseline, compare_baseline, rows, aggregated_metrics)
        txt_path = write_text(scenario_out_dir, scenario, reference_baseline, compare_baseline, rows, aggregated_metrics)
        aggregate_manifest_rows.append(
            {
                "scenario": scenario,
                "seed_count": str(len(rows)),
                "seeds": "|".join(row["seed"] for row in rows),
                "aggregate_dir": str(scenario_out_dir),
                "aggregate_csv": str(csv_path),
                "aggregate_json": str(json_path),
                "aggregate_txt": str(txt_path),
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
                "aggregate_csv",
                "aggregate_json",
                "aggregate_txt",
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
                f"reference_baseline={reference_baseline}",
                f"compare_baseline={compare_baseline}",
                f"scenario_count={len(aggregate_manifest_rows)}",
                f"aggregate_manifest={aggregate_manifest_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Aggregated {compare_display_name(reference_baseline)} vs "
        f"{compare_display_name(compare_baseline)} compare manifest: {aggregate_manifest_path}"
    )
    print(
        f"Aggregated {compare_display_name(reference_baseline)} vs "
        f"{compare_display_name(compare_baseline)} compare summary: {aggregate_summary_path}"
    )


if __name__ == "__main__":
    main()
