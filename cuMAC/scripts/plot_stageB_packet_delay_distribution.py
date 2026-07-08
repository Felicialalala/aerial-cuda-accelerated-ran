#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BINS = "0,1,5,10,20,50,100,150,200,inf"
CONTAINER_ROOT = "/opt/nvidia/cuBB/"

SERIES = [
    {
        "key": "uniform_rr",
        "scenario": "uniform",
        "leg": "uniform_rr_pf",
        "baseline": "rr",
        "label": "Uniform RR",
        "color": "#2563EB",
    },
    {
        "key": "uniform_pf",
        "scenario": "uniform",
        "leg": "uniform_rr_pf",
        "baseline": "pf",
        "label": "Uniform PF",
        "color": "#D97706",
    },
    {
        "key": "uniform_rrq",
        "scenario": "uniform",
        "leg": "uniform_rrq_pfq",
        "baseline": "rrq",
        "label": "Uniform RRQ",
        "color": "#059669",
    },
    {
        "key": "uniform_pfq",
        "scenario": "uniform",
        "leg": "uniform_rrq_pfq",
        "baseline": "pfq",
        "label": "Uniform PFQ",
        "color": "#7C3AED",
    },
    {
        "key": "boundary_rrq",
        "scenario": "boundary",
        "leg": "boundary_rrq_pfq",
        "baseline": "rrq",
        "label": "Boundary RRQ",
        "color": "#16A34A",
    },
    {
        "key": "boundary_pfq",
        "scenario": "boundary",
        "leg": "boundary_rrq_pfq",
        "baseline": "pfq",
        "label": "Boundary PFQ",
        "color": "#DB2777",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Stage-B packet delay distribution from per-run packet_delay_samples.csv files."
    )
    parser.add_argument(
        "--suite-dir",
        default="output/stageB_baseline_eval_suite_fullkpi_seed41_50",
        help="Stage-B baseline eval suite directory.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="packet_delay_samples_manifest_all.csv. Defaults to <suite-dir>/suite_aggregate/packet_delay_samples_manifest_all.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots and binned CSVs. Defaults to <suite-dir>/suite_aggregate/figures.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to map /opt/nvidia/cuBB paths when plotting outside the container.",
    )
    parser.add_argument(
        "--bins-ms",
        default=DEFAULT_BINS,
        help=f"Comma-separated delay bin edges in ms. Use inf for the last open edge. Default: {DEFAULT_BINS}",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG output DPI.")
    return parser.parse_args()


def parse_bins(spec: str) -> list[float]:
    bins: list[float] = []
    for item in spec.split(","):
        token = item.strip().lower()
        if not token:
            continue
        bins.append(math.inf if token in {"inf", "+inf", "infinity"} else float(token))
    if len(bins) < 2:
        raise ValueError("At least two bin edges are required.")
    finite = [v for v in bins if math.isfinite(v)]
    if finite != sorted(finite):
        raise ValueError("Finite bin edges must be sorted.")
    if any(math.isinf(v) for v in bins[:-1]):
        raise ValueError("Only the last bin edge may be inf.")
    return bins


def bin_label(left: float, right: float) -> str:
    if math.isinf(right):
        return f">={left:g}"
    return f"{left:g}-{right:g}"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_sample_path(csv_path: str, repo_root: Path) -> Path:
    path = Path(csv_path)
    if path.exists():
        return path
    if csv_path.startswith(CONTAINER_ROOT):
        mapped = repo_root / csv_path[len(CONTAINER_ROOT) :]
        if mapped.exists():
            return mapped
        return mapped
    if not path.is_absolute():
        mapped = repo_root / path
        if mapped.exists():
            return mapped
    return path


def load_delay_histogram(path: Path, bins: list[float]) -> tuple[np.ndarray, int]:
    delays = pd.read_csv(path, usecols=["delay_ms"])["delay_ms"].to_numpy(dtype=np.float64)
    counts, _ = np.histogram(delays, bins=np.asarray(bins, dtype=np.float64))
    return counts.astype(np.int64), int(delays.size)


def collect_seed_histograms(
    manifest_rows: list[dict[str, str]], repo_root: Path, bins: list[float]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_seed_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []

    bin_labels = [bin_label(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

    for series in SERIES:
        rows = [
            row
            for row in manifest_rows
            if row.get("leg") == series["leg"] and row.get("baseline") == series["baseline"]
        ]
        rows.sort(key=lambda row: int(row["seed"]))
        if not rows:
            raise FileNotFoundError(f"No manifest rows found for {series['label']}.")

        seed_percentages: list[np.ndarray] = []
        seed_counts: list[np.ndarray] = []

        for row in rows:
            sample_path = resolve_sample_path(row["csv"], repo_root)
            if not sample_path.exists():
                raise FileNotFoundError(f"Missing packet delay sample CSV: {sample_path}")
            counts, total = load_delay_histogram(sample_path, bins)
            pct = counts / total * 100.0 if total else np.zeros_like(counts, dtype=np.float64)
            seed_percentages.append(pct)
            seed_counts.append(counts)

            for idx, label in enumerate(bin_labels):
                per_seed_rows.append(
                    {
                        "scenario": series["scenario"],
                        "leg": series["leg"],
                        "seed": row["seed"],
                        "baseline": series["baseline"],
                        "series_key": series["key"],
                        "series_label": series["label"],
                        "bin_label": label,
                        "bin_start_ms": bins[idx],
                        "bin_end_ms": bins[idx + 1],
                        "packet_count": int(counts[idx]),
                        "packet_total": total,
                        "pct": float(pct[idx]),
                        "source_csv": str(sample_path),
                    }
                )

        pct_matrix = np.vstack(seed_percentages)
        count_matrix = np.vstack(seed_counts)
        for idx, label in enumerate(bin_labels):
            aggregate_rows.append(
                {
                    "scenario": series["scenario"],
                    "leg": series["leg"],
                    "baseline": series["baseline"],
                    "series_key": series["key"],
                    "series_label": series["label"],
                    "bin_label": label,
                    "bin_start_ms": bins[idx],
                    "bin_end_ms": bins[idx + 1],
                    "seed_count": len(rows),
                    "packet_count_total": int(count_matrix[:, idx].sum()),
                    "packet_total": int(count_matrix.sum()),
                    "pct_seed_mean": float(pct_matrix[:, idx].mean()),
                    "pct_seed_std": float(pct_matrix[:, idx].std(ddof=0)),
                }
            )

    return per_seed_rows, aggregate_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bars(
    aggregate_rows: list[dict[str, object]],
    series_keys: list[str],
    title: str,
    subtitle: str,
    output_stem: Path,
    *,
    log_y: bool,
    dpi: int,
) -> None:
    selected_series = [series for series in SERIES if series["key"] in series_keys]
    bin_labels = []
    for row in aggregate_rows:
        if row["series_key"] == selected_series[0]["key"]:
            bin_labels.append(str(row["bin_label"]))

    values_by_series: dict[str, list[float]] = {}
    for series in selected_series:
        rows = [row for row in aggregate_rows if row["series_key"] == series["key"]]
        values_by_series[series["key"]] = [float(row["pct_seed_mean"]) for row in rows]

    x = np.arange(len(bin_labels), dtype=np.float64)
    group_width = 0.82
    bar_width = group_width / max(len(selected_series), 1)

    fig_w = max(11.5, len(bin_labels) * 1.05)
    fig, ax = plt.subplots(figsize=(fig_w, 6.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for idx, series in enumerate(selected_series):
        offsets = x - group_width / 2 + bar_width / 2 + idx * bar_width
        ax.bar(
            offsets,
            values_by_series[series["key"]],
            width=bar_width * 0.92,
            label=series["label"],
            color=series["color"],
            edgecolor="white",
            linewidth=0.7,
        )

    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=16)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=10.5, color="#4B5563")
    ax.set_ylabel("Packet share per delay bucket (%)")
    ax.set_xlabel("Packet delay bucket (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=0)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.legend(loc="upper right", ncols=min(len(selected_series), 3), frameon=False)

    all_values = [v for series_values in values_by_series.values() for v in series_values if v > 0]
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(max(min(all_values) * 0.5, 0.001), max(all_values) * 1.8)
    else:
        ax.set_ylim(0, max(all_values) * 1.18)

    fig.tight_layout()
    suffix = "_logy" if log_y else ""
    png = output_stem.with_name(output_stem.name + suffix).with_suffix(".png")
    svg = output_stem.with_name(output_stem.name + suffix).with_suffix(".svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    suite_dir = Path(args.suite_dir)
    if not suite_dir.is_absolute():
        suite_dir = repo_root / suite_dir
    manifest = Path(args.manifest) if args.manifest else suite_dir / "suite_aggregate" / "packet_delay_samples_manifest_all.csv"
    if not manifest.is_absolute():
        manifest = repo_root / manifest
    output_dir = Path(args.output_dir) if args.output_dir else suite_dir / "suite_aggregate" / "figures"
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bins = parse_bins(args.bins_ms)
    manifest_rows = read_manifest(manifest)
    per_seed_rows, aggregate_rows = collect_seed_histograms(manifest_rows, repo_root, bins)

    write_csv(output_dir / "packet_delay_distribution_by_seed.csv", per_seed_rows)
    write_csv(output_dir / "packet_delay_distribution_bins.csv", aggregate_rows)

    subtitle = "Seed 41-50 mean of per-seed packet percentages; bins are left-closed, right-open except the final bin."
    plot_grouped_bars(
        aggregate_rows,
        ["uniform_rr", "uniform_pf", "uniform_rrq", "uniform_pfq"],
        "Uniform Packet Delay Distribution",
        subtitle,
        output_dir / "packet_delay_distribution_uniform",
        log_y=False,
        dpi=args.dpi,
    )
    plot_grouped_bars(
        aggregate_rows,
        ["uniform_rr", "uniform_pf", "uniform_rrq", "uniform_pfq"],
        "Uniform Packet Delay Distribution",
        subtitle,
        output_dir / "packet_delay_distribution_uniform",
        log_y=True,
        dpi=args.dpi,
    )
    plot_grouped_bars(
        aggregate_rows,
        ["uniform_rr", "uniform_rrq"],
        "Uniform Packet Delay Distribution: RR vs RRQ",
        subtitle,
        output_dir / "packet_delay_distribution_uniform_rr_vs_rrq",
        log_y=False,
        dpi=args.dpi,
    )
    plot_grouped_bars(
        aggregate_rows,
        ["uniform_rr", "uniform_rrq"],
        "Uniform Packet Delay Distribution: RR vs RRQ",
        subtitle,
        output_dir / "packet_delay_distribution_uniform_rr_vs_rrq",
        log_y=True,
        dpi=args.dpi,
    )
    plot_grouped_bars(
        aggregate_rows,
        ["uniform_pf", "uniform_pfq"],
        "Uniform Packet Delay Distribution: PF vs PFQ",
        subtitle,
        output_dir / "packet_delay_distribution_uniform_pf_vs_pfq",
        log_y=False,
        dpi=args.dpi,
    )
    plot_grouped_bars(
        aggregate_rows,
        ["uniform_pf", "uniform_pfq"],
        "Uniform Packet Delay Distribution: PF vs PFQ",
        subtitle,
        output_dir / "packet_delay_distribution_uniform_pf_vs_pfq",
        log_y=True,
        dpi=args.dpi,
    )
    plot_grouped_bars(
        aggregate_rows,
        ["boundary_rrq", "boundary_pfq"],
        "Boundary Packet Delay Distribution",
        subtitle,
        output_dir / "packet_delay_distribution_boundary",
        log_y=False,
        dpi=args.dpi,
    )
    plot_grouped_bars(
        aggregate_rows,
        ["boundary_rrq", "boundary_pfq"],
        "Boundary Packet Delay Distribution",
        subtitle,
        output_dir / "packet_delay_distribution_boundary",
        log_y=True,
        dpi=args.dpi,
    )
    plot_grouped_bars(
        aggregate_rows,
        ["boundary_rrq", "boundary_pfq"],
        "Boundary Packet Delay Distribution: RRQ vs PFQ",
        subtitle,
        output_dir / "packet_delay_distribution_boundary_rrq_vs_pfq",
        log_y=False,
        dpi=args.dpi,
    )
    plot_grouped_bars(
        aggregate_rows,
        ["boundary_rrq", "boundary_pfq"],
        "Boundary Packet Delay Distribution: RRQ vs PFQ",
        subtitle,
        output_dir / "packet_delay_distribution_boundary_rrq_vs_pfq",
        log_y=True,
        dpi=args.dpi,
    )
    plot_grouped_bars(
        aggregate_rows,
        ["uniform_rr", "uniform_pf", "uniform_rrq", "uniform_pfq", "boundary_rrq", "boundary_pfq"],
        "Requested Packet Delay Distribution Comparison",
        subtitle,
        output_dir / "packet_delay_distribution_all_requested",
        log_y=False,
        dpi=args.dpi,
    )
    plot_grouped_bars(
        aggregate_rows,
        ["uniform_rr", "uniform_pf", "uniform_rrq", "uniform_pfq", "boundary_rrq", "boundary_pfq"],
        "Requested Packet Delay Distribution Comparison",
        subtitle,
        output_dir / "packet_delay_distribution_all_requested",
        log_y=True,
        dpi=args.dpi,
    )

    print(f"Wrote plots and binned CSVs to: {output_dir}")


if __name__ == "__main__":
    main()
