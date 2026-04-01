#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Compare Stage-B RR and PF-like KPI summaries.")
    p.add_argument("--rr", required=True, help="RR run directory or RR kpi_summary.json path")
    p.add_argument("--pf", required=True, help="PF run directory or PF kpi_summary.json path")
    p.add_argument("--output-dir", required=True, help="Directory to write comparison outputs")
    p.add_argument("--top-n", type=int, default=10, help="Top-N UE deltas to include in the text summary")
    p.add_argument(
        "--compare-baseline",
        default="pf",
        help="Label for the second baseline in output file names and columns (default: pf)",
    )
    return p.parse_args()


def normalize_compare_baseline(value):
    baseline = (value or "pf").strip().lower()
    if not baseline:
        return "pf"
    safe = "".join(ch if ch.isalnum() else "_" for ch in baseline)
    return safe or "pf"


def compare_display_name(compare_baseline):
    return compare_baseline.upper()


def resolve_summary_path(path_str):
    path = Path(path_str)
    if path.is_dir():
        summary = path / "kpi_summary.json"
        if summary.exists():
            return summary
        raise FileNotFoundError(f"Cannot find kpi_summary.json in {path}")
    if path.is_file():
        return path
    raise FileNotFoundError(f"Path does not exist: {path}")


def load_summary(path_str):
    summary_path = resolve_summary_path(path_str)
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_summary_path"] = str(summary_path)
    data["_run_name"] = summary_path.parent.name
    return data


def safe_get(dct, *keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def safe_ratio(num, den):
    if den is None or abs(den) <= 1.0e-12:
        return None
    return num / den


def metric_note(summary, name):
    return safe_get(summary, "metric_definitions", name, default="")


def format_value(v, unit=None):
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
    return f"{v:.10f}" if isinstance(v, float) else str(v)


def build_metric_rows(rr, pf, compare_baseline):
    metrics = [
        {
            "name": "traffic.served_mbps_est",
            "unit": "Mbps",
            "direction": "higher_better",
            "rr": safe_get(rr, "traffic", "served_mbps_est"),
            "pf": safe_get(pf, "traffic", "served_mbps_est"),
        },
        {
            "name": "traffic.goodput_mbps",
            "unit": "Mbps",
            "direction": "higher_better",
            "rr": safe_get(rr, "traffic", "goodput_mbps"),
            "pf": safe_get(pf, "traffic", "goodput_mbps"),
        },
        {
            "name": "global_kpi.cluster_sum_throughput_mbps",
            "unit": "Mbps",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "cluster_sum_throughput_mbps"),
            "pf": safe_get(pf, "global_kpi", "cluster_sum_throughput_mbps"),
        },
        {
            "name": "global_kpi.cluster_goodput_mbps",
            "unit": "Mbps",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "cluster_goodput_mbps"),
            "pf": safe_get(pf, "global_kpi", "cluster_goodput_mbps"),
        },
        {
            "name": "global_kpi.cluster_spectral_efficiency_bps_per_hz",
            "unit": "bps/Hz",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "cluster_spectral_efficiency_bps_per_hz"),
            "pf": safe_get(pf, "global_kpi", "cluster_spectral_efficiency_bps_per_hz"),
        },
        {
            "name": "global_kpi.average_ue_throughput_mbps",
            "unit": "Mbps",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "average_ue_throughput_mbps"),
            "pf": safe_get(pf, "global_kpi", "average_ue_throughput_mbps"),
        },
        {
            "name": "global_kpi.average_ue_goodput_mbps",
            "unit": "Mbps",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "average_ue_goodput_mbps"),
            "pf": safe_get(pf, "global_kpi", "average_ue_goodput_mbps"),
        },
        {
            "name": "global_kpi.ue_throughput_jain",
            "unit": None,
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "ue_throughput_jain"),
            "pf": safe_get(pf, "global_kpi", "ue_throughput_jain"),
        },
        {
            "name": "global_kpi.ue_goodput_jain",
            "unit": None,
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "ue_goodput_jain"),
            "pf": safe_get(pf, "global_kpi", "ue_goodput_jain"),
        },
        {
            "name": "global_kpi.ue_throughput_p5_mbps",
            "unit": "Mbps",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "ue_throughput_p5_mbps"),
            "pf": safe_get(pf, "global_kpi", "ue_throughput_p5_mbps"),
        },
        {
            "name": "global_kpi.ue_goodput_p5_mbps",
            "unit": "Mbps",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "ue_goodput_p5_mbps"),
            "pf": safe_get(pf, "global_kpi", "ue_goodput_p5_mbps"),
        },
        {
            "name": "global_kpi.cell_edge_spectral_efficiency_p5_bps_per_hz",
            "unit": "bps/Hz",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "cell_edge_spectral_efficiency_p5_bps_per_hz"),
            "pf": safe_get(pf, "global_kpi", "cell_edge_spectral_efficiency_p5_bps_per_hz"),
        },
        {
            "name": "global_kpi.ue_throughput_p10_mbps",
            "unit": "Mbps",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "ue_throughput_p10_mbps"),
            "pf": safe_get(pf, "global_kpi", "ue_throughput_p10_mbps"),
        },
        {
            "name": "global_kpi.ue_goodput_p10_mbps",
            "unit": "Mbps",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "ue_goodput_p10_mbps"),
            "pf": safe_get(pf, "global_kpi", "ue_goodput_p10_mbps"),
        },
        {
            "name": "global_kpi.global_tb_bler",
            "unit": "%",
            "direction": "lower_better",
            "rr": safe_get(rr, "global_kpi", "global_tb_bler"),
            "pf": safe_get(pf, "global_kpi", "global_tb_bler"),
        },
        {
            "name": "global_kpi.global_tx_success_rate",
            "unit": "%",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "global_tx_success_rate"),
            "pf": safe_get(pf, "global_kpi", "global_tx_success_rate"),
        },
        {
            "name": "global_kpi.residual_buffer_ratio",
            "unit": "%",
            "direction": "lower_better",
            "rr": safe_get(rr, "global_kpi", "residual_buffer_ratio"),
            "pf": safe_get(pf, "global_kpi", "residual_buffer_ratio"),
        },
        {
            "name": "global_kpi.served_buffer_ratio",
            "unit": "%",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "served_buffer_ratio"),
            "pf": safe_get(pf, "global_kpi", "served_buffer_ratio"),
        },
        {
            "name": "global_kpi.backlog_free_ue_ratio",
            "unit": "%",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "backlog_free_ue_ratio"),
            "pf": safe_get(pf, "global_kpi", "backlog_free_ue_ratio"),
        },
        {
            "name": "global_kpi.prg_utilization_ratio",
            "unit": "%",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "prg_utilization_ratio"),
            "pf": safe_get(pf, "global_kpi", "prg_utilization_ratio"),
        },
        {
            "name": "traffic.queue_delay_est_ms",
            "unit": "ms",
            "direction": "lower_better",
            "rr": safe_get(rr, "traffic", "queue_delay_est_ms"),
            "pf": safe_get(pf, "traffic", "queue_delay_est_ms"),
        },
        {
            "name": "traffic.packet_delay_served_pkt_count",
            "unit": None,
            "direction": "higher_better",
            "rr": safe_get(rr, "traffic", "packet_delay_served_pkt_count"),
            "pf": safe_get(pf, "traffic", "packet_delay_served_pkt_count"),
        },
        {
            "name": "traffic.packet_delay_pending_pkt_count",
            "unit": None,
            "direction": "lower_better",
            "rr": safe_get(rr, "traffic", "packet_delay_pending_pkt_count"),
            "pf": safe_get(pf, "traffic", "packet_delay_pending_pkt_count"),
        },
        {
            "name": "traffic.packet_delay_mean_ms",
            "unit": "ms",
            "direction": "lower_better",
            "rr": safe_get(rr, "traffic", "packet_delay_mean_ms"),
            "pf": safe_get(pf, "traffic", "packet_delay_mean_ms"),
        },
        {
            "name": "traffic.packet_delay_p50_ms",
            "unit": "ms",
            "direction": "lower_better",
            "rr": safe_get(rr, "traffic", "packet_delay_p50_ms"),
            "pf": safe_get(pf, "traffic", "packet_delay_p50_ms"),
        },
        {
            "name": "traffic.packet_delay_p90_ms",
            "unit": "ms",
            "direction": "lower_better",
            "rr": safe_get(rr, "traffic", "packet_delay_p90_ms"),
            "pf": safe_get(pf, "traffic", "packet_delay_p90_ms"),
        },
        {
            "name": "traffic.packet_delay_p95_ms",
            "unit": "ms",
            "direction": "lower_better",
            "rr": safe_get(rr, "traffic", "packet_delay_p95_ms"),
            "pf": safe_get(pf, "traffic", "packet_delay_p95_ms"),
        },
        {
            "name": "global_kpi.queue_delay_p50_ms",
            "unit": "ms",
            "direction": "lower_better",
            "rr": safe_get(rr, "global_kpi", "queue_delay_p50_ms"),
            "pf": safe_get(pf, "global_kpi", "queue_delay_p50_ms"),
        },
        {
            "name": "global_kpi.queue_delay_p90_ms",
            "unit": "ms",
            "direction": "lower_better",
            "rr": safe_get(rr, "global_kpi", "queue_delay_p90_ms"),
            "pf": safe_get(pf, "global_kpi", "queue_delay_p90_ms"),
        },
        {
            "name": "global_kpi.queue_delay_p95_ms",
            "unit": "ms",
            "direction": "lower_better",
            "rr": safe_get(rr, "global_kpi", "queue_delay_p95_ms"),
            "pf": safe_get(pf, "global_kpi", "queue_delay_p95_ms"),
        },
        {
            "name": "global_kpi.ue_fraction_queue_delay_gt_10s",
            "unit": "%",
            "direction": "lower_better",
            "rr": safe_get(rr, "global_kpi", "ue_fraction_queue_delay_gt_10s"),
            "pf": safe_get(pf, "global_kpi", "ue_fraction_queue_delay_gt_10s"),
        },
        {
            "name": "global_kpi.ue_fraction_queue_delay_gt_20s",
            "unit": "%",
            "direction": "lower_better",
            "rr": safe_get(rr, "global_kpi", "ue_fraction_queue_delay_gt_20s"),
            "pf": safe_get(pf, "global_kpi", "ue_fraction_queue_delay_gt_20s"),
        },
        {
            "name": "global_kpi.scheduled_ratio_mean",
            "unit": "%",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "scheduled_ratio_mean"),
            "pf": safe_get(pf, "global_kpi", "scheduled_ratio_mean"),
        },
        {
            "name": "global_kpi.scheduled_ratio_jain",
            "unit": None,
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "scheduled_ratio_jain"),
            "pf": safe_get(pf, "global_kpi", "scheduled_ratio_jain"),
        },
        {
            "name": "global_kpi.scheduled_ratio_p5",
            "unit": "%",
            "direction": "higher_better",
            "rr": safe_get(rr, "global_kpi", "scheduled_ratio_p5"),
            "pf": safe_get(pf, "global_kpi", "scheduled_ratio_p5"),
        },
    ]

    for row in metrics:
        rr_val = row["rr"]
        pf_val = row["pf"]
        row["note"] = row.get("note") or metric_note(rr, row["name"]) or metric_note(pf, row["name"])
        row["pf_minus_rr"] = None if rr_val is None or pf_val is None else pf_val - rr_val
        row["pf_over_rr_ratio"] = None if rr_val is None or pf_val is None else safe_ratio(pf_val, rr_val)
        if rr_val is None or pf_val is None:
            row["winner"] = "unknown"
        elif abs(pf_val - rr_val) <= 1.0e-12:
            row["winner"] = "tie"
        elif row["direction"] == "higher_better":
            row["winner"] = compare_baseline if pf_val > rr_val else "rr"
        else:
            row["winner"] = compare_baseline if pf_val < rr_val else "rr"
    return metrics


def build_consistency_rows(summary, label):
    compare = summary.get("cpu_gpu_compare", {})
    return {
        "label": label,
        "instantaneous_mean_gpu_over_cpu_ratio": compare.get("instantaneous_mean_gpu_over_cpu_ratio"),
        "long_term_mean_gpu_over_cpu_ratio": compare.get("long_term_sum_mean_gpu_over_cpu_ratio"),
        "long_term_last_gpu_over_cpu_ratio": compare.get("long_term_sum_last_gpu_over_cpu_ratio"),
        "per_ue_avg_mean_gpu_over_cpu_ratio": compare.get("per_ue_avg_mean_gpu_over_cpu_ratio"),
        "instantaneous_mean_gpu_minus_cpu_mbps": compare.get("instantaneous_mean_gpu_minus_cpu_mbps"),
        "long_term_mean_gpu_minus_cpu_mbps": compare.get("long_term_sum_mean_gpu_minus_cpu_mbps"),
    }


def index_by_key(rows, *keys):
    indexed = {}
    for row in rows or []:
        key = tuple(row.get(k) for k in keys)
        indexed[key] = row
    return indexed


def build_per_ue_delta(rr, pf, top_n):
    rr_rows = index_by_key(rr.get("per_ue_kpi", []), "ue_id")
    pf_rows = index_by_key(pf.get("per_ue_kpi", []), "ue_id")
    common_ids = sorted(set(rr_rows) & set(pf_rows))
    deltas = []
    for key in common_ids:
        rr_row = rr_rows[key]
        pf_row = pf_rows[key]
        rr_thr = rr_row.get("avg_thr_mbps")
        pf_thr = pf_row.get("avg_thr_mbps")
        rr_delay = rr_row.get("queue_delay_est_ms")
        pf_delay = pf_row.get("queue_delay_est_ms")
        rr_sched = rr_row.get("scheduled_ratio")
        pf_sched = pf_row.get("scheduled_ratio")
        deltas.append(
            {
                "ue_id": rr_row.get("ue_id"),
                "cell_id_rr": rr_row.get("cell_id"),
                "cell_id_pf": pf_row.get("cell_id"),
                "rr_avg_thr_mbps": rr_thr,
                "pf_avg_thr_mbps": pf_thr,
                "pf_minus_rr_avg_thr_mbps": None if rr_thr is None or pf_thr is None else pf_thr - rr_thr,
                "pf_over_rr_avg_thr_ratio": None if rr_thr is None or pf_thr is None else safe_ratio(pf_thr, rr_thr),
                "rr_queue_delay_est_ms": rr_delay,
                "pf_queue_delay_est_ms": pf_delay,
                "pf_minus_rr_queue_delay_est_ms": None if rr_delay is None or pf_delay is None else pf_delay - rr_delay,
                "rr_scheduled_ratio": rr_sched,
                "pf_scheduled_ratio": pf_sched,
                "pf_minus_rr_scheduled_ratio": None if rr_sched is None or pf_sched is None else pf_sched - rr_sched,
            }
        )

    top_gain = sorted(
        deltas,
        key=lambda x: abs(x["pf_minus_rr_avg_thr_mbps"]) if x["pf_minus_rr_avg_thr_mbps"] is not None else -1.0,
        reverse=True,
    )[: max(top_n, 0)]
    top_delay = sorted(
        deltas,
        key=lambda x: abs(x["pf_minus_rr_queue_delay_est_ms"]) if x["pf_minus_rr_queue_delay_est_ms"] is not None else -1.0,
        reverse=True,
    )[: max(top_n, 0)]
    return {
        "matched_ue_count": len(common_ids),
        "top_abs_throughput_delta": top_gain,
        "top_abs_queue_delay_delta": top_delay,
    }


def build_per_cell_delta(rr, pf):
    rr_rows = index_by_key(rr.get("per_cell_kpi", []), "cell_id")
    pf_rows = index_by_key(pf.get("per_cell_kpi", []), "cell_id")
    common_ids = sorted(set(rr_rows) & set(pf_rows))
    rows = []
    for key in common_ids:
        rr_row = rr_rows[key]
        pf_row = pf_rows[key]
        rr_thr = rr_row.get("cell_sum_thr_mbps")
        pf_thr = pf_row.get("cell_sum_thr_mbps")
        rr_delay = rr_row.get("cell_avg_queue_delay_est_ms")
        pf_delay = pf_row.get("cell_avg_queue_delay_est_ms")
        rows.append(
            {
                "cell_id": rr_row.get("cell_id"),
                "rr_cell_sum_thr_mbps": rr_thr,
                "pf_cell_sum_thr_mbps": pf_thr,
                "pf_minus_rr_cell_sum_thr_mbps": None if rr_thr is None or pf_thr is None else pf_thr - rr_thr,
                "pf_over_rr_cell_sum_thr_ratio": None if rr_thr is None or pf_thr is None else safe_ratio(pf_thr, rr_thr),
                "rr_cell_avg_queue_delay_est_ms": rr_delay,
                "pf_cell_avg_queue_delay_est_ms": pf_delay,
                "pf_minus_rr_cell_avg_queue_delay_est_ms": None if rr_delay is None or pf_delay is None else pf_delay - rr_delay,
            }
        )
    return rows


def build_summary(rr, pf, top_n, compare_baseline):
    display_name = compare_display_name(compare_baseline)
    metric_rows = build_metric_rows(rr, pf, compare_baseline)
    compare_consistency = build_consistency_rows(pf, compare_baseline)
    per_cell_delta = build_per_cell_delta(rr, pf)
    per_ue_delta = build_per_ue_delta(rr, pf, top_n)
    return {
        "compare_baseline": compare_baseline,
        "compare_display_name": display_name,
        "rr_summary_path": rr["_summary_path"],
        "pf_summary_path": pf["_summary_path"],
        "other_summary_path": pf["_summary_path"],
        "rr_run_dir": rr.get("run_dir"),
        "pf_run_dir": pf.get("run_dir"),
        "other_run_dir": pf.get("run_dir"),
        "rr_tti_count": rr.get("tti_count"),
        "pf_tti_count": pf.get("tti_count"),
        "rr_ue_count": rr.get("ue_count"),
        "pf_ue_count": pf.get("ue_count"),
        "metric_source_note": (
            f"RR vs {display_name} should be judged mainly by traffic/global_kpi. "
            "CPU/GPU compare fields are per-run consistency checks, not baseline winners."
        ),
        "metric_definitions": rr.get("metric_definitions") or pf.get("metric_definitions") or {},
        "rr_vs_pf_metrics": metric_rows,
        "rr_vs_compare_metrics": metric_rows,
        "rr_cpu_gpu_consistency": build_consistency_rows(rr, "rr"),
        "pf_cpu_gpu_consistency": compare_consistency,
        "compare_cpu_gpu_consistency": compare_consistency,
        "per_cell_delta": per_cell_delta,
        "per_ue_delta": per_ue_delta,
    }


def write_json(out_dir, summary):
    path = out_dir / f"rr_vs_{summary['compare_baseline']}_compare.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_csv(out_dir, summary):
    compare_baseline = summary["compare_baseline"]
    path = out_dir / f"rr_vs_{compare_baseline}_compare.csv"
    lines = [
        f"metric,rr,{compare_baseline},{compare_baseline}_minus_rr,{compare_baseline}_over_rr_ratio,direction,winner,unit,note"
    ]
    for row in summary["rr_vs_pf_metrics"]:
        lines.append(
            ",".join(
                [
                    row["name"],
                    "" if row["rr"] is None else str(row["rr"]),
                    "" if row["pf"] is None else str(row["pf"]),
                    "" if row["pf_minus_rr"] is None else str(row["pf_minus_rr"]),
                    "" if row["pf_over_rr_ratio"] is None else str(row["pf_over_rr_ratio"]),
                    row["direction"],
                    row["winner"],
                    "" if row["unit"] is None else row["unit"],
                    "" if "note" not in row else row["note"],
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_text(out_dir, summary):
    compare_baseline = summary["compare_baseline"]
    display_name = summary["compare_display_name"]
    path = out_dir / f"rr_vs_{compare_baseline}_compare.txt"
    lines = [
        f"rr_summary_path: {summary['rr_summary_path']}",
        f"{compare_baseline}_summary_path: {summary['other_summary_path']}",
        f"rr_run_dir: {summary['rr_run_dir']}",
        f"{compare_baseline}_run_dir: {summary['other_run_dir']}",
        f"rr_tti_count: {summary['rr_tti_count']}",
        f"{compare_baseline}_tti_count: {summary['pf_tti_count']}",
        f"rr_ue_count: {summary['rr_ue_count']}",
        f"{compare_baseline}_ue_count: {summary['pf_ue_count']}",
        "",
        "[Interpretation]",
        summary["metric_source_note"],
        "",
        f"[RR vs {display_name} Metrics]",
    ]
    for row in summary["rr_vs_pf_metrics"]:
        lines.append(
            f"{row['name']}: rr={format_value(row['rr'], row['unit'])} "
            f"{compare_baseline}={format_value(row['pf'], row['unit'])} "
            f"{compare_baseline}_minus_rr={format_value(row['pf_minus_rr'], row['unit'])} "
            f"{compare_baseline}_over_rr_ratio={format_value(row['pf_over_rr_ratio'])} "
            f"winner={row['winner']}"
            + (f" | {row['note']}" if row.get("note") else "")
        )
    lines.extend(
        [
            "",
            "[RR CPU-GPU Consistency]",
        ]
    )
    for key, value in summary["rr_cpu_gpu_consistency"].items():
        if key == "label":
            continue
        lines.append(f"{key}: {format_value(value)}")
    lines.extend(
        [
            "",
            f"[{display_name} CPU-GPU Consistency]",
        ]
    )
    for key, value in summary["pf_cpu_gpu_consistency"].items():
        if key == "label":
            continue
        lines.append(f"{key}: {format_value(value)}")

    lines.extend(
        [
            "",
            "[Per-Cell Delta]",
        ]
    )
    for row in summary["per_cell_delta"]:
        lines.append(
            f"cell_id={row['cell_id']} "
            f"rr_cell_sum_thr_mbps={format_value(row['rr_cell_sum_thr_mbps'], 'Mbps')} "
            f"{compare_baseline}_cell_sum_thr_mbps={format_value(row['pf_cell_sum_thr_mbps'], 'Mbps')} "
            f"{compare_baseline}_minus_rr_cell_sum_thr_mbps={format_value(row['pf_minus_rr_cell_sum_thr_mbps'], 'Mbps')} "
            f"rr_cell_avg_queue_delay_est_ms={format_value(row['rr_cell_avg_queue_delay_est_ms'], 'ms')} "
            f"{compare_baseline}_cell_avg_queue_delay_est_ms={format_value(row['pf_cell_avg_queue_delay_est_ms'], 'ms')}"
        )

    lines.extend(
        [
            "",
            f"[Top UE Throughput Delta | matched_ue_count={summary['per_ue_delta']['matched_ue_count']}]",
        ]
    )
    for row in summary["per_ue_delta"]["top_abs_throughput_delta"]:
        lines.append(
            f"ue_id={row['ue_id']} "
            f"rr_avg_thr_mbps={format_value(row['rr_avg_thr_mbps'], 'Mbps')} "
            f"{compare_baseline}_avg_thr_mbps={format_value(row['pf_avg_thr_mbps'], 'Mbps')} "
            f"{compare_baseline}_minus_rr_avg_thr_mbps={format_value(row['pf_minus_rr_avg_thr_mbps'], 'Mbps')} "
            f"{compare_baseline}_over_rr_avg_thr_ratio={format_value(row['pf_over_rr_avg_thr_ratio'])}"
        )

    lines.extend(
        [
            "",
            f"[Top UE Queue Delay Delta | matched_ue_count={summary['per_ue_delta']['matched_ue_count']}]",
        ]
    )
    for row in summary["per_ue_delta"]["top_abs_queue_delay_delta"]:
        lines.append(
            f"ue_id={row['ue_id']} "
            f"rr_queue_delay_est_ms={format_value(row['rr_queue_delay_est_ms'], 'ms')} "
            f"{compare_baseline}_queue_delay_est_ms={format_value(row['pf_queue_delay_est_ms'], 'ms')} "
            f"{compare_baseline}_minus_rr_queue_delay_est_ms={format_value(row['pf_minus_rr_queue_delay_est_ms'], 'ms')} "
            f"rr_scheduled_ratio={format_value(row['rr_scheduled_ratio'])} "
            f"{compare_baseline}_scheduled_ratio={format_value(row['pf_scheduled_ratio'])}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    compare_baseline = normalize_compare_baseline(args.compare_baseline)
    rr = load_summary(args.rr)
    pf = load_summary(args.pf)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(rr, pf, args.top_n, compare_baseline)
    json_path = write_json(out_dir, summary)
    txt_path = write_text(out_dir, summary)
    csv_path = write_csv(out_dir, summary)

    display_name = compare_display_name(compare_baseline)
    print(f"RR vs {display_name} comparison written: {json_path}")
    print(f"RR vs {display_name} comparison written: {txt_path}")
    print(f"RR vs {display_name} comparison written: {csv_path}")


if __name__ == "__main__":
    main()
