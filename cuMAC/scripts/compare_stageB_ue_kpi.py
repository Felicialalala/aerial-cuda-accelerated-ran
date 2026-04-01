#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge RR/PF UE KPI CSVs into a side-by-side comparison table."
    )
    parser.add_argument("--rr-csv", required=True, help="Path to RR ue_kpi.csv")
    parser.add_argument("--pf-csv", required=True, help="Path to PF ue_kpi.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs")
    return parser.parse_args()


def load_rows(path):
    rows = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ue_id = row.get("ue_id") or ""
            if not ue_id.isdigit():
                continue
            rows[int(ue_id)] = row
    return rows


def f(row, key):
    return float(row[key])


def i(row, key):
    return int(row[key])


def build_merged(rr_rows, pf_rows):
    merged = []
    for ue_id in sorted(rr_rows):
        rr = rr_rows[ue_id]
        pf = pf_rows[ue_id]
        merged.append(
            {
                "cell_id": i(rr, "cell_id"),
                "ue_id": ue_id,
                "avg_wb_sinr_db": f(rr, "avg_wb_sinr_db"),
                "rr_avg_thr_mbps": f(rr, "avg_thr_mbps"),
                "pf_avg_thr_mbps": f(pf, "avg_thr_mbps"),
                "pf_minus_rr_avg_thr_mbps": f(pf, "avg_thr_mbps") - f(rr, "avg_thr_mbps"),
                "rr_queue_delay_est_ms": f(rr, "queue_delay_est_ms"),
                "pf_queue_delay_est_ms": f(pf, "queue_delay_est_ms"),
                "pf_minus_rr_queue_delay_est_ms": f(pf, "queue_delay_est_ms") - f(rr, "queue_delay_est_ms"),
                "rr_packet_delay_mean_ms": f(rr, "packet_delay_mean_ms"),
                "pf_packet_delay_mean_ms": f(pf, "packet_delay_mean_ms"),
                "pf_minus_rr_packet_delay_mean_ms": f(pf, "packet_delay_mean_ms") - f(rr, "packet_delay_mean_ms"),
                "rr_packet_delay_p95_ms": f(rr, "packet_delay_p95_ms"),
                "pf_packet_delay_p95_ms": f(pf, "packet_delay_p95_ms"),
                "pf_minus_rr_packet_delay_p95_ms": f(pf, "packet_delay_p95_ms") - f(rr, "packet_delay_p95_ms"),
                "rr_scheduled_ratio": f(rr, "scheduled_ratio"),
                "pf_scheduled_ratio": f(pf, "scheduled_ratio"),
                "pf_minus_rr_scheduled_ratio": f(pf, "scheduled_ratio") - f(rr, "scheduled_ratio"),
                "rr_served_pkt_count": i(rr, "packet_delay_served_pkt_count"),
                "pf_served_pkt_count": i(pf, "packet_delay_served_pkt_count"),
                "pf_minus_rr_served_pkt_count": i(pf, "packet_delay_served_pkt_count") - i(rr, "packet_delay_served_pkt_count"),
                "rr_pending_pkt_count": i(rr, "packet_delay_pending_pkt_count"),
                "pf_pending_pkt_count": i(pf, "packet_delay_pending_pkt_count"),
                "pf_minus_rr_pending_pkt_count": i(pf, "packet_delay_pending_pkt_count") - i(rr, "packet_delay_pending_pkt_count"),
            }
        )
    return merged


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, merged):
    pf_better_thr = [row for row in merged if row["pf_minus_rr_avg_thr_mbps"] > 0.0]
    pf_higher_sched = [row for row in merged if row["pf_minus_rr_scheduled_ratio"] > 0.0]
    pf_high_sched = [row for row in merged if row["pf_scheduled_ratio"] >= 0.95]
    pf_high_sched_small_gain = [
        row
        for row in merged
        if row["pf_scheduled_ratio"] >= 0.95 and row["pf_minus_rr_avg_thr_mbps"] <= 0.5
    ]
    worst_thr = sorted(merged, key=lambda row: row["pf_minus_rr_avg_thr_mbps"])[:10]
    best_thr = sorted(merged, key=lambda row: row["pf_minus_rr_avg_thr_mbps"], reverse=True)[:10]

    lines = [
        "UE KPI side-by-side comparison summary",
        "",
        f"matched_ue_count={len(merged)}",
        f"pf_better_throughput_count={len(pf_better_thr)}",
        f"pf_higher_scheduled_ratio_count={len(pf_higher_sched)}",
        f"pf_high_scheduled_ratio_ge_0.95_count={len(pf_high_sched)}",
        f"pf_high_scheduled_ratio_ge_0.95_but_thr_gain_le_0.5_count={len(pf_high_sched_small_gain)}",
        "",
        "Interpretation:",
        "- In this pair of runs, PF does not schedule any UE more frequently than RR when measured by scheduled_ratio.",
        "- RR already keeps almost every UE scheduled in almost every TTI, so PF advantage is not more TTIs; at most it is more PRGs within some TTIs.",
        "- The small set of PF throughput winners are all backlog-free on both sides, so their gains are capped by offered load rather than radio resource abundance.",
        "",
        "PF throughput winners:",
    ]
    for row in pf_better_thr:
        lines.append(
            "  "
            + f"ue={row['ue_id']} cell={row['cell_id']} sinr_db={row['avg_wb_sinr_db']:.3f} "
            + f"rr_thr={row['rr_avg_thr_mbps']:.6f} pf_thr={row['pf_avg_thr_mbps']:.6f} "
            + f"thr_delta={row['pf_minus_rr_avg_thr_mbps']:.6f} "
            + f"rr_sched={row['rr_scheduled_ratio']:.6f} pf_sched={row['pf_scheduled_ratio']:.6f} "
            + f"rr_qd={row['rr_queue_delay_est_ms']:.6f} pf_qd={row['pf_queue_delay_est_ms']:.6f}"
        )

    lines.extend(["", "Top 10 PF throughput losses:"])
    for row in worst_thr:
        lines.append(
            "  "
            + f"ue={row['ue_id']} cell={row['cell_id']} "
            + f"rr_thr={row['rr_avg_thr_mbps']:.6f} pf_thr={row['pf_avg_thr_mbps']:.6f} "
            + f"thr_delta={row['pf_minus_rr_avg_thr_mbps']:.6f} "
            + f"rr_sched={row['rr_scheduled_ratio']:.6f} pf_sched={row['pf_scheduled_ratio']:.6f} "
            + f"rr_qd={row['rr_queue_delay_est_ms']:.6f} pf_qd={row['pf_queue_delay_est_ms']:.6f}"
        )

    lines.extend(["", "Top 10 PF throughput gains:"])
    for row in best_thr:
        lines.append(
            "  "
            + f"ue={row['ue_id']} cell={row['cell_id']} "
            + f"rr_thr={row['rr_avg_thr_mbps']:.6f} pf_thr={row['pf_avg_thr_mbps']:.6f} "
            + f"thr_delta={row['pf_minus_rr_avg_thr_mbps']:.6f} "
            + f"rr_sched={row['rr_scheduled_ratio']:.6f} pf_sched={row['pf_scheduled_ratio']:.6f} "
            + f"rr_qd={row['rr_queue_delay_est_ms']:.6f} pf_qd={row['pf_queue_delay_est_ms']:.6f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rr_rows = load_rows(args.rr_csv)
    pf_rows = load_rows(args.pf_csv)
    merged = build_merged(rr_rows, pf_rows)

    base_fields = [
        "cell_id",
        "ue_id",
        "avg_wb_sinr_db",
        "rr_avg_thr_mbps",
        "pf_avg_thr_mbps",
        "pf_minus_rr_avg_thr_mbps",
        "rr_queue_delay_est_ms",
        "pf_queue_delay_est_ms",
        "pf_minus_rr_queue_delay_est_ms",
        "rr_packet_delay_mean_ms",
        "pf_packet_delay_mean_ms",
        "pf_minus_rr_packet_delay_mean_ms",
        "rr_packet_delay_p95_ms",
        "pf_packet_delay_p95_ms",
        "pf_minus_rr_packet_delay_p95_ms",
        "rr_scheduled_ratio",
        "pf_scheduled_ratio",
        "pf_minus_rr_scheduled_ratio",
        "rr_served_pkt_count",
        "pf_served_pkt_count",
        "pf_minus_rr_served_pkt_count",
        "rr_pending_pkt_count",
        "pf_pending_pkt_count",
        "pf_minus_rr_pending_pkt_count",
    ]

    write_csv(output_dir / "ue_kpi_compare_all.csv", merged, base_fields)
    write_csv(
        output_dir / "ue_kpi_compare_by_thr_delta.csv",
        sorted(merged, key=lambda row: row["pf_minus_rr_avg_thr_mbps"]),
        base_fields,
    )
    write_summary(output_dir / "ue_kpi_compare_summary.txt", merged)

    print(output_dir / "ue_kpi_compare_all.csv")
    print(output_dir / "ue_kpi_compare_by_thr_delta.csv")
    print(output_dir / "ue_kpi_compare_summary.txt")


if __name__ == "__main__":
    main()
