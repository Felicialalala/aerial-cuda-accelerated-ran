#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate Stage-B run KPI summaries into a matrix.")
    p.add_argument("--base-dir", required=True, help="Stage-B output base directory")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"base-dir does not exist: {base_dir}")

    status_map = {}
    status_file = base_dir / "matrix_status.txt"
    if status_file.exists():
        with status_file.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario = (row.get("scenario") or "").strip()
                if not scenario:
                    continue
                status_map[scenario] = {
                    "status": (row.get("status") or "").strip(),
                    "reason": (row.get("reason") or "").strip(),
                }

    rows = []
    for summary in sorted(base_dir.glob("*/kpi_summary.json")):
        scenario = summary.parent.name
        data = json.loads(summary.read_text())
        thr = data.get("throughput", {})
        trf = data.get("traffic", {})
        gkpi = data.get("global_kpi", {})
        ue_count = data.get("ue_count", 0)
        per_ue_mean = thr.get("per_ue_avg_mbps_gpu", {}).get("mean", 0.0)
        status_info = status_map.get(scenario, {})
        row = {
            "scenario": scenario,
            "status": status_info.get("status", "PASS"),
            "reason": status_info.get("reason", "ok"),
            "tti_count": data.get("tti_count", 0),
            "ue_count": ue_count,
            "inst_mean_mbps": thr.get("instantaneous_mbps_gpu", {}).get("mean", 0.0),
            "inst_p95_mbps": thr.get("instantaneous_mbps_gpu", {}).get("p95", 0.0),
            "inst_last_mbps": thr.get("instantaneous_mbps_gpu", {}).get("last", 0.0),
            "long_term_last_mbps": thr.get("long_term_sum_mbps_gpu", {}).get("last", 0.0),
            "cluster_sum_thr_mbps": gkpi.get("cluster_sum_throughput_mbps", per_ue_mean * ue_count),
            "avg_ue_thr_mbps": gkpi.get("average_ue_throughput_mbps", per_ue_mean),
            "global_tb_bler": gkpi.get("global_tb_bler", None),
            "residual_buffer_ratio": gkpi.get("residual_buffer_ratio", None),
            "ue_thr_jain": gkpi.get("ue_throughput_jain", None),
            "cell_sum_thr_jain": gkpi.get("cell_sum_throughput_jain", None),
            "ue_thr_p5_mbps": gkpi.get("ue_throughput_p5_mbps", thr.get("per_ue_avg_mbps_gpu", {}).get("p5", 0.0)),
            "ue_thr_p10_mbps": gkpi.get("ue_throughput_p10_mbps", 0.0),
            "sched_ratio_jain": gkpi.get("scheduled_ratio_jain", None),
            "sched_ratio_p5": gkpi.get("scheduled_ratio_p5", None),
            "queue_delay_p95_ms": gkpi.get("queue_delay_p95_ms", None),
            "ue_delay_gt_10s_frac": gkpi.get("ue_fraction_queue_delay_gt_10s", None),
            "ue_delay_gt_20s_frac": gkpi.get("ue_fraction_queue_delay_gt_20s", None),
            "per_ue_p5_mbps": thr.get("per_ue_avg_mbps_gpu", {}).get("p5", 0.0),
            "per_ue_mean_mbps": per_ue_mean,
            "flows": trf.get("flows", 0),
            "offered_mbps": trf.get("offered_mbps", 0.0),
            "served_mbps_est": trf.get("served_mbps_est", 0.0),
            "drop_rate": trf.get("drop_rate", None),
            "queue_delay_est_ms": trf.get("queue_delay_est_ms", None),
        }
        rows.append(row)

    for scenario, info in status_map.items():
        if any(r["scenario"] == scenario for r in rows):
            continue
        rows.append(
            {
                "scenario": scenario,
                "status": info.get("status", ""),
                "reason": info.get("reason", ""),
                "tti_count": 0,
                "ue_count": 0,
                "inst_mean_mbps": 0.0,
                "inst_p95_mbps": 0.0,
                "inst_last_mbps": 0.0,
                "long_term_last_mbps": 0.0,
                "cluster_sum_thr_mbps": 0.0,
                "avg_ue_thr_mbps": 0.0,
                "global_tb_bler": None,
                "residual_buffer_ratio": None,
                "ue_thr_jain": None,
                "cell_sum_thr_jain": None,
                "ue_thr_p5_mbps": 0.0,
                "ue_thr_p10_mbps": 0.0,
                "sched_ratio_jain": None,
                "sched_ratio_p5": None,
                "queue_delay_p95_ms": None,
                "ue_delay_gt_10s_frac": None,
                "ue_delay_gt_20s_frac": None,
                "per_ue_p5_mbps": 0.0,
                "per_ue_mean_mbps": 0.0,
                "flows": 0,
                "offered_mbps": 0.0,
                "served_mbps_est": 0.0,
                "drop_rate": None,
                "queue_delay_est_ms": None,
            }
        )

    rows.sort(key=lambda r: r["scenario"])

    csv_file = base_dir / "stageB_kpi_matrix.csv"
    txt_file = base_dir / "stageB_kpi_matrix.txt"

    if not rows:
        txt_file.write_text("No kpi_summary.json files found under base directory.\n")
        print(f"No scenario summaries found in {base_dir}")
        print(f"Matrix note written: {txt_file}")
        return

    fields = list(rows[0].keys())
    with csv_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append(f"base_dir: {base_dir}")
    lines.append(f"num_scenarios: {len(rows)}")
    lines.append("")
    for r in rows:
        lines.append(
            f"{r['scenario']} [{r['status']}/{r['reason']}]: cluster_sum_thr={r['cluster_sum_thr_mbps']:.3f} Mbps, "
            f"ue_jain={r['ue_thr_jain']}, ue_p10={r['ue_thr_p10_mbps']}, "
            f"queue_p95_ms={r['queue_delay_p95_ms']}, residual_buffer_ratio={r['residual_buffer_ratio']}"
        )
    txt_file.write_text("\n".join(lines) + "\n")

    print(f"Stage-B matrix written: {csv_file}")
    print(f"Stage-B matrix written: {txt_file}")


if __name__ == "__main__":
    main()
