#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean


def parse_args():
    p = argparse.ArgumentParser(description="Summarize Stage-A MVP KPI from output files.")
    p.add_argument("--output-dir", required=True, help="Run output directory")
    p.add_argument("--slot-duration-ms", type=float, default=0.5, help="Slot duration in ms")
    p.add_argument("--traffic-percent", type=float, default=0.0, help="Traffic UE percentage used in run")
    p.add_argument("--traffic-rate", type=float, default=5000.0, help="Traffic packet size in bytes")
    return p.parse_args()


def parse_arrays(text: str):
    pattern = re.compile(r"([A-Za-z0-9_]+)\s*=\s*\[(.*?)\];", re.S)
    parsed = {}
    for key, payload in pattern.findall(text):
        values = []
        for tok in payload.split():
            try:
                values.append(float(tok))
            except ValueError:
                continue
        parsed[key] = values
    return parsed


def pct(values, p):
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    arr = sorted(values)
    pos = (len(arr) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    w = pos - lo
    return arr[lo] * (1.0 - w) + arr[hi] * w


def basic_stats(values):
    if not values:
        return {"count": 0, "mean": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "last": 0.0}
    return {
        "count": len(values),
        "mean": mean(values),
        "p5": pct(values, 5),
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "min": min(values),
        "max": max(values),
        "last": values[-1],
    }


def parse_traffic_kpi(log_text: str):
    l1 = re.findall(
        r"TRAFFIC_KPI flows=(\d+)\s+generated_pkts=(\d+)\s+generated_bytes=(\d+)\s+accepted_bytes=(\d+)\s+dropped_bytes=(\d+)\s+flow_queued_bytes=(\d+)\s+mac_buffer_bytes=(\d+)\s+served_bytes_est=(\d+)",
        log_text,
    )
    l2 = re.findall(
        r"TRAFFIC_KPI offered_mbps=([0-9eE+\-\.]+)\s+served_mbps_est=([0-9eE+\-\.]+)\s+drop_rate=([0-9eE+\-\.]+)\s+queue_delay_est_ms=([0-9eE+\-\.]+)",
        log_text,
    )
    if not l1 or not l2:
        return None
    a = l1[-1]
    b = l2[-1]
    return {
        "flows": int(a[0]),
        "generated_pkts": int(a[1]),
        "generated_bytes": int(a[2]),
        "accepted_bytes": int(a[3]),
        "dropped_bytes": int(a[4]),
        "flow_queued_bytes": int(a[5]),
        "mac_buffer_bytes": int(a[6]),
        "served_bytes_est": int(a[7]),
        "offered_mbps": float(b[0]),
        "served_mbps_est": float(b[1]),
        "drop_rate": float(b[2]),
        "queue_delay_est_ms": float(b[3]),
    }


def parse_ue_kpi(log_text: str):
    pat_new = re.compile(
        r"UE_KPI\s+cell_id=(-?\d+)\s+ue_local_id=(-?\d+)\s+ue_id=(\d+)\s+avg_thr_mbps=([0-9eE+\-\.]+)\s+avg_mcs_tx_only=([0-9eE+\-\.]+)\s+avg_mcs_all_tti0=([0-9eE+\-\.]+)\s+scheduled_tti_count=(\d+)\s+no_tx_tti_count=(\d+)\s+scheduled_ratio=([0-9eE+\-\.]+)\s+avg_wb_sinr_db=([0-9eE+\-\.]+)\s+avg_sched_wb_sinr_db=([0-9eE+\-\.]+)\s+avg_predicted_bler=([0-9eE+\-\.]+)\s+tb_err_count=(\d+)\s+tb_bler=([0-9eE+\-\.]+)\s+flow_drop_rate=([0-9eE+\-\.]+)\s+tx_success_rate=([0-9eE+\-\.]+)\s+tx_drop_rate=([0-9eE+\-\.]+)\s+tx_total_pkts=(\d+)\s+tx_success_pkts=(\d+)\s+queue_delay_est_ms=([0-9eE+\-\.]+)\s+generated_bytes=(\d+)\s+accepted_bytes=(\d+)\s+dropped_bytes=(\d+)\s+flow_queued_bytes=(\d+)\s+mac_buffer_bytes=(\d+)\s+mcs_samples=(\d+)"
    )
    pat_legacy = re.compile(
        r"UE_KPI\s+ue_id=(\d+)\s+avg_thr_mbps=([0-9eE+\-\.]+)\s+avg_mcs=([0-9eE+\-\.]+)\s+drop_rate=([0-9eE+\-\.]+)\s+queue_delay_est_ms=([0-9eE+\-\.]+)\s+generated_bytes=(\d+)\s+accepted_bytes=(\d+)\s+dropped_bytes=(\d+)\s+flow_queued_bytes=(\d+)\s+mac_buffer_bytes=(\d+)\s+mcs_samples=(\d+)"
    )
    rows = []
    for m in pat_new.finditer(log_text):
        rows.append(
            {
                "cell_id": int(m.group(1)),
                "ue_local_id": int(m.group(2)),
                "ue_id": int(m.group(3)),
                "avg_thr_mbps": float(m.group(4)),
                "avg_mcs_tx_only": float(m.group(5)),
                "avg_mcs_all_tti0": float(m.group(6)),
                "scheduled_tti_count": int(m.group(7)),
                "no_tx_tti_count": int(m.group(8)),
                "scheduled_ratio": float(m.group(9)),
                "avg_wb_sinr_db": float(m.group(10)),
                "avg_sched_wb_sinr_db": float(m.group(11)),
                "avg_predicted_bler": float(m.group(12)),
                "tb_err_count": int(m.group(13)),
                "tb_bler": float(m.group(14)),
                "flow_drop_rate": float(m.group(15)),
                "drop_rate": float(m.group(15)),
                "tx_success_rate": float(m.group(16)),
                "tx_drop_rate": float(m.group(17)),
                "tx_total_pkts": int(m.group(18)),
                "tx_success_pkts": int(m.group(19)),
                "queue_delay_est_ms": float(m.group(20)),
                "generated_bytes": int(m.group(21)),
                "accepted_bytes": int(m.group(22)),
                "dropped_bytes": int(m.group(23)),
                "flow_queued_bytes": int(m.group(24)),
                "mac_buffer_bytes": int(m.group(25)),
                "mcs_samples": int(m.group(26)),
                "avg_mcs": float(m.group(5)),
            }
        )
    if not rows:
        for m in pat_legacy.finditer(log_text):
            ue_id = int(m.group(1))
            rows.append(
                {
                    "cell_id": -1,
                    "ue_local_id": ue_id,
                    "ue_id": ue_id,
                    "avg_thr_mbps": float(m.group(2)),
                    "avg_mcs_tx_only": float(m.group(3)),
                    "avg_mcs_all_tti0": float(m.group(3)),
                    "avg_mcs": float(m.group(3)),
                    "scheduled_tti_count": int(m.group(11)),
                    "no_tx_tti_count": 0,
                    "scheduled_ratio": 1.0,
                    "avg_wb_sinr_db": 0.0,
                    "avg_sched_wb_sinr_db": 0.0,
                    "avg_predicted_bler": 0.0,
                    "tb_err_count": 0,
                    "tb_bler": 0.0,
                    "flow_drop_rate": float(m.group(4)),
                    "drop_rate": float(m.group(4)),
                    "tx_success_rate": 0.0,
                    "tx_drop_rate": 0.0,
                    "tx_total_pkts": 0,
                    "tx_success_pkts": 0,
                    "queue_delay_est_ms": float(m.group(5)),
                    "generated_bytes": int(m.group(6)),
                    "accepted_bytes": int(m.group(7)),
                    "dropped_bytes": int(m.group(8)),
                    "flow_queued_bytes": int(m.group(9)),
                    "mac_buffer_bytes": int(m.group(10)),
                    "mcs_samples": int(m.group(11)),
                }
            )
    rows.sort(key=lambda x: (x["cell_id"], x["ue_local_id"], x["ue_id"]))
    return rows


def summarize_cell_kpi(ue_kpi):
    groups = {}
    for row in ue_kpi:
        cell_id = row.get("cell_id", -1)
        groups.setdefault(cell_id, []).append(row)

    rows = []
    for cell_id in sorted(groups):
        cell_rows = groups[cell_id]
        ue_count = len(cell_rows)
        thr_values = [r["avg_thr_mbps"] for r in cell_rows]
        delay_values = [r["queue_delay_est_ms"] for r in cell_rows]
        total_mcs_samples = sum(r.get("mcs_samples", 0) for r in cell_rows)
        total_tx_pkts = sum(r.get("tx_total_pkts", 0) for r in cell_rows)
        total_tx_success = sum(r.get("tx_success_pkts", 0) for r in cell_rows)
        total_tb_err = sum(r.get("tb_err_count", 0) for r in cell_rows)
        total_generated = sum(r.get("generated_bytes", 0) for r in cell_rows)
        total_mac_buffer = sum(r.get("mac_buffer_bytes", 0) for r in cell_rows)
        total_flow_drop_num = sum(r.get("dropped_bytes", 0) for r in cell_rows)
        total_sched_tti = sum(r.get("scheduled_tti_count", r.get("mcs_samples", 0)) for r in cell_rows)
        total_no_tx_tti = sum(r.get("no_tx_tti_count", 0) for r in cell_rows)
        avg_mcs_num = sum(r.get("avg_mcs_tx_only", r.get("avg_mcs", 0.0)) * r.get("mcs_samples", 0) for r in cell_rows)
        avg_mcs_all_tti0_num = sum(r.get("avg_mcs_all_tti0", r.get("avg_mcs", 0.0)) * (r.get("scheduled_tti_count", r.get("mcs_samples", 0)) + r.get("no_tx_tti_count", 0)) for r in cell_rows)
        avg_sinr_num = sum(r.get("avg_wb_sinr_db", 0.0) * max(r.get("scheduled_tti_count", 0) + r.get("no_tx_tti_count", 0), 1) for r in cell_rows)
        avg_sched_sinr_num = sum(r.get("avg_sched_wb_sinr_db", 0.0) * max(r.get("scheduled_tti_count", r.get("mcs_samples", 0)), 1) for r in cell_rows)
        avg_pred_bler_num = sum(r.get("avg_predicted_bler", 0.0) * max(r.get("scheduled_tti_count", r.get("mcs_samples", 0)), 1) for r in cell_rows)
        total_tti_samples = sum(r.get("scheduled_tti_count", r.get("mcs_samples", 0)) + r.get("no_tx_tti_count", 0) for r in cell_rows)
        cell_tx_success_rate = float(total_tx_success) / float(total_tx_pkts) if total_tx_pkts > 0 else 0.0
        cell_flow_drop_rate = float(total_flow_drop_num) / float(total_generated) if total_generated > 0 else 0.0

        rows.append(
            {
                "cell_id": cell_id,
                "ue_count": ue_count,
                "cell_sum_thr_mbps": sum(thr_values),
                "cell_avg_ue_thr_mbps": mean(thr_values) if thr_values else 0.0,
                "cell_avg_mcs_tx_only": avg_mcs_num / total_mcs_samples if total_mcs_samples > 0 else 0.0,
                "cell_avg_mcs_all_tti0": avg_mcs_all_tti0_num / total_tti_samples if total_tti_samples > 0 else 0.0,
                "cell_scheduled_tti_count": total_sched_tti,
                "cell_no_tx_tti_count": total_no_tx_tti,
                "cell_scheduled_ratio": float(total_sched_tti) / float(total_sched_tti + total_no_tx_tti) if (total_sched_tti + total_no_tx_tti) > 0 else 0.0,
                "cell_avg_wb_sinr_db": avg_sinr_num / total_tti_samples if total_tti_samples > 0 else 0.0,
                "cell_avg_sched_wb_sinr_db": avg_sched_sinr_num / total_sched_tti if total_sched_tti > 0 else 0.0,
                "cell_avg_predicted_bler": avg_pred_bler_num / total_sched_tti if total_sched_tti > 0 else 0.0,
                "cell_tb_err_count": total_tb_err,
                "cell_tb_bler": float(total_tb_err) / float(total_tx_pkts) if total_tx_pkts > 0 else 0.0,
                "cell_tx_success_rate": cell_tx_success_rate,
                "cell_tx_drop_rate": 1.0 - cell_tx_success_rate if total_tx_pkts > 0 else 0.0,
                "cell_tx_total_pkts": total_tx_pkts,
                "cell_tx_success_pkts": total_tx_success,
                "cell_flow_drop_rate": cell_flow_drop_rate,
                "cell_avg_queue_delay_est_ms": mean(delay_values) if delay_values else 0.0,
                "cell_total_generated_bytes": total_generated,
                "cell_total_mac_buffer_bytes": total_mac_buffer,
                "cell_total_mcs_samples": total_mcs_samples,
            }
        )
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    output_short = out_dir / "output_short.txt"
    output_full = out_dir / "output.txt"
    run_log = out_dir / "run.log"
    summary_json = out_dir / "kpi_summary.json"

    source_file = output_short if output_short.exists() else output_full
    if source_file.exists():
        arrays = parse_arrays(source_file.read_text())
        log_text = run_log.read_text() if run_log.exists() else ""

        sum_ins = arrays.get("mcType1SumInsThrRecordsGpu", arrays.get("mcType0SumInsThrRecordsGpu", []))
        sum_cell = arrays.get("mcType1SumCellThrRecordsGpu", arrays.get("mcType0SumCellThrRecordsGpu", []))
        ue_avg = arrays.get("mcType1AvgRatesGpu", arrays.get("mcType0AvgRatesGpu", arrays.get("mcType1AvgRatesGpu", [])))

        throughput_ins_mbps = [x / 1e6 for x in sum_ins]
        throughput_cell_mbps = [x / 1e6 for x in sum_cell]
        ue_avg_mbps = [x / 1e6 for x in ue_avg]

        traffic_kpi = parse_traffic_kpi(log_text)
        ue_kpi = parse_ue_kpi(log_text)
    elif summary_json.exists():
        existing = json.loads(summary_json.read_text())
        sum_ins = []
        sum_cell = []
        ue_avg = []
        throughput_ins_mbps = []
        throughput_cell_mbps = []
        ue_avg_mbps = []
        traffic_kpi = existing.get("traffic")
        ue_kpi = existing.get("per_ue_kpi", [])
        for row in ue_kpi:
            row.setdefault("cell_id", -1)
            row.setdefault("ue_local_id", row.get("ue_id", -1))
            row.setdefault("flow_drop_rate", row.get("drop_rate", 0.0))
            row.setdefault("drop_rate", row.get("flow_drop_rate", 0.0))
            row.setdefault("avg_mcs_tx_only", row.get("avg_mcs", 0.0))
            row.setdefault("avg_mcs_all_tti0", row.get("avg_mcs", 0.0))
            row.setdefault("avg_mcs", row.get("avg_mcs_tx_only", 0.0))
            row.setdefault("scheduled_tti_count", row.get("mcs_samples", 0))
            row.setdefault("no_tx_tti_count", 0)
            row.setdefault("scheduled_ratio", 1.0 if row.get("mcs_samples", 0) > 0 else 0.0)
            row.setdefault("avg_wb_sinr_db", 0.0)
            row.setdefault("avg_sched_wb_sinr_db", 0.0)
            row.setdefault("avg_predicted_bler", 0.0)
            row.setdefault("tb_err_count", 0)
            row.setdefault("tb_bler", row.get("tx_drop_rate", 0.0))
            row.setdefault("tx_success_rate", 0.0)
            row.setdefault("tx_drop_rate", 0.0)
            row.setdefault("tx_total_pkts", 0)
            row.setdefault("tx_success_pkts", 0)
            row.setdefault("generated_bytes", 0)
            row.setdefault("accepted_bytes", row.get("generated_bytes", 0))
            row.setdefault("dropped_bytes", 0)
            row.setdefault("flow_queued_bytes", 0)
            row.setdefault("mac_buffer_bytes", 0)
            row.setdefault("mcs_samples", 0)
        ue_kpi.sort(key=lambda x: (x["cell_id"], x["ue_local_id"], x["ue_id"]))
    else:
        raise FileNotFoundError(f"Cannot find output or summary file in {out_dir}")

    if traffic_kpi is None:
        flow_count = int(round(len(ue_avg) * args.traffic_percent / 100.0))
        offered_bytes_per_tti_est = flow_count * args.traffic_rate
        slot_s = args.slot_duration_ms / 1000.0
        offered_mbps_est = (offered_bytes_per_tti_est * 8.0 / slot_s) / 1e6 if slot_s > 0 else 0.0
        traffic_kpi = {
            "flows": flow_count,
            "generated_pkts": None,
            "generated_bytes": None,
            "accepted_bytes": None,
            "dropped_bytes": None,
            "flow_queued_bytes": None,
            "mac_buffer_bytes": None,
            "served_bytes_est": None,
            "offered_mbps": offered_mbps_est,
            "served_mbps_est": basic_stats(throughput_ins_mbps)["mean"],
            "drop_rate": None,
            "queue_delay_est_ms": None,
            "note": "Traffic counters were not found in run.log; offered load is estimated from CLI args.",
        }

    summary = {
        "run_dir": str(out_dir),
        "slot_duration_ms": args.slot_duration_ms,
        "tti_count": len(sum_ins),
        "ue_count": len(ue_avg),
        "throughput": {
            "instantaneous_mbps_gpu": basic_stats(throughput_ins_mbps),
            "long_term_sum_mbps_gpu": basic_stats(throughput_cell_mbps),
            "per_ue_avg_mbps_gpu": basic_stats(ue_avg_mbps),
        },
        "traffic": traffic_kpi,
        "per_ue_kpi": ue_kpi,
        "per_cell_kpi": summarize_cell_kpi(ue_kpi),
    }

    summary_txt = out_dir / "kpi_summary.txt"
    ue_kpi_csv = out_dir / "ue_kpi.csv"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = [
        f"run_dir: {summary['run_dir']}",
        f"tti_count: {summary['tti_count']}",
        f"ue_count: {summary['ue_count']}",
        "",
        "[Throughput GPU]",
        f"inst_mean_mbps: {summary['throughput']['instantaneous_mbps_gpu']['mean']:.6f}",
        f"inst_p50_mbps: {summary['throughput']['instantaneous_mbps_gpu']['p50']:.6f}",
        f"inst_p95_mbps: {summary['throughput']['instantaneous_mbps_gpu']['p95']:.6f}",
        f"inst_last_mbps: {summary['throughput']['instantaneous_mbps_gpu']['last']:.6f}",
        f"long_term_last_mbps: {summary['throughput']['long_term_sum_mbps_gpu']['last']:.6f}",
        f"per_ue_p5_mbps: {pct(ue_avg_mbps, 5):.6f}",
        f"per_ue_mean_mbps: {summary['throughput']['per_ue_avg_mbps_gpu']['mean']:.6f}",
        "",
        "[Traffic]",
        f"flows: {summary['traffic']['flows']}",
        f"offered_mbps: {summary['traffic']['offered_mbps'] if summary['traffic']['offered_mbps'] is not None else 'N/A'}",
        f"served_mbps_est: {summary['traffic']['served_mbps_est'] if summary['traffic']['served_mbps_est'] is not None else 'N/A'}",
        f"drop_rate: {summary['traffic']['drop_rate'] if summary['traffic']['drop_rate'] is not None else 'N/A'}",
        f"queue_delay_est_ms: {summary['traffic']['queue_delay_est_ms'] if summary['traffic']['queue_delay_est_ms'] is not None else 'N/A'}",
    ]
    if "note" in summary["traffic"]:
        lines.append(f"note: {summary['traffic']['note']}")

    summary_txt.write_text("\n".join(lines) + "\n")
    if ue_kpi:
        csv_lines = [
            "cell_id,ue_local_id,ue_id,avg_thr_mbps,avg_mcs_tx_only,avg_mcs_all_tti0,scheduled_tti_count,no_tx_tti_count,scheduled_ratio,avg_wb_sinr_db,avg_sched_wb_sinr_db,avg_predicted_bler,tb_err_count,tb_bler,tx_success_rate,tx_drop_rate,tx_total_pkts,tx_success_pkts,queue_delay_est_ms,generated_bytes,mac_buffer_bytes,mcs_samples"
        ]
        for r in ue_kpi:
            csv_lines.append(
                f"{r['cell_id']},{r['ue_local_id']},{r['ue_id']},{r['avg_thr_mbps']:.6f},{r['avg_mcs_tx_only']:.6f},{r['avg_mcs_all_tti0']:.6f},{r['scheduled_tti_count']},{r['no_tx_tti_count']},{r['scheduled_ratio']:.6f},{r['avg_wb_sinr_db']:.6f},{r['avg_sched_wb_sinr_db']:.6f},{r['avg_predicted_bler']:.6f},{r['tb_err_count']},{r['tb_bler']:.6f},{r['tx_success_rate']:.6f},{r['tx_drop_rate']:.6f},{r['tx_total_pkts']},{r['tx_success_pkts']},{r['queue_delay_est_ms']:.6f},{r['generated_bytes']},{r['mac_buffer_bytes']},{r['mcs_samples']}"
            )
        cell_kpi = summary["per_cell_kpi"]
        if cell_kpi:
            csv_lines.append("")
            csv_lines.append("# CELL_KPI")
            csv_lines.append(
                "cell_id,ue_count,cell_sum_thr_mbps,cell_avg_ue_thr_mbps,cell_avg_mcs_tx_only,cell_avg_mcs_all_tti0,cell_scheduled_tti_count,cell_no_tx_tti_count,cell_scheduled_ratio,cell_avg_wb_sinr_db,cell_avg_sched_wb_sinr_db,cell_avg_predicted_bler,cell_tb_err_count,cell_tb_bler,cell_tx_success_rate,cell_tx_drop_rate,cell_tx_total_pkts,cell_tx_success_pkts,cell_avg_queue_delay_est_ms,cell_total_generated_bytes,cell_total_mac_buffer_bytes,cell_total_mcs_samples"
            )
            for r in cell_kpi:
                csv_lines.append(
                    f"{r['cell_id']},{r['ue_count']},{r['cell_sum_thr_mbps']:.6f},{r['cell_avg_ue_thr_mbps']:.6f},{r['cell_avg_mcs_tx_only']:.6f},{r['cell_avg_mcs_all_tti0']:.6f},{r['cell_scheduled_tti_count']},{r['cell_no_tx_tti_count']},{r['cell_scheduled_ratio']:.6f},{r['cell_avg_wb_sinr_db']:.6f},{r['cell_avg_sched_wb_sinr_db']:.6f},{r['cell_avg_predicted_bler']:.6f},{r['cell_tb_err_count']},{r['cell_tb_bler']:.6f},{r['cell_tx_success_rate']:.6f},{r['cell_tx_drop_rate']:.6f},{r['cell_tx_total_pkts']},{r['cell_tx_success_pkts']},{r['cell_avg_queue_delay_est_ms']:.6f},{r['cell_total_generated_bytes']},{r['cell_total_mac_buffer_bytes']},{r['cell_total_mcs_samples']}"
                )
        ue_kpi_csv.write_text("\n".join(csv_lines) + "\n")
    print(f"KPI summary written: {summary_json}")
    print(f"KPI summary written: {summary_txt}")
    if ue_kpi:
        print(f"UE KPI written: {ue_kpi_csv}")


if __name__ == "__main__":
    main()
