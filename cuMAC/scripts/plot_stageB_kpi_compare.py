#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


RR_COLOR = "#D97706"
PF_COLOR = "#2563EB"
GRID_COLOR = "#D1D5DB"
TEXT_COLOR = "#111827"
SUBTEXT_COLOR = "#4B5563"
BG_COLOR = "#FFFFFF"


def parse_args():
    p = argparse.ArgumentParser(description="Plot Stage-B RR vs PF KPI comparison as SVG.")
    p.add_argument("--rr-json", required=True, help="Path to RR kpi_summary.json")
    p.add_argument("--pf-json", required=True, help="Path to PF kpi_summary.json")
    p.add_argument("--output-dir", required=True, help="Directory to write SVG outputs")
    return p.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_num(v, decimals=1):
    return f"{v:.{decimals}f}"


def fmt_metric(v, unit):
    if unit == "%":
        return f"{v:.1f}%"
    if unit == "ms":
        return f"{v:.1f} ms"
    if unit == "Mbps":
        return f"{v:.2f} Mbps"
    return f"{v:.2f}"


def bar_chart_svg(rr, pf, rr_label, pf_label):
    throughput_metrics = [
        ("Served Throughput", rr["traffic"]["served_mbps_est"], pf["traffic"]["served_mbps_est"]),
    ]
    delay_metrics = [
        ("Queue Delay", rr["traffic"]["queue_delay_est_ms"], pf["traffic"]["queue_delay_est_ms"]),
    ]
    ratio_metrics = [
        ("Served Buffer", rr["global_kpi"]["served_buffer_ratio"] * 100.0, pf["global_kpi"]["served_buffer_ratio"] * 100.0),
        ("Residual Buffer", rr["global_kpi"]["residual_buffer_ratio"] * 100.0, pf["global_kpi"]["residual_buffer_ratio"] * 100.0),
        ("Backlog-free UE", rr["global_kpi"]["backlog_free_ue_ratio"] * 100.0, pf["global_kpi"]["backlog_free_ue_ratio"] * 100.0),
    ]

    width = 1360
    height = 920
    gap = 26
    top_y = 108
    card_h = 282
    card_w = (width - 80 * 2 - gap) / 2
    ratio_y = top_y + card_h + 28

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BG_COLOR}"/>',
        f'<text x="80" y="42" font-size="28" font-weight="700" fill="{TEXT_COLOR}">Stage-B KPI Comparison: RR vs PF</text>',
        f'<text x="80" y="72" font-size="15" fill="{SUBTEXT_COLOR}">Redesigned as unit-consistent small multiples: throughput, delay, and ratio metrics each use their own axis.</text>',
        legend(width - 250, 36, rr_label, pf_label),
    ]

    parts.extend(render_grouped_panel(
        x=80,
        y=top_y,
        w=card_w,
        h=card_h,
        title="Throughput",
        subtitle="Unit: Mbps",
        metrics=throughput_metrics,
        unit="Mbps",
        y_max=None,
    ))
    parts.extend(render_grouped_panel(
        x=80 + card_w + gap,
        y=top_y,
        w=card_w,
        h=card_h,
        title="Queue Pressure",
        subtitle="Unit: ms",
        metrics=delay_metrics,
        unit="ms",
        y_max=None,
    ))
    parts.extend(render_grouped_panel(
        x=80,
        y=ratio_y,
        w=width - 160,
        h=card_h + 80,
        title="Buffer and QoS Proxies",
        subtitle="Unit: %  |  grouped bars with a shared 0-100 scale",
        metrics=ratio_metrics,
        unit="%",
        y_max=100.0,
    ))

    parts.append("</svg>")
    return "\n".join(parts)


def render_grouped_panel(x, y, w, h, title, subtitle, metrics, unit, y_max=None):
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#F8FAFC" stroke="#E5E7EB"/>',
        f'<text x="{x + 24}" y="{y + 34}" font-size="20" font-weight="700" fill="{TEXT_COLOR}">{escape(title)}</text>',
        f'<text x="{x + 24}" y="{y + 58}" font-size="13" fill="{SUBTEXT_COLOR}">{escape(subtitle)}</text>',
    ]

    plot_x = x + 68
    plot_y = y + 82
    plot_w = w - 96
    plot_h = h - 122

    if y_max is None:
        max_val = max(max(m[1], m[2]) for m in metrics)
        y_max = max_val * 1.18 if max_val > 0 else 1.0

    tick_count = 4
    tick_vals = [y_max * i / tick_count for i in range(tick_count + 1)]

    for tick in tick_vals:
        yy = plot_y + plot_h - (tick / y_max) * plot_h
        parts.append(f'<line x1="{plot_x}" y1="{yy:.2f}" x2="{plot_x + plot_w}" y2="{yy:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text x="{plot_x - 10}" y="{yy + 4:.2f}" font-size="11" text-anchor="end" fill="{SUBTEXT_COLOR}">{fmt_num(tick, 0)}</text>')

    parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="{TEXT_COLOR}" stroke-width="1.1"/>')
    parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="{TEXT_COLOR}" stroke-width="1.1"/>')
    parts.append(f'<text x="{x + 18}" y="{plot_y + plot_h / 2:.2f}" font-size="12" fill="{SUBTEXT_COLOR}" transform="rotate(-90 {x + 18} {plot_y + plot_h / 2:.2f})">{escape(unit)}</text>')

    group_w = plot_w / max(len(metrics), 1)
    bar_w = min(58, group_w * 0.24 if len(metrics) > 1 else 72)

    for idx, (label, rr_v, pf_v) in enumerate(metrics):
        cx = plot_x + group_w * (idx + 0.5)
        rr_x = cx - bar_w - 8
        pf_x = cx + 8
        rr_h = (rr_v / y_max) * plot_h
        pf_h = (pf_v / y_max) * plot_h
        rr_y = plot_y + plot_h - rr_h
        pf_y = plot_y + plot_h - pf_h

        parts.append(f'<rect x="{rr_x:.2f}" y="{rr_y:.2f}" width="{bar_w:.2f}" height="{rr_h:.2f}" rx="6" fill="{RR_COLOR}"/>')
        parts.append(f'<rect x="{pf_x:.2f}" y="{pf_y:.2f}" width="{bar_w:.2f}" height="{pf_h:.2f}" rx="6" fill="{PF_COLOR}"/>')
        parts.append(value_label(rr_x + bar_w / 2, rr_y - 10, fmt_metric(rr_v, unit), RR_COLOR))
        parts.append(value_label(pf_x + bar_w / 2, pf_y - 10, fmt_metric(pf_v, unit), PF_COLOR))
        parts.append(f'<text x="{cx:.2f}" y="{plot_y + plot_h + 28}" font-size="13" font-weight="600" text-anchor="middle" fill="{TEXT_COLOR}">{escape(label)}</text>')

    return parts


def dumbbell_svg(rr, pf, rr_label, pf_label):
    metrics = [
        ("P50 Delay", rr["global_kpi"]["queue_delay_p50_ms"], pf["global_kpi"]["queue_delay_p50_ms"]),
        ("P90 Delay", rr["global_kpi"]["queue_delay_p90_ms"], pf["global_kpi"]["queue_delay_p90_ms"]),
        ("P95 Delay", rr["global_kpi"]["queue_delay_p95_ms"], pf["global_kpi"]["queue_delay_p95_ms"]),
    ]

    width = 1180
    height = 450
    margin_left = 180
    margin_right = 60
    margin_top = 95
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_val = max(max(m[1], m[2]) for m in metrics)
    x_max = max_val * 1.12 if max_val > 0 else 1.0
    tick_count = 5
    tick_vals = [x_max * i / tick_count for i in range(tick_count + 1)]
    row_gap = plot_h / max(len(metrics), 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BG_COLOR}"/>',
        f'<text x="{margin_left}" y="42" font-size="28" font-weight="700" fill="{TEXT_COLOR}">Delay Percentile Dumbbell</text>',
        f'<text x="{margin_left}" y="72" font-size="15" fill="{SUBTEXT_COLOR}">Horizontal comparison of RR and PF delay percentiles. This makes it easy to see that P50 differs much more than P95.</text>',
        legend(width - 150, 36, rr_label, pf_label, vertical=True),
    ]

    for tick in tick_vals:
        x = margin_left + (tick / x_max) * plot_w
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_h}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{margin_top + plot_h + 26}" font-size="12" text-anchor="middle" fill="{SUBTEXT_COLOR}">{fmt_num(tick, 0)}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="{TEXT_COLOR}" stroke-width="1.2"/>')
    parts.append(f'<text x="{margin_left + plot_w / 2:.2f}" y="{height - 16}" font-size="14" text-anchor="middle" fill="{SUBTEXT_COLOR}">Delay (ms)</text>')

    for idx, (label, rr_v, pf_v) in enumerate(metrics):
        y = margin_top + row_gap * (idx + 0.5)
        rr_x = margin_left + (rr_v / x_max) * plot_w
        pf_x = margin_left + (pf_v / x_max) * plot_w
        x1, x2 = sorted([rr_x, pf_x])
        parts.append(f'<text x="{margin_left - 18}" y="{y + 5:.2f}" font-size="15" font-weight="600" text-anchor="end" fill="{TEXT_COLOR}">{escape(label)}</text>')
        parts.append(f'<line x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}" stroke="#9CA3AF" stroke-width="3"/>')
        parts.append(f'<circle cx="{rr_x:.2f}" cy="{y:.2f}" r="8" fill="{RR_COLOR}"/>')
        parts.append(f'<circle cx="{pf_x:.2f}" cy="{y:.2f}" r="8" fill="{PF_COLOR}"/>')
        parts.append(value_label(rr_x, y - 14, fmt_metric(rr_v, "ms"), RR_COLOR))
        parts.append(value_label(pf_x, y + 28, fmt_metric(pf_v, "ms"), PF_COLOR))

    parts.append("</svg>")
    return "\n".join(parts)


def legend(x, y, rr_label, pf_label, vertical=False):
    if vertical:
        gap = 26
        return (
            f'<rect x="{x}" y="{y}" width="14" height="14" rx="3" fill="{RR_COLOR}"/>'
            f'<text x="{x + 22}" y="{y + 12}" font-size="13" fill="{TEXT_COLOR}">{escape(rr_label)}</text>'
            f'<rect x="{x}" y="{y + gap}" width="14" height="14" rx="3" fill="{PF_COLOR}"/>'
            f'<text x="{x + 22}" y="{y + gap + 12}" font-size="13" fill="{TEXT_COLOR}">{escape(pf_label)}</text>'
        )
    return (
        f'<rect x="{x}" y="{y}" width="14" height="14" rx="3" fill="{RR_COLOR}"/>'
        f'<text x="{x + 22}" y="{y + 12}" font-size="13" fill="{TEXT_COLOR}">{escape(rr_label)}</text>'
        f'<rect x="{x + 90}" y="{y}" width="14" height="14" rx="3" fill="{PF_COLOR}"/>'
        f'<text x="{x + 112}" y="{y + 12}" font-size="13" fill="{TEXT_COLOR}">{escape(pf_label)}</text>'
    )


def value_label(x, y, text, color):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="12" font-weight="700" text-anchor="middle" fill="{color}">{escape(text)}</text>'


def write_metric_table(out_dir, rr, pf):
    rows = [
        ("served_mbps_est", rr["traffic"]["served_mbps_est"], pf["traffic"]["served_mbps_est"], "Mbps"),
        ("served_buffer_ratio_pct", rr["global_kpi"]["served_buffer_ratio"] * 100.0, pf["global_kpi"]["served_buffer_ratio"] * 100.0, "%"),
        ("residual_buffer_ratio_pct", rr["global_kpi"]["residual_buffer_ratio"] * 100.0, pf["global_kpi"]["residual_buffer_ratio"] * 100.0, "%"),
        ("queue_delay_est_ms", rr["traffic"]["queue_delay_est_ms"], pf["traffic"]["queue_delay_est_ms"], "ms"),
        ("backlog_free_ue_ratio_pct", rr["global_kpi"]["backlog_free_ue_ratio"] * 100.0, pf["global_kpi"]["backlog_free_ue_ratio"] * 100.0, "%"),
        ("queue_delay_p50_ms", rr["global_kpi"]["queue_delay_p50_ms"], pf["global_kpi"]["queue_delay_p50_ms"], "ms"),
        ("queue_delay_p90_ms", rr["global_kpi"]["queue_delay_p90_ms"], pf["global_kpi"]["queue_delay_p90_ms"], "ms"),
        ("queue_delay_p95_ms", rr["global_kpi"]["queue_delay_p95_ms"], pf["global_kpi"]["queue_delay_p95_ms"], "ms"),
    ]
    lines = ["metric,rr,pf,unit"]
    for metric, rr_v, pf_v, unit in rows:
        lines.append(f"{metric},{rr_v:.10f},{pf_v:.10f},{unit}")
    (out_dir / "kpi_compare.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rr = read_json(args.rr_json)
    pf = read_json(args.pf_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rr_label = "RR"
    pf_label = "PF"

    grouped_svg = bar_chart_svg(rr, pf, rr_label, pf_label)
    dumbbell = dumbbell_svg(rr, pf, rr_label, pf_label)

    (out_dir / "grouped_bar.svg").write_text(grouped_svg, encoding="utf-8")
    (out_dir / "delay_dumbbell.svg").write_text(dumbbell, encoding="utf-8")
    write_metric_table(out_dir, rr, pf)

    print(f"Wrote: {out_dir / 'grouped_bar.svg'}")
    print(f"Wrote: {out_dir / 'delay_dumbbell.svg'}")
    print(f"Wrote: {out_dir / 'kpi_compare.csv'}")


if __name__ == "__main__":
    main()
