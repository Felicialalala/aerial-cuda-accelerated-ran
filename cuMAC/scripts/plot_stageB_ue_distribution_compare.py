#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


RR_COLOR = "#D97706"
PF_COLOR = "#2563EB"
GRID_COLOR = "#D1D5DB"
TEXT_COLOR = "#111827"
SUBTEXT_COLOR = "#4B5563"
BG_COLOR = "#FFFFFF"
PANEL_BG = "#F8FAFC"
PANEL_BORDER = "#E5E7EB"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot RR/PF UE throughput and delay distributions as SVG."
    )
    parser.add_argument("--rr-csv", required=True, help="Path to RR ue_kpi.csv")
    parser.add_argument("--pf-csv", required=True, help="Path to PF ue_kpi.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write SVG outputs")
    parser.add_argument("--rr-label", default="RR", help="Legend label for RR series")
    parser.add_argument("--pf-label", default="PF", help="Legend label for PF series")
    parser.add_argument(
        "--title-prefix",
        default="Stage-B UE Distribution Compare",
        help="Prefix used in chart titles",
    )
    return parser.parse_args()


def read_ue_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ue_id = row.get("ue_id") or ""
            if not ue_id.isdigit():
                continue
            rows.append(
                {
                    "ue_id": int(ue_id),
                    "cell_id": int(row["cell_id"]),
                    "avg_thr_mbps": float(row["avg_thr_mbps"]),
                    "scheduled_ratio": float(row["scheduled_ratio"]),
                    "queue_delay_est_ms": float(row["queue_delay_est_ms"]),
                    "packet_delay_mean_ms": float(row["packet_delay_mean_ms"]),
                    "packet_delay_p95_ms": float(row["packet_delay_p95_ms"]),
                    "avg_wb_sinr_db": float(row["avg_wb_sinr_db"]),
                }
            )
    return rows


def percentile(values, p):
    xs = sorted(values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def fmt_value(value, unit):
    if unit == "Mbps":
        return f"{value:.2f} Mbps"
    if unit == "ms":
        return f"{value:.1f} ms"
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "ratio":
        return f"{value:.3f}"
    return f"{value:.3f}"


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def legend(x, y, items):
    parts = []
    cursor_x = x
    for label, color in items:
        parts.append(
            f'<rect x="{cursor_x}" y="{y}" width="14" height="14" rx="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{cursor_x + 22}" y="{y + 12}" font-size="13" fill="{TEXT_COLOR}">{escape(label)}</text>'
        )
        cursor_x += 24 + len(label) * 8 + 26
    return parts


def cdf_points(values):
    xs = sorted(values)
    n = len(xs)
    points = []
    for idx, value in enumerate(xs):
        pct = (idx + 1) * 100.0 / n
        points.append((value, pct))
    return points


def nice_linear_max(value, step):
    return max(step, math.ceil(value / step) * step)


def linear_mapper(x_min, x_max):
    span = max(1e-9, x_max - x_min)

    def _map(value):
        return (value - x_min) / span

    return _map


def log1p_mapper(x_max):
    denom = math.log10(1.0 + max(x_max, 1.0))

    def _map(value):
        return math.log10(1.0 + max(value, 0.0)) / denom

    return _map


def cdf_svg(title, subtitle, series, x_label, x_ticks, x_map, x_max, output_path):
    width = 1080
    height = 700
    margin_left = 92
    margin_right = 40
    margin_top = 86
    margin_bottom = 82
    plot_x = margin_left
    plot_y = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BG_COLOR}"/>',
        f'<rect x="18" y="18" width="{width - 36}" height="{height - 36}" rx="20" fill="{PANEL_BG}" stroke="{PANEL_BORDER}"/>',
        f'<text x="{plot_x}" y="42" font-size="28" font-weight="700" fill="{TEXT_COLOR}">{escape(title)}</text>',
        f'<text x="{plot_x}" y="68" font-size="14" fill="{SUBTEXT_COLOR}">{escape(subtitle)}</text>',
    ]
    parts.extend(legend(width - 230, 34, [(label, color) for label, color, _ in series]))

    for pct in [0, 25, 50, 75, 100]:
        y = plot_y + plot_h - (pct / 100.0) * plot_h
        parts.append(
            f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_w}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{plot_x - 10}" y="{y + 4:.2f}" font-size="12" text-anchor="end" fill="{SUBTEXT_COLOR}">{pct}</text>'
        )

    for tick_value, tick_label in x_ticks:
        x = plot_x + x_map(tick_value) * plot_w
        parts.append(
            f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_h}" stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{plot_y + plot_h + 28}" font-size="12" text-anchor="middle" fill="{SUBTEXT_COLOR}">{escape(tick_label)}</text>'
        )

    parts.append(
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="{TEXT_COLOR}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="{TEXT_COLOR}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="{plot_x + plot_w / 2:.2f}" y="{height - 20}" font-size="14" text-anchor="middle" fill="{SUBTEXT_COLOR}">{escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="24" y="{plot_y + plot_h / 2:.2f}" font-size="14" fill="{SUBTEXT_COLOR}" transform="rotate(-90 24 {plot_y + plot_h / 2:.2f})">UE percentile (%)</text>'
    )

    for label, color, values in series:
        points = cdf_points(values)
        coords = []
        for value, pct in points:
            x = plot_x + x_map(min(value, x_max)) * plot_w
            y = plot_y + plot_h - (pct / 100.0) * plot_h
            coords.append(f"{x:.2f},{y:.2f}")
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(coords)}"/>'
        )

        value_stats = [percentile(values, q) for q in (5, 50, 95)]
        text = f"{label}: p5={value_stats[0]:.2f}, p50={value_stats[1]:.2f}, p95={value_stats[2]:.2f}"
        parts.append(
            f'<text x="{plot_x}" y="{plot_y + plot_h + 56 + (0 if label == series[0][0] else 20)}" font-size="12" fill="{color}">{escape(text)}</text>'
        )

    parts.append("</svg>")
    write_text(output_path, "\n".join(parts))


def scatter_svg(title, subtitle, rr_rows, pf_rows, output_path):
    width = 1080
    height = 700
    margin_left = 82
    margin_right = 40
    margin_top = 86
    margin_bottom = 82
    plot_x = margin_left
    plot_y = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_sinr = [row["avg_wb_sinr_db"] for row in rr_rows + pf_rows]
    all_thr = [row["avg_thr_mbps"] for row in rr_rows + pf_rows]
    x_min = math.floor(min(all_sinr) / 5.0) * 5.0
    x_max = math.ceil(max(all_sinr) / 5.0) * 5.0
    y_min = 0.0
    y_max = nice_linear_max(max(all_thr) + 2.0, 5.0)
    x_map = linear_mapper(x_min, x_max)
    y_map = linear_mapper(y_min, y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BG_COLOR}"/>',
        f'<rect x="18" y="18" width="{width - 36}" height="{height - 36}" rx="20" fill="{PANEL_BG}" stroke="{PANEL_BORDER}"/>',
        f'<text x="{plot_x}" y="42" font-size="28" font-weight="700" fill="{TEXT_COLOR}">{escape(title)}</text>',
        f'<text x="{plot_x}" y="68" font-size="14" fill="{SUBTEXT_COLOR}">{escape(subtitle)}</text>',
    ]
    parts.extend(legend(width - 230, 34, [("RR", RR_COLOR), ("PF", PF_COLOR)]))

    x_ticks = list(range(int(x_min), int(x_max) + 1, 5))
    y_ticks = list(range(0, int(y_max) + 1, 5))

    for tick in x_ticks:
        x = plot_x + x_map(tick) * plot_w
        parts.append(
            f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_h}" stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{plot_y + plot_h + 28}" font-size="12" text-anchor="middle" fill="{SUBTEXT_COLOR}">{tick}</text>'
        )

    for tick in y_ticks:
        y = plot_y + plot_h - y_map(tick) * plot_h
        parts.append(
            f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_w}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{plot_x - 10}" y="{y + 4:.2f}" font-size="12" text-anchor="end" fill="{SUBTEXT_COLOR}">{tick}</text>'
        )

    parts.append(
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="{TEXT_COLOR}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="{TEXT_COLOR}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="{plot_x + plot_w / 2:.2f}" y="{height - 20}" font-size="14" text-anchor="middle" fill="{SUBTEXT_COLOR}">Average UE SINR (dB)</text>'
    )
    parts.append(
        f'<text x="24" y="{plot_y + plot_h / 2:.2f}" font-size="14" fill="{SUBTEXT_COLOR}" transform="rotate(-90 24 {plot_y + plot_h / 2:.2f})">Average UE throughput (Mbps)</text>'
    )

    for label, color, rows in [("RR", RR_COLOR, rr_rows), ("PF", PF_COLOR, pf_rows)]:
        for row in rows:
            x = plot_x + x_map(row["avg_wb_sinr_db"]) * plot_w
            y = plot_y + plot_h - y_map(row["avg_thr_mbps"]) * plot_h
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" fill-opacity="0.68"/>'
            )

        stats = (
            f'{label}: mean thr={sum(r["avg_thr_mbps"] for r in rows) / len(rows):.2f} Mbps, '
            f'p5 thr={percentile([r["avg_thr_mbps"] for r in rows], 5):.2f} Mbps'
        )
        y_text = plot_y + plot_h + 56 + (0 if label == "RR" else 20)
        parts.append(
            f'<text x="{plot_x}" y="{y_text}" font-size="12" fill="{color}">{escape(stats)}</text>'
        )

    parts.append("</svg>")
    write_text(output_path, "\n".join(parts))


def index_html(output_dir, files):
    cards = []
    for filename, title in files:
        cards.append(
            f"""
            <section class="card">
              <h2>{escape(title)}</h2>
              <img src="{escape(filename)}" alt="{escape(title)}"/>
            </section>
            """
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Stage-B UE Distribution Compare</title>
  <style>
    body {{
      margin: 24px;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background: #f3f4f6;
      color: #111827;
    }}
    h1 {{
      margin: 0 0 18px;
      font-size: 28px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(17, 24, 39, 0.06);
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 12px;
      background: #f8fafc;
    }}
  </style>
</head>
<body>
  <h1>Stage-B UE Distribution Compare</h1>
  <div class="grid">
    {"".join(cards)}
  </div>
</body>
</html>
"""


def write_summary_csv(output_dir, rr_rows, pf_rows):
    metrics = [
        ("avg_thr_mbps", "Mbps"),
        ("queue_delay_est_ms", "ms"),
        ("packet_delay_mean_ms", "ms"),
        ("packet_delay_p95_ms", "ms"),
        ("scheduled_ratio", "ratio"),
    ]
    lines = ["metric,rr_mean,rr_p5,rr_p50,rr_p95,pf_mean,pf_p5,pf_p50,pf_p95,unit"]
    for metric, unit in metrics:
        rr_values = [row[metric] for row in rr_rows]
        pf_values = [row[metric] for row in pf_rows]
        lines.append(
            ",".join(
                [
                    metric,
                    f"{sum(rr_values) / len(rr_values):.10f}",
                    f"{percentile(rr_values, 5):.10f}",
                    f"{percentile(rr_values, 50):.10f}",
                    f"{percentile(rr_values, 95):.10f}",
                    f"{sum(pf_values) / len(pf_values):.10f}",
                    f"{percentile(pf_values, 5):.10f}",
                    f"{percentile(pf_values, 50):.10f}",
                    f"{percentile(pf_values, 95):.10f}",
                    unit,
                ]
            )
        )
    write_text(output_dir / "ue_distribution_summary.csv", "\n".join(lines) + "\n")


def main():
    args = parse_args()
    rr_rows = read_ue_csv(args.rr_csv)
    pf_rows = read_ue_csv(args.pf_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    throughput_max = nice_linear_max(
        max(max(row["avg_thr_mbps"] for row in rr_rows), max(row["avg_thr_mbps"] for row in pf_rows)) + 2.0,
        5.0,
    )
    throughput_ticks = [(tick, str(tick)) for tick in range(0, int(throughput_max) + 1, 5)]
    throughput_map = linear_mapper(0.0, throughput_max)
    cdf_svg(
        title=f"{args.title_prefix}: UE Throughput CDF",
        subtitle="Per-UE average served throughput distribution across all 56 UEs.",
        series=[
            (args.rr_label, RR_COLOR, [row["avg_thr_mbps"] for row in rr_rows]),
            (args.pf_label, PF_COLOR, [row["avg_thr_mbps"] for row in pf_rows]),
        ],
        x_label="Average UE throughput (Mbps)",
        x_ticks=throughput_ticks,
        x_map=throughput_map,
        x_max=throughput_max,
        output_path=output_dir / "ue_throughput_cdf.svg",
    )

    queue_max = max(max(row["queue_delay_est_ms"] for row in rr_rows), max(row["queue_delay_est_ms"] for row in pf_rows))
    queue_ticks = [(0.0, "0"), (1.0, "1"), (10.0, "10"), (100.0, "100"), (1000.0, "1k"), (5000.0, "5k"), (8000.0, "8k")]
    cdf_svg(
        title=f"{args.title_prefix}: UE Queue Delay CDF",
        subtitle="Backlog-based queue_delay_est_ms on a log-like x-axis to preserve the tail.",
        series=[
            (args.rr_label, RR_COLOR, [row["queue_delay_est_ms"] for row in rr_rows]),
            (args.pf_label, PF_COLOR, [row["queue_delay_est_ms"] for row in pf_rows]),
        ],
        x_label="Per-UE queue delay estimate (ms, log-like scale)",
        x_ticks=queue_ticks,
        x_map=log1p_mapper(max(queue_max, 8000.0)),
        x_max=max(queue_max, 8000.0),
        output_path=output_dir / "ue_queue_delay_cdf.svg",
    )

    packet_max = max(max(row["packet_delay_mean_ms"] for row in rr_rows), max(row["packet_delay_mean_ms"] for row in pf_rows))
    packet_ticks = [(0.0, "0"), (1.0, "1"), (10.0, "10"), (50.0, "50"), (100.0, "100"), (200.0, "200"), (500.0, "500")]
    cdf_svg(
        title=f"{args.title_prefix}: UE Packet Delay CDF",
        subtitle="Per-UE packet_delay_mean_ms over served packets, shown on a log-like x-axis.",
        series=[
            (args.rr_label, RR_COLOR, [row["packet_delay_mean_ms"] for row in rr_rows]),
            (args.pf_label, PF_COLOR, [row["packet_delay_mean_ms"] for row in pf_rows]),
        ],
        x_label="Per-UE packet mean delay (ms, log-like scale)",
        x_ticks=packet_ticks,
        x_map=log1p_mapper(max(packet_max, 500.0)),
        x_max=max(packet_max, 500.0),
        output_path=output_dir / "ue_packet_delay_cdf.svg",
    )

    scatter_svg(
        title=f"{args.title_prefix}: UE Throughput vs SINR",
        subtitle="Same UE cloud under RR and PF. This highlights whether gains are concentrated on high-SINR users.",
        rr_rows=rr_rows,
        pf_rows=pf_rows,
        output_path=output_dir / "ue_throughput_vs_sinr.svg",
    )

    write_summary_csv(output_dir, rr_rows, pf_rows)

    html_files = [
        ("ue_throughput_cdf.svg", "UE Throughput CDF"),
        ("ue_queue_delay_cdf.svg", "UE Queue Delay CDF"),
        ("ue_packet_delay_cdf.svg", "UE Packet Delay CDF"),
        ("ue_throughput_vs_sinr.svg", "UE Throughput vs SINR"),
    ]
    write_text(output_dir / "index.html", index_html(output_dir, html_files))

    print(f"Wrote plots to: {output_dir}")
    for filename, _title in html_files:
        print(output_dir / filename)
    print(output_dir / "index.html")
    print(output_dir / "ue_distribution_summary.csv")


if __name__ == "__main__":
    main()
