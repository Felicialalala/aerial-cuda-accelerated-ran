#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import struct
import sys
from collections import Counter
from math import prod


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dtype_nbytes(dtype):
    table = {
        "uint8": 1,
        "int8": 1,
        "int16": 2,
        "uint16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "float": 4,
        "int64": 8,
        "uint64": 8,
        "float64": 8,
    }
    if dtype not in table:
        raise ValueError(f"unsupported dtype in schema: {dtype}")
    return table[dtype]


def layout_offset_and_size(layout, name):
    offset = 0
    for entry in layout:
        shape = entry.get("shape", [])
        count = int(prod(shape)) if shape else 1
        size = dtype_nbytes(entry["dtype"]) * count
        if entry.get("name") == name:
            return offset, size
        offset += size
        # Replay binary keeps 4-byte alignment after done(uint8).
        if entry.get("name") == "done" and entry.get("dtype") == "uint8" and count == 1:
            offset += 3
    raise KeyError(f"field not found in schema layout: {name}")


def analyze_actions(meta, schema, records_path):
    dims = meta.get("dims", {})
    n_cell = int(dims.get("n_cell", 0))
    n_active_ue = int(dims.get("n_active_ue", 0))
    n_sched_ue = int(dims.get("n_sched_ue", 0))
    n_prg = int(dims.get("n_prg", 0))
    action_alloc_len = int(dims.get("action_alloc_len", 0))

    if n_cell <= 0 or n_sched_ue <= 0 or n_active_ue <= 0 or n_prg <= 0 or action_alloc_len <= 0:
        return {"status": "skip", "reason": "invalid dims"}

    layout = schema.get("record_layout", [])
    if not layout:
        return {"status": "skip", "reason": "missing record_layout"}

    try:
        ue_off, ue_size = layout_offset_and_size(layout, "action_ue_select")
        prg_off, prg_size = layout_offset_and_size(layout, "action_prg_alloc")
    except (KeyError, ValueError) as e:
        return {"status": "skip", "reason": str(e)}

    if ue_size != 4 * n_sched_ue:
        return {
            "status": "skip",
            "reason": f"action_ue_select bytes mismatch: schema={ue_size}, expected={4 * n_sched_ue}",
        }
    if prg_size != 2 * action_alloc_len:
        return {
            "status": "skip",
            "reason": f"action_prg_alloc bytes mismatch: schema={prg_size}, expected={2 * action_alloc_len}",
        }

    record_bytes = int(meta.get("record_bytes", 0))
    record_count = int(meta.get("record_count", 0))
    if record_bytes <= 0 or record_count <= 0:
        return {"status": "skip", "reason": "invalid record_bytes/record_count"}

    slots_per_cell = (n_sched_ue // n_cell) if (n_sched_ue % n_cell == 0) else None

    ue_no_total = 0
    ue_total = 0
    prg_no_total = 0
    prg_total = 0

    ue_slot_no = [0] * n_sched_ue
    ue_slot_top = [Counter() for _ in range(n_sched_ue)]

    prg_local_slot = [Counter() for _ in range(n_cell)]

    ue_fmt = "<" + ("i" * n_sched_ue)
    prg_fmt = "<" + ("h" * action_alloc_len)

    with open(records_path, "rb") as f:
        for rec_idx in range(record_count):
            blob = f.read(record_bytes)
            if len(blob) != record_bytes:
                return {"status": "skip", "reason": f"short read at record {rec_idx}"}

            ue = struct.unpack(ue_fmt, blob[ue_off : ue_off + ue_size])
            prg = struct.unpack(prg_fmt, blob[prg_off : prg_off + prg_size])

            for s, x in enumerate(ue):
                ue_total += 1
                if x < 0:
                    ue_no_total += 1
                    ue_slot_no[s] += 1
                elif x < n_active_ue:
                    ue_slot_top[s][x] += 1

            if slots_per_cell is not None:
                for p in range(n_prg):
                    for c in range(n_cell):
                        idx = p * n_cell + c
                        if idx >= len(prg):
                            continue
                        x = prg[idx]
                        prg_total += 1
                        if x < 0:
                            prg_no_total += 1
                            continue
                        local = x - c * slots_per_cell
                        if 0 <= local < slots_per_cell:
                            prg_local_slot[c][local] += 1
            else:
                for x in prg:
                    prg_total += 1
                    if x < 0:
                        prg_no_total += 1

    ue_no_ratio = (ue_no_total / ue_total) if ue_total > 0 else 0.0
    prg_no_ratio = (prg_no_total / prg_total) if prg_total > 0 else 0.0

    ue_slot_no_ratio = [x / float(record_count) for x in ue_slot_no]
    ue_slot_top1 = []
    for s in range(n_sched_ue):
        total = sum(ue_slot_top[s].values())
        top1 = (ue_slot_top[s].most_common(1)[0][1] / total) if total > 0 else 1.0
        ue_slot_top1.append(top1)

    prg_dominant_slot_share = []
    for c in range(n_cell):
        total = sum(prg_local_slot[c].values())
        top_share = (prg_local_slot[c].most_common(1)[0][1] / total) if total > 0 else 1.0
        prg_dominant_slot_share.append(top_share)

    warnings = []
    if max(prg_dominant_slot_share, default=0.0) > 0.90:
        warnings.append("PRG labels highly concentrated on one slot in at least one cell (>90%).")
    if sum(1 for x in ue_slot_top1 if x > 0.95) >= max(1, n_sched_ue // 2):
        warnings.append("UE labels are highly deterministic for many slots (top1>95%).")
    if ue_no_ratio > 0.30:
        warnings.append("UE NO-slot ratio is high (>30%), effective scheduling supervision may be weak.")

    return {
        "status": "ok",
        "n_cell": n_cell,
        "n_sched_ue": n_sched_ue,
        "slots_per_cell": slots_per_cell,
        "ue_no_ratio": ue_no_ratio,
        "prg_no_ratio": prg_no_ratio,
        "ue_slot_no_ratio": ue_slot_no_ratio,
        "ue_slot_top1": ue_slot_top1,
        "prg_dominant_slot_share": prg_dominant_slot_share,
        "warnings": warnings,
    }


def print_action_analysis(result):
    if result.get("status") != "ok":
        print(f"Action diversity: skip ({result.get('reason', 'unknown')})")
        return

    n_cell = result["n_cell"]
    slots_per_cell = result["slots_per_cell"]
    print("Action diversity")
    print(f"  ue_no_ratio: {result['ue_no_ratio']:.6f}")
    print(f"  prg_no_ratio: {result['prg_no_ratio']:.6f}")

    if slots_per_cell is not None and slots_per_cell > 0:
        print("  ue_slot_no_ratio_by_cell:")
        for c in range(n_cell):
            beg = c * slots_per_cell
            end = beg + slots_per_cell
            vals = result["ue_slot_no_ratio"][beg:end]
            text = ",".join(f"{v:.3f}" for v in vals)
            print(f"    cell{c}: {text}")

        print("  ue_slot_top1_share_by_cell:")
        for c in range(n_cell):
            beg = c * slots_per_cell
            end = beg + slots_per_cell
            vals = result["ue_slot_top1"][beg:end]
            text = ",".join(f"{v:.3f}" for v in vals)
            print(f"    cell{c}: {text}")

    print("  prg_dominant_slot_share_by_cell:")
    for c, v in enumerate(result["prg_dominant_slot_share"]):
        print(f"    cell{c}: {v:.3f}")

    if result["warnings"]:
        print("  warnings:")
        for w in result["warnings"]:
            print(f"    - {w}")


def main():
    parser = argparse.ArgumentParser(description="Inspect cuMAC RL replay dump metadata/schema.")
    parser.add_argument("--replay-dir", required=True, help="Replay directory (contains meta/schema/bin)")
    args = parser.parse_args()

    replay_dir = os.path.abspath(args.replay_dir)
    meta_path = os.path.join(replay_dir, "rl_replay_meta.json")
    schema_path = os.path.join(replay_dir, "rl_replay_schema.json")

    if not os.path.isfile(meta_path):
        print(f"[inspect] missing meta file: {meta_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(schema_path):
        print(f"[inspect] missing schema file: {schema_path}", file=sys.stderr)
        return 1

    meta = load_json(meta_path)
    schema = load_json(schema_path)
    records_file = meta.get("records_file", "rl_replay_records.bin")
    records_path = os.path.join(replay_dir, records_file)

    if not os.path.isfile(records_path):
        print(f"[inspect] missing records file: {records_path}", file=sys.stderr)
        return 1

    file_size = os.path.getsize(records_path)
    record_bytes = int(meta.get("record_bytes", 0))
    record_count = int(meta.get("record_count", 0))
    expected_size = record_bytes * record_count
    inferred_count = (file_size // record_bytes) if record_bytes > 0 else 0

    print("Replay summary")
    print(f"  dir: {replay_dir}")
    print(f"  records: {records_path}")
    print(f"  file_size_bytes: {file_size}")
    print(f"  record_bytes: {record_bytes}")
    print(f"  record_count(meta): {record_count}")
    print(f"  record_count(inferred): {inferred_count}")
    print(f"  expected_size_bytes: {expected_size}")
    print(f"  size_match: {expected_size == file_size}")
    print(f"  dims: {json.dumps(meta.get('dims', {}), ensure_ascii=True)}")
    print(f"  feature_dims: {json.dumps(meta.get('feature_dims', {}), ensure_ascii=True)}")
    print(f"  meta: {json.dumps(meta.get('meta', {}), ensure_ascii=True)}")
    print(f"  first_layout_entry: {schema.get('record_layout', [{}])[0]}")

    analysis = analyze_actions(meta=meta, schema=schema, records_path=records_path)
    print_action_analysis(analysis)

    if expected_size != file_size:
        print("[inspect] ERROR: record size mismatch", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
