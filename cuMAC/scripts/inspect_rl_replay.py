#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    print(f"  first_layout_entry: {schema.get('record_layout', [{}])[0]}")

    if expected_size != file_size:
        print("[inspect] ERROR: record size mismatch", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

