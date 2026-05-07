#!/usr/bin/env python3
"""Check fixed-size HGraph ONNX consistency against the PyTorch export wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.stageb_hgraph.export_onnx import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUT,
    DEFAULT_ACTION_OUT,
    FixedHGraphActionOnnxWrapper,
    FixedHGraphOnnxWrapper,
    check_onnx,
    load_actor,
    make_dummy_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=None, help="ONNX model path")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="HGraph actor checkpoint")
    parser.add_argument("--output-mode", choices=["logits", "action"], default="logits")
    parser.add_argument("--use-post-eq-input", action="store_true")
    parser.add_argument("--n-cell", type=int, default=3)
    parser.add_argument("--n-ue", type=int, default=36)
    parser.add_argument("--n-prg", type=int, default=17)
    parser.add_argument("--n-sched-ue", type=int, default=36)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_mode == "action" and not args.use_post_eq_input:
        args.use_post_eq_input = True
    onnx_path = (
        args.onnx.expanduser().resolve()
        if args.onnx is not None
        else (DEFAULT_ACTION_OUT if args.output_mode == "action" else DEFAULT_OUT).resolve()
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()
    device = torch.device(args.device)

    actor, checkpoint = load_actor(checkpoint_path, device)
    wrapper_cls = FixedHGraphActionOnnxWrapper if args.output_mode == "action" else FixedHGraphOnnxWrapper
    wrapper = wrapper_cls(
        actor,
        n_cell=args.n_cell,
        n_ue=args.n_ue,
        n_prg=args.n_prg,
        n_sched_ue=args.n_sched_ue,
    ).to(device=device)
    wrapper.eval()
    actor_config = checkpoint["actor_config"]
    dummy_inputs = make_dummy_inputs(
        n_cell=args.n_cell,
        n_ue=args.n_ue,
        n_prg=args.n_prg,
        cell_feat_dim=int(actor_config["cell_feat_dim"]),
        ue_feat_dim=int(actor_config["ue_feat_dim"]),
        prg_feat_dim=int(actor_config["prg_feat_dim"]),
        device=device,
        include_post_eq=bool(args.use_post_eq_input),
    )
    result = check_onnx(onnx_path, wrapper, dummy_inputs, output_mode=str(args.output_mode))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
