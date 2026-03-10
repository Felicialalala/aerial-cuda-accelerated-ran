#!/usr/bin/env python3
"""Export Stage-B BC checkpoint to ONNX."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.gnnrl.model import build_model_from_config


class _OnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, obs_cell_features, obs_ue_features, obs_edge_index, obs_edge_attr):
        out = self.model(obs_cell_features, obs_ue_features, obs_edge_index, obs_edge_attr)
        return out["ue_logits"], out["prg_logits"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Stage-B GNN BC checkpoint to ONNX")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt or checkpoint_last.pt")
    p.add_argument("--out", default="", help="ONNX output path (default: <checkpoint_dir>/model.onnx)")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def _check_export_deps() -> None:
    missing = []
    for mod in ("onnx", "onnxscript"):
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            missing.append(mod)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"missing Python package(s): {joined}. "
            "Install with: pip install onnx onnxscript"
        )


def main() -> int:
    args = parse_args()
    _check_export_deps()
    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model_config" not in ckpt:
        raise KeyError("checkpoint missing model_config")
    cfg = ckpt["model_config"]

    model = build_model_from_config(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    out_path = Path(args.out).resolve() if args.out else ckpt_path.parent / "model.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bsz = args.batch_size
    obs_cell = torch.zeros((bsz, cfg["n_cell"], cfg["cell_feat_dim"]), dtype=torch.float32)
    obs_ue = torch.zeros((bsz, cfg["n_active_ue"], cfg["ue_feat_dim"]), dtype=torch.float32)

    # Fully connected directed graph (excluding self-loop) order used by ReplayWriter.
    edge_index = []
    for src in range(cfg["n_cell"]):
        for dst in range(cfg["n_cell"]):
            if src == dst:
                continue
            edge_index.append((src, dst))
    edge_index = torch.tensor(edge_index, dtype=torch.int64).unsqueeze(0).expand(bsz, -1, -1).contiguous()
    obs_edge_attr = torch.zeros((bsz, len(edge_index[0]), cfg["edge_feat_dim"]), dtype=torch.float32)

    wrapper = _OnnxWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (obs_cell, obs_ue, edge_index, obs_edge_attr),
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs_cell_features", "obs_ue_features", "obs_edge_index", "obs_edge_attr"],
        output_names=["ue_logits", "prg_logits"],
        dynamic_axes={
            "obs_cell_features": {0: "batch"},
            "obs_ue_features": {0: "batch"},
            "obs_edge_index": {0: "batch"},
            "obs_edge_attr": {0: "batch"},
            "ue_logits": {0: "batch"},
            "prg_logits": {0: "batch"},
        },
    )

    print(f"onnx written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
