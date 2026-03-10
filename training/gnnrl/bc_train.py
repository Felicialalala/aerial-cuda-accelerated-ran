#!/usr/bin/env python3
"""Behavior cloning training loop for Stage-B GNN policy."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.gnnrl.dataset import IGNORE_INDEX, ReplayBinaryDataset
from training.gnnrl.masks import (
    apply_prg_action_mask,
    apply_ue_action_mask,
    classification_accuracy,
    predicted_legal_ratio,
    sanitize_targets,
)
from training.gnnrl.model import ModelConfig, StageBGnnPolicy


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _epoch_loop(
    model: StageBGnnPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    n_cell: int,
    n_prg: int,
    n_active_ue: int,
    n_sched_ue: int,
    prg_loss_weight: float,
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_ue_loss = 0.0
    total_prg_loss = 0.0
    total_samples = 0

    acc_ue_sum = 0.0
    acc_prg_sum = 0.0
    legal_ue_sum = 0.0
    legal_prg_sum = 0.0
    bad_ue_sum = 0.0
    bad_prg_sum = 0.0
    batch_count = 0

    for batch in loader:
        batch = _to_device(batch, device)
        bsz = int(batch["obs_cell_features"].shape[0])
        batch_count += 1
        total_samples += bsz

        out = model(
            obs_cell_features=batch["obs_cell_features"],
            obs_ue_features=batch["obs_ue_features"],
            obs_edge_index=batch["obs_edge_index"],
            obs_edge_attr=batch["obs_edge_attr"],
        )
        ue_logits_raw = out["ue_logits"]
        prg_logits_raw = out["prg_logits"]

        ue_logits, ue_valid = apply_ue_action_mask(ue_logits_raw, batch["action_mask_ue"], n_cell=n_cell)
        prg_logits, prg_valid = apply_prg_action_mask(
            prg_logits_raw, batch["action_mask_prg_cell"], n_cell=n_cell
        )

        ue_target, ue_bad = sanitize_targets(batch["target_ue_class"], ue_valid, ignore_index=IGNORE_INDEX)
        # replay allocSol layout is [prg, cell] (index = prgIdx * nCell + cIdx)
        prg_target = batch["target_prg_class"].view(bsz, n_prg, n_cell).transpose(1, 2).contiguous()
        prg_target, prg_bad = sanitize_targets(prg_target, prg_valid, ignore_index=IGNORE_INDEX)

        ue_loss = F.cross_entropy(
            ue_logits.reshape(-1, n_active_ue + 1),
            ue_target.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        prg_loss = F.cross_entropy(
            prg_logits.reshape(-1, n_sched_ue + 1),
            prg_target.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        loss = ue_loss + prg_loss_weight * prg_loss

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += float(loss.item()) * bsz
        total_ue_loss += float(ue_loss.item()) * bsz
        total_prg_loss += float(prg_loss.item()) * bsz

        acc_ue_sum += classification_accuracy(ue_logits, ue_target, ignore_index=IGNORE_INDEX)
        acc_prg_sum += classification_accuracy(prg_logits, prg_target, ignore_index=IGNORE_INDEX)
        legal_ue_sum += predicted_legal_ratio(ue_logits, ue_valid)
        legal_prg_sum += predicted_legal_ratio(prg_logits, prg_valid)
        bad_ue_sum += float(ue_bad.float().mean().item())
        bad_prg_sum += float(prg_bad.float().mean().item())

    denom = max(total_samples, 1)
    bdenom = max(batch_count, 1)
    return {
        "loss": total_loss / denom,
        "ue_loss": total_ue_loss / denom,
        "prg_loss": total_prg_loss / denom,
        "ue_acc": acc_ue_sum / bdenom,
        "prg_acc": acc_prg_sum / bdenom,
        "ue_legal_ratio": legal_ue_sum / bdenom,
        "prg_legal_ratio": legal_prg_sum / bdenom,
        "ue_bad_target_ratio": bad_ue_sum / bdenom,
        "prg_bad_target_ratio": bad_prg_sum / bdenom,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage-B GNN policy with behavior cloning")
    p.add_argument("--replay-dir", nargs="+", required=True, help="One or more replay directories")
    p.add_argument("--out-dir", default="training/gnnrl/checkpoints", help="Output checkpoint directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--prg-loss-weight", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-cell-msg-layers", type=int, default=2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cuda" if args.device == "cuda" else "cpu")
    )

    dataset = ReplayBinaryDataset(args.replay_dir)
    dims = dataset.dims
    feat = dataset.feature_dims

    if dims.n_tot_cell != dims.n_cell:
        raise NotImplementedError(
            f"M1 currently expects n_tot_cell == n_cell, got {dims.n_tot_cell} vs {dims.n_cell}"
        )

    n_total = len(dataset)
    if n_total < 2:
        raise RuntimeError(f"dataset too small: {n_total} records")

    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    if n_train < 1:
        n_train = 1
        n_val = n_total - 1

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=g)
    train_idx = perm[:n_train].tolist()
    val_idx = perm[n_train:].tolist()

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_cfg = ModelConfig(
        n_cell=dims.n_cell,
        n_active_ue=dims.n_active_ue,
        n_sched_ue=dims.n_sched_ue,
        n_prg=dims.n_prg,
        cell_feat_dim=feat.cell,
        ue_feat_dim=feat.ue,
        edge_feat_dim=feat.edge,
        hidden_dim=args.hidden_dim,
        num_cell_msg_layers=args.num_cell_msg_layers,
    )
    model = StageBGnnPolicy(model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "checkpoint_best.pt"
    last_path = out_dir / "checkpoint_last.pt"
    summary_path = out_dir / "train_summary.json"

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = _epoch_loop(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            n_cell=dims.n_cell,
            n_prg=dims.n_prg,
            n_active_ue=dims.n_active_ue,
            n_sched_ue=dims.n_sched_ue,
            prg_loss_weight=args.prg_loss_weight,
            device=device,
        )
        with torch.no_grad():
            val_metrics = _epoch_loop(
                model=model,
                loader=val_loader,
                optimizer=None,
                n_cell=dims.n_cell,
                n_prg=dims.n_prg,
                n_active_ue=dims.n_active_ue,
                n_sched_ue=dims.n_sched_ue,
                prg_loss_weight=args.prg_loss_weight,
                device=device,
            )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model.model_config_dict(),
                    "dataset_dims": asdict(dims),
                    "dataset_feature_dims": asdict(feat),
                    "args": vars(args),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model.model_config_dict(),
                "dataset_dims": asdict(dims),
                "dataset_feature_dims": asdict(feat),
                "args": vars(args),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            last_path,
        )

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_ue_acc={train_metrics['ue_acc']:.4f} val_ue_acc={val_metrics['ue_acc']:.4f} "
            f"train_prg_acc={train_metrics['prg_acc']:.4f} val_prg_acc={val_metrics['prg_acc']:.4f}"
        )

    summary = {
        "status": "ok",
        "device": str(device),
        "num_records": n_total,
        "train_records": n_train,
        "val_records": n_val,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "model_config": model.model_config_dict(),
        "dataset_dims": asdict(dims),
        "dataset_feature_dims": asdict(feat),
        "args": vars(args),
        "history": history,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"summary written: {summary_path}")
    print(f"best checkpoint: {best_path}")
    print(f"last checkpoint: {last_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
