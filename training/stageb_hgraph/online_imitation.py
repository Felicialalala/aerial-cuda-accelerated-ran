#!/usr/bin/env python3
"""Online imitation warm start for the Stage-B sparse entity graph policy."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.gnnrl.masks import (
    apply_prg_action_mask,
    apply_ue_action_mask,
    build_slot_selection_mask,
    classification_accuracy,
    predicted_legal_ratio,
    sanitize_targets,
)
from training.stageb_hgraph.bridge_env import BridgeEnvClient, BridgeEnvConfig
from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from training.stageb_hgraph.model import HGraphModelConfig, StageBHGraphPolicy
from training.stageb_hgraph.teacher import HeuristicTeacherConfig, build_joint_teacher


def _parse_seed_list(seed_list_arg: str) -> List[int]:
    if not seed_list_arg:
        return []
    out: List[int] = []
    for token in seed_list_arg.replace(",", " ").split():
        out.append(int(token))
    return out


def _parse_sim_env_entries(entries: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kv in entries:
        if "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        out[key] = value
    return out


def _resolve_topology_seed_mode(mode: str, explicit_seed_list: List[int]) -> str:
    if mode != "auto":
        return mode
    return "list_cycle" if explicit_seed_list else "fixed"


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_prg_candidate_count(topk: int, diversity_extra: int) -> int:
    return max(1, int(topk)) + max(0, int(diversity_extra))


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class EpisodeSummary:
    episode: int
    episode_seed: int
    topology_seed: int
    steps: int
    mean_loss: float
    mean_acc: float
    mean_teacher_blank_ratio: float
    mean_reward: float
    mean_prev_assigned_ratio: float


class OnlineEpisodeRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.proc: Optional[subprocess.Popen] = None
        self.env: Optional[BridgeEnvClient] = None
        self.socket_path: Optional[str] = None
        self.live_topology_seed: Optional[int] = None
        self.cached_obs = None
        self.cached_info = None
        self.cached_reward: float = 0.0
        self.proc_log_handle = None
        self.explicit_seed_list = _parse_seed_list(args.seed_list)
        self.resolved_topology_seed_mode = _resolve_topology_seed_mode(args.topology_seed_mode, self.explicit_seed_list)
        self.base_topology_seed = int(args.topology_seed)
        self.sim_env_overrides = _parse_sim_env_entries(getattr(args, "sim_env", []))
        self.sim_log_path = Path(args.out_dir).resolve() / "sim_runtime" / "online_bridge_imitation.log"

    def _episode_seed_for(self, episode_idx: int) -> int:
        if self.resolved_topology_seed_mode == "list_cycle":
            return int(self.explicit_seed_list[(episode_idx - 1) % len(self.explicit_seed_list)])
        if self.resolved_topology_seed_mode == "sequential":
            return int(self.base_topology_seed + max(0, episode_idx - 1))
        return int(self.base_topology_seed)

    def _launch_sim(self, socket_path: str, topology_seed: int, episode_idx: int, episode_seed: int) -> subprocess.Popen:
        sim_env = os.environ.copy()
        sim_env["CUMAC_ONLINE_BRIDGE"] = "1"
        sim_env["CUMAC_ONLINE_SOCKET"] = socket_path
        sim_env["CUMAC_ONLINE_PERSISTENT"] = "1" if int(self.args.online_persistent) == 1 else "0"
        sim_env.setdefault("CUMAC_COMPACT_TTI_LOG", "1")
        sim_env.setdefault("CUMAC_COMPARE_TTI_INTERVAL", "0")
        for key, value in self.sim_env_overrides.items():
            sim_env[key] = value
        sim_env["CUMAC_TOPOLOGY_SEED"] = str(int(topology_seed))
        self._maybe_extend_ld_library_path(sim_env)

        cmd = [self.args.sim_bin] + shlex.split(self.args.sim_args)
        if self.args.sim_log_mode == "file":
            self.sim_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.proc_log_handle = open(self.sim_log_path, "a", encoding="utf-8", buffering=1)
            launch_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self.proc_log_handle.write(
                f"\n===== [{launch_ts}] episode={episode_idx} episode_seed={episode_seed} "
                f"topology_seed={topology_seed} socket={socket_path} =====\n"
            )
            return subprocess.Popen(
                cmd,
                env=sim_env,
                cwd=self.args.sim_cwd,
                stdout=self.proc_log_handle,
                stderr=subprocess.STDOUT,
            )
        return subprocess.Popen(cmd, env=sim_env, cwd=self.args.sim_cwd)

    def _maybe_extend_ld_library_path(self, sim_env: Dict[str, str]) -> None:
        sim_bin = Path(self.args.sim_bin).resolve()
        build_cumac_dir = sim_bin.parents[2] if len(sim_bin.parents) >= 3 else None
        if build_cumac_dir is None:
            return
        lib_dirs: List[str] = []
        for candidate in (
            build_cumac_dir / "src",
            build_cumac_dir / "lib" / "cumac_msg",
        ):
            if candidate.is_dir():
                lib_dirs.append(str(candidate))
        if not lib_dirs:
            return
        current = sim_env.get("LD_LIBRARY_PATH", "")
        sim_env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([current] if current else []))

    def close_episode(self, graceful: bool = True) -> None:
        if self.env is not None:
            try:
                if graceful:
                    self.env.close()
            except Exception:
                pass
            finally:
                self.env = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=self.args.sim_wait_timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
            self.proc = None
        if self.proc_log_handle is not None:
            try:
                self.proc_log_handle.flush()
                self.proc_log_handle.close()
            except Exception:
                pass
            finally:
                self.proc_log_handle = None
        if self.socket_path:
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass
            self.socket_path = None
        self.live_topology_seed = None
        self.cached_obs = None
        self.cached_info = None
        self.cached_reward = 0.0

    def _can_reuse_episode(self, topology_seed: int) -> bool:
        if int(self.args.online_persistent) != 1:
            return False
        if self.proc is None or self.env is None or self.socket_path is None:
            return False
        if self.proc.poll() is not None:
            return False
        return self.live_topology_seed == int(topology_seed)

    def can_continue_stream(self, topology_seed: int) -> bool:
        return self._can_reuse_episode(topology_seed) and self.cached_obs is not None and self.cached_info is not None

    def begin_rollout(self, episode_idx: int):
        episode_seed = self._episode_seed_for(episode_idx)
        topology_seed = int(episode_seed)
        if self.can_continue_stream(topology_seed):
            print(
                f"[episode {episode_idx:04d}] continue "
                f"episode_seed={episode_seed} topology_seed={topology_seed} "
                f"persistent={int(self.args.online_persistent)} socket={self.socket_path}"
            )
            return self.cached_obs, float(self.cached_reward), self.cached_info, episode_seed, topology_seed, True

        obs, reward, info, episode_seed, topology_seed = self.reset_episode(episode_idx)
        self.cached_obs = obs
        self.cached_info = info
        self.cached_reward = float(reward)
        return obs, float(reward), info, episode_seed, topology_seed, False

    def record_transition(self, obs, reward: float, info) -> None:
        self.cached_obs = obs
        self.cached_info = info
        self.cached_reward = float(reward)

    def reset_episode(self, episode_idx: int):
        episode_seed = self._episode_seed_for(episode_idx)
        topology_seed = int(episode_seed)
        if self._can_reuse_episode(topology_seed):
            assert self.env is not None
            print(
                f"[episode {episode_idx:04d}] reuse "
                f"episode_seed={episode_seed} topology_seed={topology_seed} "
                f"persistent={int(self.args.online_persistent)} socket={self.socket_path}"
            )
            try:
                obs, reward, info = self.env.reset(seed=episode_seed, episode_horizon=self.args.episode_horizon)
                self.cached_obs = obs
                self.cached_info = info
                self.cached_reward = float(reward)
                return obs, float(reward), info, episode_seed, topology_seed
            except Exception as exc:
                print(
                    f"[episode {episode_idx:04d}] reuse_failed "
                    f"topology_seed={topology_seed} err={exc}; relaunching simulator"
                )
                self.close_episode(graceful=False)

        self.close_episode(graceful=True)
        self.socket_path = f"{self.args.socket_path}.{os.getpid()}.{episode_idx}"
        self.proc = self._launch_sim(
            socket_path=self.socket_path,
            topology_seed=topology_seed,
            episode_idx=episode_idx,
            episode_seed=episode_seed,
        )
        print(
            f"[episode {episode_idx:04d}] launch "
            f"episode_seed={episode_seed} topology_seed={topology_seed} "
            f"persistent={int(self.args.online_persistent)} socket={self.socket_path}"
        )
        self.env = BridgeEnvClient(
            BridgeEnvConfig(socket_path=self.socket_path, connect_timeout_s=self.args.connect_timeout_s)
        )
        self.live_topology_seed = topology_seed
        obs, reward, info = self.env.reset(seed=episode_seed, episode_horizon=self.args.episode_horizon)
        self.cached_obs = obs
        self.cached_info = info
        self.cached_reward = float(reward)
        return obs, float(reward), info, episode_seed, topology_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage-B sparse-graph online imitation warm start")
    p.add_argument("--sim-bin", required=True, help="Path to multiCellSchedulerUeSelection binary")
    p.add_argument("--sim-args", default="-d 1 -b 2 -x 0 -f 0 -g 100 -r 3000", help="CLI args for the simulation binary")
    p.add_argument("--sim-cwd", default="", help="Optional cwd for launching the simulation")
    p.add_argument("--sim-env", action="append", default=[], help="Extra simulator env vars, format KEY=VALUE")
    p.add_argument("--sim-log-mode", choices=["file", "inherit"], default="file")
    p.add_argument("--socket-path", default="/tmp/cumac_stageb_hgraph_imitation.sock")
    p.add_argument("--connect-timeout-s", type=float, default=20.0)
    p.add_argument("--sim-wait-timeout", type=float, default=10.0)
    p.add_argument("--online-persistent", type=int, choices=[0, 1], default=1)
    p.add_argument("--episode-horizon", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=0, help="0 means use episode-horizon as the runner-side cap")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--topology-seed", type=int, default=42)
    p.add_argument("--topology-seed-mode", choices=["auto", "fixed", "sequential", "list_cycle"], default="auto")
    p.add_argument("--seed-list", default="", help="Comma-separated topology seeds for list_cycle mode")
    p.add_argument("--out-dir", default="training/stageb_hgraph/runs/online_imitation")
    p.add_argument("--ue-prg-topk", type=int, default=4)
    p.add_argument("--ue-prg-diversity-extra", type=int, default=0)
    p.add_argument("--tb-err-ema-alpha", type=float, default=0.10)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--message-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--teacher-mode", choices=["heuristic_graph", "pfq_passthrough"], default="pfq_passthrough")
    p.add_argument("--save-every-episodes", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.sim_cwd == "":
        args.sim_cwd = None
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.episode_horizon < 1:
        raise ValueError("--episode-horizon must be >= 1")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0")
    if args.ue_prg_diversity_extra < 0:
        raise ValueError("--ue-prg-diversity-extra must be >= 0")
    explicit_seed_list = _parse_seed_list(args.seed_list)
    if args.topology_seed_mode == "list_cycle" and not explicit_seed_list:
        raise ValueError("--topology-seed-mode list_cycle requires --seed-list")
    if explicit_seed_list and args.topology_seed_mode not in ("auto", "list_cycle"):
        raise ValueError("--seed-list may only be used with --topology-seed-mode auto or list_cycle")

    _set_seed(int(args.seed))
    device = _pick_device(args.device)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "online_imitation_metrics.jsonl"
    summary_path = out_dir / "online_imitation_summary.json"
    ckpt_last_path = out_dir / "hgraph_imitation_last.pt"
    ckpt_best_path = out_dir / "hgraph_imitation_best.pt"
    max_prg_candidates = _resolve_prg_candidate_count(args.ue_prg_topk, args.ue_prg_diversity_extra)

    snapshot_builder = FeatureSnapshotBuilder(
        SnapshotBuilderConfig(tb_err_ema_alpha=float(args.tb_err_ema_alpha))
    )
    edge_generator = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            ue_prg_topk=int(args.ue_prg_topk),
            ue_prg_diversity_extra=int(args.ue_prg_diversity_extra),
        )
    )
    teacher = build_joint_teacher(
        args.teacher_mode,
        heuristic_cfg=HeuristicTeacherConfig(max_candidates=max_prg_candidates),
        max_candidates=max_prg_candidates,
    )

    model = StageBHGraphPolicy(
        HGraphModelConfig(
            hidden_dim=int(args.hidden_dim),
            message_layers=int(args.message_layers),
            dropout=float(args.dropout),
            max_prg_candidates=max_prg_candidates,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    runner = OnlineEpisodeRunner(args)
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(args.episode_horizon)
    if max_steps < 1:
        raise ValueError("resolved max_steps must be >= 1")
    ue_loss_weight = 1.0

    all_rows: List[Dict] = []
    episode_summaries: List[EpisodeSummary] = []
    best_mean_loss = float("inf")
    global_step = 0

    try:
        with open(metrics_path, "w", encoding="utf-8") as f_jsonl:
            for episode_idx in range(1, args.episodes + 1):
                obs, _reward0, info, episode_seed, topology_seed, resumed_stream = runner.begin_rollout(episode_idx)
                if not resumed_stream:
                    snapshot_builder.reset()
                step = 0
                episode_rows: List[Dict] = []
                while True:
                    snapshot = snapshot_builder.build(obs)
                    graph = edge_generator.build_graph(snapshot)
                    meta = graph.snapshot.static_meta

                    teacher_out = teacher.label_graph(graph)
                    teacher_target_ue = teacher_out["target_ue_class"].to(device=device, dtype=torch.long)
                    teacher_target_prg = teacher_out["target_prg_class"].to(device=device, dtype=torch.long)

                    model.train(True)
                    out = model(graph)
                    ue_logits_raw = out["ue_logits"].unsqueeze(0)
                    prg_logits_raw = out["prg_logits"].unsqueeze(0)

                    action_mask_ue = (
                        meta.action_mask_ue
                        if meta.action_mask_ue is not None
                        else torch.ones((meta.n_ue,), dtype=torch.bool)
                    ).to(device=device, dtype=torch.bool).unsqueeze(0)
                    action_mask_cell_ue = meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).unsqueeze(0)
                    action_mask_prg_cell = meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).unsqueeze(0)

                    ue_logits, ue_valid = apply_ue_action_mask(
                        ue_logits_raw,
                        action_mask_ue,
                        n_cell=meta.n_cell,
                        action_mask_cell_ue=action_mask_cell_ue,
                    )
                    ue_target, ue_bad = sanitize_targets(
                        teacher_target_ue.unsqueeze(0),
                        ue_valid,
                    )
                    selected_slot_mask = build_slot_selection_mask(
                        teacher_target_ue.unsqueeze(0),
                        action_mask_ue,
                        n_cell=meta.n_cell,
                        action_mask_cell_ue=action_mask_cell_ue,
                    )
                    prg_logits, prg_valid = apply_prg_action_mask(
                        prg_logits_raw,
                        meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).reshape(1, meta.n_cell, meta.n_prg),
                        n_cell=meta.n_cell,
                        action_mask_cell_ue=action_mask_cell_ue,
                        action_mask_ue=action_mask_ue,
                        selected_slot_mask=selected_slot_mask,
                    )
                    prg_target, prg_bad = sanitize_targets(
                        teacher_target_prg.unsqueeze(0),
                        prg_valid,
                    )

                    ue_loss = F.cross_entropy(
                        ue_logits.reshape(-1, meta.n_ue + 1),
                        ue_target.reshape(-1),
                        ignore_index=-100,
                    )
                    prg_loss = F.cross_entropy(
                        prg_logits.reshape(-1, prg_logits.shape[-1]),
                        prg_target.reshape(-1),
                        ignore_index=-100,
                    )
                    loss = prg_loss + ue_loss_weight * ue_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                    optimizer.step()

                    with torch.no_grad():
                        ue_acc = classification_accuracy(ue_logits, ue_target, ignore_index=-100)
                        prg_acc = classification_accuracy(prg_logits, prg_target, ignore_index=-100)
                        joint_acc = 0.5 * (ue_acc + prg_acc)

                    assert runner.env is not None
                    next_obs, reward_raw, done, next_info = runner.env.step(
                        teacher_out["action_ue_select"].cpu().numpy(),
                        teacher_out["action_prg_alloc"].cpu().numpy(),
                    )
                    runner.record_transition(next_obs, float(reward_raw), next_info)

                    prg_feat = snapshot.prg_features.values
                    row = {
                        "episode": episode_idx,
                        "episode_seed": episode_seed,
                        "topology_seed": topology_seed,
                        "step": step,
                        "global_step": global_step,
                        "tti": int(info["tti"]),
                        "loss": float(loss.item()),
                        "ue_loss": float(ue_loss.item()),
                        "prg_loss": float(prg_loss.item()),
                        "ue_acc": ue_acc,
                        "prg_acc": prg_acc,
                        "joint_acc": joint_acc,
                        "ue_legal_ratio": predicted_legal_ratio(ue_logits, ue_valid),
                        "prg_legal_ratio": predicted_legal_ratio(prg_logits, prg_valid),
                        "ue_bad_target_ratio": float(ue_bad.float().mean().item()),
                        "prg_bad_target_ratio": float(prg_bad.float().mean().item()),
                        "teacher_blank_ratio": float(teacher_out["teacher_blank_ratio"].item()),
                        "reward_raw": float(reward_raw),
                        "candidate_mode": graph.snapshot.static_meta.candidate_mode,
                        "post_eq_layer_dim": (
                            int(graph.snapshot.static_meta.post_eq_sinr.shape[-1])
                            if graph.snapshot.static_meta.post_eq_sinr is not None
                            and graph.snapshot.static_meta.post_eq_sinr.dim() == 3
                            else (1 if graph.snapshot.static_meta.post_eq_sinr is not None else 0)
                        ),
                        "n_edges_up": int(graph.edges["E_UP"].num_edges),
                        "n_edges_pp": int(graph.edges["E_PP"].num_edges),
                        "prg_prev_assigned_ratio": float(prg_feat[:, 8].mean().item()),
                        "prg_top1_sinr_db_mean": float(prg_feat[:, 0].mean().item()),
                        "prg_same_prg_conflict_ratio_mean": float(prg_feat[:, 4].mean().item()),
                        "prg_ici_proxy_mean": float(prg_feat[:, 5].mean().item()),
                    }
                    f_jsonl.write(json.dumps(row, ensure_ascii=True) + "\n")
                    all_rows.append(row)
                    episode_rows.append(row)

                    if args.print_every > 0 and ((step % args.print_every) == 0 or step == max_steps - 1):
                        print(
                            f"[episode {episode_idx:04d}] step={step:04d} tti={row['tti']} "
                            f"loss={row['loss']:.4f} ue_acc={row['ue_acc']:.3f} prg_acc={row['prg_acc']:.3f} "
                            f"blank={row['teacher_blank_ratio']:.3f} reward={row['reward_raw']:.4f} "
                            f"mode={row['candidate_mode']} L={row['post_eq_layer_dim']} "
                            f"top1_mu={row['prg_top1_sinr_db_mean']:.2f} "
                            f"prev_on={row['prg_prev_assigned_ratio']:.3f}"
                        )

                    global_step += 1
                    step += 1
                    obs, info = next_obs, next_info
                    if done or step >= max_steps:
                        break

                mean_loss = float(np.mean([row["loss"] for row in episode_rows])) if episode_rows else 0.0
                mean_acc = float(np.mean([row["joint_acc"] for row in episode_rows])) if episode_rows else 0.0
                mean_blank = (
                    float(np.mean([row["teacher_blank_ratio"] for row in episode_rows])) if episode_rows else 0.0
                )
                mean_reward = float(np.mean([row["reward_raw"] for row in episode_rows])) if episode_rows else 0.0
                mean_prev = (
                    float(np.mean([row["prg_prev_assigned_ratio"] for row in episode_rows])) if episode_rows else 0.0
                )
                episode_summaries.append(
                    EpisodeSummary(
                        episode=episode_idx,
                        episode_seed=episode_seed,
                        topology_seed=topology_seed,
                        steps=len(episode_rows),
                        mean_loss=mean_loss,
                        mean_acc=mean_acc,
                        mean_teacher_blank_ratio=mean_blank,
                        mean_reward=mean_reward,
                        mean_prev_assigned_ratio=mean_prev,
                    )
                )

                checkpoint = {
                    "model_state": model.state_dict(),
                    "model_config": model.model_config_dict(),
                    "train_args": vars(args),
                    "episode": episode_idx,
                    "global_step": global_step,
                    "mean_loss": mean_loss,
                    "mean_acc": mean_acc,
                }
                if int(args.save_every_episodes) > 0 and (episode_idx % int(args.save_every_episodes) == 0):
                    torch.save(checkpoint, ckpt_last_path)
                if mean_loss < best_mean_loss:
                    best_mean_loss = mean_loss
                    torch.save(checkpoint, ckpt_best_path)

    finally:
        runner.close_episode(graceful=False)

    summary = {
        "episodes": int(args.episodes),
        "resolved_max_steps": int(max_steps),
        "resolved_topology_seed_mode": runner.resolved_topology_seed_mode,
        "device": str(device),
        "teacher_mode": args.teacher_mode,
        "sim_bin": args.sim_bin,
        "sim_args": args.sim_args,
        "sim_cwd": args.sim_cwd,
        "metrics_path": str(metrics_path),
        "last_checkpoint": str(ckpt_last_path),
        "best_checkpoint": str(ckpt_best_path),
        "best_mean_loss": float(best_mean_loss if np.isfinite(best_mean_loss) else 0.0),
        "episode_summaries": [asdict(ep) for ep in episode_summaries],
        "mean_loss_all": float(np.mean([row["loss"] for row in all_rows])) if all_rows else 0.0,
        "mean_ue_acc_all": float(np.mean([row["ue_acc"] for row in all_rows])) if all_rows else 0.0,
        "mean_prg_acc_all": float(np.mean([row["prg_acc"] for row in all_rows])) if all_rows else 0.0,
        "mean_acc_all": float(np.mean([row["joint_acc"] for row in all_rows])) if all_rows else 0.0,
        "mean_reward_all": float(np.mean([row["reward_raw"] for row in all_rows])) if all_rows else 0.0,
    }
    with open(summary_path, "w", encoding="utf-8") as f_summary:
        json.dump(summary, f_summary, ensure_ascii=True, indent=2)

    print(f"[online-imitation] wrote metrics: {metrics_path}")
    print(f"[online-imitation] wrote summary: {summary_path}")
    print(f"[online-imitation] last checkpoint: {ckpt_last_path}")
    print(f"[online-imitation] best checkpoint: {ckpt_best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
