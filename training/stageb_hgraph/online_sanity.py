#!/usr/bin/env python3
"""Independent online-bridge graph sanity runner for Stage-B sparse entity graphs."""

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

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.stageb_hgraph.bridge_env import BridgeEnvClient, BridgeEnvConfig
from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from training.stageb_hgraph.mask_checker import GraphMaskChecker
from training.stageb_hgraph.sanity_logger import GraphSanityLogger


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


def _pick_python() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    venv_py = repo_root / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return "python3"


def _build_noop_action(n_sched_ue: int, action_alloc_len: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((n_sched_ue,), -1, dtype=np.int32),
        np.full((action_alloc_len,), -1, dtype=np.int16),
    )


def _build_type0_slot_layout_np(
    action_mask_cell_ue: np.ndarray,
    action_mask_ue: np.ndarray,
    n_sched_ue: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sched_mask = np.asarray(action_mask_cell_ue, dtype=bool) & np.asarray(action_mask_ue, dtype=bool)[None, :]
    raw_counts = sched_mask.sum(axis=1, dtype=np.int64)
    clipped_ends = np.clip(np.cumsum(raw_counts, dtype=np.int64), 0, int(n_sched_ue))
    cell_slot_start = np.concatenate([np.zeros((1,), dtype=np.int64), clipped_ends[:-1]], axis=0)
    slot_counts = clipped_ends - cell_slot_start

    slot_to_cell = np.zeros((n_sched_ue,), dtype=np.int64)
    slot_valid_mask = np.zeros((n_sched_ue,), dtype=bool)
    for c_idx, (start, count) in enumerate(zip(cell_slot_start.tolist(), slot_counts.tolist())):
        if count <= 0:
            continue
        end = min(int(n_sched_ue), int(start + count))
        slot_to_cell[start:end] = int(c_idx)
        slot_valid_mask[start:end] = True
    return slot_to_cell, slot_valid_mask, slot_counts, cell_slot_start


def _build_random_legal_action(
    obs: Dict[str, np.ndarray],
    n_sched_ue: int,
    action_alloc_len: int,
    rng: np.random.Generator,
    blank_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    action_mask_ue = np.asarray(obs["action_mask_ue"], dtype=bool)
    action_mask_cell_ue = np.asarray(obs["action_mask_cell_ue"], dtype=bool)
    action_mask_prg_cell = np.asarray(obs["action_mask_prg_cell"], dtype=bool)

    n_cell, n_active_ue = action_mask_cell_ue.shape
    n_prg = action_mask_prg_cell.shape[1]
    ue_action = np.full((n_sched_ue,), -1, dtype=np.int32)
    prg_action = np.full((action_alloc_len,), -1, dtype=np.int16)

    slot_to_cell, slot_valid_mask, slot_counts, cell_slot_start = _build_type0_slot_layout_np(
        action_mask_cell_ue=action_mask_cell_ue,
        action_mask_ue=action_mask_ue,
        n_sched_ue=n_sched_ue,
    )

    for c_idx in range(n_cell):
        legal_ues = np.flatnonzero(action_mask_cell_ue[c_idx] & action_mask_ue)
        count = int(slot_counts[c_idx])
        if count <= 0 or legal_ues.size == 0:
            continue
        take = min(count, int(legal_ues.size))
        selected = rng.permutation(legal_ues)[:take]
        start = int(cell_slot_start[c_idx])
        ue_action[start : start + take] = selected.astype(np.int32, copy=False)

    for prg_idx in range(n_prg):
        for c_idx in range(n_cell):
            if not bool(action_mask_prg_cell[c_idx, prg_idx]):
                continue
            if rng.random() < blank_prob:
                continue
            start = int(cell_slot_start[c_idx])
            count = int(slot_counts[c_idx])
            if count <= 0:
                continue
            candidate_slots = np.arange(start, start + count, dtype=np.int16)
            if candidate_slots.size == 0:
                continue
            chosen_slot = int(rng.choice(candidate_slots))
            prg_action[prg_idx * n_cell + c_idx] = np.int16(chosen_slot)

    if slot_valid_mask.size != n_sched_ue:
        raise RuntimeError("slot layout size mismatch while building random legal action")
    return ue_action, prg_action


@dataclass(frozen=True)
class EpisodeSummary:
    episode: int
    episode_seed: int
    topology_seed: int
    steps: int
    action_source: str
    mask_ok_all: bool
    max_illegal_cross_cell_up_edges: int
    max_unschedulable_up_edges: int
    max_prg_without_legal_actions: int
    mean_edges_up: float
    mean_edges_pp: float
    mean_prg_candidate_edges: float
    mean_prg_prev_assigned_ratio: float
    prg_dynamic_ranges: Dict[str, Dict[str, float]]


_DYNAMIC_PRG_SUMMARY_KEYS = (
    "prg_prev_assigned_ratio",
    "prg_top1_sinr_db_mean",
    "prg_top2_gap_db_mean",
    "prg_neighbor_max_top1_sinr_db_mean",
    "prg_same_prg_conflict_ratio_mean",
    "prg_ici_proxy_mean",
    "prg_post_eq_p10_sinr_db_mean",
    "prg_post_eq_mean_sinr_db_mean",
    "prg_backlog_weighted_expected_goodput_mbps_mean",
    "prg_low_sinr_hol_ttl_weight_mean",
)


def _aggregate_scalar_rows(rows: List[Dict], key: str) -> Dict[str, float]:
    vals = [float(row[key]) for row in rows if key in row]
    if not vals:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


class OnlineSanityRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.proc: Optional[subprocess.Popen] = None
        self.env: Optional[BridgeEnvClient] = None
        self.socket_path: Optional[str] = None
        self.proc_log_handle = None
        self.explicit_seed_list = _parse_seed_list(args.seed_list)
        self.resolved_topology_seed_mode = _resolve_topology_seed_mode(args.topology_seed_mode, self.explicit_seed_list)
        self.base_topology_seed = int(args.topology_seed)
        self.sim_env_overrides = _parse_sim_env_entries(getattr(args, "sim_env", []))
        self.sim_log_path = Path(args.out_dir).resolve() / "sim_runtime" / "online_bridge_graph_sanity.log"

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

    def reset_episode(self, episode_idx: int):
        self.close_episode(graceful=True)
        episode_seed = self._episode_seed_for(episode_idx)
        topology_seed = int(episode_seed)
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
        obs, reward, info = self.env.reset(seed=episode_seed, episode_horizon=self.args.episode_horizon)
        return obs, float(reward), info, episode_seed, topology_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage-B sparse-graph sanity checks over the online bridge")
    p.add_argument("--sim-bin", required=True, help="Path to multiCellSchedulerUeSelection binary")
    p.add_argument("--sim-args", default="-d 1 -b 2 -x 0 -f 0 -g 100 -r 3000", help="CLI args for the simulation binary")
    p.add_argument("--sim-cwd", default="", help="Optional cwd for launching the simulation")
    p.add_argument("--sim-env", action="append", default=[], help="Extra simulator env vars, format KEY=VALUE")
    p.add_argument("--sim-log-mode", choices=["file", "inherit"], default="file")
    p.add_argument("--socket-path", default="/tmp/cumac_stageb_hgraph.sock")
    p.add_argument("--connect-timeout-s", type=float, default=20.0)
    p.add_argument("--sim-wait-timeout", type=float, default=10.0)
    p.add_argument("--online-persistent", type=int, choices=[0, 1], default=1)
    p.add_argument("--episode-horizon", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=0, help="0 means use episode-horizon as the runner-side cap")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--topology-seed", type=int, default=42)
    p.add_argument("--topology-seed-mode", choices=["auto", "fixed", "sequential", "list_cycle"], default="auto")
    p.add_argument("--seed-list", default="", help="Comma-separated topology seeds for list_cycle mode")
    p.add_argument("--out-dir", default="training/stageb_hgraph/runs/online_sanity")
    p.add_argument("--ue-prg-topk", type=int, default=4)
    p.add_argument("--tb-err-ema-alpha", type=float, default=0.10)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--action-source", choices=["noop", "random_legal"], default="noop")
    p.add_argument(
        "--random-blank-prob",
        type=float,
        default=0.25,
        help="Blank probability used when --action-source=random_legal",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="RNG seed for --action-source=random_legal",
    )
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
    if not (0.0 <= float(args.random_blank_prob) <= 1.0):
        raise ValueError("--random-blank-prob must be within [0,1]")
    explicit_seed_list = _parse_seed_list(args.seed_list)
    if args.topology_seed_mode == "list_cycle" and not explicit_seed_list:
        raise ValueError("--topology-seed-mode list_cycle requires --seed-list")
    if explicit_seed_list and args.topology_seed_mode not in ("auto", "list_cycle"):
        raise ValueError("--seed-list may only be used with --topology-seed-mode auto or list_cycle")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "graph_sanity_tti_stats.jsonl"
    summary_path = out_dir / "graph_sanity_summary.json"

    snapshot_builder = FeatureSnapshotBuilder(
        SnapshotBuilderConfig(tb_err_ema_alpha=float(args.tb_err_ema_alpha))
    )
    edge_generator = SparseEdgeGenerator(EdgeGeneratorConfig(ue_prg_topk=int(args.ue_prg_topk)))
    mask_checker = GraphMaskChecker()
    sanity_logger = GraphSanityLogger()
    runner = OnlineSanityRunner(args)
    action_rng = np.random.default_rng(int(args.random_seed))

    all_rows: List[Dict] = []
    episode_summaries: List[EpisodeSummary] = []
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(args.episode_horizon)
    if max_steps < 1:
        raise ValueError("resolved max_steps must be >= 1")

    try:
        with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
            for episode_idx in range(1, args.episodes + 1):
                snapshot_builder.reset()
                obs, _reward0, info, episode_seed, topology_seed = runner.reset_episode(episode_idx)
                step = 0
                episode_rows: List[Dict] = []
                while True:
                    snapshot = snapshot_builder.build(obs)
                    graph = edge_generator.build_graph(snapshot)
                    mask_report = mask_checker.check(graph)
                    sanity_stats = sanity_logger.summarize(graph, mask_report)
                    row = {
                        "episode": episode_idx,
                        "episode_seed": episode_seed,
                        "topology_seed": topology_seed,
                        "step": step,
                        "tti": int(info["tti"]),
                        "mask_ok": int(mask_report.ok),
                        **sanity_stats.to_dict(),
                    }
                    f_jsonl.write(json.dumps(row, ensure_ascii=True) + "\n")
                    all_rows.append(row)
                    episode_rows.append(row)
                    if args.print_every > 0 and ((step % args.print_every) == 0 or step == max_steps - 1):
                        print(
                            f"[episode {episode_idx:04d}] step={step:04d} tti={row['tti']} "
                            f"E_UP={row['n_edges_up']} E_PP={row['n_edges_pp']} "
                            f"mask_ok={row['mask_ok']} empty_prg={row['prg_without_legal_actions']} "
                            f"bad_up={row['unschedulable_up_edges']} "
                            f"mode={row['candidate_mode']} Fp={row['prg_feat_dim']} L={row['post_eq_layer_dim']} "
                            f"top1_mu={row['prg_top1_sinr_db_mean']:.2f} "
                            f"p10_mu={row['prg_post_eq_p10_sinr_db_mean']:.2f} "
                            f"bw_gp={row['prg_backlog_weighted_expected_goodput_mbps_mean']:.2f} "
                            f"low_tail={row['prg_low_sinr_hol_ttl_weight_mean']:.3f} "
                            f"conf_mu={row['prg_same_prg_conflict_ratio_mean']:.3f} "
                            f"ici_mu={row['prg_ici_proxy_mean']:.3f} "
                            f"prev_on={row['prg_prev_assigned_ratio']:.3f}"
                        )
                    if step + 1 >= max_steps:
                        break
                    assert runner.env is not None
                    dims = runner.env.dims
                    if dims is None:
                        raise RuntimeError("bridge dims missing during step loop")
                    if args.action_source == "random_legal":
                        action_ue, action_prg = _build_random_legal_action(
                            obs=obs,
                            n_sched_ue=int(dims.n_sched_ue),
                            action_alloc_len=int(dims.action_alloc_len),
                            rng=action_rng,
                            blank_prob=float(args.random_blank_prob),
                        )
                    else:
                        action_ue, action_prg = _build_noop_action(dims.n_sched_ue, dims.action_alloc_len)
                    next_obs, _reward, done, next_info = runner.env.step(action_ue, action_prg)
                    step += 1
                    obs, info = next_obs, next_info
                    if done:
                        break

                mean_edges_up = float(np.mean([row["n_edges_up"] for row in episode_rows])) if episode_rows else 0.0
                mean_edges_pp = float(np.mean([row["n_edges_pp"] for row in episode_rows])) if episode_rows else 0.0
                mean_prg_candidate_edges = (
                    float(np.mean([row["per_prg_candidate_edge_mean"] for row in episode_rows])) if episode_rows else 0.0
                )
                mean_prg_prev_assigned_ratio = (
                    float(np.mean([row["prg_prev_assigned_ratio"] for row in episode_rows])) if episode_rows else 0.0
                )
                prg_dynamic_ranges = {
                    key: _aggregate_scalar_rows(episode_rows, key) for key in _DYNAMIC_PRG_SUMMARY_KEYS
                }
                episode_summaries.append(
                    EpisodeSummary(
                        episode=episode_idx,
                        episode_seed=episode_seed,
                        topology_seed=topology_seed,
                        steps=len(episode_rows),
                        action_source=str(args.action_source),
                        mask_ok_all=all(bool(row["mask_ok"]) for row in episode_rows),
                        max_illegal_cross_cell_up_edges=max(
                            [int(row["illegal_cross_cell_up_edges"]) for row in episode_rows] or [0]
                        ),
                        max_unschedulable_up_edges=max(
                            [int(row["unschedulable_up_edges"]) for row in episode_rows] or [0]
                        ),
                        max_prg_without_legal_actions=max(
                            [int(row["prg_without_legal_actions"]) for row in episode_rows] or [0]
                        ),
                        mean_edges_up=mean_edges_up,
                        mean_edges_pp=mean_edges_pp,
                        mean_prg_candidate_edges=mean_prg_candidate_edges,
                        mean_prg_prev_assigned_ratio=mean_prg_prev_assigned_ratio,
                        prg_dynamic_ranges=prg_dynamic_ranges,
                    )
                )
                runner.close_episode(graceful=True)
    finally:
        runner.close_episode(graceful=False)

    summary = {
        "episodes": args.episodes,
        "resolved_max_steps": max_steps,
        "resolved_topology_seed_mode": runner.resolved_topology_seed_mode,
        "sim_bin": args.sim_bin,
        "sim_args": args.sim_args,
        "sim_cwd": args.sim_cwd,
        "action_source": args.action_source,
        "random_blank_prob": float(args.random_blank_prob),
        "random_seed": int(args.random_seed),
        "jsonl_path": str(jsonl_path),
        "all_mask_ok": all(bool(row["mask_ok"]) for row in all_rows) if all_rows else False,
        "max_illegal_cross_cell_up_edges": max([int(row["illegal_cross_cell_up_edges"]) for row in all_rows] or [0]),
        "max_unschedulable_up_edges": max([int(row["unschedulable_up_edges"]) for row in all_rows] or [0]),
        "max_prg_without_legal_actions": max([int(row["prg_without_legal_actions"]) for row in all_rows] or [0]),
        "prg_dynamic_ranges": {
            key: _aggregate_scalar_rows(all_rows, key) for key in _DYNAMIC_PRG_SUMMARY_KEYS
        },
        "episode_summaries": [asdict(ep) for ep in episode_summaries],
    }
    with open(summary_path, "w", encoding="utf-8") as f_summary:
        json.dump(summary, f_summary, ensure_ascii=True, indent=2)

    print(f"[graph-sanity] wrote per-TTI stats: {jsonl_path}")
    print(f"[graph-sanity] wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
