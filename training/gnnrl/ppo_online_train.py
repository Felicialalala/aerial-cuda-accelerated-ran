#!/usr/bin/env python3
"""Online PPO training for Stage-B via Unix socket bridge."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.gnnrl.action_modes import (
    ACTION_MODE_JOINT,
    build_type0_all_ue_action,
    is_prg_only_type0,
    normalize_action_mode,
)
from training.gnnrl.aerial_env import AerialEnvClient, AerialEnvConfig
from training.gnnrl.dataset import IGNORE_INDEX
from training.gnnrl.masks import apply_prg_action_mask, apply_ue_action_mask, sanitize_targets
from training.gnnrl.model import ModelConfig, StageBGnnPolicy, build_model_from_config


def _moving_average(vals: List[float], window: int) -> List[float]:
    if window <= 1 or len(vals) == 0:
        return list(vals)
    out: List[float] = []
    run_sum = 0.0
    q: List[float] = []
    for v in vals:
        q.append(v)
        run_sum += v
        if len(q) > window:
            run_sum -= q.pop(0)
        out.append(run_sum / float(len(q)))
    return out


def _write_history_csv(history: List[Dict[str, float]], csv_path: Path) -> None:
    if len(history) == 0:
        return
    keys = list(history[0].keys())
    for row in history[1:]:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for row in history:
            wr.writerow(row)


def _plot_training_curves(
    history: List[Dict[str, float]],
    png_path: Path,
    smooth_window: int,
    target_kl: Optional[float] = None,
) -> bool:
    if len(history) == 0:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skip plotting (matplotlib unavailable): {e}")
        return False

    iters = [int(r.get("iter", i + 1)) for i, r in enumerate(history)]

    def arr(key: str) -> List[float]:
        return [float(r.get(key, 0.0)) for r in history]

    curves = [
        ("objective", arr("objective")),
        ("policy_loss", arr("policy_loss")),
        ("value_loss", arr("value_loss")),
        ("approx_kl", arr("approx_kl")),
        ("entropy", arr("entropy")),
        ("clipfrac", arr("clipfrac")),
        ("rollout_reward_raw_mean", arr("rollout_reward_raw_mean")),
        ("rollout_reward_mean", arr("rollout_reward_mean")),
        ("rollout_throughput_mbps_mean", arr("rollout_throughput_mbps_mean")),
        ("rollout_total_buffer_mb_mean", arr("rollout_total_buffer_mb_mean")),
        ("rollout_tb_err_rate_mean", arr("rollout_tb_err_rate_mean")),
        ("rollout_fairness_jain_mean", arr("rollout_fairness_jain_mean")),
    ]

    n = len(curves)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.0 * nrows), sharex=True)
    if nrows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = list(axes.flatten())
    for idx, (name, y) in enumerate(curves):
        ax = axes_flat[idx]
        ax.plot(iters, y, linewidth=1.2, alpha=0.45, label=f"{name} (raw)")
        ys = _moving_average(y, smooth_window)
        if smooth_window > 1:
            ax.plot(iters, ys, linewidth=1.8, label=f"{name} (ma{smooth_window})")
        if name == "approx_kl" and target_kl is not None:
            ax.axhline(float(target_kl), color="tab:red", linestyle="--", linewidth=1.0, label="target_kl")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    for ax in axes_flat[n:]:
        ax.axis("off")
    for ax in axes_flat[max(0, n - ncols) : n]:
        ax.set_xlabel("iteration")

    fig.suptitle("Stage-B Online PPO Training Curves", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return True


class StateValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class RunningNorm:
    dim: int
    eps: float = 1.0e-6

    def __post_init__(self):
        self.count = 0
        self.mean = torch.zeros(self.dim, dtype=torch.float32)
        self.m2 = torch.zeros(self.dim, dtype=torch.float32)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().view(-1, self.dim).cpu()
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean = self.mean + delta / float(self.count)
            delta2 = row - self.mean
            self.m2 = self.m2 + delta * delta2

    def std(self) -> torch.Tensor:
        if self.count < 2:
            return torch.ones_like(self.mean)
        var = self.m2 / float(self.count)
        return torch.sqrt(var.clamp_min(self.eps))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = self.std().to(x.device)
        return (x - mean) / std


@dataclass
class RunningScalarNorm:
    eps: float = 1.0e-6

    def __post_init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        dx = x - self.mean
        self.mean += dx / float(self.count)
        dx2 = x - self.mean
        self.m2 += dx * dx2

    def std(self) -> float:
        if self.count < 2:
            return 1.0
        var = self.m2 / float(self.count)
        return float(max(var, self.eps) ** 0.5)

    def normalize(self, x: float) -> float:
        return float((x - self.mean) / self.std())


class OnlineEpisodeRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.proc: Optional[subprocess.Popen] = None
        self.env: Optional[AerialEnvClient] = None
        self.current_obs: Optional[Dict[str, np.ndarray]] = None
        self.episode_idx = 0
        self.socket_path: Optional[str] = None
        self.last_info: Dict = {}

    @property
    def dims(self):
        if self.env is None:
            return None
        return self.env.dims

    def _launch_sim(self, socket_path: str) -> subprocess.Popen:
        sim_env = os.environ.copy()
        sim_env["CUMAC_ONLINE_BRIDGE"] = "1"
        sim_env["CUMAC_ONLINE_SOCKET"] = socket_path
        sim_env["CUMAC_ONLINE_PERSISTENT"] = "1" if int(self.args.online_persistent) == 1 else "0"
        sim_env.setdefault("CUMAC_COMPACT_TTI_LOG", "1")
        sim_env.setdefault("CUMAC_COMPARE_TTI_INTERVAL", "0")

        for kv in self.args.sim_env:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            sim_env[k] = v

        cmd = [self.args.sim_bin] + shlex.split(self.args.sim_args)
        return subprocess.Popen(cmd, env=sim_env, cwd=self.args.sim_cwd)

    def _wait_process(self, timeout_s: float) -> None:
        if self.proc is None:
            return
        try:
            self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)

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
            self._wait_process(timeout_s=self.args.sim_wait_timeout)
            self.proc = None
        if self.socket_path:
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass
        self.socket_path = None
        self.current_obs = None

    def reset_episode(self) -> Dict[str, np.ndarray]:
        self.close_episode(graceful=True)
        self.episode_idx += 1
        self.socket_path = f"{self.args.socket_path}.{os.getpid()}.{self.episode_idx}"
        self.proc = self._launch_sim(socket_path=self.socket_path)

        self.env = AerialEnvClient(AerialEnvConfig(socket_path=self.socket_path, connect_timeout_s=self.args.connect_timeout_s))
        obs = self.env.reset(seed=self.args.seed + self.episode_idx, episode_horizon=self.args.episode_horizon)
        self.current_obs = obs
        self.last_info = {"episode_idx": self.episode_idx}
        return obs

    def step(self, action_ue_select: np.ndarray, action_prg_alloc: np.ndarray):
        if self.env is None or self.current_obs is None:
            raise RuntimeError("episode is not initialized")
        next_obs, reward, done, info = self.env.step(action_ue_select, action_prg_alloc)
        self.current_obs = None if done else next_obs
        self.last_info = info
        return next_obs, reward, done, info


def _load_actor_checkpoint_compat(actor: StageBGnnPolicy, ckpt_state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    Load as many compatible parameters as possible from checkpoint.
    Returns:
        loaded_params: number of matched tensors loaded
        total_params: total tensor count in target actor state dict
    """
    target_state = actor.state_dict()
    compatible = {}
    for k, v in ckpt_state.items():
        if k in target_state and target_state[k].shape == v.shape:
            compatible[k] = v
    actor.load_state_dict(compatible, strict=False)
    return len(compatible), len(target_state)


@dataclass(frozen=True)
class EnvDims:
    n_cell: int
    n_active_ue: int
    n_sched_ue: int
    n_tot_cell: int
    n_prg: int


@dataclass
class RolloutBuffer:
    obs_cell_features: List[torch.Tensor]
    obs_ue_features: List[torch.Tensor]
    obs_edge_index: List[torch.Tensor]
    obs_edge_attr: List[torch.Tensor]
    action_mask_ue: List[torch.Tensor]
    action_mask_cell_ue: List[torch.Tensor]
    action_mask_prg_cell: List[torch.Tensor]
    target_ue_class: List[torch.Tensor]
    target_prg_class: List[torch.Tensor]
    rewards_raw: List[float]
    rewards: List[float]
    reward_throughput_mbps: List[float]
    reward_total_buffer_mb: List[float]
    reward_tb_err_rate: List[float]
    reward_fairness_jain: List[float]
    dones: List[float]
    values: List[float]
    next_values: List[float]
    old_logp: List[float]
    entropies: List[float]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _obs_to_batch(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "obs_cell_features": torch.from_numpy(obs["obs_cell_features"]).to(device=device, dtype=torch.float32).unsqueeze(0),
        "obs_ue_features": torch.from_numpy(obs["obs_ue_features"]).to(device=device, dtype=torch.float32).unsqueeze(0),
        "obs_edge_index": torch.from_numpy(obs["obs_edge_index"]).to(device=device, dtype=torch.int64).unsqueeze(0),
        "obs_edge_attr": torch.from_numpy(obs["obs_edge_attr"]).to(device=device, dtype=torch.float32).unsqueeze(0),
        "action_mask_ue": torch.from_numpy(obs["action_mask_ue"]).to(device=device, dtype=torch.bool).unsqueeze(0),
        "action_mask_cell_ue": torch.from_numpy(obs["action_mask_cell_ue"]).to(device=device, dtype=torch.bool).unsqueeze(0),
        "action_mask_prg_cell": torch.from_numpy(obs["action_mask_prg_cell"]).to(device=device, dtype=torch.bool).unsqueeze(0),
    }


def _pool_state_features(cell_feat: torch.Tensor, ue_feat: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
    return torch.cat([cell_feat.mean(dim=1), ue_feat.mean(dim=1), edge_feat.mean(dim=1)], dim=-1)


def _state_vec(batch: Dict[str, torch.Tensor], state_norm: Optional[RunningNorm]) -> torch.Tensor:
    v = _pool_state_features(batch["obs_cell_features"], batch["obs_ue_features"], batch["obs_edge_attr"])
    if state_norm is not None:
        v = state_norm.normalize(v)
    return v


def _reduce_multi_categorical(logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_class = logits.shape[-1]
    logp_all = F.log_softmax(logits, dim=-1)
    probs = logp_all.exp()

    valid = target != IGNORE_INDEX
    safe_target = target.clamp(min=0, max=n_class - 1)
    chosen_logp = logp_all.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * logp_all).sum(dim=-1)

    bsz = logits.shape[0]
    valid_f = valid.reshape(bsz, -1).float()
    chosen_logp = chosen_logp.reshape(bsz, -1)
    entropy = entropy.reshape(bsz, -1)

    denom = valid_f.sum(dim=-1).clamp_min(1.0)
    mean_logp = (chosen_logp * valid_f).sum(dim=-1) / denom
    mean_entropy = (entropy * valid_f).sum(dim=-1) / denom
    return mean_logp, mean_entropy


def _action_logp_entropy(
    actor_out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    n_cell: int,
    action_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prg_logits, prg_valid = apply_prg_action_mask(actor_out["prg_logits"], batch["action_mask_prg_cell"], n_cell=n_cell)
    prg_target, _ = sanitize_targets(batch["target_prg_class"], prg_valid, ignore_index=IGNORE_INDEX)
    prg_logp, prg_entropy = _reduce_multi_categorical(prg_logits, prg_target)
    if is_prg_only_type0(action_mode):
        return prg_logp, prg_entropy

    ue_logits, ue_valid = apply_ue_action_mask(
        actor_out["ue_logits"],
        batch["action_mask_ue"],
        n_cell=n_cell,
        action_mask_cell_ue=batch["action_mask_cell_ue"],
    )
    ue_target, _ = sanitize_targets(batch["target_ue_class"], ue_valid, ignore_index=IGNORE_INDEX)
    ue_logp, ue_entropy = _reduce_multi_categorical(ue_logits, ue_target)
    return ue_logp + prg_logp, 0.5 * (ue_entropy + prg_entropy)


def _sample_action(actor_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], dims: EnvDims, action_mode: str):
    prg_logits, _ = apply_prg_action_mask(actor_out["prg_logits"], batch["action_mask_prg_cell"], n_cell=dims.n_cell)

    if is_prg_only_type0(action_mode):
        ue_action = build_type0_all_ue_action(batch["action_mask_cell_ue"], n_sched_ue=dims.n_sched_ue).squeeze(0)
        ue_class = torch.where(
            ue_action >= 0,
            ue_action,
            torch.full_like(ue_action, dims.n_active_ue),
        )
        ue_np = ue_action.detach().cpu().numpy().astype(np.int32, copy=False)
        ue_logp = prg_logits.new_zeros(())
        ue_entropy = prg_logits.new_zeros(())
    else:
        ue_logits, _ = apply_ue_action_mask(
            actor_out["ue_logits"],
            batch["action_mask_ue"],
            n_cell=dims.n_cell,
            action_mask_cell_ue=batch["action_mask_cell_ue"],
        )
        ue_dist = Categorical(logits=ue_logits.squeeze(0))
        ue_class = ue_dist.sample()  # [S]
        ue_logp = ue_dist.log_prob(ue_class).mean()
        ue_entropy = ue_dist.entropy().mean()
        ue_np = ue_class.detach().cpu().numpy().astype(np.int32, copy=False)
        ue_np = np.where(ue_np == dims.n_active_ue, -1, ue_np).astype(np.int32, copy=False)

    prg_dist = Categorical(logits=prg_logits.squeeze(0))  # [C, P, S+1]
    prg_class = prg_dist.sample()  # [C, P]
    prg_logp = prg_dist.log_prob(prg_class).mean()
    prg_entropy = prg_dist.entropy().mean()

    prg_np = prg_class.detach().cpu().numpy().astype(np.int16, copy=False)  # [C, P]
    prg_np = np.where(prg_np == dims.n_sched_ue, -1, prg_np).astype(np.int16, copy=False)
    prg_alloc = prg_np.transpose(1, 0).reshape(-1)  # [P, C] -> flatten

    return {
        "action_ue_select": ue_np,
        "action_prg_alloc": prg_alloc,
        "target_ue_class": ue_class.detach().to(torch.long),
        "target_prg_class": prg_class.detach().to(torch.long),
        "logp": float((ue_logp + prg_logp).item()),
        "entropy": float((prg_entropy if is_prg_only_type0(action_mode) else (0.5 * (ue_entropy + prg_entropy))).item()),
    }


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t_len = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype)
    for t in range(t_len - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def _collect_rollout(
    runner: OnlineEpisodeRunner,
    actor: StageBGnnPolicy,
    critic: StateValueNet,
    device: torch.device,
    dims: EnvDims,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    normalize_reward: bool,
    normalize_adv: bool,
    action_mode: str,
    state_norm: Optional[RunningNorm],
    reward_norm: Optional[RunningScalarNorm],
) -> Dict[str, torch.Tensor]:
    actor.eval()
    critic.eval()

    buf = RolloutBuffer(
        obs_cell_features=[],
        obs_ue_features=[],
        obs_edge_index=[],
        obs_edge_attr=[],
        action_mask_ue=[],
        action_mask_cell_ue=[],
        action_mask_prg_cell=[],
        target_ue_class=[],
        target_prg_class=[],
        rewards_raw=[],
        rewards=[],
        reward_throughput_mbps=[],
        reward_total_buffer_mb=[],
        reward_tb_err_rate=[],
        reward_fairness_jain=[],
        dones=[],
        values=[],
        next_values=[],
        old_logp=[],
        entropies=[],
    )

    for _ in range(rollout_steps):
        if runner.current_obs is None:
            runner.reset_episode()
        assert runner.current_obs is not None

        batch = _obs_to_batch(runner.current_obs, device=device)
        with torch.no_grad():
            state_v = _pool_state_features(batch["obs_cell_features"], batch["obs_ue_features"], batch["obs_edge_attr"])
            if state_norm is not None:
                state_norm.update(state_v)
            value = critic(_state_vec(batch, state_norm=state_norm)).item()

            actor_out = actor(
                obs_cell_features=batch["obs_cell_features"],
                obs_ue_features=batch["obs_ue_features"],
                obs_edge_index=batch["obs_edge_index"],
                obs_edge_attr=batch["obs_edge_attr"],
            )
            sampled = _sample_action(actor_out, batch, dims=dims, action_mode=action_mode)

        next_obs, reward_raw, done, info = runner.step(sampled["action_ue_select"], sampled["action_prg_alloc"])
        reward_terms = info.get("reward_terms", (0.0, 0.0, 0.0, 0.0))
        try:
            throughput_mbps = float(reward_terms[0])
            total_buffer_mb = float(reward_terms[1])
            tb_err_rate = float(reward_terms[2])
            fairness_jain = float(reward_terms[3])
        except Exception:
            throughput_mbps = 0.0
            total_buffer_mb = 0.0
            tb_err_rate = 0.0
            fairness_jain = 0.0

        if reward_norm is not None:
            reward_norm.update(float(reward_raw))
            reward = reward_norm.normalize(float(reward_raw)) if normalize_reward else float(reward_raw)
        else:
            reward = float(reward_raw)

        if done:
            next_value = 0.0
            runner.close_episode(graceful=True)
        else:
            next_batch = _obs_to_batch(next_obs, device=device)
            with torch.no_grad():
                next_state_v = _pool_state_features(next_batch["obs_cell_features"], next_batch["obs_ue_features"], next_batch["obs_edge_attr"])
                if state_norm is not None:
                    state_norm.update(next_state_v)
                next_value = critic(_state_vec(next_batch, state_norm=state_norm)).item()

        buf.obs_cell_features.append(batch["obs_cell_features"].squeeze(0).cpu())
        buf.obs_ue_features.append(batch["obs_ue_features"].squeeze(0).cpu())
        buf.obs_edge_index.append(batch["obs_edge_index"].squeeze(0).cpu())
        buf.obs_edge_attr.append(batch["obs_edge_attr"].squeeze(0).cpu())
        buf.action_mask_ue.append(batch["action_mask_ue"].squeeze(0).cpu())
        buf.action_mask_cell_ue.append(batch["action_mask_cell_ue"].squeeze(0).cpu())
        buf.action_mask_prg_cell.append(batch["action_mask_prg_cell"].squeeze(0).cpu())
        buf.target_ue_class.append(sampled["target_ue_class"].cpu())
        buf.target_prg_class.append(sampled["target_prg_class"].cpu())
        buf.rewards_raw.append(float(reward_raw))
        buf.rewards.append(float(reward))
        buf.reward_throughput_mbps.append(throughput_mbps)
        buf.reward_total_buffer_mb.append(total_buffer_mb)
        buf.reward_tb_err_rate.append(tb_err_rate)
        buf.reward_fairness_jain.append(fairness_jain)
        buf.dones.append(1.0 if done else 0.0)
        buf.values.append(float(value))
        buf.next_values.append(float(next_value))
        buf.old_logp.append(float(sampled["logp"]))
        buf.entropies.append(float(sampled["entropy"]))

    rewards_t = torch.tensor(buf.rewards, dtype=torch.float32)
    dones_t = torch.tensor(buf.dones, dtype=torch.float32)
    values_t = torch.tensor(buf.values, dtype=torch.float32)
    next_values_t = torch.tensor(buf.next_values, dtype=torch.float32)
    advantages_t, returns_t = _compute_gae(
        rewards=rewards_t,
        values=values_t,
        next_values=next_values_t,
        dones=dones_t,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    if normalize_adv:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1.0e-8)

    return {
        "obs_cell_features": torch.stack(buf.obs_cell_features),
        "obs_ue_features": torch.stack(buf.obs_ue_features),
        "obs_edge_index": torch.stack(buf.obs_edge_index),
        "obs_edge_attr": torch.stack(buf.obs_edge_attr),
        "action_mask_ue": torch.stack(buf.action_mask_ue),
        "action_mask_cell_ue": torch.stack(buf.action_mask_cell_ue),
        "action_mask_prg_cell": torch.stack(buf.action_mask_prg_cell),
        "target_ue_class": torch.stack(buf.target_ue_class),
        "target_prg_class": torch.stack(buf.target_prg_class),
        "rewards_raw": torch.tensor(buf.rewards_raw, dtype=torch.float32),
        "rewards": rewards_t,
        "reward_throughput_mbps": torch.tensor(buf.reward_throughput_mbps, dtype=torch.float32),
        "reward_total_buffer_mb": torch.tensor(buf.reward_total_buffer_mb, dtype=torch.float32),
        "reward_tb_err_rate": torch.tensor(buf.reward_tb_err_rate, dtype=torch.float32),
        "reward_fairness_jain": torch.tensor(buf.reward_fairness_jain, dtype=torch.float32),
        "dones": dones_t,
        "values": values_t,
        "next_values": next_values_t,
        "old_logp": torch.tensor(buf.old_logp, dtype=torch.float32),
        "old_entropy": torch.tensor(buf.entropies, dtype=torch.float32),
        "advantages": advantages_t,
        "returns": returns_t,
    }


def _slice_batch(buf: Dict[str, torch.Tensor], idx: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in buf.items():
        if torch.is_tensor(v) and v.shape[0] == buf["old_logp"].shape[0]:
            out[k] = v[idx].to(device)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage-B policy with online PPO through bridge")
    p.add_argument("--sim-bin", required=True, help="Path to multiCellSchedulerUeSelection binary")
    p.add_argument("--sim-args", default="-d 1 -b 0 -x 0 -f 0 -g 100 -r 5000", help="CLI args for simulation binary")
    p.add_argument("--sim-cwd", default="", help="Optional cwd for launching simulation")
    p.add_argument("--sim-env", nargs="*", default=[], help="Extra env vars, format KEY=VALUE")
    p.add_argument("--socket-path", default="/tmp/cumac_stageb_online.sock", help="Unix socket path prefix")
    p.add_argument("--connect-timeout-s", type=float, default=20.0)
    p.add_argument("--sim-wait-timeout", type=float, default=10.0)
    p.add_argument(
        "--online-persistent",
        type=int,
        choices=[0, 1],
        default=1,
        help="1: keep one simulator process alive and run continuous TTIs until close; 0: episode-style done/reset",
    )

    p.add_argument("--episode-horizon", type=int, default=400)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--ppo-epochs", type=int, default=6)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--normalize-state", type=int, choices=[0, 1], default=1)
    p.add_argument("--normalize-reward", type=int, choices=[0, 1], default=1)
    p.add_argument("--normalize-adv", type=int, choices=[0, 1], default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--value-loss", choices=["mse", "huber"], default="huber")
    p.add_argument("--value-huber-beta", type=float, default=10.0)
    p.add_argument("--plot-after-train", type=int, choices=[0, 1], default=1)
    p.add_argument("--plot-smooth-window", type=int, default=5, help="Moving-average window for training curves")

    p.add_argument("--init-policy-checkpoint", default="", help="Optional warm-start checkpoint")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-cell-msg-layers", type=int, default=2)
    p.add_argument("--action-mode", default="", help="Override action mode: joint | prg_only_type0")
    p.add_argument("--out-dir", default="training/gnnrl/checkpoints/m3_online_ppo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cuda" if args.device == "cuda" else "cpu")
    )

    if args.sim_cwd == "":
        args.sim_cwd = None

    runner = OnlineEpisodeRunner(args)
    try:
        obs0 = runner.reset_episode()
        dims0 = runner.dims
        if dims0 is None:
            raise RuntimeError("failed to get env dims from reset")
        dims = EnvDims(
            n_cell=int(dims0.n_cell),
            n_active_ue=int(dims0.n_active_ue),
            n_sched_ue=int(dims0.n_sched_ue),
            n_tot_cell=int(dims0.n_tot_cell),
            n_prg=int(dims0.n_prg),
        )

        if dims.n_tot_cell != dims.n_cell:
            raise NotImplementedError(f"online PPO currently expects n_tot_cell == n_cell, got {dims.n_tot_cell} vs {dims.n_cell}")
        if int(dims0.alloc_type) != 0:
            raise NotImplementedError(f"online PPO currently supports alloc_type=0 only, got {dims0.alloc_type}")

        requested_action_mode = normalize_action_mode(args.action_mode) if args.action_mode else ""
        if args.init_policy_checkpoint:
            ckpt = torch.load(args.init_policy_checkpoint, map_location="cpu")
            model_cfg = dict(ckpt["model_config"])
            model_cfg["action_mode"] = requested_action_mode or normalize_action_mode(
                model_cfg.get("action_mode", ACTION_MODE_JOINT)
            )
            actor = build_model_from_config(model_cfg)
            try:
                actor.load_state_dict(ckpt["model_state_dict"], strict=True)
                print(f"[init] strict checkpoint load ok: {args.init_policy_checkpoint}")
            except RuntimeError as e:
                loaded, total = _load_actor_checkpoint_compat(actor, ckpt["model_state_dict"])
                print(
                    "[init] strict checkpoint load failed, fallback to partial compatible load: "
                    f"loaded={loaded}/{total}, ckpt={args.init_policy_checkpoint}"
                )
                print(f"[init] strict load error: {e}")
            args.action_mode = model_cfg["action_mode"]
        else:
            args.action_mode = requested_action_mode or ACTION_MODE_JOINT
            cfg = ModelConfig(
                n_cell=dims.n_cell,
                n_active_ue=dims.n_active_ue,
                n_sched_ue=dims.n_sched_ue,
                n_prg=dims.n_prg,
                cell_feat_dim=5,
                ue_feat_dim=8,
                edge_feat_dim=2,
                hidden_dim=args.hidden_dim,
                num_cell_msg_layers=args.num_cell_msg_layers,
                action_mode=args.action_mode,
            )
            actor = StageBGnnPolicy(cfg)
            model_cfg = actor.model_config_dict()

        print(f"[launch] action_mode={args.action_mode}")
        print(f"[launch] sim_bin={args.sim_bin}")
        print(f"[launch] sim_args={args.sim_args}")
        if args.sim_env:
            print(f"[launch] sim_env={' '.join(args.sim_env)}")

        actor.to(device)
        critic = StateValueNet(input_dim=5 + 8 + 2, hidden_dim=model_cfg["hidden_dim"]).to(device)

        actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr, weight_decay=1e-5)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=1e-5)

        state_norm = RunningNorm(dim=5 + 8 + 2) if args.normalize_state == 1 else None
        reward_norm = RunningScalarNorm() if args.normalize_reward == 1 else None

        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        actor_best_path = out_dir / "ppo_actor_best.pt"
        actor_last_path = out_dir / "ppo_actor_last.pt"
        train_last_path = out_dir / "ppo_train_last.pt"
        summary_path = out_dir / "online_ppo_summary.json"
        history_csv_path = out_dir / "online_ppo_metrics.csv"
        curves_path = out_dir / "online_ppo_curves.png"

        history: List[Dict[str, float]] = []
        best_objective = -1.0e30
        best_iter = -1

        for it in range(1, args.iters + 1):
            rollout = _collect_rollout(
                runner=runner,
                actor=actor,
                critic=critic,
                device=device,
                dims=dims,
                rollout_steps=args.rollout_steps,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                normalize_reward=(args.normalize_reward == 1),
                normalize_adv=(args.normalize_adv == 1),
                action_mode=args.action_mode,
                state_norm=state_norm,
                reward_norm=reward_norm,
            )

            old_logp = rollout["old_logp"]
            n_steps = old_logp.shape[0]
            min_updates_before_early_stop = max(4, n_steps // max(1, args.minibatch_size))

            pol_loss_sum = 0.0
            val_loss_sum = 0.0
            ent_sum = 0.0
            kl_sum = 0.0
            clipfrac_sum = 0.0
            batch_updates = 0
            stop_early = False

            actor.train()
            critic.train()
            for _epoch in range(args.ppo_epochs):
                perm = torch.randperm(n_steps)
                for start in range(0, n_steps, args.minibatch_size):
                    idx = perm[start : start + args.minibatch_size]
                    if idx.numel() == 0:
                        continue
                    mb = _slice_batch(rollout, idx, device=device)

                    actor_out = actor(
                        obs_cell_features=mb["obs_cell_features"],
                        obs_ue_features=mb["obs_ue_features"],
                        obs_edge_index=mb["obs_edge_index"],
                        obs_edge_attr=mb["obs_edge_attr"],
                    )
                    new_logp, entropy = _action_logp_entropy(
                        actor_out,
                        mb,
                        n_cell=dims.n_cell,
                        action_mode=args.action_mode,
                    )
                    value_pred = critic(_state_vec(mb, state_norm=state_norm))

                    old_logp_mb = mb["old_logp"]
                    adv_mb = mb["advantages"]
                    ret_mb = mb["returns"]

                    ratio = torch.exp(new_logp - old_logp_mb)
                    clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                    policy_loss = -torch.min(ratio * adv_mb, clipped_ratio * adv_mb).mean()
                    if args.value_loss == "huber":
                        value_loss = F.smooth_l1_loss(value_pred, ret_mb, beta=args.value_huber_beta)
                    else:
                        value_loss = F.mse_loss(value_pred, ret_mb)
                    entropy_bonus = entropy.mean()

                    actor_loss = policy_loss - args.entropy_coef * entropy_bonus
                    critic_loss = args.value_coef * value_loss

                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                    actor_optimizer.step()

                    critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                    critic_optimizer.step()

                    approx_kl = (old_logp_mb - new_logp).mean()
                    clipfrac = ((ratio - 1.0).abs() > args.clip_eps).float().mean()

                    pol_loss_sum += float(policy_loss.item())
                    val_loss_sum += float(value_loss.item())
                    ent_sum += float(entropy_bonus.item())
                    kl_sum += float(approx_kl.item())
                    clipfrac_sum += float(clipfrac.item())
                    batch_updates += 1

                    if batch_updates >= min_updates_before_early_stop and approx_kl.item() > (1.5 * args.target_kl):
                        stop_early = True
                        break
                if stop_early:
                    break

            denom = max(batch_updates, 1)
            mean_policy_loss = pol_loss_sum / denom
            mean_value_loss = val_loss_sum / denom
            mean_entropy = ent_sum / denom
            mean_kl = kl_sum / denom
            mean_clipfrac = clipfrac_sum / denom
            objective = -mean_policy_loss + args.entropy_coef * mean_entropy

            row = {
                "iter": it,
                "objective": objective,
                "policy_loss": mean_policy_loss,
                "value_loss": mean_value_loss,
                "entropy": mean_entropy,
                "approx_kl": mean_kl,
                "clipfrac": mean_clipfrac,
                "rollout_reward_raw_mean": float(rollout["rewards_raw"].mean().item()),
                "rollout_reward_raw_std": float(rollout["rewards_raw"].std(unbiased=False).item()),
                "rollout_reward_mean": float(rollout["rewards"].mean().item()),
                "rollout_reward_std": float(rollout["rewards"].std(unbiased=False).item()),
                "rollout_adv_mean": float(rollout["advantages"].mean().item()),
                "rollout_adv_std": float(rollout["advantages"].std(unbiased=False).item()),
                "rollout_return_mean": float(rollout["returns"].mean().item()),
                "rollout_return_std": float(rollout["returns"].std(unbiased=False).item()),
                "rollout_old_logp_mean": float(rollout["old_logp"].mean().item()),
                "rollout_entropy_mean": float(rollout["old_entropy"].mean().item()),
                "rollout_throughput_mbps_mean": float(rollout["reward_throughput_mbps"].mean().item()),
                "rollout_throughput_mbps_std": float(rollout["reward_throughput_mbps"].std(unbiased=False).item()),
                "rollout_total_buffer_mb_mean": float(rollout["reward_total_buffer_mb"].mean().item()),
                "rollout_total_buffer_mb_std": float(rollout["reward_total_buffer_mb"].std(unbiased=False).item()),
                "rollout_tb_err_rate_mean": float(rollout["reward_tb_err_rate"].mean().item()),
                "rollout_tb_err_rate_std": float(rollout["reward_tb_err_rate"].std(unbiased=False).item()),
                "rollout_fairness_jain_mean": float(rollout["reward_fairness_jain"].mean().item()),
                "rollout_fairness_jain_std": float(rollout["reward_fairness_jain"].std(unbiased=False).item()),
                "early_stop_kl": bool(stop_early),
                "num_updates": batch_updates,
            }
            history.append(row)

            torch.save(
                {
                    "iter": it,
                    "model_state_dict": actor.state_dict(),
                    "model_config": actor.model_config_dict(),
                    "source": "m3_online_ppo",
                    "metrics": row,
                    "args": vars(args),
                },
                actor_last_path,
            )
            torch.save(
                {
                    "iter": it,
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "model_config": actor.model_config_dict(),
                    "actor_optimizer_state_dict": actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": critic_optimizer.state_dict(),
                    "metrics": row,
                    "args": vars(args),
                    "state_norm": {
                        "count": 0 if state_norm is None else state_norm.count,
                        "mean": [] if state_norm is None else state_norm.mean.tolist(),
                        "std": [] if state_norm is None else state_norm.std().tolist(),
                    },
                    "reward_norm": {
                        "count": 0 if reward_norm is None else reward_norm.count,
                        "mean": 0.0 if reward_norm is None else reward_norm.mean,
                        "std": 1.0 if reward_norm is None else reward_norm.std(),
                    },
                },
                train_last_path,
            )

            if objective > best_objective:
                best_objective = objective
                best_iter = it
                torch.save(
                    {
                        "iter": it,
                        "model_state_dict": actor.state_dict(),
                        "model_config": actor.model_config_dict(),
                        "source": "m3_online_ppo",
                        "metrics": row,
                        "args": vars(args),
                    },
                    actor_best_path,
                )

            print(
                f"[iter {it:03d}] objective={objective:.5f} "
                f"pol_loss={mean_policy_loss:.5f} val_loss={mean_value_loss:.5f} "
                f"entropy={mean_entropy:.5f} kl={mean_kl:.5f} clipfrac={mean_clipfrac:.5f} "
                f"thr={row['rollout_throughput_mbps_mean']:.2f}Mbps "
                f"buf={row['rollout_total_buffer_mb_mean']:.2f}MB "
                f"bler={row['rollout_tb_err_rate_mean']:.4f} fair={row['rollout_fairness_jain_mean']:.4f} "
                f"early_stop_kl={int(stop_early)}"
            )

        summary = {
            "status": "ok",
            "device": str(device),
            "env_dims": asdict(dims),
            "model_config": actor.model_config_dict(),
            "best_objective": best_objective,
            "best_iter": best_iter,
            "best_actor_checkpoint": str(actor_best_path),
            "last_actor_checkpoint": str(actor_last_path),
            "last_train_checkpoint": str(train_last_path),
            "args": vars(args),
            "history": history,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        _write_history_csv(history, history_csv_path)

        plot_ok = False
        if int(args.plot_after_train) == 1:
            plot_ok = _plot_training_curves(
                history=history,
                png_path=curves_path,
                smooth_window=max(1, int(args.plot_smooth_window)),
                target_kl=float(args.target_kl),
            )

        print(f"summary written: {summary_path}")
        print(f"metrics csv written: {history_csv_path}")
        if plot_ok:
            print(f"curves written: {curves_path}")
        print(f"best actor checkpoint: {actor_best_path}")
        return 0
    finally:
        runner.close_episode(graceful=True)


if __name__ == "__main__":
    raise SystemExit(main())
