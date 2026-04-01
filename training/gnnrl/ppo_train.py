#!/usr/bin/env python3
"""Masked PPO training loop for Stage-B using replay transitions."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.gnnrl.action_modes import ACTION_MODE_JOINT, is_prg_only_type0, normalize_action_mode
from training.gnnrl.dataset import IGNORE_INDEX, ReplayBinaryDataset
from training.gnnrl.masks import apply_prg_action_mask, apply_ue_action_mask, sanitize_targets
from training.gnnrl.model import ModelConfig, StageBGnnPolicy, build_model_from_config


class StateValueNet(nn.Module):
    """Simple critic over pooled observation features."""

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


@dataclass(frozen=True)
class NormStats:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    reward_mean: float
    reward_std: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pool_state_features(
    cell_feat: torch.Tensor,
    ue_feat: torch.Tensor,
    edge_feat: torch.Tensor,
) -> torch.Tensor:
    cell = cell_feat.mean(dim=1)
    ue = ue_feat.mean(dim=1)
    edge = edge_feat.mean(dim=1)
    return torch.cat([cell, ue, edge], dim=-1)


def _normalize_state_vector(
    vec: torch.Tensor,
    state_mean: Optional[torch.Tensor],
    state_std: Optional[torch.Tensor],
) -> torch.Tensor:
    if state_mean is None or state_std is None:
        return vec
    return (vec - state_mean) / state_std.clamp_min(1.0e-6)


def _state_vector(
    batch: Dict[str, torch.Tensor],
    state_mean: Optional[torch.Tensor] = None,
    state_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    vec = _pool_state_features(
        batch["obs_cell_features"],
        batch["obs_ue_features"],
        batch["obs_edge_attr"],
    )
    return _normalize_state_vector(vec, state_mean=state_mean, state_std=state_std)


def _next_state_vector(
    batch: Dict[str, torch.Tensor],
    state_mean: Optional[torch.Tensor] = None,
    state_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    vec = _pool_state_features(
        batch["next_cell_features"],
        batch["next_ue_features"],
        batch["next_edge_attr"],
    )
    return _normalize_state_vector(vec, state_mean=state_mean, state_std=state_std)


def _compute_norm_stats(dataset: ReplayBinaryDataset) -> NormStats:
    state_sum: Optional[torch.Tensor] = None
    state_sq_sum: Optional[torch.Tensor] = None
    state_count = 0
    reward_sum = 0.0
    reward_sq_sum = 0.0
    reward_count = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        obs_vec = _pool_state_features(
            sample["obs_cell_features"].unsqueeze(0),
            sample["obs_ue_features"].unsqueeze(0),
            sample["obs_edge_attr"].unsqueeze(0),
        ).squeeze(0).float()
        next_vec = _pool_state_features(
            sample["next_cell_features"].unsqueeze(0),
            sample["next_ue_features"].unsqueeze(0),
            sample["next_edge_attr"].unsqueeze(0),
        ).squeeze(0).float()
        for vec in (obs_vec, next_vec):
            if state_sum is None:
                state_sum = torch.zeros_like(vec)
                state_sq_sum = torch.zeros_like(vec)
            state_sum += vec
            state_sq_sum += vec * vec
            state_count += 1

        r = float(sample["reward_scalar"].item())
        reward_sum += r
        reward_sq_sum += r * r
        reward_count += 1

    if state_sum is None or state_sq_sum is None or state_count == 0 or reward_count == 0:
        raise RuntimeError("failed to compute normalization stats from replay dataset")

    state_mean = state_sum / float(state_count)
    state_var = state_sq_sum / float(state_count) - state_mean * state_mean
    state_std = torch.sqrt(state_var.clamp_min(1.0e-6))

    reward_mean = reward_sum / float(reward_count)
    reward_var = reward_sq_sum / float(reward_count) - reward_mean * reward_mean
    reward_std = max(reward_var, 1.0e-6) ** 0.5

    return NormStats(
        state_mean=state_mean,
        state_std=state_std,
        reward_mean=reward_mean,
        reward_std=reward_std,
    )


def _reduce_multi_categorical(logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        logits: [B, ..., K]
        target: [B, ...], IGNORE_INDEX entries are skipped
    Returns:
        mean_logp_per_sample: [B]
        mean_entropy_per_sample: [B]
        valid_ratio_per_sample: [B]
    """
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
    valid_ratio = valid_f.mean(dim=-1)
    return mean_logp, mean_entropy, valid_ratio


def _action_logp_entropy(
    actor_out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    n_cell: int,
    n_prg: int,
    action_mode: str,
) -> Dict[str, torch.Tensor]:
    prg_logits, prg_valid = apply_prg_action_mask(actor_out["prg_logits"], batch["action_mask_prg_cell"], n_cell=n_cell)

    prg_target = (
        batch["target_prg_class"]
        .view(batch["target_prg_class"].shape[0], n_prg, n_cell)
        .transpose(1, 2)
        .contiguous()
    )
    prg_target, prg_bad = sanitize_targets(prg_target, prg_valid, ignore_index=IGNORE_INDEX)
    prg_logp, prg_entropy, prg_valid_ratio = _reduce_multi_categorical(prg_logits, prg_target)

    if is_prg_only_type0(action_mode):
        ones = torch.ones_like(prg_logp)
        zeros = torch.zeros_like(prg_logp)
        return {
            "logp": prg_logp,
            "entropy": prg_entropy,
            "ue_valid_ratio": ones,
            "prg_valid_ratio": prg_valid_ratio,
            "ue_bad_ratio": zeros,
            "prg_bad_ratio": prg_bad.float().reshape(prg_bad.shape[0], -1).mean(dim=-1),
        }

    ue_logits, ue_valid = apply_ue_action_mask(
        actor_out["ue_logits"],
        batch["action_mask_ue"],
        n_cell=n_cell,
        action_mask_cell_ue=batch.get("action_mask_cell_ue"),
    )
    ue_target, ue_bad = sanitize_targets(batch["target_ue_class"], ue_valid, ignore_index=IGNORE_INDEX)
    ue_logp, ue_entropy, ue_valid_ratio = _reduce_multi_categorical(ue_logits, ue_target)

    return {
        "logp": ue_logp + prg_logp,
        "entropy": 0.5 * (ue_entropy + prg_entropy),
        "ue_valid_ratio": ue_valid_ratio,
        "prg_valid_ratio": prg_valid_ratio,
        "ue_bad_ratio": ue_bad.float().reshape(ue_bad.shape[0], -1).mean(dim=-1),
        "prg_bad_ratio": prg_bad.float().reshape(prg_bad.shape[0], -1).mean(dim=-1),
    }


def _sample_to_batch(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0).to(device)
        else:
            out[k] = v
    return out


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
    dataset: ReplayBinaryDataset,
    actor: StageBGnnPolicy,
    critic: StateValueNet,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
    normalize_adv: bool,
    normalize_reward: bool,
    state_mean: Optional[torch.Tensor] = None,
    state_std: Optional[torch.Tensor] = None,
    reward_mean: float = 0.0,
    reward_std: float = 1.0,
) -> Dict[str, torch.Tensor]:
    actor.eval()
    critic.eval()

    obs_cell, obs_ue, obs_edge_idx, obs_edge_attr = [], [], [], []
    next_cell, next_ue, next_edge_attr = [], [], []
    action_mask_ue, action_mask_cell_ue, action_mask_prg = [], [], []
    target_ue, target_prg = [], []
    rewards, dones = [], []
    values, next_values = [], []
    old_logp, entropies = [], []
    ue_valid_ratio, prg_valid_ratio = [], []
    ue_bad_ratio, prg_bad_ratio = [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        batch = _sample_to_batch(sample, device=device)

        with torch.no_grad():
            actor_out = actor(
                obs_cell_features=batch["obs_cell_features"],
                obs_ue_features=batch["obs_ue_features"],
                obs_edge_index=batch["obs_edge_index"],
                obs_edge_attr=batch["obs_edge_attr"],
            )
            act_stat = _action_logp_entropy(
                actor_out,
                batch,
                n_cell=dataset.dims.n_cell,
                n_prg=dataset.dims.n_prg,
                action_mode=actor.action_mode,
            )

            v = critic(_state_vector(batch, state_mean=state_mean, state_std=state_std))
            nv = critic(_next_state_vector(batch, state_mean=state_mean, state_std=state_std))

        obs_cell.append(sample["obs_cell_features"])
        obs_ue.append(sample["obs_ue_features"])
        obs_edge_idx.append(sample["obs_edge_index"])
        obs_edge_attr.append(sample["obs_edge_attr"])
        next_cell.append(sample["next_cell_features"])
        next_ue.append(sample["next_ue_features"])
        next_edge_attr.append(sample["next_edge_attr"])
        action_mask_ue.append(sample["action_mask_ue"])
        action_mask_cell_ue.append(sample["action_mask_cell_ue"])
        action_mask_prg.append(sample["action_mask_prg_cell"])
        target_ue.append(sample["target_ue_class"])
        target_prg.append(sample["target_prg_class"])

        rewards.append(sample["reward_scalar"].float())
        dones.append(sample["done"].float())
        values.append(v.squeeze(0).cpu())
        next_values.append(nv.squeeze(0).cpu())
        old_logp.append(act_stat["logp"].squeeze(0).cpu())
        entropies.append(act_stat["entropy"].squeeze(0).cpu())
        ue_valid_ratio.append(act_stat["ue_valid_ratio"].squeeze(0).cpu())
        prg_valid_ratio.append(act_stat["prg_valid_ratio"].squeeze(0).cpu())
        ue_bad_ratio.append(act_stat["ue_bad_ratio"].squeeze(0).cpu())
        prg_bad_ratio.append(act_stat["prg_bad_ratio"].squeeze(0).cpu())

    rewards_raw_t = torch.stack(rewards).float()
    if normalize_reward:
        rewards_t = (rewards_raw_t - reward_mean) / max(reward_std, 1.0e-6)
    else:
        rewards_t = rewards_raw_t
    dones_t = torch.stack(dones).float()
    values_t = torch.stack(values).float()
    next_values_t = torch.stack(next_values).float()

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
        "obs_cell_features": torch.stack(obs_cell),
        "obs_ue_features": torch.stack(obs_ue),
        "obs_edge_index": torch.stack(obs_edge_idx),
        "obs_edge_attr": torch.stack(obs_edge_attr),
        "next_cell_features": torch.stack(next_cell),
        "next_ue_features": torch.stack(next_ue),
        "next_edge_attr": torch.stack(next_edge_attr),
        "action_mask_ue": torch.stack(action_mask_ue),
        "action_mask_cell_ue": torch.stack(action_mask_cell_ue),
        "action_mask_prg_cell": torch.stack(action_mask_prg),
        "target_ue_class": torch.stack(target_ue),
        "target_prg_class": torch.stack(target_prg),
        "rewards": rewards_t,
        "rewards_raw": rewards_raw_t,
        "dones": dones_t,
        "values": values_t,
        "next_values": next_values_t,
        "old_logp": torch.stack(old_logp).float(),
        "old_entropy": torch.stack(entropies).float(),
        "advantages": advantages_t,
        "returns": returns_t,
        "ue_valid_ratio": torch.stack(ue_valid_ratio).float(),
        "prg_valid_ratio": torch.stack(prg_valid_ratio).float(),
        "ue_bad_ratio": torch.stack(ue_bad_ratio).float(),
        "prg_bad_ratio": torch.stack(prg_bad_ratio).float(),
    }


def _slice_batch(buf: Dict[str, torch.Tensor], idx: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in buf.items():
        if torch.is_tensor(v) and v.shape[0] == buf["old_logp"].shape[0]:
            out[k] = v[idx].to(device)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage-B GNN policy with masked PPO on replay transitions")
    p.add_argument("--replay-dir", nargs="+", required=True, help="One or more replay directories")
    p.add_argument("--out-dir", default="training/gnnrl/checkpoints/m2_ppo", help="Output directory")
    p.add_argument("--init-policy-checkpoint", default="", help="Optional M1 actor checkpoint for warm start")
    p.add_argument("--iters", type=int, default=50, help="PPO outer iterations")
    p.add_argument("--ppo-epochs", type=int, default=6, help="PPO epochs per iteration")
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--actor-lr", type=float, default=-1.0, help="Override actor lr when > 0")
    p.add_argument("--critic-lr", type=float, default=-1.0, help="Override critic lr when > 0")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--actor-max-grad-norm", type=float, default=-1.0, help="Override actor grad norm when > 0")
    p.add_argument("--critic-max-grad-norm", type=float, default=-1.0, help="Override critic grad norm when > 0")
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--normalize-adv", type=int, choices=[0, 1], default=1)
    p.add_argument("--normalize-state", type=int, choices=[0, 1], default=1)
    p.add_argument("--normalize-reward", type=int, choices=[0, 1], default=1)
    p.add_argument("--value-loss", choices=["mse", "huber"], default="huber")
    p.add_argument("--value-huber-beta", type=float, default=10.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-cell-msg-layers", type=int, default=2)
    p.add_argument("--action-mode", default="", help="Override action mode: joint | prg_only_type0")
    p.add_argument("--seed", type=int, default=42)
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
            f"M2 currently expects n_tot_cell == n_cell, got {dims.n_tot_cell} vs {dims.n_cell}"
        )

    requested_action_mode = normalize_action_mode(args.action_mode) if args.action_mode else ""
    if args.init_policy_checkpoint:
        ckpt = torch.load(args.init_policy_checkpoint, map_location="cpu")
        cfg_dict = dict(ckpt["model_config"])
        cfg_dict["action_mode"] = requested_action_mode or normalize_action_mode(cfg_dict.get("action_mode", ACTION_MODE_JOINT))
        actor = build_model_from_config(cfg_dict)  # validates cfg compatibility implicitly via forward shapes
        actor.load_state_dict(ckpt["model_state_dict"], strict=True)
        args.action_mode = cfg_dict["action_mode"]
    else:
        args.action_mode = requested_action_mode or ACTION_MODE_JOINT
        cfg = ModelConfig(
            n_cell=dims.n_cell,
            n_active_ue=dims.n_active_ue,
            n_sched_ue=dims.n_sched_ue,
            n_prg=dims.n_prg,
            cell_feat_dim=feat.cell,
            ue_feat_dim=feat.ue,
            edge_feat_dim=feat.edge,
            hidden_dim=args.hidden_dim,
            num_cell_msg_layers=args.num_cell_msg_layers,
            action_mode=args.action_mode,
        )
        actor = StageBGnnPolicy(cfg)
        cfg_dict = actor.model_config_dict()

    critic_in_dim = feat.cell + feat.ue + feat.edge
    critic = StateValueNet(input_dim=critic_in_dim, hidden_dim=cfg_dict["hidden_dim"])

    actor.to(device)
    critic.to(device)

    actor_lr = args.actor_lr if args.actor_lr > 0 else args.lr
    critic_lr = args.critic_lr if args.critic_lr > 0 else args.lr
    actor_max_grad_norm = args.actor_max_grad_norm if args.actor_max_grad_norm > 0 else args.max_grad_norm
    critic_max_grad_norm = args.critic_max_grad_norm if args.critic_max_grad_norm > 0 else args.max_grad_norm

    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=actor_lr, weight_decay=1e-5)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=critic_lr, weight_decay=1e-5)

    norm_stats = _compute_norm_stats(dataset)
    state_mean = norm_stats.state_mean.to(device) if args.normalize_state == 1 else None
    state_std = norm_stats.state_std.to(device) if args.normalize_state == 1 else None

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    actor_best_path = out_dir / "ppo_actor_best.pt"
    actor_last_path = out_dir / "ppo_actor_last.pt"
    train_last_path = out_dir / "ppo_train_last.pt"
    summary_path = out_dir / "ppo_summary.json"

    history: List[Dict[str, float]] = []
    best_objective = -1.0e30
    best_iter = -1

    for it in range(1, args.iters + 1):
        rollout = _collect_rollout(
            dataset=dataset,
            actor=actor,
            critic=critic,
            device=device,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            normalize_adv=(args.normalize_adv == 1),
            normalize_reward=(args.normalize_reward == 1),
            state_mean=state_mean,
            state_std=state_std,
            reward_mean=norm_stats.reward_mean,
            reward_std=norm_stats.reward_std,
        )

        old_logp = rollout["old_logp"]
        advantages = rollout["advantages"]
        returns = rollout["returns"]
        n_steps = old_logp.shape[0]
        min_updates_before_early_stop = max(4, n_steps // max(1, args.minibatch_size))

        pol_loss_sum = 0.0
        val_loss_sum = 0.0
        ent_sum = 0.0
        kl_sum = 0.0
        clipfrac_sum = 0.0
        batch_updates = 0

        stop_early = False
        for _epoch in range(args.ppo_epochs):
            perm = torch.randperm(n_steps)
            for start in range(0, n_steps, args.minibatch_size):
                idx = perm[start : start + args.minibatch_size]
                if idx.numel() == 0:
                    continue

                mb = _slice_batch(rollout, idx, device=device)
                out = actor(
                    obs_cell_features=mb["obs_cell_features"],
                    obs_ue_features=mb["obs_ue_features"],
                    obs_edge_index=mb["obs_edge_index"],
                    obs_edge_attr=mb["obs_edge_attr"],
                )
                act_stat = _action_logp_entropy(
                    out,
                    mb,
                    n_cell=dims.n_cell,
                    n_prg=dims.n_prg,
                    action_mode=args.action_mode,
                )
                new_logp = act_stat["logp"]
                entropy = act_stat["entropy"]

                value_pred = critic(_state_vector(mb, state_mean=state_mean, state_std=state_std))

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
                torch.nn.utils.clip_grad_norm_(actor.parameters(), actor_max_grad_norm)
                actor_optimizer.step()

                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), critic_max_grad_norm)
                critic_optimizer.step()

                approx_kl = (old_logp_mb - new_logp).mean()
                clipfrac = ((ratio - 1.0).abs() > args.clip_eps).float().mean()

                pol_loss_sum += float(policy_loss.item())
                val_loss_sum += float(value_loss.item())
                ent_sum += float(entropy_bonus.item())
                kl_sum += float(approx_kl.item())
                clipfrac_sum += float(clipfrac.item())
                batch_updates += 1

                # Use a tolerance margin to avoid noisy one-minibatch spikes ending an iteration too early.
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

        # Proxy objective (higher is better): -policy_loss with entropy regularization.
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
            "rollout_value_mean": float(rollout["values"].mean().item()),
            "rollout_adv_mean": float(rollout["advantages"].mean().item()),
            "rollout_adv_std": float(rollout["advantages"].std(unbiased=False).item()),
            "rollout_return_mean": float(rollout["returns"].mean().item()),
            "rollout_return_std": float(rollout["returns"].std(unbiased=False).item()),
            "rollout_old_logp_mean": float(rollout["old_logp"].mean().item()),
            "rollout_entropy_mean": float(rollout["old_entropy"].mean().item()),
            "rollout_ue_valid_ratio": float(rollout["ue_valid_ratio"].mean().item()),
            "rollout_prg_valid_ratio": float(rollout["prg_valid_ratio"].mean().item()),
            "rollout_ue_bad_ratio": float(rollout["ue_bad_ratio"].mean().item()),
            "rollout_prg_bad_ratio": float(rollout["prg_bad_ratio"].mean().item()),
            "early_stop_kl": bool(stop_early),
            "num_updates": batch_updates,
        }
        history.append(row)

        torch.save(
            {
                "iter": it,
                "model_state_dict": actor.state_dict(),
                "model_config": actor.model_config_dict(),
                "source": "m2_ppo",
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
                "norm_stats": {
                    "state_mean": norm_stats.state_mean.tolist(),
                    "state_std": norm_stats.state_std.tolist(),
                    "reward_mean": norm_stats.reward_mean,
                    "reward_std": norm_stats.reward_std,
                },
                "args": vars(args),
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
                    "source": "m2_ppo",
                    "metrics": row,
                    "args": vars(args),
                },
                actor_best_path,
            )

        print(
            f"[iter {it:03d}] objective={objective:.5f} "
            f"pol_loss={mean_policy_loss:.5f} val_loss={mean_value_loss:.5f} "
            f"entropy={mean_entropy:.5f} kl={mean_kl:.5f} clipfrac={mean_clipfrac:.5f} "
            f"early_stop_kl={int(stop_early)}"
        )

    summary = {
        "status": "ok",
        "device": str(device),
        "num_records": len(dataset),
        "dataset_dims": asdict(dims),
        "dataset_feature_dims": asdict(feat),
        "model_config": actor.model_config_dict(),
        "norm_stats": {
            "state_mean": norm_stats.state_mean.tolist(),
            "state_std": norm_stats.state_std.tolist(),
            "reward_mean": norm_stats.reward_mean,
            "reward_std": norm_stats.reward_std,
        },
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

    print(f"summary written: {summary_path}")
    print(f"best actor checkpoint: {actor_best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
