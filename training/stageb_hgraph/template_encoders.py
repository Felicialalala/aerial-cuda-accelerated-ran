from __future__ import annotations

import torch
from torch import nn


def _flatten_last(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.dim() == 2:
        return x, x.shape[:-1]
    leading = x.shape[:-1]
    return x.reshape(-1, x.shape[-1]), leading


def _restore_last(x: torch.Tensor, leading: tuple[int, ...]) -> torch.Tensor:
    if len(leading) == 1:
        return x
    return x.reshape(*leading, x.shape[-1])


class _ResidualMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CellTemplateEncoder(nn.Module):
    """Structured cell encoder over raw bridge 5-D cell features."""

    def __init__(self, in_dim: int = 5, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        third = hidden_dim // 3
        self.load_branch = nn.Sequential(nn.Linear(2, third), nn.GELU())
        self.quality_branch = nn.Sequential(nn.Linear(2, third), nn.GELU())
        self.risk_branch = nn.Sequential(nn.Linear(1, hidden_dim - 2 * third), nn.GELU())
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.residual = _ResidualMlp(out_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat, leading = _flatten_last(x)
        load_h = self.load_branch(flat[:, [0, 1]])
        quality_h = self.quality_branch(flat[:, [2, 3]])
        risk_h = self.risk_branch(flat[:, [4]])
        h = self.proj(torch.cat([load_h, quality_h, risk_h], dim=-1))
        h = self.residual(h)
        return _restore_last(h, leading)


class UeTemplateEncoder(nn.Module):
    """Structured UE encoder over raw bridge 12-D UE features."""

    def __init__(self, in_dim: int = 12, hidden_dim: int = 96, out_dim: int = 96):
        super().__init__()
        quarter = hidden_dim // 4
        self.urgency_branch = nn.Sequential(nn.Linear(3, quarter), nn.GELU())
        self.link_branch = nn.Sequential(nn.Linear(4, quarter), nn.GELU())
        self.fairness_branch = nn.Sequential(nn.Linear(2, quarter), nn.GELU())
        self.history_branch = nn.Sequential(nn.Linear(3, hidden_dim - 3 * quarter), nn.GELU())
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.residual1 = _ResidualMlp(out_dim, hidden_dim)
        self.residual2 = _ResidualMlp(out_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat, leading = _flatten_last(x)
        urgency_h = self.urgency_branch(flat[:, [0, 8, 9]])
        link_h = self.link_branch(flat[:, [1, 2, 3, 4]])
        fairness_h = self.fairness_branch(flat[:, [10, 11]])
        history_h = self.history_branch(flat[:, [5, 6, 7]])
        h = self.proj(torch.cat([urgency_h, link_h, fairness_h, history_h], dim=-1))
        h = self.residual2(self.residual1(h))
        return _restore_last(h, leading)


class PrgTemplateEncoder(nn.Module):
    """Structured PRG encoder over raw bridge 8-D PRG features."""

    def __init__(self, in_dim: int = 8, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        third = hidden_dim // 3
        self.value_branch = nn.Sequential(nn.Linear(3, third), nn.GELU())
        self.conflict_branch = nn.Sequential(nn.Linear(3, third), nn.GELU())
        self.availability_branch = nn.Sequential(nn.Linear(2, hidden_dim - 2 * third), nn.GELU())
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.residual = _ResidualMlp(out_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat, leading = _flatten_last(x)
        value_h = self.value_branch(flat[:, [0, 1, 2]])
        conflict_h = self.conflict_branch(flat[:, [4, 5, 6]])
        availability_h = self.availability_branch(flat[:, [3, 7]])
        h = self.proj(torch.cat([value_h, conflict_h, availability_h], dim=-1))
        h = self.residual(h)
        return _restore_last(h, leading)
