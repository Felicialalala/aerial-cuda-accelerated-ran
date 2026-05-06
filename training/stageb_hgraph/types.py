from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class NamedFeatureBlock:
    values: torch.Tensor
    names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.dim() != 2:
            raise ValueError(f"feature block must be rank-2 [N,F], got shape={tuple(self.values.shape)}")
        if self.values.shape[1] != len(self.names):
            raise ValueError(
                f"feature/name mismatch: shape={tuple(self.values.shape)} names={len(self.names)}"
            )

    @property
    def n_items(self) -> int:
        return int(self.values.shape[0])

    @property
    def dim(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class StaticMeta:
    n_cell: int
    n_ue: int
    n_sched_ue: int
    n_prg: int
    topology: str
    action_mask_cell_ue: torch.Tensor
    action_mask_prg_cell: torch.Tensor
    cell_adjacency: torch.Tensor
    cell_edge_attr: Optional[torch.Tensor]
    ue_serving_cell: torch.Tensor
    prg_owner_cell: torch.Tensor
    prg_local_index: torch.Tensor
    action_mask_ue: Optional[torch.Tensor] = None
    blank_action_always_allowed: bool = True
    post_eq_sinr: Optional[torch.Tensor] = None
    teacher_action_ue: Optional[torch.Tensor] = None
    teacher_action_prg: Optional[torch.Tensor] = None
    candidate_mode: str = "shared_cell_pool"

    def __post_init__(self) -> None:
        if self.action_mask_cell_ue.shape != (self.n_cell, self.n_ue):
            raise ValueError(
                "action_mask_cell_ue shape mismatch: "
                f"got={tuple(self.action_mask_cell_ue.shape)} expected={(self.n_cell, self.n_ue)}"
            )
        if self.action_mask_prg_cell.shape != (self.n_cell, self.n_prg):
            raise ValueError(
                "action_mask_prg_cell shape mismatch: "
                f"got={tuple(self.action_mask_prg_cell.shape)} expected={(self.n_cell, self.n_prg)}"
            )
        if self.action_mask_ue is not None and self.action_mask_ue.shape != (self.n_ue,):
            raise ValueError(
                "action_mask_ue shape mismatch: "
                f"got={tuple(self.action_mask_ue.shape)} expected={(self.n_ue,)}"
            )
        if self.teacher_action_ue is not None and self.teacher_action_ue.shape != (self.n_sched_ue,):
            raise ValueError(
                "teacher_action_ue shape mismatch: "
                f"got={tuple(self.teacher_action_ue.shape)} expected={(self.n_sched_ue,)}"
            )
        if self.teacher_action_prg is not None and self.teacher_action_prg.shape != (self.n_prg * self.n_cell,):
            raise ValueError(
                "teacher_action_prg shape mismatch: "
                f"got={tuple(self.teacher_action_prg.shape)} expected={(self.n_prg * self.n_cell,)}"
            )
        if self.cell_adjacency.dim() != 2 or self.cell_adjacency.shape[1] != 2:
            raise ValueError(
                f"cell_adjacency must be [E,2], got shape={tuple(self.cell_adjacency.shape)}"
            )
        if self.cell_edge_attr is not None:
            if self.cell_edge_attr.dim() != 2:
                raise ValueError(
                    f"cell_edge_attr must be [E,F], got shape={tuple(self.cell_edge_attr.shape)}"
                )
            if self.cell_edge_attr.shape[0] != self.cell_adjacency.shape[0]:
                raise ValueError(
                    "cell_edge_attr edge count mismatch: "
                    f"edge_attr={tuple(self.cell_edge_attr.shape)} cell_adjacency={tuple(self.cell_adjacency.shape)}"
                )
        if self.ue_serving_cell.shape != (self.n_ue,):
            raise ValueError(
                f"ue_serving_cell must be [{self.n_ue}], got shape={tuple(self.ue_serving_cell.shape)}"
            )
        n_prg_tokens = self.n_cell * self.n_prg
        if self.prg_owner_cell.shape != (n_prg_tokens,):
            raise ValueError(
                f"prg_owner_cell must be [{n_prg_tokens}], got shape={tuple(self.prg_owner_cell.shape)}"
            )
        if self.prg_local_index.shape != (n_prg_tokens,):
            raise ValueError(
                f"prg_local_index must be [{n_prg_tokens}], got shape={tuple(self.prg_local_index.shape)}"
            )


@dataclass(frozen=True)
class GraphFeatureSnapshot:
    cell_features: NamedFeatureBlock
    ue_features: NamedFeatureBlock
    prg_features: NamedFeatureBlock
    static_meta: StaticMeta

    @property
    def n_cell(self) -> int:
        return self.static_meta.n_cell

    @property
    def n_ue(self) -> int:
        return self.static_meta.n_ue

    @property
    def n_prg(self) -> int:
        return self.static_meta.n_prg

    @property
    def n_prg_tokens(self) -> int:
        return self.static_meta.n_cell * self.static_meta.n_prg


@dataclass(frozen=True)
class EdgeSet:
    name: str
    src_type: str
    dst_type: str
    index: torch.Tensor
    attr: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.index.dim() != 2 or self.index.shape[0] != 2:
            raise ValueError(f"edge index must be [2,E], got shape={tuple(self.index.shape)}")
        if self.attr is not None and self.attr.shape[0] != self.index.shape[1]:
            raise ValueError(
                f"edge attr count mismatch for {self.name}: edges={self.index.shape[1]} attr={self.attr.shape[0]}"
            )

    @property
    def num_edges(self) -> int:
        return int(self.index.shape[1])


@dataclass
class SparseEntityGraph:
    snapshot: GraphFeatureSnapshot
    edges: Dict[str, EdgeSet]
    cell_embeddings: Optional[torch.Tensor] = None
    ue_embeddings: Optional[torch.Tensor] = None
    prg_embeddings: Optional[torch.Tensor] = None
    stats: Dict[str, float] = field(default_factory=dict)
