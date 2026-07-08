"""Stage-B first-version sparse entity graph stack."""

from .edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from .feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from .graph_ops import (
    DecodedType0Action,
    PackedPrgCandidates,
    build_default_joint_slot_targets,
    decode_joint_actions_to_type0,
    decode_prg_choices_to_type0,
    joint_targets_from_type0_actions,
    pack_prg_candidates,
    select_and_decode_budgeted_candidate_actions,
)
from .mask_checker import GraphMaskChecker, MaskCheckReport
from .model import HGraphModelConfig, StageBHGraphPolicy, build_model_from_config
from .reward_utils import compute_stageb_v1_reward, extract_reward_metrics
from .sanity_logger import GraphSanityLogger, GraphSanityStats
from .template_encoders import CellTemplateEncoder, PrgTemplateEncoder, UeTemplateEncoder
from .teacher import GraphHeuristicTeacher, HeuristicTeacherConfig, PfqPassthroughTeacher, build_joint_teacher
from .types import EdgeSet, GraphFeatureSnapshot, NamedFeatureBlock, SparseEntityGraph, StaticMeta

__all__ = [
    "CellTemplateEncoder",
    "DecodedType0Action",
    "EdgeGeneratorConfig",
    "EdgeSet",
    "FeatureSnapshotBuilder",
    "GraphFeatureSnapshot",
    "GraphMaskChecker",
    "GraphSanityLogger",
    "GraphSanityStats",
    "GraphHeuristicTeacher",
    "HGraphModelConfig",
    "HeuristicTeacherConfig",
    "MaskCheckReport",
    "NamedFeatureBlock",
    "PackedPrgCandidates",
    "PrgTemplateEncoder",
    "SnapshotBuilderConfig",
    "SparseEdgeGenerator",
    "SparseEntityGraph",
    "StageBHGraphPolicy",
    "StaticMeta",
    "UeTemplateEncoder",
    "build_model_from_config",
    "build_default_joint_slot_targets",
    "build_joint_teacher",
    "compute_stageb_v1_reward",
    "decode_joint_actions_to_type0",
    "decode_prg_choices_to_type0",
    "extract_reward_metrics",
    "joint_targets_from_type0_actions",
    "pack_prg_candidates",
    "select_and_decode_budgeted_candidate_actions",
    "PfqPassthroughTeacher",
]
