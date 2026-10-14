"""Project Override: ARC-AGI-3 Cognitive Architecture Root Package."""

from project_override.core.semantic_evaluator import (
    D4SymmetryScorer,
    BoundaryEnclosureScorer,
    CompactionScorer,
    StructuralEntropyScorer,
    SinkhornKnoppOptimalTransport,
    SemanticEvaluator,
)

from project_override.core.dead_signature import (
    ActionSignature,
    DeadSignatureFilter,
    pack_signature,
    unpack_signature,
    compute_context_hash,
)

from project_override.core.mcts_planner import (
    DiscreteActionType,
    EpistemicMCTSPlanner,
    PlanResult,
)

from project_override.core.zero_nn_ccl import (
    BoundingBox,
    Component,
    SceneGraph,
    ZeroNNPerception,
)

from project_override.core.physics_engine import (
    Action,
    EntityType,
    Entity,
    PhysicsEvent,
    StepResult,
    InterceptResult,
    DiscretePhysicsEngine,
    PhysicsEngine,
    compute_deterministic_intercept,
)

__all__ = [
    "D4SymmetryScorer",
    "BoundaryEnclosureScorer",
    "CompactionScorer",
    "StructuralEntropyScorer",
    "SinkhornKnoppOptimalTransport",
    "SemanticEvaluator",
    "ActionSignature",
    "DeadSignatureFilter",
    "pack_signature",
    "unpack_signature",
    "compute_context_hash",
    "DiscreteActionType",
    "EpistemicMCTSPlanner",
    "PlanResult",
    "BoundingBox",
    "Component",
    "SceneGraph",
    "ZeroNNPerception",
    "Action",
    "EntityType",
    "Entity",
    "PhysicsEvent",
    "StepResult",
    "InterceptResult",
    "DiscretePhysicsEngine",
    "PhysicsEngine",
    "compute_deterministic_intercept",
]
