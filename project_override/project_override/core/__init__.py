"""Project Override: Core Architecture Package.
Exposes:
- Semantic Evaluator (D4 Symmetry, Boundary Enclosure, Compaction, Structural Shannon Entropy, Sinkhorn Optimal Transport)
- Dead Signature (64-bit packed bitwise action encoder and O(1) null transition culling)
- Epistemic MCTS Planner (Epistemic-Damped UCT search with macro-burst generation)
"""

from project_override.core.semantic_evaluator import (
    D4SymmetryResult,
    D4SymmetryScorer,
    EnclosureResult,
    BoundaryEnclosureScorer,
    CompactionResult,
    CompactionScorer,
    StructuralEntropyResult,
    EntropyTransitionResult,
    StructuralEntropyScorer,
    OptimalTransportResult,
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
    EpistemicMCTSNode,
    EpistemicMCTSPlanner,
    PlanResult,
    simulate_grid_action,
)

from project_override.core.zero_nn_ccl import (
    BoundingBox,
    Component,
    SceneGraph,
    ZeroNNPerception,
    compute_convex_hull_2d,
    rasterize_convex_polygon,
)

from project_override.core.physics_engine import (
    Action,
    EntityType,
    Entity,
    PhysicsEvent,
    StepResult,
    InterceptResult,
    StateSnapshot,
    DiscretePhysicsEngine,
    PhysicsEngine,
    compute_deterministic_intercept,
    CPPKinematicsCore,
)

__all__ = [
    # Semantic Evaluator
    "D4SymmetryResult",
    "D4SymmetryScorer",
    "EnclosureResult",
    "BoundaryEnclosureScorer",
    "CompactionResult",
    "CompactionScorer",
    "StructuralEntropyResult",
    "EntropyTransitionResult",
    "StructuralEntropyScorer",
    "OptimalTransportResult",
    "SinkhornKnoppOptimalTransport",
    "SemanticEvaluator",
    # Dead Signature
    "ActionSignature",
    "DeadSignatureFilter",
    "pack_signature",
    "unpack_signature",
    "compute_context_hash",
    # MCTS Planner
    "DiscreteActionType",
    "EpistemicMCTSNode",
    "EpistemicMCTSPlanner",
    "PlanResult",
    "simulate_grid_action",
    # Zero-NN CCL Perception
    "BoundingBox",
    "Component",
    "SceneGraph",
    "ZeroNNPerception",
    "compute_convex_hull_2d",
    "rasterize_convex_polygon",
    # Physics Engine & Kinematics
    "Action",
    "EntityType",
    "Entity",
    "PhysicsEvent",
    "StepResult",
    "InterceptResult",
    "StateSnapshot",
    "DiscretePhysicsEngine",
    "PhysicsEngine",
    "compute_deterministic_intercept",
    "CPPKinematicsCore",
    # A* Navigation & Kinematics
    "AStarNavigator",
    "CollisionHull",
    "PathResult",
    "NavAction",
    # Symbolic Puzzle Solver
    "SymbolicPuzzleSolver",
    "ClickCandidate",
    "TransitionDelta",
    # DPA Physics Predictor
    "DPAPhysicsPredictor",
    "AutomataClass",
    "SpriteEntity",
    "InterceptSolution",
    # Universal Semantic Delta & World Model Engine
    "FunctionalRole",
    "KinematicModality",
    "EntityState",
    "FrameDelta",
    "VerificationReport",
    "ArenaBackgroundSeparator",
    "UniversalSemanticDeltaEngine",
    "ActionPipelineTracker",
    "InertiaDynamicsTracker",
    "WorldModelTransitionGraph",
    # Delta-Driven Goal Induction Engine
    "GoalCategory",
    "HypothesisStatus",
    "GoalHypothesis",
    "GoalGuidance",
    "CausalWinInvariant",
    "SolvedLevelMemory",
    "DeltaGoalInductionEngine",
    # Simulated Model Planner & Verification Delta
    "SimulatedAction",
    "SimulatedState",
    "DeadlockDetector",
    "KinematicTransitionModel",
    "EpistemicProbePlan",
    "ProbeOutcome",
    "EpistemicProber",
    "VerificationDeltaReport",
    "VerificationDeltaController",
    "SimulatedModelPlanner",
    "UniversalGF2LinearSolver",
    "UniversalBooleanSATSolver",
    "UniversalIndependentSetSolver",
    "InteractiveMatrixInducer",
    "PassiveVelocityField",
    "solve_conveyor_astar",
    "OnlineWorldModelSynthesizer",
    "MultiStrideKinematicsSimulator",
]

from project_override.core.combinatorial_engine import (
    UniversalGF2LinearSolver,
    UniversalBooleanSATSolver,
    UniversalIndependentSetSolver,
    InteractiveMatrixInducer,
)

from project_override.core.passive_conveyor_engine import (
    PassiveVelocityField,
    solve_conveyor_astar,
)

from project_override.core.online_world_model_synthesizer import (
    OnlineWorldModelSynthesizer,
    MultiStrideKinematicsSimulator,
)

from project_override.core.astar_navigation import (
    AStarNavigator,
    CollisionHull,
    PathResult,
    NavAction,
)

from project_override.core.symbolic_puzzle_solver import (
    SymbolicPuzzleSolver,
    ClickCandidate,
    TransitionDelta,
)

from project_override.core.dpa_physics_predictor import (
    DPAPhysicsPredictor,
    AutomataClass,
    SpriteEntity,
    InterceptSolution,
)

from project_override.core.world_model_engine import (
    FunctionalRole,
    KinematicModality,
    EntityState,
    FrameDelta,
    VerificationReport,
    ArenaBackgroundSeparator,
    UniversalSemanticDeltaEngine,
    ActionPipelineTracker,
    InertiaDynamicsTracker,
    WorldModelTransitionGraph,
)

from project_override.core.goal_induction_engine import (
    GoalCategory,
    HypothesisStatus,
    GoalHypothesis,
    GoalGuidance,
    CausalWinInvariant,
    SolvedLevelMemory,
    DeltaGoalInductionEngine,
)

from project_override.core.combinatorial_engine import (
    UniversalGF2LinearSolver,
    UniversalBooleanSATSolver,
    UniversalIndependentSetSolver,
    InteractiveMatrixInducer,
    SatisfactionBarInfo,
    UniversalSatisfactionBarDetector,
    UniversalSATClauseSolver,
    UniversalSubmitButtonDetector,
    UniversalOptionSelectorEngine,
)

from project_override.core.passive_conveyor_engine import (
    PassiveVelocityField,
    solve_conveyor_astar,
)

from project_override.core.online_world_model_synthesizer import (
    OnlineWorldModelSynthesizer,
    MultiStrideKinematicsSimulator,
)

