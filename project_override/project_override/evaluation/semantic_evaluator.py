"""Project Override: Evaluation Module Facade for Semantic Evaluator.
Re-exports semantic evaluation metrics from core.semantic_evaluator.
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

__all__ = [
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
]
