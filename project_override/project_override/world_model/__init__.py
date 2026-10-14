"""Project Override: World Model & Context Architecture Package
Exposes AST Sandbox, Reflective World Model Engine, and Duck Harness Context Manager.
"""

from project_override.world_model.ast_sandbox import (
    ASTSecurityGate,
    ExecutionResult,
    ExecutionTimeoutError,
    SafeCompiler,
    SandboxCompilationError,
    SandboxExecutor,
    SandboxRuntimeError,
    SandboxSecurityError,
    SecurityViolationError,
    build_safe_builtins,
    build_safe_globals,
)
from project_override.world_model.world_model_engine import (
    CellMismatch,
    CompiledWorldModel,
    EntityDef,
    PriorKnowledge,
    RecordedTransition,
    ReplayVerifier,
    RuleDef,
    Rulebook,
    SolvedLevelMemory,
    SolvedLevelRecord,
    VerificationReport,
    WorldModelCompiler,
)
from project_override.world_model.context_manager import (
    BlockType,
    ContextBlock,
    DrainEvent,
    DuckContextManager,
    HelperFunction,
    HelperFunctionRegistry,
    TokenEstimator,
)

__all__ = [
    # AST Sandbox
    "ASTSecurityGate",
    "ExecutionResult",
    "ExecutionTimeoutError",
    "SafeCompiler",
    "SandboxCompilationError",
    "SandboxExecutor",
    "SandboxRuntimeError",
    "SandboxSecurityError",
    "SecurityViolationError",
    "build_safe_builtins",
    "build_safe_globals",
    # World Model Engine
    "CellMismatch",
    "CompiledWorldModel",
    "EntityDef",
    "PriorKnowledge",
    "RecordedTransition",
    "ReplayVerifier",
    "RuleDef",
    "Rulebook",
    "SolvedLevelMemory",
    "SolvedLevelRecord",
    "VerificationReport",
    "WorldModelCompiler",
    # Context Manager
    "BlockType",
    "ContextBlock",
    "DrainEvent",
    "DuckContextManager",
    "HelperFunction",
    "HelperFunctionRegistry",
    "TokenEstimator",
]
