from dataclasses import dataclass, field
import os
from typing import List

@dataclass
class AgentConfig:
    """AgentConfig v5.0 for ARC-AGI-3 NVIDIA AVO + CodeAct REPL Architecture."""
    
    # Kaggle runtime & budgets
    total_kaggle_budget_s: float = float(os.environ.get("ARC_TOTAL_BUDGET_S", "31500.0"))
    expected_num_games: int = int(os.environ.get("ARC_EXPECTED_GAMES", "110"))
    max_actions_per_game: int = int(os.environ.get("ARC_MAX_ACTIONS_PER_GAME", "450"))
    max_actions_per_level: int = int(os.environ.get("ARC_MAX_ACTIONS_PER_LEVEL", "120"))
    max_repl_plan_len: int = int(os.environ.get("ARC_MAX_REPL_PLAN_LEN", "18"))
    
    # NVIDIA AVO Trajectory Supervisor
    enable_avo_supervisor: bool = os.environ.get("ARC_ENABLE_SUPERVISOR", "1") == "1"
    stagnation_threshold: float = float(os.environ.get("ARC_STAGNATION_THRESHOLD", "0.70"))
    stagnation_window: int = int(os.environ.get("ARC_STAGNATION_WINDOW", "8"))
    strategy_audit_pct: float = float(os.environ.get("ARC_STRATEGY_AUDIT_PCT", "0.25"))
    max_undo_chain: int = int(os.environ.get("ARC_MAX_UNDO_CHAIN", "5"))
    enable_undo_backtrack: bool = os.environ.get("ARC_ENABLE_UNDO", "1") == "1"
    
    # Grounded Verifier Gate
    enable_verifier_gate: bool = os.environ.get("ARC_ENABLE_VERIFIER", "1") == "1"
    verifier_max_violations: int = int(os.environ.get("ARC_VERIFIER_MAX_VIOLATIONS", "3"))
    enable_action_guards: bool = os.environ.get("ARC_ENABLE_GUARDS", "1") == "1"
    enable_dead_signatures: bool = os.environ.get("ARC_ENABLE_DEAD_SIG", "1") == "1"
    
    # Memory & Compaction (arXiv:2610.26632)
    context_trim_high_tokens: int = int(os.environ.get("ARC_CTX_TRIM_HIGH", "118000"))
    context_trim_target_tokens: int = int(os.environ.get("ARC_CTX_TRIM_TARGET", "59000"))
    max_rolling_turns: int = int(os.environ.get("ARC_MAX_ROLLING_TURNS", "12"))
    enable_hud_masking: bool = os.environ.get("ARC_ENABLE_HUD_MASK", "1") == "1"
    
    # LLM & CodeAct Serving
    enable_llm: bool = os.environ.get("ARC_ENABLE_LLM", "1") == "1"
    llm_call_interval: int = int(os.environ.get("ARC_LLM_INTERVAL", "1"))
    llm_timeout_s: float = float(os.environ.get("ARC_LLM_TIMEOUT_S", "30.0"))
    llm_max_new_tokens: int = int(os.environ.get("ARC_LLM_MAX_TOKENS", "512"))
    vllm_launch_timeout_s: float = float(os.environ.get("ARC_VLLM_LAUNCH_TIMEOUT_S", "300.0"))
    vllm_gpu_memory_utilization: float = float(os.environ.get("ARC_VLLM_GPU_UTIL", "0.92"))
    speculative_draft_tokens: int = int(os.environ.get("ARC_MTP_TOKENS", "3"))
    enable_prefix_caching: bool = os.environ.get("ARC_PREFIX_CACHE", "1") == "1"
    
    # Model candidate paths (searched in order)
    vllm_base_urls: List[str] = field(
        default_factory=lambda: [
            os.environ.get("ARC_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:11434/v1",  # Local Ollama fallback
            "http://127.0.0.1:18300/v1",
            "http://127.0.0.1:30000/v1",
        ]
    )
    candidate_model_paths: List[str] = field(
        default_factory=lambda: [
            os.environ.get("ARC_LOCAL_MODEL_PATH", ""),
            "/kaggle/input/intel-qwen3-8-flash-next-w4a16-autoround",
            "/kaggle/input/models/dfranzen/intel-qwen3-8-flash-next-w4a16-autoround",
            "/kaggle/input/primitive-ai-qwen3-8-flash-next-mixed-nvfp4-fp8",
            "/kaggle/input/albucino-qwen3-8-flash-next-w4a16-fp8ple",
            "/kaggle/input/qwen3-8-flash-next-nvfp4",
            "/kaggle/input/qwen3-8-27b-fp8",
            "/kaggle/input/qwen3-6-27b-fp8",
            "/kaggle/input/gemma-4-31b",
            "/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/32b-instruct-awq/1",
            "/kaggle/input/qwen2.5-coder/transformers/32b-instruct-awq/1",
            "/kaggle/input/qwen-2.5-coder-32b-instruct-awq",
        ]
    )
    candidate_drafter_paths: List[str] = field(
        default_factory=lambda: [
            os.environ.get("ARC_LOCAL_DRAFTER_PATH", ""),
            "/kaggle/input/albucino-qwen3-8-flash-next-drafter",
            "/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter",
            "/kaggle/input/albucino-qwen3-8-flash-next-w4a16-fp8ple",
        ]
    )
