from dataclasses import dataclass, field
import os
from typing import List

@dataclass
class AgentConfig:
    total_kaggle_budget_s: float = float(os.environ.get("ARC_TOTAL_BUDGET_S", "31500.0"))
    expected_num_games: int = int(os.environ.get("ARC_EXPECTED_GAMES", "100"))
    max_actions_per_game: int = int(os.environ.get("ARC_MAX_ACTIONS_PER_GAME", "450"))
    max_actions_per_level: int = int(os.environ.get("ARC_MAX_ACTIONS_PER_LEVEL", "120"))
    max_repl_plan_len: int = int(os.environ.get("ARC_MAX_REPL_PLAN_LEN", "18"))
    llm_call_interval: int = int(os.environ.get("ARC_LLM_INTERVAL", "1"))
    llm_timeout_s: float = float(os.environ.get("ARC_LLM_TIMEOUT_S", "8.0"))
    llm_max_new_tokens: int = int(os.environ.get("ARC_LLM_MAX_TOKENS", "384"))
    max_rolling_turns: int = int(os.environ.get("ARC_MAX_ROLLING_TURNS", "12"))
    context_trim_high_tokens: int = int(os.environ.get("ARC_CTX_TRIM_HIGH", "118000"))
    context_trim_target_tokens: int = int(os.environ.get("ARC_CTX_TRIM_TARGET", "59000"))
    enable_hud_masking: bool = os.environ.get("ARC_ENABLE_HUD_MASK", "1") == "1"
    enable_undo_backtrack: bool = os.environ.get("ARC_ENABLE_UNDO", "1") == "1"
    enable_action_guards: bool = os.environ.get("ARC_ENABLE_GUARDS", "1") == "1"
    enable_dead_signatures: bool = os.environ.get("ARC_ENABLE_DEAD_SIG", "1") == "1"
    enable_world_model_bfs: bool = os.environ.get("ARC_ENABLE_WM_BFS", "1") == "1"
    enable_bayesian_exploration: bool = os.environ.get("ARC_ENABLE_BAYESIAN", "1") == "1"
    enable_avo_supervisor: bool = os.environ.get("ARC_ENABLE_SUPERVISOR", "1") == "1"
    enable_pattern_click_solver: bool = os.environ.get("ARC_ENABLE_CLICK_SOLVER", "1") == "1"
    enable_sokoban_planner: bool = os.environ.get("ARC_ENABLE_SOKOBAN", "1") == "1"
    enable_orthogonal_completion: bool = os.environ.get("ARC_ENABLE_ORTHO_COMP", "1") == "1"
    enable_2d_sokoban_astar: bool = os.environ.get("ARC_ENABLE_2D_SOKOBAN", "1") == "1"
    enable_iw_search: bool = os.environ.get("ARC_ENABLE_IW_SEARCH", "1") == "1"
    enable_llm: bool = os.environ.get("ARC_ENABLE_LLM", "1") == "1"
    speculative_draft_tokens: int = int(os.environ.get("ARC_MTP_TOKENS", "3"))
    vllm_base_urls: List[str] = field(
        default_factory=lambda: [
            os.environ.get("ARC_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:18300/v1",
            "http://127.0.0.1:30000/v1",
        ]
    )
    candidate_model_paths: List[str] = field(
        default_factory=lambda: [
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
            "/kaggle/input/albucino-qwen3-8-flash-next-drafter",
            "/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter",
            "/kaggle/input/albucino-qwen3-8-flash-next-w4a16-fp8ple",
        ]
    )
