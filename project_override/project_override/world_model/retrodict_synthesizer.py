"""
project_override/world_model/retrodict_synthesizer.py
Tier-2 Retrodiction-Gated LLM World-Model Synthesizer in AST Sandbox.
Implements:
1. Automatic Offline Hardware Detection (RTX PRO 6000 vLLM FP8 -> Dual-T4 llama.cpp GGUF -> Local Fallback).
2. Tycho 4-Function Rendered Moore Machine Synthesis Contract (init_state, transition, render, outcome).
3. Entity-Centric Zero-Action Retrodiction over full interaction history H_{0:t}.
4. CEGIS Counterexample Diff Repair & Aguilar Martin Play-Adequacy Verification.
"""

from __future__ import annotations
import ast
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


IMMUTABLE_SYSTEM_PREFIX = """You are the Project Override v2 Neurosymbolic World-Model Compiler for ARC-AGI-3, powered by Qwen 2.5 Coder 32B Instruct.
You NEVER emit raw step-by-step actions. Instead, you synthesize a deterministic Python class `SynthesizedWorldModel`
that models the environment as a Rendered Deterministic Moore Machine:
    class SynthesizedWorldModel:
        def init_state(self, grid: np.ndarray, hud: dict, latent: np.ndarray) -> dict: ...
        def transition(self, state: dict, action: int, click_xy: tuple = None) -> dict: ...
        def render(self, state: dict) -> np.ndarray: ...
        def outcome(self, state: dict) -> str: # Returns "ONGOING", "LEVEL_CLEAR", or "GAME_OVER"

Rules:
1. Write pure, deterministic Python using only `numpy as np` and standard built-in types.
2. Never hardcode game IDs or fixed level coordinates; express rules in terms of component colors, relative displacements, ray-casts, cellular neighborhoods, or HUD state registers.
3. Your code will be tested immediately via Entity-Centric Retrodiction against all past transitions.
"""


@dataclass
class HistoricalTransition:
    step_idx: int
    pre_grid: np.ndarray
    pre_hud: Dict[str, int]
    pre_latent: np.ndarray
    action_id: int
    click_xy: Optional[Tuple[int, int]]
    post_grid: np.ndarray
    post_hud: Dict[str, int]
    timer_mask: np.ndarray
    level_cleared: bool
    game_over: bool


@dataclass
class RetrodictionReport:
    passed_all: bool
    mean_jaccard: float
    failed_step: Optional[int] = None
    counterexample_diff: str = ""


class ASTWorldModelSandbox:
    """Executes synthesized Python world-model classes safely with strict AST node whitelisting."""

    FORBIDDEN_NAMES = {
        "open", "exec", "eval", "compile", "__import__", "os", "sys",
        "subprocess", "socket", "pathlib", "shutil", "globals", "locals",
    }

    @classmethod
    def validate_and_compile(cls, code_str: str) -> Optional[Any]:
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name not in ("numpy", "math", "collections", "itertools", "copy"):
                            return None
                elif isinstance(node, ast.Name) and node.id in cls.FORBIDDEN_NAMES:
                    return None
                elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                    if node.attr not in ("__init__", "__len__"):
                        return None

            import builtins
            safe_globals: Dict[str, Any] = {
                "__name__": "__sandbox__",
                "__builtins__": {
                    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
                    "enumerate": enumerate, "filter": filter, "float": float, "frozenset": frozenset,
                    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
                    "max": max, "min": min, "range": range, "round": round,
                    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
                    "__build_class__": builtins.__build_class__,
                },
                "np": np,
            }
            local_ns: Dict[str, Any] = {}
            exec(compile(tree, filename="<ast_sandbox>", mode="exec"), safe_globals, local_ns)
            wm_cls = local_ns.get("SynthesizedWorldModel")
            if wm_cls is not None:
                return wm_cls()
        except Exception:
            return None
        return None


class OfflineLLMRuntime:
    """
    Hardware-aware offline & local inference bridge supporting:
    - Local Testing & Development: Qwen 2.5 Coder 7B Instruct
      * Local HTTP Server: Ollama (http://localhost:11434), LM Studio (http://localhost:1234/v1), vLLM / llama-server
      * In-Process: HuggingFace Transformers + PyTorch (Apple Silicon MPS / Metal or CPU)
      * In-Process: llama.cpp GGUF Q4_K_M
    - Kaggle Competition Evaluation: Qwen 2.5 Coder 32B Instruct
      * Tier A (NVIDIA RTX PRO 6000, >= 36 GB VRAM): vLLM AWQ / FP8
      * Tier B (Dual T4, 2x16 GB = 32 GB VRAM): llama.cpp GGUF layer-splitting (-ts 0.5,0.5) or vLLM tensor_parallel_size=2
    - Offline & Test Fallback: Rule-Template Programmatic Inductive Synthesizer.
    """

    CANDIDATE_MODEL_PATHS: List[str] = [
        # Local 7B Paths
        "./models/qwen2.5-coder-7b-instruct",
        "./models/qwen-2.5-coder-7b",
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct"),
        "/kaggle/input/qwen-2-5-coder-7b-instruct",
        "/kaggle/input/qwen2-5-coder-7b-instruct-awq",
        "/kaggle/input/qwen2-5-coder-7b-instruct-gguf",
        # Kaggle Models API mount paths (format: /kaggle/input/<owner>/<model>/<framework>/<variant>/<version>)
        # qwen-lm/qwen2.5-coder/transformers/32b-instruct-awq/1  <- exact model_sources entry in kernel-metadata.json
        "/kaggle/input/qwen-lm/qwen2.5-coder/transformers/32b-instruct-awq/1",
        "/kaggle/input/qwen-lm/qwen2.5-coder/transformers/32b-instruct-awq/2",
        "/kaggle/input/qwen-lm/qwen2.5-coder/transformers/32b-instruct-awq/3",
        "/kaggle/input/qwen-lm/qwen2.5-coder/transformers/32b-instruct-gguf/1",
        "/kaggle/input/qwen-lm/qwen2.5-coder/transformers/32b-instruct-gguf/2",
        "/kaggle/input/qwen-lm/qwen2.5-coder/transformers/7b-instruct-awq/1",
        "/kaggle/input/qwen-lm/qwen2.5-coder/transformers/7b-instruct/1",
        # Legacy /kaggle/input flat dataset paths
        "/kaggle/input/qwen-2-5-coder-32b-instruct-awq",
        "/kaggle/input/qwen2-5-coder-32b-instruct-awq",
        "/kaggle/input/qwen-2-5-coder-32b-instruct-gguf",
        "/kaggle/input/qwen2-5-coder-32b-instruct-gguf",
        "/kaggle/input/qwen-2-5-coder-32b-instruct",
        "/kaggle/input/arc3-model",
    ]

    DEFAULT_LOCAL_HTTP_URLS: List[str] = [
        "http://localhost:11434",
        "http://localhost:1234/v1",
        "http://localhost:8000/v1",
        "http://127.0.0.1:11434",
    ]

    def __init__(
        self,
        model_input_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        model_size: Optional[str] = None,
        local_server_url: Optional[str] = None,
    ) -> None:
        self.is_kaggle = (
            os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "false").lower() == "true"
            or os.path.exists("/kaggle")
        )
        # Default to 7B locally for fast local testing, 32B in Kaggle offline rerun
        env_size = os.environ.get("ARC_MODEL_SIZE")
        self.model_size = (
            model_size
            or env_size
            or ("32B" if self.is_kaggle else "7B")
        ).upper()

        default_name = (
            "Qwen/Qwen2.5-Coder-32B-Instruct"
            if self.model_size == "32B"
            else "Qwen/Qwen2.5-Coder-7B-Instruct"
        )
        self.model_name = (
            model_name
            or os.environ.get("ARC_MODEL_NAME")
            or default_name
        )

        self.model_input_dir = self._resolve_model_path(model_input_dir)
        self.local_server_url = local_server_url or os.environ.get("LOCAL_LLM_URL")
        self.backend_type: str = "TEMPLATE_SYNTH"
        self.engine: Any = None
        self.tokenizer: Any = None
        self._initialize_hardware_backend()

    def _resolve_model_path(self, explicit_path: Optional[str]) -> Optional[str]:
        if explicit_path and os.path.exists(explicit_path):
            return explicit_path
        env_dir = os.environ.get("ARC_MODEL_DIR")
        if env_dir and os.path.exists(env_dir):
            return env_dir
        for p in self.CANDIDATE_MODEL_PATHS:
            if p and os.path.exists(p):
                return p
        # Fast targeted scan: Kaggle Models API mounts at /kaggle/input/<owner>/<model>/<framework>/<variant>/<version>
        # Depth-limited — avoids scanning thousands of game frame files in competition datasets
        if os.path.exists("/kaggle/input"):
            ki = "/kaggle/input"
            try:
                for l1 in os.listdir(ki):                          # owner
                    p1 = os.path.join(ki, l1)
                    if not os.path.isdir(p1):
                        continue
                    if not any(k in l1.lower() for k in ("qwen", "arc3", "model")):
                        continue
                    for l2 in os.listdir(p1):                      # model-name
                        p2 = os.path.join(p1, l2)
                        if not os.path.isdir(p2):
                            continue
                        for l3 in os.listdir(p2):                  # framework
                            p3 = os.path.join(p2, l3)
                            if not os.path.isdir(p3):
                                continue
                            for l4 in os.listdir(p3):              # variant
                                p4 = os.path.join(p3, l4)
                                if not os.path.isdir(p4):
                                    continue
                                for l5 in os.listdir(p4):          # version
                                    p5 = os.path.join(p4, l5)
                                    if os.path.isdir(p5) and os.path.exists(os.path.join(p5, "config.json")):
                                        return p5
                                # Also check flat variant dirs (version may be the leaf)
                                if os.path.exists(os.path.join(p4, "config.json")):
                                    return p4
                                for f in os.listdir(p4):
                                    if f.endswith(".gguf"):
                                        return p4
            except Exception:
                pass
            # Flat /kaggle/input/<slug> fallback (legacy dataset format)
            try:
                for slug in os.listdir(ki):
                    p = os.path.join(ki, slug)
                    if os.path.isdir(p) and any(k in slug.lower() for k in ("qwen", "32b", "arc3")):
                        if os.path.exists(os.path.join(p, "config.json")):
                            return p
                        for f in os.listdir(p):
                            if f.endswith(".gguf") and ("32b" in f.lower() or "qwen" in f.lower()):
                                return p
            except Exception:
                pass
        return explicit_path

    @staticmethod
    def _ping_server(url: str) -> bool:
        try:
            import urllib.request
            probe = url.rstrip("/")
            if probe.endswith("/v1"):
                test_endpoint = f"{probe}/models"
            else:
                test_endpoint = f"{probe}/api/tags" if "11434" in probe else f"{probe}/v1/models"
            req = urllib.request.Request(test_endpoint, headers={"User-Agent": "ARC-Agent"})
            with urllib.request.urlopen(req, timeout=0.3) as resp:
                return resp.status in (200, 204)
        except Exception:
            return False

    def _initialize_hardware_backend(self) -> None:
        # 1. Local HTTP API server check (Ollama / LM Studio / LocalAI / vLLM API)
        if not self.local_server_url:
            for probe_url in self.DEFAULT_LOCAL_HTTP_URLS:
                if self._ping_server(probe_url):
                    self.local_server_url = probe_url
                    break

        if self.local_server_url:
            self.backend_type = "LOCAL_HTTP_SERVER"
            return

        # 2. Kaggle CUDA Accelerator check (RTX 6000 or Dual T4)
        try:
            import torch
            if torch.cuda.is_available() and self.model_input_dir and os.path.exists(self.model_input_dir):
                num_gpus = torch.cuda.device_count()
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                m_path_lower = self.model_input_dir.lower()

                # High-VRAM single GPU (RTX 6000, A100 >= 36 GB VRAM)
                if vram_gb >= 36.0:
                    try:
                        from vllm import LLM
                        quant = "awq" if "awq" in m_path_lower else ("fp8" if "fp8" in m_path_lower else None)
                        self.engine = LLM(
                            model=self.model_input_dir,
                            quantization=quant,
                            kv_cache_dtype="fp8" if quant == "fp8" else "auto",
                            enable_prefix_caching=True,
                            max_model_len=4096,
                            gpu_memory_utilization=0.90,
                            enforce_eager=False,
                            trust_remote_code=True,
                        )
                        self.backend_type = f"VLLM_RTX6000_{quant.upper() if quant else 'BF16'}"
                        return
                    except Exception:
                        pass

                # Dual T4 GPUs (2x16 GB = 32 GB VRAM)
                elif num_gpus >= 2:
                    gguf_files = [f for f in os.listdir(self.model_input_dir) if f.endswith(".gguf")]
                    if gguf_files:
                        try:
                            from llama_cpp import Llama
                            self.engine = Llama(
                                model_path=os.path.join(self.model_input_dir, gguf_files[0]),
                                n_gpu_layers=99,
                                split_mode=1,
                                tensor_split=[0.5, 0.5],
                                n_ctx=4096,
                                use_mmap=True,
                                verbose=False,
                            )
                            self.backend_type = "LLAMACPP_DUAL_T4_GGUF"
                            return
                        except Exception:
                            pass
                    else:
                        try:
                            from vllm import LLM
                            quant = "awq" if "awq" in m_path_lower else ("fp8" if "fp8" in m_path_lower else None)
                            self.engine = LLM(
                                model=self.model_input_dir,
                                tensor_parallel_size=2,
                                quantization=quant,
                                max_model_len=4096,
                                gpu_memory_utilization=0.88,
                                enforce_eager=False,
                                trust_remote_code=True,
                            )
                            self.backend_type = f"VLLM_DUAL_T4_{quant.upper() if quant else 'AUTO'}"
                            return
                        except Exception:
                            pass

                # Native Transformers CUDA loader (Works on both RTX 6000 and Dual T4 with auto device_map)
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_input_dir, trust_remote_code=True)
                    self.engine = AutoModelForCausalLM.from_pretrained(
                        self.model_input_dir,
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    self.backend_type = "TRANSFORMERS_CUDA_32B"
                    return
                except Exception:
                    pass
        except Exception:
            pass

        # 3. Local In-Process PyTorch/MPS Transformers (Apple Silicon or CPU for 7B)
        if self.model_size == "7B" or not self.is_kaggle:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                should_load = bool(
                    (self.model_input_dir and os.path.exists(self.model_input_dir))
                    or os.environ.get("ARC_LOAD_LOCAL_WEIGHTS", "false").lower() == "true"
                )
                if should_load:
                    load_target = (
                        self.model_input_dir
                        if (self.model_input_dir and os.path.exists(self.model_input_dir))
                        else self.model_name
                    )
                    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                    device = "mps" if has_mps else ("cuda" if torch.cuda.is_available() else "cpu")
                    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

                    self.tokenizer = AutoTokenizer.from_pretrained(load_target, trust_remote_code=True)
                    self.engine = AutoModelForCausalLM.from_pretrained(
                        load_target,
                        torch_dtype=dtype,
                        device_map={"": device} if device == "mps" else "auto",
                        trust_remote_code=True,
                    )
                    self.backend_type = f"TRANSFORMERS_7B_{device.upper()}"
                    return
            except Exception:
                pass

        # 4. Fallback to Inductive Programmatic Synthesizer
        self.backend_type = "TEMPLATE_SYNTH"

    def _call_local_http_api(self, prompt: str, attempt_idx: int = 0) -> str:
        """Queries local Ollama or OpenAI-compatible server (LM Studio / vLLM / llama-server)."""
        import json
        import urllib.request
        server = (self.local_server_url or "http://localhost:11434").rstrip("/")
        temperature = 0.0 if attempt_idx == 0 else 0.2

        try:
            # Case 1: Ollama native generate endpoint
            if "11434" in server or not server.endswith("/v1"):
                ollama_url = f"{server}/api/generate"
                model_tag = "qwen2.5-coder:7b" if "7b" in self.model_name.lower() else "qwen2.5-coder:32b"
                payload = {
                    "model": model_tag,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": 512},
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    ollama_url, data=data, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    res_obj = json.loads(resp.read().decode("utf-8"))
                    return res_obj.get("response", "")

            # Case 2: OpenAI-compatible completions endpoint (/v1/chat/completions)
            api_url = f"{server}/chat/completions" if server.endswith("/v1") else f"{server}/v1/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": IMMUTABLE_SYSTEM_PREFIX},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": 512,
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                api_url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                res_obj = json.loads(resp.read().decode("utf-8"))
                return res_obj["choices"][0]["message"]["content"]
        except Exception:
            return ""

    def _call_transformers_7b(self, prompt: str, attempt_idx: int = 0) -> str:
        """In-process inference using PyTorch / MPS / CUDA Transformers."""
        try:
            import torch
            target_device = (
                "cuda:0"
                if torch.cuda.is_available()
                else getattr(self.engine, "device", "cpu")
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(target_device)
            with torch.no_grad():
                outputs = self.engine.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=(attempt_idx > 0),
                    temperature=0.2 if attempt_idx > 0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            return generated
        except Exception:
            return ""

    def synthesize_world_model_code(
        self,
        history: List[HistoricalTransition],
        counterexample_feedback: str = "",
        attempt_idx: int = 0,
    ) -> str:
        """Generates candidate Python code for SynthesizedWorldModel using 7B (local) or 32B (Kaggle)."""
        prompt = self._build_compact_prompt(history, counterexample_feedback)

        # 1. Local HTTP API Server (Ollama / LM Studio)
        if self.backend_type == "LOCAL_HTTP_SERVER":
            raw_text = self._call_local_http_api(prompt, attempt_idx)
            if raw_text:
                return self._extract_code_block(raw_text)

        # 2. Local In-Process Transformers (7B with MPS / CPU)
        if self.backend_type.startswith("TRANSFORMERS") and self.engine is not None:
            raw_text = self._call_transformers_7b(prompt, attempt_idx)
            if raw_text:
                return self._extract_code_block(raw_text)

        # 3. vLLM (RTX 6000 / Dual T4)
        if self.backend_type.startswith("VLLM") and self.engine is not None:
            from vllm import SamplingParams
            params = SamplingParams(
                temperature=0.0 if attempt_idx == 0 else 0.2,
                top_p=0.95,
                max_tokens=512,
                seed=42 + attempt_idx,
                stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
            )
            outputs = self.engine.generate([prompt], sampling_params=params)
            return self._extract_code_block(outputs[0].outputs[0].text)

        # 4. llama.cpp GGUF
        if self.backend_type.startswith("LLAMACPP") and self.engine is not None:
            out = self.engine(
                prompt,
                max_tokens=512,
                temperature=0.0 if attempt_idx == 0 else 0.2,
                stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
                seed=42 + attempt_idx,
            )
            return self._extract_code_block(out["choices"][0]["text"])

        # 5. Programmatic Inductive Synthesis Fallback
        return self._programmatic_inductive_synthesis(history)

    def _build_compact_prompt(
        self, history: List[HistoricalTransition], counterexample_feedback: str
    ) -> str:
        """Constructs a compact ChatML prompt strictly formatted for Qwen 2.5 Coder 32B Instruct."""
        diffs_summary: List[str] = []
        for h in history[-8:]:
            mask = (h.post_grid != h.pre_grid) & (~h.timer_mask)
            coords = np.argwhere(mask)
            changes = [
                f"({int(r)},{int(c)}):{int(h.pre_grid[r,c])}->{int(h.post_grid[r,c])}"
                for r, c in coords[:16]
            ]
            diffs_summary.append(
                f"Step {h.step_idx} Act={h.action_id} Click={h.click_xy} HUD={h.pre_hud}->{h.post_hud} "
                f"Clear={h.level_cleared} Diffs={changes}"
            )
        body = "\n".join(diffs_summary)
        user_msg = (
            f"# Observed Transitions:\n{body}\n\n"
            f"# Counterexample Feedback:\n{counterexample_feedback}\n\n"
            "Synthesize the deterministic Python class `SynthesizedWorldModel` enclosed in ```python ... ```."
        )

        return (
            f"<|im_start|>system\n{IMMUTABLE_SYSTEM_PREFIX}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            "<|im_start|>assistant\n```python\n"
        )

    @staticmethod
    def _extract_code_block(text: str) -> str:
        if "```python" in text:
            text = text.split("```python", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
        return text.strip()

    @staticmethod
    def _programmatic_inductive_synthesis(history: List[HistoricalTransition]) -> str:
        """
        Synthesizes an executable empirical Moore Machine directly from observed transition deltas
        when running in zero-GPU local validation mode.
        """
        learned_deltas: Dict[int, Tuple[int, int]] = {}
        wall_cols: Set[int] = set()
        avatar_col: int = -1

        for h in history:
            if h.action_id in (1, 2, 3, 4):
                diff = (h.post_grid != h.pre_grid) & (~h.timer_mask)
                if np.any(diff):
                    bg_col = int(np.bincount(h.pre_grid.ravel()).argmax())
                    for col in range(16):
                        if col == bg_col:
                            continue
                        p0 = np.argwhere((h.pre_grid == col) & diff)
                        p1 = np.argwhere((h.post_grid == col) & diff)
                        if len(p0) > 0 and len(p0) == len(p1):
                            dr = int(round(p1[:, 0].mean() - p0[:, 0].mean()))
                            dc = int(round(p1[:, 1].mean() - p0[:, 1].mean()))
                            if abs(dr) + abs(dc) > 0:
                                avatar_col = col
                                learned_deltas[h.action_id] = (dr, dc)
                                break

        return f"""
class SynthesizedWorldModel:
    def __init__(self):
        self.avatar_col = {avatar_col}
        self.deltas = {repr(learned_deltas)}
        self.walls = set({list(wall_cols)})

    def init_state(self, grid, hud, latent):
        return {{"grid": grid.copy(), "hud": dict(hud), "latent": latent.copy(), "cleared": False}}

    def transition(self, state, action, click_xy=None):
        g = state["grid"].copy()
        if action in self.deltas and self.avatar_col >= 0:
            dr, dc = self.deltas[action]
            pts = np.argwhere(g == self.avatar_col)
            if len(pts) > 0:
                can_move = True
                for r, c in pts:
                    nr, nc = int(r + dr), int(c + dc)
                    if not (0 <= nr < 64 and 0 <= nc < 64) or int(g[nr, nc]) in self.walls:
                        can_move = False
                        break
                if can_move:
                    bg = int(np.bincount(g.ravel()).argmax())
                    for r, c in pts:
                        g[r, c] = bg
                    for r, c in pts:
                        g[int(r + dr), int(c + dc)] = self.avatar_col
        return {{"grid": g, "hud": dict(state["hud"]), "latent": state["latent"].copy(), "cleared": False}}

    def render(self, state):
        return state["grid"]

    def outcome(self, state):
        return "LEVEL_CLEAR" if state.get("cleared", False) else "ONGOING"
"""


class RetrodictWorldModelSynthesizer:
    """
    Executes the Tier-2 CEGIS + Entity-Centric Zero-Action Retrodiction loop.
    Guarantees that no synthesized WorldModel is trusted unless it reproduces 100% of changed
    entity pixels across H_{0:t}.
    """

    def __init__(self, llm_runtime: Optional[OfflineLLMRuntime] = None) -> None:
        self.llm = llm_runtime or OfflineLLMRuntime()
        self.verified_model: Optional[Any] = None
        self.verified_code: str = ""

    @staticmethod
    def verify_retrodiction(
        candidate_model: Any, history: List[HistoricalTransition]
    ) -> RetrodictionReport:
        """
        Entity-Centric Diff Verification (Preventing Kepler's False Transition Verification Trap):
        Evaluates Jaccard similarity strictly over changed non-timer pixels (F_{k+1} ^ F_k).
        """
        if not history:
            return RetrodictionReport(passed_all=True, mean_jaccard=1.0)

        jaccards: List[float] = []
        for h in history:
            try:
                st = candidate_model.init_state(h.pre_grid, h.pre_hud, h.pre_latent)
                nxt_st = candidate_model.transition(st, h.action_id, h.click_xy)
                pred_grid = candidate_model.render(nxt_st)
            except Exception as exc:
                return RetrodictionReport(
                    passed_all=False,
                    mean_jaccard=0.0,
                    failed_step=h.step_idx,
                    counterexample_diff=f"Runtime exception at step {h.step_idx}: {exc}",
                )

            actual_diff_mask = (h.post_grid != h.pre_grid) & (~h.timer_mask)
            pred_diff_mask = (pred_grid != h.pre_grid) & (~h.timer_mask)

            actual_set = {
                (int(r), int(c), int(h.post_grid[r, c]))
                for r, c in np.argwhere(actual_diff_mask)
            }
            pred_set = {
                (int(r), int(c), int(pred_grid[r, c]))
                for r, c in np.argwhere(pred_diff_mask)
            }

            if not actual_set and not pred_set:
                jaccards.append(1.0)
            else:
                inter = len(actual_set & pred_set)
                union = max(1, len(actual_set | pred_set))
                j = float(inter) / float(union)
                jaccards.append(j)
                if j < 1.0:
                    missing = list(actual_set - pred_set)[:8]
                    spurious = list(pred_set - actual_set)[:8]
                    return RetrodictionReport(
                        passed_all=False,
                        mean_jaccard=float(np.mean(jaccards)),
                        failed_step=h.step_idx,
                        counterexample_diff=(
                            f"Step {h.step_idx} (action={h.action_id}): "
                            f"Missing actual changes={missing}, Spurious predicted changes={spurious}"
                        ),
                    )

        return RetrodictionReport(passed_all=True, mean_jaccard=1.0)

    def synthesize_and_verify(
        self,
        history: List[HistoricalTransition],
        max_cegis_iters: int = 2,
        max_total_seconds: float = 2.0,
    ) -> Optional[Any]:
        import time
        start_t = time.perf_counter()
        feedback = ""
        for attempt in range(max_cegis_iters):
            if time.perf_counter() - start_t > max_total_seconds:
                break
            code = self.llm.synthesize_world_model_code(history, feedback, attempt)
            model = ASTWorldModelSandbox.validate_and_compile(code)
            if model is None:
                feedback = "Syntax or AST Sandbox validation error. Ensure class SynthesizedWorldModel is valid."
                continue
            report = self.verify_retrodiction(model, history)
            if report.passed_all:
                self.verified_model = model
                self.verified_code = code
                return model
            feedback = report.counterexample_diff
        return None
