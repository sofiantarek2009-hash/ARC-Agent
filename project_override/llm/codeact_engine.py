import ast
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
from project_override.config import AgentConfig

logger = logging.getLogger(__name__)


def parse_codeact_response(text: str) -> Optional[Dict[str, Any]]:
    """Robust parser for LLM CodeAct responses.
    
    Supports:
    1. Fenced ```python ... ``` blocks (Duck / CodeAct style)
    2. Raw JSON objects ({"thought": "...", "subgoal": "ACTION", "action_id": 1})
    """
    if not text:
        return None
    
    py_match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if py_match:
        code_body = py_match.group(1).strip()
        if code_body:
            thought = text[: py_match.start()].strip()
            return {
                "subgoal": "CODEACT",
                "code": code_body,
                "thought": thought[:240],
            }

    # JSON fallback
    dec = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


class OfflineServerManager:
    """Manages offline vLLM / SGLang OpenAI-compatible HTTP inference server.
    
    Solves the root-cause "Ghost Engine" bug:
    Probes running endpoints (e.g. Kaggle background vLLM or local Ollama),
    and if none exists, launches vLLM via subprocess.Popen and blocks until
    http://127.0.0.1:8000/v1/models is ready.
    """
    _INSTANCE: Optional["OfflineServerManager"] = None
    _LOCK = threading.Lock()

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        os.environ.setdefault("VLLM_GDN_DECODE_KERNEL", "triton")
        self.http_endpoint: Optional[str] = None
        self.http_model_name: str = "default"
        self.process: Optional[subprocess.Popen] = None
        self.is_ready: bool = False
        self.launch_attempted: bool = False
        self.total_calls: int = 0
        self.fallback_count: int = 0

    @classmethod
    def get_instance(cls, config: Optional[AgentConfig] = None) -> "OfflineServerManager":
        if cls._INSTANCE is None:
            with cls._LOCK:
                if cls._INSTANCE is None:
                    cls._INSTANCE = cls(config=config)
        return cls._INSTANCE

    def _probe_running_server(self) -> bool:
        """Checks if a vLLM/SGLang/Ollama server is already responding."""
        for base_url in self.config.vllm_base_urls:
            if not base_url:
                continue
            try:
                probe_url = f"{base_url}/models"
                req = urllib.request.Request(probe_url, headers={"User-Agent": "ARC-Agent"})
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        models = data.get("data", [])
                        if models:
                            self.http_model_name = models[0].get("id", "default")
                        self.http_endpoint = f"{base_url}/chat/completions"
                        self.is_ready = True
                        logger.info(f"[OfflineServerManager] Connected to active endpoint at {self.http_endpoint} (model: {self.http_model_name})")
                        return True
            except Exception:
                continue
        return False

    def _resolve_model_path(self) -> Optional[str]:
        env_p = os.environ.get("ARC_LOCAL_MODEL_PATH")
        if env_p and Path(env_p).exists():
            return env_p
        for p in self.config.candidate_model_paths:
            if p and Path(p).exists():
                return p
        kaggle_in = Path("/kaggle/input")
        if kaggle_in.exists():
            for cfg in kaggle_in.rglob("config.json"):
                low = str(cfg).lower()
                if "drafter" in low:
                    continue
                if any(k in low for k in ("qwen", "gemma", "coder", "flash", "autoround", "nvfp4")):
                    return str(cfg.parent)
        return None

    def _resolve_drafter_path(self) -> Optional[str]:
        env_d = os.environ.get("ARC_LOCAL_DRAFTER_PATH")
        if env_d and Path(env_d).exists():
            return env_d
        for p in self.config.candidate_drafter_paths:
            if p and Path(p).exists():
                return p
        kaggle_in = Path("/kaggle/input")
        if kaggle_in.exists():
            for cfg in kaggle_in.rglob("config.json"):
                if "drafter" in str(cfg).lower():
                    return str(cfg.parent)
        return None

    def ensure_server_running(self) -> bool:
        """Ensures the inference engine is online, launching via subprocess if needed."""
        if not self.config.enable_llm:
            return False
        if self.is_ready:
            return True
        if self.launch_attempted:
            return self.is_ready

        with self._LOCK:
            if self.is_ready:
                return True
            self.launch_attempted = True

            # 1. Probe if server is already running
            if self._probe_running_server():
                return True

            # 2. Check if model weights exist to launch a new server
            model_path = self._resolve_model_path()
            if not model_path:
                logger.warning("[OfflineServerManager] No model weights found locally; LLM offline.")
                return False

            drafter_path = self._resolve_drafter_path()

            # 3. Launch vLLM OpenAI API server as a managed subprocess
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--gpu-memory-utilization", str(self.config.vllm_gpu_memory_utilization),
                "--max-model-len", "16384",
                "--port", "8000",
                "--trust-remote-code",
            ]
            if self.config.enable_prefix_caching:
                cmd.append("--enable-prefix-caching")

            if drafter_path:
                cmd.extend([
                    "--speculative-model", drafter_path,
                    "--num-speculative-tokens", str(self.config.speculative_draft_tokens),
                ])

            logger.info(f"[OfflineServerManager] Launching vLLM server: {' '.join(cmd)}")
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                logger.error(f"[OfflineServerManager] Failed to launch vLLM subprocess: {e}")
                return False

            # 4. Wait for server to come online
            start_t = time.time()
            deadline = start_t + self.config.vllm_launch_timeout_s
            while time.time() < deadline:
                if self._probe_running_server():
                    logger.info(f"[OfflineServerManager] vLLM server came online in {time.time() - start_t:.1f}s.")
                    return True
                time.sleep(2.0)

            logger.error("[OfflineServerManager] Timed out waiting for vLLM server startup.")
            return False

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Optional[str]:
        """Issues synchronous inference request to the OpenAI-compatible HTTP server."""
        self.total_calls += 1
        if not self.ensure_server_running() or not self.http_endpoint:
            self.fallback_count += 1
            return None

        max_tok = max_new_tokens or self.config.llm_max_new_tokens
        limit_s = timeout_s or self.config.llm_timeout_s

        payload = json.dumps({
            "model": self.http_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tok,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")

        req = urllib.request.Request(
            self.http_endpoint,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "ARC-Agent"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=limit_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.debug(f"[OfflineServerManager] Inference request failed: {exc}")
            self.fallback_count += 1
            return None

    def infer_decision(self, prompt: str) -> Optional[Dict[str, Any]]:
        raw = self.generate_text(prompt)
        if not raw:
            return None
        return parse_codeact_response(raw)


class SandboxedCodeActREPL:
    """AST-validated sandboxed execution environment for Python CodeAct programs."""

    ALLOWED_MODULES = {"numpy", "math", "collections", "itertools"}

    def __init__(self, max_plan_len: int = 18):
        self.max_plan_len = max_plan_len

    def execute_codeact(
        self,
        code_str: str,
        environment_context: Dict[str, Any],
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """Executes LLM-synthesized code in a sandboxed namespace."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            logger.debug(f"[SandboxedCodeActREPL] Syntax error in synthesized code: {e}")
            return []

        # Validate AST: no illegal imports or OS/filesystem operations
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod_name = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                root_mod = mod_name.split(".")[0] if mod_name else ""
                if root_mod not in self.ALLOWED_MODULES:
                    logger.warning(f"[SandboxedCodeActREPL] Blocked unauthorized import: {root_mod}")
                    return []

        action_plan: List[Tuple[int, Dict[str, Any]]] = []
        notes = environment_context.get("notes", {})

        def act(action_id: int, x: int = 0, y: int = 0) -> None:
            if len(action_plan) < self.max_plan_len:
                action_plan.append((int(action_id), {"x": int(x), "y": int(y)}))

        def note(key: str, value: Any) -> None:
            notes[str(key)] = value

        query_fn = environment_context.get("query_log_fn", lambda n=5: [])

        def bfs_single(start: Tuple[int, int], target: Tuple[int, int]) -> List[int]:
            from collections import deque
            import numpy as np
            grid = environment_context.get("grid")
            if grid is None:
                return []
            stride = environment_context.get("stride", 5)
            bg = environment_context.get("bg_color", 4)
            action_map = environment_context.get(
                "action_map", {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
            )
            sub_h, sub_w = grid.shape[0] // stride, grid.shape[1] // stride

            # Auto-scale world pixel coordinates to stride space if needed
            s_r, s_c = start[0], start[1]
            if s_r >= sub_h or s_c >= sub_w:
                s_r, s_c = s_r // stride, s_c // stride
            t_r, t_c = target[0], target[1]
            if t_r >= sub_h or t_c >= sub_w:
                t_r, t_c = t_r // stride, t_c // stride
            start = (s_r, s_c)
            target = (t_r, t_c)

            grid_stride = np.zeros((sub_h, sub_w), dtype=int)
            for r in range(sub_h):
                for c in range(sub_w):
                    grid_stride[r, c] = grid[r * stride + stride // 2, c * stride + stride // 2]

            q = deque([(start, [])])
            visited = {start}
            while q:
                (r, c), path = q.popleft()
                if (r, c) == target:
                    return path
                for (dr, dc), aid in action_map.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < sub_h and 0 <= nc < sub_w and (nr, nc) not in visited:
                        val = grid_stride[nr, nc]
                        if (nr, nc) == target or val != bg:
                            visited.add((nr, nc))
                            q.append(((nr, nc), path + [aid]))
            return []

        def find_path(start: Tuple[int, int], target: Any) -> List[int]:
            """General 2D pathfinder. Accepts single (r, c) target or list of [(r, c), ...] targets.
            When multiple targets are given, computes the exact minimal tour via Held-Karp Metric TSP.
            """
            if not target:
                return []
            if isinstance(target, list) and target and isinstance(target[0], (tuple, list)):
                targets = [tuple(t[:2]) for t in target]
                if len(targets) == 1:
                    return bfs_single(start, targets[0])
                if len(targets) > 8:
                    targets = targets[:8]
                nodes = [start] + targets
                k = len(nodes)
                pair_paths = {}
                dist_matrix = [[float('inf')] * k for _ in range(k)]
                for i in range(k):
                    dist_matrix[i][i] = 0
                    for j in range(k):
                        if i == j:
                            continue
                        p = bfs_single(nodes[i], nodes[j])
                        if p or nodes[i] == nodes[j]:
                            dist_matrix[i][j] = len(p)
                            pair_paths[(i, j)] = p

                reachable = [i for i in range(1, k) if dist_matrix[0][i] < float('inf')]
                if not reachable:
                    return []
                if len(reachable) == 1:
                    return pair_paths.get((0, reachable[0]), [])

                sub_nodes = [0] + reachable
                sub_k = len(sub_nodes)
                sub_dist = [[dist_matrix[sub_nodes[i]][sub_nodes[j]] for j in range(sub_k)] for i in range(sub_k)]

                memo = {}
                import itertools
                for i in range(1, sub_k):
                    memo[(1 << i, i)] = (sub_dist[0][i], [i])
                for sz in range(2, sub_k):
                    for subset in itertools.combinations(range(1, sub_k), sz):
                        mask = sum(1 << i for i in subset)
                        for last in subset:
                            prev_mask = mask ^ (1 << last)
                            best_cost = float('inf')
                            best_p = []
                            for prev in subset:
                                if prev != last and (prev_mask, prev) in memo:
                                    cost = memo[(prev_mask, prev)][0] + sub_dist[prev][last]
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_p = memo[(prev_mask, prev)][1] + [last]
                            if best_cost < float('inf'):
                                memo[(mask, last)] = (best_cost, best_p)

                full_mask = sum(1 << i for i in range(1, sub_k))
                best_total = float('inf')
                best_tour = []
                for last in range(1, sub_k):
                    if (full_mask, last) in memo:
                        cost, p = memo[(full_mask, last)]
                        if cost < best_total:
                            best_total = cost
                            best_tour = p

                full_actions = []
                curr_node = 0
                for step_idx in best_tour:
                    orig_target_node = sub_nodes[step_idx]
                    full_actions.extend(pair_paths.get((curr_node, orig_target_node), []))
                    curr_node = orig_target_node
                return full_actions
            else:
                return bfs_single(start, tuple(target[:2]))

        sandbox_globals = {
            "__builtins__": {
                "range": range, "len": len, "enumerate": enumerate, "zip": zip,
                "int": int, "float": float, "str": str, "bool": bool, "list": list,
                "dict": dict, "set": set, "tuple": tuple, "min": min, "max": max,
                "abs": abs, "sum": sum, "round": round, "print": lambda *args: None,
            },
            "act": act,
            "note": note,
            "find_path": find_path,
            "query_log": query_fn,
            "np": __import__("numpy"),
            "math": __import__("math"),
            "grid": environment_context.get("grid"),
            "prev_grid": environment_context.get("prev_grid"),
            "avatar_pos": environment_context.get("avatar_pos"),
            "targets": environment_context.get("targets", []),
            "available_actions": environment_context.get("available_actions", [1, 2, 3, 4, 5, 6]),
            "entities": environment_context.get("entities", []),
        }

        try:
            compiled = compile(tree, "<codeact>", "exec")
            exec(compiled, sandbox_globals)
        except Exception as exc:
            logger.debug(f"[SandboxedCodeActREPL] Runtime error executing synthesized code: {exc}")
            return []

        return action_plan
