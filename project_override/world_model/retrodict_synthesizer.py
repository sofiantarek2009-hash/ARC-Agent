from collections import deque
import heapq
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from project_override.core.dead_signature import DeadSignatureTable, DynamicHUDMasker, compute_arena_hash
from project_override.world_model.ast_sandbox import ASTSandbox

logger = logging.getLogger(__name__)


def _install_pure_pytorch_awq_fallback() -> None:
    """Fast uint8-unpacked GPU 4-bit AWQ dequantizer for Kaggle's offline Python 3.13 container."""
    try:
        import awq  # noqa: F401
        return
    except ImportError:
        pass

    try:
        import torch
        import torch.nn as nn
        from transformers.quantizers import AutoHfQuantizer, HfQuantizer

        class TorchAWQLinear(nn.Module):
            def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 128):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.group_size = group_size
                pack_num = 8
                self.register_buffer(
                    "qweight",
                    torch.zeros((in_features, out_features // pack_num), dtype=torch.int32),
                )
                self.register_buffer(
                    "qzeros",
                    torch.zeros((in_features // group_size, out_features // pack_num), dtype=torch.int32),
                )
                self.register_buffer(
                    "scales",
                    torch.zeros((in_features // group_size, out_features), dtype=torch.float16),
                )
                if bias:
                    self.register_buffer("bias", torch.zeros((out_features,), dtype=torch.float16))
                else:
                    self.bias = None
                self.register_buffer(
                    "_shifts",
                    torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int32).view(1, 1, 8),
                    persistent=False,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                w_unpacked = (
                    torch.bitwise_and(torch.bitwise_right_shift(self.qweight.unsqueeze(-1), self._shifts), 0x0F)
                    .to(torch.uint8)
                    .reshape(self.in_features, self.out_features)
                )
                z_unpacked = (
                    torch.bitwise_and(torch.bitwise_right_shift(self.qzeros.unsqueeze(-1), self._shifts), 0x0F)
                    .to(torch.uint8)
                    .reshape(self.in_features // self.group_size, self.out_features)
                )
                z_expanded = z_unpacked.repeat_interleave(self.group_size, dim=0)
                s_expanded = self.scales.repeat_interleave(self.group_size, dim=0)
                w_fp16 = (w_unpacked.to(torch.float16) - z_expanded.to(torch.float16)) * s_expanded
                out = torch.matmul(x.to(torch.float16), w_fp16)
                if self.bias is not None:
                    out = out + self.bias
                return out.to(x.dtype)

        class PureTorchAwqQuantizer(HfQuantizer):
            requires_calibration = False
            required_packages = []

            def __init__(self, quantization_config: Any, **kwargs: Any):
                super().__init__(quantization_config, **kwargs)
                self.group_size = getattr(quantization_config, "group_size", 128)

            def validate_environment(self, *args: Any, **kwargs: Any) -> None:
                return

            def update_torch_dtype(self, torch_dtype: Any) -> Any:
                return torch.float16

            def _process_model_before_weight_loading(self, model: nn.Module, **kwargs: Any) -> None:
                def _replace(module: nn.Module, prefix: str = "") -> None:
                    for name, child in list(module.named_children()):
                        full = f"{prefix}.{name}" if prefix else name
                        if isinstance(child, nn.Linear) and "lm_head" not in full:
                            if child.in_features % self.group_size == 0 and child.out_features % 8 == 0:
                                setattr(
                                    module,
                                    name,
                                    TorchAWQLinear(
                                        child.in_features,
                                        child.out_features,
                                        bias=child.bias is not None,
                                        group_size=self.group_size,
                                    ),
                                )
                        else:
                            _replace(child, full)
                _replace(model)

            def _process_model_after_weight_loading(self, model: nn.Module, **kwargs: Any) -> None:
                return

            @property
            def is_trainable(self) -> bool:
                return False

            @property
            def is_serializable(self) -> bool:
                return True

        AutoHfQuantizer.register("awq", PureTorchAwqQuantizer, exist_ok=True)
    except Exception:
        pass


class OfflineLLMRuntime:
    """Thread-safe Singleton for Qwen 2.5 Coder 32B Instruct AWQ on Kaggle Dual T4."""

    _INSTANCE: Optional["OfflineLLMRuntime"] = None
    _LOAD_LOCK = threading.Lock()
    _INFERENCE_LOCK = threading.Lock()

    CANDIDATE_PATHS = [
        "/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/32b-instruct-awq/1",
        "/kaggle/input/qwen2.5-coder/transformers/32b-instruct-awq/1",
        "/kaggle/input/qwen-2.5-coder-32b-instruct-awq",
    ]

    def __init__(self):
        self.model_path: Optional[str] = self._resolve_model_path()
        self.model: Any = None
        self.tokenizer: Any = None
        self.is_loaded: bool = False
        self.load_attempted: bool = False

    @classmethod
    def get_instance(cls) -> "OfflineLLMRuntime":
        if cls._INSTANCE is None:
            with cls._LOAD_LOCK:
                if cls._INSTANCE is None:
                    cls._INSTANCE = cls()
        return cls._INSTANCE

    def _resolve_model_path(self) -> Optional[str]:
        env_p = os.environ.get("ARC_LOCAL_MODEL_PATH")
        if env_p and Path(env_p).exists():
            return env_p
        for p in self.CANDIDATE_PATHS:
            if Path(p).exists():
                return p
        kaggle_in = Path("/kaggle/input")
        if kaggle_in.exists():
            for cfg in kaggle_in.rglob("config.json"):
                if "qwen" in str(cfg).lower() or "coder" in str(cfg).lower():
                    return str(cfg.parent)
        return None

    def ensure_loaded(self) -> bool:
        if self.is_loaded:
            return True
        if self.load_attempted or self.model_path is None:
            return False
        with self._LOAD_LOCK:
            if self.is_loaded:
                return True
            self.load_attempted = True
            try:
                import torch
                if not torch.cuda.is_available():
                    return False
                _install_pure_pytorch_awq_fallback()
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True, local_files_only=True
                )
                n_gpus = torch.cuda.device_count()
                max_mem = {i: "14GiB" for i in range(n_gpus)}
                max_mem["cpu"] = "24GiB"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    max_memory=max_mem,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self.model.eval()
                self.is_loaded = True
                return True
            except Exception as e:
                logger.warning(f"OfflineLLMRuntime fallback to symbolic CEGIS ({e})")
                self.is_loaded = False
                return False

    def generate_text(self, prompt: str, max_new_tokens: int = 210, timeout_s: float = 14.0) -> Optional[str]:
        if not self.ensure_loaded():
            return None
        if not self._INFERENCE_LOCK.acquire(timeout=2.0):
            return None
        try:
            import torch
            t0 = time.perf_counter()
            if (time.perf_counter() - t0) > timeout_s:
                return None
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = prompt
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        except Exception:
            return None
        finally:
            self._INFERENCE_LOCK.release()

    def infer_subgoal(self, prompt: str) -> Optional[Dict[str, Any]]:
        resp = self.generate_text(prompt, max_new_tokens=150)
        if not resp:
            return None
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


class RetrodictWorldModelSynthesizer:
    """
    Tycho Rendered Deterministic Moore Machine + Auto-Triggered Model Repair (arXiv:2607.28287)
    + 8-Archetype CEGIS Program Bank + Best-First Width Search (BFWS / IW(2)).
    """

    def __init__(self):
        self.transitions: List[Tuple[np.ndarray, int, Dict[str, Any], np.ndarray]] = []
        self.verified_fn: Optional[Callable] = None
        self.verified_code: Optional[str] = None
        self.verified_accuracy: float = 0.0
        self.needs_repair: bool = False

    def reset_level(self, keep_verified: bool = True) -> None:
        self.transitions.clear()
        self.needs_repair = False
        if not keep_verified:
            self.verified_fn = None
            self.verified_code = None
            self.verified_accuracy = 0.0

    def record_transition(
        self,
        pre_grid: np.ndarray,
        action_id: int,
        data: Dict[str, Any],
        post_grid: np.ndarray,
        hud_mask: Optional[np.ndarray] = None,
    ) -> None:
        if not np.array_equal(pre_grid, post_grid):
            self.transitions.append((pre_grid.copy(), int(action_id), dict(data or {}), post_grid.copy()))
            if len(self.transitions) > 20:
                self.transitions.pop(0)
            # Tycho Auto-Triggered Repair: check if active model mispredicts this new transition
            if self.verified_fn is not None:
                acc, _ = ASTSandbox.evaluate_retrodiction(
                    self.verified_fn, [self.transitions[-1]], hud_mask=hud_mask
                )
                if acc < 1.0:
                    self.needs_repair = True

    def _build_archetype_candidates(
        self,
        floor_color: int,
        avatar_colors: Set[int],
        wall_colors: Set[int],
        box_colors: Set[int],
        vectors: Dict[int, Tuple[int, int]],
    ) -> List[str]:
        fl = int(floor_color)
        av_repr = repr(set(int(c) for c in avatar_colors))
        wl_repr = repr(set(int(c) for c in wall_colors))
        bx_repr = repr(set(int(c) for c in box_colors))
        vc_repr = repr(vectors)

        # 1. Cardinal + Sokoban Multi-Box Push + Action-5 Brush Stamp
        c1_cardinal = f"""
def predict_next_grid(grid, action, data=None):
    vecs = {vc_repr}
    floor_c = {fl}
    av_cols = {av_repr}
    wall_cols = {wl_repr}
    box_cols = {bx_repr}
    h, w = grid.shape
    mask = np.isin(grid, list(av_cols))
    mask[:4, :] = False; mask[h-4:, :] = False; mask[:, :4] = False; mask[:, w-4:] = False
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return grid
    if action == 5:
        nxt = grid.copy()
        r0, c0 = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r0 + dr, c0 + dc
                if 4 <= nr < h - 4 and 4 <= nc < w - 4 and int(nxt[nr, nc]) == floor_c:
                    nxt[nr, nc] = min(av_cols)
        return nxt
    if action not in vecs:
        return grid
    dr, dc = vecs[action]
    for r, c in pts:
        nr, nc = int(r + dr), int(c + dc)
        if not (4 <= nr < h - 4 and 4 <= nc < w - 4):
            return grid
        if int(grid[nr, nc]) in wall_cols:
            return grid
    nxt = grid.copy()
    for r, c in pts:
        nxt[int(r), int(c)] = floor_c
    for r, c in pts:
        nr, nc = int(r + dr), int(c + dc)
        if int(grid[nr, nc]) in box_cols:
            br, bc = int(nr + dr), int(nc + dc)
            if 4 <= br < h - 4 and 4 <= bc < w - 4 and int(grid[br, bc]) not in wall_cols:
                nxt[br, bc] = grid[nr, nc]
        nxt[nr, nc] = grid[int(r), int(c)]
    return nxt
"""
        # 2. Toroidal Modulo Wrap-Around Arena
        c2_toroidal = f"""
def predict_next_grid(grid, action, data=None):
    vecs = {vc_repr}
    if action not in vecs:
        return grid
    dr, dc = vecs[action]
    floor_c = {fl}
    av_cols = {av_repr}
    wall_cols = {wl_repr}
    h, w = grid.shape
    mask = np.isin(grid, list(av_cols))
    mask[:4, :] = False; mask[h-4:, :] = False; mask[:, :4] = False; mask[:, w-4:] = False
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return grid
    nxt = grid.copy()
    for r, c in pts:
        nr = 4 + ((int(r + dr) - 4) % (h - 8))
        nc = 4 + ((int(c + dc) - 4) % (w - 8))
        if int(grid[nr, nc]) in wall_cols:
            return grid
    for r, c in pts:
        nxt[int(r), int(c)] = floor_c
    for r, c in pts:
        nr = 4 + ((int(r + dr) - 4) % (h - 8))
        nc = 4 + ((int(c + dc) - 4) % (w - 8))
        nxt[nr, nc] = grid[int(r), int(c)]
    return nxt
"""
        # 3. Ice-Slide Continuous Raycast
        c3_slide = f"""
def predict_next_grid(grid, action, data=None):
    vecs = {vc_repr}
    if action not in vecs:
        return grid
    dr, dc = vecs[action]
    ur = 0 if dr == 0 else (1 if dr > 0 else -1)
    uc = 0 if dc == 0 else (1 if dc > 0 else -1)
    floor_c = {fl}
    av_cols = {av_repr}
    wall_cols = {wl_repr}
    h, w = grid.shape
    mask = np.isin(grid, list(av_cols))
    mask[:4, :] = False; mask[h-4:, :] = False; mask[:, :4] = False; mask[:, w-4:] = False
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return grid
    k = 0
    while k < 56:
        blocked = False
        for r, c in pts:
            nr, nc = int(r + ur * (k + 1)), int(c + uc * (k + 1))
            if not (4 <= nr < h - 4 and 4 <= nc < w - 4) or int(grid[nr, nc]) in wall_cols:
                blocked = True
                break
        if blocked:
            break
        k += 1
    if k == 0:
        return grid
    nxt = grid.copy()
    for r, c in pts:
        nxt[int(r), int(c)] = floor_c
    for r, c in pts:
        nxt[int(r + ur * k), int(c + uc * k)] = grid[int(r), int(c)]
    return nxt
"""
        # 4. Gravity / Falling-Sand Cascade after Move
        c4_gravity = f"""
def predict_next_grid(grid, action, data=None):
    vecs = {vc_repr}
    if action not in vecs:
        return grid
    dr, dc = vecs[action]
    floor_c = {fl}
    av_cols = {av_repr}
    wall_cols = {wl_repr}
    h, w = grid.shape
    mask = np.isin(grid, list(av_cols))
    mask[:4, :] = False; mask[h-4:, :] = False; mask[:, :4] = False; mask[:, w-4:] = False
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return grid
    for r, c in pts:
        nr, nc = int(r + dr), int(c + dc)
        if not (4 <= nr < h - 4 and 4 <= nc < w - 4) or int(grid[nr, nc]) in wall_cols:
            return grid
    nxt = grid.copy()
    for r, c in pts:
        nxt[int(r), int(c)] = floor_c
    for r, c in pts:
        nxt[int(r + dr), int(c + dc)] = grid[int(r), int(c)]
    for _ in range(4):
        pts2 = np.argwhere(np.isin(nxt, list(av_cols)))
        can_fall = len(pts2) > 0 and all(
            int(r + 1) < h - 4 and (int(nxt[int(r + 1), int(c)]) == floor_c or int(nxt[int(r + 1), int(c)]) in av_cols)
            for r, c in pts2
        )
        if not can_fall:
            break
        prev_nxt = nxt.copy()
        for r, c in pts2:
            nxt[int(r), int(c)] = floor_c
        for r, c in pts2:
            nxt[int(r + 1), int(c)] = prev_nxt[int(r), int(c)]
    return nxt
"""
        return [c1_cardinal, c2_toroidal, c3_slide, c4_gravity]

    def synthesize_world_model(
        self,
        floor_color: Optional[int],
        avatar_colors: Set[int],
        wall_colors: Set[int],
        box_colors: Set[int],
        vectors: Dict[int, Tuple[int, int]],
        hud_mask: Optional[np.ndarray] = None,
    ) -> Optional[Callable]:
        if floor_color is None or not avatar_colors or not vectors:
            return None

        best_fn = None
        best_acc = -1.0
        best_ce = None
        best_code = None

        for code_cand in self._build_archetype_candidates(
            floor_color, avatar_colors, wall_colors, box_colors, vectors
        ):
            fn = ASTSandbox.validate_and_compile(code_cand)
            if fn is not None:
                if len(self.transitions) >= 2:
                    acc, ce = ASTSandbox.evaluate_retrodiction(fn, self.transitions, hud_mask=hud_mask)
                    if acc > best_acc:
                        best_acc = acc
                        best_fn = fn
                        best_ce = ce
                        best_code = code_cand
                    if acc >= 0.90:
                        self.verified_fn = fn
                        self.verified_code = code_cand
                        self.verified_accuracy = acc
                        self.needs_repair = False
                        return fn
                elif best_fn is None:
                    best_fn = fn
                    best_code = code_cand
                    best_acc = 0.5

        # Multi-Turn Qwen 2.5 Coder 32B AWQ CEGIS Refinement for Alien Mechanics
        llm = OfflineLLMRuntime.get_instance()
        if llm.ensure_loaded() and len(self.transitions) >= 2:
            curr_ce = best_ce
            for _ in range(2):
                cegis_prompt = (
                    "Write ONLY a pure Python function `def predict_next_grid(grid, action, data=None):` "
                    "using `np` (no imports) that predicts the next 64x64 int8 grid.\n"
                    f"Induced parameters: floor={floor_color}, avatar={avatar_colors}, walls={wall_colors}, "
                    f"boxes={box_colors}, vectors={vectors}.\n"
                    f"Counterexample on current model: {curr_ce}\n"
                    "Return ONLY valid Python code inside ```python ... ```."
                )
                resp = llm.generate_text(cegis_prompt, max_new_tokens=240)
                if not resp:
                    break
                m = re.search(r"```(?:python)?\s*(def predict_next_grid.*?)```", resp, re.DOTALL)
                cand_code = m.group(1).strip() if m else resp.strip()
                llm_fn = ASTSandbox.validate_and_compile(cand_code)
                if llm_fn is not None:
                    acc2, ce2 = ASTSandbox.evaluate_retrodiction(llm_fn, self.transitions, hud_mask=hud_mask)
                    if acc2 > best_acc:
                        best_acc = acc2
                        best_fn = llm_fn
                        best_code = cand_code
                    if acc2 >= 0.85:
                        self.verified_fn = llm_fn
                        self.verified_code = cand_code
                        self.verified_accuracy = acc2
                        self.needs_repair = False
                        return llm_fn
                    curr_ce = ce2

        if best_fn is not None:
            self.verified_fn = best_fn
            self.verified_code = best_code
            self.verified_accuracy = max(0.0, best_acc)
            self.needs_repair = False
        return best_fn

    def plan_iw_lookahead(
        self,
        start_grid: np.ndarray,
        available_actions: List[int],
        target_pos: Optional[Tuple[int, int]],
        avatar_colors: Set[int],
        dead_table: Optional[DeadSignatureTable] = None,
        hud_masker: Optional[DynamicHUDMasker] = None,
        level_idx: int = 0,
        max_expansions: int = 160,
    ) -> Optional[List[int]]:
        """
        Replay-Audited Best-First Width Search (BFWS) combining IW(1) + IW(2) Novelty
        with Goal-Distance Heuristic and Dead/Hazard Signature Pruning.
        """
        if self.verified_fn is None or not avatar_colors:
            return None
        moves = [a for a in available_actions if a in (1, 2, 3, 4, 5)]
        if not moves:
            return None

        seen_atoms: Set[Tuple[int, int, int]] = set()
        seen_pairs: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()

        def _evaluate_novelty(g: np.ndarray) -> int:
            nov = 0
            fg_atoms = []
            for r in range(4, g.shape[0] - 4, 3):
                for c in range(4, g.shape[1] - 4, 3):
                    val = int(g[r, c])
                    atom = (r, c, val)
                    fg_atoms.append(atom)
                    if atom not in seen_atoms:
                        seen_atoms.add(atom)
                        nov += 2
            if nov == 0 and len(fg_atoms) <= 48:
                for i in range(0, len(fg_atoms), 2):
                    for j in range(i + 1, min(i + 4, len(fg_atoms))):
                        p = (fg_atoms[i], fg_atoms[j])
                        if p not in seen_pairs:
                            seen_pairs.add(p)
                            nov += 1
            return nov

        _evaluate_novelty(start_grid)
        tie_breaker = 0
        open_heap: List[Tuple[float, int, np.ndarray, List[int]]] = [(0.0, tie_breaker, start_grid.copy(), [])]
        expansions = 0

        while open_heap and expansions < max_expansions:
            _, _, g, path = heapq.heappop(open_heap)
            expansions += 1
            if len(path) >= 16:
                continue
            g_hash = compute_arena_hash(g, bezel_px=4, hud_masker=hud_masker) if dead_table is not None else 0
            for a in moves:
                if dead_table is not None and dead_table.is_dead(g_hash, a, 0, 0, level_idx):
                    continue
                try:
                    nxt_g = self.verified_fn(g, a, {})
                except Exception:
                    continue
                if np.array_equal(nxt_g, g):
                    continue
                new_path = path + [a]
                if target_pos is not None:
                    tr, tc = target_pos
                    if 0 <= tr < nxt_g.shape[0] and 0 <= tc < nxt_g.shape[1]:
                        if int(nxt_g[tr, tc]) in avatar_colors:
                            return new_path
                nov = _evaluate_novelty(nxt_g)
                if nov > 0:
                    dist_pen = 0.0
                    if target_pos is not None:
                        pts = np.argwhere(np.isin(nxt_g, list(avatar_colors)))
                        if len(pts) > 0:
                            av_r, av_c = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                            dist_pen = (abs(av_r - target_pos[0]) + abs(av_c - target_pos[1])) * 0.15
                    priority = len(new_path) + dist_pen - (nov * 1.5)
                    tie_breaker += 1
                    heapq.heappush(open_heap, (priority, tie_breaker, nxt_g, new_path))
        return None
