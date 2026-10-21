import ast
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

FORBIDDEN_AST_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Delete)
FORBIDDEN_NAMES = {"eval", "exec", "open", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals"}
SAFE_BUILTINS: Dict[str, Any] = {
    "range": range, "len": len, "min": min, "max": max, "abs": abs,
    "int": int, "float": float, "bool": bool, "str": str, "tuple": tuple,
    "list": list, "set": set, "dict": dict, "enumerate": enumerate,
    "zip": zip, "sum": sum, "any": any, "all": all, "round": round, "sorted": sorted,
}


class ASTSandbox:
    @staticmethod
    def validate_and_compile(code_str: str, fn_name: str = "predict_next_grid") -> Optional[Callable]:
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, FORBIDDEN_AST_NODES):
                    return None
                if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                    return None
            safe_globals: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "np": np}
            local_ns: Dict[str, Any] = {}
            exec(compile(tree, filename="<cegis_world_model>", mode="exec"), safe_globals, local_ns)
            fn = local_ns.get(fn_name)
            return fn if callable(fn) else None
        except Exception:
            return None

    @staticmethod
    def evaluate_retrodiction(
        fn: Callable,
        transitions: List[Tuple[np.ndarray, int, Dict[str, Any], np.ndarray]],
        hud_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, Optional[str]]:
        if not transitions:
            return 0.0, None
        passed = 0
        counterexample = None
        window = transitions[-14:]
        for pre_g, act_id, data, post_g in window:
            try:
                pred = fn(pre_g.copy(), int(act_id), data or {})
                if isinstance(pred, np.ndarray) and pred.shape == post_g.shape:
                    non_abstained = pred != -1
                    if np.mean(non_abstained) < 0.70:
                        counterexample = f"Action {act_id}: coverage too low"
                        continue
                    diff = (pred != post_g) & non_abstained
                    if hud_mask is not None and hud_mask.shape == diff.shape:
                        diff = diff & (~hud_mask)
                    else:
                        diff[:4, :] = False; diff[-4:, :] = False; diff[:, :4] = False; diff[:, -4:] = False
                    n_err = int(np.sum(diff))
                    if n_err == 0:
                        passed += 1
                    elif counterexample is None:
                        rs, cs = np.where(diff)
                        r0, c0 = int(rs[0]), int(cs[0])
                        counterexample = f"Action {act_id}: {n_err} px mismatch at ({r0},{c0})"
                else:
                    counterexample = f"Action {act_id}: invalid return shape"
            except Exception as e:
                counterexample = f"Action {act_id} raised {type(e).__name__}: {e}"
        return passed / float(len(window)), counterexample


class CodeActREPL:
    def __init__(self):
        self.queued_plan: List[Any] = []
        self.workspace_vars: Dict[str, Any] = {}

    def reset_level(self) -> None:
        self.queued_plan.clear()

    def execute_codeact(
        self,
        code_str: str,
        grid: np.ndarray,
        prev_grid: Optional[np.ndarray],
        av_pos: Optional[Tuple[int, int]],
        bg_color: int,
        entities: List[Any],
        cavities: List[Tuple[int, int]],
        vectors: Dict[int, Tuple[int, int]],
        avail_actions: List[int],
        history_records: List[Dict[str, Any]],
        portals: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> Optional[List[Any]]:
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, FORBIDDEN_AST_NODES):
                    return None
                if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                    return None
        except Exception:
            return None

        self.queued_plan = []

        def _queue_actions(acts: List[Any]) -> None:
            for a in acts[:16]:
                self.queued_plan.append(a)

        def _click_xy(x: int, y: int) -> None:
            self.queued_plan.append((6, {"x": int(x), "y": int(y)}))

        def _find_color_centroids(color: int) -> List[Tuple[int, int]]:
            pts = np.argwhere(grid == int(color))
            return [(int(pts[:, 0].mean()), int(pts[:, 1].mean()))] if len(pts) > 0 else []

        def _grep_history(pattern: str) -> List[Dict[str, Any]]:
            p = str(pattern).lower()
            return [r for r in history_records if p in str(r).lower()][-8:]

        def _find_causal_triggers() -> List[Dict[str, Any]]:
            return [r for r in history_records if r.get("remote_triggers")]

        env_vars: Dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "np": np,
            "grid": grid.copy(),
            "prev_grid": prev_grid.copy() if prev_grid is not None else grid.copy(),
            "av_pos": av_pos,
            "bg_color": int(bg_color),
            "entities": [{"id": e.id, "color": e.color, "mass": e.mass, "rc": (int(round(e.centroid[0])), int(round(e.centroid[1])))} for e in entities],
            "cavities": list(cavities),
            "vectors": dict(vectors),
            "portals": dict(portals or {}),
            "avail_actions": list(avail_actions),
            "history": list(history_records[-16:]),
            "workspace": self.workspace_vars,
            "queue_actions": _queue_actions,
            "click_xy": _click_xy,
            "find_color_centroids": _find_color_centroids,
            "grep_history": _grep_history,
            "find_causal_triggers": _find_causal_triggers,
        }
        local_ns: Dict[str, Any] = {}
        try:
            exec(compile(tree, filename="<duck_codeact_repl>", mode="exec"), env_vars, local_ns)
            if "solve_turn" in local_ns and callable(local_ns["solve_turn"]):
                res = local_ns["solve_turn"](env_vars)
                if isinstance(res, list) and res:
                    return res[:16]
                if isinstance(res, int):
                    return [res]
            if self.queued_plan:
                return list(self.queued_plan)
            if "plan" in local_ns and isinstance(local_ns["plan"], list) and local_ns["plan"]:
                return local_ns["plan"][:16]
        except Exception:
            return None
        return None
