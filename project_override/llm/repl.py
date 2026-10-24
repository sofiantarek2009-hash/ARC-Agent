import ast
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from project_override.perception.observation_processor import Entity

class SandboxedREPL:
    """
    Tufa Labs Duck-style + Schema/PRO-LONG Python REPL with AST safety validation.
    Allows the model to inspect `grid`, `prev_grid`, `objects`, `avail_actions`,
    query recorded history via `query_log(action_id=None, min_delta=0)`,
    register an executable simulator via `register_world_model(step_fn, goal_fn=None)`,
    update persistent `notes`, and emit actions via `act(id, x=0, y=0)` or `plan`.
    """
    FORBIDDEN_NODES = (
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
    )
    FORBIDDEN_NAMES = {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "input",
    }

    def __init__(self, max_plan_len: int = 14):
        self.max_plan_len = max_plan_len

    def validate_ast(self, code_str: str) -> bool:
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, self.FORBIDDEN_NODES):
                return False
            if isinstance(node, ast.Name) and node.id in self.FORBIDDEN_NAMES:
                return False
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return False
        return True

    def execute_codeact(
        self,
        code_str: str,
        grid: np.ndarray,
        prev_grid: Optional[np.ndarray],
        av_pos: Optional[Tuple[int, int]],
        bg_color: int,
        entities: List[Entity],
        vectors: Dict[int, Tuple[int, int]],
        avail_actions: List[int],
        notes: Dict[str, str],
        query_log_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        register_wm_fn: Optional[Callable[..., None]] = None,
    ) -> List[Tuple[int, Dict[str, int]]]:
        if not self.validate_ast(code_str):
            return []
        emitted: List[Tuple[int, Dict[str, int]]] = []

        def act(action_id: int, x: int = 0, y: int = 0) -> None:
            aid = int(action_id)
            if aid in avail_actions and len(emitted) < self.max_plan_len:
                if aid == 6:
                    emitted.append(
                        (6, {"x": max(0, min(63, int(x))), "y": max(0, min(63, int(y)))})
                    )
                else:
                    emitted.append((aid, {}))

        def note(key: str, val: Any) -> None:
            notes[str(key)[:40]] = str(val)[:160]

        def query_log(action_id: Optional[int] = None, min_delta: int = 0) -> List[Dict[str, Any]]:
            if query_log_fn is not None:
                return query_log_fn(action_id=action_id, min_delta=min_delta)
            return []

        def register_world_model(step_fn: Any, goal_fn: Any = None) -> None:
            if register_wm_fn is not None and callable(step_fn):
                register_wm_fn(step_fn, goal_fn if callable(goal_fn) else None)

        safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "any": any,
            "all": all,
        }
        obj_dicts = [
            {
                "id": e.id,
                "color": e.color,
                "mass": e.mass,
                "r": int(round(e.centroid[0])),
                "c": int(round(e.centroid[1])),
                "bbox": e.bbox,
            }
            for e in entities
        ]
        local_env: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "np": np,
            "grid": grid.copy(),
            "prev_grid": prev_grid.copy() if prev_grid is not None else grid.copy(),
            "av_pos": av_pos,
            "bg_color": bg_color,
            "objects": obj_dicts,
            "vectors": dict(vectors),
            "avail_actions": list(avail_actions),
            "notes": notes,
            "act": act,
            "note": note,
            "query_log": query_log,
            "register_world_model": register_world_model,
            "plan": [],
        }
        try:
            exec(compile(code_str, "<repl>", "exec"), local_env, local_env)
        except Exception:
            return emitted[: self.max_plan_len]
        if not emitted and isinstance(local_env.get("plan"), list):
            for item in local_env["plan"][: self.max_plan_len]:
                if isinstance(item, int) and item in avail_actions:
                    emitted.append((item, {}))
                elif isinstance(item, tuple) and len(item) == 2 and int(item[0]) in avail_actions:
                    emitted.append((int(item[0]), dict(item[1])))
                elif (
                    isinstance(item, tuple)
                    and len(item) == 3
                    and int(item[0]) == 6
                    and 6 in avail_actions
                ):
                    emitted.append((6, {"x": int(item[1]), "y": int(item[2])}))
        return emitted[: self.max_plan_len]
