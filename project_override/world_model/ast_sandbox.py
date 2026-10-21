import ast
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


FORBIDDEN_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
)

FORBIDDEN_NAMES = {"eval", "exec", "open", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals"}


class ASTSandbox:
    """Safe AST validator and retrodiction verifier for CEGIS Python world models."""

    @staticmethod
    def validate_and_compile(code_str: str, fn_name: str = "predict_next_grid") -> Optional[Callable]:
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, FORBIDDEN_AST_NODES):
                    return None
                if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                    return None
            safe_globals: Dict[str, Any] = {
                "__builtins__": {
                    "range": range,
                    "len": len,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "tuple": tuple,
                    "list": list,
                    "set": set,
                    "dict": dict,
                    "enumerate": enumerate,
                    "zip": zip,
                    "sum": sum,
                    "any": any,
                    "all": all,
                    "round": round,
                },
                "np": np,
            }
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
    ) -> Tuple[float, Optional[str]]:
        """
        Tests synthesized predict_next_grid(grid, action, data) against historical transitions.
        Returns (accuracy_0_to_1, first_counterexample_diff_str).
        """
        if not transitions:
            return 0.0, None
        passed = 0
        counterexample = None
        for pre_g, act_id, data, post_g in transitions[-10:]:
            try:
                pred = fn(pre_g.copy(), int(act_id), data or {})
                if isinstance(pred, np.ndarray) and pred.shape == post_g.shape:
                    diff = pred != post_g
                    n_err = int(np.sum(diff))
                    if n_err == 0:
                        passed += 1
                    elif counterexample is None:
                        rs, cs = np.where(diff)
                        r0, c0 = int(rs[0]), int(cs[0])
                        counterexample = (
                            f"Action {act_id}: {n_err} px mismatch; at ({r0},{c0}) "
                            f"expected {int(post_g[r0, c0])}, got {int(pred[r0, c0])}"
                        )
                else:
                    counterexample = f"Action {act_id}: invalid return shape"
            except Exception as e:
                counterexample = f"Action {act_id} raised {type(e).__name__}: {e}"
        return passed / float(len(transitions[-10:])), counterexample
