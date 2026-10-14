"""Project Override: AST Security Gate & Safe Sandbox Compiler/Executor
ARC-AGI-3 Reflective World Model Execution Sandbox.

Enforces strict structural constraints on candidate world models:
- AST Security Gate (NodeVisitor):
  * Whitelists: numpy, math, itertools, scipy.
  * Explicitly bans while-loops (Halting Problem guard).
  * Bans file I/O, os, sys, subprocess, network, and eval/exec inside generated code.
  * Bans dunder attribute and method introspection (__subclasses__, __globals__, etc.).
- Safe compiler and executor:
  * Strict timeout execution via process isolation or high-resolution signals.
  * Isolated memory namespace with restricted safe builtins.
"""

from __future__ import annotations

import ast
import builtins
import io
import itertools
import math
import multiprocessing
import os
import signal
import sys
import threading
import time
import types
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import scipy
    import scipy.ndimage
    import scipy.spatial
    _SCIPY_AVAILABLE = True
except ImportError:
    scipy = None
    _SCIPY_AVAILABLE = False


# =====================================================================
# Security Exceptions
# =====================================================================

class SandboxSecurityError(Exception):
    """Base exception for all AST sandbox security violations."""
    pass


class SecurityViolationError(SandboxSecurityError):
    """Raised when an AST node violates sandbox security policies."""

    def __init__(
        self,
        message: str,
        lineno: Optional[int] = None,
        col_offset: Optional[int] = None,
        node_type: Optional[str] = None,
    ) -> None:
        self.message = message
        self.lineno = lineno
        self.col_offset = col_offset
        self.node_type = node_type
        location = f" at line {lineno}:{col_offset}" if lineno is not None else ""
        super().__init__(f"Security violation{location}: {message}")


class SandboxCompilationError(SandboxSecurityError):
    """Raised when candidate code fails syntax parsing or compilation."""
    pass


class ExecutionTimeoutError(TimeoutError):
    """Raised when sandboxed execution exceeds the strict time quota."""
    pass


class SandboxRuntimeError(RuntimeError):
    """Raised when candidate code raises an uncaught runtime exception."""
    pass


# =====================================================================
# AST Security Gate (NodeVisitor)
# =====================================================================

ALLOWED_TOP_MODULES: Set[str] = {"numpy", "np", "math", "itertools", "scipy"}

ALLOWED_MODULE_PREFIXES: Tuple[str, ...] = (
    "numpy.",
    "scipy.",
    "math",
    "itertools",
)

BANNED_BUILTIN_NAMES: Set[str] = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "dir",
    "help",
    "exit",
    "quit",
    "memoryview",
    "getattr",
    "setattr",
    "delattr",
    "classmethod",
    "staticmethod",
    "super",
}

BANNED_MODULE_NAMES: Set[str] = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "urllib",
    "requests",
    "http",
    "ftplib",
    "asyncio",
    "threading",
    "multiprocessing",
    "shutil",
    "pickle",
    "shelve",
    "posix",
    "nt",
    "ctypes",
    "io",
    "pathlib",
    "importlib",
    "builtins",
    "__builtin__",
    "code",
    "codeop",
    "pty",
    "commands",
    "runpy",
    "trace",
    "pdb",
    "linecache",
    "inspect",
    "webbrowser",
    "telnetlib",
    "ssl",
}

BANNED_DUNDER_ATTRIBUTES: Set[str] = {
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__builtins__",
    "__code__",
    "__closure__",
    "__class__",
    "__dict__",
    "__module__",
    "__import__",
    "gi_frame",
    "f_globals",
    "f_locals",
    "f_builtins",
    "func_globals",
    "func_code",
    "cr_frame",
}

BANNED_METHODS: Set[str] = {
    "system",
    "popen",
    "spawn",
    "fork",
    "execve",
    "kill",
    "remove",
    "unlink",
    "rmdir",
    "rename",
    "chmod",
    "chown",
    "mkdir",
    "rmtree",
    "read",
    "write",
    "readline",
    "readlines",
}


class ASTSecurityGate(ast.NodeVisitor):
    """Validates candidate Python AST against Project Override security policies.

    Enforces:
    - Banning while-loops (Halting Problem guard).
    - Whitelisting modules: numpy, math, itertools, scipy.
    - Banning file I/O, os, sys, subprocess, network, threading, multiprocessing.
    - Banning eval/exec/compile/__import__ and dunder introspection.
    """

    def __init__(self) -> None:
        self.violations: List[str] = []

    def validate(self, source_or_tree: Union[str, ast.AST]) -> None:
        """Validates code string or AST. Raises SecurityViolationError on violation."""
        if isinstance(source_or_tree, str):
            try:
                tree = ast.parse(source_or_tree, filename="<candidate_world_model>")
            except SyntaxError as e:
                raise SandboxCompilationError(f"Syntax error during AST parse: {e}") from e
        else:
            tree = source_or_tree

        self.violations.clear()
        self.visit(tree)

        if self.violations:
            raise SecurityViolationError("; ".join(self.violations))

    def visit_While(self, node: ast.While) -> Any:
        """Explicitly ban while-loops to prevent non-terminating loops."""
        self._record_violation(
            "While-loops are strictly prohibited (Halting Problem guard). Use bounded for-loops instead.",
            node,
        )
        # Continue visiting children to collect any further violations
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        """Inspect import statements. Only whitelisted modules allowed."""
        for alias in node.names:
            root_module = alias.name.split(".")[0]
            if root_module not in ALLOWED_TOP_MODULES:
                self._record_violation(
                    f"Importing module '{alias.name}' is prohibited. Whitelist: {sorted(ALLOWED_TOP_MODULES)}",
                    node,
                )
            elif alias.name in BANNED_MODULE_NAMES:
                self._record_violation(
                    f"Module '{alias.name}' is explicitly blacklisted.",
                    node,
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        """Inspect from-import statements. Only whitelisted modules allowed."""
        if node.level is not None and node.level > 0:
            self._record_violation("Relative imports are strictly prohibited.", node)

        if node.module is not None:
            root_module = node.module.split(".")[0]
            if root_module not in ALLOWED_TOP_MODULES:
                self._record_violation(
                    f"Import from module '{node.module}' is prohibited. Whitelist: {sorted(ALLOWED_TOP_MODULES)}",
                    node,
                )
            elif node.module in BANNED_MODULE_NAMES:
                self._record_violation(
                    f"Module '{node.module}' is explicitly blacklisted.",
                    node,
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        """Inspect function and method calls."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in BANNED_BUILTIN_NAMES:
                self._record_violation(
                    f"Call to prohibited builtin '{func_name}' is forbidden.",
                    node,
                )
            if func_name in BANNED_MODULE_NAMES:
                self._record_violation(
                    f"Direct call to forbidden namespace '{func_name}' is forbidden.",
                    node,
                )
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in BANNED_METHODS:
                self._record_violation(
                    f"Call to dangerous method '{attr_name}' is forbidden.",
                    node,
                )
            if attr_name.startswith("__") and attr_name.endswith("__"):
                self._record_violation(
                    f"Call to dunder method '{attr_name}' is forbidden.",
                    node,
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Inspect attribute accesses (prevent dunder escaping and sandbox escape)."""
        attr_name = node.attr
        if attr_name.startswith("__"):
            self._record_violation(
                f"Access to dunder attribute '{attr_name}' is forbidden.",
                node,
            )
        if attr_name in BANNED_DUNDER_ATTRIBUTES:
            self._record_violation(
                f"Access to restricted introspection attribute '{attr_name}' is forbidden.",
                node,
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        """Inspect variable names, preventing access to banned builtins or module names."""
        if node.id.startswith("__") and node.id.endswith("__"):
            self._record_violation(
                f"Reference to dunder identifier '{node.id}' is forbidden.",
                node,
            )
        if node.id in BANNED_BUILTIN_NAMES and isinstance(node.ctx, (ast.Load, ast.Store)):
            self._record_violation(
                f"Reference to banned builtin '{node.id}' is forbidden.",
                node,
            )
        if node.id in BANNED_MODULE_NAMES and isinstance(node.ctx, (ast.Load, ast.Store)):
            self._record_violation(
                f"Reference to banned module namespace '{node.id}' is forbidden.",
                node,
            )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._record_violation("Async function definitions are prohibited.", node)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        self._record_violation("Async for-loops are prohibited.", node)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> Any:
        self._record_violation("Async with-statements are prohibited.", node)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> Any:
        self._record_violation("Global statements are prohibited inside sandbox.", node)
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> Any:
        self._record_violation("Nonlocal statements are prohibited inside sandbox.", node)
        self.generic_visit(node)

    def _record_violation(self, reason: str, node: ast.AST) -> None:
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)
        node_name = type(node).__name__
        loc = f"[Line {lineno}:{col_offset} - {node_name}]" if lineno is not None else f"[{node_name}]"
        self.violations.append(f"{loc} {reason}")


# =====================================================================
# Safe Builtins and Namespace Factory
# =====================================================================

def _create_safe_import() -> Callable[..., Any]:
    """Returns a restricted __import__ function that respects module whitelists."""
    orig_import = builtins.__import__

    def safe_import(
        name: str,
        globals: Optional[Dict[str, Any]] = None,
        locals: Optional[Dict[str, Any]] = None,
        fromlist: Tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0:
            raise ImportError("Relative imports are forbidden in the AST sandbox.")
        root_module = name.split(".")[0]
        if root_module not in ALLOWED_TOP_MODULES:
            raise ImportError(
                f"Import of module '{name}' is forbidden in the AST sandbox. Allowed: {sorted(ALLOWED_TOP_MODULES)}"
            )
        return orig_import(name, globals, locals, fromlist, level)

    return safe_import


def build_safe_builtins() -> Dict[str, Any]:
    """Constructs a hardened, immutable-friendly builtins dictionary."""
    safe_builtins: Dict[str, Any] = {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "bytes": bytes,
        "chr": chr,
        "complex": complex,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "frozenset": frozenset,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        # Exceptions
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "AttributeError": AttributeError,
        "AssertionError": AssertionError,
        "ZeroDivisionError": ZeroDivisionError,
        "StopIteration": StopIteration,
        "RuntimeError": RuntimeError,
        # Constants
        "True": True,
        "False": False,
        "None": None,
        # Safe import hook
        "__import__": _create_safe_import(),
    }
    return safe_builtins


def build_safe_globals(custom_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pre-populates the sandbox namespace with whitelisted libraries."""
    safe_globals: Dict[str, Any] = {
        "__builtins__": build_safe_builtins(),
        "np": np,
        "numpy": np,
        "math": math,
        "itertools": itertools,
    }
    if _SCIPY_AVAILABLE and scipy is not None:
        safe_globals["scipy"] = scipy
    if custom_context:
        for k, v in custom_context.items():
            if not k.startswith("__"):
                safe_globals[k] = v
    return safe_globals


# =====================================================================
# Safe Compiler
# =====================================================================

class SafeCompiler:
    """Validates and compiles source code within the AST sandbox boundaries."""

    def __init__(self, security_gate: Optional[ASTSecurityGate] = None) -> None:
        self.security_gate = security_gate or ASTSecurityGate()

    def compile(
        self,
        source_code: str,
        filename: str = "<sandboxed_world_model>",
    ) -> types.CodeType:
        """Parses, verifies, and compiles source code into a bytecode code object."""
        if not isinstance(source_code, str):
            raise TypeError(f"source_code must be str, got {type(source_code).__name__}")

        # 1. AST Security Gate Verification
        self.security_gate.validate(source_code)

        # 2. Bytecode compilation
        try:
            code_obj = compile(source_code, filename, "exec")
            return code_obj
        except SyntaxError as e:
            raise SandboxCompilationError(f"Syntax error compiling safe code: {e}") from e


# Alias for backward compatibility
SandboxCompiler = SafeCompiler


# =====================================================================
# Execution Results & Sandbox Executor
# =====================================================================

class ExecutionResult:
    """Encapsulates the execution output, return value, and telemetry."""

    def __init__(
        self,
        success: bool,
        return_value: Any = None,
        stdout: str = "",
        stderr: str = "",
        execution_time: float = 0.0,
        error: Optional[str] = None,
        timed_out: bool = False,
    ) -> None:
        self.success = success
        self.return_value = return_value
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time = execution_time
        self.error = error
        self.timed_out = timed_out

    def __repr__(self) -> str:
        status = "TIMEOUT" if self.timed_out else ("SUCCESS" if self.success else "FAILED")
        return (
            f"<ExecutionResult status={status} time={self.execution_time * 1000:.2f}ms "
            f"ret={type(self.return_value).__name__}>"
        )


def _worker_process_entrypoint(
    source_code: str,
    func_name: Optional[str],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    conn: Any,
) -> None:
    """Child process target for isolated execution."""
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = captured_stdout, captured_stderr

    start_t = time.perf_counter()
    try:
        # Pre-verify AST gate in child as defense-in-depth
        gate = ASTSecurityGate()
        gate.validate(source_code)

        code_obj = compile(source_code, "<child_sandbox>", "exec")
        safe_env = build_safe_globals()
        exec(code_obj, safe_env)

        if func_name:
            if func_name not in safe_env:
                raise NameError(f"Function '{func_name}' was not defined in the candidate code.")
            target_fn = safe_env[func_name]
            result = target_fn(*args, **kwargs)
        else:
            result = None

        elapsed = time.perf_counter() - start_t
        sys.stdout, sys.stderr = old_stdout, old_stderr
        conn.send({
            "success": True,
            "return_value": result,
            "stdout": captured_stdout.getvalue(),
            "stderr": captured_stderr.getvalue(),
            "execution_time": elapsed,
            "error": None,
            "timed_out": False,
        })
    except Exception as exc:
        elapsed = time.perf_counter() - start_t
        sys.stdout, sys.stderr = old_stdout, old_stderr
        conn.send({
            "success": False,
            "return_value": None,
            "stdout": captured_stdout.getvalue(),
            "stderr": captured_stderr.getvalue(),
            "execution_time": elapsed,
            "error": f"{type(exc).__name__}: {str(exc)}",
            "timed_out": False,
        })
    finally:
        conn.close()


class SandboxExecutor:
    """Safely executes candidate world model code under strict timeout constraints.

    Provides two primary execution modalities:
    1. Signal-based high-resolution timer (fastest, microsecond overhead for in-process step replay).
    2. Subprocess-isolated execution with SIGKILL (guarantees halting even against C extensions/GIL locks).
    """

    def __init__(
        self,
        compiler: Optional[SafeCompiler] = None,
        default_timeout: float = 3.0,
    ) -> None:
        self.compiler = compiler or SafeCompiler()
        self.default_timeout = max(0.01, float(default_timeout))

    def execute_code(
        self,
        source_code: str,
        custom_context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        isolate_process: bool = False,
    ) -> ExecutionResult:
        """Executes a code block and returns the environment and execution telemetry."""
        timeout_val = timeout if timeout is not None else self.default_timeout

        if isolate_process:
            return self._execute_in_process(
                source_code=source_code,
                func_name=None,
                args=(),
                kwargs={},
                timeout=timeout_val,
            )

        # 1. Compile & validate AST
        code_obj = self.compiler.compile(source_code)
        safe_env = build_safe_globals(custom_context)

        return self._execute_in_thread(
            callable_target=lambda: exec(code_obj, safe_env),
            timeout=timeout_val,
            return_env=safe_env,
        )

    def call_function(
        self,
        source_code_or_obj: Union[str, types.CodeType],
        func_name: str,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        isolate_process: bool = False,
    ) -> ExecutionResult:
        """Executes a specific named function within the sandboxed environment."""
        timeout_val = timeout if timeout is not None else self.default_timeout
        kwargs = kwargs or {}

        if isolate_process and isinstance(source_code_or_obj, str):
            return self._execute_in_process(
                source_code=source_code_or_obj,
                func_name=func_name,
                args=args,
                kwargs=kwargs,
                timeout=timeout_val,
            )

        # In-process execution
        if isinstance(source_code_or_obj, str):
            code_obj = self.compiler.compile(source_code_or_obj)
        else:
            code_obj = source_code_or_obj

        safe_env = build_safe_globals()

        # Run code setup
        exec_res = self._execute_in_thread(
            callable_target=lambda: exec(code_obj, safe_env),
            timeout=timeout_val,
        )
        if not exec_res.success:
            return exec_res

        if func_name not in safe_env:
            return ExecutionResult(
                success=False,
                error=f"Function '{func_name}' was not found in compiled namespace.",
                execution_time=exec_res.execution_time,
            )

        target_fn = safe_env[func_name]
        return self._execute_in_thread(
            callable_target=lambda: target_fn(*args, **kwargs),
            timeout=timeout_val,
        )

    def _execute_in_thread(
        self,
        callable_target: Callable[[], Any],
        timeout: float,
        return_env: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Executes a callable with strict signal/thread timeout enforcement."""
        result_holder: List[Any] = []
        error_holder: List[Exception] = []
        captured_stdout = io.StringIO()
        start_t = time.perf_counter()

        # Check if we can use high-resolution SIGALRM (POSIX main thread)
        can_use_sigalrm = (
            hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
            and threading.current_thread() is threading.main_thread()
        )

        if can_use_sigalrm:
            def _alarm_handler(signum: int, frame: Any) -> None:
                raise ExecutionTimeoutError(f"Execution timed out after {timeout:.3f}s")

            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            old_stdout = sys.stdout
            sys.stdout = captured_stdout
            try:
                ret = callable_target()
                elapsed = time.perf_counter() - start_t
                return ExecutionResult(
                    success=True,
                    return_value=return_env if return_env is not None else ret,
                    stdout=captured_stdout.getvalue(),
                    execution_time=elapsed,
                )
            except ExecutionTimeoutError as te:
                elapsed = time.perf_counter() - start_t
                return ExecutionResult(
                    success=False,
                    stdout=captured_stdout.getvalue(),
                    execution_time=elapsed,
                    error=str(te),
                    timed_out=True,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start_t
                return ExecutionResult(
                    success=False,
                    stdout=captured_stdout.getvalue(),
                    execution_time=elapsed,
                    error=f"{type(exc).__name__}: {str(exc)}",
                    timed_out=False,
                )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)
                sys.stdout = old_stdout

        # Fallback to Threading + Join timeout
        def _worker() -> None:
            old_stdout = sys.stdout
            sys.stdout = captured_stdout
            try:
                ret = callable_target()
                result_holder.append(ret)
            except Exception as exc:
                error_holder.append(exc)
            finally:
                sys.stdout = old_stdout

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout)
        elapsed = time.perf_counter() - start_t

        if t.is_alive():
            return ExecutionResult(
                success=False,
                stdout=captured_stdout.getvalue(),
                execution_time=elapsed,
                error=f"Execution exceeded timeout quota of {timeout:.3f}s",
                timed_out=True,
            )

        if error_holder:
            exc = error_holder[0]
            return ExecutionResult(
                success=False,
                stdout=captured_stdout.getvalue(),
                execution_time=elapsed,
                error=f"{type(exc).__name__}: {str(exc)}",
                timed_out=False,
            )

        ret_val = return_env if return_env is not None else (result_holder[0] if result_holder else None)
        return ExecutionResult(
            success=True,
            return_value=ret_val,
            stdout=captured_stdout.getvalue(),
            execution_time=elapsed,
        )

    def _execute_in_process(
        self,
        source_code: str,
        func_name: Optional[str],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        timeout: float,
    ) -> ExecutionResult:
        """Executes candidate code inside a dedicated OS process with guaranteed termination."""
        parent_conn, child_conn = multiprocessing.Pipe()
        start_t = time.perf_counter()

        proc = multiprocessing.Process(
            target=_worker_process_entrypoint,
            args=(source_code, func_name, args, kwargs, child_conn),
        )
        proc.start()
        proc.join(timeout=timeout)
        elapsed = time.perf_counter() - start_t

        if proc.is_alive():
            # Force kill hung process
            proc.terminate()
            time.sleep(0.01)
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=0.1)
            parent_conn.close()
            return ExecutionResult(
                success=False,
                execution_time=elapsed,
                error=f"Process terminated: exceeded timeout quota of {timeout:.3f}s",
                timed_out=True,
            )

        if parent_conn.poll():
            raw_data = parent_conn.recv()
            parent_conn.close()
            return ExecutionResult(
                success=raw_data["success"],
                return_value=raw_data["return_value"],
                stdout=raw_data["stdout"],
                stderr=raw_data["stderr"],
                execution_time=raw_data["execution_time"],
                error=raw_data["error"],
                timed_out=raw_data["timed_out"],
            )

        parent_conn.close()
        return ExecutionResult(
            success=False,
            execution_time=elapsed,
            error=f"Subprocess terminated unexpectedly with exitcode {proc.exitcode}",
            timed_out=False,
        )
