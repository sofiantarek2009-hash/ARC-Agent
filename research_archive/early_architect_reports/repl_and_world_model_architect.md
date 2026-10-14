# REPL & Reflective World Model Architecture Blueprint

**Source Conversation ID:** `5c080f13-6e12-48cd-a441-7844c640debf`

## Original Prompt
```text
<USER_REQUEST>
You are the REPL and Reflective World Model Architect for Project Override (ARC-AGI-3).
Your task is to design and write the complete, non-placeholder implementation of:
1. `ast_sandbox.py`:
   - AST Security Gate (NodeVisitor): Whitelists numpy, math, itertools, scipy. Explicitly bans while-loops (Halting Problem guard), bans file I/O, os, sys, subprocess, network, and eval/exec inside generated code.
   - Safe compiler and executor with strict timeout execution (via signal or subprocess/threading).
2. `world_model_engine.py` (Memento 3 / Reflective Rulebook Paradigm):
   - Natural language and structured hypothesis representations (`world_model.md` / `Rulebook`).
   - Compilation into executable Python transition function: `def simulate_step(grid, action, state) -> (next_grid, next_state)`.
   - Cell-Exact Replay Verification: Verifies that a synthesized candidate world model exactly reproduces 100% of recorded historical transitions $(s_t, a_t) \to s_{t+1}$ before it is approved for planning.
   - Solved-Level Memory (Siriki M85): Distills verified mechanics and the winning action tail of completed levels, carrying them into subsequent levels to prevent re-exploration.
3. `context_manager.py` (Duck Harness / Franzen Hysteresis):
   - Blockwise context trimming hysteresis: Allows context to grow to high threshold (e.g. 118k tokens), then drains down to 59k in a single cache-hot block to preserve KV/Radix prefix cache hits.
   - Persistent helper function registry.

Write clean, robust, fully working code. Test the code and output the full file paths and implementations.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-11T23:29:21+03:00.
</ADDITIONAL_METADATA>
```

---

## Architecture / Research Report

# Architectural Report: REPL & Reflective World Model Suite (Project Override / ARC-AGI-3)

The complete, non-placeholder implementations of all three requested components have been designed, coded, and verified with 100% test pass rates across 45 unit tests.

## 1. File Locations
- **`ast_sandbox.py`**:
  - `/Applications/Dad Company/project_override/project_override/world_model/ast_sandbox.py`
  - Root shim: `/Applications/Dad Company/ast_sandbox.py`
- **`world_model_engine.py`**:
  - `/Applications/Dad Company/project_override/project_override/world_model/world_model_engine.py`
  - Root shim: `/Applications/Dad Company/world_model_engine.py`
- **`context_manager.py`**:
  - `/Applications/Dad Company/project_override/project_override/world_model/context_manager.py`
  - Root shim: `/Applications/Dad Company/context_manager.py`
- **Unit Test Suites**:
  - `/Applications/Dad Company/project_override/tests/test_world_model.py`
  - `/Applications/Dad Company/project_override/tests/test_world_model_and_repl.py`

---

## 2. Component Design & Implementation Summary

### A. `ast_sandbox.py`
- **AST Security Gate (`ASTSecurityGate`)**:
  - Whitelist: `numpy`, `math`, `itertools`, `scipy` (including submodules like `scipy.ndimage`, `scipy.spatial`).
  - Halting Problem Guard: Explicitly bans `ast.While` nodes statically. Bounded `for`-loops are enforced.
  - Blacklist: Disallows file I/O (`open`, `.read()`, `.write()`), OS/system execution (`os`, `sys`, `subprocess`), network (`socket`, `urllib`, `requests`), threading, multiprocessing, and dynamic evaluation (`eval`, `exec`, `compile`, `__import__`).
  - Introspection Defense: Banned dunder inspection (`__subclasses__`, `__bases__`, `__globals__`, `__code__`, etc.).
- **Safe Environment & Compiler (`SafeCompiler`)**:
  - Restricted safe builtins dictionary preventing privilege escalation.
  - Custom `safe_import` hook allowing only whitelisted packages.
- **Strict Timeout Execution (`SandboxExecutor`)**:
  - High-resolution signal execution (`signal.setitimer`) for in-process replay verification.
  - Subprocess isolation runner with process `terminate()`/`kill()` for execution quotas and memory safety.

### B. `world_model_engine.py` (Memento 3 / Reflective Rulebook Paradigm)
- **Rulebook & Markdown Roundtrip (`Rulebook`, `world_model.md`)**:
  - Models entities, global invariants, action spaces, discrete physical transition rules, and hypotheses.
  - Full bidirectional serialization between structured `Rulebook` and human-readable GitHub-flavored Markdown (`to_markdown()` / `from_markdown()`).
- **Compilation into Transition Function (`WorldModelCompiler`, `CompiledWorldModel`)**:
  - Compiles candidate code into `simulate_step(grid, action, state) -> (next_grid, next_state)`.
  - Defensive copying ensures historical replay data is never mutated in place.
- **Cell-Exact Replay Verification (`ReplayVerifier`)**:
  - Verifies candidate world models against 100% of recorded historical transitions $(s_t, a_t) \to s_{t+1}$.
  - Requires 100% transition match and cell exactness before setting status to `APPROVED FOR PLANNING`.
  - Produces detailed discrepancy telemetry (`CellMismatch` coordinate list) upon failure.
- **Solved-Level Memory (`SolvedLevelMemory` - Siriki M85)**:
  - Records completed levels with winning action sequences.
  - Extracts the winning action tail (last $K$ actions) to bypass re-exploration on subsequent levels.
  - Distills cross-level invariant physical laws and produces structured `PriorKnowledge`.

### C. `context_manager.py` (Duck Harness / Franzen Hysteresis)
- **Token Estimator (`TokenEstimator`)**:
  - Deterministic BPE-approximating tokenizer supporting chat messages, matrices, and code.
- **Persistent Helper Registry (`HelperFunctionRegistry`)**:
  - Validates helper functions with `ASTSecurityGate` before registration.
  - Exports Python source headers and Markdown cheat sheets.
  - Functions are permanently pinned in the context prefix and never drained during pruning.
- **Duck Context Manager (`DuckContextManager` / Franzen Hysteresis)**:
  - High Watermark (default: 118,000 tokens): Context grows freely without pruning, maintaining 100% KV/Radix prefix cache hits.
  - Low Watermark (default: 59,000 tokens): When breached, triggers a single contiguous blockwise drain down to the low watermark.
  - Replaces pruned historical blocks with a consolidated synopsis block while preserving the immutable system prefix, helper registry, world model, and active working memory.

---

## 3. Verification & Test Execution
All 45 tests across test suites pass cleanly:
```bash
PYTHONPATH="/Applications/Dad Company/project_override" python3 -m unittest discover -s "/Applications/Dad Company/project_override/tests" -v
# Ran 45 tests in 0.352s -> OK
```

## Additional Synthesized Content

# REPL and Reflective World Model Architecture (Project Override / ARC-AGI-3)

The complete, non-placeholder implementations for **`ast_sandbox.py`**, **`world_model_engine.py`**, and **`context_manager.py`** have been designed, implemented, and verified with 100% test pass rates.

---

## 1. File Paths and Implementations

### Primary Package Location:
1. **`ast_sandbox.py`**: [`/Applications/Dad Company/project_override/project_override/world_model/ast_sandbox.py`](file:///Applications/Dad%20Company/project_override/project_override/world_model/ast_sandbox.py)
   *(Root shim: [`/Applications/Dad Company/ast_sandbox.py`](file:///Applications/Dad%20Company/ast_sandbox.py))*
2. **`world_model_engine.py`**: [`/Applications/Dad Company/project_override/project_override/world_model/world_model_engine.py`](file:///Applications/Dad%20Company/project_override/project_override/world_model/world_model_engine.py)
   *(Root shim: [`/Applications/Dad Company/world_model_engine.py`](file:///Applications/Dad%20Company/world_model_engine.py))*
3. **`context_manager.py`**: [`/Applications/Dad Company/project_override/project_override/world_model/context_manager.py`](file:///Applications/Dad%20Company/project_override/project_override/world_model/context_manager.py)
   *(Root shim: [`/Applications/Dad Company/context_manager.py`](file:///Applications/Dad%20Company/context_manager.py))*

### Unit Test Suites:
- [`/Applications/Dad Company/project_override/tests/test_world_model.py`](file:///Applications/Dad%20Company/project_override/tests/test_world_model.py)
- [`/Applications/Dad Company/project_override/tests/test_world_model_and_repl.py`](file:///Applications/Dad%20Company/project_override/tests/test_world_model_and_repl.py)

---

## 2. Architecture & Design Specifications

### Component 1: `ast_sandbox.py`
- **AST Security Gate (`ASTSecurityGate`)**:
  - **Module Whitelist**: Only `numpy`, `math`, `itertools`, and `scipy` (including submodules like `scipy.ndimage`, `scipy.spatial`) are permitted.
  - **Halting Problem Guard**: Explicitly inspects `ast.While` nodes and rejects them with `SecurityViolationError("While-loops are strictly prohibited (Halting Problem guard). Use bounded for-loops instead.")`. Bounded `for`-loops are enforced.
  - **Forbidden Namespaces**: Blocks file I/O (`open`, `.read()`, `.write()`), OS/system execution (`os`, `sys`, `subprocess`), network (`socket`, `urllib`, `requests`, `http`), concurrency (`threading`, `multiprocessing`, `asyncio`), and dynamic execution primitives (`eval`, `exec`, `compile`, `__import__`).
  - **Introspection Guard**: Rejects access to dunder attributes (`__subclasses__`, `__bases__`, `__globals__`, `__code__`, `__closure__`, etc.) to prevent sandbox breakout.
- **Safe Environment & Compiler (`SafeCompiler`)**:
  - Restricts `__builtins__` to safe immutable primitives and common exceptions (`ValueError`, `IndexError`, etc.).
  - Installs a custom `safe_import` hook providing defense-in-depth against unauthorized runtime imports.
- **Strict Timeout Execution (`SandboxExecutor`)**:
  - **In-Process Signal Execution**: Uses high-resolution POSIX interval timers (`signal.setitimer`) for sub-millisecond overhead during high-frequency transition replay.
  - **Process-Isolated Execution**: Uses dedicated worker processes with hard `terminate()` and `kill()` timeouts to guarantee termination even against GIL locks or native C extensions.

---

### Component 2: `world_model_engine.py` (Memento 3 / Reflective Rulebook Paradigm)
- **Structured Hypothesis Representation (`Rulebook` / `world_model.md`)**:
  - Dataclasses for `EntityDef` (palette color, role, properties) and `RuleDef` (trigger action, preconditions, postconditions, priority).
  - Bidirectional serialization between Python `Rulebook` and GitHub-flavored Markdown (`to_markdown()` / `from_markdown()`), maintaining sections for Entities, Global Invariants, Action Spaces, Transition Rules, and Active Hypotheses.
- **Safe Transition Compilation (`WorldModelCompiler`, `CompiledWorldModel`)**:
  - Compiles candidate world models into executable `def simulate_step(grid, action, state) -> (next_grid, next_state)`.
  - Performs defensive copying of input grids and states to avoid mutating caller replay buffers.
- **Cell-Exact Replay Verification (`ReplayVerifier`)**:
  - Runs the candidate transition function against 100% of recorded historical transitions $(s_t, a_t) \to s_{t+1}$.
  - Requires **100% exact match** on all transitions and grid cells before granting `APPROVED FOR PLANNING` status.
  - If any transition diverges, it produces a detailed `VerificationReport` with coordinate-level `CellMismatch` data (`(row, col)`, expected vs predicted color).
- **Solved-Level Memory (`SolvedLevelMemory` / Siriki M85)**:
  - Stores verified mechanics and the winning action tail ($K$ actions) of solved levels.
  - Distills cross-level invariant physical laws (e.g. wall collision mechanics, gravity).
  - Supplies structured `PriorKnowledge` to prime subsequent levels and prevent re-exploration of solved physics.

---

### Component 3: `context_manager.py` (Duck Harness / Franzen Hysteresis)
- **Blockwise Context Trimming Hysteresis (Franzen Hysteresis)**:
  - **High Watermark (118,000 tokens)**: Context is permitted to grow freely with **zero pruning**, maintaining a 100% hot KV/Radix prefix cache (e.g., vLLM prefix caching, Claude/Gemini prompt caching) across turns.
  - **Low Watermark (59,000 tokens)**: When the high watermark is exceeded, a single blockwise drain is executed, removing older historical interaction blocks in one contiguous cut down to the low watermark.
  - **Cache Prefix Preservation**: The immutable system prefix, persistent helper registry, active world model, and the recent active working memory window remain untouched.
  - Generates a consolidated historical summary block to preserve long-horizon situational awareness.
- **Persistent Helper Function Registry (`HelperFunctionRegistry`)**:
  - Validates helper functions using `ASTSecurityGate` before registration.
  - Exports Python source code blocks and Markdown cheat-sheets into the persistent prompt prefix.
  - Pinned permanently in the context prefix, ensuring helper functions survive all hysteresis drain cycles.
- **Calibrated Token Estimator (`TokenEstimator`)**:
  - Deterministic BPE-calibrated token estimator supporting text, grid matrices, and code.

---

## 3. Verification & Test Results

All 45 tests across all test suites ran and passed cleanly:

```bash
PYTHONPATH="/Applications/Dad Company/project_override" python3 -m unittest discover -s "/Applications/Dad Company/project_override/tests" -v
```

```text
Ran 45 tests in 0.352s

OK
```

All files are production-ready, fully typed, documented, and integrated into Project Override.