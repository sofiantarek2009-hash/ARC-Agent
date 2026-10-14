"""Project Override: World Model Engine (Memento 3 / Reflective Rulebook Paradigm)
ARC-AGI-3 Reflective Hypothesis & Transition Function Synthesizer.

Core Responsibilities:
1. Natural language and structured hypothesis representations (`world_model.md` / `Rulebook`).
2. Safe compilation into executable Python transition function:
   `def simulate_step(grid, action, state) -> (next_grid, next_state)`.
3. Cell-Exact Replay Verification:
   Verifies that a synthesized candidate world model exactly reproduces 100% of recorded
   historical transitions (s_t, a_t) -> s_{t+1} before it is approved for planning.
4. Solved-Level Memory (Siriki M85):
   Distills verified mechanics and the winning action tail of completed levels,
   carrying them into subsequent levels to prevent re-exploration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
import time
import types
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from project_override.world_model.ast_sandbox import (
    ASTSecurityGate,
    ExecutionResult,
    SafeCompiler,
    SandboxCompiler,
    SandboxExecutor,
    SandboxSecurityError,
    build_safe_globals,
)


# =====================================================================
# Data Structures: Rulebook & Entities
# =====================================================================

@dataclass
class EntityDef:
    """Definition of an object, agent, or environmental entity in the grid."""
    name: str
    color_id: int
    role: str  # e.g., "agent", "wall", "goal", "box", "hazard", "background"
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityDef:
        return cls(
            name=data["name"],
            color_id=int(data["color_id"]),
            role=data.get("role", "object"),
            properties=data.get("properties", {}),
        )


@dataclass
class RuleDef:
    """A discrete physical or behavioral rule within the Rulebook."""
    rule_id: str
    name: str
    condition: str
    effect: str
    trigger_action: Optional[Union[str, int]] = None
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuleDef:
        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            trigger_action=data.get("trigger_action"),
            condition=data["condition"],
            effect=data["effect"],
            priority=int(data.get("priority", 0)),
        )


@dataclass
class Rulebook:
    """Structured hypothesis representation of the environment physics."""
    domain_name: str
    level_id: str = "level_0"
    version: int = 1
    description: str = ""
    entities: Dict[str, EntityDef] = field(default_factory=dict)
    invariants: List[str] = field(default_factory=list)
    action_space: Dict[Union[int, str], str] = field(default_factory=dict)
    rules: List[RuleDef] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    transition_code: Optional[str] = None

    def add_entity(self, entity: EntityDef) -> None:
        self.entities[entity.name] = entity

    def add_invariant(self, invariant_text: str) -> None:
        if invariant_text not in self.invariants:
            self.invariants.append(invariant_text)

    def add_rule(self, rule: RuleDef) -> None:
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def to_markdown(self) -> str:
        """Serializes the Rulebook into clean, human-readable world_model.md format."""
        lines: List[str] = [
            f"# World Model: {self.domain_name}",
            "",
            "## Metadata",
            f"- **Level ID**: {self.level_id}",
            f"- **Version**: {self.version}",
            f"- **Description**: {self.description or 'None provided.'}",
            "",
            "## Entities & Palette",
            "| Name | Color ID | Role | Properties |",
            "| :--- | :---: | :--- | :--- |",
        ]

        if not self.entities:
            lines.append("| (none) | - | - | - |")
        else:
            for e in self.entities.values():
                props_str = json.dumps(e.properties) if e.properties else "{}"
                lines.append(f"| {e.name} | {e.color_id} | {e.role} | `{props_str}` |")

        lines.extend([
            "",
            "## Global Invariants",
        ])
        if not self.invariants:
            lines.append("- (no invariants specified)")
        else:
            for inv in self.invariants:
                lines.append(f"- {inv}")

        lines.extend([
            "",
            "## Action Space",
        ])
        if not self.action_space:
            lines.append("- Default: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=ACTION_USE")
        else:
            for act_k, act_name in sorted(self.action_space.items(), key=lambda x: str(x[0])):
                lines.append(f"- **{act_k}**: {act_name}")

        lines.extend([
            "",
            "## Rules & Transition Mechanics",
        ])
        if not self.rules:
            lines.append("- (no rules defined)")
        else:
            for r in self.rules:
                lines.append(f"### Rule {r.rule_id}: {r.name}")
                lines.append(f"- **Trigger Action**: {r.trigger_action}")
                lines.append(f"- **Condition**: {r.condition}")
                lines.append(f"- **Effect**: {r.effect}")
                lines.append(f"- **Priority**: {r.priority}")
                lines.append("")

        lines.extend([
            "## Active Hypotheses",
        ])
        if not self.hypotheses:
            lines.append("- (no active hypotheses)")
        else:
            for h in self.hypotheses:
                lines.append(f"- {h}")

        lines.extend([
            "",
            "## Executable Transition Function",
            "```python",
        ])
        if self.transition_code:
            lines.append(self.transition_code.strip())
        else:
            lines.append(
                "# def simulate_step(grid, action, state):\n"
                "#     # To be synthesized by world model engine\n"
                "#     return grid.copy(), state.copy() if state else {}"
            )
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, md_text: str) -> Rulebook:
        """Parses a world_model.md markdown string into a structured Rulebook."""
        lines = [line.strip() for line in md_text.splitlines()]

        # 1. Parse Title
        domain_name = "ARC-AGI Domain"
        for line in lines:
            if line.startswith("# World Model:"):
                domain_name = line.replace("# World Model:", "").strip()
                break

        # 2. Parse Metadata
        level_id = "level_0"
        version = 1
        description = ""
        for line in lines:
            if line.startswith("- **Level ID**:") or line.startswith("- Level ID:"):
                level_id = line.split(":", 1)[1].strip().strip("*")
            elif line.startswith("- **Version**:") or line.startswith("- Version:"):
                v_str = line.split(":", 1)[1].strip().strip("*")
                try:
                    version = int(v_str)
                except ValueError:
                    pass
            elif line.startswith("- **Description**:") or line.startswith("- Description:"):
                description = line.split(":", 1)[1].strip().strip("*")

        rulebook = cls(
            domain_name=domain_name,
            level_id=level_id,
            version=version,
            description=description,
        )

        # 3. Parse Entities Table
        in_entities = False
        for line in lines:
            if line.startswith("## Entities"):
                in_entities = True
                continue
            elif in_entities and line.startswith("## "):
                in_entities = False
            elif in_entities and line.startswith("|") and not line.startswith("| Name") and not line.startswith("| :"):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 3 and parts[0] != "(none)":
                    try:
                        c_id = int(parts[1])
                        props = json.loads(parts[3].strip("`")) if len(parts) > 3 and parts[3].startswith("`") else {}
                        rulebook.add_entity(EntityDef(
                            name=parts[0],
                            color_id=c_id,
                            role=parts[2],
                            properties=props,
                        ))
                    except (ValueError, json.JSONDecodeError):
                        pass

        # 4. Parse Global Invariants
        in_invariants = False
        for line in lines:
            if line.startswith("## Global Invariants"):
                in_invariants = True
                continue
            elif in_invariants and line.startswith("## "):
                in_invariants = False
            elif in_invariants and line.startswith("- "):
                inv = line[2:].strip()
                if inv != "(no invariants specified)":
                    rulebook.add_invariant(inv)

        # 5. Parse Action Space
        in_actions = False
        for line in lines:
            if line.startswith("## Action Space"):
                in_actions = True
                continue
            elif in_actions and line.startswith("## "):
                in_actions = False
            elif in_actions and line.startswith("- "):
                act_str = line[2:].strip().replace("**", "")
                if ":" in act_str:
                    k, v = act_str.split(":", 1)
                    k_clean = k.strip()
                    try:
                        act_key = int(k_clean)
                    except ValueError:
                        act_key = k_clean
                    rulebook.action_space[act_key] = v.strip()

        # 6. Parse Rules
        rule_pattern = re.compile(r"### Rule\s+([^:]+):\s*(.*)")
        current_rule_id = None
        current_rule_name = ""
        current_trigger: Optional[str] = None
        current_cond = ""
        current_effect = ""
        current_prio = 0

        in_rules = False
        for line in lines:
            if line.startswith("## Rules & Transition Mechanics"):
                in_rules = True
                continue
            elif in_rules and line.startswith("## "):
                # Save previous rule if pending
                if current_rule_id:
                    rulebook.add_rule(RuleDef(
                        rule_id=current_rule_id,
                        name=current_rule_name,
                        trigger_action=current_trigger,
                        condition=current_cond,
                        effect=current_effect,
                        priority=current_prio,
                    ))
                    current_rule_id = None
                in_rules = False
                continue

            if in_rules:
                m = rule_pattern.match(line)
                if m:
                    if current_rule_id:
                        rulebook.add_rule(RuleDef(
                            rule_id=current_rule_id,
                            name=current_rule_name,
                            trigger_action=current_trigger,
                            condition=current_cond,
                            effect=current_effect,
                            priority=current_prio,
                        ))
                    current_rule_id = m.group(1).strip()
                    current_rule_name = m.group(2).strip()
                    current_trigger = None
                    current_cond = ""
                    current_effect = ""
                    current_prio = 0
                elif current_rule_id:
                    if line.startswith("- **Trigger Action**:") or line.startswith("- Trigger Action:"):
                        current_trigger = line.split(":", 1)[1].strip()
                    elif line.startswith("- **Condition**:") or line.startswith("- Condition:"):
                        current_cond = line.split(":", 1)[1].strip()
                    elif line.startswith("- **Effect**:") or line.startswith("- Effect:"):
                        current_effect = line.split(":", 1)[1].strip()
                    elif line.startswith("- **Priority**:") or line.startswith("- Priority:"):
                        try:
                            current_prio = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass

        if current_rule_id:
            rulebook.add_rule(RuleDef(
                rule_id=current_rule_id,
                name=current_rule_name,
                trigger_action=current_trigger,
                condition=current_cond,
                effect=current_effect,
                priority=current_prio,
            ))

        # 7. Parse Hypotheses
        in_hypo = False
        for line in lines:
            if line.startswith("## Active Hypotheses"):
                in_hypo = True
                continue
            elif in_hypo and line.startswith("## "):
                in_hypo = False
            elif in_hypo and line.startswith("- "):
                h_text = line[2:].strip()
                if h_text != "(no active hypotheses)":
                    rulebook.hypotheses.append(h_text)

        # 8. Extract Python code block
        code_blocks = re.findall(r"```python\s*(.*?)\s*```", md_text, re.DOTALL)
        for cb in code_blocks:
            if "def simulate_step" in cb:
                rulebook.transition_code = cb.strip()
                break

        return rulebook


# =====================================================================
# Transition Function Compiler
# =====================================================================

class CompiledWorldModel:
    """Wrapper around a safely compiled transition step function."""

    def __init__(
        self,
        source_code: str,
        compiled_fn: Callable[..., Tuple[np.ndarray, Dict[str, Any]]],
        compiler: SafeCompiler,
        executor: SandboxExecutor,
    ) -> None:
        self.source_code = source_code
        self.compiled_fn = compiled_fn
        self.compiler = compiler
        self.executor = executor

    def simulate_step(
        self,
        grid: np.ndarray,
        action: Union[int, str],
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Executes a simulation step with defensive copying and type enforcement."""
        grid_copy = np.array(grid, copy=True)
        state_copy = dict(state) if state is not None else {}

        # Execute compiled transition function
        next_grid, next_state = self.compiled_fn(grid_copy, action, state_copy)

        # Enforce output specifications
        if not isinstance(next_grid, np.ndarray):
            next_grid = np.array(next_grid)
        if not isinstance(next_state, dict):
            next_state = dict(next_state) if next_state is not None else {}

        return next_grid, next_state


class WorldModelCompiler:
    """Compiles candidate Python transition functions through the AST sandbox."""

    def __init__(
        self,
        compiler: Optional[SafeCompiler] = None,
        executor: Optional[SandboxExecutor] = None,
    ) -> None:
        self.compiler = compiler or SafeCompiler()
        self.executor = executor or SandboxExecutor(compiler=self.compiler)

    def compile(self, code_or_rulebook: Union[str, Rulebook]) -> CompiledWorldModel:
        """Validates and compiles source code defining `simulate_step`."""
        if isinstance(code_or_rulebook, Rulebook):
            code_str = code_or_rulebook.transition_code or ""
        else:
            code_str = str(code_or_rulebook)

        if "def simulate_step" not in code_str:
            raise SandboxSecurityError("Candidate code must define 'def simulate_step(grid, action, state)'.")

        # 1. Compile through AST gate
        code_obj = self.compiler.compile(code_str)

        # 2. Build safe isolated environment
        safe_env = build_safe_globals()
        try:
            exec(code_obj, safe_env)
        except Exception as e:
            raise SandboxSecurityError(f"Error executing module setup in sandbox: {e}") from e

        if "simulate_step" not in safe_env or not callable(safe_env["simulate_step"]):
            raise SandboxSecurityError("Symbol 'simulate_step' is not a callable function.")

        compiled_fn = safe_env["simulate_step"]
        return CompiledWorldModel(
            source_code=code_str,
            compiled_fn=compiled_fn,
            compiler=self.compiler,
            executor=self.executor,
        )


# =====================================================================
# Cell-Exact Replay Verification
# =====================================================================

class RecordedTransition:
    """A recorded empirical transition observed in the ARC environment.

    Supports both keyword and positional arguments, with flexible orderings:
    - RecordedTransition(step_index, grid, action, state=s, next_grid=ng)
    - RecordedTransition(step_index, grid, action, next_grid, state, ...)
    - RecordedTransition(step_index, grid, action, state, next_grid, ...)
    """

    def __init__(
        self,
        step_index: int,
        grid: np.ndarray,
        action: Union[int, str],
        *args: Any,
        state: Optional[Dict[str, Any]] = None,
        next_grid: Optional[np.ndarray] = None,
        next_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.step_index = int(step_index)
        self.grid = np.array(grid, copy=False)
        self.action = action

        # Resolve from positional args if provided
        if len(args) > 0 and (next_grid is None or state is None):
            a0 = args[0]
            a1 = args[1] if len(args) > 1 else None
            a2 = args[2] if len(args) > 2 else None
            if isinstance(a0, np.ndarray):
                next_grid = a0
                if state is None:
                    state = a1 if isinstance(a1, dict) else {}
                if next_state is None and isinstance(a2, dict):
                    next_state = a2
            elif isinstance(a1, np.ndarray):
                if state is None:
                    state = a0 if isinstance(a0, dict) else {}
                next_grid = a1
                if next_state is None and isinstance(a2, dict):
                    next_state = a2
            else:
                if next_grid is None:
                    next_grid = a0
                if state is None:
                    state = a1

        self.next_grid = np.array(next_grid, copy=False) if next_grid is not None else np.zeros_like(self.grid)
        self.state = dict(state) if isinstance(state, dict) else {}
        self.next_state = dict(next_state) if isinstance(next_state, dict) else None
        self.metadata = metadata or kwargs.get("metadata", {})


@dataclass
class CellMismatch:
    """Details of a cell-level discrepancy between predicted and empirical states."""
    step_index: int
    row: int
    col: int
    expected_color: int
    predicted_color: int


@dataclass
class VerificationReport:
    """Result of cell-exact replay verification over historical transitions."""
    approved: bool
    total_transitions: int
    matched_transitions: int
    failed_transitions: int
    transition_accuracy: float
    cell_accuracy: float
    total_cells_checked: int
    total_cell_mismatches: int
    mismatches: List[CellMismatch]
    first_failure_step: Optional[int] = None
    execution_error: Optional[str] = None

    @property
    def summary(self) -> str:
        status = "APPROVED FOR PLANNING" if self.approved else "REJECTED (ACCURACY < 100%)"
        return (
            f"[{status}] Transitions: {self.matched_transitions}/{self.total_transitions} "
            f"({self.transition_accuracy * 100:.1f}%), Cell Accuracy: {self.cell_accuracy * 100:.2f}%, "
            f"Cell Discrepancies: {self.total_cell_mismatches}"
        )


class ReplayVerifier:
    """Verifies candidate world models against 100% of recorded historical transitions."""

    def __init__(self, timeout_per_step: float = 0.5) -> None:
        self.timeout_per_step = max(0.01, float(timeout_per_step))

    def verify(
        self,
        candidate_model: Union[CompiledWorldModel, Callable[..., Tuple[np.ndarray, Dict[str, Any]]]],
        transitions: List[RecordedTransition],
        max_reported_mismatches: int = 50,
    ) -> VerificationReport:
        """Runs full replay verification across all transitions."""
        total_transitions = len(transitions)
        if total_transitions == 0:
            return VerificationReport(
                approved=False,
                total_transitions=0,
                matched_transitions=0,
                failed_transitions=0,
                transition_accuracy=0.0,
                cell_accuracy=0.0,
                total_cells_checked=0,
                total_cell_mismatches=0,
                mismatches=[],
                execution_error="No transitions provided for replay verification.",
            )

        sim_fn = candidate_model.simulate_step if isinstance(candidate_model, CompiledWorldModel) else candidate_model

        matched_transitions = 0
        mismatches: List[CellMismatch] = []
        total_cells_checked = 0
        total_cell_mismatches = 0
        first_failure_step: Optional[int] = None
        fatal_error: Optional[str] = None

        for t in transitions:
            try:
                pred_grid, pred_state = sim_fn(
                    grid=t.grid,
                    action=t.action,
                    state=t.state,
                )
            except Exception as exc:
                fatal_error = f"Step {t.step_index} execution error: {type(exc).__name__}: {str(exc)}"
                if first_failure_step is None:
                    first_failure_step = t.step_index
                break

            # Validate grid shape compatibility
            if pred_grid.shape != t.next_grid.shape:
                fatal_error = (
                    f"Step {t.step_index} grid shape mismatch: predicted {pred_grid.shape}, "
                    f"expected {t.next_grid.shape}"
                )
                if first_failure_step is None:
                    first_failure_step = t.step_index
                break

            cells_in_grid = t.next_grid.size
            total_cells_checked += cells_in_grid

            # Cell-by-cell discrepancy analysis
            diff_mask = pred_grid != t.next_grid
            diff_count = int(np.count_nonzero(diff_mask))

            if diff_count == 0:
                matched_transitions += 1
            else:
                total_cell_mismatches += diff_count
                if first_failure_step is None:
                    first_failure_step = t.step_index

                # Log cell mismatch coordinates
                diff_rows, diff_cols = np.where(diff_mask)
                for r, c in zip(diff_rows, diff_cols):
                    if len(mismatches) < max_reported_mismatches:
                        mismatches.append(CellMismatch(
                            step_index=t.step_index,
                            row=int(r),
                            col=int(c),
                            expected_color=int(t.next_grid[r, c]),
                            predicted_color=int(pred_grid[r, c]),
                        ))

        failed_transitions = total_transitions - matched_transitions
        transition_accuracy = matched_transitions / total_transitions if total_transitions > 0 else 0.0
        cell_accuracy = (
            (total_cells_checked - total_cell_mismatches) / total_cells_checked
            if total_cells_checked > 0
            else 0.0
        )

        # 100% cell-exact match requirement for planning approval
        approved = (matched_transitions == total_transitions) and (fatal_error is None)

        return VerificationReport(
            approved=approved,
            total_transitions=total_transitions,
            matched_transitions=matched_transitions,
            failed_transitions=failed_transitions,
            transition_accuracy=transition_accuracy,
            cell_accuracy=cell_accuracy,
            total_cells_checked=total_cells_checked,
            total_cell_mismatches=total_cell_mismatches,
            mismatches=mismatches,
            first_failure_step=first_failure_step,
            execution_error=fatal_error,
        )


# =====================================================================
# Solved-Level Memory (Siriki M85)
# =====================================================================

@dataclass
class SolvedLevelRecord:
    """Historical record of an environment level successfully solved."""
    level_id: str
    rulebook: Rulebook
    transition_code: str
    winning_action_sequence: List[Union[int, str]]
    winning_action_tail: List[Union[int, str]]
    distilled_mechanics: List[str]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriorKnowledge:
    """Distilled cross-level priors to accelerate solving new levels."""
    cross_level_invariants: List[str]
    distilled_rules: List[RuleDef]
    winning_tails: Dict[str, List[Union[int, str]]]
    universal_action_space: Dict[Union[int, str], str]
    base_transition_template: str


class SolvedLevelMemory:
    """Distills verified mechanics and winning action tails across completed levels.

    Prevents re-exploration in subsequent levels by transferring verified mechanics
    and tactical action tails (Siriki M85 paradigm).
    """

    def __init__(self, default_tail_length: int = 8) -> None:
        self.default_tail_length = default_tail_length
        self.records: Dict[str, SolvedLevelRecord] = {}

    def register_solved_level(
        self,
        level_id: str,
        rulebook: Rulebook,
        transition_code: str,
        winning_actions: List[Union[int, str]],
        tail_length: Optional[int] = None,
        distilled_mechanics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SolvedLevelRecord:
        """Registers a completed level into memory and extracts winning action tail."""
        k = tail_length if tail_length is not None else self.default_tail_length
        winning_tail = list(winning_actions[-k:]) if len(winning_actions) > 0 else []

        mechanics = distilled_mechanics or list(rulebook.invariants)

        record = SolvedLevelRecord(
            level_id=level_id,
            rulebook=rulebook,
            transition_code=transition_code,
            winning_action_sequence=list(winning_actions),
            winning_action_tail=winning_tail,
            distilled_mechanics=mechanics,
            metadata=metadata or {},
        )
        self.records[level_id] = record
        return record

    def get_level(self, level_id: str) -> Optional[SolvedLevelRecord]:
        return self.records.get(level_id)

    def distill_cross_level_invariants(self) -> List[str]:
        """Extracts invariants and mechanics confirmed across solved levels."""
        if not self.records:
            return []

        # Count occurrences of each invariant across all solved levels
        invariant_counts: Dict[str, int] = {}
        for rec in self.records.values():
            for inv in rec.distilled_mechanics:
                invariant_counts[inv] = invariant_counts.get(inv, 0) + 1

        # Sorted by frequency
        sorted_invariants = sorted(
            invariant_counts.keys(),
            key=lambda inv: invariant_counts[inv],
            reverse=True,
        )
        return sorted_invariants

    def get_prior_for_level(self, target_level_id: str) -> PriorKnowledge:
        """Provides transferred knowledge and winning tails to prime a new level."""
        invariants = self.distill_cross_level_invariants()

        winning_tails: Dict[str, List[Union[int, str]]] = {
            lid: rec.winning_action_tail for lid, rec in self.records.items()
        }

        # Consolidate universal action space
        universal_actions: Dict[Union[int, str], str] = {}
        for rec in self.records.values():
            for act_k, act_v in rec.rulebook.action_space.items():
                universal_actions[act_k] = act_v

        # Collect unique verified rules
        known_rule_ids: Set[str] = set()
        distilled_rules: List[RuleDef] = []
        for rec in self.records.values():
            for rule in rec.rulebook.rules:
                if rule.rule_id not in known_rule_ids:
                    known_rule_ids.add(rule.rule_id)
                    distilled_rules.append(rule)

        # Base transition template from most recently solved level
        base_code = ""
        if self.records:
            last_record = list(self.records.values())[-1]
            base_code = last_record.transition_code

        return PriorKnowledge(
            cross_level_invariants=invariants,
            distilled_rules=distilled_rules,
            winning_tails=winning_tails,
            universal_action_space=universal_actions,
            base_transition_template=base_code,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes memory into JSON-compatible dictionary."""
        out: Dict[str, Any] = {}
        for lid, rec in self.records.items():
            out[lid] = {
                "level_id": rec.level_id,
                "rulebook_md": rec.rulebook.to_markdown(),
                "transition_code": rec.transition_code,
                "winning_actions": rec.winning_action_sequence,
                "winning_tail": rec.winning_action_tail,
                "distilled_mechanics": rec.distilled_mechanics,
                "timestamp": rec.timestamp,
                "metadata": rec.metadata,
            }
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SolvedLevelMemory:
        """Restores memory from dictionary."""
        mem = cls()
        for lid, item in data.items():
            rb = Rulebook.from_markdown(item["rulebook_md"])
            rec = SolvedLevelRecord(
                level_id=lid,
                rulebook=rb,
                transition_code=item["transition_code"],
                winning_action_sequence=item["winning_actions"],
                winning_action_tail=item["winning_tail"],
                distilled_mechanics=item["distilled_mechanics"],
                timestamp=item.get("timestamp", time.time()),
                metadata=item.get("metadata", {}),
            )
            mem.records[lid] = rec
        return mem
