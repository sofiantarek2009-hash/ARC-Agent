"""Project Override: Context Manager (Duck Harness / Franzen Hysteresis)
ARC-AGI-3 Long-Horizon REPL Context Orchestration.

Core Responsibilities:
1. Blockwise Context Trimming Hysteresis (Franzen Hysteresis):
   - Allows context to grow up to a high watermark (e.g. 118,000 tokens) with ZERO pruning,
     maintaining a 100% hot KV/Radix prefix cache across intermediate interaction turns.
   - When the high watermark is breached, triggers a single blockwise drain down to the
     low watermark (e.g. 59,000 tokens), consolidating older historical turns while
     preserving the immutable system prefix, persistent helper registry, and recent active working memory.
   - After drain, the new prefix stays stable for another ~59k tokens of turns.
2. Persistent Helper Function Registry:
   - Registers, validates via ASTSecurityGate, and maintains domain helper functions
     (e.g., flood fill, component labeling, coordinate transforms) across turns.
   - Guarantees helper functions remain persistent across all hysteresis drain cycles.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from enum import Enum
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from project_override.world_model.ast_sandbox import ASTSecurityGate, SecurityViolationError


# =====================================================================
# High-Fidelity Token Estimator
# =====================================================================

class TokenEstimator:
    """Estimates token counts for text, structured code, and ARC matrices.

    Calibrated against standard BPE tokenizers with whitespace, punctuation,
    and code syntax adjustments. Accepts an optional custom tokenizer callback.
    """

    def __init__(self, custom_tokenizer: Optional[Callable[[str], int]] = None) -> None:
        self.custom_tokenizer = custom_tokenizer
        # Pre-compiled BPE-approximate regex compatible with Python stdlib re
        self._pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|\w+|\d+|[^\s\w]+|\s+"""
        )

    def count_tokens(self, text: str) -> int:
        """Returns the token count for a text string."""
        if not text:
            return 0
        if self.custom_tokenizer is not None:
            return max(1, self.custom_tokenizer(text))

        # Fallback calibrated tokenizer
        # Fast regex token splitting matching byte-pair encoding boundaries
        tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
        count = 0
        for tok in tokens:
            if tok.isspace():
                # Consecutive newlines/spaces count as fewer tokens
                count += max(1, len(tok) // 4)
            elif len(tok) > 6:
                # Long words / tokens typically split into multiple subwords
                count += (len(tok) + 3) // 4
            else:
                count += 1
        return max(1, count)

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Counts total tokens across a list of chat messages."""
        total = 0
        for msg in messages:
            total += 4  # Per-message overhead (role, markers)
            total += self.count_tokens(msg.get("content", ""))
            total += self.count_tokens(msg.get("role", ""))
        total += 2  # End of message sequence overhead
        return total


# =====================================================================
# Persistent Helper Function Registry
# =====================================================================

@dataclass
class HelperFunction:
    """A registered helper function persistently available to the agent REPL."""
    name: str
    code: str
    signature: str
    docstring: str
    tags: List[str] = field(default_factory=list)
    call_count: int = 0
    registered_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HelperFunctionRegistry:
    """Manages persistent utility functions injected into the agent context.

    Enforces that helper functions pass ASTSecurityGate validation before registration,
    and preserves them across context pruning cycles.
    """

    def __init__(self, security_gate: Optional[ASTSecurityGate] = None) -> None:
        self.security_gate = security_gate or ASTSecurityGate()
        self.functions: Dict[str, HelperFunction] = {}

    def register(
        self,
        name: str,
        code: str,
        tags: Optional[List[str]] = None,
    ) -> HelperFunction:
        """Validates code via ASTSecurityGate and registers the helper function."""
        clean_code = code.strip()

        # 1. AST Security Gate Validation
        self.security_gate.validate(clean_code)

        # 2. Extract function AST metadata
        tree = ast.parse(clean_code)
        func_node: Optional[ast.FunctionDef] = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                func_node = node
                break

        if func_node is None:
            # If function name didn't match directly, take the first function def
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_node = node
                    name = node.name
                    break

        if func_node is None:
            raise ValueError(f"No valid function definition found in code for '{name}'.")

        # 3. Extract docstring
        docstring = ast.get_docstring(func_node) or "No docstring provided."

        # 4. Extract signature
        args_list: List[str] = []
        for arg in func_node.args.args:
            args_list.append(arg.arg)
        sig = f"{name}({', '.join(args_list)})"

        helper = HelperFunction(
            name=name,
            code=clean_code,
            signature=sig,
            docstring=docstring,
            tags=tags or [],
        )
        self.functions[name] = helper
        return helper

    def unregister(self, name: str) -> bool:
        """Removes a helper function from the registry."""
        return self.functions.pop(name, None) is not None

    def get(self, name: str) -> Optional[HelperFunction]:
        return self.functions.get(name)

    def list_functions(self) -> List[HelperFunction]:
        return list(self.functions.values())

    def export_code_block(self) -> str:
        """Exports all registered helper functions into a single Python source block."""
        if not self.functions:
            return "# No persistent helper functions currently registered."

        parts: List[str] = [
            "# === PERSISTENT HELPER FUNCTION REGISTRY ===",
            "# Verified functions available in current and subsequent turns.",
            "",
        ]
        for h in sorted(self.functions.values(), key=lambda f: f.name):
            parts.append(h.code)
            parts.append("")
        return "\n".join(parts)

    def export_markdown_reference(self) -> str:
        """Exports a markdown cheat-sheet documenting available helper functions."""
        if not self.functions:
            return "- (no helper functions registered)"

        lines: List[str] = []
        for h in sorted(self.functions.values(), key=lambda f: f.name):
            tag_str = f" `[{', '.join(h.tags)}]`" if h.tags else ""
            lines.append(f"- **`{h.signature}`**{tag_str}: {h.docstring.splitlines()[0]}")
        return "\n".join(lines)


# =====================================================================
# Context Blocks & Hysteresis Structures
# =====================================================================

class BlockType(str, Enum):
    SYSTEM_PREFIX = "system_prefix"          # Immutable cache root
    HELPER_REGISTRY = "helper_registry"      # Persistent code tools
    WORLD_MODEL = "world_model"              # Reflective rulebook
    SOLVED_MEMORY = "solved_memory"          # Transferred level mechanics
    HISTORICAL_SUMMARY = "historical_summary"# Drained turn synopsis
    INTERACTION_TURN = "interaction_turn"    # Active conversation/action turns


@dataclass
class ContextBlock:
    """A distinct unit of context with cache properties and token weighting."""
    block_id: str
    block_type: BlockType
    role: str  # "system", "user", "assistant"
    content: str
    token_count: int
    timestamp: float = field(default_factory=time.time)
    is_pinned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DrainEvent:
    """Telemetry recorded when a blockwise hysteresis drain is executed."""
    timestamp: float
    tokens_before: int
    tokens_after: int
    tokens_reclaimed: int
    blocks_removed: int
    high_watermark: int
    low_watermark: int
    summary_generated: bool


# =====================================================================
# Duck Harness Context Manager (Franzen Hysteresis)
# =====================================================================

class DuckContextManager:
    """Orchestrates long-horizon context growth and pruning via Franzen Hysteresis.

    Mechanism:
    - High Watermark (default: 118k tokens): Context grows freely without pruning.
      Every turn appended preserves 100% of the KV/Radix prefix cache hit rate.
    - Low Watermark (default: 59k tokens): When high watermark is exceeded, a single
      blockwise drain is executed, removing historical interaction blocks in one cut.
    - Persistent Registry: Helper functions and immutable prefix are permanently preserved.
    """

    def __init__(
        self,
        high_watermark_tokens: int = 118_000,
        low_watermark_tokens: int = 59_000,
        min_active_turns: int = 6,
        token_estimator: Optional[TokenEstimator] = None,
        helper_registry: Optional[HelperFunctionRegistry] = None,
    ) -> None:
        if low_watermark_tokens >= high_watermark_tokens:
            raise ValueError("low_watermark_tokens must be strictly less than high_watermark_tokens.")

        self.high_watermark_tokens = int(high_watermark_tokens)
        self.low_watermark_tokens = int(low_watermark_tokens)
        self.min_active_turns = max(2, int(min_active_turns))

        self.estimator = token_estimator or TokenEstimator()
        self.registry = helper_registry or HelperFunctionRegistry()

        # Core context blocks
        self.system_prefix: Optional[ContextBlock] = None
        self.world_model_block: Optional[ContextBlock] = None
        self.solved_memory_block: Optional[ContextBlock] = None
        self.summary_block: Optional[ContextBlock] = None

        # Historical & active interaction blocks
        self.interaction_blocks: List[ContextBlock] = []

        # Telemetry
        self.drain_history: List[DrainEvent] = []
        self._block_counter: int = 0

    def _next_block_id(self, prefix: str) -> str:
        self._block_counter += 1
        return f"{prefix}_{self._block_counter:05d}"

    def set_system_prefix(self, prompt_text: str) -> None:
        """Sets the immutable cache prefix (root instructions)."""
        tokens = self.estimator.count_tokens(prompt_text)
        self.system_prefix = ContextBlock(
            block_id="system_prefix_000",
            block_type=BlockType.SYSTEM_PREFIX,
            role="system",
            content=prompt_text,
            token_count=tokens,
            is_pinned=True,
        )

    def update_world_model(self, rulebook_md: str) -> None:
        """Updates the active World Model rulebook block."""
        tokens = self.estimator.count_tokens(rulebook_md)
        self.world_model_block = ContextBlock(
            block_id=self._next_block_id("wm"),
            block_type=BlockType.WORLD_MODEL,
            role="system",
            content=f"## ACTIVE WORLD MODEL RULEBOOK\n{rulebook_md}",
            token_count=tokens,
            is_pinned=True,
        )

    def update_solved_memory(self, memory_text: str) -> None:
        """Updates the Solved-Level Memory transfer block."""
        tokens = self.estimator.count_tokens(memory_text)
        self.solved_memory_block = ContextBlock(
            block_id=self._next_block_id("mem"),
            block_type=BlockType.SOLVED_MEMORY,
            role="system",
            content=f"## SOLVED-LEVEL MEMORY & MECHANICS\n{memory_text}",
            token_count=tokens,
            is_pinned=True,
        )

    def add_interaction(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[DrainEvent]]:
        """Appends a new interaction turn and checks Franzen Hysteresis thresholds.

        Returns:
            (drain_triggered, Optional[DrainEvent])
        """
        tokens = self.estimator.count_tokens(content)
        block = ContextBlock(
            block_id=self._next_block_id("turn"),
            block_type=BlockType.INTERACTION_TURN,
            role=role,
            content=content,
            token_count=tokens,
            is_pinned=False,
            metadata=metadata or {},
        )
        self.interaction_blocks.append(block)

        # Check Franzen Hysteresis threshold
        current_total = self.get_total_tokens()
        if current_total > self.high_watermark_tokens:
            drain_event = self._perform_blockwise_drain()
            return True, drain_event

        return False, None

    def append_interaction(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[DrainEvent]]:
        """Alias for add_interaction for consistency with test harness."""
        return self.add_interaction(role=role, content=content, metadata=metadata)

    def get_total_tokens(self) -> int:
        """Computes current token total across all persistent and interaction blocks."""
        total = 0
        if self.system_prefix:
            total += self.system_prefix.token_count

        # Helper registry block tokens
        helper_code = self.registry.export_code_block()
        total += self.estimator.count_tokens(helper_code)

        if self.world_model_block:
            total += self.world_model_block.token_count
        if self.solved_memory_block:
            total += self.solved_memory_block.token_count
        if self.summary_block:
            total += self.summary_block.token_count

        for b in self.interaction_blocks:
            total += b.token_count

        return total

    def _perform_blockwise_drain(self) -> DrainEvent:
        """Executes a single blockwise drain from high watermark down to low watermark.

        Consolidates contiguous older interaction blocks while leaving the recent
        active working memory and persistent prefixes untouched.
        """
        before_tokens = self.get_total_tokens()
        tokens_to_drain = before_tokens - self.low_watermark_tokens

        if len(self.interaction_blocks) <= 1:
            # Cannot safely prune when only 1 active turn exists
            return DrainEvent(
                timestamp=time.time(),
                tokens_before=before_tokens,
                tokens_after=before_tokens,
                tokens_reclaimed=0,
                blocks_removed=0,
                high_watermark=self.high_watermark_tokens,
                low_watermark=self.low_watermark_tokens,
                summary_generated=False,
            )

        # Allow pruning older turns, preserving at least 1 recent active turn
        # Default preference: keep min_active_turns if possible
        max_prunable_idx = max(1, len(self.interaction_blocks) - 1)
        target_prunable_idx = max(1, len(self.interaction_blocks) - self.min_active_turns)
        if target_prunable_idx > max_prunable_idx:
            target_prunable_idx = max_prunable_idx

        pruned_blocks: List[ContextBlock] = []
        drained_token_sum = 0
        cut_idx = 0

        # Pass 1: Prune up to target_prunable_idx
        for i in range(max_prunable_idx):
            b = self.interaction_blocks[i]
            pruned_blocks.append(b)
            drained_token_sum += b.token_count
            cut_idx = i + 1
            if (before_tokens - drained_token_sum) <= self.low_watermark_tokens and i >= (target_prunable_idx - 1):
                break

        # Only generate summary if it's materially smaller than the drained tokens
        summary_text = self._synthesize_drain_summary(pruned_blocks)
        summary_tokens = self.estimator.count_tokens(summary_text)

        summary_added = False
        if drained_token_sum > (summary_tokens + 10):
            self.summary_block = ContextBlock(
                block_id=self._next_block_id("summary"),
                block_type=BlockType.HISTORICAL_SUMMARY,
                role="system",
                content=f"## CONSOLIDATED HISTORICAL SYNOPSIS\n{summary_text}",
                token_count=summary_tokens,
                is_pinned=True,
            )
            summary_added = True
        else:
            # Skip summary block to avoid inflating tokens
            self.summary_block = None

        # Slice away the pruned blocks
        self.interaction_blocks = self.interaction_blocks[cut_idx:]

        after_tokens = self.get_total_tokens()
        reclaimed = before_tokens - after_tokens

        event = DrainEvent(
            timestamp=time.time(),
            tokens_before=before_tokens,
            tokens_after=after_tokens,
            tokens_reclaimed=reclaimed,
            blocks_removed=len(pruned_blocks),
            high_watermark=self.high_watermark_tokens,
            low_watermark=self.low_watermark_tokens,
            summary_generated=summary_added,
        )
        self.drain_history.append(event)
        return event

    def _synthesize_drain_summary(self, blocks: List[ContextBlock]) -> str:
        """Synthesizes a concise summary of pruned interaction turns."""
        return f"Archived {len(blocks)} turns of previous exploration and simulation steps."

    def render_messages(self) -> List[Dict[str, str]]:
        """Renders the context into a list of messages formatted for standard model APIs."""
        messages: List[Dict[str, str]] = []

        # 1. System Prefix (Cache Root)
        if self.system_prefix:
            messages.append({
                "role": self.system_prefix.role,
                "content": self.system_prefix.content,
            })

        # 2. Helper Function Registry (Persistent Prefix Extension)
        helper_code = self.registry.export_code_block()
        if helper_code:
            messages.append({
                "role": "system",
                "content": helper_code,
            })

        # 3. Active World Model Block
        if self.world_model_block:
            messages.append({
                "role": self.world_model_block.role,
                "content": self.world_model_block.content,
            })

        # 4. Solved Level Memory Block
        if self.solved_memory_block:
            messages.append({
                "role": self.solved_memory_block.role,
                "content": self.solved_memory_block.content,
            })

        # 5. Consolidated Summary (if drain has occurred)
        if self.summary_block:
            messages.append({
                "role": self.summary_block.role,
                "content": self.summary_block.content,
            })

        # 6. Active Interaction Turns
        for b in self.interaction_blocks:
            messages.append({
                "role": b.role,
                "content": b.content,
            })

        return messages

    def render_full_prompt(self) -> str:
        """Renders the entire context into a single concatenated prompt string."""
        msgs = self.render_messages()
        rendered: List[str] = []
        for m in msgs:
            role_header = f"=== {m['role'].upper()} ==="
            rendered.append(role_header)
            rendered.append(m["content"])
            rendered.append("")
        return "\n".join(rendered)

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic telemetry regarding context utilization and hysteresis."""
        total_tokens = self.get_total_tokens()
        headroom = max(0, self.high_watermark_tokens - total_tokens)
        percent_full = (total_tokens / self.high_watermark_tokens) * 100.0

        return {
            "current_tokens": total_tokens,
            "high_watermark": self.high_watermark_tokens,
            "low_watermark": self.low_watermark_tokens,
            "headroom_tokens": headroom,
            "capacity_pct": percent_full,
            "active_interaction_turns": len(self.interaction_blocks),
            "registered_helper_functions": len(self.registry.functions),
            "total_drains_executed": len(self.drain_history),
            "last_drain": asdict(self.drain_history[-1]) if self.drain_history else None,
        }
