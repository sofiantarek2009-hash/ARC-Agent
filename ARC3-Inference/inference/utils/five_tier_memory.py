"""Archimedes-3 Five-Tier Prefix-Pinned Memory & Cross-Level Schema Transfer Engine.

Formalized from Section 11 (Memory, Transfer, and Forgetting) of ARCHITECTURAL_BLUEPRINT_ARCHIMEDES_3.md.
Maintains strictly append-only episodic tape, state graphs, and Bayesian rule posteriors with cross-level transfer decay (prior=0.85).
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class VerifiedRule:
    """Semantic causal rule with status and Bayesian confidence."""
    rule_id: str
    description: str
    status: str  # "VALIDATED", "PROBABLE", "SPECULATIVE", "FALSIFIED"
    confidence: float = 1.0
    retrodiction_accuracy: float = 1.0


class FiveTierMemoryManager:
    """Manages the 5 memory tiers of Archimedes-3:
    1. Working Memory (active prompt tail)
    2. Episodic Memory (lossless transition tape & visited state graph)
    3. Semantic/Causal Memory (verified rules & goal predicates)
    4. Procedural Memory (retained solver functions)
    5. Negative/Falsification Memory (FatalLedger & NoOpLedger)
    """

    def __init__(self, current_game_id: str = "") -> None:
        self.current_game_id = current_game_id
        self.current_level: int = 1

        # Tier 2: Episodic Memory
        self.episodic_tape: list[dict[str, Any]] = []
        self.visited_state_graph: set[int] = set()

        # Tier 3: Semantic / Causal Memory
        self.semantic_rules: dict[str, VerifiedRule] = {}
        self.candidate_goals: list[dict[str, Any]] = []

        # Tier 5: Negative Memory
        self.fatal_ledger: set[tuple[int, str]] = set()  # (state_hash, action)
        self.noop_ledger: set[tuple[int, str]] = set()   # (state_hash, action)

    def record_transition(
        self,
        turn_idx: int,
        state_hash: int,
        action: str,
        next_state_hash: int,
        gameplay_changed: bool,
        is_death: bool = False,
        is_level_completed: bool = False,
    ) -> None:
        """Appends transition to Episodic Tape and updates state graph & negative ledgers."""
        entry = {
            "turn": turn_idx,
            "level": self.current_level,
            "state_hash": state_hash,
            "action": action,
            "next_state_hash": next_state_hash,
            "gameplay_changed": gameplay_changed,
            "is_death": is_death,
            "is_level_completed": is_level_completed,
        }
        self.episodic_tape.append(entry)
        self.visited_state_graph.add(state_hash)
        self.visited_state_graph.add(next_state_hash)

        if is_death:
            self.fatal_ledger.add((state_hash, action))
        elif not gameplay_changed:
            self.noop_ledger.add((state_hash, action))

    def is_known_fatal(self, state_hash: int, action: str) -> bool:
        """Checks Tier 5 FatalLedger."""
        return (state_hash, action) in self.fatal_ledger

    def is_known_noop(self, state_hash: int, action: str) -> bool:
        """Checks Tier 5 NoOpLedger."""
        return (state_hash, action) in self.noop_ledger

    def register_rule(self, rule_id: str, description: str, status: str = "VALIDATED") -> None:
        """Registers a verified causal rule into Semantic Memory."""
        self.semantic_rules[rule_id] = VerifiedRule(
            rule_id=rule_id,
            description=description,
            status=status,
            confidence=1.0 if status == "VALIDATED" else 0.85,
        )

    def advance_level(self, new_level: int) -> None:
        """Applies Cross-Level Transfer with Bayesian Decay (Section 11.2).
        
        - Resets level-specific NoOpLedger.
        - Preserves FatalLedger.
        - Downgrades VALIDATED rules to PROBABLE (prior=0.85) until re-confirmed in new level.
        """
        self.current_level = new_level
        self.noop_ledger.clear()

        # Decay validated rules to PROBABLE
        for rule in self.semantic_rules.values():
            if rule.status == "VALIDATED":
                rule.status = "PROBABLE"
                rule.confidence = 0.85

    def wipe_cross_game(self, new_game_id: str) -> None:
        """Enforces hard cross-game isolation discipline (Section 11.2).
        
        Zero mechanics transfer across game boundaries.
        """
        self.current_game_id = new_game_id
        self.current_level = 1
        self.episodic_tape.clear()
        self.visited_state_graph.clear()
        self.semantic_rules.clear()
        self.candidate_goals.clear()
        self.fatal_ledger.clear()
        self.noop_ledger.clear()
