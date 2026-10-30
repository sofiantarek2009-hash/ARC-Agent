"""Archimedes-3 Tycho Rendered Moore Machine Base Class & Abstract AST Synthesizer.

Formalized from Section 7 of ARCHITECTURAL_BLUEPRINT_ARCHIMEDES_3.md:
M_l = (S_l, s_{l,0}, A_e, T_theta, rho_theta, omega_theta, upsilon_theta)
Provides the formal interface that world_model.py candidate classes implement.
"""
from __future__ import annotations

import copy
from typing import Any, Sequence

# Sentinel for explicit epistemic abstention on cells
ABSTAIN = -1


class RenderedMooreMachine:
    """Deterministic Rendered Moore Machine with Transition Emissions & Epistemic Abstention."""

    def __init__(self, frame_0: list[list[int]], level_idx: int = 1) -> None:
        self.state = self.init_state(frame_0, level_idx)

    def init_state(self, frame_0: list[list[int]], level_idx: int = 1) -> Any:
        """Maps initial rendered frame F_{l,0} and level index to latent state s_{l,0}."""
        return {
            "grid": [list(r) for r in frame_0],
            "level": level_idx,
            "step": 0,
            "latent_flags": {},
        }

    def transition(self, state: Any, action: str, action_data: dict[str, Any] | None = None) -> Any:
        """Deterministic transition function T_theta: S_l x A_e -> S_l."""
        nxt = copy.deepcopy(state)
        nxt["step"] += 1
        return nxt

    def render(self, state: Any) -> list[list[int]]:
        """Partial renderer rho_theta: S_l -> ({0..15} U {ABSTAIN})^{64x64}."""
        return state.get("grid", [])

    def predict_outcome(self, state: Any) -> str:
        """Terminal status classifier upsilon_theta: S_l -> {PLAYABLE, WIN, DEATH}."""
        return "PLAYABLE"

    def emit_animation(self, state: Any, action: str) -> list[list[list[int]]]:
        """Emission function omega_theta capturing transient intermediate animation sequence."""
        return []
