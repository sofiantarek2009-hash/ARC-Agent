"""Archimedes-3 Portfolio Value-of-Computation (VoC) Global Scheduler.

Formalized from Section 12.2 of ARCHITECTURAL_BLUEPRINT_ARCHIMEDES_3.md.
Dynamically prioritizes game streams based on:
1. P_solve(l | b_{e,t}) * [ w_{l,e} * (h_{l,e} / (a_{l,e} + a_rem))^2 + gamma_unlock * sum_{k>l} w_{k,e} S_prior(k) ]
2. tokens_since_progress exponential decay damping (tau_fade)
3. Cache-locality hysteresis quantum psi_switch to keep active streams resident in SGLang prefix tree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class GameStreamState:
    game_id: str
    current_level: int
    total_levels: int
    actions_expended: int
    human_baseline: int
    tokens_since_progress: int
    world_model_recall: float
    is_resident: bool = False
    turns_in_cache: int = 0


class PortfolioVoCScheduler:
    """Cache-Pinned Portfolio Value-of-Computation Global Scheduler."""

    def __init__(
        self,
        max_active_streams: int = 4,
        tau_fade: float = 30000.0,
        cache_hysteresis_bonus: float = 50.0,
        gamma_unlock: float = 0.85,
    ) -> None:
        self.max_active_streams = max_active_streams
        self.tau_fade = tau_fade
        self.cache_hysteresis_bonus = cache_hysteresis_bonus
        self.gamma_unlock = gamma_unlock
        self.active_working_set: set[str] = set()

    def compute_game_priority(self, stream: GameStreamState) -> float:
        """Computes marginal expected RHAE rate per equation in Section 12.2."""
        l = max(1, stream.current_level)
        L = max(l, stream.total_levels)

        # Level weight w_{l,e} = 2*l / (L*(L+1))
        total_weight_denom = (L * (L + 1)) / 2.0
        w_l = l / total_weight_denom

        # Estimated actions remaining based on human baseline
        a_rem = max(1, int(stream.human_baseline * 0.5))
        projected_actions = stream.actions_expended + a_rem
        efficiency_curr = min(1.15, stream.human_baseline / max(1.0, float(projected_actions))) ** 2

        # Future levels unlock option value: gamma_unlock * sum_{k=l+1}^L w_{k,e} * S_prior
        unlock_option_value = 0.0
        for k in range(l + 1, L + 1):
            w_k = k / total_weight_denom
            unlock_option_value += w_k * 0.80  # S_prior ~ 0.80

        # P_solve estimated from verified dynamic recall
        p_solve = max(0.1, min(1.0, stream.world_model_recall))

        # Expected score gain
        expected_gain = p_solve * (w_l * efficiency_curr + self.gamma_unlock * unlock_option_value)

        # Exponential decay damper when burning tokens without progress
        fading_factor = math.exp(-stream.tokens_since_progress / max(1000.0, self.tau_fade))

        base_priority = (expected_gain * 10000.0) * fading_factor

        # Cache locality bonus if already resident in SGLang KV tree
        if stream.is_resident:
            base_priority += self.cache_hysteresis_bonus

        return float(base_priority)

    def select_active_working_set(self, streams: Sequence[GameStreamState]) -> list[str]:
        """Selects the top M_active streams to execute while respecting cache hysteresis."""
        ranked = sorted(streams, key=lambda s: self.compute_game_priority(s), reverse=True)
        selected = [s.game_id for s in ranked[: self.max_active_streams]]
        self.active_working_set = set(selected)
        return selected
