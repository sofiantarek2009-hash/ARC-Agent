"""Tests for PortfolioVoCScheduler."""
import pytest
from inference.utils.portfolio_voc_scheduler import (
    PortfolioVoCScheduler,
    GameStreamState,
)


def test_portfolio_voc_scheduler_priority_calculation():
    scheduler = PortfolioVoCScheduler(max_active_streams=2)

    # Game A: Level 1 of 5, fresh, high recall
    game_a = GameStreamState(
        game_id="game_a",
        current_level=1,
        total_levels=5,
        actions_expended=5,
        human_baseline=20,
        tokens_since_progress=500,
        world_model_recall=0.95,
        is_resident=True,
    )

    # Game B: Level 4 of 5 (high weight), moderate recall
    game_b = GameStreamState(
        game_id="game_b",
        current_level=4,
        total_levels=5,
        actions_expended=15,
        human_baseline=25,
        tokens_since_progress=1000,
        world_model_recall=0.85,
        is_resident=False,
    )

    # Game C: Stalled (burned 100K tokens without progress)
    game_c = GameStreamState(
        game_id="game_c",
        current_level=1,
        total_levels=3,
        actions_expended=40,
        human_baseline=10,
        tokens_since_progress=90000,
        world_model_recall=0.10,
        is_resident=False,
    )

    p_a = scheduler.compute_game_priority(game_a)
    p_b = scheduler.compute_game_priority(game_b)
    p_c = scheduler.compute_game_priority(game_c)

    assert p_b > p_c
    assert p_a > p_c

    # Active working set selection
    selected = scheduler.select_active_working_set([game_a, game_b, game_c])
    assert len(selected) == 2
    assert "game_c" not in selected
