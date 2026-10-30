"""Tests for ProceduralMooreMachine."""
import pytest
from inference.utils.procedural_moore_machine import ProceduralMooreMachine


def test_moore_machine_initial_state():
    mm = ProceduralMooreMachine(width=8, height=8, seed=123)
    grid = mm.render()
    assert len(grid) == 8
    assert len(grid[0]) == 8
    # Borders are walls
    assert grid[0][0] == 1
    # Player at (1, 1)
    assert grid[1][1] == 4


def test_moore_machine_step_and_collision():
    mm = ProceduralMooreMachine(width=8, height=8, seed=123)
    # Move up into top wall -> no-op
    grid, changed, win = mm.step("ACTION1")
    assert not changed
    assert not win
    assert mm.player_pos == (1, 1)

    # Move right into empty space -> changes
    grid2, changed2, win2 = mm.step("ACTION4")
    assert changed2
    assert not win2
    assert mm.player_pos == (1, 2)
