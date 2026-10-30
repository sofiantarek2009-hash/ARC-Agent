"""Tests for AnimationEventCompiler."""
import pytest
import numpy as np
from inference.utils.animation_event_compiler import AnimationEventCompiler


def test_animation_event_compiler_empty():
    res = AnimationEventCompiler.compile_animation_timeline([])
    assert res["sub_frame_count"] == 0
    assert res["summary_text"] == ""


def test_animation_event_compiler_transient_flash():
    # Grid 0: cell (1, 1) is 0
    # Grid 1: cell (1, 1) flashes to 5
    # Grid 2: cell (1, 1) returns to 0
    g0 = [[0, 0], [0, 0]]
    g1 = [[0, 0], [0, 5]]
    g2 = [[0, 0], [0, 0]]

    res = AnimationEventCompiler.compile_animation_timeline([g0, g1, g2])
    assert res["sub_frame_count"] == 3
    assert (1, 1) in res["transient_flashes"]
    assert "transient flash" in res["summary_text"]


def test_animation_event_compiler_motion_events():
    g0 = [[1, 0, 0], [0, 0, 0]]
    g1 = [[0, 1, 0], [0, 0, 0]]
    g2 = [[0, 0, 1], [0, 0, 0]]

    res = AnimationEventCompiler.compile_animation_timeline([g0, g1, g2])
    assert res["sub_frame_count"] == 3
    assert len(res["moving_entities"]) == 2
    assert "kinematic motion" in res["summary_text"]
