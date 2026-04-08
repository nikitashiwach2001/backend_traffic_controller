"""Tests for grader scoring functions."""
from __future__ import annotations

import pytest

from graders import grade
from models import Action, EpisodeStep, ServerState


def _step(latency=100.0, crashed=False, incoming=40.0, allowed=40.0, queue=0):
    return EpisodeStep(
        step=0,
        state=ServerState(
            cpu_usage=0.4,
            memory_usage=0.3,
            request_rate=incoming,
            queue_length=queue,
            avg_latency=latency,
            step=0,
            crashed=crashed,
        ),
        action=Action.allow_all,
        reward=0.0,
        incoming_requests=incoming,
        allowed_requests=allowed,
        crashed=crashed,
    )


def test_empty_history_scores_zero():
    for tid in ("task_easy", "task_medium", "task_hard"):
        assert grade(tid, []) == 0.0


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        grade("nope", [_step()])


def test_easy_perfect():
    assert grade("task_easy", [_step(latency=150) for _ in range(10)]) == 1.0


def test_easy_high_latency_partial():
    assert grade("task_easy", [_step(latency=400) for _ in range(10)]) == 0.5


def test_easy_crash_zero():
    hist = [_step(latency=100), _step(latency=100, crashed=True)]
    assert grade("task_easy", hist) == 0.0


def test_medium_low_latency_full_credit():
    score = grade("task_medium", [_step(latency=150) for _ in range(20)])
    assert score == 1.0


def test_medium_high_latency_half():
    score = grade("task_medium", [_step(latency=700) for _ in range(20)])
    assert score == pytest.approx(0.5, abs=1e-3)


def test_hard_throughput_and_stability():
    hist = [_step(incoming=100, allowed=100, latency=150, queue=10) for _ in range(20)]
    score = grade("task_hard", hist)
    assert score > 0.9


def test_hard_crash_caps_score():
    hist = [_step(incoming=100, allowed=100, crashed=True) for _ in range(5)]
    assert grade("task_hard", hist) <= 0.3
