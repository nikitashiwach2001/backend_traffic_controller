"""Tests for task definitions."""
from __future__ import annotations

from tasks import EPISODE_LENGTHS, TASK_METADATA, TRAFFIC_PATTERNS


def test_all_tasks_registered():
    ids = {t.id for t in TASK_METADATA}
    assert ids == set(TRAFFIC_PATTERNS.keys()) == set(EPISODE_LENGTHS.keys())


def test_episode_lengths_positive():
    for length in EPISODE_LENGTHS.values():
        assert length > 0


def test_traffic_patterns_return_floats():
    for tid, fn in TRAFFIC_PATTERNS.items():
        for step in range(EPISODE_LENGTHS[tid]):
            v = fn(step)
            assert isinstance(v, (int, float))
            assert v >= 0


def test_easy_spike_window():
    from tasks import traffic_easy
    assert traffic_easy(0) == 40.0
    assert traffic_easy(12) == 160.0
    assert traffic_easy(20) == 40.0
