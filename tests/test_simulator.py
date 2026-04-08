"""Tests for the backend simulator math."""
from __future__ import annotations

from models import EnvConfig
from simulator import compute_next_state, initial_state


def test_initial_state_under_capacity():
    cfg = EnvConfig()
    s = initial_state(40.0, config=cfg)
    assert s.step == 0
    assert s.queue_length == 0
    assert not s.crashed
    assert 0.0 <= s.cpu_usage <= 1.0
    assert s.avg_latency >= cfg.base_latency


def test_latency_grows_with_load():
    cfg = EnvConfig()
    low, _ = compute_next_state(initial_state(20.0, cfg), 20.0, 20.0, cfg)
    high, _ = compute_next_state(initial_state(20.0, cfg), 90.0, 90.0, cfg)
    assert high.avg_latency > low.avg_latency


def test_crash_above_threshold():
    cfg = EnvConfig(server_capacity=100.0, crash_load_ratio=1.3)
    _, crashed = compute_next_state(initial_state(40.0, cfg), 200.0, 200.0, cfg)
    assert crashed is True


def test_no_crash_at_capacity():
    cfg = EnvConfig(server_capacity=100.0, crash_load_ratio=1.3)
    _, crashed = compute_next_state(initial_state(40.0, cfg), 100.0, 100.0, cfg)
    assert crashed is False


def test_queue_drains_when_idle():
    cfg = EnvConfig()
    s = initial_state(40.0, cfg)
    s.queue_length = 200
    nxt, _ = compute_next_state(s, 10.0, 10.0, cfg)
    assert nxt.queue_length < 200


def test_queue_capped_at_max():
    cfg = EnvConfig(max_queue=500)
    s = initial_state(40.0, cfg)
    s.queue_length = 490
    nxt, _ = compute_next_state(s, 1000.0, 1000.0, cfg)
    assert nxt.queue_length <= 500
