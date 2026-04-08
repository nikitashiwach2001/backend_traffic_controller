"""
Graders — deterministic scoring for each task.

Each grader receives the full episode history and returns a float in [0.0, 1.0].
"""

from __future__ import annotations

from models import EpisodeStep

# Scores must lie strictly inside (0, 1) per the submission validator,
# so we clamp every grader's output to this open interval.
SCORE_MIN = 0.01
SCORE_MAX = 0.99


def _clamp(score: float) -> float:
    return round(max(SCORE_MIN, min(SCORE_MAX, score)), 4)


# ---------------------------------------------------------------------------
# Task Easy — Single Spike
# ---------------------------------------------------------------------------

def grade_task_easy(history: list[EpisodeStep]) -> float:
    """
    Score (clamped to (0, 1)):
      ~0.99 → no crash AND avg latency across all steps < 300 ms
      ~0.50 → no crash but avg latency >= 300 ms
      ~0.01 → any crash occurred
    """
    if not history:
        return SCORE_MIN

    crashed = any(s.crashed for s in history)
    if crashed:
        return SCORE_MIN

    avg_latency = sum(s.state.avg_latency for s in history) / len(history)
    if avg_latency < 300.0:
        return _clamp(1.0)
    return _clamp(0.5)


# ---------------------------------------------------------------------------
# Task Medium — Multiple Spikes
# ---------------------------------------------------------------------------

def grade_task_medium(history: list[EpisodeStep]) -> float:
    """
    Score:
      base  = steps_without_crash / total_steps
      penalty factor for high latency: multiplied by latency_factor in [0.5, 1.0]
        latency_factor = 1.0 if avg_latency <= 200 ms
        latency_factor = 0.5 if avg_latency >= 600 ms
        linear interpolation in between
    """
    if not history:
        return SCORE_MIN

    total = len(history)
    crash_steps = sum(1 for s in history if s.crashed)
    base = (total - crash_steps) / total

    avg_latency = sum(s.state.avg_latency for s in history) / total

    low, high = 200.0, 600.0
    if avg_latency <= low:
        latency_factor = 1.0
    elif avg_latency >= high:
        latency_factor = 0.5
    else:
        latency_factor = 1.0 - 0.5 * (avg_latency - low) / (high - low)

    return _clamp(base * latency_factor)


# ---------------------------------------------------------------------------
# Task Hard — Sustained Overload
# ---------------------------------------------------------------------------

def grade_task_hard(history: list[EpisodeStep]) -> float:
    """
    Score = throughput_ratio * stability_bonus * queue_factor

    throughput_ratio = sum(allowed) / sum(incoming)   — maximize allowed traffic
    stability_bonus  = 1.0 if no crash, 0.0 if any crash
    queue_factor     = fraction of steps where queue_length < 100
    """
    if not history:
        return SCORE_MIN

    total_incoming = sum(s.incoming_requests for s in history)
    total_allowed = sum(s.allowed_requests for s in history)

    if total_incoming == 0:
        throughput_ratio = 0.0
    else:
        throughput_ratio = min(1.0, total_allowed / total_incoming)

    crashed = any(s.crashed for s in history)
    stability_bonus = 0.0 if crashed else 1.0

    # Partial credit for keeping queue under control
    low_queue_steps = sum(1 for s in history if s.state.queue_length < 100)
    queue_factor = low_queue_steps / len(history)

    # Combine: throughput matters most, stability is binary gate,
    # queue is a tie-breaker bonus
    if stability_bonus == 0.0:
        # Still give partial credit for throughput management even with a crash
        score = throughput_ratio * 0.3 * queue_factor
    else:
        score = throughput_ratio * 0.7 + queue_factor * 0.3

    return _clamp(score)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "task_easy": grade_task_easy,
    "task_medium": grade_task_medium,
    "task_hard": grade_task_hard,
}


def grade(task_id: str, history: list[EpisodeStep]) -> float:
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id!r}")
    return grader(history)
