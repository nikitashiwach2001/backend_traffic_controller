"""Task definitions — each describes a traffic pattern and episode parameters."""

from __future__ import annotations

from models import TaskInfo


# ---------------------------------------------------------------------------
# Traffic pattern generators
# Each returns incoming request rate (req/s) for a given step index (0-based).
# ---------------------------------------------------------------------------

def traffic_easy(step: int) -> float:
    """
    Task Easy — Single Spike
    Baseline 40 req/s, spike to 160 at step 10 for 5 steps, back to 40.
    """
    if 10 <= step < 15:
        return 160.0
    return 40.0


def traffic_medium(step: int) -> float:
    """
    Task Medium — Multiple Spikes
    Baseline 50 req/s, spikes of 150 req/s at steps 5–7, 15–17, 25–27.
    """
    if 5 <= step < 8:
        return 150.0
    if 15 <= step < 18:
        return 150.0
    if 25 <= step < 28:
        return 150.0
    return 50.0


def traffic_hard(step: int) -> float:
    """
    Task Hard — Sustained Overload
    Ramps from 60 → 200 req/s over 20 steps, stays at 200 for 20 more steps,
    then drops back to 80 for the final 10 steps.
    """
    if step < 20:
        # linear ramp 60 → 200
        return 60.0 + (200.0 - 60.0) * (step / 19.0)
    if step < 40:
        return 200.0
    return 80.0


TRAFFIC_PATTERNS: dict[str, callable] = {
    "task_easy": traffic_easy,
    "task_medium": traffic_medium,
    "task_hard": traffic_hard,
}

EPISODE_LENGTHS: dict[str, int] = {
    "task_easy": 30,
    "task_medium": 40,
    "task_hard": 50,
}

TASK_METADATA: list[TaskInfo] = [
    TaskInfo(
        id="task_easy",
        description=(
            "Single traffic spike: baseline 40 req/s rising to 160 req/s at step 10 "
            "for 5 steps, then back to 40. Agent must detect and throttle the spike "
            "without crashing the server."
        ),
        episode_length=30,
        difficulty="easy",
    ),
    TaskInfo(
        id="task_medium",
        description=(
            "Three traffic spikes of 150 req/s at steps 5, 15, and 25 (3 steps each), "
            "baseline 50 req/s. Agent must handle repeated bursts while maintaining "
            "throughput between spikes."
        ),
        episode_length=40,
        difficulty="medium",
    ),
    TaskInfo(
        id="task_hard",
        description=(
            "Sustained overload: traffic ramps from 60 → 200 req/s over 20 steps, "
            "stays at 200 for 20 more steps, then drops to 80. Agent must balance "
            "throughput vs. stability under prolonged high load."
        ),
        episode_length=50,
        difficulty="hard",
    ),
]
