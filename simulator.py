"""Backend simulation math — models how a real server responds to load."""

from __future__ import annotations

from models import ServerState

MAX_CAPACITY = 100.0   # requests/sec the backend can handle at full health
BASE_LATENCY = 50.0    # milliseconds at zero load
MAX_QUEUE = 500
CRASH_LOAD_RATIO = 1.3  # server crashes when 30% or more over capacity


def compute_next_state(
    current_state: ServerState,
    allowed_requests: float,
    incoming_requests: float,
) -> tuple[ServerState, bool]:
    """
    Compute the next server state after one time step.

    Returns (next_state, crashed).

    The environment exposes the *upcoming* request_rate in the observation so
    the agent can react before overload happens (see environment.py).
    Crash fires when allowed traffic exceeds 130% of capacity in a single step.
    """
    load_ratio = allowed_requests / MAX_CAPACITY

    # Latency spikes superlinearly under load
    if load_ratio <= 1.0:
        latency = BASE_LATENCY * (1.0 + load_ratio ** 2)
    else:
        latency = BASE_LATENCY * (1.0 + load_ratio ** 3)  # exponential degradation

    # Queue builds when allowed requests exceed capacity
    queue_delta = max(0.0, allowed_requests - MAX_CAPACITY)
    # Queue drains when load is under capacity (servers catch up)
    queue_drain = max(0.0, (MAX_CAPACITY - allowed_requests) * 0.3)
    new_queue = current_state.queue_length + queue_delta - queue_drain
    queue_length = int(min(MAX_QUEUE, max(0.0, new_queue)))

    # Crash if load exceeds 130% of capacity
    crashed = load_ratio > CRASH_LOAD_RATIO

    # Latency grows with queue backlog
    latency += queue_length * 0.5

    # CPU and memory track load
    cpu = min(1.0, 0.3 + load_ratio * 0.6)
    memory = min(1.0, 0.2 + load_ratio * 0.4)

    next_state = ServerState(
        cpu_usage=round(cpu, 4),
        memory_usage=round(memory, 4),
        request_rate=round(incoming_requests, 2),
        queue_length=queue_length,
        avg_latency=round(latency, 2),
        step=current_state.step + 1,
        crashed=crashed,
    )
    return next_state, crashed


def initial_state(incoming_requests: float = 40.0) -> ServerState:
    """Return a clean initial server state."""
    load_ratio = incoming_requests / MAX_CAPACITY
    latency = BASE_LATENCY * (1.0 + load_ratio ** 2)
    cpu = min(1.0, 0.3 + load_ratio * 0.6)
    memory = min(1.0, 0.2 + load_ratio * 0.4)
    return ServerState(
        cpu_usage=round(cpu, 4),
        memory_usage=round(memory, 4),
        request_rate=round(incoming_requests, 2),
        queue_length=0,
        avg_latency=round(latency, 2),
        step=0,
        crashed=False,
    )
