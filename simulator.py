"""Backend simulation math — models how a real server responds to load."""

from __future__ import annotations

from models import EnvConfig, ServerState

# Default config (used when no config is provided)
DEFAULT_CONFIG = EnvConfig()


def compute_next_state(
    current_state: ServerState,
    allowed_requests: float,
    incoming_requests: float,
    config: EnvConfig = DEFAULT_CONFIG,
) -> tuple[ServerState, bool]:
    """
    Compute the next server state after one time step.

    Returns (next_state, crashed).

    All thresholds are driven by `config` so users can simulate
    servers with different capacities, latencies, and crash points.
    """
    capacity = config.server_capacity
    base_lat = config.base_latency
    max_queue = config.max_queue

    load_ratio = allowed_requests / capacity

    # Latency spikes superlinearly under load
    if load_ratio <= 1.0:
        latency = base_lat * (1.0 + load_ratio ** 2)
    else:
        latency = base_lat * (1.0 + load_ratio ** 3)  # exponential degradation

    # Queue builds when allowed requests exceed capacity
    queue_delta = max(0.0, allowed_requests - capacity)
    # Queue drains when load is under capacity (servers catch up)
    queue_drain = max(0.0, (capacity - allowed_requests) * 0.3)
    new_queue = current_state.queue_length + queue_delta - queue_drain
    queue_length = int(min(max_queue, max(0.0, new_queue)))

    # Crash if load exceeds crash threshold
    crashed = load_ratio > config.crash_load_ratio

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


def initial_state(incoming_requests: float = 40.0, config: EnvConfig = DEFAULT_CONFIG) -> ServerState:
    """Return a clean initial server state."""
    capacity = config.server_capacity
    base_lat = config.base_latency

    load_ratio = incoming_requests / capacity
    latency = base_lat * (1.0 + load_ratio ** 2)
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
