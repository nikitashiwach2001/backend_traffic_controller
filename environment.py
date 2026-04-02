"""
Adaptive Traffic Controller — OpenEnv-compatible FastAPI environment.

Endpoints
---------
POST /reset          reset env, return initial state
POST /step           take action, return (state, reward, done, info)
GET  /state          current state
GET  /tasks          list all tasks
GET  /openenv.yaml   OpenEnv spec
GET  /health         liveness probe
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from graders import grade
from models import (
    Action,
    ACTION_ACCEPT_RATE,
    EpisodeStep,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    ServerState,
    StepRequest,
    StepResponse,
    TaskListResponse,
)
from simulator import compute_next_state, initial_state
from tasks import EPISODE_LENGTHS, TASK_METADATA, TRAFFIC_PATTERNS

# ---------------------------------------------------------------------------
# In-memory session state
# ---------------------------------------------------------------------------

class EnvSession:
    def __init__(self) -> None:
        self.task_id: str = "task_easy"
        self.state: ServerState = initial_state()
        self.step: int = 0
        self.done: bool = False
        self.history: list[EpisodeStep] = []

    def reset(self, task_id: str) -> ServerState:
        traffic_fn = TRAFFIC_PATTERNS[task_id]
        first_incoming = traffic_fn(0)
        self.task_id = task_id
        self.state = initial_state(first_incoming)
        self.step = 0
        self.done = False
        self.history = []
        return self.state

    def step_env(self, action: Action) -> tuple[ServerState, float, bool, dict[str, Any]]:
        if self.done:
            raise ValueError("Episode is done. Call /reset to start a new episode.")

        task_id = self.task_id
        traffic_fn = TRAFFIC_PATTERNS[task_id]
        max_steps = EPISODE_LENGTHS[task_id]

        incoming = traffic_fn(self.step)
        accept_rate = ACTION_ACCEPT_RATE[action]
        allowed = incoming * accept_rate

        next_state, crashed = compute_next_state(self.state, allowed, incoming)
        next_state.step = self.step + 1

        # --- Reward shaping ---
        reward = _compute_reward(
            incoming=incoming,
            allowed=allowed,
            latency=next_state.avg_latency,
            crashed=crashed,
            queue=next_state.queue_length,
        )

        ep_step = EpisodeStep(
            step=self.step,
            state=next_state,
            action=action,
            reward=reward,
            incoming_requests=incoming,
            allowed_requests=allowed,
            crashed=crashed,
        )
        self.history.append(ep_step)

        self.step += 1
        self.state = next_state
        self.done = crashed or (self.step >= max_steps)

        # Expose the *upcoming* incoming rate so the agent can react proactively.
        # This mirrors real monitoring: you see current traffic flow before deciding
        # the next throttle level.
        if not self.done:
            upcoming = traffic_fn(self.step)
            self.state.request_rate = round(upcoming, 2)

        info: dict[str, Any] = {
            "incoming_requests": incoming,
            "allowed_requests": allowed,
            "accept_rate": accept_rate,
            "crashed": crashed,
            "episode_step": self.step,
            "max_steps": max_steps,
        }

        if self.done:
            final_score = grade(task_id, self.history)
            info["final_score"] = final_score
            info["episode_done"] = True

        return next_state, reward, self.done, info


def _compute_reward(
    incoming: float,
    allowed: float,
    latency: float,
    crashed: bool,
    queue: int,
) -> float:
    if crashed:
        return -10.0

    # Throughput reward: prefer allowing more traffic (normalised to [0, 1])
    throughput_reward = allowed / max(incoming, 1.0)

    # Latency penalty: smooth penalty starting at 200 ms
    latency_penalty = max(0.0, (latency - 200.0) / 800.0)  # 0 at 200ms, 1 at 1000ms

    # Queue penalty
    queue_penalty = min(1.0, queue / 500.0)

    reward = throughput_reward - latency_penalty * 0.5 - queue_penalty * 0.3
    return round(reward, 4)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

SESSION = EnvSession()


@asynccontextmanager
async def lifespan(app: FastAPI):
    SESSION.reset("task_easy")
    yield


app = FastAPI(
    title="Adaptive Traffic Controller",
    description="OpenEnv environment for LLM-based backend traffic control",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/reset", response_model=ResetResponse)
async def reset(body: ResetRequest = ResetRequest()) -> ResetResponse:
    if body.task_id not in TRAFFIC_PATTERNS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id {body.task_id!r}. "
                   f"Valid: {list(TRAFFIC_PATTERNS.keys())}",
        )
    state = SESSION.reset(body.task_id)
    return ResetResponse(
        state=state,
        task_id=body.task_id,
        max_steps=EPISODE_LENGTHS[body.task_id],
    )


@app.post("/step", response_model=StepResponse)
async def step(body: StepRequest) -> StepResponse:
    try:
        state, reward, done, info = SESSION.step_env(body.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StepResponse(state=state, reward=reward, done=done, info=info)


@app.get("/state", response_model=ServerState)
async def get_state() -> ServerState:
    return SESSION.state


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks() -> TaskListResponse:
    return TaskListResponse(tasks=TASK_METADATA)


@app.get("/openenv.yaml", response_class=PlainTextResponse)
async def get_openenv_yaml() -> str:
    yaml_path = Path(__file__).parent / "openenv.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    return yaml_path.read_text()
