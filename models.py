from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Action(str, Enum):
    allow_all = "allow_all"
    throttle_70 = "throttle_70"
    throttle_40 = "throttle_40"
    drop_aggressive = "drop_aggressive"


ACTION_ACCEPT_RATE: dict[Action, float] = {
    Action.allow_all: 1.0,
    Action.throttle_70: 0.7,
    Action.throttle_40: 0.4,
    Action.drop_aggressive: 0.2,
}


class ServerState(BaseModel):
    cpu_usage: float = Field(..., ge=0.0, le=1.0, description="CPU utilization 0–1")
    memory_usage: float = Field(..., ge=0.0, le=1.0, description="Memory utilization 0–1")
    request_rate: float = Field(..., ge=0.0, description="Incoming requests per second")
    queue_length: int = Field(..., ge=0, description="Pending requests in queue")
    avg_latency: float = Field(..., ge=0.0, description="Average latency in milliseconds")
    step: int = Field(default=0, description="Current episode step")
    crashed: bool = Field(default=False, description="Whether server has crashed")


class ResetRequest(BaseModel):
    task_id: str = Field(default="task_easy", description="Task to run")


class ResetResponse(BaseModel):
    state: ServerState
    task_id: str
    max_steps: int


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    state: ServerState
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class EpisodeStep(BaseModel):
    step: int
    state: ServerState
    action: Action
    reward: float
    incoming_requests: float
    allowed_requests: float
    crashed: bool


class TaskInfo(BaseModel):
    id: str
    description: str
    episode_length: int
    difficulty: str


class TaskListResponse(BaseModel):
    tasks: list[TaskInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
