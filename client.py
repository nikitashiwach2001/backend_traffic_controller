"""
OpenEnv client for the Adaptive Traffic Controller.

Provides a Python API to interact with the environment server
without needing to make raw HTTP calls.
"""

from __future__ import annotations

from typing import Any

import httpx

from models import Action, EnvConfig, ServerState


class EnvClient:
    """Client for the Adaptive Traffic Controller environment."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def health(self) -> bool:
        resp = self.http.get("/health")
        return resp.status_code == 200

    def reset(
        self,
        task_id: str = "task_easy",
        config: EnvConfig | None = None,
    ) -> dict[str, Any]:
        """Reset environment. Returns {state, task_id, max_steps, config}."""
        payload: dict[str, Any] = {"task_id": task_id}
        if config is not None:
            payload["config"] = config.model_dump()
        resp = self.http.post("/reset", json=payload)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: str | Action) -> dict[str, Any]:
        """Take one step. Returns {state, reward, done, info}."""
        if isinstance(action, Action):
            action = action.value
        resp = self.http.post("/step", json={"action": action})
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict[str, Any]:
        """Get current server state."""
        resp = self.http.get("/state")
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> list[dict[str, Any]]:
        """List available tasks."""
        resp = self.http.get("/tasks")
        resp.raise_for_status()
        return resp.json()["tasks"]

    def close(self) -> None:
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
