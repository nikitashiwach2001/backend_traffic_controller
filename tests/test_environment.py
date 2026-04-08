"""End-to-end tests for the FastAPI environment endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from environment import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_tasks_list(client):
    r = client.get("/tasks")
    assert r.status_code == 200
    ids = {t["id"] for t in r.json()["tasks"]}
    assert {"task_easy", "task_medium", "task_hard"} <= ids


def test_reset_unknown_task(client):
    r = client.post("/reset", json={"task_id": "nope"})
    assert r.status_code == 400


def test_reset_and_state(client):
    r = client.post("/reset", json={"task_id": "task_easy"})
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "task_easy"
    assert body["max_steps"] > 0
    assert "state" in body

    s = client.get("/state").json()
    assert "cpu_usage" in s


def test_full_episode_runs_to_done(client):
    client.post("/reset", json={"task_id": "task_easy"})
    done = False
    for _ in range(100):
        r = client.post("/step", json={"action": "throttle_70"})
        assert r.status_code == 200
        body = r.json()
        if body["done"]:
            done = True
            assert "final_score" in body["info"]
            break
    assert done


def test_step_after_done_errors(client):
    client.post("/reset", json={"task_id": "task_easy"})
    for _ in range(100):
        if client.post("/step", json={"action": "drop_aggressive"}).json()["done"]:
            break
    r = client.post("/step", json={"action": "allow_all"})
    assert r.status_code == 400
