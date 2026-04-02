"""
LLM agent runner for the Adaptive Traffic Controller environment.

Usage:
    API_BASE_URL=<url> MODEL_NAME=<model> HF_TOKEN=<token> python inference.py

Environment variables:
    API_BASE_URL  — OpenAI-compatible base URL (e.g. HuggingFace TGI endpoint)
    MODEL_NAME    — Model identifier
    HF_TOKEN      — API key / HuggingFace token
    ENV_URL       — (optional) Traffic controller environment URL, default http://localhost:7860
"""

from __future__ import annotations

import os
import sys
import time

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ["API_BASE_URL"]
MODEL_NAME: str = os.environ["MODEL_NAME"]
HF_TOKEN: str = os.environ["HF_TOKEN"]
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860")

VALID_ACTIONS = {"allow_all", "throttle_70", "throttle_40", "drop_aggressive"}
DEFAULT_ACTION = "throttle_70"
MAX_RETRIES = 3

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a backend traffic controller agent.
Your goal: prevent server crashes while maximizing throughput.

Server state fields:
  cpu_usage      — fraction 0.0–1.0 (danger above 0.8)
  memory_usage   — fraction 0.0–1.0 (danger above 0.8)
  request_rate   — incoming requests per second
  queue_length   — pending requests (danger above 200)
  avg_latency    — milliseconds (danger above 400ms)

Available actions (choose exactly one):
  allow_all        — accept 100% of requests (use when load is safe)
  throttle_70      — accept 70%, drop 30% (use when load is moderate)
  throttle_40      — accept 40%, drop 60% (use when load is high)
  drop_aggressive  — accept 20%, drop 80% (use when crash is imminent)

Decision heuristics:
  - cpu < 0.6 AND latency < 200ms AND queue < 50  → allow_all
  - cpu < 0.75 OR latency < 300ms                  → throttle_70
  - cpu < 0.9 OR latency < 500ms OR queue < 150    → throttle_40
  - otherwise                                       → drop_aggressive

Respond with ONLY the action name, nothing else. No punctuation, no explanation."""


def _format_state(state: dict) -> str:
    return (
        f"cpu_usage={state['cpu_usage']:.3f} "
        f"memory_usage={state['memory_usage']:.3f} "
        f"request_rate={state['request_rate']:.1f} req/s "
        f"queue_length={state['queue_length']} "
        f"avg_latency={state['avg_latency']:.1f}ms "
        f"step={state.get('step', '?')}"
    )


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def get_action(state: dict) -> str:
    """Query the LLM for a throttling action given the current server state."""
    user_msg = f"Current server state: {_format_state(state)}\nChoose action:"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=20,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().lower()
            # Normalise: strip punctuation, take first token
            action = raw.split()[0].rstrip(".,;:!") if raw.split() else ""
            if action in VALID_ACTIONS:
                return action
            print(f"  [warn] LLM returned invalid action {raw!r}, attempt {attempt}/{MAX_RETRIES}")
        except Exception as exc:
            print(f"  [warn] LLM call failed ({exc}), attempt {attempt}/{MAX_RETRIES}")
            time.sleep(1)

    print(f"  [warn] falling back to default action: {DEFAULT_ACTION}")
    return DEFAULT_ACTION


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, env_url: str) -> float:
    """Run one full episode for task_id and return the final graded score."""
    http = httpx.Client(base_url=env_url, timeout=30.0)

    # Reset environment
    reset_resp = http.post("/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    data = reset_resp.json()
    state = data["state"]
    max_steps = data["max_steps"]

    print(f"  Starting {task_id} (max_steps={max_steps})")

    total_reward = 0.0
    final_score = 0.0
    step = 0

    while True:
        action = get_action(state)
        step_resp = http.post("/step", json={"action": action})
        step_resp.raise_for_status()
        result = step_resp.json()

        state = result["state"]
        reward = result["reward"]
        done = result["done"]
        info = result["info"]

        total_reward += reward
        step += 1

        crashed = info.get("crashed", False)
        print(
            f"    step={step:3d} action={action:<18s} "
            f"reward={reward:+.3f} latency={state['avg_latency']:6.1f}ms "
            f"queue={state['queue_length']:4d} cpu={state['cpu_usage']:.2f}"
            + (" [CRASH]" if crashed else "")
        )

        if done:
            final_score = info.get("final_score", 0.0)
            break

    print(f"  {task_id} done — total_reward={total_reward:.3f}, score={final_score:.3f}")
    http.close()
    return final_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    env_url = ENV_URL
    print(f"Environment URL : {env_url}")
    print(f"Model           : {MODEL_NAME}")
    print(f"API base        : {API_BASE_URL}")
    print()

    # Quick health check
    try:
        resp = httpx.get(f"{env_url}/health", timeout=10.0)
        resp.raise_for_status()
        print("Health check OK\n")
    except Exception as exc:
        print(f"[ERROR] Environment not reachable at {env_url}: {exc}")
        sys.exit(1)

    results: dict[str, float] = {}
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        print(f"=== {task_id.upper()} ===")
        score = run_task(task_id, env_url)
        results[task_id] = score
        print()

    print("=== RESULTS ===")
    for task_id, score in results.items():
        print(f"  {task_id:<15s}: {score:.3f}")
    overall = sum(results.values()) / len(results)
    print(f"  {'Overall':<15s}: {overall:.3f}")


if __name__ == "__main__":
    main()
