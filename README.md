---
title: Adaptive Traffic Controller
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
  - llm-agent
  - traffic-control
license: mit
---

# Adaptive Backend Traffic Controller

An **OpenEnv**-compatible environment where an LLM agent prevents backend server crashes by intelligently throttling incoming traffic in real-time. The agent observes server metrics, reasons about the situation, and picks the optimal throttling action — no training required, just prompting.

Built for the **Scaler × Meta PyTorch Hackathon**.

---

## Why This Matters — Traditional vs LLM Approach

### The Problem
Backend servers crash under traffic spikes. This is a real-world problem every tech company faces — Black Friday sales, viral content, DDoS attacks.

### Traditional Approaches (and their limits)

| Approach | How it works | Limitation |
|---|---|---|
| **Static rate limiting** | Fixed threshold (e.g. 100 req/s max) | Can't adapt — wastes capacity at low load, still crashes during unexpected spikes |
| **Auto-scaling** | Spin up more servers when load increases | Slow (minutes to provision), expensive, doesn't help with instant spikes |
| **Rule-based throttling** | If CPU > 80% then drop 50% | Hardcoded thresholds — different servers need different rules, no learning |
| **PID controllers** | Feedback loop adjusting admission rate | Requires manual tuning per deployment, poor at handling non-linear dynamics |

### Our Approach: LLM as Adaptive Controller
An LLM agent that:
- **Reads** real-time server metrics (CPU, memory, latency, queue) as natural language
- **Reasons** about the situation ("traffic is 160% of capacity, need to throttle aggressively")
- **Adapts** to any server configuration without code changes — just tell it the capacity in the prompt
- **Generalizes** — the same agent works for a 50 req/s server or a 500 req/s server

This environment lets anyone test whether their LLM can make these split-second decisions correctly.

---

## How the Environment Works

```
                    ┌─────────────────┐
  Traffic Spikes ──►│   Environment   │──► Server Metrics (CPU, latency, queue)
                    │  (simulator.py) │          │
                    └────────┬────────┘          ▼
                             │           ┌──────────────┐
                             │           │   LLM Agent   │
                             │           │ (inference.py) │
                             │           └──────┬───────┘
                             │                  │
                             ◄──────────────────┘
                          Action: allow_all / throttle_70 / throttle_40 / drop_aggressive
```

1. **Environment** generates traffic patterns (spikes, ramps, sustained overload)
2. **Agent** observes server state each step and picks a throttling action
3. **Environment** simulates the effect — CPU changes, latency spikes, queue builds
4. **Grader** scores the agent: did it survive? Was latency acceptable? How much traffic got through?

The agent must balance two competing goals:
- **Maximize throughput** — let as much traffic through as possible (users want fast responses)
- **Prevent crashes** — don't overload the server (crashes = total failure, score = 0)

---

## Overview

The environment simulates a backend server receiving variable traffic. The agent observes system metrics every time step and chooses a throttling action to keep the server healthy. The server's physics are modelled realistically: CPU and memory track load linearly, latency spikes superlinearly, and sustained overload causes crashes.

---

## Observation Space

| Field | Type | Range | Description |
|-------|------|--------|-------------|
| `cpu_usage` | float | 0.0 – 1.0 | CPU utilization fraction |
| `memory_usage` | float | 0.0 – 1.0 | Memory utilization fraction |
| `request_rate` | float | ≥ 0 | Incoming requests per second |
| `queue_length` | int | 0 – 500 | Pending requests in backlog |
| `avg_latency` | float | ≥ 0 | Average response latency (ms) |
| `step` | int | ≥ 0 | Current episode step |
| `crashed` | bool | — | Whether the server crashed this step |

---

## Action Space

| Action | Accept Rate | Description |
|--------|------------|-------------|
| `allow_all` | 100% | Safe load — accept all requests |
| `throttle_70` | 70% | Moderate load — drop 30% |
| `throttle_40` | 40% | High load — drop 60% |
| `drop_aggressive` | 20% | Imminent crash — drop 80% |

---

## Tasks

### Task Easy — Single Spike
- Traffic: 40 req/s baseline → 160 req/s spike at step 10 for 5 steps → back to 40
- Episode length: 30 steps
- Scoring:
  - `1.0` — no crash AND avg latency < 300 ms
  - `0.5` — no crash, but avg latency ≥ 300 ms
  - `0.0` — any crash

### Task Medium — Multiple Spikes
- Traffic: 50 req/s baseline with 3 spikes of 150 req/s at steps 5, 15, 25 (3 steps each)
- Episode length: 40 steps
- Scoring: `(steps_without_crash / total_steps) × latency_factor`
  - `latency_factor` = 1.0 at ≤ 200 ms, 0.5 at ≥ 600 ms, linear between

### Task Hard — Sustained Overload
- Traffic: ramps 60 → 200 req/s over 20 steps, stays at 200 for 20 steps, drops to 80
- Episode length: 50 steps
- Scoring: `throughput_ratio × 0.7 + queue_factor × 0.3`
  - `throughput_ratio` = total allowed / total incoming
  - `queue_factor` = fraction of steps with queue < 100

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Reset environment, returns initial state |
| `POST` | `/step` | Execute action, returns state/reward/done/info |
| `GET` | `/state` | Current server state |
| `GET` | `/tasks` | List all 3 tasks |
| `GET` | `/openenv.yaml` | OpenEnv specification |
| `GET` | `/health` | Liveness probe |

---

## Configurable Environment

The environment is fully configurable via the `/reset` endpoint. Pass a `config` object to simulate different server profiles:

```bash
curl -X POST localhost:7860/reset -H "Content-Type: application/json" \
     -d '{"task_id": "task_easy", "config": {"server_capacity": 200, "base_latency": 30}}'
```

| Parameter | Default | Description |
|---|---|---|
| `server_capacity` | 100.0 | Max requests/sec the server can handle |
| `base_latency` | 50.0 | Response time at zero load (ms) |
| `crash_load_ratio` | 1.3 | Server crashes at this multiple of capacity |
| `max_queue` | 500 | Maximum pending request queue size |
| `traffic_scale` | 1.0 | Multiplier for traffic patterns (2.0 = double traffic) |

The LLM agent adapts automatically — the system prompt includes the configured capacity so the model knows the server's limits.

---

## Setup

### Local (Python)

```bash
pip install -r requirements.txt

# Start the environment + Gradio UI
python app.py

# Smoke tests
curl -s localhost:7860/health
curl -s -X POST localhost:7860/reset -H "Content-Type: application/json" \
     -d '{"task_id": "task_easy"}' | python -m json.tool
curl -s -X POST localhost:7860/step -H "Content-Type: application/json" \
     -d '{"action": "throttle_70"}' | python -m json.tool
curl -s localhost:7860/tasks | python -m json.tool
curl -s localhost:7860/openenv.yaml
```

### Docker

```bash
docker build -t traffic-controller .
docker run -p 7860:7860 traffic-controller
```

---

## Running Inference

Set the three required environment variables then run `inference.py`:

```bash
export API_BASE_URL="https://api-inference.huggingface.co/models/<your-model>/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
export ENV_URL="http://localhost:7860"   # optional, defaults to this

python inference.py
```

Expected output:

```
Environment URL : http://localhost:7860
Model           : meta-llama/Llama-3.1-8B-Instruct
API base        : https://api-inference.huggingface.co/...

Health check OK

=== TASK_EASY ===
  Starting task_easy (max_steps=30)
    step=  1 action=allow_all          reward=+0.950 latency=  56.5ms queue=   0 cpu=0.54
    ...
  task_easy done — total_reward=27.3, score=1.000

=== RESULTS ===
  task_easy      : 1.000
  task_medium    : 0.875
  task_hard      : 0.623
  Overall        : 0.833
```

---

## Baseline Scores

Measured on the deterministic simulator. Scores are in **0.0 – 1.0**.

| Agent | task_easy | task_medium | task_hard | Overall |
|-------|-----------|-------------|-----------|---------|
| **Always allow_all** (naive) | 0.000 💥 | 0.833 | 0.300 💥 | 0.378 |
| **Always drop_aggressive** (conservative) | 1.000 | 1.000 | 0.440 | 0.813 |
| **Adaptive agent** (scales to config) | 1.000 | 1.000 | 0.500 | 0.833 |
| **LLM agent** (target) | ≥ 0.9 | ≥ 0.9 | ≥ 0.6 | ≥ 0.8 |

💥 = server crash occurred during episode

**Key insight:** The hard task is the differentiator — naive and conservative agents score ≤ 0.44 because sustained 200 req/s overload requires balancing throughput (don't drop too much) against stability (don't let load crash the server). A smart LLM agent should outperform all rule-based baselines here.

---

## Infrastructure

- Port: **7860** (HuggingFace Spaces)
- CPU: 2 vCPU
- Memory: 8 GB
- GPU: not required
- Inference timeout: < 20 minutes total

---

## Project Structure

```
.
├── app.py           # Gradio UI + mounts FastAPI endpoints
├── environment.py   # FastAPI app + episode logic
├── simulator.py     # Backend physics (latency, CPU, memory, crash)
├── models.py        # Pydantic models (state, action, config, request/response)
├── tasks.py         # Traffic patterns + task metadata
├── graders.py       # Per-task scoring functions (0.0–1.0)
├── inference.py     # LLM agent runner (OpenAI client)
├── client.py        # Python EnvClient for programmatic access
├── __init__.py      # Exports Action, ServerState, EnvClient
├── openenv.yaml     # OpenEnv spec
├── pyproject.toml   # Package metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Interactive Demo (UI Guide)

The HF Space serves a **Gradio dashboard** alongside the API. Here's what you can do:

### Controls
- **Traffic Scenario** — pick `task_easy` (single spike), `task_medium` (3 spikes), or `task_hard` (sustained ramp to 200 req/s)
- **Agent Strategy** — compare `Adaptive Agent` vs baselines (`Always Allow` crashes, `Always Throttle 40%` wastes capacity)
- **Server Configuration** — sliders to customize:
  - Server Capacity (20–500 req/s)
  - Base Latency (10–200ms)
  - Crash Threshold (1.1x–2.0x)
  - Traffic Scale (0.5x–3.0x)
- **LLM Agent** — plug in your own API key + model to test a real LLM as the controller

### Dashboard Charts (6 panels)
1. **Traffic: Incoming vs Allowed** — red line = raw traffic, green = what agent lets through, dashed = server capacity
2. **Agent Actions** — color-coded bars (green=allow, yellow=70%, orange=40%, red=drop)
3. **CPU & Memory** — server health with danger line at 80%
4. **Avg Latency** — response time with danger at 400ms
5. **Queue Length** — pending request backlog
6. **Cumulative Reward** — running score (higher = better)

### Agent Reasoning Log
Below the charts, a step-by-step log shows the agent's thinking at each step:
```
Step 10 ⚠️ | Traffic: 160 req/s | Action: 🔴 drop_aggressive → allowed 32 req/s
> Rate 160 req/s = 320% of capacity (50). Drop aggressively!
> CPU: 68% | Latency: 70ms | Queue: 0 | Reward: +0.200
```

---

## Building an Agent for This Environment

### Quick Start (Python)
```python
from client import EnvClient

with EnvClient("http://localhost:7860") as env:
    data = env.reset("task_easy")
    state = data["state"]

    while True:
        # Your agent logic here
        action = decide(state)  # returns "allow_all", "throttle_70", etc.

        result = env.step(action)
        state = result["state"]
        reward = result["reward"]

        if result["done"]:
            print(f"Score: {result['info']['final_score']}")
            break
```

### What Makes a Good Agent
1. **React to `request_rate`** — this shows upcoming traffic, not past traffic. If it exceeds capacity, throttle BEFORE the server overloads.
2. **Don't over-throttle** — `drop_aggressive` prevents crashes but kills throughput. Use it only when truly needed.
3. **Watch the queue** — if `queue_length` is building up, the server is falling behind. Throttle more until it drains.
4. **Adapt to config** — different servers have different capacities. Read the config from `/reset` response and adjust thresholds accordingly.

### Reward Function
```
reward = throughput_reward - latency_penalty * 0.5 - queue_penalty * 0.3

throughput_reward = allowed_requests / incoming_requests     (0.0–1.0)
latency_penalty   = (avg_latency - 200) / 800               (0 at 200ms, 1 at 1000ms)
queue_penalty     = queue_length / max_queue                 (0.0–1.0)
crash             = -10.0                                    (instant game over)
```

The agent is rewarded for letting traffic through, penalized for high latency/queue, and heavily punished for crashing.
