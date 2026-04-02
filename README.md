---
title: Adaptive Traffic Controller
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - traffic-control
  - llm-agent
license: mit
---

# Adaptive Backend Traffic Controller

An **OpenEnv**-compatible reinforcement learning environment where an LLM agent learns to prevent backend server crashes by intelligently throttling incoming traffic in real-time.

Built for the **Scaler × Meta PyTorch Hackathon**.

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

## Setup

### Local (Python)

```bash
pip install -r requirements.txt

# Start the environment server
uvicorn environment:app --host 0.0.0.0 --port 7860

# In another terminal, run a quick smoke test
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

# Same smoke tests work on localhost:7860
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
| **Rule-based heuristic** | 1.000 | 1.000 | 0.500 | 0.833 |
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
├── environment.py   # FastAPI app + episode logic
├── tasks.py         # Traffic patterns + task metadata
├── graders.py       # Per-task scoring functions
├── simulator.py     # Backend physics (latency, CPU, memory, crash)
├── models.py        # Pydantic models (state, action, request/response)
├── inference.py     # LLM agent runner
├── openenv.yaml     # OpenEnv spec
├── Dockerfile
├── requirements.txt
└── README.md
```
