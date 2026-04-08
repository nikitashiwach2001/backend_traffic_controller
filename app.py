"""
Gradio demo for the Adaptive Traffic Controller.

Runs the simulation with a rule-based agent and visualises every step
through interactive charts. Deploy on HF Spaces as-is.

    pip install gradio plotly
    python app.py
"""

from __future__ import annotations

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models import Action, ACTION_ACCEPT_RATE, ServerState
from simulator import compute_next_state, initial_state
from tasks import TRAFFIC_PATTERNS, EPISODE_LENGTHS

# ---------------------------------------------------------------------------
# Rule-based agent (mirrors the LLM system prompt heuristics)
# ---------------------------------------------------------------------------

ACTION_LABELS = {
    Action.allow_all: "allow_all (100%)",
    Action.throttle_70: "throttle_70 (70%)",
    Action.throttle_40: "throttle_40 (40%)",
    Action.drop_aggressive: "drop_aggressive (20%)",
}

ACTION_COLORS = {
    Action.allow_all: "#2ecc71",       # green
    Action.throttle_70: "#f1c40f",     # yellow
    Action.throttle_40: "#e67e22",     # orange
    Action.drop_aggressive: "#e74c3c", # red
}


def rule_based_agent(state: ServerState) -> Action:
    """Heuristic agent — uses both current metrics AND upcoming request_rate."""
    cpu = state.cpu_usage
    latency = state.avg_latency
    queue = state.queue_length
    rate = state.request_rate  # upcoming traffic the env exposes

    # Proactive: if upcoming traffic would exceed capacity, throttle early
    if rate > 130:
        return Action.drop_aggressive
    if rate > 100:
        return Action.throttle_40
    if rate > 70:
        return Action.throttle_70

    # Reactive: use current server health
    if cpu < 0.6 and latency < 200 and queue < 50:
        return Action.allow_all
    if cpu < 0.75 and latency < 300:
        return Action.throttle_70
    if cpu < 0.9 and latency < 500 and queue < 150:
        return Action.throttle_40
    return Action.drop_aggressive


# ---------------------------------------------------------------------------
# Random / naive baselines for comparison
# ---------------------------------------------------------------------------

def always_allow_agent(_state: ServerState) -> Action:
    return Action.allow_all


def always_throttle_agent(_state: ServerState) -> Action:
    return Action.throttle_40


AGENTS = {
    "Smart Agent (rule-based)": rule_based_agent,
    "Baseline: Always Allow": always_allow_agent,
    "Baseline: Always Throttle 40%": always_throttle_agent,
}

# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str, agent_fn):
    traffic_fn = TRAFFIC_PATTERNS[task_id]
    max_steps = EPISODE_LENGTHS[task_id]

    state = initial_state(traffic_fn(0))
    steps, cpus, mems, latencies, queues = [], [], [], [], []
    incoming_rates, allowed_rates, rewards, actions = [], [], [], []
    cumulative_reward = []
    total_reward = 0.0

    for step in range(max_steps):
        action = agent_fn(state)
        incoming = traffic_fn(step)
        accept_rate = ACTION_ACCEPT_RATE[action]
        allowed = incoming * accept_rate

        next_state, crashed = compute_next_state(state, allowed, incoming)
        next_state.step = step + 1

        # Reward (same formula as environment.py)
        throughput_reward = allowed / max(incoming, 1.0)
        latency_penalty = max(0.0, (next_state.avg_latency - 200.0) / 800.0)
        queue_penalty = min(1.0, next_state.queue_length / 500.0)
        reward = throughput_reward - latency_penalty * 0.5 - queue_penalty * 0.3
        if crashed:
            reward = -10.0
        reward = round(reward, 4)
        total_reward += reward

        steps.append(step)
        cpus.append(next_state.cpu_usage)
        mems.append(next_state.memory_usage)
        latencies.append(next_state.avg_latency)
        queues.append(next_state.queue_length)
        incoming_rates.append(incoming)
        allowed_rates.append(allowed)
        rewards.append(reward)
        actions.append(action)
        cumulative_reward.append(total_reward)

        if crashed:
            break

        # Update state for next step
        if step + 1 < max_steps:
            upcoming = traffic_fn(step + 1)
            next_state.request_rate = round(upcoming, 2)
        state = next_state

    return {
        "steps": steps,
        "cpu": cpus,
        "memory": mems,
        "latency": latencies,
        "queue": queues,
        "incoming": incoming_rates,
        "allowed": allowed_rates,
        "reward": rewards,
        "cumulative_reward": cumulative_reward,
        "actions": actions,
        "crashed": crashed,
        "total_reward": total_reward,
        "final_step": len(steps),
        "max_steps": max_steps,
    }


# ---------------------------------------------------------------------------
# Plotly charts
# ---------------------------------------------------------------------------

def build_dashboard(task_id: str, agent_name: str):
    agent_fn = AGENTS[agent_name]
    data = run_episode(task_id, agent_fn)

    steps = data["steps"]
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Traffic: Incoming vs Allowed (req/s)",
            "Agent Actions Over Time",
            "CPU & Memory Usage",
            "Avg Latency (ms)",
            "Queue Length",
            "Cumulative Reward",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    # 1) Traffic: incoming vs allowed
    fig.add_trace(go.Scatter(
        x=steps, y=data["incoming"], name="Incoming",
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=data["allowed"], name="Allowed",
        line=dict(color="#2ecc71", width=2),
        fill="tozeroy", fillcolor="rgba(46,204,113,0.1)",
    ), row=1, col=1)
    # Capacity line
    fig.add_hline(y=100, line_dash="dash", line_color="gray",
                  annotation_text="Server Capacity", row=1, col=1)

    # 2) Actions as colored bar chart
    action_colors = [ACTION_COLORS[a] for a in data["actions"]]
    action_labels = [ACTION_LABELS[a] for a in data["actions"]]
    accept_pcts = [ACTION_ACCEPT_RATE[a] * 100 for a in data["actions"]]
    fig.add_trace(go.Bar(
        x=steps, y=accept_pcts, name="Accept %",
        marker_color=action_colors,
        text=action_labels, textposition="none",
        hovertemplate="Step %{x}<br>Accept: %{y}%<br>%{text}<extra></extra>",
    ), row=1, col=2)

    # 3) CPU & Memory
    fig.add_trace(go.Scatter(
        x=steps, y=data["cpu"], name="CPU",
        line=dict(color="#3498db", width=2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=data["memory"], name="Memory",
        line=dict(color="#9b59b6", width=2),
    ), row=2, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="#e74c3c",
                  annotation_text="Danger", row=2, col=1)

    # 4) Latency
    fig.add_trace(go.Scatter(
        x=steps, y=data["latency"], name="Latency",
        line=dict(color="#e67e22", width=2),
        fill="tozeroy", fillcolor="rgba(230,126,34,0.1)",
    ), row=2, col=2)
    fig.add_hline(y=400, line_dash="dash", line_color="#e74c3c",
                  annotation_text="Danger (400ms)", row=2, col=2)

    # 5) Queue
    fig.add_trace(go.Scatter(
        x=steps, y=data["queue"], name="Queue",
        line=dict(color="#1abc9c", width=2),
        fill="tozeroy", fillcolor="rgba(26,188,156,0.1)",
    ), row=3, col=1)
    fig.add_hline(y=200, line_dash="dash", line_color="#e74c3c",
                  annotation_text="Danger (200)", row=3, col=1)

    # 6) Cumulative Reward
    fig.add_trace(go.Scatter(
        x=steps, y=data["cumulative_reward"], name="Cum. Reward",
        line=dict(color="#2c3e50", width=2.5),
        fill="tozeroy", fillcolor="rgba(44,62,80,0.08)",
    ), row=3, col=2)

    # Layout
    fig.update_layout(
        height=900,
        showlegend=False,
        template="plotly_white",
        title_text=f"Adaptive Traffic Controller — {task_id} | {agent_name}",
        title_x=0.5,
        font=dict(size=12),
        margin=dict(t=80, b=40),
    )

    # Summary
    status = "CRASHED" if data["crashed"] else "Survived"
    summary = (
        f"### Results\n"
        f"- **Status:** {status}\n"
        f"- **Steps completed:** {data['final_step']} / {data['max_steps']}\n"
        f"- **Total reward:** {data['total_reward']:.3f}\n"
        f"- **Avg reward/step:** {data['total_reward'] / max(data['final_step'], 1):.3f}\n"
    )

    return fig, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# Adaptive Traffic Controller

An LLM agent that dynamically throttles backend traffic to **prevent server crashes
while maximising throughput**. Watch how it reacts to traffic spikes in real time!

**How it works:**
1. Pick a traffic scenario (easy/medium/hard) and an agent strategy
2. The simulation runs step-by-step — each step the agent observes server metrics
   (CPU, memory, latency, queue) and decides how much traffic to allow
3. Charts show every metric and the agent's decisions over time

**Actions available to the agent:**
| Action | Traffic Allowed |
|---|---|
| `allow_all` | 100% |
| `throttle_70` | 70% |
| `throttle_40` | 40% |
| `drop_aggressive` | 20% |
"""

with gr.Blocks(
    title="Adaptive Traffic Controller",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        task_dd = gr.Dropdown(
            choices=["task_easy", "task_medium", "task_hard"],
            value="task_easy",
            label="Traffic Scenario",
        )
        agent_dd = gr.Dropdown(
            choices=list(AGENTS.keys()),
            value="Smart Agent (rule-based)",
            label="Agent Strategy",
        )
        run_btn = gr.Button("Run Simulation", variant="primary", scale=0)

    plot_out = gr.Plot(label="Dashboard")
    summary_out = gr.Markdown()

    run_btn.click(
        fn=build_dashboard,
        inputs=[task_dd, agent_dd],
        outputs=[plot_out, summary_out],
    )

    # Run on load so the page isn't empty
    demo.load(
        fn=build_dashboard,
        inputs=[task_dd, agent_dd],
        outputs=[plot_out, summary_out],
    )

if __name__ == "__main__":
    demo.launch()
