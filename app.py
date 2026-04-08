"""
Gradio demo for the Adaptive Traffic Controller.

Features:
  - Step-by-step animated simulation with live reasoning log
  - Rule-based agent OR real LLM agent (user provides API key)
  - Baseline comparisons to show why smart control matters
  - Interactive Plotly charts for all server metrics

Deploy on HF Spaces with Docker.

    pip install gradio plotly openai
    python app.py
"""

from __future__ import annotations

import time
from typing import Generator

import gradio as gr
import plotly.graph_objects as go
from openai import OpenAI
from plotly.subplots import make_subplots

from models import Action, ACTION_ACCEPT_RATE, EnvConfig, ServerState
from simulator import compute_next_state, initial_state
from tasks import TRAFFIC_PATTERNS, EPISODE_LENGTHS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_LABELS = {
    Action.allow_all: "allow_all (100%)",
    Action.throttle_70: "throttle_70 (70%)",
    Action.throttle_40: "throttle_40 (40%)",
    Action.drop_aggressive: "drop_aggressive (20%)",
}

ACTION_COLORS = {
    Action.allow_all: "#2ecc71",
    Action.throttle_70: "#f1c40f",
    Action.throttle_40: "#e67e22",
    Action.drop_aggressive: "#e74c3c",
}

ACTION_EMOJI = {
    Action.allow_all: "🟢",
    Action.throttle_70: "🟡",
    Action.throttle_40: "🟠",
    Action.drop_aggressive: "🔴",
}

VALID_ACTIONS = {"allow_all", "throttle_70", "throttle_40", "drop_aggressive"}

LLM_SYSTEM_PROMPT = """You are a backend traffic controller agent.
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

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def adaptive_agent(state: ServerState, config: EnvConfig) -> tuple[Action, str]:
    """
    Adaptive agent — all thresholds scale relative to the configured
    server_capacity. Works for ANY capacity (50, 100, 200, 500, etc.).
    """
    cpu = state.cpu_usage
    latency = state.avg_latency
    queue = state.queue_length
    rate = state.request_rate
    cap = config.server_capacity
    max_q = config.max_queue

    # Thresholds relative to configured capacity
    ratio = rate / cap  # how full is the server relative to its capacity

    if ratio > 1.3:
        reason = f"Rate {rate:.0f} req/s = {ratio:.0%} of capacity ({cap:.0f}). Drop aggressively!"
        return Action.drop_aggressive, reason
    if ratio > 1.0:
        reason = f"Rate {rate:.0f} req/s = {ratio:.0%} of capacity ({cap:.0f}). Throttle to 40%."
        return Action.throttle_40, reason
    if ratio > 0.7:
        reason = f"Rate {rate:.0f} req/s = {ratio:.0%} of capacity ({cap:.0f}). Throttle to 70%."
        return Action.throttle_70, reason

    # Reactive: check server health (these ratios are already normalized 0-1)
    queue_ratio = queue / max_q
    if cpu < 0.6 and latency < config.base_latency * 4 and queue_ratio < 0.1:
        reason = f"All clear — CPU {cpu:.0%}, latency {latency:.0f}ms, queue {queue}/{max_q}. Allow all."
        return Action.allow_all, reason
    if cpu < 0.75 and latency < config.base_latency * 6:
        reason = f"Moderate load — CPU {cpu:.0%}, latency {latency:.0f}ms. Throttle to 70%."
        return Action.throttle_70, reason
    if cpu < 0.9 and latency < config.base_latency * 10 and queue_ratio < 0.3:
        reason = f"High load — CPU {cpu:.0%}, latency {latency:.0f}ms, queue {queue}/{max_q}. Throttle to 40%."
        return Action.throttle_40, reason

    reason = f"Critical — CPU {cpu:.0%}, latency {latency:.0f}ms, queue {queue}/{max_q}. Drop aggressive!"
    return Action.drop_aggressive, reason


def always_allow_agent(state: ServerState, config: EnvConfig) -> tuple[Action, str]:
    return Action.allow_all, "No intelligence — blindly accepting all traffic regardless of load."


def always_throttle_agent(state: ServerState, config: EnvConfig) -> tuple[Action, str]:
    return Action.throttle_40, "No intelligence — always throttling to 40% regardless of conditions."


def make_llm_agent(api_base: str, api_key: str, model_name: str, config: EnvConfig):
    """Create an LLM-based agent closure with capacity-aware prompt."""
    client = OpenAI(base_url=api_base, api_key=api_key)

    # Dynamic system prompt based on configured capacity
    system_prompt = LLM_SYSTEM_PROMPT.replace(
        "prevent server crashes while maximizing throughput.",
        f"prevent server crashes while maximizing throughput.\n"
        f"Server capacity: {config.server_capacity:.0f} req/s. "
        f"Crash threshold: {config.crash_load_ratio:.0%} of capacity.",
    )

    def llm_agent(state: ServerState, _config: EnvConfig) -> tuple[Action, str]:
        user_msg = (
            f"cpu_usage={state.cpu_usage:.3f} "
            f"memory_usage={state.memory_usage:.3f} "
            f"request_rate={state.request_rate:.1f} req/s "
            f"queue_length={state.queue_length} "
            f"avg_latency={state.avg_latency:.1f}ms"
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Current server state: {user_msg}\nChoose action:"},
                ],
                max_tokens=20,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().lower()
            action_str = raw.split()[0].rstrip(".,;:!") if raw.split() else ""
            if action_str in VALID_ACTIONS:
                action = Action(action_str)
                reason = f"LLM chose `{action_str}` (raw: \"{raw}\")"
                return action, reason
            else:
                reason = f"LLM returned invalid action \"{raw}\", falling back to throttle_70"
                return Action.throttle_70, reason
        except Exception as exc:
            reason = f"LLM call failed: {exc}. Falling back to throttle_70"
            return Action.throttle_70, reason

    return llm_agent


BUILTIN_AGENTS = {
    "Adaptive Agent": adaptive_agent,
    "Baseline: Always Allow": always_allow_agent,
    "Baseline: Always Throttle 40%": always_throttle_agent,
}


# ---------------------------------------------------------------------------
# Build charts from episode data
# ---------------------------------------------------------------------------

def build_charts(data: dict, capacity: float = 100.0) -> go.Figure:
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

    # 1) Traffic
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
    fig.add_hline(y=capacity, line_dash="dash", line_color="gray",
                  annotation_text=f"Capacity ({capacity:.0f})", row=1, col=1)

    # 2) Actions
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

    fig.update_layout(
        height=900,
        showlegend=False,
        template="plotly_white",
        title_text="Adaptive Traffic Controller",
        title_x=0.5,
        font=dict(size=12),
        margin=dict(t=80, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Step-by-step streaming simulation
# ---------------------------------------------------------------------------

def run_simulation_streaming(
    task_id: str,
    agent_name: str,
    api_base: str,
    api_key: str,
    model_name: str,
    server_capacity: float,
    base_latency: float,
    crash_threshold: float,
    traffic_scale: float,
) -> Generator:
    """
    Generator that yields (plot, log_text, summary) after each step.
    Gradio streams these updates to the UI in real time.
    """
    config = EnvConfig(
        server_capacity=server_capacity,
        base_latency=base_latency,
        crash_load_ratio=crash_threshold,
        traffic_scale=traffic_scale,
    )

    # Pick agent
    if agent_name == "LLM Agent (bring your own key)":
        if not api_key.strip() or not model_name.strip():
            yield (
                None,
                "**Error:** Please provide API Base URL, API Key, and Model Name to use LLM Agent.",
                "",
            )
            return
        agent_fn = make_llm_agent(api_base.strip(), api_key.strip(), model_name.strip(), config)
    else:
        agent_fn = BUILTIN_AGENTS[agent_name]

    traffic_fn = TRAFFIC_PATTERNS[task_id]
    max_steps = EPISODE_LENGTHS[task_id]

    first_incoming = traffic_fn(0) * config.traffic_scale
    state = initial_state(first_incoming, config=config)

    # Accumulators
    data = {
        "steps": [], "cpu": [], "memory": [], "latency": [], "queue": [],
        "incoming": [], "allowed": [], "reward": [], "cumulative_reward": [],
        "actions": [],
    }
    total_reward = 0.0
    log_lines: list[str] = []
    crashed = False

    log_lines.append(f"### Simulation: {task_id} | {agent_name}")
    log_lines.append(
        f"Server capacity: **{config.server_capacity:.0f} req/s** | "
        f"Crash at: **{config.crash_load_ratio:.0%}** of capacity | "
        f"Traffic scale: **{config.traffic_scale}x** | "
        f"Max steps: **{max_steps}**"
    )
    log_lines.append("---")

    for step in range(max_steps):
        action, reason = agent_fn(state, config)
        incoming = traffic_fn(step) * config.traffic_scale
        accept_rate = ACTION_ACCEPT_RATE[action]
        allowed = incoming * accept_rate

        next_state, crashed = compute_next_state(state, allowed, incoming, config=config)
        next_state.step = step + 1

        # Reward
        throughput_reward = allowed / max(incoming, 1.0)
        latency_penalty = max(0.0, (next_state.avg_latency - 200.0) / 800.0)
        queue_penalty = min(1.0, next_state.queue_length / config.max_queue)
        reward = throughput_reward - latency_penalty * 0.5 - queue_penalty * 0.3
        if crashed:
            reward = -10.0
        reward = round(reward, 4)
        total_reward += reward

        # Store
        data["steps"].append(step)
        data["cpu"].append(next_state.cpu_usage)
        data["memory"].append(next_state.memory_usage)
        data["latency"].append(next_state.avg_latency)
        data["queue"].append(next_state.queue_length)
        data["incoming"].append(incoming)
        data["allowed"].append(allowed)
        data["reward"].append(reward)
        data["cumulative_reward"].append(total_reward)
        data["actions"].append(action)

        # Build reasoning log line
        emoji = ACTION_EMOJI[action]
        status_icon = "💀" if crashed else ("⚠️" if next_state.cpu_usage > 0.8 or next_state.avg_latency > 400 else "✅")

        log_lines.append(
            f"**Step {step}** {status_icon} | "
            f"Traffic: {incoming:.0f} req/s | "
            f"Action: {emoji} `{action.value}` → allowed {allowed:.0f} req/s\n"
            f"> {reason}\n"
            f"> CPU: {next_state.cpu_usage:.0%} | "
            f"Latency: {next_state.avg_latency:.0f}ms | "
            f"Queue: {next_state.queue_length} | "
            f"Reward: {reward:+.3f}"
        )

        if crashed:
            cap = config.server_capacity
            log_lines.append("\n## 💀 SERVER CRASHED!")
            log_lines.append(f"Load ratio: {allowed/cap:.2f}x capacity (crash threshold: {config.crash_load_ratio}x)")
            break

        # Update state
        if step + 1 < max_steps:
            upcoming = traffic_fn(step + 1) * config.traffic_scale
            next_state.request_rate = round(upcoming, 2)
        state = next_state

        # Summary so far
        summary = (
            f"### Results (step {step + 1}/{max_steps})\n"
            f"- **Status:** Running...\n"
            f"- **Total reward:** {total_reward:.3f}\n"
        )

        fig = build_charts(data, capacity=config.server_capacity)
        log_text = "\n\n".join(log_lines)

        yield fig, log_text, summary

    # Final summary
    status = "💀 CRASHED" if crashed else "✅ Survived"
    final_step = len(data["steps"])
    summary = (
        f"### Final Results\n"
        f"- **Status:** {status}\n"
        f"- **Steps completed:** {final_step} / {max_steps}\n"
        f"- **Total reward:** {total_reward:.3f}\n"
        f"- **Avg reward/step:** {total_reward / max(final_step, 1):.3f}\n"
    )

    fig = build_charts(data, capacity=config.server_capacity)
    log_text = "\n\n".join(log_lines)
    yield fig, log_text, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# Adaptive Traffic Controller

An **OpenEnv environment** where LLM agents learn to prevent backend server crashes
by intelligently throttling traffic. Configure your server, watch the agent think step-by-step!

### How it works
1. **Configure** your server — set capacity, latency, crash threshold, traffic intensity
2. Each step, the agent **observes** server metrics (CPU, memory, latency, queue)
3. The agent **decides** how much traffic to allow: 100%, 70%, 40%, or 20%
4. If too much traffic gets through, the server **crashes** (game over!)

### Try it
- Change **Server Capacity** to 50 or 200 and see how the agent adapts
- Crank up **Traffic Scale** to 2x to stress-test the agent
- Switch to **"Always Allow" baseline** to watch the server crash
- Plug in your own **LLM API key** to test a real model as the controller!
"""

AGENT_CHOICES = list(BUILTIN_AGENTS.keys()) + ["LLM Agent (bring your own key)"]

with gr.Blocks(
    title="Adaptive Traffic Controller",
    theme=gr.themes.Soft(),
    css="""
    .reasoning-log { max-height: 400px; overflow-y: auto; }
    """,
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        task_dd = gr.Dropdown(
            choices=["task_easy", "task_medium", "task_hard"],
            value="task_easy",
            label="Traffic Scenario",
            info="Easy = 1 spike, Medium = 3 spikes, Hard = sustained ramp",
        )
        agent_dd = gr.Dropdown(
            choices=AGENT_CHOICES,
            value="Adaptive Agent",
            label="Agent Strategy",
        )
        run_btn = gr.Button("Run Simulation", variant="primary", scale=0)

    # Server configuration sliders
    with gr.Accordion("Server Configuration", open=True):
        gr.Markdown("*Customize the simulated server. The adaptive agent automatically adjusts its thresholds to match.*")
        with gr.Row():
            capacity_slider = gr.Slider(
                minimum=20, maximum=500, value=100, step=10,
                label="Server Capacity (req/s)",
                info="Max requests the server can handle per second",
            )
            latency_slider = gr.Slider(
                minimum=10, maximum=200, value=50, step=5,
                label="Base Latency (ms)",
                info="Response time at zero load",
            )
            crash_slider = gr.Slider(
                minimum=1.1, maximum=2.0, value=1.3, step=0.1,
                label="Crash Threshold",
                info="Server crashes at this multiple of capacity (1.3 = 130%)",
            )
            scale_slider = gr.Slider(
                minimum=0.5, maximum=3.0, value=1.0, step=0.1,
                label="Traffic Scale",
                info="Multiply all traffic patterns by this factor",
            )

    # LLM config section (shown only when LLM Agent is selected)
    with gr.Accordion("LLM Configuration", open=True, visible=False) as llm_config:
        gr.Markdown("*Provide your own OpenAI-compatible API endpoint to test a real LLM as the traffic controller.*")
        with gr.Row():
            api_base_input = gr.Textbox(
                label="API Base URL",
                placeholder="https://api-inference.huggingface.co/v1",
                value="https://api-inference.huggingface.co/v1",
            )
            api_key_input = gr.Textbox(
                label="API Key",
                placeholder="hf_... or sk-...",
                type="password",
            )
            model_name_input = gr.Textbox(
                label="Model Name",
                placeholder="meta-llama/Llama-3.1-8B-Instruct",
                value="meta-llama/Llama-3.1-8B-Instruct",
            )

    # Toggle LLM config visibility
    def toggle_llm_config(agent_name):
        return gr.Accordion(visible=(agent_name == "LLM Agent (bring your own key)"))

    agent_dd.change(fn=toggle_llm_config, inputs=[agent_dd], outputs=[llm_config])

    # Outputs
    summary_out = gr.Markdown()
    plot_out = gr.Plot(label="Dashboard")

    with gr.Accordion("Agent Reasoning Log (step-by-step)", open=True):
        log_out = gr.Markdown(elem_classes=["reasoning-log"])

    all_inputs = [
        task_dd, agent_dd,
        api_base_input, api_key_input, model_name_input,
        capacity_slider, latency_slider, crash_slider, scale_slider,
    ]
    all_outputs = [plot_out, log_out, summary_out]

    run_btn.click(fn=run_simulation_streaming, inputs=all_inputs, outputs=all_outputs)
    demo.load(fn=run_simulation_streaming, inputs=all_inputs, outputs=all_outputs)

if __name__ == "__main__":
    # Mount the FastAPI environment endpoints (required for hackathon evaluation)
    # so /reset, /step, /state, /health, /tasks, /openenv.yaml all work
    # alongside the Gradio UI on the same port.
    from environment import app as fastapi_app

    gradio_app = gr.mount_gradio_app(fastapi_app, demo, path="/")
    import uvicorn
    uvicorn.run(gradio_app, host="0.0.0.0", port=7860)
