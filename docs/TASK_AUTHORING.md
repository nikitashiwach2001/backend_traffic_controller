# Adding a new task

A task is three small things stitched together: a function that produces
incoming traffic, a number that says how long the episode runs, and a
function that scores the finished episode. All three live at the project
root, no plugin system, no registration magic beyond appending to a dict.

Say you want a task with two short bursts. Open `tasks.py` and add:

```python
def traffic_burst(step: int) -> float:
    """Two short bursts of 180 req/s, otherwise quiet."""
    if 8 <= step < 11 or 22 <= step < 25:
        return 180.0
    return 30.0
```

Then wire it up in the same file:

```python
TRAFFIC_PATTERNS["task_burst"] = traffic_burst
EPISODE_LENGTHS["task_burst"] = 35
TASK_METADATA.append(
    TaskInfo(
        id="task_burst",
        description="Two short bursts of 180 req/s with quiet baseline.",
        episode_length=35,
        difficulty="medium",
    )
)
```

Now jump over to `graders.py` and decide what "doing well" means for this
task. For something this short, "no crash and decent latency" is probably
enough:

```python
def grade_task_burst(history: list[EpisodeStep]) -> float:
    if not history:
        return 0.0
    if any(s.crashed for s in history):
        return 0.0
    avg_latency = sum(s.state.avg_latency for s in history) / len(history)
    return 1.0 if avg_latency < 250 else 0.5

GRADERS["task_burst"] = grade_task_burst
```

That's it. Run `make test`, then `make run` and try it:

```bash
curl -s -X POST localhost:7860/reset \
     -H 'content-type: application/json' \
     -d '{"task_id": "task_burst"}'
```

A few things I've learned the hard way:

The grader has to return a number between 0 and 1. The harness assumes this
and various downstream tools (the UI, the leaderboard) will look weird if
you return something outside that range. Clamp it if you have to.

Keep the traffic function pure. No globals, no random state without a
seeded RNG, no reading files. The tests rely on these being deterministic
and so does anyone trying to reproduce a run.

If your task is meant to be hard, *test* that it's actually hard before
declaring victory. Run the existing baselines against it. A "hard" task
that `throttle_40` solves trivially isn't hard, it's just badly tuned.

And if you're tempted to make the grader complicated, resist. The simpler
graders (`task_easy`) are the ones people actually understand and trust.
The complicated one (`task_hard`) is the one I've had to explain the most.
