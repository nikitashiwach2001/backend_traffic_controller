# HTTP API

The env runs on port 7860 by default. Everything is JSON in/out. If you want
the exact field types, just look at `models.py` — it's the source of truth and
it's short.

## Health

`GET /health` returns `{"status": "ok"}`. Use it for readiness probes.

## Listing tasks

`GET /tasks` gives you back the list of tasks the env knows about:

```json
{
  "tasks": [
    { "id": "task_easy", "description": "Single traffic spike...",
      "episode_length": 30, "difficulty": "easy" }
  ]
}
```

There are three of these out of the box (easy / medium / hard). Adding more
is covered in `TASK_AUTHORING.md`.

## Starting an episode

`POST /reset` starts (or restarts) an episode. The body is just a task id,
plus an optional config block if you want to override the server model:

```json
{
  "task_id": "task_easy",
  "config": { "server_capacity": 100.0, "crash_load_ratio": 1.3 }
}
```

You can omit `config` entirely and you'll get the defaults from `EnvConfig`.
The response gives you the initial state, the task id you actually got, the
episode length, and the config that ended up being used:

```json
{
  "state": { "cpu_usage": 0.54, "memory_usage": 0.36, "request_rate": 40.0,
             "queue_length": 0, "avg_latency": 58.0, "step": 0, "crashed": false },
  "task_id": "task_easy",
  "max_steps": 30,
  "config": { "server_capacity": 100.0, "...": "..." }
}
```

If you pass a task id that doesn't exist you'll get a 400 back with the list
of valid ids in the error message.

## Taking a step

`POST /step` with one of four actions:

- `allow_all` — let everything through
- `throttle_70` — drop 30%
- `throttle_40` — drop 60%
- `drop_aggressive` — drop 80%

```json
{ "action": "throttle_70" }
```

You get back the next state, the reward for this step, whether the episode
is done, and an `info` dict with the raw incoming/allowed counts and a few
other things that are useful for debugging your agent:

```json
{
  "state": { "...": "..." },
  "reward": 0.41,
  "done": false,
  "info": {
    "incoming_requests": 40.0, "allowed_requests": 28.0,
    "accept_rate": 0.7, "crashed": false,
    "episode_step": 1, "max_steps": 30, "server_capacity": 100.0
  }
}
```

When the episode finishes, `done` flips to `true` and `info` will also have
`final_score` (between 0 and 1) and `episode_done: true`. Trying to step
after that point gives you a 400 — call `/reset` and start over.

One thing worth knowing: after each step the `state.request_rate` field is
overwritten with the *upcoming* incoming rate, not the one you just handled.
That's deliberate — it's a small concession to the agent so it can react
before a spike rather than after.

## Other endpoints

`GET /state` peeks at the current state without advancing the episode.
Handy for debugging or for a separate dashboard process.

`GET /openenv.yaml` serves the OpenEnv spec as plain text.
