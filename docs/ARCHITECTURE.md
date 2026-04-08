# How this thing is put together

The whole project is small enough that you can read it end-to-end in maybe
twenty minutes, but a quick map helps if you're just dropping in.

## The pieces

There's an LLM agent (`inference.py`) that talks HTTP to a FastAPI app
(`environment.py`). The FastAPI app holds a single in-memory `EnvSession`
that tracks "what episode are we in, what step, what's the state right now."
Whenever a step comes in, the session leans on three other modules to do the
actual work:

- `simulator.py` — pure functions. Given the current state, how many
  requests we're letting through, and a config, it tells you what the next
  state looks like and whether the server crashed. No I/O, no globals.
- `tasks.py` — the traffic patterns and episode lengths. One function per
  task that maps `step -> incoming req/s`.
- `graders.py` — once an episode finishes, the grader for that task gets
  the full history and returns a score in [0, 1].

`models.py` defines the wire format (Pydantic) and the action enum. It's the
contract between the agent and the env, so if you're changing anything that
crosses the network, start there.

`client.py` is just a thin Python wrapper around the HTTP API for people who
don't feel like writing `httpx` calls by hand.

## What a single step looks like

The agent posts an action. The session looks up the traffic function for the
current task, asks it for `incoming` at the current step, multiplies by the
accept rate for the chosen action to get `allowed`, and hands both to
`compute_next_state`. Whatever comes back gets shoved through the reward
shaper, appended to the history, and returned over the wire. If we crashed
or hit `max_steps`, the grader runs and the final score gets stuck in the
`info` dict on the way out.

## Things that are easy to miss

**It's single-tenant.** There's exactly one `EnvSession` per server process.
Two clients hitting the same server will stomp on each other's episodes.
That's fine for the use case (one agent, one env) but worth knowing.

**The look-ahead trick.** After every step, the session overwrites
`state.request_rate` with the *next* incoming rate before sending the state
back. The agent is technically reacting to the future, not the past. This
makes the tasks much more solvable than they'd otherwise be.

**The simulator is pure.** That's the single most useful property of this
codebase for testing. You can throw any state/action combo at it without
spinning up a server, which is exactly why the test suite is fast.

**Reward != score.** The per-step reward in `environment.py` is shaped for
RL-style training and isn't the same thing as the grader's final score.
Don't confuse them when you're debugging.
