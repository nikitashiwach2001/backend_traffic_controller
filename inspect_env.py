"""
Debug helper: dump current environment state and recent history as JSON.

Usage:
    python inspect_env.py                # default http://localhost:7860
    python inspect_env.py --url URL      # custom url
    python inspect_env.py --history 10   # show last N steps from a fresh episode
    python inspect_env.py --replay path  # replay a saved trace file (one JSON per line)

The script is read-only against a running environment. With --replay, it works
purely offline against a JSONL trace dumped by an agent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def cmd_live(url: str) -> int:
    try:
        r = httpx.get(f"{url}/state", timeout=5.0)
        r.raise_for_status()
    except Exception as exc:
        print(f"[error] could not reach {url}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(r.json(), indent=2))
    return 0


def cmd_replay(path: Path) -> int:
    if not path.exists():
        print(f"[error] file not found: {path}", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"# {len(rows)} steps from {path}")
    for row in rows:
        step = row.get("step", "?")
        action = row.get("action", "?")
        state = row.get("state", {})
        print(
            f"step={step:>3} action={action:<16} "
            f"latency={state.get('avg_latency', 0):6.1f}ms "
            f"queue={state.get('queue_length', 0):>4} "
            f"crashed={state.get('crashed', False)}"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect a running env or replay a trace.")
    p.add_argument("--url", default="http://localhost:7860")
    p.add_argument("--replay", type=Path, help="Replay a JSONL trace file")
    args = p.parse_args()

    if args.replay is not None:
        return cmd_replay(args.replay)
    return cmd_live(args.url)


if __name__ == "__main__":
    raise SystemExit(main())
