"""
Server entry point expected by openenv validate.

Re-exports the FastAPI ``app`` instance defined in the project root
so that ``[project.scripts] server = "server.app:app"`` resolves correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable (needed when invoked via the
# console-script entry point, where the cwd may differ).
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from environment import app  # noqa: E402, F401


def main() -> None:
    """Console-script entry point: launch FastAPI + Gradio on port 7860."""
    import gradio as gr
    import uvicorn

    from app import demo

    mounted = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(mounted, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
