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
