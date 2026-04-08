"""
Opt-in structured logging for the traffic controller.

This module is purely additive: nothing else in the codebase imports it.
To enable logging, call `setup_logging()` from your own entrypoint, e.g.::

    from observability import setup_logging, get_logger
    setup_logging()
    log = get_logger(__name__)
    log.info("env_started", extra={"task_id": "task_easy"})

Configuration is read from environment variables so it can be flipped without
code changes:

    LOG_LEVEL    DEBUG | INFO | WARNING | ERROR   (default: INFO)
    LOG_FORMAT   plain | json                     (default: plain)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any `extra={...}` fields
        for key, value in record.__dict__.items():
            if key in payload or key.startswith("_"):
                continue
            if key in (
                "args", "msg", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "name",
            ):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure the root logger. Safe to call multiple times."""
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    fmt = (fmt or os.environ.get("LOG_FORMAT", "plain")).lower()

    root = logging.getLogger()
    root.setLevel(level)
    # Clear pre-existing handlers so repeated calls don't duplicate output
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
