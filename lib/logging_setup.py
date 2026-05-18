"""Logging configured once per process.

Cron-driven jobs write to a rotating file so we can debug Monday morning
what failed Sunday night. CLI invocations also stream to stderr.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_CONFIGURED = False


def setup(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    stderr: bool = True,
) -> None:
    """Idempotent. First caller wins; later calls are no-ops."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if stderr:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if log_file is not None:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            path, maxBytes=2_000_000, backupCount=4, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # httpx and openai are chatty at INFO; bump them to WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    _CONFIGURED = True
