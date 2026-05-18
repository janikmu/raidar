"""Thin secret-access layer.

Reads from process env after loading .env (if present). Designed so the
call sites (`secrets.get("GITHUB_PAT")`) stay the same when we swap the
backend to macOS Keychain or 1Password later — only this module changes.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_LOADED = False


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    # Walk up from CWD looking for a .env; load_dotenv with no args also does this
    # but being explicit helps when jobs run from launchd with a different CWD.
    env_path = _find_dotenv()
    if env_path is not None:
        load_dotenv(env_path, override=False)
    _LOADED = True


def _find_dotenv() -> Path | None:
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def get(key: str, default: str | None = None) -> str | None:
    """Return secret `key` or `default`. Empty strings are treated as missing."""
    _ensure_loaded()
    value = os.environ.get(key, default)
    if value == "":
        return default
    return value


def require(key: str) -> str:
    """Return secret `key` or raise. Use at boundaries where a missing secret is fatal."""
    value = get(key)
    if value is None:
        raise RuntimeError(f"Required secret {key!r} is not set (check .env or environment)")
    return value
