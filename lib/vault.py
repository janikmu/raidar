"""Vault file I/O — atomic reads/writes for entities, signals, and digests.

The vault is a second git repo (path comes from `lib.config.load().vault_path`)
laid out as:

    entities/{id}.md           # YAML frontmatter + markdown body
    signals/{id}.jsonl         # append-only, one JSON object per line
    digests/YYYY-MM-DD.md      # weekly digest output
    embeddings/index.json      # owned by lib.embeddings — not touched here

This module deliberately knows nothing about *what* lives in frontmatter
beyond the few fields the filters and dedup helpers query. Schema validation
belongs to capture/enrich, not here.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import frontmatter

from lib import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    id: str
    frontmatter: dict[str, Any]
    body: str  # everything after the frontmatter block, no leading newline


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _vault_root() -> Path:
    return config.load().vault_path


def _entities_dir() -> Path:
    return _vault_root() / "entities"


def _signals_dir() -> Path:
    return _vault_root() / "signals"


def _digests_dir() -> Path:
    return _vault_root() / "digests"


def _entity_path(entity_id: str) -> Path:
    return _entities_dir() / f"{entity_id}.md"


def _signal_path(entity_id: str) -> Path:
    return _signals_dir() / f"{entity_id}.jsonl"


def _digest_path(date_iso: str) -> Path:
    return _digests_dir() / f"{date_iso}.md"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically via tempfile + os.replace in the same dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # delete=False because we close the handle ourselves and rely on os.replace.
    # dir= keeps the tempfile on the same filesystem so replace is atomic.
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(text)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup; let the original exception propagate.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


def read_entity(entity_id: str) -> Entity | None:
    """Return the Entity or None if the file doesn't exist."""
    path = _entity_path(entity_id)
    if not path.is_file():
        return None
    post = frontmatter.load(str(path))
    return Entity(id=entity_id, frontmatter=dict(post.metadata), body=post.content)


def write_entity(entity: Entity) -> None:
    """Serialize the entity and atomically replace entities/{id}.md."""
    post = frontmatter.Post(entity.body, **entity.frontmatter)
    text = frontmatter.dumps(post)
    # python-frontmatter strips trailing newline; restore it so the file ends cleanly.
    if not text.endswith("\n"):
        text += "\n"
    _atomic_write_text(_entity_path(entity.id), text)


def entity_exists(entity_id: str) -> bool:
    return _entity_path(entity_id).is_file()


def list_entities(
    status: str | None = None, type_: str | None = None
) -> list[Entity]:
    """Return entities sorted by id, filtered by frontmatter status/type."""
    entities_dir = _entities_dir()
    if not entities_dir.is_dir():
        return []

    out: list[Entity] = []
    for path in sorted(entities_dir.glob("*.md")):
        entity_id = path.stem
        entity = read_entity(entity_id)
        if entity is None:
            continue
        if status is not None and entity.frontmatter.get("status") != status:
            continue
        if type_ is not None and entity.frontmatter.get("type") != type_:
            continue
        out.append(entity)
    return out


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def append_signal(entity_id: str, snapshot: dict[str, Any]) -> None:
    """Append one JSON line to signals/{id}.jsonl, creating the file if missing."""
    path = _signal_path(entity_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(snapshot, sort_keys=False, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(line)
        fp.write("\n")


def read_signals(entity_id: str) -> list[dict[str, Any]]:
    """Return all signal snapshots for an entity. [] if the file is missing."""
    path = _signal_path(entity_id)
    if not path.is_file():
        logger.debug("signal file missing for %s", entity_id)
        return []

    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped:
                continue
            out.append(json.loads(stripped))
    out.sort(key=lambda e: e.get("date", ""))
    return out


def has_star_history_backfill(entity_id: str) -> bool:
    """True if any signal entry has source='backfill' (star history already fetched)."""
    return any(s.get("source") == "backfill" for s in read_signals(entity_id))


# ---------------------------------------------------------------------------
# History section
# ---------------------------------------------------------------------------


_HISTORY_HEADING = "## History"


def append_history(
    entity_id: str, entry_text: str, entry_date: str | None = None
) -> None:
    """Append a `- {date}: {entry_text}` bullet under `## History` in the entity body.

    If no `## History` section exists, add one to the end of the body.
    """
    entity = read_entity(entity_id)
    if entity is None:
        raise FileNotFoundError(f"entity {entity_id!r} not found")

    if entry_date is None:
        entry_date = date.today().isoformat()
    bullet = f"- {entry_date}: {entry_text}"

    body = entity.body
    if _HISTORY_HEADING in body:
        # Find the heading line and walk to the end of the section (next "## "
        # at column 0, or end of body). Append the bullet just before that boundary.
        lines = body.splitlines()
        new_lines: list[str] = []
        i = 0
        appended = False
        while i < len(lines):
            new_lines.append(lines[i])
            if not appended and lines[i].strip() == _HISTORY_HEADING:
                # Walk forward to find the end of this section.
                j = i + 1
                while j < len(lines) and not lines[j].startswith("## "):
                    new_lines.append(lines[j])
                    j += 1
                # Trim trailing blank lines inside the section before appending.
                while new_lines and new_lines[-1].strip() == "":
                    new_lines.pop()
                new_lines.append(bullet)
                # Blank line before the next section if there is one.
                if j < len(lines):
                    new_lines.append("")
                i = j
                appended = True
                continue
            i += 1
        new_body = "\n".join(new_lines)
        if body.endswith("\n") and not new_body.endswith("\n"):
            new_body += "\n"
    else:
        # No history section — append one.
        sep = "" if body == "" or body.endswith("\n\n") else ("\n" if body.endswith("\n") else "\n\n")
        new_body = f"{body}{sep}{_HISTORY_HEADING}\n{bullet}\n"

    write_entity(Entity(id=entity.id, frontmatter=entity.frontmatter, body=new_body))


def body_for_embedding(body: str) -> str:
    """Return body with the `## History` section stripped.

    Used by the embedding pipeline so churn on the history bullets doesn't
    perturb the semantic vector.
    """
    if _HISTORY_HEADING not in body:
        return body
    lines = body.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == _HISTORY_HEADING:
            # Skip until the next "## " heading or end.
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    result = "\n".join(out)
    # Trim trailing whitespace introduced by stripping the section.
    return result.rstrip() + ("\n" if body.endswith("\n") else "")


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def find_by_github_repo(github_repo: str) -> list[Entity]:
    """Return entities whose frontmatter `github_repo` matches `owner/name` exactly."""
    matches: list[Entity] = []
    for entity in list_entities():
        if entity.frontmatter.get("github_repo") == github_repo:
            matches.append(entity)
    return matches


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def write_digest(date_iso: str, body_markdown: str) -> Path:
    """Write digests/{date_iso}.md (atomic, overwriting if it exists) and return the path."""
    path = _digest_path(date_iso)
    text = body_markdown if body_markdown.endswith("\n") else body_markdown + "\n"
    _atomic_write_text(path, text)
    return path


def read_recent_digests(within_days: int) -> list[tuple[str, str]]:
    """Return (date_iso, body) tuples for digests within the last `within_days` days, newest first."""
    digests_dir = _digests_dir()
    if not digests_dir.is_dir():
        return []

    cutoff = date.today() - timedelta(days=within_days)
    out: list[tuple[date, str, str]] = []
    for path in digests_dir.glob("*.md"):
        try:
            d = datetime.strptime(path.stem, "%Y-%m-%d").date()
        except ValueError:
            logger.warning("skipping digest with non-ISO filename: %s", path.name)
            continue
        if d < cutoff:
            continue
        out.append((d, path.stem, path.read_text(encoding="utf-8")))

    out.sort(key=lambda t: t[0], reverse=True)
    return [(date_iso, body) for _, date_iso, body in out]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Quick round-trip test against the real configured vault. Writes a fake
    # entity + signal under a sentinel id, exercises the helpers, then deletes
    # everything it created.
    smoke_id = "__smoke_test__"
    entity_path = _entity_path(smoke_id)
    signal_path = _signal_path(smoke_id)

    if entity_path.exists() or signal_path.exists():
        raise RuntimeError(
            f"smoke test artifacts already exist for {smoke_id!r}; refusing to clobber"
        )

    try:
        fm = {
            "id": smoke_id,
            "type": "tool",
            "status": "watch",
            "first_seen": "2026-05-18",
            "last_evaluated": "2026-05-18",
            "source": "smoke",
            "tags": ["smoke", "test"],
            "relevance": "low",
            "github_repo": "smoke/test",
        }
        body = (
            "## What it is\n"
            "Smoke test entity.\n\n"
            "## History\n"
            "- 2026-05-18: Captured.\n"
        )
        write_entity(Entity(id=smoke_id, frontmatter=fm, body=body))
        assert entity_exists(smoke_id), "entity_exists should be true after write"

        roundtrip = read_entity(smoke_id)
        assert roundtrip is not None
        assert roundtrip.frontmatter["github_repo"] == "smoke/test"
        assert "## History" in roundtrip.body

        # Filters.
        listed = list_entities(status="watch", type_="tool")
        assert any(e.id == smoke_id for e in listed), "list_entities filter missed entity"

        by_repo = find_by_github_repo("smoke/test")
        assert any(e.id == smoke_id for e in by_repo), "find_by_github_repo missed entity"

        # Signals.
        append_signal(smoke_id, {"date": "2026-05-18", "stars": 10})
        append_signal(smoke_id, {"date": "2026-05-19", "stars": 12})
        signals = read_signals(smoke_id)
        assert signals == [
            {"date": "2026-05-18", "stars": 10},
            {"date": "2026-05-19", "stars": 12},
        ], f"unexpected signals: {signals!r}"

        # History append.
        append_history(smoke_id, "Smoke check ran.", entry_date="2026-05-20")
        after = read_entity(smoke_id)
        assert after is not None
        assert "- 2026-05-20: Smoke check ran." in after.body

        # Embedding view drops the history section.
        stripped = body_for_embedding(after.body)
        assert "## History" not in stripped
        assert "Smoke test entity." in stripped

        # Append-history when no section exists.
        no_hist = Entity(
            id=smoke_id,
            frontmatter=fm,
            body="## What it is\nNo history yet.\n",
        )
        write_entity(no_hist)
        append_history(smoke_id, "Added history.", entry_date="2026-05-21")
        rehydrated = read_entity(smoke_id)
        assert rehydrated is not None
        assert "## History" in rehydrated.body
        assert "- 2026-05-21: Added history." in rehydrated.body

        print("vault.py smoke test OK")
    finally:
        # Clean up — leave the vault as we found it.
        if entity_path.exists():
            entity_path.unlink()
        if signal_path.exists():
            signal_path.unlink()
