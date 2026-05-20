"""Vault file I/O — atomic reads/writes for concepts, artifacts, signals, and digests.

The vault is a second git repo (path comes from `lib.config.load().vault_path`)
laid out as:

    concepts/{id}.md           # concept YAML frontmatter + markdown body
    artifacts/{id}.md          # artifact YAML frontmatter + markdown body
    signals/{id}.jsonl         # append-only, one JSON object per line (artifact ids)
    digests/YYYY-MM-DD.md      # weekly digest output
    embeddings/concepts.json   # owned by lib.embeddings — not touched here
    embeddings/artifacts.json  # owned by lib.embeddings — not touched here

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
class Concept:
    """A concept file from concepts/{id}.md.

    Frontmatter keys (all optional beyond id/type):
      id, label, type ("concept"), status, first_seen, last_evaluated,
      relevance, tags, artifacts (list of {id, relationship, weight}),
      review_needed (bool).
    """

    id: str
    frontmatter: dict[str, Any]
    body: str


@dataclass(frozen=True)
class Artifact:
    """An artifact file from artifacts/{id}.md.

    Frontmatter keys (all optional beyond id/type):
      id, type (repo/paper/post/release/spec), evaluation, concept,
      relationship, first_seen, last_evaluated, source_url, github_repo, tags.
    """

    id: str
    frontmatter: dict[str, Any]
    body: str


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _vault_root() -> Path:
    return config.load().vault_path


def _concepts_dir() -> Path:
    return _vault_root() / "concepts"


def _artifacts_dir() -> Path:
    return _vault_root() / "artifacts"


def _signals_dir() -> Path:
    return _vault_root() / "signals"


def _digests_dir() -> Path:
    return _vault_root() / "digests"


def _concept_path(concept_id: str) -> Path:
    return _concepts_dir() / f"{concept_id}.md"


def _artifact_path(artifact_id: str) -> Path:
    return _artifacts_dir() / f"{artifact_id}.md"


def _signal_path(artifact_id: str) -> Path:
    return _signals_dir() / f"{artifact_id}.jsonl"


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
# Concepts
# ---------------------------------------------------------------------------


def read_concept(concept_id: str) -> "Concept | None":
    """Return the Concept or None if the file doesn't exist."""
    path = _concept_path(concept_id)
    if not path.is_file():
        return None
    post = frontmatter.load(str(path))
    return Concept(id=concept_id, frontmatter=dict(post.metadata), body=post.content)


def write_concept(concept: "Concept") -> None:
    """Serialize the concept and atomically replace concepts/{id}.md."""
    post = frontmatter.Post(concept.body, **concept.frontmatter)
    text = frontmatter.dumps(post)
    if not text.endswith("\n"):
        text += "\n"
    _atomic_write_text(_concept_path(concept.id), text)


def concept_exists(concept_id: str) -> bool:
    return _concept_path(concept_id).is_file()


def list_concepts(status: str | None = None) -> list["Concept"]:
    """Return concepts sorted by id, optionally filtered by lifecycle status."""
    concepts_dir = _concepts_dir()
    if not concepts_dir.is_dir():
        return []
    out: list[Concept] = []
    for path in sorted(concepts_dir.glob("*.md")):
        concept_id = path.stem
        concept = read_concept(concept_id)
        if concept is None:
            continue
        if status is not None and concept.frontmatter.get("status") != status:
            continue
        out.append(concept)
    return out


def find_artifacts_for_concept(concept_id: str) -> list["Artifact"]:
    """Return Artifact objects for all artifact ids listed in the concept's frontmatter.

    Missing artifact files are silently skipped (with a logged warning).
    """
    concept = read_concept(concept_id)
    if concept is None:
        return []
    artifact_entries = concept.frontmatter.get("artifacts") or []
    results: list[Artifact] = []
    for entry in artifact_entries:
        art_id = entry.get("id") if isinstance(entry, dict) else str(entry)
        if not art_id:
            continue
        art = read_artifact(art_id)
        if art is None:
            logger.warning("concept %s references missing artifact %s", concept_id, art_id)
            continue
        results.append(art)
    return results


def find_concept_for_artifact(artifact_id: str) -> "Concept | None":
    """Scan all concepts to find which one lists this artifact.

    Linear scan — acceptable at vault scale (hundreds of concepts). Returns
    the first match, or None if not found.
    """
    for concept in list_concepts():
        artifact_entries = concept.frontmatter.get("artifacts") or []
        for entry in artifact_entries:
            aid = entry.get("id") if isinstance(entry, dict) else str(entry)
            if aid == artifact_id:
                return concept
    return None


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def read_artifact(artifact_id: str) -> "Artifact | None":
    """Return the Artifact or None if the file doesn't exist."""
    path = _artifact_path(artifact_id)
    if not path.is_file():
        return None
    post = frontmatter.load(str(path))
    return Artifact(id=artifact_id, frontmatter=dict(post.metadata), body=post.content)


def write_artifact(artifact: "Artifact") -> None:
    """Serialize the artifact and atomically replace artifacts/{id}.md."""
    post = frontmatter.Post(artifact.body, **artifact.frontmatter)
    text = frontmatter.dumps(post)
    if not text.endswith("\n"):
        text += "\n"
    _atomic_write_text(_artifact_path(artifact.id), text)


def artifact_exists(artifact_id: str) -> bool:
    return _artifact_path(artifact_id).is_file()


def list_artifacts(
    evaluation: str | None = None,
    artifact_type: str | None = None,
    concept_id: str | None = None,
) -> list["Artifact"]:
    """Return artifacts sorted by id, optionally filtered by evaluation/type/concept."""
    artifacts_dir = _artifacts_dir()
    if not artifacts_dir.is_dir():
        return []
    out: list[Artifact] = []
    for path in sorted(artifacts_dir.glob("*.md")):
        artifact_id = path.stem
        artifact = read_artifact(artifact_id)
        if artifact is None:
            continue
        if evaluation is not None and artifact.frontmatter.get("evaluation") != evaluation:
            continue
        if artifact_type is not None and artifact.frontmatter.get("type") != artifact_type:
            continue
        if concept_id is not None and artifact.frontmatter.get("concept") != concept_id:
            continue
        out.append(artifact)
    return out


def find_artifact_by_github_repo(github_repo: str) -> list["Artifact"]:
    """Return artifacts whose frontmatter `github_repo` matches `owner/name` exactly."""
    matches: list[Artifact] = []
    for artifact in list_artifacts():
        if artifact.frontmatter.get("github_repo") == github_repo:
            matches.append(artifact)
    return matches


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def append_signal(artifact_id: str, snapshot: dict[str, Any]) -> None:
    """Append one JSON line to signals/{id}.jsonl, creating the file if missing."""
    path = _signal_path(artifact_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(snapshot, sort_keys=False, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(line)
        fp.write("\n")


def read_signals(artifact_id: str) -> list[dict[str, Any]]:
    """Return all signal snapshots for an artifact. [] if the file is missing."""
    path = _signal_path(artifact_id)
    if not path.is_file():
        logger.debug("signal file missing for %s", artifact_id)
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


def has_star_history_backfill(artifact_id: str) -> bool:
    """True if any signal entry has source='backfill' (star history already fetched)."""
    return any(s.get("source") == "backfill" for s in read_signals(artifact_id))


# ---------------------------------------------------------------------------
# Body helpers
# ---------------------------------------------------------------------------


_HISTORY_HEADING = "## History"


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
    # artifact + concept + signal under sentinel ids, exercises the helpers,
    # then deletes everything it created.
    smoke_aid = "__smoke_artifact__"
    smoke_cid = "__smoke_concept__"
    artifact_path = _artifact_path(smoke_aid)
    concept_path = _concept_path(smoke_cid)
    signal_path = _signal_path(smoke_aid)

    for p in (artifact_path, concept_path, signal_path):
        if p.exists():
            raise RuntimeError(
                f"smoke test artifacts already exist at {p}; refusing to clobber"
            )

    try:
        artifact_fm = {
            "id": smoke_aid,
            "type": "repo",
            "evaluation": "new",
            "concept": smoke_cid,
            "relationship": "implements",
            "first_seen": "2026-05-19",
            "last_evaluated": "2026-05-19",
            "tags": ["smoke", "test"],
            "github_repo": "smoke/test",
        }
        artifact_body = (
            "## What it is\n"
            "Smoke test artifact.\n\n"
            "## Evaluation rationale\n"
            "Just captured.\n\n"
            "## History\n"
            "- 2026-05-19: Captured.\n"
        )
        write_artifact(Artifact(id=smoke_aid, frontmatter=artifact_fm, body=artifact_body))
        assert artifact_exists(smoke_aid), "artifact_exists should be true after write"

        roundtrip = read_artifact(smoke_aid)
        assert roundtrip is not None
        assert roundtrip.frontmatter["github_repo"] == "smoke/test"
        assert "## History" in roundtrip.body

        # Filters.
        listed = list_artifacts(evaluation="new", artifact_type="repo")
        assert any(a.id == smoke_aid for a in listed), "list_artifacts filter missed"

        by_repo = find_artifact_by_github_repo("smoke/test")
        assert any(a.id == smoke_aid for a in by_repo), "find_artifact_by_github_repo missed"

        # Concept round-trip.
        concept_fm = {
            "id": smoke_cid,
            "label": "Smoke Test",
            "type": "concept",
            "status": "emerging",
            "first_seen": "2026-05-19",
            "last_evaluated": "2026-05-19",
            "relevance": "low",
            "tags": ["smoke"],
            "artifacts": [{"id": smoke_aid, "relationship": "implements", "weight": "primary"}],
        }
        concept_body = (
            "## What it is\nSmoke test concept.\n\n"
            "## Why it matters\nFor smoke testing.\n\n"
            "## Current assessment\nThis is a smoke test.\n"
        )
        write_concept(Concept(id=smoke_cid, frontmatter=concept_fm, body=concept_body))
        assert concept_exists(smoke_cid)
        ctrip = read_concept(smoke_cid)
        assert ctrip is not None and ctrip.frontmatter["label"] == "Smoke Test"

        # Cross-lookups.
        linked = find_artifacts_for_concept(smoke_cid)
        assert any(a.id == smoke_aid for a in linked), "find_artifacts_for_concept missed"
        back = find_concept_for_artifact(smoke_aid)
        assert back is not None and back.id == smoke_cid, "find_concept_for_artifact missed"

        # Signals.
        append_signal(smoke_aid, {"date": "2026-05-18", "stars": 10})
        append_signal(smoke_aid, {"date": "2026-05-19", "stars": 12})
        signals = read_signals(smoke_aid)
        assert signals == [
            {"date": "2026-05-18", "stars": 10},
            {"date": "2026-05-19", "stars": 12},
        ], f"unexpected signals: {signals!r}"

        # Embedding view drops the history section.
        stripped = body_for_embedding(artifact_body)
        assert "## History" not in stripped
        assert "Smoke test artifact." in stripped

        print("vault.py smoke test OK")
    finally:
        # Clean up — leave the vault as we found it.
        for p in (artifact_path, concept_path, signal_path):
            if p.exists():
                p.unlink()
