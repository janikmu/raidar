"""Search CLI — query the AI Radar vault.

Subcommands:
    raidar search keyword <query>
    raidar search semantic <query> [--top-k N]
    raidar search entity <id>
    raidar search signals <id>
    raidar search digest [--last N]
    raidar search list [--status S] [--type T]

Output is plain text — readable to a human in the terminal and parseable
by Claude reading stdout. Each command exits 0 on success (including the
"no matches" case), 1 on expected errors (entity not found, invalid
arguments), and 2 on infrastructure errors (e.g. the embeddings backend
is unreachable).
"""

from __future__ import annotations

import sys
from typing import Any

import openai
import typer

from lib import vault
from lib.embeddings import Index
from lib.entity_body import parse as parse_body
from lib.vault import (
    Entity,
    list_entities,
    read_entity,
    read_recent_digests,
    read_signals,
)

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------


def _first_line_what_it_is(body: str) -> str:
    """Return the first non-empty line under "## What it is", or "" if absent."""
    sections = parse_body(body)
    what = sections.get("What it is", "").strip()
    if not what:
        return ""
    for line in what.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _tags(entity: Entity) -> list[str]:
    raw = entity.frontmatter.get("tags") or []
    if isinstance(raw, str):
        return [raw]
    return [str(t) for t in raw]


def _format_entity_block(entity: Entity, header: str) -> str:
    tags = ", ".join(_tags(entity))
    summary = _first_line_what_it_is(entity.body)
    lines = [
        header,
        f"tags: {tags}",
    ]
    if summary:
        lines.append(summary)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# keyword
# ---------------------------------------------------------------------------


@app.command("keyword")
def keyword_cmd(
    query: str = typer.Argument(..., help="Substring to match (case-insensitive)."),
) -> None:
    """Substring match against id, tags, type, status, and the first line of 'What it is'."""
    q = query.lower().strip()
    if not q:
        print("ERROR: empty query.", file=sys.stderr)
        raise typer.Exit(code=1)

    matches: list[Entity] = []
    for entity in list_entities():
        fm = entity.frontmatter
        haystacks: list[str] = [
            str(fm.get("id", entity.id)),
            str(fm.get("type", "")),
            str(fm.get("status", "")),
            _first_line_what_it_is(entity.body),
        ]
        haystacks.extend(_tags(entity))
        if any(q in h.lower() for h in haystacks if h):
            matches.append(entity)

    if not matches:
        print("No matches.")
        return

    blocks: list[str] = []
    for entity in matches:
        fm = entity.frontmatter
        header = (
            f"== {entity.id} "
            f"({fm.get('type', '?')}, "
            f"{fm.get('status', '?')}, "
            f"relevance={fm.get('relevance', '?')}) =="
        )
        blocks.append(_format_entity_block(entity, header))
    print("\n\n".join(blocks))


# ---------------------------------------------------------------------------
# semantic
# ---------------------------------------------------------------------------


@app.command("semantic")
def semantic_cmd(
    query: str = typer.Argument(..., help="Natural-language query."),
    top_k: int = typer.Option(5, "--top-k", help="Number of results to return."),
) -> None:
    """Semantic search via the local embeddings index."""
    if top_k < 1:
        print("ERROR: --top-k must be >= 1.", file=sys.stderr)
        raise typer.Exit(code=1)

    index = Index()
    try:
        results = index.search(query, top_k=top_k)
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        print(
            f"ERROR: embeddings backend unreachable: {exc}",
            file=sys.stderr,
        )
        raise typer.Exit(code=2) from exc

    if not results:
        print("No matches.")
        return

    blocks: list[str] = []
    for entity_id, score in results:
        entity = read_entity(entity_id)
        header = f"== {entity_id} (score={score:.3f}) =="
        if entity is None:
            # Index references an entity that no longer exists in the vault.
            blocks.append(f"{header}\n(entity file missing from vault)")
            continue
        blocks.append(_format_entity_block(entity, header))
    print("\n\n".join(blocks))


# ---------------------------------------------------------------------------
# entity
# ---------------------------------------------------------------------------


def _serialize_entity(entity: Entity) -> str:
    """Re-render the entity as markdown (frontmatter + body) using python-frontmatter."""
    import frontmatter  # local import — only this command needs it

    post = frontmatter.Post(entity.body, **entity.frontmatter)
    text = frontmatter.dumps(post)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _signal_summary(signals: list[dict[str, Any]]) -> str:
    if not signals:
        return "(no signals recorded)"

    first = signals[0]
    last = signals[-1]

    def _fmt(snap: dict[str, Any]) -> str:
        return (
            f"{snap.get('date', '?')}  "
            f"stars={snap.get('stars', '?')} "
            f"forks={snap.get('forks', '?')} "
            f"commits_30d={snap.get('commits_30d', '?')}"
        )

    def _delta(field: str) -> str:
        a = first.get(field)
        b = last.get(field)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            d = b - a
            return f"{a}->{b} ({d:+d})" if isinstance(d, int) else f"{a}->{b} ({d:+.1f})"
        return f"{a}->{b}"

    lines = [
        f"== Signals ({len(signals)} snapshots) ==",
        f"First: {_fmt(first)}",
        f"Last:  {_fmt(last)}",
        f"Trend: stars {_delta('stars')}, commits_30d {_delta('commits_30d')}",
    ]
    return "\n".join(lines)


@app.command("entity")
def entity_cmd(
    entity_id: str = typer.Argument(..., metavar="ID", help="Entity id (filename stem)."),
) -> None:
    """Print the full entity markdown followed by a signal summary."""
    entity = read_entity(entity_id)
    if entity is None:
        print(f"Entity {entity_id!r} not found.", file=sys.stderr)
        raise typer.Exit(code=1)

    print(_serialize_entity(entity), end="")
    print()
    print(_signal_summary(read_signals(entity_id)))


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------


_DELTA_FIELDS = ("stars", "forks", "commits_30d", "open_issues")


def _format_delta_line(curr: dict[str, Any], prev: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in _DELTA_FIELDS:
        a = prev.get(field)
        b = curr.get(field)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            diff = b - a
            if isinstance(diff, int):
                parts.append(f"{field}={diff:+d}")
            else:
                parts.append(f"{field}={diff:+.1f}")
    delta_str = ", ".join(parts) if parts else "(no numeric deltas)"
    return f"  delta: {delta_str}   (vs. {prev.get('date', '?')})"


@app.command("signals")
def signals_cmd(
    entity_id: str = typer.Argument(..., metavar="ID", help="Entity id."),
) -> None:
    """Print each signal snapshot as one JSON line, with deltas vs. the previous one."""
    import json

    if not vault.entity_exists(entity_id):
        print(f"Entity {entity_id!r} not found.", file=sys.stderr)
        raise typer.Exit(code=1)

    signals = read_signals(entity_id)
    if not signals:
        print("(no signals recorded)")
        return

    prev: dict[str, Any] | None = None
    for snap in signals:
        print(json.dumps(snap, ensure_ascii=False, sort_keys=False))
        if prev is not None:
            print(_format_delta_line(snap, prev))
        prev = snap


# ---------------------------------------------------------------------------
# digest
# ---------------------------------------------------------------------------


@app.command("digest")
def digest_cmd(
    last: int = typer.Option(14, "--last", help="Window in days to include."),
) -> None:
    """Print recent digests (newest first) within the last N days."""
    if last < 1:
        print("ERROR: --last must be >= 1.", file=sys.stderr)
        raise typer.Exit(code=1)

    digests = read_recent_digests(within_days=last)
    if not digests:
        print(f"No digests within the last {last} days.")
        return

    for date_iso, body in digests:
        print(f"== Digest {date_iso} ==")
        print(body, end="" if body.endswith("\n") else "\n")
        print()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _format_row(cols: list[str], widths: list[int]) -> str:
    pieces: list[str] = []
    for i, (col, w) in enumerate(zip(cols, widths)):
        if i == len(cols) - 1:
            # Last column: don't pad — tags can be long.
            pieces.append(col)
        else:
            pieces.append(col.ljust(w))
    return "  ".join(pieces)


@app.command("list")
def list_cmd(
    status: str | None = typer.Option(None, "--status", help="Filter by status."),
    type_: str | None = typer.Option(None, "--type", help="Filter by type."),
) -> None:
    """List entities filtered by status and/or type."""
    entities = list_entities(status=status, type_=type_)
    # list_entities already sorts by id (filename stem). Keep that.

    headers = ["ID", "TYPE", "STATUS", "RELEVANCE", "LAST_EVALUATED", "TAGS"]
    rows: list[list[str]] = []
    for entity in entities:
        fm = entity.frontmatter
        rows.append(
            [
                entity.id,
                str(fm.get("type", "")),
                str(fm.get("status", "")),
                str(fm.get("relevance", "")),
                str(fm.get("last_evaluated", "")),
                ", ".join(_tags(entity)),
            ]
        )

    if not rows:
        # Still print the header so the output shape is predictable, then a note.
        print("  ".join(headers))
        filters = []
        if status is not None:
            filters.append(f"status={status}")
        if type_ is not None:
            filters.append(f"type={type_}")
        suffix = f" matching {' '.join(filters)}" if filters else ""
        print(f"(no entities{suffix})")
        return

    # Compute column widths from header + data (skip last column — tags free-form).
    widths: list[int] = []
    for col_idx, header in enumerate(headers):
        if col_idx == len(headers) - 1:
            widths.append(len(header))
            continue
        w = len(header)
        for row in rows:
            w = max(w, len(row[col_idx]))
        widths.append(w)

    print(_format_row(headers, widths))
    for row in rows:
        print(_format_row(row, widths))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app()
