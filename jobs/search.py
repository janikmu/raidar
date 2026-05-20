"""Search CLI — query the AI Radar vault.

Subcommands:
    raidar search keyword <query>
    raidar search semantic <query> [--top-k N]
    raidar search concept <id>
    raidar search artifact <id>
    raidar search signals <id>
    raidar search digest [--last N]
    raidar search list-concepts [--status S]
    raidar search list-artifacts [--evaluation E] [--type T]
    raidar search pending

Output is plain text — readable to a human in the terminal and parseable
by Claude reading stdout. Each command exits 0 on success (including the
"no matches" case), 1 on expected errors, and 2 on infrastructure errors.
"""

from __future__ import annotations

import sys
from typing import Any

import openai
import typer

from lib import vault
from lib.embeddings import Index
from lib.body import parse as parse_body
from lib.vault import (
    Artifact,
    Concept,
    list_artifacts,
    list_concepts,
    read_artifact,
    read_concept,
    read_recent_digests,
    read_signals,
)

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------


def _first_line(body: str, section: str) -> str:
    """Return the first non-empty line of a named body section."""
    sections = parse_body(body)
    text = sections.get(section, "").strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _tags_str(fm: dict) -> str:
    raw = fm.get("tags") or []
    if isinstance(raw, str):
        return raw
    return ", ".join(str(t) for t in raw)


def _format_concept_block(concept: Concept) -> str:
    fm = concept.frontmatter
    n_arts = len(fm.get("artifacts") or [])
    summary = _first_line(concept.body, "What it is")
    lines = [
        f"concept:{concept.id}  status={fm.get('status','?')}  relevance={fm.get('relevance','?')}  artifacts={n_arts}",
        f"label: {fm.get('label', concept.id)}",
        f"tags: {_tags_str(fm)}",
    ]
    if summary:
        lines.append(summary)
    return "\n".join(lines)


def _format_artifact_block(artifact: Artifact) -> str:
    fm = artifact.frontmatter
    summary = _first_line(artifact.body, "What it is")
    lines = [
        f"artifact:{artifact.id}  type={fm.get('type','?')}  evaluation={fm.get('evaluation','?')}",
        f"concept: {fm.get('concept', '?')}  relationship: {fm.get('relationship', '?')}",
        f"tags: {_tags_str(fm)}",
    ]
    if summary:
        lines.append(summary)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# keyword
# ---------------------------------------------------------------------------


@app.command()
def keyword(
    query: str = typer.Argument(..., help="Text to match against concept/artifact frontmatter."),
) -> None:
    """Search concepts and artifacts by keyword (frontmatter fields + tags)."""
    q = query.lower()

    found = False

    # Search concepts
    for concept in list_concepts():
        fm = concept.frontmatter
        haystack = " ".join([
            concept.id,
            fm.get("label", ""),
            fm.get("status", ""),
            _tags_str(fm),
        ]).lower()
        if q in haystack:
            print(_format_concept_block(concept))
            print()
            found = True

    # Search artifacts
    for artifact in list_artifacts():
        fm = artifact.frontmatter
        haystack = " ".join([
            artifact.id,
            fm.get("type", ""),
            fm.get("evaluation", ""),
            fm.get("concept", ""),
            fm.get("github_repo", ""),
            _tags_str(fm),
        ]).lower()
        if q in haystack:
            print(_format_artifact_block(artifact))
            print()
            found = True

    if not found:
        print(f"No concepts or artifacts found matching {query!r}")


# ---------------------------------------------------------------------------
# semantic
# ---------------------------------------------------------------------------


@app.command()
def semantic(
    query: str = typer.Argument(..., help="Natural language query."),
    top_k: int = typer.Option(5, "--top-k", help="Number of results to return."),
) -> None:
    """Semantic search across both concepts and artifacts."""
    from lib import config
    cfg = config.load()

    results: list[tuple[str, str, float]] = []  # (layer, id, score)

    for layer in ("concepts", "artifacts"):
        try:
            idx = Index(cfg, layer=layer)
            hits = idx.search(query, top_k=top_k)
            results.extend((layer, item_id, score) for item_id, score in hits)
        except openai.APIConnectionError as exc:
            print(f"WARNING: embedding backend unreachable for {layer}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: semantic search failed for {layer}: {exc}", file=sys.stderr)

    if not results:
        print("No semantic results found (embedding backend may be offline).")
        raise typer.Exit(code=0)

    results.sort(key=lambda x: x[2], reverse=True)
    results = results[:top_k]

    for layer, item_id, score in results:
        if layer == "concepts":
            concept = read_concept(item_id)
            if concept:
                print(f"[{score:+.3f}] {_format_concept_block(concept)}")
        else:
            artifact = read_artifact(item_id)
            if artifact:
                print(f"[{score:+.3f}] {_format_artifact_block(artifact)}")
        print()


# ---------------------------------------------------------------------------
# concept
# ---------------------------------------------------------------------------


@app.command()
def concept(
    concept_id: str = typer.Argument(..., help="Concept id."),
) -> None:
    """Display a concept file in full."""
    c = read_concept(concept_id)
    if c is None:
        print(f"ERROR: concept {concept_id!r} not found.", file=sys.stderr)
        raise typer.Exit(code=1)
    fm = c.frontmatter
    print(f"# {fm.get('label', concept_id)}  [{fm.get('status', '?')}]")
    print(f"id={concept_id}  first_seen={fm.get('first_seen','?')}  last_evaluated={fm.get('last_evaluated','?')}")
    print(f"relevance={fm.get('relevance','?')}  tags={_tags_str(fm)}")
    artifact_entries = fm.get("artifacts") or []
    if artifact_entries:
        print(f"\nArtifacts ({len(artifact_entries)}):")
        for entry in artifact_entries:
            if isinstance(entry, dict):
                print(f"  - {entry.get('id')}  ({entry.get('relationship','?')}, {entry.get('weight','?')})")
    print()
    print(c.body)


# ---------------------------------------------------------------------------
# artifact
# ---------------------------------------------------------------------------


@app.command()
def artifact(
    artifact_id: str = typer.Argument(..., help="Artifact id."),
) -> None:
    """Display an artifact file in full."""
    a = read_artifact(artifact_id)
    if a is None:
        print(f"ERROR: artifact {artifact_id!r} not found.", file=sys.stderr)
        raise typer.Exit(code=1)
    fm = a.frontmatter
    print(f"# artifact:{artifact_id}  [{fm.get('evaluation','?')}]")
    print(f"type={fm.get('type','?')}  concept={fm.get('concept','?')}  relationship={fm.get('relationship','?')}")
    print(f"first_seen={fm.get('first_seen','?')}  last_evaluated={fm.get('last_evaluated','?')}")
    if fm.get("github_repo"):
        print(f"github: https://github.com/{fm['github_repo']}")
    print(f"tags={_tags_str(fm)}")
    print()
    print(a.body)


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------


@app.command()
def signals(
    artifact_id: str = typer.Argument(..., help="Artifact id to show signals for."),
    last: int = typer.Option(20, "--last", help="Number of most recent signals to show."),
) -> None:
    """Show signal history for an artifact."""
    rows = read_signals(artifact_id)
    if not rows:
        print(f"No signals found for {artifact_id!r}")
        raise typer.Exit(code=0)
    rows = rows[-last:]
    print(f"signals for {artifact_id} (showing last {len(rows)}):\n")
    for row in rows:
        parts: list[str] = [row.get("date", "?")]
        for key in ("stars", "forks", "commits_30d", "open_issues", "evaluation"):
            val = row.get(key)
            if val is not None:
                parts.append(f"{key}={val}")
        src = row.get("source")
        if src:
            parts.append(f"[{src}]")
        print("  " + "  ".join(parts))


# ---------------------------------------------------------------------------
# digest
# ---------------------------------------------------------------------------


@app.command()
def digest(
    last: int = typer.Option(1, "--last", help="Number of recent digests to show."),
) -> None:
    """Show recent weekly digests."""
    digests = read_recent_digests(n=last)
    if not digests:
        print("No digests found.")
        raise typer.Exit(code=0)
    for date_iso, content in digests:
        print(f"{'='*60}")
        print(f"Digest: {date_iso}")
        print(f"{'='*60}")
        print(content)
        print()


# ---------------------------------------------------------------------------
# list-concepts
# ---------------------------------------------------------------------------


@app.command(name="list-concepts")
def list_concepts_cmd(
    status: str | None = typer.Option(None, "--status", help="Filter by lifecycle status."),
) -> None:
    """List all concepts, optionally filtered by status."""
    concepts = list_concepts(status=status)
    if not concepts:
        msg = f"No concepts with status={status!r}" if status else "No concepts in vault"
        print(msg)
        return
    header = f"{'ID':<40} {'STATUS':<12} {'RELEVANCE':<10} {'ARTIFACTS'}"
    print(header)
    print("-" * len(header))
    for c in concepts:
        fm = c.frontmatter
        n_arts = len(fm.get("artifacts") or [])
        print(
            f"{c.id:<40} {fm.get('status', '?'):<12} "
            f"{fm.get('relevance', '?'):<10} {n_arts}"
        )


# ---------------------------------------------------------------------------
# list-artifacts
# ---------------------------------------------------------------------------


@app.command(name="list-artifacts")
def list_artifacts_cmd(
    evaluation: str | None = typer.Option(None, "--evaluation", help="Filter by evaluation."),
    type_: str | None = typer.Option(None, "--type", metavar="TYPE", help="Filter by artifact type."),
    concept_id: str | None = typer.Option(None, "--concept", help="Filter by concept id."),
) -> None:
    """List artifacts, optionally filtered by evaluation/type/concept."""
    artifacts = list_artifacts(
        evaluation=evaluation, artifact_type=type_, concept_id=concept_id
    )
    if not artifacts:
        print("No artifacts matched the filters.")
        return
    header = f"{'ID':<45} {'TYPE':<8} {'EVAL':<12} {'CONCEPT'}"
    print(header)
    print("-" * len(header))
    for a in artifacts:
        fm = a.frontmatter
        print(
            f"{a.id:<45} {fm.get('type', '?'):<8} "
            f"{fm.get('evaluation', '?'):<12} {fm.get('concept', '?')}"
        )


# ---------------------------------------------------------------------------
# pending
# ---------------------------------------------------------------------------


@app.command()
def pending() -> None:
    """List concepts flagged with review_needed=true."""
    flagged = [
        c for c in list_concepts()
        if c.frontmatter.get("review_needed")
    ]
    if not flagged:
        print("No concepts flagged for review.")
        return
    print(f"Concepts needing review ({len(flagged)}):\n")
    for c in flagged:
        fm = c.frontmatter
        print(f"  {c.id}  [{fm.get('status','?')}]  {fm.get('label', '')}")


if __name__ == "__main__":
    app()
