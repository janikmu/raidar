"""Rename a concept — change its id (and optionally its label).

Rewrites the concept file under the new id, repoints every artifact's
`concept:` link, moves the embedding entry, and deletes the old file. The
artifact roster, prose, and per-artifact relationships are preserved. Use it
when a concept's slug no longer reflects the idea — e.g. after merges leave you
with `agent-skills-library` when the true concept is just `agent-skills`.

Usage:
    raidar rename-concept OLD NEW
    raidar rename-concept OLD NEW --label "Agent Skills"
    raidar rename-concept OLD NEW --dry-run
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import date

import typer

from lib import config as config_module
from lib import vault
from lib.body import parse_concept, render_concept
from lib.embeddings import Index
from lib.logging_setup import setup as setup_logging
from lib.vault import Artifact, Concept

log = logging.getLogger("jobs.rename")
app = typer.Typer(add_completion=False, help=__doc__)

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _history_bullets(body: str) -> list[str]:
    out: list[str] = []
    for line in parse_concept(body).get("History", "").splitlines():
        s = line.strip()
        if s.startswith("- "):
            out.append(s[2:].strip())
        elif s.startswith("-"):
            out.append(s[1:].strip())
    return out


@app.command(name="rename-concept")
def rename_concept(
    old: str = typer.Argument(..., help="Current concept id."),
    new: str = typer.Argument(..., help="New concept id (kebab-case slug)."),
    label: str = typer.Option(None, "--label", help="New human-readable label (optional)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan; write nothing."),
) -> None:
    """Rename concept OLD to NEW."""
    cfg = config_module.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)
    today = date.today().isoformat()

    if old == new:
        print("ERROR: old and new ids are the same.", file=sys.stderr)
        raise typer.Exit(code=1)
    if not _SLUG_RE.match(new):
        print(f"ERROR: {new!r} is not a valid kebab-case slug.", file=sys.stderr)
        raise typer.Exit(code=1)

    concept = vault.read_concept(old)
    if concept is None:
        print(f"ERROR: concept {old!r} not found.", file=sys.stderr)
        raise typer.Exit(code=1)
    if vault.concept_exists(new):
        print(f"ERROR: concept {new!r} already exists — use `merge-concept` instead.", file=sys.stderr)
        raise typer.Exit(code=1)

    artifacts = vault.list_artifacts(concept_id=old)
    print(f"Rename plan: {old} → {new}")
    print(f"  artifacts to repoint: {len(artifacts)}  {[a.id for a in artifacts]}")
    if label:
        print(f"  label: {concept.frontmatter.get('label')!r} → {label!r}")
    if dry_run:
        print("\n[dry-run] no files changed.")
        raise typer.Exit(code=0)

    # ---- 1. repoint artifacts ----------------------------------------------
    for a in artifacts:
        new_fm = dict(a.frontmatter)
        new_fm["concept"] = new
        vault.write_artifact(Artifact(id=a.id, frontmatter=new_fm, body=a.body))

    # ---- 2. write concept under the new id ---------------------------------
    sections = parse_concept(concept.body)
    history = _history_bullets(concept.body)
    history.append(f"{today}: Renamed from {old}.")
    artifact_rows = [
        {"id": a.id, "type": a.frontmatter.get("type", "repo"),
         "evaluation": a.frontmatter.get("evaluation", "new")}
        for a in artifacts
    ]
    new_body = render_concept(
        what_it_is=sections.get("What it is", ""),
        why_it_matters=sections.get("Why it matters", ""),
        current_assessment=sections.get("Current assessment", ""),
        artifact_rows=artifact_rows,
        history_bullets=history,
    )
    new_fm = dict(concept.frontmatter)
    new_fm["id"] = new
    if label:
        new_fm["label"] = label
    new_fm["last_evaluated"] = today
    vault.write_concept(Concept(id=new, frontmatter=new_fm, body=new_body))

    # ---- 3. delete old + move embedding ------------------------------------
    vault.delete_concept(old)
    try:
        cidx = Index(cfg, layer="concepts")
        cidx.delete(old)
        cidx.upsert(new, new_body)
    except Exception as exc:  # noqa: BLE001
        log.warning("embedding update after rename failed (%s); run `raidar reindex`", exc)

    print(f"\n✓ Renamed {old} → {new}: repointed {len(artifacts)} artifact(s).")


if __name__ == "__main__":
    app()
