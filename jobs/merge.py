"""Merge one concept into another — the remediation for duplicate/forked concepts.

Reassigns every artifact under SOURCE to TARGET, unions tags, records the merge
in TARGET's history, rebuilds TARGET's artifact-summary table and embedding, then
deletes SOURCE (file + embedding entry). TARGET is the keeper: its label, status,
relevance and prose are preserved.

Usage:
    raidar merge-concept SOURCE TARGET
    raidar merge-concept SOURCE TARGET --dry-run    # show the plan, write nothing

Example:
    raidar merge-concept agent-skills-library-2 agent-skills-library
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from typing import Any

import typer

from lib import config as config_module
from lib import vault
from lib.body import parse_concept, render_concept
from lib.embeddings import Index
from lib.logging_setup import setup as setup_logging
from lib.vault import Artifact, Concept

log = logging.getLogger("jobs.merge")
app = typer.Typer(add_completion=False, help=__doc__)


def _history_bullets(body: str) -> list[str]:
    sections = parse_concept(body)
    out: list[str] = []
    for line in sections.get("History", "").splitlines():
        s = line.strip()
        if s.startswith("- "):
            out.append(s[2:].strip())
        elif s.startswith("-"):
            out.append(s[1:].strip())
    return out


def _artifacts_under(concept: Concept) -> dict[str, dict[str, str]]:
    """Return {artifact_id: {relationship, weight}} for everything that should
    live under `concept`: its listed artifacts plus any artifact whose
    frontmatter concept: points at it (in case the two ever diverged)."""
    out: dict[str, dict[str, str]] = {}
    for entry in concept.frontmatter.get("artifacts") or []:
        if isinstance(entry, dict) and entry.get("id"):
            out[entry["id"]] = {
                "relationship": entry.get("relationship", "implements"),
                "weight": entry.get("weight", "primary"),
            }
    for art in vault.list_artifacts(concept_id=concept.id):
        out.setdefault(art.id, {
            "relationship": art.frontmatter.get("relationship", "implements"),
            "weight": "primary",
        })
    return out


@app.command(name="merge-concept")
def merge_concept(
    source: str = typer.Argument(..., help="Concept id to merge FROM (will be deleted)."),
    target: str = typer.Argument(..., help="Concept id to merge INTO (the keeper)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan; write nothing."),
) -> None:
    """Merge SOURCE concept into TARGET concept."""
    cfg = config_module.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)
    today = date.today().isoformat()

    if source == target:
        print("ERROR: source and target are the same concept.", file=sys.stderr)
        raise typer.Exit(code=1)

    src = vault.read_concept(source)
    tgt = vault.read_concept(target)
    if src is None:
        print(f"ERROR: source concept {source!r} not found.", file=sys.stderr)
        raise typer.Exit(code=1)
    if tgt is None:
        print(f"ERROR: target concept {target!r} not found.", file=sys.stderr)
        raise typer.Exit(code=1)

    src_arts = _artifacts_under(src)
    tgt_arts = _artifacts_under(tgt)
    moved = [aid for aid in src_arts if aid not in tgt_arts]

    # ---- merged artifact roster (target wins on relationship/weight) --------
    merged_entries: dict[str, dict[str, str]] = dict(tgt_arts)
    for aid, meta in src_arts.items():
        merged_entries.setdefault(aid, meta)

    # ---- merged frontmatter -------------------------------------------------
    tgt_fm: dict[str, Any] = dict(tgt.frontmatter)
    src_tags = list(src.frontmatter.get("tags") or [])
    tgt_tags = list(tgt_fm.get("tags") or [])
    merged_tags = list(dict.fromkeys([*tgt_tags, *src_tags]))[:8]

    print(f"Merge plan: {source} → {target}")
    print(f"  artifacts to move:   {len(moved)}  {moved if moved else ''}")
    print(f"  target artifacts:    {len(tgt_arts)} → {len(merged_entries)}")
    print(f"  tags:                {tgt_tags} + {src_tags} → {merged_tags}")
    print(f"  target keeps:        label={tgt_fm.get('label')!r} status={tgt_fm.get('status')!r}")
    if dry_run:
        print("\n[dry-run] no files changed.")
        raise typer.Exit(code=0)

    # ---- 1. reassign artifacts ---------------------------------------------
    for aid in src_arts:
        art = vault.read_artifact(aid)
        if art is None:
            log.warning("artifact %s listed under %s is missing; skipping", aid, source)
            continue
        if art.frontmatter.get("concept") == target:
            continue
        new_fm = dict(art.frontmatter)
        new_fm["concept"] = target
        vault.write_artifact(Artifact(id=aid, frontmatter=new_fm, body=art.body))

    # ---- 2. rebuild target body --------------------------------------------
    sections = parse_concept(tgt.body)
    history = _history_bullets(tgt.body)
    history.append(f"{today}: Merged concept {source} ({len(moved)} artifact(s) moved).")

    artifact_rows = []
    for aid in merged_entries:
        art = vault.read_artifact(aid)
        if art is None:
            continue
        artifact_rows.append({
            "id": aid,
            "type": art.frontmatter.get("type", "repo"),
            "evaluation": art.frontmatter.get("evaluation", "new"),
        })

    new_body = render_concept(
        what_it_is=sections.get("What it is", ""),
        why_it_matters=sections.get("Why it matters", ""),
        current_assessment=sections.get("Current assessment", ""),
        artifact_rows=artifact_rows,
        history_bullets=history,
    )

    tgt_fm["artifacts"] = [
        {"id": aid, "relationship": m["relationship"], "weight": m["weight"]}
        for aid, m in merged_entries.items()
    ]
    tgt_fm["tags"] = merged_tags
    tgt_fm["last_evaluated"] = today
    src_first = src.frontmatter.get("first_seen")
    tgt_first = tgt_fm.get("first_seen")
    if src_first and (not tgt_first or src_first < tgt_first):
        tgt_fm["first_seen"] = src_first

    vault.write_concept(Concept(id=target, frontmatter=tgt_fm, body=new_body))

    # ---- 3. delete source ---------------------------------------------------
    vault.delete_concept(source)

    # ---- 4. embeddings ------------------------------------------------------
    try:
        cidx = Index(cfg, layer="concepts")
        cidx.delete(source)
        cidx.upsert(target, new_body)
    except Exception as exc:  # noqa: BLE001
        log.warning("embedding update after merge failed (%s); run `raidar reindex`", exc)

    print(f"\n✓ Merged {source} into {target}: moved {len(moved)} artifact(s); deleted {source}.")
    print("  Run `raidar health` to confirm.")


if __name__ == "__main__":
    app()
