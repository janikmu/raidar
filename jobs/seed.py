"""Seed the vault with canonical concept entries from training-data knowledge.

For well-known concepts (MCP, RAG, ReAct, etc.) the LLM has reliable knowledge
from its training data alone. This job seeds those concepts with status +
prose so:

  (a) the vault has correct lifecycle status on industry-standard concepts
      from the start, instead of forcing the artifact-driven re-eval pass
      to fabricate world knowledge from a list of unfamiliar repos, and
  (b) subsequent captures attach matching artifacts to the existing concept
      instead of spawning a near-duplicate.

The seed list lives at `lib/seed_concepts.yaml`. The LLM may return
`confident: false` to decline a concept it can't speak to with high
confidence — that concept is skipped (no file written).

Usage:
    uv run python -m jobs.seed              # seed all entries from YAML, skip existing
    uv run python -m jobs.seed --id mcp     # seed only this id; repeatable
    uv run python -m jobs.seed --list       # print the seed list, exit
    uv run python -m jobs.seed --dry-run    # show plan without writing
    uv run python -m jobs.seed --force      # overwrite existing concept files
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import typer
import yaml

from lib import config as config_module
from lib import vault
from lib.body import render_concept
from lib.llm import AllProvidersFailed, Router
from lib.logging_setup import setup as setup_logging
from lib.vault import Concept

log = logging.getLogger("jobs.seed")
app = typer.Typer(add_completion=False, help=__doc__)

_SEED_YAML_PATH = Path(__file__).resolve().parent.parent / "lib" / "seed_concepts.yaml"

_STATUS_ENUM = ["emerging", "watch", "invest", "common", "superseded", "abandoned"]

_SEED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["confident"],
    "properties": {
        "confident": {"type": "boolean"},
        "status": {"type": "string", "enum": _STATUS_ENUM},
        "what_it_is": {"type": "string"},
        "why_it_matters": {"type": "string"},
        "current_assessment": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}

_SEED_SYSTEM = (
    "You are seeding a concept entry for a personal AI-tooling knowledge base. "
    "You receive a concept id, label, and one-phrase hint. Use ONLY your training-data "
    "knowledge of the AI/dev-tooling landscape to describe the concept and assign a "
    "lifecycle status based on REAL-WORLD adoption — not on artifact counts.\n\n"
    "Status rules:\n"
    "  'emerging' — recent, unclear trajectory.\n"
    "  'watch' — gaining traction, several implementations exist, not yet broadly adopted.\n"
    "  'invest' — mature, has recommended implementations, worth investing in now.\n"
    "  'common' — stable and widespread industry-standard, effectively table-stakes.\n"
    "  'superseded' — a clearly better approach has replaced this.\n"
    "  'abandoned' — community dissolved, no active implementations.\n\n"
    "Output rules:\n"
    "  - Set confident=true ONLY if you have high-confidence knowledge of this concept "
    "from training data. If unsure, return confident=false and omit the other fields.\n"
    "  - When confident=true, all of status, what_it_is, why_it_matters, current_assessment "
    "are REQUIRED.\n"
    "  - Prose sections are 2-4 sentences each, terse and factual.\n"
    "  - tags: up to 6 short kebab-case tags. Optional.\n"
    "  - Do not fabricate. Do not speculate about post-training-cutoff developments.\n"
    "Return only JSON matching the provided schema."
)


def _load_seed_list() -> list[dict[str, str]]:
    if not _SEED_YAML_PATH.is_file():
        raise FileNotFoundError(f"seed list not found: {_SEED_YAML_PATH}")
    data = yaml.safe_load(_SEED_YAML_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "concepts" not in data:
        raise ValueError(f"{_SEED_YAML_PATH}: missing top-level `concepts:` list")
    entries = data["concepts"]
    if not isinstance(entries, list):
        raise ValueError(f"{_SEED_YAML_PATH}: `concepts` must be a list")
    for e in entries:
        for k in ("id", "label", "hint"):
            if not isinstance(e.get(k), str) or not e[k].strip():
                raise ValueError(f"{_SEED_YAML_PATH}: entry missing required field {k!r}: {e!r}")
    return entries


def _seed_one(
    entry: dict[str, str],
    router: Router,
    today: str,
) -> Concept | None:
    """Call the LLM to produce a concept entry. Returns None if LLM declines or fails."""
    prompt = (
        f"## Concept\n"
        f"id: {entry['id']}\n"
        f"label: {entry['label']}\n"
        f"hint: {entry['hint']}\n\n"
        "Produce a concept entry per the schema. If you cannot speak to this "
        "concept with high confidence from training data, return confident=false."
    )

    try:
        completion = router.generate(
            task="enrichment",
            prompt=prompt,
            system=_SEED_SYSTEM,
            response_schema=_SEED_SCHEMA,
            max_tokens=1024,
        )
    except AllProvidersFailed as exc:
        log.error("seed %s: all providers failed: %s", entry["id"], exc)
        return None

    payload = completion.parsed
    if not isinstance(payload, dict):
        log.error("seed %s: non-JSON response: %s", entry["id"], completion.text[:200])
        return None

    if not payload.get("confident"):
        log.info("seed %s: LLM declined (confident=false)", entry["id"])
        return None

    missing = [k for k in ("status", "what_it_is", "why_it_matters", "current_assessment")
               if not isinstance(payload.get(k), str) or not payload[k].strip()]
    if missing:
        log.error("seed %s: confident=true but missing fields: %s", entry["id"], missing)
        return None

    tags = payload.get("tags") or []
    if not isinstance(tags, list):
        tags = []

    fm: dict[str, Any] = {
        "id": entry["id"],
        "label": entry["label"],
        "type": "concept",
        "status": payload["status"],
        "first_seen": today,
        "last_evaluated": today,
        "relevance": "high",
        "tags": [t for t in tags if isinstance(t, str)][:6],
        "artifacts": [],
        "seeded": True,
        "seeded_at": today,
        "seeded_model": f"{completion.provider_name}:{completion.model}",
    }

    body = render_concept(
        what_it_is=payload["what_it_is"],
        why_it_matters=payload["why_it_matters"],
        current_assessment=payload["current_assessment"],
        artifact_rows=[],
        history_bullets=[f"{today}: Seeded from training-data ({completion.provider_name})."],
    )
    return Concept(id=entry["id"], frontmatter=fm, body=body)


@app.command()
def seed(
    ids: list[str] = typer.Option(
        None, "--id", help="Seed only this id (repeatable). Must exist in lib/seed_concepts.yaml.",
    ),
    list_only: bool = typer.Option(False, "--list", help="Print the seed list and exit."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only — no LLM calls, no writes."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing concept files. Default skips them.",
    ),
) -> None:
    """Seed the vault with canonical concept entries from training-data knowledge."""
    cfg = config_module.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)
    today = date.today().isoformat()

    entries = _load_seed_list()
    entries_by_id = {e["id"]: e for e in entries}

    if list_only:
        print(f"Seed list ({len(entries)} entries) at {_SEED_YAML_PATH}:")
        for e in entries:
            print(f"  - {e['id']:<35}  {e['label']}")
            print(f"      hint: {e['hint']}")
        raise typer.Exit(code=0)

    if ids:
        unknown = [i for i in ids if i not in entries_by_id]
        if unknown:
            print(f"ERROR: ids not in seed list: {unknown}")
            print(f"Run `raidar seed --list` to see available ids.")
            raise typer.Exit(code=1)
        targets = [entries_by_id[i] for i in ids]
    else:
        targets = entries

    to_seed: list[dict[str, str]] = []
    skipped_existing: list[str] = []
    for e in targets:
        if vault.concept_exists(e["id"]) and not force:
            skipped_existing.append(e["id"])
        else:
            to_seed.append(e)

    print(f"\nSeed plan ({today}):")
    print(f"  to seed:           {len(to_seed)}")
    print(f"  skipped (exists):  {len(skipped_existing)}  {'(use --force to overwrite)' if skipped_existing else ''}")
    if skipped_existing:
        for i in skipped_existing:
            print(f"    - {i}")
    if not to_seed:
        print("\nNothing to do.")
        raise typer.Exit(code=0)

    if dry_run:
        print("\n[dry-run] Would call LLM for:")
        for e in to_seed:
            print(f"  - {e['id']}  ({e['label']})")
        raise typer.Exit(code=0)

    router = Router(cfg)
    if not router.available_providers("enrichment"):
        print("ERROR: no LLM providers available for task=enrichment.")
        raise typer.Exit(code=1)

    print("")
    n_written = 0
    n_declined = 0
    for i, entry in enumerate(to_seed, 1):
        concept = _seed_one(entry, router, today)
        if concept is None:
            n_declined += 1
            print(f"  [{i}/{len(to_seed)}] {entry['id']:<35}  declined / failed")
            continue
        vault.write_concept(concept)
        n_written += 1
        print(f"  [{i}/{len(to_seed)}] {entry['id']:<35}  status={concept.frontmatter['status']}")

    print(f"\nSeed complete: {n_written} written, {n_declined} declined / failed.")


if __name__ == "__main__":
    app()
