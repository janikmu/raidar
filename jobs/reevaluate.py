"""Force re-evaluation of entities using their full signal history.

Unlike the weekly enrich job (which only touches emerging/watch and uses a
short signal tail), reevaluate:

  - Processes ALL entities regardless of current status (or a filtered set).
  - Feeds the LLM the *full* signal history so backfill context is used.
  - Applies the same status heuristics as capture but with ground-truth data.
  - Can also correct coarse type classifications (e.g. 'tool' → 'agent-framework').

Useful after a bulk backfill pass when many established repos are incorrectly
sitting at 'emerging' or 'watch'.

Usage:
    raidar reevaluate                        # all entities
    raidar reevaluate --only ID              # single entity
    raidar reevaluate --status emerging      # only emerging
    raidar reevaluate --dry-run              # print plan, no writes
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

import typer
from typing import Annotated

from lib import config as config_module
from lib import vault
from lib.embeddings import Index
from lib.entity_body import HISTORY, parse, render
from lib.llm import AllProvidersFailed, Router
from lib.logging_setup import setup as setup_logging
from lib.vault import Entity

log = logging.getLogger("jobs.reevaluate")

app = typer.Typer(add_completion=False, help=__doc__)

# ---------------------------------------------------------------------------
# Schema / prompts
# ---------------------------------------------------------------------------

_STATUSES = ["emerging", "watch", "adopt", "skip", "settled"]
_TYPES = [
    "agent-framework", "llm-client", "inference", "rag",
    "evaluation", "data", "fine-tuning", "tool", "model",
    "platform", "paper", "pattern",
]

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["new_status", "new_type", "prose_changed", "rationale"],
    "properties": {
        "new_status": {"type": "string", "enum": _STATUSES},
        "new_type": {"type": "string", "enum": _TYPES},
        "prose_changed": {"type": "boolean"},
        "new_what_it_is": {"type": ["string", "null"]},
        "new_why_it_matters": {"type": ["string", "null"]},
        "new_current_assessment": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
}

_SYSTEM_PROMPT = (
    "You are re-evaluating entities in an Information Systems researcher's "
    "personal AI-tooling radar. You have access to the entity's full star "
    "history (backfill signals) plus any live enrich signals. Use this data "
    "as ground truth — do not guess from prose alone. "
    "Status rules (apply strictly): "
    "'emerging' = project < 6 months old OR < 300 stars; "
    "'watch' = 6 months–2 years old OR 300–5k stars; "
    "'adopt' = > 2 years old AND > 5k stars, or in broad production use. "
    "A well-established repo with years of history and thousands of stars "
    "must NOT stay 'emerging'. "
    "Also correct the type if the current one is too generic — pick the "
    "most specific match from the enum. "
    "Rewrite prose only when the status changes or the current prose is "
    "clearly stale/wrong; otherwise set prose_changed=false. "
    "Keep prose terse — one short paragraph per section. "
    "Return JSON matching the provided schema."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_history_bullets(body: str) -> list[str]:
    sections = parse(body)
    hist_block = sections.get(HISTORY, "")
    if not hist_block.strip():
        return []
    out: list[str] = []
    for line in hist_block.splitlines():
        s = line.strip()
        if s.startswith("- "):
            out.append(s[2:].strip())
        elif s.startswith("-"):
            out.append(s[1:].strip())
    return out


def _build_prompt(context_md: str, entity: Entity, signals: list[dict[str, Any]]) -> str:
    fm_dump = json.dumps(entity.frontmatter, indent=2, sort_keys=True, default=str)
    signals_dump = json.dumps(signals, indent=2, default=str)
    return (
        "# Researcher context\n"
        f"{context_md.strip()}\n\n"
        "# Entity under review\n"
        f"## frontmatter\n```json\n{fm_dump}\n```\n\n"
        f"## body\n{entity.body.strip()}\n\n"
        "# Full signal history (oldest first)\n"
        f"```json\n{signals_dump}\n```\n\n"
        "# Task\n"
        "Re-evaluate this entity. Correct the status and type based on the "
        "signal history above (not from the current frontmatter). "
        f"Status options: {', '.join(_STATUSES)}. "
        f"Type options: {', '.join(_TYPES)}. "
        "Return JSON per the schema. "
        "If prose_changed is false, you may leave new_what_it_is / "
        "new_why_it_matters / new_current_assessment as null."
    )


# ---------------------------------------------------------------------------
# Per-entity logic
# ---------------------------------------------------------------------------


@dataclass
class _Result:
    id: str
    old_status: str
    old_type: str
    new_status: str | None = None
    new_type: str | None = None
    prose_updated: bool = False
    rationale: str = ""
    error: str | None = None


def _reevaluate_entity(
    entity: Entity,
    *,
    cfg: config_module.Config,
    context_md: str,
    router: Router,
    dry_run: bool,
    index: Index | None,
    today: str,
) -> _Result:
    result = _Result(
        id=entity.id,
        old_status=entity.frontmatter.get("status", "unknown"),
        old_type=entity.frontmatter.get("type", "tool"),
    )

    signals = vault.read_signals(entity.id)
    prompt = _build_prompt(context_md, entity, signals)

    try:
        completion = router.generate(
            task="reevaluation",
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            response_schema=_RESPONSE_SCHEMA,
            max_tokens=2048,
        )
    except AllProvidersFailed as exc:
        result.error = f"llm: {exc}"
        return result

    payload = completion.parsed
    if not isinstance(payload, dict):
        result.error = f"llm: non-JSON response: {completion.text[:200]!r}"
        return result

    new_status = payload.get("new_status") or result.old_status
    new_type = payload.get("new_type") or result.old_type
    prose_changed = bool(payload.get("prose_changed"))
    rationale = (payload.get("rationale") or "").strip()

    if new_status not in _STATUSES:
        result.error = f"llm: invalid status {new_status!r}"
        return result
    if new_type not in _TYPES:
        result.error = f"llm: invalid type {new_type!r}"
        return result

    status_changed = new_status != result.old_status
    type_changed = new_type != result.old_type

    result.new_status = new_status
    result.new_type = new_type
    result.prose_updated = prose_changed
    result.rationale = rationale

    if dry_run:
        parts = [f"[plan] {entity.id}:"]
        if status_changed:
            parts.append(f"status {result.old_status} → {new_status}")
        else:
            parts.append(f"status unchanged ({result.old_status})")
        if type_changed:
            parts.append(f"type {result.old_type} → {new_type}")
        else:
            parts.append(f"type unchanged ({result.old_type})")
        parts.append("prose rewrite" if prose_changed else "prose unchanged")
        if rationale:
            parts.append(f"rationale={rationale!r}")
        print(" | ".join(parts))
        return result

    # ---- apply changes ------------------------------------------------
    new_frontmatter = dict(entity.frontmatter)
    new_frontmatter["last_evaluated"] = today
    if status_changed:
        new_frontmatter["status"] = new_status
    if type_changed:
        new_frontmatter["type"] = new_type

    new_body = entity.body
    if prose_changed:
        section_map = parse(entity.body)
        old_what = section_map.get("What it is", "")
        old_why = section_map.get("Why it matters", "")
        old_curr = section_map.get("Current assessment", "")
        new_body = render(
            what_it_is=payload.get("new_what_it_is") or old_what,
            why_it_matters=payload.get("new_why_it_matters") or old_why,
            current_assessment=payload.get("new_current_assessment") or old_curr,
            history_bullets=_extract_history_bullets(entity.body),
        )

    if status_changed or type_changed or prose_changed or new_frontmatter != entity.frontmatter:
        vault.write_entity(Entity(id=entity.id, frontmatter=new_frontmatter, body=new_body))

    if status_changed or type_changed or prose_changed:
        parts: list[str] = []
        if status_changed:
            parts.append(f"Status: {result.old_status} → {new_status}")
        if type_changed:
            parts.append(f"Type: {result.old_type} → {new_type}")
        if prose_changed:
            parts.append("prose rewritten")
        summary = ". ".join(parts)
        suffix = f" {rationale}" if rationale else ""
        vault.append_history(entity.id, f"Re-evaluated. {summary}.{suffix}".strip(), entry_date=today)

    if prose_changed and index is not None:
        refreshed = vault.read_entity(entity.id) or Entity(
            id=entity.id, frontmatter=new_frontmatter, body=new_body
        )
        try:
            index.upsert(entity.id, vault.body_for_embedding(refreshed.body))
        except Exception as exc:
            log.error("%s: embedding upsert failed: %s", entity.id, exc)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def reevaluate(
    only: str | None = typer.Option(None, "--only", metavar="ID", help="Single entity ID."),
    status: Annotated[list[str] | None, typer.Option("--status", help="Filter by status (repeatable).")] = None,
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan without writing."),
) -> None:
    """Re-evaluate entities using full signal history."""
    cfg = config_module.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)
    today = date.today().isoformat()

    if only:
        entity = vault.read_entity(only)
        if entity is None:
            print(f"ERROR: entity {only!r} not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        entities = [entity]
    elif status:
        entities = []
        for s in status:
            entities.extend(vault.list_entities(status=s))
    else:
        entities = vault.list_entities()

    if not entities:
        print("No entities to process.")
        return

    try:
        context_md = cfg.context_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        context_md = ""

    router = Router(cfg)
    if not router.available_providers("reevaluation"):
        print("ERROR: no LLM providers available for task=reevaluation.", file=sys.stderr)
        raise typer.Exit(code=1)

    index: Index | None = None
    if not dry_run:
        try:
            index = Index(cfg)
        except Exception as exc:
            log.error("could not load embedding index: %s", exc)

    moved = changed_type = prose_updated = errors = 0
    for entity in entities:
        try:
            result = _reevaluate_entity(
                entity,
                cfg=cfg,
                context_md=context_md,
                router=router,
                dry_run=dry_run,
                index=index,
                today=today,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            log.exception("%s: unhandled exception", entity.id)
            result = _Result(
                id=entity.id,
                old_status=entity.frontmatter.get("status", "?"),
                old_type=entity.frontmatter.get("type", "?"),
                error=f"{type(exc).__name__}: {exc}",
            )

        if result.error:
            print(f"ERROR {entity.id}: {result.error}", file=sys.stderr)
            errors += 1
        elif not dry_run:
            status_moved = result.new_status and result.new_status != result.old_status
            type_moved = result.new_type and result.new_type != result.old_type
            if status_moved:
                moved += 1
                print(f"{entity.id}: {result.old_status} → {result.new_status} ({result.old_type} → {result.new_type})")
            elif type_moved:
                changed_type += 1
                print(f"{entity.id}: type {result.old_type} → {result.new_type}")
            if result.prose_updated:
                prose_updated += 1

    prefix = "[dry-run] " if dry_run else ""
    print(
        f"\n{prefix}Done: {moved} status moved, {changed_type} type corrected, "
        f"{prose_updated} prose rewritten, {errors} errors."
    )
    if errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
