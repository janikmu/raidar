"""Force re-evaluation of artifacts using their full signal history.

Unlike the weekly enrich job (which only touches artifacts whose concepts
are in emerging/watch and uses a short signal tail), reevaluate:

  - Processes ALL artifacts regardless of current evaluation (or a filtered set).
  - Feeds the LLM the *full* signal history so backfill context is used.
  - Applies the same evaluation rules as capture but with ground-truth data.
  - Can also correct coarse artifact types (e.g. 'repo' → 'spec').

Useful after a bulk backfill pass when many established repos are incorrectly
sitting at 'new' or 'promising'.

Usage:
    raidar reevaluate                          # all artifacts
    raidar reevaluate --only ID                # single artifact
    raidar reevaluate --evaluation new         # only those at 'new'
    raidar reevaluate --dry-run                # print plan, no writes
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any

import typer

from lib import config as config_module
from lib import vault
from lib.embeddings import Index
from lib.body import parse, render_artifact
from lib.llm import AllProvidersFailed, Router
from lib.logging_setup import setup as setup_logging
from lib.vault import Artifact

log = logging.getLogger("jobs.reevaluate")

app = typer.Typer(add_completion=False, help=__doc__)

# ---------------------------------------------------------------------------
# Schema / prompts
# ---------------------------------------------------------------------------

_EVALUATIONS = ["new", "promising", "recommended", "deprecated", "hype"]
_TYPES = ["repo", "paper", "post", "release", "spec"]

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["new_evaluation", "new_type", "prose_changed", "rationale"],
    "properties": {
        "new_evaluation": {"type": "string", "enum": _EVALUATIONS},
        "new_type": {"type": "string", "enum": _TYPES},
        "prose_changed": {"type": "boolean"},
        "new_what_it_is": {"type": ["string", "null"]},
        "new_evaluation_rationale": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
}

_SYSTEM_PROMPT = (
    "You are re-evaluating artifacts in an Information Systems researcher's "
    "personal AI-tooling radar. You have access to the artifact's full star "
    "history (backfill signals) plus any live enrich signals. Use this data "
    "as ground truth — do not guess from prose alone.\n\n"
    "Evaluation rules (apply strictly):\n"
    "  'recommended' = mature, actively maintained, strong community or org "
    "backing — typically > 2 years old AND > 5k stars, or in broad production use.\n"
    "  'promising' = clear value and growing traction, not yet proven at scale — "
    "typically 6 months–2 years old or 300–5k stars.\n"
    "  'new' = recent, < 6 months old or < 300 stars, insufficient signal.\n"
    "  'deprecated' = was relevant, now abandoned, archived, or superseded.\n"
    "  'hype' = superficial signals, self-reported benchmarks, thin community, "
    "not safe to depend on.\n\n"
    "A well-established repo with years of history and thousands of stars "
    "must NOT stay 'new' or 'promising'. "
    "Also correct the type if the current one is wrong — pick the most "
    "specific match from the enum. "
    "Rewrite prose only when the evaluation changes or the current prose is "
    "clearly stale/wrong; otherwise set prose_changed=false. "
    "Keep prose terse — one short paragraph per section. "
    "Return JSON matching the provided schema."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_history_bullets(body: str) -> list[str]:
    sections = parse(body)
    hist_block = sections.get("History", "")
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


def _build_prompt(context_md: str, artifact: Artifact, signals: list[dict[str, Any]]) -> str:
    fm_dump = json.dumps(artifact.frontmatter, indent=2, sort_keys=True, default=str)
    signals_dump = json.dumps(signals, indent=2, default=str)
    return (
        "# Researcher context\n"
        f"{context_md.strip()}\n\n"
        "# Artifact under review\n"
        f"## frontmatter\n```json\n{fm_dump}\n```\n\n"
        f"## body\n{artifact.body.strip()}\n\n"
        "# Full signal history (oldest first)\n"
        f"```json\n{signals_dump}\n```\n\n"
        "# Task\n"
        "Re-evaluate this artifact. Correct the evaluation and type based on "
        "the signal history above (not from the current frontmatter). "
        f"Evaluation options: {', '.join(_EVALUATIONS)}. "
        f"Type options: {', '.join(_TYPES)}. "
        "Return JSON per the schema. "
        "If prose_changed is false, you may leave new_what_it_is / "
        "new_evaluation_rationale as null."
    )


# ---------------------------------------------------------------------------
# Per-artifact logic
# ---------------------------------------------------------------------------


@dataclass
class _Result:
    id: str
    old_evaluation: str
    old_type: str
    new_evaluation: str | None = None
    new_type: str | None = None
    prose_updated: bool = False
    rationale: str = ""
    error: str | None = None


def _reevaluate_artifact(
    artifact: Artifact,
    *,
    cfg: config_module.Config,
    context_md: str,
    router: Router,
    dry_run: bool,
    index: Index | None,
    today: str,
) -> _Result:
    result = _Result(
        id=artifact.id,
        old_evaluation=artifact.frontmatter.get("evaluation", "new"),
        old_type=artifact.frontmatter.get("type", "repo"),
    )

    signals = vault.read_signals(artifact.id)
    prompt = _build_prompt(context_md, artifact, signals)

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

    new_evaluation = payload.get("new_evaluation") or result.old_evaluation
    new_type = payload.get("new_type") or result.old_type
    prose_changed = bool(payload.get("prose_changed"))
    rationale = (payload.get("rationale") or "").strip()

    if new_evaluation not in _EVALUATIONS:
        result.error = f"llm: invalid evaluation {new_evaluation!r}"
        return result
    if new_type not in _TYPES:
        result.error = f"llm: invalid type {new_type!r}"
        return result

    evaluation_changed = new_evaluation != result.old_evaluation
    type_changed = new_type != result.old_type

    result.new_evaluation = new_evaluation
    result.new_type = new_type
    result.prose_updated = prose_changed
    result.rationale = rationale

    if dry_run:
        parts = [f"[plan] {artifact.id}:"]
        if evaluation_changed:
            parts.append(f"evaluation {result.old_evaluation} → {new_evaluation}")
        else:
            parts.append(f"evaluation unchanged ({result.old_evaluation})")
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
    new_frontmatter = dict(artifact.frontmatter)
    new_frontmatter["last_evaluated"] = today
    if evaluation_changed:
        new_frontmatter["evaluation"] = new_evaluation
    if type_changed:
        new_frontmatter["type"] = new_type

    sections = parse(artifact.body)
    history_bullets = _extract_history_bullets(artifact.body)

    summary_parts: list[str] = []
    if evaluation_changed:
        summary_parts.append(f"Evaluation: {result.old_evaluation} → {new_evaluation}")
    if type_changed:
        summary_parts.append(f"Type: {result.old_type} → {new_type}")
    if prose_changed:
        summary_parts.append("prose rewritten")
    if summary_parts:
        summary = ". ".join(summary_parts)
        suffix = f" {rationale}" if rationale else ""
        history_bullets.append(f"{today}: Re-evaluated. {summary}.{suffix}".strip())

    new_body = render_artifact(
        what_it_is=(payload.get("new_what_it_is") if prose_changed else None)
                   or sections.get("What it is", ""),
        evaluation_rationale=(payload.get("new_evaluation_rationale") if prose_changed else None)
                              or sections.get("Evaluation rationale", ""),
        history_bullets=history_bullets,
    )

    if evaluation_changed or type_changed or prose_changed or new_frontmatter != artifact.frontmatter:
        vault.write_artifact(Artifact(id=artifact.id, frontmatter=new_frontmatter, body=new_body))

    if prose_changed and index is not None:
        refreshed = vault.read_artifact(artifact.id) or Artifact(
            id=artifact.id, frontmatter=new_frontmatter, body=new_body
        )
        try:
            index.upsert(artifact.id, vault.body_for_embedding(refreshed.body))
        except Exception as exc:
            log.error("%s: embedding upsert failed: %s", artifact.id, exc)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def reevaluate(
    only: str | None = typer.Option(None, "--only", metavar="ID", help="Single artifact ID."),
    evaluation: Annotated[list[str] | None, typer.Option("--evaluation", help="Filter by evaluation (repeatable).")] = None,
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan without writing."),
) -> None:
    """Re-evaluate artifacts using full signal history."""
    cfg = config_module.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)
    today = date.today().isoformat()

    if only:
        artifact = vault.read_artifact(only)
        if artifact is None:
            print(f"ERROR: artifact {only!r} not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        artifacts = [artifact]
    elif evaluation:
        artifacts = []
        for e in evaluation:
            artifacts.extend(vault.list_artifacts(evaluation=e))
    else:
        artifacts = vault.list_artifacts()

    if not artifacts:
        print("No artifacts to process.")
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
            index = Index(cfg, layer="artifacts")
        except Exception as exc:
            log.error("could not load artifact embedding index: %s", exc)

    moved = changed_type = prose_updated = errors = 0
    for artifact in artifacts:
        try:
            result = _reevaluate_artifact(
                artifact,
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
            log.exception("%s: unhandled exception", artifact.id)
            result = _Result(
                id=artifact.id,
                old_evaluation=artifact.frontmatter.get("evaluation", "?"),
                old_type=artifact.frontmatter.get("type", "?"),
                error=f"{type(exc).__name__}: {exc}",
            )

        if result.error:
            print(f"ERROR {artifact.id}: {result.error}", file=sys.stderr)
            errors += 1
        elif not dry_run:
            eval_moved = result.new_evaluation and result.new_evaluation != result.old_evaluation
            type_moved = result.new_type and result.new_type != result.old_type
            if eval_moved:
                moved += 1
                print(f"{artifact.id}: {result.old_evaluation} → {result.new_evaluation} ({result.old_type} → {result.new_type})")
            elif type_moved:
                changed_type += 1
                print(f"{artifact.id}: type {result.old_type} → {result.new_type}")
            if result.prose_updated:
                prose_updated += 1

    prefix = "[dry-run] " if dry_run else ""
    print(
        f"\n{prefix}Done: {moved} evaluation moved, {changed_type} type corrected, "
        f"{prose_updated} prose rewritten, {errors} errors."
    )
    if errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
