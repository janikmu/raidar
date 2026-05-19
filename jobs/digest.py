"""Weekly digest job — assemble a five-minute-read of the week and write it
into the vault as `digests/YYYY-MM-DD.md`.

Pulls four streams from the vault:

  1. Enrich output (logs/last_enrich.json) — what moved this week.
  2. New entities — frontmatter.first_seen within the last 7 days.
  3. Current watch list — latest signal snapshot for each.
  4. Current adopt list — how long each has been at status=adopt.

Hands a compact, structured prompt to the LLM router (task="digest") and
writes the returned markdown via `vault.write_digest`. One LLM call total.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import typer

from lib import config, logging_setup, vault
from lib.entity_body import parse as parse_body
from lib.llm import AllProvidersFailed, Router

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help="Generate the weekly AI Radar digest.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_sentence(text: str) -> str:
    """Return the first sentence (or first line) of `text`, stripped.

    Cheap heuristic: split on `. `, `? `, `! `, or newline; cap length.
    Returns `""` for empty input.
    """
    if not text:
        return ""
    cleaned = text.strip()
    if not cleaned:
        return ""
    # Prefer the first paragraph line, then the first sentence in it.
    first_para = cleaned.split("\n\n", 1)[0].replace("\n", " ").strip()
    m = re.search(r"(.+?[.!?])(\s|$)", first_para)
    sentence = m.group(1).strip() if m else first_para
    # Trim absurdly long sentences so we don't blow the prompt budget.
    if len(sentence) > 280:
        sentence = sentence[:277].rstrip() + "..."
    return sentence


def _section_summary(body: str, section_title: str) -> str:
    """Return the first sentence of a named entity body section, or "" if missing."""
    sections = parse_body(body)
    return _first_sentence(sections.get(section_title, ""))


def _load_enrich_output(path: Path) -> dict[str, Any]:
    """Load logs/last_enrich.json. Returns an empty dict if the file is missing
    or malformed (with a logged warning) — the digest still proceeds."""
    if not path.is_file():
        logger.warning("enrich output not found at %s; proceeding without it", path)
        return {}
    try:
        with path.open(encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError as exc:
        logger.warning("enrich output at %s is not valid JSON (%s); ignoring", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("enrich output at %s is not a JSON object; ignoring", path)
        return {}
    return data


def _parse_iso_date(value: Any) -> date | None:
    """Parse a `YYYY-MM-DD` frontmatter string. Returns None on any failure."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _latest_signal(entity_id: str) -> dict[str, Any] | None:
    """Return the most recent signal snapshot for an entity, or None."""
    signals = vault.read_signals(entity_id)
    if not signals:
        return None
    return signals[-1]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _format_enrich_section(enrich: dict[str, Any], today: date) -> str:
    if not enrich:
        return (
            f"## Enrichment summary (for the week ending {today.isoformat()})\n"
            "No enrich run output available.\n"
        )

    evaluated = enrich.get("evaluated", 0)
    moved = enrich.get("moved", []) or []
    sig_changes = enrich.get("significant_signal_changes", []) or []
    errors = enrich.get("errors", 0)

    lines: list[str] = [
        f"## Enrichment summary (for the week ending {today.isoformat()})",
        f"Evaluated: {evaluated}",
        f"Moved status: {len(moved)}",
    ]
    for item in moved:
        eid = item.get("id", "?")
        frm = item.get("from", "?")
        to = item.get("to", "?")
        rationale = (item.get("rationale") or "").strip()
        lines.append(f"  - {eid}: {frm} -> {to}. Rationale: {rationale}")

    lines.append(f"Significant signal changes (no status flip): {len(sig_changes)}")
    for item in sig_changes:
        eid = item.get("id", "?")
        sd = item.get("stars_delta", 0)
        cd = item.get("commits_30d_delta", 0)
        lines.append(f"  - {eid}: stars_delta={sd}, commits_30d_delta={cd}")

    # `errors` may be a count or a list — accept either.
    err_count = len(errors) if isinstance(errors, list) else int(errors or 0)
    lines.append(f"Errors during enrich: {err_count}")
    return "\n".join(lines) + "\n"


def _format_new_section(new_entities: list[vault.Entity]) -> str:
    lines = ["## New this week"]
    if not new_entities:
        lines.append("  (none)")
        return "\n".join(lines) + "\n"
    for e in new_entities:
        type_ = e.frontmatter.get("type", "?")
        status = e.frontmatter.get("status", "?")
        relevance = e.frontmatter.get("relevance", "?")
        gist = _section_summary(e.body, "What it is")
        lines.append(
            f"  - {e.id} ({type_}, {status}, relevance={relevance}): {gist}"
        )
    return "\n".join(lines) + "\n"


def _format_watch_section(watch_entities: list[vault.Entity]) -> str:
    lines = ["## Current watch list"]
    if not watch_entities:
        lines.append("  (none)")
        return "\n".join(lines) + "\n"
    for e in watch_entities:
        snap = _latest_signal(e.id) or {}
        stars = snap.get("stars", "?")
        commits = snap.get("commits_30d", "?")
        last_eval = e.frontmatter.get("last_evaluated", "?")
        lines.append(
            f"  - {e.id}: stars={stars}, commits_30d={commits}  (last_evaluated={last_eval})"
        )
    return "\n".join(lines) + "\n"


def _format_adopt_section(adopt_entities: list[vault.Entity], today: date) -> str:
    lines = ["## Adopt reminders"]
    if not adopt_entities:
        lines.append("  (none)")
        return "\n".join(lines) + "\n"
    for e in adopt_entities:
        last_eval_raw = e.frontmatter.get("last_evaluated", "")
        last_eval_d = _parse_iso_date(last_eval_raw)
        if last_eval_d is not None:
            days = (today - last_eval_d).days
            since = f"{last_eval_raw} ({days}d ago)"
        else:
            since = "unknown"
        assessment = _section_summary(e.body, "Current assessment")
        lines.append(f"  - {e.id}: at adopt since {since}.  ({assessment})")
    return "\n".join(lines) + "\n"


_PROMPT_INSTRUCTION = (
    "Write a weekly digest of this AI tooling intelligence. Be ruthlessly concise — "
    "readable in five minutes. Use this structure: "
    "## New this week / ## Moved / ## Watch-list pulse / ## Adopt reminder. "
    "Each entry one or two lines. Skip sections that have no content. "
    "Markdown formatting. Do not invent facts not in the input."
)


def assemble_prompt(
    *,
    context_text: str,
    enrich: dict[str, Any],
    new_entities: list[vault.Entity],
    watch_entities: list[vault.Entity],
    adopt_entities: list[vault.Entity],
    today: date,
) -> str:
    """Build the single prompt string handed to the LLM."""
    parts = [
        "## My context",
        context_text.strip() or "(no context provided)",
        "",
        _format_enrich_section(enrich, today).rstrip(),
        "",
        _format_new_section(new_entities).rstrip(),
        "",
        _format_watch_section(watch_entities).rstrip(),
        "",
        _format_adopt_section(adopt_entities, today).rstrip(),
        "",
        _PROMPT_INSTRUCTION,
    ]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def _collect_new_entities(today: date, window_days: int = 7) -> list[vault.Entity]:
    cutoff = today - timedelta(days=window_days)
    out: list[vault.Entity] = []
    for entity in vault.list_entities():
        first_seen = _parse_iso_date(entity.frontmatter.get("first_seen"))
        if first_seen is None:
            continue
        if first_seen >= cutoff:
            out.append(entity)
    return out


def _load_context(path: Path) -> str:
    if not path.is_file():
        logger.warning("context file %s not found; using empty context", path)
        return ""
    return path.read_text(encoding="utf-8")


def run_digest(*, target_date: date, dry_run: bool) -> None:
    cfg = config.load()
    logging_setup.setup(level=cfg.log_level, log_file=cfg.log_file)

    logger.info("digest: target_date=%s dry_run=%s", target_date.isoformat(), dry_run)

    context_text = _load_context(cfg.context_path)
    enrich = _load_enrich_output(cfg.enrich_output_path)
    new_entities = _collect_new_entities(target_date)
    watch_entities = vault.list_entities(status="watch")
    adopt_entities = vault.list_entities(status="adopt")

    logger.info(
        "digest inputs: new=%d watch=%d adopt=%d enrich_moved=%d",
        len(new_entities),
        len(watch_entities),
        len(adopt_entities),
        len(enrich.get("moved", []) or []),
    )

    prompt = assemble_prompt(
        context_text=context_text,
        enrich=enrich,
        new_entities=new_entities,
        watch_entities=watch_entities,
        adopt_entities=adopt_entities,
        today=target_date,
    )

    router = Router(cfg)
    available = router.available_providers("digest")
    if not available:
        # Acceptance: prove input assembly works even with no LLM configured.
        logger.warning(
            "no providers configured for task=digest; printing assembled prompt only"
        )
        print(prompt)
        return

    try:
        completion = router.generate(
            task="digest",
            prompt=prompt,
            system="You are a terse weekly-digest writer for an AI tooling watchlist.",
            max_tokens=3000,
        )
    except AllProvidersFailed as exc:
        logger.error("digest: all providers failed: %s (cause: %s)", exc, exc.__cause__)
        raise typer.Exit(code=1) from exc

    body = completion.text

    moved_count = len(enrich.get("moved", []) or [])
    coverage = (
        f"(covering {len(new_entities)} new, {moved_count} moved, "
        f"{len(watch_entities)} watched, {len(adopt_entities)} to adopt)"
    )

    if dry_run:
        logger.info("dry-run: skipping vault.write_digest")
        print(body)
        print(coverage)
        return

    path = vault.write_digest(date_iso=target_date.isoformat(), body_markdown=body)
    print(f"Digest written: {path}")
    print(coverage)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    date_str: str | None = typer.Option(
        None,
        "--date",
        help="Target date in YYYY-MM-DD form (defaults to today).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the digest to stdout instead of writing it to the vault.",
    ),
) -> None:
    """Generate the weekly AI Radar digest."""
    if date_str is None:
        target = date.today()
    else:
        try:
            target = datetime.fromisoformat(date_str).date()
        except ValueError as exc:
            raise typer.BadParameter(f"--date must be YYYY-MM-DD ({exc})") from exc
    run_digest(target_date=target, dry_run=dry_run)


if __name__ == "__main__":
    app()
