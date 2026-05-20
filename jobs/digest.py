"""Weekly digest job — assemble a five-minute-read of the week and write it
into the vault as `digests/YYYY-MM-DD.md`.

Pulls four streams from the vault:

  1. Enrich output (logs/last_enrich.json) — what moved this week.
  2. New concepts — frontmatter.first_seen within the last 7 days.
  3. Current watch-list concepts with their artifact evaluations.
  4. Current invest-status concepts as invest reminders.
  5. Concepts flagged review_needed.

One LLM call total (task="digest").
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
from lib.body import parse as parse_body
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

    art_moved = enrich.get("artifact_evaluations_changed") or enrich.get("moved", []) or []
    concept_moved = enrich.get("concept_status_changed") or []
    sig_changes = enrich.get("significant_signal_changes", []) or []
    errors = enrich.get("errors", 0)

    lines: list[str] = [
        f"## Enrichment summary (for the week ending {today.isoformat()})",
    ]
    if concept_moved:
        lines.append(f"Concept status changes ({len(concept_moved)}):")
        for item in concept_moved:
            frm = item.get("from", "?")
            to = item.get("to", "?")
            rationale = (item.get("rationale") or "").strip()[:120]
            lines.append(f"  - {item.get('id','?')}: {frm} -> {to}. {rationale}")
    if art_moved:
        lines.append(f"Artifact evaluation changes ({len(art_moved)}):")
        for item in art_moved:
            frm = item.get("from", "?")
            to = item.get("to", "?")
            rationale = (item.get("rationale") or "").strip()[:120]
            lines.append(f"  - {item.get('id','?')}: {frm} -> {to}. {rationale}")
    if sig_changes:
        lines.append(f"Notable signal changes: {len(sig_changes)}")
        for item in sig_changes:
            d = item.get("deltas") or {}
            sd = d.get("stars_delta")
            lines.append(f"  - {item.get('id','?')}: Δstars={sd:+d}" if sd is not None else f"  - {item.get('id','?')}")
    err_count = len(errors) if isinstance(errors, list) else int(errors or 0)
    if err_count:
        lines.append(f"Errors during enrich: {err_count}")
    return "\n".join(lines) + "\n"


def _format_new_concepts_section(concepts: list[vault.Concept]) -> str:
    lines = ["## New concepts this week"]
    if not concepts:
        lines.append("  (none)")
        return "\n".join(lines) + "\n"
    for c in concepts:
        fm = c.frontmatter
        gist = _section_summary(c.body, "What it is")
        n_arts = len(fm.get("artifacts") or [])
        lines.append(
            f"  - {c.id} ({fm.get('status','?')}, relevance={fm.get('relevance','?')}, "
            f"{n_arts} artifact(s)): {gist}"
        )
    return "\n".join(lines) + "\n"


def _format_watch_section(watch_concepts: list[vault.Concept]) -> str:
    lines = ["## Watch-list pulse"]
    if not watch_concepts:
        lines.append("  (none)")
        return "\n".join(lines) + "\n"
    for c in watch_concepts:
        fm = c.frontmatter
        arts = fm.get("artifacts") or []
        eval_counts: dict[str, int] = {}
        for entry in arts:
            art_id = entry.get("id") if isinstance(entry, dict) else str(entry)
            art = vault.read_artifact(art_id)
            if art:
                ev = art.frontmatter.get("evaluation", "new")
                eval_counts[ev] = eval_counts.get(ev, 0) + 1
        eval_str = ", ".join(f"{k}={v}" for k, v in sorted(eval_counts.items()))
        last_eval = fm.get("last_evaluated", "?")
        lines.append(
            f"  - {c.id}: [{eval_str or 'no artifacts'}]  (last_evaluated={last_eval})"
        )
    return "\n".join(lines) + "\n"


def _format_invest_section(invest_concepts: list[vault.Concept], today: date) -> str:
    lines = ["## Invest reminders"]
    if not invest_concepts:
        lines.append("  (none)")
        return "\n".join(lines) + "\n"
    for c in invest_concepts:
        fm = c.frontmatter
        last_eval_raw = fm.get("last_evaluated", "")
        last_eval_d = _parse_iso_date(last_eval_raw)
        days = (today - last_eval_d).days if last_eval_d else None
        since = f"{last_eval_raw} ({days}d ago)" if days is not None else "unknown"
        assessment = _section_summary(c.body, "Current assessment")
        lines.append(f"  - {c.id}: at invest since {since}.  ({assessment})")
    return "\n".join(lines) + "\n"


def _format_review_section(review_concepts: list[vault.Concept]) -> str:
    lines = ["## Needs your review"]
    if not review_concepts:
        return ""
    for c in review_concepts:
        fm = c.frontmatter
        lines.append(f"  - {c.id} ({fm.get('label','')}): review_needed flag set")
    return "\n".join(lines) + "\n"


_PROMPT_INSTRUCTION = (
    "Write a weekly digest of this AI tooling intelligence. Be ruthlessly concise — "
    "readable in five minutes. Use this structure: "
    "## New This Week / ## Moved / ## Watch-List Pulse / ## Invest Reminders / ## Needs Your Review. "
    "Each entry: one or two lines. Skip sections that have no content. "
    "Markdown formatting. Do not invent facts not in the input."
)


def assemble_prompt(
    *,
    context_text: str,
    enrich: dict[str, Any],
    new_concepts: list[vault.Concept],
    watch_concepts: list[vault.Concept],
    invest_concepts: list[vault.Concept],
    review_concepts: list[vault.Concept],
    today: date,
) -> str:
    """Build the single prompt string handed to the LLM."""
    review_block = _format_review_section(review_concepts)
    parts = [
        "## My context",
        context_text.strip() or "(no context provided)",
        "",
        _format_enrich_section(enrich, today).rstrip(),
        "",
        _format_new_concepts_section(new_concepts).rstrip(),
        "",
        _format_watch_section(watch_concepts).rstrip(),
        "",
        _format_invest_section(invest_concepts, today).rstrip(),
    ]
    if review_block:
        parts += ["", review_block.rstrip()]
    parts += ["", _PROMPT_INSTRUCTION]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def _collect_new_concepts(today: date, window_days: int = 7) -> list[vault.Concept]:
    cutoff = today - timedelta(days=window_days)
    out: list[vault.Concept] = []
    for concept in vault.list_concepts():
        first_seen = _parse_iso_date(concept.frontmatter.get("first_seen"))
        if first_seen is None:
            continue
        if first_seen >= cutoff:
            out.append(concept)
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
    new_concepts = _collect_new_concepts(target_date)
    watch_concepts = vault.list_concepts(status="watch")
    invest_concepts = vault.list_concepts(status="invest")
    review_concepts = [c for c in vault.list_concepts() if c.frontmatter.get("review_needed")]

    concept_moved = len(enrich.get("concept_status_changed") or [])
    art_moved = len(enrich.get("artifact_evaluations_changed") or enrich.get("moved", []) or [])

    logger.info(
        "digest inputs: new_concepts=%d watch=%d invest=%d review=%d enrich_concept_moved=%d enrich_art_moved=%d",
        len(new_concepts), len(watch_concepts), len(invest_concepts), len(review_concepts),
        concept_moved, art_moved,
    )

    prompt = assemble_prompt(
        context_text=context_text,
        enrich=enrich,
        new_concepts=new_concepts,
        watch_concepts=watch_concepts,
        invest_concepts=invest_concepts,
        review_concepts=review_concepts,
        today=target_date,
    )

    router = Router(cfg)
    available = router.available_providers("digest")
    if not available:
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

    coverage = (
        f"(covering {len(new_concepts)} new concepts, {concept_moved} concept moves, "
        f"{art_moved} artifact eval changes, "
        f"{len(watch_concepts)} watched, {len(invest_concepts)} to invest)"
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
