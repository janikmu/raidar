"""Weekly enrich job for AI Radar.

For every entity in `status in {emerging, watch}`:

  1. Pull a fresh GitHub snapshot (if a `github_repo` is set) and append it
     to `signals/{id}.jsonl`.
  2. Compute deltas vs. the previous snapshot.
  3. Ask the LLM ("enrichment" task chain) whether the status should change
     and/or the prose should be rewritten, given full context + signal history.
  4. Apply changes — frontmatter mutation, body re-render, history bullet,
     embedding refresh.
  5. Write a summary JSON for the digest job and print a terse stdout summary.

A failure on a single entity is captured and the batch continues — one
unreachable repo or one rate-limited LLM should not poison the rest. We
deliberately do NOT touch `status in {adopt, skip, settled}` here: `adopt`
needs human signal, the other two are done.

Run modes:

    raidar enrich               # full pass
    raidar enrich --only ID     # single entity (testing)
    raidar enrich --dry-run     # no writes, plan only

`--dry-run` performs every read step (fetch, delta compute, LLM eval) but
skips the four mutating sinks: `append_signal`, `write_entity`,
`append_history`, `Index.upsert`, and the enrich-output JSON. When the LLM
isn't configured (offline dev), `--dry-run` still exercises fetch + delta
compute and prints a clear "LLM not configured" line.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

from lib import config as config_module
from lib import vault
from lib.embeddings import Index
from lib.entity_body import HISTORY, parse, render
from lib.github import (
    RepoSnapshot, TerminalError, fetch_repo, fetch_star_history,
    get_rate_limit_remaining, parse_repo_url,
)
from lib.llm import AllProvidersFailed, Router
from lib.logging_setup import setup as setup_logging
from lib.vault import Entity

log = logging.getLogger("jobs.enrich")


# ---------------------------------------------------------------------------
# Schema for the LLM response.
# ---------------------------------------------------------------------------

_STATUSES = ["emerging", "watch", "adopt", "skip", "settled"]

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status_changed", "new_status", "prose_changed", "rationale"],
    "properties": {
        "status_changed": {"type": "boolean"},
        "new_status": {"type": "string", "enum": _STATUSES},
        "prose_changed": {"type": "boolean"},
        "new_what_it_is": {"type": ["string", "null"]},
        "new_why_it_matters": {"type": ["string", "null"]},
        "new_current_assessment": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
}

_SYSTEM_PROMPT = (
    "You are the curator of an Information Systems researcher's personal "
    "AI-tooling watchlist. You are reviewing one entity given the user's "
    "context, current entity prose, full signal history, and recent deltas. "
    "Use the star history (backfill signals) as ground truth for age and "
    "growth trajectory — a repo with years of history and thousands of stars "
    "should NOT stay 'emerging'. "
    "Status heuristics: 'emerging' for < 6 months old or < 300 stars; "
    "'watch' for 6 months–2 years old or 300–5k stars; "
    "'adopt' for > 2 years old AND > 5k stars, or tools in broad production use. "
    "Only rewrite prose when something materially changed (status flip, big "
    "signal move, repo archived, scope shift). "
    "Return JSON matching the provided schema."
)


# ---------------------------------------------------------------------------
# Result types.
# ---------------------------------------------------------------------------


@dataclass
class _Deltas:
    """Per-entity signal deltas, all optional (None on first run)."""

    stars_delta: int | None
    forks_delta: int | None
    commits_30d_delta: int | None
    open_issues_delta: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "stars_delta": self.stars_delta,
            "forks_delta": self.forks_delta,
            "commits_30d_delta": self.commits_30d_delta,
            "open_issues_delta": self.open_issues_delta,
        }


@dataclass
class _EntityResult:
    id: str
    evaluated: bool = False
    moved: dict[str, Any] | None = None        # {"from": ..., "to": ..., "rationale": ...}
    prose_updated: bool = False
    significant_change: dict[str, Any] | None = None  # {"deltas": {...}}
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _today_iso() -> str:
    return date.today().isoformat()


def _significant(
    deltas: _Deltas,
    prev: dict[str, Any] | None,
    thresholds: dict[str, Any],
) -> bool:
    """Apply config thresholds to determine if a signal change is "notable"."""
    if prev is None:
        return False
    sig_cfg = thresholds.get("signal_change", {}) if thresholds else {}
    abs_star = int(sig_cfg.get("abs_star_delta", 50))
    rel_star_pct = float(sig_cfg.get("rel_star_delta_pct", 20))
    rel_commits_pct = float(sig_cfg.get("rel_commits_30d_pct", 50))

    if deltas.stars_delta is not None and abs(deltas.stars_delta) >= abs_star:
        return True

    prev_stars = prev.get("stars")
    if (
        deltas.stars_delta is not None
        and isinstance(prev_stars, (int, float))
        and prev_stars > 0
    ):
        if (deltas.stars_delta / prev_stars * 100) >= rel_star_pct:
            return True

    prev_commits = prev.get("commits_30d")
    if (
        deltas.commits_30d_delta is not None
        and isinstance(prev_commits, (int, float))
        and prev_commits > 0
    ):
        if abs(deltas.commits_30d_delta / prev_commits * 100) >= rel_commits_pct:
            return True

    return False


def _compute_deltas(
    new_snap: dict[str, Any], prev_snap: dict[str, Any] | None
) -> _Deltas:
    if prev_snap is None:
        return _Deltas(None, None, None, None)

    def _diff(field: str) -> int | None:
        a = new_snap.get(field)
        b = prev_snap.get(field)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return None
        return int(a) - int(b)

    return _Deltas(
        stars_delta=_diff("stars"),
        forks_delta=_diff("forks"),
        commits_30d_delta=_diff("commits_30d"),
        open_issues_delta=_diff("open_issues"),
    )


def _snapshot_from_repo(
    repo: RepoSnapshot, status: str, today: str
) -> dict[str, Any]:
    return {
        "date": today,
        "stars": repo.stars,
        "forks": repo.forks,
        "commits_30d": repo.commits_30d,
        "open_issues": repo.open_issues,
        "status": status,
        "source": "enrich",
    }


def _extract_history_bullets(body: str) -> list[str]:
    """Return the History section bullets (text after '- '), preserving order."""
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


def _build_prompt(
    context_md: str,
    entity: Entity,
    signals_tail: list[dict[str, Any]],
    deltas: _Deltas,
    significant: bool,
) -> str:
    """Assemble the user-side prompt for the LLM. Stable formatting matters
    less than completeness — the LLM is reading this in one shot."""
    fm_dump = json.dumps(entity.frontmatter, indent=2, sort_keys=True, default=str)
    signals_dump = json.dumps(signals_tail, indent=2, default=str)
    deltas_dump = json.dumps(deltas.as_dict(), indent=2)

    return (
        "# Researcher context\n"
        f"{context_md.strip()}\n\n"
        "# Entity under review\n"
        f"## frontmatter\n```json\n{fm_dump}\n```\n\n"
        f"## body\n{entity.body.strip()}\n\n"
        "# Signal history (most recent last)\n"
        f"```json\n{signals_dump}\n```\n\n"
        "# Deltas (current vs. previous snapshot)\n"
        f"```json\n{deltas_dump}\n```\n"
        f"significant_change_flag: {significant}\n\n"
        "# Task\n"
        "Decide whether the entity's status should change and/or whether the "
        "prose sections (`What it is`, `Why it matters`, `Current "
        "assessment`) should be rewritten. Status options: "
        f"{', '.join(_STATUSES)}. Return JSON per the schema. "
        "If status_changed is false, set new_status to the current status. "
        "If prose_changed is false, you may leave the three new_* prose fields null. "
        "Keep prose terse — one short paragraph per section."
    )


def _format_planned_change(
    entity_id: str,
    parsed: dict[str, Any],
    deltas: _Deltas,
    significant: bool,
    current_status: str,
) -> str:
    bits = [f"[plan] {entity_id}:"]
    if parsed.get("status_changed"):
        bits.append(f"status {current_status} -> {parsed.get('new_status')!r}")
    else:
        bits.append(f"status unchanged ({current_status})")
    if parsed.get("prose_changed"):
        changed = [
            name for name, key in (
                ("what_it_is", "new_what_it_is"),
                ("why_it_matters", "new_why_it_matters"),
                ("current_assessment", "new_current_assessment"),
            ) if parsed.get(key)
        ]
        bits.append("prose rewrite: " + (", ".join(changed) if changed else "(empty)"))
    else:
        bits.append("prose unchanged")
    if significant:
        bits.append(f"significant deltas={deltas.as_dict()}")
    rationale = parsed.get("rationale", "")
    if rationale:
        bits.append(f"rationale={rationale!r}")
    return " | ".join(bits)


# ---------------------------------------------------------------------------
# Core per-entity step.
# ---------------------------------------------------------------------------


def _process_entity(
    entity: Entity,
    *,
    cfg: config_module.Config,
    context_md: str,
    router: Router | None,
    dry_run: bool,
    index: Index | None,
) -> _EntityResult:
    """Process one entity end-to-end. Caller wraps in try/except."""
    today = _today_iso()
    result = _EntityResult(id=entity.id)
    current_status = entity.frontmatter.get("status", "unknown")

    # ---- 1. GitHub fetch + signal append --------------------------------
    new_snap: dict[str, Any] | None = None
    github_repo = entity.frontmatter.get("github_repo")
    if github_repo:
        parsed_repo = parse_repo_url(str(github_repo))
        if parsed_repo is None:
            log.warning("%s: github_repo=%r not parseable; skipping fetch", entity.id, github_repo)
        else:
            owner, name = parsed_repo
            try:
                repo = fetch_repo(owner, name)
            except TerminalError as exc:
                log.error("%s: GitHub TerminalError on %s/%s: %s", entity.id, owner, name, exc)
                result.error = f"github terminal: {exc}"
                return result
            if repo is None:
                log.warning("%s: GitHub returned 404 for %s/%s", entity.id, owner, name)
                if not dry_run:
                    try:
                        vault.append_history(entity.id, "Repo unreachable on enrich (404).", entry_date=today)
                    except Exception as exc:  # pragma: no cover - defensive
                        log.error("%s: failed to append 404-history: %s", entity.id, exc)
                # Continue without a new snapshot. Don't treat as fatal.
            else:
                new_snap = _snapshot_from_repo(repo, current_status, today)
                # ---- 1.5. Auto-backfill star history on first enrich --------
                # Skip if already backfilled; only proceed when rate limit is healthy.
                if not vault.has_star_history_backfill(entity.id):
                    rl = get_rate_limit_remaining()
                    if rl is None or rl > 300:
                        try:
                            history = fetch_star_history(owner, name, repo.stars)
                            if history and not dry_run:
                                for star_count, starred_at in history:
                                    vault.append_signal(entity.id, {
                                        "date": starred_at[:10],
                                        "stars": star_count,
                                        "source": "backfill",
                                    })
                                log.info(
                                    "%s: backfilled %d star history points",
                                    entity.id, len(history),
                                )
                        except Exception as exc:
                            log.warning("%s: star history backfill failed: %s", entity.id, exc)
                    else:
                        log.info(
                            "%s: skipping backfill (rate_limit_remaining=%d)", entity.id, rl
                        )

    # ---- 2. Persist new snapshot ----------------------------------------
    if new_snap is not None and not dry_run:
        vault.append_signal(entity.id, new_snap)

    # ---- 3. Compute deltas against previous snapshot --------------------
    # `read_signals` reflects what's currently on disk. After append_signal
    # the just-written snapshot is the last entry; in --dry-run it's not on
    # disk so we splice it in for delta math.
    history_signals = vault.read_signals(entity.id)
    if dry_run and new_snap is not None:
        history_signals = history_signals + [new_snap]

    prev_snap: dict[str, Any] | None = None
    if new_snap is not None and len(history_signals) >= 2:
        prev_snap = history_signals[-2]
    deltas = _compute_deltas(new_snap or {}, prev_snap) if new_snap is not None else _Deltas(None, None, None, None)
    significant = _significant(deltas, prev_snap, cfg.thresholds)
    if significant:
        result.significant_change = {"deltas": deltas.as_dict()}

    # ---- 4. LLM evaluation ----------------------------------------------
    if router is None:
        log.info("%s: LLM not configured; skipping evaluation step", entity.id)
        print(f"[dry-run] {entity.id}: LLM not configured; skipping evaluation step")
        # Even in dry-run we want to surface that fetch + deltas ran.
        if new_snap is not None:
            print(f"[dry-run] {entity.id}: snapshot={new_snap}, deltas={deltas.as_dict()}, significant={significant}")
        return result

    # Adaptive tail: use full history for the first enrich pass per entity
    # (backfill signals give crucial age/growth context); short tail for
    # subsequent weekly updates where we only need recent momentum.
    prior_enrich = [s for s in history_signals if s.get("source") == "enrich"]
    signals_tail = history_signals if not prior_enrich else history_signals[-8:]
    prompt = _build_prompt(context_md, entity, signals_tail, deltas, significant)

    try:
        completion = router.generate(
            task="enrichment",
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            response_schema=_RESPONSE_SCHEMA,
            max_tokens=2048,
        )
    except AllProvidersFailed as exc:
        log.error("%s: LLM AllProvidersFailed: %s", entity.id, exc)
        result.error = f"llm: {exc}"
        return result

    parsed_payload = completion.parsed
    if not isinstance(parsed_payload, dict):
        log.error("%s: LLM returned no parseable JSON; text=%r", entity.id, completion.text[:200])
        result.error = "llm: non-JSON response"
        return result

    result.evaluated = True

    status_changed = bool(parsed_payload.get("status_changed"))
    new_status = parsed_payload.get("new_status") or current_status
    if status_changed and new_status not in _STATUSES:
        log.error("%s: LLM proposed invalid status %r; ignoring", entity.id, new_status)
        status_changed = False
        new_status = current_status

    prose_changed = bool(parsed_payload.get("prose_changed"))
    rationale = (parsed_payload.get("rationale") or "").strip()

    # In dry-run, log the planned changes and stop short of writing.
    if dry_run:
        print(_format_planned_change(entity.id, parsed_payload, deltas, significant, current_status))
        if status_changed:
            result.moved = {"from": current_status, "to": new_status, "rationale": rationale}
        if prose_changed:
            result.prose_updated = True
        return result

    # ---- 5. Apply mutations ---------------------------------------------
    new_frontmatter = dict(entity.frontmatter)
    new_frontmatter["last_evaluated"] = today
    if status_changed:
        new_frontmatter["status"] = new_status

    new_body = entity.body
    if prose_changed:
        # Preserve the existing history bullets verbatim; the bullet we append
        # below for this evaluation goes through append_history afterwards so
        # we don't need to inline it into the rendered body.
        section_map = parse(entity.body)
        old_what = section_map.get("What it is", "")
        old_why = section_map.get("Why it matters", "")
        old_curr = section_map.get("Current assessment", "")
        new_what = parsed_payload.get("new_what_it_is") or old_what
        new_why = parsed_payload.get("new_why_it_matters") or old_why
        new_curr = parsed_payload.get("new_current_assessment") or old_curr
        history_bullets = _extract_history_bullets(entity.body)
        new_body = render(
            what_it_is=new_what,
            why_it_matters=new_why,
            current_assessment=new_curr,
            history_bullets=history_bullets,
        )

    # Persist the frontmatter+body changes (if any) before appending a
    # history bullet — append_history reads the entity off disk and would
    # otherwise drop our prose update.
    if status_changed or prose_changed or new_frontmatter != entity.frontmatter:
        vault.write_entity(Entity(id=entity.id, frontmatter=new_frontmatter, body=new_body))

    # ---- 6. History bullet ----------------------------------------------
    history_text: str | None = None
    if status_changed:
        suffix = f" {rationale}" if rationale else ""
        history_text = f"Status: {current_status} -> {new_status}.{suffix}".strip()
    elif prose_changed:
        suffix = f" {rationale}" if rationale else ""
        history_text = f"Re-evaluated.{suffix}".strip()

    if history_text:
        vault.append_history(entity.id, history_text, entry_date=today)

    # ---- 7. Embedding refresh -------------------------------------------
    if prose_changed and index is not None:
        # Re-read so we get the body with the new history bullet stripped
        # (body_for_embedding skips history anyway, but reading is cheap and
        # keeps this honest).
        refreshed = vault.read_entity(entity.id) or Entity(
            id=entity.id, frontmatter=new_frontmatter, body=new_body
        )
        try:
            index.upsert(entity.id, vault.body_for_embedding(refreshed.body))
        except Exception as exc:
            # Embedding failure shouldn't roll back the rest of the work,
            # but is worth surfacing.
            log.error("%s: embedding upsert failed: %s", entity.id, exc)

    # ---- 8. Record summary state ----------------------------------------
    if status_changed:
        result.moved = {"from": current_status, "to": new_status, "rationale": rationale}
    if prose_changed:
        result.prose_updated = True

    return result


# ---------------------------------------------------------------------------
# Batch orchestration.
# ---------------------------------------------------------------------------


def _pick_entities(only: str | None) -> list[Entity]:
    if only:
        ent = vault.read_entity(only)
        if ent is None:
            raise SystemExit(f"--only {only!r}: entity not found")
        return [ent]
    # `emerging` + `watch` only. `adopt` requires human action; `skip`/`settled` are done.
    return vault.list_entities(status="emerging") + vault.list_entities(status="watch")


def _build_router(cfg: config_module.Config) -> Router | None:
    """Construct a Router only if at least one provider in the enrichment
    chain is ready. Otherwise return None (offline dev mode)."""
    router = Router(cfg)
    if not router.available_providers("enrichment"):
        log.warning("no LLM providers available for task=enrichment; running without evaluation step")
        return None
    return router


def _write_summary(cfg: config_module.Config, today: str, results: list[_EntityResult]) -> None:
    summary = {
        "date": today,
        "evaluated": sum(1 for r in results if r.evaluated),
        "moved": [
            {"id": r.id, **r.moved}
            for r in results
            if r.moved is not None
        ],
        "significant_signal_change": [
            {"id": r.id, **r.significant_change}
            for r in results
            if r.significant_change is not None
        ],
        "prose_updated": [r.id for r in results if r.prose_updated],
        "errors": [
            {"id": r.id, "error": r.error}
            for r in results
            if r.error is not None
        ],
    }
    path = cfg.enrich_output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("enrich summary written to %s", path)


def _print_stdout_summary(results: list[_EntityResult]) -> None:
    moved = [r for r in results if r.moved is not None]
    significant = [r for r in results if r.significant_change is not None]
    errors = [r for r in results if r.error is not None]
    evaluated = sum(1 for r in results if r.evaluated)

    print(
        f"Enrich complete: {evaluated} entities evaluated, "
        f"{len(moved)} moved status, "
        f"{len(significant)} significant signal changes, "
        f"{len(errors)} error{'' if len(errors) == 1 else 's'}"
    )
    if moved:
        print("Moved:")
        # Pad ids for nicer alignment.
        id_width = max(len(r.id) for r in moved) + 1
        for r in moved:
            m = r.moved or {}
            rationale = m.get("rationale", "")
            arrow = f"{m.get('from', '?')} -> {m.get('to', '?')}"
            print(f"  - {r.id:<{id_width}} {arrow}" + (f"  ({rationale})" if rationale else ""))
    if errors:
        print("Errors:")
        for r in errors:
            print(f"  - {r.id}: {r.error}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="jobs.enrich",
        description="Weekly enrich pass: refresh signals, re-evaluate status/prose.",
    )
    p.add_argument("--only", help="Run on a single entity id (testing)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute snapshots and deltas, call the LLM, but write nothing.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = config_module.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)

    today = _today_iso()
    log.info("enrich starting (date=%s dry_run=%s only=%s)", today, args.dry_run, args.only)

    try:
        context_md = cfg.context_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning("context file %s not found; using empty context", cfg.context_path)
        context_md = ""

    entities = _pick_entities(args.only)
    log.info("processing %d entit%s", len(entities), "y" if len(entities) == 1 else "ies")

    router = _build_router(cfg)

    # Only construct the embedding Index when we'll actually use it. Loading
    # it warms a JSON file; not the end of the world but unnecessary on dry-run.
    index: Index | None = None
    if not args.dry_run:
        try:
            index = Index(cfg)
        except Exception as exc:
            log.error("could not load embedding index (continuing without): %s", exc)
            index = None

    results: list[_EntityResult] = []
    for entity in entities:
        try:
            result = _process_entity(
                entity,
                cfg=cfg,
                context_md=context_md,
                router=router,
                dry_run=args.dry_run,
                index=index,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 - we want to capture everything else
            log.exception("%s: unhandled exception during enrich", entity.id)
            result = _EntityResult(id=entity.id, error=f"{type(exc).__name__}: {exc}")
        results.append(result)

    if not args.dry_run:
        _write_summary(cfg, today, results)

    _print_stdout_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
