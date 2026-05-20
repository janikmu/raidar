"""Weekly enrich job for AI Radar — two-pass evaluation.

Pass 1 — Artifact signal refresh + re-evaluation:
  For every artifact with type=repo and evaluation != deprecated/hype:
    1. Fetch fresh GitHub snapshot, append to signals/{id}.jsonl.
    2. Compute deltas vs previous snapshot.
    3. LLM evaluates: should evaluation change? Should prose change?
    4. Apply changes: frontmatter, Evaluation rationale, History, embedding.

Pass 2 — Concept lifecycle re-evaluation:
  For every concept with status in {emerging, watch}:
    1. Gather all linked artifacts + their current evaluations + signal summaries.
    2. LLM evaluates: should lifecycle status change?
    3. Apply changes: frontmatter, Current assessment, Artifact summary table, History.

Terminal concept suppression: artifacts whose concept is common/superseded/abandoned
are skipped in Pass 1 unless a new artifact was recently added.

Run modes:
    uv run python -m jobs.enrich               # full two-pass
    uv run python -m jobs.enrich --only ID     # single artifact (testing; skips concept pass)
    uv run python -m jobs.enrich --dry-run     # no writes, LLM still called
    uv run python -m jobs.enrich --concepts-only  # skip Pass 1, only re-evaluate concepts
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import typer

from lib import config as config_module
from lib import vault
from lib.embeddings import Index
from lib.body import parse, parse_artifact, parse_concept, render_artifact, render_concept
from lib.github import (
    RepoSnapshot,
    TerminalError,
    fetch_repo,
    fetch_star_history,
    get_rate_limit_remaining,
    parse_repo_url,
)
from lib.llm import AllProvidersFailed, Router
from lib.logging_setup import setup as setup_logging
from lib.vault import Artifact, Concept

log = logging.getLogger("jobs.enrich")
app = typer.Typer(add_completion=False, help=__doc__)

# ---------------------------------------------------------------------------
# LLM schemas
# ---------------------------------------------------------------------------

_EVALUATIONS = ["new", "promising", "recommended", "deprecated", "hype"]
_CONCEPT_STATUSES = ["emerging", "watch", "invest", "common", "superseded", "abandoned"]

_ARTIFACT_EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["evaluation_changed", "new_evaluation", "prose_changed", "rationale"],
    "properties": {
        "evaluation_changed": {"type": "boolean"},
        "new_evaluation": {"type": "string", "enum": _EVALUATIONS},
        "prose_changed": {"type": "boolean"},
        "new_what_it_is": {"type": ["string", "null"]},
        "new_evaluation_rationale": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
}

_ARTIFACT_SYSTEM = (
    "You are the curator of an AI-tooling research radar. You are reviewing one artifact "
    "given its signal history and the researcher's context. "
    "Assign an evaluation status based on evidence quality, not just star count. "
    "evaluation='recommended': mature, actively maintained, strong community or org backing, worth adopting. "
    "evaluation='promising': clear value, growing traction, not yet proven at scale. "
    "evaluation='hype': thin community, self-reported benchmarks, no substantial adoption. "
    "evaluation='deprecated': archived, abandoned, or superseded by something better. "
    "evaluation='new': insufficient signal to judge. "
    "Only change prose when something materially changed (evaluation flip, big signal move, archived). "
    "Return JSON matching the provided schema."
)

_CONCEPT_EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status_changed", "new_status", "prose_changed", "rationale"],
    "properties": {
        "status_changed": {"type": "boolean"},
        "new_status": {"type": "string", "enum": _CONCEPT_STATUSES},
        "prose_changed": {"type": "boolean"},
        "new_current_assessment": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
}

_CONCEPT_SYSTEM = (
    "You are the curator of an AI-tooling research radar. You are reviewing one concept "
    "given all its artifacts and their current evaluations. "
    "Lifecycle rules: "
    "'emerging' = concept is new or unclear trajectory; "
    "'watch' = concept is gaining traction, multiple implementations, at least one 'promising'; "
    "'invest' = concept has at least one 'recommended' artifact, is proven in production; "
    "'common' = stable and widespread — table-stakes, no need to track further; "
    "'superseded' = a better approach replaced this concept (note what replaced it in rationale); "
    "'abandoned' = community dissolved, no active implementations. "
    "Status is driven by artifact quality, not just quantity. "
    "Update current_assessment only when the status changes or the landscape materially shifted. "
    "Return JSON matching the provided schema."
)

# ---------------------------------------------------------------------------
# Delta and result types (reused from pre-migration enrich)
# ---------------------------------------------------------------------------


@dataclass
class _Deltas:
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
class _ArtifactResult:
    id: str
    evaluated: bool = False
    moved: dict[str, Any] | None = None          # {from, to, rationale}
    prose_updated: bool = False
    significant_change: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class _ConceptResult:
    id: str
    evaluated: bool = False
    moved: dict[str, Any] | None = None          # {from, to, rationale}
    prose_updated: bool = False
    error: str | None = None


@dataclass
class _EnrichOutput:
    artifact_results: list[_ArtifactResult] = field(default_factory=list)
    concept_results: list[_ConceptResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Signal + delta helpers
# ---------------------------------------------------------------------------


def _today_iso() -> str:
    return date.today().isoformat()


def _significant(
    deltas: _Deltas,
    prev: dict[str, Any] | None,
    thresholds: dict[str, Any],
) -> bool:
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
        and (deltas.stars_delta / prev_stars * 100) >= rel_star_pct
    ):
        return True

    prev_commits = prev.get("commits_30d")
    if (
        deltas.commits_30d_delta is not None
        and isinstance(prev_commits, (int, float))
        and prev_commits > 0
        and abs(deltas.commits_30d_delta / prev_commits * 100) >= rel_commits_pct
    ):
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


def _snapshot_from_repo(repo: RepoSnapshot, evaluation: str, today: str) -> dict[str, Any]:
    return {
        "date": today,
        "stars": repo.stars,
        "forks": repo.forks,
        "commits_30d": repo.commits_30d,
        "open_issues": repo.open_issues,
        "evaluation": evaluation,
        "source": "enrich",
    }


def _extract_history_bullets(body: str) -> list[str]:
    sections = parse(body)
    hist_block = sections.get("History", "")
    out: list[str] = []
    for line in hist_block.splitlines():
        s = line.strip()
        if s.startswith("- "):
            out.append(s[2:].strip())
        elif s.startswith("-"):
            out.append(s[1:].strip())
    return out


def has_star_history_backfill(artifact_id: str) -> bool:
    """Return True if the signal file has any entry with source='backfill'."""
    signals = vault.read_signals(artifact_id)
    return any(s.get("source") == "backfill" for s in signals)


# ---------------------------------------------------------------------------
# Pass 1: Artifact enrichment
# ---------------------------------------------------------------------------


def _build_artifact_prompt(
    context_md: str,
    artifact: Artifact,
    signals_tail: list[dict[str, Any]],
    deltas: _Deltas,
    significant: bool,
) -> str:
    fm_dump = json.dumps(artifact.frontmatter, indent=2, sort_keys=True, default=str)
    signals_dump = json.dumps(signals_tail, indent=2, default=str)
    deltas_dump = json.dumps(deltas.as_dict(), indent=2)
    return (
        "# Researcher context\n"
        f"{context_md.strip()}\n\n"
        "# Artifact under review\n"
        f"## frontmatter\n```json\n{fm_dump}\n```\n\n"
        f"## body\n{artifact.body.strip()}\n\n"
        "# Signal history (most recent last)\n"
        f"```json\n{signals_dump}\n```\n\n"
        "# Deltas (current vs previous snapshot)\n"
        f"```json\n{deltas_dump}\n```\n"
        f"significant_change_flag: {significant}\n\n"
        "# Task\n"
        "Evaluate this artifact. If evaluation_changed, set new_evaluation. "
        "If prose_changed, provide new_what_it_is and/or new_evaluation_rationale. "
        "Keep prose terse — 2-3 sentences per section. "
        "Return JSON per the schema."
    )


def _enrich_artifact(
    artifact: Artifact,
    *,
    router: Router,
    context_md: str,
    thresholds: dict[str, Any],
    artifact_index: Index,
    today: str,
    dry_run: bool,
) -> _ArtifactResult:
    result = _ArtifactResult(id=artifact.id)
    fm = artifact.frontmatter
    github_repo = fm.get("github_repo")

    # --- 1. GitHub fetch --------------------------------------------------
    snap: RepoSnapshot | None = None
    if github_repo:
        parsed_url = parse_repo_url(github_repo)
        if parsed_url:
            owner, name = parsed_url
            rl = get_rate_limit_remaining()
            if rl is not None and rl <= 200:
                result.error = f"rate_limit_remaining={rl}; skipping"
                return result
            try:
                snap = fetch_repo(owner, name)
            except TerminalError as exc:
                result.error = f"github terminal: {exc}"
                return result

    # --- 2. Compute deltas ------------------------------------------------
    signals = vault.read_signals(artifact.id)
    # Auto-backfill star history on first enrich
    if snap and not has_star_history_backfill(artifact.id) and not dry_run:
        log.info("auto-backfilling star history for %s", artifact.id)
        try:
            history = fetch_star_history(*parse_repo_url(github_repo), snap.stars)
            for star_count, starred_at in history:
                vault.append_signal(artifact.id, {
                    "date": starred_at[:10],
                    "stars": star_count,
                    "source": "backfill",
                })
            signals = vault.read_signals(artifact.id)
        except Exception as exc:  # noqa: BLE001
            log.warning("star history backfill failed for %s: %s", artifact.id, exc)

    # Filter out backfill signals for delta computation
    real_signals = [s for s in signals if s.get("source") != "backfill"]
    prev_snap = real_signals[-1] if real_signals else None
    new_snap_dict: dict[str, Any] = {}

    if snap:
        current_evaluation = fm.get("evaluation", "new")
        new_snap_dict = _snapshot_from_repo(snap, current_evaluation, today)
        if not dry_run:
            vault.append_signal(artifact.id, new_snap_dict)

    deltas = _compute_deltas(new_snap_dict, prev_snap)
    significant = _significant(deltas, prev_snap, thresholds)
    if significant:
        result.significant_change = {"deltas": deltas.as_dict()}

    # --- 3. LLM evaluation ------------------------------------------------
    signals_tail = signals[-24:]
    prompt = _build_artifact_prompt(context_md, artifact, signals_tail, deltas, significant)
    try:
        completion = router.generate(
            task="enrichment",
            prompt=prompt,
            system=_ARTIFACT_SYSTEM,
            response_schema=_ARTIFACT_EVAL_SCHEMA,
            max_tokens=1024,
        )
    except AllProvidersFailed as exc:
        result.error = f"llm failed: {exc}"
        return result

    parsed = completion.parsed
    if not isinstance(parsed, dict):
        result.error = "llm: non-JSON response"
        return result

    result.evaluated = True
    new_evaluation = parsed.get("new_evaluation", fm.get("evaluation", "new"))
    evaluation_changed = bool(parsed.get("evaluation_changed", False))
    prose_changed = bool(parsed.get("prose_changed", False))

    if evaluation_changed:
        result.moved = {
            "from": fm.get("evaluation"),
            "to": new_evaluation,
            "rationale": parsed.get("rationale", ""),
        }

    # --- 4. Write changes -------------------------------------------------
    if not dry_run and (evaluation_changed or prose_changed or snap):
        sections = parse_artifact(artifact.body)
        history_bullets = _extract_history_bullets(artifact.body)

        what_it_is = parsed.get("new_what_it_is") if prose_changed else sections.get("What it is", "")
        eval_rationale = parsed.get("new_evaluation_rationale") if prose_changed else sections.get("Evaluation rationale", "")

        if evaluation_changed:
            history_bullets.append(
                f"{today}: evaluation {fm.get('evaluation')} → {new_evaluation}. "
                f"{parsed.get('rationale', '')[:120]}"
            )
        elif prose_changed:
            history_bullets.append(f"{today}: Prose updated.")

        new_body = render_artifact(
            what_it_is=what_it_is or sections.get("What it is", ""),
            evaluation_rationale=eval_rationale or sections.get("Evaluation rationale", ""),
            history_bullets=history_bullets,
        )

        new_fm = dict(fm)
        new_fm["evaluation"] = new_evaluation
        new_fm["last_evaluated"] = today
        if snap:
            new_fm["stars"] = snap.stars

        updated = Artifact(id=artifact.id, frontmatter=new_fm, body=new_body)
        vault.write_artifact(updated)
        result.prose_updated = prose_changed

        try:
            artifact_index.upsert(artifact.id, new_body)
        except Exception as exc:  # noqa: BLE001
            log.error("embedding upsert failed for %s: %s", artifact.id, exc)

    return result


# ---------------------------------------------------------------------------
# Pass 2: Concept lifecycle re-evaluation
# ---------------------------------------------------------------------------


def _build_concept_prompt(
    context_md: str,
    concept: Concept,
    artifacts: list[Artifact],
) -> str:
    artifact_lines: list[str] = []
    for art in artifacts:
        fm = art.frontmatter
        sections = parse_artifact(art.body)
        what = sections.get("What it is", "").split("\n")[0][:100]
        eval_val = fm.get("evaluation", "new")
        rel = fm.get("relationship", "implements")
        atype = fm.get("type", "repo")
        artifact_lines.append(
            f"- {art.id} ({atype}, {rel}, evaluation={eval_val}): {what}"
        )

    return (
        "# Researcher context\n"
        f"{context_md.strip()}\n\n"
        "# Concept under review\n"
        f"## frontmatter\n```json\n{json.dumps(concept.frontmatter, indent=2, default=str)}\n```\n\n"
        f"## body\n{concept.body.strip()}\n\n"
        "# Linked artifacts\n"
        + "\n".join(artifact_lines)
        + "\n\n"
        "# Task\n"
        "Evaluate this concept's lifecycle status based on the quality of its artifacts. "
        "If status_changed, set new_status and write new_current_assessment. "
        "Return JSON per the schema."
    )


def _enrich_concept(
    concept: Concept,
    *,
    router: Router,
    context_md: str,
    concept_index: Index,
    today: str,
    dry_run: bool,
) -> _ConceptResult:
    result = _ConceptResult(id=concept.id)
    artifacts = vault.find_artifacts_for_concept(concept.id)

    if not artifacts:
        log.debug("concept %s has no artifacts, skipping", concept.id)
        return result

    prompt = _build_concept_prompt(context_md, concept, artifacts)
    try:
        completion = router.generate(
            task="enrichment",
            prompt=prompt,
            system=_CONCEPT_SYSTEM,
            response_schema=_CONCEPT_EVAL_SCHEMA,
            max_tokens=1024,
        )
    except AllProvidersFailed as exc:
        result.error = f"llm failed: {exc}"
        return result

    parsed = completion.parsed
    if not isinstance(parsed, dict):
        result.error = "llm: non-JSON response"
        return result

    result.evaluated = True
    fm = concept.frontmatter
    new_status = parsed.get("new_status", fm.get("status", "emerging"))
    status_changed = bool(parsed.get("status_changed", False))
    prose_changed = bool(parsed.get("prose_changed", False))

    if status_changed:
        result.moved = {
            "from": fm.get("status"),
            "to": new_status,
            "rationale": parsed.get("rationale", ""),
        }

    if not dry_run and (status_changed or prose_changed):
        sections = parse_concept(concept.body)
        history_bullets = _extract_history_bullets(concept.body)

        new_assessment = (
            parsed.get("new_current_assessment") or sections.get("Current assessment", "")
        )

        if status_changed:
            history_bullets.append(
                f"{today}: status {fm.get('status')} → {new_status}. "
                f"{parsed.get('rationale', '')[:120]}"
            )
        elif prose_changed:
            history_bullets.append(f"{today}: Assessment updated.")

        # Rebuild artifact summary table from current vault state
        artifact_rows = [
            {
                "id": a.id,
                "type": a.frontmatter.get("type", "repo"),
                "evaluation": a.frontmatter.get("evaluation", "new"),
            }
            for a in artifacts
        ]

        new_body = render_concept(
            what_it_is=sections.get("What it is", ""),
            why_it_matters=sections.get("Why it matters", ""),
            current_assessment=new_assessment,
            artifact_rows=artifact_rows,
            history_bullets=history_bullets,
        )

        new_fm = dict(fm)
        new_fm["status"] = new_status
        new_fm["last_evaluated"] = today

        updated = Concept(id=concept.id, frontmatter=new_fm, body=new_body)
        vault.write_concept(updated)
        result.prose_updated = prose_changed

        try:
            concept_index.upsert(concept.id, new_body)
        except Exception as exc:  # noqa: BLE001
            log.error("concept embedding upsert failed for %s: %s", concept.id, exc)

    return result


# ---------------------------------------------------------------------------
# Output + printing
# ---------------------------------------------------------------------------

_TERMINAL_CONCEPT_STATUSES = {"common", "superseded", "abandoned"}


def _print_artifact_result(r: _ArtifactResult, dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    if r.error:
        print(f"{prefix}{r.id}: ERROR {r.error}")
        return
    if not r.evaluated:
        print(f"{prefix}{r.id}: skipped")
        return
    parts = [f"{prefix}{r.id}:"]
    if r.moved:
        parts.append(f"evaluation {r.moved['from']} → {r.moved['to']}")
    else:
        parts.append("evaluation unchanged")
    if r.prose_updated:
        parts.append("prose updated")
    if r.significant_change:
        d = r.significant_change.get("deltas", {})
        sd = d.get("stars_delta")
        parts.append(f"Δstars={sd:+d}" if sd is not None else "")
    print("  ".join(p for p in parts if p))


def _print_concept_result(r: _ConceptResult, dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    if r.error:
        print(f"{prefix}{r.id}: ERROR {r.error}")
        return
    if not r.evaluated:
        return
    parts = [f"{prefix}concept {r.id}:"]
    if r.moved:
        parts.append(f"status {r.moved['from']} → {r.moved['to']}")
    else:
        parts.append("status unchanged")
    print("  ".join(parts))


def _write_enrich_output(output: _EnrichOutput, cfg: config_module.Config) -> None:
    data = {
        "artifact_evaluations_changed": [
            {
                "id": r.id,
                "from": r.moved["from"],
                "to": r.moved["to"],
                "rationale": r.moved.get("rationale", ""),
            }
            for r in output.artifact_results
            if r.moved
        ],
        "concept_status_changed": [
            {
                "id": r.id,
                "from": r.moved["from"],
                "to": r.moved["to"],
                "rationale": r.moved.get("rationale", ""),
            }
            for r in output.concept_results
            if r.moved
        ],
        "significant_signal_changes": [
            {"id": r.id, "deltas": r.significant_change["deltas"]}
            for r in output.artifact_results
            if r.significant_change
        ],
        "errors": [
            {"id": r.id, "error": r.error}
            for r in (output.artifact_results + output.concept_results)  # type: ignore[operator]
            if r.error
        ],
    }
    out_path = cfg.enrich_output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    log.info("enrich output written to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def enrich(
    only: str | None = typer.Option(None, "--only", metavar="ID", help="Single artifact ID."),
    dry_run: bool = typer.Option(False, "--dry-run", help="No writes, LLM still called."),
    concepts_only: bool = typer.Option(
        False, "--concepts-only", help="Skip Pass 1, only run concept re-evaluation."
    ),
) -> None:
    """Weekly enrichment: refresh artifact signals + re-evaluate concepts."""
    cfg = config_module.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)
    today = _today_iso()

    try:
        context_md = cfg.context_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        context_md = ""

    router = Router(cfg)
    if not router.available_providers("enrichment"):
        print("WARNING: no providers available for task=enrichment. LLM calls will fail.")

    artifact_index = Index(cfg, layer="artifacts")
    concept_index = Index(cfg, layer="concepts")
    output = _EnrichOutput()

    # -----------------------------------------------------------------------
    # Pass 1: Artifacts
    # -----------------------------------------------------------------------
    if not concepts_only:
        if only:
            artifact = vault.read_artifact(only)
            if artifact is None:
                print(f"ERROR: artifact {only!r} not found.", file=sys.stderr)
                raise typer.Exit(code=1)
            artifacts_to_enrich = [artifact]
        else:
            all_artifacts = vault.list_artifacts()
            # Skip artifacts whose concept is in a terminal state
            terminal_concepts: set[str] = set()
            for concept in vault.list_concepts():
                if concept.frontmatter.get("status") in _TERMINAL_CONCEPT_STATUSES:
                    terminal_concepts.add(concept.id)

            artifacts_to_enrich = []
            for art in all_artifacts:
                ev = art.frontmatter.get("evaluation", "new")
                if ev in ("deprecated",):
                    continue
                concept_id = art.frontmatter.get("concept", "")
                if concept_id in terminal_concepts:
                    log.debug("skipping %s: concept %s is terminal", art.id, concept_id)
                    continue
                artifacts_to_enrich.append(art)

        print(f"\n=== Pass 1: Enriching {len(artifacts_to_enrich)} artifacts ===")
        for art in artifacts_to_enrich:
            result = _enrich_artifact(
                art,
                router=router,
                context_md=context_md,
                thresholds=cfg.thresholds,
                artifact_index=artifact_index,
                today=today,
                dry_run=dry_run,
            )
            output.artifact_results.append(result)
            _print_artifact_result(result, dry_run)

    # -----------------------------------------------------------------------
    # Pass 2: Concepts
    # -----------------------------------------------------------------------
    if not only:
        active_concepts = [
            c for c in vault.list_concepts()
            if c.frontmatter.get("status") not in _TERMINAL_CONCEPT_STATUSES
        ]
        print(f"\n=== Pass 2: Re-evaluating {len(active_concepts)} concepts ===")
        for concept in active_concepts:
            result = _enrich_concept(
                concept,
                router=router,
                context_md=context_md,
                concept_index=concept_index,
                today=today,
                dry_run=dry_run,
            )
            output.concept_results.append(result)
            _print_concept_result(result, dry_run)

    # -----------------------------------------------------------------------
    # Summary + output file
    # -----------------------------------------------------------------------
    art_moved = sum(1 for r in output.artifact_results if r.moved)
    art_errors = sum(1 for r in output.artifact_results if r.error)
    concept_moved = sum(1 for r in output.concept_results if r.moved)
    concept_errors = sum(1 for r in output.concept_results if r.error)

    print(f"\n=== Summary ===")
    print(f"Artifacts: {len(output.artifact_results)} processed, {art_moved} evaluation changes, {art_errors} errors")
    print(f"Concepts:  {len(output.concept_results)} processed, {concept_moved} status changes, {concept_errors} errors")

    if not dry_run and not only:
        _write_enrich_output(output, cfg)

    if art_errors or concept_errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
