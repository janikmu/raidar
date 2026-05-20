"""Capture job — classify an input (GitHub URL, web URL, or free text) and
write a new artifact + create/update its concept in the vault.

Usage:
    uv run python -m jobs.capture <url-or-text>
    uv run python -m jobs.capture --force <url-or-text>       # bypass dedup
    uv run python -m jobs.capture --dry-run <url-or-text>     # preview, no writes

The job determines artifact type and metadata, maps the artifact to an
existing concept (or creates a new one), writes both files, appends a
signal snapshot for repos, and updates both embedding indexes.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timezone
from typing import Any

import trafilatura
import typer

from lib import config, vault
from lib.embeddings import Index
from lib.body import render_artifact, render_artifact_summary_table, render_concept
from lib.github import (
    RepoSnapshot,
    fetch_readme,
    fetch_repo,
    parse_repo_url,
)
from lib.llm import AllProvidersFailed, Router
from lib.logging_setup import setup as setup_logging
from lib.vault import (
    Artifact,
    Concept,
    append_signal,
    artifact_exists,
    find_artifact_by_github_repo,
    write_artifact,
    write_concept,
)


log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help=__doc__)


# ---------------------------------------------------------------------------
# LLM classification schema
# ---------------------------------------------------------------------------

_ARTIFACT_TYPES = ["repo", "paper", "post", "release", "spec"]
_EVALUATIONS = ["new", "promising", "recommended", "deprecated", "hype"]
_RELATIONSHIPS = ["introduces", "implements", "extends", "applies", "discusses"]
_CONCEPT_STATUSES = ["emerging", "watch", "invest"]

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "artifact_id",
        "artifact_type",
        "evaluation",
        "what_it_is",
        "evaluation_rationale",
        "tags",
        "relevance",
        "concept_id",
        "concept_label",
        "is_new_concept",
        "relationship",
    ],
    "properties": {
        "artifact_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"},
        "artifact_type": {"type": "string", "enum": _ARTIFACT_TYPES},
        "evaluation": {"type": "string", "enum": _EVALUATIONS},
        "what_it_is": {"type": "string"},
        "evaluation_rationale": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "relevance": {"type": "string", "enum": ["low", "medium", "high"]},
        "concept_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"},
        "concept_label": {"type": "string"},
        "is_new_concept": {"type": "boolean"},
        "relationship": {"type": "string", "enum": _RELATIONSHIPS},
        "concept_what_it_is": {"type": ["string", "null"]},
        "concept_why_it_matters": {"type": ["string", "null"]},
        "review_needed": {"type": "boolean"},
        "github_repo": {"type": ["string", "null"]},
    },
}

_SYSTEM_PROMPT = (
    "You classify AI/dev-tooling artifacts for a personal research radar. "
    "You receive my context, content describing an artifact, and the list of "
    "existing concepts in the vault. "
    "Your job: classify the artifact AND decide which concept it belongs to. "
    "\n\n"
    "ARTIFACT CLASSIFICATION:\n"
    "artifact_type: 'repo'=GitHub repo, 'paper'=arXiv/technical report, "
    "'post'=blog/newsletter, 'release'=model release, 'spec'=protocol/standard.\n"
    "evaluation: 'new'=just captured; 'promising'=early positive signals; "
    "'recommended'=mature, worth adopting; 'deprecated'=abandoned/superseded; "
    "'hype'=thin community, self-reported benchmarks, not safe to depend on.\n"
    "what_it_is: 2-3 sentences describing what this thing is.\n"
    "evaluation_rationale: 2-3 sentences explaining the evaluation, citing concrete evidence.\n"
    "\n"
    "CONCEPT MAPPING:\n"
    "concept_id: id of the best matching existing concept, OR a new 2-4 word kebab-case slug.\n"
    "concept_label: human-readable 2-4 word label.\n"
    "is_new_concept: true only if no existing concept is a good fit AND you can name it "
    "clearly in 2-4 words. If ambiguous, use the closest existing concept and set "
    "review_needed=true.\n"
    "concept_what_it_is / concept_why_it_matters: required only when is_new_concept=true.\n"
    "relationship: how this artifact relates to its concept. "
    "DEFAULT 'implements' — builds, embodies, or instantiates the concept (use for the vast majority of repos, "
    "including the most prominent or anchor artifact of a concept). "
    "'extends' — considerably advances the concept beyond its original form (rare). "
    "'applies' — uses the concept incidentally. "
    "'discusses' — paper/post that comments on the concept without building it. "
    "'introduces' — RESERVED: set ONLY when the artifact is the official tracking repo for a named "
    "spec/protocol/standard that defines this concept, OR the official companion code repo of a "
    "tech report/whitepaper that originated this concept. Not for 'first/most prominent'. "
    "If unsure, use 'implements'.\n"
    "\n"
    "Return ONLY JSON matching the provided schema. Be concise."
)

_CONTENT_LIMIT = 6000


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _repo_age_days(created_at_iso: str) -> int | None:
    try:
        dt = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
        return (datetime.now(tz=timezone.utc) - dt).days
    except (ValueError, TypeError):
        return None


def _summarize_snapshot(snap: RepoSnapshot) -> str:
    age = _repo_age_days(snap.created_at)
    age_str = f"{age // 365}y {(age % 365) // 30}m ({age}d)" if age is not None else "unknown"
    return (
        f"GitHub repo: {snap.owner}/{snap.name}\n"
        f"Description: {snap.description or '(none)'}\n"
        f"Stars: {snap.stars} | Forks: {snap.forks} | "
        f"Open issues: {snap.open_issues} | Commits (last 30d): {snap.commits_30d}\n"
        f"Repo age: {age_str} | Created: {snap.created_at} | Last push: {snap.pushed_at}"
    )


def _truncate(text: str, limit: int = _CONTENT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[...truncated at {limit} chars]"


def _fetch_web_text(url: str) -> str | None:
    raw = trafilatura.fetch_url(url)
    if not raw:
        return None
    extracted = trafilatura.extract(raw)
    if not extracted or len(extracted.strip()) < 100:
        return None
    return extracted


# ---------------------------------------------------------------------------
# Concept context builder
# ---------------------------------------------------------------------------


def _concept_context() -> str:
    """Return a compact summary of existing concepts for the LLM prompt."""
    concepts = vault.list_concepts()
    if not concepts:
        return "(no concepts in vault yet)"
    lines: list[str] = []
    for c in concepts:
        label = c.frontmatter.get("label", c.id)
        status = c.frontmatter.get("status", "?")
        n_artifacts = len(c.frontmatter.get("artifacts") or [])
        lines.append(f"- {c.id} | {label} | status={status} | {n_artifacts} artifact(s)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _assemble_prompt(
    *,
    context_md: str,
    source: str,
    raw_input: str,
    snapshot: RepoSnapshot | None,
    readme: str | None,
    web_text: str | None,
    concept_context: str,
) -> str:
    parts: list[str] = []
    parts.append("## My context\n" + context_md.strip())
    parts.append("## Existing concepts in vault\n" + concept_context)

    if source == "github" and snapshot is not None:
        parts.append("## Repo snapshot\n" + _summarize_snapshot(snapshot))
        parts.append("## README\n" + (_truncate(readme) if readme else "(no README found)"))
    elif source == "web" and web_text is not None:
        parts.append(f"## Source URL\n{raw_input}")
        parts.append("## Extracted article\n" + _truncate(web_text))
    else:
        parts.append("## Free-text note\n" + raw_input.strip())

    parts.append(
        "## Task\n"
        "Classify this artifact and map it to a concept. "
        "Return ONLY JSON matching the provided schema."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def _unique_artifact_id(proposed: str) -> str:
    if not artifact_exists(proposed):
        return proposed
    n = 2
    while True:
        candidate = f"{proposed}-{n}"
        if not artifact_exists(candidate):
            log.info("id collision: %r exists, using %r", proposed, candidate)
            return candidate
        n += 1


def _unique_concept_id(proposed: str) -> str:
    if not vault.concept_exists(proposed):
        return proposed
    n = 2
    while True:
        candidate = f"{proposed}-{n}"
        if not vault.concept_exists(candidate):
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# Concept upsert
# ---------------------------------------------------------------------------


def _upsert_concept(
    concept_id: str,
    concept_label: str,
    artifact_id: str,
    relationship: str,
    artifact: Artifact,
    parsed: dict[str, Any],
    today: str,
) -> Concept:
    """Create or update the concept to include this artifact."""
    existing = vault.read_concept(concept_id)

    if existing is None:
        # New concept
        artifact_rows = [{
            "id": artifact_id,
            "type": artifact.frontmatter.get("type", "repo"),
            "evaluation": artifact.frontmatter.get("evaluation", "new"),
        }]
        body = render_concept(
            what_it_is=parsed.get("concept_what_it_is") or "",
            why_it_matters=parsed.get("concept_why_it_matters") or "",
            current_assessment="(initial assessment pending further artifacts)",
            artifact_rows=artifact_rows,
            history_bullets=[f"{today}: Concept created. First artifact: {artifact_id}."],
        )
        fm: dict[str, Any] = {
            "id": concept_id,
            "label": concept_label,
            "type": "concept",
            "status": "emerging",
            "first_seen": today,
            "last_evaluated": today,
            "relevance": artifact.frontmatter.get("relevance", "medium"),
            "tags": list(artifact.frontmatter.get("tags") or [])[:8],
            "artifacts": [{"id": artifact_id, "relationship": relationship, "weight": "primary"}],
        }
        if parsed.get("review_needed"):
            fm["review_needed"] = True
        return Concept(id=concept_id, frontmatter=fm, body=body)

    # Existing concept — append artifact if not already listed
    fm = dict(existing.frontmatter)
    artifact_list: list[dict] = list(fm.get("artifacts") or [])
    existing_ids = {a.get("id") for a in artifact_list if isinstance(a, dict)}
    if artifact_id not in existing_ids:
        artifact_list.append({"id": artifact_id, "relationship": relationship, "weight": "primary"})
        fm["artifacts"] = artifact_list

    fm["last_evaluated"] = today

    # Rebuild artifact summary table from current vault state + new artifact
    all_arts = vault.find_artifacts_for_concept(concept_id)
    # Include the new artifact even if not yet written
    if artifact_id not in {a.id for a in all_arts}:
        all_arts.append(artifact)
    artifact_rows = [
        {
            "id": a.id,
            "type": a.frontmatter.get("type", "repo"),
            "evaluation": a.frontmatter.get("evaluation", "new"),
        }
        for a in all_arts
    ]

    from lib.body import parse as parse_body
    sections = parse_body(existing.body)
    hist_block = sections.get("History", "")
    history_bullets: list[str] = []
    for line in hist_block.splitlines():
        s = line.strip()
        if s.startswith("- "):
            history_bullets.append(s[2:])
        elif s.startswith("-"):
            history_bullets.append(s[1:].strip())
    history_bullets.append(f"{today}: Added artifact {artifact_id} ({relationship}).")

    new_body = render_concept(
        what_it_is=sections.get("What it is", ""),
        why_it_matters=sections.get("Why it matters", ""),
        current_assessment=sections.get("Current assessment", ""),
        artifact_rows=artifact_rows,
        history_bullets=history_bullets,
    )

    return Concept(id=concept_id, frontmatter=fm, body=new_body)


# ---------------------------------------------------------------------------
# Shared capture logic
# ---------------------------------------------------------------------------


class CaptureSkipped(Exception):
    def __init__(self, reason: str, existing_id: str | None = None) -> None:
        super().__init__(reason)
        self.existing_id = existing_id


def _capture_one(
    input_str: str,
    cfg,
    *,
    force: bool = False,
    today: str | None = None,
) -> str:
    """Capture a single input. Returns the artifact_id."""
    today = today or date.today().isoformat()

    # ----- 1. classify input shape ----------------------------------------
    parsed_repo = parse_repo_url(input_str) if _is_url(input_str) or "/" in input_str else None
    if parsed_repo is not None and not _is_url(input_str):
        if " " in input_str or input_str.count("/") != 1:
            parsed_repo = None

    snapshot: RepoSnapshot | None = None
    readme: str | None = None
    web_text: str | None = None

    if parsed_repo is not None:
        owner, name = parsed_repo
        source = "github"
        log.info("fetching GitHub repo %s/%s", owner, name)
        snapshot = fetch_repo(owner, name)
        if snapshot is None:
            raise ValueError(f"GitHub repo {owner}/{name} not found (404).")
        readme = fetch_readme(owner, name)
    elif _is_url(input_str):
        source = "web"
        log.info("fetching web URL %s", input_str)
        web_text = _fetch_web_text(input_str)
        if web_text is None:
            log.warning("falling back to text mode for %s", input_str)
            source = "text"
    else:
        source = "text"

    # ----- 2. exact dedup (GitHub repos only) -----------------------------
    if parsed_repo is not None:
        slug = f"{parsed_repo[0]}/{parsed_repo[1]}"
        existing = find_artifact_by_github_repo(slug)
        if existing and not force:
            raise CaptureSkipped(f"already tracked as {existing[0].id}", existing[0].id)

    # ----- 3. read context + concept context ------------------------------
    try:
        context_md = cfg.context_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        context_md = ""

    concept_ctx = _concept_context()

    prompt = _assemble_prompt(
        context_md=context_md,
        source=source,
        raw_input=input_str,
        snapshot=snapshot,
        readme=readme,
        web_text=web_text,
        concept_context=concept_ctx,
    )

    # ----- 4. LLM classification ------------------------------------------
    router = Router(cfg)
    try:
        completion = router.generate(
            task="classification",
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            response_schema=_SCHEMA,
            max_tokens=2048,
        )
    except AllProvidersFailed as exc:
        raise RuntimeError(f"LLM classification failed: {exc}") from exc

    parsed = completion.parsed
    if parsed is None or not isinstance(parsed, dict):
        raise RuntimeError(
            f"LLM did not return valid JSON (provider={completion.provider_name} "
            f"text={completion.text[:200]!r})"
        )

    required = _SCHEMA["required"]
    missing = [k for k in required if k not in parsed]
    if missing:
        raise RuntimeError(f"LLM response missing required fields: {missing}")

    # Override github_repo from actual fetch
    if parsed_repo is not None:
        parsed["github_repo"] = f"{parsed_repo[0]}/{parsed_repo[1]}"

    # ----- 5. ID collision handling ---------------------------------------
    artifact_id = _unique_artifact_id(parsed["artifact_id"])
    parsed["artifact_id"] = artifact_id

    concept_id_proposed = parsed["concept_id"]
    is_new_concept = bool(parsed.get("is_new_concept", False))

    # If concept already exists, use it; if proposed as new but collides, make unique
    if is_new_concept:
        concept_id = _unique_concept_id(concept_id_proposed)
    else:
        concept_id = concept_id_proposed
        # Validate the proposed existing concept actually exists
        if not vault.concept_exists(concept_id):
            log.warning(
                "LLM proposed non-existent concept %r; treating as new concept", concept_id
            )
            is_new_concept = True

    # ----- 6. build artifact body -----------------------------------------
    artifact_body = render_artifact(
        what_it_is=parsed["what_it_is"],
        evaluation_rationale=parsed["evaluation_rationale"],
        history_bullets=[
            f"{today}: Captured. evaluation={parsed['evaluation']}."
        ],
    )

    # ----- 7. soft dedup (artifact embedding) -----------------------------
    artifact_index = Index(cfg, layer="artifacts")
    # The body is prose-only (render_artifact output) — suitable for embedding as-is.
    embed_text = artifact_body
    threshold = float(cfg.thresholds.get("semantic_dedup", 0.92))
    try:
        neighbors = artifact_index.search(embed_text, top_k=3)
    except Exception as exc:  # noqa: BLE001
        log.warning("soft dedup search failed (%s); proceeding without it", exc)
        neighbors = []

    soft_hits = [
        (nid, score)
        for nid, score in neighbors
        if score > threshold and nid != artifact_id
    ]
    if soft_hits and not force:
        top_id, top_score = soft_hits[0]
        raise CaptureSkipped(
            f"similar to {top_id!r} (score={top_score:.2f}); use --force to override",
            top_id,
        )

    # ----- 8. assemble artifact frontmatter + write -----------------------
    artifact_fm: dict[str, Any] = {
        "id": artifact_id,
        "type": parsed["artifact_type"],
        "evaluation": parsed["evaluation"],
        "concept": concept_id,
        "relationship": parsed["relationship"],
        "first_seen": today,
        "last_evaluated": today,
        "tags": list(parsed["tags"]),
        "relevance": parsed["relevance"],
        "github_repo": parsed.get("github_repo"),
    }
    if parsed.get("github_repo"):
        artifact_fm["source_url"] = f"https://github.com/{parsed['github_repo']}"

    artifact = Artifact(id=artifact_id, frontmatter=artifact_fm, body=artifact_body)
    write_artifact(artifact)

    # ----- 9. upsert concept ----------------------------------------------
    concept = _upsert_concept(
        concept_id=concept_id,
        concept_label=parsed["concept_label"],
        artifact_id=artifact_id,
        relationship=parsed["relationship"],
        artifact=artifact,
        parsed=parsed,
        today=today,
    )
    write_concept(concept)

    # ----- 10. signal snapshot for repos ----------------------------------
    if source == "github" and snapshot is not None:
        signal: dict[str, Any] = {
            "date": today,
            "stars": snapshot.stars,
            "forks": snapshot.forks,
            "commits_30d": snapshot.commits_30d,
            "open_issues": snapshot.open_issues,
            "evaluation": parsed["evaluation"],
            "source": "capture",
        }
    else:
        signal = {"date": today, "evaluation": parsed["evaluation"], "source": "capture"}
    append_signal(artifact_id, signal)

    # ----- 11. upsert embeddings ------------------------------------------
    try:
        artifact_index.upsert(artifact_id, embed_text)
    except Exception as exc:  # noqa: BLE001
        log.error("artifact embedding upsert failed for %s: %s", artifact_id, exc)

    try:
        concept_index = Index(cfg, layer="concepts")
        concept_index.upsert(concept_id, concept.body)
    except Exception as exc:  # noqa: BLE001
        log.error("concept embedding upsert failed for %s: %s", concept_id, exc)

    # ----- 12. print confirmation -----------------------------------------
    new_marker = " [NEW CONCEPT]" if is_new_concept else ""
    review_marker = " ⚠ review_needed" if parsed.get("review_needed") else ""
    print(
        f"Captured: {artifact_id} ({parsed['artifact_type']}, "
        f"evaluation={parsed['evaluation']}, relevance={parsed['relevance']})"
    )
    print(f"  concept: {concept_id} ({parsed['concept_label']}){new_marker}{review_marker}")
    print(f"  relationship: {parsed['relationship']}")
    print(f"  source: {source}")
    if parsed.get("github_repo"):
        print(f"  github_repo: {parsed['github_repo']}")
    print(f"  tags: {', '.join(parsed['tags'])}")
    if source == "github" and snapshot is not None:
        print(
            f"  signal: stars={snapshot.stars} forks={snapshot.forks} "
            f"commits_30d={snapshot.commits_30d} open_issues={snapshot.open_issues}"
        )

    return artifact_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def capture(
    input: str = typer.Argument(..., help="GitHub URL, generic URL, or free text."),
    force: bool = typer.Option(False, "--force", help="Bypass exact + soft dedup checks."),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Assemble the prompt but don't call the LLM or write to the vault.",
    ),
) -> None:
    """Capture an artifact from a URL or free-text note."""
    cfg = config.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)
    today = date.today().isoformat()

    if dry_run:
        parsed_repo = parse_repo_url(input) if _is_url(input) or "/" in input else None
        if parsed_repo is not None and not _is_url(input):
            if " " in input or input.count("/") != 1:
                parsed_repo = None
        snapshot = fetch_repo(*parsed_repo) if parsed_repo else None
        readme = fetch_readme(*parsed_repo) if parsed_repo else None
        web_text = _fetch_web_text(input) if _is_url(input) and not parsed_repo else None
        source = "github" if parsed_repo else ("web" if _is_url(input) and web_text else "text")
        try:
            context_md = cfg.context_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            context_md = ""
        prompt = _assemble_prompt(
            context_md=context_md, source=source, raw_input=input,
            snapshot=snapshot, readme=readme, web_text=web_text,
            concept_context=_concept_context(),
        )
        print("=== DRY RUN ===")
        print(f"source: {source}")
        if parsed_repo:
            print(f"github_repo: {parsed_repo[0]}/{parsed_repo[1]}")
        print("\n=== SYSTEM PROMPT ===")
        print(_SYSTEM_PROMPT)
        print("\n=== USER PROMPT ===")
        print(prompt)
        return

    try:
        _capture_one(input, cfg, force=force, today=today)
    except CaptureSkipped as exc:
        if exc.existing_id:
            print(f"Already tracked as {exc.existing_id}")
        else:
            print(f"WARNING: {exc}", file=sys.stderr)
        raise typer.Exit(code=0 if exc.existing_id else 1)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
