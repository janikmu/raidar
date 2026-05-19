"""Capture job — classify an input (GitHub URL, web URL, or free text) and
write a fresh entity to the vault.

Usage:
    raidar capture <url-or-text>
    raidar capture --force <url-or-text>
    raidar capture --update <id> <url-or-text>
    raidar capture --dry-run <url-or-text>

The job classifies the input shape (GitHub repo, generic web page, or free
text), runs exact + soft dedup against the vault, calls the LLM router for
classification, and writes the canonical entity body + first signal +
embedding. Failures print a clear error and exit non-zero — capture is a
single-shot job, not a batch.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timezone
from typing import Any

import trafilatura
import typer

from lib import config, vault
from lib.embeddings import Index
from lib.entity_body import render
from lib.github import (
    RepoSnapshot,
    fetch_readme,
    fetch_repo,
    fetch_star_history,
    get_rate_limit_remaining,
    parse_repo_url,
)
from lib.llm import AllProvidersFailed, Router
from lib.logging_setup import setup as setup_logging
from lib.vault import (
    Entity,
    append_history,
    append_signal,
    body_for_embedding,
    entity_exists,
    find_by_github_repo,
    write_entity,
)

log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help=__doc__)


# ---------------------------------------------------------------------------
# Classification schema for the LLM
# ---------------------------------------------------------------------------

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "type",
        "tags",
        "github_repo",
        "what_it_is",
        "why_it_matters",
        "current_assessment",
        "relevance",
        "initial_status",
    ],
    "properties": {
        "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"},
        "type": {"type": "string", "enum": [
            "agent-framework", "llm-client", "inference", "rag",
            "evaluation", "data", "fine-tuning", "tool", "model",
            "platform", "paper", "pattern",
        ]},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "github_repo": {"type": ["string", "null"]},
        "what_it_is": {"type": "string"},
        "why_it_matters": {"type": "string"},
        "current_assessment": {"type": "string"},
        "relevance": {"type": "string", "enum": ["low", "medium", "high"]},
        "initial_status": {"type": "string", "enum": ["emerging", "watch", "adopt"]},
    },
}

_SYSTEM_PROMPT = (
    "You classify AI/dev-tooling entities for a personal research radar. "
    "You receive my context (relevance anchor) followed by content describing "
    "an entity (a GitHub repo, an article, or a free-form note). Return ONLY "
    "JSON matching the provided schema. Be concise — each prose field should "
    "be 2-4 sentences. Tags are short kebab-case strings (3-8 tags). The id "
    "is a short kebab-case slug (lowercase, alphanumerics + hyphens) suitable "
    "as a filename stem. "
    "For initial_status use the 'Repo age' field as the primary signal: "
    "'emerging' for projects < 6 months old or < 300 stars (new/unproven); "
    "'watch' for projects 6 months–2 years old or 300–5k stars (actively growing); "
    "'adopt' for projects > 2 years old AND > 5k stars, or tools already in "
    "broad production use regardless of age. A 3-year-old 10k-star repo is NOT emerging. "
    "For type, pick the most specific match: "
    "'agent-framework' = multi-agent orchestration (LangChain, CrewAI, AutoGen); "
    "'llm-client' = model-access/routing clients (LiteLLM, Ollama, LMStudio); "
    "'inference' = serving/inference engines (vLLM, TGI, Triton); "
    "'rag' = retrieval-augmented generation pipelines or vector stores; "
    "'evaluation' = benchmarking, evals, red-teaming; "
    "'data' = dataset tooling, synthetic data, data processing; "
    "'fine-tuning' = training/PEFT/LoRA frameworks; "
    "'model' = weights or model checkpoints; "
    "'platform' = hosted SaaS or developer platforms; "
    "'paper' = research paper or arxiv preprint; "
    "'pattern' = architectural pattern, technique, or concept; "
    "'tool' = generic utility that doesn't fit a narrower category."
)

# Truncate fetched content before stuffing into the prompt to keep token cost
# bounded. Generous enough for a README or article lede.
_CONTENT_LIMIT = 6000


# ---------------------------------------------------------------------------
# Input classification
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
    if age is not None:
        age_str = f"{age // 365}y {(age % 365) // 30}m ({age}d)"
    else:
        age_str = "unknown"
    return (
        f"GitHub repo: {snap.owner}/{snap.name}\n"
        f"Description: {snap.description or '(none)'}\n"
        f"Stars: {snap.stars} | Forks: {snap.forks} | "
        f"Open issues: {snap.open_issues} | Commits (last 30d): {snap.commits_30d}\n"
        f"Default branch: {snap.default_branch} | "
        f"Language: {snap.language or '(none)'} | "
        f"License: {snap.license_spdx or '(none)'} | "
        f"Archived: {snap.archived}\n"
        f"Repo age: {age_str} | Created: {snap.created_at} | Last push: {snap.pushed_at}"
    )


def _truncate(text: str, limit: int = _CONTENT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[...truncated at {limit} chars]"


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
) -> str:
    parts: list[str] = []
    parts.append("## My context\n" + context_md.strip())

    if source == "github" and snapshot is not None:
        parts.append("## Repo snapshot\n" + _summarize_snapshot(snapshot))
        if readme:
            parts.append("## README\n" + _truncate(readme))
        else:
            parts.append("## README\n(no README found)")
    elif source == "web" and web_text is not None:
        parts.append(f"## Source URL\n{raw_input}")
        parts.append("## Extracted article\n" + _truncate(web_text))
    else:
        # text (or web-fallback)
        parts.append("## Free-text note\n" + raw_input.strip())

    parts.append(
        "## Task\n"
        "Classify this entity. Produce ONLY JSON matching the provided "
        "schema. Be concise — each prose field is 2-4 sentences. Tags are "
        "short kebab-case strings (3-8 tags)."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_id(proposed: str) -> str:
    """Append `-2`, `-3`, ... until we find an unused id."""
    if not entity_exists(proposed):
        return proposed
    n = 2
    while True:
        candidate = f"{proposed}-{n}"
        if not entity_exists(candidate):
            log.info("id collision: %r already exists, using %r", proposed, candidate)
            return candidate
        n += 1


def _fetch_web_text(url: str) -> str | None:
    """Fetch + extract main article text from a non-GitHub URL.

    Returns extracted text on success, None if extraction yielded nothing
    useful (caller falls back to text mode).
    """
    raw = trafilatura.fetch_url(url)
    if not raw:
        log.warning("trafilatura.fetch_url returned empty for %s", url)
        return None
    extracted = trafilatura.extract(raw)
    if not extracted or len(extracted.strip()) < 100:
        log.warning(
            "trafilatura extracted <100 chars from %s; falling back to text mode",
            url,
        )
        return None
    return extracted


# ---------------------------------------------------------------------------
# Shared capture logic (also used by bulk_capture)
# ---------------------------------------------------------------------------


class CaptureSkipped(Exception):
    """Raised when capture is deliberately skipped (dedup). Not an error."""

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
    """Capture a single input and return the entity_id.

    Raises CaptureSkipped on exact or soft dedup (when not forced).
    Raises RuntimeError on LLM failure or bad response.
    Raises ValueError on GitHub 404.
    """
    today = today or date.today().isoformat()

    # ----- 1. classify input shape -------------------------------------
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

    # ----- 2. exact dedup (GitHub only) --------------------------------
    if parsed_repo is not None:
        slug = f"{parsed_repo[0]}/{parsed_repo[1]}"
        existing = find_by_github_repo(slug)
        if existing and not force:
            raise CaptureSkipped(f"already tracked as {existing[0].id}", existing[0].id)

    # ----- 3. read context anchor --------------------------------------
    try:
        context_md = cfg.context_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning("context.md not found at %s; using empty context", cfg.context_path)
        context_md = ""

    prompt = _assemble_prompt(
        context_md=context_md,
        source=source,
        raw_input=input_str,
        snapshot=snapshot,
        readme=readme,
        web_text=web_text,
    )

    # ----- 4. classify with the LLM ------------------------------------
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

    if parsed_repo is not None:
        parsed["github_repo"] = f"{parsed_repo[0]}/{parsed_repo[1]}"

    # ----- 5. id collision handling ------------------------------------
    entity_id = _unique_id(parsed["id"])
    parsed["id"] = entity_id

    # ----- 6. build body + soft dedup ----------------------------------
    body = render(
        what_it_is=parsed["what_it_is"],
        why_it_matters=parsed["why_it_matters"],
        current_assessment=parsed["current_assessment"],
        history_bullets=[f"{today}: Captured. Status set to {parsed['initial_status']}."],
    )

    index = Index(cfg)
    embed_text = body_for_embedding(body)
    threshold = float(cfg.thresholds.get("semantic_dedup", 0.92))
    try:
        neighbors = index.search(embed_text, top_k=3)
    except Exception as exc:  # noqa: BLE001
        log.warning("soft dedup search failed (%s); proceeding without it", exc)
        neighbors = []

    soft_hits = [
        (nid, score)
        for nid, score in neighbors
        if score > threshold and nid != entity_id
    ]
    if soft_hits and not force:
        top_id, top_score = soft_hits[0]
        raise CaptureSkipped(
            f"similar to {top_id!r} (score={top_score:.2f}); use --force to override",
            top_id,
        )
    for nid, score in soft_hits:
        log.info("soft-dedup neighbor: %s score=%.3f (forced through)", nid, score)

    # ----- 7. assemble frontmatter + write entity ----------------------
    frontmatter_dict: dict[str, Any] = {
        "id": entity_id,
        "type": parsed["type"],
        "status": parsed["initial_status"],
        "first_seen": today,
        "last_evaluated": today,
        "source": source,
        "tags": list(parsed["tags"]),
        "relevance": parsed["relevance"],
        "github_repo": parsed["github_repo"],
    }
    write_entity(Entity(id=entity_id, frontmatter=frontmatter_dict, body=body))

    # ----- 8. append signal snapshot -----------------------------------
    if source == "github" and snapshot is not None:
        signal: dict[str, Any] = {
            "date": today,
            "stars": snapshot.stars,
            "forks": snapshot.forks,
            "commits_30d": snapshot.commits_30d,
            "open_issues": snapshot.open_issues,
            "status": parsed["initial_status"],
            "source": "capture",
        }
    else:
        signal = {"date": today, "status": parsed["initial_status"], "source": "capture"}
    append_signal(entity_id, signal)

    # ----- 9. upsert embedding -----------------------------------------
    try:
        index.upsert(entity_id, embed_text)
    except Exception as exc:  # noqa: BLE001
        log.error("embedding upsert failed for %s: %s", entity_id, exc)

    # ----- 10. backfill star history -----------------------------------
    # Backfill immediately so the first enrich pass (and reevaluate) have
    # full historical context and can assign the correct initial status.
    backfill_points = 0
    if source == "github" and snapshot is not None and snapshot.stars > 0:
        rl = get_rate_limit_remaining()
        if rl is None or rl > 300:
            try:
                history = fetch_star_history(
                    parsed_repo[0], parsed_repo[1], snapshot.stars  # type: ignore[index]
                )
                for star_count, starred_at in history:
                    append_signal(entity_id, {
                        "date": starred_at[:10],
                        "stars": star_count,
                        "source": "backfill",
                    })
                backfill_points = len(history)
            except Exception as exc:  # noqa: BLE001
                log.warning("star history backfill failed for %s: %s", entity_id, exc)
        else:
            log.info("skipping backfill for %s (rate_limit_remaining=%d)", entity_id, rl)

    # ----- 11. print confirmation --------------------------------------
    status = parsed["initial_status"]
    print(f"Captured: {entity_id} ({parsed['type']}, {status}, relevance={parsed['relevance']})")
    print(f"  source: {source}")
    print(f"  github_repo: {parsed['github_repo']}")
    print(f"  tags: {', '.join(parsed['tags'])}")
    if source == "github" and snapshot is not None:
        print(
            f"  signal: stars={snapshot.stars} forks={snapshot.forks} "
            f"commits_30d={snapshot.commits_30d} open_issues={snapshot.open_issues}"
        )
    if backfill_points:
        print(f"  backfilled: {backfill_points} star history points")

    return entity_id


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def capture(
    input: str = typer.Argument(..., help="GitHub URL, generic URL, or free text."),
    force: bool = typer.Option(False, "--force", help="Bypass exact + soft dedup checks."),
    update: str | None = typer.Option(
        None, "--update", metavar="ID",
        help="Append a history line to an existing entity instead of creating new.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Assemble the prompt and frontmatter but don't call the LLM or write to the vault.",
    ),
) -> None:
    """Capture an entity from a URL or free-text note."""
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

    if update is not None:
        parsed_repo = parse_repo_url(input) if _is_url(input) or "/" in input else None
        if parsed_repo is not None and not _is_url(input):
            if " " in input or input.count("/") != 1:
                parsed_repo = None
        if not entity_exists(update):
            print(f"ERROR: entity {update!r} does not exist.", file=sys.stderr)
            raise typer.Exit(code=1)
        if parsed_repo:
            note = f"Re-captured from github.com/{parsed_repo[0]}/{parsed_repo[1]}."
        elif _is_url(input):
            note = f"Re-captured from {input}."
        else:
            preview = input.strip().replace("\n", " ")
            note = f"Re-captured: {preview[:80]}{'...' if len(preview) > 80 else ''}"
        append_history(update, note, entry_date=today)
        print(f"Updated {update} with new history entry.")
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
