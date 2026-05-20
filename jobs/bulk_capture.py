"""Bulk-capture repos from an awesome-list or newsletter page.

Extracts all github.com/owner/repo links from the target, skips already-tracked
repos silently, and captures the rest using the same pipeline as jobs.capture.

GitHub awesome-list URLs are fetched via the GitHub API (README quality >
scraping). All other URLs are fetched with trafilatura.

Usage:
    raidar bulk-capture <url>
    raidar bulk-capture --dry-run <url>
    raidar bulk-capture --limit 20 <url>
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import date
from typing import Annotated

import trafilatura
import typer

from lib import config
from lib.github import fetch_readme, parse_repo_url
from lib.logging_setup import setup as setup_logging
from jobs.capture import CaptureSkipped, _capture_one

log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help=__doc__)

# Matches any github.com/owner/repo path (with or without scheme).
_GH_URL_RE = re.compile(
    r"(?:https?://)?github\.com/([A-Za-z0-9][A-Za-z0-9\-]{0,38}/[A-Za-z0-9_.\-]+)"
)


def _extract_github_slugs(text: str) -> list[str]:
    """Return deduplicated owner/name slugs found in text, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _GH_URL_RE.finditer(text):
        raw = "https://github.com/" + m.group(1)
        parsed = parse_repo_url(raw)
        if parsed is None:
            continue
        slug = f"{parsed[0]}/{parsed[1]}"
        # Skip obvious non-repo paths that still match the regex.
        if parsed[1].lower() in {"features", "pricing", "about", "login", "topics",
                                   "marketplace", "sponsors", "issues", "pulls",
                                   "settings", "notifications", "explore"}:
            continue
        if slug not in seen:
            seen.add(slug)
            result.append(slug)
    return result


def _fetch_source_text(url: str) -> str | None:
    """Return raw text for slug extraction.

    For GitHub repo URLs, uses the README (higher quality than HTML scrape).
    For everything else, uses trafilatura.
    """
    parsed = parse_repo_url(url)
    if parsed is not None:
        owner, name = parsed
        log.info("fetching README for %s/%s to extract links", owner, name)
        readme = fetch_readme(owner, name)
        if readme:
            return readme
        log.warning("no README for %s/%s; falling back to trafilatura", owner, name)

    log.info("fetching %s with trafilatura", url)
    raw = trafilatura.fetch_url(url)
    if not raw:
        return None
    return trafilatura.extract(raw, include_links=True) or trafilatura.extract(raw)


@app.command()
def bulk(
    url: Annotated[str, typer.Argument(help="GitHub awesome-list URL or any web page.")],
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be captured without writing."),
    limit: int | None = typer.Option(None, "--limit", help="Max repos to process (default: all)."),
    force: bool = typer.Option(False, "--force", help="Bypass soft dedup on each item."),
) -> None:
    """Bulk-capture repos extracted from an awesome-list or web page."""
    cfg = config.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)
    today = date.today().isoformat()

    text = _fetch_source_text(url)
    if not text:
        print(f"ERROR: could not fetch content from {url}", file=sys.stderr)
        raise typer.Exit(code=1)

    slugs = _extract_github_slugs(text)
    if not slugs:
        print("No GitHub repo links found in the page.")
        raise typer.Exit(code=0)

    if limit is not None:
        slugs = slugs[:limit]

    print(f"Found {len(slugs)} GitHub repos. {'(dry run)' if dry_run else ''}")

    captured = skipped = failed = 0
    for slug in slugs:
        gh_url = f"https://github.com/{slug}"
        if dry_run:
            print(f"  would capture: {slug}")
            captured += 1
            continue
        try:
            artifact_id = _capture_one(gh_url, cfg, force=force, today=today)
            log.info("bulk captured %s -> %s", slug, artifact_id)
            captured += 1
        except CaptureSkipped as exc:
            log.debug("bulk skip %s: %s", slug, exc)
            skipped += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {slug}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {captured} captured, {skipped} skipped (dedup), {failed} failed.")
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
