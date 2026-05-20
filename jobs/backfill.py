"""Backfill star history for tracked GitHub repos.

Fetches sampled stargazer timestamps from the GitHub API and writes them as
source="backfill" signal entries against each artifact id. This gives the
enrich job historical context for repos that were added to the radar long
after they first gained traction.

The enrich job also auto-backfills on first run per artifact; this standalone
script is useful for bulk backfilling your existing vault, or for re-running
with more samples.

Usage:
    raidar backfill                  # all artifacts with github_repo
    raidar backfill --only ID        # single artifact
    raidar backfill --dry-run        # show plan without writing
    raidar backfill --samples 20     # denser history (more API calls)
    raidar backfill --force          # re-backfill even if already done
    raidar backfill --min-rate 300   # abort if rate limit drops below N
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated

import typer

from lib import config, vault
from lib.github import (
    TerminalError,
    fetch_repo,
    fetch_star_history,
    get_rate_limit_remaining,
    parse_repo_url,
)
from lib.logging_setup import setup as setup_logging

log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help=__doc__)

_MIN_RATE_DEFAULT = 300


def _backfill_artifact(
    artifact: vault.Artifact,
    *,
    dry_run: bool = False,
    samples: int = 12,
    force: bool = False,
    min_rate: int = _MIN_RATE_DEFAULT,
) -> tuple[int, str]:
    """Backfill one artifact. Returns (signals_written, status_msg)."""
    github_repo = artifact.frontmatter.get("github_repo")
    if not github_repo:
        return 0, "skip: no github_repo"

    if not force and vault.has_star_history_backfill(artifact.id):
        return 0, "skip: already backfilled"

    parsed = parse_repo_url(str(github_repo))
    if parsed is None:
        return 0, f"skip: unparseable github_repo={github_repo!r}"
    owner, name = parsed

    rl = get_rate_limit_remaining()
    if rl is not None and rl <= min_rate:
        return 0, f"abort: rate_limit_remaining={rl} <= min_rate={min_rate}"

    try:
        snap = fetch_repo(owner, name)
    except TerminalError as exc:
        return 0, f"error: github terminal: {exc}"
    if snap is None:
        return 0, "skip: repo 404"
    if snap.stars == 0:
        return 0, "skip: 0 stars"

    history = fetch_star_history(owner, name, snap.stars, n_samples=samples)
    if not history:
        return 0, "skip: no star history returned"

    if not dry_run:
        for star_count, starred_at in history:
            vault.append_signal(artifact.id, {
                "date": starred_at[:10],
                "stars": star_count,
                "source": "backfill",
            })

    first_date = history[0][1][:10]
    last_date = history[-1][1][:10]
    return len(history), f"ok: {len(history)} points [{first_date} → {last_date}]"


@app.command()
def backfill(
    only: str | None = typer.Option(None, "--only", metavar="ID", help="Single artifact ID."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan without writing."),
    samples: Annotated[int, typer.Option("--samples", help="Stargazer pages to fetch per repo.")] = 12,
    force: bool = typer.Option(False, "--force", help="Re-backfill even if already done."),
    min_rate: Annotated[int, typer.Option("--min-rate", help="Abort if API calls remaining drop below this.")] = _MIN_RATE_DEFAULT,
) -> None:
    """Backfill star history for tracked GitHub repos."""
    cfg = config.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)

    if only:
        artifact = vault.read_artifact(only)
        if artifact is None:
            print(f"ERROR: artifact {only!r} not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        artifacts = [artifact]
    else:
        artifacts = vault.list_artifacts()

    done = skipped = errors = 0
    for artifact in artifacts:
        n, msg = _backfill_artifact(
            artifact, dry_run=dry_run, samples=samples, force=force, min_rate=min_rate,
        )
        prefix = "[dry-run] " if dry_run else ""
        if msg.startswith("abort"):
            print(f"{prefix}{artifact.id}: {msg}", file=sys.stderr)
            print(f"\nAborted after {done} backfilled, {skipped} skipped, {errors} errors.")
            raise typer.Exit(code=1)
        elif msg.startswith("error"):
            print(f"{prefix}{artifact.id}: {msg}", file=sys.stderr)
            errors += 1
        elif msg.startswith("skip"):
            log.debug("%s: %s", artifact.id, msg)
            skipped += 1
        else:
            print(f"{prefix}{artifact.id}: {msg}")
            done += 1

    print(f"\nDone: {done} backfilled, {skipped} skipped, {errors} errors.")
    if errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
