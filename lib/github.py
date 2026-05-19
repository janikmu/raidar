"""GitHub API client for AI Radar capture + enrich jobs.

Thin layer over the GitHub REST API. Returns a fixed snapshot shape
(`RepoSnapshot`) plus a separate README fetch. Works unauthenticated
(60 req/hour) but warns once on module use; with a PAT in
`Config.github_token` you get 5000 req/hour.

Network calls are wrapped with tenacity (retry on transient network
errors and 5xx). 404 -> None. Non-rate-limit 403 -> ValueError (auth).
Rate-limit 403 -> TerminalError (caller decides; could be hours away
from reset, so we don't sleep-and-retry).
"""

from __future__ import annotations

import base64
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from lib import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoSnapshot:
    """A point-in-time snapshot of GitHub repo metadata.

    Field notes:
      * `open_issues` mirrors GitHub's `open_issues_count`, which *includes*
        open pull requests. We accept this conflation for V1 — splitting
        would cost an extra search API call per repo.
      * `commits_30d` counts commits on the default branch in the window
        `config.github_commit_window_days` (default 30).
      * `fetched_at` is when we made the call (UTC, second precision),
        not a GitHub-provided timestamp.
    """

    owner: str
    name: str
    stars: int
    forks: int
    open_issues: int
    commits_30d: int
    description: str | None
    default_branch: str
    archived: bool
    language: str | None
    license_spdx: str | None
    pushed_at: str
    created_at: str
    fetched_at: str


class TerminalError(RuntimeError):
    """Raised when retries exhaust on a transient error, or when GitHub
    has rate-limited us (caller decides whether to wait or skip)."""


class _RetryableHTTPError(Exception):
    """Internal sentinel for 5xx responses so tenacity can retry them
    via `retry_if_exception_type`. Not part of the public API."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Module-level state: lazy client + one-time unauth warning
# ---------------------------------------------------------------------------

_client: httpx.Client | None = None
_client_lock = threading.Lock()
_unauth_warned = False
_rate_limit_remaining: int | None = None


def _get_client() -> httpx.Client:
    global _client, _unauth_warned
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        cfg = config.load()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ai-radar-tool/0.1",
        }
        if cfg.github_token:
            headers["Authorization"] = f"Bearer {cfg.github_token}"
        elif not _unauth_warned:
            logger.warning(
                "GITHUB_PAT not set; using unauthenticated GitHub API "
                "(60 req/hour). Set GITHUB_PAT in .env for 5000 req/hour."
            )
            _unauth_warned = True
        _client = httpx.Client(
            base_url=cfg.github_api_base,
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
        )
        return _client


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

# owner/name segment chars per GitHub rules (lenient: we don't enforce
# every micro-rule, just the obvious ones).
_OWNER_RE = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
_NAME_RE = r"[A-Za-z0-9._-]+"
_SLUG_RE = re.compile(rf"^({_OWNER_RE})/({_NAME_RE})$")


def parse_repo_url(url_or_slug: str) -> tuple[str, str] | None:
    """Extract (owner, name) from a variety of GitHub references.

    Accepts:
      - https://github.com/owner/name
      - http://github.com/owner/name/
      - github.com/owner/name
      - owner/name
      - https://github.com/owner/name.git
      - https://github.com/owner/name/tree/main/...

    Returns None for anything that isn't a recognizable GitHub reference.
    """
    if not url_or_slug:
        return None
    raw = url_or_slug.strip()

    # Bare slug case (no scheme, no host, just owner/name).
    if "://" not in raw and not raw.lower().startswith("github.com"):
        # Strip a possible trailing slash to be lenient.
        candidate = raw.rstrip("/")
        # Strip .git suffix if present.
        if candidate.endswith(".git"):
            candidate = candidate[: -len(".git")]
        m = _SLUG_RE.match(candidate)
        if m:
            return m.group(1), m.group(2)
        return None

    # Add scheme if missing so urlparse extracts host correctly.
    if raw.lower().startswith("github.com"):
        raw = "https://" + raw

    try:
        parsed = urlparse(raw)
    except ValueError:
        return None

    host = (parsed.netloc or "").lower()
    if host not in {"github.com", "www.github.com"}:
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None

    owner, name = parts[0], parts[1]
    if name.endswith(".git"):
        name = name[: -len(".git")]

    # Validate against the same character rules as the bare-slug branch.
    if not _SLUG_RE.match(f"{owner}/{name}"):
        return None

    return owner, name


# ---------------------------------------------------------------------------
# HTTP layer (retried)
# ---------------------------------------------------------------------------


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=1.0, max=30.0),
    retry=retry_if_exception_type(
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.RemoteProtocolError,
            _RetryableHTTPError,
        )
    ),
)
def _request(method: str, path: str, **kwargs) -> httpx.Response:
    """Issue a single GitHub API request with retry on transient failures.

    Retried: connection/read/write timeouts, RemoteProtocolError, and 5xx
    responses (via the internal `_RetryableHTTPError` sentinel).

    Not retried: 4xx (including 404, 422, 403). Caller handles those.
    """
    global _rate_limit_remaining
    client = _get_client()
    resp = client.request(method, path, **kwargs)
    if 500 <= resp.status_code < 600:
        raise _RetryableHTTPError(
            resp.status_code, f"{method} {path} -> {resp.status_code}"
        )
    rl = resp.headers.get("X-RateLimit-Remaining")
    if rl:
        try:
            _rate_limit_remaining = int(rl)
        except ValueError:
            pass
    return resp


def get_rate_limit_remaining() -> int | None:
    """Return the last observed X-RateLimit-Remaining value from any GitHub response."""
    return _rate_limit_remaining


def _check_403(resp: httpx.Response, context: str) -> None:
    """Raise the appropriate error for a 403 response.

    Rate-limit (X-RateLimit-Remaining == 0): TerminalError with reset.
    Anything else (bad token, SSO, abuse detection): ValueError.
    """
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining == "0":
        reset_epoch = resp.headers.get("X-RateLimit-Reset", "")
        try:
            reset_iso = datetime.fromtimestamp(
                int(reset_epoch), tz=timezone.utc
            ).isoformat()
        except (TypeError, ValueError):
            reset_iso = reset_epoch or "unknown"
        logger.error(
            "GitHub rate-limited on %s; resets at %s (epoch=%s)",
            context,
            reset_iso,
            reset_epoch,
        )
        raise TerminalError(f"rate limited until {reset_iso}")
    # Non-rate-limit 403 -> auth problem.
    logger.error("GitHub 403 on %s (not rate-limit): %s", context, resp.text[:200])
    raise ValueError(
        f"GitHub returned 403 on {context} (not rate-limit; check token/permissions)"
    )


# ---------------------------------------------------------------------------
# Public fetchers
# ---------------------------------------------------------------------------


def fetch_repo(owner: str, name: str) -> RepoSnapshot | None:
    """Fetch a `RepoSnapshot` for `owner/name`.

    Returns None if the repo doesn't exist (404). Raises ValueError on
    a non-rate-limit 403. Raises TerminalError after retries on transient
    failures or when GitHub has rate-limited us.
    """
    path = f"/repos/{owner}/{name}"
    try:
        resp = _request("GET", path)
    except _RetryableHTTPError as e:
        raise TerminalError(f"5xx after retries on {path}: {e}") from e

    if resp.status_code == 404:
        return None
    if resp.status_code == 403:
        _check_403(resp, path)
    if resp.status_code >= 400:
        raise TerminalError(f"{path} -> {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    default_branch = data.get("default_branch") or "main"
    commits_30d = _count_recent_commits(owner, name, default_branch)

    license_block = data.get("license") or {}
    license_spdx = license_block.get("spdx_id") if isinstance(license_block, dict) else None
    # GitHub returns "NOASSERTION" for repos with a LICENSE file that doesn't match
    # a known template. Treat that the same as missing for our purposes.
    if license_spdx in (None, "", "NOASSERTION"):
        license_spdx = None

    fetched_at = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()

    return RepoSnapshot(
        owner=data.get("owner", {}).get("login", owner),
        name=data.get("name", name),
        stars=int(data.get("stargazers_count", 0)),
        forks=int(data.get("forks_count", 0)),
        open_issues=int(data.get("open_issues_count", 0)),
        commits_30d=commits_30d,
        description=data.get("description"),
        default_branch=default_branch,
        archived=bool(data.get("archived", False)),
        language=data.get("language"),
        license_spdx=license_spdx,
        pushed_at=data.get("pushed_at", ""),
        created_at=data.get("created_at", ""),
        fetched_at=fetched_at,
    )


def _count_recent_commits(owner: str, name: str, default_branch: str) -> int:
    """Count commits on the default branch in the last N days.

    Cheapest correct approach: GET /commits?since=...&per_page=1 and read
    the `page=N` query param from the `Link: rel="last"` header. If there
    is no Link header, the count is just the number of entries returned
    (0 or 1). We pin `sha=default_branch` so this can't be confused by a
    repo whose default branch is something other than `main`.
    """
    cfg = config.load()
    since_dt = datetime.now(tz=timezone.utc) - timedelta(
        days=cfg.github_commit_window_days
    )
    since_iso = since_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    path = f"/repos/{owner}/{name}/commits"
    params = {"since": since_iso, "per_page": "1", "sha": default_branch}
    try:
        resp = _request("GET", path, params=params)
    except _RetryableHTTPError as e:
        raise TerminalError(f"5xx after retries on {path}: {e}") from e

    if resp.status_code == 404:
        # Empty repo or vanished branch -> treat as zero recent commits.
        return 0
    if resp.status_code == 409:
        # GitHub returns 409 "Git Repository is empty" for new repos.
        return 0
    if resp.status_code == 403:
        _check_403(resp, path)
    if resp.status_code >= 400:
        raise TerminalError(f"{path} -> {resp.status_code}: {resp.text[:200]}")

    link = resp.headers.get("Link")
    if link:
        last_page = _extract_last_page(link)
        if last_page is not None:
            return last_page

    # No Link header => fewer than per_page+1 results; count the body.
    try:
        body = resp.json()
    except ValueError:
        return 0
    return len(body) if isinstance(body, list) else 0


_LINK_REL_RE = re.compile(r'<([^>]+)>;\s*rel="([^"]+)"')


def _extract_last_page(link_header: str) -> int | None:
    """Pull the integer `page=` param out of the rel="last" entry."""
    for url, rel in _LINK_REL_RE.findall(link_header):
        if rel != "last":
            continue
        try:
            qs = parse_qs(urlparse(url).query)
        except ValueError:
            return None
        page_vals = qs.get("page")
        if not page_vals:
            return None
        try:
            return int(page_vals[0])
        except ValueError:
            return None
    return None


def fetch_readme(owner: str, name: str) -> str | None:
    """Fetch README text for `owner/name`, decoded as UTF-8.

    Returns None on 404 or missing README. Raises ValueError on a
    non-rate-limit 403, TerminalError after retries on transient
    failures.
    """
    path = f"/repos/{owner}/{name}/readme"
    try:
        resp = _request("GET", path)
    except _RetryableHTTPError as e:
        raise TerminalError(f"5xx after retries on {path}: {e}") from e

    if resp.status_code == 404:
        return None
    if resp.status_code == 403:
        _check_403(resp, path)
    if resp.status_code >= 400:
        raise TerminalError(f"{path} -> {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    encoding = data.get("encoding")
    content = data.get("content", "")
    if encoding == "base64":
        try:
            raw = base64.b64decode(content)
        except (ValueError, TypeError):
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")
    # Some responses (rare) may not be base64-wrapped.
    if isinstance(content, str) and content:
        return content
    return None


# ---------------------------------------------------------------------------
# Star history
# ---------------------------------------------------------------------------


_STAR_PAGE_LIMIT = 400  # GitHub returns 422 for pages beyond this regardless of per_page


def fetch_star_history(
    owner: str,
    name: str,
    total_stars: int,
    *,
    n_samples: int = 12,
) -> list[tuple[int, str]]:
    """Sample the stargazer timeline. Returns [(star_count, starred_at_iso), ...].

    Uses adaptive per_page so small repos (e.g. 119 stars) still get n_samples
    data points rather than only 1-2. For large repos (>n_samples*100 stars)
    per_page=100 is used and pages are spread evenly, capped at the GitHub
    limit of 400 pages beyond which the API returns 422.
    """
    if total_stars <= 0:
        return []

    # Adaptive per_page: for small repos use finer granularity so we get
    # n_samples temporal points instead of just ceil(stars/100).
    per_page = max(1, min(100, total_stars // n_samples))
    total_pages = min((total_stars + per_page - 1) // per_page, _STAR_PAGE_LIMIT)

    if total_pages <= n_samples:
        pages = list(range(1, total_pages + 1))
    else:
        pages_set: set[int] = set()
        for i in range(n_samples):
            pages_set.add(round(1 + i * (total_pages - 1) / (n_samples - 1)))
        pages = sorted(pages_set)

    path = f"/repos/{owner}/{name}/stargazers"
    star_accept = {"Accept": "application/vnd.github.star+json"}
    result: list[tuple[int, str]] = []

    for page in pages:
        star_count = min(page * per_page, total_stars)
        try:
            resp = _request(
                "GET", path,
                params={"per_page": str(per_page), "page": str(page)},
                headers=star_accept,
            )
        except Exception:
            logger.warning("star_history: error on page %d for %s/%s", page, owner, name)
            continue
        if resp.status_code != 200:
            logger.warning(
                "star_history: HTTP %d page %d for %s/%s",
                resp.status_code, page, owner, name,
            )
            continue
        entries = resp.json()
        if not entries or not isinstance(entries, list):
            continue
        starred_at = entries[-1].get("starred_at", "")
        if starred_at:
            result.append((star_count, starred_at))

    return sorted(result, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from lib import logging_setup

    logging_setup.setup(level="INFO")

    cfg = config.load()
    print(f"PAT present: {bool(cfg.github_token)}")

    # 1) parse_repo_url round-trip
    cases = [
        "https://github.com/anthropics/claude-code.git",
        "anthropics/claude-code",
        "https://github.com/anthropics/claude-code/tree/main/docs",
        "github.com/anthropics/claude-code/",
        "not a url",
        "https://gitlab.com/x/y",
    ]
    for c in cases:
        print(f"  parse_repo_url({c!r}) -> {parse_repo_url(c)}")

    # 2) Fetch a well-known repo. Try anthropics/claude-code first,
    # fall back to octocat/Hello-World if that 404s.
    primary = ("anthropics", "claude-code")
    fallback = ("octocat", "Hello-World")
    snap = fetch_repo(*primary)
    used = primary
    if snap is None:
        print(f"  {primary[0]}/{primary[1]} 404; falling back to octocat/Hello-World")
        snap = fetch_repo(*fallback)
        used = fallback
    if snap is None:
        raise SystemExit("both primary and fallback returned 404")
    print("snapshot:")
    for field_name in (
        "owner",
        "name",
        "stars",
        "forks",
        "open_issues",
        "commits_30d",
        "description",
        "default_branch",
        "archived",
        "language",
        "license_spdx",
        "pushed_at",
        "created_at",
        "fetched_at",
    ):
        print(f"  {field_name}: {getattr(snap, field_name)!r}")

    # 3) README preview
    readme = fetch_readme(*used)
    if readme is None:
        print("readme: <none>")
    else:
        preview = readme[:200].replace("\n", " ")
        print(f"readme[0:200]: {preview!r}")

    # 4) Known-404
    missing = fetch_repo("definitely-not-a-real-org-xyz", "nope")
    print(f"missing-repo fetch -> {missing!r}")
    assert missing is None, "expected None for missing repo"

    print("github.py smoke test OK")
