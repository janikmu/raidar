"""raidar — AI Radar command-line interface.

Usage:
    raidar init               [--vault PATH] [--seed]
    raidar seed               [--id ID]... [--list] [--dry-run] [--force]
    raidar capture <url>      [--force] [--update ID] [--dry-run] [--no-backfill]
    raidar bulk-capture <url> [--dry-run] [--limit N] [--force]
    raidar enrich             [--only ID] [--dry-run]
    raidar backfill           [--only ID] [--dry-run] [--samples N] [--force]
    raidar reevaluate         [--only ID] [--status S] [--dry-run]
    raidar digest             [--date YYYY-MM-DD] [--dry-run]
    raidar health             [--semantic] [--json] [--strict] [--threshold N]
    raidar merge-concept <source> <target>  [--dry-run]
    raidar rename-concept <old> <new>       [--label TEXT] [--dry-run]
    raidar reindex            [--layer concepts|artifacts|all] [--prune] [--dry-run]
    raidar search keyword <query>
    raidar search semantic <query>  [--top-k N]
    raidar search entity <id>
    raidar search signals <id>
    raidar search list        [--status S] [--type T]
    raidar search digest      [--last N]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

# ---------------------------------------------------------------------------
# Root app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="raidar",
    add_completion=False,
    help="raidar — personal AI tooling intelligence.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Typer-based commands
# ---------------------------------------------------------------------------

from jobs.capture import capture  # noqa: E402
from jobs.bulk_capture import bulk  # noqa: E402
from jobs.backfill import backfill  # noqa: E402
from jobs.reevaluate import reevaluate  # noqa: E402
from jobs.digest import main as _digest_cmd  # noqa: E402
from jobs.search import app as _search_app  # noqa: E402
from jobs.seed import seed as _seed_cmd  # noqa: E402
from jobs.enrich import enrich as _enrich_cmd  # noqa: E402
from jobs.health import health as _health_cmd  # noqa: E402
from jobs.merge import merge_concept as _merge_cmd  # noqa: E402
from jobs.rename import rename_concept as _rename_cmd  # noqa: E402
from jobs.reindex import reindex as _reindex_cmd  # noqa: E402

app.command("capture")(capture)
app.command("bulk-capture")(bulk)
app.command("backfill")(backfill)
app.command("reevaluate")(reevaluate)
app.command("digest")(_digest_cmd)
app.command("seed")(_seed_cmd)
app.command("enrich")(_enrich_cmd)
app.command("health")(_health_cmd)
app.command("merge-concept")(_merge_cmd)
app.command("rename-concept")(_rename_cmd)
app.command("reindex")(_reindex_cmd)
app.add_typer(_search_app, name="search")

# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

_CONTEXT_MD_TEMPLATE = """\
# Research context

<!--
  This file anchors the LLM's relevance judgments to your specific interests.
  Edit it to describe your research focus, what you track, and why.
  Keep it concise — it is prepended to every classification and evaluation prompt.
-->

I am an Information Systems researcher studying ...

## Focus areas

- ...
- ...

## What I track

I want to track tools, frameworks, and papers relevant to ...
I am NOT interested in ...

## Relevance notes

- High relevance: ...
- Medium relevance: ...
- Low relevance: ...
"""

_CONFIG_YAML_TEMPLATE = """\
# raidar — tool config.
# Paths can use ~ for home. Env-var references are resolved at runtime.

vault:
  path: {vault_path}

# Each provider is an OpenAI-compatible chat endpoint.
# base_url_env points at an env var (lets you swap proxy hosts without editing config).
# api_key_env: null means no auth header (used for local LMStudio/Ollama).
providers:
  academic-mistral:
    base_url_env: ACADEMIC_BASE_URL
    api_key_env: ACADEMIC_API_KEY
    model: "Mistral Small 4 119B 2603 KI:EZ"
    timeout_s: 60
  academic-gpt-oss:
    base_url_env: ACADEMIC_BASE_URL
    api_key_env: ACADEMIC_API_KEY
    model: "OpenAI GPT OSS 120B KI:Inferenz.nrw"
    timeout_s: 120
  local-lmstudio:
    base_url_env: LMSTUDIO_BASE_URL
    api_key_env: null
    # Set this to whatever model identifier LMStudio exposes for the loaded model.
    model: qwen/qwen3.5-9b
    timeout_s: 180
    # /no_think requests non-thinking mode for Qwen3. Some LMStudio versions
    # still route output through reasoning_content — the router handles this.
    # Drop for non-reasoning local models.
    no_think: true
    # Reasoning models need room for thinking tokens + response tokens.
    max_tokens: 16384

# Per-task provider chains. The router tries them in order; on terminal failure
# it falls through to the next.
tasks:
  classification:
    - academic-mistral
    - local-lmstudio
  enrichment:
    - academic-gpt-oss
    - academic-mistral
    - local-lmstudio
  reevaluation:
    - academic-gpt-oss
    - academic-mistral
    - local-lmstudio
  digest:
    - academic-gpt-oss
    - academic-mistral
    - local-lmstudio

# Embedding backend. LMStudio exposes /v1/embeddings when an embedding model is
# loaded. Load e.g. `text-embedding-nomic-embed-text-v1.5` and update `model`.
embeddings:
  base_url_env: LMSTUDIO_BASE_URL
  api_key_env: null
  model: text-embedding-nomic-embed-text-v1.5

github:
  api_base: https://api.github.com
  token_env: GITHUB_PAT
  commit_window_days: 30

thresholds:
  signal_change:
    abs_star_delta: 50
    rel_star_delta_pct: 20
    rel_commits_30d_pct: 50
  # Soft artifact dedup: capture warns/skips if a new artifact is this close to one already tracked.
  semantic_dedup: 0.92
  # Concept dedup gate: a proposed NEW concept this close to an existing one is
  # attached to the existing concept (flagged review_needed) instead of forked.
  # Tuned high because the local embedding model compresses agent-tooling concepts
  # (distinct ideas sit ~0.90; true duplicates ~0.92+). Lower it cautiously.
  concept_dedup: 0.93

retry:
  max_attempts: 4
  initial_backoff_s: 1.0
  max_backoff_s: 30.0

logging:
  level: INFO
  file: logs/raidar.log

paths:
  context: context.md
  enrich_output: logs/last_enrich.json
"""

_VAULT_GITIGNORE = """\
# Embeddings index is large and fully regeneratable — don't commit it.
embeddings/

# Run-state and execution logs — don't commit them.
logs/
"""

_VAULT_README = """\
# AI Radar vault

Personal AI tooling intelligence vault. Managed by [raidar](https://github.com/your/raidar).

## Structure

| Directory | Contents |
|-----------|----------|
| `concepts/` | One `.md` file per concept (the intellectual ideas being tracked) |
| `artifacts/` | One `.md` file per artifact (repos, papers, posts mapped to concepts) |
| `signals/` | Append-only `.jsonl` signal history per artifact |
| `digests/` | Weekly digest outputs |
| `embeddings/` | Local embeddings index — gitignored, regenerate with `raidar enrich` |
"""

_ENV_VARS: list[tuple[str, str, bool]] = [
    ("GITHUB_PAT",        "GitHub personal access token (for repo metadata + star history)", True),
    ("ACADEMIC_BASE_URL", "Academic proxy base URL (cloud LLM provider)",                   False),
    ("ACADEMIC_API_KEY",  "Academic proxy API key",                                          False),
    ("LMSTUDIO_BASE_URL", "LMStudio base URL, e.g. http://localhost:1234/v1 (local LLM)",   False),
]


def _check_mark(ok: bool) -> str:
    return "✓" if ok else "✗"


@app.command("init")
def init_cmd(
    vault: Optional[str] = typer.Option(
        None,
        "--vault",
        help="Vault directory path (default: ~/raidar-vault).",
    ),
    seed: bool = typer.Option(
        False,
        "--seed",
        help="After scaffolding, run `raidar seed` to seed canonical concepts "
             "(MCP, RAG, ReAct, ...) from training-data knowledge. Skipped if "
             "required env vars aren't set yet.",
    ),
) -> None:
    """Create the vault directory structure and scaffold the project config."""
    from lib import config as lib_config
    config_path = lib_config.get_config_path()
    vault_path = Path(vault).expanduser().resolve() if vault else Path("~/raidar-vault").expanduser()

    typer.echo(f"\nInitialising raidar")
    typer.echo(f"  config  : {config_path}")
    typer.echo(f"  vault   : {vault_path}\n")

    # ---- 1. Vault directories ------------------------------------------
    for subdir in ("concepts", "artifacts", "signals", "digests", "embeddings", "logs"):
        d = vault_path / subdir
        d.mkdir(parents=True, exist_ok=True)
        # Keep empty directories in git with a .gitkeep
        gitkeep = d / ".gitkeep"
        if not any(d.iterdir()):
            gitkeep.touch()
    typer.echo(f"  {_check_mark(True)} vault directories created")

    # ---- 2. Vault .gitignore -------------------------------------------
    gi = vault_path / ".gitignore"
    if not gi.exists():
        gi.write_text(_VAULT_GITIGNORE)
        typer.echo(f"  {_check_mark(True)} vault/.gitignore created")
    else:
        typer.echo(f"  · vault/.gitignore already exists — skipped")

    # ---- 3. Vault README -----------------------------------------------
    readme = vault_path / "README.md"
    if not readme.exists():
        readme.write_text(_VAULT_README)
        typer.echo(f"  {_check_mark(True)} vault/README.md created")
    else:
        typer.echo(f"  · vault/README.md already exists — skipped")

    # ---- 4. git init the vault -----------------------------------------
    git_dir = vault_path / ".git"
    if not git_dir.is_dir():
        result = subprocess.run(
            ["git", "init", str(vault_path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            typer.echo(f"  {_check_mark(True)} git init {vault_path}")
        else:
            typer.echo(f"  {_check_mark(False)} git init failed: {result.stderr.strip()}")
    else:
        typer.echo(f"  · vault is already a git repo — skipped")

    # ---- 5. context.md -------------------------------------------------
    context_path = vault_path / "context.md"
    if not context_path.exists():
        # Check if context.md exists in current working directory to copy it (migration safeguard)
        local_context = Path.cwd() / "context.md"
        if local_context.is_file():
            import shutil
            shutil.copy(local_context, context_path)
            typer.echo(f"  {_check_mark(True)} context.md copied from current directory to vault")
        else:
            context_path.write_text(_CONTEXT_MD_TEMPLATE)
            typer.echo(f"  {_check_mark(True)} context.md created in vault — edit this to anchor the LLM")
    else:
        typer.echo(f"  · context.md already exists in vault — skipped")

    # ---- 6. config.yaml ------------------------------------------------
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_CONFIG_YAML_TEMPLATE.format(vault_path=str(vault_path)))
        typer.echo(f"  {_check_mark(True)} config.yaml created at {config_path}")
    else:
        # Check if the vault path in config matches what was requested
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            configured_vault = Path(cfg.get("vault", {}).get("path", "")).expanduser()
            if configured_vault.resolve() != vault_path:
                typer.echo(
                    f"  ⚠  config.yaml exists but vault.path={cfg['vault']['path']!r} "
                    f"— update it to {vault_path} if needed"
                )
            else:
                typer.echo(f"  · config.yaml already exists at {config_path} and vault path matches — skipped")
        except Exception:
            typer.echo(f"  · config.yaml already exists at {config_path} — skipped")

    # ---- 7. .env check -------------------------------------------------
    env_path = config_path.parent / ".env"
    
    # Migration safeguard: copy local .env to global configuration folder if it exists
    local_env_path = Path.cwd() / ".env"
    if not env_path.exists() and local_env_path.is_file():
        import shutil
        shutil.copy(local_env_path, env_path)
        typer.echo(f"  {_check_mark(True)} .env copied from current directory to configuration folder")

    env_vars: dict[str, str] = {}
    if env_path.exists():
        # Parse without importing dotenv (avoid dependency at init time)
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip()

    typer.echo(f"\n  Environment variables (.env):")
    all_required_set = True
    for var, desc, required in _ENV_VARS:
        is_set = var in env_vars or __import__("os").environ.get(var)
        mark = _check_mark(bool(is_set))
        label = "(required)" if required else "(optional)"
        typer.echo(f"    {mark} {var:<25} {label}  {desc}")
        if required and not is_set:
            all_required_set = False

    if not env_path.exists():
        typer.echo(f"\n  No .env found — create one at {env_path} with the vars above.")

    # ---- 8. Optional seed ---------------------------------------------
    if seed:
        if not all_required_set:
            typer.echo("\n  --seed skipped: required env vars not set yet.")
            typer.echo("  Set them in .env, then run: raidar seed")
        else:
            typer.echo("\n  Seeding canonical concepts (raidar seed)…")
            _seed_cmd(ids=None, list_only=False, dry_run=False, force=False)

    # ---- 9. Summary ----------------------------------------------------
    typer.echo("")
    if all_required_set:
        typer.echo("  Ready. Try: raidar capture https://github.com/owner/repo")
    else:
        typer.echo("  Set the required env vars in .env, then try:")
        typer.echo("    raidar capture https://github.com/owner/repo")
    typer.echo("")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
