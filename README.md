# AI Radar — Tool Repo

Personal AI tooling intelligence system. Captures, enriches, and surfaces
knowledge about the AI tooling landscape.

This repo is the **tool**. The knowledge vault is a separate private repo
at `~/source/ai-radar-vault` (configured via `config.yaml`).

## Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Configure secrets
cp .env.example .env
# Fill in ACADEMIC_API_KEY, ACADEMIC_BASE_URL, GITHUB_PAT, etc.

# 3. Confirm config
uv run python -c "from lib import config; print(config.load().vault_path)"

# 4. Run a capture (once Task 5 lands)
uv run python -m jobs.capture https://github.com/some/repo
```

## Structure

```
ai-radar-tool/
  jobs/         on-demand and scheduled jobs (capture, enrich, digest, search)
  lib/          shared libraries (vault I/O, LLM router, embeddings, GitHub, config, secrets, logging)
  infra/        launchd plists + install script
  config.yaml   paths, provider configs, task→provider routing, thresholds
  context.md    personal relevance anchor, read by capture/enrich
  .env          local secrets (gitignored)
```

## Build status

| Task | Module | Status |
| --- | --- | --- |
| 0 | scaffold | done |
| 1 | `lib/vault.py` | todo |
| 2 | `lib/llm.py` | todo |
| 3 | `lib/embeddings.py` | todo |
| 4 | `lib/github.py` | todo |
| 5 | `jobs/capture.py` | todo |
| 6 | `jobs/enrich.py` | todo |
| 7 | `jobs/digest.py` | todo |
| 8 | `jobs/search.py` | todo |
| 9 | launchd + Cowork `SKILL.md` | todo |
| 10 | smoke test + setup docs | todo |
