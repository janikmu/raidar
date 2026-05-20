<div align="center">
  <img width="600" height="198" alt="Gemini_Generated_Image_ad00odad00odad00" src="https://github.com/user-attachments/assets/4efa0537-770a-4c79-9e6d-26037fe7b848" />
</div>
  
# AI Radar — Tool Repo

Personal AI tooling intelligence system. Captures, enriches, and surfaces
knowledge about the AI tooling landscape from URLs, notes, and scheduled
GitHub signal pulls.

This repo is the **tool**. The knowledge vault is a separate private repo
(default: `~/raidar-vault`), configured in `config.yaml`.

The vault uses a **two-layer knowledge model**:
1. **Concepts**: Intellectual ideas (e.g., `multi-agent-frameworks`) with a lifecycle status (`emerging`, `watch`, `invest`, `common`, `superseded`, `abandoned`).
2. **Artifacts**: Evidence pieces (GitHub repos, papers, blog posts) mapped to a concept, with an evaluation status (`new`, `promising`, `recommended`, `deprecated`, `hype`).

## Architecture in one breath

- **Three jobs**: `capture` (on-demand), `enrich` (Sunday 20:00 via launchd),
  `digest` (Sunday 21:00 via launchd). Each is `python -m jobs.<name>`.
- **One CLI for querying**: `python -m jobs.search` with subcommands
  (`keyword | semantic | concept | artifact | signals | digest | list-concepts | list-artifacts | pending`).
- **One LLM router** (`lib/llm.py`) routes per-task to a configured chain of
  OpenAI-wire-compatible providers (academic proxy → local LMStudio fallback).
- **Local embeddings** via LMStudio (any OpenAI-compatible embedding model), flat JSON indexes, numpy cosine.
- **No databases**. Markdown + JSONL + JSON, all git-friendly.

## Quickstart

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure secrets

```bash
cp .env.example .env
# Fill in:
#   ACADEMIC_API_KEY   — your academic OpenAI-compatible proxy key
#   ACADEMIC_BASE_URL  — proxy base URL (e.g. https://proxy.host/v1)
#   GITHUB_PAT         — GitHub personal access token (read:public_repo scope)
#   LMSTUDIO_BASE_URL  — defaults to http://localhost:1234/v1 if unset
```

### 3. Start LMStudio

LMStudio serves both the chat fallback and the embedding model on the same
OpenAI-compatible endpoint (port 1234), routing by the `model` field in each
request. Load both models in the LMStudio UI:

- **Chat model** — e.g. `qwen/qwen3.5-9b` (or any model that fits your RAM
  budget). Update `providers.local-lmstudio.model` in [config.yaml](config.yaml)
  to match the id LMStudio shows.
- **Embedding model** — e.g. `text-embedding-nomic-embed-text-v1.5`. Update
  `embeddings.model` in [config.yaml](config.yaml) similarly.

Then start the server (LMStudio UI: Developer → Start Server) so it listens
on `http://localhost:1234/v1`. Both models stay loaded concurrently — on a
32 GB M5 with 50% allocation, qwen3.5-9b (~6 GB) + nomic-embed-text (~100 MB)
fit comfortably.

### 4. Verify the install

```bash
bash infra/smoke.sh
```

You want to see "0 fail". Skips are fine if LMStudio isn't running yet — re-run
once the server is up with both models loaded to exercise the full path.

### 5. Capture your first artifact

```bash
uv run python -m jobs.capture https://github.com/some-org/some-tool
uv run python -m jobs.search concept <concept-id-the-capture-printed>
uv run python -m jobs.search artifact <artifact-id-the-capture-printed>
```

### 6. Install scheduled jobs

```bash
./infra/install_launchd.sh
launchctl list | grep airadar     # both com.airadar.enrich and com.airadar.digest should appear
```

Next enrich fires Sunday 20:00 local time; digest at 21:00. To uninstall:
`./infra/install_launchd.sh uninstall`.

### 7. Connect Cowork

Open this directory as a Claude Cowork project with filesystem access and shell
execution enabled. Cowork reads [SKILL.md](SKILL.md) automatically and learns
the CLI surface. Then paste a URL into chat — Claude will run `capture` and
report back.

## Layout

```
ai-radar-tool/                    (this repo)
  jobs/
    capture.py                    on-demand capture (URL or text -> artifact + concept)
    bulk_capture.py               bulk capture from awesome-lists and newsletter pages
    enrich.py                     weekly signal refresh + LLM re-evaluation (two passes)
    digest.py                     weekly markdown digest
    backfill.py                   bulk star-history backfill for artifacts
    reevaluate.py                 force re-evaluation of artifacts with full signal history
    seed.py                       seed canonical concepts from training-data knowledge
    search.py                     CLI: query concepts, artifacts, signals, digests
    cli.py                        unified `raidar` entry point
  lib/
    vault.py                      atomic file I/O for concepts, artifacts, signals, digests
    llm.py                        OpenAI-compatible router with per-task chains + retry
    embeddings.py                 Ollama embeddings + split numpy indexes
    github.py                     GitHub API client (httpx + tenacity)
    body.py                       canonical body renderer/parser for concepts and artifacts
    config.py                     config.yaml loader
    secrets.py                    .env / env-var access
    logging_setup.py              logging configured once per process
  infra/
    launchd/com.airadar.{enrich,digest}.plist   templates (placeholders substituted on install)
    install_launchd.sh            install / uninstall
    smoke.sh                      offline acceptance test
  config.yaml                     paths, provider configs, task chains, thresholds
  context.md                      personal relevance anchor (edit freely)
  SKILL.md                        Cowork integration instructions
  .env                            local secrets (gitignored)

ai-radar-vault/                   (separate private repo at ~/raidar-vault)
  concepts/<id>.md                YAML frontmatter + markdown body
  artifacts/<id>.md               YAML frontmatter + markdown body
  signals/<id>.jsonl              append-only weekly snapshots, one JSON per line
  digests/YYYY-MM-DD.md           weekly digests
  embeddings/concepts.json        embedding index for concepts
  embeddings/artifacts.json       embedding index for artifacts
```

## How the LLM router routes

`config.yaml` maps task names to ordered provider chains:

```yaml
tasks:
  classification:    [academic-mistral, local-lmstudio]
  enrichment:        [academic-gpt-oss, academic-mistral, local-lmstudio]
  digest:            [academic-gpt-oss, academic-mistral]
```

`Router.generate(task=...)` tries them in order; transient failures retry
per-provider (exponential backoff via tenacity), terminal failures fall
through to the next provider. Running out of providers raises
`AllProvidersFailed`. Adding a new backend means adding one entry under
`providers:` and listing it in the relevant `tasks:` chains — no code changes.

## Editing the vault by hand

Concept and Artifact files are plain markdown. You can edit `## Current assessment` directly,
or flip `status:` / `evaluation:` in the frontmatter, or remove a tag — the tool reads from
disk on every call, so changes take effect immediately. `signals/` and
`embeddings/` are tool-owned; don't edit those by hand.

## Out of scope

Email capture, arXiv ingest, social-media scraping, web UI, multi-user. See
the original spec for rationale.
