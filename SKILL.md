# AI Radar — Cowork Skill

You are operating as the query and capture interface for the AI Radar knowledge
system. The tool repo lives at `~/source/ai-knowledge`; the vault at
`~/raidar-vault`.

The vault uses a two-layer model:
1. **Concepts**: Intellectual ideas (e.g. `multi-agent-frameworks`) with a lifecycle status (`emerging`, `watch`, `invest`, `common`, `superseded`, `abandoned`).
2. **Artifacts**: Evidence pieces (GitHub repos, papers, blog posts) mapped to a concept, with an evaluation status (`new`, `promising`, `recommended`, `deprecated`, `hype`).

Operating rules:

- **Always prefer tool output over your own training knowledge** when discussing
  specific tracked items. The vault is the source of truth for what the user
  is currently watching.
- Run commands from the tool repo (`cd ~/source/ai-knowledge` if you aren't
  already). All commands use `uv run` so the project's venv is picked up.
- When the user pastes a URL or a note, **call `capture`** rather than just
  describing what would happen. Capture is idempotent (exact dedup by GitHub
  repo URL, soft warning on semantic similarity) and creates/updates both the
  artifact and its concept.
- After any capture or `enrich` runs, you can re-read the affected concept or
  artifact file directly with `python -m jobs.search concept <id>` to discuss
  what was written.

## Commands

### Capture — add or update an artifact/concept

```bash
uv run python -m jobs.capture "<url-or-text>"
uv run python -m jobs.capture --force "<input>"        # bypass dedup warning
uv run python -m jobs.capture --dry-run "<input>"      # preview, no writes
```

Accepts: a GitHub URL, any other URL (article/blog post — extracted via
trafilatura), or free-form text. The LLM automatically classifies the artifact
and maps it to an existing concept (or creates a new one).

### Search — query the vault

```bash
uv run python -m jobs.search keyword "<query>"             # frontmatter substring match (both layers)
uv run python -m jobs.search semantic "<query>"            # embedding-based (both layers, top-5)
uv run python -m jobs.search concept <id>                  # full concept prose + artifact summary
uv run python -m jobs.search artifact <id>                 # full artifact prose
uv run python -m jobs.search signals <id>                  # JSONL signal history with deltas
uv run python -m jobs.search digest --last 14              # recent digest markdown
uv run python -m jobs.search list-concepts --status watch  # table of concepts by status
uv run python -m jobs.search list-artifacts --evaluation recommended
uv run python -m jobs.search pending                       # list concepts flagged for review
```

Choose by intent:
- **keyword** — known id, tag, or status. Fast, precise, no LLM/embedding cost.
- **semantic** — conceptual question ("what are people using for agent memory").
- **concept** — once you know a concept id and want its full summary and artifact roster.
- **artifact** — to drill down into a specific tool's rationale.
- **signals** — when the user asks about momentum, deltas, or "what changed" for an artifact.
- **digest** — when the user asks "what happened this week" or wants a recap.
- **list-concepts / list-artifacts** — when the user wants a roster.
- **pending** — when checking if any concepts need manual curation.

## Maintenance commands (less common in chat)

```bash
uv run python -m jobs.enrich --only <id> --dry-run   # re-evaluate one artifact without writes
uv run python -m jobs.enrich --concepts-only         # manually run the concept re-evaluation pass
uv run python -m jobs.digest --dry-run               # preview today's digest without writing
```

The full `enrich` and `digest` runs happen automatically via launchd on Sundays
(20:00 and 21:00). Do not invoke them without `--dry-run` from a chat session
unless the user explicitly asks for a manual run.

## Output conventions

Every command writes plain text or markdown to stdout. Errors go to stderr
with non-zero exit codes:

- `0` — success (including the "no matches" case)
- `1` — expected error (id not found, bad input, LLM not configured)
- `2` — infrastructure unavailable (e.g. LMStudio not running for semantic search)

Treat exit code 2 as recoverable infrastructure: tell the user "the embedding
backend isn't running" and offer to use `keyword` instead.

## Reading the vault directly

When you need raw access (rare), the vault layout is:

```
~/raidar-vault/
  concepts/<id>.md         # frontmatter + body
  artifacts/<id>.md        # frontmatter + body
  signals/<id>.jsonl       # one snapshot per line, append-only
  digests/YYYY-MM-DD.md
```

You may `cat` a concept or artifact file directly when discussing it in detail. Don't
write to vault files by hand — go through `capture`, `enrich`, or `digest`.
