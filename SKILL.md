# AI Radar — Cowork Skill

You are operating as the query, capture, and management interface for the AI Radar knowledge system. The CLI is globally installed on the user's machine as a system-wide executable named `raidar`.

## Architecture & Paths

The CLI is completely stateless. Its paths are:
- **Global executable**: `raidar` (callable from any directory on the user's machine)
- **Active Configuration**: `~/.config/raidar/config.yaml` (stores vault paths and LLM task/provider mappings)
- **Global secrets**: `~/.config/raidar/.env` (contains API keys and URLs)
- **Knowledge Vault**: Located at the path specified in `config.yaml` (default: `~/raidar-vault`). All personal context, logs, and markdown files live here:
  - `{vault_path}/context.md` — your personal relevance anchor (describes research focus/interests)
  - `{vault_path}/logs/` — tool and automated launchd logs
  - `{vault_path}/concepts/` — intellectual concepts / YAML frontmatter + markdown
  - `{vault_path}/artifacts/` — evidence pieces (repos, papers, blog posts)
  - `{vault_path}/signals/` — automated weekly tracking snapshots (JSONL)
  - `{vault_path}/digests/` — weekly markdown digests

## Operating Rules

- **Always prefer tool output over your own training knowledge** when discussing specific tracked items. The vault is the source of truth for what the user is currently watching.
- **Run the global command directly**: You do NOT need to run `cd` or `uv run python -m jobs...`. Simply invoke `raidar` directly from whatever directory you are in.
- **Call capture on URL/note input**: When the user pastes a URL or a note, **call `raidar capture`** rather than just describing what would happen. Capture is idempotent (exact dedup by GitHub repo URL, soft warning on semantic similarity) and creates/updates both the artifact and its concept.
- **After any capture or enrich runs**, you can read the affected concept or artifact file directly with `raidar search concept <id>` or `raidar search artifact <id>` to discuss what was written.

## Commands

### Init — scaffold a new vault and config
```bash
raidar init --vault "~/raidar-vault"
```

### Capture — add or update an artifact/concept
```bash
raidar capture "<url-or-text>"
raidar capture --force "<input>"        # bypass dedup warning
raidar capture --dry-run "<input>"      # preview, no writes
```
Accepts a GitHub URL, any other web URL, or free-form text. The LLM automatically classifies the artifact, maps it to a concept (or creates a new one), and outputs the resulting IDs.

### Search — query the vault
```bash
raidar search keyword "<query>"             # frontmatter substring match (both layers)
raidar search semantic "<query>"            # embedding-based (both layers, top-5)
raidar search concept <id>                  # full concept prose + artifact summary
raidar search artifact <id>                 # full artifact prose
raidar search signals <id>                  # JSONL signal history with deltas
raidar search digest --last 14              # recent digest markdown
raidar search list-concepts --status watch  # table of concepts by status
raidar search list-artifacts --evaluation recommended
raidar search pending                       # list concepts flagged for review
```

Choose by intent:
- **keyword** — known ID, tag, or status. Fast, precise, no LLM/embedding cost.
- **semantic** — conceptual question ("what are people using for agent memory").
- **concept** — once you know a concept ID and want its full summary and artifact roster.
- **artifact** — to drill down into a specific tool's rationale.
- **signals** — when the user asks about momentum, deltas, or "what changed" for an artifact.
- **digest** — when the user asks "what happened this week" or wants a recap.
- **list-concepts / list-artifacts** — when the user wants a roster.
- **pending** — when checking if any concepts need manual curation.

### Maintenance & Jobs
```bash
raidar enrich --only <id> --dry-run   # re-evaluate one artifact without writes
raidar enrich --concepts-only         # manually run the concept re-evaluation pass
raidar digest --dry-run               # preview today's digest without writing
```
The full `enrich` and `digest` runs happen automatically via launchd on Sundays (20:00 and 21:00). Do not invoke them without `--dry-run` from a chat session unless the user explicitly asks for a manual run.

## Output Conventions & Troubleshooting

Every command writes plain text or markdown to stdout. Errors go to stderr with non-zero exit codes:
- `0` — success (including the "no matches" case)
- `1` — expected error (ID not found, bad input, LLM not configured)
- `2` — infrastructure unavailable (e.g. LMStudio not running for semantic search)

Treat exit code 2 as recoverable infrastructure: tell the user "the embedding backend isn't running" and offer to use `keyword` instead.

## Reading the Vault Directly

When you need raw file access (e.g., to read full markdown details beyond what search displays), you can view files in the vault directly:
- `cat ~/raidar-vault/concepts/<id>.md`
- `cat ~/raidar-vault/artifacts/<id>.md`
- `cat ~/raidar-vault/context.md`
Do NOT write or edit vault files by hand — always go through `raidar capture`, `raidar enrich`, or `raidar digest`.
