# The concept model

This is the doctrine that keeps the vault coherent. It defines what a *concept*
is, at what altitude to draw one, and the three mechanisms that stop the vault
drifting into duplicates. Read this before changing capture's classification
prompt or the dedup logic.

## The two layers

- **Artifact** — a single piece of evidence: a GitHub repo, paper, post, release,
  or spec. It has an `evaluation` (`new` → `promising` → `recommended`, or
  `deprecated`/`hype`).
- **Concept** — the *idea* an artifact is evidence *for*. It has a lifecycle
  `status` (`emerging` → `watch` → `invest`, plus the terminal `common` /
  `superseded` / `abandoned`).

Every artifact maps to exactly one concept. A concept gathers the artifacts that
implement it.

## What a concept is — the altitude rule

> A concept is a **capability or approach** that several artifacts could
> implement as **alternatives to one another**.

Good concepts: `agent-memory`, `llm-routing-proxies`, `prompt-optimization`,
`retrieval-augmented-generation`. Each could plausibly hold 3+ sibling tools that
compete or substitute.

Two failure modes, both of which the vault has suffered:

- **Too low (product-shaped).** Naming a concept after one tool. `caveman` and
  `ponytail` are both token/context compressors; they belong to *one*
  compression concept, not `agent-communication-compression` and
  `agent-productivity-tools`.
- **Too high (umbrella-shaped).** `agent-productivity-tools`, `agent-tooling` —
  so broad that unrelated artifacts pile in and the concept says nothing.

Litmus test: *could a second, competing artifact join this concept tomorrow?*
If only this one tool could ever fit, the altitude is too low. If a dozen
unrelated tools would fit, it's too high.

A concept with exactly one artifact is a *smell*, not a goal — it's either too
low, or it's still waiting for siblings. `raidar health` lists singletons.

**A spec is not its own concept.** When a standard/schema/protocol and its
implementations both show up, they belong to *one* concept: the spec is the
artifact that `introduces` it, the implementations `implements` it. Don't make
`agent-skills-spec` next to `agent-skills-library` — make one `agent-skills`
concept where `agentskills-spec` introduces and the libraries implement. The
`relationship` field carries that distinction; the concept stays single.

## Relationships — and the one that depends on concept kind

Each artifact links to its concept with a `relationship`:

- `introduces` — the originating spec/standard repo, or the official companion code
  of the paper/whitepaper that originated the concept. One per concept, usually.
- `implements` — builds, embodies, or instantiates the concept. The default.
- `extends` — considerably advances the concept beyond its original form (rare).
- `applies` — the artifact's primary purpose is something *else* and it merely uses
  the concept as a component.
- `discusses` — a paper/post commenting on the concept without building it.

Four of these are stable across all concepts. **`implements` vs `applies` is the
only one that depends on what the concept is.** The litmus:

> *Can an artifact use this concept while being primarily about something else?*

- **No → paradigm/category concept** (`agent-skills`, `agent-memory`,
  `agent-development-frameworks`). Membership *is* being an instance — a skills
  collection doesn't "apply skills," it *is* skills. Every member `implements`;
  `applies` is meaningless here.
- **Yes → technique/construct concept** (`retrieval-augmented-generation`,
  `react-pattern`, `function-calling`, protocols like `model-context-protocol`). A
  RAG framework `implements` RAG; a coding agent that retrieves internally
  `applies` RAG. The distinction tells you which artifacts *are* the technique vs
  which merely *use* it.

Rule of thumb: the seeded canonical concepts tend to be the construct kind; the
capture-grown ones tend to be paradigms. When unsure, use `implements`.

## The three guards against duplicates

Capture decides a concept with an LLM, which is fallible. Three mechanisms keep
it honest — they compound, they don't replace each other.

1. **Candidate retrieval (before the LLM).** Capture embeds the incoming artifact
   and retrieves the nearest existing concepts, then shows them to the LLM with
   their prose under *"Closest existing concepts — consider these first"*. The
   model recognises an existing home instead of inventing a synonym.
   (`jobs/capture.py:_candidate_block`)

2. **The dedup gate (after the LLM).** If the LLM still proposes a *new* concept,
   capture embeds the proposal and compares it to every existing concept. Above
   `thresholds.concept_dedup` (default **0.93**) the artifact is attached to the
   nearest existing concept instead, and the concept is flagged `review_needed`.
   A collision on the proposed *slug* is treated the same way — a re-derived slug
   means the same idea, so we attach rather than fork a `-2`.
   (`jobs/capture.py:_nearest_concept`)

3. **The health check (after the fact).** `raidar health` finds what slips
   through: `-N` forks, duplicate labels, and — with `--semantic` — concept pairs
   above the cosine threshold (default 0.90). It's the periodic backstop.

> **Threshold calibration.** The local embedding model
> (`nomic-embed-text`) compresses everything in agent-tooling space: genuinely
> *distinct* concepts (e.g. `agent-development-frameworks` vs
> `agent-evaluation-frameworks`) sit around cosine **0.90**, while true
> duplicates (`agent-memory` / `agent-memory-systems`) are **0.92+**. So the
> gate is deliberately strict (0.93 — only collapse near-certain duplicates) and
> the health backstop a touch looser (0.90 — surface candidates for a human).
> If you swap embedding models, re-measure: run `raidar health --semantic
> --threshold 0.80 --json` and look at where real duplicates separate from noise.

For any of this to work, **every concept must be embedded**. Seeded concepts are
embedded at seed time; if the backend was down, `raidar reindex` fills the gap.
Un-embedded concepts are invisible to guards 1 and 2 — `raidar health` reports
them as `index-drift`.

## When torn

Prefer the existing concept. If you (or the LLM) are unsure whether something is
a new concept or an instance of an existing one, attach to the existing one and
set `review_needed=true`. Splitting later (creating a concept and moving
artifacts to it) is cheap and reviewable; un-forking a `-2` after it has
accumulated artifacts and history is not.

## Maintenance workflow

```bash
raidar health                     # structural check (fast, offline)
raidar health --semantic          # + embedding-based near-duplicate pairs
raidar reindex --prune            # embed everything; drop orphan/legacy index entries
raidar merge-concept SRC DST      # fold a duplicate SRC into keeper DST
raidar rename-concept OLD NEW     # rename a concept slug (repoints artifacts + embedding)
raidar search pending             # concepts flagged review_needed
```

A healthy vault returns zero `error`-severity findings from `raidar health`.
Run it after bulk captures and before committing the vault.
