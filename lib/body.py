"""Body template renderers and parsers for the two-layer vault model.

Two kinds of body:

Concept body:
    ## What it is
    ...

    ## Why it matters
    ...

    ## Current assessment
    ...

    ## Artifact summary
    | Artifact | Type | Evaluation |
    |---|---|---|
    | id | repo | recommended |

    ## History
    - 2026-05-18: First captured.

Artifact body:
    ## What it is
    ...

    ## Evaluation rationale
    ...

    ## History
    - 2026-05-18: Captured. Evaluation: new.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

HISTORY = "History"
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def parse(body: str) -> dict[str, str]:
    """Split a body into a {section_title: section_text} dict.

    Robust to extra whitespace and missing sections. Section text excludes
    the heading line and is stripped of surrounding blank lines.
    """
    headings = list(_HEADING_RE.finditer(body))
    if not headings:
        return {}
    out: dict[str, str] = {}
    for i, m in enumerate(headings):
        title = m.group(1).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        out[title] = body[start:end].strip()
    return out


# ---------------------------------------------------------------------------
# Concept body
# ---------------------------------------------------------------------------

CONCEPT_SECTIONS = ("What it is", "Why it matters", "Current assessment", "Artifact summary")


def render_artifact_summary_table(
    artifact_rows: list[dict],
) -> str:
    """Render the artifact summary markdown table from a list of row dicts.

    Each row dict should have keys: id, type, evaluation.
    Returns an empty string if artifact_rows is empty.
    """
    if not artifact_rows:
        return ""
    lines = [
        "| Artifact | Type | Evaluation |",
        "|---|---|---|",
    ]
    for row in artifact_rows:
        aid = row.get("id", "")
        atype = row.get("type", "")
        aeval = row.get("evaluation", "")
        lines.append(f"| {aid} | {atype} | {aeval} |")
    return "\n".join(lines)


def render_concept(
    what_it_is: str,
    why_it_matters: str,
    current_assessment: str,
    artifact_rows: list[dict] | None = None,
    history_bullets: list[str] | None = None,
) -> str:
    """Render a concept body with four prose sections, optional artifact table, and history.

    `artifact_rows` is a list of dicts with keys: id, type, evaluation.
    `history_bullets` are full bullet lines (without the leading "- ").
    """
    parts = [
        f"## What it is\n{what_it_is.strip()}\n",
        f"## Why it matters\n{why_it_matters.strip()}\n",
        f"## Current assessment\n{current_assessment.strip()}\n",
    ]

    table = render_artifact_summary_table(artifact_rows or [])
    if table:
        parts.append(f"## Artifact summary\n{table}\n")
    else:
        parts.append("## Artifact summary\n(none yet)\n")

    if history_bullets:
        history_block = "## History\n" + "\n".join(f"- {b}" for b in history_bullets) + "\n"
        parts.append(history_block)

    return "\n".join(parts)


def parse_concept(body: str) -> dict[str, str]:
    """Split a concept body into a {section_title: section_text} dict.

    Same implementation as parse() — listed separately for clarity.
    """
    return parse(body)


# ---------------------------------------------------------------------------
# Artifact body
# ---------------------------------------------------------------------------

ARTIFACT_SECTIONS = ("What it is", "Evaluation rationale")


def render_artifact(
    what_it_is: str,
    evaluation_rationale: str,
    history_bullets: list[str] | None = None,
) -> str:
    """Render an artifact body with two prose sections and optional history."""
    parts = [
        f"## What it is\n{what_it_is.strip()}\n",
        f"## Evaluation rationale\n{evaluation_rationale.strip()}\n",
    ]
    if history_bullets:
        history_block = "## History\n" + "\n".join(f"- {b}" for b in history_bullets) + "\n"
        parts.append(history_block)
    return "\n".join(parts)


def parse_artifact(body: str) -> dict[str, str]:
    """Split an artifact body into a {section_title: section_text} dict."""
    return parse(body)
