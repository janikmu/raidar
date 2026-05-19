"""Canonical entity-body template — render and parse the four sections.

The spec body looks like:

    ## What it is
    ...

    ## Why it matters
    ...

    ## Current assessment
    ...

    ## History
    - 2026-05-18: Captured. Status set to watch.

Both capture (writes fresh prose) and enrich (rewrites prose on status changes)
need to produce this same shape, so the renderer lives here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SECTIONS = ("What it is", "Why it matters", "Current assessment")
HISTORY = "History"

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class EntityBody:
    what_it_is: str
    why_it_matters: str
    current_assessment: str
    history: list[str]  # bullet text only (no leading "- "), excluding the date prefix preserved as-is


def render(
    what_it_is: str,
    why_it_matters: str,
    current_assessment: str,
    history_bullets: list[str] | None = None,
) -> str:
    """Render the entity body markdown. `history_bullets` are full bullet text
    lines (without the leading "- "). If empty/None, no History section is added —
    capture should typically pass the first bullet (e.g. "2026-05-18: Captured.")."""
    parts = [
        f"## What it is\n{what_it_is.strip()}\n",
        f"## Why it matters\n{why_it_matters.strip()}\n",
        f"## Current assessment\n{current_assessment.strip()}\n",
    ]
    if history_bullets:
        history_block = "## History\n" + "\n".join(f"- {b}" for b in history_bullets) + "\n"
        parts.append(history_block)
    return "\n".join(parts)


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
