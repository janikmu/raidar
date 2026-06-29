"""Health check for the AI Radar vault.

Read-only diagnostics that surface the structural antipatterns the vault drifts
into over time: duplicate / forked concepts, dangling cross-references, orphans,
embedding-index drift, and leftover legacy files. Structural checks run offline;
``--semantic`` adds embedding-based near-duplicate concept detection (needs the
embedding backend, e.g. LMStudio).

Usage:
    raidar health                       # structural checks, human-readable
    raidar health --json                # machine-readable findings
    raidar health --semantic            # also flag near-duplicate concepts
    raidar health --semantic --threshold 0.88
    raidar health --strict              # exit non-zero on warnings too

Exit codes:
    0  no error-severity findings (and, without --strict, no/any warnings)
    1  at least one error-severity finding (or any warning under --strict)
    2  infrastructure error (e.g. embedding backend down for --semantic)

Severities:
    error  data is broken or self-contradictory (dangling refs, forked ids)
    warn   likely-wrong but not corrupting (duplicate labels, index drift)
    info   worth a glance, expected in normal operation (singletons, cruft)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import typer

from lib import config as config_module
from lib import vault
from lib.body import parse as parse_body
from lib.embeddings import Index, embed
from lib.logging_setup import setup as setup_logging
from lib.vault import Artifact, Concept

log = logging.getLogger("jobs.health")
app = typer.Typer(add_completion=False, help=__doc__)

# Canonical enums — source of truth lives in jobs/capture.py (artifacts) and
# jobs/enrich.py (concept lifecycle). Duplicated here so the health check stays
# light and offline (capture pulls in heavy web-fetch deps).
_ARTIFACT_TYPES = {"repo", "paper", "post", "release", "spec"}
_EVALUATIONS = {"new", "promising", "recommended", "deprecated", "hype"}
_CONCEPT_STATUSES = {"emerging", "watch", "invest", "common", "superseded", "abandoned"}

_SUFFIX_RE = re.compile(r"^(?P<base>.+)-(?P<n>\d+)$")

ERROR, WARN, INFO = "error", "warn", "info"
_SEVERITY_ORDER = {ERROR: 0, WARN: 1, INFO: 2}


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    check: str
    severity: str
    entity: str
    message: str
    suggestion: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class _VaultView:
    """Everything the checks need, loaded once."""

    concepts: list[Concept]
    artifacts: list[Artifact]
    concept_ids: set[str] = field(default_factory=set)
    artifact_ids: set[str] = field(default_factory=set)

    @classmethod
    def load(cls) -> "_VaultView":
        concepts = vault.list_concepts()
        artifacts = vault.list_artifacts()
        return cls(
            concepts=concepts,
            artifacts=artifacts,
            concept_ids={c.id for c in concepts},
            artifact_ids={a.id for a in artifacts},
        )


def _norm_label(fm: dict[str, Any], fallback: str) -> str:
    label = fm.get("label") or fallback
    return str(label).strip().lower()


def _artifact_entry_ids(concept: Concept) -> list[str]:
    out: list[str] = []
    for entry in concept.frontmatter.get("artifacts") or []:
        aid = entry.get("id") if isinstance(entry, dict) else str(entry)
        if aid:
            out.append(aid)
    return out


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


def check_dangling_refs(v: _VaultView) -> list[Finding]:
    out: list[Finding] = []
    for a in v.artifacts:
        cid = a.frontmatter.get("concept")
        if cid and cid not in v.concept_ids:
            out.append(Finding(
                "dangling-ref", ERROR, f"artifact:{a.id}",
                f"points to concept {cid!r} which does not exist",
                f"raidar capture --update {a.id}  (or fix the concept: field)",
            ))
    for c in v.concepts:
        for aid in _artifact_entry_ids(c):
            if aid not in v.artifact_ids:
                out.append(Finding(
                    "dangling-ref", ERROR, f"concept:{c.id}",
                    f"lists artifact {aid!r} which does not exist",
                    "remove the stale entry from the concept's artifacts: list",
                ))
    return out


def check_link_asymmetry(v: _VaultView) -> list[Finding]:
    out: list[Finding] = []
    listed: dict[str, set[str]] = {c.id: set(_artifact_entry_ids(c)) for c in v.concepts}
    for a in v.artifacts:
        cid = a.frontmatter.get("concept")
        if cid and cid in listed and a.id not in listed[cid]:
            out.append(Finding(
                "link-asymmetry", WARN, f"artifact:{a.id}",
                f"claims concept {cid!r} but that concept doesn't list it",
                "re-run enrich on the concept, or add the artifact to its list",
            ))
    for c in v.concepts:
        for aid in listed[c.id]:
            art = next((a for a in v.artifacts if a.id == aid), None)
            if art is not None and art.frontmatter.get("concept") != c.id:
                out.append(Finding(
                    "link-asymmetry", WARN, f"concept:{c.id}",
                    f"lists artifact {aid!r} but its concept: is "
                    f"{art.frontmatter.get('concept')!r}",
                    "reconcile the artifact's concept: with the concept it lives under",
                ))
    return out


def check_suffix_dup(v: _VaultView) -> tuple[list[Finding], set[str]]:
    """Concepts forked as `-N`. Returns findings + the set of flagged ids."""
    out: list[Finding] = []
    flagged: set[str] = set()
    labels: dict[str, list[str]] = {}
    for c in v.concepts:
        labels.setdefault(_norm_label(c.frontmatter, c.id), []).append(c.id)
    for c in v.concepts:
        m = _SUFFIX_RE.match(c.id)
        if not m:
            continue
        base = m.group("base")
        same_label = [x for x in labels.get(_norm_label(c.frontmatter, c.id), []) if x != c.id]
        target = None
        if base in v.concept_ids:
            target = base
        elif same_label:
            target = same_label[0]
        if target is not None:
            flagged.add(c.id)
            out.append(Finding(
                "suffix-dup", ERROR, f"concept:{c.id}",
                f"looks like a fork of {target!r} (same idea, suffixed id)",
                f"raidar merge-concept {c.id} {target}",
            ))
    return out, flagged


def check_dup_label(v: _VaultView, suffix_flagged: set[str]) -> list[Finding]:
    out: list[Finding] = []
    groups: dict[str, list[Concept]] = {}
    for c in v.concepts:
        groups.setdefault(_norm_label(c.frontmatter, c.id), []).append(c)
    for label, members in sorted(groups.items()):
        # Skip groups already explained by a -N fork (handled by suffix-dup).
        remaining = [c for c in members if c.id not in suffix_flagged]
        if len(remaining) < 2:
            continue
        ids = ", ".join(c.id for c in remaining)
        out.append(Finding(
            "dup-label", WARN, f"label:{label}",
            f"{len(remaining)} concepts share the label {label!r}: {ids}",
            "consider merging with raidar merge-concept",
        ))
    return out


def check_orphans(v: _VaultView) -> list[Finding]:
    out: list[Finding] = []
    for c in v.concepts:
        if _artifact_entry_ids(c):
            continue
        seeded = bool(c.frontmatter.get("seeded"))
        out.append(Finding(
            "orphan-concept", INFO if seeded else WARN, f"concept:{c.id}",
            "has 0 artifacts" + (" (seeded anchor — fine until a capture attaches)" if seeded else ""),
            "" if seeded else "merge into a sibling, or remove if it was created in error",
        ))
    return out


def check_singletons(v: _VaultView) -> list[Finding]:
    singles = [c.id for c in v.concepts if len(_artifact_entry_ids(c)) == 1]
    if not singles:
        return []
    return [Finding(
        "singleton-concept", INFO, "vault",
        f"{len(singles)} concept(s) hold exactly 1 artifact (fragmentation candidates): "
        + ", ".join(sorted(singles)),
        "run `raidar health --semantic` to see which could merge upward",
    )]


def check_frontmatter(v: _VaultView) -> list[Finding]:
    out: list[Finding] = []
    for c in v.concepts:
        fm = c.frontmatter
        if fm.get("id") and fm.get("id") != c.id:
            out.append(Finding(
                "frontmatter", WARN, f"concept:{c.id}",
                f"frontmatter id={fm.get('id')!r} doesn't match filename", "",
            ))
        status = fm.get("status")
        if not status:
            out.append(Finding("frontmatter", WARN, f"concept:{c.id}", "missing status", ""))
        elif status not in _CONCEPT_STATUSES:
            out.append(Finding(
                "frontmatter", WARN, f"concept:{c.id}",
                f"unknown status {status!r}", "",
            ))
    for a in v.artifacts:
        fm = a.frontmatter
        if fm.get("id") and fm.get("id") != a.id:
            out.append(Finding(
                "frontmatter", WARN, f"artifact:{a.id}",
                f"frontmatter id={fm.get('id')!r} doesn't match filename", "",
            ))
        if not fm.get("concept"):
            out.append(Finding(
                "frontmatter", WARN, f"artifact:{a.id}", "missing concept: link", "",
            ))
        ev = fm.get("evaluation")
        if ev and ev not in _EVALUATIONS:
            out.append(Finding(
                "frontmatter", WARN, f"artifact:{a.id}", f"unknown evaluation {ev!r}", "",
            ))
        atype = fm.get("type")
        if atype and atype not in _ARTIFACT_TYPES:
            out.append(Finding(
                "frontmatter", WARN, f"artifact:{a.id}", f"unknown type {atype!r}", "",
            ))
    return out


def check_index_drift(v: _VaultView, cfg: config_module.Config) -> list[Finding]:
    out: list[Finding] = []
    layers = {"concepts": v.concept_ids, "artifacts": v.artifact_ids}
    for layer, on_disk in layers.items():
        try:
            idx = Index(cfg, layer=layer)
        except Exception as exc:  # noqa: BLE001
            out.append(Finding("index-drift", WARN, f"index:{layer}", f"unreadable: {exc}", ""))
            continue
        in_index = set(idx.ids())
        missing = on_disk - in_index
        stale = in_index - on_disk
        if missing:
            out.append(Finding(
                "index-drift", WARN, f"index:{layer}",
                f"{len(missing)} {layer} not embedded: " + ", ".join(sorted(missing)[:12])
                + (" …" if len(missing) > 12 else ""),
                "raidar reindex --layer " + layer,
            ))
        if stale:
            out.append(Finding(
                "index-drift", WARN, f"index:{layer}",
                f"{len(stale)} index entries have no file: " + ", ".join(sorted(stale)[:12])
                + (" …" if len(stale) > 12 else ""),
                "raidar reindex --layer " + layer + " --prune",
            ))
    return out


def check_legacy_cruft(cfg: config_module.Config) -> list[Finding]:
    out: list[Finding] = []
    root = cfg.vault_path
    entities = root / "entities"
    if entities.is_dir():
        n = sum(1 for _ in entities.glob("*.md"))
        out.append(Finding(
            "legacy-cruft", INFO, "entities/",
            f"legacy pre-split layout still present ({n} files)",
            "migrate or remove once confirmed superseded by concepts/ + artifacts/",
        ))
    for name in ("index.json", "index.bak.json"):
        p = root / "embeddings" / name
        if p.is_file():
            out.append(Finding(
                "legacy-cruft", INFO, f"embeddings/{name}",
                "leftover single-index file from the pre-split embedding layout",
                "raidar reindex --prune  (removes legacy index files)",
            ))
    return out


def check_orphan_signals(v: _VaultView, cfg: config_module.Config) -> list[Finding]:
    sig_dir = cfg.vault_path / "signals"
    if not sig_dir.is_dir():
        return []
    orphans = [p.stem for p in sig_dir.glob("*.jsonl") if p.stem not in v.artifact_ids]
    if not orphans:
        return []
    return [Finding(
        "orphan-signal", INFO, "signals/",
        f"{len(orphans)} signal file(s) without an artifact: " + ", ".join(sorted(orphans)[:12])
        + (" …" if len(orphans) > 12 else ""),
        "",
    )]


def check_launchd_agents() -> list[Finding]:
    """macOS only: flag scheduled enrich/digest agents that were never installed
    (or were lost — e.g. after a fresh machine setup that skipped that step)."""
    import platform
    if platform.system() != "Darwin":
        return []
    from jobs.launchd import EXPECTED_LABELS, installed_plists
    missing = sorted(set(EXPECTED_LABELS) - set(installed_plists()))
    if not missing:
        return []
    return [Finding(
        "launchd", WARN, "launchd",
        f"{len(missing)} scheduled job(s) not installed: " + ", ".join(missing),
        "raidar install-launchd",
    )]


def check_review_pending(v: _VaultView) -> list[Finding]:
    flagged = [c.id for c in v.concepts if c.frontmatter.get("review_needed")]
    if not flagged:
        return []
    return [Finding(
        "review-pending", INFO, "vault",
        f"{len(flagged)} concept(s) flagged review_needed: " + ", ".join(sorted(flagged)),
        "raidar search pending",
    )]


# ---------------------------------------------------------------------------
# Semantic check
# ---------------------------------------------------------------------------


def _concept_embed_text(c: Concept) -> str:
    """Prose used to embed a concept: label + What it is + Why it matters."""
    sections = parse_body(c.body)
    label = c.frontmatter.get("label", c.id)
    return "\n".join(
        s for s in (
            str(label),
            sections.get("What it is", ""),
            sections.get("Why it matters", ""),
        ) if s
    )


def check_near_dup_concepts(
    v: _VaultView, cfg: config_module.Config, threshold: float
) -> list[Finding]:
    import numpy as np

    idx = Index(cfg, layer="concepts")
    ids: list[str] = []
    vectors: list[list[float]] = []
    embedded_live = 0
    for c in v.concepts:
        entry = idx.get(c.id)
        if entry is not None:
            ids.append(c.id)
            vectors.append(entry.vector)
            continue
        # Not in the index — embed in memory (no persistence) so the check is
        # complete even before a reindex.
        vec = embed(_concept_embed_text(c), cfg=cfg)
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            continue
        ids.append(c.id)
        vectors.append((arr / norm).tolist())
        embedded_live += 1

    if embedded_live:
        log.info("semantic check embedded %d concept(s) missing from the index", embedded_live)

    if len(ids) < 2:
        return []

    matrix = np.asarray(vectors, dtype=np.float32)
    sims = matrix @ matrix.T
    out: list[Finding] = []
    pairs: list[tuple[float, str, str]] = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sims[i, j])
            if score > threshold:
                pairs.append((score, ids[i], ids[j]))
    pairs.sort(reverse=True)
    for score, a, b in pairs:
        out.append(Finding(
            "near-dup-concept", WARN, f"concept:{a}",
            f"cosine {score:.3f} to concept {b!r} — likely the same idea",
            f"raidar merge-concept {b} {a}   (keep whichever has the better prose)",
        ))
    return out


# ---------------------------------------------------------------------------
# Runner + printing
# ---------------------------------------------------------------------------


def run_checks(
    cfg: config_module.Config, *, semantic: bool, threshold: float
) -> list[Finding]:
    v = _VaultView.load()
    findings: list[Finding] = []
    findings += check_dangling_refs(v)
    findings += check_link_asymmetry(v)
    suffix_findings, suffix_flagged = check_suffix_dup(v)
    findings += suffix_findings
    findings += check_dup_label(v, suffix_flagged)
    findings += check_orphans(v)
    findings += check_singletons(v)
    findings += check_frontmatter(v)
    findings += check_index_drift(v, cfg)
    findings += check_legacy_cruft(cfg)
    findings += check_orphan_signals(v, cfg)
    findings += check_review_pending(v)
    findings += check_launchd_agents()
    if semantic:
        findings += check_near_dup_concepts(v, cfg, threshold)
    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.check, f.entity))
    return findings


_ICON = {ERROR: "✗", WARN: "⚠", INFO: "·"}


def _print_human(findings: list[Finding], v_counts: dict[str, int]) -> None:
    if not findings:
        print("✓ vault healthy — no findings.")
    for sev in (ERROR, WARN, INFO):
        group = [f for f in findings if f.severity == sev]
        if not group:
            continue
        print(f"\n{_ICON[sev]} {sev.upper()} ({len(group)})")
        for f in group:
            print(f"  [{f.check}] {f.entity}: {f.message}")
            if f.suggestion:
                print(f"      → {f.suggestion}")
    n_err = sum(1 for f in findings if f.severity == ERROR)
    n_warn = sum(1 for f in findings if f.severity == WARN)
    n_info = sum(1 for f in findings if f.severity == INFO)
    print(
        f"\nScanned {v_counts['concepts']} concepts, {v_counts['artifacts']} artifacts — "
        f"{n_err} error, {n_warn} warn, {n_info} info."
    )


@app.command()
def health(
    json_out: bool = typer.Option(False, "--json", help="Emit findings as JSON."),
    semantic: bool = typer.Option(
        False, "--semantic", help="Also flag near-duplicate concepts (needs embedding backend)."
    ),
    threshold: float = typer.Option(
        0.90, "--threshold", help="Cosine threshold for --semantic near-duplicate detection."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero on warnings too (default: only on errors)."
    ),
) -> None:
    """Run vault health checks and report findings."""
    cfg = config_module.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)

    try:
        findings = run_checks(cfg, semantic=semantic, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        # Most likely the embedding backend is down during --semantic.
        if semantic:
            print(f"ERROR: semantic check failed (embedding backend down?): {exc}", file=sys.stderr)
            raise typer.Exit(code=2)
        raise

    v_counts = {
        "concepts": len(vault.list_concepts()),
        "artifacts": len(vault.list_artifacts()),
    }

    if json_out:
        print(json.dumps([f.as_dict() for f in findings], indent=2))
    else:
        _print_human(findings, v_counts)

    n_err = sum(1 for f in findings if f.severity == ERROR)
    n_warn = sum(1 for f in findings if f.severity == WARN)
    if n_err or (strict and n_warn):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
