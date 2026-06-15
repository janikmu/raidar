"""Rebuild the embedding indexes from the markdown on disk.

Upserts every concept/artifact body into its embedding index (a no-op when the
prose is unchanged, so it's cheap to re-run). ``--prune`` additionally drops
index entries that no longer have a file and removes the legacy single-index
files left over from the pre-split layout.

Use it to:
  - embed concepts that were written without an embedding (e.g. seeded anchors),
    so the capture dedup gate and `raidar health --semantic` can see them;
  - clean up after manual edits or merges.

Usage:
    raidar reindex                       # upsert both layers
    raidar reindex --layer concepts      # one layer only
    raidar reindex --prune               # also drop orphan entries + legacy files
    raidar reindex --dry-run             # report what would change, write nothing
"""

from __future__ import annotations

import logging
import sys

import openai
import typer

from lib import config as config_module
from lib import vault
from lib.embeddings import Index
from lib.logging_setup import setup as setup_logging

log = logging.getLogger("jobs.reindex")
app = typer.Typer(add_completion=False, help=__doc__)

_LAYERS = ("concepts", "artifacts")


def _bodies(layer: str) -> dict[str, str]:
    if layer == "concepts":
        return {c.id: c.body for c in vault.list_concepts()}
    return {a.id: a.body for a in vault.list_artifacts()}


def _reindex_layer(layer: str, cfg: config_module.Config, *, prune: bool, dry_run: bool) -> None:
    idx = Index(cfg, layer=layer)
    bodies = _bodies(layer)
    in_index = set(idx.ids())

    to_update = [eid for eid, body in bodies.items()
                 if (e := idx.get(eid)) is None or e.text_chunk != body]
    stale = sorted(in_index - set(bodies))

    print(f"\n[{layer}] {len(bodies)} on disk, {len(in_index)} in index "
          f"→ {len(to_update)} to (re)embed, {len(stale)} stale"
          + (" (will prune)" if prune and stale else ""))

    if dry_run:
        if to_update:
            print("  would embed: " + ", ".join(sorted(to_update)[:12])
                  + (" …" if len(to_update) > 12 else ""))
        if prune and stale:
            print("  would prune: " + ", ".join(stale[:12]) + (" …" if len(stale) > 12 else ""))
        return

    n_done = 0
    for eid in to_update:
        if idx.upsert(eid, bodies[eid]):
            n_done += 1
    if prune:
        for eid in stale:
            idx.delete(eid)
    print(f"  embedded {n_done}, pruned {len(stale) if prune else 0}")


def _remove_legacy(cfg: config_module.Config, *, dry_run: bool) -> None:
    emb = cfg.vault_path / "embeddings"
    for name in ("index.json", "index.bak.json"):
        p = emb / name
        if p.is_file():
            if dry_run:
                print(f"  would remove legacy {p.name}")
            else:
                p.unlink()
                print(f"  removed legacy {p.name}")


@app.command()
def reindex(
    layer: str = typer.Option("all", "--layer", help="concepts | artifacts | all"),
    prune: bool = typer.Option(
        False, "--prune", help="Drop index entries with no file + remove legacy index files."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report changes; write nothing."),
) -> None:
    """Rebuild embedding indexes from the markdown on disk."""
    cfg = config_module.load()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)

    if layer not in ("all", *_LAYERS):
        print(f"ERROR: --layer must be one of: all, {', '.join(_LAYERS)}", file=sys.stderr)
        raise typer.Exit(code=1)

    targets = _LAYERS if layer == "all" else (layer,)
    try:
        for lyr in targets:
            _reindex_layer(lyr, cfg, prune=prune, dry_run=dry_run)
    except openai.APIConnectionError as exc:
        print(f"\nERROR: embedding backend unreachable: {exc}", file=sys.stderr)
        raise typer.Exit(code=2)

    if prune:
        _remove_legacy(cfg, dry_run=dry_run)

    print("\nDone." if not dry_run else "\n[dry-run] nothing written.")


if __name__ == "__main__":
    app()
