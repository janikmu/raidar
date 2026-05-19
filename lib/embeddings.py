"""Local embedding generation + numpy cosine search for the AI Radar vault.

Design notes
------------
- Backend: OpenAI-compatible HTTP endpoint (default: Ollama at /v1/embeddings).
- Storage: a single JSON file at ``{vault_path}/embeddings/index.json``.
  Schema per entry:
      {
        "vector": [float, ...],   # L2-normalized
        "text_chunk": str,        # exact prose used to generate the vector
        "last_updated": "YYYY-MM-DD"
      }
- Change detection: byte-equal compare of ``text_chunk``. No separate hash.
- Cosine: vectors are stored already L2-normalized, so search is a single
  matrix-vector dot product. Six lines of numpy, no scikit-learn.
- Concurrency: single-writer tool. Atomic save via tempfile + os.replace.
- Scale: hundreds of entities. Flat JSON + numpy is intentionally adequate.

This module exposes a free function ``embed`` plus an ``Index`` class that
wraps load/upsert/delete/search/save. See the bottom of the file for a
smoke test runnable via ``uv run python -m lib.embeddings``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import openai
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lib.config import Config, load
from lib.logging_setup import setup as setup_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client + low-level embed()
# ---------------------------------------------------------------------------


def _make_client(cfg: Config) -> OpenAI:
    base = (cfg.embedding_provider.base_url or "").rstrip("/")
    suffix = cfg.embedding_openai_compat_suffix or ""
    full_base = base + suffix
    return OpenAI(
        base_url=full_base,
        api_key=cfg.embedding_provider.api_key or "not-needed",
        timeout=cfg.embedding_provider.timeout_s,
    )


_RETRYABLE = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


@retry(
    reraise=True,
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=30.0),
)
def embed(text: str, cfg: Config | None = None) -> list[float]:
    """Generate a single embedding via the configured OpenAI-compatible backend.

    Retries up to 4 attempts on timeout / connection / 5xx errors with
    exponential backoff. 4xx errors propagate immediately.
    """
    cfg = cfg or load()
    client = _make_client(cfg)
    resp = client.embeddings.create(
        model=cfg.embedding_provider.model,
        input=text,
    )
    return list(resp.data[0].embedding)


# ---------------------------------------------------------------------------
# Index data types
# ---------------------------------------------------------------------------


@dataclass
class IndexEntry:
    vector: list[float]
    text_chunk: str
    last_updated: str  # YYYY-MM-DD


def _l2_normalize(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        # Degenerate but legal — return as-is rather than NaN.
        return arr.tolist()
    return (arr / norm).tolist()


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class Index:
    """JSON-backed embedding index with numpy cosine search.

    Vectors are L2-normalized at upsert time, so ``search`` is just a dot
    product against a stacked matrix of normalized rows.
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg or load()
        self._dir: Path = self._cfg.vault_path / "embeddings"
        self._path: Path = self._dir / "index.json"
        self._entries: dict[str, IndexEntry] = {}
        self._load()

    # ---- persistence -----------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            self._entries = {}
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            # Corruption should not be silent — caller decides what to do.
            raise RuntimeError(
                f"embeddings index at {self._path} is corrupt JSON: {exc}"
            ) from exc
        entries: dict[str, IndexEntry] = {}
        for entity_id, payload in raw.items():
            entries[entity_id] = IndexEntry(
                vector=list(payload["vector"]),
                text_chunk=payload["text_chunk"],
                last_updated=payload["last_updated"],
            )
        self._entries = entries

    def save(self) -> None:
        """Atomic write: tempfile in the same dir, then os.replace."""
        self._dir.mkdir(parents=True, exist_ok=True)
        serializable: dict[str, dict[str, Any]] = {
            entity_id: asdict(entry) for entity_id, entry in self._entries.items()
        }
        # NamedTemporaryFile in the same dir so os.replace is atomic on the
        # same filesystem.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".index.", suffix=".json.tmp", dir=str(self._dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                json.dump(serializable, fp, indent=2, sort_keys=True)
                fp.write("\n")
            os.replace(tmp_path, self._path)
        except Exception:
            # Best-effort cleanup; don't mask the original exception.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ---- CRUD ------------------------------------------------------------

    def get(self, entity_id: str) -> IndexEntry | None:
        return self._entries.get(entity_id)

    def upsert(self, entity_id: str, text: str) -> bool:
        """Generate + store an embedding if the prose changed.

        Returns True if a new embedding was generated and persisted, False
        if the stored text_chunk already matches (no-op).
        """
        existing = self._entries.get(entity_id)
        if existing is not None and existing.text_chunk == text:
            log.debug("embeddings: %s unchanged, skipping", entity_id)
            return False

        log.info("embeddings: generating vector for %s", entity_id)
        vector = embed(text, cfg=self._cfg)
        normalized = _l2_normalize(vector)
        self._entries[entity_id] = IndexEntry(
            vector=normalized,
            text_chunk=text,
            last_updated=date.today().isoformat(),
        )
        self.save()
        return True

    def delete(self, entity_id: str) -> bool:
        if entity_id not in self._entries:
            return False
        del self._entries[entity_id]
        self.save()
        return True

    # ---- search ----------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._entries:
            return []
        query_vec = np.asarray(embed(query, cfg=self._cfg), dtype=np.float32)
        norm = float(np.linalg.norm(query_vec))
        if norm == 0.0:
            return []
        query_vec /= norm

        ids = list(self._entries.keys())
        matrix = np.asarray(
            [self._entries[i].vector for i in ids], dtype=np.float32
        )
        # Stored rows are already L2-normalized, so cosine == dot product.
        sims = matrix @ query_vec
        # argsort descending; clamp top_k.
        k = min(top_k, len(ids))
        top_idx = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in top_idx]

    # ---- introspection (handy for tests) ---------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entity_id: object) -> bool:
        return entity_id in self._entries


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke() -> int:
    setup_logging(level="INFO")
    cfg = load()

    # Probe the backend with a single attempt (no retries). If it's clearly
    # offline, bail out fast rather than waiting ~13s for tenacity to give up.
    try:
        _make_client(cfg).embeddings.create(
            model=cfg.embedding_provider.model, input="ping"
        )
    except openai.APIConnectionError as exc:
        print(f"ollama unreachable, skipping live test: {exc}")
        return 0

    idx = Index(cfg)

    a_id, b_id = "__smoke_a__", "__smoke_b__"
    a_text = (
        "AgentMemory is a persistent memory layer for AI coding agents that "
        "stores tool-call history and project context across sessions."
    )
    b_text = (
        "Polars is a high-performance DataFrame library written in Rust with "
        "Python bindings, focused on lazy query execution."
    )

    created_a = idx.upsert(a_id, a_text)

    created_b = idx.upsert(b_id, b_text)
    print(f"upserted {a_id} (new={created_a}), {b_id} (new={created_b})")

    results = idx.search("memory for coding agents", top_k=5)
    print("search 'memory for coding agents':")
    for entity_id, score in results:
        print(f"  {score:+.4f}  {entity_id}")

    # Re-upsert identical prose — should be a no-op.
    again = idx.upsert(a_id, a_text)
    assert again is False, f"expected no-op on identical text, got {again!r}"
    print(f"re-upsert {a_id} with identical text returned {again} (no-op OK)")

    # Clean up.
    idx.delete(a_id)
    idx.delete(b_id)
    idx.save()

    print("embeddings.py smoke test OK")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke())
