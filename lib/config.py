"""Load config.yaml with env-var resolution for paths and provider URLs."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from lib import secrets

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str | None
    api_key: str | None
    model: str
    timeout_s: float
    # When true, prepend /no_think to the user message to request that
    # Qwen3 / DeepSeek-R1 skip chain-of-thought reasoning.
    # Note: some LMStudio versions still route all output through reasoning_content
    # even with /no_think — the router handles this transparently.
    no_think: bool = False
    # Appended to the system prompt on every call.
    system_suffix: str | None = None
    # Minimum max_tokens to use for this provider. Useful for local reasoning
    # models (Qwen3, DeepSeek-R1) that need a large budget to think + respond.
    # If the call-site requests fewer tokens, this value wins.
    max_tokens: int | None = None


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    vault_path: Path
    providers: dict[str, ProviderConfig]
    task_chains: dict[str, list[str]]
    embedding_provider: ProviderConfig
    embedding_openai_compat_suffix: str
    github_api_base: str
    github_token: str | None
    github_commit_window_days: int
    thresholds: dict[str, Any] = field(default_factory=dict)
    retry: dict[str, Any] = field(default_factory=dict)
    log_level: str = "INFO"
    log_file: Path | None = None
    context_path: Path = Path("context.md")
    enrich_output_path: Path = Path("logs/last_enrich.json")


def _resolve_provider(name: str, raw: dict[str, Any]) -> ProviderConfig:
    base_url_env = raw.get("base_url_env")
    api_key_env = raw.get("api_key_env")
    base_url = secrets.get(base_url_env) if base_url_env else raw.get("base_url")
    api_key = secrets.get(api_key_env) if api_key_env else None
    suffix = raw.get("system_suffix")
    max_tokens_raw = raw.get("max_tokens")
    return ProviderConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=raw["model"],
        timeout_s=float(raw.get("timeout_s", 60)),
        no_think=bool(raw.get("no_think", False)),
        system_suffix=suffix if suffix else None,
        max_tokens=int(max_tokens_raw) if max_tokens_raw is not None else None,
    )


@lru_cache(maxsize=1)
def load() -> Config:
    with CONFIG_PATH.open() as fp:
        raw = yaml.safe_load(fp)

    vault_path = Path(raw["vault"]["path"]).expanduser()

    providers = {
        name: _resolve_provider(name, cfg) for name, cfg in raw["providers"].items()
    }

    embed_raw = raw["embeddings"]
    embedding_provider = _resolve_provider("__embeddings__", embed_raw)
    embedding_openai_compat_suffix = embed_raw.get("openai_compat_suffix", "")

    github_raw = raw.get("github", {})
    log_raw = raw.get("logging", {})
    paths_raw = raw.get("paths", {})

    root = CONFIG_PATH.parent
    log_file = log_raw.get("file")
    log_file_path = (root / log_file).resolve() if log_file else None
    context_path = (root / paths_raw.get("context", "context.md")).resolve()
    enrich_output_path = (root / paths_raw.get("enrich_output", "logs/last_enrich.json")).resolve()

    return Config(
        raw=raw,
        vault_path=vault_path,
        providers=providers,
        task_chains={k: list(v) for k, v in raw.get("tasks", {}).items()},
        embedding_provider=embedding_provider,
        embedding_openai_compat_suffix=embedding_openai_compat_suffix,
        github_api_base=github_raw.get("api_base", "https://api.github.com"),
        github_token=secrets.get(github_raw.get("token_env", "GITHUB_PAT")),
        github_commit_window_days=int(github_raw.get("commit_window_days", 30)),
        thresholds=raw.get("thresholds", {}),
        retry=raw.get("retry", {}),
        log_level=log_raw.get("level", "INFO"),
        log_file=log_file_path,
        context_path=context_path,
        enrich_output_path=enrich_output_path,
    )
