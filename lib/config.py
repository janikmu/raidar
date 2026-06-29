"""Load config.yaml with env-var resolution for paths and provider URLs."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from lib import secrets

def get_config_path() -> Path:
    import os
    # 1. Environment variable override
    if "RAIDAR_CONFIG" in os.environ:
        return Path(os.environ["RAIDAR_CONFIG"]).expanduser().resolve()

    # 2. Single canonical location: ~/.config/raidar/config.yaml
    #    (XDG-style, portable, and friendly to dotfile managers like chezmoi).
    return Path.home() / ".config" / "raidar" / "config.yaml"


CONFIG_PATH = get_config_path()


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
    # Refresh in case get_config_path changes dynamically (e.g. in tests)
    config_path = get_config_path()
    try:
        with config_path.open() as fp:
            raw = yaml.safe_load(fp)
    except FileNotFoundError:
        raise RuntimeError(
            f"Configuration file not found at {config_path}.\n"
            f"Please run 'raidar init' first to initialize your configuration and vault."
        )

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

    # Resolve relative paths relative to vault_path rather than config file directory
    log_file = log_raw.get("file")
    if log_file:
        log_file_path = Path(log_file).expanduser()
        if not log_file_path.is_absolute():
            log_file_path = (vault_path / log_file_path).resolve()
    else:
        log_file_path = None

    context_val = paths_raw.get("context", "context.md")
    context_path = Path(context_val).expanduser()
    if not context_path.is_absolute():
        context_path = (vault_path / context_path).resolve()

    enrich_val = paths_raw.get("enrich_output", "logs/last_enrich.json")
    enrich_output_path = Path(enrich_val).expanduser()
    if not enrich_output_path.is_absolute():
        enrich_output_path = (vault_path / enrich_output_path).resolve()

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
