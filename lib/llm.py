"""LLM router for AI Radar.

One entry point — `Router.generate(task=..., prompt=...)` — fans out across
the provider chain configured for that task. Each provider is OpenAI-wire
compatible. Transient errors retry per-provider; terminal errors fall
through to the next provider; running out of providers raises.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import openai
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from lib.config import Config, ProviderConfig, load

log = logging.getLogger(__name__)

_TRANSIENT = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)
_PERMANENT = (
    openai.AuthenticationError,
    openai.BadRequestError,
    openai.NotFoundError,
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
# Strips reasoning blocks emitted by Qwen3, DeepSeek-R1, etc. before JSON parsing.
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


class ProviderUnavailable(RuntimeError):
    """Provider lacks credentials or base_url. Router skips silently."""


class TerminalError(RuntimeError):
    """Retries exhausted or permanent failure on a provider."""


class AllProvidersFailed(RuntimeError):
    """Every provider in the task chain has failed."""


@dataclass
class Completion:
    text: str
    parsed: dict | list | None
    provider_name: str
    model: str


def _is_local(p: ProviderConfig) -> bool:
    # Local providers have api_key=None because api_key_env was null in config.
    return p.api_key is None


def _ready(p: ProviderConfig) -> bool:
    if not p.base_url:
        return False
    if not _is_local(p) and not p.api_key:
        return False
    return True


def _missing_required(parsed: dict | list | None, schema: dict) -> list[str]:
    """Return the names of required keys absent from `parsed` (top-level only).

    Returns ['<not-an-object>'] if `parsed` is not a dict and the schema expects one.
    Returns [] when nothing is missing (or there are no required keys).
    """
    required = schema.get("required") or []
    if not required:
        return []
    if not isinstance(parsed, dict):
        return ["<not-an-object>"]
    return [k for k in required if k not in parsed]


def _extract_json(text: str) -> dict | list | None:
    cleaned = _THINK_RE.sub("", text).strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(cleaned)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Router:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load()
        r = self.cfg.retry or {}
        self._max_attempts = int(r.get("max_attempts", 4))
        self._initial_backoff = float(r.get("initial_backoff_s", 1.0))
        self._max_backoff = float(r.get("max_backoff_s", 30.0))

    def available_providers(self, task: str) -> list[str]:
        return [n for n in self._chain(task) if _ready(self.cfg.providers[n])]

    def generate(
        self,
        task: str,
        prompt: str,
        *,
        system: str | None = None,
        response_schema: dict | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Completion:
        last_error: BaseException | None = None
        for name in self._chain(task):
            provider = self.cfg.providers[name]
            if not _ready(provider):
                log.debug("provider %s unavailable (missing base_url or api_key)", name)
                last_error = ProviderUnavailable(name)
                continue
            log.info(
                "llm attempt provider=%s task=%s model=%s prompt_len=%d",
                name, task, provider.model, len(prompt),
            )
            try:
                result = self._call_provider(
                    provider, prompt, system, response_schema, temperature, max_tokens,
                )
            except TerminalError as exc:
                log.error("llm provider=%s terminal: %s", name, exc)
                log.info("llm falling through from provider=%s", name)
                last_error = exc
                continue
            log.info("llm success provider=%s task=%s", name, task)
            return result
        raise AllProvidersFailed(f"all providers failed for task {task!r}") from last_error

    def _chain(self, task: str) -> list[str]:
        if task not in self.cfg.task_chains:
            raise ValueError(f"unknown task {task!r} (not in config.task_chains)")
        return self.cfg.task_chains[task]

    def _call_provider(
        self,
        provider: ProviderConfig,
        prompt: str,
        system: str | None,
        response_schema: dict | None,
        temperature: float,
        max_tokens: int | None,
    ) -> Completion:
        client = openai.OpenAI(
            base_url=provider.base_url,
            api_key=provider.api_key or "not-needed",
            timeout=provider.timeout_s,
        )
        strategies = ["json_schema", "json_object", "text"] if response_schema else ["none"]
        last_bad: openai.BadRequestError | None = None
        # Apply per-provider system suffix (arbitrary prompt injection).
        effective_system = system or ""
        if provider.system_suffix:
            effective_system = (effective_system + "\n" + provider.system_suffix).strip()
        # no_think: /no_think is prepended to the USER message (not system) so
        # it appears in the turn where Qwen3 / DeepSeek-R1 actually look for it.
        # Injecting it into the system prompt is unreliable because schema text
        # is appended after, burying the directive mid-prompt.
        for strategy in strategies:
            kwargs: dict = {
                "model": provider.model,
                "messages": self._messages(prompt, effective_system, response_schema, strategy, no_think=provider.no_think),
                "temperature": temperature,
            }
            # Provider may specify a minimum token budget (e.g. local reasoning
            # models need room to think + respond). Use the larger of the two.
            effective_max_tokens = max_tokens
            if provider.max_tokens is not None:
                effective_max_tokens = (
                    max(provider.max_tokens, max_tokens)
                    if max_tokens is not None
                    else provider.max_tokens
                )
            if effective_max_tokens is not None:
                kwargs["max_tokens"] = effective_max_tokens
            rf = self._response_format(response_schema, strategy)
            if rf is not None:
                kwargs["response_format"] = rf
            try:
                resp = self._with_retry(client, kwargs)
            except openai.BadRequestError as exc:
                last_bad = exc
                log.warning(
                    "provider=%s rejected response_format=%s: %s; trying fallback",
                    provider.name, strategy, exc,
                )
                continue
            except _PERMANENT as exc:
                raise TerminalError(f"{provider.name}: {type(exc).__name__}: {exc}") from exc
            except RetryError as exc:
                cause = exc.last_attempt.exception() if exc.last_attempt else exc
                raise TerminalError(
                    f"{provider.name}: retries exhausted ({type(cause).__name__}: {cause})"
                ) from cause

            message = resp.choices[0].message
            text = (message.content or "").strip()
            # LMStudio routes Qwen3 output into reasoning_content (a separate
            # field) instead of content when thinking mode is active — even
            # when /no_think is set, some versions still do this.  Fall back
            # to reasoning_content so the router can extract the JSON.
            if not text:
                rc = getattr(message, "reasoning_content", None)
                if rc:
                    log.debug(
                        "provider=%s strategy=%s: content empty, reading reasoning_content (%d chars)",
                        provider.name, strategy, len(rc),
                    )
                    text = rc.strip()

            parsed: dict | list | None = None
            if response_schema is not None:
                parsed = _extract_json(text)
                missing = _missing_required(parsed, response_schema) if parsed is not None else ["<empty>"]
                if parsed is None or missing:
                    # Either empty/non-JSON, or JSON without the required keys
                    # (common when a proxy silently ignores response_format, or
                    # when a reasoning model burns its budget on <think> tokens).
                    is_last = strategy == strategies[-1]
                    log.warning(
                        "provider=%s strategy=%s invalid response (text_len=%d, missing=%s); %s",
                        provider.name, strategy, len(text), missing,
                        "exhausting provider" if is_last else "trying fallback",
                    )
                    if is_last:
                        raise TerminalError(
                            f"{provider.name}: all response_format strategies returned "
                            f"invalid output (last text_len={len(text)}, missing={missing})"
                        )
                    continue
            return Completion(text=text, parsed=parsed, provider_name=provider.name, model=provider.model)

        raise TerminalError(
            f"{provider.name}: all response_format strategies failed (last_bad={last_bad})"
        ) from last_bad

    @staticmethod
    def _messages(
        prompt: str, system: str | None, response_schema: dict | None, strategy: str,
        *, no_think: bool = False,
    ) -> list[dict]:
        sys_text = system or ""
        # Always inject the schema when one is provided — many proxies silently
        # strip the OpenAI `response_format` field, so the model has to see the
        # schema in the prompt to know what to produce. Belt and suspenders even
        # when `json_schema` strict mode is honored.
        if response_schema is not None:
            required = response_schema.get("required") or []
            required_hint = (
                f"Required keys (must all be present): {', '.join(required)}."
                if required else ""
            )
            sys_text = (
                f"{sys_text}\n\nRespond with valid JSON matching this exact schema. "
                f"Do not invent additional keys. Do not omit required keys.\n"
                f"{required_hint}\n"
                f"Schema: {json.dumps(response_schema)}"
            ).strip()
        msgs: list[dict] = []
        if sys_text:
            msgs.append({"role": "system", "content": sys_text})
        user_content = f"/no_think\n\n{prompt}" if no_think else prompt
        msgs.append({"role": "user", "content": user_content})
        return msgs

    @staticmethod
    def _response_format(response_schema: dict | None, strategy: str) -> dict | None:
        if response_schema is None or strategy in ("text", "none"):
            return None
        if strategy == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema, "strict": True},
            }
        if strategy == "json_object":
            return {"type": "json_object"}
        return None

    def _with_retry(self, client: openai.OpenAI, kwargs: dict):
        @retry(
            retry=retry_if_exception_type(_TRANSIENT),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_random_exponential(multiplier=self._initial_backoff, max=self._max_backoff),
            reraise=False,
            before_sleep=lambda rs: log.warning(
                "llm retry attempt=%d exc=%s",
                rs.attempt_number,
                rs.outcome.exception() if rs.outcome else "?",
            ),
        )
        def _call():
            return client.chat.completions.create(**kwargs)

        return _call()


def _smoke() -> None:
    from lib import logging_setup

    cfg = load()
    logging_setup.setup(level=cfg.log_level, log_file=cfg.log_file)
    router = Router(cfg)

    for task in cfg.task_chains:
        print(
            f"task={task}: chain={cfg.task_chains[task]} "
            f"ready={router.available_providers(task)}"
        )

    target = "classification"
    if not router.available_providers(target):
        print("llm.py smoke test ... skipped (no providers configured)")
        return
    try:
        result = router.generate(
            task=target,
            prompt="Reply with exactly the word OK.",
            system="You are a terse test responder.",
            max_tokens=10,
        )
    except AllProvidersFailed as exc:
        print(f"llm.py smoke test FAILED: {exc} (cause: {exc.__cause__})")
        return
    print(f"provider={result.provider_name} model={result.model} text={result.text!r}")
    print("llm.py smoke test OK")


if __name__ == "__main__":
    _smoke()
