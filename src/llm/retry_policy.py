"""Retry classification shared by all LLM call paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

_CONTEXT_LENGTH_CODES = {
    "context_length_exceeded",
    "context_window_exceeded",
    "max_context_length_exceeded",
}


def _error_fields(exc: BaseException) -> tuple[str, str]:
    body: Any = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return "", str(exc).lower()

    body_mapping = cast(Mapping[str, object], body)
    raw_error = body_mapping.get("error", body_mapping)
    if not isinstance(raw_error, dict):
        return "", str(exc).lower()
    error = cast(Mapping[str, object], raw_error)

    code = str(error.get("code", "")).lower()
    text = " ".join(
        str(error.get(key, "")) for key in ("message", "type", "param")
    ).lower()
    return code, text


def is_retryable_llm_exception(exc: BaseException) -> bool:
    """Reject retries for deterministic HTTP 400 context-window failures."""
    if getattr(exc, "status_code", None) != 400:
        return True

    code, text = _error_fields(exc)
    context_length_failure = code in _CONTEXT_LENGTH_CODES or (
        "maximum context length" in text
        or (
            "context window" in text
            and any(marker in text for marker in ("exceed", "too long", "maximum"))
        )
    )
    return not context_length_failure


__all__ = ["is_retryable_llm_exception"]
