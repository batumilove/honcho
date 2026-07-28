"""Conversation-shaping helpers: token counting + tool-aware truncation.

Moved out of src/utils/clients.py as part of the migration into src/llm/.
These are pure helpers with no orchestration dependencies.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from src.exceptions import ValidationException
from src.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens in a list of messages using tiktoken."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Anthropic-style content blocks
            total += estimate_tokens(json.dumps(content))
        if "parts" in msg:
            try:
                total += estimate_tokens(json.dumps(msg["parts"]))
            except TypeError:
                # Non-JSON-serializable content (e.g. bytes) — estimate from repr.
                total += estimate_tokens(str(msg["parts"]))
        # OpenAI keeps tool arguments outside ``content``. Count the complete
        # provider-visible structure or long tool calls can evade the cap.
        if "tool_calls" in msg:
            try:
                total += estimate_tokens(json.dumps(msg["tool_calls"]))
            except TypeError:
                total += estimate_tokens(str(msg["tool_calls"]))
        for key in ("tool_call_id", "name"):
            value = msg.get(key)
            if value is not None:
                total += estimate_tokens(str(value))
    return total


def _is_tool_use_message(msg: dict[str, Any]) -> bool:
    """Check if a message contains tool calls (any format).

    Recognizes:
    - Anthropic: ``content`` is a list containing a ``{"type": "tool_use"}`` block.
    - Gemini: ``parts`` is a list containing a ``{"function_call": …}`` entry.
    - OpenAI: assistant message with a non-empty ``tool_calls`` field.
    """
    content = msg.get("content")
    if isinstance(content, list):
        for block in cast(list[dict[str, Any]], content):
            if block.get("type") == "tool_use":
                return True
    parts = msg.get("parts")
    if isinstance(parts, list):
        for part in cast(list[dict[str, Any]], parts):
            if "function_call" in part:
                return True
    return bool(msg.get("tool_calls"))


def _is_tool_result_message(msg: dict[str, Any]) -> bool:
    """Check if a message contains tool results (any format).

    Recognizes:
    - Anthropic: ``content`` is a list containing a ``{"type": "tool_result"}`` block.
    - Gemini: ``parts`` is a list containing a ``{"function_response": …}`` entry.
    - OpenAI: message with ``role == "tool"``.
    """
    content = msg.get("content")
    if isinstance(content, list):
        for block in cast(list[dict[str, Any]], content):
            if block.get("type") == "tool_result":
                return True
    parts = msg.get("parts")
    if isinstance(parts, list):
        for part in cast(list[dict[str, Any]], parts):
            if "function_response" in part:
                return True
    return msg.get("role") == "tool"


def _group_into_units(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group messages into logical conversation units.

    A unit is either:
    - A tool_use message + ALL consecutive tool_result messages that follow
    - A single non-tool message

    Keeps tool_use / tool_result pairs together so truncation never breaks
    them apart.
    """
    units: list[list[dict[str, Any]]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if _is_tool_use_message(msg):
            j = i + 1
            while j < len(messages) and _is_tool_result_message(messages[j]):
                j += 1
            unit = messages[i:j]
            if len(unit) > 1:
                units.append(unit)
                i = j
            else:
                # Orphaned tool_use with no results — skip it.
                logger.debug(f"Skipping orphaned tool_use at index {i}")
                i += 1
        elif _is_tool_result_message(msg):
            # Orphaned tool_result — skip it.
            logger.debug(f"Skipping orphaned tool_result at index {i}")
            i += 1
        else:
            units.append([msg])
            i += 1

    return units


def _validate_truncation_result(
    result: list[dict[str, Any]],
    *,
    pre_tokens: int,
    max_tokens: int,
    system_tokens: int,
    retained_units: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Log token diagnostics and enforce the truncation hard cap."""
    post_tokens = count_message_tokens(result)
    retained_unit_tokens = [count_message_tokens(unit) for unit in retained_units]
    cap_exceeded = post_tokens > max_tokens
    logger.log(
        logging.WARNING if cap_exceeded else logging.INFO,
        "Truncation diagnostics: pre_tokens=%s post_tokens=%s max_tokens=%s "
        + "system_tokens=%s retained_unit_tokens=%s cap_hit=true",
        pre_tokens,
        post_tokens,
        max_tokens,
        system_tokens,
        retained_unit_tokens,
    )
    if cap_exceeded:
        raise ValidationException(
            "Conversation remains over max_tokens after truncation: "
            + f"{post_tokens} > {max_tokens}"
        )
    return result


def truncate_messages_to_fit(
    messages: list[dict[str, Any]],
    max_tokens: int,
    preserve_system: bool = True,
) -> list[dict[str, Any]]:
    """Truncate messages to fit within a token limit while maintaining valid structure.

    Strategy:
    1. Group messages into units (tool_use + results together, or single messages)
    2. Remove oldest units first to preserve recent context
    3. Units stay intact so tool_use/tool_result pairs are never broken
    """
    current_tokens = count_message_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    logger.info(f"Truncating: {current_tokens} tokens exceeds {max_tokens} limit")

    system_messages: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") == "system" and preserve_system:
            system_messages.append(msg)
        else:
            conversation.append(msg)

    system_tokens = count_message_tokens(system_messages)
    available_tokens = max_tokens - system_tokens
    units = _group_into_units(conversation)

    if available_tokens <= 0:
        logger.warning("System message exceeds max_input_tokens")
        return _validate_truncation_result(
            messages,
            pre_tokens=current_tokens,
            max_tokens=max_tokens,
            system_tokens=system_tokens,
            retained_units=units,
        )

    if not units:
        logger.warning("No valid conversation units")
        return _validate_truncation_result(
            system_messages,
            pre_tokens=current_tokens,
            max_tokens=max_tokens,
            system_tokens=system_tokens,
            retained_units=[],
        )

    # Preserve the most recent real user query. Tool loops can otherwise evict
    # their only user message while retaining recent assistant/tool units,
    # producing an invalid provider request (for example, Qwen/vLLM rejects it
    # with "No user query found in messages"). Tool-result messages may use the
    # user role for Anthropic/Gemini shapes, so they do not qualify.
    user_query_unit = next(
        (
            unit
            for unit in reversed(units)
            if any(
                msg.get("role") == "user" and not _is_tool_result_message(msg)
                for msg in unit
            )
        ),
        None,
    )

    # Drop oldest removable units until the conversation fits. Always preserve
    # the latest unit and, when present, the most recent real user query.
    while len(units) > 1:
        flat_messages = [m for unit in units for m in unit]
        if count_message_tokens(flat_messages) <= available_tokens:
            break
        removable_index = next(
            (
                index
                for index, unit in enumerate(units[:-1])
                if unit is not user_query_unit
            ),
            None,
        )
        if removable_index is None:
            break
        removed_unit = units.pop(removable_index)
        logger.debug(
            "Dropping conversation unit with "
            + f"{len(removed_unit)} messages "
            + f"(~{count_message_tokens(removed_unit)} tokens)"
        )

    result = _validate_truncation_result(
        system_messages + [m for unit in units for m in unit],
        pre_tokens=current_tokens,
        max_tokens=max_tokens,
        system_tokens=system_tokens,
        retained_units=units,
    )
    result_tokens = count_message_tokens(result)
    logger.info(
        f"Truncation complete: {current_tokens} → {result_tokens} tokens "
        + f"({len(messages)} → {len(result)} messages)"
    )
    return result


__all__ = [
    "count_message_tokens",
    "truncate_messages_to_fit",
]
