from typing import Any

import pytest

from src.exceptions import ValidationException
from src.llm.conversation import (
    _is_tool_result_message,  # pyright: ignore[reportPrivateUsage]
    _is_tool_use_message,  # pyright: ignore[reportPrivateUsage]
    count_message_tokens,
    truncate_messages_to_fit,
)


def test_truncate_messages_to_fit_rejects_last_unit_over_limit() -> None:
    messages = [
        {"role": "user", "content": "x " * 2000},
    ]

    with pytest.raises(ValidationException, match="remains over max_tokens"):
        truncate_messages_to_fit(messages, max_tokens=1)


def test_truncate_messages_to_fit_rejects_oversized_system_context() -> None:
    messages = [
        {"role": "system", "content": "policy " * 2000},
        {"role": "user", "content": "continue"},
    ]

    with pytest.raises(ValidationException, match="remains over max_tokens"):
        truncate_messages_to_fit(messages, max_tokens=1)


def test_truncate_messages_to_fit_rejects_oversized_tool_unit() -> None:
    messages = [
        {"role": "user", "content": "investigate"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "large " * 2000,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "large result " * 2000,
        },
    ]

    with pytest.raises(ValidationException, match="remains over max_tokens"):
        truncate_messages_to_fit(messages, max_tokens=10)


def test_truncation_cap_hit_logs_token_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    messages = [{"role": "user", "content": "x " * 2000}]

    with (
        caplog.at_level("INFO", logger="src.llm.conversation"),
        pytest.raises(ValidationException),
    ):
        truncate_messages_to_fit(messages, max_tokens=1)

    assert "pre_tokens=" in caplog.text
    assert "post_tokens=" in caplog.text
    assert "max_tokens=1" in caplog.text
    assert "system_tokens=" in caplog.text
    assert "retained_unit_tokens=" in caplog.text
    assert "cap_hit=true" in caplog.text


def test_truncate_messages_to_fit_preserves_user_query_and_tool_result_pair() -> None:
    messages = [
        {"role": "user", "content": "investigate the memory"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "old result " * 1000,
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "recent result"},
    ]
    expected = [messages[0], *messages[3:]]

    truncated = truncate_messages_to_fit(
        messages,
        max_tokens=count_message_tokens(expected),
    )

    assert truncated == expected


def test_count_message_tokens_includes_openai_tool_call_arguments() -> None:
    short_call = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        }
    ]
    long_call = [
        {
            **short_call[0],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "x " * 1000,
                    },
                }
            ],
        }
    ]

    assert count_message_tokens(long_call) > count_message_tokens(short_call) + 500


def test_is_tool_use_message_detects_gemini_function_call_in_parts() -> None:
    msg: dict[str, Any] = {
        "role": "model",
        "parts": [
            {"function_call": {"name": "search", "args": {"q": "honcho"}}},
        ],
    }
    assert _is_tool_use_message(msg) is True


def test_is_tool_result_message_detects_gemini_function_response_in_parts() -> None:
    msg: dict[str, Any] = {
        "role": "user",
        "parts": [
            {"function_response": {"name": "search", "response": {"result": "ok"}}},
        ],
    }
    assert _is_tool_result_message(msg) is True


def test_is_tool_use_message_detects_anthropic_tool_use_block() -> None:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "calling lookup"},
            {"type": "tool_use", "id": "t_1", "name": "lookup", "input": {}},
        ],
    }
    assert _is_tool_use_message(msg) is True


def test_truncate_messages_to_fit_preserves_gemini_tool_pair() -> None:
    """A Gemini-shaped function_call / function_response pair must stay
    grouped when older units get dropped. Regression: before adding the
    parts-based detection, neither message would be recognized as a tool
    unit, and truncation could split or drop them individually."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "parts": [{"text": "investigate the memory"}]},
        {"role": "model", "parts": [{"text": "old context " * 1000}]},
        {
            "role": "model",
            "parts": [
                {"function_call": {"name": "lookup", "args": {}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "lookup",
                        "response": {"result": "found"},
                    }
                }
            ],
        },
    ]
    expected = [messages[0], *messages[2:]]

    truncated = truncate_messages_to_fit(
        messages,
        max_tokens=count_message_tokens(expected),
    )

    # The old bulk-text unit is dropped while the real user query and complete
    # function_call/function_response pair remain. The function_response uses
    # role=user but must not be mistaken for the query itself.
    assert truncated == expected
