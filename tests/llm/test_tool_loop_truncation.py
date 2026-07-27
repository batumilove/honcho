# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""Regression tests for `hit_input_token_cap` propagation through
`execute_tool_loop`.

The toolless path (`src/llm/api.py:325-340`) detects the cap hit up-front
by comparing input tokens against `max_input_tokens`. Before this fix,
the tool-loop path called `truncate_messages_to_fit` per iteration but
never propagated the flag — Dialectic (the main tool-loop consumer)
under-reported `hit_input_token_cap` on dialectic/representation events.

The rule is intentionally token-based, not message-count-based, so the
deriver's single-prompt path (where `truncate_messages_to_fit` keeps the
last unit even when oversized) still surfaces a real cap hit.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.exceptions import ValidationException
from src.llm import tool_loop
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import HonchoLLMCallResponse


def _make_plan() -> AttemptPlan:
    # `selected_config=None` works for these tests since `_call_with_messages`
    # passes it straight through to the mocked `honcho_llm_call_inner`.
    return AttemptPlan(
        provider="anthropic",
        model="claude-sonnet-4-5",
        client=object(),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=None,
        attempt=1,
        retry_attempts=1,
        is_fallback=False,
    )


async def _terminating_call(*_args: Any, **_kwargs: Any) -> HonchoLLMCallResponse[Any]:
    # No tool calls — execute_tool_loop terminates after iteration 1.
    return HonchoLLMCallResponse(
        content="done",
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        finish_reasons=["stop"],
        tool_calls_made=[],
    )


@pytest.mark.asyncio
async def test_hit_input_token_cap_fires_when_input_exceeds_cap():
    """When the input message list exceeds `max_input_tokens` by token
    count, the response carries `hit_input_token_cap=True`. Regression
    check for the rule switch from message-count to token-based.
    """

    # Pretend the conversation totals 200 tokens; cap is 100.
    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_terminating_call),
        patch("src.llm.conversation.count_message_tokens", side_effect=[200, 50]),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            side_effect=lambda msgs, _cap: msgs,
        ),
    ):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "huge"}],
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=lambda _name, _input: "",
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=100,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.hit_input_token_cap is True


@pytest.mark.asyncio
async def test_hit_input_token_cap_false_when_under_cap():
    """Input tokens under cap → flag stays False, no false positive."""

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_terminating_call),
        patch("src.llm.conversation.count_message_tokens", return_value=50),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            side_effect=lambda msgs, _cap: msgs,
        ),
    ):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "small"}],
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=lambda _name, _input: "",
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=100,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.hit_input_token_cap is False


@pytest.mark.asyncio
async def test_tool_loop_rejects_input_that_remains_over_cap_after_truncation():
    """Never forward a locally over-cap payload to the provider."""

    call_count = 0

    async def _must_not_call(*_args: Any, **_kwargs: Any):
        nonlocal call_count
        call_count += 1
        return await _terminating_call(*_args, **_kwargs)

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_must_not_call),
        patch("src.llm.conversation.count_message_tokens", return_value=99_999),
        # Truncate is a no-op (matches real behavior for single-message inputs).
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            side_effect=lambda msgs, _cap: msgs,
        ),
        pytest.raises(ValidationException, match="remains over max_input_tokens"),
    ):
        await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "x" * 1_000_000}],
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=lambda _name, _input: "",
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=1_000,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert call_count == 0


@pytest.mark.asyncio
async def test_tool_loop_restores_user_turn_after_truncation_drops_it():
    """Tool-only retained history must still contain a user query for providers."""

    captured_messages: list[dict[str, Any]] = []

    async def _capture_call(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        captured_messages.extend(kwargs["messages"])
        return await _terminating_call(*_args, **kwargs)

    truncated = [
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
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_capture_call),
        patch("src.llm.conversation.count_message_tokens", side_effect=[200, 50]),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            return_value=truncated,
        ),
    ):
        await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "original query"}],
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=lambda _name, _input: "",
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=100,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    user_messages = [m for m in captured_messages if m.get("role") == "user"]
    assert user_messages == [{"role": "user", "content": "original query"}]


@pytest.mark.asyncio
async def test_tool_loop_does_not_treat_gemini_tool_result_as_user_query():
    """Gemini function responses use role=user but are not genuine user turns."""

    captured_messages: list[dict[str, Any]] = []

    async def _capture_call(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        captured_messages.extend(kwargs["messages"])
        return await _terminating_call(*_args, **kwargs)

    truncated: list[dict[str, Any]] = [
        {
            "role": "model",
            "parts": [{"function_call": {"name": "lookup", "args": {}}}],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "lookup",
                        "response": {"result": "ok"},
                    }
                }
            ],
        },
    ]

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_capture_call),
        patch("src.llm.conversation.count_message_tokens", side_effect=[200, 50]),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            return_value=truncated,
        ),
    ):
        await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "original query"}],
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=lambda _name, _input: "",
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=100,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    genuine_user_messages = [
        message
        for message in captured_messages
        if message.get("role") == "user" and "content" in message
    ]
    assert genuine_user_messages == [{"role": "user", "content": "original query"}]


@pytest.mark.asyncio
async def test_synthesis_rejects_input_that_remains_over_cap_after_truncation():
    """The max-iteration synthesis path must enforce the same hard local cap."""

    call_count = 0

    async def _tool_call_then_forbidden(
        *_args: Any, **_kwargs: Any
    ) -> HonchoLLMCallResponse[Any]:
        nonlocal call_count
        call_count += 1
        return HonchoLLMCallResponse(
            content="",
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            finish_reasons=["tool_use"],
            tool_calls_made=[{"name": "noop", "input": {}, "id": "call_1"}],
        )

    async def _tool_executor(_name: str, _input: dict[str, Any]) -> str:
        return "large tool result"

    with (
        patch.object(
            tool_loop,
            "honcho_llm_call_inner",
            new=_tool_call_then_forbidden,
        ),
        patch(
            "src.llm.conversation.count_message_tokens",
            side_effect=[50, 50, 200, 200],
        ),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            side_effect=lambda msgs, _cap: msgs,
        ),
        pytest.raises(ValidationException, match="remains over max_input_tokens"),
    ):
        await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "original query"}],
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="required",
            tool_executor=_tool_executor,
            max_tool_iterations=1,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=100,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert call_count == 1


@pytest.mark.asyncio
async def test_tool_loop_retruncates_after_adding_continuity_query():
    """A compact fallback query should displace oversized tool-only history."""

    captured_messages: list[dict[str, Any]] = []

    async def _capture_call(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        captured_messages.extend(kwargs["messages"])
        return await _terminating_call(*_args, **kwargs)

    tool_only_history: list[dict[str, Any]] = [
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
        {"role": "tool", "tool_call_id": "call_1", "content": "large result"},
    ]
    continuity_only = [
        {
            "role": "user",
            "content": "Continue the original request using the retained context.",
        }
    ]

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_capture_call),
        patch(
            "src.llm.conversation.count_message_tokens",
            side_effect=[200, 150, 5],
        ),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            side_effect=[tool_only_history, continuity_only],
        ) as truncate_mock,
    ):
        await execute_tool_loop(
            prompt="ignored because messages are provided",
            max_tokens=64,
            messages=[{"role": "assistant", "content": "no user query"}],
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=lambda _name, _input: "",
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=100,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert truncate_mock.call_count == 2
    assert captured_messages == continuity_only
