from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import BadRequestError
from tenacity import RetryError, wait_none

from src.config import ConfiguredModelSettings, ModelConfig
from src.llm import tool_loop
from src.llm.api import honcho_llm_call
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import HonchoLLMCallResponse, ProviderClient


def _bad_request(
    *, message: str, param: str | None = None, code: Any = 400
) -> BadRequestError:
    body = {
        "error": {
            "message": message,
            "type": "BadRequestError",
            "param": param,
            "code": code,
        }
    }
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(400, request=request, json=body)
    return BadRequestError(message, response=response, body=body)


def _context_length_error() -> BadRequestError:
    return _bad_request(
        message=(
            "This model's maximum context length is 32768 tokens. However, you "
            "requested 8192 output tokens and your prompt contains at least 24577 "
            "input tokens, for a total of at least 32769 tokens."
        ),
        param="input_tokens",
    )


def _plan() -> AttemptPlan:
    return AttemptPlan(
        provider="openai",
        model="test-model",
        client=cast(ProviderClient, object()),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=ModelConfig(model="test-model", transport="openai"),
        attempt=1,
        retry_attempts=3,
        is_fallback=False,
    )


@pytest.mark.asyncio
async def test_toolless_context_length_bad_request_is_not_retried() -> None:
    async def raise_context_error(*_args: Any, **_kwargs: Any) -> None:
        raise _context_length_error()

    provider_call = AsyncMock(side_effect=raise_context_error)

    with (
        patch("src.llm.api.honcho_llm_call_inner", provider_call),
        patch("src.llm.runtime.client_for_model_config", return_value=object()),
        patch("src.llm.api.wait_exponential", return_value=wait_none()),
        pytest.raises(BadRequestError, match="maximum context length"),
    ):
        await honcho_llm_call(
            model_config=ConfiguredModelSettings(
                model="test-model", transport="openai"
            ),
            prompt="hello",
            max_tokens=8192,
            enable_retry=True,
            retry_attempts=3,
        )

    assert provider_call.await_count == 1


@pytest.mark.asyncio
async def test_synthesis_context_length_bad_request_is_not_retried() -> None:
    calls = 0

    async def provider_call(*_args: Any, **_kwargs: Any) -> HonchoLLMCallResponse[Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return HonchoLLMCallResponse(
                content="",
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                finish_reasons=["tool_use"],
                tool_calls_made=[{"name": "noop", "input": {}, "id": "call_1"}],
            )
        raise _context_length_error()

    async def execute_noop(_name: str, _input: dict[str, Any]) -> str:
        return "large result"

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=provider_call),
        patch("src.llm.tool_loop.wait_exponential", return_value=wait_none()),
        pytest.raises(BadRequestError, match="maximum context length"),
    ):
        await execute_tool_loop(
            prompt="hello",
            max_tokens=8192,
            messages=None,
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="required",
            tool_executor=execute_noop,
            max_tool_iterations=1,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=True,
            retry_attempts=3,
            max_input_tokens=None,
            get_attempt_plan=_plan,
            before_retry_callback=lambda _state: None,
        )

    assert calls == 2


@pytest.mark.asyncio
async def test_other_bad_request_remains_eligible_for_retry() -> None:
    async def raise_other_bad_request(*_args: Any, **_kwargs: Any) -> None:
        raise _bad_request(message="Unsupported parameter", param="temperature")

    provider_call = AsyncMock(side_effect=raise_other_bad_request)

    with (
        patch("src.llm.api.honcho_llm_call_inner", provider_call),
        patch("src.llm.runtime.client_for_model_config", return_value=object()),
        patch("src.llm.api.wait_exponential", return_value=wait_none()),
        pytest.raises(RetryError),
    ):
        await honcho_llm_call(
            model_config=ConfiguredModelSettings(
                model="test-model", transport="openai"
            ),
            prompt="hello",
            max_tokens=100,
            enable_retry=True,
            retry_attempts=3,
        )

    assert provider_call.await_count == 3
