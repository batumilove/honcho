from unittest.mock import AsyncMock, patch

import pytest

from src.config import ConfiguredModelSettings
from src.exceptions import ValidationException
from src.llm.api import honcho_llm_call
from src.llm.types import HonchoLLMCallResponse


@pytest.mark.asyncio
async def test_toolless_call_rejects_over_cap_input_before_provider_dispatch() -> None:
    provider_call = AsyncMock(
        return_value=HonchoLLMCallResponse(
            content="unexpected",
            input_tokens=1,
            output_tokens=1,
            finish_reasons=["stop"],
        )
    )

    with (
        patch("src.llm.api.honcho_llm_call_inner", provider_call),
        patch("src.llm.runtime.client_for_model_config", return_value=object()),
        pytest.raises(ValidationException, match="remains over max_tokens"),
    ):
        await honcho_llm_call(
            model_config=ConfiguredModelSettings(
                model="test-model",
                transport="openai",
            ),
            prompt="oversized " * 2000,
            max_tokens=100,
            max_input_tokens=1,
            enable_retry=False,
        )

    provider_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_toolless_call_rechecks_truncation_result_before_provider_dispatch() -> (
    None
):
    provider_call = AsyncMock(
        return_value=HonchoLLMCallResponse(
            content="unexpected",
            input_tokens=1,
            output_tokens=1,
            finish_reasons=["stop"],
        )
    )
    oversized_messages = [{"role": "user", "content": "oversized " * 2000}]

    with (
        patch("src.llm.api.honcho_llm_call_inner", provider_call),
        patch("src.llm.runtime.client_for_model_config", return_value=object()),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            return_value=oversized_messages,
        ),
        pytest.raises(ValidationException, match="Tool-less input remains over"),
    ):
        await honcho_llm_call(
            model_config=ConfiguredModelSettings(
                model="test-model",
                transport="openai",
            ),
            prompt="ignored",
            messages=oversized_messages,
            max_tokens=100,
            max_input_tokens=1,
            enable_retry=False,
        )

    provider_call.assert_not_awaited()
