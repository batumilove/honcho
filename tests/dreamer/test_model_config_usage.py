import json
from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.dreamer.specialists import DeductionSpecialist, InductionSpecialist
from src.llm import HonchoLLMCallResponse
from src.utils.tokens import estimate_tokens


def test_deduction_prompt_uses_identity_markers_framing() -> None:
    """Deduction prompt must frame the peer card as an identity store with the
    entity-agnostic prefix taxonomy, not as a human bio sheet."""
    prompt = DeductionSpecialist().build_system_prompt("alice", peer_card_enabled=True)

    assert "identity store" in prompt
    assert "stable identity markers" in prompt
    for prefix in ("IDENTITY:", "ATTRIBUTE:", "RELATIONSHIP:", "INSTRUCTION:"):
        assert prefix in prompt
    # Cross-entity examples confirm the prompt is not biased toward humans.
    assert "codebase" in prompt
    assert "team" in prompt
    # Behavioral content must be explicitly excluded.
    assert "TRAIT:" in prompt
    # The old human-shaped REQUIRED enumeration must be gone.
    assert "Family members and relationships" not in prompt
    assert "Core preferences and traits" not in prompt


def test_deduction_prompt_omits_peer_card_when_disabled() -> None:
    prompt = DeductionSpecialist().build_system_prompt("alice", peer_card_enabled=False)
    assert "PEER CARD" not in prompt
    assert "IDENTITY:" not in prompt


def test_induction_prompt_has_no_peer_card_section() -> None:
    """Induction no longer writes to the peer card; its prompt must not reference it."""
    prompt = InductionSpecialist().build_system_prompt("alice", peer_card_enabled=True)
    assert "PEER CARD" not in prompt
    assert "update_peer_card" not in prompt


def test_induction_specialist_cannot_update_peer_card() -> None:
    """Induction must have can_update_peer_card=False and no update_peer_card tool."""
    specialist = InductionSpecialist()
    assert specialist.can_update_peer_card is False

    tool_names = {t["name"] for t in specialist.get_tools()}
    assert "update_peer_card" not in tool_names
    # Sanity: induction still has the discovery and create tools it actually needs.
    assert "create_observations_inductive" in tool_names
    assert "search_memory" in tool_names


def test_deduction_specialist_can_update_peer_card() -> None:
    specialist = DeductionSpecialist()
    assert specialist.can_update_peer_card is True

    tool_names = {t["name"] for t in specialist.get_tools(peer_card_enabled=True)}
    assert "update_peer_card" in tool_names

    disabled_names = {t["name"] for t in specialist.get_tools(peer_card_enabled=False)}
    assert "update_peer_card" not in disabled_names


@pytest.mark.asyncio
async def test_deduction_specialist_uses_nested_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.METRICS, "ENABLED", False)
    specialist = DeductionSpecialist()
    mock_response = HonchoLLMCallResponse(
        content="done",
        input_tokens=10,
        output_tokens=5,
        finish_reasons=["stop"],
    )

    with (
        patch(
            "src.dreamer.specialists.crud.get_peer",
            new=AsyncMock(),
        ),
        patch(
            "src.dreamer.specialists.crud.get_peer_card",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "src.dreamer.specialists.create_tool_executor",
            new=AsyncMock(return_value=AsyncMock()),
        ),
        patch(
            "src.dreamer.specialists.honcho_llm_call",
            new=AsyncMock(return_value=mock_response),
        ) as mock_llm_call,
    ):
        result = await specialist.run(
            workspace_name="workspace",
            observer="alice",
            observed="alice",
            session_name="session",
        )

    await_args = mock_llm_call.await_args
    if await_args is None:
        raise AssertionError("Expected dreamer LLM call")
    kwargs = await_args.kwargs
    expected_config = settings.DREAM.DEDUCTION_MODEL_CONFIG
    expected_output_tokens = (
        expected_config.max_output_tokens or specialist.get_max_tokens()
    )
    tools = specialist.get_tools(peer_card_enabled=True)
    tool_definition_tokens = estimate_tokens(
        json.dumps(tools, sort_keys=True, separators=(",", ":"))
    )
    expected_input_tokens = min(
        settings.DREAM.MAX_INPUT_TOKENS,
        settings.DREAM.CONTEXT_WINDOW_TOKENS
        - expected_output_tokens
        - settings.DREAM.CONTEXT_SAFETY_MARGIN_TOKENS
        - tool_definition_tokens,
    )

    assert result.content == "done"
    assert kwargs["model_config"] == expected_config
    assert kwargs["max_input_tokens"] == expected_input_tokens
    assert "llm_settings" not in kwargs


@pytest.mark.asyncio
async def test_induction_specialist_reserves_tool_definitions_from_input_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.METRICS, "ENABLED", False)
    specialist = InductionSpecialist()
    mock_response = HonchoLLMCallResponse(
        content="done",
        input_tokens=10,
        output_tokens=5,
        finish_reasons=["stop"],
    )

    with (
        patch("src.dreamer.specialists.crud.get_peer", new=AsyncMock()),
        patch(
            "src.dreamer.specialists.create_tool_executor",
            new=AsyncMock(return_value=AsyncMock()),
        ),
        patch(
            "src.dreamer.specialists.honcho_llm_call",
            new=AsyncMock(return_value=mock_response),
        ) as mock_llm_call,
    ):
        await specialist.run(
            workspace_name="workspace",
            observer="alice",
            observed="alice",
            session_name="session",
        )

    await_args = mock_llm_call.await_args
    if await_args is None:
        raise AssertionError("Expected dreamer LLM call")
    kwargs = await_args.kwargs
    tools = specialist.get_tools()
    tool_definition_tokens = estimate_tokens(
        json.dumps(tools, sort_keys=True, separators=(",", ":"))
    )
    configured_output_tokens = settings.DREAM.INDUCTION_MODEL_CONFIG.max_output_tokens
    expected_output_tokens = configured_output_tokens or specialist.get_max_tokens()
    expected_input_tokens = min(
        settings.DREAM.MAX_INPUT_TOKENS,
        settings.DREAM.CONTEXT_WINDOW_TOKENS
        - expected_output_tokens
        - settings.DREAM.CONTEXT_SAFETY_MARGIN_TOKENS
        - tool_definition_tokens,
    )

    assert tool_definition_tokens > 0
    assert kwargs["tools"] == tools
    assert kwargs["max_input_tokens"] == expected_input_tokens
