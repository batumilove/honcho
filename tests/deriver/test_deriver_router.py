import pytest

from src.config import ConfiguredModelSettings
from src.deriver.deriver import (
    _choose_deriver_model_config,  # pyright: ignore[reportPrivateUsage]
)


def test_batch_token_cap_forces_safe_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DERIVER_ROUTER_ENABLED", "true")
    monkeypatch.setenv("DERIVER_ROUTER_FAST_MODEL", "fast-model")
    monkeypatch.setenv("DERIVER_ROUTER_FAST_BASE_URL", "http://fast.example/v1")
    base = ConfiguredModelSettings(transport="openai", model="safe-model")

    chosen, reason = _choose_deriver_model_config(
        base_model_config=base,
        messages=[],
        queued_message_count=1,
        messages_tokens=64,
        prompt_message_tokens=64,
        hit_batch_token_cap=True,
        had_previous_error=False,
    )

    assert chosen is base
    assert reason == "safe:batch-token-cap"
