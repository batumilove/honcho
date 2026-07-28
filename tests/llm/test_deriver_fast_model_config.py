import logging

import pytest

from src.config import settings
from src.deriver.deriver import (
    _build_fast_deriver_model_config,  # pyright: ignore[reportPrivateUsage]
)


def _configure_fast_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DERIVER_ROUTER_FAST_MODEL", "fast-model")
    monkeypatch.setenv("DERIVER_ROUTER_FAST_BASE_URL", "http://fast.invalid/v1")


def test_fast_deriver_invalid_transport_falls_back_to_primary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _configure_fast_route(monkeypatch)
    monkeypatch.setenv("DERIVER_ROUTER_FAST_TRANSPORT", "invalid-transport")

    with caplog.at_level(logging.WARNING):
        config = _build_fast_deriver_model_config(settings.DERIVER.MODEL_CONFIG)

    assert config is not None
    assert config.transport == settings.DERIVER.MODEL_CONFIG.transport
    assert "Invalid model transport for DERIVER_ROUTER_FAST_TRANSPORT" in caplog.text
