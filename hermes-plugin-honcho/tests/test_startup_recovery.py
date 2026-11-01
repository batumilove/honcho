"""Isolated external-plugin regressions; never imports an installed live profile."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import hermes_plugin_honcho as plugin


def config(home):
    return SimpleNamespace(
        enabled=True,
        api_key="test",
        base_url="https://honcho.invalid",
        raw={},
        host="hermes",
        hermes_home=home,
        recall_mode="tools",
        recall_sync=False,
        first_turn_base_wait=0,
        first_turn_dialectic_wait=0,
        dialectic_depth=1,
        reasoning_level_cap="high",
        query_rewrite=False,
        injection_frequency="every-turn",
        context_cadence=1,
        dialectic_cadence=1,
        dialectic_depth_levels=None,
        reasoning_heuristic=True,
        init_on_session_start=True,
        save_messages=True,
        resolve_session_name=lambda **kw: kw["session_id"],
    )


def test_eager_tools_startup_returns_while_backend_blocked_and_shares_admission(
    tmp_path,
):
    started, release, returned = threading.Event(), threading.Event(), threading.Event()
    calls = []
    providers = [plugin.HonchoMemoryProvider(), plugin.HonchoMemoryProvider()]

    def blocked(self, *args, **kwargs):
        calls.append(self)
        started.set()
        assert release.wait(10)
        self._manager = object()
        self._session_initialized = True

    def initialize():
        for provider in providers:
            provider.initialize("session", hermes_home=str(tmp_path))
        returned.set()

    with (
        patch.object(
            plugin.HonchoClientConfig,
            "from_global_config",
            return_value=config(tmp_path),
        ),
        patch.object(plugin.HonchoMemoryProvider, "_do_session_init", blocked),
    ):
        caller = threading.Thread(target=initialize)
        caller.start()
        try:
            assert started.wait(3)
            assert returned.wait(3), "agent construction waited on backend completion"
            assert len(calls) == 1, (
                "another agent started overlapping initialization for the same profile"
            )
        finally:
            release.set()
            caller.join(5)
            for provider in providers:
                if provider._init_thread:
                    provider._init_thread.join(5)
    assert not caller.is_alive()


def test_unavailable_eager_memory_retains_turn_and_author_in_own_profile(tmp_path):
    provider = plugin.HonchoMemoryProvider()
    provider._config = config(tmp_path / "shopping")
    provider._recall_mode = "tools"
    provider._session_key = "session-one"
    provider._lazy_init_kwargs = {"hermes_home": str(tmp_path / "shopping")}
    provider._lazy_init_session_id = "session-one"
    author = {"id": "human-one", "name": "Alice", "is_bot": False}
    provider.sync_turn("remember this", "okay", turn_author=author)
    author["id"] = "different-user"
    records = list(
        (tmp_path / "shopping" / "state" / "honcho-pending-turns").glob("*.json")
    )
    assert len(records) == 1
    receipt = json.loads(records[0].read_text())
    assert receipt["state"] == "not_submitted"
    assert receipt["session_key"] == "session-one"
    assert receipt["user_content"] == "remember this"
    assert receipt["assistant_content"] == "okay"
    assert receipt["turn_author"]["id"] == "human-one"
    assert records[0].stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / "research").exists()
    provider._config.save_messages = False
    provider.sync_turn("must not retain", "disabled")
    assert len(list(records[0].parent.glob("*.json"))) == 1


def test_lazy_tool_reports_pending_while_initializer_is_running(tmp_path):
    provider = plugin.HonchoMemoryProvider()
    cfg = config(tmp_path)
    cfg.init_on_session_start = False
    release = threading.Event()

    def blocked(self, *args, **kwargs):
        assert release.wait(10)
        self._manager = object()
        self._session_initialized = True

    with (
        patch.object(plugin.HonchoClientConfig, "from_global_config", return_value=cfg),
        patch.object(plugin.HonchoMemoryProvider, "_do_session_init", blocked),
    ):
        provider.initialize("session", hermes_home=str(tmp_path))
        try:
            result = json.loads(
                provider.handle_tool_call("honcho_profile", {"peer": "user"})
            )
            assert "still initializing" in result["error"]
            assert "could not be initialized" not in result["error"]
        finally:
            release.set()
            if provider._init_thread:
                provider._init_thread.join(5)
