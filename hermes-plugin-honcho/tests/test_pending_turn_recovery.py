"""Recovery records keep their original destination, policy and private body boundaries."""

import hashlib
import importlib
import json
from unittest.mock import patch

import hermes_plugin_honcho as plugin
from test_startup_recovery import config


def test_pending_bot_targets_a2a_session_and_inventory_is_read_only(tmp_path):
    provider = plugin.HonchoMemoryProvider()
    provider._config = config(tmp_path)
    provider._config.workspace_id = "workspace-one"
    provider._config.ai_peer = "recipient-one"
    provider._config.a2a_sessions = True
    provider._recall_mode = "tools"
    provider._session_key = "human-session"
    provider._lazy_init_kwargs = {"hermes_home": str(tmp_path), "user_id": "human-id"}
    author = {"id": "sender-bot", "is_bot": True}
    with patch.object(provider, "_ready_or_kick_init", return_value=False):
        provider.sync_turn(
            "private user body", "private assistant body", turn_author=author
        )
    path = next((tmp_path / "state" / "honcho-pending-turns").glob("*.json"))
    raw = path.read_bytes()
    record = json.loads(raw)
    assert record["session_key"] != provider._session_key
    assert record["session_key"] == provider._a2a_session_key(author)
    assert record["binding"]["workspace_id"] == "workspace-one"
    assert record["binding"]["runtime_user_id"] == "human-id"
    module = importlib.import_module(plugin.__name__ + ".pending_turns")
    report = module.inventory(tmp_path)
    assert report["records"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "private user body" not in json.dumps(report)
    assert "private assistant body" not in json.dumps(report)
    assert path.read_bytes() == raw


def test_inventory_holds_foreign_profile_binding_and_unknown_schema(tmp_path):
    module = importlib.import_module(plugin.__name__ + ".pending_turns")
    directory = tmp_path / "state" / "honcho-pending-turns"
    directory.mkdir(parents=True)
    foreign = {
        "version": 2,
        "binding": {"hermes_home": str(tmp_path / "other-profile")},
        "user_content": "private",
    }
    (directory / "foreign.json").write_text(json.dumps(foreign))
    (directory / "legacy.json").write_text(
        json.dumps({"version": 1, "user_content": "private"})
    )
    report = module.inventory(tmp_path)
    statuses = {r["status"] for r in report["records"]}
    assert statuses == {"profile_binding_mismatch", "legacy_or_unknown_schema"}
    assert "private" not in json.dumps(report)
    assert (directory / "foreign.json").exists()


def test_unconfigured_memory_does_not_create_recovery_records(tmp_path):
    provider = plugin.HonchoMemoryProvider()
    provider._recall_mode = "tools"
    provider._lazy_init_kwargs = {"hermes_home": str(tmp_path)}
    with patch.object(provider, "_ready_or_kick_init", return_value=False):
        provider.sync_turn("not configured", "response")
    assert not (tmp_path / "state" / "honcho-pending-turns").exists()


def test_invalid_profile_path_cannot_create_a_recovery_directory(tmp_path):
    from unittest.mock import MagicMock

    import pytest
    startup = importlib.import_module(plugin.__name__ + ".startup")
    provider = plugin.HonchoMemoryProvider()
    provider._lazy_init_kwargs = {"hermes_home": MagicMock()}
    with pytest.raises(ValueError, match="profile home"):
        startup._acquire_initializer_lock(provider)
    assert not (tmp_path / "state").exists()
