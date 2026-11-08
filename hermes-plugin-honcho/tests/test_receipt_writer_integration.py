"""Exercise the real Hermes loader and writer path with no credentials/network."""

import socket
import sqlite3
from pathlib import Path
from types import SimpleNamespace


def test_loader_and_writer_acknowledge_content_free_parts(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))

    def no_network(*a, **k):
        raise AssertionError("network forbidden in receipt tests")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    from hermes_cli.plugins import PluginManager
    from hermes_cli.plugins_manifest import PluginManifest

    manager = PluginManager(scope_key=str(tmp_path))
    manifest = PluginManifest(
        name="honcho", path=str(Path(__file__).parents[1]), source="user"
    )
    module = manager._load_directory_module(manifest)
    assert module.__name__.startswith("hermes_plugins.honcho")
    from importlib import import_module

    sessions = import_module(module.__name__ + ".session")
    cfg = SimpleNamespace(
        hermes_home=tmp_path / "profile",
        workspace_id="workspace",
        message_max_chars=25000,
    )
    cache = sessions.HonchoSession("key", "user", "assistant", "remote-session")
    sm = sessions.HonchoSessionManager()
    sm._config = cfg
    sm.get_or_create = lambda *a, **k: cache
    sm.resolve_author_peer_id = lambda *a, **k: None
    sm._authed_call = lambda op, fn: fn()

    class Peer:
        def message(self, content, **kw):
            return SimpleNamespace(content=content, **kw)

    sm._get_or_create_peer = lambda *a: Peer()

    def add(messages):
        return [
            SimpleNamespace(
                id="m" + str(i),
                workspace_id="workspace",
                session_id="remote-session",
                metadata=m.metadata,
            )
            for i, m in enumerate(messages)
        ]

    sm._sessions_cache["remote-session"] = SimpleNamespace(add_messages=add)
    sm.save = lambda cache: sm._flush_session(cache)
    p = module.HonchoMemoryProvider.__new__(module.HonchoMemoryProvider)
    p._config = cfg
    p._manager = sm
    p._session_key = "key"
    p._turn_author = {"is_bot": False}
    p._writes_enabled = lambda: True
    p._ready_or_kick_init = lambda: True
    p._sync_thread = None
    p._spawn_write = lambda fn, *args: fn()
    p.sync_turn(
        "SENSITIVE USER TEXT", "SENSITIVE ASSISTANT TEXT", session_id="local-session"
    )
    db = cfg.hermes_home / "cache/honcho-receipts/receipts.sqlite3"
    with sqlite3.connect(db) as c:
        assert c.execute(
            "SELECT source_session,human,expected,status FROM turns"
        ).fetchone() == ("local-session", 1, 2, "acknowledged")
        assert (
            c.execute(
                "SELECT count(*) FROM parts WHERE remote_id IS NOT NULL"
            ).fetchone()[0]
            == 2
        )
    assert b"SENSITIVE" not in db.read_bytes()
