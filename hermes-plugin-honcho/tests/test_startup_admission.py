"""Acceptance gaps found while reviewing the unpublished external-plugin candidate."""

import importlib.util
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import hermes_plugin_honcho as plugin
from test_startup_recovery import config


def test_outage_retention_respects_disabled_bot_ingestion(tmp_path):
    provider = plugin.HonchoMemoryProvider()
    provider._config = config(tmp_path)
    provider._config.a2a_sessions = False
    provider._recall_mode = "tools"
    provider._session_key = "human-session"
    provider._lazy_init_kwargs = {"hermes_home": str(tmp_path)}
    provider.sync_turn(
        "bot turn", "answer", turn_author={"id": "another-bot", "is_bot": True}
    )
    assert not list((tmp_path / "state" / "honcho-pending-turns").glob("*.json"))


def test_initializer_admission_survives_module_generation_change(tmp_path):
    path = Path(plugin.__file__).parent
    name = "isolated_honcho_reloaded_generation"
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    newer = importlib.util.module_from_spec(spec)
    sys.modules[name] = newer
    spec.loader.exec_module(newer)
    release = threading.Event()
    providers = [plugin.HonchoMemoryProvider(), newer.HonchoMemoryProvider()]
    calls = []

    def blocked(self, *args, **kwargs):
        calls.append(self)
        assert release.wait(10)
        self._manager = object()
        self._session_initialized = True

    with (
        patch.object(
            plugin.HonchoClientConfig,
            "from_global_config",
            return_value=config(tmp_path),
        ),
        patch.object(
            newer.HonchoClientConfig,
            "from_global_config",
            return_value=config(tmp_path),
        ),
        patch.object(plugin.HonchoMemoryProvider, "_do_session_init", blocked),
        patch.object(newer.HonchoMemoryProvider, "_do_session_init", blocked),
    ):
        try:
            for provider in providers:
                provider.initialize("session", hermes_home=str(tmp_path))
            assert len(calls) == 1, (
                "new module generation started overlapping backend work"
            )
            child = """import importlib.util, os, sys
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('startup_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
provider = SimpleNamespace(_lazy_init_kwargs={'hermes_home':sys.argv[2]}, _config=None)
fd = module._acquire_initializer_lock(provider)
if fd is None: raise SystemExit(1)
os.close(fd)
"""
            command = [
                sys.executable,
                "-c",
                child,
                str(path / "startup.py"),
                str(tmp_path),
            ]
            assert subprocess.run(command, timeout=5).returncode == 1

        finally:
            release.set()
            for provider in providers:
                if provider._init_thread:
                    provider._init_thread.join(5)

    assert subprocess.run(command, timeout=5).returncode == 0
    assert (tmp_path / "state" / "honcho-init.lock").exists()
