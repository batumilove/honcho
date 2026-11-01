"""Bound startup waiting and retain turns that could not yet reach Honcho.

Python cannot cancel a blocking SDK call. Admission stays held until it really
returns; a timed-out wait never authorizes another initializer for that profile.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)
_STARTUP_WAIT_SECONDS = 0.1


def normalize_profile_home(home):
    """Canonicalize a concrete path in the caller before dispatching any work."""
    if not issubclass(type(home), (str, Path)):
        raise ValueError("Honcho startup requires a concrete profile-home path")
    return Path(home).expanduser().resolve()


def owner_home(provider):
    """Use the captured lifecycle scope, never the process's ambient profile."""
    kwargs = provider._lazy_init_kwargs or {}
    binding = getattr(provider, "_startup_binding", None) or {}
    home = (
        kwargs.get("hermes_home")
        or binding.get("hermes_home")
        or getattr(provider._config, "hermes_home", None)
    )
    if not issubclass(type(home), (str, Path)) or not Path(home).expanduser().is_absolute():
        raise ValueError("Honcho startup requires a captured absolute profile home")
    return Path(home).expanduser().resolve()


def _acquire_initializer_lock(provider):
    """Keep the carrier inode: separate module generations/processes must lock the same file."""
    directory = owner_home(provider) / "state"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(directory / "honcho-init.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            return None
        raise
    return fd


def start_initializer(provider, target, spawn):
    """Admit one actual initializer per profile across module reloads and processes.

    The worker owns the descriptor until target returns. A join timeout neither
    releases admission nor unlinks its carrier, and closing on failure permits recovery.
    """
    fd = _acquire_initializer_lock(provider)
    if fd is None:
        return None

    def run():
        try:
            provider._init_failed = not bool(target())
        finally:
            os.close(fd)

    try:
        provider._init_failed = True
        worker = spawn(run, name="honcho-session-init", owner=provider)
        provider._init_thread = worker
        worker.start()
        return worker
    except BaseException:
        os.close(fd)
        raise


def turn_capture_allowed(provider, author):
    if not author.get("is_bot"):
        return True
    return bool(
        getattr(provider._config, "a2a_sessions", True)
        and str(author.get("id") or "").strip()
    )


def retain_if_requested(provider, user_content, assistant_content, author):
    # Unconfigured and deliberately lazy tools-only providers do not opt into capture.
    if provider._config is None:
        return
    if provider._recall_mode == "tools" and not getattr(
        provider._config, "init_on_session_start", False
    ):
        return
    retain_unsubmitted_turn(provider, user_content, assistant_content, author)


def session_unavailable_text(provider):
    """Report a running initializer as pending after the caller's wait ends."""
    worker = provider._init_thread
    if worker and worker.is_alive():
        return "Honcho session is still initializing; try again shortly."
    if provider._init_auth_failure:
        return f"Honcho memory authentication failed: {provider._init_auth_failure}"
    if provider._init_peer_failure:
        return provider._peer_failure_text()
    return "Honcho session could not be initialized."


def recovery_binding(provider):
    """Capture destinations and identity inputs, never credentials, before startup clears kwargs."""
    cfg, kwargs = provider._config, provider._lazy_init_kwargs or {}
    aliases = getattr(cfg, "user_peer_aliases", {}) or {}
    return {
        "hermes_home": str(owner_home(provider)),
        "workspace_id": getattr(cfg, "workspace_id", None),
        "base_url": getattr(cfg, "base_url", None),
        "host": getattr(cfg, "host", None),
        "peer_name": getattr(cfg, "peer_name", None),
        "ai_peer": getattr(cfg, "ai_peer", None),
        "runtime_user_id": kwargs.get("user_id"),
        "runtime_user_id_alt": kwargs.get("user_id_alt"),
        "peer_aliases_sha256": hashlib.sha256(
            json.dumps(aliases, sort_keys=True).encode()
        ).hexdigest(),
    }


def retain_unsubmitted_turn(provider, user_content, assistant_content, turn_author):
    """Durable recovery evidence, not an acknowledged remote write or an automatic replay queue.

    Reconciliation owns replay: a later process must never blindly duplicate a
    turn that an operator may already have recovered. Files are never expired.
    """
    if not user_content and not assistant_content:
        return
    directory = owner_home(provider) / "state" / "honcho-pending-turns"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    author = dict(turn_author or {})
    session_key = (
        provider._a2a_session_key(author)
        if author.get("is_bot")
        else provider._session_key
    )
    binding = getattr(provider, "_startup_binding", None) or recovery_binding(provider)
    record = {
        "version": 2,
        "state": "not_submitted",
        "created_at": time.time(),
        "session_key": session_key,
        "origin_session_key": provider._session_key,
        "binding": dict(binding),
        "user_content": user_content,
        "assistant_content": assistant_content,
        "turn_author": author,
    }
    # Publish only a complete fsynced record. A crash before rename leaves the
    # private partial as recovery evidence rather than exposing a truncated JSON.
    final = directory / f"{uuid.uuid4().hex}.json"
    partial = final.with_suffix(".partial")
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, final)
    if os.name != "nt":
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    logger.warning(
        "Honcho memory degraded: turn retained locally as not_submitted; reconciliation required (%s)",
        final.name,
    )
