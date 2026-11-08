"""Opaque request correlation at the provider boundary; no conversation data."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit, urlunsplit

PURPOSE_STAGES = {
    "deriver.representation": "deriver",
    "dialectic.answer": "dialectic",
    "dream.deduction": "dream",
    "dream.induction": "dream",
    "summary.short": "summary",
    "summary.long": "summary",
}
STAGES = frozenset({*PURPOSE_STAGES.values(), "embedding", "unattributed"})


def request_headers(
    purpose: str | None = None, *, embedding: bool = False
) -> dict[str, str]:
    """One ID per application attempt; SDK retries retain the correlation ID."""
    purpose = (
        purpose if isinstance(purpose, str) and purpose in PURPOSE_STAGES else None
    )
    headers = {
        "X-Honcho-Request-ID": uuid.uuid4().hex,
        "X-Honcho-Stage": "embedding"
        if embedding
        else PURPOSE_STAGES.get(purpose, "unattributed")
        if purpose
        else "unattributed",
    }
    if purpose is not None:
        headers["X-Honcho-Call-Purpose"] = purpose
    return headers


def validated_headers(value: object) -> dict[str, str]:
    """Never turn provider_params into arbitrary/authentication HTTP headers."""
    if not isinstance(value, dict):
        return {}
    value = cast(dict[str, object], value)
    ident = value.get("X-Honcho-Request-ID")
    stage = value.get("X-Honcho-Stage")
    if (
        not isinstance(ident, str)
        or not re.fullmatch(r"[a-f0-9]{32}", ident)
        or not isinstance(stage, str)
        or stage not in STAGES
    ):
        return {}
    result = {"X-Honcho-Request-ID": ident, "X-Honcho-Stage": stage}
    purpose = value.get("X-Honcho-Call-Purpose")
    if (
        isinstance(purpose, str)
        and purpose in PURPOSE_STAGES
        and PURPOSE_STAGES[purpose] == stage
    ):
        result["X-Honcho-Call-Purpose"] = purpose
    return result


# Optional durable caller-side receipts also cover failures before the proxy can
# receive a request. Enable only with a private, persistent deployment bind.


def _record(
    headers: object,
    model: str,
    endpoint: str,
    *,
    outcome: Literal["success", "error", "cancelled"] | None = None,
    error: BaseException | None = None,
) -> None:
    path = os.environ.get("HONCHO_REQUEST_RECEIPT_PATH")
    safe = validated_headers(headers)
    if not path or not safe:
        return
    try:
        file = Path(path)
        file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(file, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        c = sqlite3.connect(file, timeout=1)
        try:
            c.execute("PRAGMA synchronous=FULL")
            c.execute(
                "CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, started REAL NOT NULL, finished REAL, stage TEXT NOT NULL, purpose TEXT, model TEXT NOT NULL, endpoint TEXT NOT NULL, outcome TEXT, error_class TEXT, http_status INTEGER)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS request_finished ON requests(finished)"
            )
            c.execute("CREATE INDEX IF NOT EXISTS request_started ON requests(started)")
            with c:
                ident = safe["X-Honcho-Request-ID"]
                if outcome is None:
                    u = urlsplit(endpoint)
                    endpoint = urlunsplit(
                        (
                            u.scheme,
                            (u.hostname or "") + (":" + str(u.port) if u.port else ""),
                            "",
                            "",
                            "",
                        )
                    )
                    model = (
                        model
                        if isinstance(model, str)
                        and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", model)
                        else "unknown"
                    )
                    c.execute(
                        "INSERT INTO requests VALUES(?,?,NULL,?,?,?,?,NULL,NULL,NULL)",
                        (
                            ident,
                            time.time(),
                            safe["X-Honcho-Stage"],
                            safe.get("X-Honcho-Call-Purpose"),
                            model,
                            endpoint,
                        ),
                    )
                    c.execute(
                        "DELETE FROM requests WHERE id IN (SELECT id FROM requests WHERE finished < ? LIMIT 500)",
                        (time.time() - 7 * 86400,),
                    )
                else:
                    status = getattr(error, "status_code", None)
                    status = status if type(status) is int else None
                    c.execute(
                        "UPDATE requests SET finished=?,outcome=?,error_class=?,http_status=? WHERE id=?",
                        (
                            time.time(),
                            outcome,
                            type(error).__name__ if error else None,
                            status,
                            ident,
                        ),
                    )
        finally:
            c.close()
    except Exception:
        logging.getLogger(__name__).warning("Provider request receipt unavailable")


@contextmanager
def request_attempt(headers: object, *, model: str, endpoint: str) -> Iterator[None]:
    _record(headers, model, endpoint)
    try:
        yield
    except BaseException as exc:
        outcome = (
            "cancelled"
            if isinstance(
                exc,
                asyncio.CancelledError | GeneratorExit | KeyboardInterrupt | SystemExit,
            )
            else "error"
        )
        _record(headers, model, endpoint, outcome=outcome, error=exc)
        raise
    else:
        _record(headers, model, endpoint, outcome="success")
