"""Content-free, profile-bound receipts for the completed-turn writer boundary.

A receipt proves only an invocation admitted by sync_turn, not transcript-wide
coverage. Ambiguous writes are never replayed by this module.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
 id TEXT PRIMARY KEY, source_session TEXT NOT NULL, workspace TEXT NOT NULL,
 session_key TEXT NOT NULL, remote_session TEXT, created REAL NOT NULL,
 updated REAL NOT NULL, human INTEGER NOT NULL, expected INTEGER NOT NULL,
 status TEXT NOT NULL, error_class TEXT);
CREATE TABLE IF NOT EXISTS parts (
 turn_id TEXT NOT NULL REFERENCES turns(id), ordinal INTEGER NOT NULL,
 role TEXT NOT NULL, remote_id TEXT, PRIMARY KEY(turn_id, ordinal));
CREATE INDEX IF NOT EXISTS receipt_created ON turns(created);
"""


class ReceiptStore:
    def __init__(self, home: Path):
        self.path = Path(home) / "cache" / "honcho-receipts" / "receipts.sqlite3"

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Exclusive creation preserves private permissions without changing umask
        # process-wide while other gateway threads are writing files.
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        c = sqlite3.connect(self.path, timeout=5)
        try:
            c.execute("PRAGMA synchronous=FULL")
            c.executescript(SCHEMA)
            with c:
                yield c
        finally:
            c.close()

    def begin(self, source_session, workspace, session_key, roles, *, human):
        ident, now = uuid.uuid4().hex, time.time()
        with self.connect() as c:
            c.execute(
                "INSERT INTO turns VALUES(?,?,?,?,?,?,?,?,?,?,NULL)",
                (
                    ident,
                    source_session,
                    workspace,
                    session_key,
                    None,
                    now,
                    now,
                    int(human),
                    len(roles),
                    "pending",
                ),
            )
            c.executemany(
                "INSERT INTO parts VALUES(?,?,?,NULL)",
                [(ident, i, role) for i, role in enumerate(roles)],
            )
            # Finite retention only for fully acknowledged receipts. Outstanding
            # debt is preserved until the owner reconciles it.
            old = time.time() - 7 * 86400
            c.execute(
                "DELETE FROM parts WHERE turn_id IN (SELECT id FROM turns WHERE status='acknowledged' AND created < ?)",
                (old,),
            )
            c.execute(
                "DELETE FROM turns WHERE status='acknowledged' AND created < ?", (old,)
            )
        return ident

    def bind(self, ident, remote_session):
        with self.connect() as c:
            c.execute(
                "UPDATE turns SET remote_session=?,updated=? WHERE id=? AND (remote_session IS NULL OR remote_session=?)",
                (remote_session, time.time(), ident, remote_session),
            )

    def uncertain(self, identifiers, error_class):
        with self.connect() as c:
            c.executemany(
                "UPDATE turns SET status='unknown',error_class=?,updated=? WHERE id=? AND status!='acknowledged'",
                [(error_class[:80], time.time(), i) for i in set(identifiers)],
            )

    def acknowledge(self, identifiers, response):
        """Require echoed metadata and exact workspace/session for each part.

        Returned ordering is not assumed. Missing, mismatched or duplicate parts
        remain unknown; a successful HTTP response alone is insufficient.
        """
        identifiers = set(identifiers)
        with self.connect() as c:
            seen = set()
            for message in response:
                meta = getattr(message, "metadata", None)
                marker = (
                    meta.get("hermes_receipt", {}) if isinstance(meta, dict) else {}
                )
                ident, part = marker.get("id"), marker.get("part")
                if ident not in identifiers or type(part) is not int:
                    continue
                row = c.execute(
                    "SELECT workspace,remote_session FROM turns WHERE id=?", (ident,)
                ).fetchone()
                key = (ident, part)
                if (
                    not row
                    or key in seen
                    or not isinstance(getattr(message, "id", None), str)
                    or not message.id
                    or (message.workspace_id, message.session_id) != row
                ):
                    raise ValueError("receipt_response_identity_mismatch")
                seen.add(key)
                cur = c.execute(
                    "UPDATE parts SET remote_id=? WHERE turn_id=? AND ordinal=?",
                    (message.id, ident, part),
                )
                if cur.rowcount != 1:
                    raise ValueError("receipt_response_part_mismatch")
            for ident in identifiers:
                missing = c.execute(
                    "SELECT count(*) FROM parts WHERE turn_id=? AND remote_id IS NULL",
                    (ident,),
                ).fetchone()[0]
                c.execute(
                    "UPDATE turns SET status=?,error_class=?,updated=? WHERE id=?",
                    (
                        "unknown" if missing else "acknowledged",
                        "receipt_response_incomplete" if missing else None,
                        time.time(),
                        ident,
                    ),
                )
