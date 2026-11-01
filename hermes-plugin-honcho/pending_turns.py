"""Read-only inventory of unavailable-memory records. Never contacts Honcho or replays data."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

_MAX_RECORD_BYTES = 8 * 1024 * 1024


def _summary(path: Path, home: Path) -> dict:
    result = {"file": path.name}
    if path.is_symlink() or not path.is_file():
        return {**result, "status": "unsafe_file_type"}
    if path.stat().st_size > _MAX_RECORD_BYTES:
        return {**result, "status": "oversized_for_inventory"}
    raw = path.read_bytes()
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        record = json.loads(raw)
    except (ValueError, UnicodeError):
        return {**result, "status": "invalid_json"}
    if not isinstance(record, dict) or record.get("version") != 2:
        return {**result, "status": "legacy_or_unknown_schema"}
    binding = record.get("binding")
    if not isinstance(binding, dict) or binding.get("hermes_home") != str(home):
        return {**result, "status": "profile_binding_mismatch"}
    author = record.get("turn_author")
    if (
        not isinstance(author, dict)
        or record.get("state") != "not_submitted"
        or not all(
            isinstance(record.get(k), str)
            for k in ("user_content", "assistant_content", "session_key")
        )
    ):
        return {**result, "status": "invalid_record_shape"}
    return {
        **result,
        "status": "review_required",
        "state": record.get("state"),
        "binding": binding,
        "session_key": record.get("session_key"),
        "bot_authored": bool(author.get("is_bot")),
        "user_chars": len(record.get("user_content") or ""),
        "assistant_chars": len(record.get("assistant_content") or ""),
    }


def inventory(home: Path, limit: int = 50) -> dict:
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    home = Path(home).expanduser().resolve()
    directory = home / "state" / "honcho-pending-turns"
    if directory.is_symlink():
        raise ValueError("pending-turn directory must not be a symlink")
    paths = list(itertools.islice(directory.glob("*.json"), limit + 1))
    return {
        "hermes_home": str(home),
        "truncated": len(paths) > limit,
        "records": [_summary(path, home) for path in paths[:limit]],
        "partial_files": [
            path.name for path in itertools.islice(directory.glob("*.partial"), limit)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home", required=True, type=Path, help="explicit owning profile home"
    )
    parser.add_argument("--limit", default=50, type=int)
    args = parser.parse_args()
    print(json.dumps(inventory(args.home, args.limit), indent=2))


if __name__ == "__main__":
    main()
