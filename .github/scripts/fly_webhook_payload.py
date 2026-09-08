"""Build private Fly deploy webhook request files without shell interpolation.

Reads GITHUB_EVENT_NAME, GITHUB_REF_NAME, INPUT_VERSION, IMAGE_LABEL_PREFIX,
WEBHOOK_SECRET, and WEBHOOK_URL from the environment. Writes mode-0600
`payload.json` and `curl-config` atomically, so secrets never appear in argv.
"""

import json
import os
import tempfile
from pathlib import Path


def _curl_config_quote(value: str, *, name: str) -> str:
    """Encode one curl-config quoted value, rejecting line injection."""
    if any(char in value for char in ("\r", "\n", "\0")):
        raise ValueError(f"{name} must not contain CR, LF, or NUL")
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _write_private_atomic(path: Path, content: str) -> None:
    """Replace path atomically with a file private from creation onward."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    event = os.environ["GITHUB_EVENT_NAME"]
    if event == "workflow_dispatch":
        version = os.environ["INPUT_VERSION"]
        image_label = os.environ["IMAGE_LABEL_PREFIX"] + version
    else:
        version = os.environ["GITHUB_REF_NAME"].removeprefix("v")
        image_label = os.environ["IMAGE_LABEL_PREFIX"] + os.environ["GITHUB_REF_NAME"]

    secret = _curl_config_quote(os.environ["WEBHOOK_SECRET"], name="WEBHOOK_SECRET")
    base_url = os.environ["WEBHOOK_URL"]
    if not base_url.startswith(("https://", "http://")):
        raise ValueError("WEBHOOK_URL must use http or https")
    url = _curl_config_quote(
        base_url.rstrip("/") + "/webhooks/v1/add_honcho_version",
        name="WEBHOOK_URL",
    )

    payload = json.dumps({"version": version, "image_label": image_label})
    config = (
        'header = "Content-Type: application/json"\n'
        f'header = "Authorization: Bearer {secret}"\n'
        f'url = "{url}"\n'
    )

    _write_private_atomic(Path("payload.json"), payload)
    _write_private_atomic(Path("curl-config"), config)


if __name__ == "__main__":
    main()
