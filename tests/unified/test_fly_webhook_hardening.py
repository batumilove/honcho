"""
Fly webhook payloads must be JSON-encoded and secrets must never reach argv.

Private helpers are intentionally imported from the workflow policy suite.
"""
# pyright: reportPrivateUsage=none, reportImplicitRelativeImport=none

import importlib.util
import json
import os
import shlex
import stat
import subprocess
import tempfile
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Protocol, cast

import pytest
import yaml
from test_github_actions_pins import (
    FLY_DEPLOY_WORKFLOWS,
    WORKFLOWS_DIR,
    _iter_mapping_nodes,
    _load_workflow,
    _mapping_child,
)

SECRET_NAMES = (
    ("fly-deploy.yml", "TEST_ENV_WEBHOOK_SECRET", "TEST_ENV_URL"),
    ("fly-deploy-prod.yml", "PROD_ENV_WEBHOOK_SECRET", "PROD_ENV_URL"),
)
ENCODER_SCRIPT = Path(".github") / "scripts" / "fly_webhook_payload.py"


class CurlConfigQuote(Protocol):
    def __call__(self, value: str, *, name: str) -> str: ...


def _step_by_name(workflow_name: str, step_name: str) -> yaml.MappingNode:
    """Return the named step mapping from a workflow's prompt-service job."""
    doc = _load_workflow(WORKFLOWS_DIR / workflow_name)
    assert isinstance(doc, yaml.MappingNode)
    jobs = _mapping_child(doc, "jobs")
    assert isinstance(jobs, yaml.MappingNode)
    job = _mapping_child(jobs, "prompt-service")
    assert isinstance(job, yaml.MappingNode)
    steps = _mapping_child(job, "steps")
    assert isinstance(steps, yaml.SequenceNode)
    for step in steps.value:
        if isinstance(step, yaml.MappingNode):
            name = _mapping_child(step, "name")
            if isinstance(name, yaml.ScalarNode) and name.value == step_name:
                return step
    raise AssertionError(f"step {step_name!r} not found in {workflow_name}")


def _webhook_run(path: Path) -> tuple[str, yaml.MappingNode]:
    """Return the curl webhook step's run text and mapping."""
    mapping = _step_by_name(path.name, "Send POST request")
    run = _mapping_child(mapping, "run")
    assert isinstance(run, yaml.ScalarNode)
    return run.value, mapping


@pytest.mark.parametrize("workflow_name", FLY_DEPLOY_WORKFLOWS)
def test_webhook_json_is_encoded_not_shell_interpolated(workflow_name: str) -> None:
    run, _ = _webhook_run(WORKFLOWS_DIR / workflow_name)

    # The curl step must post the pre-encoded file, not build JSON inline.
    import re as _re

    assert not _re.search(r"(?:^|\s)-d(?:\s|$)", run), (
        f"webhook body must not be assembled inline in {workflow_name}"
    )
    commands = [
        line.strip()
        for line in run.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert commands == [
        "curl --fail --silent -K curl-config --data-binary @payload.json -o /dev/null"
    ], f"unexpected webhook command in {workflow_name}: {commands}"
    assert shlex.split(commands[0]) == [
        "curl",
        "--fail",
        "--silent",
        "-K",
        "curl-config",
        "--data-binary",
        "@payload.json",
        "-o",
        "/dev/null",
    ]

    # A dedicated step must encode the payload via the checked-in encoder script.
    encoder = _step_by_name(workflow_name, "Encode webhook JSON")
    run_node = _mapping_child(encoder, "run")
    assert isinstance(run_node, yaml.ScalarNode)
    assert run_node.value.strip() == f"python3 {ENCODER_SCRIPT}"
    assert ENCODER_SCRIPT.is_file(), f"missing encoder script {ENCODER_SCRIPT}"
    script = ENCODER_SCRIPT.read_text(encoding="utf-8")
    assert "json.dump" in script, "encoder must use the json module"


@pytest.mark.parametrize(("workflow_name", "secret_name", "url_name"), SECRET_NAMES)
def test_webhook_secrets_reach_curl_without_command_interpolation(
    workflow_name: str, secret_name: str, url_name: str
) -> None:
    _, mapping = _webhook_run(WORKFLOWS_DIR / workflow_name)
    env = _mapping_child(mapping, "env")
    # The curl step itself must have no env at all: secrets stay in curl-config.
    assert env is None, f"curl step must not carry env in {workflow_name}"

    encoder = _step_by_name(workflow_name, "Encode webhook JSON")
    encoder_env = _mapping_child(encoder, "env")
    assert isinstance(encoder_env, yaml.MappingNode)
    expected_env = {
        "GITHUB_EVENT_NAME": "${{ github.event_name }}",
        "GITHUB_REF_NAME": "${{ github.ref_name }}",
        "INPUT_VERSION": "${{ github.event.inputs.version }}",
        "IMAGE_LABEL_PREFIX": (
            "honcho-prod-image:deployment-"
            if workflow_name == "fly-deploy-prod.yml"
            else "honcho-image:deployment-"
        ),
        "WEBHOOK_SECRET": f"${{{{ secrets.{secret_name} }}}}",
        "WEBHOOK_URL": f"${{{{ secrets.{url_name} }}}}",
    }
    actual_env = {}
    for key, value in encoder_env.value:
        assert isinstance(key, yaml.ScalarNode)
        assert isinstance(value, yaml.ScalarNode)
        assert key.value not in actual_env
        actual_env[key.value] = value.value
    assert actual_env == expected_env


@pytest.mark.parametrize(("workflow_name", "secret_name", "url_name"), SECRET_NAMES)
def test_secret_values_do_not_appear_in_workflow_text(
    workflow_name: str, secret_name: str, url_name: str
) -> None:
    text = (WORKFLOWS_DIR / workflow_name).read_text(encoding="utf-8")
    # Secrets may only be referenced through env bindings, never inline.
    inline = text.split("env:")[-1]
    assert f"secrets.{secret_name}" in inline, workflow_name
    assert f"secrets.{url_name}" in inline, workflow_name
    # No secret expression may appear inside any run block.
    for mapping in _iter_mapping_nodes(_load_workflow(WORKFLOWS_DIR / workflow_name)):
        run = _mapping_child(mapping, "run")
        if isinstance(run, yaml.ScalarNode):
            assert "secrets." not in run.value, (
                f"secret expression leaked into run block in {workflow_name}"
            )


def test_encoder_step_binds_every_env_name_the_script_reads() -> None:
    """Fail closed if the script starts reading an env name the workflow omits."""
    import re as _re

    script = ENCODER_SCRIPT.read_text(encoding="utf-8")
    read_names = set(_re.findall(r'os\.environ\["([A-Z_]+)"\]', script))
    assert "GITHUB_EVENT_NAME" in read_names
    for workflow_name in FLY_DEPLOY_WORKFLOWS:
        encoder = _step_by_name(workflow_name, "Encode webhook JSON")
        env = _mapping_child(encoder, "env")
        assert isinstance(env, yaml.MappingNode)
        bound = {k.value for k, _ in env.value if isinstance(k, yaml.ScalarNode)}
        missing = read_names - bound
        assert not missing, f"{workflow_name}: script reads unbound env {missing}"
        # Checkout must precede the encoder step so the script exists at runtime.
        doc = _load_workflow(WORKFLOWS_DIR / workflow_name)
        assert isinstance(doc, yaml.MappingNode)
        assert _mapping_child(doc, "env") is None
        jobs = _mapping_child(doc, "jobs")
        assert isinstance(jobs, yaml.MappingNode)
        job = _mapping_child(jobs, "prompt-service")
        assert isinstance(job, yaml.MappingNode)
        job_keys = [
            key.value for key, _ in job.value if isinstance(key, yaml.ScalarNode)
        ]
        assert job_keys == ["name", "needs", "runs-on", "steps"]
        assert _mapping_child(job, "env") is None
        expected_deploy_job = (
            "deploy-honcho-prod-image"
            if workflow_name == "fly-deploy-prod.yml"
            else "deploy-honcho-image"
        )
        needs = _mapping_child(job, "needs")
        runs_on = _mapping_child(job, "runs-on")
        assert isinstance(needs, yaml.ScalarNode) and needs.value == expected_deploy_job
        assert isinstance(runs_on, yaml.ScalarNode) and runs_on.value == "ubuntu-latest"
        steps_node = _mapping_child(job, "steps")
        assert isinstance(steps_node, yaml.SequenceNode)
        assert len(steps_node.value) == 3
        assert all(isinstance(step, yaml.MappingNode) for step in steps_node.value)
        steps = cast(list[yaml.MappingNode], steps_node.value)
        encoder_indices: list[int] = []
        for index, step in enumerate(steps):
            name = _mapping_child(step, "name")
            if (
                isinstance(name, yaml.ScalarNode)
                and name.value == "Encode webhook JSON"
            ):
                encoder_indices.append(index)
        assert encoder_indices == [1], f"{workflow_name}: encoder must be step 2"
        checkout = steps[0]
        uses = _mapping_child(checkout, "uses")
        assert isinstance(uses, yaml.ScalarNode)
        assert uses.value == (
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
        )
        checkout_with = _mapping_child(checkout, "with")
        assert isinstance(checkout_with, yaml.MappingNode)
        persist = _mapping_child(checkout_with, "persist-credentials")
        assert isinstance(persist, yaml.ScalarNode) and persist.value == "false"
        send_name = _mapping_child(steps[2], "name")
        assert isinstance(send_name, yaml.ScalarNode)
        assert send_name.value == "Send POST request"


@pytest.mark.parametrize("workflow_name", FLY_DEPLOY_WORKFLOWS)
def test_fly_workflow_triggers_and_permissions_are_exact(workflow_name: str) -> None:
    doc = _load_workflow(WORKFLOWS_DIR / workflow_name)
    assert isinstance(doc, yaml.MappingNode)
    permissions = _mapping_child(doc, "permissions")
    assert isinstance(permissions, yaml.MappingNode)
    assert [
        (key.value, value.value)
        for key, value in permissions.value
        if isinstance(key, yaml.ScalarNode) and isinstance(value, yaml.ScalarNode)
    ] == [("contents", "read")]

    triggers = _mapping_child(doc, "on")
    assert isinstance(triggers, yaml.MappingNode)
    assert [
        key.value for key, _ in triggers.value if isinstance(key, yaml.ScalarNode)
    ] == [
        "push",
        "workflow_dispatch",
    ]
    push = _mapping_child(triggers, "push")
    dispatch = _mapping_child(triggers, "workflow_dispatch")
    assert isinstance(push, yaml.MappingNode)
    assert isinstance(dispatch, yaml.MappingNode)
    tags = _mapping_child(push, "tags")
    assert isinstance(tags, yaml.SequenceNode)
    assert [tag.value for tag in tags.value if isinstance(tag, yaml.ScalarNode)] == [
        "v*"
    ]
    inputs = _mapping_child(dispatch, "inputs")
    assert isinstance(inputs, yaml.MappingNode)
    version = _mapping_child(inputs, "version")
    assert isinstance(version, yaml.MappingNode)
    assert [
        (key.value, value.value)
        for key, value in version.value
        if isinstance(key, yaml.ScalarNode) and isinstance(value, yaml.ScalarNode)
    ] == [
        ("description", "Version to deploy (without v prefix)"),
        ("required", "true"),
        ("type", "string"),
        ("default", "manual"),
    ]


def test_encoder_produces_exact_private_files_and_curl_sends_exact_bytes(
    tmp_path: Path,
) -> None:
    """Exercise the real encoder and curl against a local capture server."""
    captured: dict[str, bytes | str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            captured["path"] = self.path
            captured["authorization"] = self.headers["Authorization"]
            captured["body"] = self.rfile.read(length)
            self.send_response(204)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    secret = 's3cr et"x\\tail'
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF_NAME": 'v1.2."3',
        "INPUT_VERSION": "ignored",
        "IMAGE_LABEL_PREFIX": 'img:deployment-"',
        "WEBHOOK_SECRET": secret,
        "WEBHOOK_URL": f"http://127.0.0.1:{server.server_port}",
    }
    completed = subprocess.run(
        ["python3", str(ENCODER_SCRIPT.resolve())],
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = tmp_path / "payload.json"
    config_path = tmp_path / "curl-config"
    expected_payload = json.dumps(
        {
            "version": '1.2."3',
            "image_label": 'img:deployment-"v1.2."3',
        }
    ).encode()
    assert payload.read_bytes() == expected_payload
    expected_config = (
        'header = "Content-Type: application/json"\n'
        'header = "Authorization: Bearer s3cr et\\"x\\\\tail"\n'
        f'url = "http://127.0.0.1:{server.server_port}/webhooks/v1/add_honcho_version"\n'
    ).encode()
    assert config_path.read_bytes() == expected_config
    assert stat.S_IMODE(payload.stat().st_mode) == 0o600
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600

    curl = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "-K",
            str(config_path),
            "--data-binary",
            f"@{payload}",
            "-o",
            "/dev/null",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    thread.join(timeout=2)
    server.server_close()
    assert curl.returncode == 0, curl.stderr
    assert captured == {
        "path": "/webhooks/v1/add_honcho_version",
        "authorization": f"Bearer {secret}",
        "body": expected_payload,
    }


@pytest.mark.parametrize(
    (
        "workflow_name",
        "event",
        "ref",
        "input_version",
        "expected_version",
        "expected_label",
    ),
    (
        (
            "fly-deploy.yml",
            "push",
            "v2.3.4",
            "ignored",
            "2.3.4",
            "honcho-image:deployment-v2.3.4",
        ),
        (
            "fly-deploy.yml",
            "workflow_dispatch",
            "main",
            "2.3.4",
            "2.3.4",
            "honcho-image:deployment-2.3.4",
        ),
        (
            "fly-deploy-prod.yml",
            "push",
            "v2.3.4",
            "ignored",
            "2.3.4",
            "honcho-prod-image:deployment-v2.3.4",
        ),
        (
            "fly-deploy-prod.yml",
            "workflow_dispatch",
            "main",
            "2.3.4",
            "2.3.4",
            "honcho-prod-image:deployment-2.3.4",
        ),
    ),
)
def test_encoder_preserves_trigger_and_environment_label_semantics(
    tmp_path: Path,
    workflow_name: str,
    event: str,
    ref: str,
    input_version: str,
    expected_version: str,
    expected_label: str,
) -> None:
    prefix = (
        "honcho-prod-image:deployment-"
        if workflow_name == "fly-deploy-prod.yml"
        else "honcho-image:deployment-"
    )
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": event,
        "GITHUB_REF_NAME": ref,
        "INPUT_VERSION": input_version,
        "IMAGE_LABEL_PREFIX": prefix,
        "WEBHOOK_SECRET": "secret",
        "WEBHOOK_URL": "https://hook.example",
    }
    completed = subprocess.run(
        ["python3", str(ENCODER_SCRIPT.resolve())],
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "payload.json").read_bytes() == json.dumps(
        {"version": expected_version, "image_label": expected_label}
    ).encode()


def test_private_writer_is_private_before_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location("fly_webhook_payload", ENCODER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    writer = cast(Callable[[Path, str], None], module._write_private_atomic)

    original_mkstemp = tempfile.mkstemp
    original_replace = os.replace
    events: list[str] = []

    def checked_mkstemp(*, prefix: str, dir: Path) -> tuple[int, str]:
        fd, name = original_mkstemp(prefix=prefix, dir=dir)
        assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o600
        events.append("private-created")
        return fd, name

    def checked_replace(source: str | Path, destination: str | Path) -> None:
        assert events == ["private-created"]
        assert stat.S_IMODE(Path(source).stat().st_mode) == 0o600
        events.append("atomic-replace")
        original_replace(source, destination)

    monkeypatch.setattr(tempfile, "mkstemp", checked_mkstemp)
    monkeypatch.setattr(os, "replace", checked_replace)
    target = tmp_path / "sealed"
    writer(target, "secret")
    assert events == ["private-created", "atomic-replace"]
    assert target.read_text() == "secret"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_encoder_escapes_url_quotes_and_backslashes(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF_NAME": "v1",
        "INPUT_VERSION": "ignored",
        "IMAGE_LABEL_PREFIX": "img:deployment-",
        "WEBHOOK_SECRET": "secret",
        "WEBHOOK_URL": 'https://hook.example/base"\\tail',
    }
    completed = subprocess.run(
        ["python3", str(ENCODER_SCRIPT.resolve())],
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "curl-config").read_bytes() == (
        b'header = "Content-Type: application/json"\n'
        b'header = "Authorization: Bearer secret"\n'
        b'url = "https://hook.example/base\\"\\\\tail/webhooks/v1/add_honcho_version"\n'
    )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("WEBHOOK_SECRET", "secret\nheader = injected"),
        ("WEBHOOK_SECRET", "secret\rheader = injected"),
        ("WEBHOOK_URL", "https://host\nurl = injected"),
        ("WEBHOOK_URL", "https://host\rurl = injected"),
    ),
)
def test_encoder_rejects_config_line_injection(
    tmp_path: Path, name: str, value: str
) -> None:
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF_NAME": "v1",
        "INPUT_VERSION": "ignored",
        "IMAGE_LABEL_PREFIX": "img:deployment-",
        "WEBHOOK_SECRET": "secret",
        "WEBHOOK_URL": "https://hook.example",
        name: value,
    }
    completed = subprocess.run(
        ["python3", str(ENCODER_SCRIPT.resolve())],
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert not (tmp_path / "payload.json").exists()
    assert not (tmp_path / "curl-config").exists()


@pytest.mark.parametrize("name", ("WEBHOOK_SECRET", "WEBHOOK_URL"))
def test_encoder_rejects_nul_config_injection(name: str) -> None:
    spec = importlib.util.spec_from_file_location("fly_webhook_payload", ENCODER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    quote = cast(CurlConfigQuote, module._curl_config_quote)
    with pytest.raises(ValueError, match="CR, LF, or NUL"):
        quote("value\0directive", name=name)


def test_encoder_script_has_no_hardcoded_secrets() -> None:
    script = ENCODER_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("flyctl", "FLY_API_TOKEN"):
        assert forbidden not in script, forbidden
