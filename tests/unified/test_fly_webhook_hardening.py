"""
Fly webhook payloads must be JSON-encoded and secrets must never reach argv.

Private helpers are intentionally imported from the workflow policy suite.
"""
# pyright: reportPrivateUsage=none, reportImplicitRelativeImport=none

import json
import os
import shlex
import stat
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

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
    for name, value in {
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
    }.items():
        node = _mapping_child(encoder_env, name)
        assert isinstance(node, yaml.ScalarNode) and node.value == value, name


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
        jobs = _mapping_child(doc, "jobs")
        assert isinstance(jobs, yaml.MappingNode)
        job = _mapping_child(jobs, "prompt-service")
        assert isinstance(job, yaml.MappingNode)
        steps_node = _mapping_child(job, "steps")
        assert isinstance(steps_node, yaml.SequenceNode)
        steps = [
            step for step in steps_node.value if isinstance(step, yaml.MappingNode)
        ]
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
    ("name", "value"),
    (
        ("WEBHOOK_SECRET", "secret\nheader = injected"),
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


def test_encoder_script_has_no_hardcoded_secrets() -> None:
    script = ENCODER_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("flyctl", "FLY_API_TOKEN"):
        assert forbidden not in script, forbidden
