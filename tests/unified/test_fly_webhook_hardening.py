"""
Fly webhook payloads must be JSON-encoded and secrets must never reach argv.

Private helpers are intentionally imported from the workflow policy suite.
"""
# pyright: reportPrivateUsage=none, reportImplicitRelativeImport=none

import subprocess
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
ENCODER_SCRIPT = (
    Path(".github") / "scripts" / "fly_webhook_payload.py"
)


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
    for mapping in _iter_mapping_nodes(_load_workflow(path)):
        run = _mapping_child(mapping, "run")
        if isinstance(run, yaml.ScalarNode) and "curl --fail" in run.value:
            return run.value, mapping
    raise AssertionError(f"webhook step not found in {path}")


@pytest.mark.parametrize("workflow_name", FLY_DEPLOY_WORKFLOWS)
def test_webhook_json_is_encoded_not_shell_interpolated(workflow_name: str) -> None:
    run, _ = _webhook_run(WORKFLOWS_DIR / workflow_name)

    # The curl step must post the pre-encoded file, not build JSON inline.
    import re as _re

    assert not _re.search(r"(?:^|\s)-d(?:\s|$)", run), (
        f"webhook body must not be assembled inline in {workflow_name}"
    )
    assert "--data-binary @payload.json" in run, (
        f"webhook must post encoded payload.json in {workflow_name}"
    )

    # A dedicated step must encode the payload via the checked-in encoder script.
    encoder = _step_by_name(workflow_name, "Encode webhook JSON")
    run_node = _mapping_child(encoder, "run")
    assert isinstance(run_node, yaml.ScalarNode)
    assert str(ENCODER_SCRIPT) in run_node.value, (
        f"encoder step must run {ENCODER_SCRIPT} in {workflow_name}"
    )
    assert ENCODER_SCRIPT.is_file(), f"missing encoder script {ENCODER_SCRIPT}"
    script = ENCODER_SCRIPT.read_text(encoding="utf-8")
    assert "json.dump" in script, "encoder must use the json module"


@pytest.mark.parametrize(
    ("workflow_name", "secret_name", "url_name"), SECRET_NAMES
)
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
        "WEBHOOK_SECRET": f"${{{{ secrets.{secret_name} }}}}",
        "WEBHOOK_URL": f"${{{{ secrets.{url_name} }}}}",
    }.items():
        node = _mapping_child(encoder_env, name)
        assert isinstance(node, yaml.ScalarNode) and node.value == value, name


@pytest.mark.parametrize(
    ("workflow_name", "secret_name", "url_name"), SECRET_NAMES
)
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
        bound = {
            k.value
            for k, _ in env.value
            if isinstance(k, yaml.ScalarNode)
        }
        missing = read_names - bound
        assert not missing, f"{workflow_name}: script reads unbound env {missing}"
        # Checkout must precede the encoder step so the script exists at runtime.
        doc = _load_workflow(WORKFLOWS_DIR / workflow_name)
        steps = [
            node
            for node in _iter_mapping_nodes(doc)
            if any(
                isinstance((u := _mapping_child(node, "uses")), yaml.ScalarNode)
                and u.value.startswith("actions/checkout@")
                for _ in [0]
            )
        ]
        assert steps, f"{workflow_name}: prompt-service job must check out the repo"


def test_encoder_produces_valid_json_and_argv_free_curl_config() -> None:
    """Run the real encoder with hostile inputs; verify outputs byte-exactly."""
    workdir = WORKFLOWS_DIR.parent.parent
    hostile_env = {
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF_NAME": 'v1.2."3',
        "INPUT_VERSION": "ignored",
        "IMAGE_LABEL_PREFIX": 'img:deployment-"',
        "WEBHOOK_SECRET": 's3cr et"x',
        "WEBHOOK_URL": "https://hook.example",
    }
    completed = subprocess.run(
        ["python3", str(ENCODER_SCRIPT)],
        capture_output=True,
        cwd=workdir,
        env=hostile_env,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    try:
        payload_text = (workdir / "payload.json").read_text()
        dq = chr(34)
        bs = chr(92)
        expected_payload = (
            '{"version": "1.2.'
            + bs
            + dq
            + '3", "image_label": "img:deployment-'
            + bs
            + dq
            + 'v1.2.'
            + bs
            + dq
            + '3"}'
        )
        assert payload_text == expected_payload, payload_text
        config = (workdir / "curl-config").read_text()
        assert config == (
            'header = "Content-Type: application/json"\n'
            'header = "Authorization: Bearer s3cr et\'x"\n'
            'url = "https://hook.example/webhooks/v1/add_honcho_version"\n'
        )
        # curl itself must parse the config without complaint (DNS failure is fine).
        curl = subprocess.run(
            [
                "curl",
                "--fail",
                "-sS",
                "-K",
                str(workdir / "curl-config"),
                "--data-binary",
                f"@{workdir / 'payload.json'}",
                "-o",
                "/dev/null",
                "--max-time",
                "2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert curl.returncode in (0, 6, 7, 28), curl.stderr
    finally:
        for name in ("payload.json", "curl-config"):
            (workdir / name).unlink(missing_ok=True)


def test_encoder_script_has_no_hardcoded_secrets() -> None:
    script = ENCODER_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("flyctl", "FLY_API_TOKEN"):
        assert forbidden not in script, forbidden
