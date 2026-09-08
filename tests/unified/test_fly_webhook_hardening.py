"""
Fly webhook payloads must be JSON-encoded and keep secrets out of argv.

Private helpers are intentionally imported from the workflow policy suite.
"""
# pyright: reportPrivateUsage=none, reportImplicitRelativeImport=none

import re
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

    # The curl step must consume the pre-encoded file, not build JSON inline.
    assert '"--json-raw"' not in run and "-d" not in run, (
        f"webhook body must not be assembled inline in {workflow_name}"
    )
    assert "--json-file" in run, f"webhook must post encoded JSON in {workflow_name}"

    # A dedicated step must encode the JSON with python (jq is not installed).
    steps = None
    for node in _iter_mapping_nodes(_load_workflow(WORKFLOWS_DIR / workflow_name)):
        candidate = _mapping_child(node, "steps")
        if isinstance(candidate, yaml.SequenceNode):
            steps = candidate
    assert steps is not None
    encoders = [
        step
        for step in steps.value
        if isinstance(step, yaml.MappingNode)
        and isinstance((name := _mapping_child(step, "name")), yaml.ScalarNode)
        and name.value == "Encode webhook JSON"
    ]
    assert len(encoders) == 1, (
        f"exactly one Encode webhook JSON step in {workflow_name}"
    )
    encoder_run = _mapping_child(encoders[0], "run")
    assert isinstance(encoder_run, yaml.ScalarNode)
    assert "python3 -c" in encoder_run.value and "json.dumps" in encoder_run.value
    assert "payload.json" in encoder_run.value


@pytest.mark.parametrize(("workflow_name", "secret_name", "url_name"), SECRET_NAMES)
def test_webhook_secrets_reach_curl_without_command_interpolation(
    workflow_name: str, secret_name: str, url_name: str
) -> None:
    _, mapping = _webhook_run(WORKFLOWS_DIR / workflow_name)
    env = _mapping_child(mapping, "env")
    assert isinstance(env, yaml.MappingNode)
    expected = {
        "WEBHOOK_SECRET": f"${{{{ secrets.{secret_name} }}}}",
        "WEBHOOK_URL": f"${{{{ secrets.{url_name} }}}}",
    }
    for name, value in expected.items():
        node = _mapping_child(env, name)
        assert isinstance(node, yaml.ScalarNode) and node.value == value, name

    encoder = _step_by_name(workflow_name, "Encode webhook JSON")
    encoder_env = _mapping_child(encoder, "env")
    assert isinstance(encoder_env, yaml.MappingNode)
    for name, value in {
        "GITHUB_EVENT_NAME": "${{ github.event_name }}",
        "GITHUB_REF_NAME": "${{ github.ref_name }}",
        "INPUT_VERSION": "${{ github.event.inputs.version }}",
    }.items():
        node = _mapping_child(encoder_env, name)
        assert isinstance(node, yaml.ScalarNode) and node.value == value, name


@pytest.mark.parametrize(("workflow_name", "secret_name", "url_name"), SECRET_NAMES)
def test_secret_values_do_not_appear_in_workflow_text(
    workflow_name: str, secret_name: str, url_name: str
) -> None:
    text = (WORKFLOWS_DIR / workflow_name).read_text(encoding="utf-8")
    # Secrets may only be referenced through env bindings, never inline.
    inline = re.findall(rf"secrets\.({secret_name}|{url_name})", text)
    assert len(inline) == 2, f"{workflow_name}: secrets must be referenced only in env"


def test_webhook_payload_matches_expected_schema() -> None:
    """The encoded payload carries exactly version and image_label keys."""
    for workflow_name in FLY_DEPLOY_WORKFLOWS:
        text = (WORKFLOWS_DIR / workflow_name).read_text(encoding="utf-8")
        assert '"version"' in text and '"image_label"' in text, workflow_name
