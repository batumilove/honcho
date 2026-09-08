import re
import subprocess
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"
FULL_SHA = re.compile(r"[0-9a-f]{40}")

APPROVED_ACTION_REFS = {
    "actions/attest-build-provenance": "4d101475d8b20a2381f78447822ac1eab6504dd8",
    "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
    "actions/setup-python": "5fda3b95a4ea91299a34e894583c3862153e4b97",
    "astral-sh/setup-uv": "20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
    "aws-actions/aws-secretsmanager-get-secrets": (
        "2cb1a461cbd4865ac4299648312e4704c646cd53"
    ),
    "aws-actions/configure-aws-credentials": (
        "cbe3b392738ccf3f987d68400dafcf4b0624a56c"
    ),
    "docker/build-push-action": "53b7df96c91f9c12dcc8a07bcb9ccacbed38856a",
    "docker/login-action": "dbcb813823bdd20940b903addbd779551569679f",
    "docker/metadata-action": "dc802804100637a589fabce1cb79ff13a1411302",
    "docker/setup-buildx-action": "37fe631027851001ddb9b187196cc803df7f5f0e",
    "dorny/paths-filter": "ceb8a2b8f2d89434be7ff52d3de7ec3738c5cc9d",
    "oven-sh/setup-bun": "0c5077e51419868618aeaa5fe8019c62421857d6",
    "superfly/flyctl-actions/setup-flyctl": (
        "ed8efb33836e8b2096c7fd3ba1c8afe303ebbff1"
    ),
}


APPROVED_ACTION_COUNTS = {
    "actions/attest-build-provenance": 1,
    "actions/checkout": 8,
    "actions/setup-python": 2,
    "astral-sh/setup-uv": 2,
    "aws-actions/aws-secretsmanager-get-secrets": 1,
    "aws-actions/configure-aws-credentials": 1,
    "docker/build-push-action": 1,
    "docker/login-action": 1,
    "docker/metadata-action": 1,
    "docker/setup-buildx-action": 1,
    "dorny/paths-filter": 1,
    "oven-sh/setup-bun": 1,
    "superfly/flyctl-actions/setup-flyctl": 4,
}
APPROVED_LOCAL_USES = "./.github/workflows/start-fly-runner.yml"
FLY_DEPLOY_WORKFLOWS = ("fly-deploy.yml", "fly-deploy-prod.yml")


def _iter_mapping_nodes(node: Node) -> Iterator[MappingNode]:
    """Yield every mapping from a parsed YAML node tree."""
    if isinstance(node, MappingNode):
        yield node
        for key, child in node.value:
            yield from _iter_mapping_nodes(key)
            yield from _iter_mapping_nodes(child)
    elif isinstance(node, SequenceNode):
        for child in node.value:
            yield from _iter_mapping_nodes(child)


def _iter_scalar_values(node: Node) -> Iterator[str]:
    """Yield every scalar value from a parsed YAML node tree."""
    if isinstance(node, ScalarNode):
        yield node.value
    elif isinstance(node, MappingNode):
        for key, child in node.value:
            yield from _iter_scalar_values(key)
            yield from _iter_scalar_values(child)
    elif isinstance(node, SequenceNode):
        for child in node.value:
            yield from _iter_scalar_values(child)


def _mapping_child(node: MappingNode, name: str) -> Node | None:
    """Return a mapping value by scalar key name."""
    for key, child in node.value:
        if isinstance(key, ScalarNode) and key.value == name:
            return child
    return None


def _load_workflow(path: Path) -> Node:
    workflow = cast(
        Node | None,
        yaml.compose(  # pyright: ignore[reportUnknownMemberType]
            path.read_text(encoding="utf-8")
        ),
    )
    assert workflow is not None, f"empty workflow: {path}"
    return workflow


def _iter_uses(node: Node) -> Iterator[str]:
    """Yield every GitHub Actions `uses` value from a parsed YAML node tree."""
    if isinstance(node, MappingNode):
        for key, child in node.value:
            if isinstance(key, ScalarNode) and key.value == "uses":
                assert (
                    isinstance(child, ScalarNode)
                    and child.tag == "tag:yaml.org,2002:str"
                ), "uses value must be a string"
                yield child.value
            yield from _iter_uses(child)
    elif isinstance(node, SequenceNode):
        for child in node.value:
            yield from _iter_uses(child)


@pytest.mark.parametrize(
    "workflow_text",
    [
        "steps:\n  - uses:\n      nested: value\n",
        "steps:\n  - uses:\n      - action\n",
        "steps:\n  - uses:\n",
        "steps:\n  - uses: 123\n",
    ],
)
def test_non_string_uses_is_rejected(workflow_text: str) -> None:
    workflow = cast(
        Node | None,
        yaml.compose(  # pyright: ignore[reportUnknownMemberType]
            workflow_text
        ),
    )
    assert workflow is not None

    with pytest.raises(AssertionError, match="uses value must be a string"):
        list(_iter_uses(workflow))


def test_external_actions_use_approved_immutable_refs() -> None:
    observed_actions: Counter[str] = Counter()
    observed_local_uses: Counter[str] = Counter()

    for workflow_path in sorted(WORKFLOWS_DIR.glob("*.y*ml")):
        workflow = cast(
            Node | None,
            yaml.compose(  # pyright: ignore[reportUnknownMemberType]
                workflow_path.read_text(encoding="utf-8")
            ),
        )
        assert workflow is not None, f"empty workflow: {workflow_path}"
        for uses in _iter_uses(workflow):
            if uses.startswith("./"):
                observed_local_uses[uses] += 1
                continue

            action, separator, ref = uses.partition("@")
            assert separator, f"missing ref in {workflow_path}: {uses}"
            assert FULL_SHA.fullmatch(ref), (
                f"external action must use a full immutable SHA in "
                f"{workflow_path}: {uses}"
            )
            assert action in APPROVED_ACTION_REFS, (
                f"unreviewed external action in {workflow_path}: {action}"
            )
            assert ref == APPROVED_ACTION_REFS[action], (
                f"unexpected ref for {action} in {workflow_path}: {ref}"
            )
            observed_actions[action] += 1

    assert set(APPROVED_ACTION_COUNTS) == set(APPROVED_ACTION_REFS)
    assert observed_actions == Counter(APPROVED_ACTION_COUNTS)
    assert observed_local_uses == Counter({APPROVED_LOCAL_USES: 1})


def test_workflows_do_not_contain_non_breaking_spaces() -> None:
    for workflow_path in sorted(WORKFLOWS_DIR.glob("*.y*ml")):
        assert "\N{NO-BREAK SPACE}" not in workflow_path.read_text(encoding="utf-8"), (
            f"non-breaking space in {workflow_path}"
        )


def test_checkout_steps_do_not_persist_credentials() -> None:
    checkout_steps = 0

    for workflow_path in sorted(WORKFLOWS_DIR.glob("*.y*ml")):
        for mapping in _iter_mapping_nodes(_load_workflow(workflow_path)):
            uses = _mapping_child(mapping, "uses")
            if not (
                isinstance(uses, ScalarNode)
                and uses.value.startswith("actions/checkout@")
            ):
                continue

            checkout_steps += 1
            with_node = _mapping_child(mapping, "with")
            assert isinstance(with_node, MappingNode), (
                f"checkout step must define with.persist-credentials in {workflow_path}"
            )
            persist_credentials = _mapping_child(with_node, "persist-credentials")
            assert (
                isinstance(persist_credentials, ScalarNode)
                and persist_credentials.tag == "tag:yaml.org,2002:bool"
                and persist_credentials.value == "false"
            ), f"checkout credentials must not persist in {workflow_path}"

    assert checkout_steps == APPROVED_ACTION_COUNTS["actions/checkout"]


def test_action_policy_runs_for_every_workflow_change() -> None:
    workflow = _load_workflow(WORKFLOWS_DIR / "unittest.yml")
    assert isinstance(workflow, MappingNode)
    workflow_scope = ".github/workflows/**"

    triggers = _mapping_child(workflow, "on")
    assert isinstance(triggers, MappingNode)
    for trigger_name in ("push", "pull_request"):
        trigger = _mapping_child(triggers, trigger_name)
        assert isinstance(trigger, MappingNode)
        paths = _mapping_child(trigger, "paths")
        assert isinstance(paths, SequenceNode)
        assert workflow_scope in list(_iter_scalar_values(paths)), (
            f"FastAPI {trigger_name} scope must include every workflow file"
        )

    jobs = _mapping_child(workflow, "jobs")
    assert isinstance(jobs, MappingNode)
    changes = _mapping_child(jobs, "changes")
    assert isinstance(changes, MappingNode)
    steps = _mapping_child(changes, "steps")
    assert isinstance(steps, SequenceNode)
    filter_steps = [
        step
        for step in steps.value
        if isinstance(step, MappingNode)
        and isinstance((uses := _mapping_child(step, "uses")), ScalarNode)
        and uses.value.startswith("dorny/paths-filter@")
    ]
    assert len(filter_steps) == 1
    with_node = _mapping_child(filter_steps[0], "with")
    assert isinstance(with_node, MappingNode)
    filters = _mapping_child(with_node, "filters")
    assert isinstance(filters, ScalarNode)
    filter_workflow = cast(
        Node | None,
        yaml.compose(  # pyright: ignore[reportUnknownMemberType]
            filters.value
        ),
    )
    assert isinstance(filter_workflow, MappingNode)
    python_filter = _mapping_child(filter_workflow, "python")
    assert isinstance(python_filter, SequenceNode)
    assert workflow_scope in list(_iter_scalar_values(python_filter)), (
        "FastAPI Python filter must include every workflow file"
    )


@pytest.mark.parametrize(
    ("changes_result", "python_changed", "python_result", "expected_returncode"),
    [
        ("failure", "false", "skipped", 1),
        ("success", "true", "skipped", 1),
        ("success", "true", "success", 0),
        ("success", "false", "skipped", 0),
        ("success", "false", "success", 1),
        ("success", "", "skipped", 1),
    ],
)
def test_status_fails_closed(
    changes_result: str,
    python_changed: str,
    python_result: str,
    expected_returncode: int,
) -> None:
    workflow = _load_workflow(WORKFLOWS_DIR / "unittest.yml")
    assert isinstance(workflow, MappingNode)
    jobs = _mapping_child(workflow, "jobs")
    assert isinstance(jobs, MappingNode)
    test_status = _mapping_child(jobs, "test-status")
    assert isinstance(test_status, MappingNode)
    steps = _mapping_child(test_status, "steps")
    assert isinstance(steps, SequenceNode) and len(steps.value) == 1
    step = steps.value[0]
    assert isinstance(step, MappingNode)
    run = _mapping_child(step, "run")
    env = _mapping_child(step, "env")
    assert isinstance(run, ScalarNode)
    assert isinstance(env, MappingNode)

    expected_env = {
        "CHANGES_RESULT": "${{ needs.changes.result }}",
        "PYTHON_CHANGED": "${{ needs.changes.outputs.python }}",
        "PYTHON_RESULT": "${{ needs.test-python.result }}",
    }
    for name, expected_value in expected_env.items():
        value = _mapping_child(env, name)
        assert isinstance(value, ScalarNode) and value.value == expected_value
    assert "${{" not in run.value

    completed = subprocess.run(
        ["bash", "-c", run.value],
        check=False,
        capture_output=True,
        env={
            "CHANGES_RESULT": changes_result,
            "PYTHON_CHANGED": python_changed,
            "PYTHON_RESULT": python_result,
        },
        text=True,
    )
    assert completed.returncode == expected_returncode, completed.stdout


def test_fly_deploy_shell_does_not_interpolate_github_values() -> None:
    unsafe_expression = re.compile(
        r"\$\{\{[^}\n]*(?:\bversion\b|ref_name)[^}\n]*}}"
    )
    expected_env = {
        "GITHUB_EVENT_NAME": "${{ github.event_name }}",
        "GITHUB_REF_NAME": "${{ github.ref_name }}",
        "INPUT_VERSION": "${{ github.event.inputs.version }}",
    }

    for workflow_name in FLY_DEPLOY_WORKFLOWS:
        workflow_path = WORKFLOWS_DIR / workflow_name
        relevant_steps = 0

        for mapping in _iter_mapping_nodes(_load_workflow(workflow_path)):
            run = _mapping_child(mapping, "run")
            if not isinstance(run, ScalarNode):
                continue
            assert unsafe_expression.search(run.value) is None, (
                f"GitHub value interpolated directly into shell in {workflow_path}"
            )
            if "flyctl deploy" not in run.value and "fly_webhook_payload" not in run.value:
                continue

            relevant_steps += 1
            env = _mapping_child(mapping, "env")
            assert isinstance(env, MappingNode)
            for name, expected_value in expected_env.items():
                value = _mapping_child(env, name)
                assert isinstance(value, ScalarNode) and value.value == expected_value

            if "flyctl deploy" in run.value:
                assert 'IMAGE_LABEL="deployment-${INPUT_VERSION}"' in run.value
                assert 'IMAGE_LABEL="deployment-${GITHUB_REF_NAME}"' in run.value
                assert 'if [[ "$GITHUB_EVENT_NAME" == "workflow_dispatch" ]]' in run.value
            else:
                # Version branching moved into the encoder script; the shell only runs it.
                assert "fly_webhook_payload" in run.value

        assert relevant_steps == 2
