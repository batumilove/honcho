import re
from collections.abc import Iterator
from pathlib import Path
from typing import cast

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


def _iter_uses(node: Node) -> Iterator[str]:
    """Yield every GitHub Actions `uses` value from a parsed YAML node tree."""
    if isinstance(node, MappingNode):
        for key, child in node.value:
            if (
                isinstance(key, ScalarNode)
                and key.value == "uses"
                and isinstance(child, ScalarNode)
            ):
                yield child.value
            yield from _iter_uses(child)
    elif isinstance(node, SequenceNode):
        for child in node.value:
            yield from _iter_uses(child)


def test_external_actions_use_approved_immutable_refs() -> None:
    observed_actions: set[str] = set()

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
            observed_actions.add(action)

    assert observed_actions == set(APPROVED_ACTION_REFS)
