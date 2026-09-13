"""Fail-closed policy tests: Fly runner token handling.

Guards the ephemeral Fly test-runner workflows so GitHub Actions tokens
(`GH_TOKEN_ACTIONS`, `GITHUB_TOKEN`, `FLY_API_TOKEN_TESTING`) never appear
in ``run:`` shell expressions, process argv, or job-wide secret
inheritance, and cleanup fails closed when a resource still exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
START_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "start-fly-runner.yml"
UNIFIED_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "unified-tests.yml"

TOKEN_ENV_KEYS = ("GH_TOKEN_VALUE", "GITHUB_TOKEN", "GH_TOKEN")
# Any ${{ ... }} interpolation inside a run body.
INTERP_RE = re.compile(r"\$\{\{.*?\}\}", re.DOTALL)


def _load(path: Path) -> str:
    assert path.is_file(), f"missing workflow: {path}"
    return path.read_text(encoding="utf-8")


def _run_blocks(path: Path) -> list[str]:
    doc = yaml.safe_load(_load(path))
    blocks: list[str] = []
    for job in doc.get("jobs", {}).values():
        if "steps" not in job:
            continue
        for step in job["steps"]:
            if "run" in step:
                blocks.append(step["run"])
    return blocks


def _jobs(path: Path) -> dict:
    return yaml.safe_load(_load(path)).get("jobs", {})


def _secret_refs(node) -> list[str]:
    """Recursively collect secret names referenced via ${{ secrets.X }}."""
    refs: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            refs.extend(_secret_refs(value))
    elif isinstance(node, list):
        for value in node:
            refs.extend(_secret_refs(value))
    elif isinstance(node, str):
        refs.extend(re.findall(r"\$\{\{\s*secrets\.([A-Za-z0-9_]+)\s*\}\}", node))
    return refs


class TestNoInterpolationInRunBlocks:
    def test_start_runner_run_blocks_have_no_workflow_expressions(self) -> None:
        for i, block in enumerate(_run_blocks(START_WORKFLOW)):
            hit = INTERP_RE.search(block)
            assert hit is None, (
                f"start-fly-runner run block #{i} interpolates {hit.group(0)!r}; "
                "pass values via step env instead"
            )

    def test_unified_tests_run_blocks_have_no_workflow_expressions(self) -> None:
        for i, block in enumerate(_run_blocks(UNIFIED_WORKFLOW)):
            hit = INTERP_RE.search(block)
            assert hit is None, (
                f"unified-tests run block #{i} interpolates {hit.group(0)!r}; "
                "pass values via step env instead"
            )


class TestNoJobWideTokenEnv:
    def test_unified_cleanup_job_declares_no_token_env(self) -> None:
        job = _jobs(UNIFIED_WORKFLOW)["cleanup-machine"]
        env = job.get("env", {})
        leaked = [
            key
            for key, value in env.items()
            if key in TOKEN_ENV_KEYS
            or any(
                name in ("GH_TOKEN_ACTIONS", "GITHUB_TOKEN")
                for name in _secret_refs([value])
            )
        ]
        assert not leaked, (
            f"cleanup-machine job env binds tokens {leaked} for every step; "
            "scope them to the exact consuming steps"
        )

    def test_start_runner_jobs_declare_no_token_env(self) -> None:
        for name, job in _jobs(START_WORKFLOW).items():
            env = job.get("env", {})
            leaked = [
                key
                for key, value in env.items()
                if key in TOKEN_ENV_KEYS
                or any(
                    name_ in ("GH_TOKEN_ACTIONS", "GITHUB_TOKEN")
                    for name_ in _secret_refs([value])
                )
            ]
            assert not leaked, (
                f"{name} job env binds tokens {leaked} for every step; scope "
                "them to the exact consuming steps"
            )


class TestTokensNeverInArgv:
    def test_flyctl_secret_upload_reads_stdin_only(self) -> None:
        blocks = _run_blocks(START_WORKFLOW)
        # Exactly one flyctl secrets invocation must exist and be a stdin pipe.
        secret_lines = [
            line
            for block in blocks
            for line in block.splitlines()
            if "flyctl secrets" in line
        ]
        assert len(secret_lines) == 1, (
            f"expected exactly one flyctl secrets invocation, found {secret_lines}"
        )
        line = secret_lines[0]
        assert "|" in line, f"flyctl secrets must read the token from stdin: {line}"
        flyctl_side = line.split("|", 1)[1]
        for key in TOKEN_ENV_KEYS:
            assert f"${key}" not in flyctl_side, (
                f"flyctl secrets may receive {key} in argv: {line}"
            )
        assert "upload" in flyctl_side and re.search(r"-\s*$", flyctl_side), (
            f"flyctl secrets must be an upload reading stdin '-': {line}"
        )
        # The producing (left) side of the pipe must carry the token value.
        left = line.split("|", 1)[0]
        assert "GH_TOKEN_VALUE" in left, (
            f"stdin pipe must write the secret value: {line}"
        )

    def test_curl_reads_auth_from_private_header_file(self) -> None:
        for path in (START_WORKFLOW, UNIFIED_WORKFLOW):
            for block in _run_blocks(path):
                for line in block.splitlines():
                    if "curl" not in line:
                        continue
                    for key in TOKEN_ENV_KEYS:
                        assert f"${key}" not in line, (
                            f"{path.name}: curl line may receive token in argv: {line}"
                        )
                # Every curl to api.github.com must use --header @
                for line in block.splitlines():
                    if "api.github.com" in line and "curl" not in line.split("\\")[0]:
                        continue
                    if "curl" in line and "api.github.com" in line:
                        assert (
                            '--header @"$HEADER_FILE"' in line or "--header @" in line
                        ), (
                            f"{path.name}: api.github.com curl must read auth from "
                            f"a header file: {line}"
                        )
            # Token-producing header files must be 0600 mktemp + trap-removed
            text = _load(path)
            if "HEADER_FILE=" in text:
                assert "mktemp" in text, (
                    f"{path.name}: HEADER_FILE must come from mktemp"
                )
                assert "chmod 0600" in text, (
                    f"{path.name}: HEADER_FILE must be chmod 0600"
                )
                assert 'rm -f "$HEADER_FILE"' in text, (
                    f"{path.name}: HEADER_FILE must be removed on exit"
                )


class TestCleanupFailClosed:
    def test_cleanup_requires_machine_gone_via_status_check(self) -> None:
        blocks = _run_blocks(UNIFIED_WORKFLOW)
        cleanup = "\n".join(blocks)
        # Success must be defined by a post-destroy existence check.
        assert re.search(
            r"flyctl machines status.*&&|if flyctl machines status", cleanup
        ), "cleanup must verify the machine is gone via flyctl machines status"
        assert "::error::" in cleanup and "exit 1" in cleanup, (
            "a still-existing machine after cleanup must fail the job with ::error::"
        )

    def test_runner_delete_accepts_204_or_404_only(self) -> None:
        blocks = _run_blocks(UNIFIED_WORKFLOW)
        cleanup = "\n".join(blocks)
        assert '"204"' in cleanup and '"404"' in cleanup, (
            "runner delete must accept 204 (deleted) and 404 (already gone) as success"
        )
        # Any other code path must exit 1.
        assert re.search(r"else\n.*::error::.*\n.*exit 1", cleanup), (
            "non-204/404 runner delete must fail the job"
        )

    def test_runner_delete_matches_exact_name_with_no_label_fallback(self) -> None:
        blocks = _run_blocks(UNIFIED_WORKFLOW)
        cleanup = "\n".join(blocks)
        assert "select(.name == $name)" in cleanup, (
            "runner lookup must select on the exact runner name"
        )
        assert "FALLBACK_LABEL" not in cleanup and "--arg label" not in cleanup, (
            "runner deletion must not fall back to a bare run-id label match"
        )
