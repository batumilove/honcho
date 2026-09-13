"""Fail-closed policy tests: Fly runner token handling.

Guards the ephemeral Fly test-runner workflows so GitHub Actions tokens
(`GH_TOKEN_ACTIONS`, `GITHUB_TOKEN`) never appear in ``run:`` shell
expressions, process argv, or job-wide secret inheritance, and cleanup
reports failures accurately instead of masking them as success.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
START_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "start-fly-runner.yml"
UNIFIED_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "unified-tests.yml"

TOKEN_SECRET_KEYS = ("GH_TOKEN_ACTIONS", "GITHUB_TOKEN", "FLY_API_TOKEN_TESTING")
# ${{ secrets.NAME }} or ${{ env.NAME }} style interpolations inside run bodies.
INTERP_RE = re.compile(r"\$\{\{\s*(secrets|env|needs)[^}]*\}\}")


def _load(path: Path) -> str:
    assert path.is_file(), f"missing workflow: {path}"
    return path.read_text(encoding="utf-8")


def _run_blocks(text: str) -> list[str]:
    """Extract every `run:` script body from workflow YAML text."""
    import yaml

    doc = yaml.safe_load(text)
    blocks: list[str] = []
    for job in doc.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "run" in step:
                blocks.append(step["run"])
    return blocks


class TestNoTokenInterpolationInRunBlocks:
    def test_start_runner_never_interpolates_secrets_into_run(self) -> None:
        text = _load(START_WORKFLOW)
        for i, block in enumerate(_run_blocks(text)):
            hits = INTERP_RE.findall(block)
            assert not hits, (
                f"start-fly-runner run block #{i} interpolates workflow "
                f"expressions ({hits}); pass values via step env instead"
            )

    def test_unified_cleanup_never_interpolates_secrets_into_run(self) -> None:
        text = _load(UNIFIED_WORKFLOW)
        for i, block in enumerate(_run_blocks(text)):
            hits = INTERP_RE.findall(block)
            assert not hits, (
                f"unified-tests run block #{i} interpolates workflow "
                f"expressions ({hits}); pass values via step env instead"
            )


class TestNoJobWideSecretEnv:
    def test_unified_cleanup_job_declares_no_token_env(self) -> None:
        import yaml

        doc = yaml.safe_load(_load(UNIFIED_WORKFLOW))
        env = doc["jobs"]["cleanup-machine"].get("env", {})
        for key in TOKEN_SECRET_KEYS:
            assert key not in env, (
                f"cleanup-machine job env binds {key} for every step; "
                "scope it to the exact consuming steps"
            )

    def test_start_runner_job_declares_no_token_env(self) -> None:
        import yaml

        doc = yaml.safe_load(_load(START_WORKFLOW))
        for name, job in doc.get("jobs", {}).items():
            env = job.get("env", {})
            for key in TOKEN_SECRET_KEYS:
                assert key not in env, (
                    f"{name} job env binds {key} for every step; scope it "
                    "to the exact consuming steps"
                )


class TestTokensNeverInArgv:
    def test_flyctl_secret_set_reads_stdin_not_argv(self) -> None:
        text = _load(START_WORKFLOW)
        for block in _run_blocks(text):
            for line in block.splitlines():
                if "flyctl secrets" in line:
                    # Inspect only the flyctl side of any pipeline: a token
                    # piped via stdin is fine, a token argument is not.
                    flyctl_side = line.split("|", 1)[-1]
                    assert (
                        "$GH_TOKEN" not in flyctl_side
                        and "$GITHUB_TOKEN" not in flyctl_side
                    ), f"flyctl secrets invocation may expose token in argv: {line}"

    def test_curl_never_receives_token_on_command_line(self) -> None:
        for path in (START_WORKFLOW, UNIFIED_WORKFLOW):
            for block in _run_blocks(_load(path)):
                for line in block.splitlines():
                    if line.strip().startswith("curl") or " curl " in line:
                        assert (
                            "$GITHUB_TOKEN" not in line
                            and "${GITHUB_TOKEN}" not in line
                        ), f"{path.name}: curl line may receive token via argv: {line}"


class TestCleanupFailClosed:
    def test_cleanup_does_not_mask_failures_as_success(self) -> None:
        text = _load(UNIFIED_WORKFLOW)
        blocks = _run_blocks(text)
        cleanup = "\n".join(blocks)
        # A destroyed-or-missing machine is fine; a *failed destroy* must not
        # be swallowed into exit 0 without at least surfacing an error marker.
        assert (
            "|| echo" not in cleanup or "::error::" in cleanup or "exit 1" in cleanup
        ), (
            "cleanup steps convert failures into successful exits without "
            "failing the job or emitting an error annotation"
        )

    def test_cleanup_verifies_runner_identity_before_delete(self) -> None:
        text = _load(UNIFIED_WORKFLOW)
        cleanup = "\n".join(_run_blocks(text))
        assert "runner" in cleanup.lower()
        assert (
            re.search(r"select\(\.name\s*==", cleanup) or "exact" in cleanup.lower()
        ), "runner deletion must match the exact runner name, not a bare label fallback"
