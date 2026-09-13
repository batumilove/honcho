"""Fail-closed policy tests: Fly runner token handling.

Structurally binds the ephemeral Fly test-runner workflows so GitHub Actions
and Fly tokens never appear in ``run:`` shell expressions, process argv, or
job-wide secret inheritance, and cleanup fails closed when a resource still
exists. Assertions operate on parsed YAML and anchored regexes so a single
safe occurrence cannot mask an unsafe one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
START_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "start-fly-runner.yml"
UNIFIED_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "unified-tests.yml"
WORKFLOWS = (START_WORKFLOW, UNIFIED_WORKFLOW)

# Secret names whose values are tokens.
TOKEN_SECRET_NAMES = frozenset(
    {"GH_TOKEN_ACTIONS", "GITHUB_TOKEN", "FLY_API_TOKEN_TESTING"}
)
# Env keys whose values would be tokens at runtime.
TOKEN_ENV_KEYS = frozenset(
    {"GITHUB_TOKEN", "GH_TOKEN", "GH_TOKEN_VALUE", "FLY_API_TOKEN"}
)

INTERP_RE = re.compile(r"\$\{\{.*?\}\}", re.DOTALL)


def _load(path: Path) -> str:
    assert path.is_file(), f"missing workflow: {path}"
    return path.read_text(encoding="utf-8")


def _doc(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load(_load(path))
    return doc


def _jobs(path: Path) -> dict[str, Any]:
    jobs: dict[str, Any] = _doc(path).get("jobs", {})
    return jobs


def _run_steps(path: Path) -> list[tuple[str, str, str]]:
    """Yield (job_name, step_name, run_body) for every runnable step."""
    out: list[tuple[str, str, str]] = []
    for job_name, job in _jobs(path).items():
        for step in job.get("steps", []) or []:
            if "run" in step:
                out.append((job_name, step.get("name", "<unnamed>"), step["run"]))
    return out


def _secret_refs(node: Any) -> set[str]:
    """Recursively collect secret names referenced via ${{ secrets.X }}."""
    refs: set[str] = set()
    if isinstance(node, dict):
        for value in node.values():
            refs |= _secret_refs(value)
    elif isinstance(node, list):
        for value in node:
            refs |= _secret_refs(value)
    elif isinstance(node, str):
        refs |= set(re.findall(r"\$\{\{\s*secrets\.([A-Za-z0-9_]+)\s*\}\}", node))
    return refs


class TestNoInterpolationInRunBlocks:
    def test_no_workflow_expressions_in_any_run_block(self) -> None:
        violations: list[str] = []
        for path in WORKFLOWS:
            for job_name, step_name, body in _run_steps(path):
                hit = INTERP_RE.search(body)
                if hit:
                    violations.append(
                        f"{path.name}:{job_name}/{step_name}: {hit.group(0)!r}"
                    )
        assert not violations, (
            "run blocks must not interpolate workflow expressions "
            f"(pass values via step env): {violations}"
        )


class TestNoJobWideTokenEnv:
    def test_no_job_env_binds_token_secrets_or_token_keys(self) -> None:
        violations: list[str] = []
        for path in WORKFLOWS:
            for job_name, job in _jobs(path).items():
                for key, value in (job.get("env") or {}).items():
                    token_by_key = key in TOKEN_ENV_KEYS
                    token_by_value = bool(_secret_refs([value]) & TOKEN_SECRET_NAMES)
                    if token_by_key or token_by_value:
                        violations.append(f"{path.name}:{job_name}.env.{key}")
        assert not violations, (
            f"job-wide env binds tokens for every step; scope to consuming "
            f"steps: {violations}"
        )

    def test_step_env_token_bindings_exist_only_in_consuming_steps(self) -> None:
        # Sanity: tokens do still reach the steps that need them (the change
        # must not have simply deleted the plumbing).
        start = _jobs(START_WORKFLOW)["start-runner"]["steps"]
        wait = next(s for s in start if s.get("id") == "wait-for-runner")
        assert _secret_refs(wait.get("env", {})) & {"GH_TOKEN_ACTIONS"}, (
            "wait-for-runner step must still receive GH_TOKEN_ACTIONS via env"
        )
        unified = _jobs(UNIFIED_WORKFLOW)["cleanup-machine"]["steps"]
        cleanup = next(s for s in unified if s.get("name") == "Cleanup GitHub runner")
        assert _secret_refs(cleanup.get("env", {})) & {"GH_TOKEN_ACTIONS"}, (
            "Cleanup GitHub runner step must still receive GH_TOKEN_ACTIONS via env"
        )


class TestTokensNeverInArgv:
    def test_flyctl_secret_upload_is_exactly_one_stdin_pipe(self) -> None:
        secret_lines = [
            (path.name, job, step, line)
            for path in WORKFLOWS
            for job, step, body in _run_steps(path)
            for line in body.splitlines()
            if "flyctl secrets" in line
        ]
        assert len(secret_lines) == 1, (
            f"expected exactly one flyctl secrets invocation, found {secret_lines}"
        )
        fname, job, step, line = secret_lines[0]
        pipe_re = re.compile(
            r"^\s*printf\s+'GH_TOKEN=%s'\s+\"\$GH_TOKEN_VALUE\"\s*\|\s*"
            r"flyctl\s+secrets\s+upload\s+-a\s+\"\$\{?FLY_RUNNER_APP\}?\"\s+-$"
        )
        assert pipe_re.match(line), (
            f"{fname}:{job}/{step}: flyctl secrets must be exactly "
            f"printf 'GH_TOKEN=%s' \"$GH_TOKEN_VALUE\" | flyctl secrets upload "
            f'-a "$FLY_RUNNER_APP" - ; got: {line!r}'
        )

    def test_every_api_github_curl_reads_headers_from_file(self) -> None:
        for path in WORKFLOWS:
            for job, step, body in _run_steps(path):
                # Join continuation lines so multiline curl invocations are seen.
                logical = re.sub(r"\\\n\s*", " ", body)
                for line in logical.splitlines():
                    if "curl" in line and "api.github.com" in line:
                        assert '--header @"$HEADER_FILE"' in line, (
                            f"{path.name}:{job}/{step}: api.github.com curl must "
                            f'read auth via --header @"$HEADER_FILE": {line.strip()!r}'
                        )
                        for key in sorted(TOKEN_ENV_KEYS):
                            assert f"${key}" not in line, (
                                f"{path.name}:{job}/{step}: curl argv may "
                                f"contain {key}: {line.strip()!r}"
                            )

    def test_header_file_lifecycle_in_token_using_steps(self) -> None:
        header_re = re.compile(
            r"HEADER_FILE=\$\(mktemp\)\n"
            r"\s*chmod 0600 \"\$HEADER_FILE\"\n"
            r"\s*printf 'Authorization: Bearer %s\\nAccept: application/vnd\.github(\.v3)?\+json\\n' "
            r"\"\$GITHUB_TOKEN\" >\"\$HEADER_FILE\"\n"
            r"\s*trap 'rm -f \"\$HEADER_FILE\"' EXIT"
        )
        for path in WORKFLOWS:
            for job, step, body in _run_steps(path):
                if "$HEADER_FILE" not in body:
                    continue
                assert header_re.search(body), (
                    f"{path.name}:{job}/{step}: HEADER_FILE must be created by "
                    "mktemp, chmod 0600, written by printf, and trap-removed on "
                    "EXIT in that order"
                )


class TestCleanupFailClosed:
    def _cleanup_steps(self) -> dict[str, str]:
        return {
            s.get("name", "<unnamed>"): s["run"]
            for s in _jobs(UNIFIED_WORKFLOW)["cleanup-machine"]["steps"]
            if "run" in s
        }

    def test_machine_cleanup_fail_closed_shape(self) -> None:
        body = self._cleanup_steps()["Cleanup fly machine"]
        # The existence check must gate the error+exit branch directly.
        gate_re = re.compile(
            r"if flyctl machines status \"\$MACHINE_ID\" -a \"\$FLY_RUNNER_APP\" "
            r">/dev/null 2>&1; then\n"
            r"\s*echo \"::error::Machine \${MACHINE_ID} still exists after cleanup\"\n"
            r"\s*exit 1\n"
            r"\s*fi"
        )
        assert gate_re.search(body), (
            "machine cleanup must run `flyctl machines status` and, when it "
            "succeeds (machine exists), emit ::error:: and exit 1 in the same "
            "branch"
        )
        # No failure-masking construct may appear after the status gate.
        gate = gate_re.search(body)
        assert gate, "unreachable"  # pragma: no cover
        tail = body[gate.start() :]
        for mask in ("|| true", "|| echo", "|| :", "|| exit 0"):
            assert mask not in tail, (
                f"post-verification failures must not be masked (found {mask!r})"
            )

    def test_runner_delete_success_is_exactly_204_or_404(self) -> None:
        body = self._cleanup_steps()["Cleanup GitHub runner"]
        cond_re = re.compile(
            r'^\s*if \[ "\$HTTP_CODE" = "204" \] \|\| \[ "\$HTTP_CODE" = "404" \]; then\n'
            r"\s*echo \"Successfully deleted runner \(HTTP \${HTTP_CODE}; 404 = already gone\)\.\"\n"
            r"\s*else\n"
            r"\s*echo \"::error::Failed to delete runner \${RUNNER_NAME} \(id \${RUNNER_ID}\)\. "
            r"HTTP code: \$HTTP_CODE\"\n"
            r"\s*exit 1\n"
            r"\s*fi$",
            re.MULTILINE,
        )
        assert cond_re.search(body), (
            "runner delete must accept exactly HTTP 204 or 404 as success and "
            "fail with ::error:: + exit 1 otherwise, in one tied branch"
        )

    def test_runner_lookup_is_exactly_one_name_select_no_label_fallback(self) -> None:
        body = self._cleanup_steps()["Cleanup GitHub runner"]
        # Partition the script at every `jq` token start; each chunk (up to
        # the next jq invocation) is one invocation regardless of newlines or
        # same-line multiples. Count chunks that reference the runners array.
        starts = [m.start() for m in re.finditer(r"\bjq\b", body)]
        assert starts, "no jq runner lookup found at all"
        chunks = []
        for i, s in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(body)
            chunks.append(body[s:end])
        lookups = [c for c in chunks if "runners" in c]
        assert len(lookups) == 1, (
            f"exactly one runner lookup expected, found {len(lookups)}: {lookups}"
        )
        assert re.search(r"jq\s+-r\s+--arg\s+name\s", lookups[0]), (
            f"runner lookup must key on --arg name, got: {lookups[0]!r}"
        )
        assert "select(.name == $name)" in body, (
            "runner lookup must select on the exact runner name"
        )
        assert "FALLBACK_LABEL" not in body and "--arg label" not in body, (
            "runner deletion must not fall back to a bare run-id label match"
        )
