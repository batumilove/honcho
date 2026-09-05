from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github/workflows/unified-tests.yml"


def test_fly_unified_tests_requires_explicit_repository_opt_in() -> None:
    workflow_lines = WORKFLOW.read_text().splitlines()
    job_start = workflow_lines.index("  start-runner:")
    job_end = next(
        (
            index
            for index, line in enumerate(workflow_lines[job_start + 1 :], job_start + 1)
            if line.startswith("  ") and not line.startswith("    ")
        ),
        len(workflow_lines),
    )
    start_runner_job = workflow_lines[job_start:job_end]

    assert "    if: vars.FLY_UNIFIED_TESTS_ENABLED == 'true'" in start_runner_job
