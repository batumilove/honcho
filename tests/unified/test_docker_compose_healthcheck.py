from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_api_healthcheck_allows_database_migrations_to_finish() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.yml.example").read_text(encoding="utf-8")
    )
    healthcheck = compose["services"]["api"]["healthcheck"]

    assert healthcheck["start_period"] == "60s"
    assert healthcheck["retries"] >= 10
