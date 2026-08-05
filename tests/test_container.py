from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_container_runs_non_root_with_health_check_and_persistent_database():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()

    assert "USER app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/_stcore/health" in dockerfile
    assert "CHATGPTFOLIO_DB_PATH=/app/state/market_data.duckdb" in dockerfile


def test_compose_exposes_dashboard_and_mounts_state_volume():
    compose = (PROJECT_ROOT / "compose.yaml").read_text()

    assert '"8501:8501"' in compose
    assert "chatgptfolio-state:/app/state" in compose


def test_github_workflow_only_publishes_on_manual_dispatch():
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "container.yml").read_text()

    assert "workflow_dispatch:" in workflow
    assert "push: ${{ github.event_name == 'workflow_dispatch' }}" in workflow
    assert "ghcr.io/${{ github.repository }}" in workflow
