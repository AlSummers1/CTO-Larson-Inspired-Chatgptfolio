from datetime import date

import pandas as pd

from stockpicker.data import load_lab_runs, save_lab_research_details, save_lab_research_snapshot, save_lab_run
from stockpicker.lab import LabConfig, optimize_decision, paper_orders, research_packet, validate_decision


def _candidates():
    return pd.DataFrame([{"symbol": "NVDA", "name": "NVIDIA", "theme": "AI", "latest_close": 100.0, "score": 90.0}])


def test_packet_is_point_in_time_and_bounded():
    packet = research_packet(_candidates(), date(2026, 7, 1), "SPY")
    assert packet["as_of_date"] == "2026-07-01"
    assert packet["candidates"][0]["symbol"] == "NVDA"


def test_validator_rejects_symbols_outside_packet_and_bad_weights():
    result = validate_decision({"holdings": [{"symbol": "BAD", "weight": 0.8}], "cash_weight": 0.2}, _candidates(), LabConfig())
    assert not result.valid
    assert "outside" in " ".join(result.errors)


def test_paper_orders_include_estimated_cost():
    decision = {"holdings": [{"symbol": "NVDA", "weight": 0.1}], "cash_weight": 0.9}
    orders = paper_orders(pd.DataFrame(), decision, LabConfig(paper_capital=10_000, slippage_bps=10))
    assert orders.loc[orders.symbol == "NVDA", "notional"].iloc[0] == 1000
    assert orders.loc[orders.symbol == "NVDA", "estimated_cost"].iloc[0] == 1


def test_lab_records_do_not_require_or_store_credentials(tmp_path):
    db_path = tmp_path / "lab.duckdb"
    packet_hash = save_lab_research_snapshot({"as_of_date": "2026-07-01", "candidates": []}, db_path)
    run_id = save_lab_run({"as_of_date": date(2026, 7, 1), "benchmark": "SPY", "model": "gpt-5.5", "status": "model_failed", "packet_hash": packet_hash, "error_message": "missing key"}, db_path)
    runs = load_lab_runs(db_path)
    assert run_id == 1
    assert runs["model"].tolist() == ["gpt-5.5"]


def test_optimizer_reconciles_weights_and_honors_position_and_sector_caps():
    candidates = pd.DataFrame([
        {"symbol": "AAA", "score": 90, "sector": "Tech"},
        {"symbol": "BBB", "score": 80, "sector": "Tech"},
        {"symbol": "CCC", "score": 70, "sector": "Health"},
    ])
    model = {
        "abstain_reason": None, "portfolio_thesis": "test", "cash_target": 0,
        "holdings": [
            {"symbol": "AAA", "conviction": 100}, {"symbol": "BBB", "conviction": 90}, {"symbol": "CCC", "conviction": 80},
        ],
    }
    config = LabConfig(max_position=0.2, max_sector_exposure=0.3)
    result = optimize_decision(model, candidates, config)
    assert abs(sum(row["weight"] for row in result["holdings"]) + result["cash_weight"] - 1) < 0.000001
    assert all(row["weight"] <= 0.2 for row in result["holdings"])
    assert result["cash_weight"] >= 0.4
    assert validate_decision(result, candidates, config).valid


def test_packet_records_explicit_missing_research_without_fetching():
    packet = research_packet(_candidates(), date(2026, 7, 1), "SPY", LabConfig(), refresh_research=False)
    assert packet["coverage"]["fundamentals"] == "not_requested"
    assert packet["missing_evidence_policy"].startswith("No evidence")
    assert packet["candidates"][0]["combined_score"] == 90.0


def test_research_details_persist_source_level_evidence(tmp_path):
    db_path = tmp_path / "evidence.duckdb"
    packet = {
        "as_of_date": "2026-07-01", "candidates": [], "security_master": [],
        "candidate_scores": [{"symbol": "NVDA", "technical_score": 90, "fundamental_score": 80, "combined_score": 86, "liquidity_score": 90, "sector": "Tech", "score_json": "{}"}],
        "evidence": [{"evidence_id": "e1", "symbol": "NVDA", "category": "news", "title": "Test", "summary": "Test", "source_url": "https://example.com", "publisher": "Test", "published_at": "2026-07-01", "fetched_at": "2026-07-01", "content_hash": "x", "status": "available"}],
    }
    packet_hash = save_lab_research_snapshot(packet, db_path)
    save_lab_research_details(packet_hash, packet, db_path)
    import duckdb
    with duckdb.connect(str(db_path), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM lab_evidence").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM lab_factor_scores").fetchone()[0] == 1
