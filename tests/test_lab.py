from datetime import date

import pandas as pd

from stockpicker.data import load_lab_runs, save_lab_research_snapshot, save_lab_run
from stockpicker.lab import LabConfig, paper_orders, research_packet, validate_decision


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
