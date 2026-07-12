from pathlib import Path

import pandas as pd
import pytest

from stockpicker.data import (
    approve_paper_rebalance,
    default_paper_portfolio,
    execute_pending_paper_rebalances,
    init_database,
    load_paper_nav,
    load_paper_benchmarks,
    load_paper_positions,
    rebuild_paper_nav,
    load_latest_portfolio_snapshot,
    load_matching_portfolio_snapshot,
    load_portfolio_rebalance_log,
    read_cached_prices,
    save_portfolio_snapshot,
    save_lab_run,
    save_lab_research_snapshot,
    write_prices,
)


pytest.importorskip("duckdb")


def test_duckdb_cache_round_trip(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    prices = pd.DataFrame(
        {
            "symbol": ["NVDA", "NVDA"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.0, 11.0],
            "volume": [1000, 1200],
        }
    )
    write_prices(prices, db_path)
    cached = read_cached_prices(["NVDA"], db_path)
    assert len(cached) == 2
    assert cached["close"].tolist() == [10.0, 11.0]


def test_approved_paper_rebalance_fills_next_open_and_builds_nav(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    packet_hash = save_lab_research_snapshot(
        {"as_of_date": "2026-01-02", "candidates": [{"symbol": "NVDA", "score": 99.0}]}, db_path
    )
    run_id = save_lab_run(
        {"as_of_date": "2026-01-02", "benchmark": "SPY", "model": "gpt-5.5", "status": "proposal_ready", "packet_hash": packet_hash}, db_path
    )
    portfolio_id = default_paper_portfolio(1_000.0, db_path)
    rebalance_id = approve_paper_rebalance(
        run_id,
        {"holdings": [{"symbol": "NVDA", "weight": 0.8}], "cash_weight": 0.2},
        {"paper_capital": 1_000.0, "slippage_bps": 0.0, "commission_bps": 0.0},
        pd.Timestamp("2026-01-02").date(),
        portfolio_id,
        db_path,
    )
    prices = pd.DataFrame(
        {
            "symbol": ["NVDA", "NVDA", "SPY", "SPY"],
            "date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-02", "2026-01-03"]),
            "open": [10.0, 10.0, 100.0, 100.0],
            "high": [10.0, 11.0, 100.0, 101.0],
            "low": [10.0, 10.0, 100.0, 100.0],
            "close": [10.0, 11.0, 100.0, 101.0],
            "volume": [1_000_000] * 4,
        }
    )
    assert execute_pending_paper_rebalances(prices, db_path) == [rebalance_id]
    positions = load_paper_positions(portfolio_id, db_path)
    assert positions["symbol"].tolist() == ["NVDA"]
    assert positions["shares"].iloc[0] == pytest.approx(80.0)
    rebuild_paper_nav(prices, portfolio_id, db_path)
    nav = load_paper_nav(portfolio_id, db_path)
    assert nav["nav"].iloc[-1] == pytest.approx(1_080.0)
    assert "TECHNICAL" in load_paper_benchmarks(portfolio_id, db_path)["symbol"].tolist()


def test_unapproved_run_cannot_change_paper_positions(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    default_paper_portfolio(1_000.0, db_path)
    prices = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    assert execute_pending_paper_rebalances(prices, db_path) == []
    assert load_paper_positions(db_path=db_path).empty


def test_portfolio_snapshot_round_trip(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    holdings = pd.DataFrame(
        [
            {
                "symbol": "NVDA",
                "name": "NVIDIA",
                "seed_universe": "SMH Universe",
                "weight": 0.12,
                "allocator_score": 95.0,
                "score": 94.0,
                "benchmark_adjusted_score": 95.0,
                "relative_strength_score": 100.0,
                "larsson_regime": "Bull",
                "reason_codes": "score_improved",
                "data_warnings": "",
            }
        ]
    )
    changes = pd.DataFrame(
        [
            {
                "symbol": "NVDA",
                "previous_weight": 0.0,
                "target_weight": 0.12,
                "delta": 0.12,
                "action": "added",
                "reason_codes": "accepted_change",
            }
        ]
    )

    snapshot_id = save_portfolio_snapshot(
        holdings,
        {
            "benchmark": "QQQ",
            "benchmark_regime": "Bull",
            "selected_seeds": "SMH Universe",
            "sensitivity": "Balanced",
            "config_json": "{}",
        },
        changes,
        db_path,
    )
    loaded_holdings, metadata = load_latest_portfolio_snapshot(db_path)
    log = load_portfolio_rebalance_log(db_path)

    assert snapshot_id == 1
    assert metadata["benchmark"] == "QQQ"
    assert loaded_holdings["symbol"].tolist() == ["NVDA"]
    assert loaded_holdings["weight"].iloc[0] == pytest.approx(0.12)
    assert {"daily_trend", "weekly_trend", "monthly_trend", "drawdown_52w", "extension_sma50d", "rrg_quadrant"}.issubset(
        loaded_holdings.columns
    )
    assert log["action"].tolist() == ["added"]


def test_portfolio_snapshot_scoped_lookup(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    holdings = pd.DataFrame(
        [
            {
                "symbol": "NVDA",
                "name": "NVIDIA",
                "seed_universe": "SMH Universe",
                "weight": 0.12,
                "allocator_score": 95.0,
                "score": 94.0,
                "benchmark_adjusted_score": 95.0,
                "relative_strength_score": 100.0,
                "larsson_regime": "Bull",
                "daily_trend": "Up",
                "weekly_trend": "Up",
                "monthly_trend": "Up",
                "drawdown_52w": -0.1,
                "extension_sma50d": 0.05,
                "rrg_quadrant": "Leading",
                "reason_codes": "score_improved",
                "data_warnings": "",
            }
        ]
    )
    save_portfolio_snapshot(
        holdings,
        {
            "benchmark": "QQQ",
            "benchmark_regime": "Bull",
            "selected_seeds": "Healthcare Universe,SMH Universe",
            "sensitivity": "Balanced",
            "config_hash": "abc123",
            "config_json": "{}",
        },
        db_path=db_path,
    )

    matched, metadata = load_matching_portfolio_snapshot("QQQ", "Healthcare Universe,SMH Universe", "abc123", db_path)
    wrong_benchmark, _ = load_matching_portfolio_snapshot("SMH", "Healthcare Universe,SMH Universe", "abc123", db_path)
    wrong_seeds, _ = load_matching_portfolio_snapshot("QQQ", "SMH Universe", "abc123", db_path)

    assert metadata["config_hash"] == "abc123"
    assert matched["rrg_quadrant"].tolist() == ["Leading"]
    assert wrong_benchmark.empty
    assert wrong_seeds.empty


def test_portfolio_snapshot_insert_uses_column_names_for_existing_schema(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    init_database(db_path)
    import duckdb

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE OR REPLACE TABLE portfolio_holdings (
                snapshot_id INTEGER,
                symbol VARCHAR,
                name VARCHAR,
                seed_universe VARCHAR,
                weight DOUBLE,
                allocator_score DOUBLE,
                score DOUBLE,
                benchmark_adjusted_score DOUBLE,
                relative_strength_score DOUBLE,
                larsson_regime VARCHAR,
                daily_trend VARCHAR,
                weekly_trend VARCHAR,
                monthly_trend VARCHAR,
                drawdown_52w DOUBLE,
                extension_sma50d DOUBLE,
                rrg_quadrant VARCHAR,
                data_warnings VARCHAR,
                reason_codes VARCHAR
            )
            """
        )
    holdings = pd.DataFrame(
        [
            {
                "symbol": "NVDA",
                "name": "NVIDIA",
                "seed_universe": "SMH Universe",
                "weight": 0.12,
                "allocator_score": 95.0,
                "score": 94.0,
                "benchmark_adjusted_score": 95.0,
                "relative_strength_score": 100.0,
                "larsson_regime": "Bull",
                "daily_trend": "Up",
                "weekly_trend": "Up",
                "monthly_trend": "Up",
                "drawdown_52w": -0.1,
                "extension_sma50d": 0.05,
                "rrg_quadrant": "Leading",
                "reason_codes": "score_improved, rrg_seed_improving",
                "data_warnings": "",
            }
        ]
    )

    save_portfolio_snapshot(
        holdings,
        {
            "benchmark": "QQQ",
            "benchmark_regime": "Bull",
            "selected_seeds": "SMH Universe",
            "sensitivity": "Balanced",
            "config_hash": "abc123",
            "config_json": "{}",
        },
        db_path=db_path,
    )

    loaded, _ = load_latest_portfolio_snapshot(db_path)

    assert loaded["reason_codes"].tolist() == ["score_improved, rrg_seed_improving"]
    assert loaded["data_warnings"].tolist() == [""]
