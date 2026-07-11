from pathlib import Path

import pandas as pd
import pytest

from stockpicker.data import (
    init_database,
    load_latest_portfolio_snapshot,
    load_matching_portfolio_snapshot,
    load_portfolio_rebalance_log,
    read_cached_prices,
    save_portfolio_snapshot,
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
