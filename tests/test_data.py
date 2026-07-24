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
    load_paper_experiments,
    load_paper_fills,
    load_paper_positions,
    load_paper_rebalances,
    migrate_legacy_pending_paper_rebalances,
    rebuild_paper_nav,
    paper_valuation_symbols,
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
    legacy_portfolio_id = default_paper_portfolio(1_000.0, db_path)
    rebalance_id = approve_paper_rebalance(
        run_id,
        {"holdings": [{"symbol": "NVDA", "weight": 0.8}], "cash_weight": 0.2},
        {"paper_capital": 1_000.0, "slippage_bps": 0.0, "commission_bps": 0.0},
        pd.Timestamp("2026-01-02").date(),
        legacy_portfolio_id,
        db_path,
    )
    experiments = load_paper_experiments(db_path)
    assert len(experiments) == 1
    portfolio_id = int(experiments.iloc[0]["portfolio_id"])
    assert portfolio_id != legacy_portfolio_id
    assert experiments.iloc[0]["run_id"] == run_id
    assert experiments.iloc[0]["initial_capital"] == pytest.approx(100_000.0)
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
    assert positions["shares"].iloc[0] == pytest.approx(8_000.0)
    rebuild_paper_nav(prices, portfolio_id, db_path)
    nav = load_paper_nav(portfolio_id, db_path)
    assert nav["nav"].iloc[-1] == pytest.approx(108_000.0)
    assert "TECHNICAL" in load_paper_benchmarks(portfolio_id, db_path)["symbol"].tolist()
    valuation_symbols = paper_valuation_symbols(portfolio_id, db_path)
    assert {"NVDA", "SPY", "QQQ", "GLD", "BTC-USD"}.issubset(valuation_symbols)
    import duckdb
    with duckdb.connect(str(db_path), read_only=True) as conn:
        first_rebuilt_at = conn.execute("SELECT rebuilt_at FROM lab_paper_valuation_state WHERE portfolio_id = ?", [portfolio_id]).fetchone()[0]
    rebuild_paper_nav(prices, portfolio_id, db_path)
    with duckdb.connect(str(db_path), read_only=True) as conn:
        second_rebuilt_at = conn.execute("SELECT rebuilt_at FROM lab_paper_valuation_state WHERE portfolio_id = ?", [portfolio_id]).fetchone()[0]
    assert second_rebuilt_at == first_rebuilt_at


def test_unapproved_run_cannot_change_paper_positions(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    default_paper_portfolio(1_000.0, db_path)
    prices = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    assert execute_pending_paper_rebalances(prices, db_path) == []
    assert load_paper_positions(db_path=db_path).empty


def _proposal_run(db_path: Path, symbol: str, as_of_date: str = "2026-01-02") -> int:
    packet_hash = save_lab_research_snapshot(
        {"as_of_date": as_of_date, "candidates": [{"symbol": symbol, "score": 99.0}]}, db_path
    )
    return save_lab_run(
        {"as_of_date": as_of_date, "benchmark": "SPY", "model": "gpt-5.5", "status": "proposal_ready", "packet_hash": packet_hash},
        db_path,
    )


def test_each_approval_creates_an_isolated_100k_experiment(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    legacy_portfolio_id = default_paper_portfolio(1_000.0, db_path)
    first_run, second_run = _proposal_run(db_path, "AAA"), _proposal_run(db_path, "BBB")
    first_rebalance = approve_paper_rebalance(first_run, {"holdings": [{"symbol": "AAA", "weight": 0.8}], "cash_weight": 0.2}, {"slippage_bps": 0.0, "commission_bps": 0.0}, pd.Timestamp("2026-01-02").date(), legacy_portfolio_id, db_path)
    second_rebalance = approve_paper_rebalance(second_run, {"holdings": [{"symbol": "BBB", "weight": 0.6}], "cash_weight": 0.4}, {"slippage_bps": 0.0, "commission_bps": 0.0}, pd.Timestamp("2026-01-02").date(), legacy_portfolio_id, db_path)

    experiments = load_paper_experiments(db_path).sort_values("run_id")
    assert experiments["run_id"].tolist() == [first_run, second_run]
    assert experiments["initial_capital"].tolist() == pytest.approx([100_000.0, 100_000.0])
    assert experiments["portfolio_id"].nunique() == 2
    assert legacy_portfolio_id not in experiments["portfolio_id"].tolist()
    first_portfolio, second_portfolio = experiments["portfolio_id"].astype(int).tolist()
    assert load_paper_rebalances(first_portfolio, db_path)["rebalance_id"].tolist() == [first_rebalance]
    assert load_paper_rebalances(second_portfolio, db_path)["rebalance_id"].tolist() == [second_rebalance]


def test_paper_experiment_includes_frozen_sensitivity_and_basket_summary(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    run_id = _proposal_run(db_path, "NVDA")
    approve_paper_rebalance(
        run_id,
        {
            "holdings": [
                {"symbol": "NVDA", "weight": 0.60, "sector": "SMH Universe"},
                {"symbol": "LLY", "weight": 0.40, "sector": "Healthcare Universe"},
            ],
            "cash_weight": 0.0,
        },
        {"max_turnover": 0.60, "slippage_bps": 0.0, "commission_bps": 0.0},
        pd.Timestamp("2026-01-02").date(),
        db_path=db_path,
    )

    experiment = load_paper_experiments(db_path).iloc[0]

    assert experiment["sensitivity"] == "Aggressive"
    assert experiment["basket_description"] == "Long chips & healthcare"


def test_legacy_pending_runs_23_and_24_migrate_idempotently_before_the_next_open(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    legacy_portfolio_id = default_paper_portfolio(db_path=db_path)
    import duckdb

    frozen_decisions = {
        23: '{"holdings": [{"symbol": "AAA", "weight": 0.8}], "cash_weight": 0.2}',
        24: '{"holdings": [{"symbol": "BBB", "weight": 0.6}], "cash_weight": 0.4}',
    }
    with duckdb.connect(str(db_path)) as conn:
        for run_id in [23, 24]:
            conn.execute(
                """INSERT INTO lab_runs (run_id, created_at, as_of_date, benchmark, model, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [run_id, pd.Timestamp("2026-07-13 10:00"), pd.Timestamp("2026-07-13").date(), "SPY", f"model-{run_id}", "proposal_ready"],
            )
            conn.execute(
                """INSERT INTO lab_paper_rebalances
                   (portfolio_id, run_id, approved_at, status, config_json, decision_json, note)
                   VALUES (?, ?, ?, 'pending', ?, ?, ?)""",
                [legacy_portfolio_id, run_id, pd.Timestamp("2026-07-13 11:00"), '{"slippage_bps": 5}', frozen_decisions[run_id], "Awaiting next open after 2026-07-13"],
            )

    migrated = migrate_legacy_pending_paper_rebalances(db_path)
    assert len(migrated) == 2
    assert migrate_legacy_pending_paper_rebalances(db_path) == []
    experiments = load_paper_experiments(db_path).sort_values("run_id")
    assert experiments["run_id"].tolist() == [23, 24]
    assert experiments["has_pending"].tolist() == [True, True]
    assert all(name.startswith("Run ") and "model-" in name for name in experiments["name"])
    assert load_paper_rebalances(legacy_portfolio_id, db_path).empty
    for _, experiment in experiments.iterrows():
        rebalance = load_paper_rebalances(int(experiment["portfolio_id"]), db_path).iloc[0]
        assert rebalance["status"] == "pending"
        assert rebalance["decision_json"] == frozen_decisions[int(experiment["run_id"])]
    no_next_open = pd.DataFrame({"symbol": ["AAA", "BBB"], "date": pd.to_datetime(["2026-07-13", "2026-07-13"]), "open": [10.0, 20.0], "high": [10.0, 20.0], "low": [10.0, 20.0], "close": [10.0, 20.0], "volume": [1_000_000, 1_000_000]})
    assert execute_pending_paper_rebalances(no_next_open, db_path) == []


def test_pending_experiments_fill_and_value_independently(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    first_run, second_run = _proposal_run(db_path, "AAA"), _proposal_run(db_path, "BBB")
    first_rebalance = approve_paper_rebalance(first_run, {"holdings": [{"symbol": "AAA", "weight": 0.8}], "cash_weight": 0.2}, {"slippage_bps": 0.0, "commission_bps": 0.0}, pd.Timestamp("2026-01-02").date(), db_path=db_path)
    second_rebalance = approve_paper_rebalance(second_run, {"holdings": [{"symbol": "BBB", "weight": 0.7}], "cash_weight": 0.3}, {"slippage_bps": 0.0, "commission_bps": 0.0}, pd.Timestamp("2026-01-02").date(), db_path=db_path)
    prices = pd.DataFrame({
        "symbol": ["AAA", "BBB", "SPY", "QQQ", "GLD", "BTC-USD"] * 2,
        "date": pd.to_datetime(["2026-01-02"] * 6 + ["2026-01-03"] * 6),
        "open": [10.0, 20.0, 100.0, 200.0, 50.0, 40_000.0] * 2,
        "high": [11.0, 22.0, 101.0, 202.0, 51.0, 40_100.0] * 2,
        "low": [10.0, 20.0, 99.0, 198.0, 49.0, 39_900.0] * 2,
        "close": [10.0, 20.0, 100.0, 200.0, 50.0, 40_000.0, 11.0, 22.0, 101.0, 202.0, 51.0, 40_100.0],
        "volume": [1_000_000] * 12,
    })
    assert execute_pending_paper_rebalances(prices, db_path) == [first_rebalance, second_rebalance]
    experiments = load_paper_experiments(db_path).sort_values("run_id")
    first_portfolio, second_portfolio = experiments["portfolio_id"].astype(int).tolist()
    assert load_paper_positions(first_portfolio, db_path)["symbol"].tolist() == ["AAA"]
    assert load_paper_positions(second_portfolio, db_path)["symbol"].tolist() == ["BBB"]
    for portfolio_id in [first_portfolio, second_portfolio]:
        rebuild_paper_nav(prices, portfolio_id, db_path)
        assert not load_paper_nav(portfolio_id, db_path).empty
        assert {"SPY", "QQQ", "GLD", "BTC-USD"}.issubset(load_paper_benchmarks(portfolio_id, db_path)["symbol"])
    assert {"AAA", "BBB", "SPY", "QQQ", "GLD", "BTC-USD"}.issubset(paper_valuation_symbols(db_path=db_path))


def test_new_approval_cannot_change_a_filled_experiment(tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    first_run = _proposal_run(db_path, "AAA")
    approve_paper_rebalance(first_run, {"holdings": [{"symbol": "AAA", "weight": 1.0}], "cash_weight": 0.0}, {"slippage_bps": 0.0, "commission_bps": 0.0}, pd.Timestamp("2026-01-02").date(), db_path=db_path)
    prices = pd.DataFrame({"symbol": ["AAA", "AAA", "BBB", "BBB"], "date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-02", "2026-01-03"]), "open": [10.0, 10.0, 20.0, 20.0], "high": [10.0, 11.0, 20.0, 21.0], "low": [10.0, 10.0, 20.0, 20.0], "close": [10.0, 11.0, 20.0, 21.0], "volume": [1_000_000] * 4})
    execute_pending_paper_rebalances(prices, db_path)
    first_portfolio = int(load_paper_experiments(db_path).iloc[0]["portfolio_id"])
    before = load_paper_positions(first_portfolio, db_path).copy()
    rebuild_paper_nav(prices, first_portfolio, db_path)
    nav_before = load_paper_nav(first_portfolio, db_path).copy()
    fills_before = load_paper_fills(first_portfolio, db_path).copy()
    second_run = _proposal_run(db_path, "BBB")
    approve_paper_rebalance(second_run, {"holdings": [{"symbol": "BBB", "weight": 1.0}], "cash_weight": 0.0}, {"slippage_bps": 0.0, "commission_bps": 0.0}, pd.Timestamp("2026-01-02").date(), first_portfolio, db_path)
    execute_pending_paper_rebalances(prices, db_path)
    pd.testing.assert_frame_equal(before, load_paper_positions(first_portfolio, db_path))
    pd.testing.assert_frame_equal(fills_before, load_paper_fills(first_portfolio, db_path))
    pd.testing.assert_frame_equal(nav_before, rebuild_paper_nav(prices, first_portfolio, db_path))


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
