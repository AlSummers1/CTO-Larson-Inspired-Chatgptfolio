from pathlib import Path

import pandas as pd
import pytest

import stockpicker.data as data_module

from stockpicker.data import (
    PRICE_COLUMNS,
    download_prices,
    fetch_price_history,
    init_database,
    load_fetch_log,
    load_latest_portfolio_snapshot,
    load_matching_portfolio_snapshot,
    load_portfolio_rebalance_log,
    read_cached_prices,
    save_portfolio_snapshot,
    write_prices,
)


pytest.importorskip("duckdb")


def downloaded_frame(column_order: str) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=2, tz="UTC")
    values = {
        "AAA": {
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Close": [10.25, 11.25],
            "Volume": [1_000, 1_100],
        },
        "BBB": {
            "Open": [20.0, 21.0],
            "High": [20.5, 21.5],
            "Low": [19.5, 20.5],
            "Close": [20.25, 21.25],
            "Volume": [2_000, 2_100],
        },
    }
    columns = {}
    for symbol, fields in values.items():
        for field, field_values in fields.items():
            key = (symbol, field) if column_order == "ticker_first" else (field, symbol)
            columns[key] = field_values
    frame = pd.DataFrame(columns, index=dates)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


@pytest.mark.parametrize("column_order", ["ticker_first", "field_first"])
def test_download_prices_supports_yfinance_multiindex_orientations(monkeypatch, column_order: str):
    raw = downloaded_frame(column_order)
    calls = []

    class FakeYFinance:
        @staticmethod
        def download(**kwargs):
            calls.append(kwargs)
            return raw

    monkeypatch.setattr(data_module, "_import_yfinance", lambda: FakeYFinance())

    prices = download_prices(["bbb", "AAA"], years=5)

    assert calls[0]["tickers"] == "BBB AAA"
    assert calls[0]["period"] == "5y"
    assert prices.groupby("symbol")["close"].apply(list).to_dict() == {
        "AAA": [10.25, 11.25],
        "BBB": [20.25, 21.25],
    }
    assert prices["date"].dt.tz is None


def test_download_prices_skips_symbol_without_close(monkeypatch):
    raw = pd.DataFrame(
        {"Open": [10.0], "Volume": [1_000]},
        index=pd.bdate_range("2024-01-01", periods=1),
    )

    class FakeYFinance:
        @staticmethod
        def download(**_kwargs):
            return raw

    monkeypatch.setattr(data_module, "_import_yfinance", lambda: FakeYFinance())

    prices = download_prices(["AAA"])

    assert prices.empty
    assert prices.columns.tolist() == PRICE_COLUMNS


def test_fetch_price_history_downloads_only_cache_misses(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    cached = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": pd.to_datetime(["2024-01-02"]),
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.25],
            "volume": [1_000],
        }
    )
    downloaded = cached.assign(
        symbol="BBB",
        open=20.0,
        high=20.5,
        low=19.5,
        close=20.25,
        volume=2_000,
    )
    write_prices(cached, db_path)
    calls = []

    def fake_download(symbols, years=10):
        calls.append((symbols, years))
        return downloaded

    monkeypatch.setattr(data_module, "download_prices", fake_download)

    prices = fetch_price_history(["aaa", "BBB"], years=7, db_path=db_path)
    log = load_fetch_log(db_path)

    assert calls == [(["BBB"], 7)]
    assert prices["symbol"].tolist() == ["AAA", "BBB"]
    assert log[["symbol", "status", "rows"]].to_dict("records") == [
        {"symbol": "BBB", "status": "ok", "rows": 1}
    ]


def test_failed_refresh_preserves_cache_and_records_failure(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "market_data.duckdb"
    cached = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": pd.to_datetime(["2024-01-02"]),
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.25],
            "volume": [1_000],
        }
    )
    write_prices(cached, db_path)
    monkeypatch.setattr(
        data_module,
        "download_prices",
        lambda _symbols, years=10: pd.DataFrame(columns=PRICE_COLUMNS),
    )

    prices = fetch_price_history(["AAA"], refresh=True, db_path=db_path)
    log = load_fetch_log(db_path)

    assert prices["close"].tolist() == [10.25]
    assert log.loc[0, "symbol"] == "AAA"
    assert log.loc[0, "status"] == "failed"
    assert log.loc[0, "rows"] == 0
    assert "No rows returned" in log.loc[0, "message"]


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


def test_database_path_can_be_redirected_for_container_storage(monkeypatch, tmp_path: Path):
    container_path = tmp_path / "state" / "market_data.duckdb"
    monkeypatch.setenv("CHATGPTFOLIO_DB_PATH", str(container_path))

    assert data_module.configured_db_path() == container_path


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


def test_existing_database_upgrades_additively_and_idempotently(tmp_path: Path):
    db_path = tmp_path / "legacy.duckdb"
    import duckdb

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE prices (
                symbol VARCHAR, date DATE, open DOUBLE, high DOUBLE,
                low DOUBLE, close DOUBLE, volume DOUBLE,
                PRIMARY KEY (symbol, date)
            )
            """
        )
        conn.execute("INSERT INTO prices VALUES ('AAA', '2024-01-05', 1, 1, 1, 1, 100)")

    init_database(db_path)
    init_database(db_path)

    with duckdb.connect(str(db_path)) as conn:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        preserved = conn.execute("SELECT symbol, close FROM prices").fetchall()

    assert {"analysis_runs", "signal_snapshots", "signal_events", "weekly_reports", "saved_weekly_briefs"}.issubset(tables)
    assert preserved == [("AAA", 1.0)]
