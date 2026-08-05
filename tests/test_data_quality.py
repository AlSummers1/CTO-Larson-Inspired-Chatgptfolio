import pandas as pd
import pytest

from stockpicker.data_quality import DERIVED_200W_NOTE, compute_data_health


def prices_for(calendar, rows):
    records = []
    for symbol, dates, closes in rows:
        for date, close in zip(dates, closes):
            records.append({"symbol": symbol, "date": pd.Timestamp(date), "close": close})
    return pd.DataFrame(records)


def test_benchmark_calendar_excludes_weekends_and_holidays():
    benchmark_dates = pd.to_datetime(["2024-07-03", "2024-07-05", "2024-07-08"])
    prices = prices_for(
        benchmark_dates,
        [
            ("QQQ", benchmark_dates, [100, 101, 102]),
            ("AAA", benchmark_dates[:2], [10, 11]),
        ],
    )

    health = compute_data_health(prices, "QQQ", lookback_sessions=3).set_index("symbol")

    assert health.loc["AAA", "trading_sessions_behind"] == 1
    assert health.loc["AAA", "freshness_status"] == "Delayed"
    assert health.loc["AAA", "missing_sessions_60"] == 1
    assert "2024-07-04" not in health.loc["AAA", "missing_session_dates"]


def test_one_session_is_delayed_and_two_sessions_are_stale():
    dates = pd.bdate_range("2024-01-01", periods=4)
    prices = prices_for(
        dates,
        [
            ("QQQ", dates, [100, 101, 102, 103]),
            ("ONE", dates[:-1], [10, 11, 12]),
            ("TWO", dates[:-2], [20, 21]),
        ],
    )

    health = compute_data_health(prices, "QQQ").set_index("symbol")

    assert health.loc["QQQ", "freshness_status"] == "Fresh"
    assert health.loc["ONE", "freshness_status"] == "Delayed"
    assert health.loc["TWO", "freshness_status"] == "Stale"


def test_missing_benchmark_uses_approximate_broad_market_fallback():
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = prices_for(
        dates,
        [
            ("AAA", dates, [10, 11, 12]),
            ("BBB", dates, [20, 21, 22]),
        ],
    )

    health = compute_data_health(prices, "MISSING")

    assert health["calendar_approximate"].all()
    assert set(health["calendar_source"]) == {"broad-market fallback"}
    assert health["reference_market_date"].max() == dates[-1]


def test_coverage_missing_rows_fetch_status_and_unchanged_close_streak():
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = prices_for(
        dates,
        [
            ("QQQ", dates, [100, 101, 102, 103, 104]),
            ("AAA", dates.delete(2), [10, 11, 12, 12]),
        ],
    )
    fetch_log = pd.DataFrame(
        [
            {"symbol": "AAA", "fetched_at": "2024-01-05 10:00", "status": "ok"},
            {"symbol": "AAA", "fetched_at": "2024-01-06 10:00", "status": "failed"},
        ]
    )

    health = compute_data_health(prices, "QQQ", lookback_sessions=5, fetch_log=fetch_log).set_index("symbol")

    assert health.loc["AAA", "coverage_pct_60"] == pytest.approx(80.0)
    assert health.loc["AAA", "missing_sessions_60"] == 1
    assert health.loc["AAA", "fetch_status"] == "failed"
    assert health.loc["AAA", "latest_successful_fetch"] == pd.Timestamp("2024-01-05 10:00")
    assert health.loc["AAA", "unchanged_close_streak"] == 1
    assert health.loc["AAA", "derived_series_note"] == DERIVED_200W_NOTE


def test_invalid_lookback_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        compute_data_health(pd.DataFrame(), "QQQ", lookback_sessions=0)
