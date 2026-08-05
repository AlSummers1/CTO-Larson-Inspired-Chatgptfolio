import pandas as pd
import pytest

from stockpicker.signals import (
    RRGConfig,
    SignalConfig,
    compute_rrg_basket_history,
    compute_rrg_history,
    compute_signal_table,
    equal_weight_basket_history,
    larsson_regime_history,
    latest_drawdown,
    percent_extension,
    relative_strength_metrics,
    resample_weekly_close,
    rolling_sma,
    slope_over,
)


def make_prices(symbol: str, periods: int, start: float, step: float) -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-01", periods=periods)
    closes = [start + step * index for index in range(periods)]
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1_000_000,
        }
    )


def make_prices_from_closes(symbol: str, closes: list[float], start: str = "2022-01-03") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1_000_000,
        }
    )


def make_weekly_prices(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-03", periods=len(closes), freq="W-FRI")
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1_000_000,
        }
    )


def rrg_fixture(symbol: str, relative_line: list[float], benchmark_start: int = 0) -> pd.DataFrame:
    benchmark = make_weekly_prices("QQQ", [100.0] * len(relative_line)).iloc[benchmark_start:].copy()
    stock = make_weekly_prices(symbol, [100.0 * value for value in relative_line])
    return pd.concat([stock, benchmark], ignore_index=True)


def test_rolling_sma_uses_full_window():
    series = pd.Series([1, 2, 3, 4])
    sma = rolling_sma(series, 3)
    assert pd.isna(sma.iloc[1])
    assert sma.iloc[-1] == pytest.approx(3)


def test_slope_and_extension_math():
    series = pd.Series([10, 11, 12, 15])
    assert slope_over(series, 2) == pytest.approx(4)
    assert percent_extension(120, 100) == pytest.approx(0.20)


def test_drawdown_math():
    close = pd.Series([100, 120, 90, 108])
    assert latest_drawdown(close) == pytest.approx(-0.10)
    assert latest_drawdown(close, lookback=2) == pytest.approx(0.0)


def test_equal_weight_basket_history_uses_normalized_common_start():
    stock_a = make_prices_from_closes("AAA", [100.0, 110.0, 120.0])
    stock_b = make_prices_from_closes("BBB", [50.0, 55.0, 50.0])
    benchmark = make_prices_from_closes("QQQ", [200.0, 202.0, 204.0])

    basket, used, missing = equal_weight_basket_history(
        pd.concat([stock_a, stock_b, benchmark], ignore_index=True),
        ["AAA", "BBB"],
        "QQQ",
    )

    assert used == ["AAA", "BBB"]
    assert missing == []
    assert basket["date"].iloc[0] == stock_a["date"].iloc[0]
    assert basket["basket_index"].tolist() == pytest.approx([1.0, 1.1, 1.1])


def test_equal_weight_basket_history_skips_missing_symbols():
    stock = make_prices_from_closes("AAA", [100.0, 110.0, 120.0])

    basket, used, missing = equal_weight_basket_history(stock, ["AAA", "MISSING"], "QQQ")

    assert used == ["AAA"]
    assert missing == ["MISSING"]
    assert basket["basket_index"].tolist() == pytest.approx([1.0, 1.1, 1.2])


def test_equal_weight_basket_deduplicates_and_excludes_benchmark():
    stock = make_prices_from_closes("AAA", [100.0, 110.0, 120.0])
    benchmark = make_prices_from_closes("QQQ", [200.0, 210.0, 220.0])

    basket, used, missing = equal_weight_basket_history(
        pd.concat([stock, benchmark], ignore_index=True),
        ["AAA", "AAA", "QQQ", "MISSING", "MISSING"],
        "QQQ",
    )

    assert used == ["AAA"]
    assert missing == ["MISSING"]
    assert basket["basket_index"].tolist() == pytest.approx([1.0, 1.1, 1.2])


def test_equal_weight_basket_rebases_at_common_history_start():
    older_stock = make_prices_from_closes("AAA", [100.0, 110.0, 120.0, 130.0])
    newer_stock = make_prices_from_closes("BBB", [50.0, 55.0], start="2022-01-05")

    basket, used, missing = equal_weight_basket_history(
        pd.concat([older_stock, newer_stock], ignore_index=True),
        ["AAA", "BBB"],
        "QQQ",
    )

    assert used == ["AAA", "BBB"]
    assert missing == []
    assert basket["date"].tolist() == newer_stock["date"].tolist()
    assert basket["basket_index"].tolist() == pytest.approx([1.0, (130 / 120 + 55 / 50) / 2])


def test_equal_weight_basket_drawdown_uses_basket_index():
    stock_a = make_prices_from_closes("AAA", [100.0, 200.0, 100.0])
    stock_b = make_prices_from_closes("BBB", [100.0, 100.0, 100.0])

    basket, _, _ = equal_weight_basket_history(
        pd.concat([stock_a, stock_b], ignore_index=True),
        ["AAA", "BBB"],
        "QQQ",
    )

    assert basket["basket_index"].tolist() == pytest.approx([1.0, 1.5, 1.0])
    assert basket["drawdown"].tolist() == pytest.approx([0.0, 0.0, -1 / 3])


def test_weekly_resampling_returns_friday_close():
    prices = make_prices("TEST", 10, 10, 1)
    weekly = resample_weekly_close(prices)
    assert len(weekly) == 2
    assert weekly.iloc[0] == pytest.approx(14)


def test_signal_table_identifies_uptrend_and_filters():
    stock = make_prices("NVDA", 1500, 20, 0.10)
    benchmark = make_prices("QQQ", 1500, 100, 0.05)
    signals = compute_signal_table(pd.concat([stock, benchmark], ignore_index=True), "QQQ", SignalConfig())
    row = signals[signals["symbol"] == "NVDA"].iloc[0]
    assert row["larsson_regime"] == "Bull"
    assert row["daily_trend"] == "Up"
    assert row["weekly_trend"] == "Up"
    assert row["monthly_trend"] == "Up"
    assert row["pass_200d"] == True
    assert row["pass_200w"] == True
    assert row["benchmark_larsson_regime"] == "Bull"


def test_signal_table_identifies_downtrend_and_warnings():
    stock = make_prices("WEAK", 260, 100, -0.10)
    signals = compute_signal_table(stock, "QQQ", SignalConfig())
    row = signals.iloc[0]
    assert row["daily_trend"] == "Down"
    assert row["pass_200d"] == False
    assert pd.isna(row["pass_200w"])
    assert "insufficient 200W history" in row["data_warnings"]


def test_larsson_regime_history_encodes_bull_bear_and_transition():
    config = SignalConfig(larsson_history_weeks=12)

    bull = larsson_regime_history(make_prices("BULL", 1500, 20, 0.10), "BULL", config)
    bear = larsson_regime_history(make_prices("BEAR", 1500, 300, -0.10), "BEAR", config)
    transition = larsson_regime_history(make_prices("FLAT", 1500, 100, 0), "FLAT", config)

    assert len(bull) == 12
    assert len(bear) == 12
    assert len(transition) == 12
    assert set(bull["larsson_regime_value"]) == {1.0}
    assert set(bear["larsson_regime_value"]) == {0.0}
    assert set(transition["larsson_regime_value"]) == {0.5}


def test_signal_table_includes_capped_larsson_history_columns():
    config = SignalConfig(larsson_history_weeks=9)
    signals = compute_signal_table(make_prices("NVDA", 1500, 20, 0.10), "QQQ", config)
    row = signals.iloc[0]

    assert len(row["larsson_regime_history"]) == 9
    assert set(row["larsson_regime_history"]).issubset({0.0, 0.5, 1.0})
    assert len(row["larsson_regime_history_labels"].split(", ")) == 9


def test_relative_strength_uses_aligned_3m_6m_12m_return_spreads():
    benchmark = make_prices_from_closes("QQQ", [100.0] * 253)
    stock = make_prices_from_closes("NVDA", [100.0] * 253)
    stock.loc[126:, "close"] = 120.0
    stock.loc[189:, "close"] = 130.0
    stock.loc[252:, "close"] = 140.0

    metrics = relative_strength_metrics(pd.concat([stock, benchmark], ignore_index=True), "NVDA", "QQQ")

    assert metrics["rs_3m"] == pytest.approx(140 / 130 - 1)
    assert metrics["rs_6m"] == pytest.approx(140 / 120 - 1)
    assert metrics["rs_12m"] == pytest.approx(0.40)
    assert metrics["relative_strength_score"] > 50
    assert metrics["relative_strength_label"] == "Outperforming"


def test_relative_strength_handles_misaligned_and_insufficient_history():
    benchmark = make_prices_from_closes("QQQ", [100.0] * 253, start="2022-01-03")
    stock = make_prices_from_closes("NVDA", [100.0] * 220, start="2022-02-01")
    stock.loc[stock.index[-1], "close"] = 120.0

    metrics = relative_strength_metrics(pd.concat([stock, benchmark], ignore_index=True), "NVDA", "QQQ")

    assert metrics["rs_3m"] > 0
    assert pd.isna(metrics["rs_12m"])
    assert metrics["relative_strength_score"] > 50


def test_relative_strength_neutral_when_benchmark_missing():
    stock = make_prices("NVDA", 50, 100, 1)
    metrics = relative_strength_metrics(stock, "NVDA", "QQQ")

    assert metrics["relative_strength_score"] == 50.0
    assert metrics["relative_strength_label"] == "Neutral"
    assert pd.isna(metrics["rs_3m"])


def test_signal_table_includes_relative_strength_columns():
    stock = make_prices("NVDA", 1500, 20, 0.10)
    benchmark = make_prices("QQQ", 1500, 100, 0.05)
    signals = compute_signal_table(pd.concat([stock, benchmark], ignore_index=True), "QQQ", SignalConfig())
    row = signals[signals["symbol"] == "NVDA"].iloc[0]

    assert row["rs_3m"] > 0
    assert row["rs_6m"] > 0
    assert row["rs_12m"] > 0
    assert row["relative_strength_score"] > 50


def test_rrg_history_classifies_all_quadrants():
    config = RRGConfig(ratio_window_weeks=4, momentum_window_weeks=4, tail_weeks=6)
    patterns = {
        "LEAD": [1 + (index / 39) ** 2 for index in range(40)],
        "WEAK": [1 + index * 0.02 for index in range(40)],
        "LAG": [2 - index * 0.02 for index in range(40)],
        "IMPR": [2 - 0.8 * ((index / 39) ** 0.3) for index in range(40)],
    }
    prices = pd.concat(
        [rrg_fixture(symbol, relative_line) for symbol, relative_line in patterns.items()],
        ignore_index=True,
    )

    rrg = compute_rrg_history(prices, list(patterns), "QQQ", config)
    latest = rrg.sort_values("date").groupby("symbol").tail(1).set_index("symbol")

    assert latest.loc["LEAD", "quadrant"] == "Leading"
    assert latest.loc["WEAK", "quadrant"] == "Weakening"
    assert latest.loc["LAG", "quadrant"] == "Lagging"
    assert latest.loc["IMPR", "quadrant"] == "Improving"


def test_rrg_history_trims_tails_and_handles_aligned_dates():
    config = RRGConfig(ratio_window_weeks=4, momentum_window_weeks=4, tail_weeks=5)
    prices = rrg_fixture("NVDA", [1 + (index / 39) ** 2 for index in range(40)], benchmark_start=4)

    rrg = compute_rrg_history(prices, ["NVDA"], "QQQ", config)

    assert len(rrg) == 5
    assert rrg["symbol"].unique().tolist() == ["NVDA"]
    assert rrg["date"].min() >= pd.Timestamp("2020-01-31")


def test_rrg_history_skips_insufficient_history():
    config = RRGConfig(ratio_window_weeks=10, momentum_window_weeks=10, tail_weeks=6)
    prices = rrg_fixture("NVDA", [1.0, 1.1, 1.2, 1.3])

    rrg = compute_rrg_history(prices, ["NVDA"], "QQQ", config)

    assert rrg.empty


def test_rrg_basket_history_uses_equal_weight_normalized_basket():
    config = RRGConfig(ratio_window_weeks=4, momentum_window_weeks=4, tail_weeks=8)
    benchmark = make_weekly_prices("QQQ", [100.0] * 40)
    low_price_winner = make_weekly_prices("LOW", [10.0 * (1 + index / 39) for index in range(40)])
    high_price_flat = make_weekly_prices("HIGH", [10_000.0] * 40)
    prices = pd.concat([benchmark, low_price_winner, high_price_flat], ignore_index=True)

    rrg = compute_rrg_basket_history(prices, {"Test Universe": ["LOW", "HIGH"]}, "QQQ", config)

    low_norm = resample_weekly_close(low_price_winner) / 10.0
    high_norm = resample_weekly_close(high_price_flat) / 10_000.0
    expected_basket = (low_norm + high_norm) / 2
    relative_line = expected_basket / resample_weekly_close(benchmark)
    expected_ratio = 100 * relative_line / rolling_sma(relative_line, config.ratio_window_weeks)
    expected_momentum = 100 * expected_ratio / rolling_sma(expected_ratio, config.momentum_window_weeks)

    latest = rrg.sort_values("date").iloc[-1]
    assert latest["rs_ratio"] == pytest.approx(expected_ratio.dropna().iloc[-1])
    assert latest["rs_momentum"] == pytest.approx(expected_momentum.dropna().iloc[-1])
    assert latest["symbol_count"] == 2
    assert latest["missing_count"] == 0


def test_rrg_basket_history_skips_missing_symbols_and_trims_tail():
    config = RRGConfig(ratio_window_weeks=4, momentum_window_weeks=4, tail_weeks=5)
    prices = pd.concat(
        [
            make_weekly_prices("QQQ", [100.0] * 40),
            make_weekly_prices("AAA", [100.0 + index for index in range(40)]),
        ],
        ignore_index=True,
    )

    rrg = compute_rrg_basket_history(prices, {"Seed": ["AAA", "MISSING"]}, "QQQ", config)

    assert len(rrg) == 5
    latest = rrg.sort_values("date").iloc[-1]
    assert latest["symbols_used"] == "AAA"
    assert latest["symbols_missing"] == "MISSING"
    assert latest["symbol_count"] == 1
    assert latest["missing_count"] == 1


def test_rrg_basket_history_handles_aligned_dates():
    config = RRGConfig(ratio_window_weeks=4, momentum_window_weeks=4, tail_weeks=6)
    benchmark = make_weekly_prices("QQQ", [100.0] * 40)
    early = make_weekly_prices("EARLY", [100.0 + index for index in range(40)])
    late = make_weekly_prices("LATE", [50.0 + index for index in range(30)])
    late["date"] = late["date"] + pd.offsets.Week(10)
    prices = pd.concat([benchmark, early, late], ignore_index=True)

    rrg = compute_rrg_basket_history(prices, {"Aligned": ["EARLY", "LATE"]}, "QQQ", config)

    assert not rrg.empty
    assert rrg["date"].min() >= late["date"].min()


def test_rrg_basket_history_classifies_all_quadrants():
    config = RRGConfig(ratio_window_weeks=4, momentum_window_weeks=4, tail_weeks=6)
    patterns = {
        "Leading Seed": [1 + (index / 39) ** 2 for index in range(40)],
        "Weakening Seed": [1 + index * 0.02 for index in range(40)],
        "Lagging Seed": [2 - index * 0.02 for index in range(40)],
        "Improving Seed": [2 - 0.8 * ((index / 39) ** 0.3) for index in range(40)],
    }
    prices = [make_weekly_prices("QQQ", [100.0] * 40)]
    seed_map = {}
    for index, (seed_name, relative_line) in enumerate(patterns.items()):
        symbol = f"SEED{index}"
        prices.append(make_weekly_prices(symbol, [100.0 * value for value in relative_line]))
        seed_map[seed_name] = [symbol]

    rrg = compute_rrg_basket_history(pd.concat(prices, ignore_index=True), seed_map, "QQQ", config)
    latest = rrg.sort_values("date").groupby("seed_name").tail(1).set_index("seed_name")

    assert latest.loc["Leading Seed", "quadrant"] == "Leading"
    assert latest.loc["Weakening Seed", "quadrant"] == "Weakening"
    assert latest.loc["Lagging Seed", "quadrant"] == "Lagging"
    assert latest.loc["Improving Seed", "quadrant"] == "Improving"
