from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TREND_STATES = ("Up", "Mixed", "Down")
LARSSON_STATES = ("Bull", "Transition", "Bear")
LARSSON_HISTORY_VALUES = {
    "Bear": 0.0,
    "Transition": 0.5,
    "Bull": 1.0,
}
RELATIVE_STRENGTH_WINDOWS = {
    "3m": 63,
    "6m": 126,
    "12m": 252,
}


@dataclass(frozen=True)
class SignalConfig:
    daily_trend_window: int = 50
    daily_slope_days: int = 20
    daily_filter_window: int = 200
    weekly_trend_window: int = 30
    weekly_slope_weeks: int = 10
    weekly_filter_window: int = 200
    monthly_trend_window: int = 10
    monthly_slope_months: int = 4
    drawdown_lookback_days: int = 252
    larsson_slope_days: int = 20
    larsson_history_weeks: int = 26


@dataclass(frozen=True)
class RRGConfig:
    ratio_window_weeks: int = 10
    momentum_window_weeks: int = 10
    tail_weeks: int = 12


RRG_COLUMNS = ["symbol", "date", "rs_ratio", "rs_momentum", "quadrant"]
RRG_BASKET_COLUMNS = [
    "seed_name",
    "date",
    "rs_ratio",
    "rs_momentum",
    "quadrant",
    "symbols_used",
    "symbols_missing",
    "symbol_count",
    "missing_count",
]
BASKET_COLUMNS = ["date", "basket_index", "drawdown"]


def rolling_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average with full-window requirements."""
    return series.rolling(window=window, min_periods=window).mean()


def resample_weekly_close(prices: pd.DataFrame) -> pd.Series:
    """Convert daily prices to Friday-ending weekly closes."""
    return prices.set_index("date")["close"].resample("W-FRI").last().dropna()


def resample_monthly_close(prices: pd.DataFrame) -> pd.Series:
    """Convert daily prices to month-end closes."""
    return prices.set_index("date")["close"].resample("ME").last().dropna()


def slope_over(series: pd.Series, periods: int) -> float:
    valid = series.dropna()
    if len(valid) <= periods:
        return float("nan")
    return float(valid.iloc[-1] - valid.iloc[-periods - 1])


def percent_extension(price: float, average: float) -> float:
    if pd.isna(price) or pd.isna(average) or average == 0:
        return float("nan")
    return float(price / average - 1)


def total_return_over(close: pd.Series, periods: int) -> float:
    """Return total return over a trading-day lookback."""
    valid = close.dropna()
    if len(valid) <= periods:
        return float("nan")
    start = float(valid.iloc[-periods - 1])
    end = float(valid.iloc[-1])
    if start <= 0:
        return float("nan")
    return float(end / start - 1)


def latest_drawdown(close: pd.Series, lookback: int | None = None) -> float:
    series = close.dropna()
    if lookback is not None:
        series = series.tail(lookback)
    if series.empty:
        return float("nan")
    peak = float(series.max())
    if peak <= 0:
        return float("nan")
    return float(series.iloc[-1] / peak - 1)


def drawdown_history(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    daily = _daily_for_symbol(prices, symbol)
    if daily.empty:
        return pd.DataFrame(columns=["date", "drawdown"])
    close = daily.set_index("date")["close"]
    drawdown = close / close.cummax() - 1
    return drawdown.rename("drawdown").reset_index()


def equal_weight_basket_history(
    prices: pd.DataFrame,
    symbols: list[str],
    benchmark: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a normalized equal-weight basket for active non-benchmark symbols."""
    frames = []
    symbols_used: list[str] = []
    missing_symbols: list[str] = []
    seen_symbols = []
    for symbol in symbols:
        if symbol not in seen_symbols:
            seen_symbols.append(symbol)

    for symbol in seen_symbols:
        if symbol == benchmark:
            continue
        daily = _daily_for_symbol(prices, symbol)
        if daily.empty:
            missing_symbols.append(symbol)
            continue
        close = daily.set_index("date")["close"].dropna()
        close = close[close > 0]
        if close.empty:
            missing_symbols.append(symbol)
            continue
        frames.append(close.rename(symbol))
        symbols_used.append(symbol)

    if not frames:
        return pd.DataFrame(columns=BASKET_COLUMNS), [], missing_symbols

    aligned = pd.concat(frames, axis=1, join="inner").dropna(how="any")
    if aligned.empty:
        return pd.DataFrame(columns=BASKET_COLUMNS), [], symbols_used + missing_symbols

    normalized = aligned.divide(aligned.iloc[0])
    basket = normalized.mean(axis=1)
    drawdown = basket / basket.cummax() - 1
    history = pd.DataFrame(
        {
            "date": basket.index,
            "basket_index": basket.values,
            "drawdown": drawdown.values,
        }
    )
    return history.reset_index(drop=True), symbols_used, missing_symbols


def _daily_for_symbol(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    frame = prices[prices["symbol"] == symbol].copy()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def _state_from_price_average_slope(price: float, average: float, slope: float) -> str:
    if pd.isna(price) or pd.isna(average) or pd.isna(slope):
        return "Mixed"
    if price > average and slope >= 0:
        return "Up"
    if price < average and slope < 0:
        return "Down"
    return "Mixed"


def _larsson_state(price: float, proxy: float, slope: float) -> str:
    if pd.isna(price) or pd.isna(proxy) or pd.isna(slope):
        return "Transition"
    if price > proxy and slope >= 0:
        return "Bull"
    if price < proxy and slope < 0:
        return "Bear"
    return "Transition"


def _latest(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    value = series.iloc[-1]
    return float(value) if not pd.isna(value) else float("nan")


def _bool_or_none(price: float, average: float) -> bool | None:
    if pd.isna(price) or pd.isna(average):
        return None
    return bool(price > average)


def rrg_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    """Classify an RRG point around the 100/100 center."""
    if pd.isna(rs_ratio) or pd.isna(rs_momentum):
        return "n/a"
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "Leading"
    if rs_ratio >= 100 and rs_momentum < 100:
        return "Weakening"
    if rs_ratio < 100 and rs_momentum < 100:
        return "Lagging"
    return "Improving"


def compute_rrg_history(
    prices: pd.DataFrame,
    symbols: list[str],
    benchmark: str,
    config: RRGConfig | None = None,
) -> pd.DataFrame:
    """Compute weekly RRG-style ratio and momentum history versus a benchmark."""
    config = config or RRGConfig()
    if prices.empty:
        return pd.DataFrame(columns=RRG_COLUMNS)

    requested = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols if str(symbol).strip()))
    benchmark = benchmark.strip().upper()
    benchmark_daily = _daily_for_symbol(prices, benchmark)
    if benchmark_daily.empty:
        return pd.DataFrame(columns=RRG_COLUMNS)

    benchmark_weekly = resample_weekly_close(benchmark_daily).rename("benchmark_close")
    frames = []
    for symbol in requested:
        if symbol == benchmark:
            continue
        symbol_daily = _daily_for_symbol(prices, symbol)
        if symbol_daily.empty:
            continue

        symbol_weekly = resample_weekly_close(symbol_daily).rename("symbol_close")
        aligned = pd.concat([symbol_weekly, benchmark_weekly], axis=1, join="inner").dropna()
        if aligned.empty:
            continue

        aligned = aligned[(aligned["symbol_close"] > 0) & (aligned["benchmark_close"] > 0)].copy()
        relative_line = aligned["symbol_close"] / aligned["benchmark_close"]
        rs_ratio = 100 * relative_line / rolling_sma(relative_line, config.ratio_window_weeks)
        rs_momentum = 100 * rs_ratio / rolling_sma(rs_ratio, config.momentum_window_weeks)
        symbol_history = pd.DataFrame(
            {
                "symbol": symbol,
                "date": aligned.index,
                "rs_ratio": rs_ratio,
                "rs_momentum": rs_momentum,
            }
        ).dropna(subset=["rs_ratio", "rs_momentum"])
        if symbol_history.empty:
            continue

        symbol_history["quadrant"] = [
            rrg_quadrant(row.rs_ratio, row.rs_momentum)
            for row in symbol_history.itertuples(index=False)
        ]
        frames.append(symbol_history.tail(config.tail_weeks))

    if not frames:
        return pd.DataFrame(columns=RRG_COLUMNS)
    return pd.concat(frames, ignore_index=True)[RRG_COLUMNS]


def compute_rrg_basket_history(
    prices: pd.DataFrame,
    seed_universes: dict[str, list[str]],
    benchmark: str,
    config: RRGConfig | None = None,
) -> pd.DataFrame:
    """Compute RRG-style history for equal-weight seed-universe baskets."""
    config = config or RRGConfig()
    if prices.empty or not seed_universes:
        return pd.DataFrame(columns=RRG_BASKET_COLUMNS)

    benchmark = benchmark.strip().upper()
    benchmark_daily = _daily_for_symbol(prices, benchmark)
    if benchmark_daily.empty:
        return pd.DataFrame(columns=RRG_BASKET_COLUMNS)

    benchmark_weekly = resample_weekly_close(benchmark_daily).rename("benchmark_close")
    frames = []
    for seed_name, raw_symbols in seed_universes.items():
        requested = list(dict.fromkeys(symbol.strip().upper() for symbol in raw_symbols if str(symbol).strip()))
        weekly_frames = []
        symbols_used: list[str] = []
        symbols_missing: list[str] = []
        for symbol in requested:
            if symbol == benchmark:
                continue
            symbol_daily = _daily_for_symbol(prices, symbol)
            if symbol_daily.empty:
                symbols_missing.append(symbol)
                continue

            symbol_weekly = resample_weekly_close(symbol_daily)
            symbol_weekly = symbol_weekly[symbol_weekly > 0]
            if symbol_weekly.empty:
                symbols_missing.append(symbol)
                continue

            weekly_frames.append(symbol_weekly.rename(symbol))
            symbols_used.append(symbol)

        if not weekly_frames:
            continue

        aligned_symbols = pd.concat(weekly_frames, axis=1, join="inner").dropna(how="any")
        if aligned_symbols.empty:
            continue

        normalized = aligned_symbols.divide(aligned_symbols.iloc[0])
        basket = normalized.mean(axis=1).rename("basket_close")
        aligned = pd.concat([basket, benchmark_weekly], axis=1, join="inner").dropna()
        if aligned.empty:
            continue

        aligned = aligned[(aligned["basket_close"] > 0) & (aligned["benchmark_close"] > 0)].copy()
        relative_line = aligned["basket_close"] / aligned["benchmark_close"]
        rs_ratio = 100 * relative_line / rolling_sma(relative_line, config.ratio_window_weeks)
        rs_momentum = 100 * rs_ratio / rolling_sma(rs_ratio, config.momentum_window_weeks)
        seed_history = pd.DataFrame(
            {
                "seed_name": seed_name,
                "date": aligned.index,
                "rs_ratio": rs_ratio,
                "rs_momentum": rs_momentum,
            }
        ).dropna(subset=["rs_ratio", "rs_momentum"])
        if seed_history.empty:
            continue

        seed_history["quadrant"] = [
            rrg_quadrant(row.rs_ratio, row.rs_momentum)
            for row in seed_history.itertuples(index=False)
        ]
        seed_history["symbols_used"] = ", ".join(symbols_used)
        seed_history["symbols_missing"] = ", ".join(symbols_missing)
        seed_history["symbol_count"] = len(symbols_used)
        seed_history["missing_count"] = len(symbols_missing)
        frames.append(seed_history.tail(config.tail_weeks))

    if not frames:
        return pd.DataFrame(columns=RRG_BASKET_COLUMNS)
    return pd.concat(frames, ignore_index=True)[RRG_BASKET_COLUMNS]


def _relative_strength_label(score: float) -> str:
    if pd.isna(score):
        return "Neutral"
    if score >= 70:
        return "Outperforming"
    if score <= 30:
        return "Lagging"
    return "Neutral"


def _score_return_spread(spread: float) -> float:
    if pd.isna(spread):
        return 50.0
    return max(min(50.0 + spread * 100.0, 100.0), 0.0)


def relative_strength_metrics(
    prices: pd.DataFrame,
    symbol: str,
    benchmark: str,
) -> dict[str, object]:
    """Compare a symbol's trailing returns with the benchmark on aligned dates."""
    if symbol == benchmark:
        return {
            "rs_3m": 0.0,
            "rs_6m": 0.0,
            "rs_12m": 0.0,
            "relative_strength_score": 50.0,
            "relative_strength_label": "Benchmark",
        }

    symbol_daily = _daily_for_symbol(prices, symbol)
    benchmark_daily = _daily_for_symbol(prices, benchmark)
    if symbol_daily.empty or benchmark_daily.empty:
        return {
            "rs_3m": float("nan"),
            "rs_6m": float("nan"),
            "rs_12m": float("nan"),
            "relative_strength_score": 50.0,
            "relative_strength_label": "Neutral",
        }

    aligned = (
        symbol_daily[["date", "close"]]
        .rename(columns={"close": "symbol_close"})
        .merge(
            benchmark_daily[["date", "close"]].rename(columns={"close": "benchmark_close"}),
            on="date",
            how="inner",
        )
        .sort_values("date")
    )
    if aligned.empty:
        return {
            "rs_3m": float("nan"),
            "rs_6m": float("nan"),
            "rs_12m": float("nan"),
            "relative_strength_score": 50.0,
            "relative_strength_label": "Neutral",
        }

    symbol_close = aligned["symbol_close"]
    benchmark_close = aligned["benchmark_close"]
    spreads: dict[str, float] = {}
    scores = []
    for label, periods in RELATIVE_STRENGTH_WINDOWS.items():
        symbol_return = total_return_over(symbol_close, periods)
        benchmark_return = total_return_over(benchmark_close, periods)
        spread = (
            float("nan")
            if pd.isna(symbol_return) or pd.isna(benchmark_return)
            else float(symbol_return - benchmark_return)
        )
        spreads[f"rs_{label}"] = spread
        scores.append(_score_return_spread(spread))

    score = float(sum(scores) / len(scores)) if scores else 50.0
    return {
        **spreads,
        "relative_strength_score": round(score, 1),
        "relative_strength_label": _relative_strength_label(score),
    }


def _larsson_proxy_daily(daily: pd.DataFrame, config: SignalConfig) -> pd.DataFrame:
    close = daily.set_index("date")["close"]
    sma200d = rolling_sma(close, config.daily_filter_window)
    weekly_close = close.resample("W-FRI").last().dropna()
    sma200w = rolling_sma(weekly_close, config.weekly_filter_window)
    sma200w_daily = sma200w.reindex(close.index, method="ffill")
    proxy = pd.concat([sma200d, sma200w_daily], axis=1).max(axis=1, skipna=True)
    return pd.DataFrame(
        {
            "date": close.index,
            "close": close.values,
            "sma50d": rolling_sma(close, config.daily_trend_window).values,
            "sma200d": sma200d.values,
            "sma200w_ffill": sma200w_daily.values,
            "larsson_line_proxy": proxy.values,
        }
    )


def _larsson_regime_history_from_daily(daily: pd.DataFrame, config: SignalConfig) -> pd.DataFrame:
    indicator = _larsson_proxy_daily(daily, config)
    if indicator.empty:
        return pd.DataFrame(columns=["date", "larsson_regime", "larsson_regime_value"])

    history = indicator[["date", "close", "larsson_line_proxy"]].copy()
    history["proxy_slope"] = history["larsson_line_proxy"] - history["larsson_line_proxy"].shift(
        config.larsson_slope_days
    )
    weekly = (
        history.set_index("date")[["close", "larsson_line_proxy", "proxy_slope"]]
        .resample("W-FRI")
        .last()
        .dropna(subset=["close"])
    )
    if weekly.empty:
        return pd.DataFrame(columns=["date", "larsson_regime", "larsson_regime_value"])

    regimes = [
        _larsson_state(row.close, row.larsson_line_proxy, row.proxy_slope)
        for row in weekly.itertuples(index=False)
    ]
    weekly_history = pd.DataFrame(
        {
            "date": weekly.index,
            "larsson_regime": regimes,
            "larsson_regime_value": [LARSSON_HISTORY_VALUES[regime] for regime in regimes],
        }
    )
    return weekly_history.tail(config.larsson_history_weeks).reset_index(drop=True)


def larsson_regime_history(
    prices: pd.DataFrame,
    symbol: str,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """Return weekly Larsson-proxy regime history for a symbol."""
    config = config or SignalConfig()
    daily = _daily_for_symbol(prices, symbol)
    if daily.empty:
        return pd.DataFrame(columns=["date", "larsson_regime", "larsson_regime_value"])
    return _larsson_regime_history_from_daily(daily, config)


def price_history_with_indicators(
    prices: pd.DataFrame,
    symbol: str,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """Return daily close data with chartable moving-average indicators."""
    config = config or SignalConfig()
    daily = _daily_for_symbol(prices, symbol)
    if daily.empty:
        return pd.DataFrame(columns=["date", "close", "sma50d", "sma200d", "sma200w_ffill", "larsson_line_proxy"])
    return _larsson_proxy_daily(daily, config)


def weekly_history_with_indicators(
    prices: pd.DataFrame,
    symbol: str,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """Return weekly closes with 30W and 200W moving averages."""
    config = config or SignalConfig()
    daily = _daily_for_symbol(prices, symbol)
    if daily.empty:
        return pd.DataFrame(columns=["date", "weekly_close", "sma30w", "sma200w"])
    weekly_close = resample_weekly_close(daily)
    return pd.DataFrame(
        {
            "date": weekly_close.index,
            "weekly_close": weekly_close.values,
            "sma30w": rolling_sma(weekly_close, config.weekly_trend_window).values,
            "sma200w": rolling_sma(weekly_close, config.weekly_filter_window).values,
        }
    )


def _warning_text(warnings: list[str]) -> str:
    return "; ".join(dict.fromkeys(warnings))


def _compute_symbol_signals(symbol: str, prices: pd.DataFrame, benchmark: str, config: SignalConfig) -> dict[str, object]:
    daily = _daily_for_symbol(prices, symbol)
    warnings: list[str] = []
    if daily.empty:
        return {
            "symbol": symbol,
            "is_benchmark": symbol == benchmark,
            "latest_date": pd.NaT,
            "latest_close": float("nan"),
            "larsson_regime": "Transition",
            "larsson_regime_history": [],
            "larsson_regime_history_labels": "",
            "daily_trend": "Mixed",
            "weekly_trend": "Mixed",
            "monthly_trend": "Mixed",
            "pass_200d": None,
            "pass_200w": None,
            "rs_3m": float("nan"),
            "rs_6m": float("nan"),
            "rs_12m": float("nan"),
            "relative_strength_score": 50.0,
            "relative_strength_label": "Neutral",
            "data_warnings": "no usable price rows",
        }

    close = daily.set_index("date")["close"]
    weekly_close = close.resample("W-FRI").last().dropna()
    monthly_close = close.resample("ME").last().dropna()

    sma50d = rolling_sma(close, config.daily_trend_window)
    sma200d = rolling_sma(close, config.daily_filter_window)
    sma30w = rolling_sma(weekly_close, config.weekly_trend_window)
    sma200w = rolling_sma(weekly_close, config.weekly_filter_window)
    sma10m = rolling_sma(monthly_close, config.monthly_trend_window)

    sma200w_daily = sma200w.reindex(close.index, method="ffill")
    larsson_proxy = pd.concat([sma200d, sma200w_daily], axis=1).max(axis=1, skipna=True)
    latest_close = _latest(close)
    latest_sma50d = _latest(sma50d)
    latest_sma200d = _latest(sma200d)
    latest_sma30w = _latest(sma30w)
    latest_sma200w = _latest(sma200w)
    latest_sma10m = _latest(sma10m)
    latest_proxy = _latest(larsson_proxy)

    if len(close.dropna()) < config.daily_filter_window:
        warnings.append("insufficient 200D history")
    if len(weekly_close.dropna()) < config.weekly_filter_window:
        warnings.append("insufficient 200W history")
    if len(weekly_close.dropna()) < config.weekly_trend_window:
        warnings.append("insufficient 30W history")
    if len(monthly_close.dropna()) < config.monthly_trend_window:
        warnings.append("insufficient 10M history")

    daily_trend = _state_from_price_average_slope(latest_close, latest_sma50d, slope_over(sma50d, config.daily_slope_days))
    weekly_trend = _state_from_price_average_slope(
        _latest(weekly_close),
        latest_sma30w,
        slope_over(sma30w, config.weekly_slope_weeks),
    )
    monthly_trend = _state_from_price_average_slope(
        _latest(monthly_close),
        latest_sma10m,
        slope_over(sma10m, config.monthly_slope_months),
    )
    larsson_regime = _larsson_state(latest_close, latest_proxy, slope_over(larsson_proxy, config.larsson_slope_days))
    larsson_history = _larsson_regime_history_from_daily(daily, config)
    rs_metrics = relative_strength_metrics(prices, symbol, benchmark)

    pass_200d = _bool_or_none(latest_close, latest_sma200d)
    pass_200w = None if len(weekly_close.dropna()) < config.weekly_filter_window else _bool_or_none(_latest(weekly_close), latest_sma200w)

    return {
        "symbol": symbol,
        "is_benchmark": symbol == benchmark,
        "latest_date": close.index.max(),
        "latest_close": latest_close,
        "larsson_line_proxy": latest_proxy,
        "larsson_regime": larsson_regime,
        "larsson_regime_history": larsson_history["larsson_regime_value"].tolist(),
        "larsson_regime_history_labels": ", ".join(larsson_history["larsson_regime"].tolist()),
        "daily_trend": daily_trend,
        "weekly_trend": weekly_trend,
        "monthly_trend": monthly_trend,
        "sma50d": latest_sma50d,
        "sma200d": latest_sma200d,
        "sma30w": latest_sma30w,
        "sma200w": latest_sma200w,
        "sma10m": latest_sma10m,
        "pass_200d": pass_200d,
        "pass_200w": pass_200w,
        **rs_metrics,
        "drawdown_52w": latest_drawdown(close, config.drawdown_lookback_days),
        "drawdown_history": latest_drawdown(close),
        "extension_sma50d": percent_extension(latest_close, latest_sma50d),
        "extension_sma200d": percent_extension(latest_close, latest_sma200d),
        "extension_sma30w": percent_extension(latest_close, latest_sma30w),
        "data_warnings": _warning_text(warnings),
    }


def compute_signal_table(
    prices: pd.DataFrame,
    benchmark: str,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """Compute one latest technical signal row per symbol."""
    config = config or SignalConfig()
    if prices.empty:
        return pd.DataFrame()

    prepared = prices.copy()
    prepared["symbol"] = prepared["symbol"].astype(str).str.upper()
    prepared["date"] = pd.to_datetime(prepared["date"])
    benchmark = benchmark.upper()
    symbols = sorted(prepared["symbol"].dropna().unique())
    rows = [_compute_symbol_signals(symbol, prepared, benchmark, config) for symbol in symbols]
    signal_table = pd.DataFrame(rows)

    benchmark_row = signal_table[signal_table["symbol"] == benchmark]
    benchmark_regime = benchmark_row["larsson_regime"].iloc[0] if not benchmark_row.empty else "n/a"
    signal_table["benchmark"] = benchmark
    signal_table["benchmark_larsson_regime"] = benchmark_regime
    return signal_table
