from __future__ import annotations

from typing import Iterable

import pandas as pd


DATA_HEALTH_COLUMNS = [
    "symbol",
    "latest_price_date",
    "reference_market_date",
    "trading_sessions_behind",
    "coverage_pct_60",
    "missing_sessions_60",
    "missing_session_dates",
    "latest_successful_fetch",
    "fetch_status",
    "unchanged_close_streak",
    "freshness_status",
    "calendar_approximate",
    "calendar_source",
    "derived_series_note",
]

DERIVED_200W_NOTE = (
    "The weekly 200W component is a derived weekly series carried between weekly "
    "observations; raw prices are never forward-filled."
)


def _empty_health() -> pd.DataFrame:
    return pd.DataFrame(columns=DATA_HEALTH_COLUMNS)


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "close"}
    if prices.empty or not required.issubset(prices.columns):
        return pd.DataFrame(columns=["symbol", "date", "close"])
    frame = prices[["symbol", "date", "close"]].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return (
        frame.dropna(subset=["symbol", "date", "close"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def _reference_calendar(frame: pd.DataFrame, benchmark: str) -> tuple[pd.DatetimeIndex, bool, str]:
    benchmark_dates = frame.loc[frame["symbol"] == benchmark, "date"].drop_duplicates().sort_values()
    if not benchmark_dates.empty:
        return pd.DatetimeIndex(benchmark_dates), False, benchmark

    if frame.empty:
        return pd.DatetimeIndex([]), True, "broad-market fallback"

    # Prefer the latest date with broad cross-symbol observation. Looking only at
    # the recent window avoids selecting an old date merely because the universe
    # grew over time.
    recent_dates = frame["date"].drop_duplicates().sort_values().tail(10)
    recent = frame[frame["date"].isin(recent_dates)]
    counts = recent.groupby("date")["symbol"].nunique()
    broadest_count = int(counts.max())
    broad_dates = counts[counts == broadest_count].index
    reference_date = pd.Timestamp(max(broad_dates)).normalize()
    broadly_observed = (
        frame.loc[frame["date"] <= reference_date, "date"].drop_duplicates().sort_values()
    )
    return pd.DatetimeIndex(broadly_observed), True, "broad-market fallback"


def _latest_fetch_by_symbol(fetch_log: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if fetch_log is None or fetch_log.empty or not {"symbol", "status"}.issubset(fetch_log.columns):
        return {}
    log = fetch_log.copy()
    log["symbol"] = log["symbol"].astype(str).str.strip().str.upper()
    log["fetched_at"] = pd.to_datetime(log["fetched_at"], errors="coerce") if "fetched_at" in log.columns else pd.NaT
    log = log.sort_values("fetched_at", ascending=False)
    latest = log.drop_duplicates("symbol", keep="first").set_index("symbol")
    successful = log[log["status"].astype(str).str.lower().isin({"ok", "success"})]
    latest_success = (
        successful.drop_duplicates("symbol", keep="first").set_index("symbol")
        if not successful.empty
        else pd.DataFrame()
    )
    result: dict[str, dict[str, object]] = {}
    for symbol, row in latest.iterrows():
        successful_at = None
        if not latest_success.empty and symbol in latest_success.index:
            value = latest_success.loc[symbol, "fetched_at"]
            successful_at = None if pd.isna(value) else pd.Timestamp(value)
        result[str(symbol)] = {
            "fetch_status": str(row.get("status", "unknown") or "unknown"),
            "latest_successful_fetch": successful_at,
        }
    return result


def _unchanged_close_streak(close: Iterable[float]) -> int:
    values = list(close)
    if len(values) < 2:
        return 0
    streak = 0
    latest = values[-1]
    for prior in reversed(values[:-1]):
        if pd.isna(prior) or pd.isna(latest) or float(prior) != float(latest):
            break
        streak += 1
        latest = prior
    return streak


def _freshness_status(sessions_behind: int | None) -> str:
    if sessions_behind is None:
        return "Unknown"
    if sessions_behind == 0:
        return "Fresh"
    if sessions_behind == 1:
        return "Delayed"
    return "Stale"


def compute_data_health(
    prices: pd.DataFrame,
    benchmark: str,
    lookback_sessions: int = 60,
    fetch_log: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate inspectable per-symbol freshness and market-session coverage.

    The selected benchmark's observed dates are the primary market calendar. If
    that series is unavailable, the latest broadly observed date and the observed
    cross-symbol dates are used as an explicitly approximate fallback.
    """
    if lookback_sessions <= 0:
        raise ValueError("lookback_sessions must be positive")

    frame = _normalize_prices(prices)
    if frame.empty:
        return _empty_health()

    benchmark = str(benchmark).strip().upper()
    calendar, approximate, calendar_source = _reference_calendar(frame, benchmark)
    reference_date = pd.Timestamp(calendar.max()).normalize() if len(calendar) else pd.NaT
    coverage_dates = calendar[-lookback_sessions:]
    fetch_by_symbol = _latest_fetch_by_symbol(fetch_log)

    rows: list[dict[str, object]] = []
    for symbol in sorted(frame["symbol"].unique()):
        symbol_frame = frame[frame["symbol"] == symbol].sort_values("date")
        symbol_dates = pd.DatetimeIndex(symbol_frame["date"].drop_duplicates())
        latest_date = pd.Timestamp(symbol_dates.max()).normalize() if len(symbol_dates) else pd.NaT
        sessions_behind: int | None
        if pd.isna(latest_date) or pd.isna(reference_date):
            sessions_behind = None
        else:
            sessions_behind = int((calendar > latest_date).sum())

        observed = set(symbol_dates)
        missing_dates = [pd.Timestamp(date) for date in coverage_dates if pd.Timestamp(date) not in observed]
        expected = len(coverage_dates)
        coverage_pct = 0.0 if expected == 0 else round(100.0 * (expected - len(missing_dates)) / expected, 1)
        fetch = fetch_by_symbol.get(symbol, {})
        rows.append(
            {
                "symbol": symbol,
                "latest_price_date": latest_date,
                "reference_market_date": reference_date,
                "trading_sessions_behind": sessions_behind,
                "coverage_pct_60": coverage_pct,
                "missing_sessions_60": len(missing_dates),
                "missing_session_dates": ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates),
                "latest_successful_fetch": fetch.get("latest_successful_fetch"),
                "fetch_status": fetch.get("fetch_status", "unknown"),
                "unchanged_close_streak": _unchanged_close_streak(symbol_frame["close"]),
                "freshness_status": _freshness_status(sessions_behind),
                "calendar_approximate": approximate,
                "calendar_source": calendar_source,
                "derived_series_note": DERIVED_200W_NOTE,
            }
        )

    return pd.DataFrame(rows, columns=DATA_HEALTH_COLUMNS)
