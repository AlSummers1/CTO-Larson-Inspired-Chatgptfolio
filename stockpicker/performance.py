from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

import pandas as pd


INITIAL_PAPER_NAV = 100_000.0
CHECKPOINT_COLUMNS = [
    "saved_brief_id",
    "title",
    "saved_at",
    "evidence_date",
    "benchmark",
    "status",
    "effective_date",
    "holding_count",
    "cash_weight",
    "diagnostic",
    "weights",
]


@dataclass(frozen=True)
class SavedBriefPortfolioHistory:
    nav: pd.DataFrame
    benchmarks: pd.DataFrame
    checkpoints: pd.DataFrame
    latest_holdings: pd.DataFrame
    required_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    latest_valuation_date: pd.Timestamp | None


def _empty_history(checkpoints: pd.DataFrame | None = None, required_symbols: tuple[str, ...] = ()) -> SavedBriefPortfolioHistory:
    return SavedBriefPortfolioHistory(
        nav=pd.DataFrame(columns=["date", "nav", "cash", "drawdown"]),
        benchmarks=pd.DataFrame(columns=["date", "series", "symbol", "nav"]),
        checkpoints=(checkpoints.copy() if checkpoints is not None else pd.DataFrame(columns=CHECKPOINT_COLUMNS)),
        latest_holdings=pd.DataFrame(columns=["symbol", "shares", "value", "weight"]),
        required_symbols=required_symbols,
        missing_symbols=required_symbols,
        latest_valuation_date=None,
    )


def _payload(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _timestamp(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _allocation_weights(evidence: object) -> tuple[dict[str, float], str]:
    if not isinstance(evidence, list):
        return {}, "No captured paper allocation."
    weights: dict[str, float] = {}
    invalid_weight = False
    for item in evidence:
        if not isinstance(item, dict) or item.get("kind") != "paper_allocation":
            continue
        symbol = str(item.get("symbol") or "").strip()
        symbol = "Cash" if symbol.lower() == "cash" else symbol.upper()
        try:
            weight = float(item.get("weight", 0.0))
        except (TypeError, ValueError):
            invalid_weight = True
            continue
        if not symbol or not math.isfinite(weight) or weight < 0:
            invalid_weight = True
            continue
        if weight > 0:
            weights[symbol] = weights.get(symbol, 0.0) + weight
    if invalid_weight:
        return {}, "The captured allocation contains an invalid weight."
    if not weights:
        return {}, "No captured paper allocation."
    total = sum(weights.values())
    if not 0.99 <= total <= 1.01:
        return {}, f"Captured allocation totals {total:.2%}; expected approximately 100%."
    return {symbol: weight / total for symbol, weight in weights.items()}, ""


def build_saved_brief_checkpoints(saved_briefs: list[dict[str, object]]) -> pd.DataFrame:
    """Convert immutable saved reports into auditable portfolio checkpoints."""
    rows: list[dict[str, object]] = []
    for brief in saved_briefs:
        load_error = str(brief.get("tracking_error") or "").strip()
        report_payload = _payload(brief.get("evidence_json"))
        evidence_date = _timestamp(report_payload.get("effective_market_date"))
        benchmark = str(report_payload.get("benchmark") or brief.get("benchmark") or "").strip().upper()
        weights, allocation_diagnostic = _allocation_weights(report_payload.get("evidence"))
        diagnostic = allocation_diagnostic
        status = "Ready"
        if load_error:
            status = "Invalid"
            diagnostic = load_error
        elif evidence_date is None:
            status = "Invalid"
            diagnostic = "The saved brief has no valid evidence date."
        elif not benchmark:
            status = "Invalid"
            diagnostic = "The saved brief has no benchmark."
        elif allocation_diagnostic:
            status = "Skipped" if allocation_diagnostic == "No captured paper allocation." else "Invalid"
        saved_at = pd.to_datetime(brief.get("saved_at"), errors="coerce")
        rows.append(
            {
                "saved_brief_id": int(brief.get("saved_brief_id") or 0),
                "title": str(brief.get("saved_title") or brief.get("title") or "Saved brief"),
                "saved_at": saved_at,
                "evidence_date": evidence_date,
                "benchmark": benchmark,
                "status": status,
                "effective_date": pd.NaT,
                "holding_count": sum(symbol != "Cash" for symbol in weights),
                "cash_weight": float(weights.get("Cash", 0.0)),
                "diagnostic": diagnostic,
                "weights": weights,
            }
        )
    if not rows:
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    checkpoints = pd.DataFrame(rows, columns=CHECKPOINT_COLUMNS)
    checkpoints = checkpoints.sort_values(
        ["evidence_date", "saved_at", "saved_brief_id"],
        na_position="last",
    ).reset_index(drop=True)
    valid_dates = checkpoints.dropna(subset=["evidence_date"])
    for _, same_date in valid_dates.groupby("evidence_date", sort=False):
        if len(same_date) <= 1:
            continue
        winner_index = same_date.sort_values(["saved_at", "saved_brief_id"], na_position="first").index[-1]
        for index in same_date.index:
            if index == winner_index:
                continue
            checkpoints.at[index, "status"] = "Superseded"
            checkpoints.at[index, "diagnostic"] = "A more recently saved brief has the same evidence date."
    return checkpoints


def saved_brief_tracking_symbols(checkpoints: pd.DataFrame) -> tuple[str, ...]:
    symbols = {"SPY"}
    if checkpoints.empty:
        return tuple()
    active = checkpoints[checkpoints["status"] == "Ready"]
    for row in active.itertuples(index=False):
        symbols.update(symbol for symbol in row.weights if symbol != "Cash")
        if row.benchmark:
            symbols.add(str(row.benchmark).upper())
    return tuple(sorted(symbols)) if not active.empty else tuple()


def _prepared_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "open", "close"}
    if prices.empty or not required.issubset(prices.columns):
        return pd.DataFrame(columns=["symbol", "date", "open", "close"])
    frame = prices[["symbol", "date", "open", "close"]].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "date", "close"])
    frame = frame[frame["close"] > 0]
    return frame.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")


def _first_common_open(
    opens: pd.DataFrame,
    symbols: set[str],
    after: pd.Timestamp,
    not_before: pd.Timestamp | None,
) -> pd.Timestamp | None:
    if not symbols:
        return None
    candidate_dates = opens.index[opens.index > after]
    if not_before is not None:
        candidate_dates = candidate_dates[candidate_dates >= not_before]
    for candidate in candidate_dates:
        values = opens.loc[candidate].reindex(sorted(symbols))
        if values.notna().all() and (values > 0).all():
            return pd.Timestamp(candidate)
    return None


def build_saved_brief_portfolio_history(
    checkpoints: pd.DataFrame,
    prices: pd.DataFrame,
    initial_nav: float = INITIAL_PAPER_NAV,
) -> SavedBriefPortfolioHistory:
    """Simulate one evolving paper portfolio from saved-brief checkpoints."""
    planned = checkpoints.copy()
    required_symbols = saved_brief_tracking_symbols(planned)
    if planned.empty or not required_symbols:
        return _empty_history(planned, required_symbols)
    frame = _prepared_prices(prices)
    available = set(frame["symbol"].unique()) if not frame.empty else set()
    missing_symbols = tuple(sorted(set(required_symbols) - available))
    if frame.empty:
        return _empty_history(planned, required_symbols)

    opens = frame.pivot(index="date", columns="symbol", values="open").sort_index()
    closes = frame.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    current_symbols: set[str] = set()
    current_benchmark = ""
    last_execution: pd.Timestamp | None = None
    active_indices = planned.index[planned["status"] == "Ready"].tolist()
    for index in active_indices:
        row = planned.loc[index]
        target_symbols = {symbol for symbol in row["weights"] if symbol != "Cash"}
        execution_symbols = current_symbols | target_symbols | {str(row["benchmark"]).upper()}
        if last_execution is None:
            execution_symbols.add("SPY")
        elif current_benchmark:
            execution_symbols.add(current_benchmark)
        execution_date = _first_common_open(
            opens,
            execution_symbols,
            pd.Timestamp(row["evidence_date"]),
            last_execution,
        )
        if execution_date is None:
            absent = sorted(symbol for symbol in execution_symbols if symbol not in opens.columns)
            planned.at[index, "status"] = "Pending"
            planned.at[index, "diagnostic"] = (
                "No eligible cached market open is available yet."
                + (f" Missing price history: {', '.join(absent)}." if absent else "")
            )
            continue
        planned.at[index, "status"] = "Applied"
        planned.at[index, "effective_date"] = execution_date
        planned.at[index, "diagnostic"] = "Rebalanced at the next eligible cached open after the evidence date."
        current_symbols = target_symbols
        current_benchmark = str(row["benchmark"]).upper()
        last_execution = execution_date

    applied = planned[planned["status"] == "Applied"].sort_values(
        ["effective_date", "evidence_date", "saved_at", "saved_brief_id"]
    )
    if applied.empty:
        return SavedBriefPortfolioHistory(
            nav=pd.DataFrame(columns=["date", "nav", "cash", "drawdown"]),
            benchmarks=pd.DataFrame(columns=["date", "series", "symbol", "nav"]),
            checkpoints=planned,
            latest_holdings=pd.DataFrame(columns=["symbol", "shares", "value", "weight"]),
            required_symbols=required_symbols,
            missing_symbols=missing_symbols,
            latest_valuation_date=None,
        )

    first_execution = pd.Timestamp(applied.iloc[0]["effective_date"])
    valuation_dates = closes.index[closes.index >= first_execution]
    portfolio_shares: dict[str, float] = {}
    portfolio_cash = float(initial_nav)
    dynamic_symbol = ""
    dynamic_shares = 0.0
    spy_shares = 0.0
    all_brief_benchmarks = set(applied["benchmark"].astype(str).str.upper())
    dedupe_spy = all_brief_benchmarks == {"SPY"}
    nav_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []

    checkpoints_by_date = {
        pd.Timestamp(date): group.sort_values(["evidence_date", "saved_at", "saved_brief_id"])
        for date, group in applied.groupby("effective_date", sort=True)
    }
    for current_date in valuation_dates:
        current_date = pd.Timestamp(current_date)
        for _, checkpoint in checkpoints_by_date.get(current_date, pd.DataFrame()).iterrows():
            open_row = opens.loc[current_date]
            if portfolio_shares:
                open_nav = portfolio_cash + sum(
                    shares * float(open_row[symbol]) for symbol, shares in portfolio_shares.items()
                )
            else:
                open_nav = portfolio_cash
            weights = checkpoint["weights"]
            portfolio_cash = open_nav * float(weights.get("Cash", 0.0))
            portfolio_shares = {
                symbol: open_nav * float(weight) / float(open_row[symbol])
                for symbol, weight in weights.items()
                if symbol != "Cash" and float(weight) > 0
            }

            new_benchmark = str(checkpoint["benchmark"]).upper()
            dynamic_value = float(initial_nav) if not dynamic_symbol else dynamic_shares * float(open_row[dynamic_symbol])
            dynamic_symbol = new_benchmark
            dynamic_shares = dynamic_value / float(open_row[dynamic_symbol])
            if spy_shares == 0.0:
                spy_shares = float(initial_nav) / float(open_row["SPY"])

        close_row = closes.loc[current_date]
        portfolio_value = portfolio_cash + sum(
            shares * float(close_row[symbol]) for symbol, shares in portfolio_shares.items()
        )
        nav_rows.append({"date": current_date, "nav": portfolio_value, "cash": portfolio_cash})
        if dynamic_symbol:
            benchmark_rows.append(
                {
                    "date": current_date,
                    "series": "Brief benchmark" if not dedupe_spy else "Brief benchmark · SPY",
                    "symbol": dynamic_symbol,
                    "nav": dynamic_shares * float(close_row[dynamic_symbol]),
                }
            )
        if spy_shares and not dedupe_spy:
            benchmark_rows.append(
                {"date": current_date, "series": "SPY", "symbol": "SPY", "nav": spy_shares * float(close_row["SPY"])}
            )

    nav = pd.DataFrame(nav_rows)
    nav["drawdown"] = nav["nav"] / nav["nav"].cummax() - 1.0
    benchmarks = pd.DataFrame(benchmark_rows, columns=["date", "series", "symbol", "nav"])
    last_close = closes.loc[valuation_dates[-1]]
    holding_rows = [
        {"symbol": symbol, "shares": shares, "value": shares * float(last_close[symbol])}
        for symbol, shares in sorted(portfolio_shares.items())
    ]
    if portfolio_cash > 0:
        holding_rows.append({"symbol": "Cash", "shares": 0.0, "value": portfolio_cash})
    latest_holdings = pd.DataFrame(holding_rows, columns=["symbol", "shares", "value"])
    if not latest_holdings.empty:
        latest_holdings["weight"] = latest_holdings["value"] / float(latest_holdings["value"].sum())
    else:
        latest_holdings["weight"] = pd.Series(dtype=float)
    return SavedBriefPortfolioHistory(
        nav=nav[["date", "nav", "cash", "drawdown"]],
        benchmarks=benchmarks,
        checkpoints=planned,
        latest_holdings=latest_holdings[["symbol", "shares", "value", "weight"]],
        required_symbols=required_symbols,
        missing_symbols=missing_symbols,
        latest_valuation_date=pd.Timestamp(nav["date"].max()) if not nav.empty else None,
    )
