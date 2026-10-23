"""Auditable OpenAI-backed paper-portfolio workflow.

The model proposes allocations; deterministic validation is always authoritative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
import getpass
import hashlib
import json
import os
import platform
import re
import subprocess
from time import perf_counter
from typing import Any

import pandas as pd

from .research import alpha_vantage_api_key, fetch_alpha_vantage_news, fetch_macro_snapshot, fetch_gdelt_news, fetch_sec_company_research, fundamental_score

MODEL_OPTIONS = ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini")
DEFAULT_MODEL = "gpt-5.5"
PROMPT_VERSION = "chatgptfolio-v5-theme-breadth-compact"
SEVERITY_ORDER = ("critical", "high", "medium", "low", "unresolved")
SEVERITY_COLORS = {"critical": "#dc2626", "high": "#f97316", "medium": "#eab308", "low": "#22c55e", "unresolved": "#94a3b8"}
MODEL_EVIDENCE_SUMMARY_CHARS = 500
MODEL_EVIDENCE_LIMITS = {"fundamental": 1, "corporate_event": 3, "news": 3}


@dataclass(frozen=True)
class LabConfig:
    model: str = DEFAULT_MODEL
    max_holdings: int = 15
    max_position: float = 0.12
    max_sector_exposure: float = 0.40
    max_turnover: float = 0.35
    paper_capital: float = 100_000.0
    slippage_bps: float = 5.0
    commission_bps: float = 0.0
    min_dollar_volume: float = 1_000_000.0
    research_limit: int = 30


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _price_series(prices: pd.DataFrame, symbol: str) -> pd.Series:
    if prices.empty or "symbol" not in prices:
        return pd.Series(dtype=float)
    rows = prices[prices["symbol"].astype(str).str.upper() == symbol.upper()][["date", "close"]].copy()
    if rows.empty:
        return pd.Series(dtype=float)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["close"] = pd.to_numeric(rows["close"], errors="coerce")
    return rows.dropna().drop_duplicates("date", keep="last").sort_values("date").set_index("date")["close"]


def entry_signal(prices: pd.DataFrame, symbol: str, regime: str = "", daily_trend: str = "", weekly_trend: str = "", monthly_trend: str = "") -> dict[str, object]:
    """Return reproducible, non-predictive entry guidance from local adjusted prices."""
    close = _price_series(prices, symbol)
    unresolved = {"entry_label": "Unresolved", "entry_score": None, "preferred_entry_low": None, "preferred_entry_high": None, "trend_invalidation": None, "latest_price": None, "extension_50d": None, "atr_14d": None, "reason": "insufficient price history"}
    if len(close) < 50:
        return unresolved
    latest = float(close.iloc[-1])
    sma20, sma50 = float(close.tail(20).mean()), float(close.tail(50).mean())
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None
    high20 = float(close.tail(20).max())
    atr_proxy = float(close.pct_change().abs().tail(14).mean() * latest)
    extension = latest / sma50 - 1.0 if sma50 else None
    constructive = regime == "Bull" and daily_trend == weekly_trend == monthly_trend == "Up" and (sma200 is None or latest > sma200)
    low, high = min(sma20, sma50) * 0.98, max(sma20, sma50) * 1.02
    invalidation = sma50 - 1.5 * atr_proxy
    if not constructive:
        label, score, reason = "Avoid", 15, "trend/regime is not fully constructive"
    elif extension is not None and extension > 0.08:
        label, score, reason = "Wait — don't chase", 35, "price is more than 8% above its 50-day average"
    elif low <= latest <= high:
        label, score, reason = "Add", 85, "price is in the calculated moving-average entry zone"
    elif latest >= high20 * 0.98:
        label, score, reason = "Wait — confirm", 65, "near a 20-day high; await a confirmed breakout or pullback"
    else:
        label, score, reason = "Wait — confirm", 55, "constructive trend, but price is outside the preferred entry zone"
    return {"entry_label": label, "entry_score": score, "preferred_entry_low": round(low, 2), "preferred_entry_high": round(high, 2), "trend_invalidation": round(invalidation, 2), "latest_price": round(latest, 2), "extension_50d": round(extension or 0.0, 4), "atr_14d": round(atr_proxy, 2), "reason": reason}


def correlation_summary(prices: pd.DataFrame, symbols: list[str], held_symbols: list[str] | None = None) -> dict[str, object]:
    """Compute bounded daily/weekly return correlations for an immutable packet."""
    symbols = list(dict.fromkeys(str(symbol).upper() for symbol in symbols if symbol))
    held = {str(symbol).upper() for symbol in (held_symbols or [])}
    matrix_symbols = list(dict.fromkeys(symbols + sorted(held)))
    closes = pd.DataFrame({symbol: _price_series(prices, symbol) for symbol in matrix_symbols}).sort_index()
    daily = closes.pct_change().tail(50)
    weekly = closes.resample("W-FRI").last().pct_change().tail(26) if not closes.empty else pd.DataFrame()
    daily_corr = daily.corr(min_periods=40)
    weekly_corr = weekly.corr(min_periods=20)
    pairs: list[dict[str, object]] = []
    clusters: dict[str, set[str]] = {symbol: {symbol} for symbol in matrix_symbols}
    for index, left in enumerate(symbols):
        for right in symbols[index + 1:]:
            value = daily_corr.loc[left, right] if left in daily_corr.index and right in daily_corr.columns else float("nan")
            weekly_value = weekly_corr.loc[left, right] if left in weekly_corr.index and right in weekly_corr.columns else float("nan")
            if pd.notna(value) and float(value) >= 0.80:
                pairs.append({"left": left, "right": right, "daily_correlation": round(float(value), 3), "weekly_correlation": None if pd.isna(weekly_value) else round(float(weekly_value), 3), "severity": "high"})
                merged = clusters[left] | clusters[right]
                for member in merged:
                    clusters[member] = merged
    candidates = []
    for symbol in symbols:
        others = [other for other in symbols if other != symbol]
        values = daily_corr.loc[symbol, others].dropna() if symbol in daily_corr.index and others else pd.Series(dtype=float)
        to_held = daily_corr.loc[symbol, [other for other in held if other != symbol and other in daily_corr.columns]].dropna() if symbol in daily_corr.index and held else pd.Series(dtype=float)
        candidates.append({"symbol": symbol, "daily_observations": int(daily[symbol].dropna().shape[0]) if symbol in daily else 0, "weekly_observations": int(weekly[symbol].dropna().shape[0]) if symbol in weekly else 0, "average_daily_correlation": None if values.empty else round(float(values.mean()), 3), "max_daily_correlation": None if values.empty else round(float(values.max()), 3), "average_correlation_to_held": None if to_held.empty else round(float(to_held.mean()), 3), "cluster": " · ".join(sorted(clusters[symbol])) if len(clusters[symbol]) > 1 else "independent", "status": "available" if len(values) else "unresolved"})
    matrix = daily_corr.round(3).where(pd.notna(daily_corr), None).to_dict() if not daily_corr.empty else {}
    return {"daily_window": 50, "daily_min_observations": 40, "weekly_window": 26, "weekly_min_observations": 20, "candidate_summary": candidates, "high_correlation_pairs": pairs, "daily_matrix": matrix, "status": "available" if not daily_corr.empty else "unresolved"}


def credential_status(last_error: str | None = None) -> str:
    """Return a safe UI status without probing or exposing an API key."""
    if last_error and ("401" in last_error or "invalid_api_key" in last_error):
        return "rejected"
    return "ready" if openai_api_key() else "missing"


def packet_context_metadata(
    packet: dict[str, object],
    config: LabConfig,
    api_usage: dict[str, int] | None = None,
) -> dict[str, object]:
    """A reproducible context estimate, not a billing estimate or secret trace."""
    full_serialized = json.dumps(packet, default=str, separators=(",", ":"))
    model_packet = compact_model_packet(packet)
    model_serialized = json.dumps(model_packet, default=str, separators=(",", ":"))
    prompt = prompt_for_packet(packet, config)
    reduction = 0.0 if not full_serialized else 1.0 - len(model_serialized) / len(full_serialized)
    metadata: dict[str, object] = {
        "packet_characters": len(full_serialized),
        "full_packet_characters": len(full_serialized),
        "model_packet_characters": len(model_serialized),
        "prompt_characters": len(prompt),
        "estimated_input_tokens": round(len(prompt) / 4),
        "estimated_packet_reduction_pct": round(100.0 * reduction, 1),
        "candidate_count": len(packet.get("candidates", [])),
        "evidence_count": len(packet.get("evidence", [])),
        "model_evidence_count": len(model_packet.get("evidence", [])),
        "theme_breadth_count": len(model_packet.get("theme_breadth", [])),
        "model": config.model,
        "prompt_version": PROMPT_VERSION,
    }
    if api_usage:
        metadata["api_usage"] = api_usage
    return metadata


def preflight_packet(packet: dict[str, object], credential: str) -> ValidationResult:
    """Stop before an API call if basic decision inputs cannot be supported."""
    errors: list[str] = []
    coverage = packet.get("coverage", {}) if isinstance(packet, dict) else {}
    if credential != "ready":
        errors.append("OpenAI credential is not ready")
    if coverage.get("prices") != "available" or not packet.get("candidates"):
        errors.append("required price coverage is unavailable")
    research_available = sum(coverage.get(key) == "available" for key in ("fundamentals", "events", "macro", "news"))
    if research_available == 0:
        errors.append("minimum external evidence coverage is unavailable; refresh research before running")
    return ValidationResult(not errors, errors)


def recent_news_payload(
    evidence: list[dict[str, object]],
    window_start: datetime | None,
    window_end: datetime | None,
) -> dict[str, object]:
    """Keep the recent, ticker-scoped news context explicit for the model audit."""
    by_symbol: dict[str, list[dict[str, object]]] = {}
    for item in evidence:
        if not isinstance(item, dict) or item.get("category") != "news" or item.get("status") != "available":
            continue
        symbol = str(item.get("symbol", "")).upper()
        if not symbol:
            continue
        by_symbol.setdefault(symbol, []).append(
            {
                key: item.get(key)
                for key in [
                    "evidence_id", "title", "publisher", "published_at", "source_url",
                    "news_channel", "relevance_score", "sentiment_score", "sentiment_label",
                ]
            }
        )
    return {
        "window_start": window_start.isoformat() if window_start else None,
        "window_end": window_end.isoformat() if window_end else None,
        "by_symbol": by_symbol,
        "policy": "Alpha Vantage is primary. GDELT is used only for symbols without fresh Alpha Vantage coverage.",
    }


def _liquidity_score(row: pd.Series) -> float:
    volume = pd.to_numeric(row.get("avg_dollar_volume", row.get("volume", 0)), errors="coerce")
    if pd.isna(volume) or volume <= 0:
        return 50.0
    return float(max(0, min(100, 50 + 10 * (len(str(int(volume))) - 6))))


def theme_breadth_reference(seed_universe: object) -> str:
    """Return a stable packet reference for one fixed seed universe."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(seed_universe).strip().lower()).strip("-")
    return f"packet:theme_breadth:{slug or 'unresolved'}"


def _optional_number(value: object, digits: int = 4) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(number) else round(float(number), digits)


def _seed_membership_list(value: object) -> list[str]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = value.split(" · ")
    else:
        raw = []
    return list(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))


def normalize_theme_breadth(rows: list[dict[str, object]] | None) -> list[dict[str, object]]:
    """Freeze latest seed breadth snapshots into a compact, JSON-safe shape."""
    normalized: list[dict[str, object]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        seed_universe = str(row.get("seed_universe", "")).strip()
        if not seed_universe:
            continue
        coverage = _optional_number(row.get("coverage"))
        confirmation = str(row.get("confirmation", "Unresolved") or "Unresolved")
        if coverage is None or coverage < 0.75:
            confirmation = "Unresolved"
        normalized.append(
            {
                "seed_universe": seed_universe,
                "reference": theme_breadth_reference(seed_universe),
                "observation_date": str(row.get("observation_date") or row.get("date") or "") or None,
                "breadth": _optional_number(row.get("breadth")),
                "breadth_4w_average": _optional_number(row.get("breadth_4w_average")),
                "breadth_change": _optional_number(row.get("breadth_change")),
                "roc_50d": _optional_number(row.get("roc_50d")),
                "confirmation": confirmation,
                "coverage": coverage,
                "eligible_count": int(_optional_number(row.get("eligible_count"), 0) or 0),
                "universe_size": int(_optional_number(row.get("universe_size"), 0) or 0),
            }
        )
    return normalized


def research_packet(
    candidates: pd.DataFrame,
    as_of_date: date,
    benchmark: str,
    config: LabConfig | None = None,
    refresh_research: bool = False,
    refresh_news: bool = False,
    security_master: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    held_symbols: list[str] | None = None,
    theme_breadth: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Create a bounded, timestamped packet using only recorded source evidence.

    Live public connectors are opt-in so simply rendering the dashboard never causes
    dozens of network requests. ``refresh_news`` lets analysis runs always collect
    fresh ticker news even when the broader evidence refresh is disabled.
    """
    config = config or LabConfig()
    columns = [column for column in ["symbol", "name", "theme", "seed_memberships", "sector", "industry", "latest_close", "score", "benchmark_adjusted_score", "relative_strength_score", "drawdown_52w", "extension_sma50d", "larsson_regime", "daily_trend", "weekly_trend", "monthly_trend", "data_warnings"] if column in candidates]
    facts = candidates[columns].head(config.research_limit).copy() if not candidates.empty else pd.DataFrame(columns=columns)
    if "sector" not in facts:
        facts["sector"] = facts.get("theme", "Unclassified")
    if "industry" not in facts:
        facts["industry"] = "Unclassified"
    facts["symbol"] = facts.get("symbol", pd.Series(dtype=str)).astype(str).str.upper()
    facts["eligible"] = facts["symbol"].ne("")
    facts["liquidity_score"] = facts.apply(_liquidity_score, axis=1) if not facts.empty else pd.Series(dtype=float)
    symbols = facts["symbol"].tolist()
    fundamentals: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    macro: dict[str, Any] = {"as_of_date": as_of_date.isoformat(), "regime": "unavailable", "series": {}}
    source_status = {
        "prices": "available", "fundamentals": "not_requested", "events": "not_requested",
        "macro": "not_requested", "news": "not_requested", "alpha_vantage_news": "not_configured",
        "gdelt_news": "not_requested",
    }
    connector_timings: dict[str, float] = {}
    source_diagnostics: dict[str, str] = {}
    news_window_start: datetime | None = None
    news_window_end: datetime | None = None
    recent_news_evidence: list[dict[str, object]] = []
    macro_evidence: list[dict[str, object]] = []
    sec_evidence: list[dict[str, object]] = []
    if refresh_research:
        started_at = perf_counter()
        macro, macro_evidence = fetch_macro_snapshot(as_of_date)
        connector_timings["macro"] = round((perf_counter() - started_at) * 1000, 1)
        started_at = perf_counter()
        fundamentals, sec_evidence = fetch_sec_company_research(symbols, config.research_limit)
        connector_timings["fundamentals"] = round((perf_counter() - started_at) * 1000, 1)
        connector_timings["events"] = connector_timings["fundamentals"]
        source_status.update({
            "fundamentals": "available" if fundamentals else "unavailable",
            "events": "available" if sec_evidence else "unavailable",
            "macro": "available" if macro.get("series") else "unavailable",
        })
        macro_failures = [str(item.get("reason", "")) for item in macro.get("series", {}).values() if isinstance(item, dict) and item.get("reason")]
        sec_failures = [str(item.get("reason", "")) for item in fundamentals.values() if isinstance(item, dict) and item.get("reason")]
        if macro_failures:
            source_diagnostics["macro"] = macro_failures[0][:160]
        if sec_failures:
            source_diagnostics["fundamentals"] = sec_failures[0][:160]
            source_diagnostics["events"] = sec_failures[0][:160]

    if refresh_research or refresh_news:
        names = {str(row.get("symbol", "")).upper(): str(row.get("name", row.get("symbol", ""))) for _, row in facts.iterrows()}
        alpha_news_evidence: list[dict[str, Any]] = []
        alpha_diagnostic: str | None = None
        alpha_available = bool(alpha_vantage_api_key())
        news_window_end = datetime.now(timezone.utc)
        news_window_start = news_window_end - timedelta(hours=48)
        if alpha_available:
            started_at = perf_counter()
            alpha_news_evidence, alpha_diagnostic = fetch_alpha_vantage_news(
                symbols,
                as_of_date,
                window_end=news_window_end,
            )
            connector_timings["alpha_vantage_news"] = round((perf_counter() - started_at) * 1000, 1)
        else:
            alpha_diagnostic = "credential not configured"
        alpha_covered_symbols = {
            str(item.get("symbol", "")).upper()
            for item in alpha_news_evidence
            if item.get("status") == "available"
        }
        fallback_symbols = [symbol for symbol in symbols if symbol not in alpha_covered_symbols]
        gdelt_news_evidence: list[dict[str, Any]] = []
        if fallback_symbols:
            started_at = perf_counter()
            alpha_urls = {
                str(item.get("source_url", ""))
                for item in alpha_news_evidence
                if item.get("status") == "available" and item.get("source_url")
            }
            gdelt_news_evidence = fetch_gdelt_news(
                fallback_symbols,
                names,
                config.research_limit,
                window_end=news_window_end,
                seen_urls=alpha_urls,
            )
            connector_timings["gdelt_news"] = round((perf_counter() - started_at) * 1000, 1)
        recent_news_evidence = alpha_news_evidence + gdelt_news_evidence
        available_recent_news = any(item.get("status") == "available" for item in recent_news_evidence)
        source_status.update({
            "news": "available" if available_recent_news else "unavailable",
            "alpha_vantage_news": "available" if alpha_news_evidence else ("unavailable" if alpha_available else "not_configured"),
            "gdelt_news": "available" if any(item.get("status") == "available" for item in gdelt_news_evidence) else ("unavailable" if fallback_symbols else "not_requested"),
        })
        if any(item.get("status") == "unavailable" for item in gdelt_news_evidence):
            source_diagnostics["gdelt_news"] = "One or more GDELT fallback requests were unavailable; retry on the next analysis."
        if not available_recent_news:
            source_diagnostics["news"] = "No fresh ticker-linked news was available in the prior 48 hours."
        if alpha_diagnostic:
            source_diagnostics["alpha_vantage_news"] = alpha_diagnostic
    evidence = macro_evidence + sec_evidence + recent_news_evidence
    scores: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for _, row in facts.iterrows():
        item = row.to_dict()
        symbol = str(item["symbol"])
        if "seed_memberships" in item:
            item["seed_memberships"] = _seed_membership_list(item.get("seed_memberships"))
        fund_score, factors = fundamental_score(fundamentals.get(symbol, {}))
        technical = float(pd.to_numeric(item.get("score", 0), errors="coerce") or 0)
        combined = round(technical if fund_score is None else 0.6 * technical + 0.4 * fund_score, 2)
        item["fundamental_score"] = fund_score
        item["combined_score"] = combined
        item["fundamental_factors"] = factors
        item["entry"] = entry_signal(
            prices if prices is not None else pd.DataFrame(), symbol,
            str(item.get("larsson_regime", "")), str(item.get("daily_trend", "")),
            str(item.get("weekly_trend", "")), str(item.get("monthly_trend", "")),
        )
        enriched.append(item)
        scores.append({"symbol": symbol, "technical_score": technical, "fundamental_score": fund_score, "combined_score": combined, "liquidity_score": item["liquidity_score"], "sector": item.get("sector", "Unclassified"), "score_json": json.dumps(factors, default=str)})
    enriched.sort(key=lambda item: float(item.get("combined_score", 0)), reverse=True)
    correlations = correlation_summary(prices if prices is not None else pd.DataFrame(), symbols, held_symbols)
    correlation_lookup = {str(item["symbol"]): item for item in correlations["candidate_summary"]}
    for item in enriched:
        item["correlation"] = correlation_lookup.get(str(item.get("symbol", "")), {"status": "unresolved"})
    for column in facts.select_dtypes(include=["datetime", "datetimetz"]).columns:
        facts[column] = facts[column].astype(str)
    quality = {}
    for key, status in source_status.items():
        available = status == "available"
        provenance = "primary" if key in {"fundamentals", "events", "macro"} else ("secondary" if key in {"news", "alpha_vantage_news", "gdelt_news"} else "curated")
        quality[key] = {"freshness": "current" if available else "unknown", "completeness": 1.0 if available else 0.0, "provenance": provenance, "conflict": "none", "score": 100 if available else 0}
    return {
        "packet_version": PROMPT_VERSION,
        "as_of_date": as_of_date.isoformat(), "benchmark": benchmark,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {"name": "local DuckDB/yfinance price cache", "fetched_at": datetime.now(timezone.utc).isoformat(), "status": "available"},
            {"name": "SEC EDGAR", "status": source_status["fundamentals"]},
            {"name": "FRED", "status": source_status["macro"]},
            {"name": "Alpha Vantage primary ticker news", "status": source_status["alpha_vantage_news"]},
            {"name": "GDELT fallback news discovery", "status": source_status["gdelt_news"]},
        ],
        "coverage": source_status, "source_quality": quality, "connector_timings_ms": connector_timings,
        "source_diagnostics": source_diagnostics, "macro": macro, "candidates": enriched,
        "theme_breadth": normalize_theme_breadth(theme_breadth),
        "recent_news": recent_news_payload(recent_news_evidence, news_window_start, news_window_end),
        "candidate_scores": scores, "evidence": evidence, "correlations": correlations,
        "security_master": (security_master if security_master is not None else facts)[[column for column in ["symbol", "name", "sector", "industry", "eligible", "effective_date", "source_url", "fetched_at"] if column in (security_master if security_master is not None else facts)]].to_dict(orient="records"),
        "missing_evidence_policy": "No evidence may be invented. Abstain when the packet cannot support a claim.",
    }


def _compact_value(value: object) -> object:
    """Remove empty values and normalize scalars for the model-facing packet."""
    if isinstance(value, dict):
        return {
            str(key): compacted
            for key, item in value.items()
            if (compacted := _compact_value(item)) not in (None, "", [], {})
        }
    if isinstance(value, (list, tuple)):
        return [compacted for item in value if (compacted := _compact_value(item)) not in (None, "", [], {})]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return None if pd.isna(value) else round(value, 4)
    if hasattr(value, "item"):
        try:
            return _compact_value(value.item())
        except (TypeError, ValueError):
            pass
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _compact_evidence(records: object) -> list[dict[str, object]]:
    """Bound source evidence while keeping the complete originals in the audit packet."""
    if not isinstance(records, list):
        return []
    fields = [
        "evidence_id", "symbol", "category", "title", "summary", "publisher",
        "published_at", "source_tier", "source_domain", "news_channel",
        "relevance_score", "sentiment_score", "sentiment_label",
    ]
    counts: dict[tuple[str, str], int] = {}
    compacted: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict) or record.get("status") != "available":
            continue
        category = str(record.get("category", ""))
        if category == "macro":
            continue
        symbol = str(record.get("symbol", "")).upper()
        key = (symbol, category)
        limit = MODEL_EVIDENCE_LIMITS.get(category, 3)
        if counts.get(key, 0) >= limit:
            continue
        item = {field: record.get(field) for field in fields if field in record}
        summary = str(item.get("summary", "") or "")
        if len(summary) > MODEL_EVIDENCE_SUMMARY_CHARS:
            item["summary"] = summary[: MODEL_EVIDENCE_SUMMARY_CHARS - 3].rstrip() + "..."
        cleaned = _compact_value(item)
        if isinstance(cleaned, dict):
            compacted.append(cleaned)
            counts[key] = counts.get(key, 0) + 1
    return compacted


def compact_model_packet(packet: dict[str, object]) -> dict[str, object]:
    """Return the bounded packet sent to GPT while preserving the full audit packet."""
    candidate_fields = [
        "symbol", "name", "seed_memberships", "theme", "sector", "industry",
        "latest_close", "score", "benchmark_adjusted_score", "relative_strength_score",
        "drawdown_52w", "extension_sma50d", "larsson_regime", "daily_trend",
        "weekly_trend", "monthly_trend", "data_warnings", "liquidity_score",
        "fundamental_score", "combined_score", "fundamental_factors", "entry", "correlation",
    ]
    candidates = []
    for candidate in packet.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        cleaned = _compact_value({field: candidate.get(field) for field in candidate_fields if field in candidate})
        if isinstance(cleaned, dict):
            candidates.append(cleaned)

    correlations = packet.get("correlations", {})
    compact_correlations = {}
    if isinstance(correlations, dict):
        correlation_fields = [
            "daily_window", "daily_min_observations", "weekly_window",
            "weekly_min_observations", "candidate_summary", "high_correlation_pairs", "status",
        ]
        compact_correlations = {
            field: correlations.get(field)
            for field in correlation_fields
            if field in correlations
        }

    model_packet = {
        "packet_version": packet.get("packet_version"),
        "as_of_date": packet.get("as_of_date"),
        "benchmark": packet.get("benchmark"),
        "coverage": packet.get("coverage", {}),
        "source_diagnostics": packet.get("source_diagnostics", {}),
        "macro": packet.get("macro", {}),
        "theme_breadth": packet.get("theme_breadth", []),
        "candidates": candidates,
        "correlations": compact_correlations,
        "evidence": _compact_evidence(packet.get("evidence", [])),
        "missing_evidence_policy": packet.get("missing_evidence_policy"),
    }
    cleaned = _compact_value(model_packet)
    return cleaned if isinstance(cleaned, dict) else {}


def allowed_packet_references(packet: dict[str, object]) -> list[str]:
    """Return the finite set of citation labels supported by a frozen packet.

    Literal evidence IDs are the preferred citations.  The small set of
    ``packet:`` references covers deterministic technical/correlation fields
    which are intentionally part of the same immutable packet, but do not have
    separate source records.
    """
    references = {
        str(item.get("evidence_id"))
        for item in packet.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    references.update({
        "packet:macro",
        "packet:candidates",
        "packet:candidates.entry",
        "packet:candidates.fundamental_factors",
        "packet:correlations",
    })
    correlations = packet.get("correlations", {})
    if isinstance(correlations, dict) and correlations.get("high_correlation_pairs"):
        references.add("packet:correlations.high_correlation_pairs")
    if isinstance(correlations, dict) and correlations.get("daily_matrix"):
        references.add("packet:correlations.daily_matrix")
    for breadth in packet.get("theme_breadth", []):
        if isinstance(breadth, dict) and breadth.get("reference"):
            references.add(str(breadth["reference"]))
    for candidate in packet.get("candidates", []):
        if not isinstance(candidate, dict) or not candidate.get("symbol"):
            continue
        symbol = str(candidate["symbol"]).upper()
        references.add(f"packet:candidates:{symbol}")
        if isinstance(correlations, dict) and correlations.get("daily_matrix"):
            references.add(f"packet:correlations.daily_matrix.{symbol}")
    return sorted(references)


def prompt_for_packet(packet: dict[str, object], config: LabConfig) -> str:
    model_packet = compact_model_packet(packet)
    allowed_references = allowed_packet_references(model_packet)
    return f"""You are Chatgptfolio's research underwriter. Use ONLY the timestamped research packet below. Do not invent news, fundamentals, prices, citations, targets, or events. This is paper-trading research, not investment advice. Prefer abstaining when evidence is insufficient.

Choose at most {config.max_holdings} liquid US equity/ETF symbols already in the packet. Long only; no options, leverage, shorts, or symbols outside the packet. The deterministic allocator—not you—sets final position sizes. Give each candidate an integer conviction from 0 to 100 and cite evidence IDs from the packet. Treat the packet's correlation data as a diversification preference: avoid redundant high-correlation combinations when conviction is comparable. Explain downside triggers and bull/base/bear valuation scenarios. Classify every investment risk as critical, high, medium, low, or unresolved. Every risk-flag evidence reference must exactly match one item in ALLOWED_EVIDENCE_REFERENCES below; return an empty array when no permitted citation supports the risk. Do NOT return entry prices, targets, stops, or valuation levels: the UI attaches deterministic price guidance from the packet. Cash target is a preference only and must be between 0 and 0.50. Active portfolio limits: max position {config.max_position:.1%}; sector cap {config.max_sector_exposure:.1%}; turnover budget {config.max_turnover:.1%}.

THEME_BREADTH_POLICY: `theme_breadth` is advisory context. Strong participation may support a thesis and deteriorating participation may support a risk, but breadth must never automatically exclude a candidate or determine position size. Treat `Unresolved` confirmations or coverage below 75% as insufficient breadth evidence. Use each candidate's `seed_memberships` to connect it to one or more seed snapshots.

RECENT_NEWS_POLICY: News evidence contains only the prior 48 hours of ticker-linked Alpha Vantage news, ranked by per-ticker relevance and sentiment strength, plus GDELT fallback articles only when Alpha Vantage had no fresh coverage for that symbol. Relevance and sentiment are evidence inputs, never automatic buy/sell instructions. Assess materiality yourself alongside source quality, technicals, fundamentals, breadth, and risks; ignore weak, irrelevant, or conflicting articles. If news materially informs a thesis, catalyst, or risk, include its evidence ID in that holding's `evidence_refs` or the relevant risk flag.

Return JSON matching the requested schema.

ALLOWED_EVIDENCE_REFERENCES:
{json.dumps(allowed_references)}

RESEARCH_PACKET:
{json.dumps(model_packet, sort_keys=True, default=str, separators=(",", ":"))}"""


DECISION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "abstain_reason": {"type": ["string", "null"]},
        "portfolio_thesis": {"type": "string"},
        "portfolio_risks": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"category": {"type": "string"}, "severity": {"type": "string", "enum": list(SEVERITY_ORDER)}, "rationale": {"type": "string"}, "evidence_refs": {"type": "array", "items": {"type": "string"}}}, "required": ["category", "severity", "rationale", "evidence_refs"]}},
        "cash_target": {"type": "number", "minimum": 0, "maximum": 0.5},
        "holdings": {"type": "array", "maxItems": 15, "items": {"type": "object", "additionalProperties": False, "properties": {"symbol": {"type": "string"}, "score": {"type": "number", "minimum": 0, "maximum": 100}, "confidence": {"type": "number", "minimum": 0, "maximum": 100}, "conviction": {"type": "number", "minimum": 0, "maximum": 100}, "thesis": {"type": "string"}, "catalysts": {"type": "array", "items": {"type": "string"}}, "risks": {"type": "array", "items": {"type": "string"}}, "risk_flags": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"category": {"type": "string"}, "severity": {"type": "string", "enum": list(SEVERITY_ORDER)}, "rationale": {"type": "string"}, "evidence_refs": {"type": "array", "items": {"type": "string"}}}, "required": ["category", "severity", "rationale", "evidence_refs"]}}, "downside_triggers": {"type": "array", "items": {"type": "string"}}, "scenarios": {"type": "object", "additionalProperties": False, "properties": {"bear": {"type": "string"}, "base": {"type": "string"}, "bull": {"type": "string"}}, "required": ["bear", "base", "bull"]}, "evidence_refs": {"type": "array", "items": {"type": "string"}}}, "required": ["symbol", "score", "confidence", "conviction", "thesis", "catalysts", "risks", "risk_flags", "downside_triggers", "scenarios", "evidence_refs"]}},
    }, "required": ["abstain_reason", "portfolio_thesis", "portfolio_risks", "cash_target", "holdings"],
}


def decision_schema_for_packet(packet: dict[str, object]) -> dict[str, Any]:
    """Constrain structured risk citations to the current immutable packet."""
    schema = json.loads(json.dumps(DECISION_SCHEMA))
    references = allowed_packet_references(compact_model_packet(packet)) or [""]
    reference_schema = {"type": "string", "enum": references}
    risk_reference_items = [
        schema["properties"]["portfolio_risks"]["items"],
        schema["properties"]["holdings"]["items"]["properties"]["risk_flags"]["items"],
    ]
    for risk_item in risk_reference_items:
        risk_item["properties"]["evidence_refs"]["items"] = reference_schema
    return schema


def evidence_health(packet: dict[str, object]) -> list[dict[str, str]]:
    """Classify evidence completeness deterministically for the UI and audit trail."""
    coverage = packet.get("coverage", {}) if isinstance(packet, dict) else {}
    labels = {"prices": "Price history", "fundamentals": "SEC fundamentals", "events": "SEC filings", "macro": "FRED macro", "news": "Recent ticker news", "alpha_vantage_news": "Alpha Vantage primary news", "gdelt_news": "GDELT fallback news"}
    output = []
    for key, label in labels.items():
        status = str(coverage.get(key, "unresolved"))
        if key == "prices" and status != "available": severity = "critical"
        elif status == "unavailable": severity = "critical" if key in {"fundamentals", "events"} else "high"
        elif status in {"not_requested", "not_configured"}: severity = "unresolved"
        elif status == "available": severity = "low"
        else: severity = "medium"
        quality = packet.get("source_quality", {}).get(key, {}) if isinstance(packet, dict) else {}
        output.append({"source": label, "status": status, "severity": severity, "freshness": str(quality.get("freshness", "unknown")), "quality_score": str(quality.get("score", 0))})
    return output


def severity_counts(items: list[dict[str, object]]) -> dict[str, int]:
    return {severity: sum(1 for item in items if item.get("severity") == severity) for severity in SEVERITY_ORDER}


def response_usage_metadata(response: object) -> dict[str, int]:
    """Normalize Responses API token accounting without depending on SDK internals."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    def value(source: object, key: str) -> int:
        raw = source.get(key, 0) if isinstance(source, dict) else getattr(source, key, 0)
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    input_details = usage.get("input_tokens_details", {}) if isinstance(usage, dict) else getattr(usage, "input_tokens_details", {})
    output_details = usage.get("output_tokens_details", {}) if isinstance(usage, dict) else getattr(usage, "output_tokens_details", {})
    return {
        "input_tokens": value(usage, "input_tokens"),
        "cached_input_tokens": value(input_details, "cached_tokens"),
        "cache_write_input_tokens": value(input_details, "cache_write_tokens"),
        "output_tokens": value(usage, "output_tokens"),
        "reasoning_tokens": value(output_details, "reasoning_tokens"),
        "total_tokens": value(usage, "total_tokens"),
    }


class OpenAIPortfolioModel:
    def __init__(self, model: str) -> None:
        if model not in MODEL_OPTIONS:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model

    def decide(self, packet: dict[str, object], config: LabConfig) -> tuple[dict[str, object], str, str, dict[str, int]]:
        api_key = openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured and no Chatgptfolio key was found in macOS Keychain. Add it to your environment or save it with: security add-generic-password -a \"$USER\" -s chatgptfolio-openai -w")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=self.model,
            input=prompt_for_packet(packet, config),
            reasoning={"effort": "high"} if self.model == DEFAULT_MODEL else {"effort": "medium"},
            text={"format": {"type": "json_schema", "name": "portfolio_decision", "strict": True, "schema": decision_schema_for_packet(packet)}},
        )
        raw = response.output_text
        decision = json.loads(raw)
        returned_model = getattr(response, "model", self.model)
        return decision, raw, returned_model, response_usage_metadata(response)


def openai_api_key() -> str | None:
    """Resolve an API key without persisting or logging it.

    Environment configuration has priority. On macOS, a local-only installation
    may instead keep the key in the user's Keychain under ``chatgptfolio-openai``.
    """
    environment_key = os.getenv("OPENAI_API_KEY")
    if environment_key:
        return environment_key
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-a", getpass.getuser(), "-s", "chatgptfolio-openai", "-w"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def optimize_decision(
    model_decision: dict[str, object],
    candidates: pd.DataFrame,
    config: LabConfig,
    previous: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Convert model conviction into deterministic target weights and residual cash."""
    output = json.loads(json.dumps(model_decision, default=str))
    allowed = candidates.copy()
    allowed["symbol"] = allowed.get("symbol", pd.Series(dtype=str)).astype(str).str.upper()
    sector_values = allowed["sector"] if "sector" in allowed else (allowed["theme"] if "theme" in allowed else pd.Series("Unclassified", index=allowed.index))
    allowed["sector"] = sector_values.fillna("Unclassified")
    lookup = allowed.set_index("symbol").to_dict(orient="index") if not allowed.empty else {}
    selected = []
    for item in model_decision.get("holdings", []):
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper()
        if symbol not in lookup or float(item.get("conviction", 0)) <= 0:
            continue
        selected.append((symbol, item, lookup[symbol]))
    selected = sorted(selected, key=lambda row: float(row[1].get("conviction", 0)), reverse=True)[: config.max_holdings]
    requested_cash = float(model_decision.get("cash_target", 0) or 0)
    requested_cash = min(0.5, max(0.0, requested_cash))
    target_equity = 1.0 - requested_cash
    sector_used: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    remaining = target_equity
    total_conviction = sum(float(item.get("conviction", 0)) for _, item, _ in selected) or 1.0
    for symbol, item, metadata in selected:
        sector = str(metadata.get("sector", "Unclassified"))
        desired = target_equity * float(item.get("conviction", 0)) / total_conviction
        sector_room = max(0.0, config.max_sector_exposure - sector_used.get(sector, 0.0))
        weight = max(0.0, min(config.max_position, sector_room, desired, remaining))
        if weight <= 0.00001:
            continue
        sector_used[sector] = sector_used.get(sector, 0.0) + weight
        remaining -= weight
        enriched = dict(item)
        enriched.update({"symbol": symbol, "weight": round(weight, 8), "sector": sector})
        rows.append(enriched)
    # Fill remaining capacity proportionally without breaking any deterministic cap.
    for _ in range(10):
        if remaining <= 0.00001 or not rows:
            break
        eligible = [row for row in rows if row["weight"] < config.max_position - 1e-8 and sector_used.get(str(row["sector"]), 0) < config.max_sector_exposure - 1e-8]
        if not eligible:
            break
        denominator = sum(float(row.get("conviction", 0)) for row in eligible) or len(eligible)
        used = 0.0
        for row in eligible:
            sector = str(row["sector"])
            room = min(config.max_position - float(row["weight"]), config.max_sector_exposure - sector_used[sector])
            addition = min(room, remaining * (float(row.get("conviction", 0)) / denominator))
            row["weight"] = round(float(row["weight"]) + addition, 8)
            sector_used[sector] += addition
            used += addition
        remaining -= used
        if used <= 1e-8:
            break
    cash = round(max(0.0, 1.0 - sum(float(row["weight"]) for row in rows)), 8)
    output["holdings"] = rows
    output["cash_weight"] = cash
    output["optimization"] = {"requested_cash": requested_cash, "allocated_equity": round(1.0 - cash, 8), "residual_cash": cash, "sector_weights": {key: round(value, 8) for key, value in sector_used.items()}, "binding_constraints": ["position_cap" if any(float(row["weight"]) >= config.max_position - 1e-6 for row in rows) else None, "sector_cap" if any(value >= config.max_sector_exposure - 1e-6 for value in sector_used.values()) else None]}
    output["optimization"]["binding_constraints"] = [value for value in output["optimization"]["binding_constraints"] if value]
    return output


def validate_decision(decision: dict[str, object], candidates: pd.DataFrame, config: LabConfig, previous: pd.DataFrame | None = None, packet: dict[str, object] | None = None) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    holdings = decision.get("holdings", [])
    if not isinstance(holdings, list):
        return ValidationResult(False, ["holdings must be an array"])
    allowed = set(candidates.get("symbol", pd.Series(dtype=str)).astype(str).str.upper())
    symbols = [str(item.get("symbol", "")).upper() for item in holdings if isinstance(item, dict)]
    if len(holdings) > config.max_holdings: errors.append("too many holdings")
    if len(symbols) != len(set(symbols)): errors.append("duplicate holdings")
    if any(symbol not in allowed for symbol in symbols): errors.append("proposal contains symbols outside the research packet")
    weights = [float(item.get("weight", -1)) for item in holdings if isinstance(item, dict)]
    if any(weight < 0 or weight > config.max_position for weight in weights): errors.append("position weight violates limit")
    cash = float(decision.get("cash_weight", -1))
    if cash < 0 or cash > 1: errors.append("cash weight must be between zero and one")
    if abs(sum(weights) + cash - 1.0) > 0.001: errors.append("weights plus cash must equal 1.0")
    sector_lookup = candidates.copy()
    sector_lookup["symbol"] = sector_lookup.get("symbol", pd.Series(dtype=str)).astype(str).str.upper()
    sector_values = sector_lookup["sector"] if "sector" in sector_lookup else (sector_lookup["theme"] if "theme" in sector_lookup else pd.Series("Unclassified", index=sector_lookup.index))
    sector_lookup["sector"] = sector_values.fillna("Unclassified")
    sectors = sector_lookup.set_index("symbol")["sector"].to_dict() if not sector_lookup.empty else {}
    sector_totals: dict[str, float] = {}
    for item, weight in zip(holdings, weights):
        sector = str(sectors.get(str(item.get("symbol", "")).upper(), "Unclassified")) if isinstance(item, dict) else "Unclassified"
        sector_totals[sector] = sector_totals.get(sector, 0) + weight
    if any(weight > config.max_sector_exposure + 1e-8 for weight in sector_totals.values()):
        errors.append("sector exposure violates limit")
    if previous is not None and not previous.empty:
        old = dict(zip(previous.get("symbol", []), previous.get("weight", [])))
        new = {str(item.get("symbol", "")).upper(): weight for item, weight in zip(holdings, weights) if isinstance(item, dict)}
        turnover = sum(abs(float(old.get(symbol, 0)) - float(new.get(symbol, 0))) for symbol in set(old) | set(new)) / 2
        if turnover > config.max_turnover + 1e-8:
            errors.append("turnover violates limit")
    if packet is not None:
        evidence_ids = set(allowed_packet_references(compact_model_packet(packet)))
        flags = list(decision.get("portfolio_risks", []))
        for item in holdings:
            if isinstance(item, dict):
                flags.extend(item.get("risk_flags", []))
        for flag in flags:
            if not isinstance(flag, dict):
                errors.append("risk flag must be an object")
                continue
            if flag.get("severity") not in SEVERITY_ORDER:
                errors.append("risk flag has invalid severity")
            refs = {str(ref) for ref in flag.get("evidence_refs", [])}
            if refs and not refs.issubset(evidence_ids):
                errors.append("risk flag references evidence outside the research packet")
        selected = set(symbols)
        correlation = packet.get("correlations", {}) if isinstance(packet, dict) else {}
        for pair in correlation.get("high_correlation_pairs", []) if isinstance(correlation, dict) else []:
            if isinstance(pair, dict) and {str(pair.get("left", "")), str(pair.get("right", ""))}.issubset(selected):
                warnings.append(f"high daily correlation: {pair.get('left')} / {pair.get('right')} ({float(pair.get('daily_correlation', 0)):.2f})")
        summary = {str(item.get("symbol")): item for item in correlation.get("candidate_summary", []) if isinstance(item, dict)} if isinstance(correlation, dict) else {}
        for symbol in selected:
            value = summary.get(symbol, {}).get("average_correlation_to_held")
            if isinstance(value, (int, float)) and value > 0.70:
                warnings.append(f"{symbol} has high average correlation to current holdings ({value:.2f})")
    return ValidationResult(not errors, list(dict.fromkeys(errors)), list(dict.fromkeys(warnings)))


def paper_orders(previous: pd.DataFrame, decision: dict[str, object], config: LabConfig) -> pd.DataFrame:
    prior = dict(zip(previous.get("symbol", []), previous.get("weight", []))) if not previous.empty else {}
    targets = {str(row["symbol"]).upper(): float(row["weight"]) for row in decision.get("holdings", [])}
    targets["Cash"] = float(decision.get("cash_weight", 0))
    rows = []
    for symbol in sorted(set(prior) | set(targets)):
        old, new = float(prior.get(symbol, 0)), float(targets.get(symbol, 0))
        delta = new - old
        if abs(delta) < 0.00001: continue
        notional = abs(delta) * config.paper_capital
        cost = notional * (config.slippage_bps + config.commission_bps) / 10_000
        rows.append({"symbol": symbol, "action": "BUY" if delta > 0 else "SELL", "previous_weight": old, "target_weight": new, "notional": notional, "estimated_cost": cost, "status": "proposed"})
    return pd.DataFrame(rows, columns=["symbol", "action", "previous_weight", "target_weight", "notional", "estimated_cost", "status"])


def decision_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def config_payload(config: LabConfig) -> dict[str, object]:
    return asdict(config)
