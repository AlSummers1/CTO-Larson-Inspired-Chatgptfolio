"""Auditable OpenAI-backed paper-portfolio workflow.

The model proposes allocations; deterministic validation is always authoritative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import hashlib
import json
import os
from typing import Any

import pandas as pd

from .research import fetch_macro_snapshot, fetch_news, fetch_sec_company_research, fundamental_score

MODEL_OPTIONS = ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini")
DEFAULT_MODEL = "gpt-5.5"
PROMPT_VERSION = "chatgptfolio-v2-research-lab"


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


def _liquidity_score(row: pd.Series) -> float:
    volume = pd.to_numeric(row.get("avg_dollar_volume", row.get("volume", 0)), errors="coerce")
    if pd.isna(volume) or volume <= 0:
        return 50.0
    return float(max(0, min(100, 50 + 10 * (len(str(int(volume))) - 6))))


def research_packet(
    candidates: pd.DataFrame,
    as_of_date: date,
    benchmark: str,
    config: LabConfig | None = None,
    refresh_research: bool = False,
    security_master: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Create a bounded, timestamped packet using only recorded source evidence.

    Live public connectors are opt-in so simply rendering the dashboard never causes
    dozens of network requests.  Missing coverage is surfaced to the model and UI.
    """
    config = config or LabConfig()
    columns = [column for column in ["symbol", "name", "theme", "sector", "industry", "latest_close", "score", "benchmark_adjusted_score", "relative_strength_score", "drawdown_52w", "extension_sma50d", "larsson_regime", "daily_trend", "weekly_trend", "monthly_trend", "data_warnings"] if column in candidates]
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
    source_status = {"prices": "available", "fundamentals": "not_requested", "events": "not_requested", "macro": "not_requested", "news": "not_requested"}
    if refresh_research:
        macro, macro_evidence = fetch_macro_snapshot(as_of_date)
        fundamentals, sec_evidence = fetch_sec_company_research(symbols, config.research_limit)
        news_evidence = fetch_news(symbols, config.research_limit)
        evidence = macro_evidence + sec_evidence + news_evidence
        source_status.update({
            "fundamentals": "available" if fundamentals else "unavailable",
            "events": "available" if sec_evidence else "unavailable",
            "macro": "available" if macro.get("series") else "unavailable",
            "news": "available" if any(item.get("status") == "available" for item in news_evidence) else "unavailable",
        })
    scores: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for _, row in facts.iterrows():
        item = row.to_dict()
        symbol = str(item["symbol"])
        fund_score, factors = fundamental_score(fundamentals.get(symbol, {}))
        technical = float(pd.to_numeric(item.get("score", 0), errors="coerce") or 0)
        combined = round(technical if fund_score is None else 0.6 * technical + 0.4 * fund_score, 2)
        item["fundamental_score"] = fund_score
        item["combined_score"] = combined
        item["fundamental_factors"] = factors
        enriched.append(item)
        scores.append({"symbol": symbol, "technical_score": technical, "fundamental_score": fund_score, "combined_score": combined, "liquidity_score": item["liquidity_score"], "sector": item.get("sector", "Unclassified"), "score_json": json.dumps(factors, default=str)})
    enriched.sort(key=lambda item: float(item.get("combined_score", 0)), reverse=True)
    for column in facts.select_dtypes(include=["datetime", "datetimetz"]).columns:
        facts[column] = facts[column].astype(str)
    return {
        "packet_version": PROMPT_VERSION,
        "as_of_date": as_of_date.isoformat(), "benchmark": benchmark,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {"name": "local DuckDB/yfinance price cache", "fetched_at": datetime.now(timezone.utc).isoformat(), "status": "available"},
            {"name": "SEC EDGAR", "status": source_status["fundamentals"]},
            {"name": "FRED", "status": source_status["macro"]},
            {"name": "curated Google News RSS", "status": source_status["news"]},
        ],
        "coverage": source_status, "macro": macro, "candidates": enriched,
        "candidate_scores": scores, "evidence": evidence,
        "security_master": (security_master if security_master is not None else facts)[[column for column in ["symbol", "name", "sector", "industry", "eligible", "effective_date", "source_url", "fetched_at"] if column in (security_master if security_master is not None else facts)]].to_dict(orient="records"),
        "missing_evidence_policy": "No evidence may be invented. Abstain when the packet cannot support a claim.",
    }


def prompt_for_packet(packet: dict[str, object], config: LabConfig) -> str:
    return f"""You are Chatgptfolio's research underwriter. Use ONLY the timestamped research packet below. Do not invent news, fundamentals, prices, citations, targets, or events. This is paper-trading research, not investment advice. Prefer abstaining when evidence is insufficient.

Choose at most {config.max_holdings} liquid US equity/ETF symbols already in the packet. Long only; no options, leverage, shorts, or symbols outside the packet. The deterministic allocator—not you—sets final position sizes. Give each candidate an integer conviction from 0 to 100 and cite evidence IDs from the packet. Explain downside triggers and bull/base/bear valuation scenarios. Cash target is a preference only and must be between 0 and 0.50. Active portfolio limits: max position {config.max_position:.1%}; sector cap {config.max_sector_exposure:.1%}; turnover budget {config.max_turnover:.1%}.

Return JSON matching the requested schema.

RESEARCH_PACKET:
{json.dumps(packet, sort_keys=True, default=str)}"""


DECISION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "abstain_reason": {"type": ["string", "null"]},
        "portfolio_thesis": {"type": "string"},
        "cash_target": {"type": "number", "minimum": 0, "maximum": 0.5},
        "holdings": {"type": "array", "maxItems": 15, "items": {"type": "object", "additionalProperties": False, "properties": {"symbol": {"type": "string"}, "score": {"type": "number", "minimum": 0, "maximum": 100}, "confidence": {"type": "number", "minimum": 0, "maximum": 100}, "conviction": {"type": "number", "minimum": 0, "maximum": 100}, "thesis": {"type": "string"}, "catalysts": {"type": "array", "items": {"type": "string"}}, "risks": {"type": "array", "items": {"type": "string"}}, "downside_triggers": {"type": "array", "items": {"type": "string"}}, "scenarios": {"type": "object", "additionalProperties": False, "properties": {"bear": {"type": "string"}, "base": {"type": "string"}, "bull": {"type": "string"}}, "required": ["bear", "base", "bull"]}, "evidence_refs": {"type": "array", "items": {"type": "string"}}}, "required": ["symbol", "score", "confidence", "conviction", "thesis", "catalysts", "risks", "downside_triggers", "scenarios", "evidence_refs"]}},
    }, "required": ["abstain_reason", "portfolio_thesis", "cash_target", "holdings"],
}


class OpenAIPortfolioModel:
    def __init__(self, model: str) -> None:
        if model not in MODEL_OPTIONS:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model

    def decide(self, packet: dict[str, object], config: LabConfig) -> tuple[dict[str, object], str, str]:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured. Add it to your environment; it is never stored by Chatgptfolio.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            input=prompt_for_packet(packet, config),
            reasoning={"effort": "high"} if self.model == DEFAULT_MODEL else {"effort": "medium"},
            text={"format": {"type": "json_schema", "name": "portfolio_decision", "strict": True, "schema": DECISION_SCHEMA}},
        )
        raw = response.output_text
        decision = json.loads(raw)
        returned_model = getattr(response, "model", self.model)
        return decision, raw, returned_model


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


def validate_decision(decision: dict[str, object], candidates: pd.DataFrame, config: LabConfig, previous: pd.DataFrame | None = None) -> ValidationResult:
    errors: list[str] = []
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
    return ValidationResult(not errors, errors)


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
