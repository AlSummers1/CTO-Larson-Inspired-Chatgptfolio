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

MODEL_OPTIONS = ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini")
DEFAULT_MODEL = "gpt-5.5"
PROMPT_VERSION = "chatgptfolio-v1"


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


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def research_packet(candidates: pd.DataFrame, as_of_date: date, benchmark: str) -> dict[str, object]:
    """Build a bounded, timestamped packet from locally available market evidence."""
    columns = [column for column in ["symbol", "name", "theme", "latest_close", "score", "benchmark_adjusted_score", "relative_strength_score", "drawdown_52w", "extension_sma50d", "larsson_regime", "daily_trend", "weekly_trend", "monthly_trend", "data_warnings"] if column in candidates]
    facts = candidates[columns].head(60).copy() if not candidates.empty else pd.DataFrame(columns=columns)
    for column in facts.select_dtypes(include=["datetime", "datetimetz"]).columns:
        facts[column] = facts[column].astype(str)
    return {
        "as_of_date": as_of_date.isoformat(),
        "benchmark": benchmark,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": [{"name": "local DuckDB/yfinance price cache", "fetched_at": datetime.now(timezone.utc).isoformat(), "status": "available"}],
        "coverage": {"prices": "available", "fundamentals": "not yet connected", "events": "not yet connected", "news": "not yet connected"},
        "candidates": facts.to_dict(orient="records"),
    }


def prompt_for_packet(packet: dict[str, object], config: LabConfig) -> str:
    return f"""You are Chatgptfolio's research assistant. Use ONLY the research packet below. Do not invent news, fundamentals, prices, or citations. This is a paper-trading proposal, not investment advice. Prefer abstaining when evidence is insufficient. Select at most {config.max_holdings} liquid US equity/ETF symbols in the packet. Long only; no options, leverage, short positions, or symbols outside the packet. Weights plus cash_weight must equal 1.0.\n\nReturn JSON matching the requested schema.\n\nRESEARCH_PACKET:\n{json.dumps(packet, sort_keys=True, default=str)}"""


DECISION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "abstain_reason": {"type": ["string", "null"]},
        "portfolio_thesis": {"type": "string"},
        "cash_weight": {"type": "number", "minimum": 0, "maximum": 1},
        "holdings": {"type": "array", "maxItems": 15, "items": {"type": "object", "additionalProperties": False, "properties": {"symbol": {"type": "string"}, "weight": {"type": "number", "minimum": 0, "maximum": 1}, "score": {"type": "number", "minimum": 0, "maximum": 100}, "confidence": {"type": "number", "minimum": 0, "maximum": 100}, "thesis": {"type": "string"}, "catalysts": {"type": "array", "items": {"type": "string"}}, "risks": {"type": "array", "items": {"type": "string"}}, "evidence_refs": {"type": "array", "items": {"type": "string"}}}, "required": ["symbol", "weight", "score", "confidence", "thesis", "catalysts", "risks", "evidence_refs"]}},
    }, "required": ["abstain_reason", "portfolio_thesis", "cash_weight", "holdings"],
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


def validate_decision(decision: dict[str, object], candidates: pd.DataFrame, config: LabConfig) -> ValidationResult:
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
