"""Point-in-time public research connectors for the paper-trading lab.

The connectors intentionally return partial coverage instead of fabricating data.  Each
record is normalized into auditable evidence before it reaches a model prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = os.getenv("CHATGPTFOLIO_USER_AGENT", "Chatgptfolio research lab contact@example.invalid")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=20) as response:  # noqa: S310 - fixed public endpoints
        return json.loads(response.read().decode("utf-8"))


def _get_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=20) as response:  # noqa: S310 - fixed/public RSS endpoints
        return response.read().decode("utf-8", errors="replace")


def evidence_record(
    symbol: str,
    category: str,
    title: str,
    summary: str,
    source_url: str,
    publisher: str,
    published_at: str | None = None,
    status: str = "available",
) -> dict[str, str | None]:
    fetched_at = _now()
    digest = hashlib.sha256(f"{symbol}|{category}|{title}|{source_url}|{published_at}".encode()).hexdigest()
    return {
        "evidence_id": digest[:24], "symbol": symbol.upper(), "category": category,
        "title": title[:500], "summary": summary[:2_000], "source_url": source_url,
        "publisher": publisher, "published_at": published_at, "fetched_at": fetched_at,
        "content_hash": digest, "status": status,
    }


def load_sp500_security_master() -> pd.DataFrame:
    """Load the current public S&P 500 constituent table for a frozen run snapshot."""
    try:
        tables = pd.read_html(WIKIPEDIA_SP500_URL)
        table = tables[0]
        result = pd.DataFrame({
            "symbol": table["Symbol"].astype(str).str.replace(".", "-", regex=False),
            "name": table["Security"].astype(str),
            "sector": table.get("GICS Sector", pd.Series("Unclassified", index=table.index)).astype(str),
            "industry": table.get("GICS Sub-Industry", pd.Series("Unclassified", index=table.index)).astype(str),
        })
        result["eligible"] = True
        result["effective_date"] = date.today().isoformat()
        result["source_url"] = WIKIPEDIA_SP500_URL
        result["fetched_at"] = _now()
        return result
    except Exception:
        return pd.DataFrame(columns=["symbol", "name", "sector", "industry", "eligible", "effective_date", "source_url", "fetched_at"])


def fetch_macro_snapshot(as_of_date: date) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    series = {"FEDFUNDS": "Federal funds rate", "CPIAUCSL": "Consumer price index", "UNRATE": "Unemployment rate", "T10Y2Y": "10Y-2Y Treasury spread"}
    values: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []
    for code, label in series.items():
        url = FRED_URL.format(series=code)
        try:
            frame = pd.read_csv(url)
            frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
            frame = frame[frame["observation_date"].dt.date <= as_of_date].dropna()
            if frame.empty:
                continue
            row = frame.iloc[-1]
            value = float(row[code])
            values[code] = {"label": label, "value": value, "observation_date": str(row["observation_date"].date())}
            evidence.append(evidence_record("_MACRO", "macro", label, f"{label}: {value} on {row['observation_date'].date()}", url, "FRED", str(row["observation_date"].date())))
        except Exception as exc:
            values[code] = {"label": label, "status": "unavailable", "reason": str(exc)[:160]}
    spread = values.get("T10Y2Y", {}).get("value")
    funds = values.get("FEDFUNDS", {}).get("value")
    regime = "risk_neutral"
    if isinstance(spread, (int, float)) and spread < 0:
        regime = "curve_inverted"
    elif isinstance(funds, (int, float)) and funds >= 5:
        regime = "restrictive_rates"
    return {"as_of_date": as_of_date.isoformat(), "regime": regime, "series": values}, evidence


def _latest_fact(facts: dict[str, Any], names: list[str], unit: str = "USD") -> tuple[float | None, str | None]:
    for name in names:
        fact = facts.get(name, {})
        units = fact.get("units", {})
        rows = units.get(unit) or next(iter(units.values()), [])
        rows = [row for row in rows if row.get("val") is not None and row.get("form") in {"10-K", "10-Q"}]
        if rows:
            row = sorted(rows, key=lambda item: (item.get("filed", ""), item.get("end", "")))[-1]
            return float(row["val"]), row.get("filed")
    return None, None


def fetch_sec_company_research(symbols: list[str], limit: int = 30) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Fetch SEC facts and filing events for a bounded candidate set."""
    try:
        mapping = _get_json(SEC_TICKERS_URL)
        by_symbol = {str(item["ticker"]).upper(): int(item["cik_str"]) for item in mapping.values()}
    except Exception:
        return {}, []
    output: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for symbol in [str(value).upper() for value in symbols[:limit]]:
        cik = by_symbol.get(symbol)
        if not cik:
            continue
        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        try:
            facts = _get_json(facts_url).get("facts", {}).get("us-gaap", {})
            revenue, revenue_date = _latest_fact(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"])
            net_income, _ = _latest_fact(facts, ["NetIncomeLoss"])
            assets, _ = _latest_fact(facts, ["Assets"])
            liabilities, _ = _latest_fact(facts, ["Liabilities"])
            cash, _ = _latest_fact(facts, ["CashAndCashEquivalentsAtCarryingValue"])
            margin = (net_income / revenue) if revenue and net_income is not None else None
            debt_ratio = (liabilities / assets) if assets and liabilities is not None else None
            output[symbol] = {"revenue": revenue, "net_margin": margin, "assets": assets, "debt_ratio": debt_ratio, "cash": cash, "reported_at": revenue_date, "source_url": facts_url}
            evidence.append(evidence_record(symbol, "fundamental", "SEC company facts", f"Revenue: {revenue}; net margin: {margin}; debt/assets: {debt_ratio}.", facts_url, "SEC EDGAR", revenue_date))
            submissions = _get_json(submissions_url).get("filings", {}).get("recent", {})
            forms = submissions.get("form", [])
            dates = submissions.get("filingDate", [])
            accessions = submissions.get("accessionNumber", [])
            primary_docs = submissions.get("primaryDocument", [])
            for form, filed, accession, document in list(zip(forms, dates, accessions, primary_docs))[:20]:
                if form not in {"10-K", "10-Q", "8-K"}:
                    continue
                accession_no_dash = str(accession).replace("-", "")
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dash}/{document}"
                evidence.append(evidence_record(symbol, "corporate_event", f"SEC {form} filed", f"{form} filed with the SEC.", url, "SEC EDGAR", filed))
        except Exception as exc:
            output[symbol] = {"status": "unavailable", "reason": str(exc)[:160], "source_url": facts_url}
    return output, evidence


def fetch_news(symbols: list[str], limit: int = 20, items_per_symbol: int = 3) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for symbol in [str(value).upper() for value in symbols[:limit]]:
        url = f"https://news.google.com/rss/search?q={quote_plus(symbol + ' stock')}&hl=en-US&gl=US&ceid=US:en"
        try:
            root = ET.fromstring(_get_text(url))
            for item in root.findall("./channel/item")[:items_per_symbol]:
                evidence.append(evidence_record(symbol, "news", item.findtext("title", "Untitled"), item.findtext("description", ""), item.findtext("link", url), item.findtext("source", "Google News"), item.findtext("pubDate")))
        except Exception:
            evidence.append(evidence_record(symbol, "news", "News feed unavailable", "No curated news was included for this symbol.", url, "Google News", status="unavailable"))
    return evidence


def fundamental_score(fundamentals: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    if not fundamentals or fundamentals.get("status") == "unavailable":
        return None, {"status": "unavailable"}
    points = 50.0
    margin = fundamentals.get("net_margin")
    debt = fundamentals.get("debt_ratio")
    if isinstance(margin, (int, float)):
        points += max(-20, min(25, margin * 100))
    if isinstance(debt, (int, float)):
        points -= max(-20, min(20, (debt - 0.5) * 20))
    return round(max(0, min(100, points)), 2), {"net_margin": margin, "debt_ratio": debt, "reported_at": fundamentals.get("reported_at")}
