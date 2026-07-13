"""Point-in-time public research connectors for the paper-trading lab.

The connectors intentionally return partial coverage instead of fabricating data.  Each
record is normalized into auditable evidence before it reaches a model prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import getpass
import hashlib
import io
import json
import os
import platform
import subprocess
from time import monotonic
from typing import Any, Protocol
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = os.getenv("CHATGPTFOLIO_USER_AGENT", "Chatgptfolio research lab contact@example.invalid")
CONNECTOR_TIMEOUT_SECONDS = 10
MACRO_BUDGET_SECONDS = 30
COMPANY_RESEARCH_BUDGET_SECONDS = 45
NEWS_BUDGET_SECONDS = 30
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DENYLIST = {"prnewswire.com", "businesswire.com", "globenewswire.com", "accesswire.com", "news.google.com", "finance.yahoo.com", "reddit.com", "x.com", "twitter.com"}
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_KEYCHAIN_SERVICE = "chatgptfolio-alpha-vantage"


class EnrichmentProvider(Protocol):
    """Optional paid-data seam; no provider is enabled in the free local build."""

    def fundamentals(self, symbols: list[str]) -> dict[str, dict[str, Any]]: ...

    def earnings_calendar(self, symbols: list[str]) -> list[dict[str, Any]]: ...

    def news(self, symbols: list[str]) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class SourceQuality:
    freshness: str
    completeness: float
    provenance: str
    conflict: str = "none"

    @property
    def score(self) -> float:
        penalties = {"none": 0, "possible": 15, "confirmed": 40}
        freshness_score = {"current": 100, "stale": 65, "unknown": 35}.get(self.freshness, 35)
        return max(0.0, round(freshness_score * self.completeness - penalties.get(self.conflict, 25), 1))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=CONNECTOR_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed public endpoints
        return json.loads(response.read().decode("utf-8"))


def _get_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=CONNECTOR_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed/public RSS endpoints
        return response.read().decode("utf-8", errors="replace")


def alpha_vantage_api_key() -> str | None:
    """Resolve optional Alpha Vantage credentials without logging or persisting them."""
    environment_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if environment_key:
        return environment_key
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", getpass.getuser(), "-s", ALPHA_VANTAGE_KEYCHAIN_SERVICE, "-w"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def evidence_record(
    symbol: str,
    category: str,
    title: str,
    summary: str,
    source_url: str,
    publisher: str,
    published_at: str | None = None,
    status: str = "available",
    source_tier: str = "primary",
    source_domain: str | None = None,
) -> dict[str, str | None]:
    fetched_at = _now()
    digest = hashlib.sha256(f"{symbol}|{category}|{title}|{source_url}|{published_at}".encode()).hexdigest()
    return {
        "evidence_id": digest[:24], "symbol": symbol.upper(), "category": category,
        "title": title[:500], "summary": summary[:2_000], "source_url": source_url,
        "publisher": publisher, "published_at": published_at, "fetched_at": fetched_at,
        "content_hash": digest, "status": status,
        "source_tier": source_tier, "source_domain": source_domain or urlparse(source_url).netloc.lower(),
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
    started_at = monotonic()
    for code, label in series.items():
        url = FRED_URL.format(series=code)
        if monotonic() - started_at >= MACRO_BUDGET_SECONDS:
            values[code] = {"label": label, "status": "unavailable", "reason": "FRED connector time budget reached"}
            continue
        try:
            frame = pd.read_csv(io.StringIO(_get_text(url)))
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
    started_at = monotonic()
    for symbol in [str(value).upper() for value in symbols[:limit]]:
        if monotonic() - started_at >= COMPANY_RESEARCH_BUDGET_SECONDS:
            output[symbol] = {"status": "unavailable", "reason": "SEC connector time budget reached"}
            continue
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


def fetch_gdelt_news(symbols: list[str], company_names: dict[str, str] | None = None, limit: int = 30, items_per_symbol: int = 5) -> list[dict[str, Any]]:
    """Fetch bounded secondary news discovery; primary claims stay tied to SEC evidence."""
    evidence: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    started_at = monotonic()
    for symbol in [str(value).upper() for value in symbols[:limit]]:
        company = (company_names or {}).get(symbol, symbol)
        query = f'"{company}"'
        url = f"{GDELT_DOC_URL}?{urlencode({'query': query, 'mode': 'artlist', 'format': 'json', 'maxrecords': items_per_symbol, 'timespan': '7d'})}"
        if monotonic() - started_at >= NEWS_BUDGET_SECONDS:
            evidence.append(evidence_record(symbol, "news", "GDELT time budget reached", "No secondary news was included before the connector time budget expired.", url, "GDELT", status="unavailable", source_tier="secondary_discovery"))
            continue
        try:
            payload = _get_json(url)
            accepted = 0
            for article in payload.get("articles", []):
                article_url = str(article.get("url", ""))
                domain = str(article.get("domain", urlparse(article_url).netloc)).lower().removeprefix("www.")
                if not article_url or article_url in seen_urls or domain in GDELT_DENYLIST:
                    continue
                seen_urls.add(article_url)
                evidence.append(evidence_record(symbol, "news", str(article.get("title", "Untitled")), str(article.get("socialimage", "")), article_url, domain or "GDELT", str(article.get("seendate", "")) or None, source_tier="secondary_discovery", source_domain=domain))
                accepted += 1
                if accepted >= items_per_symbol:
                    break
            if accepted == 0:
                evidence.append(evidence_record(symbol, "news", "No eligible secondary articles", "GDELT returned no non-aggregated article for this company in the requested window.", url, "GDELT", status="unavailable", source_tier="secondary_discovery"))
        except Exception as exc:
            evidence.append(evidence_record(symbol, "news", "GDELT news unavailable", str(exc)[:160], url, "GDELT", status="unavailable", source_tier="secondary_discovery"))
    return evidence


def fetch_alpha_vantage_news(
    symbols: list[str],
    as_of_date: date,
    items_per_symbol: int = 5,
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch one bounded, ticker-linked Alpha Vantage news batch.

    A single multi-ticker request avoids consuming a request per candidate.  The
    credential is used only in the outgoing request; persisted evidence retains
    a public URL with the key removed.
    """
    api_key = alpha_vantage_api_key()
    if not api_key:
        return [], "credential not configured"
    requested = list(dict.fromkeys(str(value).upper() for value in symbols if str(value).strip()))
    if not requested:
        return [], None
    requested = requested[:30]
    start = as_of_date - timedelta(days=7)
    public_params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ",".join(requested),
        "time_from": start.strftime("%Y%m%dT0000"),
        "limit": min(200, max(25, len(requested) * items_per_symbol)),
    }
    public_url = f"{ALPHA_VANTAGE_URL}?{urlencode(public_params)}"
    request_url = f"{public_url}&{urlencode({'apikey': api_key})}"
    try:
        payload = _get_json(request_url)
    except Exception:
        return [], "request failed; retry on the next refresh"
    provider_message = str(payload.get("Note") or payload.get("Information") or payload.get("Error Message") or "").strip()
    if provider_message:
        return [], provider_message[:160]
    feed = payload.get("feed", [])
    if not isinstance(feed, list):
        return [], "response did not contain a news feed"
    evidence: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    counts: dict[str, int] = {symbol: 0 for symbol in requested}
    for article in feed:
        if not isinstance(article, dict):
            continue
        article_url = str(article.get("url", ""))
        if not article_url or article_url in seen_urls:
            continue
        article_symbols = {
            str(item.get("ticker", "")).upper()
            for item in article.get("ticker_sentiment", [])
            if isinstance(item, dict)
        }.intersection(requested)
        if not article_symbols:
            continue
        seen_urls.add(article_url)
        for symbol in sorted(article_symbols):
            if counts[symbol] >= items_per_symbol:
                continue
            sentiment = next(
                (item for item in article.get("ticker_sentiment", []) if isinstance(item, dict) and str(item.get("ticker", "")).upper() == symbol),
                {},
            )
            sentiment_label = str(sentiment.get("ticker_sentiment_label", ""))
            sentiment_score = str(sentiment.get("ticker_sentiment_score", ""))
            summary = str(article.get("summary", ""))
            if sentiment_label:
                summary = f"Alpha Vantage sentiment: {sentiment_label} ({sentiment_score}). {summary}".strip()
            evidence.append(
                evidence_record(
                    symbol,
                    "news",
                    str(article.get("title", "Untitled")),
                    summary,
                    article_url,
                    str(article.get("source", "Alpha Vantage")),
                    str(article.get("time_published", "")) or None,
                    source_tier="secondary_enrichment",
                )
            )
            counts[symbol] += 1
    return evidence, None if evidence else "no ticker-linked articles in the requested window"


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
