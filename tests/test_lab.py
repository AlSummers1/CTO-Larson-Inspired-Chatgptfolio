from datetime import date, datetime, timezone

import pandas as pd
import pytest

from stockpicker.data import load_lab_runs, save_lab_research_details, save_lab_research_snapshot, save_lab_run
from stockpicker.lab import LabConfig, allowed_packet_references, correlation_summary, decision_schema_for_packet, entry_signal, evidence_health, openai_api_key, optimize_decision, paper_orders, prompt_for_packet, research_packet, validate_decision
from stockpicker.research import alpha_vantage_api_key, fetch_alpha_vantage_news, fetch_gdelt_news


def _candidates():
    return pd.DataFrame([{"symbol": "NVDA", "name": "NVIDIA", "theme": "AI", "latest_close": 100.0, "score": 90.0}])


def _prices(symbol: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"symbol": symbol, "date": pd.bdate_range("2026-01-01", periods=len(values)), "open": values, "high": values, "low": values, "close": values, "volume": 1_000_000})


def test_packet_is_point_in_time_and_bounded():
    packet = research_packet(_candidates(), date(2026, 7, 1), "SPY")
    assert packet["as_of_date"] == "2026-07-01"
    assert packet["candidates"][0]["symbol"] == "NVDA"


def test_validator_rejects_symbols_outside_packet_and_bad_weights():
    result = validate_decision({"holdings": [{"symbol": "BAD", "weight": 0.8}], "cash_weight": 0.2}, _candidates(), LabConfig())
    assert not result.valid
    assert "outside" in " ".join(result.errors)


def test_paper_orders_include_estimated_cost():
    decision = {"holdings": [{"symbol": "NVDA", "weight": 0.1}], "cash_weight": 0.9}
    orders = paper_orders(pd.DataFrame(), decision, LabConfig(paper_capital=10_000, slippage_bps=10))
    assert orders.loc[orders.symbol == "NVDA", "notional"].iloc[0] == 1000
    assert orders.loc[orders.symbol == "NVDA", "estimated_cost"].iloc[0] == 1


def test_lab_records_do_not_require_or_store_credentials(tmp_path):
    db_path = tmp_path / "lab.duckdb"
    packet_hash = save_lab_research_snapshot({"as_of_date": "2026-07-01", "candidates": []}, db_path)
    run_id = save_lab_run({"as_of_date": date(2026, 7, 1), "benchmark": "SPY", "model": "gpt-5.5", "status": "model_failed", "packet_hash": packet_hash, "error_message": "missing key"}, db_path)
    runs = load_lab_runs(db_path)
    assert run_id == 1
    assert runs["model"].tolist() == ["gpt-5.5"]


def test_optimizer_reconciles_weights_and_honors_position_and_sector_caps():
    candidates = pd.DataFrame([
        {"symbol": "AAA", "score": 90, "sector": "Tech"},
        {"symbol": "BBB", "score": 80, "sector": "Tech"},
        {"symbol": "CCC", "score": 70, "sector": "Health"},
    ])
    model = {
        "abstain_reason": None, "portfolio_thesis": "test", "cash_target": 0,
        "holdings": [
            {"symbol": "AAA", "conviction": 100}, {"symbol": "BBB", "conviction": 90}, {"symbol": "CCC", "conviction": 80},
        ],
    }
    config = LabConfig(max_position=0.2, max_sector_exposure=0.3)
    result = optimize_decision(model, candidates, config)
    assert abs(sum(row["weight"] for row in result["holdings"]) + result["cash_weight"] - 1) < 0.000001
    assert all(row["weight"] <= 0.2 for row in result["holdings"])
    assert result["cash_weight"] >= 0.4
    assert validate_decision(result, candidates, config).valid


def test_packet_records_explicit_missing_research_without_fetching():
    packet = research_packet(_candidates(), date(2026, 7, 1), "SPY", LabConfig(), refresh_research=False)
    assert packet["coverage"]["fundamentals"] == "not_requested"
    assert packet["missing_evidence_policy"].startswith("No evidence")
    assert packet["candidates"][0]["combined_score"] == 90.0
    assert packet["connector_timings_ms"] == {}


def test_news_refresh_runs_without_the_broader_evidence_refresh(monkeypatch):
    import stockpicker.lab as lab

    alpha_evidence = [{"evidence_id": "alpha-1", "symbol": "NVDA", "category": "news", "title": "Update", "summary": "Test", "source_url": "https://alpha.example/nvda", "publisher": "Alpha", "published_at": "2026-07-13T11:00:00+00:00", "status": "available", "news_channel": "alpha_vantage_primary"}]
    monkeypatch.setattr(lab, "fetch_macro_snapshot", lambda as_of: pytest.fail("macro refresh should stay disabled"))
    monkeypatch.setattr(lab, "fetch_sec_company_research", lambda symbols, limit: pytest.fail("SEC refresh should stay disabled"))
    monkeypatch.setattr(lab, "alpha_vantage_api_key", lambda: "configured")
    monkeypatch.setattr(lab, "fetch_alpha_vantage_news", lambda symbols, as_of, **kwargs: (alpha_evidence, None))

    packet = research_packet(_candidates(), date(2026, 7, 13), "SPY", refresh_research=False, refresh_news=True)

    assert packet["coverage"]["fundamentals"] == "not_requested"
    assert packet["coverage"]["alpha_vantage_news"] == "available"
    assert packet["recent_news"]["window_end"] is not None


def test_correlation_packet_flags_high_pairs_with_enough_observations():
    values = [100 + index for index in range(70)]
    summary = correlation_summary(pd.concat([_prices("AAA", values), _prices("BBB", values)], ignore_index=True), ["AAA"], held_symbols=["BBB"])
    assert summary["candidate_summary"][0]["daily_observations"] >= 40
    assert summary["candidate_summary"][0]["average_correlation_to_held"] == 1.0


def test_entry_signal_is_deterministic_and_does_not_require_model_prices():
    result = entry_signal(_prices("AAA", [100 + index * 0.1 for index in range(250)]), "AAA", "Bull", "Up", "Up", "Up")
    assert result["entry_label"] in {"Add", "Wait — confirm", "Wait — don't chase"}
    assert result["preferred_entry_low"] < result["preferred_entry_high"]
    assert entry_signal(_prices("AAA", [100] * 60), "AAA", "Bear", "Down", "Down", "Down")["entry_label"] == "Avoid"


def test_validator_reports_correlation_warning_without_rejecting_valid_weights():
    candidates = pd.DataFrame([{"symbol": "AAA", "sector": "One"}, {"symbol": "BBB", "sector": "Two"}])
    packet = {"evidence": [], "correlations": {"high_correlation_pairs": [{"left": "AAA", "right": "BBB", "daily_correlation": 0.9}], "candidate_summary": []}}
    result = validate_decision({"holdings": [{"symbol": "AAA", "weight": 0.4}, {"symbol": "BBB", "weight": 0.4}], "cash_weight": 0.2}, candidates, LabConfig(max_position=0.5), packet=packet)
    assert result.valid
    assert "high daily correlation" in " ".join(result.warnings)


def test_prompt_forbids_model_invented_entry_levels():
    prompt = prompt_for_packet({"candidates": [], "recent_news": {"by_symbol": {}}}, LabConfig())

    assert "Do NOT return entry prices" in prompt
    assert "Relevance and sentiment are evidence inputs, never automatic buy/sell instructions" in prompt
    assert "recent-news evidence ID" in prompt


def test_gdelt_evidence_is_secondary_and_deduplicated(monkeypatch):
    import stockpicker.research as research

    monkeypatch.setattr(research, "_get_json", lambda url: {"articles": [
        {"title": "Eligible", "url": "https://example.com/story", "domain": "example.com", "seendate": "20260711T120000Z"},
        {"title": "Duplicate", "url": "https://example.com/story", "domain": "example.com"},
        {"title": "Syndicated", "url": "https://www.prnewswire.com/story", "domain": "prnewswire.com"},
    ]})
    rows = fetch_gdelt_news(["NVDA"], {"NVDA": "NVIDIA"})
    available = [row for row in rows if row["status"] == "available"]
    assert len(available) == 1
    assert available[0]["source_tier"] == "secondary_discovery"


def test_alpha_vantage_news_uses_one_keyless_audit_url_and_ticker_links(monkeypatch):
    import stockpicker.research as research

    requested_urls = []
    monkeypatch.setattr(research, "alpha_vantage_api_key", lambda: "test-key")
    monkeypatch.setattr(research, "_get_json", lambda url: requested_urls.append(url) or {"feed": [
        {
            "title": "Semiconductor demand update",
            "summary": "A relevant update.",
            "url": "https://publisher.example/article",
            "source": "Publisher",
            "time_published": "20260713T1000",
            "ticker_sentiment": [
                {"ticker": "NVDA", "ticker_sentiment_label": "Bullish", "ticker_sentiment_score": "0.4"},
                {"ticker": "AMD", "ticker_sentiment_label": "Neutral", "ticker_sentiment_score": "0.0"},
            ],
        }
    ]})

    rows, diagnostic = fetch_alpha_vantage_news(["NVDA", "AMD"], date(2026, 7, 13))

    assert diagnostic is None
    assert len(requested_urls) == 1
    assert "test-key" in requested_urls[0]
    assert {row["symbol"] for row in rows} == {"NVDA", "AMD"}
    assert all(row["source_tier"] == "secondary_enrichment" for row in rows)
    assert all("test-key" not in str(row["source_url"]) for row in rows)


def test_alpha_vantage_news_keeps_fresh_articles_by_relevance_then_sentiment(monkeypatch):
    import stockpicker.research as research

    requested_urls = []
    monkeypatch.setattr(research, "alpha_vantage_api_key", lambda: "test-key")
    monkeypatch.setattr(research, "_get_json", lambda url: requested_urls.append(url) or {"feed": [
        {
            "title": "High relevance, modest sentiment", "summary": "A", "url": "https://publisher.example/a", "source": "Publisher",
            "time_published": "20260713T1100",
            "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.9", "ticker_sentiment_label": "Bullish", "ticker_sentiment_score": "0.1"}],
        },
        {
            "title": "High relevance, strong sentiment", "summary": "B", "url": "https://publisher.example/b", "source": "Publisher",
            "time_published": "20260713T1000",
            "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.9", "ticker_sentiment_label": "Bearish", "ticker_sentiment_score": "-0.8"}],
        },
        {
            "title": "Lower relevance", "summary": "C", "url": "https://publisher.example/c", "source": "Publisher",
            "time_published": "20260713T1130",
            "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.8", "ticker_sentiment_label": "Bullish", "ticker_sentiment_score": "0.95"}],
        },
        {
            "title": "Outside the 48-hour window", "summary": "Old", "url": "https://publisher.example/old", "source": "Publisher",
            "time_published": "20260711T1159",
            "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "1.0", "ticker_sentiment_label": "Bullish", "ticker_sentiment_score": "0.99"}],
        },
    ]})

    rows, diagnostic = fetch_alpha_vantage_news(
        ["NVDA"],
        date(2026, 7, 13),
        items_per_symbol=2,
        window_end=datetime(2026, 7, 13, 12, tzinfo=timezone.utc),
    )

    assert diagnostic is None
    assert "sort=RELEVANCE" in requested_urls[0]
    assert "time_from=20260711T1200" in requested_urls[0]
    assert "time_to=20260713T1200" in requested_urls[0]
    assert [row["title"] for row in rows] == ["High relevance, strong sentiment", "High relevance, modest sentiment"]
    assert rows[0]["relevance_score"] == 0.9
    assert rows[0]["sentiment_score"] == -0.8
    assert rows[0]["sentiment_label"] == "Bearish"
    assert all(row["news_channel"] == "alpha_vantage_primary" for row in rows)


def test_alpha_vantage_key_reads_macos_keychain_when_environment_is_missing(monkeypatch):
    import stockpicker.research as research

    class Result:
        returncode = 0
        stdout = "keychain-key\n"

    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.setattr(research.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(research.subprocess, "run", lambda *args, **kwargs: Result())
    assert alpha_vantage_api_key() == "keychain-key"


def test_refreshed_packet_includes_alpha_vantage_source_health(monkeypatch):
    import stockpicker.lab as lab

    alpha_evidence = [{"evidence_id": "alpha-1", "symbol": "NVDA", "category": "news", "title": "Update", "summary": "Test", "source_url": "https://example.com", "publisher": "Publisher", "published_at": None, "fetched_at": "2026-07-13T12:00:00+00:00", "content_hash": "alpha", "status": "available", "source_tier": "secondary_enrichment", "source_domain": "example.com"}]
    monkeypatch.setattr(lab, "fetch_macro_snapshot", lambda as_of: ({"series": {"FEDFUNDS": {"value": 4.0}}}, []))
    monkeypatch.setattr(lab, "fetch_sec_company_research", lambda symbols, limit: ({"NVDA": {}}, []))
    monkeypatch.setattr(lab, "fetch_gdelt_news", lambda symbols, names, limit, **kwargs: [])
    monkeypatch.setattr(lab, "alpha_vantage_api_key", lambda: "configured")
    monkeypatch.setattr(lab, "fetch_alpha_vantage_news", lambda symbols, as_of, **kwargs: (alpha_evidence, None))

    packet = research_packet(_candidates(), date(2026, 7, 13), "SPY", refresh_research=True)

    assert packet["coverage"]["alpha_vantage_news"] == "available"
    assert packet["sources"][-2]["name"] == "Alpha Vantage primary ticker news"
    assert packet["evidence"][-1]["evidence_id"] == "alpha-1"
    assert packet["recent_news"]["by_symbol"]["NVDA"][0]["evidence_id"] == "alpha-1"


def test_refreshed_packet_uses_gdelt_only_for_alpha_uncovered_symbols(monkeypatch):
    import stockpicker.lab as lab

    candidates = pd.DataFrame([
        {"symbol": "NVDA", "name": "NVIDIA", "theme": "AI", "latest_close": 100.0, "score": 90.0},
        {"symbol": "AMD", "name": "AMD", "theme": "AI", "latest_close": 80.0, "score": 80.0},
    ])
    alpha_evidence = [{"evidence_id": "alpha-1", "symbol": "NVDA", "category": "news", "title": "NVDA", "summary": "Test", "source_url": "https://alpha.example/nvda", "publisher": "Alpha", "published_at": "2026-07-13T11:00:00+00:00", "status": "available", "news_channel": "alpha_vantage_primary"}]
    gdelt_calls = []
    gdelt_evidence = [{"evidence_id": "gdelt-1", "symbol": "AMD", "category": "news", "title": "AMD", "summary": "Test", "source_url": "https://gdelt.example/amd", "publisher": "GDELT", "published_at": "2026-07-13T11:00:00+00:00", "status": "available", "news_channel": "gdelt_fallback"}]
    monkeypatch.setattr(lab, "fetch_macro_snapshot", lambda as_of: ({"series": {"FEDFUNDS": {"value": 4.0}}}, []))
    monkeypatch.setattr(lab, "fetch_sec_company_research", lambda symbols, limit: ({"NVDA": {}, "AMD": {}}, []))
    monkeypatch.setattr(lab, "alpha_vantage_api_key", lambda: "configured")
    monkeypatch.setattr(lab, "fetch_alpha_vantage_news", lambda symbols, as_of, **kwargs: (alpha_evidence, None))
    monkeypatch.setattr(lab, "fetch_gdelt_news", lambda symbols, names, limit, **kwargs: gdelt_calls.append(symbols) or gdelt_evidence)

    packet = research_packet(candidates, date(2026, 7, 13), "SPY", refresh_research=True)

    assert gdelt_calls == [["AMD"]]
    assert packet["coverage"]["news"] == "available"
    assert packet["coverage"]["gdelt_news"] == "available"
    assert set(packet["recent_news"]["by_symbol"]) == {"NVDA", "AMD"}


def test_packet_freezes_supplied_security_master():
    master = pd.DataFrame([{"symbol": "NVDA", "name": "NVIDIA", "sector": "Information Technology", "industry": "Semiconductors", "eligible": True, "effective_date": "2026-07-01"}])
    packet = research_packet(_candidates(), date(2026, 7, 1), "SPY", LabConfig(), security_master=master)
    assert packet["security_master"][0]["sector"] == "Information Technology"


def test_research_details_persist_source_level_evidence(tmp_path):
    db_path = tmp_path / "evidence.duckdb"
    packet = {
        "as_of_date": "2026-07-01", "candidates": [], "security_master": [],
        "candidate_scores": [{"symbol": "NVDA", "technical_score": 90, "fundamental_score": 80, "combined_score": 86, "liquidity_score": 90, "sector": "Tech", "score_json": "{}"}],
        "evidence": [{"evidence_id": "e1", "symbol": "NVDA", "category": "news", "title": "Test", "summary": "Test", "source_url": "https://example.com", "publisher": "Test", "published_at": "Fri, 10 Jul 2026 17:19:00 GMT", "fetched_at": "2026-07-01", "content_hash": "x", "status": "available"}],
    }
    packet_hash = save_lab_research_snapshot(packet, db_path)
    save_lab_research_details(packet_hash, packet, db_path)
    import duckdb
    with duckdb.connect(str(db_path), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM lab_evidence").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM lab_factor_scores").fetchone()[0] == 1


def test_openai_key_prefers_environment_over_keychain(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    assert openai_api_key() == "environment-key"


def test_openai_key_reads_macos_keychain_when_environment_is_missing(monkeypatch):
    import stockpicker.lab as lab

    class Result:
        returncode = 0
        stdout = "keychain-key\n"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(lab.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(lab.subprocess, "run", lambda *args, **kwargs: Result())
    assert openai_api_key() == "keychain-key"


def test_evidence_health_marks_missing_price_as_critical_and_unrequested_as_unresolved():
    health = evidence_health({"coverage": {"prices": "unavailable", "fundamentals": "not_requested", "events": "available", "macro": "available", "news": "available"}})
    levels = {item["source"]: item["severity"] for item in health}
    assert levels["Price history"] == "critical"
    assert levels["SEC fundamentals"] == "unresolved"


def test_validator_rejects_risk_reference_outside_packet():
    candidates = _candidates()
    decision = {"holdings": [{"symbol": "NVDA", "weight": 0.1, "risk_flags": [{"severity": "high", "evidence_refs": ["missing"]}]}], "cash_weight": 0.9, "portfolio_risks": []}
    result = validate_decision(decision, candidates, LabConfig(), packet={"evidence": []})
    assert not result.valid
    assert "outside" in " ".join(result.errors)


def test_packet_risk_references_allow_frozen_evidence_and_defined_technical_paths():
    candidates = pd.DataFrame([{"symbol": "AAA", "sector": "One"}])
    packet = {
        "evidence": [{"evidence_id": "source-1"}],
        "candidates": [{"symbol": "AAA"}],
        "correlations": {},
    }
    assert "source-1" in allowed_packet_references(packet)
    assert "packet:correlations.daily_matrix.AAA" in allowed_packet_references(packet)
    decision = {
        "holdings": [{"symbol": "AAA", "weight": 0.5, "risk_flags": [{"severity": "medium", "evidence_refs": ["packet:correlations.daily_matrix.AAA"]}]}],
        "cash_weight": 0.5,
        "portfolio_risks": [{"severity": "low", "evidence_refs": ["source-1"]}],
    }
    result = validate_decision(decision, candidates, LabConfig(max_position=0.5, max_sector_exposure=0.6), packet=packet)
    assert result.valid
    schema = decision_schema_for_packet(packet)
    refs = schema["properties"]["portfolio_risks"]["items"]["properties"]["evidence_refs"]["items"]["enum"]
    assert refs == allowed_packet_references(packet)


def test_validator_deduplicates_repeated_invalid_risk_references():
    candidates = pd.DataFrame([{"symbol": "AAA", "sector": "One"}])
    decision = {
        "holdings": [{"symbol": "AAA", "weight": 0.5, "risk_flags": [{"severity": "high", "evidence_refs": ["unknown"]}]}],
        "cash_weight": 0.5,
        "portfolio_risks": [{"severity": "high", "evidence_refs": ["unknown"]}],
    }
    result = validate_decision(decision, candidates, LabConfig(max_position=0.5, max_sector_exposure=0.6), packet={"evidence": []})
    assert result.errors == ["risk flag references evidence outside the research packet"]
