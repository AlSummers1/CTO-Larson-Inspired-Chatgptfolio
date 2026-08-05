import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import stockpicker.reports as reports_module
from stockpicker.activity import capture_analysis_snapshot
from stockpicker.data import save_portfolio_snapshot
from stockpicker.reports import (
    AICommentaryResult,
    REPORT_SECTIONS,
    build_paper_holding_allocation,
    build_paper_rrg_distribution,
    build_report_brief,
    build_weekly_report,
    generate_ai_commentary,
    load_saved_weekly_brief,
    load_saved_weekly_briefs,
    load_weekly_reports,
    openai_api_key,
    parse_commentary_sections,
    save_ai_commentary,
    save_weekly_brief,
)
from stockpicker.signals import SignalConfig


pytest.importorskip("duckdb")


def _seed_snapshot(db_path: Path, stale=False, score=90.0):
    signals = pd.DataFrame(
        [
            {"symbol": "QQQ", "latest_date": "2024-01-05", "larsson_regime": "Bull", "score": 80.0, "pass_200d": True, "pass_200w": True},
            {"symbol": "BBB", "latest_date": "2024-01-05", "larsson_regime": "Transition", "score": 70.0, "pass_200d": True, "pass_200w": False},
            {"symbol": "AAA", "latest_date": "2024-01-05", "larsson_regime": "Bull", "score": score, "pass_200d": True, "pass_200w": True},
        ]
    )
    health = pd.DataFrame(
        [
            {"symbol": symbol, "latest_price_date": "2024-01-05", "reference_market_date": "2024-01-05", "trading_sessions_behind": 2 if stale and symbol == "BBB" else 0, "freshness_status": "Stale" if stale and symbol == "BBB" else "Fresh", "calendar_approximate": False}
            for symbol in ["QQQ", "AAA", "BBB"]
        ]
    )
    return capture_analysis_snapshot(signals, health, "QQQ", ["AAA", "BBB", "QQQ"], SignalConfig(), db_path=db_path)


def test_report_has_required_sections_deterministic_order_and_stale_warning(tmp_path: Path):
    db_path = tmp_path / "reports.duckdb"
    _seed_snapshot(db_path, stale=True)

    report = build_weekly_report(db_path)

    positions = [report.markdown.index(f"## {section}") for section in REPORT_SECTIONS]
    assert positions == sorted(positions)
    assert "BBB: Stale" in report.markdown
    assert "raw prices are never forward-filled" in report.markdown
    assert report.markdown.index("AAA: score 90.0") < report.markdown.index("BBB: score 70.0")
    assert all(item["evidence_id"] in report.markdown for item in report.evidence_payload["evidence"])


def test_regeneration_only_creates_version_when_evidence_changes(tmp_path: Path):
    db_path = tmp_path / "reports.duckdb"
    _seed_snapshot(db_path, score=90.0)
    first = build_weekly_report(db_path)
    same = build_weekly_report(db_path)
    _seed_snapshot(db_path, score=91.0)
    changed = build_weekly_report(db_path)

    assert first.created_new
    assert not same.created_new
    assert same.report_id == first.report_id
    assert changed.created_new
    assert changed.version == first.version + 1


def test_report_brief_prioritizes_eligible_leaders_and_risks(tmp_path: Path):
    db_path = tmp_path / "brief.duckdb"
    _seed_snapshot(db_path, stale=True)

    report = build_weekly_report(db_path)
    brief = build_report_brief(report.evidence_payload)

    assert brief.benchmark == "QQQ"
    assert brief.benchmark_regime == "Bull"
    assert brief.fresh_count == 2
    assert brief.stock_count == 3
    assert [item.subject for item in brief.leaders] == ["AAA"]
    assert [item.subject for item in brief.risks] == ["BBB"]
    assert "stale data" in brief.risks[0].detail
    assert brief.evidence_count == len(report.evidence_payload["evidence"])


def test_report_brief_caps_sections_and_reports_overflow():
    evidence = [
        {
            "evidence_id": f"SG-{index}",
            "kind": "signal_snapshot",
            "subject_type": "stock",
            "subject": f"S{index}",
            "freshness_status": "Fresh",
            "larsson_regime": "Bull",
            "rrg_quadrant": "Leading",
            "pass_200d": True,
            "score": 100 - index,
        }
        for index in range(4)
    ]
    payload = {"benchmark": "QQQ", "effective_market_date": "2024-01-05", "evidence": evidence}

    brief = build_report_brief(payload, item_limit=2)

    assert [item.subject for item in brief.leaders] == ["S0", "S1"]
    assert brief.leader_overflow == 2


def test_missing_credentials_leave_local_report_usable(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "reports.duckdb"
    _seed_snapshot(db_path)
    report = build_weekly_report(db_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(reports_module.platform, "system", lambda: "Linux")

    result = generate_ai_commentary(report.evidence_payload)

    assert result.status == "missing_key"
    assert report.markdown.startswith("# Weekly Technical Analysis Report")


def test_openai_key_prefers_environment_over_keychain(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", " environment-key ")
    monkeypatch.setattr(
        reports_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Keychain should not be read when an environment key exists."),
    )

    assert openai_api_key() == "environment-key"


def test_openai_key_reads_current_users_macos_keychain(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="keychain-key\n")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(reports_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(reports_module.getpass, "getuser", lambda: "test-user")
    monkeypatch.setattr(reports_module.subprocess, "run", fake_run)

    assert openai_api_key() == "keychain-key"
    assert calls[0][0] == [
        "security",
        "find-generic-password",
        "-a",
        "test-user",
        "-s",
        "chatgptfolio-openai",
        "-w",
    ]
    assert calls[0][1]["capture_output"] is True


def test_openai_key_skips_keychain_outside_macos(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(reports_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        reports_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Keychain should not be read outside macOS."),
    )

    assert openai_api_key() is None


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.output, Exception):
            raise self.output
        return SimpleNamespace(output_text=json.dumps(self.output), usage={"input_tokens": 10, "output_tokens": 5})


def _ai_payload():
    return {"evidence": [{"evidence_id": "SG-ABC", "kind": "signal_snapshot", "subject": "AAA"}]}


def _valid_ai_output(evidence_id="SG-ABC"):
    return {
        "summary": [{"text": "AAA is constructive.", "evidence_ids": [evidence_id]}],
        "opportunities": [],
        "risks": [],
        "watch_items": [],
    }


def test_ai_commentary_rejects_unknown_evidence_references():
    responses = FakeResponses(_valid_ai_output("UNKNOWN"))
    result = generate_ai_commentary(_ai_payload(), client=SimpleNamespace(responses=responses))

    assert result.status == "invalid_evidence"
    assert not result.commentary
    assert responses.calls[0]["store"] is False
    assert responses.calls[0]["model"] == "gpt-5.6-terra"


def test_ai_commentary_rejects_invalid_schema():
    responses = FakeResponses({"summary": []})
    result = generate_ai_commentary(_ai_payload(), client=SimpleNamespace(responses=responses))

    assert result.status == "invalid_schema"
    assert not result.commentary


def test_valid_ai_commentary_is_labeled_and_persisted(tmp_path: Path):
    db_path = tmp_path / "reports.duckdb"
    _seed_snapshot(db_path)
    report = build_weekly_report(db_path)
    first_id = report.evidence_payload["evidence"][0]["evidence_id"]
    responses = FakeResponses(_valid_ai_output(first_id))

    result = generate_ai_commentary(report.evidence_payload, client=SimpleNamespace(responses=responses))
    save_ai_commentary(report.report_id, result, db_path)
    stored = load_weekly_reports(db_path).iloc[0]

    assert result.status == "success"
    assert "Interpretation Only" in result.commentary
    assert stored["generation_status"] == "success"
    assert stored["ai_model"] == "gpt-5.6-terra"
    assert first_id in stored["commentary"]
    assert parse_commentary_sections(stored["commentary_json"])["summary"][0]["text"]
    assert not pd.isna(stored["commentary_created_at"])


def test_legacy_commentary_without_structured_json_has_safe_fallback():
    assert parse_commentary_sections(None) == {}
    assert parse_commentary_sections("") == {}
    assert parse_commentary_sections('{"summary": []}') == {}


def test_named_weekly_briefs_round_trip_and_allow_duplicate_titles(tmp_path: Path):
    db_path = tmp_path / "saved-briefs.duckdb"
    _seed_snapshot(db_path)
    report = build_weekly_report(db_path)

    first = save_weekly_brief("  QQQ Weekly Brief  ", report.report_id, db_path)
    second = save_weekly_brief("QQQ Weekly Brief", report.report_id, db_path)
    history = load_saved_weekly_briefs(db_path)
    loaded = load_saved_weekly_brief(first.saved_brief_id, db_path)

    assert first.title == "QQQ Weekly Brief"
    assert second.saved_brief_id != first.saved_brief_id
    assert history["title"].tolist() == ["QQQ Weekly Brief", "QQQ Weekly Brief"]
    assert loaded["saved_title"] == "QQQ Weekly Brief"
    assert loaded["source_report_id"] == report.report_id
    assert loaded["markdown"] == report.markdown
    assert json.loads(loaded["evidence_json"]) == report.evidence_payload


def test_saved_weekly_brief_is_immutable_after_source_commentary_changes(tmp_path: Path):
    db_path = tmp_path / "immutable-brief.duckdb"
    _seed_snapshot(db_path)
    report = build_weekly_report(db_path)
    original_sections = {
        "summary": [{"text": "Original view.", "evidence_ids": [report.evidence_payload["evidence"][0]["evidence_id"]]}],
        "opportunities": [],
        "risks": [],
        "watch_items": [],
    }
    save_ai_commentary(
        report.report_id,
        AICommentaryResult(status="success", commentary="Original commentary", sections=original_sections, model="test-model"),
        db_path,
    )
    saved = save_weekly_brief("Immutable", report.report_id, db_path)
    save_ai_commentary(
        report.report_id,
        AICommentaryResult(status="success", commentary="Updated commentary", sections={**original_sections, "summary": [{"text": "Updated view.", "evidence_ids": original_sections["summary"][0]["evidence_ids"]}]}, model="test-model"),
        db_path,
    )

    loaded = load_saved_weekly_brief(saved.saved_brief_id, db_path)

    assert loaded["commentary"] == "Original commentary"
    assert parse_commentary_sections(loaded["commentary_json"])["summary"][0]["text"] == "Original view."


@pytest.mark.parametrize(
    ("title", "message"),
    [("   ", "required"), ("x" * 121, "120 characters")],
)
def test_saved_weekly_brief_title_validation(tmp_path: Path, title: str, message: str):
    db_path = tmp_path / "title-validation.duckdb"
    _seed_snapshot(db_path)
    report = build_weekly_report(db_path)

    with pytest.raises(ValueError, match=message):
        save_weekly_brief(title, report.report_id, db_path)


def test_saved_weekly_brief_rejects_missing_source_and_malformed_snapshot(tmp_path: Path):
    db_path = tmp_path / "invalid-saved-brief.duckdb"
    _seed_snapshot(db_path)
    report = build_weekly_report(db_path)
    with pytest.raises(ValueError, match="source weekly report"):
        save_weekly_brief("Missing", 999, db_path)

    saved = save_weekly_brief("Broken", report.report_id, db_path)
    with reports_module._connect(db_path) as conn:
        conn.execute(
            "UPDATE saved_weekly_briefs SET snapshot_json = 'not-json' WHERE saved_brief_id = ?",
            [saved.saved_brief_id],
        )

    with pytest.raises(ValueError, match="invalid"):
        load_saved_weekly_brief(saved.saved_brief_id, db_path)


def test_paper_rrg_distribution_uses_captured_portfolio_weights():
    payload = {
        "evidence": [
            {"kind": "paper_allocation", "symbol": "AAA", "weight": 0.20, "rrg_quadrant": "Leading"},
            {"kind": "paper_allocation", "symbol": "BBB", "weight": 0.10, "rrg_quadrant": "Leading"},
            {"kind": "paper_allocation", "symbol": "CCC", "weight": 0.30, "rrg_quadrant": "Lagging"},
            {"kind": "paper_allocation", "symbol": "Cash", "weight": 0.25, "rrg_quadrant": ""},
            {"kind": "paper_allocation", "symbol": "DDD", "weight": 0.15, "rrg_quadrant": "Unknown"},
            {"kind": "signal_snapshot", "subject": "IGNORED", "rrg_quadrant": "Improving"},
        ]
    }

    distribution = build_paper_rrg_distribution(payload)

    assert distribution["quadrant"].tolist() == ["Leading", "Lagging", "Cash / Unclassified"]
    assert distribution["weight"].tolist() == pytest.approx([0.30, 0.30, 0.40])
    assert distribution["holding_count"].tolist() == [2, 1, 2]
    assert "AAA 20.0%" in distribution.iloc[0]["holdings"]
    assert "Cash 25.0%" in distribution.iloc[2]["holdings"]


def test_paper_rrg_distribution_has_calm_empty_result():
    assert build_paper_rrg_distribution({"evidence": []}).empty
    assert build_paper_rrg_distribution({"evidence": [{"kind": "paper_allocation", "weight": 0}]}).empty


def test_paper_holding_allocation_uses_only_captured_positive_weights():
    payload = {
        "evidence": [
            {"kind": "paper_allocation", "symbol": "AAA", "weight": 0.40},
            {"kind": "paper_allocation", "symbol": "AAA", "weight": 0.10},
            {"kind": "paper_allocation", "symbol": "Cash", "weight": 0.50},
            {"kind": "paper_allocation", "symbol": "IGNORED", "weight": 0.0},
            {"kind": "signal_snapshot", "symbol": "NOT-A-HOLDING", "weight": 1.0},
        ]
    }

    allocation = build_paper_holding_allocation(payload)

    assert allocation["symbol"].tolist() == ["AAA", "Cash"]
    assert allocation["weight"].tolist() == pytest.approx([0.50, 0.50])
    assert build_paper_holding_allocation({"evidence": []}).empty


def test_api_failure_returns_concise_diagnostic():
    responses = FakeResponses(RuntimeError("secret internal details"))
    result = generate_ai_commentary(_ai_payload(), client=SimpleNamespace(responses=responses))

    assert result.status == "api_error"
    assert "secret internal details" not in result.diagnostic


def test_timeout_and_refusal_leave_no_commentary():
    timeout = generate_ai_commentary(
        _ai_payload(),
        client=SimpleNamespace(responses=FakeResponses(TimeoutError("late"))),
    )

    class RefusalResponses:
        @staticmethod
        def create(**_kwargs):
            return SimpleNamespace(output_text="", usage={"input_tokens": 2})

    refusal = generate_ai_commentary(_ai_payload(), client=SimpleNamespace(responses=RefusalResponses()))

    assert timeout.status == "timeout" and not timeout.commentary
    assert refusal.status == "refusal" and not refusal.commentary


def test_local_report_never_constructs_api_client(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "reports.duckdb"
    _seed_snapshot(db_path)
    called = []
    monkeypatch.setattr(reports_module, "_import_openai", lambda: called.append(True))

    build_weekly_report(db_path)

    assert called == []
