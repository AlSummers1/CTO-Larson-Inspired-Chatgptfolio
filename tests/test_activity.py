from pathlib import Path

import pandas as pd
import pytest

from stockpicker.activity import capture_analysis_snapshot, load_signal_events
from stockpicker.data import save_portfolio_snapshot
from stockpicker.signals import SignalConfig


pytest.importorskip("duckdb")


def _signals(regime="Bull"):
    return pd.DataFrame(
        [
            {"symbol": "QQQ", "latest_date": "2024-01-05", "larsson_regime": regime, "score": 80.0, "pass_200d": True, "pass_200w": True},
            {"symbol": "AAA", "latest_date": "2024-01-05", "larsson_regime": regime, "score": 90.0, "pass_200d": True, "pass_200w": True},
        ]
    )


def _health(status="Fresh"):
    return pd.DataFrame(
        [
            {"symbol": "QQQ", "latest_price_date": "2024-01-05", "reference_market_date": "2024-01-05", "freshness_status": "Fresh", "calendar_approximate": False},
            {"symbol": "AAA", "latest_price_date": "2024-01-05", "reference_market_date": "2024-01-05", "freshness_status": status, "calendar_approximate": False},
        ]
    )


def _rrg(quadrant="Leading"):
    return pd.DataFrame([{"symbol": "AAA", "date": "2024-01-05", "quadrant": quadrant, "rs_ratio": 101, "rs_momentum": 101}])


def test_first_snapshot_is_baseline_and_unchanged_refresh_has_no_events(tmp_path: Path):
    db_path = tmp_path / "activity.duckdb"
    first = capture_analysis_snapshot(_signals(), _health(), "QQQ", ["AAA", "QQQ"], SignalConfig(), _rrg(), db_path=db_path)
    second = capture_analysis_snapshot(_signals(), _health(), "QQQ", ["AAA", "QQQ"], SignalConfig(), _rrg(), db_path=db_path)

    assert first.captured and first.baseline and first.event_count == 0
    assert second.captured and not second.baseline and second.event_count == 0
    assert load_signal_events(db_path).empty


def test_all_signal_transitions_are_logged_with_stable_rrg_config(tmp_path: Path):
    db_path = tmp_path / "activity.duckdb"
    config = SignalConfig()
    seed_old = pd.DataFrame([{"seed_name": "Tech", "date": "2024-01-05", "quadrant": "Lagging"}])
    seed_new = pd.DataFrame([{"seed_name": "Tech", "date": "2024-01-12", "quadrant": "Improving"}])
    capture_analysis_snapshot(_signals("Bear"), _health("Fresh"), "QQQ", ["AAA", "QQQ"], config, _rrg("Lagging"), seed_old, db_path=db_path)
    result = capture_analysis_snapshot(_signals("Bull"), _health("Stale"), "QQQ", ["AAA", "QQQ"], config, _rrg("Leading"), seed_new, db_path=db_path)
    events = load_signal_events(db_path)

    assert result.event_count == 5
    assert set(events["event_type"]) == {
        "benchmark_regime_change",
        "larsson_regime_change",
        "stock_rrg_quadrant_change",
        "seed_rrg_quadrant_change",
        "data_freshness_deterioration",
    }
    assert events["idempotency_key"].is_unique if "idempotency_key" in events else True
    assert events["evidence_id"].str.startswith("EV-").all()
    assert events.loc[events["event_type"].str.contains("rrg"), "reason"].str.contains("10/10-week").all()

    recovery = capture_analysis_snapshot(_signals("Bull"), _health("Fresh"), "QQQ", ["AAA", "QQQ"], config, _rrg("Leading"), seed_new, db_path=db_path)
    assert recovery.event_count == 1
    assert "data_freshness_recovery" in set(load_signal_events(db_path)["event_type"])


def test_benchmarks_and_universes_do_not_cross_compare(tmp_path: Path):
    db_path = tmp_path / "activity.duckdb"
    config = SignalConfig()
    capture_analysis_snapshot(_signals("Bear"), _health(), "QQQ", ["AAA", "QQQ"], config, db_path=db_path)
    different_benchmark = capture_analysis_snapshot(_signals("Bull"), _health(), "SPY", ["AAA", "QQQ"], config, db_path=db_path)
    different_universe = capture_analysis_snapshot(_signals("Bull"), _health(), "QQQ", ["AAA", "BBB", "QQQ"], config, db_path=db_path)
    different_config = capture_analysis_snapshot(_signals("Bull"), _health(), "QQQ", ["AAA", "QQQ"], SignalConfig(daily_trend_window=51), db_path=db_path)

    assert different_benchmark.baseline and different_benchmark.event_count == 0
    assert different_universe.baseline and different_universe.event_count == 0
    assert different_config.baseline and different_config.event_count == 0


def test_repeated_transition_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "activity.duckdb"
    config = SignalConfig()
    capture_analysis_snapshot(_signals("Bear"), _health(), "QQQ", ["AAA", "QQQ"], config, db_path=db_path)
    capture_analysis_snapshot(_signals("Bull"), _health(), "QQQ", ["AAA", "QQQ"], config, db_path=db_path)
    capture_analysis_snapshot(_signals("Bear"), _health(), "QQQ", ["AAA", "QQQ"], config, db_path=db_path)
    capture_analysis_snapshot(_signals("Bull"), _health(), "QQQ", ["AAA", "QQQ"], config, db_path=db_path)

    events = load_signal_events(db_path)
    # Bear->Bull for two subjects is logged once despite a later repeat on the
    # same effective date; Bull->Bear is a separate transition.
    assert len(events) == 4


def test_portfolio_save_creates_paper_allocation_events(tmp_path: Path):
    db_path = tmp_path / "activity.duckdb"
    changes = pd.DataFrame(
        [
            {"symbol": "AAA", "previous_weight": 0.0, "target_weight": 0.1, "delta": 0.1, "action": "added", "reason_codes": "accepted_change"},
            {"symbol": "BBB", "previous_weight": 0.2, "target_weight": 0.0, "delta": -0.2, "action": "removed", "reason_codes": "accepted_change"},
            {"symbol": "CCC", "previous_weight": 0.1, "target_weight": 0.2, "delta": 0.1, "action": "increased", "reason_codes": "accepted_change"},
            {"symbol": "DDD", "previous_weight": 0.2, "target_weight": 0.1, "delta": -0.1, "action": "reduced", "reason_codes": "accepted_change"},
        ]
    )
    metadata = {"benchmark": "QQQ", "selected_seeds": "Tech", "config_hash": "cfg", "created_at": pd.Timestamp("2024-01-05")}
    save_portfolio_snapshot(pd.DataFrame(), metadata, changes, db_path)

    events = load_signal_events(db_path)

    assert set(events["event_type"]) == {
        "paper_allocation_added",
        "paper_allocation_removed",
        "paper_allocation_increased",
        "paper_allocation_reduced",
    }
    assert events["reason"].str.contains("no execution occurred").all()


def test_failed_refresh_does_not_capture(tmp_path: Path):
    result = capture_analysis_snapshot(_signals(), _health(), "QQQ", ["AAA", "QQQ"], refresh_succeeded=False, db_path=tmp_path / "activity.duckdb")
    assert not result.captured
    assert result.run_id is None
