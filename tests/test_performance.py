from __future__ import annotations

import json

import pandas as pd
import pytest

from stockpicker.performance import (
    build_saved_brief_checkpoints,
    build_saved_brief_portfolio_history,
    saved_brief_tracking_symbols,
)


def _brief(
    brief_id: int,
    evidence_date: str,
    allocations: list[tuple[str, float]] | None,
    *,
    benchmark: str = "QQQ",
    saved_at: str | None = None,
    title: str | None = None,
) -> dict[str, object]:
    evidence = [] if allocations is None else [
        {"kind": "paper_allocation", "symbol": symbol, "weight": weight}
        for symbol, weight in allocations
    ]
    return {
        "saved_brief_id": brief_id,
        "saved_title": title or f"Brief {brief_id}",
        "saved_at": saved_at or f"{evidence_date} 12:00:00",
        "evidence_json": json.dumps(
            {
                "benchmark": benchmark,
                "effective_market_date": evidence_date,
                "evidence": evidence,
            }
        ),
    }


def _price_rows(symbol: str, values: list[tuple[str, float, float]]) -> list[dict[str, object]]:
    return [
        {"symbol": symbol, "date": date, "open": open_price, "close": close_price}
        for date, open_price, close_price in values
    ]


def test_checkpoints_use_latest_saved_brief_for_same_evidence_date():
    checkpoints = build_saved_brief_checkpoints(
        [
            _brief(1, "2024-01-01", [("AAA", 1.0)], saved_at="2024-01-02 09:00:00"),
            _brief(2, "2024-01-01", [("BBB", 1.0)], saved_at="2024-01-02 10:00:00"),
        ]
    )

    assert checkpoints["status"].tolist() == ["Superseded", "Ready"]
    assert checkpoints.iloc[1]["weights"] == {"BBB": 1.0}
    assert saved_brief_tracking_symbols(checkpoints) == ("BBB", "QQQ", "SPY")


def test_checkpoints_coalesce_symbols_and_validate_allocations():
    checkpoints = build_saved_brief_checkpoints(
        [
            _brief(1, "2024-01-01", [("AAA", 0.4), ("AAA", 0.1), ("Cash", 0.5)]),
            _brief(2, "2024-01-02", None),
            _brief(3, "2024-01-03", [("AAA", 0.2)]),
        ]
    )

    assert checkpoints.iloc[0]["weights"] == pytest.approx({"AAA": 0.5, "Cash": 0.5})
    assert checkpoints.iloc[0]["holding_count"] == 1
    assert checkpoints.iloc[1]["status"] == "Skipped"
    assert checkpoints.iloc[2]["status"] == "Invalid"


def test_portfolio_starts_next_open_and_rebalances_without_changing_open_nav():
    checkpoints = build_saved_brief_checkpoints(
        [
            _brief(1, "2024-01-01", [("AAA", 0.5), ("Cash", 0.5)]),
            _brief(2, "2024-01-02", [("BBB", 1.0)], benchmark="SPY"),
        ]
    )
    prices = pd.DataFrame(
        [
            *_price_rows("AAA", [("2024-01-02", 10, 11), ("2024-01-03", 12, 12)]),
            *_price_rows("BBB", [("2024-01-03", 20, 22)]),
            *_price_rows("QQQ", [("2024-01-02", 100, 102), ("2024-01-03", 104, 105)]),
            *_price_rows("SPY", [("2024-01-02", 200, 202), ("2024-01-03", 204, 206)]),
        ]
    )

    result = build_saved_brief_portfolio_history(checkpoints, prices)

    assert result.checkpoints["status"].tolist() == ["Applied", "Applied"]
    assert result.checkpoints["effective_date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert result.nav["nav"].tolist() == pytest.approx([105_000.0, 121_000.0])
    assert result.nav.iloc[-1]["drawdown"] == pytest.approx(0.0)
    assert result.latest_holdings["symbol"].tolist() == ["BBB"]
    assert result.latest_holdings.iloc[0]["shares"] == pytest.approx(5_500.0)

    dynamic = result.benchmarks[result.benchmarks["series"] == "Brief benchmark"]
    spy = result.benchmarks[result.benchmarks["series"] == "SPY"]
    assert dynamic["symbol"].tolist() == ["QQQ", "SPY"]
    assert dynamic["nav"].tolist() == pytest.approx([102_000.0, 104_000.0 / 204.0 * 206.0])
    assert spy["nav"].tolist() == pytest.approx([101_000.0, 103_000.0])


def test_cash_only_checkpoint_is_valid_and_holds_value_flat():
    checkpoints = build_saved_brief_checkpoints([_brief(1, "2024-01-01", [("Cash", 1.0)], benchmark="SPY")])
    prices = pd.DataFrame(
        _price_rows("SPY", [("2024-01-02", 100, 101), ("2024-01-03", 102, 103)])
    )

    result = build_saved_brief_portfolio_history(checkpoints, prices)

    assert result.nav["nav"].tolist() == pytest.approx([100_000.0, 100_000.0])
    assert result.latest_holdings.iloc[0]["symbol"] == "Cash"
    assert result.benchmarks["series"].unique().tolist() == ["Brief benchmark · SPY"]


def test_pending_checkpoint_does_not_block_a_later_executable_checkpoint():
    checkpoints = build_saved_brief_checkpoints(
        [
            _brief(1, "2024-01-01", [("AAA", 1.0)]),
            _brief(2, "2024-01-02", [("MISSING", 1.0)]),
            _brief(3, "2024-01-03", [("BBB", 1.0)]),
        ]
    )
    prices = pd.DataFrame(
        [
            *_price_rows("AAA", [("2024-01-02", 10, 10), ("2024-01-03", 10, 10), ("2024-01-04", 10, 10)]),
            *_price_rows("BBB", [("2024-01-04", 20, 20)]),
            *_price_rows("QQQ", [("2024-01-02", 100, 100), ("2024-01-03", 100, 100), ("2024-01-04", 100, 100)]),
            *_price_rows("SPY", [("2024-01-02", 100, 100), ("2024-01-03", 100, 100), ("2024-01-04", 100, 100)]),
        ]
    )

    result = build_saved_brief_portfolio_history(checkpoints, prices)

    assert result.checkpoints["status"].tolist() == ["Applied", "Pending", "Applied"]
    assert "MISSING" in result.missing_symbols
    assert result.latest_holdings["symbol"].tolist() == ["BBB"]


def test_no_market_open_after_evidence_leaves_calm_pending_history():
    checkpoints = build_saved_brief_checkpoints([_brief(1, "2024-01-05", [("AAA", 1.0)])])
    prices = pd.DataFrame(
        [
            *_price_rows("AAA", [("2024-01-05", 10, 10)]),
            *_price_rows("QQQ", [("2024-01-05", 100, 100)]),
            *_price_rows("SPY", [("2024-01-05", 100, 100)]),
        ]
    )

    result = build_saved_brief_portfolio_history(checkpoints, prices)

    assert result.nav.empty
    assert result.checkpoints.iloc[0]["status"] == "Pending"
    assert "No eligible cached market open" in result.checkpoints.iloc[0]["diagnostic"]
