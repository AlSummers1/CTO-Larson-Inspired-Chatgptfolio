import pandas as pd
import pytest

from stockpicker.portfolio import (
    PortfolioConfig,
    allocate_portfolio,
    apply_sensitivity_to_target,
    build_allocator_prompt,
    build_portfolio_candidates,
    cash_target_for_regime,
    compare_portfolios,
    canonical_selected_seeds,
    portfolio_config_hash,
    rrg_pressure,
    rrg_seed_summary,
)


def ranked_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "name": "AAA Corp",
                "score": 95.0,
                "benchmark_adjusted_score": 96.0,
                "relative_strength_score": 90.0,
                "larsson_regime": "Bull",
                "daily_trend": "Up",
                "weekly_trend": "Up",
                "monthly_trend": "Up",
                "pass_200d": True,
                "pass_200w": True,
                "drawdown_52w": -0.05,
                "extension_sma50d": 0.05,
                "data_warnings": "",
            },
            {
                "symbol": "BBB",
                "name": "BBB Corp",
                "score": 85.0,
                "benchmark_adjusted_score": 86.0,
                "relative_strength_score": 80.0,
                "larsson_regime": "Bull",
                "daily_trend": "Up",
                "weekly_trend": "Up",
                "monthly_trend": "Up",
                "pass_200d": True,
                "pass_200w": True,
                "drawdown_52w": -0.08,
                "extension_sma50d": 0.03,
                "data_warnings": "",
            },
            {
                "symbol": "CCC",
                "name": "CCC Corp",
                "score": 80.0,
                "benchmark_adjusted_score": 81.0,
                "relative_strength_score": 75.0,
                "larsson_regime": "Bear",
                "daily_trend": "Down",
                "weekly_trend": "Down",
                "monthly_trend": "Mixed",
                "pass_200d": True,
                "pass_200w": False,
                "drawdown_52w": -0.40,
                "extension_sma50d": -0.10,
                "data_warnings": "",
            },
            {
                "symbol": "DDD",
                "name": "DDD Corp",
                "score": 70.0,
                "benchmark_adjusted_score": 71.0,
                "relative_strength_score": 65.0,
                "larsson_regime": "Bull",
                "daily_trend": "Up",
                "weekly_trend": "Mixed",
                "monthly_trend": "Mixed",
                "pass_200d": False,
                "pass_200w": True,
                "drawdown_52w": -0.10,
                "extension_sma50d": -0.02,
                "data_warnings": "",
            },
        ]
    )


def membership_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "seed_universe": "Seed A"},
            {"symbol": "BBB", "seed_universe": "Seed B"},
            {"symbol": "CCC", "seed_universe": "Seed B"},
            {"symbol": "DDD", "seed_universe": "Seed A"},
        ]
    )


def test_rrg_pressure_rewards_improving_and_penalizes_weakening():
    improving_score, improving_reason = rrg_pressure("Leading", 1.0, 1.0)
    weakening_score, weakening_reason = rrg_pressure("Weakening", -1.0, -1.0)

    assert improving_score > weakening_score
    assert improving_reason == "rrg_seed_improving"
    assert weakening_reason == "rrg_seed_weakening"


def test_rrg_seed_summary_uses_latest_tail_delta():
    history = pd.DataFrame(
        [
            {"seed_name": "Seed A", "date": pd.Timestamp("2024-01-05"), "rs_ratio": 99.0, "rs_momentum": 98.0, "quadrant": "Improving"},
            {"seed_name": "Seed A", "date": pd.Timestamp("2024-01-12"), "rs_ratio": 101.0, "rs_momentum": 102.0, "quadrant": "Leading"},
        ]
    )

    summary = rrg_seed_summary(history)

    assert summary.loc[0, "seed_universe"] == "Seed A"
    assert summary.loc[0, "rrg_quadrant"] == "Leading"
    assert summary.loc[0, "rs_ratio_delta"] == pytest.approx(2.0)
    assert summary.loc[0, "rs_momentum_delta"] == pytest.approx(4.0)


def test_candidate_selection_excludes_weak_and_failed_200d_by_default():
    config = PortfolioConfig(selected_seeds=("Seed A", "Seed B"), top_candidates_per_seed=5)
    rrg_summary = pd.DataFrame(
        [
            {"seed_universe": "Seed A", "rrg_quadrant": "Leading", "rrg_score": 100.0, "rrg_reason": "rrg_seed_improving"},
            {"seed_universe": "Seed B", "rrg_quadrant": "Lagging", "rrg_score": 15.0, "rrg_reason": "rrg_seed_weakening"},
        ]
    )

    candidates = build_portfolio_candidates(ranked_fixture(), membership_fixture(), rrg_summary, config)

    assert candidates["symbol"].tolist() == ["AAA", "BBB"]
    assert "CCC" not in set(candidates["symbol"])
    assert "DDD" not in set(candidates["symbol"])
    assert candidates.loc[candidates["symbol"] == "AAA", "rrg_quadrant"].iloc[0] == "Leading"


def test_allocation_respects_caps_and_cash_rule():
    candidates = pd.DataFrame(
        [
            {**row, "seed_universe": "Seed A", "allocator_score": score, "rrg_quadrant": "Leading", "reason_codes": "score_improved"}
            for row, score in zip(ranked_fixture().head(2).to_dict(orient="records"), [100.0, 80.0])
        ]
    )
    config = PortfolioConfig(selected_seeds=("Seed A",), max_position=0.40, max_seed_exposure=0.70)

    portfolio = allocate_portfolio(candidates, config, "Transition")

    assert portfolio["weight"].sum() == pytest.approx(1.0)
    assert portfolio.loc[portfolio["symbol"] == "Cash", "weight"].iloc[0] >= 0.15 - 1e-9
    assert portfolio.loc[portfolio["symbol"] != "Cash", "weight"].max() <= 0.40 + 1e-9
    assert portfolio.loc[portfolio["seed_universe"] == "Seed A", "weight"].sum() <= 0.70 + 1e-9


def test_cash_target_by_benchmark_regime():
    assert cash_target_for_regime("Bull") == 0.0
    assert cash_target_for_regime("Transition") == 0.15
    assert cash_target_for_regime("Bear") == 0.35


def test_sensitivity_suppresses_small_changes_and_turnover():
    previous = pd.DataFrame(
        [
            {"symbol": "AAA", "weight": 0.10, "name": "AAA", "seed_universe": "Seed A"},
            {"symbol": "Cash", "weight": 0.90, "name": "Cash", "seed_universe": "Cash"},
        ]
    )
    target = pd.DataFrame(
        [
            {"symbol": "AAA", "weight": 0.115, "name": "AAA", "seed_universe": "Seed A"},
            {"symbol": "BBB", "weight": 0.30, "name": "BBB", "seed_universe": "Seed B"},
            {"symbol": "Cash", "weight": 0.585, "name": "Cash", "seed_universe": "Cash"},
        ]
    )

    conservative = PortfolioConfig(selected_seeds=("Seed A", "Seed B"), sensitivity="Conservative")
    changes, suppressed = compare_portfolios(previous, target, conservative)

    assert "AAA" in set(suppressed["symbol"])
    assert "BBB" in set(suppressed["symbol"])
    assert "Cash" in set(changes["symbol"])


def test_replacement_threshold_suppresses_weak_replacement():
    previous = pd.DataFrame(
        [
            {"symbol": "AAA", "weight": 0.20, "allocator_score": 90.0, "seed_universe": "Seed A", "rrg_quadrant": "Leading"},
            {"symbol": "Cash", "weight": 0.80, "allocator_score": 0.0, "seed_universe": "Cash", "rrg_quadrant": ""},
        ]
    )
    target = pd.DataFrame(
        [
            {"symbol": "BBB", "weight": 0.20, "allocator_score": 96.0, "seed_universe": "Seed B", "rrg_quadrant": "Leading"},
            {"symbol": "Cash", "weight": 0.80, "allocator_score": 0.0, "seed_universe": "Cash", "rrg_quadrant": ""},
        ]
    )

    changes, suppressed = compare_portfolios(
        previous,
        target,
        PortfolioConfig(selected_seeds=("Seed A", "Seed B"), sensitivity="Balanced"),
    )

    assert "BBB" in set(suppressed["symbol"])
    assert "replacement_threshold_not_met" in set(suppressed["reason_codes"])
    assert "BBB" not in set(changes.get("symbol", []))


def test_aggressive_replacement_threshold_allows_smaller_replacement():
    previous = pd.DataFrame(
        [
            {"symbol": "AAA", "weight": 0.20, "allocator_score": 90.0, "seed_universe": "Seed A", "rrg_quadrant": "Leading"},
            {"symbol": "Cash", "weight": 0.80, "allocator_score": 0.0, "seed_universe": "Cash", "rrg_quadrant": ""},
        ]
    )
    target = pd.DataFrame(
        [
            {"symbol": "BBB", "weight": 0.20, "allocator_score": 96.0, "seed_universe": "Seed B", "rrg_quadrant": "Leading"},
            {"symbol": "Cash", "weight": 0.80, "allocator_score": 0.0, "seed_universe": "Cash", "rrg_quadrant": ""},
        ]
    )

    changes, suppressed = compare_portfolios(
        previous,
        target,
        PortfolioConfig(selected_seeds=("Seed A", "Seed B"), sensitivity="Aggressive"),
    )

    assert "BBB" in set(changes["symbol"])
    assert "BBB" not in set(suppressed.get("symbol", []))


def test_conservative_requires_rrg_quadrant_confirmation_for_increase():
    previous = pd.DataFrame(
        [
            {"symbol": "AAA", "weight": 0.10, "allocator_score": 90.0, "seed_universe": "Seed A", "rrg_quadrant": "Leading"},
            {"symbol": "Cash", "weight": 0.90, "allocator_score": 0.0, "seed_universe": "Cash", "rrg_quadrant": ""},
        ]
    )
    target = pd.DataFrame(
        [
            {"symbol": "AAA", "weight": 0.15, "allocator_score": 95.0, "seed_universe": "Seed A", "rrg_quadrant": "Leading"},
            {"symbol": "Cash", "weight": 0.85, "allocator_score": 0.0, "seed_universe": "Cash", "rrg_quadrant": ""},
        ]
    )

    _, suppressed = compare_portfolios(
        previous,
        target,
        PortfolioConfig(selected_seeds=("Seed A",), sensitivity="Conservative"),
    )

    assert "AAA" in set(suppressed["symbol"])
    assert "rrg_quadrant_confirmation_required" in set(suppressed["reason_codes"])


def test_portfolio_config_hash_is_seed_order_independent():
    first = PortfolioConfig(selected_seeds=("SMH Universe", "Healthcare Universe"))
    second = PortfolioConfig(selected_seeds=("Healthcare Universe", "SMH Universe"))

    assert canonical_selected_seeds(first.selected_seeds) == "Healthcare Universe,SMH Universe"
    assert portfolio_config_hash(first) == portfolio_config_hash(second)


def test_apply_sensitivity_keeps_previous_weight_for_suppressed_change():
    previous = pd.DataFrame(
        [
            {"symbol": "AAA", "weight": 0.10, "name": "AAA", "seed_universe": "Seed A"},
            {"symbol": "Cash", "weight": 0.90, "name": "Cash", "seed_universe": "Cash"},
        ]
    )
    target = pd.DataFrame(
        [
            {"symbol": "AAA", "weight": 0.115, "name": "AAA", "seed_universe": "Seed A"},
            {"symbol": "Cash", "weight": 0.885, "name": "Cash", "seed_universe": "Cash"},
        ]
    )
    for column in [
        "allocator_score",
        "score",
        "benchmark_adjusted_score",
        "relative_strength_score",
        "larsson_regime",
        "daily_trend",
        "weekly_trend",
        "monthly_trend",
        "drawdown_52w",
        "extension_sma50d",
        "rrg_quadrant",
        "reason_codes",
        "data_warnings",
    ]:
        previous[column] = 0 if "score" in column or column in {"drawdown_52w", "extension_sma50d"} else ""
        target[column] = 0 if "score" in column or column in {"drawdown_52w", "extension_sma50d"} else ""

    adjusted = apply_sensitivity_to_target(previous, target, PortfolioConfig(selected_seeds=("Seed A",), sensitivity="Conservative"))

    assert adjusted.loc[adjusted["symbol"] == "AAA", "weight"].iloc[0] == pytest.approx(0.10)


def test_turnover_suppressed_replacement_keeps_existing_holding():
    previous = pd.DataFrame(
        [
            {"symbol": "AAA", "weight": 0.50, "name": "AAA", "seed_universe": "Seed A"},
            {"symbol": "Cash", "weight": 0.50, "name": "Cash", "seed_universe": "Cash"},
        ]
    )
    target = pd.DataFrame(
        [
            {"symbol": "BBB", "weight": 0.50, "name": "BBB", "seed_universe": "Seed B"},
            {"symbol": "Cash", "weight": 0.50, "name": "Cash", "seed_universe": "Cash"},
        ]
    )
    for column in [
        "allocator_score",
        "score",
        "benchmark_adjusted_score",
        "relative_strength_score",
        "larsson_regime",
        "daily_trend",
        "weekly_trend",
        "monthly_trend",
        "drawdown_52w",
        "extension_sma50d",
        "rrg_quadrant",
        "reason_codes",
        "data_warnings",
    ]:
        previous[column] = 0 if "score" in column or column in {"drawdown_52w", "extension_sma50d"} else ""
        target[column] = 0 if "score" in column or column in {"drawdown_52w", "extension_sma50d"} else ""
    config = PortfolioConfig(selected_seeds=("Seed A", "Seed B"), sensitivity="Conservative")

    changes, suppressed = compare_portfolios(previous, target, config)
    adjusted = apply_sensitivity_to_target(previous, target, config)

    assert "BBB" in set(suppressed["symbol"])
    assert "AAA" in set(suppressed["symbol"])
    assert "paired_addition_suppressed" in set(suppressed["reason_codes"])
    assert adjusted.loc[adjusted["symbol"] == "AAA", "weight"].iloc[0] == pytest.approx(0.50)
    assert adjusted.loc[adjusted["symbol"] == "BBB", "weight"].iloc[0] == pytest.approx(0.0)
    assert adjusted["weight"].sum() == pytest.approx(1.0)


def test_llm_prompt_contains_context_without_api_call():
    config = PortfolioConfig(selected_seeds=("Seed A",), sensitivity="Balanced")
    prompt = build_allocator_prompt(
        pd.DataFrame(),
        pd.DataFrame([{"symbol": "AAA", "weight": 0.1}]),
        pd.DataFrame(),
        pd.DataFrame(),
        "QQQ",
        "Bull",
        config,
    )

    assert '"benchmark": "QQQ"' in prompt
    assert '"sensitivity": "Balanced"' in prompt
    assert "Do not invent tickers" in prompt
