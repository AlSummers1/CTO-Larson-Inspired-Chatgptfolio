import pandas as pd
import pytest

from stockpicker.scoring import RankingConfig, drawdown_extension_credit, rank_universe


def test_rank_universe_scores_components_and_neutral_200w():
    signals = pd.DataFrame(
        [
            {
                "symbol": "NVDA",
                "is_benchmark": False,
                "larsson_regime": "Bull",
                "daily_trend": "Up",
                "weekly_trend": "Up",
                "monthly_trend": "Up",
                "pass_200d": True,
                "pass_200w": None,
                "drawdown_52w": -0.10,
                "extension_sma50d": 0.05,
                "extension_sma200d": 0.20,
            }
        ]
    )
    ranked = rank_universe(signals, RankingConfig())
    assert ranked.loc[0, "score"] == pytest.approx(95.0)
    assert ranked.loc[0, "pass_200w_label"] == "n/a"


def test_rank_universe_excludes_benchmark_by_default():
    signals = pd.DataFrame(
        [
            {
                "symbol": "QQQ",
                "is_benchmark": True,
                "larsson_regime": "Bull",
                "daily_trend": "Up",
                "weekly_trend": "Up",
                "monthly_trend": "Up",
                "pass_200d": True,
                "pass_200w": True,
                "drawdown_52w": -0.01,
                "extension_sma50d": 0.01,
                "extension_sma200d": 0.01,
            },
            {
                "symbol": "MSFT",
                "is_benchmark": False,
                "larsson_regime": "Transition",
                "daily_trend": "Mixed",
                "weekly_trend": "Mixed",
                "monthly_trend": "Mixed",
                "pass_200d": False,
                "pass_200w": False,
                "drawdown_52w": -0.30,
                "extension_sma50d": -0.05,
                "extension_sma200d": -0.05,
            },
        ]
    )
    ranked = rank_universe(signals)
    assert ranked["symbol"].tolist() == ["MSFT"]


def test_drawdown_extension_credit_penalizes_damage_and_extension():
    constructive = pd.Series({"drawdown_52w": -0.08, "extension_sma50d": 0.06, "extension_sma200d": 0.30})
    damaged = pd.Series({"drawdown_52w": -0.55, "extension_sma50d": 0.40, "extension_sma200d": 0.90})
    assert drawdown_extension_credit(constructive) == pytest.approx(1.0)
    assert drawdown_extension_credit(damaged) == pytest.approx(0.0)


def base_signal(symbol: str, daily: str, rs_score: float) -> dict[str, object]:
    return {
        "symbol": symbol,
        "is_benchmark": False,
        "larsson_regime": "Bull",
        "daily_trend": daily,
        "weekly_trend": "Up",
        "monthly_trend": "Up",
        "pass_200d": True,
        "pass_200w": True,
        "drawdown_52w": -0.05,
        "extension_sma50d": 0.03,
        "extension_sma200d": 0.10,
        "relative_strength_score": rs_score,
    }


def test_benchmark_adjusted_score_preserves_default_base_sorting():
    signals = pd.DataFrame(
        [
            base_signal("BASE", "Up", 0),
            base_signal("RELATIVE", "Mixed", 100),
        ]
    )

    ranked = rank_universe(signals, RankingConfig(use_relative_strength=False))

    assert ranked["symbol"].tolist() == ["BASE", "RELATIVE"]
    assert ranked.loc[0, "score"] > ranked.loc[1, "score"]
    assert "benchmark_adjusted_score" in ranked.columns


def test_benchmark_adjusted_score_sorts_when_enabled():
    signals = pd.DataFrame(
        [
            base_signal("BASE", "Up", 0),
            base_signal("RELATIVE", "Mixed", 100),
        ]
    )

    ranked = rank_universe(signals, RankingConfig(use_relative_strength=True))

    assert ranked["symbol"].tolist() == ["RELATIVE", "BASE"]
    relative = ranked[ranked["symbol"] == "RELATIVE"].iloc[0]
    assert relative["benchmark_adjusted_score"] == pytest.approx(round(0.85 * relative["score"] + 0.15 * 100, 1))
