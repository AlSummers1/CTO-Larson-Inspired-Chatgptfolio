from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RankingConfig:
    larsson_weight: float = 15.0
    daily_trend_weight: float = 15.0
    weekly_trend_weight: float = 20.0
    monthly_trend_weight: float = 15.0
    pass_200d_weight: float = 15.0
    pass_200w_weight: float = 10.0
    drawdown_extension_weight: float = 10.0
    fundamental_weight: float = 0.0
    catalyst_weight: float = 0.0
    relative_strength_weight: float = 0.15
    use_relative_strength: bool = False
    exclude_benchmark: bool = True


LARSSON_CREDIT = {
    "Bull": 1.0,
    "Transition": 0.5,
    "Bear": 0.0,
}

TREND_CREDIT = {
    "Up": 1.0,
    "Mixed": 0.5,
    "Down": 0.0,
}


def _finite(value: object) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def _bool_credit(value: object, neutral_if_missing: bool) -> float:
    if value is None or pd.isna(value):
        return 0.5 if neutral_if_missing else 0.0
    return 1.0 if bool(value) else 0.0


def drawdown_extension_credit(row: pd.Series) -> float:
    """Reward constructive pullbacks and penalize severe damage or hot extension."""
    drawdown = _finite(row.get("drawdown_52w"))
    ext50 = _finite(row.get("extension_sma50d"))
    ext200 = _finite(row.get("extension_sma200d"))
    if drawdown is None:
        return 0.5

    credit = 1.0
    if drawdown <= -0.50:
        credit -= 0.50
    elif drawdown <= -0.35:
        credit -= 0.35
    elif drawdown <= -0.20:
        credit -= 0.15

    if ext50 is not None:
        if ext50 >= 0.35:
            credit -= 0.35
        elif ext50 >= 0.20:
            credit -= 0.20
        elif ext50 <= -0.15:
            credit -= 0.10

    if ext200 is not None:
        if ext200 >= 0.80:
            credit -= 0.25
        elif ext200 >= 0.50:
            credit -= 0.10
        elif ext200 < 0:
            credit -= 0.15

    return max(min(credit, 1.0), 0.0)


def _status_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return "Pass" if bool(value) else "Fail"


def rank_universe(signals: pd.DataFrame, config: RankingConfig | None = None) -> pd.DataFrame:
    """Apply the transparent 0-100 ranking model to a signal table."""
    config = config or RankingConfig()
    if signals.empty:
        return signals.copy()

    ranked = signals.copy()
    if config.exclude_benchmark and "is_benchmark" in ranked.columns:
        ranked = ranked[~ranked["is_benchmark"].fillna(False)].copy()

    ranked["larsson_score"] = ranked["larsson_regime"].map(LARSSON_CREDIT).fillna(0.5) * config.larsson_weight
    ranked["daily_trend_score"] = ranked["daily_trend"].map(TREND_CREDIT).fillna(0.5) * config.daily_trend_weight
    ranked["weekly_trend_score"] = ranked["weekly_trend"].map(TREND_CREDIT).fillna(0.5) * config.weekly_trend_weight
    ranked["monthly_trend_score"] = ranked["monthly_trend"].map(TREND_CREDIT).fillna(0.5) * config.monthly_trend_weight
    ranked["pass_200d_score"] = ranked["pass_200d"].map(lambda value: _bool_credit(value, neutral_if_missing=False)) * config.pass_200d_weight
    ranked["pass_200w_score"] = ranked["pass_200w"].map(lambda value: _bool_credit(value, neutral_if_missing=True)) * config.pass_200w_weight
    ranked["drawdown_extension_score"] = ranked.apply(drawdown_extension_credit, axis=1) * config.drawdown_extension_weight

    if "fundamental_score" not in ranked.columns:
        ranked["fundamental_score"] = 0.0
    if "catalyst_score" not in ranked.columns:
        ranked["catalyst_score"] = 0.0

    ranked["fundamental_component"] = pd.to_numeric(ranked["fundamental_score"], errors="coerce").fillna(0.0) * config.fundamental_weight
    ranked["catalyst_component"] = pd.to_numeric(ranked["catalyst_score"], errors="coerce").fillna(0.0) * config.catalyst_weight
    score_columns = [
        "larsson_score",
        "daily_trend_score",
        "weekly_trend_score",
        "monthly_trend_score",
        "pass_200d_score",
        "pass_200w_score",
        "drawdown_extension_score",
        "fundamental_component",
        "catalyst_component",
    ]
    ranked["score"] = ranked[score_columns].sum(axis=1).round(1)
    if "relative_strength_score" not in ranked.columns:
        ranked["relative_strength_score"] = 50.0
    ranked["relative_strength_score"] = pd.to_numeric(
        ranked["relative_strength_score"], errors="coerce"
    ).fillna(50.0)
    base_weight = 1.0 - config.relative_strength_weight
    ranked["benchmark_adjusted_score"] = (
        base_weight * ranked["score"] + config.relative_strength_weight * ranked["relative_strength_score"]
    ).round(1)
    ranked["pass_200d_label"] = ranked["pass_200d"].map(_status_label)
    ranked["pass_200w_label"] = ranked["pass_200w"].map(_status_label)
    sort_score = "benchmark_adjusted_score" if config.use_relative_strength else "score"
    ranked = ranked.sort_values([sort_score, "symbol"], ascending=[False, True]).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked
