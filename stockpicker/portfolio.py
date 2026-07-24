from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


SENSITIVITY_PROFILES = {
    "Conservative": {
        "rrg_weight": 0.10,
        "replacement_threshold": 12.0,
        "min_weight_change": 0.03,
        "max_turnover": 0.20,
        "requires_quadrant_change": True,
    },
    "Balanced": {
        "rrg_weight": 0.20,
        "replacement_threshold": 8.0,
        "min_weight_change": 0.02,
        "max_turnover": 0.35,
        "requires_quadrant_change": False,
    },
    "Aggressive": {
        "rrg_weight": 0.35,
        "replacement_threshold": 5.0,
        "min_weight_change": 0.01,
        "max_turnover": 0.60,
        "requires_quadrant_change": False,
    },
}

PORTFOLIO_COLUMNS = [
    "symbol",
    "name",
    "seed_universe",
    "weight",
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
]


@dataclass(frozen=True)
class PortfolioConfig:
    selected_seeds: tuple[str, ...]
    max_holdings: int = 12
    top_candidates_per_seed: int = 5
    max_position: float = 0.12
    max_seed_exposure: float = 0.45
    min_position: float = 0.02
    score_power: float = 1.8
    use_relative_strength: bool = True
    include_cash_rule: bool = True
    allow_weak_regime_candidates: bool = False
    sensitivity: str = "Balanced"


def sensitivity_profile(name: str) -> dict[str, float | bool]:
    return SENSITIVITY_PROFILES.get(name, SENSITIVITY_PROFILES["Balanced"]).copy()


def canonical_selected_seeds(selected_seeds: tuple[str, ...] | list[str]) -> str:
    return ",".join(sorted(str(seed) for seed in selected_seeds if str(seed).strip()))


def portfolio_config_payload(config: PortfolioConfig) -> dict[str, object]:
    return {
        "selected_seeds": sorted(config.selected_seeds),
        "max_holdings": config.max_holdings,
        "top_candidates_per_seed": config.top_candidates_per_seed,
        "max_position": round(config.max_position, 6),
        "max_seed_exposure": round(config.max_seed_exposure, 6),
        "min_position": round(config.min_position, 6),
        "score_power": round(config.score_power, 6),
        "use_relative_strength": config.use_relative_strength,
        "include_cash_rule": config.include_cash_rule,
        "allow_weak_regime_candidates": config.allow_weak_regime_candidates,
        "sensitivity": config.sensitivity,
    }


def portfolio_config_hash(config: PortfolioConfig) -> str:
    payload = json.dumps(portfolio_config_payload(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def cash_target_for_regime(regime: str, include_cash_rule: bool = True) -> float:
    if not include_cash_rule:
        return 0.0
    if regime == "Bear":
        return 0.35
    if regime == "Transition":
        return 0.15
    return 0.0


def rrg_pressure(quadrant: object, ratio_delta: float = 0.0, momentum_delta: float = 0.0) -> tuple[float, str]:
    quadrant = str(quadrant) if not pd.isna(quadrant) else "n/a"
    base = {
        "Leading": 100.0,
        "Improving": 72.0,
        "Weakening": 42.0,
        "Lagging": 15.0,
    }.get(quadrant, 50.0)
    movement_bonus = max(min((ratio_delta + momentum_delta) * 2.0, 10.0), -10.0)
    score = max(min(base + movement_bonus, 100.0), 0.0)
    if quadrant in {"Leading", "Improving"}:
        reason = "rrg_seed_improving"
    elif quadrant in {"Weakening", "Lagging"}:
        reason = "rrg_seed_weakening"
    else:
        reason = "rrg_seed_neutral"
    return score, reason


def rrg_seed_summary(rrg_history: pd.DataFrame) -> pd.DataFrame:
    columns = ["seed_universe", "rrg_quadrant", "rrg_score", "rrg_reason", "rs_ratio_delta", "rs_momentum_delta"]
    if rrg_history.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for seed_name, group in rrg_history.groupby("seed_name", sort=False):
        group = group.sort_values("date")
        latest = group.iloc[-1]
        previous = group.iloc[-2] if len(group) >= 2 else latest
        ratio_delta = float(latest["rs_ratio"] - previous["rs_ratio"])
        momentum_delta = float(latest["rs_momentum"] - previous["rs_momentum"])
        score, reason = rrg_pressure(latest["quadrant"], ratio_delta, momentum_delta)
        rows.append(
            {
                "seed_universe": seed_name,
                "rrg_quadrant": latest["quadrant"],
                "rrg_score": score,
                "rrg_reason": reason,
                "rs_ratio_delta": ratio_delta,
                "rs_momentum_delta": momentum_delta,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_portfolio_candidates(
    ranked: pd.DataFrame,
    seed_membership: pd.DataFrame,
    rrg_summary: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.DataFrame:
    if ranked.empty or seed_membership.empty:
        return pd.DataFrame()

    selected = list(config.selected_seeds)
    membership = seed_membership[seed_membership["seed_universe"].isin(selected)].copy()
    merged = membership.merge(ranked, on="symbol", how="inner", suffixes=("_seed", ""))
    if merged.empty:
        return merged

    merged["seed_order"] = merged["seed_universe"].map({label: index for index, label in enumerate(selected)})
    merged = merged.sort_values(["symbol", "seed_order"]).drop_duplicates("symbol", keep="first")
    if not config.allow_weak_regime_candidates:
        merged = merged[merged["larsson_regime"] != "Bear"].copy()
    merged = merged[merged["pass_200d"] == True].copy()
    if merged.empty:
        return merged

    rrg = rrg_summary.rename(columns={"rrg_quadrant": "rrg_quadrant"})
    merged = merged.merge(rrg, on="seed_universe", how="left")
    merged["rrg_score"] = pd.to_numeric(merged["rrg_score"], errors="coerce").fillna(50.0)
    merged["rrg_quadrant"] = merged["rrg_quadrant"].fillna("n/a")
    merged["rrg_reason"] = merged["rrg_reason"].fillna("rrg_seed_neutral")

    base_col = "benchmark_adjusted_score" if config.use_relative_strength else "score"
    base_score = pd.to_numeric(merged[base_col], errors="coerce").fillna(0.0)
    profile = sensitivity_profile(config.sensitivity)
    rrg_weight = float(profile["rrg_weight"])
    merged["allocator_score"] = ((1.0 - rrg_weight) * base_score + rrg_weight * merged["rrg_score"]).round(1)
    merged["reason_codes"] = merged.apply(_candidate_reason_codes, axis=1)
    return (
        merged.sort_values(["seed_universe", "allocator_score", "symbol"], ascending=[True, False, True])
        .groupby("seed_universe", as_index=False)
        .head(config.top_candidates_per_seed)
        .sort_values(["allocator_score", "symbol"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _candidate_reason_codes(row: pd.Series) -> str:
    reasons = ["score_improved"]
    rrg_reason = str(row.get("rrg_reason", ""))
    if rrg_reason:
        reasons.append(rrg_reason)
    if row.get("pass_200w") is False:
        reasons.append("failed_200w")
    drawdown = pd.to_numeric(pd.Series([row.get("drawdown_52w")]), errors="coerce").iloc[0]
    if not pd.isna(drawdown) and drawdown <= -0.35:
        reasons.append("large_drawdown")
    return ", ".join(dict.fromkeys(reasons))


def allocate_portfolio(
    candidates: pd.DataFrame,
    config: PortfolioConfig,
    benchmark_regime: str,
) -> pd.DataFrame:
    if candidates.empty:
        return _cash_only(1.0, "no_eligible_candidates")

    cash_weight = cash_target_for_regime(benchmark_regime, config.include_cash_rule)
    equity_target = max(1.0 - cash_weight, 0.0)
    selected = candidates.head(config.max_holdings).copy()
    selected = _score_to_weights(selected, config.score_power, equity_target)
    selected = _enforce_caps(selected, config.max_position, config.max_seed_exposure, equity_target)
    selected = selected[selected["weight"] >= config.min_position].copy()
    if not selected.empty:
        selected = _score_to_weights(selected, config.score_power, equity_target)
        selected = _enforce_caps(selected, config.max_position, config.max_seed_exposure, equity_target)

    allocated = float(selected["weight"].sum()) if not selected.empty else 0.0
    cash_weight = max(1.0 - allocated, cash_weight)
    selected["reason_codes"] = selected["reason_codes"].map(lambda text: _append_reason(text, "cash_rule_applied") if cash_target_for_regime(benchmark_regime, config.include_cash_rule) > 0 else text)
    output = selected[[column for column in PORTFOLIO_COLUMNS if column in selected.columns]].copy()
    cash = _cash_only(cash_weight, "cash_rule_applied" if cash_weight > 0 else "residual_cash")
    if output.empty:
        return cash
    return pd.concat([output, cash], ignore_index=True)[PORTFOLIO_COLUMNS]


def _score_to_weights(frame: pd.DataFrame, score_power: float, target_weight: float) -> pd.DataFrame:
    output = frame.copy()
    scores = pd.to_numeric(output["allocator_score"], errors="coerce").fillna(0.0).clip(lower=1.0)
    powered = scores.pow(score_power)
    total = float(powered.sum())
    output["weight"] = 0.0 if total <= 0 else powered / total * target_weight
    return output


def _enforce_caps(frame: pd.DataFrame, max_position: float, max_seed_exposure: float, target_weight: float) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        return output
    output["weight"] = output["weight"].clip(upper=max_position)
    for _ in range(8):
        seed_totals = output.groupby("seed_universe")["weight"].transform("sum")
        over_seed = seed_totals > max_seed_exposure
        if over_seed.any():
            output.loc[over_seed, "weight"] *= max_seed_exposure / seed_totals[over_seed]
            output.loc[over_seed, "reason_codes"] = output.loc[over_seed, "reason_codes"].map(
                lambda text: _append_reason(text, "seed_cap_applied")
            )
        output.loc[output["weight"] >= max_position, "reason_codes"] = output.loc[
            output["weight"] >= max_position, "reason_codes"
        ].map(lambda text: _append_reason(text, "position_cap_applied"))
        deficit = target_weight - float(output["weight"].sum())
        if deficit <= 0.0001:
            break
        seed_totals = output.groupby("seed_universe")["weight"].transform("sum")
        eligible = (output["weight"] < max_position - 0.0001) & (seed_totals < max_seed_exposure - 0.0001)
        if not eligible.any():
            break
        position_room = max_position - output.loc[eligible, "weight"]
        seed_room = max_seed_exposure - seed_totals[eligible]
        room = pd.concat([position_room.rename("position_room"), seed_room.rename("seed_room")], axis=1).min(axis=1)
        add = room / room.sum() * deficit
        output.loc[eligible, "weight"] += add.clip(upper=room)
    return output


def _append_reason(text: object, reason: str) -> str:
    parts = [part.strip() for part in str(text or "").split(",") if part.strip()]
    parts.append(reason)
    return ", ".join(dict.fromkeys(parts))


def _cash_only(weight: float, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "Cash",
                "name": "Cash",
                "seed_universe": "Cash",
                "weight": max(min(weight, 1.0), 0.0),
                "allocator_score": 0.0,
                "score": 0.0,
                "benchmark_adjusted_score": 0.0,
                "relative_strength_score": 50.0,
                "larsson_regime": "",
                "daily_trend": "",
                "weekly_trend": "",
                "monthly_trend": "",
                "drawdown_52w": 0.0,
                "extension_sma50d": 0.0,
                "rrg_quadrant": "",
                "reason_codes": reason,
                "data_warnings": "",
            }
        ],
        columns=PORTFOLIO_COLUMNS,
    )


def compare_portfolios(previous: pd.DataFrame, target: pd.DataFrame, config: PortfolioConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = sensitivity_profile(config.sensitivity)
    min_change = float(profile["min_weight_change"])
    max_turnover = float(profile["max_turnover"])
    replacement_threshold = float(profile["replacement_threshold"])
    requires_quadrant_change = bool(profile["requires_quadrant_change"])
    prev = _weight_map(previous)
    curr = _weight_map(target)
    previous_scores = _field_map(previous, "allocator_score", 0.0)
    target_scores = _field_map(target, "allocator_score", 0.0)
    previous_seeds = _field_map(previous, "seed_universe", "")
    target_seeds = _field_map(target, "seed_universe", "")
    previous_quadrants_by_seed = _seed_quadrant_map(previous)
    target_quadrants_by_seed = _seed_quadrant_map(target)
    candidates = []
    for symbol in sorted(set(prev) | set(curr)):
        old_weight = prev.get(symbol, 0.0)
        new_weight = curr.get(symbol, 0.0)
        delta = new_weight - old_weight
        if abs(delta) < 1e-9:
            action = "unchanged"
        elif old_weight == 0:
            action = "added"
        elif new_weight == 0:
            action = "removed"
        elif delta > 0:
            action = "increased"
        else:
            action = "reduced"
        row = {"symbol": symbol, "previous_weight": old_weight, "target_weight": new_weight, "delta": delta, "action": action}
        candidates.append(row)

    rows = []
    suppressed = []
    turnover = 0.0
    accepted_positive = 0.0
    suppressed_positive = 0.0
    cash_delta = curr.get("Cash", 0.0) - prev.get("Cash", 0.0)
    removed_candidates = [candidate for candidate in candidates if candidate["symbol"] != "Cash" and candidate["delta"] < 0]
    best_removed_score = max((float(previous_scores.get(candidate["symbol"], 0.0)) for candidate in removed_candidates), default=0.0)

    for row in [candidate for candidate in candidates if candidate["symbol"] != "Cash" and candidate["delta"] > 0]:
        if abs(row["delta"]) < min_change:
            row["reason_codes"] = "below_minimum_change"
            suppressed.append(row)
            continue
        if row["action"] == "added" and removed_candidates:
            target_score = float(target_scores.get(row["symbol"], 0.0))
            if target_score < best_removed_score + replacement_threshold:
                row["reason_codes"] = "replacement_threshold_not_met"
                suppressed.append(row)
                suppressed_positive += row["delta"]
                continue
        target_seed = str(target_seeds.get(row["symbol"], ""))
        current_quadrant = str(target_quadrants_by_seed.get(target_seed, ""))
        previous_quadrant = str(previous_quadrants_by_seed.get(target_seed, ""))
        if (
            requires_quadrant_change
            and target_seed
            and current_quadrant
            and previous_quadrant
            and current_quadrant == previous_quadrant
        ):
            row["reason_codes"] = "rrg_quadrant_confirmation_required"
            suppressed.append(row)
            suppressed_positive += row["delta"]
            continue
        projected_turnover = turnover + row["delta"]
        if projected_turnover > max_turnover:
            row["reason_codes"] = "turnover_limit_applied"
            suppressed.append(row)
            suppressed_positive += row["delta"]
            continue
        turnover = projected_turnover
        accepted_positive += row["delta"]
        row["reason_codes"] = "accepted_change"
        rows.append(row)

    sell_budget = accepted_positive + max(cash_delta, 0.0)
    accepted_negative = 0.0
    for row in [candidate for candidate in candidates if candidate["symbol"] != "Cash" and candidate["delta"] < 0]:
        if abs(row["delta"]) < min_change:
            row["reason_codes"] = "below_minimum_change"
            suppressed.append(row)
            continue
        if suppressed_positive > 0 and accepted_negative + abs(row["delta"]) > sell_budget + 1e-9:
            row["reason_codes"] = "paired_addition_suppressed"
            suppressed.append(row)
            continue
        accepted_negative += abs(row["delta"])
        row["reason_codes"] = "accepted_change"
        rows.append(row)

    for row in [candidate for candidate in candidates if candidate["symbol"] == "Cash" or abs(candidate["delta"]) < 1e-9]:
        if row["action"] != "unchanged" and abs(row["delta"]) < min_change:
            row["reason_codes"] = "below_minimum_change"
            suppressed.append(row)
            continue
        row["reason_codes"] = "accepted_change" if row["action"] != "unchanged" else ""
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(suppressed)


def apply_sensitivity_to_target(previous: pd.DataFrame, target: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    if previous.empty:
        return target.copy()
    changes, suppressed = compare_portfolios(previous, target, config)
    if suppressed.empty:
        return target.copy()
    adjusted = target.copy()
    previous_weights = _weight_map(previous)
    for row in suppressed.itertuples(index=False):
        if row.symbol in set(adjusted["symbol"]):
            adjusted.loc[adjusted["symbol"] == row.symbol, "weight"] = previous_weights.get(row.symbol, 0.0)
        elif previous_weights.get(row.symbol, 0.0) > 0:
            previous_row = previous[previous["symbol"] == row.symbol].head(1).copy()
            if not previous_row.empty:
                adjusted = pd.concat([adjusted, previous_row], ignore_index=True)
    non_cash = adjusted["symbol"] != "Cash"
    non_cash_total = float(adjusted.loc[non_cash, "weight"].sum())
    cash_weight = max(1.0 - non_cash_total, 0.0)
    adjusted = adjusted[adjusted["symbol"] != "Cash"].copy()
    return pd.concat([adjusted, _cash_only(cash_weight, "residual_cash")], ignore_index=True)[PORTFOLIO_COLUMNS]


def _weight_map(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or "symbol" not in frame.columns or "weight" not in frame.columns:
        return {}
    return {
        str(row.symbol): float(row.weight)
        for row in frame[["symbol", "weight"]].itertuples(index=False)
    }


def _field_map(frame: pd.DataFrame, column: str, default: object) -> dict[str, object]:
    if frame.empty or "symbol" not in frame.columns or column not in frame.columns:
        return {}
    return {
        str(row.symbol): getattr(row, column, default)
        for row in frame[["symbol", column]].itertuples(index=False)
    }


def _seed_quadrant_map(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty or "seed_universe" not in frame.columns or "rrg_quadrant" not in frame.columns:
        return {}
    non_cash = frame[frame["seed_universe"] != "Cash"].copy()
    if non_cash.empty:
        return {}
    return (
        non_cash.dropna(subset=["seed_universe"])
        .drop_duplicates("seed_universe", keep="first")
        .set_index("seed_universe")["rrg_quadrant"]
        .astype(str)
        .to_dict()
    )


def build_allocator_prompt(
    previous: pd.DataFrame,
    target: pd.DataFrame,
    changes: pd.DataFrame,
    suppressed: pd.DataFrame,
    benchmark: str,
    benchmark_regime: str,
    config: PortfolioConfig,
) -> str:
    payload = {
        "instruction": "Explain this deterministic model portfolio. Do not invent tickers. Do not suggest trades outside the candidate list.",
        "benchmark": benchmark,
        "benchmark_regime": benchmark_regime,
        "sensitivity": config.sensitivity,
        "selected_seeds": list(config.selected_seeds),
        "previous_allocation": _compact_records(previous),
        "target_allocation": _compact_records(target),
        "accepted_changes": _compact_records(changes),
        "suppressed_changes": _compact_records(suppressed),
    }
    return json.dumps(payload, indent=2, default=str)


def portfolio_snapshot_metadata(
    benchmark: str,
    benchmark_regime: str,
    config: PortfolioConfig,
) -> dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "benchmark": benchmark,
        "benchmark_regime": benchmark_regime,
        "selected_seeds": canonical_selected_seeds(config.selected_seeds),
        "sensitivity": config.sensitivity,
        "config_hash": portfolio_config_hash(config),
        "config_json": json.dumps(portfolio_config_payload(config), sort_keys=True, default=str),
    }


def _compact_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    return frame.round(4).to_dict(orient="records")
