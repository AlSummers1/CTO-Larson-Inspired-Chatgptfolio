from __future__ import annotations

from pathlib import Path
from datetime import date
import json
from time import perf_counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stockpicker.data import (
    approve_paper_rebalance,
    cached_symbol_count,
    DEFAULT_DB_PATH,
    execute_pending_paper_rebalances,
    fetch_price_history,
    latest_cache_timestamp,
    load_fetch_log,
    load_latest_portfolio_snapshot,
    load_matching_portfolio_snapshot,
    load_portfolio_rebalance_log,
    load_lab_runs,
    load_paper_experiments,
    load_paper_fills,
    load_paper_portfolio_dashboard,
    load_source_health,
    paper_valuation_symbols,
    rebuild_paper_nav,
    save_lab_research_snapshot,
    save_lab_research_details,
    save_lab_run,
    save_portfolio_snapshot,
)
from stockpicker.portfolio import (
    PortfolioConfig,
    SENSITIVITY_PROFILES,
    allocate_portfolio,
    apply_sensitivity_to_target,
    build_portfolio_candidates,
    cash_target_for_regime,
    compare_portfolios,
    canonical_selected_seeds,
    portfolio_snapshot_metadata,
    portfolio_config_hash,
    rrg_seed_summary,
)
from stockpicker.scoring import RankingConfig, rank_universe
from stockpicker.signals import (
    RRGConfig,
    SignalConfig,
    compute_rrg_basket_history,
    compute_rrg_history,
    compute_signal_table,
    drawdown_history,
    equal_weight_basket_history,
    larsson_regime_history,
    price_history_with_indicators,
    weekly_history_with_indicators,
)
from stockpicker.universe import UNIVERSE_COLUMNS, load_universe, parse_ticker_text
from stockpicker.lab import (
    DEFAULT_MODEL,
    MODEL_OPTIONS,
    LabConfig,
    OpenAIPortfolioModel,
    SEVERITY_COLORS,
    SEVERITY_ORDER,
    config_payload as lab_config_payload,
    compact_model_packet,
    credential_status,
    decision_hash,
    decision_schema_for_packet,
    evidence_health,
    paper_orders,
    optimize_decision,
    packet_context_metadata,
    preflight_packet,
    prompt_for_packet,
    research_packet,
    severity_counts,
    validate_decision,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MANUAL_ONLY_UNIVERSE = "Manual only"
DEFAULT_SEED_UNIVERSE = "SMH Universe"
SEED_UNIVERSES: dict[str, Path | None] = {
    DEFAULT_SEED_UNIVERSE: PROJECT_ROOT / "data" / "universe_smh.csv",
    "Healthcare Universe": PROJECT_ROOT / "data" / "universe_healthcare.csv",
    "GLP-1 and Obesity Universe": PROJECT_ROOT / "data" / "universe_glp1_obesity.csv",
    "Defence and Space Universe": PROJECT_ROOT / "data" / "universe_defence_space.csv",
    "Data Center Infrastructure Universe": PROJECT_ROOT / "data" / "universe_data_center_infrastructure.csv",
    "Power Grid and Electrification Universe": PROJECT_ROOT / "data" / "universe_power_grid_electrification.csv",
    "Utilities Universe": PROJECT_ROOT / "data" / "universe_utilities.csv",
    "Clean Energy Universe": PROJECT_ROOT / "data" / "universe_clean_energy.csv",
    "Nuclear Energy Universe": PROJECT_ROOT / "data" / "universe_nuclear_energy.csv",
    "Rare Earths and Strategic Materials Universe": PROJECT_ROOT / "data" / "universe_strategic_materials.csv",
    "Crypto Universe": PROJECT_ROOT / "data" / "universe_crypto.csv",
    "Quantum Universe": PROJECT_ROOT / "data" / "universe_quantum.csv",
    "Cybersecurity Universe": PROJECT_ROOT / "data" / "universe_cybersecurity.csv",
    "Magnificent Seven Universe": PROJECT_ROOT / "data" / "universe_magnificent_seven.csv",
    "Space Exploration Universe": PROJECT_ROOT / "data" / "universe_space_exploration.csv",
    MANUAL_ONLY_UNIVERSE: None,
}
SEED_UNIVERSE_DISPLAY_NAMES = {
    "SMH Universe": "Semiconductors (SMH)",
    "Healthcare Universe": "Healthcare",
    "GLP-1 and Obesity Universe": "GLP-1 & Obesity",
    "Defence and Space Universe": "Defence & Space",
    "Data Center Infrastructure Universe": "Data Center Infra",
    "Power Grid and Electrification Universe": "Power Grid",
    "Utilities Universe": "Utilities",
    "Clean Energy Universe": "Clean Energy",
    "Nuclear Energy Universe": "Nuclear Energy",
    "Rare Earths and Strategic Materials Universe": "Rare Earths & Materials",
    "Crypto Universe": "Crypto",
    "Quantum Universe": "Quantum",
    "Cybersecurity Universe": "Cybersecurity",
    "Magnificent Seven Universe": "Mag7",
    "Space Exploration Universe": "Space Exploration",
}
LEGACY_SEED_UNIVERSE_NAMES = {
    "Magnificent Seven + SpaceX Universe": "Magnificent Seven Universe",
}
RRG_COLORS = {
    "Leading": "#16a34a",
    "Weakening": "#f59e0b",
    "Lagging": "#dc2626",
    "Improving": "#2563eb",
}
RRG_METADATA_COLUMNS = [
    "symbol",
    "name",
    "theme",
    "score",
    "benchmark_adjusted_score",
    "relative_strength_score",
    "rs_3m",
    "rs_6m",
    "rs_12m",
]
DEFAULT_RRG_SEED_UNIVERSES = [
    "SMH Universe",
    "Healthcare Universe",
    "Defence and Space Universe",
    "Cybersecurity Universe",
]
DEFAULT_PORTFOLIO_SEED_UNIVERSES = DEFAULT_RRG_SEED_UNIVERSES
VIEW_OPTIONS = ["Rankings", "Detail", "Benchmark", "RRG", "Theme Breadth", "Portfolio", "Chatgptfolio Lab", "Paper Portfolio", "System Health", "Data Audit"]
THEME_BREADTH_SMA_DAYS = 20
THEME_BREADTH_SMOOTHING_WEEKS = 4
THEME_BREADTH_ROC_DAYS = 50
THEME_BREADTH_MIN_COVERAGE = 0.75
THEME_BREADTH_PERCENT_COLUMNS = (
    "breadth",
    "breadth_4w_average",
    "breadth_change",
    "roc_50d",
    "coverage",
)
RRG_CONTROL_HELP = {
    "tail_weeks": "How many recent weekly RRG points to draw as each ticker's rotation trail.",
    "ratio_smoothing": "Weeks used to smooth the stock-vs-benchmark relative trend. Higher values make the RS Ratio less jumpy.",
    "momentum_smoothing": "Weeks used to smooth relative momentum. Higher values make the vertical RRG movement steadier.",
}


st.set_page_config(page_title="CTO-Style Stock Ranking", layout="wide")


def record_timing(name: str, started_at: float, cache_state: str = "not cached", details: str = "") -> float:
    """Keep a local, non-secret performance trace for the System Health view."""
    elapsed_ms = round((perf_counter() - started_at) * 1000, 1)
    st.session_state.setdefault("performance_diagnostics", {})[name] = {
        "operation": name,
        "duration_ms": elapsed_ms,
        "cache_state": cache_state,
        "details": details,
    }
    return elapsed_ms


def record_duration(name: str, duration_ms: float, cache_state: str = "not cached", details: str = "") -> None:
    st.session_state.setdefault("performance_diagnostics", {})[name] = {
        "operation": name,
        "duration_ms": round(float(duration_ms), 1),
        "cache_state": cache_state,
        "details": details,
    }


def session_cache_state(name: str, signature: object) -> str:
    """Describe whether this browser session reused the same cached-call key."""
    key = f"_performance_key_{name}"
    reused = st.session_state.get(key) == signature
    st.session_state[key] = signature
    return "session cache key reused" if reused else "new cache key"


@st.cache_data
def cached_universe(path: str, file_mtime_ns: int) -> pd.DataFrame:
    return load_universe(Path(path))


@st.cache_data(show_spinner=False)
def cached_prices(
    symbols: tuple[str, ...],
    refresh_key: int,
    refresh: bool,
    years: int,
    db_path: str,
) -> pd.DataFrame:
    return fetch_price_history(
        list(symbols),
        refresh=refresh,
        years=years,
        db_path=Path(db_path),
        download_missing=refresh,
    )


@st.cache_data(show_spinner=False)
def cached_signal_table(prices: pd.DataFrame, benchmark: str, larsson_history_weeks: int) -> pd.DataFrame:
    return compute_signal_table(
        prices,
        benchmark=benchmark,
        config=SignalConfig(larsson_history_weeks=larsson_history_weeks),
    )


@st.cache_data(show_spinner=False)
def cached_rrg_history(
    prices: pd.DataFrame,
    symbols: tuple[str, ...],
    benchmark: str,
    ratio_window: int,
    momentum_window: int,
    tail_weeks: int,
) -> pd.DataFrame:
    return compute_rrg_history(
        prices,
        list(symbols),
        benchmark,
        RRGConfig(
            ratio_window_weeks=ratio_window,
            momentum_window_weeks=momentum_window,
            tail_weeks=tail_weeks,
        ),
    )


@st.cache_data(show_spinner=False)
def cached_rrg_basket_history(
    prices: pd.DataFrame,
    seed_universes: dict[str, list[str]],
    benchmark: str,
    ratio_window: int,
    momentum_window: int,
    tail_weeks: int,
) -> pd.DataFrame:
    normalized_seed_map = {
        seed_name: list(dict.fromkeys(symbol.strip().upper() for symbol in symbols if str(symbol).strip()))
        for seed_name, symbols in sorted(seed_universes.items())
    }
    return compute_rrg_basket_history(
        prices,
        normalized_seed_map,
        benchmark,
        RRGConfig(
            ratio_window_weeks=ratio_window,
            momentum_window_weeks=momentum_window,
            tail_weeks=tail_weeks,
        ),
    )


@st.cache_data(show_spinner=False)
def cached_theme_breadth_history(
    prices: pd.DataFrame,
    symbols: tuple[str, ...],
    sma_days: int = THEME_BREADTH_SMA_DAYS,
    smoothing_weeks: int = THEME_BREADTH_SMOOTHING_WEEKS,
    roc_days: int = THEME_BREADTH_ROC_DAYS,
) -> pd.DataFrame:
    """Return weekly 20D breadth and daily-rebalanced equal-weight ROC from local prices."""
    columns = [
        "date",
        "breadth",
        "breadth_4w_average",
        "breadth_change",
        "eligible_count",
        "universe_size",
        "coverage",
        "theme_index",
        "roc_50d",
        "return_coverage",
        "roc_min_coverage",
    ]
    requested = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
    if not requested or prices.empty or not {"symbol", "date", "close"}.issubset(prices.columns):
        return pd.DataFrame(columns=columns)

    frame = prices.loc[prices["symbol"].astype(str).str.upper().isin(requested), ["date", "symbol", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "symbol", "close"])
    frame = frame[frame["close"].gt(0)].drop_duplicates(["date", "symbol"], keep="last")
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame = frame.sort_values(["symbol", "date"])
    frame["average"] = frame.groupby("symbol", sort=False)["close"].transform(
        lambda values: values.rolling(window=sma_days, min_periods=sma_days).mean()
    )
    close = frame.pivot(index="date", columns="symbol", values="close").sort_index().reindex(columns=requested)
    average = frame.pivot(index="date", columns="symbol", values="average").sort_index().reindex(columns=requested)
    eligible = close.notna() & average.notna()
    above_average = (close > average).where(eligible)
    weekly_eligible = eligible.resample("W-FRI").last().fillna(False)
    # Resampler.last() skips nulls by default. Without replacing them first, a
    # constituent missing on the final weekly observation can retain an earlier
    # True value in the numerator while being absent from the eligible
    # denominator, producing impossible breadth readings above 100%.
    weekly_above = above_average.eq(True).resample("W-FRI").last() & weekly_eligible
    weekly = pd.DataFrame(index=weekly_eligible.index)
    weekly["eligible_count"] = weekly_eligible.sum(axis=1).astype(int)
    weekly["universe_size"] = len(requested)
    weekly["coverage"] = weekly["eligible_count"] / weekly["universe_size"]
    weekly["breadth"] = weekly_above.sum(axis=1, skipna=True) / weekly["eligible_count"].replace(0, float("nan"))
    weekly["breadth_4w_average"] = weekly["breadth"].rolling(smoothing_weeks, min_periods=smoothing_weeks).mean()
    weekly["breadth_change"] = weekly["breadth"].diff()

    returns = close.pct_change(fill_method=None).replace([float("inf"), float("-inf")], float("nan"))
    daily_return = returns.mean(axis=1, skipna=True)
    return_coverage = returns.notna().sum(axis=1) / len(requested)
    theme_index = (1.0 + daily_return.fillna(0.0)).cumprod()
    daily = pd.DataFrame(
        {
            "theme_index": theme_index,
            "roc_50d": theme_index.pct_change(roc_days, fill_method=None),
            "return_coverage": return_coverage,
            "roc_min_coverage": return_coverage.rolling(roc_days, min_periods=roc_days).min(),
        }
    )
    weekly = weekly.join(daily.resample("W-FRI").last(), how="left")
    weekly = weekly[weekly["eligible_count"].gt(0)].reset_index(names="date")
    return weekly[columns]


def empty_universe() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIVERSE_COLUMNS)


def seed_universe_options() -> list[str]:
    return list(SEED_UNIVERSES)


def seed_universe_display_name(seed_label: str) -> str:
    """Keep the compact sidebar selector readable without changing stored labels."""
    return SEED_UNIVERSE_DISPLAY_NAMES.get(seed_label, seed_label)


def seed_universe_label_after_migration(current_label: object, options: list[str]) -> str:
    """Migrate a previously selected seed to its current, available label."""
    candidate = LEGACY_SEED_UNIVERSE_NAMES.get(str(current_label), str(current_label))
    return candidate if candidate in options else DEFAULT_SEED_UNIVERSE


def benchmark_regime_cash_rule_status(regime: str, enabled: bool) -> str:
    """Describe the exact regime-driven cash reserve applied to the allocator."""
    reserve = cash_target_for_regime(regime, enabled)
    if not enabled:
        return f"Regime cash rule is off — {regime} benchmark reserve: 0%."
    return f"Regime cash rule is on — {regime} benchmark reserve: {reserve:.0%}."


def seed_rrg_universe_options() -> list[str]:
    return [label for label, path in SEED_UNIVERSES.items() if path is not None]


def default_rrg_seed_universes() -> list[str]:
    return seed_rrg_universe_options()


def rrg_seed_universes_after_default_migration(current_labels: object, options: list[str]) -> list[str]:
    """Move sessions with the old curated default to the current all-seeds default."""
    if not isinstance(current_labels, list):
        return options.copy()
    valid_current = [label for label in current_labels if label in options]
    legacy_default = [label for label in DEFAULT_RRG_SEED_UNIVERSES if label in options]
    if valid_current == legacy_default and set(valid_current) != set(options):
        return options.copy()
    return valid_current


def default_portfolio_seed_universes() -> list[str]:
    return seed_rrg_universe_options()


def portfolio_seed_universes_after_default_migration(current_labels: object, options: list[str]) -> list[str]:
    """Expand the old four-seed default while preserving custom selections."""
    if not isinstance(current_labels, list):
        return options.copy()
    valid_current = [label for label in current_labels if label in options]
    legacy_default = [label for label in DEFAULT_PORTFOLIO_SEED_UNIVERSES if label in options]
    if valid_current == legacy_default and set(valid_current) != set(options):
        return options.copy()
    return valid_current


def select_seed_universe(seed_label: str, seed_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if seed_label == MANUAL_ONLY_UNIVERSE:
        return empty_universe()
    return seed_frames[seed_label].copy()


def build_active_universe(seed: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if not symbols:
        return empty_universe()

    seed_by_symbol = seed.drop_duplicates("symbol", keep="first").set_index("symbol")
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        if symbol in seed_by_symbol.index:
            row = seed_by_symbol.loc[symbol].to_dict()
            row["symbol"] = symbol
        else:
            row = {
                "symbol": symbol,
                "name": "",
                "theme": "Manual",
                "fundamental_score": 0.0,
                "catalyst_score": 0.0,
                "notes": "",
            }
        rows.append(row)
    return pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)


def seed_theme_options(universe: pd.DataFrame) -> list[str]:
    if "theme" not in universe.columns:
        return []
    return sorted(theme for theme in universe["theme"].dropna().unique() if theme)


def seed_symbols_for_labels(seed_frames: dict[str, pd.DataFrame], labels: list[str]) -> list[str]:
    symbols: list[str] = []
    for label in labels:
        seed = seed_frames.get(label)
        if seed is None or seed.empty or "symbol" not in seed.columns:
            continue
        symbols.extend(seed["symbol"].astype(str).tolist())
    return list(dict.fromkeys(symbols))


def seed_universe_symbol_map(seed_frames: dict[str, pd.DataFrame], labels: list[str]) -> dict[str, list[str]]:
    return {
        label: seed_frames[label]["symbol"].astype(str).tolist()
        for label in labels
        if label in seed_frames
    }


def selected_rrg_seed_labels_for_fetch(rrg_view: str, selected_labels: object) -> list[str]:
    if rrg_view != "Seed universes":
        return []
    labels = selected_labels if isinstance(selected_labels, list) else default_rrg_seed_universes()
    valid_options = set(seed_rrg_universe_options())
    return [label for label in labels if label in valid_options]


def selected_portfolio_seed_labels_for_fetch(selected_labels: object, portfolio_data_enabled: bool) -> list[str]:
    if not portfolio_data_enabled:
        return []
    labels = selected_labels if isinstance(selected_labels, list) else default_portfolio_seed_universes()
    valid_options = set(seed_rrg_universe_options())
    return [label for label in labels if label in valid_options]


def prices_for_symbols(prices: pd.DataFrame, symbols: set[str]) -> pd.DataFrame:
    if prices.empty or "symbol" not in prices.columns:
        return prices.copy()
    return prices[prices["symbol"].isin(symbols)].copy()


def seed_membership_frame(seed_frames: dict[str, pd.DataFrame], labels: list[str]) -> pd.DataFrame:
    frames = []
    for label in labels:
        seed = seed_frames.get(label)
        if seed is None or seed.empty:
            continue
        frame = seed.copy()
        frame["seed_universe"] = label
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*UNIVERSE_COLUMNS, "seed_universe"])
    return pd.concat(frames, ignore_index=True)


def all_seed_universe(seed_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one row per ticker while retaining every seed membership."""
    membership = seed_membership_frame(seed_frames, list(seed_frames))
    if membership.empty:
        return empty_universe().assign(seed_memberships=pd.Series(dtype=str))
    membership["symbol"] = membership["symbol"].astype(str).str.upper()
    rows = []
    for symbol, group in membership.groupby("symbol", sort=True):
        first = group.iloc[0].to_dict()
        first["symbol"] = symbol
        first["seed_memberships"] = " · ".join(sorted(group["seed_universe"].dropna().unique()))
        first["theme"] = first["seed_memberships"]
        rows.append(first)
    return pd.DataFrame(rows)[[*UNIVERSE_COLUMNS, "seed_memberships"]]


def theme_breadth_confirmation(summary: pd.Series | dict[str, object]) -> str:
    """Classify agreement between participation and 50-session theme momentum."""
    getter = summary.get if hasattr(summary, "get") else lambda _key, default=None: default
    breadth = pd.to_numeric(pd.Series([getter("breadth", None)]), errors="coerce").iloc[0]
    average = pd.to_numeric(pd.Series([getter("breadth_4w_average", None)]), errors="coerce").iloc[0]
    roc = pd.to_numeric(pd.Series([getter("roc_50d", None)]), errors="coerce").iloc[0]
    coverage = pd.to_numeric(pd.Series([getter("coverage", None)]), errors="coerce").iloc[0]
    roc_coverage = pd.to_numeric(pd.Series([getter("roc_min_coverage", None)]), errors="coerce").iloc[0]
    if (
        pd.isna(breadth)
        or pd.isna(average)
        or pd.isna(roc)
        or pd.isna(coverage)
        or pd.isna(roc_coverage)
        or coverage < THEME_BREADTH_MIN_COVERAGE
        or roc_coverage < THEME_BREADTH_MIN_COVERAGE
    ):
        return "Unresolved"
    if breadth > average and roc > 0:
        return "Confirmed advance"
    if breadth < average and roc < 0:
        return "Confirmed deterioration"
    return "Mixed"


def latest_theme_breadth_summary(
    history: pd.DataFrame,
    seed_name: str,
    universe_size: int | None = None,
) -> pd.DataFrame:
    """Return the latest transparent breadth/ROC record for one built-in seed universe."""
    columns = [
        "seed_universe",
        "date",
        "breadth",
        "breadth_4w_average",
        "breadth_change",
        "roc_50d",
        "coverage",
        "eligible_count",
        "universe_size",
        "missing_or_short_history",
        "confirmation",
    ]
    if history.empty:
        return pd.DataFrame(
            [
                {
                    "seed_universe": seed_name,
                    "universe_size": int(universe_size or 0),
                    "missing_or_short_history": int(universe_size or 0),
                    "confirmation": "Unresolved",
                }
            ],
            columns=columns,
        )
    latest = history.sort_values("date").tail(1).copy()
    latest.insert(0, "seed_universe", seed_name)
    latest["missing_or_short_history"] = latest["universe_size"] - latest["eligible_count"]
    latest["confirmation"] = latest.apply(theme_breadth_confirmation, axis=1)
    return latest[[column for column in columns if column in latest.columns]]


def theme_breadth_packet_rows(
    prices: pd.DataFrame,
    seed_frames: dict[str, pd.DataFrame],
    as_of_date: date,
) -> list[dict[str, object]]:
    """Build latest point-in-time breadth snapshots for every fixed seed universe."""
    cutoff = pd.Timestamp(as_of_date).normalize()
    snapshots: list[dict[str, object]] = []
    for label in seed_rrg_universe_options():
        theme_symbols = seed_symbols_for_labels(seed_frames, [label])
        theme_prices = prices_for_symbols(prices, set(theme_symbols))
        if not theme_prices.empty and "date" in theme_prices:
            observed_dates = pd.to_datetime(theme_prices["date"], errors="coerce").dt.normalize()
            theme_prices = theme_prices.loc[observed_dates.le(cutoff)].copy()
            observed_dates = observed_dates.loc[observed_dates.le(cutoff)].dropna()
            observation_date = observed_dates.max().date().isoformat() if not observed_dates.empty else None
        else:
            observation_date = None
        history = cached_theme_breadth_history(theme_prices, tuple(theme_symbols))
        summary = latest_theme_breadth_summary(history, label, len(theme_symbols)).iloc[0].to_dict()
        summary["observation_date"] = observation_date
        snapshots.append(summary)
    return snapshots


def theme_breadth_chart(history: pd.DataFrame, seed_name: str) -> go.Figure:
    """Draw weekly percentage-above-20D breadth with its four-week average."""
    fig = go.Figure()
    if history.empty:
        fig.update_layout(title=f"{seed_universe_display_name(seed_name)} breadth", height=360)
        return fig
    fig.add_trace(go.Scatter(x=history["date"], y=history["breadth"], mode="lines+markers", name="% above 20D"))
    fig.add_trace(go.Scatter(x=history["date"], y=history["breadth_4w_average"], mode="lines", name="4W average"))
    fig.update_layout(
        title=f"{seed_universe_display_name(seed_name)}: % above 20D SMA",
        yaxis_title="Participation",
        yaxis={"tickformat": ".0%", "range": [0, 1]},
        xaxis_title="",
        height=360,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def theme_roc_chart(history: pd.DataFrame, seed_name: str) -> go.Figure:
    """Draw 50-session ROC of the locally constructed equal-weight theme index."""
    fig = go.Figure()
    if history.empty:
        fig.update_layout(title=f"{seed_universe_display_name(seed_name)} 50D ROC", height=360)
        return fig
    fig.add_trace(go.Scatter(x=history["date"], y=history["roc_50d"], mode="lines", name="50D ROC"))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(
        title=f"{seed_universe_display_name(seed_name)}: equal-weight theme index 50D ROC",
        yaxis_title="Rate of change",
        yaxis={"tickformat": ".0%"},
        xaxis_title="",
        height=360,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def severity_dot_strip(items: list[dict[str, object]], label: str) -> None:
    counts = severity_counts(items)
    dots = " ".join(f"<span style='color:{SEVERITY_COLORS[level]};font-size:1.35rem'>●</span>" * counts[level] for level in SEVERITY_ORDER)
    summary = " · ".join(f"{counts[level]} {level}" for level in SEVERITY_ORDER if counts[level]) or "No findings"
    st.markdown(f"**{label}:** {dots}<br><small>{summary}</small>", unsafe_allow_html=True)


def seed_text_mode(seed_label: str, seed: pd.DataFrame) -> str:
    symbols = ",".join(seed["symbol"].astype(str).tolist()) if "symbol" in seed.columns else ""
    return f"{seed_label}:{symbols}"


def pct(value: object, decimals: int = 1) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "n/a"
    return f"{float(number):.{decimals}%}"


def money(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "n/a"
    return f"${float(number):,.2f}"


def normalize_paper_series(series: pd.DataFrame, value_column: str = "nav", start_value: float = 100_000.0) -> pd.DataFrame:
    """Normalize one experiment or benchmark series to its own nominal start."""
    if series.empty or value_column not in series:
        return pd.DataFrame(columns=[*series.columns, "normalized_nav"])
    normalized = series.dropna(subset=[value_column]).sort_values("date").copy()
    if normalized.empty or float(normalized.iloc[0][value_column]) == 0:
        normalized["normalized_nav"] = float("nan")
        return normalized
    normalized["normalized_nav"] = start_value * normalized[value_column] / float(normalized.iloc[0][value_column])
    return normalized


SHARED_MARKET_BENCHMARKS = ("SPY", "QQQ", "GLD", "BTC-USD")
RRG_REGIME_ORDER = ("Leading", "Improving", "Weakening", "Lagging", "Unresolved")
RRG_STRENGTH_QUADRANTS = {"Leading", "Improving"}
RRG_WEAKNESS_QUADRANTS = {"Weakening", "Lagging"}


def paper_chart_trace_visibility(symbol: object) -> bool | str:
    """Start broad market references hidden while keeping run-specific traces visible."""
    return "legendonly" if str(symbol).upper() in SHARED_MARKET_BENCHMARKS else True


def paper_experiment_display_name(experiment: pd.Series | dict[str, object]) -> str:
    """Return a descriptive UI label while leaving the immutable run name intact."""
    getter = experiment.get if hasattr(experiment, "get") else lambda _key, default=None: default
    base_name = str(getter("name", "Experiment"))
    sensitivity = str(getter("sensitivity", "Balanced") or "Balanced")
    basket_description = str(getter("basket_description", "Long selected equities") or "Long selected equities")
    return f"{base_name} — {sensitivity} — {basket_description}"


def paper_comparison_benchmark_anchor(experiments: pd.DataFrame) -> int | None:
    """Return the earliest filled selected experiment for shared market references."""
    if experiments.empty or not {"portfolio_id", "fill_date"}.issubset(experiments.columns):
        return None
    filled = experiments.copy()
    filled["fill_date"] = pd.to_datetime(filled["fill_date"], errors="coerce")
    filled = filled.dropna(subset=["fill_date"]).sort_values(["fill_date", "portfolio_id"])
    return int(filled.iloc[0]["portfolio_id"]) if not filled.empty else None


def shared_market_benchmark_series(benchmarks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Normalize the one shared market-reference set; TECHNICAL stays run-specific."""
    if benchmarks.empty or not {"symbol", "date", "nav"}.issubset(benchmarks.columns):
        return {}
    output: dict[str, pd.DataFrame] = {}
    normalized_symbols = benchmarks["symbol"].astype(str).str.upper()
    for symbol in SHARED_MARKET_BENCHMARKS:
        normalized = normalize_paper_series(benchmarks.loc[normalized_symbols == symbol])
        if not normalized.empty:
            output[symbol] = normalized
    return output


def paper_comparison_start_warning(experiments: pd.DataFrame) -> str | None:
    """Explain when selected experiments do not share a comparable start date."""
    if experiments.empty or "fill_date" not in experiments:
        return None
    starts = pd.to_datetime(experiments["fill_date"], errors="coerce").dropna().dt.date.unique()
    if len(starts) > 1:
        return "Selected experiments have different start dates. Each series is normalized to $100,000 at its own start, so the comparison is not directly like-for-like."
    return None


def paper_holding_display_frame(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    seed_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build ticker/name display rows with local cached valuation for filled paper holdings."""
    columns = ["symbol", "name", "label", "shares", "average_cost", "latest_close", "value", "weight"]
    if positions.empty or "symbol" not in positions:
        return pd.DataFrame(columns=columns)

    holdings = positions.copy()
    holdings["symbol"] = holdings["symbol"].astype(str).str.strip().str.upper()
    holdings["shares"] = pd.to_numeric(holdings.get("shares"), errors="coerce").fillna(0.0)
    holdings["average_cost"] = pd.to_numeric(holdings.get("average_cost"), errors="coerce")
    holdings = holdings[holdings["symbol"].ne("") & holdings["shares"].gt(0)].copy()
    if holdings.empty:
        return pd.DataFrame(columns=columns)

    if prices.empty or not {"symbol", "date", "close"}.issubset(prices.columns):
        latest = pd.DataFrame(columns=["symbol", "latest_close"])
    else:
        latest = prices.copy()
        latest["symbol"] = latest["symbol"].astype(str).str.strip().str.upper()
        latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
        latest["close"] = pd.to_numeric(latest["close"], errors="coerce")
        latest = (
            latest[latest["symbol"].isin(holdings["symbol"])]
            .dropna(subset=["date", "close"])
            .sort_values(["symbol", "date"])
            .groupby("symbol", as_index=False)
            .tail(1)[["symbol", "close"]]
            .rename(columns={"close": "latest_close"})
        )

    if seed_metadata.empty or not {"symbol", "name"}.issubset(seed_metadata.columns):
        names = pd.DataFrame(columns=["symbol", "name"])
    else:
        names = seed_metadata[["symbol", "name"]].copy()
        names["symbol"] = names["symbol"].astype(str).str.strip().str.upper()
        names["name"] = names["name"].fillna("").astype(str).str.strip()
        names = names.drop_duplicates("symbol", keep="first")

    display = holdings.merge(latest, on="symbol", how="left").merge(names, on="symbol", how="left")
    display["name"] = display["name"].fillna("").astype(str).str.strip()
    display.loc[display["name"].eq(""), "name"] = display.loc[display["name"].eq(""), "symbol"]
    display["valuation_price"] = display["latest_close"].where(display["latest_close"].notna(), display["average_cost"])
    display["value"] = display["shares"] * display["valuation_price"]
    display["value"] = pd.to_numeric(display["value"], errors="coerce").fillna(0.0)
    total_value = float(display["value"].sum())
    display["weight"] = display["value"] / total_value if total_value > 0 else 0.0
    display["label"] = [
        symbol if name == symbol else f"{symbol} — {name}"
        for symbol, name in zip(display["symbol"], display["name"])
    ]
    return display[columns].sort_values("value", ascending=False).reset_index(drop=True)


def paper_value_pie_chart(values: pd.DataFrame, label_column: str, value_column: str, title: str) -> go.Figure:
    """Render a donut chart from already-valued paper holdings or exposures."""
    fig = go.Figure()
    if values.empty or label_column not in values.columns or value_column not in values.columns:
        fig.update_layout(title=title, height=360)
        return fig
    chart_values = values.copy()
    chart_values[value_column] = pd.to_numeric(chart_values[value_column], errors="coerce").fillna(0.0)
    if float(chart_values[value_column].sum()) <= 0:
        fig.update_layout(title=title, height=360)
        return fig
    fig.add_trace(
        go.Pie(
            labels=chart_values[label_column],
            values=chart_values[value_column],
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:$,.2f}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=360, margin={"l": 10, "r": 10, "t": 50, "b": 10})
    return fig


def paper_rrg_snapshot(
    prices: pd.DataFrame,
    symbols: list[str],
    benchmark: str,
    as_of_date: object | None = None,
) -> pd.DataFrame:
    """Classify current paper holdings using the app's stock-level weekly RRG engine."""
    requested = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols if str(symbol).strip()))
    columns = ["symbol", "rrg_date", "rrg_quadrant"]
    if not requested:
        return pd.DataFrame(columns=columns)

    if prices.empty or not {"symbol", "date", "close"}.issubset(prices.columns):
        return pd.DataFrame({"symbol": requested, "rrg_date": pd.NaT, "rrg_quadrant": "Unresolved"})

    scoped_prices = prices.copy()
    scoped_prices["symbol"] = scoped_prices["symbol"].astype(str).str.strip().str.upper()
    scoped_prices["date"] = pd.to_datetime(scoped_prices["date"], errors="coerce")
    if as_of_date is not None:
        scoped_prices = scoped_prices[scoped_prices["date"].le(pd.to_datetime(as_of_date))]

    history = compute_rrg_history(
        scoped_prices,
        requested,
        benchmark,
        RRGConfig(ratio_window_weeks=10, momentum_window_weeks=10, tail_weeks=1),
    )
    if history.empty:
        latest = pd.DataFrame(columns=["symbol", "date", "quadrant"])
    else:
        latest = history.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)

    snapshot = pd.DataFrame({"symbol": requested}).merge(
        latest[["symbol", "date", "quadrant"]],
        on="symbol",
        how="left",
    )
    snapshot["rrg_date"] = pd.to_datetime(snapshot["date"], errors="coerce")
    snapshot["rrg_quadrant"] = snapshot["quadrant"].fillna("Unresolved")
    return snapshot[columns]


def paper_rrg_regime_frame(holdings: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Aggregate current holding value by stock-level RRG quadrant."""
    columns = ["rrg_quadrant", "value", "weight"]
    if holdings.empty or "symbol" not in holdings or "value" not in holdings:
        return pd.DataFrame(columns=columns)

    exposures = holdings[["symbol", "value"]].copy()
    exposures["symbol"] = exposures["symbol"].astype(str).str.strip().str.upper()
    exposures["value"] = pd.to_numeric(exposures["value"], errors="coerce").fillna(0.0)
    if snapshot.empty or "symbol" not in snapshot or "rrg_quadrant" not in snapshot:
        exposures["rrg_quadrant"] = "Unresolved"
    else:
        clean_snapshot = snapshot[["symbol", "rrg_quadrant"]].copy()
        clean_snapshot["symbol"] = clean_snapshot["symbol"].astype(str).str.strip().str.upper()
        exposures = exposures.merge(clean_snapshot, on="symbol", how="left")
        exposures["rrg_quadrant"] = exposures["rrg_quadrant"].fillna("Unresolved")
    exposures.loc[~exposures["rrg_quadrant"].isin(RRG_REGIME_ORDER), "rrg_quadrant"] = "Unresolved"

    grouped = exposures.groupby("rrg_quadrant", as_index=False)["value"].sum()
    ordered = pd.DataFrame({"rrg_quadrant": list(RRG_REGIME_ORDER)}).merge(grouped, on="rrg_quadrant", how="left")
    ordered["value"] = ordered["value"].fillna(0.0)
    total_value = float(ordered["value"].sum())
    ordered["weight"] = ordered["value"] / total_value if total_value > 0 else 0.0
    return ordered[columns]


def _rrg_quadrant_weight(regime: pd.DataFrame, quadrants: set[str]) -> float:
    if regime.empty or "rrg_quadrant" not in regime or "weight" not in regime:
        return 0.0
    return float(regime.loc[regime["rrg_quadrant"].isin(quadrants), "weight"].sum())


def _largest_rrg_move(current: pd.DataFrame, previous: pd.DataFrame) -> tuple[str, float]:
    merged = pd.DataFrame({"rrg_quadrant": list(RRG_REGIME_ORDER)}).merge(
        current[["rrg_quadrant", "weight"]].rename(columns={"weight": "current_weight"}),
        on="rrg_quadrant",
        how="left",
    ).merge(
        previous[["rrg_quadrant", "weight"]].rename(columns={"weight": "previous_weight"}),
        on="rrg_quadrant",
        how="left",
    )
    merged[["current_weight", "previous_weight"]] = merged[["current_weight", "previous_weight"]].fillna(0.0)
    merged["abs_delta"] = (merged["current_weight"] - merged["previous_weight"]).abs()
    largest = merged.sort_values(["abs_delta", "rrg_quadrant"], ascending=[False, True]).iloc[0]
    return str(largest["rrg_quadrant"]), float(largest["abs_delta"])


def paper_rrg_regime_sentence(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    previous_date: object | None,
) -> str:
    """Return a deterministic one-line description of the portfolio's RRG rotation."""
    if current.empty or float(pd.to_numeric(current.get("value", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) <= 0:
        return "⚪ RRG regime is unresolved because the holdings or benchmark lack enough price history."

    unresolved_weight = _rrg_quadrant_weight(current, {"Unresolved"})
    if unresolved_weight >= 0.999:
        return "⚪ RRG regime is unresolved because the holdings or benchmark lack enough price history."

    if previous.empty or previous_date is None:
        largest = current.sort_values(["weight", "rrg_quadrant"], ascending=[False, True]).iloc[0]
        return f"➖ First RRG regime snapshot; largest sleeve is {largest['rrg_quadrant']} at {float(largest['weight']):.1%}."

    previous_label = pd.to_datetime(previous_date).date().isoformat()
    strength_delta = _rrg_quadrant_weight(current, RRG_STRENGTH_QUADRANTS) - _rrg_quadrant_weight(previous, RRG_STRENGTH_QUADRANTS)
    weakness_delta = _rrg_quadrant_weight(current, RRG_WEAKNESS_QUADRANTS) - _rrg_quadrant_weight(previous, RRG_WEAKNESS_QUADRANTS)
    strength_weight = _rrg_quadrant_weight(current, RRG_STRENGTH_QUADRANTS)
    weakness_weight = _rrg_quadrant_weight(current, RRG_WEAKNESS_QUADRANTS)
    threshold = 0.0005

    if strength_delta > threshold and weakness_delta <= threshold:
        return f"🟢 RRG improved since {previous_label}: Leading/Improving exposure rose by {strength_delta * 100:.1f} pts to {strength_weight:.1%}."
    if weakness_delta > threshold and strength_delta <= threshold:
        return f"🔴 RRG weakened since {previous_label}: Weakening/Lagging exposure rose by {weakness_delta * 100:.1f} pts to {weakness_weight:.1%}."

    quadrant, largest_delta = _largest_rrg_move(current, previous)
    if largest_delta > threshold:
        return f"🟡 RRG rotation was mixed since {previous_label}: largest move was {quadrant} by {largest_delta * 100:.1f} pts."

    largest = current.sort_values(["weight", "rrg_quadrant"], ascending=[False, True]).iloc[0]
    return f"➖ No RRG regime change since {previous_label}; largest sleeve remains {largest['rrg_quadrant']} at {float(largest['weight']):.1%}."


def paper_rrg_regime_analysis(
    prices: pd.DataFrame,
    holdings: pd.DataFrame,
    benchmark: str,
) -> tuple[pd.DataFrame, str]:
    """Build current RRG exposure and its deterministic latest-vs-prior summary."""
    if holdings.empty or prices.empty or "date" not in prices:
        current = paper_rrg_regime_frame(holdings, pd.DataFrame())
        return current, paper_rrg_regime_sentence(current, pd.DataFrame(), None)

    symbols = holdings["symbol"].astype(str).str.strip().str.upper().tolist()
    scoped = prices.copy()
    scoped["symbol"] = scoped["symbol"].astype(str).str.strip().str.upper()
    scoped["date"] = pd.to_datetime(scoped["date"], errors="coerce")
    relevant_symbols = set(symbols) | {str(benchmark).strip().upper()}
    available_dates = sorted(scoped.loc[scoped["symbol"].isin(relevant_symbols), "date"].dropna().unique())
    latest_date = available_dates[-1] if available_dates else None
    previous_date = available_dates[-2] if len(available_dates) > 1 else None

    current_snapshot = paper_rrg_snapshot(scoped, symbols, benchmark, latest_date)
    current = paper_rrg_regime_frame(holdings, current_snapshot)
    previous = (
        paper_rrg_regime_frame(holdings, paper_rrg_snapshot(scoped, symbols, benchmark, previous_date))
        if previous_date is not None
        else pd.DataFrame()
    )
    return current, paper_rrg_regime_sentence(current, previous, previous_date)


BASKET_OVERLAY_COLOR = "#facc15"


def scaled_basket_overlay(history: pd.DataFrame, basket_history: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if history.empty or basket_history.empty or value_column not in history.columns:
        return pd.DataFrame(columns=["date", "value"])
    base = (
        history[["date", value_column]]
        .dropna()
        .merge(basket_history[["date", "basket_index"]], on="date", how="inner")
        .sort_values("date")
    )
    if base.empty:
        return pd.DataFrame(columns=["date", "value"])
    start_value = float(base[value_column].iloc[0])
    if pd.isna(start_value) or start_value <= 0:
        return pd.DataFrame(columns=["date", "value"])
    return pd.DataFrame(
        {
            "date": base["date"],
            "value": base["basket_index"] * start_value,
        }
    )


def basket_drawdown_overlay(basket_history: pd.DataFrame) -> pd.DataFrame:
    if basket_history.empty:
        return pd.DataFrame(columns=["date", "drawdown"])
    return basket_history[["date", "drawdown"]].copy()


def price_chart(history: pd.DataFrame, symbol: str, overlay: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=history["date"], y=history["sma50d"], mode="lines", name="SMA 50D"))
    fig.add_trace(go.Scatter(x=history["date"], y=history["sma200d"], mode="lines", name="SMA 200D"))
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["larsson_line_proxy"],
            mode="lines",
            name="Larsson proxy",
            line={"dash": "dash"},
        )
    )
    if overlay is not None and not overlay.empty:
        fig.add_trace(
            go.Scatter(
                x=overlay["date"],
                y=overlay["value"],
                mode="lines",
                name="Seed EW Basket",
                line={"color": BASKET_OVERLAY_COLOR, "dash": "dash", "width": 2},
            )
        )
    fig.update_layout(title=f"{symbol} Daily Trend", xaxis_title="", yaxis_title="Adjusted price")
    return fig


def weekly_chart(history: pd.DataFrame, symbol: str, overlay: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["weekly_close"], mode="lines", name="Weekly close"))
    fig.add_trace(go.Scatter(x=history["date"], y=history["sma30w"], mode="lines", name="SMA 30W"))
    fig.add_trace(go.Scatter(x=history["date"], y=history["sma200w"], mode="lines", name="SMA 200W"))
    if overlay is not None and not overlay.empty:
        fig.add_trace(
            go.Scatter(
                x=overlay["date"],
                y=overlay["value"],
                mode="lines",
                name="Seed EW Basket",
                line={"color": BASKET_OVERLAY_COLOR, "dash": "dash", "width": 2},
            )
        )
    fig.update_layout(title=f"{symbol} Weekly Trend", xaxis_title="", yaxis_title="Adjusted price")
    return fig


def drawdown_chart(history: pd.DataFrame, symbol: str, overlay: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["drawdown"], mode="lines", name="Drawdown"))
    if overlay is not None and not overlay.empty:
        fig.add_trace(
            go.Scatter(
                x=overlay["date"],
                y=overlay["drawdown"],
                mode="lines",
                name="Seed EW Drawdown",
                line={"color": BASKET_OVERLAY_COLOR, "dash": "dash", "width": 2},
            )
        )
    fig.update_layout(title=f"{symbol} Drawdown From Fetched-History Peak", xaxis_title="", yaxis_tickformat=".0%")
    return fig


def filter_ranked_by_seed_themes(ranked: pd.DataFrame, selected_themes: list[str]) -> pd.DataFrame:
    if not selected_themes or "theme" not in ranked.columns:
        return ranked.iloc[0:0].copy()
    return ranked[ranked["theme"].isin(selected_themes)].copy()


def latest_rrg_with_metadata(rrg_history: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    if rrg_history.empty:
        return pd.DataFrame()
    metadata_columns = [column for column in RRG_METADATA_COLUMNS if column in ranked.columns]
    return (
        rrg_history.sort_values("date")
        .groupby("symbol", as_index=False)
        .tail(1)
        .merge(ranked[metadata_columns], on="symbol", how="left")
    )


def latest_seed_rrg(rrg_history: pd.DataFrame) -> pd.DataFrame:
    if rrg_history.empty:
        return pd.DataFrame()
    return rrg_history.sort_values("date").groupby("seed_name", as_index=False).tail(1)


def rrg_selection_from_event(event: object, fig: go.Figure) -> str | None:
    """Return the selected RRG symbol from a Streamlit Plotly selection event."""
    if not hasattr(event, "get"):
        return None
    selection = event.get("selection")
    if not hasattr(selection, "get"):
        return None
    points = selection.get("points")
    if not points:
        return None
    point = points[0]
    if not hasattr(point, "get"):
        return None
    curve_number = point.get("curve_number", point.get("curveNumber"))
    if not isinstance(curve_number, int) or curve_number >= len(fig.data):
        return None
    trace_name = getattr(fig.data[curve_number], "name", None)
    return str(trace_name) if trace_name else None


def toggle_rrg_selection(current_symbol: str | None, clicked_symbol: str | None) -> str | None:
    if not clicked_symbol:
        return current_symbol
    return None if clicked_symbol == current_symbol else clicked_symbol


def rrg_chart(
    rrg_history: pd.DataFrame,
    latest_table: pd.DataFrame,
    benchmark: str,
    selected_symbol: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    if rrg_history.empty:
        fig.update_layout(
            title=f"Relative Rotation Graph vs {benchmark}",
            xaxis_title="RS Ratio",
            yaxis_title="RS Momentum",
            height=680,
        )
        return fig

    latest_by_symbol = latest_table.set_index("symbol") if not latest_table.empty else pd.DataFrame()
    x_min = min(98.0, float(rrg_history["rs_ratio"].min()) - 1)
    x_max = max(102.0, float(rrg_history["rs_ratio"].max()) + 1)
    y_min = min(98.0, float(rrg_history["rs_momentum"].min()) - 1)
    y_max = max(102.0, float(rrg_history["rs_momentum"].max()) + 1)

    for symbol, group in rrg_history.groupby("symbol", sort=False):
        group = group.sort_values("date")
        latest = group.iloc[-1]
        quadrant = str(latest["quadrant"])
        color = RRG_COLORS.get(quadrant, "#64748b")
        is_selected = selected_symbol == symbol
        is_dimmed = bool(selected_symbol) and not is_selected
        tail_width = 4 if is_selected else 2
        marker_size = 16 if is_selected else 11
        marker_line = {"color": "#ffffff", "width": 3 if is_selected else 1}
        trace_opacity = 1.0 if is_selected else (0.14 if is_dimmed else 0.45)
        marker_opacity = 1.0 if is_selected else (0.28 if is_dimmed else 1.0)
        meta = latest_by_symbol.loc[symbol] if symbol in latest_by_symbol.index else {}
        name = meta.get("name", "") if hasattr(meta, "get") else ""
        theme = meta.get("theme", "") if hasattr(meta, "get") else ""
        score = meta.get("score", float("nan")) if hasattr(meta, "get") else float("nan")
        benchmark_score = meta.get("benchmark_adjusted_score", float("nan")) if hasattr(meta, "get") else float("nan")
        hover_text = (
            f"<b>{symbol}</b><br>"
            f"{name}<br>"
            f"Theme: {theme}<br>"
            f"Quadrant: {quadrant}<br>"
            f"RS Ratio: {latest['rs_ratio']:.1f}<br>"
            f"RS Momentum: {latest['rs_momentum']:.1f}<br>"
            f"Base Score: {score:.1f}<br>"
            f"Benchmark Score: {benchmark_score:.1f}"
        )

        fig.add_trace(
            go.Scatter(
                x=group["rs_ratio"],
                y=group["rs_momentum"],
                mode="lines",
                line={"color": color, "width": tail_width},
                opacity=trace_opacity,
                hoverinfo="skip",
                name=f"{symbol} path",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[latest["rs_ratio"]],
                y=[latest["rs_momentum"]],
                mode="markers+text",
                marker={"color": color, "size": marker_size, "line": marker_line, "opacity": marker_opacity},
                text=[symbol],
                textposition="top center",
                hovertemplate=hover_text + "<extra></extra>",
                name=symbol,
                showlegend=False,
            )
        )
        if len(group) >= 2:
            previous = group.iloc[-2]
            fig.add_annotation(
                x=latest["rs_ratio"],
                y=latest["rs_momentum"],
                ax=previous["rs_ratio"],
                ay=previous["rs_momentum"],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2.4 if is_selected else 1.4,
                arrowcolor=color,
                opacity=1.0 if is_selected else (0.18 if is_dimmed else 0.75),
            )

    fig.add_vline(x=100, line_width=1, line_dash="dash", line_color="#94a3b8")
    fig.add_hline(y=100, line_width=1, line_dash="dash", line_color="#94a3b8")
    fig.add_annotation(x=x_max, y=y_max, text="Leading", showarrow=False, font={"color": RRG_COLORS["Leading"]})
    fig.add_annotation(x=x_max, y=y_min, text="Weakening", showarrow=False, font={"color": RRG_COLORS["Weakening"]})
    fig.add_annotation(x=x_min, y=y_min, text="Lagging", showarrow=False, font={"color": RRG_COLORS["Lagging"]})
    fig.add_annotation(x=x_min, y=y_max, text="Improving", showarrow=False, font={"color": RRG_COLORS["Improving"]})
    fig.update_layout(
        title=f"Relative Rotation Graph vs {benchmark}",
        xaxis_title="RS Ratio (relative trend)",
        yaxis_title="RS Momentum",
        xaxis={"range": [x_min, x_max]},
        yaxis={"range": [y_min, y_max]},
        height=720,
        hovermode="closest",
    )
    return fig


def rrg_seed_chart(
    rrg_history: pd.DataFrame,
    latest_table: pd.DataFrame,
    benchmark: str,
    selected_seed: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    if rrg_history.empty:
        fig.update_layout(
            title=f"Seed Universe Relative Rotation Graph vs {benchmark}",
            xaxis_title="RS Ratio",
            yaxis_title="RS Momentum",
            height=680,
        )
        return fig

    latest_by_seed = latest_table.set_index("seed_name") if not latest_table.empty else pd.DataFrame()
    x_min = min(98.0, float(rrg_history["rs_ratio"].min()) - 1)
    x_max = max(102.0, float(rrg_history["rs_ratio"].max()) + 1)
    y_min = min(98.0, float(rrg_history["rs_momentum"].min()) - 1)
    y_max = max(102.0, float(rrg_history["rs_momentum"].max()) + 1)

    for seed_name, group in rrg_history.groupby("seed_name", sort=False):
        group = group.sort_values("date")
        latest = group.iloc[-1]
        quadrant = str(latest["quadrant"])
        color = RRG_COLORS.get(quadrant, "#64748b")
        is_selected = selected_seed == seed_name
        is_dimmed = bool(selected_seed) and not is_selected
        tail_width = 4 if is_selected else 2
        marker_size = 18 if is_selected else 13
        trace_opacity = 1.0 if is_selected else (0.14 if is_dimmed else 0.45)
        marker_opacity = 1.0 if is_selected else (0.28 if is_dimmed else 1.0)
        meta = latest_by_seed.loc[seed_name] if seed_name in latest_by_seed.index else {}
        symbols_used = meta.get("symbols_used", "") if hasattr(meta, "get") else ""
        symbols_missing = meta.get("symbols_missing", "") if hasattr(meta, "get") else ""
        symbol_count = int(meta.get("symbol_count", 0)) if hasattr(meta, "get") else 0
        missing_count = int(meta.get("missing_count", 0)) if hasattr(meta, "get") else 0
        hover_text = (
            f"<b>{seed_name}</b><br>"
            f"Quadrant: {quadrant}<br>"
            f"RS Ratio: {latest['rs_ratio']:.1f}<br>"
            f"RS Momentum: {latest['rs_momentum']:.1f}<br>"
            f"Symbols used: {symbol_count}<br>"
            f"Missing symbols: {missing_count}<br>"
            f"Used: {symbols_used}"
        )
        if symbols_missing:
            hover_text += f"<br>Missing: {symbols_missing}"

        fig.add_trace(
            go.Scatter(
                x=group["rs_ratio"],
                y=group["rs_momentum"],
                mode="lines",
                line={"color": color, "width": tail_width},
                opacity=trace_opacity,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[latest["rs_ratio"]],
                y=[latest["rs_momentum"]],
                mode="markers+text",
                marker={"color": color, "size": marker_size, "line": {"color": "white", "width": 3 if is_selected else 1}, "opacity": marker_opacity},
                text=[seed_name.replace(" Universe", "")],
                textposition="top center",
                hovertemplate=hover_text + "<extra></extra>",
                name=seed_name,
                showlegend=False,
            )
        )
        if len(group) >= 2:
            previous = group.iloc[-2]
            fig.add_annotation(
                x=latest["rs_ratio"],
                y=latest["rs_momentum"],
                ax=previous["rs_ratio"],
                ay=previous["rs_momentum"],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2.4 if is_selected else 1.4,
                arrowcolor=color,
                opacity=1.0 if is_selected else (0.18 if is_dimmed else 0.75),
            )

    fig.add_vline(x=100, line_width=1, line_dash="dash", line_color="#94a3b8")
    fig.add_hline(y=100, line_width=1, line_dash="dash", line_color="#94a3b8")
    fig.add_annotation(x=x_max, y=y_max, text="Leading", showarrow=False, font={"color": RRG_COLORS["Leading"]})
    fig.add_annotation(x=x_max, y=y_min, text="Weakening", showarrow=False, font={"color": RRG_COLORS["Weakening"]})
    fig.add_annotation(x=x_min, y=y_min, text="Lagging", showarrow=False, font={"color": RRG_COLORS["Lagging"]})
    fig.add_annotation(x=x_min, y=y_max, text="Improving", showarrow=False, font={"color": RRG_COLORS["Improving"]})
    fig.update_layout(
        title=f"Seed Universe Relative Rotation Graph vs {benchmark}",
        xaxis_title="RS Ratio (relative trend)",
        yaxis_title="RS Momentum",
        xaxis={"range": [x_min, x_max]},
        yaxis={"range": [y_min, y_max]},
        height=720,
        hovermode="closest",
    )
    return fig


def ranking_column_config() -> dict[str, object]:
    return {
        "rank": st.column_config.NumberColumn(
            "Rank",
            help="Current displayed rank after the selected filters and sorting mode.",
            format="%d",
        ),
        "symbol": st.column_config.TextColumn(
            "Symbol",
            help="Ticker symbol for the stock.",
        ),
        "name": st.column_config.TextColumn(
            "Name",
            help="Company or instrument name from the seed universe or ticker metadata.",
        ),
        "theme": st.column_config.TextColumn(
            "Theme",
            help="Built-in seed group used for filtering, such as semiconductors or cloud platforms.",
        ),
        "benchmark_adjusted_score": st.column_config.NumberColumn(
            "Benchmark Score",
            help="Optional ranking score: 85% Base Score plus 15% RS Score. Used for sorting only when enabled in the sidebar.",
            format="%.1f",
        ),
        "score": st.column_config.NumberColumn(
            "Base Score",
            help="Core 0-100 technical score from regime, trend, 200D/200W filters, drawdown, and extension.",
            format="%.1f",
        ),
        "relative_strength_score": st.column_config.NumberColumn(
            "RS Score",
            help="0-100 relative-strength score versus the selected benchmark. Around 50 is neutral.",
            format="%.1f",
        ),
        "relative_strength_label": st.column_config.TextColumn(
            "RS Label",
            help="Plain-language relative-strength bucket versus the selected benchmark.",
        ),
        "rs_3m": st.column_config.NumberColumn(
            "RS 3M",
            help="Three-month stock return minus benchmark return on aligned trading dates.",
            format="%.1f%%",
        ),
        "rs_6m": st.column_config.NumberColumn(
            "RS 6M",
            help="Six-month stock return minus benchmark return on aligned trading dates.",
            format="%.1f%%",
        ),
        "rs_12m": st.column_config.NumberColumn(
            "RS 12M",
            help="Twelve-month stock return minus benchmark return on aligned trading dates.",
            format="%.1f%%",
        ),
        "larsson_regime": st.column_config.TextColumn(
            "Larsson Regime",
            help="Current Larsson-proxy regime approximation: Bull, Transition, or Bear.",
        ),
        "larsson_regime_history": st.column_config.BarChartColumn(
            "Regime History",
            width=120,
            help="Weekly Larsson-proxy history: high bars are Bull, mid bars are Transition, low bars are Bear.",
            y_min=0,
            y_max=1,
            color="#22c55e",
        ),
        "daily_trend": st.column_config.TextColumn(
            "Daily Trend",
            help="Daily trend state from price versus the 50D average and the 50D slope.",
        ),
        "weekly_trend": st.column_config.TextColumn(
            "Weekly Trend",
            help="Weekly trend state from weekly close versus the 30W average and the 30W slope.",
        ),
        "monthly_trend": st.column_config.TextColumn(
            "Monthly Trend",
            help="Monthly trend state from monthly close versus the 10M average and the 10M slope.",
        ),
        "pass_200d_label": st.column_config.TextColumn(
            "200D",
            help="Whether the latest adjusted close is above the 200-day moving average.",
        ),
        "pass_200w_label": st.column_config.TextColumn(
            "200W",
            help="Whether the latest weekly close is above the 200-week moving average. n/a means insufficient history.",
        ),
        "latest_close": st.column_config.NumberColumn(
            "Latest",
            help="Latest cached adjusted close.",
            format="$%.2f",
        ),
        "drawdown_52w": st.column_config.NumberColumn(
            "52W DD",
            help="Current drawdown from the highest close in the last 52 weeks.",
            format="%.1f%%",
        ),
        "extension_sma50d": st.column_config.NumberColumn(
            "Ext 50D",
            help="How far the latest close is above or below the 50-day moving average.",
            format="%.1f%%",
        ),
        "extension_sma200d": st.column_config.NumberColumn(
            "Ext 200D",
            help="How far the latest close is above or below the 200-day moving average.",
            format="%.1f%%",
        ),
        "extension_sma30w": st.column_config.NumberColumn(
            "Ext 30W",
            help="How far the latest close is above or below the 30-week moving average.",
            format="%.1f%%",
        ),
        "data_warnings": st.column_config.TextColumn(
            "Warnings",
            help="Data-quality or insufficient-history notes that affect interpretation.",
        ),
    }


def rrg_column_config() -> dict[str, object]:
    return {
        "symbol": st.column_config.TextColumn("Symbol", help="Ticker symbol plotted on the RRG."),
        "name": st.column_config.TextColumn("Name", help="Company or instrument name from the seed universe."),
        "theme": st.column_config.TextColumn("Theme", help="Built-in seed group used by the RRG theme filter."),
        "quadrant": st.column_config.TextColumn(
            "Quadrant",
            help="Current RRG quadrant: Leading, Weakening, Lagging, or Improving.",
        ),
        "rs_ratio": st.column_config.NumberColumn(
            "RS Ratio",
            help="RRG x-axis: relative trend versus the benchmark, centered near 100.",
            format="%.1f",
        ),
        "rs_momentum": st.column_config.NumberColumn(
            "RS Momentum",
            help="RRG y-axis: momentum of relative trend, centered near 100.",
            format="%.1f",
        ),
        "relative_strength_score": st.column_config.NumberColumn(
            "RS Score",
            help="0-100 relative-strength score versus the selected benchmark. Around 50 is neutral.",
            format="%.1f",
        ),
        "rs_3m": st.column_config.NumberColumn(
            "RS 3M",
            help="Three-month stock return minus benchmark return on aligned trading dates.",
            format="%.1f%%",
        ),
        "rs_6m": st.column_config.NumberColumn(
            "RS 6M",
            help="Six-month stock return minus benchmark return on aligned trading dates.",
            format="%.1f%%",
        ),
        "rs_12m": st.column_config.NumberColumn(
            "RS 12M",
            help="Twelve-month stock return minus benchmark return on aligned trading dates.",
            format="%.1f%%",
        ),
        "score": st.column_config.NumberColumn(
            "Base Score",
            help="Core 0-100 technical score from the Rankings tab.",
            format="%.1f",
        ),
        "benchmark_adjusted_score": st.column_config.NumberColumn(
            "Benchmark Score",
            help="Optional score combining 85% Base Score and 15% RS Score.",
            format="%.1f",
        ),
    }


def rrg_seed_column_config() -> dict[str, object]:
    return {
        "seed_name": st.column_config.TextColumn(
            "Seed Universe",
            help="Built-in seed preset represented by one equal-weight basket dot.",
        ),
        "quadrant": st.column_config.TextColumn(
            "Quadrant",
            help="Current RRG quadrant: Leading, Weakening, Lagging, or Improving.",
        ),
        "rs_ratio": st.column_config.NumberColumn(
            "RS Ratio",
            help="RRG x-axis: seed-basket relative trend versus the benchmark, centered near 100.",
            format="%.1f",
        ),
        "rs_momentum": st.column_config.NumberColumn(
            "RS Momentum",
            help="RRG y-axis: momentum of the seed-basket relative trend, centered near 100.",
            format="%.1f",
        ),
        "symbol_count": st.column_config.NumberColumn(
            "Symbols Used",
            help="Number of seed tickers with usable price history included in the equal-weight basket.",
            format="%d",
        ),
        "missing_count": st.column_config.NumberColumn(
            "Missing",
            help="Number of seed tickers skipped because usable price history was unavailable.",
            format="%d",
        ),
        "symbols_used": st.column_config.TextColumn(
            "Included Tickers",
            help="Tickers included in the equal-weight seed basket.",
        ),
        "symbols_missing": st.column_config.TextColumn(
            "Missing Tickers",
            help="Tickers skipped from the seed basket because usable price data was unavailable.",
        ),
    }


def theme_breadth_column_config() -> dict[str, object]:
    return {
        "seed_universe": st.column_config.TextColumn("Theme", help="Fixed built-in seed universe used for this local breadth calculation."),
        "date": st.column_config.DateColumn("As of", help="Latest Friday-ended weekly observation."),
        "breadth": st.column_config.NumberColumn("Above 20D", help="Eligible constituents above their own 20-day SMA.", format="%.1f%%"),
        "breadth_4w_average": st.column_config.NumberColumn("4W Avg", help="Simple average of the latest four weekly breadth readings.", format="%.1f%%"),
        "breadth_change": st.column_config.NumberColumn("1W Change", help="Latest breadth minus the prior weekly reading.", format="%+.1f%%"),
        "roc_50d": st.column_config.NumberColumn("50D ROC", help="50-trading-session percent change of the dashboard equal-weight theme index.", format="%+.1f%%"),
        "confirmation": st.column_config.TextColumn("Confirmation", help="Agreement between breadth versus its 4W average and positive/negative 50D ROC."),
        "coverage": st.column_config.NumberColumn("Coverage", help="Eligible constituents with sufficient 20D history divided by the fixed universe size.", format="%.0f%%"),
        "eligible_count": st.column_config.NumberColumn("Eligible", help="Constituents with a valid close and 20D SMA at the observation date.", format="%d"),
        "universe_size": st.column_config.NumberColumn("Universe", help="Fixed number of tickers in the current built-in seed CSV.", format="%d"),
        "missing_or_short_history": st.column_config.NumberColumn("Unavailable", help="Constituents missing a valid close or enough 20D history at the observation date.", format="%d"),
    }


def theme_breadth_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Scale ratio-valued breadth fields for percent-formatted table columns."""
    display = summary.copy()
    for column in THEME_BREADTH_PERCENT_COLUMNS:
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce") * 100.0
    return display


def portfolio_column_config() -> dict[str, object]:
    return {
        "weight": st.column_config.NumberColumn("Weight", help="Model portfolio target weight.", format="%.1f%%"),
        "symbol": st.column_config.TextColumn("Symbol", help="Ticker or Cash row."),
        "name": st.column_config.TextColumn("Name", help="Company name from the seed universe."),
        "seed_universe": st.column_config.TextColumn("Seed Universe", help="Seed preset that supplied this holding."),
        "allocator_score": st.column_config.NumberColumn("Allocator Score", help="Deterministic score used to size positions.", format="%.1f"),
        "score": st.column_config.NumberColumn("Base Score", help="Core technical ranking score.", format="%.1f"),
        "benchmark_adjusted_score": st.column_config.NumberColumn("Benchmark Score", help="Base score blended with relative strength.", format="%.1f"),
        "relative_strength_score": st.column_config.NumberColumn("RS Score", help="0-100 score versus the selected benchmark.", format="%.1f"),
        "larsson_regime": st.column_config.TextColumn("Regime", help="Current Larsson-proxy regime."),
        "rrg_quadrant": st.column_config.TextColumn("RRG", help="Seed-level RRG quadrant behind this holding."),
        "drawdown_52w": st.column_config.NumberColumn("52W DD", help="Current drawdown from 52-week high.", format="%.1f%%"),
        "extension_sma50d": st.column_config.NumberColumn("Ext 50D", help="Extension versus 50-day moving average.", format="%.1f%%"),
        "reason_codes": st.column_config.TextColumn("Reasons", help="Deterministic reason codes for inclusion, caps, or cash."),
        "data_warnings": st.column_config.TextColumn("Warnings", help="Data-quality or insufficient-history notes."),
    }


def portfolio_change_column_config() -> dict[str, object]:
    return {
        "symbol": st.column_config.TextColumn("Symbol", help="Ticker whose model weight changed."),
        "previous_weight": st.column_config.NumberColumn("Previous", help="Saved baseline portfolio weight.", format="%.1f%%"),
        "target_weight": st.column_config.NumberColumn("Target", help="Current target model weight.", format="%.1f%%"),
        "delta": st.column_config.NumberColumn("Change", help="Target weight minus previous weight.", format="%.1f%%"),
        "action": st.column_config.TextColumn("Action", help="Added, removed, increased, reduced, or unchanged."),
        "reason_codes": st.column_config.TextColumn("Reasons", help="Why the change was accepted or suppressed."),
    }


def allocation_pie_chart(portfolio: pd.DataFrame, name_column: str, title: str) -> go.Figure:
    fig = go.Figure()
    if portfolio.empty or name_column not in portfolio.columns:
        fig.update_layout(title=title, height=360)
        return fig
    grouped = portfolio.groupby(name_column, as_index=False, sort=False)["weight"].sum()
    fig.add_trace(go.Pie(labels=grouped[name_column], values=grouped["weight"], hole=0.45))
    fig.update_layout(title=title, height=360)
    return fig


def portfolio_rrg_allocation_frame(applied_target: pd.DataFrame, rrg_history: pd.DataFrame) -> pd.DataFrame:
    """Group applied non-cash allocation by each holding's latest stock-level RRG quadrant."""
    columns = ["rrg_quadrant", "weight"]
    if applied_target.empty or not {"symbol", "weight"}.issubset(applied_target.columns):
        return pd.DataFrame(columns=columns)

    holdings = applied_target[["symbol", "weight"]].copy()
    holdings["symbol"] = holdings["symbol"].astype(str).str.strip().str.upper()
    holdings["weight"] = pd.to_numeric(holdings["weight"], errors="coerce").fillna(0.0)
    holdings = holdings[holdings["symbol"].ne("") & holdings["symbol"].ne("CASH") & holdings["weight"].gt(0)].copy()
    if holdings.empty:
        return pd.DataFrame(columns=columns)

    if rrg_history.empty or not {"symbol", "date", "quadrant"}.issubset(rrg_history.columns):
        latest = pd.DataFrame(columns=["symbol", "quadrant"])
    else:
        latest = rrg_history.copy()
        latest["symbol"] = latest["symbol"].astype(str).str.strip().str.upper()
        latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
        latest = (
            latest[latest["symbol"].isin(holdings["symbol"])]
            .dropna(subset=["date"])
            .sort_values(["symbol", "date"])
            .groupby("symbol", as_index=False)
            .tail(1)[["symbol", "quadrant"]]
        )

    grouped = holdings.merge(latest, on="symbol", how="left")
    grouped["rrg_quadrant"] = grouped["quadrant"].fillna("Unresolved")
    grouped.loc[~grouped["rrg_quadrant"].isin(RRG_REGIME_ORDER), "rrg_quadrant"] = "Unresolved"
    grouped = grouped.groupby("rrg_quadrant", as_index=False)["weight"].sum()
    return (
        pd.DataFrame({"rrg_quadrant": list(RRG_REGIME_ORDER)})
        .merge(grouped, on="rrg_quadrant", how="left")
        .fillna({"weight": 0.0})[columns]
    )


def symbol_from_dataframe_selection(event: object, ranked: pd.DataFrame) -> str | None:
    if ranked.empty or not hasattr(event, "get"):
        return None
    selection = event.get("selection")
    if not hasattr(selection, "get"):
        return None
    rows = selection.get("rows")
    if not rows:
        return None
    row_index = rows[0]
    if not isinstance(row_index, int) or row_index >= len(ranked):
        return None
    return str(ranked.iloc[row_index]["symbol"])


def dashboard_view_after_pending_request(current_view: object, requested_view: object) -> str:
    """Choose a valid dashboard view before its navigation widget is created."""
    if requested_view in VIEW_OPTIONS:
        return str(requested_view)
    if current_view in VIEW_OPTIONS:
        return str(current_view)
    return "Rankings"


def display_rankings(ranked: pd.DataFrame) -> object:
    columns = [
        "rank",
        "symbol",
        "name",
        "theme",
        "benchmark_adjusted_score",
        "score",
        "relative_strength_score",
        "relative_strength_label",
        "rs_3m",
        "rs_6m",
        "rs_12m",
        "larsson_regime",
        "larsson_regime_history",
        "daily_trend",
        "weekly_trend",
        "monthly_trend",
        "pass_200d_label",
        "pass_200w_label",
        "latest_close",
        "drawdown_52w",
        "extension_sma50d",
        "extension_sma200d",
        "extension_sma30w",
        "data_warnings",
    ]
    existing = [column for column in columns if column in ranked.columns]
    return st.dataframe(
        ranked[existing],
        width="stretch",
        hide_index=True,
        column_config=ranking_column_config(),
        key="rankings_table",
        on_select="rerun",
        selection_mode="single-row",
    )


def main() -> None:
    seed_frames = {
        label: cached_universe(str(path), path.stat().st_mtime_ns)
        for label, path in SEED_UNIVERSES.items()
        if path is not None
    }

    st.sidebar.header("Inputs")
    benchmark = st.sidebar.text_input("Benchmark", value="QQQ").strip().upper() or "QQQ"
    years = st.sidebar.slider("Price history", min_value=3, max_value=15, value=10, step=1)
    larsson_history_weeks = st.sidebar.slider("Larsson history weeks", min_value=8, max_value=104, value=26, step=1)
    st.sidebar.subheader("Seed Universe")
    options = seed_universe_options()
    # This runs before the widget is instantiated, so sessions that still hold
    # the retired Mag7 + SpaceX value migrate without a stale-widget error.
    st.session_state["seed_universe"] = seed_universe_label_after_migration(
        st.session_state.get("seed_universe"), options
    )
    seed_label = st.sidebar.selectbox(
        "Seed universe",
        options,
        index=options.index(DEFAULT_SEED_UNIVERSE),
        format_func=seed_universe_display_name,
        help="Choose the stock basket used for rankings and charts. The full universe name remains available after selection.",
    )
    active_seed = select_seed_universe(seed_label, seed_frames)
    default_text = "\n".join(active_seed["symbol"].tolist())
    current_seed_text_mode = seed_text_mode(seed_label, active_seed)
    if st.session_state.get("ticker_text_mode") != current_seed_text_mode:
        st.session_state["ticker_text"] = default_text
        st.session_state["ticker_text_mode"] = current_seed_text_mode
    ticker_text = st.sidebar.text_area("Tickers", height=220, key="ticker_text")
    require_200d = st.sidebar.checkbox("Require 200D pass", value=False)
    require_200w = st.sidebar.checkbox("Require 200W pass", value=False)
    use_relative_strength = st.sidebar.checkbox("Use relative strength in ranking", value=False)
    sidebar_refresh_clicked = st.sidebar.button("Refresh cache")
    # The Paper Portfolio control below requests a rerun, so the same explicit
    # refresh path is used regardless of where the user starts it.
    refresh_clicked = sidebar_refresh_clicked or bool(st.session_state.pop("paper_portfolio_refresh_requested", False))

    requested_dashboard_view = st.session_state.pop("pending_dashboard_view", None)
    st.session_state["dashboard_view"] = dashboard_view_after_pending_request(
        st.session_state.get("dashboard_view"), requested_dashboard_view
    )
    active_view = st.segmented_control(
        "Dashboard view",
        VIEW_OPTIONS,
        key="dashboard_view",
        label_visibility="collapsed",
    )

    manual_symbols = parse_ticker_text(ticker_text)
    universe = build_active_universe(active_seed, manual_symbols)
    rrg_seed_labels = selected_rrg_seed_labels_for_fetch(
        st.session_state.get("rrg_view", "Stocks"),
        st.session_state.get("rrg_seed_universes", default_rrg_seed_universes()),
    ) if active_view == "RRG" else []
    if active_view == "Portfolio" and not st.session_state.get("_portfolio_seed_universes_all_default_migrated", False):
        st.session_state["portfolio_seed_universes"] = portfolio_seed_universes_after_default_migration(
            st.session_state.get("portfolio_seed_universes"),
            seed_rrg_universe_options(),
        )
        st.session_state["_portfolio_seed_universes_all_default_migrated"] = True
    portfolio_data_enabled = bool(st.session_state.get("portfolio_data_enabled", False))
    portfolio_seed_labels_for_fetch = selected_portfolio_seed_labels_for_fetch(
        st.session_state.get("portfolio_seed_universes", default_portfolio_seed_universes()),
        portfolio_data_enabled,
    ) if active_view == "Portfolio" else []
    rrg_seed_symbols = seed_symbols_for_labels(seed_frames, rrg_seed_labels)
    portfolio_seed_symbols = seed_symbols_for_labels(seed_frames, portfolio_seed_labels_for_fetch)
    theme_breadth_symbols = seed_symbols_for_labels(seed_frames, seed_rrg_universe_options()) if active_view == "Theme Breadth" else []
    all_seed = all_seed_universe(seed_frames)
    lab_symbols = set(all_seed["symbol"]) if active_view == "Chatgptfolio Lab" else set()
    paper_symbols = set(paper_valuation_symbols(db_path=DEFAULT_DB_PATH)) if active_view in {"Paper Portfolio", "Chatgptfolio Lab"} else set()
    active_symbols = set(universe["symbol"]) if active_view not in {"Paper Portfolio", "System Health", "Theme Breadth"} else set()
    symbols = sorted(active_symbols.union(rrg_seed_symbols).union(portfolio_seed_symbols).union(theme_breadth_symbols).union(lab_symbols).union(paper_symbols).union({benchmark}))
    refresh_key = st.session_state.get("refresh_key", 0)
    if refresh_clicked:
        refresh_key += 1
        st.session_state["refresh_key"] = refresh_key

    st.title("CTO-Style Stock Ranking")

    try:
        price_started_at = perf_counter()
        cached_before = cached_symbol_count(symbols, DEFAULT_DB_PATH)
        price_cache_state = session_cache_state("prices", (tuple(symbols), refresh_key, years))
        prices = cached_prices(tuple(symbols), refresh_key, refresh_clicked, years, str(DEFAULT_DB_PATH))
        missing_symbol_count = max(0, len(symbols) - cached_before)
        loaded_detail = "manual refresh" if refresh_clicked else ("warm local cache" if missing_symbol_count == 0 else f"{missing_symbol_count} symbols missing from local cache")
        record_timing("Price cache", price_started_at, price_cache_state, loaded_detail)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    if not refresh_clicked:
        cached_symbols = set(prices["symbol"].astype(str).str.upper()) if not prices.empty and "symbol" in prices else set()
        missing_symbols = [symbol for symbol in symbols if symbol not in cached_symbols]
        if missing_symbols:
            preview = ", ".join(missing_symbols[:20])
            suffix = "" if len(missing_symbols) <= 20 else f", and {len(missing_symbols) - 20} more"
            st.warning(f"{len(missing_symbols)} requested symbols are not in the local price cache: {preview}{suffix}. Press Refresh cache to download market data.")

    if prices.empty:
        st.warning("No price data available yet. Refresh the cache after installing the required dependencies.")
        return

    signals = pd.DataFrame()
    ranked = pd.DataFrame()
    needs_primary_signals = active_view not in {"Chatgptfolio Lab", "Paper Portfolio", "System Health", "Theme Breadth"}
    if needs_primary_signals:
        config = SignalConfig(larsson_history_weeks=larsson_history_weeks)
        active_signal_symbols = set(universe["symbol"]).union({benchmark})
        signal_prices = prices_for_symbols(prices, active_signal_symbols)
        signal_started_at = perf_counter()
        signal_cache_state = session_cache_state("primary_signals", (tuple(sorted(active_signal_symbols)), benchmark, larsson_history_weeks, len(signal_prices), str(signal_prices["date"].max())))
        signals = cached_signal_table(signal_prices, benchmark, larsson_history_weeks)
        record_timing("Technical signals", signal_started_at, signal_cache_state, f"{len(active_signal_symbols)} symbols")
        signals = signals.merge(universe, on="symbol", how="left")
        signals[["name", "theme", "notes"]] = signals[["name", "theme", "notes"]].fillna("")
        signals[["fundamental_score", "catalyst_score"]] = signals[["fundamental_score", "catalyst_score"]].fillna(0.0)
        ranked = rank_universe(signals, RankingConfig(use_relative_strength=use_relative_strength))
        if require_200d:
            ranked = ranked[ranked["pass_200d"] == True].copy()
        if require_200w:
            ranked = ranked[ranked["pass_200w"] == True].copy()
        ranked = ranked.reset_index(drop=True)
        ranked["rank"] = range(1, len(ranked) + 1)
        benchmark_row = signals[signals["symbol"] == benchmark]
        benchmark_regime = benchmark_row["larsson_regime"].iloc[0] if not benchmark_row.empty else "n/a"
        benchmark_close = benchmark_row["latest_close"].iloc[0] if not benchmark_row.empty else float("nan")
        benchmark_drawdown = benchmark_row["drawdown_52w"].iloc[0] if not benchmark_row.empty else float("nan")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Benchmark", benchmark)
        c2.metric("Benchmark Regime", benchmark_regime)
        c3.metric("Benchmark Close", money(benchmark_close))
        c4.metric("Benchmark 52W Drawdown", pct(benchmark_drawdown))
        c5.metric("Cache Freshness", latest_cache_timestamp(DEFAULT_DB_PATH))
        s1, s2, s3 = st.columns(3)
        s1.metric("Symbols Ranked", len(ranked))
        s2.metric("Passing 200D", int((ranked["pass_200d"] == True).sum()) if not ranked.empty else 0)
        s3.metric("Passing 200W", int((ranked["pass_200w"] == True).sum()) if not ranked.empty else 0)

    if active_view == "Rankings":
        display = ranked.copy()
        for column in [
            "rs_3m",
            "rs_6m",
            "rs_12m",
            "drawdown_52w",
            "extension_sma50d",
            "extension_sma200d",
            "extension_sma30w",
        ]:
            display[column] = display[column] * 100
        ranking_event = display_rankings(display)
        selected_from_table = symbol_from_dataframe_selection(ranking_event, ranked)
        if selected_from_table:
            st.session_state["detail_symbol"] = selected_from_table
            # The navigation widget has already been created this run. Defer
            # changing its state until the next rerun to satisfy Streamlit's
            # widget-state contract.
            st.session_state["pending_dashboard_view"] = "Detail"
            st.rerun()

    elif active_view == "Detail":
        if ranked.empty:
            st.info("No symbols match the active filters.")
        else:
            detail_symbols = ranked["symbol"].tolist()
            current_detail_symbol = st.session_state.get("detail_symbol")
            if current_detail_symbol not in detail_symbols:
                st.session_state["detail_symbol"] = detail_symbols[0]
            selected = st.selectbox("Symbol", detail_symbols, key="detail_symbol")
            selected_row = ranked[ranked["symbol"] == selected].iloc[0]
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Score", f"{selected_row['score']:.1f}")
            d2.metric("Latest", money(selected_row["latest_close"]))
            d3.metric("52W Drawdown", pct(selected_row["drawdown_52w"]))
            d4.metric("Extension vs 200D", pct(selected_row["extension_sma200d"]))

            daily_history = price_history_with_indicators(prices, selected, config)
            weekly_history = weekly_history_with_indicators(prices, selected, config)
            dd_history = drawdown_history(prices, selected)
            st.plotly_chart(price_chart(daily_history, selected), use_container_width=True)
            st.plotly_chart(weekly_chart(weekly_history, selected), use_container_width=True)
            st.plotly_chart(drawdown_chart(dd_history, selected), use_container_width=True)

            raw_columns = [
                "symbol",
                "score",
                "benchmark_adjusted_score",
                "relative_strength_score",
                "relative_strength_label",
                "rs_3m",
                "rs_6m",
                "rs_12m",
                "larsson_score",
                "daily_trend_score",
                "weekly_trend_score",
                "monthly_trend_score",
                "pass_200d_score",
                "pass_200w_score",
                "drawdown_extension_score",
                "fundamental_component",
                "catalyst_component",
                "data_warnings",
                "notes",
            ]
            detail_values = selected_row[[column for column in raw_columns if column in selected_row.index]]
            st.dataframe(detail_values.astype(str).to_frame("value"), width="stretch")

    elif active_view == "Benchmark":
        if benchmark_row.empty:
            st.info(f"No benchmark data available for {benchmark}.")
        else:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Benchmark", benchmark)
            b2.metric("Regime", benchmark_regime)
            b3.metric("Latest", money(benchmark_close))
            b4.metric("52W Drawdown", pct(benchmark_drawdown))

            basket_history, basket_symbols_used, basket_missing_symbols = equal_weight_basket_history(
                prices,
                manual_symbols,
                benchmark,
            )
            st.caption(
                f"Seed EW Basket: {len(basket_symbols_used)} / {len(manual_symbols)} active symbols used."
            )
            if basket_missing_symbols:
                st.caption(f"Missing basket data: {', '.join(basket_missing_symbols)}")

            benchmark_history = larsson_regime_history(prices, benchmark, config)
            st.dataframe(
                pd.DataFrame(
                    {
                        "benchmark": [benchmark],
                        "larsson_regime_history": [benchmark_history["larsson_regime_value"].tolist()],
                        "larsson_regime_history_labels": [", ".join(benchmark_history["larsson_regime"].tolist())],
                    }
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "larsson_regime_history": st.column_config.BarChartColumn(
                        "Regime History",
                        width=240,
                        help="Weekly benchmark Larsson-proxy history: high bars = Bull, mid bars = Transition, low bars = Bear.",
                        y_min=0,
                        y_max=1,
                        color="#22c55e",
                    )
                },
            )
            benchmark_daily = price_history_with_indicators(prices, benchmark, config)
            benchmark_weekly = weekly_history_with_indicators(prices, benchmark, config)
            benchmark_drawdown_history = drawdown_history(prices, benchmark)
            basket_weekly = basket_history.copy()
            if not basket_weekly.empty:
                basket_weekly = (
                    basket_weekly.set_index("date")["basket_index"]
                    .resample("W-FRI")
                    .last()
                    .dropna()
                    .rename("basket_index")
                    .reset_index()
                )
            st.plotly_chart(
                price_chart(
                    benchmark_daily,
                    benchmark,
                    scaled_basket_overlay(benchmark_daily, basket_history, "close"),
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                weekly_chart(
                    benchmark_weekly,
                    benchmark,
                    scaled_basket_overlay(benchmark_weekly, basket_weekly, "weekly_close"),
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                drawdown_chart(
                    benchmark_drawdown_history,
                    benchmark,
                    basket_drawdown_overlay(basket_history),
                ),
                use_container_width=True,
            )

    elif active_view == "RRG":
        rrg_view = st.radio(
            "RRG view",
            ["Stocks", "Seed universes"],
            horizontal=True,
            key="rrg_view",
            help="Choose whether each dot is an individual stock or an equal-weight built-in seed universe.",
        )
        tail_weeks = st.slider(
            "Tail weeks",
            min_value=4,
            max_value=26,
            value=12,
            step=1,
            help=RRG_CONTROL_HELP["tail_weeks"],
        )
        ratio_window = st.slider(
            "Ratio smoothing",
            min_value=4,
            max_value=20,
            value=10,
            step=1,
            help=RRG_CONTROL_HELP["ratio_smoothing"],
        )
        momentum_window = st.slider(
            "Momentum smoothing",
            min_value=4,
            max_value=20,
            value=10,
            step=1,
            help=RRG_CONTROL_HELP["momentum_smoothing"],
        )

        if rrg_view == "Stocks":
            theme_options = seed_theme_options(universe)
            selected_themes = st.multiselect("Seed themes", theme_options, default=theme_options)
            theme_ranked = filter_ranked_by_seed_themes(ranked, selected_themes)
            symbol_options = theme_ranked["symbol"].tolist()
            default_rrg_symbols = symbol_options[:20]
            selected_rrg_symbols = st.multiselect("Symbols", symbol_options, default=default_rrg_symbols)

            if not selected_rrg_symbols:
                st.info("Select at least one seed theme and symbol to draw the RRG.")
            else:
                rrg_history = cached_rrg_history(
                    prices,
                    tuple(selected_rrg_symbols),
                    benchmark,
                    ratio_window,
                    momentum_window,
                    tail_weeks,
                )
                if rrg_history.empty:
                    st.info("No RRG rows available. Try a longer price history or fewer smoothing weeks.")
                else:
                    latest_rrg = latest_rrg_with_metadata(rrg_history, ranked)
                    selected_symbol = st.session_state.get("selected_rrg_symbol")
                    if selected_symbol not in set(selected_rrg_symbols):
                        selected_symbol = None
                        st.session_state["selected_rrg_symbol"] = None
                    if selected_symbol:
                        st.caption(f"Focused RRG symbol: {selected_symbol}. Click it again to clear the focus.")
                    else:
                        st.caption("Click a ticker dot to highlight its current point and recent path.")
                    rrg_fig = rrg_chart(rrg_history, latest_rrg, benchmark, selected_symbol=selected_symbol)
                    rrg_event = st.plotly_chart(
                        rrg_fig,
                        use_container_width=True,
                        key="rrg_stock_chart",
                        on_select="rerun",
                        selection_mode="points",
                    )
                    clicked_symbol = rrg_selection_from_event(rrg_event, rrg_fig)
                    new_selected_symbol = toggle_rrg_selection(selected_symbol, clicked_symbol)
                    if new_selected_symbol != selected_symbol:
                        st.session_state["selected_rrg_symbol"] = new_selected_symbol
                        st.rerun()
                    display_rrg = latest_rrg.copy()
                    for column in ["rs_3m", "rs_6m", "rs_12m"]:
                        display_rrg[column] = display_rrg[column] * 100
                    st.dataframe(
                        display_rrg[
                            [
                                "symbol",
                                "name",
                                "theme",
                                "quadrant",
                                "rs_ratio",
                                "rs_momentum",
                                "relative_strength_score",
                                "rs_3m",
                                "rs_6m",
                                "rs_12m",
                                "score",
                                "benchmark_adjusted_score",
                            ]
                        ].sort_values(["quadrant", "symbol"]),
                        width="stretch",
                        hide_index=True,
                        column_config=rrg_column_config(),
                    )
        else:
            seed_options = seed_rrg_universe_options()
            if not st.session_state.get("_rrg_seed_universes_all_default_migrated", False):
                st.session_state["rrg_seed_universes"] = rrg_seed_universes_after_default_migration(
                    st.session_state.get("rrg_seed_universes"),
                    seed_options,
                )
                st.session_state["_rrg_seed_universes_all_default_migrated"] = True
            selected_seed_labels = st.multiselect(
                "Seed universes",
                seed_options,
                key="rrg_seed_universes",
                help="Built-in seed presets to compare as equal-weight basket dots.",
            )
            if not selected_seed_labels:
                st.info("Select at least one seed universe to draw the RRG.")
            else:
                seed_history = cached_rrg_basket_history(
                    prices,
                    seed_universe_symbol_map(seed_frames, selected_seed_labels),
                    benchmark,
                    ratio_window,
                    momentum_window,
                    tail_weeks,
                )
                if seed_history.empty:
                    st.info("No seed RRG rows available. Try a longer price history or fewer smoothing weeks.")
                else:
                    latest_seed_history = latest_seed_rrg(seed_history)
                    selected_rrg_seed = st.session_state.get("selected_rrg_seed")
                    if selected_rrg_seed not in set(selected_seed_labels):
                        selected_rrg_seed = None
                        st.session_state["selected_rrg_seed"] = None
                    if selected_rrg_seed:
                        st.caption(f"Focused RRG seed universe: {selected_rrg_seed}. Click it again to clear the focus.")
                    else:
                        st.caption("Click a seed-universe dot to highlight its current point and recent path.")
                    seed_fig = rrg_seed_chart(seed_history, latest_seed_history, benchmark, selected_seed=selected_rrg_seed)
                    seed_event = st.plotly_chart(
                        seed_fig,
                        use_container_width=True,
                        key="rrg_seed_chart",
                        on_select="rerun",
                        selection_mode="points",
                    )
                    clicked_seed = rrg_selection_from_event(seed_event, seed_fig)
                    new_selected_seed = toggle_rrg_selection(selected_rrg_seed, clicked_seed)
                    if new_selected_seed != selected_rrg_seed:
                        st.session_state["selected_rrg_seed"] = new_selected_seed
                        st.rerun()
                    st.dataframe(
                        latest_seed_history[
                            [
                                "seed_name",
                                "quadrant",
                                "rs_ratio",
                                "rs_momentum",
                                "symbol_count",
                                "missing_count",
                                "symbols_used",
                                "symbols_missing",
                            ]
                        ].sort_values(["quadrant", "seed_name"]),
                        width="stretch",
                        hide_index=True,
                        column_config=rrg_seed_column_config(),
                    )

    elif active_view == "Theme Breadth":
        theme_options = seed_rrg_universe_options()
        st.subheader("Theme Breadth")
        st.caption(
            "Local-only participation and momentum for every fixed built-in seed universe. "
            "This is a dashboard equal-weight theme index, not Jordi Visser's/JV's index; it does not affect rankings or allocations."
        )
        st.caption(
            f"Breadth is the Friday-ended percentage above each stock's {THEME_BREADTH_SMA_DAYS}D SMA; "
            f"the line is a {THEME_BREADTH_SMOOTHING_WEEKS}-week average and ROC (Rate of Change) is {THEME_BREADTH_ROC_DAYS} trading sessions. "
            f"A confirmation requires at least {THEME_BREADTH_MIN_COVERAGE:.0%} coverage."
        )
        theme_histories: dict[str, pd.DataFrame] = {}
        theme_summaries: list[pd.DataFrame] = []
        for label in theme_options:
            theme_symbols = seed_symbols_for_labels(seed_frames, [label])
            history = cached_theme_breadth_history(
                prices_for_symbols(prices, set(theme_symbols)),
                tuple(theme_symbols),
            )
            theme_histories[label] = history
            theme_summaries.append(latest_theme_breadth_summary(history, label, len(theme_symbols)))

        summary = pd.concat(theme_summaries, ignore_index=True) if theme_summaries else pd.DataFrame()
        if summary.empty:
            st.info("No cached theme data is available yet. Press Refresh cache to download the built-in theme universes.")
        else:
            summary["theme"] = summary["seed_universe"].map(seed_universe_display_name)
            table_columns = [
                "theme",
                "date",
                "breadth",
                "breadth_4w_average",
                "breadth_change",
                "roc_50d",
                "confirmation",
                "coverage",
                "eligible_count",
                "universe_size",
                "missing_or_short_history",
            ]
            display_summary = theme_breadth_display(
                summary[[column for column in table_columns if column in summary.columns]].sort_values("theme")
            )
            table_config = theme_breadth_column_config()
            table_config["theme"] = st.column_config.TextColumn("Theme", help="Fixed built-in seed universe used for this local breadth calculation.")
            st.dataframe(display_summary, width="stretch", hide_index=True, column_config=table_config)

            focused_theme = st.selectbox(
                "Theme detail",
                theme_options,
                format_func=seed_universe_display_name,
                key="theme_breadth_focus",
                help="Choose one built-in universe for the two historical charts below.",
            )
            focused_history = theme_histories[focused_theme]
            focused_summary = latest_theme_breadth_summary(
                focused_history,
                focused_theme,
                len(seed_symbols_for_labels(seed_frames, [focused_theme])),
            ).iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Above 20D", pct(focused_summary.get("breadth")))
            m2.metric("4W Breadth Average", pct(focused_summary.get("breadth_4w_average")))
            m3.metric("Equal-Weight 50D ROC", pct(focused_summary.get("roc_50d")))
            eligible_count = pd.to_numeric(pd.Series([focused_summary.get("eligible_count")]), errors="coerce").fillna(0).iloc[0]
            universe_size = pd.to_numeric(pd.Series([focused_summary.get("universe_size")]), errors="coerce").fillna(0).iloc[0]
            unavailable_count = pd.to_numeric(pd.Series([focused_summary.get("missing_or_short_history")]), errors="coerce").fillna(0).iloc[0]
            coverage_detail = f"{int(eligible_count)} / {int(universe_size)} eligible"
            m4.metric("Coverage", pct(focused_summary.get("coverage")), coverage_detail)
            st.caption(f"**{focused_summary.get('confirmation', 'Unresolved')}** · {coverage_detail} · unavailable or short history: {int(unavailable_count)}")
            chart_left, chart_right = st.columns(2)
            chart_left.plotly_chart(theme_breadth_chart(focused_history, focused_theme), use_container_width=True)
            chart_right.plotly_chart(theme_roc_chart(focused_history, focused_theme), use_container_width=True)

    elif active_view == "Portfolio":
        portfolio_seed_options = seed_rrg_universe_options()
        selected_portfolio_seeds = st.multiselect(
            "Portfolio seed universes",
            portfolio_seed_options,
            default=default_portfolio_seed_universes(),
            key="portfolio_seed_universes",
            help="Seed presets eligible for the model portfolio.",
        )
        pc1, pc2, pc3, pc4 = st.columns(4)
        max_holdings = pc1.slider("Max holdings", min_value=4, max_value=25, value=12, step=1, help="Maximum number of stocks the allocator may hold; cash can still be retained.")
        top_per_seed = pc2.slider("Top candidates per seed", min_value=2, max_value=10, value=5, step=1, help="How many highest-ranked candidates each selected seed universe contributes to the allocation pool.")
        max_position_pct = pc3.slider("Max position", min_value=5, max_value=25, value=12, step=1, help="Largest allowed weight for any individual stock after allocation.")
        max_seed_exposure_pct = pc4.slider("Max seed exposure", min_value=20, max_value=80, value=45, step=5, help="Largest combined weight allowed for stocks from one seed universe.")

        pc5, pc6, pc7 = st.columns(3)
        sensitivity = pc5.selectbox("Allocator sensitivity", list(SENSITIVITY_PROFILES), index=1, help="Controls how readily the allocator accepts portfolio changes: Conservative needs stronger evidence; Aggressive changes more readily.")
        use_portfolio_rs = pc6.checkbox("Use relative strength", value=True, help="Blend each stock's relative performance versus the selected benchmark into its ranking score.")
        allow_weak_candidates = pc7.checkbox("Allow weak-regime candidates", value=False, help="Include candidates whose Larsson-proxy regime is Bear. By default, those candidates are excluded.")
        include_cash_rule = st.checkbox("Use benchmark-regime cash rule", value=True, help="Reserve at least 15% cash in a Transition benchmark regime and 35% in a Bear regime. Turn this off to target no regime-driven cash reserve.")
        st.caption(benchmark_regime_cash_rule_status(benchmark_regime, include_cash_rule))

        portfolio_config = PortfolioConfig(
            selected_seeds=tuple(selected_portfolio_seeds),
            max_holdings=max_holdings,
            top_candidates_per_seed=top_per_seed,
            max_position=max_position_pct / 100,
            max_seed_exposure=max_seed_exposure_pct / 100,
            use_relative_strength=use_portfolio_rs,
            include_cash_rule=include_cash_rule,
            allow_weak_regime_candidates=allow_weak_candidates,
            sensitivity=sensitivity,
        )
        active_config_hash = portfolio_config_hash(portfolio_config)
        active_seed_key = canonical_selected_seeds(portfolio_config.selected_seeds)
        portfolio_data_enabled = bool(st.session_state.get("portfolio_data_enabled", False))
        if not portfolio_data_enabled:
            st.info("Portfolio universe data is not loaded yet. Load it when you want to build the model portfolio.")
            if st.button("Load portfolio universe data"):
                st.session_state["portfolio_data_enabled"] = True
                st.rerun()
        else:
            portfolio_membership = seed_membership_frame(seed_frames, selected_portfolio_seeds)
            portfolio_symbols = set(portfolio_membership["symbol"]) if not portfolio_membership.empty else set()
            portfolio_prices = prices_for_symbols(prices, portfolio_symbols.union({benchmark}))
            available_symbols = set(portfolio_prices["symbol"]) if not portfolio_prices.empty else set()
            missing_portfolio_symbols = sorted(portfolio_symbols - available_symbols)
            if missing_portfolio_symbols:
                st.warning(
                    "Some selected portfolio symbols are not cached yet. Use Refresh cache to fetch: "
                    + ", ".join(missing_portfolio_symbols[:20])
                )
            portfolio_signal_started_at = perf_counter()
            portfolio_signal_cache_state = session_cache_state("portfolio_signals", (tuple(sorted(portfolio_symbols)), benchmark, larsson_history_weeks, len(portfolio_prices), str(portfolio_prices["date"].max())))
            portfolio_signals = cached_signal_table(portfolio_prices, benchmark, larsson_history_weeks)
            record_timing("Portfolio technical ranking", portfolio_signal_started_at, portfolio_signal_cache_state, f"{len(portfolio_symbols)} symbols")
            portfolio_signals = portfolio_signals.merge(
                portfolio_membership.drop_duplicates("symbol", keep="first"),
                on="symbol",
                how="left",
            )
            if not portfolio_signals.empty:
                portfolio_signals[["name", "theme", "notes", "seed_universe"]] = portfolio_signals[
                    ["name", "theme", "notes", "seed_universe"]
                ].fillna("")
                portfolio_signals[["fundamental_score", "catalyst_score"]] = portfolio_signals[
                    ["fundamental_score", "catalyst_score"]
                ].fillna(0.0)
            portfolio_ranked = rank_universe(
                portfolio_signals,
                RankingConfig(use_relative_strength=use_portfolio_rs),
            )

            portfolio_rrg_history = cached_rrg_basket_history(
                prices,
                seed_universe_symbol_map(seed_frames, selected_portfolio_seeds),
                benchmark,
                RRGConfig().ratio_window_weeks,
                RRGConfig().momentum_window_weeks,
                RRGConfig().tail_weeks,
            )
            portfolio_rrg_summary = rrg_seed_summary(portfolio_rrg_history)
            candidates = build_portfolio_candidates(portfolio_ranked, portfolio_membership, portfolio_rrg_summary, portfolio_config)
            raw_target = allocate_portfolio(candidates, portfolio_config, benchmark_regime)
            previous_portfolio, previous_metadata = load_matching_portfolio_snapshot(
                benchmark=benchmark,
                selected_seeds=active_seed_key,
                config_hash=active_config_hash,
                db_path=DEFAULT_DB_PATH,
            )
            latest_portfolio, latest_metadata = load_latest_portfolio_snapshot(DEFAULT_DB_PATH)
            applied_target = apply_sensitivity_to_target(previous_portfolio, raw_target, portfolio_config)
            changes, suppressed = compare_portfolios(previous_portfolio, raw_target, portfolio_config)
            portfolio_holding_symbols = (
                applied_target.loc[applied_target["symbol"].astype(str).str.upper() != "CASH", "symbol"]
                .astype(str)
                .str.upper()
                .tolist()
                if not applied_target.empty and "symbol" in applied_target.columns
                else []
            )
            stock_rrg_history = cached_rrg_history(
                portfolio_prices,
                tuple(portfolio_holding_symbols),
                benchmark,
                RRGConfig().ratio_window_weeks,
                RRGConfig().momentum_window_weeks,
                RRGConfig().tail_weeks,
            ) if portfolio_holding_symbols else pd.DataFrame()
            stock_rrg_allocation = portfolio_rrg_allocation_frame(applied_target, stock_rrg_history)

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Holdings", int((applied_target["symbol"] != "Cash").sum()) if not applied_target.empty else 0)
            p2.metric("Cash", pct(applied_target.loc[applied_target["symbol"] == "Cash", "weight"].sum() if not applied_target.empty else 1.0))
            p3.metric("Sensitivity", sensitivity)
            p4.metric("Config", active_config_hash)
            if previous_metadata:
                st.caption(f"Matching baseline found: {previous_metadata.get('created_at')}")
            elif latest_metadata:
                st.caption("No matching baseline. Latest saved baseline belongs to different settings.")
            else:
                st.caption("No matching baseline.")

            if st.button("Save current allocation as baseline"):
                saved_changes, _ = compare_portfolios(previous_portfolio, applied_target, portfolio_config)
                snapshot_id = save_portfolio_snapshot(
                    applied_target,
                    portfolio_snapshot_metadata(benchmark, benchmark_regime, portfolio_config),
                    saved_changes,
                    DEFAULT_DB_PATH,
                )
                st.success(f"Saved portfolio snapshot #{snapshot_id}.")

            target_display = applied_target.copy()
            for column in ["weight", "drawdown_52w", "extension_sma50d"]:
                if column in target_display.columns:
                    target_display[column] = target_display[column] * 100
            st.subheader("Applied Allocation After Sensitivity Rules")
            st.dataframe(
                target_display,
                width="stretch",
                hide_index=True,
                column_config=portfolio_column_config(),
            )

            cc1, cc2, cc3 = st.columns(3)
            cc1.plotly_chart(allocation_pie_chart(applied_target, "symbol", "Allocation by Holding"), use_container_width=True)
            cc2.plotly_chart(allocation_pie_chart(applied_target, "seed_universe", "Allocation by Seed Universe"), use_container_width=True)
            cc3.plotly_chart(allocation_pie_chart(stock_rrg_allocation, "rrg_quadrant", "Allocation by Stock RRG Quadrant"), use_container_width=True)

            st.subheader("Changes Since Matching Baseline")
            change_display = changes.copy()
            for column in ["previous_weight", "target_weight", "delta"]:
                if column in change_display.columns:
                    change_display[column] = change_display[column] * 100
            st.dataframe(change_display, width="stretch", hide_index=True, column_config=portfolio_change_column_config())

            st.subheader("Suppressed Changes")
            suppressed_display = suppressed.copy()
            for column in ["previous_weight", "target_weight", "delta"]:
                if column in suppressed_display.columns:
                    suppressed_display[column] = suppressed_display[column] * 100
            st.dataframe(suppressed_display, width="stretch", hide_index=True, column_config=portfolio_change_column_config())

            st.subheader("RRG Strategy Signals")
            st.dataframe(portfolio_rrg_summary, width="stretch", hide_index=True)

            with st.expander("Raw Target Allocation"):
                raw_display = raw_target.copy()
                for column in ["weight", "drawdown_52w", "extension_sma50d"]:
                    if column in raw_display.columns:
                        raw_display[column] = raw_display[column] * 100
                st.dataframe(raw_display, width="stretch", hide_index=True, column_config=portfolio_column_config())

            with st.expander("Candidate Pool"):
                candidate_display = candidates.copy()
                for column in ["drawdown_52w", "extension_sma50d"]:
                    if column in candidate_display.columns:
                        candidate_display[column] = candidate_display[column] * 100
                st.dataframe(candidate_display, width="stretch", hide_index=True)

            with st.expander("Saved Rebalance Log"):
                show_all_logs = st.checkbox("Show all saved rebalance logs", value=False)
                matching_snapshot_id = previous_metadata.get("snapshot_id") if previous_metadata else None
                log = load_portfolio_rebalance_log(
                    DEFAULT_DB_PATH,
                    snapshot_id=None if show_all_logs else matching_snapshot_id,
                )
                for column in ["previous_weight", "target_weight", "delta"]:
                    if column in log.columns:
                        log[column] = log[column] * 100
                st.dataframe(log, width="stretch", hide_index=True, column_config=portfolio_change_column_config())

    elif active_view == "Chatgptfolio Lab":
        st.title("Chatgptfolio Lab")
        st.caption("Auditable OpenAI research and paper trading. This does not place real trades or provide investment advice.")
        lab_candidate_universe = st.radio("Candidate universe", ["All seed tickers", "Current selected seed"], horizontal=True, key="lab_candidate_universe", help="All seed tickers technically ranks the deduplicated union before research is bounded to the top 30.")
        lab_universe = all_seed if lab_candidate_universe == "All seed tickers" else universe
        lab_signal_prices = prices_for_symbols(prices, set(lab_universe["symbol"]).union({benchmark}))
        lab_signal_started_at = perf_counter()
        lab_signal_cache_state = session_cache_state("lab_signals", (lab_candidate_universe, tuple(sorted(lab_universe["symbol"])), benchmark, larsson_history_weeks, len(lab_signal_prices), str(lab_signal_prices["date"].max())))
        lab_signals = cached_signal_table(lab_signal_prices, benchmark, larsson_history_weeks)
        record_timing("Lab technical ranking", lab_signal_started_at, lab_signal_cache_state, f"{len(lab_universe)} symbols")
        lab_signals = lab_signals.merge(lab_universe, on="symbol", how="left")
        for column in ["name", "theme", "notes"]:
            if column in lab_signals:
                lab_signals[column] = lab_signals[column].fillna("")
        for column in ["fundamental_score", "catalyst_score"]:
            if column in lab_signals:
                lab_signals[column] = lab_signals[column].fillna(0.0)
        lab_ranked = rank_universe(lab_signals, RankingConfig(use_relative_strength=use_relative_strength))
        if require_200d:
            lab_ranked = lab_ranked[lab_ranked["pass_200d"] == True].copy()
        if require_200w:
            lab_ranked = lab_ranked[lab_ranked["pass_200w"] == True].copy()
        lab_ranked = lab_ranked.reset_index(drop=True)
        lab_ranked["rank"] = range(1, len(lab_ranked) + 1)
        lab_candidates = lab_ranked.head(30).copy()
        lab_state = st.session_state.get("lab_state", {})
        # Every approved proposal starts an isolated experiment, so no prior
        # experiment's positions are inputs to a new model decision.
        lab_held_symbols: list[str] = []

        st.subheader("1. Run setup")
        lc1, lc2, lc3, lc4 = st.columns(4)
        selected_model = lc1.selectbox("OpenAI model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL), help="Model choice is recorded per run; there is no automatic fallback.")
        as_of_date = date.today()
        lab_theme_breadth = theme_breadth_packet_rows(
            prices_for_symbols(prices, set(all_seed["symbol"])),
            seed_frames,
            as_of_date,
        )
        lc2.metric("Research date", as_of_date.isoformat(), help="Live Lab decisions are current-date research only. Historical replay stays disabled until every source has an archived point-in-time packet.")
        paper_capital = 100_000.0
        lc3.metric("Experiment capital", money(paper_capital), help="Every approved run opens a separate, independent paper experiment at this nominal capital.")
        slippage_bps = lc4.number_input("Slippage (bps)", min_value=0.0, value=5.0, step=1.0)
        lc5, lc6, lc7 = st.columns(3)
        lab_max_holdings = lc5.slider("Maximum holdings", 1, 15, 12)
        lab_max_position = lc6.slider("Maximum position (%)", 1, 25, 12) / 100
        commission_bps = lc7.number_input("Commission (bps)", min_value=0.0, value=0.0, step=0.5)
        lc8, lc9 = st.columns(2)
        lab_max_sector = lc8.slider("Maximum sector exposure (%)", 10, 80, 40, step=5) / 100
        refresh_research = lc9.checkbox("Refresh SEC, FRED, and filing evidence", value=True, help="Refreshes the broader public research packet when an analysis is run. Recent ticker news is always refreshed for an analysis.")
        lab_config = LabConfig(model=selected_model, max_holdings=lab_max_holdings, max_position=lab_max_position, max_sector_exposure=lab_max_sector, paper_capital=paper_capital, slippage_bps=slippage_bps, commission_bps=commission_bps)

        st.subheader("Connection & Data Health")
        last_model_error = st.session_state.get("lab_credential_error")
        credential = credential_status(last_model_error)
        health_one, health_two, health_three = st.columns(3)
        health_one.metric("OpenAI credential", credential.title())
        cached_all_seed_count = cached_symbol_count(all_seed["symbol"], DEFAULT_DB_PATH)
        health_two.metric("All-seeds cache", f"{cached_all_seed_count}/{len(all_seed)} tickers")
        health_three.metric("Price cache", latest_cache_timestamp(DEFAULT_DB_PATH))
        if cached_all_seed_count < len(all_seed):
            st.caption("First all-seeds load may take longer; subsequent runs reuse the local DuckDB price cache.")
        if credential == "missing":
            st.warning("No credential is available to the app. The key is never displayed or stored in DuckDB.")
        elif credential == "rejected":
            st.error("The last OpenAI request was rejected. Replace the credential, then rerun; the key is never displayed here.")

        st.subheader("2. Research packet")
        packet = research_packet(lab_candidates, as_of_date, benchmark, lab_config, refresh_research=False, prices=lab_signal_prices, held_symbols=lab_held_symbols, theme_breadth=lab_theme_breadth)
        display_packet = lab_state.get("packet", packet)
        source_status, coverage_status, candidate_status = st.columns(3)
        source_status.metric("Price source", display_packet["coverage"]["prices"].title())
        coverage_status.metric("Research completeness", "Ready to refresh" if refresh_research else "Cached prices only")
        candidate_status.metric("Candidates", len(lab_candidates))
        st.caption(f"{len(lab_ranked)} technically eligible tickers were ranked; only the top {len(lab_candidates)} enter the bounded research packet. Missing sources remain explicit and are never filled by model inference.")
        st.subheader("Research Overview")
        context = packet_context_metadata(display_packet, lab_config, lab_state.get("api_usage"))
        context_one, context_two, context_three = st.columns(3)
        context_one.metric("Estimated GPT input", f"{int(context['estimated_input_tokens']):,} tokens")
        context_two.metric("Packet reduction", f"{float(context['estimated_packet_reduction_pct']):.0f}%")
        context_three.metric("Seed breadth", f"{int(context['theme_breadth_count'])} themes")
        health = evidence_health(display_packet)
        severity_dot_strip(health, "Evidence health")
        st.dataframe(pd.DataFrame(health), width="stretch", hide_index=True)
        macro = display_packet.get("macro", {})
        if macro.get("series"):
            st.caption(f"Macro regime: **{macro.get('regime', 'unavailable')}**")
            st.dataframe(pd.DataFrame(macro["series"]).T, width="stretch", hide_index=False)
        breadth_preview = pd.DataFrame(display_packet.get("theme_breadth", []))
        if not breadth_preview.empty:
            breadth_columns = [
                "seed_universe", "observation_date", "breadth", "breadth_4w_average",
                "breadth_change", "roc_50d", "confirmation", "coverage",
                "eligible_count", "universe_size",
            ]
            st.caption("Latest fixed-seed breadth supplied to GPT as advisory context.")
            st.dataframe(
                theme_breadth_display(breadth_preview[[column for column in breadth_columns if column in breadth_preview]]),
                width="stretch",
                hide_index=True,
                column_config=theme_breadth_column_config(),
            )
        correlation_preview = display_packet.get("correlations", {})
        if correlation_preview:
            st.caption(f"Correlation coverage: {correlation_preview.get('daily_window', 50)} trading days / {correlation_preview.get('weekly_window', 26)} weeks. High-pair threshold: 0.80.")
            high_pairs = pd.DataFrame(correlation_preview.get("high_correlation_pairs", []))
            if not high_pairs.empty:
                st.warning("High-correlation candidate pairs are a diversification warning, not an automatic exclusion.")
                st.dataframe(high_pairs, width="stretch", hide_index=True)
        with st.expander("Immutable packet audit (raw JSON)"):
            st.json(display_packet)
        with st.expander("Compact packet sent to GPT"):
            st.json(compact_model_packet(display_packet))

        st.subheader("3. Candidate review")
        review = lab_ranked.copy()
        if "seed_memberships" in review:
            membership_options = sorted({seed for value in review["seed_memberships"].dropna() for seed in str(value).split(" · ")})
            membership_filter = st.multiselect("Filter all-seeds review by membership", membership_options)
            if membership_filter:
                review = review[review["seed_memberships"].map(lambda value: bool(set(str(value).split(" · ")).intersection(membership_filter)))]
        review_columns = [column for column in ["rank", "symbol", "name", "seed_memberships", "theme", "latest_close", "score", "benchmark_adjusted_score", "relative_strength_score", "larsson_regime", "daily_trend", "weekly_trend", "monthly_trend", "data_warnings"] if column in review]
        st.dataframe(review[review_columns], width="stretch", hide_index=True)

        if st.button("Run ChatGPT paper-portfolio analysis", type="primary"):
            research_started_at = perf_counter()
            packet = research_packet(lab_candidates, as_of_date, benchmark, lab_config, refresh_research=refresh_research, refresh_news=True, prices=lab_signal_prices, held_symbols=lab_held_symbols, theme_breadth=lab_theme_breadth)
            record_timing("Research collection", research_started_at, "live refresh" if refresh_research else "packet-only", f"{len(packet.get('evidence', []))} evidence records")
            for source, duration_ms in packet.get("connector_timings_ms", {}).items():
                record_duration(f"{str(source).title()} connector", float(duration_ms), "sequential public source", str(packet.get("coverage", {}).get(source, "unknown")))
            packet_hash = save_lab_research_snapshot(packet, DEFAULT_DB_PATH)
            save_lab_research_details(packet_hash, packet, DEFAULT_DB_PATH)
            readiness = preflight_packet(packet, credential)
            if not readiness.valid:
                run_id = save_lab_run({
                    "as_of_date": as_of_date, "benchmark": benchmark, "model": selected_model,
                    "status": "preflight_blocked", "packet_hash": packet_hash,
                    "config_json": json.dumps(lab_config_payload(lab_config)),
                    "context_metadata_json": json.dumps(packet_context_metadata(packet, lab_config)),
                    "error_message": "; ".join(readiness.errors),
                }, DEFAULT_DB_PATH)
                st.error(f"Analysis was not run (record #{run_id}): " + "; ".join(readiness.errors))
                return
            try:
                model_started_at = perf_counter()
                model_decision, raw_response, returned_model, api_usage = OpenAIPortfolioModel(selected_model).decide(packet, lab_config)
                record_timing("OpenAI proposal", model_started_at, "live API", selected_model)
                previous_weights = pd.DataFrame(columns=["symbol", "weight"])
                decision = optimize_decision(model_decision, lab_candidates, lab_config, previous_weights)
                validation = validate_decision(decision, lab_candidates, lab_config, previous_weights, packet=packet)
                status = "proposal_ready" if validation.valid else "validation_failed"
                run_id = save_lab_run({
                    "as_of_date": as_of_date,
                    "benchmark": benchmark,
                    "model": selected_model,
                    "returned_model": returned_model,
                    "status": status,
                    "packet_hash": packet_hash,
                    "prompt_hash": decision_hash({
                        "prompt": prompt_for_packet(packet, lab_config),
                        "schema": decision_schema_for_packet(packet),
                        "model": selected_model,
                    }),
                    "config_json": json.dumps(lab_config_payload(lab_config)),
                    "request_settings_json": json.dumps({"reasoning_effort": "high" if selected_model == DEFAULT_MODEL else "medium", "response_format": "strict_json_schema"}),
                    "context_metadata_json": json.dumps(packet_context_metadata(packet, lab_config, api_usage)),
                    "response_json": json.dumps({"model_decision": model_decision, "optimized_decision": decision}),
                    "raw_response": raw_response,
                    "validation_json": json.dumps({"valid": validation.valid, "errors": validation.errors, "warnings": validation.warnings}),
                }, DEFAULT_DB_PATH)
                st.session_state.pop("lab_credential_error", None)
                st.session_state["lab_state"] = {"run_id": run_id, "decision": decision, "validation": validation, "model": selected_model, "packet": packet, "config": lab_config_payload(lab_config), "as_of_date": as_of_date.isoformat(), "api_usage": api_usage}
                st.rerun()
            except Exception as exc:
                st.session_state["lab_credential_error"] = str(exc)
                run_id = save_lab_run({"as_of_date": as_of_date, "benchmark": benchmark, "model": selected_model, "status": "model_failed", "packet_hash": packet_hash, "config_json": json.dumps(lab_config_payload(lab_config)), "context_metadata_json": json.dumps(packet_context_metadata(packet, lab_config)), "error_message": str(exc)}, DEFAULT_DB_PATH)
                st.error(f"Analysis was not run (record #{run_id}): {exc}")

        st.subheader("4. ChatGPT proposal")
        if not lab_state:
            st.info("No proposal yet. Run the analysis after reviewing the research packet.")
        else:
            validation = lab_state["validation"]
            decision = lab_state["decision"]
            state1, state2, state3 = st.columns(3)
            state1.metric("Model", lab_state["model"])
            state2.metric("Validation", "Ready" if validation.valid else "Failed")
            state3.metric("Run ID", lab_state["run_id"])
            usage = lab_state.get("api_usage", {})
            if usage:
                st.caption(
                    f"API tokens: {int(usage.get('input_tokens', 0)):,} input · "
                    f"{int(usage.get('cached_input_tokens', 0)):,} cached input · "
                    f"{int(usage.get('output_tokens', 0)):,} output · "
                    f"{int(usage.get('reasoning_tokens', 0)):,} reasoning"
                )
            if validation.valid:
                st.success("Proposal passed deterministic limits. Review before approving paper orders.")
            else:
                st.error("Proposal is blocked: " + "; ".join(validation.errors))
            if decision.get("abstain_reason"):
                st.warning("Model abstained: " + str(decision["abstain_reason"]))
            for warning in validation.warnings:
                st.warning("Diversification: " + warning)
            st.write(decision.get("portfolio_thesis", ""))
            optimization = decision.get("optimization", {})
            if optimization:
                st.caption(f"Deterministic allocator: {optimization.get('allocated_equity', 0):.1%} equity / {decision.get('cash_weight', 0):.1%} cash. Binding constraints: {', '.join(optimization.get('binding_constraints', [])) or 'none'}.")
            proposal = pd.DataFrame(decision.get("holdings", []))
            if proposal.empty:
                st.info("No holdings proposed.")
            else:
                entry_lookup = {str(candidate.get("symbol", "")): candidate.get("entry", {}) for candidate in lab_state.get("packet", {}).get("candidates", [])}
                proposal["entry_label"] = proposal["symbol"].map(lambda symbol: entry_lookup.get(str(symbol), {}).get("entry_label", "Unresolved"))
                proposal["preferred_entry_zone"] = proposal["symbol"].map(lambda symbol: "–" if not entry_lookup.get(str(symbol), {}).get("preferred_entry_low") else f"${entry_lookup[str(symbol)]['preferred_entry_low']:.2f}–${entry_lookup[str(symbol)]['preferred_entry_high']:.2f}")
                st.dataframe(proposal, width="stretch", hide_index=True)

            st.subheader("Entry and diversification guidance")
            deterministic = pd.DataFrame([
                {"symbol": candidate.get("symbol"), **candidate.get("entry", {}), **{key: value for key, value in candidate.get("correlation", {}).items() if key in {"average_daily_correlation", "max_daily_correlation", "average_correlation_to_held", "cluster", "status"}}}
                for candidate in lab_state.get("packet", {}).get("candidates", [])
                if str(candidate.get("symbol", "")) in set(proposal.get("symbol", []))
            ])
            if not deterministic.empty:
                st.dataframe(deterministic, width="stretch", hide_index=True)
            correlations = lab_state.get("packet", {}).get("correlations", {})
            matrix = pd.DataFrame(correlations.get("daily_matrix", {}))
            if not matrix.empty:
                matrix = matrix.reindex(index=matrix.columns, columns=matrix.columns)
                st.plotly_chart(go.Figure(data=go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True, colorbar={"title": "50D corr"})).update_layout(height=440, margin={"l": 10, "r": 10, "t": 20, "b": 10}), use_container_width=True, config={"displayModeBar": False})

            risk_items = []
            for flag in decision.get("portfolio_risks", []):
                if isinstance(flag, dict):
                    risk_items.append({"symbol": "Portfolio", **flag})
            for holding in decision.get("holdings", []):
                if isinstance(holding, dict):
                    for flag in holding.get("risk_flags", []):
                        if isinstance(flag, dict):
                            risk_items.append({"symbol": holding.get("symbol", ""), **flag})
            st.subheader("Investment risk")
            severity_dot_strip(risk_items, "Model-assessed risks")
            if risk_items:
                st.dataframe(pd.DataFrame(risk_items), width="stretch", hide_index=True)
            else:
                st.info("No model risk flags were returned for this run.")

            st.subheader("Candidate evidence")
            packet_evidence = pd.DataFrame(lab_state.get("packet", {}).get("evidence", []))
            for candidate in lab_state.get("packet", {}).get("candidates", []):
                symbol = candidate.get("symbol", "")
                with st.expander(f"{symbol} — score {candidate.get('combined_score', 'n/a')}"):
                    st.json({key: candidate.get(key) for key in ["name", "seed_memberships", "sector", "technical_score", "fundamental_score", "combined_score", "fundamental_factors", "entry", "correlation"] if key in candidate})
                    rows = packet_evidence[packet_evidence["symbol"] == symbol] if not packet_evidence.empty and "symbol" in packet_evidence else pd.DataFrame()
                    if rows.empty:
                        st.warning("No refreshed company evidence was included; treat this candidate as unresolved.")
                    else:
                        st.dataframe(rows[[column for column in ["category", "title", "publisher", "source_tier", "source_domain", "published_at", "status", "source_url"] if column in rows]], width="stretch", hide_index=True)

            st.subheader("5. Paper rebalance")
            prior_weights = pd.DataFrame(columns=["symbol", "weight"])
            orders = paper_orders(prior_weights, decision, lab_config)
            if orders.empty:
                st.info("No paper orders proposed.")
            else:
                orders_display = orders.copy()
                for column in ["previous_weight", "target_weight"]:
                    orders_display[column] = orders_display[column] * 100
                st.dataframe(orders_display, width="stretch", hide_index=True)
                if validation.valid and st.button("Approve paper rebalance", type="primary"):
                    rebalance_id = approve_paper_rebalance(
                        int(lab_state["run_id"]), decision, lab_state.get("config", lab_config_payload(lab_config)),
                        date.fromisoformat(lab_state.get("as_of_date", as_of_date.isoformat())), db_path=DEFAULT_DB_PATH,
                    )
                    st.success(f"Independent $100,000 experiment approved (rebalance #{rebalance_id}). It will fill at the next cached market open; no broker was contacted.")

            with st.expander("Decision audit"):
                st.json(decision)

            with st.expander("Research evidence used"):
                evidence = pd.DataFrame(lab_state.get("packet", {}).get("evidence", []))
                if evidence.empty:
                    st.info("No refreshed external evidence was included in this run.")
                else:
                    st.dataframe(evidence[[column for column in ["symbol", "category", "title", "publisher", "source_tier", "source_domain", "published_at", "status", "source_url"] if column in evidence]], width="stretch", hide_index=True)

        st.subheader("6. Audit & performance")
        runs = load_lab_runs(DEFAULT_DB_PATH)
        if runs.empty:
            st.info("No prior lab runs saved.")
        else:
            st.dataframe(runs, width="stretch", hide_index=True)
            st.caption("Runs preserve immutable packets and source timestamps. Historical model replay is disabled until every source is archived with point-in-time availability metadata.")

    elif active_view == "Paper Portfolio":
        st.title("Paper Portfolio")
        st.caption("Every approved proposal creates an isolated $100,000 paper experiment. No experiment shares cash, positions, fills, or NAV with another.")
        if st.button("Refresh market data", icon=":material/refresh:", help="Fetch current prices for the paper holdings and benchmarks, then process any eligible pending paper rebalance."):
            st.session_state["paper_portfolio_refresh_requested"] = True
            st.rerun()
        valuation_started_at = perf_counter()
        completed_rebalances = execute_pending_paper_rebalances(prices, DEFAULT_DB_PATH)
        experiments = load_paper_experiments(DEFAULT_DB_PATH)
        for portfolio_id in experiments.get("portfolio_id", pd.Series(dtype=int)).tolist():
            rebuild_paper_nav(prices, int(portfolio_id), DEFAULT_DB_PATH, force=bool(completed_rebalances))
        record_timing(
            "Paper NAV refresh", valuation_started_at,
            "rebuilt after fill" if completed_rebalances else "valuation fingerprint reused when unchanged",
            f"{len(completed_rebalances)} new fills",
        )
        experiments = load_paper_experiments(DEFAULT_DB_PATH)
        st.subheader("Experiments")
        if experiments.empty:
            st.info("No independent experiment exists yet. Approve a validated Lab proposal to create one; it will remain pending until its next cached market open.")
        else:
            experiment_table = experiments.copy()
            experiment_table["pending_state"] = experiment_table["has_pending"].map(lambda pending: "Pending next open" if pending else "Filled")
            experiment_table["return"] = experiment_table["return"].map(pct)
            experiment_table["drawdown"] = experiment_table["drawdown"].map(pct)
            experiment_table["nav"] = experiment_table["nav"].map(money)
            experiment_table = experiment_table.rename(columns={"name": "experiment", "sensitivity": "sensitivity", "basket_description": "basket"})
            st.dataframe(
                experiment_table[[column for column in ["experiment", "sensitivity", "basket", "model", "run_id", "status", "approval_date", "fill_date", "nav", "return", "drawdown", "holding_count", "pending_state"] if column in experiment_table]],
                width="stretch",
                hide_index=True,
                column_config={
                    "experiment": st.column_config.TextColumn("Experiment"),
                    "sensitivity": st.column_config.TextColumn("Sensitivity", help="Frozen turnover sensitivity from the approved Lab run."),
                    "basket": st.column_config.TextColumn("Basket", help="Brief summary of the largest themes in the approved holdings."),
                },
            )

            experiment_options = experiments["portfolio_id"].astype(int).tolist()
            experiment_names = {
                int(row["portfolio_id"]): paper_experiment_display_name(row)
                for _, row in experiments.iterrows()
            }
            selected_experiments = st.multiselect(
                "Compare active experiments",
                experiment_options,
                default=experiment_options,
                format_func=lambda portfolio_id: experiment_names[int(portfolio_id)],
                help="Each experiment is normalized to its own $100,000 start. Shared market references use the earliest selected filled experiment.",
            )
            selected_summary = experiments[experiments["portfolio_id"].isin(selected_experiments)]
            comparison_warning = paper_comparison_start_warning(selected_summary)
            if comparison_warning:
                st.warning(comparison_warning)
            if not selected_experiments:
                st.info("Select one or more experiments to compare their performance.")
            else:
                comparison = go.Figure()
                plotted = False
                comparison_dashboards: dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
                for portfolio_id in selected_experiments:
                    nav, benchmarks, rebalances, _ = load_paper_portfolio_dashboard(int(portfolio_id), DEFAULT_DB_PATH)
                    comparison_dashboards[int(portfolio_id)] = (nav, benchmarks, rebalances)
                    normalized_nav = normalize_paper_series(nav)
                    if normalized_nav.empty:
                        continue
                    experiment_name = experiment_names[int(portfolio_id)]
                    comparison.add_trace(go.Scatter(x=normalized_nav["date"], y=normalized_nav["normalized_nav"], name=experiment_name, line={"width": 3}))
                    plotted = True
                    executed = rebalances[rebalances["status"] == "executed"] if not rebalances.empty else rebalances
                    if not executed.empty:
                        markers = normalized_nav.merge(executed[["effective_date", "rebalance_id"]], left_on="date", right_on="effective_date", how="inner")
                        if not markers.empty:
                            comparison.add_trace(go.Scatter(x=markers["date"], y=markers["normalized_nav"], mode="markers", name=f"{experiment_name} fill", marker={"size": 9, "symbol": "diamond", "color": "#f97316"}, text=[f"Rebalance #{value}" for value in markers["rebalance_id"]]))
                benchmark_anchor_id = paper_comparison_benchmark_anchor(selected_summary)
                if benchmark_anchor_id is not None and benchmark_anchor_id in comparison_dashboards:
                    _, anchor_benchmarks, _ = comparison_dashboards[benchmark_anchor_id]
                    for symbol, normalized_benchmark in shared_market_benchmark_series(anchor_benchmarks).items():
                        comparison.add_trace(go.Scatter(x=normalized_benchmark["date"], y=normalized_benchmark["normalized_nav"], name=symbol, line={"width": 1.25, "dash": "dot"}, visible=paper_chart_trace_visibility(symbol)))
                if plotted:
                    comparison.update_layout(height=460, margin={"l": 10, "r": 10, "t": 25, "b": 10}, yaxis_title="Normalized value ($100,000 start)")
                    st.plotly_chart(comparison, use_container_width=True, config={"displayModeBar": False})
                    if benchmark_anchor_id is not None:
                        anchor = selected_summary[selected_summary["portfolio_id"] == benchmark_anchor_id].iloc[0]
                        anchor_date = pd.to_datetime(anchor["fill_date"]).date().isoformat()
                        st.caption(f"Shared SPY, QQQ, GLD, and BTC-USD references are anchored to {experiment_names[benchmark_anchor_id]}'s {anchor_date} fill. TECHNICAL is run-specific and appears only in Experiment detail.")
                    else:
                        st.caption("Shared market references begin once a selected experiment has filled. TECHNICAL is run-specific and appears only in Experiment detail.")
                else:
                    st.info("The selected experiments are pending. Their comparison series will begin at the eligible cached market open.")

            selected_detail_id = st.selectbox(
                "Experiment detail",
                experiment_options,
                format_func=lambda portfolio_id: experiment_names[int(portfolio_id)],
            )
            detail = experiments[experiments["portfolio_id"] == selected_detail_id].iloc[0]
            nav, benchmarks, rebalances, positions = load_paper_portfolio_dashboard(int(selected_detail_id), DEFAULT_DB_PATH)
            fills = load_paper_fills(int(selected_detail_id), DEFAULT_DB_PATH)
            executed = rebalances[rebalances["status"] == "executed"] if not rebalances.empty else rebalances
            pending = rebalances[rebalances["status"] == "pending"] if not rebalances.empty else rebalances
            st.subheader(experiment_names[int(selected_detail_id)])
            st.caption("The sensitivity and basket description are derived from this experiment's frozen approval; the run ID, decision, and audit packet remain unchanged.")
            metric_one, metric_two, metric_three, metric_four = st.columns(4)
            metric_one.metric("NAV", money(detail["nav"]), pct(detail["return"]))
            metric_two.metric("Max drawdown", pct(detail["drawdown"]))
            metric_three.metric("Holdings", str(int(detail["holding_count"])))
            metric_four.metric("State", "Pending next open" if bool(detail["has_pending"]) else "Filled")
            paper_holdings = paper_holding_display_frame(positions, prices, all_seed)
            if not paper_holdings.empty:
                paper_rrg_regime, paper_rrg_sentence = paper_rrg_regime_analysis(prices, paper_holdings, benchmark)
                allocation_column, rrg_column = st.columns(2)
                allocation_column.plotly_chart(
                    paper_value_pie_chart(paper_holdings, "label", "value", "Current holdings by ticker"),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                rrg_column.plotly_chart(
                    paper_value_pie_chart(paper_rrg_regime, "rrg_quadrant", "value", "Current RRG regime"),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.caption(paper_rrg_sentence)
            if nav.empty:
                st.info("This experiment has not filled yet. Its frozen decision will remain pending until every required holding has a cached opening price after the research date.")
                if not pending.empty:
                    st.dataframe(pending[[column for column in ["rebalance_id", "run_id", "approved_at", "status", "note"] if column in pending]], width="stretch", hide_index=True)
            else:
                detail_chart = go.Figure()
                normalized_nav = normalize_paper_series(nav)
                detail_chart.add_trace(go.Scatter(x=normalized_nav["date"], y=normalized_nav["normalized_nav"], name="Experiment", line={"width": 3, "color": "#2563eb"}))
                for symbol, benchmark_rows in benchmarks.groupby("symbol") if not benchmarks.empty else []:
                    normalized_benchmark = normalize_paper_series(benchmark_rows)
                    if not normalized_benchmark.empty:
                        detail_chart.add_trace(go.Scatter(x=normalized_benchmark["date"], y=normalized_benchmark["normalized_nav"], name=str(symbol), line={"width": 1.5}, visible=paper_chart_trace_visibility(symbol)))
                if not executed.empty:
                    markers = normalized_nav.merge(executed[["effective_date", "rebalance_id"]], left_on="date", right_on="effective_date", how="inner")
                    if not markers.empty:
                        detail_chart.add_trace(go.Scatter(x=markers["date"], y=markers["normalized_nav"], mode="markers", name="Fill", marker={"size": 10, "symbol": "diamond", "color": "#f97316"}, text=[f"Rebalance #{value}" for value in markers["rebalance_id"]]))
                detail_chart.update_layout(height=430, margin={"l": 10, "r": 10, "t": 25, "b": 10}, yaxis_title="Normalized value ($100,000 start)")
                st.plotly_chart(detail_chart, use_container_width=True, config={"displayModeBar": False})
            st.subheader("Current holdings")
            if positions.empty:
                st.info("No filled positions yet.")
            else:
                st.dataframe(positions, width="stretch", hide_index=True)
            st.subheader("Fill audit trail")
            if fills.empty:
                st.info("No fills recorded yet.")
            else:
                st.dataframe(fills, width="stretch", hide_index=True)
            st.subheader("Rebalance timeline and frozen evidence")
            if rebalances.empty:
                st.info("No approvals recorded.")
            else:
                st.caption(f"Evidence / run link: Chatgptfolio Lab run #{int(detail['run_id'])}; frozen packet {str(rebalances.iloc[0].get('packet_hash', 'unavailable'))[:16]}…")
                st.dataframe(rebalances[[column for column in ["rebalance_id", "run_id", "approved_at", "effective_date", "status", "starting_nav", "ending_cash", "model", "packet_hash", "note"] if column in rebalances]], width="stretch", hide_index=True)

    elif active_view == "System Health":
        st.title("System Health")
        st.caption("Local-only readiness and data provenance. API secrets are never displayed, logged, or stored in DuckDB.")
        latest_runs = load_lab_runs(DEFAULT_DB_PATH)
        latest_error = str(latest_runs.iloc[0]["error_message"]) if not latest_runs.empty and pd.notna(latest_runs.iloc[0].get("error_message")) else None
        status_one, status_two, status_three = st.columns(3)
        status_one.metric("OpenAI credential", credential_status(latest_error).title())
        status_two.metric("Price cache", latest_cache_timestamp(DEFAULT_DB_PATH))
        status_three.metric("All-seeds readiness", f"{len(all_seed)} unique tickers")
        st.subheader("Local performance diagnostics")
        performance = st.session_state.get("performance_diagnostics", {})
        if performance:
            st.dataframe(pd.DataFrame(performance.values()), width="stretch", hide_index=True)
        else:
            st.info("Open a dashboard view or run a Lab analysis to collect local timing diagnostics for this browser session.")
        source_health = load_source_health(DEFAULT_DB_PATH)
        st.subheader("Source health")
        if source_health.empty:
            st.info("No research packet has been stored yet. Run a refreshed Lab analysis to establish connector health.")
        else:
            st.dataframe(source_health, width="stretch", hide_index=True)
        st.subheader("Model attempt diagnostics")
        if latest_runs.empty:
            st.info("No model attempts have been recorded.")
        else:
            st.dataframe(latest_runs, width="stretch", hide_index=True)
        st.caption("Historical model replay is intentionally unavailable. Only frozen packets with evidence available before the decision date may later be used for replay.")

    elif active_view == "Data Audit":
        st.subheader("Fetch Log")
        st.dataframe(load_fetch_log(DEFAULT_DB_PATH), width="stretch", hide_index=True)
        st.subheader("Signal Table")
        st.dataframe(signals, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
