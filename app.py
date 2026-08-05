from __future__ import annotations

from collections.abc import MutableMapping
from datetime import timedelta
import json
from pathlib import Path
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stockpicker.data import (
    DEFAULT_DB_PATH,
    fetch_price_history,
    latest_cache_timestamp,
    load_fetch_log,
    load_latest_portfolio_snapshot,
    load_matching_portfolio_snapshot,
    load_portfolio_rebalance_log,
    read_cached_prices,
    save_portfolio_snapshot,
)
from stockpicker.activity import (
    capture_analysis_snapshot,
    load_latest_analysis_snapshot,
    load_signal_events,
)
from stockpicker.data_quality import DERIVED_200W_NOTE, compute_data_health
from stockpicker.portfolio import (
    PortfolioConfig,
    SENSITIVITY_PROFILES,
    allocate_portfolio,
    apply_sensitivity_to_target,
    build_allocator_prompt,
    build_portfolio_candidates,
    compare_portfolios,
    canonical_selected_seeds,
    portfolio_snapshot_metadata,
    portfolio_config_hash,
    rrg_seed_summary,
)
from stockpicker.scoring import RankingConfig, rank_universe
from stockpicker.performance import (
    INITIAL_PAPER_NAV,
    SavedBriefPortfolioHistory,
    build_saved_brief_checkpoints,
    build_saved_brief_portfolio_history,
    saved_brief_tracking_symbols,
)
from stockpicker.reports import (
    BriefItem,
    build_paper_holding_allocation,
    build_paper_rrg_distribution,
    build_weekly_report,
    build_report_brief,
    generate_ai_commentary,
    load_saved_weekly_brief,
    load_saved_weekly_briefs,
    load_weekly_report,
    load_weekly_reports,
    parse_commentary_sections,
    save_ai_commentary,
    save_weekly_brief,
)
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
    MANUAL_ONLY_UNIVERSE: None,
}
RRG_COLORS = {
    "Leading": "#16a34a",
    "Weakening": "#f59e0b",
    "Lagging": "#dc2626",
    "Improving": "#2563eb",
}
MIN_VISIBLE_ALLOCATION_WEIGHT = 0.001
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
VIEW_OPTIONS = ["Rankings", "Detail", "Benchmark", "RRG", "Portfolio", "Activity", "Data Audit"]
RANKING_DISPLAY_COLUMNS = [
    "rank",
    "symbol",
    "name",
    "theme",
    "larsson_regime_history",
    "benchmark_adjusted_score",
    "score",
    "relative_strength_score",
    "relative_strength_label",
    "rs_3m",
    "rs_6m",
    "rs_12m",
    "larsson_regime",
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
RRG_CONTROL_HELP = {
    "tail_weeks": "How many recent weekly RRG points to draw as each ticker's rotation trail.",
    "ratio_smoothing": "Weeks used to smooth the stock-vs-benchmark relative trend. Higher values make the RS Ratio less jumpy.",
    "momentum_smoothing": "Weeks used to smooth relative momentum. Higher values make the vertical RRG movement steadier.",
}


st.set_page_config(page_title="CTO-Style Stock Ranking", layout="wide")


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
    return fetch_price_history(list(symbols), refresh=refresh, years=years, db_path=Path(db_path))


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


def empty_universe() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIVERSE_COLUMNS)


def seed_universe_options() -> list[str]:
    return list(SEED_UNIVERSES)


def seed_rrg_universe_options() -> list[str]:
    return [label for label, path in SEED_UNIVERSES.items() if path is not None]


def default_rrg_seed_universes() -> list[str]:
    return seed_rrg_universe_options()


def default_portfolio_seed_universes() -> list[str]:
    options = seed_rrg_universe_options()
    return [label for label in DEFAULT_PORTFOLIO_SEED_UNIVERSES if label in options]


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


def refresh_batch_succeeded(fetch_log: pd.DataFrame, symbols: list[str] | tuple[str, ...]) -> bool:
    """Return true only when the newest explicit fetch batch succeeded in full."""
    requested = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    if not requested or fetch_log.empty or not {"symbol", "fetched_at", "status"}.issubset(fetch_log.columns):
        return False
    log = fetch_log.copy()
    log["symbol"] = log["symbol"].astype(str).str.strip().str.upper()
    log["fetched_at"] = pd.to_datetime(log["fetched_at"], errors="coerce")
    newest = log["fetched_at"].max()
    if pd.isna(newest):
        return False
    batch = log[log["fetched_at"] == newest]
    statuses = batch.drop_duplicates("symbol", keep="first").set_index("symbol")["status"].astype(str).str.lower()
    return requested.issubset(set(statuses.index)) and statuses.loc[sorted(requested)].isin({"ok", "success"}).all()


def data_health_overview(data_health: pd.DataFrame) -> dict[str, object]:
    if data_health.empty:
        return {"market_date": "Unknown", "fresh_pct": 0.0, "stale_count": 0}
    total = len(data_health)
    fresh = int((data_health["freshness_status"] == "Fresh").sum())
    reference = pd.to_datetime(data_health["reference_market_date"], errors="coerce").max()
    return {
        "market_date": "Unknown" if pd.isna(reference) else reference.strftime("%Y-%m-%d"),
        "fresh_pct": 100.0 * fresh / total if total else 0.0,
        "stale_count": int((data_health["freshness_status"] == "Stale").sum()),
    }


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
    basket_values = pd.to_numeric(base["basket_index"], errors="coerce")
    basket_start_value = basket_values.iloc[0]
    if (
        pd.isna(start_value)
        or start_value <= 0
        or pd.isna(basket_start_value)
        or basket_start_value <= 0
    ):
        return pd.DataFrame(columns=["date", "value"])
    return pd.DataFrame(
        {
            "date": base["date"],
            "value": basket_values / basket_start_value * start_value,
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
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["sma200w"],
            mode="lines",
            name="SMA 200W (derived weekly series)",
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


def rrg_seed_chart(rrg_history: pd.DataFrame, latest_table: pd.DataFrame, benchmark: str) -> go.Figure:
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
                line={"color": color, "width": 2},
                opacity=0.45,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[latest["rs_ratio"]],
                y=[latest["rs_momentum"]],
                mode="markers+text",
                marker={"color": color, "size": 13, "line": {"color": "white", "width": 1}},
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
                arrowwidth=1.4,
                arrowcolor=color,
                opacity=0.75,
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
        "larsson_regime_history": st.column_config.BarChartColumn(
            "Regime History",
            width=120,
            help="Weekly Larsson-proxy history: high bars are Bull, mid bars are Transition, low bars are Bear.",
            y_min=0,
            y_max=1,
            color="#22c55e",
        ),
        "larsson_regime": st.column_config.TextColumn("Regime", help="Current Larsson-proxy regime."),
        "rrg_quadrant": st.column_config.TextColumn("RRG", help="Seed-level RRG quadrant behind this holding."),
        "drawdown_52w": st.column_config.NumberColumn("52W DD", help="Current drawdown from 52-week high.", format="%.1f%%"),
        "extension_sma50d": st.column_config.NumberColumn("Ext 50D", help="Extension versus 50-day moving average.", format="%.1f%%"),
        "reason_codes": st.column_config.TextColumn("Reasons", help="Deterministic reason codes for inclusion, caps, or cash."),
        "data_warnings": st.column_config.TextColumn("Warnings", help="Data-quality or insufficient-history notes."),
    }


def portfolio_allocation_display(target: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    """Add current regime history to a portfolio table without changing stored allocations."""
    display = target.copy()
    history_by_symbol: dict[str, object] = {}
    if not ranked.empty and {"symbol", "larsson_regime_history"}.issubset(ranked.columns):
        history_by_symbol = (
            ranked.drop_duplicates("symbol", keep="first")
            .set_index("symbol")["larsson_regime_history"]
            .to_dict()
        )
    display["larsson_regime_history"] = display.get("symbol", pd.Series(index=display.index, dtype=object)).map(
        history_by_symbol
    )
    display["larsson_regime_history"] = display["larsson_regime_history"].map(
        lambda value: list(value) if isinstance(value, (list, tuple)) else []
    )

    columns = [column for column in display.columns if column != "larsson_regime_history"]
    if "relative_strength_score" in columns:
        insert_at = columns.index("relative_strength_score") + 1
        columns.insert(insert_at, "larsson_regime_history")
    else:
        columns.append("larsson_regime_history")
    return display[columns]


def portfolio_change_column_config() -> dict[str, object]:
    return {
        "symbol": st.column_config.TextColumn("Symbol", help="Ticker whose model weight changed."),
        "previous_weight": st.column_config.NumberColumn("Previous", help="Saved baseline portfolio weight.", format="%.1f%%"),
        "target_weight": st.column_config.NumberColumn("Target", help="Current target model weight.", format="%.1f%%"),
        "delta": st.column_config.NumberColumn("Change", help="Target weight minus previous weight.", format="%.1f%%"),
        "action": st.column_config.TextColumn("Action", help="Added, removed, increased, reduced, or unchanged."),
        "reason_codes": st.column_config.TextColumn("Reasons", help="Why the change was accepted or suppressed."),
    }


def visible_allocation_groups(
    portfolio: pd.DataFrame,
    name_column: str,
    min_weight: float = MIN_VISIBLE_ALLOCATION_WEIGHT,
) -> pd.DataFrame:
    if portfolio.empty or name_column not in portfolio.columns or "weight" not in portfolio.columns:
        return pd.DataFrame(columns=[name_column, "weight"])
    visible = portfolio[[name_column, "weight"]].copy()
    visible[name_column] = visible[name_column].fillna("Unknown").astype(str)
    visible["weight"] = pd.to_numeric(visible["weight"], errors="coerce").fillna(0.0)
    grouped = visible.groupby(name_column, as_index=False)["weight"].sum()
    return grouped[grouped["weight"] >= float(min_weight)].reset_index(drop=True)


def allocation_pie_chart(
    portfolio: pd.DataFrame,
    name_column: str,
    title: str,
    *,
    height: int = 360,
) -> go.Figure:
    fig = go.Figure()
    grouped = visible_allocation_groups(portfolio, name_column)
    if grouped.empty:
        fig.update_layout(title=title, height=height)
        return fig
    fig.add_trace(
        go.Pie(
            labels=grouped[name_column],
            values=grouped["weight"],
            hole=0.45,
            texttemplate="%{percent:.1%}",
            textposition="auto",
            hovertemplate="%{label}<br>Allocation: %{value:.1%}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=height)
    return fig


def paper_rrg_distribution_chart(distribution: pd.DataFrame) -> go.Figure:
    colors = {**RRG_COLORS, "Cash / Unclassified": "#64748b"}
    fig = go.Figure()
    if distribution.empty:
        fig.update_layout(title="Paper Portfolio RRG Distribution", height=340)
        return fig
    fig.add_trace(
        go.Pie(
            labels=distribution["quadrant"],
            values=distribution["weight"],
            customdata=distribution[["holding_count", "holdings"]],
            marker={"colors": [colors.get(value, "#64748b") for value in distribution["quadrant"]]},
            hole=0.45,
            sort=False,
            textinfo="label+percent",
            hovertemplate=(
                "%{label}<br>Portfolio weight: %{value:.1%}<br>"
                "Holdings: %{customdata[0]}<br>%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Paper Portfolio RRG Distribution",
        height=360,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
    )
    return fig


def saved_brief_portfolio_chart(
    history: SavedBriefPortfolioHistory,
    selected_saved_brief_id: int,
) -> go.Figure:
    fig = go.Figure()
    if history.nav.empty:
        fig.update_layout(height=420)
        return fig
    fig.add_trace(
        go.Scatter(
            x=history.nav["date"],
            y=history.nav["nav"],
            name="Saved brief portfolio",
            mode="lines",
            line={"color": "#38bdf8", "width": 3},
            hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: $%{y:,.0f}<extra></extra>",
        )
    )
    benchmark_colors = {"Brief benchmark": "#f59e0b", "Brief benchmark · SPY": "#f59e0b", "SPY": "#94a3b8"}
    for series, rows in history.benchmarks.groupby("series", sort=False):
        fig.add_trace(
            go.Scatter(
                x=rows["date"],
                y=rows["nav"],
                name=str(series),
                mode="lines",
                line={"color": benchmark_colors.get(str(series), "#94a3b8"), "width": 1.5, "dash": "dot"},
                customdata=rows[["symbol"]],
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: $%{y:,.0f}<br>Active: %{customdata[0]}<extra></extra>",
            )
        )
    applied = history.checkpoints[history.checkpoints["status"] == "Applied"].copy()
    if not applied.empty:
        markers = applied.merge(
            history.nav[["date", "nav"]],
            left_on="effective_date",
            right_on="date",
            how="left",
        )
        markers["selected"] = markers["saved_brief_id"].astype(int) == int(selected_saved_brief_id)
        for selected, rows in markers.groupby("selected", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=rows["effective_date"],
                    y=rows["nav"],
                    name="Selected checkpoint" if selected else "Saved brief checkpoint",
                    mode="markers",
                    marker={
                        "symbol": "diamond",
                        "size": 13 if selected else 9,
                        "color": "#f97316" if selected else "#facc15",
                        "line": {"color": "#ffffff", "width": 1 if selected else 0},
                    },
                    customdata=rows[["title", "evidence_date", "saved_brief_id"]],
                    hovertemplate=(
                        "%{customdata[0]}<br>Evidence: %{customdata[1]|%Y-%m-%d}<br>"
                        "Rebalanced: %{x|%Y-%m-%d}<br>Portfolio: $%{y:,.0f}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        height=440,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        yaxis_title="Value ($100,000 start)",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5},
    )
    return fig


def saved_brief_tracking_metrics(history: SavedBriefPortfolioHistory) -> dict[str, float]:
    if history.nav.empty:
        return {}
    current_nav = float(history.nav.iloc[-1]["nav"])
    portfolio_return = current_nav / INITIAL_PAPER_NAV - 1.0
    max_drawdown = float(history.nav["drawdown"].min())
    dynamic = history.benchmarks[history.benchmarks["series"].astype(str).str.startswith("Brief benchmark")]
    benchmark_return = float(dynamic.iloc[-1]["nav"]) / INITIAL_PAPER_NAV - 1.0 if not dynamic.empty else float("nan")
    return {
        "current_nav": current_nav,
        "return": portfolio_return,
        "max_drawdown": max_drawdown,
        "versus_benchmark": portfolio_return - benchmark_return,
    }


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


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def parse_report_payload(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(_text(value) or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def activity_event_category(event_type: object) -> str:
    event = _text(event_type).lower()
    if "paper_allocation" in event:
        return "Portfolio"
    if "rrg" in event or "quadrant" in event:
        return "RRG"
    if "regime" in event:
        return "Regime"
    if "freshness" in event or "data" in event or "coverage" in event:
        return "Data"
    return "Other"


def activity_event_label(event_type: object) -> str:
    labels = {
        "benchmark_regime_change": "Benchmark regime changed",
        "larsson_regime_change": "Regime changed",
        "stock_rrg_quadrant_change": "RRG quadrant changed",
        "seed_rrg_quadrant_change": "RRG quadrant changed",
        "freshness_status_change": "Data freshness changed",
        "paper_allocation_added": "Added to paper portfolio",
        "paper_allocation_removed": "Removed from paper portfolio",
        "paper_allocation_changed": "Paper weight changed",
    }
    event = _text(event_type)
    return labels.get(event, event.replace("_", " ").title())


def filter_activity_events(
    events: pd.DataFrame,
    *,
    categories: list[str],
    subject: str = "All",
    window_days: int | None = 30,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    filtered = events.copy()
    filtered["event_category"] = filtered["event_type"].map(activity_event_category)
    filtered = filtered[filtered["event_category"].isin(categories)]
    if subject != "All":
        filtered = filtered[filtered["subject"].astype(str) == subject]
    filtered["effective_market_date"] = pd.to_datetime(filtered["effective_market_date"], errors="coerce")
    if window_days is not None:
        reference = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
        filtered = filtered[filtered["effective_market_date"] >= reference - timedelta(days=window_days)]
    return filtered.sort_values(["effective_market_date", "event_type", "subject"], ascending=[False, True, True])


def report_evidence_table(payload: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    effective_date = _text(payload.get("effective_market_date"))[:10]
    for item in payload.get("evidence", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Evidence ID": _text(item.get("evidence_id")),
                "Date": _text(item.get("effective_market_date") or effective_date)[:10],
                "Type": _text(item.get("event_type") or item.get("kind")),
                "Subject": _text(item.get("subject") or item.get("symbol")),
                "Value": _text(item.get("new_value") or item.get("larsson_regime") or item.get("weight")),
            }
        )
    return pd.DataFrame(rows, columns=["Evidence ID", "Date", "Type", "Subject", "Value"])


def queue_weekly_report_selection(state: MutableMapping[str, object], report_id: int) -> None:
    """Defer a report-widget selection until the next Streamlit run."""
    state["pending_weekly_report_id"] = int(report_id)


def apply_pending_weekly_report_selection(state: MutableMapping[str, object]) -> None:
    """Apply a deferred report selection before its widget is instantiated."""
    pending_report_id = state.pop("pending_weekly_report_id", None)
    if pending_report_id is not None:
        state["selected_weekly_report_id"] = int(pending_report_id)


def queue_saved_weekly_brief_selection(state: MutableMapping[str, object], saved_brief_id: int) -> None:
    """Switch to a newly saved brief on the next Streamlit run."""
    state["pending_saved_weekly_brief_id"] = int(saved_brief_id)
    state["pending_weekly_brief_library"] = "Saved"


def apply_pending_saved_weekly_brief_selection(state: MutableMapping[str, object]) -> None:
    pending_library = state.pop("pending_weekly_brief_library", None)
    if pending_library in {"Generated", "Saved"}:
        state["weekly_brief_library"] = pending_library
    pending_brief_id = state.pop("pending_saved_weekly_brief_id", None)
    if pending_brief_id is not None:
        state["selected_saved_weekly_brief_id"] = int(pending_brief_id)


def default_saved_brief_title(selected_report: dict[str, object], payload: dict[str, object]) -> str:
    benchmark = _text(selected_report.get("benchmark") or payload.get("benchmark")) or "Market"
    effective_date = _text(payload.get("effective_market_date"))[:10] or "undated"
    return f"{benchmark} Weekly Brief — {effective_date}"


def _render_brief_items(items: tuple[BriefItem, ...], empty_message: str, overflow: int = 0) -> None:
    if not items:
        st.caption(empty_message)
        return
    for item in items:
        st.markdown(f"**{item.subject}** · {item.label}")
        st.caption(item.detail)
    if overflow:
        st.caption(f"{overflow} more in the full report")


COMMENTARY_SECTION_STYLES = {
    "summary": ("🧭", "Key takeaways"),
    "opportunities": ("🚀", "Opportunities"),
    "risks": ("⚠️", "Risks"),
    "watch_items": ("👀", "Watch items"),
}


def split_commentary_points(value: object) -> list[str]:
    """Split model prose at explicit observation and interpretation markers."""
    text = _text(value).strip()
    if not text:
        return []
    points = re.split(r"(?=\b(?:Observation|Interpretation):\s*)", text, flags=re.IGNORECASE)
    return [point.strip() for point in points if point.strip()]


def commentary_point_markdown(point: str, section: str) -> str:
    labeled = re.match(r"^(Observation|Interpretation):\s*(.*)$", point.strip(), flags=re.IGNORECASE | re.DOTALL)
    if labeled:
        label = labeled.group(1).capitalize()
        icon = "🔎" if label == "Observation" else "💡"
        return f"- {icon} **{label}:** {labeled.group(2).strip()}"
    icon = COMMENTARY_SECTION_STYLES.get(section, ("•", ""))[0]
    return f"- {icon} {point.strip()}"


def _render_chatgptfolio_commentary(
    selected_report: dict[str, object],
    report_id: int,
    *,
    allow_generation: bool = True,
) -> None:
    with st.container(border=True):
        st.markdown("#### Chatgptfolio view")
        sections = parse_commentary_sections(selected_report.get("commentary_json"))
        legacy_commentary = _text(selected_report.get("commentary"))
        status = _text(selected_report.get("generation_status")) or "not_requested"
        if sections:
            summary_items = sections.get("summary", [])
            if summary_items:
                summary_points = [
                    point
                    for item in summary_items
                    for point in split_commentary_points(item.get("text"))
                ]
                summary_icon, summary_label = COMMENTARY_SECTION_STYLES["summary"]
                summary_markdown = "\n".join(
                    commentary_point_markdown(point, "summary") for point in summary_points
                )
                st.markdown(f"**{summary_icon} {summary_label}**\n\n{summary_markdown}")
            else:
                st.caption("No evidence-backed summary was identified.")
            supporting_count = sum(
                len(split_commentary_points(item.get("text")))
                for section in ["opportunities", "risks", "watch_items"]
                for item in sections.get(section, [])
            )
            if supporting_count:
                with st.expander(f"Supporting observations ({supporting_count})"):
                    for section in ["opportunities", "risks", "watch_items"]:
                        items = sections.get(section, [])
                        if not items:
                            continue
                        icon, label = COMMENTARY_SECTION_STYLES[section]
                        section_points: list[str] = []
                        for item in items:
                            refs = ", ".join(item.get("evidence_ids", []))
                            for point in split_commentary_points(item.get("text")):
                                reference = f" · `{refs}`" if refs else ""
                                section_points.append(f"{commentary_point_markdown(point, section)}{reference}")
                        st.markdown(f"**{icon} {label}**\n\n" + "\n".join(section_points))
        elif legacy_commentary:
            st.caption("This older report has commentary in the legacy format.")
            with st.expander("Open full commentary"):
                st.markdown(legacy_commentary)
        else:
            st.caption(
                "Generate a short, evidence-linked interpretation for this report. It cannot change rankings or allocations."
                if allow_generation
                else "No Chatgptfolio commentary was captured in this saved brief."
            )

        if sections or legacy_commentary:
            metadata: list[str] = []
            if _text(selected_report.get("ai_model")):
                metadata.append(_text(selected_report.get("ai_model")))
            if _text(selected_report.get("commentary_created_at")):
                created = pd.Timestamp(selected_report["commentary_created_at"])
                metadata.append(created.strftime("%Y-%m-%d %H:%M"))
            if metadata:
                st.caption(" · ".join(metadata))
        if status not in {"success", "not_requested"} and _text(selected_report.get("diagnostic")):
            st.warning(_text(selected_report.get("diagnostic")))

        if allow_generation and st.button(
            "Generate commentary" if not legacy_commentary else "Regenerate commentary",
            key=f"generate_commentary_{report_id}",
            type="secondary",
        ):
            evidence_payload = parse_report_payload(selected_report.get("evidence_json"))
            with st.spinner("Building evidence-linked commentary…"):
                result = generate_ai_commentary(evidence_payload)
            save_ai_commentary(report_id, result, DEFAULT_DB_PATH)
            if result.status == "success":
                st.session_state["activity_notice"] = f"Commentary generated with {result.model}."
                st.session_state["activity_notice_kind"] = "success"
            else:
                st.session_state["activity_notice"] = result.diagnostic
                st.session_state["activity_notice_kind"] = "warning"
            st.rerun()


def _saved_briefs_for_tracking(saved_history: pd.DataFrame) -> list[dict[str, object]]:
    briefs: list[dict[str, object]] = []
    for row in saved_history.sort_values(["created_at", "saved_brief_id"]).itertuples(index=False):
        try:
            briefs.append(load_saved_weekly_brief(int(row.saved_brief_id), DEFAULT_DB_PATH))
        except ValueError as exc:
            briefs.append(
                {
                    "saved_brief_id": int(row.saved_brief_id),
                    "saved_title": str(row.title),
                    "saved_at": row.created_at,
                    "evidence_json": "",
                    "tracking_error": str(exc),
                }
            )
    return briefs


def _render_saved_brief_portfolio_tracker(
    saved_history: pd.DataFrame,
    selected_saved_brief_id: int,
) -> None:
    tracking_briefs = _saved_briefs_for_tracking(saved_history)
    checkpoints = build_saved_brief_checkpoints(tracking_briefs)
    required_symbols = saved_brief_tracking_symbols(checkpoints)

    with st.container(border=True):
        title_col, refresh_col = st.columns([4, 1])
        title_col.markdown("#### Saved Brief Portfolio History")
        title_col.caption("One evolving hypothetical portfolio, rebalanced from immutable saved-brief allocations.")
        refresh_clicked = refresh_col.button(
            "Refresh tracker prices",
            key="refresh_saved_brief_tracker_prices",
            width="stretch",
            disabled=not required_symbols,
            help="Refresh only saved holdings, their report benchmarks, and SPY.",
        )
        if refresh_clicked:
            with st.spinner(f"Refreshing {len(required_symbols)} tracker symbols…"):
                fetch_price_history(list(required_symbols), refresh=True, years=10, db_path=DEFAULT_DB_PATH)
            st.session_state["activity_notice"] = "Saved-brief tracker prices refreshed."
            st.session_state["activity_notice_kind"] = "success"
            st.rerun()

        prices = read_cached_prices(list(required_symbols), DEFAULT_DB_PATH) if required_symbols else pd.DataFrame()
        history = build_saved_brief_portfolio_history(checkpoints, prices)
        if history.missing_symbols:
            st.warning("Missing cached tracker history: " + ", ".join(history.missing_symbols))

        if history.nav.empty:
            ready_or_pending = history.checkpoints[
                history.checkpoints["status"].isin(["Ready", "Pending"])
            ] if not history.checkpoints.empty else pd.DataFrame()
            if ready_or_pending.empty:
                st.info("No usable paper allocation was captured in the saved brief library.")
            else:
                st.info(
                    "Awaiting the first eligible cached market open after the saved evidence date. "
                    "Refresh tracker prices after a new market session to begin the history."
                )
        else:
            metrics = saved_brief_tracking_metrics(history)
            metric_one, metric_two, metric_three, metric_four = st.columns(4)
            metric_one.metric("Current value", money(metrics["current_nav"]), pct(metrics["return"]))
            metric_two.metric("Total return", pct(metrics["return"]))
            metric_three.metric("Max drawdown", pct(metrics["max_drawdown"]))
            metric_four.metric("Vs brief benchmark", pct(metrics["versus_benchmark"]))
            st.plotly_chart(
                saved_brief_portfolio_chart(history, selected_saved_brief_id),
                use_container_width=True,
                key="saved_brief_portfolio_history",
                config={"displayModeBar": False},
            )
            latest_date = pd.Timestamp(history.latest_valuation_date).strftime("%Y-%m-%d")
            st.caption(
                f"Valued through {latest_date} from adjusted local prices. Rebalances use the next eligible cached open; "
                "fractional shares, zero costs, no cash yield, and no broker execution."
            )

        with st.expander(f"Checkpoint audit ({len(history.checkpoints)})"):
            if history.checkpoints.empty:
                st.caption("No saved brief checkpoints are available.")
            else:
                audit = history.checkpoints[
                    [
                        "saved_brief_id",
                        "title",
                        "evidence_date",
                        "effective_date",
                        "status",
                        "benchmark",
                        "holding_count",
                        "cash_weight",
                        "diagnostic",
                    ]
                ].copy()
                audit["cash_weight"] = audit["cash_weight"] * 100
                st.dataframe(
                    audit,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "saved_brief_id": st.column_config.NumberColumn("Brief", format="#%d"),
                        "title": st.column_config.TextColumn("Title"),
                        "evidence_date": st.column_config.DateColumn("Evidence Date", format="YYYY-MM-DD"),
                        "effective_date": st.column_config.DateColumn("Simulated Rebalance", format="YYYY-MM-DD"),
                        "status": st.column_config.TextColumn("Status"),
                        "benchmark": st.column_config.TextColumn("Benchmark"),
                        "holding_count": st.column_config.NumberColumn("Holdings", format="%d"),
                        "cash_weight": st.column_config.NumberColumn("Cash", format="%.1f%%"),
                        "diagnostic": st.column_config.TextColumn("Diagnostic"),
                    },
                )


def _render_weekly_brief() -> None:
    apply_pending_weekly_report_selection(st.session_state)
    apply_pending_saved_weekly_brief_selection(st.session_state)
    st.markdown("### Weekly Brief")
    st.caption("A decision-ready summary of the latest local evidence. Full Markdown and evidence remain available below.")

    library_mode = st.segmented_control(
        "Brief library",
        ["Generated", "Saved"],
        default="Generated",
        key="weekly_brief_library",
        label_visibility="collapsed",
    ) or "Generated"

    notice = st.session_state.pop("activity_notice", "")
    notice_kind = st.session_state.pop("activity_notice_kind", "info")
    if notice:
        getattr(st, notice_kind if notice_kind in {"success", "warning", "info"} else "info")(notice)

    selected_report: dict[str, object] = {}
    saved_history = pd.DataFrame()
    content_key = ""
    allow_commentary_generation = library_mode == "Generated"
    if library_mode == "Saved":
        saved_history = load_saved_weekly_briefs(DEFAULT_DB_PATH)
        if saved_history.empty:
            st.info("No named briefs have been saved yet. Select Generated, then use Save brief to create one.")
            return
        saved_labels = {
            int(row.saved_brief_id): (
                f"{row.title} · {pd.Timestamp(row.created_at).strftime('%b %d, %Y %H:%M')} · #{int(row.saved_brief_id)}"
            )
            for row in saved_history.itertuples(index=False)
        }
        saved_ids = list(saved_labels)
        if st.session_state.get("selected_saved_weekly_brief_id") not in saved_ids:
            st.session_state["selected_saved_weekly_brief_id"] = saved_ids[0]
        saved_brief_id = int(
            st.selectbox(
                "Saved briefs",
                saved_ids,
                format_func=lambda brief_id: saved_labels[brief_id],
                key="selected_saved_weekly_brief_id",
                label_visibility="collapsed",
            )
        )
        try:
            selected_report = load_saved_weekly_brief(saved_brief_id, DEFAULT_DB_PATH)
        except ValueError as exc:
            st.warning(str(exc))
            return
        if not selected_report:
            st.warning("The selected saved brief no longer exists.")
            return
        content_key = f"saved_{saved_brief_id}"
    else:
        report_history = load_weekly_reports(DEFAULT_DB_PATH)
        selector_col, generate_col, save_col = st.columns([4, 1, 1])
        if report_history.empty:
            selector_col.caption("No report history yet")
        else:
            report_labels = {
                int(row.report_id): (
                    f"{pd.Timestamp(row.created_at).strftime('%b %d, %Y')} · "
                    f"{row.benchmark} · v{int(row.report_version)}"
                )
                for row in report_history.itertuples(index=False)
            }
            report_ids = list(report_labels)
            if st.session_state.get("selected_weekly_report_id") not in report_ids:
                st.session_state["selected_weekly_report_id"] = report_ids[0]
            selector_col.selectbox(
                "Report history",
                report_ids,
                format_func=lambda report_id: report_labels[report_id],
                key="selected_weekly_report_id",
                label_visibility="collapsed",
            )
        if generate_col.button("Generate latest", type="primary", width="stretch"):
            try:
                generated_report = build_weekly_report(DEFAULT_DB_PATH)
                queue_weekly_report_selection(st.session_state, generated_report.report_id)
                st.session_state["activity_notice"] = (
                    f"Created report v{generated_report.version}."
                    if generated_report.created_new
                    else "Evidence is unchanged; the existing report is selected."
                )
                st.session_state["activity_notice_kind"] = "success" if generated_report.created_new else "info"
                st.rerun()
            except RuntimeError as exc:
                st.warning(str(exc))
        if report_history.empty:
            save_col.caption("Save brief")
            st.info("Generate the first weekly brief after a successful data refresh.")
            return

        report_id = int(st.session_state["selected_weekly_report_id"])
        selected_report = load_weekly_report(report_id, DEFAULT_DB_PATH)
        payload_for_title = parse_report_payload(selected_report.get("evidence_json"))
        with save_col.popover("Save brief", width="stretch"):
            st.caption("Create an immutable copy of this report and its current commentary.")
            with st.form(f"save_weekly_brief_{report_id}"):
                saved_title = st.text_input(
                    "Brief title",
                    value=default_saved_brief_title(selected_report, payload_for_title),
                    max_chars=120,
                )
                if st.form_submit_button("Save snapshot", type="primary", width="stretch"):
                    try:
                        saved = save_weekly_brief(saved_title, report_id, DEFAULT_DB_PATH)
                    except ValueError as exc:
                        st.warning(str(exc))
                    else:
                        queue_saved_weekly_brief_selection(st.session_state, saved.saved_brief_id)
                        st.session_state["activity_notice"] = f"Saved “{saved.title}”."
                        st.session_state["activity_notice_kind"] = "success"
                        st.rerun()
        content_key = f"report_{report_id}"

    payload = parse_report_payload(selected_report.get("evidence_json"))
    brief = build_report_brief(payload)
    if library_mode == "Saved":
        saved_at = pd.Timestamp(selected_report.get("saved_at")).strftime("%Y-%m-%d %H:%M")
        st.caption(
            f"Saved {saved_at} · immutable snapshot · evidence date {brief.effective_market_date} · "
            f"{brief.evidence_count} evidence records"
        )
    else:
        st.caption(
            f"Evidence date {brief.effective_market_date} · {brief.evidence_count} evidence records · "
            "local data only"
        )
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Market stance", brief.benchmark_regime, help=f"Current Larsson-proxy regime for {brief.benchmark}.")
    kpi2.metric("Data confidence", f"{brief.fresh_count}/{brief.stock_count} fresh", f"{brief.freshness_pct:.0f}%", delta_color="off")
    kpi3.metric("Material changes", brief.material_change_count, help="All captured signal changes in this report window.")
    kpi4.metric("Portfolio changes", brief.allocation_change_count, help="Changes to the saved paper allocation only.")

    source_report_id = int(selected_report.get("source_report_id") or selected_report.get("report_id") or 0)
    _render_chatgptfolio_commentary(
        selected_report,
        source_report_id,
        allow_generation=allow_commentary_generation,
    )

    holding_allocation = build_paper_holding_allocation(payload)
    with st.container(border=True):
        if holding_allocation.empty:
            st.markdown("#### Allocation by Holding")
            st.caption("No saved paper-allocation evidence was captured in this brief.")
        else:
            st.plotly_chart(
                allocation_pie_chart(
                    holding_allocation,
                    "symbol",
                    "Allocation by Holding",
                    height=560,
                ),
                use_container_width=True,
                key=f"paper_holding_allocation_{content_key}",
            )
            st.caption("Holding weights captured with this brief; historical snapshots never use the live portfolio.")

    rrg_distribution = build_paper_rrg_distribution(payload)
    with st.container(border=True):
        if rrg_distribution.empty:
            st.markdown("#### Paper Portfolio RRG Distribution")
            st.caption("No saved paper-allocation evidence was captured in this brief.")
        else:
            st.plotly_chart(
                paper_rrg_distribution_chart(rrg_distribution),
                use_container_width=True,
                key=f"paper_rrg_distribution_{content_key}",
            )
            st.caption("Portfolio weights grouped by the RRG quadrant captured with this brief.")

    if library_mode == "Saved":
        _render_saved_brief_portfolio_tracker(saved_history, int(selected_report["saved_brief_id"]))

    leaders_col, risks_col = st.columns(2)
    with leaders_col.container(border=True):
        st.markdown("#### Leaders")
        st.caption("Fresh Bull candidates above the 200-day trend, ranked by score.")
        _render_brief_items(brief.leaders, "No candidate meets the leader criteria in this report.", brief.leader_overflow)
    with risks_col.container(border=True):
        st.markdown("#### Risks to review")
        st.caption("Data issues and deteriorating technical states, ordered by urgency.")
        _render_brief_items(brief.risks, "No deterministic risk flag is active.", brief.risk_overflow)

    changes_col, impact_col = st.columns(2)
    with changes_col.container(border=True):
        st.markdown("#### What changed")
        _render_brief_items(brief.changes, "No material signal change was captured.", brief.change_overflow)
    with impact_col.container(border=True):
        st.markdown("#### Paper portfolio impact")
        if brief.allocation_changes:
            _render_brief_items(brief.allocation_changes, "", brief.allocation_overflow)
        elif brief.has_paper_allocation:
            st.caption("A saved paper portfolio exists; no allocation change was recorded in this report window.")
        else:
            st.caption("No saved paper allocation is available for this report.")

    with st.expander("Full deterministic report"):
        st.markdown(_text(selected_report.get("markdown")))
        st.download_button(
            "Download Markdown",
            data=_text(selected_report.get("markdown")),
            file_name=f"weekly-ta-report-v{int(selected_report.get('report_version', 1))}.md",
            mime="text/markdown",
            key=f"download_report_{content_key}",
        )
    with st.expander(f"Evidence ledger ({brief.evidence_count})"):
        st.dataframe(report_evidence_table(payload), width="stretch", hide_index=True)
    st.caption("Informational only. No brokerage connection, order generation, or execution occurs.")


def _render_change_log() -> None:
    st.markdown("### Change Log")
    st.caption("A chronological audit trail of regime, RRG, data-quality, and paper-portfolio changes.")
    all_events = load_signal_events(DEFAULT_DB_PATH, limit=2_000)
    if all_events.empty:
        st.info("No changes logged yet. The first successful refresh establishes a baseline; later comparable refreshes create events.")
        return

    categories = ["Regime", "RRG", "Portfolio", "Data", "Other"]
    subjects = sorted(all_events["subject"].dropna().astype(str).unique())
    with st.popover("Filters"):
        selected_categories = st.multiselect("Change type", categories, default=categories, key="change_log_categories")
        selected_subject = st.selectbox("Symbol / subject", ["All", *subjects], key="change_log_subject")
        time_window = st.selectbox("Time window", ["30 days", "7 days", "90 days", "All"], key="change_log_window")
    window_days = {"7 days": 7, "30 days": 30, "90 days": 90}.get(time_window)
    filtered = filter_activity_events(
        all_events,
        categories=selected_categories,
        subject=selected_subject,
        window_days=window_days,
    ) if selected_categories else all_events.iloc[0:0].copy()

    counts = filtered.get("event_category", pd.Series(dtype=str)).value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime", int(counts.get("Regime", 0)))
    c2.metric("RRG", int(counts.get("RRG", 0)))
    c3.metric("Portfolio", int(counts.get("Portfolio", 0)))
    c4.metric("Data", int(counts.get("Data", 0)))
    st.caption(f"{len(filtered)} changes match the current filters.")

    if filtered.empty:
        st.info("No changes match these filters.")
        return
    with st.container(border=True):
        for row in filtered.head(20).itertuples(index=False):
            date_label = pd.Timestamp(row.effective_market_date).strftime("%b %d, %Y") if not pd.isna(row.effective_market_date) else "Unknown date"
            st.markdown(f"**{row.subject}** · {activity_event_label(row.event_type)}")
            st.caption(f"{date_label} · {_text(row.old_value) or '—'} → {_text(row.new_value) or '—'}")
            if _text(row.reason):
                st.caption(_text(row.reason))
    if len(filtered) > 20:
        st.caption(f"Showing the latest 20 of {len(filtered)} changes. The audit table contains the full filtered history.")
    with st.expander("Audit table & evidence"):
        columns = [
            "effective_market_date", "event_type", "subject", "old_value", "new_value",
            "evidence_id", "reason", "benchmark", "signal_config_hash", "run_id", "evidence_json",
        ]
        st.dataframe(filtered[[column for column in columns if column in filtered.columns]], width="stretch", hide_index=True)


def display_rankings(ranked: pd.DataFrame) -> object:
    existing = [column for column in RANKING_DISPLAY_COLUMNS if column in ranked.columns]
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
    seed_label = st.sidebar.selectbox(
        "Seed universe",
        options,
        index=options.index(DEFAULT_SEED_UNIVERSE),
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
    refresh_clicked = st.sidebar.button("Refresh cache")

    manual_symbols = parse_ticker_text(ticker_text)
    universe = build_active_universe(active_seed, manual_symbols)
    rrg_seed_labels = selected_rrg_seed_labels_for_fetch(
        st.session_state.get("rrg_view", "Stocks"),
        st.session_state.get("rrg_seed_universes", default_rrg_seed_universes()),
    )
    portfolio_data_enabled = bool(st.session_state.get("portfolio_data_enabled", False))
    portfolio_seed_labels_for_fetch = selected_portfolio_seed_labels_for_fetch(
        st.session_state.get("portfolio_seed_universes", default_portfolio_seed_universes()),
        portfolio_data_enabled,
    )
    rrg_seed_symbols = seed_symbols_for_labels(seed_frames, rrg_seed_labels)
    portfolio_seed_symbols = seed_symbols_for_labels(seed_frames, portfolio_seed_labels_for_fetch)
    symbols = sorted(set(universe["symbol"]).union(rrg_seed_symbols).union(portfolio_seed_symbols).union({benchmark}))
    refresh_key = st.session_state.get("refresh_key", 0)
    if refresh_clicked:
        refresh_key += 1
        st.session_state["refresh_key"] = refresh_key

    st.title("CTO-Style Stock Ranking")

    try:
        prices = cached_prices(tuple(symbols), refresh_key, refresh_clicked, years, str(DEFAULT_DB_PATH))
    except RuntimeError as exc:
        st.error(str(exc))
        return

    if prices.empty:
        st.warning("No price data available yet. Refresh the cache after installing the required dependencies.")
        return

    fetch_log = load_fetch_log(DEFAULT_DB_PATH)
    data_health = compute_data_health(prices, benchmark, lookback_sessions=60, fetch_log=fetch_log)
    config = SignalConfig(larsson_history_weeks=larsson_history_weeks)
    active_signal_symbols = set(universe["symbol"]).union({benchmark})
    signal_prices = prices_for_symbols(prices, active_signal_symbols)
    signals = cached_signal_table(signal_prices, benchmark, larsson_history_weeks)
    signals = signals.merge(universe, on="symbol", how="left")
    signals[["name", "theme", "notes"]] = signals[["name", "theme", "notes"]].fillna("")
    signals[["fundamental_score", "catalyst_score"]] = signals[["fundamental_score", "catalyst_score"]].fillna(0.0)
    analysis_signals = rank_universe(
        signals,
        RankingConfig(use_relative_strength=use_relative_strength, exclude_benchmark=False),
    )
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

    stable_rrg_config = RRGConfig()
    stable_stock_rrg = cached_rrg_history(
        signal_prices,
        tuple(sorted(active_signal_symbols)),
        benchmark,
        stable_rrg_config.ratio_window_weeks,
        stable_rrg_config.momentum_window_weeks,
        stable_rrg_config.tail_weeks,
    )
    snapshot_seed_map = (
        seed_universe_symbol_map(seed_frames, [seed_label])
        if seed_label != MANUAL_ONLY_UNIVERSE
        else {}
    )
    stable_seed_rrg = cached_rrg_basket_history(
        prices,
        snapshot_seed_map,
        benchmark,
        stable_rrg_config.ratio_window_weeks,
        stable_rrg_config.momentum_window_weeks,
        stable_rrg_config.tail_weeks,
    ) if snapshot_seed_map else pd.DataFrame()

    if refresh_clicked:
        refresh_succeeded = refresh_batch_succeeded(fetch_log, tuple(symbols))
        capture = capture_analysis_snapshot(
            analysis_signals,
            data_health,
            benchmark,
            active_signal_symbols,
            signal_config=config,
            stock_rrg=stable_stock_rrg,
            seed_rrg=stable_seed_rrg,
            refresh_succeeded=refresh_succeeded,
            db_path=DEFAULT_DB_PATH,
        )
        if capture.captured:
            if capture.baseline:
                st.success("Refresh captured as the first comparable analysis baseline.")
            else:
                st.success(f"Refresh captured with {capture.event_count} new change event(s).")
        else:
            st.warning(f"Analysis snapshot not captured: {capture.diagnostic}")

    latest_snapshot_rows, latest_run = load_latest_analysis_snapshot(DEFAULT_DB_PATH)
    latest_run_events = (
        load_signal_events(DEFAULT_DB_PATH, run_id=str(latest_run["run_id"]))
        if latest_run
        else pd.DataFrame()
    )
    overview = data_health_overview(data_health)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Benchmark / Regime", f"{benchmark} · {benchmark_regime}")
    c2.metric("Market-data Date", overview["market_date"])
    c3.metric("Fresh-symbol Coverage", f"{overview['fresh_pct']:.0f}%")
    c4.metric("Stale Symbols", overview["stale_count"])
    c5.metric("Changes Since Prior Snapshot", len(latest_run_events))

    if st.session_state.get("dashboard_view") not in VIEW_OPTIONS:
        st.session_state["dashboard_view"] = "Rankings"
    active_view = st.segmented_control(
        "Dashboard view",
        VIEW_OPTIONS,
        default=st.session_state["dashboard_view"],
        key="dashboard_view",
        label_visibility="collapsed",
    )

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
            st.session_state["dashboard_view"] = "Detail"
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
            selected_health = data_health[data_health["symbol"] == selected]
            freshness = selected_health["freshness_status"].iloc[0] if not selected_health.empty else "Unknown"
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Score", f"{selected_row['score']:.1f}")
            d2.metric("Latest", money(selected_row["latest_close"]))
            d3.metric("52W Drawdown", pct(selected_row["drawdown_52w"]))
            d4.metric("Extension vs 200D", pct(selected_row["extension_sma200d"]))
            d5.metric("Data Freshness", freshness)

            daily_history = price_history_with_indicators(prices, selected, config)
            weekly_history = weekly_history_with_indicators(prices, selected, config)
            dd_history = drawdown_history(prices, selected)
            st.plotly_chart(price_chart(daily_history, selected), use_container_width=True)
            st.plotly_chart(weekly_chart(weekly_history, selected), use_container_width=True)
            st.caption(DERIVED_200W_NOTE)
            st.plotly_chart(drawdown_chart(dd_history, selected), use_container_width=True)

            notes = str(selected_row.get("notes", "") or "").strip()
            st.caption(f"Seed notes (read-only): {notes or 'None'}")
            ticker_events = load_signal_events(DEFAULT_DB_PATH, symbol=selected, limit=100)
            with st.expander("Recent signal timeline", expanded=not ticker_events.empty):
                if ticker_events.empty:
                    st.info("No logged changes for this ticker yet. Changes begin after two comparable refresh snapshots.")
                else:
                    st.dataframe(
                        ticker_events[
                            ["effective_market_date", "event_type", "old_value", "new_value", "evidence_id", "reason"]
                        ],
                        width="stretch",
                        hide_index=True,
                    )

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
            selected_seed_labels = st.multiselect(
                "Seed universes",
                seed_options,
                default=default_rrg_seed_universes(),
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
                    st.plotly_chart(rrg_seed_chart(seed_history, latest_seed_history, benchmark), use_container_width=True)
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
        max_holdings = pc1.slider("Max holdings", min_value=4, max_value=25, value=12, step=1)
        top_per_seed = pc2.slider("Top candidates per seed", min_value=2, max_value=10, value=5, step=1)
        max_position_pct = pc3.slider("Max position", min_value=5, max_value=25, value=12, step=1)
        max_seed_exposure_pct = pc4.slider("Max seed exposure", min_value=20, max_value=80, value=45, step=5)

        pc5, pc6, pc7 = st.columns(3)
        sensitivity = pc5.selectbox("Allocator sensitivity", list(SENSITIVITY_PROFILES), index=1)
        use_portfolio_rs = pc6.checkbox("Use relative strength", value=True)
        allow_weak_candidates = pc7.checkbox("Allow weak-regime candidates", value=False)
        include_cash_rule = st.checkbox("Use benchmark-regime cash rule", value=True)

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
            portfolio_signals = cached_signal_table(portfolio_prices, benchmark, larsson_history_weeks)
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
                snapshot_metadata = portfolio_snapshot_metadata(benchmark, benchmark_regime, portfolio_config)
                snapshot_metadata["effective_market_date"] = overview["market_date"]
                snapshot_id = save_portfolio_snapshot(
                    applied_target,
                    snapshot_metadata,
                    saved_changes,
                    DEFAULT_DB_PATH,
                )
                st.success(f"Saved paper-allocation snapshot #{snapshot_id}.")

            target_display = portfolio_allocation_display(applied_target, portfolio_ranked)
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

            cc1, cc2 = st.columns(2)
            cc1.plotly_chart(allocation_pie_chart(applied_target, "symbol", "Allocation by Holding"), use_container_width=True)
            cc2.plotly_chart(allocation_pie_chart(applied_target, "seed_universe", "Allocation by Seed Universe"), use_container_width=True)

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

            with st.expander("Deterministic allocator evidence preview"):
                st.code(
                    build_allocator_prompt(
                        previous_portfolio,
                        applied_target,
                        changes,
                        suppressed,
                        benchmark,
                        benchmark_regime,
                        portfolio_config,
                    ),
                    language="json",
                )

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

    elif active_view == "Activity":
        st.subheader("Activity")
        activity_mode = st.segmented_control(
            "Activity view",
            ["Weekly Brief", "Change Log"],
            default="Weekly Brief",
            key="activity_mode",
            label_visibility="collapsed",
        )
        if activity_mode == "Change Log":
            _render_change_log()
        else:
            _render_weekly_brief()

    elif active_view == "Data Audit":
        st.subheader("Data Freshness & Coverage")
        st.caption(DERIVED_200W_NOTE)
        st.dataframe(data_health, width="stretch", hide_index=True)
        with st.expander("Raw fetch log"):
            st.dataframe(fetch_log, width="stretch", hide_index=True)
        with st.expander("Raw signal table"):
            st.dataframe(signals, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
