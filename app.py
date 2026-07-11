from __future__ import annotations

from pathlib import Path
from datetime import date
import json

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
    load_lab_runs,
    save_lab_paper_orders,
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
    decision_hash,
    evidence_health,
    paper_orders,
    optimize_decision,
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
    MANUAL_ONLY_UNIVERSE: None,
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
VIEW_OPTIONS = ["Rankings", "Detail", "Benchmark", "RRG", "Portfolio", "Chatgptfolio Lab", "Data Audit"]
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
    options = seed_rrg_universe_options()
    return [label for label in DEFAULT_RRG_SEED_UNIVERSES if label in options]


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
    grouped = portfolio.groupby(name_column, as_index=False)["weight"].sum()
    fig.add_trace(go.Pie(labels=grouped[name_column], values=grouped["weight"], hole=0.45))
    fig.update_layout(title=title, height=360)
    return fig


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
    all_seed = all_seed_universe(seed_frames)
    lab_uses_all_seeds = st.session_state.get("dashboard_view") == "Chatgptfolio Lab" and st.session_state.get("lab_candidate_universe", "All seed tickers") == "All seed tickers"
    lab_symbols = set(all_seed["symbol"]) if lab_uses_all_seeds else set()
    symbols = sorted(set(universe["symbol"]).union(rrg_seed_symbols).union(portfolio_seed_symbols).union(lab_symbols).union({benchmark}))
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

    config = SignalConfig(larsson_history_weeks=larsson_history_weeks)
    active_signal_symbols = set(universe["symbol"]).union({benchmark})
    signal_prices = prices_for_symbols(prices, active_signal_symbols)
    signals = cached_signal_table(signal_prices, benchmark, larsson_history_weeks)
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

    if st.session_state.get("dashboard_view") not in VIEW_OPTIONS:
        st.session_state["dashboard_view"] = "Rankings"
    active_view = st.segmented_control(
        "Dashboard view",
        VIEW_OPTIONS,
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
        lab_signals = cached_signal_table(lab_signal_prices, benchmark, larsson_history_weeks)
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

        st.subheader("1. Run setup")
        lc1, lc2, lc3, lc4 = st.columns(4)
        selected_model = lc1.selectbox("OpenAI model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL), help="Model choice is recorded per run; there is no automatic fallback.")
        as_of_date = lc2.date_input("As-of date", value=date.today(), max_value=date.today(), help="Current version analyses the locally cached, current-market candidate table. Historical replay is a future extension.")
        paper_capital = lc3.number_input("Paper capital ($)", min_value=1_000.0, value=100_000.0, step=10_000.0)
        slippage_bps = lc4.number_input("Slippage (bps)", min_value=0.0, value=5.0, step=1.0)
        lc5, lc6, lc7 = st.columns(3)
        lab_max_holdings = lc5.slider("Maximum holdings", 1, 15, 12)
        lab_max_position = lc6.slider("Maximum position (%)", 1, 25, 12) / 100
        commission_bps = lc7.number_input("Commission (bps)", min_value=0.0, value=0.0, step=0.5)
        lc8, lc9 = st.columns(2)
        lab_max_sector = lc8.slider("Maximum sector exposure (%)", 10, 80, 40, step=5) / 100
        refresh_research = lc9.checkbox("Refresh public research evidence", value=True, help="Fetches bounded SEC, FRED, and curated-news evidence only when an analysis is run.")
        lab_config = LabConfig(model=selected_model, max_holdings=lab_max_holdings, max_position=lab_max_position, max_sector_exposure=lab_max_sector, paper_capital=paper_capital, slippage_bps=slippage_bps, commission_bps=commission_bps)

        st.subheader("2. Research packet")
        packet = research_packet(lab_candidates, as_of_date, benchmark, lab_config, refresh_research=False)
        display_packet = lab_state.get("packet", packet)
        source_status, coverage_status, candidate_status = st.columns(3)
        source_status.metric("Price source", display_packet["coverage"]["prices"].title())
        coverage_status.metric("Research completeness", "Ready to refresh" if refresh_research else "Cached prices only")
        candidate_status.metric("Candidates", len(lab_candidates))
        st.caption(f"{len(lab_ranked)} technically eligible tickers were ranked; only the top {len(lab_candidates)} enter the bounded research packet. Missing sources remain explicit and are never filled by model inference.")
        st.subheader("Research Overview")
        health = evidence_health(display_packet)
        severity_dot_strip(health, "Evidence health")
        st.dataframe(pd.DataFrame(health), width="stretch", hide_index=True)
        macro = display_packet.get("macro", {})
        if macro.get("series"):
            st.caption(f"Macro regime: **{macro.get('regime', 'unavailable')}**")
            st.dataframe(pd.DataFrame(macro["series"]).T, width="stretch", hide_index=False)
        with st.expander("Immutable packet audit (raw JSON)"):
            st.json(display_packet)

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
            packet = research_packet(lab_candidates, as_of_date, benchmark, lab_config, refresh_research=refresh_research)
            packet_hash = save_lab_research_snapshot(packet, DEFAULT_DB_PATH)
            save_lab_research_details(packet_hash, packet, DEFAULT_DB_PATH)
            try:
                model_decision, raw_response, returned_model = OpenAIPortfolioModel(selected_model).decide(packet, lab_config)
                decision = optimize_decision(model_decision, lab_candidates, lab_config)
                validation = validate_decision(decision, lab_candidates, lab_config, packet=packet)
                status = "proposal_ready" if validation.valid else "validation_failed"
                run_id = save_lab_run({
                    "as_of_date": as_of_date,
                    "benchmark": benchmark,
                    "model": selected_model,
                    "returned_model": returned_model,
                    "status": status,
                    "packet_hash": packet_hash,
                    "prompt_hash": decision_hash({"packet": packet, "config": lab_config_payload(lab_config)}),
                    "config_json": json.dumps(lab_config_payload(lab_config)),
                    "response_json": json.dumps({"model_decision": model_decision, "optimized_decision": decision}),
                    "raw_response": raw_response,
                    "validation_json": json.dumps({"valid": validation.valid, "errors": validation.errors, "warnings": validation.warnings}),
                }, DEFAULT_DB_PATH)
                st.session_state["lab_state"] = {"run_id": run_id, "decision": decision, "validation": validation, "model": selected_model, "packet": packet}
                st.rerun()
            except Exception as exc:
                run_id = save_lab_run({"as_of_date": as_of_date, "benchmark": benchmark, "model": selected_model, "status": "model_failed", "packet_hash": packet_hash, "config_json": json.dumps(lab_config_payload(lab_config)), "error_message": str(exc)}, DEFAULT_DB_PATH)
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
            if validation.valid:
                st.success("Proposal passed deterministic limits. Review before approving paper orders.")
            else:
                st.error("Proposal is blocked: " + "; ".join(validation.errors))
            if decision.get("abstain_reason"):
                st.warning("Model abstained: " + str(decision["abstain_reason"]))
            st.write(decision.get("portfolio_thesis", ""))
            optimization = decision.get("optimization", {})
            if optimization:
                st.caption(f"Deterministic allocator: {optimization.get('allocated_equity', 0):.1%} equity / {decision.get('cash_weight', 0):.1%} cash. Binding constraints: {', '.join(optimization.get('binding_constraints', [])) or 'none'}.")
            proposal = pd.DataFrame(decision.get("holdings", []))
            if proposal.empty:
                st.info("No holdings proposed.")
            else:
                st.dataframe(proposal, width="stretch", hide_index=True)

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
                    st.json({key: candidate.get(key) for key in ["name", "seed_memberships", "sector", "technical_score", "fundamental_score", "combined_score", "fundamental_factors"] if key in candidate})
                    rows = packet_evidence[packet_evidence["symbol"] == symbol] if not packet_evidence.empty and "symbol" in packet_evidence else pd.DataFrame()
                    if rows.empty:
                        st.warning("No refreshed company evidence was included; treat this candidate as unresolved.")
                    else:
                        st.dataframe(rows[[column for column in ["category", "title", "publisher", "published_at", "status", "source_url"] if column in rows]], width="stretch", hide_index=True)

            st.subheader("5. Paper rebalance")
            orders = paper_orders(pd.DataFrame(), decision, lab_config)
            if orders.empty:
                st.info("No paper orders proposed.")
            else:
                orders_display = orders.copy()
                for column in ["previous_weight", "target_weight"]:
                    orders_display[column] = orders_display[column] * 100
                st.dataframe(orders_display, width="stretch", hide_index=True)
                if validation.valid and st.button("Approve and save paper orders"):
                    save_lab_paper_orders(int(lab_state["run_id"]), orders, DEFAULT_DB_PATH)
                    st.success("Paper orders saved. No broker was contacted.")

            with st.expander("Decision audit"):
                st.json(decision)

            with st.expander("Research evidence used"):
                evidence = pd.DataFrame(lab_state.get("packet", {}).get("evidence", []))
                if evidence.empty:
                    st.info("No refreshed external evidence was included in this run.")
                else:
                    st.dataframe(evidence[[column for column in ["symbol", "category", "title", "publisher", "published_at", "status", "source_url"] if column in evidence]], width="stretch", hide_index=True)

        st.subheader("6. Audit & performance")
        runs = load_lab_runs(DEFAULT_DB_PATH)
        if runs.empty:
            st.info("No prior lab runs saved.")
        else:
            st.dataframe(runs, width="stretch", hide_index=True)
            st.caption("Runs preserve an immutable packet and source timestamps. Historical replay must reject evidence published or fetched after its as-of date.")

    elif active_view == "Data Audit":
        st.subheader("Fetch Log")
        st.dataframe(load_fetch_log(DEFAULT_DB_PATH), width="stretch", hide_index=True)
        st.subheader("Signal Table")
        st.dataframe(signals, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
