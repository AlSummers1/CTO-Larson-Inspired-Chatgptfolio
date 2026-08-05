import importlib

import pandas as pd
import pytest
from pathlib import Path

from stockpicker.universe import load_universe


pytest.importorskip("streamlit")


def test_app_import_has_main():
    app = importlib.import_module("app")
    assert callable(app.main)


def test_rrg_theme_filter_preserves_seed_metadata():
    app = importlib.import_module("app")
    ranked = pd.DataFrame(
        [
            {"symbol": "NVDA", "name": "NVIDIA", "theme": "AI accelerators", "score": 95},
            {"symbol": "MSFT", "name": "Microsoft", "theme": "Cloud and AI platforms", "score": 90},
        ]
    )

    filtered = app.filter_ranked_by_seed_themes(ranked, ["AI accelerators"])

    assert filtered["symbol"].tolist() == ["NVDA"]
    assert filtered["name"].tolist() == ["NVIDIA"]
    assert filtered["theme"].tolist() == ["AI accelerators"]


def test_latest_rrg_with_metadata_preserves_name_and_theme():
    app = importlib.import_module("app")
    rrg_history = pd.DataFrame(
        [
            {"symbol": "NVDA", "date": pd.Timestamp("2024-01-05"), "rs_ratio": 101, "rs_momentum": 99, "quadrant": "Weakening"},
            {"symbol": "NVDA", "date": pd.Timestamp("2024-01-12"), "rs_ratio": 103, "rs_momentum": 101, "quadrant": "Leading"},
        ]
    )
    ranked = pd.DataFrame(
        [
            {
                "symbol": "NVDA",
                "name": "NVIDIA",
                "theme": "AI accelerators",
                "score": 95.0,
                "benchmark_adjusted_score": 96.0,
                "relative_strength_score": 100.0,
                "rs_3m": 0.1,
                "rs_6m": 0.2,
                "rs_12m": 0.3,
            }
        ]
    )

    latest = app.latest_rrg_with_metadata(rrg_history, ranked)

    assert len(latest) == 1
    assert latest.iloc[0]["quadrant"] == "Leading"
    assert latest.iloc[0]["name"] == "NVIDIA"
    assert latest.iloc[0]["theme"] == "AI accelerators"


def test_ranking_column_config_includes_header_help_for_score_columns():
    app = importlib.import_module("app")

    config = app.ranking_column_config()
    expected_columns = [
        "benchmark_adjusted_score",
        "score",
        "relative_strength_score",
        "relative_strength_label",
        "rs_3m",
        "rs_6m",
        "rs_12m",
    ]

    for column in expected_columns:
        assert config[column]["help"]


def test_regime_history_is_between_theme_and_benchmark_score():
    app = importlib.import_module("app")

    assert app.RANKING_DISPLAY_COLUMNS.index("theme") + 1 == app.RANKING_DISPLAY_COLUMNS.index(
        "larsson_regime_history"
    )
    assert app.RANKING_DISPLAY_COLUMNS.index("larsson_regime_history") + 1 == app.RANKING_DISPLAY_COLUMNS.index(
        "benchmark_adjusted_score"
    )


def test_portfolio_allocation_display_adds_regime_history_after_rs_score():
    app = importlib.import_module("app")
    target = pd.DataFrame(
        [
            {"symbol": "NVDA", "relative_strength_score": 95.0, "larsson_regime": "Bull"},
            {"symbol": "Cash", "relative_strength_score": 50.0, "larsson_regime": ""},
        ]
    )
    ranked = pd.DataFrame(
        [{"symbol": "NVDA", "larsson_regime_history": [0.5, 1.0, 1.0]}]
    )

    display = app.portfolio_allocation_display(target, ranked)

    assert display.columns.tolist() == [
        "symbol",
        "relative_strength_score",
        "larsson_regime_history",
        "larsson_regime",
    ]
    assert display.loc[display["symbol"] == "NVDA", "larsson_regime_history"].iloc[0] == [0.5, 1.0, 1.0]
    assert display.loc[display["symbol"] == "Cash", "larsson_regime_history"].iloc[0] == []


def test_saved_brief_portfolio_chart_and_metrics_highlight_selected_checkpoint():
    app = importlib.import_module("app")
    history = app.SavedBriefPortfolioHistory(
        nav=pd.DataFrame(
            [
                {"date": pd.Timestamp("2024-01-02"), "nav": 100_000.0, "cash": 0.0, "drawdown": 0.0},
                {"date": pd.Timestamp("2024-01-03"), "nav": 105_000.0, "cash": 0.0, "drawdown": 0.0},
            ]
        ),
        benchmarks=pd.DataFrame(
            [
                {"date": pd.Timestamp("2024-01-02"), "series": "Brief benchmark", "symbol": "QQQ", "nav": 100_000.0},
                {"date": pd.Timestamp("2024-01-03"), "series": "Brief benchmark", "symbol": "QQQ", "nav": 102_000.0},
                {"date": pd.Timestamp("2024-01-02"), "series": "SPY", "symbol": "SPY", "nav": 100_000.0},
                {"date": pd.Timestamp("2024-01-03"), "series": "SPY", "symbol": "SPY", "nav": 101_000.0},
            ]
        ),
        checkpoints=pd.DataFrame(
            [
                {
                    "saved_brief_id": 7,
                    "title": "Selected",
                    "evidence_date": pd.Timestamp("2024-01-01"),
                    "effective_date": pd.Timestamp("2024-01-02"),
                    "status": "Applied",
                }
            ]
        ),
        latest_holdings=pd.DataFrame(),
        required_symbols=("QQQ", "SPY"),
        missing_symbols=(),
        latest_valuation_date=pd.Timestamp("2024-01-03"),
    )

    metrics = app.saved_brief_tracking_metrics(history)
    figure = app.saved_brief_portfolio_chart(history, 7)

    assert metrics["current_nav"] == 105_000.0
    assert metrics["return"] == pytest.approx(0.05)
    assert metrics["versus_benchmark"] == pytest.approx(0.03)
    assert [trace.name for trace in figure.data][:3] == ["Saved brief portfolio", "Brief benchmark", "SPY"]
    selected_trace = next(trace for trace in figure.data if trace.name == "Selected checkpoint")
    assert selected_trace.marker.size == 13


def test_chatgptfolio_commentary_splits_and_formats_observations():
    app = importlib.import_module("app")
    text = (
        "Observation: QQQ remains in a Bull regime. "
        "Interpretation: The backdrop is constructive. "
        "Observation: Daily momentum is mixed."
    )

    points = app.split_commentary_points(text)

    assert points == [
        "Observation: QQQ remains in a Bull regime.",
        "Interpretation: The backdrop is constructive.",
        "Observation: Daily momentum is mixed.",
    ]
    assert app.commentary_point_markdown(points[0], "summary") == (
        "- 🔎 **Observation:** QQQ remains in a Bull regime."
    )
    assert app.commentary_point_markdown(points[1], "summary") == (
        "- 💡 **Interpretation:** The backdrop is constructive."
    )
    assert app.commentary_point_markdown("Watch the next close.", "watch_items") == "- 👀 Watch the next close."


def test_rrg_control_help_text_is_available():
    app = importlib.import_module("app")

    assert app.RRG_CONTROL_HELP["tail_weeks"]
    assert app.RRG_CONTROL_HELP["ratio_smoothing"]
    assert app.RRG_CONTROL_HELP["momentum_smoothing"]


def test_seed_universe_options_default_to_smh():
    app = importlib.import_module("app")

    options = app.seed_universe_options()

    assert app.DEFAULT_SEED_UNIVERSE == "SMH Universe"
    assert options[0] == app.DEFAULT_SEED_UNIVERSE
    assert app.MANUAL_ONLY_UNIVERSE in options


def test_seed_rrg_options_exclude_manual_only_and_all_are_selected_by_default():
    app = importlib.import_module("app")

    options = app.seed_rrg_universe_options()
    defaults = app.default_rrg_seed_universes()

    assert app.MANUAL_ONLY_UNIVERSE not in options
    assert defaults
    assert defaults == options


def test_select_seed_universe_switches_presets():
    app = importlib.import_module("app")
    smh_seed = load_universe(Path("data/universe_smh.csv"))
    healthcare_seed = load_universe(Path("data/universe_healthcare.csv"))
    seed_frames = {
        "SMH Universe": smh_seed,
        "Healthcare Universe": healthcare_seed,
    }

    default_seed = app.select_seed_universe("SMH Universe", seed_frames)
    healthcare_active = app.select_seed_universe("Healthcare Universe", seed_frames)
    manual_only = app.select_seed_universe(app.MANUAL_ONLY_UNIVERSE, seed_frames)

    assert set(default_seed["symbol"]) == set(smh_seed["symbol"])
    assert set(healthcare_active["symbol"]) == set(healthcare_seed["symbol"])
    assert "NVDA" not in set(healthcare_active["symbol"])
    assert manual_only.empty


def test_build_active_universe_preserves_seed_metadata_and_marks_manual():
    app = importlib.import_module("app")
    healthcare_seed = load_universe(Path("data/universe_healthcare.csv"))

    active = app.build_active_universe(healthcare_seed, ["UNH", "TSLA"])

    assert active["symbol"].tolist() == ["UNH", "TSLA"]
    assert active.loc[active["symbol"] == "UNH", "name"].iloc[0] == "UnitedHealth Group"
    assert active.loc[active["symbol"] == "UNH", "theme"].iloc[0] == "Managed care"
    assert active.loc[active["symbol"] == "TSLA", "theme"].iloc[0] == "Manual"


def test_seed_theme_options_follow_active_universe():
    app = importlib.import_module("app")
    smh_seed = load_universe(Path("data/universe_smh.csv"))
    healthcare_seed = load_universe(Path("data/universe_healthcare.csv"))

    smh_themes = app.seed_theme_options(smh_seed)
    healthcare_themes = app.seed_theme_options(healthcare_seed)

    assert "AI accelerators" in smh_themes
    assert "Managed care" in healthcare_themes
    assert "AI accelerators" not in healthcare_themes


def test_seed_text_mode_changes_when_symbols_change():
    app = importlib.import_module("app")
    original = pd.DataFrame({"symbol": ["UNH", "IQV"]})
    updated = pd.DataFrame({"symbol": ["UNH", "LLY"]})

    assert app.seed_text_mode("Healthcare Universe", original) != app.seed_text_mode(
        "Healthcare Universe",
        updated,
    )


def test_seed_symbol_helpers_use_selected_seed_frames():
    app = importlib.import_module("app")
    smh_seed = load_universe(Path("data/universe_smh.csv"))
    healthcare_seed = load_universe(Path("data/universe_healthcare.csv"))
    seed_frames = {
        "SMH Universe": smh_seed,
        "Healthcare Universe": healthcare_seed,
    }

    labels = ["Healthcare Universe"]
    symbol_map = app.seed_universe_symbol_map(seed_frames, labels)
    symbols = app.seed_symbols_for_labels(seed_frames, labels)

    assert set(symbol_map) == {"Healthcare Universe"}
    assert "LLY" in symbol_map["Healthcare Universe"]
    assert "NVDA" not in symbol_map["Healthcare Universe"]
    assert symbols == symbol_map["Healthcare Universe"]


def test_rrg_seed_labels_for_fetch_only_when_seed_view_active():
    app = importlib.import_module("app")

    selected = ["Healthcare Universe", app.MANUAL_ONLY_UNIVERSE, "Not Registered"]

    assert app.selected_rrg_seed_labels_for_fetch("Stocks", selected) == []
    assert app.selected_rrg_seed_labels_for_fetch("Seed universes", selected) == ["Healthcare Universe"]
    assert app.selected_rrg_seed_labels_for_fetch("Seed universes", "bad-state") == app.default_rrg_seed_universes()


def test_portfolio_seed_labels_for_fetch_only_when_enabled():
    app = importlib.import_module("app")

    selected = ["Healthcare Universe", app.MANUAL_ONLY_UNIVERSE, "Not Registered"]

    assert app.selected_portfolio_seed_labels_for_fetch(selected, False) == []
    assert app.selected_portfolio_seed_labels_for_fetch(selected, True) == ["Healthcare Universe"]
    assert app.selected_portfolio_seed_labels_for_fetch("bad-state", True) == app.default_portfolio_seed_universes()


def test_refresh_batch_requires_every_requested_symbol_to_succeed():
    app = importlib.import_module("app")
    log = pd.DataFrame(
        [
            {"symbol": "AAA", "fetched_at": "2024-01-05 10:00", "status": "ok"},
            {"symbol": "QQQ", "fetched_at": "2024-01-05 10:00", "status": "ok"},
        ]
    )

    assert app.refresh_batch_succeeded(log, ("AAA", "QQQ"))
    assert not app.refresh_batch_succeeded(log, ("AAA", "BBB", "QQQ"))


def test_data_health_overview_is_compact_and_counts_stale_symbols():
    app = importlib.import_module("app")
    health = pd.DataFrame(
        {
            "reference_market_date": pd.to_datetime(["2024-01-05", "2024-01-05"]),
            "freshness_status": ["Fresh", "Stale"],
        }
    )

    overview = app.data_health_overview(health)

    assert overview == {"market_date": "2024-01-05", "fresh_pct": 50.0, "stale_count": 1}


def test_activity_view_is_registered():
    app = importlib.import_module("app")
    assert "Activity" in app.VIEW_OPTIONS


def test_activity_events_are_grouped_and_filtered_for_change_log():
    app = importlib.import_module("app")
    events = pd.DataFrame(
        [
            {"effective_market_date": "2024-01-30", "event_type": "larsson_regime_change", "subject": "AAA"},
            {"effective_market_date": "2024-01-20", "event_type": "stock_rrg_quadrant_change", "subject": "BBB"},
            {"effective_market_date": "2023-12-01", "event_type": "paper_allocation_added", "subject": "AAA"},
        ]
    )

    filtered = app.filter_activity_events(
        events,
        categories=["Regime", "RRG"],
        window_days=30,
        today=pd.Timestamp("2024-02-01"),
    )

    assert filtered["subject"].tolist() == ["AAA", "BBB"]
    assert filtered["event_category"].tolist() == ["Regime", "RRG"]
    assert app.activity_event_category("paper_allocation_changed") == "Portfolio"


def test_report_payload_and_evidence_table_fail_closed():
    app = importlib.import_module("app")

    assert app.parse_report_payload("not-json") == {}
    table = app.report_evidence_table(
        {
            "effective_market_date": "2024-01-05",
            "evidence": [
                {"evidence_id": "SG-1", "kind": "signal_snapshot", "subject": "AAA", "larsson_regime": "Bull"}
            ],
        }
    )

    assert table.to_dict("records") == [
        {"Evidence ID": "SG-1", "Date": "2024-01-05", "Type": "signal_snapshot", "Subject": "AAA", "Value": "Bull"}
    ]


def test_weekly_report_selection_is_deferred_until_the_next_run():
    app = importlib.import_module("app")
    state = {"selected_weekly_report_id": 1}

    app.queue_weekly_report_selection(state, 2)

    assert state["selected_weekly_report_id"] == 1
    assert state["pending_weekly_report_id"] == 2

    app.apply_pending_weekly_report_selection(state)

    assert state["selected_weekly_report_id"] == 2
    assert "pending_weekly_report_id" not in state


def test_newly_saved_weekly_brief_switches_library_on_next_run():
    app = importlib.import_module("app")
    state = {"weekly_brief_library": "Generated"}

    app.queue_saved_weekly_brief_selection(state, 7)

    assert state["weekly_brief_library"] == "Generated"
    app.apply_pending_saved_weekly_brief_selection(state)
    assert state["weekly_brief_library"] == "Saved"
    assert state["selected_saved_weekly_brief_id"] == 7


def test_default_saved_brief_title_uses_benchmark_and_evidence_date():
    app = importlib.import_module("app")

    title = app.default_saved_brief_title(
        {"benchmark": "QQQ"},
        {"effective_market_date": "2024-01-05"},
    )

    assert title == "QQQ Weekly Brief — 2024-01-05"


def test_paper_rrg_distribution_chart_uses_rrg_colors_and_hover_details():
    app = importlib.import_module("app")
    distribution = pd.DataFrame(
        [
            {"quadrant": "Leading", "weight": 0.6, "holding_count": 2, "holdings": "AAA 40.0%, BBB 20.0%"},
            {"quadrant": "Cash / Unclassified", "weight": 0.4, "holding_count": 1, "holdings": "Cash 40.0%"},
        ]
    )

    figure = app.paper_rrg_distribution_chart(distribution)

    assert list(figure.data[0].labels) == ["Leading", "Cash / Unclassified"]
    assert list(figure.data[0].marker.colors) == [app.RRG_COLORS["Leading"], "#64748b"]
    assert "Holdings" in figure.data[0].hovertemplate


def test_allocation_pie_hides_zero_and_sub_threshold_residuals():
    app = importlib.import_module("app")
    portfolio = pd.DataFrame(
        [
            {"symbol": "AAA", "seed_universe": "Seed A", "weight": 0.50},
            {"symbol": "BBB", "seed_universe": "Seed B", "weight": 0.4999999999999998},
            {"symbol": "Cash", "seed_universe": "Cash", "weight": 2.22e-16},
            {"symbol": "REMOVED", "seed_universe": "Seed C", "weight": 0.0},
            {"symbol": "TINY", "seed_universe": "Seed C", "weight": 0.0005},
        ]
    )

    grouped = app.visible_allocation_groups(portfolio, "symbol")
    figure = app.allocation_pie_chart(portfolio, "symbol", "Allocation by Holding")

    assert grouped["symbol"].tolist() == ["AAA", "BBB"]
    assert list(figure.data[0].labels) == ["AAA", "BBB"]
    assert list(figure.data[0].values) == pytest.approx([0.50, 0.4999999999999998])
    assert figure.data[0].texttemplate == "%{percent:.1%}"


def test_allocation_pie_keeps_legitimate_minimum_positions():
    app = importlib.import_module("app")
    portfolio = pd.DataFrame([{"symbol": "AAA", "weight": 0.02}, {"symbol": "Cash", "weight": 0.98}])

    grouped = app.visible_allocation_groups(portfolio, "symbol")

    assert grouped["symbol"].tolist() == ["AAA", "Cash"]


def test_allocation_pie_accepts_taller_activity_layout():
    app = importlib.import_module("app")
    portfolio = pd.DataFrame([{"symbol": "AAA", "weight": 1.0}])

    figure = app.allocation_pie_chart(
        portfolio,
        "symbol",
        "Allocation by Holding",
        height=560,
    )

    assert figure.layout.height == 560


def test_prices_for_symbols_keeps_rankings_scoped_to_active_universe():
    app = importlib.import_module("app")
    prices = pd.DataFrame(
        {
            "symbol": ["NVDA", "QQQ", "LLY"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "close": [100.0, 100.0, 100.0],
        }
    )

    filtered = app.prices_for_symbols(prices, {"NVDA", "QQQ"})

    assert filtered["symbol"].tolist() == ["NVDA", "QQQ"]


def test_chart_helpers_add_basket_overlay_traces():
    app = importlib.import_module("app")
    dates = pd.bdate_range("2024-01-01", periods=3)
    daily = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0, 102.0, 104.0],
            "sma50d": [None, None, None],
            "sma200d": [None, None, None],
            "larsson_line_proxy": [None, None, None],
        }
    )
    weekly = pd.DataFrame(
        {
            "date": dates,
            "weekly_close": [100.0, 102.0, 104.0],
            "sma30w": [None, None, None],
            "sma200w": [None, None, None],
        }
    )
    drawdown = pd.DataFrame({"date": dates, "drawdown": [0.0, 0.0, -0.1]})
    price_overlay = pd.DataFrame({"date": dates, "value": [100.0, 103.0, 106.0]})
    dd_overlay = pd.DataFrame({"date": dates, "drawdown": [0.0, -0.02, -0.05]})

    daily_fig = app.price_chart(daily, "QQQ", price_overlay)
    weekly_fig = app.weekly_chart(weekly, "QQQ", price_overlay)
    drawdown_fig = app.drawdown_chart(drawdown, "QQQ", dd_overlay)

    assert daily_fig.data[-1].name == "Seed EW Basket"
    assert weekly_fig.data[-1].name == "Seed EW Basket"
    assert drawdown_fig.data[-1].name == "Seed EW Drawdown"
    assert daily_fig.data[-1].line.color == app.BASKET_OVERLAY_COLOR
    assert daily_fig.data[-1].line.dash == "dash"


def test_chart_helpers_skip_empty_basket_overlay():
    app = importlib.import_module("app")
    dates = pd.bdate_range("2024-01-01", periods=3)
    daily = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0, 102.0, 104.0],
            "sma50d": [None, None, None],
            "sma200d": [None, None, None],
            "larsson_line_proxy": [None, None, None],
        }
    )
    weekly = pd.DataFrame(
        {
            "date": dates,
            "weekly_close": [100.0, 102.0, 104.0],
            "sma30w": [None, None, None],
            "sma200w": [None, None, None],
        }
    )
    drawdown = pd.DataFrame({"date": dates, "drawdown": [0.0, 0.0, -0.1]})

    assert len(app.price_chart(daily, "QQQ", pd.DataFrame()).data) == 4
    assert len(app.weekly_chart(weekly, "QQQ", pd.DataFrame()).data) == 3
    assert len(app.drawdown_chart(drawdown, "QQQ", pd.DataFrame()).data) == 1


def test_rrg_seed_chart_adds_one_latest_marker_per_seed():
    app = importlib.import_module("app")
    history = pd.DataFrame(
        [
            {
                "seed_name": "SMH Universe",
                "date": pd.Timestamp("2024-01-05"),
                "rs_ratio": 99.0,
                "rs_momentum": 99.0,
                "quadrant": "Lagging",
                "symbols_used": "NVDA",
                "symbols_missing": "",
                "symbol_count": 1,
                "missing_count": 0,
            },
            {
                "seed_name": "SMH Universe",
                "date": pd.Timestamp("2024-01-12"),
                "rs_ratio": 101.0,
                "rs_momentum": 101.0,
                "quadrant": "Leading",
                "symbols_used": "NVDA",
                "symbols_missing": "",
                "symbol_count": 1,
                "missing_count": 0,
            },
            {
                "seed_name": "Healthcare Universe",
                "date": pd.Timestamp("2024-01-12"),
                "rs_ratio": 98.0,
                "rs_momentum": 102.0,
                "quadrant": "Improving",
                "symbols_used": "LLY",
                "symbols_missing": "",
                "symbol_count": 1,
                "missing_count": 0,
            },
        ]
    )

    latest = app.latest_seed_rrg(history)
    fig = app.rrg_seed_chart(history, latest, "QQQ")
    marker_traces = [trace for trace in fig.data if trace.mode == "markers+text"]

    assert len(marker_traces) == 2
    assert {trace.name for trace in marker_traces} == {"SMH Universe", "Healthcare Universe"}


def test_rrg_chart_highlights_selected_symbol_and_dims_others():
    app = importlib.import_module("app")
    history = pd.DataFrame(
        [
            {"symbol": "NVDA", "date": pd.Timestamp("2024-01-05"), "rs_ratio": 99.0, "rs_momentum": 99.0, "quadrant": "Lagging"},
            {"symbol": "NVDA", "date": pd.Timestamp("2024-01-12"), "rs_ratio": 101.0, "rs_momentum": 101.0, "quadrant": "Leading"},
            {"symbol": "AMD", "date": pd.Timestamp("2024-01-05"), "rs_ratio": 98.0, "rs_momentum": 102.0, "quadrant": "Improving"},
            {"symbol": "AMD", "date": pd.Timestamp("2024-01-12"), "rs_ratio": 97.0, "rs_momentum": 101.0, "quadrant": "Improving"},
        ]
    )
    latest = app.latest_rrg_with_metadata(
        history,
        pd.DataFrame(
            [
                {"symbol": "NVDA", "name": "NVIDIA", "theme": "Semiconductors", "score": 90.0, "benchmark_adjusted_score": 91.0},
                {"symbol": "AMD", "name": "AMD", "theme": "Semiconductors", "score": 80.0, "benchmark_adjusted_score": 81.0},
            ]
        ),
    )

    fig = app.rrg_chart(history, latest, "QQQ", selected_symbol="NVDA")
    nvda_marker = next(trace for trace in fig.data if trace.name == "NVDA")
    amd_marker = next(trace for trace in fig.data if trace.name == "AMD")
    nvda_path = next(trace for trace in fig.data if trace.name == "NVDA path")
    amd_path = next(trace for trace in fig.data if trace.name == "AMD path")

    assert nvda_marker.marker.size > amd_marker.marker.size
    assert nvda_path.line.width > amd_path.line.width
    assert amd_path.opacity < nvda_path.opacity


def test_rrg_selection_helpers_extract_and_toggle_symbol():
    app = importlib.import_module("app")
    fig = app.rrg_chart(
        pd.DataFrame(
            [
                {"symbol": "NVDA", "date": pd.Timestamp("2024-01-05"), "rs_ratio": 101.0, "rs_momentum": 101.0, "quadrant": "Leading"},
            ]
        ),
        pd.DataFrame(),
        "QQQ",
    )
    event = {"selection": {"points": [{"curve_number": 1}]}}

    clicked = app.rrg_selection_from_event(event, fig)

    assert clicked == "NVDA"
    assert app.toggle_rrg_selection(None, clicked) == "NVDA"
    assert app.toggle_rrg_selection("NVDA", clicked) is None


def test_symbol_from_dataframe_selection_uses_selected_row_position():
    app = importlib.import_module("app")
    ranked = pd.DataFrame(
        [
            {"symbol": "NVDA", "rank": 1},
            {"symbol": "AMD", "rank": 2},
        ]
    )

    symbol = app.symbol_from_dataframe_selection({"selection": {"rows": [1]}}, ranked)

    assert symbol == "AMD"
    assert app.symbol_from_dataframe_selection({"selection": {"rows": []}}, ranked) is None


def test_rrg_seed_chart_empty_state_has_no_traces():
    app = importlib.import_module("app")

    fig = app.rrg_seed_chart(pd.DataFrame(), pd.DataFrame(), "QQQ")

    assert len(fig.data) == 0


def test_scaled_basket_overlay_uses_benchmark_start_value():
    app = importlib.import_module("app")
    dates = pd.bdate_range("2024-01-01", periods=3)
    history = pd.DataFrame({"date": dates, "close": [200.0, 210.0, 220.0]})
    basket = pd.DataFrame({"date": dates, "basket_index": [1.0, 1.1, 0.9]})

    overlay = app.scaled_basket_overlay(history, basket, "close")

    assert overlay["value"].tolist() == pytest.approx([200.0, 220.0, 180.0])


def test_scaled_basket_overlay_rebases_at_first_shared_date():
    app = importlib.import_module("app")
    basket_dates = pd.bdate_range("2024-01-01", periods=4)
    history = pd.DataFrame(
        {
            "date": basket_dates[2:],
            "close": [200.0, 210.0],
        }
    )
    basket = pd.DataFrame(
        {
            "date": basket_dates,
            "basket_index": [1.0, 1.1, 1.2, 1.8],
        }
    )

    overlay = app.scaled_basket_overlay(history, basket, "close")

    assert overlay["value"].tolist() == pytest.approx([200.0, 300.0])


def test_scaled_basket_overlay_rejects_invalid_shared_base():
    app = importlib.import_module("app")
    dates = pd.bdate_range("2024-01-01", periods=2)
    history = pd.DataFrame({"date": dates, "close": [200.0, 210.0]})
    basket = pd.DataFrame({"date": dates, "basket_index": [0.0, 1.0]})

    overlay = app.scaled_basket_overlay(history, basket, "close")

    assert overlay.empty


def test_scaled_basket_overlay_returns_empty_without_shared_dates():
    app = importlib.import_module("app")
    history = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=2),
            "close": [200.0, 210.0],
        }
    )
    basket = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-02-01", periods=2),
            "basket_index": [1.0, 1.1],
        }
    )

    overlay = app.scaled_basket_overlay(history, basket, "close")

    assert overlay.empty
