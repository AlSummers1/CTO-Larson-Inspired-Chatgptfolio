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


def test_seed_rrg_options_exclude_manual_only_and_defaults_are_registered():
    app = importlib.import_module("app")

    options = app.seed_rrg_universe_options()
    defaults = app.default_rrg_seed_universes()

    assert app.MANUAL_ONLY_UNIVERSE not in options
    assert defaults == options


def test_rrg_seed_default_migration_selects_all_seeds_from_old_curated_default():
    app = importlib.import_module("app")

    options = app.seed_rrg_universe_options()

    assert app.rrg_seed_universes_after_default_migration(app.DEFAULT_RRG_SEED_UNIVERSES, options) == options
    assert app.rrg_seed_universes_after_default_migration(["Healthcare Universe"], options) == ["Healthcare Universe"]


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


def test_all_seed_universe_deduplicates_and_preserves_memberships():
    app = importlib.import_module("app")
    seed_frames = {
        "Seed A": pd.DataFrame([{"symbol": "AAA", "name": "A", "theme": "Theme", "fundamental_score": 0, "catalyst_score": 0, "notes": ""}, {"symbol": "B", "name": "B", "theme": "Theme", "fundamental_score": 0, "catalyst_score": 0, "notes": ""}]),
        "Seed B": pd.DataFrame([{"symbol": "AAA", "name": "A", "theme": "Theme", "fundamental_score": 0, "catalyst_score": 0, "notes": ""}]),
    }
    union = app.all_seed_universe(seed_frames)
    assert union["symbol"].tolist() == ["AAA", "B"]
    assert union.loc[union["symbol"] == "AAA", "seed_memberships"].iloc[0] == "Seed A · Seed B"


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


def test_portfolio_defaults_to_all_fixed_seeds_and_migrates_only_the_old_default():
    app = importlib.import_module("app")
    options = app.seed_rrg_universe_options()
    legacy_default = [label for label in app.DEFAULT_PORTFOLIO_SEED_UNIVERSES if label in options]
    custom = ["Healthcare Universe"]

    assert app.default_portfolio_seed_universes() == options
    assert app.portfolio_seed_universes_after_default_migration(None, options) == options
    assert app.portfolio_seed_universes_after_default_migration(legacy_default, options) == options
    assert app.portfolio_seed_universes_after_default_migration(custom, options) == custom


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


def test_rrg_seed_chart_highlights_selected_seed_and_dims_others():
    app = importlib.import_module("app")
    history = pd.DataFrame(
        [
            {"seed_name": "SMH Universe", "date": pd.Timestamp("2024-01-05"), "rs_ratio": 99.0, "rs_momentum": 99.0, "quadrant": "Lagging"},
            {"seed_name": "SMH Universe", "date": pd.Timestamp("2024-01-12"), "rs_ratio": 101.0, "rs_momentum": 101.0, "quadrant": "Leading"},
            {"seed_name": "Healthcare Universe", "date": pd.Timestamp("2024-01-05"), "rs_ratio": 98.0, "rs_momentum": 102.0, "quadrant": "Improving"},
            {"seed_name": "Healthcare Universe", "date": pd.Timestamp("2024-01-12"), "rs_ratio": 97.0, "rs_momentum": 101.0, "quadrant": "Improving"},
        ]
    )
    latest = app.latest_seed_rrg(history)

    fig = app.rrg_seed_chart(history, latest, "QQQ", selected_seed="SMH Universe")
    selected_marker = next(trace for trace in fig.data if trace.name == "SMH Universe")
    dimmed_marker = next(trace for trace in fig.data if trace.name == "Healthcare Universe")
    selected_path = next(trace for trace in fig.data if trace.mode == "lines" and trace.line.width == 4)
    dimmed_path = next(trace for trace in fig.data if trace.mode == "lines" and trace.line.width == 2)

    assert selected_marker.marker.size > dimmed_marker.marker.size
    assert dimmed_marker.marker.opacity < selected_marker.marker.opacity
    assert dimmed_path.opacity < selected_path.opacity


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


def test_dashboard_view_after_pending_request_applies_only_valid_request():
    app = importlib.import_module("app")

    assert app.dashboard_view_after_pending_request("Rankings", "Detail") == "Detail"
    assert app.dashboard_view_after_pending_request("Detail", "not-a-view") == "Detail"
    assert app.dashboard_view_after_pending_request("not-a-view", None) == "Rankings"


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


def test_paper_comparison_normalizes_each_series_and_warns_on_staggered_starts():
    app = importlib.import_module("app")
    nav = pd.DataFrame(
        {"date": pd.to_datetime(["2026-01-02", "2026-01-03"]), "nav": [98_000.0, 107_800.0]}
    )
    normalized = app.normalize_paper_series(nav)
    experiments = pd.DataFrame(
        {"fill_date": pd.to_datetime(["2026-01-02", "2026-02-03"])}
    )

    assert normalized["normalized_nav"].tolist() == pytest.approx([100_000.0, 110_000.0])
    assert "not directly like-for-like" in app.paper_comparison_start_warning(experiments)
    assert app.paper_comparison_start_warning(experiments.iloc[:1]) is None


def test_shared_paper_market_references_use_the_earliest_fill_and_exclude_technical():
    app = importlib.import_module("app")
    experiments = pd.DataFrame(
        {
            "portfolio_id": [24, 23, 26],
            "fill_date": pd.to_datetime(["2026-07-14", "2026-07-13", None]),
        }
    )
    benchmarks = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "QQQ", "QQQ", "TECHNICAL", "TECHNICAL"],
            "date": pd.to_datetime(["2026-07-13", "2026-07-14"] * 3),
            "nav": [99_000.0, 108_900.0, 101_000.0, 111_100.0, 100_000.0, 120_000.0],
        }
    )

    shared = app.shared_market_benchmark_series(benchmarks)

    assert app.paper_comparison_benchmark_anchor(experiments) == 23
    assert list(shared) == ["SPY", "QQQ"]
    assert all(frame["normalized_nav"].iloc[0] == pytest.approx(100_000.0) for frame in shared.values())
    assert "TECHNICAL" not in shared
    assert app.paper_chart_trace_visibility("SPY") == "legendonly"
    assert app.paper_chart_trace_visibility("BTC-USD") == "legendonly"
    assert app.paper_chart_trace_visibility("TECHNICAL") is True


def test_seed_display_labels_are_compact_and_cash_rule_status_is_explicit():
    app = importlib.import_module("app")

    assert app.seed_universe_display_name("Rare Earths and Strategic Materials Universe") == "Rare Earths & Materials"
    assert app.seed_universe_display_name("Magnificent Seven Universe") == "Mag7"
    assert app.seed_universe_display_name("Space Exploration Universe") == "Space Exploration"
    assert app.seed_universe_label_after_migration("Magnificent Seven + SpaceX Universe", app.seed_universe_options()) == "Magnificent Seven Universe"
    assert app.seed_universe_label_after_migration("removed universe", app.seed_universe_options()) == "SMH Universe"
    assert app.benchmark_regime_cash_rule_status("Bull", True).endswith("reserve: 0%.")
    assert app.benchmark_regime_cash_rule_status("Transition", True).endswith("reserve: 15%.")
    assert app.benchmark_regime_cash_rule_status("Bear", True).endswith("reserve: 35%.")
    assert "off" in app.benchmark_regime_cash_rule_status("Bear", False)


def test_paper_experiment_display_name_includes_sensitivity_and_basket():
    app = importlib.import_module("app")

    label = app.paper_experiment_display_name(
        {"name": "Run 26 — gpt-5.5 — 2026-07-15", "sensitivity": "Balanced", "basket_description": "Long healthcare & chips"}
    )

    assert label == "Run 26 — gpt-5.5 — 2026-07-15 — Balanced — Long healthcare & chips"


def test_paper_holding_display_frame_resolves_names_and_cached_values():
    app = importlib.import_module("app")
    positions = pd.DataFrame(
        [
            {"symbol": "nvda", "shares": 10, "average_cost": 90.0},
            {"symbol": "XYZ", "shares": 2, "average_cost": 5.0},
        ]
    )
    prices = pd.DataFrame(
        [
            {"symbol": "NVDA", "date": "2026-07-18", "close": 110.0},
            {"symbol": "NVDA", "date": "2026-07-19", "close": 120.0},
        ]
    )
    metadata = pd.DataFrame([{"symbol": "NVDA", "name": "NVIDIA"}])

    display = app.paper_holding_display_frame(positions, prices, metadata)

    nvda = display[display["symbol"] == "NVDA"].iloc[0]
    unknown = display[display["symbol"] == "XYZ"].iloc[0]
    assert nvda["label"] == "NVDA — NVIDIA"
    assert nvda["latest_close"] == pytest.approx(120.0)
    assert nvda["value"] == pytest.approx(1200.0)
    assert unknown["name"] == "XYZ"
    assert unknown["value"] == pytest.approx(10.0)


def test_paper_rrg_regime_frame_aggregates_expected_quadrants():
    app = importlib.import_module("app")
    holdings = pd.DataFrame(
        [
            {"symbol": "AAA", "value": 60.0},
            {"symbol": "BBB", "value": 40.0},
        ]
    )
    snapshot = pd.DataFrame(
        [
            {"symbol": "AAA", "rrg_quadrant": "Leading"},
            {"symbol": "BBB", "rrg_quadrant": "Lagging"},
        ]
    )

    regime = app.paper_rrg_regime_frame(holdings, snapshot)

    assert regime.loc[regime["rrg_quadrant"] == "Leading", "weight"].iloc[0] == pytest.approx(0.6)
    assert regime.loc[regime["rrg_quadrant"] == "Lagging", "weight"].iloc[0] == pytest.approx(0.4)
    assert regime["rrg_quadrant"].tolist() == ["Leading", "Improving", "Weakening", "Lagging", "Unresolved"]


def test_paper_rrg_snapshot_handles_unresolved_holdings_without_crashing():
    app = importlib.import_module("app")
    prices = pd.DataFrame([{"symbol": "AAA", "date": "2026-07-19", "close": 10.0}])

    snapshot = app.paper_rrg_snapshot(prices, ["AAA"], "SPY", "2026-07-19")

    assert snapshot["symbol"].tolist() == ["AAA"]
    assert snapshot["rrg_quadrant"].tolist() == ["Unresolved"]


def test_paper_rrg_regime_sentence_templates_are_deterministic():
    app = importlib.import_module("app")

    def regime(weights: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"rrg_quadrant": quadrant, "value": weight * 100.0, "weight": weight}
                for quadrant, weight in weights.items()
            ]
        )

    previous_date = "2026-07-18"
    assert app.paper_rrg_regime_sentence(
        regime({"Leading": 0.70, "Lagging": 0.30}),
        regime({"Leading": 0.40, "Lagging": 0.60}),
        previous_date,
    ) == "🟢 RRG improved since 2026-07-18: Leading/Improving exposure rose by 30.0 pts to 70.0%."
    assert app.paper_rrg_regime_sentence(
        regime({"Leading": 0.40, "Weakening": 0.60}),
        regime({"Leading": 0.70, "Weakening": 0.30}),
        previous_date,
    ) == "🔴 RRG weakened since 2026-07-18: Weakening/Lagging exposure rose by 30.0 pts to 60.0%."
    assert app.paper_rrg_regime_sentence(
        regime({"Leading": 0.45, "Weakening": 0.45, "Unresolved": 0.10}),
        regime({"Leading": 0.30, "Weakening": 0.30, "Unresolved": 0.40}),
        previous_date,
    ) == "🟡 RRG rotation was mixed since 2026-07-18: largest move was Unresolved by 30.0 pts."
    assert app.paper_rrg_regime_sentence(
        regime({"Leading": 0.60, "Lagging": 0.40}),
        regime({"Leading": 0.60, "Lagging": 0.40}),
        previous_date,
    ) == "➖ No RRG regime change since 2026-07-18; largest sleeve remains Leading at 60.0%."
    assert app.paper_rrg_regime_sentence(
        regime({"Unresolved": 1.0}),
        regime({"Unresolved": 1.0}),
        previous_date,
    ) == "⚪ RRG regime is unresolved because the holdings or benchmark lack enough price history."


def test_paper_portfolio_pie_figures_are_non_empty_for_filled_experiment_helpers():
    app = importlib.import_module("app")
    holdings = pd.DataFrame(
        [
            {"symbol": "AAA", "label": "AAA — Alpha", "value": 75.0},
            {"symbol": "BBB", "label": "BBB — Beta", "value": 25.0},
        ]
    )
    regime = app.paper_rrg_regime_frame(
        holdings,
        pd.DataFrame(
            [
                {"symbol": "AAA", "rrg_quadrant": "Leading"},
                {"symbol": "BBB", "rrg_quadrant": "Improving"},
            ]
        ),
    )

    holdings_fig = app.paper_value_pie_chart(holdings, "label", "value", "Current holdings by ticker")
    regime_fig = app.paper_value_pie_chart(regime, "rrg_quadrant", "value", "Current RRG regime")

    assert len(holdings_fig.data) == 1
    assert len(regime_fig.data) == 1
    assert sum(holdings_fig.data[0]["values"]) == pytest.approx(100.0)
    assert sum(regime_fig.data[0]["values"]) == pytest.approx(100.0)
    assert list(regime_fig.data[0]["labels"]) == ["Leading", "Improving", "Weakening", "Lagging", "Unresolved"]


def test_portfolio_rrg_allocation_frame_uses_latest_stock_quadrants_and_excludes_cash():
    app = importlib.import_module("app")
    applied_target = pd.DataFrame(
        [
            {"symbol": "AAA", "weight": 0.30},
            {"symbol": "BBB", "weight": 0.25},
            {"symbol": "CCC", "weight": 0.15},
            {"symbol": "Cash", "weight": 0.30},
        ]
    )
    rrg_history = pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2026-07-17", "quadrant": "Lagging"},
            {"symbol": "AAA", "date": "2026-07-24", "quadrant": "Leading"},
            {"symbol": "BBB", "date": "2026-07-24", "quadrant": "Weakening"},
        ]
    )

    allocation = app.portfolio_rrg_allocation_frame(applied_target, rrg_history)

    assert allocation["rrg_quadrant"].tolist() == ["Leading", "Improving", "Weakening", "Lagging", "Unresolved"]
    assert allocation.loc[allocation["rrg_quadrant"] == "Leading", "weight"].iloc[0] == pytest.approx(0.30)
    assert allocation.loc[allocation["rrg_quadrant"] == "Weakening", "weight"].iloc[0] == pytest.approx(0.25)
    assert allocation.loc[allocation["rrg_quadrant"] == "Unresolved", "weight"].iloc[0] == pytest.approx(0.15)
    assert allocation["weight"].sum() == pytest.approx(0.70)


def test_portfolio_rrg_allocation_pie_is_non_empty_for_applied_holdings():
    app = importlib.import_module("app")
    allocation = app.portfolio_rrg_allocation_frame(
        pd.DataFrame(
            [
                {"symbol": "AAA", "weight": 0.60},
                {"symbol": "BBB", "weight": 0.40},
            ]
        ),
        pd.DataFrame(
            [
                {"symbol": "AAA", "date": "2026-07-24", "quadrant": "Improving"},
                {"symbol": "BBB", "date": "2026-07-24", "quadrant": "Lagging"},
            ]
        ),
    )

    fig = app.allocation_pie_chart(allocation, "rrg_quadrant", "Allocation by Stock RRG Quadrant")

    assert len(fig.data) == 1
    assert list(fig.data[0]["labels"]) == ["Leading", "Improving", "Weakening", "Lagging", "Unresolved"]
    assert sum(fig.data[0]["values"]) == pytest.approx(1.0)


def _theme_prices(symbols: dict[str, list[float]]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=80)
    rows = []
    for symbol, closes in symbols.items():
        for offset, close in enumerate(closes):
            rows.append({"symbol": symbol, "date": dates[offset], "close": close})
    return pd.DataFrame(rows)


def test_theme_breadth_history_uses_20d_eligibility_weekly_smoothing_and_equal_weight_roc():
    app = importlib.import_module("app")
    prices = _theme_prices(
        {
            "AAA": [100.0 + index for index in range(80)],
            "BBB": [50.0 + 2 * index for index in range(80)],
        }
    )

    history = app.cached_theme_breadth_history(prices, ("AAA", "BBB"))
    latest = history.iloc[-1]
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    expected_index = (1.0 + close.pct_change(fill_method=None).mean(axis=1).fillna(0.0)).cumprod()

    assert (pd.to_datetime(history["date"]).dt.weekday == 4).all()
    assert latest["eligible_count"] == 2
    assert latest["coverage"] == pytest.approx(1.0)
    assert latest["breadth"] == pytest.approx(1.0)
    assert latest["breadth_4w_average"] == pytest.approx(1.0)
    assert latest["roc_50d"] == pytest.approx(expected_index.pct_change(50).loc[pd.Timestamp(latest["date"])])
    assert latest["roc_min_coverage"] == pytest.approx(1.0)


def test_theme_breadth_short_history_reduces_coverage_and_is_unresolved():
    app = importlib.import_module("app")
    prices = _theme_prices(
        {
            "AAA": [100.0 + index for index in range(80)],
            "BBB": [50.0 + index for index in range(80)],
            "NEW": [20.0 + index for index in range(10)],
        }
    )
    prices.loc[prices["symbol"] == "NEW", "date"] = pd.bdate_range("2024-04-08", periods=10)

    history = app.cached_theme_breadth_history(prices, ("AAA", "BBB", "NEW"))
    summary = app.latest_theme_breadth_summary(history, "Test Theme", 3).iloc[0]

    assert summary["eligible_count"] == 2
    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["missing_or_short_history"] == 1
    assert summary["confirmation"] == "Unresolved"


def test_theme_breadth_uses_each_symbols_last_20_observations_across_isolated_gaps():
    app = importlib.import_module("app")
    prices = _theme_prices(
        {
            "AAA": [100.0 + index for index in range(80)],
            "BBB": [50.0 + index for index in range(80)],
        }
    )
    gap_date = prices.loc[prices["symbol"] == "AAA", "date"].sort_values().iloc[-10]
    prices = prices[~((prices["symbol"] == "AAA") & (prices["date"] == gap_date))]

    latest = app.cached_theme_breadth_history(prices, ("AAA", "BBB")).iloc[-1]

    assert latest["eligible_count"] == 2
    assert latest["coverage"] == pytest.approx(1.0)


def test_theme_breadth_does_not_carry_stale_weekly_values_into_partial_coverage():
    app = importlib.import_module("app")
    prices = _theme_prices(
        {
            "AAA": [100.0 + index for index in range(80)],
            "BBB": [50.0 + index for index in range(80)],
            "CCC": [25.0 + index for index in range(80)],
        }
    )
    final_date = prices["date"].max()
    prices = prices[~(prices["symbol"].isin(["AAA", "BBB"]) & prices["date"].eq(final_date))]

    history = app.cached_theme_breadth_history(prices, ("AAA", "BBB", "CCC"))
    latest = history.iloc[-1]

    assert latest["eligible_count"] == 1
    assert latest["coverage"] == pytest.approx(1 / 3)
    assert latest["breadth"] == pytest.approx(1.0)
    assert history["breadth"].between(0.0, 1.0).all()
    assert history["breadth_4w_average"].dropna().between(0.0, 1.0).all()


def test_theme_breadth_display_scales_ratios_for_percent_formatted_columns():
    app = importlib.import_module("app")
    summary = pd.DataFrame(
        [
            {
                "breadth": 0.8,
                "breadth_4w_average": 0.75,
                "breadth_change": -0.05,
                "roc_50d": 0.125,
                "coverage": 0.6,
                "eligible_count": 12,
            }
        ]
    )

    display = app.theme_breadth_display(summary).iloc[0]

    assert display["breadth"] == pytest.approx(80.0)
    assert display["breadth_4w_average"] == pytest.approx(75.0)
    assert display["breadth_change"] == pytest.approx(-5.0)
    assert display["roc_50d"] == pytest.approx(12.5)
    assert display["coverage"] == pytest.approx(60.0)
    assert display["eligible_count"] == 12


def test_theme_breadth_confirmation_templates_are_deterministic():
    app = importlib.import_module("app")
    base = {"coverage": 1.0, "roc_min_coverage": 1.0}

    assert app.theme_breadth_confirmation({**base, "breadth": 0.8, "breadth_4w_average": 0.6, "roc_50d": 0.1}) == "Confirmed advance"
    assert app.theme_breadth_confirmation({**base, "breadth": 0.4, "breadth_4w_average": 0.6, "roc_50d": -0.1}) == "Confirmed deterioration"
    assert app.theme_breadth_confirmation({**base, "breadth": 0.8, "breadth_4w_average": 0.6, "roc_50d": -0.1}) == "Mixed"
    assert app.theme_breadth_confirmation({**base, "breadth": 0.8, "breadth_4w_average": 0.6, "roc_50d": 0.1, "coverage": 0.5}) == "Unresolved"


def test_theme_breadth_ui_helpers_produce_charts_and_register_view():
    app = importlib.import_module("app")
    prices = _theme_prices({"AAA": [100.0 + index for index in range(80)]})
    history = app.cached_theme_breadth_history(prices, ("AAA",))

    breadth_figure = app.theme_breadth_chart(history, "SMH Universe")
    roc_figure = app.theme_roc_chart(history, "SMH Universe")

    assert "Theme Breadth" in app.VIEW_OPTIONS
    assert len(breadth_figure.data) == 2
    assert len(roc_figure.data) == 1


def test_theme_breadth_packet_rows_cover_every_fixed_seed_with_actual_observation_date():
    app = importlib.import_module("app")
    prices = _theme_prices({"AAA": [100.0 + index for index in range(80)]})
    seed_frames = {
        label: pd.DataFrame([{"symbol": "AAA"}])
        for label in app.seed_rrg_universe_options()
    }
    latest_date = pd.to_datetime(prices["date"]).max().date()

    rows = app.theme_breadth_packet_rows(prices, seed_frames, latest_date)

    assert [row["seed_universe"] for row in rows] == app.seed_rrg_universe_options()
    assert all(row["observation_date"] == latest_date.isoformat() for row in rows)
    assert all(row["eligible_count"] == 1 for row in rows)
