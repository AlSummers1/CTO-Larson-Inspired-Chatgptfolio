from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache
import json
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from .universe import normalize_symbol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "market_data.duckdb"
PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]
PORTFOLIO_HOLDING_COLUMNS = [
    "snapshot_id",
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
NUMERIC_PORTFOLIO_COLUMNS = {
    "weight",
    "allocator_score",
    "score",
    "benchmark_adjusted_score",
    "relative_strength_score",
    "drawdown_52w",
    "extension_sma50d",
}
PAPER_EXPERIMENT_CAPITAL = 100_000.0
LEGACY_PAPER_PORTFOLIO_NAME = "Chatgptfolio Paper Portfolio"

# These compact names are deliberately presentation-only.  The frozen decision
# still retains the full source basket names in its audit trail.
PAPER_BASKET_LABELS = {
    "SMH Universe": "chips",
    "Healthcare Universe": "healthcare",
    "GLP-1 and Obesity Universe": "healthcare",
    "Defence and Space Universe": "defense & space",
    "Data Center Infrastructure Universe": "data-center infrastructure",
    "Power Grid and Electrification Universe": "grid electrification",
    "Utilities Universe": "utilities",
    "Clean Energy Universe": "clean energy",
    "Nuclear Energy Universe": "nuclear energy",
    "Rare Earths and Strategic Materials Universe": "strategic materials",
    "Crypto Universe": "crypto assets",
    "Quantum Universe": "quantum",
    "Cybersecurity Universe": "cybersecurity",
    "Magnificent Seven Universe": "mega-cap technology",
    "Space Exploration Universe": "space exploration",
}


def _import_duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required. Install dependencies with: pip install -r requirements.txt") from exc
    return duckdb


def _connect(db_path: Path = DEFAULT_DB_PATH):
    duckdb = _import_duckdb()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def _paper_experiment_name(run_id: int, model: object, approved_at: object) -> str:
    """Build the stable, human-readable name for an independent experiment."""
    approval_date = pd.to_datetime(approved_at, errors="coerce")
    approval_label = approval_date.date().isoformat() if pd.notna(approval_date) else "unknown-date"
    model_label = str(model or "unknown-model")
    return f"Run {int(run_id)} — {model_label} — {approval_label}"


def _json_object(value: object) -> dict[str, object]:
    """Return a JSON object without letting legacy audit rows break the UI."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _paper_experiment_sensitivity(config_json: object) -> str:
    """Translate frozen turnover settings into a short, human-readable label."""
    config = _json_object(config_json)
    explicit = str(config.get("sensitivity", config.get("experiment_sensitivity", ""))).strip().title()
    if explicit in {"Conservative", "Balanced", "Aggressive"}:
        return explicit
    turnover = pd.to_numeric(pd.Series([config.get("max_turnover")]), errors="coerce").iloc[0]
    if pd.isna(turnover):
        # Earlier approved runs did not record this setting.  Their Lab default
        # was a 35% turnover limit, which is the balanced profile.
        return "Balanced"
    if float(turnover) <= 0.25:
        return "Conservative"
    if float(turnover) >= 0.50:
        return "Aggressive"
    return "Balanced"


def _paper_experiment_basket_description(decision_json: object) -> str:
    """Summarize the largest frozen thematic sleeves in an approved decision."""
    decision = _json_object(decision_json)
    holdings = decision.get("holdings", [])
    if not isinstance(holdings, list):
        return "Long selected equities"
    exposure: dict[str, float] = {}
    for holding in holdings:
        if not isinstance(holding, dict):
            continue
        raw_baskets = str(
            holding.get("seed_universe") or holding.get("basket") or holding.get("sector") or holding.get("theme") or ""
        )
        if not raw_baskets:
            continue
        weight = pd.to_numeric(pd.Series([holding.get("weight", 0.0)]), errors="coerce").iloc[0]
        position_weight = float(weight) if pd.notna(weight) and float(weight) > 0 else 1.0
        baskets = [value.strip() for value in raw_baskets.split(" · ") if value.strip()]
        if not baskets:
            continue
        for basket in baskets:
            label = PAPER_BASKET_LABELS.get(basket, basket.removesuffix(" Universe").lower())
            exposure[label] = exposure.get(label, 0.0) + position_weight / len(baskets)
    if not exposure:
        return "Long selected equities"
    labels = [label for label, _ in sorted(exposure.items(), key=lambda item: (-item[1], item[0]))[:3]]
    if len(labels) == 1:
        return f"Long {labels[0]}"
    if len(labels) == 2:
        return f"Long {labels[0]} & {labels[1]}"
    return f"Long {labels[0]}, {labels[1]} & {labels[2]}"


def _ensure_paper_experiment(conn, run_id: int, model: object, approved_at: object) -> int:
    """Return the one independent experiment belonging to a Lab run."""
    existing = conn.execute(
        "SELECT portfolio_id FROM lab_paper_portfolios WHERE run_id = ? ORDER BY portfolio_id LIMIT 1",
        [int(run_id)],
    ).fetchone()
    if existing:
        return int(existing[0])
    return int(
        conn.execute(
            """INSERT INTO lab_paper_portfolios
               (name, initial_capital, created_at, status, run_id, model, approval_date)
               VALUES (?, ?, ?, 'active', ?, ?, ?) RETURNING portfolio_id""",
            [
                _paper_experiment_name(run_id, model, approved_at),
                PAPER_EXPERIMENT_CAPITAL,
                datetime.now(timezone.utc).replace(tzinfo=None),
                int(run_id),
                str(model or "unknown-model"),
                pd.to_datetime(approved_at, errors="coerce").date() if pd.notna(pd.to_datetime(approved_at, errors="coerce")) else None,
            ],
        ).fetchone()[0]
    )


def _migrate_legacy_pending_paper_rebalances(conn) -> list[int]:
    """Split old default-portfolio pending approvals into immutable experiments.

    Only pending rows in the historical default destination are moved.  Fills,
    decisions, costs, and frozen research remain untouched, making this safe to
    run on every database initialization.
    """
    pending = conn.execute(
        """SELECT r.rebalance_id, r.run_id, r.approved_at, l.model
           FROM lab_paper_rebalances r
           JOIN lab_paper_portfolios p ON r.portfolio_id = p.portfolio_id
           LEFT JOIN lab_runs l ON r.run_id = l.run_id
           WHERE r.status = 'pending' AND p.name = ?
           ORDER BY r.rebalance_id""",
        [LEGACY_PAPER_PORTFOLIO_NAME],
    ).fetchall()
    migrated: list[int] = []
    for rebalance_id, run_id, approved_at, model in pending:
        experiment_id = _ensure_paper_experiment(conn, int(run_id), model, approved_at)
        conn.execute(
            "UPDATE lab_paper_rebalances SET portfolio_id = ? WHERE rebalance_id = ?",
            [experiment_id, int(rebalance_id)],
        )
        migrated.append(int(rebalance_id))
    if migrated:
        conn.execute(
            "UPDATE lab_paper_portfolios SET status = 'legacy' WHERE name = ?",
            [LEGACY_PAPER_PORTFOLIO_NAME],
        )
    return migrated


@lru_cache(maxsize=32)
def init_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the local DuckDB cache tables if they do not exist."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, date)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fetch_log (
                symbol VARCHAR,
                fetched_at TIMESTAMP,
                start_date DATE,
                end_date DATE,
                rows INTEGER,
                status VARCHAR,
                message VARCHAR
            )
            """
        )
        conn.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS portfolio_snapshot_id_seq START 1
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                snapshot_id INTEGER DEFAULT nextval('portfolio_snapshot_id_seq'),
                created_at TIMESTAMP,
                benchmark VARCHAR,
                benchmark_regime VARCHAR,
                selected_seeds VARCHAR,
                sensitivity VARCHAR,
                config_hash VARCHAR,
                config_json VARCHAR
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_holdings (
                snapshot_id INTEGER,
                symbol VARCHAR,
                name VARCHAR,
                seed_universe VARCHAR,
                weight DOUBLE,
                allocator_score DOUBLE,
                score DOUBLE,
                benchmark_adjusted_score DOUBLE,
                relative_strength_score DOUBLE,
                larsson_regime VARCHAR,
                daily_trend VARCHAR,
                weekly_trend VARCHAR,
                monthly_trend VARCHAR,
                drawdown_52w DOUBLE,
                extension_sma50d DOUBLE,
                rrg_quadrant VARCHAR,
                reason_codes VARCHAR,
                data_warnings VARCHAR
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_rebalance_log (
                snapshot_id INTEGER,
                symbol VARCHAR,
                previous_weight DOUBLE,
                target_weight DOUBLE,
                delta DOUBLE,
                action VARCHAR,
                reason_codes VARCHAR
            )
            """
        )
        conn.execute("ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS config_hash VARCHAR")
        for column in ["daily_trend", "weekly_trend", "monthly_trend", "rrg_quadrant"]:
            conn.execute(f"ALTER TABLE portfolio_holdings ADD COLUMN IF NOT EXISTS {column} VARCHAR")
        for column in ["drawdown_52w", "extension_sma50d"]:
            conn.execute(f"ALTER TABLE portfolio_holdings ADD COLUMN IF NOT EXISTS {column} DOUBLE")
        conn.execute("CREATE SEQUENCE IF NOT EXISTS lab_run_id_seq START 1")
        conn.execute("CREATE TABLE IF NOT EXISTS lab_research_snapshots (snapshot_hash VARCHAR PRIMARY KEY, as_of_date DATE, created_at TIMESTAMP, packet_json VARCHAR)")
        conn.execute("CREATE TABLE IF NOT EXISTS lab_runs (run_id INTEGER DEFAULT nextval('lab_run_id_seq'), created_at TIMESTAMP, as_of_date DATE, benchmark VARCHAR, model VARCHAR, returned_model VARCHAR, status VARCHAR, packet_hash VARCHAR, prompt_hash VARCHAR, config_json VARCHAR, response_json VARCHAR, raw_response VARCHAR, validation_json VARCHAR, error_message VARCHAR)")
        conn.execute("CREATE TABLE IF NOT EXISTS lab_paper_orders (run_id INTEGER, symbol VARCHAR, action VARCHAR, previous_weight DOUBLE, target_weight DOUBLE, notional DOUBLE, estimated_cost DOUBLE, status VARCHAR)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_security_master (
                snapshot_hash VARCHAR, symbol VARCHAR, name VARCHAR, sector VARCHAR, industry VARCHAR,
                eligible BOOLEAN, effective_date DATE, source_url VARCHAR, fetched_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_evidence (
                evidence_id VARCHAR PRIMARY KEY, snapshot_hash VARCHAR, symbol VARCHAR, category VARCHAR,
                title VARCHAR, summary VARCHAR, source_url VARCHAR, publisher VARCHAR, published_at TIMESTAMP,
                fetched_at TIMESTAMP, content_hash VARCHAR, status VARCHAR, source_tier VARCHAR, source_domain VARCHAR
            )
        """)
        conn.execute("ALTER TABLE lab_evidence ADD COLUMN IF NOT EXISTS source_tier VARCHAR")
        conn.execute("ALTER TABLE lab_evidence ADD COLUMN IF NOT EXISTS source_domain VARCHAR")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_factor_scores (
                snapshot_hash VARCHAR, symbol VARCHAR, technical_score DOUBLE, fundamental_score DOUBLE,
                combined_score DOUBLE, liquidity_score DOUBLE, sector VARCHAR, score_json VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_nav (
                run_id INTEGER, date DATE, nav DOUBLE, benchmark_nav DOUBLE, technical_nav DOUBLE,
                drawdown DOUBLE, turnover DOUBLE
            )
        """)
        # The original lab tables are an immutable model/audit record.  These
        # tables intentionally keep approved paper execution separate, so a
        # rejected or merely exploratory model run can never alter portfolio
        # performance.
        conn.execute("CREATE SEQUENCE IF NOT EXISTS lab_paper_portfolio_id_seq START 1")
        conn.execute("CREATE SEQUENCE IF NOT EXISTS lab_paper_rebalance_id_seq START 1")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_paper_portfolios (
                portfolio_id INTEGER DEFAULT nextval('lab_paper_portfolio_id_seq') PRIMARY KEY,
                name VARCHAR UNIQUE, initial_capital DOUBLE, created_at TIMESTAMP, status VARCHAR,
                run_id INTEGER, model VARCHAR, approval_date DATE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_paper_rebalances (
                rebalance_id INTEGER DEFAULT nextval('lab_paper_rebalance_id_seq') PRIMARY KEY,
                portfolio_id INTEGER, run_id INTEGER, approved_at TIMESTAMP, effective_date DATE,
                status VARCHAR, starting_nav DOUBLE, ending_cash DOUBLE, config_json VARCHAR,
                decision_json VARCHAR, note VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_paper_fills (
                rebalance_id INTEGER, symbol VARCHAR, action VARCHAR, quantity DOUBLE,
                fill_price DOUBLE, notional DOUBLE, estimated_cost DOUBLE, filled_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_paper_positions (
                rebalance_id INTEGER, symbol VARCHAR, shares DOUBLE, average_cost DOUBLE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_paper_daily_nav (
                portfolio_id INTEGER, date DATE, nav DOUBLE, cash DOUBLE, drawdown DOUBLE,
                turnover DOUBLE, PRIMARY KEY (portfolio_id, date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_paper_benchmark_nav (
                portfolio_id INTEGER, symbol VARCHAR, date DATE, nav DOUBLE,
                PRIMARY KEY (portfolio_id, symbol, date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_source_health (
                snapshot_hash VARCHAR, source VARCHAR, status VARCHAR, severity VARCHAR,
                freshness VARCHAR, coverage DOUBLE, last_successful_at TIMESTAMP,
                failure_reason VARCHAR, retry_state VARCHAR, quality_score DOUBLE
            )
        """)
        conn.execute("ALTER TABLE lab_runs ADD COLUMN IF NOT EXISTS request_settings_json VARCHAR")
        conn.execute("ALTER TABLE lab_runs ADD COLUMN IF NOT EXISTS context_metadata_json VARCHAR")
        conn.execute("ALTER TABLE lab_paper_portfolios ADD COLUMN IF NOT EXISTS run_id INTEGER")
        conn.execute("ALTER TABLE lab_paper_portfolios ADD COLUMN IF NOT EXISTS model VARCHAR")
        conn.execute("ALTER TABLE lab_paper_portfolios ADD COLUMN IF NOT EXISTS approval_date DATE")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lab_paper_valuation_state (
                portfolio_id INTEGER PRIMARY KEY, input_fingerprint VARCHAR, rebuilt_at TIMESTAMP
            )
        """)
        _migrate_legacy_pending_paper_rebalances(conn)
        # The former singleton is retained as an auditable record only.  It is
        # never an active experiment or a destination for future approvals.
        conn.execute(
            "UPDATE lab_paper_portfolios SET status = 'legacy' WHERE name = ? AND run_id IS NULL",
            [LEGACY_PAPER_PORTFOLIO_NAME],
        )


def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()))


def read_cached_prices(symbols: list[str], db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Read cached adjusted OHLCV data for the requested symbols."""
    requested = _normalize_symbols(symbols)
    if not requested:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    init_database(db_path)
    symbol_frame = pd.DataFrame({"symbol": requested})
    with _connect(db_path) as conn:
        conn.register("_requested_symbols", symbol_frame)
        frame = conn.execute(
            """
            SELECT p.symbol, p.date, p.open, p.high, p.low, p.close, p.volume
            FROM prices p
            INNER JOIN _requested_symbols s ON p.symbol = s.symbol
            ORDER BY p.symbol, p.date
            """
        ).fetchdf()
        conn.unregister("_requested_symbols")

    if frame.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[PRICE_COLUMNS]


def write_prices(prices: pd.DataFrame, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Replace cached rows for symbols present in the supplied price frame."""
    if prices.empty:
        return

    prepared = prices[PRICE_COLUMNS].copy()
    prepared["symbol"] = prepared["symbol"].astype(str).map(normalize_symbol)
    prepared["date"] = pd.to_datetime(prepared["date"]).dt.date

    init_database(db_path)
    with _connect(db_path) as conn:
        conn.register("_prices_to_write", prepared)
        conn.execute(
            """
            DELETE FROM prices
            USING (SELECT DISTINCT symbol FROM _prices_to_write) incoming
            WHERE prices.symbol = incoming.symbol
            """
        )
        conn.execute(
            """
            INSERT INTO prices
            SELECT symbol, date, open, high, low, close, volume
            FROM _prices_to_write
            """
        )
        conn.unregister("_prices_to_write")


def _record_fetch_log(
    symbols: list[str],
    prices: pd.DataFrame,
    failed: set[str],
    db_path: Path,
) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        symbol_prices = prices[prices["symbol"] == symbol] if not prices.empty else pd.DataFrame()
        if symbol in failed or symbol_prices.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "fetched_at": now,
                    "start_date": None,
                    "end_date": None,
                    "rows": 0,
                    "status": "failed",
                    "message": "No rows returned by yfinance.",
                }
            )
            continue
        rows.append(
            {
                "symbol": symbol,
                "fetched_at": now,
                "start_date": symbol_prices["date"].min(),
                "end_date": symbol_prices["date"].max(),
                "rows": len(symbol_prices),
                "status": "ok",
                "message": "",
            }
        )

    init_database(db_path)
    with _connect(db_path) as conn:
        conn.register("_fetch_log_rows", pd.DataFrame(rows))
        conn.execute("INSERT INTO fetch_log SELECT * FROM _fetch_log_rows")
        conn.unregister("_fetch_log_rows")


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required. Install dependencies with: pip install -r requirements.txt") from exc
    return yf


def _column_lookup(frame: pd.DataFrame, desired: str) -> str | None:
    normalized = {str(column).strip().lower().replace("_", " "): column for column in frame.columns}
    return normalized.get(desired)


def _standardize_downloaded_symbol(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    open_col = _column_lookup(frame, "open")
    high_col = _column_lookup(frame, "high")
    low_col = _column_lookup(frame, "low")
    close_col = _column_lookup(frame, "close") or _column_lookup(frame, "adj close")
    volume_col = _column_lookup(frame, "volume")
    if close_col is None:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    standardized = pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.to_datetime(frame.index),
            "open": pd.to_numeric(frame[open_col], errors="coerce") if open_col else pd.NA,
            "high": pd.to_numeric(frame[high_col], errors="coerce") if high_col else pd.NA,
            "low": pd.to_numeric(frame[low_col], errors="coerce") if low_col else pd.NA,
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
            "volume": pd.to_numeric(frame[volume_col], errors="coerce") if volume_col else 0,
        }
    )
    standardized = standardized.dropna(subset=["date", "close"]).copy()
    standardized["date"] = pd.to_datetime(standardized["date"], utc=True).dt.tz_convert(None).dt.normalize()
    return standardized[PRICE_COLUMNS]


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    if not isinstance(raw.columns, pd.MultiIndex):
        return raw

    upper_levels = [
        [str(value).upper() for value in raw.columns.get_level_values(level)]
        for level in range(raw.columns.nlevels)
    ]
    for level_index, values in enumerate(upper_levels):
        if symbol in values:
            actual_value = raw.columns.get_level_values(level_index)[values.index(symbol)]
            return raw.xs(actual_value, axis=1, level=level_index)
    return pd.DataFrame()


def download_prices(symbols: list[str], years: int = 10) -> pd.DataFrame:
    """Download adjusted daily OHLCV data from yfinance."""
    requested = _normalize_symbols(symbols)
    if not requested:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    yf = _import_yfinance()
    raw = yf.download(
        tickers=" ".join(requested),
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    frames = []
    for symbol in requested:
        symbol_frame = _extract_symbol_frame(raw, symbol)
        standardized = _standardize_downloaded_symbol(symbol, symbol_frame)
        if not standardized.empty:
            frames.append(standardized)
    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def fetch_price_history(
    symbols: list[str],
    refresh: bool = False,
    years: int = 10,
    db_path: Path = DEFAULT_DB_PATH,
    download_missing: bool = False,
) -> pd.DataFrame:
    """Fetch adjusted daily price history, using DuckDB as a local cache."""
    requested = _normalize_symbols(symbols)
    if not requested:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    cached = pd.DataFrame(columns=PRICE_COLUMNS) if refresh else read_cached_prices(requested, db_path)
    cached_symbols = set(cached["symbol"].unique()) if not cached.empty else set()
    to_download = requested if refresh else ([symbol for symbol in requested if symbol not in cached_symbols] if download_missing else [])

    if to_download:
        downloaded = download_prices(to_download, years=years)
        if not downloaded.empty:
            write_prices(downloaded, db_path)
        downloaded_symbols = set(downloaded["symbol"].unique()) if not downloaded.empty else set()
        _record_fetch_log(to_download, downloaded, set(to_download) - downloaded_symbols, db_path)

    return read_cached_prices(requested, db_path)


def load_fetch_log(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Return fetch attempts ordered from newest to oldest."""
    init_database(db_path)
    with _connect(db_path) as conn:
        frame = conn.execute(
            """
            SELECT symbol, fetched_at, start_date, end_date, rows, status, message
            FROM fetch_log
            ORDER BY fetched_at DESC, symbol
            """
        ).fetchdf()
    return frame


def cached_symbol_count(symbols: Iterable[str], db_path: Path = DEFAULT_DB_PATH) -> int:
    """Count symbols with any local history without triggering a network fetch."""
    requested = _normalize_symbols(symbols)
    if not requested:
        return 0
    init_database(db_path)
    with _connect(db_path) as conn:
        frame = pd.DataFrame({"symbol": requested})
        conn.register("_cache_symbols", frame)
        count = conn.execute("SELECT count(DISTINCT p.symbol) FROM prices p JOIN _cache_symbols s ON p.symbol = s.symbol").fetchone()[0]
        conn.unregister("_cache_symbols")
    return int(count or 0)


def latest_cache_timestamp(db_path: Path = DEFAULT_DB_PATH) -> str:
    """Return a compact cache freshness label for the dashboard."""
    log = load_fetch_log(db_path)
    if log.empty:
        return "Never"
    fetched_at = pd.to_datetime(log["fetched_at"]).max()
    return fetched_at.strftime("%Y-%m-%d %H:%M UTC")


def save_lab_research_snapshot(packet: dict[str, object], db_path: Path = DEFAULT_DB_PATH) -> str:
    """Persist an immutable, point-in-time research packet and return its hash."""
    init_database(db_path)
    payload = json.dumps(packet, sort_keys=True, default=str, separators=(",", ":"))
    packet_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO lab_research_snapshots VALUES (?, ?, ?, ?)",
            [packet_hash, packet.get("as_of_date"), datetime.now(timezone.utc).replace(tzinfo=None), payload],
        )
    return packet_hash


def save_lab_research_details(packet_hash: str, packet: dict[str, object], db_path: Path = DEFAULT_DB_PATH) -> None:
    """Persist source-level evidence and factor scores without ever storing credentials."""
    init_database(db_path)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    evidence = packet.get("evidence", [])
    scores = packet.get("candidate_scores", [])
    master = packet.get("security_master", [])
    with _connect(db_path) as conn:
        if isinstance(evidence, list) and evidence:
            frame = pd.DataFrame(evidence)
            required = ["evidence_id", "symbol", "category", "title", "summary", "source_url", "publisher", "published_at", "fetched_at", "content_hash", "status", "source_tier", "source_domain"]
            for column in required:
                if column not in frame:
                    frame[column] = None
            # RSS feeds commonly use RFC 2822 dates (for example, "Fri, 10 Jul
            # 2026 17:19:00 GMT"). Normalize every source timestamp before the
            # DuckDB insert so source-specific date formats never break a run.
            for column in ["published_at", "fetched_at"]:
                frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True).dt.tz_convert(None)
            frame["snapshot_hash"] = packet_hash
            conn.register("_lab_evidence", frame[["evidence_id", "snapshot_hash", *required[1:]]])
            conn.execute("INSERT OR IGNORE INTO lab_evidence (evidence_id, snapshot_hash, symbol, category, title, summary, source_url, publisher, published_at, fetched_at, content_hash, status, source_tier, source_domain) SELECT * FROM _lab_evidence")
            conn.unregister("_lab_evidence")
        if isinstance(scores, list) and scores:
            frame = pd.DataFrame(scores)
            for column in ["symbol", "technical_score", "fundamental_score", "combined_score", "liquidity_score", "sector", "score_json"]:
                if column not in frame:
                    frame[column] = None
            frame["snapshot_hash"] = packet_hash
            conn.register("_lab_scores", frame[["snapshot_hash", "symbol", "technical_score", "fundamental_score", "combined_score", "liquidity_score", "sector", "score_json"]])
            conn.execute("INSERT INTO lab_factor_scores SELECT * FROM _lab_scores")
            conn.unregister("_lab_scores")
        if isinstance(master, list) and master:
            frame = pd.DataFrame(master)
            for column in ["symbol", "name", "sector", "industry", "eligible", "effective_date", "source_url", "fetched_at"]:
                if column not in frame:
                    frame[column] = None
            frame["snapshot_hash"] = packet_hash
            frame["fetched_at"] = frame["fetched_at"].fillna(now)
            conn.register("_lab_master", frame[["snapshot_hash", "symbol", "name", "sector", "industry", "eligible", "effective_date", "source_url", "fetched_at"]])
            conn.execute("INSERT INTO lab_security_master SELECT * FROM _lab_master")
            conn.unregister("_lab_master")
        coverage = packet.get("coverage", {}) if isinstance(packet, dict) else {}
        diagnostics = packet.get("source_diagnostics", {}) if isinstance(packet, dict) else {}
        source_rows = []
        source_labels = {
            "prices": "Price cache",
            "fundamentals": "SEC EDGAR fundamentals",
            "events": "SEC EDGAR filings",
            "macro": "FRED macro",
            "news": "GDELT news discovery",
            "alpha_vantage_news": "Alpha Vantage ticker-linked news",
        }
        for key, label in source_labels.items():
            status = str(coverage.get(key, "unresolved"))
            packet_quality = packet.get("source_quality", {}).get(key, {}) if isinstance(packet, dict) else {}
            severity = "low" if status == "available" else ("critical" if key == "prices" or status == "unavailable" else "unresolved")
            source_rows.append({
                "snapshot_hash": packet_hash, "source": label, "status": status,
                "severity": severity, "freshness": str(packet_quality.get("freshness", "current" if status == "available" else "unknown")),
                "coverage": float(packet_quality.get("completeness", 1.0 if status == "available" else 0.0)),
                "last_successful_at": now if status == "available" else None,
                "failure_reason": None if status == "available" else str(diagnostics.get(key, "Connector returned no usable evidence; retry on the next refresh.")),
                "retry_state": "ready" if status == "available" else "retryable",
                "quality_score": float(packet_quality.get("score", 100.0 if status == "available" else 0.0)),
            })
        source_frame = pd.DataFrame(source_rows)
        conn.register("_lab_source_health", source_frame)
        conn.execute("INSERT INTO lab_source_health SELECT * FROM _lab_source_health")
        conn.unregister("_lab_source_health")


def save_lab_run(record: dict[str, object], db_path: Path = DEFAULT_DB_PATH) -> int:
    """Store an LLM decision without storing credentials."""
    init_database(db_path)
    fields = ["created_at", "as_of_date", "benchmark", "model", "returned_model", "status", "packet_hash", "prompt_hash", "config_json", "response_json", "raw_response", "validation_json", "error_message", "request_settings_json", "context_metadata_json"]
    values = [record.get(field) for field in fields]
    values[0] = values[0] or datetime.now(timezone.utc).replace(tzinfo=None)
    with _connect(db_path) as conn:
        return int(conn.execute(
            "INSERT INTO lab_runs (" + ", ".join(fields) + ") VALUES (" + ", ".join(["?"] * len(fields)) + ") RETURNING run_id",
            values,
        ).fetchone()[0])


def save_lab_paper_orders(run_id: int, orders: pd.DataFrame, db_path: Path = DEFAULT_DB_PATH) -> None:
    if orders.empty:
        return
    init_database(db_path)
    columns = ["run_id", "symbol", "action", "previous_weight", "target_weight", "notional", "estimated_cost", "status"]
    prepared = orders.copy()
    prepared["run_id"] = run_id
    with _connect(db_path) as conn:
        conn.register("_lab_orders", prepared[columns])
        conn.execute("INSERT INTO lab_paper_orders SELECT * FROM _lab_orders")
        conn.unregister("_lab_orders")


def load_lab_runs(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    init_database(db_path)
    with _connect(db_path) as conn:
        return conn.execute("SELECT run_id, created_at, as_of_date, benchmark, model, returned_model, status, packet_hash, error_message FROM lab_runs ORDER BY run_id DESC").fetchdf()


def default_paper_portfolio(initial_capital: float = 100_000.0, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Return the historical default destination, retained only for audit compatibility."""
    init_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT portfolio_id FROM lab_paper_portfolios WHERE name = ?", [LEGACY_PAPER_PORTFOLIO_NAME]).fetchone()
        if row:
            return int(row[0])
        return int(conn.execute(
            "INSERT INTO lab_paper_portfolios (name, initial_capital, created_at, status) VALUES (?, ?, ?, ?) RETURNING portfolio_id",
            [LEGACY_PAPER_PORTFOLIO_NAME, initial_capital, datetime.now(timezone.utc).replace(tzinfo=None), "legacy"],
        ).fetchone()[0])


def migrate_legacy_pending_paper_rebalances(db_path: Path = DEFAULT_DB_PATH) -> list[int]:
    """Run the idempotent one-time split for old default-portfolio approvals."""
    init_database(db_path)
    with _connect(db_path) as conn:
        return _migrate_legacy_pending_paper_rebalances(conn)


def load_paper_experiments(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Return active independent paper experiments with their current summary."""
    init_database(db_path)
    columns = [
        "portfolio_id", "name", "run_id", "model", "status", "approval_date", "fill_date",
        "initial_capital", "nav", "return", "drawdown", "holding_count", "has_pending",
        "sensitivity", "basket_description",
    ]
    with _connect(db_path) as conn:
        portfolios = conn.execute(
            """SELECT portfolio_id, name, run_id, model, status, approval_date, initial_capital
               FROM lab_paper_portfolios WHERE status = 'active' ORDER BY approval_date, portfolio_id"""
        ).fetchdf()
        rebalances = conn.execute(
            """SELECT portfolio_id, rebalance_id, status, effective_date, config_json, decision_json
               FROM lab_paper_rebalances ORDER BY rebalance_id"""
        ).fetchdf()
        nav = conn.execute(
            """SELECT portfolio_id, date, nav, drawdown FROM lab_paper_daily_nav
               ORDER BY portfolio_id, date"""
        ).fetchdf()
        positions = conn.execute(
            """SELECT r.portfolio_id, r.rebalance_id, p.symbol
               FROM lab_paper_rebalances r JOIN lab_paper_positions p ON r.rebalance_id = p.rebalance_id
               WHERE r.status = 'executed'"""
        ).fetchdf()
    if portfolios.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for _, experiment in portfolios.iterrows():
        portfolio_id = int(experiment["portfolio_id"])
        experiment_rebalances = rebalances[rebalances["portfolio_id"] == portfolio_id]
        executed = experiment_rebalances[experiment_rebalances["status"] == "executed"]
        experiment_nav = nav[nav["portfolio_id"] == portfolio_id]
        latest_rebalance = experiment_rebalances.sort_values("rebalance_id").iloc[-1] if not experiment_rebalances.empty else None
        latest_nav = float(experiment_nav.iloc[-1]["nav"]) if not experiment_nav.empty else float("nan")
        capital = pd.to_numeric(pd.Series([experiment["initial_capital"]]), errors="coerce").iloc[0]
        initial_capital = float(capital) if pd.notna(capital) and float(capital) > 0 else PAPER_EXPERIMENT_CAPITAL
        latest_rebalance_id = int(executed["rebalance_id"].max()) if not executed.empty else None
        holding_count = 0
        if latest_rebalance_id is not None:
            holding_count = int(len(positions[(positions["portfolio_id"] == portfolio_id) & (positions["rebalance_id"] == latest_rebalance_id)]))
        rows.append({
            "portfolio_id": portfolio_id,
            "name": experiment["name"],
            "run_id": experiment["run_id"],
            "model": experiment["model"],
            "status": experiment["status"],
            "approval_date": experiment["approval_date"],
            "fill_date": executed["effective_date"].max() if not executed.empty else None,
            "initial_capital": initial_capital,
            "nav": latest_nav,
            "return": latest_nav / initial_capital - 1.0 if pd.notna(latest_nav) and initial_capital else float("nan"),
            "drawdown": float(experiment_nav["drawdown"].min()) if not experiment_nav.empty else float("nan"),
            "holding_count": holding_count,
            "has_pending": bool((experiment_rebalances["status"] == "pending").any()),
            "sensitivity": _paper_experiment_sensitivity(latest_rebalance["config_json"] if latest_rebalance is not None else None),
            "basket_description": _paper_experiment_basket_description(latest_rebalance["decision_json"] if latest_rebalance is not None else None),
        })
    return pd.DataFrame(rows, columns=columns)


def load_paper_rebalances(portfolio_id: int | None = None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    portfolio_id = portfolio_id or default_paper_portfolio(db_path=db_path)
    init_database(db_path)
    with _connect(db_path) as conn:
        return conn.execute(
            """SELECT r.*, l.as_of_date, l.model, l.returned_model, l.packet_hash
               FROM lab_paper_rebalances r LEFT JOIN lab_runs l ON r.run_id = l.run_id
               WHERE r.portfolio_id = ? ORDER BY r.approved_at DESC""", [portfolio_id]
        ).fetchdf()


def load_paper_positions(portfolio_id: int | None = None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    portfolio_id = portfolio_id or default_paper_portfolio(db_path=db_path)
    init_database(db_path)
    with _connect(db_path) as conn:
        return conn.execute(
            """SELECT p.symbol, p.shares, p.average_cost, r.rebalance_id, r.effective_date, r.ending_cash
               FROM lab_paper_rebalances r JOIN lab_paper_positions p ON r.rebalance_id = p.rebalance_id
               WHERE r.portfolio_id = ? AND r.status = 'executed'
                 AND r.rebalance_id = (SELECT max(rebalance_id) FROM lab_paper_rebalances WHERE portfolio_id = ? AND status = 'executed')
               ORDER BY p.symbol""", [portfolio_id, portfolio_id]
        ).fetchdf()


def paper_portfolio_weights(prices: pd.DataFrame, portfolio_id: int | None = None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Value the latest executed positions at their most recent cached close."""
    positions = load_paper_positions(portfolio_id, db_path)
    if positions.empty:
        return pd.DataFrame(columns=["symbol", "weight"])
    latest = prices.copy()
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
    latest = latest.sort_values("date").groupby("symbol", as_index=False).tail(1)
    valued = positions.merge(latest[["symbol", "close"]], on="symbol", how="left")
    valued["value"] = valued["shares"] * valued["close"].fillna(0.0)
    cash = float(positions["ending_cash"].iloc[0] or 0.0)
    total = float(valued["value"].sum() + cash)
    valued["weight"] = valued["value"] / total if total else 0.0
    return valued[["symbol", "weight"]]


def paper_valuation_symbols(portfolio_id: int | None = None, db_path: Path = DEFAULT_DB_PATH) -> list[str]:
    """Return valuation symbols for one experiment, or every active experiment."""
    init_database(db_path)
    symbols = {"SPY", "QQQ", "GLD", "BTC-USD"}
    with _connect(db_path) as conn:
        if portfolio_id is None:
            portfolio_ids = [int(row[0]) for row in conn.execute(
                "SELECT portfolio_id FROM lab_paper_portfolios WHERE status = 'active'"
            ).fetchall()]
        else:
            portfolio_ids = [int(portfolio_id)]
        if not portfolio_ids:
            return sorted(symbols)
        placeholders = ", ".join(["?"] * len(portfolio_ids))
        held = conn.execute(
            f"""SELECT DISTINCT p.symbol FROM lab_paper_positions p
               JOIN lab_paper_rebalances r ON p.rebalance_id = r.rebalance_id
               WHERE r.portfolio_id IN ({placeholders}) AND r.status = 'executed'""", portfolio_ids
        ).fetchall()
        symbols.update(str(row[0]).upper() for row in held)
        packets = conn.execute(
            f"""SELECT r.status, r.config_json, r.decision_json, s.packet_json FROM lab_paper_rebalances r
               JOIN lab_runs l ON r.run_id = l.run_id
               JOIN lab_research_snapshots s ON l.packet_hash = s.snapshot_hash
               WHERE r.portfolio_id IN ({placeholders}) AND r.status IN ('executed', 'pending')""", portfolio_ids
        ).fetchall()
    for status, config_json, decision_json, packet_json in packets:
        try:
            config, packet = json.loads(config_json or "{}"), json.loads(packet_json)
            if status == "pending":
                decision = json.loads(decision_json or "{}")
                symbols.update(str(item.get("symbol", "")).upper() for item in decision.get("holdings", []) if item.get("symbol"))
            else:
                candidates = sorted(packet.get("candidates", []), key=lambda item: float(item.get("score", 0) or 0), reverse=True)
                symbols.update(str(item.get("symbol", "")).upper() for item in candidates[: int(config.get("max_holdings", 15))] if item.get("symbol"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return sorted(symbols)


def load_paper_portfolio_dashboard(portfolio_id: int | None = None, db_path: Path = DEFAULT_DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Paper Portfolio screen from one DuckDB connection."""
    portfolio_id = portfolio_id or default_paper_portfolio(db_path=db_path)
    init_database(db_path)
    with _connect(db_path) as conn:
        nav = conn.execute("SELECT date, nav, cash, drawdown, turnover FROM lab_paper_daily_nav WHERE portfolio_id = ? ORDER BY date", [portfolio_id]).fetchdf()
        benchmarks = conn.execute("SELECT symbol, date, nav FROM lab_paper_benchmark_nav WHERE portfolio_id = ? ORDER BY date", [portfolio_id]).fetchdf()
        rebalances = conn.execute(
            """SELECT r.*, l.as_of_date, l.model, l.returned_model, l.packet_hash
               FROM lab_paper_rebalances r LEFT JOIN lab_runs l ON r.run_id = l.run_id
               WHERE r.portfolio_id = ? ORDER BY r.approved_at DESC""", [portfolio_id]
        ).fetchdf()
        positions = conn.execute(
            """SELECT p.symbol, p.shares, p.average_cost, r.rebalance_id, r.effective_date, r.ending_cash
               FROM lab_paper_rebalances r JOIN lab_paper_positions p ON r.rebalance_id = p.rebalance_id
               WHERE r.portfolio_id = ? AND r.status = 'executed'
                 AND r.rebalance_id = (SELECT max(rebalance_id) FROM lab_paper_rebalances WHERE portfolio_id = ? AND status = 'executed')
               ORDER BY p.symbol""", [portfolio_id, portfolio_id]
        ).fetchdf()
    return nav, benchmarks, rebalances, positions


def load_paper_fills(portfolio_id: int, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load the immutable fill audit trail for one experiment."""
    init_database(db_path)
    with _connect(db_path) as conn:
        return conn.execute(
            """SELECT r.rebalance_id, r.run_id, r.effective_date, f.symbol, f.action, f.quantity,
                      f.fill_price, f.notional, f.estimated_cost, f.filled_at
               FROM lab_paper_rebalances r JOIN lab_paper_fills f ON r.rebalance_id = f.rebalance_id
               WHERE r.portfolio_id = ? ORDER BY f.filled_at, r.rebalance_id, f.symbol""",
            [int(portfolio_id)],
        ).fetchdf()


def approve_paper_rebalance(
    run_id: int,
    decision: dict[str, object],
    config: dict[str, object],
    as_of_date: date,
    portfolio_id: int | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Approve a run as its own $100,000 pending paper experiment.

    ``portfolio_id`` remains accepted for callers from the single-portfolio
    version, but is deliberately ignored: an approval must never rebalance or
    otherwise mutate a different experiment.
    """
    init_database(db_path)
    with _connect(db_path) as conn:
        existing = conn.execute("SELECT rebalance_id FROM lab_paper_rebalances WHERE run_id = ?", [run_id]).fetchone()
        if existing:
            return int(existing[0])
        run = conn.execute("SELECT model FROM lab_runs WHERE run_id = ?", [int(run_id)]).fetchone()
        model = run[0] if run else None
        approved_at = datetime.now(timezone.utc).replace(tzinfo=None)
        experiment_id = _ensure_paper_experiment(conn, int(run_id), model, approved_at)
        return int(conn.execute(
            """INSERT INTO lab_paper_rebalances
               (portfolio_id, run_id, approved_at, status, config_json, decision_json, note)
               VALUES (?, ?, ?, 'pending', ?, ?, ?) RETURNING rebalance_id""",
            [experiment_id, run_id, approved_at, json.dumps(config, default=str), json.dumps(decision, default=str), f"Awaiting next open after {as_of_date.isoformat()}"],
        ).fetchone()[0])


def _latest_positions_for_rebalance(conn, portfolio_id: int) -> tuple[pd.DataFrame, float]:
    prior = conn.execute(
        "SELECT rebalance_id, ending_cash FROM lab_paper_rebalances WHERE portfolio_id = ? AND status = 'executed' ORDER BY rebalance_id DESC LIMIT 1",
        [portfolio_id],
    ).fetchone()
    if not prior:
        capital = conn.execute("SELECT initial_capital FROM lab_paper_portfolios WHERE portfolio_id = ?", [portfolio_id]).fetchone()
        return pd.DataFrame(columns=["symbol", "shares", "average_cost"]), float(capital[0] if capital else 0.0)
    rows = conn.execute("SELECT symbol, shares, average_cost FROM lab_paper_positions WHERE rebalance_id = ?", [prior[0]]).fetchdf()
    return rows, float(prior[1] or 0.0)


def execute_pending_paper_rebalances(prices: pd.DataFrame, db_path: Path = DEFAULT_DB_PATH) -> list[int]:
    """Convert pending approvals into next-open simulated fills using supplied cache data.

    No fill is manufactured when a required open is unavailable; the approval remains
    pending until a subsequent cache refresh supplies the actual next trading day.
    """
    init_database(db_path)
    frame = prices.copy()
    if frame.empty:
        return []
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    completed: list[int] = []
    with _connect(db_path) as conn:
        pending = conn.execute(
            """SELECT r.rebalance_id, r.portfolio_id, r.decision_json, r.config_json, l.as_of_date
               FROM lab_paper_rebalances r JOIN lab_runs l ON r.run_id = l.run_id
               WHERE r.status = 'pending' ORDER BY r.rebalance_id"""
        ).fetchdf()
        for _, rebalance in pending.iterrows():
            decision = json.loads(rebalance["decision_json"])
            config = json.loads(rebalance["config_json"])
            targets = {str(item["symbol"]).upper(): float(item["weight"]) for item in decision.get("holdings", [])}
            previous, cash = _latest_positions_for_rebalance(conn, int(rebalance["portfolio_id"]))
            symbols = set(targets).union(set(previous.get("symbol", pd.Series(dtype=str)).astype(str).str.upper()))
            after = pd.to_datetime(rebalance["as_of_date"]).date()
            eligible_days = frame[(frame["symbol"].isin(symbols)) & (frame["date"] > after)] if symbols else pd.DataFrame()
            if eligible_days.empty:
                continue
            fill_date = min(eligible_days["date"])
            opens = frame[(frame["date"] == fill_date) & (frame["symbol"].isin(symbols))].set_index("symbol")
            if any(symbol not in opens.index or pd.isna(opens.loc[symbol, "open"]) for symbol in symbols):
                continue
            prior_shares = dict(zip(previous.get("symbol", []), previous.get("shares", [])))
            nav = cash + sum(float(prior_shares.get(symbol, 0.0)) * float(opens.loc[symbol, "open"]) for symbol in symbols)
            if nav <= 0:
                continue
            slippage = float(config.get("slippage_bps", 0.0)) / 10_000
            commission = float(config.get("commission_bps", 0.0)) / 10_000
            desired = {symbol: nav * weight / float(opens.loc[symbol, "open"]) for symbol, weight in targets.items()}
            fills: list[dict[str, object]] = []
            # Sells first: this makes cash accounting conservative and deterministic.
            for symbol in sorted(symbols):
                old, new = float(prior_shares.get(symbol, 0.0)), float(desired.get(symbol, 0.0))
                if new >= old - 1e-10:
                    continue
                quantity, price = old - new, float(opens.loc[symbol, "open"]) * (1 - slippage)
                notional, cost = quantity * price, quantity * price * commission
                cash += notional - cost
                fills.append({"symbol": symbol, "action": "SELL", "quantity": quantity, "fill_price": price, "notional": notional, "estimated_cost": cost})
            buys = [(symbol, max(0.0, float(desired.get(symbol, 0.0)) - float(prior_shares.get(symbol, 0.0)))) for symbol in sorted(targets)]
            buy_cost = sum(qty * float(opens.loc[symbol, "open"]) * (1 + slippage + commission) for symbol, qty in buys)
            scale = min(1.0, cash / buy_cost) if buy_cost > 0 else 1.0
            for symbol, quantity in buys:
                quantity *= max(0.0, scale)
                if quantity <= 1e-10:
                    continue
                price = float(opens.loc[symbol, "open"]) * (1 + slippage)
                notional, cost = quantity * price, quantity * price * commission
                cash -= notional + cost
                fills.append({"symbol": symbol, "action": "BUY", "quantity": quantity, "fill_price": price, "notional": notional, "estimated_cost": cost})
            final_shares = {symbol: float(desired.get(symbol, 0.0)) if symbol in targets else 0.0 for symbol in symbols}
            if scale < 1.0:
                for symbol, quantity in buys:
                    final_shares[symbol] = float(prior_shares.get(symbol, 0.0)) + quantity * scale
            conn.execute("DELETE FROM lab_paper_fills WHERE rebalance_id = ?", [int(rebalance["rebalance_id"])])
            conn.execute("DELETE FROM lab_paper_positions WHERE rebalance_id = ?", [int(rebalance["rebalance_id"])])
            for fill in fills:
                conn.execute("INSERT INTO lab_paper_fills VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [int(rebalance["rebalance_id"]), *fill.values(), datetime.combine(fill_date, datetime.min.time())])
            for symbol, shares in final_shares.items():
                if shares > 1e-10:
                    price = float(opens.loc[symbol, "open"])
                    conn.execute("INSERT INTO lab_paper_positions VALUES (?, ?, ?, ?)", [int(rebalance["rebalance_id"]), symbol, shares, price])
            conn.execute("UPDATE lab_paper_rebalances SET status = 'executed', effective_date = ?, starting_nav = ?, ending_cash = ?, note = ? WHERE rebalance_id = ?", [fill_date, nav, max(0.0, cash), "Filled at next cached market open", int(rebalance["rebalance_id"])])
            completed.append(int(rebalance["rebalance_id"]))
    return completed


def _paper_valuation_fingerprint(frame: pd.DataFrame, rebalances: pd.DataFrame) -> str:
    """Cheap, stable invalidation key for a portfolio valuation input set."""
    price_summary = {
        "rows": len(frame),
        "symbols": sorted(frame["symbol"].astype(str).str.upper().unique().tolist()),
        "start": str(pd.to_datetime(frame["date"], errors="coerce").min()),
        "end": str(pd.to_datetime(frame["date"], errors="coerce").max()),
        "close_sum": round(float(pd.to_numeric(frame["close"], errors="coerce").sum()), 6),
    }
    rebalance_summary = rebalances[[column for column in ["rebalance_id", "status", "effective_date", "ending_cash"] if column in rebalances]].astype(str).to_dict(orient="records")
    return hashlib.sha256(json.dumps({"prices": price_summary, "rebalances": rebalance_summary}, sort_keys=True).encode("utf-8")).hexdigest()


def rebuild_paper_nav(
    prices: pd.DataFrame,
    portfolio_id: int | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    benchmarks: tuple[str, ...] = ("SPY", "QQQ", "GLD", "BTC-USD"),
    force: bool = False,
) -> pd.DataFrame:
    """Recalculate daily NAV only when its price/rebalance inputs have changed."""
    portfolio_id = portfolio_id or default_paper_portfolio(db_path=db_path)
    init_database(db_path)
    frame = prices.copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", "nav", "cash", "drawdown", "turnover"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    rebalances = load_paper_rebalances(portfolio_id, db_path)
    rebalances = rebalances[rebalances["status"] == "executed"].sort_values("effective_date") if not rebalances.empty else rebalances
    if rebalances.empty:
        return pd.DataFrame(columns=["date", "nav", "cash", "drawdown", "turnover"])
    fingerprint = _paper_valuation_fingerprint(frame, rebalances)
    with _connect(db_path) as conn:
        state = conn.execute("SELECT input_fingerprint FROM lab_paper_valuation_state WHERE portfolio_id = ?", [portfolio_id]).fetchone()
    if not force and state and state[0] == fingerprint:
        return load_paper_nav(portfolio_id, db_path)
    positions_by_rebalance: dict[int, pd.DataFrame] = {}
    technical_targets: dict[int, list[str]] = {}
    with _connect(db_path) as conn:
        for _, rebalance in rebalances.iterrows():
            rebalance_id = int(rebalance["rebalance_id"])
            positions_by_rebalance[int(rebalance_id)] = conn.execute("SELECT symbol, shares FROM lab_paper_positions WHERE rebalance_id = ?", [int(rebalance_id)]).fetchdf()
            packet_row = conn.execute(
                """SELECT s.packet_json FROM lab_runs l JOIN lab_research_snapshots s ON l.packet_hash = s.snapshot_hash
                   WHERE l.run_id = ?""", [int(rebalance["run_id"])]
            ).fetchone()
            if packet_row:
                packet = json.loads(packet_row[0])
                config = json.loads(rebalance["config_json"] or "{}")
                candidates = sorted(packet.get("candidates", []), key=lambda item: float(item.get("score", item.get("technical_score", 0)) or 0), reverse=True)
                technical_targets[rebalance_id] = [str(item.get("symbol", "")).upper() for item in candidates[: int(config.get("max_holdings", 15))] if item.get("symbol")]
    start = pd.to_datetime(rebalances["effective_date"].min()).date()
    days = sorted(frame.loc[frame["date"] >= start, "date"].dropna().unique())
    closes = frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()
    rows = []
    technical_rows = []
    technical_shares: dict[str, float] = {}
    technical_cash = 0.0
    technical_rebalance_id: int | None = None
    for current_day in days:
        eligible = rebalances[pd.to_datetime(rebalances["effective_date"]).dt.date <= current_day]
        if eligible.empty or current_day not in closes.index:
            continue
        current = eligible.iloc[-1]
        holdings = positions_by_rebalance[int(current["rebalance_id"])]
        values = 0.0
        for _, holding in holdings.iterrows():
            if holding["symbol"] in closes.columns and pd.notna(closes.loc[current_day, holding["symbol"]]):
                values += float(holding["shares"]) * float(closes.loc[current_day, holding["symbol"]])
        nav = values + float(current["ending_cash"] or 0.0)
        rows.append({"portfolio_id": portfolio_id, "date": current_day, "nav": nav, "cash": float(current["ending_cash"] or 0.0), "turnover": 0.0})
        rebalance_id = int(current["rebalance_id"])
        if technical_rebalance_id != rebalance_id:
            if technical_shares:
                technical_cash += sum(float(shares) * float(closes.loc[current_day, symbol]) for symbol, shares in technical_shares.items() if symbol in closes.columns and pd.notna(closes.loc[current_day, symbol]))
            else:
                technical_cash = float(current["starting_nav"] or nav)
            targets = [symbol for symbol in technical_targets.get(rebalance_id, []) if symbol in closes.columns and pd.notna(closes.loc[current_day, symbol])]
            technical_shares = {symbol: technical_cash / len(targets) / float(closes.loc[current_day, symbol]) for symbol in targets} if targets else {}
            technical_cash = 0.0 if targets else technical_cash
            technical_rebalance_id = rebalance_id
        technical_value = technical_cash + sum(float(shares) * float(closes.loc[current_day, symbol]) for symbol, shares in technical_shares.items() if symbol in closes.columns and pd.notna(closes.loc[current_day, symbol]))
        if technical_value > 0:
            technical_rows.append({"portfolio_id": portfolio_id, "symbol": "TECHNICAL", "date": current_day, "nav": technical_value})
    nav_frame = pd.DataFrame(rows)
    if nav_frame.empty:
        return pd.DataFrame(columns=["date", "nav", "cash", "drawdown", "turnover"])
    nav_frame["drawdown"] = nav_frame["nav"] / nav_frame["nav"].cummax() - 1.0
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM lab_paper_daily_nav WHERE portfolio_id = ?", [portfolio_id])
        conn.register("_paper_nav", nav_frame[["portfolio_id", "date", "nav", "cash", "drawdown", "turnover"]])
        conn.execute("INSERT INTO lab_paper_daily_nav SELECT * FROM _paper_nav")
        conn.unregister("_paper_nav")
        conn.execute("DELETE FROM lab_paper_benchmark_nav WHERE portfolio_id = ?", [portfolio_id])
        initial_nav = float(nav_frame.iloc[0]["nav"])
        for symbol in benchmarks:
            if symbol not in closes.columns:
                continue
            series = closes[symbol].reindex(nav_frame["date"]).dropna()
            if series.empty:
                continue
            normalized = initial_nav * series / float(series.iloc[0])
            benchmark_frame = pd.DataFrame({"portfolio_id": portfolio_id, "symbol": symbol, "date": normalized.index, "nav": normalized.values})
            conn.register("_benchmark_nav", benchmark_frame)
            conn.execute("INSERT INTO lab_paper_benchmark_nav SELECT * FROM _benchmark_nav")
            conn.unregister("_benchmark_nav")
        if technical_rows:
            technical_frame = pd.DataFrame(technical_rows)
            conn.register("_technical_nav", technical_frame)
            conn.execute("INSERT INTO lab_paper_benchmark_nav SELECT * FROM _technical_nav")
            conn.unregister("_technical_nav")
        conn.execute(
            "INSERT OR REPLACE INTO lab_paper_valuation_state VALUES (?, ?, ?)",
            [portfolio_id, fingerprint, datetime.now(timezone.utc).replace(tzinfo=None)],
        )
    return nav_frame[["date", "nav", "cash", "drawdown", "turnover"]]


def load_paper_nav(portfolio_id: int | None = None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    portfolio_id = portfolio_id or default_paper_portfolio(db_path=db_path)
    init_database(db_path)
    with _connect(db_path) as conn:
        return conn.execute("SELECT date, nav, cash, drawdown, turnover FROM lab_paper_daily_nav WHERE portfolio_id = ? ORDER BY date", [portfolio_id]).fetchdf()


def load_paper_benchmarks(portfolio_id: int | None = None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    portfolio_id = portfolio_id or default_paper_portfolio(db_path=db_path)
    init_database(db_path)
    with _connect(db_path) as conn:
        return conn.execute("SELECT symbol, date, nav FROM lab_paper_benchmark_nav WHERE portfolio_id = ? ORDER BY date", [portfolio_id]).fetchdf()


def load_source_health(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    init_database(db_path)
    with _connect(db_path) as conn:
        return conn.execute("""SELECT * EXCLUDE (row_num) FROM (
            SELECT *, row_number() OVER (PARTITION BY source ORDER BY last_successful_at DESC NULLS LAST) AS row_num
            FROM lab_source_health) WHERE row_num = 1 ORDER BY source""").fetchdf()


def save_portfolio_snapshot(
    holdings: pd.DataFrame,
    metadata: dict[str, object],
    changes: pd.DataFrame | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Persist a model-portfolio snapshot and optional rebalance log."""
    init_database(db_path)
    snapshot_row = pd.DataFrame(
        [
            {
                "created_at": metadata.get("created_at", datetime.now(timezone.utc).replace(tzinfo=None)),
                "benchmark": metadata.get("benchmark", ""),
                "benchmark_regime": metadata.get("benchmark_regime", ""),
                "selected_seeds": metadata.get("selected_seeds", ""),
                "sensitivity": metadata.get("sensitivity", ""),
                "config_hash": metadata.get("config_hash", ""),
                "config_json": metadata.get("config_json", json.dumps({})),
            }
        ]
    )
    with _connect(db_path) as conn:
        conn.register("_portfolio_snapshot", snapshot_row)
        snapshot_id = conn.execute(
            """
            INSERT INTO portfolio_snapshots (
                created_at, benchmark, benchmark_regime, selected_seeds, sensitivity, config_hash, config_json
            )
            SELECT created_at, benchmark, benchmark_regime, selected_seeds, sensitivity, config_hash, config_json
            FROM _portfolio_snapshot
            RETURNING snapshot_id
            """
        ).fetchone()[0]
        conn.unregister("_portfolio_snapshot")

        if not holdings.empty:
            prepared_holdings = holdings.copy()
            prepared_holdings["snapshot_id"] = snapshot_id
            for column in PORTFOLIO_HOLDING_COLUMNS:
                if column not in prepared_holdings.columns:
                    prepared_holdings[column] = 0.0 if column in NUMERIC_PORTFOLIO_COLUMNS else ""
            conn.register("_portfolio_holdings", prepared_holdings[PORTFOLIO_HOLDING_COLUMNS])
            holding_columns_sql = ", ".join(PORTFOLIO_HOLDING_COLUMNS)
            conn.execute(
                f"""
                INSERT INTO portfolio_holdings ({holding_columns_sql})
                SELECT {holding_columns_sql}
                FROM _portfolio_holdings
                """
            )
            conn.unregister("_portfolio_holdings")

        if changes is not None and not changes.empty:
            log_columns = ["snapshot_id", "symbol", "previous_weight", "target_weight", "delta", "action", "reason_codes"]
            prepared_changes = changes.copy()
            prepared_changes["snapshot_id"] = snapshot_id
            for column in log_columns:
                if column not in prepared_changes.columns:
                    prepared_changes[column] = ""
            conn.register("_portfolio_changes", prepared_changes[log_columns])
            log_columns_sql = ", ".join(log_columns)
            conn.execute(
                f"""
                INSERT INTO portfolio_rebalance_log ({log_columns_sql})
                SELECT {log_columns_sql}
                FROM _portfolio_changes
                """
            )
            conn.unregister("_portfolio_changes")
    return int(snapshot_id)


def load_latest_portfolio_snapshot(db_path: Path = DEFAULT_DB_PATH) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the most recent model-portfolio snapshot."""
    return load_matching_portfolio_snapshot(db_path=db_path)


def load_matching_portfolio_snapshot(
    benchmark: str | None = None,
    selected_seeds: str | None = None,
    config_hash: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the latest portfolio snapshot, optionally scoped to matching settings."""
    init_database(db_path)
    filters = []
    params: list[object] = []
    if benchmark is not None:
        filters.append("benchmark = ?")
        params.append(benchmark)
    if selected_seeds is not None:
        filters.append("selected_seeds = ?")
        params.append(selected_seeds)
    if config_hash is not None:
        filters.append("config_hash = ?")
        params.append(config_hash)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    with _connect(db_path) as conn:
        snapshot = conn.execute(
            f"""
            SELECT snapshot_id, created_at, benchmark, benchmark_regime, selected_seeds, sensitivity, config_hash, config_json
            FROM portfolio_snapshots
            {where_clause}
            ORDER BY created_at DESC, snapshot_id DESC
            LIMIT 1
            """,
            params,
        ).fetchdf()
        if snapshot.empty:
            return pd.DataFrame(), {}
        snapshot_id = int(snapshot["snapshot_id"].iloc[0])
        holdings = conn.execute(
            """
            SELECT symbol, name, seed_universe, weight, allocator_score, score, benchmark_adjusted_score,
                   relative_strength_score, larsson_regime, daily_trend, weekly_trend, monthly_trend,
                   drawdown_52w, extension_sma50d, rrg_quadrant, reason_codes, data_warnings
            FROM portfolio_holdings
            WHERE snapshot_id = ?
            ORDER BY weight DESC, symbol
            """,
            [snapshot_id],
        ).fetchdf()
    metadata = snapshot.iloc[0].to_dict()
    holdings = _normalize_portfolio_holdings(holdings)
    return holdings, metadata


def load_portfolio_rebalance_log(
    db_path: Path = DEFAULT_DB_PATH,
    snapshot_id: int | None = None,
) -> pd.DataFrame:
    """Return saved portfolio rebalance changes from newest to oldest."""
    init_database(db_path)
    where_clause = "WHERE l.snapshot_id = ?" if snapshot_id is not None else ""
    params = [snapshot_id] if snapshot_id is not None else []
    with _connect(db_path) as conn:
        return conn.execute(
            f"""
            SELECT l.snapshot_id, s.created_at, l.symbol, l.previous_weight, l.target_weight,
                   l.delta, l.action, l.reason_codes
            FROM portfolio_rebalance_log l
            LEFT JOIN portfolio_snapshots s ON l.snapshot_id = s.snapshot_id
            {where_clause}
            ORDER BY s.created_at DESC, l.symbol
            """,
            params,
        ).fetchdf()


def _normalize_portfolio_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame(columns=[column for column in PORTFOLIO_HOLDING_COLUMNS if column != "snapshot_id"])
    normalized = holdings.copy()
    for column in PORTFOLIO_HOLDING_COLUMNS:
        if column == "snapshot_id":
            continue
        if column not in normalized.columns:
            normalized[column] = 0.0 if column in NUMERIC_PORTFOLIO_COLUMNS else ""
    return normalized[[column for column in PORTFOLIO_HOLDING_COLUMNS if column != "snapshot_id"]]
