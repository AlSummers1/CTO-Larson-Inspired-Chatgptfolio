from __future__ import annotations

from datetime import datetime, timezone
import json
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
) -> pd.DataFrame:
    """Fetch adjusted daily price history, using DuckDB as a local cache."""
    requested = _normalize_symbols(symbols)
    if not requested:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    cached = pd.DataFrame(columns=PRICE_COLUMNS) if refresh else read_cached_prices(requested, db_path)
    cached_symbols = set(cached["symbol"].unique()) if not cached.empty else set()
    to_download = requested if refresh else [symbol for symbol in requested if symbol not in cached_symbols]

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


def latest_cache_timestamp(db_path: Path = DEFAULT_DB_PATH) -> str:
    """Return a compact cache freshness label for the dashboard."""
    log = load_fetch_log(db_path)
    if log.empty:
        return "Never"
    fetched_at = pd.to_datetime(log["fetched_at"]).max()
    return fetched_at.strftime("%Y-%m-%d %H:%M UTC")


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
