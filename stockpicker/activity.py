from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import pandas as pd

from .data import DEFAULT_DB_PATH, _connect, init_database


EVENT_COLUMNS = [
    "event_id",
    "created_at",
    "effective_market_date",
    "event_type",
    "subject_type",
    "subject",
    "old_value",
    "new_value",
    "benchmark",
    "universe_hash",
    "signal_config_hash",
    "run_id",
    "evidence_id",
    "reason",
    "evidence_json",
]

SNAPSHOT_COLUMNS = [
    "run_id",
    "subject_type",
    "subject",
    "effective_market_date",
    "larsson_regime",
    "rrg_quadrant",
    "freshness_status",
    "score",
    "pass_200d",
    "pass_200w",
    "reason_codes",
    "data_warnings",
    "evidence_json",
]

FRESHNESS_ORDER = {"Fresh": 0, "Delayed": 1, "Stale": 2, "Unknown": 3}


@dataclass(frozen=True)
class SnapshotCaptureResult:
    captured: bool
    run_id: str | None
    event_count: int
    baseline: bool
    diagnostic: str = ""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _hash_payload(value: object, length: int | None = None) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def active_universe_hash(symbols: list[str] | tuple[str, ...] | set[str]) -> str:
    normalized = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    return _hash_payload(normalized, 16)


def canonical_signal_configuration(config: object | None) -> tuple[str, str]:
    if config is None:
        payload: object = {}
    elif is_dataclass(config):
        payload = asdict(config)
    elif isinstance(config, dict):
        payload = config
    else:
        payload = vars(config)
    encoded = _canonical_json(payload)
    return _hash_payload(payload, 16), encoded


def _value(row: pd.Series, column: str, default: object = "") -> object:
    value = row.get(column, default)
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return default
    return value


def _as_bool_or_none(value: object) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return bool(value)


def _latest_rrg_map(frame: pd.DataFrame | None, subject_column: str) -> dict[str, pd.Series]:
    if frame is None or frame.empty or subject_column not in frame.columns:
        return {}
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared.get("date"), errors="coerce")
    prepared = prepared.sort_values("date").drop_duplicates(subject_column, keep="last")
    return {str(row[subject_column]): row for _, row in prepared.iterrows()}


def _snapshot_rows(
    run_id: str,
    signals: pd.DataFrame,
    data_health: pd.DataFrame,
    stock_rrg: pd.DataFrame | None,
    seed_rrg: pd.DataFrame | None,
) -> list[dict[str, object]]:
    health_map = (
        {str(row["symbol"]): row for _, row in data_health.iterrows()}
        if not data_health.empty and "symbol" in data_health.columns
        else {}
    )
    stock_rrg_map = _latest_rrg_map(stock_rrg, "symbol")
    rows: list[dict[str, object]] = []
    for _, signal in signals.sort_values("symbol").iterrows():
        symbol = str(signal["symbol"])
        health = health_map.get(symbol, pd.Series(dtype=object))
        rrg = stock_rrg_map.get(symbol, pd.Series(dtype=object))
        effective_date = _value(health, "latest_price_date", _value(signal, "latest_date", None))
        evidence = {
            "symbol": symbol,
            "latest_price_date": effective_date,
            "reference_market_date": _value(health, "reference_market_date", None),
            "trading_sessions_behind": _value(health, "trading_sessions_behind", None),
            "coverage_pct_60": _value(health, "coverage_pct_60", None),
            "missing_sessions_60": _value(health, "missing_sessions_60", None),
            "unchanged_close_streak": _value(health, "unchanged_close_streak", None),
            "fetch_status": _value(health, "fetch_status", "unknown"),
            "calendar_approximate": _value(health, "calendar_approximate", False),
            "larsson_regime": _value(signal, "larsson_regime", ""),
            "daily_trend": _value(signal, "daily_trend", ""),
            "weekly_trend": _value(signal, "weekly_trend", ""),
            "monthly_trend": _value(signal, "monthly_trend", ""),
            "score": _value(signal, "score", None),
            "benchmark_adjusted_score": _value(signal, "benchmark_adjusted_score", None),
            "pass_200d": _as_bool_or_none(signal.get("pass_200d")),
            "pass_200w": _as_bool_or_none(signal.get("pass_200w")),
            "rrg_quadrant": _value(rrg, "quadrant", ""),
            "reason_codes": _value(signal, "reason_codes", ""),
            "data_warnings": _value(signal, "data_warnings", ""),
            "name": _value(signal, "name", ""),
            "theme": _value(signal, "theme", ""),
        }
        rows.append(
            {
                "run_id": run_id,
                "subject_type": "stock",
                "subject": symbol,
                "effective_market_date": effective_date,
                "larsson_regime": evidence["larsson_regime"],
                "rrg_quadrant": evidence["rrg_quadrant"],
                "freshness_status": _value(health, "freshness_status", "Unknown"),
                "score": evidence["score"],
                "pass_200d": evidence["pass_200d"],
                "pass_200w": evidence["pass_200w"],
                "reason_codes": evidence["reason_codes"],
                "data_warnings": evidence["data_warnings"],
                "evidence_json": _canonical_json(evidence),
            }
        )

    if seed_rrg is not None and not seed_rrg.empty and "seed_name" in seed_rrg.columns:
        latest_seeds = seed_rrg.copy()
        latest_seeds["date"] = pd.to_datetime(latest_seeds["date"], errors="coerce")
        latest_seeds = latest_seeds.sort_values("date").drop_duplicates("seed_name", keep="last")
        for _, seed in latest_seeds.sort_values("seed_name").iterrows():
            evidence = {
                "seed_name": str(seed["seed_name"]),
                "date": _value(seed, "date", None),
                "rrg_quadrant": _value(seed, "quadrant", ""),
                "rs_ratio": _value(seed, "rs_ratio", None),
                "rs_momentum": _value(seed, "rs_momentum", None),
                "symbols_used": _value(seed, "symbols_used", ""),
                "symbols_missing": _value(seed, "symbols_missing", ""),
            }
            rows.append(
                {
                    "run_id": run_id,
                    "subject_type": "seed_universe",
                    "subject": str(seed["seed_name"]),
                    "effective_market_date": _value(seed, "date", None),
                    "larsson_regime": "",
                    "rrg_quadrant": evidence["rrg_quadrant"],
                    "freshness_status": "",
                    "score": None,
                    "pass_200d": None,
                    "pass_200w": None,
                    "reason_codes": "",
                    "data_warnings": "",
                    "evidence_json": _canonical_json(evidence),
                }
            )
    return rows


def _event_key(
    event_type: str,
    subject: str,
    effective_market_date: object,
    old_value: object,
    new_value: object,
    benchmark: str,
    signal_config_hash: str,
) -> str:
    return _hash_payload(
        {
            "event_type": event_type,
            "subject": subject,
            "effective_market_date": effective_market_date,
            "old_value": old_value,
            "new_value": new_value,
            "benchmark": benchmark,
            "signal_config_hash": signal_config_hash,
        }
    )


def _make_event(
    *,
    event_type: str,
    subject_type: str,
    subject: str,
    old_value: object,
    new_value: object,
    effective_market_date: object,
    benchmark: str,
    universe_hash: str,
    signal_config_hash: str,
    run_id: str | None,
    reason: str,
    evidence: dict[str, object],
    created_at: datetime | None = None,
) -> dict[str, object]:
    key = _event_key(
        event_type,
        subject,
        effective_market_date,
        old_value,
        new_value,
        benchmark,
        signal_config_hash,
    )
    evidence_id = f"EV-{key[:12].upper()}"
    return {
        "event_id": evidence_id,
        "idempotency_key": key,
        "created_at": created_at or _utc_now(),
        "effective_market_date": effective_market_date,
        "event_type": event_type,
        "subject_type": subject_type,
        "subject": subject,
        "old_value": str(old_value),
        "new_value": str(new_value),
        "benchmark": benchmark,
        "universe_hash": universe_hash,
        "signal_config_hash": signal_config_hash,
        "run_id": run_id,
        "evidence_id": evidence_id,
        "reason": reason,
        "evidence_json": _canonical_json(evidence),
    }


def _derive_events(
    current: list[dict[str, object]],
    previous: pd.DataFrame,
    *,
    benchmark: str,
    universe_hash: str,
    signal_config_hash: str,
    run_id: str,
) -> list[dict[str, object]]:
    if previous.empty:
        return []
    prior = {
        (str(row.subject_type), str(row.subject)): row
        for row in previous.itertuples(index=False)
    }
    events: list[dict[str, object]] = []
    for row in current:
        key = (str(row["subject_type"]), str(row["subject"]))
        old = prior.get(key)
        if old is None:
            continue
        common = {
            "subject_type": str(row["subject_type"]),
            "subject": str(row["subject"]),
            "effective_market_date": row["effective_market_date"],
            "benchmark": benchmark,
            "universe_hash": universe_hash,
            "signal_config_hash": signal_config_hash,
            "run_id": run_id,
        }
        evidence = {
            "previous_snapshot": old.run_id,
            "current_snapshot": run_id,
            "benchmark": benchmark,
            "signal_config_hash": signal_config_hash,
            "old_evidence": json.loads(old.evidence_json or "{}"),
            "new_evidence": json.loads(str(row["evidence_json"]) or "{}"),
        }

        new_regime = str(row["larsson_regime"] or "")
        old_regime = str(old.larsson_regime or "")
        if new_regime and old_regime and new_regime != old_regime:
            event_type = "benchmark_regime_change" if row["subject"] == benchmark else "larsson_regime_change"
            events.append(
                _make_event(
                    event_type=event_type,
                    old_value=old_regime,
                    new_value=new_regime,
                    reason=f"Larsson-proxy regime changed from {old_regime} to {new_regime}.",
                    evidence=evidence,
                    **common,
                )
            )

        new_quadrant = str(row["rrg_quadrant"] or "")
        old_quadrant = str(old.rrg_quadrant or "")
        if new_quadrant and old_quadrant and new_quadrant != old_quadrant:
            event_type = "seed_rrg_quadrant_change" if row["subject_type"] == "seed_universe" else "stock_rrg_quadrant_change"
            events.append(
                _make_event(
                    event_type=event_type,
                    old_value=old_quadrant,
                    new_value=new_quadrant,
                    reason=f"RRG quadrant changed from {old_quadrant} to {new_quadrant} using the stable 10/10-week configuration.",
                    evidence=evidence,
                    **common,
                )
            )

        if row["subject_type"] == "stock":
            new_freshness = str(row["freshness_status"] or "Unknown")
            old_freshness = str(old.freshness_status or "Unknown")
            if new_freshness != old_freshness:
                new_order = FRESHNESS_ORDER.get(new_freshness, FRESHNESS_ORDER["Unknown"])
                old_order = FRESHNESS_ORDER.get(old_freshness, FRESHNESS_ORDER["Unknown"])
                direction = "deterioration" if new_order > old_order else "recovery"
                events.append(
                    _make_event(
                        event_type=f"data_freshness_{direction}",
                        old_value=old_freshness,
                        new_value=new_freshness,
                        reason=f"Data freshness {direction}: {old_freshness} to {new_freshness}.",
                        evidence=evidence,
                        **common,
                    )
                )
    return events


def _insert_events(conn: object, events: list[dict[str, object]]) -> int:
    inserted = 0
    for event in events:
        exists = conn.execute(
            "SELECT 1 FROM signal_events WHERE idempotency_key = ? LIMIT 1",
            [event["idempotency_key"]],
        ).fetchone()
        if exists:
            continue
        columns = ["idempotency_key", *EVENT_COLUMNS]
        placeholders = ", ".join("?" for _ in columns)
        conn.execute(
            f"INSERT INTO signal_events ({', '.join(columns)}) VALUES ({placeholders})",
            [event[column] for column in columns],
        )
        inserted += 1
    return inserted


def capture_analysis_snapshot(
    signals: pd.DataFrame,
    data_health: pd.DataFrame,
    benchmark: str,
    active_symbols: list[str] | tuple[str, ...] | set[str],
    signal_config: object | None = None,
    stock_rrg: pd.DataFrame | None = None,
    seed_rrg: pd.DataFrame | None = None,
    refresh_succeeded: bool = True,
    db_path: Path = DEFAULT_DB_PATH,
) -> SnapshotCaptureResult:
    """Persist a refresh snapshot and derive idempotent, comparable events."""
    if not refresh_succeeded:
        return SnapshotCaptureResult(False, None, 0, False, "Refresh did not complete successfully.")
    if signals.empty:
        return SnapshotCaptureResult(False, None, 0, False, "No signals were available after refresh.")

    benchmark = str(benchmark).strip().upper()
    universe_hash = active_universe_hash(active_symbols)
    config_hash, config_json = canonical_signal_configuration(signal_config)
    run_id = uuid.uuid4().hex
    created_at = _utc_now()
    rows = _snapshot_rows(run_id, signals, data_health, stock_rrg, seed_rrg)
    effective_dates = [pd.Timestamp(row["effective_market_date"]) for row in rows if row["effective_market_date"] is not None and not pd.isna(row["effective_market_date"])]
    effective_date = max(effective_dates).date() if effective_dates else None
    benchmark_rows = signals[signals["symbol"].astype(str) == benchmark]
    benchmark_regime = str(benchmark_rows.iloc[0].get("larsson_regime", "n/a")) if not benchmark_rows.empty else "n/a"
    approximate = bool(data_health["calendar_approximate"].any()) if "calendar_approximate" in data_health.columns and not data_health.empty else True

    init_database(db_path)
    with _connect(db_path) as conn:
        previous_run = conn.execute(
            """
            SELECT run_id
            FROM analysis_runs
            WHERE benchmark = ? AND universe_hash = ? AND signal_config_hash = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [benchmark, universe_hash, config_hash],
        ).fetchone()
        previous = pd.DataFrame()
        if previous_run:
            previous = conn.execute(
                """
                SELECT run_id, subject_type, subject, effective_market_date, larsson_regime,
                       rrg_quadrant, freshness_status, score, pass_200d, pass_200w,
                       reason_codes, data_warnings, evidence_json
                FROM signal_snapshots
                WHERE run_id = ?
                """,
                [previous_run[0]],
            ).fetchdf()

        conn.execute(
            """
            INSERT INTO analysis_runs (
                run_id, created_at, effective_market_date, benchmark, universe_hash,
                signal_config_hash, signal_config_json, benchmark_regime,
                calendar_approximate, refresh_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [run_id, created_at, effective_date, benchmark, universe_hash, config_hash, config_json, benchmark_regime, approximate, "success"],
        )
        if rows:
            snapshot_frame = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
            conn.register("_signal_snapshot_rows", snapshot_frame)
            conn.execute(
                f"INSERT INTO signal_snapshots ({', '.join(SNAPSHOT_COLUMNS)}) SELECT {', '.join(SNAPSHOT_COLUMNS)} FROM _signal_snapshot_rows"
            )
            conn.unregister("_signal_snapshot_rows")

        events = _derive_events(
            rows,
            previous,
            benchmark=benchmark,
            universe_hash=universe_hash,
            signal_config_hash=config_hash,
            run_id=run_id,
        )
        inserted = _insert_events(conn, events)
    return SnapshotCaptureResult(True, run_id, inserted, previous.empty)


def load_signal_events(
    db_path: Path = DEFAULT_DB_PATH,
    event_types: list[str] | tuple[str, ...] | None = None,
    symbol: str | None = None,
    since: object | None = None,
    run_id: str | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Load normalized events using optional Activity-view filters."""
    init_database(db_path)
    filters: list[str] = []
    params: list[object] = []
    if event_types:
        placeholders = ", ".join("?" for _ in event_types)
        filters.append(f"event_type IN ({placeholders})")
        params.extend(event_types)
    if symbol:
        filters.append("subject = ?")
        params.append(str(symbol).strip().upper())
    if since is not None:
        filters.append("effective_market_date >= ?")
        params.append(pd.Timestamp(since).date())
    if run_id:
        filters.append("run_id = ?")
        params.append(run_id)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(max(int(limit), 1))
    with _connect(db_path) as conn:
        return conn.execute(
            f"""
            SELECT {', '.join(EVENT_COLUMNS)}
            FROM signal_events
            {where}
            ORDER BY effective_market_date DESC, created_at DESC, event_id
            LIMIT ?
            """,
            params,
        ).fetchdf()


def load_latest_analysis_snapshot(db_path: Path = DEFAULT_DB_PATH) -> tuple[pd.DataFrame, dict[str, object]]:
    init_database(db_path)
    with _connect(db_path) as conn:
        run = conn.execute(
            """
            SELECT run_id, created_at, effective_market_date, benchmark, universe_hash,
                   signal_config_hash, signal_config_json, benchmark_regime,
                   calendar_approximate, refresh_status
            FROM analysis_runs
            WHERE refresh_status = 'success'
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchdf()
        if run.empty:
            return pd.DataFrame(columns=SNAPSHOT_COLUMNS), {}
        run_id = str(run.iloc[0]["run_id"])
        snapshots = conn.execute(
            f"SELECT {', '.join(SNAPSHOT_COLUMNS)} FROM signal_snapshots WHERE run_id = ? ORDER BY subject_type, subject",
            [run_id],
        ).fetchdf()
    return snapshots, run.iloc[0].to_dict()


def record_paper_allocation_events(
    snapshot_id: int,
    changes: pd.DataFrame,
    metadata: dict[str, object],
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Normalize saved model-portfolio deltas as paper allocation events."""
    if changes.empty:
        return 0
    benchmark = str(metadata.get("benchmark", "")).strip().upper()
    config_hash = str(metadata.get("config_hash", ""))
    created_at = pd.Timestamp(metadata.get("created_at", _utc_now())).to_pydatetime().replace(tzinfo=None)
    effective_date = pd.Timestamp(metadata.get("effective_market_date", created_at)).date()
    universe_hash = _hash_payload(str(metadata.get("selected_seeds", "")), 16)
    events: list[dict[str, object]] = []
    for row in changes.sort_values("symbol").itertuples(index=False):
        action = str(getattr(row, "action", "")).lower()
        if action not in {"added", "removed", "increased", "reduced"}:
            continue
        old_value = round(float(getattr(row, "previous_weight", 0.0)), 8)
        new_value = round(float(getattr(row, "target_weight", 0.0)), 8)
        evidence = {
            "portfolio_snapshot_id": snapshot_id,
            "previous_weight": old_value,
            "target_weight": new_value,
            "delta": round(float(getattr(row, "delta", new_value - old_value)), 8),
            "action": action,
            "reason_codes": str(getattr(row, "reason_codes", "")),
            "paper_only": True,
        }
        events.append(
            _make_event(
                event_type=f"paper_allocation_{action}",
                subject_type="paper_allocation",
                subject=str(row.symbol),
                old_value=old_value,
                new_value=new_value,
                effective_market_date=effective_date,
                benchmark=benchmark,
                universe_hash=universe_hash,
                signal_config_hash=config_hash,
                run_id=None,
                reason=f"Paper allocation {action}; no execution occurred.",
                evidence=evidence,
                created_at=created_at,
            )
        )
    init_database(db_path)
    with _connect(db_path) as conn:
        return _insert_events(conn, events)
