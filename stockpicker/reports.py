from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import getpass
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from .activity import load_latest_analysis_snapshot, load_signal_events
from .data import (
    DEFAULT_DB_PATH,
    _connect,
    init_database,
    load_latest_portfolio_snapshot,
    load_portfolio_rebalance_log,
)
from .data_quality import DERIVED_200W_NOTE


REPORT_SECTIONS = [
    "Data Health & Cautions",
    "Benchmark Regime & Material Changes",
    "Strongest Candidates",
    "Deteriorating Holdings or Candidates",
    "Regime & RRG Movers",
    "Paper Allocation Changes",
    "Evidence",
]
AI_SECTIONS = ["summary", "opportunities", "risks", "watch_items"]
RRG_DISTRIBUTION_ORDER = ["Leading", "Weakening", "Lagging", "Improving", "Cash / Unclassified"]
SAVED_BRIEF_SNAPSHOT_VERSION = 1
SAVED_BRIEF_TITLE_MAX_LENGTH = 120
SAVED_REPORT_FIELDS = [
    "report_id",
    "created_at",
    "report_version",
    "analysis_run_id",
    "benchmark",
    "universe_hash",
    "evidence_hash",
    "markdown",
    "evidence_json",
    "commentary",
    "commentary_json",
    "commentary_created_at",
    "ai_model",
    "prompt_hash",
    "generation_status",
    "usage_json",
    "diagnostic",
]


@dataclass(frozen=True)
class WeeklyReportResult:
    report_id: int
    version: int
    markdown: str
    evidence_payload: dict[str, object]
    evidence_hash: str
    created_new: bool
    analysis_run_id: str
    benchmark: str


@dataclass(frozen=True)
class AICommentaryResult:
    status: str
    commentary: str = ""
    sections: dict[str, list[dict[str, object]]] | None = None
    model: str = ""
    prompt_hash: str = ""
    usage: dict[str, object] | None = None
    diagnostic: str = ""


@dataclass(frozen=True)
class SavedWeeklyBriefResult:
    saved_brief_id: int
    title: str
    created_at: datetime
    source_report_id: int


@dataclass(frozen=True)
class BriefItem:
    subject: str
    label: str
    detail: str
    evidence_id: str


@dataclass(frozen=True)
class ReportBrief:
    benchmark: str
    effective_market_date: str
    benchmark_regime: str
    fresh_count: int
    stock_count: int
    freshness_pct: float
    material_change_count: int
    allocation_change_count: int
    evidence_count: int
    leaders: tuple[BriefItem, ...]
    leader_overflow: int
    risks: tuple[BriefItem, ...]
    risk_overflow: int
    changes: tuple[BriefItem, ...]
    change_overflow: int
    allocation_changes: tuple[BriefItem, ...]
    allocation_overflow: int
    has_paper_allocation: bool


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


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _evidence_id(prefix: str, payload: object) -> str:
    return f"{prefix}-{_stable_hash(payload)[:12].upper()}"


def _parse_evidence(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_commentary_sections(value: object) -> dict[str, list[dict[str, object]]]:
    """Return validated stored commentary sections, or an empty mapping for legacy rows."""
    if isinstance(value, dict):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value or "{}"))
        except (TypeError, ValueError):
            return {}
    if not isinstance(parsed, dict):
        return {}
    output: dict[str, list[dict[str, object]]] = {}
    for section in AI_SECTIONS:
        items = parsed.get(section)
        if not isinstance(items, list):
            return {}
        cleaned: list[dict[str, object]] = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                return {}
            evidence_ids = item.get("evidence_ids")
            if not isinstance(evidence_ids, list):
                return {}
            cleaned.append({"text": item["text"], "evidence_ids": [str(value) for value in evidence_ids]})
        output[section] = cleaned
    return output


def _brief_event_label(event_type: str) -> str:
    labels = {
        "benchmark_regime_change": "Benchmark regime",
        "larsson_regime_change": "Regime",
        "stock_rrg_quadrant_change": "RRG quadrant",
        "seed_rrg_quadrant_change": "RRG quadrant",
        "freshness_status_change": "Data freshness",
        "paper_allocation_added": "Added to paper portfolio",
        "paper_allocation_removed": "Removed from paper portfolio",
        "paper_allocation_changed": "Paper weight changed",
    }
    return labels.get(event_type, event_type.replace("_", " ").title())


def build_report_brief(evidence_payload: dict[str, object], *, item_limit: int = 5) -> ReportBrief:
    """Create a compact, deterministic presentation model from an auditable report payload."""
    limit = max(1, int(item_limit))
    benchmark = str(evidence_payload.get("benchmark") or "—")
    effective_date = str(evidence_payload.get("effective_market_date") or "unknown")[:10]
    evidence = [item for item in evidence_payload.get("evidence", []) if isinstance(item, dict)]
    snapshots = [item for item in evidence if item.get("kind") == "signal_snapshot" and item.get("subject_type") == "stock"]
    events = [item for item in evidence if item.get("kind") == "change_event"]
    allocations = [item for item in evidence if item.get("kind") == "paper_allocation"]
    candidates = [item for item in snapshots if str(item.get("subject")) != benchmark]
    benchmark_row = next((item for item in snapshots if str(item.get("subject")) == benchmark), None)

    fresh_count = sum(str(item.get("freshness_status")) == "Fresh" for item in snapshots)
    stock_count = len(snapshots)
    freshness_pct = (fresh_count / stock_count * 100.0) if stock_count else 0.0

    eligible_leaders = [
        item for item in candidates
        if item.get("freshness_status") == "Fresh"
        and item.get("larsson_regime") == "Bull"
        and item.get("pass_200d") is True
    ]
    eligible_leaders.sort(key=lambda item: (-(float(item.get("score") or -1)), str(item.get("subject"))))
    leaders = tuple(
        BriefItem(
            subject=str(item.get("subject") or "—"),
            label=f"Score {_fmt_score(item.get('score'))}",
            detail=" · ".join(
                value for value in [
                    str(item.get("larsson_regime") or "No regime"),
                    str(item.get("rrg_quadrant") or "No RRG classification"),
                    "above 200D",
                ] if value
            ),
            evidence_id=str(item.get("evidence_id") or ""),
        )
        for item in eligible_leaders[:limit]
    )
    held_symbols = {str(item.get("symbol")) for item in allocations}
    risk_candidates: list[tuple[int, str, BriefItem]] = []
    for item in candidates:
        subject = str(item.get("subject") or "—")
        freshness = str(item.get("freshness_status") or "Unknown")
        regime = str(item.get("larsson_regime") or "")
        quadrant = str(item.get("rrg_quadrant") or "")
        warning = str(item.get("data_warnings") or "").strip()
        severity = 0
        reasons: list[str] = []
        if freshness in {"Stale", "Unknown"}:
            severity = max(severity, 5)
            reasons.append(freshness.lower() + " data")
        elif freshness == "Delayed":
            severity = max(severity, 4)
            reasons.append("delayed data")
        if regime == "Bear":
            severity = max(severity, 4)
            reasons.append("Bear regime")
        elif regime == "Transition":
            severity = max(severity, 3)
            reasons.append("Transition regime")
        if warning:
            severity = max(severity, 3)
            reasons.append(warning)
        if subject in held_symbols and quadrant in {"Weakening", "Lagging"}:
            severity = max(severity, 2)
            reasons.append(f"{quadrant} paper holding")
        if severity:
            risk_candidates.append(
                (
                    severity,
                    subject,
                    BriefItem(
                        subject=subject,
                        label="Review",
                        detail=" · ".join(dict.fromkeys(reasons)),
                        evidence_id=str(item.get("evidence_id") or ""),
                    ),
                )
            )
    risk_candidates.sort(key=lambda value: (-value[0], value[1]))
    risks = tuple(value[2] for value in risk_candidates[:limit])

    events.sort(
        key=lambda item: (
            str(item.get("effective_market_date") or ""),
            str(item.get("event_type") or ""),
            str(item.get("subject") or ""),
        ),
        reverse=True,
    )

    def event_item(item: dict[str, object]) -> BriefItem:
        old_value = str(item.get("old_value") or "—")
        new_value = str(item.get("new_value") or "—")
        return BriefItem(
            subject=str(item.get("subject") or "—"),
            label=_brief_event_label(str(item.get("event_type") or "")),
            detail=str(item.get("reason") or f"{old_value} → {new_value}"),
            evidence_id=str(item.get("evidence_id") or ""),
        )

    change_items = tuple(event_item(item) for item in events[:limit])
    allocation_events = [item for item in events if str(item.get("event_type") or "").startswith("paper_allocation_")]
    allocation_items = tuple(event_item(item) for item in allocation_events[:limit])
    return ReportBrief(
        benchmark=benchmark,
        effective_market_date=effective_date,
        benchmark_regime=str(benchmark_row.get("larsson_regime") or "Unknown") if benchmark_row else "Unknown",
        fresh_count=fresh_count,
        stock_count=stock_count,
        freshness_pct=freshness_pct,
        material_change_count=len(events),
        allocation_change_count=len(allocation_events),
        evidence_count=len(evidence),
        leaders=leaders,
        leader_overflow=max(0, len(eligible_leaders) - limit),
        risks=risks,
        risk_overflow=max(0, len(risk_candidates) - limit),
        changes=change_items,
        change_overflow=max(0, len(events) - limit),
        allocation_changes=allocation_items,
        allocation_overflow=max(0, len(allocation_events) - limit),
        has_paper_allocation=bool(allocations),
    )


def build_paper_rrg_distribution(evidence_payload: dict[str, object]) -> pd.DataFrame:
    """Aggregate captured paper-allocation weights by RRG quadrant."""
    grouped: dict[str, dict[str, object]] = {
        quadrant: {"weight": 0.0, "holdings": []}
        for quadrant in RRG_DISTRIBUTION_ORDER
    }
    for item in evidence_payload.get("evidence", []):
        if not isinstance(item, dict) or item.get("kind") != "paper_allocation":
            continue
        try:
            weight = float(item.get("weight") or 0.0)
        except (TypeError, ValueError):
            continue
        if pd.isna(weight) or weight <= 0:
            continue
        symbol = str(item.get("symbol") or "Unknown")
        quadrant = str(item.get("rrg_quadrant") or "")
        if symbol.casefold() == "cash" or quadrant not in RRG_DISTRIBUTION_ORDER[:-1]:
            quadrant = "Cash / Unclassified"
        grouped[quadrant]["weight"] = float(grouped[quadrant]["weight"]) + weight
        grouped[quadrant]["holdings"].append((symbol, weight))

    rows: list[dict[str, object]] = []
    for quadrant in RRG_DISTRIBUTION_ORDER:
        holdings = sorted(grouped[quadrant]["holdings"], key=lambda value: (-value[1], value[0]))
        if not holdings:
            continue
        rows.append(
            {
                "quadrant": quadrant,
                "weight": float(grouped[quadrant]["weight"]),
                "holding_count": len(holdings),
                "holdings": ", ".join(f"{symbol} {weight:.1%}" for symbol, weight in holdings),
            }
        )
    return pd.DataFrame(rows, columns=["quadrant", "weight", "holding_count", "holdings"])


def build_paper_holding_allocation(evidence_payload: dict[str, object]) -> pd.DataFrame:
    """Return captured paper weights by holding without consulting the live portfolio."""
    weights: dict[str, float] = {}
    for item in evidence_payload.get("evidence", []):
        if not isinstance(item, dict) or item.get("kind") != "paper_allocation":
            continue
        try:
            weight = float(item.get("weight") or 0.0)
        except (TypeError, ValueError):
            continue
        if pd.isna(weight) or weight <= 0:
            continue
        symbol = str(item.get("symbol") or "Unknown").strip() or "Unknown"
        weights[symbol] = weights.get(symbol, 0.0) + weight
    rows = [
        {"symbol": symbol, "weight": weight}
        for symbol, weight in sorted(weights.items(), key=lambda value: (-value[1], value[0]))
    ]
    return pd.DataFrame(rows, columns=["symbol", "weight"])


def _clean_scalar(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _snapshot_evidence(snapshots: pd.DataFrame, benchmark: str) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for row in snapshots.sort_values(["subject_type", "subject"]).itertuples(index=False):
        details = _parse_evidence(row.evidence_json)
        payload = {
            "kind": "signal_snapshot",
            "subject_type": str(row.subject_type),
            "subject": str(row.subject),
            "effective_market_date": _clean_scalar(row.effective_market_date),
            "larsson_regime": str(row.larsson_regime or ""),
            "rrg_quadrant": str(row.rrg_quadrant or ""),
            "freshness_status": str(row.freshness_status or ""),
            "score": _clean_scalar(row.score),
            "pass_200d": _clean_scalar(row.pass_200d),
            "pass_200w": _clean_scalar(row.pass_200w),
            "reason_codes": str(row.reason_codes or ""),
            "data_warnings": str(row.data_warnings or ""),
            "details": details,
        }
        prefix = "BM" if str(row.subject) == benchmark else ("SE" if str(row.subject_type) == "seed_universe" else "SG")
        evidence.append({"evidence_id": _evidence_id(prefix, payload), **payload})
    return evidence


def _event_evidence(events: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    if events.empty:
        return output
    for row in events.sort_values(["effective_market_date", "event_type", "subject", "event_id"]).itertuples(index=False):
        output.append(
            {
                "evidence_id": str(row.evidence_id),
                "kind": "change_event",
                "effective_market_date": _clean_scalar(row.effective_market_date),
                "event_type": str(row.event_type),
                "subject": str(row.subject),
                "old_value": str(row.old_value),
                "new_value": str(row.new_value),
                "reason": str(row.reason),
                "details": _parse_evidence(row.evidence_json),
            }
        )
    return output


def _portfolio_evidence(holdings: pd.DataFrame, metadata: dict[str, object]) -> list[dict[str, object]]:
    if holdings.empty:
        return []
    output: list[dict[str, object]] = []
    for row in holdings.sort_values(["weight", "symbol"], ascending=[False, True]).to_dict("records"):
        payload = {
            "kind": "paper_allocation",
            "portfolio_snapshot_id": _clean_scalar(metadata.get("snapshot_id")),
            "symbol": str(row.get("symbol", "")),
            "weight": round(float(row.get("weight", 0.0) or 0.0), 8),
            "allocator_score": _clean_scalar(row.get("allocator_score")),
            "larsson_regime": str(row.get("larsson_regime", "") or ""),
            "rrg_quadrant": str(row.get("rrg_quadrant", "") or ""),
            "reason_codes": str(row.get("reason_codes", "") or ""),
            "data_warnings": str(row.get("data_warnings", "") or ""),
        }
        output.append({"evidence_id": _evidence_id("PA", payload), **payload})
    return output


def _bullet(text: str, evidence_id: str) -> str:
    return f"- {text} [{evidence_id}]"


def _fmt_score(value: object) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def _render_report(payload: dict[str, object]) -> str:
    benchmark = str(payload["benchmark"])
    effective_date = str(payload.get("effective_market_date") or "unknown")[:10]
    evidence = list(payload.get("evidence", []))
    snapshots = [item for item in evidence if item.get("kind") == "signal_snapshot"]
    events = [item for item in evidence if item.get("kind") == "change_event"]
    allocations = [item for item in evidence if item.get("kind") == "paper_allocation"]
    stocks = [item for item in snapshots if item.get("subject_type") == "stock"]
    benchmark_rows = [item for item in stocks if item.get("subject") == benchmark]
    candidates = [item for item in stocks if item.get("subject") != benchmark]

    counts = {status: sum(item.get("freshness_status") == status for item in stocks) for status in ["Fresh", "Delayed", "Stale", "Unknown"]}
    lines = [
        f"# Weekly Technical Analysis Report — {effective_date}",
        "",
        f"Benchmark: **{benchmark}** · Evidence is from the latest explicit local refresh.",
        "",
        "## Data Health & Cautions",
        "",
        f"- Fresh: {counts['Fresh']} · Delayed: {counts['Delayed']} · Stale: {counts['Stale']} · Unknown: {counts['Unknown']}.",
    ]
    cautions = [item for item in stocks if item.get("freshness_status") in {"Delayed", "Stale", "Unknown"} or item.get("data_warnings")]
    if cautions:
        for item in sorted(cautions, key=lambda value: (str(value.get("freshness_status")), str(value.get("subject")))):
            detail = item.get("details", {})
            warning = str(item.get("data_warnings") or "no additional signal warning")
            lines.append(
                _bullet(
                    f"{item['subject']}: {item.get('freshness_status')}; {detail.get('trading_sessions_behind', 'n/a')} sessions behind; {warning}.",
                    str(item["evidence_id"]),
                )
            )
    else:
        lines.append("- No delayed or stale symbols in the captured universe.")
    if payload.get("calendar_approximate"):
        lines.append("- Benchmark calendar unavailable; calendar-dependent coverage is approximate.")
    lines.extend(["", f"> {DERIVED_200W_NOTE}", "", "## Benchmark Regime & Material Changes", ""])
    if benchmark_rows:
        item = benchmark_rows[0]
        lines.append(_bullet(f"{benchmark} Larsson-proxy regime: **{item.get('larsson_regime') or 'n/a'}**.", str(item["evidence_id"])))
    benchmark_events = [item for item in events if item.get("event_type") == "benchmark_regime_change"]
    for item in benchmark_events:
        lines.append(_bullet(str(item["reason"]), str(item["evidence_id"])))
    if not benchmark_events:
        lines.append("- No benchmark regime change was recorded in the latest evidence window.")

    lines.extend(["", "## Strongest Candidates", ""])
    strongest = sorted(candidates, key=lambda item: (-(float(item.get("score") or -1)), str(item.get("subject"))))[:5]
    if strongest:
        for item in strongest:
            passes = []
            if item.get("pass_200d") is True:
                passes.append("passed 200D")
            if item.get("pass_200w") is True:
                passes.append("passed 200W")
            reason = ", ".join(passes) if passes else "review technical filters"
            lines.append(_bullet(f"{item['subject']}: score {_fmt_score(item.get('score'))}; {item.get('larsson_regime') or 'n/a'} regime; {reason}.", str(item["evidence_id"])))
    else:
        lines.append("- No candidates were present in the latest snapshot.")

    lines.extend(["", "## Deteriorating Holdings or Candidates", ""])
    deteriorating = [
        item for item in candidates
        if item.get("freshness_status") in {"Delayed", "Stale", "Unknown"}
        or item.get("larsson_regime") == "Bear"
        or item.get("rrg_quadrant") in {"Weakening", "Lagging"}
    ]
    if deteriorating:
        for item in sorted(deteriorating, key=lambda value: str(value.get("subject"))):
            labels = [str(item.get("freshness_status") or "Unknown"), str(item.get("larsson_regime") or "n/a")]
            if item.get("rrg_quadrant"):
                labels.append(str(item["rrg_quadrant"]))
            lines.append(_bullet(f"{item['subject']}: {' · '.join(labels)}.", str(item["evidence_id"])))
    else:
        lines.append("- No captured candidate met the deterministic deterioration criteria.")

    lines.extend(["", "## Regime & RRG Movers", ""])
    movers = [item for item in events if item.get("event_type") in {"larsson_regime_change", "benchmark_regime_change", "stock_rrg_quadrant_change", "seed_rrg_quadrant_change"}]
    if movers:
        for item in movers:
            lines.append(_bullet(str(item["reason"]), str(item["evidence_id"])))
    else:
        lines.append("- No regime or RRG changes were recorded in the evidence window.")

    lines.extend(["", "## Paper Allocation Changes", ""])
    allocation_events = [item for item in events if str(item.get("event_type", "")).startswith("paper_allocation_")]
    if allocation_events:
        for item in allocation_events:
            lines.append(_bullet(str(item["reason"]), str(item["evidence_id"])))
    elif allocations:
        lines.append("- A saved paper portfolio exists, with no allocation change recorded in the evidence window.")
    else:
        lines.append("- No saved paper allocation is available.")

    lines.extend(["", "## Evidence", "", "| Evidence ID | Date | Type | Subject | Value |", "|---|---|---|---|---|"])
    for item in sorted(evidence, key=lambda value: str(value["evidence_id"])):
        subject = str(item.get("subject", item.get("symbol", "")))
        date = str(item.get("effective_market_date", effective_date) or effective_date)[:10]
        kind = str(item.get("event_type", item.get("kind", "")))
        value = str(item.get("new_value", item.get("larsson_regime", item.get("weight", ""))))
        value = value.replace("|", "/")
        lines.append(f"| {item['evidence_id']} | {date} | {kind} | {subject} | {value} |")

    lines.extend(
        [
            "",
            "---",
            "This report is informational technical analysis of local data. It does not generate orders, connect to a broker, or constitute investment advice.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_weekly_report(db_path: Path = DEFAULT_DB_PATH) -> WeeklyReportResult:
    """Build and idempotently persist a deterministic report from local evidence."""
    snapshots, run = load_latest_analysis_snapshot(db_path)
    if not run:
        raise RuntimeError("Refresh data successfully before building a weekly report.")
    benchmark = str(run["benchmark"])
    effective = pd.Timestamp(run.get("effective_market_date"))
    since = effective - timedelta(days=7) if not pd.isna(effective) else None
    events = load_signal_events(db_path, since=since, limit=1000)
    holdings, portfolio_metadata = load_latest_portfolio_snapshot(db_path)
    evidence = [
        *_snapshot_evidence(snapshots, benchmark),
        *_event_evidence(events),
        *_portfolio_evidence(holdings, portfolio_metadata),
    ]
    evidence = sorted(evidence, key=lambda item: str(item["evidence_id"]))
    payload: dict[str, object] = {
        "benchmark": benchmark,
        "effective_market_date": _clean_scalar(run.get("effective_market_date")),
        "universe_hash": str(run.get("universe_hash", "")),
        "signal_config_hash": str(run.get("signal_config_hash", "")),
        "calendar_approximate": bool(run.get("calendar_approximate", False)),
        "evidence": evidence,
    }
    evidence_hash = _stable_hash(payload)
    markdown = _render_report(payload)

    init_database(db_path)
    with _connect(db_path) as conn:
        existing = conn.execute(
            """
            SELECT report_id, report_version, markdown, evidence_json
            FROM weekly_reports WHERE evidence_hash = ? LIMIT 1
            """,
            [evidence_hash],
        ).fetchone()
        if existing:
            return WeeklyReportResult(
                int(existing[0]), int(existing[1]), str(existing[2]), json.loads(existing[3]),
                evidence_hash, False, str(run["run_id"]), benchmark,
            )
        version = int(
            conn.execute(
                "SELECT COALESCE(MAX(report_version), 0) + 1 FROM weekly_reports WHERE benchmark = ?",
                [benchmark],
            ).fetchone()[0]
        )
        report_id = int(
            conn.execute(
                """
                INSERT INTO weekly_reports (
                    created_at, report_version, analysis_run_id, benchmark, universe_hash,
                    evidence_hash, markdown, evidence_json, commentary, commentary_json,
                    commentary_created_at, ai_model, prompt_hash, generation_status, usage_json, diagnostic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', '{}', NULL, '', '', 'not_requested', '{}', '')
                RETURNING report_id
                """,
                [_utc_now(), version, str(run["run_id"]), benchmark, str(run.get("universe_hash", "")), evidence_hash, markdown, _canonical_json(payload)],
            ).fetchone()[0]
        )
    return WeeklyReportResult(report_id, version, markdown, payload, evidence_hash, True, str(run["run_id"]), benchmark)


def load_weekly_reports(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    init_database(db_path)
    with _connect(db_path) as conn:
        return conn.execute(
            """
            SELECT report_id, created_at, report_version, analysis_run_id, benchmark,
                   universe_hash, evidence_hash, markdown, evidence_json, commentary,
                   commentary_json, commentary_created_at, ai_model, prompt_hash,
                   generation_status, usage_json, diagnostic
            FROM weekly_reports
            ORDER BY created_at DESC, report_id DESC
            """
        ).fetchdf()


def load_weekly_report(report_id: int, db_path: Path = DEFAULT_DB_PATH) -> dict[str, object]:
    reports = load_weekly_reports(db_path)
    selected = reports[reports["report_id"] == int(report_id)]
    return selected.iloc[0].to_dict() if not selected.empty else {}


def save_weekly_brief(
    title: str,
    source_report_id: int,
    db_path: Path = DEFAULT_DB_PATH,
) -> SavedWeeklyBriefResult:
    """Persist an immutable presentation snapshot of a generated weekly report."""
    cleaned_title = str(title if title is not None else "").strip()
    if not cleaned_title:
        raise ValueError("Brief title is required.")
    if len(cleaned_title) > SAVED_BRIEF_TITLE_MAX_LENGTH:
        raise ValueError(f"Brief title must be {SAVED_BRIEF_TITLE_MAX_LENGTH} characters or fewer.")
    source_report = load_weekly_report(int(source_report_id), db_path)
    if not source_report:
        raise ValueError("The source weekly report no longer exists.")

    snapshot = {
        "schema_version": SAVED_BRIEF_SNAPSHOT_VERSION,
        "report": {field: source_report.get(field) for field in SAVED_REPORT_FIELDS},
    }
    created_at = _utc_now()
    with _connect(db_path) as conn:
        saved_brief_id = int(
            conn.execute(
                """
                INSERT INTO saved_weekly_briefs (
                    created_at, title, source_report_id, snapshot_version, snapshot_json
                ) VALUES (?, ?, ?, ?, ?)
                RETURNING saved_brief_id
                """,
                [
                    created_at,
                    cleaned_title,
                    int(source_report_id),
                    SAVED_BRIEF_SNAPSHOT_VERSION,
                    _canonical_json(snapshot),
                ],
            ).fetchone()[0]
        )
    return SavedWeeklyBriefResult(saved_brief_id, cleaned_title, created_at, int(source_report_id))


def load_saved_weekly_briefs(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """List named weekly briefs without expanding their snapshot payloads."""
    init_database(db_path)
    with _connect(db_path) as conn:
        return conn.execute(
            """
            SELECT saved_brief_id, created_at, title, source_report_id, snapshot_version
            FROM saved_weekly_briefs
            ORDER BY created_at DESC, saved_brief_id DESC
            """
        ).fetchdf()


def load_saved_weekly_brief(saved_brief_id: int, db_path: Path = DEFAULT_DB_PATH) -> dict[str, object]:
    """Load an immutable saved brief as the same presentation shape as a generated report."""
    init_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT created_at, title, source_report_id, snapshot_version, snapshot_json
            FROM saved_weekly_briefs
            WHERE saved_brief_id = ?
            LIMIT 1
            """,
            [int(saved_brief_id)],
        ).fetchone()
    if row is None:
        return {}
    try:
        snapshot = json.loads(str(row[4] or "{}"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Saved brief content is invalid and cannot be loaded.") from exc
    if (
        int(row[3]) != SAVED_BRIEF_SNAPSHOT_VERSION
        or not isinstance(snapshot, dict)
        or snapshot.get("schema_version") != SAVED_BRIEF_SNAPSHOT_VERSION
        or not isinstance(snapshot.get("report"), dict)
    ):
        raise ValueError("Saved brief content uses an unsupported or invalid format.")
    report = dict(snapshot["report"])
    report.update(
        {
            "saved_brief_id": int(saved_brief_id),
            "saved_title": str(row[1]),
            "saved_at": row[0],
            "source_report_id": int(row[2]),
            "is_saved_brief": True,
        }
    )
    return report


def _import_openai():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The OpenAI SDK is not installed. Install dependencies from requirements.txt.") from exc
    return OpenAI


def _commentary_schema() -> dict[str, object]:
    item = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        "required": ["text", "evidence_ids"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {section: {"type": "array", "items": item} for section in AI_SECTIONS},
        "required": AI_SECTIONS,
        "additionalProperties": False,
    }


def _usage_dict(usage: object) -> dict[str, object]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {"value": str(usage)}


def _validate_commentary(parsed: object, valid_ids: set[str]) -> tuple[str, str]:
    if not isinstance(parsed, dict) or set(parsed) != set(AI_SECTIONS):
        return "invalid_schema", "AI response did not contain the required sections."
    for section in AI_SECTIONS:
        items = parsed.get(section)
        if not isinstance(items, list):
            return "invalid_schema", f"AI section '{section}' was not a list."
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                return "invalid_schema", f"AI section '{section}' contained an invalid item."
            references = item.get("evidence_ids")
            if not isinstance(references, list) or not references:
                return "invalid_schema", "Every AI assertion must reference at least one evidence ID."
            unknown = sorted(set(map(str, references)) - valid_ids)
            if unknown:
                return "invalid_evidence", f"AI response referenced unknown evidence IDs: {', '.join(unknown)}."
    return "success", ""


def _render_commentary(parsed: dict[str, object]) -> str:
    labels = {
        "summary": "Summary",
        "opportunities": "Opportunities",
        "risks": "Risks",
        "watch_items": "Watch Items",
    }
    lines = ["## AI Commentary — Interpretation Only", ""]
    for section in AI_SECTIONS:
        lines.extend([f"### {labels[section]}", ""])
        items = parsed[section]
        if not items:
            lines.append("- No evidence-backed item identified.")
        for item in items:
            refs = ", ".join(item["evidence_ids"])
            lines.append(f"- {item['text']} [{refs}]")
        lines.append("")
    lines.append("AI commentary cannot change scores or paper allocations. Informational only; no trade or execution is implied.")
    return "\n".join(lines).strip() + "\n"


def openai_api_key() -> str | None:
    """Resolve an API key without persisting or logging it.

    Environment configuration has priority. On macOS, a local installation may
    instead keep the key in the current user's Keychain under
    ``chatgptfolio-openai``.
    """
    environment_key = os.getenv("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                getpass.getuser(),
                "-s",
                "chatgptfolio-openai",
                "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    keychain_key = result.stdout.strip()
    return keychain_key if result.returncode == 0 and keychain_key else None


def generate_ai_commentary(
    evidence_payload: dict[str, object],
    *,
    model: str | None = None,
    client: object | None = None,
    timeout_seconds: float = 30.0,
) -> AICommentaryResult:
    """Generate validated evidence-linked interpretation after an explicit call."""
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-terra")
    compact_payload = _canonical_json(evidence_payload)
    schema = _commentary_schema()
    instructions = (
        "Interpret the supplied technical-analysis evidence. Return only the requested structured sections. "
        "Every assertion must cite one or more supplied evidence IDs. Do not invent facts, change scores, "
        "suggest execution, or imply brokerage activity. Distinguish observations from interpretation."
    )
    prompt_hash = _stable_hash({"instructions": instructions, "schema": schema, "evidence": evidence_payload, "model": selected_model})
    api_key = openai_api_key() if client is None else None
    if client is None and not api_key:
        return AICommentaryResult(
            status="missing_key",
            model=selected_model,
            prompt_hash=prompt_hash,
            usage={},
            diagnostic=(
                "No OpenAI API key was found in the environment or macOS Keychain. "
                "The deterministic local report remains available."
            ),
        )
    valid_ids = {
        str(item.get("evidence_id"))
        for item in evidence_payload.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    try:
        if client is None:
            OpenAI = _import_openai()
            client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=1)
        response = client.responses.create(
            model=selected_model,
            reasoning={"effort": "low"},
            store=False,
            input=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": compact_payload},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "weekly_ta_commentary",
                    "strict": True,
                    "schema": schema,
                },
                "verbosity": "low",
            },
        )
        output_text = str(getattr(response, "output_text", "") or "").strip()
        if not output_text:
            return AICommentaryResult(
                status="refusal",
                model=selected_model,
                prompt_hash=prompt_hash,
                usage=_usage_dict(getattr(response, "usage", None)),
                diagnostic="The model returned no commentary, possibly because the request was refused.",
            )
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            return AICommentaryResult(
                status="invalid_schema",
                model=selected_model,
                prompt_hash=prompt_hash,
                usage=_usage_dict(getattr(response, "usage", None)),
                diagnostic="The AI response was not valid structured JSON.",
            )
        validation_status, diagnostic = _validate_commentary(parsed, valid_ids)
        if validation_status != "success":
            return AICommentaryResult(
                status=validation_status,
                model=selected_model,
                prompt_hash=prompt_hash,
                usage=_usage_dict(getattr(response, "usage", None)),
                diagnostic=diagnostic,
            )
        return AICommentaryResult(
            status="success",
            commentary=_render_commentary(parsed),
            sections=parse_commentary_sections(parsed),
            model=selected_model,
            prompt_hash=prompt_hash,
            usage=_usage_dict(getattr(response, "usage", None)),
        )
    except Exception as exc:  # SDK exception classes vary between compatible releases.
        is_timeout = isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()
        status = "timeout" if is_timeout else "api_error"
        diagnostic = "OpenAI request timed out." if is_timeout else f"OpenAI request failed: {type(exc).__name__}."
        return AICommentaryResult(status=status, model=selected_model, prompt_hash=prompt_hash, diagnostic=diagnostic)


def save_ai_commentary(report_id: int, result: AICommentaryResult, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Persist only the final commentary and compact generation metadata."""
    init_database(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE weekly_reports
            SET commentary = ?, commentary_json = ?, commentary_created_at = ?, ai_model = ?,
                prompt_hash = ?, generation_status = ?, usage_json = ?, diagnostic = ?
            WHERE report_id = ?
            """,
            [
                result.commentary,
                _canonical_json(result.sections or {}),
                _utc_now(),
                result.model,
                result.prompt_hash,
                result.status,
                _canonical_json(result.usage or {}),
                result.diagnostic,
                int(report_id),
            ],
        )
