"""Prepare an auditable weekly research packet from a candidate CSV.

Example:
    python -m stockpicker.weekly --candidates data/candidates.csv --refresh-research
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import pandas as pd

from .data import DEFAULT_DB_PATH, save_lab_research_details, save_lab_research_snapshot
from .lab import LabConfig, research_packet
from .research import load_sp500_security_master
from .data import fetch_price_history
from .scoring import rank_universe
from .signals import compute_signal_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_local_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"--output must stay inside the project directory: {PROJECT_ROOT}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a point-in-time Chatgptfolio weekly research packet.")
    parser.add_argument("--candidates", type=Path, help="CSV with at least symbol and technical score columns.")
    parser.add_argument("--sp500-master", action="store_true", help="Fetch and freeze the current public S&P 500 constituent master; use it as the candidate universe.")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="ISO date used to bound evidence.")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--refresh-research", action="store_true", help="Fetch bounded public SEC, FRED, and news evidence.")
    parser.add_argument("--refresh-prices", action="store_true", help="Refresh price history instead of using the local cache.")
    parser.add_argument("--years", type=int, default=5, help="Price-history years used for S&P 500 technical screening.")
    parser.add_argument("--output", type=Path, help="Optional JSON copy of the immutable packet.")
    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of)
    if not args.candidates and not args.sp500_master:
        parser.error("provide --candidates or --sp500-master")
    if args.candidates and args.sp500_master:
        parser.error("use either --candidates or --sp500-master, not both")
    try:
        output_path = _project_local_output(args.output) if args.output else None
    except ValueError as exc:
        parser.error(str(exc))
    master = load_sp500_security_master() if args.sp500_master else None
    candidates = master.copy() if master is not None else pd.read_csv(args.candidates)
    if master is not None:
        metadata = master[["symbol", "name", "sector"]].copy()
        metadata["theme"] = metadata["sector"]
        metadata["fundamental_score"] = 0.0
        metadata["catalyst_score"] = 0.0
        prices = fetch_price_history(
            metadata["symbol"].tolist() + [args.benchmark],
            refresh=args.refresh_prices,
            years=args.years,
            download_missing=args.refresh_prices,
        )
        signals = compute_signal_table(prices, args.benchmark)
        candidates = rank_universe(signals.merge(metadata, on="symbol", how="left"))
    if "score" not in candidates:
        candidates["score"] = 0.0
    if "symbol" not in candidates:
        parser.error("candidate CSV must include a symbol column")
    packet = research_packet(candidates, as_of, args.benchmark, LabConfig(), refresh_research=args.refresh_research, security_master=master)
    packet_hash = save_lab_research_snapshot(packet, DEFAULT_DB_PATH)
    save_lab_research_details(packet_hash, packet, DEFAULT_DB_PATH)
    if output_path:
        output_path.write_text(json.dumps(packet, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"packet_hash": packet_hash, "candidates": len(packet["candidates"]), "coverage": packet["coverage"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
