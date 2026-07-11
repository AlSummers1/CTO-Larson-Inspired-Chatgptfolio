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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a point-in-time Chatgptfolio weekly research packet.")
    parser.add_argument("--candidates", required=True, type=Path, help="CSV with at least symbol and technical score columns.")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="ISO date used to bound evidence.")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--refresh-research", action="store_true", help="Fetch bounded public SEC, FRED, and news evidence.")
    parser.add_argument("--output", type=Path, help="Optional JSON copy of the immutable packet.")
    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of)
    candidates = pd.read_csv(args.candidates)
    if "symbol" not in candidates:
        parser.error("candidate CSV must include a symbol column")
    packet = research_packet(candidates, as_of, args.benchmark, LabConfig(), refresh_research=args.refresh_research)
    packet_hash = save_lab_research_snapshot(packet, DEFAULT_DB_PATH)
    save_lab_research_details(packet_hash, packet, DEFAULT_DB_PATH)
    if args.output:
        args.output.write_text(json.dumps(packet, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"packet_hash": packet_hash, "candidates": len(packet["candidates"]), "coverage": packet["coverage"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
