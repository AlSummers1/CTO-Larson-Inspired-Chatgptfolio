from __future__ import annotations

from pathlib import Path

import pandas as pd


UNIVERSE_COLUMNS = [
    "symbol",
    "name",
    "theme",
    "fundamental_score",
    "catalyst_score",
    "notes",
]


def normalize_symbol(symbol: str) -> str:
    """Normalize tickers for yfinance and table joins."""
    return symbol.strip().upper()


def parse_ticker_text(text: str) -> list[str]:
    """Parse tickers separated by commas, spaces, or newlines."""
    cleaned = text.replace(",", " ").replace("\n", " ")
    symbols = [normalize_symbol(part) for part in cleaned.split() if part.strip()]
    return list(dict.fromkeys(symbols))


def load_universe(path: Path) -> pd.DataFrame:
    """Load a seed universe and ensure optional v1 scoring columns exist."""
    frame = pd.read_csv(path)
    for column in UNIVERSE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0 if column.endswith("_score") else ""

    frame = frame[UNIVERSE_COLUMNS].copy()
    frame = frame[frame["symbol"].notna()].copy()
    frame["symbol"] = frame["symbol"].astype(str).map(normalize_symbol)
    frame = frame[frame["symbol"] != ""]
    frame["fundamental_score"] = pd.to_numeric(frame["fundamental_score"], errors="coerce").fillna(0.0)
    frame["catalyst_score"] = pd.to_numeric(frame["catalyst_score"], errors="coerce").fillna(0.0)
    frame[["name", "theme", "notes"]] = frame[["name", "theme", "notes"]].fillna("")
    return frame.drop_duplicates("symbol", keep="first").reset_index(drop=True)


def merge_manual_tickers(seed: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Merge manually entered tickers into the seed universe."""
    seed_symbols = set(seed["symbol"])
    extra_rows = [
        {
            "symbol": symbol,
            "name": "",
            "theme": "Manual",
            "fundamental_score": 0.0,
            "catalyst_score": 0.0,
            "notes": "",
        }
        for symbol in tickers
        if symbol not in seed_symbols
    ]
    if not extra_rows:
        return seed.copy()
    return pd.concat([seed, pd.DataFrame(extra_rows)], ignore_index=True)
