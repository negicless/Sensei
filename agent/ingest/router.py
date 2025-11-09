# agent/ingest/router.py
from __future__ import annotations
import re
from agent.ingest.price_stooq import fetch_price_stooq as fetch_jp_daily
from agent.ingest.price_us import get_price_df_us

def is_us_ticker(code: str) -> bool:
    c = _normalize_ticker(code)
    # JP tickers in your bot are typically 4 digits (e.g., 7013)
    if re.fullmatch(r"\d{4}", c):
        return False
    # everything else (letters like FUBO, NVDA, BRK.B) treat as US
    return True

def fetch_history(code: str, tf: str):
    norm = _normalize_ticker(code)
    if is_us_ticker(norm):
        return get_price_df_us(norm, tf)
    else:
        if tf == "1d":
            return fetch_jp_daily(norm)
        raise ValueError("Intraday JP not implemented yet")

def _normalize_ticker(raw: str) -> str:
    s = raw.strip().upper()
    s = s.lstrip('$')
    s = re.sub(r"\.(T|JP|SS|SZ|HK|KS|KQ|AX|TO|V|OL|PA|L|SI|NZ|SA|MX|F|DE|SW|MI|VI|SG)$", "", s)
    s = re.sub(r"\.(T|JP|SS|SZ|HK|KS|KQ|AX|TO|V|OL|PA|L|SI|NZ|SA|MX|F|DE|SW|MI|VI|SG)$", "", s)
    # Add more normalization rules if needed
    return s