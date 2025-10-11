# agent/ingest/price_stooq.py
from __future__ import annotations
import io
import pandas as pd
from agent.ingest.http import get  # your pooled session with timeouts


def fetch_price_stooq(code: str) -> pd.DataFrame:
    """
    Fast EOD prices from Stooq. Returns DataFrame with columns:
    date,o,h,l,c,v   (UTC dates, no tz)
    """
    sym = f"{str(code).zfill(4).lower()}.jp"  # e.g., 7013 -> 7013.jp
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"  # daily CSV
    r = get(url)  # has timeout/retry
    r.raise_for_status()
    if not r.text or "Error" in r.text:
        raise RuntimeError("stooq returned no data")

    df = pd.read_csv(io.StringIO(r.text))
    # Stooq CSV columns: Date,Open,High,Low,Close,Volume
    if df.empty:
        raise RuntimeError("stooq empty")
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "o",
            "High": "h",
            "Low": "l",
            "Close": "c",
            "Volume": "v",
        }
    )
    # Ensure types
    df["date"] = pd.to_datetime(df["date"], utc=False)
    df = df[["date", "o", "h", "l", "c", "v"]].dropna()
    # Stooq delivers oldest->newest; keep as-is for our pipeline
    return df
