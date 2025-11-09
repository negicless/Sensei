# agent/ingest/price_us.py
from __future__ import annotations
import io
import re
import pandas as pd
import numpy as np
from datetime import timezone
from dateutil import tz
from agent.ingest.http import get  # your pooled session with timeouts

# ---------- symbol helpers ----------
_YF_SYMBOL_FIX = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

def _to_yf(sym: str) -> str:
    s = sym.strip().upper()
    s = s.lstrip("$")
    # if user typed FUBO.T / FUBO.US, drop the suffix for US tickers
    s = re.sub(r"\.(US|T|JP|SS|SZ|HK|KS|KQ|AX|TO|V|OL|PA|L|SI|NZ|SA|MX|F|DE|SW|MI|VI|SG)$", "", s)
    return _YF_SYMBOL_FIX.get(s, s)

def _to_stooq_us(sym: str) -> str:
    # stooq expects lowercase + .us (e.g., fubo.us)
    return f"{sym.strip().lower()}.us"

# ---------- EOD: Stooq (US) ----------
def fetch_us_daily_stooq(symbol: str) -> pd.DataFrame:
    """
    EOD prices for US tickers from Stooq.
    Returns df with columns: date,o,h,l,c,v (UTC-naive dates)
    """
    sym = _to_stooq_us(symbol)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = get(url); r.raise_for_status()
    if not r.text or r.text.lower().startswith("error"):
        raise RuntimeError(f"Stooq returned no data for {symbol}")
    df = pd.read_csv(io.StringIO(r.text))
    # Standardize
    df.rename(columns={
        "Date": "date", "Open": "o", "High": "h",
        "Low": "l", "Close": "c", "Volume": "v"
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[["date","o","h","l","c","v"]].dropna()
    return df

# ---------- Intraday/Resample: Yahoo ----------
def _download_yf(symbol: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    s = _to_yf(symbol)
    df = yf.download(s, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol} ({period}/{interval})")
    # index is tz-aware DatetimeIndex (UTC). Normalize to naive UTC for your pipeline.
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.rename(columns={
        "Open":"o","High":"h","Low":"l","Close":"c","Adj Close":"ac","Volume":"v"
    })[["o","h","l","c","v"]]
    df = df.reset_index().rename(columns={"index":"date","Datetime":"date"})
    return df

def fetch_us_intraday_yf(symbol: str, period: str="90d", interval: str="15m") -> pd.DataFrame:
    """
    Intraday US prices (15m default) from Yahoo.
    Returns: date,o,h,l,c,v (UTC naive)
    """
    return _download_yf(symbol, period=period, interval=interval)

def resample_to_4h(df_intraday: pd.DataFrame) -> pd.DataFrame:
    """
    From intraday df (date as UTC-naive), build 4H OHLCV.
    """
    df = df_intraday.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    rule = "4H"
    o = df["o"].resample(rule).first()
    h = df["h"].resample(rule).max()
    l = df["l"].resample(rule).min()
    c = df["c"].resample(rule).last()
    v = df["v"].resample(rule).sum()
    out = pd.concat([o,h,l,c,v], axis=1).dropna().reset_index()
    out.rename(columns={"date":"date"}, inplace=True)
    # Make sure sorted and types are clean
    return out[["date","o","h","l","c","v"]]

# ---------- Unified entry ----------
def get_price_df_us(symbol: str, tf: str) -> pd.DataFrame:
    """
    tf: '1d' for EOD daily history
        '15m' intraday (90d)
        '4h' resampled from 15m
    """
    tf = tf.lower()
    if tf == "1d":
        return fetch_us_daily_stooq(symbol)
    elif tf == "15m":
        return fetch_us_intraday_yf(symbol, period="90d", interval="15m")
    elif tf == "4h":
        src = fetch_us_intraday_yf(symbol, period="60d", interval="30m")  # denser base
        return resample_to_4h(src)
    else:
        raise ValueError(f"Unsupported tf: {tf}")
