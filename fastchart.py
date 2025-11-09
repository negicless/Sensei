# agent/fastchart.py
# Minimal, fast, fail-safe chart pipeline for /chart. No fundamentals/news.

from __future__ import annotations

import os
import io
import re
import time
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np

# ---------- matplotlib speed hygiene (must be before pyplot import) ----------
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "./.mplcache")
os.makedirs("./.mplcache", exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

# ---------- tiny in-memory cache (5 min TTL) ----------
_TTL_SEC = 300
_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}

# ---------- output dir ----------
_OUT_DIR = "./outputs/charts"
os.makedirs(_OUT_DIR, exist_ok=True)

# ---------- optional deps ----------
try:
    import yfinance as yf
except Exception:
    yf = None  # fallback will handle


# ---------- helpers ----------
_JP_ALNUM = re.compile(r"^(?P<num>\d{3,4})(?P<suf>[A-Z])?$")  # e.g., 7011 / 270A
_INTRADAY = re.compile(r"^(?P<intv>\d+[smhd]):(?P<per>\d+[dwmy])$", re.I)  # e.g., 15m:5d


def _now() -> float:
    return time.time()


def _cache_key(sym: str, period: str, interval: str) -> str:
    return f"{sym}|{period}|{interval}"


def _symbol_map(code: str) -> Tuple[str, bool]:
    """
    Map user code to yfinance symbol.
    Returns (symbol, is_jp). JP detects for fallback to Stooq.
    JP rules:
      - 3-4 digits, optional single trailing capital letter (e.g., 270A) -> '.T'
    Otherwise, treat as global symbol (e.g., FUBO, AAPL).
    """
    c = code.strip().upper()
    if c.endswith(".T"):
        return c, True
    m = _JP_ALNUM.match(c)
    if m:
        return f"{c}.T", True
    return c, False


def _horizon_to_yf(h: str) -> Tuple[str, str]:
    """
    Convert a horizon string to (period, interval) for yfinance.
    Supports:
      - '5d', '1mo', '3mo', '6m'/'6mo', '1y', '2y'
      - intraday '15m:5d', '30m:10d', '1h:60d', etc.
    Defaults to ('6mo', '1d') if unknown.
    """
    if not h:
        return "6mo", "1d"

    h = h.strip().lower()
    mi = _INTRADAY.match(h)
    if mi:
        return mi.group("per").lower(), mi.group("intv").lower()

    # normalize common aliases
    alias = {
        "6m": "6mo",
        "3m": "3mo",
        "1m": "1mo",
        "2m": "2mo",
        "12m": "1y",
    }
    h = alias.get(h, h)

    valid_periods = {
        "5d", "7d", "10d", "14d", "30d",
        "1mo", "2mo", "3mo", "6mo",
        "1y", "2y", "3y",
    }
    if h in valid_periods:
        # choose interval based on span
        if h.endswith("d"):
            # short ranges -> allow higher res daily
            return h, "1d"
        if h in {"1mo", "2mo"}:
            return h, "1d"
        if h in {"3mo", "6mo"}:
            return h, "1d"
        return h, "1d"

    # default
    return "6mo", "1d"


def _trim_df(df: pd.DataFrame, max_rows: int = 240) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if len(df) > max_rows:
        return df.tail(max_rows).copy()
    return df.copy()


def _download_yf(sym: str, period: str, interval: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    # yfinance sometimes hangs when threads=True; force single-thread
    df = yf.download(
        sym,
        period=period,
        interval=interval,
        progress=False,
        threads=False,
        auto_adjust=False,
    )
    # yfinance returns multi-index columns sometimes
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].capitalize() for c in df.columns]  # ('Close', ...) style
    else:
        df.columns = [c.capitalize() for c in df.columns]
    # unify column names
    rename = {"Adj close": "AdjClose"}
    df = df.rename(columns=rename)
    return df


def _fetch_stooq_daily_jp(code_or_sym: str, timeout: float = 6.0) -> pd.DataFrame:
    """
    Fast EOD fallback for JP tickers via Stooq.
    Expects numeric/alnum code like '7011' or '270A' (with or without '.T').
    """
    import requests

    base = code_or_sym.upper().replace(".T", "")
    # Stooq JP: '7011.jp', '270a.jp'
    sym = f"{base.lower()}.jp"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    text = r.text.strip()
    if not text or text.startswith("<!DOCTYPE"):
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(text))
    # Stooq columns: Date,Open,High,Low,Close,Volume
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.rename(
        columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
    )
    df = df.set_index("Date")
    return df


def _get_price(sym: str, is_jp: bool, period: str, interval: str) -> pd.DataFrame:
    key = _cache_key(sym, period, interval)
    t = _now()
    hit = _CACHE.get(key)
    if hit and (t - hit[0]) < _TTL_SEC:
        return hit[1].copy()

    # 1) Try yfinance first
    df = _download_yf(sym, period, interval)
    if df is None or df.empty or "Close" not in df.columns:
        # 2) JP fallback to Stooq daily (only if interval is daily-like)
        if is_jp and interval in {"1d", "1wk"}:
            df = _fetch_stooq_daily_jp(sym)
            # normalize columns to yfinance-like
            if not df.empty:
                # ensure standard columns exist
                for col in ["Open", "High", "Low", "Close"]:
                    if col not in df.columns and "Close" in df.columns:
                        df[col] = df["Close"]
        else:
            df = pd.DataFrame()

    if df is None or df.empty:
        raise ValueError(f"no price data for {sym} (period={period}, interval={interval})")

    # Some yfinance intraday frames use lowercase columns
    cols = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=cols)

    _CACHE[key] = (t, df.copy())
    return df.copy()


def _format_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.10, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(EngFormatter(unit=""))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(8)
    for label in ax.get_yticklabels():
        label.set_fontsize(8)


@dataclass
class ChartSpec:
    code: str
    horizon: str
    width: int = 720    # pixels
    height: int = 360   # pixels
    dpi: int = 100      # 720x360 @100dpi -> figsize(7.2,3.6)


def render_chart_fast(code: str, horizon: str = "6m", out_dir: Optional[str] = None) -> str:
    """
    Render a minimal price line chart for the given code and horizon.
    - Supports '6m','1y','5d','3mo','2y' or intraday '15m:5d','30m:10d'.
    - Auto JP mapping (e.g., 7011 -> 7011.T, 270A -> 270A.T).
    - 5 min cache, Stooq fallback for JP daily bars.

    Returns: path to saved PNG.
    Raises: ValueError if no data.
    """
    out_dir = out_dir or _OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    sym, is_jp = _symbol_map(code)
    period, interval = _horizon_to_yf(horizon)

    logging.getLogger(__name__).info(f"[fastchart] yfinance symbol → {sym} ({period}/{interval})")

    df = _get_price(sym, is_jp, period, interval)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"no usable price data for {sym}")

    # Trim rows to keep plotting light
    df = _trim_df(df, max_rows=300)

    # Build figure
    spec = ChartSpec(code=code, horizon=horizon)
    figsize = (spec.width / spec.dpi, spec.height / spec.dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=spec.dpi)
    # Minimalist line
    ax.plot(df.index, df["Close"].astype(float).values, lw=1.4)

    # Optional: show last price as a subtle marker + text
    try:
        last_ts = df.index[-1]
        last_px = float(df["Close"].iloc[-1])
        ax.scatter([last_ts], [last_px], s=16, zorder=3)
        ax.text(
            last_ts,
            last_px,
            f"  {last_px:.2f}",
            va="center",
            ha="left",
            fontsize=8,
        )
    except Exception:
        pass

    title = f"{code} — {horizon.upper()}  ·  Close"
    ax.set_title(title, fontsize=10, pad=6)
    _format_axes(ax)
    fig.tight_layout()

    # Save
    safe_code = re.sub(r"[^\w\-\.]+", "_", code)
    safe_h = re.sub(r"[^\w\-\.]+", "_", horizon)
    out_path = os.path.join(out_dir, f"{safe_code}_{safe_h}.png")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return out_path


# ----- CLI for quick manual testing -----
if __name__ == "__main__":
    import sys

    c = sys.argv[1] if len(sys.argv) > 1 else "7011"
    h = sys.argv[2] if len(sys.argv) > 2 else "6m"
    try:
        p = render_chart_fast(c, h)
        print(p)
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"[fastchart] {c}: {e}")
        sys.exit(1)
