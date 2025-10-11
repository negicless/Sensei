# agent/fastchart.py
# Minimal, fast, fail-safe chart pipeline for /chart. No fundamentals/news.

from __future__ import annotations
import os, io, time
from typing import Tuple
import pandas as pd
import numpy as np

# -------- matplotlib speed hygiene (must be before pyplot import) --------
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "./.mplcache")
os.makedirs("./.mplcache", exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import EngFormatter

# -------- tiny in-memory cache (5 min TTL) --------
_TTL_SEC = 300
_PRICE_CACHE: dict[str, Tuple[float, pd.DataFrame]] = {}

# -------- tiny HTTP helper using your pooled session --------
try:
    from agent.ingest.http import get  # your improved session with timeouts
except Exception:
    import requests


class DataUnavailable(Exception):
    """Raised when no price source returns data for the given code."""

    pass

    def get(url, **kw):  # fallback direct GET with timeout
        kw.setdefault("timeout", (3, 5))
        return requests.get(url, **kw)


def _fetch_stooq_prices(code: str) -> pd.DataFrame:
    sym = f"{str(code).zfill(4).lower()}.jp"  # 7013 -> 7013.jp
    bases = ["stooq.com", "stooq.pl"]  # try both
    last_err = None
    for base in bases:
        url = f"https://{base}/q/d/l/?s={sym}&i=d"
        try:
            r = get(url, timeout=(3, 4))
            r.raise_for_status()
            txt = r.text
            if not txt or txt.startswith("Error"):
                last_err = RuntimeError("stooq returned no data")
                continue
            df = pd.read_csv(io.StringIO(txt))  # Date,Open,High,Low,Close,Volume
            if df.empty:
                last_err = RuntimeError("stooq empty")
                continue
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
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[["date", "o", "h", "l", "c", "v"]].dropna()
            if df.empty:
                last_err = RuntimeError("stooq cleaned to empty")
                continue
            return df
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("stooq failed")


def _fetch_internal_yahoojp(code: str) -> pd.DataFrame:
    """Use your existing Yahoo JP ingestor (fast locally)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from agent.ingest.price_yahoojp import fetch_price_history

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fetch_price_history, code)
        try:
            df = fut.result(timeout=4.0)  # hard cap
        except TimeoutError as e:
            raise RuntimeError("internal yahoojp timeout") from e

    if df is None or df.empty:
        raise RuntimeError("internal yahoojp empty")

    need = {"date", "o", "h", "l", "c", "v"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"internal yahoojp bad columns: {df.columns}")

    df = df[["date", "o", "h", "l", "c", "v"]].dropna()
    if df.empty:
        raise RuntimeError("internal yahoojp cleaned to empty")
    return df


def _fetch_yfinance_prices(code: str) -> pd.DataFrame:
    """Fallback via yfinance (.T)."""
    try:
        import yfinance as yf, requests
    except Exception as e:
        raise RuntimeError("yfinance unavailable") from e

    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})
    tkr = yf.Ticker(f"{str(code).zfill(4)}.T", session=sess)
    df = tkr.history(period="10y", auto_adjust=False, timeout=4)
    if df is None or df.empty:
        raise RuntimeError("yfinance empty")

    df = df.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "o",
            "High": "h",
            "Low": "l",
            "Close": "c",
            "Volume": "v",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[["date", "o", "h", "l", "c", "v"]].dropna()
    if df.empty:
        raise RuntimeError("yfinance cleaned to empty")
    return df


def _fetch_yfinance_prices(code: str) -> pd.DataFrame:
    """Fallback via yfinance (.T)."""
    try:
        import yfinance as yf, requests
    except Exception as e:
        raise RuntimeError("yfinance unavailable") from e

    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})
    tkr = yf.Ticker(f"{str(code).zfill(4)}.T", session=sess)
    df = tkr.history(period="10y", auto_adjust=False, timeout=4)
    if df is None or df.empty:
        raise RuntimeError("yfinance empty")

    df = df.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "o",
            "High": "h",
            "Low": "l",
            "Close": "c",
            "Volume": "v",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[["date", "o", "h", "l", "c", "v"]].dropna()
    if df.empty:
        raise RuntimeError("yfinance cleaned to empty")
    return df


def _get_prices_cached(code: str) -> pd.DataFrame:
    now = time.time()
    key = str(code).zfill(4)

    hit = _PRICE_CACHE.get(key)
    if hit and (now - hit[0]) < _TTL_SEC:
        return hit[1]

    last_err = None
    for fetcher in (
        _fetch_stooq_prices,
        _fetch_internal_yahoojp,
        _fetch_yfinance_prices,
    ):
        try:
            df = fetcher(key)
            _PRICE_CACHE[key] = (now, df)
            return df
        except Exception as e:
            last_err = e
            continue

    raise DataUnavailable(
        f"No price data found for {key}. The code may be invalid, delisted, or unsupported."
    ) from last_err


def _window_df(px: pd.DataFrame, horizon: str) -> pd.DataFrame:
    horizon = (horizon or "1y").lower().strip()
    weekly = horizon.endswith("w")
    span = horizon[:-1] if weekly else horizon
    rows_map = {"6m": 130, "1y": 260, "2y": 520, "5y": 1300, "10y": 2600}
    rows = rows_map.get(span, 260)
    df = px.copy()
    if weekly:
        df = (
            df.set_index("date")
            .resample("W-FRI")
            .agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
            .dropna()
            .reset_index()
        )
    return df.tail(rows).copy()


def _render_fast_candles(
    code: str, df: pd.DataFrame, horizon: str, out_dir="outputs/charts"
) -> str:
    if df is None or df.empty or len(df) < 20:
        raise RuntimeError("not enough data to render")

    # ensure types
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["o", "h", "l", "c", "v"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "o", "h", "l", "c"])

    # EMAs (fast)
    emas = [8, 21, 34, 50, 100, 200]
    for e in emas:
        df[f"ema{e}"] = df["c"].ewm(span=e, adjust=False).mean()

    # trend text
    last = df.iloc[-1]
    bull = all(last[f"ema{a}"] > last[f"ema{b}"] for a, b in zip(emas, emas[1:]))
    bear = all(last[f"ema{a}"] < last[f"ema{b}"] for a, b in zip(emas, emas[1:]))
    trend_txt = "Strong Uptrend" if bull else ("Strong Downtrend" if bear else "Mixed")
    align_txt = (
        "Perfect Bullish Swing Alignment"
        if bull
        else (
            "Perfect Bearish Swing Alignment" if bear else "Neutral / Mixed Alignment"
        )
    )

    # simple S/R bands
    look = min(80, len(df))
    res = df["h"].rolling(5).max().iloc[-look:].iloc[-1]
    sup = df["l"].rolling(5).min().iloc[-look:].iloc[-1]

    # figure
    os.makedirs(out_dir, exist_ok=True)
    weekly = horizon.lower().endswith("w")
    out_path = os.path.join(
        out_dir, f"{code}_{'weekly' if weekly else 'daily'}_swing.png"
    )

    plt.rcParams["font.family"] = "DejaVu Sans"  # avoid heavy font scans
    fig = plt.figure(figsize=(11.5, 5.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)

    # vectorized candlesticks
    x = np.arange(len(df))
    up = df["c"].values >= df["o"].values
    down = ~up

    # wicks (LineCollection)
    segs = np.stack(
        [np.column_stack([x, df["l"].values]), np.column_stack([x, df["h"].values])],
        axis=1,
    )
    wick_colors = np.where(up, "#0ECB81", "#F6465D").tolist()  # <-- 1D, not [:, None]
    lc = LineCollection(segs, colors=wick_colors, linewidths=1.0)
    ax.add_collection(lc)

    # candle bodies
    body_w = 0.6
    ax.bar(
        x[up],
        (df["c"] - df["o"]).values[up],
        bottom=df["o"].values[up],
        width=body_w,
        color="#0ECB81",
        edgecolor="#0ECB81",
        align="center",
    )
    ax.bar(
        x[down],
        (df["c"] - df["o"]).values[down],
        bottom=df["o"].values[down],
        width=body_w,
        color="#F6465D",
        edgecolor="#F6465D",
        align="center",
    )

    # EMAs
    ema_colors = {
        8: "#FF7F0E",
        21: "#E31A1C",
        34: "#FDBF6F",
        50: "#2CA02C",
        100: "#1F77B4",
        200: "#7F3FBF",
    }
    for e in emas:
        ax.plot(x, df[f"ema{e}"].values, lw=1.5, color=ema_colors[e], alpha=0.95)

    # S/R
    ax.axhline(res, ls="--", lw=1.4, color="#F6465D", alpha=0.9)
    ax.axhline(sup, ls="--", lw=1.4, color="#0ECB81", alpha=0.9)

    # cosmetics
    ax.set_title(f"{code} Chart — {trend_txt} — {align_txt}", fontweight="bold", pad=8)
    ax.grid(True, alpha=0.25)
    ax.yaxis.tick_right()
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Price", rotation=270, labelpad=16)

    # volume
    axv.bar(x, df["v"].values, color=np.where(up, "#0ECB81", "#F6465D"), width=0.6)
    axv.grid(True, alpha=0.15)
    axv.yaxis.set_major_formatter(EngFormatter())
    axv.yaxis.tick_right()
    axv.spines["left"].set_visible(False)
    axv.set_ylabel("Vol", rotation=270, labelpad=18)

    # x ticks as dates (sparse)
    locs = np.linspace(0, len(df) - 1, 6, dtype=int)
    axv.set_xticks(locs)
    axv.set_xticklabels(
        [df["date"].dt.strftime("%Y-%m-%d").iloc[i] for i in locs],
        rotation=30,
        ha="right",
    )

    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        raise RuntimeError("chart not written")
    return out_path


# -------- public API --------
def render_chart_fast(code: str, horizon: str = "1y") -> str:
    """
    Ultra-fast chart path used by /chart.
    - Stooq EOD prices with hard timeout
    - 5min TTL cache
    - fast candlestick renderer
    """
    # timings (quick visibility)
    t0 = time.perf_counter()
    px = _get_prices_cached(code)  # cached fetch
    t1 = time.perf_counter()
    view = _window_df(px, horizon)  # trim/resample
    t2 = time.perf_counter()
    path = _render_fast_candles(str(code).zfill(4), view, horizon)
    t3 = time.perf_counter()
    print(
        f"[FAST-CHART] {code} ms: fetch={(t1-t0)*1000:.0f} window={(t2-t1)*1000:.0f} render={(t3-t2)*1000:.0f} total={(t3-t0)*1000:.0f}"
    )
    return path


# optional: command-line test
if __name__ == "__main__":
    import sys

    c = (sys.argv[1] if len(sys.argv) > 1 else "7013").zfill(4)
    h = sys.argv[2] if len(sys.argv) > 2 else "1y"
    print(render_chart_fast(c, h))
