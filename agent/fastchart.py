# agent/fastchart.py
# Minimal, fast, fail-safe chart pipeline for /chart. No fundamentals/news.

from __future__ import annotations
import os, io, time
from typing import Tuple
import pandas as pd
import numpy as np
import re
from datetime import timedelta


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
    horizon = (horizon or "14d").lower().strip()
    df = px.copy().sort_values("date").reset_index(drop=True)

    # --- weekly mode via classic suffix 'W' on month/year spans (e.g. '2yW') ---
    weekly_flag = horizon.endswith("w") and horizon[:-1] in {
        "6m",
        "1y",
        "2y",
        "5y",
        "10y",
    }
    span = horizon[:-1] if weekly_flag else horizon

    # Classic month/year presets (trading-day approximations)
    rows_map = {"6m": 130, "1y": 260, "2y": 520, "5y": 1300, "10y": 2600}

    # New: 'Xd' (days) and 'Xwk' (weeks) daily windows
    m_days = re.fullmatch(r"(\d+)\s*d", span)
    m_wks = re.fullmatch(r"(\d+)\s*wk", span)

    # Minimum rows we want for indicators/EMAs to look okay
    MIN_ROWS = 30

    if weekly_flag:
        dfw = (
            df.set_index("date")
            .resample("W-FRI")
            .agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
            .dropna()
            .reset_index()
        )
        rows = rows_map.get(span, 260)
        return dfw.tail(max(10, rows)).copy()

    # Daily path
    if m_days:
        ndays = int(m_days.group(1))
        # Approximate trading bars: ~5/7 of calendar days, with a little headroom
        approx_rows = int(ndays * 5 / 7 * 1.2)
        need_rows = max(MIN_ROWS, approx_rows)
        # Prefer a calendar cutoff, but ensure we meet MIN_ROWS via tail fallback
        cutoff = df["date"].max() - timedelta(days=ndays)
        view = df[df["date"] >= cutoff]
        if len(view) < need_rows:
            view = df.tail(need_rows)
        return view.copy()

    if m_wks:
        nw = int(m_wks.group(1))
        approx_rows = int(nw * 5 * 1.2)  # ~5 trading days per week
        need_rows = max(MIN_ROWS, approx_rows)
        cutoff = df["date"].max() - timedelta(days=7 * nw)
        view = df[df["date"] >= cutoff]
        if len(view) < need_rows:
            view = df.tail(need_rows)
        return view.copy()

    # Classic month/year spans (daily)
    rows = rows_map.get(span, 260)
    return df.tail(max(MIN_ROWS, rows)).copy()


def _render_fast_candles(
    code: str, df: pd.DataFrame, horizon: str, out_dir: str = "outputs/charts"
) -> str:
    """
    TradingView-style renderer (tight spacing):
      • Auto width scales with candle count (fits data)
      • Wider candle/volume bodies (smaller gaps)
      • EMA ribbon + simple S/R + colored volume
      • Balanced spacing so x-labels don't collide with volume bars
    Env vars:
      FASTCHART_THEME=tv-dark|tv-light
      FASTCHART_DPI, FASTCHART_FIG_H
      FASTCHART_BAR_PX, FASTCHART_MIN_W, FASTCHART_MAX_W, FASTCHART_SIDE_PX
    """
    if df is None or df.empty or len(df) < 20:
        raise RuntimeError("not enough data to render")

    # ---------- Theme ----------
    THEME = os.getenv("FASTCHART_THEME", "tv-dark").lower()
    TV_DARK = {
        "bg": "#131722",
        "panel": "#131722",
        "frame": "#2a2e39",
        "grid": "#1f2940",
        "grid_a": 0.6,
        "grid_w": 0.6,
        "tick": "#B2B5BE",
        "label": "#B2B5BE",
        "title": "#E0E3EB",
        "up": "#26a69a",
        "down": "#ef5350",
        "sr_up": "#26a69a",
        "sr_dn": "#ef5350",
    }
    TV_LIGHT = {
        "bg": "#ffffff",
        "panel": "#ffffff",
        "frame": "#d1d4dc",
        "grid": "#e6e9f2",
        "grid_a": 1.0,
        "grid_w": 0.6,
        "tick": "#6a6d7c",
        "label": "#6a6d7c",
        "title": "#111111",
        "up": "#26a69a",
        "down": "#ef5350",
        "sr_up": "#26a69a",
        "sr_dn": "#ef5350",
    }
    th = TV_DARK if THEME == "tv-dark" else TV_LIGHT

    # ---------- Data prep ----------
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["o", "h", "l", "c", "v"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "o", "h", "l", "c"]).reset_index(drop=True)

    # EMAs
    emas = [8, 21, 34, 50, 100, 200]
    for e in emas:
        df[f"ema{e}"] = df["c"].ewm(span=e, adjust=False).mean()

    # Trend text
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

    # Simple S/R
    look = min(80, len(df))
    res = df["h"].rolling(5).max().iloc[-look:].iloc[-1]
    sup = df["l"].rolling(5).min().iloc[-look:].iloc[-1]

    # ---------- Auto-fit figure size (tighter defaults) ----------
    FIG_DPI = int(os.getenv("FASTCHART_DPI", "140"))
    FIG_H = float(os.getenv("FASTCHART_FIG_H", "6"))
    BAR_PX = float(os.getenv("FASTCHART_BAR_PX", "5"))  # tighter than 7
    MIN_W = float(os.getenv("FASTCHART_MIN_W", "8.0"))  # allow slightly smaller
    MAX_W = float(os.getenv("FASTCHART_MAX_W", "16"))
    SIDE_PX = float(os.getenv("FASTCHART_SIDE_PX", "120"))  # smaller side margin

    n = len(df)
    fig_w = (n * BAR_PX + SIDE_PX) / FIG_DPI
    fig_w = max(MIN_W, min(MAX_W, fig_w))

    os.makedirs(out_dir, exist_ok=True)
    weekly = horizon.lower().endswith("w")
    out_path = os.path.join(
        out_dir, f"{code}_{'weekly' if weekly else 'daily'}_swing.png"
    )

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(fig_w, FIG_H), facecolor=th["bg"])
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.2], hspace=0.10)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)

    # Panel backgrounds + grid/spines/ticks
    for a in (ax, axv):
        a.set_facecolor(th["panel"])
        a.grid(True, color=th["grid"], alpha=th["grid_a"], linewidth=th["grid_w"])
        a.spines["left"].set_visible(False)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_color(th["frame"])
        a.spines["bottom"].set_color(th["frame"])
        a.tick_params(colors=th["tick"])
        a.yaxis.tick_right()
        a.yaxis.set_label_position("right")

    # ---------- Candles (wider bodies → smaller gaps) ----------
    x = np.arange(n)
    o = df["o"].values
    h = df["h"].values
    l = df["l"].values
    c = df["c"].values
    v = df["v"].values
    up = c >= o

    # Wicks
    segs = np.stack([np.column_stack([x, l]), np.column_stack([x, h])], axis=1)
    lc = LineCollection(
        segs, colors=np.where(up, th["up"], th["down"]).tolist(), linewidths=1.0
    )
    ax.add_collection(lc)

    # Wider bodies; scale a bit with n so very long windows don't overfill
    if n >= 260:
        body_w = 0.68
    elif n >= 160:
        body_w = 0.72
    elif n >= 100:
        body_w = 0.78
    else:
        body_w = 0.84  # short windows = chunkier candles

    ax.bar(
        x[up],
        (c - o)[up],
        bottom=o[up],
        width=body_w,
        color=th["up"],
        edgecolor=th["up"],
        align="center",
    )
    ax.bar(
        x[~up],
        (c - o)[~up],
        bottom=o[~up],
        width=body_w,
        color=th["down"],
        edgecolor=th["down"],
        align="center",
    )

    # EMAs
    ema_colors = {
        8: "#ff9800",
        21: "#f06292",
        34: "#ffcc80",
        50: "#4caf50",
        100: "#42a5f5",
        200: "#9575cd",
    }
    for e in emas:
        ax.plot(x, df[f"ema{e}"].values, lw=1.5, color=ema_colors[e], alpha=0.95)

    # S/R
    ax.axhline(res, ls="--", lw=1.2, color=th["sr_dn"], alpha=0.9)
    ax.axhline(sup, ls="--", lw=1.2, color=th["sr_up"], alpha=0.9)

    # Titles / labels
    ax.set_title(
        f"{code} Chart — {trend_txt} — {align_txt}",
        fontweight="bold",
        pad=8,
        color=th["title"],
    )
    ax.set_ylabel("Price", rotation=270, labelpad=16, color=th["label"])

    # ---------- Volume (match candle width; slightly wider) ----------
    vol_w = min(0.95, body_w + 0.06)
    axv.bar(x, v, color=np.where(up, th["up"], th["down"]), width=vol_w, align="center")
    axv.set_ylabel("Vol", rotation=270, labelpad=18, color=th["label"])
    axv.yaxis.set_major_formatter(EngFormatter())
    axv.set_ylim(0, v.max() * 1.15)

    # ---------- X ticks (dates) ----------
    ticks = 6 if n >= 120 else (5 if n >= 60 else 4)
    locs = np.linspace(0, n - 1, ticks, dtype=int)
    axv.set_xticks(locs)
    axv.set_xticklabels(
        [df["date"].dt.strftime("%Y-%m-%d").iloc[i] for i in locs],
        rotation=30,
        ha="right",
        color=th["tick"],
    )
    axv.tick_params(axis="x", pad=12, length=0)
    fig.subplots_adjust(bottom=0.16)

    # Save
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=th["bg"])
    plt.close(fig)

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        raise RuntimeError("chart not written")
    return out_path


# -------- public API --------
def render_chart_fast(code: str, horizon: str = "2w") -> str:
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
