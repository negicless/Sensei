# agent/fastchart.py
# Minimal, fast, fail-safe chart pipeline for /chart. No fundamentals/news.

from __future__ import annotations
import os, io, time, re
from typing import Tuple
from datetime import timedelta
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

# ✅ Use resolve() as the single source of truth (no suffix/zero-fill here)
from agent.ingest.tickers import resolve, normalize


# -------- tiny in-memory cache (5 min TTL) --------
_TTL_SEC = 300
_PRICE_CACHE: dict[str, Tuple[float, pd.DataFrame]] = {}

# -------- tiny HTTP helper using your pooled session --------
try:
    from agent.ingest.https import get  # pooled session with timeouts
except Exception:
    import requests
    def get(url, **kw):  # fallback direct GET with timeout
        kw.setdefault("timeout", (3, 5))
        return requests.get(url, **kw)


class DataUnavailable(Exception):
    """Raised when no price source returns data for the given code."""
    pass


# ------------------------------- Fetchers -------------------------------

def _fetch_stooq_prices(jp_code_4d: str) -> pd.DataFrame:
    """
    Stooq supports 4-digit JP EOD only (e.g., 7013.jp).
    Do NOT call for JP '###A' codes.
    """
    if not re.fullmatch(r"\d{4}", jp_code_4d):
        raise RuntimeError("stooq only supports JP 4-digit codes")
    sym = f"{jp_code_4d.lower()}.jp"
    bases = ["stooq.com", "stooq.pl"]
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


def _fetch_internal_yahoojp(yahoo_symbol: str) -> pd.DataFrame:
    """Use your existing Yahoo JP ingestor (fast locally)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from agent.ingest.price_yahoojp import fetch_price_history

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fetch_price_history, yahoo_symbol)
        try:
            df = fut.result(timeout=4.0)
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


def _fetch_yf_intraday(code_raw: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """
    Fetch intraday (1m–90m) via yfinance using resolve() for the symbol.
    """
    try:
        import yfinance as yf
        from yfinance.exceptions import YFDataException
    except Exception as e:
        raise RuntimeError("yfinance unavailable") from e

    market, symbol = resolve(code_raw)
    print(f"[fastchart] yfinance symbol → {symbol}")

    try:
        tkr = yf.Ticker(symbol)
        df = tkr.history(period=period, interval=interval, auto_adjust=False)
    except YFDataException as e:
        raise RuntimeError(f"yfinance intraday error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"yfinance intraday fetch failed: {e}") from e

    if df is None or df.empty:
        raise RuntimeError(f"yfinance intraday empty ({interval}/{period}) for {symbol}")

    df = df.reset_index().rename(
        columns={
            "Datetime": "date",
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
    return df


def _fetch_yf_eod(yahoo_symbol: str) -> pd.DataFrame:
    """Generic EOD via yfinance using the already-resolved symbol."""
    try:
        import yfinance as yf
        from yfinance.exceptions import YFDataException
    except Exception as e:
        raise RuntimeError("yfinance unavailable") from e

    try:
        tkr = yf.Ticker(yahoo_symbol)
        df = tkr.history(period="10y", auto_adjust=False)
    except YFDataException as e:
        raise RuntimeError(f"yfinance EOD error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"yfinance EOD fetch failed: {e}") from e

    if df is None or df.empty:
        raise RuntimeError(f"yfinance empty for {yahoo_symbol}")

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
    return df


def _get_prices_cached(code_raw: str) -> pd.DataFrame:
    """
    EOD path (cached). Uses resolve() to choose sources and cache key.
    JP 4-digit: try internal YahooJP → yfinance → stooq
    JP 3D+1L : try internal YahooJP → yfinance (no stooq)
    US/etc.  : yfinance
    """
    now = time.time()
    market, symbol = resolve(code_raw)
    cache_key = symbol  # robust for JP/US; includes suffix where needed

    hit = _PRICE_CACHE.get(cache_key)
    if hit and (now - hit[0]) < _TTL_SEC:
        return hit[1]

    last_err = None
    try_order = []

    if market == "JP":
        try_order = [
            lambda: _fetch_internal_yahoojp(symbol),
            lambda: _fetch_yf_eod(symbol),
        ]
        # If it’s JP 4-digit, add stooq fallback
        base = symbol.split(".", 1)[0]
        if re.fullmatch(r"\d{4}", base):
            try_order.append(lambda: _fetch_stooq_prices(base))
    else:
        # US / others
        try_order = [lambda: _fetch_yf_eod(symbol)]

    for fetcher in try_order:
        try:
            df = fetcher()
            _PRICE_CACHE[cache_key] = (now, df)
            return df
        except Exception as e:
            last_err = e
            continue

    raise DataUnavailable(
        f"No price data found for '{code_raw}' ({symbol}). The code may be invalid, delisted, or unsupported."
    ) from last_err


# --------------------------- Intraday helpers ---------------------------

_INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}

def _parse_intraday(horizon: str) -> tuple[bool, str, str]:
    """
    Returns (is_intraday, interval, period)
    Accepts '5m' or '5m:10d' (interval:period). Defaults: 5m:5d.
    """
    if not horizon:
        return (False, "", "")
    h = horizon.strip().lower()
    if ":" in h:
        interval, period = h.split(":", 1)
        interval, period = interval.strip(), period.strip()
    else:
        interval, period = h, "5d"
    if interval in _INTRADAY_INTERVALS:
        if not re.fullmatch(r"\d+\s*[dhw]$", period):
            period = "5d"
        return (True, interval, period)
    return (False, "", "")


# ------------------------------- Windowing ------------------------------

def _window_df(px: pd.DataFrame, horizon: str) -> pd.DataFrame:
    horizon = (horizon or "14d").lower().strip()
    df = px.copy().sort_values("date").reset_index(drop=True)

    weekly_flag = horizon.endswith("w") and horizon[:-1] in {"6m", "1y", "2y", "5y", "10y"}
    span = horizon[:-1] if weekly_flag else horizon

    rows_map = {"6m": 130, "1y": 260, "2y": 520, "5y": 1300, "10y": 2600}
    m_days = re.fullmatch(r"(\d+)\s*d", span)
    m_wks  = re.fullmatch(r"(\d+)\s*w(k)?", span)

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

    if m_days:
        ndays = int(m_days.group(1))
        approx_rows = int(ndays * 5 / 7 * 1.2)
        need_rows = max(MIN_ROWS, approx_rows)
        cutoff = df["date"].max() - timedelta(days=ndays)
        view = df[df["date"] >= cutoff]
        if len(view) < need_rows:
            view = df.tail(need_rows)
        return view.copy()

    if m_wks:
        nw = int(m_wks.group(1))
        approx_rows = int(nw * 5 * 1.2)
        need_rows = max(MIN_ROWS, approx_rows)
        cutoff = df["date"].max() - timedelta(days=7 * nw)
        view = df[df["date"] >= cutoff]
        if len(view) < need_rows:
            view = df.tail(need_rows)
        return view.copy()

    rows = rows_map.get(span, 260)
    return df.tail(max(MIN_ROWS, rows)).copy()


# ------------------------------ Rendering -------------------------------

def _label_for(market: str, yahoo_symbol: str) -> str:
    """
    Human label for titles/filenames (no suffixes).
    JP: '7013.T' -> '7013', '247A.T' -> '247A'
    US: keep symbol (e.g., 'FUBO', 'BRK-B')
    Index: '^N225' -> 'NIKKEI225'
    """
    if yahoo_symbol.startswith("^"):
        return "NIKKEI225" if yahoo_symbol.upper() == "^N225" else yahoo_symbol.lstrip("^")
    if market == "JP":
        return yahoo_symbol.split(".", 1)[0]
    return yahoo_symbol


def _render_fast_candles(
    code_label: str, df: pd.DataFrame, horizon: str, out_dir: str = "outputs/charts"
) -> str:
    """
    Institutional Dark Intraday Renderer
    """
    if df is None or df.empty or len(df) < 20:
        raise RuntimeError("Not enough data to render")

    # ---------- Theme ----------
    th = {
        "bg":   "#0d1117",
        "panel":"#0d1117",
        "frame":"#1e2736",
        "grid": "#1c2333",
        "grid_a": 0.25,
        "grid_w": 0.5,
        "tick": "#b3b8c3",
        "label":"##b3b8c3".replace("##","#"),
        "title":"#e2e5ec",
        "up":   "#26a69a",
        "down": "#ef5350",
        "sr_up":"#26a69a",
        "sr_dn":"#ef5350",
    }

    # ---------- Data prep ----------
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["o", "h", "l", "c", "v"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "o", "h", "l", "c"]).reset_index(drop=True)

    n = len(df)
    is_intraday, interval, period = _parse_intraday(horizon)

    # --- EMAs ---
    emas = [5, 25, 75, 200]
    ema_colors = {5: "#00BFFF", 25: "#FFD700", 75: "#FF6347", 200: "#9575cd"}
    for e in emas:
        df[f"ema{e}"] = df["c"].ewm(span=e, adjust=False).mean()

    EMA_LW = float(os.getenv("FASTCHART_EMA_LW", "1.8"))
    EMA_ALPHA = float(os.getenv("FASTCHART_EMA_ALPHA", "0.9"))

    last = df.iloc[-1]
    bull = all(last[f"ema{a}"] > last[f"ema{b}"] for a, b in zip(emas, emas[1:]))
    bear = all(last[f"ema{a}"] < last[f"ema{b}"] for a, b in zip(emas, emas[1:]))

    trend_txt = "Strong Uptrend" if bull else ("Strong Downtrend" if bear else "Mixed")
    align_txt = (
        "Perfect Bullish Swing Alignment"
        if bull
        else ("Perfect Bearish Swing Alignment" if bear else "Neutral / Mixed Alignment")
    )

    look = min(80, len(df))
    res = df["h"].rolling(5).max().iloc[-look:].max()
    sup = df["l"].rolling(5).min().iloc[-look:].min()

    # ---------- Figure layout ----------
    FIG_DPI = int(os.getenv("FASTCHART_DPI", "140"))
    FIG_H   = float(os.getenv("FASTCHART_FIG_H", "6.5"))
    BAR_PX  = float(os.getenv("FASTCHART_BAR_PX", "8"))
    MIN_W, MAX_W = 8.0, 18.0
    SIDE_PX = 120

    fig_w = (n * BAR_PX + SIDE_PX) / FIG_DPI
    fig_w = max(MIN_W, min(MAX_W, fig_w))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{code_label}_{'intra' if is_intraday else 'daily'}_swing.png")

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(fig_w, FIG_H), facecolor=th["bg"])
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.2], hspace=0.10)
    ax  = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)

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

    x = np.arange(n)
    o, h, l, c, v = df["o"].values, df["h"].values, df["l"].values, df["c"].values, df["v"].values
    up = c >= o

    segs = np.stack([np.column_stack([x, l]), np.column_stack([x, h])], axis=1)
    lc = LineCollection(segs, colors=np.where(up, th["up"], th["down"]).tolist(), linewidths=1.0)
    ax.add_collection(lc)

    body_w = 0.8 if is_intraday else 0.72
    ax.bar(x[up],   (c - o)[up],   bottom=o[up],   width=body_w, color=th["up"],   edgecolor="#0d1117", linewidth=0.4)
    ax.bar(x[~up],  (c - o)[~up],  bottom=o[~up],  width=body_w, color=th["down"], edgecolor="#0d1117", linewidth=0.4)

    for e in emas:
        ax.plot(x, df[f"ema{e}"].values, lw=EMA_LW, alpha=EMA_ALPHA,
                color=ema_colors[e], solid_capstyle="round")

    if os.getenv("FASTCHART_ZONES", "1") == "1":
        ax.axhspan(res * 0.995, res * 1.005, color=th["sr_dn"], alpha=0.10)
        ax.axhspan(sup * 0.995, sup * 1.005, color=th["sr_up"], alpha=0.10)
        mid = (res + sup) / 2.0
        mid_color = "#00BFFF" if bull else ("#FF7F7F" if bear else "#6CA6CD")
        ax.axhline(mid, color=mid_color, ls="--", lw=1.0, alpha=0.70)
        band_bps = int(os.getenv("FASTCHART_MID_BAND_BPS", "0"))
        if band_bps > 0:
            w = mid * (band_bps / 10000.0)
            ax.axhspan(mid - w, mid + w, color=mid_color, alpha=0.08)

    ax.set_title(f"{code_label} Chart — {trend_txt} — {align_txt}",
                 color=th["title"], fontweight="bold", pad=10)
    ax.set_ylabel("Price", rotation=270, labelpad=18, color=th["label"])
    ax.set_ylim(df["l"].min() * 0.995, df["h"].max() * 1.005)

    axv.bar(x, v, color=np.where(up, th["up"], th["down"]), width=min(0.95, body_w + 0.06))
    axv.set_ylabel("Vol", rotation=270, labelpad=18, color=th["label"])
    axv.yaxis.set_major_formatter(EngFormatter())
    axv.set_ylim(0, (v.max() if np.isfinite(v.max()) else 0) * 1.2)

    TZNAME = os.getenv("FASTCHART_TZ", "Asia/Tokyo")
    try:
        import pytz
        tz = pytz.timezone(TZNAME)
    except Exception:
        tz = None

    s = df["date"]
    if tz is not None:
        try:
            s_disp = s.dt.tz_localize("UTC").dt.tz_convert(tz) if s.dt.tz is None else s.dt.tz_convert(tz)
        except Exception:
            s_disp = s
    else:
        s_disp = s

    ticks = 8 if is_intraday else (6 if n > 120 else 4)
    locs = np.linspace(0, n - 1, ticks, dtype=int)
    same_day = s_disp.dt.date.nunique() == 1
    fmt = "%H:%M" if (is_intraday and same_day) else ("%m-%d %H:%M" if is_intraday else "%m-%d")
    labels = [s_disp.dt.strftime(fmt).iloc[i] for i in locs]
    axv.set_xticks(locs)
    axv.set_xticklabels(labels, rotation=25, ha="right", color=th["tick"])
    axv.tick_params(axis="x", pad=12, length=0)

    fig.subplots_adjust(bottom=0.15)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor=th["bg"])
    plt.close(fig)

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        raise RuntimeError("chart not written")
    return out_path


# ------------------------------- Public API -----------------------------

def render_chart_fast(code_raw: str, horizon: str = "14d") -> str:
    """
    Ultra-fast chart path.
      - Daily/EOD: cached multi-source fetch, then horizon windowing
      - Intraday (e.g., '5m' or '15m:30d'): yfinance intraday fetch (no windowing)
    """
    t0 = time.perf_counter()

    is_intra, interval, period = _parse_intraday(horizon)
    market, symbol = resolve(code_raw)
    code_label = _label_for(market, symbol)  # title/filename without suffixes

    # --- Fetch ---
    if is_intra:
        px = _fetch_yf_intraday(code_raw, interval=interval, period=period)
    else:
        px = _get_prices_cached(code_raw)

    t1 = time.perf_counter()

    # --- Window / View ---
    view = px.copy() if is_intra else _window_df(px, horizon)

    t2 = time.perf_counter()

    # --- Render ---
    path = _render_fast_candles(code_label, view, horizon)

    t3 = time.perf_counter()
    print(
        f"[FAST-CHART] {code_raw} ms: fetch={(t1-t0)*1000:.0f} "
        f"window={(t2-t1)*1000:.0f} render={(t3-t2)*1000:.0f} total={(t3-t0)*1000:.0f} "
        f"{'(intra ' + interval + ':' + period + ')' if is_intra else '(daily)'}  symbol={symbol}"
    )
    return path


# optional: command-line test
if __name__ == "__main__":
    import sys
    c = sys.argv[1] if len(sys.argv) > 1 else "7013"
    h = sys.argv[2] if len(sys.argv) > 2 else "14d"
    print(render_chart_fast(c, h))
