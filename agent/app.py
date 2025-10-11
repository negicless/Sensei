# agent/app.py

from pathlib import Path
import pandas as pd

# --- Pipeline imports ---
from agent.ingest.kabutan import fetch_kabutan_snapshot
from agent.ingest.price_yahoojp import fetch_price_history
from agent.parse.fundamentals import parse_fundamentals
from agent.parse.news import parse_news_items
from agent.signal.tech import add_indicators, find_levels
from agent.signal.setups import scan_setups
from agent.report.build_markdown import render_md
from agent.report.to_pdf import md_to_pdf
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from agent.ingest.price_stooq import fetch_price_stooq
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# -------- Chart horizon --------
DEFAULT_HORIZON = "1y"  # 6m | 1y | 2y | 5y | 10y | 2yW (weekly)


def _window_rows(horizon: str) -> tuple[int, bool]:
    horizon = (horizon or DEFAULT_HORIZON).lower()
    weekly = horizon.endswith("w")
    base = horizon.rstrip("w")
    days = {"6m": 130, "1y": 260, "2y": 520, "5y": 1300, "10y": 2600}.get(base, 260)
    return days, weekly


def _window_df(px: pd.DataFrame, horizon: str) -> pd.DataFrame:
    rows, weekly = _window_rows(horizon)
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


def _ensure_dirs():
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    Path("outputs/charts").mkdir(parents=True, exist_ok=True)


# agent/app.py  — replace the whole _make_chart_png() with this


def _make_chart_png(code: str, df_view: pd.DataFrame, horizon: str) -> str | None:
    import os, numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import EngFormatter

    if df_view is None or df_view.empty or len(df_view) < 20:
        return None

    df = df_view.copy()
    # Ensure types
    df["date"] = pd.to_datetime(df["date"])
    for c in ["o", "h", "l", "c", "v"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "o", "h", "l", "c"])

    # EMA ribbon
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

    # Support / Resistance
    look = min(80, len(df))
    res = df["h"].rolling(5).max().iloc[-look:].iloc[-1]
    sup = df["l"].rolling(5).min().iloc[-look:].iloc[-1]

    # Build figure
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)

    # —— Fast candlesticks (vectorized) ——
    x = np.arange(len(df))
    up = df["c"].values >= df["o"].values
    down = ~up

    # wicks
    segs = np.stack(
        [np.column_stack([x, df["l"].values]), np.column_stack([x, df["h"].values])],
        axis=1,
    )
    lc = LineCollection(segs, colors=np.where(up[:, None], "#0ECB81", "#F6465D"))
    ax.add_collection(lc)

    # bodies
    body_w = 0.6
    for side, color in [(up, "#0ECB81"), (down, "#F6465D")]:
        xs = x[side]
        o = df["o"].values[side]
        c = df["c"].values[side]
        ax.bar(
            xs,
            c - o,
            bottom=o,
            width=body_w,
            align="center",
            color=color,
            edgecolor=color,
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
        ax.plot(x, df[f"ema{e}"].values, lw=1.6, color=ema_colors[e], alpha=0.95)

    # S/R
    ax.axhline(res, ls="--", lw=1.4, color="#F6465D", alpha=0.9)
    ax.axhline(sup, ls="--", lw=1.4, color="#0ECB81", alpha=0.9)

    ax.set_title(f"{code} Chart — {trend_txt} — {align_txt}", fontweight="bold", pad=10)
    ax.grid(True, alpha=0.25)
    ax.yaxis.tick_right()
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Price", rotation=270, labelpad=18)

    # Volume
    axv.bar(x, df["v"].values, color=np.where(up, "#0ECB81", "#F6465D"), width=0.6)
    axv.grid(True, alpha=0.15)
    axv.yaxis.set_major_formatter(EngFormatter())
    axv.yaxis.tick_right()
    axv.spines["left"].set_visible(False)
    axv.set_ylabel("Volume", rotation=270, labelpad=22)

    # X ticks as dates
    locs = np.linspace(0, len(df) - 1, 6, dtype=int)
    axv.set_xticks(locs)
    axv.set_xticklabels(
        [df["date"].dt.strftime("%Y-%m-%d").iloc[i] for i in locs],
        rotation=30,
        ha="right",
    )

    # Save
    weekly = horizon.lower().endswith("w")
    out = f"outputs/charts/{code}_{'weekly' if weekly else 'daily'}_swing.png"
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out if os.path.exists(out) and os.path.getsize(out) > 1024 else None


# -------- Public API (simple, no progress) --------
def run_for_code(code: str, horizon: str = DEFAULT_HORIZON):
    """
    Snapshot -> parse -> prices -> indicators -> levels -> setups -> MD -> PDF -> chart.
    Everything computed on the SAME window as the chart.
    """
    code = str(code).zfill(4)
    _ensure_dirs()

    html = fetch_kabutan_snapshot(code)
    funda = parse_fundamentals(html)
    news = parse_news_items(html)

    # Fetch once, then window it for all downstream steps
    px_full = fetch_price_history(code)
    df_view = _window_df(px_full, horizon)

    df_view = add_indicators(df_view)
    levels = find_levels(df_view)
    ideas = scan_setups(df_view, funda)

    md_path = f"outputs/reports/{code}.md"
    pdf_path = f"outputs/reports/{code}.pdf"
    md = render_md(code, funda, news, df_view, levels, ideas)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    md_to_pdf(md_path, pdf_path)

    chart_path = _make_chart_png(code, df_view, horizon)
    return pdf_path, chart_path


# Lightweight ideas (no PDF/chart)
def build_ideas_for_code(code: str, horizon: str = DEFAULT_HORIZON):
    code = str(code).zfill(4)
    html = fetch_kabutan_snapshot(code)
    funda = parse_fundamentals(html)
    px_full = fetch_price_history(code)
    df_view = _window_df(px_full, horizon)
    df_view = add_indicators(df_view)
    return scan_setups(df_view, funda)


# --- NEW: chart-only public API (no PDF, no news/fundamentals) ---
def render_chart_only(code: str, horizon: str = DEFAULT_HORIZON) -> str:

    # Try Stooq first (fast), with a hard timeout; fallback to YahooJP
    def _fetch_stooq():
        return fetch_price_stooq(code)

    def _fetch_yj():
        return fetch_price_history(code)

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_fetch_stooq)
        try:
            px_full = fut.result(timeout=6)
        except Exception:
            # fallback
            px_full = _fetch_yj()

    df_view = _window_df(px_full, horizon)
    if df_view.empty:
        raise RuntimeError("No price data for chart window")

    df_view = add_indicators(df_view)
    chart_path = _make_chart_png(code, df_view, horizon)
    if not chart_path:
        raise RuntimeError("Chart rendering failed")
    return chart_path
    import time, logging
