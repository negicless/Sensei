# -*- coding: utf-8 -*-
# agent/signal/levels.py — Mentor-style levels
# - Weekly detail: W (current), W-1 (previous), W-low (lowest low within last N weeks)
# - Modes per TF: current / body / auto, plus 4H Donchian freeze
# - 30m row appended by default
# - Adaptive bias to keep 4H distinct from Weekly
# - No ATH column in renders

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import re
from decimal import Decimal, ROUND_HALF_UP
__all__ = [
    "LevelsConfig",
    "LevelRow",
    "compute_levels_sheet",
    "render_levels_sheet_img",
    "compute_and_render",
    "as_markdown_table",
]


def _is_jp_ticker(code: str, yf_symbol: str | None = None) -> bool:
    """Detect if the ticker is Japanese (4 digits or ends with .T)."""
    if yf_symbol and yf_symbol.upper().endswith(".T"):
        return True
    return bool(re.fullmatch(r"\d{3,4}[A-Z]?", str(code)))

def _fmt_yen(x) -> str:
    """Round and format as whole yen (no decimals)."""
    if x is None or x != x:
        return "-"
    try:
        return str(int(Decimal(str(x)).quantize(0, rounding=ROUND_HALF_UP)))
    except Exception:
        return str(int(round(float(x))))

def _fmt_float(x) -> str:
    """Standard float with two decimals for non-JP tickers."""
    if x is None or x != x:
        return "-"
    return f"{x:.2f}"

def _fmt_range(lo, hi, fmtfunc):
    return f"{fmtfunc(lo)} – {fmtfunc(hi)}"

def _fmt_list(values, fmtfunc):
    return ", ".join(fmtfunc(v) for v in values if v is not None)


# --------------------------------------------------------------------------- #
#                                  CONFIG                                     #
# --------------------------------------------------------------------------- #

@dataclass
class LevelsConfig:
    # Base TFs (30m auto-appended if include_m30=True)
    tfs: Tuple[str, ...] = ("W", "D", "4H")

    # Include intraday 30m row
    include_m30: bool = True

    # Mentor smoothing (how many most-recent BODIES to average)
    smooth_bars_W: int = 3
    smooth_bars_D: int = 2
    smooth_bars_H4: int = 2
    smooth_bars_M30: int = 1

    # Per-TF computation mode: "current" | "body" | "auto" | ("donchian" for 4H)
    range_mode_W: str = "auto"
    range_mode_D: str = "auto"
    range_mode_H4: str = "auto"
    range_mode_M30: str = "auto"
    expansion_mult: float = 1.5

    # 4H Donchian controls (when range_mode_H4 == "donchian")
    donchian_bars_H4: int = 8
    h4_mode: str = "current"  # legacy/ignored but accepted

    # Adaptive 4H vs Weekly separation
    h4_bias_when_matches_weekly: bool = True
    h4_bias_compress: float = 0.25
    h4_bias_eps_ratio: float = 0.02

    # Mentor weekly detail block
    weekly_detail: bool = True          # if True, W becomes W/W-1/W-low
    weekly_detail_span: int = 4         # look back N weekly candles to pick the lowest low

    # legacy/tolerated (accepted but not used)
    ath_override: Optional[float] = None
    wick_filter_mult: float = 1.5
    weekly_mode: str = "current"
    daily_mode: str = "donchian"
    h4_donchian_auto: bool = False
    h4_auto_vol_window: int = 12
    h4_auto_high_atr_pct: float = 3.0
    h4_auto_med_atr_pct: float  = 1.8
    daily_window_days: int = 7
    weekly_window_weeks: int = 4
    period_days: int = 5
    horizon: str = "default"
    include_volume: bool = True
    ema_periods: Tuple[int, ...] = (20, 50, 200)
    debug: bool = False


# --------------------------------------------------------------------------- #
#                                   UTILS                                     #
# --------------------------------------------------------------------------- #

def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to o/h/l/c/v and ensure sorted DatetimeIndex."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    colmap = {
        "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v",
        "o": "o", "h": "h", "l": "l", "c": "c", "v": "v",
        "Date": "date", "Datetime": "date", "timestamp": "date", "time": "date",
    }
    df = df.rename(columns={c: colmap.get(c, colmap.get(c.lower(), c)) for c in df.columns})

    for k in ["o", "h", "l", "c"]:
        if k not in df.columns:
            raise ValueError(f"Missing column {k}")
    if "v" not in df.columns:
        df["v"] = 0.0

    for k in ["o", "h", "l", "c", "v"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")

    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = next((c for c in ["date", "datetime", "dt", "timestamp", "time"] if c in df.columns), None)
        dt = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.to_datetime(df.index, errors="coerce")
    else:
        dt = df.index

    mask = pd.Series(dt).notna().values
    if not mask.any():
        raise ValueError("Could not parse any datetimes for index")

    df = df.loc[mask].copy()
    dt = pd.DatetimeIndex(dt[mask])
    try:
        dt = dt.tz_localize(None)
    except Exception:
        pass
    df.index = dt
    return df.sort_index().dropna(subset=["o", "h", "l", "c"])


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule_map = {"W": "W-FRI", "D": "1D", "4H": "4H", "30m": "30T"}
    rule = rule_map[tf]
    agg = {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
    r = df.resample(rule, label="right", closed="right").agg(agg).dropna()
    return r.dropna(subset=["o", "h", "l", "c"])


def _swing_highs(df: pd.DataFrame, k: int = 3, lookback: int = 200) -> list[float]:
    """Local maxima excluding the very last bar; newest-first."""
    if len(df) < 3:
        return []
    x = df.tail(lookback + 3).iloc[:-1]
    h = x["h"].values
    peaks = [float(h[i]) for i in range(1, len(h) - 1) if h[i] > h[i-1] and h[i] > h[i+1]]
    peaks = peaks[-12:]
    return [round(v, 2) for v in peaks[-k:]][::-1]



# ----------------------- mentor computation helpers ------------------------ #

def _smoothed_body_extents(tf_df: pd.DataFrame, n: int) -> tuple[float, float]:
    """Average of last n candle BODIES (min/max of open/close)."""
    n = max(1, int(n))
    last_n = tf_df.tail(n)
    lows = last_n[["o", "c"]].min(axis=1).mean()
    highs = last_n[["o", "c"]].max(axis=1).mean()
    return round(float(lows), 2), round(float(highs), 2)


def _current_range_extents(tf_df: pd.DataFrame) -> tuple[float, float]:
    """Current candle low/high (wicks allowed)."""
    last = tf_df.tail(1).iloc[0]
    return round(float(last["l"]), 2), round(float(last["h"]), 2)


def _auto_pick_mode(tf_df: pd.DataFrame, mult: float, fallback: str = "body") -> str:
    """Switch to 'current' when last bar is an expansion vs recent body ranges."""
    if len(tf_df) < 3:
        return "current"
    last = tf_df.tail(1).iloc[0]
    rng_curr = float(last["h"] - last["l"])
    body_rng = (tf_df["c"] - tf_df["o"]).abs().tail(5).mean()
    try:
        body_rng = float(body_rng)
    except Exception:
        body_rng = rng_curr
    return "current" if rng_curr >= mult * body_rng else fallback


def _levels_for_tf_general(tf_df: pd.DataFrame, n: int, mode: str, mult: float) -> tuple[float, float]:
    """General TF handler (W/D/30m) + also used for 4H when mode is not 'donchian'."""
    if mode == "current":
        return _current_range_extents(tf_df)
    if mode == "body":
        return _smoothed_body_extents(tf_df, n)
    picked = _auto_pick_mode(tf_df, mult, fallback="body")
    return _current_range_extents(tf_df) if picked == "current" else _smoothed_body_extents(tf_df, n)


def _levels_for_h4(tf_df: pd.DataFrame, cfg: LevelsConfig) -> tuple[float, float]:
    """Special handler for 4H to support Donchian mode."""
    mode = getattr(cfg, "range_mode_H4", "auto")
    if mode == "donchian":
        N = max(1, int(getattr(cfg, "donchian_bars_H4", 8)))
        tail = tf_df.tail(N)
        bottom = round(float(tail["l"].min()), 2)
        top = round(float(tail["h"].max()), 2)
        return bottom, top
    return _levels_for_tf_general(tf_df, cfg.smooth_bars_H4, mode, cfg.expansion_mult)


# -------------------- adaptive H4 vs Weekly bias helpers ------------------- #

def _apply_bias_toward_mid(bottom: float, mid: float, top: float, compress: float) -> tuple[float, float]:
    """Move bottom/top toward mid by `compress` fraction (centered)."""
    b = mid - (mid - bottom) * (1.0 - compress)
    t = mid + (top - mid) * (1.0 - compress)
    return round(float(b), 2), round(float(t), 2)


def _is_almost_same_range(b1: float, t1: float, b2: float, t2: float, eps: float) -> bool:
    """Close if endpoints within eps or 4H nearly sits inside weekly span (≥80%)."""
    if abs(b1 - b2) <= eps and abs(t1 - t2) <= eps:
        return True
    span1, span2 = (t1 - b1), (t2 - b2)
    if span2 <= 0 or span1 <= 0:
        return False
    return (b1 >= b2 - eps) and (t1 <= t2 + eps) and (span1 / span2 >= 0.8)


# --------------------------------------------------------------------------- #
#                                    CORE                                     #
# --------------------------------------------------------------------------- #

@dataclass
class LevelRow:
    tf: str
    current_candle: str
    bottom: float
    mid: float
    top: float
    support_res: str
    prev_highs: str
    ath: float  # kept for back-compat (not rendered)


def _weekly_detail_rows(weekly_df: pd.DataFrame, cfg: LevelsConfig, prev_highs_txt: str) -> List[LevelRow]:
    """
    Build three rows:
      - 'W'     : current weekly candle (last bar)
      - 'W-1'   : previous weekly candle (last-1)
      - 'W-low' : candle with the lowest 'l' within cfg.weekly_detail_span
    Each row: Bottom=low, Mid=(low+high)/2, Top=high (full wick range).
    Support & Res = Bottom & Mid.
    """
    rows: List[LevelRow] = []
    if len(weekly_df) == 0:
        return rows

    def make_row(label: str, i_abs: int) -> Optional[LevelRow]:
        if i_abs < 0 or i_abs >= len(weekly_df):
            return None
        r = weekly_df.iloc[i_abs]
        lo, hi = round(float(r["l"]), 2), round(float(r["h"]), 2)
        mid = round((lo + hi) / 2.0, 2)
        current_txt = f"{lo} – {hi}"
        return LevelRow(
            tf=label,
            current_candle=current_txt,
            bottom=lo,
            mid=mid,
            top=hi,
            support_res=f"{lo} & {mid}",
            prev_highs=prev_highs_txt,
            ath=float("nan"),
        )

    idx_last = len(weekly_df) - 1
    row_curr = make_row("W", idx_last)
    if row_curr: rows.append(row_curr)

    if len(weekly_df) >= 2:
        row_prev = make_row("W-1", idx_last - 1)
        if row_prev: rows.append(row_prev)

    span = max(1, int(getattr(cfg, "weekly_detail_span", 4)))
    tail = weekly_df.tail(span)
    if len(tail) > 0:
        lowest_idx = tail["l"].idxmin()
        # convert index label → absolute integer position
        i_abs = weekly_df.index.get_loc(lowest_idx)
        row_low = make_row("W-low", i_abs)
        if row_low: rows.append(row_low)

    return rows


def compute_levels_sheet(
    df: pd.DataFrame,
    config: Optional[LevelsConfig] = None,
    *,
    tfs: Tuple[str, ...] = ("W", "D", "4H"),
    ath_override: Optional[float] = None,
    df_full: Optional[pd.DataFrame] = None,
    cfg: Optional[LevelsConfig] = None,
    symbol: Optional[str] = None,
    **kwargs,
) -> List[LevelRow]:
    """
    Mentor-aligned computation with per-TF mode:
      - 'current' → Bottom/Top = current low/high, Mid = 50%
      - 'body'    → Bottom/Top = smoothed body lows/highs
      - 'auto'    → switch to 'current' on expansion candles, else 'body'
      - 'donchian' (4H only) → Bottom/Top from last N 4H bars (min/max)
    Appends a 30m row if enabled.
    Weekly detail (W/W-1/W-low) replaces the single 'W' row when enabled.
    """
    base = _ensure_ohlc(df)

    if config is None and cfg is not None:
        config = cfg
    if config is None:
        config = LevelsConfig()

    tfs_list = list(config.tfs)
    include_m30 = getattr(config, "include_m30", True)
    if include_m30 and "30m" not in tfs_list:
        tfs_list.append("30m")

    tf_to_n = {"W": config.smooth_bars_W, "D": config.smooth_bars_D, "4H": config.smooth_bars_H4, "30m": config.smooth_bars_M30}
    tf_to_mode = {
        "W": getattr(config, "range_mode_W", "auto"),
        "D": getattr(config, "range_mode_D", "auto"),
        "4H": getattr(config, "range_mode_H4", "auto"),
        "30m": getattr(config, "range_mode_M30", "auto"),
    }
    mult = config.expansion_mult

    rows: List[LevelRow] = []
    weekly_ref = None  # for adaptive 4H vs Weekly bias

    # If weekly detail is enabled, we will expand 'W' into W/W-1/W-low and skip the plain one.
    want_weekly_detail = bool(getattr(config, "weekly_detail", True))

    for tf in tfs_list:
        tf_df = _resample(base, tf)
        if len(tf_df) == 0:
            tf_df = base.copy()

        # ---- special weekly detail block ----
        if tf == "W" and want_weekly_detail:
            # prev highs based on weekly series
            highs_w = _swing_highs(tf_df, k=3)
            prev_highs_txt_w = ", ".join(map(str, highs_w)) if highs_w else ""
            rows.extend(_weekly_detail_rows(tf_df, config, prev_highs_txt_w))
            # set weekly_ref for 4H bias using the *current* weekly candle (first of the trio)
            if rows:
                wb, wm, wt = rows[0].bottom, rows[0].mid, rows[0].top
                weekly_ref = (wb, wm, wt)
            continue  # skip normal 'W' computation

        # reference: true current range (for display)
        last = tf_df.tail(1).iloc[0]
        current_txt = f"{round(float(last['l']),2)} – {round(float(last['h']),2)}"

        # compute range per TF
        if tf == "4H":
            bottom, top = _levels_for_h4(tf_df, config)
        else:
            bottom, top = _levels_for_tf_general(tf_df, tf_to_n.get(tf, 1), tf_to_mode.get(tf, "auto"), mult)
        mid = round((bottom + top) / 2.0, 2)

        # adaptive bias if 4H ≈ Weekly
        if tf == "W" and not want_weekly_detail:
            weekly_ref = (bottom, mid, top)
        if tf == "4H" and weekly_ref is not None and getattr(config, "h4_bias_when_matches_weekly", True):
            wb, wm, wt = weekly_ref
            w_span = max(wt - wb, 1e-9)
            eps = float(getattr(config, "h4_bias_eps_ratio", 0.02)) * w_span
            if _is_almost_same_range(bottom, top, wb, wt, eps):
                compress = float(getattr(config, "h4_bias_compress", 0.25))
                bottom, top = _apply_bias_toward_mid(bottom, mid, top, compress)
                mid = round((bottom + top) / 2.0, 2)

        support_res = f"{round(bottom,2)} & {mid}"
        highs = _swing_highs(tf_df, k=3)
        prev_highs_txt = ", ".join(map(str, highs)) if highs else ""

        rows.append(
            LevelRow(
                tf=tf,
                current_candle=current_txt,
                bottom=round(bottom, 2),
                mid=mid,
                top=round(top, 2),
                support_res=support_res,
                prev_highs=prev_highs_txt,
                ath=float("nan"),  # kept for back-compat; not rendered
            )
        )

    return rows


# --------------------------------------------------------------------------- #
#                               IMAGE RENDERER                                #
# --------------------------------------------------------------------------- #

def _font(kind="regular", size=15):
    prefer = [
        ("SegoeUI-Bold.ttf", "SegoeUI.ttf"),
        ("arialbd.ttf", "arial.ttf"),
        ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"),
    ]
    for b, r in prefer:
        try:
            return ImageFont.truetype(b if kind == "bold" else r, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_levels_sheet_img(
    rows: List[LevelRow],
    title: str = "Instrument",
    path: str = "./outputs/levels_sheet.png",
    *,
    scale: float = 2.0,
    dpi: int = 220,
    symbol: Optional[str] = None,
) -> str:
    """Compact, high-contrast table (no ATH) at high resolution."""
    # ---- choose formatter (JP = integer yen, else 2dp) ----
    # fallback: infer code from the first token of title if symbol is not provided
    code_guess = (title.split("—", 1)[0].strip().split()[0] if title else None)
    is_jp = _is_jp_ticker(code_guess or "", symbol)
    fmt_num = _fmt_yen if is_jp else _fmt_float

    # helpers to reformat existing text fields (current_candle/support_res/prev_highs)
    import re as _re

    def _refmt_range_text(txt: str) -> str:
        # find two numbers in "A – B" and reformat them
        nums = _re.findall(r"-?\d+(?:\.\d+)?", txt or "")
        if len(nums) >= 2:
            a, b = float(nums[0]), float(nums[1])
            return f"{fmt_num(a)} – {fmt_num(b)}"
        return txt or "-"

    def _refmt_pair_text(txt: str) -> str:
        # find two numbers in "A & B"
        nums = _re.findall(r"-?\d+(?:\.\d+)?", txt or "")
        if len(nums) >= 2:
            a, b = float(nums[0]), float(nums[1])
            return f"{fmt_num(a)} & {fmt_num(b)}"
        return txt or "-"

    def _refmt_list_text(txt: str) -> str:
        if not txt:
            return ""
        parts = []
        for m in _re.findall(r"-?\d+(?:\.\d+)?", txt):
            parts.append(fmt_num(float(m)))
        return ", ".join(parts)

    headers = ["TF", "Current Candle", "Bottom", "Mid", "Top", "Support & Res", "Previous Highs"]
    data = [
        [
            r.tf,
            _refmt_range_text(r.current_candle),
            "-" if math.isnan(r.bottom) else fmt_num(r.bottom),
            "-" if math.isnan(r.mid) else fmt_num(r.mid),
            "-" if math.isnan(r.top) else fmt_num(r.top),
            _refmt_pair_text(r.support_res),
            _refmt_list_text(r.prev_highs),
        ]
        for r in rows
    ]

    pad_x, row_h, header_h, title_h = int(14*scale), int(32*scale), int(36*scale), int(20*scale)
    font  = _font("regular", int(12*scale))
    bold  = _font("bold",    int(12*scale))
    title_font = _font("bold", int(13*scale))
    small = _font("regular", int(12*scale))

    def w(t, f=font):
        d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        try:
            return int(d.textlength(t, font=f))
        except Exception:
            return d.textbbox((0, 0), t, font=f)[2]

    widths = [max([w(h, bold)] + [w(r[i], font) for r in data]) + 2*pad_x for i, h in enumerate(headers)]
    wrap_col = headers.index("Previous Highs")

    def wrap(txt, lim=22):
        if not txt or len(txt) <= lim:
            return txt
        parts, line = [], ""
        for token in txt.split(", "):
            add = (", " if line else "") + token
            if len(line) + len(add) > lim:
                parts.append(line); line = token
            else:
                line += add
        if line:
            parts.append(line)
        return "\n".join(parts)

    for r in data:
        r[wrap_col] = wrap(r[wrap_col])

    def multiw(s):
        return max(w(line) for line in s.split("\n")) + 2*pad_x

    widths[wrap_col] = max(w(headers[wrap_col], bold), max(multiw(r[wrap_col]) for r in data))
    W, H = int(sum(widths)), int(title_h + header_h + len(data)*row_h + 40*scale)

    bg_dark, row_dark, head_col = (18,18,22), (26,26,30), (35,35,40)
    border, txt_main = (50,50,55), (230,230,235)

    img = Image.new("RGB", (W, H), bg_dark)
    draw = ImageDraw.Draw(img)

    draw.text((pad_x, int(6*scale)), f"{title} — Levels", fill=txt_main, font=title_font)

    y, x = int(title_h), 0
    for i, htxt in enumerate(headers):
        draw.rectangle([x, y, x+widths[i], y+header_h], fill=head_col, outline=border)
        draw.text((x+pad_x, y+int(10*scale)), htxt, fill=txt_main, font=bold)
        x += widths[i]

    y += header_h
    for r in data:
        x = 0
        for i, cell in enumerate(r):
            draw.rectangle([x, y, x+widths[i], y+row_h], fill=row_dark, outline=border)
            if i == wrap_col and "\n" in cell:
                draw.multiline_text((x+pad_x, y+int(6*scale)), cell, fill=txt_main, font=small, spacing=int(2*scale))
            else:
                draw.text((x+pad_x, y+int(8*scale)), cell, fill=txt_main, font=font)
            x += widths[i]
        y += row_h

    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path, dpi=(dpi, dpi))
    return path




# --------------------------------------------------------------------------- #
#                               MARKDOWN TABLE                                #
# --------------------------------------------------------------------------- #

def as_markdown_table(rows: List[LevelRow], title: str | None = None, symbol: Optional[str] = None) -> str:
    headers = ["TF", "Current Candle", "Bottom", "Mid", "Top", "Support & Res", "Previous Highs"]
    lines = []
    if title:
        lines.append(f"**{title} — Levels**\n")

    code_guess = (title.split("—", 1)[0].strip().split()[0] if title else None)
    is_jp = _is_jp_ticker(code_guess or "", symbol)
    fmt_num = _fmt_yen if is_jp else _fmt_float

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        row = [
            r.tf,
            # reformat embedded numbers in strings too
            re.sub(r"-?\d+(?:\.\d+)?", lambda m: fmt_num(float(m.group(0))), r.current_candle),
            "-" if math.isnan(r.bottom) else fmt_num(r.bottom),
            "-" if math.isnan(r.mid) else fmt_num(r.mid),
            "-" if math.isnan(r.top) else fmt_num(r.top),
            re.sub(r"-?\d+(?:\.\d+)?", lambda m: fmt_num(float(m.group(0))), r.support_res),
            re.sub(r"-?\d+(?:\.\d+)?", lambda m: fmt_num(float(m.group(0))), r.prev_highs or ""),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)



# --------------------------------------------------------------------------- #
#                            ONE-LINER CONVENIENCE                            #
# --------------------------------------------------------------------------- #

def compute_and_render(
    df: pd.DataFrame,
    *,
    title: str,
    config: Optional[LevelsConfig] = None,
    ath_override: Optional[float] = None,   # accepted for back-compat (ignored by render)
    out_path: str = "./outputs/levels_sheet.png",
    df_full: Optional[pd.DataFrame] = None, # accepted for back-compat (ignored by render)
    cfg: Optional[LevelsConfig] = None,
    symbol: Optional[str] = None,
    scale: float = 2.0,
    dpi: int = 220,
    **kwargs,
) -> str:
    rows = compute_levels_sheet(
        df,
        config=config,
        ath_override=ath_override,
        df_full=df_full,
        cfg=cfg,
        symbol=symbol,
        **kwargs,
    )
    return render_levels_sheet_img(rows, title=title, path=out_path, scale=scale, dpi=dpi, symbol=symbol)
