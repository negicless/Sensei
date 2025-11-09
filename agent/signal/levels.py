# agent/signal/levels.py
# Levels for: 1W (closest 4 weeks), 1D (most recent day), 4H (last 1D Donchian)
# Falls back to lower TF bars while preserving *time span* (so Donchian is correct).
# Public API:
#   compute_levels(df) -> dict
#   render_levels_table_img(levels_dict, title="") -> str (png path)
#   as_markdown_table(levels_dict) -> str
#   compute_levels_sheet(df) -> dict
#   render_levels_sheet_img(sheet_dict, title="") -> str (png path)

from __future__ import annotations
import math, os, tempfile, re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Only for input validation in /levels command handler (no suffix logic here)
try:
    from agent.ingest.tickers import resolve
except Exception:
    resolve = None  # optional import for library-only usage

# ---- Data guards ----

def _ensure_df(d: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "o", "h", "l", "c", "v"]
    miss = [c for c in cols if c not in d.columns]
    if miss:
        raise ValueError(f"DataFrame missing columns: {miss}")
    x = d.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for c in ["o","h","l","c","v"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["o","h","l","c"])
    return x

# ---- Tick model (generic; adjust per exchange if needed) ----

def _tick_size(price: float) -> float:
    if price >= 1000: return 1.0
    if price >= 100:  return 0.1
    if price >= 10:   return 0.05
    if price >= 1:    return 0.01
    return 0.001

def _round_tick(x: float) -> float:
    t = _tick_size(abs(x))
    return round(x / t) * t

def _zone(lo: float, hi: float, widen_ticks: int = 0) -> Tuple[float, float]:
    if math.isnan(lo) or math.isnan(hi):
        return (float("nan"), float("nan"))
    lo, hi = float(lo), float(hi)
    if hi < lo:
        lo, hi = hi, lo
    if widen_ticks:
        lo -= widen_ticks * _tick_size(lo)
        hi += widen_ticks * _tick_size(hi)
    return (_round_tick(lo), _round_tick(hi))

def _fmt_zone(z: Tuple[float, float]) -> str:
    a, b = z
    if np.isnan(a) or np.isnan(b): return "-"
    return f"{a:g}" if a == b else f"{a:g} – {b:g}"

# ---- Highs utility ----

def _swing_highs(d: pd.DataFrame, left: int = 2, right: int = 2, k: int = 3) -> List[float]:
    """Lightweight swing-high detector for 'previous highs' column."""
    h = d["h"].to_numpy()
    res: List[float] = []
    n = len(h)
    for i in range(left, n - right):
        if all(h[i] > h[i - j - 1] for j in range(left)) and all(h[i] >= h[i + j + 1] for j in range(right)):
            res.append(_round_tick(float(h[i])))
            if len(res) >= k:
                break
    res.sort(reverse=True)
    return res

# ---- Resampling ----

def _ohlcv_resample(x: pd.DataFrame, rule: str) -> pd.DataFrame:
    y = _ensure_df(x).copy()
    y = y.set_index("date")
    agg = {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
    r = y.resample(rule, label="left", closed="left").agg(agg)
    r = r.dropna(subset=["o","h","l","c"]).reset_index().rename(columns={"date": "date"})
    return r

def _to_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "W":  return _ohlcv_resample(df, "W-FRI")
    if tf == "D":  return _ohlcv_resample(df, "D")
    if tf == "4H": return _ohlcv_resample(df, "4h")
    raise ValueError(f"Unsupported tf: {tf}")

# ---- Donchian on time-window (anchor = last timestamp) ----

def _donchian_last_by_timewin(d: pd.DataFrame, period: str) -> Tuple[float, float, float]:
    x = _ensure_df(d).set_index(pd.to_datetime(d["date"], errors="coerce"))
    x = x.dropna(subset=["o","h","l","c"])
    if len(x) < 1:
        raise ValueError("Not enough bars for time-window Donchian")

    end = x.index[-1]
    start = end - pd.Timedelta(period)
    window = x.loc[start:end]
    if window.empty:
        window = x.tail(1)

    top = float(window["h"].max())
    bot = float(window["l"].min())
    mid = 0.5 * (top + bot)
    return top, bot, mid

def _cur_candle_zone(d: pd.DataFrame) -> Tuple[float, float]:
    last_lo = float(d["l"].iloc[-1]); last_hi = float(d["h"].iloc[-1])
    return _zone(last_lo, last_hi, widen_ticks=1)

# ---- Core per-TF with fallbacks ----

def _build_payload(top: float, bot: float, d_anchor: pd.DataFrame) -> Dict[str, str]:
    mid = 0.5 * (top + bot)
    bottom_zone = _zone(bot * 0.997, bot * 1.003)
    top_zone    = _zone(top * 0.997, top * 1.003)
    cur_zone    = _cur_candle_zone(d_anchor)
    prev_highs  = _swing_highs(d_anchor, left=2, right=2, k=3)
    prev_highs_z = [_fmt_zone(_zone(ph, ph)) for ph in prev_highs]
    ath = float(d_anchor["h"].max()) if len(d_anchor) else float("nan")
    return {
        "current_candle": _fmt_zone(cur_zone),
        "support_bottom_channel": _fmt_zone(bottom_zone),
        "mid_50": f"{_round_tick(mid):g}",
        "top_sr": _fmt_zone(top_zone),
        "previous_highs": ", ".join(prev_highs_z) if prev_highs_z else "-",
        "ath": f"{_round_tick(ath):g}" if not np.isnan(ath) else "-",
    }

def _compute_levels_for_tf(df_all: Dict[str, pd.DataFrame], tf_label: str) -> Dict[str, str]:
    """
    W: 4 recent weeks; if <4 bars, use 28D Donchian on D or 4H
    D: most recent daily candle; if missing, use 1D on 4H
    4H: Donchian over last 1D; if missing, use 1D on BASE
    """
    d_tf = df_all.get(tf_label)

    if tf_label == "W":
        if d_tf is not None and len(d_tf) >= 4:
            d = d_tf.dropna(subset=["o","h","l","c"]).reset_index(drop=True)
            top = float(d["h"].tail(4).max()); bot = float(d["l"].tail(4).min())
            return _build_payload(top, bot, d)
        for lo_tf in ("D", "4H"):
            d_lo = df_all.get(lo_tf)
            if d_lo is None or len(d_lo) < 1: continue
            try:
                top, bot, _ = _donchian_last_by_timewin(d_lo, "28D")
                return _build_payload(top, bot, d_lo.reset_index(drop=True))
            except Exception:
                continue

    if tf_label == "D":
        if d_tf is not None and len(d_tf) >= 1:
            d = d_tf.dropna(subset=["o","h","l","c"]).reset_index(drop=True)
            top = float(d["h"].iloc[-1]); bot = float(d["l"].iloc[-1])
            return _build_payload(top, bot, d)
        d_lo = df_all.get("4H")
        if d_lo is not None and len(d_lo) >= 1:
            top, bot, _ = _donchian_last_by_timewin(d_lo, "1D")
            return _build_payload(top, bot, d_lo.reset_index(drop=True))

    if tf_label == "4H":
        if d_tf is not None and len(d_tf) >= 1:
            d = d_tf.dropna(subset=["o","h","l","c"]).reset_index(drop=True)
            top, bot, _ = _donchian_last_by_timewin(d, "1D")
            return _build_payload(top, bot, d)
        d_base = df_all.get("BASE")
        if d_base is not None and len(d_base) >= 1:
            top, bot, _ = _donchian_last_by_timewin(d_base, "1D")
            d_anchor = _to_tf(d_base, "4H")
            return _build_payload(top, bot, d_anchor.reset_index(drop=True))

    raise ValueError(f"Not enough data to compute {tf_label} levels (even with fallback)")

def compute_levels(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    base = _ensure_df(df)
    tf_dict = {"BASE": base, "W": _to_tf(base, "W"), "D": _to_tf(base, "D"), "4H": _to_tf(base, "4H")}
    out: Dict[str, Dict[str, str]] = {}
    for tf in ("W", "D", "4H"):
        try:
            out[tf] = _compute_levels_for_tf(tf_dict, tf)
        except Exception:
            # best-effort: skip missing TFs, but fail if all missing
            pass
    if not out:
        raise ValueError("Not enough data across all timeframes to compute levels")
    return out

# ---- Renderers ----

def _draw_table_image(rows: List[List[str]], title: str, path: str) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise RuntimeError("Pillow (PIL) is required to render images. pip install pillow") from e

    pad_x, pad_y = 28, 20
    row_h, header_h, title_h = 36, 46, 60
    # Wider columns for 'Previous Highs' overflow protection
    col_w = [160, 210, 210, 160, 340, 160]

    width = sum(col_w) + pad_x * 2
    height = title_h + header_h + row_h * len(rows) + pad_y * 2

    img = Image.new("RGB", (width, height), (16, 16, 20))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_hdr   = ImageFont.truetype("arial.ttf", 18)
        font_cell  = ImageFont.truetype("arial.ttf", 17)
    except Exception:
        font_title = ImageFont.load_default()
        font_hdr   = ImageFont.load_default()
        font_cell  = ImageFont.load_default()

    # Title
    draw.text((pad_x, pad_y), f"{title or 'Instrument'} — Levels", fill=(240,240,245), font=font_title)
    y = pad_y + title_h

    # Headers
    headers = ["TF", "Current Candle", "Bottom", "Mid", "Top", "ATH"]
    x = pad_x
    for i, h in enumerate(headers):
        draw.rectangle([x, y, x + col_w[i], y + header_h], outline=(80,80,90), fill=(30,30,36))
        draw.text((x + 8, y + 12), h, fill=(210,210,220), font=font_hdr)
        x += col_w[i]
    y += header_h

    # Rows (auto-wrap)
    for r in rows:
        x = pad_x
        max_row_h = row_h
        wrapped_cells: List[List[str]] = []
        for i, cell in enumerate(r):
            text = str(cell or "-")
            max_chars = max(6, int(col_w[i] // 9))  # naive char-wrap width
            chunks, rest = [], text
            while len(rest) > max_chars:
                cut = rest.rfind(" ", 0, max_chars)
                if cut == -1: cut = max_chars
                chunks.append(rest[:cut]); rest = rest[cut:].lstrip()
            chunks.append(rest)
            wrapped_cells.append(chunks)
            max_row_h = max(max_row_h, 20 * len(chunks) + 8)

        for i, chunks in enumerate(wrapped_cells):
            draw.rectangle([x, y, x + col_w[i], y + max_row_h], outline=(50,50,60), fill=(22,22,26))
            for j, line in enumerate(chunks):
                draw.text((x + 8, y + 6 + j * 20), line, fill=(235,235,240), font=font_cell)
            x += col_w[i]
        y += max_row_h

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path

def render_levels_table_img(levels: Dict[str, Dict[str, str]], title: str = "") -> str:
    order = ["W", "D", "4H"]
    rows: List[List[str]] = []
    for tf in order:
        b = levels.get(tf)
        if not b: continue
        rows.append([
            tf,
            b.get("current_candle", "-"),
            b.get("support_bottom_channel", "-"),
            b.get("mid_50", "-"),
            b.get("top_sr", "-"),
            b.get("ath", "-"),
        ])
    module_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    out_dir = os.path.join(module_dir, "out")
    path = os.path.join(out_dir, "levels_table.png")
    return _draw_table_image(rows, title, path)

def as_markdown_table(levels: Dict[str, Dict[str, str]]) -> str:
    order = ["W", "D", "4H"]
    headers = ["TF", "Current Candle", "Bottom", "Mid", "Top", "ATH"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for tf in order:
        b = levels.get(tf)
        if not b: continue
        row = [
            tf,
            b.get("current_candle", "-"),
            b.get("support_bottom_channel", "-"),
            b.get("mid_50", "-"),
            b.get("top_sr", "-"),
            b.get("ath", "-"),
        ]
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines)

# ---- Mentor sheet variant ----

def compute_levels_sheet(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    lv = compute_levels(df)
    out: Dict[str, Dict[str, str]] = {}
    def _parse_first(s: str) -> Optional[float]:
        try:
            if not s or s == "-": return None
            if "–" in s:
                return float(s.split("–", 1)[0].strip())
            return float(s.strip())
        except Exception:
            return None

    for tf, b in lv.items():
        out[tf] = {
            "bottom": _parse_first(b.get("support_bottom_channel", "-")),
            "mid":    _parse_first(b.get("mid_50", "-")),
            "top":    _parse_first(b.get("top_sr", "-")),
            "current_candle": b.get("current_candle", "-"),
            "previous_highs": b.get("previous_highs", "-"),
            "ath": b.get("ath", "-"),
        }
    return out

def render_levels_sheet_img(sheet: Dict[str, Dict[str, str]], title: str = "") -> str:
    order = ["W", "D", "4H"]
    rows: List[List[str]] = []
    for tf in order:
        b = sheet.get(tf)
        if not b: continue
        rows.append([
            tf,
            b.get("current_candle", "-"),
            f"{b.get('bottom', '-')}",
            f"{b.get('mid', '-')}",
            f"{b.get('top', '-')}",
            f"{b.get('ath', '-')}",
        ])
    module_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    out_dir = os.path.join(module_dir, "out")
    path = os.path.join(out_dir, "levels_sheet.png")
    return _draw_table_image(rows, title, path)

# ---- Bot handler (optional helper) ----

def levels_cmd(update, context):
    raw = (context.args[0] if context.args else "").strip()
    if not raw:
        update.message.reply_text("Usage: /levels <ticker>  e.g., /levels 7013 or /levels FUBO")
        return
    if resolve:
        try:
            resolve(raw)  # validate ticker only; suffixes handled upstream
        except ValueError as ve:
            update.message.reply_text(str(ve))
            return
