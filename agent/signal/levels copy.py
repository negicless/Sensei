from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def render_levels_table_img(levels: dict, ticker: str, out_path: str = None) -> str:
    """
    Render mentor-style levels as a dark-themed PNG table.
    """
    rows = []
    order = [("W","WEEKLY"),("D","DAILY"),("4H","4HOUR")]
    for key, label in order:
        if key not in levels:
            continue
        lv = levels[key]
        rows.append([
            label,
            lv["support_bottom_channel"],
            lv["mid_50"],
            lv["top_sr"],
            lv["previous_highs"],
            lv["ath"],
            lv["current_candle"],
        ])

    columns = ["TF","Bottom (Support)","Mid 50%","Top (S/R)","Previous Highs","ATH","Current Candle"]
    df = pd.DataFrame(rows, columns=columns)

    # ---------- style ----------
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.5 * len(df)))
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

        # ---------- table ----------
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )

    # auto font scaling and wrapping
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)  # taller cells

    # Compute column widths based on text length
    col_widths = []
    for col in df.columns:
        max_len = max(len(str(v)) for v in df[col].values)
        col_widths.append(0.02 * max(8, min(max_len, 30)))  # adapt but cap width
    total = sum(col_widths)
    for i, w in enumerate(col_widths):
        table.auto_set_column_width(i)
        for (row, col), cell in table.get_celld().items():
            if col == i:
                cell.set_width(w / total)

    # Style & wrap each cell’s text
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2a2e39")
        cell.set_linewidth(0.5)
        txt = cell.get_text().get_text()
        # Simple wrap at ~18 chars
        if len(txt) > 18 and " " not in txt:
            wrapped = "\n".join([txt[i:i+18] for i in range(0, len(txt), 18)])
            cell.get_text().set_text(wrapped)
        if row == 0:
            cell.set_text_props(color="#e2e5ec", weight="bold")
            cell.set_facecolor("#1e2736")
        else:
            cell.set_text_props(color="#b3b8c3", wrap=True)
            cell.set_facecolor("#0d1117")


    # Title
    ax.set_title(
        f"${ticker} — Levels",
        color="#e2e5ec",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

   # Save
    if not out_path:
        out_path = f"outputs/charts/{ticker}_levels.png"
    dirpath = os.path.dirname(out_path) or "outputs/charts"
    os.makedirs(dirpath, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return out_path

# --- resampling helper (keeps the latest bar = “closest” period) ---
def to_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    tf = tf.upper()
    rule_map = {"W": "W-FRI", "D": "1D", "4H": "4H"}
    if tf not in rule_map:
        raise ValueError(f"Unknown timeframe {tf}")

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")
    try:
        if d["date"].dt.tz is None:
            d["date"] = d["date"].dt.tz_localize("UTC")
    except Exception:
        pass
    d = d.set_index("date")

    rule = rule_map[tf]
    out = (
        d.resample(rule)
         .agg({"o":"first","h":"max","l":"min","c":"last","v":"sum"})
         .dropna()
         .reset_index()
    )
    return out



# -------- core indicators --------
def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    h, l, c = df["h"], df["l"], df["c"]
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum(abs(h - prev_c), abs(l - prev_c)))
    return tr.rolling(n, min_periods=1).mean()

# -------- swing points (fractals) --------
def swing_highs(df: pd.DataFrame, left: int = 2, right: int = 2, k: int = 2) -> list[float]:
    """Return last k swing highs (most recent first)."""
    h = df["h"].values
    idxs = []
    for i in range(left, len(h) - right):
        if h[i] == max(h[i-left:i+right+1]):
            idxs.append(i)
    out = [float(df["h"].iloc[i]) for i in reversed(idxs[-k:])]
    return out

# -------- tick-aware rounding --------
def tick_for_price(p: float) -> float:
    """
    Simple tick sizing. Adjust if you want JP unit tables.
    """
    if p < 10:   return 0.01
    if p < 50:   return 0.1
    if p < 500:  return 0.5
    if p < 3000: return 1.0
    if p < 10000:return 5.0
    return round(p * 0.001, 0)  # fallback step

def round_to_tick(p: float) -> float:
    t = tick_for_price(p)
    return float(np.round(p / t) * t)

def zone(lo: float, hi: float, widen_ticks: int = 2) -> tuple[float, float]:
    """Create a neat printable zone [lo, hi] widened by a couple of ticks."""
    t = tick_for_price((lo + hi) / 2)
    return (round_to_tick(lo - widen_ticks*t), round_to_tick(hi + widen_ticks*t))

# -------- printer --------
def fmt_zone(z: tuple[float,float]) -> str:
    a, b = z
    if a == b: return f"{a:g}"
    return f"{a:g}-{b:g}"

# -------- main levels per TF --------
def compute_levels_for_tf(df_tf: pd.DataFrame, tf_label: str) -> dict:
    """Compute levels for a single timeframe; requires minimal bars per TF."""
    d = df_tf.copy().dropna(subset=["o","h","l","c"]).reset_index(drop=True)

    # Adaptive minimums (light but robust)
    min_req = {"W": 8, "D": 30, "4H": 60}
    need = min_req.get(tf_label, 30)
    if len(d) < need:
        raise ValueError(f"Not enough {tf_label} bars (have {len(d)}, need {need})")

    # Core indicators
    d["ema20"] = ema(d["c"], 20)
    d["atr20"] = atr(d, 20)

    top  = d["ema20"].iloc[-1] + 2.0 * d["atr20"].iloc[-1]
    bot  = d["ema20"].iloc[-1] - 2.0 * d["atr20"].iloc[-1]
    mid  = 0.5 * (top + bot)

    # Current candle zone
    last_lo, last_hi = float(d["l"].iloc[-1]), float(d["h"].iloc[-1])
    cur_zone = zone(last_lo, last_hi, widen_ticks=1)

    # Channel zones
    bottom_zone = zone(bot*0.997, bot*1.003)
    mid_zone    = (round_to_tick(mid), round_to_tick(mid))  # printed as a line
    top_zone    = zone(top*0.997, top*1.003)

    # Previous highs (two recent swings) & ATH
    prev_highs = swing_highs(d, left=2, right=2, k=2)
    prev_highs_z = [fmt_zone(zone(ph, ph)) for ph in prev_highs]
    ath = float(d["h"].max())

    return {
        "current_candle": fmt_zone(cur_zone),
        "support_bottom_channel": fmt_zone(bottom_zone),
        "mid_50": f"{mid_zone[0]:g}",
        "top_sr": fmt_zone(top_zone),
        "previous_highs": ", ".join(prev_highs_z) if prev_highs_z else "-",
        "ath": f"{round_to_tick(ath):g}",
    }


def compute_levels(df: pd.DataFrame) -> dict:
    """
    Try W, D, 4H in that order; skip TFs that don't meet the minimum bars.
    Raise only if ALL fail.
    """
    out = {}
    for tf in ("W", "D", "4H"):
        d_tf = to_tf(df, tf)
        try:
            out[tf] = compute_levels_for_tf(d_tf, tf)
        except Exception:
            # Skip thin TF silently
            continue

    if not out:
        raise ValueError("Not enough data across all timeframes to compute levels")
    return out


def as_markdown_table(levels: dict, ticker: str) -> str:
    """
    Render mentor-style table.
    """
    lines = [f"**${ticker} — Levels**",
             "",
             "| TF | Bottom (Support) | Mid 50% | Top (S/R) | Previous Highs | ATH | Current Candle |",
             "|---:|:------------------|:--------:|:----------|:---------------|:----:|:---------------|"]
    order = [("W","WEEKLY"),("D","DAILY"),("4H","4 Hour")]
    for key, label in order:
        lv = levels[key]
        lines.append(
            f"| {label} | {lv['support_bottom_channel']} | {lv['mid_50']} | {lv['top_sr']} | "
            f"{lv['previous_highs']} | {lv['ath']} | {lv['current_candle']} |"
        )
    return "\n".join(lines)
