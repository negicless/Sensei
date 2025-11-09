# agent/report/levels_table.py
# Converts mentor-style levels dict into a PNG table (dark institutional theme)

import os
import pandas as pd
import matplotlib.pyplot as plt

def render_levels_table(levels: dict, ticker: str, out_dir="outputs/charts") -> str:
    """
    Render mentor-style levels as PNG image.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ticker}_levels.png")

    # Arrange data
    rows = []
    order = [("W", "WEEKLY"), ("D", "DAILY"), ("4H", "4 HOUR")]
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
    cols = ["TF", "Bottom (Support)", "Mid 50%", "Top (S/R)", "Previous Highs", "ATH", "Current Candle"]
    df = pd.DataFrame(rows, columns=cols)

    # ---- Plot ----
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 1.1 + 0.5*len(df)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Hide axes
    ax.axis("off")

    # Colors
    header_color = "#161b22"
    cell_color = "#0d1117"
    text_color = "#e2e5ec"
    edge_color = "#222830"

    # Build table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="#58a6ff", weight="bold")
        else:
            cell.set_facecolor(cell_color)
            cell.set_text_props(color=text_color)

    ax.set_title(f"${ticker} — Institutional Levels", color="#58a6ff", fontsize=12, pad=10)
    plt.tight_layout()
    fig.savefig(path, dpi=180, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)
    return path
