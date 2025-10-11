from agent.ingest.kabutan import fetch_kabutan_snapshot
from agent.ingest.price_yahoojp import fetch_price_history
from agent.parse.fundamentals import parse_fundamentals
from agent.parse.news import parse_news_items
from agent.signal.tech import add_indicators, find_levels
from agent.signal.setups import scan_setups
from agent.report.build_markdown import render_md
from agent.report.to_pdf import md_to_pdf
from pathlib import Path

def run_for_code(code: str):
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    Path("outputs/charts").mkdir(parents=True, exist_ok=True)
    html = fetch_kabutan_snapshot(code)
    funda = parse_fundamentals(html)
    news = parse_news_items(html)

    px = fetch_price_history(code)
    px = add_indicators(px)
    levels = find_levels(px)

    md = render_md(code, funda, news, px, levels, scan_setups(px, funda))
    md_path = f"outputs/reports/{code}.md"
    pdf_path = f"outputs/reports/{code}.pdf"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    md_to_pdf(md_path, pdf_path)

    # Save a simple chart PNG (mplfinance can be added later)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        px.tail(120).set_index("date")["c"].plot()
        plt.title(f"{code} Close (last 120)")
        chart_path = f"outputs/charts/{code}_daily.png"
        plt.savefig(chart_path, dpi=140, bbox_inches="tight")
        plt.close()
    except Exception:
        chart_path = None

    return pdf_path, chart_path

def build_ideas_for_code(code: str):
    html = fetch_kabutan_snapshot(code)
    funda = parse_fundamentals(html)
    px = fetch_price_history(code)
    px = add_indicators(px)
    return scan_setups(px, funda)
