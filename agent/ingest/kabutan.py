import time, random
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = "https://kabutan.jp/stock/?code={code}"

def fetch_kabutan_snapshot(code: str) -> str:
    out = Path(f"data/raw_html/{code}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out.read_text(encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0 (SenseiBot)")
        page.goto(BASE.format(code=code), wait_until="load", timeout=60000)
        time.sleep(1.5 + random.random()*1.0)
        html = page.content()
        browser.close()
    out.write_text(html, encoding="utf-8")
    return html
