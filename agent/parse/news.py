from bs4 import BeautifulSoup
from datetime import datetime

def parse_news_items(html: str):
    soup = BeautifulSoup(html, "lxml")
    items = []
    # very generic: look for links in news sections; adapt selectors as you refine
    for a in soup.select("a"):
        t = (a.get_text() or "").strip()
        href = a.get("href") or ""
        if t and "news" in href:
            items.append({"date": "", "title": t, "link": href})
    # Trim to top 5
    return items[:5]
