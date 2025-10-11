import re
from bs4 import BeautifulSoup

LABELS = {
    "per": r"PER[^0-9]*([\d\.]+)",
    "pbr": r"PBR[^0-9]*([\d\.]+)",
    "yield": r"配当.*?([\d\.]+)\%",
    "equity_ratio": r"自己資本比率[^0-9]*([\d\.]+)\%",
    "mktcap": r"時価総額[^0-9]*([\d,]+)",
    "credit_ratio": r"信用倍率[^0-9]*([\d\.]+)",
}

def parse_fundamentals(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    out = {}
    for k, pat in LABELS.items():
        m = re.search(pat, text)
        if m:
            out[k] = float(m.group(1).replace(",", ""))
    return out
