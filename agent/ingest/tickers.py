import re, json
from pathlib import Path

# Strip common exchange suffixes from user input (we'll apply our own later)
_SUFFIX_RE = re.compile(
    r"\.(US|T|JP|SS|SZ|HK|KS|KQ|AX|TO|V|OL|PA|L|SI|NZ|SA|MX|F|DE|SW|MI|VI|SG)$",
    re.I,
)

ALIAS_FILES = [
    Path("aliases.json"),                       # repo root (easy to edit)
    Path("agent/ingest/aliases.json"),          # alt location
]

def _load_aliases() -> dict:
    for p in ALIAS_FILES:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}

ALIASES = _load_aliases()

# Yahoo class-share quirks
SPECIAL_US = {"BRK.B": "BRK-B", "BF.B": "BF-B"}

# --- JP formats ---
# 1) 4 digits
_JP_4D_RE = re.compile(r"^\d{4}$")
# 2) 3 digits + 1 letter (e.g., 247A)
_JP_3D1L_RE = re.compile(r"^(\d{3})([A-Z])$")

# US symbol (letters, digits, optional dots, up to 10 total)
_US_RE = re.compile(r"^[A-Z][A-Z0-9\.-]{0,9}$")

def normalize(raw: str) -> str:
    """
    Normalize a user-supplied ticker to a clean, suffix-free, uppercased token.
    Examples:
      "7013.T" -> "7013"
      "247a.jp" -> "247A"
      "$fubo" -> "FUBO"
    """
    s = str(raw).strip().upper().lstrip("$")
    s = _SUFFIX_RE.sub("", s)
    return s

def resolve(raw: str) -> tuple[str, str]:
    """
    Returns (market, yahoo_symbol)
      JP 4-digit        : '####.T'
      JP 3D+1L (e.g., 247A) : '247A.T'
      US equities       : 'SYMBOL' (with BRK.B -> BRK-B, BF.B -> BF-B fixes)
    Aliases (optional) can remap any input to another token (e.g., "7203" -> "TOYOF").
    """
    s = normalize(raw)

    # 1) Aliases first (lets you map, override, or redirect)
    if s in ALIASES:
        return resolve(ALIASES[s])

    # 2) JP: standard 4-digit equities
    if _JP_4D_RE.fullmatch(s):
        return ("JP", f"{s}.T")

    # 3) JP: 3 digits + 1 letter (e.g., 247A) — supported natively
    m = _JP_3D1L_RE.fullmatch(s)
    if m:
        code = f"{m.group(1)}{m.group(2)}"  # already uppercased by normalize()
        return ("JP", f"{code}.T")

    # 4) US: letters/dots/digits (class shares etc.)
    if _US_RE.fullmatch(s):
        return ("US", SPECIAL_US.get(s, s))

    # 5) Otherwise invalid
    raise ValueError(
        f"Unsupported ticker '{raw}'. "
        "Use JP 4-digit (e.g., 7013), JP 3-digit+1 letter (e.g., 247A), or a US symbol (e.g., FUBO)."
    )
