import os

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_RISK_PCT = float(os.getenv("DEFAULT_RISK_PCT", "0.8"))
DATA_DIR = os.getenv("DATA_DIR", "data")
TZ = "Asia/Tokyo"
