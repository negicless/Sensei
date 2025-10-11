# Sensei Research Telegram Bot (JP Stocks)

Pure-code Telegram bot that scrapes Kabutan snapshots, pulls OHLCV via yfinance, computes indicators,
scans setups (compression breakout / trend pullback), and returns an institutional-style PDF + chart.

## Quickstart (Local)

1. **Python**: 3.11 recommended
2. Create env & install:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   playwright install chromium
   ```
3. Set environment:
   ```bash
   cp .env.example .env
   # edit .env to set TELEGRAM_BOT_TOKEN and defaults
   ```
4. Run bot (polling, simplest):
   ```bash
   python -m bot.main
   ```

## Commands
- `/report 7013` — returns PDF + PNG chart
- `/idea 7013` — trade ideas only (entry/SL/targets with R)
- `/watchlist add 7013 | /watchlist rm 7013 | /watchlist show`
- `/scan` — scans your watchlist for signals

## Notes
- Data: Kabutan pages are rendered once via Playwright and cached. Prices via yfinance (7013.T, etc.).
- Be polite with scraping: the code uses jittered sleeps and caching.
- This is for personal research & education. Verify ToS/robots and your jurisdiction’s rules.
