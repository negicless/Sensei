# -*- coding: utf-8 -*-
# bot/main.py

# --- Headless matplotlib (must be before pyplot anywhere) ---
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "./.mplcache")
os.makedirs("./.mplcache", exist_ok=True)

import re
import asyncio
import logging
import datetime as dt
from pathlib import Path

import pandas as pd
import pytz
from telegram import Update, InputFile
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from storage import WatchlistDB
from config import TELEGRAM_BOT_TOKEN, TZ

# ---- Sensei agent APIs ----
from agent.app import run_for_code, build_ideas_for_code
from agent.fastchart import render_chart_fast  # chart-only path (internally uses resolve())
from agent.ingest.tickers import resolve       # single source of truth

# Levels module (updated API)
from agent.signal.levels import (
    compute_levels, compute_levels_sheet,
    render_levels_table_img, render_levels_sheet_img,
    as_markdown_table,
)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sensei-bot")

# ---------- Globals ----------
JST = pytz.timezone(TZ or "Asia/Tokyo")
DB = WatchlistDB("data/bot.db")

# ---------- Paths ----------
LEVELS_OUTDIR = Path("out/levels")
LEVELS_OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
_INTRADAY_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h","2h","4h"}
SPAN_DEFAULT = "1y"  # default when user only says 'day'/'week'

def _looks_intraday_token(tok: str) -> bool:
    t = (tok or "").strip().lower()
    if ":" in t:
        left, _ = t.split(":", 1)
        return left in _INTRADAY_INTERVALS
    return t in _INTRADAY_INTERVALS

def _resolve_horizon(args_tail):
    if not args_tail:
        return SPAN_DEFAULT
    toks = [str(t).strip() for t in args_tail if str(t).strip()]
    if not toks:
        return SPAN_DEFAULT
    if _looks_intraday_token(toks[0]):
        return toks[0]

    span, weekly = None, None

    def is_span(x):
        xl = x.lower()
        return xl in {"6m","1y","2y","5y","10y"} or (
            xl.endswith("w") and xl[:-1] in {"6m","1y","2y","5y","10y"}
        )

    for t in (tok.lower() for tok in toks):
        if t in {"day","d"}:
            weekly = False
        elif t in {"week","w"}:
            weekly = True
        elif is_span(t):
            if t.endswith("w"):
                span = t[:-1]; weekly = True
            else:
                span = t
    span = span or SPAN_DEFAULT
    if weekly is True:  return f"{span}W"
    if weekly is False: return span
    return span

def _drop_last_partial_bar(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    last = pd.to_datetime(df["date"].iloc[-1])
    if last.date() == pd.Timestamp.utcnow().date():
        return df.iloc[:-1] if len(df) > 1 else df
    return df

def _label_for(market: str, yahoo_symbol: str) -> str:
    """Suffix-free label for UI/filenames."""
    if yahoo_symbol.startswith("^"):
        return "NIKKEI225" if yahoo_symbol.upper() == "^N225" else yahoo_symbol.lstrip("^")
    if market == "JP":
        return yahoo_symbol.split(".", 1)[0]   # '7013.T' -> '7013', '247A.T' -> '247A'
    return yahoo_symbol                        # 'FUBO', 'BRK-B', etc.

# ==========================================================
# /start & /ping
# ==========================================================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Sensei is online.\n"
        "Commands:\n"
        "/report 7013 [6m|1y|2y|5y|10y|2yW|day|week|5m:1d|15m:5d|1h:10d]\n"
        "/chart  7013 [same syntax]\n"
        "/idea   7013 [horizon]\n"
        "/levels 7013\n"
        "/watchlist add 7013 | /watchlist show | /watchlist rm 7013\n"
        "/scan"
    )

async def ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        await update.message.reply_text(f"Got it: {update.message.text[:60]}")

# ==========================================================
# /report
# ==========================================================
async def report_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text(
            "Usage: /report 7013 [6m|1y|2y|5y|10y|2yW|day|week|5m:1d|15m:5d|1h:10d]"
        )
    raw = ctx.args[0]
    horizon = _resolve_horizon(ctx.args[1:]) if len(ctx.args) > 1 else SPAN_DEFAULT
    try:
        market, symbol = resolve(raw)
    except ValueError as e:
        return await update.message.reply_text(str(e))
    label = _label_for(market, symbol)

    logger.info("Report requested for %s (%s) by %s", label, horizon, update.effective_user.id)
    status_msg = await update.message.reply_text(
        f"🟢 Task received: building report for **{label}** ({horizon}) …",
        parse_mode="Markdown",
    )

    try:
        loop = asyncio.get_running_loop()
        pdf_path, chart_path = await loop.run_in_executor(None, lambda: run_for_code(raw, horizon))
    except Exception as e:
        logger.exception("Report failed for %s", label)
        return await status_msg.edit_text(f"⚠️ {label}: failed to build report.\nDetails: {e!r}")

    await status_msg.edit_text(f"✅ {label}: report ready. Sending files…")
    ts = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")

    try:
        if pdf_path and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1024:
            with open(pdf_path, "rb") as f:
                await update.message.reply_document(
                    InputFile(f, filename=f"{label}_report.pdf"),
                    caption=f"{label} report (JST {ts})",
                )
        else:
            await update.message.reply_text("PDF looked empty; skipped sending.")
    except Exception as e:
        await update.message.reply_text(f"PDF send failed: {e!r}")

    if chart_path and os.path.exists(chart_path) and os.path.getsize(chart_path) > 1024:
        try:
            with open(chart_path, "rb") as f:
                await update.message.reply_photo(f)
        except BadRequest:
            try:
                await update.message.reply_document(
                    InputFile(chart_path, filename=os.path.basename(chart_path)),
                    caption="(Sent as file due to Telegram image processing)",
                )
            except Exception as e2:
                await update.message.reply_text(f"Chart send failed: {e2!r}")
        except Exception as e:
            await update.message.reply_text(f"Chart send failed: {e!r}")

# ==========================================================
# /idea
# ==========================================================
async def idea_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /idea 7013 [horizon]")
    raw = ctx.args[0]
    horizon = _resolve_horizon(ctx.args[1:]) if len(ctx.args) > 1 else SPAN_DEFAULT
    try:
        market, symbol = resolve(raw)
    except ValueError as e:
        return await update.message.reply_text(str(e))
    label = _label_for(market, symbol)

    logger.info("Ideas requested for %s (%s) by %s", label, horizon, update.effective_user.id)
    try:
        loop = asyncio.get_running_loop()
        ideas = await loop.run_in_executor(None, lambda: build_ideas_for_code(raw, horizon))
    except Exception as e:
        logger.exception("Idea pipeline failed for %s", label)
        return await update.message.reply_text(f"ERROR building ideas for {label}: {e!r}")

    if not ideas:
        return await update.message.reply_text("No A-grade setup today.")

    lines = [f"**{label} Trade Ideas** ({horizon})"]
    for sig in ideas:
        r_parts = []
        for i, row in enumerate(sig.get("r_table", []), start=1):
            r_parts.append(f"T{i}:{row.get('R','?')}R")
        targets = sig.get("targets", [])
        targets_str = ", ".join([f"{t:.2f}" for t in targets]) if targets else "—"

        entry = sig.get("entry")
        stop = sig.get("stop")
        entry_stop = (
            f"Entry: {entry:.2f}, SL: {stop:.2f}"
            if isinstance(entry, (int, float)) and isinstance(stop, (int, float))
            else f"Entry: {entry} SL: {stop}"
        )

        lines += [
            f"• {sig.get('name','Idea')}",
            f"  {entry_stop}",
            f"  Targets: {targets_str}",
            f"  R: {', '.join(r_parts) if r_parts else '—'}",
        ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

# ==========================================================
# /chart  (fast path)
# ==========================================================
async def chart_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text(
            "Usage: /chart 7013 [6m|1y|2y|5y|10y|2yW|day|week|5m:1d|15m:5d|1h:10d]"
        )
    raw = ctx.args[0]
    horizon = _resolve_horizon(ctx.args[1:]) if len(ctx.args) > 1 else SPAN_DEFAULT
    try:
        market, symbol = resolve(raw)
    except ValueError as e:
        return await update.message.reply_text(str(e))
    label = _label_for(market, symbol)

    logger.info("Chart requested for %s (%s) by %s", label, horizon, update.effective_user.id)
    status_msg = await update.message.reply_text(
        f"🟢 Generating chart for **{label}** ({horizon}) …",
        parse_mode="Markdown",
    )

    try:
        loop = asyncio.get_running_loop()
        chart_path = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: render_chart_fast(raw, horizon)),
            timeout=30,
        )
    except asyncio.TimeoutError:
        logger.warning("Chart timed out for %s (%s)", label, horizon)
        return await status_msg.edit_text(f"⚠️ {label}: chart timed out. Try again later.")
    except Exception as e:
        logger.exception("Chart build failed for %s", label)
        return await status_msg.edit_text(f"⚠️ {label}: chart build failed.\nDetails: {e!r}")

    ts = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    try:
        if chart_path and os.path.exists(chart_path) and os.path.getsize(chart_path) > 1024:
            try:
                with open(chart_path, "rb") as f:
                    await update.message.reply_photo(
                        f, caption=f"{label} ({horizon.upper()}) — JST {ts}"
                    )
            except BadRequest:
                await update.message.reply_document(
                    InputFile(chart_path, filename=os.path.basename(chart_path)),
                    caption=f"{label} ({horizon.upper()}) — JST {ts}\n(Sent as file due to Telegram image processing)",
                )
            await status_msg.delete()
        else:
            await status_msg.edit_text("Chart looked empty; skipped sending.")
    except Exception as e:
        await status_msg.edit_text(f"Chart send failed: {e!r}")

# ==========================================================
# /levels  (mentor-style table + optional chart)
# ==========================================================
async def levels_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /levels <ticker>  (e.g., /levels 247A)")
    raw = ctx.args[0]
    # Optional preview chart horizon (defaults to 15m:5d)
    horizon = _resolve_horizon(ctx.args[1:]) if len(ctx.args) > 1 else "15m:5d"

    try:
        market, symbol = resolve(raw)  # validate + canonical symbol
    except ValueError as e:
        return await update.message.reply_text(str(e))
    label = _label_for(market, symbol)
    logger.info("Levels requested for %s (%s) by %s", label, horizon, update.effective_user.id)

    try:
        # Prefer dense intraday for best 4H resampling; fallback to EOD
        df = None
        try:
            # private, but fine in-process for now
            from agent.fastchart import _fetch_yf_intraday  # type: ignore
            df = _fetch_yf_intraday(raw, interval="15m", period="60d")
            df = _drop_last_partial_bar(df)
        except Exception:
            # fallback: let fastchart’s EOD cache loader do the heavy lifting
            from agent.fastchart import _get_prices_cached  # type: ignore
            df = _get_prices_cached(raw)

        if df is None or df.empty:
            return await update.message.reply_text(f"No price data for {label}.")

        # Compute + send markdown table
        lv = compute_levels(df)
        md = as_markdown_table(lv)
        await update.message.reply_markdown(md)

        # Render dark table image (auto out path inside module)
        img_path = render_levels_table_img(lv, title=label)
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f:
                await ctx.bot.send_photo(chat_id=update.effective_chat.id, photo=f)

        # Optional preview chart
        chart_path = await asyncio.get_running_loop().run_in_executor(
            None, lambda: render_chart_fast(raw, horizon)
        )
        if chart_path and os.path.exists(chart_path):
            with open(chart_path, "rb") as f:
                await update.message.reply_photo(f)

    except Exception as e:
        logger.exception("Levels build failed for %s", label)
        await update.message.reply_text(f"Levels error for {label}: {e!r}")

# ==========================================================
# /watchlist & /scan
# ==========================================================
async def watchlist_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /watchlist add|rm|show [code]")
    action = ctx.args[0].lower()
    uid = str(update.effective_user.id)

    if action == "show":
        wl = DB.list_codes(uid)
        return await update.message.reply_text("Watchlist: " + ", ".join(wl) if wl else "Empty.")

    if len(ctx.args) < 2:
        return await update.message.reply_text("Provide a code (e.g., 7013 or 247A).")

    code = ctx.args[1].strip().upper()
    if action == "add":
        DB.add_code(uid, code)
        return await update.message.reply_text(f"Added {code}.")
    if action == "rm":
        DB.remove_code(uid, code)
        return await update.message.reply_text(f"Removed {code}.")
    await update.message.reply_text("Unknown action. Use add|rm|show.")

async def scan_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    wl = DB.list_codes(uid)
    if not wl:
        return await update.message.reply_text("Your watchlist is empty.")

    await update.message.reply_text(f"Scanning {len(wl)} codes …")
    hits = []
    loop = asyncio.get_running_loop()
    for code in wl:
        try:
            ideas = await loop.run_in_executor(None, lambda c=code: build_ideas_for_code(c))
            if ideas:
                hits.append((code, ideas[0].get("name", "Idea")))
        except Exception as e:
            logger.warning("Scan failed for %s: %r", code, e)

    msg = "Signals:\n" + "\n".join(f"{c}: {n}" for c, n in hits) if hits else "No A-grade signals."
    await update.message.reply_text(msg)

# ==========================================================
# Error handler
# ==========================================================
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = repr(context.error)
    logger.exception("Unhandled error: %s", err)
    try:
        if isinstance(update, Update) and update.effective_chat:
            await update.effective_chat.send_message(f"Bot ERROR: {err[:1000]}")
    except Exception:
        pass

# ==========================================================
# App entry
# ==========================================================
def main():
    token = TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing. Put it in .env or export the env var.")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("report", report_cmd))
    app.add_handler(CommandHandler("chart", chart_cmd))
    app.add_handler(CommandHandler("idea", idea_cmd))
    app.add_handler(CommandHandler("levels", levels_cmd))
    app.add_handler(CommandHandler("watchlist", watchlist_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ping))
    app.add_error_handler(on_error)
    app.run_polling()

if __name__ == "__main__":
    main()
