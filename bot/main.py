# -*- coding: utf-8 -*-
# bot/main.py

# --- Make matplotlib cache writable & headless (must be before pyplot anywhere) ---
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "./.mplcache")
os.makedirs("./.mplcache", exist_ok=True)

import asyncio
import logging
import datetime as dt
import pytz

from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.error import BadRequest

from storage import WatchlistDB
from config import TELEGRAM_BOT_TOKEN, TZ

# --- Sensei agent APIs ---
from agent.app import run_for_code, build_ideas_for_code
from agent.fastchart import render_chart_fast  # fast chart-only path

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sensei-bot")

# ---------- Globals ----------
JST = pytz.timezone(TZ or "Asia/Tokyo")
DB = WatchlistDB("data/bot.db")

# ---------- Horizon resolver (day/week or spans) ----------
SPAN_DEFAULT = "1y"  # default when user only says 'day'/'week'


def _resolve_horizon(args_tail):
    if not args_tail:
        return SPAN_DEFAULT
    toks = [str(t).strip().lower() for t in args_tail if str(t).strip()]
    span = None
    weekly = None

    def is_span(x):
        return x in {"6m", "1y", "2y", "5y", "10y"} or (
            x.endswith("w") and x[:-1] in {"6m", "1y", "2y", "5y", "10y"}
        )

    for t in toks:
        if t in {"day", "d"}:
            weekly = False
        elif t in {"week", "w"}:
            weekly = True
        elif is_span(t):
            if t.endswith("w"):
                span = t[:-1]
                weekly = True
            else:
                span = t
    span = span or SPAN_DEFAULT
    if weekly is True:
        return f"{span}W"
    if weekly is False:
        return span
    return span


# ---------- Error handler ----------
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = repr(context.error)
    logger.exception("Unhandled error: %s", err)
    try:
        if isinstance(update, Update) and update.effective_chat:
            await update.effective_chat.send_message(f"Bot ERROR: {err[:1000]}")
    except Exception:
        pass


# ---------- Basics ----------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Sensei is online.\n"
        "Commands:\n"
        "/report 7013 [6m|1y|2y|5y|10y|2yW|day|week]\n"
        "/chart  7013 [same syntax]\n"
        "/idea   7013\n"
        "/watchlist add 7013 | /watchlist show | /watchlist rm 7013\n"
        "/scan"
    )


async def ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        await update.message.reply_text(f"Got it: {update.message.text[:60]}")


# ---------- Commands ----------
async def report_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text(
            "Usage: /report 7013 [6m|1y|2y|5y|10y|2yW|day|week]"
        )
    code = ctx.args[0].zfill(4)
    horizon = _resolve_horizon(ctx.args[1:]) if len(ctx.args) > 1 else SPAN_DEFAULT
    logger.info(
        "Report requested for %s (%s) by %s", code, horizon, update.effective_user.id
    )

    status_msg = await update.message.reply_text(
        f"🟢 Task received: building report for **{code}** ({horizon}) …",
        parse_mode="Markdown",
    )

    try:
        loop = asyncio.get_running_loop()
        pdf_path, chart_path = await loop.run_in_executor(
            None, lambda: run_for_code(code, horizon)
        )
    except Exception as e:
        logger.exception("Report failed for %s", code)
        return await status_msg.edit_text(
            f"⚠️ {code}: failed to build report.\nDetails: {e!r}"
        )

    await status_msg.edit_text(f"✅ {code}: report ready. Sending files…")
    ts = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    try:
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1024:
            with open(pdf_path, "rb") as f:
                await update.message.reply_document(
                    InputFile(f, filename=f"{code}_report.pdf"),
                    caption=f"{code} report (JST {ts})",
                )
        else:
            await update.message.reply_text(
                "PDF looked empty; skipped sending the file."
            )
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


# ---------- Ideas (setup scan only) ----------
async def idea_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # Usage: /idea 7013 [6m|1y|2y|5y|10y|2yW|day|week]
    if not ctx.args:
        return await update.message.reply_text("Usage: /idea 7013 [horizon]")
    code = ctx.args[0].zfill(4)
    horizon = _resolve_horizon(ctx.args[1:]) if len(ctx.args) > 1 else SPAN_DEFAULT
    logger.info(
        "Ideas requested for %s (%s) by %s", code, horizon, update.effective_user.id
    )

    try:
        loop = asyncio.get_running_loop()
        ideas = await loop.run_in_executor(
            None, lambda: build_ideas_for_code(code, horizon)
        )
    except Exception as e:
        logger.exception("Idea pipeline failed for %s", code)
        return await update.message.reply_text(
            f"ERROR building ideas for {code}: {e!r}"
        )

    if not ideas:
        return await update.message.reply_text("No A-grade setup today.")

    # Compact text rendering
    lines = [f"**{code} Trade Ideas** ({horizon})"]
    for sig in ideas:
        r_parts = []
        for i, row in enumerate(sig.get("r_table", []), start=1):
            r_parts.append(f"T{i}:{row.get('R','?')}R")
        targets = sig.get("targets", [])
        targets_str = ", ".join([f"{t:.2f}" for t in targets]) if targets else "—"

        entry = sig.get("entry")
        stop = sig.get("stop")
        if isinstance(entry, (int, float)) and isinstance(stop, (int, float)):
            entry_stop = f"Entry: {entry:.2f}, SL: {stop:.2f}"
        else:
            entry_stop = f"Entry: {entry} SL: {stop}"

        lines += [
            f"• {sig.get('name','Idea')}",
            f"  {entry_stop}",
            f"  Targets: {targets_str}",
            f"  R: {', '.join(r_parts) if r_parts else '—'}",
        ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# --- chart-only (fast path) ---
async def chart_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text(
            "Usage: /chart 7013 [6m|1y|2y|5y|10y|2yW|day|week]"
        )
    code = ctx.args[0].zfill(4)
    horizon = _resolve_horizon(ctx.args[1:]) if len(ctx.args) > 1 else SPAN_DEFAULT
    logger.info(
        "Chart requested for %s (%s) by %s", code, horizon, update.effective_user.id
    )

    status_msg = await update.message.reply_text(
        f"🟢 Generating chart for **{code}** ({horizon}) …",
        parse_mode="Markdown",
    )

    try:
        loop = asyncio.get_running_loop()
        chart_path = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: render_chart_fast(code, horizon)),
            timeout=20,  # end-to-end cap from bot layer
        )
    except asyncio.TimeoutError:
        logger.warning("Chart timed out for %s (%s)", code, horizon)
        return await status_msg.edit_text(
            f"⚠️ {code}: chart timed out. Try again later."
        )
    except Exception as e:
        logger.exception("Chart build failed for %s", code)
        return await status_msg.edit_text(
            f"⚠️ {code}: chart build failed.\nDetails: {e!r}"
        )

    ts = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    try:
        if (
            chart_path
            and os.path.exists(chart_path)
            and os.path.getsize(chart_path) > 1024
        ):
            try:
                with open(chart_path, "rb") as f:
                    await update.message.reply_photo(
                        f, caption=f"{code} ({horizon.upper()}) — JST {ts}"
                    )
            except BadRequest:
                await update.message.reply_document(
                    InputFile(chart_path, filename=os.path.basename(chart_path)),
                    caption=f"{code} ({horizon.upper()}) — JST {ts}\n(Sent as file due to Telegram image processing)",
                )
            await status_msg.delete()
        else:
            await status_msg.edit_text("Chart looked empty; skipped sending the file.")
    except Exception as e:
        await status_msg.edit_text(f"Chart send failed: {e!r}")


async def watchlist_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /watchlist add|rm|show [code]")
    action = ctx.args[0].lower()
    uid = str(update.effective_user.id)

    if action == "show":
        wl = DB.list_codes(uid)
        return await update.message.reply_text(
            "Watchlist: " + ", ".join(wl) if wl else "Empty."
        )

    if len(ctx.args) < 2:
        return await update.message.reply_text("Provide a 4-digit code (e.g., 7013).")

    code = ctx.args[1].zfill(4)
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
            ideas = await loop.run_in_executor(
                None, lambda c=code: build_ideas_for_code(c)
            )
            if ideas:
                hits.append((code, ideas[0].get("name", "Idea")))
        except Exception as e:
            logger.warning("Scan failed for %s: %r", code, e)

    msg = (
        "Signals:\n" + "\n".join(f"{c}: {n}" for c, n in hits)
        if hits
        else "No A-grade signals."
    )
    await update.message.reply_text(msg)


# ---------- App entry ----------
def main():
    token = TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN missing. Put it in .env or export the env var."
        )

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("report", report_cmd))
    app.add_handler(CommandHandler("chart", chart_cmd))
    app.add_handler(CommandHandler("idea", idea_cmd))
    app.add_handler(CommandHandler("watchlist", watchlist_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ping))
    app.add_error_handler(on_error)

    app.run_polling()


if __name__ == "__main__":
    main()
