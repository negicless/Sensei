import os, asyncio, logging, pytz, datetime as dt
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from storage import WatchlistDB
from config import TELEGRAM_BOT_TOKEN, TZ
from agent.app import run_for_code, build_ideas_for_code

logging.basicConfig(level=logging.INFO)
JST = pytz.timezone(TZ)
DB = WatchlistDB("data/bot.db")

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Sensei Research Bot ready.\n"
        "/report 7013 | /idea 7013 | /watchlist add 7013 | /watchlist show | /scan"
    )

async def report_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: 
        return await update.message.reply_text("Usage: /report 7013")
    code = ctx.args[0].zfill(4)
    await update.message.reply_text(f"Building report for {code} …")
    pdf_path, chart_path = await asyncio.get_event_loop().run_in_executor(None, lambda: run_for_code(code))
    ts = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    await update.message.reply_document(InputFile(pdf_path), caption=f"{code} report (JST {ts})")
    if chart_path and os.path.exists(chart_path):
        await update.message.reply_photo(InputFile(chart_path))

async def idea_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /idea 7013")
    code = ctx.args[0].zfill(4)
    ideas = await asyncio.get_event_loop().run_in_executor(None, lambda: build_ideas_for_code(code))
    if not ideas:
        return await update.message.reply_text("No A-grade setup today.")
    lines = [f"**{code} Trade Ideas**"]
    for sig in ideas:
        lines += [
            f"• {sig['name']}",
            f"  Entry: {sig['entry']:.2f}, SL: {sig['stop']:.2f}",
            f"  Targets: {', '.join(f'{t:.2f}' for t in sig['targets'])}",
            f"  R: {', '.join(f"T{i+1}:{r['R']}R" for i,r in enumerate(sig['r_table']))}"
        ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def watchlist_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # /watchlist add 7013 | /watchlist rm 7013 | /watchlist show
    if not ctx.args:
        return await update.message.reply_text("Usage: /watchlist add|rm|show [code]")
    action = ctx.args[0].lower()
    uid = str(update.effective_user.id)
    if action == "show":
        wl = DB.list_codes(uid)
        return await update.message.reply_text("Watchlist: " + ", ".join(wl) if wl else "Empty.")
    if len(ctx.args) < 2:
        return await update.message.reply_text("Provide a 4-digit code.")
    code = ctx.args[1].zfill(4)
    if action == "add":
        DB.add_code(uid, code); return await update.message.reply_text(f"Added {code}.")
    if action == "rm":
        DB.remove_code(uid, code); return await update.message.reply_text(f"Removed {code}.")
    await update.message.reply_text("Unknown action.")

async def scan_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    wl = DB.list_codes(uid)
    if not wl: return await update.message.reply_text("Your watchlist is empty.")
    await update.message.reply_text(f"Scanning {len(wl)} codes …")
    hits = []
    loop = asyncio.get_event_loop()
    for code in wl:
        ideas = await loop.run_in_executor(None, lambda c=code: build_ideas_for_code(c))
        if ideas: hits.append((code, ideas[0]['name']))
    msg = "Signals:\n" + "\n".join(f"{c}: {n}" for c,n in hits) if hits else "No A-grade signals."
    await update.message.reply_text(msg)

def main():
    token = TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in .env or environment")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("report", report_cmd))
    app.add_handler(CommandHandler("idea", idea_cmd))
    app.add_handler(CommandHandler("watchlist", watchlist_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.run_polling()

if __name__ == "__main__":
    main()
