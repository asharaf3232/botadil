# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🤖 الملف الرئيسي (telegram_bot.py) | النسخة الكاملة والنهائية 🤖 ---
# =======================================================================================

import logging
import json
import os
from datetime import datetime

# --- استيراد المكتبات الأساسية ---
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest

# --- استيراد الوحدات المخصصة للمشروع ---
from config import *
from database import init_database, save_settings, load_settings, get_active_trades_from_db
from exchanges import bot_state, initialize_exchanges, calculate_full_portfolio
from core_logic import perform_scan, track_open_trades

# --- إعداد مسجل الأحداث (Logger) ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'a', 'utf-8'), logging.StreamHandler()])
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger("MinesweeperBot_v6")

# =======================================================================================
# --- Communications ---
# =======================================================================================

async def send_telegram_message(bot, signal_data, is_new=False, is_opportunity=False, update_type=None, edit_message_id=None, return_message_object=False):
    message, keyboard, target_chat = "", None, TELEGRAM_CHAT_ID
    def format_price(price): 
        if price is None: return "N/A"
        return f"{price:,.8f}" if price < 0.01 else f"{price:,.4f}"

    if 'custom_message' in signal_data:
        message, target_chat = signal_data['custom_message'], signal_data.get('target_chat', TELEGRAM_CHAT_ID)
        if 'keyboard' in signal_data: keyboard = signal_data['keyboard']
    elif is_new:
        target_chat = TELEGRAM_SIGNAL_CHANNEL_ID
        trade_type_title = "🚨 صفقة حقيقية 🚨" if signal_data.get('is_real_trade') else "✅ توصية شراء جديدة"
        title = f"**{trade_type_title} | {signal_data['symbol']}**"
        entry, tp, sl = signal_data['entry_price'], signal_data['take_profit'], signal_data['stop_loss']
        tp_percent = ((tp - entry) / entry * 100) if entry else 0
        sl_percent = ((entry - sl) / entry * 100) if entry else 0
        id_line = f"\n*للمتابعة: /check {signal_data.get('trade_id', 'N/A')}*"
        message = (f"{title}\n\n"
                   f"🔹 **المنصة:** {signal_data['exchange']}\n"
                   f"📈 **الدخول:** `{format_price(entry)}`\n"
                   f"🎯 **الهدف:** `{format_price(tp)}` (+{tp_percent:.2f}%)\n"
                   f"🛑 **الوقف:** `{format_price(sl)}` (-{sl_percent:.2f}%)"
                   f"{id_line}")
    elif update_type == 'tsl_activation':
        message = (f"**🚀 تأمين الأرباح! | #{signal_data['id']} {signal_data['symbol']}**\n\n"
                   f"تم رفع وقف الخسارة إلى نقطة الدخول. الصفقة الآن بدون مخاطرة!")

    if not message: return
    try:
        if edit_message_id:
            sent_message = await bot.edit_message_text(chat_id=target_chat, message_id=edit_message_id, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
        else:
            sent_message = await bot.send_message(chat_id=target_chat, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
        if return_message_object: return sent_message
    except Exception as e:
        logger.error(f"Telegram send/edit error: {e}")

# =======================================================================================
# --- Telegram UI Handlers ---
# =======================================================================================

main_menu_keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"], ["ℹ️ مساعدة"]]

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("💣 أهلاً بك في بوت **كاسحة الألغام**!", reply_markup=ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True), parse_mode=ParseMode.MARKDOWN)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("استخدم الأزرار للتنقل.")

async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active_trades = get_active_trades_from_db()
    real_trades = [t for t in active_trades if t['trade_mode'] == 'real']
    virtual_trades = [t for t in active_trades if t['trade_mode'] == 'virtual']

    real_portfolio = await calculate_full_portfolio(bot_state.exchanges.get('kucoin')) # Change to your primary exchange if needed
    
    dashboard_text = (
        f"🖥️ **لوحة التحكم** 🖥️\n\n"
        f"**الصفقات النشطة:** {len(active_trades)} (حقيقي: {len(real_trades)}, وهمي: {len(virtual_trades)})\n"
        f"**رصيد المحفظة الحقيقية (تقريبي):** ${real_portfolio.get('total_usdt', 0):.2f}\n"
        f"**رصيد المحفظة الوهمية:** ${bot_state.settings.get('virtual_portfolio_balance_usdt', 0):.2f}"
    )
    await update.message.reply_text(dashboard_text, parse_mode=ParseMode.MARKDOWN)

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("هذه هي قائمة الإعدادات (سيتم تنفيذها لاحقاً).")

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text == "Dashboard 🖥️":
        await dashboard_command(update, context)
    elif text == "⚙️ الإعدادات":
        await settings_command(update, context)
    elif text == "ℹ️ مساعدة":
        await help_command(update, context)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

# =======================================================================================
# --- Bot Startup and Main Loop ---
# =======================================================================================

async def post_init(application: Application):
    logger.info("Post-init: Initializing exchanges...")
    await initialize_exchanges()
    if not bot_state.public_exchanges:
        logger.critical("CRITICAL: No public exchange clients connected.")
        return

    job_queue = application.job_queue
    job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name='perform_scan')
    job_queue.run_repeating(track_open_trades, interval=TRACK_INTERVAL_SECONDS, first=20, name='track_open_trades')
    
    logger.info("Jobs scheduled successfully.")
    await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🚀 *بوت كاسحة الألغام (v6.6) جاهز للعمل!*", parse_mode=ParseMode.MARKDOWN)

async def post_shutdown(application: Application):
    all_exchanges = list(bot_state.exchanges.values()) + list(bot_state.public_exchanges.values())
    unique_exchanges = list({id(ex): ex for ex in all_exchanges}.values())
    await asyncio.gather(*[ex.close() for ex in unique_exchanges])
    logger.info("All exchange connections closed gracefully.")

def main():
    if not TELEGRAM_BOT_TOKEN or 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN:
        print("FATAL ERROR: TELEGRAM_BOT_TOKEN is not set.")
        exit()

    load_settings()
    init_database()

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    application.add_error_handler(error_handler)
    
    logger.info("Application configured. Starting polling...")
    application.run_polling()

if __name__ == '__main__':
    print("🚀 Starting Mineseper Bot v6.6...")
    main()
