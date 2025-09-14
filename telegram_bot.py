# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🤖 الملف الرئيسي (telegram_bot.py) | بوت كاسحة الألغام v6.6 🤖 ---
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
from database import init_database, save_settings, load_settings
from exchanges import bot_state, initialize_exchanges, get_total_real_portfolio_value_usdt, get_exchange_adapter, get_real_balance, calculate_full_portfolio
from strategies import SCANNERS
from core_logic import (perform_scan, track_open_trades, check_market_regime, 
                        get_fear_and_greed_index, _reconstruct_and_save_trade,
                        execute_manual_trade)

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

    elif is_new or is_opportunity:
        target_chat = TELEGRAM_SIGNAL_CHANNEL_ID
        strength_stars = '⭐' * signal_data.get('strength', 1)
        trade_type_title = "🚨 صفقة حقيقية 🚨" if signal_data.get('is_real_trade') else "✅ توصية شراء جديدة"
        title = f"**{trade_type_title} | {signal_data['symbol']}**" if is_new else f"**💡 فرصة محتملة | {signal_data['symbol']}**"
        entry, tp, sl = signal_data['entry_price'], signal_data['take_profit'], signal_data['stop_loss']
        tp_percent, sl_percent = ((tp - entry) / entry * 100), ((entry - sl) / entry * 100)
        id_line = f"\n*للمتابعة اضغط: /check {signal_data.get('trade_id', 'N/A')}*" if is_new else ""
        reasons_en = signal_data['reason'].split(' + ')
        reasons_ar = ' + '.join([STRATEGY_NAMES_AR.get(r, r) for r in reasons_en])
        message = (f"**Signal Alert | تنبيه إشارة**\n"
                   f"------------------------------------\n"
                   f"{title}\n"
                   f"------------------------------------\n"
                   f"🔹 **المنصة:** {signal_data['exchange']}\n⭐ **قوة الإشارة:** {strength_stars}\n"
                   f"🔍 **الاستراتيجية:** {reasons_ar}\n\n"
                   f"📈 **نقطة الدخول:** `{format_price(entry)}`\n"
                   f"🎯 **الهدف:** `{format_price(tp)}` (+{tp_percent:.2f}%)\n"
                   f"🛑 **الوقف:** `{format_price(sl)}` (-{sl_percent:.2f}%)"
                   f"{id_line}")
    elif update_type == 'tsl_activation':
        message = (f"**🚀 تأمين الأرباح! | #{signal_data['id']} {signal_data['symbol']}**\n\n"
                   f"تم رفع وقف الخسارة إلى نقطة الدخول.\n"
                   f"**هذه الصفقة الآن مؤمَّنة بالكامل وبدون مخاطرة!**\n\n"
                   f"*دع الأرباح تنمو!*")
    elif update_type == 'tsl_update_real':
        message = (f"**🔔 تنبيه تحديث وقف الخسارة (صفقة حقيقية) 🔔**\n\n"
                   f"**صفقة:** `#{signal_data['id']} {signal_data['symbol']}`\n\n"
                   f"وصل السعر إلى `{format_price(signal_data['current_price'])}`.\n"
                   f"**إجراء مقترح:** قم بتعديل أمر وقف الخسارة يدوياً إلى `{format_price(signal_data['new_sl'])}` لتأمين الأرباح.")

    if not message: return
    try:
        if edit_message_id:
            sent_message = await bot.edit_message_text(chat_id=target_chat, message_id=edit_message_id, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
        else:
            sent_message = await bot.send_message(chat_id=target_chat, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
        if return_message_object: return sent_message
    except BadRequest as e:
        if 'Message is not modified' not in str(e): logger.error(f"Telegram BadRequest: {e}")
    except Exception as e:
        logger.error(f"General error in send_telegram_message: {e}")

# =======================================================================================
# --- Telegram UI Handlers ---
# =======================================================================================

main_menu_keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"], ["ℹ️ مساعدة"]]
settings_menu_keyboard = [
    ["🏁 أنماط جاهزة", "🎭 تفعيل/تعطيل الماسحات"], 
    ["🔧 تعديل المعايير", "🚨 التحكم بالتداول الحقيقي"],
    ["🔙 القائمة الرئيسية"]
]

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = "💣 أهلاً بك في بوت **كاسحة الألغام**!\n\n*(الإصدار 6.6 - الهيكل الجديد)*\n\nاختر من القائمة للبدء."
    await update.message.reply_text(welcome_message, reply_markup=ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True), parse_mode=ParseMode.MARKDOWN)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "**💣 أوامر بوت كاسحة الألغام 💣**\n\n"
        "`/start` - لعرض القائمة الرئيسية.\n"
        "`/check <ID>` - لمتابعة صفقة معينة.\n"
        "`/trade` - لبدء تداول يدوي."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

# ... (Include all other Telegram handlers: show_dashboard_command, show_settings_menu, etc.)
# ... (These functions will call the logic from other modules as needed)
# ... For brevity, I will only include the main handler structure and the main() function

async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # This handler now manages all text inputs, including settings changes and menu navigation
    if not update.message or not update.message.text: return
    text = update.message.text
    # ... [Full logic for universal_text_handler]
    
async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # This handler now manages all button presses
    query = update.callback_query
    await query.answer()
    # ... [Full logic for button_callback_handler]

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

# =======================================================================================
# --- Bot Startup and Main Loop ---
# =======================================================================================

async def post_init(application: Application):
    """Function to run after the bot is initialized but before polling starts."""
    if NLTK_AVAILABLE:
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("Downloading NLTK data for sentiment analysis...")
            nltk.download('vader_lexicon')
    
    logger.info("Post-init: Initializing exchanges...")
    await initialize_exchanges()
    if not bot_state.public_exchanges:
        logger.critical("CRITICAL: No public exchange clients connected. Bot cannot run.")
        return

    job_queue = application.job_queue
    job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name='perform_scan')
    job_queue.run_repeating(track_open_trades, interval=TRACK_INTERVAL_SECONDS, first=20, name='track_open_trades')
    
    logger.info("Jobs scheduled successfully.")
    await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🚀 *بوت كاسحة الألغام (v6.6) جاهز للعمل!*", parse_mode=ParseMode.MARKDOWN)

async def post_shutdown(application: Application):
    """Function to run gracefully on bot shutdown."""
    all_exchanges = list(bot_state.exchanges.values()) + list(bot_state.public_exchanges.values())
    unique_exchanges = list({id(ex): ex for ex in all_exchanges}.values())
    await asyncio.gather(*[ex.close() for ex in unique_exchanges])
    logger.info("All exchange connections closed gracefully.")

def main():
    """Sets up and runs the entire bot application."""
    if not TELEGRAM_BOT_TOKEN or 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN:
        print("FATAL ERROR: TELEGRAM_BOT_TOKEN is not set.")
        exit()

    # Load settings and initialize database before starting
    load_settings()
    init_database()

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(60.0)
        .read_timeout(60.0)
        .connection_pool_size(50)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # Register all handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    # ... add other command handlers ...
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    application.add_error_handler(error_handler)
    
    logger.info("Application configured. Starting polling...")
    application.run_polling()

if __name__ == '__main__':
    print("🚀 Starting Mineseper Bot v6.6 (Modular Architecture)...")
    try:
        main()
    except Exception as e:
        logging.critical(f"Bot stopped due to a critical unhandled error: {e}", exc_info=True)
