# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 OKX Mastermind Trader v27.5 (Rescue Edition) 🚀 ---
# =======================================================================================
# هذا إصدار خاص ومؤقت. سيقوم تلقائياً بنسخ الصفقات من قاعدة البيانات القديمة
# (v26.db) إلى الجديدة (v27.db) عند بدء التشغيل لأول مرة.
# بعد التأكد من عودة صفقاتك، يرجى التحديث إلى الإصدار النظيف.
# =======================================================================================

# --- Core Libraries ---
import asyncio
import os
import logging
import json
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hmac
import base64
from collections import defaultdict
import copy
import sqlite3 # Required for rescue operation

# --- Database & Networking ---
import aiosqlite
import websockets
import websockets.exceptions
import httpx
import feedparser

# --- Data Analysis & CCXT ---
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt

# --- Optional NLP Library ---
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not found. News sentiment analysis will be disabled.")


# --- Telegram & Environment ---
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest, TimedOut, Forbidden
from dotenv import load_dotenv

# =======================================================================================
# --- ⚙️ Core Configuration ⚙️ ---
# =======================================================================================
load_dotenv()

OKX_API_KEY = os.getenv('OKX_API_KEY')
OKX_API_SECRET = os.getenv('OKX_API_SECRET')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') 

TIMEFRAME = '15m'
SCAN_INTERVAL_SECONDS = 900
SUPERVISOR_INTERVAL_SECONDS = 120

APP_ROOT = '.'
OLD_DB_FILE = os.path.join(APP_ROOT, 'mastermind_trader_v26.db')
DB_FILE = os.path.join(APP_ROOT, 'mastermind_trader_v27.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'mastermind_trader_settings_v27.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Mastermind_Trader")

# =======================================================================================
# --- 🆘 Rescue Logic 🆘 ---
# =======================================================================================
def run_rescue_operation():
    """
    ينسخ الصفقات من قاعدة بيانات قديمة إلى جديدة. يعمل مرة واحدة فقط.
    """
    RESCUE_FLAG_FILE = os.path.join(APP_ROOT, '.rescue_done')
    if os.path.exists(RESCUE_FLAG_FILE):
        logger.info("Rescue operation already completed. Skipping.")
        return

    logger.info("--- Starting Trade Rescue Operation ---")
    
    if not os.path.exists(OLD_DB_FILE):
        logger.warning(f"Old database '{OLD_DB_FILE}' not found. No trades to rescue.")
        open(RESCUE_FLAG_FILE, 'w').close()
        return

    if not os.path.exists(DB_FILE):
        logger.error(f"New database '{DB_FILE}' not found. Cannot perform rescue.")
        return

    try:
        conn_old = sqlite3.connect(OLD_DB_FILE)
        conn_new = sqlite3.connect(DB_FILE)
        cursor_old = conn_old.cursor()
        cursor_new = conn_new.cursor()

        cursor_old.execute("SELECT * FROM trades WHERE status = 'active' OR status = 'pending'")
        trades_to_rescue = cursor_old.fetchall()

        if not trades_to_rescue:
            logger.info("No active or pending trades found in old database.")
            open(RESCUE_FLAG_FILE, 'w').close()
            return

        logger.info(f"Found {len(trades_to_rescue)} trades to rescue.")

        column_names = [description[0] for description in cursor_old.description]
        cursor_new.execute("PRAGMA table_info(trades)")
        new_columns = [row[1] for row in cursor_new.fetchall()]
        
        shared_columns = [col for col in column_names if col in new_columns]
        placeholders = ', '.join(['?'] * len(shared_columns))
        columns_str = ', '.join(shared_columns)
        insert_query = f"INSERT OR IGNORE INTO trades ({columns_str}) VALUES ({placeholders})"
        
        rescued_count = 0
        for trade in trades_to_rescue:
            trade_dict = dict(zip(column_names, trade))
            values_to_insert = tuple(trade_dict[col] for col in shared_columns)
            cursor_new.execute(insert_query, values_to_insert)
            rescued_count += 1

        conn_new.commit()
        conn_old.close()
        conn_new.close()

        logger.info(f"Successfully rescued {rescued_count} trades.")
        open(RESCUE_FLAG_FILE, 'w').close() # Create flag file to prevent re-running
        return f"✅ **عملية الإنقاذ اكتملت!**\nتم استعادة `{rescued_count}` صفقة بنجاح. يرجى الآن التحديث إلى الإصدار النظيف."

    except Exception as e:
        logger.critical(f"CRITICAL ERROR during rescue operation: {e}")
        return f"🚨 **فشل حرج أثناء عملية الإنقاذ!**\nالخطأ: `{e}`. يرجى مراجعة السجلات."


# =======================================================================================
# --- Global Bot State & Locks ---
# =======================================================================================
class BotState:
    def __init__(self):
        self.settings = {}
        self.active_preset_name = "مخصص"
        self.last_signal_time = {}
        self.application = None
        self.exchange = None
        self.market_mood = {"mood": "UNKNOWN", "reason": "تحليل لم يتم بعد"}
        self.private_ws = None
        self.public_ws = None
        self.trade_guardian = None
        self.last_scan_info = {}
        self.all_markets = []
        self.last_markets_fetch = 0

bot_data = BotState()
scan_lock = asyncio.Lock()
trade_management_lock = asyncio.Lock()

# ... (باقي الكود من إصدار v27.1 بدون تغيير) ...
# The rest of the v27.1 code is identical. For brevity, it is not repeated here.
# Please copy the rest of the code from the file you provided.
# The only changes are in the configuration section and the addition of the rescue logic.
# The main function should call the rescue logic.

# ... (Copy the entire code from binance_trader (36).py from here down) ...
# Make sure to modify the `main` function as shown below.

def main():
    logger.info("--- Starting OKX Mastermind Trader v27.5 (Rescue Edition) ---")
    
    # Initialize database first to ensure the file exists
    asyncio.run(init_database())
    
    # Run the rescue operation synchronously before starting the bot
    rescue_message = run_rescue_operation()
    
    load_settings()
    
    app_builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    # Pass the rescue message to post_init
    app_builder.post_init(lambda app: post_init(app, rescue_message)).post_shutdown(post_shutdown)
    application = app_builder.build()
    
    # ... (the rest of the main function is the same)
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("scan", manual_scan_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    
    application.run_polling()

async def post_init(application: Application, rescue_message: str = None):
    bot_data.application = application
    # ... (rest of post_init is the same)

    # Send the rescue message if it exists
    if rescue_message:
        await application.bot.send_message(TELEGRAM_CHAT_ID, rescue_message, parse_mode=ParseMode.MARKDOWN)

    try:
        await application.bot.send_message(TELEGRAM_CHAT_ID, "*🚀 OKX Mastermind Trader v27.5 (إصدار إنقاذ) بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)
    except Forbidden:
        logger.critical(f"FATAL: Bot is not authorized for chat ID {TELEGRAM_CHAT_ID}.")
        return
    logger.info("--- Bot is now fully operational ---")

# (You need to fill in the rest of the code from your file)
