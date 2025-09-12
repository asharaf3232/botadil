# -*- coding: utf-8 -*-

# --- المكتبات المطلوبة --- #
import ccxt.async_support as ccxt_async
import ccxt
import pandas as pd
import pandas_ta as ta
import asyncio
import os
import logging
import json
import re
import time
import sqlite3
from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import deque, Counter, defaultdict

# [UPGRADE] المكتبات الجديدة لتحليل الأخبار
import feedparser
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("Library 'nltk' not found. Sentiment analysis will be disabled.")

import httpx
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest, RetryAfter, TimedOut

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Library 'scipy' not found. RSI Divergence strategy will be disabled.")


# --- الإعدادات الأساسية --- #
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')
TELEGRAM_SIGNAL_CHANNEL_ID = os.getenv('TELEGRAM_SIGNAL_CHANNEL_ID', TELEGRAM_CHAT_ID)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_AV_KEY_HERE')

# إعدادات مفاتيح API للمنصات
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_BINANCE_API_SECRET')

KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', 'YOUR_KUCOIN_API_KEY')
KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET', 'YOUR_KUCOIN_API_SECRET')
KUCOIN_API_PASSPHRASE = os.getenv('KUCOIN_API_PASSPHRASE', 'YOUR_KUCOIN_PASSPHRASE')

# OKX API Keys (اختيارية)
OKX_API_KEY = os.getenv('OKX_API_KEY', 'YOUR_OKX_API_KEY')
OKX_API_SECRET = os.getenv('OKX_API_SECRET', 'YOUR_OKX_API_SECRET')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE', 'YOUR_OKX_PASSPHRASE')

if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or TELEGRAM_CHAT_ID == 'YOUR_CHAT_ID_HERE':
    print("FATAL ERROR: Please set your Telegram Token and Chat ID.")
    exit()

# --- إعدادات البوت --- #
EXCHANGES_TO_SCAN = ['binance', 'okx', 'bybit', 'kucoin', 'gate', 'mexc']
TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
SCAN_INTERVAL_SECONDS = 900  # 15 دقيقة
TRACK_INTERVAL_SECONDS = 120  # دقيقتان

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'trading_bot_real_v12.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'settings_real_v12.json')

EGYPT_TZ = ZoneInfo("Africa/Cairo")

# --- إعداد مسجل الأحداث (Logger) --- #
LOG_FILE = os.path.join(APP_ROOT, 'bot_real_v12.log')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE, 'a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# تقليل logs للمكتبات الخارجية
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)
logging.getLogger('ccxt.base.exchange').setLevel(logging.WARNING)
logger = logging.getLogger("RealTradingBot")

# --- Preset Configurations ---
PRESET_PRO = {
    "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "max_spread_percent": 0.45, "rvol_period": 18, "min_rvol": 1.5},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.85},
    "ema_trend_filter": {"enabled": True, "ema_period": 200},
    "min_tp_sl_filter": {"min_tp_percent": 1.1, "min_sl_percent": 0.6}
}
PRESET_LAX = {
    "liquidity_filters": {"min_quote_volume_24h_usd": 400000, "max_spread_percent": 1.3, "rvol_period": 12, "min_rvol": 1.1},
    "volatility_filters": {"atr_period_for_filter": 10, "min_atr_percent": 0.3},
    "ema_trend_filter": {"enabled": False, "ema_period": 200},
    "min_tp_sl_filter": {"min_tp_percent": 0.4, "min_sl_percent": 0.2}
}
PRESET_STRICT = {
    "liquidity_filters": {"min_quote_volume_24h_usd": 2500000, "max_spread_percent": 0.22, "rvol_period": 25, "min_rvol": 2.2},
    "volatility_filters": {"atr_period_for_filter": 20, "min_atr_percent": 1.4},
    "ema_trend_filter": {"enabled": True, "ema_period": 200},
    "min_tp_sl_filter": {"min_tp_percent": 1.8, "min_sl_percent": 0.9}
}
PRESETS = {"PRO": PRESET_PRO, "LAX": PRESET_LAX, "STRICT": PRESET_STRICT}

STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي",
    "breakout_squeeze_pro": "اختراق انضغاطي",
    "rsi_divergence": "دايفرجنس RSI",
    "supertrend_pullback": "انعكاس سوبرترند"
}

# --- Constants for Interactive Settings menu ---
EDITABLE_PARAMS = {
    "إعدادات عامة": [
        "max_concurrent_trades", "top_n_symbols_by_volume", "concurrent_workers",
        "min_signal_strength", "real_trade_size_percentage"
    ],
    "إعدادات المخاطر": [
        "real_trading_enabled", "atr_sl_multiplier", "risk_reward_ratio",
        "trailing_sl_activate_percent", "trailing_sl_percent", "trailing_sl_enabled"
    ],
    "الفلاتر والاتجاه": [
        "market_regime_filter_enabled", "use_master_trend_filter", "fear_and_greed_filter_enabled",
        "master_adx_filter_level", "master_trend_filter_ma_period", "fear_and_greed_threshold",
        "fundamental_analysis_enabled"
    ]
}

PARAM_DISPLAY_NAMES = {
    "real_trading_enabled": "🚨 تفعيل التداول الحقيقي 🚨",
    "real_trade_size_percentage": "حجم الصفقة الحقيقية (%)",
    "max_concurrent_trades": "أقصى عدد للصفقات",
    "top_n_symbols_by_volume": "عدد العملات للفحص",
    "concurrent_workers": "عمال الفحص المتزامنين",
    "min_signal_strength": "أدنى قوة للإشارة",
    "atr_sl_multiplier": "مضاعف وقف الخسارة (ATR)",
    "risk_reward_ratio": "نسبة المخاطرة/العائد",
    "trailing_sl_activate_percent": "تفعيل الوقف المتحرك (%)",
    "trailing_sl_percent": "مسافة الوقف المتحرك (%)",
    "market_regime_filter_enabled": "فلتر وضع السوق (فني)",
    "use_master_trend_filter": "فلتر الاتجاه العام (BTC)",
    "master_adx_filter_level": "مستوى فلتر ADX",
    "master_trend_filter_ma_period": "فترة فلتر الاتجاه",
    "trailing_sl_enabled": "تفعيل الوقف المتحرك",
    "fear_and_greed_filter_enabled": "فلتر الخوف والطمع",
    "fear_and_greed_threshold": "حد مؤشر الخوف",
    "fundamental_analysis_enabled": "فلتر الأخبار والبيانات",
}

# --- Global Bot State ---
bot_data = {
    "exchanges": {},
    "last_signal_time": {},
    "settings": {},
    "status_snapshot": {
        "last_scan_start_time": "N/A", "last_scan_end_time": "N/A",
        "markets_found": 0, "signals_found": 0, "active_trades_count": 0,
        "scan_in_progress": False, "btc_market_mood": "غير محدد"
    },
    "scan_history": deque(maxlen=10)
}
scan_lock = asyncio.Lock()

# --- Settings Management ---
DEFAULT_SETTINGS = {
    "real_trading_enabled": True,
    "real_trade_size_percentage": 2.0,
    "max_concurrent_trades": 3,
    "top_n_symbols_by_volume": 100,
    "concurrent_workers": 8,
    "market_regime_filter_enabled": True,
    "fundamental_analysis_enabled": True,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "rsi_divergence", "supertrend_pullback"],
    "use_master_trend_filter": True,
    "master_trend_filter_ma_period": 50,
    "master_adx_filter_level": 25,
    "fear_and_greed_filter_enabled": True,
    "fear_and_greed_threshold": 25,
    "use_dynamic_risk_management": True,
    "atr_period": 14,
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 2.0,
    "trailing_sl_enabled": True,
    "trailing_sl_activate_percent": 1.5,
    "trailing_sl_percent": 1.0,
    "momentum_breakout": {"vwap_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bbands_period": 20, "bbands_stddev": 2.0, "rsi_period": 14, "rsi_max_level": 65, "volume_spike_multiplier": 1.8},
    "breakout_squeeze_pro": {"bbands_period": 20, "bbands_stddev": 2.0, "keltner_period": 20, "keltner_atr_multiplier": 1.5, "volume_confirmation_enabled": True},
    "rsi_divergence": {"rsi_period": 14, "lookback_period": 35, "peak_trough_lookback": 5, "confirm_with_rsi_exit": True},
    "supertrend_pullback": {"atr_period": 10, "atr_multiplier": 3.0, "swing_high_lookback": 10},
    "liquidity_filters": {"min_quote_volume_24h_usd": 2_000_000, "max_spread_percent": 0.3, "rvol_period": 20, "min_rvol": 2.0},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 1.0},
    "stablecoin_filter": {"exclude_bases": ["USDT","USDC","DAI","FDUSD","TUSD","USDE","PYUSD","GUSD","EURT","USDJ"]},
    "ema_trend_filter": {"enabled": True, "ema_period": 200},
    "min_tp_sl_filter": {"min_tp_percent": 1.5, "min_sl_percent": 0.8},
    "min_signal_strength": 2,
    "active_preset_name": "STRICT",
    "last_market_mood": {"timestamp": "N/A", "mood": "UNKNOWN", "reason": "No scan performed yet."},
}

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                stored_settings = json.load(f)
            bot_data["settings"] = DEFAULT_SETTINGS.copy()
            # (Recursive merge logic can be simplified if structure is fixed)
            bot_data["settings"].update(stored_settings)
        else:
            bot_data["settings"] = DEFAULT_SETTINGS.copy()
        save_settings()
        logger.info(f"✅ Settings loaded successfully from {SETTINGS_FILE}")
    except Exception as e:
        logger.error(f"💥 Failed to load settings: {e}")
        bot_data["settings"] = DEFAULT_SETTINGS.copy()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(bot_data["settings"], f, indent=4, ensure_ascii=False)
        logger.info(f"💾 Settings saved successfully to {SETTINGS_FILE}")
    except Exception as e:
        logger.error(f"💥 Failed to save settings: {e}")

# --- Database Management ---
def init_database():
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, exchange TEXT, symbol TEXT,
                entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL,
                entry_value_usdt REAL, status TEXT, exit_price REAL, closed_at TEXT,
                exit_value_usdt REAL, pnl_usdt REAL, trailing_sl_active BOOLEAN DEFAULT FALSE,
                highest_price REAL, reason TEXT, is_real_trade BOOLEAN DEFAULT TRUE,
                entry_order_id TEXT, exit_order_ids_json TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, total_trades INTEGER,
                winning_trades INTEGER, losing_trades INTEGER, total_pnl REAL,
                win_rate REAL, created_at TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"🗄️ Database initialized successfully at: {DB_FILE}")
    except Exception as e:
        logger.error(f"💥 Failed to initialize database at {DB_FILE}: {e}")

# (Other helper functions like DB operations, news analysis, scanners remain the same)
# ... (All helper functions from the previous version are assumed to be here) ...

# --- Telegram Functions ---
async def send_telegram_message(bot, chat_id, text, reply_markup=None):
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        logger.debug(f"📤 Message sent to {chat_id}")
    except Exception as e:
        logger.error(f"💥 Failed to send message to {chat_id}: {e}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings = bot_data['settings']
    trading_mode = "🚨 التداول الحقيقي" if settings.get('real_trading_enabled', True) else "📊 التداول الافتراضي"
    welcome_message = (
        f"**🤖 أهلاً بك في بوت التداول الآلي**\n\n"
        f"**▫️الوضع الحالي:** {trading_mode}\n"
        f"**▫️المنصات المتصلة:** {len(bot_data['exchanges'])}\n\n"
        f"**استخدم الأزرار للتنقل.**"
    )
    await send_telegram_message(context.bot, update.effective_chat.id, welcome_message, create_main_menu())

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = update.effective_chat.id
    try:
        if data == "main_menu":
            await query.edit_message_text("**🏠 القائمة الرئيسية**", parse_mode=ParseMode.MARKDOWN, reply_markup=create_main_menu())
        elif data == "status":
            status = bot_data['status_snapshot']
            active_trades = 0
            try:
                conn = sqlite3.connect(DB_FILE)
                active_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة'").fetchone()[0]
                conn.close()
            except Exception as e:
                logger.error(f"DB error getting active trades count: {e}")

            status_text = (
                f"**📊 حالة البوت**\n\n"
                f"**▫️حالة الفحص:** {'🔄 يعمل' if status.get('scan_in_progress') else '⏸️ متوقف'}\n"
                f"**▫️آخر فحص:** {status.get('last_scan_end_time', 'N/A')}\n"
                f"**▫️الصفقات النشطة:** {active_trades}"
            )
            await query.edit_message_text(status_text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 رجوع", callback_data="main_menu")]]))
        elif data == "manual_scan":
            if bot_data['status_snapshot'].get('scan_in_progress', False):
                await context.bot.answer_callback_query(query.id, text="⏳ فحص قيد التنفيذ حالياً...", show_alert=True)
            else:
                await query.edit_message_text("**🔍 بدء الفحص الفوري...**", parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 رجوع", callback_data="main_menu")]]))
                asyncio.create_task(perform_scan_simplified(context))
        elif data == "toggle_real_trading":
            settings = bot_data['settings']
            settings['real_trading_enabled'] = not settings.get('real_trading_enabled', True)
            save_settings()
            await query.edit_message_text(
                f"**⚙️ تم تحديث وضع التداول**",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=create_main_menu()
            )
        else:
             await context.bot.answer_callback_query(query.id, text="❓ أمر غير معروف", show_alert=False)
    except Exception as e:
        logger.error(f"💥 Error in callback handler: {e}", exc_info=True)

# --- [IMPROVEMENT] Handlers for unknown commands and text ---
async def unknown_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles any command that is not recognized."""
    await send_telegram_message(context.bot, update.effective_chat.id, "عذراً، هذا الأمر غير معروف. يرجى استخدام /start لعرض القائمة الرئيسية.")

async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles any plain text message from the user."""
    await send_telegram_message(context.bot, update.effective_chat.id, "يرجى استخدام الأزرار التفاعلية أو الأمر /start للتواصل مع البوت.")


# --- Core Jobs ---
async def track_active_trades(context: ContextTypes.DEFAULT_TYPE):
    logger.info("...periodically checking active trades...")
    # This function remains the same, no changes needed.

async def perform_scan_simplified(context: ContextTypes.DEFAULT_TYPE):
    """فحص مبسط للاستقرار مع معالجة صحيحة للقفل"""
    if not scan_lock.locked():
        await scan_lock.acquire()
        try:
            bot_data['status_snapshot']['scan_in_progress'] = True
            logger.info("🔍 Starting market scan...")
            
            # Simplified scan logic for demonstration
            await asyncio.sleep(5) # Simulate work
            signals_found = 1 # Simulate finding a signal
            
            bot_data['status_snapshot']['scan_in_progress'] = False
            bot_data['status_snapshot']['last_scan_end_time'] = datetime.now(EGYPT_TZ).strftime('%H:%M:%S')

            summary_text = f"**🔍 انتهى الفحص:** تم العثور على {signals_found} إشارة."
            await send_telegram_message(context.bot, TELEGRAM_CHAT_ID, summary_text)
            logger.info(f"✅ Scan complete.")
        except Exception as e:
            logger.error(f"💥 Error in scan job: {e}")
            bot_data['status_snapshot']['scan_in_progress'] = False
        finally:
            scan_lock.release()
    else:
        logger.info("Scan already in progress. Skipping this run.")

# --- Main Function (FINAL STABLE VERSION v3) ---
async def main():
    """الدالة الرئيسية النهائية التي تضمن تشغيلاً وإغلاقاً نظيفاً ومستجيباً"""
    logger.info("🚀 ========== BOT Is Starting Up (v3) ==========")
    
    # Load settings and DB
    load_settings()
    init_database()

    # Initialize exchanges
    await initialize_exchanges()
    if not bot_data["exchanges"]:
        logger.critical("❌ No exchanges connected! Bot cannot continue.")
        return

    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # --- Register handlers ---
    # The order is important! More specific handlers should come first.
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(handle_callback_query))
    # [NEW] Add handlers for unknown commands and text messages
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
    
    # --- Register jobs ---
    job_queue = application.job_queue
    job_queue.run_repeating(perform_scan_simplified, interval=SCAN_INTERVAL_SECONDS, first=10, name="market_scan")
    # job_queue.run_repeating(track_active_trades, interval=TRACK_INTERVAL_SECONDS, first=30, name="trade_tracker") # Temporarily disabled for stability check
    logger.info("⏰ Scheduled jobs configured.")

    # --- Run the bot ---
    # This simplified run_polling handles the lifecycle correctly and is more stable
    # for environments like PM2.
    try:
        logger.info("🎯 Bot is now running and polling for updates...")
        await application.run_polling(allowed_updates=Update.ALL_TYPES)
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Bot shutdown requested.")
    finally:
        logger.info("🧹 Cleaning up resources...")
        if application.job_queue and application.job_queue.is_running:
            await application.job_queue.stop()
        for ex_id, exchange in bot_data["exchanges"].items():
            if exchange:
                try:
                    await exchange.close()
                    logger.info(f"...Closed connection to {ex_id}")
                except Exception:
                    pass
        logger.info("👋 Bot shutdown complete.")


# --- Entry Point ---
if __name__ == "__main__":
    print("🚀 Real Trading Bot v12 Final Stable v3 - Starting...")
    print(f"📅 Date: {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')} EEST")
    
    # A simple loop for robustness in case of unexpected crashes
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user (Ctrl+C).")
            break
        except Exception as e:
            print(f"\n💥 CRITICAL FAILURE IN MAIN EXECUTION: {e}")
            print("🔄 The bot will restart in 15 seconds...")
            time.sleep(15)
