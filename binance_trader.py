# -*- coding: utf-8 -*-

# =======================================================================================

# --- 💣 بوت كاسحة الألغام (Minesweeper Bot) v5.0 (إصدار العقل المدبر) 💣 ---

# =======================================================================================

# --- سجل التغييرات الكامل (v5.0) ---

#

# --- قسم الذكاء والتحكم الذاتي (Auto-Pilot) ---

# 1.  [ميزة كبرى] إضافة قائمة "🧠 التحكم الذاتي" في الإعدادات.

# 2.  [ميزة] "إدارة المخاطر الديناميكية": يمكن للبوت الآن تعديل حجم الصفقة

#     تلقائيًا بناءً على مزاج السوق (إيجابي = 100%، محايد = 60%).

# 3.  [ميزة] "التبديل الذكي للأنماط": سيعود البوت لاقتراح تغيير نمط الفلترة

#     (e.g., from STRICT to PRO) بناءً على نشاط السوق.

# 4.  [ميزة] "التعلم الذاتي للاستراتيجيات": يمكن للبوت الآن مراجعة أداء

#     الاستراتيجيات وتعطيل الفاشلة منها تلقائيًا وإعلامك بذلك.

#

# --- قسم أدوات التداول المتقدمة ---

# 5.  [ميزة كبرى] "التداول اليدوي المتقدم": إضافة القدرة على إلغاء الأوامر

#     المفتوحة ووضع أوامر معلقة (Limit/Stop) مباشرة من البوت.

# 6.  [ميزة أمان] "زر الذعر": إضافة أداة لإغلاق كل الصفقات على منصة معينة

#     بعد تأكيد نصي بكلمة "تأكيد".

# 7.  [ميزة] "إغلاق صفقة فردية": عند مراجعة صفقة نشطة، يظهر زر جديد

#     لإغلاقها يدويًا بسعر السوق.

#

# --- قسم الشفافية والموثوقية ---

# 8.  [ميزة] "نظام الإشعارات الآمن": إذا فشل إرسال إشعار صفقة إلى القناة،

#     سيتم إرساله تلقائيًا إلى محادثتك الخاصة مع رسالة تنبيه.

# 9.  [ميزة] "رسالة تأكيد مفصلة": تصميم رسالة جديدة غنية بالمعلومات

#     تظهر بعد تنفيذ كل صفقة حقيقية بنجاح.

#

# --- قسم إدارة البيانات ---

# 10. [ميزة] "إدارة قاعدة البيانات": قائمة جديدة في الإعدادات تتيح لك

#     تنظيف السجل من الصفقات القديمة وتصدير كل بياناتك إلى ملف CSV.

#

# --- إصلاحات الأخطاء النهائية ---

# 11. [إصلاح حاسم] حل مشكلة 'sqlite3.Row' object has no attribute 'get'

#     التي كانت تعطل "تقرير المخاطر".

# 12. [إصلاح حاسم] إعادة هيكلة "تقرير المزامنة" ليعمل مع كل المنصات.

#

# ... (جميع الإصلاحات السابقة من v4.4 موجودة)

# =======================================================================================





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

import csv # [جديد] لإدارة ملفات CSV

from datetime import datetime, time as dt_time, timedelta, timezone

from zoneinfo import ZoneInfo

from collections import deque, Counter, defaultdict



# [UPGRADE] المكتبات الجديدة لتحليل الأخبار

import feedparser

# [جديد] إضافة مكتبة numpy

import numpy as np

try:

    import nltk

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    NLTK_AVAILABLE = True

except ImportError:

    NLTK_AVAILABLE = False

    logging.warning("Library 'nltk' not found. Sentiment analysis will be disabled.")



# [تعديل] استيراد مكتبة httpx للطلبات غير المتزامنة

import httpx

from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup, InputFile

from telegram.constants import ParseMode

# [جديد] استيراد للتحكم في مهلة الشبكة

from telegram.request import HTTPXRequest

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



# [تعديل] إضافة متغيرات مفاتيح API الخاصة بمنصة Binance

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY')

BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_BINANCE_API_SECRET')



# [جديد] إضافة متغيرات مفاتيح API الخاصة بمنصة KuCoin

KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', 'YOUR_KUCOIN_API_KEY')

KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET', 'YOUR_KUCOIN_API_SECRET')

KUCOIN_API_PASSPHRASE = os.getenv('KUCOIN_API_PASSPHRASE', 'YOUR_KUCOIN_API_PASSPHRASE')





if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or TELEGRAM_CHAT_ID == 'YOUR_CHAT_ID_HERE':

    print("FATAL ERROR: Please set your Telegram Token and Chat ID.")

    exit()

if ALPHA_VANTAGE_API_KEY == 'YOUR_AV_KEY_HERE':

    logging.warning("Alpha Vantage API key not set. Economic calendar will be disabled.")





# --- إعدادات البوت --- #

EXCHANGES_TO_SCAN = ['binance', 'okx', 'bybit', 'kucoin', 'gate', 'mexc']

TIMEFRAME = '15m'

HIGHER_TIMEFRAME = '1h'

SCAN_INTERVAL_SECONDS = 900

TRACK_INTERVAL_SECONDS = 120



APP_ROOT = '.'

# [تعديل] تحديث أسماء الملفات للبوت الجديد

DB_FILE = os.path.join(APP_ROOT, 'minesweeper_bot.db')

SETTINGS_FILE = os.path.join(APP_ROOT, 'minesweeper_settings.json')

CSV_EXPORT_FILE = os.path.join(APP_ROOT, 'trades_export.csv') # [جديد]



EGYPT_TZ = ZoneInfo("Africa/Cairo")



# --- إعداد مسجل الأحداث (Logger) --- #

LOG_FILE = os.path.join(APP_ROOT, 'minesweeper_bot.log')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'a'), logging.StreamHandler()])

logging.getLogger('httpx').setLevel(logging.WARNING)

logging.getLogger('apscheduler').setLevel(logging.WARNING)

logging.getLogger('telegram').setLevel(logging.WARNING)

logging.getLogger('requests').setLevel(logging.WARNING)

logger = logging.getLogger("MinesweeperBot")





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

PRESET_VERY_LAX = {

  "liquidity_filters": {"min_quote_volume_24h_usd": 200000, "max_spread_percent": 2.0, "rvol_period": 10, "min_rvol": 0.8},

  "volatility_filters": {"atr_period_for_filter": 10, "min_atr_percent": 0.2},

  "ema_trend_filter": {"enabled": False, "ema_period": 200},

  "min_tp_sl_filter": {"min_tp_percent": 0.3, "min_sl_percent": 0.15}

}

PRESETS = {"PRO": PRESET_PRO, "LAX": PRESET_LAX, "STRICT": PRESET_STRICT, "VERY_LAX": PRESET_VERY_LAX}



# [تعديل] إضافة أسماء الاستراتيجيات الجديدة

STRATEGY_NAMES_AR = {

    "momentum_breakout": "زخم اختراقي",

    "breakout_squeeze_pro": "اختراق انضغاطي",

    "support_rebound": "ارتداد الدعم",

    "whale_radar": "رادار الحيتان",

    "sniper_pro": "القناص المحترف",

}





# --- Constants for Interactive Settings menu ---

EDITABLE_PARAMS = {

    "إعدادات عامة": [

        "max_concurrent_trades", "top_n_symbols_by_volume", "concurrent_workers",

        "min_signal_strength"

    ],

    "إعدادات المخاطر": [

        "automate_real_tsl", "real_trade_size_usdt", "virtual_trade_size_percentage",

        "atr_sl_multiplier", "risk_reward_ratio", "trailing_sl_activation_percent", "trailing_sl_callback_percent"

    ],

    "الفلاتر والاتجاه": [

        "market_regime_filter_enabled", "use_master_trend_filter", "fear_and_greed_filter_enabled",

        "master_adx_filter_level", "master_trend_filter_ma_period", "trailing_sl_enabled", "fear_and_greed_threshold",

        "fundamental_analysis_enabled"

    ]

}

PARAM_DISPLAY_NAMES = {

    "automate_real_tsl": "🤖 أتمتة الوقف المتحرك الحقيقي",

    "real_trade_size_usdt": "💵 حجم الصفقة الحقيقية ($)",

    "virtual_trade_size_percentage": "📊 حجم الصفقة الوهمية (%)",

    "max_concurrent_trades": "أقصى عدد للصفقات",

    "top_n_symbols_by_volume": "عدد العملات للفحص",

    "concurrent_workers": "عمال الفحص المتزامنين",

    "min_signal_strength": "أدنى قوة للإشارة",

    "atr_sl_multiplier": "مضاعف وقف الخسارة (ATR)",

    "risk_reward_ratio": "نسبة المخاطرة/العائد",

    "trailing_sl_activation_percent": "تفعيل الوقف المتحرك (%)",

    "trailing_sl_callback_percent": "مسافة الوقف المتحرك (%)",

    "market_regime_filter_enabled": "فلتر وضع السوق (فني)",

    "use_master_trend_filter": "فلتر الاتجاه العام (BTC)",

    "master_adx_filter_level": "مستوى فلتر ADX",

    "master_trend_filter_ma_period": "فترة فلتر الاتجاه",

    "trailing_sl_enabled": "تفعيل الوقف المتحرك",

    "fear_and_greed_filter_enabled": "فلتر الخوف والطمع",

    "fear_and_greed_threshold": "حد مؤشر الخوف",

    "fundamental_analysis_enabled": "فلتر الأخبار والبيانات",

}



# [جديد] إعدادات التحكم الذاتي

AUTOPILOT_PARAMS = {

    "dynamic_risk_enabled": "🧠 إدارة المخاطر الديناميكية",

    "smart_preset_switching_enabled": "⚙️ التبديل الذكي للأنماط",

    "auto_disable_strategies_enabled": "📉 التعلم الذاتي للاستراتيجيات",

}





# --- Global Bot State ---

bot_data = {

    "exchanges": {},

    "public_exchanges": {},

    "last_signal_time": {},

    "settings": {},

    "status_snapshot": {

        "last_scan_start_time": None, "last_scan_end_time": None,

        "markets_found": 0, "signals_found": 0, "active_trades_count": 0,

        "scan_in_progress": False, "btc_market_mood": "غير محدد"

    },

    "scan_history": deque(maxlen=10)

}

scan_lock = asyncio.Lock()

report_lock = asyncio.Lock()



# --- Settings Management ---

DEFAULT_SETTINGS = {

    "real_trading_per_exchange": {ex: False for ex in EXCHANGES_TO_SCAN}, 

    "automate_real_tsl": False, 

    "real_trade_size_usdt": 15.0,

    "virtual_portfolio_balance_usdt": 1000.0, 

    "virtual_trade_size_percentage": 5.0, 

    "max_concurrent_trades": 10, 

    "top_n_symbols_by_volume": 250, 

    "concurrent_workers": 10,

    "market_regime_filter_enabled": True, 

    "fundamental_analysis_enabled": True,

    "active_scanners": list(STRATEGY_NAMES_AR.keys()),

    "use_master_trend_filter": True, "master_trend_filter_ma_period": 50, "master_adx_filter_level": 22,

    "fear_and_greed_filter_enabled": True, "fear_and_greed_threshold": 30,

    "use_dynamic_risk_management": True, "atr_period": 14, "atr_sl_multiplier": 2.5, "risk_reward_ratio": 2.0,

    "trailing_sl_enabled": True, "trailing_sl_activation_percent": 1.5, "trailing_sl_callback_percent": 1.0,

    

    # [جديد] إعدادات التحكم الذاتي

    "autopilot_settings": {

        "dynamic_risk_enabled": False,

        "smart_preset_switching_enabled": True,

        "auto_disable_strategies_enabled": False,

        "auto_disable_min_trades": 20,

        "auto_disable_win_rate_threshold": 40.0,

    },



    # إعدادات الاستراتيجيات

    "momentum_breakout": {"vwap_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bbands_period": 20, "bbands_stddev": 2.0, "rsi_period": 14, "rsi_max_level": 68, "volume_spike_multiplier": 1.5},

    "breakout_squeeze_pro": {"bbands_period": 20, "bbands_stddev": 2.0, "keltner_period": 20, "keltner_atr_multiplier": 1.5, "volume_confirmation_enabled": True},

    "sniper_pro": {"compression_hours": 6, "max_volatility_percent": 12.0},

    "whale_radar": {"wall_threshold_usdt": 30000},

    

    # الفلاتر

    "liquidity_filters": {"min_quote_volume_24h_usd": 1_000_000, "max_spread_percent": 0.5, "rvol_period": 20, "min_rvol": 1.5},

    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.8},

    "stablecoin_filter": {"exclude_bases": ["USDT","USDC","DAI","FDUSD","TUSD","USDE","PYUSD","GUSD","EURT","USDJ"]},

    "ema_trend_filter": {"enabled": True, "ema_period": 200},

    "min_tp_sl_filter": {"min_tp_percent": 1.0, "min_sl_percent": 0.5},

    "min_signal_strength": 1,

    

    # الحالة

    "active_preset_name": "PRO",

    "last_market_mood": {"timestamp": "N/A", "mood": "UNKNOWN", "reason": "No scan performed yet."},

    "last_suggestion_time": 0

}





def load_settings():

    try:

        if os.path.exists(SETTINGS_FILE):

            with open(SETTINGS_FILE, 'r') as f: bot_data["settings"] = json.load(f)

        else:

            bot_data["settings"] = DEFAULT_SETTINGS.copy()

            save_settings()

            logger.info("No settings file found. Created a new one with default values.")

            return



        updated = False

        # [جديد] آلية ترحيل الإعدادات القديمة

        if "real_trading_enabled" in bot_data["settings"]:

            logger.warning("Old 'real_trading_enabled' setting found. Migrating to new per-exchange format.")

            old_value = bot_data["settings"].pop("real_trading_enabled")

            bot_data["settings"]["real_trading_per_exchange"] = {ex: old_value for ex in EXCHANGES_TO_SCAN}

            updated = True

            

        for key, value in DEFAULT_SETTINGS.items():

            if key not in bot_data["settings"]:

                bot_data["settings"][key] = value; updated = True

            elif isinstance(value, dict):

                # Handle nested dictionaries like autopilot_settings

                for sub_key, sub_value in value.items():

                    if sub_key not in bot_data["settings"].get(key, {}):

                        bot_data["settings"][key][sub_key] = sub_value; updated = True

        if updated: save_settings()

        

        logger.info(f"Settings loaded successfully from {SETTINGS_FILE}")

    except Exception as e:

        logger.error(f"Failed to load settings: {e}")

        bot_data["settings"] = DEFAULT_SETTINGS.copy()





def save_settings():

    try:

        with open(SETTINGS_FILE, 'w') as f: json.dump(bot_data["settings"], f, indent=4)

        logger.info(f"Settings saved successfully to {SETTINGS_FILE}")

    except Exception as e:

        logger.error(f"Failed to save settings: {e}")



# --- Database Management ---

def migrate_database():

    """Checks for missing columns in the database and adds them. This is critical for updates."""

    logger.info("Checking database schema...")

    try:

        conn = sqlite3.connect(DB_FILE, timeout=10)

        cursor = conn.cursor()

        

        required_columns = {

            "id": "INTEGER PRIMARY KEY AUTOINCREMENT", "timestamp": "TEXT", "exchange": "TEXT",

            "symbol": "TEXT", "entry_price": "REAL", "take_profit": "REAL", "stop_loss": "REAL",

            "quantity": "REAL", "entry_value_usdt": "REAL", "status": "TEXT", "exit_price": "REAL",

            "closed_at": "TEXT", "exit_value_usdt": "REAL", "pnl_usdt": "REAL",

            "trailing_sl_active": "BOOLEAN", "highest_price": "REAL", "reason": "TEXT",

            "is_real_trade": "BOOLEAN", "trade_mode": "TEXT DEFAULT 'virtual'",

            "entry_order_id": "TEXT", "exit_order_ids_json": "TEXT"

        }

        

        cursor.execute("PRAGMA table_info(trades)")

        existing_columns = {row[1] for row in cursor.fetchall()}

        

        for col_name, col_type in required_columns.items():

            if col_name not in existing_columns:

                logger.warning(f"Database schema mismatch. Missing column '{col_name}'. Adding it now.")

                cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")

                logger.info(f"Column '{col_name}' added successfully.")

        

        conn.commit()

        conn.close()

        logger.info("Database schema check complete.")

    except Exception as e:

        logger.error(f"CRITICAL: Database migration failed: {e}", exc_info=True)





def init_database():

    try:

        conn = sqlite3.connect(DB_FILE, timeout=10)

        cursor = conn.cursor()

        

        cursor.execute('''

            CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT)

        ''')

        

        conn.commit()

        conn.close()

        

        migrate_database()

        

        logger.info(f"Database initialized and schema verified at: {DB_FILE}")

    except Exception as e:

        logger.error(f"Failed to initialize database at {DB_FILE}: {e}")



def log_recommendation_to_db(signal):

    try:

        conn = sqlite3.connect(DB_FILE, timeout=10)

        cursor = conn.cursor()

        sql = '''INSERT INTO trades (timestamp, exchange, symbol, entry_price, take_profit, stop_loss, quantity, entry_value_usdt, status, trailing_sl_active, highest_price, reason, trade_mode, entry_order_id, exit_order_ids_json)

                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

        params = (

            signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),

            signal['exchange'],

            signal['symbol'],

            signal.get('verified_entry_price', signal['entry_price']), 

            signal['take_profit'],

            signal['stop_loss'],

            signal.get('verified_quantity', signal['quantity']), 

            signal.get('verified_entry_value', signal['entry_value_usdt']), 

            'نشطة',

            False,

            signal.get('verified_entry_price', signal['entry_price']),

            signal['reason'],

            'real' if signal.get('is_real_trade') else 'virtual',

            signal.get('entry_order_id'),

            signal.get('exit_order_ids_json')

        )

        cursor.execute(sql, params)

        trade_id = cursor.lastrowid

        conn.commit()

        conn.close()

        return trade_id

    except Exception as e:

        logger.error(f"Failed to log recommendation to DB: {e}")

        return None



# --- Fundamental & News Analysis Section ---

async def get_alpha_vantage_economic_events():

    if ALPHA_VANTAGE_API_KEY == 'YOUR_AV_KEY_HERE':

        logger.warning("Alpha Vantage API key is not set. Skipping economic calendar check.")

        return []

    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    params = {'function': 'ECONOMIC_CALENDAR', 'horizon': '3month', 'apikey': ALPHA_VANTAGE_API_KEY}

    try:

        async with httpx.AsyncClient() as client:

            response = await client.get('https://www.alphavantage.co/query', params=params, timeout=20)

            response.raise_for_status()



        data_str = response.text

        if "premium" in data_str.lower():

             logger.error("Alpha Vantage API returned a premium feature error for Economic Calendar.")

             return []

        lines = data_str.strip().split('\r\n')

        if len(lines) < 2: return []

        header = [h.strip() for h in lines[0].split(',')]

        high_impact_events = []

        for line in lines[1:]:

            values = [v.strip() for v in line.split(',')]

            event = dict(zip(header, values))

            if event.get('releaseDate', '') == today_str and event.get('impact', '').lower() == 'high' and event.get('country', '') in ['USD', 'EUR']:

                high_impact_events.append(event.get('event', 'Unknown Event'))

        if high_impact_events: logger.warning(f"High-impact events today via Alpha Vantage: {high_impact_events}")

        return high_impact_events

    except httpx.RequestError as e:

        logger.error(f"Failed to fetch economic calendar data from Alpha Vantage: {e}")

        return None



def get_latest_crypto_news(limit=15):

    urls = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]

    headlines = []

    for url in urls:

        try:

            feed = feedparser.parse(url)

            headlines.extend(entry.title for entry in feed.entries[:5])

        except Exception as e:

            logger.error(f"Failed to fetch news from {url}: {e}")

    return list(set(headlines))[:limit]



def analyze_sentiment_of_headlines(headlines):

    if not headlines or not NLTK_AVAILABLE: return 0.0

    sia = SentimentIntensityAnalyzer()

    total_compound_score = sum(sia.polarity_scores(headline)['compound'] for headline in headlines)

    return total_compound_score / len(headlines) if headlines else 0.0



async def get_fundamental_market_mood():

    high_impact_events = await get_alpha_vantage_economic_events()

    if high_impact_events is None: return "DANGEROUS", -1.0, "فشل جلب البيانات الاقتصادية"

    if high_impact_events: return "DANGEROUS", -0.9, f"أحداث هامة اليوم: {', '.join(high_impact_events)}"

    latest_headlines = get_latest_crypto_news()

    sentiment_score = analyze_sentiment_of_headlines(latest_headlines)

    logger.info(f"Market sentiment score based on news: {sentiment_score:.2f}")

    if sentiment_score > 0.25: return "POSITIVE", sentiment_score, f"مشاعر إيجابية (الدرجة: {sentiment_score:.2f})"

    elif sentiment_score < -0.25: return "NEGATIVE", sentiment_score, f"مشاعر سلبية (الدرجة: {sentiment_score:.2f})"

    else: return "NEUTRAL", sentiment_score, f"مشاعر محايدة (الدرجة: {sentiment_score:.2f})"





# --- Advanced Scanners ---

def find_col(df_columns, prefix):

    try: return next(col for col in df_columns if col.startswith(prefix))

    except StopIteration: return None



def analyze_momentum_breakout(df, params, rvol, adx_value, exchange, symbol):

    df.ta.vwap(append=True)

    df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)

    df.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'], append=True)

    df.ta.rsi(length=params['rsi_period'], append=True)

    macd_col, macds_col, bbu_col, rsi_col = (

        find_col(df.columns, f"MACD_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}"),

        find_col(df.columns, f"MACDs_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}"),

        find_col(df.columns, f"BBU_{params['bbands_period']}_"),

        find_col(df.columns, f"RSI_{params['rsi_period']}")

    )

    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None

    last, prev = df.iloc[-2], df.iloc[-3]

    rvol_ok = rvol >= bot_data['settings']['liquidity_filters']['min_rvol']

    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and

        last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and

        last[rsi_col] < params['rsi_max_level'] and rvol_ok):

        return {"reason": "momentum_breakout", "type": "long"}

    return None



def analyze_breakout_squeeze_pro(df, params, rvol, adx_value, exchange, symbol):

    df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)

    df.ta.kc(length=params['keltner_period'], scalar=params['keltner_atr_multiplier'], append=True)

    df.ta.obv(append=True)

    bbu_col, bbl_col, kcu_col, kcl_col = (

        find_col(df.columns, f"BBU_{params['bbands_period']}_"), find_col(df.columns, f"BBL_{params['bbands_period']}_"),

        find_col(df.columns, f"KCUe_{params['keltner_period']}_"), find_col(df.columns, f"KCLEe_{params['keltner_period']}_")

    )

    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None

    last, prev = df.iloc[-2], df.iloc[-3]

    is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]

    if is_in_squeeze:

        breakout_fired = last['close'] > last[bbu_col]

        volume_ok = not params.get('volume_confirmation_enabled', True) or last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5

        rvol_ok = rvol >= bot_data['settings']['liquidity_filters']['min_rvol']

        obv_rising = df['OBV'].iloc[-2] > df['OBV'].iloc[-3]

        if breakout_fired and rvol_ok and obv_rising:

            if params.get('volume_confirmation_enabled', True) and not volume_ok: return None

            return {"reason": "breakout_squeeze_pro", "type": "long"}

    return None



def find_support_resistance(high_prices, low_prices, window=10):

    supports, resistances = [], []

    if len(high_prices) < (2 * window + 1):

        return [], []

        

    for i in range(window, len(high_prices) - window):

        if high_prices[i] == max(high_prices[i-window:i+window+1]): resistances.append(high_prices[i])

        if low_prices[i] == min(low_prices[i-window:i+window+1]): supports.append(low_prices[i])

    if not supports and not resistances: return [], []



    def cluster_levels(levels, tolerance_percent=0.5):

        if not levels: return []

        clustered = []

        levels.sort()

        current_cluster = [levels[0]]

        for level in levels[1:]:

            if (level - current_cluster[-1]) / current_cluster[-1] * 100 < tolerance_percent:

                current_cluster.append(level)

            else:

                clustered.append(np.mean(current_cluster))

                current_cluster = [level]

        if current_cluster:

            clustered.append(np.mean(current_cluster))

        return clustered



    return cluster_levels(supports), cluster_levels(resistances)



def analyze_sniper_pro(df, params, rvol, adx_value, exchange, symbol):

    try:

        compression_candles = int(params.get("compression_hours", 6) * 4) 

        if len(df) < compression_candles + 2:

            return None



        compression_df = df.iloc[-compression_candles-1:-1]

        highest_high = compression_df['high'].max()

        lowest_low = compression_df['low'].min()



        volatility = (highest_high - lowest_low) / lowest_low * 100 if lowest_low > 0 else float('inf')



        if volatility < params.get("max_volatility_percent", 12.0):

            last_candle = df.iloc[-2]

            if last_candle['close'] > highest_high:

                avg_volume = compression_df['volume'].mean()

                if last_candle['volume'] > avg_volume * 2:

                    return {"reason": "sniper_pro", "type": "long"}

    except Exception as e:

        logger.warning(f"Sniper Pro scan failed for {symbol}: {e}")

    return None



async def analyze_whale_radar(df, params, rvol, adx_value, exchange, symbol):

    try:

        threshold = params.get("wall_threshold_usdt", 30000)

        ob = await exchange.fetch_order_book(symbol, limit=20)

        if not ob or not ob.get('bids'): return None



        bids = ob.get('bids', [])

        # [إصلاح] جعل الكود أكثر صلابة ضد البيانات غير المتوقعة من المنصة

        total_bid_value = 0

        for item in bids[:10]:

            if isinstance(item, list) and len(item) >= 2:

                price, qty = item[0], item[1]

                total_bid_value += float(price) * float(qty)



        if total_bid_value > threshold:

            return {"reason": "whale_radar", "type": "long"}

    except Exception as e:

        logger.warning(f"Whale Radar scan failed for {symbol}: {e}")

    return None



async def analyze_support_rebound(df, params, rvol, adx_value, exchange, symbol):

    try:

        ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)

        if not ohlcv_1h or len(ohlcv_1h) < 50: return None



        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        current_price = df_1h['close'].iloc[-1]



        supports, _ = find_support_resistance(df_1h['high'].to_numpy(), df_1h['low'].to_numpy(), window=5)

        if not supports: return None



        closest_support = max([s for s in supports if s < current_price], default=None)

        if not closest_support: return None



        if (current_price - closest_support) / closest_support * 100 < 1.0:

            last_candle_15m = df.iloc[-2]

            avg_volume_15m = df['volume'].rolling(window=20).mean().iloc[-2]



            if last_candle_15m['close'] > last_candle_15m['open'] and last_candle_15m['volume'] > avg_volume_15m * 1.5:

                 return {"reason": "support_rebound", "type": "long"}

    except Exception as e:

        logger.warning(f"Support Rebound scan failed for {symbol}: {e}")

    return None





SCANNERS = {

    "momentum_breakout": analyze_momentum_breakout,

    "breakout_squeeze_pro": analyze_breakout_squeeze_pro,

    "support_rebound": analyze_support_rebound,

    "whale_radar": analyze_whale_radar,

    "sniper_pro": analyze_sniper_pro,

}



# --- Core Bot Functions ---

async def initialize_exchanges():

    async def connect(ex_id):

        try:

            public_exchange = getattr(ccxt_async, ex_id)({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})

            await public_exchange.load_markets()

            bot_data["public_exchanges"][ex_id] = public_exchange

            logger.info(f"Connected to {ex_id} with PUBLIC client.")

        except Exception as e:

            logger.error(f"Failed to connect PUBLIC client for {ex_id}: {e}")

            if 'public_exchange' in locals(): await public_exchange.close()



        params = {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}

        authenticated = False

        if ex_id == 'binance' and BINANCE_API_KEY != 'YOUR_BINANCE_API_KEY':

            logger.info("Binance API Keys found. Initializing with private client.")

            params.update({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET})

            authenticated = True



        if ex_id == 'kucoin' and KUCOIN_API_KEY != 'YOUR_KUCOIN_API_KEY':

            logger.info("KuCoin API Keys found. Initializing with private client.")

            params.update({'apiKey': KUCOIN_API_KEY, 'secret': KUCOIN_API_SECRET, 'password': KUCOIN_API_PASSPHRASE})

            authenticated = True



        if authenticated:

            try:

                private_exchange = getattr(ccxt_async, ex_id)(params)

                await private_exchange.load_markets()

                bot_data["exchanges"][ex_id] = private_exchange

                logger.info(f"Connected to {ex_id} with PRIVATE (authenticated) client.")

            except Exception as e:

                logger.error(f"Failed to connect PRIVATE client for {ex_id}: {e}")

                if 'private_exchange' in locals(): await private_exchange.close()

        else:

             if ex_id in bot_data["public_exchanges"]:

                 bot_data["exchanges"][ex_id] = bot_data["public_exchanges"][ex_id]



    await asyncio.gather(*[connect(ex_id) for ex_id in EXCHANGES_TO_SCAN])





async def aggregate_top_movers():

    all_tickers = []

    async def fetch(ex_id, ex):

        try: return [dict(t, exchange=ex_id) for t in (await ex.fetch_tickers()).values()]

        except Exception: return []

    results = await asyncio.gather(*[fetch(ex_id, ex) for ex_id, ex in bot_data["public_exchanges"].items()])

    for res in results: all_tickers.extend(res)

    settings = bot_data['settings']

    excluded_bases = settings['stablecoin_filter']['exclude_bases']

    min_volume = settings['liquidity_filters']['min_quote_volume_24h_usd']

    usdt_tickers = [t for t in all_tickers if t.get('symbol') and t['symbol'].upper().endswith('/USDT') and t['symbol'].split('/')[0] not in excluded_bases and t.get('quoteVolume', 0) and t['quoteVolume'] >= min_volume and not any(k in t['symbol'].upper() for k in ['UP','DOWN','3L','3S','BEAR','BULL'])]

    sorted_tickers = sorted(usdt_tickers, key=lambda t: t.get('quoteVolume', 0), reverse=True)

    unique_symbols = {t['symbol']: {'exchange': t['exchange'], 'symbol': t['symbol']} for t in sorted_tickers}

    final_list = list(unique_symbols.values())[:settings['top_n_symbols_by_volume']]

    logger.info(f"Aggregated markets. Found {len(all_tickers)} tickers -> Post-filter: {len(usdt_tickers)} -> Selected top {len(final_list)} unique pairs.")

    bot_data['status_snapshot']['markets_found'] = len(final_list)

    return final_list



async def get_higher_timeframe_trend(exchange, symbol, ma_period):

    try:

        ohlcv_htf = await exchange.fetch_ohlcv(symbol, HIGHER_TIMEFRAME, limit=ma_period + 5)

        if len(ohlcv_htf) < ma_period: return None, "Not enough HTF data"

        df_htf = pd.DataFrame(ohlcv_htf, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        df_htf[f'SMA_{ma_period}'] = ta.sma(df_htf['close'], length=ma_period)

        last_candle = df_htf.iloc[-1]

        is_bullish = last_candle['close'] > last_candle[f'SMA_{ma_period}']

        return is_bullish, "Bullish" if is_bullish else "Bearish"

    except Exception as e:

        return None, f"Error: {e}"



async def worker(queue, results_list, settings, failure_counter):

    while not queue.empty():

        market_info = await queue.get()

        symbol = market_info.get('symbol', 'N/A')

        exchange = bot_data["public_exchanges"].get(market_info['exchange'])

        if not exchange or not settings.get('active_scanners'):

            queue.task_done()

            continue

        try:

            liq_filters, vol_filters, ema_filters = settings['liquidity_filters'], settings['volatility_filters'], settings['ema_trend_filter']



            orderbook = await exchange.fetch_order_book(symbol, limit=20)

            if not orderbook or not orderbook['bids'] or not orderbook['asks']:

                logger.debug(f"Reject {symbol}: Could not fetch order book."); continue



            best_bid, best_ask = orderbook['bids'][0][0], orderbook['asks'][0][0]

            if best_bid <= 0: logger.debug(f"Reject {symbol}: Invalid bid price."); continue



            spread_percent = ((best_ask - best_bid) / best_bid) * 100

            if spread_percent > liq_filters['max_spread_percent']:

                logger.debug(f"Reject {symbol}: High Spread ({spread_percent:.2f}% > {liq_filters['max_spread_percent']}%)"); continue



            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)

            if len(ohlcv) < ema_filters['ema_period']:

                logger.debug(f"Skipping {symbol}: Not enough data ({len(ohlcv)} candles) for EMA calculation."); continue



            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)



            df['volume_sma'] = ta.sma(df['volume'], length=liq_filters['rvol_period'])

            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] <= 0:

                logger.debug(f"Skipping {symbol}: Invalid SMA volume."); continue



            rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]

            if rvol < liq_filters['min_rvol']:

                logger.debug(f"Reject {symbol}: Low RVOL ({rvol:.2f} < {liq_filters['min_rvol']})"); continue



            atr_col_name = f"ATRr_{vol_filters['atr_period_for_filter']}"

            df.ta.atr(length=vol_filters['atr_period_for_filter'], append=True)

            last_close = df['close'].iloc[-2]

            if last_close <= 0: logger.debug(f"Skipping {symbol}: Invalid close price."); continue



            atr_percent = (df[atr_col_name].iloc[-2] / last_close) * 100

            if atr_percent < vol_filters['min_atr_percent']:

                logger.debug(f"Reject {symbol}: Low ATR% ({atr_percent:.2f}% < {vol_filters['min_atr_percent']}%)"); continue



            ema_col_name = f"EMA_{ema_filters['ema_period']}"

            df.ta.ema(length=ema_filters['ema_period'], append=True)

            if ema_col_name not in df.columns or pd.isna(df[ema_col_name].iloc[-2]):

                logger.debug(f"Skipping {symbol}: EMA_{ema_filters['ema_period']} could not be calculated.")

                continue



            if ema_filters['enabled'] and last_close < df[ema_col_name].iloc[-2]:

                logger.debug(f"Reject {symbol}: Below EMA{ema_filters['ema_period']}"); continue



            if settings.get('use_master_trend_filter'):

                is_htf_bullish, reason = await get_higher_timeframe_trend(exchange, symbol, settings['master_trend_filter_ma_period'])

                if not is_htf_bullish:

                    logger.debug(f"HTF Trend Filter FAILED for {symbol}: {reason}"); continue



            df.ta.adx(append=True)

            adx_col = find_col(df.columns, 'ADX_')

            adx_value = df[adx_col].iloc[-2] if adx_col and pd.notna(df[adx_col].iloc[-2]) else 0

            if settings.get('use_master_trend_filter') and adx_value < settings['master_adx_filter_level']:

                logger.debug(f"ADX Filter FAILED for {symbol}: {adx_value:.2f} < {settings['master_adx_filter_level']}"); continue



            confirmed_reasons = []

            for scanner_name in settings['active_scanners']:

                scanner_func = SCANNERS.get(scanner_name)

                if not scanner_func: continue

                

                scanner_params = settings.get(scanner_name, {})

                if asyncio.iscoroutinefunction(scanner_func):

                    result = await scanner_func(df.copy(), scanner_params, rvol, adx_value, exchange, symbol)

                else:

                    result = scanner_func(df.copy(), scanner_params, rvol, adx_value, exchange, symbol)



                if result and result.get("type") == "long":

                    confirmed_reasons.append(result['reason'])





            if confirmed_reasons and len(confirmed_reasons) >= settings.get("min_signal_strength", 1):

                reason_str = ' + '.join(confirmed_reasons)

                entry_price = df.iloc[-2]['close']

                df.ta.atr(length=settings['atr_period'], append=True)

                current_atr = df.iloc[-2].get(find_col(df.columns, f"ATRr_{settings['atr_period']}"), 0)

                if settings.get("use_dynamic_risk_management", False) and current_atr > 0:

                    risk_per_unit = current_atr * settings['atr_sl_multiplier']

                    stop_loss, take_profit = entry_price - risk_per_unit, entry_price + (risk_per_unit * settings['risk_reward_ratio'])

                else:

                    sl_percent = settings.get("stop_loss_percentage", 2.0)

                    tp_percent = settings.get("take_profit_percentage", 4.0)

                    stop_loss, take_profit = entry_price * (1 - sl_percent / 100), entry_price * (1 + tp_percent / 100)



                tp_percent_calc, sl_percent_calc = ((take_profit - entry_price) / entry_price * 100), ((entry_price - stop_loss) / entry_price * 100)

                min_filters = settings['min_tp_sl_filter']

                if tp_percent_calc >= min_filters['min_tp_percent'] and sl_percent_calc >= min_filters['min_sl_percent']:

                    results_list.append({"symbol": symbol, "exchange": market_info['exchange'].capitalize(), "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "timestamp": df.index[-2], "reason": reason_str, "strength": len(confirmed_reasons)})

                else:

                    logger.debug(f"Reject {symbol} Signal: Small TP/SL (TP: {tp_percent_calc:.2f}%, SL: {sl_percent_calc:.2f}%)")



        except ccxt.RateLimitExceeded as e:

            logger.warning(f"Rate limit exceeded for {symbol} on {market_info['exchange']}. Pausing...: {e}")

            await asyncio.sleep(10)

        except ccxt.NetworkError as e:

            logger.warning(f"Network error for {symbol}: {e}")

        except Exception as e:

            logger.error(f"CRITICAL ERROR in worker for {symbol}: {e}", exc_info=True)

            failure_counter[0] += 1

        finally:

            queue.task_done()



async def get_real_balance(exchange_id, currency='USDT'):

    try:

        exchange = bot_data["exchanges"].get(exchange_id.lower())

        if not exchange or not exchange.apiKey:

            logger.warning(f"Cannot fetch balance: {exchange_id.capitalize()} client not authenticated.")

            return 0.0



        balance = await exchange.fetch_balance()

        return balance['free'].get(currency, 0.0)

    except Exception as e:

        logger.error(f"Error fetching {exchange_id.capitalize()} balance for {currency}: {e}")

        return 0.0



async def place_real_trade(signal):

    exchange_id = signal['exchange'].lower()

    exchange = bot_data["exchanges"].get(exchange_id)

    settings = bot_data['settings']

    symbol = signal['symbol']



    if not exchange or not exchange.apiKey:

        return {'success': False, 'data': f"Client not authenticated for {exchange_id.capitalize()}."}



    try:

        usdt_balance = await get_real_balance(exchange_id, 'USDT')

        user_trade_amount_usdt = settings.get("real_trade_size_usdt", 15.0)



        # [جديد] تطبيق إدارة المخاطر الديناميكية

        autopilot = settings.get('autopilot_settings', {})

        if autopilot.get('dynamic_risk_enabled', False):

            mood = settings.get('last_market_mood', {}).get('mood', 'UNKNOWN')

            if mood == 'POSITIVE':

                multiplier = 1.0

            elif mood == 'NEUTRAL':

                multiplier = 0.6

            else: # NEGATIVE or DANGEROUS

                multiplier = 0.3

            user_trade_amount_usdt *= multiplier

            logger.info(f"Dynamic Risk is ON. Market mood is {mood}. Adjusted trade size to ${user_trade_amount_usdt:.2f}")



        # [تحسين] التحقق من الحد الأدنى لحجم الصفقة في المنصة

        markets = await exchange.load_markets()

        market_info = markets.get(symbol)

        if not market_info:

            return {'success': False, 'data': f"Could not find market info for {symbol}."}



        min_notional = 0

        if 'minNotional' in market_info.get('limits', {}).get('cost', {}):

             min_notional = market_info['limits']['cost']['minNotional']

        elif exchange_id == 'kucoin': # KuCoin has a different structure

            min_notional = float(market_info.get('info', {}).get('minProvideSize', 5.0))



        trade_amount_usdt = max(user_trade_amount_usdt, min_notional or 0)

        if min_notional and min_notional > user_trade_amount_usdt:

             logger.warning(f"User trade size ${user_trade_amount_usdt} for {symbol} is below exchange minimum of ${min_notional}. Using exchange minimum.")





        if usdt_balance < trade_amount_usdt:

            return {'success': False, 'data': f"رصيدك الحالي ${usdt_balance:.2f} غير كافٍ لفتح صفقة بقيمة ${trade_amount_usdt:.2f}."}

        

        quantity = trade_amount_usdt / signal['entry_price']

        formatted_quantity = exchange.amount_to_precision(symbol, quantity)

    except Exception as e:

        return {'success': False, 'data': f"Pre-flight check failed: {e}"}



    buy_order = None

    try:

        logger.info(f"Placing MARKET BUY order for {formatted_quantity} of {symbol} on {exchange_id.capitalize()}")

        buy_order = await exchange.create_market_buy_order(symbol, float(formatted_quantity))

        logger.info(f"Initial response for BUY order {buy_order.get('id', 'N/A')} received.")

    except Exception as e:

        logger.error(f"Placing BUY order for {symbol} failed immediately: {e}", exc_info=True)

        return {'success': False, 'data': f"حدث خطأ من المنصة عند محاولة الشراء: `{str(e)}`"}



    try:

        await asyncio.sleep(2) 

        verified_order = await exchange.fetch_order(buy_order['id'], symbol)

        

        if verified_order and verified_order.get('status') == 'closed' and verified_order.get('filled', 0) > 0:

            verified_price = verified_order.get('average', signal['entry_price'])

            verified_quantity = verified_order.get('filled')

            verified_cost = verified_order.get('cost', verified_price * verified_quantity)

            logger.info(f"BUY order {buy_order['id']} VERIFIED. Filled {verified_quantity} @ {verified_price}")

        else:

            raise Exception(f"Order {buy_order['id']} not confirmed as filled. Status: {verified_order.get('status')}")



    except Exception as e:

        logger.error(f"VERIFICATION FAILED for BUY order {buy_order.get('id', 'N/A')}: {e}", exc_info=True)

        return {'success': False, 'manual_check_required': True, 'data': f"تم إرسال أمر الشراء لكن فشل التحقق منه. **يرجى التحقق من المنصة يدوياً!** Order ID: `{buy_order.get('id', 'N/A')}`. Error: `{e}`"}



    # [ترقية أمان حرجة] منطق الخروج الموحد باستخدام OCO

    exit_order_ids = {}

    try:

        tp_price = exchange.price_to_precision(symbol, signal['take_profit'])

        sl_price = exchange.price_to_precision(symbol, signal['stop_loss'])

        sl_trigger_price = exchange.price_to_precision(symbol, signal['stop_loss'] * (1 - 0.001)) # Trigger just before the limit price



        # Binance has a dedicated OCO order type

        if exchange.id == 'binance':

            logger.info(f"Placing OCO for {symbol} on Binance. TP: {tp_price}, SL Trigger: {sl_trigger_price}, SL Limit: {sl_price}")

            oco_params = {'stopLimitPrice': sl_price}

            oco_order = await exchange.create_order(symbol, 'oco', 'sell', verified_quantity, price=tp_price, stopPrice=sl_trigger_price, params=oco_params)

            exit_order_ids = {"oco_id": oco_order['id']}

        # KuCoin supports OCO via params on a stop limit order

        elif exchange.id == 'kucoin':

            logger.info(f"Placing OCO for {symbol} on KuCoin. TP Trigger: {tp_price}, SL Trigger: {sl_trigger_price}, SL Limit: {sl_price}")

            params = {

                'stop': 'loss', # Specify stop loss

                'takeProfitPrice': tp_price # Add take profit trigger

            }

            oco_order = await exchange.create_order(symbol, 'stop_limit', 'sell', verified_quantity, price=sl_price, stopPrice=sl_trigger_price, params=params)

            exit_order_ids = {"oco_id": oco_order['id']}

        else:

            # Fallback for exchanges without unified OCO (less safe)

            logger.warning(f"Exchange {exchange.id} does not have a supported OCO method in this bot. Placing separate TP/SL orders.")

            tp_order = await exchange.create_limit_sell_order(symbol, verified_quantity, float(tp_price))

            sl_order = await exchange.create_stop_loss_order(symbol, 'sell', verified_quantity, float(sl_price)) # Assuming unified method

            exit_order_ids = {"tp_id": tp_order['id'], "sl_id": sl_order['id']}

        

        logger.info(f"Successfully placed exit orders for {symbol} with IDs: {exit_order_ids}")

    except Exception as e:

        logger.error(f"Failed to place exit orders for {symbol} after successful buy: {e}", exc_info=True)

        return {'success': True, 'exit_orders_failed': True, 'data': f"تم شراء {symbol} بنجاح، **لكن فشل وضع أوامر الخروج**. يرجى وضع الوقف والهدف يدوياً!"}



    return {

        'success': True,

        'data': {

            "entry_order_id": buy_order['id'],

            "exit_order_ids_json": json.dumps(exit_order_ids),

            "verified_quantity": verified_quantity,

            "verified_entry_price": verified_price,

            "verified_entry_value": verified_cost

        }

    }





async def perform_scan(context: ContextTypes.DEFAULT_TYPE):

    async with scan_lock:

        if bot_data['status_snapshot']['scan_in_progress']:

            logger.warning("Scan attempted while another was in progress. Skipped."); return

        settings = bot_data["settings"]

        if settings.get('fundamental_analysis_enabled', True):

            mood, mood_score, mood_reason = await get_fundamental_market_mood()

            bot_data['settings']['last_market_mood'] = {"timestamp": datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M'), "mood": mood, "reason": mood_reason}

            save_settings()

            logger.info(f"Fundamental Market Mood: {mood} - Reason: {mood_reason}")

            if mood in ["NEGATIVE", "DANGEROUS"]:

                await send_telegram_message(context.bot, {'custom_message': f"**⚠️ تم إيقاف الفحص التلقائي مؤقتاً**\n\n**السبب:** مزاج السوق سلبي/خطر.\n**التفاصيل:** {mood_reason}.\n\n*سيتم استئناف الفحص عندما تتحسن الظروف.*", 'target_chat': TELEGRAM_CHAT_ID}); return



        is_market_ok, btc_reason = await check_market_regime()

        bot_data['status_snapshot']['btc_market_mood'] = "إيجابي ✅" if is_market_ok else "سلبي ❌"



        if settings.get('market_regime_filter_enabled', True) and not is_market_ok:

            logger.info(f"Skipping scan: {btc_reason}")

            await send_telegram_message(context.bot, {'custom_message': f"**⚠️ تم إيقاف الفحص التلقائي مؤقتاً**\n\n**السبب:** مزاج السوق سلبي/خطر.\n**التفاصيل:** {btc_reason}.\n\n*سيتم استئناف الفحص عندما تتحسن الظروف.*", 'target_chat': TELEGRAM_CHAT_ID}); return



        status = bot_data['status_snapshot']

        status.update({"scan_in_progress": True, "last_scan_start_time": datetime.now(EGYPT_TZ)})

        

        try:

            conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة' AND trade_mode = 'virtual'")

            active_virtual_trades = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة' AND trade_mode = 'real'")

            active_real_trades = cursor.fetchone()[0]

            conn.close()

            active_trades_count = active_virtual_trades + active_real_trades

        except Exception as e:

            logger.error(f"DB Error in perform_scan: {e}"); active_trades_count = settings.get("max_concurrent_trades", 10)



        top_markets = await aggregate_top_movers()

        if not top_markets:

            logger.info("Scan complete: No markets to scan."); status['scan_in_progress'] = False; return



        queue = asyncio.Queue(); [await queue.put(market) for market in top_markets]

        signals, failure_counter = [], [0]

        worker_tasks = [asyncio.create_task(worker(queue, signals, settings, failure_counter)) for _ in range(settings['concurrent_workers'])]

        await queue.join(); [task.cancel() for task in worker_tasks]



        total_signals_found = len(signals)



        signals.sort(key=lambda s: s.get('strength', 0), reverse=True)

        new_trades, opportunities = 0, 0

        last_signal_time = bot_data['last_signal_time']



        for signal in signals:

            if time.time() - last_signal_time.get(signal['symbol'], 0) <= (SCAN_INTERVAL_SECONDS * 4):

                logger.info(f"Signal for {signal['symbol']} skipped due to cooldown."); continue



            signal_exchange_id = signal['exchange'].lower()

            per_exchange_settings = settings.get("real_trading_per_exchange", {})

            is_real_mode_enabled = per_exchange_settings.get(signal_exchange_id, False)



            exchange_is_tradeable = signal_exchange_id in bot_data["exchanges"] and bot_data["exchanges"][signal_exchange_id].apiKey

            attempt_real_trade = is_real_mode_enabled and exchange_is_tradeable

            signal['is_real_trade'] = attempt_real_trade



            if attempt_real_trade:

                await send_telegram_message(context.bot, {'custom_message': f"**🔎 تم العثور على إشارة حقيقية لـ `{signal['symbol']}`... جاري محاولة التنفيذ على `{signal['exchange']}`.**"})

                trade_result = await place_real_trade(signal)

                

                if trade_result['success']:

                    signal.update(trade_result['data'])

                    if log_recommendation_to_db(signal):

                        await send_telegram_message(context.bot, signal, is_new=True) # This will now be detailed

                        new_trades += 1

                    else: 

                        await send_telegram_message(context.bot, {'custom_message': f"**⚠️ خطأ حرج:** تم تنفيذ صفقة `{signal['symbol']}` لكن فشل تسجيلها. **يرجى المتابعة اليدوية فوراً!**\nتفاصيل: `{trade_result['data']}`"})

                else:

                    await send_telegram_message(context.bot, {'custom_message': f"**❌ فشل تنفيذ صفقة `{signal['symbol']}`**\n\n**السبب:** {trade_result['data']}"})



            else: 

                if active_trades_count < settings.get("max_concurrent_trades", 10):

                    trade_amount_usdt = settings["virtual_portfolio_balance_usdt"] * (settings["virtual_trade_size_percentage"] / 100)

                    signal.update({'quantity': trade_amount_usdt / signal['entry_price'], 'entry_value_usdt': trade_amount_usdt})

                    if trade_id := log_recommendation_to_db(signal):

                        signal['trade_id'] = trade_id

                        await send_telegram_message(context.bot, signal, is_new=True) # Virtual trades will also be sent now

                        new_trades += 1

                else:

                    await send_telegram_message(context.bot, signal, is_opportunity=True)

                    opportunities += 1



            await asyncio.sleep(0.5)

            last_signal_time[signal['symbol']] = time.time()



        failures = failure_counter[0]

        logger.info(f"Scan complete. Found: {total_signals_found}, Entered: {new_trades}, Opportunities: {opportunities}, Failures: {failures}.")

        

        status['last_scan_end_time'] = datetime.now(EGYPT_TZ)

        scan_start_time = status.get('last_scan_start_time')

        scan_duration = (status['last_scan_end_time'] - scan_start_time).total_seconds() if isinstance(scan_start_time, datetime) else 0



        summary_message = (f"**🔬 ملخص الفحص الأخير**\n\n"

                           f"- **الحالة:** اكتمل بنجاح\n"

                           f"- **وضع السوق (BTC):** {status['btc_market_mood']}\n"

                           f"- **المدة:** {scan_duration:.0f} ثانية\n"

                           f"- **العملات المفحوصة:** {len(top_markets)}\n\n"

                           f"- - - - - - - - - - - - - - - - - -\n"

                           f"- **إجمالي الإشارات المكتشفة:** {total_signals_found}\n"

                           f"- **✅ صفقات جديدة فُتحت:** {new_trades}\n"

                           f"- **💡 فرص للمراقبة:** {opportunities}\n"

                           f"- **⚠️ أخطاء في التحليل:** {failures}\n"

                           f"- - - - - - - - - - - - - - - - - -\n\n"

                           f"*الفحص التالي مجدول تلقائياً.*")



        await send_telegram_message(context.bot, {'custom_message': summary_message, 'target_chat': TELEGRAM_CHAT_ID})



        status['scan_in_progress'] = False



        bot_data['scan_history'].append({'signals': total_signals_found, 'failures': failures})

        if settings.get('autopilot_settings', {}).get('smart_preset_switching_enabled', True):

            await analyze_performance_and_suggest(context)



# [ميزة] نظام الإشعارات الآمن + رسائل مفصلة

async def send_telegram_message(bot, signal_data, is_new=False, is_opportunity=False, update_type=None):

    message, keyboard, target_chat = "", None, TELEGRAM_CHAT_ID

    def format_price(price): 

        if price is None: return "N/A"

        return f"{price:,.8f}" if price < 0.01 else f"{price:,.4f}"



    if 'custom_message' in signal_data:

        message, target_chat = signal_data['custom_message'], signal_data.get('target_chat', TELEGRAM_CHAT_ID)

        if 'keyboard' in signal_data: keyboard = signal_data['keyboard']



    elif is_new:

        target_chat = TELEGRAM_SIGNAL_CHANNEL_ID

        is_real = signal_data.get('is_real_trade', False)

        

        # [ميزة] تصميم الرسالة المفصلة للصفقات الحقيقية

        if is_real:

            entry = signal_data.get('verified_entry_price', signal_data['entry_price'])

            value = signal_data.get('verified_entry_value', signal_data['entry_value_usdt'])

            tp, sl = signal_data['take_profit'], signal_data['stop_loss']

            tp_percent, sl_percent = ((tp - entry) / entry * 100), ((entry - sl) / entry * 100)

            

            total_portfolio = await get_total_real_portfolio_value_usdt()

            portfolio_percent = (value / total_portfolio * 100) if total_portfolio > 0 else 0



            message = (

                f"**🚨 تم تنفيذ صفقة حقيقية بنجاح 🚨**\n"

                f"------------------------------------\n"

                f"**عملة:** `{signal_data['symbol']}`\n"

                f"**منصة:** `{signal_data['exchange']}`\n"

                f"------------------------------------\n"

                f"**📈 سعر التنفيذ الفعلي:** `{format_price(entry)}`\n"

                f"**💵 القيمة الإجمالية:** `${value:.2f}`\n"

                f"**📊 نسبة من المحفظة:** `{portfolio_percent:.2f}%`\n"

                f"------------------------------------\n"

                f"**🎯 الهدف:** `{format_price(tp)}` (+{tp_percent:.2f}%)\n"

                f"**🛑 الوقف:** `{format_price(sl)}` (-{sl_percent:.2f}%)"

            )

        else: # الرسالة العادية للصفقات الوهمية والفرص

            title = "**📈 صفقة وهمية جديدة**" if not is_opportunity else "**💡 فرصة محتملة**"

            entry, tp, sl = signal_data['entry_price'], signal_data['take_profit'], signal_data['stop_loss']

            tp_percent, sl_percent = ((tp - entry) / entry * 100), ((entry - sl) / entry * 100)

            reasons_en = signal_data['reason'].split(' + ')

            reasons_ar = ' + '.join([STRATEGY_NAMES_AR.get(r, r) for r in reasons_en])

            

            message = (

                f"**{title} | {signal_data['symbol']}**\n"

                f"------------------------------------\n"

                f"🔹 **المنصة:** {signal_data['exchange']}\n"

                f"🔍 **الاستراتيجية:** {reasons_ar}\n\n"

                f"📈 **نقطة الدخول:** `{format_price(entry)}`\n"

                f"🎯 **الهدف:** `{format_price(tp)}` (+{tp_percent:.2f}%)\n"

                f"🛑 **الوقف:** `{format_price(sl)}` (-{sl_percent:.2f}%)"

            )



    elif is_opportunity:

         target_chat = TELEGRAM_SIGNAL_CHANNEL_ID

         entry, tp, sl = signal_data['entry_price'], signal_data['take_profit'], signal_data['stop_loss']

         tp_percent, sl_percent = ((tp - entry) / entry * 100), ((entry - sl) / entry * 100)

         reasons_en = signal_data['reason'].split(' + ')

         reasons_ar = ' + '.join([STRATEGY_NAMES_AR.get(r, r) for r in reasons_en])

         message = (

            f"**💡 فرصة محتملة | {signal_data['symbol']}**\n"

            f"------------------------------------\n"

            f"🔹 **المنصة:** {signal_data['exchange']}\n"

            f"🔍 **الاستراتيجية:** {reasons_ar}\n\n"

            f"📈 **نقطة الدخول:** `{format_price(entry)}`\n"

            f"🎯 **الهدف:** `{format_price(tp)}` (+{tp_percent:.2f}%)\n"

            f"🛑 **الوقف:** `{format_price(sl)}` (-{sl_percent:.2f}%)"

        )

    elif update_type == 'tsl_activation':

        message = (f"**🚀 تأمين الأرباح! | #{signal_data['id']} {signal_data['symbol']}**\n\n"

                   f"تم رفع وقف الخسارة إلى نقطة الدخول.\n"

                   f"**هذه الصفقة الآن مؤمَّنة بالكامل وبدون مخاطرة!**\n\n"

                   f"*دع الأرباح تنمو!*")

    elif update_type == 'tsl_update_real_auto':

        message = (f"**✅ تم تحديث الوقف المتحرك تلقائيًا**\n\n"

                   f"**صفقة:** `#{signal_data['id']} {signal_data['symbol']}`\n"

                   f"**الوقف الجديد:** `{format_price(signal_data['new_sl'])}`")

    elif update_type == 'tsl_update_real':

        message = (f"**🔔 تنبيه تحديث وقف الخسارة (صفقة حقيقية) 🔔**\n\n"

                   f"**صفقة:** `#{signal_data['id']} {signal_data['symbol']}`\n\n"

                   f"وصل السعر إلى `{format_price(signal_data['current_price'])}`.\n"

                   f"**إجراء مقترح:** قم بتعديل أمر وقف الخسارة يدوياً إلى `{format_price(signal_data['new_sl'])}` لتأمين الأرباح.")





    if not message: return

    

    # [ميزة] نظام الإشعارات الآمن

    try:

        await bot.send_message(chat_id=target_chat, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)

    except Exception as e:

        logger.error(f"Failed to send message to target_chat {target_chat}: {e}")

        # إذا فشل الإرسال إلى القناة، أرسل إلى المحادثة الخاصة

        if str(target_chat) == str(TELEGRAM_SIGNAL_CHANNEL_ID) and str(target_chat) != str(TELEGRAM_CHAT_ID):

            try:

                warning_text = "⚠️ **فشل الإرسال إلى القناة!** تم إرسال الإشعار هنا بدلاً من ذلك. يرجى التحقق من صلاحيات البوت في القناة.\n\n"

                await bot.send_message(

                    chat_id=TELEGRAM_CHAT_ID,

                    text=warning_text + message,

                    parse_mode=ParseMode.MARKDOWN,

                    reply_markup=keyboard

                )

            except Exception as admin_e:

                logger.error(f"Failed to send fallback message to admin chat: {admin_e}")



async def track_open_trades(context: ContextTypes.DEFAULT_TYPE):

    try:

        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute("SELECT * FROM trades WHERE status = 'نشطة'")

        active_trades = [dict(row) for row in cursor.fetchall()]; conn.close()

    except Exception as e: logger.error(f"DB error in track_open_trades: {e}"); return

    bot_data['status_snapshot']['active_trades_count'] = len(active_trades)

    if not active_trades: return



    for trade in active_trades:

        exchange_id = trade['exchange'].lower()

        exchange = bot_data["exchanges"].get(exchange_id)

        if not exchange:

            logger.warning(f"Exchange {trade['exchange']} not found for tracking trade #{trade['id']}.")

            continue



        try:

            public_exchange = bot_data['public_exchanges'].get(exchange_id)

            ticker = await public_exchange.fetch_ticker(trade['symbol'])

            current_price = ticker.get('last') or ticker.get('close')

            if not current_price:

                logger.warning(f"Could not fetch price for {trade['symbol']} on {trade['exchange']}")

                continue



            current_stop_loss = trade.get('stop_loss') or 0

            current_take_profit = trade.get('take_profit')



            if current_take_profit is not None and current_price >= current_take_profit:

                await close_trade_in_db(context, trade, current_price, 'ناجحة')

                continue

            if current_stop_loss > 0 and current_price <= current_stop_loss:

                await close_trade_in_db(context, trade, current_price, 'فاشلة')

                continue



            settings = bot_data["settings"]

            if settings.get('trailing_sl_enabled', True):

                highest_price = max(trade.get('highest_price', current_price) or current_price, current_price)

                

                if not trade.get('trailing_sl_active'):

                    activation_price = trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100)

                    if current_price >= activation_price:

                        new_sl = trade['entry_price']

                        if new_sl > current_stop_loss:

                            is_real_automated = trade.get('trade_mode') == 'real' and settings.get('automate_real_tsl', False)

                            if is_real_automated:

                                await update_real_trade_sl(context, exchange, trade, new_sl, highest_price, is_activation=True)

                            elif trade.get('trade_mode') == 'real': # Manual notification

                                await send_telegram_message(context.bot, {**trade, "new_sl": new_sl, "current_price": current_price}, update_type='tsl_update_real')

                                await update_trade_sl_in_db(context, trade, new_sl, highest_price, is_activation=True, silent=True)

                            else: # Virtual trade

                                await update_trade_sl_in_db(context, trade, new_sl, highest_price, is_activation=True)

                

                elif trade.get('trailing_sl_active'):

                    new_sl = highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)

                    if new_sl > current_stop_loss:

                        is_real_automated = trade.get('trade_mode') == 'real' and settings.get('automate_real_tsl', False)

                        if is_real_automated:

                            await update_real_trade_sl(context, exchange, trade, new_sl, highest_price)

                        elif trade.get('trade_mode') == 'real': # Manual notification

                            await send_telegram_message(context.bot, {**trade, "new_sl": new_sl, "current_price": current_price}, update_type='tsl_update_real')

                            await update_trade_sl_in_db(context, trade, new_sl, highest_price, silent=True)

                        else: # Virtual trade

                             await update_trade_sl_in_db(context, trade, new_sl, highest_price)

                

                if highest_price > (trade.get('highest_price') or 0):

                    await update_trade_peak_price_in_db(trade['id'], highest_price)



        except Exception as e:

            logger.error(f"Error tracking trade #{trade['id']} ({trade['symbol']}): {e}", exc_info=True)



# [جديد] دالة لأتمتة تحديث الوقف المتحرك الحقيقي

async def update_real_trade_sl(context, exchange, trade, new_sl, highest_price, is_activation=False):

    symbol = trade['symbol']

    logger.info(f"AUTOMATING TSL UPDATE for real trade #{trade['id']} ({symbol}). New SL: {new_sl}")

    

    try:

        # 1. إلغاء الأمر الحالي

        exit_ids = json.loads(trade.get('exit_order_ids_json', '{}'))

        if not (oco_id := exit_ids.get('oco_id')):

            raise ValueError("No OCO ID found to cancel for automated TSL.")

        

        await exchange.cancel_order(oco_id, symbol)

        logger.info(f"Successfully cancelled old OCO order {oco_id} for {symbol}.")

        await asyncio.sleep(1) # تأخير بسيط لضمان الإلغاء



        # 2. إنشاء الأمر الجديد

        tp_price = exchange.price_to_precision(symbol, trade['take_profit'])

        sl_price = exchange.price_to_precision(symbol, new_sl)

        sl_trigger_price = exchange.price_to_precision(symbol, new_sl * (1 - 0.001))

        quantity = trade['quantity']

        

        new_exit_ids = {}

        if exchange.id == 'binance':

            oco_params = {'stopLimitPrice': sl_price}

            new_oco_order = await exchange.create_order(symbol, 'oco', 'sell', quantity, price=tp_price, stopPrice=sl_trigger_price, params=oco_params)

            new_exit_ids = {"oco_id": new_oco_order['id']}

        elif exchange.id == 'kucoin':

            params = {'stop': 'loss', 'takeProfitPrice': tp_price}

            new_oco_order = await exchange.create_order(symbol, 'stop_limit', 'sell', quantity, price=sl_price, stopPrice=sl_trigger_price, params=params)

            new_exit_ids = {"oco_id": new_oco_order['id']}

        else:

            raise NotImplementedError(f"Automated TSL not implemented for {exchange.id}")



        logger.info(f"Successfully created new OCO order {new_exit_ids.get('oco_id')} for {symbol}.")



        # 3. تحديث قاعدة البيانات وإرسال إشعار

        await update_trade_sl_in_db(context, trade, new_sl, highest_price, is_activation=is_activation, new_exit_ids_json=json.dumps(new_exit_ids), silent=True)

        await send_telegram_message(context.bot, {**trade, "new_sl": new_sl}, update_type='tsl_update_real_auto')





    except Exception as e:

        logger.critical(f"CRITICAL FAILURE in automated TSL for trade #{trade['id']} ({symbol}): {e}", exc_info=True)

        await send_telegram_message(context.bot, {'custom_message': f"**🚨 فشل حرج في أتمتة الوقف المتحرك 🚨**\n\n**صفقة:** `#{trade['id']} {symbol}`\n**الخطأ:** `{e}`\n\n**قد تكون الصفقة الآن بدون حماية! يرجى المتابعة اليدوية فوراً!**"})





async def close_trade_in_db(context: ContextTypes.DEFAULT_TYPE, trade: dict, exit_price: float, status: str):

    pnl_usdt = (exit_price - trade['entry_price']) * trade['quantity']

    if trade.get('trade_mode') == 'virtual':

        bot_data['settings']['virtual_portfolio_balance_usdt'] += pnl_usdt

        save_settings()



    closed_at_str = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')

    start_dt_naive = datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S')

    # [إصلاح حاسم] استخدام الطريقة الصحيحة لجعل التاريخ على دراية بالمنطقة الزمنية

    start_dt = start_dt_naive.replace(tzinfo=EGYPT_TZ)

    end_dt = datetime.now(EGYPT_TZ)

    duration = end_dt - start_dt

    days, remainder = divmod(duration.total_seconds(), 86400)

    hours, remainder = divmod(remainder, 3600)

    minutes, _ = divmod(remainder, 60)

    duration_str = f"{int(days)}d {int(hours)}h {int(minutes)}m" if days > 0 else f"{int(hours)}h {int(minutes)}m"



    try:

        conn = sqlite3.connect(DB_FILE, timeout=10)

        cursor = conn.cursor()

        cursor.execute("UPDATE trades SET status=?, exit_price=?, closed_at=?, exit_value_usdt=?, pnl_usdt=? WHERE id=?",

                       (status, exit_price, closed_at_str, exit_price * trade['quantity'], pnl_usdt, trade['id']))

        conn.commit()

        conn.close()

    except Exception as e:

        logger.error(f"DB update failed while closing trade #{trade['id']}: {e}")

        return

    

    trade_type_str = "(صفقة حقيقية)" if trade.get('trade_mode') == 'real' else ""

    pnl_percent = (pnl_usdt / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0

    message = ""

    if status == 'ناجحة':

        message = (f"**📦 إغلاق صفقة {trade_type_str} | #{trade['id']} {trade['symbol']}**\n\n"

                   f"**الحالة: ✅ ناجحة (تم تحقيق الهدف)**\n"

                   f"💰 **الربح:** `${pnl_usdt:+.2f}` (`{pnl_percent:+.2f}%`)\n\n"

                   f"- **مدة الصفقة:** {duration_str}")

    else: 

        message = (f"**📦 إغلاق صفقة {trade_type_str} | #{trade['id']} {trade['symbol']}**\n\n"

                   f"**الحالة: ❌ فاشلة (تم ضرب الوقف)**\n"

                   f"💰 **الخسارة:** `${pnl_usdt:.2f}` (`{pnl_percent:.2f}%`)\n\n"

                   f"- **مدة الصفقة:** {duration_str}")



    await send_telegram_message(context.bot, {'custom_message': message, 'target_chat': TELEGRAM_SIGNAL_CHANNEL_ID})



async def update_trade_sl_in_db(context: ContextTypes.DEFAULT_TYPE, trade: dict, new_sl: float, highest_price: float, is_activation: bool = False, silent: bool = False, new_exit_ids_json: str = None):

    try:

        conn = sqlite3.connect(DB_FILE, timeout=10)

        cursor = conn.cursor()

        

        sql = "UPDATE trades SET stop_loss=?, highest_price=?, trailing_sl_active=? "

        params = [new_sl, highest_price, True]

        

        if new_exit_ids_json:

            sql += ", exit_order_ids_json=? "

            params.append(new_exit_ids_json)



        sql += "WHERE id=?"

        params.append(trade['id'])



        cursor.execute(sql, tuple(params))

        conn.commit()

        conn.close()

        

        log_msg = f"Trailing SL {'activated' if is_activation else 'updated'} for trade #{trade['id']}. New SL: {new_sl}"

        if new_exit_ids_json:

            log_msg += f", New Exit IDs: {new_exit_ids_json}"

        logger.info(log_msg)



        if not silent and is_activation:

            await send_telegram_message(context.bot, {**trade, "new_sl": new_sl}, update_type='tsl_activation')

    except Exception as e:

        logger.error(f"Failed to update SL for trade #{trade['id']} in DB: {e}")



async def update_trade_peak_price_in_db(trade_id: int, highest_price: float):

    try:

        conn = sqlite3.connect(DB_FILE, timeout=10)

        cursor = conn.cursor()

        cursor.execute("UPDATE trades SET highest_price=? WHERE id=?", (highest_price, trade_id))

        conn.commit()

        conn.close()

    except Exception as e:

        logger.error(f"Failed to update peak price for trade #{trade_id} in DB: {e}")





async def get_fear_and_greed_index():

    try:

        async with httpx.AsyncClient() as client:

            response = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)

            response.raise_for_status()

            if data := response.json().get('data', []):

                return int(data[0]['value'])

    except Exception as e:

        logger.error(f"Could not fetch Fear and Greed Index: {e}")

    return None



async def check_market_regime():

    settings = bot_data['settings']

    is_technically_bullish, is_sentiment_bullish, fng_index = True, True, "N/A"

    try:

        if binance := bot_data["public_exchanges"].get('binance'):

            ohlcv = await binance.fetch_ohlcv('BTC/USDT', '4h', limit=55)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            df['sma50'] = ta.sma(df['close'], length=50)

            is_technically_bullish = df['close'].iloc[-1] > df['sma50'].iloc[-1]

    except Exception as e:

        logger.error(f"Error checking BTC trend: {e}")

    if settings.get("fear_and_greed_filter_enabled", True):

        if (fng_value := await get_fear_and_greed_index()) is not None:

            fng_index = fng_value

            is_sentiment_bullish = fng_index >= settings.get("fear_and_greed_threshold", 30)

    if not is_technically_bullish:

        return False, "اتجاه BTC هابط (تحت متوسط 50 على 4 ساعات)."

    if not is_sentiment_bullish:

        return False, f"مشاعر خوف شديد (مؤشر F&G: {fng_index} تحت الحد {settings.get('fear_and_greed_threshold')})."

    return True, "وضع السوق مناسب لصفقات الشراء."



async def analyze_performance_and_suggest(context: ContextTypes.DEFAULT_TYPE):

    settings = bot_data['settings']

    history = bot_data['scan_history']



    if len(history) < 5 or (time.time() - settings.get('last_suggestion_time', 0)) < 7200:

        return



    avg_signals = sum(item['signals'] for item in history) / len(history)

    current_preset = settings.get('active_preset_name', 'PRO')



    suggestion, market_desc, reason = None, None, None



    if avg_signals < 0.5 and current_preset == "STRICT":

        suggestion = "PRO"

        market_desc = "السوق يبدو بطيئاً جداً والإشارات شحيحة."

        reason = "نمط 'PRO' أكثر توازناً وقد يساعدنا في التقاط المزيد من الفرص المناسبة دون التضحية بالكثير من الجودة."

    elif avg_signals < 1 and current_preset == "PRO":

        suggestion = "LAX"

        market_desc = "عدد الفرص المكتشفة منخفض نسبياً."

        reason = "نمط 'LAX' (متساهل) سيوسع نطاق البحث، مما قد يزيد من عدد الإشارات في سوق هادئ."

    elif avg_signals > 8 and current_preset in ["LAX", "VERY_LAX"]:

        suggestion = "PRO"

        market_desc = "السوق نشط جداً وهناك عدد كبير من الإشارات (ضوضاء)."

        reason = "نمط 'PRO' سيساعد في فلترة الإشارات الأضعف والتركيز على الفرص ذات الجودة الأعلى."

    elif avg_signals > 12 and current_preset == "PRO":

        suggestion = "STRICT"

        market_desc = "السوق متقلب وهناك فيضان من الإشارات."

        reason = "نمط 'STRICT' (متشدد) سيطبق أقوى الفلاتر لاصطياد أفضل الفرص فقط في هذا السوق المتقلب."



    if suggestion and suggestion != current_preset:

        message = (f"**💡 اقتراح ذكي لتحسين الأداء**\n\n"

                   f"*مرحباً! بناءً على تحليل آخر {len(history)} فحص، لاحظت تغيراً في طبيعة السوق.*\n\n"

                   f"**الملاحظة:**\n- {market_desc}\n\n"

                   f"**الاقتراح:**\n- أقترح تغيير نمط الإعدادات من `{current_preset}` إلى **`{suggestion}`**.\n\n"

                   f"**السبب:**\n- {reason}\n\n"

                   f"*هل توافق على تطبيق هذا التغيير؟*")



        keyboard = InlineKeyboardMarkup([

            [InlineKeyboardButton("✅ نعم، قم بتطبيق النمط المقترح", callback_data=f"suggest_accept_{suggestion}")],

            [InlineKeyboardButton("❌ لا شكراً، تجاهل الاقتراح", callback_data="suggest_decline")]

        ])



        await send_telegram_message(context.bot, {'custom_message': message, 'keyboard': keyboard})

        bot_data['settings']['last_suggestion_time'] = time.time()

        save_settings()





# --- Reports and Telegram Commands ---

def generate_performance_report_string(trade_mode_filter='all'):

    REPORT_DAYS = 30

    if not os.path.exists(DB_FILE): return "❌ خطأ: لم يتم العثور على ملف قاعدة البيانات."

    try:

        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row; cursor = conn.cursor()

        start_date = (datetime.now() - timedelta(days=REPORT_DAYS)).strftime('%Y-%m-%d %H:%M:%S')

        

        query = "SELECT reason, status, entry_price, highest_price FROM trades WHERE status IN ('ناجحة', 'فاشلة') AND timestamp >= ?"

        params = [start_date]

        if trade_mode_filter != 'all':

            query += " AND trade_mode = ?"

            params.append(trade_mode_filter)



        cursor.execute(query, params); trades = cursor.fetchall(); conn.close()

    except Exception as e: return f"❌ حدث خطأ غير متوقع: {e}"

    if not trades: return f"ℹ️ لا توجد صفقات مغلقة في آخر {REPORT_DAYS} يومًا لهذا الفلتر."

    

    strategy_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'max_profits': []})

    for trade in trades:

        reasons = (trade['reason'] or 'unknown').split(' + ')

        for reason in reasons:

            stats = strategy_stats[reason]

            stats['total'] += 1

            if trade['status'] == 'ناجحة': stats['successful'] += 1

            if trade['entry_price'] is not None and trade['entry_price'] > 0 and trade['highest_price'] is not None:

                stats['max_profits'].append(((trade['highest_price'] - trade['entry_price']) / trade['entry_price']) * 100)

    

    report_lines = [f"📊 **تقرير أداء الاستراتيجيات (آخر {REPORT_DAYS} يومًا)** 📊", "="*35]

    for reason_en, stats in sorted(strategy_stats.items(), key=lambda item: item[1]['total'], reverse=True):

        reason_ar = STRATEGY_NAMES_AR.get(reason_en, reason_en)

        if total_trades := stats['total']:

            success_rate = (stats['successful'] / total_trades) * 100

            avg_max_profit = sum(stats['max_profits']) / len(stats['max_profits']) if stats['max_profits'] else 0

            report_lines.extend([f"--- **{reason_ar}** ---", f"- **إجمالي التوصيات:** {total_trades}", f"- **نسبة النجاح:** {success_rate:.1f}%", f"- **متوسط أقصى ربح:** {avg_max_profit:.2f}%", ""])

    return "\n".join(report_lines)



main_menu_keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"], ["ℹ️ مساعدة"]]

settings_menu_keyboard = [

    ["🏁 أنماط جاهزة", "🎭 تفعيل/تعطيل الماسحات"], 

    ["🔧 تعديل المعايير", "🚨 التحكم بالتداول الحقيقي"],

    ["🧠 التحكم الذاتي", "🗄️ إدارة قاعدة البيانات"],

    ["🔙 القائمة الرئيسية"]

]



async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    welcome_message = "💣 أهلاً بك في بوت **كاسحة الألغام**!\n\n*(الإصدار 5.0 - العقل المدبر)*\n\nاختر من القائمة للبدء."

    await update.message.reply_text(welcome_message, reply_markup=ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True), parse_mode=ParseMode.MARKDOWN)



async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    target_message = update.message or update.callback_query.message

    keyboard = InlineKeyboardMarkup([

        [InlineKeyboardButton("📊 الإحصائيات العامة", callback_data="dashboard_stats"), InlineKeyboardButton("📈 الصفقات النشطة", callback_data="dashboard_active_trades")],

        [InlineKeyboardButton("📜 تقرير أداء الاستراتيجيات", callback_data="dashboard_strategy_report")],

        [InlineKeyboardButton("📸 لقطة للمحفظة", callback_data="dashboard_snapshot"), InlineKeyboardButton("ρίск تقرير المخاطر", callback_data="dashboard_risk")],

        [InlineKeyboardButton("🔄 مزامنة ومطابقة المحفظة", callback_data="dashboard_sync")],

        [InlineKeyboardButton("🛠️ أدوات التداول", callback_data="dashboard_tools"), InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="dashboard_debug")],

        [InlineKeyboardButton("🔄 تحديث", callback_data="dashboard_refresh")]

    ])

    message_text = "🖥️ *لوحة التحكم الرئيسية*\n\nاختر التقرير أو البيانات التي تريد عرضها:"



    try:

        if update.callback_query:

             await target_message.edit_text(message_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)

        else:

            await target_message.reply_text(message_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)

    except BadRequest as e:

        if "Message is not modified" in str(e):

            pass 

        else:

            logger.error(f"Error in show_dashboard_command: {e}")

            if update.callback_query:

                await context.bot.send_message(chat_id=target_message.chat_id, text=message_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)





async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE): await (update.message or update.callback_query.message).reply_text("اختر الإعداد:", reply_markup=ReplyKeyboardMarkup(settings_menu_keyboard, resize_keyboard=True))



def get_scanners_keyboard():

    active_scanners = bot_data["settings"].get("active_scanners", [])

    keyboard = [[InlineKeyboardButton(f"{'✅' if name in active_scanners else '❌'} {STRATEGY_NAMES_AR.get(name, name)}", callback_data=f"toggle_scanner_{name}")] for name in STRATEGY_NAMES_AR.keys()]

    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])

    return InlineKeyboardMarkup(keyboard)



def get_presets_keyboard():

    return InlineKeyboardMarkup([

        [InlineKeyboardButton("🚦 احترافية (متوازنة)", callback_data="preset_PRO"), InlineKeyboardButton("🎯 متشددة", callback_data="preset_STRICT")],

        [InlineKeyboardButton("🌙 متساهلة", callback_data="preset_LAX"), InlineKeyboardButton("⚠️ فائق التساهل", callback_data="preset_VERY_LAX")],

        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")]

    ])

    

# [جديد] دالة لعرض قائمة التحكم بالتداول الحقيقي

async def show_real_trading_control_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):

    target_message = update.message or update.callback_query.message

    settings = bot_data["settings"].get("real_trading_per_exchange", {})

    keyboard = []

    for ex_id in EXCHANGES_TO_SCAN:

        is_enabled = settings.get(ex_id, False)

        status_emoji = '✅' if is_enabled else '❌'

        button_text = f"{status_emoji} {ex_id.capitalize()}"

        callback_data = f"toggle_real_trade_{ex_id}"

        keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])

    

    await target_message.reply_text(

        "**🚨 التحكم بالتداول الحقيقي 🚨**\n\nاختر المنصة لتفعيل أو تعطيل التداول عليها:",

        reply_markup=InlineKeyboardMarkup(keyboard),

        parse_mode=ParseMode.MARKDOWN

    )



async def show_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):

    target_message = update.message or update.callback_query.message

    await target_message.reply_text("اختر نمط إعدادات جاهز:", reply_markup=get_presets_keyboard())

async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):

    target_message = update.message or update.callback_query.message

    await target_message.reply_text("اختر الماسحات لتفعيلها أو تعطيلها:", reply_markup=get_scanners_keyboard())

async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):

    keyboard, settings = [], bot_data["settings"]

    for category, params in EDITABLE_PARAMS.items():

        keyboard.append([InlineKeyboardButton(f"--- {category} ---", callback_data="ignore")])

        for row in [params[i:i + 2] for i in range(0, len(params), 2)]:

            button_row = []

            for param_key in row:

                display_name = PARAM_DISPLAY_NAMES.get(param_key, param_key)

                current_value = settings.get(param_key, "N/A")

                text = f"{display_name}: {'مُفعّل ✅' if current_value else 'مُعطّل ❌'}" if isinstance(current_value, bool) else f"{display_name}: {current_value}"

                button_row.append(InlineKeyboardButton(text, callback_data=f"param_{param_key}"))

            keyboard.append(button_row)

    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])

    message_text = "⚙️ *الإعدادات المتقدمة* ⚙️\n\nاختر الإعداد الذي تريد تعديله بالضغط عليه:"

    target_message = update.callback_query.message if update.callback_query else update.message

    try:

        if update.callback_query:

            await target_message.edit_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

        else:

            sent_message = await target_message.reply_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

            context.user_data['settings_menu_id'] = sent_message.message_id

    except BadRequest as e:

        if "Message is not modified" not in str(e): logger.error(f"Error editing parameters menu: {e}")



async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    help_text = (

        "**💣 أوامر بوت كاسحة الألغام 💣**\n\n"

        "`/start` - لعرض القائمة الرئيسية وبدء التفاعل.\n"

        "`/check <ID>` - لمتابعة حالة صفقة معينة باستخدام رقمها.\n"

        "`/trade` - لبدء عملية تداول يدوية لاختبار الاتصال بالمنصات."

    )

    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)



# [إصلاح حاسم] تعديل دوال التقارير لترجع المحتوى

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_mode_filter='all'):

    try:

        conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor();

        

        query = "SELECT status, COUNT(*), SUM(pnl_usdt) FROM trades"

        params = []

        

        filter_conditions = []

        if trade_mode_filter != 'all':

            filter_conditions.append("trade_mode = ?")

            params.append(trade_mode_filter)



        if filter_conditions:

            query += " WHERE " + " AND ".join(filter_conditions)



        query += " GROUP BY status"

        cursor.execute(query, params)

        

        stats_data = cursor.fetchall(); conn.close()

        counts = {s: c for s, c, p in stats_data}; pnl = {s: (p or 0) for s, c, p in stats_data}

        total, active, successful, failed = sum(counts.values()), counts.get('نشطة', 0), counts.get('ناجحة', 0), counts.get('فاشلة', 0)

        closed = successful + failed; win_rate = (successful / closed * 100) if closed > 0 else 0; total_pnl = sum(pnl.values())

        preset_name = bot_data["settings"].get("active_preset_name", "N/A")

        

        mode_title_map = {'all': '(الكل)', 'real': '(حقيقي فقط)', 'virtual': '(وهمي فقط)'}

        title = mode_title_map.get(trade_mode_filter, '')



        # [إصلاح حاسم] عرض الرصيد الحقيقي أو الوهمي بناءً على الفلتر

        balance_lines = []

        if trade_mode_filter == 'real':

            real_balance = await get_total_real_portfolio_value_usdt()

            balance_lines.append(f"💰 *إجمالي قيمة المحفظة الحقيقية:* `${real_balance:.2f}`")

        elif trade_mode_filter == 'virtual':

            balance_lines.append(f"📈 *الرصيد الافتراضي:* `${bot_data['settings']['virtual_portfolio_balance_usdt']:.2f}`")

        else: # 'all'

            real_balance = await get_total_real_portfolio_value_usdt()

            balance_lines.append(f"💰 *قيمة المحفظة الحقيقية:* `${real_balance:.2f}`")

            balance_lines.append(f"📈 *الرصيد الافتراضي:* `${bot_data['settings']['virtual_portfolio_balance_usdt']:.2f}`")



        balance_section = "\n".join(balance_lines)



        stats_msg = (f"*📊 إحصائيات المحفظة {title}*\n\n"

                       f"{balance_section}\n"

                       f"💰 *إجمالي الربح/الخسارة:* `${total_pnl:+.2f}`\n"

                       f"⚙️ *النمط الحالي:* `{preset_name}`\n\n"

                       f"- *إجمالي الصفقات:* `{total}` (`{active}` نشطة)\n"

                       f"- *الناجحة:* `{successful}` | *الربح:* `${pnl.get('ناجحة', 0):.2f}`\n"

                       f"- *الفاشلة:* `{failed}` | *الخسارة:* `${abs(pnl.get('فاشلة', 0)):.2f}`\n"

                       f"- *معدل النجاح:* `{win_rate:.2f}%`")

        return stats_msg, None

    except Exception as e:

        logger.error(f"Error in stats_command: {e}", exc_info=True)

        return "خطأ في جلب الإحصائيات.", None



async def strategy_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_mode_filter='all'):

    report_string = generate_performance_report_string(trade_mode_filter)

    return report_string, None



async def send_daily_report(context: ContextTypes.DEFAULT_TYPE):

    today_str = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d')

    logger.info(f"Generating detailed daily report for {today_str}...")

    try:

        conn = sqlite3.connect(DB_FILE, timeout=10)

        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        

        cursor.execute("SELECT * FROM trades WHERE DATE(closed_at) = ? AND trade_mode = 'real'", (today_str,))

        closed_real_today = [dict(row) for row in cursor.fetchall()]

        

        cursor.execute("SELECT * FROM trades WHERE DATE(closed_at) = ? AND trade_mode = 'virtual'", (today_str,))

        closed_virtual_today = [dict(row) for row in cursor.fetchall()]

        conn.close()



        parts = [f"**🗓️ التقرير اليومي المفصل | {today_str}**\n"]



        def generate_section(title, trades):

            if not trades:

                return [f"\n--- **{title}** ---\nلم يتم إغلاق أي صفقات اليوم."]

            

            wins = [t for t in trades if t['status'] == 'ناجحة']

            losses = [t for t in trades if t['status'] == 'فاشلة']

            total_pnl = sum(t['pnl_usdt'] for t in trades if t['pnl_usdt'] is not None)

            win_rate = (len(wins) / len(trades) * 100) if trades else 0



            section_parts = [f"\n--- **{title}** ---"]

            section_parts.append(f"  - الربح/الخسارة الصافي: `${total_pnl:+.2f}`")

            section_parts.append(f"  - ✅ الرابحة: {len(wins)} | ❌ الخاسرة: {len(losses)}")

            section_parts.append(f"  - معدل النجاح: {win_rate:.1f}%")

            return section_parts



        parts.extend(generate_section("💰 الأداء الحقيقي", closed_real_today))

        parts.extend(generate_section("📊 الأداء الوهمي", closed_virtual_today))



        parts.append("\n\n*رسالة اليوم: \"النجاح في التداول هو نتيجة للانضباط والصبر والتعلم المستمر.\"*")

        report_message = "\n".join(parts)



        await send_telegram_message(context.bot, {'custom_message': report_message, 'target_chat': TELEGRAM_SIGNAL_CHANNEL_ID})

    except Exception as e:

        logger.error(f"Failed to generate detailed daily report: {e}", exc_info=True)



async def daily_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    target_message = update.callback_query.message if update.callback_query else update.message

    await target_message.reply_text("⏳ جاري إرسال التقرير اليومي المفصل...")

    await send_daily_report(context)

    await target_message.reply_text("✅ تم إرسال التقرير بنجاح إلى القناة.")



async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    target_message = update.callback_query.message if update.callback_query else update.message

    await target_message.reply_text("⏳ جاري إعداد تقرير التشخيص الشامل...")

    settings = bot_data.get("settings", {})

    parts = [f"**🕵️‍♂️ تقرير التشخيص الشامل**\n\n*تم إنشاؤه في: {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')}*"]



    parts.append("\n- - - - - - - - - - - - - - - - - -")

    parts.append("**[ ⚙️ حالة النظام والبيئة ]**")

    parts.append(f"- `NLTK (تحليل الأخبار):` {'متاحة ✅' if NLTK_AVAILABLE else 'غير متاحة ❌'}")

    parts.append(f"- `SciPy (تحليل الدايفرجنس):` {'متاحة ✅' if SCIPY_AVAILABLE else 'غير متاحة ❌'}")

    parts.append(f"- `Alpha Vantage (بيانات اقتصادية):` {'موجود ✅' if ALPHA_VANTAGE_API_KEY != 'YOUR_AV_KEY_HERE' else 'مفقود ⚠️'}")



    parts.append("\n**[ 📊 حالة السوق الحالية ]**")

    mood_info = settings.get("last_market_mood", {})

    # [إصلاح حرج] جعل جلب البيانات الخارجية أكثر صلابة

    try:

        fng_value = await get_fear_and_greed_index()

        fng_text = "غير متاح"

        if fng_value is not None:

            classification = "خوف شديد" if fng_value < 25 else "خوف" if fng_value < 45 else "محايد" if fng_value < 55 else "طمع" if fng_value < 75 else "طمع شديد"

            fng_text = f"{fng_value} ({classification})"

    except Exception as e:

        fng_text = f"فشل الجلب ({e})"

    parts.append(f"- **المزاج الأساسي (أخبار):** `{mood_info.get('mood', 'N/A')}`")

    parts.append(f"  - `{mood_info.get('reason', 'N/A')}`")

    parts.append(f"- **المزاج الفني (BTC):** `{bot_data['status_snapshot']['btc_market_mood']}`")

    parts.append(f"- **مؤشر الخوف والطمع:** `{fng_text}`")



    status = bot_data['status_snapshot']

    scan_duration = "N/A"

    # [إصلاح حرج] التعامل مع كائنات datetime مباشرة

    if isinstance(status.get('last_scan_end_time'), datetime) and isinstance(status.get('last_scan_start_time'), datetime):

        duration_sec = (status['last_scan_end_time'] - status['last_scan_start_time']).total_seconds()

        scan_duration = f"{duration_sec:.0f} ثانية"

    parts.append("\n**[ 🔬 أداء آخر فحص ]**")

    parts.append(f"- **وقت البدء:** `{status.get('last_scan_start_time', 'N/A')}`")

    parts.append(f"- **المدة:** `{scan_duration}`")

    parts.append(f"- **العملات المفحوصة:** `{status['markets_found']}`")

    parts.append(f"- **فشل في تحليل:** `{(bot_data['scan_history'][-1]['failures'] if bot_data['scan_history'] else 'N/A')}` عملات")



    parts.append("\n**[ 🔧 الإعدادات النشطة ]**")

    parts.append(f"- **النمط الحالي:** `{settings.get('active_preset_name', 'N/A')}`")

    parts.append(f"- **الماسحات المفعلة:** `{', '.join(settings.get('active_scanners', []))}`")

    lf, vf = settings['liquidity_filters'], settings['volatility_filters']

    parts.append("- **فلاتر السيولة:**")

    parts.append(f"  - `حجم التداول الأدنى:` ${lf['min_quote_volume_24h_usd']:,}")

    parts.append(f"  - `أقصى سبريد مسموح:` {lf['max_spread_percent']}%")

    parts.append(f"  - `الحد الأدنى لـ RVOL:` {lf['min_rvol']}")

    parts.append("- **فلتر التقلب:**")

    parts.append(f"  - `الحد الأدنى لـ ATR:` {vf['min_atr_percent']}%")



    parts.append("\n**[ 🔩 حالة العمليات الداخلية ]**")

    if context.job_queue:

        try:

            scan_job = context.job_queue.get_jobs_by_name('perform_scan')

            track_job = context.job_queue.get_jobs_by_name('track_open_trades')

            scan_next = scan_job[0].next_t.astimezone(EGYPT_TZ).strftime('%H:%M:%S') if scan_job and scan_job[0].next_t else 'N/A'

            track_next = track_job[0].next_t.astimezone(EGYPT_TZ).strftime('%H:%M:%S') if track_job and track_job[0].next_t else 'N/A'

            parts.append("- **المهام المجدولة:**")

            parts.append(f"  - `فحص العملات:` {'يعمل'}, *التالي في: {scan_next}*")

            parts.append(f"  - `متابعة الصفقات:` {'يعمل'}, *التالي في: {track_next}*")

        except Exception as e:

            parts.append(f"- **المهام المجدولة:** فشل الفحص ({e})")

            

    parts.append("- **الاتصال بالمنصات:**")

    for ex_id in EXCHANGES_TO_SCAN:

        is_private_connected = ex_id in bot_data.get('exchanges', {}) and bot_data['exchanges'][ex_id].apiKey

        is_public_connected = ex_id in bot_data.get('public_exchanges', {})

        status_text = f"عام: {'✅' if is_public_connected else '❌'} | خاص: {'✅' if is_private_connected else '❌'}"

        parts.append(f"  - `{ex_id.capitalize()}:` {status_text}")





    parts.append("- **قاعدة البيانات:**")

    try:

        conn = sqlite3.connect(DB_FILE, timeout=5); cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM trades"); total_trades = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة'"); active_trades = cursor.fetchone()[0]

        conn.close()

        db_size = os.path.getsize(DB_FILE) / (1024 * 1024)

        parts.append(f"  - `الاتصال:` ناجح ✅")

        parts.append(f"  - `حجم الملف:` {db_size:.2f} MB")

        parts.append(f"  - `إجمالي الصفقات:` {total_trades} ({active_trades} نشطة)")

    except Exception as e: parts.append(f"  - `الاتصال:` فشل ❌ ({e})")

    parts.append("- - - - - - - - - - - - - - - - - -")



    await target_message.reply_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN)





async def check_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_id_from_callback=None):

    target = update.callback_query.message if trade_id_from_callback else update.message

    def format_price(price): return f"{price:,.8f}" if price < 0.01 else f"{price:,.4f}"

    try:

        trade_id = trade_id_from_callback or int(context.args[0])

        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row; cursor = conn.cursor()

        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))

        trade_row = cursor.fetchone(); conn.close()

        if not trade_row: await target.reply_text(f"لم يتم العثور على صفقة بالرقم `{trade_id}`."); return

        trade = dict(trade_row)

        

        message = ""

        keyboard = None



        if trade['status'] != 'نشطة':

            pnl_percent = (trade['pnl_usdt'] / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0

            closed_at_dt = datetime.strptime(trade['closed_at'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=EGYPT_TZ)

            message = f"📋 *ملخص الصفقة #{trade_id}*\n\n*العملة:* `{trade['symbol']}`\n*الحالة:* `{trade['status']}`\n*تاريخ الإغلاق:* `{closed_at_dt.strftime('%Y-%m-%d %I:%M %p')}`\n*الربح/الخسارة:* `${trade.get('pnl_usdt', 0):+.2f} ({pnl_percent:+.2f}%)`"

        else:

            if not (exchange := bot_data["public_exchanges"].get(trade['exchange'].lower())): await target.reply_text("المنصة غير متصلة."); return

            if not (ticker := await exchange.fetch_ticker(trade['symbol'])) or not (current_price := ticker.get('last') or ticker.get('close')):

                await target.reply_text(f"لم أتمكن من جلب السعر الحالي لـ `{trade['symbol']}`."); return

            live_pnl = (current_price - trade['entry_price']) * trade['quantity']

            live_pnl_percent = (live_pnl / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0

            message = (f"📈 *متابعة حية للصفقة #{trade_id}*\n\n"

                       f"▫️ *العملة:* `{trade['symbol']}` | *الحالة:* `نشطة`\n"

                       f"▫️ *سعر الدخول:* `${format_price(trade['entry_price'])}`\n"

                       f"▫️ *السعر الحالي:* `${format_price(current_price)}`\n\n"

                       f"💰 *الربح/الخسارة الحالية:*\n`${live_pnl:+.2f} ({live_pnl_percent:+.2f}%)`")

            # [ميزة] إضافة زر الإغلاق اليدوي

            if trade['trade_mode'] == 'real':

                keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🚨 إغلاق الآن بسعر السوق", callback_data=f"manual_close_{trade_id}")]])



        await target.reply_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)

    except (ValueError, IndexError): await target.reply_text("رقم صفقة غير صالح. مثال: `/check 17`")

    except Exception as e: logger.error(f"Error in check_trade_command: {e}", exc_info=True); await target.reply_text("حدث خطأ.")



async def show_active_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_mode_filter='all'):

    try:

        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row; cursor = conn.cursor()

        

        query = "SELECT id, symbol, entry_value_usdt, exchange FROM trades WHERE status = 'نشطة'"

        params = []

        if trade_mode_filter != 'all':

            query += " AND trade_mode = ?"

            params.append(trade_mode_filter)

        query += " ORDER BY id DESC"



        cursor.execute(query, params)

        active_trades = cursor.fetchall(); conn.close()

        

        if not active_trades:

            return "لا توجد صفقات نشطة حالياً لهذا الفلتر.", None

            

        keyboard = [[InlineKeyboardButton(f"#{t['id']} | {t['symbol']} | ${t['entry_value_usdt']:.2f} | {t['exchange']}", callback_data=f"check_{t['id']}")] for t in active_trades]

        return "اختر صفقة لمتابعتها:", InlineKeyboardMarkup(keyboard)

    except Exception as e:

        logger.error(f"Error in show_active_trades: {e}")

        return "خطأ في جلب الصفقات.", None



async def execute_manual_trade(exchange_id, symbol, amount_usdt, side, context: ContextTypes.DEFAULT_TYPE, order_type='market', price=None, stop_price=None, tp_price=None, sl_price=None):

    logger.info(f"Attempting MANUAL {order_type.upper()} {side.upper()} for {symbol} on {exchange_id}")

    exchange = bot_data["exchanges"].get(exchange_id.lower())

    if not exchange or not exchange.apiKey:

        return {"success": False, "error": f"لا يمكن تنفيذ الأمر. لم يتم توثيق الاتصال بمنصة {exchange_id.capitalize()}."}



    try:

        ticker = await exchange.fetch_ticker(symbol)

        current_price = ticker.get('last') or ticker.get('close')

        if not current_price:

            return {"success": False, "error": f"لم أتمكن من جلب السعر الحالي لـ {symbol}."}



        quantity = float(amount_usdt) / (price or current_price)

        formatted_quantity = exchange.amount_to_precision(symbol, quantity)



        order_receipt = None

        if order_type == 'market':

            if side == 'buy':

                order_receipt = await exchange.create_market_buy_order(symbol, float(formatted_quantity))

            elif side == 'sell':

                order_receipt = await exchange.create_market_sell_order(symbol, float(formatted_quantity))

        elif order_type == 'limit':

            if side == 'buy':

                order_receipt = await exchange.create_limit_buy_order(symbol, float(formatted_quantity), float(price))

            elif side == 'sell':

                 order_receipt = await exchange.create_limit_sell_order(symbol, float(formatted_quantity), float(price))

        elif order_type == 'oco_limit' and side == 'buy':

             # This is a complex case, not standard for entry, usually for exit.

             # We can implement a STOP-LIMIT buy as a proxy

             order_receipt = await exchange.create_order(symbol, 'stop_limit', 'buy', float(formatted_quantity), price=price, params={'stopPrice': stop_price})



        await asyncio.sleep(2)

        order = await exchange.fetch_order(order_receipt['id'], symbol)



        logger.info(f"MANUAL ORDER SUCCESS: {order}")



        filled_quantity = order.get('filled', 0)

        status = order.get('status', 'unknown')

        cost = order.get('cost', 0)



        success_message = (

            f"**✅ تم وضع الأمر اليدوي بنجاح**\n\n"

            f"**المنصة:** `{exchange_id.capitalize()}`\n"

            f"**العملة:** `{symbol}`\n"

            f"**النوع:** `{order_type.upper()} {side.upper()}`\n\n"

            f"--- **تفاصيل الأمر** ---\n"

            f"**ID:** `{order['id']}`\n"

            f"**الحالة:** `{status}`\n"

            f"**الكمية:** `{formatted_quantity}`\n"

            f"**السعر:** `{price or 'Market'}`\n"

            f"**التكلفة الإجمالية (عند التنفيذ):** `${cost or 'N/A'}`"

        )

        return {"success": True, "message": success_message}



    except ccxt.InsufficientFunds as e:

        error_msg = f"❌ فشل: رصيد غير كافٍ على {exchange_id.capitalize()}."

        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}")

        return {"success": False, "error": error_msg}

    except ccxt.InvalidOrder as e:

        error_msg = f"❌ فشل: أمر غير صالح. قد يكون المبلغ أقل من الحد الأدنى للمنصة.\n`{e}`"

        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}")

        return {"success": False, "error": error_msg}

    except ccxt.ExchangeError as e:

        error_msg = f"❌ فشل: خطأ من المنصة.\n`{e}`"

        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}")

        return {"success": False, "error": error_msg}

    except Exception as e:

        error_msg = f"❌ فشل: حدث خطأ غير متوقع.\n`{e}`"

        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}", exc_info=True)

        return {"success": False, "error": error_msg}



async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):

    query = update.callback_query; await query.answer(); data = query.data

    user_data = context.user_data



    # [إصلاح جذري] إعادة هيكلة كاملة لمعالج التقارير

    if data.startswith("dashboard_") and data.endswith(('_all', '_real', '_virtual')):

        if report_lock.locked():

            await query.answer("⏳ تقرير آخر قيد الإعداد، يرجى الانتظار...", show_alert=False)

            return

            

        async with report_lock:

            try:

                # [إصلاح حاسم] الطريقة الصحيحة لتحليل نص الزر

                parts = data.split('_')

                trade_mode_filter = parts[-1]

                report_type = '_'.join(parts[1:-1])



                await query.edit_message_text(f"⏳ جاري إعداد تقرير **{report_type.replace('_', ' ').capitalize()}**...", parse_mode=ParseMode.MARKDOWN)



                report_content, keyboard = None, None



                if report_type == "stats":

                    report_content, keyboard = await stats_command(update, context, trade_mode_filter=trade_mode_filter)

                elif report_type == "active_trades":

                    report_content, keyboard = await show_active_trades_command(update, context, trade_mode_filter=trade_mode_filter)

                elif report_type == "strategy_report":

                    report_content, keyboard = await strategy_report_command(update, context, trade_mode_filter=trade_mode_filter)



                if report_content:

                    await query.edit_message_text(

                        text=report_content, 

                        reply_markup=keyboard, 

                        parse_mode=ParseMode.MARKDOWN

                    )

                else:

                    await query.edit_message_text("❌ فشل إعداد التقرير.")



            except Exception as e:

                logger.error(f"Error in dashboard filter handler: {e}", exc_info=True)

                try:

                    await query.edit_message_text("❌ حدث خطأ غير متوقع أثناء إعداد التقرير.")

                except Exception:

                    pass

        return



    # --- Dashboard Main Actions ---

    if data.startswith("dashboard_"):

        action = data.split("_", 1)[1]

        

        if action in ["stats", "active_trades", "strategy_report"]:

            keyboard = InlineKeyboardMarkup([

                [InlineKeyboardButton("📊 الكل (وهمي + حقيقي)", callback_data=f"dashboard_{action}_all")],

                [InlineKeyboardButton("📈 حقيقي فقط", callback_data=f"dashboard_{action}_real"), InlineKeyboardButton("📉 وهمي فقط", callback_data=f"dashboard_{action}_virtual")],

                [InlineKeyboardButton("🔙 العودة", callback_data="dashboard_refresh")]

            ])

            await query.edit_message_text(f"اختر نوع السجل لعرض **{action.replace('_', ' ').capitalize()}**:", reply_markup=keyboard)

            return



        if action == "debug": 

            await query.edit_message_text("⏳ جاري إعداد تقرير التشخيص...", parse_mode=ParseMode.MARKDOWN)

            await debug_command(update, context)

        elif action == "refresh": await show_dashboard_command(update, context)

        elif action == "snapshot": await portfolio_snapshot_command(update, context)

        elif action == "risk": await risk_report_command(update, context)

        elif action == "sync": await sync_portfolio_command(update, context)

        elif action == "tools":

             # [ميزة] قائمة أدوات التداول المتقدمة

             keyboard = [

                 [InlineKeyboardButton("⚡️ أمر فوري (Market)", callback_data="tools_manual_trade_market")],

                 [InlineKeyboardButton("⏳ أمر معلق (Limit/Stop)", callback_data="tools_manual_trade_limit")],

                 [InlineKeyboardButton("❌ إلغاء أمر مفتوح", callback_data="tools_cancel_order")],

                 [InlineKeyboardButton("💰 عرض رصيدي", callback_data="tools_balance")],

                 [InlineKeyboardButton("📖 أوامري المفتوحة", callback_data="tools_openorders")],

                 [InlineKeyboardButton("🚨 زر الذعر (إغلاق الكل)", callback_data="tools_panic_button")],

                 [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="dashboard_refresh")]

             ]

             await query.edit_message_text("🛠️ *أدوات التداول المتقدمة*\n\nاختر الأداة التي تريد استخدامها:", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

        return



    elif data.startswith("tools_"):

        parts = data.split("_")

        tool = parts[1]

        if tool == "manual": await manual_trade_command(update, context, order_type=parts[2])

        elif tool == "balance": await balance_command(update, context)

        elif tool == "openorders": await open_orders_command(update, context)

        elif tool == "cancel": await cancel_order_command(update, context)

        elif tool == "panic": await panic_button_command(update, context)

        return



    elif data.startswith("manual_close_"):

        trade_id = int(data.split("_")[-1])

        context.user_data['awaiting_confirmation'] = {

            "action": "manual_close_trade",

            "trade_id": trade_id

        }

        await query.message.reply_text(f"⚠️ **تأكيد الإغلاق**\n\nأنت على وشك إغلاق الصفقة #{trade_id} بسعر السوق. هل أنت متأكد؟\n\n**للتأكيد، أرسل كلمة `تأكيد`**")

        return



    # [جديد] معالج زر اختيار منصة لقطة المحفظة

    elif data.startswith("snapshot_exchange_"):

        exchange_id = data.split("_", 2)[2]

        await process_portfolio_snapshot(update, context, exchange_id)

        return

        

    # [جديد] معالج زر اختيار منصة المزامنة

    elif data.startswith("sync_exchange_"):

        exchange_id = data.split("_", 2)[2]

        await process_sync_portfolio(update, context, exchange_id)

        return

        

    # [جديد] معالج زر اختيار منصة زر الذعر

    elif data.startswith("panic_exchange_"):

        exchange_id = data.split("_", 2)[2]

        context.user_data['awaiting_confirmation'] = {

            "action": "panic_close",

            "exchange_id": exchange_id

        }

        await query.message.edit_text(f"⚠️ **تحذير أخير** ⚠️\n\nأنت على وشك إغلاق **جميع** الصفقات الحقيقية المفتوحة على منصة `{exchange_id}` بسعر السوق.\n\n**هذا الإجراء لا يمكن التراجع عنه.**\n\nللتأكيد، أرسل كلمة `تأكيد`.", parse_mode=ParseMode.MARKDOWN)

        return

        

    # [جديد] معالج زر اختيار منصة لإلغاء أمر

    elif data.startswith("cancel_order_exchange_"):

        exchange_id = data.split("_", 3)[3]

        await list_orders_to_cancel(update, context, exchange_id)

        return

        

    # [جديد] معالج زر إلغاء أمر محدد

    elif data.startswith("cancel_this_order_"):

        parts = data.split("_")

        exchange_id, order_id = parts[3], parts[4]

        await cancel_specific_order(update, context, exchange_id, order_id)

        return



    elif data.startswith("preset_"):

        preset_name = data.split("_", 1)[1]

        if preset_data := PRESETS.get(preset_name):

            bot_data["settings"]['liquidity_filters'] = preset_data['liquidity_filters']

            bot_data["settings"]['volatility_filters'] = preset_data['volatility_filters']

            bot_data["settings"]['ema_trend_filter'] = preset_data['ema_trend_filter']

            bot_data["settings"]['min_tp_sl_filter'] = preset_data['min_tp_sl_filter']

            bot_data["settings"]["active_preset_name"] = preset_name

            save_settings()

            preset_titles = {"PRO": "احترافي", "STRICT": "متشدد", "LAX": "متساهل", "VERY_LAX": "فائق التساهل"}

            lf, vf = preset_data['liquidity_filters'], preset_data['volatility_filters']

            confirmation_text = f"✅ *تم تفعيل النمط: {preset_titles.get(preset_name, preset_name)}*\n\n*أهم القيم:*\n`- min_rvol: {lf['min_rvol']}`\n`- max_spread: {lf['max_spread_percent']}%`\n`- min_atr: {vf['min_atr_percent']}%`"

            try: await query.edit_message_text(confirmation_text, parse_mode=ParseMode.MARKDOWN, reply_markup=get_presets_keyboard())

            except BadRequest as e:

                if "Message is not modified" not in str(e): raise

    elif data.startswith("param_"):

        param_key = data.split("_", 1)[1]

        context.user_data['awaiting_input_for_param'] = param_key

        context.user_data['settings_menu_id'] = query.message.message_id

        current_value = bot_data["settings"].get(param_key)

        if isinstance(current_value, bool):

            bot_data["settings"][param_key] = not current_value

            bot_data["settings"]["active_preset_name"] = "Custom"; save_settings()

            await query.answer(f"✅ تم تبديل '{PARAM_DISPLAY_NAMES.get(param_key, param_key)}'")

            await show_parameters_menu(update, context)

        else:

            await query.edit_message_text(f"📝 *تعديل '{PARAM_DISPLAY_NAMES.get(param_key, param_key)}'*\n\n*القيمة الحالية:* `{current_value}`\n\nالرجاء إرسال القيمة الجديدة.", parse_mode=ParseMode.MARKDOWN)

    

    # [جديد] معالج أزرار التحكم الذاتي

    elif data.startswith("toggle_autopilot_"):

        param_key = data.split("_", 2)[2]

        settings = bot_data["settings"].get("autopilot_settings", {})

        settings[param_key] = not settings.get(param_key, False)

        bot_data["settings"]["autopilot_settings"] = settings

        save_settings()

        await query.answer(f"تم {'تفعيل' if settings[param_key] else 'تعطيل'} {AUTOPILOT_PARAMS.get(param_key)}")

        # Refresh the menu

        await show_autopilot_menu(update, context)

        if query.message: await query.message.delete()

        return



    # [جديد] معالج أزرار إدارة قاعدة البيانات

    elif data.startswith("db_manage_"):

        action = data.split("_", 2)[2]

        context.user_data['awaiting_confirmation'] = {"action": f"db_{action}"}

        

        message = ""

        if action == "clean":

            message = "أنت على وشك حذف جميع الصفقات **المغلقة** الأقدم من 30 يومًا."

        elif action == "purge":

            message = "أنت على وشك حذف **جميع الصفقات المغلقة** من قاعدة البيانات."

        

        await query.message.edit_text(f"⚠️ **تأكيد الإجراء** ⚠️\n\n{message}\n\n**هذا الإجراء لا يمكن التراجع عنه.**\n\nللتأكيد، أرسل كلمة `تأكيد`.", parse_mode=ParseMode.MARKDOWN)

        return

    elif data == "db_manage_export":

        await export_db_command(update, context)

        return



    elif data.startswith("toggle_scanner_"):

        scanner_name = data.split("_", 2)[2]

        active_scanners = bot_data["settings"].get("active_scanners", []).copy()

        if scanner_name in active_scanners:

            if len(active_scanners) > 1: # Prevent disabling the last scanner

                active_scanners.remove(scanner_name)

            else:

                await query.answer("لا يمكن تعطيل آخر ماسح!", show_alert=True)

                return

        else:

            active_scanners.append(scanner_name)

        bot_data["settings"]["active_scanners"] = active_scanners; save_settings()

        try: await query.edit_message_text(text="اختر الماسحات لتفعيلها أو تعطيلها:", reply_markup=get_scanners_keyboard())

        except BadRequest as e:

            if "Message is not modified" not in str(e): raise

    # [جديد] معالجة زر التحكم بالتداول الحقيقي

    elif data.startswith("toggle_real_trade_"):

        exchange_id = data.split("_", 3)[3]

        settings = bot_data["settings"].get("real_trading_per_exchange", {})

        settings[exchange_id] = not settings.get(exchange_id, False)

        bot_data["settings"]["real_trading_per_exchange"] = settings

        save_settings()

        await query.answer(f"تم {'تفعيل' if settings[exchange_id] else 'تعطيل'} التداول على {exchange_id.capitalize()}")

        # Refresh the menu

        await show_real_trading_control_menu(update, context)

        if query.message: await query.message.delete()

        return



    elif data == "back_to_settings":

        if query.message: await query.message.delete()

        await context.bot.send_message(chat_id=query.message.chat_id, text="اختر الإعداد:", reply_markup=ReplyKeyboardMarkup(settings_menu_keyboard, resize_keyboard=True))

    elif data.startswith("check_"):

        await check_trade_command(update, context, trade_id_from_callback=int(data.split("_")[1]))



    elif data.startswith("suggest_"):

        action = data.split("_", 1)[1]

        if action.startswith("accept"):

            preset_name = data.split("_")[2]

            if preset_data := PRESETS.get(preset_name):

                bot_data["settings"]['liquidity_filters'] = preset_data['liquidity_filters']

                bot_data["settings"]['volatility_filters'] = preset_data['volatility_filters']

                bot_data["settings"]['ema_trend_filter'] = preset_data['ema_trend_filter']

                bot_data["settings"]['min_tp_sl_filter'] = preset_data['min_tp_sl_filter']

                bot_data["settings"]["active_preset_name"] = preset_name

                save_settings()

                await query.edit_message_text(f"✅ **تم قبول الاقتراح!**\n\nتم تغيير النمط بنجاح إلى `{preset_name}`.", parse_mode=ParseMode.MARKDOWN)

        elif action == "decline":

            await query.edit_message_text("👍 **تم تجاهل الاقتراح.**\n\nسيستمر البوت بالعمل على الإعدادات الحالية.", parse_mode=ParseMode.MARKDOWN)



async def manual_trade_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):

    query = update.callback_query; await query.answer(); data = query.data

    user_data = context.user_data



    if 'manual_trade' not in user_data:

        await query.edit_message_text("⚠️ انتهت هذه الجلسة. ابدأ من جديد باستخدام /trade.")

        return



    state = user_data['manual_trade'].get('state')



    if data == "manual_trade_cancel":

        user_data.pop('manual_trade', None)

        await query.edit_message_text("👍 تم إلغاء عملية التداول اليدوي.")

        return



    if state == 'awaiting_exchange':

        exchange = data.split("_")[-1]

        user_data['manual_trade']['exchange'] = exchange

        user_data['manual_trade']['state'] = 'awaiting_symbol'

        await query.edit_message_text(f"اخترت منصة: *{exchange.capitalize()}*\n\nالآن، أرسل رمز العملة (مثال: `BTC/USDT`).", parse_mode=ParseMode.MARKDOWN)



    elif state == 'awaiting_side':

        side = data.split("_")[-1]

        user_data['manual_trade']['side'] = side

        user_data['manual_trade']['state'] = 'confirming'



        trade_data = user_data['manual_trade']

        await query.edit_message_text("⏳ جاري تنفيذ الأمر...", reply_markup=None)



        result = await execute_manual_trade(

            exchange_id=trade_data['exchange'],

            symbol=trade_data['symbol'],

            amount_usdt=trade_data['amount'],

            side=trade_data['side'],

            order_type=trade_data['order_type'],

            price=trade_data.get('price'),

            stop_price=trade_data.get('stop_price')

        )



        if result['success']:

            await query.edit_message_text(result['message'], parse_mode=ParseMode.MARKDOWN)

        else:

            await query.edit_message_text(result['error'], parse_mode=ParseMode.MARKDOWN)



        user_data.pop('manual_trade', None)



async def tools_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):

    query = update.callback_query

    await query.answer()

    data = query.data

    user_data = context.user_data

    tool_name, action, value = data.split("_", 2)



    tool_key = f"{tool_name}_tool"

    if tool_key not in user_data:

        await query.edit_message_text("⚠️ انتهت هذه الجلسة. ابدأ من جديد من لوحة التحكم.")

        return



    if action == "exchange":

        user_data[tool_key]['exchange'] = value

        if tool_name == "balance":

            await query.edit_message_text(f"💰 جاري جلب الأرصدة من *{value.capitalize()}*...", parse_mode=ParseMode.MARKDOWN)

            await fetch_and_display_balance(value, query)

            user_data.pop(tool_key, None)

        else:

            user_data[tool_key]['state'] = 'awaiting_symbol'

            await query.edit_message_text(f"اخترت منصة: *{value.capitalize()}*\n\nالآن، أرسل رمز العملة (مثال: `BTC/USDT`)\nأو أرسل `الكل` لعرض البيانات لجميع العملات.", parse_mode=ParseMode.MARKDOWN)



async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not update.message or not update.message.text:

        return



    user_data = context.user_data

    text = update.message.text.strip()

    

    # [ميزة] معالج كلمات التأكيد

    if 'awaiting_confirmation' in user_data and text.lower() == 'تأكيد':

        action_data = user_data.pop('awaiting_confirmation')

        action = action_data['action']

        

        if action == "panic_close":

            await panic_close_all_trades_confirmed(update, context, action_data['exchange_id'])

        elif action == "manual_close_trade":

            await manual_close_trade_confirmed(update, context, action_data['trade_id'])

        elif action == "db_clean":

            await cleanup_db_command(update, context, days=30)

        elif action == "db_purge":

            await cleanup_db_command(update, context, purge_all=True)

        return



    active_tool = None

    for tool_key in ['openorders_tool', 'mytrades_tool', 'manual_trade']:

        if tool_key in user_data:

            active_tool = tool_key

            break



    if active_tool:

        state = user_data[active_tool].get('state')

        if state == 'awaiting_symbol':

            symbol = text.upper()

            exchange_id = user_data[active_tool]['exchange']



            if symbol.lower() in ["all", "الكل"]:

                symbol = None

            elif '/' not in symbol:

                await update.message.reply_text("❌ رمز غير صالح. الرجاء إرسال الرمز بالتنسيق الصحيح (مثال: `BTC/USDT`) أو كلمة `الكل`.")

                return



            if active_tool == 'openorders_tool':

                await update.message.reply_text(f"📖 جاري جلب أوامرك المفتوحة لـ *{symbol or 'الكل'}*...", parse_mode=ParseMode.MARKDOWN)

                await fetch_and_display_open_orders(exchange_id, symbol, update.message)

            elif active_tool == 'mytrades_tool':

                await update.message.reply_text(f"📜 جاري جلب سجل تداولاتك لـ *{symbol or 'الكل'}*...", parse_mode=ParseMode.MARKDOWN)

                await fetch_and_display_my_trades(exchange_id, symbol, update.message)

            elif active_tool == 'manual_trade':

                 user_data['manual_trade']['symbol'] = symbol

                 user_data['manual_trade']['state'] = 'awaiting_amount'

                 await update.message.reply_text(f"رمز العملة: *{symbol}*\n\nالآن، أدخل المبلغ بـ USDT (مثال: `15`).", parse_mode=ParseMode.MARKDOWN)



            if active_tool != 'manual_trade':

                user_data.pop(active_tool, None)

            return



        elif active_tool == 'manual_trade' and state == 'awaiting_amount':

            try:

                amount = float(text)

                if amount <= 0: raise ValueError("Amount must be positive")

                user_data['manual_trade']['amount'] = amount

                if user_data['manual_trade']['order_type'] == 'market':

                     user_data['manual_trade']['state'] = 'awaiting_side'

                     keyboard = [

                         [InlineKeyboardButton("📈 شراء (Buy)", callback_data="manual_trade_side_buy"),

                          InlineKeyboardButton("📉 بيع (Sell)", callback_data="manual_trade_side_sell")],

                         [InlineKeyboardButton("❌ إلغاء", callback_data="manual_trade_cancel")]

                     ]

                     await update.message.reply_text(f"المبلغ: *${amount}*\n\nاختر نوع الأمر:", reply_markup=InlineKeyboardMarkup(keyboard))

                else: # Limit/Stop order

                    user_data['manual_trade']['state'] = 'awaiting_price'

                    await update.message.reply_text(f"المبلغ: *${amount}*\n\nالآن، أرسل سعر الدخول المطلوب.", parse_mode=ParseMode.MARKDOWN)



            except ValueError:

                await update.message.reply_text("❌ مبلغ غير صالح. الرجاء إرسال رقم فقط (مثال: `15` أو `20.5`).")

            return

            

        elif active_tool == 'manual_trade' and state == 'awaiting_price':

            try:

                price = float(text)

                if price <= 0: raise ValueError("Price must be positive")

                user_data['manual_trade']['price'] = price

                user_data['manual_trade']['state'] = 'awaiting_side'

                keyboard = [

                     [InlineKeyboardButton("📈 شراء معلق (Limit Buy)", callback_data="manual_trade_side_buy")],

                     [InlineKeyboardButton("📉 بيع معلق (Limit Sell)", callback_data="manual_trade_side_sell")],

                     [InlineKeyboardButton("❌ إلغاء", callback_data="manual_trade_cancel")]

                ]

                await update.message.reply_text(f"سعر الدخول: *{price}*\n\nاختر نوع الأمر المعلق:", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

            except ValueError:

                await update.message.reply_text("❌ سعر غير صالح. الرجاء إرسال رقم فقط.")

            return



    menu_handlers = {

        "Dashboard 🖥️": show_dashboard_command,

        "ℹ️ مساعدة": help_command,

        "⚙️ الإعدادات": show_settings_menu,

        "🔧 تعديل المعايير": show_parameters_menu,

        "🔙 القائمة الرئيسية": start_command,

        "🎭 تفعيل/تعطيل الماسحات": show_scanners_menu,

        "🏁 أنماط جاهزة": show_presets_menu,

        "🚨 التحكم بالتداول الحقيقي": show_real_trading_control_menu,

        "🧠 التحكم الذاتي": show_autopilot_menu,

        "🗄️ إدارة قاعدة البيانات": show_db_management_menu,

    }

    if text in menu_handlers:

        for key in list(user_data.keys()):

            if key.startswith(('manual_trade', 'openorders_tool', 'mytrades_tool', 'balance_tool', 'awaiting_')) :

                user_data.pop(key)



        handler = menu_handlers[text]

        await handler(update, context)

        return



    if param := user_data.pop('awaiting_input_for_param', None):

        value_str = update.message.text

        settings_menu_id = context.user_data.pop('settings_menu_id', None)

        chat_id = update.message.chat_id

        await context.bot.delete_message(chat_id=chat_id, message_id=update.message.message_id)

        settings = bot_data["settings"]

        try:

            current_type = type(settings.get(param, ''))

            new_value = current_type(value_str)

            if isinstance(settings.get(param), bool):

                new_value = value_str.lower() in ['true', '1', 'yes', 'on', 'نعم', 'تفعيل']

            settings[param] = new_value

            settings["active_preset_name"] = "Custom"

            save_settings()

            if settings_menu_id: context.user_data['settings_menu_id'] = settings_menu_id

            await show_parameters_menu(update, context)

            confirm_msg = await update.message.reply_text(f"✅ تم تحديث **{PARAM_DISPLAY_NAMES.get(param, param)}** إلى `{new_value}`.", parse_mode=ParseMode.MARKDOWN)

            context.job_queue.run_once(lambda ctx: ctx.bot.delete_message(chat_id, confirm_msg.message_id), 4)

        except (ValueError, KeyError):

            if settings_menu_id:

                await context.bot.edit_message_text(chat_id=chat_id, message_id=settings_menu_id, text="❌ قيمة غير صالحة. الرجاء المحاولة مرة أخرى.")

                context.job_queue.run_once(lambda _: show_parameters_menu(update, context), 3)

        return



async def manual_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE, order_type='market'):

    context.user_data['manual_trade'] = {'state': 'awaiting_exchange', 'order_type': order_type}

    keyboard = [

        [InlineKeyboardButton(ex.capitalize(), callback_data=f"manual_trade_exchange_{ex}")] 

        for ex in bot_data["exchanges"].keys() if bot_data["exchanges"][ex].apiKey

    ]

    keyboard.append([InlineKeyboardButton("❌ إلغاء", callback_data="manual_trade_cancel")])



    message_text = f"✍️ **بدء تداول يدوي ({order_type.capitalize()})**\n\nاختر المنصة التي تريد تنفيذ الأمر عليها:"

    if update.callback_query:

        await update.callback_query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard))

    else:

        await update.message.reply_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard))



async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    context.user_data['balance_tool'] = {'state': 'awaiting_exchange'}

    keyboard = [

        [InlineKeyboardButton(ex.capitalize(), callback_data=f"balance_exchange_{ex}")]

         for ex in bot_data["exchanges"].keys() if bot_data["exchanges"][ex].apiKey

    ]

    keyboard.append([InlineKeyboardButton("🔙 العودة للأدوات", callback_data="dashboard_tools")])

    await update.callback_query.edit_message_text("💰 **عرض الرصيد**\n\nاختر المنصة لعرض أرصدتك:", reply_markup=InlineKeyboardMarkup(keyboard))



async def open_orders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    context.user_data['openorders_tool'] = {'state': 'awaiting_exchange'}

    keyboard = [

        [InlineKeyboardButton(ex.capitalize(), callback_data=f"openorders_exchange_{ex}")]

         for ex in bot_data["exchanges"].keys() if bot_data["exchanges"][ex].apiKey

    ]

    keyboard.append([InlineKeyboardButton("🔙 العودة للأدوات", callback_data="dashboard_tools")])

    await update.callback_query.edit_message_text("📖 **أوامري المفتوحة**\n\nاختر المنصة:", reply_markup=InlineKeyboardMarkup(keyboard))



async def my_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    context.user_data['mytrades_tool'] = {'state': 'awaiting_exchange'}

    keyboard = [

        [InlineKeyboardButton(ex.capitalize(), callback_data=f"mytrades_exchange_{ex}")]

         for ex in bot_data["exchanges"].keys() if bot_data["exchanges"][ex].apiKey

    ]

    keyboard.append([InlineKeyboardButton("🔙 العودة للأدوات", callback_data="dashboard_tools")])

    await update.callback_query.edit_message_text("📜 **سجل تداولاتي**\n\nاختر المنصة:", reply_markup=InlineKeyboardMarkup(keyboard))



async def fetch_and_display_balance(exchange_id, query):

    exchange = bot_data["exchanges"].get(exchange_id.lower())

    if not exchange or not exchange.apiKey:

        await query.edit_message_text(f"❌ خطأ: لم يتم توثيق الاتصال بمنصة {exchange_id.capitalize()}.")

        return



    try:

        balance = await exchange.fetch_balance()

        total_balance = balance.get('total', {})



        public_exchange = bot_data['public_exchanges'].get(exchange_id.lower())

        tickers = await public_exchange.fetch_tickers()



        assets = []

        for currency, amount in total_balance.items():

            if amount > 0:

                usdt_value = 0

                if currency == 'USDT':

                    usdt_value = amount

                elif f"{currency}/USDT" in tickers:

                    usdt_value = amount * tickers[f"{currency}/USDT"]['last']



                if usdt_value > 1:

                    assets.append({'currency': currency, 'amount': amount, 'usdt_value': usdt_value})



        assets.sort(key=lambda x: x['usdt_value'], reverse=True)



        if not assets:

            await query.edit_message_text(f"ℹ️ لا توجد أرصدة كبيرة (> $1) على منصة {exchange_id.capitalize()}.")

            return



        message_lines = [f"**💰 رصيدك على {exchange_id.capitalize()}**\n"]

        total_usdt_value = sum(a['usdt_value'] for a in assets)

        message_lines.append(f"__**إجمالي القيمة التقديرية:**__ `${total_usdt_value:,.2f}`\n")



        for asset in assets[:15]:

            message_lines.append(f"- `{asset['currency']}`: `{asset['amount']:.4f}` (~`${asset['usdt_value']:.2f}`)")



        await query.edit_message_text("\n".join(message_lines), parse_mode=ParseMode.MARKDOWN)



    except Exception as e:

        logger.error(f"Error fetching balance for {exchange_id}: {e}")

        await query.edit_message_text(f"❌ حدث خطأ أثناء جلب الرصيد من {exchange_id.capitalize()}.")



async def fetch_and_display_open_orders(exchange_id, symbol, message):

    exchange = bot_data["exchanges"].get(exchange_id.lower())

    if not exchange or not exchange.apiKey:

        await message.reply_text(f"❌ خطأ: لم يتم توثيق الاتصال بمنصة {exchange_id.capitalize()}.")

        return

    try:

        open_orders = await exchange.fetch_open_orders(symbol)



        if not open_orders:

            await message.reply_text(f"✅ لا توجد لديك أوامر مفتوحة لـ `{symbol or 'الكل'}` على {exchange_id.capitalize()}.")

            return



        lines = [f"**📖 أوامرك المفتوحة لـ `{symbol or 'الكل'}` على {exchange_id.capitalize()}**\n"]

        for order in open_orders:

            side_emoji = "🔼" if order['side'] == 'buy' else "🔽"

            lines.append(

                f"`{order['symbol']}` {side_emoji} `{order['side'].upper()}`\n"

                f"  - **الكمية:** `{order['amount']}`\n"

                f"  - **السعر:** `{order['price']}`\n"

                f"  - **النوع:** `{order['type']}`"

            )



        await message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)



    except Exception as e:

        logger.error(f"Error fetching open orders for {symbol} on {exchange_id}: {e}")

        await message.reply_text(f"❌ فشل جلب الأوامر المفتوحة. تأكد من صحة الرمز: `{symbol or ''}`.")



async def fetch_and_display_my_trades(exchange_id, symbol, message):

    exchange = bot_data["exchanges"].get(exchange_id.lower())

    if not exchange or not exchange.apiKey:

        await message.reply_text(f"❌ خطأ: لم يتم توثيق الاتصال بمنصة {exchange_id.capitalize()}.")

        return

    try:

        my_trades = await exchange.fetch_my_trades(symbol, limit=20)



        if not my_trades:

            await message.reply_text(f"✅ لا يوجد لديك سجل تداول لـ `{symbol or 'الكل'}` على {exchange_id.capitalize()}.")

            return



        lines = [f"**📜 آخر 20 من تداولاتك لـ `{symbol or 'الكل'}` على {exchange_id.capitalize()}**\n"]



        for trade in reversed(my_trades):

            trade_time = datetime.fromtimestamp(trade['timestamp'] / 1000, tz=EGYPT_TZ).strftime('%Y-%m-%d %H:%M')

            side_emoji = "🔼" if trade['side'] == 'buy' else "🔽"

            fee = trade.get('fee', {})

            fee_str = f"{fee.get('cost', 0):.4f} {fee.get('currency', '')}"

            lines.append(

                f"`{trade_time}` | `{trade['symbol']}` {side_emoji} `{trade['side'].upper()}`\n"

                f"  - **الكمية:** `{trade['amount']}`\n"

                f"  - **السعر:** `{trade['price']}`\n"

                f"  - **الرسوم:** `{fee_str}`"

            )



        await message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:

        logger.error(f"Error fetching my trades for {symbol} on {exchange_id}: {e}")

        await message.reply_text(f"❌ فشل جلب سجل تداولاتك. تأكد من صحة الرمز: `{symbol or ''}`.")





async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None: logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)



# [جديد] دالة مساعدة لحساب إجمالي قيمة المحفظة الحقيقية

async def get_total_real_portfolio_value_usdt():

    total_usdt_value = 0

    # Loop through all connected private exchanges

    for exchange in bot_data["exchanges"].values():

        if not exchange.apiKey:

            continue

        try:

            balance = await exchange.fetch_balance()

            # Use a cached version of tickers if available to speed up, otherwise fetch

            if not hasattr(exchange, '_tickers_cache') or (time.time() - exchange._tickers_cache_time > 60):

                 exchange._tickers_cache = await exchange.fetch_tickers()

                 exchange._tickers_cache_time = time.time()

            

            tickers = exchange._tickers_cache



            for currency, amount in balance.get('total', {}).items():

                if amount > 0:

                    usdt_value = 0

                    if currency == 'USDT':

                        usdt_value = amount

                    elif f"{currency}/USDT" in tickers and tickers[f"{currency}/USDT"].get('last'):

                        usdt_value = amount * tickers[f"{currency}/USDT"]['last']

                    

                    if usdt_value > 0.1: # Count smaller amounts too for total value

                        total_usdt_value += usdt_value

        except Exception as e:

            logger.error(f"Could not calculate real portfolio value for {exchange.id}: {e}")

    return total_usdt_value



# [إصلاح حاسم] تعديل دالة لقطة المحفظة

async def portfolio_snapshot_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    target_message = update.callback_query.message

    

    connected_exchanges = [ex for ex in bot_data["exchanges"].values() if ex.apiKey]

    

    if not connected_exchanges:

        await target_message.edit_text("❌ **فشل:** لم يتم العثور على أي منصة متصلة بحساب حقيقي. يرجى التأكد من إعداد مفاتيح API.")

        return



    if len(connected_exchanges) == 1:

        await process_portfolio_snapshot(update, context, connected_exchanges[0].id)

    else:

        keyboard = []

        for ex in connected_exchanges:

            keyboard.append([InlineKeyboardButton(f"📸 {ex.id.capitalize()}", callback_data=f"snapshot_exchange_{ex.id}")])

        keyboard.append([InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="dashboard_refresh")])

        

        await target_message.edit_text(

            "**📸 لقطة للمحفظة**\n\nلديك أكثر من منصة متصلة. يرجى اختيار المنصة التي تريد عرض لقطة لمحفظتها:",

            reply_markup=InlineKeyboardMarkup(keyboard)

        )



async def process_portfolio_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE, exchange_id: str):

    target_message = update.callback_query.message

    await target_message.edit_text(f"📸 **لقطة للمحفظة**\n\n⏳ جارِ الاتصال بمنصة {exchange_id.capitalize()} وجلب البيانات...")



    exchange = bot_data["exchanges"].get(exchange_id)

    if not exchange:

        await target_message.edit_text(f"❌ **فشل:** خطأ في العثور على منصة {exchange_id.capitalize()} المتصلة.")

        return



    try:

        balance = await exchange.fetch_balance()

        all_assets = balance.get('total', {})

        tickers = await exchange.fetch_tickers()

        

        portfolio_assets = []

        total_usdt_value = 0

        for currency, amount in all_assets.items():

            if amount > 0:

                usdt_value = 0

                if currency == 'USDT':

                    usdt_value = amount

                elif f"{currency}/USDT" in tickers and tickers[f"{currency}/USDT"].get('last'):

                    usdt_value = amount * tickers[f"{currency}/USDT"]['last']

                

                if usdt_value > 1:

                    portfolio_assets.append({'currency': currency, 'amount': amount, 'usdt_value': usdt_value})

                    total_usdt_value += usdt_value

        

        portfolio_assets.sort(key=lambda x: x['usdt_value'], reverse=True)

        

        all_recent_trades = []

        for asset in portfolio_assets:

            try:

                symbol = f"{asset['currency']}/USDT"

                if symbol in exchange.markets:

                    trades = await exchange.fetch_my_trades(symbol=symbol, limit=5)

                    all_recent_trades.extend(trades)

            except Exception as e:

                logger.warning(f"Could not fetch trades for {asset['currency']}: {e}")

        

        all_recent_trades.sort(key=lambda x: x['timestamp'], reverse=True)

        recent_trades = all_recent_trades[:20]



        parts = [f"**📸 لقطة لمحفظة {exchange.id.capitalize()}**\n"]

        parts.append(f"__**إجمالي القيمة التقديرية:**__ `${total_usdt_value:,.2f}`\n")



        parts.append("--- **الأرصدة الحالية (> $1)** ---")

        for asset in portfolio_assets[:15]:

            parts.append(f"- **{asset['currency']}**: `{asset['amount']:.4f}` *~`${asset['usdt_value']:.2f}`*")

        

        parts.append("\n--- **آخر 20 عملية تداول** ---")

        if not recent_trades:

            parts.append("لا يوجد سجل تداولات حديث.")

        else:

            for trade in reversed(recent_trades): 

                side_emoji = "🟢" if trade['side'] == 'buy' else "🔴"

                parts.append(f"`{trade['symbol']}` {side_emoji} `{trade['side'].upper()}` | الكمية: `{trade['amount']}` | السعر: `{trade['price']}`")

        

        await target_message.edit_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN)



    except Exception as e:

        logger.error(f"Error generating portfolio snapshot: {e}", exc_info=True)

        await target_message.edit_text(f"❌ **فشل:** حدث خطأ أثناء جلب بيانات المحفظة.\n`{e}`")



async def risk_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    target_message = update.callback_query.message

    await target_message.edit_text("ρίск **تقرير المخاطر**\n\n⏳ جارِ تحليل الصفقات النشطة...")



    try:

        conn = sqlite3.connect(DB_FILE, timeout=10)

        conn.row_factory = sqlite3.Row

        

        real_trades = conn.cursor().execute("SELECT * FROM trades WHERE status = 'نشطة' AND trade_mode = 'real'").fetchall()

        virtual_trades = conn.cursor().execute("SELECT * FROM trades WHERE status = 'نشطة' AND trade_mode = 'virtual'").fetchall()

        conn.close()



        parts = ["**ρίск تقرير المخاطر الحالي**\n"]

        

        # [إصلاح] تحويل الصفوف إلى قواميس قبل المعالجة

        real_trades = [dict(row) for row in real_trades]

        virtual_trades = [dict(row) for row in virtual_trades]



        def generate_risk_section(title, trades, portfolio_value):

            if not trades:

                return [f"\n--- **{title}** ---\n✅ لا توجد صفقات نشطة حالياً."]

            

            # The .get method is now valid as these are dicts

            valid_trades = [t for t in trades if all(k in t and t[k] is not None for k in ['entry_value_usdt', 'entry_price', 'stop_loss', 'quantity'])]

            

            total_at_risk = sum(t['entry_value_usdt'] for t in valid_trades)

            potential_loss = sum((t['entry_price'] - t['stop_loss']) * t['quantity'] for t in valid_trades)

            symbol_concentration = Counter(t['symbol'] for t in valid_trades)



             section_parts = [f"\n--- **{title}** ---"]
            section_parts.append(f"- **عدد الصفقات:** {len(valid_trades)}")
            section_parts.append(f"- **إجمالي رأس المال بالصفقات:** `${total_at_risk:,.2f}`")
            if portfolio_value > 0:
                section_parts.append(f"- **نسبة التعرض:** `{(total_at_risk / portfolio_value) * 100:.2f}%` من المحفظة")
            section_parts.append(f"- **أقصى خسارة محتملة:** `$-{potential_loss:,.2f}` (إذا ضُرب كل الوقف)")
            
            if symbol_concentration:
                most_common = symbol_concentration.most_common(1)[0]
                section_parts.append(f"- **العملة الأكثر تركيزاً:** `{most_common[0]}` ({most_common[1]} صفقات)")
            
            return section_parts

        real_portfolio_value = await get_total_real_portfolio_value_usdt()
        parts.extend(generate_risk_section("🚨 المخاطر الحقيقية", real_trades, real_portfolio_value))
        
        virtual_portfolio_value = bot_data['settings']['virtual_portfolio_balance_usdt']
        parts.extend(generate_risk_section("📊 المخاطر الوهمية", virtual_trades, virtual_portfolio_value))

        await target_message.edit_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error generating risk report: {e}", exc_info=True)
        await target_message.edit_text(f"❌ **فشل:** حدث خطأ أثناء إعداد تقرير المخاطر.\n`{e}`")

async def sync_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message
    
    connected_exchanges = [ex for ex in bot_data["exchanges"].values() if ex.apiKey]
    
    if not connected_exchanges:
        await target_message.edit_text("❌ **فشل:** لم يتم العثور على أي منصة متصلة بحساب حقيقي.")
        return

    if len(connected_exchanges) == 1:
        await process_sync_portfolio(update, context, connected_exchanges[0].id)
    else:
        keyboard = [[InlineKeyboardButton(f"🔄 {ex.id.capitalize()}", callback_data=f"sync_exchange_{ex.id}")] for ex in connected_exchanges]
        keyboard.append([InlineKeyboardButton("🔙 العودة", callback_data="dashboard_refresh")])
        await target_message.edit_text("اختر المنصة لمزامنة المحفظة معها:", reply_markup=InlineKeyboardMarkup(keyboard))

async def process_sync_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE, exchange_id: str):
    target_message = update.callback_query.message
    await target_message.edit_text(f"🔄 **مزامنة ومطابقة المحفظة**\n\n⏳ جارِ الاتصال بمنصة {exchange_id.capitalize()} ومقارنة البيانات...")

    exchange = bot_data["exchanges"].get(exchange_id)
    if not exchange:
        await target_message.edit_text(f"❌ **فشل:** خطأ في العثور على منصة {exchange_id.capitalize()}.")
        return
        
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        bot_trades_raw = conn.cursor().execute("SELECT symbol FROM trades WHERE status = 'نشطة' AND trade_mode = 'real' AND exchange = ?", (exchange_id.capitalize(),)).fetchall()
        bot_symbols = {item[0] for item in bot_trades_raw}
        conn.close()

        balance = await exchange.fetch_balance()
        exchange_symbols = set()
        for currency, amount in balance.get('total', {}).items():
            if amount > 0:
                symbol = f"{currency}/USDT"
                if symbol in exchange.markets:
                    usdt_value = 0
                    if currency == 'USDT': usdt_value = amount
                    else: 
                        ticker = await exchange.fetch_ticker(symbol)
                        usdt_value = amount * ticker.get('last', 0)
                    
                    if usdt_value > settings.get('real_trade_size_usdt', 15.0) * 0.5:
                        exchange_symbols.add(symbol)
        
        matched_symbols = bot_symbols.intersection(exchange_symbols)
        bot_only_symbols = bot_symbols.difference(exchange_symbols)
        exchange_only_symbols = exchange_symbols.difference(bot_symbols)

        parts = [f"**🔄 تقرير مزامنة المحفظة ({exchange.id.capitalize()})**\n"]
        parts.append(f"تمت مقارنة `{len(bot_symbols)}` صفقة مسجلة في البوت مع `{len(exchange_symbols)}` عملة مملوكة في المنصة (بقيمة > 50% من حجم الصفقة).\n")

        parts.append(f"--- **✅ صفقات متطابقة ({len(matched_symbols)})** ---")
        if matched_symbols: parts.extend([f"- `{s}`" for s in matched_symbols])
        else: parts.append("لا توجد صفقات متطابقة حالياً.")

        parts.append(f"\n--- **⚠️ صفقات في المنصة فقط ({len(exchange_only_symbols)})** ---")
        parts.append("*هذه هي الصفقات القديمة أو التي تم شراؤها يدوياً.*")
        if exchange_only_symbols: parts.extend([f"- `{s}`" for s in exchange_only_symbols])
        else: parts.append("لا توجد صفقات غير مسجلة في البوت.")

        parts.append(f"\n--- **❓ صفقات في البوت فقط ({len(bot_only_symbols)})** ---")
        parts.append("*هذه الصفقات قد تكون أُغلقت يدوياً أو أن قيمتها أقل من الحد. يجب التحقق منها.*")
        if bot_only_symbols: parts.extend([f"- `{s}`" for s in bot_only_symbols])
        else: parts.append("لا توجد صفقات غير متطابقة.")

        await target_message.edit_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error during portfolio sync: {e}", exc_info=True)
        await target_message.edit_text(f"❌ **فشل:** حدث خطأ أثناء مزامنة المحفظة.\n`{e}`")


async def post_init(application: Application):
    if NLTK_AVAILABLE:
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError: logger.info("Downloading NLTK data (vader_lexicon)..."); nltk.download('vader_lexicon')
    logger.info("Post-init: Initializing exchanges...")
    await initialize_exchanges()
    if not bot_data["public_exchanges"]: logger.critical("CRITICAL: No public exchange clients connected. Bot cannot run."); return

    if any(bot_data['settings'].get('real_trading_per_exchange', {}).values()):
        if not any(ex.apiKey for ex in bot_data.get("exchanges", {}).values()):
            logger.critical("CRITICAL: Real trading is enabled, but no API keys are set!")
            await application.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text="**🚨 خطأ فادح: مفاتيح التداول مفقودة 🚨**\n\nتم تفعيل التداول الحقيقي، ولكن لم يتم العثور على أي مفاتيح API. سيستمر البوت في العمل بالوضع الوهمي فقط.",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
             logger.info("Real trading is enabled and at least one set of API keys is present.")

    logger.info("Exchanges initialized. Setting up job queue...")
    job_queue = application.job_queue
    job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name='perform_scan')
    job_queue.run_repeating(track_open_trades, interval=TRACK_INTERVAL_SECONDS, first=20, name='track_open_trades')
    job_queue.run_daily(send_daily_report, time=dt_time(hour=23, minute=55, tzinfo=EGYPT_TZ), name='daily_report')
    # [جديد] إضافة مهمة تقييم الاستراتيجيات
    job_queue.run_daily(evaluate_strategies_periodically, time=dt_time(hour=1, minute=0, tzinfo=EGYPT_TZ), name='evaluate_strategies')
    
    logger.info(f"Jobs scheduled. Daily report at 23:55 {EGYPT_TZ}.")
    try:
        await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🚀 *بوت كاسحة الألغام (v5.0) جاهز للعمل!*", parse_mode=ParseMode.MARKDOWN)
    except TimedOut:
        logger.warning("Failed to send startup message due to a network timeout. The bot is running anyway.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending the startup message: {e}")
    logger.info("Post-init finished.")
    
async def post_shutdown(application: Application):
    all_exchanges = list(bot_data["exchanges"].values()) + list(bot_data["public_exchanges"].values())
    unique_exchanges = list({id(ex): ex for ex in all_exchanges}.values())
    await asyncio.gather(*[ex.close() for ex in unique_exchanges])
    logger.info("All exchange connections closed.")

def main():
    reset_file = os.path.join(APP_ROOT, '_reset_db_please.txt')
    if os.path.exists(reset_file):
        logger.warning("Reset file found! Deleting database and resetting.")
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            logger.info("Old database file deleted.")
        os.remove(reset_file)
        logger.info("Reset file deleted. Bot will start with a fresh database.")

    load_settings()
    init_database()

    request = HTTPXRequest(connect_timeout=60.0, read_timeout=60.0, pool_timeout=60.0)
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).post_init(post_init).post_shutdown(post_shutdown).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("check", check_trade_command))
    application.add_handler(CommandHandler("trade", manual_trade_command))
    
    application.add_handler(CallbackQueryHandler(manual_trade_button_handler, pattern="^manual_trade_"))
    application.add_handler(CallbackQueryHandler(tools_button_handler, pattern="^(balance|openorders|mytrades)_"))
    application.add_handler(CallbackQueryHandler(button_callback_handler))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    application.add_error_handler(error_handler)

    logger.info("Application configured. Starting polling...")
    application.run_polling()


if __name__ == '__main__':
    print("🚀 Starting Minesweeper Bot v5.0 (The Mastermind Release)...")
    try:
        main()
    except Exception as e:
        logging.critical(f"Bot stopped due to a critical unhandled error in the main loop: {e}", exc_info=True
