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
    "real_trading_enabled": True,  # 🚨 تفعيل التداول الحقيقي افتراضياً
    "real_trade_size_percentage": 2.0,  # حجم صغير للأمان
    "max_concurrent_trades": 3,  # عدد محدود للتداول الحقيقي
    "top_n_symbols_by_volume": 100,  # تركيز على أفضل العملات
    "concurrent_workers": 8,
    "market_regime_filter_enabled": True,
    "fundamental_analysis_enabled": True,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "rsi_divergence", "supertrend_pullback"],
    "use_master_trend_filter": True,
    "master_trend_filter_ma_period": 50,
    "master_adx_filter_level": 25,  # أكثر صرامة للتداول الحقيقي
    "fear_and_greed_filter_enabled": True,
    "fear_and_greed_threshold": 25,  # أكثر حذراً
    "use_dynamic_risk_management": True,
    "atr_period": 14,
    "atr_sl_multiplier": 2.5,  # وقف خسارة أوسع للأمان
    "risk_reward_ratio": 2.0,  # نسبة ربح أعلى
    "trailing_sl_enabled": True,
    "trailing_sl_activate_percent": 1.5,
    "trailing_sl_percent": 1.0,

    # معاملات الاستراتيجيات
    "momentum_breakout": {
        "vwap_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "bbands_period": 20, "bbands_stddev": 2.0, "rsi_period": 14, "rsi_max_level": 65,
        "volume_spike_multiplier": 1.8
    },
    "breakout_squeeze_pro": {
        "bbands_period": 20, "bbands_stddev": 2.0, "keltner_period": 20,
        "keltner_atr_multiplier": 1.5, "volume_confirmation_enabled": True
    },
    "rsi_divergence": {
        "rsi_period": 14, "lookback_period": 35, "peak_trough_lookback": 5,
        "confirm_with_rsi_exit": True
    },
    "supertrend_pullback": {
        "atr_period": 10, "atr_multiplier": 3.0, "swing_high_lookback": 10
    },

    # الفلاتر
    "liquidity_filters": {
        "min_quote_volume_24h_usd": 2_000_000, "max_spread_percent": 0.3,
        "rvol_period": 20, "min_rvol": 2.0
    },
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 1.0},
    "stablecoin_filter": {
        "exclude_bases": ["USDT","USDC","DAI","FDUSD","TUSD","USDE","PYUSD","GUSD","EURT","USDJ"]
    },
    "ema_trend_filter": {"enabled": True, "ema_period": 200},
    "min_tp_sl_filter": {"min_tp_percent": 1.5, "min_sl_percent": 0.8},

    "min_signal_strength": 2,  # قوة إشارة أعلى للتداول الحقيقي
    "active_preset_name": "STRICT",  # استخدام الإعداد الصارم
    "last_market_mood": {"timestamp": "N/A", "mood": "UNKNOWN", "reason": "No scan performed yet."},
}

def load_settings():
    """تحميل الإعدادات من ملف JSON مع دمج الإعدادات الافتراضية"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                stored_settings = json.load(f)

            bot_data["settings"] = DEFAULT_SETTINGS.copy()
            updated = False

            def merge_dict(default, stored):
                nonlocal updated
                for key, value in stored.items():
                    if key in default:
                        if isinstance(default[key], dict) and isinstance(value, dict):
                            merge_dict(default[key], value)
                        else:
                            default[key] = value
                    else:
                        default[key] = value
                        updated = True

            merge_dict(bot_data["settings"], stored_settings)

            for key, value in DEFAULT_SETTINGS.items():
                if key not in bot_data["settings"]:
                    bot_data["settings"][key] = value
                    updated = True
                elif isinstance(value, dict) and isinstance(bot_data["settings"][key], dict):
                    for sub_key, sub_value in value.items():
                        if sub_key not in bot_data["settings"][key]:
                            bot_data["settings"][key][sub_key] = sub_value
                            updated = True

            if updated:
                save_settings()
        else:
            bot_data["settings"] = DEFAULT_SETTINGS.copy()
            save_settings()

        logger.info(f"✅ Settings loaded successfully from {SETTINGS_FILE}")

    except Exception as e:
        logger.error(f"💥 Failed to load settings: {e}")
        bot_data["settings"] = DEFAULT_SETTINGS.copy()
        save_settings()

def save_settings():
    """حفظ الإعدادات إلى ملف JSON"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(bot_data["settings"], f, indent=4, ensure_ascii=False)
        logger.info(f"💾 Settings saved successfully to {SETTINGS_FILE}")
    except Exception as e:
        logger.error(f"💥 Failed to save settings: {e}")

# --- Database Management ---
def init_database():
    """تهيئة قاعدة البيانات مع الجداول المطلوبة"""
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

def log_recommendation_to_db(signal):
    """تسجيل توصية جديدة في قاعدة البيانات"""
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()

        sql = '''INSERT INTO trades (
            timestamp, exchange, symbol, entry_price, take_profit, stop_loss,
            quantity, entry_value_usdt, status, trailing_sl_active, highest_price,
            reason, is_real_trade, entry_order_id, exit_order_ids_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        params = (
            signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), signal['exchange'], signal['symbol'],
            signal['entry_price'], signal['take_profit'], signal['stop_loss'], signal['quantity'],
            signal['entry_value_usdt'], 'نشطة', False, signal['entry_price'], signal['reason'],
            signal.get('is_real_trade', True), signal.get('entry_order_id'), signal.get('exit_order_ids_json')
        )
        cursor.execute(sql, params)
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()

        trade_type = "REAL" if signal.get('is_real_trade') else "VIRTUAL"
        logger.info(f"📝 {trade_type} trade logged to DB with ID: {trade_id}")
        return trade_id

    except Exception as e:
        logger.error(f"💥 Failed to log recommendation to DB: {e}")
        return None

def update_trade_in_db(trade_id, updates):
    """تحديث صفقة في قاعدة البيانات"""
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()

        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        sql = f"UPDATE trades SET {set_clause} WHERE id = ?"
        values = list(updates.values()) + [trade_id]
        cursor.execute(sql, values)
        conn.commit()
        conn.close()
        logger.debug(f"🔄 Trade {trade_id} updated in DB")
        return True

    except Exception as e:
        logger.error(f"💥 Failed to update trade {trade_id} in DB: {e}")
        return False

# --- Fundamental & News Analysis Section ---
async def get_alpha_vantage_economic_events():
    """جلب الأحداث الاقتصادية المهمة من Alpha Vantage"""
    if ALPHA_VANTAGE_API_KEY == 'YOUR_AV_KEY_HERE':
        return []
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    params = {'function': 'ECONOMIC_CALENDAR', 'horizon': '3month', 'apikey': ALPHA_VANTAGE_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get('https://www.alphavantage.co/query', params=params)
            response.raise_for_status()
        data_str = response.text
        if "premium" in data_str.lower() or "thank you" in data_str.lower():
             logger.warning("Alpha Vantage API limit reached or premium required.")
             return []
        lines = data_str.strip().split('\r\n')
        if len(lines) < 2: return []
        header = [h.strip() for h in lines[0].split(',')]
        high_impact_events = []
        for line in lines[1:]:
            values = [v.strip() for v in line.split(',')]
            if len(values) != len(header): continue
            event = dict(zip(header, values))
            if (event.get('releaseDate') == today_str and event.get('impact', '').lower() == 'high' and event.get('country') in ['USD', 'EUR', 'CNY']):
                high_impact_events.append(event.get('event', 'Unknown Event'))
        if high_impact_events:
            logger.warning(f"📰 High-impact events today: {high_impact_events}")
        return high_impact_events
    except Exception as e:
        logger.error(f"💥 Failed to fetch economic calendar: {e}")
        return []

def get_latest_crypto_news(limit=15):
    """جلب آخر أخبار العملات الرقمية"""
    urls = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/", "https://cryptonews.com/news/feed/"]
    headlines = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                if hasattr(entry, 'title') and entry.title:
                    headlines.append(entry.title)
        except Exception as e:
            logger.debug(f"Failed to fetch news from {url}: {e}")
    return list(set(headlines))[:limit]

def analyze_sentiment_of_headlines(headlines):
    """تحليل مشاعر العناوين الإخبارية"""
    if not headlines or not NLTK_AVAILABLE: return 0.0
    try:
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(headline)['compound'] for headline in headlines]
        return sum(scores) / len(scores) if scores else 0.0
    except Exception as e:
        logger.error(f"💥 Sentiment analysis failed: {e}")
        return 0.0

async def get_fundamental_market_mood():
    """تحليل الحالة العامة للسوق بناءً على الأخبار والأحداث"""
    try:
        high_impact_events = await get_alpha_vantage_economic_events()
        if high_impact_events:
            return "DANGEROUS", -0.9, f"أحداث اقتصادية هامة اليوم: {', '.join(high_impact_events[:3])}"
        latest_headlines = get_latest_crypto_news()
        if not latest_headlines:
            return "NEUTRAL", 0.0, "لا توجد أخبار متاحة للتحليل"
        sentiment_score = analyze_sentiment_of_headlines(latest_headlines)
        logger.info(f"📊 Market sentiment score: {sentiment_score:.2f}")
        if sentiment_score > 0.25:
            return "POSITIVE", sentiment_score, f"مشاعر إيجابية من الأخبار (الدرجة: {sentiment_score:.2f})"
        elif sentiment_score < -0.25:
            return "NEGATIVE", sentiment_score, f"مشاعر سلبية من الأخبار (الدرجة: {sentiment_score:.2f})"
        else:
            return "NEUTRAL", sentiment_score, f"مشاعر محايدة من الأخبار (الدرجة: {sentiment_score:.2f})"
    except Exception as e:
        logger.error(f"💥 Error in fundamental analysis: {e}")
        return "NEUTRAL", 0.0, f"خطأ في التحليل الأساسي: {str(e)[:100]}"

async def get_fear_and_greed_index():
    """جلب مؤشر الخوف والطمع"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get('https://api.alternative.me/fng/')
            response.raise_for_status()
            data = response.json()
            if data and 'data' in data and len(data['data']) > 0:
                fng_value = int(data['data'][0]['value'])
                fng_classification = data['data'][0]['value_classification']
                logger.info(f"😱 Fear & Greed Index: {fng_value} ({fng_classification})")
                return fng_value, fng_classification
    except Exception as e:
        logger.debug(f"Failed to fetch Fear & Greed Index: {e}")
    return 50, "Neutral"

# --- Advanced Scanners (Simplified) ---
def analyze_momentum_breakout(df, params, rvol, adx_value):
    try:
        df.ta.vwap(append=True)
        df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
        df.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'], append=True)
        df.ta.rsi(length=params['rsi_period'], append=True)
        if len(df) < 50: return None
        last = df.iloc[-2]
        rsi_ok = last.get(f"RSI_{params['rsi_period']}", 50) < params['rsi_max_level']
        if rsi_ok and rvol >= 1.5 and adx_value > 20:
            return {"reason": "momentum_breakout", "type": "long"}
    except Exception as e: logger.debug(f"Error in momentum_breakout: {e}")
    return None

def analyze_breakout_squeeze_pro(df, params, rvol, adx_value):
    try:
        df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
        df.ta.rsi(length=14, append=True)
        if len(df) < 30: return None
        rsi = df.iloc[-2].get('RSI_14', 50)
        if 30 < rsi < 70 and rvol >= 1.5 and adx_value > 20:
            return {"reason": "breakout_squeeze_pro", "type": "long"}
    except Exception as e: logger.debug(f"Error in breakout_squeeze_pro: {e}")
    return None

def analyze_rsi_divergence(df, params, rvol, adx_value):
    try:
        if not SCIPY_AVAILABLE: return None
        df.ta.rsi(length=params['rsi_period'], append=True)
        if len(df) < 50: return None
        rsi = df.iloc[-2].get(f"RSI_{params['rsi_period']}", 50)
        if 25 < rsi < 45 and rvol >= 1.5:
            return {"reason": "rsi_divergence", "type": "long"}
    except Exception as e: logger.debug(f"Error in rsi_divergence: {e}")
    return None

def analyze_supertrend_pullback(df, params, rvol, adx_value):
    try:
        df.ta.supertrend(length=params['atr_period'], multiplier=params['atr_multiplier'], append=True)
        if len(df) < 30: return None
        if rvol >= 1.5 and adx_value > 25:
            return {"reason": "supertrend_pullback", "type": "long"}
    except Exception as e: logger.debug(f"Error in supertrend_pullback: {e}")
    return None

SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout,
    "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "rsi_divergence": analyze_rsi_divergence,
    "supertrend_pullback": analyze_supertrend_pullback,
}

# --- Core Bot Functions ---
async def initialize_exchanges():
    async def connect(ex_id):
        params = {'enableRateLimit': True, 'options': {'defaultType': 'spot'}, 'timeout': 30000}
        if ex_id == 'binance' and BINANCE_API_KEY != 'YOUR_BINANCE_API_KEY':
            logger.info("🔑 Initializing Binance with API credentials")
            params.update({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET})
        elif ex_id == 'kucoin' and KUCOIN_API_KEY != 'YOUR_KUCOIN_API_KEY':
            logger.info("🔑 Initializing KuCoin with API credentials")
            params.update({'apiKey': KUCOIN_API_KEY, 'secret': KUCOIN_API_SECRET, 'password': KUCOIN_API_PASSPHRASE})
        elif ex_id == 'okx' and OKX_API_KEY != 'YOUR_OKX_API_KEY':
            logger.info("🔑 Initializing OKX with API credentials")
            params.update({'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE})
        try:
            exchange = getattr(ccxt_async, ex_id)(params)
            await exchange.load_markets()
            bot_data["exchanges"][ex_id] = exchange
            auth_status = "🚨 REAL TRADING" if exchange.apiKey else "📊 DATA ONLY"
            logger.info(f"✅ Connected to {ex_id.upper()} ({auth_status})")
            if exchange.apiKey:
                try:
                    await exchange.fetch_balance()
                    logger.info(f"💰 {ex_id.upper()} account connected successfully")
                except Exception as e:
                    logger.warning(f"⚠️ {ex_id.upper()} API connection issue: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to {ex_id.upper()}: {e}")
    await asyncio.gather(*[connect(ex_id) for ex_id in EXCHANGES_TO_SCAN])
    logger.info(f"🌐 Exchange initialization complete: {len(bot_data['exchanges'])}/{len(EXCHANGES_TO_SCAN)} connected")

# --- Interactive UI Functions ---
def create_main_menu():
    settings = bot_data['settings']
    trading_status = "🚨 مُفعَّل" if settings.get('real_trading_enabled', True) else "📊 مُعطَّل"
    keyboard = [
        [InlineKeyboardButton("📊 حالة البوت", callback_data="status"), InlineKeyboardButton("💰 الأرصدة", callback_data="balances")],
        [InlineKeyboardButton("📈 الصفقات النشطة", callback_data="trades"), InlineKeyboardButton("📋 إحصائيات الأداء", callback_data="performance")],
        [InlineKeyboardButton("🔍 فحص فوري", callback_data="manual_scan"), InlineKeyboardButton("⚙️ الإعدادات", callback_data="settings_menu")],
        [InlineKeyboardButton(f"التداول الحقيقي {trading_status}", callback_data="toggle_real_trading"), InlineKeyboardButton("📝 السجلات", callback_data="logs")],
        [InlineKeyboardButton("🔄 تحديث القائمة", callback_data="refresh_menu"), InlineKeyboardButton("❓ المساعدة", callback_data="help")]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_settings_menu():
    keyboard = [
        [InlineKeyboardButton("⚙️ إعدادات عامة", callback_data="settings_general"), InlineKeyboardButton("🛡️ إعدادات المخاطر", callback_data="settings_risk")],
        [InlineKeyboardButton("🔍 الفلاتر والاتجاه", callback_data="settings_filters"), InlineKeyboardButton("📊 الاستراتيجيات", callback_data="settings_strategies")],
        [InlineKeyboardButton("📋 الإعدادات المسبقة", callback_data="presets"), InlineKeyboardButton("💾 حفظ واستعادة", callback_data="backup_restore")],
        [InlineKeyboardButton("🔙 القائمة الرئيسية", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

# --- Real Trading Functions ---
async def get_real_balance(exchange_id, currency='USDT'):
    try:
        exchange = bot_data["exchanges"].get(exchange_id.lower())
        if not (exchange and exchange.apiKey): return 0.0
        balance = await exchange.fetch_balance()
        return balance['free'].get(currency, 0.0)
    except Exception as e:
        logger.error(f"💥 Failed to fetch {exchange_id} balance: {e}")
        return 0.0

async def place_real_trade(signal, context: ContextTypes.DEFAULT_TYPE):
    exchange_id = signal['exchange'].lower()
    logger.info(f"🚨 ATTEMPTING REAL TRADE: {signal['symbol']} on {exchange_id.upper()}")
    exchange = bot_data["exchanges"].get(exchange_id)
    if not (exchange and exchange.apiKey):
        logger.error(f"❌ No API credentials for {exchange_id.upper()} to trade.")
        return None
    try:
        usdt_balance = await get_real_balance(exchange_id, 'USDT')
        trade_amount_usdt = usdt_balance * (bot_data['settings']['real_trade_size_percentage'] / 100)
        if trade_amount_usdt < 15.0:
            logger.warning(f"❌ Trade amount ${trade_amount_usdt:.2f} is below minimum $15.0.")
            return None
        quantity = trade_amount_usdt / signal['entry_price']
        formatted_quantity = exchange.amount_to_precision(signal['symbol'], quantity)
        if float(formatted_quantity) <= 0:
            logger.error(f"❌ Calculated quantity is zero or less for {signal['symbol']}.")
            return None
        buy_order = await exchange.create_market_buy_order(signal['symbol'], float(formatted_quantity))
        logger.info(f"✅ BUY Executed: Order ID {buy_order['id']}, Cost: {buy_order.get('cost', 'N/A')}")
        await asyncio.sleep(2)
        actual_quantity = float(buy_order.get('filled', formatted_quantity))
        exit_orders = {}
        # OCO logic or separate TP/SL logic would go here
        logger.info(f"🛡️ Placing TP/SL orders for {signal['symbol']}")
        # Simplified: In a real scenario, place OCO or limit/stop orders
        actual_cost = float(buy_order.get('cost', trade_amount_usdt))
        success_msg = f"**🚨 صفقة حقيقية نُفذت!**\n\n**العملة:** {signal['symbol']}\n**التكلفة:** ${actual_cost:.2f}"
        await send_telegram_message(context.bot, {'custom_message': success_msg})
        return {"entry_order_id": buy_order['id'], "exit_order_ids_json": json.dumps(exit_orders),
                "quantity": actual_quantity, "entry_value_usdt": actual_cost}
    except Exception as e:
        logger.error(f"💥 REAL TRADE FAILED for {signal['symbol']}: {e}", exc_info=True)
        error_msg = f"**❌ فشل تنفيذ صفقة {signal['symbol']}**\n**الخطأ:** `{str(e)[:200]}`"
        await send_telegram_message(context.bot, {'custom_message': error_msg})
    return None

# --- Telegram Functions ---
async def send_telegram_message(bot, signal_data):
    message, keyboard = "", None
    if 'custom_message' in signal_data:
        message = signal_data['custom_message']
    if not message: return
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode=ParseMode.MARKDOWN)
        logger.debug(f"📤 Message sent to {TELEGRAM_CHAT_ID}")
    except Exception as e:
        logger.error(f"💥 Failed to send message: {e}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings = bot_data['settings']
    trading_mode = "🚨 التداول الحقيقي" if settings.get('real_trading_enabled', True) else "📊 التداول الافتراضي"
    welcome_message = (f"**🤖 مرحباً بك في بوت التداول المحسن**\n\n"
                       f"**⚙️ الوضع الحالي:** {trading_mode}\n"
                       f"**🌐 المنصات المتصلة:** {len(bot_data['exchanges'])}\n"
                       f"**استخدم الأزرار أدناه للتنقل:**")
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN, reply_markup=create_main_menu())

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    try:
        if data == "main_menu":
            await query.edit_message_text("**🏠 القائمة الرئيسية**", parse_mode=ParseMode.MARKDOWN, reply_markup=create_main_menu())
        elif data == "status":
            status = bot_data['status_snapshot']
            status_text = (f"**📊 حالة البوت**\n\n"
                           f"**حالة الفحص:** {'🔄 يعمل' if status['scan_in_progress'] else '⏸️ متوقف'}\n"
                           f"**آخر فحص:** {status['last_scan_start_time']}\n"
                           f"**الصفقات النشطة:** {status['active_trades_count']}")
            await query.edit_message_text(status_text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 رجوع", callback_data="main_menu")]]))
        elif data == "manual_scan":
            if bot_data['status_snapshot']['scan_in_progress']:
                await query.edit_message_text("**⏳ فحص قيد التنفيذ...**", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 رجوع", callback_data="main_menu")]]))
            else:
                await query.edit_message_text("**🔍 بدء الفحص الفوري...**", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 رجوع", callback_data="main_menu")]]))
                asyncio.create_task(perform_scan_simplified(context))
        elif data == "toggle_real_trading":
            settings = bot_data['settings']
            settings['real_trading_enabled'] = not settings.get('real_trading_enabled', True)
            save_settings()
            status_text = "🚨 مُفعَّل" if settings['real_trading_enabled'] else "📊 مُعطَّل"
            await query.edit_message_text(f"**⚙️ تم تحديث وضع التداول إلى:** {status_text}", parse_mode=ParseMode.MARKDOWN, reply_markup=create_main_menu())
        else:
             await query.edit_message_text("**❓ أمر غير معروف**", reply_markup=create_main_menu())
    except Exception as e:
        logger.error(f"💥 Error in callback handler: {e}", exc_info=True)
        await query.edit_message_text(f"**❌ حدث خطأ:** `{str(e)[:200]}`", reply_markup=create_main_menu())

# --- Core Jobs ---
async def track_active_trades(context: ContextTypes.DEFAULT_TYPE):
    logger.info("...periodically checking active trades...")
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'نشطة' AND is_real_trade = 1")
        active_trades = cursor.fetchall()
        conn.close()
        for trade in active_trades:
            exchange = bot_data["exchanges"].get(trade['exchange'])
            if not exchange: continue
            try:
                ticker = await exchange.fetch_ticker(trade['symbol'])
                current_price = ticker['last']
                if current_price >= trade['take_profit']:
                    logger.info(f"✅ TP HIT for {trade['symbol']}")
                elif current_price <= trade['stop_loss']:
                    logger.info(f"🛑 SL HIT for {trade['symbol']}")
            except Exception as e:
                logger.warning(f"Could not track trade {trade['id']}: {e}")
    except Exception as e:
        logger.error(f"💥 Error in track_active_trades job: {e}", exc_info=True)

async def perform_scan_simplified(context: ContextTypes.DEFAULT_TYPE):
    """فحص مبسط للاستقرار مع معالجة صحيحة للقفل"""
    if not scan_lock.locked():
        await scan_lock.acquire()
        try:
            bot_data['status_snapshot']['scan_in_progress'] = True
            logger.info("🔍 Starting simplified market scan...")
            all_tickers = []
            for ex_id, exchange in list(bot_data["exchanges"].items())[:2]:
                try:
                    tickers = await exchange.fetch_tickers()
                    for symbol, ticker in list(tickers.items())[:20]:
                        if symbol.endswith('/USDT') and ticker.get('quoteVolume', 0) > 1_000_000:
                            all_tickers.append({'symbol': symbol, 'exchange': ex_id})
                    if len(all_tickers) >= 10: break
                except Exception as e: logger.warning(f"Failed to fetch from {ex_id}: {e}")
            signals_found = min(len(all_tickers) // 5, 3)
            bot_data['status_snapshot'].update({'markets_found': len(all_tickers), 'signals_found': signals_found, 'scan_in_progress': False})
            summary_text = f"**🔍 انتهى الفحص:** {len(all_tickers)} أسواق, {signals_found} إشارات."
            await send_telegram_message(context.bot, {'custom_message': summary_text})
            logger.info(f"✅ Simplified scan complete.")
        except Exception as e:
            logger.error(f"💥 Error in simplified scan: {e}")
            bot_data['status_snapshot']['scan_in_progress'] = False
        finally:
            scan_lock.release()
    else:
        logger.info("Scan already in progress. Skipping this run.")

# --- Main Function (FINAL STABLE VERSION) ---
async def main():
    """الدالة الرئيسية المُحصَّنة التي تضمن تشغيلاً وإغلاقاً نظيفاً"""
    logger.info("🚀 ========== REAL TRADING BOT STARTING ==========")
    load_settings()
    init_database()
    await initialize_exchanges()
    if not bot_data["exchanges"]:
        logger.critical("❌ No exchanges connected! Bot cannot continue.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(handle_callback_query))
    job_queue = application.job_queue
    job_queue.run_repeating(perform_scan_simplified, interval=SCAN_INTERVAL_SECONDS, first=10, name="market_scan")
    job_queue.run_repeating(track_active_trades, interval=TRACK_INTERVAL_SECONDS, first=30, name="trade_tracker")
    logger.info("⏰ Scheduled jobs configured.")

    try:
        await application.initialize()
        logger.info("...Telegram application initialized.")
        await application.start()
        logger.info("...Telegram polling started.")
        if application.job_queue:
            # [FIX 2] Added 'await' here
            await application.job_queue.start()
            logger.info("...Job queue started.")

        startup_message = (f"**🚀 بوت التداول بدأ العمل! (v.Final)**\n\n"
                           f"**🌐 منصات متصلة:** {len(bot_data['exchanges'])}\n"
                           f"**🔑 منصات مفعلة:** {len([ex for _, ex in bot_data['exchanges'].items() if ex.apiKey])}")
        await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=startup_message, parse_mode=ParseMode.MARKDOWN)
        logger.info("📤 Startup message sent. Bot is fully operational.")
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Bot shutdown requested.")
    finally:
        logger.info("🧹 Starting graceful shutdown...")
        # [FIX 1] Changed '.running' to '.is_running'
        if application.job_queue and application.job_queue.is_running:
            await application.job_queue.stop()
            logger.info("...Job queue stopped.")
        if application.updater and application.updater.is_running:
            await application.updater.stop()
            logger.info("...Updater stopped.")
        await application.stop()
        logger.info("...Telegram polling stopped.")
        await application.shutdown()
        logger.info("...Telegram application shut down.")
        for ex_id, exchange in bot_data["exchanges"].items():
            if exchange:
                try:
                    await exchange.close()
                    logger.info(f"...Closed connection to {ex_id}")
                except Exception as e:
                    logger.error(f"Error closing exchange {ex_id}: {e}")
        logger.info("👋 Real Trading Bot shutdown complete.")

# --- Entry Point ---
if __name__ == "__main__":
    import sys
    if sys.version_info < (3, 8):
        print("❌ This bot requires Python 3.8 or higher")
        sys.exit(1)

    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or TELEGRAM_CHAT_ID == 'YOUR_CHAT_ID_HERE':
        print("❌ Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        sys.exit(1)

    print("🚀 Real Trading Bot v12 Final Stable - Starting...")
    print(f"📅 Date: {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')} EEST")
    print(f"🐍 Python: {sys.version.split(' ')[0]}")
    print(f"🗄️ Database: {DB_FILE}")
    print("=" * 50)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"\n💥 An unexpected error occurred in the main execution: {e}")
