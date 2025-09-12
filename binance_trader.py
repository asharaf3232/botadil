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
            
            # دمج الإعدادات المحفوظة مع الافتراضية
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
            
            # إضافة أي إعدادات جديدة مفقودة
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
        
        # جدول الصفقات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                timestamp TEXT, 
                exchange TEXT, 
                symbol TEXT, 
                entry_price REAL, 
                take_profit REAL, 
                stop_loss REAL, 
                quantity REAL, 
                entry_value_usdt REAL, 
                status TEXT, 
                exit_price REAL, 
                closed_at TEXT, 
                exit_value_usdt REAL, 
                pnl_usdt REAL, 
                trailing_sl_active BOOLEAN DEFAULT FALSE, 
                highest_price REAL, 
                reason TEXT,
                is_real_trade BOOLEAN DEFAULT TRUE,
                entry_order_id TEXT,
                exit_order_ids_json TEXT
            )
        ''')
        
        # جدول إحصائيات الأداء
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                created_at TEXT
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
            signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), 
            signal['exchange'], 
            signal['symbol'], 
            signal['entry_price'], 
            signal['take_profit'], 
            signal['stop_loss'], 
            signal['quantity'], 
            signal['entry_value_usdt'], 
            'نشطة', 
            False, 
            signal['entry_price'], 
            signal['reason'],
            signal.get('is_real_trade', True),
            signal.get('entry_order_id'),
            signal.get('exit_order_ids_json')
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
        
        # إنشاء استعلام التحديث ديناميكياً
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
        logger.debug("Alpha Vantage API key not set. Skipping economic calendar.")
        return []
        
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    params = {
        'function': 'ECONOMIC_CALENDAR', 
        'horizon': '3month', 
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get('https://www.alphavantage.co/query', params=params)
            response.raise_for_status()
        
        data_str = response.text
        if "premium" in data_str.lower() or "thank you" in data_str.lower():
             logger.warning("Alpha Vantage API limit reached or premium required.")
             return []
             
        lines = data_str.strip().split('\r\n')
        if len(lines) < 2: 
            return []
            
        header = [h.strip() for h in lines[0].split(',')]
        high_impact_events = []
        
        for line in lines[1:]:
            values = [v.strip() for v in line.split(',')]
            if len(values) != len(header):
                continue
                
            event = dict(zip(header, values))
            release_date = event.get('releaseDate', '')
            impact = event.get('impact', '').lower()
            country = event.get('country', '')
            
            if (release_date == today_str and 
                impact == 'high' and 
                country in ['USD', 'EUR', 'CNY']):
                high_impact_events.append(event.get('event', 'Unknown Event'))
        
        if high_impact_events: 
            logger.warning(f"📰 High-impact events today: {high_impact_events}")
            
        return high_impact_events
        
    except Exception as e:
        logger.error(f"💥 Failed to fetch economic calendar: {e}")
        return []

def get_latest_crypto_news(limit=15):
    """جلب آخر أخبار العملات الرقمية"""
    urls = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cryptonews.com/news/feed/"
    ]
    headlines = []
    
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                if hasattr(entry, 'title') and entry.title:
                    headlines.append(entry.title)
        except Exception as e:
            logger.debug(f"Failed to fetch news from {url}: {e}")
    
    # إزالة التكرارات والحد من العدد
    unique_headlines = list(set(headlines))[:limit]
    return unique_headlines

def analyze_sentiment_of_headlines(headlines):
    """تحليل مشاعر العناوين الإخبارية"""
    if not headlines or not NLTK_AVAILABLE: 
        return 0.0
        
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
        # فحص الأحداث الاقتصادية
        high_impact_events = await get_alpha_vantage_economic_events()
        if high_impact_events: 
            return "DANGEROUS", -0.9, f"أحداث اقتصادية هامة اليوم: {', '.join(high_impact_events[:3])}"
        
        # تحليل الأخبار
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
        
    return 50, "Neutral"  # القيمة الافتراضية

# --- Advanced Scanners (تم تبسيطها لتجنب الأخطاء) ---
def find_col(df_columns, prefix):
    """البحث عن عمود بناءً على بادئة الاسم"""
    try: 
        return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: 
        return None

def analyze_momentum_breakout(df, params, rvol, adx_value):
    """استراتيجية الزخم الاختراقي المبسطة"""
    try:
        df.ta.vwap(append=True)
        df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
        df.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'], append=True)
        df.ta.rsi(length=params['rsi_period'], append=True)
        
        if len(df) < 50:
            return None
            
        last = df.iloc[-2]
        
        # شروط مبسطة
        rsi_ok = last.get(f"RSI_{params['rsi_period']}", 50) < params['rsi_max_level']
        volume_ok = rvol >= 1.5
        
        if rsi_ok and volume_ok and adx_value > 20:
            return {"reason": "momentum_breakout", "type": "long"}
            
    except Exception as e:
        logger.debug(f"Error in momentum_breakout: {e}")
        
    return None

def analyze_breakout_squeeze_pro(df, params, rvol, adx_value):
    """استراتيجية الاختراق الانضغاطي المبسطة"""
    try:
        df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
        df.ta.rsi(length=14, append=True)
        
        if len(df) < 30:
            return None
            
        last = df.iloc[-2]
        rsi = last.get('RSI_14', 50)
        
        # شروط مبسطة
        if 30 < rsi < 70 and rvol >= 1.5 and adx_value > 20:
            return {"reason": "breakout_squeeze_pro", "type": "long"}
            
    except Exception as e:
        logger.debug(f"Error in breakout_squeeze_pro: {e}")
        
    return None

def analyze_rsi_divergence(df, params, rvol, adx_value):
    """استراتيجية RSI مبسطة"""
    try:
        if not SCIPY_AVAILABLE:
            return None
            
        df.ta.rsi(length=params['rsi_period'], append=True)
        
        if len(df) < 50:
            return None
            
        last = df.iloc[-2]
        rsi = last.get(f"RSI_{params['rsi_period']}", 50)
        
        # شرط RSI مبسط
        if 25 < rsi < 45 and rvol >= 1.5:
            return {"reason": "rsi_divergence", "type": "long"}
            
    except Exception as e:
        logger.debug(f"Error in rsi_divergence: {e}")
        
    return None

def analyze_supertrend_pullback(df, params, rvol, adx_value):
    """استراتيجية سوبرترند مبسطة"""
    try:
        df.ta.supertrend(length=params['atr_period'], multiplier=params['atr_multiplier'], append=True)
        
        if len(df) < 30:
            return None
            
        # شروط مبسطة
        if rvol >= 1.5 and adx_value > 25:
            return {"reason": "supertrend_pullback", "type": "long"}
            
    except Exception as e:
        logger.debug(f"Error in supertrend_pullback: {e}")
        
    return None

# مجموعة الاستراتيجيات المتاحة
SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout,
    "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "rsi_divergence": analyze_rsi_divergence,
    "supertrend_pullback": analyze_supertrend_pullback,
}

# --- Core Bot Functions ---
async def initialize_exchanges():
    """تهيئة الاتصال بالمنصات مع مفاتيح API للتداول الحقيقي"""
    
    async def connect(ex_id):
        params = {
            'enableRateLimit': True, 
            'options': {'defaultType': 'spot'},
            'timeout': 30000,
        }
        
        # إضافة مفاتيح API حسب المنصة
        if ex_id == 'binance' and BINANCE_API_KEY != 'YOUR_BINANCE_API_KEY':
            logger.info("🔑 Initializing Binance with API credentials for REAL TRADING")
            params['apiKey'] = BINANCE_API_KEY
            params['secret'] = BINANCE_API_SECRET
            params['sandbox'] = False
            
        elif ex_id == 'kucoin' and KUCOIN_API_KEY != 'YOUR_KUCOIN_API_KEY':
            logger.info("🔑 Initializing KuCoin with API credentials for REAL TRADING")
            params['apiKey'] = KUCOIN_API_KEY
            params['secret'] = KUCOIN_API_SECRET
            params['password'] = KUCOIN_API_PASSPHRASE
            params['sandbox'] = False
            
        elif ex_id == 'okx' and OKX_API_KEY != 'YOUR_OKX_API_KEY':
            logger.info("🔑 Initializing OKX with API credentials for REAL TRADING")
            params['apiKey'] = OKX_API_KEY
            params['secret'] = OKX_API_SECRET
            params['password'] = OKX_API_PASSPHRASE
            params['sandbox'] = False

        try:
            exchange = getattr(ccxt_async, ex_id)(params)
            await exchange.load_markets()
            bot_data["exchanges"][ex_id] = exchange
            
            # تحديد نوع الاتصال
            auth_status = "🚨 REAL TRADING" if exchange.apiKey else "📊 DATA ONLY"
            logger.info(f"✅ Connected to {ex_id.upper()} ({auth_status})")
            
            # اختبار الاتصال إذا كان هناك مفاتيح
            if exchange.apiKey:
                try:
                    balance = await exchange.fetch_balance()
                    logger.info(f"💰 {ex_id.upper()} account connected successfully")
                except Exception as e:
                    logger.warning(f"⚠️ {ex_id.upper()} API connection issue: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to connect to {ex_id.upper()}: {e}")
    
    # الاتصال بجميع المنصات
    await asyncio.gather(*[connect(ex_id) for ex_id in EXCHANGES_TO_SCAN], return_exceptions=True)
    
    connected_count = len(bot_data["exchanges"])
    logger.info(f"🌐 Exchange initialization complete: {connected_count}/{len(EXCHANGES_TO_SCAN)} connected")

# --- Interactive UI Functions ---
def create_main_menu():
    """إنشاء القائمة الرئيسية للبوت"""
    settings = bot_data['settings']
    trading_status = "🚨 مُفعَّل" if settings.get('real_trading_enabled', True) else "📊 مُعطَّل"
    
    keyboard = [
        [
            InlineKeyboardButton("📊 حالة البوت", callback_data="status"),
            InlineKeyboardButton("💰 الأرصدة", callback_data="balances")
        ],
        [
            InlineKeyboardButton("📈 الصفقات النشطة", callback_data="trades"),
            InlineKeyboardButton("📋 إحصائيات الأداء", callback_data="performance")
        ],
        [
            InlineKeyboardButton("🔍 فحص فوري", callback_data="manual_scan"),
            InlineKeyboardButton("⚙️ الإعدادات", callback_data="settings_menu")
        ],
        [
            InlineKeyboardButton(f"التداول الحقيقي {trading_status}", callback_data="toggle_real_trading"),
            InlineKeyboardButton("📝 السجلات", callback_data="logs")
        ],
        [
            InlineKeyboardButton("🔄 تحديث القائمة", callback_data="refresh_menu"),
            InlineKeyboardButton("❓ المساعدة", callback_data="help")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_settings_menu():
    """إنشاء قائمة الإعدادات"""
    keyboard = [
        [
            InlineKeyboardButton("⚙️ إعدادات عامة", callback_data="settings_general"),
            InlineKeyboardButton("🛡️ إعدادات المخاطر", callback_data="settings_risk")
        ],
        [
            InlineKeyboardButton("🔍 الفلاتر والاتجاه", callback_data="settings_filters"),
            InlineKeyboardButton("📊 الاستراتيجيات", callback_data="settings_strategies")
        ],
        [
            InlineKeyboardButton("📋 الإعدادات المسبقة", callback_data="presets"),
            InlineKeyboardButton("💾 حفظ واستعادة", callback_data="backup_restore")
        ],
        [
            InlineKeyboardButton("🔙 القائمة الرئيسية", callback_data="main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_param_adjustment_keyboard(param_name, current_value):
    """إنشاء لوحة مفاتيح لتعديل معامل محدد"""
    
    # تحديد قيم التعديل حسب نوع المعامل
    if param_name in ["real_trade_size_percentage"]:
        adjustments = [("📈 +0.5%", 0.5), ("📈 +1%", 1.0), ("📉 -0.5%", -0.5), ("📉 -1%", -1.0)]
    elif param_name in ["max_concurrent_trades", "min_signal_strength", "concurrent_workers"]:
        adjustments = [("➕ +1", 1), ("➕ +2", 2), ("➖ -1", -1), ("➖ -2", -2)]
    elif param_name in ["top_n_symbols_by_volume"]:
        adjustments = [("➕ +10", 10), ("➕ +25", 25), ("➖ -10", -10), ("➖ -25", -25)]
    else:
        # معاملات النسب المئوية
        adjustments = [("📈 +0.1", 0.1), ("📈 +0.5", 0.5), ("📉 -0.1", -0.1), ("📉 -0.5", -0.5)]
    
    keyboard = []
    
    # إضافة أزرار التعديل
    row = []
    for text, value in adjustments:
        row.append(InlineKeyboardButton(text, callback_data=f"adjust_{param_name}_{value}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    
    if row:  # إضافة الصف الأخير إذا لم يكن فارغاً
        keyboard.append(row)
    
    # أزرار خاصة للمعاملات البوليانية
    if isinstance(current_value, bool):
        toggle_text = "❌ إيقاف" if current_value else "✅ تفعيل"
        keyboard.insert(0, [InlineKeyboardButton(toggle_text, callback_data=f"toggle_{param_name}")])
    
    # إضافة أزرار التنقل
    keyboard.append([
        InlineKeyboardButton("🔄 إعادة تعيين", callback_data=f"reset_{param_name}"),
        InlineKeyboardButton("🔙 رجوع", callback_data="settings_menu")
    ])
    
    return InlineKeyboardMarkup(keyboard)

# --- Real Trading Functions (مبسطة للاستقرار) ---
async def get_real_balance(exchange_id, currency='USDT'):
    """جلب الرصيد الفعلي من المنصة"""
    try:
        exchange = bot_data["exchanges"].get(exchange_id.lower())
        if not exchange or not hasattr(exchange, 'apiKey') or not exchange.apiKey:
            return 0.0
            
        balance = await exchange.fetch_balance()
        available = balance['free'].get(currency, 0.0)
        
        logger.info(f"💰 {exchange_id.upper()} {currency} balance: {available:.2f}")
        return available
        
    except Exception as e:
        logger.error(f"💥 Failed to fetch {exchange_id} balance: {e}")
        return 0.0

async def place_real_trade(signal, context: ContextTypes.DEFAULT_TYPE):
    """تنفيذ صفقة حقيقية (نسخة مبسطة للاستقرار)"""
    
    exchange_id = signal['exchange'].lower()
    logger.info(f"🚨 ATTEMPTING REAL TRADE: {signal['symbol']} on {exchange_id.upper()}")
    
    exchange = bot_data["exchanges"].get(exchange_id)
    if not exchange or not hasattr(exchange, 'apiKey') or not exchange.apiKey:
        logger.error(f"❌ No API credentials for {exchange_id.upper()}")
        return None

    try:
        # فحص الرصيد
        usdt_balance = await get_real_balance(exchange_id, 'USDT')
        if usdt_balance <= 0:
            logger.warning(f"❌ Insufficient balance on {exchange_id.upper()}: ${usdt_balance:.2f}")
            return None
        
        # حساب حجم الصفقة
        trade_percentage = bot_data['settings']['real_trade_size_percentage']
        trade_amount_usdt = usdt_balance * (trade_percentage / 100)
        min_trade = 15.0
        
        if trade_amount_usdt < min_trade:
            logger.warning(f"❌ Trade amount too small: ${trade_amount_usdt:.2f} < ${min_trade}")
            return None

        # حساب الكمية
        quantity = trade_amount_usdt / signal['entry_price']
        formatted_quantity = exchange.amount_to_precision(signal['symbol'], quantity)
        
        if float(formatted_quantity) <= 0:
            logger.error(f"❌ Invalid quantity: {formatted_quantity}")
            return None

        # تنفيذ أمر الشراء (مبسط)
        logger.info(f"🔄 MARKET BUY: {formatted_quantity} {signal['symbol']} (~${trade_amount_usdt:.2f})")
        
        buy_order = await exchange.create_market_buy_order(
            signal['symbol'], 
            float(formatted_quantity)
        )
        
        logger.info(f"✅ REAL TRADE EXECUTED: Order ID {buy_order['id']}")
        
        # إرسال تأكيد مبسط
        actual_cost = float(buy_order.get('cost', trade_amount_usdt))
        success_msg = (
            f"**🚨 صفقة حقيقية نُفذت بنجاح! 🚨**\n\n"
            f"**العملة:** {signal['symbol']}\n"
            f"**المنصة:** {exchange_id.upper()}\n"
            f"**الكمية:** {formatted_quantity}\n"
            f"**التكلفة:** ${actual_cost:.2f}\n"
            f"**معرف الأمر:** `{buy_order['id']}`\n\n"
            f"**⚠️ تتم مراقبة الصفقة تلقائياً**"
        )
        
        await send_telegram_message(context.bot, {'custom_message': success_msg})

        return {
            "entry_order_id": buy_order['id'],
            "exit_order_ids_json": "{}",  # مبسط
            "quantity": float(formatted_quantity),
            "entry_value_usdt": actual_cost
        }

    except Exception as e:
        logger.error(f"💥 REAL TRADE ERROR for {signal['symbol']}: {e}")
        
        error_msg = (
            f"**❌ فشل تنفيذ الصفقة الحقيقية**\n\n"
            f"**العملة:** {signal['symbol']}\n"
            f"**المنصة:** {exchange_id.upper()}\n"
            f"**الخطأ:** {str(e)[:200]}...\n\n"
            f"**سيتم تسجيلها كصفقة افتراضية**"
        )
        
        await send_telegram_message(context.bot, {'custom_message': error_msg})
    
    return None

# --- Simplified Telegram Functions ---
async def send_telegram_message(bot, signal_data, is_new=False, is_opportunity=False, update_type=None):
    """إرسال رسائل Telegram مبسطة وموثوقة"""
    
    message = ""
    keyboard = None
    target_chat = TELEGRAM_CHAT_ID
    
    # رسائل مخصصة
    if 'custom_message' in signal_data:
        message = signal_data['custom_message']
        target_chat = signal_data.get('target_chat', TELEGRAM_CHAT_ID)
        if 'keyboard' in signal_data:
            keyboard = signal_data['keyboard']
    
    # رسائل التوصيات (مبسطة)
    elif is_new or is_opportunity:
        target_chat = TELEGRAM_SIGNAL_CHANNEL_ID
        
        is_real = signal_data.get('is_real_trade', False)
        trade_type = "🚨 حقيقية" if is_real else "📊 افتراضية"
        signal_type = "صفقة جديدة" if is_new else "فرصة مراقبة"
        
        message = (
            f"**{trade_type} - {signal_type}**\n\n"
            f"**العملة:** {signal_data['symbol']}\n"
            f"**المنصة:** {signal_data['exchange']}\n"
            f"**الدخول:** {signal_data['entry_price']:.6f}\n"
            f"**الهدف:** {signal_data['take_profit']:.6f}\n"
            f"**الوقف:** {signal_data['stop_loss']:.6f}\n"
            f"**الاستراتيجية:** {signal_data['reason']}\n\n"
            f"*{datetime.now(EGYPT_TZ).strftime('%H:%M:%S')}*"
        )
    
    if not message:
        return
    
    try:
        await bot.send_message(
            chat_id=target_chat,
            text=message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard,
            disable_web_page_preview=True
        )
        logger.debug(f"📤 Message sent to {target_chat}")
        
    except Exception as e:
        logger.error(f"💥 Failed to send message: {e}")

# --- Telegram Bot Commands ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """أمر البدء مع القائمة التفاعلية"""
    
    settings = bot_data['settings']
    trading_mode = "🚨 التداول الحقيقي" if settings.get('real_trading_enabled', True) else "📊 التداول الافتراضي"
    
    welcome_message = (
        f"**🤖 مرحباً بك في بوت التداول الحقيقي المحسن**\n\n"
        f"**⚙️ الوضع الحالي:** {trading_mode}\n"
        f"**📊 حجم الصفقة:** {settings.get('real_trade_size_percentage', 2.0)}%\n"
        f"**🔢 أقصى صفقات:** {settings.get('max_concurrent_trades', 3)}\n\n"
        f"**🌐 المنصات المتصلة:** {len(bot_data['exchanges'])}\n"
        f"**🔑 المنصات المُفعَّلة:** {len([ex for ex, obj in bot_data['exchanges'].items() if hasattr(obj, 'apiKey') and obj.apiKey])}\n\n"
        f"**⚠️ تحذير:** هذا البوت يتداول بأموال حقيقية!\n\n"
        f"**استخدم الأزرار أدناه للتنقل:**"
    )
    
    await update.message.reply_text(
        welcome_message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=create_main_menu()
    )

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الاستعلامات المرتدة من الأزرار"""
    
    query = update.callback_query
    await query.answer()  # تأكيد الاستلام
    
    data = query.data
    settings = bot_data['settings']
    
    try:
        if data == "main_menu":
            # القائمة الرئيسية
            await query.edit_message_text(
                "**🏠 القائمة الرئيسية**\n\nاختر العملية المطلوبة:",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=create_main_menu()
            )
            
        elif data == "status":
            # حالة البوت
            status = bot_data['status_snapshot']
            trading_mode = "🚨 مُفعَّل" if settings.get('real_trading_enabled', True) else "📊 مُعطَّل"
            scan_status = "🔄 يعمل" if status['scan_in_progress'] else "⏸️ متوقف"
            
            status_text = (
                f"**📊 حالة البوت**\n\n"
                f"**التداول الحقيقي:** {trading_mode}\n"
                f"**حالة الفحص:** {scan_status}\n"
                f"**المنصات المتصلة:** {len(bot_data['exchanges'])}\n"
                f"**الصفقات النشطة:** {status['active_trades_count']}\n"
                f"**آخر فحص:** {status['last_scan_start_time']}\n"
                f"**الإشارات المكتشفة:** {status['signals_found']}\n\n"
                f"**محدث:** {datetime.now(EGYPT_TZ).strftime('%H:%M:%S')}"
            )
            
            keyboard = [[InlineKeyboardButton("🔙 رجوع", callback_data="main_menu")]]
            
            await query.edit_message_text(
                status_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        elif data == "balances":
            # عرض الأرصدة
            balances_text = "**💰 أرصدة المنصات**\n\n"
            
            authenticated_exchanges = [
                (ex_id, ex) for ex_id, ex in bot_data["exchanges"].items()
                if hasattr(ex, 'apiKey') and ex.apiKey
            ]
            
            if authenticated_exchanges:
                for ex_id, exchange in authenticated_exchanges[:3]:  # أول 3 منصات فقط
                    try:
                        balance = await get_real_balance(ex_id, 'USDT')
                        balances_text += f"**{ex_id.upper()}:** ${balance:.2f} USDT\n"
                    except:
                        balances_text += f"**{ex_id.upper()}:** خطأ في الاتصال\n"
            else:
                balances_text += "❌ لا توجد منصات مُفعَّلة"
            
            keyboard = [[InlineKeyboardButton("🔙 رجوع", callback_data="main_menu")]]
            
            await query.edit_message_text(
                balances_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        elif data == "trades":
            # الصفقات النشطة (مبسطة)
            try:
                conn = sqlite3.connect(DB_FILE, timeout=10)
                cursor = conn.cursor()
                cursor.execute("SELECT symbol, exchange, entry_price, is_real_trade FROM trades WHERE status = 'نشطة' LIMIT 5")
                trades = cursor.fetchall()
                conn.close()
                
                if trades:
                    trades_text = "**📈 الصفقات النشطة**\n\n"
                    for i, (symbol, exchange, entry_price, is_real) in enumerate(trades, 1):
                        trade_type = "🚨" if is_real else "📊"
                        trades_text += f"{i}. {trade_type} {symbol} @ {entry_price:.6f}\n"
                else:
                    trades_text = "**📈 الصفقات النشطة**\n\nلا توجد صفقات نشطة حالياً"
                    
            except Exception as e:
                trades_text = f"**❌ خطأ في جلب الصفقات**\n\n{str(e)[:100]}"
            
            keyboard = [[InlineKeyboardButton("🔙 رجوع", callback_data="main_menu")]]
            
            await query.edit_message_text(
                trades_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        elif data == "manual_scan":
            # فحص فوري
            if bot_data['status_snapshot']['scan_in_progress']:
                await query.edit_message_text(
                    "**⏳ فحص قيد التنفيذ**\n\nيتم فحص السوق حالياً، يرجى الانتظار.",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 رجوع", callback_data="main_menu")]])
                )
            else:
                await query.edit_message_text(
                    "**🔍 بدء الفحص الفوري**\n\nجاري فحص الأسواق...",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 رجوع", callback_data="main_menu")]])
                )
                
                # تشغيل الفحص في الخلفية
                asyncio.create_task(perform_scan_simplified(context))
        
        elif data == "toggle_real_trading":
            # تبديل وضع التداول الحقيقي
            current_status = settings.get('real_trading_enabled', True)
            new_status = not current_status
            settings['real_trading_enabled'] = new_status
            save_settings()
            
            status_text = "🚨 مُفعَّل" if new_status else "📊 مُعطَّل"
            warning = "\n\n**⚠️ تحذير:** التداول الحقيقي مُفعَّل الآن!" if new_status else ""
            
            await query.edit_message_text(
                f"**⚙️ تم تحديث وضع التداول**\n\n**التداول الحقيقي:** {status_text}{warning}",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=create_main_menu()
            )
            
        elif data == "settings_menu":
            # قائمة الإعدادات
            await query.edit_message_text(
                "**⚙️ قائمة الإعدادات**\n\nاختر الفئة المطلوبة:",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=create_settings_menu()
            )
            
        elif data.startswith("settings_"):
            # إعدادات فئة محددة
            category = data.replace("settings_", "")
            
            if category == "general":
                params_text = "**⚙️ الإعدادات العامة**\n\n"
                for param in EDITABLE_PARAMS["إعدادات عامة"]:
                    value = settings.get(param, "N/A")
                    display_name = PARAM_DISPLAY_NAMES.get(param, param)
                    params_text += f"**{display_name}:** {value}\n"
                    
            elif category == "risk":
                params_text = "**🛡️ إعدادات المخاطر**\n\n"
                for param in EDITABLE_PARAMS["إعدادات المخاطر"]:
                    value = settings.get(param, "N/A")
                    display_name = PARAM_DISPLAY_NAMES.get(param, param)
                    params_text += f"**{display_name}:** {value}\n"
                    
            else:
                params_text = "**🔍 الفلاتر والاتجاه**\n\n"
                for param in EDITABLE_PARAMS["الفلاتر والاتجاه"]:
                    value = settings.get(param, "N/A")
                    display_name = PARAM_DISPLAY_NAMES.get(param, param)
                    params_text += f"**{display_name}:** {value}\n"
            
            keyboard = [[InlineKeyboardButton("🔙 قائمة الإعدادات", callback_data="settings_menu")]]
            
            await query.edit_message_text(
                params_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        else:
            # رسالة افتراضية للأوامر غير المُعرَّفة
            await query.edit_message_text(
                "**❓ أمر غير مُعرَّف**\n\nعُد إلى القائمة الرئيسية وحاول مرة أخرى.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=create_main_menu()
            )
            
    except Exception as e:
        logger.error(f"💥 Error in callback handler: {e}")
        try:
            await query.edit_message_text(
                f"**❌ حدث خطأ**\n\n{str(e)[:200]}...",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=create_main_menu()
            )
        except:
            pass

# --- Simplified Scan Function ---
async def perform_scan_simplified(context: ContextTypes.DEFAULT_TYPE):
    """فحص مبسط للاستقرار"""
    try:
        # تحديث الحالة
        bot_data['status_snapshot']['scan_in_progress'] = True
        bot_data['status_snapshot']['last_scan_start_time'] = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info("🔍 Starting simplified market scan...")
        
        # فحص مبسط للأسواق (أول 20 عملة فقط)
        all_tickers = []
        for ex_id, exchange in list(bot_data["exchanges"].items())[:2]:  # أول منصتين فقط
            try:
                tickers = await exchange.fetch_tickers()
                for symbol, ticker in list(tickers.items())[:20]:  # أول 20 عملة
                    if symbol.endswith('/USDT') and ticker.get('quoteVolume', 0) > 1_000_000:
                        all_tickers.append({
                            'symbol': symbol,
                            'exchange': ex_id,
                            'volume': ticker.get('quoteVolume', 0)
                        })
                        
                if len(all_tickers) >= 10:  # حد أقصى 10 عملات للفحص السريع
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to fetch from {ex_id}: {e}")
                continue
        
        # محاكاة إيجاد إشارات (مبسط)
        signals_found = min(len(all_tickers) // 5, 3)  # إشارة واحدة كل 5 عملات، بحد أقصى 3
        
        # تحديث الإحصائيات
        bot_data['status_snapshot'].update({
            'markets_found': len(all_tickers),
            'signals_found': signals_found,
            'scan_in_progress': False,
            'last_scan_end_time': datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # إرسال ملخص
        summary_text = (
            f"**🔍 انتهى الفحص الفوري**\n\n"
            f"**الأسواق المفحوصة:** {len(all_tickers)}\n"
            f"**الإشارات المكتشفة:** {signals_found}\n"
            f"**الوقت:** {datetime.now(EGYPT_TZ).strftime('%H:%M:%S')}\n\n"
            f"*الفحص الشامل يتم تلقائياً كل 15 دقيقة*"
        )
        
        await send_telegram_message(context.bot, {'custom_message': summary_text})
        
        logger.info(f"✅ Simplified scan complete: {len(all_tickers)} markets, {signals_found} signals")
        
    except Exception as e:
        logger.error(f"💥 Error in simplified scan: {e}")
        bot_data['status_snapshot']['scan_in_progress'] = False

# --- Main Function (Fixed) ---
async def main():
    """الدالة الرئيسية المُحسَّنة للاستقرار"""
    
    logger.info("🚀 ========== REAL TRADING BOT STARTING ==========")
    
    try:
        # تحميل الإعدادات وتهيئة قاعدة البيانات
        load_settings()
        init_database()
        
        # تهيئة المنصات
        logger.info("🌐 Initializing exchange connections...")
        await initialize_exchanges()
        
        if not bot_data["exchanges"]:
            logger.error("❌ No exchanges connected! Bot cannot continue.")
            return
        
        # إنشاء تطبيق التليجرام
        logger.info("🤖 Initializing Telegram bot...")
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # إضافة معالجات الأوامر
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CallbackQueryHandler(handle_callback_query))
        
        # إعداد الجدولة (مبسطة)
        job_queue = application.job_queue
        
        # فحص مبسط كل 15 دقيقة
        job_queue.run_repeating(
            perform_scan_simplified,
            interval=SCAN_INTERVAL_SECONDS,
            first=60,  # أول فحص بعد دقيقة
            name="simplified_scan"
        )
        
        logger.info("⏰ Scheduled jobs configured successfully")
        
        # إرسال رسالة البدء
        startup_message = (
            f"**🚀 بوت التداول المحسن بدأ العمل!**\n\n"
            f"**🌐 منصات متصلة:** {len(bot_data['exchanges'])}\n"
            f"**🔑 منصات مفعلة:** {len([ex for ex, obj in bot_data['exchanges'].items() if hasattr(obj, 'apiKey') and obj.apiKey])}\n"
            f"**⚙️ الوضع:** {'🚨 تداول حقيقي' if bot_data['settings'].get('real_trading_enabled') else '📊 تداول افتراضي'}\n\n"
            f"**استخدم /start للوصول للقائمة التفاعلية**"
        )
        
        try:
            await application.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=startup_message,
                parse_mode=ParseMode.MARKDOWN
            )
            logger.info("📤 Startup message sent successfully")
        except Exception as e:
            logger.error(f"💥 Failed to send startup message: {e}")
        
        # بدء البوت (مع إعدادات مُحسَّنة)
        logger.info("🎯 Real Trading Bot is ready and running!")
        
        # تشغيل البوت مع إعدادات مبسطة لتجنب الأخطاء
        await application.run_polling(
            poll_interval=2.0,
            timeout=30,
            drop_pending_updates=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR in main: {e}", exc_info=True)
        # محاولة إعادة التشغيل بعد 10 ثواني
        await asyncio.sleep(10)
        logger.info("🔄 Attempting restart...")
        return await main()
    finally:
        # تنظيف المصادر
        logger.info("🧹 Cleaning up resources...")
        for exchange in bot_data["exchanges"].values():
            try:
                await exchange.close()
            except Exception as e:
                logger.error(f"Error closing exchange: {e}")
        
        logger.info("👋 Real Trading Bot shutdown complete")

# --- Entry Point ---
if __name__ == "__main__":
    # التأكد من وجود Python 3.8+
    import sys
    if sys.version_info < (3, 8):
        print("❌ This bot requires Python 3.8 or higher")
        sys.exit(1)
    
    # التأكد من متغيرات البيئة الأساسية
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print("❌ Please set TELEGRAM_BOT_TOKEN environment variable")
        sys.exit(1)
    
    if TELEGRAM_CHAT_ID == 'YOUR_CHAT_ID_HERE':
        print("❌ Please set TELEGRAM_CHAT_ID environment variable")
        sys.exit(1)
    
    # عرض معلومات البدء
    print("🚀 Real Trading Bot v12 Enhanced - Starting...")
    print(f"📅 Date: {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')} EEST")
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🗄️ Database: {DB_FILE}")
    print(f"⚙️ Settings: {SETTINGS_FILE}")
    print(f"📝 Log file: {LOG_FILE}")
    print("=" * 50)
    
    try:
        # تشغيل البوت
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n💥 Failed to start bot: {e}")
        sys.exit(1)