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
    # تحميل البيانات المطلوبة
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
        "min_signal_strength"
    ],
    "إعدادات المخاطر": [
        "real_trading_enabled", "real_trade_size_percentage", "atr_sl_multiplier", "risk_reward_ratio",
        "trailing_sl_activate_percent", "trailing_sl_percent"
    ],
    "الفلاتر والاتجاه": [
        "market_regime_filter_enabled", "use_master_trend_filter", "fear_and_greed_filter_enabled",
        "master_adx_filter_level", "master_trend_filter_ma_period", "trailing_sl_enabled", "fear_and_greed_threshold",
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

# --- [API UPGRADE] Fundamental & News Analysis Section ---
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

async def check_market_regime():
    """فحص حالة السوق العامة بناءً على BTC وعوامل أخرى"""
    try:
        settings = bot_data['settings']
        
        # العثور على منصة متاحة لفحص BTC
        btc_exchange = None
        for ex_name in ['binance', 'okx', 'bybit']:
            if ex_name in bot_data["exchanges"]:
                btc_exchange = bot_data["exchanges"][ex_name]
                break
                
        if not btc_exchange:
            return True, "لا توجد منصة متاحة لفحص BTC - السماح بالتداول"
        
        # جلب بيانات BTC/USDT
        try:
            ohlcv = await btc_exchange.fetch_ohlcv('BTC/USDT', '1h', limit=50)
        except Exception as e:
            logger.warning(f"Failed to fetch BTC data: {e}")
            return True, "فشل جلب بيانات BTC - السماح بالتداول"
            
        if len(ohlcv) < 20:
            return True, "بيانات BTC غير كافية - السماح بالتداول"
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # حساب المؤشرات الفنية
        df.ta.sma(length=20, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.adx(append=True)
        
        last_candle = df.iloc[-1]
        current_price = last_candle['close']
        sma20 = last_candle.get('SMA_20', current_price)
        rsi = last_candle.get('RSI_14', 50)
        adx = last_candle.get('ADX_14', 20)
        
        # فحص مؤشر الخوف والطمع
        if settings.get('fear_and_greed_filter_enabled', True):
            fng_value, fng_classification = await get_fear_and_greed_index()
            threshold = settings.get('fear_and_greed_threshold', 25)
            
            if fng_value < threshold:
                return False, f"مؤشر الخوف والطمع منخفض: {fng_value} ({fng_classification})"
        
        # تحليل الحالة الفنية
        is_above_sma = current_price > sma20
        is_oversold = rsi < 30
        is_overbought = rsi > 70
        is_trending = adx > settings.get('master_adx_filter_level', 25)
        
        # منطق القرار
        if is_overbought and not is_above_sma:
            return False, f"BTC في منطقة تشبع شرائي وتحت SMA20 (RSI: {rsi:.1f})"
            
        if is_oversold and is_trending:
            return False, f"BTC في منطقة تشبع بيعي مع اتجاه قوي (RSI: {rsi:.1f}, ADX: {adx:.1f})"
            
        if not is_trending and not is_above_sma:
            return False, f"BTC في حالة تذبذب وتحت SMA20 (ADX: {adx:.1f})"
            
        return True, f"حالة السوق مناسبة (BTC فوق SMA20: {is_above_sma}, RSI: {rsi:.1f}, ADX: {adx:.1f})"
        
    except Exception as e:
        logger.error(f"💥 Error in check_market_regime: {e}")
        return True, "خطأ في فحص السوق - السماح بالتداول للأمان"

# --- Advanced Scanners ---
def find_col(df_columns, prefix):
    """البحث عن عمود بناءً على بادئة الاسم"""
    try: 
        return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: 
        return None

def analyze_momentum_breakout(df, params, rvol, adx_value):
    """استراتيجية الزخم الاختراقي"""
    try:
        df.ta.vwap(append=True)
        df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
        df.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'], append=True)
        df.ta.rsi(length=params['rsi_period'], append=True)
        
        # البحث عن الأعمدة المطلوبة
        macd_col = find_col(df.columns, f"MACD_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}")
        macds_col = find_col(df.columns, f"MACDs_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}")
        bbu_col = find_col(df.columns, f"BBU_{params['bbands_period']}_")
        rsi_col = find_col(df.columns, f"RSI_{params['rsi_period']}")
        
        if not all([macd_col, macds_col, bbu_col, rsi_col]): 
            return None
            
        last, prev = df.iloc[-2], df.iloc[-3]
        
        # الشروط
        macd_crossover = prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col]
        price_above_bb = last['close'] > last[bbu_col]
        price_above_vwap = last['close'] > last["VWAP_D"]
        rsi_not_overbought = last[rsi_col] < params['rsi_max_level']
        rvol_ok = rvol >= bot_data['settings']['liquidity_filters']['min_rvol']
        
        # تأكيد الحجم (للتداول الحقيقي)
        volume_spike = df['volume'].iloc[-2] > df['volume'].rolling(20).mean().iloc[-2] * params.get('volume_spike_multiplier', 1.8)
        
        if all([macd_crossover, price_above_bb, price_above_vwap, rsi_not_overbought, rvol_ok, volume_spike]):
            return {"reason": "momentum_breakout", "type": "long"}
            
    except Exception as e:
        logger.debug(f"Error in momentum_breakout: {e}")
        
    return None

def analyze_breakout_squeeze_pro(df, params, rvol, adx_value):
    """استراتيجية الاختراق الانضغاطي المحسنة"""
    try:
        df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
        df.ta.kc(length=params['keltner_period'], scalar=params['keltner_atr_multiplier'], append=True)
        df.ta.obv(append=True)
        
        # البحث عن الأعمدة
        bbu_col = find_col(df.columns, f"BBU_{params['bbands_period']}_")
        bbl_col = find_col(df.columns, f"BBL_{params['bbands_period']}_")
        kcu_col = find_col(df.columns, f"KCUe_{params['keltner_period']}_")
        kcl_col = find_col(df.columns, f"KCLe_{params['keltner_period']}_")
        
        if not all([bbu_col, bbl_col, kcu_col, kcl_col]): 
            return None
            
        last, prev = df.iloc[-2], df.iloc[-3]
        
        # فحص حالة الانضغاط
        is_in_squeeze = (prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col])
        
        if is_in_squeeze:
            breakout_fired = last['close'] > last[bbu_col]
            rvol_ok = rvol >= bot_data['settings']['liquidity_filters']['min_rvol']
            
            # تأكيد الحجم
            volume_ok = True
            if params.get('volume_confirmation_enabled', True):
                avg_volume = df['volume'].rolling(20).mean().iloc[-2]
                volume_ok = last['volume'] > avg_volume * 1.8
            
            # تأكيد OBV
            obv_rising = df['OBV'].iloc[-2] > df['OBV'].iloc[-3]
            
            if breakout_fired and rvol_ok and volume_ok and obv_rising:
                return {"reason": "breakout_squeeze_pro", "type": "long"}
                
    except Exception as e:
        logger.debug(f"Error in breakout_squeeze_pro: {e}")
        
    return None

def analyze_rsi_divergence(df, params, rvol, adx_value):
    """استراتيجية انحراف مؤشر القوة النسبية"""
    if not SCIPY_AVAILABLE: 
        return None
        
    try:
        df.ta.rsi(length=params['rsi_period'], append=True)
        rsi_col = find_col(df.columns, f"RSI_{params['rsi_period']}")
        
        if not rsi_col or df[rsi_col].isnull().all(): 
            return None
            
        # أخذ عينة من البيانات للتحليل
        lookback = min(params['lookback_period'], len(df) - 1)
        subset = df.iloc[-lookback:].copy()
        
        if len(subset) < params['peak_trough_lookback'] * 2:
            return None
            
        # البحث عن القيعان في السعر والـ RSI
        distance = params['peak_trough_lookback']
        price_troughs_idx, _ = find_peaks(-subset['low'].values, distance=distance)
        rsi_troughs_idx, _ = find_peaks(-subset[rsi_col].values, distance=distance)
        
        if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
            # أخذ آخر قاعين
            p_low1_idx, p_low2_idx = price_troughs_idx[-2], price_troughs_idx[-1]
            r_low1_idx, r_low2_idx = rsi_troughs_idx[-2], rsi_troughs_idx[-1]
            
            # فحص الانحراف (السعر ينخفض والـ RSI يرتفع)
            price_makes_lower_low = subset.iloc[p_low2_idx]['low'] < subset.iloc[p_low1_idx]['low']
            rsi_makes_higher_low = subset.iloc[r_low2_idx][rsi_col] > subset.iloc[r_low1_idx][rsi_col]
            
            is_divergence = price_makes_lower_low and rsi_makes_higher_low
            
            if is_divergence:
                # تأكيدات إضافية
                rsi_exits_oversold = True
                if params['confirm_with_rsi_exit']:
                    rsi_exits_oversold = (subset.iloc[r_low1_idx][rsi_col] < 35 and 
                                        subset.iloc[-2][rsi_col] > 40)
                
                # تأكيد السعر
                confirmation_price = subset.iloc[p_low2_idx:]['high'].max()
                price_confirmed = df.iloc[-2]['close'] > confirmation_price
                
                if rsi_exits_oversold and price_confirmed:
                    return {"reason": "rsi_divergence", "type": "long"}
                    
    except Exception as e:
        logger.debug(f"Error in rsi_divergence: {e}")
        
    return None

def analyze_supertrend_pullback(df, params, rvol, adx_value):
    """استراتيجية انعكاس السوبرترند"""
    try:
        df.ta.supertrend(length=params['atr_period'], multiplier=params['atr_multiplier'], append=True)
        st_dir_col = find_col(df.columns, f"SUPERTd_{params['atr_period']}_")
        
        # إضافة EMA للتأكيد
        ema_period = bot_data['settings']['ema_trend_filter']['ema_period']
        df.ta.ema(length=ema_period, append=True)
        ema_col = find_col(df.columns, f'EMA_{ema_period}')
        
        if not st_dir_col or not ema_col or pd.isna(df[ema_col].iloc[-2]): 
            return None
            
        last, prev = df.iloc[-2], df.iloc[-3]
        
        # إشارة انعكاس السوبرترند (من bearish إلى bullish)
        supertrend_bullish_flip = (prev[st_dir_col] == -1 and last[st_dir_col] == 1)
        
        if supertrend_bullish_flip:
            settings = bot_data['settings']
            
            # الشروط الإضافية
            price_above_ema = last['close'] > last[ema_col]
            strong_trend = adx_value >= settings['master_adx_filter_level']
            good_volume = rvol >= settings['liquidity_filters']['min_rvol']
            
            # كسر أعلى سعر حديث (breakout)
            swing_lookback = params.get('swing_high_lookback', 10)
            if len(df) >= swing_lookback + 2:
                recent_swing_high = df['high'].iloc[-(swing_lookback+2):-2].max()
                breakout_confirmed = last['close'] > recent_swing_high
            else:
                breakout_confirmed = True  # إذا لم تكن البيانات كافية
            
            if price_above_ema and strong_trend and good_volume and breakout_confirmed:
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
            'timeout': 30000,  # 30 ثانية timeout
        }
        
        # إضافة مفاتيح API حسب المنصة
        if ex_id == 'binance' and BINANCE_API_KEY != 'YOUR_BINANCE_API_KEY':
            logger.info("🔑 Initializing Binance with API credentials for REAL TRADING")
            params['apiKey'] = BINANCE_API_KEY
            params['secret'] = BINANCE_API_SECRET
            params['sandbox'] = False  # تأكد من أنه ليس sandbox
            
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
            # لا نغلق الاتصال هنا لأنه قد لا يكون مفتوحاً
    
    # الاتصال بجميع المنصات
    await asyncio.gather(*[connect(ex_id) for ex_id in EXCHANGES_TO_SCAN], return_exceptions=True)
    
    connected_count = len(bot_data["exchanges"])
    logger.info(f"🌐 Exchange initialization complete: {connected_count}/{len(EXCHANGES_TO_SCAN)} connected")

async def aggregate_top_movers():
    """جمع أفضل العملات من جميع المنصات المتصلة"""
    all_tickers = []
    
    async def fetch_tickers(ex_id, ex):
        try: 
            tickers = await ex.fetch_tickers()
            result = []
            for symbol, ticker in tickers.items():
                if ticker:  # تأكد من وجود بيانات
                    ticker_copy = dict(ticker)
                    ticker_copy['exchange'] = ex_id
                    result.append(ticker_copy)
            return result
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch tickers from {ex_id}: {e}")
            return []
    
    # جلب البيانات من جميع المنصات بالتوازي
    results = await asyncio.gather(
        *[fetch_tickers(ex_id, ex) for ex_id, ex in bot_data["exchanges"].items()],
        return_exceptions=True
    )
    
    # دمج النتائج
    for result in results:
        if isinstance(result, list):  # تجنب الأخطاء
            all_tickers.extend(result)
        
    if not all_tickers:
        logger.warning("⚠️ No market data received from any exchange")
        return []
        
    settings = bot_data['settings']
    excluded_bases = settings['stablecoin_filter']['exclude_bases']
    min_volume = settings['liquidity_filters']['min_quote_volume_24h_usd']
    
    # تصفية العملات
    filtered_tickers = []
    for ticker in all_tickers:
        symbol = ticker.get('symbol', '')
        quote_volume = ticker.get('quoteVolume', 0)
        percentage = ticker.get('percentage')
        
        if (symbol.upper().endswith('/USDT') and
            symbol.split('/')[0] not in excluded_bases and
            quote_volume >= min_volume and
            percentage is not None and
            not any(keyword in symbol.upper() for keyword in ['UP','DOWN','3L','3S','BEAR','BULL','LEVERAGED'])):
            filtered_tickers.append(ticker)
    
    # ترتيب حسب الحجم
    sorted_tickers = sorted(filtered_tickers, key=lambda t: t.get('quoteVolume', 0), reverse=True)
    
    # إزالة التكرارات (نفس العملة من منصات مختلفة)
    unique_symbols = {}
    for ticker in sorted_tickers:
        symbol = ticker['symbol']
        if symbol not in unique_symbols:
            unique_symbols[symbol] = {
                'exchange': ticker['exchange'], 
                'symbol': symbol,
                'volume': ticker.get('quoteVolume', 0)
            }
    
    # اختيار أفضل العملات
    final_list = list(unique_symbols.values())[:settings['top_n_symbols_by_volume']]
    
    logger.info(f"📊 Market aggregation: {len(all_tickers)} total → {len(filtered_tickers)} filtered → {len(final_list)} selected")
    bot_data['status_snapshot']['markets_found'] = len(final_list)
    
    return final_list

async def get_higher_timeframe_trend(exchange, symbol, ma_period):
    """فحص الاتجاه في الإطار الزمني الأعلى"""
    try:
        ohlcv_htf = await exchange.fetch_ohlcv(symbol, HIGHER_TIMEFRAME, limit=ma_period + 10)
        if len(ohlcv_htf) < ma_period: 
            return None, f"Not enough data: {len(ohlcv_htf)} < {ma_period}"
            
        df_htf = pd.DataFrame(ohlcv_htf, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        sma_values = ta.sma(df_htf['close'], length=ma_period)
        
        if pd.isna(sma_values.iloc[-1]):
            return None, "SMA calculation failed"
            
        last_close = df_htf.iloc[-1]['close']
        last_sma = sma_values.iloc[-1]
        is_bullish = last_close > last_sma
        
        return is_bullish, "Bullish" if is_bullish else "Bearish"
        
    except Exception as e:
        return None, f"Error: {str(e)[:50]}"

async def worker(queue, results_list, settings, failure_counter):
    """عامل معالجة لتحليل العملات"""
    
    while not queue.empty():
        try:
            market_info = await queue.get()
            symbol = market_info.get('symbol', 'N/A')
            exchange_id = market_info.get('exchange', 'unknown')
            exchange = bot_data["exchanges"].get(exchange_id)
            
            if not exchange or not settings.get('active_scanners'):
                continue
                
            # جلب الفلاتر
            liq_filters = settings['liquidity_filters']
            vol_filters = settings['volatility_filters']
            ema_filters = settings['ema_trend_filter']

            # 1. فحص orderbook والانتشار
            try:
                orderbook = await exchange.fetch_order_book(symbol, limit=10)
                if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
                    continue

                best_bid, best_ask = orderbook['bids'][0][0], orderbook['asks'][0][0]
                if best_bid <= 0 or best_ask <= 0: 
                    continue

                spread_percent = ((best_ask - best_bid) / best_bid) * 100
                if spread_percent > liq_filters['max_spread_percent']:
                    continue
                    
            except Exception:
                continue

            # 2. جلب البيانات التاريخية
            try:
                required_candles = max(ema_filters['ema_period'], 200) + 20
                ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=required_candles)
                
                if len(ohlcv) < ema_filters['ema_period']:
                    continue

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
                
            except Exception:
                continue

            # 3. فحص RVOL
            try:
                volume_sma = ta.sma(df['volume'], length=liq_filters['rvol_period'])
                if pd.isna(volume_sma.iloc[-2]) or volume_sma.iloc[-2] <= 0:
                    continue

                current_volume = df['volume'].iloc[-2]
                rvol = current_volume / volume_sma.iloc[-2]
                
                if rvol < liq_filters['min_rvol']:
                    continue
                    
            except Exception:
                continue

            # 4. فحص التقلبات (ATR)
            try:
                atr_values = ta.atr(df['high'], df['low'], df['close'], length=vol_filters['atr_period_for_filter'])
                last_close = df['close'].iloc[-2]
                last_atr = atr_values.iloc[-2]
                
                if pd.isna(last_atr) or last_close <= 0:
                    continue
                    
                atr_percent = (last_atr / last_close) * 100
                if atr_percent < vol_filters['min_atr_percent']:
                    continue
                    
            except Exception:
                continue

            # 5. فحص EMA Trend Filter
            try:
                if ema_filters['enabled']:
                    ema_values = ta.ema(df['close'], length=ema_filters['ema_period'])
                    if pd.isna(ema_values.iloc[-2]):
                        continue
                        
                    if last_close < ema_values.iloc[-2]:
                        continue
                        
            except Exception:
                continue

            # 6. فحص Higher Timeframe Trend
            if settings.get('use_master_trend_filter'):
                try:
                    is_htf_bullish, htf_reason = await get_higher_timeframe_trend(
                        exchange, symbol, settings['master_trend_filter_ma_period']
                    )
                    if not is_htf_bullish:
                        continue
                except Exception:
                    continue

            # 7. حساب ADX
            try:
                adx_values = ta.adx(df['high'], df['low'], df['close'])
                adx_col = find_col(adx_values.columns if hasattr(adx_values, 'columns') else [], 'ADX')
                
                if adx_col and not pd.isna(adx_values[adx_col].iloc[-2]):
                    adx_value = adx_values[adx_col].iloc[-2]
                else:
                    adx_value = 20  # قيمة افتراضية
                
                if settings.get('use_master_trend_filter') and adx_value < settings['master_adx_filter_level']:
                    continue
                    
            except Exception:
                adx_value = 20
                if settings.get('use_master_trend_filter') and adx_value < settings['master_adx_filter_level']:
                    continue

            # 8. تشغيل الاستراتيجيات
            confirmed_signals = []
            for scanner_name in settings['active_scanners']:
                try:
                    if scanner_name in SCANNERS:
                        scanner_params = settings.get(scanner_name, {})
                        result = SCANNERS[scanner_name](df.copy(), scanner_params, rvol, adx_value)
                        
                        if result and result.get("type") == "long":
                            confirmed_signals.append(result['reason'])
                except Exception as e:
                    logger.debug(f"Scanner {scanner_name} failed for {symbol}: {e}")

            # 9. التحقق من قوة الإشارة
            if len(confirmed_signals) >= settings.get("min_signal_strength", 2):
                try:
                    reason_str = ' + '.join(confirmed_signals)
                    entry_price = df.iloc[-2]['close']
                    
                    # حساب TP/SL
                    if settings.get("use_dynamic_risk_management", True):
                        try:
                            atr_values = ta.atr(df['high'], df['low'], df['close'], length=settings['atr_period'])
                            current_atr = atr_values.iloc[-2]
                            
                            if pd.notna(current_atr) and current_atr > 0:
                                risk_per_unit = current_atr * settings['atr_sl_multiplier']
                                stop_loss = entry_price - risk_per_unit
                                take_profit = entry_price + (risk_per_unit * settings['risk_reward_ratio'])
                            else:
                                raise ValueError("Invalid ATR")
                        except:
                            # Fallback إلى النسب الثابتة
                            stop_loss = entry_price * 0.98  # 2%
                            take_profit = entry_price * 1.04  # 4%
                    else:
                        stop_loss = entry_price * 0.98
                        take_profit = entry_price * 1.04
                    
                    tp_percent = ((take_profit - entry_price) / entry_price * 100)
                    sl_percent = ((entry_price - stop_loss) / entry_price * 100)
                    
                    # فحص الحدود الدنيا
                    min_tp = settings['min_tp_sl_filter']['min_tp_percent']
                    min_sl = settings['min_tp_sl_filter']['min_sl_percent']
                    
                    if tp_percent >= min_tp and sl_percent >= min_sl:
                        signal = {
                            "symbol": symbol,
                            "exchange": exchange_id.capitalize(),
                            "entry_price": entry_price,
                            "take_profit": take_profit,
                            "stop_loss": stop_loss,
                            "timestamp": df.index[-2],
                            "reason": reason_str,
                            "strength": len(confirmed_signals),
                            "tp_percent": tp_percent,
                            "sl_percent": sl_percent,
                            "rvol": rvol,
                            "adx": adx_value
                        }
                        
                        results_list.append(signal)
                        logger.info(f"✅ SIGNAL: {symbol} | Strength: {len(confirmed_signals)} | TP: +{tp_percent:.1f}% | SL: -{sl_percent:.1f}%")
                        
                except Exception as e:
                    logger.error(f"Error processing signal for {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"💥 Worker error: {e}")
            failure_counter[0] += 1
            
        finally:
            queue.task_done()

# --- Real Trading Functions ---
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
    """تنفيذ صفقة حقيقية على المنصة"""
    
    exchange_id = signal['exchange'].lower()
    logger.info(f"🚨 EXECUTING REAL TRADE: {signal['symbol']} on {exchange_id.upper()}")
    
    exchange = bot_data["exchanges"].get(exchange_id)
    if not exchange or not hasattr(exchange, 'apiKey') or not exchange.apiKey:
        error_msg = f"❌ No API credentials for {exchange_id.upper()}"
        logger.error(error_msg)
        await send_telegram_message(context.bot, {
            'custom_message': f"**خطأ في التنفيذ**\n\n{error_msg}\nالعملة: {signal['symbol']}"
        })
        return None

    try:
        # 1. فحص الرصيد
        usdt_balance = await get_real_balance(exchange_id, 'USDT')
        if usdt_balance <= 0:
            await send_telegram_message(context.bot, {
                'custom_message': f"**❌ رصيد غير كافٍ**\n\nالمنصة: {exchange_id.upper()}\nالرصيد: ${usdt_balance:.2f}"
            })
            return None
        
        # 2. حساب حجم الصفقة
        trade_percentage = bot_data['settings']['real_trade_size_percentage']
        trade_amount_usdt = usdt_balance * (trade_percentage / 100)
        min_trade = 15.0  # حد أدنى
        
        if trade_amount_usdt < min_trade:
            await send_telegram_message(context.bot, {
                'custom_message': f"**⚠️ حجم صفقة صغير**\n\nالمطلوب: ${trade_amount_usdt:.2f}\nالحد الأدنى: ${min_trade}\nالرصيد: ${usdt_balance:.2f}"
            })
            return None

        # 3. معلومات السوق
        markets = await exchange.load_markets()
        market = markets.get(signal['symbol'])
        if not market:
            logger.error(f"❌ Market {signal['symbol']} not found")
            return None
        
        # 4. حساب الكمية
        quantity = trade_amount_usdt / signal['entry_price']
        formatted_quantity = exchange.amount_to_precision(signal['symbol'], quantity)
        
        if float(formatted_quantity) <= 0:
            logger.error(f"❌ Invalid quantity: {formatted_quantity}")
            return None

        # 5. تنفيذ أمر الشراء
        logger.info(f"🔄 MARKET BUY: {formatted_quantity} {signal['symbol']} (~${trade_amount_usdt:.2f})")
        
        buy_order = await exchange.create_market_buy_order(
            signal['symbol'], 
            float(formatted_quantity)
        )
        
        logger.info(f"✅ BUY ORDER EXECUTED: {buy_order['id']}")
        
        # انتظار قصير
        await asyncio.sleep(3)

        # 6. أوامر الخروج
        tp_price = exchange.price_to_precision(signal['symbol'], signal['take_profit'])
        sl_price = exchange.price_to_precision(signal['symbol'], signal['stop_loss'])
        exit_orders = {}

        # أوامر الخروج حسب المنصة
        try:
            if exchange_id == 'binance':
                # OCO order للبينانس
                sl_trigger = exchange.price_to_precision(signal['symbol'], signal['stop_loss'] * 1.001)
                
                oco_order = await exchange.create_order(
                    signal['symbol'], 'oco', 'sell', float(formatted_quantity),
                    price=tp_price, stopPrice=sl_trigger,
                    params={'stopLimitPrice': sl_price}
                )
                
                exit_orders = {"oco_id": oco_order['id']}
                logger.info(f"✅ BINANCE OCO placed: {oco_order['id']}")
                
            elif exchange_id == 'kucoin':
                # أوامر منفصلة للكوكوين
                tp_order = await exchange.create_limit_sell_order(
                    signal['symbol'], float(formatted_quantity), float(tp_price)
                )
                
                sl_trigger = exchange.price_to_precision(signal['symbol'], signal['stop_loss'] * 1.002)
                sl_order = await exchange.create_order(
                    signal['symbol'], 'stop_limit', 'sell', float(formatted_quantity),
                    float(sl_price), params={'stopPrice': float(sl_trigger)}
                )
                
                exit_orders = {"tp_id": tp_order['id'], "sl_id": sl_order['id']}
                logger.info(f"✅ KUCOIN TP/SL placed: {tp_order['id']}, {sl_order['id']}")
                
            else:
                logger.warning(f"⚠️ Exit orders not implemented for {exchange_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to place exit orders: {e}")
            # يمكن المتابعة بدون أوامر الخروج

        # 7. تأكيد التنفيذ
        actual_cost = float(buy_order.get('cost', trade_amount_usdt))
        
        success_msg = (
            f"**🚨 صفقة حقيقية تم تنفيذها! 🚨**\n\n"
            f"📊 **تفاصيل الصفقة:**\n"
            f"• المنصة: {exchange_id.upper()}\n"
            f"• العملة: `{signal['symbol']}`\n"
            f"• الكمية: `{formatted_quantity}`\n"
            f"• التكلفة: `${actual_cost:.2f}`\n"
            f"• سعر الدخول: `{signal['entry_price']:.6f}`\n\n"
            f"🎯 **أهداف الخروج:**\n"
            f"• الهدف: `{tp_price}` (+{signal.get('tp_percent', 0):.1f}%)\n"
            f"• الوقف: `{sl_price}` (-{signal.get('sl_percent', 0):.1f}%)\n\n"
            f"📋 **معرف الأمر:** `{buy_order['id']}`\n\n"
            f"**🔐 أوامر الخروج تعمل تلقائياً**"
        )
        
        await send_telegram_message(context.bot, {'custom_message': success_msg})

        return {
            "entry_order_id": buy_order['id'],
            "exit_order_ids_json": json.dumps(exit_orders),
            "quantity": float(formatted_quantity),
            "entry_value_usdt": actual_cost
        }

    except ccxt.InsufficientFunds as e:
        error_msg = f"💸 رصيد غير كافٍ على {exchange_id.upper()}"
        logger.error(f"{error_msg}: {e}")
        await send_telegram_message(context.bot, {'custom_message': f"**❌ {error_msg}**"})
        
    except ccxt.ExchangeError as e:
        error_msg = f"🏛️ خطأ من منصة {exchange_id.upper()}"
        logger.error(f"{error_msg}: {e}")
        await send_telegram_message(context.bot, {'custom_message': f"**❌ {error_msg}**\n`{str(e)[:100]}...`"})
        
    except Exception as e:
        error_msg = f"💥 خطأ حرج في تنفيذ الصفقة"
        logger.error(f"{error_msg}: {e}", exc_info=True)
        await send_telegram_message(context.bot, {'custom_message': f"**{error_msg}**\n{signal['symbol']} على {exchange_id.upper()}"})
    
    return None

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    """الفحص الرئيسي للسوق وتنفيذ الصفقات"""
    
    # حماية من التنفيذ المتوازي
    async with scan_lock:
        if bot_data['status_snapshot']['scan_in_progress']:
            logger.warning("⚠️ Scan already in progress, skipping...")
            return
            
        settings = bot_data["settings"]
        
        # إعداد حالة الفحص
        status = bot_data['status_snapshot']
        status.update({
            "scan_in_progress": True,
            "last_scan_start_time": datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "signals_found": 0
        })
        
        logger.info("🚀 ========== STARTING REAL TRADING SCAN ==========")
        
        try:
            # 1. فحص التحليل الأساسي
            if settings.get('fundamental_analysis_enabled', True):
                mood, mood_score, mood_reason = await get_fundamental_market_mood()
                bot_data['settings']['last_market_mood'] = {
                    "timestamp": datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M'),
                    "mood": mood,
                    "reason": mood_reason
                }
                save_settings()
                
                logger.info(f"📰 Fundamental Analysis: {mood} - {mood_reason}")
                
                if mood in ["NEGATIVE", "DANGEROUS"]:
                    await send_telegram_message(context.bot, {
                        'custom_message': f"**⚠️ إيقاف الفحص - تحليل أساسي سلبي**\n\n**الحالة:** {mood}\n**السبب:** {mood_reason}\n\n*سيتم المحاولة في الفحص التالي*"
                    })
                    status['scan_in_progress'] = False
                    return
            
            # 2. فحص وضع السوق
            is_market_ok, btc_reason = await check_market_regime()
            status['btc_market_mood'] = "إيجابي ✅" if is_market_ok else "سلبي ❌"
            
            if settings.get('market_regime_filter_enabled', True) and not is_market_ok:
                logger.info(f"🔒 Market regime blocked: {btc_reason}")
                await send_telegram_message(context.bot, {
                    'custom_message': f"**🔒 إيقاف الفحص - وضع السوق سلبي**\n\n**السبب:** {btc_reason}\n\n*سيتم المحاولة في الفحص التالي*"
                })
                status['scan_in_progress'] = False
                return
            
            # 3. فحص الصفقات النشطة
            try:
                conn = sqlite3.connect(DB_FILE, timeout=10)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة'")
                active_trades_count = cursor.fetchone()[0]
                conn.close()
                logger.info(f"📊 Active trades: {active_trades_count}/{settings.get('max_concurrent_trades', 3)}")
            except Exception as e:
                logger.error(f"💥 DB Error: {e}")
                active_trades_count = 0
            
            # 4. جلب أفضل الأسواق
            top_markets = await aggregate_top_movers()
            if not top_markets:
                logger.warning("⚠️ No markets found for scanning")
                status['scan_in_progress'] = False
                return
            
            # 5. إعداد المعالجة المتوازية
            queue = asyncio.Queue()
            for market in top_markets:
                await queue.put(market)
            
            signals = []
            failure_counter = [0]
            
            # 6. تشغيل العمال
            logger.info(f"🔄 Starting {settings['concurrent_workers']} workers to analyze {len(top_markets)} markets...")
            
            workers = [
                asyncio.create_task(worker(queue, signals, settings, failure_counter))
                for _ in range(settings['concurrent_workers'])
            ]
            
            # انتظار انتهاء جميع المهام
            await queue.join()
            
            # إلغاء العمال
            for w in workers:
                w.cancel()
            
            await asyncio.gather(*workers, return_exceptions=True)
            
            # 7. معالجة النتائج
            signals.sort(key=lambda s: s.get('strength', 0), reverse=True)
            
            new_trades = 0
            opportunities = 0
            last_signal_time = bot_data['last_signal_time']
            cooldown_seconds = SCAN_INTERVAL_SECONDS * 3  # 45 دقيقة cooldown
            
            logger.info(f"🔍 Analysis complete: {len(signals)} signals found")
            
            # 8. تنفيذ الصفقات
            for signal in signals:
                try:
                    symbol = signal['symbol']
                    
                    # فحص cooldown
                    if time.time() - last_signal_time.get(symbol, 0) <= cooldown_seconds:
                        logger.debug(f"⏰ {symbol}: Still in cooldown")
                        continue
                    
                    # تحديد نوع التداول
                    can_trade_real = (settings.get('real_trading_enabled', True) and
                                    signal['exchange'].lower() in ['binance', 'kucoin', 'okx'] and
                                    active_trades_count < settings.get("max_concurrent_trades", 3))
                    
                    if can_trade_real:
                        # محاولة التداول الحقيقي
                        logger.info(f"🎯 Attempting REAL TRADE: {symbol}")
                        order_result = await place_real_trade(signal, context)
                        
                        if order_result:
                            # نجح التداول الحقيقي
                            signal.update({
                                'is_real_trade': True,
                                'entry_order_id': order_result['entry_order_id'],
                                'exit_order_ids_json': order_result['exit_order_ids_json'],
                                'quantity': order_result['quantity'],
                                'entry_value_usdt': order_result['entry_value_usdt']
                            })
                            
                            if trade_id := log_recommendation_to_db(signal):
                                signal['trade_id'] = trade_id
                                await send_telegram_message(context.bot, signal, is_new=True)
                                active_trades_count += 1
                                new_trades += 1
                                logger.info(f"✅ REAL TRADE EXECUTED: {symbol}")
                        else:
                            # فشل التداول الحقيقي
                            signal['is_real_trade'] = False
                            await send_telegram_message(context.bot, signal, is_opportunity=True)
                            opportunities += 1
                            logger.warning(f"❌ Real trade failed for {symbol}, logged as opportunity")
                    
                    elif active_trades_count >= settings.get("max_concurrent_trades", 3):
                        # وصل للحد الأقصى
                        signal['is_real_trade'] = False
                        await send_telegram_message(context.bot, signal, is_opportunity=True)
                        opportunities += 1
                        logger.info(f"📊 Max trades reached, {symbol} as opportunity")
                    
                    else:
                        # التداول الحقيقي غير مفعل أو المنصة غير مدعومة
                        signal['is_real_trade'] = False
                        
                        # تسجيل كتداول افتراضي
                        trade_amount = 1000 * (settings.get('real_trade_size_percentage', 2.0) / 100)
                        signal.update({
                            'quantity': trade_amount / signal['entry_price'],
                            'entry_value_usdt': trade_amount
                        })
                        
                        if trade_id := log_recommendation_to_db(signal):
                            signal['trade_id'] = trade_id
                            await send_telegram_message(context.bot, signal, is_new=True)
                            new_trades += 1
                            logger.info(f"📝 Virtual trade logged: {symbol}")
                    
                    # تحديث الوقت
                    last_signal_time[symbol] = time.time()
                    await asyncio.sleep(1)  # تجنب spam
                    
                except Exception as e:
                    logger.error(f"💥 Error processing signal {signal.get('symbol', 'UNKNOWN')}: {e}")
            
            # 9. إرسال الملخص
            failures = failure_counter[0]
            scan_end_time = datetime.now(EGYPT_TZ)
            scan_start_time = datetime.strptime(status['last_scan_start_time'], '%Y-%m-%d %H:%M:%S')
            scan_duration = (scan_end_time - scan_start_time).total_seconds()
            
            trading_mode = "🚨 التداول الحقيقي" if settings.get('real_trading_enabled', True) else "📊 التداول الافتراضي"
            
            summary = (
                f"**🔬 ملخص فحص السوق**\n\n"
                f"**⚙️ الوضع:** {trading_mode}\n"
                f"**🕐 المدة:** {scan_duration:.0f} ثانية\n"
                f"**📊 الأسواق:** {len(top_markets)}\n"
                f"**🎯 حالة السوق:** {status['btc_market_mood']}\n\n"
                f"**📈 النتائج:**\n"
                f"• **إشارات مكتشفة:** {len(signals)}\n"
                f"• **✅ صفقات جديدة:** {new_trades}\n"
                f"• **💡 فرص للمراقبة:** {opportunities}\n"
                f"• **⚠️ أخطاء:** {failures}\n\n"
                f"**📊 الصفقات النشطة:** {active_trades_count}/{settings.get('max_concurrent_trades', 3)}\n\n"
                f"*الفحص التالي خلال {SCAN_INTERVAL_SECONDS//60} دقيقة*"
            )
            
            await send_telegram_message(context.bot, {'custom_message': summary})
            
            # 10. تحديث الإحصائيات
            status.update({
                'signals_found': new_trades + opportunities,
                'last_scan_end_time': scan_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'active_trades_count': active_trades_count,
                'scan_in_progress': False
            })
            
            bot_data['scan_history'].append({
                'timestamp': status['last_scan_end_time'],
                'signals': len(signals),
                'trades': new_trades,
                'opportunities': opportunities,
                'failures': failures,
                'duration': scan_duration
            })
            
            logger.info(f"🏁 SCAN COMPLETE: {len(signals)} signals, {new_trades} trades, {opportunities} opportunities in {scan_duration:.0f}s")
            
        except Exception as e:
            logger.error(f"💥 CRITICAL ERROR in perform_scan: {e}", exc_info=True)
            
            status['scan_in_progress'] = False
            await send_telegram_message(context.bot, {
                'custom_message': f"**💥 خطأ حرج في فحص السوق**\n\n**الخطأ:** {str(e)[:200]}...\n\n**سيتم إعادة المحاولة في الفحص التالي**"
            })

async def track_open_trades(context: ContextTypes.DEFAULT_TYPE):
    """تتبع ومراقبة الصفقات المفتوحة"""
    
    try:
        # جلب الصفقات النشطة
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'نشطة'")
        active_trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if not active_trades:
            bot_data['status_snapshot']['active_trades_count'] = 0
            return
        
        logger.info(f"👀 Tracking {len(active_trades)} active trades")
        bot_data['status_snapshot']['active_trades_count'] = len(active_trades)
        
    except Exception as e:
        logger.error(f"💥 Database error in trade tracking: {e}")
        return

    async def check_single_trade(trade):
        """فحص صفقة واحدة"""
        try:
            exchange = bot_data["exchanges"].get(trade['exchange'].lower())
            if not exchange:
                return None

            # جلب السعر الحالي
            ticker = await exchange.fetch_ticker(trade['symbol'])
            current_price = ticker.get('last') or ticker.get('close')
            
            if not current_price or current_price <= 0:
                return None

            # تحديث أعلى سعر
            highest_price = max(trade.get('highest_price', current_price), current_price)
            
            # فحص شروط الخروج الأساسية
            if current_price >= trade['take_profit']:
                return {
                    'trade_id': trade['id'],
                    'action': 'close',
                    'status': 'ناجحة',
                    'exit_price': current_price,
                    'highest_price': highest_price,
                    'reason': 'وصل للهدف'
                }
            elif current_price <= trade['stop_loss']:
                return {
                    'trade_id': trade['id'],
                    'action': 'close',
                    'status': 'فاشلة',
                    'exit_price': current_price,
                    'highest_price': highest_price,
                    'reason': 'وصل لوقف الخسارة'
                }

            # إدارة الوقف المتحرك
            settings = bot_data["settings"]
            if settings.get('trailing_sl_enabled', True):
                
                # تفعيل الوقف المتحرك
                if not trade.get('trailing_sl_active'):
                    activation_percent = settings.get('trailing_sl_activate_percent', 1.5)
                    activation_price = trade['entry_price'] * (1 + activation_percent / 100)
                    
                    if current_price >= activation_price:
                        new_sl = trade['entry_price']  # نقل لنقطة التعادل
                        
                        return {
                            'trade_id': trade['id'],
                            'action': 'activate_trailing',
                            'new_sl': new_sl,
                            'highest_price': highest_price
                        }
                
                # تحديث الوقف المتحرك
                elif trade.get('trailing_sl_active'):
                    trailing_percent = settings.get('trailing_sl_percent', 1.0)
                    new_sl = highest_price * (1 - trailing_percent / 100)
                    
                    if new_sl > trade['stop_loss']:
                        return {
                            'trade_id': trade['id'],
                            'action': 'update_trailing',
                            'new_sl': new_sl,
                            'highest_price': highest_price
                        }

            # تحديث أعلى سعر فقط
            if highest_price > trade.get('highest_price', 0):
                return {
                    'trade_id': trade['id'],
                    'action': 'update_highest',
                    'highest_price': highest_price
                }

            return None

        except Exception as e:
            logger.error(f"💥 Error checking trade {trade['id']}: {e}")
            return None

    # فحص جميع الصفقات
    updates = []
    for trade in active_trades:
        try:
            update = await check_single_trade(trade)
            if update:
                updates.append(update)
            await asyncio.sleep(0.5)  # تجنب rate limiting
        except Exception as e:
            logger.error(f"💥 Error in trade check loop: {e}")

    # تطبيق التحديثات
    for update in updates:
        try:
            trade_id = update['trade_id']
            action = update['action']
            
            if action == 'close':
                # إغلاق الصفقة
                trade = next((t for t in active_trades if t['id'] == trade_id), None)
                if not trade:
                    continue
                
                # حساب P&L
                pnl_usdt = (update['exit_price'] - trade['entry_price']) * trade['quantity']
                exit_value_usdt = update['exit_price'] * trade['quantity']
                
                # تحديث قاعدة البيانات
                db_updates = {
                    'status': update['status'],
                    'exit_price': update['exit_price'],
                    'closed_at': datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_value_usdt': exit_value_usdt,
                    'pnl_usdt': pnl_usdt,
                    'highest_price': update['highest_price']
                }
                
                if update_trade_in_db(trade_id, db_updates):
                    # إرسال إشعار الإغلاق
                    profit_emoji = "🎉" if pnl_usdt > 0 else "😞"
                    profit_text = f"+${pnl_usdt:.2f}" if pnl_usdt > 0 else f"${pnl_usdt:.2f}"
                    trade_type = "🚨 حقيقية" if trade.get('is_real_trade') else "📊 افتراضية"
                    
                    close_message = (
                        f"**{profit_emoji} صفقة مغلقة #{trade_id}**\n\n"
                        f"**📊 النوع:** {trade_type}\n"
                        f"**💰 العملة:** {trade['symbol']}\n"
                        f"**📈 الدخول:** {trade['entry_price']:.6f}\n"
                        f"**📉 الخروج:** {update['exit_price']:.6f}\n"
                        f"**💵 النتيجة:** {profit_text}\n"
                        f"**📋 السبب:** {update['reason']}\n\n"
                        f"*تم الإغلاق تلقائياً*"
                    )
                    
                    await send_telegram_message(context.bot, {'custom_message': close_message})
                    logger.info(f"🔔 Trade #{trade_id} closed: {update['status']} with P&L: ${pnl_usdt:.2f}")
            
            elif action == 'activate_trailing':
                # تفعيل الوقف المتحرك
                db_updates = {
                    'trailing_sl_active': True,
                    'stop_loss': update['new_sl'],
                    'highest_price': update['highest_price']
                }
                
                if update_trade_in_db(trade_id, db_updates):
                    trade = next((t for t in active_trades if t['id'] == trade_id), None)
                    
                    activation_message = (
                        f"**🚀 تأمين الأرباح - الصفقة #{trade_id}**\n\n"
                        f"**💰 العملة:** {trade['symbol'] if trade else 'N/A'}\n"
                        f"**🔒 تم تفعيل الوقف المتحرك**\n"
                        f"**📈 الوقف الجديد:** {update['new_sl']:.6f}\n\n"
                        f"**هذه الصفقة الآن مؤمَّنة بالكامل! 🛡️**"
                    )
                    
                    await send_telegram_message(context.bot, {'custom_message': activation_message})
                    logger.info(f"🚀 Trailing SL activated for trade #{trade_id}")
            
            elif action == 'update_trailing':
                # تحديث الوقف المتحرك
                db_updates = {
                    'stop_loss': update['new_sl'],
                    'highest_price': update['highest_price']
                }
                
                update_trade_in_db(trade_id, db_updates)
                logger.info(f"📈 Trailing SL updated for trade #{trade_id}: {update['new_sl']:.6f}")
            
            elif action == 'update_highest':
                # تحديث أعلى سعر فقط
                update_trade_in_db(trade_id, {'highest_price': update['highest_price']})
                logger.debug(f"📊 Highest price updated for trade #{trade_id}")
            
        except Exception as e:
            logger.error(f"💥 Failed to process trade update {update.get('trade_id', 'UNKNOWN')}: {e}")

async def send_telegram_message(bot, signal_data, is_new=False, is_opportunity=False, update_type=None):
    """إرسال رسائل Telegram مُحسَّنة للتداول الحقيقي"""
    
    message = ""
    keyboard = None
    target_chat = TELEGRAM_CHAT_ID
    
    def format_price(price):
        if price < 0.01:
            return f"{price:,.8f}"
        elif price < 1:
            return f"{price:,.6f}"
        else:
            return f"{price:,.4f}"
    
    # رسائل مخصصة
    if 'custom_message' in signal_data:
        message = signal_data['custom_message']
        target_chat = signal_data.get('target_chat', TELEGRAM_CHAT_ID)
        if 'keyboard' in signal_data:
            keyboard = signal_data['keyboard']
    
    # رسائل التوصيات
    elif is_new or is_opportunity:
        target_chat = TELEGRAM_SIGNAL_CHANNEL_ID
        strength_stars = '⭐' * signal_data.get('strength', 1)
        
        # تحديد نوع الصفقة والعنوان
        is_real = signal_data.get('is_real_trade', False)
        
        if is_new and is_real:
            title = f"**🚨 صفقة حقيقية تم تنفيذها! | {signal_data['symbol']}**"
            trade_emoji = "🚨"
            trade_type = "صفقة حقيقية"
        elif is_new:
            title = f"**📊 توصية افتراضية | {signal_data['symbol']}**"
            trade_emoji = "📊"
            trade_type = "صفقة افتراضية"
        else:
            title = f"**💡 فرصة للمراقبة | {signal_data['symbol']}**"
            trade_emoji = "💡"
            trade_type = "فرصة مراقبة"
        
        # البيانات الأساسية
        entry = signal_data['entry_price']
        tp = signal_data['take_profit']
        sl = signal_data['stop_loss']
        tp_percent = signal_data.get('tp_percent', ((tp - entry) / entry * 100))
        sl_percent = signal_data.get('sl_percent', ((entry - sl) / entry * 100))
        
        # البيانات الفنية
        rvol = signal_data.get('rvol', 0)
        adx = signal_data.get('adx', 0)
        
        # ترجمة الاستراتيجيات
        reasons_en = signal_data['reason'].split(' + ')
        reasons_ar = ' + '.join([STRATEGY_NAMES_AR.get(r, r) for r in reasons_en])
        
        # معلومات المتابعة
        follow_info = ""
        if is_new and 'trade_id' in signal_data:
            follow_info = f"\n📋 **للمتابعة:** `/check {signal_data['trade_id']}`"
        
        # معلومات الأمان
        safety_info = ""
        if is_real:
            safety_info = f"\n🔐 **الحماية:** أوامر الخروج تلقائية"
        
        message = (
            f"**{trade_emoji} إشارة تداول - {trade_type.upper()}**\n"
            f"{'═' * 40}\n"
            f"{title}\n"
            f"{'═' * 40}\n\n"
            f"🏛️ **المنصة:** {signal_data['exchange']}\n"
            f"⭐ **قوة الإشارة:** {strength_stars} ({signal_data.get('strength', 1)})\n"
            f"🔍 **الاستراتيجية:** {reasons_ar}\n"
            f"📊 **المؤشرات:** RVOL {rvol:.1f} | ADX {adx:.1f}\n\n"
            f"**📈 تفاصيل التداول:**\n"
            f"• **نقطة الدخول:** `{format_price(entry)}`\n"
            f"• **الهدف:** `{format_price(tp)}` **+{tp_percent:.1f}%** 🎯\n"
            f"• **وقف الخسارة:** `{format_price(sl)}` **-{sl_percent:.1f}%** 🛑\n"
            f"• **نسبة المخاطرة/العائد:** 1:{(tp_percent/sl_percent):.1f}\n"
            f"{safety_info}{follow_info}\n\n"
            f"*{'تم التنفيذ تلقائياً' if is_real else 'للمتابعة والتحليل'}*"
        )
    
    # رسائل التحديثات
    elif update_type == 'tsl_activation':
        message = (
            f"**🚀 تأمين الأرباح! | #{signal_data['id']} {signal_data['symbol']}**\n\n"
            f"تم تفعيل الوقف المتحرك وتأمين الصفقة.\n"
            f"**🔒 هذه الصفقة الآن محمية من الخسائر!**\n\n"
            f"*دع الأرباح تنمو! 📈*"
        )
    
    if not message:
        logger.warning("Empty message in send_telegram_message")
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
        
    except RetryAfter as e:
        logger.warning(f"⏰ Telegram rate limit, waiting {e.retry_after}s")
        await asyncio.sleep(e.retry_after)
        try:
            await bot.send_message(
                chat_id=target_chat,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
        except Exception as retry_e:
            logger.error(f"💥 Failed to send message after retry: {retry_e}")
            
    except Exception as e:
        logger.error(f"💥 Failed to send Telegram message: {e}")

# --- Telegram Bot Commands ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """أمر البدء مع معلومات التداول الحقيقي"""
    
    settings = bot_data['settings']
    trading_mode = "🚨 التداول الحقيقي مُفعَّل" if settings.get('real_trading_enabled', True) else "📊 التداول الافتراضي"
    trade_size = settings.get('real_trade_size_percentage', 2.0)
    max_trades = settings.get('max_concurrent_trades', 3)
    
    # فحص المنصات المتصلة
    authenticated_exchanges = []
    for ex_id, exchange in bot_data["exchanges"].items():
        if hasattr(exchange, 'apiKey') and exchange.apiKey:
            authenticated_exchanges.append(ex_id.upper())
    
    auth_info = f"**🔑 المنصات المُفعَّلة:** {', '.join(authenticated_exchanges) if authenticated_exchanges else 'لا توجد'}"
    
    welcome_message = (
        f"**🤖 مرحباً بك في بوت التداول الحقيقي**\n\n"
        f"**⚙️ الوضع الحالي:** {trading_mode}\n"
        f"**📊 حجم كل صفقة:** {trade_size}% من الرصيد\n"
        f"**🔢 أقصى عدد صفقات:** {max_trades}\n"
        f"{auth_info}\n\n"
        f"**📋 الأوامر المتاحة:**\n"
        f"• `/status` - حالة البوت والأسواق\n"
        f"• `/trades` - الصفقات النشطة\n"
        f"• `/balance` - عرض الأرصدة\n"
        f"• `/scan` - فحص فوري للسوق\n"
        f"• `/settings` - تعديل الإعدادات\n"
        f"• `/performance` - إحصائيات الأداء\n\n"
        f"**⚠️ تحذير هام:**\n"
        f"هذا البوت يتداول بأموال حقيقية! تأكد من صحة جميع الإعدادات.\n\n"
        f"**🔄 التشغيل التلقائي:**\n"
        f"• فحص السوق: كل {SCAN_INTERVAL_SECONDS//60} دقيقة\n"
        f"• مراقبة الصفقات: كل {TRACK_INTERVAL_SECONDS//60} دقيقة\n\n"
        f"*البوت يعمل الآن! 🚀*"
    )
    
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض حالة البوت التفصيلية"""
    
    settings = bot_data['settings']
    status = bot_data['status_snapshot']
    
    # معلومات التداول
    trading_mode = "🚨 التداول الحقيقي" if settings.get('real_trading_enabled', True) else "📊 التداول الافتراضي"
    scan_status = "🔄 يفحص الآن..." if status['scan_in_progress'] else "🟢 جاهز"
    
    # المنصات المتصلة
    exchange_status = []
    for ex_id, exchange in bot_data["exchanges"].items():
        if hasattr(exchange, 'apiKey') and exchange.apiKey:
            exchange_status.append(f"🔑 {ex_id.upper()}")
        else:
            exchange_status.append(f"📊 {ex_id.upper()}")
    
    # إحصائيات الفحص الأخير
    last_scan = status['last_scan_start_time']
    if last_scan == 'N/A':
        last_scan_info = "لم يتم الفحص بعد"
    else:
        last_scan_info = f"آخر فحص: {last_scan}"
    
    # متوسط وقت الفحص
    recent_scans = list(bot_data['scan_history'])[-5:] if bot_data['scan_history'] else []
    avg_duration = sum(s.get('duration', 0) for s in recent_scans) / len(recent_scans) if recent_scans else 0
    
    status_message = (
        f"**🤖 حالة بوت التداول الحقيقي**\n\n"
        f"**⚙️ وضع التشغيل:** {trading_mode}\n"
        f"**🔄 الحالة:** {scan_status}\n"
        f"**🏛️ المنصات:** {len(bot_data['exchanges'])}/6 متصلة\n"
        f"{'   • ' + chr(10) + '   • '.join(exchange_status)}\n\n"
        f"**📊 آخر نشاط:**\n"
        f"• {last_scan_info}\n"
        f"• **الأسواق المفحوصة:** {status['markets_found']}\n"
        f"• **الإشارات الجديدة:** {status['signals_found']}\n"
        f"• **حالة السوق العامة:** {status['btc_market_mood']}\n"
        f"• **متوسط وقت الفحص:** {avg_duration:.0f} ثانية\n\n"
        f"**📈 الصفقات:**\n"
        f"• **النشطة حالياً:** {status['active_trades_count']}/{settings.get('max_concurrent_trades', 3)}\n\n"
        f"**⚙️ الإعدادات الرئيسية:**\n"
        f"• **حجم الصفقة:** {settings.get('real_trade_size_percentage', 2.0)}%\n"
        f"• **قوة الإشارة المطلوبة:** {settings.get('min_signal_strength', 2)}\n"
        f"• **عدد العمال:** {settings.get('concurrent_workers', 8)}\n"
        f"• **فلتر السوق:** {'مُفعَّل' if settings.get('market_regime_filter_enabled') else 'مُعطَّل'}\n"
        f"• **تحليل الأخبار:** {'مُفعَّل' if settings.get('fundamental_analysis_enabled') else 'مُعطَّل'}\n\n"
        f"*الفحص التالي خلال {SCAN_INTERVAL_SECONDS//60} دقيقة*"
    )
    
    await update.message.reply_text(status_message, parse_mode=ParseMode.MARKDOWN)

async def trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض الصفقات النشطة"""
    
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM trades 
            WHERE status = 'نشطة' 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        active_trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if not active_trades:
            await update.message.reply_text(
                "**📊 الصفقات النشطة**\n\n"
                "لا توجد صفقات نشطة حالياً.\n\n"
                "*استخدم `/scan` لفحص السوق والعثور على فرص جديدة*",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        trades_text = "**📊 الصفقات النشطة**\n" + "═" * 30 + "\n\n"
        
        for i, trade in enumerate(active_trades, 1):
            trade_type = "🚨 حقيقية" if trade.get('is_real_trade') else "📊 افتراضية"
            entry_time = trade['timestamp'][:16] if trade['timestamp'] else "N/A"
            
            # حساب P&L الحالي (نحتاج السعر الحالي)
            current_pnl = "يتم الحساب..."
            
            trades_text += (
                f"**{i}. {trade['symbol']}** ({trade_type})\n"
                f"• **المنصة:** {trade['exchange']}\n"
                f"• **الدخول:** {trade['entry_price']:.6f} ({entry_time})\n"
                f"• **الهدف:** {trade['take_profit']:.6f}\n"
                f"• **الوقف:** {trade['stop_loss']:.6f}\n"
                f"• **الكمية:** {trade['quantity']:.4f}\n"
                f"• **الوقف المتحرك:** {'✅' if trade.get('trailing_sl_active') else '❌'}\n"
                f"• **ID:** #{trade['id']}\n\n"
            )
        
        trades_text += f"*إجمالي: {len(active_trades)} صفقة نشطة*"
        
        await update.message.reply_text(trades_text, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error(f"💥 Error in trades_command: {e}")
        await update.message.reply_text(
            "**❌ خطأ في جلب الصفقات**\n\n"
            "حدث خطأ أثناء جلب بيانات الصفقات من قاعدة البيانات.",
            parse_mode=ParseMode.MARKDOWN
        )

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض أرصدة المنصات المتصلة"""
    
    balances_text = "**💰 أرصدة المنصات**\n" + "═" * 25 + "\n\n"
    total_usdt = 0
    
    authenticated_exchanges = [
        (ex_id, ex) for ex_id, ex in bot_data["exchanges"].items()
        if hasattr(ex, 'apiKey') and ex.apiKey
    ]
    
    if not authenticated_exchanges:
        await update.message.reply_text(
            "**⚠️ لا توجد منصات مُفعَّلة**\n\n"
            "لم يتم العثور على أي منصة بمفاتيح API صحيحة.\n"
            "يرجى التأكد من ضبط متغيرات البيئة للمفاتيح.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    for ex_id, exchange in authenticated_exchanges:
        try:
            usdt_balance = await get_real_balance(ex_id, 'USDT')
            
            if usdt_balance > 0:
                balances_text += f"🔑 **{ex_id.upper()}**\n"
                balances_text += f"   • USDT: `{usdt_balance:.2f}`\n\n"
                total_usdt += usdt_balance
            else:
                balances_text += f"📊 **{ex_id.upper()}**\n"
                balances_text += f"   • USDT: `{usdt_balance:.2f}`\n\n"
                
        except Exception as e:
            balances_text += f"❌ **{ex_id.upper()}**\n"
            balances_text += f"   • خطأ: {str(e)[:50]}...\n\n"
    
    # معلومات إضافية
    trade_percentage = bot_data['settings'].get('real_trade_size_percentage', 2.0)
    potential_trade_size = total_usdt * (trade_percentage / 100)
    
    balances_text += (
        f"**📊 الملخص:**\n"
        f"• **إجمالي USDT:** `{total_usdt:.2f}`\n"
        f"• **حجم الصفقة الواحدة:** `{potential_trade_size:.2f}` ({trade_percentage}%)\n"
        f"• **عدد الصفقات المحتمل:** `{int(total_usdt / potential_trade_size) if potential_trade_size > 0 else 0}`\n\n"
        f"*محدث: {datetime.now(EGYPT_TZ).strftime('%H:%M:%S')}*"
    )
    
    await update.message.reply_text(balances_text, parse_mode=ParseMode.MARKDOWN)

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تشغيل فحص فوري للسوق"""
    
    if bot_data['status_snapshot']['scan_in_progress']:
        await update.message.reply_text(
            "**⏳ فحص قيد التنفيذ**\n\n"
            "يتم تنفيذ فحص للسوق حالياً. يرجى الانتظار حتى انتهائه.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    await update.message.reply_text(
        "**🚀 بدء الفحص الفوري**\n\n"
        "جاري فحص السوق والبحث عن الفرص...\n"
        "سيتم إرسال النتائج عند الانتهاء.",
        parse_mode=ParseMode.MARKDOWN
    )
    
    # تشغيل الفحص
    try:
        await perform_scan(context)
    except Exception as e:
        logger.error(f"💥 Manual scan failed: {e}")
        await update.message.reply_text(
            f"**❌ فشل الفحص الفوري**\n\n"
            f"حدث خطأ أثناء الفحص: {str(e)[:100]}...\n\n"
            f"سيتم إعادة المحاولة في الفحص التلقائي التالي.",
            parse_mode=ParseMode.MARKDOWN
        )

async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض إحصائيات الأداء"""
    
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        
        # إحصائيات عامة
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'ناجحة'")
        winning_trades = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'فاشلة'")
        losing_trades = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة'")
        active_trades = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(pnl_usdt) FROM trades WHERE pnl_usdt IS NOT NULL")
        total_pnl = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM trades WHERE is_real_trade = 1")
        real_trades = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(pnl_usdt) FROM trades WHERE is_real_trade = 1 AND pnl_usdt IS NOT NULL")
        real_pnl = cursor.fetchone()[0] or 0
        
        # إحصائيات الأسبوع الماضي
        week_ago = datetime.now(EGYPT_TZ) - timedelta(days=7)
        cursor.execute("""
            SELECT COUNT(*), SUM(pnl_usdt) 
            FROM trades 
            WHERE timestamp >= ? AND pnl_usdt IS NOT NULL
        """, (week_ago.strftime('%Y-%m-%d %H:%M:%S'),))
        
        week_result = cursor.fetchone()
        week_trades, week_pnl = week_result[0] or 0, week_result[1] or 0
        
        conn.close()
        
        # حساب النسب
        win_rate = (winning_trades / (winning_trades + losing_trades) * 100) if (winning_trades + losing_trades) > 0 else 0
        
        # معلومات الفحص الأخير
        scan_history = list(bot_data['scan_history'])
        recent_scans = scan_history[-5:] if scan_history else []
        avg_signals = sum(s.get('signals', 0) for s in recent_scans) / len(recent_scans) if recent_scans else 0
        avg_failures = sum(s.get('failures', 0) for s in recent_scans) / len(recent_scans) if recent_scans else 0
        
        performance_text = (
            f"**📈 إحصائيات الأداء**\n"
            f"{'═' * 30}\n\n"
            f"**📊 الصفقات الإجمالية:**\n"
            f"• **المجموع:** {total_trades}\n"
            f"• **ناجحة:** {winning_trades} ✅\n"
            f"• **فاشلة:** {losing_trades} ❌\n"
            f"• **نشطة:** {active_trades} 🔄\n"
            f"• **معدل النجاح:** {win_rate:.1f}%\n\n"
            f"**💰 الأرباح والخسائر:**\n"
            f"• **إجمالي P&L:** `${total_pnl:.2f}`\n"
            f"• **P&L الصفقات الحقيقية:** `${real_pnl:.2f}`\n"
            f"• **P&L الأسبوع الماضي:** `${week_pnl:.2f}` ({week_trades} صفقة)\n\n"
            f"**🔍 إحصائيات الفحص:**\n"
            f"• **متوسط الإشارات/فحص:** {avg_signals:.1f}\n"
            f"• **متوسط الأخطاء/فحص:** {avg_failures:.1f}\n"
            f"• **إجمالي الفحوص:** {len(scan_history)}\n\n"
            f"**🚨 الصفقات الحقيقية:**\n"
            f"• **العدد:** {real_trades}/{total_trades}\n"
            f"• **النسبة:** {(real_trades/total_trades*100) if total_trades > 0 else 0:.1f}%\n\n"
            f"*محدث: {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')}*"
        )
        
        await update.message.reply_text(performance_text, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error(f"💥 Error in performance_command: {e}")
        await update.message.reply_text(
            "**❌ خطأ في جلب الإحصائيات**\n\n"
            "حدث خطأ أثناء جلب إحصائيات الأداء من قاعدة البيانات.",
            parse_mode=ParseMode.MARKDOWN
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض قائمة الأوامر المتاحة"""
    
    help_text = (
        f"**🤖 دليل أوامر بوت التداول الحقيقي**\n"
        f"{'═' * 40}\n\n"
        f"**📋 الأوامر الأساسية:**\n"
        f"• `/start` - رسالة الترحيب والمعلومات الأساسية\n"
        f"• `/status` - حالة البوت ومعلومات التشغيل\n"
        f"• `/help` - عرض هذه القائمة\n\n"
        f"**📊 مراقبة التداول:**\n"
        f"• `/trades` - عرض الصفقات النشطة\n"
        f"• `/balance` - أرصدة المنصات المتصلة\n"
        f"• `/performance` - إحصائيات الأداء\n\n"
        f"**🔄 التحكم في الفحص:**\n"
        f"• `/scan` - فحص فوري للسوق\n\n"
        f"**⚙️ الإعدادات:**\n"
        f"• `/settings` - تعديل إعدادات البوت\n\n"
        f"**🆘 المساعدة:**\n"
        f"• `/check [ID]` - تفاصيل صفقة محددة\n\n"
        f"**⚠️ ملاحظات هامة:**\n"
        f"• البوت يعمل تلقائياً كل {SCAN_INTERVAL_SECONDS//60} دقيقة\n"
        f"• يتم مراقبة الصفقات كل {TRACK_INTERVAL_SECONDS//60} دقيقة\n"
        f"• جميع الأوامر تعمل فقط للمستخدمين المُخوَّلين\n\n"
        f"**🚨 تذكير:** هذا البوت يتداول بأموال حقيقية!"
    )
    
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

# --- Error Handler ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """معالج الأخطاء العام"""
    
    logger.error(f"💥 Exception while handling update {update}: {context.error}", exc_info=context.error)
    
    # إرسال رسالة للمطور في حالة أخطاء حرجة
    if update and hasattr(update, 'effective_chat'):
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"**❌ حدث خطأ في البوت**\n\n"
                     f"نعتذر، حدث خطأ أثناء معالجة طلبك.\n"
                     f"تم تسجيل الخطأ وسيتم إصلاحه قريباً.\n\n"
                     f"*يمكنك المحاولة مرة أخرى بعد قليل*",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.error(f"💥 Failed to send error message: {e}")

# --- Main Function ---
async def main():
    """الدالة الرئيسية لتشغيل البوت"""
    
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
        application.add_handler(CommandHandler("status", status_command))
        application.add_handler(CommandHandler("trades", trades_command))
        application.add_handler(CommandHandler("balance", balance_command))
        application.add_handler(CommandHandler("scan", scan_command))
        application.add_handler(CommandHandler("performance", performance_command))
        application.add_handler(CommandHandler("help", help_command))
        
        # معالج الأخطاء
        application.add_error_handler(error_handler)
        
        # إعداد الجدولة
        job_queue = application.job_queue
        
        # فحص السوق كل 15 دقيقة
        job_queue.run_repeating(
            perform_scan,
            interval=SCAN_INTERVAL_SECONDS,
            first=30,  # أول فحص بعد 30 ثانية
            name="market_scan"
        )
        
        # تتبع الصفقات كل دقيقتين
        job_queue.run_repeating(
            track_open_trades,
            interval=TRACK_INTERVAL_SECONDS,
            first=60,  # أول فحص بعد دقيقة
            name="trade_tracking"
        )
        
        logger.info("⏰ Scheduled jobs configured successfully")
        
        # إرسال رسالة البدء
        settings = bot_data['settings']
        startup_message = (
            f"**🚨 بوت التداول الحقيقي بدأ العمل! 🚨**\n\n"
            f"**⚙️ الوضع:** {'🚨 تداول حقيقي' if settings.get('real_trading_enabled') else '📊 تداول افتراضي'}\n"
            f"**📊 حجم الصفقات:** {settings.get('real_trade_size_percentage', 2.0)}%\n"
            f"**🎯 أقصى صفقات:** {settings.get('max_concurrent_trades', 3)}\n"
            f"**🏛️ منصات متصلة:** {len(bot_data['exchanges'])}\n\n"
            f"**⏰ الجدولة:**\n"
            f"• فحص السوق: كل {SCAN_INTERVAL_SECONDS//60} دقيقة\n"
            f"• مراقبة الصفقات: كل {TRACK_INTERVAL_SECONDS//60} دقيقة\n\n"
            f"**⚠️ تحذير:** البوت يتداول بأموال حقيقية!\n\n"
            f"*استخدم /help لعرض الأوامر المتاحة*"
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
        
        # بدء البوت
        logger.info("🎯 Real Trading Bot is ready and running!")
        logger.info(f"📊 Connected exchanges: {list(bot_data['exchanges'].keys())}")
        logger.info(f"🔑 Authenticated exchanges: {[ex for ex, obj in bot_data['exchanges'].items() if hasattr(obj, 'apiKey') and obj.apiKey]}")
        
        # تشغيل البوت
        await application.run_polling(
            poll_interval=1,
            timeout=10,
            bootstrap_retries=-1,  # إعادة المحاولة إلى ما لا نهاية
            read_timeout=20,
            write_timeout=20,
            connect_timeout=20,
            pool_timeout=20
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR in main: {e}", exc_info=True)
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
    print("🚀 Real Trading Bot v12 - Starting...")
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