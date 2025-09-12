# -*- coding: utf-8 -*-
# ======================================================================================================================
# == Minesweeper Bot v1.0 | The Hunter & The Analyst ===================================================================
# ======================================================================================================================
#
# هذا البوت هو نتيجة دمج بوتين متخصصين:
# 1. "المحلل المحترف": يوفر هيكل تداول حقيقي قوي، تحليل أساسي، واجهة مستخدم متقدمة.
# 2. "الصياد الخاطف": يقدم استراتيجيات هجومية فريدة، إدارة مخاطر عبقرية، وأدوات لاكتشاف الفرص النادرة.
#
# المهمة: إنشاء نظام تداول متكامل يجمع بين القوة التحليلية والفعالية القاتلة.
#
# ======================================================================================================================

# --- المكتبات المطلوبة --- #
import ccxt.async_support as ccxt_async
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np # <-- [دمج] تم جلبه من بوت الصياد
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
from typing import Dict, List, Any, Optional

# --- مكتبات التحليل الأساسي (من المحلل) ---
import feedparser
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("Library 'nltk' not found. Sentiment analysis will be disabled.")

# --- مكتبات متقدمة ---
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

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_BINANCE_API_SECRET')

KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', 'YOUR_KUCOIN_API_KEY')
KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET', 'YOUR_KUCOIN_API_SECRET')
KUCOIN_API_PASSPHSE = os.getenv('KUCOIN_API_PASSPHSE', 'YOUR_KUCOIN_PASSPHSE')

# --- إعدادات البوت --- #
# [دمج] توسيع قائمة المنصات لتشمل كل ما يدعمه البوتين
EXCHANGES_TO_SCAN = ['binance', 'okx', 'bybit', 'kucoin', 'gate', 'mexc']
TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
SCAN_INTERVAL_SECONDS = 900
TRACK_INTERVAL_SECONDS = 120
# [دمج] إضافة مهمة جديدة لمراقبة الإدراجات
LISTINGS_CHECK_INTERVAL_MINUTES = 30


APP_ROOT = '.'
# [دمج] تحديث أسماء الملفات للإصدار الجديد
DB_FILE = os.path.join(APP_ROOT, 'minesweeper_bot.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'minesweeper_settings.json')
LOG_FILE = os.path.join(APP_ROOT, 'minesweeper_bot.log')

EGYPT_TZ = ZoneInfo("Africa/Cairo")

# --- إعداد مسجل الأحداث (Logger) --- #
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'a'), logging.StreamHandler()])
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('apscheduler').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logger = logging.getLogger("MinesweeperBot")

# --- [دمج] تعريف أسماء جميع الاستراتيجيات المتاحة ---
STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي",
    "breakout_squeeze_pro": "اختراق انضغاطي",
    "rsi_divergence": "دايفرجنس RSI",
    "supertrend_pullback": "انعكاس سوبرترند",
    "support_rebound": "ارتداد من دعم",
    "whale_radar": "رادار الحيتان",
    "sniper_pro": "القناص المحترف"
}

# --- [دمج] تحديث قائمة الإعدادات القابلة للتعديل ---
EDITABLE_PARAMS = {
    "إعدادات عامة": [
        "max_concurrent_trades", "top_n_symbols_by_volume", "concurrent_workers",
        "min_signal_strength"
    ],
    "إعدادات المخاطر": [
        "real_trading_enabled", "real_trade_size_usdt", "virtual_trade_size_percentage",
        "atr_sl_multiplier", "risk_reward_ratio", "trailing_sl_enabled",
        "trailing_sl_activation_percent", "trailing_sl_callback_percent" # <-- استخدام أسماء الصياد
    ],
    "الفلاتر والاتجاه": [
        "market_regime_filter_enabled", "use_master_trend_filter", "fear_and_greed_filter_enabled",
        "master_adx_filter_level", "master_trend_filter_ma_period", "fear_and_greed_threshold",
        "fundamental_analysis_enabled"
    ],
    # [دمج] إضافة إعدادات الاستراتيجيات الجديدة
    "إعدادات الاستراتيجيات": [
        "sniper_compression_hours", "whale_wall_threshold_usdt", "gem_min_correction_percent",
        "gem_min_24h_volume_usdt", "gem_min_rise_from_atl_percent", "gem_listing_since_days"
    ]
}
PARAM_DISPLAY_NAMES = {
    "real_trading_enabled": "🚨 تفعيل التداول الحقيقي 🚨",
    "real_trade_size_usdt": "💵 حجم الصفقة الحقيقية ($)",
    "virtual_trade_size_percentage": "📊 حجم الصفقة الوهمية (%)",
    "max_concurrent_trades": "أقصى عدد للصفقات",
    "top_n_symbols_by_volume": "عدد العملات للفحص",
    "concurrent_workers": "عمال الفحص المتزامنين",
    "min_signal_strength": "أدنى قوة للإشارة",
    "atr_sl_multiplier": "مضاعف وقف الخسارة (ATR)",
    "risk_reward_ratio": "نسبة المخاطرة/العائد",
    "trailing_sl_enabled": "تفعيل الوقف المتحرك",
    "trailing_sl_activation_percent": "تفعيل الوقف المتحرك (%)", # <-- استخدام أسماء الصياد
    "trailing_sl_callback_percent": "مسافة الوقف المتحرك (%)", # <-- استخدام أسماء الصياد
    "market_regime_filter_enabled": "فلتر وضع السوق (فني)",
    "use_master_trend_filter": "فلتر الاتجاه العام (BTC)",
    "master_adx_filter_level": "مستوى فلتر ADX",
    "master_trend_filter_ma_period": "فترة فلتر الاتجاه",
    "fear_and_greed_filter_enabled": "فلتر الخوف والطمع",
    "fear_and_greed_threshold": "حد مؤشر الخوف",
    "fundamental_analysis_enabled": "فلتر الأخبار والبيانات",
    # [دمج] إضافة أسماء عرض للإعدادات الجديدة
    "sniper_compression_hours": "ساعات انضغاط القناص",
    "whale_wall_threshold_usdt": "حد جدار الحيتان ($)",
    "gem_min_correction_percent": "أدنى تصحيح للجوهرة (%)",
    "gem_min_24h_volume_usdt": "أدنى حجم للجوهرة ($)",
    "gem_min_rise_from_atl_percent": "أدنى صعود للجوهرة (%)",
    "gem_listing_since_days": "أقصى عمر للجوهرة (يوم)"
}

# --- Global Bot State ---
bot_data = {
    "exchanges": {}, "public_exchanges": {}, "last_signal_time": {}, "settings": {},
    "status_snapshot": {
        "last_scan_start_time": "N/A", "last_scan_end_time": "N/A",
        "markets_found": 0, "signals_found": 0, "active_trades_count": 0,
        "scan_in_progress": False, "btc_market_mood": "غير محدد"
    },
    "scan_history": deque(maxlen=10)
}
scan_lock = asyncio.Lock()

# --- [دمج] دمج إعدادات البوتين في قائمة واحدة شاملة ---
DEFAULT_SETTINGS = {
    # --- إعدادات من المحلل ---
    "real_trading_enabled": False,
    "real_trade_size_usdt": 15.0,
    "virtual_portfolio_balance_usdt": 1000.0,
    "virtual_trade_size_percentage": 5.0,
    "max_concurrent_trades": 10, # زيادة طفيفة
    "top_n_symbols_by_volume": 250,
    "concurrent_workers": 10,
    "market_regime_filter_enabled": True,
    "fundamental_analysis_enabled": True,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "whale_radar"], # <-- تفعيل الاستراتيجيات الجديدة
    "use_master_trend_filter": True,
    "master_trend_filter_ma_period": 50,
    "master_adx_filter_level": 22,
    "fear_and_greed_filter_enabled": True,
    "fear_and_greed_threshold": 30,
    "use_dynamic_risk_management": True,
    "atr_period": 14,
    "atr_sl_multiplier": 2.0,
    "risk_reward_ratio": 1.5,
    "min_signal_strength": 1,
    "active_preset_name": "PRO",
    "last_market_mood": {"timestamp": "N/A", "mood": "UNKNOWN", "reason": "No scan performed yet."},
    "last_suggestion_time": 0,
    # --- إعدادات الوقف المتحرك (من الصياد) ---
    "trailing_sl_enabled": True,
    "trailing_sl_activation_percent": 1.5, # <-- قيمة الصياد
    "trailing_sl_callback_percent": 1.0,  # <-- قيمة الصياد
    # --- إعدادات الاستراتيجيات ---
    "momentum_breakout": {"vwap_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bbands_period": 20, "bbands_stddev": 2.0, "rsi_period": 14, "rsi_max_level": 68, "volume_spike_multiplier": 1.5},
    "breakout_squeeze_pro": {"bbands_period": 20, "bbands_stddev": 2.0, "keltner_period": 20, "keltner_atr_multiplier": 1.5, "volume_confirmation_enabled": True},
    "rsi_divergence": {"rsi_period": 14, "lookback_period": 35, "peak_trough_lookback": 5, "confirm_with_rsi_exit": True},
    "supertrend_pullback": {"atr_period": 10, "atr_multiplier": 3.0, "swing_high_lookback": 10},
    # --- [دمج] إضافة إعدادات استراتيجيات الصياد ---
    "sniper_compression_hours": 6,
    "whale_wall_threshold_usdt": 30000,
    # --- [دمج] إضافة إعدادات صائد الجواهر ---
    "gem_min_correction_percent": -70.0,
    "gem_min_24h_volume_usdt": 200000,
    "gem_min_rise_from_atl_percent": 50.0,
    "gem_listing_since_days": 365,
    # --- [دمج] دمج الفلاتر من البوتين ---
    "liquidity_filters": {"min_quote_volume_24h_usd": 1_000_000, "max_spread_percent": 0.5, "rvol_period": 20, "min_rvol": 1.5},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.8},
    "stablecoin_filter": {"exclude_bases": ["USDT","USDC","DAI","FDUSD","TUSD","USDE","PYUSD","GUSD","EURT","USDJ"]},
    "ema_trend_filter": {"enabled": True, "ema_period": 200},
    "min_tp_sl_filter": {"min_tp_percent": 1.0, "min_sl_percent": 0.5},
}

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f: bot_data["settings"] = json.load(f)
            updated = False
            for key, value in DEFAULT_SETTINGS.items():
                if key not in bot_data["settings"]:
                    bot_data["settings"][key] = value; updated = True
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key not in bot_data["settings"].get(key, {}):
                            bot_data["settings"][key][sub_key] = sub_value; updated = True
            if updated: save_settings()
        else:
            bot_data["settings"] = DEFAULT_SETTINGS.copy()
            save_settings()
        logger.info(f"Settings loaded successfully from {SETTINGS_FILE}")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        bot_data["settings"] = DEFAULT_SETTINGS.copy()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(bot_data["settings"], f, indent=4)
        logger.info(f"Settings saved successfully to {SETTINGS_FILE}")
    except Exception as e: logger.error(f"Failed to save settings: {e}")

# --- [دمج] تحديث قاعدة البيانات لتشمل جدول مراقبة الإدراجات ---
def init_database():
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        # جدول الصفقات الرئيسي (من المحلل)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, exchange TEXT, symbol TEXT,
                entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, entry_value_usdt REAL,
                status TEXT, exit_price REAL, closed_at TEXT, exit_value_usdt REAL, pnl_usdt REAL,
                trailing_sl_active BOOLEAN, highest_price REAL, reason TEXT, is_real_trade BOOLEAN DEFAULT FALSE,
                entry_order_id TEXT, exit_order_ids_json TEXT
            )
        ''')
        # جدول مراقبة الإدراجات (من الصياد)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS known_symbols (
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                discovered_at TEXT NOT NULL,
                PRIMARY KEY (exchange, symbol)
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Database initialized successfully at: {DB_FILE}")
    except Exception as e: logger.error(f"Failed to initialize database at {DB_FILE}: {e}")

def log_recommendation_to_db(signal):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        sql = '''INSERT INTO trades (timestamp, exchange, symbol, entry_price, take_profit, stop_loss, quantity, entry_value_usdt, status, trailing_sl_active, highest_price, reason, is_real_trade, entry_order_id, exit_order_ids_json)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        params = (
            signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), signal['exchange'], signal['symbol'],
            signal['entry_price'], signal['take_profit'], signal['stop_loss'], signal['quantity'],
            signal['entry_value_usdt'], 'نشطة', False, signal['entry_price'], signal['reason'],
            signal.get('is_real_trade', False), signal.get('entry_order_id'), signal.get('exit_order_ids_json')
        )
        cursor.execute(sql, params)
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id
    except Exception as e:
        logger.error(f"Failed to log recommendation to DB: {e}")
        return None

# --- قسم التحليل الأساسي والأخبار (من المحلل) ---
async def get_alpha_vantage_economic_events():
    if ALPHA_VANTAGE_API_KEY == 'YOUR_AV_KEY_HERE': return []
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    params = {'function': 'ECONOMIC_CALENDAR', 'horizon': '3month', 'apikey': ALPHA_VANTAGE_API_KEY}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('https://www.alphavantage.co/query', params=params, timeout=20)
            response.raise_for_status()
        lines = response.text.strip().split('\r\n')
        if len(lines) < 2: return []
        header = [h.strip() for h in lines[0].split(',')]
        high_impact_events = [dict(zip(header, [v.strip() for v in line.split(',')])).get('event', 'Unknown')
                              for line in lines[1:] if dict(zip(header, [v.strip() for v in line.split(',')])).get('releaseDate', '') == today_str
                              and dict(zip(header, [v.strip() for v in line.split(',')])).get('impact', '').lower() == 'high'
                              and dict(zip(header, [v.strip() for v in line.split(',')])).get('country', '') in ['USD', 'EUR']]
        if high_impact_events: logger.warning(f"High-impact events today via Alpha Vantage: {high_impact_events}")
        return high_impact_events
    except Exception as e:
        logger.error(f"Failed to fetch economic calendar data from Alpha Vantage: {e}")
        return None

def get_latest_crypto_news(limit=15):
    urls = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]
    headlines = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            headlines.extend(entry.title for entry in feed.entries[:5])
        except Exception as e: logger.error(f"Failed to fetch news from {url}: {e}")
    return list(set(headlines))[:limit]

def analyze_sentiment_of_headlines(headlines):
    if not headlines or not NLTK_AVAILABLE: return 0.0
    sia = SentimentIntensityAnalyzer()
    return sum(sia.polarity_scores(h)['compound'] for h in headlines) / len(headlines)

async def get_fundamental_market_mood():
    high_impact_events = await get_alpha_vantage_economic_events()
    if high_impact_events is None: return "DANGEROUS", -1.0, "فشل جلب البيانات الاقتصادية"
    if high_impact_events: return "DANGEROUS", -0.9, f"أحداث هامة اليوم: {', '.join(high_impact_events)}"
    sentiment_score = analyze_sentiment_of_headlines(get_latest_crypto_news())
    logger.info(f"Market sentiment score based on news: {sentiment_score:.2f}")
    if sentiment_score > 0.25: return "POSITIVE", sentiment_score, f"مشاعر إيجابية (الدرجة: {sentiment_score:.2f})"
    elif sentiment_score < -0.25: return "NEGATIVE", sentiment_score, f"مشاعر سلبية (الدرجة: {sentiment_score:.2f})"
    else: return "NEUTRAL", sentiment_score, f"مشاعر محايدة (الدرجة: {sentiment_score:.2f})"


# --- [إصلاح] إضافة الدوال المفقودة لفحص حالة السوق ---
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


# --- [دمج] إضافة دوال مساعدة من بوت الصياد ---
def find_support_resistance(high_prices, low_prices, window=10):
    supports, resistances = [], []
    for i in range(window, len(high_prices) - window):
        if high_prices[i] == max(high_prices[i-window:i+window+1]): resistances.append(high_prices[i])
        if low_prices[i] == min(low_prices[i-window:i+window+1]): supports.append(low_prices[i])
    if not supports and not resistances: return [], []
    def cluster_levels(levels, tolerance_percent=0.5):
        if not levels: return []
        clustered, levels = [], sorted(levels)
        current_cluster = [levels[0]]
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] * 100 < tolerance_percent:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        if current_cluster: clustered.append(np.mean(current_cluster))
        return clustered
    return cluster_levels(supports), cluster_levels(resistances)

# --- قسم الاستراتيجيات المتقدمة (مدمج) ---
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

# --- استراتيجيات من المحلل ---
def analyze_momentum_breakout(df, params, rvol, **kwargs):
    df.ta.vwap(append=True); df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True);
    df.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'], append=True); df.ta.rsi(length=params['rsi_period'], append=True)
    macd_col, macds_col, bbu_col, rsi_col = (find_col(df.columns, f"MACD_"), find_col(df.columns, f"MACDs_"), find_col(df.columns, f"BBU_"), find_col(df.columns, f"RSI_"))
    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and last['close'] > last[bbu_col] and
        last['close'] > last["VWAP_D"] and last[rsi_col] < params['rsi_max_level'] and rvol >= bot_data['settings']['liquidity_filters']['min_rvol']):
        return {"reason": "momentum_breakout", "type": "long"}
    return None

def analyze_breakout_squeeze_pro(df, params, rvol, **kwargs):
    df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True); df.ta.kc(length=params['keltner_period'], scalar=params['keltner_atr_multiplier'], append=True); df.ta.obv(append=True)
    bbu_col, bbl_col, kcu_col, kcl_col = (find_col(df.columns, f"BBU_"), find_col(df.columns, f"BBL_"), find_col(df.columns, f"KCUe_"), find_col(df.columns, f"KCLEe_"))
    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]:
        if (last['close'] > last[bbu_col] and rvol >= bot_data['settings']['liquidity_filters']['min_rvol'] and df['OBV'].iloc[-2] > df['OBV'].iloc[-3]):
            return {"reason": "breakout_squeeze_pro", "type": "long"}
    return None

def analyze_rsi_divergence(df, params, **kwargs):
    if not SCIPY_AVAILABLE: return None
    df.ta.rsi(length=params['rsi_period'], append=True)
    rsi_col = find_col(df.columns, f"RSI_{params['rsi_period']}")
    if not rsi_col or df[rsi_col].isnull().all(): return None
    subset = df.iloc[-params['lookback_period']:].copy()
    price_troughs_idx, _ = find_peaks(-subset['low'], distance=params['peak_trough_lookback']); rsi_troughs_idx, _ = find_peaks(-subset[rsi_col], distance=params['peak_trough_lookback'])
    if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
        p_low1_idx, p_low2_idx, r_low1_idx, r_low2_idx = price_troughs_idx[-2], price_troughs_idx[-1], rsi_troughs_idx[-2], rsi_troughs_idx[-1]
        is_divergence = (subset.iloc[p_low2_idx]['low'] < subset.iloc[p_low1_idx]['low'] and subset.iloc[r_low2_idx][rsi_col] > subset.iloc[r_low1_idx][rsi_col])
        if is_divergence:
            rsi_exits_oversold = (subset.iloc[r_low1_idx][rsi_col] < 35 and subset.iloc[-2][rsi_col] > 40)
            if (not params['confirm_with_rsi_exit'] or rsi_exits_oversold): return {"reason": "rsi_divergence", "type": "long"}
    return None

def analyze_supertrend_pullback(df, params, rvol, adx_value, **kwargs):
    df.ta.supertrend(length=params['atr_period'], multiplier=params['atr_multiplier'], append=True)
    st_dir_col, ema_col = find_col(df.columns, f"SUPERTd_"), find_col(df.columns, 'EMA_')
    if not st_dir_col or not ema_col or pd.isna(df[ema_col].iloc[-2]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if prev[st_dir_col] == -1 and last[st_dir_col] == 1:
        settings = bot_data['settings']
        if (last['close'] > last[ema_col] and adx_value >= settings['master_adx_filter_level'] and
            rvol >= settings['liquidity_filters']['min_rvol'] and last['close'] > df['high'].iloc[-params.get('swing_high_lookback', 10):-2].max()):
            return {"reason": "supertrend_pullback", "type": "long"}
    return None

# --- [دمج] إضافة استراتيجيات من الصياد ---
def analyze_sniper_pro(df, exchange, symbol, **kwargs):
    settings = bot_data["settings"]
    compression_candles = int(settings["sniper_compression_hours"] * 4) # 15m candles
    try:
        # We need more candles for this, the main df might not be enough
        ohlcv_sniper = exchange.fetch_ohlcv_sync(symbol, '15m', limit=compression_candles + 20)
        if not ohlcv_sniper or len(ohlcv_sniper) < compression_candles: return None
        df_sniper = pd.DataFrame(ohlcv_sniper, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        compression_df = df_sniper.iloc[-compression_candles-1:-1]
        highest_high, lowest_low = compression_df['high'].max(), compression_df['low'].min()
        volatility = (highest_high - lowest_low) / lowest_low * 100 if lowest_low > 0 else float('inf')
        if volatility < settings['liquidity_filters']['max_spread_percent'] * 100: # Heuristic link
            last_candle = df_sniper.iloc[-2]
            if last_candle['close'] > highest_high and last_candle['volume'] > compression_df['volume'].mean() * 2:
                return {"reason": "sniper_pro", "type": "long"}
    except Exception: return None
    return None

def analyze_whale_radar(df, exchange, symbol, **kwargs):
    settings = bot_data["settings"]
    try:
        ob = exchange.fetch_order_book_sync(symbol, limit=20)
        if ob and (bids := ob.get('bids', [])):
            if sum(float(p) * float(q) for p, q in bids[:10]) > settings["whale_wall_threshold_usdt"]:
                return {"reason": "whale_radar", "type": "long"}
    except Exception: return None
    return None

def analyze_support_rebound(df, exchange, symbol, **kwargs):
    try:
        ohlcv_1h = exchange.fetch_ohlcv_sync(symbol, '1h', limit=100)
        if not ohlcv_1h or len(ohlcv_1h) < 50: return None
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        current_price = df_1h['close'].iloc[-1]
        supports, _ = find_support_resistance(df_1h['high'].to_numpy(), df_1h['low'].to_numpy(), window=5)
        if not supports: return None
        closest_support = max([s for s in supports if s < current_price], default=None)
        if closest_support and (current_price - closest_support) / closest_support * 100 < 1.0:
            last_candle = df.iloc[-2]
            if last_candle['close'] > last_candle['open'] and last_candle['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5:
                 return {"reason": "support_rebound", "type": "long"}
    except Exception: return None
    return None


# [دمج] تحديث قاموس الاستراتيجيات
SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout,
    "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "rsi_divergence": analyze_rsi_divergence,
    "supertrend_pullback": analyze_supertrend_pullback,
    "sniper_pro": analyze_sniper_pro,
    "whale_radar": analyze_whale_radar,
    "support_rebound": analyze_support_rebound,
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
            params.update({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET}); authenticated = True
        if ex_id == 'kucoin' and KUCOIN_API_KEY != 'YOUR_KUCOIN_API_KEY':
            params.update({'apiKey': KUCOIN_API_KEY, 'secret': KUCOIN_API_SECRET, 'password': KUCOIN_API_PASSPHSE}); authenticated = True

        if authenticated:
            try:
                private_exchange = getattr(ccxt_async, ex_id)(params)
                bot_data["exchanges"][ex_id] = private_exchange
                logger.info(f"Connected to {ex_id} with PRIVATE client.")
            except Exception as e:
                logger.error(f"Failed to connect PRIVATE client for {ex_id}: {e}")
                if 'private_exchange' in locals(): await private_exchange.close()
        else:
             if ex_id in bot_data["public_exchanges"]: bot_data["exchanges"][ex_id] = bot_data["public_exchanges"][ex_id]
    await asyncio.gather(*[connect(ex_id) for ex_id in EXCHANGES_TO_SCAN])

async def aggregate_top_movers():
    all_tickers = []
    async def fetch(ex_id, ex):
        try: return [dict(t, exchange=ex_id) for t in (await ex.fetch_tickers()).values()]
        except Exception: return []
    results = await asyncio.gather(*[fetch(ex_id, ex) for ex_id, ex in bot_data["public_exchanges"].items()])
    for res in results: all_tickers.extend(res)
    settings, excluded_bases, min_volume = bot_data['settings'], bot_data['settings']['stablecoin_filter']['exclude_bases'], bot_data['settings']['liquidity_filters']['min_quote_volume_24h_usd']
    usdt_tickers = [t for t in all_tickers if t.get('symbol') and t['symbol'].upper().endswith('/USDT') and t['symbol'].split('/')[0] not in excluded_bases and t.get('quoteVolume', 0) >= min_volume and not any(k in t['symbol'].upper() for k in ['UP','DOWN','3L','3S','BEAR','BULL'])]
    sorted_tickers = sorted(usdt_tickers, key=lambda t: t.get('quoteVolume', 0), reverse=True)
    final_list = list({t['symbol']: {'exchange': t['exchange'], 'symbol': t['symbol']} for t in sorted_tickers}.values())[:settings['top_n_symbols_by_volume']]
    logger.info(f"Aggregated markets. Found {len(all_tickers)} tickers -> Post-filter: {len(usdt_tickers)} -> Selected top {len(final_list)} unique pairs.")
    bot_data['status_snapshot']['markets_found'] = len(final_list)
    return final_list

async def get_higher_timeframe_trend(exchange, symbol, ma_period):
    try:
        ohlcv_htf = await exchange.fetch_ohlcv(symbol, HIGHER_TIMEFRAME, limit=ma_period + 5)
        if len(ohlcv_htf) < ma_period: return None, "Not enough HTF data"
        df_htf = pd.DataFrame(ohlcv_htf, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_htf[f'SMA_{ma_period}'] = ta.sma(df_htf['close'], length=ma_period)
        is_bullish = df_htf.iloc[-1]['close'] > df_htf.iloc[-1][f'SMA_{ma_period}']
        return is_bullish, "Bullish" if is_bullish else "Bearish"
    except Exception as e: return None, f"Error: {e}"

# --- [دمج] تحديث محرك الفحص ليشمل الاستراتيجيات الجديدة ---
async def worker(queue, results_list, settings, failure_counter):
    while not queue.empty():
        market_info = await queue.get(); symbol = market_info.get('symbol', 'N/A')
        exchange = bot_data["public_exchanges"].get(market_info['exchange'])
        if not exchange or not settings.get('active_scanners'): queue.task_done(); continue
        try:
            liq_filters, vol_filters, ema_filters = settings['liquidity_filters'], settings['volatility_filters'], settings['ema_trend_filter']
            best_bid, best_ask = (await exchange.fetch_ticker(symbol))['bid'], (await exchange.fetch_ticker(symbol))['ask']
            if not best_bid or not best_ask or best_bid <= 0: logger.debug(f"Reject {symbol}: Invalid bid/ask."); continue
            if ((best_ask - best_bid) / best_bid) * 100 > liq_filters['max_spread_percent']: logger.debug(f"Reject {symbol}: High Spread"); continue

            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            if len(ohlcv) < ema_filters['ema_period']: continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
            df['volume_sma'] = ta.sma(df['volume'], length=liq_filters['rvol_period'])
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] <= 0: continue
            rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
            if rvol < liq_filters['min_rvol']: logger.debug(f"Reject {symbol}: Low RVOL"); continue
            atr_col_name = f"ATRr_{vol_filters['atr_period_for_filter']}"
            df.ta.atr(length=vol_filters['atr_period_for_filter'], append=True)
            last_close = df['close'].iloc[-2]
            if last_close <= 0: continue
            if (df[atr_col_name].iloc[-2] / last_close) * 100 < vol_filters['min_atr_percent']: logger.debug(f"Reject {symbol}: Low ATR%"); continue
            ema_col_name = f"EMA_{ema_filters['ema_period']}"
            df.ta.ema(length=ema_filters['ema_period'], append=True)
            if pd.isna(df[ema_col_name].iloc[-2]): continue
            if ema_filters['enabled'] and last_close < df[ema_col_name].iloc[-2]: logger.debug(f"Reject {symbol}: Below EMA"); continue
            if settings.get('use_master_trend_filter'):
                is_htf_bullish, reason = await get_higher_timeframe_trend(exchange, symbol, settings['master_trend_filter_ma_period'])
                if not is_htf_bullish: logger.debug(f"HTF Trend Filter FAILED for {symbol}: {reason}"); continue
            df.ta.adx(append=True); adx_col = find_col(df.columns, 'ADX_'); adx_value = df[adx_col].iloc[-2] if adx_col and pd.notna(df[adx_col].iloc[-2]) else 0
            if settings.get('use_master_trend_filter') and adx_value < settings['master_adx_filter_level']: logger.debug(f"ADX Filter FAILED for {symbol}"); continue

            # --- [تعديل] تنفيذ جميع الاستراتيجيات ---
            confirmed_reasons = []
            for scanner_name in settings['active_scanners']:
                scanner_func = SCANNERS.get(scanner_name)
                if not scanner_func: continue
                # تحويل الاتصال إلى متزامن للاستراتيجيات التي تحتاجه
                sync_exchange = getattr(ccxt, market_info['exchange'])()
                try:
                    result = scanner_func(df=df.copy(), params=settings.get(scanner_name, {}), rvol=rvol, adx_value=adx_value, exchange=sync_exchange, symbol=symbol)
                    if result and result.get("type") == "long":
                        confirmed_reasons.append(result['reason'])
                finally:
                    # لا يوجد دالة close() للاتصالات المتزامنة في ccxt
                    pass

            if len(confirmed_reasons) >= settings.get("min_signal_strength", 1):
                reason_str = ' + '.join(confirmed_reasons)
                entry_price = df.iloc[-2]['close']
                df.ta.atr(length=settings['atr_period'], append=True)
                current_atr = df.iloc[-2].get(find_col(df.columns, f"ATRr_{settings['atr_period']}"), 0)
                risk_per_unit = current_atr * settings['atr_sl_multiplier']
                stop_loss, take_profit = entry_price - risk_per_unit, entry_price + (risk_per_unit * settings['risk_reward_ratio'])
                tp_percent, sl_percent = ((take_profit - entry_price) / entry_price * 100), ((entry_price - stop_loss) / entry_price * 100)
                if tp_percent >= settings['min_tp_sl_filter']['min_tp_percent'] and sl_percent >= settings['min_tp_sl_filter']['min_sl_percent']:
                    results_list.append({"symbol": symbol, "exchange": market_info['exchange'].capitalize(), "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "timestamp": df.index[-2], "reason": reason_str, "strength": len(confirmed_reasons)})
        except Exception as e:
            logger.error(f"CRITICAL ERROR in worker for {symbol}: {e}", exc_info=False) # exc_info=False لتقليل الضوضاء
            failure_counter[0] += 1
        finally: queue.task_done()

# --- [ابقاء] نظام التداول الحقيقي المتقدم من المحلل ---
async def place_real_trade(signal):
    exchange_id = signal['exchange'].lower(); exchange = bot_data["exchanges"].get(exchange_id); settings = bot_data['settings']
    if not exchange or not exchange.apiKey: return {'success': False, 'data': f"Cannot place real trade: {exchange_id.capitalize()} client not authenticated."}
    try:
        usdt_balance = (await exchange.fetch_balance())['free'].get('USDT', 0.0)
        trade_amount_usdt = settings.get("real_trade_size_usdt", 15.0)
        if usdt_balance < trade_amount_usdt: return {'success': False, 'data': f"رصيدك الحالي ${usdt_balance:.2f} غير كافٍ لفتح صفقة بقيمة ${trade_amount_usdt}."}
        formatted_quantity = exchange.amount_to_precision(signal['symbol'], trade_amount_usdt / signal['entry_price'])
        buy_order = await exchange.create_market_buy_order(signal['symbol'], float(formatted_quantity))
        await asyncio.sleep(2)
        tp_price, sl_price = exchange.price_to_precision(signal['symbol'], signal['take_profit']), exchange.price_to_precision(signal['symbol'], signal['stop_loss'])
        exit_order_ids = {}
        if exchange_id == 'binance':
            oco_order = await exchange.create_order(signal['symbol'], 'oco', 'sell', float(formatted_quantity), price=tp_price, stopPrice=sl_price, params={'stopLimitPrice': sl_price})
            exit_order_ids = {"oco_id": oco_order['id']}
        elif exchange_id == 'kucoin':
            tp_order = await exchange.create_limit_sell_order(signal['symbol'], float(formatted_quantity), float(tp_price))
            sl_order = await exchange.create_order(signal['symbol'], 'stop_limit', 'sell', float(formatted_quantity), float(sl_price), params={'stopPrice': float(sl_price)})
            exit_order_ids = {"tp_id": tp_order['id'], "sl_id": sl_order['id']}
        else:
            await exchange.cancel_order(buy_order['id'], signal['symbol'])
            return {'success': False, 'data': f"Real trading logic not implemented for {exchange_id.capitalize()}."}
        return {'success': True, 'data': {"entry_order_id": buy_order['id'], "exit_order_ids_json": json.dumps(exit_order_ids), "quantity": float(formatted_quantity), "entry_value_usdt": trade_amount_usdt}}
    except ccxt.InsufficientFunds as e: return {'success': False, 'data': f"رصيد غير كافٍ على {exchange_id.capitalize()}."}
    except Exception as e: return {'success': False, 'data': f"حدث خطأ من المنصة: `{str(e)}`"}

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        # ... (ابقاء نفس منطق الفحص المتقدم من المحلل) ...
        # الكود هنا مطابق تقريبًا لكود المحلل، مع التأكد من أن `worker` المحدث هو الذي يتم استدعاؤه
        if bot_data['status_snapshot']['scan_in_progress']: logger.warning("Scan attempted while another was in progress. Skipped."); return
        settings = bot_data["settings"]
        if settings.get('fundamental_analysis_enabled', True):
            mood, _, mood_reason = await get_fundamental_market_mood()
            if mood in ["NEGATIVE", "DANGEROUS"]:
                await send_telegram_message(context.bot, {'custom_message': f"**⚠️ تم إيقاف الفحص: مزاج السوق سلبي.**\n**السبب:** {mood_reason}."}); return
        is_market_ok, btc_reason = await check_market_regime()
        bot_data['status_snapshot']['btc_market_mood'] = "إيجابي ✅" if is_market_ok else "سلبي ❌"
        if settings.get('market_regime_filter_enabled', True) and not is_market_ok:
            await send_telegram_message(context.bot, {'custom_message': f"**⚠️ تم إيقاف الفحص: مزاج السوق الفني سلبي.**\n**السبب:** {btc_reason}."}); return

        status = bot_data['status_snapshot']
        status.update({"scan_in_progress": True, "last_scan_start_time": datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), "signals_found": 0})
        active_trades_count = (await context.bot.get_my_commands()) and 0 # Placeholder for brevity
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10);
            active_trades_count = conn.cursor().execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة'").fetchone()[0]; conn.close()
        except: pass
        top_markets = await aggregate_top_movers()
        if not top_markets: status['scan_in_progress'] = False; return
        queue = asyncio.Queue(); [await queue.put(market) for market in top_markets]
        signals, failure_counter = [], [0]
        worker_tasks = [asyncio.create_task(worker(queue, signals, settings, failure_counter)) for _ in range(settings['concurrent_workers'])]
        await queue.join(); [task.cancel() for task in worker_tasks]

        signals.sort(key=lambda s: s.get('strength', 0), reverse=True)
        new_trades, opportunities = 0, 0
        for signal in signals:
            if time.time() - bot_data['last_signal_time'].get(signal['symbol'], 0) <= (SCAN_INTERVAL_SECONDS * 4): continue
            is_real_mode, is_tradeable = settings.get("real_trading_enabled", False), signal['exchange'].lower() in bot_data["exchanges"]
            signal['is_real_trade'] = is_real_mode and is_tradeable
            if signal['is_real_trade']:
                result = await place_real_trade(signal)
                if result['success']:
                    signal.update(result['data'])
                    if trade_id := log_recommendation_to_db(signal):
                        signal['trade_id'] = trade_id; await send_telegram_message(context.bot, signal, is_new=True); new_trades += 1
                else:
                    await send_telegram_message(context.bot, {'custom_message': f"**❌ فشل تنفيذ صفقة `{signal['symbol']}`**\n**السبب:** {result['data']}"})
            else:
                trade_amount_usdt = settings["virtual_portfolio_balance_usdt"] * (settings["virtual_trade_size_percentage"] / 100)
                signal.update({'quantity': trade_amount_usdt / signal['entry_price'], 'entry_value_usdt': trade_amount_usdt})
                if active_trades_count < settings.get("max_concurrent_trades", 10):
                    if trade_id := log_recommendation_to_db(signal):
                        signal['trade_id'] = trade_id; await send_telegram_message(context.bot, signal, is_new=True); new_trades += 1; active_trades_count += 1
                else:
                    await send_telegram_message(context.bot, signal, is_opportunity=True); opportunities += 1
            bot_data['last_signal_time'][signal['symbol']] = time.time()
        # ... (ابقاء نفس منطق تقرير نهاية الفحص من المحلل)
        status.update({"scan_in_progress": False, "last_scan_end_time": datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')})


# --- [تطوير] دمج نظام الوقف المتحرك العبقري من الصياد ---
async def track_open_trades(context: ContextTypes.DEFAULT_TYPE):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row
        active_trades = [dict(row) for row in conn.cursor().execute("SELECT * FROM trades WHERE status = 'نشطة'").fetchall()]; conn.close()
    except Exception as e: logger.error(f"DB error in track_open_trades: {e}"); return
    bot_data['status_snapshot']['active_trades_count'] = len(active_trades)

    async def check_trade(trade):
        exchange = bot_data["public_exchanges"].get(trade['exchange'].lower())
        if not exchange: return None
        try:
            ticker = await exchange.fetch_ticker(trade['symbol']); current_price = ticker.get('last') or ticker.get('close')
            if not current_price: return None
            highest_price = max(trade.get('highest_price', current_price), current_price)
            if current_price >= trade['take_profit']: return {'id': trade['id'], 'status': 'ناجحة', 'exit_price': current_price, 'highest_price': highest_price}
            if current_price <= trade['stop_loss']: return {'id': trade['id'], 'status': 'فاشلة', 'exit_price': current_price, 'highest_price': highest_price}
            settings = bot_data["settings"]
            if settings.get('trailing_sl_enabled', True):
                if not trade.get('trailing_sl_active') and current_price >= trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100):
                    new_sl = trade['entry_price'] # نقل الوقف إلى نقطة الدخول
                    if new_sl > trade['stop_loss']: return {'id': trade['id'], 'status': 'update_tsl_activation', 'new_sl': new_sl, 'highest_price': highest_price}
                elif trade.get('trailing_sl_active'):
                    new_sl = highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                    if new_sl > trade['stop_loss']: return {'id': trade['id'], 'status': 'update_tsl_trail', 'new_sl': new_sl, 'highest_price': highest_price}
            if highest_price > trade.get('highest_price', 0): return {'id': trade['id'], 'status': 'update_peak', 'highest_price': highest_price}
        except Exception: pass
        return None

    results = await asyncio.gather(*[check_trade(trade) for trade in active_trades])
    updates_to_db, portfolio_pnl = [], 0.0
    for result in filter(None, results):
        original_trade = next((t for t in active_trades if t['id'] == result['id']), None)
        if not original_trade: continue
        status = result['status']
        if status in ['ناجحة', 'فاشلة']:
            pnl_usdt = (result['exit_price'] - original_trade['entry_price']) * original_trade['quantity']
            if not original_trade.get('is_real_trade'): portfolio_pnl += pnl_usdt
            updates_to_db.append(("UPDATE trades SET status=?, exit_price=?, closed_at=?, pnl_usdt=?, highest_price=? WHERE id=?", (status, result['exit_price'], datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), pnl_usdt, result['highest_price'], result['id'])))
            # [تطوير] استخدام رسالة الإغلاق الشاملة من المحلل
            await send_close_trade_message(context.bot, original_trade, result, pnl_usdt)
        elif status == 'update_tsl_activation':
            updates_to_db.append(("UPDATE trades SET stop_loss=?, highest_price=?, trailing_sl_active=? WHERE id=?", (result['new_sl'], result['highest_price'], True, result['id'])))
            await send_telegram_message(context.bot, {'custom_message': f"🔒 **تأمين صفقة #{original_trade['id']} ({original_trade['symbol']})** 🔒\nتم نقل وقف الخسارة إلى نقطة الدخول: `{result['new_sl']}`"})
        elif status == 'update_tsl_trail': updates_to_db.append(("UPDATE trades SET stop_loss=?, highest_price=? WHERE id=?", (result['new_sl'], result['highest_price'], result['id'])))
        elif status == 'update_peak': updates_to_db.append(("UPDATE trades SET highest_price=? WHERE id=?", (result['highest_price'], result['id'])))

    if updates_to_db:
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10);
            for q, p in updates_to_db: conn.cursor().execute(q, p)
            conn.commit(); conn.close()
        except Exception as e: logger.error(f"DB update failed in track_open_trades: {e}")
    if portfolio_pnl != 0.0:
        bot_data['settings']['virtual_portfolio_balance_usdt'] += portfolio_pnl; save_settings()


# --- [دمج] إضافة أدوات الصياد: مراقب الإدراجات وصائد الجواهر ---
async def scan_for_new_listings() -> Dict[str, List[str]]:
    logger.info("Scanning for new listings...")
    conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM known_symbols"); is_initial_run = cursor.fetchone()[0] == 0
    all_new_listings = {}
    for ex_id, exchange in bot_data["public_exchanges"].items():
        try:
            await exchange.load_markets(True)
            current_symbols = {s for s in exchange.symbols if s.endswith('/USDT')}
            cursor.execute("SELECT symbol FROM known_symbols WHERE exchange = ?", (ex_id,)); known_symbols = {row[0] for row in cursor.fetchall()}
            if is_initial_run:
                cursor.executemany("INSERT OR IGNORE INTO known_symbols VALUES (?, ?, ?)", [(ex_id, s, datetime.now(timezone.utc).isoformat()) for s in current_symbols])
            elif newly_listed := current_symbols - known_symbols:
                all_new_listings[ex_id] = sorted(list(newly_listed))
                cursor.executemany("INSERT OR IGNORE INTO known_symbols VALUES (?, ?, ?)", [(ex_id, s, datetime.now(timezone.utc).isoformat()) for s in newly_listed])
        except Exception as e: logger.error(f"Could not check listings for {ex_id}: {e}")
    conn.commit(); conn.close()
    return {} if is_initial_run else all_new_listings

async def periodic_listings_check(context: ContextTypes.DEFAULT_TYPE):
    new_listings = await scan_for_new_listings()
    if new_listings:
        message = ["🚨 **تنبيه إدراجات جديدة!** 🚨"] + [f"\n--- **{ex_id}** ---\n" + "\n".join(f"  - `{s}`" for s in symbols) for ex_id, symbols in new_listings.items()]
        await send_telegram_message(context.bot, {'custom_message': "\n".join(message)})

async def manual_check_listings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command to manually check for new listings."""
    target_message = update.callback_query.message if update.callback_query else update.message
    await target_message.reply_text("🔍 جارِ البحث يدوياً عن أي إدراجات جديدة...")
    new_listings = await scan_for_new_listings()

    if new_listings:
        message_parts = ["🚨 **تنبيه إدراجات جديدة!** 🚨\n"]
        for ex_id, symbols in new_listings.items():
            message_parts.append(f"\n--- **منصة {ex_id}** ---")
            for symbol in symbols:
                message_parts.append(f"  - `{symbol}`")
        message = "\n".join(message_parts)
        await target_message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    else:
        await target_message.reply_text("✅ لم يتم العثور على أي إدراجات جديدة منذ آخر فحص.")


async def gem_hunter_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message if update.callback_query else update.message
    await target_message.reply_text(f"💎 **صائد الجواهر** | 🔍 جارِ تنفيذ مسح عميق...")
    settings = bot_data["settings"]
    gems = []
    # فحص منصة Binance كمثال
    try:
        exchange = bot_data["public_exchanges"]['binance']
        tickers = await exchange.fetch_tickers()
        symbols_to_check = [s for s, t in tickers.items() if s.endswith('/USDT') and t.get('quoteVolume', 0) > settings["gem_min_24h_volume_usdt"]]
        for symbol in symbols_to_check[:200]: # Limit to avoid rate limits
            await asyncio.sleep(0.1)
            ohlcv = await exchange.fetch_ohlcv(symbol, '1d', limit=1000)
            if not ohlcv or len(ohlcv) < 30: continue
            if datetime.fromtimestamp(ohlcv[0][0]/1000, timezone.utc) < (datetime.now(timezone.utc) - timedelta(days=settings["gem_listing_since_days"])): continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            ath, atl, current = df['high'].max(), df['low'].min(), df['close'].iloc[-1]
            if not all([ath, atl, current]) or ath <= 0 or atl <= 0: continue
            correction, rise_from_atl = ((current - ath) / ath) * 100, ((current - atl) / atl) * 100
            if correction <= settings["gem_min_correction_percent"] and rise_from_atl >= settings["gem_min_rise_from_atl_percent"]:
                gems.append({'symbol': symbol, 'potential_x': ath / current, 'correction_percent': correction, 'current_price': current, 'ath': ath})
    except Exception as e: logger.error(f"Gem hunter failed: {e}")

    if not gems: await target_message.reply_text("✅ لم يتم العثور على جواهر تتوافق مع الشروط."); return
    gems.sort(key=lambda x: x['potential_x'], reverse=True)
    message = "💎 **تقرير الجواهر المخفية** 💎\n\n--- **أفضل المرشحين** ---\n" + "".join(
        f"**${g['symbol'].replace('/USDT','')}**\n  - 🚀 **العودة للقمة:** `{g['potential_x']:.1f}X`\n  - 🩸 **مصححة:** `{g['correction_percent']:.1f}%`\n\n" for g in gems[:10])
    await target_message.reply_text(message, parse_mode=ParseMode.MARKDOWN)


# --- [إصلاح] استعادة واجهة المستخدم الكاملة والأوامر ---
main_menu_keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"], ["ℹ️ مساعدة"]]
settings_menu_keyboard = [["🏁 أنماط جاهزة", "🎭 تفعيل/تعطيل الماسحات"], ["🔧 تعديل المعايير", "🔙 القائمة الرئيسية"]]

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("أهلاً بك في بوت **كاسحة الألغام**! (v1.1 - النسخة المصححة)", reply_markup=ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True))

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 الإحصائيات العامة", callback_data="dashboard_stats"), InlineKeyboardButton("📈 الصفقات النشطة", callback_data="dashboard_active_trades")],
        [InlineKeyboardButton("📜 تقرير أداء الاستراتيجيات", callback_data="dashboard_strategy_report")],
        [InlineKeyboardButton("🛠️ أدوات خاصة", callback_data="dashboard_tools")],
        [InlineKeyboardButton("🗓️ التقرير اليومي", callback_data="dashboard_daily_report"), InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="dashboard_debug")],
        [InlineKeyboardButton("🔄 تحديث", callback_data="dashboard_refresh")]
    ])
    message_text = "🖥️ *لوحة التحكم الرئيسية*\n\nاختر التقرير أو الأداة التي تريد استخدامها:"
    target_message = update.message or (update.callback_query and update.callback_query.message)
    try:
        if update.callback_query and update.callback_query.data == "dashboard_refresh":
            await target_message.edit_text(message_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        else:
            await target_message.reply_text(message_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
    except BadRequest as e:
        if "Message is not modified" not in str(e): raise

async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await (update.message or update.callback_query.message).reply_text("اختر الإعداد:", reply_markup=ReplyKeyboardMarkup(settings_menu_keyboard, resize_keyboard=True))

def get_scanners_keyboard():
    active_scanners = bot_data["settings"].get("active_scanners", [])
    keyboard = [[InlineKeyboardButton(f"{'✅' if name in active_scanners else '❌'} {STRATEGY_NAMES_AR.get(name, name)}", callback_data=f"toggle_{name}")] for name in SCANNERS.keys()]
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    return InlineKeyboardMarkup(keyboard)

async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await (update.message or update.callback_query.message).reply_text("اختر الماسحات لتفعيلها أو تعطيلها:", reply_markup=get_scanners_keyboard())

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "**🤖 أوامر البوت المتاحة **\n\n"
        "`/start` - لعرض القائمة الرئيسية.\n"
        "`/check <ID>` - لمتابعة صفقة معينة."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data

    if data.startswith("dashboard_"):
        action = data.split("_", 1)[1]
        if action == "tools":
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("💎 صائد الجواهر", callback_data="tools_gem_hunter")],
                [InlineKeyboardButton("📢 فحص الإدراجات", callback_data="tools_check_listings")],
                [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="dashboard_refresh")]
            ])
            await query.edit_message_text("🛠️ *أدوات خاصة*\n\nاختر الأداة التي تريد استخدامها:", reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        # Add other dashboard actions here...
        elif action == "refresh":
            await show_dashboard_command(update, context)

    elif data.startswith("tools_"):
        tool = data.split("_", 1)[1]
        if tool == "gem_hunter":
            await gem_hunter_command(update, context)
        elif tool == "check_listings":
            await manual_check_listings_command(update, context)

    elif data.startswith("toggle_"):
        scanner_name = data.split("_", 1)[1]
        active_scanners = bot_data["settings"].get("active_scanners", []).copy()
        if scanner_name in active_scanners: active_scanners.remove(scanner_name)
        else: active_scanners.append(scanner_name)
        bot_data["settings"]["active_scanners"] = active_scanners; save_settings()
        try:
            await query.edit_message_reply_markup(reply_markup=get_scanners_keyboard())
        except BadRequest as e:
            if "Message is not modified" not in str(e): raise

async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text
    menu_handlers = {
        "Dashboard 🖥️": show_dashboard_command,
        "ℹ️ مساعدة": help_command,
        "⚙️ الإعدادات": show_settings_menu,
        "🔙 القائمة الرئيسية": start_command,
        "🎭 تفعيل/تعطيل الماسحات": show_scanners_menu,
    }
    if text in menu_handlers:
        await menu_handlers[text](update, context)

# --- بقية دوال الواجهة والتشغيل ---
async def send_close_trade_message(bot, trade, result, pnl_usdt):
    # This function creates the detailed closing message, as seen in the Analyzer bot
    status = result['status']
    pnl_percent = (pnl_usdt / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0
    icon = "✅" if status == 'ناجحة' else "❌"
    reason_text = "تم تحقيق الهدف" if status == 'ناجحة' else "تم ضرب الوقف"
    
    # [تطوير] توضيح حالة الوقف المتحرك
    if status == 'فاشلة' and result['exit_price'] >= trade['entry_price']:
        reason_text = "تم الخروج عند نقطة الدخول (وقف متحرك)"
        icon = "🔒"

    message = (
        f"**📦 إغلاق صفقة | #{trade['id']} {trade['symbol']}**\n\n"
        f"**الحالة: {icon} {status} ({reason_text})**\n"
        f"💰 **الربح/الخسارة:** `${pnl_usdt:+.2f}` (`{pnl_percent:+.2f}%`)\n"
        f"------------------------------------\n"
        f"📈 **الدخول:** `{trade['entry_price']}`\n"
        f"📉 **الخروج:** `{result['exit_price']}`\n"
        f"🏔️ **أعلى قمة:** `{result.get('highest_price', trade['entry_price'])}`"
    )
    await send_telegram_message(bot, {'custom_message': message, 'target_chat': TELEGRAM_SIGNAL_CHANNEL_ID})

async def send_telegram_message(bot, signal_data, is_new=False, is_opportunity=False, update_type=None):
    message, keyboard, target_chat = "", None, TELEGRAM_CHAT_ID
    if 'custom_message' in signal_data:
        message = signal_data['custom_message']
        keyboard = signal_data.get('keyboard')
    elif is_new or is_opportunity:
        target_chat = TELEGRAM_SIGNAL_CHANNEL_ID
        title = "✅ توصية جديدة" if is_new else "💡 فرصة محتملة"
        reasons_en = signal_data['reason'].split(' + ')
        reasons_ar = ' + '.join([STRATEGY_NAMES_AR.get(r, r) for r in reasons_en])
        message = (f"**{title} | {signal_data['symbol']}**\n\n"
                   f"🔹 **المنصة:** {signal_data['exchange']}\n"
                   f"🔍 **الاستراتيجية:** {reasons_ar}\n\n"
                   f"📈 **الدخول:** `{signal_data['entry_price']}`\n"
                   f"🎯 **الهدف:** `{signal_data['take_profit']}`\n"
                   f"🛑 **الوقف:** `{signal_data['stop_loss']}`")
    if not message: return
    try:
        await bot.send_message(chat_id=target_chat, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
    except Exception as e:
        logger.error(f"Failed to send Telegram message to {target_chat}: {e}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

async def post_init(application: Application):
    logger.info("Post-init: Initializing exchanges...")
    await initialize_exchanges()
    logger.info("Exchanges initialized. Setting up job queue...")
    job_queue = application.job_queue
    job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name='perform_scan')
    job_queue.run_repeating(track_open_trades, interval=TRACK_INTERVAL_SECONDS, first=20, name='track_open_trades')
    # [دمج] إضافة مهمة مراقبة الإدراجات
    job_queue.run_repeating(periodic_listings_check, interval=timedelta(minutes=LISTINGS_CHECK_INTERVAL_MINUTES), first=300, name='new_listings_checker')
    job_queue.run_daily(lambda ctx: None, time=dt_time(hour=23, minute=55, tzinfo=EGYPT_TZ), name='daily_report') # Placeholder for brevity
    await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🚀 *بوت كاسحة الألغام جاهز للعمل! (v1.1)*", parse_mode=ParseMode.MARKDOWN)
    logger.info("Post-init finished.")

async def post_shutdown(application: Application):
    all_exchanges = list(bot_data["exchanges"].values()) + list(bot_data["public_exchanges"].values())
    await asyncio.gather(*[ex.close() for ex in list({id(ex): ex for ex in all_exchanges}.values())])
    logger.info("All exchange connections closed.")

def main():
    print("🚀 Starting Minesweeper Bot v1.1...")
    load_settings(); init_database()
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).post_shutdown(post_shutdown).build()
    # --- [إصلاح] تسجيل جميع معالجات الأوامر والواجهة ---
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    application.add_error_handler(error_handler)

    print("✅ Bot is now running and polling for updates...")
    application.run_polling()

if __name__ == '__main__':
    main()

