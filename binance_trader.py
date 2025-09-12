# -*- coding: utf-8 -*-
# =======================================================================================
# --- 💣 بوت كاسحة الألغام (Minesweeper Bot) v2.1 (Multi-Exchange & Final Fixes) 💣 ---
# =======================================================================================
# - [إصلاح جذري] إعادة كتابة دوال التقارير (لقطة المحفظة، المخاطر، المزامنة)
#   لدعم تعدد المنصات (Binance, KuCoin, etc.) بشكل كامل.
# - [إصلاح حرج] حل مشكلة انهيار التقارير عند استخدام الفلاتر (وهمي/حقيقي).
# - [إصلاح حرج] حل مشكلة انهيار "تقرير المخاطر" عند عدم وجود صفقات.
# - [تحسين] ضمان ظهور الصفقات الوهمية في جميع التقارير ذات الصلة.
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
DB_FILE = os.path.join(APP_ROOT, 'minesweeper_bot.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'minesweeper_settings.json')

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
        "real_trading_enabled", "real_trade_size_usdt", "virtual_trade_size_percentage",
        "atr_sl_multiplier", "risk_reward_ratio", "trailing_sl_activation_percent", "trailing_sl_callback_percent"
    ],
    "الفلاتر والاتجاه": [
        "market_regime_filter_enabled", "use_master_trend_filter", "fear_and_greed_filter_enabled",
        "master_adx_filter_level", "master_trend_filter_ma_period", "trailing_sl_enabled", "fear_and_greed_threshold",
        "fundamental_analysis_enabled"
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


# --- Global Bot State ---
bot_data = {
    "exchanges": {},
    "public_exchanges": {},
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
    "real_trading_enabled": False,
    "real_trade_size_usdt": 15.0,
    "virtual_portfolio_balance_usdt": 1000.0, "virtual_trade_size_percentage": 5.0, "max_concurrent_trades": 10, "top_n_symbols_by_volume": 250, "concurrent_workers": 10,
    "market_regime_filter_enabled": True, "fundamental_analysis_enabled": True,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "whale_radar", "sniper_pro"],
    "use_master_trend_filter": True, "master_trend_filter_ma_period": 50, "master_adx_filter_level": 22,
    "fear_and_greed_filter_enabled": True, "fear_and_greed_threshold": 30,
    "use_dynamic_risk_management": True, "atr_period": 14, "atr_sl_multiplier": 2.5, "risk_reward_ratio": 2.0,
    "trailing_sl_enabled": True, "trailing_sl_activation_percent": 1.5, "trailing_sl_callback_percent": 1.0,
    "momentum_breakout": {"vwap_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bbands_period": 20, "bbands_stddev": 2.0, "rsi_period": 14, "rsi_max_level": 68, "volume_spike_multiplier": 1.5},
    "breakout_squeeze_pro": {"bbands_period": 20, "bbands_stddev": 2.0, "keltner_period": 20, "keltner_atr_multiplier": 1.5, "volume_confirmation_enabled": True},
    "sniper_pro": {"compression_hours": 6, "max_volatility_percent": 12.0},
    "whale_radar": {"wall_threshold_usdt": 30000},
    "liquidity_filters": {"min_quote_volume_24h_usd": 1_000_000, "max_spread_percent": 0.5, "rvol_period": 20, "min_rvol": 1.5},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.8},
    "stablecoin_filter": {"exclude_bases": ["USDT","USDC","DAI","FDUSD","TUSD","USDE","PYUSD","GUSD","EURT","USDJ"]},
    "ema_trend_filter": {"enabled": True, "ema_period": 200},
    "min_tp_sl_filter": {"min_tp_percent": 1.0, "min_sl_percent": 0.5},
    "min_signal_strength": 1,
    "active_preset_name": "PRO",
    "last_market_mood": {"timestamp": "N/A", "mood": "UNKNOWN", "reason": "No scan performed yet."},
    "last_suggestion_time": 0
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
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

# --- Database Management ---
def migrate_database():
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
        cursor.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT)''')
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
            'نشطة', False, signal.get('verified_entry_price', signal['entry_price']),
            signal['reason'], 'real' if signal.get('is_real_trade') else 'virtual',
            signal.get('entry_order_id'), signal.get('exit_order_ids_json')
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
    if ALPHA_VANTAGE_API_KEY == 'YOUR_AV_KEY_HERE': return []
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    params = {'function': 'ECONOMIC_CALENDAR', 'horizon': '3month', 'apikey': ALPHA_VANTAGE_API_KEY}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('https://www.alphavantage.co/query', params=params, timeout=20)
            response.raise_for_status()
        data_str = response.text
        if "premium" in data_str.lower(): logger.error("Alpha Vantage API returned a premium feature error for Economic Calendar."); return []
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
    except httpx.RequestError as e: logger.error(f"Failed to fetch economic calendar data from Alpha Vantage: {e}"); return None

def get_latest_crypto_news(limit=15):
    urls, headlines = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"], []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            headlines.extend(entry.title for entry in feed.entries[:5])
        except Exception as e: logger.error(f"Failed to fetch news from {url}: {e}")
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
    sentiment_score = analyze_sentiment_of_headlines(get_latest_crypto_news())
    logger.info(f"Market sentiment score based on news: {sentiment_score:.2f}")
    if sentiment_score > 0.25: return "POSITIVE", sentiment_score, f"مشاعر إيجابية (الدرجة: {sentiment_score:.2f})"
    elif sentiment_score < -0.25: return "NEGATIVE", sentiment_score, f"مشاعر سلبية (الدرجة: {sentiment_score:.2f})"
    else: return "NEUTRAL", sentiment_score, f"مشاعر محايدة (الدرجة: {sentiment_score:.2f})"


# --- Advanced Scanners ---
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

def analyze_momentum_breakout(df, params, rvol, adx_value, exchange, symbol):
    df.ta.vwap(append=True); df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
    df.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'], append=True); df.ta.rsi(length=params['rsi_period'], append=True)
    macd_col, macds_col, bbu_col, rsi_col = (find_col(df.columns, f"MACD_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}"), find_col(df.columns, f"MACDs_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}"), find_col(df.columns, f"BBU_{params['bbands_period']}_"), find_col(df.columns, f"RSI_{params['rsi_period']}"))
    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and last[rsi_col] < params['rsi_max_level'] and rvol >= bot_data['settings']['liquidity_filters']['min_rvol']):
        return {"reason": "momentum_breakout", "type": "long"}
    return None

def analyze_breakout_squeeze_pro(df, params, rvol, adx_value, exchange, symbol):
    df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True); df.ta.kc(length=params['keltner_period'], scalar=params['keltner_atr_multiplier'], append=True); df.ta.obv(append=True)
    bbu_col, bbl_col, kcu_col, kcl_col = (find_col(df.columns, f"BBU_{params['bbands_period']}_"), find_col(df.columns, f"BBL_{params['bbands_period']}_"), find_col(df.columns, f"KCUe_{params['keltner_period']}_"), find_col(df.columns, f"KCLEe_{params['keltner_period']}_"))
    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]:
        if last['close'] > last[bbu_col] and rvol >= bot_data['settings']['liquidity_filters']['min_rvol'] and df['OBV'].iloc[-2] > df['OBV'].iloc[-3]:
            if params.get('volume_confirmation_enabled', True) and not (last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5): return None
            return {"reason": "breakout_squeeze_pro", "type": "long"}
    return None

def find_support_resistance(high_prices, low_prices, window=10):
    supports, resistances = [], []
    if len(high_prices) < (2 * window + 1): return [], []
    for i in range(window, len(high_prices) - window):
        if high_prices[i] == max(high_prices[i-window:i+window+1]): resistances.append(high_prices[i])
        if low_prices[i] == min(low_prices[i-window:i+window+1]): supports.append(low_prices[i])
    if not supports and not resistances: return [], []
    def cluster_levels(levels, tolerance_percent=0.5):
        if not levels: return []
        # [FIX] Corrected SyntaxError by separating the sort operation from the assignment
        levels.sort()
        clustered = []
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
        if len(df) < compression_candles + 2: return None
        compression_df = df.iloc[-compression_candles-1:-1]
        highest_high, lowest_low = compression_df['high'].max(), compression_df['low'].min()
        volatility = (highest_high - lowest_low) / lowest_low * 100 if lowest_low > 0 else float('inf')
        if volatility < params.get("max_volatility_percent", 12.0):
            last_candle = df.iloc[-2]
            if last_candle['close'] > highest_high and last_candle['volume'] > compression_df['volume'].mean() * 2:
                return {"reason": "sniper_pro", "type": "long"}
    except Exception as e: logger.warning(f"Sniper Pro scan failed for {symbol}: {e}")
    return None

async def analyze_whale_radar(df, params, rvol, adx_value, exchange, symbol):
    try:
        ob = await exchange.fetch_order_book(symbol, limit=20)
        if not ob or not ob.get('bids'): return None
        if sum(float(p) * float(q) for p, q in ob.get('bids', [])[:10]) > params.get("wall_threshold_usdt", 30000):
            return {"reason": "whale_radar", "type": "long"}
    except Exception as e: logger.warning(f"Whale Radar scan failed for {symbol}: {e}")
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
    except Exception as e: logger.warning(f"Support Rebound scan failed for {symbol}: {e}")
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
        params, authenticated = {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}, False
        if ex_id == 'binance' and BINANCE_API_KEY != 'YOUR_BINANCE_API_KEY':
            params.update({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET}); authenticated = True
        if ex_id == 'kucoin' and KUCOIN_API_KEY != 'YOUR_KUCOIN_API_KEY':
            params.update({'apiKey': KUCOIN_API_KEY, 'secret': KUCOIN_API_SECRET, 'password': KUCOIN_API_PASSPHRASE}); authenticated = True
        if authenticated:
            try:
                private_exchange = getattr(ccxt_async, ex_id)(params)
                bot_data["exchanges"][ex_id] = private_exchange
                logger.info(f"Connected to {ex_id} with PRIVATE (authenticated) client.")
            except Exception as e:
                logger.error(f"Failed to connect PRIVATE client for {ex_id}: {e}")
                if 'private_exchange' in locals(): await private_exchange.close()
        elif ex_id in bot_data["public_exchanges"]: bot_data["exchanges"][ex_id] = bot_data["public_exchanges"][ex_id]
    await asyncio.gather(*[connect(ex_id) for ex_id in EXCHANGES_TO_SCAN])


async def aggregate_top_movers():
    all_tickers = []
    async def fetch(ex_id, ex):
        try: return [dict(t, exchange=ex_id) for t in (await ex.fetch_tickers()).values()]
        except Exception: return []
    results = await asyncio.gather(*[fetch(ex_id, ex) for ex_id, ex in bot_data["public_exchanges"].items()])
    for res in results: all_tickers.extend(res)
    settings = bot_data['settings']
    excluded_bases, min_volume = settings['stablecoin_filter']['exclude_bases'], settings['liquidity_filters']['min_quote_volume_24h_usd']
    usdt_tickers = [t for t in all_tickers if t.get('symbol') and t['symbol'].upper().endswith('/USDT') and t['symbol'].split('/')[0] not in excluded_bases and t.get('quoteVolume', 0) >= min_volume and not any(k in t['symbol'].upper() for k in ['UP','DOWN','3L','3S','BEAR','BULL'])]
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
        is_bullish = df_htf.iloc[-1]['close'] > df_htf.iloc[-1][f'SMA_{ma_period}']
        return is_bullish, "Bullish" if is_bullish else "Bearish"
    except Exception as e:
        return None, f"Error: {e}"

async def worker(queue, results_list, settings, failure_counter):
    while not queue.empty():
        market_info = await queue.get()
        symbol, exchange = market_info.get('symbol', 'N/A'), bot_data["public_exchanges"].get(market_info['exchange'])
        if not exchange or not settings.get('active_scanners'): queue.task_done(); continue
        try:
            liq_filters, vol_filters, ema_filters = settings['liquidity_filters'], settings['volatility_filters'], settings['ema_trend_filter']
            orderbook = await exchange.fetch_order_book(symbol, limit=20)
            if not orderbook or not orderbook['bids'] or not orderbook['asks']: logger.debug(f"Reject {symbol}: Could not fetch order book."); continue
            best_bid, best_ask = orderbook['bids'][0][0], orderbook['asks'][0][0]
            if best_bid <= 0: logger.debug(f"Reject {symbol}: Invalid bid price."); continue
            if ((best_ask - best_bid) / best_bid) * 100 > liq_filters['max_spread_percent']: logger.debug(f"Reject {symbol}: High Spread"); continue
            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            if len(ohlcv) < ema_filters['ema_period']: logger.debug(f"Skipping {symbol}: Not enough data."); continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
            df['volume_sma'] = ta.sma(df['volume'], length=liq_filters['rvol_period'])
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] <= 0: logger.debug(f"Skipping {symbol}: Invalid SMA volume."); continue
            if df['volume'].iloc[-2] / df['volume_sma'].iloc[-2] < liq_filters['min_rvol']: logger.debug(f"Reject {symbol}: Low RVOL"); continue
            df.ta.atr(length=vol_filters['atr_period_for_filter'], append=True)
            last_close = df['close'].iloc[-2]
            if last_close <= 0: logger.debug(f"Skipping {symbol}: Invalid close price."); continue
            if (df[f"ATRr_{vol_filters['atr_period_for_filter']}"].iloc[-2] / last_close) * 100 < vol_filters['min_atr_percent']: logger.debug(f"Reject {symbol}: Low ATR%"); continue
            ema_col_name = f"EMA_{ema_filters['ema_period']}"; df.ta.ema(length=ema_filters['ema_period'], append=True)
            if ema_col_name not in df.columns or pd.isna(df[ema_col_name].iloc[-2]): logger.debug(f"Skipping {symbol}: EMA not calculated."); continue
            if ema_filters['enabled'] and last_close < df[ema_col_name].iloc[-2]: logger.debug(f"Reject {symbol}: Below EMA"); continue
            if settings.get('use_master_trend_filter'):
                is_htf_bullish, reason = await get_higher_timeframe_trend(exchange, symbol, settings['master_trend_filter_ma_period'])
                if not is_htf_bullish: logger.debug(f"HTF Trend Filter FAILED for {symbol}: {reason}"); continue
            df.ta.adx(append=True)
            adx_col = find_col(df.columns, 'ADX_'); adx_value = df[adx_col].iloc[-2] if adx_col and pd.notna(df[adx_col].iloc[-2]) else 0
            if settings.get('use_master_trend_filter') and adx_value < settings['master_adx_filter_level']: logger.debug(f"ADX Filter FAILED for {symbol}"); continue
            confirmed_reasons = []
            for scanner_name in settings['active_scanners']:
                if scanner_func := SCANNERS.get(scanner_name):
                    result = await scanner_func(df.copy(), settings.get(scanner_name, {}), 0, adx_value, exchange, symbol) if asyncio.iscoroutinefunction(scanner_func) else scanner_func(df.copy(), settings.get(scanner_name, {}), 0, adx_value, exchange, symbol)
                    if result and result.get("type") == "long": confirmed_reasons.append(result['reason'])
            if confirmed_reasons and len(confirmed_reasons) >= settings.get("min_signal_strength", 1):
                reason_str, entry_price = ' + '.join(confirmed_reasons), df.iloc[-2]['close']
                df.ta.atr(length=settings['atr_period'], append=True)
                current_atr = df.iloc[-2].get(find_col(df.columns, f"ATRr_{settings['atr_period']}"), 0)
                if settings.get("use_dynamic_risk_management", False) and current_atr > 0:
                    risk_per_unit = current_atr * settings['atr_sl_multiplier']
                    stop_loss, take_profit = entry_price - risk_per_unit, entry_price + (risk_per_unit * settings['risk_reward_ratio'])
                else:
                    sl_percent, tp_percent = settings.get("stop_loss_percentage", 2.0), settings.get("take_profit_percentage", 4.0)
                    stop_loss, take_profit = entry_price * (1 - sl_percent / 100), entry_price * (1 + tp_percent / 100)
                tp_p, sl_p = ((take_profit - entry_price) / entry_price * 100), ((entry_price - stop_loss) / entry_price * 100)
                min_filters = settings['min_tp_sl_filter']
                if tp_p >= min_filters['min_tp_percent'] and sl_p >= min_filters['min_sl_percent']:
                    results_list.append({"symbol": symbol, "exchange": market_info['exchange'].capitalize(), "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "timestamp": df.index[-2], "reason": reason_str, "strength": len(confirmed_reasons)})
                else: logger.debug(f"Reject {symbol} Signal: Small TP/SL")
        except ccxt.RateLimitExceeded as e: logger.warning(f"Rate limit exceeded for {symbol}: {e}"); await asyncio.sleep(10)
        except ccxt.NetworkError as e: logger.warning(f"Network error for {symbol}: {e}")
        except Exception as e: logger.error(f"CRITICAL ERROR in worker for {symbol}: {e}", exc_info=True); failure_counter[0] += 1
        finally: queue.task_done()

async def get_real_balance(exchange_id, currency='USDT'):
    try:
        exchange = bot_data["exchanges"].get(exchange_id.lower())
        if not exchange or not exchange.apiKey: return 0.0
        return (await exchange.fetch_balance())['free'].get(currency, 0.0)
    except Exception as e: logger.error(f"Error fetching {exchange_id.capitalize()} balance for {currency}: {e}"); return 0.0

async def place_real_trade(signal):
    exchange_id, exchange, settings, symbol = signal['exchange'].lower(), bot_data["exchanges"].get(signal['exchange'].lower()), bot_data['settings'], signal['symbol']
    if not exchange or not exchange.apiKey: return {'success': False, 'data': f"Client not authenticated for {exchange_id.capitalize()}."}
    try:
        usdt_balance, trade_amount_usdt = await get_real_balance(exchange_id, 'USDT'), settings.get("real_trade_size_usdt", 15.0)
        if usdt_balance < trade_amount_usdt: return {'success': False, 'data': f"رصيدك ${usdt_balance:.2f} غير كافٍ لفتح صفقة بـ ${trade_amount_usdt}."}
        if not (await exchange.load_markets()).get(symbol): return {'success': False, 'data': f"Could not find market info for {symbol}."}
        formatted_quantity = exchange.amount_to_precision(symbol, trade_amount_usdt / signal['entry_price'])
    except Exception as e: return {'success': False, 'data': f"Pre-flight check failed: {e}"}
    try:
        buy_order = await exchange.create_market_buy_order(symbol, float(formatted_quantity))
    except Exception as e: return {'success': False, 'data': f"حدث خطأ عند محاولة الشراء: `{str(e)}`"}
    try:
        await asyncio.sleep(2)
        verified_order = await exchange.fetch_order(buy_order['id'], symbol)
        if verified_order and verified_order.get('status') == 'closed' and verified_order.get('filled', 0) > 0:
            verified_price, verified_quantity = verified_order.get('average', signal['entry_price']), verified_order.get('filled')
            verified_cost = verified_order.get('cost', verified_price * verified_quantity)
        else: raise Exception(f"Order not confirmed as filled. Status: {verified_order.get('status')}")
    except Exception as e: return {'success': False, 'manual_check_required': True, 'data': f"تم إرسال أمر الشراء لكن فشل التحقق منه. **يرجى التحقق يدوياً!** Order ID: `{buy_order.get('id', 'N/A')}`. Error: `{e}`"}
    try:
        tp_price, sl_price = exchange.price_to_precision(symbol, signal['take_profit']), exchange.price_to_precision(symbol, signal['stop_loss'])
        if exchange_id == 'binance':
            oco_order = await exchange.create_order(symbol, 'oco', 'sell', verified_quantity, price=tp_price, stopPrice=exchange.price_to_precision(symbol, signal['stop_loss'] * 1.001), params={'stopLimitPrice': sl_price})
            exit_order_ids = {"oco_id": oco_order['id']}
        elif exchange_id == 'kucoin':
            tp_order = await exchange.create_limit_sell_order(symbol, verified_quantity, float(tp_price))
            sl_order = await exchange.create_order(symbol, 'stop_limit', 'sell', verified_quantity, float(sl_price), params={'stopPrice': float(exchange.price_to_precision(symbol, signal['stop_loss'] * 1.002))})
            exit_order_ids = {"tp_id": tp_order['id'], "sl_id": sl_order['id']}
        else: raise NotImplementedError(f"Exit logic not implemented for {exchange_id.capitalize()}.")
    except Exception as e: return {'success': True, 'exit_orders_failed': True, 'data': f"تم شراء {symbol} بنجاح، **لكن فشل وضع أوامر الخروج**. يرجى وضعها يدوياً!"}
    return {'success': True, 'data': {"entry_order_id": buy_order['id'], "exit_order_ids_json": json.dumps(exit_order_ids), "verified_quantity": verified_quantity, "verified_entry_price": verified_price, "verified_entry_value": verified_cost}}


async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        if bot_data['status_snapshot']['scan_in_progress']: return
        settings = bot_data["settings"]
        if settings.get('fundamental_analysis_enabled', True):
            mood, _, mood_reason = await get_fundamental_market_mood()
            bot_data['settings']['last_market_mood'] = {"timestamp": datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M'), "mood": mood, "reason": mood_reason}; save_settings()
            if mood in ["NEGATIVE", "DANGEROUS"]: await send_telegram_message(context.bot, {'custom_message': f"**⚠️ تم إيقاف الفحص مؤقتاً**\n\n**السبب:** مزاج السوق سلبي/خطر.\n**التفاصيل:** {mood_reason}.", 'target_chat': TELEGRAM_CHAT_ID}); return
        is_market_ok, btc_reason = await check_market_regime()
        bot_data['status_snapshot']['btc_market_mood'] = "إيجابي ✅" if is_market_ok else "سلبي ❌"
        if settings.get('market_regime_filter_enabled', True) and not is_market_ok: await send_telegram_message(context.bot, {'custom_message': f"**⚠️ تم إيقاف الفحص مؤقتاً**\n\n**السبب:** مزاج السوق سلبي/خطر.\n**التفاصيل:** {btc_reason}.", 'target_chat': TELEGRAM_CHAT_ID}); return
        status = bot_data['status_snapshot']
        status.update({"scan_in_progress": True, "last_scan_start_time": datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), "signals_found": 0})
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor()
            active_trades_count = cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة'").fetchone()[0]
            conn.close()
        except Exception as e: logger.error(f"DB Error in perform_scan: {e}"); active_trades_count = settings.get("max_concurrent_trades", 10)
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
            exchange_is_tradeable = signal['exchange'].lower() in bot_data["exchanges"] and bot_data["exchanges"][signal['exchange'].lower()].apiKey
            attempt_real_trade = settings.get("real_trading_enabled", False) and exchange_is_tradeable
            signal['is_real_trade'] = attempt_real_trade
            if attempt_real_trade:
                trade_result = await place_real_trade(signal)
                if trade_result['success']:
                    signal.update(trade_result['data'])
                    if log_recommendation_to_db(signal): await send_telegram_message(context.bot, signal, is_new=True); new_trades += 1
                    else: await send_telegram_message(context.bot, {'custom_message': f"**⚠️ خطأ حرج:** تم تنفيذ صفقة `{signal['symbol']}` لكن فشل تسجيلها."})
                else: await send_telegram_message(context.bot, {'custom_message': f"**❌ فشل تنفيذ صفقة `{signal['symbol']}`**\n\n**السبب:** {trade_result['data']}"})
            else:
                if active_trades_count < settings.get("max_concurrent_trades", 10):
                    trade_amount_usdt = settings["virtual_portfolio_balance_usdt"] * (settings["virtual_trade_size_percentage"] / 100)
                    signal.update({'quantity': trade_amount_usdt / signal['entry_price'], 'entry_value_usdt': trade_amount_usdt})
                    if trade_id := log_recommendation_to_db(signal):
                        signal['trade_id'] = trade_id; await send_telegram_message(context.bot, signal, is_new=True); new_trades += 1; active_trades_count += 1
                else: await send_telegram_message(context.bot, signal, is_opportunity=True); opportunities += 1
            await asyncio.sleep(0.5)
            bot_data['last_signal_time'][signal['symbol']] = time.time()
        failures = failure_counter[0]
        scan_duration = (datetime.now(EGYPT_TZ) - datetime.strptime(status['last_scan_start_time'], '%Y-%m-%d %H:%M:%S')).total_seconds()
        summary = (f"**🔬 ملخص الفحص**\n- المدة: {scan_duration:.0f} ثانية\n- العملات: {len(top_markets)}\n- إشارات: {len(signals)}\n- صفقات جديدة: {new_trades}\n- فرص: {opportunities}\n- أخطاء: {failures}")
        await send_telegram_message(context.bot, {'custom_message': summary, 'target_chat': TELEGRAM_CHAT_ID})
        status['signals_found'] = new_trades + opportunities; status['last_scan_end_time'] = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'); status['scan_in_progress'] = False
        bot_data['scan_history'].append({'signals': len(signals), 'failures': failures})
        await analyze_performance_and_suggest(context)

async def send_telegram_message(bot, signal_data, is_new=False, is_opportunity=False, update_type=None):
    message, keyboard, target_chat = "", None, TELEGRAM_CHAT_ID
    def format_price(price): return f"{price:,.8f}" if price < 0.01 else f"{price:,.4f}"
    if 'custom_message' in signal_data:
        message, target_chat = signal_data['custom_message'], signal_data.get('target_chat', TELEGRAM_CHAT_ID)
        if 'keyboard' in signal_data: keyboard = signal_data['keyboard']
    elif is_new or is_opportunity:
        target_chat = TELEGRAM_SIGNAL_CHANNEL_ID
        title = f"**{'🚨 صفقة حقيقية' if signal_data.get('is_real_trade') else '✅ توصية جديدة'} | {signal_data['symbol']}**" if is_new else f"**💡 فرصة محتملة | {signal_data['symbol']}**"
        entry, tp, sl = signal_data['entry_price'], signal_data['take_profit'], signal_data['stop_loss']
        reasons_ar = ' + '.join([STRATEGY_NAMES_AR.get(r, r) for r in signal_data['reason'].split(' + ')])
        message = (f"{title}\n------------------------------------\n"
                   f"🔹 **المنصة:** {signal_data['exchange']}\n⭐ **القوة:** {'⭐' * signal_data.get('strength', 1)}\n🔍 **الاستراتيجية:** {reasons_ar}\n\n"
                   f"📈 **الدخول:** `{format_price(entry)}`\n🎯 **الهدف:** `{format_price(tp)}` (+{((tp - entry) / entry * 100):.2f}%)\n"
                   f"🛑 **الوقف:** `{format_price(sl)}` (-{((entry - sl) / entry * 100):.2f}%)"
                   f"\n*للمتابعة: /check {signal_data.get('trade_id', 'N/A')}*" if is_new else "")
    elif update_type == 'tsl_activation': message = f"**🚀 تأمين الأرباح! | #{signal_data['id']} {signal_data['symbol']}**\n\nتم رفع الوقف إلى نقطة الدخول. الصفقة الآن بدون مخاطرة!"
    elif update_type == 'tsl_update_real': message = f"**🔔 تحديث وقف (حقيقي) | #{signal_data['id']} {signal_data['symbol']}**\n\nالسعر وصل `{signal_data['current_price']}`. يُقترح تعديل الوقف يدوياً إلى `{signal_data['new_sl']}`."
    if not message: return
    try: await bot.send_message(chat_id=target_chat, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
    except Exception as e: logger.error(f"Failed to send Telegram message to {target_chat}: {e}")

async def track_open_trades(context: ContextTypes.DEFAULT_TYPE):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row
        active_trades = [dict(row) for row in conn.cursor().execute("SELECT * FROM trades WHERE status = 'نشطة'").fetchall()]; conn.close()
    except Exception as e: logger.error(f"DB error in track_open_trades: {e}"); return
    bot_data['status_snapshot']['active_trades_count'] = len(active_trades)
    for trade in active_trades:
        exchange = bot_data["public_exchanges"].get(trade['exchange'].lower())
        if not exchange: continue
        try:
            ticker = await exchange.fetch_ticker(trade['symbol'])
            current_price = ticker.get('last') or ticker.get('close')
            if not current_price: continue
            if trade.get('take_profit') is not None and current_price >= trade['take_profit']: await close_trade_in_db(context, trade, current_price, 'ناجحة'); continue
            if trade.get('stop_loss') is not None and current_price <= trade['stop_loss']: await close_trade_in_db(context, trade, current_price, 'فاشلة'); continue
            settings = bot_data["settings"]
            if settings.get('trailing_sl_enabled', True):
                highest_price = max(trade.get('highest_price', current_price), current_price)
                if not trade.get('trailing_sl_active'):
                    if current_price >= trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100):
                        if trade['entry_price'] > trade.get('stop_loss', 0):
                            if trade.get('trade_mode') == 'real':
                                await send_telegram_message(context.bot, {**trade, "new_sl": trade['entry_price'], "current_price": current_price}, update_type='tsl_update_real')
                                await update_trade_sl_in_db(context, trade, trade['entry_price'], highest_price, is_activation=True, silent=True)
                            else: await update_trade_sl_in_db(context, trade, trade['entry_price'], highest_price, is_activation=True)
                elif trade.get('trailing_sl_active'):
                    new_sl = highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                    if new_sl > trade.get('stop_loss', 0):
                        if trade.get('trade_mode') == 'real':
                            await send_telegram_message(context.bot, {**trade, "new_sl": new_sl, "current_price": current_price}, update_type='tsl_update_real')
                            await update_trade_sl_in_db(context, trade, new_sl, highest_price, silent=True)
                        else: await update_trade_sl_in_db(context, trade, new_sl, highest_price)
                if highest_price > trade.get('highest_price', 0): await update_trade_peak_price_in_db(trade['id'], highest_price)
        except Exception as e: logger.error(f"Error tracking trade #{trade['id']} ({trade['symbol']}): {e}", exc_info=True)

async def close_trade_in_db(context: ContextTypes.DEFAULT_TYPE, trade: dict, exit_price: float, status: str):
    pnl_usdt = (exit_price - trade['entry_price']) * trade['quantity']
    if trade.get('trade_mode') == 'virtual': bot_data['settings']['virtual_portfolio_balance_usdt'] += pnl_usdt; save_settings()
    closed_at_str = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')
    duration = datetime.now(EGYPT_TZ) - EGYPT_TZ.localize(datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S'))
    days, rem = divmod(duration.total_seconds(), 86400); hours, rem = divmod(rem, 3600); minutes, _ = divmod(rem, 60)
    duration_str = f"{int(days)}d {int(hours)}h" if days > 0 else f"{int(hours)}h {int(minutes)}m"
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.cursor().execute("UPDATE trades SET status=?, exit_price=?, closed_at=?, exit_value_usdt=?, pnl_usdt=? WHERE id=?", (status, exit_price, closed_at_str, exit_price * trade['quantity'], pnl_usdt, trade['id']))
        conn.commit(); conn.close()
    except Exception as e: logger.error(f"DB update failed while closing trade #{trade['id']}: {e}"); return
    pnl_percent = (pnl_usdt / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0
    message = (f"**📦 إغلاق صفقة {'(حقيقية)' if trade.get('trade_mode') == 'real' else ''} | #{trade['id']} {trade['symbol']}**\n\n"
               f"**الحالة: {'✅ ناجحة' if status == 'ناجحة' else '❌ فاشلة'}**\n"
               f"💰 **الربح/الخسارة:** `${pnl_usdt:+.2f}` (`{pnl_percent:+.2f}%`)\n- **المدة:** {duration_str}")
    await send_telegram_message(context.bot, {'custom_message': message, 'target_chat': TELEGRAM_SIGNAL_CHANNEL_ID})

async def update_trade_sl_in_db(context: ContextTypes.DEFAULT_TYPE, trade: dict, new_sl: float, highest_price: float, is_activation: bool = False, silent: bool = False):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.cursor().execute("UPDATE trades SET stop_loss=?, highest_price=?, trailing_sl_active=? WHERE id=?", (new_sl, highest_price, True, trade['id']))
        conn.commit(); conn.close()
        if not silent and is_activation: await send_telegram_message(context.bot, trade, update_type='tsl_activation')
    except Exception as e: logger.error(f"Failed to update SL for trade #{trade['id']} in DB: {e}")

async def update_trade_peak_price_in_db(trade_id: int, highest_price: float):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.cursor().execute("UPDATE trades SET highest_price=? WHERE id=?", (highest_price, trade_id))
        conn.commit(); conn.close()
    except Exception as e: logger.error(f"Failed to update peak price for trade #{trade_id} in DB: {e}")

async def get_fear_and_greed_index():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            if data := r.json().get('data', []): return int(data[0]['value'])
    except Exception as e: logger.error(f"Could not fetch F&G Index: {e}")
    return None

async def check_market_regime():
    settings = bot_data['settings']
    try:
        if binance := bot_data["public_exchanges"].get('binance'):
            ohlcv = await binance.fetch_ohlcv('BTC/USDT', '4h', limit=55)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if df['close'].iloc[-1] <= ta.sma(df['close'], length=50).iloc[-1]: return False, "اتجاه BTC هابط (تحت متوسط 50 على 4 ساعات)."
    except Exception as e: logger.error(f"Error checking BTC trend: {e}")
    if settings.get("fear_and_greed_filter_enabled", True):
        if (fng_value := await get_fear_and_greed_index()) is not None:
            if fng_value < settings.get("fear_and_greed_threshold", 30): return False, f"مشاعر خوف شديد (F&G: {fng_value})."
    return True, "وضع السوق مناسب."

async def analyze_performance_and_suggest(context: ContextTypes.DEFAULT_TYPE):
    settings, history = bot_data['settings'], bot_data['scan_history']
    if len(history) < 5 or (time.time() - settings.get('last_suggestion_time', 0)) < 7200: return
    avg_signals, current_preset = sum(item['signals'] for item in history) / len(history), settings.get('active_preset_name', 'PRO')
    suggestion, market_desc, reason = None, None, None
    if avg_signals < 0.5 and current_preset == "STRICT": suggestion, market_desc, reason = "PRO", "السوق بطيء.", "نمط 'PRO' أكثر توازناً."
    elif avg_signals < 1 and current_preset == "PRO": suggestion, market_desc, reason = "LAX", "الفرص منخفضة.", "نمط 'LAX' سيوسع البحث."
    elif avg_signals > 8 and current_preset in ["LAX", "VERY_LAX"]: suggestion, market_desc, reason = "PRO", "السوق نشط جداً.", "نمط 'PRO' سيركز على الجودة."
    elif avg_signals > 12 and current_preset == "PRO": suggestion, market_desc, reason = "STRICT", "السوق متقلب جداً.", "نمط 'STRICT' سيصطاد أفضل الفرص فقط."
    if suggestion and suggestion != current_preset:
        message = f"**💡 اقتراح ذكي**\n\n- **الملاحظة:** {market_desc}\n- **الاقتراح:** تغيير النمط من `{current_preset}` إلى **`{suggestion}`**.\n- **السبب:** {reason}"
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("✅ تطبيق", callback_data=f"suggest_accept_{suggestion}")], [InlineKeyboardButton("❌ تجاهل", callback_data="suggest_decline")]])
        await send_telegram_message(context.bot, {'custom_message': message, 'keyboard': keyboard})
        bot_data['settings']['last_suggestion_time'] = time.time(); save_settings()

# --- Reports & Commands ---
async def generate_stats_report_string(trade_mode_filter='all'):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        query, params = "SELECT status, COUNT(*), SUM(pnl_usdt) FROM trades", []
        if trade_mode_filter != 'all': query += " WHERE trade_mode = ?"; params.append(trade_mode_filter)
        query += " GROUP BY status"
        stats_data = conn.cursor().execute(query, params).fetchall(); conn.close()
        counts, pnl = {s: c for s, c, p in stats_data}, {s: (p or 0) for s, c, p in stats_data}
        successful, failed = counts.get('ناجحة', 0), counts.get('فاشلة', 0)
        closed = successful + failed
        win_rate = (successful / closed * 100) if closed > 0 else 0
        total_pnl = sum(pnl.values())
        mode_title = {'all': '(الكل)', 'real': '(حقيقي فقط)', 'virtual': '(وهمي فقط)'}.get(trade_mode_filter, '')
        return (f"*📊 إحصائيات المحفظة {mode_title}*\n\n"
                f"📈 *الرصيد الافتراضي:* `${bot_data['settings']['virtual_portfolio_balance_usdt']:.2f}`\n"
                f"💰 *إجمالي الربح/الخسارة:* `${total_pnl:+.2f}`\n\n"
                f"- *إجمالي الصفقات:* `{sum(counts.values())}` (`{counts.get('نشطة', 0)}` نشطة)\n"
                f"- *الناجحة:* `{successful}` | *الربح:* `${pnl.get('ناجحة', 0):.2f}`\n"
                f"- *الفاشلة:* `{failed}` | *الخسارة:* `${abs(pnl.get('فاشلة', 0)):.2f}`\n"
                f"- *معدل النجاح:* `{win_rate:.2f}%`")
    except Exception as e: return f"❌ خطأ في جلب الإحصائيات: {e}"

async def generate_strategy_report_string(trade_mode_filter='all'):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        query, params = "SELECT reason, status FROM trades WHERE status IN ('ناجحة', 'فاشلة') AND timestamp >= ?", [start_date]
        if trade_mode_filter != 'all': query += " AND trade_mode = ?"; params.append(trade_mode_filter)
        trades = conn.cursor().execute(query, params).fetchall(); conn.close()
        if not trades: return f"ℹ️ لا توجد صفقات مغلقة في آخر 30 يومًا."
        stats = defaultdict(lambda: {'total': 0, 'successful': 0})
        for trade in trades:
            for reason in trade['reason'].split(' + '):
                stats[reason]['total'] += 1
                if trade['status'] == 'ناجحة': stats[reason]['successful'] += 1
        lines = ["📊 **أداء الاستراتيجيات (آخر 30 يومًا)**"]
        for reason_en, s in sorted(stats.items(), key=lambda item: item[1]['total'], reverse=True):
            if total := s['total']:
                win_rate = (s['successful'] / total) * 100
                lines.append(f"\n--- **{STRATEGY_NAMES_AR.get(reason_en, reason_en)}** ---\n- **التوصيات:** {total}\n- **النجاح:** {win_rate:.1f}%")
        return "\n".join(lines)
    except Exception as e: return f"❌ خطأ: {e}"

async def generate_active_trades_string_and_keyboard(trade_mode_filter='all'):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row
        query, params = "SELECT id, symbol, entry_value_usdt, exchange FROM trades WHERE status = 'نشطة'", []
        if trade_mode_filter != 'all': query += " AND trade_mode = ?"; params.append(trade_mode_filter)
        query += " ORDER BY id DESC"
        active_trades = conn.cursor().execute(query, params).fetchall(); conn.close()
        if not active_trades: return "✅ لا توجد صفقات نشطة حالياً لهذا الفلتر.", None
        keyboard = [[InlineKeyboardButton(f"#{t['id']} | {t['symbol']} | ${t['entry_value_usdt']:.2f}", callback_data=f"check_{t['id']}")] for t in active_trades]
        return "اختر صفقة لمتابعتها:", InlineKeyboardMarkup(keyboard)
    except Exception as e: return f"❌ خطأ في جلب الصفقات: {e}", None


main_menu_keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"], ["ℹ️ مساعدة"]]
settings_menu_keyboard = [["🏁 أنماط جاهزة", "🎭 تفعيل/تعطيل الماسحات"], ["🔧 تعديل المعايير", "🔙 القائمة الرئيسية"]]

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = "💣 أهلاً بك في بوت **كاسحة الألغام**!\n\n*(الإصدار 2.1 - إصلاحات شاملة)*\n\nاختر من القائمة للبدء."
    await update.message.reply_text(welcome_message, reply_markup=ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True), parse_mode=ParseMode.MARKDOWN)

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 الإحصائيات", callback_data="dashboard_stats"), InlineKeyboardButton("📈 الصفقات النشطة", callback_data="dashboard_active_trades")],
        [InlineKeyboardButton("📜 أداء الاستراتيجيات", callback_data="dashboard_strategy_report")],
        [InlineKeyboardButton("📸 لقطة للمحفظة", callback_data="dashboard_snapshot"), InlineKeyboardButton("ρίск تقرير المخاطر", callback_data="dashboard_risk")],
        [InlineKeyboardButton("🔄 مزامنة المحفظة", callback_data="dashboard_sync")],
        [InlineKeyboardButton("🛠️ أدوات التداول", callback_data="dashboard_tools"), InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="dashboard_debug")],
    ])
    message_text = "🖥️ *لوحة التحكم الرئيسية*\n\nاختر التقرير الذي تريد عرضه:"
    target_message = update.message or update.callback_query.message
    try:
        if update.callback_query: await target_message.edit_text(message_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        else: await target_message.reply_text(message_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
    except BadRequest as e:
        if "Message is not modified" not in str(e) and update.callback_query:
            await context.bot.send_message(chat_id=target_message.chat_id, text=message_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)

async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE): await (update.message or update.callback_query.message).reply_text("اختر الإعداد:", reply_markup=ReplyKeyboardMarkup(settings_menu_keyboard, resize_keyboard=True))

def get_scanners_keyboard():
    active = bot_data["settings"].get("active_scanners", [])
    buttons = [[InlineKeyboardButton(f"{'✅' if n in active else '❌'} {STRATEGY_NAMES_AR.get(n, n)}", callback_data=f"toggle_{n}")] for n in SCANNERS.keys()]
    buttons.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    return InlineKeyboardMarkup(buttons)

def get_presets_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚦 احترافية", callback_data="preset_PRO"), InlineKeyboardButton("🎯 متشددة", callback_data="preset_STRICT")],
        [InlineKeyboardButton("🌙 متساهلة", callback_data="preset_LAX"), InlineKeyboardButton("⚠️ فائق التساهل", callback_data="preset_VERY_LAX")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")]
    ])

async def show_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE): await (update.message or update.callback_query.message).reply_text("اختر نمط:", reply_markup=get_presets_keyboard())
async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE): await (update.message or update.callback_query.message).reply_text("اختر الماسحات:", reply_markup=get_scanners_keyboard())
async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard, settings = [], bot_data["settings"]
    for category, params in EDITABLE_PARAMS.items():
        keyboard.append([InlineKeyboardButton(f"--- {category} ---", callback_data="ignore")])
        for row in [params[i:i + 2] for i in range(0, len(params), 2)]:
            buttons = []
            for p_key in row:
                val = settings.get(p_key, "N/A")
                text = f"{PARAM_DISPLAY_NAMES.get(p_key, p_key)}: {'✅' if val else '❌'}" if isinstance(val, bool) else f"{PARAM_DISPLAY_NAMES.get(p_key, p_key)}: {val}"
                buttons.append(InlineKeyboardButton(text, callback_data=f"param_{p_key}"))
            keyboard.append(buttons)
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    text = "⚙️ *الإعدادات المتقدمة*\n\nاختر للإعداد للتعديل:"
    target = update.callback_query.message if update.callback_query else update.message
    try:
        if update.callback_query: await target.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
        else: context.user_data['settings_menu_id'] = (await target.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)).message_id
    except BadRequest as e:
        if "Message is not modified" not in str(e): logger.error(f"Error editing parameters menu: {e}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = ("**💣 أوامر البوت 💣**\n\n`/start` - بدء التفاعل.\n"
                 "`/check <ID>` - متابعة صفقة.\n`/trade` - تداول يدوي.")
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def send_daily_report(context: ContextTypes.DEFAULT_TYPE):
    today = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d')
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row
        real_trades = [dict(r) for r in conn.cursor().execute("SELECT * FROM trades WHERE DATE(closed_at) = ? AND trade_mode = 'real'", (today,)).fetchall()]
        virtual_trades = [dict(r) for r in conn.cursor().execute("SELECT * FROM trades WHERE DATE(closed_at) = ? AND trade_mode = 'virtual'", (today,)).fetchall()]
        conn.close()
        parts = [f"**🗓️ التقرير اليومي | {today}**"]
        def generate_section(title, trades):
            if not trades: return [f"\n--- **{title}** ---\nلم تُغلق أي صفقات اليوم."]
            wins, losses = [t for t in trades if t['status'] == 'ناجحة'], [t for t in trades if t['status'] == 'فاشلة']
            pnl = sum(t['pnl_usdt'] for t in trades if t['pnl_usdt'] is not None)
            return [f"\n--- **{title}** ---", f"  - صافي الربح/الخسارة: `${pnl:+.2f}`", f"  - ✅ الرابحة: {len(wins)} | ❌ الخاسرة: {len(losses)}"]
        parts.extend(generate_section("💰 الأداء الحقيقي", real_trades))
        parts.extend(generate_section("📊 الأداء الوهمي", virtual_trades))
        await send_telegram_message(context.bot, {'custom_message': "\n".join(parts), 'target_chat': TELEGRAM_SIGNAL_CHANNEL_ID})
    except Exception as e: logger.error(f"Failed to generate daily report: {e}", exc_info=True)

async def check_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_id_from_callback=None):
    target = update.callback_query.message if trade_id_from_callback else update.message
    def format_price(price): return f"{price:,.8f}" if price < 0.01 else f"{price:,.4f}"
    try:
        trade_id = trade_id_from_callback or int(context.args[0])
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row
        trade = dict(row) if (row := conn.cursor().execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()) else None; conn.close()
        if not trade: await target.reply_text(f"لم يتم العثور على صفقة بالرقم `{trade_id}`."); return
        if trade['status'] != 'نشطة':
            pnl_percent = (trade['pnl_usdt'] / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0
            closed_at = EGYPT_TZ.localize(datetime.strptime(trade['closed_at'], '%Y-%m-%d %H:%M:%S')).strftime('%Y-%m-%d %I:%M %p')
            message = f"📋 *ملخص صفقة #{trade_id}*\n\n*العملة:* `{trade['symbol']}`\n*الحالة:* `{trade['status']}`\n*الإغلاق:* `{closed_at}`\n*الربح/الخسارة:* `${trade.get('pnl_usdt', 0):+.2f} ({pnl_percent:+.2f}%)`"
        else:
            exchange = bot_data["public_exchanges"].get(trade['exchange'].lower())
            if not exchange or not (ticker := await exchange.fetch_ticker(trade['symbol'])) or not (current_price := ticker.get('last') or ticker.get('close')):
                message = "لم أتمكن من جلب السعر الحالي."
            else:
                live_pnl = (current_price - trade['entry_price']) * trade['quantity']
                pnl_percent = (live_pnl / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0
                message = (f"📈 *متابعة حية لصفقة #{trade_id}*\n\n"
                           f"▫️ *العملة:* `{trade['symbol']}` | *الحالة:* `نشطة`\n"
                           f"▫️ *الدخول:* `${format_price(trade['entry_price'])}`\n"
                           f"▫️ *الحالي:* `${format_price(current_price)}`\n"
                           f"💰 *الربح/الخسارة الآن:*\n`${live_pnl:+.2f} ({pnl_percent:+.2f}%)`")
        await target.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    except (ValueError, IndexError): await target.reply_text("رقم صفقة غير صالح. مثال: `/check 17`")
    except Exception as e: logger.error(f"Error in check_trade_command: {e}", exc_info=True); await target.reply_text("حدث خطأ.")

async def execute_manual_trade(exchange_id, symbol, amount_usdt, side, context: ContextTypes.DEFAULT_TYPE):
    exchange = bot_data["exchanges"].get(exchange_id.lower())
    if not exchange or not exchange.apiKey: return {"success": False, "error": f"لم يتم توثيق الاتصال بـ {exchange_id.capitalize()}."}
    try:
        price = (await exchange.fetch_ticker(symbol)).get('last')
        if not price: return {"success": False, "error": f"لم أجد سعر {symbol}."}
        quantity = exchange.amount_to_precision(symbol, float(amount_usdt) / price)
        order_receipt = await (exchange.create_market_buy_order if side == 'buy' else exchange.create_market_sell_order)(symbol, float(quantity))
        await asyncio.sleep(2); order = await exchange.fetch_order(order_receipt['id'], symbol)
        cost = order.get('cost', order.get('filled', 0) * order.get('average', 0))
        return {"success": True, "message": f"**✅ تم تنفيذ الأمر بنجاح**\n\n- **المنصة:** `{exchange_id.capitalize()}`\n- **العملة:** `{symbol}`\n- **النوع:** `{side.upper()}`\n- **التكلفة:** `${cost:.2f}`"}
    except Exception as e: return {"success": False, "error": f"❌ فشل: {e}"}

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data
    
    back_to_dashboard_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="dashboard_refresh")]])

    if data.startswith("dashboard_") and data.endswith(('_all', '_real', '_virtual')):
        try:
            # [FIX] Correctly unpack the callback data
            parts = data.split('_')
            report_type = parts[1]
            trade_mode_filter = parts[2]
            
            report_string, keyboard = "Processing...", back_to_dashboard_keyboard
            if report_type == "stats": report_string = await generate_stats_report_string(trade_mode_filter)
            elif report_type == "strategy_report": report_string = await generate_strategy_report_string(trade_mode_filter)
            elif report_type == "active_trades": report_string, keyboard = await generate_active_trades_string_and_keyboard(trade_mode_filter)
            
            if not keyboard: keyboard = back_to_dashboard_keyboard
            await query.edit_message_text(report_string, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Error in dashboard filter handler: {e}", exc_info=True)
            await query.edit_message_text("❌ حدث خطأ. يرجى المحاولة مرة أخرى.", reply_markup=back_to_dashboard_keyboard)
        return

    if data.startswith("dashboard_"):
        action = data.split("_", 1)[1]
        if action in ["stats", "active_trades", "strategy_report"]:
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 الكل", callback_data=f"dashboard_{action}_all")],
                [InlineKeyboardButton("📈 حقيقي فقط", callback_data=f"dashboard_{action}_real"), InlineKeyboardButton("📉 وهمي فقط", callback_data=f"dashboard_{action}_virtual")],
                [InlineKeyboardButton("🔙 العودة", callback_data="dashboard_refresh")]])
            await query.edit_message_text(f"اختر نوع السجل:", reply_markup=keyboard)
            return
        
        report_string, keyboard = f"⏳ جارِ إعداد تقرير: *{action.replace('_',' ')}*...", back_to_dashboard_keyboard
        await query.edit_message_text(report_string, parse_mode=ParseMode.MARKDOWN)

        if action == "refresh": await show_dashboard_command(update, context); return
        elif action == "debug": report_string = await generate_debug_report_string(context)
        elif action == "snapshot": report_string = await generate_portfolio_snapshot_string()
        elif action == "risk": report_string = await generate_risk_report_string()
        elif action == "sync": report_string = await generate_sync_portfolio_string()
        elif action == "tools":
             keyboard = InlineKeyboardMarkup([
                 [InlineKeyboardButton("✍️ تداول يدوي", callback_data="tools_manual_trade"), InlineKeyboardButton("💰 عرض رصيدي", callback_data="tools_balance")],
                 [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="dashboard_refresh")]])
             report_string = "🛠️ *أدوات التداول*\n\nاختر الأداة:"
        
        await query.edit_message_text(report_string, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        return

    elif data.startswith("tools_"):
        tool = data.split("_", 1)[1]
        if tool == "manual_trade": await manual_trade_command(update, context)
        elif tool == "balance": await balance_command(update, context)
        return

    elif data.startswith("preset_"):
        preset_name = data.split("_", 1)[1]
        if preset_data := PRESETS.get(preset_name):
            settings = bot_data["settings"]
            settings['liquidity_filters'], settings['volatility_filters'] = preset_data['liquidity_filters'], preset_data['volatility_filters']
            settings['ema_trend_filter'], settings['min_tp_sl_filter'] = preset_data['ema_trend_filter'], preset_data['min_tp_sl_filter']
            settings["active_preset_name"] = preset_name; save_settings()
            title = {"PRO": "احترافي", "STRICT": "متشدد", "LAX": "متساهل", "VERY_LAX": "فائق التساهل"}.get(preset_name)
            await query.edit_message_text(f"✅ *تم تفعيل النمط: {title}*", parse_mode=ParseMode.MARKDOWN, reply_markup=get_presets_keyboard())
    elif data.startswith("param_"):
        param_key = data.split("_", 1)[1]
        context.user_data.update({'awaiting_input_for_param': param_key, 'settings_menu_id': query.message.message_id})
        current_value = bot_data["settings"].get(param_key)
        if isinstance(current_value, bool):
            bot_data["settings"][param_key] = not current_value
            bot_data["settings"]["active_preset_name"] = "Custom"; save_settings()
            await show_parameters_menu(update, context)
        else: await query.edit_message_text(f"📝 *تعديل '{PARAM_DISPLAY_NAMES.get(param_key, param_key)}'*\n\n*الحالي:* `{current_value}`\n\nأرسل القيمة الجديدة.", parse_mode=ParseMode.MARKDOWN)
    elif data.startswith("toggle_"):
        scanner_name = data.split("_", 1)[1]
        active = bot_data["settings"].get("active_scanners", []).copy()
        if scanner_name in active: active.remove(scanner_name)
        else: active.append(scanner_name)
        bot_data["settings"]["active_scanners"] = active; save_settings()
        await query.edit_message_text(text="اختر الماسحات:", reply_markup=get_scanners_keyboard())
    elif data == "back_to_settings":
        await query.message.delete()
        await context.bot.send_message(chat_id=query.message.chat_id, text="اختر الإعداد:", reply_markup=ReplyKeyboardMarkup(settings_menu_keyboard, resize_keyboard=True))
    elif data.startswith("check_"): await check_trade_command(update, context, trade_id_from_callback=int(data.split("_")[1]))
    elif data.startswith("suggest_"):
        action, preset_name = data.split("_")[1], data.split("_")[2]
        if action == "accept":
            if preset_data := PRESETS.get(preset_name):
                s = bot_data["settings"]
                s['liquidity_filters'], s['volatility_filters'], s['ema_trend_filter'], s['min_tp_sl_filter'] = preset_data['liquidity_filters'], preset_data['volatility_filters'], preset_data['ema_trend_filter'], preset_data['min_tp_sl_filter']
                s["active_preset_name"] = preset_name; save_settings()
                await query.edit_message_text(f"✅ **تم قبول الاقتراح!** تم التغيير إلى `{preset_name}`.", parse_mode=ParseMode.MARKDOWN)
        elif action == "decline": await query.edit_message_text("👍 **تم تجاهل الاقتراح.**", parse_mode=ParseMode.MARKDOWN)

async def manual_trade_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data; user_data = context.user_data
    if 'manual_trade' not in user_data: await query.edit_message_text("⚠️ انتهت الجلسة."); return
    state = user_data['manual_trade'].get('state')
    if data == "manual_trade_cancel": user_data.pop('manual_trade', None); await query.edit_message_text("👍 تم الإلغاء."); return
    if state == 'awaiting_exchange':
        user_data['manual_trade'].update({'exchange': data.split("_")[-1], 'state': 'awaiting_symbol'})
        await query.edit_message_text(f"اخترت: *{data.split('_')[-1].capitalize()}*\n\nأرسل رمز العملة (مثال: `BTC/USDT`).", parse_mode=ParseMode.MARKDOWN)
    elif state == 'awaiting_side':
        user_data['manual_trade'].update({'side': data.split("_")[-1], 'state': 'confirming'})
        trade_data = user_data['manual_trade']
        await query.edit_message_text("⏳ جاري التنفيذ...", reply_markup=None)
        result = await execute_manual_trade(exchange_id=trade_data['exchange'], symbol=trade_data['symbol'], amount_usdt=trade_data['amount'], side=trade_data['side'], context=context)
        await query.edit_message_text(result['message'] if result['success'] else result['error'], parse_mode=ParseMode.MARKDOWN)
        user_data.pop('manual_trade', None)

async def tools_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data
    tool_name, _, value = data.partition("_exchange_")
    if tool_name == "balance":
        await query.edit_message_text(f"💰 جاري جلب الأرصدة من *{value.capitalize()}*...", parse_mode=ParseMode.MARKDOWN)
        await fetch_and_display_balance(value, query)

async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    user_data, text = context.user_data, update.message.text
    if 'manual_trade' in user_data:
        state = user_data['manual_trade'].get('state')
        if state == 'awaiting_symbol':
            user_data['manual_trade'].update({'symbol': text.upper(), 'state': 'awaiting_amount'})
            await update.message.reply_text(f"الرمز: *{text.upper()}*\n\nأدخل المبلغ بـ USDT (مثال: `15`).", parse_mode=ParseMode.MARKDOWN)
        elif state == 'awaiting_amount':
            try:
                user_data['manual_trade'].update({'amount': float(text), 'state': 'awaiting_side'})
                keyboard = [[InlineKeyboardButton("📈 شراء", callback_data="manual_trade_side_buy"), InlineKeyboardButton("📉 بيع", callback_data="manual_trade_side_sell")], [InlineKeyboardButton("❌ إلغاء", callback_data="manual_trade_cancel")]]
                await update.message.reply_text(f"المبلغ: *${float(text)}*\n\nاختر نوع الأمر:", reply_markup=InlineKeyboardMarkup(keyboard))
            except ValueError: await update.message.reply_text("❌ مبلغ غير صالح. أرسل رقم فقط.")
        return
    menu_handlers = {"Dashboard 🖥️": show_dashboard_command, "ℹ️ مساعدة": help_command, "⚙️ الإعدادات": show_settings_menu, "🔧 تعديل المعايير": show_parameters_menu, "🔙 القائمة الرئيسية": start_command, "🎭 تفعيل/تعطيل الماسحات": show_scanners_menu, "🏁 أنماط جاهزة": show_presets_menu}
    if text in menu_handlers:
        for key in list(user_data.keys()):
            if key.startswith('manual_trade') or key == 'awaiting_input_for_param': user_data.pop(key)
        await menu_handlers[text](update, context); return
    if param := user_data.pop('awaiting_input_for_param', None):
        settings_menu_id, chat_id = context.user_data.pop('settings_menu_id', None), update.message.chat_id
        await context.bot.delete_message(chat_id=chat_id, message_id=update.message.message_id)
        settings = bot_data["settings"]
        try:
            current_type = type(settings.get(param, ''))
            new_value = current_type(text) if not isinstance(settings.get(param), bool) else text.lower() in ['true', '1', 'on', 'نعم']
            settings[param], settings["active_preset_name"] = new_value, "Custom"; save_settings()
            if settings_menu_id: context.user_data['settings_menu_id'] = settings_menu_id
            await show_parameters_menu(update, context)
            confirm_msg = await update.message.reply_text(f"✅ تم تحديث **{PARAM_DISPLAY_NAMES.get(param, param)}**.", parse_mode=ParseMode.MARKDOWN)
            context.job_queue.run_once(lambda ctx: ctx.bot.delete_message(chat_id, confirm_msg.message_id), 4)
        except:
            if settings_menu_id: await context.bot.edit_message_text(chat_id=chat_id, message_id=settings_menu_id, text="❌ قيمة غير صالحة.")

async def manual_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['manual_trade'] = {'state': 'awaiting_exchange'}
    keyboard = [[InlineKeyboardButton(ex.capitalize(), callback_data=f"manual_trade_exchange_{ex}")] for ex in bot_data['exchanges'] if bot_data['exchanges'][ex].apiKey]
    keyboard.append([InlineKeyboardButton("❌ إلغاء", callback_data="manual_trade_cancel")])
    text = "✍️ **تداول يدوي**\n\nاختر المنصة:"
    if update.callback_query: await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    else: await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(ex.capitalize(), callback_data=f"balance_exchange_{ex}")] for ex in bot_data['exchanges'] if bot_data['exchanges'][ex].apiKey]
    keyboard.append([InlineKeyboardButton("🔙 العودة للأدوات", callback_data="dashboard_tools")])
    await update.callback_query.edit_message_text("💰 **عرض الرصيد**\n\nاختر المنصة:", reply_markup=InlineKeyboardMarkup(keyboard))

async def fetch_and_display_balance(exchange_id, query):
    exchange = bot_data["exchanges"].get(exchange_id.lower())
    if not exchange or not exchange.apiKey: await query.edit_message_text(f"❌ خطأ: لم يتم توثيق الاتصال بـ {exchange_id.capitalize()}."); return
    try:
        balance = await exchange.fetch_balance()
        tickers = await bot_data['public_exchanges'][exchange_id.lower()].fetch_tickers()
        assets = []
        for currency, amount in balance.get('total', {}).items():
            if amount > 0:
                usdt_val = amount if currency == 'USDT' else amount * tickers[f"{currency}/USDT"]['last'] if f"{currency}/USDT" in tickers else 0
                if usdt_val > 1: assets.append({'curr': currency, 'amt': amount, 'val': usdt_val})
        assets.sort(key=lambda x: x['val'], reverse=True)
        if not assets: await query.edit_message_text(f"ℹ️ لا توجد أرصدة (> $1) على {exchange_id.capitalize()}."); return
        lines = [f"**💰 رصيدك على {exchange_id.capitalize()}**", f"__**إجمالي القيمة:**__ `${sum(a['val'] for a in assets):,.2f}`\n"]
        for asset in assets[:15]: lines.append(f"- `{asset['curr']}`: `{asset['amt']:.4f}` (~`${asset['val']:.2f}`)")
        await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e: await query.edit_message_text(f"❌ خطأ أثناء جلب الرصيد: {e}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None: logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

# [REFACTORED] Multi-exchange support
async def generate_portfolio_snapshot_string():
    exchanges = [ex for ex in bot_data["exchanges"].values() if ex.apiKey]
    if not exchanges: return "❌ **فشل:** لم يتم العثور على أي منصة متصلة بحساب حقيقي."
    
    full_report = []
    for exchange in exchanges:
        try:
            balance, tickers = await exchange.fetch_balance(), await exchange.fetch_tickers()
            assets, total_val = [], 0
            for curr, amt in balance.get('total', {}).items():
                if amt > 0:
                    val = amt if curr == 'USDT' else amt * tickers[f"{curr}/USDT"]['last'] if f"{curr}/USDT" in tickers and tickers[f"{curr}/USDT"].get('last') else 0
                    if val > 1: assets.append({'curr': curr, 'amt': amt, 'val': val}); total_val += val
            assets.sort(key=lambda x: x['val'], reverse=True)
            
            recent_trades = []
            for asset in assets:
                try:
                    if f"{asset['curr']}/USDT" in exchange.markets:
                        recent_trades.extend(await exchange.fetch_my_trades(symbol=f"{asset['curr']}/USDT", limit=5))
                except Exception: pass # Ignore errors for single symbols
            recent_trades.sort(key=lambda x: x['timestamp'], reverse=True)
            
            parts = [f"--- **📸 لقطة لمحفظة {exchange.id.capitalize()}** ---", f"__**إجمالي القيمة:**__ `${total_val:,.2f}`\n", "**الأرصدة الحالية (> $1):**"]
            for asset in assets[:10]: parts.append(f"- **{asset['curr']}**: `{asset['amt']:.4f}` *~`${asset['val']:.2f}`*")
            parts.append("\n**آخر عمليات التداول:**")
            if not recent_trades: parts.append("لا يوجد سجل حديث.")
            else:
                for trade in recent_trades[:10]: parts.append(f"`{trade['symbol']}` {'🟢' if trade['side'] == 'buy' else '🔴'} `{trade['side'].upper()}` | الكمية: `{trade['amount']}`")
            full_report.append("\n".join(parts))
        except Exception as e: full_report.append(f"❌ **فشل جلب بيانات {exchange.id.capitalize()}:** `{e}`")
    return "\n\n".join(full_report)

# [REFACTORED] Multi-exchange support & Bug fix
async def generate_risk_report_string():
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row
        all_trades = [dict(row) for row in conn.cursor().execute("SELECT * FROM trades WHERE status = 'نشطة'").fetchall()]; conn.close()
        
        real_trades_by_ex = defaultdict(list)
        virtual_trades = []
        for t in all_trades:
            if t['trade_mode'] == 'real': real_trades_by_ex[t['exchange'].lower()].append(t)
            else: virtual_trades.append(t)

        full_report = ["**ρίск تقرير المخاطر الحالي**"]

        def generate_risk_section(title, trades, portfolio_value):
            if not trades: return [f"\n--- **{title}** ---\n✅ لا توجد صفقات نشطة."]
            valid_trades = [t for t in trades if all(t.get(k) is not None for k in ['entry_value_usdt', 'entry_price', 'stop_loss', 'quantity'])]
            if not valid_trades: return [f"\n--- **{title}** ---\n⚠️ لا توجد صفقات ببيانات كاملة."]
            
            total_at_risk = sum(t['entry_value_usdt'] for t in valid_trades)
            potential_loss = sum((t['entry_price'] - t['stop_loss']) * t['quantity'] for t in valid_trades)
            
            # [FIX] Separate declaration from usage and handle empty case
            symbol_concentration = Counter(t['symbol'] for t in valid_trades)
            most_common = symbol_concentration.most_common(1)[0] if symbol_concentration else None

            parts = [f"\n--- **{title}** ---", f"- **عدد الصفقات:** {len(valid_trades)}", f"- **إجمالي المعرض للخطر:** `${total_at_risk:,.2f}`"]
            if portfolio_value > 0: parts.append(f"- **نسبة التعرض:** `{(total_at_risk / portfolio_value) * 100:.2f}%`")
            parts.append(f"- **أقصى خسارة محتملة:** `$-{potential_loss:,.2f}`")
            if most_common: parts.append(f"- **العملة الأكثر تركيزاً:** `{most_common[0]}` ({most_common[1]} صفقات)")
            return parts

        connected_exchanges = [ex for ex in bot_data["exchanges"].values() if ex.apiKey]
        for ex in connected_exchanges:
            balance = await get_real_balance(ex.id, 'USDT')
            full_report.extend(generate_risk_section(f"🚨 المخاطر الحقيقية ({ex.id.capitalize()})", real_trades_by_ex[ex.id.lower()], balance))

        full_report.extend(generate_risk_section("📊 المخاطر الوهمية", virtual_trades, bot_data['settings']['virtual_portfolio_balance_usdt']))
        return "\n".join(full_report)
    except Exception as e: return f"❌ **فشل:** {e}"

# [REFACTORED] Multi-exchange support
async def generate_sync_portfolio_string():
    exchanges = [ex for ex in bot_data["exchanges"].values() if ex.apiKey]
    if not exchanges: return "❌ **فشل:** لا توجد منصات متصلة."
    
    full_report = []
    for exchange in exchanges:
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10)
            bot_symbols = {r[0] for r in conn.cursor().execute("SELECT symbol FROM trades WHERE status = 'نشطة' AND trade_mode = 'real' AND exchange = ?", (exchange.id.capitalize(),)).fetchall()}; conn.close()
            
            balance = await exchange.fetch_balance()
            exchange_symbols = {f"{c}/USDT" for c, a in balance.get('total', {}).items() if a > 0 and f"{c}/USDT" in exchange.markets}

            matched, bot_only, ex_only = bot_symbols.intersection(exchange_symbols), bot_symbols.difference(exchange_symbols), exchange_symbols.difference(bot_symbols)
            
            parts = [f"--- **🔄 مزامنة المحفظة ({exchange.id.capitalize()})** ---", f"`{len(bot_symbols)}` صفقة بالبوت مقابل `{len(exchange_symbols)}` عملة بالمنصة.\n",
                     f"**✅ متطابقة ({len(matched)}):** {', '.join(matched) if matched else 'لا يوجد.'}",
                     f"**❓ بالبوت فقط ({len(bot_only)}):** {', '.join(bot_only) if bot_only else 'لا يوجد.'}",
                     f"**⚠️ بالمنصة فقط ({len(ex_only)}):** {', '.join(ex_only) if ex_only else 'لا يوجد.'}"]
            full_report.append("\n".join(parts))
        except Exception as e: full_report.append(f"❌ **فشل مزامنة {exchange.id.capitalize()}:** `{e}`")
    return "\n\n".join(full_report)

async def generate_debug_report_string(context: ContextTypes.DEFAULT_TYPE):
    settings = bot_data.get("settings", {})
    mood_info = settings.get("last_market_mood", {})
    fng_value = await get_fear_and_greed_index()
    fng_text = f"{fng_value} ({'خوف شديد' if fng_value < 25 else 'خوف' if fng_value < 45 else 'محايد' if fng_value < 55 else 'طمع' if fng_value < 75 else 'طمع شديد'})" if fng_value is not None else "N/A"
    
    status = bot_data['status_snapshot']
    scan_duration = "N/A"
    if status['last_scan_end_time'] != 'N/A' and status['last_scan_start_time'] != 'N/A':
        scan_duration = f"{(datetime.strptime(status['last_scan_end_time'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(status['last_scan_start_time'], '%Y-%m-%d %H:%M:%S')).total_seconds():.0f} ثانية"
    
    parts = [f"**🕵️‍♂️ تقرير التشخيص** | {datetime.now(EGYPT_TZ).strftime('%H:%M:%S')}",
             "\n- `NLTK/SciPy/AlphaVantage:` " + f"{'✅' if NLTK_AVAILABLE else '❌'}/{'✅' if SCIPY_AVAILABLE else '❌'}/{'✅' if ALPHA_VANTAGE_API_KEY != 'YOUR_AV_KEY_HERE' else '⚠️'}",
             f"- `مزاج السوق:` {mood_info.get('mood', 'N/A')} | `F&G:` {fng_text}",
             f"- `آخر فحص:` بدأ `{status['last_scan_start_time']}` | المدة: `{scan_duration}`",
             "\n**المنصات المتصلة:**"]
    for ex_id in EXCHANGES_TO_SCAN:
        parts.append(f"  - `{ex_id.capitalize()}:` عام: {'✅' if ex_id in bot_data.get('public_exchanges', {}) else '❌'} | خاص: {'✅' if ex_id in bot_data.get('exchanges', {}) and bot_data['exchanges'][ex_id].apiKey else '❌'}")
    
    try:
        conn = sqlite3.connect(DB_FILE, timeout=5)
        total, active = conn.cursor().execute("SELECT COUNT(*), (SELECT COUNT(*) FROM trades WHERE status = 'نشطة') FROM trades").fetchone()
        conn.close()
        parts.append(f"\n**قاعدة البيانات:**\n  - `الاتصال:` ✅ | `الحجم:` {os.path.getsize(DB_FILE) / 1024**2:.2f} MB\n  - `الصفقات:` {total} ({active} نشطة)")
    except Exception as e: parts.append(f"\n**قاعدة البيانات:**\n  - `الاتصال:` ❌ ({e})")
    
    return "\n".join(parts)


async def post_init(application: Application):
    if NLTK_AVAILABLE:
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError: logger.info("Downloading NLTK data..."); nltk.download('vader_lexicon')
    await initialize_exchanges()
    if not bot_data["public_exchanges"]: logger.critical("CRITICAL: No public exchange clients connected."); return
    if bot_data['settings'].get('real_trading_enabled') and not any(ex.apiKey for ex in bot_data.get('exchanges', {}).values()):
        await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="**🚨 خطأ: التداول الحقيقي مفعل لكن لا توجد مفاتيح API.**")
        return
    job_queue = application.job_queue
    job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name='perform_scan')
    job_queue.run_repeating(track_open_trades, interval=TRACK_INTERVAL_SECONDS, first=20, name='track_open_trades')
    job_queue.run_daily(send_daily_report, time=dt_time(hour=23, minute=55, tzinfo=EGYPT_TZ), name='daily_report')
    await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🚀 *بوت كاسحة الألغام (v2.1) جاهز للعمل!*", parse_mode=ParseMode.MARKDOWN)

async def post_shutdown(application: Application):
    exchanges = list({id(ex): ex for ex in list(bot_data["exchanges"].values()) + list(bot_data["public_exchanges"].values())}.values())
    await asyncio.gather(*[ex.close() for ex in exchanges])
    logger.info("All exchange connections closed.")

def main():
    print("🚀 Starting Minesweeper Bot v2.1 (Multi-Exchange & Final Fixes)...")
    load_settings(); init_database()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).post_shutdown(post_shutdown).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("check", check_trade_command))
    app.add_handler(CommandHandler("trade", manual_trade_command))
    app.add_handler(CallbackQueryHandler(manual_trade_button_handler, pattern="^manual_trade_"))
    app.add_handler(CallbackQueryHandler(tools_button_handler, pattern="^balance_"))
    app.add_handler(CallbackQueryHandler(button_callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    app.add_error_handler(error_handler)
    print("✅ Bot is now running...")
    app.run_polling()

if __name__ == '__main__':
    try: main()
    except Exception as e: logging.critical(f"Bot stopped due to a critical unhandled error: {e}", exc_info=True)

