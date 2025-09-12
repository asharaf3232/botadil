# -*- coding: utf-8 -*-
# =================================================================================================
# == 💣 Minesweeper Bot v1.0 | كاسحة الألغام 💣 =====================================================
# =================================================================================================
#
# MISSION LOG:
# - PHASE 1: FOUNDATION ESTABLISHED. Using the robust multi-exchange framework of 'Analyzer Bot'.
# - PHASE 2: HUNTER'S ARSENAL INTEGRATED. Ported winning strategies from 'FOMO Hunter':
#   -> Support Rebound, Whale Radar, Sniper Pro.
# - PHASE 3: SECRET WEAPON ACTIVATED. Replaced the existing Trailing SL with the superior,
#   battle-tested logic from 'FOMO Hunter' for maximum profit protection.
#
# =================================================================================================

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
import numpy as np # <-- تم جلبه من صياد الفومو

# [UPGRADE] المكتبات الجديدة لتحليل الأخبار
import feedparser
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
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

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_BINANCE_API_SECRET')
KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', 'YOUR_KUCOIN_API_KEY')
KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET', 'YOUR_KUCOIN_API_SECRET')
KUCOIN_API_PASSPHRASE = os.getenv('KUCOIN_API_PASSPHRASE', 'YOUR_KUCOIN_PASSPHRASE')


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

# [PHASE 2] دمج أسماء الاستراتيجيات الجديدة
STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي",
    "breakout_squeeze_pro": "اختراق انضغاطي",
    "rsi_divergence": "دايفرجنس RSI",
    "supertrend_pullback": "انعكاس سوبرترند",
    "support_rebound": "ارتداد من الدعم",
    "whale_radar": "رادار الحيتان",
    "sniper_pro": "القناص المحترف"
}


# --- Constants for Interactive Settings menu ---
# [PHASE 3] تحديث قائمة الإعدادات بمنطق الـ Trailing SL الجديد
EDITABLE_PARAMS = {
    "إعدادات عامة": [
        "max_concurrent_trades", "top_n_symbols_by_volume", "concurrent_workers",
        "min_signal_strength"
    ],
    "إعدادات المخاطر": [
        "real_trading_enabled", "real_trade_size_usdt", "virtual_trade_size_percentage", 
        "atr_sl_multiplier", "risk_reward_ratio"
    ],
     "الوقف المتحرك (Trailing SL)": [
        "trailing_sl_enabled", "trailing_sl_activation_percent", "trailing_sl_callback_percent"
    ],
    "الفلاتر والاتجاه": [
        "market_regime_filter_enabled", "use_master_trend_filter", "fear_and_greed_filter_enabled",
        "master_adx_filter_level", "master_trend_filter_ma_period", "fear_and_greed_threshold",
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
    "trailing_sl_enabled": "تفعيل الوقف المتحرك",
    "trailing_sl_activation_percent": "تفعيل الوقف عند ربح (%)", # <-- [PHASE 3]
    "trailing_sl_callback_percent": "مسافة التتبع من القمة (%)", # <-- [PHASE 3]
    "market_regime_filter_enabled": "فلتر وضع السوق (فني)",
    "use_master_trend_filter": "فلتر الاتجاه العام (BTC)",
    "master_adx_filter_level": "مستوى فلتر ADX",
    "master_trend_filter_ma_period": "فترة فلتر الاتجاه",
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
# [PHASE 3] تحديث الإعدادات الافتراضية
DEFAULT_SETTINGS = {
    "real_trading_enabled": False,
    "real_trade_size_usdt": 15.0,
    "virtual_portfolio_balance_usdt": 1000.0, "virtual_trade_size_percentage": 5.0, "max_concurrent_trades": 5, "top_n_symbols_by_volume": 250, "concurrent_workers": 10,
    "market_regime_filter_enabled": True, "fundamental_analysis_enabled": True,
    # [PHASE 2] إضافة الاستراتيجيات الجديدة إلى القائمة الافتراضية
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "whale_radar", "sniper_pro"],
    "use_master_trend_filter": True, "master_trend_filter_ma_period": 50, "master_adx_filter_level": 22,
    "fear_and_greed_filter_enabled": True, "fear_and_greed_threshold": 30,
    "use_dynamic_risk_management": True, "atr_period": 14, "atr_sl_multiplier": 2.5, "risk_reward_ratio": 2.0,
    # [PHASE 3] اعتماد إعدادات الـ Trailing SL من صياد الفومو
    "trailing_sl_enabled": True, 
    "trailing_sl_activation_percent": 1.5, 
    "trailing_sl_callback_percent": 1.0,
    # إعدادات الاستراتيجيات القديمة
    "momentum_breakout": {"rsi_max_level": 68},
    "breakout_squeeze_pro": {"keltner_atr_multiplier": 1.5},
    "rsi_divergence": {},
    "supertrend_pullback": {},
    # [PHASE 2] إضافة إعدادات الاستراتيجيات الجديدة
    "sniper_pro": {"sniper_compression_hours": 6, "sniper_max_volatility_percent": 18.0},
    "whale_radar": {"whale_wall_threshold_usdt": 30000},
    "support_rebound": {},
    # الفلاتر
    "liquidity_filters": {"min_quote_volume_24h_usd": 1_000_000, "max_spread_percent": 0.5, "rvol_period": 20, "min_rvol": 1.5},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 1.0},
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
# [PHASE 3] تحديث جدول قاعدة البيانات ليتوافق مع Trailing SL الجديد
def init_database():
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                timestamp TEXT, 
                exchange TEXT, 
                symbol TEXT, 
                entry_price REAL, 
                take_profit REAL, 
                initial_stop_loss REAL, 
                current_stop_loss REAL,
                quantity REAL, 
                entry_value_usdt REAL, 
                status TEXT, 
                exit_price REAL, 
                closed_at TEXT, 
                exit_value_usdt REAL, 
                pnl_usdt REAL, 
                trailing_sl_active BOOLEAN, 
                highest_price REAL, 
                reason TEXT,
                is_real_trade BOOLEAN DEFAULT FALSE,
                entry_order_id TEXT,
                exit_order_ids_json TEXT
            )
        ''')
        # التأكد من وجود الأعمدة الجديدة
        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN initial_stop_loss REAL;")
            cursor.execute("ALTER TABLE trades ADD COLUMN current_stop_loss REAL;")
        except sqlite3.OperationalError:
            pass # الأعمدة موجودة بالفعل
        conn.commit()
        conn.close()
        logger.info(f"Database initialized successfully at: {DB_FILE}")
    except Exception as e:
        logger.error(f"Failed to initialize database at {DB_FILE}: {e}")

def log_recommendation_to_db(signal):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        sql = '''INSERT INTO trades (timestamp, exchange, symbol, entry_price, take_profit, 
                                  initial_stop_loss, current_stop_loss, quantity, entry_value_usdt, 
                                  status, trailing_sl_active, highest_price, reason, is_real_trade, 
                                  entry_order_id, exit_order_ids_json) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        params = (
            signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), 
            signal['exchange'], 
            signal['symbol'], 
            signal['entry_price'], 
            signal['take_profit'], 
            signal['stop_loss'], # Initial SL
            signal['stop_loss'], # Current SL starts as Initial
            signal['quantity'], 
            signal['entry_value_usdt'], 
            'نشطة', 
            False, 
            signal['entry_price'], 
            signal['reason'],
            signal.get('is_real_trade', False),
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


# --- [PHASE 2] دمج الاستراتيجيات الجديدة ودوالها المساعدة ---
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None
    
def find_support_resistance(high_prices, low_prices, window=10):
    supports, resistances = [], []
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

async def analyze_momentum_breakout(exchange, symbol, df, params, rvol, adx_value):
    df.ta.vwap(append=True)
    df.ta.bbands(length=20, std=2.0, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.rsi(length=14, append=True)
    macd_col, macds_col, bbu_col, rsi_col = (
        find_col(df.columns, f"MACD_12_26_9"), find_col(df.columns, f"MACDs_12_26_9"),
        find_col(df.columns, f"BBU_20_2.0"), find_col(df.columns, f"RSI_14")
    )
    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and
        last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and
        last[rsi_col] < params.get('rsi_max_level', 68)):
        return {"reason": "momentum_breakout", "entry_price": last['close']}
    return None

async def analyze_breakout_squeeze_pro(exchange, symbol, df, params, rvol, adx_value):
    df.ta.bbands(length=20, std=2.0, append=True)
    df.ta.kc(length=20, scalar=params.get('keltner_atr_multiplier', 1.5), append=True)
    df.ta.obv(append=True)
    bbu_col, bbl_col, kcu_col, kcl_col = (
        find_col(df.columns, "BBU_20_2.0"), find_col(df.columns, "BBL_20_2.0"),
        find_col(df.columns, "KCUe_20_"), find_col(df.columns, "KCLEe_20_")
    )
    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]
    if is_in_squeeze:
        breakout_fired = last['close'] > last[bbu_col]
        volume_ok = last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5
        obv_rising = df['OBV'].iloc[-2] > df['OBV'].iloc[-3]
        if breakout_fired and volume_ok and obv_rising:
            return {"reason": "breakout_squeeze_pro", "entry_price": last['close']}
    return None

async def analyze_rsi_divergence(exchange, symbol, df, params, rvol, adx_value):
    # This strategy is complex and kept from the original for diversity.
    # It might be less frequent but can provide high-quality signals.
    return None # Placeholder, can be fully integrated later if needed.

async def analyze_supertrend_pullback(exchange, symbol, df, params, rvol, adx_value):
    # This strategy is also kept for diversity.
    return None # Placeholder, can be fully integrated later if needed.

# --- استراتيجيات "صياد الفومو" المدمجة ---
async def analyze_sniper_pro(exchange, symbol, df, params, rvol, adx_value):
    compression_candles = int(params.get("sniper_compression_hours", 6) * 4) # 6 hours * 4 candles/hour
    if len(df) < compression_candles: return None

    compression_df = df.iloc[-compression_candles-1:-1]
    highest_high = compression_df['high'].max()
    lowest_low = compression_df['low'].min()

    volatility = (highest_high - lowest_low) / lowest_low * 100 if lowest_low > 0 else float('inf')

    if volatility < params.get("sniper_max_volatility_percent", 18.0):
        last_candle = df.iloc[-2]
        if last_candle['close'] > highest_high:
            avg_volume = compression_df['volume'].mean()
            if last_candle['volume'] > avg_volume * 2:
                return {"reason": "sniper_pro", "entry_price": last_candle['close']}
    return None

async def analyze_whale_radar(exchange, symbol, df, params, rvol, adx_value):
    threshold = params.get("whale_wall_threshold_usdt", 30000)
    try:
        ob = await exchange.fetch_order_book(symbol, limit=20)
        if not ob or not ob.get('bids'): return None

        total_bid_value = sum(float(price) * float(qty) for price, qty in ob['bids'][:10])

        if total_bid_value > threshold:
            reason = f"whale_radar ({total_bid_value/1000:.0f}K)"
            return {"reason": reason, "entry_price": df['close'].iloc[-2]}
    except Exception:
        pass # Fail silently on order book fetch errors
    return None

async def analyze_support_rebound(exchange, symbol, df, params, rvol, adx_value):
    try:
        # We need more data for this, fetching 1h
        ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
        if not ohlcv_1h or len(ohlcv_1h) < 50: return None

        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        current_price = df_1h['close'].iloc[-1]

        supports, _ = find_support_resistance(df_1h['high'].to_numpy(), df_1h['low'].to_numpy(), window=5)
        if not supports: return None

        closest_support = max([s for s in supports if s < current_price], default=None)
        if not closest_support: return None

        # Check if price is very close to support (e.g., within 1%)
        if (current_price - closest_support) / closest_support * 100 < 1.0:
            # Check for confirmation on the 15m chart (the one we passed in)
            last_candle = df.iloc[-2]
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-2]

            # Look for a bullish candle with increasing volume
            if last_candle['close'] > last_candle['open'] and last_candle['volume'] > avg_volume * 1.5:
                 return {"reason": "support_rebound", "entry_price": current_price}
        return None
    except Exception:
        return None

# [PHASE 2] القائمة الموحدة للاستراتيجيات
SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout,
    "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "support_rebound": analyze_support_rebound,
    "whale_radar": analyze_whale_radar,
    "sniper_pro": analyze_sniper_pro,
    # "rsi_divergence": analyze_rsi_divergence, # Disabled for now
    # "supertrend_pullback": analyze_supertrend_pullback, # Disabled for now
}


# --- Core Bot Functions ---
# ... The rest of the code from binance_trader_final.py
# ... From initialize_exchanges() to the end.
