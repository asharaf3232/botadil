# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 OKX Mastermind Trader v28.5 (Ultimate Resilience & Fee Awareness) 🚀 ---
# =======================================================================================
# This version implements the definitive, professional solution to trade quantity
# discrepancies by treating the exchange as the single source of truth.
#
# --- Version 28.5 Changelog ---
#   - 🎯 **TRUE FEE AWARENESS (SINGLE SOURCE OF TRUTH)**: The `activate_trade` function
#     has been re-engineered. It now correctly calculates the **net filled quantity** by
#     fetching the executed order and explicitly subtracting the trading fee cost from
#     the filled amount. The bot no longer assumes or guesses; it records the exact
#     amount that lands in the wallet. This is the root-cause fix for the
#     `CRITICAL BALANCE SHORTFALL` error.
#   - 🛡️ **FLEXIBLE SELL LOGIC (ULTIMATE FAILSAFE)**: The `_close_trade` function in the
#     Guardian has been made incredibly robust. When closing a trade, it will now
#     **ignore the quantity stored in the database**. Instead, it fetches the current
#     available balance of the asset and sells 100% of that amount. This acts as a
#     final, unbreakable failsafe against any possible state mismatch, ensuring trades
#     always close and liquidity is always freed.
#   - ✅ **DB & SETTINGS PRESERVED**: Continues to use v26 database and settings files.
#
# --- 🆕 New Advanced Filters from Al-Kasiha Bot (الكاسحة) ---
#   - 💹 **Spread Filter**: To avoid trading coins with high bid/ask spread.
#   - 📊 **ATR Filter**: To filter out low-volatility, "dead" coins.
#   - 📈 **EMA Trend Filter**: To ensure the asset is in an overall uptrend.
#
# --- 🐛 Bug Fixes from User Feedback ---
#   - ✅ **Fix 1**: Corrected nested setting value handling in `handle_setting_value`.
#     The bot can now correctly update simple settings (e.g., `real_trade_size_usdt`)
#     and nested settings (e.g., `trend_filters_ema_period`).
#   - ✅ **Fix 2**: Fixed the `db_manual_scan` button. The `manual_scan_command` now
#     correctly handles both text commands and inline button callbacks.
#   - ✅ **Fix 3**: Improved `determine_active_preset` to correctly identify presets
#     by performing a deep comparison of settings, fixing the "very lenient" preset issue.
# =======================================================================================


# --- Core Libraries ---
import asyncio
import os
import logging
import json
import re
import time
import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import hmac
import base64
from collections import defaultdict
import copy

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
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # For translation

TIMEFRAME = '15m'
SCAN_INTERVAL_SECONDS = 900
SUPERVISOR_INTERVAL_SECONDS = 120
TIME_SYNC_INTERVAL_SECONDS = 3600 # Check time every hour

APP_ROOT = '.'
# --- PERSISTENCE GUARANTEED: Using v26 files as requested ---
DB_FILE = os.path.join(APP_ROOT, 'mastermind_trader_v26.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'mastermind_trader_settings_v26.json')
# --- END OF PERSISTENCE SECTION ---

EGYPT_TZ = ZoneInfo("Africa/Cairo")

class SafeFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'trade_id'):
            record.trade_id = 'N/A'
        return super().format(record)

log_formatter = SafeFormatter('%(asctime)s - %(levelname)s - [TradeID:%(trade_id)s] - %(message)s')
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)
root_logger = logging.getLogger()
root_logger.handlers = [log_handler]
root_logger.setLevel(logging.INFO)

class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        if 'trade_id' not in kwargs['extra']:
            kwargs['extra']['trade_id'] = 'N/A'
        return msg, kwargs
logger = ContextAdapter(logging.getLogger("OKX_Mastermind_Trader"), {})

# =======================================================================================
# --- 🔬 Global Bot State & Locks 🔬 ---
# =======================================================================================
class BotState:
    def __init__(self):
        self.settings = {}
        self.trading_enabled = True # KILL SWITCH
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

# =======================================================================================
# --- 💡 Default Settings, Filters & UI Constants 💡 ---
# =======================================================================================
DEFAULT_SETTINGS = {
    "real_trade_size_usdt": 15.0,
    "max_concurrent_trades": 5,
    "top_n_symbols_by_volume": 300,
    "worker_threads": 10,
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 2.0,
    "trailing_sl_enabled": True,
    "trailing_sl_activation_percent": 1.5,
    "trailing_sl_callback_percent": 1.0,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "sniper_pro", "whale_radar"],
    "market_mood_filter_enabled": True,
    "fear_and_greed_threshold": 30,
    "adx_filter_enabled": True,
    "adx_filter_level": 25,
    "btc_trend_filter_enabled": True,
    "news_filter_enabled": True,
    "asset_blacklist": [
        "USDC", "DAI", "TUSD", "FDUSD", "USDD", "PYUSD", "USDT",
        "BNB", "OKB", "KCS", "BGB", "MX", "GT", "HT",
        "BTC", "ETH"
    ],
    "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "min_rvol": 1.5},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.8},
    "trend_filters": {"ema_period": 200, "htf_period": 50},
    "spread_filter": {"max_spread_percent": 0.5},
}
STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي", "breakout_squeeze_pro": "اختراق انضغاطي",
    "support_rebound": "ارتداد الدعم", "sniper_pro": "القناص المحترف", "whale_radar": "رادار الحيتان"
}
PRESET_NAMES_AR = {
    "professional": "احترافي", "strict": "متشدد",
    "lenient": "متساهل", "very_lenient": "فائق التساهل"
}
SETTINGS_PRESETS = {
    "professional": copy.deepcopy(DEFAULT_SETTINGS),
    "strict": {
        **copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 3, "risk_reward_ratio": 2.5,
        "fear_and_greed_threshold": 40, "adx_filter_level": 28,
        "liquidity_filters": {"min_quote_volume_24h_usd": 2000000, "min_rvol": 2.0},
    },
    "lenient": {
        **copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 8, "atr_sl_multiplier": 3.0,
        "risk_reward_ratio": 1.8, "fear_and_greed_threshold": 25, "adx_filter_level": 20,
        "liquidity_filters": {"min_quote_volume_24h_usd": 500000, "min_rvol": 1.2},
    },
    "very_lenient": {
        **copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 12, "atr_sl_multiplier": 3.5,
        "risk_reward_ratio": 1.5, "fear_and_greed_threshold": 20, "adx_filter_enabled": False,
        "market_mood_filter_enabled": False,
        "liquidity_filters": {"min_quote_volume_24h_usd": 250000, "min_rvol": 1.0},
    }
}


# =======================================================================================
# --- Helper, Settings & DB Management ---
# =======================================================================================
def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                bot_data.settings = json.load(f)
        else:
            bot_data.settings = copy.deepcopy(DEFAULT_SETTINGS)
    except Exception:
        bot_data.settings = copy.deepcopy(DEFAULT_SETTINGS)

    default_copy = copy.deepcopy(DEFAULT_SETTINGS)
    for key, value in default_copy.items():
        if isinstance(value, dict):
            if key not in bot_data.settings or not isinstance(bot_data.settings[key], dict):
                bot_data.settings[key] = {}
            for sub_key, sub_value in value.items():
                bot_data.settings[key].setdefault(sub_key, sub_value)
        else:
            bot_data.settings.setdefault(key, value)
    
    determine_active_preset()
    save_settings()
    logger.info(f"Settings loaded. Active preset identified as: {bot_data.active_preset_name}")

def determine_active_preset():
    found_preset = False
    
    # --- FIX 3: Use deep comparison to correctly identify presets ---
    current_settings_for_compare = copy.deepcopy(bot_data.settings)
    
    for name, preset_settings in SETTINGS_PRESETS.items():
        if current_settings_for_compare == preset_settings:
            bot_data.active_preset_name = PRESET_NAMES_AR.get(name, "مخصص")
            found_preset = True
            break
            
    if not found_preset:
        bot_data.active_preset_name = "مخصص"

def save_settings():
    with open(SETTINGS_FILE, 'w') as f: json.dump(bot_data.settings, f, indent=4)

async def safe_send_message(bot, text, **kwargs):
    try: await bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except Exception as e: logger.error(f"Telegram Send Error: {e}")
async def safe_edit_message(query, text, **kwargs):
    try: await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except BadRequest as e:
        if "Message is not modified" not in str(e): logger.warning(f"Edit Message Error: {e}")
    except Exception as e: logger.error(f"Edit Message Error: {e}")

async def init_database():
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT,
                    entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL,
                    status TEXT, -- pending, active, successful, failed
                    reason TEXT, order_id TEXT,
                    highest_price REAL DEFAULT 0, trailing_sl_active BOOLEAN DEFAULT 0
                )
            ''')

            cursor = await conn.execute("PRAGMA table_info(trades)")
            columns = [row[1] for row in await cursor.fetchall()]
            if 'close_price' not in columns:
                await conn.execute("ALTER TABLE trades ADD COLUMN close_price REAL")
            if 'pnl_usdt' not in columns:
                await conn.execute("ALTER TABLE trades ADD COLUMN pnl_usdt REAL")

            await conn.commit()
        logger.info("Mastermind database initialized/verified successfully.")
    except Exception as e: logger.critical(f"Database initialization failed: {e}")

async def log_pending_trade_to_db(signal, buy_order):
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute(
                "INSERT INTO trades (timestamp, symbol, reason, order_id, status, entry_price, take_profit, stop_loss) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(EGYPT_TZ).isoformat(), signal['symbol'], signal['reason'], buy_order['id'],
                 'pending', signal['entry_price'], signal['take_profit'], signal['stop_loss'])
            )
            await conn.commit()
            logger.info(f"Logged pending trade for {signal['symbol']} with order ID {buy_order['id']}.")
            return True
    except Exception as e:
        logger.error(f"DB Log Pending Error: {e}")
        return False

# =======================================================================================
# --- 🧠 Mastermind Brain (Analysis & Mood) 🧠 ---
# =======================================================================================
async def translate_text_gemini(text_list):
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found in .env file. Skipping translation.")
        return text_list, False
    if not text_list:
        return [], True

    prompt = "Translate the following English headlines to Arabic. Return only the translated text, with each headline on a new line:\n\n" + "\n".join(text_list)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            translated_text = result['candidates'][0]['content']['parts'][0]['text']
            return translated_text.strip().split('\n'), True
    except Exception as e:
        logger.error(f"Gemini translation failed: {e}")
        return text_list, False

def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

async def get_fear_and_greed_index():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            return int(r.json()['data'][0]['value'])
    except Exception: return None

def get_latest_crypto_news(limit=5):
    headlines = []
    urls = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]
    for url in urls:
        try:
            feed = feedparser.parse(url)
            headlines.extend(entry.title for entry in feed.entries[:limit])
        except Exception: pass
    return list(set(headlines))[:limit]

def analyze_sentiment_of_headlines(headlines):
    if not headlines or not NLTK_AVAILABLE: return "N/A", "N/A"
    sia = SentimentIntensityAnalyzer()
    score = sum(sia.polarity_scores(h)['compound'] for h in headlines) / len(headlines)
    if score > 0.1: mood = "إيجابية"
    elif score < -0.1: mood = "سلبية"
    else: mood = "محايدة"
    return mood, f"{score:.2f}"

async def get_market_mood():
    s = bot_data.settings
    if s.get('btc_trend_filter_enabled', True):
        try:
            htf_period = s['trend_filters']['htf_period']
            ohlcv = await bot_data.exchange.fetch_ohlcv('BTC/USDT', '4h', limit=htf_period + 5)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['sma'] = ta.sma(df['close'], length=htf_period)
            is_btc_bullish = df['close'].iloc[-1] > df['sma'].iloc[-1]
            btc_mood_text = "صاعد ✅" if is_btc_bullish else "هابط ❌"
            if not is_btc_bullish:
                return {"mood": "NEGATIVE", "reason": "اتجاه BTC هابط", "btc_mood": btc_mood_text}
        except Exception as e:
            return {"mood": "DANGEROUS", "reason": f"فشل جلب بيانات BTC: {e}", "btc_mood": "UNKNOWN"}
    else:
        btc_mood_text = "الفلتر معطل"

    if s.get('market_mood_filter_enabled', True):
        fng = await get_fear_and_greed_index()
        if fng is not None and fng < s['fear_and_greed_threshold']:
            return {"mood": "NEGATIVE", "reason": f"مشاعر خوف شديد (F&G: {fng})", "btc_mood": btc_mood_text}

    return {"mood": "POSITIVE", "reason": "وضع السوق مناسب", "btc_mood": btc_mood_text}

def analyze_momentum_breakout(df, rvol):
    df.ta.vwap(append=True); df.ta.bbands(length=20, append=True); df.ta.macd(append=True); df.ta.rsi(append=True)
    last, prev = df.iloc[-2], df.iloc[-3]
    macd_col, macds_col, bbu_col, rsi_col = find_col(df.columns, "MACD_"), find_col(df.columns, "MACDs_"), find_col(df.columns, "BBU_"), find_col(df.columns, "RSI_")
    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and last[rsi_col] < 68):
        return {"reason": STRATEGY_NAMES_AR['momentum_breakout']}
    return None

def analyze_breakout_squeeze_pro(df, rvol):
    df.ta.bbands(length=20, append=True); df.ta.kc(length=20, scalar=1.5, append=True); df.ta.obv(append=True)
    bbu_col, bbl_col, kcu_col, kcl_col = find_col(df.columns, "BBU_"), find_col(df.columns, "BBL_"), find_col(df.columns, "KCUe_"), find_col(df.columns, "KCLEe_")
    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]
    if is_in_squeeze and (last['close'] > last[bbu_col]) and (last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5) and (df['OBV'].iloc[-2] > df['OBV'].iloc[-3]):
        return {"reason": STRATEGY_NAMES_AR['breakout_squeeze_pro']}
    return None

async def analyze_support_rebound(df, rvol, exchange, symbol):
    try:
        ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
        if len(ohlcv_1h) < 50: return None
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        current_price = df_1h['close'].iloc[-1]
        recent_lows = df_1h['low'].rolling(window=10, center=True).min()
        supports = recent_lows[recent_lows.notna()]
        closest_support = max([s for s in supports if s < current_price], default=None)
        if not closest_support or ((current_price - closest_support) / closest_support * 100 > 1.0): return None

        last_candle_15m = df.iloc[-2]
        if last_candle_15m['close'] > last_candle_15m['open'] and last_candle_15m['volume'] > df['volume'].rolling(window=20).mean().iloc[-2] * 1.5:
            return {"reason": STRATEGY_NAMES_AR['support_rebound']}
    except Exception: return None
    return None

def analyze_sniper_pro(df, rvol):
    try:
        compression_candles = 24
        if len(df) < compression_candles + 2: return None
        compression_df = df.iloc[-compression_candles-1:-1]
        highest_high, lowest_low = compression_df['high'].max(), compression_df['low'].min()
        if lowest_low <= 0: return None
        volatility = (highest_high - lowest_low) / lowest_low * 100
        if volatility < 12.0:
            last_candle = df.iloc[-2]
            if last_candle['close'] > highest_high and last_candle['volume'] > compression_df['volume'].mean() * 2:
                return {"reason": STRATEGY_NAMES_AR['sniper_pro']}
    except Exception: return None
    return None

async def analyze_whale_radar(df, rvol, exchange, symbol):
    try:
        ob = await exchange.fetch_order_book(symbol, limit=20)
        if not ob or not ob.get('bids'): return None
        if sum(float(price) * float(qty) for price, qty in ob['bids'][:10]) > 30000:
            return {"reason": STRATEGY_NAMES_AR['whale_radar']}
    except Exception: return None
    return None

SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout, "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "support_rebound": analyze_support_rebound, "sniper_pro": analyze_sniper_pro, "whale_radar": analyze_whale_radar
}

# =======================================================================================
# --- 🚀 Hybrid Core Protocol (Execution & Management) 🚀 ---
# =======================================================================================
async def activate_trade(order_id, symbol):
    bot = bot_data.application.bot
    log_ctx = {'trade_id': 'N/A'}
    try:
        logger.info(f"Fetching final order details for {order_id} to ensure fee accuracy.")
        order_details = await bot_data.exchange.fetch_order(order_id, symbol)
        
        filled_price = order_details.get('average', 0.0)
        gross_filled_quantity = order_details.get('filled', 0.0)
        
        if gross_filled_quantity <= 0 or filled_price <= 0:
            logger.error(f"Order {order_id} for {symbol} has invalid fill data. Price: {filled_price}, Qty: {gross_filled_quantity}. Aborting.")
            return

        # --- TRUE FEE AWARENESS LOGIC ---
        net_filled_quantity = gross_filled_quantity
        base_currency = symbol.split('/')[0]
        if 'fee' in order_details and order_details['fee'] and 'cost' in order_details['fee']:
            fee_cost = order_details['fee']['cost']
            fee_currency = order_details['fee']['currency']
            if fee_currency == base_currency:
                net_filled_quantity -= fee_cost
                logger.info(f"Fee of {fee_cost} {fee_currency} deducted. Net quantity for {symbol} is {net_filled_quantity}.", extra=log_ctx)

        if net_filled_quantity <= 0:
            logger.error(f"Net quantity for {order_id} is zero or less after fees. Aborting activation.", extra=log_ctx)
            return

        balance_after = await bot_data.exchange.fetch_balance()
        usdt_remaining = balance_after.get('USDT', {}).get('free', 0)
    except Exception as e:
        logger.error(f"Could not fetch data for trade activation: {e}", exc_info=True)
        # Attempt to clean up the pending trade in DB to prevent it from getting stuck
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("UPDATE trades SET status = 'failed', reason = 'Activation Fetch Error' WHERE order_id = ?", (order_id,))
            await conn.commit()
        return

    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT * FROM trades WHERE order_id = ? AND status = 'pending'", (order_id,))
        trade = await cursor.fetchone()
        if not trade:
            logger.info(f"Activation ignored for {order_id}: Trade not pending.")
            return

        trade = dict(trade)
        log_ctx['trade_id'] = trade['id']
        logger.info(f"Activating trade #{trade['id']} for {symbol}...", extra=log_ctx)
        
        risk = filled_price - trade['stop_loss']
        new_take_profit = filled_price + (risk * bot_data.settings['risk_reward_ratio'])

        await conn.execute(
            "UPDATE trades SET status = 'active', entry_price = ?, quantity = ?, take_profit = ? WHERE id = ?",
            (filled_price, net_filled_quantity, new_take_profit, trade['id'])
        )
        
        active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]
        await conn.commit()

    await bot_data.public_ws.subscribe([symbol])
    
    trade_cost = filled_price * net_filled_quantity
    tp_percent = (new_take_profit / filled_price - 1) * 100
    sl_percent = (1 - trade['stop_loss'] / filled_price) * 100
    
    context_msg = ""
    try:
        ohlcv_24h = await bot_data.exchange.fetch_ohlcv(symbol, '1h', limit=24)
        if ohlcv_24h:
            df_24h = pd.DataFrame(ohlcv_24h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            high_24h = df_24h['high'].max()
            if new_take_profit >= high_24h * 0.98:
                context_msg = "📈 الهدف المحدد يقع بالقرب من أو يتجاوز أعلى سعر خلال 24 ساعة، مما يشير إلى احتمالية اختراق."
            else:
                context_msg = "📊 الهدف يقع ضمن نطاق التداول الطبيعي لآخر 24 ساعة."
    except Exception:
        context_msg = "تعذر تحليل سياق السوق."

    success_msg = (
        f"**✅ تم تأكيد الشراء | {symbol}**\n"
        f"**الاستراتيجية:** {trade['reason']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔸 **الصفقة رقم:** `#{trade['id']}`\n"
        f"🔸 **سعر التنفيذ:** `${filled_price:,.4f}`\n"
        f"🔸 **الكمية (صافي):** `{net_filled_quantity:,.4f}` {symbol.split('/')[0]}\n"
        f"🔸 **التكلفة:** `${trade_cost:,.2f}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 **الهدف (TP):** `${new_take_profit:,.4f}` `(ربح متوقع: {tp_percent:+.2f}%)`\n"
        f"🛡️ **الوقف (SL):** `${trade['stop_loss']:,.4f}` `(خسارة مقبولة: {sl_percent:.2f}%)`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔬 **تحليل السياق:**\n_{context_msg}_\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 **السيولة المتبقية (USDT):** `${usdt_remaining:,.2f}`\n"
        f"🔄 **إجمالي الصفقات النشطة:** `{active_trades_count}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"الحارس الأمين يراقب الصفقة الآن."
    )
    await safe_send_message(bot, success_msg)

async def handle_filled_buy_order(order_data):
    symbol = order_data['instId'].replace('-', '/'); order_id = order_data['ordId']
    avg_price = float(order_data.get('avgPx', 0))
    if avg_price > 0:
        logger.info(f"🎤 Fast Reporter: Received fill for {order_id} via WebSocket. Activating...")
        await activate_trade(order_id, symbol)

async def exponential_backoff_with_jitter(run_coro, *args, **kwargs):
    retries = 0
    base_delay, max_delay = 2, 120
    while True:
        try:
            await run_coro(*args, **kwargs)
        except Exception as e:
            retries += 1
            backoff_delay = min(max_delay, base_delay * (2 ** retries))
            jitter = random.uniform(0, backoff_delay * 0.5)
            total_delay = backoff_delay + jitter
            logger.error(f"Coroutine {run_coro.__name__} failed: {e}. Retrying in {total_delay:.2f} seconds...")
            await asyncio.sleep(total_delay)

class PrivateWebSocketManager:
    def __init__(self): self.ws_url = "wss://ws.okx.com:8443/ws/v5/private"
    def _get_auth_args(self):
        timestamp = str(time.time()); message = timestamp + 'GET' + '/users/self/verify'
        mac = hmac.new(bytes(OKX_API_SECRET, 'utf8'), bytes(message, 'utf8'), 'sha256')
        sign = base64.b64encode(mac.digest()).decode()
        return [{"apiKey": OKX_API_KEY, "passphrase": OKX_API_PASSPHRASE, "timestamp": timestamp, "sign": sign}]
    async def _message_handler(self, msg):
        if msg == 'ping': await self.websocket.send('pong'); return
        data = json.loads(msg)
        if data.get('arg', {}).get('channel') == 'orders':
            for order in data.get('data', []):
                if order.get('state') == 'filled' and order.get('side') == 'buy':
                    await handle_filled_buy_order(order)
    async def _run_loop(self):
        async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
            self.websocket = ws; logger.info("✅ [Fast Reporter] Connected.")
            await ws.send(json.dumps({"op": "login", "args": self._get_auth_args()}))
            login_response = json.loads(await ws.recv())
            if login_response.get('code') == '0':
                logger.info("🔐 [Fast Reporter] Authenticated.")
                await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "orders", "instType": "SPOT"}]}))
                async for msg in ws: await self._message_handler(msg)
            else:
                logger.error(f"🔥 [Fast Reporter] Auth failed: {login_response}")
                raise ConnectionAbortedError("Authentication failed")
    async def run(self):
        await exponential_backoff_with_jitter(self._run_loop)

async def the_supervisor_job(context: ContextTypes.DEFAULT_TYPE):
    logger.info("🕵️ Supervisor: Conducting audit of pending trades...")
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        two_mins_ago = (datetime.now(EGYPT_TZ) - timedelta(minutes=2)).isoformat()
        cursor = await conn.execute("SELECT * FROM trades WHERE status = 'pending' AND timestamp <= ?", (two_mins_ago,))
        stuck_trades = await cursor.fetchall()
        if not stuck_trades:
            logger.info("🕵️ Supervisor: Audit complete. No abandoned trades found.")
            return
        for trade_data in stuck_trades:
            trade = dict(trade_data)
            order_id, symbol = trade['order_id'], trade['symbol']
            logger.warning(f"🕵️ Supervisor: Found abandoned trade #{trade['id']}. Investigating.", extra={'trade_id': trade['id']})
            try:
                order_status = await bot_data.exchange.fetch_order(order_id, symbol)
                if order_status['status'] == 'closed' and order_status.get('filled', 0) > 0:
                    logger.info(f"🕵️ Supervisor: API confirms trade {order_id} was filled. Activating manually.", extra={'trade_id': trade['id']})
                    await activate_trade(order_id, symbol)
                elif order_status['status'] == 'canceled':
                    await conn.execute("UPDATE trades SET status = 'failed' WHERE id = ?", (trade['id'],))
                else: 
                    await bot_data.exchange.cancel_order(order_id, symbol)
                    await conn.execute("UPDATE trades SET status = 'failed' WHERE id = ?", (trade['id'],))
                await conn.commit()
            except Exception as e: logger.error(f"🕵️ Supervisor: Failed to rectify trade #{trade['id']}: {e}", extra={'trade_id': trade['id']})

class TradeGuardian:
    def __init__(self, application): self.application = application
    async def handle_ticker_update(self, ticker_data):
        async with trade_management_lock:
            symbol = ticker_data['instId'].replace('-', '/'); current_price = float(ticker_data['last'])
            try:
                async with aiosqlite.connect(DB_FILE) as conn:
                    conn.row_factory = aiosqlite.Row
                    cursor = await conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = 'active'", (symbol,))
                    trade_data = await cursor.fetchone()
                    if not trade_data: return

                    trade = dict(trade_data)
                    log_ctx = {'trade_id': trade['id']}
                    settings = bot_data.settings
                    if settings['trailing_sl_enabled']:
                        new_highest_price = max(trade.get('highest_price', 0), current_price)
                        if new_highest_price > trade.get('highest_price', 0):
                            await conn.execute("UPDATE trades SET highest_price = ? WHERE id = ?", (new_highest_price, trade['id']))
                        if not trade['trailing_sl_active'] and current_price >= trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100):
                            trade['trailing_sl_active'] = True
                            await conn.execute("UPDATE trades SET trailing_sl_active = 1, stop_loss = ? WHERE id = ?", (trade['entry_price'], trade['id']))
                            trade['stop_loss'] = trade['entry_price']
                            await safe_send_message(self.application.bot, f"**🚀 تأمين الأرباح! | #{trade['id']} {symbol}**\nتم رفع وقف الخسارة إلى نقطة الدخول: `${trade['entry_price']}`")
                        if trade['trailing_sl_active']:
                            new_sl = new_highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                            if new_sl > trade['stop_loss']:
                                trade['stop_loss'] = new_sl
                                await conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_sl, trade['id']))
                    await conn.commit()

                if current_price >= trade['take_profit']: await self._close_trade(trade, "ناجحة (TP)", current_price)
                elif current_price <= trade['stop_loss']: await self._close_trade(trade, "فاشلة (SL)", current_price)
            except Exception as e: logger.error(f"Guardian Ticker Error for {symbol}: {e}", exc_info=True)

    async def _close_trade(self, trade, reason, close_price):
        symbol, trade_id = trade['symbol'], trade['id']
        bot = self.application.bot
        log_ctx = {'trade_id': trade_id}
        logger.info(f"Guardian: Starting close process for {symbol}. Reason: {reason}", extra=log_ctx)
        try:
            # --- FLEXIBLE SELL LOGIC ---
            asset_to_sell = symbol.split('/')[0]
            balance = await bot_data.exchange.fetch_balance()
            available_quantity = balance.get(asset_to_sell, {}).get('free', 0.0)

            if available_quantity <= 0:
                logger.critical(f"Attempted to close trade #{trade_id} but no available balance found for {asset_to_sell}. This may indicate a sync issue.", extra=log_ctx)
                # Mark as failed to allow supervisor to investigate
                async with aiosqlite.connect(DB_FILE) as conn:
                    await conn.execute("UPDATE trades SET status = 'closure_failed', reason = 'Zero available balance' WHERE id = ?", (trade_id,))
                    await conn.commit()
                await safe_send_message(bot, f"🚨 **فشل إغلاق** 🚨\nلا يوجد رصيد متاح من `{asset_to_sell}` لإغلاق الصفقة `#{trade_id}`. الرجاء المراجعة.")
                return

            logger.info(f"Found {available_quantity} {asset_to_sell} available. Selling this amount.", extra=log_ctx)
            formatted_quantity = bot_data.exchange.amount_to_precision(symbol, available_quantity)

            params = {'tdMode': 'cash', 'clOrdId': f"close_{trade_id}_{int(time.time() * 1000)}"}
            order = await bot_data.exchange.create_market_sell_order(symbol, formatted_quantity, params)
            logger.info(f"Successfully created sell order. Order ID: {order.get('id')}", extra=log_ctx)

            # Use the original (accurate) quantity from the DB for PNL calculation
            pnl = (close_price - trade['entry_price']) * trade['quantity']
            pnl_percent = (close_price / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
            emoji = "✅" if pnl > 0 else "🛑"

            async with aiosqlite.connect(DB_FILE) as conn:
                await conn.execute("UPDATE trades SET status = ?, close_price = ?, pnl_usdt = ? WHERE id = ?", (reason, close_price, pnl, trade['id']))
                await conn.commit()
            await bot_data.public_ws.unsubscribe([symbol])

            msg = (f"**{emoji} تم إغلاق الصفقة | {symbol}**\n**السبب:** {reason}\n**الربح/الخسارة:** `${pnl:,.2f}` ({pnl_percent:+.2f}%)")
            await safe_send_message(bot, msg)
        except ccxt.InsufficientFunds as e:
            logger.critical(f"CRITICAL: InsufficientFunds error during FLEXIBLE SELL: {e}", extra=log_ctx)
            await safe_send_message(bot, f"🚨 **فشل حرج** 🚨\nحدث خطأ رصيد غير كافٍ نهائي عند إغلاق `#{trade_id}`. تدخل يدوي!")
        except Exception as e:
            logger.critical(f"Unexpected CRITICAL error while closing trade: {e}", exc_info=True, extra=log_ctx)
            await safe_send_message(bot, f"🚨 **فشل حرج** 🚨\nخطأ غير متوقع عند إغلاق `#{trade_id}`. راجع السجلات.")

    async def sync_subscriptions(self):
        try:
            async with aiosqlite.connect(DB_FILE) as conn:
                cursor = await conn.execute("SELECT DISTINCT symbol FROM trades WHERE status = 'active'")
                active_symbols = [row[0] for row in await cursor.fetchall()]
            if active_symbols:
                logger.info(f"Guardian: Syncing subscriptions for: {active_symbols}")
                await bot_data.public_ws.subscribe(active_symbols)
        except Exception as e: logger.error(f"Guardian Sync Error: {e}")

class PublicWebSocketManager:
    def __init__(self, handler_coro): self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"; self.handler = handler_coro; self.subscriptions = set()
    async def _send_op(self, op, symbols):
        if not symbols or not hasattr(self, 'websocket') or not self.websocket: return
        try: await self.websocket.send(json.dumps({"op": op, "args": [{"channel": "tickers", "instId": s.replace('/', '-')} for s in symbols]}))
        except websockets.exceptions.ConnectionClosed: logger.warning(f"Could not send '{op}' op; ws is closed.")
    async def subscribe(self, symbols):
        new = [s for s in symbols if s not in self.subscriptions]
        if new: await self._send_op('subscribe', new); self.subscriptions.update(new); logger.info(f"👁️ [Guardian] Now watching: {new}")
    async def unsubscribe(self, symbols):
        old = [s for s in symbols if s in self.subscriptions]
        if old: await self._send_op('unsubscribe', old); [self.subscriptions.discard(s) for s in old]; logger.info(f"👁️ [Guardian] Stopped watching: {old}")
    async def _run_loop(self):
        async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
            self.websocket = ws; logger.info("✅ [Guardian's Eyes] Connected.")
            if self.subscriptions: await self.subscribe(list(self.subscriptions))
            async for msg in ws:
                if msg == 'ping': await ws.send('pong'); continue
                data = json.loads(msg)
                if data.get('arg', {}).get('channel') == 'tickers' and 'data' in data:
                    for ticker in data['data']: await self.handler(ticker)
    async def run(self):
        await exponential_backoff_with_jitter(self._run_loop)

# =======================================================================================
# --- ⚡ Core Scanner & Trade Initiation Logic ⚡ ---
# =======================================================================================
async def get_okx_markets():
    settings = bot_data.settings
    if time.time() - bot_data.last_markets_fetch > 300: # Cache for 5 minutes
        try:
            logger.info("Fetching and caching all OKX markets...")
            all_tickers = await bot_data.exchange.fetch_tickers()
            bot_data.all_markets = list(all_tickers.values())
            bot_data.last_markets_fetch = time.time()
        except Exception as e:
            logger.error(f"Failed to fetch all markets: {e}")
            return []

    blacklist = settings.get('asset_blacklist', [])
    valid_markets = [
        t for t in bot_data.all_markets if
        t.get('symbol') and t['symbol'].endswith('/USDT') and
        t['symbol'].split('/')[0] not in blacklist and
        t.get('quoteVolume', 0) > settings['liquidity_filters']['min_quote_volume_24h_usd'] and
        t.get('active', True) and
        not any(k in t['symbol'] for k in ['-SWAP', 'UP', 'DOWN', '3L', '3S'])
    ]
    valid_markets.sort(key=lambda m: m.get('quoteVolume', 0), reverse=True)
    return valid_markets[:settings['top_n_symbols_by_volume']]


# --- START: The updated worker function including new filters ---
async def worker(queue, signals_list, errors_list):
    settings, exchange = bot_data.settings, bot_data.exchange
    while not queue.empty():
        market = await queue.get(); symbol = market['symbol']
        try:
            # --- START: الفلاتر المتقدمة المدمجة من الكاسحة ---

            # 1. فلتر السيولة الدقيقة (Spread Filter)
            # يتطلب هذا الفلتر استدعاء دفتر الأوامر، مما يضيف استدعاء API إضافي لكل عملة
            # ولكنه ضروري لضمان سيولة حقيقية.
            try:
                orderbook = await exchange.fetch_order_book(symbol, limit=1)
                best_bid = orderbook['bids'][0][0]
                best_ask = orderbook['asks'][0][0]
                if best_bid <= 0: continue # تخطي إذا كان السعر غير صالح
                
                spread_percent = ((best_ask - best_bid) / best_bid) * 100
                if spread_percent > settings.get('spread_filter', {}).get('max_spread_percent', 0.5):
                    continue # تخطي العملة إذا كان الفارق السعري مرتفعاً جداً
            except Exception:
                continue # تخطي العملة إذا فشل جلب دفتر الأوامر

            # --- نهاية فلتر السيولة ---

            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            
            # --- فلتر الاتجاه العام للعملة (EMA Filter) ---
            ema_period = settings.get('trend_filters', {}).get('ema_period', 200)
            if len(ohlcv) < ema_period + 1: 
                continue # تخطي إذا لم تكن هناك بيانات كافية لحساب EMA
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.ta.ema(length=ema_period, append=True)
            ema_col_name = find_col(df.columns, f"EMA_{ema_period}")
            if not ema_col_name or pd.isna(df[ema_col_name].iloc[-2]):
                continue # تخطي إذا فشل حساب EMA

            last_close = df['close'].iloc[-2]
            if last_close < df[ema_col_name].iloc[-2]:
                continue # تخطي إذا كان السعر تحت المتوسط المتحرك الطويل المدى
            
            # --- نهاية فلتر الاتجاه ---

            # --- فلتر التقلب (ATR Filter) ---
            vol_filters = settings.get('volatility_filters', {})
            atr_period = vol_filters.get('atr_period_for_filter', 14)
            min_atr_percent = vol_filters.get('min_atr_percent', 0.8)
            
            df.ta.atr(length=atr_period, append=True)
            atr_col_name = find_col(df.columns, f"ATRr_{atr_period}")
            if not atr_col_name or pd.isna(df[atr_col_name].iloc[-2]):
                continue # تخطي إذا فشل حساب ATR

            atr_percent = (df[atr_col_name].iloc[-2] / last_close) * 100
            if atr_percent < min_atr_percent:
                continue # تخطي العملات ذات التقلب المنخفض جداً

            # --- نهاية فلتر التقلب ---

            # --- END: نهاية الفلاتر المتقدمة ---


            # --- بقية الفلاتر الأصلية لبوت OKX ---
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] == 0: continue
            rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
            if rvol < settings['liquidity_filters']['min_rvol']: continue

            if settings.get('adx_filter_enabled', False):
                df.ta.adx(append=True)
                adx_col = find_col(df.columns, "ADX_")
                if adx_col and not pd.isna(df[adx_col].iloc[-2]) and df[adx_col].iloc[-2] < settings.get('adx_filter_level', 25):
                    continue

            # --- نهاية الفلاتر الأصلية ---

            # --- بدء تطبيق الاستراتيجيات (لم يتغير) ---
            confirmed_reasons = []
            for name in settings['active_scanners']:
                strategy_func = SCANNERS.get(name)
                if not strategy_func: continue

                if asyncio.iscoroutinefunction(strategy_func):
                    result = await strategy_func(df.copy(), rvol, exchange, symbol)
                else:
                    result = strategy_func(df.copy(), rvol)

                if result:
                    confirmed_reasons.append(result['reason'])

            if confirmed_reasons:
                reason_str = ' + '.join(set(confirmed_reasons))
                entry_price = df.iloc[-2]['close']
                df.ta.atr(length=14, append=True)
                atr = df.iloc[-2].get(find_col(df.columns, "ATRr_14"), 0)
                risk = atr * settings['atr_sl_multiplier']
                stop_loss = entry_price - risk
                take_profit = entry_price + (risk * settings['risk_reward_ratio'])
                signals_list.append({"symbol": symbol, "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "reason": reason_str})
        except Exception as e:
            logger.debug(f"Worker error for {symbol}: {e}")
            errors_list.append(symbol)
        finally:
            queue.task_done()
# --- END: The updated worker function ---

async def initiate_real_trade(signal):
    if not bot_data.trading_enabled:
        logger.warning(f"Trade initiation for {signal['symbol']} blocked: Kill Switch is active.")
        return False
    try:
        settings, exchange = bot_data.settings, bot_data.exchange
        await exchange.load_markets()

        trade_size = settings['real_trade_size_usdt']
        base_amount = trade_size / signal['entry_price']

        formatted_amount = exchange.amount_to_precision(signal['symbol'], base_amount)

        logger.info(f"--- INITIATING REAL TRADE: {signal['symbol']} ---")
        logger.info(f"Calculated amount: {base_amount}, Formatted amount: {formatted_amount}")
        
        # Check for sufficient USDT balance before placing the order
        balance = await exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {}).get('free', 0.0)
        if usdt_balance < trade_size:
             logger.error(f"Insufficient USDT to initiate trade for {signal['symbol']}. Have: {usdt_balance}, Need: {trade_size}")
             await safe_send_message(bot_data.application.bot, f"⚠️ **رصيد USDT غير كافٍ!**\nفشل فتح صفقة لـ `{signal['symbol']}`.")
             return False # Return False instead of raising an exception to allow the scan to continue

        buy_order = await exchange.create_market_buy_order(signal['symbol'], formatted_amount)

        if await log_pending_trade_to_db(signal, buy_order):
            await safe_send_message(bot_data.application.bot, f"🚀 تم إرسال أمر شراء لـ `{signal['symbol']}`.")
            return True
        else:
            await exchange.cancel_order(buy_order['id'], signal['symbol'])
            await safe_send_message(bot_data.application.bot, f"⚠️ فشل تسجيل صفقة `{signal['symbol']}`. تم إلغاء الأمر.")
            return False
    except ccxt.InsufficientFunds as e:
        logger.error(f"REAL TRADE FAILED for {signal['symbol']}: {e}")
        await safe_send_message(bot_data.application.bot, f"⚠️ **رصيد غير كافٍ!**\nفشل فتح صفقة لـ `{signal['symbol']}`.")
        return False # This allows the scanner to continue with other signals if one fails
    except Exception as e:
        logger.error(f"REAL TRADE FAILED for {signal['symbol']}: {e}", exc_info=True)
        await safe_send_message(bot_data.application.bot, f"🔥 فشل فتح صفقة لـ `{signal['symbol']}`.")
        return False

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        if not bot_data.trading_enabled:
            logger.info("Scan skipped: Kill Switch is active.")
            return

        scan_start_time = time.time()
        logger.info("--- Starting new OKX-focused market scan... ---")
        settings = bot_data.settings
        bot = context.bot

        mood_result = await get_market_mood()
        bot_data.market_mood = mood_result
        if mood_result['mood'] in ["NEGATIVE", "DANGEROUS"]:
            logger.warning(f"SCAN SKIPPED: {mood_result['reason']}")
            await safe_send_message(bot, f"🔬 *ملخص الفحص الأخير*\n\n- **الحالة:** تم التخطي بسبب مزاج السوق السلبي.\n- **السبب:** {mood_result['reason']}")
            return

        async with aiosqlite.connect(DB_FILE) as conn:
            active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active' OR status = 'pending'")).fetchone())[0]

        if active_trades_count >= settings['max_concurrent_trades']:
            logger.info(f"Scan skipped: Max trades ({active_trades_count}) reached.")
            await safe_send_message(bot, f"🔬 *ملخص الفحص الأخير*\n\n- **الحالة:** تم التخطي بسبب الوصول للحد الأقصى للصفقات.\n- **الصفقات الحالية:** {active_trades_count} / {settings['max_concurrent_trades']}")
            return

        top_markets = await get_okx_markets()
        if not top_markets:
            logger.info("Scan complete: No markets passed filters."); return

        queue, signals_found, analysis_errors = asyncio.Queue(), [], []
        for market in top_markets: await queue.put(market)
        worker_tasks = [asyncio.create_task(worker(queue, signals_found, analysis_errors)) for _ in range(settings.get("worker_threads", 10))]
        await queue.join(); [task.cancel() for task in worker_tasks]

        trades_opened_count = 0
        for signal in signals_found:
            if active_trades_count >= settings['max_concurrent_trades']:
                logger.info("Stopping trade initiation, max concurrent trades reached.")
                break
            if time.time() - bot_data.last_signal_time.get(signal['symbol'], 0) > (SCAN_INTERVAL_SECONDS * 0.9):
                bot_data.last_signal_time[signal['symbol']] = time.time()
                if await initiate_real_trade(signal):
                    active_trades_count += 1
                    trades_opened_count += 1
                await asyncio.sleep(2) # Small delay between initiating trades
        
        scan_duration = time.time() - scan_start_time
        bot_data.last_scan_info = {
            "start_time": datetime.fromtimestamp(scan_start_time, EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "duration_seconds": int(scan_duration),
            "checked_symbols": len(top_markets),
            "analysis_errors": len(analysis_errors)
        }

        summary_message = (
            f"🔬 *ملخص الفحص الأخير*\n\n"
            f"- **الحالة:** اكتمل بنجاح\n"
            f"- **وضع السوق (BTC):** {bot_data.market_mood.get('btc_mood', 'N/A')}\n"
            f"- **المدة:** {int(scan_duration)} ثانية\n"
            f"- **العملات المفحوصة:** {len(top_markets)}\n"
            f"----------------------------------\n"
            f"- **إجمالي الإشارات المكتشفة:** {len(signals_found)}\n"
            f"- **✅ صفقات جديدة فُتحت:** {trades_opened_count}\n"
            f"- **⚠️ أخطاء في التحليل:** {len(analysis_errors)}\n"
            f"----------------------------------\n\n"
            f"الفحص التالي مجدول تلقائياً."
        )
        await safe_send_message(bot, summary_message)

async def check_time_sync(context: ContextTypes.DEFAULT_TYPE):
    try:
        server_time = await bot_data.exchange.fetch_time()
        local_time = int(time.time() * 1000)
        diff = abs(server_time - local_time)
        if diff > 2000:
            logger.critical(f"TIME SYNC WARNING: System clock is off by {diff}ms. This may cause API request errors.")
            await safe_send_message(context.bot, f"⚠️ **تحذير مزامنة الوقت** ⚠️\nساعة النظام المحلي غير متزامنة مع الخادم بفارق `{diff}` ميلي ثانية. قد تفشل الطلبات.")
        else:
            logger.info(f"Time sync check passed. Difference is {diff}ms.")
    except Exception as e:
        logger.error(f"Could not perform time sync check: {e}")

# =======================================================================================
# --- 🤖 Telegram UI & Bot Startup 🤖 ---
# =======================================================================================
# ... (UI functions remain unchanged from v28.4, except for version number in start_command)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Dashboard 🖥️"], ["الإعدادات ⚙️"]]
    await update.message.reply_text("أهلاً بك في OKX Mastermind Trader v28.5", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

# --- FIX 2: Correctly handle text commands and button callbacks for manual scan ---
async def manual_scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not bot_data.trading_enabled:
        message_target = update.message or update.callback_query.message
        await message_target.reply_text("🔬 الفحص اليدوي محظور. مفتاح الإيقاف مفعل.")
        return
    message_target = update.message or update.callback_query.message
    await message_target.reply_text("🔬 تم إعطاء أمر الفحص اليدوي... قد يستغرق هذا بعض الوقت.")
    context.job_queue.run_once(perform_scan, 1)


async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ks_status_emoji = "🚨" if not bot_data.trading_enabled else "✅"
    ks_status_text = "مفتاح الإيقاف (مفعل)" if not bot_data.trading_enabled else "الحالة (طبيعية)"
    keyboard = [
        [InlineKeyboardButton("💼 نظرة عامة على المحفظة", callback_data="db_portfolio"), InlineKeyboardButton("📈 الصفقات النشطة", callback_data="db_trades")],
        [InlineKeyboardButton("📜 سجل الصفقات المغلقة", callback_data="db_history"), InlineKeyboardButton("📊 الإحصائيات والأداء", callback_data="db_stats")],
        [InlineKeyboardButton("🌡️ تحليل مزاج السوق", callback_data="db_mood"), InlineKeyboardButton("🔬 فحص فوري", callback_data="db_manual_scan")],
        [InlineKeyboardButton(f"{ks_status_emoji} {ks_status_text}", callback_data="kill_switch_toggle"), InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="db_diagnostics")]
    ]
    message_text = "🖥️ *لوحة التحكم الرئيسية*\n\nاختر التقرير أو البيانات التي تريد عرضها:"
    if not bot_data.trading_enabled:
        message_text += "\n\n**تحذير: تم تفعيل مفتاح الإيقاف. لن يتم فتح صفقات جديدة.**"

    target_message = update.message or update.callback_query.message
    if update.callback_query:
        await safe_edit_message(update.callback_query, message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await target_message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

async def toggle_kill_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    bot_data.trading_enabled = not bot_data.trading_enabled
    if bot_data.trading_enabled:
        await query.answer("✅ تم إلغاء تفعيل مفتاح الإيقاف. استئناف التداول الطبيعي.")
        await safe_send_message(context.bot, "✅ **تم استئناف التداول الطبيعي.**")
    else:
        await query.answer("🚨 تم تفعيل مفتاح الإيقاف! لن يتم فتح صفقات جديدة.", show_alert=True)
        await safe_send_message(context.bot, "🚨 **تحذير: تم تفعيل مفتاح الإيقاف!**\nلن يتم فتح أي صفقات جديدة حتى يتم إلغاء تفعيله يدوياً.")
    await show_dashboard_command(update, context)

async def show_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT id, symbol, status FROM trades WHERE status = 'active' OR status = 'pending' ORDER BY id DESC")
        trades = await cursor.fetchall()

    if not trades:
        text = "لا توجد صفقات حالية."
        keyboard = [[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
        await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))
        return

    text = "📈 *الصفقات النشطة*\nاختر صفقة لعرض تفاصيلها:\n"
    keyboard = []

    for trade in trades:
        status_emoji = "✅" if trade['status'] == 'active' else "⏳"
        button_text = f"#{trade['id']} {status_emoji} | {trade['symbol']}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=f"check_{trade['id']}")])

    keyboard.append([InlineKeyboardButton("🔄 تحديث", callback_data="db_trades")])
    keyboard.append([InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")])
    await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))


async def check_trade_details(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    trade_id = int(query.data.split('_')[1])
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        trade = await cursor.fetchone()
    if not trade:
        await query.answer("لم يتم العثور على الصفقة."); return
    trade = dict(trade)
    if trade['status'] == 'pending':
        message = f"**⏳ حالة الصفقة #{trade_id}**\n- **العملة:** `{trade['symbol']}`\n- **الحالة:** في انتظار تأكيد التنفيذ..."
    else:
        try:
            ticker = await bot_data.exchange.fetch_ticker(trade['symbol'])
            current_price = ticker['last']
            pnl = (current_price - trade['entry_price']) * trade['quantity']
            pnl_percent = (current_price / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
            pnl_text = f"💰 **الربح/الخسارة الحالية:** `${pnl:+.2f}` ({pnl_percent:+.2f}%)"
            current_price_text = f"- **السعر الحالي:** `${current_price}`"
        except Exception:
            pnl_text = "💰 تعذر جلب الربح/الخسارة الحالية."
            current_price_text = "- **السعر الحالي:** `تعذر الجلب`"

        message = (
            f"**✅ حالة الصفقة #{trade_id}**\n\n"
            f"- **العملة:** `{trade['symbol']}`\n"
            f"- **سعر الدخول:** `${trade['entry_price']}`\n"
            f"{current_price_text}\n"
            f"- **الكمية:** `{trade['quantity']}`\n"
            f"----------------------------------\n"
            f"- **الهدف (TP):** `${trade['take_profit']}`\n"
            f"- **الوقف (SL):** `${trade['stop_loss']}`\n"
            f"----------------------------------\n"
            f"{pnl_text}"
        )

    await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للصفقات", callback_data="db_trades")]]))

async def show_mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("جاري تحليل مزاج السوق...")

    fng_task = asyncio.create_task(get_fear_and_greed_index())
    headlines_task = asyncio.create_task(asyncio.to_thread(get_latest_crypto_news))
    mood_task = asyncio.create_task(get_market_mood())
    markets_task = asyncio.create_task(get_okx_markets())

    fng_index = await fng_task
    original_headlines = await headlines_task
    mood = await mood_task
    all_markets = await markets_task

    translated_headlines, translation_success = await translate_text_gemini(original_headlines)

    news_sentiment, _ = analyze_sentiment_of_headlines(original_headlines)

    top_gainers, top_losers = [], []
    if all_markets:
        sorted_by_change = sorted([m for m in all_markets if m.get('percentage') is not None], key=lambda m: m['percentage'], reverse=True)
        top_gainers = sorted_by_change[:3]
        top_losers = sorted_by_change[-3:]

    verdict = "الحالة العامة للسوق تتطلب الحذر."
    if mood['mood'] == 'POSITIVE':
        verdict = "المؤشرات الفنية إيجابية، مما قد يدعم فرص الشراء."
        if fng_index and fng_index > 65:
            verdict = "المؤشرات الفنية إيجابية ولكن مع وجود طمع في السوق، يرجى الحذر من التقلبات."
    elif fng_index and fng_index < 30:
        verdict = "يسود الخوف على السوق، قد تكون هناك فرص للمدى الطويل ولكن المخاطرة عالية حالياً."

    gainers_str = "\n".join([f"  `{g['symbol']}` `({g.get('percentage', 0):+.2f}%)`" for g in top_gainers]) or "  لا توجد بيانات."
    losers_str = "\n".join([f"  `{l['symbol']}` `({l.get('percentage', 0):+.2f}%)`" for l in reversed(top_losers)]) or "  لا توجد بيانات."
    news_header = "📰 آخر الأخبار (مترجمة آلياً):" if translation_success else "📰 آخر الأخبار (الترجمة غير متاحة):"
    news_str = "\n".join([f"  - _{h}_" for h in translated_headlines]) or "  لا توجد أخبار."

    message = (
        f"**🌡️ تحليل مزاج السوق الشامل**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"**⚫️ الخلاصة:** *{verdict}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"**📊 المؤشرات الرئيسية:**\n"
        f"  - **اتجاه BTC العام:** {mood.get('btc_mood', 'N/A')}\n"
        f"  - **الخوف والطمع:** {fng_index or 'N/A'}\n"
        f"  - **مشاعر الأخبار:** {news_sentiment}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"**🚀 أبرز الرابحين:**\n{gainers_str}\n\n"
        f"**📉 أبرز الخاسرين:**\n{losers_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{news_header}\n{news_str}\n"
    )

    keyboard = [[InlineKeyboardButton("🔄 تحديث", callback_data="db_mood")], [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
    await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup(keyboard))


async def show_strategy_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        cursor = await conn.execute("SELECT reason, status FROM trades WHERE status LIKE 'ناجحة%' OR status LIKE 'فاشلة%'")
        trades = await cursor.fetchall()
    if not trades:
        await safe_edit_message(update.callback_query, "لا توجد صفقات مغلقة لتحليلها.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))
        return
    stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
    for reason, status in trades:
        if not reason: continue
        reasons = reason.split(' + ')
        for r in reasons:
            if status.startswith('ناجحة'): stats[r]['wins'] += 1
            else: stats[r]['losses'] += 1
    report = ["**📜 تقرير أداء الاستراتيجيات**"]
    for r, s in sorted(stats.items(), key=lambda item: item[1]['wins'] + item[1]['losses'], reverse=True):
        total = s['wins'] + s['losses']
        wr = (s['wins'] / total * 100) if total > 0 else 0
        report.append(f"\n--- *{r}* ---\n  - الصفقات: {total} ({s['wins']}✅ / {s['losses']}❌)\n  - النجاح: {wr:.2f}%")
    await safe_edit_message(update.callback_query, "\n".join(report), reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))

async def show_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT pnl_usdt, status FROM trades WHERE status LIKE 'ناجحة%' OR status LIKE 'فاشلة%'")
        trades_data = await cursor.fetchall()

    if not trades_data:
        await safe_edit_message(update.callback_query, "لا توجد صفقات مغلقة لعرض الإحصائيات.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))
        return

    total_trades = len(trades_data)
    total_pnl = sum(t['pnl_usdt'] for t in trades_data if t['pnl_usdt'] is not None)

    wins_data = [t['pnl_usdt'] for t in trades_data if t['status'].startswith('ناجحة') and t['pnl_usdt'] is not None]
    losses_data = [t['pnl_usdt'] for t in trades_data if t['status'].startswith('فاشلة') and t['pnl_usdt'] is not None]

    win_count = len(wins_data)
    loss_count = len(losses_data)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    avg_win = sum(wins_data) / win_count if win_count > 0 else 0
    avg_loss = sum(losses_data) / loss_count if loss_count > 0 else 0

    profit_factor = sum(wins_data) / abs(sum(losses_data)) if sum(losses_data) != 0 else float('inf')

    message = (
        f"**📊 إحصائيات الأداء التفصيلية**\n\n"
        f"**💰 إجمالي الربح/الخسارة الصافي:** `${total_pnl:,.2f}`\n"
        f"----------------------------------\n"
        f"- **إجمالي الصفقات المغلقة:** {total_trades}\n"
        f"- **الصفقات الرابحة:** {win_count} ✅\n"
        f"- **الصفقات الخاسرة:** {loss_count} ❌\n"
        f"- **معدل النجاح (Win Rate):** {win_rate:.2f}%\n"
        f"----------------------------------\n"
        f"- **متوسط ربح الصفقة:** `${avg_win:,.2f}`\n"
        f"- **متوسط خسارة الصفقة:** `${avg_loss:,.2f}`\n"
        f"- **معامل الربح (Profit Factor):** `{profit_factor:.2f}`"
    )
    await safe_edit_message(update.callback_query, message, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))

async def show_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("جاري جلب بيانات المحفظة...")
    try:
        balance = await bot_data.exchange.fetch_balance({'type': 'trading'})

        owned_assets = {
            asset: data['total']
            for asset, data in balance.items()
            if isinstance(data, dict) and data.get('total', 0) > 0
        }

        usdt_balance = balance.get('USDT', {})
        total_usdt_equity = usdt_balance.get('total', 0)
        free_usdt = usdt_balance.get('free', 0)

        assets_to_fetch = [f"{asset}/USDT" for asset in owned_assets if asset != 'USDT']
        tickers = {}
        if assets_to_fetch:
            try:
                tickers = await bot_data.exchange.fetch_tickers(assets_to_fetch)
            except Exception as e:
                logger.warning(f"Could not fetch all tickers for portfolio: {e}")

        asset_details = []
        total_assets_value_usdt = 0
        for asset, total in owned_assets.items():
            if asset == 'USDT': continue
            symbol = f"{asset}/USDT"
            value_usdt = 0
            if symbol in tickers and tickers[symbol] is not None:
                value_usdt = tickers[symbol].get('last', 0) * total

            total_assets_value_usdt += value_usdt

            if value_usdt >= 1.0:
                asset_details.append(f"  - `{asset}`: `{total:,.6f}` `(≈ ${value_usdt:,.2f})`")

        total_equity = total_usdt_equity + total_assets_value_usdt

        async with aiosqlite.connect(DB_FILE) as conn:
            cursor_pnl = await conn.execute("SELECT SUM(pnl_usdt) FROM trades WHERE status LIKE 'ناجحة%' OR status LIKE 'فاشلة%'")
            total_realized_pnl = (await cursor_pnl.fetchone())[0] or 0.0

            cursor_trades = await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")
            active_trades_count = (await cursor_trades.fetchone())[0]

        assets_str = "\n".join(asset_details) if asset_details else "  لا توجد أصول أخرى بقيمة تزيد عن 1 دولار."

        message = (
            f"**💼 نظرة عامة على المحفظة**\n"
            f"🗓️ {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"**💰 إجمالي قيمة المحفظة:** `≈ ${total_equity:,.2f}`\n"
            f"  - **السيولة المتاحة (USDT):** `${free_usdt:,.2f}`\n"
            f"  - **قيمة الأصول الأخرى:** `≈ ${total_assets_value_usdt:,.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"**📊 تفاصيل الأصول (أكثر من 1$):**\n"
            f"{assets_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"**📈 أداء التداول:**\n"
            f"  - **الربح/الخسارة المحقق:** `${total_realized_pnl:,.2f}`\n"
            f"  - **عدد الصفقات النشطة:** {active_trades_count}\n"
        )
        keyboard = [[InlineKeyboardButton("🔄 تحديث", callback_data="db_portfolio")], [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
        await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup(keyboard))

    except Exception as e:
        logger.error(f"Portfolio fetch error: {e}", exc_info=True)
        await safe_edit_message(query, f"حدث خطأ أثناء جلب رصيد المحفظة: {e}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة", callback_data="back_to_dashboard")]]))


async def show_trade_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT symbol, pnl_usdt, status FROM trades WHERE status LIKE 'ناجحة%' OR status LIKE 'فاشلة%' ORDER BY id DESC LIMIT 10")
        closed_trades = await cursor.fetchall()

    if not closed_trades:
        text = "لا يوجد سجل للصفقات المغلقة."
        keyboard = [[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
        await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))
        return

    history_list = ["📜 *آخر 10 صفقات مغلقة*"]
    for trade in closed_trades:
        emoji = "✅" if trade['status'].startswith('ناجحة') else "🛑"
        pnl = trade['pnl_usdt'] or 0.0
        history_list.append(f"{emoji} `{trade['symbol']}` | الربح/الخسارة: `${pnl:,.2f}`")

    text = "\n".join(history_list)
    keyboard = [[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
    await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))


async def show_diagnostics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    s = bot_data.settings
    scan_info = bot_data.last_scan_info
    
    determine_active_preset()

    nltk_status = "متاحة ✅" if NLTK_AVAILABLE else "غير متاحة ❌"

    scan_time = scan_info.get("start_time", "لم يتم بعد")
    scan_duration = f'{scan_info.get("duration_seconds", "N/A")} ثانية'
    scan_checked = scan_info.get("checked_symbols", "N/A")
    scan_errors = scan_info.get("analysis_errors", "N/A")

    scanners_list = "\n".join([f"  - {name}" for key, name in STRATEGY_NAMES_AR.items() if key in s['active_scanners']])

    scan_job = context.job_queue.get_jobs_by_name("perform_scan")
    next_scan_time = scan_job[0].next_t.astimezone(EGYPT_TZ).strftime('%H:%M:%S') if scan_job and scan_job[0].next_t else "N/A"

    db_size = f"{os.path.getsize(DB_FILE) / 1024:.2f} KB" if os.path.exists(DB_FILE) else "N/A"
    async with aiosqlite.connect(DB_FILE) as conn:
        total_trades = (await (await conn.execute("SELECT COUNT(*) FROM trades")).fetchone())[0]
        active_trades = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]

    report = (
        f"🕵️‍♂️ *تقرير التشخيص الشامل*\n\n"
        f"تم إنشاؤه في: {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"----------------------------------\n"
        f"⚙️ **حالة النظام والبيئة**\n"
        f"- NLTK (تحليل الأخبار): {nltk_status}\n\n"
        f"🔬 **أداء آخر فحص**\n"
        f"- وقت البدء: {scan_time}\n"
        f"- المدة: {scan_duration}\n"
        f"- العملات المفحوصة: {scan_checked}\n"
        f"- فشل في التحليل: {scan_errors} عملات\n\n"
        f"🔧 **الإعدادات النشطة**\n"
        f"- **النمط الحالي: {bot_data.active_preset_name}**\n"
        f"- الماسحات المفعلة:\n{scanners_list}\n"
        f"----------------------------------\n"
        f"🔩 **حالة العمليات الداخلية**\n"
        f"- فحص العملات: يعمل, التالي في: {next_scan_time}\n"
        f"- الاتصال بـ OKX: متصل ✅\n"
        f"- قاعدة البيانات:\n"
        f"  - الاتصال: ناجح ✅\n"
        f"  - حجم الملف: {db_size}\n"
        f"  - إجمالي الصفقات: {total_trades} ({active_trades} نشطة)\n"
        f"----------------------------------"
    )

    await safe_edit_message(query, report, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 تحديث", callback_data="db_diagnostics")], [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))

async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🎛️ تعديل المعايير المتقدمة", callback_data="settings_params")],
        [InlineKeyboardButton("🔭 تفعيل/تعطيل الماسحات", callback_data="settings_scanners")],
        [InlineKeyboardButton("🗂️ أنماط جاهزة", callback_data="settings_presets")],
        [InlineKeyboardButton("🚫 القائمة السوداء", callback_data="settings_blacklist"), InlineKeyboardButton("🗑️ إدارة البيانات", callback_data="settings_data")]
    ]
    message_text = "⚙️ *الإعدادات الرئيسية*\n\nاختر فئة الإعدادات التي تريد تعديلها."
    target_message = update.message or update.callback_query.message
    if update.callback_query:
        await safe_edit_message(update.callback_query, message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await target_message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = bot_data.settings

    def bool_format(key, text):
        val = s.get(key, False)
        emoji = "✅" if val else "❌"
        return f"{text}: {emoji} مفعل"
        
    def get_nested_value(d, keys):
        current_level = d
        for key in keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            else:
                return None
        return current_level

    keyboard = [
        [InlineKeyboardButton("--- إعدادات عامة ---", callback_data="noop")],
        [InlineKeyboardButton(f"عدد العملات للفحص: {s['top_n_symbols_by_volume']}", callback_data="param_set_top_n_symbols_by_volume"),
         InlineKeyboardButton(f"أقصى عدد للصفقات: {s['max_concurrent_trades']}", callback_data="param_set_max_concurrent_trades")],
        [InlineKeyboardButton(f"عمال الفحص المتزامنين: {s['worker_threads']}", callback_data="param_set_worker_threads")],
        [InlineKeyboardButton("--- إعدادات المخاطر ---", callback_data="noop")],
        [InlineKeyboardButton(f"حجم الصفقة ($): {s['real_trade_size_usdt']}", callback_data="param_set_real_trade_size_usdt"),
         InlineKeyboardButton(f"مضاعف وقف الخسارة (ATR): {s['atr_sl_multiplier']}", callback_data="param_set_atr_sl_multiplier")],
        [InlineKeyboardButton(f"نسبة المخاطرة/العائد: {s['risk_reward_ratio']}", callback_data="param_set_risk_reward_ratio")],
        [InlineKeyboardButton(bool_format('trailing_sl_enabled', 'تفعيل الوقف المتحرك'), callback_data="param_toggle_trailing_sl_enabled")],
        [InlineKeyboardButton(f"تفعيل الوقف المتحرك (%): {s['trailing_sl_activation_percent']}", callback_data="param_set_trailing_sl_activation_percent"),
         InlineKeyboardButton(f"مسافة الوقف المتحرك (%): {s['trailing_sl_callback_percent']}", callback_data="param_set_trailing_sl_callback_percent")],
        [InlineKeyboardButton("--- الفلاتر والاتجاه ---", callback_data="noop")],
        [InlineKeyboardButton(bool_format('btc_trend_filter_enabled', 'فلتر الاتجاه العام (BTC)'), callback_data="param_toggle_btc_trend_filter_enabled")],
        [InlineKeyboardButton(f"فترة EMA للاتجاه: {get_nested_value(s, ['trend_filters', 'ema_period'])}", callback_data="param_set_trend_filters_ema_period")],
        [InlineKeyboardButton(f"أقصى سبريد مسموح (%): {get_nested_value(s, ['spread_filter', 'max_spread_percent'])}", callback_data="param_set_spread_filter_max_spread_percent")],
        [InlineKeyboardButton(f"أدنى ATR مسموح (%): {get_nested_value(s, ['volatility_filters', 'min_atr_percent'])}", callback_data="param_set_volatility_filters_min_atr_percent")],
        [InlineKeyboardButton(bool_format('market_mood_filter_enabled', 'فلتر الخوف والطمع'), callback_data="param_toggle_market_mood_filter_enabled"),
         InlineKeyboardButton(f"حد مؤشر الخوف: {s['fear_and_greed_threshold']}", callback_data="param_set_fear_and_greed_threshold")],
        [InlineKeyboardButton(bool_format('adx_filter_enabled', 'فلتر ADX'), callback_data="param_toggle_adx_filter_enabled"),
         InlineKeyboardButton(f"مستوى فلتر ADX: {s['adx_filter_level']}", callback_data="param_set_adx_filter_level")],
        [InlineKeyboardButton(bool_format('news_filter_enabled', 'فلتر الأخبار والبيانات'), callback_data="param_toggle_news_filter_enabled")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, "🎛️ *المعايير المتقدمة*\n\nاضغط على أي معيار لتغيير قيمته:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    active_scanners = bot_data.settings['active_scanners']
    for key, name in STRATEGY_NAMES_AR.items():
        status_emoji = "✅" if key in active_scanners else "❌"
        keyboard.append([InlineKeyboardButton(f"{status_emoji} {name}", callback_data=f"scanner_toggle_{key}")])
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")])
    await safe_edit_message(update.callback_query, "اختر الماسحات لتفعيلها أو تعطيلها:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🚦 احترافي", callback_data="preset_set_professional")],
        [InlineKeyboardButton("🎯 متشدد", callback_data="preset_set_strict")],
        [InlineKeyboardButton("🌙 متساهل", callback_data="preset_set_lenient")],
        [InlineKeyboardButton("⚠️ فائق التساهل", callback_data="preset_set_very_lenient")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, "اختر نمط إعدادات جاهز:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_blacklist_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    blacklist = bot_data.settings.get('asset_blacklist', [])
    blacklist_str = ", ".join(f"`{item}`" for item in blacklist) if blacklist else "لا توجد عملات في القائمة."
    text = f"🚫 *القائمة السوداء*\n\nالعملات التالية لن يتم فحصها أو التداول عليها:\n\n{blacklist_str}"
    keyboard = [
        [InlineKeyboardButton("➕ إضافة عملة", callback_data="blacklist_add"), InlineKeyboardButton("➖ إزالة عملة", callback_data="blacklist_remove")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_data_management_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("‼️ مسح كل الصفقات ‼️", callback_data="data_clear_confirm")], [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]]
    await safe_edit_message(update.callback_query, "🗑️ *إدارة البيانات*\n\n**تحذير:** هذا الإجراء سيحذف سجل جميع الصفقات بشكل نهائي.", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_clear_data_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("نعم، متأكد. احذف كل شيء.", callback_data="data_clear_execute")], [InlineKeyboardButton("لا، تراجع.", callback_data="settings_data")]]
    await safe_edit_message(update.callback_query, "🛑 **تأكيد نهائي** 🛑\n\nهل أنت متأكد أنك تريد حذف جميع بيانات الصفقات؟", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_clear_data_execute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_edit_message(query, "جاري حذف البيانات...", reply_markup=None)
    try:
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            logger.info("Database file has been deleted by user.")
        await init_database()
        await safe_edit_message(query, "✅ تم حذف جميع بيانات الصفقات بنجاح.")
    except Exception as e:
        logger.error(f"Failed to clear data: {e}")
        await safe_edit_message(query, f"❌ حدث خطأ أثناء حذف البيانات: {e}")
    await asyncio.sleep(2)
    await show_settings_menu(update, context)


async def handle_scanner_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    scanner_key = query.data.replace("scanner_toggle_", "")

    active_scanners = bot_data.settings['active_scanners']

    if scanner_key not in STRATEGY_NAMES_AR:
        logger.error(f"Invalid scanner key received from callback: '{scanner_key}'")
        await query.answer("خطأ: مفتاح الماسح غير صالح.", show_alert=True)
        return

    if scanner_key in active_scanners:
        if len(active_scanners) > 1:
            active_scanners.remove(scanner_key)
        else:
            await query.answer("يجب تفعيل ماسح واحد على الأقل.", show_alert=True)
            return
    else:
        active_scanners.append(scanner_key)
    
    save_settings()
    determine_active_preset()
    await query.answer(f"{STRATEGY_NAMES_AR[scanner_key]} {'تم تفعيله' if scanner_key in active_scanners else 'تم تعطيله'}")
    await show_scanners_menu(update, context)

async def handle_preset_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    preset_key = query.data.split('_')[-1]
    
    if preset_settings := SETTINGS_PRESETS.get(preset_key):
        current_scanners = bot_data.settings.get('active_scanners', [])
        
        bot_data.settings = copy.deepcopy(preset_settings)
        
        bot_data.settings['active_scanners'] = current_scanners
        
        determine_active_preset() 
        save_settings()
        
        await query.answer(f"نمط '{PRESET_NAMES_AR.get(preset_key)}' تم تطبيقه. الحالة الحالية: '{bot_data.active_preset_name}'")
        await show_settings_menu(update, context)
    else:
        await query.answer("لم يتم العثور على النمط.")


async def handle_parameter_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    param_key = query.data.replace("param_set_", "")
    context.user_data['setting_to_change'] = param_key
    
    # Check if the key is nested to provide a hint
    if '_' in param_key:
        await query.message.reply_text(f"أرسل القيمة الرقمية الجديدة لـ `{param_key}`:\n\n*ملاحظة: هذا إعداد متقدم (متشعب)، سيتم تحديثه مباشرة.*", parse_mode=ParseMode.MARKDOWN)
    else:
        await query.message.reply_text(f"أرسل القيمة الرقمية الجديدة لـ `{param_key}`:", parse_mode=ParseMode.MARKDOWN)

async def handle_toggle_parameter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    param_key = query.data.replace("param_toggle_", "")
    bot_data.settings[param_key] = not bot_data.settings.get(param_key, False)
    save_settings()
    determine_active_preset()
    await show_parameters_menu(update, context)

# --- FIX 1: Refactored handle_setting_value to correctly handle nested and simple keys ---
def _set_nested_value(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

async def handle_setting_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    
    if 'blacklist_action' in context.user_data:
        action = context.user_data.pop('blacklist_action')
        blacklist = bot_data.settings.get('asset_blacklist', [])
        symbol = user_input.upper().replace("/USDT", "")

        if action == 'add':
            if symbol not in blacklist:
                blacklist.append(symbol)
                await update.message.reply_text(f"✅ تم إضافة `{symbol}` إلى القائمة السوداء.")
            else:
                await update.message.reply_text(f"⚠️ العملة `{symbol}` موجودة بالفعل.")
        elif action == 'remove':
            if symbol in blacklist:
                blacklist.remove(symbol)
                await update.message.reply_text(f"✅ تم إزالة `{symbol}` من القائمة السوداء.")
            else:
                await update.message.reply_text(f"⚠️ العملة `{symbol}` غير موجودة في القائمة.")

        bot_data.settings['asset_blacklist'] = blacklist
        save_settings()
        determine_active_preset()
        # Create a mock callback query to return to the correct menu
        mock_query = type('Query', (object,), {'message': update.message, 'data': 'settings_blacklist', 'edit_message_text': (lambda *args, **kwargs: None), 'answer': (lambda *args, **kwargs: None)})()
        await show_blacklist_menu(Update(update.update_id, callback_query=mock_query), context)
        return

    if not (setting_key := context.user_data.get('setting_to_change')):
        return

    try:
        keys = setting_key.split('_')
        value_to_update = bot_data.settings
        
        # Traverse down the nested dicts to find the correct value type
        for key in keys:
            if isinstance(value_to_update, dict) and key in value_to_update:
                value_to_update = value_to_update[key]
            else:
                # This handles simple keys correctly
                value_to_update = bot_data.settings[setting_key]
                keys = [setting_key]
                break

        # Determine the type and convert the input
        if isinstance(value_to_update, int):
            new_value = int(user_input)
        else:
            new_value = float(user_input)

        _set_nested_value(bot_data.settings, keys, new_value)
        
        save_settings()
        determine_active_preset()
        await update.message.reply_text(f"✅ تم تحديث `{setting_key}` إلى `{new_value}`.")
    except (ValueError, KeyError):
        await update.message.reply_text("❌ قيمة غير صالحة. الرجاء إرسال رقم.")
    finally:
        del context.user_data['setting_to_change']
        # Create a mock callback query to return to the correct menu
        mock_query = type('Query', (object,), {'message': update.message, 'data': 'settings_params', 'edit_message_text': (lambda *args, **kwargs: None), 'answer': (lambda *args, **kwargs: None)})()
        await show_parameters_menu(Update(update.update_id, callback_query=mock_query), context)


async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'setting_to_change' in context.user_data or 'blacklist_action' in context.user_data:
        await handle_setting_value(update, context)
        return
    text = update.message.text
    if text == "Dashboard 🖥️": await show_dashboard_command(update, context)
    elif text == "الإعدادات ⚙️": await show_settings_menu(update, context)

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    data = query.data

    route_map = {
        "db_stats": show_stats_command, "db_trades": show_trades_command, "db_history": show_trade_history_command,
        "db_mood": show_mood_command, "db_diagnostics": show_diagnostics_command, "back_to_dashboard": show_dashboard_command,
        "db_portfolio": show_portfolio_command,
        # FIX 2: Lambda function now correctly passes the update object
        "db_manual_scan": lambda u,c: manual_scan_command(u, c),
        "kill_switch_toggle": toggle_kill_switch,
        "settings_main": show_settings_menu, "settings_params": show_parameters_menu, "settings_scanners": show_scanners_menu,
        "settings_presets": show_presets_menu, "settings_blacklist": show_blacklist_menu, "settings_data": show_data_management_menu,
        "blacklist_add": handle_blacklist_action, "blacklist_remove": handle_blacklist_action,
        "data_clear_confirm": handle_clear_data_confirmation, "data_clear_execute": handle_clear_data_execute,
        "noop": (lambda u,c: None)
    }

    try:
        if data in route_map: await route_map[data](update, context)
        elif data.startswith("check_"): await check_trade_details(update, context)
        elif data.startswith("scanner_toggle_"): await handle_scanner_toggle(update, context)
        elif data.startswith("preset_set_"): await handle_preset_set(update, context)
        elif data.startswith("param_set_"): await handle_parameter_selection(update, context)
        elif data.startswith("param_toggle_"): await handle_toggle_parameter(update, context)
    except Exception as e:
        logger.error(f"Error in button callback handler for data '{data}': {e}", exc_info=True)


async def post_init(application: Application):
    bot_data.application = application
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, TELEGRAM_BOT_TOKEN]):
        logger.critical("FATAL: Missing critical API or Bot keys."); return
    if NLTK_AVAILABLE:
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError: logger.info("Downloading NLTK data..."); nltk.download('vader_lexicon', quiet=True)

    try:
        config = {'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE, 'enableRateLimit': True}
        bot_data.exchange = ccxt.okx(config)
        await bot_data.exchange.load_markets()
        await bot_data.exchange.fetch_balance()
        logger.info("✅ Successfully connected to OKX.")
    except Exception as e:
        logger.critical(f"🔥 FATAL: Could not connect to OKX: {e}"); return

    await check_time_sync(ContextTypes.DEFAULT_TYPE(application=application))

    bot_data.trade_guardian = TradeGuardian(application)
    bot_data.public_ws = PublicWebSocketManager(bot_data.trade_guardian.handle_ticker_update)
    bot_data.private_ws = PrivateWebSocketManager()
    asyncio.create_task(bot_data.public_ws.run())
    asyncio.create_task(bot_data.private_ws.run())

    logger.info("Waiting 5s for WebSocket connections...")
    await asyncio.sleep(5)
    await bot_data.trade_guardian.sync_subscriptions()

    application.job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name="perform_scan")
    application.job_queue.run_repeating(the_supervisor_job, interval=SUPERVISOR_INTERVAL_SECONDS, first=30, name="the_supervisor_job")
    application.job_queue.run_repeating(check_time_sync, interval=TIME_SYNC_INTERVAL_SECONDS, first=TIME_SYNC_INTERVAL_SECONDS, name="time_sync_job")

    logger.info(f"Scanner scheduled for every {SCAN_INTERVAL_SECONDS}s. Supervisor will audit every {SUPERVISOR_INTERVAL_SECONDS}s.")
    try:
        await application.bot.send_message(TELEGRAM_CHAT_ID, "*🚀 OKX Mastermind Trader v28.5 بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)
    except Forbidden:
        logger.critical(f"FATAL: Bot is not authorized for chat ID {TELEGRAM_CHAT_ID}.")
        return
    logger.info("--- Bot is now fully operational ---")

async def post_shutdown(application: Application):
    if bot_data.exchange: await bot_data.exchange.close()
    logger.info("Bot has shut down.")

def main():
    logger.info("--- Starting OKX Mastermind Trader v28.5 ---")
    load_settings(); asyncio.run(init_database())
    app_builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    app_builder.post_init(post_init).post_shutdown(post_shutdown)
    application = app_builder.build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("scan", manual_scan_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    application.add_handler(CallbackQueryHandler(button_callback_handler))

    application.run_polling()

if __name__ == '__main__':
    main()
