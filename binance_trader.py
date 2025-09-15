# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 بوت OKX القناص v5.3 (The Mastermind - Stable) - النسخة النهائية المتكاملة 🚀 ---
# =======================================================================================
# هذا الإصدار هو إصدار تصحيحي شامل بناءً على الملاحظات الحية:
# - [إصلاح حاسم] إصلاح الخطأ البرمجي (AttributeError) الذي كان يمنع بدء تشغيل البوت.
# - [إكمال الميزات] برمجة زر "تقرير أداء الاستراتيجيات" و "تقرير التشخيص" ليعملا بشكل كامل ومستقر.
# - [إضافة] إضافة "حالة اتصال المنصة" إلى تقرير التشخيص.
# - [تحسينات] مراجعة شاملة للكود لضمان الاستقرار والموثوقية.
#
# للتثبيت: pip install "ccxt[async]" pandas pandas-ta python-telegram-bot httpx feedparser nltk
# =======================================================================================

# --- المكتبات المطلوبة ---
import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import asyncio
import os
import logging
import json
import time
import types
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict, Counter
import sqlite3
import httpx
import feedparser

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest

# =======================================================================================
# --- ⚙️ الإعدادات الأساسية ⚙️ ---
# =======================================================================================
OKX_API_KEY = os.getenv('OKX_API_KEY', 'YOUR_OKX_API_KEY')
OKX_API_SECRET = os.getenv('OKX_API_SECRET', 'YOUR_OKX_API_SECRET')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE', 'YOUR_OKX_PASSPHRASE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID')

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'okx_mastermind_v5.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'okx_mastermind_settings_v5.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Mastermind_v5")

class BotState:
    def __init__(self):
        self.exchange = None
        self.settings = {}
        self.last_signal_time = {}
        self.market_mood = {"mood": "UNKNOWN", "reason": "تحليل لم يتم بعد", "btc_mood": "UNKNOWN", "fng": "N/A", "news": "N/A"}
        self.scan_stats = {"last_start": None, "last_duration": "N/A", "markets_scanned": 0, "failures": 0}

bot_state = BotState()
scan_lock = asyncio.Lock()

# =======================================================================================
# --- [MERGE] الثوابت والقواميس الخاصة بواجهة المستخدم ---
# =======================================================================================
DEFAULT_SETTINGS = {
    "active_preset": "PRO",
    "real_trade_size_usdt": 15.0,
    "top_n_symbols_by_volume": 250,
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 2.0,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "whale_radar", "sniper_pro"],
    "market_mood_filter_enabled": True,
    "fear_and_greed_threshold": 30,
    "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "max_spread_percent": 0.5, "min_rvol": 1.5},
    "volatility_filters": {"min_atr_percent": 0.8},
    "trend_filters": {"ema_period": 200, "htf_period": 50},
    "min_tp_sl_filter": {"min_tp_percent": 1.0, "min_sl_percent": 0.5},
    "scan_interval_seconds": 900,
    "track_interval_seconds": 60,
    "trailing_sl_enabled": True,
    "trailing_sl_activation_percent": 1.5,
    "trailing_sl_callback_percent": 1.0
}

PRESETS = {
    "PRO": {"liquidity_filters": {"min_rvol": 1.5}, "volatility_filters": {"min_atr_percent": 0.8}, "name": "🚦 احترافية (متوازنة)"},
    "STRICT": {"liquidity_filters": {"min_rvol": 2.2}, "volatility_filters": {"min_atr_percent": 1.4}, "name": "🎯 متشددة"},
    "LAX": {"liquidity_filters": {"min_rvol": 1.1}, "volatility_filters": {"min_atr_percent": 0.4}, "name": "🌙 متساهلة"},
    "VERY_LAX": {"liquidity_filters": {"min_rvol": 0.8}, "volatility_filters": {"min_atr_percent": 0.2}, "name": "⚠️ فائق التساهل"}
}

EDITABLE_PARAMS = {
    "إعدادات المخاطر": ["real_trade_size_usdt", "atr_sl_multiplier", "risk_reward_ratio"],
    "إعدادات الوقف المتحرك": ["trailing_sl_enabled", "trailing_sl_activation_percent", "trailing_sl_callback_percent"],
    "إعدادات الفحص والمزاج": ["top_n_symbols_by_volume", "fear_and_greed_threshold", "market_mood_filter_enabled"]
}
PARAM_DISPLAY_NAMES = {
    "real_trade_size_usdt": "💵 حجم الصفقة ($)", "atr_sl_multiplier": "مضاعف وقف الخسارة (ATR)",
    "risk_reward_ratio": "نسبة المخاطرة/العائد", "trailing_sl_enabled": "⚙️ تفعيل الوقف المتحرك",
    "trailing_sl_activation_percent": "تفعيل الوقف المتحرك (%)", "trailing_sl_callback_percent": "مسافة تتبع الوقف (%)",
    "top_n_symbols_by_volume": "عدد العملات للفحص", "fear_and_greed_threshold": "حد مؤشر الخوف",
    "market_mood_filter_enabled": "فلتر مزاج السوق"
}
STRATEGIES_MAP = {
    "momentum_breakout": {"func_name": "analyze_momentum_breakout", "name": "زخم اختراقي"},
    "breakout_squeeze_pro": {"func_name": "analyze_breakout_squeeze_pro", "name": "اختراق انضغاطي"},
    "support_rebound": {"func_name": "analyze_support_rebound", "name": "ارتداد الدعم"},
    "sniper_pro": {"func_name": "analyze_sniper_pro", "name": "القناص المحترف"},
    "whale_radar": {"func_name": "analyze_whale_radar", "name": "رادار الحيتان"},
}

# =======================================================================================
# --- دوال المساعدة (الإعدادات، قاعدة البيانات، تحليل المزاج) 🗄️ ---
# =======================================================================================
def escape_markdown(text: str) -> str:
    """Helper function to escape telegram markdown symbols."""
    if not isinstance(text, str):
        text = str(text)
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                bot_state.settings = json.load(f)
        else:
            bot_state.settings = DEFAULT_SETTINGS.copy()
        for key, value in DEFAULT_SETTINGS.items():
            if key not in bot_state.settings:
                bot_state.settings[key] = value
            elif isinstance(value, dict):
                 for sub_key, sub_value in value.items():
                      if sub_key not in bot_state.settings.get(key, {}):
                           bot_state.settings[key][sub_key] = sub_value
        save_settings()
        logger.info("Settings loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        bot_state.settings = DEFAULT_SETTINGS.copy()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(bot_state.settings, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

def init_database():
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
                entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, entry_value_usdt REAL,
                status TEXT DEFAULT 'active', exit_price REAL, closed_at TEXT, pnl_usdt REAL,
                reason TEXT, order_id TEXT, algo_id TEXT,
                highest_price REAL, trailing_sl_active BOOLEAN DEFAULT 0
            )''')
        conn.commit()
        existing_cols = [c[1] for c in cursor.execute('PRAGMA table_info(trades);').fetchall()]
        for col in ['highest_price', 'trailing_sl_active', 'algo_id']:
            if col not in existing_cols:
                cursor.execute(f'ALTER TABLE trades ADD COLUMN {col} ' + ('REAL' if col == 'highest_price' else 'TEXT' if col == 'algo_id' else 'BOOLEAN DEFAULT 0'))
                conn.commit()
        conn.close()
        logger.info(f"Database initialized/verified at: {DB_FILE}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def log_trade_to_db(signal, order_receipt, algo_id):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        sql = '''INSERT INTO trades (timestamp, symbol, entry_price, take_profit, stop_loss, quantity, entry_value_usdt, reason, order_id, algo_id, highest_price)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        avg_price = order_receipt.get('average', signal['entry_price'])
        filled_qty = order_receipt.get('filled', 0)
        params = (
            datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), signal['symbol'],
            avg_price, signal['final_tp'], signal['final_sl'], filled_qty,
            avg_price * filled_qty, signal['reason'], order_receipt.get('id'),
            algo_id, avg_price
        )
        cursor.execute(sql, params)
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id
    except Exception as e:
        logger.error(f"Failed to log trade to DB: {e}")
        return None

async def get_fear_and_greed_index():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            return int(r.json()['data'][0]['value'])
    except Exception: return None

def get_latest_crypto_news():
    headlines = []
    urls = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]
    for url in urls:
        try:
            feed = feedparser.parse(url)
            headlines.extend(entry.title for entry in feed.entries[:5])
        except Exception: pass
    return list(set(headlines))[:10]

def analyze_sentiment_of_headlines(headlines):
    if not headlines or not NLTK_AVAILABLE: return 0.0, "N/A"
    sia = SentimentIntensityAnalyzer()
    score = sum(sia.polarity_scores(h)['compound'] for h in headlines) / len(headlines)
    if score > 0.1: mood = "إيجابية"
    elif score < -0.1: mood = "سلبية"
    else: mood = "محايدة"
    return score, f"{mood} (الدرجة: {score:.2f})"

async def get_market_mood():
    try:
        exchange = bot_state.exchange
        htf_period = bot_state.settings['trend_filters']['htf_period']
        ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '4h', limit=htf_period + 5)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['sma'] = ta.sma(df['close'], length=htf_period)
        is_btc_bullish = df['close'].iloc[-1] > df['sma'].iloc[-1]
        btc_mood_text = "إيجابي ✅" if is_btc_bullish else "سلبي ❌"
        if not is_btc_bullish:
            return {"mood": "NEGATIVE", "reason": "اتجاه BTC هابط (تحت متوسط 50 على 4 ساعات)", "btc_mood": btc_mood_text, "fng": "N/A", "news": "N/A"}
    except Exception as e:
        logger.warning(f"Could not fetch BTC trend: {e}")
        return {"mood": "DANGEROUS", "reason": "فشل جلب بيانات BTC", "btc_mood": "UNKNOWN", "fng": "N/A", "news": "N/A"}
    
    fng = await get_fear_and_greed_index()
    fng_text = str(fng) if fng is not None else "N/A"
    if fng is not None and fng < bot_state.settings['fear_and_greed_threshold']:
        return {"mood": "NEGATIVE", "reason": f"مشاعر خوف شديد (مؤشر F&G: {fng})", "btc_mood": btc_mood_text, "fng": fng_text, "news": "N/A"}
        
    _, news_mood_text = analyze_sentiment_of_headlines(get_latest_crypto_news())

    return {"mood": "POSITIVE", "reason": "وضع السوق مناسب", "btc_mood": btc_mood_text, "fng": fng_text, "news": news_mood_text}

# =======================================================================================
# --- 🧠 العقل: دوال التحليل والاستراتيجيات 🧠 ---
# =======================================================================================
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None
def analyze_momentum_breakout(df, rvol):
    df.ta.vwap(append=True); df.ta.bbands(length=20, std=2.0, append=True); df.ta.macd(fast=12, slow=26, signal=9, append=True); df.ta.rsi(length=14, append=True)
    last, prev = df.iloc[-2], df.iloc[-3]
    macd_col, macds_col, bbu_col, rsi_col = find_col(df.columns, "MACD_"), find_col(df.columns, "MACDs_"), find_col(df.columns, "BBU_"), find_col(df.columns, "RSI_")
    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and last[rsi_col] < 68):
        return {"reason": STRATEGIES_MAP['momentum_breakout']['name'], "type": "long"}
    return None
def analyze_breakout_squeeze_pro(df, rvol):
    df.ta.bbands(length=20, std=2.0, append=True); df.ta.kc(length=20, scalar=1.5, append=True); df.ta.obv(append=True)
    bbu_col, bbl_col, kcu_col, kcl_col = find_col(df.columns, "BBU_20"), find_col(df.columns, "BBL_20"), find_col(df.columns, "KCUe_20"), find_col(df.columns, "KCLEe_20")
    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]
    if is_in_squeeze:
        breakout_fired = last['close'] > last[bbu_col]
        volume_ok = last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5
        obv_rising = df['OBV'].iloc[-2] > df['OBV'].iloc[-3]
        if breakout_fired and volume_ok and obv_rising: return {"reason": STRATEGIES_MAP['breakout_squeeze_pro']['name'], "type": "long"}
    return None
async def analyze_support_rebound(df, rvol, exchange, symbol):
    try:
        ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
        if not ohlcv_1h or len(ohlcv_1h) < 50: return None
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        current_price = df_1h['close'].iloc[-1]
        recent_lows = df_1h['low'].rolling(window=10, center=True).min()
        supports = recent_lows[recent_lows.notna()]
        closest_support = max([s for s in supports if s < current_price], default=None)
        if not closest_support: return None
        if (current_price - closest_support) / closest_support * 100 < 1.0:
            last_candle_15m = df.iloc[-2]
            avg_volume_15m = df['volume'].rolling(window=20).mean().iloc[-2]
            if last_candle_15m['close'] > last_candle_15m['open'] and last_candle_15m['volume'] > avg_volume_15m * 1.5:
                return {"reason": STRATEGIES_MAP['support_rebound']['name'], "type": "long"}
    except Exception: return None
    return None
def analyze_sniper_pro(df, rvol):
    try:
        compression_candles = int(6 * 4)
        if len(df) < compression_candles + 2: return None
        compression_df = df.iloc[-compression_candles-1:-1]
        highest_high, lowest_low = compression_df['high'].max(), compression_df['low'].min()
        volatility = (highest_high - lowest_low) / lowest_low * 100 if lowest_low > 0 else float('inf')
        if volatility < 12.0:
            last_candle = df.iloc[-2]
            if last_candle['close'] > highest_high and last_candle['volume'] > compression_df['volume'].mean() * 2:
                return {"reason": STRATEGIES_MAP['sniper_pro']['name'], "type": "long"}
    except Exception: return None
    return None
async def analyze_whale_radar(df, rvol, exchange, symbol):
    try:
        ob = await exchange.fetch_order_book(symbol, limit=20)
        if not ob or not ob.get('bids'): return None
        total_bid_value = sum(float(price) * float(qty) for price, qty in ob['bids'][:10])
        if total_bid_value > 30000:
            return {"reason": STRATEGIES_MAP['whale_radar']['name'], "type": "long"}
    except Exception: return None
    return None

# =======================================================================================
# --- المراقب الذكي: منطق الوقف المتحرك ---
# =======================================================================================
async def place_okx_oco_order(exchange, symbol, quantity, tp_price, sl_price):
    try:
        inst_id = exchange.market_id(symbol)
        payload = {
            'instId': inst_id, 'tdMode': 'cash', 'side': 'sell', 'ordType': 'oco',
            'sz': str(exchange.amount_to_precision(symbol, quantity)),
            'tpTriggerPx': exchange.price_to_precision(symbol, tp_price), 'tpOrdPx': '-1',
            'slTriggerPx': exchange.price_to_precision(symbol, sl_price), 'slOrdPx': '-1',
        }
        response = await exchange.private_post_trade_order_algo(payload)
        if response and response.get('data') and response['data'][0].get('algoId'):
            return response['data'][0]['algoId']
        raise ccxt.ExchangeError(f"Failed to place OCO, invalid response: {response}")
    except Exception as e:
        logger.error(f"Error placing OCO for {symbol}: {e}")
        raise

async def track_open_trades(context: ContextTypes.DEFAULT_TYPE):
    settings = bot_state.settings
    if not settings.get('trailing_sl_enabled', False): return
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    active_trades = [dict(row) for row in conn.cursor().execute("SELECT * FROM trades WHERE status = 'active'").fetchall()]
    conn.close()
    if not active_trades: return
    exchange, bot = bot_state.exchange, context.bot

    for trade in active_trades:
        try:
            ticker = await exchange.fetch_ticker(trade['symbol'])
            current_price = ticker.get('last')
            if not current_price: continue
            highest_price = max(trade.get('highest_price', 0) or current_price, current_price)
            new_sl, is_activation = None, False
            if not trade['trailing_sl_active']:
                activation_price = trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100)
                if current_price >= activation_price:
                    new_sl, is_activation = trade['entry_price'], True
            else:
                callback_price = highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                if callback_price > trade['stop_loss']: new_sl = callback_price
            if new_sl and new_sl > trade['stop_loss']:
                logger.info(f"{'ACTIVATING' if is_activation else 'UPDATING'} TSL for trade #{trade['id']}. New SL: {new_sl}")
                await exchange.private_post_trade_cancel_algos([{'instId': exchange.market_id(trade['symbol']), 'algoId': trade['algo_id']}])
                new_algo_id = await place_okx_oco_order(exchange, trade['symbol'], trade['quantity'], trade['take_profit'], new_sl)
                with sqlite3.connect(DB_FILE) as conn:
                    conn.cursor().execute("UPDATE trades SET stop_loss=?, highest_price=?, trailing_sl_active=1, algo_id=? WHERE id=?",
                                          (new_sl, highest_price, new_algo_id, trade['id']))
                if is_activation:
                    await bot.send_message(TELEGRAM_CHAT_ID, f"**🚀 تأمين الأرباح! | #{trade['id']} {trade['symbol']}**\n\nتم رفع وقف الخسارة إلى نقطة الدخول.", parse_mode=ParseMode.MARKDOWN)
            elif highest_price > (trade.get('highest_price') or 0):
                with sqlite3.connect(DB_FILE) as conn:
                    conn.cursor().execute("UPDATE trades SET highest_price=? WHERE id=?", (highest_price, trade['id']))
        except ccxt.OrderNotFound:
             logger.warning(f"Order for trade #{trade['id']} seems filled or cancelled.")
        except Exception as e:
            logger.error(f"Error in TSL for trade #{trade['id']}: {e}")

# =======================================================================================
# --- 🦾 جسد البوت: منطق التشغيل والفحص والتداول 🦾 ---
# =======================================================================================
async def execute_atomic_trade(signal):
    symbol, settings, exchange, bot = signal['symbol'], bot_state.settings, bot_state.exchange, application.bot
    logger.info(f"Attempting ATOMIC trade for {symbol} using attachAlgoOrds.")
    try:
        quantity_to_buy = settings['real_trade_size_usdt'] / signal['entry_price']
        tp_price_str = exchange.price_to_precision(symbol, signal['take_profit'])
        sl_price_str = exchange.price_to_precision(symbol, signal['stop_loss'])
        attached_algo_orders = [
            {'tpTriggerPx': tp_price_str, 'tpOrdPx': '-1', 'side': 'sell'},
            {'slTriggerPx': sl_price_str, 'slOrdPx': '-1', 'side': 'sell'}]
        params = {'tdMode': 'cash', 'attachAlgoOrds': attached_algo_orders, 'clOrdId': f'mastermind_{int(time.time()*1000)}'}
        order_receipt = await exchange.create_order(symbol=symbol, type='market', side='buy', amount=quantity_to_buy, params=params)
        logger.info(f"Atomic order request sent. Order ID: {order_receipt.get('id')}")
        max_retries = 10
        for i in range(max_retries):
            await asyncio.sleep(2.5)
            verified_order = await exchange.fetch_order(order_receipt.get('id'), symbol)
            if verified_order and verified_order.get('status') == 'filled':
                logger.info(f"✅ VERIFIED: Main order {verified_order.get('id')} is filled.")
                await asyncio.sleep(1)
                open_orders = await exchange.fetch_open_orders(symbol)
                algo_order = next((o for o in open_orders if o.get('clOrdId') == verified_order.get('clOrdId')), None)
                algo_id = algo_order.get('id') if algo_order else 'unknown'
                avg_price = verified_order.get('average', signal['entry_price'])
                original_risk = signal['entry_price'] - signal['stop_loss']
                signal['final_sl'] = avg_price - original_risk
                signal['final_tp'] = avg_price + (original_risk * settings['risk_reward_ratio'])
                trade_id = log_trade_to_db(signal, verified_order, algo_id)
                tp_percent = (signal['final_tp'] - avg_price) / avg_price * 100 if avg_price > 0 else 0
                sl_percent = (avg_price - signal['final_sl']) / avg_price * 100 if avg_price > 0 else 0
                success_msg = (
                    f"**✅ صفقة ذرية ناجحة | {symbol} (ID: {trade_id})**\n"
                    f"------------------------------------\n"
                    f"🔍 **الاستراتيجية:** {signal['reason']}\n\n"
                    f"📈 **متوسط الشراء:** `{avg_price:,.4f}`\n"
                    f"🎯 **الهدف:** `{signal['final_tp']:,.4f}` (+{tp_percent:.2f}%)\n"
                    f"🛑 **الوقف:** `{signal['final_sl']:,.4f}` (-{sl_percent:.2f}%)\n\n"
                    f"***الصفقة مؤمنة بالكامل بشكل ذري.***")
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=success_msg, parse_mode=ParseMode.MARKDOWN)
                return
        raise Exception("Failed to verify order and protection status.")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during atomic trade for {symbol}: {e}")
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"**🔥🔥🔥 فشل ذري حرج - {symbol}**\n\n**الخطأ:** `{str(e)}`", parse_mode=ParseMode.MARKDOWN)

async def worker(queue, signals_list, failure_counter):
    settings, exchange = bot_state.settings, bot_state.exchange
    while not queue.empty():
        market = await queue.get()
        symbol = market.get('symbol')
        try:
            orderbook = await exchange.fetch_order_book(symbol, limit=1)
            if not orderbook['bids'] or not orderbook['asks']: continue
            spread = (orderbook['asks'][0][0] - orderbook['bids'][0][0]) / orderbook['bids'][0][0] * 100
            if spread > settings['liquidity_filters']['max_spread_percent']: continue
            ohlcv = await exchange.fetch_ohlcv(symbol, '15m', limit=settings['trend_filters']['ema_period'] + 20)
            if len(ohlcv) < settings['trend_filters']['ema_period'] + 10: continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] == 0: continue
            rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
            if rvol < settings['liquidity_filters']['min_rvol']: continue
            df.ta.atr(length=14, append=True); atr_col = find_col(df.columns, 'ATRr_')
            last_close = df['close'].iloc[-2]
            if not atr_col or last_close == 0: continue
            atr_percent = (df[atr_col].iloc[-2] / last_close) * 100
            if atr_percent < settings['volatility_filters']['min_atr_percent']: continue
            ema_period = settings['trend_filters']['ema_period']
            df.ta.ema(length=ema_period, append=True)
            ema_col = find_col(df.columns, f'EMA_{ema_period}')
            if not ema_col or pd.isna(df[ema_col].iloc[-2]) or last_close < df[ema_col].iloc[-2]: continue
            htf_period = settings['trend_filters']['htf_period']
            htf_ohlcv = await exchange.fetch_ohlcv(symbol, '1h', limit=htf_period + 5)
            if len(htf_ohlcv) < htf_period: continue
            df_htf = pd.DataFrame(htf_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_htf['sma'] = ta.sma(df_htf['close'], length=htf_period)
            if df_htf['close'].iloc[-1] < df_htf['sma'].iloc[-1]: continue
            confirmed_reasons = []
            for name in settings['active_scanners']:
                strategy_info = STRATEGIES_MAP.get(name)
                if not strategy_info: continue
                strategy_func = globals()[strategy_info['func_name']]
                if asyncio.iscoroutinefunction(strategy_func):
                    result = await strategy_func(df.copy(), rvol, exchange, symbol)
                else:
                    result = strategy_func(df.copy(), rvol)
                if result: confirmed_reasons.append(result['reason'])
            if confirmed_reasons:
                reason_str = ' + '.join(confirmed_reasons)
                entry_price = last_close
                df.ta.atr(length=14, append=True); atr_col = find_col(df.columns, "ATRr_14")
                current_atr = df.iloc[-2].get(atr_col, 0)
                if current_atr > 0:
                    risk = current_atr * settings['atr_sl_multiplier']
                    stop_loss, take_profit = entry_price - risk, entry_price + (risk * settings['risk_reward_ratio'])
                    if (take_profit/entry_price - 1)*100 >= settings['min_tp_sl_filter']['min_tp_percent'] and (1 - stop_loss/entry_price)*100 >= settings['min_tp_sl_filter']['min_sl_percent']:
                        signals_list.append({"symbol": symbol, "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "reason": reason_str})
        except Exception as e:
            logger.debug(f"Worker error for {symbol}: {e}")
            failure_counter[0] += 1
        finally: queue.task_done()

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        settings, bot = bot_state.settings, context.bot
        bot_state.scan_stats['last_start'] = datetime.now(EGYPT_TZ)
        logger.info("--- Starting new market scan... ---")
        if settings['market_mood_filter_enabled']:
            mood_result = await get_market_mood()
            bot_state.market_mood = mood_result
            if mood_result['mood'] in ["NEGATIVE", "DANGEROUS"]:
                logger.warning(f"SCAN SKIPPED: {mood_result['reason']}")
                await bot.send_message(TELEGRAM_CHAT_ID, f"⚠️ **تم إيقاف الفحص مؤقتًا**\n**السبب:** {mood_result['reason']}", parse_mode=ParseMode.MARKDOWN)
                if mood_result['mood'] == "NEGATIVE" and settings['active_preset'] != "STRICT":
                     await bot.send_message(TELEGRAM_CHAT_ID, f"💡 **اقتراح ذكي:** مزاج السوق سلبي. هل تريد التحول إلى النمط **المتشدد** لتقليل المخاطرة؟",
                                           reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("✅ نعم، حوّل إلى النمط المتشدد", callback_data="preset_STRICT")]]),
                                           parse_mode=ParseMode.MARKDOWN)
                return
        exchange = bot_state.exchange
        try:
            tickers = await exchange.fetch_tickers()
            usdt_markets = [m for m in tickers.values() if m.get('symbol', '').endswith('/USDT') and not any(k in m['symbol'] for k in ['-SWAP', 'UP', 'DOWN', '3L', '3S']) and m.get('quoteVolume', 0) > settings['liquidity_filters']['min_quote_volume_24h_usd']]
            usdt_markets.sort(key=lambda m: m.get('quoteVolume', 0), reverse=True)
            top_markets = usdt_markets[:settings['top_n_symbols_by_volume']]
            bot_state.scan_stats['markets_scanned'] = len(top_markets)
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return
        queue, signals_found, failure_counter = asyncio.Queue(), [], [0]
        for market in top_markets: await queue.put(market)
        worker_tasks = [asyncio.create_task(worker(queue, signals_found, failure_counter)) for _ in range(10)]
        await queue.join()
        for task in worker_tasks: task.cancel()
        
        bot_state.scan_stats['failures'] = failure_counter[0]
        duration = (datetime.now(EGYPT_TZ) - bot_state.scan_stats['last_start']).total_seconds()
        bot_state.scan_stats['last_duration'] = f"{duration:.0f} ثانية"

        scan_summary = (
            f"🔬 **ملخص الفحص الأخير**\n\n"
            f"- **الحالة:** اكتمل بنجاح\n"
            f"- **وضع السوق:** {bot_state.market_mood['mood']} ({bot_state.market_mood['btc_mood']})\n"
            f"- **المدة:** {bot_state.scan_stats['last_duration']}\n"
            f"- **العملات المفحوصة:** {bot_state.scan_stats['markets_scanned']}\n\n"
            f"------------------------------------\n"
            f"- **إجمالي الإشارات المكتشفة:** {len(signals_found)}\n"
        )
        new_trades = 0
        if signals_found:
            logger.info(f"+++ Scan complete. Found {len(signals_found)} signals! +++")
            for signal in signals_found:
                if time.time() - bot_state.last_signal_time.get(signal['symbol'], 0) < settings['scan_interval_seconds'] * 2.5: continue
                bot_state.last_signal_time[signal['symbol']] = time.time()
                await execute_atomic_trade(signal)
                new_trades += 1
                await asyncio.sleep(10)
        else:
            logger.info("--- Scan complete. No new signals found. ---")
        scan_summary += f"- **✅ صفقات جديدة فُتحت:** {new_trades}\n"
        scan_summary += f"- **⚠️ أخطاء في التحليل:** {bot_state.scan_stats['failures']}"
        await bot.send_message(TELEGRAM_CHAT_ID, scan_summary, parse_mode=ParseMode.MARKDOWN)

# =======================================================================================
# --- 📱 واجهة التحكم عبر تليجرام (النسخة الكاملة) 📱 ---
# =======================================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"]]
    await update.message.reply_text("أهلاً بك في بوت OKX القناص v5.3 (The Mastermind)", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True))

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 الإحصائيات العامة", callback_data="dashboard_stats")],
        [InlineKeyboardButton("📈 الصفقات النشطة", callback_data="dashboard_active_trades")],
        [InlineKeyboardButton("📜 تقرير أداء الاستراتيجيات", callback_data="dashboard_strategy_report")],
        [InlineKeyboardButton("🌡️ حالة مزاج السوق", callback_data="dashboard_mood"), InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="dashboard_diagnostics")]
    ]
    await update.message.reply_text("🖥️ *لوحة التحكم الرئيسية*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["🎭 تفعيل/تعطيل الماسحات", "🔧 تعديل المعايير"], ["🏁 الأنماط الجاهزة"], ["🔙 القائمة الرئيسية"]]
    await update.message.reply_text("اختر الإعداد:", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True))

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text
    menu_map = {"Dashboard 🖥️": show_dashboard_command, "⚙️ الإعدادات": show_settings_menu,
                "🎭 تفعيل/تعطيل الماسحات": show_scanners_menu, "🔧 تعديل المعايير": show_parameters_menu,
                "🏁 الأنماط الجاهزة": show_presets_menu, "🔙 القائمة الرئيسية": start_command}
    if text in menu_map: await menu_map[text](update, context)

async def input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'awaiting_input_for_param' in context.user_data:
        param_key, msg_to_del = context.user_data.pop('awaiting_input_for_param')
        new_value_str = update.message.text
        settings = bot_state.settings
        try:
            current_value = settings.get(param_key)
            if isinstance(current_value, bool): new_value = new_value_str.lower() in ['true', '1', 'on', 'yes', 'نعم', 'تفعيل']
            elif isinstance(current_value, float): new_value = float(new_value_str)
            else: new_value = int(new_value_str)
            settings[param_key] = new_value
            save_settings()
            await update.message.reply_text(f"✅ تم تحديث **{PARAM_DISPLAY_NAMES.get(param_key, param_key)}** إلى `{new_value}`.", parse_mode=ParseMode.MARKDOWN)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=msg_to_del)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
        except (ValueError, TypeError):
            await update.message.reply_text("❌ قيمة غير صالحة. الرجاء المحاولة مرة أخرى.")

async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active = bot_state.settings.get("active_scanners", [])
    keyboard = [[InlineKeyboardButton(f"{'✅' if k in active else '❌'} {v['name']}", callback_data=f"toggle_scanner_{k}")] for k, v in STRATEGIES_MAP.items()]
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    await update.message.reply_text("اختر الماسحات لتفعيلها أو تعطيلها:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(v['name'], callback_data=f"preset_{k}")] for k,v in PRESETS.items()]
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    await update.message.reply_text("اختر نمط إعدادات جاهز:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard, settings = [], bot_state.settings
    for category, params in EDITABLE_PARAMS.items():
        keyboard.append([InlineKeyboardButton(f"--- {category} ---", callback_data="ignore")])
        for param_key in params:
            name = PARAM_DISPLAY_NAMES.get(param_key, param_key)
            value = settings.get(param_key, "N/A")
            text = f"{name}: {'مُفعّل ✅' if value else 'مُعطّل ❌'}" if isinstance(value, bool) else f"{name}: {value}"
            keyboard.append([InlineKeyboardButton(text, callback_data=f"param_{param_key}")])
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    await update.message.reply_text("⚙️ *الإعدادات المتقدمة*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    try:
        if data.startswith("dashboard_"):
            await query.message.delete()
            report_type = data.split("_", 1)[1]
            if report_type == "stats":
                with sqlite3.connect(DB_FILE) as conn:
                    stats = conn.cursor().execute("SELECT status, COUNT(*), SUM(pnl_usdt) FROM trades WHERE status != 'active' GROUP BY status").fetchall()
                counts, pnl = defaultdict(int), defaultdict(float)
                for status, count, p in stats: counts[status], pnl[status] = count, p or 0
                wins = sum(v for k, v in counts.items() if k and k.startswith('ناجحة'))
                losses = counts.get('فاشلة (وقف خسارة)', 0); closed = wins + losses
                win_rate = (wins / closed * 100) if closed > 0 else 0
                total_pnl = sum(pnl.values())
                await query.message.reply_text(f"*📊 الإحصائيات العامة*\n- الصفقات المغلقة: {closed}\n- نسبة النجاح: {win_rate:.2f}%\n- صافي الربح/الخسارة: ${total_pnl:+.2f}", parse_mode=ParseMode.MARKDOWN)
            elif report_type == "active_trades":
                with sqlite3.connect(DB_FILE) as conn:
                    conn.row_factory = sqlite3.Row
                    trades = conn.cursor().execute("SELECT id, symbol, entry_value_usdt FROM trades WHERE status = 'active' ORDER BY id DESC").fetchall()
                if not trades: return await query.message.reply_text("لا توجد صفقات نشطة حالياً.")
                keyboard = [[InlineKeyboardButton(f"#{t['id']} | {t['symbol']} | ${t['entry_value_usdt']:.2f}", callback_data=f"check_{t['id']}")] for t in trades]
                await query.message.reply_text("اختر صفقة لمتابعتها:", reply_markup=InlineKeyboardMarkup(keyboard))
            elif report_type == "strategy_report":
                with sqlite3.connect(DB_FILE) as conn:
                    trades = conn.cursor().execute("SELECT reason, status, pnl_usdt FROM trades WHERE status != 'active'").fetchall()
                if not trades: return await query.message.reply_text("لا توجد صفقات مغلقة لتحليلها.")
                stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})
                for reason, status, pnl_val in trades:
                    if not reason or not status: continue
                    reasons = reason.split(' + ')
                    for r in reasons:
                        s = stats[r.strip()]
                        if status.startswith('ناجحة'): s['wins'] += 1
                        else: s['losses'] += 1
                        if pnl_val: s['pnl'] += pnl_val / len(reasons)
                report = ["**📜 تقرير أداء الاستراتيجيات**"]
                for r, s in stats.items():
                    total = s['wins'] + s['losses']
                    wr = (s['wins'] / total * 100) if total > 0 else 0
                    report.append(f"\n--- *{escape_markdown(r)}* ---\n  - الصفقات: {total} ({s['wins']}✅ / {s['losses']}❌)\n  - النجاح: {wr:.2f}%\n  - صافي الربح: ${s['pnl']:+.2f}")
                await query.message.reply_text("\n".join(report), parse_mode=ParseMode.MARKDOWN)
            elif report_type == "mood":
                mood = bot_state.market_mood
                await query.message.reply_text(f"*🌡️ حالة مزاج السوق*\n- **النتيجة:** {mood['mood']}\n- **السبب:** {escape_markdown(mood['reason'])}\n- **مؤشر BTC:** {mood['btc_mood']}\n- **الخوف والطمع:** {mood['fng']}\n- **الأخبار:** {escape_markdown(mood['news'])}", parse_mode=ParseMode.MARKDOWN)
            elif report_type == "diagnostics":
                mood, scan, settings = bot_state.market_mood, bot_state.scan_stats, bot_state.settings
                with sqlite3.connect(DB_FILE) as conn:
                    total_trades, active_trades = conn.cursor().execute("SELECT COUNT(*) FROM trades").fetchone()[0], conn.cursor().execute("SELECT COUNT(*) FROM trades WHERE status = 'active'").fetchone()[0]
                conn_status = "متصل ✅" if bot_state.exchange and bot_state.exchange.check_required_credentials() else "غير متصل ❌"
                report = [f"**🕵️‍♂️ تقرير التشخيص الشامل (v5.2)**\n",
                          f"--- **📊 حالة السوق الحالية** ---\n- **المزاج العام:** {mood['mood']} ({escape_markdown(mood['reason'])})\n- **مؤشر BTC:** {mood['btc_mood']}\n- **الخوف والطمع:** {mood['fng']}\n",
                          f"--- **🔬 أداء آخر فحص** ---\n- **وقت البدء:** {scan['last_start']}\n- **المدة:** {scan['last_duration']}\n- **العملات المفحوصة:** {scan['markets_scanned']}\n- **فشل في تحليل:** {scan['failures']} عملات\n",
                          f"--- **🔧 الإعدادات النشطة** ---\n- **النمط الحالي:** {settings['active_preset']}\n- **الماسحات المفعلة:** {escape_markdown(', '.join(settings['active_scanners']))}\n",
                          f"--- **🔩 حالة العمليات الداخلية** ---\n- **اتصال المنصة:** {conn_status}\n- **قاعدة البيانات:** متصلة ✅ ({total_trades} صفقة / {active_trades} نشطة)"]
                await query.message.reply_text("\n".join(report), parse_mode=ParseMode.MARKDOWN)

        elif data.startswith("toggle_scanner_"):
            scanner_name = data.split("_", 2)[2]
            active = bot_state.settings.get("active_scanners", []).copy()
            if scanner_name in active: active.remove(scanner_name)
            else: active.append(scanner_name)
            bot_state.settings["active_scanners"] = active; save_settings()
            keyboard = [[InlineKeyboardButton(f"{'✅' if k in active else '❌'} {v['name']}", callback_data=f"toggle_scanner_{k}")] for k, v in STRATEGIES_MAP.items()]
            keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))
            
        elif data.startswith("preset_"):
            preset_name = data.split("_", 1)[1]
            if preset_data := PRESETS.get(preset_name):
                bot_state.settings['liquidity_filters'].update(preset_data['liquidity_filters'])
                bot_state.settings['volatility_filters'].update(preset_data['volatility_filters'])
                bot_state.settings["active_preset"] = preset_name
                save_settings()
                await query.edit_message_text(f"✅ تم تفعيل النمط: **{preset_data['name']}**", parse_mode=ParseMode.MARKDOWN)

        elif data.startswith("param_"):
            param_key = data.split("_", 1)[1]
            if isinstance(bot_state.settings.get(param_key), bool):
                 bot_state.settings[param_key] = not bot_state.settings[param_key]; save_settings()
                 await query.message.delete(); await show_parameters_menu(update, context)
            else:
                 msg = await query.message.reply_text(f"📝 *تعديل '{PARAM_DISPLAY_NAMES.get(param_key, param_key)}'*\n*القيمة الحالية:* `{bot_state.settings.get(param_key)}`\n\nأرسل القيمة الجديدة.", parse_mode=ParseMode.MARKDOWN)
                 context.user_data['awaiting_input_for_param'] = (param_key, msg.message_id)

        elif data == "back_to_settings":
            await query.message.delete()
    except BadRequest as e:
        if "Message is not modified" not in str(e):
            logger.error(f"Telegram BadRequest in button handler: {e}")
            await query.message.reply_text(f"حدث خطأ في عرض البيانات: {e}")
    except Exception as e:
        logger.error(f"General error in button handler: {e}", exc_info=True)
        await query.message.reply_text("حدث خطأ غير متوقع.")

# =======================================================================================
# --- 🚀 نقطة انطلاق البوت 🚀 ---
# =======================================================================================
application = None
async def post_init(app: Application):
    global application
    application = app
    if NLTK_AVAILABLE:
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError: logger.info("Downloading NLTK data..."); nltk.download('vader_lexicon')
    logger.info("🚀 Starting OKX Mastermind v5.3 (Stable)...")
    if 'YOUR_OKX_API_KEY' in OKX_API_KEY or 'YOUR_BOT_TOKEN' in TELEGRAM_BOT_TOKEN:
        logger.critical("FATAL: API keys or Bot Token are not set."); return

    # [FIX v5.3] The correct, stable way to monkey-patch ccxt
    original_fetch2 = ccxt.base.exchange.Exchange.fetch2
    def patched_fetch2(self, path, api='public', method='GET', params=None, headers=None, body=None, config=None, context=None):
        params = params or {}
        if self.id == 'okx':
            if (path == 'trade/order-algo') or (path == 'trade/order' and 'attachAlgoOrds' in params):
                if params.get("side") == "sell":
                    params.pop("tgtCcy", None)
        return original_fetch2(self, path, api, method, params, headers, body, config)

    ccxt.base.exchange.Exchange.fetch2 = patched_fetch2
    logger.info("Applied STABLE global monkey-patch for CCXT.")
    
    bot_state.exchange = ccxt.okx({'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE, 'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    
    scan_interval = bot_state.settings.get("scan_interval_seconds", 900)
    track_interval = bot_state.settings.get("track_interval_seconds", 60)
    app.job_queue.run_repeating(perform_scan, interval=scan_interval, first=10, name="perform_scan")
    app.job_queue.run_repeating(track_open_trades, interval=track_interval, first=30, name="track_trades")
    logger.info(f"Scan job every {scan_interval}s. Tracker job every {track_interval}s.")
    await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="*🚀 بوت OKX The Mastermind v5.3 بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)

def main():
    load_settings()
    init_database()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start_command))
    # Regex updated to better handle various inputs and avoid conflicts
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex(r'^[+-]?(\d*\.\d+|\d+\.?\d*)([eE][+-]?\d+)?$'), text_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, input_handler))
    app.add_handler(CallbackQueryHandler(button_callback_handler))
    app.run_polling()

if __name__ == '__main__':
    main()

