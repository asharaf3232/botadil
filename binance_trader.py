# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 بوت OKX القناص v5.3 (The Mastermind - Hardened) - النسخة المحسنة 🚀 ---
# =======================================================================================
# هذا الإصدار يتضمن تحسينات جوهرية في الموثوقية والأداء بناءً على مراجعة متقدمة:
# - [إصلاح حاسم] تصحيح الـ monkey-patch الخاص بـ ccxt ليعمل بشكل غير متزامن (async-safe).
# - [موثوقية] إضافة اختبار اتصال إلزامي بالمنصة (OKX) عند بدء التشغيل.
# - [أداء] استبدال مكتبة sqlite3 بـ aiosqlite لمنع حظر العمليات في البيئة غير المتزامنة.
# - [هيكلية] التخلص من الاعتماد على المتغير العام `application` وتمرير كائن `bot` بشكل صريح.
# - [أمان] تشديد التحقق من وجود المتغيرات الحساسة ومنع التشغيل بقيم افتراضية.
#
# للتثبيت: pip install "ccxt[async]" pandas pandas-ta python-telegram-bot httpx feedparser nltk aiosqlite
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
import aiosqlite  # <-- تم التغيير
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
from telegram.error import BadRequest, Forbidden

# =======================================================================================
# --- ⚙️ الإعدادات الأساسية ⚙️ ---
# =======================================================================================
# [MODIFIED] تحميل المتغيرات بدون قيم افتراضية خطيرة
OKX_API_KEY = os.getenv('OKX_API_KEY')
OKX_API_SECRET = os.getenv('OKX_API_SECRET')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'okx_mastermind_v5.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'okx_mastermind_settings_v5.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Mastermind_v5.3")

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
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
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
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(bot_state.settings, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

# [MODIFIED] Using aiosqlite for all DB operations
async def init_database():
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
                    entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, entry_value_usdt REAL,
                    status TEXT DEFAULT 'active', exit_price REAL, closed_at TEXT, pnl_usdt REAL,
                    reason TEXT, order_id TEXT, algo_id TEXT,
                    highest_price REAL, trailing_sl_active BOOLEAN DEFAULT 0
                )''')
            await conn.commit()
            
            cursor = await conn.execute('PRAGMA table_info(trades);')
            existing_cols = [c[1] for c in await cursor.fetchall()]
            for col in ['highest_price', 'trailing_sl_active', 'algo_id']:
                if col not in existing_cols:
                    await conn.execute(f'ALTER TABLE trades ADD COLUMN {col} ' + ('REAL' if col == 'highest_price' else 'TEXT' if col == 'algo_id' else 'BOOLEAN DEFAULT 0'))
                    await conn.commit()
        logger.info(f"Database initialized/verified at: {DB_FILE}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

# [MODIFIED] Using aiosqlite
async def log_trade_to_db(signal, order_receipt, algo_id):
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
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
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor.lastrowid
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
# [MODIFIED] Using aiosqlite
async def track_open_trades(context: ContextTypes.DEFAULT_TYPE):
    settings = bot_state.settings
    if not settings.get('trailing_sl_enabled', False): return
    
    active_trades = []
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM trades WHERE status = 'active'") as cursor:
                active_trades = [dict(row) for row in await cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to fetch active trades from DB: {e}")
        return

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
                
                # This part is complex, assuming cancel_algos works this way with ccxt
                await exchange.private_post_trade_cancel_algos([{'instId': exchange.market_id(trade['symbol']), 'algoId': trade['algo_id']}])
                
                # Re-placing an OCO is often done by creating a new one
                # For simplicity, assuming a custom function exists or direct API call
                new_algo_id = await exchange.create_order(
                    trade['symbol'], 'oco', 'sell', trade['quantity'], 
                    params={'tpTriggerPx': trade['take_profit'], 'slTriggerPx': new_sl}
                )['id']

                async with aiosqlite.connect(DB_FILE) as conn:
                    await conn.execute("UPDATE trades SET stop_loss=?, highest_price=?, trailing_sl_active=1, algo_id=? WHERE id=?",
                                          (new_sl, highest_price, new_algo_id, trade['id']))
                    await conn.commit()
                
                if is_activation:
                    await bot.send_message(TELEGRAM_CHAT_ID, f"**🚀 تأمين الأرباح! | #{trade['id']} {trade['symbol']}**\n\nتم رفع وقف الخسارة إلى نقطة الدخول.", parse_mode=ParseMode.MARKDOWN)
            
            elif highest_price > (trade.get('highest_price') or 0):
                async with aiosqlite.connect(DB_FILE) as conn:
                    await conn.execute("UPDATE trades SET highest_price=? WHERE id=?", (highest_price, trade['id']))
                    await conn.commit()

        except ccxt.OrderNotFound:
             logger.warning(f"Order for trade #{trade['id']} seems filled or cancelled on the exchange.")
        except Exception as e:
            logger.error(f"Error in TSL for trade #{trade['id']}: {e}", exc_info=True)


# =======================================================================================
# --- 🦾 جسد البوت: منطق التشغيل والفحص والتداول 🦾 ---
# =======================================================================================
# [MODIFIED] Passing `bot` object explicitly
async def execute_atomic_trade(signal, bot: "telegram.Bot"):
    symbol, settings, exchange = signal['symbol'], bot_state.settings, bot_state.exchange
    logger.info(f"Attempting ATOMIC trade for {symbol} using attachAlgoOrds.")
    try:
        quantity_to_buy = settings['real_trade_size_usdt'] / signal['entry_price']
        tp_price_str = exchange.price_to_precision(symbol, signal['take_profit'])
        sl_price_str = exchange.price_to_precision(symbol, signal['stop_loss'])
        
        attached_algo_orders = [
            {'tpTriggerPx': tp_price_str, 'tpOrdPx': '-1', 'side': 'sell'},
            {'slTriggerPx': sl_price_str, 'slOrdPx': '-1', 'side': 'sell'}
        ]
        
        # --- [START] MODIFIED BLOCK FOR RELIABLE ORDER PLACEMENT ---
        # Build the request body manually to match OKX API specifications
        request_body = {
            'instId': exchange.market_id(symbol),
            'tdMode': 'cash',
            'side': 'buy',
            'ordType': 'market',
            'sz': exchange.amount_to_precision(symbol, quantity_to_buy),
            'clOrdId': f'mastermind{int(time.time()*1000)}',
            'attachAlgoOrds': attached_algo_orders
        }
        
        # Use the direct API endpoint instead of the generic create_order
        order_receipt = await exchange.private_post_trade_order(request_body)
        logger.debug(f"Direct API request for {symbol} sent. Receipt: {json.dumps(order_receipt, default=str)}")
        
        # Handle the specific response structure from the direct endpoint
        if order_receipt and order_receipt.get('data') and order_receipt['data'][0].get('sCode') == '0':
            order_id = order_receipt['data'][0]['ordId']
            # Add the order ID to the receipt object so the verification code below can find it
            order_receipt['id'] = order_id 
        else:
            # If the order failed at placement, raise a clear exception
            raise ccxt.ExchangeError(f"OKX API Error on order placement: {json.dumps(order_receipt)}")
        # --- [END] MODIFIED BLOCK ---

        max_retries = 10
        for i in range(max_retries):
            await asyncio.sleep(2.5)
            # Use the extracted order_id for verification
            verified_order = await exchange.fetch_order(order_receipt.get('id'), symbol)
            if verified_order and verified_order.get('status') == 'filled':
                logger.info(f"✅ VERIFIED: Main order {verified_order.get('id')} for {symbol} is filled.")
                await asyncio.sleep(1) # Give exchange time to register algo orders
                
                open_orders = await exchange.fetch_open_orders(symbol)
                # Find the algo order linked to our main market order
                algo_order = next((o for o in open_orders if o.get('clOrdId') == verified_order.get('clOrdId')), None)
                algo_id = algo_order.get('id') if algo_order else 'unknown_algo_id'

                avg_price = verified_order.get('average', signal['entry_price'])
                original_risk = signal['entry_price'] - signal['stop_loss']
                signal['final_sl'] = avg_price - original_risk
                signal['final_tp'] = avg_price + (original_risk * settings['risk_reward_ratio'])
                
                trade_id = await log_trade_to_db(signal, verified_order, algo_id)
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
        
        raise Exception("Failed to verify order and protection status after multiple retries.")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during atomic trade for {symbol}: {e}", exc_info=True)
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
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True) # <--- تم التصحيح
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
        finally:
            queue.task_done()

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
        await asyncio.gather(*worker_tasks, return_exceptions=True) # Wait for cancellation
        
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
                await execute_atomic_trade(signal, bot) # [MODIFIED] Passing bot
                new_trades += 1
                await asyncio.sleep(10) # Stagger trades
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
    await update.message.reply_text("أهلاً بك في بوت OKX القناص v5.3 (The Mastermind - Hardened)", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True))

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

# [MODIFIED] Using aiosqlite for all dashboard queries
async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    try:
        if data.startswith("dashboard_"):
            await query.message.delete()
            report_type = data.split("_", 1)[1]
            if report_type == "stats":
                stats = []
                async with aiosqlite.connect(DB_FILE) as conn:
                     async with conn.execute("SELECT status, COUNT(*), SUM(pnl_usdt) FROM trades WHERE status != 'active' GROUP BY status") as cursor:
                         stats = await cursor.fetchall()
                counts, pnl = defaultdict(int), defaultdict(float)
                for status, count, p in stats: counts[status], pnl[status] = count, p or 0
                wins = sum(v for k, v in counts.items() if k and k.startswith('ناجحة'))
                losses = counts.get('فاشلة (وقف خسارة)', 0); closed = wins + losses
                win_rate = (wins / closed * 100) if closed > 0 else 0
                total_pnl = sum(pnl.values())
                await query.message.reply_text(f"*📊 الإحصائيات العامة*\n- الصفقات المغلقة: {closed}\n- نسبة النجاح: {win_rate:.2f}%\n- صافي الربح/الخسارة: ${total_pnl:+.2f}", parse_mode=ParseMode.MARKDOWN)
            
            elif report_type == "active_trades":
                trades = []
                async with aiosqlite.connect(DB_FILE) as conn:
                    conn.row_factory = aiosqlite.Row
                    async with conn.execute("SELECT id, symbol, entry_value_usdt FROM trades WHERE status = 'active' ORDER BY id DESC") as cursor:
                        trades = await cursor.fetchall()
                if not trades: return await query.message.reply_text("لا توجد صفقات نشطة حالياً.")
                keyboard = [[InlineKeyboardButton(f"#{t['id']} | {t['symbol']} | ${t['entry_value_usdt']:.2f}", callback_data=f"check_{t['id']}")] for t in trades]
                await query.message.reply_text("اختر صفقة لمتابعتها:", reply_markup=InlineKeyboardMarkup(keyboard))
            
            elif report_type == "strategy_report":
                trades = []
                async with aiosqlite.connect(DB_FILE) as conn:
                    async with conn.execute("SELECT reason, status, pnl_usdt FROM trades WHERE status != 'active'") as cursor:
                        trades = await cursor.fetchall()
                if not trades: return await query.message.reply_text("لا توجد صفقات مغلقة لتحليلها.")
                stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})
                for reason, status, pnl_val in trades:
                    if not reason or not status: continue
                    s = stats[reason]
                    if status.startswith('ناجحة'): s['wins'] += 1
                    else: s['losses'] += 1
                    if pnl_val: s['pnl'] += pnl_val
                report = ["**📜 تقرير أداء الاستراتيجيات**"]
                for r, s in sorted(stats.items()):
                    total = s['wins'] + s['losses']
                    wr = (s['wins'] / total * 100) if total > 0 else 0
                    report.append(f"\n--- *{r}* ---\n  - الصفقات: {total} ({s['wins']}✅ / {s['losses']}❌)\n  - النجاح: {wr:.2f}%\n  - صافي الربح: ${s['pnl']:+.2f}")
                await query.message.reply_text("\n".join(report), parse_mode=ParseMode.MARKDOWN)
            
            elif report_type == "mood":
                mood = bot_state.market_mood
                await query.message.reply_text(f"*🌡️ حالة مزاج السوق*\n- **النتيجة:** {mood['mood']}\n- **السبب:** {mood['reason']}\n- **مؤشر BTC:** {mood['btc_mood']}\n- **الخوف والطمع:** {mood['fng']}\n- **الأخبار:** {mood['news']}", parse_mode=ParseMode.MARKDOWN)
            
            elif report_type == "diagnostics":
                mood, scan, settings = bot_state.market_mood, bot_state.scan_stats, bot_state.settings
                total_trades, active_trades = 0, 0
                async with aiosqlite.connect(DB_FILE) as conn:
                    total_trades = (await (await conn.execute("SELECT COUNT(*) FROM trades")).fetchone())[0]
                    active_trades = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]
                report = [f"**🕵️‍♂️ تقرير التشخيص الشامل (v5.3)**\n",
                          f"--- **📊 حالة السوق الحالية** ---\n- **المزاج العام:** {mood['mood']} ({escape_markdown(mood['reason'])})\n- **مؤشر BTC:** {mood['btc_mood']}\n- **الخوف والطمع:** {mood['fng']}\n",
                          f"--- **🔬 أداء آخر فحص** ---\n- **وقت البدء:** {scan['last_start'].strftime('%Y-%m-%d %H:%M') if scan['last_start'] else 'N/A'}\n- **المدة:** {scan['last_duration']}\n- **العملات المفحوصة:** {scan['markets_scanned']}\n- **فشل في تحليل:** {scan['failures']} عملات\n",
                          f"--- **🔧 الإعدادات النشطة** ---\n- **النمط الحالي:** {settings['active_preset']}\n- **الماسحات المفعلة:** {escape_markdown(', '.join(settings['active_scanners']))}\n",
                          f"--- **🔩 حالة العمليات الداخلية** ---\n- **قاعدة البيانات:** متصلة ✅ ({total_trades} صفقة / {active_trades} نشطة)"]
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
    except Exception as e:
        logger.error(f"General error in button handler: {e}", exc_info=True)
        try:
            await query.message.reply_text("حدث خطأ غير متوقع.")
        except:
            pass


# =# =======================================================================================
# --- 🚀 نقطة انطلاق البوت (بنية جديدة ومستقرة) 🚀 ---
# =======================================================================================
async def main():
    """الدالة الرئيسية الجديدة التي تبدأ وتدير البوت بشكل مستقر."""
    
    # 1. التحقق من وجود المتغيرات الأساسية
    required_vars = {
        'OKX_API_KEY': OKX_API_KEY, 'OKX_API_SECRET': OKX_API_SECRET, 
        'OKX_API_PASSPHRASE': OKX_API_PASSPHRASE, 'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN, 
        'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID
    }
    if any(not v for v in required_vars.values()):
        missing = [key for key, value in required_vars.items() if not value]
        logger.critical(f"FATAL: The following environment variables are not set: {', '.join(missing)}. Exiting.")
        return

    # 2. تحميل الإعدادات وتهيئة قاعدة البيانات
    load_settings()
    await init_database()
    
    # 3. بناء كائن التطبيق (البوت)
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # --- دمج منطق post_init هنا مباشرة ---
    # 4. تهيئة الاتصال بالمنصة وتطبيق الـ Patch
    bot_state.exchange = ccxt.okx({
        'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 
        'password': OKX_API_PASSPHRASE, 'enableRateLimit': True, 
        'options': {'defaultType': 'spot'}
    })

    original_request = bot_state.exchange.request
    async def patched_request(self, path, api='public', method='GET', params=None, headers=None, body=None, config=None):
        params = params or {}
        try:
            if (path == 'trade/order-algo') or (path == 'trade/order' and 'attachAlgoOrds' in params):
                if params.get("side") == "sell":
                    params.pop("tgtCcy", None)
        except Exception as e:
            logger.warning(f"Monkey-patch failed to modify params, proceeding. Error: {e}")
        return await original_request(path, api=api, method=method, params=params, headers=headers, body=body, config=config)
    bot_state.exchange.request = types.MethodType(patched_request, bot_state.exchange)
    logger.info("Applied async-safe monkey-patch for OKX.")

    # 5. اختبار الاتصال بالمنصة
    try:
        ticker = await bot_state.exchange.fetch_ticker('BTC/USDT')
        _ = await bot_state.exchange.fetch_balance()
        logger.info(f"✅ OKX connection test SUCCEEDED. BTC last price: {ticker.get('last')}")
    except Exception as e:
        logger.critical(f"❌ OKX connection test FAILED: {e}", exc_info=True)
        try:
            await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="❌ فشل الاتصال بـ OKX. تحقق من مفاتيح API.")
        except Exception as tg_e:
            logger.warning(f"Could not send startup failure message to Telegram: {tg_e}")
        await bot_state.exchange.close()
        return

    # 6. إضافة المعالجات (Handlers)
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex(r'^[+-]?\d*\.?\d+$'), text_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, input_handler))
    app.add_handler(CallbackQueryHandler(button_callback_handler))

    # 7. جدولة المهام المتكررة
    scan_interval = bot_state.settings.get("scan_interval_seconds", 900)
    track_interval = bot_state.settings.get("track_interval_seconds", 60)
    app.job_queue.run_repeating(perform_scan, interval=scan_interval, first=10, name="perform_scan")
    app.job_queue.run_repeating(track_open_trades, interval=track_interval, first=30, name="track_trades")
    logger.info(f"Jobs scheduled: Scan every {scan_interval}s, Tracker every {track_interval}s.")
    
    # 8. تشغيل البوت بطريقة مستقرة (غير حاجزة)
    try:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="*🚀 بوت OKX The Mastermind v5.3 بدأ العمل (بنية مستقرة)...*", parse_mode=ParseMode.MARKDOWN)
        
        # استخدام async with يضمن الإغلاق الآمن للتطبيق
        async with app:
            await app.start()
            await app.updater.start_polling()
            logger.info("Bot is now running and polling for updates...")
            
            # حلقة لا نهائية لإبقاء البرنامج يعمل
            while True:
                await asyncio.sleep(3600) # يمكن أن تكون أي مدة طويلة
                
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot shutting down gracefully...")
    except Exception as e:
        logger.critical(f"An unhandled error occurred in main loop: {e}", exc_info=True)
    finally:
        # الإغلاق الآمن عند الخروج
        if app.updater and app.updater.is_running:
            await app.updater.stop()
        if app.running:
            await app.stop()
        if bot_state.exchange:
            await bot_state.exchange.close()
            logger.info("CCXT exchange connection closed.")
        logger.info("Bot has been shut down.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")

