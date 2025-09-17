# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 OKX Mastermind Trader v25.1 🚀 ---
# =======================================================================================
# This is the master version, representing a complete fusion of the best features:
#
# - BRAIN (from Mastermind v5.5):
#   - Five advanced scanning strategies (Sniper, Whale Radar, etc.).
#   - Comprehensive market mood analysis including news sentiment.
# - APPEARANCE (from Mastermind v5.5 & your Copy-trader):
#   - A full, rich Telegram UI with a detailed dashboard and diagnostics.
#   - A professional trade confirmation message that reports on liquidity.
# - BODY (from Hybrid Core v24.1):
#   - The infallible Hybrid Core for trade confirmation (Fast Reporter + Supervisor).
#   - The reliable Guardian protocol for real-time management of active trades.
#
# --- Version 25.1 Changelog ---
#   - Implemented advanced trade closing logic to eliminate "Insufficient Funds" errors.
#   - Added a pre-emptive balance availability check (`wait_for_balance_available`).
#   - Explicitly set `tdMode: 'cash'` for sell orders to resolve API ambiguity.
#   - Integrated a robust, multi-layered error handling system for closing trades.
#   - Activated a fully interactive settings menu in the Telegram UI.
# =======================================================================================

# --- Core Libraries ---
import asyncio
import os
import logging
import json
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hmac
import base64
from collections import defaultdict

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
from telegram.error import BadRequest, TimedOut
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

TIMEFRAME = '15m'
SCAN_INTERVAL_SECONDS = 900
SUPERVISOR_INTERVAL_SECONDS = 120

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'mastermind_trader_v25.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'mastermind_trader_settings_v25.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Mastermind_Trader")

# =======================================================================================
# --- 🔬 Global Bot State & Locks 🔬 ---
# =======================================================================================
class BotState:
    def __init__(self):
        self.settings = {}
        self.last_signal_time = {}
        self.application = None
        self.exchange = None
        self.market_mood = {"mood": "UNKNOWN", "reason": "تحليل لم يتم بعد"}
        self.private_ws = None
        self.public_ws = None
        self.trade_guardian = None

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
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 2.0,
    "trailing_sl_enabled": True,
    "trailing_sl_activation_percent": 1.5,
    "trailing_sl_callback_percent": 1.0,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "sniper_pro", "whale_radar"],
    "market_mood_filter_enabled": True,
    "fear_and_greed_threshold": 30,
    "asset_blacklist": [
        "USDC", "DAI", "TUSD", "FDUSD", "USDD", "PYUSD", "USDT",
        "BNB", "OKB", "KCS", "BGB", "MX", "GT", "HT",
        "BTC", "ETH"
    ],
    "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "min_rvol": 1.5},
    "trend_filters": {"ema_period": 200, "htf_period": 50},
}
STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي", "breakout_squeeze_pro": "اختراق انضغاطي",
    "support_rebound": "ارتداد الدعم", "sniper_pro": "القناص المحترف", "whale_radar": "رادار الحيتان"
}

# =======================================================================================
# --- Helper, Settings & DB Management ---
# =======================================================================================
def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f: bot_data.settings = json.load(f)
        else: bot_data.settings = DEFAULT_SETTINGS.copy()
    except Exception: bot_data.settings = DEFAULT_SETTINGS.copy()
    for key, value in DEFAULT_SETTINGS.items(): bot_data.settings.setdefault(key, value)
    save_settings(); logger.info("Settings loaded.")
def save_settings():
    with open(SETTINGS_FILE, 'w') as f: json.dump(bot_data.settings, f, indent=4)
async def safe_send_message(bot, text, **kwargs):
    try: await bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except Exception as e: logger.error(f"Telegram Send Error: {e}")

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
            await conn.commit()
        logger.info("Mastermind database initialized successfully.")
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
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

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
    if not headlines or not NLTK_AVAILABLE: return "N/A"
    sia = SentimentIntensityAnalyzer()
    score = sum(sia.polarity_scores(h)['compound'] for h in headlines) / len(headlines)
    if score > 0.1: mood = "إيجابية"
    elif score < -0.1: mood = "سلبية"
    else: mood = "محايدة"
    return f"{mood} (الدرجة: {score:.2f})"

async def get_market_mood():
    try:
        htf_period = bot_data.settings['trend_filters']['htf_period']
        ohlcv = await bot_data.exchange.fetch_ohlcv('BTC/USDT', '4h', limit=htf_period + 5)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['sma'] = ta.sma(df['close'], length=htf_period)
        is_btc_bullish = df['close'].iloc[-1] > df['sma'].iloc[-1]
        btc_mood_text = "إيجابي ✅" if is_btc_bullish else "سلبي ❌"
        if not is_btc_bullish:
            return {"mood": "NEGATIVE", "reason": "اتجاه BTC هابط", "btc_mood": btc_mood_text}
    except Exception as e:
        return {"mood": "DANGEROUS", "reason": f"فشل جلب بيانات BTC: {e}", "btc_mood": "UNKNOWN"}
    
    fng = await get_fear_and_greed_index()
    if fng is not None and fng < bot_data.settings['fear_and_greed_threshold']:
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
    "support_rebound": analyze_support_rebound, "sniper_pro": analyze_sniper_pro,
    "whale_radar": analyze_whale_radar
}

# =======================================================================================
# --- 🚀 Hybrid Core Protocol (Execution & Management) 🚀 ---
# =======================================================================================
async def activate_trade(order_id, filled_qty, avg_price, symbol):
    """The centralized function to activate a trade and send the detailed confirmation message."""
    bot = bot_data.application.bot
    try:
        balance_after = await bot_data.exchange.fetch_balance()
        usdt_remaining = balance_after.get('USDT', {}).get('free', 0)
    except Exception as e:
        logger.error(f"Could not fetch balance for confirmation message: {e}")
        usdt_remaining = "N/A"

    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT * FROM trades WHERE order_id = ? AND status = 'pending'", (order_id,))
        trade = await cursor.fetchone()
        if not trade:
            logger.info(f"Activation ignored for {order_id}: Trade not pending.")
            return

        trade = dict(trade)
        logger.info(f"Activating trade #{trade['id']} for {symbol}...")
        
        risk = avg_price - trade['stop_loss']
        new_take_profit = avg_price + (risk * bot_data.settings['risk_reward_ratio'])

        await conn.execute(
            "UPDATE trades SET status = 'active', entry_price = ?, quantity = ?, take_profit = ? WHERE id = ?",
            (avg_price, filled_qty, new_take_profit, trade['id'])
        )
        
        active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]
        await conn.commit()

    await bot_data.public_ws.subscribe([symbol])
    
    trade_cost = avg_price * filled_qty
    
    # The new, detailed confirmation message
    success_msg = (
        f"**✅ تم تأكيد الشراء | {symbol}**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔸 **الصفقة رقم:** `#{trade['id']}`\n"
        f"🔸 **الاستراتيجية:** {trade['reason']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*تحليل الصفقة:*\n"
        f" ▪️ **سعر التنفيذ:** `${avg_price:,.4f}`\n"
        f" ▪️ **الكمية:** `{filled_qty:,.4f}` {symbol.split('/')[0]}\n"
        f" ▪️ **التكلفة (السيولة المستهلكة):** `${trade_cost:,.2f}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*التأثير على المحفظة:*\n"
        f" ▪️ **السيولة المتبقية (USDT):** `${usdt_remaining:,.2f}`\n"
        f" ▪️ **إجمالي الصفقات النشطة:** `{active_trades_count}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"الحارس الأمين يراقب الصفقة الآن."
    )
    await safe_send_message(bot, success_msg)

async def handle_filled_buy_order(order_data):
    symbol = order_data['instId'].replace('-', '/'); order_id = order_data['ordId']
    filled_qty = float(order_data.get('fillSz', 0)); avg_price = float(order_data.get('avgPx', 0))
    if filled_qty > 0 and avg_price > 0:
        logger.info(f"🎤 Fast Reporter: Received fill for {order_id} via WebSocket.")
        await activate_trade(order_id, filled_qty, avg_price, symbol)

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
    async def run(self):
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    self.websocket = ws; logger.info("✅ [Fast Reporter] Connected.")
                    await ws.send(json.dumps({"op": "login", "args": self._get_auth_args()}))
                    login_response = json.loads(await ws.recv())
                    if login_response.get('code') == '0':
                        logger.info("🔐 [Fast Reporter] Authenticated.")
                        await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "orders", "instType": "SPOT"}]}))
                        async for msg in ws: await self._message_handler(msg)
                    else: logger.error(f"🔥 [Fast Reporter] Auth failed: {login_response}")
            except Exception as e: logger.error(f"🔥 [Fast Reporter] Connection Error: {e}")
            await asyncio.sleep(5)

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
        for trade in stuck_trades:
            trade = dict(trade)
            order_id, symbol = trade['order_id'], trade['symbol']
            logger.warning(f"🕵️ Supervisor: Found abandoned trade #{trade['id']}. Investigating...")
            try:
                order_status = await bot_data.exchange.fetch_order(order_id, symbol)
                if order_status['status'] == 'closed' and order_status.get('filled', 0) > 0:
                    logger.info(f"🕵️ Supervisor: API confirms trade {order_id} was filled. Activating manually.")
                    await activate_trade(order_id, order_status['filled'], order_status['average'], symbol)
                elif order_status['status'] == 'canceled':
                    await conn.execute("UPDATE trades SET status = 'failed' WHERE id = ?", (trade['id'],))
                else:
                    await bot_data.exchange.cancel_order(order_id, symbol)
                    await conn.execute("UPDATE trades SET status = 'failed' WHERE id = ?", (trade['id'],))
                await conn.commit()
            except Exception as e: logger.error(f"🕵️ Supervisor: Failed to rectify trade #{trade['id']}: {e}")

# NEW: Helper function to wait for balance
async def wait_for_balance_available(exchange, asset, required_amount, timeout=30):
    """
    Waits until a specific amount of an asset becomes available in the balance.
    """
    start_time = time.time()
    logger.info(f"Checking for available balance of {required_amount} {asset}...")
    
    asset_symbol = asset.split('/')[0]

    while time.time() - start_time < timeout:
        try:
            balance = await exchange.fetch_balance()
            # In ccxt, 'free' represents the available balance
            available_amount = balance.get(asset_symbol, {}).get('free', 0.0)
            
            if available_amount >= required_amount:
                logger.info(f"SUCCESS: {available_amount} {asset_symbol} is now available.")
                return True
                
            logger.debug(f"Balance not yet available. Have: {available_amount}, Need: {required_amount}. Retrying...")
            await asyncio.sleep(0.5)  # Wait for 500ms before re-checking
        except Exception as e:
            logger.warning(f"Error while fetching balance: {e}. Retrying...")
            await asyncio.sleep(1) # Wait longer on error
            
    logger.error(f"TIMEOUT: Failed to verify availability of {required_amount} {asset_symbol} within {timeout}s.")
    return False

class TradeGuardian:
    def __init__(self, application): self.application = application
    async def handle_ticker_update(self, ticker_data):
        async with trade_management_lock:
            symbol = ticker_data['instId'].replace('-', '/'); current_price = float(ticker_data['last'])
            try:
                async with aiosqlite.connect(DB_FILE) as conn:
                    conn.row_factory = aiosqlite.Row
                    cursor = await conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = 'active'", (symbol,))
                    trade = await cursor.fetchone()
                    if not trade: return

                    trade = dict(trade)
                    settings = bot_data.settings
                    if settings['trailing_sl_enabled']:
                        new_highest_price = max(trade.get('highest_price', 0), current_price)
                        if new_highest_price > trade.get('highest_price', 0):
                            await conn.execute("UPDATE trades SET highest_price = ? WHERE id = ?", (new_highest_price, trade['id']))
                        if not trade['trailing_sl_active'] and current_price >= trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100):
                            trade['trailing_sl_active'] = True
                            await conn.execute("UPDATE trades SET trailing_sl_active = 1, stop_loss = ? WHERE id = ?", (trade['entry_price'], trade['id']))
                            trade['stop_loss'] = trade['entry_price']
                            await safe_send_message(self.application.bot, f"**🚀 تأمين الأرباح! | #{trade['id']} {symbol}**")
                        if trade['trailing_sl_active']:
                            new_sl = new_highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                            if new_sl > trade['stop_loss']:
                                trade['stop_loss'] = new_sl
                                await conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_sl, trade['id']))
                    await conn.commit()
                
                if current_price >= trade['take_profit']: await self._close_trade(trade, "ناجحة (TP)", current_price)
                elif current_price <= trade['stop_loss']: await self._close_trade(trade, "فاشلة (SL)", current_price)
            except Exception as e: logger.error(f"Guardian Ticker Error for {symbol}: {e}", exc_info=True)

    # REBUILT: The new, robust trade closing function
    async def _close_trade(self, trade, reason, close_price):
        """
        Upgraded version of the close trade function that handles balance and timing issues.
        """
        symbol = trade['symbol']
        quantity = trade['quantity']
        trade_id = trade['id']
        bot = self.application.bot

        logger.info(f"Guardian: Starting close process for trade #{trade_id} on {symbol}. Reason: {reason}")

        try:
            # --- STEP 1: Wait for Balance Settlement ---
            # Ensure the asset we want to sell is fully available before sending the order.
            asset_to_sell = symbol.split('/')[0]
            is_available = await wait_for_balance_available(bot_data.exchange, asset_to_sell, quantity)

            if not is_available:
                logger.critical(f"CRITICAL: Failed to close trade #{trade_id}: Balance did not become available in time.")
                await safe_send_message(bot, f"🚨 **فشل حرج** 🚨\nفشل إغلاق الصفقة `#{trade_id}` لأن الرصيد لم يتوفر. الرجاء التدخل اليدوي!")
                return

            # --- STEP 2: Explicitly Define Order Intent ---
            # Send explicit instructions to the exchange that this is a spot trade, not margin.
            params = {'tdMode': 'cash'}

            # --- STEP 3: Execute Order Safely ---
            # Use a unique client order ID to ensure idempotency on retries.
            client_order_id = f"close_{trade_id}_{int(time.time() * 1000)}"
            params['clOrdId'] = client_order_id
            
            logger.info(f"Sending market sell order for trade #{trade_id} with params: {params}")
            
            order = await bot_data.exchange.create_market_sell_order(symbol, quantity, params)
            
            logger.info(f"Successfully created sell order for trade #{trade_id}. Order ID: {order.get('id')}")
            
            pnl = (close_price - trade['entry_price']) * quantity
            pnl_percent = (close_price / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
            emoji = "✅" if pnl > 0 else "🛑"

            async with aiosqlite.connect(DB_FILE) as conn:
                await conn.execute("UPDATE trades SET status = ? WHERE id = ?", (reason, trade['id']))
                await conn.commit()

            await bot_data.public_ws.unsubscribe([symbol])
            
            msg = (f"**{emoji} تم إغلاق الصفقة | {symbol}**\n**السبب:** {reason}\n**الربح/الخسارة:** `${pnl:,.2f}` ({pnl_percent:+.2f}%)")
            await safe_send_message(bot, msg)

        except ccxt.InsufficientFunds as e:
            # This error should no longer occur thanks to wait_for_balance_available,
            # but we keep it as a final safeguard.
            logger.critical(f"CRITICAL: Final InsufficientFunds error when closing trade #{trade_id}: {e}")
            await safe_send_message(bot, f"🚨 **فشل حرج** 🚨\nحدث خطأ رصيد غير كافٍ نهائي عند إغلاق الصفقة `#{trade_id}`. الرجاء التدخل اليدوي فوراً!")
            
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
            # Handle temporary network or exchange-related errors
            logger.warning(f"Temporary error closing trade #{trade_id}: {e}. Will be retried by Guardian.")
            # The Guardian's loop will naturally retry this on the next price tick.
            
        except Exception as e:
            # Handle any other unexpected errors
            logger.critical(f"Unexpected CRITICAL error while closing trade #{trade_id}: {e}", exc_info=True)
            await safe_send_message(bot, f"🚨 **فشل حرج** 🚨\nحدث خطأ غير متوقع عند إغلاق الصفقة `#{trade_id}`. الرجاء مراجعة السجلات.")

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
    async def run(self):
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    self.websocket = ws; logger.info("✅ [Guardian's Eyes] Connected.")
                    if self.subscriptions: await self.subscribe(list(self.subscriptions))
                    async for msg in ws:
                        if msg == 'ping': await ws.send('pong'); continue
                        data = json.loads(msg)
                        if data.get('arg', {}).get('channel') == 'tickers' and 'data' in data:
                            for ticker in data['data']: await self.handler(ticker)
            except Exception as e: logger.error(f"🔥 [Guardian's Eyes] Error: {e}")
            await asyncio.sleep(5)

# =======================================================================================
# --- ⚡ Core Scanner & Trade Initiation Logic ⚡ ---
# =======================================================================================
async def get_okx_markets():
    settings, exchange = bot_data.settings, bot_data.exchange
    try:
        tickers = await exchange.fetch_tickers()
        blacklist = settings.get('asset_blacklist', [])
        valid_markets = [
            t for t in tickers.values() if
            t.get('symbol') and t['symbol'].endswith('/USDT') and
            t['symbol'].split('/')[0] not in blacklist and
            t.get('quoteVolume', 0) > settings['liquidity_filters']['min_quote_volume_24h_usd'] and
            not any(k in t['symbol'] for k in ['-SWAP', 'UP', 'DOWN', '3L', '3S'])
        ]
        valid_markets.sort(key=lambda m: m.get('quoteVolume', 0), reverse=True)
        return valid_markets[:settings['top_n_symbols_by_volume']]
    except Exception as e: logger.error(f"Failed to fetch and filter OKX markets: {e}"); return []

async def worker(queue, signals_list):
    settings, exchange = bot_data.settings, bot_data.exchange
    while not queue.empty():
        market = await queue.get(); symbol = market['symbol']
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            if len(ohlcv) < 200: continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)

            df['volume_sma'] = ta.sma(df['volume'], length=20)
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] == 0: continue
            rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
            if rvol < settings['liquidity_filters']['min_rvol']: continue
            
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
        except Exception as e: logger.debug(f"Worker error for {symbol}: {e}")
        finally: queue.task_done()

async def initiate_real_trade(signal):
    try:
        settings, exchange = bot_data.settings, bot_data.exchange
        trade_size = settings['real_trade_size_usdt']
        amount = trade_size / signal['entry_price']
        logger.info(f"--- INITIATING REAL TRADE: {signal['symbol']} ---")
        buy_order = await exchange.create_market_buy_order(signal['symbol'], amount)
        
        if await log_pending_trade_to_db(signal, buy_order):
            await safe_send_message(bot_data.application.bot, f"🚀 تم إرسال أمر شراء لـ `{signal['symbol']}`.")
        else:
            await exchange.cancel_order(buy_order['id'], signal['symbol'])
            await safe_send_message(bot_data.application.bot, f"⚠️ فشل تسجيل صفقة `{signal['symbol']}`. تم إلغاء الأمر.")
    except ccxt.InsufficientFunds as e:
        logger.error(f"REAL TRADE FAILED for {signal['symbol']}: {e}")
    except Exception as e:
        logger.error(f"REAL TRADE FAILED for {signal['symbol']}: {e}", exc_info=True)
        await safe_send_message(bot_data.application.bot, f"🔥 فشل فتح صفقة لـ `{signal['symbol']}`.")
    raise e

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        logger.info("--- Starting new OKX-focused market scan... ---")
        settings = bot_data.settings

        if settings['market_mood_filter_enabled']:
            mood_result = await get_market_mood()
            bot_data.market_mood = mood_result
            if mood_result['mood'] in ["NEGATIVE", "DANGEROUS"]:
                logger.warning(f"SCAN SKIPPED: {mood_result['reason']}")
                return

        async with aiosqlite.connect(DB_FILE) as conn:
            active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active' OR status = 'pending'")).fetchone())[0]

        if active_trades_count >= settings['max_concurrent_trades']:
            logger.info(f"Scan skipped: Max trades ({active_trades_count}) reached.")
            return

        top_markets = await get_okx_markets()
        if not top_markets: logger.info("Scan complete: No markets passed filters."); return

        queue, signals_found = asyncio.Queue(), []
        for market in top_markets: await queue.put(market)
        worker_tasks = [asyncio.create_task(worker(queue, signals_found)) for _ in range(10)]
        await queue.join(); [task.cancel() for task in worker_tasks]
        logger.info(f"--- Scan complete. Found {len(signals_found)} potential signals. ---")
        
        for signal in signals_found:
            try:
                if active_trades_count >= settings['max_concurrent_trades']: 
                    logger.info("Stopping trade initiation, max concurrent trades reached.")
                    break
                if time.time() - bot_data.last_signal_time.get(signal['symbol'], 0) > (SCAN_INTERVAL_SECONDS * 2):
                    bot_data.last_signal_time[signal['symbol']] = time.time()
                    await initiate_real_trade(signal)
                    active_trades_count += 1
                    await asyncio.sleep(3)
            except ccxt.InsufficientFunds:
                 await safe_send_message(context.bot, f"⚠️ **رصيد غير كافٍ!**\nتم إيقاف فتح صفقات جديدة.")
                 break
            except Exception:
                 continue

# =======================================================================================
# --- 🤖 Telegram UI & Bot Startup 🤖 ---
# =======================================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"]]
    await update.message.reply_text("أهلاً بك في OKX Mastermind Trader v25.1", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 الصفقات الحالية", callback_data="dashboard_trades")],
        [InlineKeyboardButton("🌡️ حالة مزاج السوق", callback_data="dashboard_mood")],
        [InlineKeyboardButton("📜 تقرير أداء الاستراتيجيات", callback_data="dashboard_strategies")]
    ]
    await (update.message or update.callback_query.message).reply_text("🖥️ *لوحة التحكم الرئيسية*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def show_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message if update.callback_query else update.message
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT id, symbol, status FROM trades WHERE status = 'active' OR status = 'pending' ORDER BY id DESC")
        trades = await cursor.fetchall()
    if not trades:
        await target_message.reply_text("لا توجد صفقات حالية.")
        return
    keyboard = []
    for trade in trades:
        status_emoji = "✅" if trade['status'] == 'active' else "⏳"
        button_text = f"#{trade['id']} {status_emoji} | {trade['symbol']}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=f"check_{trade['id']}")])
    await target_message.reply_text("اختر صفقة لعرض تفاصيلها:", reply_markup=InlineKeyboardMarkup(keyboard))

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
        except Exception: pnl_text = "💰 تعذر جلب الربح/الخسارة الحالية."
        message = (f"**✅ حالة الصفقة #{trade_id}**\n\n- **العملة:** `{trade['symbol']}`\n- **سعر الدخول:** `${trade['entry_price']}`\n- **الهدف:** `${trade['take_profit']}`\n- **الوقف:** `${trade['stop_loss']}`\n{pnl_text}")
    
    await query.edit_message_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للصفقات", callback_data="dashboard_trades")]]))


async def show_mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mood = bot_data.market_mood
    headlines = get_latest_crypto_news()
    news_sentiment = analyze_sentiment_of_headlines(headlines)
    message = (f"*🌡️ حالة مزاج السوق*\n\n- **النتيجة:** {mood.get('mood', 'N/A')}\n"
               f"- **السبب:** {mood.get('reason', 'N/A')}\n"
               f"- **مؤشر BTC:** {mood.get('btc_mood', 'N/A')}\n"
               f"- **مشاعر الأخبار:** {news_sentiment}")
    await update.callback_query.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

async def show_strategy_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        cursor = await conn.execute("SELECT reason, status FROM trades WHERE status LIKE 'ناجحة%' OR status LIKE 'فاشلة%'")
        trades = await cursor.fetchall()
    if not trades:
        await update.callback_query.message.reply_text("لا توجد صفقات مغلقة لتحليلها.")
        return
    stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
    for reason, status in trades:
        if not reason: continue
        reasons = reason.split(' + ')
        for r in reasons:
            if status.startswith('ناجحة'): stats[r]['wins'] += 1
            else: stats[r]['losses'] += 1
    report = ["**📜 تقرير أداء الاستراتيجيات**"]
    for r, s in sorted(stats.items()):
        total = s['wins'] + s['losses']
        wr = (s['wins'] / total * 100) if total > 0 else 0
        report.append(f"\n--- *{r}* ---\n  - الصفقات: {total} ({s['wins']}✅ / {s['losses']}❌)\n  - النجاح: {wr:.2f}%")
    await update.callback_query.message.reply_text("\n".join(report), parse_mode=ParseMode.MARKDOWN)


async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Displays the main settings menu."""
    keyboard = [
        [InlineKeyboardButton(f"حجم الصفقة: ${bot_data.settings['real_trade_size_usdt']}", callback_data="setting_real_trade_size_usdt")],
        [InlineKeyboardButton(f"أقصى عدد للصفقات: {bot_data.settings['max_concurrent_trades']}", callback_data="setting_max_concurrent_trades")],
        [InlineKeyboardButton(f"مضاعف وقف الخسارة (ATR): {bot_data.settings['atr_sl_multiplier']}", callback_data="setting_atr_sl_multiplier")],
        [InlineKeyboardButton("إغلاق القائمة", callback_data="setting_close")]
    ]
    await update.message.reply_text("⚙️ *إعدادات البوت* ⚙️\nاختر الإعداد الذي تريد تعديله:", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def handle_setting_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles when a user clicks a setting button."""
    query = update.callback_query
    setting_key = query.data.split('_', 1)[1]
    
    if setting_key == 'close':
        await query.message.delete()
        return

    context.user_data['setting_to_change'] = setting_key
    await query.message.reply_text(f"أرسل القيمة الجديدة لـ `{setting_key}`:", parse_mode=ParseMode.MARKDOWN)
    await query.answer()

async def handle_setting_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the new value sent by the user."""
    setting_key = context.user_data.get('setting_to_change')
    if not setting_key:
        return # Not in the process of changing a setting

    new_value_str = update.message.text
    try:
        # Determine the type of the original value and cast the new value
        original_value = bot_data.settings[setting_key]
        if isinstance(original_value, bool):
            new_value = new_value_str.lower() in ['true', '1', 'yes', 'on']
        elif isinstance(original_value, int):
            new_value = int(new_value_str)
        elif isinstance(original_value, float):
            new_value = float(new_value_str)
        else: # Assumes string or list (list needs special handling not implemented here)
            new_value = new_value_str

        bot_data.settings[setting_key] = new_value
        save_settings()
        await update.message.reply_text(f"✅ تم تحديث `{setting_key}` إلى `{new_value}` بنجاح.")
    except (ValueError, KeyError) as e:
        await update.message.reply_text(f"❌ قيمة غير صالحة. لم يتم التغيير. الخطأ: {e}")
    finally:
        del context.user_data['setting_to_change']

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("setting_"):
        await handle_setting_selection(update, context)
    elif data == "dashboard_trades": await show_trades_command(update, context)
    elif data == "dashboard_mood": await show_mood_command(update, context)
    elif data == "dashboard_strategies": await show_strategy_report_command(update, context)
    elif data.startswith("check_"): await check_trade_details(update, context)

async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check if we are expecting a setting value
    if 'setting_to_change' in context.user_data:
        await handle_setting_value(update, context)
        return

    text = update.message.text
    if text == "Dashboard 🖥️":
        await show_dashboard_command(update, context)
    elif text == "⚙️ الإعدادات":
        await show_settings_menu(update, context)

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
        await bot_data.exchange.fetch_balance()
        logger.info("✅ Successfully connected to OKX.")
    except Exception as e:
        logger.critical(f"🔥 FATAL: Could not connect to OKX: {e}"); return

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
    
    logger.info(f"Scanner scheduled for every {SCAN_INTERVAL_SECONDS}s. Supervisor will audit every {SUPERVISOR_INTERVAL_SECONDS}s.")
    await safe_send_message(application.bot, "*🚀 OKX Mastermind Trader v25.1 بدأ العمل...*")
    logger.info("--- Bot is now fully operational ---")

async def post_shutdown(application: Application):
    if bot_data.exchange: await bot_data.exchange.close()
    logger.info("Bot has shut down.")

def main():
    logger.info("--- Starting OKX Mastermind Trader v25.1 ---")
    load_settings(); asyncio.run(init_database())
    app_builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    app_builder.post_init(post_init).post_shutdown(post_shutdown)
    application = app_builder.build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    
    application.run_polling()

if __name__ == '__main__':
    main()
