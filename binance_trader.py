# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 OKX Guardian Trader v22.0 🚀 ---
# =======================================================================================
# This is the definitive, reliable version built on proven, stable components.
#
# ARCHITECTURE:
# 1. FOCUS: Operates exclusively on OKX.
# 2. INTELLIGENCE: Uses a multi-strategy brain with a powerful asset blacklist.
# 3. EXECUTION: Employs the "Guardian" protocol. After a buy order is filled,
#    the bot itself takes full responsibility for monitoring the trade's price
#    in real-time via a dedicated Public WebSocket. It actively watches to
#    close the trade at TP or SL.
#
# DEPRECATED AND REMOVED:
# - The unreliable "Postman" (OCO) protocol due to its instability.
# - All multi-exchange logic.
# =======================================================================================

# --- Core Libraries ---
import asyncio
import os
import logging
import json
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import hmac
import base64

# --- Database & Networking ---
import aiosqlite
import websockets
import websockets.exceptions

# --- Data Analysis & CCXT ---
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt

# --- Telegram & Environment ---
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest, TimedOut
from dotenv import load_dotenv

# --- Optional Libraries ---
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Scipy not found. RSI Divergence strategy will be disabled.")

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

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'guardian_trader_v22.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'guardian_trader_settings_v22.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Guardian_Trader")

# =======================================================================================
# --- 🔬 Global Bot State & Locks 🔬 ---
# =======================================================================================
class BotState:
    def __init__(self):
        self.settings = {}
        self.last_signal_time = {}
        self.application = None
        self.exchange = None
        # --- The Guardian's Tools ---
        self.private_ws = None # The 'ear' for order confirmations
        self.public_ws = None  # The 'eyes' for price tracking
        self.trade_guardian = None # The 'brain' for managing active trades

bot_data = BotState()
scan_lock = asyncio.Lock()
trade_management_lock = asyncio.Lock()

# =======================================================================================
# --- 💡 Default Settings & Filters 💡 ---
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
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "rsi_divergence", "supertrend_pullback"],
    "min_signal_strength": 1,
    "asset_blacklist": [
        "USDC", "DAI", "TUSD", "FDUSD", "USDD", "PYUSD", # Stablecoins
        "BNB", "OKB", "KCS", "BGB", "MX", "GT", "HT",    # Exchange Tokens
        "BTC", "ETH"                                    # Giants
    ],
    "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "max_spread_percent": 0.5},
    "strategy_params": {
        "momentum_breakout": {"rsi_max_level": 68},
        "breakout_squeeze_pro": {"bbands_period": 20, "keltner_period": 20, "keltner_atr_multiplier": 1.5},
        "rsi_divergence": {"rsi_period": 14, "lookback_period": 35},
        "supertrend_pullback": {"atr_period": 10, "atr_multiplier": 3.0}
    }
}
STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي", "breakout_squeeze_pro": "اختراق انضغاطي",
    "rsi_divergence": "دايفرجنس RSI", "supertrend_pullback": "انعكاس سوبرترند"
}

# =======================================================================================
# --- Helper & Settings Management ---
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

# =======================================================================================
# --- 💽 Database Management 💽 ---
# =======================================================================================
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
        logger.info("Guardian database initialized successfully.")
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
    except Exception as e: logger.error(f"DB Log Pending Error: {e}")

# =======================================================================================
# --- 🧠 Advanced Scanners (The Brain) 🧠 ---
# =======================================================================================
# [Analysis functions remain unchanged]
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

def analyze_momentum_breakout(df, params):
    df.ta.vwap(append=True); df.ta.bbands(length=20, append=True); df.ta.macd(append=True); df.ta.rsi(append=True)
    last, prev = df.iloc[-2], df.iloc[-3]
    macd_col, macds_col, bbu_col, rsi_col = find_col(df.columns, "MACD_"), find_col(df.columns, "MACDs_"), find_col(df.columns, "BBU_"), find_col(df.columns, "RSI_")
    if not all([macd_col, macds_col, bbu_col, rsi_col, "VWAP_D"]): return None
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and last[rsi_col] < params.get('rsi_max_level', 68)):
        return {"reason": "momentum_breakout"}
    return None

def analyze_breakout_squeeze_pro(df, params):
    p = params
    df.ta.bbands(length=p['bbands_period'], append=True); df.ta.kc(length=p['keltner_period'], scalar=p['keltner_atr_multiplier'], append=True); df.ta.obv(append=True)
    bbu_col, bbl_col, kcu_col, kcl_col = find_col(df.columns, f"BBU_"), find_col(df.columns, f"BBL_"), find_col(df.columns, f"KCUe_"), find_col(df.columns, f"KCLEe_")
    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]
    if is_in_squeeze and (last['close'] > last[bbu_col]) and (df['OBV'].iloc[-2] > df['OBV'].iloc[-3]):
        return {"reason": "breakout_squeeze_pro"}
    return None

def analyze_rsi_divergence(df, params):
    if not SCIPY_AVAILABLE: return None
    p = params
    df.ta.rsi(length=p['rsi_period'], append=True)
    rsi_col = find_col(df.columns, f"RSI_")
    if not rsi_col or df[rsi_col].isnull().all(): return None
    subset = df.iloc[-p['lookback_period']:].copy()
    price_troughs_idx, _ = find_peaks(-subset['low'], distance=5)
    rsi_troughs_idx, _ = find_peaks(-subset[rsi_col], distance=5)
    if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
        p_low1_idx, p_low2_idx = price_troughs_idx[-2], price_troughs_idx[-1]
        r_low1_idx, r_low2_idx = rsi_troughs_idx[-2], rsi_troughs_idx[-1]
        is_divergence = (subset.iloc[p_low2_idx]['low'] < subset.iloc[p_low1_idx]['low'] and subset.iloc[r_low2_idx][rsi_col] > subset.iloc[r_low1_idx][rsi_col])
        if is_divergence: return {"reason": "rsi_divergence"}
    return None

def analyze_supertrend_pullback(df, params):
    p = params
    df.ta.supertrend(length=p['atr_period'], multiplier=p['atr_multiplier'], append=True)
    st_dir_col = find_col(df.columns, f"SUPERTd_")
    if not st_dir_col: return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if prev[st_dir_col] == -1 and last[st_dir_col] == 1:
        return {"reason": "supertrend_pullback"}
    return None

SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout, "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "rsi_divergence": analyze_rsi_divergence, "supertrend_pullback": analyze_supertrend_pullback
}

# =======================================================================================
# --- 🛡️ The Guardian Protocol (Reliable Execution Body) 🛡️ ---
# =======================================================================================
async def handle_filled_buy_order(order_data):
    symbol = order_data['instId'].replace('-', '/'); order_id = order_data['ordId']
    filled_qty = float(order_data.get('fillSz', 0)); avg_price = float(order_data.get('avgPx', 0))
    if filled_qty == 0 or avg_price == 0: return

    logger.info(f"✅ Order {order_id} filled for {symbol}. Activating Guardian protocol.")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM trades WHERE order_id = ?", (order_id,))
            trade = dict(await cursor.fetchone())
            
            risk = avg_price - trade['stop_loss']
            new_take_profit = avg_price + (risk * bot_data.settings['risk_reward_ratio'])

            await conn.execute(
                "UPDATE trades SET status = 'active', entry_price = ?, quantity = ?, take_profit = ? WHERE order_id = ?",
                (avg_price, filled_qty, new_take_profit, order_id)
            )
            await conn.commit()

        # This is the critical step: Tell the Guardian to start watching this symbol
        await bot_data.public_ws.subscribe([symbol])
        
        success_msg = f"**✅ تم تنفيذ الشراء | {symbol}**\n\nالصفقة الآن نشطة والحارس الأمين يراقبها لحظة بلحظة."
        await safe_send_message(bot_data.application.bot, success_msg)
    except Exception as e:
        logger.error(f"Handle Fill Error for {order_id}: {e}", exc_info=True)

class TradeGuardian:
    """The heart of the execution body. Monitors active trades via live WebSocket ticker data."""
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
                    # This logic handles Trailing Stop Loss if enabled
                    settings = bot_data.settings
                    if settings['trailing_sl_enabled']:
                        new_highest_price = max(trade.get('highest_price', 0), current_price)
                        if new_highest_price > trade.get('highest_price', 0):
                            await conn.execute("UPDATE trades SET highest_price = ? WHERE id = ?", (new_highest_price, trade['id']))

                        if not trade['trailing_sl_active'] and current_price >= trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100):
                            trade['trailing_sl_active'] = True
                            await conn.execute("UPDATE trades SET trailing_sl_active = 1 WHERE id = ?", (trade['id'],))
                            logger.info(f"Guardian: TSL activated for trade #{trade['id']}.")
                            await safe_send_message(self.application.bot, f"**🚀 تأمين الأرباح! | #{trade['id']} {symbol}**")

                        if trade['trailing_sl_active']:
                            new_sl = new_highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                            if new_sl > trade['stop_loss']:
                                trade['stop_loss'] = new_sl
                                await conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_sl, trade['id']))
                    await conn.commit()
                
                # Check for TP/SL breach
                if current_price >= trade['take_profit']: await self._close_trade(trade, "ناجحة (TP)", current_price)
                elif current_price <= trade['stop_loss']: await self._close_trade(trade, "فاشلة (SL)", current_price)

            except Exception as e: logger.error(f"Guardian Ticker Error for {symbol}: {e}", exc_info=True)

    async def _close_trade(self, trade, reason, close_price):
        symbol = trade['symbol']
        logger.info(f"Guardian: Closing trade #{trade['id']} for {symbol}. Reason: {reason}")
        try:
            # In a real scenario, we place the MARKET SELL order here
            # For simplicity, we assume it's filled instantly at the trigger price
            await bot_data.exchange.create_market_sell_order(symbol, trade['quantity'])
            
            pnl = (close_price - trade['entry_price']) * trade['quantity']
            async with aiosqlite.connect(DB_FILE) as conn:
                await conn.execute("UPDATE trades SET status = ?, exit_price = ? WHERE id = ?", (reason, close_price, trade['id']))
                await conn.commit()

            await bot_data.public_ws.unsubscribe([symbol])
            pnl_percent = (close_price / trade['entry_price'] - 1) * 100
            emoji = "✅" if pnl > 0 else "🛑"
            msg = (f"**{emoji} تم إغلاق الصفقة | {symbol}**\n**السبب:** {reason}\n**الربح/الخسارة:** `${pnl:,.2f}` ({pnl_percent:+.2f}%)")
            await safe_send_message(self.application.bot, msg)
        except Exception as e: logger.critical(f"Guardian Close Trade Error #{trade['id']}: {e}", exc_info=True)

    async def sync_subscriptions(self):
        try:
            async with aiosqlite.connect(DB_FILE) as conn:
                cursor = await conn.execute("SELECT DISTINCT symbol FROM trades WHERE status = 'active'")
                active_symbols = [row[0] for row in await cursor.fetchall()]
            if active_symbols:
                logger.info(f"Guardian: Syncing subscriptions for: {active_symbols}")
                await bot_data.public_ws.subscribe(active_symbols)
        except Exception as e: logger.error(f"Guardian Sync Error: {e}")

class PrivateWebSocketManager:
    # [Identical to previous versions, handles order confirmations]
    def __init__(self): self.ws_url = "wss://ws.okx.com:8443/ws/v5/private"; self.websocket = None
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
                    asyncio.create_task(handle_filled_buy_order(order))
    async def run(self):
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    self.websocket = ws; logger.info("✅ [WS-Private] Connected.")
                    await ws.send(json.dumps({"op": "login", "args": self._get_auth_args()}))
                    login_response = json.loads(await ws.recv())
                    if login_response.get('code') == '0':
                        logger.info("🔐 [WS-Private] Authenticated.")
                        await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "orders", "instType": "SPOT"}]}))
                        async for msg in ws: await self._message_handler(msg)
                    else: logger.error(f"🔥 [WS-Private] Auth failed: {login_response}")
            except Exception as e: logger.error(f"🔥 [WS-Private] Connection Error: {e}")
            self.websocket = None; logger.warning("⚠️ [WS-Private] Disconnected. Reconnecting..."); await asyncio.sleep(5)

class PublicWebSocketManager:
    # [Identical to previous versions, handles price tickers for the Guardian]
    def __init__(self, handler_coro): self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"; self.handler = handler_coro; self.subscriptions = set(); self.websocket = None
    async def _send_op(self, op, symbols):
        if not symbols or self.websocket is None: return
        try: await self.websocket.send(json.dumps({"op": op, "args": [{"channel": "tickers", "instId": s.replace('/', '-')} for s in symbols]}))
        except websockets.exceptions.ConnectionClosed: logger.warning(f"Could not send '{op}' op; websocket is closed.")
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
                    self.websocket = ws; logger.info("✅ [WS-Public] Guardian's eyes are open.")
                    if self.subscriptions: await self.subscribe(list(self.subscriptions))
                    async for msg in ws:
                        if msg == 'ping': await ws.send('pong'); continue
                        data = json.loads(msg)
                        if data.get('arg', {}).get('channel') == 'tickers' and 'data' in data:
                            for ticker in data['data']: asyncio.create_task(self.handler(ticker))
            except Exception as e: logger.error(f"🔥 [WS-Public] Error: {e}")
            self.websocket = None; logger.warning("⚠️ [WS-Public] Disconnected. Reconnecting..."); await asyncio.sleep(5)

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

async def worker(queue, results_list):
    settings, exchange = bot_data.settings, bot_data.exchange
    while not queue.empty():
        market = await queue.get(); symbol = market['symbol']
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            if len(ohlcv) < 200: continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)

            confirmed_reasons = {SCANNERS[name](df.copy(), params)['reason']
                                 for name in settings['active_scanners']
                                 if (params := settings.get('strategy_params', {}).get(name)) and SCANNERS[name](df.copy(), params)}
            
            if len(confirmed_reasons) >= settings.get("min_signal_strength", 1):
                reason_str = ' + '.join(confirmed_reasons)
                entry_price = df.iloc[-2]['close']
                df.ta.atr(length=14, append=True)
                atr = df.iloc[-2].get(find_col(df.columns, "ATRr_14"), 0)
                risk = atr * settings['atr_sl_multiplier']
                stop_loss = entry_price - risk
                take_profit = entry_price + (risk * settings['risk_reward_ratio'])
                results_list.append({"symbol": symbol, "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "reason": reason_str})
        except Exception as e: logger.debug(f"Worker error for {symbol}: {e}")
        finally: queue.task_done()

async def initiate_real_trade(signal):
    try:
        settings, exchange = bot_data.settings, bot_data.exchange
        trade_size = settings['real_trade_size_usdt']
        amount = trade_size / signal['entry_price']
        logger.info(f"--- INITIATING REAL TRADE: {signal['symbol']} ---")
        buy_order = await exchange.create_market_buy_order(signal['symbol'], amount)
        await log_pending_trade_to_db(signal, buy_order)
        await safe_send_message(bot_data.application.bot, f"🚀 تم إرسال أمر شراء لـ `{signal['symbol']}`.")
    except Exception as e:
        logger.error(f"REAL TRADE FAILED for {signal['symbol']}: {e}", exc_info=True)
        await safe_send_message(bot_data.application.bot, f"🔥 فشل فتح صفقة لـ `{signal['symbol']}`.")

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        logger.info("--- Starting new OKX-focused market scan... ---")
        async with aiosqlite.connect(DB_FILE) as conn:
            active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active' OR status = 'pending'")).fetchone())[0]

        settings = bot_data.settings
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
            if active_trades_count >= settings['max_concurrent_trades']: break
            if time.time() - bot_data.last_signal_time.get(signal['symbol'], 0) > (SCAN_INTERVAL_SECONDS * 2):
                bot_data.last_signal_time[signal['symbol']] = time.time()
                await initiate_real_trade(signal)
                active_trades_count += 1
                await asyncio.sleep(2)

# =======================================================================================
# --- 🤖 Telegram UI & Bot Startup 🤖 ---
# =======================================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("أهلاً بك في OKX Guardian Trader v22.0")

async def post_init(application: Application):
    bot_data.application = application
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, TELEGRAM_BOT_TOKEN]):
        logger.critical("FATAL: Missing critical API or Bot keys."); return

    try:
        config = {'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE, 'enableRateLimit': True}
        bot_data.exchange = ccxt.okx(config)
        await bot_data.exchange.fetch_balance()
        logger.info("✅ Successfully connected to OKX.")
    except Exception as e:
        logger.critical(f"🔥 FATAL: Could not connect to OKX: {e}"); return

    # --- Initialize and start the Guardian's components ---
    bot_data.trade_guardian = TradeGuardian(application)
    bot_data.public_ws = PublicWebSocketManager(bot_data.trade_guardian.handle_ticker_update)
    bot_data.private_ws = PrivateWebSocketManager()
    asyncio.create_task(bot_data.public_ws.run())
    asyncio.create_task(bot_data.private_ws.run())
    
    logger.info("Waiting 5s for WebSocket connections...")
    await asyncio.sleep(5)
    await bot_data.trade_guardian.sync_subscriptions() # Important for restarts

    application.job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name="perform_scan")
    logger.info(f"Scanner scheduled for every {SCAN_INTERVAL_SECONDS} seconds.")
    await safe_send_message(application.bot, "*🚀 OKX Guardian Trader v22.0 بدأ العمل...*")
    logger.info("--- Bot is now fully operational ---")

async def post_shutdown(application: Application):
    if bot_data.exchange: await bot_data.exchange.close()
    logger.info("Bot has shut down.")

def main():
    logger.info("--- Starting OKX Guardian Trader v22.0 ---")
    load_settings(); asyncio.run(init_database())
    app_builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    app_builder.post_init(post_init).post_shutdown(post_shutdown)
    application = app_builder.build()
    
    application.add_handler(CommandHandler("start", start_command))
    # Add other handlers as needed for the UI
    
    application.run_polling()

if __name__ == '__main__':
    main()

