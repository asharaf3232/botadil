# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 OKX Ultimate Trader v21.0 (The Postman) 🚀 ---
# =======================================================================================
# This is the definitive version, fusing the best elements from all predecessors.
#
# ARCHITECTURE:
# 1. FOCUS: Operates exclusively on OKX for both scanning and real trading.
# 2. INTELLIGENCE: Utilizes a multi-strategy analytical brain with advanced asset filters.
# 3. EXECUTION: Employs the "Postman" protocol. After a buy order is filled,
#    it immediately dispatches an OCO (One-Cancels-the-Other) order to the
#    exchange itself. This ensures trades are protected by TP/SL directly on
#    the exchange servers, offering maximum reliability and speed.
#
# DEPRECATED:
# - Multi-exchange scanning logic.
# - The self-hosted TradeGuardian and Public WebSocket price tracker are now
#   obsolete, replaced by the far superior server-side OCO protection.
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
from collections import defaultdict
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

# --- API Keys & Telegram ---
OKX_API_KEY = os.getenv('OKX_API_KEY')
OKX_API_SECRET = os.getenv('OKX_API_SECRET')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# --- Bot Settings ---
TIMEFRAME = '15m'
SCAN_INTERVAL_SECONDS = 900

# --- File Paths & Logging ---
APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'ultimate_trader_v21.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'ultimate_trader_settings_v21.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Ultimate_Trader")

# =======================================================================================
# --- 🔬 Global Bot State & Locks 🔬 ---
# =======================================================================================
class BotState:
    def __init__(self):
        self.settings = {}
        self.last_signal_time = {}
        self.application = None
        self.exchange = None # This will be our single, powerful connection to OKX
        self.private_ws = None

bot_data = BotState()
scan_lock = asyncio.Lock()

# =======================================================================================
# --- 💡 Default Settings & Filters 💡 ---
# =======================================================================================
DEFAULT_SETTINGS = {
    "real_trade_size_usdt": 15.0,
    "max_concurrent_trades": 5,
    "top_n_symbols_by_volume": 300,
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 2.0,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "rsi_divergence", "supertrend_pullback"],
    "min_signal_strength": 1,
    # --- The new advanced asset filter ---
    "asset_blacklist": [
        # Stablecoins
        "USDC", "DAI", "TUSD", "FDUSD", "USDD", "PYUSD",
        # Exchange Tokens (for Sharia compliance)
        "BNB", "OKB", "KCS", "BGB", "MX", "GT", "HT",
        # "Giants" (for strategic reasons)
        "BTC", "ETH"
    ],
    "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "max_spread_percent": 0.5},
    "strategy_params": {
        "momentum_breakout": {"rsi_max_level": 68},
        "breakout_squeeze_pro": {"bbands_period": 20, "keltner_period": 20, "keltner_atr_multiplier": 1.5},
        "rsi_divergence": {"rsi_period": 14, "lookback_period": 35, "confirm_with_rsi_exit": True},
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
                    reason TEXT, order_id TEXT, algo_id TEXT -- algo_id for OCO
                )
            ''')
            await conn.commit()
        logger.info("Ultimate database initialized successfully.")
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
# [Note: Analysis functions remain the same, they are the core intelligence]
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
# --- 🦾 The Postman Protocol (Execution Body) 🦾 ---
# =======================================================================================
async def handle_filled_buy_order(order_data):
    """
    The Postman's core task: Triggered by the Private WS on a successful buy.
    It immediately dispatches the OCO protection order.
    """
    symbol = order_data['instId'].replace('-', '/'); order_id = order_data['ordId']
    filled_qty = float(order_data.get('fillSz', 0)); avg_price = float(order_data.get('avgPx', 0))
    if filled_qty == 0 or avg_price == 0: return

    logger.info(f"📬 [Postman] Received fill for order {order_id} ({symbol}). Preparing OCO protection...")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM trades WHERE order_id = ? AND status = 'pending'", (order_id,))
            trade = await cursor.fetchone()
            if not trade:
                logger.warning(f"[Postman] No matching 'pending' trade for order {order_id}. Ignoring.")
                return

            trade = dict(trade)
            # Recalculate TP/SL based on actual fill price for max accuracy
            risk = avg_price - trade['stop_loss']
            final_tp = avg_price + (risk * bot_data.settings['risk_reward_ratio'])

            # Prepare the OCO order parameters for the exchange API
            oco_params = {
                'instId': bot_data.exchange.market_id(symbol), 'tdMode': 'cash', 'side': 'sell', 'ordType': 'oco',
                'sz': bot_data.exchange.amount_to_precision(symbol, filled_qty),
                'tpTriggerPx': bot_data.exchange.price_to_precision(symbol, final_tp), 'tpOrdPx': '-1', # -1 for market order
                'slTriggerPx': bot_data.exchange.price_to_precision(symbol, trade['stop_loss']), 'slOrdPx': '-1'
            }

            # Attempt to place the OCO order with retries
            for attempt in range(3):
                oco_receipt = await bot_data.exchange.private_post_trade_order_algo(oco_params)
                if oco_receipt and oco_receipt.get('data') and oco_receipt['data'][0].get('sCode') == '0':
                    algo_id = oco_receipt['data'][0]['algoId']
                    logger.info(f"✅ [Postman] OCO protection delivered for trade ID {trade['id']}. Algo ID: {algo_id}")

                    await conn.execute(
                        "UPDATE trades SET status = 'active', entry_price = ?, quantity = ?, take_profit = ?, algo_id = ? WHERE id = ?",
                        (avg_price, filled_qty, final_tp, algo_id, trade['id'])
                    )
                    await conn.commit()

                    tp_percent = (final_tp / avg_price - 1) * 100
                    sl_percent = (1 - trade['stop_loss'] / avg_price) * 100
                    success_msg = (f"**✅🛡️ صفقة مصفحة | {symbol} (ID: {trade['id']})**\n"
                                   f"🔍 **الاستراتيجية:** {trade['reason']}\n\n"
                                   f"📈 **الشراء:** `{avg_price:,.4f}`\n"
                                   f"🎯 **الهدف:** `{final_tp:,.4f}` (+{tp_percent:.2f}%)\n"
                                   f"🛑 **الوقف:** `{trade['stop_loss']:,.4f}` (-{sl_percent:.2f}%)\n\n"
                                   f"***تم تأمين الصفقة بنجاح على خوادم المنصة.***")
                    await safe_send_message(bot_data.application.bot, success_msg)
                    return # Mission accomplished
                else:
                    logger.warning(f"[Postman] OCO placement attempt {attempt + 1} failed. Retrying...")
                    await asyncio.sleep(2)
            
            raise Exception(f"All Postman attempts to place OCO for trade #{trade['id']} failed.")

    except Exception as e:
        logger.critical(f"🔥 [Postman] CRITICAL FAILURE while protecting {order_id}: {e}", exc_info=True)
        error_message = (f"**🔥🔥🔥 فشل حرج لساعي البريد - {symbol}**\n\n"
                         f"🚨 **خطر!** تم تنفيذ الشراء ولكن **فشل وضع الحماية تلقائيًا**.\n"
                         f"**معرف الأمر:** `{order_id}`\n\n"
                         f"**❗️ تدخل يدوي فوري ضروري لوضع وقف الخسارة!**")
        await safe_send_message(bot_data.application.bot, error_message)

class PrivateWebSocketManager:
    """The bot's 'ear', listening for order fill confirmations from OKX."""
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
            self.websocket = None; logger.warning("⚠️ [WS-Private] Disconnected. Reconnecting in 5s..."); await asyncio.sleep(5)

# =======================================================================================
# --- ⚡ Core Scanner & Trade Initiation Logic ⚡ ---
# =======================================================================================
async def get_okx_markets():
    """Fetches, filters, and sorts markets exclusively from OKX."""
    settings = bot_data.settings
    exchange = bot_data.exchange
    try:
        tickers = await exchange.fetch_tickers()
        # Apply all filters: USDT pair, volume, and the new blacklist
        blacklist = settings.get('asset_blacklist', [])
        
        valid_markets = []
        for ticker in tickers.values():
            symbol = ticker.get('symbol')
            if not symbol or not symbol.endswith('/USDT'): continue
            
            base_currency = symbol.split('/')[0]
            if base_currency in blacklist: continue
            
            if ticker.get('quoteVolume', 0) < settings['liquidity_filters']['min_quote_volume_24h_usd']: continue
            
            # Additional check for derivative-like names
            if any(k in symbol for k in ['-SWAP', 'UP', 'DOWN', '3L', '3S']): continue

            valid_markets.append(ticker)
            
        # Sort by volume and take the top N
        valid_markets.sort(key=lambda m: m.get('quoteVolume', 0), reverse=True)
        return valid_markets[:settings['top_n_symbols_by_volume']]
    except Exception as e:
        logger.error(f"Failed to fetch and filter OKX markets: {e}")
        return []

async def worker(queue, results_list, failure_counter):
    settings = bot_data.settings
    exchange = bot_data.exchange
    while not queue.empty():
        market = await queue.get(); symbol = market['symbol']
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            if len(ohlcv) < 200: continue
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            confirmed_reasons = []
            for name in settings['active_scanners']:
                params = settings.get('strategy_params', {}).get(name, {})
                if result := SCANNERS[name](df.copy(), params):
                    confirmed_reasons.append(result['reason'])

            if len(confirmed_reasons) >= settings.get("min_signal_strength", 1):
                reason_str = ' + '.join(map(str, set(confirmed_reasons)))
                entry_price = df.iloc[-2]['close']
                df.ta.atr(length=14, append=True)
                atr_col = find_col(df.columns, "ATRr_14")
                current_atr = df.iloc[-2].get(atr_col, 0)

                if current_atr > 0 and entry_price > 0:
                    risk = current_atr * settings['atr_sl_multiplier']
                    stop_loss = entry_price - risk
                    take_profit = entry_price + (risk * settings['risk_reward_ratio'])
                    results_list.append({"symbol": symbol, "entry_price": entry_price, 
                                         "take_profit": take_profit, "stop_loss": stop_loss, 
                                         "reason": reason_str, "strength": len(confirmed_reasons)})
        except Exception as e:
            logger.debug(f"Worker error for {symbol}: {e}")
            failure_counter[0] += 1
        finally: queue.task_done()

async def initiate_real_trade(signal):
    settings = bot_data.settings
    exchange = bot_data.exchange
    try:
        trade_size_usdt = settings['real_trade_size_usdt']
        amount = trade_size_usdt / signal['entry_price']

        logger.info(f"--- INITIATING REAL TRADE: {signal['symbol']} ---")
        buy_order = await exchange.create_market_buy_order(signal['symbol'], amount)
        logger.info(f"Order sent to OKX. ID: {buy_order['id']}. Awaiting Postman confirmation...")

        await log_pending_trade_to_db(signal, buy_order)
        await safe_send_message(bot_data.application.bot, f"🚀 تم إرسال أمر شراء لـ `{signal['symbol']}`. في انتظار تأكيد التنفيذ...")
    except Exception as e:
        logger.error(f"REAL TRADE FAILED for {signal['symbol']}: {e}", exc_info=True)
        await safe_send_message(bot_data.application.bot, f"🔥 فشل فتح صفقة لـ `{signal['symbol']}`.")

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        logger.info("--- Starting new OKX-focused market scan... ---")
        async with aiosqlite.connect(DB_FILE) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active' OR status = 'pending'")
            active_trades_count = (await cursor.fetchone())[0]

        settings = bot_data.settings
        if active_trades_count >= settings['max_concurrent_trades']:
            logger.info(f"Scan skipped: Max concurrent trades ({active_trades_count}) reached.")
            return

        top_markets = await get_okx_markets()
        if not top_markets:
            logger.info("Scan complete: No markets passed asset and liquidity filters."); return

        queue, signals_found, failure_counter = asyncio.Queue(), [], [0]
        for market in top_markets: await queue.put(market)

        worker_tasks = [asyncio.create_task(worker(queue, signals_found, failure_counter)) for _ in range(10)]
        await queue.join(); [task.cancel() for task in worker_tasks]

        logger.info(f"--- Scan complete. Found {len(signals_found)} potential signals. ---")
        
        for signal in sorted(signals_found, key=lambda s: s.get('strength', 0), reverse=True):
            if active_trades_count >= settings['max_concurrent_trades']: break
            if time.time() - bot_data.last_signal_time.get(signal['symbol'], 0) > (SCAN_INTERVAL_SECONDS * 2):
                bot_data.last_signal_time[signal['symbol']] = time.time()
                await initiate_real_trade(signal)
                active_trades_count += 1
                await asyncio.sleep(5) # Small delay between initiating trades

# =======================================================================================
# --- 🤖 Telegram UI & Bot Startup 🤖 ---
# =======================================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("أهلاً بك في OKX Ultimate Trader v21.0 (The Postman)")

# [Note: Other UI functions like dashboard, settings etc. would be here]

async def post_init(application: Application):
    bot_data.application = application
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, TELEGRAM_BOT_TOKEN]):
        logger.critical("FATAL: Missing critical API or Bot keys."); return

    try:
        exchange_config = {'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE, 'enableRateLimit': True}
        bot_data.exchange = ccxt.okx(exchange_config)
        await bot_data.exchange.fetch_balance()
        logger.info("✅ Successfully connected to OKX for scanning and trading.")
    except Exception as e:
        logger.critical(f"🔥 FATAL: Could not connect to OKX: {e}"); return

    bot_data.private_ws = PrivateWebSocketManager()
    asyncio.create_task(bot_data.private_ws.run())

    application.job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name="perform_scan")
    logger.info(f"Scanner scheduled for every {SCAN_INTERVAL_SECONDS} seconds.")
    await safe_send_message(application.bot, "*🚀 OKX Ultimate Trader (The Postman) بدأ العمل...*")
    logger.info("--- Bot is now fully operational ---")

async def post_shutdown(application: Application):
    if bot_data.exchange: await bot_data.exchange.close()
    logger.info("Bot has shut down.")

def main():
    logger.info("--- Starting OKX Ultimate Trader v21.0 ---")
    load_settings(); asyncio.run(init_database())
    app_builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    app_builder.post_init(post_init).post_shutdown(post_shutdown)
    application = app_builder.build()
    
    application.add_handler(CommandHandler("start", start_command))
    # Add other handlers as needed
    
    application.run_polling()

if __name__ == '__main__':
    main()

