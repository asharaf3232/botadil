# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 OKX Bot v8.4 (The Sentinel) 🚀 ---
# =======================================================================================
# This version introduces a real-time, dual-WebSocket architecture for trade
# management and fully implements Trailing Stop Loss functionality.
# =======================================================================================

# --- Libraries ---
import asyncio
import os
import logging
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
import aiosqlite
import httpx
import websockets
import hmac
import base64
import time

# --- Heavy Libraries (Lazy Loaded) ---
ccxt = None
pd = None
ta = None

from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest
from dotenv import load_dotenv

# =======================================================================================
# --- ⚙️ Core Setup ⚙️ ---
# =======================================================================================
load_dotenv() 

OKX_API_KEY = os.getenv('OKX_API_KEY')
OKX_API_SECRET = os.getenv('OKX_API_SECRET')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'okx_sentinel_v8_4.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'okx_sentinel_settings_v8_4.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Sentinel_v8.4")

class BotState:
    def __init__(self):
        self.exchange = None
        self.settings = {}
        self.market_mood = {"mood": "UNKNOWN", "reason": "تحليل لم يتم بعد"}
        self.scan_stats = {"last_start": None, "last_duration": "N/A"}
        self.application = None
        self.trade_guardian = None
        self.public_ws = None
        self.private_ws = None

bot_state = BotState()
scan_lock = asyncio.Lock()
trade_management_lock = asyncio.Lock()

# --- (UI Constants and other static data remain unchanged) ---
DEFAULT_SETTINGS = {
    "active_preset": "PRO", "real_trade_size_usdt": 15.0, "top_n_symbols_by_volume": 250,
    "atr_sl_multiplier": 2.5, "risk_reward_ratio": 2.0,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "whale_radar", "sniper_pro"],
    "market_mood_filter_enabled": True, "fear_and_greed_threshold": 30,
    "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "max_spread_percent": 0.5, "min_rvol": 1.5},
    "volatility_filters": {"min_atr_percent": 0.8},
    "trend_filters": {"ema_period": 200, "htf_period": 50},
    "min_tp_sl_filter": {"min_tp_percent": 1.0, "min_sl_percent": 0.5},
    "scan_interval_seconds": 900,
    "trailing_sl_enabled": True, "trailing_sl_activation_percent": 1.5, "trailing_sl_callback_percent": 1.0
}
# ... (Other constants like PRESETS, PARAM_DISPLAY_NAMES, etc.)

# --- (Helper functions like ensure_libraries_loaded, init_database etc. remain unchanged) ---
async def ensure_libraries_loaded():
    global pd, ta, ccxt
    if pd is None: logger.info("Loading pandas library..."); import pandas as pd_lib; pd = pd_lib
    if ta is None: logger.info("Loading pandas-ta library..."); import pandas_ta as ta_lib; ta = ta_lib
    if ccxt is None: logger.info("Loading ccxt library..."); import ccxt.async_support as ccxt_lib; ccxt = ccxt_lib

async def init_database():
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
                    entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, entry_value_usdt REAL,
                    status TEXT DEFAULT 'active', exit_price REAL, closed_at TEXT, pnl_usdt REAL,
                    reason TEXT, order_id TEXT, algo_id TEXT,
                    highest_price REAL DEFAULT 0, trailing_sl_active BOOLEAN DEFAULT 0
                )''')
            await conn.commit()
    except Exception as e: logger.error(f"Failed to initialize database: {e}")

# ... (rest of helper functions)

# =======================================================================================
# ---  Sentinel Protocol: Real-time Trade Management ---
# =======================================================================================
class TradeGuardian:
    def __init__(self, exchange, settings):
        self.exchange = exchange
        self.settings = settings

    async def handle_ticker_update(self, ticker_data):
        async with trade_management_lock:
            try:
                symbol = ticker_data['instId'].replace('-', '/')
                current_price = float(ticker_data['last'])

                async with aiosqlite.connect(DB_FILE) as conn:
                    conn.row_factory = aiosqlite.Row
                    cursor = await conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = 'active'", (symbol,))
                    trade = await cursor.fetchone()

                    if not trade: return
                    trade = dict(trade)
                    
                    # --- Trailing Stop Loss Logic ---
                    new_highest_price = max(trade.get('highest_price', 0), current_price)
                    if new_highest_price > trade.get('highest_price', 0):
                        await conn.execute("UPDATE trades SET highest_price = ? WHERE id = ?", (new_highest_price, trade['id']))
                    
                    if self.settings['trailing_sl_enabled'] and not trade['trailing_sl_active']:
                        activation_price = trade['entry_price'] * (1 + self.settings['trailing_sl_activation_percent'] / 100)
                        if current_price >= activation_price:
                            trade['trailing_sl_active'] = True
                            await conn.execute("UPDATE trades SET trailing_sl_active = 1 WHERE id = ?", (trade['id'],))
                            logger.info(f"Sentinel: Trailing SL activated for trade #{trade['id']} ({symbol}).")
                    
                    if trade['trailing_sl_active']:
                        callback_percent = self.settings['trailing_sl_callback_percent'] / 100
                        new_sl = new_highest_price * (1 - callback_percent)
                        if new_sl > trade['stop_loss']:
                            trade['stop_loss'] = new_sl
                            await conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_sl, trade['id']))

                    await conn.commit() # Commit price and SL updates

                # --- Decision Logic ---
                if current_price >= trade['take_profit']:
                    await self._close_trade(trade, f"ناجحة (TP)", current_price)
                elif current_price <= trade['stop_loss']:
                    reason = f"فاشلة (TSL)" if trade['trailing_sl_active'] else f"فاشلة (SL)"
                    await self._close_trade(trade, reason, current_price)

            except Exception as e:
                logger.error(f"Sentinel: Error in ticker handler: {e}", exc_info=True)

    async def _close_trade(self, trade, reason, current_price):
        symbol = trade['symbol']
        logger.info(f"Sentinel: Triggered '{reason}' for trade #{trade['id']} ({symbol}). Attempting to close.")
        try:
            sell_order = await self.exchange.create_market_sell_order(symbol, trade['quantity'])
            final_exit_price = float(sell_order.get('average', current_price))
            pnl = (final_exit_price - trade['entry_price']) * trade['quantity']

            async with aiosqlite.connect(DB_FILE) as conn:
                await conn.execute(
                    "UPDATE trades SET status = ?, exit_price = ?, closed_at = ?, pnl_usdt = ? WHERE id = ?",
                    (reason, final_exit_price, datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), pnl, trade['id'])
                )
                await conn.commit()
            
            await bot_state.public_ws.unsubscribe([symbol])

            pnl_percent = (final_exit_price / trade['entry_price'] - 1) * 100
            result_emoji = "✅" if pnl > 0 else "🛑"
            msg = (f"**{result_emoji} تم إغلاق الصفقة | {symbol} (ID: {trade['id']})**\n\n"
                   f"**السبب:** {reason}\n**سعر الخروج:** `{final_exit_price:,.4f}`\n"
                   f"**الربح/الخسارة:** `${pnl:,.2f}` ({pnl_percent:+.2f}%)")
            await bot_state.application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.critical(f"Sentinel: CRITICAL FAILURE closing trade #{trade['id']}: {e}", exc_info=True)
            # Handle failure...

    async def sync_subscriptions(self):
        async with aiosqlite.connect(DB_FILE) as conn:
            cursor = await conn.execute("SELECT DISTINCT symbol FROM trades WHERE status = 'active'")
            active_symbols = [row[0] for row in await cursor.fetchall()]
        
        if active_symbols:
            logger.info(f"Sentinel: Syncing subscriptions for symbols: {active_symbols}")
            await bot_state.public_ws.subscribe(active_symbols)

# =======================================================================================
# --- WebSocket Managers ---
# =======================================================================================
class PrivateWebSocketManager:
    # This class remains the same, handling private order updates
    def __init__(self): self.ws_url = "wss://ws.okx.com:8443/ws/v5/private"
    def _get_auth_args(self):
        timestamp = str(time.time())
        message = timestamp + 'GET' + '/users/self/verify'
        mac = hmac.new(bytes(OKX_API_SECRET, encoding='utf8'), bytes(message, encoding='utf8'), digestmod='sha256')
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
                async with websockets.connect(self.ws_url) as ws:
                    self.websocket = ws
                    logger.info("✅ [WS-Private] Connected. Authenticating...")
                    await ws.send(json.dumps({"op": "login", "args": self._get_auth_args()}))
                    login_resp = json.loads(await ws.recv())
                    if login_resp.get('code') == '0':
                        logger.info("🔐 [WS-Private] Authenticated. Subscribing to orders...")
                        await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "orders", "instType": "SPOT"}]}))
                        async for msg in ws: await self._message_handler(msg)
                    else: logger.error(f"🔥 [WS-Private] Auth failed: {login_resp}")
            except Exception as e: logger.error(f"🔥 [WS-Private] Error: {e}")
            logger.warning("⚠️ [WS-Private] Disconnected. Reconnecting in 5s..."); await asyncio.sleep(5)

class PublicWebSocketManager:
    def __init__(self, handler_coro):
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.handler = handler_coro
        self.subscriptions = set()
    async def _send_subscription(self, op, symbols):
        if not symbols: return
        args = [{"channel": "tickers", "instId": s.replace('/', '-')} for s in symbols]
        await self.websocket.send(json.dumps({"op": op, "args": args}))
    async def subscribe(self, symbols):
        new_symbols = [s for s in symbols if s not in self.subscriptions]
        if new_symbols:
            await self._send_subscription('subscribe', new_symbols)
            self.subscriptions.update(new_symbols)
            logger.info(f"✅ [WS-Public] Subscribed to tickers: {new_symbols}")
    async def unsubscribe(self, symbols):
        to_unsubscribe = [s for s in symbols if s in self.subscriptions]
        if to_unsubscribe:
            await self._send_subscription('unsubscribe', to_unsubscribe)
            for s in to_unsubscribe: self.subscriptions.discard(s)
            logger.info(f"🗑️ [WS-Public] Unsubscribed from tickers: {to_unsubscribe}")
    async def run(self):
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.websocket = ws
                    logger.info("✅ [WS-Public] Connected.")
                    await self.subscribe(list(self.subscriptions)) # Resubscribe on reconnect
                    async for msg in ws:
                        if msg == 'ping': await ws.send('pong'); continue
                        data = json.loads(msg)
                        if data.get('arg', {}).get('channel') == 'tickers' and 'data' in data:
                            for ticker in data['data']:
                                asyncio.create_task(self.handler(ticker))
            except Exception as e: logger.error(f"🔥 [WS-Public] Error: {e}")
            logger.warning("⚠️ [WS-Public] Disconnected. Reconnecting in 5s..."); await asyncio.sleep(5)

# =======================================================================================
# --- Core Bot Logic (Modified for new architecture) ---
# =======================================================================================
async def handle_filled_buy_order(order_data):
    symbol = order_data['instId'].replace('-', '/')
    order_id = order_data['ordId']
    avg_price = float(order_data.get('avgPx', 0))
    filled_qty = float(order_data.get('fillSz', 0))
    if filled_qty == 0 or avg_price == 0: return

    logger.info(f"📬 [Postman] Fill event for {symbol}. Activating Sentinel monitoring...")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            cursor = await conn.execute("SELECT * FROM trades WHERE order_id = ? AND status = 'pending_protection'", (order_id,))
            trade = await cursor.fetchone()
            if not trade: return
            
            # Recalculate TP/SL based on actual entry price
            # Note: The original trade data is a tuple, needs indexing.
            original_entry, original_sl = trade[3], trade[5]
            original_risk = original_entry - original_sl
            final_tp = avg_price + (original_risk * bot_state.settings['risk_reward_ratio'])
            final_sl = avg_price - original_risk
            
            await conn.execute(
                "UPDATE trades SET status = 'active', entry_price = ?, quantity = ?, take_profit = ?, stop_loss = ?, highest_price = ? WHERE order_id = ?",
                (avg_price, filled_qty, final_tp, final_sl, avg_price, order_id)
            )
            await conn.commit()
        
        await bot_state.trade_guardian.sync_subscriptions()

        # ... (Send success message to Telegram)
    except Exception as e:
        logger.critical(f"🔥 [Postman] CRITICAL FAILURE activating trade {order_id}: {e}", exc_info=True)
        # ... (Send failure message to Telegram)

# --- (perform_scan, initiate_trade and other analysis functions are unchanged) ---
# ...

# =======================================================================================
# --- 🚀 Main Bot Startup ---
# =======================================================================================
async def main():
    logger.info("--- Bot v8.4 (The Sentinel) starting ---")
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.critical("FATAL: Missing environment variables."); return
        
    # --- Load settings, DB, and CCXT ---
    # ... (load_settings(), init_database(), etc.)
    await ensure_libraries_loaded()
    bot_state.exchange = ccxt.okx({'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE})

    # --- Initialize Core Components ---
    bot_state.trade_guardian = TradeGuardian(bot_state.exchange, bot_state.settings)
    bot_state.public_ws = PublicWebSocketManager(bot_state.trade_guardian.handle_ticker_update)
    bot_state.private_ws = PrivateWebSocketManager()
    
    # --- Setup Telegram ---
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    bot_state.application = app
    # ... (Add all your Telegram handlers here)

    # --- Start Background Tasks ---
    public_ws_task = asyncio.create_task(bot_state.public_ws.run())
    private_ws_task = asyncio.create_task(bot_state.private_ws.run())
    
    # Sync subscriptions for any trades that were active before a restart
    await asyncio.sleep(5) # Give WS time to connect
    await bot_state.trade_guardian.sync_subscriptions()
    
    # --- Setup Job Queue ---
    scan_interval = bot_state.settings.get("scan_interval_seconds", 900)
    app.job_queue.run_repeating(perform_scan, interval=scan_interval, first=10)
    
    try:
        await bot_state.exchange.fetch_balance()
        logger.info("✅ OKX connection test SUCCEEDED.")
        await app.bot.send_message(TELEGRAM_CHAT_ID, "*🚀 بوت The Sentinel v8.4 بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)
        
        async with app:
            await app.start()
            await app.updater.start_polling()
            logger.info("Bot is now running...")
            await asyncio.gather(public_ws_task, private_ws_task)

    except Exception as e:
        logger.critical(f"Unhandled error in main loop: {e}", exc_info=True)
    finally:
        # ... (graceful shutdown logic)
        if bot_state.exchange: await bot_state.exchange.close()
        logger.info("Bot has been shut down.")

if __name__ == '__main__':
    # You need to fill in the missing functions (settings, helpers, telegram UI, scan logic)
    # from your previous version for this file to be complete.
    asyncio.run(main())

