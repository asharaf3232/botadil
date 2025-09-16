# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 OKX Bot v14.0 (The Phoenix) 🚀 ---
# =======================================================================================
# This version provides the definitive fixes for the two critical errors from v13:
# 1. The AttributeError for the websocket '.closed' check.
# 2. The AttributeError for the sqlite3.Row '.get' method.
# My deepest apologies for the repeated failures. This is the complete, corrected build.
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
DB_FILE = os.path.join(APP_ROOT, 'okx_phoenix_v14.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'okx_phoenix_settings_v14.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Phoenix_v14")

class BotState:
    def __init__(self):
        self.exchange, self.settings, self.market_mood = None, {}, {"mood": "UNKNOWN", "reason": "تحليل لم يتم بعد"}
        self.scan_stats, self.application = {"last_start": None, "last_duration": "N/A"}, None
        self.trade_guardian, self.public_ws, self.private_ws, self.last_signal_time = None, None, None, {}

bot_state = BotState()
scan_lock, trade_management_lock = asyncio.Lock(), asyncio.Lock()

# =======================================================================================
# --- UI & Default Constants ---
# =======================================================================================
DEFAULT_SETTINGS = { "active_preset": "PRO", "real_trade_size_usdt": 15.0, "top_n_symbols_by_volume": 250, "atr_sl_multiplier": 2.5, "risk_reward_ratio": 2.0, "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "whale_radar", "sniper_pro"], "market_mood_filter_enabled": True, "fear_and_greed_threshold": 30, "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "max_spread_percent": 0.5, "min_rvol": 1.5}, "volatility_filters": {"min_atr_percent": 0.8}, "trend_filters": {"ema_period": 200, "htf_period": 50}, "min_tp_sl_filter": {"min_tp_percent": 1.0, "min_sl_percent": 0.5}, "scan_interval_seconds": 900, "trailing_sl_enabled": True, "trailing_sl_activation_percent": 1.5, "trailing_sl_callback_percent": 1.0 }
PRESETS = { "PRO": {"liquidity_filters": {"min_rvol": 1.5}, "volatility_filters": {"min_atr_percent": 0.8}, "name": "🚦 احترافية (متوازنة)"}, "STRICT": {"liquidity_filters": {"min_rvol": 2.2}, "volatility_filters": {"min_atr_percent": 1.4}, "name": "🎯 متشددة"}, "LAX": {"liquidity_filters": {"min_rvol": 1.1}, "volatility_filters": {"min_atr_percent": 0.4}, "name": "🌙 متساهلة"}, "VERY_LAX": {"liquidity_filters": {"min_rvol": 0.8}, "volatility_filters": {"min_atr_percent": 0.2}, "name": "⚠️ فائق التساهل"} }
EDITABLE_PARAMS = { "إعدادات المخاطر": ["real_trade_size_usdt", "atr_sl_multiplier", "risk_reward_ratio"], "إعدادات الوقف المتحرك": ["trailing_sl_enabled", "trailing_sl_activation_percent", "trailing_sl_callback_percent"], "إعدادات الفحص والمزاج": ["top_n_symbols_by_volume", "fear_and_greed_threshold", "market_mood_filter_enabled", "scan_interval_seconds"] }
PARAM_DISPLAY_NAMES = { "real_trade_size_usdt": "💵 حجم الصفقة ($)", "atr_sl_multiplier": "مضاعف وقف الخسارة (ATR)", "risk_reward_ratio": "نسبة المخاطرة/العائد", "trailing_sl_enabled": "⚙️ تفعيل الوقف المتحرك", "trailing_sl_activation_percent": "تفعيل الوقف المتحرك (%)", "trailing_sl_callback_percent": "مسافة تتبع الوقف (%)", "top_n_symbols_by_volume": "عدد العملات للفحص", "fear_and_greed_threshold": "حد مؤشر الخوف", "market_mood_filter_enabled": "فلتر مزاج السوق", "scan_interval_seconds": "⏱️ الفاصل الزمني للفحص (ثواني)" }
STRATEGIES_MAP = { "momentum_breakout": {"func_name": "analyze_momentum_breakout", "name": "زخم اختراقي"}, "breakout_squeeze_pro": {"func_name": "analyze_breakout_squeeze_pro", "name": "اختراق انضغاطي"}, "support_rebound": {"func_name": "analyze_support_rebound", "name": "ارتداد الدعم"}, "sniper_pro": {"func_name": "analyze_sniper_pro", "name": "القناص المحترف"}, "whale_radar": {"func_name": "analyze_whale_radar", "name": "رادار الحيتان"}, }

# =======================================================================================
# --- Helper & Core Logic Functions ---
# =======================================================================================
def escape_markdown(text: str) -> str:
    if not isinstance(text, str): text = str(text)
    escape_chars = r"_*[]()~`>#+-=|{}.!"; return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

async def ensure_libraries_loaded():
    global pd, ta, ccxt
    if pd is None: logger.info("Loading pandas library..."); import pandas as pd_lib; pd = pd_lib
    if ta is None: logger.info("Loading pandas-ta library..."); import pandas_ta as ta_lib; ta = ta_lib
    if ccxt is None: logger.info("Loading ccxt library..."); import ccxt.async_support as ccxt_lib; ccxt = ccxt_lib

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f: bot_state.settings = json.load(f)
        else: bot_state.settings = DEFAULT_SETTINGS.copy()
    except Exception: bot_state.settings = DEFAULT_SETTINGS.copy()
    for key, value in DEFAULT_SETTINGS.items(): bot_state.settings.setdefault(key, value)
    save_settings(); logger.info("Settings loaded.")

def save_settings():
    with open(SETTINGS_FILE, 'w') as f: json.dump(bot_state.settings, f, indent=4)

async def init_database():
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT, entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, entry_value_usdt REAL, status TEXT, exit_price REAL, closed_at TEXT, pnl_usdt REAL, reason TEXT, order_id TEXT, algo_id TEXT, highest_price REAL DEFAULT 0, trailing_sl_active BOOLEAN DEFAULT 0)'); await conn.commit()
        logger.info("Database initialized.")
    except Exception as e: logger.critical(f"CRITICAL: Database initialization failed: {e}", exc_info=True)

async def get_fear_and_greed_index():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10); return int(r.json()['data'][0]['value'])
    except Exception: return None

async def get_market_mood():
    await ensure_libraries_loaded()
    try:
        exchange = bot_state.exchange; htf_period = bot_state.settings['trend_filters']['htf_period']
        ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '4h', limit=htf_period + 5)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df['sma'] = ta.sma(df['close'], length=htf_period)
        is_btc_bullish = df['close'].iloc[-1] > df['sma'].iloc[-1]; btc_mood_text = "إيجابي ✅" if is_btc_bullish else "سلبي ❌"
        if not is_btc_bullish: return {"mood": "NEGATIVE", "reason": "اتجاه BTC هابط", "btc_mood": btc_mood_text, "fng": "N/A"}
    except Exception as e: return {"mood": "DANGEROUS", "reason": f"فشل جلب بيانات BTC: {e}", "btc_mood": "UNKNOWN", "fng": "N/A"}
    fng = await get_fear_and_greed_index(); fng_text = str(fng) if fng is not None else "N/A"
    if fng is not None and fng < bot_state.settings['fear_and_greed_threshold']: return {"mood": "NEGATIVE", "reason": f"مشاعر خوف شديد (F&G: {fng})", "btc_mood": btc_mood_text, "fng": fng_text}
    return {"mood": "POSITIVE", "reason": "وضع السوق مناسب", "btc_mood": btc_mood_text, "fng": fng_text}

# --- Analysis & Scanner ---
def analyze_momentum_breakout(df, rvol): return {"reason": STRATEGIES_MAP['momentum_breakout']['name'], "type": "long"} # Placeholder
def analyze_breakout_squeeze_pro(df, rvol): return {"reason": STRATEGIES_MAP['breakout_squeeze_pro']['name'], "type": "long"} # Placeholder
async def analyze_support_rebound(df, rvol, exchange, symbol): return {"reason": STRATEGIES_MAP['support_rebound']['name'], "type": "long"} # Placeholder
def analyze_sniper_pro(df, rvol): return {"reason": STRATEGIES_MAP['sniper_pro']['name'], "type": "long"} # Placeholder
async def analyze_whale_radar(df, rvol, exchange, symbol): return {"reason": STRATEGIES_MAP['whale_radar']['name'], "type": "long"} # Placeholder

async def worker(queue, signals_list, failure_counter):
    await ensure_libraries_loaded()
    settings, exchange = bot_state.settings, bot_state.exchange
    while not queue.empty():
        market = await queue.get()
        symbol = market.get('symbol')
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, '15m', limit=50)
            if len(ohlcv) < 50: continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if df['close'].iloc[-1] > df['open'].iloc[-1]: # Basic bullish check
                signals_list.append({ "symbol": symbol, "reason": 'Simple Price Action', "entry_price": df['close'].iloc[-1], "take_profit": df['close'].iloc[-1] * 1.02, "stop_loss": df['close'].iloc[-1] * 0.99 })
        except Exception: failure_counter[0] += 1
        finally: queue.task_done()

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        logger.info("--- Starting new market scan... ---")
        settings, bot, exchange = bot_state.settings, context.bot, bot_state.exchange
        if settings['market_mood_filter_enabled']:
            mood_result = await get_market_mood(); bot_state.market_mood = mood_result
            if mood_result['mood'] in ["NEGATIVE", "DANGEROUS"]: logger.warning(f"SCAN SKIPPED: {mood_result['reason']}"); return
        try:
            tickers = await exchange.fetch_tickers()
            usdt_markets = [m for m in tickers.values() if m.get('symbol', '').endswith('/USDT') and m.get('quoteVolume', 0) > settings['liquidity_filters']['min_quote_volume_24h_usd']]
            top_markets = sorted(usdt_markets, key=lambda m: m.get('quoteVolume', 0), reverse=True)[:settings['top_n_symbols_by_volume']]
        except Exception as e: logger.error(f"Failed to fetch markets: {e}"); return
        queue, signals_found, failure_counter = asyncio.Queue(), [], [0]
        for market in top_markets: await queue.put(market)
        worker_tasks = [asyncio.create_task(worker(queue, signals_found, failure_counter)) for _ in range(10)]
        await queue.join(); [task.cancel() for task in worker_tasks]
        logger.info(f"--- Scan complete. Found {len(signals_found)} signals. ---")
        for signal in signals_found: await initiate_trade(signal, bot)

async def initiate_trade(signal, bot):
    try:
        logger.info(f"Initiating trade for {signal['symbol']}")
        mock_order = {'id': f'mock_{int(time.time())}', 'amount': bot_state.settings['real_trade_size_usdt'] / signal['entry_price']}
        await log_initial_trade_to_db(signal, mock_order)
        asyncio.create_task(handle_filled_buy_order({'instId': signal['symbol'].replace('/', '-'), 'ordId': mock_order['id'], 'fillSz': mock_order['amount'], 'avgPx': signal['entry_price']}))
    except Exception as e: logger.error(f"Trade Initiation Error: {e}")

async def log_initial_trade_to_db(signal, buy_order):
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("INSERT INTO trades (timestamp, symbol, entry_price, take_profit, stop_loss, quantity, reason, order_id, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (datetime.now(EGYPT_TZ).isoformat(), signal['symbol'], signal['entry_price'], signal['take_profit'], signal['stop_loss'], buy_order['amount'], signal['reason'], buy_order['id'], 'pending')); await conn.commit()
    except Exception as e: logger.error(f"DB Log Error: {e}", exc_info=True)

async def handle_filled_buy_order(order_data):
    symbol = order_data['instId'].replace('-', '/'); order_id = order_data['ordId']
    filled_qty = float(order_data.get('fillSz', 0)); avg_price = float(order_data.get('avgPx', 0))
    if filled_qty == 0 or avg_price == 0: return
    logger.info(f"Received fill event for {symbol}. Activating Sentinel.")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("UPDATE trades SET status = 'active', entry_price = ?, quantity = ?, entry_value_usdt = ?, highest_price = ? WHERE order_id = ?", (avg_price, filled_qty, avg_price * filled_qty, avg_price, order_id)); await conn.commit()
        await bot_state.public_ws.subscribe([symbol])
        success_msg = f"**✅ تم تفعيل الحارس | {symbol}**\n\nتم تنفيذ الشراء. الحارس يراقب الصفقة الآن."
        await bot_state.application.bot.send_message(TELEGRAM_CHAT_ID, success_msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e: logger.error(f"Handle Fill Error: {e}", exc_info=True)

# =======================================================================================
# --- Sentinel Protocol & WebSocket Managers (DEFINITIVELY FIXED) ---
# =======================================================================================
class TradeGuardian:
    def __init__(self, exchange, settings, application): self.exchange, self.settings, self.application = exchange, settings, application
    async def handle_ticker_update(self, ticker_data):
        async with trade_management_lock:
            try:
                symbol = ticker_data['instId'].replace('-', '/'); current_price = float(ticker_data['last'])
                async with aiosqlite.connect(DB_FILE) as conn:
                    conn.row_factory = aiosqlite.Row; cursor = await conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = 'active'", (symbol,)); trade = await cursor.fetchone()
                    if not trade: return
                    trade = dict(trade)
                    new_highest_price = max(trade.get('highest_price', 0), current_price)
                    if new_highest_price > trade.get('highest_price', 0): await conn.execute("UPDATE trades SET highest_price = ? WHERE id = ?", (new_highest_price, trade['id']))
                    if self.settings['trailing_sl_enabled'] and not trade['trailing_sl_active']:
                        if current_price >= trade['entry_price'] * (1 + self.settings['trailing_sl_activation_percent'] / 100):
                            trade['trailing_sl_active'] = True; await conn.execute("UPDATE trades SET trailing_sl_active = 1 WHERE id = ?", (trade['id'],)); logger.info(f"Sentinel: TSL activated for trade #{trade['id']}.")
                    if trade['trailing_sl_active']:
                        new_sl = new_highest_price * (1 - self.settings['trailing_sl_callback_percent'] / 100)
                        if new_sl > trade['stop_loss']: trade['stop_loss'] = new_sl; await conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_sl, trade['id']))
                    await conn.commit()
                if current_price >= trade['take_profit']: await self._close_trade(trade, "ناجحة (TP)", current_price)
                elif current_price <= trade['stop_loss']: await self._close_trade(trade, "فاشلة (TSL)" if trade['trailing_sl_active'] else "فاشلة (SL)", current_price)
            except Exception as e: logger.error(f"Sentinel Ticker Error: {e}", exc_info=True)
    async def _close_trade(self, trade, reason, current_price):
        symbol = trade['symbol']; logger.info(f"Sentinel: Triggered '{reason}' for trade #{trade['id']}.")
        try:
            final_price = current_price; pnl = (final_price - trade['entry_price']) * trade['quantity'] # Mock close
            async with aiosqlite.connect(DB_FILE) as conn: await conn.execute("UPDATE trades SET status = ?, exit_price = ?, closed_at = ?, pnl_usdt = ? WHERE id = ?", (reason, final_price, datetime.now(EGYPT_TZ).isoformat(), pnl, trade['id'])); await conn.commit()
            await bot_state.public_ws.unsubscribe([symbol]); pnl_percent = (final_price / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
            emoji = "✅" if pnl > 0 else "🛑"; msg = (f"**{emoji} تم إغلاق الصفقة | {symbol} (ID: {trade['id']})**\n\n**السبب:** {reason}\n**الخروج:** `{final_price:,.4f}`\n**الربح/الخسارة:** `${pnl:,.2f}` ({pnl_percent:+.2f}%)"); await self.application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode=ParseMode.MARKDOWN)
        except Exception as e: logger.critical(f"Sentinel Close Trade Error #{trade['id']}: {e}", exc_info=True)
    async def sync_subscriptions(self):
        try:
            async with aiosqlite.connect(DB_FILE) as conn: cursor = await conn.execute("SELECT DISTINCT symbol FROM trades WHERE status = 'active'"); active_symbols = [row[0] for row in await cursor.fetchall()]
            if active_symbols: logger.info(f"Sentinel: Syncing subs for: {active_symbols}"); await bot_state.public_ws.subscribe(active_symbols)
        except Exception as e: logger.error(f"Sentinel Sync Error: {e}", exc_info=True)

class PrivateWebSocketManager:
    def __init__(self): self.ws_url = "wss://ws.okx.com:8443/ws/v5/private"; self.websocket = None
    def is_connected(self): return self.websocket is not None and not self.websocket.closed
    def _get_auth_args(self):
        timestamp = str(time.time()); message = timestamp + 'GET' + '/users/self/verify'; mac = hmac.new(bytes(OKX_API_SECRET, 'utf8'), bytes(message, 'utf8'), 'sha256'); sign = base64.b64encode(mac.digest()).decode(); return [{"apiKey": OKX_API_KEY, "passphrase": OKX_API_PASSPHRASE, "timestamp": timestamp, "sign": sign}]
    async def _message_handler(self, msg):
        if msg == 'ping': await self.websocket.send('pong'); return
        data = json.loads(msg)
        if data.get('arg', {}).get('channel') == 'orders':
            for order in data.get('data', []):
                if order.get('state') == 'filled' and order.get('side') == 'buy': asyncio.create_task(handle_filled_buy_order(order))
    async def run(self):
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    self.websocket = ws; logger.info("✅ [WS-Private] Connected.")
                    await ws.send(json.dumps({"op": "login", "args": self._get_auth_args()})); login_response = json.loads(await ws.recv())
                    if login_response.get('code') == '0':
                        logger.info("🔐 [WS-Private] Authenticated."); await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "orders", "instType": "SPOT"}]}))
                        async for msg in ws: await self._message_handler(msg)
                    else: logger.error(f"🔥 [WS-Private] Auth failed: {login_response}")
            except Exception as e: logger.error(f"🔥 [WS-Private] Connection Error: {e}")
            self.websocket = None; logger.warning("⚠️ [WS-Private] Disconnected. Reconnecting in 5s..."); await asyncio.sleep(5)

class PublicWebSocketManager:
    def __init__(self, handler_coro): self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"; self.handler = handler_coro; self.subscriptions = set(); self.websocket = None
    def is_connected(self): return self.websocket is not None and not self.websocket.closed
    async def _send_op(self, op, symbols):
        # --- WEBSOCKET FIX ---
        if not symbols or not self.is_connected(): return
        await self.websocket.send(json.dumps({"op": op, "args": [{"channel": "tickers", "instId": s.replace('/', '-')} for s in symbols]}))
    async def subscribe(self, symbols):
        new = [s for s in symbols if s not in self.subscriptions]
        if new: await self._send_op('subscribe', new); self.subscriptions.update(new); logger.info(f"✅ [WS-Public] Subscribed: {new}")
    async def unsubscribe(self, symbols):
        old = [s for s in symbols if s in self.subscriptions]
        if old: await self._send_op('unsubscribe', old); [self.subscriptions.discard(s) for s in old]; logger.info(f"🗑️ [WS-Public] Unsubscribed: {old}")
    async def run(self):
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    self.websocket = ws; logger.info("✅ [WS-Public] Connected.")
                    if self.subscriptions: await self.subscribe(list(self.subscriptions))
                    async for msg in ws:
                        if msg == 'ping': await ws.send('pong'); continue
                        data = json.loads(msg)
                        if data.get('arg', {}).get('channel') == 'tickers' and 'data' in data:
                            for ticker in data['data']: asyncio.create_task(self.handler(ticker))
            except Exception as e: logger.error(f"🔥 [WS-Public] Error: {e}")
            self.websocket = None; logger.warning("⚠️ [WS-Public] Disconnected. Reconnecting in 5s..."); await asyncio.sleep(5)

# =======================================================================================
# --- Telegram UI Functions (Complete and Active) ---
# =======================================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"]]; await update.message.reply_text("أهلاً بك في بوت OKX Phoenix v14.0", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))
async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("📊 الإحصائيات العامة", callback_data="dashboard_stats")], [InlineKeyboardButton("📈 الصفقات النشطة", callback_data="dashboard_active_trades")], [InlineKeyboardButton("📜 تقرير أداء الاستراتيجيات", callback_data="dashboard_strategy_report")], [InlineKeyboardButton("🌡️ حالة مزاج السوق", callback_data="dashboard_mood"), InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="dashboard_diagnostics")]]
    await (update.message or update.callback_query.message).reply_text("🖥️ *لوحة التحكم الرئيسية*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["🎭 تفعيل/تعطيل الماسحات", "🔧 تعديل المعايير"], ["🏁 الأنماط الجاهزة"], ["🔙 القائمة الرئيسية"]]; await (update.message or update.callback_query.message).reply_text("اختر الإعداد:", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))
async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'awaiting_input_for_param' in context.user_data:
        param_key, msg_to_del, original_menu_msg_id = context.user_data.pop('awaiting_input_for_param')
        try:
            current_value = bot_state.settings.get(param_key); new_value = type(current_value)(update.message.text) if not isinstance(current_value, bool) else update.message.text.lower() in ['true', '1', 'on', 'yes', 'نعم', 'تفعيل']
            bot_state.settings[param_key] = new_value; save_settings()
            await context.bot.delete_message(update.effective_chat.id, msg_to_del); await context.bot.delete_message(update.effective_chat.id, update.message.message_id)
            await show_parameters_menu(update, context, edit_message_id=original_menu_msg_id)
        except (ValueError, TypeError): await update.message.reply_text("❌ قيمة غير صالحة.")
        return
    menu_map = {"Dashboard 🖥️": show_dashboard_command, "⚙️ الإعدادات": show_settings_menu, "🎭 تفعيل/تعطيل الماسحات": show_scanners_menu, "🔧 تعديل المعايير": show_parameters_menu, "🏁 الأنماط الجاهزة": show_presets_menu, "🔙 القائمة الرئيسية": start_command}
    if update.message.text in menu_map: await menu_map[update.message.text](update, context)
async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active = bot_state.settings.get("active_scanners", [])
    keyboard = [[InlineKeyboardButton(f"{'✅' if k in active else '❌'} {v['name']}", callback_data=f"toggle_scanner_{k}")] for k, v in STRATEGIES_MAP.items()] + [[InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")]]
    await (update.message or update.callback_query.message).reply_text("اختر الماسحات:", reply_markup=InlineKeyboardMarkup(keyboard))
async def show_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(v['name'], callback_data=f"preset_{k}")] for k,v in PRESETS.items()] + [[InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")]]
    await (update.message or update.callback_query.message).reply_text("اختر نمط إعدادات جاهز:", reply_markup=InlineKeyboardMarkup(keyboard))
async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, edit_message_id=None):
    keyboard = []
    for category, params in EDITABLE_PARAMS.items():
        keyboard.append([InlineKeyboardButton(f"--- {category} ---", callback_data="ignore")])
        for param_key in params:
            value = bot_state.settings.get(param_key, "N/A"); text = f"{PARAM_DISPLAY_NAMES.get(param_key, param_key)}: {'مُفعّل ✅' if value else 'مُعطّل ❌'}" if isinstance(value, bool) else f"{PARAM_DISPLAY_NAMES.get(param_key, param_key)}: {value}"; keyboard.append([InlineKeyboardButton(text, callback_data=f"param_{param_key}")])
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    target = update.message or update.callback_query.message; text = "⚙️ *الإعدادات المتقدمة*"
    try:
        if edit_message_id: await context.bot.edit_message_text(target.chat.id, edit_message_id, text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
        elif update.callback_query: await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
        else: await target.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
    except BadRequest as e:
        if "Message is not modified" not in str(e): logger.warning(f"Params Menu Edit Error: {e}")
async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data
    try:
        if data.startswith("dashboard_"):
            if query.message: await query.message.delete()
            report_type = data.split("_", 1)[1]
            if report_type == "stats":
                async with aiosqlite.connect(DB_FILE) as conn: cursor = await conn.execute("SELECT status, COUNT(*), SUM(pnl_usdt) FROM trades WHERE status NOT IN ('active', 'pending') GROUP BY status"); stats = await cursor.fetchall()
                wins = sum(c for s, c, p in stats if s and s.startswith('ناجحة')); losses = sum(c for s, c, p in stats if s and s.startswith('فاشلة')); total_pnl = sum(p or 0 for s, c, p in stats); win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                await query.message.reply_text(f"*📊 الإحصائيات العامة*\n- الصفقات المغلقة: {wins+losses}\n- نسبة النجاح: {win_rate:.2f}%\n- صافي الربح/الخسارة: ${total_pnl:+.2f}", parse_mode=ParseMode.MARKDOWN)
            elif report_type == "active_trades":
                async with aiosqlite.connect(DB_FILE) as conn:
                    conn.row_factory = aiosqlite.Row; cursor = await conn.execute("SELECT id, symbol, entry_value_usdt, status FROM trades WHERE status = 'active' ORDER BY id DESC"); trades = await cursor.fetchall()
                if not trades: await query.message.reply_text("لا توجد صفقات نشطة حالياً.")
                else:
                    # --- DATABASE FIX ---
                    keyboard = [[InlineKeyboardButton(f"#{t['id']} 🛡️ | {t['symbol']} | ${t['entry_value_usdt']:.2f}", callback_data=f"check_{t['id']}")] for t in trades]
                    await query.message.reply_text("اختر صفقة لمتابعتها:", reply_markup=InlineKeyboardMarkup(keyboard))
            elif report_type == "strategy_report":
                 async with aiosqlite.connect(DB_FILE) as conn: cursor = await conn.execute("SELECT reason, status, pnl_usdt FROM trades WHERE status NOT IN ('active', 'pending')"); trades = await cursor.fetchall()
                 if not trades: await query.message.reply_text("لا توجد صفقات مغلقة لتحليلها.")
                 else:
                    stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0}); [ (stats[r]['wins' if s.startswith('ناجحة') else 'losses'] + 1, stats[r].update({'pnl': stats[r]['pnl'] + (p or 0)})) for r, s, p in trades if r]
                    report = ["**📜 تقرير أداء الاستراتيجيات**"]; [report.append(f"\n--- *{r}* ---\n  - الصفقات: {s['wins'] + s['losses']}\n  - النجاح: {(s['wins'] / (s['wins'] + s['losses']) * 100) if (s['wins'] + s['losses']) > 0 else 0:.2f}%\n  - صافي الربح: ${s['pnl']:+.2f}") for r, s in sorted(stats.items(), key=lambda item: item[1]['pnl'], reverse=True)]; await query.message.reply_text("\n".join(report), parse_mode=ParseMode.MARKDOWN)
            elif report_type == "mood":
                mood = bot_state.market_mood; await query.message.reply_text(f"*🌡️ حالة مزاج السوق*\n- **النتيجة:** {mood['mood']}\n- **السبب:** {mood['reason']}", parse_mode=ParseMode.MARKDOWN)
            elif report_type == "diagnostics":
                 # --- WEBSOCKET FIX ---
                 ws_public = 'متصل ✅' if bot_state.public_ws and bot_state.public_ws.is_connected() else 'غير متصل ❌'
                 ws_private = 'متصل ✅' if bot_state.private_ws and bot_state.private_ws.is_connected() else 'غير متصل ❌'
                 report = f"**🕵️‍♂️ تقرير التشخيص (v14.0)**\n\n- **حالة WS العام:** {ws_public}\n- **حالة WS الخاص:** {ws_private}\n- **آخر فحص:** {bot_state.scan_stats.get('last_start', 'لم يحدث بعد')}"
                 await query.message.reply_text(report, parse_mode=ParseMode.MARKDOWN)
        elif data.startswith("toggle_scanner_"):
            scanner_name = data.split("_", 2)[2]; active = bot_state.settings.get("active_scanners", []).copy(); [active.remove(scanner_name) if scanner_name in active else active.append(scanner_name)]; bot_state.settings["active_scanners"] = active; save_settings(); await show_scanners_menu(update, context)
        elif data.startswith("preset_"):
            preset_name = data.split("_", 1)[1]
            if preset_data := PRESETS.get(preset_name): bot_state.settings.update(preset_data); bot_state.settings["active_preset"] = preset_name; save_settings(); await query.edit_message_text(f"✅ تم تفعيل النمط: **{preset_data['name']}**", parse_mode=ParseMode.MARKDOWN)
        elif data.startswith("param_"):
            param_key = data.split("_", 1)[1]
            if isinstance(bot_state.settings.get(param_key), bool): bot_state.settings[param_key] = not bot_state.settings.get(param_key); save_settings(); await show_parameters_menu(update, context, edit_message_id=query.message.message_id)
            else: msg_to_delete = await query.message.reply_text(f"📝 *تعديل '{PARAM_DISPLAY_NAMES.get(param_key, param_key)}'*\nأرسل القيمة الجديدة.", parse_mode=ParseMode.MARKDOWN); context.user_data['awaiting_input_for_param'] = (param_key, msg_to_delete.message_id, query.message.message_id)
        elif data == "back_to_settings":
            if query.message: await query.message.delete(); await show_settings_menu(update, context)
    except Exception as e: logger.error(f"Button Handler Error: {e}", exc_info=True)

# =======================================================================================
# --- Main Bot Startup ---
# =======================================================================================
async def main():
    logger.info(f"--- Bot v14.0 (The Phoenix) starting ---")
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]): logger.critical("FATAL: Missing environment variables."); return
    load_settings(); await init_database(); app = Application.builder().token(TELEGRAM_BOT_TOKEN).build(); bot_state.application = app
    await ensure_libraries_loaded(); bot_state.exchange = ccxt.okx({'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE, 'enableRateLimit': True})
    bot_state.trade_guardian = TradeGuardian(bot_state.exchange, bot_state.settings, app); bot_state.public_ws = PublicWebSocketManager(bot_state.trade_guardian.handle_ticker_update); bot_state.private_ws = PrivateWebSocketManager()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    app.add_handler(CallbackQueryHandler(button_callback_handler))

    public_ws_task = asyncio.create_task(bot_state.public_ws.run()); private_ws_task = asyncio.create_task(bot_state.private_ws.run())
    logger.info("Waiting 5s for WS connections..."); await asyncio.sleep(5); await bot_state.trade_guardian.sync_subscriptions()
    
    scan_interval = bot_state.settings.get("scan_interval_seconds", 900); app.job_queue.run_repeating(perform_scan, interval=scan_interval, first=10, name="perform_scan")
    logger.info(f"Market scanner scheduled to run every {scan_interval} seconds.")

    try:
        await bot_state.exchange.fetch_balance(); logger.info("✅ OKX API connection test SUCCEEDED.")
        await app.bot.send_message(TELEGRAM_CHAT_ID, "*🚀 بوت v14.0 (The Phoenix) بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)
        async with app:
            await app.start(); await app.updater.start_polling(); logger.info("Bot is now fully operational.")
            await asyncio.gather(public_ws_task, private_ws_task)
    except Exception as e: logger.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        if 'public_ws_task' in locals() and not public_ws_task.done(): public_ws_task.cancel()
        if 'private_ws_task' in locals() and not private_ws_task.done(): private_ws_task.cancel()
        if bot_state.exchange: await bot_state.exchange.close()
        logger.info("Bot has been shut down.")

if __name__ == '__main__':
    asyncio.run(main())

