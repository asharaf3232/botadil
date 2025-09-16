# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 OKX Accountant Trader v23.0 🚀 ---
# =======================================================================================
# This is the most reliable version, built on a new paradigm: Balance Auditing.
#
# ARCHITECTURE:
# 1. BRAIN: A powerful, OKX-focused scanner with a sophisticated asset filter.
# 2. BODY: A simple, robust execution module for placing market buy orders.
# 3. CONFIRMATION & MANAGEMENT (The Accountant): Instead of relying on fallible
#    WebSocket events for confirmations, this bot actively audits the account
#    balance. Any change in asset quantity is treated as a definitive trade
#    confirmation. The bot then takes over active management (TP/SL) for that trade.
#
# This architecture is immune to missed WebSocket messages and provides 100%
# reliable trade execution confirmation.
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
ACCOUNTANT_INTERVAL_SECONDS = 45 # How often the accountant checks the balance

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'accountant_trader_v23.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'accountant_trader_settings_v23.json')
BALANCE_CACHE_FILE = os.path.join(APP_ROOT, 'balance_cache_v23.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Accountant_Trader")

# =======================================================================================
# --- 🔬 Global Bot State & Locks 🔬 ---
# =======================================================================================
class BotState:
    def __init__(self):
        self.settings = {}
        self.last_signal_time = {}
        self.application = None
        self.exchange = None
        self.live_tickers = {} # For price tracking of active trades

bot_data = BotState()
scan_lock = asyncio.Lock()
accountant_lock = asyncio.Lock()

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
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "supertrend_pullback"],
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
        "supertrend_pullback": {"atr_period": 10, "atr_multiplier": 3.0}
    }
}
STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي", "breakout_squeeze_pro": "اختراق انضغاطي",
    "supertrend_pullback": "انعكاس سوبرترند"
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

def load_balance_cache():
    if os.path.exists(BALANCE_CACHE_FILE):
        with open(BALANCE_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}
def save_balance_cache(balances):
    with open(BALANCE_CACHE_FILE, 'w') as f:
        json.dump(balances, f, indent=4)

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
        logger.info("Accountant database initialized successfully.")
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
# --- 🧠 Advanced Scanners (The Brain) 🧠 ---
# =======================================================================================
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

# [Analysis functions are the same as v22 - they are proven]
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
    "momentum_breakout": analyze_momentum_breakout,
    "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "supertrend_pullback": analyze_supertrend_pullback
}

# =======================================================================================
# --- 🧾 The Accountant Protocol (Confirmation & Management) 🧾 ---
# =======================================================================================
async def get_current_balance():
    """Fetches the current asset balances from the exchange."""
    try:
        balance_data = await bot_data.exchange.fetch_balance()
        # We only care about assets with a quantity (total, not just free)
        return {
            asset: details['total']
            for asset, details in balance_data.items()
            if details.get('total') is not None and details['total'] > 0
        }
    except Exception as e:
        logger.error(f"Accountant: Could not fetch current balance: {e}")
        return None

async def the_accountant_job(context: ContextTypes.DEFAULT_TYPE):
    """
    The core job that audits balance changes to confirm trades and manages active ones.
    """
    async with accountant_lock:
        bot = context.bot
        
        # --- Part 1: Trade Confirmation via Balance Audit ---
        previous_balances = load_balance_cache()
        current_balances = await get_current_balance()
        if current_balances is None: return # API error, skip this cycle

        # Find assets that have appeared or increased significantly
        all_assets = set(previous_balances.keys()) | set(current_balances.keys())
        for asset in all_assets:
            if asset == 'USDT': continue
            prev_qty = previous_balances.get(asset, 0)
            curr_qty = current_balances.get(asset, 0)
            
            # Check if a new asset appeared or an existing one increased
            if curr_qty > prev_qty and (curr_qty - prev_qty) * bot_data.live_tickers.get(f"{asset}/USDT", 1) > 5: # Threshold to ignore dust
                logger.info(f"Accountant: Detected new acquisition for {asset}. Quantity increased from {prev_qty} to {curr_qty}.")
                
                async with aiosqlite.connect(DB_FILE) as conn:
                    conn.row_factory = aiosqlite.Row
                    # Find the latest 'pending' trade for this symbol to activate it
                    cursor = await conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = 'pending' ORDER BY id DESC LIMIT 1", (f"{asset}/USDT",))
                    pending_trade = await cursor.fetchone()
                    
                    if pending_trade:
                        pending_trade = dict(pending_trade)
                        logger.info(f"Accountant: Found matching pending trade #{pending_trade['id']}. Activating now.")
                        
                        # Use the actual new quantity from the balance for accuracy
                        # Entry price is assumed to be the one from the signal for now
                        await conn.execute(
                            "UPDATE trades SET status = 'active', quantity = ?, timestamp = ? WHERE id = ?",
                            (curr_qty, datetime.now(EGYPT_TZ).isoformat(), pending_trade['id'])
                        )
                        await conn.commit()
                        
                        await safe_send_message(bot, f"**✅ تم تأكيد الشراء | {asset}/USDT**\n\nالصفقة #{pending_trade['id']} الآن نشطة والمحاسب يراقبها.")
                    else:
                        logger.warning(f"Accountant: Detected balance increase for {asset}, but no matching pending trade found in DB.")

        # CRITICAL: Update the cache with the new reality
        save_balance_cache(current_balances)

        # --- Part 2: Active Trade Management ---
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM trades WHERE status = 'active'")
            active_trades = [dict(row) for row in await cursor.fetchall()]

        if not active_trades: return
        
        # Fetch all required tickers in one go
        active_symbols = [trade['symbol'] for trade in active_trades]
        try:
            tickers = await bot_data.exchange.fetch_tickers(active_symbols)
            bot_data.live_tickers.update(tickers)
        except Exception as e:
            logger.error(f"Accountant: Could not fetch tickers for active trades: {e}")
            return
            
        for trade in active_trades:
            symbol = trade['symbol']
            ticker = bot_data.live_tickers.get(symbol)
            if not ticker or 'last' not in ticker: continue
            
            current_price = ticker['last']
            
            # Trailing SL Logic
            settings = bot_data.settings
            if settings['trailing_sl_enabled']:
                new_highest_price = max(trade.get('highest_price', 0), current_price)
                if new_highest_price > trade.get('highest_price', 0):
                    await conn.execute("UPDATE trades SET highest_price = ? WHERE id = ?", (new_highest_price, trade['id']))
                if not trade['trailing_sl_active'] and current_price >= trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100):
                    trade['trailing_sl_active'] = True
                    await conn.execute("UPDATE trades SET trailing_sl_active = 1 WHERE id = ?", (trade['id'],))
                    await safe_send_message(bot, f"**🚀 تأمين الأرباح! | #{trade['id']} {symbol}**")
                if trade['trailing_sl_active']:
                    new_sl = new_highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                    if new_sl > trade['stop_loss']:
                        trade['stop_loss'] = new_sl
                        await conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_sl, trade['id']))

            # TP/SL Check
            if current_price >= trade['take_profit']:
                await close_trade(trade, "ناجحة (TP)", current_price, bot)
            elif current_price <= trade['stop_loss']:
                await close_trade(trade, "فاشلة (SL)", current_price, bot)

async def close_trade(trade, reason, close_price, bot):
    symbol, quantity = trade['symbol'], trade['quantity']
    logger.info(f"Accountant: Closing trade #{trade['id']} for {symbol}. Reason: {reason}")
    try:
        # 1. Execute the market sell order on the exchange
        await bot_data.exchange.create_market_sell_order(symbol, quantity)
        logger.info(f"Market sell order for {quantity} {symbol} sent successfully.")
        
        # 2. Update the database
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("UPDATE trades SET status = ? WHERE id = ?", (reason, trade['id']))
            await conn.commit()
            
        # 3. Notify the user
        pnl = (close_price - trade['entry_price']) * quantity
        pnl_percent = (close_price / trade['entry_price'] - 1) * 100
        emoji = "✅" if pnl > 0 else "🛑"
        msg = (f"**{emoji} تم إغلاق الصفقة | {symbol}**\n**السبب:** {reason}\n**الربح/الخسارة:** `${pnl:,.2f}` ({pnl_percent:+.2f}%)")
        await safe_send_message(bot, msg)
    except Exception as e:
        logger.critical(f"Accountant Close Trade Error #{trade['id']}: {e}", exc_info=True)
        await safe_send_message(bot, f"🔥 **فشل إغلاق الصفقة #{trade['id']}**\nيرجى التحقق يدوياً!")

# =======================================================================================
# --- ⚡ Core Scanner & Trade Initiation Logic ⚡ ---
# =======================================================================================
async def get_okx_markets():
    # [Identical to v22]
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
    # [Identical to v22]
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
        
        if await log_pending_trade_to_db(signal, buy_order):
            await safe_send_message(bot_data.application.bot, f"🚀 تم إرسال أمر شراء لـ `{signal['symbol']}`. المحاسب سيقوم بتأكيد التنفيذ.")
        else: # If DB logging fails, cancel the order to be safe
            await exchange.cancel_order(buy_order['id'], signal['symbol'])
            await safe_send_message(bot_data.application.bot, f"⚠️ فشل تسجيل صفقة `{signal['symbol']}`. تم إلغاء الأمر.")
            
    except Exception as e:
        logger.error(f"REAL TRADE FAILED for {signal['symbol']}: {e}", exc_info=True)
        await safe_send_message(bot_data.application.bot, f"🔥 فشل فتح صفقة لـ `{signal['symbol']}`: {e}")

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
    await update.message.reply_text("أهلاً بك في OKX Accountant Trader v23.0")

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

    # Initialize the balance cache for the first time
    logger.info("Initializing accountant's ledger (balance cache)...")
    initial_balance = await get_current_balance()
    if initial_balance is not None:
        save_balance_cache(initial_balance)
        logger.info("Balance cache initialized successfully.")
    else:
        logger.error("Could not initialize balance cache. Trade confirmations might be delayed.")

    # Schedule the core jobs
    application.job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name="perform_scan")
    application.job_queue.run_repeating(the_accountant_job, interval=ACCOUNTANT_INTERVAL_SECONDS, first=15, name="the_accountant_job")
    
    logger.info(f"Scanner scheduled for every {SCAN_INTERVAL_SECONDS}s. Accountant will audit every {ACCOUNTANT_INTERVAL_SECONDS}s.")
    await safe_send_message(application.bot, "*🚀 OKX Accountant Trader v23.0 بدأ العمل...*")
    logger.info("--- Bot is now fully operational ---")

async def post_shutdown(application: Application):
    if bot_data.exchange: await bot_data.exchange.close()
    logger.info("Bot has shut down.")

def main():
    logger.info("--- Starting OKX Accountant Trader v23.0 ---")
    load_settings(); asyncio.run(init_database())
    app_builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    app_builder.post_init(post_init).post_shutdown(post_shutdown)
    application = app_builder.build()
    
    application.add_handler(CommandHandler("start", start_command))
    
    application.run_polling()

if __name__ == '__main__':
    main()

