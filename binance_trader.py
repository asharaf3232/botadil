# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 بوت OKX القناص v4.0 (The Guardian Sniper) - النسخة التنفيذية النهائية 🚀 ---
# =======================================================================================
# هذا الإصدار يدمج كل الميزات المطلوبة في حزمة واحدة قوية:
# 1. المحرك الذري الموثوق (v2.0).
# 2. الواجهة التفاعلية الكاملة (من البوت الأصلي).
# 3. المراقب الذكي للوقف المتحرك (v4.0).
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
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict, Counter
import sqlite3

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
DB_FILE = os.path.join(APP_ROOT, 'okx_guardian_sniper_v4.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'okx_guardian_sniper_settings_v4.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Guardian_Sniper_v4")

class BotState:
    def __init__(self):
        self.exchange = None
        self.settings = {}
        self.last_signal_time = {}

bot_state = BotState()
scan_lock = asyncio.Lock()

# =======================================================================================
# --- [MERGE] الثوابت والقواميس الخاصة بواجهة المستخدم ---
# =======================================================================================
DEFAULT_SETTINGS = {
    "real_trade_size_usdt": 15.0,
    "top_n_symbols_by_volume": 200,
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 2.0,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "whale_radar", "sniper_pro"],
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

EDITABLE_PARAMS = {
    "إعدادات المخاطر": [
        "real_trade_size_usdt", "atr_sl_multiplier", "risk_reward_ratio"
    ],
    "إعدادات الوقف المتحرك": [
        "trailing_sl_enabled", "trailing_sl_activation_percent", "trailing_sl_callback_percent"
    ],
    "إعدادات الفحص": [
        "top_n_symbols_by_volume", "scan_interval_seconds", "track_interval_seconds"
    ]
}
PARAM_DISPLAY_NAMES = {
    "real_trade_size_usdt": "💵 حجم الصفقة ($)",
    "atr_sl_multiplier": "مضاعف وقف الخسارة (ATR)",
    "risk_reward_ratio": "نسبة المخاطرة/العائد",
    "trailing_sl_enabled": "⚙️ تفعيل الوقف المتحرك",
    "trailing_sl_activation_percent": "تفعيل الوقف المتحرك (%)",
    "trailing_sl_callback_percent": "مسافة تتبع الوقف (%)",
    "top_n_symbols_by_volume": "عدد العملات للفحص",
    "scan_interval_seconds": "فترة الفحص (ثانية)",
    "track_interval_seconds": "فترة المراقبة (ثانية)"
}
STRATEGIES_MAP = {
    "momentum_breakout": {"func": "analyze_momentum_breakout", "name": "زخم اختراقي"},
    "breakout_squeeze_pro": {"func": "analyze_breakout_squeeze_pro", "name": "اختراق انضغاطي"},
    "support_rebound": {"func": "analyze_support_rebound", "name": "ارتداد الدعم"},
    "sniper_pro": {"func": "analyze_sniper_pro", "name": "القناص المحترف"},
    "whale_radar": {"func": "analyze_whale_radar", "name": "رادار الحيتان"},
}

# =======================================================================================
# --- دوال المساعدة (الإعدادات وقاعدة البيانات) 🗄️ ---
# =======================================================================================

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
        for col in ['highest_price', 'trailing_sl_active', 'algo_id']:
            try:
                cursor.execute(f'ALTER TABLE trades ADD COLUMN {col} ' + ('REAL' if col == 'highest_price' else 'TEXT' if col == 'algo_id' else 'BOOLEAN DEFAULT 0'))
                conn.commit()
            except sqlite3.OperationalError: pass
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

# =======================================================================================
# --- 🧠 العقل: دوال التحليل والاستراتيجيات 🧠 ---
# =======================================================================================
# (هذا القسم لم يتغير، فهو يحتوي على نفس الاستراتيجيات القوية)
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None
def analyze_momentum_breakout(df, rvol):
    df.ta.vwap(append=True); df.ta.bbands(length=20, std=2.0, append=True); df.ta.macd(fast=12, slow=26, signal=9, append=True); df.ta.rsi(length=14, append=True)
    last, prev = df.iloc[-2], df.iloc[-3]
    macd_col, macds_col, bbu_col, rsi_col = find_col(df.columns, "MACD_"), find_col(df.columns, "MACDs_"), find_col(df.columns, "BBU_"), find_col(df.columns, "RSI_")
    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and last[rsi_col] < 68):
        return {"reason": "Momentum Breakout", "type": "long"}
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
        if breakout_fired and volume_ok and obv_rising: return {"reason": "Breakout Squeeze", "type": "long"}
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
                return {"reason": "Support Rebound", "type": "long"}
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
                return {"reason": "Sniper Pro", "type": "long"}
    except Exception: return None
    return None
async def analyze_whale_radar(df, rvol, exchange, symbol):
    try:
        ob = await exchange.fetch_order_book(symbol, limit=20)
        if not ob or not ob.get('bids'): return None
        total_bid_value = sum(float(price) * float(qty) for price, qty in ob['bids'][:10])
        if total_bid_value > 30000:
            return {"reason": "Whale Radar", "type": "long"}
    except Exception: return None
    return None

# =======================================================================================
# --- [NEW v4.0] المراقب الذكي: منطق الوقف المتحرك ---
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
            
            new_sl = None
            is_activation = False
            
            if not trade['trailing_sl_active']:
                activation_price = trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100)
                if current_price >= activation_price:
                    new_sl = trade['entry_price']
                    is_activation = True
            else:
                callback_price = highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                if callback_price > trade['stop_loss']:
                    new_sl = callback_price
            
            if new_sl and new_sl > trade['stop_loss']:
                logger.info(f"{'ACTIVATING' if is_activation else 'UPDATING'} Trailing SL for trade #{trade['id']}. New SL: {new_sl}")
                await exchange.private_post_trade_cancel_algos([{'instId': exchange.market_id(trade['symbol']), 'algoId': trade['algo_id']}])
                new_algo_id = await place_okx_oco_order(exchange, trade['symbol'], trade['quantity'], trade['take_profit'], new_sl)
                
                conn = sqlite3.connect(DB_FILE)
                conn.cursor().execute("UPDATE trades SET stop_loss=?, highest_price=?, trailing_sl_active=1, algo_id=? WHERE id=?",
                                      (new_sl, highest_price, new_algo_id, trade['id']))
                conn.commit()
                conn.close()
                if is_activation:
                    await bot.send_message(TELEGRAM_CHAT_ID, f"**🚀 تأمين الأرباح! | #{trade['id']} {trade['symbol']}**\n\nتم رفع وقف الخسارة إلى نقطة الدخول. الصفقة الآن بدون مخاطرة!", parse_mode=ParseMode.MARKDOWN)

            elif highest_price > (trade.get('highest_price') or 0):
                conn = sqlite3.connect(DB_FILE)
                conn.cursor().execute("UPDATE trades SET highest_price=? WHERE id=?", (highest_price, trade['id']))
                conn.commit()
                conn.close()
        except ccxt.OrderNotFound:
             logger.warning(f"Order for trade #{trade['id']} not found on exchange. It might have been filled or manually cancelled.")
             # A full implementation would check position status and close the trade in DB if needed.
        except Exception as e:
            logger.error(f"Error in trailing stop logic for trade #{trade['id']}: {e}")

# =======================================================================================
# --- 🦾 جسد البوت: منطق التشغيل والفحص والتداول 🦾 ---
# =======================================================================================
async def execute_atomic_trade(signal):
    symbol = signal['symbol']
    settings = bot_state.settings
    exchange = bot_state.exchange
    bot = application.bot
    logger.info(f"Attempting ATOMIC trade for {symbol} using attachAlgoOrds.")
    try:
        quantity_to_buy = settings['real_trade_size_usdt'] / signal['entry_price']
        tp_price_str = exchange.price_to_precision(symbol, signal['take_profit'])
        sl_price_str = exchange.price_to_precision(symbol, signal['stop_loss'])
        attached_algo_orders = [
            {'tpTriggerPx': tp_price_str, 'tpOrdPx': '-1', 'side': 'sell'},
            {'slTriggerPx': sl_price_str, 'slOrdPx': '-1', 'side': 'sell'}
        ]
        params = {'tdMode': 'cash', 'attachAlgoOrds': attached_algo_orders, 'clOrdId': f'sniper_{int(time.time()*1000)}'}
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
                algo_order = next((o for o in open_orders if o.get('clientOrderId') == verified_order.get('clientOrderId')), None)
                algo_id = algo_order.get('id') if algo_order else 'unknown'
                
                avg_price = verified_order.get('average', signal['entry_price'])
                original_risk = signal['entry_price'] - signal['stop_loss']
                signal['final_sl'] = avg_price - original_risk
                signal['final_tp'] = avg_price + (original_risk * settings['risk_reward_ratio'])
                trade_id = log_trade_to_db(signal, verified_order, algo_id)
                tp_percent = (signal['final_tp'] - avg_price) / avg_price * 100
                sl_percent = (avg_price - signal['final_sl']) / avg_price * 100
                success_msg = (
                    f"**✅ صفقة ذرية ناجحة | {symbol} (ID: {trade_id})**\n"
                    f"------------------------------------\n"
                    f"🔍 **الاستراتيجية:** {signal['reason']}\n\n"
                    f"📈 **متوسط الشراء:** `{avg_price:,.4f}`\n"
                    f"🎯 **الهدف:** `{signal['final_tp']:,.4f}` (+{tp_percent:.2f}%)\n"
                    f"🛑 **الوقف:** `{signal['final_sl']:,.4f}` (-{sl_percent:.2f}%)\n\n"
                    f"***الصفقة مؤمنة بالكامل بشكل ذري.***"
                )
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=success_msg, parse_mode=ParseMode.MARKDOWN)
                return
        raise Exception("Failed to verify order and protection status.")
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during atomic trade for {symbol}: {e}")
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"**🔥🔥🔥 فشل ذري حرج - {symbol}**\n\n**الخطأ:** `{str(e)}`", parse_mode=ParseMode.MARKDOWN)

async def worker(queue, signals_list):
    # This is the full worker logic, adapted to use the settings dictionary
    settings = bot_state.settings
    exchange = bot_state.exchange
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
                strategy_func = globals()[strategy_info['func']]
                
                if asyncio.iscoroutinefunction(strategy_func):
                    result = await strategy_func(df.copy(), rvol, exchange, symbol)
                else:
                    result = strategy_func(df.copy(), rvol)
                if result: confirmed_reasons.append(result['reason'])

            if confirmed_reasons:
                reason_str = ' + '.join(confirmed_reasons)
                entry_price = last_close
                df.ta.atr(length=14, append=True) # ATR for SL calc
                atr_col = find_col(df.columns, f"ATRr_14")
                current_atr = df.iloc[-2].get(atr_col, 0)
                if current_atr > 0:
                    risk_per_unit = current_atr * settings['atr_sl_multiplier']
                    stop_loss = entry_price - risk_per_unit
                    take_profit = entry_price + (risk_per_unit * settings['risk_reward_ratio'])
                    
                    tp_perc = (take_profit - entry_price) / entry_price * 100
                    sl_perc = (entry_price - stop_loss) / entry_price * 100
                    if tp_perc >= settings['min_tp_sl_filter']['min_tp_percent'] and sl_perc >= settings['min_tp_sl_filter']['min_sl_percent']:
                        signals_list.append({"symbol": symbol, "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "reason": reason_str})
        except Exception: pass
        finally: queue.task_done()

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        settings = bot_state.settings
        exchange = bot_state.exchange
        logger.info("--- Starting new market scan... ---")
        try:
            tickers = await exchange.fetch_tickers()
            usdt_markets = [m for m in tickers.values() if m.get('symbol', '').endswith('/USDT') and not any(k in m['symbol'] for k in ['-SWAP', 'UP', 'DOWN', '3L', '3S']) and m.get('quoteVolume', 0) > settings['liquidity_filters']['min_quote_volume_24h_usd']]
            usdt_markets.sort(key=lambda m: m.get('quoteVolume', 0), reverse=True)
            top_markets = usdt_markets[:settings['top_n_symbols_by_volume']]
        except Exception as e:
            logger.error(f"Failed to fetch markets from OKX: {e}")
            return

        queue = asyncio.Queue()
        for market in top_markets: await queue.put(market)
        signals_found = []
        worker_tasks = [asyncio.create_task(worker(queue, signals_found)) for _ in range(10)]
        await queue.join()
        for task in worker_tasks: task.cancel()

        if signals_found:
            logger.info(f"+++ Scan complete. Found {len(signals_found)} potential signals! +++")
            for signal in signals_found:
                symbol = signal['symbol']
                if time.time() - bot_state.last_signal_time.get(symbol, 0) < settings['scan_interval_seconds'] * 2.5:
                    continue
                bot_state.last_signal_time[symbol] = time.time()
                await execute_atomic_trade(signal)
                await asyncio.sleep(10)

# =======================================================================================
# --- 📱 واجهة التحكم عبر تليجرام 📱 ---
# =======================================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"], ["ℹ️ مساعدة"]]
    await update.message.reply_text("أهلاً بك في بوت OKX القناص v4.0 (الحارس)", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("📈 الصفقات النشطة", callback_data="active_trades")]]
    await update.message.reply_text("🖥️ *لوحة التحكم الرئيسية*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
    
async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["🎭 تفعيل/تعطيل الماسحات", "🔧 تعديل المعايير"], ["🔙 القائمة الرئيسية"]]
    await update.message.reply_text("اختر الإعداد:", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active_scanners = bot_state.settings.get("active_scanners", [])
    keyboard = [[InlineKeyboardButton(f"{'✅' if key in active_scanners else '❌'} {value['name']}", callback_data=f"toggle_scanner_{key}")] for key, value in STRATEGIES_MAP.items()]
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    await update.message.reply_text("اختر الماسحات لتفعيلها أو تعطيلها:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    settings = bot_state.settings
    for category, params in EDITABLE_PARAMS.items():
        keyboard.append([InlineKeyboardButton(f"--- {category} ---", callback_data="ignore")])
        for param_key in params:
            display_name = PARAM_DISPLAY_NAMES.get(param_key, param_key)
            current_value = settings.get(param_key, "N/A")
            text = f"{display_name}: {'مُفعّل ✅' if current_value else 'مُعطّل ❌'}" if isinstance(current_value, bool) else f"{display_name}: {current_value}"
            keyboard.append([InlineKeyboardButton(text, callback_data=f"param_{param_key}")])
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    await update.message.reply_text("⚙️ *الإعدادات المتقدمة*\n\nاختر الإعداد الذي تريد تعديله:", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    menu_map = {
        "Dashboard 🖥️": show_dashboard_command, "⚙️ الإعدادات": show_settings_menu,
        "🎭 تفعيل/تعطيل الماسحات": show_scanners_menu, "🔧 تعديل المعايير": show_parameters_menu,
        "🔙 القائمة الرئيسية": start_command
    }
    if text in menu_map:
        await menu_map[text](update, context)

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data.startswith("toggle_scanner_"):
        scanner_name = data.split("toggle_scanner_")[1]
        active_scanners = bot_state.settings.get("active_scanners", []).copy()
        if scanner_name in active_scanners: active_scanners.remove(scanner_name)
        else: active_scanners.append(scanner_name)
        bot_state.settings["active_scanners"] = active_scanners
        save_settings()
        
        keyboard = [[InlineKeyboardButton(f"{'✅' if key in active_scanners else '❌'} {value['name']}", callback_data=f"toggle_scanner_{key}")] for key, value in STRATEGIES_MAP.items()]
        keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
        await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data.startswith("param_"):
        param_key = data.split("param_")[1]
        context.user_data['awaiting_input_for_param'] = param_key
        await query.message.reply_text(f"📝 *تعديل '{PARAM_DISPLAY_NAMES.get(param_key, param_key)}'*\n\n*القيمة الحالية:* `{bot_state.settings.get(param_key)}`\n\nأرسل القيمة الجديدة.", parse_mode=ParseMode.MARKDOWN)

    elif data == "active_trades":
        conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row
        trades = conn.cursor().execute("SELECT id, symbol, entry_value_usdt FROM trades WHERE status = 'active' ORDER BY id DESC").fetchall()
        conn.close()
        if not trades:
            await query.message.reply_text("لا توجد صفقات نشطة حالياً.")
            return
        keyboard = [[InlineKeyboardButton(f"#{t['id']} | {t['symbol']} | ${t['entry_value_usdt']:.2f}", callback_data=f"check_{t['id']}")] for t in trades]
        await query.message.reply_text("اختر صفقة لمتابعتها:", reply_markup=InlineKeyboardMarkup(keyboard))

async def input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'awaiting_input_for_param' in context.user_data:
        param_key = context.user_data.pop('awaiting_input_for_param')
        new_value_str = update.message.text
        settings = bot_state.settings
        try:
            current_value = settings.get(param_key)
            if isinstance(current_value, bool):
                new_value = new_value_str.lower() in ['true', '1', 'on', 'yes', 'نعم']
            elif isinstance(current_value, float):
                new_value = float(new_value_str)
            else:
                new_value = int(new_value_str)
            
            settings[param_key] = new_value
            save_settings()
            await update.message.reply_text(f"✅ تم تحديث **{PARAM_DISPLAY_NAMES.get(param_key, param_key)}** إلى `{new_value}`.", parse_mode=ParseMode.MARKDOWN)
        except (ValueError, TypeError):
            await update.message.reply_text("❌ قيمة غير صالحة. الرجاء المحاولة مرة أخرى.")

# =======================================================================================
# --- 🚀 نقطة انطلاق البوت 🚀 ---
# =======================================================================================
application = None
async def post_init(app: Application):
    global application
    application = app
    logger.info("🚀 Starting OKX Guardian Sniper v4.0...")
    if 'YOUR_OKX_API_KEY' in OKX_API_KEY or 'YOUR_BOT_TOKEN' in TELEGRAM_BOT_TOKEN:
        logger.critical("FATAL: API keys or Bot Token are not set.")
        return
    bot_state.exchange = ccxt.okx({'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE, 'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    def _request_patch(self, path, api="private", method="POST", params=None, headers=None, body=None, config=None, context=None):
        if params is None: params = {}
        if (path == 'trade/order-algo') or (path == 'trade/order' and 'attachAlgoOrds' in params):
            if params.get("side") == "sell": params.pop("tgtCcy", None)
        return self.fetch2(path, api, method, params, headers, body, config, context)
    bot_state.exchange.request = types.MethodType(_request_patch, bot_state.exchange)
    logger.info("Applied monkey-patch to fix OKX 'tgtCcy' parameter issue.")
    
    scan_interval = bot_state.settings.get("scan_interval_seconds", 900)
    track_interval = bot_state.settings.get("track_interval_seconds", 60)
    app.job_queue.run_repeating(perform_scan, interval=scan_interval, first=10)
    app.job_queue.run_repeating(track_open_trades, interval=track_interval, first=30)
    logger.info(f"Scan job scheduled every {scan_interval}s. Tracker job scheduled every {track_interval}s.")
    await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="*🚀 بوت OKX الحارس v4.0 بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)

def main():
    load_settings()
    init_database()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex('^[0-9.,]+$'), text_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex('^[0-9.,truefalseonoffyesnoنعملا]+$'), input_handler))
    app.add_handler(CallbackQueryHandler(button_callback_handler))
    app.run_polling()

if __name__ == '__main__':
    main()
