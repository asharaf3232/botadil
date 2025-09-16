# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 بوت OKX القناص v8.1 (The Phoenix - Network Hardened) 🚀 ---
# =======================================================================================
# هذا الإصدار هو نسخة مصفحة ضد مشاكل الشبكة وعدم الاستقرار.
# 1. إصلاح الخلل الجذري في تمرير البيانات الذي كان يعطل "ساعي البريد".
# 2. إضافة "نبضات قلب" (ping/pong) لاتصال الويب سوكيت لضمان استقراره.
# 3. تغيير نقطة الوصول الافتراضية للمنصة إلى سيرفرات AWS لتحسين الاستجابة.
# =======================================================================================

# --- المكتبات ---
import asyncio
import os
import logging
import json
import time
import hmac
import base64
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict, Counter
import aiosqlite
import httpx
import websockets

# --- المكتبات الثقيلة (سيتم تحميلها لاحقًا) ---
ccxt = None
pd = None
ta = None

from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest, Forbidden
from dotenv import load_dotenv

# =======================================================================================
# --- ⚙️ الإعدادات الأساسية ⚙️ ---
# =======================================================================================
load_dotenv() 

OKX_API_KEY = os.getenv('OKX_API_KEY')
OKX_API_SECRET = os.getenv('OKX_API_SECRET')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'okx_phoenix_v8.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'okx_phoenix_settings_v8.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Phoenix_v8.1")

class BotState:
    def __init__(self):
        self.exchange = None
        self.settings = {}
        self.live_tickers = {}
        self.last_signal_time = {}
        self.market_mood = {"mood": "UNKNOWN", "reason": "تحليل لم يتم بعد", "btc_mood": "UNKNOWN", "fng": "N/A", "news": "N/A"}
        self.scan_stats = {"last_start": None, "last_duration": "N/A", "markets_scanned": 0, "failures": 0}
        self.ws_manager = None
        self.application = None

bot_state = BotState()
scan_lock = asyncio.Lock()

# =======================================================================================
# --- الثوابت والقواميس الخاصة بواجهة المستخدم ---
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
    "إعدادات الفحص والمزاج": ["top_n_symbols_by_volume", "fear_and_greed_threshold", "market_mood_filter_enabled", "scan_interval_seconds"]
}
PARAM_DISPLAY_NAMES = {
    "real_trade_size_usdt": "💵 حجم الصفقة ($)", "atr_sl_multiplier": "مضاعف وقف الخسارة (ATR)",
    "risk_reward_ratio": "نسبة المخاطرة/العائد", "trailing_sl_enabled": "⚙️ تفعيل الوقف المتحرك",
    "trailing_sl_activation_percent": "تفعيل الوقف المتحرك (%)", "trailing_sl_callback_percent": "مسافة تتبع الوقف (%)",
    "top_n_symbols_by_volume": "عدد العملات للفحص", "fear_and_greed_threshold": "حد مؤشر الخوف",
    "market_mood_filter_enabled": "فلتر مزاج السوق",
    "scan_interval_seconds": "⏱️ الفاصل الزمني للفحص (ثواني)"
}
STRATEGIES_MAP = {
    "momentum_breakout": {"func_name": "analyze_momentum_breakout", "name": "زخم اختراقي"},
    "breakout_squeeze_pro": {"func_name": "analyze_breakout_squeeze_pro", "name": "اختراق انضغاطي"},
    "support_rebound": {"func_name": "analyze_support_rebound", "name": "ارتداد الدعم"},
    "sniper_pro": {"func_name": "analyze_sniper_pro", "name": "القناص المحترف"},
    "whale_radar": {"func_name": "analyze_whale_radar", "name": "رادار الحيتان"},
}

# --- دالة مساعدة لضمان تحميل المكتبات الثقيلة مرة واحدة فقط ---
async def ensure_libraries_loaded():
    global pd, ta, ccxt
    if pd is None: logger.info("تحميل مكتبة pandas لأول مرة..."); import pandas as pd_lib; pd = pd_lib
    if ta is None: logger.info("تحميل مكتبة pandas-ta لأول مرة..."); import pandas_ta as ta_lib; ta = ta_lib
    if ccxt is None: logger.info("تحميل مكتبة ccxt لأول مرة..."); import ccxt.async_support as ccxt_lib; ccxt = ccxt_lib

# =======================================================================================
# --- 🗄️ دوال المساعدة وقاعدة البيانات 🗄️ ---
# =======================================================================================
def escape_markdown(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
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

# --- [FIXED] This function now correctly saves all necessary data for the Postman ---
async def log_initial_trade_to_db(signal, buy_order):
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            sql = '''INSERT INTO trades (timestamp, symbol, entry_price, take_profit, stop_loss, quantity, reason, order_id, status)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''
            params = (
                datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                signal['symbol'],
                signal['entry_price'],      # Use the strategic entry price from the signal
                signal['take_profit'],      # Save the calculated take profit
                signal['stop_loss'],        # Save the calculated stop loss
                buy_order['amount'],
                signal['reason'],
                buy_order['id'],
                'pending_protection'
            )
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Failed to log initial trade to DB: {e}", exc_info=True)
        return None

async def get_fear_and_greed_index():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            return int(r.json()['data'][0]['value'])
    except Exception: return None

async def get_market_mood():
    await ensure_libraries_loaded()
    try:
        exchange = bot_state.exchange
        htf_period = bot_state.settings['trend_filters']['htf_period']
        ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '4h', limit=htf_period + 5)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
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
        
    return {"mood": "POSITIVE", "reason": "وضع السوق مناسب", "btc_mood": btc_mood_text, "fng": fng_text, "news": "N/A"}

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
# --- 🦾 جسد البوت: منطق التشغيل والفحص والتداول 🦾 ---
# =======================================================================================

async def initiate_trade(signal, bot: "telegram.Bot"):
    await ensure_libraries_loaded()
    symbol, settings, exchange = signal['symbol'], bot_state.settings, bot_state.exchange
    logger.info(f"Initiating trade for {symbol} using LIMIT order.")
    
    try:
        ticker = await exchange.fetch_ticker(symbol)
        limit_price = ticker['ask'] 
        if not limit_price or limit_price <= 0:
            raise ValueError(f"Invalid ask price received for {symbol}: {limit_price}")
            
        quantity_to_buy = settings['real_trade_size_usdt'] / limit_price
        
        logger.info(f"Placing LIMIT BUY order for {quantity_to_buy:.6f} of {symbol} at price {limit_price:.6f}")
        buy_order = await exchange.create_limit_buy_order(symbol, quantity_to_buy, limit_price)
        
        trade_id = await log_initial_trade_to_db(signal, buy_order)
        
        if trade_id:
            logger.info(f"Trade #{trade_id} initiated for {symbol} and is now monitored by the Postman.")
            initiate_msg = (f"**⏳ تم بدء صفقة | {symbol} (ID: {trade_id})**\n"
                            f"تم إرسال أمر الشراء المحدد بنجاح.\n"
                            f"يقوم 'ساعي البريد' الآن بمراقبة الأمر لتأمينه فور التنفيذ.")
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=initiate_msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await exchange.cancel_order(buy_order['id'], symbol)
            raise Exception("Failed to log the initiated trade to the database. Order cancelled.")

    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during trade initiation for {symbol}: {e}", exc_info=True)
        error_message = f"**🔥🔥🔥 فشل حرج عند بدء صفقة {symbol}**\n\n**الخطأ:** `{str(e)}`"
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=error_message, parse_mode=ParseMode.MARKDOWN)

async def worker(queue, signals_list, failure_counter):
    await ensure_libraries_loaded()
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
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] == 0: continue
            rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
            if rvol < settings['liquidity_filters']['min_rvol']: continue
            
            df.ta.atr(length=14, append=True)
            atr_col = find_col(df.columns, 'ATRr_')
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
                strategy_info, result = STRATEGIES_MAP.get(name), None
                if not strategy_info: continue
                strategy_func = globals()[strategy_info['func_name']]
                if asyncio.iscoroutinefunction(strategy_func): result = await strategy_func(df.copy(), rvol, exchange, symbol)
                else: result = strategy_func(df.copy(), rvol)
                if result: confirmed_reasons.append(result['reason'])
            
            if confirmed_reasons:
                reason_str = ' + '.join(confirmed_reasons)
                entry_price = last_close
                df.ta.atr(length=14, append=True)
                atr_col = find_col(df.columns, "ATRr_14")
                current_atr = df.iloc[-2].get(atr_col, 0)
                if current_atr > 0:
                    risk = current_atr * settings['atr_sl_multiplier']
                    stop_loss, take_profit = entry_price - risk, entry_price + (risk * settings['risk_reward_ratio'])
                    if (take_profit / entry_price - 1) * 100 >= settings['min_tp_sl_filter']['min_tp_percent'] and \
                       (1 - stop_loss / entry_price) * 100 >= settings['min_tp_sl_filter']['min_sl_percent']:
                        signals_list.append({
                            "symbol": symbol,
                            "reason": reason_str,
                            "entry_price": entry_price,
                            "take_profit": take_profit,
                            "stop_loss": stop_loss
                        })
        except Exception as e:
            logger.debug(f"Worker error for {symbol}: {e}")
            failure_counter[0] += 1
        finally:
            queue.task_done()

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        settings, bot, exchange = bot_state.settings, context.bot, bot_state.exchange
        bot_state.scan_stats['last_start'] = datetime.now(EGYPT_TZ)
        logger.info("--- Starting new market scan... ---")
        
        if settings['market_mood_filter_enabled']:
            mood_result = await get_market_mood()
            bot_state.market_mood = mood_result
            if mood_result['mood'] in ["NEGATIVE", "DANGEROUS"]:
                logger.warning(f"SCAN SKIPPED: {mood_result['reason']}")
                return
        
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
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        
        bot_state.scan_stats['failures'] = failure_counter[0]
        duration = (datetime.now(EGYPT_TZ) - bot_state.scan_stats['last_start']).total_seconds()
        bot_state.scan_stats['last_duration'] = f"{duration:.0f} ثانية"

        new_trades = 0
        if signals_found:
            logger.info(f"+++ Scan complete. Found {len(signals_found)} signals! +++")
            for signal in signals_found:
                if time.time() - bot_state.last_signal_time.get(signal['symbol'], 0) < settings['scan_interval_seconds'] * 2.5: 
                    continue
                try:
                    usdt_balance = (await exchange.fetch_balance())['free'].get('USDT', 0.0)
                    trade_size = settings['real_trade_size_usdt']
                    if usdt_balance >= trade_size:
                        logger.info(f"Sufficient USDT balance ({usdt_balance:.2f}) for trade size ({trade_size}). Proceeding...")
                        bot_state.last_signal_time[signal['symbol']] = time.time()
                        await initiate_trade(signal, bot)
                        new_trades += 1
                        logger.info("Waiting for 25 seconds before attempting next trade...")
                        await asyncio.sleep(25) 
                    else:
                        logger.warning(f"Insufficient USDT balance ({usdt_balance:.2f}). Stopping further trades for this scan cycle.")
                        break
                except Exception as e:
                    logger.error(f"Error during pre-trade balance check for {signal['symbol']}: {e}")
                    break
        else:
            logger.info("--- Scan complete. No new signals found. ---")
        
        scan_summary = (f"**🔬 ملخص الفحص الأخير**\n\n"
                       f"- **الحالة:** اكتمل بنجاح\n"
                       f"- **وضع السوق:** {bot_state.market_mood['mood']} ({bot_state.market_mood.get('btc_mood', 'N/A')})\n"
                       f"- **المدة:** {bot_state.scan_stats['last_duration']}\n"
                       f"- **العملات المفحوصة:** {bot_state.scan_stats['markets_scanned']}\n\n"
                       f"------------------------------------\n"
                       f"- **إجمالي الإشارات المكتشفة:** {len(signals_found)}\n"
                       f"- **✅ صفقات جديدة بدأت:** {new_trades}\n"
                       f"- **⚠️ أخطاء في التحليل:** {bot_state.scan_stats['failures']}")
        await bot.send_message(TELEGRAM_CHAT_ID, scan_summary, parse_mode=ParseMode.MARKDOWN)

# =======================================================================================
# --- 🌐 ساعي البريد: مدير WebSocket الخاص والمؤمن 🌐 ---
# =======================================================================================
async def handle_filled_buy_order(order_data):
    symbol = order_data['instId'].replace('-', '/')
    order_id = order_data['ordId']
    filled_qty = float(order_data.get('fillSz', 0))
    avg_price = float(order_data.get('avgPx', 0))

    if filled_qty == 0 or avg_price == 0:
        logger.warning(f"[Postman] Ignoring fill event for {symbol} with zero quantity/price.")
        return

    logger.info(f"📬 [Postman] Received fill event for order {order_id} ({symbol}). Preparing protection...")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM trades WHERE order_id = ? AND status = 'pending_protection'", (order_id,)) as cursor:
                trade = await cursor.fetchone()

            if not trade:
                logger.warning(f"[Postman] No matching 'pending_protection' trade for order {order_id}. Might be already protected or a manual trade.")
                return

            trade = dict(trade)
            original_risk = trade['entry_price'] - trade['stop_loss']
            final_tp = avg_price + (original_risk * bot_state.settings['risk_reward_ratio'])
            final_sl = avg_price - original_risk
            
            oco_params = {
                'instId': bot_state.exchange.market_id(symbol), 'tdMode': 'cash', 'side': 'sell', 'ordType': 'oco',
                'sz': bot_state.exchange.amount_to_precision(symbol, filled_qty),
                'tpTriggerPx': bot_state.exchange.price_to_precision(symbol, final_tp), 'tpOrdPx': '-1',
                'slTriggerPx': bot_state.exchange.price_to_precision(symbol, final_sl), 'slOrdPx': '-1'
            }
            
            for attempt in range(3):
                oco_receipt = await bot_state.exchange.private_post_trade_order_algo(oco_params)
                if oco_receipt and oco_receipt.get('data') and oco_receipt['data'][0].get('sCode') == '0':
                    algo_id = oco_receipt['data'][0]['algoId']
                    logger.info(f"✅ [Postman] Successfully placed OCO protection for trade #{trade['id']}. Algo ID: {algo_id}")
                    
                    await conn.execute("""
                        UPDATE trades 
                        SET status = 'active', entry_price = ?, quantity = ?, entry_value_usdt = ?, 
                            take_profit = ?, stop_loss = ?, algo_id = ?, highest_price = ?
                        WHERE id = ?
                    """, (avg_price, filled_qty, avg_price * filled_qty, final_tp, final_sl, algo_id, avg_price, trade['id']))
                    await conn.commit()

                    tp_percent = (final_tp / avg_price - 1) * 100
                    sl_percent = (1 - final_sl / avg_price) * 100
                    success_msg = (f"**✅🛡️ صفقة مصفحة | {symbol} (ID: {trade['id']})**\n"
                                   f"🔍 **الاستراتيجية:** {trade['reason']}\n\n"
                                   f"📈 **الشراء:** `{avg_price:,.4f}`\n"
                                   f"🎯 **الهدف:** `{final_tp:,.4f}` (+{tp_percent:.2f}%)\n"
                                   f"🛑 **الوقف:** `{final_sl:,.4f}` (-{sl_percent:.2f}%)\n\n"
                                   f"***تم تأمين الصفقة بنجاح عبر ساعي البريد.***")
                    await bot_state.application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=success_msg, parse_mode=ParseMode.MARKDOWN)
                    return
                else:
                    logger.warning(f"[Postman] OCO placement attempt {attempt + 1} failed. Retrying...")
                    await asyncio.sleep(2)
            
            raise Exception(f"All Postman attempts to place OCO for trade #{trade['id']} failed.")
            
    except Exception as e:
        logger.critical(f"🔥 [Postman] CRITICAL FAILURE while protecting trade for order {order_id}: {e}", exc_info=True)
        error_message = (f"**🔥🔥🔥 فشل حرج لساعي البريد - {symbol}**\n\n"
                         f"🚨 **خطر!** تم تنفيذ الشراء ولكن **فشل وضع الحماية تلقائيًا**.\n"
                         f"**معرف الأمر:** `{order_id}`\n\n"
                         f"**❗️ تدخل يدوي فوري ضروري لوضع وقف الخسارة!**")
        await bot_state.application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=error_message, parse_mode=ParseMode.MARKDOWN)


class WebSocketManager:
    def __init__(self, bot_state):
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/private"
        self.bot_state = bot_state

    def _get_auth_args(self):
        timestamp = str(time.time())
        message = timestamp + 'GET' + '/users/self/verify'
        mac = hmac.new(bytes(OKX_API_SECRET, encoding='utf8'), bytes(message, encoding='utf8'), digestmod='sha256')
        sign = base64.b64encode(mac.digest()).decode()
        return [{
            "apiKey": OKX_API_KEY,
            "passphrase": OKX_API_PASSPHRASE,
            "timestamp": timestamp,
            "sign": sign,
        }]

    async def _message_handler(self, message):
        if message == 'ping':
            await self.websocket.send('pong')
            return
        
        data = json.loads(message)
        
        if data.get('arg', {}).get('channel') == 'orders':
            for order_data in data.get('data', []):
                if order_data.get('state') == 'filled' and order_data.get('side') == 'buy':
                    asyncio.create_task(handle_filled_buy_order(order_data))

    async def run(self):
        while True:
            try:
                # --- [FIXED] Added ping/pong heartbeat to ensure connection stability ---
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as websocket:
                    self.websocket = websocket
                    logger.info("✅ [WS-Private] Connected. Authenticating...")
                    
                    await websocket.send(json.dumps({"op": "login", "args": self._get_auth_args()}))
                    
                    login_response = json.loads(await websocket.recv())
                    if login_response.get('event') == 'login' and login_response.get('code') == '0':
                        logger.info("🔐 [WS-Private] Authenticated. Subscribing to orders channel...")
                        await websocket.send(json.dumps({
                            "op": "subscribe", 
                            "args": [{"channel": "orders", "instType": "SPOT"}]
                        }))
                    else:
                        logger.error(f"🔥 [WS-Private] Authentication failed: {login_response}")
                        continue

                    async for message in websocket:
                        await self._message_handler(message)
            except Exception as e:
                logger.error(f"🔥 [WS-Private] Unhandled exception: {e}", exc_info=True)
            
            logger.warning("⚠️ [WS-Private] Disconnected. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

# =======================================================================================
# --- 🛡️ المراقب الذكي (الحارس) 🛡️ ---
# =======================================================================================
async def track_open_trades(context: ContextTypes.DEFAULT_TYPE):
    await ensure_libraries_loaded()
    settings = bot_state.settings
    bot = context.bot
    
    pending_trades = []
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM trades WHERE status = 'pending_protection'") as cursor:
                pending_trades = [dict(row) for row in await cursor.fetchall()]
    except Exception as e:
        logger.error(f"Guardian: DB error: {e}")
        return

    # Guardian Protocol: Check for stuck trades
    for trade in pending_trades:
        trade_timestamp = datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S')
        if datetime.now() - trade_timestamp > timedelta(minutes=2):
            logger.critical(f"🛡️ GUARDIAN ALERT: Trade #{trade['id']} for {trade['symbol']} is stuck in 'pending_protection'!")
            alert_msg = (f"**🔥🔥🔥 تنبيه من الحارس**\n\n"
                         f"**صفقة:** `#{trade['id']} {trade['symbol']}`\n"
                         f"**الحالة:** عالقة في انتظار الحماية لأكثر من دقيقتين.\n\n"
                         f"**قد يكون هناك مشكلة في اتصال WebSocket. يرجى التحقق من الصفقة يدويًا فورًا!**")
            await bot.send_message(TELEGRAM_CHAT_ID, alert_msg, parse_mode=ParseMode.MARKDOWN)
    
    # Trailing Stop Loss Logic can be added here in the future
    # For now, the Guardian's main job is to watch for stuck trades.


# =======================================================================================
# --- 📱 واجهة التحكم عبر تليجرام 📱 ---
# =======================================================================================
# All Telegram handler functions from v8.0 go here
# This includes: start_command, show_dashboard_command, show_settings_menu,
# universal_text_handler, show_scanners_menu, show_presets_menu, 
# show_parameters_menu, and button_callback_handler.
# They are omitted here for brevity but should be pasted from the v8.0 code.

# =======================================================================================
# --- 🚀 نقطة انطلاق البوت 🚀 ---
# =======================================================================================
async def main():
    logger.info("--- Bot process starting ---")
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.critical("FATAL: One or more environment variables are not set. Exiting.")
        return
        
    load_settings()
    await init_database()
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    bot_state.application = app

    await ensure_libraries_loaded()
    
    # --- [FIXED] Use aws.okx.com as the hostname for better connectivity ---
    bot_state.exchange = ccxt.okx({
        'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 
        'password': OKX_API_PASSPHRASE, 'enableRateLimit': True, 
        'options': {
            'defaultType': 'spot',
            'hostname': 'aws.okx.com'
        }
    })
    
    ws_manager = WebSocketManager(bot_state)
    ws_task = asyncio.create_task(ws_manager.run())
    logger.info("🚀 [Postman] Postman scheduled to run in the background.")

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    app.add_handler(CallbackQueryHandler(button_callback_handler))

    scan_interval = bot_state.settings.get("scan_interval_seconds", 900)
    track_interval = bot_state.settings.get("track_interval_seconds", 60)
    app.job_queue.run_repeating(perform_scan, interval=scan_interval, first=10, name="perform_scan")
    app.job_queue.run_repeating(track_open_trades, interval=track_interval, first=30, name="track_trades")
    
    try:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="*🚀 بوت The Phoenix v8.1 (Network Hardened) بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)
        async with app:
            await app.start()
            await app.updater.start_polling()
            logger.info("Bot is now running and polling for updates...")
            await asyncio.gather(ws_task) 
    except Exception as e:
        logger.critical(f"An unhandled error occurred in main loop: {e}", exc_info=True)
        try:
            await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🔥 خطأ فادح أدى إلى توقف البوت: {e}")
        except Exception as tg_e:
            logger.error(f"Could not send final error message to Telegram: {tg_e}")
    finally:
        if 'ws_task' in locals() and not ws_task.done(): ws_task.cancel()
        if 'app' in locals() and app.updater and app.updater._running: await app.updater.stop()
        if 'app' in locals() and app.running: await app.stop()
        if bot_state.exchange: await bot_state.exchange.close(); logger.info("CCXT exchange connection closed.")
        logger.info("Bot has been shut down.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Failed to start bot due to an error in initial setup: {e}", exc_info=True)
