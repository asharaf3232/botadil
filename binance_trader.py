# -*- coding: utf-8 -*-

# --- المكتبات المطلوبة --- #
import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import asyncio
import os
import logging
import json
import time
import sqlite3
from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import deque
from pathlib import Path

# [UPGRADE] المكتبات الجديدة لتحليل الأخبار
import feedparser
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("Library 'nltk' not found. Sentiment analysis will be disabled.")

import requests
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Library 'scipy' not found. RSI Divergence strategy will be disabled.")


# --- الإعدادات الأساسية --- #
# !!! هام: قم بتعيين هذه المتغيرات في بيئة التشغيل الخاصة بك
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')
TELEGRAM_SIGNAL_CHANNEL_ID = os.getenv('TELEGRAM_SIGNAL_CHANNEL_ID', TELEGRAM_CHAT_ID)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_AV_KEY_HERE')

# --- [تداول حقيقي] إعدادات مفاتيح Binance API --- #
# !!! هام جدًا: لا تكتب مفاتيحك هنا مباشرة. استخدم متغيرات البيئة.
# قم بتعيينها في نظامك كالتالي:
# export BINANCE_API_KEY="your_api_key"
# export BINANCE_API_SECRET="your_secret_key"
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_BINANCE_SECRET_KEY')


if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or TELEGRAM_CHAT_ID == 'YOUR_CHAT_ID_HERE':
    print("FATAL ERROR: Please set your Telegram Token and Chat ID.")
    exit()


# --- إعدادات البوت --- #
# [تعديل] تم حصر العمل على منصة Binance فقط
EXCHANGE_TO_USE = 'binance'
TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
SCAN_INTERVAL_SECONDS = 900
TRACK_INTERVAL_SECONDS = 120 # لتتبع الصفقات الوهمية والحقيقية

APP_ROOT = '.'
# [تعديل] تم تحديث أسماء الملفات لتعكس الإصدار الجديد
DB_FILE = os.path.join(APP_ROOT, 'binance_trader.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'binance_trader_settings.json')


EGYPT_TZ = ZoneInfo("Africa/Cairo")

# --- إعداد مسجل الأحداث (Logger) --- #
LOG_FILE = os.path.join(APP_ROOT, 'binance_trader.log')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'a'), logging.StreamHandler()])
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('apscheduler').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logger = logging.getLogger("TradingBot")

# --- Preset Configurations ---
PRESET_PRO = {
  "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "max_spread_percent": 0.45, "rvol_period": 18, "min_rvol": 1.5},
  "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.85},
  "ema_trend_filter": {"enabled": True, "ema_period": 200},
  "min_tp_sl_filter": {"min_tp_percent": 1.1, "min_sl_percent": 0.6}
}
# (يمكن إضافة باقي الأنماط هنا بنفس الطريقة)
PRESETS = {"PRO": PRESET_PRO}

STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي", "breakout_squeeze_pro": "اختراق انضغاطي",
    "rsi_divergence": "دايفرجنس RSI", "supertrend_pullback": "انعكاس سوبرترند"
}

# --- Constants for Interactive Settings menu ---
EDITABLE_PARAMS = {
    "إعدادات عامة": ["max_concurrent_trades", "top_n_symbols_by_volume", "min_signal_strength"],
    "إعدادات المخاطر": ["REAL_TRADING_ENABLED", "trade_size_usdt", "atr_sl_multiplier", "risk_reward_ratio"],
    "الفلاتر والاتجاه": ["market_regime_filter_enabled", "use_master_trend_filter", "fear_and_greed_filter_enabled", "trailing_sl_enabled"]
}
PARAM_DISPLAY_NAMES = {
    "REAL_TRADING_ENABLED": "وضع التداول الحقيقي",
    "trade_size_usdt": "حجم الصفقة (USDT)",
    "max_concurrent_trades": "أقصى عدد للصفقات", "top_n_symbols_by_volume": "عدد العملات للفحص",
    "min_signal_strength": "أدنى قوة للإشارة", "atr_sl_multiplier": "مضاعف وقف الخسارة (ATR)",
    "risk_reward_ratio": "نسبة المخاطرة/العائد", "market_regime_filter_enabled": "فلتر وضع السوق (فني)",
    "use_master_trend_filter": "فلتر الاتجاه العام (BTC)", "trailing_sl_enabled": "تفعيل الوقف المتحرك",
    "fear_and_greed_filter_enabled": "فلتر الخوف والطمع"
}

# --- Global Bot State ---
bot_data = {
    "exchange": None, # [تعديل] كائن واحد للمنصة
    "last_signal_time": {}, "settings": {},
    "status_snapshot": {
        "last_scan_time": "N/A", "markets_found": 0, "signals_found": 0,
        "active_trades_count": 0, "scan_in_progress": False, "btc_market_mood": "غير محدد",
        "trading_mode": "وهمي 📝"
    },
    "scan_history": deque(maxlen=10)
}
scan_lock = asyncio.Lock()

# --- Settings Management ---
DEFAULT_SETTINGS = {
    "REAL_TRADING_ENABLED": False, "trade_size_usdt": 20.0,
    "virtual_portfolio_balance_usdt": 1000.0, "virtual_trade_size_percentage": 5.0,
    "max_concurrent_trades": 5, "top_n_symbols_by_volume": 250,
    "market_regime_filter_enabled": True, "fundamental_analysis_enabled": True,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "rsi_divergence", "supertrend_pullback"],
    "use_master_trend_filter": True, "master_trend_filter_ma_period": 50, "master_adx_filter_level": 22,
    "fear_and_greed_filter_enabled": True, "fear_and_greed_threshold": 30,
    "use_dynamic_risk_management": True, "atr_period": 14, "atr_sl_multiplier": 2.0, "risk_reward_ratio": 1.5,
    "trailing_sl_enabled": True, "trailing_sl_activate_percent": 2.0, "trailing_sl_percent": 1.5,
    "momentum_breakout": {"rsi_max_level": 68},
    "breakout_squeeze_pro": {"bbands_period": 20, "keltner_period": 20, "keltner_atr_multiplier": 1.5},
    "rsi_divergence": {"rsi_period": 14, "lookback_period": 35},
    "supertrend_pullback": {"atr_period": 10, "atr_multiplier": 3.0},
    "liquidity_filters": {"min_quote_volume_24h_usd": 1_000_000, "max_spread_percent": 0.5, "min_rvol": 1.5},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.8},
    "ema_trend_filter": {"enabled": True, "ema_period": 200},
    "min_tp_sl_filter": {"min_tp_percent": 1.0, "min_sl_percent": 0.5},
    "min_signal_strength": 1, "active_preset_name": "PRO",
}

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f: bot_data["settings"] = json.load(f)
            # تحديث الإعدادات الافتراضية إذا كانت هناك قيم جديدة
            for key, value in DEFAULT_SETTINGS.items():
                if key not in bot_data["settings"]: bot_data["settings"][key] = value
            save_settings()
        else:
            bot_data["settings"] = DEFAULT_SETTINGS.copy(); save_settings()
        
        mode = "حقيقي 🟢" if bot_data["settings"].get("REAL_TRADING_ENABLED") else "وهمي 📝"
        bot_data['status_snapshot']['trading_mode'] = mode
        logger.info(f"Settings loaded successfully. Current Trading Mode: {mode}")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}"); bot_data["settings"] = DEFAULT_SETTINGS.copy()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(bot_data["settings"], f, indent=4)
        logger.info(f"Settings saved successfully")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

# --- Database Management ---
def init_database():
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT,
                entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL,
                entry_value_usdt REAL, status TEXT, exit_price REAL, closed_at TEXT,
                pnl_usdt REAL, trailing_sl_active BOOLEAN, highest_price REAL,
                reason TEXT, is_real_trade BOOLEAN, entry_order_id TEXT, sl_tp_order_id TEXT
            )
        ''')
        conn.commit(); conn.close()
        logger.info(f"Database initialized successfully at: {DB_FILE}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def log_trade_to_db(signal, is_real=False, order_ids=None):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor()
        ids = order_ids or {}
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, entry_price, take_profit, stop_loss,
            quantity, entry_value_usdt, status, highest_price, reason, is_real_trade,
            entry_order_id, sl_tp_order_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), signal['symbol'],
            signal['entry_price'], signal['take_profit'], signal['stop_loss'],
            signal['quantity'], signal['entry_value_usdt'], 'نشطة', signal['entry_price'],
            signal['reason'], is_real, ids.get('entry_order_id'), ids.get('sl_tp_order_id')
        ))
        trade_id = cursor.lastrowid
        conn.commit(); conn.close()
        return trade_id
    except Exception as e:
        logger.error(f"Failed to log trade to DB: {e}"); return None

# --- [تداول حقيقي] وظائف تنفيذ الصفقات --- #
async def execute_real_trade_on_binance(signal):
    settings = bot_data['settings']; exchange = bot_data['exchange']; symbol = signal['symbol']
    try:
        balance = await exchange.fetch_balance(); usdt_balance = balance['total'].get('USDT', 0)
        if usdt_balance < settings['trade_size_usdt']:
            return {"success": False, "message": f"رصيد USDT غير كافٍ. المطلوب: {settings['trade_size_usdt']}, المتاح: {usdt_balance:.2f}"}

        await exchange.load_markets(True) # تحديث معلومات السوق
        market = exchange.market(symbol)
        
        amount_to_buy = settings['trade_size_usdt'] / signal['entry_price']
        quantity = exchange.amount_to_precision(symbol, amount_to_buy)

        logger.info(f"REAL TRADE: Placing MARKET BUY for {quantity} {symbol}")
        buy_order = await exchange.create_market_buy_order(symbol, quantity)
        await asyncio.sleep(2) # انتظار للتأكد من التنفيذ
        
        filled_order = await exchange.fetch_order(buy_order['id'], symbol)
        if not filled_order or filled_order['status'] != 'closed':
            return {"success": False, "message": "فشل التحقق من تنفيذ أمر الشراء."}
        
        actual_quantity = filled_order['filled']; actual_entry_price = filled_order['average']
        signal.update({'quantity': actual_quantity, 'entry_price': actual_entry_price, 'entry_value_usdt': filled_order['cost']})
        
        tp_price = exchange.price_to_precision(symbol, signal['take_profit'])
        sl_price = exchange.price_to_precision(symbol, signal['stop_loss'])
        
        logger.info(f"REAL TRADE: Placing OCO SELL for {actual_quantity} {symbol} -> TP: {tp_price}, SL: {sl_price}")
        oco_order = await exchange.create_order(symbol, 'oco', 'sell', actual_quantity, price=tp_price, stopPrice=sl_price, params={'stopLimitPrice': sl_price})
        
        return {"success": True, "message": "تم تنفيذ الصفقة بنجاح.", "order_ids": {"entry_order_id": buy_order['id'], "sl_tp_order_id": oco_order.get('orderListId')}, "filled_signal": signal}
    except Exception as e:
        logger.critical(f"CRITICAL REAL TRADE ERROR for {symbol}: {e}", exc_info=True)
        return {"success": False, "message": f"خطأ من المنصة: {e}"}

# --- Advanced Scanners ---
# (تم تبسيطها قليلًا لعدم تكرار حساب المؤشرات)
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

def analyze_momentum_breakout(df, params, rvol):
    macd_col, macds_col, bbu_col, rsi_col = find_col(df.columns, "MACD_"), find_col(df.columns, "MACDs_"), find_col(df.columns, "BBU_"), find_col(df.columns, "RSI_")
    last, prev = df.iloc[-2], df.iloc[-3]
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and last[rsi_col] < params['rsi_max_level']):
        return {"reason": "momentum_breakout"}
    return None
def analyze_breakout_squeeze_pro(df, params, rvol):
    bbu_col, bbl_col, kcu_col, kcl_col = find_col(df.columns, "BBU_"), find_col(df.columns, "BBL_"), find_col(df.columns, "KCUe_"), find_col(df.columns, "KCLEe_")
    last, prev = df.iloc[-2], df.iloc[-3]
    is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]
    if is_in_squeeze and last['close'] > last[bbu_col] and df['OBV'].iloc[-2] > df['OBV'].iloc[-3]:
        return {"reason": "breakout_squeeze_pro"}
    return None
def analyze_rsi_divergence(df, params, rvol):
    if not SCIPY_AVAILABLE: return None
    rsi_col = find_col(df.columns, f"RSI_")
    subset = df.iloc[-params['lookback_period']:].copy()
    price_troughs_idx, _ = find_peaks(-subset['low'], distance=5)
    rsi_troughs_idx, _ = find_peaks(-subset[rsi_col], distance=5)
    if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
        p_low1_idx, p_low2_idx = price_troughs_idx[-2], price_troughs_idx[-1]
        r_low1_idx, r_low2_idx = rsi_troughs_idx[-2], rsi_troughs_idx[-1]
        is_divergence = (subset.iloc[p_low2_idx]['low'] < subset.iloc[p_low1_idx]['low'] and subset.iloc[r_low2_idx][rsi_col] > subset.iloc[r_low1_idx][rsi_col])
        if is_divergence and df.iloc[-2]['close'] > subset.iloc[p_low2_idx:]['high'].max():
            return {"reason": "rsi_divergence"}
    return None
def analyze_supertrend_pullback(df, params, rvol):
    st_dir_col, ema_col = find_col(df.columns, "SUPERTd_"), find_col(df.columns, 'EMA_')
    last, prev = df.iloc[-2], df.iloc[-3]
    if prev[st_dir_col] == -1 and last[st_dir_col] == 1 and last['close'] > last[ema_col] and last['close'] > df['high'].iloc[-12:-2].max():
        return {"reason": "supertrend_pullback"}
    return None

SCANNERS = {"momentum_breakout": analyze_momentum_breakout, "breakout_squeeze_pro": analyze_breakout_squeeze_pro, "rsi_divergence": analyze_rsi_divergence, "supertrend_pullback": analyze_supertrend_pullback}

# --- Core Bot Functions ---
async def initialize_exchange():
    real_trading = bot_data["settings"].get("REAL_TRADING_ENABLED", False)
    config = {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
    if real_trading:
        if BINANCE_API_KEY != 'YOUR_BINANCE_API_KEY' and BINANCE_API_SECRET != 'YOUR_BINANCE_SECRET_KEY':
            config['apiKey'] = BINANCE_API_KEY; config['secret'] = BINANCE_API_SECRET
            logger.info("Binance Real Trading mode: API keys loaded.")
        else:
            logger.error("Real Trading is ENABLED but API keys are NOT SET. Reverting to paper mode.")
            bot_data["settings"]["REAL_TRADING_ENABLED"] = False; save_settings()
            bot_data['status_snapshot']['trading_mode'] = "وهمي 📝"
    
    exchange = getattr(ccxt, EXCHANGE_TO_USE)(config)
    try:
        await exchange.load_markets(); bot_data["exchange"] = exchange
        logger.info(f"Connected to {EXCHANGE_TO_USE.capitalize()} successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to {EXCHANGE_TO_USE.capitalize()}: {e}")
        await exchange.close(); bot_data["exchange"] = None
        return False

async def reinitialize_exchange():
    logger.info("Re-initializing Binance connection due to trading mode change...")
    if bot_data["exchange"]:
        try: await bot_data["exchange"].close(); logger.info("Existing connection closed.")
        except Exception: pass
    return await initialize_exchange()

async def worker(queue, results_list, settings, failure_counter):
    exchange = bot_data["exchange"]
    while not queue.empty():
        symbol = await queue.get()
        try:
            liq_filters, vol_filters, ema_filters, min_tp_sl = settings['liquidity_filters'], settings['volatility_filters'], settings['ema_trend_filter'], settings['min_tp_sl_filter']
            
            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            if len(ohlcv) < ema_filters['ema_period']: continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
            
            df['volume_sma'] = ta.sma(df['volume'], length=20); rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
            if rvol < liq_filters['min_rvol']: continue
            
            df.ta.atr(length=vol_filters['atr_period_for_filter'], append=True); last_close = df['close'].iloc[-2]
            atr_percent = (df[find_col(df.columns, "ATRr_")].iloc[-2] / last_close) * 100
            if atr_percent < vol_filters['min_atr_percent']: continue
            
            df.ta.ema(length=ema_filters['ema_period'], append=True)
            if ema_filters['enabled'] and last_close < df[find_col(df.columns, "EMA_")].iloc[-2]: continue
            
            # Pre-calculate indicators for all strategies
            df.ta.bbands(length=20, append=True); df.ta.kc(length=20, append=True)
            df.ta.supertrend(length=10, multiplier=3, append=True); df.ta.adx(append=True)
            df.ta.rsi(length=14, append=True); df.ta.macd(append=True); df.ta.vwap(append=True); df.ta.obv(append=True)

            confirmed_reasons = [result['reason'] for name in settings['active_scanners'] if (result := SCANNERS[name](df.copy(), settings.get(name, {}), rvol))]
            
            if confirmed_reasons and len(confirmed_reasons) >= settings.get("min_signal_strength", 1):
                entry_price = last_close
                df.ta.atr(length=settings['atr_period'], append=True); current_atr = df.iloc[-2].get(find_col(df.columns, f"ATRr_"), 0)
                risk_per_unit = current_atr * settings['atr_sl_multiplier']
                stop_loss, take_profit = entry_price - risk_per_unit, entry_price + (risk_per_unit * settings['risk_reward_ratio'])
                
                if ((take_profit / entry_price - 1) * 100) >= min_tp_sl['min_tp_percent'] and ((1 - stop_loss / entry_price) * 100) >= min_tp_sl['min_sl_percent']:
                    results_list.append({"symbol": symbol, "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "timestamp": df.index[-2], "reason": ' + '.join(confirmed_reasons), "strength": len(confirmed_reasons)})
        except Exception as e:
            if 'RateLimitExceeded' in str(e): await asyncio.sleep(10)
            else: failure_counter[0] += 1
        finally:
            queue.task_done()

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        if bot_data['status_snapshot']['scan_in_progress']: return
        settings = bot_data["settings"]
        is_real_trading = settings.get("REAL_TRADING_ENABLED", False)

        is_market_ok, btc_reason = await check_market_regime()
        bot_data['status_snapshot']['btc_market_mood'] = "إيجابي ✅" if is_market_ok else "سلبي ❌"
        if settings.get('market_regime_filter_enabled', True) and not is_market_ok:
            logger.info(f"Skipping scan: {btc_reason}"); return

        status = bot_data['status_snapshot']; exchange = bot_data['exchange']
        status.update({"scan_in_progress": True, "last_scan_time": datetime.now(EGYPT_TZ).strftime('%H:%M:%S'), "signals_found": 0})
        
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة' AND is_real_trade = ?", (is_real_trading,))
            active_trades_count = cursor.fetchone()[0]; conn.close()
        except Exception as e:
            logger.error(f"DB Error getting active trades: {e}"); active_trades_count = settings["max_concurrent_trades"]

        # [تعديل] جلب الأسواق من Binance فقط
        try:
            all_tickers = await exchange.fetch_tickers()
            min_volume = settings['liquidity_filters']['min_quote_volume_24h_usd']
            top_markets = [s for s, t in all_tickers.items() if s.endswith('/USDT') and t.get('quoteVolume', 0) > min_volume]
            top_markets = sorted(top_markets, key=lambda s: all_tickers[s]['quoteVolume'], reverse=True)[:settings['top_n_symbols_by_volume']]
            status['markets_found'] = len(top_markets)
        except Exception as e:
            logger.error(f"Failed to fetch markets from Binance: {e}"); status['scan_in_progress'] = False; return

        queue = asyncio.Queue(); [await queue.put(market) for market in top_markets]
        signals, failure_counter = [], [0]
        worker_tasks = [asyncio.create_task(worker(queue, signals, settings, failure_counter)) for _ in range(10)]
        await queue.join(); [task.cancel() for task in worker_tasks]
        
        signals.sort(key=lambda s: s.get('strength', 0), reverse=True)
        new_trades = 0
        
        for signal in signals:
            if active_trades_count >= settings.get("max_concurrent_trades", 5): break
            if time.time() - bot_data['last_signal_time'].get(signal['symbol'], 0) <= (SCAN_INTERVAL_SECONDS * 3): continue
            
            if is_real_trading:
                trade_result = await execute_real_trade_on_binance(signal)
                if trade_result["success"]:
                    filled_signal = trade_result["filled_signal"]
                    if trade_id := log_trade_to_db(filled_signal, is_real=True, order_ids=trade_result["order_ids"]):
                        filled_signal['trade_id'] = trade_id
                        await send_telegram_message(context.bot, filled_signal, is_new=True)
                        active_trades_count += 1; new_trades += 1
                else:
                    await send_telegram_message(context.bot, {'custom_message': f"**❌ فشل تنفيذ صفقة حقيقية**\n- **العملة:** `{signal['symbol']}`\n- **السبب:** `{trade_result['message']}`"})
            else:
                trade_amount_usdt = settings["virtual_portfolio_balance_usdt"] * (settings["virtual_trade_size_percentage"] / 100)
                signal.update({'quantity': trade_amount_usdt / signal['entry_price'], 'entry_value_usdt': trade_amount_usdt})
                if trade_id := log_trade_to_db(signal, is_real=False):
                    signal['trade_id'] = trade_id; await send_telegram_message(context.bot, signal, is_new=True)
                    active_trades_count += 1; new_trades += 1
            
            bot_data['last_signal_time'][signal['symbol']] = time.time()
        
        logger.info(f"Scan complete. Found: {len(signals)}, Entered: {new_trades}, Failures: {failure_counter[0]}.")
        status.update({'signals_found': len(signals), 'scan_in_progress': False})

async def send_telegram_message(bot, signal_data, is_new=False):
    message, keyboard = "", None
    def format_price(price): return f"{price:,.8f}" if price < 0.01 else f"{price:,.4f}"
    
    if 'custom_message' in signal_data: message = signal_data['custom_message']
    elif is_new:
        is_real = bot_data["settings"].get("REAL_TRADING_ENABLED", False)
        mode_icon = "🟢" if is_real else "📝"; mode_text = "صفقة حقيقية" if is_real else "توصية وهمية"
        title = f"**{mode_icon} {mode_text} | {signal_data['symbol']}**"
        entry, tp, sl = signal_data['entry_price'], signal_data['take_profit'], signal_data['stop_loss']
        tp_percent, sl_percent = ((tp - entry) / entry * 100), ((entry - sl) / entry * 100)
        reasons_ar = ' + '.join([STRATEGY_NAMES_AR.get(r, r) for r in signal_data['reason'].split(' + ')])
        message = (f"{title}\n"
                   f"⭐ **قوة الإشارة:** {'⭐' * signal_data.get('strength', 1)}\n"
                   f"🔍 **الاستراتيجية:** {reasons_ar}\n\n"
                   f"📈 **الدخول:** `{format_price(entry)}`\n"
                   f"🎯 **الهدف:** `{format_price(tp)}` (+{tp_percent:.2f}%)\n"
                   f"🛑 **الوقف:** `{format_price(sl)}` (-{sl_percent:.2f}%)\n"
                   f"*للمتابعة: /check {signal_data['trade_id']}*")
        if is_real: message += "\n\n**تنبيه: تم وضع الأوامر الفعلية في حسابك.**"
    
    if not message: return
    try: await bot.send_message(chat_id=TELEGRAM_SIGNAL_CHANNEL_ID, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
    except Exception as e: logger.error(f"Failed to send Telegram message: {e}")

async def track_trades_job(context: ContextTypes.DEFAULT_TYPE):
    settings = bot_data['settings']
    is_real_trading = settings.get("REAL_TRADING_ENABLED", False)

    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'نشطة' AND is_real_trade = ?", (is_real_trading,))
        active_trades = [dict(row) for row in cursor.fetchall()]; conn.close()
    except Exception as e: logger.error(f"DB error in track_trades_job: {e}"); return
    
    bot_data['status_snapshot']['active_trades_count'] = len(active_trades)
    if not active_trades: return
    
    if is_real_trading: await check_real_trades_status(context, active_trades)
    else: await check_paper_trades_status(context, active_trades)

async def check_paper_trades_status(context, active_trades):
    exchange = bot_data["exchange"]
    updates_to_db, portfolio_pnl = [], 0.0
    symbols_to_fetch = list(set([t['symbol'] for t in active_trades]))
    try: tickers = await exchange.fetch_tickers(symbols_to_fetch)
    except Exception: return

    for trade in active_trades:
        current_price = tickers.get(trade['symbol'], {}).get('last')
        if not current_price: continue

        highest_price = max(trade.get('highest_price', current_price), current_price)
        status, exit_price = None, None
        
        if current_price >= trade['take_profit']: status, exit_price = 'ناجحة', current_price
        elif current_price <= trade['stop_loss']: status, exit_price = 'فاشلة', current_price
        
        if status:
            pnl_usdt = (exit_price - trade['entry_price']) * trade['quantity']
            portfolio_pnl += pnl_usdt
            updates_to_db.append(("UPDATE trades SET status=?, exit_price=?, closed_at=CURRENT_TIMESTAMP, pnl_usdt=?, highest_price=? WHERE id=?", (status, exit_price, pnl_usdt, highest_price, trade['id'])))
            # (منطق إرسال رسالة الإغلاق يمكن إضافته هنا)

    if updates_to_db:
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor()
            for q, p in updates_to_db: cursor.execute(q, p)
            conn.commit(); conn.close()
        except Exception as e: logger.error(f"DB update failed in paper trade tracking: {e}")
    if portfolio_pnl != 0.0:
        bot_data['settings']['virtual_portfolio_balance_usdt'] += portfolio_pnl; save_settings()

async def check_real_trades_status(context, active_trades):
    exchange = bot_data["exchange"]
    for trade in active_trades:
        try:
            # التحقق مما إذا كان أمر OCO لا يزال مفتوحًا
            orders = await exchange.fetch_open_orders(trade['symbol'])
            oco_still_open = any(str(o.get('orderListId')) == str(trade['sl_tp_order_id']) for o in orders)
            
            if not oco_still_open:
                # إذا لم يكن مفتوحًا، فهذا يعني أنه تم تنفيذه أو إلغاؤه
                closed_orders = await exchange.fetch_closed_orders(trade['symbol'], limit=10)
                for order in closed_orders:
                    if str(order.get('orderListId')) == str(trade['sl_tp_order_id']) and order['status'] == 'closed':
                        status = 'ناجحة' if order['price'] >= trade['take_profit'] else 'فاشلة'
                        exit_price = order['average']
                        pnl_usdt = (exit_price - trade['entry_price']) * order['filled']
                        
                        conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor()
                        cursor.execute("UPDATE trades SET status=?, exit_price=?, closed_at=CURRENT_TIMESTAMP, pnl_usdt=? WHERE id=?", (status, exit_price, pnl_usdt, trade['id']))
                        conn.commit(); conn.close()
                        
                        logger.info(f"REAL trade #{trade['id']} for {trade['symbol']} detected as CLOSED. Status: {status}")
                        # (منطق إرسال رسالة الإغلاق يمكن إضافته هنا)
                        break
        except Exception as e:
            logger.error(f"Error checking real trade status for {trade['symbol']} (ID: {trade['id']}): {e}")

# --- Helper Functions & Telegram Commands ---
async def check_market_regime():
    try:
        ohlcv = await bot_data["exchange"].fetch_ohlcv('BTC/USDT', '4h', limit=55)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['sma50'] = ta.sma(df['close'], length=50)
        is_technically_bullish = df['close'].iloc[-1] > df['sma50'].iloc[-1]
    except Exception: is_technically_bullish = True
    
    if not is_technically_bullish: return False, "اتجاه BTC هابط (تحت متوسط 50 على 4 ساعات)."
    return True, "وضع السوق مناسب لصفقات الشراء."

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("أهلاً بك في بوت تداول Binance!", reply_markup=ReplyKeyboardMarkup([["Dashboard 🖥️", "⚙️ الإعدادات"]], resize_keyboard=True))

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trading_mode = bot_data['status_snapshot']['trading_mode']
    message_text = f"🖥️ *لوحة التحكم*\n\n**وضع التداول الحالي: {trading_mode}**"
    await (update.message or update.callback_query.message).reply_text(message_text, parse_mode=ParseMode.MARKDOWN)

async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard, settings = [], bot_data["settings"]
    for category, params in EDITABLE_PARAMS.items():
        keyboard.append([InlineKeyboardButton(f"--- {category} ---", callback_data="ignore")])
        for param_key in params:
            display_name = PARAM_DISPLAY_NAMES.get(param_key, param_key)
            current_value = settings.get(param_key)
            text = f"{display_name}: {'مُفعّل ✅' if current_value else 'مُعطّل ❌'}" if isinstance(current_value, bool) else f"{display_name}: {current_value}"
            keyboard.append([InlineKeyboardButton(text, callback_data=f"param_{param_key}")])
    message_text = "⚙️ *الإعدادات*\n\nاختر الإعداد الذي تريد تعديله:"
    await (update.message or update.callback_query.message).reply_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def check_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # (يمكن إضافة منطق هذا الأمر لاحقًا إذا لزم الأمر)
    await update.message.reply_text("ميزة متابعة الصفقة قيد التطوير.")

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data
    
    if data.startswith("param_"):
        param_key = data.split("_", 1)[1]
        current_value = bot_data["settings"].get(param_key)
        if isinstance(current_value, bool):
            new_value = not current_value
            if param_key == "REAL_TRADING_ENABLED":
                if new_value and (BINANCE_API_KEY == 'YOUR_BINANCE_API_KEY' or BINANCE_API_SECRET == 'YOUR_BINANCE_SECRET_KEY'):
                    await query.answer("⚠️ لا يمكن التفعيل! مفاتيح Binance API غير مهيأة.", show_alert=True)
                    return
                await query.answer(f"‼️ سيتم إعادة تشغيل الاتصال. قد يستغرق الأمر لحظات.", show_alert=True)
                bot_data["settings"][param_key] = new_value; save_settings()
                if await reinitialize_exchange():
                    bot_data['status_snapshot']['trading_mode'] = "حقيقي 🟢" if new_value else "وهمي 📝"
                    await query.message.reply_text(f"✅ تم تبديل وضع التداول إلى: {bot_data['status_snapshot']['trading_mode']}")
                else:
                    bot_data["settings"][param_key] = not new_value; save_settings() # Revert
                    bot_data['status_snapshot']['trading_mode'] = "وهمي 📝"
                    await query.message.reply_text("❌ فشل الاتصال بـ Binance. تم الإبقاء على الوضع الوهمي.")
            else:
                bot_data["settings"][param_key] = new_value; save_settings()
            
            try: await query.delete_message()
            except: pass
            await show_settings_menu(update, context)

# (تم تبسيط معالج النصوص لأنه لم تعد هناك حاجة للحوارات المعقدة)
async def main_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    handlers = {"Dashboard 🖥️": show_dashboard_command, "⚙️ الإعدادات": show_settings_menu}
    if handler := handlers.get(update.message.text): await handler(update, context)

async def post_init(application: Application):
    if NLTK_AVAILABLE:
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError: logger.info("Downloading NLTK data..."); nltk.download('vader_lexicon')
    
    if await initialize_exchange():
        job_queue = application.job_queue
        job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name='perform_scan')
        job_queue.run_repeating(track_trades_job, interval=TRACK_INTERVAL_SECONDS, first=20, name='track_trades')
        trading_mode = bot_data['status_snapshot']['trading_mode']
        await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🚀 *بوت تداول Binance جاهز للعمل!*\n\n**وضع التشغيل الحالي: {trading_mode}**", parse_mode=ParseMode.MARKDOWN)
    else:
        await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"❌ *فشل الاتصال بـ Binance!* لا يمكن تشغيل البوت. يرجى مراجعة سجل الأخطاء.")

async def post_shutdown(application: Application):
    if bot_data["exchange"]: await bot_data["exchange"].close(); logger.info("Binance connection closed.")

def main():
    print("🚀 Starting Binance Trading Bot...")
    load_settings(); init_database()
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).post_shutdown(post_shutdown).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("check", check_trade_command))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, main_text_handler))

    print("✅ Bot is now running and polling for updates...")
    application.run_polling()

if __name__ == '__main__':
    main()
