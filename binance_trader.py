# -*- coding: utf-8 -*-
# =======================================================================================
# --- 💣 بوت كاسحة الألغام (Minesweeper Bot) v5.0 (Architectural Refactor) 💣 ---
# =======================================================================================
# --- سجل التغييرات v5.0 ---
#
# 1. [إعادة هيكلة] تطبيق نمط المحول (Adapter Pattern) للتعامل مع المنصات.
#    - تم إنشاء كلاس أساسي `ExchangeAdapter` وكلاسات فرعية (`BinanceAdapter`, `KuCoinAdapter`).
#    - تم نقل منطق وضع أوامر الخروج والوقف المتحرك الخاص بكل منصة إلى المحول الخاص بها.
#    - النتيجة: الكود الرئيسي أصبح أنظف وأكثر قابلية للتوسع لإضافة منصات جديدة.
#
# 2. [إعادة هيكلة] تطبيق كلاس إدارة الحالة (State Management).
#    - تم تجميع كل متغيرات البوت العامة في كلاس واحد `BotState` بدلاً من القاموس `bot_data`.
#    - النتيجة: الكود أصبح أكثر تنظيماً وسهولة في التتبع.
#
# 3. [تحسين أداء] تطبيق التزامن في متابعة الصفقات.
#    - دالة `track_open_trades` تستخدم الآن `asyncio.gather` لمتابعة كل الصفقات في نفس الوقت.
#    - النتيجة: متابعة أسرع وأكثر كفاءة، خاصة مع عدد كبير من الصفقات المفتوحة.
#
# 4. [إصلاح نهائي] تطبيق الحل الصحيح لمنصة KuCoin.
#    - المحول الخاص بـ KuCoin (`KuCoinAdapter`) يقوم الآن بإنشاء أمرين منفصلين (TP و SL).
#    - تم إصلاح منطق الوقف المتحرك لـ KuCoin ليعمل بنظام "الإلغاء ثم الاستبدال" للأمرين معاً.
#
# =======================================================================================

# --- المكتبات المطلوبة ---
import ccxt.async_support as ccxt_async
import ccxt
import pandas as pd
import pandas_ta as ta
import asyncio
import os
import logging
import json
import re
import time
import sqlite3
from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import deque, Counter, defaultdict
import numpy as np
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
from telegram.request import HTTPXRequest
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest, RetryAfter, TimedOut

# --- الإعدادات الأساسية ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')
TELEGRAM_SIGNAL_CHANNEL_ID = os.getenv('TELEGRAM_SIGNAL_CHANNEL_ID', TELEGRAM_CHAT_ID)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_AV_KEY_HERE')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_BINANCE_API_SECRET')
KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', 'YOUR_KUCOIN_API_KEY')
KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET', 'YOUR_KUCOIN_API_SECRET')
KUCOIN_API_PASSPHRASE = os.getenv('KUCOIN_API_PASSPHRASE', 'YOUR_KUCOIN_API_PASSPHRASE')

# --- إعدادات البوت ---
EXCHANGES_TO_SCAN = ['binance', 'okx', 'bybit', 'kucoin', 'gate', 'mexc']
TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
SCAN_INTERVAL_SECONDS = 900
TRACK_INTERVAL_SECONDS = 45 # تم تقليل المدة للاستجابة السريعة مع التزامن

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v5.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'minesweeper_settings_v5.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")

# --- إعداد مسجل الأحداث (Logger) ---
LOG_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v5.log')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'a'), logging.StreamHandler()])
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger("MinesweeperBot_v5")


# =======================================================================================
# --- 🚀 [v5.0] إعادة الهيكلة: إدارة الحالة والمنصات 🚀 ---
# =======================================================================================

class BotState:
    """كلاس مركزي لإدارة كل حالة البوت لزيادة التنظيم."""
    def __init__(self):
        self.exchanges = {}  # Clients مصادق عليهم
        self.public_exchanges = {}  # Clients عامة
        self.last_signal_time = {}
        self.settings = {}
        self.status_snapshot = {
            "last_scan_start_time": None, "last_scan_end_time": None,
            "markets_found": 0, "signals_found": 0, "active_trades_count": 0,
            "scan_in_progress": False, "btc_market_mood": "غير محدد"
        }
        self.scan_history = deque(maxlen=10)

bot_state = BotState()

class ExchangeAdapter:
    """كلاس أساسي مجرد لنمط المحول. يحدد الواجهة المشتركة لجميع المنصات."""
    def __init__(self, exchange_client):
        self.exchange = exchange_client

    async def place_exit_orders(self, signal, verified_quantity):
        raise NotImplementedError("يجب على المحول الفرعي تنفيذ place_exit_orders")

    async def update_trailing_stop_loss(self, trade, new_sl):
        raise NotImplementedError("يجب على المحول الفرعي تنفيذ update_trailing_stop_loss")

class BinanceAdapter(ExchangeAdapter):
    """محول خاص بمنصة Binance، يستخدم أوامر OCO."""
    async def place_exit_orders(self, signal, verified_quantity):
        symbol = signal['symbol']
        tp_price = self.exchange.price_to_precision(symbol, signal['take_profit'])
        sl_price = self.exchange.price_to_precision(symbol, signal['stop_loss'])
        sl_trigger_price = self.exchange.price_to_precision(symbol, signal['stop_loss'])
        
        logger.info(f"BinanceAdapter: Placing OCO for {symbol}. TP: {tp_price}, SL Trigger: {sl_trigger_price}")
        oco_params = {'stopLimitPrice': sl_price}
        oco_order = await self.exchange.create_order(symbol, 'oco', 'sell', verified_quantity, price=tp_price, stopPrice=sl_trigger_price, params=oco_params)
        return {"oco_id": oco_order['id']}

    async def update_trailing_stop_loss(self, trade, new_sl):
        symbol = trade['symbol']
        exit_ids = json.loads(trade.get('exit_order_ids_json', '{}'))
        oco_id_to_cancel = exit_ids.get('oco_id')
        if not oco_id_to_cancel:
            raise ValueError("Binance trade is missing its OCO ID for TSL update.")

        logger.info(f"BinanceAdapter: Cancelling old OCO order {oco_id_to_cancel} for {symbol}.")
        await self.exchange.cancel_order(oco_id_to_cancel, symbol)
        await asyncio.sleep(2)

        quantity = trade['quantity']
        tp_price = self.exchange.price_to_precision(symbol, trade['take_profit'])
        sl_price = self.exchange.price_to_precision(symbol, new_sl)
        sl_trigger_price = self.exchange.price_to_precision(symbol, new_sl)
        
        logger.info(f"BinanceAdapter: Creating new OCO for {symbol} with new SL: {sl_price}")
        oco_params = {'stopLimitPrice': sl_price}
        new_oco_order = await self.exchange.create_order(symbol, 'oco', 'sell', quantity, price=tp_price, stopPrice=sl_trigger_price, params=oco_params)
        return {"oco_id": new_oco_order['id']}

class KuCoinAdapter(ExchangeAdapter):
    """محول خاص بمنصة KuCoin، يستخدم أمرين منفصلين."""
    async def place_exit_orders(self, signal, verified_quantity):
        symbol = signal['symbol']
        tp_price = self.exchange.price_to_precision(symbol, signal['take_profit'])
        sl_price = self.exchange.price_to_precision(symbol, signal['stop_loss'])
        sl_trigger_price = self.exchange.price_to_precision(symbol, signal['stop_loss'])
        
        logger.info(f"KuCoinAdapter: Placing separate TP and SL orders for {symbol}.")
        
        tp_order = await self.exchange.create_order(symbol, 'limit', 'sell', verified_quantity, price=tp_price)
        logger.info(f"KuCoinAdapter: Take Profit order placed with ID: {tp_order['id']}")
        
        sl_params = {'triggerPrice': sl_trigger_price, 'stop': 'loss'}
        sl_order = await self.exchange.create_order(symbol, 'stop_limit', 'sell', verified_quantity, price=sl_price, params=sl_params)
        logger.info(f"KuCoinAdapter: Stop Loss order placed with ID: {sl_order['id']}")
        
        return {"tp_id": tp_order['id'], "sl_id": sl_order['id']}

    async def update_trailing_stop_loss(self, trade, new_sl):
        symbol = trade['symbol']
        exit_ids = json.loads(trade.get('exit_order_ids_json', '{}'))
        tp_id_to_cancel = exit_ids.get('tp_id')
        sl_id_to_cancel = exit_ids.get('sl_id')
        if not tp_id_to_cancel or not sl_id_to_cancel:
            raise ValueError("KuCoin trade is missing TP or SL order ID for TSL update.")

        logger.info(f"KuCoinAdapter: Cancelling old orders for {symbol}. TP_ID: {tp_id_to_cancel}, SL_ID: {sl_id_to_cancel}")
        await self.exchange.cancel_order(tp_id_to_cancel, symbol)
        await self.exchange.cancel_order(sl_id_to_cancel, symbol)
        await asyncio.sleep(2)

        quantity = trade['quantity']
        tp_price = self.exchange.price_to_precision(symbol, trade['take_profit'])
        sl_price = self.exchange.price_to_precision(symbol, new_sl)
        sl_trigger_price = self.exchange.price_to_precision(symbol, new_sl)

        logger.info(f"KuCoinAdapter: Creating new separate orders for {symbol} with new SL: {sl_price}")
        new_tp_order = await self.exchange.create_order(symbol, 'limit', 'sell', quantity, price=tp_price)
        new_sl_params = {'triggerPrice': sl_trigger_price, 'stop': 'loss'}
        new_sl_order = await self.exchange.create_order(symbol, 'stop_limit', 'sell', quantity, price=sl_price, params=new_sl_params)
        
        return {"tp_id": new_tp_order['id'], "sl_id": new_sl_order['id']}

def get_exchange_adapter(exchange_id: str):
    """Factory function to get the correct adapter for an exchange."""
    exchange_client = bot_state.exchanges.get(exchange_id.lower())
    if not exchange_client:
        return None
        
    adapter_map = {
        'binance': BinanceAdapter,
        'kucoin': KuCoinAdapter,
    }
    AdapterClass = adapter_map.get(exchange_id.lower())
    if AdapterClass:
        return AdapterClass(exchange_client)
    
    logger.warning(f"No specific adapter found for {exchange_id}. Some features like automated TSL might not work.")
    return None

# =======================================================================================
# --- Configurations and Constants ---
# =======================================================================================

PRESET_PRO = {
  "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "max_spread_percent": 0.45, "rvol_period": 18, "min_rvol": 1.5},
  "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.85},
  "ema_trend_filter": {"enabled": True, "ema_period": 200},
  "min_tp_sl_filter": {"min_tp_percent": 1.1, "min_sl_percent": 0.6}
}
# ... (All other PRESETS remain the same)

STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي", "breakout_squeeze_pro": "اختراق انضغاطي",
    "support_rebound": "ارتداد الدعم", "whale_radar": "رادار الحيتان", "sniper_pro": "القناص المحترف",
}

EDITABLE_PARAMS = {
    # ... (Same as before)
}
PARAM_DISPLAY_NAMES = {
    # ... (Same as before)
}

DEFAULT_SETTINGS = {
    "real_trading_per_exchange": {ex: False for ex in EXCHANGES_TO_SCAN}, 
    "automate_real_tsl": False,
    "real_trade_size_usdt": 15.0,
    "virtual_portfolio_balance_usdt": 1000.0, "virtual_trade_size_percentage": 5.0, "max_concurrent_trades": 10, "top_n_symbols_by_volume": 250, "concurrent_workers": 10,
    "market_regime_filter_enabled": True, "fundamental_analysis_enabled": True,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "whale_radar", "sniper_pro"],
    "use_master_trend_filter": True, "master_trend_filter_ma_period": 50, "master_adx_filter_level": 22,
    "fear_and_greed_filter_enabled": True, "fear_and_greed_threshold": 30,
    "use_dynamic_risk_management": True, "atr_period": 14, "atr_sl_multiplier": 2.5, "risk_reward_ratio": 2.0,
    "trailing_sl_enabled": True, "trailing_sl_activation_percent": 1.5, "trailing_sl_callback_percent": 1.0,
    "momentum_breakout": {"vwap_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bbands_period": 20, "bbands_stddev": 2.0, "rsi_period": 14, "rsi_max_level": 68, "volume_spike_multiplier": 1.5},
    "breakout_squeeze_pro": {"bbands_period": 20, "bbands_stddev": 2.0, "keltner_period": 20, "keltner_atr_multiplier": 1.5, "volume_confirmation_enabled": True},
    "sniper_pro": {"compression_hours": 6, "max_volatility_percent": 12.0},
    "whale_radar": {"wall_threshold_usdt": 30000},
    "liquidity_filters": {"min_quote_volume_24h_usd": 1_000_000, "max_spread_percent": 0.5, "rvol_period": 20, "min_rvol": 1.5},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.8},
    "stablecoin_filter": {"exclude_bases": ["USDT","USDC","DAI","FDUSD","TUSD","USDE","PYUSD","GUSD","EURT","USDJ"]},
    "ema_trend_filter": {"enabled": True, "ema_period": 200},
    "min_tp_sl_filter": {"min_tp_percent": 1.0, "min_sl_percent": 0.5},
    "min_signal_strength": 1,
    "active_preset_name": "PRO",
    "last_market_mood": {"timestamp": "N/A", "mood": "UNKNOWN", "reason": "No scan performed yet."},
    "last_suggestion_time": 0
}

# =======================================================================================
# --- Helper Functions (Settings, DB, Analysis, etc.) ---
# =======================================================================================

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                bot_state.settings = json.load(f)
        else:
            bot_state.settings = DEFAULT_SETTINGS.copy()
            save_settings()
            return
        
        updated = False
        if "real_trading_enabled" in bot_state.settings:
            old_value = bot_state.settings.pop("real_trading_enabled")
            bot_state.settings["real_trading_per_exchange"] = {ex: old_value for ex in EXCHANGES_TO_SCAN}
            updated = True
            
        for key, value in DEFAULT_SETTINGS.items():
            if key not in bot_state.settings:
                bot_state.settings[key] = value; updated = True
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in bot_state.settings.get(key, {}):
                        bot_state.settings[key][sub_key] = sub_value; updated = True
        if updated:
            save_settings()
        
        logger.info(f"Settings loaded successfully into BotState.")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        bot_state.settings = DEFAULT_SETTINGS.copy()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(bot_state.settings, f, indent=4)
        logger.info(f"Settings saved successfully from BotState.")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

def migrate_database():
    logger.info("Checking database schema...")
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        
        required_columns = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT", "timestamp": "TEXT", "exchange": "TEXT",
            "symbol": "TEXT", "entry_price": "REAL", "take_profit": "REAL", "stop_loss": "REAL",
            "quantity": "REAL", "entry_value_usdt": "REAL", "status": "TEXT", "exit_price": "REAL",
            "closed_at": "TEXT", "exit_value_usdt": "REAL", "pnl_usdt": "REAL",
            "trailing_sl_active": "BOOLEAN", "highest_price": "REAL", "reason": "TEXT",
            "is_real_trade": "BOOLEAN", "trade_mode": "TEXT DEFAULT 'virtual'",
            "entry_order_id": "TEXT", "exit_order_ids_json": "TEXT"
        }
        
        cursor.execute("PRAGMA table_info(trades)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        for col_name, col_type in required_columns.items():
            if col_name not in existing_columns:
                logger.warning(f"Database schema mismatch. Missing column '{col_name}'. Adding it now.")
                cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                logger.info(f"Column '{col_name}' added successfully.")
        
        conn.commit()
        conn.close()
        logger.info("Database schema check complete.")
    except Exception as e:
        logger.error(f"CRITICAL: Database migration failed: {e}", exc_info=True)

def init_database():
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT)')
        conn.commit()
        conn.close()
        migrate_database()
        logger.info(f"Database initialized and schema verified at: {DB_FILE}")
    except Exception as e:
        logger.error(f"Failed to initialize database at {DB_FILE}: {e}")

def log_recommendation_to_db(signal):
    # ... (Same as before, no changes needed here)
    pass
# ... (All scanner functions, find_col, find_support_resistance, fundamental analysis functions are the same) ...

# =======================================================================================
# --- Core Bot Logic ---
# =======================================================================================

async def initialize_exchanges():
    """Initializes exchange clients and populates BotState."""
    async def connect(ex_id):
        try:
            public_exchange = getattr(ccxt_async, ex_id)({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
            await public_exchange.load_markets()
            bot_state.public_exchanges[ex_id] = public_exchange
            logger.info(f"Connected to {ex_id} with PUBLIC client.")
        except Exception as e:
            logger.error(f"Failed to connect PUBLIC client for {ex_id}: {e}")

        params = {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
        authenticated = False
        if ex_id == 'binance' and BINANCE_API_KEY != 'YOUR_BINANCE_API_KEY':
            params.update({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET})
            authenticated = True
        if ex_id == 'kucoin' and KUCOIN_API_KEY != 'YOUR_KUCOIN_API_KEY':
            params.update({'apiKey': KUCOIN_API_KEY, 'secret': KUCOIN_API_SECRET, 'password': KUCOIN_API_PASSPHRASE})
            authenticated = True

        if authenticated:
            try:
                private_exchange = getattr(ccxt_async, ex_id)(params)
                await private_exchange.load_markets()
                bot_state.exchanges[ex_id] = private_exchange
                logger.info(f"Connected to {ex_id} with PRIVATE client.")
            except Exception as e:
                logger.error(f"Failed to connect PRIVATE client for {ex_id}: {e}")
        
    await asyncio.gather(*[connect(ex_id) for ex_id in EXCHANGES_TO_SCAN])
    logger.info("All exchange connections initialized in BotState.")


async def place_real_trade(signal):
    exchange_id = signal['exchange'].lower()
    adapter = get_exchange_adapter(exchange_id)
    if not adapter:
        return {'success': False, 'data': f"No trade adapter available for {exchange_id.capitalize()}."}

    exchange = adapter.exchange
    settings = bot_state.settings
    symbol = signal['symbol']

    try:
        usdt_balance = await get_real_balance(exchange_id, 'USDT')
        user_trade_amount_usdt = settings.get("real_trade_size_usdt", 15.0)

        markets = await exchange.load_markets()
        market_info = markets.get(symbol)
        if not market_info:
            return {'success': False, 'data': f"Could not find market info for {symbol}."}

        min_notional = 0
        if 'minNotional' in market_info.get('limits', {}).get('cost', {}):
             min_notional = market_info['limits']['cost']['minNotional']
        elif exchange_id == 'kucoin':
            min_notional = float(market_info.get('info', {}).get('minProvideSize', 5.0))

        trade_amount_usdt = max(user_trade_amount_usdt, min_notional)
        if min_notional > user_trade_amount_usdt:
             logger.warning(f"User trade size ${user_trade_amount_usdt} for {symbol} is below exchange minimum of ${min_notional}. Using exchange minimum.")

        if usdt_balance < trade_amount_usdt:
            return {'success': False, 'data': f"رصيدك الحالي ${usdt_balance:.2f} غير كافٍ لفتح صفقة بقيمة ${trade_amount_usdt:.2f}."}
        
        quantity = trade_amount_usdt / signal['entry_price']
        formatted_quantity = exchange.amount_to_precision(symbol, quantity)
    except Exception as e:
        return {'success': False, 'data': f"Pre-flight check failed: {e}"}

    buy_order = None
    try:
        logger.info(f"Placing MARKET BUY order for {formatted_quantity} of {symbol} on {exchange_id.capitalize()}")
        buy_order = await exchange.create_market_buy_order(symbol, float(formatted_quantity))
        logger.info(f"Initial response for BUY order {buy_order.get('id', 'N/A')} received.")
    except Exception as e:
        logger.error(f"Placing BUY order for {symbol} failed immediately: {e}", exc_info=True)
        return {'success': False, 'data': f"حدث خطأ من المنصة عند محاولة الشراء: `{str(e)}`"}

    verified_order = None
    verified_price, verified_quantity, verified_cost = 0, 0, 0
    try:
        max_attempts = 5
        delay_seconds = 3
        for attempt in range(max_attempts):
            logger.info(f"Verifying BUY order {buy_order.get('id', 'N/A')}... (Attempt {attempt + 1}/{max_attempts})")
            try:
                order_status = await exchange.fetch_order(buy_order['id'], symbol)
                if order_status and order_status.get('status') == 'closed' and order_status.get('filled', 0) > 0:
                    verified_order = order_status
                    break 
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay_seconds)
            except ccxt.OrderNotFound:
                logger.warning(f"Order {buy_order.get('id', 'N/A')} not found, retrying...")
                await asyncio.sleep(delay_seconds)
            except Exception as fetch_e:
                logger.error(f"Error during order verification: {fetch_e}")
                await asyncio.sleep(delay_seconds)

        if verified_order:
            verified_price = verified_order.get('average', signal['entry_price'])
            verified_quantity = verified_order.get('filled')
            verified_cost = verified_order.get('cost', verified_price * verified_quantity)
            logger.info(f"BUY order {buy_order['id']} VERIFIED. Filled {verified_quantity} @ {verified_price}")
        else:
            raise Exception(f"Order could not be confirmed as filled after {max_attempts} attempts.")
    except Exception as e:
        logger.error(f"VERIFICATION FAILED for BUY order {buy_order.get('id', 'N/A')}: {e}", exc_info=True)
        return {'success': False, 'manual_check_required': True, 'data': f"تم إرسال أمر الشراء لكن فشل التحقق منه. **يرجى التحقق من المنصة يدوياً!** ID: `{buy_order.get('id', 'N/A')}`. Error: `{e}`"}

    try:
        exit_order_ids = await adapter.place_exit_orders(signal, verified_quantity)
        logger.info(f"Adapter successfully placed exit orders for {symbol} with IDs: {exit_order_ids}")
        return {
            'success': True, 'exit_orders_failed': False,
            'data': {
                "entry_order_id": buy_order['id'], "exit_order_ids_json": json.dumps(exit_order_ids),
                "verified_quantity": verified_quantity, "verified_entry_price": verified_price,
                "verified_entry_value": verified_cost
            }
        }
    except Exception as e:
        logger.error(f"Adapter failed to place exit orders for {symbol}: {e}", exc_info=True)
        error_data = {
            "entry_order_id": buy_order['id'], "exit_order_ids_json": json.dumps({}),
            "verified_quantity": verified_quantity, "verified_entry_price": verified_price,
            "verified_entry_value": verified_cost
        }
        return {'success': True, 'exit_orders_failed': True, 'data': error_data}


async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    # ... (Logic at the start is the same: scan_lock, fundamental check, market regime check) ...
    pass

async def track_open_trades(context: ContextTypes.DEFAULT_TYPE):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'نشطة'")
        active_trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        logger.error(f"DB error in track_open_trades: {e}")
        return

    bot_state.status_snapshot['active_trades_count'] = len(active_trades)
    if not active_trades:
        return

    tasks = [check_single_trade(trade, context) for trade in active_trades]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in results:
        if isinstance(res, Exception):
            logger.error(f"An exception occurred during concurrent trade tracking: {res}")


async def check_single_trade(trade: dict, context: ContextTypes.DEFAULT_TYPE):
    exchange_id = trade['exchange'].lower()
    public_exchange = bot_state.public_exchanges.get(exchange_id)
    if not public_exchange:
        logger.warning(f"No public exchange client for tracking trade #{trade['id']}.")
        return

    try:
        ticker = await public_exchange.fetch_ticker(trade['symbol'])
        current_price = ticker.get('last') or ticker.get('close')
        if not current_price:
            logger.warning(f"Could not fetch price for {trade['symbol']}")
            return

        # --- TP/SL Check ---
        current_stop_loss = trade.get('stop_loss') or 0
        current_take_profit = trade.get('take_profit')
        if current_take_profit is not None and current_price >= current_take_profit:
            await close_trade_in_db(context, trade, current_price, 'ناجحة')
            return
        if current_stop_loss > 0 and current_price <= current_stop_loss:
            await close_trade_in_db(context, trade, current_price, 'فاشلة')
            return

        # --- Trailing SL Logic ---
        settings = bot_state.settings
        if settings.get('trailing_sl_enabled', True):
            highest_price = max(trade.get('highest_price', current_price) or current_price, current_price)
            
            if not trade.get('trailing_sl_active'):
                activation_price = trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100)
                if current_price >= activation_price:
                    new_sl = trade['entry_price']
                    if new_sl > current_stop_loss:
                        await handle_tsl_update(context, trade, new_sl, highest_price, is_activation=True)
            elif trade.get('trailing_sl_active'):
                new_sl = highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                if new_sl > current_stop_loss:
                    await handle_tsl_update(context, trade, new_sl, highest_price)
            
            if highest_price > (trade.get('highest_price') or 0):
                await update_trade_peak_price_in_db(trade['id'], highest_price)

    except Exception as e:
        logger.error(f"Error in check_single_trade for #{trade['id']}: {e}", exc_info=True)


async def handle_tsl_update(context, trade, new_sl, highest_price, is_activation=False):
    settings = bot_state.settings
    is_real_automated = trade.get('trade_mode') == 'real' and settings.get('automate_real_tsl', False)

    if is_real_automated:
        await update_real_trade_sl(context, trade, new_sl, highest_price, is_activation)
    elif trade.get('trade_mode') == 'real':
        await send_telegram_message(context.bot, {**trade, "new_sl": new_sl, "current_price": highest_price}, update_type='tsl_update_real')
        await update_trade_sl_in_db(context, trade, new_sl, highest_price, is_activation=is_activation, silent=True)
    else: # Virtual trade
        await update_trade_sl_in_db(context, trade, new_sl, highest_price, is_activation=is_activation)


async def update_real_trade_sl(context, trade, new_sl, highest_price, is_activation=False):
    exchange_id = trade['exchange'].lower()
    symbol = trade['symbol']
    logger.info(f"AUTOMATING TSL UPDATE for real trade #{trade['id']} ({symbol}). New SL: {new_sl}")

    adapter = get_exchange_adapter(exchange_id)
    if not adapter:
        logger.error(f"Cannot automate TSL for {symbol}: No adapter for {exchange_id}.")
        return

    try:
        new_exit_ids = await adapter.update_trailing_stop_loss(trade, new_sl)
        await update_trade_sl_in_db(context, trade, new_sl, highest_price, is_activation=is_activation, new_exit_ids_json=json.dumps(new_exit_ids))
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE in automated TSL for #{trade['id']} ({symbol}): {e}", exc_info=True)
        # ... send alert ...


# ... (The rest of the helper functions: close_trade_in_db, update_trade_sl_in_db, etc. are the same) ...
# ... (All Telegram Handlers are the same, they just need to use `bot_state` now) ...

# =======================================================================================
# --- Bot Startup and Main Loop ---
# =======================================================================================

async def post_init(application: Application):
    if NLTK_AVAILABLE:
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError: logger.info("Downloading NLTK data..."); nltk.download('vader_lexicon')
    
    logger.info("Post-init: Initializing exchanges...")
    await initialize_exchanges()
    if not bot_state.public_exchanges: 
        logger.critical("CRITICAL: No public exchanges connected. Bot cannot run.")
        return

    job_queue = application.job_queue
    job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name='perform_scan')
    job_queue.run_repeating(track_open_trades, interval=TRACK_INTERVAL_SECONDS, first=20, name='track_open_trades')
    # ... daily report job ...

    logger.info("Jobs scheduled.")
    await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🚀 *بوت كاسحة الألغام (v5.0) جاهز للعمل!*", parse_mode=ParseMode.MARKDOWN)

async def post_shutdown(application: Application):
    all_exchanges = list(bot_state.exchanges.values()) + list(bot_state.public_exchanges.values())
    unique_exchanges = list({id(ex): ex for ex in all_exchanges}.values())
    await asyncio.gather(*[ex.close() for ex in unique_exchanges])
    logger.info("All exchange connections closed.")


def main():
    """Sets up and runs the bot application."""
    load_settings()
    init_database()

    request = HTTPXRequest(connect_timeout=60.0, read_timeout=60.0)
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).post_init(post_init).post_shutdown(post_shutdown).build()

    # Add all command and message handlers here
    # application.add_handler(...)
    
    logger.info("Application configured. Starting polling...")
    application.run_polling()


if __name__ == '__main__':
    print("🚀 Starting Mineseper Bot v5.0 (Architectural Refactor)...")
    try:
        main()
    except Exception as e:
        logging.critical(f"Bot stopped due to a critical unhandled error: {e}", exc_info=True)

