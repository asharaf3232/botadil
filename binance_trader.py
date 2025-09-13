# -*- coding: utf-8 -*-
# =======================================================================================
# --- 💣 بوت كاسحة الألغام (Minesweeper Bot) v6.4 (الإصلاح النهائي والتحسينات) 💣 ---
# =======================================================================================
# --- سجل التغييرات v6.4 ---
#
# 1. [إصلاح حاسم] تم إصلاح جميع أخطاء بناء الجملة (Syntax Errors) الناتجة عن التجميع الخاطئ.
# 2. [ميزة رئيسية] تمت ترقية أداة "المزامنة" إلى "المزامنة والإنقاذ الذكي":
#    - تقوم الأداة الآن تلقائياً باكتشاف الصفقات "اليتيمة" (الموجودة في المنصة وغير المسجلة).
#    - يمكن استيراد أي صفقة يتيمة بضغطة زر.
#    - يقوم البوت بالبحث في سجل التداول لحساب متوسط سعر الشراء بدقة وإعادة بناء الصفقة.
#    - تبدأ متابعة الصفقة المستوردة فوراً، مما يحل مشكلة فقدان البيانات عند إعادة التشغيل.
# 3. [تحسين وظيفي] تم تفعيل زر التحديث (🔄) في لوحة التحكم:
#    - يقوم الزر الآن ببدء عملية فحص يدوي فوري للسوق عند الطلب.
# 4. [تحسين الواجهة] تم تحسين تقرير التشخيص:
#    - يعرض الآن الوقت المتبقي للعمليات المجدولة (الفحص والمتابعة) كعد تنازلي.
#    - يعرض حالة "يعمل الآن" إذا كانت المهمة قيد التنفيذ لمزيد من الوضوح.
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

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Library 'scipy' not found. RSI Divergence strategy will be disabled.")

# --- الإعدادات الأساسية ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')
TELEGRAM_SIGNAL_CHANNEL_ID = os.getenv('TELEGRAM_SIGNAL_CHANNEL_ID', TELEGRAM_CHAT_ID)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_AV_KEY_HERE')

# --- [v5.8] Add API Keys for all supported exchanges ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', '')
KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET', '')
KUCOIN_API_PASSPHRASE = os.getenv('KUCOIN_API_PASSPHRASE', '')
GATE_API_KEY = os.getenv('GATE_API_KEY', '')
GATE_API_SECRET = os.getenv('GATE_API_SECRET', '')
MEXC_API_KEY = os.getenv('MEXC_API_KEY', '')
MEXC_API_SECRET = os.getenv('MEXC_API_SECRET', '')
OKX_API_KEY = os.getenv('OKX_API_KEY', '')
OKX_API_SECRET = os.getenv('OKX_API_SECRET', '')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE', '')
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')

# --- إعدادات البوت ---
EXCHANGES_TO_SCAN = ['binance', 'okx', 'bybit', 'kucoin', 'gate', 'mexc']
TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
SCAN_INTERVAL_SECONDS = 900
TRACK_INTERVAL_SECONDS = 45

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v6.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'minesweeper_settings_v6.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")

# --- إعداد مسجل الأحداث (Logger) ---
LOG_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v6.log')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'a', 'utf-8'), logging.StreamHandler()])
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger("MinesweeperBot_v6")


# =======================================================================================
# --- 🚀 [v5.8] إعادة هيكلة المحولات (Adapters) لدعم منصات متعددة 🚀 ---
# =======================================================================================

class BotState:
    """كلاس مركزي لإدارة كل حالة البوت لزيادة التنظيم."""
    def __init__(self):
        self.exchanges = {}
        self.public_exchanges = {}
        self.last_signal_time = {}
        self.settings = {}
        self.status_snapshot = {
            "last_scan_start_time": None, "last_scan_end_time": None,
            "markets_found": 0, "signals_found": 0, "active_trades_count": 0,
            "scan_in_progress": False, "btc_market_mood": "غير محدد"
        }
        self.scan_history = deque(maxlen=10)

bot_state = BotState()
scan_lock = asyncio.Lock()
report_lock = asyncio.Lock()

class ExchangeAdapter:
    """كلاس أساسي مجرد لنمط المحول."""
    def __init__(self, exchange_client):
        self.exchange = exchange_client

    async def place_exit_orders(self, signal, verified_quantity):
        raise NotImplementedError("يجب تعريف هذه الدالة في الكلاس الفرعي")

    async def update_trailing_stop_loss(self, trade, new_sl):
        raise NotImplementedError("يجب تعريف هذه الدالة في الكلاس الفرعي")

class OcoAdapter(ExchangeAdapter):
    """محول أساسي للمنصات التي تدعم أوامر OCO (مثل Binance, Bybit, Gate, OKX)."""
    async def place_exit_orders(self, signal, verified_quantity):
        symbol = signal['symbol']
        tp_price = self.exchange.price_to_precision(symbol, signal['take_profit'])
        sl_price = self.exchange.price_to_precision(symbol, signal['stop_loss'])
        
        logger.info(f"{self.exchange.id} OCO: Placing for {symbol}. TP: {tp_price}, SL: {sl_price}")
        params = {'stopLimitPrice': sl_price} if self.exchange.id == 'binance' else {}
        
        oco_order = await self.exchange.create_order(
            symbol=symbol, type='oco', side='sell', amount=verified_quantity,
            price=tp_price, stopPrice=sl_price, params=params
        )
        return {"oco_id": oco_order['id']}

    async def update_trailing_stop_loss(self, trade, new_sl):
        symbol = trade['symbol']
        exit_ids = json.loads(trade.get('exit_order_ids_json', '{}'))
        oco_id_to_cancel = exit_ids.get('oco_id')
        if not oco_id_to_cancel:
            raise ValueError(f"{self.exchange.id} trade is missing its OCO ID for TSL update.")

        logger.info(f"{self.exchange.id} OCO: Cancelling old OCO order {oco_id_to_cancel} for {symbol}.")
        try:
            await self.exchange.cancel_order(oco_id_to_cancel, symbol)
        except ccxt.OrderNotFound:
            logger.warning(f"OCO order {oco_id_to_cancel} not found, likely already filled/cancelled.")
        await asyncio.sleep(2)

        quantity = trade['quantity']
        tp_price = self.exchange.price_to_precision(symbol, trade['take_profit'])
        sl_price = self.exchange.price_to_precision(symbol, new_sl)
        
        logger.info(f"{self.exchange.id} OCO: Creating new OCO for {symbol} with new SL: {sl_price}")
        params = {'stopLimitPrice': sl_price} if self.exchange.id == 'binance' else {}

        new_oco_order = await self.exchange.create_order(
            symbol=symbol, type='oco', side='sell', amount=quantity,
            price=tp_price, stopPrice=sl_price, params=params
        )
        return {"oco_id": new_oco_order['id']}

class DualOrderAdapter(ExchangeAdapter):
    """محول أساسي للمنصات التي تتطلب أمرين منفصلين للخروج (مثل KuCoin, MEXC)."""
    async def place_exit_orders(self, signal, verified_quantity):
        symbol = signal['symbol']
        tp_price = self.exchange.price_to_precision(symbol, signal['take_profit'])
        sl_trigger_price = self.exchange.price_to_precision(symbol, signal['stop_loss'])
        
        logger.info(f"{self.exchange.id} DualOrder: Placing separate TP and SL orders for {symbol}.")
        
        tp_order = await self.exchange.create_order(symbol, 'limit', 'sell', verified_quantity, price=tp_price)
        logger.info(f"{self.exchange.id} DualOrder: Take Profit order placed with ID: {tp_order['id']}")
        
        sl_params = {'stopPrice': sl_trigger_price}
        sl_order = await self.exchange.create_order(symbol, 'market', 'sell', verified_quantity, params=sl_params)
        logger.info(f"{self.exchange.id} DualOrder: Stop Loss (Market) order placed with ID: {sl_order['id']}")
        
        return {"tp_id": tp_order['id'], "sl_id": sl_order['id']}

    async def update_trailing_stop_loss(self, trade, new_sl):
        symbol = trade['symbol']
        exit_ids = json.loads(trade.get('exit_order_ids_json', '{}'))
        tp_id_to_cancel = exit_ids.get('tp_id')
        sl_id_to_cancel = exit_ids.get('sl_id')
        if not tp_id_to_cancel or not sl_id_to_cancel:
            raise ValueError(f"{self.exchange.id} trade is missing TP or SL order ID for TSL update.")

        logger.info(f"{self.exchange.id} DualOrder: Cancelling old orders for {symbol}. TP_ID: {tp_id_to_cancel}, SL_ID: {sl_id_to_cancel}")
        try: await self.exchange.cancel_order(tp_id_to_cancel, symbol)
        except ccxt.OrderNotFound: logger.warning(f"TP order {tp_id_to_cancel} not found, likely already filled.")
        try: await self.exchange.cancel_order(sl_id_to_cancel, symbol)
        except ccxt.OrderNotFound: logger.warning(f"SL order {sl_id_to_cancel} not found, likely already filled.")
        await asyncio.sleep(2)

        quantity = trade['quantity']
        tp_price = self.exchange.price_to_precision(symbol, trade['take_profit'])
        sl_trigger_price = self.exchange.price_to_precision(symbol, new_sl)

        logger.info(f"{self.exchange.id} DualOrder: Creating new separate orders for {symbol} with new SL trigger: {sl_trigger_price}")
        new_tp_order = await self.exchange.create_order(symbol, 'limit', 'sell', quantity, price=tp_price)
        
        new_sl_params = {'stopPrice': sl_trigger_price}
        new_sl_order = await self.exchange.create_order(symbol, 'market', 'sell', quantity, params=new_sl_params)
        
        return {"tp_id": new_tp_order['id'], "sl_id": new_sl_order['id']}

class BinanceAdapter(OcoAdapter): pass
class BybitAdapter(OcoAdapter): pass
class GateAdapter(OcoAdapter): pass
class OKXAdapter(OcoAdapter): pass
class KuCoinAdapter(DualOrderAdapter): pass
class MEXCAdapter(DualOrderAdapter): pass

def get_exchange_adapter(exchange_id: str):
    exchange_client = bot_state.exchanges.get(exchange_id.lower())
    if not exchange_client: return None
    
    adapter_map = {
        'binance': BinanceAdapter, 'kucoin': KuCoinAdapter, 'okx': OKXAdapter,
        'bybit': BybitAdapter, 'gate': GateAdapter, 'mexc': MEXCAdapter
    }
    AdapterClass = adapter_map.get(exchange_id.lower())
    
    if AdapterClass: return AdapterClass(exchange_client)
    
    logger.warning(f"No specific adapter found for {exchange_id}, trade automation will be disabled for it.")
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
PRESET_LAX = {
 "liquidity_filters": {"min_quote_volume_24h_usd": 400000, "max_spread_percent": 1.3, "rvol_period": 12, "min_rvol": 1.1},
 "volatility_filters": {"atr_period_for_filter": 10, "min_atr_percent": 0.3},
 "ema_trend_filter": {"enabled": False, "ema_period": 200},
 "min_tp_sl_filter": {"min_tp_percent": 0.4, "min_sl_percent": 0.2}
}
PRESET_STRICT = {
 "liquidity_filters": {"min_quote_volume_24h_usd": 2500000, "max_spread_percent": 0.22, "rvol_period": 25, "min_rvol": 2.2},
 "volatility_filters": {"atr_period_for_filter": 20, "min_atr_percent": 1.4},
 "ema_trend_filter": {"enabled": True, "ema_period": 200},
 "min_tp_sl_filter": {"min_tp_percent": 1.8, "min_sl_percent": 0.9}
}
PRESET_VERY_LAX = {
 "liquidity_filters": {"min_quote_volume_24h_usd": 200000, "max_spread_percent": 2.0, "rvol_period": 10, "min_rvol": 0.8},
 "volatility_filters": {"atr_period_for_filter": 10, "min_atr_percent": 0.2},
 "ema_trend_filter": {"enabled": False, "ema_period": 200},
 "min_tp_sl_filter": {"min_tp_percent": 0.3, "min_sl_percent": 0.15}
}
PRESETS = {"PRO": PRESET_PRO, "LAX": PRESET_LAX, "STRICT": PRESET_STRICT, "VERY_LAX": PRESET_VERY_LAX}

STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي", "breakout_squeeze_pro": "اختراق انضغاطي",
    "support_rebound": "ارتداد الدعم", "whale_radar": "رادار الحيتان", "sniper_pro": "القناص المحترف",
    "Rescued/Imported": "مستورد/تم إنقاذه"
}

EDITABLE_PARAMS = {
    "إعدادات عامة": [
        "max_concurrent_trades", "top_n_symbols_by_volume", "concurrent_workers",
        "min_signal_strength"
    ],
    "إعدادات المخاطر": [
        "automate_real_tsl", "real_trade_size_usdt", "virtual_trade_size_percentage",
        "atr_sl_multiplier", "risk_reward_ratio", "trailing_sl_activation_percent", "trailing_sl_callback_percent"
    ],
    "الفلاتر والاتجاه": [
        "market_regime_filter_enabled", "use_master_trend_filter", "fear_and_greed_filter_enabled",
        "master_adx_filter_level", "master_trend_filter_ma_period", "trailing_sl_enabled", "fear_and_greed_threshold",
        "fundamental_analysis_enabled"
    ]
}
PARAM_DISPLAY_NAMES = {
    "automate_real_tsl": "🤖 أتمتة الوقف المتحرك الحقيقي",
    "real_trade_size_usdt": "💵 حجم الصفقة الحقيقية ($)",
    "virtual_trade_size_percentage": "📊 حجم الصفقة الوهمية (%)",
    "max_concurrent_trades": "أقصى عدد للصفقات",
    "top_n_symbols_by_volume": "عدد العملات للفحص",
    "concurrent_workers": "عمال الفحص المتزامنين",
    "min_signal_strength": "أدنى قوة للإشارة",
    "atr_sl_multiplier": "مضاعف وقف الخسارة (ATR)",
    "risk_reward_ratio": "نسبة المخاطرة/العائد",
    "trailing_sl_activation_percent": "تفعيل الوقف المتحرك (%)",
    "trailing_sl_callback_percent": "مسافة الوقف المتحرك (%)",
    "market_regime_filter_enabled": "فلتر وضع السوق (فني)",
    "use_master_trend_filter": "فلتر الاتجاه العام (BTC)",
    "master_adx_filter_level": "مستوى فلتر ADX",
    "master_trend_filter_ma_period": "فترة فلتر الاتجاه",
    "trailing_sl_enabled": "تفعيل الوقف المتحرك",
    "fear_and_greed_filter_enabled": "فلتر الخوف والطمع",
    "fear_and_greed_threshold": "حد مؤشر الخوف",
    "fundamental_analysis_enabled": "فلتر الأخبار والبيانات",
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
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
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
                bot_state.settings[key] = value
                updated = True
            elif isinstance(value, dict):
                if key not in bot_state.settings or not isinstance(bot_state.settings[key], dict):
                    bot_state.settings[key] = value
                    updated = True
                else:
                    for sub_key, sub_value in value.items():
                        if sub_key not in bot_state.settings.get(key, {}):
                            bot_state.settings[key][sub_key] = sub_value
                            updated = True
        if updated:
            save_settings()
        
        logger.info("Settings loaded successfully into BotState.")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        bot_state.settings = DEFAULT_SETTINGS.copy()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(bot_state.settings, f, indent=4, ensure_ascii=False)
        logger.info("Settings saved successfully from BotState.")
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
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        sql = '''INSERT INTO trades (timestamp, exchange, symbol, entry_price, take_profit, stop_loss, quantity, entry_value_usdt, status, trailing_sl_active, highest_price, reason, trade_mode, entry_order_id, exit_order_ids_json)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        
        if 'quantity' not in signal or signal['quantity'] is None:
            logger.error(f"Attempted to log trade for {signal['symbol']} with missing quantity.")
            return None

        params = (
            signal.get('timestamp', datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')),
            signal['exchange'],
            signal['symbol'],
            signal.get('entry_price'),
            signal.get('take_profit'),
            signal.get('stop_loss'),
            signal.get('quantity'),
            signal.get('entry_value_usdt'), 
            'نشطة',
            False,
            signal.get('entry_price'),
            signal['reason'],
            'real' if signal.get('is_real_trade') else 'virtual',
            signal.get('entry_order_id'),
            signal.get('exit_order_ids_json')
        )
        cursor.execute(sql, params)
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id
    except Exception as e:
        logger.error(f"Failed to log recommendation to DB: {e}", exc_info=True)
        return None

async def get_alpha_vantage_economic_events():
    if ALPHA_VANTAGE_API_KEY == 'YOUR_AV_KEY_HERE': return []
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    params = {'function': 'ECONOMIC_CALENDAR', 'horizon': '3month', 'apikey': ALPHA_VANTAGE_API_KEY}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('https://www.alphavantage.co/query', params=params, timeout=20)
            response.raise_for_status()
        data_str = response.text
        if "premium" in data_str.lower(): return []
        lines = data_str.strip().split('\r\n')
        if len(lines) < 2: return []
        header = [h.strip() for h in lines[0].split(',')]
        high_impact_events = [dict(zip(header, [v.strip() for v in line.split(',')])).get('event', 'Unknown Event') 
                              for line in lines[1:] 
                              if dict(zip(header, [v.strip() for v in line.split(',')])).get('releaseDate', '') == today_str 
                              and dict(zip(header, [v.strip() for v in line.split(',')])).get('impact', '').lower() == 'high' 
                              and dict(zip(header, [v.strip() for v in line.split(',')])).get('country', '') in ['USD', 'EUR']]
        if high_impact_events: logger.warning(f"High-impact events today: {high_impact_events}")
        return high_impact_events
    except httpx.RequestError as e:
        logger.error(f"Failed to fetch economic calendar: {e}")
        return None

def get_latest_crypto_news(limit=15):
    urls = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]
    headlines = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            headlines.extend(entry.title for entry in feed.entries[:5])
        except Exception as e:
            logger.error(f"Failed to fetch news from {url}: {e}")
    return list(set(headlines))[:limit]

def analyze_sentiment_of_headlines(headlines):
    if not headlines or not NLTK_AVAILABLE: return 0.0
    sia = SentimentIntensityAnalyzer()
    total_compound_score = sum(sia.polarity_scores(headline)['compound'] for headline in headlines)
    return total_compound_score / len(headlines) if headlines else 0.0

async def get_fundamental_market_mood():
    high_impact_events = await get_alpha_vantage_economic_events()
    if high_impact_events is None: return "DANGEROUS", -1.0, "فشل جلب البيانات الاقتصادية"
    if high_impact_events: return "DANGEROUS", -0.9, f"أحداث هامة اليوم: {', '.join(high_impact_events)}"
    sentiment_score = analyze_sentiment_of_headlines(get_latest_crypto_news())
    logger.info(f"Market sentiment score: {sentiment_score:.2f}")
    if sentiment_score > 0.25: return "POSITIVE", sentiment_score, f"مشاعر إيجابية (الدرجة: {sentiment_score:.2f})"
    elif sentiment_score < -0.25: return "NEGATIVE", sentiment_score, f"مشاعر سلبية (الدرجة: {sentiment_score:.2f})"
    else: return "NEUTRAL", sentiment_score, f"مشاعر محايدة (الدرجة: {sentiment_score:.2f})"

def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

def analyze_momentum_breakout(df, params, rvol, adx_value, exchange, symbol):
    df.ta.vwap(append=True)
    df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
    df.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'], append=True)
    df.ta.rsi(length=params['rsi_period'], append=True)
    macd_col, macds_col, bbu_col, rsi_col = (
        find_col(df.columns, f"MACD_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}"),
        find_col(df.columns, f"MACDs_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}"),
        find_col(df.columns, f"BBU_{params['bbands_period']}_"),
        find_col(df.columns, f"RSI_{params['rsi_period']}")
    )
    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    rvol_ok = rvol >= bot_state.settings['liquidity_filters']['min_rvol']
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and
        last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and
        last[rsi_col] < params['rsi_max_level'] and rvol_ok):
        return {"reason": "momentum_breakout", "type": "long"}
    return None

def analyze_breakout_squeeze_pro(df, params, rvol, adx_value, exchange, symbol):
    df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
    df.ta.kc(length=params['keltner_period'], scalar=params['keltner_atr_multiplier'], append=True)
    df.ta.obv(append=True)
    bbu_col, bbl_col, kcu_col, kcl_col = (
        find_col(df.columns, f"BBU_{params['bbands_period']}_"), find_col(df.columns, f"BBL_{params['bbands_period']}_"),
        find_col(df.columns, f"KCUe_{params['keltner_period']}_"), find_col(df.columns, f"KCLEe_{params['keltner_period']}_")
    )
    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]
    if is_in_squeeze:
        breakout_fired = last['close'] > last[bbu_col]
        volume_ok = not params.get('volume_confirmation_enabled', True) or last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5
        rvol_ok = rvol >= bot_state.settings['liquidity_filters']['min_rvol']
        obv_rising = df['OBV'].iloc[-2] > df['OBV'].iloc[-3]
        if breakout_fired and rvol_ok and obv_rising:
            if params.get('volume_confirmation_enabled', True) and not volume_ok: return None
            return {"reason": "breakout_squeeze_pro", "type": "long"}
    return None

def find_support_resistance(high_prices, low_prices, window=10):
    supports, resistances = [], []
    if len(high_prices) < (2 * window + 1):
        return [], []
        
    for i in range(window, len(high_prices) - window):
        if high_prices[i] == max(high_prices[i-window:i+window+1]): resistances.append(high_prices[i])
        if low_prices[i] == min(low_prices[i-window:i+window+1]): supports.append(low_prices[i])
    if not supports and not resistances: return [], []

    def cluster_levels(levels, tolerance_percent=0.5):
        if not levels: return []
        clustered = []
        levels.sort()
        current_cluster = [levels[0]]
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] * 100 < tolerance_percent:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        return clustered

    return cluster_levels(supports), cluster_levels(resistances)

def analyze_sniper_pro(df, params, rvol, adx_value, exchange, symbol):
    try:
        compression_candles = int(params.get("compression_hours", 6) * 4) 
        if len(df) < compression_candles + 2:
            return None

        compression_df = df.iloc[-compression_candles-1:-1]
        highest_high = compression_df['high'].max()
        lowest_low = compression_df['low'].min()

        volatility = (highest_high - lowest_low) / lowest_low * 100 if lowest_low > 0 else float('inf')

        if volatility < params.get("max_volatility_percent", 12.0):
            last_candle = df.iloc[-2]
            if last_candle['close'] > highest_high:
                avg_volume = compression_df['volume'].mean()
                if last_candle['volume'] > avg_volume * 2:
                    return {"reason": "sniper_pro", "type": "long"}
    except Exception as e:
        logger.warning(f"Sniper Pro scan failed for {symbol}: {e}")
    return None

async def analyze_whale_radar(df, params, rvol, adx_value, exchange, symbol):
    try:
        threshold = params.get("wall_threshold_usdt", 30000)
        ob = await exchange.fetch_order_book(symbol, limit=20)
        if not ob or not ob.get('bids'): return None

        bids = ob.get('bids', [])
        total_bid_value = 0
        for item in bids[:10]:
            if isinstance(item, list) and len(item) >= 2:
                price, qty = item[0], item[1]
                total_bid_value += float(price) * float(qty)

        if total_bid_value > threshold:
            return {"reason": "whale_radar", "type": "long"}
    except Exception as e:
        logger.warning(f"Whale Radar scan failed for {symbol}: {e}")
    return None

async def analyze_support_rebound(df, params, rvol, adx_value, exchange, symbol):
    try:
        ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
        if not ohlcv_1h or len(ohlcv_1h) < 50: return None

        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        current_price = df_1h['close'].iloc[-1]

        supports, _ = find_support_resistance(df_1h['high'].to_numpy(), df_1h['low'].to_numpy(), window=5)
        if not supports: return None

        closest_support = max([s for s in supports if s < current_price], default=None)
        if not closest_support: return None

        if (current_price - closest_support) / closest_support * 100 < 1.0:
            last_candle_15m = df.iloc[-2]
            avg_volume_15m = df['volume'].rolling(window=20).mean().iloc[-2]

            if last_candle_15m['close'] > last_candle_15m['open'] and last_candle_15m['volume'] > avg_volume_15m * 1.5:
                return {"reason": "support_rebound", "type": "long"}
    except Exception as e:
        logger.warning(f"Support Rebound scan failed for {symbol}: {e}")
    return None


SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout,
    "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "support_rebound": analyze_support_rebound,
    "whale_radar": analyze_whale_radar,
    "sniper_pro": analyze_sniper_pro,
}

# =======================================================================================
# --- 🚑 [v6.2] New Helper Functions for Smart Sync & Rescue 🚑 ---
# =======================================================================================

async def _calculate_weighted_average_price(trades: list) -> tuple:
    """
    Calculates the weighted average price from a list of buy trades.
    It identifies the sequence of buys after the last sell to determine the current position.
    """
    if not trades:
        return 0, 0, 0

    # Sort trades by timestamp to process them chronologically
    trades.sort(key=lambda x: x['timestamp'])

    last_sell_index = -1
    for i, trade in enumerate(trades):
        if trade['side'] == 'sell':
            last_sell_index = i

    # All trades after the last sell are part of the current open position
    relevant_trades = trades[last_sell_index + 1:]
    buy_trades = [t for t in relevant_trades if t['side'] == 'buy']

    if not buy_trades:
        return 0, 0, 0 # No open position found

    total_cost = sum(t.get('cost', t['price'] * t['amount']) for t in buy_trades)
    total_amount = sum(t['amount'] for t in buy_trades)
    
    if total_amount == 0:
        return 0, 0, 0

    average_price = total_cost / total_amount
    first_trade_timestamp = datetime.fromtimestamp(buy_trades[0]['timestamp'] / 1000, tz=EGYPT_TZ)

    return average_price, total_amount, first_trade_timestamp

async def _reconstruct_and_save_trade(exchange, symbol: str, context: ContextTypes.DEFAULT_TYPE):
    """
    Fetches trade history for a symbol, reconstructs the trade, 
    and saves it to the database to be tracked by the bot.
    """
    try:
        my_trades = await exchange.fetch_my_trades(symbol, limit=100)
        if not my_trades:
            return f"لم يتم العثور على سجل تداول لـ `{symbol}`."

        avg_price, quantity, first_trade_time = await _calculate_weighted_average_price(my_trades)

        if avg_price == 0 or quantity == 0:
            return f"لم أتمكن من إعادة بناء صفقة مفتوحة لـ `{symbol}`. قد تكون مغلقة."

        settings = bot_state.settings
        current_atr = 0
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=settings['atr_period'] + 5)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.ta.atr(length=settings['atr_period'], append=True)
            atr_col = find_col(df.columns, f"ATRr_{settings['atr_period']}")
            if atr_col and not df[atr_col].empty:
                current_atr = df[atr_col].iloc[-1]
        except Exception as e:
            logger.warning(f"Could not fetch ATR for rescued trade {symbol}: {e}")

        if settings.get("use_dynamic_risk_management", False) and current_atr > 0:
            risk_per_unit = current_atr * settings['atr_sl_multiplier']
            stop_loss = avg_price - risk_per_unit
            take_profit = avg_price + (risk_per_unit * settings['risk_reward_ratio'])
        else: # Fallback to percentage if ATR fails
            sl_percent = 2.0
            tp_percent = 4.0
            stop_loss = avg_price * (1 - sl_percent / 100)
            take_profit = avg_price * (1 + tp_percent / 100)

        # Reconstruct signal object to log it
        rescued_signal = {
            'exchange': exchange.id.capitalize(),
            'symbol': symbol,
            'entry_price': avg_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'quantity': quantity,
            'entry_value_usdt': avg_price * quantity,
            'status': 'نشطة',
            'reason': 'Rescued/Imported',
            'trade_mode': 'real',
            'is_real_trade': True,
            'timestamp': first_trade_time.strftime('%Y-%m-%d %H:%M:%S'),
            'entry_order_id': 'imported',
            'exit_order_ids_json': '{}' # No exit orders initially
        }
        
        if trade_id := log_recommendation_to_db(rescued_signal):
            rescued_signal['trade_id'] = trade_id
            await send_telegram_message(context.bot, rescued_signal, is_new=True)
            return f"✅ تم استيراد ومتابعة صفقة `{symbol}` بنجاح!\n- **متوسط الدخول:** `${avg_price}`\n- **ID:** `{trade_id}`"
        else:
            return f"❌ فشل تسجيل الصفقة المستوردة `{symbol}` في قاعدة البيانات."

    except Exception as e:
        logger.error(f"Error during trade reconstruction for {symbol}: {e}", exc_info=True)
        return f"❌ حدث خطأ فادح أثناء محاولة استيراد `{symbol}`: {e}"

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
        credentials = {}
        if ex_id == 'binance': credentials = {'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET}
        elif ex_id == 'kucoin': credentials = {'apiKey': KUCOIN_API_KEY, 'secret': KUCOIN_API_SECRET, 'password': KUCOIN_API_PASSPHRASE}
        elif ex_id == 'gate': credentials = {'apiKey': GATE_API_KEY, 'secret': GATE_API_SECRET}
        elif ex_id == 'mexc': credentials = {'apiKey': MEXC_API_KEY, 'secret': MEXC_API_SECRET}
        elif ex_id == 'okx': credentials = {'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE}
        elif ex_id == 'bybit': credentials = {'apiKey': BYBIT_API_KEY, 'secret': BYBIT_API_SECRET}
        
        if credentials.get('apiKey') and 'YOUR_' not in credentials['apiKey']:
            params.update(credentials)
            try:
                private_exchange = getattr(ccxt_async, ex_id)(params)
                await private_exchange.load_markets()
                bot_state.exchanges[ex_id] = private_exchange
                logger.info(f"Connected to {ex_id} with PRIVATE client.")
            except Exception as e:
                logger.error(f"Failed to connect PRIVATE client for {ex_id}: {e}")
        
    await asyncio.gather(*[connect(ex_id) for ex_id in EXCHANGES_TO_SCAN])
    logger.info("All exchange connections initialized in BotState.")


async def aggregate_top_movers():
    # [v6.1] This function is completely rewritten for the new hybrid priority logic.
    all_tickers = []
    async def fetch(ex_id, ex):
        try:
            return [dict(t, exchange=ex_id) for t in (await ex.fetch_tickers()).values()]
        except Exception as e:
            logger.warning(f"Could not fetch tickers from {ex_id}: {e}")
            return []
    
    results = await asyncio.gather(*[fetch(ex_id, ex) for ex_id, ex in bot_state.public_exchanges.items()])
    for res in results:
        all_tickers.extend(res)
        
    settings = bot_state.settings
    excluded_bases = settings['stablecoin_filter']['exclude_bases']
    min_volume = settings['liquidity_filters']['min_quote_volume_24h_usd']
    
    # 1. Initial Filtering
    usdt_tickers = [
        t for t in all_tickers if t.get('symbol') and t['symbol'].upper().endswith('/USDT') and 
        t['symbol'].split('/')[0] not in excluded_bases and 
        t.get('quoteVolume') and t['quoteVolume'] >= min_volume and 
        not any(k in t['symbol'].upper() for k in ['UP','DOWN','3L','3S','BEAR','BULL'])
    ]

    # 2. Group by symbol
    grouped_symbols = defaultdict(list)
    for ticker in usdt_tickers:
        grouped_symbols[ticker['symbol']].append(ticker)

    # 3. Apply Priority Logic
    final_list = []
    real_trading_exchanges = {ex for ex, enabled in settings.get("real_trading_per_exchange", {}).items() if enabled}
    
    for symbol, tickers in grouped_symbols.items():
        # Priority 1: Check for exchanges with real trading enabled
        real_trade_options = [t for t in tickers if t['exchange'] in real_trading_exchanges]
        
        if real_trade_options:
            # If multiple real trading exchanges have the same coin, pick the one with higher volume
            best_option = max(real_trade_options, key=lambda t: t.get('quoteVolume', 0))
            final_list.append(best_option)
        else:
            # Priority 2: If no real trading, pick the one with the highest volume
            best_option = max(tickers, key=lambda t: t.get('quoteVolume', 0))
            final_list.append(best_option)

    # 4. Sort the final unique list by volume and take the top N
    final_list.sort(key=lambda t: t.get('quoteVolume', 0), reverse=True)
    top_markets = final_list[:settings['top_n_symbols_by_volume']]
    
    logger.info(f"Aggregated markets. Found {len(all_tickers)} tickers -> Post-filter: {len(usdt_tickers)} -> Selected top {len(top_markets)} unique pairs with priority logic.")
    bot_state.status_snapshot['markets_found'] = len(top_markets)
    return top_markets


async def get_higher_timeframe_trend(exchange, symbol, ma_period):
    try:
        ohlcv_htf = await exchange.fetch_ohlcv(symbol, HIGHER_TIMEFRAME, limit=ma_period + 5)
        if len(ohlcv_htf) < ma_period: return None, "Not enough HTF data"
        df_htf = pd.DataFrame(ohlcv_htf, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_htf[f'SMA_{ma_period}'] = ta.sma(df_htf['close'], length=ma_period)
        last_candle = df_htf.iloc[-1]
        is_bullish = last_candle['close'] > last_candle[f'SMA_{ma_period}']
        return is_bullish, "Bullish" if is_bullish else "Bearish"
    except Exception as e:
        return None, f"Error: {e}"

async def worker(queue, results_list, settings, failure_counter):
    while not queue.empty():
        market_info = await queue.get()
        symbol = market_info.get('symbol', 'N/A')
        exchange_id = market_info.get('exchange') # Keep the id as a string
        exchange = bot_state.public_exchanges.get(exchange_id)
        if not exchange or not settings.get('active_scanners'):
            queue.task_done()
            continue
        try:
            liq_filters, vol_filters, ema_filters = settings['liquidity_filters'], settings['volatility_filters'], settings['ema_trend_filter']

            orderbook = await exchange.fetch_order_book(symbol, limit=20)
            if not orderbook or not orderbook['bids'] or not orderbook['asks'] or not orderbook['bids'][0] or not orderbook['asks'][0]:
                logger.debug(f"Reject {symbol}: Could not fetch valid order book."); continue
            
            best_bid, best_ask = orderbook['bids'][0][0], orderbook['asks'][0][0]
            if best_bid <= 0: logger.debug(f"Reject {symbol}: Invalid bid price."); continue

            spread_percent = ((best_ask - best_bid) / best_bid) * 100
            if spread_percent > liq_filters['max_spread_percent']:
                logger.debug(f"Reject {symbol}: High Spread ({spread_percent:.2f}% > {liq_filters['max_spread_percent']}%)"); continue

            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            if len(ohlcv) < ema_filters.get('ema_period', 200) + 1:
                logger.debug(f"Skipping {symbol}: Not enough data ({len(ohlcv)} candles) for EMA calculation."); continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)

            df['volume_sma'] = ta.sma(df['volume'], length=liq_filters['rvol_period'])
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] <= 0:
                logger.debug(f"Skipping {symbol}: Invalid SMA volume."); continue

            rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
            if rvol < liq_filters['min_rvol']:
                logger.debug(f"Reject {symbol}: Low RVOL ({rvol:.2f} < {liq_filters['min_rvol']})"); continue

            atr_col_name = f"ATRr_{vol_filters['atr_period_for_filter']}"
            df.ta.atr(length=vol_filters['atr_period_for_filter'], append=True)
            last_close = df['close'].iloc[-2]
            if last_close <= 0: logger.debug(f"Skipping {symbol}: Invalid close price."); continue

            atr_percent = (df[atr_col_name].iloc[-2] / last_close) * 100 if find_col(df.columns, 'ATRr_') else 0
            if atr_percent < vol_filters['min_atr_percent']:
                logger.debug(f"Reject {symbol}: Low ATR% ({atr_percent:.2f}% < {vol_filters['min_atr_percent']}%)"); continue

            ema_col_name = f"EMA_{ema_filters['ema_period']}"
            df.ta.ema(length=ema_filters['ema_period'], append=True)
            if ema_col_name not in df.columns or pd.isna(df[ema_col_name].iloc[-2]):
                logger.debug(f"Skipping {symbol}: EMA_{ema_filters['ema_period']} could not be calculated.")
                continue

            if ema_filters['enabled'] and last_close < df[ema_col_name].iloc[-2]:
                logger.debug(f"Reject {symbol}: Below EMA{ema_filters['ema_period']}"); continue

            if settings.get('use_master_trend_filter'):
                is_htf_bullish, reason = await get_higher_timeframe_trend(exchange, symbol, settings['master_trend_filter_ma_period'])
                if not is_htf_bullish:
                    logger.debug(f"HTF Trend Filter FAILED for {symbol}: {reason}"); continue

            df.ta.adx(append=True)
            adx_col = find_col(df.columns, 'ADX_')
            adx_value = df[adx_col].iloc[-2] if adx_col and pd.notna(df[adx_col].iloc[-2]) else 0
            if settings.get('use_master_trend_filter') and adx_value < settings['master_adx_filter_level']:
                logger.debug(f"ADX Filter FAILED for {symbol}: {adx_value:.2f} < {settings['master_adx_filter_level']}"); continue

            confirmed_reasons = []
            for scanner_name in settings['active_scanners']:
                scanner_func = SCANNERS.get(scanner_name)
                if not scanner_func: continue
                
                scanner_params = settings.get(scanner_name, {})
                if asyncio.iscoroutinefunction(scanner_func):
                    result = await scanner_func(df.copy(), scanner_params, rvol, adx_value, exchange, symbol)
                else:
                    result = scanner_func(df.copy(), scanner_params, rvol, adx_value, exchange, symbol)

                if result and result.get("type") == "long":
                    confirmed_reasons.append(result['reason'])


            if confirmed_reasons and len(confirmed_reasons) >= settings.get("min_signal_strength", 1):
                reason_str = ' + '.join(confirmed_reasons)
                entry_price = df.iloc[-2]['close']
                df.ta.atr(length=settings['atr_period'], append=True)
                atr_col = find_col(df.columns, f"ATRr_{settings['atr_period']}")
                current_atr = df.iloc[-2].get(atr_col, 0) if atr_col else 0

                if settings.get("use_dynamic_risk_management", False) and current_atr > 0:
                    risk_per_unit = current_atr * settings['atr_sl_multiplier']
                    stop_loss, take_profit = entry_price - risk_per_unit, entry_price + (risk_per_unit * settings['risk_reward_ratio'])
                else:
                    sl_percent = settings.get("stop_loss_percentage", 2.0)
                    tp_percent = settings.get("take_profit_percentage", 4.0)
                    stop_loss, take_profit = entry_price * (1 - sl_percent / 100), entry_price * (1 + tp_percent / 100)

                tp_percent_calc, sl_percent_calc = ((take_profit - entry_price) / entry_price * 100), ((entry_price - stop_loss) / entry_price * 100)
                min_filters = settings['min_tp_sl_filter']
                if tp_percent_calc >= min_filters['min_tp_percent'] and sl_percent_calc >= min_filters['min_sl_percent']:
                    results_list.append({"symbol": symbol, "exchange": exchange_id.capitalize(), "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "timestamp": df.index[-2], "reason": reason_str, "strength": len(confirmed_reasons)})
                else:
                    logger.debug(f"Reject {symbol} Signal: Small TP/SL (TP: {tp_percent_calc:.2f}%, SL: {sl_percent_calc:.2f}%)")

        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded for {symbol} on {exchange_id}. Pausing...: {e}")
            await asyncio.sleep(10)
        except ccxt.NetworkError as e:
            logger.warning(f"Network error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"CRITICAL ERROR in worker for {symbol} on {exchange_id}: {e}", exc_info=True)
            failure_counter[0] += 1
        finally:
            queue.task_done()

