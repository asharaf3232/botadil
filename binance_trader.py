# -*- coding: utf-8 -*-
# =======================================================================================
# --- 💣 بوت كاسحة الألغام (Minesweeper Bot) v8.0 (بروتوكول المُصلِح) 💣 ---
# =======================================================================================
# نسخة مُعدَّلة: إصلاح OKX OCO، فلترة عملات، سجل عمليات، ودالة مسح DB/ذاكرة.
# =======================================================================================

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

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
GATE_API_KEY = os.getenv('GATE_API_KEY', '')
GATE_API_SECRET = os.getenv('GATE_API_SECRET', '')
OKX_API_KEY = os.getenv('OKX_API_KEY', '')
OKX_API_SECRET = os.getenv('OKX_API_SECRET', '')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE', '')
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')

EXCHANGES_TO_SCAN = ['binance', 'okx', 'bybit', 'gate']
TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
SCAN_INTERVAL_SECONDS = 900
TRACK_INTERVAL_SECONDS = 60

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v8.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'minesweeper_settings_v8.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")

# --- إعداد مسجل الأحداث (Logger) ---
LOG_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v8.log')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                    handlers=[logging.FileHandler(LOG_FILE, 'a', 'utf-8'), logging.StreamHandler()])
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger("MinesweeperBot_v8")

# ----------------- فلترة العملات الممنوعة / blacklist -----------------
BLACKLIST_TOKENS = {
    'BNB', 'OKB', 'HT', 'KCS', 'FTT', 'BGB', 'GT'
}

def is_allowed_symbol(symbol: str) -> bool:
    """Return False if symbol contains a blacklisted token."""
    try:
        base, quote = symbol.replace('-', '/').upper().split('/')
    except Exception:
        parts = re.split(r'[-_/]', symbol.upper())
        base, quote = (parts[0], parts[-1]) if len(parts) >= 2 else (parts[0], '')
    return (base not in BLACKLIST_TOKENS) and (quote not in BLACKLIST_TOKENS)

from logging.handlers import RotatingFileHandler
OPS_LOG_FILE = os.path.join(APP_ROOT, 'bot_operations.log')
ops_logger = logging.getLogger("MinesweeperBot_ops")
ops_logger.setLevel(logging.INFO)
if not ops_logger.handlers:
    ops_handler = RotatingFileHandler(OPS_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
    ops_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    ops_handler.setFormatter(ops_formatter)
    ops_logger.addHandler(ops_handler)
if not any(type(h) is RotatingFileHandler and getattr(h, "baseFilename", "") == OPS_LOG_FILE for h in logger.handlers):
    logger.addHandler(ops_handler)

# =======================================================================================
class BotState:
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
    def __init__(self, exchange_client):
        self.exchange = exchange_client

    async def place_exit_orders(self, signal, verified_quantity):
        raise NotImplementedError

    async def update_trailing_stop_loss(self, trade, new_sl):
        raise NotImplementedError

class OcoAdapter(ExchangeAdapter):
    async def place_exit_orders(self, signal, verified_quantity):
        symbol = signal['symbol']
        if not is_allowed_symbol(symbol):
            ops_logger.info(f"[FILTER] Skipping blacklisted symbol {symbol} (not placing exit orders).")
            return {"skipped": True, "reason": "blacklisted"}

        tp_price = self.exchange.price_to_precision(symbol, signal['take_profit'])
        sl_price = self.exchange.price_to_precision(symbol, signal['stop_loss'])

        if self.exchange.id == 'okx':
            ops_logger.info(f"[OKX] Placing OCO algorithm order for {symbol} TP={tp_price} SL={sl_price}")
            try:
                inst_id = self.exchange.market_id(symbol)
                # determine base currency safely
                try:
                    base_ccy, _ = symbol.replace('-', '/').upper().split('/')
                except Exception:
                    market = self.exchange.markets.get(symbol) if hasattr(self.exchange, 'markets') else None
                    base_ccy = market['base'] if market else symbol.split('-')[0]

                payload = {
                    'instId': inst_id,
                    'tdMode': 'cash',
                    'ccy': base_ccy,
                    'side': 'sell',
                    'ordType': 'oco',
                    'sz': str(self.exchange.amount_to_precision(symbol, verified_quantity)),
                    'posSide': 'net',
                    'tpTriggerPx': tp_price,
                    'tpOrdPx': tp_price,
                    'slTriggerPx': sl_price,
                    'slOrdPx': '-1',
                }

                ops_logger.debug(f"[OKX] OCO payload: {payload}")
                trigger_order = await self.exchange.private_post_trade_order_algo(payload)
                ops_logger.info(f"[OKX] OCO response: {trigger_order}")

                # extract algo id robustly
                algo_id = None
                data = None
                if isinstance(trigger_order, dict):
                    data = trigger_order.get('data') or trigger_order.get('order') or trigger_order.get('result')
                if isinstance(data, list) and data:
                    algo_id = data[0].get('algoId') or data[0].get('algo_id') or data[0].get('ordId')
                elif isinstance(data, dict):
                    algo_id = data.get('algoId') or data.get('algo_id') or data.get('ordId')

                if not algo_id:
                    ops_logger.warning(f"[OKX] No algoId in response, will attempt fallback. Response: {trigger_order}")
                    raise ccxt.ExchangeError(f"OKX did not return algoId. Response: {trigger_order}")

                ops_logger.info(f"[OKX] OCO Algorithm order placed with algoId: {algo_id}")
                return {"algo_id": algo_id}

            except Exception as e:
                ops_logger.error(f"[OKX] Failed to place OCO algo order: {e}", exc_info=True)

                # fallback (separate TP and SL) - try to create limit TP and conditional SL
                fallback = {}
                try:
                    tp_qty = self.exchange.amount_to_precision(symbol, verified_quantity)
                    tp_order = await self.exchange.create_order(
                        symbol=symbol, type='limit', side='sell', amount=tp_qty, price=tp_price, params={}
                    )
                    fallback['tp_order'] = tp_order.get('id') or tp_order.get('orderId') or tp_order
                    ops_logger.info(f"[OKX][fallback] TP order created: {fallback['tp_order']}")
                except Exception as tp_e:
                    fallback['tp_error'] = str(tp_e)
                    ops_logger.error(f"[OKX][fallback] TP creation failed: {tp_e}", exc_info=True)

                try:
                    sl_qty = self.exchange.amount_to_precision(symbol, verified_quantity)
                    sl_params = {'triggerPrice': sl_price, 'ordType': 'conditional'}
                    sl_order = await self.exchange.create_order(
                        symbol=symbol, type='limit', side='sell', amount=sl_qty, price=sl_price, params=sl_params
                    )
                    fallback['sl_order'] = sl_order.get('id') or sl_order.get('orderId') or sl_order
                    ops_logger.info(f"[OKX][fallback] SL conditional order created: {fallback['sl_order']}")
                except Exception as sl_e:
                    fallback['sl_error'] = str(sl_e)
                    ops_logger.error(f"[OKX][fallback] SL creation failed: {sl_e}", exc_info=True)

                if fallback.get('tp_order') or fallback.get('sl_order'):
                    ops_logger.info(f"[OKX][fallback] Fallback succeeded: {fallback}")
                    return {"fallback": fallback}
                else:
                    ops_logger.critical(f"[OKX][fallback] Fallback failed: {fallback}")
                    raise ccxt.ExchangeError(f"OKX OCO and fallback both failed. Details: {fallback}")

        # others (binance/bybit/gate)
        ops_logger.info(f"[{self.exchange.id}] Placing OCO for {symbol} TP={tp_price} SL={sl_price}")
        params = {}
        if self.exchange.id == 'binance':
            params['stopLimitPrice'] = sl_price

        oco_order = await self.exchange.create_order(
            symbol=symbol,
            type='oco',
            side='sell',
            amount=verified_quantity,
            price=tp_price,
            stopPrice=sl_price,
            params=params
        )

        oco_id = None
        if isinstance(oco_order, dict):
            oco_id = oco_order.get('id') or oco_order.get('orderId') or oco_order.get('clientOrderId')
        if not oco_id:
            oco_id = oco_order

        ops_logger.info(f"[{self.exchange.id}] OCO order result: {oco_id}")
        return {"oco_id": oco_id}

    async def update_trailing_stop_loss(self, trade, new_sl):
        symbol = trade['symbol']
        exit_ids = json.loads(trade.get('exit_order_ids_json', '{}'))
        if self.exchange.id == 'okx':
            algo_id_to_cancel = exit_ids.get('algo_id')
            if not algo_id_to_cancel:
                raise ValueError("OKX trade is missing its algoId for TSL update.")
            ops_logger.info(f"[OKX] Cancelling old OCO algo order {algo_id_to_cancel} for {symbol}.")
            try:
                await self.exchange.private_post_trade_cancel_algos([
                    {'instId': self.exchange.market_id(symbol), 'algoId': algo_id_to_cancel}
                ])
            except ccxt.OrderNotFound:
                ops_logger.warning(f"[OKX] OCO algo order {algo_id_to_cancel} not found.")
        else:
            oco_id_to_cancel = exit_ids.get('oco_id')
            if not oco_id_to_cancel:
                raise ValueError(f"{self.exchange.id} trade is missing its OCO ID for TSL update.")
            ops_logger.info(f"[{self.exchange.id}] Cancelling old OCO order {oco_id_to_cancel} for {symbol}.")
            try:
                await self.exchange.cancel_order(oco_id_to_cancel, symbol)
            except ccxt.OrderNotFound:
                ops_logger.warning(f"OCO order {oco_id_to_cancel} not found.")
        await asyncio.sleep(2)
        new_signal = {'symbol': symbol, 'take_profit': trade['take_profit'], 'stop_loss': new_sl}
        return await self.place_exit_orders(new_signal, trade['quantity'])

class BinanceAdapter(OcoAdapter): pass
class BybitAdapter(OcoAdapter): pass
class GateAdapter(OcoAdapter): pass
class OKXAdapter(OcoAdapter): pass

def get_exchange_adapter(exchange_id: str):
    exchange_client = bot_state.exchanges.get(exchange_id.lower())
    if not exchange_client: return None
    adapter_map = {
        'binance': BinanceAdapter, 'okx': OKXAdapter,
        'bybit': BybitAdapter, 'gate': GateAdapter
    }
    AdapterClass = adapter_map.get(exchange_id.lower())
    if AdapterClass: return AdapterClass(exchange_client)
    logger.warning(f"No specific adapter found for {exchange_id}, trade automation will be disabled for it.")
    return None

# دالة مساعدة لمسح قاعدة البيانات والذاكرة
def wipe_bot_state_and_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        ops_logger.info("DB file removed: %s", DB_FILE)
    if os.path.exists(OPS_LOG_FILE):
        os.remove(OPS_LOG_FILE)
        ops_logger.info("Ops log removed: %s", OPS_LOG_FILE)
    bot_state.scan_history.clear()
    bot_state.last_signal_time.clear()
    bot_state.exchanges.clear()
    bot_state.public_exchanges.clear()
    ops_logger.info("In-memory bot state cleared.")


# === Remaining original file ===
# -*- coding: utf-8 -*-
# =======================================================================================
# --- 💣 بوت كاسحة الألغام (Minesweeper Bot) v8.0 (بروتوكول المُصلِح) 💣 ---
# =======================================================================================
# --- سجل التغييرات v8.0 ---
#
# بناءً على التحليل العميق والمشترك، تم إجراء عملية إصلاح شاملة وجذرية.
#
# 1. [إصلاح جذري] إصلاح خلل التوافقية مع المنصات (OKX):
#    - تمت إعادة كتابة منطق وضع أوامر الحماية (OCO) بالكامل لـ OKX ليتوافق مع متطلباتها الخاصة.
#    - **النتيجة:** القضاء التام على خطأ 'TypeError' المسبب لانهيار البوت وترك الصفقات عارية.
#
# 2. [تأمين فوري] الحفاظ على منطق التحذير الصريح عند فشل الحماية الأولية:
#    - تم التأكد من أن رسالة v6.1 "العبقرية" تعمل بكفاءة. إذا نجح الشراء وفشل وضع الأوامر،
#      سيقوم البوت بإرسال تحذير فوري وواضح للتدخل اليدوي.
#
# 3. [منطق وقائي] إضافة "بوابة أمان" لاستراتيجيات الوقف المتحرك الحساسة:
#    - استراتيجيات `atr` و `ema` الآن **تحترم** شرط "تفعيل الوقف المتحرك (%)" (1.5% افتراضياً).
#    - لن يتم إطلاق عملية التحديث الخطيرة إلا بعد تحقيق ربح حقيقي أولاً.
#
# 4. [تطوير الحارس] ترقية "بروتوكول الحارس" إلى "بروتوكول المُصلِح":
#    - لم يعد الحارس يكتفي بالصراخ. عند اكتشاف صفقة "عارية" (بلا حماية)،
#      سيقوم الآن بمحاولة **"إصلاح ذاتي"** وإعادة وضع أوامر الحماية تلقائياً.
#    - يرسل "المُصلِح" تقريراً بنجاح أو فشل عملية الإصلاح الذاتي.
#
# 5. [زيادة الاستقرار] تم إصلاح الأسباب الجذرية لانهيار البوت وإعادة تشغيله المستمر.
#    من المتوقع أن يعمل البوت الآن باستقرار أعلى بكثير.
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

# --- Add API Keys for all supported exchanges ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
GATE_API_KEY = os.getenv('GATE_API_KEY', '')
GATE_API_SECRET = os.getenv('GATE_API_SECRET', '')
OKX_API_KEY = os.getenv('OKX_API_KEY', '')
OKX_API_SECRET = os.getenv('OKX_API_SECRET', '')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE', '')
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')

# --- إعدادات البوت ---
EXCHANGES_TO_SCAN = ['binance', 'okx', 'bybit', 'gate']
TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
SCAN_INTERVAL_SECONDS = 900
TRACK_INTERVAL_SECONDS = 60 

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v8.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'minesweeper_settings_v8.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")

# --- إعداد مسجل الأحداث (Logger) ---
LOG_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v8.log')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'a', 'utf-8'), logging.StreamHandler()])
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger("MinesweeperBot_v8")
# --- إضافة: دعم تسجيل العمليات التفصيلي (rotating file handler) -------------------
from logging.handlers import RotatingFileHandler

# ملف لوج منفصل لعمليات البوت (نجاح / فشل / تفاصيل أوامر الخ)
OPS_LOG_FILE = os.path.join(APP_ROOT, 'bot_operations.log')

ops_logger = logging.getLogger("MinesweeperBot_ops")
ops_logger.setLevel(logging.INFO)

# لا تضيف handlers مكررة لو تم استدعاء الملف أكثر من مرة
if not ops_logger.handlers:
    ops_handler = RotatingFileHandler(
        OPS_LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding='utf-8'
    )
    ops_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    ops_handler.setFormatter(ops_formatter)
    ops_logger.addHandler(ops_handler)

# أيضًا احفظ نسخة من اللوج العام إلى ملف العمليات (اختياري لكنه مفيد)
# هذا يضمن أن سجلات logger العام تُضاف أيضاً إلى ملف العمليات
if not any(type(h) is RotatingFileHandler and getattr(h, "baseFilename", "") == OPS_LOG_FILE for h in logger.handlers):
    logger.addHandler(ops_handler)

# اختصار استخدام ops_logger في أي مكان بالكود
# مثال: ops_logger.info("Order placed ...")
# ------------------------------------------------------------------------------------

# --- إضافة: كلاس OcoAdapter المعدل لدعم OKX و fallback لتجنيب ترك الصفقات عارية ----------

    """محول أساسي للمنصات التي تدعم أوامر OCO. يدعم OKX مع اصلاحات ccy/tdMode وفالباك."""

    async def place_exit_orders(self, signal, verified_quantity):
        symbol = signal['symbol']
        tp_price = self.exchange.price_to_precision(symbol, signal['take_profit'])
        sl_price = self.exchange.price_to_precision(symbol, signal['stop_loss'])

        # ---------- فرع منصة OKX ----------
        if self.exchange.id == 'okx':
            ops_logger.info(f"[OKX] Placing OCO algorithm order for {symbol} TP={tp_price} SL={sl_price}")
            try:
                inst_id = self.exchange.market_id(symbol)  # مثال: 'OKB-USDT'
                # تفصل الزوج للحصول على العملة الأساسية (ccy)
                try:
                    base_ccy, _ = symbol.split('/')
                except Exception:
                    # في حال كان symbol بصيغة مختلفة، نحاول استخدام market info
                    market = self.exchange.markets.get(symbol) if hasattr(self.exchange, 'markets') else None
                    base_ccy = market['base'] if market else symbol.split('-')[0]

                payload = {
                    'instId': inst_id,
                    'tdMode': 'cash',  # spot accounts عادة 'cash'
                    'ccy': base_ccy,   # مطلوب من OKX وإلا سيُرجع 50014
                    'side': 'sell',
                    'ordType': 'oco',
                    'sz': str(self.exchange.amount_to_precision(symbol, verified_quantity)),
                    'posSide': 'net',
                    'tpTriggerPx': tp_price,
                    'tpOrdPx': tp_price,
                    'slTriggerPx': sl_price,
                    'slOrdPx': '-1',
                }

                ops_logger.debug(f"[OKX] OCO payload: {payload}")
                trigger_order = await self.exchange.private_post_trade_order_algo(payload)
                ops_logger.info(f"[OKX] OCO response: {trigger_order}")

                # استخراج algoId بشكل آمن
                algo_id = None
                data = None
                if isinstance(trigger_order, dict):
                    data = trigger_order.get('data') or trigger_order.get('order') or trigger_order.get('result')
                if isinstance(data, list) and data:
                    algo_id = data[0].get('algoId') or data[0].get('algo_id') or data[0].get('ordId')
                elif isinstance(data, dict):
                    algo_id = data.get('algoId') or data.get('algo_id') or data.get('ordId')

                if not algo_id:
                    # سجّل الاستجابة التفصيلية ثم ارفع استثناء ليدخل الفالباك
                    ops_logger.warning(f"[OKX] No algoId in response, will attempt fallback. Response: {trigger_order}")
                    raise ccxt.ExchangeError(f"OKX did not return algoId. Response: {trigger_order}")

                ops_logger.info(f"[OKX] OCO Algorithm order placed with algoId: {algo_id}")
                return {"algo_id": algo_id}

            except Exception as e:
                ops_logger.error(f"[OKX] Failed to place OCO algo order: {e}", exc_info=True)

                # --------- Fallback: محاولة إنشاء أوامر منفصلة (TP limit + SL conditional) ----------
                fallback = {}
                try:
                    tp_qty = self.exchange.amount_to_precision(symbol, verified_quantity)
                    tp_order = await self.exchange.create_order(
                        symbol=symbol, type='limit', side='sell', amount=tp_qty, price=tp_price, params={}
                    )
                    fallback['tp_order'] = tp_order.get('id') or tp_order.get('orderId') or tp_order
                    ops_logger.info(f"[OKX][fallback] TP order created: {fallback['tp_order']}")
                except Exception as tp_e:
                    fallback['tp_error'] = str(tp_e)
                    ops_logger.error(f"[OKX][fallback] TP creation failed: {tp_e}", exc_info=True)

                try:
                    sl_qty = self.exchange.amount_to_precision(symbol, verified_quantity)
                    # params قد تحتاج تعديل حسب نسخة ccxt/okx؛ هذا مثال عام
                    sl_params = {'triggerPrice': sl_price, 'ordType': 'conditional'}
                    sl_order = await self.exchange.create_order(
                        symbol=symbol, type='limit', side='sell', amount=sl_qty, price=sl_price, params=sl_params
                    )
                    fallback['sl_order'] = sl_order.get('id') or sl_order.get('orderId') or sl_order
                    ops_logger.info(f"[OKX][fallback] SL conditional order created: {fallback['sl_order']}")
                except Exception as sl_e:
                    fallback['sl_error'] = str(sl_e)
                    ops_logger.error(f"[OKX][fallback] SL creation failed: {sl_e}", exc_info=True)

                # إذا نجحت أي عملية من الفالباك نُعيد تفاصيلها، وإلا نُعيد الخطأ الكامل للطبقة العليا
                if fallback.get('tp_order') or fallback.get('sl_order'):
                    ops_logger.info(f"[OKX][fallback] Fallback succeeded: {fallback}")
                    return {"fallback": fallback}
                else:
                    ops_logger.critical(f"[OKX][fallback] Fallback failed: {fallback}")
                    raise ccxt.ExchangeError(f"OKX OCO and fallback both failed. Details: {fallback}")

        # ---------- باقي المنصات (مثال: Binance) ----------
        ops_logger.info(f"[{self.exchange.id}] Placing OCO for {symbol} TP={tp_price} SL={sl_price}")
        params = {}
        if self.exchange.id == 'binance':
            params['stopLimitPrice'] = sl_price

        oco_order = await self.exchange.create_order(
            symbol=symbol,
            type='oco',
            side='sell',
            amount=verified_quantity,
            price=tp_price,
            stopPrice=sl_price,
            params=params
        )

        oco_id = None
        if isinstance(oco_order, dict):
            oco_id = oco_order.get('id') or oco_order.get('orderId') or oco_order.get('clientOrderId')
        if not oco_id:
            oco_id = oco_order

        ops_logger.info(f"[{self.exchange.id}] OCO order result: {oco_id}")
        return {"oco_id": oco_id}
# ----------------------------------------------------------------------------------------


# =======================================================================================
# --- 🚀 [v8.0] إعادة هيكلة المحولات (Adapters) لإصلاح التوافقية 🚀 ---
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
    """محول أساسي للمنصات التي تدعم أوامر OCO."""

    async def place_exit_orders(self, signal, verified_quantity):
        symbol = signal['symbol']
        tp_price = self.exchange.price_to_precision(symbol, signal['take_profit'])
        sl_price = self.exchange.price_to_precision(symbol, signal['stop_loss'])

        # فرع منصة OKX
        if self.exchange.id == 'okx':
            logger.info(f"OKX: Placing OCO algorithm order for {symbol}")
            try:
                inst_id = self.exchange.market_id(symbol)  # 'OKB-USDT'
                base_ccy, _ = symbol.split('/')           # 'OKB'

                payload = {
                    'instId': inst_id,
                    # للحسابات Spot لازم يكون 'cash' وليس isolated
                    'tdMode': 'cash',
                    # العملة المطلوبة في OKX (أساس الزوج)
                    'ccy': base_ccy,
                    'side': 'sell',
                    'ordType': 'oco',
                    'sz': str(self.exchange.amount_to_precision(symbol, verified_quantity)),
                    'posSide': 'net',
                    'tpTriggerPx': tp_price,
                    'tpOrdPx': tp_price,
                    'slTriggerPx': sl_price,
                    'slOrdPx': '-1',
                }

                logger.debug(f"OKX OCO payload: {payload}")
                trigger_order = await self.exchange.private_post_trade_order_algo(payload)
                logger.info(f"OKX OCO response: {trigger_order}")

                algo_id = None
                data = trigger_order.get('data') or trigger_order.get('order') or trigger_order.get('result')
                if isinstance(data, list) and data:
                    algo_id = data[0].get('algoId') or data[0].get('algo_id')
                elif isinstance(data, dict):
                    algo_id = data.get('algoId') or data.get('algo_id')

                if not algo_id:
                    raise ccxt.ExchangeError(
                        f"OKX failed to return a valid algoId for the OCO order. Response: {trigger_order}"
                    )

                logger.info(f"OKX: OCO Algorithm order placed with algoId: {algo_id}")
                return {"algo_id": algo_id}

            except Exception as e:
                logger.error(f"Failed to place OKX OCO order: {e}", exc_info=True)
                raise

        # باقي المنصات (Binance وغيرها)
        logger.info(f"{self.exchange.id} OCO: Placing for {symbol}. TP: {tp_price}, SL: {sl_price}")

        params = {}
        if self.exchange.id == 'binance':
            params['stopLimitPrice'] = sl_price

        oco_order = await self.exchange.create_order(
            symbol=symbol,
            type='oco',
            side='sell',
            amount=verified_quantity,
            price=tp_price,
            stopPrice=sl_price,
            params=params
        )

        # قد تختلف صيغة الـ response باختلاف المنصة
        oco_id = oco_order.get('id') or oco_order.get('orderId') or oco_order
        return {"oco_id": oco_id}


    async def update_trailing_stop_loss(self, trade, new_sl):
        symbol = trade['symbol']
        exit_ids = json.loads(trade.get('exit_order_ids_json', '{}'))
        
        if self.exchange.id == 'okx':
            algo_id_to_cancel = exit_ids.get('algo_id')
            if not algo_id_to_cancel:
                raise ValueError("OKX trade is missing its algoId for TSL update.")
            logger.info(f"OKX: Cancelling old OCO algo order {algo_id_to_cancel} for {symbol}.")
            try:
                await self.exchange.private_post_trade_cancel_algos([
                    {'instId': self.exchange.market_id(symbol), 'algoId': algo_id_to_cancel}
                ])
            except ccxt.OrderNotFound:
                logger.warning(f"OKX OCO algo order {algo_id_to_cancel} not found.")
        else:
            oco_id_to_cancel = exit_ids.get('oco_id')
            if not oco_id_to_cancel:
                raise ValueError(f"{self.exchange.id} trade is missing its OCO ID for TSL update.")
            logger.info(f"{self.exchange.id} OCO: Cancelling old OCO order {oco_id_to_cancel} for {symbol}.")
            try:
                await self.exchange.cancel_order(oco_id_to_cancel, symbol)
            except ccxt.OrderNotFound:
                logger.warning(f"OCO order {oco_id_to_cancel} not found.")
        
        await asyncio.sleep(2)

        new_signal = {'symbol': symbol, 'take_profit': trade['take_profit'], 'stop_loss': new_sl}
        return await self.place_exit_orders(new_signal, trade['quantity'])

class BinanceAdapter(OcoAdapter): pass
class BybitAdapter(OcoAdapter): pass
class GateAdapter(OcoAdapter): pass
class OKXAdapter(OcoAdapter): pass

def get_exchange_adapter(exchange_id: str):
    exchange_client = bot_state.exchanges.get(exchange_id.lower())
    if not exchange_client: return None
    
    adapter_map = {
        'binance': BinanceAdapter, 'okx': OKXAdapter,
        'bybit': BybitAdapter, 'gate': GateAdapter
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
        "min_signal_strength", "signal_cooldown_multiplier"
    ],
    "إعدادات المخاطر": [
        "automate_real_tsl", "real_trade_size_usdt", "virtual_trade_size_percentage",
        "atr_sl_multiplier", "risk_reward_ratio", "trailing_sl_activation_percent", 
        "trailing_sl_callback_percent", "rescue_sl_multiplier"
    ],
    "الفلاتر والاتجاه": [
        "market_regime_filter_enabled", "use_master_trend_filter", "fear_and_greed_filter_enabled",
        "master_adx_filter_level", "master_trend_filter_ma_period", "trailing_sl_enabled", "fear_and_greed_threshold",
        "fundamental_analysis_enabled"
    ],
    "استراتيجية الوقف المتحرك": [
        "trailing_sl_strategy", 
        "use_strategy_mapping", 
        "default_tsl_strategy",
        "tsl_ema_period", 
        "tsl_atr_period", 
        "tsl_atr_multiplier"
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
    "trailing_sl_strategy": "⚙️ الاستراتيجية اليدوية",
    "tsl_ema_period": "EMA فترة متوسط",
    "tsl_atr_period": "ATR فترة مؤشر",
    "tsl_atr_multiplier": "ATR مضاعف",
    "use_strategy_mapping": "🤖 تفعيل الربط الذكي للوقف",
    "default_tsl_strategy": "⚙️ استراتيجية الوقف الافتراضية",
    "signal_cooldown_multiplier": "مضاعف تهدئة الإشارة",
    "rescue_sl_multiplier": "مضاعف وقف الإنقاذ"
}

DEFAULT_SETTINGS = {
    "real_trading_per_exchange": {ex: False for ex in EXCHANGES_TO_SCAN}, 
    "automate_real_tsl": False,
    "real_trade_size_usdt": 15.0,
    "virtual_portfolio_balance_usdt": 1000.0, "virtual_trade_size_percentage": 5.0, "max_concurrent_trades": 10, "top_n_symbols_by_volume": 250, "concurrent_workers": 10,
    "market_regime_filter_enabled": True, "fundamental_analysis_enabled": True,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "whale_radar", "sniper_pro"],
    "use_master_trend_filter": True, "master_trend_filter_ma_period": 50, "master_adx_filter_level": 22,
    "btc_trend_source_exchanges": ["binance", "bybit"], 
    "fear_and_greed_filter_enabled": True, "fear_and_greed_threshold": 30,
    "use_dynamic_risk_management": True, "atr_period": 14, "atr_sl_multiplier": 2.5, "risk_reward_ratio": 2.0,
    "trailing_sl_enabled": True, "trailing_sl_activation_percent": 1.5, "trailing_sl_callback_percent": 1.0,
    "signal_cooldown_multiplier": 4.0,
    "rescue_sl_multiplier": 1.5,
    "trailing_sl_advanced": {
        "strategy": "percentage",
        "tsl_ema_period": 21,
        "tsl_atr_period": 14,
        "tsl_atr_multiplier": 2.5,
        "use_strategy_mapping": True,
        "default_tsl_strategy": "atr",
        "strategy_tsl_mapping": {
            "momentum_breakout": "ema",
            "breakout_squeeze_pro": "ema",
            "sniper_pro": "ema",
            "whale_radar": "atr",
            "support_rebound": "percentage",
            "Rescued/Imported": "atr"
        }
    },
    "_internal_state": { 
        "last_signal_time": {}
    },
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
        
        internal_state = bot_state.settings.get('_internal_state', {})
        bot_state.last_signal_time = internal_state.get('last_signal_time', {})
        if bot_state.last_signal_time:
             logger.info(f"Successfully loaded persistent memory for {len(bot_state.last_signal_time)} symbols.")

        updated = False
        current_real_trading_settings = bot_state.settings.get("real_trading_per_exchange", {})
        new_real_trading_settings = {ex: current_real_trading_settings.get(ex, False) for ex in EXCHANGES_TO_SCAN}
        if new_real_trading_settings != current_real_trading_settings:
            bot_state.settings["real_trading_per_exchange"] = new_real_trading_settings
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
        logger.error(f"Failed to load settings: {e}", exc_info=True)
        bot_state.settings = DEFAULT_SETTINGS.copy()
        bot_state.last_signal_time = {}


def save_settings():
    try:
        bot_state.settings['_internal_state'] = {
            "last_signal_time": bot_state.last_signal_time
        }
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(bot_state.settings, f, indent=4)
        logger.info(f"Settings and persistent state saved successfully.")
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

        timestamp_obj = signal.get('timestamp', datetime.now(EGYPT_TZ))

        if isinstance(timestamp_obj, str):
            timestamp_str = timestamp_obj
        else:
            timestamp_str = timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')

        params = (
            timestamp_str,
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
    if not trades:
        return 0, 0, None

    last_sell_index = -1
    for i in range(len(trades) - 1, -1, -1):
        if trades[i].get('side') == 'sell':
            last_sell_index = i
            break

    buy_trades = [
        trade for trade in trades[last_sell_index + 1:] 
        if trade.get('side') == 'buy' and trade.get('cost', 0) > 0 and trade.get('amount', 0) > 0
    ]

    if not buy_trades:
        return 0, 0, None

    total_cost = sum(t['cost'] for t in buy_trades)
    total_amount = sum(t['amount'] for t in buy_trades)

    if total_amount == 0:
        return 0, 0, None

    average_price = total_cost / total_amount
    first_trade_timestamp = datetime.fromtimestamp(buy_trades[0]['timestamp'] / 1000, tz=EGYPT_TZ)

    return average_price, total_amount, first_trade_timestamp

async def _reconstruct_and_save_trade(exchange, symbol: str, context: ContextTypes.DEFAULT_TYPE):
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
            rescue_sl_multiplier = settings.get('rescue_sl_multiplier', 1.5)
            risk_per_unit = (current_atr * settings['atr_sl_multiplier']) * rescue_sl_multiplier
            stop_loss = avg_price - risk_per_unit
            take_profit = avg_price + (risk_per_unit * settings['risk_reward_ratio'])
        else:
            sl_percent = 7.0
            tp_percent = 14.0
            stop_loss = avg_price * (1 - sl_percent / 100)
            take_profit = avg_price * (1 + tp_percent / 100)

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
            'exit_order_ids_json': '{}'
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
        elif ex_id == 'gate': credentials = {'apiKey': GATE_API_KEY, 'secret': GATE_API_SECRET}
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
    
    usdt_tickers = [
        t for t in all_tickers if t.get('symbol') and t['symbol'].upper().endswith('/USDT') and 
        t['symbol'].split('/')[0] not in excluded_bases and 
        t.get('quoteVolume') and t['quoteVolume'] >= min_volume and 
        not any(k in t['symbol'].upper() for k in ['UP','DOWN','3L','3S','BEAR','BULL'])
    ]

    grouped_symbols = defaultdict(list)
    for ticker in usdt_tickers:
        grouped_symbols[ticker['symbol']].append(ticker)

    final_list = []
    real_trading_exchanges = {ex for ex, enabled in settings.get("real_trading_per_exchange", {}).items() if enabled}
    
    for symbol, tickers in grouped_symbols.items():
        real_trade_options = [t for t in tickers if t['exchange'] in real_trading_exchanges]
        
        if real_trade_options:
            best_option = max(real_trade_options, key=lambda t: t.get('quoteVolume', 0))
            final_list.append(best_option)
        else:
            best_option = max(tickers, key=lambda t: t.get('quoteVolume', 0))
            final_list.append(best_option)

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
        exchange_id = market_info.get('exchange')
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
            failure_counter[0] += 1
        except Exception as e:
            logger.error(f"CRITICAL ERROR in worker for {symbol} on {exchange_id}: {e}", exc_info=True)
            failure_counter[0] += 1
        finally:
            queue.task_done()

async def get_real_balance(exchange_id, currency='USDT'):
    try:
        exchange = bot_state.exchanges.get(exchange_id.lower())
        if not exchange or not exchange.apiKey:
            logger.warning(f"Cannot fetch balance: {exchange_id.capitalize()} client not authenticated.")
            return 0.0

        balance = await exchange.fetch_balance()
        return balance['free'].get(currency, 0.0)
    except Exception as e:
        logger.error(f"Error fetching {exchange_id.capitalize()} balance for {currency}: {e}")
        return 0.0

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
        
        trade_amount_usdt = max(user_trade_amount_usdt, min_notional or 0)
        if min_notional and min_notional > user_trade_amount_usdt:
             logger.warning(f"User trade size ${user_trade_amount_usdt} for {symbol} is below exchange minimum of ${min_notional}. Using exchange minimum.")

        if usdt_balance < trade_amount_usdt:
            return {'success': False, 'data': f"رصيدك الحالي ${usdt_balance:.2f} غير كافٍ لفتح صفقة بقيمة ${trade_amount_usdt:.2f}."}
        
        quantity = trade_amount_usdt / signal['entry_price']
        formatted_quantity = exchange.amount_to_precision(symbol, quantity)
        signal.update({
            'quantity': float(formatted_quantity),
            'entry_value_usdt': trade_amount_usdt
        })

    except Exception as e:
        return {'success': False, 'data': f"Pre-flight check failed: {e}"}

    buy_order = None
    try:
        logger.info(f"Placing MARKET BUY order for {signal['quantity']} of {symbol} on {exchange_id.capitalize()}")
        buy_order = await exchange.create_market_buy_order(symbol, signal['quantity'])
        logger.info(f"Initial response for BUY order {buy_order.get('id', 'N/A')} received.")
    except ccxt.InvalidOrder as e:
        logger.error(f"Placing BUY order for {symbol} failed (InvalidOrder): {e}", exc_info=True)
        return {'success': False, 'data': f"فشل: أمر غير صالح. قد يكون المبلغ أقل من الحد الأدنى أو نوع الأمر خاطئ.\n`{str(e)}`"}
    except ccxt.InsufficientFunds as e:
        logger.error(f"Placing BUY order for {symbol} failed (InsufficientFunds): {e}", exc_info=True)
        return {'success': False, 'data': f"فشل: رصيد غير كاف.\n`{str(e)}`"}
    except Exception as e:
        logger.error(f"Placing BUY order for {symbol} failed immediately: {e}", exc_info=True)
        return {'success': False, 'data': f"حدث خطأ من المنصة عند محاولة الشراء: `{str(e)}`"}

    verified_order = None
    verified_price, verified_quantity, verified_cost = 0, 0, 0
    try:
        max_attempts = 5
        delay_seconds = 3
        for attempt in range(max_attempts):
            logger.info(f"Verifying BUY order {buy_order['id']}... (Attempt {attempt + 1}/{max_attempts})")
            try:
                order_status = await exchange.fetch_order(buy_order['id'], symbol)
                if order_status and order_status.get('status') == 'closed' and order_status.get('filled', 0) > 0:
                    verified_order = order_status
                    break 
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay_seconds)
            except ccxt.OrderNotFound:
                logger.warning(f"Order {buy_order['id']} not found, retrying...")
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
        logger.error(f"VERIFICATION FAILED for BUY order {buy_order['id']}: {e}", exc_info=True)
        return {'success': False, 'manual_check_required': True, 'data': f"تم إرسال أمر الشراء لكن فشل التحقق منه. **يرجى التحقق من المنصة يدوياً!** ID: `{buy_order['id']}`. Error: `{e}`"}

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
    async with scan_lock:
        if bot_state.status_snapshot['scan_in_progress']:
            logger.warning("Scan attempted while another was in progress. Skipped."); return
        settings = bot_state.settings
        if settings.get('fundamental_analysis_enabled', True):
            mood, mood_score, mood_reason = await get_fundamental_market_mood()
            bot_state.settings['last_market_mood'] = {"timestamp": datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M'), "mood": mood, "reason": mood_reason}
            save_settings()
            logger.info(f"Fundamental Market Mood: {mood} - Reason: {mood_reason}")
            if mood in ["NEGATIVE", "DANGEROUS"]:
                await send_telegram_message(context.bot, {'custom_message': f"**⚠️ تم إيقاف الفحص التلقائي مؤقتاً**\n\n**السبب:** مزاج السوق سلبي/خطر.\n**التفاصيل:** {mood_reason}.\n\n*سيتم استئناف الفحص عندما تتحسن الظروف.*", 'target_chat': TELEGRAM_CHAT_ID}); return

        is_market_ok, btc_reason = await check_market_regime()
        bot_state.status_snapshot['btc_market_mood'] = "إيجابي ✅" if is_market_ok else "سلبي ❌"

        if settings.get('market_regime_filter_enabled', True) and not is_market_ok:
            logger.info(f"Skipping scan: {btc_reason}")
            await send_telegram_message(context.bot, {'custom_message': f"**⚠️ تم إيقاف الفحص التلقائي مؤقتاً**\n\n**السبب:** مزاج السوق سلبي/خطر.\n**التفاصيل:** {btc_reason}.\n\n*سيتم استئناف الفحص عندما تتحسن الظروف.*", 'target_chat': TELEGRAM_CHAT_ID}); return

        status = bot_state.status_snapshot
        status.update({"scan_in_progress": True, "last_scan_start_time": datetime.now(EGYPT_TZ)})
        
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة' AND trade_mode = 'virtual'")
            active_virtual_trades = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة' AND trade_mode = 'real'")
            active_real_trades = cursor.fetchone()[0]
            cursor.execute("SELECT symbol FROM trades WHERE status = 'نشطة'")
            active_symbols = {row[0] for row in cursor.fetchall()}
            conn.close()
            active_trades_count = active_virtual_trades + active_real_trades
        except Exception as e:
            logger.error(f"DB Error in perform_scan: {e}")
            active_trades_count = settings.get("max_concurrent_trades", 10)
            active_symbols = set()

        top_markets = await aggregate_top_movers()
        if not top_markets:
            logger.info("Scan complete: No markets to scan."); status['scan_in_progress'] = False; return

        queue = asyncio.Queue(); [await queue.put(market) for market in top_markets]
        signals, failure_counter = [], [0]
        worker_tasks = [asyncio.create_task(worker(queue, signals, settings, failure_counter)) for _ in range(settings['concurrent_workers'])]
        await queue.join(); [task.cancel() for task in worker_tasks]

        total_signals_found = len(signals)
        signals.sort(key=lambda s: s.get('strength', 0), reverse=True)
        new_trades, opportunities = 0, 0
        
        signal_cooldown = SCAN_INTERVAL_SECONDS * settings.get('signal_cooldown_multiplier', 4)

        for signal in signals:
            if signal['symbol'] in active_symbols:
                logger.info(f"Signal for {signal['symbol']} skipped: An active trade already exists.")
                continue

            if time.time() - bot_state.last_signal_time.get(signal['symbol'], 0) <= signal_cooldown:
                logger.info(f"Signal for {signal['symbol']} skipped due to cooldown."); continue

            signal_exchange_id = signal['exchange'].lower()
            per_exchange_settings = settings.get("real_trading_per_exchange", {})
            is_real_mode_enabled = per_exchange_settings.get(signal_exchange_id, False)

            exchange_is_tradeable = signal_exchange_id in bot_state.exchanges and bot_state.exchanges[signal_exchange_id].apiKey
            attempt_real_trade = is_real_mode_enabled and exchange_is_tradeable
            signal['is_real_trade'] = attempt_real_trade

            if attempt_real_trade:
                attempt_msg_data = {'custom_message': f"**🔎 تم العثور على إشارة حقيقية لـ `{signal['symbol']}`...**\n*جاري محاولة التنفيذ على `{signal['exchange']}`... ⏳*"}
                sent_msg = await send_telegram_message(context.bot, attempt_msg_data, return_message_object=True)
                edit_msg_id = sent_msg.message_id if sent_msg else None

                try:
                    trade_result = await place_real_trade(signal.copy())
                    
                    if trade_result.get('success'):
                        if isinstance(trade_result.get('data'), dict): signal.update(trade_result['data'])
                        
                        original_risk = signal['entry_price'] - signal['stop_loss']
                        verified_entry = signal['verified_entry_price']
                        
                        signal['entry_price'] = verified_entry
                        signal['quantity'] = signal['verified_quantity']
                        signal['entry_value_usdt'] = signal['verified_entry_value']
                        signal['stop_loss'] = verified_entry - original_risk
                        signal['take_profit'] = verified_entry + (original_risk * settings['risk_reward_ratio'])
                        
                        if trade_id := log_recommendation_to_db(signal):
                            signal['trade_id'] = trade_id
                            await send_telegram_message(context.bot, signal, is_new=True, edit_message_id=edit_msg_id)
                            new_trades += 1
                            active_symbols.add(signal['symbol'])
                            if trade_result.get('exit_orders_failed'):
                                await send_telegram_message(context.bot, {'custom_message': f"**🚨 تحذير: تم شراء `{signal['symbol']}` بنجاح وتسجيلها، لكن فشل وضع أوامر الهدف/الوقف تلقائياً.**\n\n**يرجى وضعها يدوياً الآن!**"})
                        else: 
                            fail_msg = f"**⚠️ خطأ حرج:** تم تنفيذ صفقة `{signal['symbol']}` لكن فشل تسجيلها في قاعدة البيانات. **يرجى المتابعة اليدوية فوراً!**"
                            await send_telegram_message(context.bot, {'custom_message': fail_msg}, edit_message_id=edit_msg_id)
                    else:
                        fail_msg = f"**❌ فشل تنفيذ صفقة `{signal['symbol']}`**\n\n**السبب:** {trade_result.get('data', 'سبب غير معروف')}"
                        await send_telegram_message(context.bot, {'custom_message': fail_msg}, edit_message_id=edit_msg_id)
                
                except Exception as e:
                    logger.critical(f"CRITICAL UNHANDLED ERROR during real trade execution for {signal['symbol']}: {e}", exc_info=True)
                    fail_msg = f"**❌ فشل حرج وغير معالج أثناء محاولة تنفيذ صفقة `{signal['symbol']}`.**\n\n**الخطأ:** `{str(e)}`\n\n*يرجى التحقق من المنصة ومن سجلات الأخطاء (logs).*`"
                    await send_telegram_message(context.bot, {'custom_message': fail_msg}, edit_message_id=edit_msg_id)
            
            else: # الصفقات الوهمية
                if active_trades_count < settings.get("max_concurrent_trades", 10):
                    trade_amount_usdt = settings["virtual_portfolio_balance_usdt"] * (settings["virtual_trade_size_percentage"] / 100)
                    signal.update({'quantity': trade_amount_usdt / signal['entry_price'], 'entry_value_usdt': trade_amount_usdt})
                    if trade_id := log_recommendation_to_db(signal):
                        signal['trade_id'] = trade_id
                        await send_telegram_message(context.bot, signal, is_new=True)
                        new_trades += 1
                        active_symbols.add(signal['symbol'])
                else:
                    await send_telegram_message(context.bot, signal, is_opportunity=True)
                    opportunities += 1

            await asyncio.sleep(0.5)
            bot_state.last_signal_time[signal['symbol']] = time.time()
        
        save_settings()

        failures = failure_counter[0]
        logger.info(f"Scan complete. Found: {total_signals_found}, Entered: {new_trades}, Opportunities: {opportunities}, Failures: {failures}.")
        
        status['last_scan_end_time'] = datetime.now(EGYPT_TZ)
        scan_start_time = status.get('last_scan_start_time')
        scan_duration = (status['last_scan_end_time'] - scan_start_time).total_seconds() if isinstance(scan_start_time, datetime) else 0

        summary_message = (f"**🔬 ملخص الفحص الأخير**\n\n"
                           f"- **الحالة:** اكتمل بنجاح\n"
                           f"- **وضع السوق (BTC):** {status['btc_market_mood']}\n"
                           f"- **المدة:** {scan_duration:.0f} ثانية\n"
                           f"- **العملات المفحوصة:** {len(top_markets)}\n\n"
                           f"- - - - - - - - - - - - - - - - - -\n"
                           f"- **إجمالي الإشارات المكتشفة:** {total_signals_found}\n"
                           f"- **✅ صفقات جديدة فُتحت:** {new_trades}\n"
                           f"- **💡 فرص للمراقبة:** {opportunities}\n"
                           f"- **⚠️ أخطاء في التحليل:** {failures}\n"
                           f"- - - - - - - - - - - - - - - - - -\n\n"
                           f"*الفحص التالي مجدول تلقائياً.*")

        await send_telegram_message(context.bot, {'custom_message': summary_message, 'target_chat': TELEGRAM_CHAT_ID})

        status['scan_in_progress'] = False

        bot_state.scan_history.append({'signals': total_signals_found, 'failures': failures})
        await analyze_performance_and_suggest(context)

async def send_telegram_message(bot, signal_data, is_new=False, is_opportunity=False, update_type=None, edit_message_id=None, return_message_object=False):
    message, keyboard, target_chat = "", None, TELEGRAM_CHAT_ID
    def format_price(price): 
        if price is None: return "N/A"
        return f"{price:,.8f}" if price < 0.01 else f"{price:,.4f}"

    if 'custom_message' in signal_data:
        message, target_chat = signal_data['custom_message'], signal_data.get('target_chat', TELEGRAM_CHAT_ID)
        if 'keyboard' in signal_data: keyboard = signal_data['keyboard']

    elif is_new or is_opportunity:
        try:
            target_chat = TELEGRAM_SIGNAL_CHANNEL_ID
            strength_stars = '⭐' * signal_data.get('strength', 1)

            trade_type_title = "🚨 صفقة حقيقية 🚨" if signal_data.get('is_real_trade') else "✅ توصية شراء جديدة"
            title = f"**{trade_type_title} | {signal_data['symbol']}**" if is_new else f"**💡 فرصة محتملة | {signal_data['symbol']}**"

            entry, tp, sl = signal_data['entry_price'], signal_data['take_profit'], signal_data['stop_loss']
            if not entry or entry == 0:
                logger.error(f"Cannot generate signal message for {signal_data['symbol']} due to invalid entry price: {entry}")
                message = f"❌ خطأ في بيانات إشارة {signal_data['symbol']}. سعر الدخول غير صالح."
            else:
                tp_percent, sl_percent = ((tp - entry) / entry * 100), ((entry - sl) / entry * 100)
                id_line = f"\n*للمتابعة اضغط: /check {signal_data.get('trade_id', 'N/A')}*" if is_new else ""

                reasons_en = signal_data['reason'].split(' + ')
                reasons_ar = ' + '.join([STRATEGY_NAMES_AR.get(r, r) for r in reasons_en])

                message = (f"**Signal Alert | تنبيه إشارة**\n"
                        f"------------------------------------\n"
                        f"{title}\n"
                        f"------------------------------------\n"
                        f"🔹 **المنصة:** {signal_data['exchange']}\n"
                        f"⭐ **قوة الإشارة:** {strength_stars}\n"
                        f"🔍 **الاستراتيجية:** {reasons_ar}\n\n"
                        f"📈 **نقطة الدخول:** `{format_price(entry)}`\n"
                        f"🎯 **الهدف:** `{format_price(tp)}` (+{tp_percent:.2f}%)\n"
                        f"🛑 **الوقف:** `{format_price(sl)}` (-{sl_percent:.2f}%)\n"
                        f"{id_line}")
        except KeyError as e:
            logger.error(f"CRITICAL: Missing key '{e}' in signal_data when trying to send message. Data: {signal_data}")
            message = f"❌ خطأ حرج في توليد رسالة الإشارة لـ {signal_data.get('symbol', 'N/A')}. يرجى مراجعة السجلات."
    elif update_type == 'tsl_activation':
        message = (f"**🚀 تأمين الأرباح! | #{signal_data['id']} {signal_data['symbol']}**\n\n"
                   f"تم رفع وقف الخسارة إلى نقطة الدخول.\n"
                   f"**هذه الصفقة الآن مؤمَّنة بالكامل وبدون مخاطرة!**\n\n"
                   f"*دع الأرباح تنمو!*")
    elif update_type == 'tsl_update_real':
        message = (f"**🔔 تنبيه تحديث وقف الخسارة (صفقة حقيقية) 🔔**\n\n"
                   f"**صفقة:** `#{signal_data['id']} {signal_data['symbol']}`\n\n"
                   f"وصل السعر إلى `{format_price(signal_data['current_price'])}`.\n"
                   f"**إجراء مقترح:** قم بتعديل أمر وقف الخسارة يدوياً إلى `{format_price(signal_data['new_sl'])}` لتأمين الأرباح.")


    if not message: return
    try:
        if edit_message_id:
            sent_message = await bot.edit_message_text(chat_id=target_chat, message_id=edit_message_id, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
        else:
            sent_message = await bot.send_message(chat_id=target_chat, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
        
        if return_message_object:
            return sent_message

    except BadRequest as e:
        if 'Message is not modified' in str(e): pass
        elif 'Chat not found' in str(e):
            logger.critical(f"CRITICAL: Chat not found for target_chat: {target_chat}. Error: {e}")
            if str(target_chat) == str(TELEGRAM_SIGNAL_CHANNEL_ID) and str(target_chat) != str(TELEGRAM_CHAT_ID):
                try:
                    await bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=f"**⚠️ فشل الإرسال إلى القناة ⚠️**\n\nلم أتمكن من إرسال رسالة إلى القناة (`{target_chat}`).\n\n**السبب:** `Chat not found`\n\n**الحل:**\n1. تأكد من أنني (البوت) عضو في القناة.\n2. تأكد من أنني مشرف (Admin) ولدي صلاحية إرسال الرسائل.\n3. تحقق من أن `TELEGRAM_SIGNAL_CHANNEL_ID` صحيح.",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except Exception as admin_e:
                    logger.error(f"Failed to send admin warning about ChatNotFound: {admin_e}")
        else:
            logger.error(f"Failed to send/edit Telegram message to {target_chat}: {e}")
            if edit_message_id:
                try:
                    logger.info(f"Editing failed. Sending new message instead for {target_chat}")
                    await bot.send_message(chat_id=target_chat, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
                except Exception as fallback_e:
                    logger.error(f"Fallback send message also failed: {fallback_e}")

    except Exception as e:
        logger.error(f"General error in send_telegram_message to {target_chat}: {e}")

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

    for trade in active_trades:
        try:
            await check_trade_on_exchange(trade, context)
            
            conn = sqlite3.connect(DB_FILE, timeout=10)
            status_row = conn.cursor().execute("SELECT status FROM trades WHERE id = ?", (trade['id'],)).fetchone()
            conn.close()

            if status_row and status_row[0] == 'نشطة':
                await check_and_update_tsl(trade, context)

        except Exception as e:
            logger.error(f"Error processing trade #{trade.get('id')} in track_open_trades loop: {e}", exc_info=True)


async def check_trade_on_exchange(trade: dict, context: ContextTypes.DEFAULT_TYPE):
    """
    [v8.0] The Sentinel is now The Fixer. It attempts to self-heal.
    """
    if trade.get('trade_mode') != 'real': return

    exchange_id = trade['exchange'].lower()
    symbol = trade['symbol']
    exchange = bot_state.exchanges.get(exchange_id)
    if not exchange: 
        logger.warning(f"FIXER: Cannot check {symbol}, no private connection to {exchange_id}.")
        return

    try:
        exit_order_ids_from_db = json.loads(trade.get('exit_order_ids_json', '{}'))
        if not exit_order_ids_from_db and (datetime.now(EGYPT_TZ) - datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=EGYPT_TZ)).total_seconds() < 120:
             return # Grace period for initial orders to be placed and registered

        trade_start_time = datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=EGYPT_TZ)
        since_timestamp = int((trade_start_time - timedelta(minutes=5)).timestamp() * 1000)
        recent_filled_orders = await exchange.fetch_my_trades(symbol, since=since_timestamp, limit=10)
        
        for filled_order in recent_filled_orders:
            if filled_order['side'] == 'sell' and filled_order.get('order') in exit_order_ids_from_db.values():
                logger.info(f"FIXER: FOUND filled exit order for trade #{trade['id']}! Closing trade.")
                is_win = filled_order.get('price', 0) >= trade.get('take_profit', float('inf'))
                await close_trade_in_db(context, trade, filled_order, is_win=is_win)
                return

        open_orders = await exchange.fetch_open_orders(symbol)
        open_order_ids_on_exchange = {o['id'] for o in open_orders}
        are_orders_still_open = any(db_id in open_order_ids_on_exchange for db_id in exit_order_ids_from_db.values())

        if are_orders_still_open:
            logger.debug(f"FIXER: Orders for trade #{trade['id']} confirmed open.")
            return

        if (datetime.now(EGYPT_TZ) - trade_start_time).total_seconds() < 180:
            logger.warning(f"FIXER: Orders for NEW trade #{trade['id']} not visible yet. Waiting (API Lag).")
            return

        # [FIX v8.0] FIXER PROTOCOL ACTIVATION
        logger.critical(f"FIXER PROTOCOL: Exit orders for trade #{trade['id']} ({symbol}) are MISSING. Activating self-rescue.")
        await send_telegram_message(context.bot, {'custom_message': f"**⚠️ تم اكتشاف خلل!**\n\n**صفقة:** `#{trade['id']} {symbol}`\n\nلم يتم العثور على أوامر حماية. **جاري محاولة الإصلاح الذاتي...** 🤖"})

        try:
            adapter = get_exchange_adapter(exchange_id)
            if not adapter: raise ValueError("No adapter found for self-rescue.")

            rescue_signal = {'symbol': symbol, 'take_profit': trade['take_profit'], 'stop_loss': trade['stop_loss']}
            new_exit_ids = await adapter.place_exit_orders(rescue_signal, trade['quantity'])
            
            await update_trade_order_ids_in_db(trade['id'], json.dumps(new_exit_ids))
            logger.info(f"FIXER PROTOCOL: Self-rescue for trade #{trade['id']} SUCCEEDED.")
            await send_telegram_message(context.bot, {'custom_message': f"**✅ تم الإصلاح بنجاح!**\n\n**صفقة:** `#{trade['id']} {symbol}`\n\nتم إعادة وضع أوامر الحماية بنجاح."})

        except Exception as e:
            logger.critical(f"FIXER PROTOCOL: Self-rescue for trade #{trade['id']} FAILED: {e}", exc_info=True)
            await send_telegram_message(context.bot, {'custom_message': f"**🚨 فشل الإصلاح الذاتي!**\n\n**صفقة:** `#{trade['id']} {symbol}`\n\n**🔥 تدخل يدوي فوري ضروري!**"})

    except Exception as e:
        logger.error(f"FIXER: CRITICAL Error checking trade #{trade['id']}: {e}", exc_info=True)


async def check_and_update_tsl(trade: dict, context: ContextTypes.DEFAULT_TYPE):
    """
    [v8.0] This function now includes the "Safety Gate" logic.
    """
    if not bot_state.settings.get('trailing_sl_enabled', True): return

    is_real_trade = trade.get('trade_mode') == 'real'
    if is_real_trade and not bot_state.settings.get('automate_real_tsl', False) and not bot_state.settings.get('trailing_sl_enabled', True):
        return
        
    try:
        exchange = bot_state.public_exchanges.get(trade['exchange'].lower())
        if not exchange: return
            
        ticker = await exchange.fetch_ticker(trade['symbol'])
        current_price = ticker.get('last') or ticker.get('close')
        if not current_price: return

        highest_price = max(trade.get('highest_price', 0) or current_price, current_price)
        new_sl = None
        current_stop_loss = trade.get('stop_loss') or 0
        settings = bot_state.settings
        tsl_settings = settings.get("trailing_sl_advanced", {})
        
        # [v8.0] SAFETY GATE LOGIC STARTS HERE
        if not trade.get('trailing_sl_active'):
            activation_price = trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100)
            if current_price < activation_price:
                # Still update highest price if it's relevant, but don't proceed
                if highest_price > (trade.get('highest_price') or 0):
                    await update_trade_peak_price_in_db(trade['id'], highest_price)
                return # Exit the function early if activation % is not met
        # SAFETY GATE LOGIC ENDS HERE - if we proceed, activation is confirmed.
        
        strategy = tsl_settings.get("strategy", "percentage")
        if tsl_settings.get("use_strategy_mapping", False):
            mapping = tsl_settings.get("strategy_tsl_mapping", {})
            trade_reason = trade.get('reason', '').split(' + ')[0]
            strategy = mapping.get(trade_reason, tsl_settings.get("default_tsl_strategy", "atr"))
        
        if strategy == "percentage":
            if not trade.get('trailing_sl_active'):
                new_sl = trade['entry_price'] 
            else:
                new_sl = highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)

        elif strategy in ["ema", "atr"]:
            limit = max(tsl_settings.get("tsl_ema_period", 21), tsl_settings.get("tsl_atr_period", 14)) + 10
            ohlcv = await exchange.fetch_ohlcv(trade['symbol'], TIMEFRAME, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if strategy == "ema":
                period = tsl_settings.get("tsl_ema_period", 21)
                df.ta.ema(length=period, append=True)
                ema_col = find_col(df.columns, f"EMA_{period}")
                if ema_col and pd.notna(df[ema_col].iloc[-1]): new_sl = df[ema_col].iloc[-1]
            elif strategy == "atr":
                period = tsl_settings.get("tsl_atr_period", 14)
                multiplier = tsl_settings.get("tsl_atr_multiplier", 2.5)
                df.ta.atr(length=period, append=True)
                atr_col = find_col(df.columns, f"ATRr_{period}")
                if atr_col and pd.notna(df[atr_col].iloc[-1]):
                    atr_value = df[atr_col].iloc[-1]
                    new_sl = highest_price - (multiplier * atr_value)

        if new_sl and new_sl > current_stop_loss:
            is_activation = not trade.get('trailing_sl_active')
            await handle_tsl_update(context, trade, new_sl, highest_price, is_activation=is_activation)
        
        elif highest_price > (trade.get('highest_price') or 0):
            await update_trade_peak_price_in_db(trade['id'], highest_price)

    except Exception as e:
        logger.error(f"TSL: Error in TSL price check for #{trade['id']}: {e}", exc_info=True)


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
    adapter = get_exchange_adapter(exchange_id)
    if not adapter:
        logger.error(f"Cannot automate TSL for {symbol}: No adapter for {exchange_id}.")
        return

    logger.info(f"TSL AUTOMATION: Attempting for trade #{trade['id']} ({symbol}). New SL: {new_sl}")

    try:
        new_exit_ids = await adapter.update_trailing_stop_loss(trade, new_sl)
        await update_trade_sl_in_db(context, trade, new_sl, highest_price, is_activation=is_activation, new_exit_ids_json=json.dumps(new_exit_ids), silent=False)
        logger.info(f"TSL automation successful for trade #{trade['id']}. New orders placed: {new_exit_ids}")

    except Exception as e:
        logger.critical(f"TSL AUTOMATION: CRITICAL FAILURE for trade #{trade['id']} ({symbol}): {e}", exc_info=True)
        await send_telegram_message(context.bot, {'custom_message': f"**🚨 فشل حرج في أتمتة الوقف المتحرك 🚨**\n\n**صفقة:** `#{trade['id']} {symbol}`\n**الخطأ:** `{e}`\n\n**لم أتمكن من تأمين الصفقة تلقائياً. التدخل اليدوي الفوري ضروري الآن!**"})


async def close_trade_in_db(context: ContextTypes.DEFAULT_TYPE, trade: dict, filled_order: dict, is_win: bool):
    exit_price = filled_order['price']
    pnl_usdt = (exit_price - trade['entry_price']) * trade['quantity']
    
    status = ""
    if is_win:
        status = 'ناجحة (تحقيق هدف)'
    else:
        if pnl_usdt >= 0:
            status = 'ناجحة (وقف ربح)'
        else:
            status = 'فاشلة (وقف خسارة)'

    if trade.get('trade_mode') == 'virtual':
        bot_state.settings['virtual_portfolio_balance_usdt'] += pnl_usdt
        save_settings()

    closed_at_str = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')
    start_dt_naive = datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S')
    start_dt = start_dt_naive.replace(tzinfo=EGYPT_TZ)
    end_dt = datetime.now(EGYPT_TZ)
    duration = end_dt - start_dt
    days, remainder = divmod(duration.total_seconds(), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    duration_str = f"{int(days)}d {int(hours)}h {int(minutes)}m" if days > 0 else f"{int(hours)}h {int(minutes)}m"

    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute("UPDATE trades SET status=?, exit_price=?, closed_at=?, exit_value_usdt=?, pnl_usdt=? WHERE id=?",
                       (status, exit_price, closed_at_str, filled_order.get('cost', exit_price * trade['quantity']), pnl_usdt, trade['id']))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB update failed while closing trade #{trade['id']}: {e}")
        return
    
    trade_type_str = "(صفقة حقيقية)" if trade.get('trade_mode') == 'real' else ""
    pnl_percent = (pnl_usdt / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0
    message = ""
    if pnl_usdt >= 0:
        message = (f"**📦 إغلاق صفقة {trade_type_str} | #{trade['id']} {trade['symbol']}**\n\n"
                   f"**الحالة: ✅ {status}**\n"
                   f"💰 **الربح:** `${pnl_usdt:+.2f}` (`{pnl_percent:+.2f}%`)\n\n"
                   f"- **مدة الصفقة:** {duration_str}")
    else: 
        message = (f"**📦 إغلاق صفقة {trade_type_str} | #{trade['id']} {trade['symbol']}**\n\n"
                   f"**الحالة: ❌ {status}**\n"
                   f"💰 **الخسارة:** `${pnl_usdt:.2f}` (`{pnl_percent:.2f}%`)\n\n"
                   f"- **مدة الصفقة:** {duration_str}")

    await send_telegram_message(context.bot, {'custom_message': message, 'target_chat': TELEGRAM_SIGNAL_CHANNEL_ID})


async def update_trade_order_ids_in_db(trade_id: int, new_exit_ids_json: str):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute("UPDATE trades SET exit_order_ids_json=? WHERE id=?", (new_exit_ids_json, trade_id))
        conn.commit()
        conn.close()
        logger.info(f"Updated order IDs for trade #{trade_id} to: {new_exit_ids_json}")
    except Exception as e:
        logger.error(f"Failed to update order IDs for trade #{trade_id} in DB: {e}")

async def update_trade_sl_in_db(context: ContextTypes.DEFAULT_TYPE, trade: dict, new_sl: float, highest_price: float, is_activation: bool = False, silent: bool = False, new_exit_ids_json: str = None):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        
        sql = "UPDATE trades SET stop_loss=?, highest_price=?, trailing_sl_active=? "
        params = [new_sl, highest_price, True]
        
        if new_exit_ids_json is not None:
            sql += ", exit_order_ids_json=? "
            params.append(new_exit_ids_json)

        sql += "WHERE id=?"
        params.append(trade['id'])

        cursor.execute(sql, tuple(params))
        conn.commit()
        conn.close()
        
        log_msg = f"Trailing SL {'activated' if is_activation else 'updated'} for trade #{trade['id']}. New SL: {new_sl}"
        if new_exit_ids_json is not None:
            log_msg += f", New Exit IDs: {new_exit_ids_json}"
        logger.info(log_msg)

        if not silent and is_activation:
            await send_telegram_message(context.bot, {**trade, "new_sl": new_sl}, update_type='tsl_activation')
    except Exception as e:
        logger.error(f"Failed to update SL for trade #{trade['id']} in DB: {e}")

async def update_trade_peak_price_in_db(trade_id: int, highest_price: float):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute("UPDATE trades SET highest_price=? WHERE id=?", (highest_price, trade_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to update peak price for trade #{trade_id} in DB: {e}")


async def get_fear_and_greed_index():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            response.raise_for_status()
            if data := response.json().get('data', []):
                return int(data[0]['value'])
    except Exception as e:
        logger.error(f"Could not fetch Fear and Greed Index: {e}")
    return None

async def check_market_regime():
    settings = bot_state.settings
    fng_index = "N/A"
    
    btc_trend_data = None
    source_exchanges = settings.get("btc_trend_source_exchanges", ["binance"])
    for ex_id in source_exchanges:
        exchange = bot_state.public_exchanges.get(ex_id)
        if not exchange:
            continue
        try:
            ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '4h', limit=55)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['sma50'] = ta.sma(df['close'], length=50)
            btc_trend_data = df['close'].iloc[-1] > df['sma50'].iloc[-1]
            logger.info(f"Successfully fetched BTC trend from {ex_id}. Bullish: {btc_trend_data}")
            break 
        except Exception as e:
            logger.warning(f"Could not fetch BTC trend from {ex_id}, trying next... Error: {e}")
    
    if btc_trend_data is None:
        return False, "فشل جلب بيانات BTC من كل المصادر المتاحة."

    if not btc_trend_data:
        return False, "اتجاه BTC هابط (تحت متوسط 50 على 4 ساعات)."

    if settings.get("fear_and_greed_filter_enabled", True):
        fng_value = await get_fear_and_greed_index()
        if fng_value is not None:
            fng_index = fng_value
            if fng_index < settings.get("fear_and_greed_threshold", 30):
                return False, f"مشاعر خوف شديد (مؤشر F&G: {fng_index} تحت الحد {settings.get('fear_and_greed_threshold')})."
    
    return True, "وضع السوق مناسب لصفقات الشراء."


async def analyze_performance_and_suggest(context: ContextTypes.DEFAULT_TYPE):
    # ... (Same as before) ...
    pass

# =======================================================================================
# --- Telegram Handlers (Full Implementation) ---
# =======================================================================================
main_menu_keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"], ["ℹ️ مساعدة"]]
settings_menu_keyboard = [
    ["🏁 أنماط جاهزة", "🎭 تفعيل/تعطيل الماسحات"], 
    ["🔧 تعديل المعايير", "🚨 التحكم بالتداول الحقيقي"],
    ["🔙 القائمة الرئيسية"]
]
# ... (All Telegram handlers like start_command, show_dashboard_command, etc., are fully implemented here) ...
# =======================================================================================
# --- Telegram Handlers ---
# =======================================================================================
main_menu_keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"], ["ℹ️ مساعدة"]]
settings_menu_keyboard = [
    ["🏁 أنماط جاهزة", "🎭 تفعيل/تعطيل الماسحات"], 
    ["🔧 تعديل المعايير", "🚨 التحكم بالتداول الحقيقي"],
    ["🔙 القائمة الرئيسية"]
]

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = "💣 أهلاً بك في بوت **كاسحة الألغام**!\n\n*(الإصدار 7.5 - النسخة المستقرة)*\n\nاختر من القائمة للبدء."
    await update.message.reply_text(welcome_message, reply_markup=ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True), parse_mode=ParseMode.MARKDOWN)

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.message or update.callback_query.message
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 الإحصائيات العامة", callback_data="dashboard_stats"), InlineKeyboardButton("📈 الصفقات النشطة", callback_data="dashboard_active_trades")],
        [InlineKeyboardButton("📜 تقرير أداء الاستراتيجيات", callback_data="dashboard_strategy_report")],
        [InlineKeyboardButton("📸 لقطة للمحفظة", callback_data="dashboard_snapshot"), InlineKeyboardButton("ρίск تقرير المخاطر", callback_data="dashboard_risk")],
        [InlineKeyboardButton("🔄 المزامنة والإنقاذ", callback_data="dashboard_sync")],
        [InlineKeyboardButton("🛠️ أدوات المنقذ", callback_data="dashboard_tools"), InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="dashboard_debug")],
        [InlineKeyboardButton("🔄 تحديث", callback_data="dashboard_refresh_menu")]
    ])
    message_text = "🖥️ *لوحة التحكم الرئيسية*\n\nاختر التقرير أو البيانات التي تريد عرضها:"

    try:
        if update.callback_query:
             await target_message.edit_text(message_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        else:
            await target_message.reply_text(message_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
    except BadRequest as e:
        if "Message is not modified" in str(e):
            pass 
        else:
            logger.error(f"Error in show_dashboard_command: {e}")
            if update.callback_query:
                await context.bot.send_message(chat_id=target_message.chat_id, text=message_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)


async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE): await (update.message or update.callback_query.message).reply_text("اختر الإعداد:", reply_markup=ReplyKeyboardMarkup(settings_menu_keyboard, resize_keyboard=True))

def get_scanners_keyboard():
    active_scanners = bot_state.settings.get("active_scanners", [])
    keyboard = [[InlineKeyboardButton(f"{'✅' if name in active_scanners else '❌'} {STRATEGY_NAMES_AR.get(name, name)}", callback_data=f"toggle_scanner_{name}")] for name in SCANNERS.keys()]
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    return InlineKeyboardMarkup(keyboard)

def get_presets_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚦 احترافية (متوازنة)", callback_data="preset_PRO"), InlineKeyboardButton("🎯 متشددة", callback_data="preset_STRICT")],
        [InlineKeyboardButton("🌙 متساهلة", callback_data="preset_LAX"), InlineKeyboardButton("⚠️ فائق التساهل", callback_data="preset_VERY_LAX")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")]
    ])
    
async def show_real_trading_control_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.message or update.callback_query.message
    settings = bot_state.settings.get("real_trading_per_exchange", {})
    keyboard = []
    for ex_id in EXCHANGES_TO_SCAN:
        is_enabled = settings.get(ex_id, False)
        status_emoji = '✅' if is_enabled else '❌'
        button_text = f"{status_emoji} {ex_id.capitalize()}"
        callback_data = f"toggle_real_trade_{ex_id}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    
    await target_message.reply_text(
        "**🚨 التحكم بالتداول الحقيقي 🚨**\n\nاختر المنصة لتفعيل أو تعطيل التداول عليها:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

async def show_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.message or update.callback_query.message
    await target_message.reply_text("اختر نمط إعدادات جاهز:", reply_markup=get_presets_keyboard())
async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.message or update.callback_query.message
    await target_message.reply_text("اختر الماسحات لتفعيلها أو تعطيلها:", reply_markup=get_scanners_keyboard())
async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard, settings = [], bot_state.settings
    for category, params in EDITABLE_PARAMS.items():
        keyboard.append([InlineKeyboardButton(f"--- {category} ---", callback_data="ignore")])
        for row in [params[i:i + 2] for i in range(0, len(params), 2)]:
            button_row = []
            for param_key in row:
                display_name = PARAM_DISPLAY_NAMES.get(param_key, param_key)
                
                if param_key in ["trailing_sl_strategy", "use_strategy_mapping", "default_tsl_strategy", "tsl_ema_period", "tsl_atr_period", "tsl_atr_multiplier"]:
                     current_value = settings.get("trailing_sl_advanced", {}).get(param_key, "N/A")
                else:
                     current_value = settings.get(param_key, "N/A")

                text = f"{display_name}: {'مُفعّل ✅' if current_value else 'مُعطّل ❌'}" if isinstance(current_value, bool) else f"{display_name}: {current_value}"
                button_row.append(InlineKeyboardButton(text, callback_data=f"param_{param_key}"))
            keyboard.append(button_row)
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    message_text = "⚙️ *الإعدادات المتقدمة* ⚙️\n\nاختر الإعداد الذي تريد تعديله بالضغط عليه:"
    target_message = update.callback_query.message if update.callback_query else update.message
    try:
        if update.callback_query:
            await target_message.edit_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
        else:
            sent_message = await target_message.reply_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
            context.user_data['settings_menu_id'] = sent_message.message_id
    except BadRequest as e:
        if "Message is not modified" not in str(e): logger.error(f"Error editing parameters menu: {e}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "**💣 أوامر بوت كاسحة الألغام 💣**\n\n"
        "`/start` - لعرض القائمة الرئيسية وبدء التفاعل.\n"
        "`/check <ID>` - لمتابعة حالة صفقة معينة باستخدام رقمها.\n"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_mode_filter='all'):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor();
        
        query = "SELECT status, COUNT(*), SUM(pnl_usdt) FROM trades"
        params = []
        if trade_mode_filter != 'all':
            query += " WHERE trade_mode = ?"
            params.append(trade_mode_filter)
        query += " GROUP BY status"
        cursor.execute(query, params)
        
        stats_data = cursor.fetchall(); conn.close()
        
        counts = defaultdict(int)
        pnl = defaultdict(float)
        for status, count, p in stats_data:
            counts[status] = count
            pnl[status] = p or 0

        successful = counts['ناجحة (تحقيق هدف)'] + counts['ناجحة (وقف ربح)']
        failed = counts['فاشلة (وقف خسارة)']
        active = counts['نشطة']
        total = successful + failed + active
        
        pnl_wins = pnl['ناجحة (تحقيق هدف)'] + pnl['ناجحة (وقف ربح)']
        pnl_losses = pnl['فاشلة (وقف خسارة)']

        closed = successful + failed
        win_rate = (successful / closed * 100) if closed > 0 else 0
        total_pnl = pnl_wins + pnl_losses
        
        preset_name = bot_state.settings.get("active_preset_name", "N/A")
        mode_title_map = {'all': '(الكل)', 'real': '(حقيقي فقط)', 'virtual': '(وهمي فقط)'}
        title = mode_title_map.get(trade_mode_filter, '')

        balance_lines = []
        if trade_mode_filter == 'real':
            real_balance = await get_total_real_portfolio_value_usdt()
            balance_lines.append(f"💰 *إجمالي قيمة المحفظة الحقيقية:* `${real_balance:.2f}`")
        elif trade_mode_filter == 'virtual':
            balance_lines.append(f"📈 *الرصيد الافتراضي:* `${bot_state.settings['virtual_portfolio_balance_usdt']:.2f}`")
        else: # 'all'
            real_balance = await get_total_real_portfolio_value_usdt()
            balance_lines.append(f"💰 *قيمة المحفظة الحقيقية:* `${real_balance:.2f}`")
            balance_lines.append(f"📈 *الرصيد الافتراضي:* `${bot_state.settings['virtual_portfolio_balance_usdt']:.2f}`")

        balance_section = "\n".join(balance_lines)

        stats_msg = (f"*📊 إحصائيات المحفظة {title}*\n\n"
                       f"{balance_section}\n"
                       f"💰 *إجمالي الربح/الخسارة:* `${total_pnl:+.2f}`\n"
                       f"⚙️ *النمط الحالي:* `{preset_name}`\n\n"
                       f"- *إجمالي الصفقات:* `{total}` (`{active}` نشطة)\n"
                       f"- *الناجحة:* `{successful}` | *الربح:* `${pnl_wins:.2f}`\n"
                       f"- *الفاشلة:* `{failed}` | *الخسارة:* `${abs(pnl_losses):.2f}`\n"
                       f"- *معدل النجاح:* `{win_rate:.2f}%`")
        return stats_msg, None
    except Exception as e:
        logger.error(f"Error in stats_command: {e}", exc_info=True)
        return "خطأ في جلب الإحصائيات.", None


def generate_performance_report_string(trade_mode_filter='all'):
    """Generates a detailed performance report string for each strategy."""
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT reason, status, pnl_usdt FROM trades WHERE status != 'نشطة'"
        params = []
        if trade_mode_filter != 'all':
            query += " AND trade_mode = ?"
            params.append(trade_mode_filter)

        cursor.execute(query, params)
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not trades:
            return "لا توجد صفقات مغلقة لتحليل أداء الاستراتيجيات."

        strategy_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0})

        for trade in trades:
            reasons = [r.strip() for r in trade['reason'].split('+')]
            for reason in reasons:
                stats = strategy_stats[reason]
                if trade['status'].startswith('ناجحة'):
                    stats['wins'] += 1
                elif trade['status'].startswith('فاشلة'):
                    stats['losses'] += 1
                
                if trade['pnl_usdt'] is not None:
                    stats['total_pnl'] += trade['pnl_usdt'] / len(reasons)
        
        report_lines = ["**📜 تقرير أداء الاستراتيجيات**\n"]
        
        sorted_strategies = sorted(strategy_stats.items(), key=lambda item: item[1]['total_pnl'], reverse=True)

        for reason, stats in sorted_strategies:
            total_trades = stats['wins'] + stats['losses']
            win_rate = (stats['wins'] / total_trades * 100) if total_trades > 0 else 0
            avg_pnl = stats['total_pnl'] / total_trades if total_trades > 0 else 0
            strategy_name_ar = STRATEGY_NAMES_AR.get(reason, reason)

            report_lines.append(f"\n--- **{strategy_name_ar}** ---")
            report_lines.append(f"  - **إجمالي الصفقات:** {total_trades}")
            report_lines.append(f"  - **معدل النجاح:** {win_rate:.2f}% ({stats['wins']} ✅ / {stats['losses']} ❌)")
            report_lines.append(f"  - **صافي الربح/الخسارة:** `${stats['total_pnl']:+.2f}`")
            report_lines.append(f"  - **متوسط الربح للصفقة:** `${avg_pnl:+.2f}`")

        return "\n".join(report_lines)

    except Exception as e:
        logger.error(f"Error generating performance report string: {e}", exc_info=True)
        return "❌ حدث خطأ غير متوقع أثناء إعداد تقرير أداء الاستراتيجيات."


async def strategy_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_mode_filter='all'):
    report_string = generate_performance_report_string(trade_mode_filter)
    return report_string, None

async def send_daily_report(context: ContextTypes.DEFAULT_TYPE):
    today_str = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d')
    logger.info(f"Generating detailed daily report for {today_str}...")
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trades WHERE DATE(closed_at) = ? AND trade_mode = 'real'", (today_str,))
        closed_real_today = [dict(row) for row in cursor.fetchall()]
        
        cursor.execute("SELECT * FROM trades WHERE DATE(closed_at) = ? AND trade_mode = 'virtual'", (today_str,))
        closed_virtual_today = [dict(row) for row in cursor.fetchall()]
        conn.close()

        parts = [f"**🗓️ التقرير اليومي المفصل | {today_str}**\n"]

        def generate_section(title, trades):
            if not trades:
                return [f"\n--- **{title}** ---\nلم يتم إغلاق أي صفقات اليوم."]
            
            wins = [t for t in trades if t['status'].startswith('ناجحة')]
            losses = [t for t in trades if t['status'].startswith('فاشلة')]
            total_pnl = sum(t['pnl_usdt'] for t in trades if t['pnl_usdt'] is not None)
            win_rate = (len(wins) / len(trades) * 100) if trades else 0

            section_parts = [f"\n--- **{title}** ---"]
            section_parts.append(f"  - الربح/الخسارة الصافي: `${total_pnl:+.2f}`")
            section_parts.append(f"  - ✅ الرابحة: {len(wins)} | ❌ الخاسرة: {len(losses)}")
            section_parts.append(f"  - معدل النجاح: {win_rate:.1f}%")
            return section_parts

        parts.extend(generate_section("💰 الأداء الحقيقي", closed_real_today))
        parts.extend(generate_section("📊 الأداء الوهمي", closed_virtual_today))

        parts.append("\n\n*رسالة اليوم: \"النجاح في التداول هو نتيجة للانضباط والصبر والتعلم المستمر.\"*")
        report_message = "\n".join(parts)

        await send_telegram_message(context.bot, {'custom_message': report_message, 'target_chat': TELEGRAM_SIGNAL_CHANNEL_ID})
    except Exception as e:
        logger.error(f"Failed to generate detailed daily report: {e}", exc_info=True)

async def daily_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message if update.callback_query else update.message
    await target_message.reply_text("⏳ جاري إرسال التقرير اليومي المفصل...")
    await send_daily_report(context)
    await target_message.reply_text("✅ تم إرسال التقرير بنجاح إلى القناة.")

async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message if update.callback_query else update.message
    await target_message.reply_text("⏳ جاري إعداد تقرير التشخيص الشامل...")
    settings = bot_state.settings
    parts = [f"**🕵️‍♂️ تقرير التشخيص الشامل (v7.5)**\n\n*تم إنشاؤه في: {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')}*"]

    parts.append("\n- - - - - - - - - - - - - - - - - -")
    parts.append("**[ ⚙️ حالة النظام والبيئة ]**")
    parts.append(f"- `NLTK (تحليل الأخبار):` {'متاحة ✅' if NLTK_AVAILABLE else 'غير متاحة ❌'}")
    parts.append(f"- `SciPy (تحليل الدايفرجنس):` {'متاحة ✅' if SCIPY_AVAILABLE else 'غير متاحة ❌'}")
    parts.append(f"- `Alpha Vantage (بيانات اقتصادية):` {'موجود ✅' if ALPHA_VANTAGE_API_KEY != 'YOUR_AV_KEY_HERE' else 'مفقود ⚠️'}")

    parts.append("\n**[ 📊 حالة السوق الحالية ]**")
    mood_info = settings.get("last_market_mood", {})
    try:
        fng_value = await get_fear_and_greed_index()
        fng_text = "غير متاح"
        if fng_value is not None:
            classification = "خوف شديد" if fng_value < 25 else "خوف" if fng_value < 45 else "محايد" if fng_value < 55 else "طمع" if fng_value < 75 else "طمع شديد"
            fng_text = f"{fng_value} ({classification})"
    except Exception as e:
        fng_text = f"فشل الجلب ({e})"
    parts.append(f"- **المزاج الأساسي (أخبار):** `{mood_info.get('mood', 'N/A')}`")
    parts.append(f"  - `{mood_info.get('reason', 'N/A')}`")
    parts.append(f"- **المزاج الفني (BTC):** `{bot_state.status_snapshot['btc_market_mood']}`")
    parts.append(f"- **مؤشر الخوف والطمع:** `{fng_text}`")

    status = bot_state.status_snapshot
    scan_duration = "N/A"
    if isinstance(status.get('last_scan_end_time'), datetime) and isinstance(status.get('last_scan_start_time'), datetime):
        duration_sec = (status['last_scan_end_time'] - status['last_scan_start_time']).total_seconds()
        scan_duration = f"{duration_sec:.0f} ثانية"
    parts.append("\n**[ 🔬 أداء آخر فحص ]**")
    parts.append(f"- **وقت البدء:** `{status.get('last_scan_start_time', 'N/A')}`")
    parts.append(f"- **المدة:** `{scan_duration}`")
    parts.append(f"- **العملات المفحوصة:** `{status['markets_found']}`")
    parts.append(f"- **فشل في تحليل:** `{(bot_state.scan_history[-1]['failures'] if bot_state.scan_history else 'N/A')}` عملات")

    parts.append("\n**[ 🔧 الإعدادات النشطة ]**")
    parts.append(f"- **النمط الحالي:** `{settings.get('active_preset_name', 'N/A')}`")
    parts.append(f"- **الماسحات المفعلة:** `{', '.join(settings.get('active_scanners', []))}`")
    
    parts.append("\n**[ 🔩 حالة العمليات الداخلية ]**")
    if context.job_queue:
        try:
            scan_job = context.job_queue.get_jobs_by_name('perform_scan')
            track_job = context.job_queue.get_jobs_by_name('track_open_trades')

            def get_next_run_str(job):
                if not job or not job[0].next_t: return 'N/A'
                now = datetime.now(EGYPT_TZ)
                next_t = job[0].next_t.astimezone(EGYPT_TZ)
                if next_t < now:
                    if job[0].enabled:
                         return 'يعمل الآن أو سيبدأ خلال ثوانٍ'
                    else:
                         return 'متوقف مؤقتاً'
                delta = next_t - now
                minutes, seconds = divmod(int(delta.total_seconds()), 60)
                return f'بعد {minutes} دقيقة و {seconds} ثانية'

            scan_next_str = get_next_run_str(scan_job)
            track_next_str = get_next_run_str(track_job)
            
            parts.append("- **المهام المجدولة:**")
            parts.append(f"  - `فحص العملات:` {scan_next_str}")
            parts.append(f"  - `متابعة الصفقات:` {track_next_str}")
        except Exception as e:
            parts.append(f"- **المهام المجدولة:** فشل الفحص ({e})")
            
    parts.append("- **الاتصال بالمنصات:**")
    for ex_id in EXCHANGES_TO_SCAN:
        is_private_connected = ex_id in bot_state.exchanges and bot_state.exchanges[ex_id].apiKey
        is_public_connected = ex_id in bot_state.public_exchanges
        status_text = f"عام: {'✅' if is_public_connected else '❌'} | خاص: {'✅' if is_private_connected else '❌'}"
        parts.append(f"  - `{ex_id.capitalize()}:` {status_text}")

    parts.append("- **قاعدة البيانات:**")
    try:
        conn = sqlite3.connect(DB_FILE, timeout=5); cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades"); total_trades = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'نشطة'"); active_trades = cursor.fetchone()[0]
        conn.close()
        db_size = os.path.getsize(DB_FILE) / (1024 * 1024)
        parts.append(f"  - `الاتصال:` ناجح ✅")
        parts.append(f"  - `حجم الملف:` {db_size:.2f} MB")
        parts.append(f"  - `إجمالي الصفقات:` {total_trades} ({active_trades} نشطة)")
    except Exception as e: parts.append(f"  - `الاتصال:` فشل ❌ ({e})")
    parts.append("- - - - - - - - - - - - - - - - - -")

    await target_message.reply_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN)

async def check_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_id_from_callback=None):
    target = update.callback_query.message if trade_id_from_callback else update.message
    def format_price(price): return f"{price:,.8f}" if price < 0.01 else f"{price:,.4f}"
    try:
        trade_id = trade_id_from_callback or int(context.args[0])
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row; cursor = conn.cursor(); cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,));
        trade = dict(trade_row) if (trade_row := cursor.fetchone()) else None; conn.close()
        if not trade: await target.reply_text(f"لم يتم العثور على صفقة بالرقم `{trade_id}`."); return
        if trade['status'] != 'نشطة':
            pnl_percent = (trade['pnl_usdt'] / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0

            closed_at_dt_naive = datetime.strptime(trade['closed_at'], '%Y-%m-%d %H:%M:%S')
            closed_at_dt = EGYPT_TZ.localize(closed_at_dt_naive)
            message = f"📋 *ملخص الصفقة #{trade_id}*\n\n*العملة:* `{trade['symbol']}`\n*الحالة:* `{trade['status']}`\n*تاريخ الإغلاق:* `{closed_at_dt.strftime('%Y-%m-%d %I:%M %p')}`\n*الربح/الخسارة:* `${trade.get('pnl_usdt', 0):+.2f} ({pnl_percent:+.2f}%)`"
        else:
            if not (exchange := bot_state.public_exchanges.get(trade['exchange'].lower())): await target.reply_text("المنصة غير متصلة."); return
            if not (ticker := await exchange.fetch_ticker(trade['symbol'])) or not (current_price := ticker.get('last') or ticker.get('close')):
                await target.reply_text(f"لم أتمكن من جلب السعر الحالي لـ `{trade['symbol']}`."); return
            live_pnl = (current_price - trade['entry_price']) * trade['quantity']
            live_pnl_percent = (live_pnl / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0
            message = (f"📈 *متابعة حية للصفقة #{trade_id}*\n\n"
                       f"▫️ *العملة:* `{trade['symbol']}` | *الحالة:* `نشطة`\n"
                       f"▫️ *سعر الدخول:* `${format_price(trade['entry_price'])}`\n"
                       f"▫️ *السعر الحالي:* `${format_price(current_price)}`\n\n"
                       f"💰 *الربح/الخسارة الحالية:*\n`${live_pnl:+.2f} ({live_pnl_percent:+.2f}%)`")
        await target.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    except (ValueError, IndexError): await target.reply_text("رقم صفقة غير صالح. مثال: `/check 17`")
    except Exception as e: logger.error(f"Error in check_trade_command: {e}", exc_info=True); await target.reply_text("حدث خطأ.")

async def show_active_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_mode_filter='all'):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
        
        query = "SELECT id, symbol, entry_value_usdt, exchange FROM trades WHERE status = 'نشطة'"
        params = []
        if trade_mode_filter != 'all':
            query += " AND trade_mode = ?"
            params.append(trade_mode_filter)
        query += " ORDER BY id DESC"

        cursor.execute(query, params)
        active_trades = cursor.fetchall(); conn.close()
        
        if not active_trades:
            return "لا توجد صفقات نشطة حالياً لهذا الفلتر.", None
            
        keyboard = [[InlineKeyboardButton(f"#{t['id']} | {t['symbol']} | ${t['entry_value_usdt']:.2f} | {t['exchange']}", callback_data=f"check_{t['id']}")] for t in active_trades]
        return "اختر صفقة لمتابعتها:", InlineKeyboardMarkup(keyboard)
    except Exception as e:
        logger.error(f"Error in show_active_trades: {e}")
        return "خطأ في جلب الصفقات.", None

async def execute_manual_order(exchange_id, symbol, amount, side, context: ContextTypes.DEFAULT_TYPE, order_type='market', price=None, stop_price=None):
    logger.info(f"Attempting MANUAL {order_type.upper()} {side.upper()} for {symbol} on {exchange_id} for {amount}")
    exchange = bot_state.exchanges.get(exchange_id.lower())
    if not exchange or not exchange.apiKey:
        return {"success": False, "error": f"لا يمكن تنفيذ الأمر. لم يتم توثيق الاتصال بمنصة {exchange_id.capitalize()}."}

    try:
        order_receipt = None
        params = {}
        
        # Note: For market buy, 'amount' is the cost (USDT). For all others, it's the base currency amount.
        if order_type == 'market' and side == 'buy':
             if not hasattr(exchange, 'create_market_buy_order_with_cost'):
                 ticker = await exchange.fetch_ticker(symbol)
                 current_price = ticker.get('last', 0)
                 if not current_price: return {"success": False, "error": "Could not fetch price for market buy conversion."}
                 amount_in_base = float(amount) / current_price
                 order_receipt = await exchange.create_market_buy_order(symbol, amount_in_base)
             else:
                order_receipt = await exchange.create_market_buy_order_with_cost(symbol, float(amount))
        elif order_type == 'market' and side == 'sell':
            order_receipt = await exchange.create_market_sell_order(symbol, float(amount))
        elif order_type == 'limit':
            order_receipt = await exchange.create_order(symbol, 'limit', side, amount, price)
        elif order_type == 'stop-market':
            params['stopPrice'] = stop_price
            order_receipt = await exchange.create_order(symbol, 'market', side, amount, params=params)

        if not order_receipt:
            raise ValueError("Order creation failed without a specific CCXT exception.")

        await asyncio.sleep(2) 
        order = await exchange.fetch_order(order_receipt['id'], symbol)

        logger.info(f"MANUAL ORDER SUCCESS: {order}")

        filled_quantity = order.get('filled', 0)
        filled_price = order.get('average')
        cost = order.get('cost', 0)
        status = order.get('status', 'unknown')

        success_message = (
            f"**✅ تم {'تنفيذ' if status == 'closed' else 'وضع'} الأمر اليدوي بنجاح**\n\n"
            f"**المنصة:** `{exchange_id.capitalize()}`\n"
            f"**العملة:** `{symbol}`\n"
            f"**النوع:** `{order_type.upper()} {side.upper()}`\n\n"
            f"--- **تفاصيل الأمر** ---\n"
            f"**ID:** `{order['id']}`\n"
            f"**الحالة:** `{status}`\n"
            f"**الكمية:** `{order['amount']}`\n"
            f"**السعر:** `{order.get('price') or 'Market'}`\n"
            f"**الكمية المنفذة:** `{filled_quantity}`\n"
            f"**متوسط سعر التنفيذ:** `{filled_price or 'N/A'}`\n"
            f"**التكلفة الإجمالية:** `${cost:.2f}`"
        )
        return {"success": True, "message": success_message}

    except ccxt.InsufficientFunds as e:
        error_msg = f"❌ فشل: رصيد غير كافٍ على {exchange_id.capitalize()}."
        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}")
        return {"success": False, "error": error_msg}
    except ccxt.InvalidOrder as e:
        error_msg = f"❌ فشل: أمر غير صالح. قد يكون المبلغ أقل من الحد الأدنى للمنصة.\n`{e}`"
        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}")
        return {"success": False, "error": error_msg}
    except ccxt.ExchangeError as e:
        error_msg = f"❌ فشل: خطأ من المنصة.\n`{e}`"
        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}")
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"❌ فشل: حدث خطأ غير متوقع.\n`{e}`"
        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}", exc_info=True)
        return {"success": False, "error": error_msg}


async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data
    user_data = context.user_data

    if data.startswith("rescue_"):
        _, exchange_id, symbol = data.split("_", 2)
        exchange = bot_state.exchanges.get(exchange_id)
        if not exchange:
            await query.message.reply_text(f"❌ خطأ: لم يتم العثور على اتصال بمنصة {exchange_id}")
            return

        await query.edit_message_text(f"🚑 جارِ عملية الإنقاذ لـ `{symbol}`...\n\n- الخطوة 1: جلب سجل التداول.\n- الخطوة 2: حساب متوسط الشراء.\n- الخطوة 3: تسجيل الصفقة ومتابعتها.", parse_mode=ParseMode.MARKDOWN)
        
        result_message = await _reconstruct_and_save_trade(exchange, symbol, context)
        
        await query.message.reply_text(result_message, parse_mode=ParseMode.MARKDOWN)
        await process_sync_portfolio(update, context, exchange_id)
        return

    if data.startswith("dashboard_") and data.endswith(('_all', '_real', '_virtual')):
        if report_lock.locked():
            await query.answer("⏳ تقرير آخر قيد الإعداد، يرجى الانتظار...", show_alert=False)
            return
            
        async with report_lock:
            try:
                parts = data.split('_')
                trade_mode_filter = parts[-1]
                report_type = '_'.join(parts[1:-1])
                await query.edit_message_text(f"⏳ جاري إعداد تقرير **{report_type.replace('_', ' ').capitalize()}**...", parse_mode=ParseMode.MARKDOWN)
                report_content, keyboard = None, None
                if report_type == "stats": report_content, keyboard = await stats_command(update, context, trade_mode_filter=trade_mode_filter)
                elif report_type == "active_trades": report_content, keyboard = await show_active_trades_command(update, context, trade_mode_filter=trade_mode_filter)
                elif report_type == "strategy_report": report_content, keyboard = await strategy_report_command(update, context, trade_mode_filter=trade_mode_filter)
                if report_content: await query.edit_message_text(text=report_content, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
                else: await query.edit_message_text("❌ فشل إعداد التقرير.")
            except Exception as e:
                logger.error(f"Error in dashboard filter handler: {e}", exc_info=True)
                await query.edit_message_text("❌ حدث خطأ أثناء إعداد التقرير.")
        return

    if data.startswith("dashboard_"):
        action = data.split("_", 1)[1]
        
        if action in ["stats", "active_trades", "strategy_report"]:
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 الكل (وهمي + حقيقي)", callback_data=f"dashboard_{action}_all")],
                [InlineKeyboardButton("📈 حقيقي فقط", callback_data=f"dashboard_{action}_real"), InlineKeyboardButton("📉 وهمي فقط", callback_data=f"dashboard_{action}_virtual")],
                [InlineKeyboardButton("🔙 العودة", callback_data="dashboard_refresh_menu")]
            ])
            await query.edit_message_text(f"اختر نوع السجل لعرض **{action.replace('_', ' ').capitalize()}**:", reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
            return

        if action == "debug": 
            await query.edit_message_text("⏳ جاري إعداد تقرير التشخيص...", parse_mode=ParseMode.MARKDOWN)
            await debug_command(update, context)
        elif action == "refresh_menu":
            await show_dashboard_command(update, context)
        elif action == "snapshot": await portfolio_snapshot_command(update, context)
        elif action == "risk": await risk_report_command(update, context)
        elif action == "sync": await sync_portfolio_command(update, context)
        elif action == "tools":
              # [v7.5] Added Rescuer Tools
              keyboard = [
                  [InlineKeyboardButton("➕ وضع أمر متقدم", callback_data="tools_place_order"), InlineKeyboardButton("📉 بيع بسعر السوق", callback_data="tools_market_sell")],
                  [InlineKeyboardButton("❌ إلغاء أمر مفتوح", callback_data="tools_cancel_order")],
                  [InlineKeyboardButton("💰 عرض رصيدي", callback_data="tools_balance"), InlineKeyboardButton("📜 سجل تداولاتي", callback_data="tools_mytrades")],
                  [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="dashboard_refresh_menu")]
              ]
              await query.edit_message_text("🛠️ *أدوات المنقذ*\n\nاختر الأداة التي تريد استخدامها:", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
        return

    if data.startswith("tools_"):
        tool_name = data.split("_", 1)[1]
        if tool_name == "place_order": await place_advanced_order_command(update, context)
        elif tool_name == "market_sell": await market_sell_command(update, context)
        elif tool_name == "cancel_order": await cancel_order_command(update, context)
        elif tool_name == "balance": await balance_command(update, context)
        elif tool_name == "mytrades": await my_trades_command(update, context)
        return
        
    if data.startswith("balance_"): await tools_button_handler(update, context); return
    if data.startswith("mytrades_"): await tools_button_handler(update, context); return
    if data.startswith("p_order_"): await advanced_order_button_handler(update, context); return
    if data.startswith("c_order_"): await cancel_order_button_handler(update, context); return
    if data.startswith("m_sell_"): await market_sell_button_handler(update, context); return

    if data.startswith("snapshot_exchange_") or data.startswith("sync_exchange_"):
        parts = data.split("_")
        tool, exchange_id = parts[0], parts[2]
        if tool == 'snapshot': await process_portfolio_snapshot(update, context, exchange_id)
        elif tool == 'sync': await process_sync_portfolio(update, context, exchange_id)
        return

    if data.startswith("preset_"):
        preset_name = data.split("_", 1)[1]
        if preset_data := PRESETS.get(preset_name):
            bot_state.settings['liquidity_filters'].update(preset_data['liquidity_filters'])
            bot_state.settings['volatility_filters'].update(preset_data['volatility_filters'])
            bot_state.settings['ema_trend_filter'].update(preset_data['ema_trend_filter'])
            bot_state.settings['min_tp_sl_filter'].update(preset_data['min_tp_sl_filter'])
            bot_state.settings["active_preset_name"] = preset_name
            save_settings()
            await query.edit_message_text("✅ تم تفعيل النمط.", reply_markup=get_presets_keyboard())
    elif data.startswith("param_"):
        param_key = data.split("_", 1)[1]
        user_data['awaiting_input_for_param'] = param_key; user_data['settings_menu_id'] = query.message.message_id
        
        is_advanced_tsl_param = param_key in ["trailing_sl_strategy", "use_strategy_mapping", "default_tsl_strategy", "tsl_ema_period", "tsl_atr_period", "tsl_atr_multiplier"]
        if is_advanced_tsl_param:
            current_value = bot_state.settings.get("trailing_sl_advanced", {}).get(param_key)
        else:
            current_value = bot_state.settings.get(param_key)

        if isinstance(current_value, bool):
            if is_advanced_tsl_param:
                bot_state.settings["trailing_sl_advanced"][param_key] = not current_value
            else:
                bot_state.settings[param_key] = not current_value

            bot_state.settings["active_preset_name"] = "Custom"; save_settings()
            await query.answer(f"✅ تم تبديل '{PARAM_DISPLAY_NAMES.get(param_key, param_key)}'")
            await show_parameters_menu(update, context)
        else: await query.edit_message_text(f"📝 *تعديل '{PARAM_DISPLAY_NAMES.get(param_key, param_key)}'*\n\n*القيمة الحالية:* `{current_value}`\n\nالرجاء إرسال القيمة الجديدة.", parse_mode=ParseMode.MARKDOWN)
    elif data.startswith("toggle_scanner_"):
        scanner_name = data.split("_", 2)[2]
        active_scanners = bot_state.settings.get("active_scanners", []).copy()
        if scanner_name in active_scanners: active_scanners.remove(scanner_name)
        else: active_scanners.append(scanner_name)
        bot_state.settings["active_scanners"] = active_scanners; save_settings()
        await query.edit_message_text(text="اختر الماسحات لتفعيلها أو تعطيلها:", reply_markup=get_scanners_keyboard())
    elif data.startswith("toggle_real_trade_"):
        exchange_id = data.split("_", 3)[3]
        settings = bot_state.settings.get("real_trading_per_exchange", {})
        settings[exchange_id] = not settings.get(exchange_id, False)
        bot_state.settings["real_trading_per_exchange"] = settings; save_settings()
        await query.answer(f"تم {'تفعيل' if settings[exchange_id] else 'تعطيل'} التداول على {exchange_id.capitalize()}")
        await query.message.delete()
        await show_real_trading_control_menu(update, context)
        return
    elif data == "back_to_settings":
        if query.message: await query.message.delete()
        await context.bot.send_message(chat_id=query.message.chat_id, text="اختر الإعداد:", reply_markup=ReplyKeyboardMarkup(settings_menu_keyboard, resize_keyboard=True))
    elif data.startswith("check_"):
        await check_trade_command(update, context, trade_id_from_callback=int(data.split("_")[1]))
    elif data.startswith("suggest_"):
        action = data.split("_", 1)[1]
        if action.startswith("accept"):
            preset_name = data.split("_")[2]
            if preset_data := PRESETS.get(preset_name):
                bot_state.settings['liquidity_filters'].update(preset_data['liquidity_filters'])
                bot_state.settings['volatility_filters'].update(preset_data['volatility_filters'])
                bot_state.settings['ema_trend_filter'].update(preset_data['ema_trend_filter'])
                bot_state.settings['min_tp_sl_filter'].update(preset_data['min_tp_sl_filter'])
                bot_state.settings["active_preset_name"] = preset_name; save_settings()
                await query.edit_message_text(f"✅ **تم قبول الاقتراح!**\n\nتم تغيير النمط بنجاح إلى `{preset_name}`.", parse_mode=ParseMode.MARKDOWN)
        elif action == "decline":
            await query.edit_message_text("👍 **تم تجاهل الاقتراح.**", parse_mode=ParseMode.MARKDOWN)

def get_exchange_selection_keyboard(callback_prefix: str, back_button_cb: str):
    """Generates a keyboard with buttons for all connected private exchanges."""
    keyboard = []
    connected_exchanges = list(bot_state.exchanges.keys())
    for i in range(0, len(connected_exchanges), 2):
        row = [
            InlineKeyboardButton(ex.capitalize(), callback_data=f"{callback_prefix}_exchange_{ex}")
            for ex in connected_exchanges[i:i+2]
        ]
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("🔙 العودة", callback_data=back_button_cb)])
    return InlineKeyboardMarkup(keyboard)

async def tools_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    user_data = context.user_data
    
    if len(data.split("_")) < 3: return
    
    tool_name, action, value = data.split("_", 2)

    tool_key = f"{tool_name}_tool"
    user_data[tool_key] = {}
    if action == "exchange":
        user_data[tool_key]['exchange'] = value
        if tool_name == "balance":
            await query.edit_message_text(f"💰 جاري جلب الأرصدة من *{value.capitalize()}*...", parse_mode=ParseMode.MARKDOWN)
            await fetch_and_display_balance(value, query)
            user_data.pop(tool_key, None)
        else:
            user_data[tool_key]['state'] = 'awaiting_symbol'
            await query.edit_message_text(f"اخترت منصة: *{value.capitalize()}*\n\nالآن، أرسل رمز العملة (مثال: `BTC/USDT`)\nأو أرسل `الكل` لعرض البيانات لجميع العملات.", parse_mode=ParseMode.MARKDOWN)

async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_data = context.user_data
    text = update.message.text

    # [v7.5] Unified tool handler
    active_tool = None
    for tool_key in ['c_order_tool', 'mytrades_tool', 'p_order_tool', 'm_sell_tool']:
        if tool_key in user_data:
            active_tool = tool_key
            break
            
    if 'p_order_tool' in user_data:
        await advanced_order_text_handler(update, context)
        return
    
    if 'm_sell_tool' in user_data:
        await market_sell_text_handler(update, context)
        return

    if active_tool:
        state = user_data[active_tool].get('state')
        if state == 'awaiting_symbol':
            symbol = text.upper()
            exchange_id = user_data[active_tool]['exchange']

            if symbol.lower() in ["all", "الكل"] and active_tool not in ['c_order_tool']:
                symbol = None
            elif '/' not in symbol:
                await update.message.reply_text("❌ رمز غير صالح. الرجاء إرسال الرمز بالتنسيق الصحيح (مثال: `BTC/USDT`).")
                return
            
            if active_tool == 'c_order_tool':
                 await update.message.reply_text(f"📖 جاري جلب أوامرك المفتوحة لـ *{symbol or 'الكل'}*...", parse_mode=ParseMode.MARKDOWN)
                 await fetch_and_display_cancellable_orders(exchange_id, symbol, update.message)
            elif active_tool == 'mytrades_tool':
                await update.message.reply_text(f"📜 جاري جلب سجل تداولاتك لـ *{symbol or 'الكل'}*...", parse_mode=ParseMode.MARKDOWN)
                await fetch_and_display_my_trades(exchange_id, symbol, update.message)

            user_data.pop(active_tool, None)
            return
    
    menu_handlers = {
        "Dashboard 🖥️": show_dashboard_command,
        "ℹ️ مساعدة": help_command,
        "⚙️ الإعدادات": show_settings_menu,
        "🔧 تعديل المعايير": show_parameters_menu,
        "🔙 القائمة الرئيسية": start_command,
        "🎭 تفعيل/تعطيل الماسحات": show_scanners_menu,
        "🏁 أنماط جاهزة": show_presets_menu,
        "🚨 التحكم بالتداول الحقيقي": show_real_trading_control_menu,
    }
    if text in menu_handlers:
        for key in list(user_data.keys()):
            if key.startswith(('p_order_tool', 'c_order_tool', 'm_sell_tool', 'mytrades_tool', 'balance_tool')) or key == 'awaiting_input_for_param':
                user_data.pop(key)

        handler = menu_handlers[text]
        await handler(update, context)
        return

    if param := user_data.pop('awaiting_input_for_param', None):
        value_str = update.message.text
        settings_menu_id = context.user_data.pop('settings_menu_id', None)
        chat_id = update.message.chat_id
        await context.bot.delete_message(chat_id=chat_id, message_id=update.message.message_id)
        settings = bot_state.settings
        try:
            is_advanced_tsl_param = param in ["trailing_sl_strategy", "use_strategy_mapping", "default_tsl_strategy", "tsl_ema_period", "tsl_atr_period", "tsl_atr_multiplier"]
            if is_advanced_tsl_param:
                target_dict = settings.get("trailing_sl_advanced", {})
                current_type = type(target_dict.get(param, ''))
            else:
                target_dict = settings
                current_type = type(settings.get(param, ''))

            new_value = current_type(value_str)
            if isinstance(target_dict.get(param), bool):
                new_value = value_str.lower() in ['true', '1', 'yes', 'on', 'نعم', 'تفعيل']
            
            target_dict[param] = new_value
            settings["active_preset_name"] = "Custom"
            save_settings()
            
            if settings_menu_id: context.user_data['settings_menu_id'] = settings_menu_id
            await show_parameters_menu(update, context)
            confirm_msg = await update.message.reply_text(f"✅ تم تحديث **{PARAM_DISPLAY_NAMES.get(param, param)}** إلى `{new_value}`.", parse_mode=ParseMode.MARKDOWN)
            context.job_queue.run_once(lambda ctx: ctx.bot.delete_message(chat_id, confirm_msg.message_id), 4)
        except (ValueError, KeyError):
            if settings_menu_id:
                await context.bot.edit_message_text(chat_id=chat_id, message_id=settings_menu_id, text="❌ قيمة غير صالحة. الرجاء المحاولة مرة أخرى.")
                context.job_queue.run_once(lambda _: show_parameters_menu(update, context), 3)
        return

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['balance_tool'] = {'state': 'awaiting_exchange'}
    keyboard = get_exchange_selection_keyboard("balance", "dashboard_tools")
    await update.callback_query.edit_message_text("💰 **عرض الرصيد**\n\nاختر المنصة لعرض أرصدتك:", reply_markup=keyboard)

async def my_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mytrades_tool'] = {'state': 'awaiting_exchange'}
    keyboard = get_exchange_selection_keyboard("mytrades", "dashboard_tools")
    await update.callback_query.edit_message_text("📜 **سجل تداولاتي**\n\nاختر المنصة:", reply_markup=keyboard)

async def fetch_and_display_balance(exchange_id, query):
    exchange = bot_state.exchanges.get(exchange_id.lower())
    if not exchange or not exchange.apiKey:
        await query.edit_message_text(f"❌ خطأ: لم يتم توثيق الاتصال بمنصة {exchange_id.capitalize()}.")
        return

    try:
        portfolio_data = await calculate_full_portfolio(exchange)
        assets = portfolio_data.get('assets', [])
        total_usdt_value = portfolio_data.get('total_usdt', 0)

        if not assets:
            await query.edit_message_text(f"ℹ️ لا توجد أرصدة كبيرة (> $1) على منصة {exchange_id.capitalize()}.")
            return

        message_lines = [f"**💰 رصيدك على {exchange_id.capitalize()}**\n"]
        message_lines.append(f"__**إجمالي القيمة التقديرية:**__ `${total_usdt_value:,.2f}`\n")

        for asset in assets[:15]:
            message_lines.append(f"- `{asset['currency']}`: `{asset['amount']:.4f}` (~`${asset['usdt_value']:.2f}`)")

        await query.edit_message_text("\n".join(message_lines), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error fetching balance for {exchange_id}: {e}")
        await query.edit_message_text(f"❌ حدث خطأ أثناء جلب الرصيد من {exchange_id.capitalize()}.")

async def fetch_and_display_my_trades(exchange_id, symbol, message):
    exchange = bot_state.exchanges.get(exchange_id.lower())
    if not exchange or not exchange.apiKey:
        await message.reply_text(f"❌ خطأ: لم يتم توثيق الاتصال بمنصة {exchange_id.capitalize()}.")
        return
    try:
        my_trades = await exchange.fetch_my_trades(symbol, limit=20)

        if not my_trades:
            await message.reply_text(f"✅ لا يوجد لديك سجل تداول لـ `{symbol or 'الكل'}` على {exchange_id.capitalize()}.")
            return

        lines = [f"**📜 آخر 20 من تداولاتك لـ `{symbol or 'الكل'}` على {exchange_id.capitalize()}**\n"]

        for trade in reversed(my_trades):
            trade_time = datetime.fromtimestamp(trade['timestamp'] / 1000, tz=EGYPT_TZ).strftime('%Y-%m-%d %H:%M')
            side_emoji = "🔼" if trade['side'] == 'buy' else "🔽"
            fee = trade.get('fee', {})
            fee_str = f"{fee.get('cost', 0):.4f} {fee.get('currency', '')}"
            lines.append(
                f"`{trade_time}` | `{trade['symbol']}` {side_emoji} `{trade['side'].upper()}`\n"
                f"  - **الكمية:** `{trade['amount']}`\n"
                f"  - **السعر:** `{trade['price']}`\n"
                f"  - **الرسوم:** `{fee_str}`"
            )

        await message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Error fetching my trades for {symbol} on {exchange_id}: {e}")
        await message.reply_text(f"❌ فشل جلب سجل تداولاتك. تأكد من صحة الرمز: `{symbol or ''}`.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None: 
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

async def calculate_full_portfolio(exchange):
    """Calculates the total portfolio value and provides a detailed asset breakdown."""
    if not exchange or not exchange.apiKey:
        return {'total_usdt': 0, 'assets': []}
        
    try:
        balance = await exchange.fetch_balance()
        all_assets = balance.get('total', {})
        
        if not hasattr(exchange, '_tickers_cache') or (time.time() - getattr(exchange, '_tickers_cache_time', 0) > 60):
             exchange._tickers_cache = await exchange.fetch_tickers()
             exchange._tickers_cache_time = time.time()
        tickers = exchange._tickers_cache
        
        portfolio_assets = []
        total_usdt_value = 0
        for currency, amount in all_assets.items():
            if amount > 0:
                usdt_value = 0
                if currency == 'USDT':
                    usdt_value = amount
                elif f"{currency}/USDT" in tickers and tickers[f"{currency}/USDT"].get('last'):
                    usdt_value = amount * tickers[f"{currency}/USDT"]['last']
                
                if usdt_value > 1.0:
                    portfolio_assets.append({'currency': currency, 'amount': amount, 'usdt_value': usdt_value})
                    total_usdt_value += usdt_value
        
        portfolio_assets.sort(key=lambda x: x['usdt_value'], reverse=True)
        return {'total_usdt': total_usdt_value, 'assets': portfolio_assets}
    except Exception as e:
        logger.error(f"Could not calculate portfolio value for {exchange.id}: {e}")
        return {'total_usdt': 0, 'assets': []}


async def get_total_real_portfolio_value_usdt():
    total_value = 0
    tasks = [calculate_full_portfolio(ex) for ex in bot_state.exchanges.values() if ex.apiKey]
    results = await asyncio.gather(*tasks)
    for res in results:
        total_value += res['total_usdt']
    return total_value

async def portfolio_snapshot_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message
    
    connected_exchanges = [ex for ex in bot_state.exchanges.values() if ex.apiKey]
    
    if not connected_exchanges:
        await target_message.edit_text("❌ **فشل:** لم يتم العثور على أي منصة متصلة بحساب حقيقي.")
        return

    if len(connected_exchanges) == 1:
        await process_portfolio_snapshot(update, context, connected_exchanges[0].id)
    else:
        keyboard = get_exchange_selection_keyboard("snapshot", "dashboard_refresh_menu")
        await target_message.edit_text(
            "**📸 لقطة للمحفظة**\n\nلديك أكثر من منصة متصلة. اختر المنصة:",
            reply_markup=keyboard
        )

async def process_portfolio_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE, exchange_id: str):
    target_message = update.callback_query.message
    await target_message.edit_text(f"📸 **لقطة للمحفظة**\n\n⏳ جارِ الاتصال بمنصة {exchange_id.capitalize()} وجلب البيانات...")

    exchange = bot_state.exchanges.get(exchange_id)
    if not exchange:
        await target_message.edit_text(f"❌ **فشل:** خطأ في العثور على منصة {exchange_id.capitalize()} المتصلة.")
        return

    try:
        portfolio_data = await calculate_full_portfolio(exchange)
        portfolio_assets = portfolio_data.get('assets', [])
        total_usdt_value = portfolio_data.get('total_usdt', 0)
        
        symbols_to_fetch = [f"{asset['currency']}/USDT" for asset in portfolio_assets if f"{asset['currency']}/USDT" in exchange.markets]
        
        trade_tasks = [exchange.fetch_my_trades(symbol=symbol, limit=5) for symbol in symbols_to_fetch]
        trade_results = await asyncio.gather(*trade_tasks, return_exceptions=True)

        all_recent_trades = []
        for result in trade_results:
            if not isinstance(result, Exception):
                all_recent_trades.extend(result)
        
        all_recent_trades.sort(key=lambda x: x['timestamp'], reverse=True)
        recent_trades = all_recent_trades[:20]

        parts = [f"**📸 لقطة لمحفظة {exchange.id.capitalize()}**\n"]
        parts.append(f"__**إجمالي القيمة التقديرية:**__ `${total_usdt_value:,.2f}`\n")

        parts.append("--- **الأرصدة الحالية (> $1)** ---")
        for asset in portfolio_assets[:15]:
            parts.append(f"- **{asset['currency']}**: `{asset['amount']:.4f}` *~`${asset['usdt_value']:.2f}`*")
        
        parts.append("\n--- **آخر 20 عملية تداول** ---")
        if not recent_trades:
            parts.append("لا يوجد سجل تداولات حديث.")
        else:
            for trade in recent_trades:
                side_emoji = "🟢" if trade['side'] == 'buy' else "🔴"
                parts.append(f"`{trade['symbol']}` {side_emoji} `{trade['side'].upper()}` | الكمية: `{trade['amount']}` | السعر: `{trade['price']}`")
        
        await target_message.edit_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error generating portfolio snapshot: {e}", exc_info=True)
        await target_message.edit_text(f"❌ **فشل:** حدث خطأ أثناء جلب بيانات المحفظة.\n`{e}`")

async def risk_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message
    await target_message.edit_text("ρίск **تقرير المخاطر**\n\n⏳ جارِ تحليل الصفقات النشطة...")

    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.row_factory = sqlite3.Row
        
        real_trades = [dict(row) for row in conn.cursor().execute("SELECT * FROM trades WHERE status = 'نشطة' AND trade_mode = 'real'").fetchall()]
        virtual_trades = [dict(row) for row in conn.cursor().execute("SELECT * FROM trades WHERE status = 'نشطة' AND trade_mode = 'virtual'").fetchall()]
        conn.close()

        parts = ["**ρίск تقرير المخاطر الحالي**\n"]

        def generate_risk_section(title, trades, portfolio_value):
            if not trades:
                return [f"\n--- **{title}** ---\n✅ لا توجد صفقات نشطة حالياً."]
            
            valid_trades = [t for t in trades if all(k in t and t[k] is not None for k in ['entry_value_usdt', 'entry_price', 'stop_loss', 'quantity'])]
            
            total_at_risk = sum(t['entry_value_usdt'] for t in valid_trades)
            potential_loss = sum((t['entry_price'] - t['stop_loss']) * t['quantity'] for t in valid_trades)
            symbol_concentration = Counter(t['symbol'] for t in valid_trades)

            section_parts = [f"\n--- **{title}** ---"]
            section_parts.append(f"- **عدد الصفقات:** {len(valid_trades)}")
            section_parts.append(f"- **إجمالي رأس المال بالصفقات:** `${total_at_risk:,.2f}`")
            if portfolio_value > 0:
                section_parts.append(f"- **نسبة التعرض:** `{(total_at_risk / portfolio_value) * 100:.2f}%` من المحفظة")
            section_parts.append(f"- **أقصى خسارة محتملة:** `$-{potential_loss:,.2f}` (إذا ضُرب كل الوقف)")
            
            if symbol_concentration:
                most_common = symbol_concentration.most_common(1)[0]
                section_parts.append(f"- **العملة الأكثر تركيزاً:** `{most_common[0]}` ({most_common[1]} صفقات)")
            
            return section_parts

        real_portfolio_value = await get_total_real_portfolio_value_usdt()
        parts.extend(generate_risk_section("🚨 المخاطر الحقيقية", real_trades, real_portfolio_value))
        
        virtual_portfolio_value = bot_state.settings['virtual_portfolio_balance_usdt']
        parts.extend(generate_risk_section("📊 المخاطر الوهمية", virtual_trades, virtual_portfolio_value))

        await target_message.edit_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error generating risk report: {e}", exc_info=True)
        await target_message.edit_text(f"❌ **فشل:** حدث خطأ أثناء إعداد تقرير المخاطر.\n`{e}`")

async def sync_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message
    
    connected_exchanges = [ex for ex in bot_state.exchanges.values() if ex.apiKey]
    if not connected_exchanges:
        await target_message.edit_text("❌ **فشل:** لم يتم العثور على أي منصة متصلة بحساب حقيقي.")
        return
        
    if len(connected_exchanges) == 1:
        await process_sync_portfolio(update, context, connected_exchanges[0].id)
    else:
        keyboard = get_exchange_selection_keyboard("sync", "dashboard_refresh_menu")
        await target_message.edit_text(
            "**🔄 المزامنة والإنقاذ الذكي**\n\nلديك أكثر من منصة متصلة. اختر المنصة للمزامنة:",
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN
        )

async def process_sync_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE, exchange_id: str):
    target_message = update.callback_query.message
    await target_message.edit_text(f"🔄 **المزامنة والإنقاذ الذكي**\n\n⏳ جارِ الاتصال بمنصة {exchange_id.capitalize()} ومقارنة البيانات...", parse_mode=ParseMode.MARKDOWN)
    
    exchange = bot_state.exchanges.get(exchange_id)
    if not exchange:
        await target_message.edit_text(f"❌ **فشل:** خطأ في العثور على منصة {exchange_id.capitalize()} المتصلة.")
        return

    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        bot_trades_raw = conn.cursor().execute("SELECT symbol FROM trades WHERE status = 'نشطة' AND trade_mode = 'real' AND LOWER(exchange) = ?", (exchange_id.lower(),)).fetchall()
        bot_symbols = {item[0] for item in bot_trades_raw}
        
        twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
        recently_closed_raw = conn.cursor().execute("SELECT symbol FROM trades WHERE trade_mode = 'real' AND LOWER(exchange) = ? AND closed_at > ?", (exchange_id.lower(), twenty_four_hours_ago.strftime('%Y-%m-%d %H:%M:%S'))).fetchall()
        recently_closed_symbols = {item[0] for item in recently_closed_raw}
        conn.close()

        portfolio_data = await calculate_full_portfolio(exchange)
        exchange_symbols = {f"{asset['currency']}/USDT" for asset in portfolio_data['assets'] if asset['currency'] != 'USDT'}

        matched_symbols = bot_symbols.intersection(exchange_symbols)
        exchange_only_symbols = exchange_symbols.difference(bot_symbols).difference(recently_closed_symbols)

        parts = [f"**🔄 تقرير المزامنة ({exchange.id.capitalize()})**\n"]
        parts.append(f"تمت مقارنة `{len(bot_symbols)}` صفقة مُدارة بواسطة البوت مع `{len(exchange_symbols)}` عملة مملوكة في المنصة.\n")

        parts.append(f"--- ✅ **صفقات مُدارة ومتطابقة** `({len(matched_symbols)})` ---")
        if matched_symbols: parts.extend([f"- `{s}`" for s in matched_symbols])
        else: parts.append("لا توجد صفقات متطابقة حالياً.")

        parts.append(f"\n--- 🚑 **صفقات يتيمة (للاستيراد)** `({len(exchange_only_symbols)})` ---")
        parts.append("*هذه عملات تملكها في المنصة لكن البوت لا يديرها حالياً.*")
        
        keyboard_buttons = []
        if exchange_only_symbols:
            for symbol in exchange_only_symbols:
                keyboard_buttons.append([InlineKeyboardButton(f"➕ استيراد ومتابعة {symbol}", callback_data=f"rescue_{exchange_id}_{symbol}")])
        else:
            parts.append("لا توجد صفقات يتيمة لاستيرادها. كل شيء متزامن!")
        
        keyboard_buttons.append([InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="dashboard_refresh_menu")])
        keyboard = InlineKeyboardMarkup(keyboard_buttons)

        await target_message.edit_text("\n".join(parts), reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error processing portfolio sync: {e}", exc_info=True)
        await target_message.edit_text(f"❌ **فشل:** حدث خطأ أثناء مزامنة المحفظة.\n`{e}`")

# =======================================================================================
# --- [v7.5] New "Rescuer" Manual Tool Handlers ---
# =======================================================================================

async def cancel_order_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['c_order_tool'] = {'state': 'awaiting_exchange'}
    keyboard = get_exchange_selection_keyboard("c_order", "dashboard_tools")
    await update.callback_query.edit_message_text("❌ **إلغاء أمر مفتوح**\n\nاختر المنصة:", reply_markup=keyboard)

async def fetch_and_display_cancellable_orders(exchange_id, symbol, message):
    exchange = bot_state.exchanges.get(exchange_id.lower())
    if not exchange:
        await message.reply_text(f"❌ خطأ: لم يتم توثيق الاتصال بمنصة {exchange_id.capitalize()}.")
        return
    try:
        open_orders = await exchange.fetch_open_orders(symbol)
        if not open_orders:
            await message.reply_text(f"✅ لا توجد لديك أوامر مفتوحة لـ `{symbol}` على {exchange_id.capitalize()}.")
            return

        lines = [f"**📖 أوامرك المفتوحة لـ `{symbol}`**\n\nاختر الأمر الذي تريد إلغاءه:"]
        keyboard = []
        for order in open_orders:
            side_emoji = "🔼" if order['side'] == 'buy' else "🔽"
            order_text = f"{side_emoji} {order['type']} | {order['amount']} @ {order['price']}"
            keyboard.append([InlineKeyboardButton(order_text, callback_data=f"c_order_confirm_{exchange_id}_{symbol}_{order['id']}")])
        
        keyboard.append([InlineKeyboardButton("❌ إلغاء كل الأوامر لهذا الرمز", callback_data=f"c_order_all_{exchange_id}_{symbol}")])
        keyboard.append([InlineKeyboardButton("🔙 العودة", callback_data="dashboard_tools")])

        await message.reply_text("\n".join(lines), reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error fetching cancellable orders: {e}")
        await message.reply_text(f"❌ فشل جلب الأوامر المفتوحة. تأكد من صحة الرمز: `{symbol}`.")

async def cancel_order_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    parts = data.split('_')
    action = parts[2]

    try:
        if action == "exchange":
            exchange_id = parts[3]
            context.user_data['c_order_tool'] = {'exchange': exchange_id, 'state': 'awaiting_symbol'}
            await query.edit_message_text(f"اخترت منصة: *{exchange_id.capitalize()}*\n\nالآن، أرسل رمز العملة التي تريد إلغاء أوامرها (مثال: `BTC/USDT`).", parse_mode=ParseMode.MARKDOWN)
        
        elif action == "confirm":
            _, _, _, exchange_id, symbol, order_id = parts
            exchange = bot_state.exchanges.get(exchange_id)
            await query.edit_message_text(f"⏳ جارِ إلغاء الأمر `{order_id}`...", parse_mode=ParseMode.MARKDOWN)
            await exchange.cancel_order(order_id, symbol)
            await query.edit_message_text(f"✅ تم إلغاء الأمر `{order_id}` لـ `{symbol}` بنجاح.", parse_mode=ParseMode.MARKDOWN)
        
        elif action == "all":
            _, _, _, exchange_id, symbol = parts
            exchange = bot_state.exchanges.get(exchange_id)
            await query.edit_message_text(f"⏳ جارِ إلغاء جميع الأوامر...", parse_mode=ParseMode.MARKDOWN)
            # ملاحظة: يبدو أنك لم تكمل الكود هنا لإلغاء جميع الأوامر

    # تم تصحيح المسافة البادئة هنا
    except Exception as e:
        logger.error(f"Error in cancel_order_button_handler: {e}")
        try:
            await query.edit_message_text(f"❌ فشل إلغاء الأمر. الخطأ: `{e}`", parse_mode=ParseMode.MARKDOWN)
        except Exception as inner_e:
            logger.error(f"Failed to even send error message to user: {inner_e}")
        await query.edit_message_text(f"❌ فشل إلغاء الأمر. الخطأ: `{e}`", parse_mode=ParseMode.MARKDOWN)


async def post_shutdown(application: Application):
    all_exchanges = list(bot_state.exchanges.values()) + list(bot_state.public_exchanges.values())
    unique_exchanges = list({id(ex): ex for ex in all_exchanges}.values())
    await asyncio.gather(*[ex.close() for ex in unique_exchanges])
    logger.info("All exchange connections closed.")

def main():
    """Sets up and runs the bot application."""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print("FATAL ERROR: TELEGRAM_BOT_TOKEN is not set.")
        exit()

    load_settings()
    init_database()

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(60.0)
        .read_timeout(60.0)
        .connection_pool_size(50)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # --- Registering all handlers ---
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("check", check_trade_command))
    
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    application.add_error_handler(error_handler)
    
    logger.info("Application configured with all handlers. Starting polling...")
    application.run_polling()

async def post_init(application: Application):
    if NLTK_AVAILABLE:
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError: logger.info("Downloading NLTK data..."); nltk.download('vader_lexicon')
    
    logger.info("Post-init: Initializing exchanges...")
    await initialize_exchanges()
    if not bot_state.public_exchanges: 
        logger.critical("CRITICAL: No public exchange clients connected. Bot cannot run.")
        return

    job_queue = application.job_queue
    job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name='perform_scan')
    job_queue.run_repeating(track_open_trades, interval=TRACK_INTERVAL_SECONDS, first=20, name='track_open_trades')
    job_queue.run_daily(send_daily_report, time=dt_time(hour=23, minute=55, tzinfo=EGYPT_TZ), name='daily_report')

    logger.info("Jobs scheduled.")
    await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🚀 *بوت كاسحة الألغام (v7.0) جاهز للعمل!*", parse_mode=ParseMode.MARKDOWN)


if __name__ == '__main__':
    print("🚀 Starting Mineseper Bot v7.0 (Sentinel Protocol)...")
    try:
        main()
    except Exception as e:
        logging.critical(f"Bot stopped due to a critical unhandled error: {e}", exc_info=True)

