# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 OKX Bot v12.0 (The Phoenix - The Forensic Investigator) 🚀 ---
# =======================================================================================
# This version addresses the final root cause: intermittent API failures when fetching
# the balance. The `ensure_trading_balance` function is now a "Forensic Investigator".
# It no longer trusts a single API call. It will retry fetching the balance multiple
# times if the initial attempt fails or returns an empty result. This provides extreme
# resilience against temporary network/API glitches that previously caused false
# "Insufficient Funds" errors. This is the definitive reliability fix.
# =======================================================================================

# --- Libraries ---
import asyncio
import os
import logging
import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict
import aiosqlite
import httpx
import websockets
import hmac
import base64
import time
import traceback

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
DB_FILE = os.path.join(APP_ROOT, 'okx_phoenix_v12.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'okx_phoenix_settings_v12.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("OKX_Phoenix_v12_Investigator")

class BotState:
    def __init__(self):
        self.exchange = None
        self.settings = {}
        self.last_signal_time = {}
        self.market_mood = {"mood": "UNKNOWN", "reason": "تحليل لم يتم بعد"}
        self.scan_stats = {"last_start": None, "last_duration": "N/A"}
        self.ws_manager = None
        self.application = None
        self.trade_manager = None 

bot_state = BotState()
scan_lock = asyncio.Lock()

# =======================================================================================
# --- UI Constants ---
# =======================================================================================
DEFAULT_SETTINGS = {
    "active_preset": "PRO", "real_trade_size_usdt": 15.0, "top_n_symbols_by_volume": 200,
    "atr_sl_multiplier": 2.5, "risk_reward_ratio": 2.0,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "whale_radar", "sniper_pro"],
    "market_mood_filter_enabled": True, "fear_and_greed_threshold": 30,
    "liquidity_filters": {"min_quote_volume_24h_usd": 5000000, "max_spread_percent": 0.5, "min_rvol": 1.5},
    "volatility_filters": {"min_atr_percent": 0.8}, "trend_filters": {"ema_period": 200, "htf_period": 50},
    "min_tp_sl_filter": {"min_tp_percent": 1.0, "min_sl_percent": 0.5},
    "scan_interval_seconds": 900, "track_interval_seconds": 60, "trailing_sl_enabled": True,
    "trailing_sl_activation_percent": 1.5, "trailing_sl_callback_percent": 1.0
}
PRESETS = {
    "PRO": {"liquidity_filters": {"min_quote_volume_24h_usd": 5000000, "min_rvol": 1.5}, "volatility_filters": {"min_atr_percent": 0.8}, "name": "🚦 احترافية (سيولة عالية)"},
    "STRICT": {"liquidity_filters": {"min_quote_volume_24h_usd": 10000000, "min_rvol": 2.2}, "volatility_filters": {"min_atr_percent": 1.4}, "name": "🎯 متشددة (سيولة فائقة)"},
    "LAX": {"liquidity_filters": {"min_quote_volume_24h_usd": 2000000, "min_rvol": 1.1}, "volatility_filters": {"min_atr_percent": 0.4}, "name": "🌙 متساهلة (سيولة متوسطة)"},
    "VERY_LAX": {"liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "min_rvol": 0.8}, "volatility_filters": {"min_atr_percent": 0.2}, "name": "⚠️ فائق التساهل (تجريبي)"}
}
EDITABLE_PARAMS = {
    "إعدادات المخاطر": ["real_trade_size_usdt", "atr_sl_multiplier", "risk_reward_ratio"],
    "إعدادات الوقف المتحرك": ["trailing_sl_enabled", "trailing_sl_activation_percent", "trailing_sl_callback_percent"],
    "إعدادات الفحص والسيولة": ["top_n_symbols_by_volume", "fear_and_greed_threshold", "market_mood_filter_enabled", "scan_interval_seconds", "min_quote_volume_24h_usd"]
}
PARAM_DISPLAY_NAMES = {
    "real_trade_size_usdt": "💵 حجم الصفقة ($)", "atr_sl_multiplier": "مضاعف وقف الخسارة (ATR)",
    "risk_reward_ratio": "نسبة المخاطرة/العائد", "trailing_sl_enabled": "⚙️ تفعيل الوقف المتحرك",
    "trailing_sl_activation_percent": "تفعيل الوقف المتحرك (%)", "trailing_sl_callback_percent": "مسافة تتبع الوقف (%)",
    "top_n_symbols_by_volume": "عدد العملات للفحص", "fear_and_greed_threshold": "حد مؤشر الخوف",
    "market_mood_filter_enabled": "فلتر مزاج السوق", "scan_interval_seconds": "⏱️ الفاصل الزمني للفحص (ثواني)",
    "min_quote_volume_24h_usd": "حد السيولة الأدنى ($)"
}
STRATEGIES_MAP = {
    "momentum_breakout": {"func_name": "analyze_momentum_breakout", "name": "زخم اختراقي"},
    "breakout_squeeze_pro": {"func_name": "analyze_breakout_squeeze_pro", "name": "اختراق انضغاطي"},
    "support_rebound": {"func_name": "analyze_support_rebound", "name": "ارتداد الدعم"},
    "sniper_pro": {"func_name": "analyze_sniper_pro", "name": "القناص المحترف"},
    "whale_radar": {"func_name": "analyze_whale_radar", "name": "رادار الحيتان"},
}

# =======================================================================================
# --- 🛠️ Helper Functions (Defined before use) 🛠️ ---
# =======================================================================================
async def ensure_libraries_loaded():
    global pd, ta, ccxt
    if pd is None: logger.info("تحميل مكتبة pandas لأول مرة..."); import pandas as pd_lib; pd = pd_lib
    if ta is None: logger.info("تحميل مكتبة pandas-ta لأول مرة..."); import pandas_ta as ta_lib; ta = ta_lib
    if ccxt is None: logger.info("تحميل مكتبة ccxt لأول مرة..."); import ccxt.async_support as ccxt_lib; ccxt = ccxt_lib

def escape_markdown(text: str) -> str:
    if not isinstance(text, str): text = str(text)
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def create_safe_task(coro):
    async def task_wrapper():
        try:
            await coro
        except Exception as e:
            logger.critical(f"Unhandled exception in background task '{coro.__name__}': {e}", exc_info=True)
    asyncio.create_task(task_wrapper())

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f: bot_state.settings = json.load(f)
        else: bot_state.settings = DEFAULT_SETTINGS.copy()
        for key, default_value in DEFAULT_SETTINGS.items():
            if key not in bot_state.settings:
                bot_state.settings[key] = default_value
            elif isinstance(default_value, dict):
                for sub_key, sub_default_value in default_value.items():
                    if sub_key not in bot_state.settings.get(key, {}):
                        bot_state.settings[key][sub_key] = sub_default_value
        save_settings()
        logger.info("Settings loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        bot_state.settings = DEFAULT_SETTINGS.copy()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(bot_state.settings, f, indent=4)
    except Exception as e: logger.error(f"Failed to save settings: {e}")

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
        logger.info(f"Database initialized/verified at: {DB_FILE}")
    except Exception as e: logger.error(f"Failed to initialize database: {e}")

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
        df['sma'] = ta.sma(df['close'], length=htf_period)
        is_btc_bullish = df['close'].iloc[-1] > df['sma'].iloc[-1]
        btc_mood_text = "إيجابي ✅" if is_btc_bullish else "سلبي ❌"
        if not is_btc_bullish:
            return {"mood": "NEGATIVE", "reason": "اتجاه BTC هابط (تحت متوسط 50 على 4 ساعات)", "btc_mood": btc_mood_text, "fng": "N/A"}
    except Exception as e:
        logger.warning(f"Could not fetch BTC trend: {e}")
        return {"mood": "DANGEROUS", "reason": "فشل جلب بيانات BTC", "btc_mood": "UNKNOWN", "fng": "N/A"}
    
    fng = await get_fear_and_greed_index()
    fng_text = str(fng) if fng is not None else "N/A"
    if fng is not None and fng < bot_state.settings['fear_and_greed_threshold']:
        return {"mood": "NEGATIVE", "reason": f"مشاعر خوف شديد (مؤشر F&G: {fng})", "btc_mood": btc_mood_text, "fng": fng_text}
        
    return {"mood": "POSITIVE", "reason": "وضع السوق مناسب", "btc_mood": btc_mood_text, "fng": fng_text}

# =======================================================================================
# --- 🔬 Analysis Functions 🔬 ---
# =======================================================================================
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None
# ... (All analysis functions are identical to previous version) ...
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
# --- 🎻 The Symphony Conductor: TradeManager 🎻 ---
# =======================================================================================
class TradeManager:
    # ... (Class definition as before) ...
    def __init__(self, exchange, bot):
        self.exchange = exchange
        self.bot = bot
        self.active_trades = {}
        self.tsl_locks = defaultdict(asyncio.Lock)
        logger.info("🎼 TradeManager (The Conductor) has been initialized.")

    async def register_new_trade(self, signal, buy_order):
        trade_id = await self.log_trade_to_db(signal, buy_order)
        if trade_id:
            trade_info = {
                'id': trade_id, 'symbol': signal['symbol'], 'status': 'pending_protection',
                'order_id': buy_order['id'], 'reason': signal['reason'], 
                'entry_price_signal': signal['entry_price'], 'stop_loss_signal': signal['stop_loss'],
                'rr_ratio': signal['risk_reward_ratio']
            }
            self.active_trades[buy_order['id']] = trade_info
            logger.info(f"Conductor: Registered new trade #{trade_id} ({signal['symbol']}) for monitoring.")
            msg = f"**⏳ تم بدء صفقة | {signal['symbol']} (ID: {trade_id})**\nتم إرسال أمر الشراء المحدد بنجاح.\nيقوم 'قائد الأوركسترا' الآن بمراقبة الأمر لتأمينه."
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
        return trade_id

    async def process_fill_event(self, order_data):
        order_id = order_data['ordId']
        symbol = order_data['instId'].replace('-', '/')
        logger.info(f"Conductor: Received fill event for order {order_id} ({symbol}).")
        
        trade_info = self.active_trades.get(order_id)
        if not trade_info or trade_info['status'] != 'pending_protection':
            logger.warning(f"Conductor: Received fill for an unknown or already-processed order {order_id}. Ignoring.")
            return

        try:
            filled_qty = float(order_data.get('fillSz', 0))
            avg_price = float(order_data.get('avgPx', 0))
            if filled_qty == 0 or avg_price == 0:
                raise ValueError("Fill event has zero quantity or price.")

            trade_info['status'] = 'protecting'
            
            original_risk = trade_info['entry_price_signal'] - trade_info['stop_loss_signal']
            final_tp = avg_price + (original_risk * trade_info['rr_ratio'])
            final_sl = avg_price - original_risk

            oco_params = {
                'instId': self.exchange.market_id(symbol), 'tdMode': 'cash', 'side': 'sell', 'ordType': 'oco',
                'sz': str(self.exchange.amount_to_precision(symbol, filled_qty)),
                'tpTriggerPx': str(self.exchange.price_to_precision(symbol, final_tp)), 'tpOrdPx': '-1',
                'slTriggerPx': str(self.exchange.price_to_precision(symbol, final_sl)), 'slOrdPx': '-1'
            }
            
            for attempt in range(3):
                oco_receipt = await self.exchange.private_post_trade_order_algo(oco_params)
                if oco_receipt and oco_receipt['data'][0].get('sCode') == '0':
                    algo_id = oco_receipt['data'][0]['algoId']
                    logger.info(f"Conductor: Successfully placed OCO for trade #{trade_info['id']}. Algo ID: {algo_id}")
                    await self.update_trade_in_db_as_active(trade_info['id'], avg_price, filled_qty, final_tp, final_sl, algo_id)
                    
                    trade_info.update({
                        'status': 'active', 'algo_id': algo_id, 'quantity': filled_qty,
                        'entry_price': avg_price, 'stop_loss': final_sl, 'take_profit': final_tp,
                        'highest_price': avg_price, 'trailing_sl_active': False
                    })
                    
                    tp_percent = (final_tp / avg_price - 1) * 100
                    sl_percent = (1 - final_sl / avg_price) * 100
                    msg = (f"**✅🛡️ صفقة مصفحة | {symbol} (ID: {trade_info['id']})**\n🔍 **الاستراتيجية:** {trade_info['reason']}\n\n"
                           f"📈 **الشراء:** `{avg_price:,.4f}`\n🎯 **الهدف:** `{final_tp:,.4f}` (+{tp_percent:.2f}%)\n"
                           f"🛑 **الوقف:** `{final_sl:,.4f}` (-{sl_percent:.2f}%)\n\n***تم تأمين الصفقة بنجاح عبر قائد الأوركسترا.***")
                    await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
                    return
                else:
                    logger.warning(f"Conductor: OCO attempt {attempt+1} failed for {symbol}. Response: {json.dumps(oco_receipt)}")
                    await asyncio.sleep(2)
            
            raise Exception("All OCO placement attempts failed.")

        except Exception as e:
            logger.critical(f"Conductor: CRITICAL FAILURE while protecting trade #{trade_info.get('id', 'N/A')} ({symbol}): {e}", exc_info=True)
            msg = (f"**🔥🔥🔥 فشل حرج في الأوركسترا - {symbol}**\n\n🚨 **خطر!** تم تنفيذ الشراء ولكن **فشل وضع الحماية تلقائيًا**.\n"
                   f"**معرف الأمر:** `{order_id}`\n\n**❗️ تدخل يدوي فوري ضروري لوضع وقف الخسارة!**")
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
            if trade_info: trade_info['status'] = 'failed_protection'
    
    async def track_all_trades(self, settings):
        active_trades_copy = list(self.active_trades.values())
        for trade in active_trades_copy:
            if trade.get('status') == 'active' and settings.get('trailing_sl_enabled', False):
                await self.process_trailing_stop_loss(trade, settings)

    async def process_trailing_stop_loss(self, trade, settings):
        async with self.tsl_locks[trade['id']]:
            try:
                symbol = trade['symbol']
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = ticker.get('last')
                if not current_price: return

                highest_price = max(trade.get('highest_price', trade['entry_price']), current_price)
                if highest_price > trade.get('highest_price', 0):
                    trade['highest_price'] = highest_price
                    await self.update_db_field(trade['id'], 'highest_price', highest_price)
                
                activation_price = trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100)
                
                if not trade.get('trailing_sl_active') and current_price >= activation_price:
                    logger.info(f"TSL Activated for {symbol} at price {current_price}")
                    trade['trailing_sl_active'] = True
                    await self.update_db_field(trade['id'], 'trailing_sl_active', 1)
                    await self.bot.send_message(TELEGRAM_CHAT_ID, f"**🚀 الوقف المتحرك مُفعّل لـ {symbol}**", parse_mode=ParseMode.MARKDOWN)
                
                if trade.get('trailing_sl_active'):
                    new_stop_loss = highest_price * (1 - (settings['trailing_sl_callback_percent'] / 100))
                    if new_stop_loss > trade['stop_loss']:
                        logger.info(f"Trailing SL for {symbol}. New SL: {new_stop_loss}")
                        await self.exchange.cancel_order(trade['algo_id'], symbol, {'algo': True})
                        
                        new_oco_params = {
                            'instId': self.exchange.market_id(symbol), 'tdMode': 'cash', 'side': 'sell', 'ordType': 'oco',
                            'sz': str(self.exchange.amount_to_precision(symbol, trade['quantity'])),
                            'tpTriggerPx': str(self.exchange.price_to_precision(symbol, trade['take_profit'])), 'tpOrdPx': '-1',
                            'slTriggerPx': str(self.exchange.price_to_precision(symbol, new_stop_loss)), 'slOrdPx': '-1'
                        }
                        new_oco_receipt = await self.exchange.private_post_trade_order_algo(new_oco_params)
                        if new_oco_receipt and new_oco_receipt['data'][0].get('sCode') == '0':
                            new_algo_id = new_oco_receipt['data'][0]['algoId']
                            trade['stop_loss'] = new_stop_loss
                            trade['algo_id'] = new_algo_id
                            await self.update_db_after_tsl(trade['id'], new_stop_loss, new_algo_id)
                            sl_percent = (1 - new_stop_loss / trade['entry_price']) * 100
                            await self.bot.send_message(TELEGRAM_CHAT_ID, f"**📈 تم رفع وقف الخسارة لـ {symbol}**\n**الوقف الجديد:** `{new_stop_loss:,.4f}` (الآن عند {sl_percent:+.2f}%)", parse_mode=ParseMode.MARKDOWN)
                        else:
                            logger.error(f"Failed to place new TSL OCO for {symbol}. Response: {json.dumps(new_oco_receipt)}")
            except Exception as e:
                logger.error(f"Error during TSL for trade #{trade['id']} ({trade.get('symbol', 'N/A')}): {e}", exc_info=True)

    async def log_trade_to_db(self, signal, buy_order):
        try:
            async with aiosqlite.connect(DB_FILE) as conn:
                sql = '''INSERT INTO trades (timestamp, symbol, entry_price, take_profit, stop_loss, quantity, reason, order_id, status, entry_value_usdt, highest_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
                params = (datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), signal['symbol'], signal['entry_price'], signal['take_profit'], signal['stop_loss'], buy_order['amount'], signal['reason'], buy_order['id'], 'pending_protection', buy_order['cost'], signal['entry_price'])
                cursor = await conn.execute(sql, params)
                await conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"DB_HELPER: Failed to log initial trade: {e}", exc_info=True); return None
    async def update_trade_in_db_as_active(self, db_id, avg_price, filled_qty, final_tp, final_sl, algo_id):
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("UPDATE trades SET status = 'active', entry_price = ?, quantity = ?, entry_value_usdt = ?, take_profit = ?, stop_loss = ?, algo_id = ?, highest_price = ? WHERE id = ?", (avg_price, filled_qty, avg_price * filled_qty, final_tp, final_sl, algo_id, avg_price, db_id)); await conn.commit()
    async def update_db_field(self, db_id, field, value):
         async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute(f"UPDATE trades SET {field} = ? WHERE id = ?", (value, db_id)); await conn.commit()
    async def update_db_after_tsl(self, db_id, new_sl, new_algo_id):
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("UPDATE trades SET stop_loss = ?, algo_id = ? WHERE id = ?", (new_sl, new_algo_id, db_id)); await conn.commit()

# =======================================================================================
# --- 💸 The Treasurer & Core Logic 💸 ---
# =======================================================================================
class WebSocketManager:
    def __init__(self, trade_manager):
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/private?brokerId=aws"
        self.websocket = None
        self.trade_manager = trade_manager
        logger.info(f"[WS-Manager] Initialized.")
    def is_connected(self):
        return self.websocket is not None and self.websocket.state == websockets.protocol.State.OPEN
    def _get_auth_args(self):
        timestamp = str(time.time()); message = timestamp + 'GET' + '/users/self/verify'
        mac = hmac.new(bytes(OKX_API_SECRET, 'utf8'), bytes(message, 'utf8'), 'sha256')
        sign = base64.b64encode(mac.digest()).decode()
        return [{"apiKey": OKX_API_KEY, "passphrase": OKX_API_PASSPHRASE, "timestamp": timestamp, "sign": sign}]
    async def _message_handler(self, message):
        if message == 'ping': await self.websocket.send('pong'); return
        try:
            data = json.loads(message)
            if data.get('arg', {}).get('channel') == 'orders':
                for order_data in data.get('data', []):
                    if order_data.get('state') == 'filled' and order_data.get('side') == 'buy':
                        create_safe_task(self.trade_manager.process_fill_event(order_data))
        except Exception as e: logger.error(f"Error in WebSocket message handler: {e}", exc_info=True)
    async def run(self):
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as websocket:
                    self.websocket = websocket
                    logger.info("✅ [WS-Private] Connected. Authenticating...")
                    await websocket.send(json.dumps({"op": "login", "args": self._get_auth_args()}))
                    login_response = json.loads(await websocket.recv())
                    if login_response.get('event') == 'login' and login_response.get('code') == '0':
                        logger.info("🔐 [WS-Private] Authenticated. Subscribing...")
                        await websocket.send(json.dumps({"op": "subscribe", "args": [{"channel": "orders", "instType": "SPOT"}]}))
                        sub_response = json.loads(await websocket.recv())
                        if sub_response.get('event') == 'subscribe': logger.info(f"📈 [WS-Private] Subscribed to: {sub_response.get('arg')}")
                    else: logger.error(f"🔥 [WS-Private] Authentication failed: {login_response}"); await asyncio.sleep(10); continue
                    async for message in websocket: await self._message_handler(message)
            except Exception as e: logger.error(f"🔥 [WS-Private] Unhandled exception in WebSocket loop: {e}", exc_info=True)
            self.websocket = None; logger.info("Reconnecting in 5 seconds..."); await asyncio.sleep(5)

async def ensure_trading_balance(exchange, required_amount):
    for attempt in range(5):
        try:
            balances = await exchange.fetch_balance()
            if not balances:
                logger.warning(f"Treasurer (Attempt {attempt+1}/5): fetch_balance returned empty. Retrying...")
                await asyncio.sleep(2)
                continue

            trading_balance = balances.get('trading', {}).get('USDT', {}).get('free', 0.0)
            if trading_balance >= required_amount:
                logger.info(f"Treasurer: Sufficient USDT in TRADING account ({trading_balance:.2f}).")
                return True

            funding_balance = balances.get('funding', {}).get('USDT', {}).get('free', 0.0)
            amount_to_transfer = required_amount - trading_balance
            
            if funding_balance >= amount_to_transfer:
                logger.info(f"Treasurer: Insufficient TRADING balance. Transferring {amount_to_transfer:.2f} USDT from FUNDING.")
                await exchange.transfer('USDT', amount_to_transfer, 'funding', 'trading')
                logger.info("Treasurer: Transfer successful.")
                return True
            else:
                logger.error(f"Treasurer: Insufficient funds in both TRADING ({trading_balance:.2f}) and FUNDING ({funding_balance:.2f}) accounts.")
                return False
        except Exception as e:
            logger.error(f"Treasurer (Attempt {attempt+1}/5): Error during balance check/transfer: {e}", exc_info=True)
            await asyncio.sleep(2)
    logger.critical("Treasurer: All attempts to fetch balance and transfer funds failed.")
    return False

async def initiate_trade(signal, bot: "telegram.Bot"):
    await ensure_libraries_loaded()
    symbol, settings, exchange = signal['symbol'], bot_state.settings, bot_state.exchange
    logger.info(f"Initiating trade for {symbol}.")
    try:
        trade_size = settings['real_trade_size_usdt']
        if not await ensure_trading_balance(exchange, trade_size * 1.02):
            raise Exception("Treasurer check failed. Insufficient funds after retries.")
        ticker = await exchange.fetch_ticker(symbol)
        limit_price = ticker['ask'] 
        if not limit_price or limit_price <= 0: raise ValueError(f"Invalid ask price: {limit_price}")
        quantity_to_buy = trade_size / limit_price
        buy_order = await exchange.create_limit_buy_order(symbol, quantity_to_buy, limit_price)
        signal['risk_reward_ratio'] = settings['risk_reward_ratio']
        await bot_state.trade_manager.register_new_trade(signal, buy_order)
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE during trade initiation for {symbol}: {e}", exc_info=True)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"**🔥🔥🔥 فشل حرج عند بدء صفقة {symbol}**\n\n**الخطأ:** `{str(e)}`", parse_mode=ParseMode.MARKDOWN)

async def worker(queue, signals_list, failure_counter):
    # ... (Worker logic is identical to previous version) ...
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
                    if (take_profit/entry_price - 1)*100 >= settings['min_tp_sl_filter']['min_tp_percent'] and \
                       (1 - stop_loss/entry_price)*100 >= settings['min_tp_sl_filter']['min_sl_percent']:
                        signals_list.append({"symbol": symbol, "reason": reason_str, "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss})
        except Exception as e:
            logger.debug(f"Worker error for {symbol}: {e}")
            failure_counter[0] += 1
        finally: queue.task_done()

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
            min_vol = settings.get('liquidity_filters', {}).get('min_quote_volume_24h_usd', 5000000)
            usdt_markets = [m for m in tickers.values() if m.get('symbol','').endswith('/USDT') and not any(k in m['symbol'] for k in ['-SWAP','UP','DOWN','3L','3S']) and m.get('quoteVolume', 0) > min_vol]
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
        
        bot_state.scan_stats['failures'] = failure_counter[0]
        duration = (datetime.now(EGYPT_TZ) - bot_state.scan_stats['last_start']).total_seconds()
        bot_state.scan_stats['last_duration'] = f"{duration:.0f} ثانية"
        new_trades = 0
        if signals_found:
            logger.info(f"+++ Scan complete. Found {len(signals_found)} signals! +++")
            for signal in signals_found:
                if time.time() - bot_state.last_signal_time.get(signal['symbol'], 0) < settings['scan_interval_seconds'] * 2.5: continue
                try:
                    await initiate_trade(signal, bot)
                    new_trades += 1
                    logger.info("Waiting for 25 seconds before attempting next trade...")
                    await asyncio.sleep(25)
                except Exception as e:
                    # initiate_trade already logs and sends a message, so we just break the loop
                    logger.error(f"Stopping scan loop due to critical error in initiate_trade for {signal['symbol']}.")
                    break
        else: logger.info("--- Scan complete. No new signals found. ---")
        summary = (f"**🔬 ملخص الفحص الأخير**\n\n- **الحالة:** اكتمل بنجاح\n- **وضع السوق:** {bot_state.market_mood['mood']} ({bot_state.market_mood.get('btc_mood', 'N/A')})\n"
                   f"- **المدة:** {bot_state.scan_stats['last_duration']}\n- **العملات المفحوصة:** {bot_state.scan_stats['markets_scanned']}\n\n------------------------------------\n"
                   f"- **إجمالي الإشارات:** {len(signals_found)}\n- **✅ صفقات جديدة:** {new_trades}\n- **⚠️ أخطاء التحليل:** {bot_state.scan_stats['failures']}")
        await bot.send_message(TELEGRAM_CHAT_ID, summary, parse_mode=ParseMode.MARKDOWN)

async def track_open_trades(context: ContextTypes.DEFAULT_TYPE):
    if bot_state.trade_manager:
        await bot_state.trade_manager.track_all_trades(bot_state.settings)

# =======================================================================================
# --- 📱 Telegram UI Handlers 📱 ---
# =======================================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Dashboard 🖥️"], ["⚙️ الإعدادات"]]
    await update.message.reply_text("أهلاً بك في بوت OKX القناص v11.0 (The Final Blueprint)", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True))

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 الإحصائيات العامة", callback_data="dashboard_stats")],
        [InlineKeyboardButton("📈 الصفقات النشطة", callback_data="dashboard_active_trades")],
        [InlineKeyboardButton("📜 تقرير أداء الاستراتيجيات", callback_data="dashboard_strategy_report")],
        [InlineKeyboardButton("🌡️ حالة مزاج السوق", callback_data="dashboard_mood"), InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="dashboard_diagnostics")]
    ]
    target_message = update.message or update.callback_query.message
    await target_message.reply_text("🖥️ *لوحة التحكم الرئيسية*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["🎭 تفعيل/تعطيل الماسحات", "🔧 تعديل المعايير"], ["🏁 الأنماط الجاهزة"], ["🔙 القائمة الرئيسية"]]
    target_message = update.message or update.callback_query.message
    await target_message.reply_text("اختر الإعداد:", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True))
async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text
    if 'awaiting_input_for_param' in context.user_data:
        param_key, msg_to_del, original_menu_msg_id = context.user_data.pop('awaiting_input_for_param')
        new_value_str = update.message.text; settings = bot_state.settings
        try:
            target_dict = settings
            if param_key == "min_quote_volume_24h_usd": target_dict = settings['liquidity_filters']
            current_value = target_dict.get(param_key)
            if isinstance(current_value, bool): new_value = new_value_str.lower() in ['true', '1', 'on', 'yes', 'نعم', 'تفعيل']
            elif isinstance(current_value, float): new_value = float(new_value_str)
            else: new_value = int(new_value_str)
            target_dict[param_key] = new_value; save_settings()
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=msg_to_del)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
            await show_parameters_menu(update, context, edit_message_id=original_menu_msg_id)
            confirm_msg = await update.message.reply_text(f"✅ تم تحديث **{PARAM_DISPLAY_NAMES.get(param_key, param_key)}** إلى `{new_value}`.", parse_mode=ParseMode.MARKDOWN)
            context.job_queue.run_once(lambda ctx: ctx.bot.delete_message(confirm_msg.chat.id, confirm_msg.message_id), 5)
        except (ValueError, TypeError): await update.message.reply_text("❌ قيمة غير صالحة. الرجاء المحاولة مرة أخرى.")
        return
    menu_map = {"Dashboard 🖥️": show_dashboard_command, "⚙️ الإعدادات": show_settings_menu, "🎭 تفعيل/تعطيل الماسحات": show_scanners_menu, "🔧 تعديل المعايير": show_parameters_menu, "🏁 الأنماط الجاهزة": show_presets_menu, "🔙 القائمة الرئيسية": start_command}
    if text in menu_map: await menu_map[text](update, context)
async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active = bot_state.settings.get("active_scanners", [])
    keyboard = [[InlineKeyboardButton(f"{'✅' if k in active else '❌'} {v['name']}", callback_data=f"toggle_scanner_{k}")] for k, v in STRATEGIES_MAP.items()]
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    await (update.message or update.callback_query.message).reply_text("اختر الماسحات لتفعيلها أو تعطيلها:", reply_markup=InlineKeyboardMarkup(keyboard))
async def show_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(v['name'], callback_data=f"preset_{k}")] for k,v in PRESETS.items()]
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    await (update.message or update.callback_query.message).reply_text("اختر نمط إعدادات جاهز:", reply_markup=InlineKeyboardMarkup(keyboard))
async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, edit_message_id=None):
    keyboard, settings = [], bot_state.settings
    for category, params in EDITABLE_PARAMS.items():
        keyboard.append([InlineKeyboardButton(f"--- {category} ---", callback_data="ignore")])
        for param_key in params:
            value = settings.get(param_key)
            if param_key == "min_quote_volume_24h_usd": value = settings.get('liquidity_filters', {}).get(param_key)
            name = PARAM_DISPLAY_NAMES.get(param_key, param_key)
            text = f"{name}: {'مُفعّل ✅' if value else 'مُعطّل ❌'}" if isinstance(value, bool) else f"{name}: {value}"
            keyboard.append([InlineKeyboardButton(text, callback_data=f"param_{param_key}")])
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="back_to_settings")])
    target_message, message_text = update.message or update.callback_query.message, "⚙️ *الإعدادات المتقدمة*"
    try:
        if edit_message_id: await context.bot.edit_message_text(chat_id=target_message.chat_id, message_id=edit_message_id, text=message_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
        elif update.callback_query: await update.callback_query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
        else: await target_message.reply_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
    except BadRequest as e:
        if "Message is not modified" not in str(e): logger.warning(f"Could not edit parameters menu: {e}")
async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer()
    data = query.data
    try:
        if data.startswith("dashboard_"):
            if query.message: 
                try: await query.message.delete()
                except BadRequest: pass
            report_type = data.split("_", 1)[1]
            if report_type == "stats":
                async with aiosqlite.connect(DB_FILE) as conn:
                     cursor = await conn.execute("SELECT status, COUNT(*), SUM(pnl_usdt) FROM trades WHERE status NOT IN ('active', 'pending_protection', 'failed_protection') GROUP BY status")
                     stats = await cursor.fetchall()
                counts, pnl = defaultdict(int), defaultdict(float)
                for status, count, p in stats: counts[status], pnl[status] = count, p or 0
                wins = sum(v for k,v in counts.items() if k and k.startswith('ناجحة')); losses = counts.get('فاشلة (وقف خسارة)', 0); closed = wins + losses
                win_rate = (wins / closed * 100) if closed > 0 else 0; total_pnl = sum(pnl.values())
                await query.message.reply_text(f"*📊 الإحصائيات العامة*\n- الصفقات المغلقة: {closed}\n- نسبة النجاح: {win_rate:.2f}%\n- صافي الربح/الخسارة: ${total_pnl:+.2f}", parse_mode=ParseMode.MARKDOWN)
            elif report_type == "active_trades":
                if not bot_state.trade_manager or not bot_state.trade_manager.active_trades:
                    return await query.message.reply_text("لا توجد صفقات نشطة حالياً.")
                keyboard = []; active_trades_list = sorted(bot_state.trade_manager.active_trades.values(), key=lambda t: t['id'], reverse=True)
                for t in active_trades_list:
                    status_emoji = "🛡️" if t.get('status') == 'active' else "⏳" if t.get('status') == 'pending_protection' else "🔥"
                    entry_value = t.get('entry_price', 0) * t.get('quantity', 0)
                    button_text = f"#{t['id']} {status_emoji} | {t['symbol']} | ${entry_value:.2f}"
                    keyboard.append([InlineKeyboardButton(button_text, callback_data=f"check_{t['id']}")])
                await query.message.reply_text("اختر صفقة لمتابعتها:", reply_markup=InlineKeyboardMarkup(keyboard))
            elif report_type == "strategy_report":
                async with aiosqlite.connect(DB_FILE) as conn:
                    cursor = await conn.execute("SELECT reason, status, pnl_usdt FROM trades WHERE status NOT IN ('active', 'pending_protection')")
                    trades = await cursor.fetchall()
                if not trades: return await query.message.reply_text("لا توجد صفقات مغلقة لتحليلها.")
                stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})
                for reason, status, pnl_val in trades:
                    if not reason or not status: continue
                    s = stats[reason]
                    if status.startswith('ناجحة'): s['wins'] += 1
                    else: s['losses'] += 1
                    if pnl_val: s['pnl'] += pnl_val
                report = ["**📜 تقرير أداء الاستراتيجيات**"]
                for r, s in sorted(stats.items(), key=lambda item: item[1]['pnl'], reverse=True):
                    total = s['wins'] + s['losses']; wr = (s['wins'] / total * 100) if total > 0 else 0
                    report.append(f"\n--- *{r}* ---\n  - الصفقات: {total} ({s['wins']}✅ / {s['losses']}❌)\n  - النجاح: {wr:.2f}%\n  - صافي الربح: ${s['pnl']:+.2f}")
                await query.message.reply_text("\n".join(report), parse_mode=ParseMode.MARKDOWN)
            elif report_type == "diagnostics":
                mood, scan, settings = bot_state.market_mood, bot_state.scan_stats, bot_state.settings; total_trades, active_trades_count = 0, 0
                async with aiosqlite.connect(DB_FILE) as conn:
                    total_trades = (await (await conn.execute("SELECT COUNT(*) FROM trades")).fetchone())[0]
                if bot_state.trade_manager: active_trades_count = len([t for t in bot_state.trade_manager.active_trades.values() if t['status'] in ['active', 'pending_protection']])
                ws_status = 'متصل ✅' if bot_state.ws_manager and bot_state.ws_manager.is_connected() else 'غير متصل ❌'
                scanners_text = escape_markdown(', '.join(settings.get('active_scanners',[])))
                report = [f"**🕵️‍♂️ تقرير التشخيص الشامل (v11.0)**\n",
                          f"--- **📊 حالة السوق الحالية** ---\n- **المزاج العام:** {mood['mood']} ({escape_markdown(mood['reason'])})\n- **مؤشر BTC:** {mood.get('btc_mood', 'N/A')}\n",
                          f"--- **🔬 أداء آخر فحص** ---\n- **وقت البدء:** {scan.get('last_start', 'N/A')}\n",
                          f"--- **🔧 الإعدادات النشطة** ---\n- **النمط الحالي:** {settings.get('active_preset', 'N/A')}\n- **الماسحات المفعلة:** {scanners_text}\n",
                          f"--- **🔩 حالة العمليات الداخلية** ---\n- **قاعدة البيانات:** متصلة ✅ ({total_trades} صفقة تاريخية)\n"
                          f"- **مدير الصفقات (الذاكرة):** {active_trades_count} صفقات نشطة\n"
                          f"- **ساعي البريد (WS):** {ws_status}"]
                await query.message.reply_text("\n".join(report), parse_mode=ParseMode.MARKDOWN)
        elif data.startswith("toggle_scanner_"):
            scanner_name = data.split("_", 2)[2]; active = bot_state.settings.get("active_scanners", []).copy()
            if scanner_name in active: active.remove(scanner_name)
            else: active.append(scanner_name)
            bot_state.settings["active_scanners"] = active; save_settings(); await show_scanners_menu(update, context)
        elif data.startswith("preset_"):
            preset_name = data.split("_", 1)[1]
            if preset_data := PRESETS.get(preset_name):
                for key, value in preset_data.items():
                    if isinstance(value, dict): bot_state.settings.get(key, {}).update(value)
                    else: bot_state.settings[key] = value
                bot_state.settings["active_preset"] = preset_name; save_settings()
                await query.edit_message_text(f"✅ تم تفعيل النمط: **{preset_data['name']}**", parse_mode=ParseMode.MARKDOWN)
        elif data.startswith("param_"):
            param_key = data.split("_", 1)[1]
            current_val = None
            if param_key in bot_state.settings: current_val = bot_state.settings[param_key]
            elif param_key in bot_state.settings.get('liquidity_filters', {}): current_val = bot_state.settings['liquidity_filters'][param_key]
            if isinstance(current_val, bool):
                 target_dict = bot_state.settings if param_key not in bot_state.settings.get('liquidity_filters', {}) else bot_state.settings['liquidity_filters']
                 target_dict[param_key] = not target_dict[param_key]; save_settings()
                 await show_parameters_menu(update, context, edit_message_id=query.message.message_id)
            else:
                 msg_to_delete = await query.message.reply_text(f"📝 *تعديل '{PARAM_DISPLAY_NAMES.get(param_key, param_key)}'*\n*القيمة الحالية:* `{current_val}`\n\nأرسل القيمة الجديدة.", parse_mode=ParseMode.MARKDOWN)
                 context.user_data['awaiting_input_for_param'] = (param_key, msg_to_delete.message_id, query.message.message_id)
        elif data == "back_to_settings":
            if query.message: await query.message.delete()
    except Exception as e: logger.error(f"Error in button handler: {e}", exc_info=True)

# =======================================================================================
# --- 🚀 Main Application Entry Point 🚀 ---
# =======================================================================================
async def main():
    logger.info("--- Bot process starting ---")
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.critical("FATAL: Environment variables not set. Exiting."); return
    
    load_settings()
    await init_database()
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    bot_state.application = app
    
    await ensure_libraries_loaded()
    
    bot_state.exchange = ccxt.okx({
        'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE, 
        'enableRateLimit': True, 'timeout': 60000,
        'options': {'defaultType': 'spot', 'hostname': 'aws.okx.com'}
    })

    bot_state.trade_manager = TradeManager(bot_state.exchange, app.bot)
    
    ws_manager = WebSocketManager(bot_state.trade_manager)
    bot_state.ws_manager = ws_manager
    ws_task = create_safe_task(ws_manager.run())
    logger.info("🚀 [Symphony] Conductor is ready. WebSocket is tuning.")

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    app.add_handler(CallbackQueryHandler(button_callback_handler))

    scan_interval = bot_state.settings.get("scan_interval_seconds", 900)
    track_interval = bot_state.settings.get("track_interval_seconds", 60)
    app.job_queue.run_repeating(perform_scan, interval=scan_interval, first=10, name="perform_scan")
    app.job_queue.run_repeating(track_open_trades, interval=track_interval, first=30, name="track_trades")
    
    try:
        await bot_state.exchange.fetch_balance()
        logger.info("✅ OKX connection test SUCCEEDED.")
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="*🚀 بوت The Phoenix v11.0 (The Final Blueprint) بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)
        
        async with app:
            await app.start()
            await app.updater.start_polling()
            logger.info("Bot is now running and polling for updates...")
            # We don't await the ws_task directly here, as it's a fire-and-forget safe task.
            # We need a way to keep the main coroutine alive.
            # A simple forever sleep is a common pattern.
            await asyncio.Event().wait()

    except Exception as e:
        logger.critical(f"An unhandled error occurred in main loop: {e}", exc_info=True)
    finally:
        # Proper shutdown sequence
        if bot_state.ws_manager and bot_state.ws_manager.websocket:
            await bot_state.ws_manager.websocket.close()
        if hasattr(app, 'updater') and app.updater and app.updater._running: await app.updater.stop()
        if app.running: await app.stop()
        if bot_state.exchange: await bot_state.exchange.close(); logger.info("CCXT exchange connection closed.")
        logger.info("Bot has been shut down.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot shutdown requested.")
    except Exception as e:
        logger.critical(f"Failed to start bot due to an error in initial setup: {e}", exc_info=True)
