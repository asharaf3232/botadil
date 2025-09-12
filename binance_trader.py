# -*- coding: utf-8 -*-

# --- المكتبات المطلوبة --- #
import ccxt.async_support as ccxt_async
import ccxt
import pandas as pd
import pandas_ta as ta
import asyncio
import os
import logging
import json
import time
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import deque

import feedparser
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("Library 'nltk' not found. Sentiment analysis will be disabled.")

import httpx
# --- تعديل مهم هنا --- #
from telegram import Update
from telegram.constants import ParseMode # تم نقل ParseMode إلى هنا في الإصدارات الجديدة
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import BadRequest, RetryAfter, TimedOut

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Library 'scipy' not found. RSI Divergence strategy will be disabled.")

from apscheduler.schedulers.asyncio import AsyncIOScheduler

# --- الإعدادات الأساسية --- #
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')
TELEGRAM_SIGNAL_CHANNEL_ID = os.getenv('TELEGRAM_SIGNAL_CHANNEL_ID', TELEGRAM_CHAT_ID)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_AV_KEY_HERE')

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_BINANCE_API_SECRET')

KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', 'YOUR_KUCOIN_API_KEY')
KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET', 'YOUR_KUCOIN_API_SECRET')
KUCOIN_API_PASSPHRASE = os.getenv('KUCOIN_API_PASSPHRASE', 'YOUR_KUCOIN_PASSPHRASE')

if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or TELEGRAM_CHAT_ID == 'YOUR_CHAT_ID_HERE':
    print("FATAL ERROR: Please set your Telegram Token and Chat ID in environment variables or directly in the script.")
    exit()
if ALPHA_VANTAGE_API_KEY == 'YOUR_AV_KEY_HERE':
    logging.warning("Alpha Vantage API key not set. Economic calendar will be disabled.")

# --- إعدادات البوت --- #
APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'trading_bot_v11.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'settings.json')
LOG_FILE = os.path.join(APP_ROOT, 'bot_v11.log')

SCAN_INTERVAL_SECONDS = 900
TRACK_INTERVAL_SECONDS = 120
EGYPT_TZ = ZoneInfo("Africa/Cairo")

# --- Logger --- #
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'a', 'utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- الإعدادات الافتراضية --- #
DEFAULT_SETTINGS = {
    "real_trading_enabled": False,
    "virtual_trade_size_percentage": 5.0,
    "max_concurrent_trades": 5,
    "top_n_symbols_by_volume": 250,
    "concurrent_workers": 10,
    "market_regime_filter_enabled": True,
    "fundamental_analysis_enabled": True,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "rsi_divergence", "supertrend_pullback"],
    "use_master_trend_filter": True,
    "master_trend_filter_ma_period": 50,
    "higher_timeframe": "1h",
    "timeframe": "15m",
    "master_adx_filter_level": 22,
    "fear_and_greed_filter_enabled": True,
    "fear_and_greed_threshold": 30,
    "use_dynamic_risk_management": True,
    "atr_period": 14,
    "atr_sl_multiplier": 2.0,
    "risk_reward_ratio": 1.5,
    "take_profit_percentage": 4.0,
    "stop_loss_percentage": 2.0,
    "trailing_sl_enabled": True,
    "trailing_sl_activate_percent": 2.0,
    "trailing_sl_percent": 1.5,
    "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "max_spread_percent": 0.5, "rvol_period": 20, "min_rvol": 1.5},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.8},
    "stablecoin_filter": {"exclude_bases": ["USDT","USDC","DAI","FDUSD","TUSD","USDE","PYUSD","GUSD","EURT","USDJ"]},
    "ema_trend_filter": {"enabled": True, "ema_period": 200},
    "min_tp_sl_filter": {"min_tp_percent": 1.0, "min_sl_percent": 0.5},
    "min_signal_strength": 1,
    "momentum_breakout": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bbands_period": 20, "bbands_stddev": 2.0, "rsi_period": 14, "rsi_max_level": 70},
    "breakout_squeeze_pro": {"bbands_period": 20, "bbands_stddev": 2.0, "keltner_period": 20, "keltner_atr_multiplier": 1.5, "volume_confirmation_enabled": True},
    "rsi_divergence": {"rsi_period": 14, "lookback_period": 40, "peak_trough_lookback": 5, "confirm_with_rsi_exit": True},
    "supertrend_pullback": {"atr_period": 10, "atr_multiplier": 3.0, "swing_high_lookback": 10},
    "last_market_mood": {"timestamp": "N/A", "mood": "UNKNOWN", "reason": "No scan performed yet."},
    "exchanges_to_scan": ["binance", "okx", "bybit", "kucoin", "gate", "mexc"]
}

# --- متغيرات الحالة العامة --- #
bot_data = {
    "exchanges": {},
    "last_signal_time": {},
    "settings": DEFAULT_SETTINGS.copy(),
    "status_snapshot": {
        "last_scan_start_time": "N/A",
        "last_scan_end_time": "N/A",
        "markets_found": 0,
        "signals_found": 0,
        "active_trades_count": 0,
        "scan_in_progress": False,
        "btc_market_mood": "غير محدد"
    },
    "scan_history": deque(maxlen=10)
}

scan_lock = asyncio.Lock()

# --- إدارة الإعدادات --- #
def load_settings():
    global bot_data
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                loaded_settings = json.load(f)
                bot_data["settings"].update(loaded_settings)
                logger.info("Settings loaded from settings.json")
        else:
            save_settings() # Create the file if it doesn't exist
    except Exception as e:
        logger.error(f"Could not load settings: {e}")

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(bot_data["settings"], f, indent=4)
        logger.info("Settings saved to settings.json")
    except Exception as e:
        logger.error(f"Could not save settings: {e}")

# --- قاعدة البيانات --- #
def init_database():
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, exchange TEXT, symbol TEXT,
                entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, entry_value_usdt REAL,
                status TEXT, exit_price REAL, closed_at TEXT, exit_value_usdt REAL, pnl_usdt REAL,
                trailing_sl_active BOOLEAN, highest_price REAL, reason TEXT, is_real_trade BOOLEAN DEFAULT FALSE,
                entry_order_id TEXT, exit_order_ids_json TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def db_query(query, params=(), fetchone=False, commit=False):
    try:
        with sqlite3.connect(DB_FILE, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if commit:
                conn.commit()
                return cursor.lastrowid
            return cursor.fetchone() if fetchone else cursor.fetchall()
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        return None if fetchone else []

# --- دوال التحليل الأساسي والأخبار --- #
async def get_alpha_vantage_economic_events():
    if ALPHA_VANTAGE_API_KEY == 'YOUR_AV_KEY_HERE': return []
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    params = {'function': 'ECONOMIC_CALENDAR', 'horizon': '3month', 'apikey': ALPHA_VANTAGE_API_KEY}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('https://www.alphavantage.co/query', params=params, timeout=20)
            response.raise_for_status()
        data_str = response.text
        if "premium" in data_str.lower():
            logger.error("Alpha Vantage API: Economic Calendar is a premium feature.")
            return []
        lines = data_str.strip().split('\r\n')
        if len(lines) < 2: return []
        header = [h.strip() for h in lines[0].split(',')]
        high_impact_events = [dict(zip(header, [v.strip() for v in line.split(',')])) for line in lines[1:]]
        today_events = [e.get('event', 'Unknown') for e in high_impact_events if e.get('releaseDate', '') == today_str and e.get('impact', '').lower() == 'high' and e.get('country', '') in ['USD', 'EUR']]
        if today_events:
            logger.warning(f"High-impact events today via Alpha Vantage: {today_events}")
        return today_events
    except httpx.RequestError as e:
        logger.error(f"Failed to fetch economic calendar from Alpha Vantage: {e}")
        return None

def get_latest_crypto_news(limit=15):
    urls, headlines = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"], []
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
    total_score = sum(sia.polarity_scores(h)['compound'] for h in headlines)
    return total_score / len(headlines) if headlines else 0.0

async def get_fundamental_market_mood():
    high_impact_events = await get_alpha_vantage_economic_events()
    if high_impact_events is None: return "DANGEROUS", -1.0, "فشل جلب البيانات الاقتصادية"
    if high_impact_events: return "DANGEROUS", -0.9, f"أحداث هامة اليوم: {', '.join(high_impact_events)}"
    sentiment_score = analyze_sentiment_of_headlines(get_latest_crypto_news())
    logger.info(f"Market sentiment score based on news: {sentiment_score:.2f}")
    if sentiment_score > 0.25: return "POSITIVE", sentiment_score, f"مشاعر إيجابية (الدرجة: {sentiment_score:.2f})"
    if sentiment_score < -0.25: return "NEGATIVE", sentiment_score, f"مشاعر سلبية (الدرجة: {sentiment_score:.2f})"
    return "NEUTRAL", sentiment_score, f"مشاعر محايدة (الدرجة: {sentiment_score:.2f})"

# --- استراتيجيات التحليل الفني --- #
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

def analyze_momentum_breakout(df, params, rvol, adx_value):
    df.ta.vwap(append=True)
    df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
    df.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'], append=True)
    df.ta.rsi(length=params['rsi_period'], append=True)
    macd_col, macds_col = find_col(df.columns, f"MACD_"), find_col(df.columns, f"MACDs_")
    bbu_col, rsi_col = find_col(df.columns, f"BBU_"), find_col(df.columns, f"RSI_")
    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and
        last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and
        last[rsi_col] < params['rsi_max_level']):
        return {"reason": "momentum_breakout", "type": "long"}
    return None

def analyze_breakout_squeeze_pro(df, params, rvol, adx_value):
    df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
    df.ta.kc(length=params['keltner_period'], scalar=params['keltner_atr_multiplier'], append=True)
    df.ta.obv(append=True)
    bbu_col, bbl_col = find_col(df.columns, f"BBU_"), find_col(df.columns, f"BBL_")
    kcu_col, kcl_col = find_col(df.columns, f"KCUe_"), find_col(df.columns, f"KCLEe_")
    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]:
        if last['close'] > last[bbu_col] and df['OBV'].iloc[-2] > df['OBV'].iloc[-3]:
            return {"reason": "breakout_squeeze_pro", "type": "long"}
    return None

def analyze_rsi_divergence(df, params, rvol, adx_value):
    if not SCIPY_AVAILABLE: return None
    df.ta.rsi(length=params['rsi_period'], append=True)
    rsi_col = find_col(df.columns, f"RSI_")
    if not rsi_col or df[rsi_col].isnull().all(): return None
    subset = df.iloc[-params['lookback_period']:].copy()
    price_troughs_idx, _ = find_peaks(-subset['low'], distance=params['peak_trough_lookback'])
    rsi_troughs_idx, _ = find_peaks(-subset[rsi_col], distance=params['peak_trough_lookback'])
    if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
        p_low1_idx, p_low2_idx = price_troughs_idx[-2], price_troughs_idx[-1]
        r_low1_idx, r_low2_idx = rsi_troughs_idx[-2], rsi_troughs_idx[-1]
        is_divergence = (subset.iloc[p_low2_idx]['low'] < subset.iloc[p_low1_idx]['low'] and subset.iloc[r_low2_idx][rsi_col] > subset.iloc[r_low1_idx][rsi_col])
        if is_divergence:
            rsi_exits_oversold = (subset.iloc[r_low1_idx][rsi_col] < 35 and df.iloc[-2][rsi_col] > 40)
            confirmation_price = subset.iloc[p_low2_idx:]['high'].max()
            price_confirmed = df.iloc[-2]['close'] > confirmation_price
            if (not params['confirm_with_rsi_exit'] or rsi_exits_oversold) and price_confirmed:
                return {"reason": "rsi_divergence", "type": "long"}
    return None

def analyze_supertrend_pullback(df, params, rvol, adx_value):
    df.ta.supertrend(length=params['atr_period'], multiplier=params['atr_multiplier'], append=True)
    st_dir_col, ema_col = find_col(df.columns, f"SUPERTd_"), find_col(df.columns, 'EMA_')
    if not st_dir_col or not ema_col or pd.isna(df[ema_col].iloc[-2]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if prev[st_dir_col] == -1 and last[st_dir_col] == 1:
        if last['close'] > last[ema_col] and adx_value >= bot_data['settings']['master_adx_filter_level']:
             recent_swing_high = df['high'].iloc[-params.get('swing_high_lookback', 10):-2].max()
             if last['close'] > recent_swing_high:
                return {"reason": "supertrend_pullback", "type": "long"}
    return None

SCANNERS = {"momentum_breakout": analyze_momentum_breakout, "breakout_squeeze_pro": analyze_breakout_squeeze_pro, "rsi_divergence": analyze_rsi_divergence, "supertrend_pullback": analyze_supertrend_pullback}

# --- تهيئة المنصات --- #
async def initialize_exchanges():
    exchanges_to_scan = bot_data['settings'].get("exchanges_to_scan", [])
    async def connect(ex_id):
        params = {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
        if ex_id == 'binance' and BINANCE_API_KEY != 'YOUR_BINANCE_API_KEY': params.update({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET})
        if ex_id == 'kucoin' and KUCOIN_API_KEY != 'YOUR_KUCOIN_API_KEY': params.update({'apiKey': KUCOIN_API_KEY, 'secret': KUCOIN_API_SECRET, 'password': KUCOIN_API_PASSPHRASE})
        exchange = getattr(ccxt_async, ex_id)(params)
        try:
            await exchange.load_markets()
            bot_data["exchanges"][ex_id] = exchange
            logger.info(f"Connected to {ex_id}.")
        except Exception as e:
            logger.error(f"Failed to connect to {ex_id}: {e}")
            await exchange.close()
    await asyncio.gather(*[connect(ex_id) for ex_id in exchanges_to_scan])

# --- الوظائف المساعدة --- #
async def aggregate_top_movers():
    all_tickers = []
    async def fetch(ex_id, ex):
        try: return [dict(t, exchange=ex_id) for t in (await ex.fetch_tickers()).values()]
        except Exception: return []
    results = await asyncio.gather(*[fetch(ex_id, ex) for ex_id, ex in bot_data["exchanges"].items()])
    for res in results: all_tickers.extend(res)
    settings = bot_data['settings']
    excluded_bases = settings['stablecoin_filter']['exclude_bases']
    min_volume = settings['liquidity_filters']['min_quote_volume_24h_usd']
    usdt_tickers = [t for t in all_tickers if t.get('symbol') and t['symbol'].upper().endswith('/USDT') and t['symbol'].split('/')[0] not in excluded_bases and t.get('quoteVolume', 0) >= min_volume and not any(k in t['symbol'].upper() for k in ['UP','DOWN','3L','3S','BEAR','BULL'])]
    sorted_tickers = sorted(usdt_tickers, key=lambda t: t.get('quoteVolume', 0), reverse=True)
    unique_symbols = {t['symbol']: {'exchange': t['exchange'], 'symbol': t['symbol']} for t in sorted_tickers}
    final_list = list(unique_symbols.values())[:settings['top_n_symbols_by_volume']]
    bot_data['status_snapshot']['markets_found'] = len(final_list)
    return final_list

async def get_higher_timeframe_trend(exchange, symbol, ma_period):
    try:
        ohlcv_htf = await exchange.fetch_ohlcv(symbol, bot_data['settings']['higher_timeframe'], limit=ma_period + 5)
        if len(ohlcv_htf) < ma_period: return None, "Not enough HTF data"
        df_htf = pd.DataFrame(ohlcv_htf, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_htf[f'SMA_{ma_period}'] = ta.sma(df_htf['close'], length=ma_period)
        last_candle = df_htf.iloc[-1]
        is_bullish = last_candle['close'] > last_candle[f'SMA_{ma_period}']
        return is_bullish, "Bullish" if is_bullish else "Bearish"
    except Exception as e:
        logger.error(f"Error fetching HTF trend for {symbol} on {exchange.id}: {e}")
        return None, f"Error: {e}"

# --- عامل المعالجة --- #
async def worker(queue, results_list, settings, failure_counter):
    while not queue.empty():
        market_info = await queue.get()
        symbol, ex_id = market_info.get('symbol', 'N/A'), market_info['exchange']
        exchange = bot_data["exchanges"].get(ex_id)
        if not exchange or not settings.get('active_scanners'):
            queue.task_done()
            continue
        try:
            liq_filters, vol_filters, ema_filters = settings['liquidity_filters'], settings['volatility_filters'], settings['ema_trend_filter']
            orderbook = await exchange.fetch_order_book(symbol, limit=1)
            if not orderbook or not orderbook['bids'] or not orderbook['asks']:
                queue.task_done(); continue
            best_bid, best_ask = orderbook['bids'][0][0], orderbook['asks'][0][0]
            if best_bid <= 0 or ((best_ask - best_bid) / best_bid) * 100 > liq_filters['max_spread_percent']:
                queue.task_done(); continue
            
            ohlcv = await exchange.fetch_ohlcv(symbol, settings['timeframe'], limit=ema_filters['ema_period'] + 20)
            if len(ohlcv) < ema_filters['ema_period']:
                queue.task_done(); continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
            
            df['volume_sma'] = ta.sma(df['volume'], length=liq_filters['rvol_period'])
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] <= 0:
                queue.task_done(); continue
            rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
            if rvol < liq_filters['min_rvol']:
                queue.task_done(); continue
            
            df.ta.atr(length=vol_filters['atr_period_for_filter'], append=True)
            atr_col = find_col(df.columns, f"ATRr_")
            last_close = df['close'].iloc[-2]
            if last_close <= 0 or (df[atr_col].iloc[-2] / last_close) * 100 < vol_filters['min_atr_percent']:
                queue.task_done(); continue
            
            ema_col = f"EMA_{ema_filters['ema_period']}"
            df.ta.ema(length=ema_filters['ema_period'], append=True)
            if ema_col not in df.columns or pd.isna(df[ema_col].iloc[-2]) or (ema_filters['enabled'] and last_close < df[ema_col].iloc[-2]):
                queue.task_done(); continue
            
            if settings.get('use_master_trend_filter'):
                is_htf_bullish, _ = await get_higher_timeframe_trend(exchange, symbol, settings['master_trend_filter_ma_period'])
                if is_htf_bullish is None or not is_htf_bullish:
                    queue.task_done(); continue
            
            df.ta.adx(append=True)
            adx_col, adx_value = find_col(df.columns, 'ADX_'), 0
            if adx_col and pd.notna(df[adx_col].iloc[-2]): adx_value = df[adx_col].iloc[-2]
            if settings.get('use_master_trend_filter') and adx_value < settings['master_adx_filter_level']:
                queue.task_done(); continue

            confirmed_reasons = [res['reason'] for scanner_name in settings['active_scanners'] if (res := SCANNERS[scanner_name](df.copy(), settings.get(scanner_name, {}), rvol, adx_value)) and res.get("type") == "long"]
            
            if confirmed_reasons and len(confirmed_reasons) >= settings.get("min_signal_strength", 1):
                entry_price = df.iloc[-2]['close']
                df.ta.atr(length=settings['atr_period'], append=True)
                current_atr = df.iloc[-2].get(find_col(df.columns, f"ATRr_"), 0)
                if settings.get("use_dynamic_risk_management", False) and current_atr > 0:
                    risk_per_unit = current_atr * settings['atr_sl_multiplier']
                    stop_loss, take_profit = entry_price - risk_per_unit, entry_price + (risk_per_unit * settings['risk_reward_ratio'])
                else:
                    stop_loss = entry_price * (1 - settings['stop_loss_percentage'] / 100)
                    take_profit = entry_price * (1 + settings['take_profit_percentage'] / 100)
                
                tp_percent = ((take_profit - entry_price) / entry_price) * 100
                sl_percent = ((entry_price - stop_loss) / entry_price) * 100
                min_filters = settings['min_tp_sl_filter']
                if tp_percent >= min_filters['min_tp_percent'] and sl_percent >= min_filters['min_sl_percent']:
                    results_list.append({"symbol": symbol, "exchange": ex_id.capitalize(), "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "timestamp": df.index[-2], "reason": ' + '.join(confirmed_reasons), "strength": len(confirmed_reasons)})
            queue.task_done()
        except ccxt.RateLimitExceeded: await asyncio.sleep(10)
        except ccxt.NetworkError as e: logger.warning(f"Network error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"CRITICAL ERROR in worker for {symbol}: {e}", exc_info=False)
            failure_counter[0] += 1
        finally:
            if not queue.empty() and queue.qsize() % 50 == 0:
                logger.info(f"Worker queue size: {queue.qsize()}")
            if 'task_done' not in str(queue.task_done): queue.task_done()

# --- وظائف التداول الحقيقي --- #
async def get_real_balance(exchange_id, currency='USDT'):
    exchange = bot_data["exchanges"].get(exchange_id.lower())
    if not exchange or not exchange.apiKey:
        logger.warning(f"Cannot fetch balance: {exchange_id.capitalize()} client not authenticated.")
        return 0.0
    try:
        balance = await exchange.fetch_balance()
        return balance['free'].get(currency, 0.0)
    except Exception as e:
        logger.error(f"Error fetching {exchange_id.capitalize()} balance for {currency}: {e}")
        return 0.0

async def place_real_trade(signal, context: ContextTypes.DEFAULT_TYPE):
    ex_id = signal['exchange'].lower()
    exchange = bot_data["exchanges"].get(ex_id)
    if not exchange or not exchange.apiKey:
        logger.error(f"Cannot place real trade for {signal['symbol']}: {ex_id.capitalize()} client not authenticated.")
        return None
    try:
        usdt_balance = await get_real_balance(ex_id, 'USDT')
        trade_size_percent = bot_data['settings']['virtual_trade_size_percentage']
        trade_amount_usdt = usdt_balance * (trade_size_percent / 100)
        if trade_amount_usdt < 10:
            logger.warning(f"Skipping real trade for {signal['symbol']}. Trade amount ${trade_amount_usdt:.2f} is too low.")
            return None
        
        market_info = exchange.markets.get(signal['symbol'])
        if not market_info:
            logger.error(f"Market {signal['symbol']} not found on {ex_id.capitalize()}")
            return None
        
        quantity = exchange.amount_to_precision(signal['symbol'], trade_amount_usdt / signal['entry_price'])
        logger.info(f"Placing MARKET BUY for {quantity} {signal['symbol']} on {ex_id.capitalize()}")
        buy_order = await exchange.create_market_buy_order(signal['symbol'], float(quantity))
        
        # ملاحظة: أوامر TP/SL معقدة وتختلف بين المنصات. هذا مثال مبسط.
        # قد تحتاج إلى منطق أكثر تعقيدًا للتحقق من تنفيذ الأوامر وإلغائها عند إغلاق الصفقة يدويًا.
        await asyncio.sleep(2) # انتظر قليلاً للتأكد من تنفيذ أمر الشراء
        
        tp_price = exchange.price_to_precision(signal['symbol'], signal['take_profit'])
        sl_price = exchange.price_to_precision(signal['symbol'], signal['stop_loss'])
        
        logger.info(f"Placing TP ({tp_price}) and SL ({sl_price}) orders for {signal['symbol']}")
        # مثال لأمر OCO على Binance (One-Cancels-the-Other)
        if exchange.has.get('createOco'):
            oco_order = await exchange.create_order(signal['symbol'], 'oco', 'sell', float(quantity), price=tp_price, stopPrice=sl_price, params={'stopLimitPrice': sl_price})
            exit_order_ids = {"oco_id": oco_order['id']}
        else: # منصات أخرى قد تتطلب أوامر منفصلة
            tp_order = await exchange.create_limit_sell_order(signal['symbol'], float(quantity), float(tp_price))
            sl_order = await exchange.create_stop_limit_sell_order(signal['symbol'], float(quantity), float(sl_price), float(sl_price))
            exit_order_ids = {"tp_id": tp_order['id'], "sl_id": sl_order['id']}

        await send_telegram_message(context.bot, {'custom_message': f"🚨 صفقة حقيقية نفذت على {ex_id.capitalize()} 🚨\n- العملة: {signal['symbol']}\n- الكمية: {quantity}\n- أمر الشراء ID: {buy_order['id']}"})
        return {"entry_order_id": buy_order['id'], "exit_order_ids_json": json.dumps(exit_order_ids), "quantity": float(quantity), "entry_value_usdt": trade_amount_usdt}

    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds on {ex_id.capitalize()}: {e}")
        await send_telegram_message(context.bot, {'custom_message': f"❌ فشل التنفيذ: رصيد غير كافٍ على {ex_id.capitalize()}"})
    except Exception as e:
        logger.error(f"Critical error placing real trade on {ex_id.capitalize()}: {e}", exc_info=True)
    return None

# --- إرسال رسائل تليجرام --- #
async def send_telegram_message(bot, signal_data, is_new=False, is_opportunity=False, update_type=None):
    message, target_chat = "", TELEGRAM_CHAT_ID
    def format_price(p): return f"{p:,.8f}" if p < 0.01 else f"{p:,.4f}"

    if 'custom_message' in signal_data:
        message, target_chat = signal_data['custom_message'], signal_data.get('target_chat', TELEGRAM_CHAT_ID)
    elif is_new or is_opportunity:
        target_chat = TELEGRAM_SIGNAL_CHANNEL_ID
        strength = '⭐' * signal_data.get('strength', 1)
        title = f"✅ توصية شراء | {signal_data['symbol']}" if is_new else f"💡 فرصة محتملة | {signal_data['symbol']}"
        entry, tp, sl = signal_data['entry_price'], signal_data['take_profit'], signal_data['stop_loss']
        tp_p, sl_p = ((tp - entry) / entry * 100), ((entry - sl) / entry * 100)
        id_line = f"\n*للمتابعة: /check {signal_data['trade_id']}*" if is_new else ""
        reasons_ar = ' + '.join([{"momentum_breakout": "زخم", "breakout_squeeze_pro": "انضغاط", "rsi_divergence": "دايفرجنس", "supertrend_pullback": "سوبرترند"}.get(r, r) for r in signal_data['reason'].split(' + ')])
        message = (f"{title}\n------------------------------------\n"
                   f"🔹 المنصة: {signal_data['exchange']}\n⭐ القوة: {strength}\n🔍 الاستراتيجية: {reasons_ar}\n\n"
                   f"📈 دخول: `{format_price(entry)}`\n"
                   f"🎯 هدف: `{format_price(tp)}` (+{tp_p:.2f}%)\n"
                   f"🛑 وقف: `{format_price(sl)}` (-{sl_p:.2f}%)"
                   f"{id_line}")
    elif update_type:
        pnl_str = f"+${signal_data['pnl']:.2f}" if signal_data['pnl'] > 0 else f"-${abs(signal_data['pnl']):.2f}"
        if update_type == 'tsl_activation': message = f"🚀 تأمين أرباح #{signal_data['id']} {signal_data['symbol']}!\nتم رفع الوقف إلى نقطة الدخول. الصفقة الآن بدون مخاطرة."
        elif update_type == 'tp_hit': message = f"✅💰 هدف محقق #{signal_data['id']} {signal_data['symbol']}!\nتم إغلاق الصفقة على ربح {pnl_str}."
        elif update_type == 'sl_hit': message = f"❌🛑 وقف خسارة #{signal_data['id']} {signal_data['symbol']}.\nتم إغلاق الصفقة على خسارة {pnl_str}."

    if not message: return
    try:
        await bot.send_message(chat_id=target_chat, text=message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e: logger.error(f"Failed to send Telegram message to {target_chat}: {e}")

# --- المهام المجدولة --- #
async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        if bot_data['status_snapshot']['scan_in_progress']: return
        settings = bot_data["settings"]
        
        if settings.get('fundamental_analysis_enabled', True):
            mood, _, mood_reason = await get_fundamental_market_mood()
            bot_data['settings']['last_market_mood'] = {"timestamp": datetime.now(EGYPT_TZ).strftime('%H:%M'), "mood": mood, "reason": mood_reason}
            if mood in ["NEGATIVE", "DANGEROUS"]:
                logger.warning(f"Scan paused due to fundamental mood: {mood_reason}")
                return

        status = bot_data['status_snapshot']
        status.update({"scan_in_progress": True, "last_scan_start_time": datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), "signals_found": 0})
        
        active_trades_count = db_query("SELECT COUNT(*) FROM trades WHERE status = 'نشطة'", fetchone=True)[0] or 0
        top_markets = await aggregate_top_movers()
        if not top_markets:
            status['scan_in_progress'] = False; return

        queue = asyncio.Queue()
        for market in top_markets: await queue.put(market)

        signals, failure_counter = [], [0]
        workers = [asyncio.create_task(worker(queue, signals, settings, failure_counter)) for _ in range(settings['concurrent_workers'])]
        await queue.join()
        for w in workers: w.cancel()

        signals.sort(key=lambda s: s.get('strength', 0), reverse=True)
        new_trades, opportunities = 0, 0
        last_signal = bot_data['last_signal_time']

        for signal in signals:
            if time.time() - last_signal.get(signal['symbol'], 0) <= (SCAN_INTERVAL_SECONDS * 2):
                continue

            trade_amount = settings["virtual_trade_size_percentage"] / 100
            signal.update({'entry_value_usdt': 1000 * trade_amount, 'quantity': (1000 * trade_amount) / signal['entry_price']}) # Based on virtual balance
            
            is_real = settings.get('real_trading_enabled', False) and signal['exchange'].lower() in ['binance', 'kucoin']
            if is_real:
                order_result = await place_real_trade(signal, context)
                if order_result: signal.update(order_result, is_real_trade=True)
                else: signal['is_real_trade'] = False
            else: signal['is_real_trade'] = False

            if active_trades_count < settings.get("max_concurrent_trades", 5):
                trade_id = db_query('''INSERT INTO trades (timestamp, exchange, symbol, entry_price, take_profit, stop_loss, quantity, entry_value_usdt, status, trailing_sl_active, highest_price, reason, is_real_trade, entry_order_id, exit_order_ids_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), signal['exchange'], signal['symbol'], signal['entry_price'], signal['take_profit'], signal['stop_loss'], signal['quantity'], signal['entry_value_usdt'], 'نشطة', False, signal['entry_price'], signal['reason'], signal.get('is_real_trade', False), signal.get('entry_order_id'), signal.get('exit_order_ids_json')), commit=True)
                if trade_id:
                    signal['trade_id'] = trade_id
                    await send_telegram_message(context.bot, signal, is_new=True)
                    active_trades_count += 1; new_trades += 1
            else:
                await send_telegram_message(context.bot, signal, is_opportunity=True)
                opportunities += 1
            
            last_signal[signal['symbol']] = time.time()
            await asyncio.sleep(0.5)

        logger.info(f"Scan complete. Found: {len(signals)}, New: {new_trades}, Opps: {opportunities}, Fails: {failure_counter[0]}.")
        status.update({'signals_found': new_trades + opportunities, 'last_scan_end_time': datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), 'scan_in_progress': False})

async def track_open_trades(context: ContextTypes.DEFAULT_TYPE):
    active_trades = db_query("SELECT id, exchange, symbol, entry_price, take_profit, stop_loss, trailing_sl_active, highest_price FROM trades WHERE status = 'نشطة'")
    bot_data['status_snapshot']['active_trades_count'] = len(active_trades)
    if not active_trades: return
    
    settings = bot_data["settings"]
    for trade in active_trades:
        trade_id, ex_id_cap, symbol, entry, tp, sl, tsl_active, highest = trade
        ex_id = ex_id_cap.lower()
        exchange = bot_data['exchanges'].get(ex_id)
        if not exchange: continue
        
        try:
            ticker = await exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            pnl = (current_price - entry) / entry * 100

            if current_price >= tp:
                db_query("UPDATE trades SET status = 'مغلقة (ربح)', exit_price = ?, closed_at = ?, pnl_usdt = ? WHERE id = ?", (current_price, datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), (current_price - entry) * (1000 / entry), trade_id), commit=True)
                await send_telegram_message(context.bot, {"id": trade_id, "symbol": symbol, "pnl": (current_price - entry) * (1000 / entry)}, update_type='tp_hit')
                continue

            if current_price <= sl:
                db_query("UPDATE trades SET status = 'مغلقة (خسارة)', exit_price = ?, closed_at = ?, pnl_usdt = ? WHERE id = ?", (current_price, datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), (current_price - entry) * (1000 / entry), trade_id), commit=True)
                await send_telegram_message(context.bot, {"id": trade_id, "symbol": symbol, "pnl": (current_price - entry) * (1000 / entry)}, update_type='sl_hit')
                continue

            if settings.get('trailing_sl_enabled'):
                if not tsl_active and pnl >= settings['trailing_sl_activate_percent']:
                    new_sl = entry
                    db_query("UPDATE trades SET trailing_sl_active = 1, stop_loss = ?, highest_price = ? WHERE id = ?", (new_sl, current_price, trade_id), commit=True)
                    await send_telegram_message(context.bot, {"id": trade_id, "symbol": symbol}, update_type='tsl_activation')
                elif tsl_active:
                    new_highest = max(highest, current_price)
                    new_sl = new_highest * (1 - settings['trailing_sl_percent'] / 100)
                    if new_sl > sl:
                        db_query("UPDATE trades SET stop_loss = ?, highest_price = ? WHERE id = ?", (new_sl, new_highest, trade_id), commit=True)
                        logger.info(f"TSL for trade #{trade_id} ({symbol}) updated to {new_sl:.4f}")

        except Exception as e:
            logger.error(f"Error tracking trade #{trade_id} ({symbol}): {e}")

# --- أوامر تليجرام --- #
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("أهلاً بك! بوت التداول قيد التشغيل.\nاستخدم /help لعرض الأوامر المتاحة.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 *أوامر البوت المتاحة*\n\n"
        "`/status` - عرض الحالة العامة للبوت والصفقات النشطة.\n"
        "`/check <ID>` - عرض تفاصيل صفقة معينة.\n"
        "`/settings` - عرض الإعدادات الحالية.\n"
        "`/help` - عرض هذه الرسالة."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status = bot_data['status_snapshot']
    settings = bot_data['settings']
    mood = settings['last_market_mood']
    active_trades = db_query("SELECT id, symbol, entry_price FROM trades WHERE status = 'نشطة'")
    
    trades_str = "\n".join([f"- `{t[0]}`: {t[1]} @ {t[2]:.4f}" for t in active_trades]) if active_trades else "لا يوجد صفقات نشطة."
    
    status_msg = (
        f"📊 *حالة البوت*\n\n"
        f"وضع التداول الحقيقي: {' مفعل ✅' if settings['real_trading_enabled'] else ' معطل ❌'}\n"
        f"آخر فحص بدأ: {status['last_scan_start_time']}\n"
        f"آخر فحص انتهى: {status['last_scan_end_time']}\n"
        f"مزاج السوق (أخبار): {mood['mood']} ({mood['reason']})\n\n"
        f"📂 *الصفقات النشطة ({len(active_trades)})*\n{trades_str}"
    )
    await update.message.reply_text(status_msg, parse_mode=ParseMode.MARKDOWN)

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings_str = json.dumps(bot_data['settings'], indent=2, ensure_ascii=False)
    # Telegram has a message character limit of 4096. Truncate if necessary.
    if len(settings_str) > 4000:
        settings_str = settings_str[:4000] + "\n... (truncated)"
    await update.message.reply_text(f"<pre>{settings_str}</pre>", parse_mode=ParseMode.HTML)
    
async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        trade_id = int(context.args[0])
        trade = db_query("SELECT * FROM trades WHERE id = ?", (trade_id,), fetchone=True)
        if not trade:
            await update.message.reply_text("لم يتم العثور على صفقة بهذا الرقم.")
            return
        
        # Unpack trade data (adjust indices based on your table structure)
        (id, ts, ex, sym, entry, tp, sl, qty, val, stat, exit_p, closed, exit_val, pnl, tsl_act, high, reason, is_real, _, _) = trade
        
        real_str = "حقيقية 🚨" if is_real else "افتراضية 🧪"
        details_msg = (
            f"🔍 *تفاصيل الصفقة #{id} ({sym})*\n\n"
            f"الحالة: *{stat}* ({real_str})\n"
            f"المنصة: {ex}\n"
            f"السبب: {reason}\n\n"
            f"تاريخ الدخول: {ts}\n"
            f"سعر الدخول: `{entry:.4f}`\n"
            f"الهدف: `{tp:.4f}`\n"
            f"الوقف: `{sl:.4f}`\n"
            f"أعلى سعر مسجل: `{high:.4f}`\n"
            f"الوقف المتحرك مفعل: {'نعم' if tsl_act else 'لا'}\n"
        )
        if stat != 'نشطة':
            details_msg += f"\nتاريخ الإغلاق: {closed}\nسعر الخروج: `{exit_p:.4f}`\nالربح/الخسارة: `${pnl:.2f}`"
        
        await update.message.reply_text(details_msg, parse_mode=ParseMode.MARKDOWN)

    except (IndexError, ValueError):
        await update.message.reply_text("الرجاء إدخال رقم الصفقة بعد الأمر. مثال: `/check 123`")

# --- دالة التشغيل الرئيسية --- #
async def main():
    logger.info("Starting bot...")
    
    # تحميل الإعدادات وتهيئة قاعدة البيانات
    load_settings()
    init_database()
    
    # تهيئة منصات التداول
    await initialize_exchanges()
    if not bot_data["exchanges"]:
        logger.fatal("No exchanges could be initialized. Exiting.")
        return

    # تهيئة تطبيق تليجرام
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # إضافة معالجات الأوامر
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("check", check_command))
    
    # تهيئة وجدولة المهام
    scheduler = AsyncIOScheduler(timezone=EGYPT_TZ)
    scheduler.add_job(perform_scan, 'interval', seconds=SCAN_INTERVAL_SECONDS, args=[application])
    scheduler.add_job(track_open_trades, 'interval', seconds=TRACK_INTERVAL_SECONDS, args=[application])
    scheduler.start()
    
    logger.info("Bot is running and scheduler is active.")
    
    # تشغيل البوت
    await application.initialize()
    await application.updater.start_polling()
    await application.start()
    
    # إبقاء السكربت يعمل
    while True:
        await asyncio.sleep(3600)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)


