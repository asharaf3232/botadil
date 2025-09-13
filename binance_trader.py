# -*- coding: utf-8 -*-
# =======================================================================================
# --- ًں’£ ط¨ظˆطھ ظƒط§ط³ط­ط© ط§ظ„ط£ظ„ط؛ط§ظ… (Minesweeper Bot) v5.5 (Truly Final & Complete) ًں’£ ---
# =======================================================================================
# --- ط³ط¬ظ„ ط§ظ„طھط؛ظٹظٹط±ط§طھ v5.5 ---
#
# 1. [ط¥طµظ„ط§ط­ ط­ط§ط³ظ…] طھظ… ظ…ظ„ط، ظƒظ„ ط§ظ„ط¯ظˆط§ظ„ ط§ظ„ظپط§ط±ط؛ط© (pass) ط¨ط§ظ„ظ…ظ†ط·ظ‚ ط§ظ„ظƒط§ظ…ظ„ ظ…ظ† ط§ظ„ظ†ط³ط®ط© 4.5.
# 2. [ط¥طµظ„ط§ط­ ظ‡ظٹظƒظ„ظٹ] طھظ… ط§ظ„طھط£ظƒط¯ ظ…ظ† ط£ظ† ظƒظ„ ط§ظ„ط¯ظˆط§ظ„ ظ…ط¹ط±ظپط© ظ‚ط¨ظ„ ط§ط³طھط¯ط¹ط§ط¦ظ‡ط§.
# 3. [ط¯ظ…ط¬ ظƒط§ظ…ظ„] طھظ… ط¯ظ…ط¬ ظƒظ„ ط§ظ„ط¯ظˆط§ظ„ ظˆط§ظ„ظ…ظ†ط·ظ‚ ظ…ظ† ط§ظ„ظ†ط³ط®ط© 4.5 ظپظٹ ط§ظ„ظ‡ظٹظƒظ„ط© ط§ظ„ط¬ط¯ظٹط¯ط© v5.
# 4. [ط¥ط¹ط§ط¯ط© ظ‡ظٹظƒظ„ط©] طھط·ط¨ظٹظ‚ ظ†ظ…ط· ط§ظ„ظ…ط­ظˆظ„ (Adapter Pattern) ظ„ظ„طھط¹ط§ظ…ظ„ ظ…ط¹ ط§ظ„ظ…ظ†طµط§طھ.
# 5. [ط¥ط¹ط§ط¯ط© ظ‡ظٹظƒظ„ط©] طھط·ط¨ظٹظ‚ ظƒظ„ط§ط³ ط¥ط¯ط§ط±ط© ط§ظ„ط­ط§ظ„ط© (State Management).
# 6. [طھط­ط³ظٹظ† ط£ط¯ط§ط،] طھط·ط¨ظٹظ‚ ط§ظ„طھط²ط§ظ…ظ† ظپظٹ ظ…طھط§ط¨ط¹ط© ط§ظ„طµظپظ‚ط§طھ.
# 7. [ط¥طµظ„ط§ط­ ظ†ظ‡ط§ط¦ظٹ] طھط·ط¨ظٹظ‚ ط§ظ„ط­ظ„ ط§ظ„طµط­ظٹط­ ظ„ظ…ظ†طµط© KuCoin (ط£ظ…ط±ط§ظ† ظ…ظ†ظپطµظ„ط§ظ†).
#
# =======================================================================================

# --- ط§ظ„ظ…ظƒطھط¨ط§طھ ط§ظ„ظ…ط·ظ„ظˆط¨ط© ---
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

# --- ط§ظ„ط¥ط¹ط¯ط§ط¯ط§طھ ط§ظ„ط£ط³ط§ط³ظٹط© ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')
TELEGRAM_SIGNAL_CHANNEL_ID = os.getenv('TELEGRAM_SIGNAL_CHANNEL_ID', TELEGRAM_CHAT_ID)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_AV_KEY_HERE')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_BINANCE_API_SECRET')
KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', 'YOUR_KUCOIN_API_KEY')
KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET', 'YOUR_KUCOIN_API_SECRET')
KUCOIN_API_PASSPHRASE = os.getenv('KUCOIN_API_PASSPHRASE', 'YOUR_KUCOIN_API_PASSPHRASE')

# --- ط¥ط¹ط¯ط§ط¯ط§طھ ط§ظ„ط¨ظˆطھ ---
EXCHANGES_TO_SCAN = ['binance', 'okx', 'bybit', 'kucoin', 'gate', 'mexc']
TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
SCAN_INTERVAL_SECONDS = 900
TRACK_INTERVAL_SECONDS = 45

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v5.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'minesweeper_settings_v5.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")

# --- ط¥ط¹ط¯ط§ط¯ ظ…ط³ط¬ظ„ ط§ظ„ط£ط­ط¯ط§ط« (Logger) ---
LOG_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v5.log')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'a'), logging.StreamHandler()])
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger("MinesweeperBot_v5")


# =======================================================================================
# --- ًںڑ€ [v5.0] ط¥ط¹ط§ط¯ط© ط§ظ„ظ‡ظٹظƒظ„ط©: ط¥ط¯ط§ط±ط© ط§ظ„ط­ط§ظ„ط© ظˆط§ظ„ظ…ظ†طµط§طھ ًںڑ€ ---
# =======================================================================================

class BotState:
    """ظƒظ„ط§ط³ ظ…ط±ظƒط²ظٹ ظ„ط¥ط¯ط§ط±ط© ظƒظ„ ط­ط§ظ„ط© ط§ظ„ط¨ظˆطھ ظ„ط²ظٹط§ط¯ط© ط§ظ„طھظ†ط¸ظٹظ…."""
    def __init__(self):
        self.exchanges = {}
        self.public_exchanges = {}
        self.last_signal_time = {}
        self.settings = {}
        self.status_snapshot = {
            "last_scan_start_time": None, "last_scan_end_time": None,
            "markets_found": 0, "signals_found": 0, "active_trades_count": 0,
            "scan_in_progress": False, "btc_market_mood": "ط؛ظٹط± ظ…ط­ط¯ط¯"
        }
        self.scan_history = deque(maxlen=10)

bot_state = BotState()
scan_lock = asyncio.Lock()
report_lock = asyncio.Lock()

class ExchangeAdapter:
    """ظƒظ„ط§ط³ ط£ط³ط§ط³ظٹ ظ…ط¬ط±ط¯ ظ„ظ†ظ…ط· ط§ظ„ظ…ط­ظˆظ„."""
    def __init__(self, exchange_client):
        self.exchange = exchange_client

    async def place_exit_orders(self, signal, verified_quantity):
        raise NotImplementedError

    async def update_trailing_stop_loss(self, trade, new_sl):
        raise NotImplementedError

class BinanceAdapter(ExchangeAdapter):
    """ظ…ط­ظˆظ„ ط®ط§طµ ط¨ظ…ظ†طµط© BinanceطŒ ظٹط³طھط®ط¯ظ… ط£ظˆط§ظ…ط± OCO."""
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
    """ظ…ط­ظˆظ„ ط®ط§طµ ط¨ظ…ظ†طµط© KuCoinطŒ ظٹط³طھط®ط¯ظ… ط£ظ…ط±ظٹظ† ظ…ظ†ظپطµظ„ظٹظ†."""
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
    exchange_client = bot_state.exchanges.get(exchange_id.lower())
    if not exchange_client:
        return None
        
    adapter_map = { 'binance': BinanceAdapter, 'kucoin': KuCoinAdapter }
    AdapterClass = adapter_map.get(exchange_id.lower())
    if AdapterClass:
        return AdapterClass(exchange_client)
    
    logger.warning(f"No specific adapter found for {exchange_id}.")
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
    "momentum_breakout": "ط²ط®ظ… ط§ط®طھط±ط§ظ‚ظٹ", "breakout_squeeze_pro": "ط§ط®طھط±ط§ظ‚ ط§ظ†ط¶ط؛ط§ط·ظٹ",
    "support_rebound": "ط§ط±طھط¯ط§ط¯ ط§ظ„ط¯ط¹ظ…", "whale_radar": "ط±ط§ط¯ط§ط± ط§ظ„ط­ظٹطھط§ظ†", "sniper_pro": "ط§ظ„ظ‚ظ†ط§طµ ط§ظ„ظ…ط­طھط±ظپ",
}

EDITABLE_PARAMS = {
    "ط¥ط¹ط¯ط§ط¯ط§طھ ط¹ط§ظ…ط©": [
        "max_concurrent_trades", "top_n_symbols_by_volume", "concurrent_workers",
        "min_signal_strength"
    ],
    "ط¥ط¹ط¯ط§ط¯ط§طھ ط§ظ„ظ…ط®ط§ط·ط±": [
        "automate_real_tsl", "real_trade_size_usdt", "virtual_trade_size_percentage",
        "atr_sl_multiplier", "risk_reward_ratio", "trailing_sl_activation_percent", "trailing_sl_callback_percent"
    ],
    "ط§ظ„ظپظ„ط§طھط± ظˆط§ظ„ط§طھط¬ط§ظ‡": [
        "market_regime_filter_enabled", "use_master_trend_filter", "fear_and_greed_filter_enabled",
        "master_adx_filter_level", "master_trend_filter_ma_period", "trailing_sl_enabled", "fear_and_greed_threshold",
        "fundamental_analysis_enabled"
    ]
}
PARAM_DISPLAY_NAMES = {
    "automate_real_tsl": "ًں¤– ط£طھظ…طھط© ط§ظ„ظˆظ‚ظپ ط§ظ„ظ…طھط­ط±ظƒ ط§ظ„ط­ظ‚ظٹظ‚ظٹ",
    "real_trade_size_usdt": "ًں’µ ط­ط¬ظ… ط§ظ„طµظپظ‚ط© ط§ظ„ط­ظ‚ظٹظ‚ظٹط© ($)",
    "virtual_trade_size_percentage": "ًں“ٹ ط­ط¬ظ… ط§ظ„طµظپظ‚ط© ط§ظ„ظˆظ‡ظ…ظٹط© (%)",
    "max_concurrent_trades": "ط£ظ‚طµظ‰ ط¹ط¯ط¯ ظ„ظ„طµظپظ‚ط§طھ",
    "top_n_symbols_by_volume": "ط¹ط¯ط¯ ط§ظ„ط¹ظ…ظ„ط§طھ ظ„ظ„ظپط­طµ",
    "concurrent_workers": "ط¹ظ…ط§ظ„ ط§ظ„ظپط­طµ ط§ظ„ظ…طھط²ط§ظ…ظ†ظٹظ†",
    "min_signal_strength": "ط£ط¯ظ†ظ‰ ظ‚ظˆط© ظ„ظ„ط¥ط´ط§ط±ط©",
    "atr_sl_multiplier": "ظ…ط¶ط§ط¹ظپ ظˆظ‚ظپ ط§ظ„ط®ط³ط§ط±ط© (ATR)",
    "risk_reward_ratio": "ظ†ط³ط¨ط© ط§ظ„ظ…ط®ط§ط·ط±ط©/ط§ظ„ط¹ط§ط¦ط¯",
    "trailing_sl_activation_percent": "طھظپط¹ظٹظ„ ط§ظ„ظˆظ‚ظپ ط§ظ„ظ…طھط­ط±ظƒ (%)",
    "trailing_sl_callback_percent": "ظ…ط³ط§ظپط© ط§ظ„ظˆظ‚ظپ ط§ظ„ظ…طھط­ط±ظƒ (%)",
    "market_regime_filter_enabled": "ظپظ„طھط± ظˆط¶ط¹ ط§ظ„ط³ظˆظ‚ (ظپظ†ظٹ)",
    "use_master_trend_filter": "ظپظ„طھط± ط§ظ„ط§طھط¬ط§ظ‡ ط§ظ„ط¹ط§ظ… (BTC)",
    "master_adx_filter_level": "ظ…ط³طھظˆظ‰ ظپظ„طھط± ADX",
    "master_trend_filter_ma_period": "ظپطھط±ط© ظپظ„طھط± ط§ظ„ط§طھط¬ط§ظ‡",
    "trailing_sl_enabled": "طھظپط¹ظٹظ„ ط§ظ„ظˆظ‚ظپ ط§ظ„ظ…طھط­ط±ظƒ",
    "fear_and_greed_filter_enabled": "ظپظ„طھط± ط§ظ„ط®ظˆظپ ظˆط§ظ„ط·ظ…ط¹",
    "fear_and_greed_threshold": "ط­ط¯ ظ…ط¤ط´ط± ط§ظ„ط®ظˆظپ",
    "fundamental_analysis_enabled": "ظپظ„طھط± ط§ظ„ط£ط®ط¨ط§ط± ظˆط§ظ„ط¨ظٹط§ظ†ط§طھ",
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
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        sql = '''INSERT INTO trades (timestamp, exchange, symbol, entry_price, take_profit, stop_loss, quantity, entry_value_usdt, status, trailing_sl_active, highest_price, reason, trade_mode, entry_order_id, exit_order_ids_json)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        params = (
            datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            signal['exchange'],
            signal['symbol'],
            signal.get('verified_entry_price', signal['entry_price']), 
            signal['take_profit'],
            signal['stop_loss'],
            signal.get('verified_quantity', signal['quantity']), 
            signal.get('verified_entry_value', signal['entry_value_usdt']), 
            'ظ†ط´ط·ط©',
            False,
            signal.get('verified_entry_price', signal['entry_price']),
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
        logger.error(f"Failed to log recommendation to DB: {e}")
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
    if high_impact_events is None: return "DANGEROUS", -1.0, "ظپط´ظ„ ط¬ظ„ط¨ ط§ظ„ط¨ظٹط§ظ†ط§طھ ط§ظ„ط§ظ‚طھطµط§ط¯ظٹط©"
    if high_impact_events: return "DANGEROUS", -0.9, f"ط£ط­ط¯ط§ط« ظ‡ط§ظ…ط© ط§ظ„ظٹظˆظ…: {', '.join(high_impact_events)}"
    sentiment_score = analyze_sentiment_of_headlines(get_latest_crypto_news())
    logger.info(f"Market sentiment score: {sentiment_score:.2f}")
    if sentiment_score > 0.25: return "POSITIVE", sentiment_score, f"ظ…ط´ط§ط¹ط± ط¥ظٹط¬ط§ط¨ظٹط© (ط§ظ„ط¯ط±ط¬ط©: {sentiment_score:.2f})"
    elif sentiment_score < -0.25: return "NEGATIVE", sentiment_score, f"ظ…ط´ط§ط¹ط± ط³ظ„ط¨ظٹط© (ط§ظ„ط¯ط±ط¬ط©: {sentiment_score:.2f})"
    else: return "NEUTRAL", sentiment_score, f"ظ…ط´ط§ط¹ط± ظ…ط­ط§ظٹط¯ط© (ط§ظ„ط¯ط±ط¬ط©: {sentiment_score:.2f})"

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


async def aggregate_top_movers():
    all_tickers = []
    async def fetch(ex_id, ex):
        try: return [dict(t, exchange=ex_id) for t in (await ex.fetch_tickers()).values()]
        except Exception: return []
    results = await asyncio.gather(*[fetch(ex_id, ex) for ex_id, ex in bot_state.public_exchanges.items()])
    for res in results: all_tickers.extend(res)
    settings = bot_state.settings
    excluded_bases = settings['stablecoin_filter']['exclude_bases']
    min_volume = settings['liquidity_filters']['min_quote_volume_24h_usd']
    usdt_tickers = [t for t in all_tickers if t.get('symbol') and t['symbol'].upper().endswith('/USDT') and t['symbol'].split('/')[0] not in excluded_bases and t.get('quoteVolume', 0) and t['quoteVolume'] >= min_volume and not any(k in t['symbol'].upper() for k in ['UP','DOWN','3L','3S','BEAR','BULL'])]
    sorted_tickers = sorted(usdt_tickers, key=lambda t: t.get('quoteVolume', 0), reverse=True)
    unique_symbols = {t['symbol']: {'exchange': t['exchange'], 'symbol': t['symbol']} for t in sorted_tickers}
    final_list = list(unique_symbols.values())[:settings['top_n_symbols_by_volume']]
    logger.info(f"Aggregated markets. Found {len(all_tickers)} tickers -> Post-filter: {len(usdt_tickers)} -> Selected top {len(final_list)} unique pairs.")
    bot_state.status_snapshot['markets_found'] = len(final_list)
    return final_list

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
        exchange = bot_state.public_exchanges.get(market_info['exchange'])
        if not exchange or not settings.get('active_scanners'):
            queue.task_done()
            continue
        try:
            liq_filters, vol_filters, ema_filters = settings['liquidity_filters'], settings['volatility_filters'], settings['ema_trend_filter']

            orderbook = await exchange.fetch_order_book(symbol, limit=20)
            if not orderbook or not orderbook['bids'] or not orderbook['asks']:
                logger.debug(f"Reject {symbol}: Could not fetch order book."); continue

            best_bid, best_ask = orderbook['bids'][0][0], orderbook['asks'][0][0]
            if best_bid <= 0: logger.debug(f"Reject {symbol}: Invalid bid price."); continue

            spread_percent = ((best_ask - best_bid) / best_bid) * 100
            if spread_percent > liq_filters['max_spread_percent']:
                logger.debug(f"Reject {symbol}: High Spread ({spread_percent:.2f}% > {liq_filters['max_spread_percent']}%)"); continue

            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            if len(ohlcv) < ema_filters['ema_period']:
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

            atr_percent = (df[atr_col_name].iloc[-2] / last_close) * 100
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
                current_atr = df.iloc[-2].get(find_col(df.columns, f"ATRr_{settings['atr_period']}"), 0)
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
                    results_list.append({"symbol": symbol, "exchange": market_info['exchange'].capitalize(), "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "timestamp": df.index[-2], "reason": reason_str, "strength": len(confirmed_reasons)})
                else:
                    logger.debug(f"Reject {symbol} Signal: Small TP/SL (TP: {tp_percent_calc:.2f}%, SL: {sl_percent_calc:.2f}%)")

        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded for {symbol} on {market_info['exchange']}. Pausing...: {e}")
            await asyncio.sleep(10)
        except ccxt.NetworkError as e:
            logger.warning(f"Network error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"CRITICAL ERROR in worker for {symbol}: {e}", exc_info=True)
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
        elif exchange_id == 'kucoin':
            min_notional = float(market_info.get('info', {}).get('minProvideSize', 5.0))

        trade_amount_usdt = max(user_trade_amount_usdt, min_notional)
        if min_notional > user_trade_amount_usdt:
             logger.warning(f"User trade size ${user_trade_amount_usdt} for {symbol} is below exchange minimum of ${min_notional}. Using exchange minimum.")

        if usdt_balance < trade_amount_usdt:
            return {'success': False, 'data': f"ط±طµظٹط¯ظƒ ط§ظ„ط­ط§ظ„ظٹ ${usdt_balance:.2f} ط؛ظٹط± ظƒط§ظپظچ ظ„ظپطھط­ طµظپظ‚ط© ط¨ظ‚ظٹظ…ط© ${trade_amount_usdt:.2f}."}
        
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
        return {'success': False, 'data': f"ط­ط¯ط« ط®ط·ط£ ظ…ظ† ط§ظ„ظ…ظ†طµط© ط¹ظ†ط¯ ظ…ط­ط§ظˆظ„ط© ط§ظ„ط´ط±ط§ط،: `{str(e)}`"}

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
        return {'success': False, 'manual_check_required': True, 'data': f"طھظ… ط¥ط±ط³ط§ظ„ ط£ظ…ط± ط§ظ„ط´ط±ط§ط، ظ„ظƒظ† ظپط´ظ„ ط§ظ„طھط­ظ‚ظ‚ ظ…ظ†ظ‡. **ظٹط±ط¬ظ‰ ط§ظ„طھط­ظ‚ظ‚ ظ…ظ† ط§ظ„ظ…ظ†طµط© ظٹط¯ظˆظٹط§ظ‹!** ID: `{buy_order.get('id', 'N/A')}`. Error: `{e}`"}

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
                await send_telegram_message(context.bot, {'custom_message': f"**âڑ ï¸ڈ طھظ… ط¥ظٹظ‚ط§ظپ ط§ظ„ظپط­طµ ط§ظ„طھظ„ظ‚ط§ط¦ظٹ ظ…ط¤ظ‚طھط§ظ‹**\n\n**ط§ظ„ط³ط¨ط¨:** ظ…ط²ط§ط¬ ط§ظ„ط³ظˆظ‚ ط³ظ„ط¨ظٹ/ط®ط·ط±.\n**ط§ظ„طھظپط§طµظٹظ„:** {mood_reason}.\n\n*ط³ظٹطھظ… ط§ط³طھط¦ظ†ط§ظپ ط§ظ„ظپط­طµ ط¹ظ†ط¯ظ…ط§ طھطھط­ط³ظ† ط§ظ„ط¸ط±ظˆظپ.*", 'target_chat': TELEGRAM_CHAT_ID}); return

        is_market_ok, btc_reason = await check_market_regime()
        bot_state.status_snapshot['btc_market_mood'] = "ط¥ظٹط¬ط§ط¨ظٹ âœ…" if is_market_ok else "ط³ظ„ط¨ظٹ â‌Œ"

        if settings.get('market_regime_filter_enabled', True) and not is_market_ok:
            logger.info(f"Skipping scan: {btc_reason}")
            await send_telegram_message(context.bot, {'custom_message': f"**âڑ ï¸ڈ طھظ… ط¥ظٹظ‚ط§ظپ ط§ظ„ظپط­طµ ط§ظ„طھظ„ظ‚ط§ط¦ظٹ ظ…ط¤ظ‚طھط§ظ‹**\n\n**ط§ظ„ط³ط¨ط¨:** ظ…ط²ط§ط¬ ط§ظ„ط³ظˆظ‚ ط³ظ„ط¨ظٹ/ط®ط·ط±.\n**ط§ظ„طھظپط§طµظٹظ„:** {btc_reason}.\n\n*ط³ظٹطھظ… ط§ط³طھط¦ظ†ط§ظپ ط§ظ„ظپط­طµ ط¹ظ†ط¯ظ…ط§ طھطھط­ط³ظ† ط§ظ„ط¸ط±ظˆظپ.*", 'target_chat': TELEGRAM_CHAT_ID}); return

        status = bot_state.status_snapshot
        status.update({"scan_in_progress": True, "last_scan_start_time": datetime.now(EGYPT_TZ)})
        
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'ظ†ط´ط·ط©' AND trade_mode = 'virtual'")
            active_virtual_trades = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'ظ†ط´ط·ط©' AND trade_mode = 'real'")
            active_real_trades = cursor.fetchone()[0]
            conn.close()
            active_trades_count = active_virtual_trades + active_real_trades
        except Exception as e:
            logger.error(f"DB Error in perform_scan: {e}"); active_trades_count = settings.get("max_concurrent_trades", 10)

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
        last_signal_time = bot_state.last_signal_time

        for signal in signals:
            if time.time() - last_signal_time.get(signal['symbol'], 0) <= (SCAN_INTERVAL_SECONDS * 4):
                logger.info(f"Signal for {signal['symbol']} skipped due to cooldown."); continue

            signal_exchange_id = signal['exchange'].lower()
            per_exchange_settings = settings.get("real_trading_per_exchange", {})
            is_real_mode_enabled = per_exchange_settings.get(signal_exchange_id, False)

            exchange_is_tradeable = signal_exchange_id in bot_state.exchanges and bot_state.exchanges[signal_exchange_id].apiKey
            attempt_real_trade = is_real_mode_enabled and exchange_is_tradeable
            signal['is_real_trade'] = attempt_real_trade

            if attempt_real_trade:
                await send_telegram_message(context.bot, {'custom_message': f"**ًں”ژ طھظ… ط§ظ„ط¹ط«ظˆط± ط¹ظ„ظ‰ ط¥ط´ط§ط±ط© ط­ظ‚ظٹظ‚ظٹط© ظ„ظ€ `{signal['symbol']}`... ط¬ط§ط±ظٹ ظ…ط­ط§ظˆظ„ط© ط§ظ„طھظ†ظپظٹط° ط¹ظ„ظ‰ `{signal['exchange']}`.**"})
                try:
                    trade_result = await place_real_trade(signal)
                    
                    if trade_result.get('success'):
                        if isinstance(trade_result.get('data'), dict):
                            signal.update(trade_result['data'])
                        
                        if log_recommendation_to_db(signal):
                            await send_telegram_message(context.bot, signal, is_new=True)
                            new_trades += 1
                            if trade_result.get('exit_orders_failed'):
                                await send_telegram_message(context.bot, {'custom_message': f"**ًںڑ¨ طھط­ط°ظٹط±:** طھظ… ط´ط±ط§ط، `{signal['symbol']}` ط¨ظ†ط¬ط§ط­ ظˆطھط³ط¬ظٹظ„ظ‡ط§طŒ **ظ„ظƒظ† ظپط´ظ„ ظˆط¶ط¹ ط£ظˆط§ظ…ط± ط§ظ„ظ‡ط¯ظپ/ط§ظ„ظˆظ‚ظپ طھظ„ظ‚ط§ط¦ظٹط§ظ‹.**\n\n**ظٹط±ط¬ظ‰ ظˆط¶ط¹ظ‡ط§ ظٹط¯ظˆظٹط§ظ‹ ط§ظ„ط¢ظ†!**"})
                        else: 
                            await send_telegram_message(context.bot, {'custom_message': f"**âڑ ï¸ڈ ط®ط·ط£ ط­ط±ط¬:** طھظ… طھظ†ظپظٹط° طµظپظ‚ط© `{signal['symbol']}` ظ„ظƒظ† ظپط´ظ„ طھط³ط¬ظٹظ„ظ‡ط§ ظپظٹ ظ‚ط§ط¹ط¯ط© ط§ظ„ط¨ظٹط§ظ†ط§طھ. **ظٹط±ط¬ظ‰ ط§ظ„ظ…طھط§ط¨ط¹ط© ط§ظ„ظٹط¯ظˆظٹط© ظپظˆط±ط§ظ‹!**"})
                    else:
                        await send_telegram_message(context.bot, {'custom_message': f"**â‌Œ ظپط´ظ„ طھظ†ظپظٹط° طµظپظ‚ط© `{signal['symbol']}`**\n\n**ط§ظ„ط³ط¨ط¨:** {trade_result.get('data', 'ط³ط¨ط¨ ط؛ظٹط± ظ…ط¹ط±ظˆظپ')}"})
                
                except Exception as e:
                    logger.critical(f"CRITICAL UNHANDLED ERROR during real trade execution for {signal['symbol']}: {e}", exc_info=True)
                    await send_telegram_message(context.bot, {'custom_message': f"**â‌Œ ظپط´ظ„ ط­ط±ط¬ ظˆط؛ظٹط± ظ…ط¹ط§ظ„ط¬ ط£ط«ظ†ط§ط، ظ…ط­ط§ظˆظ„ط© طھظ†ظپظٹط° طµظپظ‚ط© `{signal['symbol']}`.**\n\n**ط§ظ„ط®ط·ط£:** `{str(e)}`\n\n*ظٹط±ط¬ظ‰ ط§ظ„طھط­ظ‚ظ‚ ظ…ظ† ط§ظ„ظ…ظ†طµط© ظˆظ…ظ† ط³ط¬ظ„ط§طھ ط§ظ„ط£ط®ط·ط§ط، (logs).*"})
            
            else: # ط§ظ„طµظپظ‚ط§طھ ط§ظ„ظˆظ‡ظ…ظٹط©
                if active_trades_count < settings.get("max_concurrent_trades", 10):
                    trade_amount_usdt = settings["virtual_portfolio_balance_usdt"] * (settings["virtual_trade_size_percentage"] / 100)
                    signal.update({'quantity': trade_amount_usdt / signal['entry_price'], 'entry_value_usdt': trade_amount_usdt})
                    if trade_id := log_recommendation_to_db(signal):
                        signal['trade_id'] = trade_id
                        await send_telegram_message(context.bot, signal, is_new=True)
                        new_trades += 1
                else:
                    await send_telegram_message(context.bot, signal, is_opportunity=True)
                    opportunities += 1

            await asyncio.sleep(0.5)
            last_signal_time[signal['symbol']] = time.time()
        
        failures = failure_counter[0]
        logger.info(f"Scan complete. Found: {total_signals_found}, Entered: {new_trades}, Opportunities: {opportunities}, Failures: {failures}.")
        
        status['last_scan_end_time'] = datetime.now(EGYPT_TZ)
        scan_start_time = status.get('last_scan_start_time')
        scan_duration = (status['last_scan_end_time'] - scan_start_time).total_seconds() if isinstance(scan_start_time, datetime) else 0

        summary_message = (f"**ًں”¬ ظ…ظ„ط®طµ ط§ظ„ظپط­طµ ط§ظ„ط£ط®ظٹط±**\n\n"
                           f"- **ط§ظ„ط­ط§ظ„ط©:** ط§ظƒطھظ…ظ„ ط¨ظ†ط¬ط§ط­\n"
                           f"- **ظˆط¶ط¹ ط§ظ„ط³ظˆظ‚ (BTC):** {status['btc_market_mood']}\n"
                           f"- **ط§ظ„ظ…ط¯ط©:** {scan_duration:.0f} ط«ط§ظ†ظٹط©\n"
                           f"- **ط§ظ„ط¹ظ…ظ„ط§طھ ط§ظ„ظ…ظپط­ظˆطµط©:** {len(top_markets)}\n\n"
                           f"- - - - - - - - - - - - - - - - - -\n"
                           f"- **ط¥ط¬ظ…ط§ظ„ظٹ ط§ظ„ط¥ط´ط§ط±ط§طھ ط§ظ„ظ…ظƒطھط´ظپط©:** {total_signals_found}\n"
                           f"- **âœ… طµظپظ‚ط§طھ ط¬ط¯ظٹط¯ط© ظپظڈطھط­طھ:** {new_trades}\n"
                           f"- **ًں’، ظپط±طµ ظ„ظ„ظ…ط±ط§ظ‚ط¨ط©:** {opportunities}\n"
                           f"- **âڑ ï¸ڈ ط£ط®ط·ط§ط، ظپظٹ ط§ظ„طھط­ظ„ظٹظ„:** {failures}\n"
                           f"- - - - - - - - - - - - - - - - - -\n\n"
                           f"*ط§ظ„ظپط­طµ ط§ظ„طھط§ظ„ظٹ ظ…ط¬ط¯ظˆظ„ طھظ„ظ‚ط§ط¦ظٹط§ظ‹.*")

        await send_telegram_message(context.bot, {'custom_message': summary_message, 'target_chat': TELEGRAM_CHAT_ID})

        status['scan_in_progress'] = False

        bot_state.scan_history.append({'signals': total_signals_found, 'failures': failures})
        await analyze_performance_and_suggest(context)

async def send_telegram_message(bot, signal_data, is_new=False, is_opportunity=False, update_type=None):
    message, keyboard, target_chat = "", None, TELEGRAM_CHAT_ID
    def format_price(price): 
        if price is None: return "N/A"
        return f"{price:,.8f}" if price < 0.01 else f"{price:,.4f}"

    if 'custom_message' in signal_data:
        message, target_chat = signal_data['custom_message'], signal_data.get('target_chat', TELEGRAM_CHAT_ID)
        if 'keyboard' in signal_data: keyboard = signal_data['keyboard']

    elif is_new or is_opportunity:
        target_chat = TELEGRAM_SIGNAL_CHANNEL_ID
        strength_stars = 'â­گ' * signal_data.get('strength', 1)

        trade_type_title = "ًںڑ¨ طµظپظ‚ط© ط­ظ‚ظٹظ‚ظٹط© ًںڑ¨" if signal_data.get('is_real_trade') else "âœ… طھظˆطµظٹط© ط´ط±ط§ط، ط¬ط¯ظٹط¯ط©"
        title = f"**{trade_type_title} | {signal_data['symbol']}**" if is_new else f"**ًں’، ظپط±طµط© ظ…ط­طھظ…ظ„ط© | {signal_data['symbol']}**"

        entry, tp, sl = signal_data['entry_price'], signal_data['take_profit'], signal_data['stop_loss']
        tp_percent, sl_percent = ((tp - entry) / entry * 100), ((entry - sl) / entry * 100)
        id_line = f"\n*ظ„ظ„ظ…طھط§ط¨ط¹ط© ط§ط¶ط؛ط·: /check {signal_data.get('trade_id', 'N/A')}*" if is_new else ""

        reasons_en = signal_data['reason'].split(' + ')
        reasons_ar = ' + '.join([STRATEGY_NAMES_AR.get(r, r) for r in reasons_en])

        message = (f"**Signal Alert | طھظ†ط¨ظٹظ‡ ط¥ط´ط§ط±ط©**\n"
                   f"------------------------------------\n"
                   f"{title}\n"
                   f"------------------------------------\n"
                   f"ًں”¹ **ط§ظ„ظ…ظ†طµط©:** {signal_data['exchange']}\n"
                   f"â­گ **ظ‚ظˆط© ط§ظ„ط¥ط´ط§ط±ط©:** {strength_stars}\n"
                   f"ًں”چ **ط§ظ„ط§ط³طھط±ط§طھظٹط¬ظٹط©:** {reasons_ar}\n\n"
                   f"ًں“ˆ **ظ†ظ‚ط·ط© ط§ظ„ط¯ط®ظˆظ„:** `{format_price(entry)}`\n"
                   f"ًںژ¯ **ط§ظ„ظ‡ط¯ظپ:** `{format_price(tp)}` (+{tp_percent:.2f}%)\n"
                   f"ًں›‘ **ط§ظ„ظˆظ‚ظپ:** `{format_price(sl)}` (-{sl_percent:.2f}%)"
                   f"{id_line}")
    elif update_type == 'tsl_activation':
        message = (f"**ًںڑ€ طھط£ظ…ظٹظ† ط§ظ„ط£ط±ط¨ط§ط­! | #{signal_data['id']} {signal_data['symbol']}**\n\n"
                   f"طھظ… ط±ظپط¹ ظˆظ‚ظپ ط§ظ„ط®ط³ط§ط±ط© ط¥ظ„ظ‰ ظ†ظ‚ط·ط© ط§ظ„ط¯ط®ظˆظ„.\n"
                   f"**ظ‡ط°ظ‡ ط§ظ„طµظپظ‚ط© ط§ظ„ط¢ظ† ظ…ط¤ظ…ظژظ‘ظ†ط© ط¨ط§ظ„ظƒط§ظ…ظ„ ظˆط¨ط¯ظˆظ† ظ…ط®ط§ط·ط±ط©!**\n\n"
                   f"*ط¯ط¹ ط§ظ„ط£ط±ط¨ط§ط­ طھظ†ظ…ظˆ!*")
    elif update_type == 'tsl_update_real':
        message = (f"**ًں”” طھظ†ط¨ظٹظ‡ طھط­ط¯ظٹط« ظˆظ‚ظپ ط§ظ„ط®ط³ط§ط±ط© (طµظپظ‚ط© ط­ظ‚ظٹظ‚ظٹط©) ًں””**\n\n"
                   f"**طµظپظ‚ط©:** `#{signal_data['id']} {signal_data['symbol']}`\n\n"
                   f"ظˆطµظ„ ط§ظ„ط³ط¹ط± ط¥ظ„ظ‰ `{format_price(signal_data['current_price'])}`.\n"
                   f"**ط¥ط¬ط±ط§ط، ظ…ظ‚طھط±ط­:** ظ‚ظ… ط¨طھط¹ط¯ظٹظ„ ط£ظ…ط± ظˆظ‚ظپ ط§ظ„ط®ط³ط§ط±ط© ظٹط¯ظˆظٹط§ظ‹ ط¥ظ„ظ‰ `{format_price(signal_data['new_sl'])}` ظ„طھط£ظ…ظٹظ† ط§ظ„ط£ط±ط¨ط§ط­.")


    if not message: return
    try:
        await bot.send_message(chat_id=target_chat, text=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
    except BadRequest as e:
        if 'Chat not found' in str(e):
            logger.critical(f"CRITICAL: Chat not found for target_chat: {target_chat}. The bot might not be an admin or the ID is wrong. Error: {e}")
            if str(target_chat) == str(TELEGRAM_SIGNAL_CHANNEL_ID) and str(target_chat) != str(TELEGRAM_CHAT_ID):
                try:
                    await bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=f"**âڑ ï¸ڈ ظپط´ظ„ ط§ظ„ط¥ط±ط³ط§ظ„ ط¥ظ„ظ‰ ط§ظ„ظ‚ظ†ط§ط© âڑ ï¸ڈ**\n\nظ„ظ… ط£طھظ…ظƒظ† ظ…ظ† ط¥ط±ط³ط§ظ„ ط±ط³ط§ظ„ط© ط¥ظ„ظ‰ ط§ظ„ظ‚ظ†ط§ط© (`{target_chat}`).\n\n**ط§ظ„ط³ط¨ط¨:** `Chat not found`\n\n**ط§ظ„ط­ظ„:**\n1. طھط£ظƒط¯ ظ…ظ† ط£ظ†ظ†ظٹ (ط§ظ„ط¨ظˆطھ) ط¹ط¶ظˆ ظپظٹ ط§ظ„ظ‚ظ†ط§ط©.\n2. طھط£ظƒط¯ ظ…ظ† ط£ظ†ظ†ظٹ ظ…ط´ط±ظپ (Admin) ظپظٹ ط§ظ„ظ‚ظ†ط§ط© ظˆظ„ط¯ظٹ طµظ„ط§ط­ظٹط© ط¥ط±ط³ط§ظ„ ط§ظ„ط±ط³ط§ط¦ظ„.\n3. طھط­ظ‚ظ‚ ظ…ظ† ط£ظ† `TELEGRAM_SIGNAL_CHANNEL_ID` طµط­ظٹط­.",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except Exception as admin_e:
                    logger.error(f"Failed to send admin warning about ChatNotFound: {admin_e}")
        else:
            logger.error(f"Failed to send Telegram message to {target_chat} (BadRequest): {e}")
    except Exception as e:
        logger.error(f"Failed to send Telegram message to {target_chat}: {e}")

async def track_open_trades(context: ContextTypes.DEFAULT_TYPE):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'ظ†ط´ط·ط©'")
        active_trades = [dict(row) for row in cursor.fetchall()]; conn.close()
    except Exception as e: logger.error(f"DB error in track_open_trades: {e}"); return
    
    bot_state.status_snapshot['active_trades_count'] = len(active_trades)
    if not active_trades: return

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

        current_stop_loss = trade.get('stop_loss') or 0
        current_take_profit = trade.get('take_profit')
        if current_take_profit is not None and current_price >= current_take_profit:
            await close_trade_in_db(context, trade, current_price, 'ظ†ط§ط¬ط­ط©')
            return
        if current_stop_loss > 0 and current_price <= current_stop_loss:
            await close_trade_in_db(context, trade, current_price, 'ظپط§ط´ظ„ط©')
            return

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
        logger.critical(f"CRITICAL FAILURE in automated TSL for trade #{trade['id']} ({symbol}): {e}", exc_info=True)
        await send_telegram_message(context.bot, {'custom_message': f"**ًںڑ¨ ظپط´ظ„ ط­ط±ط¬ ظپظٹ ط£طھظ…طھط© ط§ظ„ظˆظ‚ظپ ط§ظ„ظ…طھط­ط±ظƒ ًںڑ¨**\n\n**طµظپظ‚ط©:** `#{trade['id']} {symbol}`\n**ط§ظ„ط®ط·ط£:** `{e}`\n\n**ظ‚ط¯ طھظƒظˆظ† ط§ظ„طµظپظ‚ط© ط§ظ„ط¢ظ† ط¨ط¯ظˆظ† ط­ظ…ط§ظٹط©! ظٹط±ط¬ظ‰ ط§ظ„ظ…طھط§ط¨ط¹ط© ط§ظ„ظٹط¯ظˆظٹط© ظپظˆط±ط§ظ‹!**"})


async def close_trade_in_db(context: ContextTypes.DEFAULT_TYPE, trade: dict, exit_price: float, status: str):
    pnl_usdt = (exit_price - trade['entry_price']) * trade['quantity']
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
                       (status, exit_price, closed_at_str, exit_price * trade['quantity'], pnl_usdt, trade['id']))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB update failed while closing trade #{trade['id']}: {e}")
        return
    
    trade_type_str = "(طµظپظ‚ط© ط­ظ‚ظٹظ‚ظٹط©)" if trade.get('trade_mode') == 'real' else ""
    pnl_percent = (pnl_usdt / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0
    message = ""
    if status == 'ظ†ط§ط¬ط­ط©':
        message = (f"**ًں“¦ ط¥ط؛ظ„ط§ظ‚ طµظپظ‚ط© {trade_type_str} | #{trade['id']} {trade['symbol']}**\n\n"
                   f"**ط§ظ„ط­ط§ظ„ط©: âœ… ظ†ط§ط¬ط­ط© (طھظ… طھط­ظ‚ظٹظ‚ ط§ظ„ظ‡ط¯ظپ)**\n"
                   f"ًں’° **ط§ظ„ط±ط¨ط­:** `${pnl_usdt:+.2f}` (`{pnl_percent:+.2f}%`)\n\n"
                   f"- **ظ…ط¯ط© ط§ظ„طµظپظ‚ط©:** {duration_str}")
    else: 
        message = (f"**ًں“¦ ط¥ط؛ظ„ط§ظ‚ طµظپظ‚ط© {trade_type_str} | #{trade['id']} {trade['symbol']}**\n\n"
                   f"**ط§ظ„ط­ط§ظ„ط©: â‌Œ ظپط§ط´ظ„ط© (طھظ… ط¶ط±ط¨ ط§ظ„ظˆظ‚ظپ)**\n"
                   f"ًں’° **ط§ظ„ط®ط³ط§ط±ط©:** `${pnl_usdt:.2f}` (`{pnl_percent:.2f}%`)\n\n"
                   f"- **ظ…ط¯ط© ط§ظ„طµظپظ‚ط©:** {duration_str}")

    await send_telegram_message(context.bot, {'custom_message': message, 'target_chat': TELEGRAM_SIGNAL_CHANNEL_ID})

async def update_trade_sl_in_db(context: ContextTypes.DEFAULT_TYPE, trade: dict, new_sl: float, highest_price: float, is_activation: bool = False, silent: bool = False, new_exit_ids_json: str = None):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        
        sql = "UPDATE trades SET stop_loss=?, highest_price=?, trailing_sl_active=? "
        params = [new_sl, highest_price, True]
        
        if new_exit_ids_json:
            sql += ", exit_order_ids_json=? "
            params.append(new_exit_ids_json)

        sql += "WHERE id=?"
        params.append(trade['id'])

        cursor.execute(sql, tuple(params))
        conn.commit()
        conn.close()
        
        log_msg = f"Trailing SL {'activated' if is_activation else 'updated'} for trade #{trade['id']}. New SL: {new_sl}"
        if new_exit_ids_json:
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
    is_technically_bullish, is_sentiment_bullish, fng_index = True, True, "N/A"
    try:
        if binance := bot_state.public_exchanges.get('binance'):
            ohlcv = await binance.fetch_ohlcv('BTC/USDT', '4h', limit=55)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['sma50'] = ta.sma(df['close'], length=50)
            is_technically_bullish = df['close'].iloc[-1] > df['sma50'].iloc[-1]
    except Exception as e:
        logger.error(f"Error checking BTC trend: {e}")
    if settings.get("fear_and_greed_filter_enabled", True):
        if (fng_value := await get_fear_and_greed_index()) is not None:
            fng_index = fng_value
            is_sentiment_bullish = fng_index >= settings.get("fear_and_greed_threshold", 30)
    if not is_technically_bullish:
        return False, "ط§طھط¬ط§ظ‡ BTC ظ‡ط§ط¨ط· (طھط­طھ ظ…طھظˆط³ط· 50 ط¹ظ„ظ‰ 4 ط³ط§ط¹ط§طھ)."
    if not is_sentiment_bullish:
        return False, f"ظ…ط´ط§ط¹ط± ط®ظˆظپ ط´ط¯ظٹط¯ (ظ…ط¤ط´ط± F&G: {fng_index} طھط­طھ ط§ظ„ط­ط¯ {settings.get('fear_and_greed_threshold')})."
    return True, "ظˆط¶ط¹ ط§ظ„ط³ظˆظ‚ ظ…ظ†ط§ط³ط¨ ظ„طµظپظ‚ط§طھ ط§ظ„ط´ط±ط§ط،."

async def analyze_performance_and_suggest(context: ContextTypes.DEFAULT_TYPE):
    settings = bot_state.settings
    history = bot_state.scan_history

    if len(history) < 5 or (time.time() - settings.get('last_suggestion_time', 0)) < 7200:
        return

    avg_signals = sum(item['signals'] for item in history) / len(history)
    current_preset = settings.get('active_preset_name', 'PRO')

    suggestion, market_desc, reason = None, None, None

    if avg_signals < 0.5 and current_preset == "STRICT":
        suggestion = "PRO"
        market_desc = "ط§ظ„ط³ظˆظ‚ ظٹط¨ط¯ظˆ ط¨ط·ظٹط¦ط§ظ‹ ط¬ط¯ط§ظ‹ ظˆط§ظ„ط¥ط´ط§ط±ط§طھ ط´ط­ظٹط­ط©."
        reason = "ظ†ظ…ط· 'PRO' ط£ظƒط«ط± طھظˆط§ط²ظ†ط§ظ‹ ظˆظ‚ط¯ ظٹط³ط§ط¹ط¯ظ†ط§ ظپظٹ ط§ظ„طھظ‚ط§ط· ط§ظ„ظ…ط²ظٹط¯ ظ…ظ† ط§ظ„ظپط±طµ ط§ظ„ظ…ظ†ط§ط³ط¨ط© ط¯ظˆظ† ط§ظ„طھط¶ط­ظٹط© ط¨ط§ظ„ظƒط«ظٹط± ظ…ظ† ط§ظ„ط¬ظˆط¯ط©."
    elif avg_signals < 1 and current_preset == "PRO":
        suggestion = "LAX"
        market_desc = "ط¹ط¯ط¯ ط§ظ„ظپط±طµ ط§ظ„ظ…ظƒطھط´ظپط© ظ…ظ†ط®ظپط¶ ظ†ط³ط¨ظٹط§ظ‹."
        reason = "ظ†ظ…ط· 'LAX' (ظ…طھط³ط§ظ‡ظ„) ط³ظٹظˆط³ط¹ ظ†ط·ط§ظ‚ ط§ظ„ط¨ط­ط«طŒ ظ…ظ…ط§ ظ‚ط¯ ظٹط²ظٹط¯ ظ…ظ† ط¹ط¯ط¯ ط§ظ„ط¥ط´ط§ط±ط§طھ ظپظٹ ط³ظˆظ‚ ظ‡ط§ط¯ط¦."
    elif avg_signals > 8 and current_preset in ["LAX", "VERY_LAX"]:
        suggestion = "PRO"
        market_desc = "ط§ظ„ط³ظˆظ‚ ظ†ط´ط· ط¬ط¯ط§ظ‹ ظˆظ‡ظ†ط§ظƒ ط¹ط¯ط¯ ظƒط¨ظٹط± ظ…ظ† ط§ظ„ط¥ط´ط§ط±ط§طھ (ط¶ظˆط¶ط§ط،)."
        reason = "ظ†ظ…ط· 'PRO' ط³ظٹط³ط§ط¹ط¯ ظپظٹ ظپظ„طھط±ط© ط§ظ„ط¥ط´ط§ط±ط§طھ ط§ظ„ط£ط¶ط¹ظپ ظˆط§ظ„طھط±ظƒظٹط² ط¹ظ„ظ‰ ط§ظ„ظپط±طµ ط°ط§طھ ط§ظ„ط¬ظˆط¯ط© ط§ظ„ط£ط¹ظ„ظ‰."
    elif avg_signals > 12 and current_preset == "PRO":
        suggestion = "STRICT"
        market_desc = "ط§ظ„ط³ظˆظ‚ ظ…طھظ‚ظ„ط¨ ظˆظ‡ظ†ط§ظƒ ظپظٹط¶ط§ظ† ظ…ظ† ط§ظ„ط¥ط´ط§ط±ط§طھ."
        reason = "ظ†ظ…ط· 'STRICT' (ظ…طھط´ط¯ط¯) ط³ظٹط·ط¨ظ‚ ط£ظ‚ظˆظ‰ ط§ظ„ظپظ„ط§طھط± ظ„ط§طµط·ظٹط§ط¯ ط£ظپط¶ظ„ ط§ظ„ظپط±طµ ظپظ‚ط· ظپظٹ ظ‡ط°ط§ ط§ظ„ط³ظˆظ‚ ط§ظ„ظ…طھظ‚ظ„ط¨."

    if suggestion and suggestion != current_preset:
        message = (f"**ًں’، ط§ظ‚طھط±ط§ط­ ط°ظƒظٹ ظ„طھط­ط³ظٹظ† ط§ظ„ط£ط¯ط§ط،**\n\n"
                   f"*ظ…ط±ط­ط¨ط§ظ‹! ط¨ظ†ط§ط،ظ‹ ط¹ظ„ظ‰ طھط­ظ„ظٹظ„ ط¢ط®ط± {len(history)} ظپط­طµطŒ ظ„ط§ط­ط¸طھ طھط؛ظٹط±ط§ظ‹ ظپظٹ ط·ط¨ظٹط¹ط© ط§ظ„ط³ظˆظ‚.*\n\n"
                   f"**ط§ظ„ظ…ظ„ط§ط­ط¸ط©:**\n- {market_desc}\n\n"
                   f"**ط§ظ„ط§ظ‚طھط±ط§ط­:**\n- ط£ظ‚طھط±ط­ طھط؛ظٹظٹط± ظ†ظ…ط· ط§ظ„ط¥ط¹ط¯ط§ط¯ط§طھ ظ…ظ† `{current_preset}` ط¥ظ„ظ‰ **`{suggestion}`**.\n\n"
                   f"**ط§ظ„ط³ط¨ط¨:**\n- {reason}\n\n"
                   f"*ظ‡ظ„ طھظˆط§ظپظ‚ ط¹ظ„ظ‰ طھط·ط¨ظٹظ‚ ظ‡ط°ط§ ط§ظ„طھط؛ظٹظٹط±طں*")

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("âœ… ظ†ط¹ظ…طŒ ظ‚ظ… ط¨طھط·ط¨ظٹظ‚ ط§ظ„ظ†ظ…ط· ط§ظ„ظ…ظ‚طھط±ط­", callback_data=f"suggest_accept_{suggestion}")],
            [InlineKeyboardButton("â‌Œ ظ„ط§ ط´ظƒط±ط§ظ‹طŒ طھط¬ط§ظ‡ظ„ ط§ظ„ط§ظ‚طھط±ط§ط­", callback_data="suggest_decline")]
        ])

        await send_telegram_message(context.bot, {'custom_message': message, 'keyboard': keyboard})
        bot_state.settings['last_suggestion_time'] = time.time()
        save_settings()

# =======================================================================================
# --- Telegram Handlers ---
# =======================================================================================
main_menu_keyboard = [["Dashboard ًں–¥ï¸ڈ"], ["âڑ™ï¸ڈ ط§ظ„ط¥ط¹ط¯ط§ط¯ط§طھ"], ["â„¹ï¸ڈ ظ…ط³ط§ط¹ط¯ط©"]]
settings_menu_keyboard = [
    ["ًںڈپ ط£ظ†ظ…ط§ط· ط¬ط§ظ‡ط²ط©", "ًںژ­ طھظپط¹ظٹظ„/طھط¹ط·ظٹظ„ ط§ظ„ظ…ط§ط³ط­ط§طھ"], 
    ["ًں”§ طھط¹ط¯ظٹظ„ ط§ظ„ظ…ط¹ط§ظٹظٹط±", "ًںڑ¨ ط§ظ„طھط­ظƒظ… ط¨ط§ظ„طھط¯ط§ظˆظ„ ط§ظ„ط­ظ‚ظٹظ‚ظٹ"],
    ["ًں”™ ط§ظ„ظ‚ط§ط¦ظ…ط© ط§ظ„ط±ط¦ظٹط³ظٹط©"]
]

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = "ًں’£ ط£ظ‡ظ„ط§ظ‹ ط¨ظƒ ظپظٹ ط¨ظˆطھ **ظƒط§ط³ط­ط© ط§ظ„ط£ظ„ط؛ط§ظ…**!\n\n*(ط§ظ„ط¥طµط¯ط§ط± 5.2 - ط§ظ„ظ‡ظٹظƒظ„ط© ط§ظ„ظ†ظ‡ط§ط¦ظٹط©)*\n\nط§ط®طھط± ظ…ظ† ط§ظ„ظ‚ط§ط¦ظ…ط© ظ„ظ„ط¨ط¯ط،."
    await update.message.reply_text(welcome_message, reply_markup=ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True), parse_mode=ParseMode.MARKDOWN)

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.message or update.callback_query.message
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("ًں“ٹ ط§ظ„ط¥ط­طµط§ط¦ظٹط§طھ ط§ظ„ط¹ط§ظ…ط©", callback_data="dashboard_stats"), InlineKeyboardButton("ًں“ˆ ط§ظ„طµظپظ‚ط§طھ ط§ظ„ظ†ط´ط·ط©", callback_data="dashboard_active_trades")],
        [InlineKeyboardButton("ًں“œ طھظ‚ط±ظٹط± ط£ط¯ط§ط، ط§ظ„ط§ط³طھط±ط§طھظٹط¬ظٹط§طھ", callback_data="dashboard_strategy_report")],
        [InlineKeyboardButton("ًں“¸ ظ„ظ‚ط·ط© ظ„ظ„ظ…ط­ظپط¸ط©", callback_data="dashboard_snapshot"), InlineKeyboardButton("دپخ¯رپذ؛ طھظ‚ط±ظٹط± ط§ظ„ظ…ط®ط§ط·ط±", callback_data="dashboard_risk")],
        [InlineKeyboardButton("ًں”„ ظ…ط²ط§ظ…ظ†ط© ظˆظ…ط·ط§ط¨ظ‚ط© ط§ظ„ظ…ط­ظپط¸ط©", callback_data="dashboard_sync")],
        [InlineKeyboardButton("ًں› ï¸ڈ ط£ط¯ظˆط§طھ ط§ظ„طھط¯ط§ظˆظ„", callback_data="dashboard_tools"), InlineKeyboardButton("ًں•µï¸ڈâ€چâ™‚ï¸ڈ طھظ‚ط±ظٹط± ط§ظ„طھط´ط®ظٹطµ", callback_data="dashboard_debug")],
        [InlineKeyboardButton("ًں”„ طھط­ط¯ظٹط«", callback_data="dashboard_refresh")]
    ])
    message_text = "ًں–¥ï¸ڈ *ظ„ظˆط­ط© ط§ظ„طھط­ظƒظ… ط§ظ„ط±ط¦ظٹط³ظٹط©*\n\nط§ط®طھط± ط§ظ„طھظ‚ط±ظٹط± ط£ظˆ ط§ظ„ط¨ظٹط§ظ†ط§طھ ط§ظ„طھظٹ طھط±ظٹط¯ ط¹ط±ط¶ظ‡ط§:"

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


async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE): await (update.message or update.callback_query.message).reply_text("ط§ط®طھط± ط§ظ„ط¥ط¹ط¯ط§ط¯:", reply_markup=ReplyKeyboardMarkup(settings_menu_keyboard, resize_keyboard=True))

def get_scanners_keyboard():
    active_scanners = bot_state.settings.get("active_scanners", [])
    keyboard = [[InlineKeyboardButton(f"{'âœ…' if name in active_scanners else 'â‌Œ'} {STRATEGY_NAMES_AR.get(name, name)}", callback_data=f"toggle_scanner_{name}")] for name in SCANNERS.keys()]
    keyboard.append([InlineKeyboardButton("ًں”™ ط§ظ„ط¹ظˆط¯ط© ظ„ظ„ط¥ط¹ط¯ط§ط¯ط§طھ", callback_data="back_to_settings")])
    return InlineKeyboardMarkup(keyboard)

def get_presets_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("ًںڑ¦ ط§ط­طھط±ط§ظپظٹط© (ظ…طھظˆط§ط²ظ†ط©)", callback_data="preset_PRO"), InlineKeyboardButton("ًںژ¯ ظ…طھط´ط¯ط¯ط©", callback_data="preset_STRICT")],
        [InlineKeyboardButton("ًںŒ™ ظ…طھط³ط§ظ‡ظ„ط©", callback_data="preset_LAX"), InlineKeyboardButton("âڑ ï¸ڈ ظپط§ط¦ظ‚ ط§ظ„طھط³ط§ظ‡ظ„", callback_data="preset_VERY_LAX")],
        [InlineKeyboardButton("ًں”™ ط§ظ„ط¹ظˆط¯ط© ظ„ظ„ط¥ط¹ط¯ط§ط¯ط§طھ", callback_data="back_to_settings")]
    ])
    
async def show_real_trading_control_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.message or update.callback_query.message
    settings = bot_state.settings.get("real_trading_per_exchange", {})
    keyboard = []
    for ex_id in EXCHANGES_TO_SCAN:
        is_enabled = settings.get(ex_id, False)
        status_emoji = 'âœ…' if is_enabled else 'â‌Œ'
        button_text = f"{status_emoji} {ex_id.capitalize()}"
        callback_data = f"toggle_real_trade_{ex_id}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])
    keyboard.append([InlineKeyboardButton("ًں”™ ط§ظ„ط¹ظˆط¯ط© ظ„ظ„ط¥ط¹ط¯ط§ط¯ط§طھ", callback_data="back_to_settings")])
    
    await target_message.reply_text(
        "**ًںڑ¨ ط§ظ„طھط­ظƒظ… ط¨ط§ظ„طھط¯ط§ظˆظ„ ط§ظ„ط­ظ‚ظٹظ‚ظٹ ًںڑ¨**\n\nط§ط®طھط± ط§ظ„ظ…ظ†طµط© ظ„طھظپط¹ظٹظ„ ط£ظˆ طھط¹ط·ظٹظ„ ط§ظ„طھط¯ط§ظˆظ„ ط¹ظ„ظٹظ‡ط§:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

async def show_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.message or update.callback_query.message
    await target_message.reply_text("ط§ط®طھط± ظ†ظ…ط· ط¥ط¹ط¯ط§ط¯ط§طھ ط¬ط§ظ‡ط²:", reply_markup=get_presets_keyboard())
async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.message or update.callback_query.message
    await target_message.reply_text("ط§ط®طھط± ط§ظ„ظ…ط§ط³ط­ط§طھ ظ„طھظپط¹ظٹظ„ظ‡ط§ ط£ظˆ طھط¹ط·ظٹظ„ظ‡ط§:", reply_markup=get_scanners_keyboard())
async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard, settings = [], bot_state.settings
    for category, params in EDITABLE_PARAMS.items():
        keyboard.append([InlineKeyboardButton(f"--- {category} ---", callback_data="ignore")])
        for row in [params[i:i + 2] for i in range(0, len(params), 2)]:
            button_row = []
            for param_key in row:
                display_name = PARAM_DISPLAY_NAMES.get(param_key, param_key)
                current_value = settings.get(param_key, "N/A")
                text = f"{display_name}: {'ظ…ظڈظپط¹ظ‘ظ„ âœ…' if current_value else 'ظ…ظڈط¹ط·ظ‘ظ„ â‌Œ'}" if isinstance(current_value, bool) else f"{display_name}: {current_value}"
                button_row.append(InlineKeyboardButton(text, callback_data=f"param_{param_key}"))
            keyboard.append(button_row)
    keyboard.append([InlineKeyboardButton("ًں”™ ط§ظ„ط¹ظˆط¯ط© ظ„ظ„ط¥ط¹ط¯ط§ط¯ط§طھ", callback_data="back_to_settings")])
    message_text = "âڑ™ï¸ڈ *ط§ظ„ط¥ط¹ط¯ط§ط¯ط§طھ ط§ظ„ظ…طھظ‚ط¯ظ…ط©* âڑ™ï¸ڈ\n\nط§ط®طھط± ط§ظ„ط¥ط¹ط¯ط§ط¯ ط§ظ„ط°ظٹ طھط±ظٹط¯ طھط¹ط¯ظٹظ„ظ‡ ط¨ط§ظ„ط¶ط؛ط· ط¹ظ„ظٹظ‡:"
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
        "**ًں’£ ط£ظˆط§ظ…ط± ط¨ظˆطھ ظƒط§ط³ط­ط© ط§ظ„ط£ظ„ط؛ط§ظ… ًں’£**\n\n"
        "`/start` - ظ„ط¹ط±ط¶ ط§ظ„ظ‚ط§ط¦ظ…ط© ط§ظ„ط±ط¦ظٹط³ظٹط© ظˆط¨ط¯ط، ط§ظ„طھظپط§ط¹ظ„.\n"
        "`/check <ID>` - ظ„ظ…طھط§ط¨ط¹ط© ط­ط§ظ„ط© طµظپظ‚ط© ظ…ط¹ظٹظ†ط© ط¨ط§ط³طھط®ط¯ط§ظ… ط±ظ‚ظ…ظ‡ط§.\n"
        "`/trade` - ظ„ط¨ط¯ط، ط¹ظ…ظ„ظٹط© طھط¯ط§ظˆظ„ ظٹط¯ظˆظٹط© ظ„ط§ط®طھط¨ط§ط± ط§ظ„ط§طھطµط§ظ„ ط¨ط§ظ„ظ…ظ†طµط§طھ."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_mode_filter='all'):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); cursor = conn.cursor();
        
        query = "SELECT status, COUNT(*), SUM(pnl_usdt) FROM trades"
        params = []
        
        filter_conditions = []
        if trade_mode_filter != 'all':
            filter_conditions.append("trade_mode = ?")
            params.append(trade_mode_filter)

        if filter_conditions:
            query += " WHERE " + " AND ".join(filter_conditions)

        query += " GROUP BY status"
        cursor.execute(query, params)
        
        stats_data = cursor.fetchall(); conn.close()
        counts = {s: c for s, c, p in stats_data}; pnl = {s: (p or 0) for s, c, p in stats_data}
        total, active, successful, failed = sum(counts.values()), counts.get('ظ†ط´ط·ط©', 0), counts.get('ظ†ط§ط¬ط­ط©', 0), counts.get('ظپط§ط´ظ„ط©', 0)
        closed = successful + failed; win_rate = (successful / closed * 100) if closed > 0 else 0; total_pnl = sum(pnl.values())
        preset_name = bot_state.settings.get("active_preset_name", "N/A")
        
        mode_title_map = {'all': '(ط§ظ„ظƒظ„)', 'real': '(ط­ظ‚ظٹظ‚ظٹ ظپظ‚ط·)', 'virtual': '(ظˆظ‡ظ…ظٹ ظپظ‚ط·)'}
        title = mode_title_map.get(trade_mode_filter, '')

        balance_lines = []
        if trade_mode_filter == 'real':
            real_balance = await get_total_real_portfolio_value_usdt()
            balance_lines.append(f"ًں’° *ط¥ط¬ظ…ط§ظ„ظٹ ظ‚ظٹظ…ط© ط§ظ„ظ…ط­ظپط¸ط© ط§ظ„ط­ظ‚ظٹظ‚ظٹط©:* `${real_balance:.2f}`")
        elif trade_mode_filter == 'virtual':
            balance_lines.append(f"ًں“ˆ *ط§ظ„ط±طµظٹط¯ ط§ظ„ط§ظپطھط±ط§ط¶ظٹ:* `${bot_state.settings['virtual_portfolio_balance_usdt']:.2f}`")
        else: # 'all'
            real_balance = await get_total_real_portfolio_value_usdt()
            balance_lines.append(f"ًں’° *ظ‚ظٹظ…ط© ط§ظ„ظ…ط­ظپط¸ط© ط§ظ„ط­ظ‚ظٹظ‚ظٹط©:* `${real_balance:.2f}`")
            balance_lines.append(f"ًں“ˆ *ط§ظ„ط±طµظٹط¯ ط§ظ„ط§ظپطھط±ط§ط¶ظٹ:* `${bot_state.settings['virtual_portfolio_balance_usdt']:.2f}`")

        balance_section = "\n".join(balance_lines)

        stats_msg = (f"*ًں“ٹ ط¥ط­طµط§ط¦ظٹط§طھ ط§ظ„ظ…ط­ظپط¸ط© {title}*\n\n"
                       f"{balance_section}\n"
                       f"ًں’° *ط¥ط¬ظ…ط§ظ„ظٹ ط§ظ„ط±ط¨ط­/ط§ظ„ط®ط³ط§ط±ط©:* `${total_pnl:+.2f}`\n"
                       f"âڑ™ï¸ڈ *ط§ظ„ظ†ظ…ط· ط§ظ„ط­ط§ظ„ظٹ:* `{preset_name}`\n\n"
                       f"- *ط¥ط¬ظ…ط§ظ„ظٹ ط§ظ„طµظپظ‚ط§طھ:* `{total}` (`{active}` ظ†ط´ط·ط©)\n"
                       f"- *ط§ظ„ظ†ط§ط¬ط­ط©:* `{successful}` | *ط§ظ„ط±ط¨ط­:* `${pnl.get('ظ†ط§ط¬ط­ط©', 0):.2f}`\n"
                       f"- *ط§ظ„ظپط§ط´ظ„ط©:* `{failed}` | *ط§ظ„ط®ط³ط§ط±ط©:* `${abs(pnl.get('ظپط§ط´ظ„ط©', 0)):.2f}`\n"
                       f"- *ظ…ط¹ط¯ظ„ ط§ظ„ظ†ط¬ط§ط­:* `{win_rate:.2f}%`")
        return stats_msg, None
    except Exception as e:
        logger.error(f"Error in stats_command: {e}", exc_info=True)
        return "ط®ط·ط£ ظپظٹ ط¬ظ„ط¨ ط§ظ„ط¥ط­طµط§ط¦ظٹط§طھ.", None

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

        parts = [f"**ًں—“ï¸ڈ ط§ظ„طھظ‚ط±ظٹط± ط§ظ„ظٹظˆظ…ظٹ ط§ظ„ظ…ظپطµظ„ | {today_str}**\n"]

        def generate_section(title, trades):
            if not trades:
                return [f"\n--- **{title}** ---\nظ„ظ… ظٹطھظ… ط¥ط؛ظ„ط§ظ‚ ط£ظٹ طµظپظ‚ط§طھ ط§ظ„ظٹظˆظ…."]
            
            wins = [t for t in trades if t['status'] == 'ظ†ط§ط¬ط­ط©']
            losses = [t for t in trades if t['status'] == 'ظپط§ط´ظ„ط©']
            total_pnl = sum(t['pnl_usdt'] for t in trades if t['pnl_usdt'] is not None)
            win_rate = (len(wins) / len(trades) * 100) if trades else 0

            section_parts = [f"\n--- **{title}** ---"]
            section_parts.append(f"  - ط§ظ„ط±ط¨ط­/ط§ظ„ط®ط³ط§ط±ط© ط§ظ„طµط§ظپظٹ: `${total_pnl:+.2f}`")
            section_parts.append(f"  - âœ… ط§ظ„ط±ط§ط¨ط­ط©: {len(wins)} | â‌Œ ط§ظ„ط®ط§ط³ط±ط©: {len(losses)}")
            section_parts.append(f"  - ظ…ط¹ط¯ظ„ ط§ظ„ظ†ط¬ط§ط­: {win_rate:.1f}%")
            return section_parts

        parts.extend(generate_section("ًں’° ط§ظ„ط£ط¯ط§ط، ط§ظ„ط­ظ‚ظٹظ‚ظٹ", closed_real_today))
        parts.extend(generate_section("ًں“ٹ ط§ظ„ط£ط¯ط§ط، ط§ظ„ظˆظ‡ظ…ظٹ", closed_virtual_today))

        parts.append("\n\n*ط±ط³ط§ظ„ط© ط§ظ„ظٹظˆظ…: \"ط§ظ„ظ†ط¬ط§ط­ ظپظٹ ط§ظ„طھط¯ط§ظˆظ„ ظ‡ظˆ ظ†طھظٹط¬ط© ظ„ظ„ط§ظ†ط¶ط¨ط§ط· ظˆط§ظ„طµط¨ط± ظˆط§ظ„طھط¹ظ„ظ… ط§ظ„ظ…ط³طھظ…ط±.\"*")
        report_message = "\n".join(parts)

        await send_telegram_message(context.bot, {'custom_message': report_message, 'target_chat': TELEGRAM_SIGNAL_CHANNEL_ID})
    except Exception as e:
        logger.error(f"Failed to generate detailed daily report: {e}", exc_info=True)

async def daily_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message if update.callback_query else update.message
    await target_message.reply_text("âڈ³ ط¬ط§ط±ظٹ ط¥ط±ط³ط§ظ„ ط§ظ„طھظ‚ط±ظٹط± ط§ظ„ظٹظˆظ…ظٹ ط§ظ„ظ…ظپطµظ„...")
    await send_daily_report(context)
    await target_message.reply_text("âœ… طھظ… ط¥ط±ط³ط§ظ„ ط§ظ„طھظ‚ط±ظٹط± ط¨ظ†ط¬ط§ط­ ط¥ظ„ظ‰ ط§ظ„ظ‚ظ†ط§ط©.")

async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message if update.callback_query else update.message
    await target_message.reply_text("âڈ³ ط¬ط§ط±ظٹ ط¥ط¹ط¯ط§ط¯ طھظ‚ط±ظٹط± ط§ظ„طھط´ط®ظٹطµ ط§ظ„ط´ط§ظ…ظ„...")
    settings = bot_state.settings
    parts = [f"**ًں•µï¸ڈâ€چâ™‚ï¸ڈ طھظ‚ط±ظٹط± ط§ظ„طھط´ط®ظٹطµ ط§ظ„ط´ط§ظ…ظ„**\n\n*طھظ… ط¥ظ†ط´ط§ط¤ظ‡ ظپظٹ: {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')}*"]

    parts.append("\n- - - - - - - - - - - - - - - - - -")
    parts.append("**[ âڑ™ï¸ڈ ط­ط§ظ„ط© ط§ظ„ظ†ط¸ط§ظ… ظˆط§ظ„ط¨ظٹط¦ط© ]**")
    parts.append(f"- `NLTK (طھط­ظ„ظٹظ„ ط§ظ„ط£ط®ط¨ط§ط±):` {'ظ…طھط§ط­ط© âœ…' if NLTK_AVAILABLE else 'ط؛ظٹط± ظ…طھط§ط­ط© â‌Œ'}")
    parts.append(f"- `SciPy (طھط­ظ„ظٹظ„ ط§ظ„ط¯ط§ظٹظپط±ط¬ظ†ط³):` {'ظ…طھط§ط­ط© âœ…' if SCIPY_AVAILABLE else 'ط؛ظٹط± ظ…طھط§ط­ط© â‌Œ'}")
    parts.append(f"- `Alpha Vantage (ط¨ظٹط§ظ†ط§طھ ط§ظ‚طھطµط§ط¯ظٹط©):` {'ظ…ظˆط¬ظˆط¯ âœ…' if ALPHA_VANTAGE_API_KEY != 'YOUR_AV_KEY_HERE' else 'ظ…ظپظ‚ظˆط¯ âڑ ï¸ڈ'}")

    parts.append("\n**[ ًں“ٹ ط­ط§ظ„ط© ط§ظ„ط³ظˆظ‚ ط§ظ„ط­ط§ظ„ظٹط© ]**")
    mood_info = settings.get("last_market_mood", {})
    try:
        fng_value = await get_fear_and_greed_index()
        fng_text = "ط؛ظٹط± ظ…طھط§ط­"
        if fng_value is not None:
            classification = "ط®ظˆظپ ط´ط¯ظٹط¯" if fng_value < 25 else "ط®ظˆظپ" if fng_value < 45 else "ظ…ط­ط§ظٹط¯" if fng_value < 55 else "ط·ظ…ط¹" if fng_value < 75 else "ط·ظ…ط¹ ط´ط¯ظٹط¯"
            fng_text = f"{fng_value} ({classification})"
    except Exception as e:
        fng_text = f"ظپط´ظ„ ط§ظ„ط¬ظ„ط¨ ({e})"
    parts.append(f"- **ط§ظ„ظ…ط²ط§ط¬ ط§ظ„ط£ط³ط§ط³ظٹ (ط£ط®ط¨ط§ط±):** `{mood_info.get('mood', 'N/A')}`")
    parts.append(f"  - `{mood_info.get('reason', 'N/A')}`")
    parts.append(f"- **ط§ظ„ظ…ط²ط§ط¬ ط§ظ„ظپظ†ظٹ (BTC):** `{bot_state.status_snapshot['btc_market_mood']}`")
    parts.append(f"- **ظ…ط¤ط´ط± ط§ظ„ط®ظˆظپ ظˆط§ظ„ط·ظ…ط¹:** `{fng_text}`")

    status = bot_state.status_snapshot
    scan_duration = "N/A"
    if isinstance(status.get('last_scan_end_time'), datetime) and isinstance(status.get('last_scan_start_time'), datetime):
        duration_sec = (status['last_scan_end_time'] - status['last_scan_start_time']).total_seconds()
        scan_duration = f"{duration_sec:.0f} ط«ط§ظ†ظٹط©"
    parts.append("\n**[ ًں”¬ ط£ط¯ط§ط، ط¢ط®ط± ظپط­طµ ]**")
    parts.append(f"- **ظˆظ‚طھ ط§ظ„ط¨ط¯ط،:** `{status.get('last_scan_start_time', 'N/A')}`")
    parts.append(f"- **ط§ظ„ظ…ط¯ط©:** `{scan_duration}`")
    parts.append(f"- **ط§ظ„ط¹ظ…ظ„ط§طھ ط§ظ„ظ…ظپط­ظˆطµط©:** `{status['markets_found']}`")
    parts.append(f"- **ظپط´ظ„ ظپظٹ طھط­ظ„ظٹظ„:** `{(bot_state.scan_history[-1]['failures'] if bot_state.scan_history else 'N/A')}` ط¹ظ…ظ„ط§طھ")

    parts.append("\n**[ ًں”§ ط§ظ„ط¥ط¹ط¯ط§ط¯ط§طھ ط§ظ„ظ†ط´ط·ط© ]**")
    parts.append(f"- **ط§ظ„ظ†ظ…ط· ط§ظ„ط­ط§ظ„ظٹ:** `{settings.get('active_preset_name', 'N/A')}`")
    parts.append(f"- **ط§ظ„ظ…ط§ط³ط­ط§طھ ط§ظ„ظ…ظپط¹ظ„ط©:** `{', '.join(settings.get('active_scanners', []))}`")
    lf, vf = settings['liquidity_filters'], settings['volatility_filters']
    parts.append("- **ظپظ„ط§طھط± ط§ظ„ط³ظٹظˆظ„ط©:**")
    parts.append(f"  - `ط­ط¬ظ… ط§ظ„طھط¯ط§ظˆظ„ ط§ظ„ط£ط¯ظ†ظ‰:` ${lf['min_quote_volume_24h_usd']:,}")
    parts.append(f"  - `ط£ظ‚طµظ‰ ط³ط¨ط±ظٹط¯ ظ…ط³ظ…ظˆط­:` {lf['max_spread_percent']}%")
    parts.append(f"  - `ط§ظ„ط­ط¯ ط§ظ„ط£ط¯ظ†ظ‰ ظ„ظ€ RVOL:` {lf['min_rvol']}")
    parts.append("- **ظپظ„طھط± ط§ظ„طھظ‚ظ„ط¨:**")
    parts.append(f"  - `ط§ظ„ط­ط¯ ط§ظ„ط£ط¯ظ†ظ‰ ظ„ظ€ ATR:` {vf['min_atr_percent']}%")

    parts.append("\n**[ ًں”© ط­ط§ظ„ط© ط§ظ„ط¹ظ…ظ„ظٹط§طھ ط§ظ„ط¯ط§ط®ظ„ظٹط© ]**")
    if context.job_queue:
        try:
            scan_job = context.job_queue.get_jobs_by_name('perform_scan')
            track_job = context.job_queue.get_jobs_by_name('track_open_trades')
            scan_next = scan_job[0].next_t.astimezone(EGYPT_TZ).strftime('%H:%M:%S') if scan_job and scan_job[0].next_t else 'N/A'
            track_next = track_job[0].next_t.astimezone(EGYPT_TZ).strftime('%H:%M:%S') if track_job and track_job[0].next_t else 'N/A'
            parts.append("- **ط§ظ„ظ…ظ‡ط§ظ… ط§ظ„ظ…ط¬ط¯ظˆظ„ط©:**")
            parts.append(f"  - `ظپط­طµ ط§ظ„ط¹ظ…ظ„ط§طھ:` {'ظٹط¹ظ…ظ„'}, *ط§ظ„طھط§ظ„ظٹ ظپظٹ: {scan_next}*")
            parts.append(f"  - `ظ…طھط§ط¨ط¹ط© ط§ظ„طµظپظ‚ط§طھ:` {'ظٹط¹ظ…ظ„'}, *ط§ظ„طھط§ظ„ظٹ ظپظٹ: {track_next}*")
        except Exception as e:
            parts.append(f"- **ط§ظ„ظ…ظ‡ط§ظ… ط§ظ„ظ…ط¬ط¯ظˆظ„ط©:** ظپط´ظ„ ط§ظ„ظپط­طµ ({e})")
            
    parts.append("- **ط§ظ„ط§طھطµط§ظ„ ط¨ط§ظ„ظ…ظ†طµط§طھ:**")
    for ex_id in EXCHANGES_TO_SCAN:
        is_private_connected = ex_id in bot_state.exchanges and bot_state.exchanges[ex_id].apiKey
        is_public_connected = ex_id in bot_state.public_exchanges
        status_text = f"ط¹ط§ظ…: {'âœ…' if is_public_connected else 'â‌Œ'} | ط®ط§طµ: {'âœ…' if is_private_connected else 'â‌Œ'}"
        parts.append(f"  - `{ex_id.capitalize()}:` {status_text}")


    parts.append("- **ظ‚ط§ط¹ط¯ط© ط§ظ„ط¨ظٹط§ظ†ط§طھ:**")
    try:
        conn = sqlite3.connect(DB_FILE, timeout=5); cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades"); total_trades = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'ظ†ط´ط·ط©'"); active_trades = cursor.fetchone()[0]
        conn.close()
        db_size = os.path.getsize(DB_FILE) / (1024 * 1024)
        parts.append(f"  - `ط§ظ„ط§طھطµط§ظ„:` ظ†ط§ط¬ط­ âœ…")
        parts.append(f"  - `ط­ط¬ظ… ط§ظ„ظ…ظ„ظپ:` {db_size:.2f} MB")
        parts.append(f"  - `ط¥ط¬ظ…ط§ظ„ظٹ ط§ظ„طµظپظ‚ط§طھ:` {total_trades} ({active_trades} ظ†ط´ط·ط©)")
    except Exception as e: parts.append(f"  - `ط§ظ„ط§طھطµط§ظ„:` ظپط´ظ„ â‌Œ ({e})")
    parts.append("- - - - - - - - - - - - - - - - - -")

    await target_message.reply_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN)


async def check_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_id_from_callback=None):
    target = update.callback_query.message if trade_id_from_callback else update.message
    def format_price(price): return f"{price:,.8f}" if price < 0.01 else f"{price:,.4f}"
    try:
        trade_id = trade_id_from_callback or int(context.args[0])
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row; cursor = conn.cursor(); cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,));
        trade = dict(trade_row) if (trade_row := cursor.fetchone()) else None; conn.close()
        if not trade: await target.reply_text(f"ظ„ظ… ظٹطھظ… ط§ظ„ط¹ط«ظˆط± ط¹ظ„ظ‰ طµظپظ‚ط© ط¨ط§ظ„ط±ظ‚ظ… `{trade_id}`."); return
        if trade['status'] != 'ظ†ط´ط·ط©':
            pnl_percent = (trade['pnl_usdt'] / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0

            closed_at_dt_naive = datetime.strptime(trade['closed_at'], '%Y-%m-%d %H:%M:%S')
            closed_at_dt = EGYPT_TZ.localize(closed_at_dt_naive)
            message = f"ًں“‹ *ظ…ظ„ط®طµ ط§ظ„طµظپظ‚ط© #{trade_id}*\n\n*ط§ظ„ط¹ظ…ظ„ط©:* `{trade['symbol']}`\n*ط§ظ„ط­ط§ظ„ط©:* `{trade['status']}`\n*طھط§ط±ظٹط® ط§ظ„ط¥ط؛ظ„ط§ظ‚:* `{closed_at_dt.strftime('%Y-%m-%d %I:%M %p')}`\n*ط§ظ„ط±ط¨ط­/ط§ظ„ط®ط³ط§ط±ط©:* `${trade.get('pnl_usdt', 0):+.2f} ({pnl_percent:+.2f}%)`"
        else:
            if not (exchange := bot_state.public_exchanges.get(trade['exchange'].lower())): await target.reply_text("ط§ظ„ظ…ظ†طµط© ط؛ظٹط± ظ…طھطµظ„ط©."); return
            if not (ticker := await exchange.fetch_ticker(trade['symbol'])) or not (current_price := ticker.get('last') or ticker.get('close')):
                await target.reply_text(f"ظ„ظ… ط£طھظ…ظƒظ† ظ…ظ† ط¬ظ„ط¨ ط§ظ„ط³ط¹ط± ط§ظ„ط­ط§ظ„ظٹ ظ„ظ€ `{trade['symbol']}`."); return
            live_pnl = (current_price - trade['entry_price']) * trade['quantity']
            live_pnl_percent = (live_pnl / trade['entry_value_usdt'] * 100) if trade.get('entry_value_usdt', 0) > 0 else 0
            message = (f"ًں“ˆ *ظ…طھط§ط¨ط¹ط© ط­ظٹط© ظ„ظ„طµظپظ‚ط© #{trade_id}*\n\n"
                       f"â–«ï¸ڈ *ط§ظ„ط¹ظ…ظ„ط©:* `{trade['symbol']}` | *ط§ظ„ط­ط§ظ„ط©:* `ظ†ط´ط·ط©`\n"
                       f"â–«ï¸ڈ *ط³ط¹ط± ط§ظ„ط¯ط®ظˆظ„:* `${format_price(trade['entry_price'])}`\n"
                       f"â–«ï¸ڈ *ط§ظ„ط³ط¹ط± ط§ظ„ط­ط§ظ„ظٹ:* `${format_price(current_price)}`\n\n"
                       f"ًں’° *ط§ظ„ط±ط¨ط­/ط§ظ„ط®ط³ط§ط±ط© ط§ظ„ط­ط§ظ„ظٹط©:*\n`${live_pnl:+.2f} ({live_pnl_percent:+.2f}%)`")
        await target.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    except (ValueError, IndexError): await target.reply_text("ط±ظ‚ظ… طµظپظ‚ط© ط؛ظٹط± طµط§ظ„ط­. ظ…ط«ط§ظ„: `/check 17`")
    except Exception as e: logger.error(f"Error in check_trade_command: {e}", exc_info=True); await target.reply_text("ط­ط¯ط« ط®ط·ط£.")

async def show_active_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE, trade_mode_filter='all'):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
        
        query = "SELECT id, symbol, entry_value_usdt, exchange FROM trades WHERE status = 'ظ†ط´ط·ط©'"
        params = []
        if trade_mode_filter != 'all':
            query += " AND trade_mode = ?"
            params.append(trade_mode_filter)
        query += " ORDER BY id DESC"

        cursor.execute(query, params)
        active_trades = cursor.fetchall(); conn.close()
        
        if not active_trades:
            return "ظ„ط§ طھظˆط¬ط¯ طµظپظ‚ط§طھ ظ†ط´ط·ط© ط­ط§ظ„ظٹط§ظ‹ ظ„ظ‡ط°ط§ ط§ظ„ظپظ„طھط±.", None
            
        keyboard = [[InlineKeyboardButton(f"#{t['id']} | {t['symbol']} | ${t['entry_value_usdt']:.2f} | {t['exchange']}", callback_data=f"check_{t['id']}")] for t in active_trades]
        return "ط§ط®طھط± طµظپظ‚ط© ظ„ظ…طھط§ط¨ط¹طھظ‡ط§:", InlineKeyboardMarkup(keyboard)
    except Exception as e:
        logger.error(f"Error in show_active_trades: {e}")
        return "ط®ط·ط£ ظپظٹ ط¬ظ„ط¨ ط§ظ„طµظپظ‚ط§طھ.", None

async def execute_manual_trade(exchange_id, symbol, amount_usdt, side, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Attempting MANUAL {side.upper()} for {symbol} on {exchange_id} for ${amount_usdt}")
    exchange = bot_state.exchanges.get(exchange_id.lower())
    if not exchange or not exchange.apiKey:
        return {"success": False, "error": f"ظ„ط§ ظٹظ…ظƒظ† طھظ†ظپظٹط° ط§ظ„ط£ظ…ط±. ظ„ظ… ظٹطھظ… طھظˆط«ظٹظ‚ ط§ظ„ط§طھطµط§ظ„ ط¨ظ…ظ†طµط© {exchange_id.capitalize()}."}

    try:
        ticker = await exchange.fetch_ticker(symbol)
        current_price = ticker.get('last') or ticker.get('close')
        if not current_price:
            return {"success": False, "error": f"ظ„ظ… ط£طھظ…ظƒظ† ظ…ظ† ط¬ظ„ط¨ ط§ظ„ط³ط¹ط± ط§ظ„ط­ط§ظ„ظٹ ظ„ظ€ {symbol}."}

        quantity = float(amount_usdt) / current_price
        formatted_quantity = exchange.amount_to_precision(symbol, quantity)

        order_receipt = None
        if side == 'buy':
            order_receipt = await exchange.create_market_buy_order(symbol, float(formatted_quantity))
        elif side == 'sell':
            order_receipt = await exchange.create_market_sell_order(symbol, float(formatted_quantity))

        await asyncio.sleep(2)
        order = await exchange.fetch_order(order_receipt['id'], symbol)

        logger.info(f"MANUAL ORDER SUCCESS: {order}")

        filled_quantity = order.get('filled', 0)
        filled_price = order.get('average', current_price)
        cost = order.get('cost', 0)

        if not cost and filled_quantity and filled_price:
            cost = filled_quantity * filled_price

        success_message = (
            f"**âœ… طھظ… طھظ†ظپظٹط° ط§ظ„ط£ظ…ط± ط§ظ„ظٹط¯ظˆظٹ ط¨ظ†ط¬ط§ط­**\n\n"
            f"**ط§ظ„ظ…ظ†طµط©:** `{exchange_id.capitalize()}`\n"
            f"**ط§ظ„ط¹ظ…ظ„ط©:** `{symbol}`\n"
            f"**ط§ظ„ظ†ظˆط¹:** `{side.upper()}`\n\n"
            f"--- **طھظپط§طµظٹظ„ ط§ظ„ط£ظ…ط±** ---\n"
            f"**ID:** `{order['id']}`\n"
            f"**ط§ظ„ظƒظ…ظٹط© ط§ظ„ظ…ظ†ظپط°ط©:** `{filled_quantity}`\n"
            f"**ظ…طھظˆط³ط· ط³ط¹ط± ط§ظ„طھظ†ظپظٹط°:** `{filled_price}`\n"
            f"**ط§ظ„طھظƒظ„ظپط© ط§ظ„ط¥ط¬ظ…ط§ظ„ظٹط©:** `${cost:.2f}`"
        )
        return {"success": True, "message": success_message}

    except ccxt.InsufficientFunds as e:
        error_msg = f"â‌Œ ظپط´ظ„: ط±طµظٹط¯ ط؛ظٹط± ظƒط§ظپظچ ط¹ظ„ظ‰ {exchange_id.capitalize()}."
        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}")
        return {"success": False, "error": error_msg}
    except ccxt.InvalidOrder as e:
        error_msg = f"â‌Œ ظپط´ظ„: ط£ظ…ط± ط؛ظٹط± طµط§ظ„ط­. ظ‚ط¯ ظٹظƒظˆظ† ط§ظ„ظ…ط¨ظ„ط؛ ط£ظ‚ظ„ ظ…ظ† ط§ظ„ط­ط¯ ط§ظ„ط£ط¯ظ†ظ‰ ظ„ظ„ظ…ظ†طµط©.\n`{e}`"
        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}")
        return {"success": False, "error": error_msg}
    except ccxt.ExchangeError as e:
        error_msg = f"â‌Œ ظپط´ظ„: ط®ط·ط£ ظ…ظ† ط§ظ„ظ…ظ†طµط©.\n`{e}`"
        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}")
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"â‌Œ ظپط´ظ„: ط­ط¯ط« ط®ط·ط£ ط؛ظٹط± ظ…طھظˆظ‚ط¹.\n`{e}`"
        logger.error(f"MANUAL TRADE FAILED: {error_msg} - {e}", exc_info=True)
        return {"success": False, "error": error_msg}

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data
    user_data = context.user_data

    if data.startswith("dashboard_") and data.endswith(('_all', '_real', '_virtual')):
        if report_lock.locked():
            await query.answer("âڈ³ طھظ‚ط±ظٹط± ط¢ط®ط± ظ‚ظٹط¯ ط§ظ„ط¥ط¹ط¯ط§ط¯طŒ ظٹط±ط¬ظ‰ ط§ظ„ط§ظ†طھط¸ط§ط±...", show_alert=False)
            return
            
        async with report_lock:
            try:
                parts = data.split('_')
                trade_mode_filter = parts[-1]
                report_type = '_'.join(parts[1:-1])

                await query.edit_message_text(f"âڈ³ ط¬ط§ط±ظٹ ط¥ط¹ط¯ط§ط¯ طھظ‚ط±ظٹط± **{report_type.replace('_', ' ').capitalize()}**...", parse_mode=ParseMode.MARKDOWN)

                report_content, keyboard = None, None

                if report_type == "stats":
                    report_content, keyboard = await stats_command(update, context, trade_mode_filter=trade_mode_filter)
                elif report_type == "active_trades":
                    report_content, keyboard = await show_active_trades_command(update, context, trade_mode_filter=trade_mode_filter)
                elif report_type == "strategy_report":
                    report_content, keyboard = await strategy_report_command(update, context, trade_mode_filter=trade_mode_filter)

                if report_content:
                    await query.edit_message_text(text=report_content, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
                else:
                    await query.edit_message_text("â‌Œ ظپط´ظ„ ط¥ط¹ط¯ط§ط¯ ط§ظ„طھظ‚ط±ظٹط±.")

            except Exception as e:
                logger.error(f"Error in dashboard filter handler: {e}", exc_info=True)
                try:
                    await query.edit_message_text("â‌Œ ط­ط¯ط« ط®ط·ط£ ط؛ظٹط± ظ…طھظˆظ‚ط¹ ط£ط«ظ†ط§ط، ط¥ط¹ط¯ط§ط¯ ط§ظ„طھظ‚ط±ظٹط±.")
                except Exception: pass
        return

    if data.startswith("dashboard_"):
        action = data.split("_", 1)[1]
        
        if action in ["stats", "active_trades", "strategy_report"]:
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("ًں“ٹ ط§ظ„ظƒظ„ (ظˆظ‡ظ…ظٹ + ط­ظ‚ظٹظ‚ظٹ)", callback_data=f"dashboard_{action}_all")],
                [InlineKeyboardButton("ًں“ˆ ط­ظ‚ظٹظ‚ظٹ ظپظ‚ط·", callback_data=f"dashboard_{action}_real"), InlineKeyboardButton("ًں“‰ ظˆظ‡ظ…ظٹ ظپظ‚ط·", callback_data=f"dashboard_{action}_virtual")],
                [InlineKeyboardButton("ًں”™ ط§ظ„ط¹ظˆط¯ط©", callback_data="dashboard_refresh")]
            ])
            await query.edit_message_text(f"ط§ط®طھط± ظ†ظˆط¹ ط§ظ„ط³ط¬ظ„ ظ„ط¹ط±ط¶ **{action.replace('_', ' ').capitalize()}**:", reply_markup=keyboard)
            return

        if action == "debug": 
            await query.edit_message_text("âڈ³ ط¬ط§ط±ظٹ ط¥ط¹ط¯ط§ط¯ طھظ‚ط±ظٹط± ط§ظ„طھط´ط®ظٹطµ...", parse_mode=ParseMode.MARKDOWN)
            await debug_command(update, context)
        elif action == "refresh": await show_dashboard_command(update, context)
        elif action == "snapshot": await portfolio_snapshot_command(update, context)
        elif action == "risk": await risk_report_command(update, context)
        elif action == "sync": await sync_portfolio_command(update, context)
        elif action == "tools":
             keyboard = [
                 [InlineKeyboardButton("âœچï¸ڈ طھط¯ط§ظˆظ„ ظٹط¯ظˆظٹ", callback_data="tools_manual_trade"), InlineKeyboardButton("ًں’° ط¹ط±ط¶ ط±طµظٹط¯ظٹ", callback_data="tools_balance")],
                 [InlineKeyboardButton("ًں“– ط£ظˆط§ظ…ط±ظٹ ط§ظ„ظ…ظپطھظˆط­ط©", callback_data="tools_openorders"), InlineKeyboardButton("ًں“œ ط³ط¬ظ„ طھط¯ط§ظˆظ„ط§طھظٹ", callback_data="tools_mytrades")],
                 [InlineKeyboardButton("ًں”™ ط§ظ„ط¹ظˆط¯ط© ظ„ظ„ظˆط­ط© ط§ظ„طھط­ظƒظ…", callback_data="dashboard_refresh")]
             ]
             await query.edit_message_text("ًں› ï¸ڈ *ط£ط¯ظˆط§طھ ط§ظ„طھط¯ط§ظˆظ„*\n\nط§ط®طھط± ط§ظ„ط£ط¯ط§ط© ط§ظ„طھظٹ طھط±ظٹط¯ ط§ط³طھط®ط¯ط§ظ…ظ‡ط§:", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
        return

    elif data.startswith("tools_"):
        tool = data.split("_", 1)[1]
        if tool == "manual_trade": await manual_trade_command(update, context)
        elif tool == "balance": await balance_command(update, context)
        elif tool == "openorders": await open_orders_command(update, context)
        elif tool == "mytrades": await my_trades_command(update, context)
        return

    elif data.startswith("snapshot_exchange_"):
        exchange_id = data.split("_", 2)[2]
        await process_portfolio_snapshot(update, context, exchange_id)
        return

    elif data.startswith("preset_"):
        preset_name = data.split("_", 1)[1]
        if preset_data := PRESETS.get(preset_name):
            bot_state.settings['liquidity_filters'] = preset_data['liquidity_filters']
            bot_state.settings['volatility_filters'] = preset_data['volatility_filters']
            bot_state.settings['ema_trend_filter'] = preset_data['ema_trend_filter']
            bot_state.settings['min_tp_sl_filter'] = preset_data['min_tp_sl_filter']
            bot_state.settings["active_preset_name"] = preset_name
            save_settings()
            preset_titles = {"PRO": "ط§ط­طھط±ط§ظپظٹ", "STRICT": "ظ…طھط´ط¯ط¯", "LAX": "ظ…طھط³ط§ظ‡ظ„", "VERY_LAX": "ظپط§ط¦ظ‚ ط§ظ„طھط³ط§ظ‡ظ„"}
            lf, vf = preset_data['liquidity_filters'], preset_data['volatility_filters']
            confirmation_text = f"âœ… *طھظ… طھظپط¹ظٹظ„ ط§ظ„ظ†ظ…ط·: {preset_titles.get(preset_name, preset_name)}*\n\n*ط£ظ‡ظ… ط§ظ„ظ‚ظٹظ…:*\n`- min_rvol: {lf['min_rvol']}`\n`- max_spread: {lf['max_spread_percent']}%`\n`- min_atr: {vf['min_atr_percent']}%`"
            try: await query.edit_message_text(confirmation_text, parse_mode=ParseMode.MARKDOWN, reply_markup=get_presets_keyboard())
            except BadRequest as e:
                if "Message is not modified" not in str(e): raise
    elif data.startswith("param_"):
        param_key = data.split("_", 1)[1]
        context.user_data['awaiting_input_for_param'] = param_key
        context.user_data['settings_menu_id'] = query.message.message_id
        current_value = bot_state.settings.get(param_key)
        if isinstance(current_value, bool):
            bot_state.settings[param_key] = not current_value
            bot_state.settings["active_preset_name"] = "Custom"; save_settings()
            await query.answer(f"âœ… طھظ… طھط¨ط¯ظٹظ„ '{PARAM_DISPLAY_NAMES.get(param_key, param_key)}'")
            await show_parameters_menu(update, context)
        else:
            await query.edit_message_text(f"ًں“‌ *طھط¹ط¯ظٹظ„ '{PARAM_DISPLAY_NAMES.get(param_key, param_key)}'*\n\n*ط§ظ„ظ‚ظٹظ…ط© ط§ظ„ط­ط§ظ„ظٹط©:* `{current_value}`\n\nط§ظ„ط±ط¬ط§ط، ط¥ط±ط³ط§ظ„ ط§ظ„ظ‚ظٹظ…ط© ط§ظ„ط¬ط¯ظٹط¯ط©.", parse_mode=ParseMode.MARKDOWN)
    elif data.startswith("toggle_scanner_"):
        scanner_name = data.split("_", 2)[2]
        active_scanners = bot_state.settings.get("active_scanners", []).copy()
        if scanner_name in active_scanners: active_scanners.remove(scanner_name)
        else: active_scanners.append(scanner_name)
        bot_state.settings["active_scanners"] = active_scanners; save_settings()
        try: await query.edit_message_text(text="ط§ط®طھط± ط§ظ„ظ…ط§ط³ط­ط§طھ ظ„طھظپط¹ظٹظ„ظ‡ط§ ط£ظˆ طھط¹ط·ظٹظ„ظ‡ط§:", reply_markup=get_scanners_keyboard())
        except BadRequest as e:
            if "Message is not modified" not in str(e): raise
    elif data.startswith("toggle_real_trade_"):
        exchange_id = data.split("_", 3)[3]
        settings = bot_state.settings.get("real_trading_per_exchange", {})
        settings[exchange_id] = not settings.get(exchange_id, False)
        bot_state.settings["real_trading_per_exchange"] = settings
        save_settings()
        await query.answer(f"طھظ… {'طھظپط¹ظٹظ„' if settings[exchange_id] else 'طھط¹ط·ظٹظ„'} ط§ظ„طھط¯ط§ظˆظ„ ط¹ظ„ظ‰ {exchange_id.capitalize()}")
        await show_real_trading_control_menu(update, context)
        if query.message: await query.message.delete()
        return

    elif data == "back_to_settings":
        if query.message: await query.message.delete()
        await context.bot.send_message(chat_id=query.message.chat_id, text="ط§ط®طھط± ط§ظ„ط¥ط¹ط¯ط§ط¯:", reply_markup=ReplyKeyboardMarkup(settings_menu_keyboard, resize_keyboard=True))
    elif data.startswith("check_"):
        await check_trade_command(update, context, trade_id_from_callback=int(data.split("_")[1]))

    elif data.startswith("suggest_"):
        action = data.split("_", 1)[1]
        if action.startswith("accept"):
            preset_name = data.split("_")[2]
            if preset_data := PRESETS.get(preset_name):
                bot_state.settings['liquidity_filters'] = preset_data['liquidity_filters']
                bot_state.settings['volatility_filters'] = preset_data['volatility_filters']
                bot_state.settings['ema_trend_filter'] = preset_data['ema_trend_filter']
                bot_state.settings['min_tp_sl_filter'] = preset_data['min_tp_sl_filter']
                bot_state.settings["active_preset_name"] = preset_name
                save_settings()
                await query.edit_message_text(f"âœ… **طھظ… ظ‚ط¨ظˆظ„ ط§ظ„ط§ظ‚طھط±ط§ط­!**\n\nطھظ… طھط؛ظٹظٹط± ط§ظ„ظ†ظ…ط· ط¨ظ†ط¬ط§ط­ ط¥ظ„ظ‰ `{preset_name}`.", parse_mode=ParseMode.MARKDOWN)
        elif action == "decline":
            await query.edit_message_text("ًں‘چ **طھظ… طھط¬ط§ظ‡ظ„ ط§ظ„ط§ظ‚طھط±ط§ط­.**\n\nط³ظٹط³طھظ…ط± ط§ظ„ط¨ظˆطھ ط¨ط§ظ„ط¹ظ…ظ„ ط¹ظ„ظ‰ ط§ظ„ط¥ط¹ط¯ط§ط¯ط§طھ ط§ظ„ط­ط§ظ„ظٹط©.", parse_mode=ParseMode.MARKDOWN)

async def manual_trade_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data
    user_data = context.user_data

    if 'manual_trade' not in user_data:
        await query.edit_message_text("âڑ ï¸ڈ ط§ظ†طھظ‡طھ ظ‡ط°ظ‡ ط§ظ„ط¬ظ„ط³ط©. ط§ط¨ط¯ط£ ظ…ظ† ط¬ط¯ظٹط¯ ط¨ط§ط³طھط®ط¯ط§ظ… /trade.")
        return

    state = user_data['manual_trade'].get('state')

    if data == "manual_trade_cancel":
        user_data.pop('manual_trade', None)
        await query.edit_message_text("ًں‘چ طھظ… ط¥ظ„ط؛ط§ط، ط¹ظ…ظ„ظٹط© ط§ظ„طھط¯ط§ظˆظ„ ط§ظ„ظٹط¯ظˆظٹ.")
        return

    if state == 'awaiting_exchange':
        exchange = data.split("_")[-1]
        user_data['manual_trade']['exchange'] = exchange
        user_data['manual_trade']['state'] = 'awaiting_symbol'
        await query.edit_message_text(f"ط§ط®طھط±طھ ظ…ظ†طµط©: *{exchange.capitalize()}*\n\nط§ظ„ط¢ظ†طŒ ط£ط±ط³ظ„ ط±ظ…ط² ط§ظ„ط¹ظ…ظ„ط© (ظ…ط«ط§ظ„: `BTC/USDT`).", parse_mode=ParseMode.MARKDOWN)

    elif state == 'awaiting_side':
        side = data.split("_")[-1]
        user_data['manual_trade']['side'] = side
        user_data['manual_trade']['state'] = 'confirming'

        trade_data = user_data['manual_trade']
        await query.edit_message_text("âڈ³ ط¬ط§ط±ظٹ طھظ†ظپظٹط° ط§ظ„ط£ظ…ط±...", reply_markup=None)

        result = await execute_manual_trade(
            exchange_id=trade_data['exchange'],
            symbol=trade_data['symbol'],
            amount_usdt=trade_data['amount'],
            side=trade_data['side'],
            context=context
        )

        if result['success']:
            await query.edit_message_text(result['message'], parse_mode=ParseMode.MARKDOWN)
        else:
            await query.edit_message_text(result['error'], parse_mode=ParseMode.MARKDOWN)

        user_data.pop('manual_trade', None)

async def tools_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    user_data = context.user_data
    tool_name, action, value = data.split("_", 2)

    tool_key = f"{tool_name}_tool"
    user_data[tool_key] = {} # Reset or initialize the tool session
    if action == "exchange":
        user_data[tool_key]['exchange'] = value
        if tool_name == "balance":
            await query.edit_message_text(f"ًں’° ط¬ط§ط±ظٹ ط¬ظ„ط¨ ط§ظ„ط£ط±طµط¯ط© ظ…ظ† *{value.capitalize()}*...", parse_mode=ParseMode.MARKDOWN)
            await fetch_and_display_balance(value, query)
            user_data.pop(tool_key, None)
        else:
            user_data[tool_key]['state'] = 'awaiting_symbol'
            await query.edit_message_text(f"ط§ط®طھط±طھ ظ…ظ†طµط©: *{value.capitalize()}*\n\nط§ظ„ط¢ظ†طŒ ط£ط±ط³ظ„ ط±ظ…ط² ط§ظ„ط¹ظ…ظ„ط© (ظ…ط«ط§ظ„: `BTC/USDT`)\nط£ظˆ ط£ط±ط³ظ„ `ط§ظ„ظƒظ„` ظ„ط¹ط±ط¶ ط§ظ„ط¨ظٹط§ظ†ط§طھ ظ„ط¬ظ…ظٹط¹ ط§ظ„ط¹ظ…ظ„ط§طھ.", parse_mode=ParseMode.MARKDOWN)

async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_data = context.user_data
    text = update.message.text

    active_tool = None
    for tool_key in ['openorders_tool', 'mytrades_tool', 'manual_trade']:
        if tool_key in user_data:
            active_tool = tool_key
            break

    if active_tool:
        state = user_data[active_tool].get('state')
        if state == 'awaiting_symbol':
            symbol = text.upper()
            exchange_id = user_data[active_tool]['exchange']

            if symbol.lower() in ["all", "ط§ظ„ظƒظ„"]:
                symbol = None
            elif '/' not in symbol:
                await update.message.reply_text("â‌Œ ط±ظ…ط² ط؛ظٹط± طµط§ظ„ط­. ط§ظ„ط±ط¬ط§ط، ط¥ط±ط³ط§ظ„ ط§ظ„ط±ظ…ط² ط¨ط§ظ„طھظ†ط³ظٹظ‚ ط§ظ„طµط­ظٹط­ (ظ…ط«ط§ظ„: `BTC/USDT`) ط£ظˆ ظƒظ„ظ…ط© `ط§ظ„ظƒظ„`.")
                return

            if active_tool == 'openorders_tool':
                await update.message.reply_text(f"ًں“– ط¬ط§ط±ظٹ ط¬ظ„ط¨ ط£ظˆط§ظ…ط±ظƒ ط§ظ„ظ…ظپطھظˆط­ط© ظ„ظ€ *{symbol or 'ط§ظ„ظƒظ„'}*...", parse_mode=ParseMode.MARKDOWN)
                await fetch_and_display_open_orders(exchange_id, symbol, update.message)
            elif active_tool == 'mytrades_tool':
                await update.message.reply_text(f"ًں“œ ط¬ط§ط±ظٹ ط¬ظ„ط¨ ط³ط¬ظ„ طھط¯ط§ظˆظ„ط§طھظƒ ظ„ظ€ *{symbol or 'ط§ظ„ظƒظ„'}*...", parse_mode=ParseMode.MARKDOWN)
                await fetch_and_display_my_trades(exchange_id, symbol, update.message)
            elif active_tool == 'manual_trade':
                 user_data['manual_trade']['symbol'] = symbol
                 user_data['manual_trade']['state'] = 'awaiting_amount'
                 await update.message.reply_text(f"ط±ظ…ط² ط§ظ„ط¹ظ…ظ„ط©: *{symbol}*\n\nط§ظ„ط¢ظ†طŒ ط£ط¯ط®ظ„ ط§ظ„ظ…ط¨ظ„ط؛ ط¨ظ€ USDT (ظ…ط«ط§ظ„: `15`).", parse_mode=ParseMode.MARKDOWN)

            if active_tool != 'manual_trade':
                user_data.pop(active_tool, None)
            return

        elif active_tool == 'manual_trade' and state == 'awaiting_amount':
            try:
                amount = float(text)
                if amount <= 0: raise ValueError("Amount must be positive")
                user_data['manual_trade']['amount'] = amount
                user_data['manual_trade']['state'] = 'awaiting_side'
                keyboard = [
                    [InlineKeyboardButton("ًں“ˆ ط´ط±ط§ط، (Buy)", callback_data="manual_trade_side_buy"),
                     InlineKeyboardButton("ًں“‰ ط¨ظٹط¹ (Sell)", callback_data="manual_trade_side_sell")],
                    [InlineKeyboardButton("â‌Œ ط¥ظ„ط؛ط§ط،", callback_data="manual_trade_cancel")]
                ]
                await update.message.reply_text(f"ط§ظ„ظ…ط¨ظ„ط؛: *${amount}*\n\nط§ط®طھط± ظ†ظˆط¹ ط§ظ„ط£ظ…ط±:", reply_markup=InlineKeyboardMarkup(keyboard))
            except ValueError:
                await update.message.reply_text("â‌Œ ظ…ط¨ظ„ط؛ ط؛ظٹط± طµط§ظ„ط­. ط§ظ„ط±ط¬ط§ط، ط¥ط±ط³ط§ظ„ ط±ظ‚ظ… ظپظ‚ط· (ظ…ط«ط§ظ„: `15` ط£ظˆ `20.5`).")
            return

    menu_handlers = {
        "Dashboard ًں–¥ï¸ڈ": show_dashboard_command,
        "â„¹ï¸ڈ ظ…ط³ط§ط¹ط¯ط©": help_command,
        "âڑ™ï¸ڈ ط§ظ„ط¥ط¹ط¯ط§ط¯ط§طھ": show_settings_menu,
        "ًں”§ طھط¹ط¯ظٹظ„ ط§ظ„ظ…ط¹ط§ظٹظٹط±": show_parameters_menu,
        "ًں”™ ط§ظ„ظ‚ط§ط¦ظ…ط© ط§ظ„ط±ط¦ظٹط³ظٹط©": start_command,
        "ًںژ­ طھظپط¹ظٹظ„/طھط¹ط·ظٹظ„ ط§ظ„ظ…ط§ط³ط­ط§طھ": show_scanners_menu,
        "ًںڈپ ط£ظ†ظ…ط§ط· ط¬ط§ظ‡ط²ط©": show_presets_menu,
        "ًںڑ¨ ط§ظ„طھط­ظƒظ… ط¨ط§ظ„طھط¯ط§ظˆظ„ ط§ظ„ط­ظ‚ظٹظ‚ظٹ": show_real_trading_control_menu,
    }
    if text in menu_handlers:
        for key in list(user_data.keys()):
            if key.startswith(('manual_trade', 'openorders_tool', 'mytrades_tool', 'balance_tool')) or key == 'awaiting_input_for_param':
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
            current_type = type(settings.get(param, ''))
            new_value = current_type(value_str)
            if isinstance(settings.get(param), bool):
                new_value = value_str.lower() in ['true', '1', 'yes', 'on', 'ظ†ط¹ظ…', 'طھظپط¹ظٹظ„']
            settings[param] = new_value
            settings["active_preset_name"] = "Custom"
            save_settings()
            if settings_menu_id: context.user_data['settings_menu_id'] = settings_menu_id
            await show_parameters_menu(update, context)
            confirm_msg = await update.message.reply_text(f"âœ… طھظ… طھط­ط¯ظٹط« **{PARAM_DISPLAY_NAMES.get(param, param)}** ط¥ظ„ظ‰ `{new_value}`.", parse_mode=ParseMode.MARKDOWN)
            context.job_queue.run_once(lambda ctx: ctx.bot.delete_message(chat_id, confirm_msg.message_id), 4)
        except (ValueError, KeyError):
            if settings_menu_id:
                await context.bot.edit_message_text(chat_id=chat_id, message_id=settings_menu_id, text="â‌Œ ظ‚ظٹظ…ط© ط؛ظٹط± طµط§ظ„ط­ط©. ط§ظ„ط±ط¬ط§ط، ط§ظ„ظ…ط­ط§ظˆظ„ط© ظ…ط±ط© ط£ط®ط±ظ‰.")
                context.job_queue.run_once(lambda _: show_parameters_menu(update, context), 3)
        return

async def manual_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['manual_trade'] = {'state': 'awaiting_exchange'}
    keyboard = [
        [InlineKeyboardButton("Binance", callback_data="manual_trade_exchange_binance"),
         InlineKeyboardButton("KuCoin", callback_data="manual_trade_exchange_kucoin")],
        [InlineKeyboardButton("â‌Œ ط¥ظ„ط؛ط§ط،", callback_data="manual_trade_cancel")]
    ]

    message_text = "âœچï¸ڈ **ط¨ط¯ط، طھط¯ط§ظˆظ„ ظٹط¯ظˆظٹ**\n\nط§ط®طھط± ط§ظ„ظ…ظ†طµط© ط§ظ„طھظٹ طھط±ظٹط¯ طھظ†ظپظٹط° ط§ظ„ط£ظ…ط± ط¹ظ„ظٹظ‡ط§:"
    if update.callback_query:
        await update.callback_query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard))

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['balance_tool'] = {'state': 'awaiting_exchange'}
    keyboard = [
        [InlineKeyboardButton("Binance", callback_data="balance_exchange_binance"),
         InlineKeyboardButton("KuCoin", callback_data="balance_exchange_kucoin")],
        [InlineKeyboardButton("ًں”™ ط§ظ„ط¹ظˆط¯ط© ظ„ظ„ط£ط¯ظˆط§طھ", callback_data="dashboard_tools")]
    ]
    await update.callback_query.edit_message_text("ًں’° **ط¹ط±ط¶ ط§ظ„ط±طµظٹط¯**\n\nط§ط®طھط± ط§ظ„ظ…ظ†طµط© ظ„ط¹ط±ط¶ ط£ط±طµط¯طھظƒ:", reply_markup=InlineKeyboardMarkup(keyboard))

async def open_orders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['openorders_tool'] = {'state': 'awaiting_exchange'}
    keyboard = [
        [InlineKeyboardButton("Binance", callback_data="openorders_exchange_binance"),
         InlineKeyboardButton("KuCoin", callback_data="openorders_exchange_kucoin")],
        [InlineKeyboardButton("ًں”™ ط§ظ„ط¹ظˆط¯ط© ظ„ظ„ط£ط¯ظˆط§طھ", callback_data="dashboard_tools")]
    ]
    await update.callback_query.edit_message_text("ًں“– **ط£ظˆط§ظ…ط±ظٹ ط§ظ„ظ…ظپطھظˆط­ط©**\n\nط§ط®طھط± ط§ظ„ظ…ظ†طµط©:", reply_markup=InlineKeyboardMarkup(keyboard))

async def my_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mytrades_tool'] = {'state': 'awaiting_exchange'}
    keyboard = [
        [InlineKeyboardButton("Binance", callback_data="mytrades_exchange_binance"),
         InlineKeyboardButton("KuCoin", callback_data="mytrades_exchange_kucoin")],
        [InlineKeyboardButton("ًں”™ ط§ظ„ط¹ظˆط¯ط© ظ„ظ„ط£ط¯ظˆط§طھ", callback_data="dashboard_tools")]
    ]
    await update.callback_query.edit_message_text("ًں“œ **ط³ط¬ظ„ طھط¯ط§ظˆظ„ط§طھظٹ**\n\nط§ط®طھط± ط§ظ„ظ…ظ†طµط©:", reply_markup=InlineKeyboardMarkup(keyboard))

async def fetch_and_display_balance(exchange_id, query):
    exchange = bot_state.exchanges.get(exchange_id.lower())
    if not exchange or not exchange.apiKey:
        await query.edit_message_text(f"â‌Œ ط®ط·ط£: ظ„ظ… ظٹطھظ… طھظˆط«ظٹظ‚ ط§ظ„ط§طھطµط§ظ„ ط¨ظ…ظ†طµط© {exchange_id.capitalize()}.")
        return

    try:
        balance = await exchange.fetch_balance()
        total_balance = balance.get('total', {})

        public_exchange = bot_state.public_exchanges.get(exchange_id.lower())
        tickers = await public_exchange.fetch_tickers()

        assets = []
        for currency, amount in total_balance.items():
            if amount > 0:
                usdt_value = 0
                if currency == 'USDT':
                    usdt_value = amount
                elif f"{currency}/USDT" in tickers:
                    usdt_value = amount * tickers[f"{currency}/USDT"]['last']

                if usdt_value > 1:
                    assets.append({'currency': currency, 'amount': amount, 'usdt_value': usdt_value})

        assets.sort(key=lambda x: x['usdt_value'], reverse=True)

        if not assets:
            await query.edit_message_text(f"â„¹ï¸ڈ ظ„ط§ طھظˆط¬ط¯ ط£ط±طµط¯ط© ظƒط¨ظٹط±ط© (> $1) ط¹ظ„ظ‰ ظ…ظ†طµط© {exchange_id.capitalize()}.")
            return

        message_lines = [f"**ًں’° ط±طµظٹط¯ظƒ ط¹ظ„ظ‰ {exchange_id.capitalize()}**\n"]
        total_usdt_value = sum(a['usdt_value'] for a in assets)
        message_lines.append(f"__**ط¥ط¬ظ…ط§ظ„ظٹ ط§ظ„ظ‚ظٹظ…ط© ط§ظ„طھظ‚ط¯ظٹط±ظٹط©:**__ `${total_usdt_value:,.2f}`\n")

        for asset in assets[:15]:
            message_lines.append(f"- `{asset['currency']}`: `{asset['amount']:.4f}` (~`${asset['usdt_value']:.2f}`)")

        await query.edit_message_text("\n".join(message_lines), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error fetching balance for {exchange_id}: {e}")
        await query.edit_message_text(f"â‌Œ ط­ط¯ط« ط®ط·ط£ ط£ط«ظ†ط§ط، ط¬ظ„ط¨ ط§ظ„ط±طµظٹط¯ ظ…ظ† {exchange_id.capitalize()}.")

async def fetch_and_display_open_orders(exchange_id, symbol, message):
    exchange = bot_state.exchanges.get(exchange_id.lower())
    if not exchange or not exchange.apiKey:
        await message.reply_text(f"â‌Œ ط®ط·ط£: ظ„ظ… ظٹطھظ… طھظˆط«ظٹظ‚ ط§ظ„ط§طھطµط§ظ„ ط¨ظ…ظ†طµط© {exchange_id.capitalize()}.")
        return
    try:
        open_orders = await exchange.fetch_open_orders(symbol)

        if not open_orders:
            await message.reply_text(f"âœ… ظ„ط§ طھظˆط¬ط¯ ظ„ط¯ظٹظƒ ط£ظˆط§ظ…ط± ظ…ظپطھظˆط­ط© ظ„ظ€ `{symbol or 'ط§ظ„ظƒظ„'}` ط¹ظ„ظ‰ {exchange_id.capitalize()}.")
            return

        lines = [f"**ًں“– ط£ظˆط§ظ…ط±ظƒ ط§ظ„ظ…ظپطھظˆط­ط© ظ„ظ€ `{symbol or 'ط§ظ„ظƒظ„'}` ط¹ظ„ظ‰ {exchange_id.capitalize()}**\n"]
        for order in open_orders:
            side_emoji = "ًں”¼" if order['side'] == 'buy' else "ًں”½"
            lines.append(
                f"`{order['symbol']}` {side_emoji} `{order['side'].upper()}`\n"
                f"  - **ط§ظ„ظƒظ…ظٹط©:** `{order['amount']}`\n"
                f"  - **ط§ظ„ط³ط¹ط±:** `{order['price']}`\n"
                f"  - **ط§ظ„ظ†ظˆط¹:** `{order['type']}`"
            )

        await message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error fetching open orders for {symbol} on {exchange_id}: {e}")
        await message.reply_text(f"â‌Œ ظپط´ظ„ ط¬ظ„ط¨ ط§ظ„ط£ظˆط§ظ…ط± ط§ظ„ظ…ظپطھظˆط­ط©. طھط£ظƒط¯ ظ…ظ† طµط­ط© ط§ظ„ط±ظ…ط²: `{symbol or ''}`.")

async def fetch_and_display_my_trades(exchange_id, symbol, message):
    exchange = bot_state.exchanges.get(exchange_id.lower())
    if not exchange or not exchange.apiKey:
        await message.reply_text(f"â‌Œ ط®ط·ط£: ظ„ظ… ظٹطھظ… طھظˆط«ظٹظ‚ ط§ظ„ط§طھطµط§ظ„ ط¨ظ…ظ†طµط© {exchange_id.capitalize()}.")
        return
    try:
        my_trades = await exchange.fetch_my_trades(symbol, limit=20)

        if not my_trades:
            await message.reply_text(f"âœ… ظ„ط§ ظٹظˆط¬ط¯ ظ„ط¯ظٹظƒ ط³ط¬ظ„ طھط¯ط§ظˆظ„ ظ„ظ€ `{symbol or 'ط§ظ„ظƒظ„'}` ط¹ظ„ظ‰ {exchange_id.capitalize()}.")
            return

        lines = [f"**ًں“œ ط¢ط®ط± 20 ظ…ظ† طھط¯ط§ظˆظ„ط§طھظƒ ظ„ظ€ `{symbol or 'ط§ظ„ظƒظ„'}` ط¹ظ„ظ‰ {exchange_id.capitalize()}**\n"]

        for trade in reversed(my_trades):
            trade_time = datetime.fromtimestamp(trade['timestamp'] / 1000, tz=EGYPT_TZ).strftime('%Y-%m-%d %H:%M')
            side_emoji = "ًں”¼" if trade['side'] == 'buy' else "ًں”½"
            fee = trade.get('fee', {})
            fee_str = f"{fee.get('cost', 0):.4f} {fee.get('currency', '')}"
            lines.append(
                f"`{trade_time}` | `{trade['symbol']}` {side_emoji} `{trade['side'].upper()}`\n"
                f"  - **ط§ظ„ظƒظ…ظٹط©:** `{trade['amount']}`\n"
                f"  - **ط§ظ„ط³ط¹ط±:** `{trade['price']}`\n"
                f"  - **ط§ظ„ط±ط³ظˆظ…:** `{fee_str}`"
            )

        await message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Error fetching my trades for {symbol} on {exchange_id}: {e}")
        await message.reply_text(f"â‌Œ ظپط´ظ„ ط¬ظ„ط¨ ط³ط¬ظ„ طھط¯ط§ظˆظ„ط§طھظƒ. طھط£ظƒط¯ ظ…ظ† طµط­ط© ط§ظ„ط±ظ…ط²: `{symbol or ''}`.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None: logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

async def get_total_real_portfolio_value_usdt():
    total_usdt_value = 0
    for exchange in bot_state.exchanges.values():
        if not exchange.apiKey:
            continue
        try:
            balance = await exchange.fetch_balance()
            if not hasattr(exchange, '_tickers_cache') or (time.time() - exchange._tickers_cache_time > 60):
                 exchange._tickers_cache = await exchange.fetch_tickers()
                 exchange._tickers_cache_time = time.time()
            
            tickers = exchange._tickers_cache

            for currency, amount in balance.get('total', {}).items():
                if amount > 0:
                    usdt_value = 0
                    if currency == 'USDT':
                        usdt_value = amount
                    elif f"{currency}/USDT" in tickers and tickers[f"{currency}/USDT"].get('last'):
                        usdt_value = amount * tickers[f"{currency}/USDT"]['last']
                    
                    if usdt_value > 0.1:
                        total_usdt_value += usdt_value
        except Exception as e:
            logger.error(f"Could not calculate real portfolio value for {exchange.id}: {e}")
    return total_usdt_value

async def portfolio_snapshot_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message
    
    connected_exchanges = [ex for ex in bot_state.exchanges.values() if ex.apiKey]
    
    if not connected_exchanges:
        await target_message.edit_text("â‌Œ **ظپط´ظ„:** ظ„ظ… ظٹطھظ… ط§ظ„ط¹ط«ظˆط± ط¹ظ„ظ‰ ط£ظٹ ظ…ظ†طµط© ظ…طھطµظ„ط© ط¨ط­ط³ط§ط¨ ط­ظ‚ظٹظ‚ظٹ.")
        return

    if len(connected_exchanges) == 1:
        await process_portfolio_snapshot(update, context, connected_exchanges[0].id)
    else:
        keyboard = []
        for ex in connected_exchanges:
            keyboard.append([InlineKeyboardButton(f"ًں“¸ {ex.id.capitalize()}", callback_data=f"snapshot_exchange_{ex.id}")])
        keyboard.append([InlineKeyboardButton("ًں”™ ط§ظ„ط¹ظˆط¯ط© ظ„ظ„ظˆط­ط© ط§ظ„طھط­ظƒظ…", callback_data="dashboard_refresh")])
        
        await target_message.edit_text(
            "**ًں“¸ ظ„ظ‚ط·ط© ظ„ظ„ظ…ط­ظپط¸ط©**\n\nظ„ط¯ظٹظƒ ط£ظƒط«ط± ظ…ظ† ظ…ظ†طµط© ظ…طھطµظ„ط©. ط§ط®طھط± ط§ظ„ظ…ظ†طµط©:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

async def process_portfolio_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE, exchange_id: str):
    target_message = update.callback_query.message
    await target_message.edit_text(f"ًں“¸ **ظ„ظ‚ط·ط© ظ„ظ„ظ…ط­ظپط¸ط©**\n\nâڈ³ ط¬ط§ط±ظگ ط§ظ„ط§طھطµط§ظ„ ط¨ظ…ظ†طµط© {exchange_id.capitalize()} ظˆط¬ظ„ط¨ ط§ظ„ط¨ظٹط§ظ†ط§طھ...")

    exchange = bot_state.exchanges.get(exchange_id)
    if not exchange:
        await target_message.edit_text(f"â‌Œ **ظپط´ظ„:** ط®ط·ط£ ظپظٹ ط§ظ„ط¹ط«ظˆط± ط¹ظ„ظ‰ ظ…ظ†طµط© {exchange_id.capitalize()} ط§ظ„ظ…طھطµظ„ط©.")
        return

    try:
        balance = await exchange.fetch_balance()
        all_assets = balance.get('total', {})
        tickers = await exchange.fetch_tickers()
        
        portfolio_assets = []
        total_usdt_value = 0
        for currency, amount in all_assets.items():
            if amount > 0:
                usdt_value = 0
                if currency == 'USDT':
                    usdt_value = amount
                elif f"{currency}/USDT" in tickers and tickers[f"{currency}/USDT"].get('last'):
                    usdt_value = amount * tickers[f"{currency}/USDT"]['last']
                
                if usdt_value > 1:
                    portfolio_assets.append({'currency': currency, 'amount': amount, 'usdt_value': usdt_value})
                    total_usdt_value += usdt_value
        
        portfolio_assets.sort(key=lambda x: x['usdt_value'], reverse=True)
        
        all_recent_trades = []
        for asset in portfolio_assets:
            try:
                symbol = f"{asset['currency']}/USDT"
                if symbol in exchange.markets:
                    trades = await exchange.fetch_my_trades(symbol=symbol, limit=5)
                    all_recent_trades.extend(trades)
            except Exception as e:
                logger.warning(f"Could not fetch trades for {asset['currency']}: {e}")
        
        all_recent_trades.sort(key=lambda x: x['timestamp'], reverse=True)
        recent_trades = all_recent_trades[:20]

        parts = [f"**ًں“¸ ظ„ظ‚ط·ط© ظ„ظ…ط­ظپط¸ط© {exchange.id.capitalize()}**\n"]
        parts.append(f"__**ط¥ط¬ظ…ط§ظ„ظٹ ط§ظ„ظ‚ظٹظ…ط© ط§ظ„طھظ‚ط¯ظٹط±ظٹط©:**__ `${total_usdt_value:,.2f}`\n")

        parts.append("--- **ط§ظ„ط£ط±طµط¯ط© ط§ظ„ط­ط§ظ„ظٹط© (> $1)** ---")
        for asset in portfolio_assets[:15]:
            parts.append(f"- **{asset['currency']}**: `{asset['amount']:.4f}` *~`${asset['usdt_value']:.2f}`*")
        
        parts.append("\n--- **ط¢ط®ط± 20 ط¹ظ…ظ„ظٹط© طھط¯ط§ظˆظ„** ---")
        if not recent_trades:
            parts.append("ظ„ط§ ظٹظˆط¬ط¯ ط³ط¬ظ„ طھط¯ط§ظˆظ„ط§طھ ط­ط¯ظٹط«.")
        else:
            for trade in reversed(recent_trades): 
                side_emoji = "ًںں¢" if trade['side'] == 'buy' else "ًں”´"
                parts.append(f"`{trade['symbol']}` {side_emoji} `{trade['side'].upper()}` | ط§ظ„ظƒظ…ظٹط©: `{trade['amount']}` | ط§ظ„ط³ط¹ط±: `{trade['price']}`")
        
        await target_message.edit_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error generating portfolio snapshot: {e}", exc_info=True)
        await target_message.edit_text(f"â‌Œ **ظپط´ظ„:** ط­ط¯ط« ط®ط·ط£ ط£ط«ظ†ط§ط، ط¬ظ„ط¨ ط¨ظٹط§ظ†ط§طھ ط§ظ„ظ…ط­ظپط¸ط©.\n`{e}`")

async def risk_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message
    await target_message.edit_text("دپخ¯رپذ؛ **طھظ‚ط±ظٹط± ط§ظ„ظ…ط®ط§ط·ط±**\n\nâڈ³ ط¬ط§ط±ظگ طھط­ظ„ظٹظ„ ط§ظ„طµظپظ‚ط§طھ ط§ظ„ظ†ط´ط·ط©...")

    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.row_factory = sqlite3.Row
        
        real_trades = conn.cursor().execute("SELECT * FROM trades WHERE status = 'ظ†ط´ط·ط©' AND trade_mode = 'real'").fetchall()
        virtual_trades = conn.cursor().execute("SELECT * FROM trades WHERE status = 'ظ†ط´ط·ط©' AND trade_mode = 'virtual'").fetchall()
        conn.close()

        parts = ["**دپخ¯رپذ؛ طھظ‚ط±ظٹط± ط§ظ„ظ…ط®ط§ط·ط± ط§ظ„ط­ط§ظ„ظٹ**\n"]

        def generate_risk_section(title, trades, portfolio_value):
            if not trades:
                return [f"\n--- **{title}** ---\nâœ… ظ„ط§ طھظˆط¬ط¯ طµظپظ‚ط§طھ ظ†ط´ط·ط© ط­ط§ظ„ظٹط§ظ‹."]
            
            valid_trades = [t for t in trades if all(t.get(k) is not None for k in ['entry_value_usdt', 'entry_price', 'stop_loss', 'quantity'])]
            
            total_at_risk = sum(t['entry_value_usdt'] for t in valid_trades)
            potential_loss = sum((t['entry_price'] - t['stop_loss']) * t['quantity'] for t in valid_trades)
            symbol_concentration = Counter(t['symbol'] for t in valid_trades)

            section_parts = [f"\n--- **{title}** ---"]
            section_parts.append(f"- **ط¹ط¯ط¯ ط§ظ„طµظپظ‚ط§طھ:** {len(valid_trades)}")
            section_parts.append(f"- **ط¥ط¬ظ…ط§ظ„ظٹ ط±ط£ط³ ط§ظ„ظ…ط§ظ„ ط¨ط§ظ„طµظپظ‚ط§طھ:** `${total_at_risk:,.2f}`")
            if portfolio_value > 0:
                section_parts.append(f"- **ظ†ط³ط¨ط© ط§ظ„طھط¹ط±ط¶:** `{(total_at_risk / portfolio_value) * 100:.2f}%` ظ…ظ† ط§ظ„ظ…ط­ظپط¸ط©")
            section_parts.append(f"- **ط£ظ‚طµظ‰ ط®ط³ط§ط±ط© ظ…ط­طھظ…ظ„ط©:** `$-{potential_loss:,.2f}` (ط¥ط°ط§ ط¶ظڈط±ط¨ ظƒظ„ ط§ظ„ظˆظ‚ظپ)")
            
            if symbol_concentration:
                most_common = symbol_concentration.most_common(1)[0]
                section_parts.append(f"- **ط§ظ„ط¹ظ…ظ„ط© ط§ظ„ط£ظƒط«ط± طھط±ظƒظٹط²ط§ظ‹:** `{most_common[0]}` ({most_common[1]} طµظپظ‚ط§طھ)")
            
            return section_parts

        exchange = next((ex for ex in bot_state.exchanges.values() if ex.apiKey), None)
        real_portfolio_value = 0
        if exchange:
            real_portfolio_value = await get_real_balance(exchange.id, 'USDT')
        parts.extend(generate_risk_section("ًںڑ¨ ط§ظ„ظ…ط®ط§ط·ط± ط§ظ„ط­ظ‚ظٹظ‚ظٹط©", real_trades, real_portfolio_value))
        
        virtual_portfolio_value = bot_state.settings['virtual_portfolio_balance_usdt']
        parts.extend(generate_risk_section("ًں“ٹ ط§ظ„ظ…ط®ط§ط·ط± ط§ظ„ظˆظ‡ظ…ظٹط©", virtual_trades, virtual_portfolio_value))

        await target_message.edit_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error generating risk report: {e}", exc_info=True)
        await target_message.edit_text(f"â‌Œ **ظپط´ظ„:** ط­ط¯ط« ط®ط·ط£ ط£ط«ظ†ط§ط، ط¥ط¹ط¯ط§ط¯ طھظ‚ط±ظٹط± ط§ظ„ظ…ط®ط§ط·ط±.\n`{e}`")

async def sync_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_message = update.callback_query.message
    await target_message.edit_text("ًں”„ **ظ…ط²ط§ظ…ظ†ط© ظˆظ…ط·ط§ط¨ظ‚ط© ط§ظ„ظ…ط­ظپط¸ط©**\n\nâڈ³ ط¬ط§ط±ظگ ط§ظ„ط§طھطµط§ظ„ ط¨ط§ظ„ظ…ظ†طµط© ظˆظ…ظ‚ط§ط±ظ†ط© ط§ظ„ط¨ظٹط§ظ†ط§طھ...")

    exchange = next((ex for ex in bot_state.exchanges.values() if ex.apiKey), None)
    if not exchange:
        await target_message.edit_text("â‌Œ **ظپط´ظ„:** ظ„ظ… ظٹطھظ… ط§ظ„ط¹ط«ظˆط± ط¹ظ„ظ‰ ط£ظٹ ظ…ظ†طµط© ظ…طھطµظ„ط© ط¨ط­ط³ط§ط¨ ط­ظ‚ظٹظ‚ظٹ.")
        return
        
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        bot_trades_raw = conn.cursor().execute("SELECT symbol FROM trades WHERE status = 'ظ†ط´ط·ط©' AND trade_mode = 'real'").fetchall()
        bot_symbols = {item[0] for item in bot_trades_raw}
        conn.close()

        balance = await exchange.fetch_balance()
        exchange_symbols = set()
        for currency, amount in balance.get('total', {}).items():
            if amount > 0 and f"{currency}/USDT" in exchange.markets:
                exchange_symbols.add(f"{currency}/USDT")

        matched_symbols = bot_symbols.intersection(exchange_symbols)
        bot_only_symbols = bot_symbols.difference(exchange_symbols)
        exchange_only_symbols = exchange_symbols.difference(bot_symbols)

        parts = [f"**ًں”„ طھظ‚ط±ظٹط± ظ…ط²ط§ظ…ظ†ط© ط§ظ„ظ…ط­ظپط¸ط© ({exchange.id.capitalize()})**\n"]
        parts.append(f"طھظ…طھ ظ…ظ‚ط§ط±ظ†ط© `{len(bot_symbols)}` طµظپظ‚ط© ظ…ط³ط¬ظ„ط© ظپظٹ ط§ظ„ط¨ظˆطھ ظ…ط¹ `{len(exchange_symbols)}` ط¹ظ…ظ„ط© ظ…ظ…ظ„ظˆظƒط© ظپظٹ ط§ظ„ظ…ظ†طµط©.\n")

        parts.append(f"--- **âœ… طµظپظ‚ط§طھ ظ…طھط·ط§ط¨ظ‚ط© ({len(matched_symbols)})** ---")
        if matched_symbols:
            parts.extend([f"- `{s}`" for s in matched_symbols])
        else:
            parts.append("ظ„ط§ طھظˆط¬ط¯ طµظپظ‚ط§طھ ظ…طھط·ط§ط¨ظ‚ط© ط­ط§ظ„ظٹط§ظ‹.")

        parts.append(f"\n--- **âڑ ï¸ڈ طµظپظ‚ط§طھ ظپظٹ ط§ظ„ظ…ظ†طµط© ظپظ‚ط· ({len(exchange_only_symbols)})** ---")
        parts.append("*ظ‡ط°ظ‡ ظ‡ظٹ ط§ظ„طµظپظ‚ط§طھ ط§ظ„ط´ط¨ط­ظٹط© ط§ظ„ظ‚ط¯ظٹظ…ط© ط£ظˆ ط§ظ„طھظٹ طھظ… ط´ط±ط§ط¤ظ‡ط§ ظٹط¯ظˆظٹط§ظ‹.*")
        if exchange_only_symbols:
            parts.extend([f"- `{s}`" for s in exchange_only_symbols])
        else:
            parts.append("ظ„ط§ طھظˆط¬ط¯ طµظپظ‚ط§طھ ط؛ظٹط± ظ…ط³ط¬ظ„ط© ظپظٹ ط§ظ„ط¨ظˆطھ.")

        parts.append(f"\n--- **â‌“ طµظپظ‚ط§طھ ظپظٹ ط§ظ„ط¨ظˆطھ ظپظ‚ط· ({len(bot_only_symbols)})** ---")
        parts.append("*ظ‡ط°ظ‡ ط§ظ„طµظپظ‚ط§طھ ظ‚ط¯ طھظƒظˆظ† ط£ظڈط؛ظ„ظ‚طھ ظٹط¯ظˆظٹط§ظ‹. ظٹط¬ط¨ ط§ظ„طھط­ظ‚ظ‚ ظ…ظ†ظ‡ط§.*")
        if bot_only_symbols:
            parts.extend([f"- `{s}`" for s in bot_only_symbols])
        else:
            parts.append("ظ„ط§ طھظˆط¬ط¯ طµظپظ‚ط§طھ ط؛ظٹط± ظ…طھط·ط§ط¨ظ‚ط©.")

        await target_message.edit_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error during portfolio sync: {e}", exc_info=True)
        await target_message.edit_text(f"â‌Œ **ظپط´ظ„:** ط­ط¯ط« ط®ط·ط£ ط£ط«ظ†ط§ط، ظ…ط²ط§ظ…ظ†ط© ط§ظ„ظ…ط­ظپط¸ط©.\n`{e}`")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None: 
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

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
        logger.critical("CRITICAL: No public exchange clients connected. Bot cannot run.")
        return

    job_queue = application.job_queue
    job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name='perform_scan')
    job_queue.run_repeating(track_open_trades, interval=TRACK_INTERVAL_SECONDS, first=20, name='track_open_trades')
    job_queue.run_daily(send_daily_report, time=dt_time(hour=23, minute=55, tzinfo=EGYPT_TZ), name='daily_report')

    logger.info("Jobs scheduled.")
    await application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"ًںڑ€ *ط¨ظˆطھ ظƒط§ط³ط­ط© ط§ظ„ط£ظ„ط؛ط§ظ… (v5.2) ط¬ط§ظ‡ط² ظ„ظ„ط¹ظ…ظ„!*", parse_mode=ParseMode.MARKDOWN)

async def post_shutdown(application: Application):
    all_exchanges = list(bot_state.exchanges.values()) + list(bot_state.public_exchanges.values())
    unique_exchanges = list({id(ex): ex for ex in all_exchanges}.values())
    await asyncio.gather(*[ex.close() for ex in unique_exchanges])
    logger.info("All exchange connections closed.")

def main():
    """Sets up and runs the bot application."""
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print("FATAL ERROR: TELEGRAM_BOT_TOKEN is not set.")
        exit()

    load_settings()
    init_database()

    request = HTTPXRequest(connect_timeout=60.0, read_timeout=60.0)
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).post_init(post_init).post_shutdown(post_shutdown).build()

    # --- Registering all handlers ---
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("check", check_trade_command))
    application.add_handler(CommandHandler("trade", manual_trade_command))
    
    application.add_handler(CallbackQueryHandler(manual_trade_button_handler, pattern="^manual_trade_"))
    application.add_handler(CallbackQueryHandler(tools_button_handler, pattern="^(balance|openorders|mytrades)_"))
    application.add_handler(CallbackQueryHandler(button_callback_handler))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    application.add_error_handler(error_handler)
    
    logger.info("Application configured with all handlers. Starting polling...")
    application.run_polling()


if __name__ == '__main__':
    print("ًںڑ€ Starting Mineseper Bot v5.2 (Full & Final Version)...")
    try:
        main()
    except Exception as e:
        logging.critical(f"Bot stopped due to a critical unhandled error: {e}", exc_info=True)
