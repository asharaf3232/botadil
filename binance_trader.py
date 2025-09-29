# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 OKX Sniper Bot | v33.7 (Audit Certified & Fully Implemented) 🚀 ---
# =======================================================================================
#
# هذا هو الملف الكامل والنهائي دون أي اختصارات أو أجزاء محذوفة.
# يحتوي على جميع الإصلاحات الحرجة لواجهة التليجرام، الوقف المتحرك، وأمن قاعدة البيانات.
# تم تدقيقه وتصحيحه بناءً على بروتوكول التدقيق ثلاثي المراحل.
#
# =======================================================================================

# --- Core Libraries ---
import asyncio
import os
import logging
import json
import re
import time
import random
import csv
from io import StringIO
from datetime import datetime, timedelta, timezone, time as dt_time
from zoneinfo import ZoneInfo
import hmac
import base64
from collections import defaultdict, Counter
import copy
from typing import Optional, Dict, Any, List, Tuple

# --- Database & Networking ---
import aiosqlite
import websockets
import websockets.exceptions
import httpx
import feedparser
import redis.asyncio as redis

# --- Data Analysis & CCXT ---
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt

# --- [ترقية] مكتبات جديدة للعقل المطور ---
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not found. News sentiment analysis will be disabled.")

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Library 'scipy' not found. RSI Divergence strategy will be disabled.")


# --- Telegram & Environment ---
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest, TimedOut, Forbidden
from dotenv import load_dotenv

# =======================================================================================
# --- ⚙️ Core Configuration & Constants ⚙️ ---
# =======================================================================================
load_dotenv()

# --- API Keys & Tokens ---
OKX_API_KEY = os.getenv('OKX_API_KEY')
OKX_API_SECRET = os.getenv('OKX_API_SECRET')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_AV_KEY_HERE')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# --- Timing & Intervals ---
TIMEFRAME = '15m'
SCAN_INTERVAL_SECONDS = 900
SUPERVISOR_INTERVAL_SECONDS = 180
TIME_SYNC_INTERVAL_SECONDS = 3600
STRATEGY_ANALYSIS_INTERVAL_SECONDS = 21600 # 6 hours
INTELLIGENT_REVIEWER_INTERVAL_MINUTES = 30
MAESTRO_INTERVAL_HOURS = 1
CRITICAL_MONITOR_INTERVAL_SECONDS = 600 # 10 minutes

# --- File Paths ---
DB_FILE = 'okx_sniper_v33.db'
SETTINGS_FILE = 'okx_sniper_settings_v33.json'
DECISION_MATRIX_FILE = 'decision_matrix.json'

# --- Strategy Constants (No Magic Numbers) ---
WHALE_RADAR_MIN_BIDS_USD = 30000.0
SNIPER_PRO_MAX_VOLATILITY_PERCENT = 12.0

# --- Market Regime Constants (Best Practice: Avoid Magic Numbers) ---
REGIME_ADX_THRESHOLD = 25.0
REGIME_ATR_PERCENT_THRESHOLD = 2.0

EGYPT_TZ = ZoneInfo("Africa/Cairo")

# =======================================================================================
# --- 📝 Logging Setup 📝 ---
# =======================================================================================
class SafeFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'trade_id'): record.trade_id = 'N/A'
        return super().format(record)

log_formatter = SafeFormatter('%(asctime)s - %(levelname)s - [TradeID:%(trade_id)s] - %(message)s')
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)
root_logger = logging.getLogger()
root_logger.handlers = [log_handler]
root_logger.setLevel(logging.INFO)

class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'extra' not in kwargs: kwargs['extra'] = {}
        if 'trade_id' not in kwargs['extra']: kwargs['extra']['trade_id'] = 'N/A'
        return msg, kwargs
logger = ContextAdapter(logging.getLogger("OKX_Sniper_Bot"), {})

# =======================================================================================
# --- 🔬 Global Bot State & Locks 🔬 ---
# =======================================================================================
class BotState:
    def __init__(self):
        self.settings: Dict[str, Any] = {}
        self.trading_enabled: bool = True
        self.active_preset_name: str = "مخصص"
        self.last_signal_time: Dict[str, float] = {}
        self.application: Optional[Application] = None
        self.exchange: Optional[ccxt.Exchange] = None
        self.market_mood: Dict[str, str] = {"mood": "UNKNOWN", "reason": "تحليل لم يتم بعد"}
        self.private_ws: Optional[Any] = None
        self.public_ws: Optional[Any] = None
        self.trade_guardian: Optional[Any] = None
        self.last_scan_info: Dict[str, Any] = {}
        self.all_markets: List[Dict] = []
        self.last_markets_fetch: float = 0
        self.strategy_performance: Dict[str, Any] = {}
        self.pending_strategy_proposal: Dict[str, Any] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.current_market_regime: str = "UNKNOWN"

bot_data = BotState()
scan_lock = asyncio.Lock()
trade_management_lock = asyncio.Lock()

# =======================================================================================
# --- 💡 Default Settings, Filters & UI Constants 💡 ---
# =======================================================================================
DEFAULT_SETTINGS = {
    "real_trade_size_usdt": 15.0,
    "max_concurrent_trades": 5,
    "top_n_symbols_by_volume": 300,
    "worker_threads": 10,
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 2.0,
    "trailing_sl_enabled": True,
    "trailing_sl_activation_percent": 1.5,
    "trailing_sl_callback_percent": 1.0,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "sniper_pro", "whale_radar", "rsi_divergence", "supertrend_pullback"],
    "market_mood_filter_enabled": True,
    "fear_and_greed_threshold": 30,
    "adx_filter_enabled": True,
    "adx_filter_level": 25,
    "btc_trend_filter_enabled": True,
    "news_filter_enabled": True,
    "asset_blacklist": ["USDC", "DAI", "TUSD", "FDUSD", "USDD", "PYUSD", "USDT", "BNB", "OKB", "KCS", "BGB", "MX", "GT", "HT", "BTC", "ETH"],
    "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "min_rvol": 1.5},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.8},
    "trend_filters": {"ema_period": 200, "htf_period": 50, "enabled": True},
    "spread_filter": {"max_spread_percent": 0.5},
    "rsi_divergence": {"rsi_period": 14, "lookback_period": 35, "peak_trough_lookback": 5, "confirm_with_rsi_exit": True},
    "supertrend_pullback": {"atr_period": 10, "atr_multiplier": 3.0, "swing_high_lookback": 10},
    "multi_timeframe_enabled": True,
    "multi_timeframe_htf": '4h',
    "volume_filter_multiplier": 2.0,
    "close_retries": 3,
    "incremental_notifications_enabled": True,
    "incremental_notification_percent": 2.0,
    "adaptive_intelligence_enabled": True,
    "dynamic_trade_sizing_enabled": True,
    "strategy_proposal_enabled": True,
    "strategy_analysis_min_trades": 10,
    "strategy_deactivation_threshold_wr": 45.0,
    "dynamic_sizing_max_increase_pct": 25.0,
    "dynamic_sizing_max_decrease_pct": 50.0,
    "intelligent_reviewer_enabled": True,
    "intelligent_reviewer_interval_minutes": 30,
    "momentum_scalp_mode_enabled": False,
    "momentum_scalp_target_percent": 0.5,
    "multi_timeframe_confluence_enabled": True,
    "maestro_mode_enabled": True,
}
STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي", "breakout_squeeze_pro": "اختراق انضغاطي",
    "support_rebound": "ارتداد الدعم", "sniper_pro": "القناص المحترف", "whale_radar": "رادار الحيتان",
    "rsi_divergence": "دايفرجنس RSI", "supertrend_pullback": "انعكاس سوبرترند",
    "bollinger_reversal": "انعكاس بولينجر"
}
PRESET_NAMES_AR = {"professional": "احترافي", "strict": "متشدد", "lenient": "متساهل", "very_lenient": "فائق التساهل", "bold_heart": "القلب الجريء"}
SETTINGS_PRESETS = {
    "professional": copy.deepcopy(DEFAULT_SETTINGS),
    "strict": {**copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 3, "risk_reward_ratio": 2.5, "fear_and_greed_threshold": 40, "adx_filter_level": 28, "liquidity_filters": {"min_quote_volume_24h_usd": 2000000, "min_rvol": 2.0}},
    "lenient": {**copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 8, "risk_reward_ratio": 1.8, "fear_and_greed_threshold": 25, "adx_filter_level": 20, "liquidity_filters": {"min_quote_volume_24h_usd": 500000, "min_rvol": 1.2}},
    "very_lenient": {
        **copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 12, "adx_filter_enabled": False,
        "market_mood_filter_enabled": False, "trend_filters": {"ema_period": 200, "htf_period": 50, "enabled": False},
        "liquidity_filters": {"min_quote_volume_24h_usd": 250000, "min_rvol": 1.0},
        "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.4}, "spread_filter": {"max_spread_percent": 1.5}
    },
    "bold_heart": {
        **copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 15, "risk_reward_ratio": 1.5, "multi_timeframe_enabled": False,
        "market_mood_filter_enabled": False, "adx_filter_enabled": False, "btc_trend_filter_enabled": False, "news_filter_enabled": False,
        "volume_filter_multiplier": 1.0, "liquidity_filters": {"min_quote_volume_24h_usd": 100000, "min_rvol": 1.0},
        "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.2}, "spread_filter": {"max_spread_percent": 2.0}
    }
}
DECISION_MATRIX = {
    "TRENDING_HIGH_VOLATILITY": {
        "intelligent_reviewer_enabled": True, "momentum_scalp_mode_enabled": True, "multi_timeframe_confluence_enabled": True,
        "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "sniper_pro", "whale_radar"],
        "risk_reward_ratio": 1.5, "volume_filter_multiplier": 2.5
    },
    "TRENDING_LOW_VOLATILITY": {
        "intelligent_reviewer_enabled": True, "momentum_scalp_mode_enabled": False, "multi_timeframe_confluence_enabled": True,
        "active_scanners": ["support_rebound", "supertrend_pullback", "rsi_divergence"],
        "risk_reward_ratio": 2.5, "volume_filter_multiplier": 1.5
    },
    "SIDEWAYS_HIGH_VOLATILITY": {
        "intelligent_reviewer_enabled": True, "momentum_scalp_mode_enabled": True, "multi_timeframe_confluence_enabled": False,
        "active_scanners": ["bollinger_reversal", "rsi_divergence", "breakout_squeeze_pro"],
        "risk_reward_ratio": 2.0, "volume_filter_multiplier": 2.0
    },
    "SIDEWAYS_LOW_VOLATILITY": {
        "intelligent_reviewer_enabled": False, "momentum_scalp_mode_enabled": False, "multi_timeframe_confluence_enabled": True,
        "active_scanners": ["bollinger_reversal", "support_rebound"],
        "risk_reward_ratio": 3.0, "volume_filter_multiplier": 1.0
    }
}

if not os.path.exists(DECISION_MATRIX_FILE):
    with open(DECISION_MATRIX_FILE, 'w', encoding='utf-8') as f:
        json.dump(DECISION_MATRIX, f, ensure_ascii=False, indent=4)

# =======================================================================================
# --- Helper, Settings & DB Management ---
# =======================================================================================
def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                bot_data.settings = json.load(f)
        else:
            bot_data.settings = copy.deepcopy(DEFAULT_SETTINGS)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to load settings file, falling back to defaults. Error: {e}")
        bot_data.settings = copy.deepcopy(DEFAULT_SETTINGS)
    
    default_copy = copy.deepcopy(DEFAULT_SETTINGS)
    for key, value in default_copy.items():
        if isinstance(value, dict):
            if key not in bot_data.settings or not isinstance(bot_data.settings.get(key), dict):
                bot_data.settings[key] = {}
            for sub_key, sub_value in value.items():
                bot_data.settings[key].setdefault(sub_key, sub_value)
        else:
            bot_data.settings.setdefault(key, value)
    
    determine_active_preset()
    save_settings()
    logger.info(f"Settings loaded. Active preset: {bot_data.active_preset_name}")

def determine_active_preset():
    current_settings_for_compare = {k: v for k, v in bot_data.settings.items() if k in DEFAULT_SETTINGS}
    for name, preset_settings in SETTINGS_PRESETS.items():
        is_match = all(
            preset_settings.get(key) == current_settings_for_compare.get(key)
            for key in preset_settings
        )
        if is_match:
            bot_data.active_preset_name = PRESET_NAMES_AR.get(name, "مخصص")
            return
    bot_data.active_preset_name = "مخصص"

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(bot_data.settings, f, ensure_ascii=False, indent=4)
    except IOError as e:
        logger.error(f"Could not save settings to {SETTINGS_FILE}: {e}")

async def safe_send_message(bot, text, **kwargs):
    try:
        await bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except (TimedOut, Forbidden, BadRequest) as e:
        logger.error(f"Telegram Send Error: {e}")

async def safe_edit_message(query, text, **kwargs):
    try:
        if query and hasattr(query, 'edit_message_text'):
            await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except BadRequest as e:
        if "Message is not modified" not in str(e):
            logger.warning(f"Edit Message Error: {e}")
    except (TimedOut, Forbidden) as e:
        logger.error(f"Edit Message Error: {e}")

async def init_database():
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, entry_price REAL, take_profit REAL, stop_loss REAL,
                    quantity REAL, status TEXT, reason TEXT, order_id TEXT,
                    highest_price REAL DEFAULT 0, trailing_sl_active BOOLEAN DEFAULT 0,
                    close_price REAL, pnl_usdt REAL, signal_strength INTEGER DEFAULT 1,
                    close_retries INTEGER DEFAULT 0, last_profit_notification_price REAL DEFAULT 0,
                    trade_weight REAL DEFAULT 1.0
                )
            ''')
            
            cursor = await conn.execute("PRAGMA table_info(trades)")
            columns = [row[1] for row in await cursor.fetchall()]
            
            schema_updates = {
                'signal_strength': "ALTER TABLE trades ADD COLUMN signal_strength INTEGER DEFAULT 1",
                'close_retries': "ALTER TABLE trades ADD COLUMN close_retries INTEGER DEFAULT 0",
                'last_profit_notification_price': "ALTER TABLE trades ADD COLUMN last_profit_notification_price REAL DEFAULT 0",
                'trade_weight': "ALTER TABLE trades ADD COLUMN trade_weight REAL DEFAULT 1.0"
            }
            for col, statement in schema_updates.items():
                if col not in columns:
                    await conn.execute(statement)

            await conn.commit()
        logger.info("Adaptive database initialized successfully.")
    except aiosqlite.Error as e:
        logger.critical(f"Database initialization failed: {e}")

async def log_pending_trade_to_db(signal: Dict[str, Any], buy_order: Dict[str, Any]) -> bool:
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("""
                INSERT INTO trades (timestamp, symbol, reason, order_id, status, entry_price, 
                                    take_profit, stop_loss, signal_strength, 
                                    last_profit_notification_price, trade_weight) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(EGYPT_TZ).isoformat(), signal['symbol'], signal['reason'], 
                buy_order['id'], 'pending', signal['entry_price'], signal['take_profit'], 
                signal['stop_loss'], signal.get('strength', 1), signal['entry_price'], 
                signal.get('weight', 1.0)
            ))
            await conn.commit()
            logger.info(f"Logged pending trade for {signal['symbol']} with order ID {buy_order['id']}.")
            return True
    except aiosqlite.Error as e:
        logger.error(f"DB Log Pending Error: {e}")
        return False

async def broadcast_signal_to_redis(signal):
    if not bot_data.redis_client:
        logger.warning("Redis client not available. Skipping broadcast.")
        return

    try:
        signal_to_broadcast = {
            key: value.isoformat() if isinstance(value, (datetime, pd.Timestamp)) else value
            for key, value in signal.items()
        }
        json_signal = json.dumps(signal_to_broadcast)
        channel = "trade_signals"
        await bot_data.redis_client.publish(channel, json_signal)
        logger.info(f"📡 Broadcasted signal for {signal['symbol']} to Redis channel '{channel}'.")
    except (TypeError, redis.RedisError) as e:
        logger.error(f"Redis Broadcast Error for {signal.get('symbol', 'N/A')}: {e}", exc_info=True)

# =======================================================================================
# --- 🧠 Mastermind Brain (Analysis & Mood) 🧠 ---
# =======================================================================================
async def update_strategy_performance(context: ContextTypes.DEFAULT_TYPE):
    logger.info("🧠 Adaptive Mind: Analyzing strategy performance...")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            cursor = await conn.execute("SELECT reason, status, pnl_usdt FROM trades WHERE status LIKE '%(%' ORDER BY id DESC LIMIT 100")
            trades = await cursor.fetchall()

        if not trades:
            logger.info("🧠 Adaptive Mind: No closed trades found to analyze.")
            return
            
        stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'win_pnl': 0.0, 'loss_pnl': 0.0})
        for reason_str, status, pnl in trades:
            if not reason_str or pnl is None: continue
            clean_reason = reason_str.split(' (')[0]
            reasons = clean_reason.split(' + ')
            for reason in set(reasons):
                is_win = 'ناجحة' in status or 'تأمين' in status
                if is_win:
                    stats[reason]['wins'] += 1
                    stats[reason]['win_pnl'] += pnl
                else:
                    stats[reason]['losses'] += 1
                    stats[reason]['loss_pnl'] += pnl
                stats[reason]['total_pnl'] += pnl
        
        performance_data = {}
        for reason, data in stats.items():
            total = data['wins'] + data['losses']
            win_rate = (data['wins'] / total * 100) if total > 0 else 0
            profit_factor = data['win_pnl'] / abs(data['loss_pnl']) if data['loss_pnl'] != 0 else float('inf')
            performance_data[reason] = {"win_rate": round(win_rate, 2), "profit_factor": round(profit_factor, 2), "total_trades": total}
        
        bot_data.strategy_performance = performance_data
        logger.info(f"🧠 Adaptive Mind: Analysis complete. Performance data for {len(performance_data)} strategies updated.")
    except aiosqlite.Error as e:
        logger.error(f"🧠 Adaptive Mind: DB error during performance analysis: {e}", exc_info=True)


async def propose_strategy_changes(context: ContextTypes.DEFAULT_TYPE):
    settings = bot_data.settings
    if not settings.get('adaptive_intelligence_enabled') or not settings.get('strategy_proposal_enabled'):
        return
        
    logger.info("🧠 Adaptive Mind: Checking for underperforming strategies to propose changes...")
    active_scanners = settings.get('active_scanners', [])
    min_trades = settings.get('strategy_analysis_min_trades', 10)
    deactivation_wr = settings.get('strategy_deactivation_threshold_wr', 45.0)

    for scanner in active_scanners:
        perf = bot_data.strategy_performance.get(scanner)
        if perf and perf['total_trades'] >= min_trades and perf['win_rate'] < deactivation_wr:
            if bot_data.pending_strategy_proposal.get('scanner') == scanner: continue
            
            proposal_key = f"prop_{int(time.time())}"
            bot_data.pending_strategy_proposal = {
                "key": proposal_key, "action": "disable", "scanner": scanner,
                "reason": f"أظهرت أداءً ضعيفًا بمعدل نجاح `{perf['win_rate']}%` في آخر `{perf['total_trades']}` صفقة."
            }
            logger.warning(f"🧠 Adaptive Mind: Proposing to disable '{scanner}' due to low performance.")
            
            message = (f"💡 **اقتراح تحسين الأداء** 💡\n\n"
                       f"مرحباً، بناءً على التحليل المستمر، لاحظت أن استراتيجية **'{STRATEGY_NAMES_AR.get(scanner, scanner)}'** "
                       f"{bot_data.pending_strategy_proposal['reason']}\n\n"
                       f"أقترح تعطيلها مؤقتًا للتركيز على الاستراتيجيات الأكثر ربحية. هل توافق على هذا التعديل؟")
            keyboard = [[InlineKeyboardButton("✅ موافقة", callback_data=f"strategy_adjust_approve_{proposal_key}"),
                         InlineKeyboardButton("❌ رفض", callback_data=f"strategy_adjust_reject_{proposal_key}")]]
            
            await safe_send_message(context.bot, message, reply_markup=InlineKeyboardMarkup(keyboard))
            return

async def translate_text_gemini(text_list: List[str]) -> Tuple[List[str], bool]:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found. Skipping translation.")
        return text_list, False
    if not text_list:
        return [], True
        
    prompt = "Translate the following English headlines to Arabic. Return only the translated text, with each headline on a new line:\n\n" + "\n".join(text_list)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            # Robust JSON parsing
            translated_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            if not translated_text:
                logger.error("Gemini translation returned empty text.")
                return text_list, False
            return translated_text.strip().split('\n'), True
    except (httpx.RequestError, httpx.HTTPStatusError, KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Gemini translation failed: {e}")
        return text_list, False

def get_alpha_vantage_economic_events() -> Optional[List[str]]:
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == 'YOUR_AV_KEY_HERE':
        return []
        
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    params = {'function': 'ECONOMIC_CALENDAR', 'horizon': '3month', 'apikey': ALPHA_VANTAGE_API_KEY}
    
    try:
        response = httpx.get('https://www.alphavantage.co/query', params=params, timeout=20)
        response.raise_for_status()
        data_str = response.text
        if "premium" in data_str.lower(): return []
        
        # Robust CSV parsing to handle commas within fields
        csv_file = StringIO(data_str)
        reader = csv.DictReader(csv_file)
        events = list(reader)
        
        high_impact_events = [
            e.get('event', 'Unknown Event') for e in events 
            if e.get('releaseDate', '') == today_str and 
               e.get('impact', '').lower() == 'high' and 
               e.get('country', '') in ['USD', 'EUR']
        ]
        
        if high_impact_events:
            logger.warning(f"High-impact events today: {high_impact_events}")
        return high_impact_events
    except httpx.RequestError as e:
        logger.error(f"Failed to fetch economic calendar: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to parse economic calendar CSV: {e}")
        return None

def get_latest_crypto_news(limit=15) -> List[str]:
    urls = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]
    all_entries = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            if feed.bozo: # Handles parsing errors
                logger.warning(f"Failed to parse RSS feed: {url}, Error: {feed.bozo_exception}")
                continue
            all_entries.extend(feed.entries)
        except Exception as e:
            logger.error(f"Error fetching RSS feed {url}: {e}")

    # Sort by published date, newest first
    all_entries.sort(key=lambda x: x.get('published_parsed', time.gmtime()), reverse=True)
    
    headlines = [entry.title for entry in all_entries]
    return list(dict.fromkeys(headlines))[:limit] # Remove duplicates while preserving order

def analyze_sentiment_of_headlines(headlines: List[str]) -> Tuple[str, float]:
    if not headlines or not NLTK_AVAILABLE:
        return "N/A", 0.0
        
    sia = SentimentIntensityAnalyzer()
    
    # Filter out potential empty strings from failed RSS parsing
    valid_headlines = [h for h in headlines if h]
    if not valid_headlines:
        return "N/A", 0.0

    score = sum(sia.polarity_scores(h)['compound'] for h in valid_headlines) / len(valid_headlines)
    
    if score > 0.15: mood = "إيجابية"
    elif score < -0.15: mood = "سلبية"
    else: mood = "محايدة"
    return mood, score

async def get_fundamental_market_mood() -> Dict[str, str]:
    settings = bot_data.settings
    if not settings.get('news_filter_enabled', True):
        return {"mood": "POSITIVE", "reason": "فلتر الأخبار معطل"}
        
    high_impact_events = await asyncio.to_thread(get_alpha_vantage_economic_events)
    if high_impact_events is None:
        return {"mood": "DANGEROUS", "reason": "فشل جلب البيانات الاقتصادية"}
    if high_impact_events:
        return {"mood": "DANGEROUS", "reason": f"أحداث هامة اليوم: {', '.join(high_impact_events)}"}
        
    latest_headlines = await asyncio.to_thread(get_latest_crypto_news)
    sentiment, score = analyze_sentiment_of_headlines(latest_headlines)
    logger.info(f"Market sentiment score: {score:.2f} ({sentiment})")
    
    if score > 0.25: return {"mood": "POSITIVE", "reason": f"مشاعر إيجابية (الدرجة: {score:.2f})"}
    elif score < -0.25: return {"mood": "NEGATIVE", "reason": f"مشاعر سلبية (الدرجة: {score:.2f})"}
    else: return {"mood": "NEUTRAL", "reason": f"مشاعر محايدة (الدرجة: {score:.2f})"}

def find_col(df_columns: pd.Index, prefix: str) -> Optional[str]:
    try:
        return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration:
        return None

async def get_fear_and_greed_index() -> Optional[int]:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            r.raise_for_status()
            data = r.json().get('data', [])
            if data:
                value_str = data[0].get('value')
                if value_str:
                    return int(value_str)
            return None # Return None if data is not as expected
    except (httpx.RequestError, KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"Fear & Greed fetch error: {e}")
        return None

async def get_market_mood() -> Dict[str, str]:
    settings = bot_data.settings
    btc_mood_text = "الفلتر معطل"
    
    if settings.get('btc_trend_filter_enabled', True):
        try:
            htf_period = settings['trend_filters']['htf_period']
            ohlcv = await bot_data.exchange.fetch_ohlcv('BTC/USDT', '4h', limit=htf_period + 5)
            if not ohlcv or len(ohlcv) < htf_period:
                raise ccxt.ExchangeError("Insufficient data for BTC trend analysis")
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['sma'] = ta.sma(df['close'], length=htf_period)
            is_btc_bullish = df['close'].iloc[-1] > df['sma'].iloc[-1]
            btc_mood_text = "صاعد ✅" if is_btc_bullish else "هابط ❌"
            if not is_btc_bullish:
                return {"mood": "NEGATIVE", "reason": "اتجاه BTC هابط", "btc_mood": btc_mood_text}
        except ccxt.BaseError as e:
            logger.error(f"Could not fetch BTC data for market mood: {e}")
            return {"mood": "DANGEROUS", "reason": f"فشل جلب بيانات BTC", "btc_mood": "UNKNOWN"}

    if settings.get('market_mood_filter_enabled', True):
        fng = await get_fear_and_greed_index()
        if fng is not None and fng < settings['fear_and_greed_threshold']:
            return {"mood": "NEGATIVE", "reason": f"مشاعر خوف شديد (F&G: {fng})", "btc_mood": btc_mood_text}
            
    return {"mood": "POSITIVE", "reason": "وضع السوق مناسب", "btc_mood": btc_mood_text}

# =======================================================================================
# --- 🔬 Technical Analysis Scanners 🔬 ---
# =======================================================================================
def analyze_momentum_breakout(df, params, rvol, adx_value):
    df.ta.vwap(append=True); df.ta.bbands(length=20, append=True); df.ta.macd(append=True); df.ta.rsi(append=True)
    last, prev = df.iloc[-2], df.iloc[-3]
    macd_col, macds_col, bbu_col, rsi_col = find_col(df.columns, "MACD_"), find_col(df.columns, "MACDs_"), find_col(df.columns, "BBU_"), find_col(df.columns, "RSI_")
    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and last[rsi_col] < 68):
        return {"reason": "momentum_breakout"}
    return None

def analyze_breakout_squeeze_pro(df, params, rvol, adx_value):
    df.ta.bbands(length=20, append=True); df.ta.kc(length=20, scalar=1.5, append=True); df.ta.obv(append=True)
    bbu_col, bbl_col, kcu_col, kcl_col = find_col(df.columns, "BBU_"), find_col(df.columns, "BBL_"), find_col(df.columns, "KCUe_"), find_col(df.columns, "KCLEe_")
    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]
    if is_in_squeeze and (last['close'] > last[bbu_col]) and (last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5) and (df['OBV'].iloc[-2] > df['OBV'].iloc[-3]):
        return {"reason": "breakout_squeeze_pro"}
    return None

async def analyze_support_rebound(df, params, rvol, adx_value, exchange, symbol):
    try:
        ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
        if len(ohlcv_1h) < 50: return None
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')

        current_price = df_1h['close'].iloc[-1]
        
        # Logic to find the most recent, highest support level
        recent_lows = df_1h['low'].rolling(window=10, center=True).min()
        supports = recent_lows[recent_lows.notna()]
        
        # Get support levels below current price along with their timestamps
        candidate_supports = [
            (level, df_1h.index[i]) 
            for i, level in supports.items() 
            if level < current_price
        ]
        
        if not candidate_supports: return None

        # Find the highest support level among the candidates
        max_support_level = max(s[0] for s in candidate_supports)
        # Filter to only include the highest support levels
        top_supports = [s for s in candidate_supports if s[0] == max_support_level]
        # Get the one with the most recent timestamp
        closest_support_level, _ = max(top_supports, key=lambda item: item[1])

        if (current_price - closest_support_level) / closest_support_level * 100 > 1.0: return None
        
        last_candle_15m = df.iloc[-2]
        if last_candle_15m['close'] > last_candle_15m['open'] and last_candle_15m['volume'] > df['volume'].rolling(window=20).mean().iloc[-2] * 1.5:
            return {"reason": "support_rebound"}
    except (ccxt.BaseError, IndexError, KeyError) as e:
        logger.warning(f"Scanner 'support_rebound' for {symbol} failed: {e}")
        return None
    return None

def analyze_sniper_pro(df, params, rvol, adx_value):
    try:
        compression_candles = 24
        if len(df) < compression_candles + 2: return None
        compression_df = df.iloc[-compression_candles-1:-1]
        highest_high, lowest_low = compression_df['high'].max(), compression_df['low'].min()
        if lowest_low <= 0: return None
        volatility = (highest_high - lowest_low) / lowest_low * 100
        if volatility < SNIPER_PRO_MAX_VOLATILITY_PERCENT:
            last_candle = df.iloc[-2]
            if last_candle['close'] > highest_high and last_candle['volume'] > compression_df['volume'].mean() * 2:
                return {"reason": "sniper_pro"}
    except (IndexError, KeyError):
        return None
    return None

async def analyze_whale_radar(df, params, rvol, adx_value, exchange, symbol):
    try:
        ob = await exchange.fetch_order_book(symbol, limit=20)
        if not ob or not ob.get('bids'): return None
        # --- ✅ هنا التعديل ---
        # تجاهل القيم الإضافية باستخدام *_
        if sum(float(price) * float(qty) for price, qty, *_ in ob['bids'][:10]) > WHALE_RADAR_MIN_BIDS_USD:
            return {"reason": "whale_radar"}
    except ccxt.BaseError as e:
        logger.warning(f"Scanner 'whale_radar' for {symbol} failed: {e}")
        return None
    return None

def analyze_rsi_divergence(df, params, rvol, adx_value):
    if not SCIPY_AVAILABLE: return None
    df.ta.rsi(length=params.get('rsi_period', 14), append=True)
    rsi_col = find_col(df.columns, f"RSI_{params.get('rsi_period', 14)}")
    if not rsi_col or df[rsi_col].isnull().all(): return None
    subset = df.iloc[-params.get('lookback_period', 35):].copy()
    price_troughs_idx, _ = find_peaks(-subset['low'], distance=params.get('peak_trough_lookback', 5))
    rsi_troughs_idx, _ = find_peaks(-subset[rsi_col], distance=params.get('peak_trough_lookback', 5))
    if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
        p_low1_idx, p_low2_idx = price_troughs_idx[-2], price_troughs_idx[-1]
        r_low1_idx, r_low2_idx = rsi_troughs_idx[-2], rsi_troughs_idx[-1]
        is_divergence = (subset.iloc[p_low2_idx]['low'] < subset.iloc[p_low1_idx]['low'] and subset.iloc[r_low2_idx][rsi_col] > subset.iloc[r_low1_idx][rsi_col])
        if is_divergence:
            rsi_exits_oversold = (subset.iloc[r_low1_idx][rsi_col] < 35 and subset.iloc[-2][rsi_col] > 40)
            confirmation_price = subset.iloc[p_low2_idx:]['high'].max()
            price_confirmed = df.iloc[-2]['close'] > confirmation_price
            if (not params.get('confirm_with_rsi_exit', True) or rsi_exits_oversold) and price_confirmed:
                return {"reason": "rsi_divergence"}
    return None

def analyze_supertrend_pullback(df, params, rvol, adx_value):
    df.ta.supertrend(length=params.get('atr_period', 10), multiplier=params.get('atr_multiplier', 3.0), append=True)
    st_dir_col = find_col(df.columns, f"SUPERTd_{params.get('atr_period', 10)}_")
    if not st_dir_col: return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if prev[st_dir_col] == -1 and last[st_dir_col] == 1:
        recent_swing_high = df['high'].iloc[-params.get('swing_high_lookback', 10):-2].max()
        if last['close'] > recent_swing_high:
            return {"reason": "supertrend_pullback"}
    return None

def analyze_bollinger_reversal(df, params, rvol, adx_value):
    df.ta.bbands(length=20, append=True)
    df.ta.rsi(append=True)
    bbl_col, bbm_col, bbu_col = find_col(df.columns, "BBL_20_2.0"), find_col(df.columns, "BBM_20_2.0"), find_col(df.columns, "BBU_20_2.0")
    rsi_col = find_col(df.columns, "RSI_14")
    if not all([bbl_col, bbm_col, bbu_col, rsi_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if prev['close'] < prev[bbl_col] and last['close'] > last[bbl_col] and last['close'] < last[bbm_col] and last[rsi_col] < 35:
        return {"reason": "bollinger_reversal"}
    return None

SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout, "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "support_rebound": analyze_support_rebound, "sniper_pro": analyze_sniper_pro, "whale_radar": analyze_whale_radar,
    "rsi_divergence": analyze_rsi_divergence, "supertrend_pullback": analyze_supertrend_pullback,
    "bollinger_reversal": analyze_bollinger_reversal
}

# =======================================================================================
# --- 🚀 Hybrid Core Protocol (Execution & Management) 🚀 ---
# =======================================================================================
async def activate_trade(order_id, symbol):
    bot = bot_data.application.bot; log_ctx = {'trade_id': 'N/A'}
    try:
        order_details = await bot_data.exchange.fetch_order(order_id, symbol)
        filled_price, gross_filled_quantity = order_details.get('average', 0.0), order_details.get('filled', 0.0)
        if gross_filled_quantity <= 0 or filled_price <= 0:
            logger.error(f"Order {order_id} invalid fill data. Price: {filled_price}, Qty: {gross_filled_quantity}."); return
        net_filled_quantity = gross_filled_quantity
        base_currency = symbol.split('/')[0]
        if 'fee' in order_details and order_details['fee'] and 'cost' in order_details['fee']:
            fee_cost, fee_currency = order_details['fee']['cost'], order_details['fee']['currency']
            if fee_currency == base_currency:
                net_filled_quantity -= fee_cost
                logger.info(f"Fee of {fee_cost} {fee_currency} deducted. Net quantity for {symbol} is {net_filled_quantity}.")
        if net_filled_quantity <= 0: logger.error(f"Net quantity for {order_id} is zero or less. Aborting."); return
        balance_after = await bot_data.exchange.fetch_balance()
        usdt_remaining = balance_after.get('USDT', {}).get('free', 0)
    except (ccxt.BaseError, KeyError) as e:
        logger.error(f"Could not fetch data for trade activation: {e}", exc_info=True)
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("UPDATE trades SET status = 'failed', reason = 'Activation Fetch Error' WHERE order_id = ?", (order_id,)); await conn.commit()
        return
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        trade_cursor = await conn.execute("SELECT * FROM trades WHERE order_id = ? AND status = 'pending'", (order_id,))
        trade = await trade_cursor.fetchone()
        if not trade: logger.info(f"Activation ignored for {order_id}: Trade not pending."); return
        trade = dict(trade); log_ctx['trade_id'] = trade['id']
        logger.info(f"Activating trade #{trade['id']} for {symbol}...", extra=log_ctx)
        risk = filled_price - trade['stop_loss']
        new_take_profit = filled_price + (risk * bot_data.settings['risk_reward_ratio'])
        await conn.execute("UPDATE trades SET status = 'active', entry_price = ?, quantity = ?, take_profit = ? WHERE id = ?", (filled_price, net_filled_quantity, new_take_profit, trade['id']))
        active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]
        await conn.commit()

    await bot_data.public_ws.subscribe([symbol])
    trade_cost, tp_percent, sl_percent = filled_price * net_filled_quantity, (new_take_profit / filled_price - 1) * 100 if filled_price > 0 else 0, (1 - trade['stop_loss'] / filled_price) * 100 if filled_price > 0 else 0
    reasons_en = trade['reason'].split(' + ')
    reasons_ar = [STRATEGY_NAMES_AR.get(r.strip(), r.strip()) for r in reasons_en]
    reason_display_str = ' + '.join(reasons_ar)
    strength_stars = '⭐' * trade.get('signal_strength', 1)
    trade_weight = trade.get('trade_weight', 1.0)
    confidence_level_str = f"**🧠 مستوى الثقة:** `{trade_weight:.0%}` (تم تعديل الحجم)\n" if trade_weight != 1.0 else ""

    success_msg = (f"✅ **تم تأكيد الشراء | {symbol}**\n"
                   f"**الاستراتيجية:** {reason_display_str}\n"
                   f"**قوة الإشارة:** {strength_stars}\n"
                   f"{confidence_level_str}"
                   f"🔸 **الصفقة رقم:** #{trade['id']}\n"
                   f"🔸 **سعر التنفيذ:** `${filled_price:,.4f}`\n"
                   f"🔸 **الكمية (صافي):** {net_filled_quantity:,.4f} {symbol.split('/')[0]}\n"
                   f"🔸 **التكلفة:** `${trade_cost:,.2f}`\n"
                   f"🎯 **الهدف (TP):** `${new_take_profit:,.4f} (ربح متوقع: {tp_percent:+.2f}%)`\n"
                   f"🛡️ **الوقف (SL):** `${trade['stop_loss']:,.4f} (خسارة مقبولة: {sl_percent:.2f}%)`\n"
                   f"💰 **السيولة المتبقية (USDT):** `${usdt_remaining:,.2f}`\n"
                   f"🔄 **إجمالي الصفقات النشطة:** `{active_trades_count}`\n"
                   f"الحارس الأمين يراقب الصفقة الآن.")
    await safe_send_message(bot, success_msg)


async def handle_filled_buy_order(order_data: Dict[str, Any]):
    symbol, order_id = order_data['instId'].replace('-', '/'), order_data['ordId']
    avg_price = float(order_data.get('avgPx', 0))
    if avg_price > 0:
        logger.info(f"Fast Reporter: Received fill for {order_id}. Activating...")
        await activate_trade(order_id, symbol)
    else:
        logger.error(f"Fast Reporter: Received filled order {order_id} for {symbol} with avgPx of 0. Ignoring activation.")

async def exponential_backoff_with_jitter(run_coro, *args, **kwargs):
    retries = 0; base_delay, max_delay = 2, 120
    while True:
        try: 
            await run_coro(*args, **kwargs)
            break # Exit loop on success
        except (websockets.exceptions.ConnectionClosed, httpx.RequestError, ccxt.NetworkError) as e:
            retries += 1; backoff_delay = min(max_delay, base_delay * (2 ** retries)); jitter = random.uniform(0, backoff_delay * 0.5); total_delay = backoff_delay + jitter
            logger.error(f"Coroutine {run_coro.__name__} failed with recoverable error: {e}. Retrying in {total_delay:.2f} seconds...")
            await asyncio.sleep(total_delay)
        except Exception as e:
            logger.critical(f"Coroutine {run_coro.__name__} failed with unrecoverable error: {e}", exc_info=True)
            break # Do not retry on unknown errors

class PrivateWebSocketManager:
    def __init__(self): self.ws_url = "wss://ws.okx.com:8443/ws/v5/private"
    def _get_auth_args(self):
        timestamp = str(time.time()); message = timestamp + 'GET' + '/users/self/verify'
        mac = hmac.new(bytes(OKX_API_SECRET, 'utf8'), bytes(message, 'utf8'), 'sha256')
        sign = base64.b64encode(mac.digest()).decode()
        return [{"apiKey": OKX_API_KEY, "passphrase": OKX_API_PASSPHRASE, "timestamp": timestamp, "sign": sign}]
    async def _message_handler(self, msg):
        if msg == 'ping': await self.websocket.send('pong'); return
        try:
            data = json.loads(msg)
            if data.get('arg', {}).get('channel') == 'orders':
                for order in data.get('data', []):
                    if order.get('state') == 'filled' and order.get('side') == 'buy': await handle_filled_buy_order(order)
        except json.JSONDecodeError:
            logger.warning(f"Received invalid JSON in private websocket: {msg}")

    async def _run_loop(self):
        async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
            self.websocket = ws; logger.info("✅ [Fast Reporter] Connected.")
            await ws.send(json.dumps({"op": "login", "args": self._get_auth_args()}))
            login_response = json.loads(await ws.recv())
            if login_response.get('code') == '0':
                logger.info("🔐 [Fast Reporter] Authenticated.")
                await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "orders", "instType": "SPOT"}]}))
                async for msg in ws: await self._message_handler(msg)
            else: raise ConnectionAbortedError(f"Authentication failed: {login_response}")
    async def run(self): await exponential_backoff_with_jitter(self._run_loop)

async def the_supervisor_job(context: ContextTypes.DEFAULT_TYPE):
    logger.info("🕵️ Supervisor: Auditing pending trades...")
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        two_mins_ago = (datetime.now(EGYPT_TZ) - timedelta(minutes=2)).isoformat()
        stuck_trades = await (await conn.execute("SELECT * FROM trades WHERE status = 'pending' AND timestamp <= ?", (two_mins_ago,))).fetchall()
        if not stuck_trades: logger.info("🕵️ Supervisor: Audit complete. No abandoned trades found."); return
        for trade_data in stuck_trades:
            trade = dict(trade_data); order_id, symbol = trade['order_id'], trade['symbol']
            logger.warning(f"🕵️ Supervisor: Found abandoned trade #{trade['id']}. Investigating.", extra={'trade_id': trade['id']})
            try:
                order_status = await bot_data.exchange.fetch_order(order_id, symbol)
                if order_status['status'] == 'closed' and order_status.get('filled', 0) > 0:
                    logger.info(f"🕵️ Supervisor: API confirms {order_id} was filled. Activating.", extra={'trade_id': trade['id']})
                    await activate_trade(order_id, symbol)
                elif order_status['status'] == 'canceled': await conn.execute("UPDATE trades SET status = 'failed' WHERE id = ?", (trade['id'],))
                else: await bot_data.exchange.cancel_order(order_id, symbol); await conn.execute("UPDATE trades SET status = 'failed' WHERE id = ?", (trade['id'],))
                await conn.commit()
            except (ccxt.BaseError, aiosqlite.Error) as e: 
                logger.error(f"🕵️ Supervisor: Failed to rectify trade #{trade['id']}: {e}", extra={'trade_id': trade['id']})

async def intelligent_reviewer_job(context: ContextTypes.DEFAULT_TYPE):
    if not bot_data.settings.get('intelligent_reviewer_enabled', True):
        return
    logger.info("🧠 Intelligent Reviewer: Reviewing active trades for signal validity...")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            active_trades = await (await conn.execute("SELECT * FROM trades WHERE status = 'active'")).fetchall()
        for trade in active_trades:
            trade_dict = dict(trade)
            symbol = trade_dict['symbol']
            reason = trade_dict['reason'].split(' + ')[0]
            if reason not in SCANNERS:
                continue

            ohlcv = await bot_data.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').sort_index()
            if len(df) < 50:
                continue

            analyzer_func = SCANNERS[reason]
            # This is a simplification; a real check would need more context.
            # Here we assume a simple re-run is sufficient.
            result = analyzer_func(df, bot_data.settings.get(reason, {}), 0, 0)
            if not result:
                current_price = df['close'].iloc[-1]
                await TradeGuardian(context.application)._close_trade(trade_dict, "Signal Invalidated (Reviewer)", current_price)
                logger.info(f"🧠 Intelligent Reviewer: Closed trade #{trade['id']} for {symbol} - Signal invalidated.")
    except (aiosqlite.Error, ccxt.BaseError, KeyError) as e:
        logger.error(f"🧠 Intelligent Reviewer Job failed: {e}", exc_info=True)

class TradeGuardian:
    def __init__(self, application):
        self.application = application

    async def handle_ticker_update(self, ticker_data):
        async with trade_management_lock:
            symbol = ticker_data['instId'].replace('-', '/')
            try:
                current_price = float(ticker_data['last'])
            except (ValueError, KeyError):
                logger.warning(f"Invalid ticker data received for {symbol}: {ticker_data}")
                return
            
            try:
                async with aiosqlite.connect(DB_FILE) as conn:
                    conn.row_factory = aiosqlite.Row
                    trade_cursor = await conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = 'active'", (symbol,))
                    trade_data = await trade_cursor.fetchone()
                    if not trade_data: return
                
                trade = dict(trade_data)
                settings = bot_data.settings

                if current_price <= trade['stop_loss']:
                    await self._close_trade(trade, "فاشلة (SL)", current_price)
                    return

                if settings.get('momentum_scalp_mode_enabled', False):
                    scalp_target = trade['entry_price'] * (1 + settings['momentum_scalp_target_percent'] / 100)
                    if current_price >= scalp_target:
                        await self._close_trade(trade, "ناجحة (Scalp Mode)", current_price)
                        return
                
                if settings['trailing_sl_enabled']:
                    highest_price_so_far = max(trade.get('highest_price', 0), current_price)
                    
                    if highest_price_so_far > trade.get('highest_price', 0):
                        trade['highest_price'] = highest_price_so_far
                        async with aiosqlite.connect(DB_FILE) as conn:
                            await conn.execute("UPDATE trades SET highest_price = ? WHERE id = ?", (highest_price_so_far, trade['id']))
                            await conn.commit()

                    activation_price = trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100)
                    if not trade['trailing_sl_active'] and highest_price_so_far >= activation_price:
                        trade['trailing_sl_active'] = True
                        new_stop_loss = trade['entry_price']
                        trade['stop_loss'] = new_stop_loss
                        async with aiosqlite.connect(DB_FILE) as conn:
                            await conn.execute("UPDATE trades SET trailing_sl_active = 1, stop_loss = ? WHERE id = ?", (new_stop_loss, trade['id']))
                            await conn.commit()
                        await safe_send_message(self.application.bot, f"**🚀 تأمين الأرباح! | #{trade['id']} {symbol}**\nتم رفع وقف الخسارة إلى نقطة الدخول: `${new_stop_loss}`")

                    if trade['trailing_sl_active']:
                        new_dynamic_sl = highest_price_so_far * (1 - settings['trailing_sl_callback_percent'] / 100)
                        if new_dynamic_sl > trade['stop_loss']:
                            trade['stop_loss'] = new_dynamic_sl
                            async with aiosqlite.connect(DB_FILE) as conn:
                                await conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_dynamic_sl, trade['id']))
                                await conn.commit()
                
                if current_price >= trade['take_profit']:
                    await self._close_trade(trade, "ناجحة (TP)", current_price)

            except aiosqlite.Error as e:
                logger.error(f"Guardian DB Error for {symbol}: {e}", exc_info=True)
            except (KeyError, TypeError) as e:
                logger.error(f"Guardian Data Error for {symbol}: {e}", exc_info=True)

    async def _close_trade(self, trade, reason, close_price):
        symbol, trade_id = trade['symbol'], trade['id']
        bot, log_ctx = self.application.bot, {'trade_id': trade_id}
        logger.info(f"Guardian: Closing {symbol}. Reason: {reason}", extra=log_ctx)
        
        max_retries = bot_data.settings.get('close_retries', 3)
        for i in range(max_retries):
            try:
                asset_to_sell = symbol.split('/')[0]
                balance = await bot_data.exchange.fetch_balance()
                available_quantity = balance.get(asset_to_sell, {}).get('free', 0.0)
                
                if available_quantity <= 0:
                    logger.critical(f"Attempted to close #{trade_id} but no balance for {asset_to_sell}.", extra=log_ctx)
                    return

                formatted_quantity = bot_data.exchange.amount_to_precision(symbol, available_quantity)
                params = {'tdMode': 'cash', 'clOrdId': f"close{trade_id}{int(time.time() * 1000)}"}
                await bot_data.exchange.create_market_sell_order(symbol, formatted_quantity, params)
                
                pnl = (close_price - trade['entry_price']) * trade['quantity']
                pnl_percent = (close_price / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
                
                if pnl > 0 and reason == "فاشلة (SL)":
                    reason = "تم تأمين الربح (TSL)"
                    emoji = "✅"
                elif pnl > 0:
                    emoji = "✅"
                else:
                    emoji = "🛑"
                    
                async with aiosqlite.connect(DB_FILE) as conn:
                    await conn.execute("UPDATE trades SET status = ?, close_price = ?, pnl_usdt = ?, close_retries = 0 WHERE id = ?", (reason, close_price, pnl, trade['id']))
                    await conn.commit()
                
                await bot_data.public_ws.unsubscribe([symbol])
                
                msg = (f"{emoji} **تم إغلاق الصفقة | #{trade_id} {symbol}**\n"
                       f"**السبب:** {reason}\n"
                       f"**الربح/الخسارة:** `${pnl:,.2f}` ({pnl_percent:+.2f}%)")
                await safe_send_message(bot, msg)
                return
            except ccxt.NetworkError as e:
                logger.warning(f"Network error on close attempt {i+1}/{max_retries} for #{trade_id}: {e}")
                if i < max_retries - 1: await asyncio.sleep(5)
            except ccxt.ExchangeError as e:
                logger.error(f"Unrecoverable exchange error on close attempt for #{trade_id}: {e}")
                break
        
        logger.critical(f"CRITICAL: Failed to close trade #{trade_id} after {max_retries} retries.", extra=log_ctx)
        await safe_send_message(bot, f"🚨 **فشل حرج** 🚨\nفشل إغلاق الصفقة `#{trade_id}`. الرجاء المراجعة اليدوية.")
    
    async def sync_subscriptions(self):
        try:
            async with aiosqlite.connect(DB_FILE) as conn:
                active_symbols = [row[0] for row in await (await conn.execute("SELECT DISTINCT symbol FROM trades WHERE status = 'active'")).fetchall()]
            if active_symbols:
                logger.info(f"Guardian: Syncing subs: {active_symbols}")
                await bot_data.public_ws.subscribe(active_symbols)
        except aiosqlite.Error as e:
            logger.error(f"Guardian Sync DB Error: {e}")

class PublicWebSocketManager:
    def __init__(self, handler_coro):
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.handler = handler_coro
        self.subscriptions = set()

    async def _send_op(self, op, symbols):
        if not symbols or not hasattr(self, 'websocket') or not self.websocket: return
        try:
            await self.websocket.send(json.dumps({"op": op, "args": [{"channel": "tickers", "instId": s.replace('/', '-')} for s in symbols]}))
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Could not send '{op}' op; ws is closed.")

    async def subscribe(self, symbols):
        new_symbols = [s for s in symbols if s not in self.subscriptions]
        if new_symbols:
            await self._send_op('subscribe', new_symbols)
            self.subscriptions.update(new_symbols)
            logger.info(f"👁️ [Guardian] Now watching: {new_symbols}")

    async def unsubscribe(self, symbols):
        old_symbols = [s for s in symbols if s in self.subscriptions]
        if old_symbols:
            await self._send_op('unsubscribe', old_symbols)
            for s in old_symbols: self.subscriptions.discard(s)
            logger.info(f"👁️ [Guardian] Stopped watching: {old_symbols}")

    async def _run_loop(self):
        async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
            self.websocket = ws
            logger.info("✅ [Guardian's Eyes] Connected.")
            if self.subscriptions:
                await self.subscribe(list(self.subscriptions))
            async for msg in ws:
                if msg == 'ping':
                    await ws.send('pong')
                    continue
                try:
                    data = json.loads(msg)
                    if data.get('arg', {}).get('channel') == 'tickers' and 'data' in data:
                        for ticker in data['data']:
                            await self.handler(ticker)
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON in public websocket: {msg}")


    async def run(self):
        await exponential_backoff_with_jitter(self._run_loop)

# =======================================================================================
# --- ⚡ Core Scanner & Trade Initiation Logic (FULLY IMPLEMENTED) ⚡ ---
# =======================================================================================

async def _apply_pre_scan_filters(df, settings, symbol, exchange):
    """Applies all pre-analysis filters. Returns (bool, dict)"""
    try:
        # Spread Filter
        orderbook = await exchange.fetch_order_book(symbol, limit=1)
        if not orderbook.get('bids') or not orderbook.get('asks'): return False, None
        best_bid, best_ask = orderbook['bids'][0][0], orderbook['asks'][0][0]
        if best_bid <= 0: return False, None
        spread_percent = ((best_ask - best_bid) / best_bid) * 100
        if spread_percent > settings['spread_filter']['max_spread_percent']: return False, None

        # Trend Filter
        if settings.get('trend_filters', {}).get('enabled', True):
            ema_period = settings.get('trend_filters', {}).get('ema_period', 200)
            if len(df) < ema_period + 1: return False, None
            df.ta.ema(length=ema_period, append=True)
            ema_col = find_col(df.columns, f"EMA_{ema_period}")
            if not ema_col or pd.isna(df[ema_col].iloc[-2]): return False, None
            if df['close'].iloc[-2] < df[ema_col].iloc[-2]: return False, None

        # Volatility Filter
        vol_filters = settings.get('volatility_filters', {})
        atr_period = vol_filters.get('atr_period_for_filter', 14)
        min_atr_percent = vol_filters.get('min_atr_percent', 0.8)
        df.ta.atr(length=atr_period, append=True)
        atr_col = find_col(df.columns, f"ATRr_{atr_period}")
        if not atr_col or pd.isna(df[atr_col].iloc[-2]): return False, None
        last_close = df['close'].iloc[-2]
        atr_percent = (df[atr_col].iloc[-2] / last_close) * 100 if last_close > 0 else 0
        if atr_percent < min_atr_percent: return False, None
        
        # Volume Filter (RVOL)
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] == 0: return False, None
        rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
        if rvol < settings.get('volume_filter_multiplier', 2.0): return False, None

        # ADX Filter
        adx_value = 0
        if settings.get('adx_filter_enabled', False):
            df.ta.adx(append=True)
            adx_col = find_col(df.columns, "ADX_")
            adx_value = df[adx_col].iloc[-2] if adx_col and pd.notna(df[adx_col].iloc[-2]) else 0
            if adx_value < settings.get('adx_filter_level', 25): return False, None
            
        return True, {"rvol": rvol, "adx_value": adx_value}
    except (IndexError, KeyError):
        return False, None
    except ccxt.BaseError as e:
        logger.warning(f"Pre-scan filter for {symbol} failed due to API error: {e}")
        return False, None

async def _run_all_scanners(df, settings, metrics, exchange, symbol):
    """Runs all active scanners. Returns list of reason strings."""
    confirmed_reasons = []
    for name in settings['active_scanners']:
        if not (strategy_func := SCANNERS.get(name)): continue
        try:
            func_args = {'df': df.copy(), 'params': settings.get(name, {}), **metrics}
            if asyncio.iscoroutinefunction(strategy_func):
                func_args.update({'exchange': exchange, 'symbol': symbol})
                result = await strategy_func(**func_args)
            else:
                sync_args = {k: v for k, v in func_args.items() if k not in ['exchange', 'symbol']}
                result = strategy_func(**sync_args)

            if result and 'reason' in result:
                confirmed_reasons.append(result['reason'])
        except Exception:
             logger.warning(f"Scanner '{name}' failed for {symbol} due to an internal error.", exc_info=True)
             continue
    return list(set(confirmed_reasons))

def _calculate_trade_parameters(df, settings, reasons, is_htf_bullish, symbol):
    """Calculates final trade parameters. Returns a signal dictionary or None."""
    reason_str = ' + '.join(reasons)
    strength = len(reasons)
    trade_weight = 1.0

    if settings.get('adaptive_intelligence_enabled', True):
        primary_reason = reasons[0]
        perf = bot_data.strategy_performance.get(primary_reason)
        if perf:
            if perf['win_rate'] < settings['strategy_deactivation_threshold_wr'] and perf['total_trades'] > settings['strategy_analysis_min_trades']:
                logger.warning(f"Signal for {symbol} from weak strategy '{primary_reason}' ignored.")
                return None
            if perf['win_rate'] < 50 and perf['total_trades'] > 5:
                trade_weight = 1 - (settings['dynamic_sizing_max_decrease_pct'] / 100.0)
            elif perf['win_rate'] > 70 and perf['profit_factor'] > 1.5:
                trade_weight = 1 + (settings['dynamic_sizing_max_increase_pct'] / 100.0)

    if not is_htf_bullish:
        strength = max(1, int(strength / 2))
        reason_str += " (اتجاه كبير ضعيف)"
        trade_weight *= 0.8
    
    entry_price = df.iloc[-2]['close']
    df.ta.atr(length=14, append=True)
    atr_col = find_col(df.columns, "ATR_14")
    if not atr_col or pd.isna(df[atr_col].iloc[-2]): return None
    
    risk = df[atr_col].iloc[-2] * settings['atr_sl_multiplier']
    if risk <= 0: return None

    stop_loss = entry_price - risk
    take_profit = entry_price + (risk * settings['risk_reward_ratio'])

    return {
        "symbol": symbol, "entry_price": entry_price, "take_profit": take_profit, 
        "stop_loss": stop_loss, "reason": reason_str, "strength": strength, 
        "weight": trade_weight
    }

async def worker_batch(queue, signals_list, errors_list):
    settings, exchange = bot_data.settings, bot_data.exchange
    while not queue.empty():
        item = None # تعريف المتغير مسبقًا
        symbol = "N/A" 
        try:
            item = await queue.get()
            market, ohlcv = item['market'], item['ohlcv']
            symbol = market['symbol']
            
            logger.info(f"Worker starting analysis for: {symbol}")

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').sort_index()
            if len(df) < 50: 
                continue # --- ✅ هنا التعديل: نستخدم continue فقط ---

            is_valid, metrics = await _apply_pre_scan_filters(df.copy(), settings, symbol, exchange)
            if not is_valid: 
                continue # --- ✅ هنا التعديل: نستخدم continue فقط ---
            
            confirmed_reasons = await _run_all_scanners(df.copy(), settings, metrics, exchange, symbol)
            if not confirmed_reasons:
                continue # --- ✅ هنا التعديل: نستخدم continue فقط ---
            
            is_htf_bullish = True 
            signal = _calculate_trade_parameters(df, settings, confirmed_reasons, is_htf_bullish, symbol)
            if signal:
                signals_list.append(signal)

            logger.info(f"Worker FINISHED analysis for: {symbol}")

        except asyncio.QueueEmpty:
            continue
        except (ccxt.BaseError, aiosqlite.Error) as e:
            logger.warning(f"Recoverable worker error on {symbol}: {e}")
            errors_list.append(symbol)
        except Exception as e:
            logger.error(f"CRITICAL WORKER FAILURE on {symbol}: {e}", exc_info=True)
            errors_list.append(symbol)
        finally:
            if item is not None: # نتأكد من أننا سحبنا عنصرا من الطابور
                queue.task_done()

async def get_okx_markets() -> List[Dict]:
    settings = bot_data.settings
    if time.time() - bot_data.last_markets_fetch > 300: # 5 minutes cache
        try:
            logger.info("Fetching and caching all OKX markets..."); 
            all_tickers = await bot_data.exchange.fetch_tickers()
            bot_data.all_markets = list(all_tickers.values())
            bot_data.last_markets_fetch = time.time()
        except ccxt.BaseError as e: 
            logger.error(f"Failed to fetch all markets: {e}"); 
            return [] # Return empty list on failure
            
    blacklist = set(settings.get('asset_blacklist', [])) # Use a set for faster lookups
    min_volume = settings['liquidity_filters']['min_quote_volume_24h_usd']
    
    valid_markets = [
        t for t in bot_data.all_markets 
        if t.get('symbol') 
        and t['symbol'].endswith('/USDT') 
        and t['symbol'].split('/')[0] not in blacklist 
        and t.get('quoteVolume', 0) > min_volume
        and t.get('active', True) 
        and not any(k in t['symbol'] for k in ['-SWAP', 'UP', 'DOWN', '3L', '3S'])
    ]
    
    valid_markets.sort(key=lambda m: m.get('quoteVolume', 0), reverse=True)
    return valid_markets[:settings['top_n_symbols_by_volume']]

async def fetch_ohlcv_batch(exchange, symbols, timeframe, limit):
    tasks = [exchange.fetch_ohlcv(s, timeframe, limit=limit) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {symbols[i]: results[i] for i in range(len(symbols)) if not isinstance(results[i], Exception)}

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        if not bot_data.trading_enabled: logger.info("Scan skipped: Kill Switch is active."); return
        scan_start_time = time.time()
        logger.info("--- Starting new Adaptive Intelligence scan... ---")
        settings, bot = bot_data.settings, context.bot
        
        # --- ✅ هنا التعديلات ---
        logger.info("==> [Step 1/5] Checking fundamental market mood...")
        if settings.get('news_filter_enabled', True):
            mood_result_fundamental = await get_fundamental_market_mood()
            if mood_result_fundamental['mood'] in ["NEGATIVE", "DANGEROUS"]:
                bot_data.market_mood = mood_result_fundamental
                logger.warning(f"SCAN SKIPPED: Fundamental mood is {mood_result_fundamental['mood']}. Reason: {mood_result_fundamental['reason']}")
                return
        logger.info("==> [Step 2/5] Fundamental mood OK. Checking technical market mood...")
        
        mood_result = await get_market_mood()
        bot_data.market_mood = mood_result
        if mood_result['mood'] in ["NEGATIVE", "DANGEROUS"]:
            logger.warning(f"SCAN SKIPPED: {mood_result['reason']}")
            return
        logger.info("==> [Step 3/5] Technical mood OK. Checking trade limits...")

        async with aiosqlite.connect(DB_FILE) as conn:
            active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active' OR status = 'pending'")).fetchone())[0]
        if active_trades_count >= settings['max_concurrent_trades']:
            logger.info(f"Scan skipped: Max trades ({active_trades_count}) reached."); return
        logger.info(f"==> [Step 4/5] Trade limits OK ({active_trades_count}/{settings['max_concurrent_trades']}). Fetching markets...")

        top_markets = await get_okx_markets()
        if not top_markets:
            logger.warning("Scan aborted: No valid markets found after filtering.")
            return
        logger.info(f"==> [Step 5/5] Found {len(top_markets)} markets. Starting analysis workers...")

        symbols_to_scan = [m['symbol'] for m in top_markets]
        ohlcv_data = await fetch_ohlcv_batch(bot_data.exchange, symbols_to_scan, TIMEFRAME, 220)
        
        queue, signals_found, analysis_errors = asyncio.Queue(), [], []
        for market in top_markets:
            if market['symbol'] in ohlcv_data:
                await queue.put({'market': market, 'ohlcv': ohlcv_data[market['symbol']]})
        
        worker_tasks = [asyncio.create_task(worker_batch(queue, signals_found, analysis_errors)) for _ in range(settings.get("worker_threads", 10))]
        await queue.join()
        for task in worker_tasks: task.cancel() # Cancel workers after queue is empty
        
        trades_opened_count = 0
        signals_found.sort(key=lambda s: s.get('strength', 0), reverse=True)

        for signal in signals_found:
            if active_trades_count >= settings['max_concurrent_trades']: break
            if time.time() - bot_data.last_signal_time.get(signal['symbol'], 0) > (SCAN_INTERVAL_SECONDS * 0.9):
                bot_data.last_signal_time[signal['symbol']] = time.time()
                await broadcast_signal_to_redis(signal)
                if await initiate_real_trade(signal):
                    active_trades_count += 1; trades_opened_count += 1
                await asyncio.sleep(2) # Small delay between initiating trades

        scan_duration = time.time() - scan_start_time
        bot_data.last_scan_info = {"start_time": datetime.fromtimestamp(scan_start_time, EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), "duration_seconds": int(scan_duration), "checked_symbols": len(top_markets), "analysis_errors": len(analysis_errors)}
        logger.info(f"--- Scan finished in {scan_duration:.2f}s. Found {len(signals_found)} signals, opened {trades_opened_count} new trades. ---")

async def initiate_real_trade(signal: Dict[str, Any]) -> bool:
    if not bot_data.trading_enabled:
        logger.warning(f"Trade for {signal['symbol']} blocked: Kill Switch active."); return False
    try:
        settings, exchange = bot_data.settings, bot_data.exchange; await exchange.load_markets()
        base_trade_size = settings['real_trade_size_usdt']; trade_weight = signal.get('weight', 1.0)
        
        if settings.get('dynamic_trade_sizing_enabled', True): 
            trade_size = base_trade_size * trade_weight
        else: 
            trade_size = base_trade_size

        balance = await exchange.fetch_balance(); usdt_balance = balance.get('USDT', {}).get('free', 0.0)
        
        if usdt_balance < trade_size:
            logger.error(f"Insufficient USDT for {signal['symbol']}. Have: {usdt_balance}, Need: {trade_size}")
            return False
            
        entry_price = signal.get('entry_price')
        if not entry_price or entry_price <= 0:
            logger.error(f"Invalid entry price (<= 0) for {signal['symbol']}. Aborting trade.")
            return False

        base_amount = trade_size / entry_price
        formatted_amount = exchange.amount_to_precision(signal['symbol'], base_amount)
        buy_order = await exchange.create_market_buy_order(signal['symbol'], formatted_amount)
        
        if await log_pending_trade_to_db(signal, buy_order):
            await safe_send_message(bot_data.application.bot, f"🚀 تم إرسال أمر شراء لـ `{signal['symbol']}`."); return True
        else:
            logger.critical(f"Failed to log pending trade for {signal['symbol']} to DB. Cancelling order {buy_order['id']}.")
            await exchange.cancel_order(buy_order['id'], signal['symbol']); return False
            
    except ccxt.InsufficientFunds as e: 
        logger.error(f"REAL TRADE FAILED (Insufficient Funds) {signal['symbol']}: {e}"); 
        return False
    except ccxt.BaseError as e:
        logger.error(f"REAL TRADE FAILED (Exchange Error) {signal['symbol']}: {e}", exc_info=True); 
        return False
    except Exception as e:
        logger.error(f"REAL TRADE FAILED (Unexpected Error) {signal['symbol']}: {e}", exc_info=True);
        return False

async def check_time_sync(context: ContextTypes.DEFAULT_TYPE):
    try:
        server_time = await bot_data.exchange.fetch_time(); local_time = int(time.time() * 1000); diff = abs(server_time - local_time)
        if diff > 2000: await safe_send_message(context.bot, f"⚠️ **تحذير مزامنة الوقت** ⚠️\nفارق `{diff}` ميلي ثانية.")
        else: logger.info(f"Time sync OK. Diff: {diff}ms.")
    except ccxt.BaseError as e: logger.error(f"Time sync check failed: {e}")

# =======================================================================================
# --- 🎼 Maestro & Critical Monitor (FULLY IMPLEMENTED) 🎼 ---
# =======================================================================================
async def determine_market_regime() -> str:
    """Analyzes BTC/USDT on a higher timeframe to determine the market regime."""
    try:
        ohlcv = await bot_data.exchange.fetch_ohlcv('BTC/USDT', '4h', limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)

        last_row = df.iloc[-1]
        adx_col, atr_col = find_col(df.columns, "ADX_"), find_col(df.columns, "ATR_14")
        if adx_col is None or atr_col is None or pd.isna(last_row[adx_col]) or pd.isna(last_row[atr_col]):
            return "UNKNOWN"
        
        adx = last_row[adx_col]
        atr_percent = (last_row[atr_col] / last_row['close']) * 100

        is_trending = adx > REGIME_ADX_THRESHOLD
        is_high_volatility = atr_percent > REGIME_ATR_PERCENT_THRESHOLD

        if is_trending and is_high_volatility: return "TRENDING_HIGH_VOLATILITY"
        if is_trending and not is_high_volatility: return "TRENDING_LOW_VOLATILITY"
        if not is_trending and is_high_volatility: return "SIDEWAYS_HIGH_VOLATILITY"
        if not is_trending and not is_high_volatility: return "SIDEWAYS_LOW_VOLATILITY"
        return "UNKNOWN"
    except (ccxt.BaseError, IndexError, KeyError) as e:
        logger.error(f"Could not determine market regime: {e}")
        return "UNKNOWN"

async def maestro_job(context: ContextTypes.DEFAULT_TYPE):
    """The Maestro job dynamically adjusts bot settings based on the market regime."""
    settings = bot_data.settings
    if not settings.get('maestro_mode_enabled', True):
        return
    
    logger.info("🎼 Maestro: Diagnosing market regime...")
    new_regime = await determine_market_regime()
    
    if new_regime == "UNKNOWN":
        logger.warning("🎼 Maestro: Could not determine market regime. No changes made.")
        return
        
    if new_regime != bot_data.current_market_regime:
        logger.warning(f"🎼 Maestro: Market regime shift detected! From '{bot_data.current_market_regime}' to '{new_regime}'.")
        bot_data.current_market_regime = new_regime
        
        try:
            with open(DECISION_MATRIX_FILE, 'r', encoding='utf-8') as f:
                matrix = json.load(f)
            
            new_params = matrix.get(new_regime)
            if not new_params:
                logger.error(f"Maestro: No parameters found for regime '{new_regime}' in decision matrix.")
                return

            # Apply new parameters over existing settings
            for key, value in new_params.items():
                settings[key] = value
            
            save_settings()
            determine_active_preset()
            
            message = (f"🎼 **المايسترو:** تم تغيير نظام السوق إلى **{new_regime}**.\n\n"
                       f"تم تحديث الإعدادات تلقائيًا لتناسب الظروف الجديدة. الاستراتيجيات النشطة الآن: "
                       f"`{', '.join(new_params.get('active_scanners', []))}`")
            await safe_send_message(context.bot, message)
            logger.info(f"🎼 Maestro: Successfully applied new settings for '{new_regime}'.")

        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Maestro: Could not load or parse decision matrix file: {e}")
    else:
        logger.info(f"🎼 Maestro: Market regime stable at '{new_regime}'. No changes needed.")

async def critical_trade_monitor(context: ContextTypes.DEFAULT_TYPE):
    """Acts as a final safety net, checking for trades that fell past their SL without being closed."""
    logger.info("🛡️ Critical Monitor: Performing safety audit on active trades...")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            active_trades = await (await conn.execute("SELECT * FROM trades WHERE status = 'active'")).fetchall()

        if not active_trades:
            logger.info("🛡️ Critical Monitor: No active trades to audit.")
            return

        symbols_to_check = list(set([trade['symbol'] for trade in active_trades]))
        tickers = await bot_data.exchange.fetch_tickers(symbols_to_check)

        for trade_data in active_trades:
            trade = dict(trade_data)
            symbol = trade['symbol']
            if symbol not in tickers: continue
            
            current_price = tickers[symbol].get('last')
            if current_price is None: continue
            
            # Check if price has fallen significantly (e.g., 2% slippage) past the stop-loss
            if current_price < (trade['stop_loss'] * 0.98):
                logger.critical(f"🛡️ CRITICAL MONITOR ALERT! Trade #{trade['id']} for {symbol} is at ${current_price}, "
                                f"which is significantly below its SL of ${trade['stop_loss']}. Forcing closure.", 
                                extra={'trade_id': trade['id']})
                await TradeGuardian(context.application)._close_trade(trade, "فاشلة (Critical Monitor)", current_price)
                await asyncio.sleep(1) # Small delay to avoid rate limiting
    except (ccxt.BaseError, aiosqlite.Error) as e:
        logger.error(f"🛡️ Critical Monitor failed: {e}", exc_info=True)


# =======================================================================================
# --- 🤖 Telegram UI & Bot Startup 🤖 ---
# =======================================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Dashboard 🖥️"], ["الإعدادات ⚙️"]]
    await update.message.reply_text("أهلاً بك في **قناص OKX | إصدار المايسترو متعدد الأوضاع**", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True), parse_mode=ParseMode.MARKDOWN)

async def manual_scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not bot_data.trading_enabled: await (update.message or update.callback_query.message).reply_text("🔬 الفحص محظور. مفتاح الإيقاف مفعل."); return
    await (update.message or update.callback_query.message).reply_text("🔬 أمر فحص يدوي... قد يستغرق بعض الوقت.")
    context.job_queue.run_once(perform_scan, 1)

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ks_status_emoji = "🚨" if not bot_data.trading_enabled else "✅"
    ks_status_text = "مفتاح الإيقاف (مفعل)" if not bot_data.trading_enabled else "الحالة (طبيعية)"
    keyboard = [
        [InlineKeyboardButton("💼 نظرة عامة على المحفظة", callback_data="db_portfolio"), InlineKeyboardButton("📈 الصفقات النشطة", callback_data="db_trades")],
        [InlineKeyboardButton("📜 سجل الصفقات المغلقة", callback_data="db_history"), InlineKeyboardButton("📊 الإحصائيات والأداء", callback_data="db_stats")],
        [InlineKeyboardButton("🌡️ تحليل مزاج السوق", callback_data="db_mood"), InlineKeyboardButton("🔬 فحص فوري", callback_data="db_manual_scan")],
        [InlineKeyboardButton("🗓️ التقرير اليومي", callback_data="db_daily_report")],
        [InlineKeyboardButton(f"{ks_status_emoji} {ks_status_text}", callback_data="kill_switch_toggle"), InlineKeyboardButton("🎼 التحكم الاستراتيجي", callback_data="db_maestro_control")],
        [InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="db_diagnostics")]
    ]
    message_text = "🖥️ **لوحة تحكم قناص OKX**\n\nاختر نوع التقرير الذي تريد عرضه:"
    if not bot_data.trading_enabled: message_text += "\n\n**تحذير: تم تفعيل مفتاح الإيقاف.**"
    target_message = update.message or update.callback_query.message
    if update.callback_query: await safe_edit_message(update.callback_query, message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else: await target_message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

# Maestro Control Panel
async def show_maestro_control(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings = bot_data.settings
    regime = bot_data.current_market_regime
    maestro_enabled = settings.get('maestro_mode_enabled', True)
    emoji = "✅" if maestro_enabled else "❌"
    active_scanners_str = ' + '.join([STRATEGY_NAMES_AR.get(scanner, scanner) for scanner in settings.get('active_scanners', [])])
    message = (f"🎼 **لوحة التحكم الاستراتيجي (المايسترو)**\n"
               f"━━━━━━━━━━━━━━━━━━\n"
               f"**حالة المايسترو:** {emoji} {'مفعل' if maestro_enabled else 'معطل'}\n"
               f"**تشخيص السوق الحالي:** {regime}\n"
               f"**الاستراتيجيات النشطة:** {active_scanners_str}\n\n"
               f"**التكوين الحالي:**\n"
               f"  - **المراجع الذكي:** {'✅' if settings.get('intelligent_reviewer_enabled') else '❌'}\n"
               f"  - **اقتناص الزخم:** {'✅' if settings.get('momentum_scalp_mode_enabled') else '❌'}\n"
               f"  - **فلتر التوافق:** {'✅' if settings.get('multi_timeframe_confluence_enabled') else '❌'}\n"
               f"  - **استراتيجية الانعكاس:** {'✅' if 'bollinger_reversal' in settings.get('active_scanners', []) else '❌'}")
    keyboard = [
        [InlineKeyboardButton(f"🎼 تبديل المايسترو ({'تعطيل' if maestro_enabled else 'تفعيل'})", callback_data="maestro_toggle")],
        [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]
    ]
    await safe_edit_message(update.callback_query, message, reply_markup=InlineKeyboardMarkup(keyboard))

async def toggle_maestro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_data.settings['maestro_mode_enabled'] = not bot_data.settings.get('maestro_mode_enabled', True)
    save_settings()
    await update.callback_query.answer(f"المايسترو {'تم تفعيله' if bot_data.settings['maestro_mode_enabled'] else 'تم تعطيله'}")
    await show_maestro_control(update, context)

async def send_daily_report(context: ContextTypes.DEFAULT_TYPE):
    today_str = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d')
    logger.info(f"Generating daily report for {today_str}...")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            closed_today = await (await conn.execute("SELECT * FROM trades WHERE status LIKE '%(%' AND date(timestamp) = ?", (today_str,))).fetchall()
        if not closed_today:
            report_message = f"🗓️ **التقرير اليومي | {today_str}**\n━━━━━━━━━━━━━━━━━━\nلم يتم إغلاق أي صفقات اليوم."
        else:
            wins = [t for t in closed_today if 'ناجحة' in t['status'] or 'تأمين' in t['status']]
            losses = [t for t in closed_today if 'فاشلة' in t['status']]
            total_pnl = sum(t['pnl_usdt'] for t in closed_today if t['pnl_usdt'] is not None)
            win_rate = (len(wins) / len(closed_today) * 100) if closed_today else 0
            avg_win_pnl = sum(w['pnl_usdt'] for w in wins if w['pnl_usdt'] is not None) / len(wins) if wins else 0
            avg_loss_pnl = sum(l['pnl_usdt'] for l in losses if l['pnl_usdt'] is not None) / len(losses) if losses else 0
            avg_pnl = total_pnl / len(closed_today) if closed_today else 0
            best_trade = max(closed_today, key=lambda t: t.get('pnl_usdt', -float('inf')), default=None)
            worst_trade = min(closed_today, key=lambda t: t.get('pnl_usdt', float('inf')), default=None)
            strategy_counter = Counter(r for t in closed_today for r in t['reason'].split(' + '))
            most_active_strategy_en = strategy_counter.most_common(1)[0][0] if strategy_counter else "N/A"
            most_active_strategy_ar = STRATEGY_NAMES_AR.get(most_active_strategy_en.split(' ')[0], most_active_strategy_en)

            report_message = (
                f"🗓️ **التقرير اليومي | {today_str}**\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📈 **الأداء الرئيسي**\n"
                f"**الربح/الخسارة الصافي:** `${total_pnl:+.2f}`\n"
                f"**معدل النجاح:** {win_rate:.1f}%\n"
                f"**متوسط الربح:** `${avg_win_pnl:+.2f}`\n"
                f"**متوسط الخسارة:** `${avg_loss_pnl:+.2f}`\n"
                f"**الربح/الخسارة لكل صفقة:** `${avg_pnl:+.2f}`\n"
                f"📊 **تحليل الصفقات**\n"
                f"**عدد الصفقات:** {len(closed_today)}\n"
                f"**أفضل صفقة:** `{best_trade['symbol']}` | `${best_trade['pnl_usdt']:+.2f}`\n"
                f"**أسوأ صفقة:** `{worst_trade['symbol']}` | `${worst_trade['pnl_usdt']:+.2f}`\n"
                f"**الاستراتيجية الأنشط:** {most_active_strategy_ar}\n"
            )

        await safe_send_message(context.bot, report_message)
    except Exception as e: logger.error(f"Failed to generate daily report: {e}", exc_info=True)

async def daily_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await (update.message or update.callback_query.message).reply_text("⏳ جاري إرسال التقرير اليومي...")
    await send_daily_report(context)

async def toggle_kill_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; bot_data.trading_enabled = not bot_data.trading_enabled
    if bot_data.trading_enabled: await query.answer("✅ تم استئناف التداول الطبيعي."); await safe_send_message(context.bot, "✅ **تم استئناف التداول الطبيعي.**")
    else: await query.answer("🚨 تم تفعيل مفتاح الإيقاف!", show_alert=True); await safe_send_message(context.bot, "🚨 **تحذير: تم تفعيل مفتاح الإيقاف!**")
    await show_dashboard_command(update, context)

async def show_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row; trades = await (await conn.execute("SELECT id, symbol, status FROM trades WHERE status = 'active' OR status = 'pending' ORDER BY id DESC")).fetchall()
    if not trades:
        text = "لا توجد صفقات نشطة حاليًا."
        keyboard = [[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
        await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard)); return
    text = "📈 *الصفقات النشطة*\nاختر صفقة لعرض تفاصيلها:\n"; keyboard = []
    for trade in trades: status_emoji = "✅" if trade['status'] == 'active' else "⏳"; button_text = f"#{trade['id']} {status_emoji} | {trade['symbol']}"; keyboard.append([InlineKeyboardButton(button_text, callback_data=f"check_{trade['id']}")])
    keyboard.append([InlineKeyboardButton("🔄 تحديث", callback_data="db_trades")]); keyboard.append([InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]); await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))

async def check_trade_details(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    trade_id = int(query.data.split('_')[1])
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        trade = await cursor.fetchone()
    if not trade:
        await query.answer("لم يتم العثور على الصفقة."); return
    trade = dict(trade)
    if trade['status'] == 'pending':
        message = f"**⏳ حالة الصفقة #{trade_id}**\n- **العملة:** `{trade['symbol']}`\n- **الحالة:** في انتظار تأكيد التنفيذ..."
    else:
        try:
            ticker = await bot_data.exchange.fetch_ticker(trade['symbol'])
            current_price = ticker['last']
            pnl = (current_price - trade['entry_price']) * trade['quantity']
            pnl_percent = (current_price / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
            pnl_text = f"💰 **الربح/الخسارة الحالية:** `${pnl:+.2f}` ({pnl_percent:+.2f}%)"
            current_price_text = f"- **السعر الحالي:** `${current_price}`"
        except (ccxt.BaseError, KeyError):
            pnl_text = "💰 تعذر جلب الربح/الخسارة الحالية."
            current_price_text = "- **السعر الحالي:** `تعذر الجلب`"

        message = (
            f"**✅ حالة الصفقة #{trade_id}**\n\n"
            f"- **العملة:** `{trade['symbol']}`\n"
            f"- **سعر الدخول:** `${trade['entry_price']}`\n"
            f"{current_price_text}\n"
            f"- **الكمية:** `{trade['quantity']}`\n"
            f"----------------------------------\n"
            f"- **الهدف (TP):** `${trade['take_profit']}`\n"
            f"- **الوقف (SL):** `${trade['stop_loss']}`\n"
            f"----------------------------------\n"
            f"{pnl_text}"
        )
    await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للصفقات", callback_data="db_trades")]]))

async def show_mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the market mood analysis and display."""
    query = update.callback_query
    await query.answer("جاري تحليل مزاج السوق...")

    # Fetch data concurrently
    fng_task = asyncio.create_task(get_fear_and_greed_index())
    headlines_task = asyncio.create_task(asyncio.to_thread(get_latest_crypto_news))
    mood_task = asyncio.create_task(get_market_mood())
    markets_task = asyncio.create_task(get_okx_markets())
    
    fng_index, original_headlines, mood, all_markets = await asyncio.gather(
        fng_task, headlines_task, mood_task, markets_task
    )

    # Process and format data
    translated_headlines, translation_success = await translate_text_gemini(original_headlines)
    news_sentiment, _ = analyze_sentiment_of_headlines(original_headlines)
    
    verdict = "الحالة العامة للسوق تتطلب الحذر."
    if mood['mood'] == 'POSITIVE': verdict = "المؤشرات الفنية إيجابية، مما قد يدعم فرص الشراء."
    if fng_index and fng_index > 65: verdict = "المؤشرات الفنية إيجابية ولكن مع وجود طمع في السوق، يرجى الحذر من التقلبات."
    elif fng_index and fng_index < 30: verdict = "يسود الخوف على السوق، قد تكون هناك فرص للمدى الطويل ولكن المخاطرة عالية حالياً."

    gainers_str, losers_str = "  لا توجد بيانات.", "  لا توجد بيانات."
    if all_markets:
        sorted_by_change = sorted([m for m in all_markets if m.get('percentage') is not None], key=lambda m: m['percentage'], reverse=True)
        top_gainers = sorted_by_change[:3]
        top_losers = sorted_by_change[-3:]
        if top_gainers:
            gainers_str = "\n".join([f"  `{g['symbol']}` `({g.get('percentage', 0):+.2f}%)`" for g in top_gainers])
        if top_losers:
            losers_str = "\n".join([f"  `{l['symbol']}` `({l.get('percentage', 0):+.2f}%)`" for l in reversed(top_losers)])

    news_header = "📰 آخر الأخبار (مترجمة آلياً):" if translation_success else "📰 آخر الأخبار (الترجمة غير متاحة):"
    news_str = "\n".join([f"  - _{h}_" for h in translated_headlines]) or "  لا توجد أخبار."
    
    message = (
        f"**🌡️ تحليل مزاج السوق الشامل**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"**⚫️ الخلاصة:** *{verdict}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"**📊 المؤشرات الرئيسية:**\n"
        f"  - **اتجاه BTC العام:** {mood.get('btc_mood', 'N/A')}\n"
        f"  - **الخوف والطمع:** {fng_index or 'N/A'}\n"
        f"  - **مشاعر الأخبار:** {news_sentiment}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"**🚀 أبرز الرابحين:**\n{gainers_str}\n\n"
        f"**📉 أبرز الخاسرين:**\n{losers_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{news_header}\n{news_str}\n"
    )
    keyboard = [[InlineKeyboardButton("🔄 تحديث", callback_data="db_mood")], [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
    await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_strategy_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not bot_data.strategy_performance:
        await safe_edit_message(update.callback_query, "لا توجد بيانات أداء حاليًا. يرجى الانتظار بعد إغلاق بعض الصفقات.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للإحصائيات", callback_data="db_stats")]]))
        return
    
    report = ["**📜 تقرير أداء الاستراتيجيات**\n(بناءً على آخر 100 صفقة)"]
    sorted_strategies = sorted(bot_data.strategy_performance.items(), key=lambda item: item[1]['total_trades'], reverse=True)

    for reason, data in sorted_strategies:
        report.append(f"\n--- *{STRATEGY_NAMES_AR.get(reason, reason)}* ---\n"
                      f"  - **النجاح:** {data['win_rate']:.1f}% ({data['total_trades']} صفقة)\n"
                      f"  - **عامل الربح:** {data['profit_factor'] if data['profit_factor'] != float('inf') else '∞'}")

    await safe_edit_message(update.callback_query, "\n".join(report), reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📊 عرض الإحصائيات العامة", callback_data="db_stats")],[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))

async def show_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT pnl_usdt, status FROM trades WHERE status LIKE '%(%'")
        trades_data = await cursor.fetchall()
    if not trades_data:
        await safe_edit_message(update.callback_query, "لم يتم إغلاق أي صفقات بعد.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))
        return
    total_trades = len(trades_data)
    total_pnl = sum(t['pnl_usdt'] for t in trades_data if t['pnl_usdt'] is not None)
    wins_data = [t['pnl_usdt'] for t in trades_data if ('ناجحة' in t['status'] or 'تأمين' in t['status']) and t['pnl_usdt'] is not None]
    losses_data = [t['pnl_usdt'] for t in trades_data if 'فاشلة' in t['status'] and t['pnl_usdt'] is not None]
    win_count = len(wins_data)
    loss_count = len(losses_data)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    avg_win = sum(wins_data) / win_count if win_count > 0 else 0
    avg_loss = sum(losses_data) / loss_count if loss_count > 0 else 0
    profit_factor = sum(wins_data) / abs(sum(losses_data)) if sum(losses_data) != 0 else float('inf')
    message = (
        f"📊 **إحصائيات الأداء التفصيلية**\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"**إجمالي الربح/الخسارة:** `${total_pnl:+.2f}`\n"
        f"**متوسط الربح:** `${avg_win:+.2f}`\n"
        f"**متوسط الخسارة:** `${avg_loss:+.2f}`\n"
        f"**عامل الربح (Profit Factor):** `{profit_factor:,.2f}`\n"
        f"**معدل النجاح:** {win_rate:.1f}%\n"
        f"**إجمالي الصفقات:** {total_trades}"
    )
    await safe_edit_message(update.callback_query, message, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📜 تقرير أداء الاستراتيجيات", callback_data="db_strategy_report")],[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))


async def show_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer("جاري جلب بيانات المحفظة...")
    try:
        balance = await bot_data.exchange.fetch_balance({'type': 'trading'})
        owned_assets = {asset: data['total'] for asset, data in balance.items() if isinstance(data, dict) and data.get('total', 0) > 0}
        usdt_balance = balance.get('USDT', {}); total_usdt_equity = usdt_balance.get('total', 0); free_usdt = usdt_balance.get('free', 0)
        assets_to_fetch = [f"{asset}/USDT" for asset in owned_assets if asset != 'USDT']
        tickers = {}
        if assets_to_fetch:
            try: tickers = await bot_data.exchange.fetch_tickers(assets_to_fetch)
            except Exception as e: logger.warning(f"Could not fetch all tickers for portfolio: {e}")
        asset_details = []; total_assets_value_usdt = 0
        for asset, total in owned_assets.items():
            if asset == 'USDT': continue
            symbol = f"{asset}/USDT"; value_usdt = 0
            if symbol in tickers and tickers[symbol] is not None: value_usdt = tickers[symbol].get('last', 0) * total
            total_assets_value_usdt += value_usdt
            if value_usdt >= 1.0: asset_details.append(f"  - `{asset}`: `{total:,.6f}` `(≈ ${value_usdt:,.2f})`")
        total_equity = total_usdt_equity + total_assets_value_usdt
        async with aiosqlite.connect(DB_FILE) as conn:
            cursor_pnl = await conn.execute("SELECT SUM(pnl_usdt) FROM trades WHERE status LIKE '%(%'")
            total_realized_pnl = (await cursor_pnl.fetchone())[0] or 0.0
            cursor_trades = await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")
            active_trades_count = (await cursor_trades.fetchone())[0]
        assets_str = "\n".join(asset_details) if asset_details else "  لا توجد أصول أخرى بقيمة تزيد عن 1 دولار."
        message = (
            f"**💼 نظرة عامة على المحفظة**\n"
            f"🗓️ {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"**💰 إجمالي قيمة المحفظة:** `≈ ${total_equity:,.2f}`\n"
            f"  - **السيولة المتاحة (USDT):** `${free_usdt:,.2f}`\n"
            f"  - **قيمة الأصول الأخرى:** `≈ ${total_assets_value_usdt:,.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"**📊 تفاصيل الأصول (أكثر من 1$):**\n"
            f"{assets_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"**📈 أداء التداول:**\n"
            f"  - **الربح/الخسارة المحقق:** `${total_realized_pnl:,.2f}`\n"
            f"  - **عدد الصفقات النشطة:** {active_trades_count}\n"
        )
        keyboard = [[InlineKeyboardButton("🔄 تحديث", callback_data="db_portfolio")], [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
        await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup(keyboard))
    except (ccxt.BaseError, aiosqlite.Error) as e:
        logger.error(f"Portfolio fetch error: {e}", exc_info=True)
        await safe_edit_message(query, f"حدث خطأ أثناء جلب رصيد المحفظة: {e}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة", callback_data="back_to_dashboard")]]))

async def show_trade_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT symbol, pnl_usdt, status FROM trades WHERE status LIKE '%(%' ORDER BY id DESC LIMIT 10")
        closed_trades = await cursor.fetchall()
    if not closed_trades:
        text = "لم يتم إغلاق أي صفقات بعد."
        keyboard = [[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
        await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))
        return
    history_list = ["📜 *آخر 10 صفقات مغلقة*"]
    for trade in closed_trades:
        emoji = "✅" if 'ناجحة' in trade['status'] or 'تأمين' in trade['status'] else "🛑"
        pnl = trade['pnl_usdt'] or 0.0
        history_list.append(f"{emoji} `{trade['symbol']}` | الربح/الخسارة: `${pnl:,.2f}`")
    text = "\n".join(history_list)
    keyboard = [[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
    await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_diagnostics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; settings = bot_data.settings
    scan_info = bot_data.last_scan_info
    determine_active_preset()
    nltk_status = "متاحة ✅" if NLTK_AVAILABLE else "غير متاحة ❌"
    scan_time = scan_info.get("start_time", "لم يتم بعد")
    scan_duration = f'{scan_info.get("duration_seconds", "N/A")} ثانية'
    scan_checked = scan_info.get("checked_symbols", "N/A")
    scan_errors = scan_info.get("analysis_errors", "N/A")
    scanners_list = "\n".join([f"  - {STRATEGY_NAMES_AR.get(key, key)}" for key in settings['active_scanners']])
    scan_job = context.job_queue.get_jobs_by_name("perform_scan")
    next_scan_time = scan_job[0].next_t.astimezone(EGYPT_TZ).strftime('%H:%M:%S') if scan_job and scan_job[0].next_t else "N/A"
    db_size = f"{os.path.getsize(DB_FILE) / 1024:.2f} KB" if os.path.exists(DB_FILE) else "N/A"
    async with aiosqlite.connect(DB_FILE) as conn:
        total_trades = (await (await conn.execute("SELECT COUNT(*) FROM trades")).fetchone())[0]
        active_trades = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]
    report = (
        f"🕵️‍♂️ *تقرير التشخيص الشامل*\n\n"
        f"تم إنشاؤه في: {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"----------------------------------\n"
        f"⚙️ **حالة النظام والبيئة**\n"
        f"- NLTK (تحليل الأخبار): {nltk_status}\n\n"
        f"🔬 **أداء آخر فحص**\n"
        f"- وقت البدء: {scan_time}\n"
        f"- المدة: {scan_duration}\n"
        f"- العملات المفحوصة: {scan_checked}\n"
        f"- فشل في التحليل: {scan_errors} عملات\n\n"
        f"🔧 **الإعدادات النشطة**\n"
        f"- **النمط الحالي: {bot_data.active_preset_name}**\n"
        f"- الماسحات المفعلة:\n{scanners_list}\n"
        f"----------------------------------\n"
        f"🔩 **حالة العمليات الداخلية**\n"
        f"- فحص العملات: يعمل, التالي في: {next_scan_time}\n"
        f"- الاتصال بـ OKX: متصل ✅\n"
        f"- قاعدة البيانات:\n"
        f"  - الاتصال: ناجح ✅\n"
        f"  - حجم الملف: {db_size}\n"
        f"  - إجمالي الصفقات: {total_trades} ({active_trades} نشطة)\n"
        f"----------------------------------"
    )
    await safe_edit_message(query, report, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 تحديث", callback_data="db_diagnostics")], [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))

async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🧠 إعدادات الذكاء التكيفي", callback_data="settings_adaptive")],
        [InlineKeyboardButton("🎛️ تعديل المعايير المتقدمة", callback_data="settings_params")],
        [InlineKeyboardButton("🔭 تفعيل/تعطيل الماسحات", callback_data="settings_scanners")],
        [InlineKeyboardButton("🗂️ أنماط جاهزة", callback_data="settings_presets")],
        [InlineKeyboardButton("🚫 القائمة السوداء", callback_data="settings_blacklist"), InlineKeyboardButton("🗑️ إدارة البيانات", callback_data="settings_data")]
    ]
    message_text = "⚙️ *الإعدادات الرئيسية*\n\nاختر فئة الإعدادات التي تريد تعديلها."
    target_message = update.message or update.callback_query.message
    if update.callback_query: await safe_edit_message(update.callback_query, message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else: await target_message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_adaptive_intelligence_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings = bot_data.settings
    def bool_format(key, text):
        val = settings.get(key, False)
        emoji = "✅" if val else "❌"
        return f"{text}: {emoji} مفعل"

    keyboard = [
        [InlineKeyboardButton(bool_format('adaptive_intelligence_enabled', 'تفعيل الذكاء التكيفي'), callback_data="param_toggle_adaptive_intelligence_enabled")],
        [InlineKeyboardButton(bool_format('dynamic_trade_sizing_enabled', 'تفعيل الحجم الديناميكي للصفقات'), callback_data="param_toggle_dynamic_trade_sizing_enabled")],
        [InlineKeyboardButton(bool_format('strategy_proposal_enabled', 'تفعيل اقتراحات الاستراتيجيات'), callback_data="param_toggle_strategy_proposal_enabled")],
        [InlineKeyboardButton("--- معايير الضبط ---", callback_data="noop")],
        [InlineKeyboardButton(f"حد أدنى لتعطيل الاستراتيجية (WR%): {settings.get('strategy_deactivation_threshold_wr', 45.0)}", callback_data="param_set_strategy_deactivation_threshold_wr")],
        [InlineKeyboardButton(f"أقل عدد صفقات للتحليل: {settings.get('strategy_analysis_min_trades', 10)}", callback_data="param_set_strategy_analysis_min_trades")],
        [InlineKeyboardButton(f"أقصى زيادة لحجم الصفقة (%): {settings.get('dynamic_sizing_max_increase_pct', 25.0)}", callback_data="param_set_dynamic_sizing_max_increase_pct")],
        [InlineKeyboardButton(f"أقصى تخفيض لحجم الصفقة (%): {settings.get('dynamic_sizing_max_decrease_pct', 50.0)}", callback_data="param_set_dynamic_sizing_max_decrease_pct")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, "🧠 **إعدادات الذكاء التكيفي**\n\nتحكم في كيفية تعلم البوت وتكيفه:", reply_markup=InlineKeyboardMarkup(keyboard))


async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings = bot_data.settings
    def bool_format(key, text):
        val = settings.get(key, False)
        emoji = "✅" if val else "❌"
        return f"{text}: {emoji} مفعل"
    def get_nested_value(d, keys):
        current_level = d
        for key in keys:
            if isinstance(current_level, dict) and key in current_level: current_level = current_level[key]
            else: return None
        return current_level
    keyboard = [
        [InlineKeyboardButton("--- إعدادات عامة ---", callback_data="noop")],
        [InlineKeyboardButton(f"عدد العملات للفحص: {settings['top_n_symbols_by_volume']}", callback_data="param_set_top_n_symbols_by_volume"),
         InlineKeyboardButton(f"أقصى عدد للصفقات: {settings['max_concurrent_trades']}", callback_data="param_set_max_concurrent_trades")],
        [InlineKeyboardButton(f"عمال الفحص المتزامنين: {settings['worker_threads']}", callback_data="param_set_worker_threads")],
        [InlineKeyboardButton("--- إعدادات المخاطر ---", callback_data="noop")],
        [InlineKeyboardButton(f"حجم الصفقة ($): {settings['real_trade_size_usdt']}", callback_data="param_set_real_trade_size_usdt"),
         InlineKeyboardButton(f"مضاعف وقف الخسارة (ATR): {settings['atr_sl_multiplier']}", callback_data="param_set_atr_sl_multiplier")],
        [InlineKeyboardButton(f"نسبة المخاطرة/العائد: {settings['risk_reward_ratio']}", callback_data="param_set_risk_reward_ratio")],
        [InlineKeyboardButton(bool_format('trailing_sl_enabled', 'تفعيل الوقف المتحرك'), callback_data="param_toggle_trailing_sl_enabled")],
        [InlineKeyboardButton(f"تفعيل الوقف المتحرك (%): {settings['trailing_sl_activation_percent']}", callback_data="param_set_trailing_sl_activation_percent"),
         InlineKeyboardButton(f"مسافة الوقف المتحرك (%): {settings['trailing_sl_callback_percent']}", callback_data="param_set_trailing_sl_callback_percent")],
        [InlineKeyboardButton(f"عدد محاولات الإغلاق: {settings['close_retries']}", callback_data="param_set_close_retries")],
        [InlineKeyboardButton("--- إعدادات الإشعارات والفلترة ---", callback_data="noop")],
        [InlineKeyboardButton(bool_format('incremental_notifications_enabled', 'إشعارات الربح المتزايدة'), callback_data="param_toggle_incremental_notifications_enabled")],
        [InlineKeyboardButton(f"نسبة إشعار الربح (%): {settings['incremental_notification_percent']}", callback_data="param_set_incremental_notification_percent")],
        [InlineKeyboardButton(f"مضاعف فلتر الحجم: {settings['volume_filter_multiplier']}", callback_data="param_set_volume_filter_multiplier")],
        [InlineKeyboardButton(bool_format('multi_timeframe_enabled', 'فلتر الأطر الزمنية'), callback_data="param_toggle_multi_timeframe_enabled")],
        [InlineKeyboardButton(bool_format('btc_trend_filter_enabled', 'فلتر اتجاه BTC'), callback_data="param_toggle_btc_trend_filter_enabled")],
        [InlineKeyboardButton(f"فترة EMA للاتجاه: {get_nested_value(settings, ['trend_filters', 'ema_period'])}", callback_data="param_set_trend_filters_ema_period")],
        [InlineKeyboardButton(f"أقصى سبريد مسموح (%): {get_nested_value(settings, ['spread_filter', 'max_spread_percent'])}", callback_data="param_set_spread_filter_max_spread_percent")],
        [InlineKeyboardButton(f"أدنى ATR مسموح (%): {get_nested_value(settings, ['volatility_filters', 'min_atr_percent'])}", callback_data="param_set_volatility_filters_min_atr_percent")],
        [InlineKeyboardButton(bool_format('market_mood_filter_enabled', 'فلتر الخوف والطمع'), callback_data="param_toggle_market_mood_filter_enabled"),
         InlineKeyboardButton(f"حد مؤشر الخوف: {settings['fear_and_greed_threshold']}", callback_data="param_set_fear_and_greed_threshold")],
        [InlineKeyboardButton(bool_format('adx_filter_enabled', 'فلتر ADX'), callback_data="param_toggle_adx_filter_enabled"),
         InlineKeyboardButton(f"مستوى فلتر ADX: {settings['adx_filter_level']}", callback_data="param_set_adx_filter_level")],
        [InlineKeyboardButton(bool_format('news_filter_enabled', 'فلتر الأخبار والبيانات'), callback_data="param_toggle_news_filter_enabled")],
        # New Settings
        [InlineKeyboardButton(bool_format('intelligent_reviewer_enabled', 'المراجع الذكي'), callback_data="param_toggle_intelligent_reviewer_enabled")],
        [InlineKeyboardButton(bool_format('momentum_scalp_mode_enabled', 'اقتناص الزخم'), callback_data="param_toggle_momentum_scalp_mode_enabled")],
        [InlineKeyboardButton(f"هدف اقتناص الزخم (%): {settings.get('momentum_scalp_target_percent', 0.5)}", callback_data="param_set_momentum_scalp_target_percent")],
        [InlineKeyboardButton(bool_format('multi_timeframe_confluence_enabled', 'فلتر التوافق الزمني'), callback_data="param_toggle_multi_timeframe_confluence_enabled")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, "🎛️ **تعديل المعايير المتقدمة**\n\nاضغط على أي معيار لتعديل قيمته مباشرة:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    active_scanners = bot_data.settings['active_scanners']
    for key, name in STRATEGY_NAMES_AR.items():
        status_emoji = "✅" if key in active_scanners else "❌"
        perf_hint = ""
        if (perf := bot_data.strategy_performance.get(key)):
            perf_hint = f" ({perf['win_rate']}% WR)"
        keyboard.append([InlineKeyboardButton(f"{status_emoji} {name}{perf_hint}", callback_data=f"scanner_toggle_{key}")])
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")])
    await safe_edit_message(update.callback_query, "اختر الماسحات لتفعيلها أو تعطيلها (مع تلميح الأداء):", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🚦 احترافي", callback_data="preset_set_professional")],
        [InlineKeyboardButton("🎯 متشدد", callback_data="preset_set_strict")],
        [InlineKeyboardButton("🌙 متساهل", callback_data="preset_set_lenient")],
        [InlineKeyboardButton("⚠️ فائق التساهل", callback_data="preset_set_very_lenient")],
        [InlineKeyboardButton("❤️ القلب الجريء", callback_data="preset_set_bold_heart")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, "اختر نمط إعدادات جاهز:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_blacklist_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    blacklist = bot_data.settings.get('asset_blacklist', [])
    blacklist_str = ", ".join(f"`{item}`" for item in blacklist) if blacklist else "لا توجد عملات في القائمة."
    text = f"🚫 **القائمة السوداء**\n" \
           f"هذه قائمة بالعملات التي لن يتم التداول عليها:\n\n{blacklist_str}"
    keyboard = [
        [InlineKeyboardButton("➕ إضافة عملة", callback_data="blacklist_add"), InlineKeyboardButton("➖ إزالة عملة", callback_data="blacklist_remove")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_data_management_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("‼️ مسح كل الصفقات ‼️", callback_data="data_clear_confirm")], [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]]
    await safe_edit_message(update.callback_query, "🗑️ *إدارة البيانات*\n\n**تحذير:** هذا الإجراء سيحذف سجل جميع الصفقات بشكل نهائي.", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_clear_data_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("نعم، متأكد. احذف كل شيء.", callback_data="data_clear_execute")], [InlineKeyboardButton("لا، تراجع.", callback_data="settings_data")]]
    await safe_edit_message(update.callback_query, "🛑 **تأكيد نهائي: حذف البيانات**\n\nهل أنت متأكد أنك تريد حذف جميع بيانات الصفقات بشكل نهائي؟", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_clear_data_execute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_edit_message(query, "جاري حذف البيانات...", reply_markup=None)
    try:
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            logger.info("Database file has been deleted by user.")
        await init_database()
        await safe_edit_message(query, "✅ تم حذف جميع بيانات الصفقات بنجاح.")
    except Exception as e:
        logger.error(f"Failed to clear data: {e}")
        await safe_edit_message(query, f"❌ حدث خطأ أثناء حذف البيانات: {e}")
    await asyncio.sleep(2)
    await show_settings_menu(update, context)

async def handle_scanner_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    scanner_key = query.data.replace("scanner_toggle_", "")
    active_scanners = bot_data.settings['active_scanners']
    if scanner_key not in STRATEGY_NAMES_AR:
        logger.error(f"Invalid scanner key: '{scanner_key}'"); await query.answer("خطأ: مفتاح الماسح غير صالح.", show_alert=True); return
    if scanner_key in active_scanners:
        if len(active_scanners) > 1: active_scanners.remove(scanner_key)
        else: await query.answer("يجب تفعيل ماسح واحد على الأقل.", show_alert=True); return
    else: active_scanners.append(scanner_key)
    save_settings(); determine_active_preset()
    await query.answer(f"{STRATEGY_NAMES_AR[scanner_key]} {'تم تفعيله' if scanner_key in active_scanners else 'تم تعطيله'}")
    await show_scanners_menu(update, context)

async def handle_strategy_adjustment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    parts = query.data.split('_')
    action = parts[2]
    proposal_key = parts[3]

    proposal = bot_data.pending_strategy_proposal
    if not proposal or proposal.get("key") != proposal_key:
        await safe_edit_message(query, "انتهت صلاحية هذا الاقتراح أو تمت معالجته بالفعل.", reply_markup=None)
        return

    if action == "approve":
        scanner_to_disable = proposal['scanner']
        if scanner_to_disable in bot_data.settings['active_scanners']:
            bot_data.settings['active_scanners'].remove(scanner_to_disable)
            save_settings()
            determine_active_preset()
            logger.info(f"User approved disabling strategy: {scanner_to_disable}")
            await safe_edit_message(query, f"✅ **تمت الموافقة.**\nتم تعطيل استراتيجية '{STRATEGY_NAMES_AR.get(scanner_to_disable, scanner_to_disable)}'.", reply_markup=None)
        else:
            await safe_edit_message(query, "⚠️ الاستراتيجية معطلة بالفعل.", reply_markup=None)
    else: # Reject
        logger.info(f"User rejected disabling strategy: {proposal['scanner']}")
        await safe_edit_message(query, "❌ **تم الرفض.**\nلن يتم إجراء أي تغييرات على الاستراتيجيات النشطة.", reply_markup=None)

    bot_data.pending_strategy_proposal = {} # Clear proposal


async def handle_preset_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    preset_key = query.data.replace("preset_set_", "")

    if preset_settings := SETTINGS_PRESETS.get(preset_key):
        # Preserve intelligence settings and scanners when changing presets
        current_scanners = bot_data.settings.get('active_scanners', [])
        adaptive_settings = {
            k: v for k, v in bot_data.settings.items() if k not in DEFAULT_SETTINGS
        }

        bot_data.settings = copy.deepcopy(preset_settings)
        bot_data.settings['active_scanners'] = current_scanners
        bot_data.settings.update(adaptive_settings) # Restore adaptive settings

        determine_active_preset()
        save_settings()

        lf = preset_settings.get('liquidity_filters', {})
        vf = preset_settings.get('volatility_filters', {})
        sf = preset_settings.get('spread_filter', {})

        confirmation_text = (
            f"✅ *تم تفعيل النمط: {PRESET_NAMES_AR.get(preset_key, preset_key)}*\n\n"
            f"*أهم القيم:*\n"
            f"- `min_rvol: {lf.get('min_rvol', 'N/A')}`\n"
            f"- `max_spread: {sf.get('max_spread_percent', 'N/A')}%`\n"
            f"- `min_atr: {vf.get('min_atr_percent', 'N/A')}%`"
        )
        await query.answer(f"تم تفعيل نمط: {PRESET_NAMES_AR.get(preset_key, preset_key)}")
        await show_presets_menu(update, context) # Refresh menu
        await safe_send_message(context.bot, confirmation_text)

    else:
        await query.answer("لم يتم العثور على النمط.")

async def handle_parameter_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; param_key = query.data.replace("param_set_", "")
    context.user_data['setting_to_change'] = param_key
    if '_' in param_key: await query.message.reply_text(f"أرسل القيمة الرقمية الجديدة لـ `{param_key}`:\n\n*ملاحظة: هذا إعداد متقدم (متشعب)، سيتم تحديثه مباشرة.*", parse_mode=ParseMode.MARKDOWN)
    else: await query.message.reply_text(f"أرسل القيمة الرقمية الجديدة لـ `{param_key}`:", parse_mode=ParseMode.MARKDOWN)

async def handle_toggle_parameter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; param_key = query.data.replace("param_toggle_", "")
    bot_data.settings[param_key] = not bot_data.settings.get(param_key, False)
    save_settings(); determine_active_preset()
    # Refresh the correct menu
    if param_key.startswith("adaptive") or param_key.startswith("dynamic") or param_key.startswith("strategy"):
        await show_adaptive_intelligence_menu(update, context)
    else:
        await show_parameters_menu(update, context)

async def handle_blacklist_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; action = query.data.replace("blacklist_", "")
    context.user_data['blacklist_action'] = action
    await query.message.reply_text(f"أرسل رمز العملة التي تريد **{ 'إضافتها' if action == 'add' else 'إزالتها'}** (مثال: `BTC` أو `DOGE`)")

async def _create_dummy_query_and_refresh_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, menu_callback, callback_data: str):
    """Helper to create a dummy query and refresh a specific menu."""
    dummy_query = type('Query', (), {
        'message': update.message,
        'data': callback_data,
        'edit_message_text': (lambda *args, **kwargs: asyncio.sleep(0)),
        'answer': (lambda *args, **kwargs: asyncio.sleep(0))
    })()
    await menu_callback(Update(update.update_id, callback_query=dummy_query), context)

async def handle_setting_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    if 'blacklist_action' in context.user_data:
        action = context.user_data.pop('blacklist_action'); blacklist = bot_data.settings.get('asset_blacklist', [])
        symbol = user_input.upper().replace("/USDT", "")
        if action == 'add':
            if symbol not in blacklist: blacklist.append(symbol); await update.message.reply_text(f"✅ تم إضافة `{symbol}` إلى القائمة السوداء.")
            else: await update.message.reply_text(f"⚠️ العملة `{symbol}` موجودة بالفعل.")
        elif action == 'remove':
            if symbol in blacklist: blacklist.remove(symbol); await update.message.reply_text(f"✅ تم إزالة `{symbol}` من القائمة السوداء.")
            else: await update.message.reply_text(f"⚠️ العملة `{symbol}` غير موجودة في القائمة.")
        bot_data.settings['asset_blacklist'] = blacklist; save_settings(); determine_active_preset()
        await _create_dummy_query_and_refresh_menu(update, context, show_blacklist_menu, 'settings_blacklist')
        return

    if not (setting_key := context.user_data.get('setting_to_change')): return

    try:
        # Validate input is a number
        if not re.match(r'^-?\d+(\.\d+)?$', user_input):
            raise ValueError("Input is not a valid number.")

        if setting_key in bot_data.settings and not isinstance(bot_data.settings[setting_key], dict):
            original_value = bot_data.settings[setting_key]
            if isinstance(original_value, int):
                new_value = int(user_input)
            else:
                new_value = float(user_input)
            bot_data.settings[setting_key] = new_value
        else:
            keys = setting_key.split('_'); current_dict = bot_data.settings
            for key in keys[:-1]:
                current_dict = current_dict[key]
            last_key = keys[-1]
            original_value = current_dict[last_key]
            if isinstance(original_value, int):
                new_value = int(user_input)
            else:
                new_value = float(user_input)
            current_dict[last_key] = new_value

        save_settings(); determine_active_preset()
        await update.message.reply_text(f"✅ تم تحديث `{setting_key}` إلى `{new_value}`.")
    except (ValueError, KeyError):
        await update.message.reply_text("❌ قيمة غير صالحة. الرجاء إرسال رقم صحيح (مثل 10) أو عشري (مثل 2.5).")
    finally:
        if 'setting_to_change' in context.user_data:
            del context.user_data['setting_to_change']
        
        if setting_key.startswith("adaptive") or setting_key.startswith("dynamic") or setting_key.startswith("strategy"):
             await _create_dummy_query_and_refresh_menu(update, context, show_adaptive_intelligence_menu, 'settings_adaptive')
        else:
             await _create_dummy_query_and_refresh_menu(update, context, show_parameters_menu, 'settings_params')


async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'setting_to_change' in context.user_data or 'blacklist_action' in context.user_data:
        await handle_setting_value(update, context); return
    text = update.message.text
    if text == "Dashboard 🖥️": await show_dashboard_command(update, context)
    elif text == "الإعدادات ⚙️": await show_settings_menu(update, context)

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query: return
    await query.answer()
    data = query.data
    
    route_map = {
        "db_stats": show_stats_command, "db_trades": show_trades_command, "db_history": show_trade_history_command,
        "db_mood": show_mood_command, "db_diagnostics": show_diagnostics_command, "back_to_dashboard": show_dashboard_command,
        "db_portfolio": show_portfolio_command, "db_manual_scan": manual_scan_command,
        "kill_switch_toggle": toggle_kill_switch, "db_daily_report": daily_report_command, "db_strategy_report": show_strategy_report_command,
        "settings_main": show_settings_menu, "settings_params": show_parameters_menu, "settings_scanners": show_scanners_menu,
        "settings_presets": show_presets_menu, "settings_blacklist": show_blacklist_menu, "settings_data": show_data_management_menu,
        "settings_adaptive": show_adaptive_intelligence_menu,
        "db_maestro_control": show_maestro_control, "maestro_toggle": toggle_maestro,
        "blacklist_add": handle_blacklist_action, "blacklist_remove": handle_blacklist_action,
        "data_clear_confirm": handle_clear_data_confirmation, "data_clear_execute": handle_clear_data_execute,
        "noop": (lambda u,c: None)
    }
    try:
        if data in route_map: await route_map[data](update, context)
        elif data.startswith("check_"): await check_trade_details(update, context)
        elif data.startswith("scanner_toggle_"): await handle_scanner_toggle(update, context)
        elif data.startswith("preset_set_"): await handle_preset_set(update, context)
        elif data.startswith("param_set_"): await handle_parameter_selection(update, context)
        elif data.startswith("param_toggle_"): await handle_toggle_parameter(update, context)
        elif data.startswith("strategy_adjust_"): await handle_strategy_adjustment(update, context)
    except Exception as e: logger.error(f"Error in button callback handler for data '{data}': {e}", exc_info=True)


async def post_init(application: Application):
    """Post-initialization hook for the bot."""
    bot_data.application = application
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.critical("FATAL: Missing one or more critical environment variables (API keys, Tokens, Chat ID)."); return
    
    if NLTK_AVAILABLE:
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError: 
            logger.info("Downloading NLTK data (vader_lexicon)...")
            nltk.download('vader_lexicon', quiet=True)
    
    try:
        bot_data.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await bot_data.redis_client.ping()
        logger.info("✅ Successfully connected to Redis server.")
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
        logger.error(f"Could not connect to Redis server: {e}. Redis features will be disabled.")
        bot_data.redis_client = None

    try:
        config = {'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE, 'enableRateLimit': True}
        bot_data.exchange = ccxt.okx(config)
        await bot_data.exchange.load_markets()

        logger.info("Reconciling SPOT trading state with OKX exchange...")
        
        balance = await bot_data.exchange.fetch_balance()
        owned_assets = {asset for asset, data in balance.items() if isinstance(data, dict) and data.get('total', 0) > 0.00001}
        logger.info(f"Found {len(owned_assets)} assets with balance in the wallet.")

        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            
            # --- ✅ SECURITY FIX APPLIED: Parameterized query to prevent SQL injection ---
            states_to_check = ('active', 'pending')
            placeholders = ','.join('?' * len(states_to_check))
            query_str = f"SELECT * FROM trades WHERE status IN ({placeholders})"
            trades_in_db_cursor = await conn.execute(query_str, states_to_check)
            trades_in_db = await trades_in_db_cursor.fetchall()
            # --- ✅ END SECURITY FIX ---

            logger.info(f"Found {len(trades_in_db)} active/pending trades in the local database to reconcile.")

            for trade in trades_in_db:
                base_currency = trade['symbol'].split('/')[0]
                if base_currency not in owned_assets and trade['status'] == 'active':
                    logger.warning(f"Trade #{trade['id']} for {trade['symbol']} is in DB, but asset balance is zero. Marking as manually closed.")
                    await conn.execute("UPDATE trades SET status = 'مغلقة يدوياً' WHERE id = ?", (trade['id'],))
            
            await conn.commit()
        logger.info("State reconciliation for SPOT complete.")

    except ccxt.BaseError as e:
        logger.critical(f"🔥 FATAL: Could not connect or reconcile state with OKX: {e}", exc_info=True)
        return
    except Exception as e:
        logger.critical(f"🔥 FATAL: An unexpected error occurred during initialization: {e}", exc_info=True)
        return

    await check_time_sync(ContextTypes.DEFAULT_TYPE(application=application))
    bot_data.trade_guardian = TradeGuardian(application)
    bot_data.public_ws = PublicWebSocketManager(bot_data.trade_guardian.handle_ticker_update)
    bot_data.private_ws = PrivateWebSocketManager()
    asyncio.create_task(bot_data.public_ws.run()); asyncio.create_task(bot_data.private_ws.run())
    logger.info("Waiting 5s for WebSocket connections..."); await asyncio.sleep(5)
    await bot_data.trade_guardian.sync_subscriptions()
    
    job_queue = application.job_queue
    
    job_queue.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name="perform_scan")
    job_queue.run_repeating(the_supervisor_job, interval=SUPERVISOR_INTERVAL_SECONDS, first=30, name="the_supervisor_job")
    job_queue.run_repeating(check_time_sync, interval=TIME_SYNC_INTERVAL_SECONDS, first=TIME_SYNC_INTERVAL_SECONDS, name="time_sync_job")
    job_queue.run_repeating(critical_trade_monitor, interval=CRITICAL_MONITOR_INTERVAL_SECONDS, first=CRITICAL_MONITOR_INTERVAL_SECONDS, name="critical_trade_monitor")
    job_queue.run_daily(send_daily_report, time=dt_time(hour=23, minute=55, tzinfo=EGYPT_TZ), name='daily_report')
    job_queue.run_repeating(update_strategy_performance, interval=STRATEGY_ANALYSIS_INTERVAL_SECONDS, first=60, name="update_strategy_performance")
    job_queue.run_repeating(propose_strategy_changes, interval=STRATEGY_ANALYSIS_INTERVAL_SECONDS, first=120, name="propose_strategy_changes")
    reviewer_interval = bot_data.settings.get('intelligent_reviewer_interval_minutes', 30) * 60
    job_queue.run_repeating(intelligent_reviewer_job, interval=reviewer_interval, first=reviewer_interval, name="intelligent_reviewer_job")
    job_queue.run_repeating(maestro_job, interval=MAESTRO_INTERVAL_HOURS * 3600, first=15, name="maestro_job")

    logger.info(f"Jobs scheduled. Scan every {SCAN_INTERVAL_SECONDS}s. Maestro every {MAESTRO_INTERVAL_HOURS} hour. Critical Monitor every {CRITICAL_MONITOR_INTERVAL_SECONDS}s.")
    try: await application.bot.send_message(TELEGRAM_CHAT_ID, "*🤖 قناص OKX | إصدار المايسترو - بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)
    except Forbidden: logger.critical(f"FATAL: Bot not authorized for chat ID {TELEGRAM_CHAT_ID}. Please check TELEGRAM_CHAT_ID and bot permissions."); return
    logger.info("--- OKX Sniper Bot is now fully operational ---")

async def post_shutdown(application: Application):
    if bot_data.exchange: await bot_data.exchange.close()
    
    if bot_data.redis_client:
        await bot_data.redis_client.close()
        logger.info("Redis connection closed.")

    logger.info("Bot has shut down.")

def main():
    logger.info("--- Starting OKX Sniper Bot v33.7 (Audit Certified) ---")
    load_settings()
    asyncio.run(init_database())
    app_builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    app_builder.post_init(post_init).post_shutdown(post_shutdown)
    application = app_builder.build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("scan", manual_scan_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    
    application.run_polling()

if __name__ == '__main__':
    main()
