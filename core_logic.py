# -*- coding: utf-8 -*-
# =======================================================================================
# --- ❤️‍🩹 ملف المنطق الأساسي (core_logic.py) | بوت كاسحة الألغام v6.6 ❤️‍🩹 ---
# =======================================================================================

import asyncio
import logging
import json
import time
import pandas as pd
import pandas_ta as ta
import httpx
import feedparser
import ccxt
from datetime import datetime, timezone
from collections import defaultdict

# --- استيراد الوحدات المخصصة ---
from config import *
from database import (log_trade_to_db, get_active_trades_from_db, close_trade_in_db as db_close_trade,
                      update_trade_sl_in_db, update_trade_peak_price_in_db, save_settings)
from exchanges import bot_state, scan_lock, get_exchange_adapter, get_real_balance
from strategies import SCANNERS, find_col

# استيراد مشروط لمكتبات التحليل
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# سيتم استيراد دوال إرسال الرسائل عند الحاجة لتجنب الاستيراد الدائري
# from telegram_bot import send_telegram_message

logger = logging.getLogger("MinesweeperBot_v6")

# =======================================================================================
# --- Market Analysis & Sentiment Functions (No changes here) ---
# =======================================================================================

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


async def get_fear_and_greed_index():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            response.raise_for_status()
            data = response.json().get('data', [])
            if data:
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

# =======================================================================================
# --- 🔥 NEW & CORRECTED TRADE TRACKING LOGIC 🔥 ---
# =======================================================================================

async def process_trade_closure(context, trade, exit_price, is_win):
    """Handles the logic for closing a trade, including DB update and notification."""
    from telegram_bot import send_telegram_message  # Local import
    
    pnl_usdt = (exit_price - trade['entry_price']) * trade['quantity']
    pnl_percent = (pnl_usdt / trade['entry_value_usdt']) * 100 if trade['entry_value_usdt'] > 0 else 0
    status = "ربح ✅" if is_win else "خسارة ❌"
    
    # Update trade in the database
    db_close_trade(trade['id'], status, exit_price, pnl_usdt)
    
    # Send notification
    try:
        trade_duration = datetime.now(EGYPT_TZ) - datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S')
        hours, remainder = divmod(trade_duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        duration_str = f"{int(hours)}h {int(minutes)}m"

        trade_type_str = "(صفقة حقيقية) " if trade.get('trade_mode') == 'real' else ""
        message = (
            f"📦 **إغلاق صفقة {trade_type_str}| #{trade['id']} {trade['symbol']}**\n\n"
            f"**الحالة:** {status}\n"
            f"💰 **الربح/الخسارة:** `${pnl_usdt:,.2f}` ({pnl_percent:,.2f}%)\n\n"
            f"▪️ **سعر الدخول:** `{trade['entry_price']}`\n"
            f"▪️ **سعر الخروج:** `{exit_price}`\n"
            f"▪️ **مدة الصفقة:** {duration_str}"
        )
        await send_telegram_message(context.bot, {'custom_message': message, 'target_chat': TELEGRAM_SIGNAL_CHANNEL_ID})
    except Exception as e:
        logger.error(f"Failed to send trade closure notification for trade #{trade['id']}: {e}")


async def check_single_trade(trade, context, tickers_data):
    """
    Checks a single trade against the latest market data.
    This function now contains the CRITICAL fix.
    """
    symbol = trade['symbol']
    ticker = tickers_data.get(symbol)
    settings = bot_state.settings

    # --- 💣 THE CRITICAL FIX IS HERE 💣 ---
    # Validate the ticker data before using it to prevent false closures.
    if not ticker or not isinstance(ticker.get('last'), (int, float)) or ticker.get('last') <= 0:
        logger.warning(f"Received invalid or missing price for {symbol}. Skipping check for this cycle to avoid errors.")
        return # Skip this trade until the next cycle

    current_price = ticker['last']

    # --- Check for Take Profit or Stop Loss hit ---
    if current_price >= trade['take_profit']:
        logger.info(f"✅ TAKE PROFIT hit for trade #{trade['id']} {symbol} at price {current_price}.")
        await process_trade_closure(context, trade, trade['take_profit'], is_win=True)
        return

    if current_price <= trade['stop_loss']:
        logger.info(f"❌ STOP LOSS hit for trade #{trade['id']} {symbol} at price {current_price}.")
        await process_trade_closure(context, trade, trade['stop_loss'], is_win=False)
        return

    # --- Trailing Stop Loss (TSL) Logic ---
    if settings.get('trailing_sl_enabled', True):
        highest_price = max(trade.get('highest_price', trade['entry_price']), current_price)
        
        if highest_price != trade.get('highest_price'):
            update_trade_peak_price_in_db(trade['id'], highest_price)
            trade['highest_price'] = highest_price # Update in-memory trade object

        # Activation: Check if price has risen enough to activate TSL
        activation_price = trade['entry_price'] * (1 + settings.get('trailing_sl_activation_percent', 1.5) / 100)
        
        if not trade.get('trailing_sl_active') and current_price >= activation_price:
            new_sl = trade['entry_price'] # Move SL to entry
            logger.info(f"🚀 TSL ACTIVATION for trade #{trade['id']} {symbol}. Moving SL to entry price: {new_sl}")
            update_trade_sl_in_db(trade['id'], new_sl, highest_price)
            # Send notification for activation
            from telegram_bot import send_telegram_message
            await send_telegram_message(context.bot, signal_data=trade, update_type='tsl_activation')

        # Trailing: If TSL is active, check if we need to trail the stop loss up
        elif trade.get('trailing_sl_active'):
            callback_percent = settings.get('trailing_sl_callback_percent', 1.0) / 100
            potential_new_sl = highest_price * (1 - callback_percent)
            
            if potential_new_sl > trade['stop_loss']:
                logger.info(f"📈 TSL UPDATE for trade #{trade['id']} {symbol}. New SL: {potential_new_sl}")
                update_trade_sl_in_db(trade['id'], potential_new_sl, highest_price)


async def track_open_trades(context):
    """Fetches all active trades and checks their status in batches."""
    active_trades = get_active_trades_from_db()
    if not active_trades:
        return

    bot_state.status_snapshot['active_trades_count'] = len(active_trades)
    
    # Group symbols by exchange to make batch requests
    trades_by_exchange = defaultdict(list)
    for trade in active_trades:
        trades_by_exchange[trade['exchange']].append(trade)

    all_tickers_data = {}
    
    # Fetch tickers in batches for each exchange
    for exchange_id, trades in trades_by_exchange.items():
        exchange = bot_state.public_exchanges.get(exchange_id)
        if not exchange:
            logger.warning(f"Cannot track trades on {exchange_id}, public client not available.")
            continue
        
        symbols = [trade['symbol'] for trade in trades]
        try:
            tickers = await exchange.fetch_tickers(symbols)
            all_tickers_data.update(tickers)
        except Exception as e:
            logger.error(f"Failed to fetch tickers for tracking on {exchange_id}: {e}")

    if not all_tickers_data:
        logger.error("Could not fetch any ticker data for tracking open trades.")
        return

    # Check each trade with the fetched data
    tasks = [check_single_trade(trade, context, all_tickers_data) for trade in active_trades]
    await asyncio.gather(*tasks)
    logger.info(f"Tracking complete for {len(active_trades)} active trades.")


# =======================================================================================
# --- Other core functions (placeholders/unchanged) ---
# =======================================================================================

async def perform_scan(context):
    # This function's internal logic for scanning remains the same.
    # It finds signals and then they are logged to the DB.
    # The new tracking logic above will then correctly manage them.
    logger.info("Scanning for new signals...")
    # ... [Your existing perform_scan logic would go here] ...
    logger.info("Scan complete.")

async def place_real_trade(signal):
    # This function's logic remains the same.
    # ... [Your existing place_real_trade logic would go here] ...
    return {'success': False, 'data': "This is a placeholder."}

# ... [The rest of your functions like aggregate_top_movers, worker, etc. remain unchanged] ...
# ... [Make sure to copy them back into the final file] ...
