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
from datetime import datetime
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
# --- Market Analysis & Sentiment Functions ---
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
    excluded_bases = settings.get('stablecoin_filter', {}).get('exclude_bases', [])
    min_volume = settings.get('liquidity_filters', {}).get('min_quote_volume_24h_usd', 1000000)
    
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
    top_markets = final_list[:settings.get('top_n_symbols_by_volume', 250)]
    
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
            liq_filters = settings['liquidity_filters']
            vol_filters = settings['volatility_filters']
            ema_filters = settings['ema_trend_filter']

            # ... [The entire complex logic of the worker function] ...
            # This is a placeholder for brevity in this example.
            # In the real file, the full worker logic would be here.

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

async def place_real_trade(signal):
    # ... [The entire logic of the place_real_trade function] ...
    return {'success': False, 'data': "Placeholder"}

async def perform_scan(context):
    from telegram_bot import send_telegram_message # Local import to avoid circular dependency
    # ... [The entire logic of the perform_scan function] ...
    # This includes calling the worker, processing signals, and calling place_real_trade.
    # It must call save_settings() at the end to persist the last_signal_time.
    logger.info("Scan complete.")

async def track_open_trades(context):
    # ... [The entire logic of the v6.5 refactored track_open_trades with batch fetching] ...
    logger.info("Tracking complete.")

async def check_single_trade(trade, context, prefetched_data):
    # ... [The entire logic of the v6.5 refactored check_single_trade] ...
    # It calls process_trade_closure when a trade needs to be closed.
    pass

async def handle_tsl_update(context, trade, new_sl, highest_price, is_activation=False):
    # ... [The entire logic of handle_tsl_update] ...
    pass

async def update_real_trade_sl(context, trade, new_sl, highest_price, is_activation=False):
    # ... [The entire logic of update_real_trade_sl] ...
    pass

async def process_trade_closure(context, trade, exit_price, is_win):
    from telegram_bot import send_telegram_message # Local import
    # ... [The entire logic of the new process_trade_closure function] ...
    pass

async def _calculate_weighted_average_price(trades):
    # ... [The entire logic of _calculate_weighted_average_price] ...
    return 0, 0, None

async def _reconstruct_and_save_trade(exchange, symbol, context):
    # ... [The entire logic of _reconstruct_and_save_trade] ...
    return "Placeholder for rescue message."

async def analyze_performance_and_suggest(context):
    from telegram_bot import send_telegram_message # Local import
    # ... [The entire logic of analyze_performance_and_suggest] ...
    pass
