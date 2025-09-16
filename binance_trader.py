# -*- coding: utf-8 -*-
"""
OKX Trading Bot (single-file)
- يركب 'العقل التحليلي' الذي أرفقته ويدمجها في بوت تداول حقيقي على OKX (spot)
- يتضمن: تحميل lazy للمكتبات، مسح أسواق، فحص استراتيجيات، تنفيذ أمر LIMIT buy، تأمين الصفقة (OCO)
- يحوي حماية إضافية: فحص رصيد قبل وضع OCO، واستخدام "Watchdog" لمهام الخلفية.

**ملاحظة مهمة:** قبل التشغيل ضع المتغيرات البيئية: OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE,
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
تأكد من تثبيت الحزم: ccxt, pandas, pandas_ta, aiosqlite, python-telegram-bot, httpx, websockets

هذا ملف مرجعي؛ عدِّل القيم الافتراضية ليتناسب مع حسابك واحتياجاتك.
"""

import os, asyncio, json, logging, time, hmac, base64
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict

# lazy heavy libs
ccxt = None
pd = None
ta = None

# async http
import httpx
import aiosqlite
import websockets

from telegram import ParseMode
from telegram.ext import Application

# -----------------------------------------------------------------------------
# Config / State
# -----------------------------------------------------------------------------
EGYPT_TZ = ZoneInfo("Africa/Cairo")
SETTINGS_FILE = 'okx_bot_settings.json'
DB_FILE = 'okx_bot_trades.db'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OKX_AnalyticBot')

OKX_API_KEY = os.getenv('OKX_API_KEY')
OKX_API_SECRET = os.getenv('OKX_API_SECRET')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

class BotState:
    def __init__(self):
        self.exchange = None
        self.application = None
        self.ws_manager = None
        self.settings = {}
        self.last_signal_time = {}

bot_state = BotState()

DEFAULT_SETTINGS = {
    "real_trade_size_usdt": 10.0,
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 2.0,
    "top_n_symbols_by_volume": 200,
    "min_quote_volume_24h_usd": 2_000_000,
    "min_rvol": 1.2,
    "min_atr_percent": 0.4,
    "active_scanners": ["momentum_breakout","breakout_squeeze_pro","support_rebound","sniper_pro","whale_radar"],
    "scan_interval_seconds": 900,
    "trailing_sl_enabled": True,
    "trailing_sl_activation_percent": 1.5,
    "trailing_sl_callback_percent": 1.0
}

# Strategy names mapping required by analysis functions
STRATEGIES_MAP = {
    "momentum_breakout": {"func_name": "analyze_momentum_breakout", "name": "زخم اختراقي"},
    "breakout_squeeze_pro": {"func_name": "analyze_breakout_squeeze_pro", "name": "اختراق انضغاطي"},
    "support_rebound": {"func_name": "analyze_support_rebound", "name": "ارتداد الدعم"},
    "sniper_pro": {"func_name": "analyze_sniper_pro", "name": "القناص المحترف"},
    "whale_radar": {"func_name": "analyze_whale_radar", "name": "رادار الحيتان"}
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
async def ensure_libraries_loaded():
    global ccxt, pd, ta
    if pd is None:
        import pandas as pd_lib
        pd = pd_lib
        logger.info('pandas loaded')
    if ta is None:
        import pandas_ta as ta_lib
        ta = ta_lib
        logger.info('pandas_ta loaded')
    if ccxt is None:
        import ccxt.async_support as ccxt_lib
        ccxt = ccxt_lib
        logger.info('ccxt loaded')


def create_safe_task(coro_func):
    """Wrap coroutine factory to ensure exceptions are logged."""
    async def wrapper():
        try:
            await coro_func
        except Exception as e:
            logger.critical(f'Unhandled exception in background task: {e}', exc_info=True)
    return asyncio.create_task(wrapper())


def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(bot_state.settings, f, indent=2)
    except Exception as e:
        logger.warning('Could not save settings: %s', e)


def load_settings():
    bot_state.settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                data = json.load(f)
                bot_state.settings.update(data)
        except Exception as e:
            logger.warning('Failed to load user settings: %s', e)
    save_settings()

# -----------------------------------------------------------------------------
# ===  العقل: دوال التحليل والاستراتيجيات  ===
# (نقلت دوالك كما هي مع تعديل طفيف للمتغيرات العامّة)
# -----------------------------------------------------------------------------

def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None


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
    except Exception:
        return None
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
    except Exception:
        return None
    return None


async def analyze_whale_radar(df, rvol, exchange, symbol):
    try:
        ob = await exchange.fetch_order_book(symbol, limit=20)
        if not ob or not ob.get('bids'): return None
        total_bid_value = sum(float(price) * float(qty) for price, qty in ob['bids'][:10])
        if total_bid_value > 30000:
            return {"reason": STRATEGIES_MAP['whale_radar']['name'], "type": "long"}
    except Exception:
        return None
    return None

# -----------------------------------------------------------------------------
# Core: trade initiation & protecting (with balance checks to avoid InsufficientFunds)
# -----------------------------------------------------------------------------
async def init_database():
    async with aiosqlite.connect(DB_FILE) as conn:
        await conn.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
            entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, entry_value_usdt REAL,
            status TEXT DEFAULT 'pending_protection', order_id TEXT, algo_id TEXT, highest_price REAL, trailing_sl_active INTEGER DEFAULT 0
        )''')
        await conn.commit()


async def log_initial_trade_to_db(signal, buy_order):
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            sql = '''INSERT INTO trades (timestamp, symbol, entry_price, take_profit, stop_loss, quantity, entry_value_usdt, order_id)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)'''
            params = (
                datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                signal['symbol'], signal['entry_price'], signal['take_profit'], signal['stop_loss'], buy_order['amount'], buy_order['cost'], buy_order['id']
            )
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.error('DB log failed: %s', e, exc_info=True)
        return None


async def initiate_trade(signal):
    await ensure_libraries_loaded()
    exchange = bot_state.exchange
    app = bot_state.application
    try:
        ticker = await exchange.fetch_ticker(signal['symbol'])
        limit_price = ticker['ask']
        qty = bot_state.settings['real_trade_size_usdt'] / limit_price
        logger.info(f'Placing LIMIT BUY {signal["symbol"]} qty={qty:.6f} @ {limit_price}')
        buy_order = await exchange.create_limit_buy_order(signal['symbol'], qty, limit_price)
        trade_id = await log_initial_trade_to_db(signal, buy_order)
        if trade_id:
            await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f'⏳ تم بدء صفقة | {signal["symbol"]} (ID: {trade_id})')
        else:
            logger.warning('Trade logged failed; cancelling order...')
            try:
                await exchange.cancel_order(buy_order['id'], signal['symbol'])
            except Exception:
                pass
    except Exception as e:
        logger.error('Initiate trade failed: %s', e, exc_info=True)
        await bot_state.application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f'🔥 فشل بدء الصفقة: {e}')


async def handle_filled_buy_order(order_data):
    # order_data expected from OKX ws with fields like instId, ordId
    symbol = order_data['instId'].replace('-', '/')
    order_id = order_data['ordId']
    logger.info(f'[Auditor] Fill event for {order_id} {symbol}')
    try:
        # confirm via fetch_order
        for _ in range(5):
            try:
                ord = await bot_state.exchange.fetch_order(order_id, symbol)
                if ord and ord.get('status') in ['closed','filled'] and ord.get('filled',0)>0:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.6)

        # find trade in DB
        trade = None
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM trades WHERE order_id = ? AND status = 'pending_protection'", (order_id,)) as cursor:
                trade = await cursor.fetchone()
        if not trade:
            logger.error('No DB trade matched for order %s', order_id)
            await bot_state.application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f'🔥 لم يتم العثور على سجل الصفقة لــ {order_id}. تحقق يدوياً.')
            return

        trade = dict(trade)
        filled_qty = float(ord.get('filled', trade.get('quantity', 0)))
        avg_price = float(ord.get('average', trade.get('entry_price', 0)))
        original_risk = trade['entry_price'] - trade['stop_loss']
        final_tp = avg_price + (original_risk * bot_state.settings['risk_reward_ratio'])
        final_sl = avg_price - original_risk

        # ===== balance check BEFORE placing OCO =====
        balances = await bot_state.exchange.fetch_balance()
        asset = symbol.split('/')[0]
        free_asset = float(balances.get('free', {}).get(asset, 0) or 0)
        logger.info('balance for %s free=%s filled_qty=%s', asset, free_asset, filled_qty)
        if free_asset + 1e-8 < filled_qty:
            # Not enough base asset to place selling OCO — warn and abort
            await bot_state.application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=(
                f'🚨 نقص رصيد {asset} بعد تنفيذ الشراء. مطلوب {filled_qty}, متوفر {free_asset}.\n'
                '❗️ الرجاء التدخل يدوياً لوضع وقف الخسارة.'
            ))
            return

        # Prepare OCO params for OKX (using private API endpoint via ccxt)
        instId = bot_state.exchange.market_id(symbol)
        sz = str(bot_state.exchange.amount_to_precision(symbol, filled_qty))
        oco_params = {
            'instId': instId, 'tdMode': 'cash', 'side': 'sell', 'ordType': 'oco',
            'sz': sz,
            'tpTriggerPx': str(bot_state.exchange.price_to_precision(symbol, final_tp)), 'tpOrdPx': '-1',
            'slTriggerPx': str(bot_state.exchange.price_to_precision(symbol, final_sl)), 'slOrdPx': '-1'
        }

        # attempt placing algo order
        for attempt in range(3):
            try:
                resp = await bot_state.exchange.private_post_trade_order_algo(oco_params)
                if resp and resp.get('data') and resp['data'][0].get('sCode') == '0':
                    algo_id = resp['data'][0]['algoId']
                    async with aiosqlite.connect(DB_FILE) as conn:
                        await conn.execute("UPDATE trades SET status='active', algo_id=?, quantity=?, entry_price=?, take_profit=?, stop_loss=?, highest_price=? WHERE id=?",
                                           (algo_id, filled_qty, avg_price, final_tp, final_sl, avg_price, trade['id']))
                        await conn.commit()
                    await bot_state.application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=(
                        f'✅ تم تأمين الصفقة #{trade["id"]} {symbol}\nالشراء: {avg_price}\nهدف: {final_tp}\nوقف: {final_sl}'
                    ))
                    return
                else:
                    logger.warning('OCO attempt failed: %s', resp)
                    await asyncio.sleep(1)
            except ccxt.base.errors.InsufficientFunds as e:
                logger.error('InsufficientFunds while placing OCO: %s', e)
                await bot_state.application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f'🔥 خطأ رصيد أثناء وضع OCO: {e}')
                return
            except Exception as e:
                logger.warning('OCO error: %s', e, exc_info=True)
                await asyncio.sleep(1)

        raise Exception('Failed to place OCO after attempts')

    except Exception as e:
        logger.critical('Auditor critical failure: %s', e, exc_info=True)
        await bot_state.application.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=(
            f'🔥🔥🔥 فشل حرج لساعي البريد - {symbol}\nمعرف الأمر: {order_id}\n
❗️ تدخل يدوي ضروري.'
        ))

# -----------------------------------------------------------------------------
# Scanner & orchestration
# -----------------------------------------------------------------------------
async def worker_scan(queue, signals):
    await ensure_libraries_loaded()
    exchange = bot_state.exchange
    settings = bot_state.settings
    while not queue.empty():
        market = await queue.get()
        sym = market.get('symbol')
        try:
            ob = await exchange.fetch_order_book(sym, limit=1)
            if not ob.get('bids') or not ob.get('asks'):
                queue.task_done(); continue
            spread = (ob['asks'][0][0]-ob['bids'][0][0])/ob['bids'][0][0]*100
            if spread > 1.0: queue.task_done(); continue
            ohlcv = await exchange.fetch_ohlcv(sym, '15m', limit=250)
            if len(ohlcv) < 60: queue.task_done(); continue
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2]==0:
                queue.task_done(); continue
            rvol = df['volume'].iloc[-2]/df['volume_sma'].iloc[-2]
            if rvol < settings['min_rvol']: queue.task_done(); continue
            df.ta.atr(length=14, append=True)
            atr_col = find_col(df.columns, 'ATRr_')
            last_close = df['close'].iloc[-2]
            if not atr_col or last_close==0: queue.task_done(); continue
            atr_percent = (df[atr_col].iloc[-2]/last_close)*100
            if atr_percent < settings['min_atr_percent']: queue.task_done(); continue
            ema_col = None
            df.ta.ema(length=200, append=True)
            ema_col = find_col(df.columns, 'EMA_200')
            if ema_col and last_close < df[ema_col].iloc[-2]: queue.task_done(); continue

            confirmed = []
            for name in settings['active_scanners']:
                strategy_info = STRATEGIES_MAP.get(name)
                if not strategy_info: continue
                strategy_func = globals()[strategy_info['func_name']]
                if asyncio.iscoroutinefunction(strategy_func):
                    res = await strategy_func(df.copy(), rvol, exchange, sym)
                else:
                    res = strategy_func(df.copy(), rvol)
                if res: confirmed.append(res['reason'])
            if confirmed:
                reason = ' + '.join(confirmed)
                entry_price = last_close
                df.ta.atr(length=14, append=True)
                atrc = find_col(df.columns, 'ATRr_14')
                current_atr = df.iloc[-2].get(atrc, 0)
                if current_atr>0:
                    risk = current_atr * settings['atr_sl_multiplier']
                    stop_loss = entry_price - risk
                    take_profit = entry_price + (risk * settings['risk_reward_ratio'])
                    signals.append({"symbol": sym, "reason": reason, "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit})
        except Exception as e:
            logger.debug('worker error %s: %s', sym, e)
        finally:
            queue.task_done()


async def perform_scan():
    await ensure_libraries_loaded()
    exchange = bot_state.exchange
    settings = bot_state.settings
    logger.info('Starting scan...')
    tickers = await exchange.fetch_tickers()
    min_vol = settings['min_quote_volume_24h_usd']
    usdt_markets = [m for m in tickers.values() if m.get('symbol','').endswith('/USDT') and m.get('quoteVolume',0)>min_vol]
    usdt_markets.sort(key=lambda m: m.get('quoteVolume',0), reverse=True)
    top = usdt_markets[:settings['top_n_symbols_by_volume']]
    q = asyncio.Queue()
    for m in top: await q.put(m)
    signals = []
    workers = [asyncio.create_task(worker_scan(q, signals)) for _ in range(8)]
    await q.join()
    for w in workers: w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    logger.info('Scan finished. Found %d signals', len(signals))
    # process signals sequentially, check balance then trade
    for s in signals:
        if time.time() - bot_state.last_signal_time.get(s['symbol'],0) < settings['scan_interval_seconds']*2: continue
        balances = await bot_state.exchange.fetch_balance()
        usdt_free = float(balances.get('free', {}).get('USDT', 0) or 0)
        if usdt_free >= settings['real_trade_size_usdt']:
            bot_state.last_signal_time[s['symbol']] = time.time()
            await initiate_trade(s)
            await asyncio.sleep(5)
        else:
            logger.warning('Insufficient USDT for trade: %s', usdt_free)

# -----------------------------------------------------------------------------
# WebSocket manager (listen fills) - simplified to OKX private orders channel
# -----------------------------------------------------------------------------
class WebSocketManager:
    def __init__(self, exchange):
        self.ws_url = 'wss://ws.okx.com:8443/ws/v5/private'
        self.websocket = None

    def _get_auth_args(self):
        timestamp = str(time.time())
        message = timestamp + 'GET' + '/users/self/verify'
        mac = hmac.new(bytes(OKX_API_SECRET, 'utf8'), bytes(message, 'utf8'), 'sha256')
        sign = base64.b64encode(mac.digest()).decode()
        return [{"apiKey": OKX_API_KEY, "passphrase": OKX_API_PASSPHRASE, "timestamp": timestamp, "sign": sign}]

    async def _message_handler(self, message):
        if message == 'ping': await self.websocket.send('pong'); return
        data = json.loads(message)
        if data.get('arg', {}).get('channel') == 'orders':
            for order in data.get('data', []):
                if order.get('state') == 'filled' and order.get('side') == 'buy':
                    # schedule auditor with watchdog
                    create_safe_task(handle_filled_buy_order(order))

    async def run(self):
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.websocket = ws
                    await ws.send(json.dumps({'op':'login','args': self._get_auth_args()}))
                    resp = json.loads(await ws.recv())
                    if resp.get('event')=='login' and resp.get('code')=='0':
                        await ws.send(json.dumps({'op':'subscribe','args':[{'channel':'orders','instType':'SPOT'}]}))
                        # listen
                    async for msg in ws:
                        await self._message_handler(msg)
            except Exception as e:
                logger.warning('WS error %s', e)
            await asyncio.sleep(5)

# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------
async def main():
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.critical('Set environment variables first')
        return
    load_settings()
    await init_database()
    await ensure_libraries_loaded()
    # setup exchange
    bot_state.exchange = ccxt.okx({
        'apiKey': OKX_API_KEY, 'secret': OKX_API_SECRET, 'password': OKX_API_PASSPHRASE,
        'enableRateLimit': True, 'options': {'defaultType': 'spot'}
    })
    # telegram
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    bot_state.application = app

    # ws manager
    ws = WebSocketManager(bot_state.exchange)
    bot_state.ws_manager = ws
    create_safe_task(ws.run())

    # periodic scanner
    async def periodic_scanner():
        while True:
            try:
                await perform_scan()
            except Exception as e:
                logger.error('Scanner error: %s', e, exc_info=True)
            await asyncio.sleep(bot_state.settings.get('scan_interval_seconds', 900))

    create_safe_task(periodic_scanner())

    # run telegram polling (application.run_polling is blocking so run it in task)
    create_safe_task(app.initialize())
    create_safe_task(app.start())
    logger.info('Bot started. Polling and scanning...')

    # keep loop alive
    while True:
        await asyncio.sleep(3600)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Exiting')
