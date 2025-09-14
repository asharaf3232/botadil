# -*- coding: utf-8 -*-
# =======================================================================================
# --- 📡 ملف المنصات (exchanges.py) | بوت كاسحة الألغام v6.6 📡 ---
# =======================================================================================

import ccxt.async_support as ccxt_async
import ccxt
import asyncio
import logging
import json
import time
from collections import deque, defaultdict

# استيراد الإعدادات والمفاتيح من ملف config.py
from config import (
    EXCHANGES_TO_SCAN,
    BINANCE_API_KEY, BINANCE_API_SECRET,
    KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSPHRASE,
    GATE_API_KEY, GATE_API_SECRET,
    MEXC_API_KEY, MEXC_API_SECRET,
    OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE,
    BYBIT_API_KEY, BYBIT_API_SECRET
)

logger = logging.getLogger("MinesweeperBot_v6")

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

# إنشاء نسخة واحدة يمكن الوصول إليها من كل الملفات
bot_state = BotState()
scan_lock = asyncio.Lock()
report_lock = asyncio.Lock()


# =======================================================================================
# --- 🚀 إعادة هيكلة المحولات (Adapters) لدعم منصات متعددة 🚀 ---
# =======================================================================================

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
