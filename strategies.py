# -*- coding: utf-8 -*-
# =======================================================================================
# --- 📈 ملف الاستراتيجيات (strategies.py) | بوت كاسحة الألغام v6.6 📈 ---
# =======================================================================================

import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import asyncio

# استيراد الحالة المشتركة للبوت للوصول إلى الإعدادات
from exchanges import bot_state

# التحقق من وجود مكتبة التحليل المتقدم
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger("MinesweeperBot_v6")

# =======================================================================================
# --- Utility Functions ---
# =======================================================================================

def find_col(df_columns, prefix):
    """Finds the full column name in a DataFrame from its prefix."""
    try:
        return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration:
        return None

# =======================================================================================
# --- Strategy Analysis Functions ---
# =======================================================================================

def analyze_momentum_breakout(df, params, rvol, adx_value, exchange, symbol):
    """Analyzes data for the Momentum Breakout strategy."""
    try:
        df.ta.vwap(append=True)
        df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
        df.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'], append=True)
        df.ta.rsi(length=params['rsi_period'], append=True)
        
        macd_col = find_col(df.columns, f"MACD_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}")
        macds_col = find_col(df.columns, f"MACDs_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}")
        bbu_col = find_col(df.columns, f"BBU_{params['bbands_period']}_")
        rsi_col = find_col(df.columns, f"RSI_{params['rsi_period']}")
        
        if not all([macd_col, macds_col, bbu_col, rsi_col]):
            return None
            
        last, prev = df.iloc[-2], df.iloc[-3]
        rvol_ok = rvol >= bot_state.settings['liquidity_filters']['min_rvol']
        
        if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and
            last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and
            last[rsi_col] < params['rsi_max_level'] and rvol_ok):
            return {"reason": "momentum_breakout", "type": "long"}
    except Exception as e:
        logger.debug(f"Error in analyze_momentum_breakout for {symbol}: {e}")
    return None

def analyze_breakout_squeeze_pro(df, params, rvol, adx_value, exchange, symbol):
    """Analyzes data for the Breakout Squeeze Pro strategy."""
    try:
        df.ta.bbands(length=params['bbands_period'], std=params['bbands_stddev'], append=True)
        df.ta.kc(length=params['keltner_period'], scalar=params['keltner_atr_multiplier'], append=True)
        df.ta.obv(append=True)
        
        bbu_col = find_col(df.columns, f"BBU_{params['bbands_period']}_")
        bbl_col = find_col(df.columns, f"BBL_{params['bbands_period']}_")
        kcu_col = find_col(df.columns, f"KCUe_{params['keltner_period']}_")
        kcl_col = find_col(df.columns, f"KCLEe_{params['keltner_period']}_")
        
        if not all([bbu_col, bbl_col, kcu_col, kcl_col]):
            return None
            
        last, prev = df.iloc[-2], df.iloc[-3]
        is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]
        
        if is_in_squeeze:
            breakout_fired = last['close'] > last[bbu_col]
            volume_ok = not params.get('volume_confirmation_enabled', True) or last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5
            rvol_ok = rvol >= bot_state.settings['liquidity_filters']['min_rvol']
            obv_rising = df['OBV'].iloc[-2] > df['OBV'].iloc[-3]
            
            if breakout_fired and rvol_ok and obv_rising and volume_ok:
                return {"reason": "breakout_squeeze_pro", "type": "long"}
    except Exception as e:
        logger.debug(f"Error in analyze_breakout_squeeze_pro for {symbol}: {e}")
    return None

def find_support_resistance(high_prices, low_prices, window=10):
    """Helper function to find support and resistance levels."""
    supports, resistances = [], []
    if len(high_prices) < (2 * window + 1):
        return [], []
        
    for i in range(window, len(high_prices) - window):
        if high_prices[i] == max(high_prices[i-window:i+window+1]):
            resistances.append(high_prices[i])
        if low_prices[i] == min(low_prices[i-window:i+window+1]):
            supports.append(low_prices[i])
            
    if not supports and not resistances:
        return [], []

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
    """Analyzes data for the Sniper Pro (compression breakout) strategy."""
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
    """Analyzes order book for the Whale Radar strategy."""
    try:
        threshold = params.get("wall_threshold_usdt", 30000)
        ob = await exchange.fetch_order_book(symbol, limit=20)
        if not ob or not ob.get('bids'):
            return None

        bids = ob.get('bids', [])
        total_bid_value = sum(float(price) * float(qty) for price, qty in bids[:10] if isinstance(price, (int, float)) and isinstance(qty, (int, float)))

        if total_bid_value > threshold:
            return {"reason": "whale_radar", "type": "long"}
    except Exception as e:
        logger.warning(f"Whale Radar scan failed for {symbol}: {e}")
    return None

async def analyze_support_rebound(df, params, rvol, adx_value, exchange, symbol):
    """Analyzes for bounces off of significant support levels."""
    try:
        ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
        if not ohlcv_1h or len(ohlcv_1h) < 50:
            return None

        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        current_price = df_1h['close'].iloc[-1]

        supports, _ = find_support_resistance(df_1h['high'].to_numpy(), df_1h['low'].to_numpy(), window=5)
        if not supports:
            return None

        closest_support = max([s for s in supports if s < current_price], default=None)
        if not closest_support:
            return None

        # Check if price is within 1% of the closest support
        if (current_price - closest_support) / closest_support * 100 < 1.0:
            last_candle_15m = df.iloc[-2]
            avg_volume_15m = df['volume'].rolling(window=20).mean().iloc[-2]

            # Look for a bullish confirmation candle on the 15m chart with increased volume
            if last_candle_15m['close'] > last_candle_15m['open'] and last_candle_15m['volume'] > avg_volume_15m * 1.5:
                return {"reason": "support_rebound", "type": "long"}
    except Exception as e:
        logger.warning(f"Support Rebound scan failed for {symbol}: {e}")
    return None


# =======================================================================================
# --- Scanners Dictionary ---
# =======================================================================================

# This dictionary maps strategy names from settings to their analysis functions.
SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout,
    "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "support_rebound": analyze_support_rebound,
    "whale_radar": analyze_whale_radar,
    "sniper_pro": analyze_sniper_pro,
}
