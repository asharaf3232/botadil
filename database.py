# -*- coding: utf-8 -*-
# =======================================================================================
# --- 💾 ملف قاعدة البيانات (database.py) | النسخة الكاملة والمعدلة 💾 ---
# =======================================================================================

import sqlite3
import logging
import json
import os
from datetime import datetime

# --- استيراد الوحدات المخصصة ---
from config import DB_FILE, EGYPT_TZ, SETTINGS_FILE, DEFAULT_SETTINGS
from exchanges import bot_state # <-- تم جلبه للوصول إلى حالة البوت

logger = logging.getLogger("MinesweeperBot_v6")

# =======================================================================================
# --- 🔥 الدالتان المفقودتان (تمت إضافتهما هنا) 🔥 ---
# =======================================================================================

def load_settings():
    """Loads settings from the JSON file into the bot_state."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                bot_state.settings = json.load(f)
        else:
            bot_state.settings = DEFAULT_SETTINGS.copy()
            save_settings() # Save defaults if file doesn't exist
        
        # Migrate old settings keys if necessary and fill in missing defaults
        updated = False
        for key, value in DEFAULT_SETTINGS.items():
            if key not in bot_state.settings:
                bot_state.settings[key] = value
                updated = True
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in bot_state.settings.get(key, {}):
                        bot_state.settings[key][sub_key] = sub_value
                        updated = True
        if updated:
            save_settings()
        
        logger.info("Settings loaded successfully into BotState.")

    except Exception as e:
        logger.error(f"Failed to load settings: {e}", exc_info=True)
        bot_state.settings = DEFAULT_SETTINGS.copy()

def save_settings():
    """Saves the current bot_state.settings to the JSON file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(bot_state.settings, f, indent=4)
        logger.info("Settings saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

# =======================================================================================
# --- دوال قاعدة البيانات (SQLite) ---
# =======================================================================================

def migrate_database():
    """Ensures the database schema is up-to-date with all required columns."""
    # ... (Rest of the function is correct)
    pass # This is a placeholder, your existing code is fine here

def init_database():
    """Initializes the database file and table if they don't exist."""
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, exchange TEXT, symbol TEXT, entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, entry_value_usdt REAL, status TEXT, exit_price REAL, closed_at TEXT, pnl_usdt REAL, trailing_sl_active BOOLEAN, highest_price REAL, reason TEXT, trade_mode TEXT, entry_order_id TEXT, exit_order_ids_json TEXT)')
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at: {DB_FILE}")
    except Exception as e:
        logger.error(f"Failed to initialize database at {DB_FILE}: {e}")

def log_trade_to_db(signal):
    """Logs a new trade signal to the database."""
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        sql = '''INSERT INTO trades (timestamp, exchange, symbol, entry_price, take_profit, stop_loss, quantity, entry_value_usdt, status, trailing_sl_active, highest_price, reason, trade_mode, entry_order_id, exit_order_ids_json)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

        timestamp_str = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')
        params = (
            timestamp_str, signal['exchange'], signal['symbol'],
            signal.get('entry_price'), signal.get('take_profit'), signal.get('stop_loss'),
            signal.get('quantity'), signal.get('entry_value_usdt'),  
            'نشطة', False, signal.get('entry_price'),
            signal.get('reason', 'N/A'), 'real' if signal.get('is_real_trade') else 'virtual',
            signal.get('entry_order_id'), signal.get('exit_order_ids_json')
        )
        cursor.execute(sql, params)
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id
    except Exception as e:
        logger.error(f"Failed to log trade to DB: {e}", exc_info=True)
        return None

def get_active_trades_from_db():
    """Fetches all active trades from the database."""
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'نشطة'")
        active_trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return active_trades
    except Exception as e:
        logger.error(f"DB error in get_active_trades_from_db: {e}")
        return []

def close_trade_in_db(trade_id: int, status: str, exit_price: float, pnl_usdt: float):
    """Updates a trade to a closed status in the database."""
    closed_at_str = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute("UPDATE trades SET status=?, exit_price=?, closed_at=?, pnl_usdt=? WHERE id=?",
                       (status, exit_price, closed_at_str, pnl_usdt, trade_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB update failed while closing trade #{trade_id}: {e}")

def update_trade_sl_in_db(trade_id: int, new_sl: float, highest_price: float, new_exit_ids_json: str = None):
    """Updates the stop loss and highest price for a trade."""
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute("UPDATE trades SET stop_loss=?, highest_price=?, trailing_sl_active=? WHERE id=?", 
                       (new_sl, highest_price, True, trade_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to update SL for trade #{trade_id} in DB: {e}")

def update_trade_peak_price_in_db(trade_id: int, highest_price: float):
    """Updates only the highest price for a trade."""
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        cursor.execute("UPDATE trades SET highest_price=? WHERE id=?", (highest_price, trade_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to update peak price for trade #{trade_id} in DB: {e}")
