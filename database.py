# -*- coding: utf-8 -*-
# =======================================================================================
# --- 💾 ملف قاعدة البيانات (database.py) | بوت كاسحة الألغام v6.6 💾 ---
# =======================================================================================

import sqlite3
import logging
import json
from datetime import datetime
from config import DB_FILE, EGYPT_TZ

logger = logging.getLogger("MinesweeperBot_v6")

def migrate_database():
    """Ensures the database schema is up-to-date with all required columns."""
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
    """Initializes the database file and table if they don't exist."""
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

def log_trade_to_db(signal):
    """Logs a new trade signal to the database."""
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        sql = '''INSERT INTO trades (timestamp, exchange, symbol, entry_price, take_profit, stop_loss, quantity, entry_value_usdt, status, trailing_sl_active, highest_price, reason, trade_mode, entry_order_id, exit_order_ids_json)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

        if 'quantity' not in signal or signal['quantity'] is None:
            logger.error(f"Attempted to log trade for {signal['symbol']} with missing quantity.")
            return None

        timestamp_obj = signal.get('timestamp', datetime.now(EGYPT_TZ))
        timestamp_str = timestamp_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp_obj, datetime) else str(timestamp_obj)

        params = (
            timestamp_str, signal['exchange'], signal['symbol'],
            signal.get('entry_price'), signal.get('take_profit'), signal.get('stop_loss'),
            signal.get('quantity'), signal.get('entry_value_usdt'),  
            'نشطة', False, signal.get('entry_price'),
            signal['reason'], 'real' if signal.get('is_real_trade') else 'virtual',
            signal.get('entry_order_id'), signal.get('exit_order_ids_json')
        )
        cursor.execute(sql, params)
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id
    except Exception as e:
        logger.error(f"Failed to log recommendation to DB: {e}", exc_info=True)
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
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # First, get the quantity to calculate exit_value_usdt
        cursor.execute("SELECT quantity FROM trades WHERE id=?", (trade_id,))
        trade = cursor.fetchone()
        if not trade:
            logger.error(f"Cannot close trade #{trade_id}: Not found in DB.")
            return
        
        exit_value_usdt = exit_price * trade['quantity']
        
        cursor.execute("UPDATE trades SET status=?, exit_price=?, closed_at=?, exit_value_usdt=?, pnl_usdt=? WHERE id=?",
                       (status, exit_price, closed_at_str, exit_value_usdt, pnl_usdt, trade_id))
        conn.commit()
        conn.close()
        logger.info(f"Successfully closed trade #{trade_id} in DB with status '{status}'.")
    except Exception as e:
        logger.error(f"DB update failed while closing trade #{trade_id}: {e}")

def update_trade_sl_in_db(trade_id: int, new_sl: float, highest_price: float, new_exit_ids_json: str = None):
    """Updates the stop loss, highest price, and optionally order IDs for a trade."""
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.cursor()
        
        sql = "UPDATE trades SET stop_loss=?, highest_price=?, trailing_sl_active=? "
        params = [new_sl, highest_price, True]
        
        if new_exit_ids_json is not None:
            sql += ", exit_order_ids_json=? "
            params.append(new_exit_ids_json)

        sql += "WHERE id=?"
        params.append(trade_id)

        cursor.execute(sql, tuple(params))
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
