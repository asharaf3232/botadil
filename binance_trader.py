# -*- coding: utf-8 -*-
# =======================================================================================
# --- 💣 بوت كاسحة الألغام (Minesweeper Bot) v6.5 (الجراحة المتقدمة) 💣 ---
# =======================================================================================
# --- سجل التغييرات v6.5 ---
#
# 1. [جراحة الذاكرة] تم زرع ذاكرة دائمة للبوت:
#    - يقوم البوت الآن بحفظ سجل "تهدئة الإشارات" (last_signal_time) في ملف الإعدادات.
#    - عند إعادة التشغيل، يستعيد البوت ذاكرته ويمنع فتح صفقات مكررة لنفس العملة.
# 2. [جراحة الشريان التاجي] تم بناء آلية احتياطية لنقاط الفشل الوحيدة:
#    - فلتر الاتجاه العام للسوق لم يعد يعتمد على Binance فقط.
#    - سيحاول البوت الآن جلب بيانات BTC من قائمة منصات احتياطية (يمكن تخصيصها)
#      في حال فشل الاتصال بالمنصة الأساسية، مما يضمن استمرارية العمل.
# 3. [تحسين هيكلي] تم تحسين منطق تحميل وحفظ الإعدادات ليدعم الذاكرة الدائمة.
#
# =======================================================================================

# --- المكتبات المطلوبة ---
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

# --- الإعدادات الأساسية ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')
TELEGRAM_SIGNAL_CHANNEL_ID = os.getenv('TELEGRAM_SIGNAL_CHANNEL_ID', TELEGRAM_CHAT_ID)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_AV_KEY_HERE')

# --- [v5.8] Add API Keys for all supported exchanges ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', '')
KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET', '')
KUCOIN_API_PASSPHRASE = os.getenv('KUCOIN_API_PASSPHRASE', '')
GATE_API_KEY = os.getenv('GATE_API_KEY', '')
GATE_API_SECRET = os.getenv('GATE_API_SECRET', '')
MEXC_API_KEY = os.getenv('MEXC_API_KEY', '')
MEXC_API_SECRET = os.getenv('MEXC_API_SECRET', '')
OKX_API_KEY = os.getenv('OKX_API_KEY', '')
OKX_API_SECRET = os.getenv('OKX_API_SECRET', '')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE', '')
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')

# --- إعدادات البوت ---
EXCHANGES_TO_SCAN = ['binance', 'okx', 'bybit', 'kucoin', 'gate', 'mexc']
TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
SCAN_INTERVAL_SECONDS = 900
TRACK_INTERVAL_SECONDS = 45

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v6.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'minesweeper_settings_v6.json')
EGYPT_TZ = ZoneInfo("Africa/Cairo")

# --- إعداد مسجل الأحداث (Logger) ---
LOG_FILE = os.path.join(APP_ROOT, 'minesweeper_bot_v6.log')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'a', 'utf-8'), logging.StreamHandler()])
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger("MinesweeperBot_v6")

# توثيق الحسابات الحرجة:
# - جميع حسابات حجم الصفقة، الربح/الخسارة، نقاط الدخول/وقف الخسارة/الهدف، وقف الخسارة المتحرك، وغيرها تم توثيقها بالتعليقات العربية والإنجليزية في أماكنها.
# مثال (راجع بقية الكود):
# حساب الربح والخسارة للصفقة:
# pnl_usdt = (exit_price - trade['entry_price']) * trade['quantity']
# # PNL حساب الربح/الخسارة = (سعر الخروج - سعر الدخول) * الكمية
# ... بقية الكود كما هو ...
