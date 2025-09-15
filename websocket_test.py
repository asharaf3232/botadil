# -*- coding: utf-8 -*-
# =================================================================
# --- 🚀 المرحلة الأولى: بناء أساس الـ WebSocket (الإصدار 0.1) ---
# =================================================================
# الهدف: الاتصال بـ WebSocket العام لمنصة OKX والاشتراك في
#        بيانات الأسعار اللحظية (tickers) لعملة BTC/USDT.
#
# للتثبيت: pip install websockets
# =================================================================

import asyncio
import websockets
import json
import logging

# --- الإعدادات الأساسية ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
OKX_PUBLIC_WEBSOCKET_URL = "wss://ws.okx.com:8443/ws/v5/public"

async def okx_websocket_listener():
    """
    يتصل بـ OKX WebSocket، يشترك في قناة الأسعار، ويستمع للتحديثات.
    """
    # 1. إعداد رسالة الاشتراك (Subscribe Message)
    #    نحن نطلب من المنصة أن ترسل لنا تحديثات من قناة 'tickers'
    #    لأداة التداول 'BTC-USDT'.
    subscribe_message = {
        "op": "subscribe",
        "args": [
            {
                "channel": "tickers",
                "instId": "BTC-USDT"
            }
        ]
    }

    # 2. حلقة اتصال لا نهائية لضمان إعادة الاتصال
    while True:
        try:
            # استخدام 'async with' يضمن إغلاق الاتصال بشكل سليم
            async with websockets.connect(OKX_PUBLIC_WEBSOCKET_URL) as websocket:
                logging.info("✅ تم الاتصال بنجاح بـ OKX WebSocket.")
                
                # 3. إرسال رسالة الاشتراك للمنصة
                await websocket.send(json.dumps(subscribe_message))
                logging.info(f"📤 تم إرسال طلب الاشتراك: {subscribe_message}")

                # 4. حلقة استماع للرسائل الواردة
                while True:
                    # انتظر وصول رسالة من الخادم
                    response = await websocket.recv()
                    
                    # OKX ترسل رسالة "ping" كل 30 ثانية للحفاظ على الاتصال.
                    # يجب أن نرد بـ "pong" لتجنب قطع الاتصال.
                    if response == 'ping':
                        await websocket.send('pong')
                        logging.info("➡️ استلمت 'ping'، أرسلت 'pong' للحفاظ على الاتصال.")
                        continue

                    # تحويل الرسالة (النص) إلى قاموس بايثون (JSON)
                    data = json.loads(response)

                    # طباعة بيانات السعر اللحظي فقط
                    if 'data' in data and data.get('arg', {}).get('channel') == 'tickers':
                        ticker_data = data['data'][0]
                        last_price = ticker_data.get('last')
                        logging.info(f"📈 تحديث سعر BTC/USDT: {last_price}")
                    else:
                        # طباعة أي رسائل أخرى (مثل تأكيد الاشتراك)
                        logging.info(f"ℹ️ رسالة من الخادم: {data}")

        except websockets.exceptions.ConnectionClosed as e:
            logging.error(f"❌ انقطع الاتصال بالـ WebSocket: {e}. سأحاول إعادة الاتصال بعد 5 ثوانٍ...")
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"🔥 حدث خطأ غير متوقع: {e}. سأحاول إعادة الاتصال بعد 5 ثوانٍ...")
            await asyncio.sleep(5)

# --- نقطة انطلاق البرنامج ---
if __name__ == "__main__":
    try:
        # تشغيل الدالة غير المتزامنة
        asyncio.run(okx_websocket_listener())
    except KeyboardInterrupt:
        logging.info("🛑 تم إيقاف البرنامج من قبل المستخدم.")

