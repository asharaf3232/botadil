# ملف: webhook_server.py (الإصدار الأولي للاختبار)

import logging
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebhookServer")

@app.post("/tradingview-webhook")
async def receive_webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"SUCCESS: Received signal from TradingView: {data}")
        return {"status": "success", "data_received": data}
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing signal")

@app.on_event("startup")
async def startup_event():
    logger.info("Webhook server started. Listening on http://0.0.0.0:8000/tradingview-webhook")
