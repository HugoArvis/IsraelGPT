from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import asyncio
import json
import os
from datetime import datetime, timezone

from config import PRIMARY_TICKER, WS_MARKET_HOURS_INTERVAL, WS_OFF_HOURS_INTERVAL
from ws_manager import WSManager
from scheduler import TradingScheduler
from risk_manager import RiskManager
from alpaca_connector import AlpacaConnector

ws_manager = WSManager()
scheduler = TradingScheduler()
risk_manager = RiskManager()
alpaca = AlpacaConnector()

# In-memory state (replace with persistent store for production)
_state = {
    "model_status": "stopped",
    "start_time": None,
    "last_signal": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Trading backend starting up")
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    yield
    scheduler.stop()
    logger.info("Trading backend shut down")


app = FastAPI(title="Trading AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend build if present
_HERE = os.path.dirname(os.path.abspath(__file__))
_DIST = os.path.join(_HERE, "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    uptime = ""
    if _state["start_time"]:
        delta = datetime.now(timezone.utc) - _state["start_time"]
        hours, rem = divmod(int(delta.total_seconds()), 3600)
        uptime = f"{hours}h {rem // 60}m"
    return {"model_status": _state["model_status"], "uptime": uptime}


@app.get("/api/portfolio")
def get_portfolio():
    try:
        return alpaca.get_portfolio()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/trades")
def get_trades():
    try:
        return alpaca.get_trade_history()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/metrics")
def get_metrics():
    try:
        return alpaca.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/model/start")
def model_start():
    if _state["model_status"] == "active":
        return {"status": "already_active"}
    scheduler.start()
    _state["model_status"] = "active"
    _state["start_time"] = datetime.now(timezone.utc)
    logger.info("Model started")
    return {"status": "started"}


@app.post("/api/model/stop")
def model_stop():
    scheduler.stop()
    _state["model_status"] = "stopped"
    logger.info("Model stopped")
    return {"status": "stopped"}


@app.post("/api/model/run-now")
async def model_run_now():
    """Trigger inference in background — returns immediately, dashboard updates via WebSocket."""
    if _state["model_status"] != "active":
        raise HTTPException(status_code=400, detail="Model must be started first")

    async def _run():
        try:
            from data.fetch_prices import fetch_latest_prices
            from data.feature_engineering import build_features
            import asyncio
            logger.info("Manual run-now triggered")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, fetch_latest_prices)
            features = await loop.run_in_executor(None, build_features)
            result = await loop.run_in_executor(None, alpaca.run_inference_and_trade, features)
            if result:
                _state["last_signal"] = result
                logger.info(f"run-now complete: score={result.get('score')} strategy={result.get('strategy')}")
        except Exception as e:
            logger.error(f"run-now failed: {e}")

    asyncio.create_task(_run())
    return {"status": "running", "detail": "Inference started — dashboard will update in ~30s via WebSocket"}


@app.post("/api/model/kill")
async def model_kill(body: dict):
    if not body.get("confirm"):
        raise HTTPException(status_code=400, detail="confirm=true required")
    scheduler.stop()
    _state["model_status"] = "stopped"
    try:
        alpaca.liquidate_all()
        logger.warning("KILL SWITCH activated — all positions liquidated")
    except Exception as e:
        logger.error(f"Kill switch liquidation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "killed", "message": "All positions liquidated"}


# ── WebSocket ─────────────────────────────────────────────────────────────────

def _is_market_hours() -> bool:
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.weekday() < 5 and 9 <= now.hour < 16


def _build_live_payload() -> dict:
    last = _state.get("last_signal") or {}
    horizons = last.get("horizons", {})
    portfolio = {}
    try:
        portfolio = alpaca.get_portfolio()
    except Exception:
        pass
    return {
        "type": "live_update",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": last.get("ticker", PRIMARY_TICKER),
        # Combined multi-horizon signal
        "score":                last.get("score", 5.0),
        "confidence":           last.get("confidence", 0.0),
        "position_pct":         last.get("position_pct", 0),
        "predicted_return":     last.get("predicted_return", 0.0),
        "predicted_return_5d":  last.get("predicted_return_5d", 0.0),
        "predicted_return_21d": last.get("predicted_return_21d", 0.0),
        "stress":               last.get("stress", 0.0),
        "strategy":             last.get("strategy", "—"),
        # Per-horizon breakdown
        "weekly":  horizons.get("weekly",  {"predicted_return_pct": 0.0}),
        "monthly": horizons.get("monthly", {"predicted_return_pct": 0.0}),
        # Portfolio
        "portfolio_value":  portfolio.get("total_value", 100_000.0),
        "pnl_today_pct":    portfolio.get("pnl_today_pct", 0.0),
        "pnl_total_pct":    portfolio.get("pnl_total_pct", 0.0),
        "sharpe_rolling":   last.get("sharpe_rolling", 0.0),
        "drawdown_current": last.get("drawdown_current", 0.0),
        "model_status":     _state["model_status"],
    }


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            interval = WS_MARKET_HOURS_INTERVAL if _is_market_hours() else WS_OFF_HOURS_INTERVAL
            payload = _build_live_payload()
            await ws_manager.send_personal(json.dumps(payload), websocket)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
