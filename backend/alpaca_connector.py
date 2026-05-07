import os
from datetime import datetime, timezone
from loguru import logger
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, PRIMARY_TICKER

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — running in stub mode")


class AlpacaConnector:
    def __init__(self):
        self._client = None
        self._start_value = 100_000.0
        self._trades: list[dict] = []
        if _ALPACA_AVAILABLE and ALPACA_API_KEY and ALPACA_API_KEY != "your_key_here":
            self._client = TradingClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                paper=True,
                url_override=ALPACA_BASE_URL if ALPACA_BASE_URL else None,
            )
            logger.info("Alpaca paper trading client initialized")
        else:
            logger.warning("Alpaca client not configured — paper trade mode (stub)")

    def get_portfolio(self) -> dict:
        if self._client:
            try:
                account = self._client.get_account()
                equity = float(account.equity)
                last_equity = float(account.last_equity)
                pnl_today = (equity - last_equity) / last_equity * 100 if last_equity else 0
                pnl_total = (equity - self._start_value) / self._start_value * 100
                positions = []
                for pos in self._client.get_all_positions():
                    positions.append({
                        "ticker": pos.symbol,
                        "qty": float(pos.qty),
                        "market_value": float(pos.market_value),
                        "unrealized_pnl": float(pos.unrealized_pl),
                    })
                return {
                    "total_value": equity,
                    "pnl_today_pct": round(pnl_today, 2),
                    "pnl_total_pct": round(pnl_total, 2),
                    "positions": positions,
                }
            except Exception as e:
                logger.error(f"get_portfolio failed: {e}")
        return {
            "total_value": 100_000.0,
            "pnl_today_pct": 0.0,
            "pnl_total_pct": 0.0,
            "positions": [],
        }

    def get_trade_history(self) -> list[dict]:
        if self._client:
            try:
                orders = self._client.get_orders()
                return [
                    {
                        "id": str(o.id),
                        "ticker": o.symbol,
                        "side": o.side.value,
                        "qty": float(o.qty or 0),
                        "filled_avg_price": float(o.filled_avg_price or 0),
                        "filled_at": str(o.filled_at),
                        "status": o.status.value,
                    }
                    for o in orders
                ]
            except Exception as e:
                logger.error(f"get_trade_history failed: {e}")
        return self._trades

    def get_metrics(self) -> dict:
        portfolio = self.get_portfolio()
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "n_trades": len(self._trades),
            "total_value": portfolio["total_value"],
        }

    def place_order(self, ticker: str, side: str, notional: float):
        if not self._client:
            logger.info(f"[STUB] Would place {side} order for {ticker} notional=${notional:,.0f}")
            self._trades.append({
                "ticker": ticker,
                "side": side,
                "notional": notional,
                "time": datetime.now(timezone.utc).isoformat(),
            })
            return
        try:
            req = MarketOrderRequest(
                symbol=ticker,
                notional=round(notional, 2),
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            logger.info(f"Order submitted: {order.id} {side} {ticker} ${notional:,.0f}")
        except Exception as e:
            logger.error(f"Order placement failed for {ticker}: {e}")

    def liquidate_all(self):
        if self._client:
            self._client.close_all_positions(cancel_orders=True)
            logger.warning("All positions closed via Alpaca")
        else:
            logger.warning("[STUB] Would liquidate all positions")

    def run_inference_and_trade(self, features) -> dict:
        """Called by scheduler after features are built. Returns the signal dict."""
        logger.info("Running multi-horizon inference pipeline...")
        try:
            from models.strategy_scorer import MultiHorizonScorer
            from risk_manager import RiskManager

            scorer = MultiHorizonScorer().load()
            if not scorer.is_ready:
                logger.warning("No trained TFT model found — skipping inference")
                return {}

            result = scorer.score(features)
            portfolio = self.get_portfolio()

            risk = RiskManager()
            score, position_pct, should_trade = risk.validate_order(
                ticker=PRIMARY_TICKER,
                p_sell=result["p_sell"],
                p_hold=result["p_hold"],
                p_buy=result["p_buy"],
                portfolio_value=portfolio["total_value"],
                start_value=self._start_value,
                avg_volume=5_000_000,
            )
            # Override score/position with risk-validated values
            result["score"]        = score
            result["position_pct"] = position_pct

            logger.info(
                f"Signal: score={score:.2f}  "
                f"daily={result['horizons']['daily']['signal']}  "
                f"weekly={result['horizons']['weekly']['signal']}  "
                f"monthly={result['horizons']['monthly']['signal']}  "
                f"→ {result['strategy']}"
            )

            if should_trade:
                notional = abs(position_pct / 100.0) * portfolio["total_value"]
                side = "buy" if position_pct > 0 else "sell"
                self.place_order(PRIMARY_TICKER, side, notional)

            return result
        except Exception as e:
            logger.error(f"Inference pipeline error: {e}")
            return {}
