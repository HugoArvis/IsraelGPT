from loguru import logger
from config import (
    MAX_PORTFOLIO_DRAWDOWN,
    MAX_POSITION_PCT,
    CONFIDENCE_THRESHOLD,
    MIN_AVG_VOLUME,
    EARNINGS_BLACKOUT_HOURS,
)


class RiskManager:
    """Hard-coded risk rules. Never bypass."""

    def check_confidence(self, p_sell: float, p_hold: float, p_buy: float) -> tuple[float, float]:
        """Return (score, position_pct). Forces neutral if confidence too low."""
        confidence = max(p_sell, p_hold, p_buy)
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(f"Confidence {confidence:.3f} < {CONFIDENCE_THRESHOLD} — forcing neutral")
            return 5.0, 0.0
        score = 5.0 + (p_buy - p_sell) * 5.0
        score = max(0.0, min(10.0, score))
        position_pct = (score - 5.0) / 5.0 * 100.0
        return score, position_pct

    def clamp_position(self, position_pct: float, portfolio_value: float, ticker: str) -> float:
        """Enforce max position per ticker rule."""
        max_pct = MAX_POSITION_PCT * 100.0
        if abs(position_pct) > max_pct:
            logger.warning(
                f"Position {position_pct:.1f}% for {ticker} exceeds max {max_pct:.0f}% — clamping"
            )
            return max_pct * (1 if position_pct > 0 else -1)
        return position_pct

    def check_global_stop_loss(self, current_value: float, start_value: float) -> bool:
        """Return True if global stop-loss triggered (drawdown > 20%)."""
        if start_value <= 0:
            return False
        drawdown = (start_value - current_value) / start_value
        if drawdown >= MAX_PORTFOLIO_DRAWDOWN:
            logger.critical(
                f"GLOBAL STOP-LOSS: drawdown {drawdown:.1%} >= {MAX_PORTFOLIO_DRAWDOWN:.0%}"
            )
            return True
        return False

    def check_volume(self, avg_volume: float, ticker: str) -> bool:
        """Return True if ticker is liquid enough to trade."""
        if avg_volume < MIN_AVG_VOLUME:
            logger.warning(f"{ticker} avg volume {avg_volume:,.0f} < {MIN_AVG_VOLUME:,} — skipping")
            return False
        return True

    def check_earnings_blackout(self, hours_to_earnings: float | None, ticker: str) -> bool:
        """Return True if in earnings blackout window (should freeze position)."""
        if hours_to_earnings is not None and 0 < hours_to_earnings <= EARNINGS_BLACKOUT_HOURS:
            logger.warning(
                f"{ticker} earnings in {hours_to_earnings:.0f}h — blackout active, no trade"
            )
            return True
        return False

    def validate_order(
        self,
        ticker: str,
        p_sell: float,
        p_hold: float,
        p_buy: float,
        portfolio_value: float,
        start_value: float,
        avg_volume: float,
        hours_to_earnings: float | None = None,
    ) -> tuple[float, float, bool]:
        """
        Full risk check pipeline.
        Returns (score, position_pct, should_trade).
        """
        if self.check_global_stop_loss(portfolio_value, start_value):
            return 5.0, 0.0, False

        if not self.check_volume(avg_volume, ticker):
            return 5.0, 0.0, False

        if self.check_earnings_blackout(hours_to_earnings, ticker):
            return 5.0, 0.0, False

        score, position_pct = self.check_confidence(p_sell, p_hold, p_buy)
        if position_pct == 0.0:
            return score, 0.0, False

        position_pct = self.clamp_position(position_pct, portfolio_value, ticker)
        return score, position_pct, True
