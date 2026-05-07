from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from config import SCHEDULE_HOUR, SCHEDULE_MINUTE, SCHEDULE_TIMEZONE


def _daily_job():
    """Main daily trading decision pipeline at 15:30 NY."""
    logger.info("Daily trading job started")
    try:
        # Import here to avoid circular imports at startup
        from data.fetch_news import fetch_and_embed_news
        from data.fetch_prices import fetch_latest_prices
        from data.feature_engineering import build_features

        logger.info("Fetching latest prices and news...")
        fetch_latest_prices()
        fetch_and_embed_news()
        features = build_features()
        logger.info(f"Features built: {features.shape if hasattr(features, 'shape') else 'ok'}")

        # Inference + order placement handled by alpaca_connector
        # (model must already be trained and loaded)
        from alpaca_connector import AlpacaConnector
        import main as _main  # update shared state so WebSocket picks it up
        connector = AlpacaConnector()
        result = connector.run_inference_and_trade(features)
        if result:
            _main._state["last_signal"] = result
    except Exception as e:
        logger.error(f"Daily job failed: {e}")


class TradingScheduler:
    def __init__(self):
        self._scheduler = BackgroundScheduler()
        self._running = False

    def start(self):
        if self._running:
            return
        trigger = CronTrigger(
            hour=SCHEDULE_HOUR,
            minute=SCHEDULE_MINUTE,
            timezone=SCHEDULE_TIMEZONE,
        )
        self._scheduler.add_job(_daily_job, trigger, id="daily_trade", replace_existing=True)
        self._scheduler.start()
        self._running = True
        logger.info(f"Scheduler started — daily job at {SCHEDULE_HOUR}:{SCHEDULE_MINUTE:02d} {SCHEDULE_TIMEZONE}")

    def stop(self):
        if self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped")
