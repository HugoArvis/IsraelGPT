from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from config import (
    SCHEDULE_HOUR, SCHEDULE_MINUTE, SCHEDULE_TIMEZONE,
    RETRAIN_DAY_OF_WEEK, RETRAIN_HOUR, RETRAIN_MINUTE,
)


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


def _weekly_retrain_job():
    """Sunday evening retraining — keeps models calibrated to the current regime."""
    logger.info("Weekly retraining job started")
    try:
        from data.fetch_prices import fetch_all_prices
        from data.fetch_news import fetch_and_embed_news
        from data.feature_engineering import build_features
        from models.lgbm_model import LGBMModel
        from models.catboost_model import CatBoostModel

        logger.info("Fetching latest data for retraining...")
        fetch_all_prices(force=False)
        fetch_and_embed_news()
        features = build_features()

        for horizon in ("5d", "21d"):
            lgbm = LGBMModel.train(features, horizon=horizon)
            lgbm.save()
            cb = CatBoostModel.train(features, horizon=horizon)
            cb.save()

        logger.info("Weekly retraining complete — LightGBM + CatBoost updated for all horizons")
    except Exception as e:
        logger.error(f"Weekly retraining failed: {e}")


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

        retrain_trigger = CronTrigger(
            day_of_week=RETRAIN_DAY_OF_WEEK,
            hour=RETRAIN_HOUR,
            minute=RETRAIN_MINUTE,
            timezone=SCHEDULE_TIMEZONE,
        )
        self._scheduler.add_job(_weekly_retrain_job, retrain_trigger, id="weekly_retrain", replace_existing=True)

        self._scheduler.start()
        self._running = True
        logger.info(
            f"Scheduler started — daily trade at {SCHEDULE_HOUR}:{SCHEDULE_MINUTE:02d}, "
            f"weekly retrain on {RETRAIN_DAY_OF_WEEK.upper()} {RETRAIN_HOUR}:{RETRAIN_MINUTE:02d} {SCHEDULE_TIMEZONE}"
        )

    def stop(self):
        if self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped")
