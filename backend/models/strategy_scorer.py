import numpy as np
import pandas as pd
from loguru import logger
from config import CONFIDENCE_THRESHOLD, SCORE_NEUTRAL, HORIZON_WEIGHTS


def _dominant(p_sell: float, p_hold: float, p_buy: float) -> str:
    return ["SELL", "HOLD", "BUY"][int(np.argmax([p_sell, p_hold, p_buy]))]


def _strategy_text(daily: str, weekly: str, monthly: str, confidence: float) -> str:
    if confidence < CONFIDENCE_THRESHOLD:
        return "Low confidence — hold, no trade"

    signals = [daily, weekly, monthly]
    buy_count  = signals.count("BUY")
    sell_count = signals.count("SELL")

    if buy_count == 3:
        return "All horizons aligned → strong buy"
    if sell_count == 3:
        return "All horizons aligned → strong sell"
    if buy_count == 2 and monthly == "BUY":
        return "Weekly & monthly aligned → buy, filter short-term noise"
    if sell_count == 2 and monthly == "SELL":
        return "Weekly & monthly aligned → sell, filter short-term noise"
    if buy_count == 2 and monthly != "BUY":
        return "Short-term buy signal — awaiting monthly confirmation"
    if sell_count == 2 and monthly != "SELL":
        return "Short-term sell signal — awaiting monthly confirmation"
    if monthly == "BUY" and weekly == "SELL":
        return "Long-term bullish, short-term pullback — hold or reduce cautiously"
    if monthly == "SELL" and weekly == "BUY":
        return "Long-term bearish, short-term bounce — reduce risk"
    return "Mixed signals — hold"


class MultiHorizonScorer:
    """
    Combines three horizon models into one final score and strategy.

    Models:
      daily  (1d)  → LightGBM   weight 0.20  — noise filter
      weekly (5d)  → TFT        weight 0.50  — primary signal
      monthly(21d) → LightGBM   weight 0.30  — trend confirmation

    Call load() once at startup, then score(features) on each inference cycle.
    """

    def __init__(self):
        self._tft = None
        self._lgbm_daily = None
        self._lgbm_monthly = None

    def load(self) -> "MultiHorizonScorer":
        from models.tft_model import TFTModel
        from models.lgbm_model import LGBMModel

        self._tft           = TFTModel.load_latest()
        self._lgbm_daily    = LGBMModel.load_latest(horizon="1d")
        self._lgbm_monthly  = LGBMModel.load_latest(horizon="21d")

        loaded = sum(m is not None for m in [self._tft, self._lgbm_daily, self._lgbm_monthly])
        logger.info(f"MultiHorizonScorer: {loaded}/3 models loaded")
        return self

    @property
    def is_ready(self) -> bool:
        return self._tft is not None

    def score(self, features: pd.DataFrame) -> dict:
        """
        Run all three models and return a combined result dict:
          score, position_pct, p_sell, p_hold, p_buy, confidence,
          strategy (text), horizons (per-horizon breakdown).
        """
        w = HORIZON_WEIGHTS

        # Daily — LightGBM 1d
        if self._lgbm_daily:
            p_sell_d, p_hold_d, p_buy_d = self._lgbm_daily.predict_latest(features)
        else:
            p_sell_d, p_hold_d, p_buy_d = 0.333, 0.334, 0.333

        # Weekly — TFT (primary model)
        if self._tft:
            p_sell_w, p_hold_w, p_buy_w = self._tft.predict_latest(features)
        else:
            p_sell_w, p_hold_w, p_buy_w = 0.333, 0.334, 0.333

        # Monthly — LightGBM 21d
        if self._lgbm_monthly:
            p_sell_m, p_hold_m, p_buy_m = self._lgbm_monthly.predict_latest(features)
        else:
            p_sell_m, p_hold_m, p_buy_m = 0.333, 0.334, 0.333

        # Weighted combination
        p_sell = w["daily"] * p_sell_d + w["weekly"] * p_sell_w + w["monthly"] * p_sell_m
        p_hold = w["daily"] * p_hold_d + w["weekly"] * p_hold_w + w["monthly"] * p_hold_m
        p_buy  = w["daily"] * p_buy_d  + w["weekly"] * p_buy_w  + w["monthly"] * p_buy_m

        confidence = float(max(p_sell, p_hold, p_buy))

        if confidence < CONFIDENCE_THRESHOLD:
            score        = SCORE_NEUTRAL
            position_pct = 0.0
        else:
            score        = float(np.clip(5.0 + (p_buy - p_sell) * 5.0, 0.0, 10.0))
            position_pct = (score - 5.0) / 5.0 * 100.0

        daily_signal   = _dominant(p_sell_d, p_hold_d, p_buy_d)
        weekly_signal  = _dominant(p_sell_w, p_hold_w, p_buy_w)
        monthly_signal = _dominant(p_sell_m, p_hold_m, p_buy_m)

        return {
            "score":        round(score, 2),
            "position_pct": round(position_pct, 1),
            "p_sell":       round(p_sell, 4),
            "p_hold":       round(p_hold, 4),
            "p_buy":        round(p_buy, 4),
            "confidence":   round(confidence, 4),
            "strategy":     _strategy_text(daily_signal, weekly_signal, monthly_signal, confidence),
            "horizons": {
                "daily":   {"signal": daily_signal,   "p_sell": round(p_sell_d, 4), "p_hold": round(p_hold_d, 4), "p_buy": round(p_buy_d, 4)},
                "weekly":  {"signal": weekly_signal,  "p_sell": round(p_sell_w, 4), "p_hold": round(p_hold_w, 4), "p_buy": round(p_buy_w, 4)},
                "monthly": {"signal": monthly_signal, "p_sell": round(p_sell_m, 4), "p_hold": round(p_hold_m, 4), "p_buy": round(p_buy_m, 4)},
            },
        }
