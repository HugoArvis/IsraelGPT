import numpy as np
import pandas as pd
from loguru import logger
from config import CONFIDENCE_THRESHOLD, SCORE_NEUTRAL


def _dominant(p_sell: float, p_hold: float, p_buy: float) -> str:
    return ["SELL", "HOLD", "BUY"][int(np.argmax([p_sell, p_hold, p_buy]))]


def _strategy_text(weekly: str, monthly: str, confidence: float, conflicting: bool) -> str:
    if confidence < CONFIDENCE_THRESHOLD:
        return "Low confidence — hold, no trade"
    if conflicting:
        return "5d/21d conflict — hold, wait for resolution"
    if weekly == monthly:
        if weekly == "BUY":
            return "5d and 21d aligned → strong buy"
        if weekly == "SELL":
            return "5d and 21d aligned → strong sell"
        return "Both horizons neutral — hold"
    if weekly == "BUY" and monthly == "HOLD":
        return "Short-term buy — monthly neutral, size down"
    if weekly == "SELL" and monthly == "HOLD":
        return "Short-term sell — monthly neutral, size down"
    if weekly == "HOLD" and monthly == "BUY":
        return "Long-term bullish, short-term pause — hold position"
    if weekly == "HOLD" and monthly == "SELL":
        return "Long-term bearish, short-term pause — reduce risk"
    return "Mixed signals — hold"


class MultiHorizonScorer:
    """
    Fuses 5d (tactical) and 21d (trend) LightGBM models into one signal.

    Fusion rules:
    1. Regime-adaptive weights — in stress/crisis the 21d trend weight rises
       (calm: 5d=0.65/21d=0.35  →  crisis: 5d=0.45/21d=0.55)
    2. Conflict penalty — if 5d and 21d are in opposing non-HOLD directions,
       directional probabilities are compressed toward HOLD by 25 pp.
    """

    def __init__(self):
        self._lgbm_5d  = None
        self._lgbm_21d = None

    def load(self) -> "MultiHorizonScorer":
        from models.lgbm_model import LGBMModel
        self._lgbm_5d  = LGBMModel.load_latest(horizon="5d")
        self._lgbm_21d = LGBMModel.load_latest(horizon="21d")
        loaded = sum(m is not None for m in [self._lgbm_5d, self._lgbm_21d])
        logger.info(f"MultiHorizonScorer: {loaded}/2 models loaded (5d + 21d)")
        return self

    @property
    def is_ready(self) -> bool:
        return self._lgbm_5d is not None

    def score(self, features: pd.DataFrame) -> dict:
        # --- per-horizon raw probabilities ---
        if self._lgbm_5d:
            p_sell_5, p_hold_5, p_buy_5 = self._lgbm_5d.predict_latest(features)
        else:
            p_sell_5, p_hold_5, p_buy_5 = 0.333, 0.334, 0.333

        if self._lgbm_21d:
            p_sell_21, p_hold_21, p_buy_21 = self._lgbm_21d.predict_latest(features)
        else:
            p_sell_21, p_hold_21, p_buy_21 = 0.333, 0.334, 0.333

        # --- regime stress score: 0 = calm, 1 = crisis ---
        try:
            last            = features.iloc[-1]
            vix_z           = float(last.get("vix_z", 0.0))
            market_drawdown = float(last.get("market_drawdown", 0.0))
            stress = float(np.clip(vix_z / 3.0 + abs(market_drawdown) * 5.0, 0.0, 1.0))
        except Exception:
            stress = 0.0

        # --- regime-adaptive weights ---
        w5  = 0.65 - 0.20 * stress   # 0.65 (calm) → 0.45 (crisis)
        w21 = 1.0 - w5               # 0.35 (calm) → 0.55 (crisis)

        p_sell = w5 * p_sell_5 + w21 * p_sell_21
        p_hold = w5 * p_hold_5 + w21 * p_hold_21
        p_buy  = w5 * p_buy_5  + w21 * p_buy_21

        # --- conflict penalty: opposing non-HOLD signals compress toward HOLD ---
        sig5  = int(np.argmax([p_sell_5, p_hold_5, p_buy_5]))
        sig21 = int(np.argmax([p_sell_21, p_hold_21, p_buy_21]))
        conflicting = (sig5 != 1 and sig21 != 1 and sig5 != sig21)
        if conflicting:
            penalty = 0.25
            p_sell *= (1.0 - penalty)
            p_buy  *= (1.0 - penalty)
            p_hold  = 1.0 - p_sell - p_buy

        confidence = float(max(p_sell, p_hold, p_buy))

        if confidence < CONFIDENCE_THRESHOLD:
            score        = SCORE_NEUTRAL
            position_pct = 0.0
        else:
            score        = float(np.clip(5.0 + (p_buy - p_sell) * 5.0, 0.0, 10.0))
            position_pct = (score - 5.0) / 5.0 * 100.0

        weekly_signal  = _dominant(p_sell_5,  p_hold_5,  p_buy_5)
        monthly_signal = _dominant(p_sell_21, p_hold_21, p_buy_21)

        return {
            "score":        round(score, 2),
            "position_pct": round(position_pct, 1),
            "p_sell":       round(p_sell, 4),
            "p_hold":       round(p_hold, 4),
            "p_buy":        round(p_buy, 4),
            "confidence":   round(confidence, 4),
            "conflicting":  conflicting,
            "stress":       round(stress, 3),
            "strategy":     _strategy_text(weekly_signal, monthly_signal, confidence, conflicting),
            "horizons": {
                "weekly":  {"signal": weekly_signal,  "p_sell": round(p_sell_5,  4), "p_hold": round(p_hold_5,  4), "p_buy": round(p_buy_5,  4)},
                "monthly": {"signal": monthly_signal, "p_sell": round(p_sell_21, 4), "p_hold": round(p_hold_21, 4), "p_buy": round(p_buy_21, 4)},
            },
        }
