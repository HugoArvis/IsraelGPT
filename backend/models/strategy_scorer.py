import numpy as np
import pandas as pd
from loguru import logger
from config import CONFIDENCE_THRESHOLD, SCORE_NEUTRAL, MAX_PREDICTED_RETURN, HORIZON_WEIGHTS


def _strategy_text(ret: float, confidence: float, stress: float) -> str:
    """Human-readable description of the signal."""
    if confidence < CONFIDENCE_THRESHOLD:
        return "Weak signal — no trade"
    suffix = "  [stress: position reduced]" if stress > 0.60 else ""
    pct = ret * 100
    if ret > MAX_PREDICTED_RETURN * 0.80:
        return f"Strong outperformance expected (+{pct:.1f}%) → strong buy{suffix}"
    if ret > MAX_PREDICTED_RETURN * 0.40:
        return f"Moderate outperformance (+{pct:.1f}%) → partial buy{suffix}"
    if ret > MAX_PREDICTED_RETURN * 0.10:
        return f"Slight outperformance (+{pct:.1f}%) → small buy{suffix}"
    if ret < -MAX_PREDICTED_RETURN * 0.80:
        return f"Strong underperformance ({pct:.1f}%) → strong sell{suffix}"
    if ret < -MAX_PREDICTED_RETURN * 0.40:
        return f"Moderate underperformance ({pct:.1f}%) → partial sell{suffix}"
    if ret < -MAX_PREDICTED_RETURN * 0.10:
        return f"Slight underperformance ({pct:.1f}%) → small sell{suffix}"
    return f"Near-zero predicted alpha ({pct:.2f}%) — hold{suffix}"


class MultiHorizonScorer:
    """
    Fuses 5d and 21d regression predictions into a single position signal.

    The model outputs predicted forward relative return vs SP500.
    Position is scaled linearly to MAX_PREDICTED_RETURN (±100% at ±3% alpha).

    Fusion:
    - 21d dominates (90%) — 5d near-random on recent data (31% direction accuracy)
    - In stress regimes, position is reduced by up to 50%
    - Signals below the confidence threshold produce zero position
    """

    def __init__(self):
        self._model_5d  = None
        self._model_21d = None

    def load(self) -> "MultiHorizonScorer":
        from models.lgbm_model import LGBMModel
        from models.catboost_model import CatBoostModel
        # 5d: CatBoost (IC 0.046, better rank signal — but direction acc ~50%, 5% weight only)
        # 21d: LightGBM (IC 0.123, direction acc 55.4% — primary decision driver)
        self._model_5d  = CatBoostModel.load_latest(horizon="5d")
        self._model_21d = LGBMModel.load_latest(horizon="21d")
        loaded = sum(m is not None for m in [self._model_5d, self._model_21d])
        logger.info(f"MultiHorizonScorer: {loaded}/2 models loaded (CatBoost-5d + LGBM-21d)")
        return self

    @property
    def is_ready(self) -> bool:
        return self._model_21d is not None

    def score(self, features: pd.DataFrame) -> dict:
        # --- per-horizon predicted returns ---
        ret_5d  = self._model_5d.predict_latest(features)  if self._model_5d  else 0.0
        ret_21d = self._model_21d.predict_latest(features) if self._model_21d else 0.0

        # --- regime stress: 0 = calm, 1 = crisis ---
        try:
            last            = features.iloc[-1]
            vix_z           = float(last.get("vix_z", 0.0))
            market_drawdown = float(last.get("market_drawdown", 0.0))
            stress = float(np.clip(vix_z / 3.0 + abs(market_drawdown) * 5.0, 0.0, 1.0))
        except Exception:
            stress = 0.0

        # --- horizon fusion: 5d noise-weighted, 21d dominates ---
        w5  = HORIZON_WEIGHTS["weekly"]   # 0.10 (calm) → 0.05 (crisis)
        w21 = HORIZON_WEIGHTS["monthly"]  # 0.90 (calm) → 0.95 (crisis)
        # Apply stress adjustment
        w5  = max(0.02, w5  - 0.05 * stress)
        w21 = 1.0 - w5
        ret = w5 * ret_5d + w21 * ret_21d

        # --- position sizing: linear scale, capped at ±MAX_PREDICTED_RETURN ---
        raw_position = float(np.clip(ret / MAX_PREDICTED_RETURN, -1.0, 1.0))

        # Confidence = normalised signal strength (0→1)
        confidence = min(abs(ret) / MAX_PREDICTED_RETURN, 1.0)

        if confidence < CONFIDENCE_THRESHOLD:
            position_pct = 0.0
            score        = SCORE_NEUTRAL
        else:
            # Stress reduces position by up to 50% in crisis regimes
            stress_factor = 1.0 - 0.5 * stress
            position_pct  = round(raw_position * stress_factor * 100.0, 1)
            score         = float(np.clip(5.0 + raw_position * 5.0, 0.0, 10.0))

        return {
            "score":                   round(score, 2),
            "position_pct":            position_pct,
            "predicted_return_5d":     round(ret_5d  * 100, 2),   # % vs SP500
            "predicted_return_21d":    round(ret_21d * 100, 2),
            "predicted_return":        round(ret     * 100, 2),    # combined signal
            "confidence":              round(confidence, 4),
            "stress":                  round(stress, 3),
            "strategy":                _strategy_text(ret, confidence, stress),
            "horizons": {
                "weekly":  {"predicted_return_pct": round(ret_5d  * 100, 2)},
                "monthly": {"predicted_return_pct": round(ret_21d * 100, 2)},
            },
        }
