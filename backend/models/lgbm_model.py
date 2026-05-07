import os
import pickle
import numpy as np
import pandas as pd
from loguru import logger
from config import MODEL_DIR, TEMPORAL_WEIGHT_HALF_LIFE
from data.feature_engineering import FEATURE_COLUMNS


class LGBMModel:
    """
    LightGBM multi-class classifier (SELL/HOLD/BUY).
    Uses a flat feature vector per day — no time-series structure needed.
    Same predict_latest() interface as TFTModel / NHiTSModel.
    """

    CKPT_PATH = None  # set at class level for easy override in tests

    def __init__(self, model=None):
        self._model = model

    @property
    def _ckpt(self):
        return self.CKPT_PATH or os.path.join(MODEL_DIR, "lgbm_latest.pkl")

    @classmethod
    def train(cls, features: pd.DataFrame) -> "LGBMModel":
        """
        Train on the full features DataFrame (already walk-forward split by caller).
        Returns a fitted instance.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError("lightgbm not installed — run: pip install lightgbm")

        df = features.copy()
        df = df.dropna(subset=FEATURE_COLUMNS + ["label"])

        X = df[FEATURE_COLUMNS].values.astype(np.float32)
        y = df["label"].astype(int).values

        # Exponential decay weights: recent rows count more
        sample_weight = None
        if TEMPORAL_WEIGHT_HALF_LIFE is not None:
            dates = pd.to_datetime(df.index)
            days_from_end = (dates.max() - dates).dt.days.values
            decay = np.log(2) / TEMPORAL_WEIGHT_HALF_LIFE
            sample_weight = np.exp(-decay * days_from_end).astype(np.float32)
            logger.info(f"LightGBM temporal weights: half_life={TEMPORAL_WEIGHT_HALF_LIFE}d")

        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight={0: 3.0, 1: 1.0, 2: 3.0},  # match TFT class weights
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y, sample_weight=sample_weight)
        logger.info(f"LightGBM trained on {len(X)} samples, {X.shape[1]} features")
        return cls(model)

    def save(self, path: str | None = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = path or self._ckpt
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info(f"LightGBM saved to {path}")

    @classmethod
    def load_latest(cls) -> "LGBMModel | None":
        inst = cls()
        path = inst._ckpt
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"LightGBM loaded from {path}")
            return cls(model)
        except Exception as e:
            logger.error(f"Failed to load LightGBM: {e}")
            return None

    def predict_latest(self, features: pd.DataFrame) -> tuple[float, float, float]:
        """Return (p_sell, p_hold, p_buy) for the most recent row."""
        if self._model is None:
            return 0.333, 0.334, 0.333
        try:
            row = features[FEATURE_COLUMNS].iloc[[-1]].fillna(0).values.astype(np.float32)
            probs = self._model.predict_proba(row)[0]  # shape (3,) — SELL, HOLD, BUY
            return float(probs[0]), float(probs[1]), float(probs[2])
        except Exception as e:
            logger.error(f"LightGBM prediction failed: {e}")
            return 0.333, 0.334, 0.333

    def evaluate(self, features: pd.DataFrame) -> dict:
        """Return accuracy + per-class report on the provided feature DataFrame."""
        if self._model is None:
            return {}
        try:
            from sklearn.metrics import accuracy_score, classification_report
            df = features.dropna(subset=FEATURE_COLUMNS + ["label"])
            X = df[FEATURE_COLUMNS].values.astype(np.float32)
            y = df["label"].astype(int).values
            preds = self._model.predict(X)
            acc = float(accuracy_score(y, preds))
            report = classification_report(y, preds, target_names=["SELL", "HOLD", "BUY"],
                                           output_dict=True)
            return {"accuracy": acc, "report": report}
        except Exception as e:
            logger.error(f"LightGBM evaluation failed: {e}")
            return {}
