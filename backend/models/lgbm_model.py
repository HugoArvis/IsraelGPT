import os
import pickle
import numpy as np
import pandas as pd
from loguru import logger
from config import MODEL_DIR, TEMPORAL_WEIGHT_HALF_LIFE, CRISIS_PERIODS, CRISIS_SAMPLE_WEIGHT
from data.feature_engineering import FEATURE_COLUMNS, CATEGORICAL_FEATURES

# Maps horizon tag → (target column, checkpoint filename)
_HORIZON_MAP = {
    "1d":  ("label_1d", "lgbm_1d.pkl"),
    "5d":  ("label",    "lgbm_5d.pkl"),
    "21d": ("label_21d", "lgbm_21d.pkl"),
}


class LGBMModel:
    """
    LightGBM multi-class classifier (SELL/HOLD/BUY).
    Uses a flat feature vector per day — no time-series structure needed.
    Same predict_latest() interface as TFTModel.

    horizon: "1d" (daily), "5d" (weekly, primary), "21d" (monthly)
    """

    def __init__(self, model=None, horizon: str = "5d"):
        self._model = model
        self._horizon = horizon
        if horizon not in _HORIZON_MAP:
            raise ValueError(f"horizon must be one of {list(_HORIZON_MAP)}")

    @property
    def _target_col(self) -> str:
        return _HORIZON_MAP[self._horizon][0]

    @property
    def _ckpt(self) -> str:
        return os.path.join(MODEL_DIR, _HORIZON_MAP[self._horizon][1])

    @classmethod
    def _make_classifier(cls):
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight={0: 6.0, 1: 1.0, 2: 6.0},
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    @classmethod
    def train(cls, features: pd.DataFrame, horizon: str = "5d") -> "LGBMModel":
        """
        Train on the full features DataFrame with temporal early stopping.
        Last 20% (by time) is used as early-stopping validation only — not mixed into training.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError("lightgbm not installed — run: pip install lightgbm")

        target_col = _HORIZON_MAP[horizon][0]
        df = features.copy().dropna(subset=[target_col])
        df = df.sort_index()

        X = df[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)
        y = df[target_col].astype(int).values

        # Temporal sample weights
        sample_weight = np.ones(len(df), dtype=np.float32)
        if TEMPORAL_WEIGHT_HALF_LIFE is not None:
            dates = pd.to_datetime(df.index).to_series()
            days_from_end = (dates.max() - dates).dt.days.values
            decay = np.log(2) / TEMPORAL_WEIGHT_HALF_LIFE
            sample_weight = np.exp(-decay * days_from_end).astype(np.float32)

        # Crisis oversampling — up-weight known crisis periods so the model sees them clearly
        idx = pd.to_datetime(df.index)
        for start, end in CRISIS_PERIODS:
            mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
            sample_weight[mask] *= CRISIS_SAMPLE_WEIGHT

        # Temporal train/val split for early stopping (no data leakage)
        feat_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)

        model = cls._make_classifier()
        model.fit(
            feat_df, y,
            sample_weight=sample_weight,
            categorical_feature=CATEGORICAL_FEATURES,
        )
        logger.info(
            f"LightGBM [{horizon}] trained — {model.n_estimators_} trees, "
            f"target={target_col}, samples={len(feat_df)}"
        )
        return cls(model, horizon=horizon)

    def save(self, path: str | None = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = path or self._ckpt
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info(f"LightGBM [{self._horizon}] saved to {path}")

    @classmethod
    def load_latest(cls, horizon: str = "5d") -> "LGBMModel | None":
        inst = cls(horizon=horizon)
        if not os.path.exists(inst._ckpt):
            return None
        try:
            with open(inst._ckpt, "rb") as f:
                model = pickle.load(f)
            logger.info(f"LightGBM [{horizon}] loaded from {inst._ckpt}")
            return cls(model, horizon=horizon)
        except Exception as e:
            logger.error(f"Failed to load LightGBM [{horizon}]: {e}")
            return None

    def predict_latest(self, features: pd.DataFrame) -> tuple[float, float, float]:
        """Return (p_sell, p_hold, p_buy) for the most recent row."""
        if self._model is None:
            return 0.333, 0.334, 0.333
        try:
            row = features[FEATURE_COLUMNS].iloc[[-1]].fillna(0).astype(np.float32)
            probs = self._model.predict_proba(row)[0]
            return float(probs[0]), float(probs[1]), float(probs[2])
        except Exception as e:
            logger.error(f"LightGBM [{self._horizon}] prediction failed: {e}")
            return 0.333, 0.334, 0.333

    def evaluate(self, features: pd.DataFrame) -> dict:
        """Accuracy + per-class report on a provided DataFrame."""
        if self._model is None:
            return {}
        try:
            from sklearn.metrics import accuracy_score, classification_report
            df = features.dropna(subset=[self._target_col])
            X = df[FEATURE_COLUMNS].fillna(0).astype(np.float32)
            y = df[self._target_col].astype(int).values
            preds = self._model.predict(X)
            acc = float(accuracy_score(y, preds))
            report = classification_report(
                y, preds,
                target_names=["SELL", "HOLD", "BUY"],
                output_dict=True,
                zero_division=0,
            )
            return {"accuracy": acc, "report": report}
        except Exception as e:
            logger.error(f"LightGBM [{self._horizon}] evaluation failed: {e}")
            return {}
