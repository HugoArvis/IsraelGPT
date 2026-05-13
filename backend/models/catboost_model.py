import os
import json
import numpy as np
import pandas as pd
from loguru import logger
from config import MODEL_DIR, TEMPORAL_WEIGHT_HALF_LIFE, CRISIS_PERIODS, CRISIS_SAMPLE_WEIGHT
from data.feature_engineering import FEATURE_COLUMNS, CATEGORICAL_FEATURES

_HORIZON_MAP = {
    "5d":  ("relative_return",     "catboost_5d.cbm"),
    "21d": ("relative_return_21d", "catboost_21d.cbm"),
}

_CAT_INDICES = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES if c in FEATURE_COLUMNS]


class CatBoostModel:
    """
    CatBoost regressor predicting forward relative return vs SP500.
    Same predict_latest() / evaluate() interface as LGBMModel.
    Positive output = expected outperformance, negative = underperformance.
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
    def _load_best_params(cls, horizon: str) -> dict | None:
        path = os.path.join(MODEL_DIR, f"catboost_best_params_{horizon}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    @classmethod
    def _make_regressor(cls, custom_params: dict | None = None):
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise RuntimeError("catboost not installed — run: pip install catboost")
        base = dict(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            random_seed=42,
            verbose=0,
        )
        if custom_params:
            base.update(custom_params)
        return CatBoostRegressor(**base)

    @classmethod
    def train(cls, features: pd.DataFrame, horizon: str = "5d") -> "CatBoostModel":
        try:
            from catboost import Pool
        except ImportError:
            raise RuntimeError("catboost not installed — run: pip install catboost")

        target_col = _HORIZON_MAP[horizon][0]
        df = features.copy().dropna(subset=[target_col])
        df = df.sort_index()

        X = df[FEATURE_COLUMNS].fillna(0)
        y = df[target_col].astype(np.float32).values

        # Temporal sample weights
        sample_weight = np.ones(len(df), dtype=np.float32)
        if TEMPORAL_WEIGHT_HALF_LIFE is not None:
            dates = pd.to_datetime(df.index).to_series()
            days_from_end = (dates.max() - dates).dt.days.values
            decay = np.log(2) / TEMPORAL_WEIGHT_HALF_LIFE
            sample_weight = np.exp(-decay * days_from_end).astype(np.float32)

        # Crisis oversampling
        idx = pd.to_datetime(df.index)
        for start, end in CRISIS_PERIODS:
            mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
            sample_weight[mask] *= CRISIS_SAMPLE_WEIGHT

        train_pool = Pool(
            data=X,
            label=y,
            weight=sample_weight,
            cat_features=_CAT_INDICES,
        )

        best_params = cls._load_best_params(horizon)
        if best_params:
            logger.info(f"CatBoost [{horizon}] using tuned hyperparameters")
        model = cls._make_regressor(custom_params=best_params)
        model.fit(train_pool)

        logger.info(
            f"CatBoost [{horizon}] trained — {model.tree_count_} trees, "
            f"target={target_col}, samples={len(X)}"
        )
        return cls(model, horizon=horizon)

    def save(self, path: str | None = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = path or self._ckpt
        self._model.save_model(path)
        logger.info(f"CatBoost [{self._horizon}] saved to {path}")

    @classmethod
    def load_latest(cls, horizon: str = "5d") -> "CatBoostModel | None":
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            return None
        inst = cls(horizon=horizon)
        if not os.path.exists(inst._ckpt):
            return None
        try:
            model = CatBoostRegressor()
            model.load_model(inst._ckpt)
            logger.info(f"CatBoost [{horizon}] loaded from {inst._ckpt}")
            return cls(model, horizon=horizon)
        except Exception as e:
            logger.error(f"Failed to load CatBoost [{horizon}]: {e}")
            return None

    def predict_latest(self, features: pd.DataFrame) -> float:
        """Return predicted forward relative return for the most recent row."""
        if self._model is None:
            return 0.0
        try:
            row = features[FEATURE_COLUMNS].iloc[[-1]].fillna(0)
            return float(self._model.predict(row)[0])
        except Exception as e:
            logger.error(f"CatBoost [{self._horizon}] prediction failed: {e}")
            return 0.0

    def evaluate(self, features: pd.DataFrame) -> dict:
        """Direction accuracy, Spearman IC, MAE and RMSE on a provided DataFrame."""
        if self._model is None:
            return {}
        try:
            from sklearn.metrics import mean_absolute_error
            from scipy import stats
            df = features.dropna(subset=[self._target_col])
            X = df[FEATURE_COLUMNS].fillna(0)
            y_true = df[self._target_col].astype(np.float32).values
            y_pred = self._model.predict(X).astype(np.float32)

            dir_acc = float((np.sign(y_pred) == np.sign(y_true)).mean())
            ic, _   = stats.spearmanr(y_pred, y_true)
            mae     = float(mean_absolute_error(y_true, y_pred))
            rmse    = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

            return {
                "direction_accuracy": dir_acc,
                "spearman_ic":        float(ic) if not np.isnan(ic) else 0.0,
                "mae":                mae,
                "rmse":               rmse,
            }
        except Exception as e:
            logger.error(f"CatBoost [{self._horizon}] evaluation failed: {e}")
            return {}
