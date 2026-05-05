import os
import glob
import torch
import numpy as np
import pandas as pd
from loguru import logger
from config import (
    MODEL_DIR, ENCODER_LENGTH, PREDICTION_HORIZON,
    TFT_HIDDEN_SIZE, TFT_ATTENTION_HEAD_SIZE, TFT_DROPOUT, TFT_LSTM_LAYERS,
)
from data.feature_engineering import FEATURE_COLUMNS


class TFTModel:
    """
    Wrapper around pytorch-forecasting's TemporalFusionTransformer.
    Provides a stable interface for training, saving, loading, and inference.
    """

    def __init__(self, model=None):
        self._model = model

    @classmethod
    def from_dataset(cls, dataset):
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import CrossEntropy

        model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=1e-3,
            hidden_size=TFT_HIDDEN_SIZE,
            attention_head_size=TFT_ATTENTION_HEAD_SIZE,
            dropout=TFT_DROPOUT,
            hidden_continuous_size=32,
            lstm_layers=TFT_LSTM_LAYERS,
            output_size=3,         # SELL / HOLD / BUY
            loss=CrossEntropy(),
            reduce_on_plateau_patience=4,
            log_interval=10,
        )
        return cls(model)

    def save(self, path: str | None = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if path is None:
            path = os.path.join(MODEL_DIR, "tft_latest.ckpt")
        if self._model is not None:
            torch.save(self._model.state_dict(), path)
            logger.info(f"TFT model saved to {path}")

    @classmethod
    def load_latest(cls) -> "TFTModel | None":
        checkpoints = glob.glob(os.path.join(MODEL_DIR, "*.ckpt"))
        if not checkpoints:
            logger.warning(f"No checkpoints found in {MODEL_DIR}")
            return None
        latest = max(checkpoints, key=os.path.getmtime)
        logger.info(f"Loading checkpoint: {latest}")
        try:
            from pytorch_forecasting import TemporalFusionTransformer
            model = TemporalFusionTransformer.load_from_checkpoint(latest)
            return cls(model)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest}: {e}")
            return None

    def predict_latest(self, features: pd.DataFrame) -> tuple[float, float, float]:
        """
        Run inference on the most recent ENCODER_LENGTH rows of features.
        Returns (p_sell, p_hold, p_buy).
        """
        if self._model is None:
            logger.error("Model not loaded")
            return 0.333, 0.334, 0.333

        self._model.eval()
        # Build a minimal dataloader for the last window
        try:
            from training.dataset import build_inference_dataloader
            dl = build_inference_dataloader(features)
            with torch.no_grad():
                predictions = self._model.predict(dl, mode="raw")
            probs = torch.softmax(predictions["prediction"][0, -1], dim=-1).numpy()
            return float(probs[0]), float(probs[1]), float(probs[2])
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.333, 0.334, 0.333
