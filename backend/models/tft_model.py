import os
import glob
import torch
import numpy as np
import pandas as pd
from loguru import logger
from config import (
    MODEL_DIR, ENCODER_LENGTH, PREDICTION_HORIZON,
    TFT_HIDDEN_SIZE, TFT_ATTENTION_HEAD_SIZE, TFT_DROPOUT, TFT_LSTM_LAYERS,
    TFT_WEIGHT_DECAY,
)
from data.feature_engineering import FEATURE_COLUMNS


class TFTModel:
    """
    Wrapper around pytorch-forecasting's TemporalFusionTransformer.
    Provides a stable interface for training, saving, loading, and inference.
    """

    def __init__(self, model=None):
        self._model = model
        self._ensemble: list = []  # extra fold models for averaged inference

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
            hidden_continuous_size=16,     # halved alongside hidden_size
            lstm_layers=TFT_LSTM_LAYERS,
            output_size=3,                 # SELL / HOLD / BUY
            loss=CrossEntropy(),
            reduce_on_plateau_patience=2,  # lr scheduler also cuts early
            log_interval=10,
            optimizer_kwargs={"weight_decay": TFT_WEIGHT_DECAY},
        )
        return cls(model)

    def save(self, path: str | None = None):
        import shutil
        os.makedirs(MODEL_DIR, exist_ok=True)
        if path is None:
            path = os.path.join(MODEL_DIR, "tft_latest.ckpt")
        # ModelCheckpoint already saved proper Lightning checkpoints for each fold.
        # Copy the most recent fold checkpoint so load_from_checkpoint works correctly.
        fold_ckpts = glob.glob(os.path.join(MODEL_DIR, "tft_fold*.ckpt"))
        if fold_ckpts:
            best = max(fold_ckpts, key=os.path.getmtime)
            shutil.copy2(best, path)
            logger.info(f"TFT model saved to {path} (copied from {os.path.basename(best)})")
        elif self._model is not None:
            # Fallback: save Lightning checkpoint via the model's internal method
            self._model.trainer.save_checkpoint(path) if hasattr(self._model, "trainer") and self._model.trainer else None
            logger.warning(f"No fold checkpoints found — attempted direct save to {path}")

    @classmethod
    def load_latest(cls) -> "TFTModel | None":
        from pytorch_forecasting import TemporalFusionTransformer

        # Load all fold checkpoints as an ensemble
        fold_ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, "tft_fold*.ckpt")))
        ensemble_models = []
        for ckpt in fold_ckpts:
            try:
                m = TemporalFusionTransformer.load_from_checkpoint(ckpt)
                m.eval()
                ensemble_models.append(m)
                logger.info(f"Ensemble: loaded {os.path.basename(ckpt)}")
            except Exception as e:
                logger.warning(f"Could not load {ckpt} for ensemble: {e}")

        if ensemble_models:
            instance = cls(ensemble_models[0])
            instance._ensemble = ensemble_models
            logger.info(f"Ensemble of {len(ensemble_models)} fold model(s) ready")
            return instance

        # Fallback to tft_latest.ckpt if no fold checkpoints found
        latest_path = os.path.join(MODEL_DIR, "tft_latest.ckpt")
        all_ckpts = glob.glob(os.path.join(MODEL_DIR, "*.ckpt"))
        if not all_ckpts:
            logger.warning(f"No checkpoints found in {MODEL_DIR}")
            return None
        latest = latest_path if os.path.exists(latest_path) else max(all_ckpts, key=os.path.getmtime)
        logger.info(f"Loading single checkpoint: {latest}")
        try:
            model = TemporalFusionTransformer.load_from_checkpoint(latest)
            return cls(model)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest}: {e}")
            return None

    def predict_latest(self, features: pd.DataFrame) -> tuple[float, float, float]:
        """
        Run inference and return (p_sell, p_hold, p_buy).
        If multiple fold models are loaded, averages their probability distributions
        before computing the final score — ensemble reduces prediction variance.
        """
        if self._model is None:
            logger.error("Model not loaded")
            return 0.333, 0.334, 0.333

        models = self._ensemble if self._ensemble else [self._model]
        try:
            from training.dataset import build_inference_dataloader
            dl = build_inference_dataloader(features)
            all_probs = []
            with torch.no_grad():
                for m in models:
                    m.eval()
                    preds = m.predict(dl, mode="raw")
                    probs = torch.softmax(preds["prediction"][0, -1], dim=-1)
                    all_probs.append(probs)
            avg_probs = torch.stack(all_probs).mean(dim=0).numpy()
            logger.debug(f"Ensemble ({len(models)} models) probs: {avg_probs}")
            return float(avg_probs[0]), float(avg_probs[1]), float(avg_probs[2])
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.333, 0.334, 0.333
