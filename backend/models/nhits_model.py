import os
import glob
import shutil
import torch
import pandas as pd
from loguru import logger
from config import MODEL_DIR, TFT_HIDDEN_SIZE, TFT_MAX_EPOCHS, TFT_BATCH_SIZE, TFT_WEIGHT_DECAY


class NHiTSModel:
    """N-HiTS wrapper — same interface as TFTModel for drop-in comparison."""

    def __init__(self, model=None):
        self._model = model
        self._ensemble: list = []

    @classmethod
    def from_dataset(cls, dataset):
        from pytorch_forecasting import NHiTS
        from pytorch_forecasting.metrics import CrossEntropy

        model = NHiTS.from_dataset(
            dataset,
            learning_rate=1e-3,
            hidden_size=TFT_HIDDEN_SIZE,
            output_size=3,
            loss=CrossEntropy(),
            backcast_loss_ratio=0.1,
            optimizer_kwargs={"weight_decay": TFT_WEIGHT_DECAY},
        )
        return cls(model)

    def save(self, path: str | None = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if path is None:
            path = os.path.join(MODEL_DIR, "nhits_latest.ckpt")
        fold_ckpts = glob.glob(os.path.join(MODEL_DIR, "nhits_fold*.ckpt"))
        if fold_ckpts:
            best = max(fold_ckpts, key=os.path.getmtime)
            shutil.copy2(best, path)
            logger.info(f"N-HiTS saved to {path} (from {os.path.basename(best)})")

    @classmethod
    def load_latest(cls) -> "NHiTSModel | None":
        from pytorch_forecasting import NHiTS

        fold_ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, "nhits_fold*.ckpt")))
        ensemble = []
        for ckpt in fold_ckpts:
            try:
                m = NHiTS.load_from_checkpoint(ckpt)
                m.eval()
                ensemble.append(m)
            except Exception as e:
                logger.warning(f"Could not load {ckpt}: {e}")

        if ensemble:
            inst = cls(ensemble[0])
            inst._ensemble = ensemble
            logger.info(f"N-HiTS ensemble of {len(ensemble)} fold(s) ready")
            return inst

        path = os.path.join(MODEL_DIR, "nhits_latest.ckpt")
        if not os.path.exists(path):
            return None
        try:
            model = NHiTS.load_from_checkpoint(path)
            return cls(model)
        except Exception as e:
            logger.error(f"Failed to load N-HiTS: {e}")
            return None

    def predict_latest(self, features: pd.DataFrame) -> tuple[float, float, float]:
        if self._model is None:
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
            avg = torch.stack(all_probs).mean(dim=0).numpy()
            return float(avg[0]), float(avg[1]), float(avg[2])
        except Exception as e:
            logger.error(f"N-HiTS prediction failed: {e}")
            return 0.333, 0.334, 0.333
