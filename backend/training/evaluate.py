import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from config import ENCODER_LENGTH, PREDICTION_HORIZON
from data.feature_engineering import FEATURE_COLUMNS, build_features
from models.tft_model import TFTModel


def walk_forward_evaluate(features_df: pd.DataFrame, n_folds: int = 5) -> dict:
    """
    Evaluate TFT directional accuracy on out-of-sample walk-forward folds.
    Exit criterion: directional accuracy > 55%.
    """
    from training.dataset import build_timeseries_dataset
    import torch

    all_dates = sorted(features_df.index.unique())
    fold_size = len(all_dates) // (n_folds + 1)
    all_preds, all_labels = [], []

    model_wrapper = TFTModel.load_latest()
    if model_wrapper is None or model_wrapper._model is None:
        logger.error("No trained model to evaluate")
        return {}

    model = model_wrapper._model
    model.eval()

    for fold in range(1, n_folds + 1):
        val_start = all_dates[fold_size * fold]
        val_end = all_dates[min(fold_size * (fold + 1) - 1, len(all_dates) - 1)]
        val_df = features_df[(features_df.index >= val_start) & (features_df.index <= val_end)]

        if val_df.empty:
            continue

        try:
            dataset = build_timeseries_dataset(features_df, cutoff_date=str(val_end.date()))
            dl = dataset.to_dataloader(train=False, batch_size=32, num_workers=0)
            with torch.no_grad():
                preds = model.predict(dl)
            predicted_labels = preds.argmax(dim=-1).numpy().flatten()
            true_labels = val_df["label"].values[:len(predicted_labels)]
            all_preds.extend(predicted_labels.tolist())
            all_labels.extend(true_labels.tolist())
            fold_acc = accuracy_score(true_labels, predicted_labels)
            logger.info(f"Fold {fold}: accuracy={fold_acc:.3f}")
        except Exception as e:
            logger.error(f"Fold {fold} evaluation error: {e}")

    if not all_preds:
        return {}

    overall_acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["SELL", "HOLD", "BUY"], output_dict=True)
    logger.info(f"Overall directional accuracy: {overall_acc:.3f}")
    logger.info(f"Exit criterion (>55%): {'PASS' if overall_acc > 0.55 else 'FAIL'}")
    return {"accuracy": overall_acc, "report": report, "exit_criterion_met": overall_acc > 0.55}


if __name__ == "__main__":
    features = build_features()
    results = walk_forward_evaluate(features)
    print(results)
