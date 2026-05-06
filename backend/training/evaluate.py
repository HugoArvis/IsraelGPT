import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from config import ENCODER_LENGTH, PREDICTION_HORIZON
from data.feature_engineering import FEATURE_COLUMNS, build_features
from models.tft_model import TFTModel
from training.dataset import build_timeseries_dataset


def walk_forward_evaluate(features_df: pd.DataFrame, n_folds: int = 5) -> dict:
    """
    Evaluate directional accuracy on walk-forward folds.
    Exit criterion: accuracy > 55%.
    Collects predictions and labels from the same batch to guarantee alignment.
    """
    model_wrapper = TFTModel.load_latest()
    if model_wrapper is None or model_wrapper._model is None:
        logger.error("No trained model to evaluate")
        return {}

    model = model_wrapper._model
    model.eval()

    all_dates = sorted(features_df.index.unique())
    fold_size = len(all_dates) // (n_folds + 1)
    all_preds, all_labels = [], []

    for fold in range(1, n_folds + 1):
        val_cut = str(all_dates[min(fold_size * (fold + 1) - 1, len(all_dates) - 1)].date())

        try:
            val_ds = build_timeseries_dataset(features_df, cutoff_date=val_cut)
            val_dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

            fold_preds, fold_labels = [], []

            with torch.no_grad():
                for x, y in val_dl:
                    out = model(x)
                    # out["prediction"]: (batch, horizon, n_classes)
                    batch_preds = out["prediction"][:, 0, :].argmax(dim=-1).cpu().numpy()
                    # y is (target_tensor, weight) — target: (batch, horizon) as encoded ints
                    target = y[0] if isinstance(y, (list, tuple)) else y
                    batch_labels = target[:, 0].long().cpu().numpy()
                    fold_preds.extend(batch_preds.tolist())
                    fold_labels.extend(batch_labels.tolist())

            fold_acc = accuracy_score(fold_labels, fold_preds)
            logger.info(f"Fold {fold}: accuracy={fold_acc:.3f}  ({len(fold_preds)} samples)")
            all_preds.extend(fold_preds)
            all_labels.extend(fold_labels)

        except Exception as e:
            logger.error(f"Fold {fold} evaluation error: {e}")

    if not all_preds:
        logger.error("No predictions collected — all folds failed")
        return {}

    overall_acc = accuracy_score(all_labels, all_preds)
    passed = overall_acc > 0.55
    report = classification_report(
        all_labels, all_preds,
        target_names=["SELL", "HOLD", "BUY"],
        output_dict=True,
        zero_division=0,
    )
    logger.info(f"Overall accuracy: {overall_acc:.3f}")
    logger.info(f"Exit criterion (>55%): {'PASS ✓' if passed else 'FAIL — retrain or tune config.py'}")
    return {"accuracy": overall_acc, "report": report, "exit_criterion_met": passed}


if __name__ == "__main__":
    features = build_features()
    results = walk_forward_evaluate(features)
    if results:
        print(f"\nAccuracy : {results['accuracy']:.3f}")
        print(f"Exit criterion (>55%): {'PASS' if results['exit_criterion_met'] else 'FAIL'}")
        rep = results["report"]
        for cls in ["SELL", "HOLD", "BUY"]:
            if cls in rep:
                print(f"  {cls}: precision={rep[cls]['precision']:.2f}  recall={rep[cls]['recall']:.2f}  f1={rep[cls]['f1-score']:.2f}")
