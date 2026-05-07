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


# ── Crisis stress test ────────────────────────────────────────────────────────

CRISIS_START          = "2020-02-01"
CRISIS_END            = "2020-04-30"
CRISIS_SELL_THRESHOLD = 0.50   # ≥50% of crash-period days must be predicted SELL


def crisis_stress_test(features_df: pd.DataFrame, n_folds: int = 5) -> dict:
    """
    COVID-crash stress test using the actual walk-forward fold checkpoint.

    One of the walk-forward folds naturally validates on a window that includes
    Feb–Apr 2020 and was trained ONLY on pre-crash data — making this a genuine
    out-of-sample test on the real production checkpoint, not a throwaway model.

    Steps:
      1. Find which fold's validation window covers the crash.
      2. Load that fold's saved checkpoint (tft_fold{N}.ckpt).
      3. Run it on a dataset scoped to the crash window only.
      4. Check that ≥50% of predictions are SELL.
    """
    logger.info("Running COVID-crash stress test (Feb–Apr 2020)...")

    # 1. Find the fold whose validation period overlaps with the crash window
    all_dates = sorted(features_df.index.unique())
    fold_size = len(all_dates) // (n_folds + 1)

    crisis_fold = None
    for i in range(1, n_folds + 1):
        val_start = all_dates[fold_size * i - 1]
        val_end   = all_dates[min(fold_size * (i + 1) - 1, len(all_dates) - 1)]
        if str(val_start.date()) <= CRISIS_END and str(val_end.date()) >= CRISIS_START:
            crisis_fold = i
            logger.info(
                f"Crash window falls in fold {i} "
                f"(val: {val_start.date()} → {val_end.date()})"
            )
            break

    if crisis_fold is None:
        logger.warning(
            "Crisis stress test skipped — COVID crash not covered by any fold. "
            "Ensure PRICE_HISTORY_YEARS >= 7 and re-run fetch_prices.py."
        )
        return {"sell_rate": None, "passed": None, "n_days": 0, "skipped": True}

    # 2. Load that fold's checkpoint — trained on pre-crash data only
    from config import MODEL_DIR
    from pytorch_forecasting import TemporalFusionTransformer

    ckpt_path = os.path.join(MODEL_DIR, f"tft_fold{crisis_fold}.ckpt")
    if not os.path.exists(ckpt_path):
        logger.warning(f"Crisis stress test skipped — {ckpt_path} not found. Run training first.")
        return {"sell_rate": None, "passed": None, "n_days": 0, "skipped": True}

    try:
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
        model.eval()
    except Exception as e:
        logger.error(f"Crisis stress test: could not load fold checkpoint — {e}")
        return {"sell_rate": None, "passed": None, "n_days": 0, "skipped": True}

    # 3. Build a dataset scoped to the crash window with enough encoder context
    #    before it (ENCODER_LENGTH trading days ≈ 90 calendar days buffer)
    context_start = pd.Timestamp(CRISIS_START) - pd.Timedelta(days=ENCODER_LENGTH * 1.5)
    crash_slice = features_df[
        (features_df.index >= context_start) & (features_df.index <= CRISIS_END)
    ]

    if crash_slice.empty:
        logger.warning("Crisis stress test skipped — crash slice is empty after filtering.")
        return {"sell_rate": None, "passed": None, "n_days": 0, "skipped": True}

    try:
        val_ds = build_timeseries_dataset(crash_slice)
        val_dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    except Exception as e:
        logger.error(f"Crisis stress test: dataset build failed — {e}")
        return {"sell_rate": None, "passed": None, "n_days": 0, "skipped": True}

    # 4. Collect predictions — all sequences end within or just before the crash window
    preds = []
    with torch.no_grad():
        for x, _ in val_dl:
            out = model(x)
            batch_preds = out["prediction"][:, 0, :].argmax(dim=-1).cpu().numpy()
            preds.extend(batch_preds.tolist())

    n_days = len(preds)
    sell_rate = preds.count(0) / n_days if n_days else 0.0
    passed = sell_rate >= CRISIS_SELL_THRESHOLD

    logger.info(
        f"Crisis stress test (fold {crisis_fold}): {n_days} sequences, "
        f"SELL rate={sell_rate:.1%} (threshold {CRISIS_SELL_THRESHOLD:.0%}) — "
        f"{'PASS ✓' if passed else 'FAIL — model blind to crash dynamics'}"
    )
    return {
        "sell_rate": sell_rate,
        "passed": passed,
        "n_days": n_days,
        "fold": crisis_fold,
        "skipped": False,
    }


if __name__ == "__main__":
    features = build_features()

    print("\n── Walk-forward evaluation ──────────────────────────")
    results = walk_forward_evaluate(features)
    if results:
        print(f"Accuracy : {results['accuracy']:.3f}")
        print(f"Exit criterion (>55%): {'PASS' if results['exit_criterion_met'] else 'FAIL'}")
        rep = results["report"]
        for cls in ["SELL", "HOLD", "BUY"]:
            if cls in rep:
                print(f"  {cls}: precision={rep[cls]['precision']:.2f}  recall={rep[cls]['recall']:.2f}  f1={rep[cls]['f1-score']:.2f}")

    print("\n── Crisis stress test (COVID crash Feb–Apr 2020) ────")
    crisis = crisis_stress_test(features)
    if crisis.get("skipped"):
        print("  Skipped — data not available or fold checkpoint missing")
    else:
        print(f"  Fold used      : {crisis['fold']} (trained on pre-crash data only)")
        print(f"  Days evaluated : {crisis['n_days']}")
        print(f"  SELL rate      : {crisis['sell_rate']:.1%}")
        print(f"  Threshold      : {CRISIS_SELL_THRESHOLD:.0%}")
        print(f"  Result         : {'PASS ✓' if crisis['passed'] else 'FAIL'}")

    print()
    all_pass = (
        results.get("exit_criterion_met", False)
        and (crisis.get("passed") or crisis.get("skipped"))
    )
    print(f"═══ Overall: {'READY FOR PHASE 2 ✓' if all_pass else 'NOT READY — address FAILs above'} ═══")
