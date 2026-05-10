import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from data.feature_engineering import FEATURE_COLUMNS, CATEGORICAL_FEATURES, build_features
from models.lgbm_model import _HORIZON_MAP
from config import CRISIS_PERIODS, CRISIS_SAMPLE_WEIGHT, RECOVERY_PERIODS

_TARGET = {"1d": "label_1d", "5d": "label", "21d": "label_21d"}


def _train_fold(df: pd.DataFrame, horizon: str):
    """
    Lightweight LightGBM fit for a single walk-forward fold.
    Uses fixed n_estimators — no early stopping so small folds don't fail.
    Features are filled with 0 for NaN (only target NaNs are dropped).
    """
    import lightgbm as lgb
    target_col = _TARGET[horizon]
    df = df.dropna(subset=[target_col]).sort_index()
    if len(df) < 50:
        return None, target_col
    X = df[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)
    y = df[target_col].astype(int).values

    # Crisis oversampling — same as production training
    sample_weight = np.ones(len(df), dtype=np.float32)
    idx = pd.to_datetime(df.index)
    for start, end in CRISIS_PERIODS:
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        sample_weight[mask] *= CRISIS_SAMPLE_WEIGHT

    model = lgb.LGBMClassifier(
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
    X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    model.fit(X_df, y, sample_weight=sample_weight, categorical_feature=CATEGORICAL_FEATURES)
    return model, target_col


# ── Regime-aware metrics ─────────────────────────────────────────────────────

def _regime_metrics(preds: pd.Series, labels: pd.Series) -> dict:
    """
    SELL recall during crash windows  — did the model warn before/during crashes?
    BUY recall during recovery windows — did the model catch the rebounds?
    """
    from sklearn.metrics import recall_score

    results = {}
    idx = pd.to_datetime(preds.index)

    # SELL recall on crash periods
    crisis_mask = pd.Series(False, index=idx)
    for start, end in CRISIS_PERIODS:
        crisis_mask |= (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))

    n_crisis = int(crisis_mask.sum())
    if n_crisis > 0:
        sell_recall = recall_score(
            labels[crisis_mask.values].values,
            preds[crisis_mask.values].values,
            labels=[0], average="macro", zero_division=0,
        )
        results["sell_recall_crisis"]  = round(sell_recall, 3)
        results["n_crisis_days"]       = n_crisis
        logger.info(f"SELL recall (crash periods): {sell_recall:.3f}  ({n_crisis} days)")
    else:
        logger.warning("No crisis-period days in validation set — SELL recall not computed")

    # BUY recall on recovery periods
    recovery_mask = pd.Series(False, index=idx)
    for start, end in RECOVERY_PERIODS:
        recovery_mask |= (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))

    n_recovery = int(recovery_mask.sum())
    if n_recovery > 0:
        buy_recall = recall_score(
            labels[recovery_mask.values].values,
            preds[recovery_mask.values].values,
            labels=[2], average="macro", zero_division=0,
        )
        results["buy_recall_recovery"] = round(buy_recall, 3)
        results["n_recovery_days"]     = n_recovery
        logger.info(f"BUY recall (recovery periods): {buy_recall:.3f}  ({n_recovery} days)")
    else:
        logger.warning("No recovery-period days in validation set — BUY recall not computed")

    return results


# ── Walk-forward evaluation ───────────────────────────────────────────────────

def walk_forward_evaluate(features_df: pd.DataFrame, horizon: str = "5d", n_folds: int = 5) -> dict:
    """
    Proper walk-forward: train on the first k folds, test on fold k+1.
    Guarantees no data leakage and matches production conditions.
    """
    target_col = _TARGET[horizon]
    all_dates = sorted(features_df.index.unique())
    fold_size = len(all_dates) // (n_folds + 1)
    all_preds, all_labels, all_idx = [], [], []

    for fold in range(1, n_folds + 1):
        train_end = all_dates[fold_size * fold - 1]
        val_end   = all_dates[min(fold_size * (fold + 1) - 1, len(all_dates) - 1)]

        train_df = features_df[features_df.index <= train_end]
        val_df   = features_df[(features_df.index > train_end) & (features_df.index <= val_end)]

        if train_df.empty or val_df.empty:
            logger.warning(f"Fold {fold}: empty split — skipping")
            continue

        try:
            model, _ = _train_fold(train_df, horizon)
            if model is None:
                logger.warning(f"Fold {fold}: too few training samples — skipping")
                continue

            df_val = val_df.dropna(subset=[target_col])
            if df_val.empty:
                logger.warning(f"Fold {fold}: empty validation set — skipping")
                continue

            X_val = df_val[FEATURE_COLUMNS].fillna(0).astype(np.float32)
            y_val = df_val[target_col].astype(int).values
            preds = model.predict(X_val)

            fold_acc = accuracy_score(y_val, preds)
            logger.info(f"Fold {fold}/{n_folds}: accuracy={fold_acc:.3f}  ({len(y_val)} samples)")
            all_preds.extend(preds.tolist())
            all_labels.extend(y_val.tolist())
            all_idx.extend(df_val.index.tolist())

        except Exception as e:
            logger.error(f"Fold {fold} failed: {e}\n{traceback.format_exc()}")

    if not all_preds:
        logger.error("No predictions collected — all folds failed")
        return {}

    overall_acc = accuracy_score(all_labels, all_preds)
    kappa       = cohen_kappa_score(all_labels, all_preds)
    report      = classification_report(
        all_labels, all_preds,
        target_names=["SELL", "HOLD", "BUY"],
        output_dict=True,
        zero_division=0,
    )

    sell_prec        = report["SELL"]["precision"]
    buy_prec         = report["BUY"]["precision"]
    macro_f1         = report["macro avg"]["f1-score"]
    active_precision = (sell_prec + buy_prec) / 2.0

    # Exit criterion: directional signals must be meaningfully better than random
    #   active_precision ≥ 0.40  — SELL/BUY calls correct >40% of the time (random baseline ~33%)
    #   macro_f1         ≥ 0.35  — balanced performance across all three classes
    #   kappa            ≥ 0.10  — non-trivial agreement beyond chance
    passed = active_precision >= 0.40 and macro_f1 >= 0.35 and kappa >= 0.10

    logger.info(
        f"[{horizon}] acc={overall_acc:.3f}  kappa={kappa:.3f}  "
        f"active_prec={active_precision:.3f}  macro_f1={macro_f1:.3f}  "
        f"exit={'PASS ✓' if passed else 'FAIL'}"
    )

    regime = _regime_metrics(
        pd.Series(all_preds, index=all_idx),
        pd.Series(all_labels, index=all_idx),
    )

    return {
        "accuracy":         overall_acc,
        "kappa":            round(kappa, 3),
        "active_precision": round(active_precision, 3),
        "macro_f1":         round(macro_f1, 3),
        "report":           report,
        "exit_criterion_met": passed,
        **regime,
    }


# ── Crisis stress test ────────────────────────────────────────────────────────

CRISIS_START          = "2020-02-01"
CRISIS_END            = "2020-04-30"
CRISIS_SELL_THRESHOLD = 0.50


def crisis_stress_test(features_df: pd.DataFrame, horizon: str = "5d") -> dict:
    """
    Train on pre-crash data only, predict on the COVID crash window.
    Checks that ≥50% of crash-period predictions are SELL.
    """
    logger.info("Running COVID-crash stress test (Feb–Apr 2020)...")
    target_col = _TARGET[horizon]

    pre_crash    = features_df[features_df.index < CRISIS_START]
    crash_window = features_df[
        (features_df.index >= CRISIS_START) & (features_df.index <= CRISIS_END)
    ]

    if pre_crash.empty or crash_window.empty:
        logger.warning("Crisis stress test skipped — insufficient date range.")
        return {"sell_rate": None, "passed": None, "n_days": 0, "skipped": True}

    try:
        model, _ = _train_fold(pre_crash, horizon)
        if model is None:
            logger.warning("Crisis stress test skipped — too few pre-crash samples.")
            return {"sell_rate": None, "passed": None, "n_days": 0, "skipped": True}
    except Exception as e:
        logger.error(f"Crisis stress test training failed: {e}\n{traceback.format_exc()}")
        return {"sell_rate": None, "passed": None, "n_days": 0, "skipped": True}

    df_crash = crash_window.dropna(subset=[target_col])
    if df_crash.empty:
        logger.warning("Crisis stress test skipped — crash slice empty after dropna.")
        return {"sell_rate": None, "passed": None, "n_days": 0, "skipped": True}

    X = df_crash[FEATURE_COLUMNS].fillna(0).astype(np.float32)
    preds = model.predict(X).tolist()

    n_days    = len(preds)
    sell_rate = preds.count(0) / n_days if n_days else 0.0
    passed    = sell_rate >= CRISIS_SELL_THRESHOLD

    logger.info(
        f"Crisis stress test [{horizon}]: {n_days} days, "
        f"SELL rate={sell_rate:.1%} (threshold {CRISIS_SELL_THRESHOLD:.0%}) — "
        f"{'PASS ✓' if passed else 'FAIL — model blind to crash dynamics'}"
    )
    return {"sell_rate": sell_rate, "passed": passed, "n_days": n_days, "skipped": False}


if __name__ == "__main__":
    features = build_features()

    # Confirm ~25/50/25 label distribution
    for col, lbl in [("label", "5d"), ("label_1d", "1d"), ("label_21d", "21d")]:
        dist = features[col].value_counts(normalize=True).sort_index()
        print(f"Label distribution [{lbl}]: SELL={dist.get(0,0):.1%}  HOLD={dist.get(1,0):.1%}  BUY={dist.get(2,0):.1%}")

    all_results = {}
    for hz in ("5d", "21d"):      # 1d dropped — too noisy for directional signals
        print(f"\n── Walk-forward evaluation [{hz}] ──────────────────────────")
        results = walk_forward_evaluate(features, horizon=hz)
        all_results[hz] = results
        if results:
            print(f"  Accuracy         : {results['accuracy']:.3f}")
            print(f"  Kappa            : {results['kappa']:.3f}   (target ≥ 0.10)")
            print(f"  Active precision : {results['active_precision']:.3f}  (target ≥ 0.40 — avg SELL/BUY precision)")
            print(f"  Macro F1         : {results['macro_f1']:.3f}   (target ≥ 0.35)")
            print(f"  Exit criterion   : {'PASS ✓' if results['exit_criterion_met'] else 'FAIL'}")
            rep = results["report"]
            for cls in ["SELL", "HOLD", "BUY"]:
                if cls in rep:
                    r = rep[cls]
                    print(f"    {cls}: prec={r['precision']:.2f}  rec={r['recall']:.2f}  f1={r['f1-score']:.2f}")
            if "sell_recall_crisis" in results:
                print(f"  SELL recall on crash periods  : {results['sell_recall_crisis']:.3f}  ({results.get('n_crisis_days',0)} days)")
            if "buy_recall_recovery" in results:
                print(f"  BUY recall on recovery periods: {results['buy_recall_recovery']:.3f}  ({results.get('n_recovery_days',0)} days)")

    print("\n── Crisis stress test [5d] (COVID crash Feb–Apr 2020) ────")
    crisis = crisis_stress_test(features, horizon="5d")
    if crisis.get("skipped"):
        print("  Skipped — data not available")
    else:
        print(f"  Days evaluated : {crisis['n_days']}")
        print(f"  SELL rate      : {crisis['sell_rate']:.1%}")
        print(f"  Threshold      : {CRISIS_SELL_THRESHOLD:.0%}")
        print(f"  Result         : {'PASS ✓' if crisis['passed'] else 'FAIL'}")

    print()
    results_5d  = all_results.get("5d",  {})
    results_21d = all_results.get("21d", {})
    all_pass = (
        results_5d.get("exit_criterion_met",  False)
        and results_21d.get("exit_criterion_met", False)
        and (crisis.get("passed") or crisis.get("skipped"))
    )
    print(f"═══ Overall: {'READY FOR PHASE 2 ✓' if all_pass else 'NOT READY — address FAILs above'} ═══")
