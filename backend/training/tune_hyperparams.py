"""
Hyperparameter tuning with Optuna.
Uses a 60/20/20 temporal split: train → val (Optuna objective) → test (final report).
Best params saved as JSON in MODEL_DIR and auto-loaded by LGBMModel / CatBoostModel.

Usage:
    py training/tune_hyperparams.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from loguru import logger
from config import MODEL_DIR, TEMPORAL_WEIGHT_HALF_LIFE, CRISIS_PERIODS, CRISIS_SAMPLE_WEIGHT
from data.feature_engineering import FEATURE_COLUMNS, CATEGORICAL_FEATURES

_REGRESSION_TARGETS = {"5d": "relative_return", "21d": "relative_return_21d"}


# ── Shared helpers ────────────────────────────────────────────────────────────

def three_way_split(features: pd.DataFrame, train_pct: float = 0.60, val_pct: float = 0.20):
    """60/20/20 temporal split — no data leakage across boundaries."""
    all_dates = sorted(pd.to_datetime(features.index).unique())
    train_end = all_dates[int(len(all_dates) * train_pct)]
    val_end   = all_dates[int(len(all_dates) * (train_pct + val_pct))]
    idx = pd.to_datetime(features.index)
    train_df = features[idx <  train_end]
    val_df   = features[(idx >= train_end) & (idx < val_end)]
    test_df  = features[idx >= val_end]
    return train_df, val_df, test_df, train_end, val_end


def _compute_weights(df: pd.DataFrame) -> np.ndarray:
    """Same temporal decay + crisis oversampling used in model.train()."""
    weights = np.ones(len(df), dtype=np.float32)
    if TEMPORAL_WEIGHT_HALF_LIFE is not None:
        dates = pd.to_datetime(df.index).to_series()
        days_from_end = (dates.max() - dates).dt.days.values
        decay = np.log(2) / TEMPORAL_WEIGHT_HALF_LIFE
        weights = np.exp(-decay * days_from_end).astype(np.float32)
    idx = pd.to_datetime(df.index)
    for start, end in CRISIS_PERIODS:
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        weights[mask] *= CRISIS_SAMPLE_WEIGHT
    return weights


def save_best_params(params: dict, model_name: str, horizon: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{model_name}_best_params_{horizon}.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Best params saved → {path}")


def load_best_params(model_name: str, horizon: str) -> dict | None:
    path = os.path.join(MODEL_DIR, f"{model_name}_best_params_{horizon}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── LightGBM tuning ───────────────────────────────────────────────────────────

def tune_lgbm(train_df: pd.DataFrame, val_df: pd.DataFrame,
              horizon: str, n_trials: int = 40) -> dict:
    """
    Optuna study for LightGBM. Uses early stopping to find optimal n_estimators.
    Objective: macro-F1 on the validation set (unweighted — real-world performance).
    """
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    target_col = _REGRESSION_TARGETS[horizon]

    train_df = train_df.dropna(subset=[target_col]).sort_index()
    val_df   = val_df.dropna(subset=[target_col]).sort_index()

    X_train = train_df[FEATURE_COLUMNS].fillna(0).astype(np.float32)
    y_train = train_df[target_col].astype(np.float32).values
    X_val   = val_df[FEATURE_COLUMNS].fillna(0).astype(np.float32)
    y_val   = val_df[target_col].astype(np.float32).values
    weights = _compute_weights(train_df)

    cat_indices = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES if c in FEATURE_COLUMNS]

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators      = 1500,   # capped by early stopping
            learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 31, 127),
            max_depth         = trial.suggest_int("max_depth", 4, 9),
            min_child_samples = trial.suggest_int("min_child_samples", 20, 100),
            subsample         = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 3.0),
            alpha             = trial.suggest_float("alpha", 0.5, 2.0),  # Huber threshold
            objective         = "huber",
            random_state      = 42,
            n_jobs            = -1,
            verbose           = -1,
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=weights,
            categorical_feature=cat_indices,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(60, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        trial.set_user_attr("best_iteration", model.best_iteration_)
        preds = model.predict(X_val)
        # Spearman IC: rank correlation between predicted and actual returns
        from scipy import stats
        ic, _ = stats.spearmanr(preds, y_val)
        return float(ic) if not np.isnan(ic) else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params.copy()
    best["n_estimators"] = study.best_trial.user_attrs.get("best_iteration", 800)
    best["n_estimators"] = max(100, int(best["n_estimators"] * 1.05))  # small buffer
    logger.info(f"LightGBM [{horizon}] best val macro-F1={study.best_value:.4f} | params={best}")
    return best


# ── CatBoost tuning ───────────────────────────────────────────────────────────

def tune_catboost(train_df: pd.DataFrame, val_df: pd.DataFrame,
                  horizon: str, n_trials: int = 25) -> dict:
    """
    Optuna study for CatBoost. Uses early stopping (iterations=1000).
    Objective: macro-F1 on the validation set.
    """
    import optuna
    from catboost import CatBoostRegressor, Pool
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    target_col = _REGRESSION_TARGETS[horizon]

    train_df = train_df.dropna(subset=[target_col]).sort_index()
    val_df   = val_df.dropna(subset=[target_col]).sort_index()

    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df[target_col].astype(np.float32).values
    X_val   = val_df[FEATURE_COLUMNS].fillna(0)
    y_val   = val_df[target_col].astype(np.float32).values
    weights = _compute_weights(train_df)

    cat_indices = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES if c in FEATURE_COLUMNS]
    train_pool = Pool(X_train, y_train, weight=weights, cat_features=cat_indices)
    val_pool   = Pool(X_val,   y_val,                   cat_features=cat_indices)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            iterations          = 1000,
            learning_rate       = trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            depth               = trial.suggest_int("depth", 4, 8),
            l2_leaf_reg         = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
            loss_function       = "RMSE",
            random_seed         = 42,
            verbose             = 0,
            early_stopping_rounds = 60,
        )
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool)
        trial.set_user_attr("best_iteration", model.best_iteration_)
        preds = model.predict(X_val).astype(np.float32)
        from scipy import stats
        ic, _ = stats.spearmanr(preds, y_val)
        return float(ic) if not np.isnan(ic) else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params.copy()
    best["iterations"] = study.best_trial.user_attrs.get("best_iteration", 800)
    best["iterations"] = max(100, int(best["iterations"] * 1.05))
    logger.info(f"CatBoost [{horizon}] best val macro-F1={study.best_value:.4f} | params={best}")
    return best


# ── Full tuning pipeline ──────────────────────────────────────────────────────

def run_tuning(features: pd.DataFrame, n_lgbm: int = 40, n_catboost: int = 25):
    """
    Full pipeline:
      1. 60/20/20 temporal split
      2. Optuna: tune LightGBM and CatBoost on train→val
      3. Save best params as JSON (auto-loaded by model.train() from now on)
      4. Retrain final models on train+val with best params
      5. Report test-set accuracy for both models, both horizons
    """
    from models.lgbm_model import LGBMModel
    from models.catboost_model import CatBoostModel
    from sklearn.metrics import accuracy_score, f1_score

    train_df, val_df, test_df, train_cut, val_cut = three_way_split(features)
    train_val_df = pd.concat([train_df, val_df])

    print(f"\n  Train : up to {train_cut.date()}  ({len(train_df):,} rows)")
    print(f"  Val   : {train_cut.date()} → {val_cut.date()}  ({len(val_df):,} rows)")
    print(f"  Test  : from {val_cut.date()}  ({len(test_df):,} rows)\n")

    summary = {}

    for horizon in ("5d", "21d"):
        summary[horizon] = {}

        # ── LightGBM ──────────────────────────────────────────────
        print(f"\n  [LightGBM {horizon}] Tuning ({n_lgbm} trials)...")
        lgbm_params = tune_lgbm(train_df, val_df, horizon, n_trials=n_lgbm)
        save_best_params(lgbm_params, "lgbm", horizon)

        print(f"  [LightGBM {horizon}] Retraining on train+val with best params...")
        lgbm_final = LGBMModel.train(train_val_df, horizon=horizon)
        lgbm_final.save()
        lgbm_result = lgbm_final.evaluate(test_df)
        summary[horizon]["lgbm"] = lgbm_result

        # ── CatBoost ──────────────────────────────────────────────
        print(f"\n  [CatBoost {horizon}] Tuning ({n_catboost} trials)...")
        cb_params = tune_catboost(train_df, val_df, horizon, n_trials=n_catboost)
        save_best_params(cb_params, "catboost", horizon)

        print(f"  [CatBoost {horizon}] Retraining on train+val with best params...")
        cb_final = CatBoostModel.train(train_val_df, horizon=horizon)
        cb_final.save()
        cb_result = cb_final.evaluate(test_df)
        summary[horizon]["catboost"] = cb_result

    # ── Results table ──────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  Post-Tuning Test Results (models trained on 80%, evaluated on 20%)")
    print("=" * 74)
    print(f"  {'Horizon':<8} {'Model':<12} {'Dir.Acc':>8}  {'IC':>8}  {'MAE%':>6}  {'RMSE%':>6}")
    print("-" * 74)
    for horizon in ("5d", "21d"):
        for name, label in (("lgbm", "LightGBM"), ("catboost", "CatBoost")):
            res  = summary[horizon].get(name, {})
            da   = res.get("direction_accuracy", float("nan"))
            ic   = res.get("spearman_ic",        float("nan"))
            mae  = res.get("mae",                float("nan")) * 100
            rmse = res.get("rmse",               float("nan")) * 100
            print(f"  {horizon:<8} {label:<12} {da:>7.1%}  {ic:>8.4f}  {mae:>5.2f}%  {rmse:>5.2f}%")
        print()
    print("=" * 74)
    print("\n  Best params saved — model.train() will use them automatically.\n")


if __name__ == "__main__":
    from data.feature_engineering import build_features
    from train import _apply_rolling_window
    features = build_features()
    features = _apply_rolling_window(features)
    run_tuning(features)
