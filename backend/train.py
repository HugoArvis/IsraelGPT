"""
Training launcher — run this instead of calling training scripts directly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROFILES = {
    "1": {
        "name": "LightGBM — Full (all 15 tickers)",
        "description": (
            "All 15 tickers · 2 horizons (5d / 21d)\n"
            "  Training time: < 2 min"
        ),
        "tickers": None,
        "n_folds": 5,
        "mode": "lgbm",
    },
    "2": {
        "name": "LightGBM — Quick test (3 tickers)",
        "description": (
            "AAPL · MSFT · GOOGL · 2 horizons (5d / 21d)\n"
            "  Training time: < 15 sec"
        ),
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "n_folds": 3,
        "mode": "lgbm",
    },
    "3": {
        "name": "TFT — Full (GPU recommended)",
        "description": (
            "All 15 tickers · hidden=32 · 50 epochs max · 5 folds\n"
            "  RTX 3060: ~40-75 min"
        ),
        "tickers": None,
        "hidden_size": 32,
        "batch_size": 64,
        "max_epochs": 50,
        "n_folds": 5,
        "encoder_length": 60,
        "mode": "tft",
    },
    "4": {
        "name": "CatBoost — Full (all 15 tickers)",
        "description": (
            "All 15 tickers · 2 horizons (5d / 21d)\n"
            "  Training time: < 5 min"
        ),
        "tickers": None,
        "n_folds": 5,
        "mode": "catboost",
    },
    "5": {
        "name": "Compare LightGBM vs CatBoost — Full",
        "description": (
            "60/20/20 temporal split · both models · both horizons\n"
            "  Val set for model selection · test set for final honest accuracy\n"
            "  Training time: < 15 min"
        ),
        "tickers": None,
        "mode": "compare",
    },
    "6": {
        "name": "Compare LightGBM vs CatBoost — Quick (3 tickers)",
        "description": (
            "AAPL · MSFT · GOOGL only — fast comparison run\n"
            "  Training time: < 1 min"
        ),
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "mode": "compare",
    },
    "7": {
        "name": "Hyperparameter Tuning — LightGBM + CatBoost (Optuna)",
        "description": (
            "40 trials/horizon for LightGBM · 25 trials/horizon for CatBoost\n"
            "  Saves best params → auto-loaded by all future training runs\n"
            "  Training time: ~30-60 min"
        ),
        "tickers": None,
        "mode": "tune",
        "n_lgbm": 40,
        "n_catboost": 25,
    },
    "8": {
        "name": "Hyperparameter Tuning — Quick (10 trials each)",
        "description": (
            "Fast tuning run to verify setup · 10 trials each model/horizon\n"
            "  Training time: ~5-10 min"
        ),
        "tickers": None,
        "mode": "tune",
        "n_lgbm": 10,
        "n_catboost": 10,
    },
}


def print_menu():
    print("\n" + "=" * 66)
    print("  Trading AI — Training Launcher")
    print("=" * 66)
    for key, p in PROFILES.items():
        print(f"\n  [{key}] {p['name']}")
        for line in p["description"].splitlines():
            print(f"      {line}")
    print("\n  [q] Quit")
    print("=" * 66)


def pick_profile() -> dict | None:
    print_menu()
    while True:
        choice = input("\nSelect profile: ").strip().lower()
        if choice == "q":
            return None
        if choice in PROFILES:
            return PROFILES[choice]
        print("  Invalid choice — enter 1–8 or q.")


def _apply_rolling_window(features):
    """Filter features to the last ROLLING_WINDOW_YEARS of data before training.
    Addresses regime non-stationarity — old dynamics hurt recent-period accuracy."""
    import pandas as pd
    from config import ROLLING_WINDOW_YEARS
    if ROLLING_WINDOW_YEARS is None:
        return features
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=ROLLING_WINDOW_YEARS)
    windowed = features[pd.to_datetime(features.index) >= cutoff]
    excluded = len(features) - len(windowed)
    print(f"  Rolling window ({ROLLING_WINDOW_YEARS}y): {len(windowed):,} rows kept, {excluded:,} older rows excluded\n")
    return windowed


def apply_tft_profile(profile: dict):
    import config as cfg
    cfg.TFT_HIDDEN_SIZE  = profile["hidden_size"]
    cfg.TFT_BATCH_SIZE   = profile["batch_size"]
    cfg.TFT_MAX_EPOCHS   = profile["max_epochs"]
    cfg.ENCODER_LENGTH   = profile["encoder_length"]
    if profile["tickers"] is not None:
        cfg.TICKERS = profile["tickers"]


def run_lgbm_all_horizons(features, profile):
    from models.lgbm_model import LGBMModel
    for horizon in ("5d", "21d"):
        print(f"\n  Training LightGBM [{horizon}]...")
        m = LGBMModel.train(features, horizon=horizon)
        m.save()
        print(f"  [{horizon}] saved.")


def run_catboost_all_horizons(features, profile):
    from models.catboost_model import CatBoostModel
    for horizon in ("5d", "21d"):
        print(f"\n  Training CatBoost [{horizon}]...")
        m = CatBoostModel.train(features, horizon=horizon)
        m.save()
        print(f"  [{horizon}] saved.")


def run_tft(features, profile):
    from training.train_supervised import train_supervised
    print(f"\n  Training TFT ({profile['n_folds']} fold(s))...")
    return train_supervised(features, n_folds=profile["n_folds"])


def _three_way_split(features):
    """60/20/20 temporal split — val for model selection, test for final honest eval."""
    import pandas as pd
    from training.tune_hyperparams import three_way_split
    return three_way_split(features)


def _print_comparison_table(results: dict):
    """Regression metrics: direction accuracy, Spearman IC, MAE for both models."""
    print("\n" + "=" * 84)
    print("  LightGBM vs CatBoost — Regression: predict forward relative return vs SP500")
    print("  60/20/20 temporal split  |  Val = model selection  |  Test = honest eval")
    print("=" * 84)
    hdr = (f"  {'Horizon':<8} {'Model':<12} {'Val DirAcc':>10}  {'Test DirAcc':>11}  "
           f"{'IC (test)':>9}  {'MAE%':>6}  {'RMSE%':>6}")
    print(hdr)
    print("-" * 84)

    for horizon in ("5d", "21d"):
        for model_name, label in (("lgbm", "LightGBM"), ("catboost", "CatBoost")):
            res      = results.get(horizon, {}).get(model_name, {})
            val_da   = res.get("val",  {}).get("direction_accuracy", float("nan"))
            test_da  = res.get("test", {}).get("direction_accuracy", float("nan"))
            test_ic  = res.get("test", {}).get("spearman_ic",        float("nan"))
            test_mae = res.get("test", {}).get("mae",                 float("nan")) * 100
            test_rmse= res.get("test", {}).get("rmse",                float("nan")) * 100
            print(f"  {horizon:<8} {label:<12} {val_da:>9.1%}  {test_da:>10.1%}  "
                  f"{test_ic:>9.4f}  {test_mae:>5.2f}%  {test_rmse:>5.2f}%")
        print()

    print("-" * 84)
    print("  IC > 0.02 = useful signal  |  Dir. Acc > 52% = consistent directional edge")
    for horizon in ("5d", "21d"):
        lgbm_ic = results.get(horizon, {}).get("lgbm",     {}).get("test", {}).get("spearman_ic", 0)
        cb_ic   = results.get(horizon, {}).get("catboost", {}).get("test", {}).get("spearman_ic", 0)
        if lgbm_ic > cb_ic:
            winner = f"LightGBM  (IC {lgbm_ic:.4f} vs {cb_ic:.4f})"
        elif cb_ic > lgbm_ic:
            winner = f"CatBoost  (IC {cb_ic:.4f} vs {lgbm_ic:.4f})"
        else:
            winner = "Tie"
        print(f"  {horizon} winner: {winner}")
    print("=" * 84)


def run_comparison(features, profile):
    import pandas as pd
    from models.lgbm_model import LGBMModel
    from models.catboost_model import CatBoostModel

    train_df, val_df, test_df, train_cut, val_cut = _three_way_split(features)
    train_val_df = pd.concat([train_df, val_df])

    print(f"\n  Train : up to {train_cut.date()}  ({len(train_df):,} rows)")
    print(f"  Val   : {train_cut.date()} → {val_cut.date()}  ({len(val_df):,} rows)")
    print(f"  Test  : from {val_cut.date()}  ({len(test_df):,} rows)\n")
    print("  NOTE: accuracy reported from models trained on 60% (train only).")
    print("        Saved production models are retrained on 80% (train+val).\n")

    results = {}
    for horizon in ("5d", "21d"):
        results[horizon] = {}

        print(f"  Training LightGBM [{horizon}]...")
        lgbm_eval = LGBMModel.train(train_df, horizon=horizon)
        lgbm_prod = LGBMModel.train(train_val_df, horizon=horizon)
        lgbm_prod.save()
        results[horizon]["lgbm"] = {
            "val":  lgbm_eval.evaluate(val_df),
            "test": lgbm_eval.evaluate(test_df),
        }

        print(f"  Training CatBoost [{horizon}]...")
        cb_eval = CatBoostModel.train(train_df, horizon=horizon)
        cb_prod = CatBoostModel.train(train_val_df, horizon=horizon)
        cb_prod.save()
        results[horizon]["catboost"] = {
            "val":  cb_eval.evaluate(val_df),
            "test": cb_eval.evaluate(test_df),
        }

    _print_comparison_table(results)


def main():
    profile = pick_profile()
    if profile is None:
        print("Cancelled.")
        return

    print(f"\n  Starting: {profile['name']}")
    tickers = profile.get("tickers")
    if tickers:
        print(f"  Tickers : {tickers}")
    else:
        print("  Tickers : all 15")
    print()

    from data.feature_engineering import build_features
    print("Building features...")
    features = build_features()

    if tickers:
        features = features[features["ticker"].isin(tickers)]
        print(f"  Filtered to {len(features):,} rows\n")

    mode = profile["mode"]

    # Apply rolling window to focus training on recent regime (skipped for TFT — needs sequences)
    if mode in ("lgbm", "catboost", "compare"):
        features = _apply_rolling_window(features)

    if mode == "lgbm":
        run_lgbm_all_horizons(features, profile)
        print("\n" + "=" * 54)
        print("  Training complete — LightGBM checkpoints saved")
        print("  Models: lgbm_5d.pkl  lgbm_21d.pkl")
        print("=" * 54)
        print("\n  Next step: py -m training.evaluate\n")

    elif mode == "catboost":
        run_catboost_all_horizons(features, profile)
        print("\n" + "=" * 54)
        print("  Training complete — CatBoost checkpoints saved")
        print("  Models: catboost_5d.cbm  catboost_21d.cbm")
        print("=" * 54)
        print("\n  Next step: py -m training.evaluate\n")

    elif mode == "compare":
        run_comparison(features, profile)
        print("\n  Production models saved (trained on 80%). Run profile 7 to tune hyperparameters.\n")

    elif mode == "tune":
        from training.tune_hyperparams import run_tuning
        run_tuning(features, n_lgbm=profile["n_lgbm"], n_catboost=profile["n_catboost"])
        print("\n  Tuning complete. Re-run profile 1 or 5 to train/compare with best params.\n")

    elif mode == "tft":
        apply_tft_profile(profile)
        model = run_tft(features, profile)
        if model:
            print("\n  TFT training complete. Checkpoint saved to models/checkpoints/")
            print("  Next step: py training/evaluate.py\n")
        else:
            print("\n  Training produced no model — check errors above.\n")


if __name__ == "__main__":
    main()
