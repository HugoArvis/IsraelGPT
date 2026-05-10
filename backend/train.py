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
}


def print_menu():
    print("\n" + "=" * 62)
    print("  Trading AI — Training Launcher")
    print("=" * 62)
    for key, p in PROFILES.items():
        print(f"\n  [{key}] {p['name']}")
        for line in p["description"].splitlines():
            print(f"      {line}")
    print("\n  [q] Quit")
    print("=" * 62)


def pick_profile() -> dict | None:
    print_menu()
    while True:
        choice = input("\nSelect profile: ").strip().lower()
        if choice == "q":
            return None
        if choice in PROFILES:
            return PROFILES[choice]
        print("  Invalid choice — enter 1, 2, 3, or q.")


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

    # Train 5d and 21d only — 1d is too noisy for directional signals
    for horizon in ("5d", "21d"):
        print(f"\n  Training LightGBM [{horizon}]...")
        m = LGBMModel.train(features, horizon=horizon)
        m.save()
        print(f"  [{horizon}] saved.")


def run_tft(features, profile):
    from training.train_supervised import train_supervised
    print(f"\n  Training TFT ({profile['n_folds']} fold(s))...")
    return train_supervised(features, n_folds=profile["n_folds"])


def main():
    profile = pick_profile()
    if profile is None:
        print("Cancelled.")
        return

    print(f"\n  Starting: {profile['name']}")
    if profile["tickers"]:
        print(f"  Tickers : {profile['tickers']}")
    else:
        print("  Tickers : all 15")
    print(f"  Folds   : {profile['n_folds']}\n")

    from data.feature_engineering import build_features
    print("Building features...")
    features = build_features()

    if profile["tickers"]:
        features = features[features["ticker"].isin(profile["tickers"])]
        print(f"  Filtered to {len(features)} rows\n")

    mode = profile["mode"]

    if mode == "lgbm":
        run_lgbm_all_horizons(features, profile)
        print("\n" + "=" * 50)
        print("  Training complete — LightGBM checkpoints saved")
        print("  Models: lgbm_5d.pkl  lgbm_21d.pkl")
        print("=" * 50)
        print("\n  Next step: py -m training.evaluate\n")

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
