"""
Training launcher — run this instead of calling train_supervised.py directly.
Presents a menu to choose between Full and Light training profiles.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Profile definitions ───────────────────────────────────────────────────────

PROFILES = {
    "1": {
        "name": "Full model",
        "description": (
            "All 15 tickers · hidden=64 · 50 epochs max · 5 folds\n"
            "  CPU: ~20-40 hours   |   RTX 3060: ~45-90 min"
        ),
        "tickers": None,           # None = use all from config
        "hidden_size": 64,
        "batch_size": 64,
        "max_epochs": 50,
        "n_folds": 5,
        "encoder_length": 60,
    },
    "2": {
        "name": "Light (test run)",
        "description": (
            "3 tickers · hidden=32 · 10 epochs max · 2 folds\n"
            "  CPU: ~30-60 min     |   RTX 3060: ~5-10 min"
        ),
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "hidden_size": 32,
        "batch_size": 128,
        "max_epochs": 10,
        "n_folds": 2,
        "encoder_length": 30,
    },
}

# ── Menu ─────────────────────────────────────────────────────────────────────

def print_menu():
    print("\n" + "=" * 58)
    print("  Trading AI — Training Launcher")
    print("=" * 58)
    for key, p in PROFILES.items():
        print(f"\n  [{key}] {p['name']}")
        for line in p["description"].splitlines():
            print(f"      {line}")
    print("\n  [q] Quit")
    print("=" * 58)


def pick_profile() -> dict | None:
    print_menu()
    while True:
        choice = input("\nSelect profile: ").strip().lower()
        if choice == "q":
            return None
        if choice in PROFILES:
            return PROFILES[choice]
        print("  Invalid choice — enter 1, 2, or q.")


# ── Apply profile overrides ───────────────────────────────────────────────────

def apply_profile(profile: dict):
    import config as cfg

    cfg.TFT_HIDDEN_SIZE = profile["hidden_size"]
    cfg.TFT_BATCH_SIZE = profile["batch_size"]
    cfg.TFT_MAX_EPOCHS = profile["max_epochs"]
    cfg.ENCODER_LENGTH = profile["encoder_length"]

    if profile["tickers"] is not None:
        cfg.TICKERS = profile["tickers"]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    profile = pick_profile()
    if profile is None:
        print("Cancelled.")
        return

    print(f"\n  Starting: {profile['name']}")
    print(f"  Tickers : {profile['tickers'] or 'all 15'}")
    print(f"  Hidden  : {profile['hidden_size']}")
    print(f"  Epochs  : {profile['max_epochs']} max")
    print(f"  Folds   : {profile['n_folds']}")
    print(f"  Encoder : {profile['encoder_length']} days\n")

    apply_profile(profile)

    from data.feature_engineering import build_features
    from training.train_supervised import train_supervised

    print("Building features...")
    features = build_features()

    if profile["tickers"]:
        features = features[features["ticker"].isin(profile["tickers"])]
        print(f"Filtered to {len(features)} rows for {profile['tickers']}")

    print(f"\nStarting training ({profile['n_folds']} fold(s))...\n")
    model = train_supervised(features, n_folds=profile["n_folds"])

    if model:
        print("\n  Training complete. Checkpoint saved to models/checkpoints/")
        print("  Next step: python training/evaluate.py\n")
    else:
        print("\n  Training produced no model — check errors above.\n")


if __name__ == "__main__":
    main()
