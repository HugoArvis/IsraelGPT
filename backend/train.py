"""
Training launcher — run this instead of calling train_supervised.py directly.
Presents a menu to choose between Full, Light, and Compare-All profiles.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Profile definitions ───────────────────────────────────────────────────────

PROFILES = {
    "1": {
        "name": "Full model (TFT only)",
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
        "mode": "tft",
    },
    "2": {
        "name": "Light test run (TFT only)",
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
        "mode": "tft",
    },
    "3": {
        "name": "Compare all models (TFT · N-HiTS · LightGBM)",
        "description": (
            "3 tickers · hidden=32 · 10 epochs max · 2 folds each\n"
            "  Trains all 3 models then prints a side-by-side accuracy table\n"
            "  CPU: ~60-120 min    |   RTX 3060: ~10-20 min"
        ),
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "hidden_size": 32,
        "batch_size": 128,
        "max_epochs": 10,
        "n_folds": 2,
        "encoder_length": 30,
        "mode": "compare",
    },
}

# ── Menu ─────────────────────────────────────────────────────────────────────

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


# ── Apply profile overrides ───────────────────────────────────────────────────

def apply_profile(profile: dict):
    import config as cfg

    cfg.TFT_HIDDEN_SIZE = profile["hidden_size"]
    cfg.TFT_BATCH_SIZE = profile["batch_size"]
    cfg.TFT_MAX_EPOCHS = profile["max_epochs"]
    cfg.ENCODER_LENGTH = profile["encoder_length"]

    if profile["tickers"] is not None:
        cfg.TICKERS = profile["tickers"]


# ── Single-model training (TFT or N-HiTS) ────────────────────────────────────

def run_tft(features, profile):
    from training.train_supervised import train_supervised
    print(f"\nTraining TFT ({profile['n_folds']} fold(s))...")
    model = train_supervised(features, n_folds=profile["n_folds"])
    return model


def run_nhits(features, profile):
    from training.train_nhits import train_nhits
    print(f"\nTraining N-HiTS ({profile['n_folds']} fold(s))...")
    model = train_nhits(features, n_folds=profile["n_folds"])
    return model


def run_lgbm(features):
    from models.lgbm_model import LGBMModel
    print("\nTraining LightGBM...")
    model = LGBMModel.train(features)
    model.save()
    return model


# ── Evaluate accuracy on held-out last fold ──────────────────────────────────

def _eval_seq_model(model_obj, features, profile):
    """Walk-forward accuracy for TFT/N-HiTS using the last validation fold."""
    import torch
    from training.dataset import build_dataset

    n = len(features)
    fold_size = n // (profile["n_folds"] + 1)
    split = n - fold_size  # last fold as held-out val
    train_df = features.iloc[:split]
    val_df = features.iloc[split:]

    try:
        _, val_ds = build_dataset(train_df, val_df)
        from torch.utils.data import DataLoader
        val_dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        preds_list, labels_list = [], []
        inner_model = model_obj._model
        with torch.no_grad():
            for x, y in val_dl:
                out = inner_model(x)
                batch_preds = out["prediction"][:, 0, :].argmax(dim=-1).cpu().numpy()
                target = y[0] if isinstance(y, (list, tuple)) else y
                batch_labels = target[:, 0].long().cpu().numpy()
                preds_list.extend(batch_preds.tolist())
                labels_list.extend(batch_labels.tolist())
        correct = sum(p == l for p, l in zip(preds_list, labels_list))
        return correct / len(preds_list) if preds_list else 0.0
    except Exception as e:
        from loguru import logger
        logger.warning(f"Seq-model eval failed: {e}")
        return 0.0


def _eval_lgbm(model_obj, features, profile):
    """Accuracy on the temporally last fold."""
    n = len(features)
    fold_size = n // (profile["n_folds"] + 1)
    val_df = features.iloc[n - fold_size:]
    result = model_obj.evaluate(val_df)
    return result.get("accuracy", 0.0)


# ── Comparison table ─────────────────────────────────────────────────────────

def print_comparison(results: dict):
    print("\n" + "=" * 50)
    print("  Model Comparison — Out-of-sample Accuracy")
    print("=" * 50)
    best_name = max(results, key=lambda k: results[k])
    for name, acc in results.items():
        marker = "  ← BEST" if name == best_name else ""
        print(f"  {name:<12}  {acc*100:6.2f}%{marker}")
    print("=" * 50)
    print(f"\n  Recommendation: use {best_name}")
    print("  (retrain full profile with that model for production)\n")


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

    print("Building features...")
    features = build_features()

    if profile["tickers"]:
        features = features[features["ticker"].isin(profile["tickers"])]
        print(f"Filtered to {len(features)} rows for {profile['tickers']}")

    mode = profile["mode"]

    # ── TFT only ─────────────────────────────────────────────────────────────
    if mode == "tft":
        model = run_tft(features, profile)
        if model:
            print("\n  Training complete. Checkpoint saved to models/checkpoints/")
            print("  Next step: python training/evaluate.py\n")
        else:
            print("\n  Training produced no model — check errors above.\n")

    # ── Compare all ──────────────────────────────────────────────────────────
    elif mode == "compare":
        results = {}

        # TFT
        tft = run_tft(features, profile)
        if tft:
            acc = _eval_seq_model(tft, features, profile)
            results["TFT"] = acc
            print(f"  TFT accuracy: {acc*100:.2f}%")

        # N-HiTS
        try:
            nhits = run_nhits(features, profile)
            if nhits:
                acc = _eval_seq_model(nhits, features, profile)
                results["N-HiTS"] = acc
                print(f"  N-HiTS accuracy: {acc*100:.2f}%")
        except Exception as e:
            print(f"  N-HiTS skipped: {e}")

        # LightGBM
        try:
            lgbm = run_lgbm(features)
            acc = _eval_lgbm(lgbm, features, profile)
            results["LightGBM"] = acc
            print(f"  LightGBM accuracy: {acc*100:.2f}%")
        except Exception as e:
            print(f"  LightGBM skipped: {e}")

        if results:
            print_comparison(results)
        else:
            print("\n  No models trained successfully.\n")


if __name__ == "__main__":
    main()
