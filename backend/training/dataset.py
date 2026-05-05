import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from loguru import logger
from config import ENCODER_LENGTH, PREDICTION_HORIZON
from data.feature_engineering import FEATURE_COLUMNS


def build_timeseries_dataset(features: pd.DataFrame, cutoff_date: str | None = None):
    """
    Build a pytorch-forecasting TimeSeriesDataSet for 3-class classification.
    cutoff_date: if set, only use data before this date (walk-forward safe).
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    df = features.copy()
    if cutoff_date:
        df = df[df.index <= cutoff_date]

    df = df.reset_index()
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days.astype(int)
    df["group"] = df["ticker"].astype(str) if "ticker" in df.columns else "default"

    # Integer labels 0/1/2 → NaNLabelEncoder tells pytorch-forecasting this is classification
    df["label"] = df["label"].astype(int).astype(str)  # NaNLabelEncoder needs strings

    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns and c != "label"]

    # Fill any NaNs in feature columns so the dataset doesn't reject rows
    df[feat_cols] = df[feat_cols].fillna(0.0)

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="label",
        group_ids=["group"],
        min_encoder_length=ENCODER_LENGTH // 2,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=PREDICTION_HORIZON,
        time_varying_unknown_reals=feat_cols,
        target_normalizer=NaNLabelEncoder(add_nan=False),
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )
    return dataset


def build_inference_dataloader(features: pd.DataFrame, batch_size: int = 1):
    """Build a single-sample dataloader for live inference on the latest window."""
    dataset = build_timeseries_dataset(features)
    return dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
