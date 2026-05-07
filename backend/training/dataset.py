import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from loguru import logger
from config import ENCODER_LENGTH, PREDICTION_HORIZON, TFT_BATCH_SIZE, TEMPORAL_WEIGHT_HALF_LIFE
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


def build_weighted_train_dataloader(dataset, batch_size: int | None = None, half_life: int | None = None):
    """
    Training DataLoader with exponential decay sample weights.
    Sequences ending on more recent dates are sampled more often.
    Validation DataLoaders should always use the plain to_dataloader(train=False).
    """
    half_life = half_life if half_life is not None else TEMPORAL_WEIGHT_HALF_LIFE
    batch_size = batch_size or TFT_BATCH_SIZE

    if half_life is None:
        return dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

    # dataset.index is a DataFrame where each row = one training sequence.
    # time_idx corresponds to the *start* of the encoder window; adding
    # max_encoder_length gives the approximate end (= prediction target date).
    idx_series = dataset.index["time_idx"]
    end_idx = idx_series + dataset.max_encoder_length  # recency proxy
    max_end = end_idx.max()

    decay = np.log(2) / half_life  # so weight halves every half_life days
    weights = np.exp(-decay * (max_end - end_idx).values)
    weights = torch.FloatTensor(weights)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    logger.info(
        f"Weighted sampler: half_life={half_life}d, "
        f"weight range [{weights.min():.3f}, {weights.max():.3f}]"
    )
    # Pass sampler; shuffle must be False when a sampler is provided
    return dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0, sampler=sampler)


def build_inference_dataloader(features: pd.DataFrame, batch_size: int = 1):
    """Build a single-sample dataloader for live inference on the latest window."""
    dataset = build_timeseries_dataset(features)
    return dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
