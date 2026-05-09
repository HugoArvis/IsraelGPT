import torch
import lightning as pl


class FeatureNoiseCallback(pl.Callback):
    """
    Injects small Gaussian noise into encoder features on every training batch.

    This is a data-augmentation regularizer: the model never sees the exact same
    feature vector twice, which prevents memorisation of training sequences.
    Noise is scaled relative to each feature's per-batch std so it stays small
    regardless of the feature's absolute magnitude.

    Disabled automatically during validation (on_train_batch_start only fires
    during training steps, not val steps).
    """

    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.noise_std <= 0:
            return
        x, _ = batch
        if not isinstance(x, dict) or "encoder_cont" in x is False:
            return
        enc = x["encoder_cont"]                          # (batch, time, features)
        # Scale noise by per-feature std so it's proportional to each feature's range
        feat_std = enc.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        noise = torch.randn_like(enc) * feat_std * self.noise_std
        x["encoder_cont"] = enc + noise
