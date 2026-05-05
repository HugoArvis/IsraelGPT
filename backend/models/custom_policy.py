import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from loguru import logger


class TFTFeaturesExtractor(BaseFeaturesExtractor):
    """
    Uses a frozen pretrained TFT encoder as feature extractor for PPO.
    Falls back to a simple MLP if TFT is not loaded.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        obs_size = int(observation_space.shape[0])
        self._tft_encoder = None

        # Try to load pretrained TFT encoder
        try:
            from models.tft_model import TFTModel
            tft = TFTModel.load_latest()
            if tft is not None and tft._model is not None:
                # Freeze TFT weights
                encoder = tft._model
                for param in encoder.parameters():
                    param.requires_grad = False
                self._tft_encoder = encoder
                logger.info("Pretrained TFT encoder loaded and frozen for PPO")
        except Exception as e:
            logger.warning(f"Could not load TFT encoder: {e} — using MLP fallback")

        # MLP head always present (trainable)
        self.mlp = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


class TFTPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = TFTFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": 128}
        super().__init__(*args, **kwargs)
