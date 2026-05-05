import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
from config import ENCODER_LENGTH
from rl.reward_functions import compute_reward
from data.feature_engineering import FEATURE_COLUMNS


class TradingEnv(gym.Env):
    """
    Single-ticker trading environment.
    observation_space: normalized 60-day feature window + current_position + drawdown
    action_space: Box(0.0, 10.0) — conviction score
    """

    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame, ticker: str = "AAPL", episode_length: int = 252):
        super().__init__()
        self.ticker_data = features[features["ticker"] == ticker].copy() if "ticker" in features.columns else features.copy()
        self.episode_length = episode_length
        self.feature_cols = [c for c in FEATURE_COLUMNS if c in self.ticker_data.columns]

        obs_dim = len(self.feature_cols) * ENCODER_LENGTH + 2  # +position +drawdown
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32)

        self._episode_start = 0
        self._step_idx = 0
        self._position_pct = 0.0
        self._portfolio_value = 100_000.0
        self._start_value = 100_000.0
        self._peak_value = 100_000.0
        self._returns: list[float] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(0, len(self.ticker_data) - self.episode_length - ENCODER_LENGTH)
        self._episode_start = self.np_random.integers(0, max_start + 1)
        self._step_idx = 0
        self._position_pct = 0.0
        self._portfolio_value = 100_000.0
        self._start_value = 100_000.0
        self._peak_value = 100_000.0
        self._returns = []
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        start = self._episode_start + self._step_idx
        end = start + ENCODER_LENGTH
        window = self.ticker_data.iloc[start:end][self.feature_cols].values.astype(np.float32)

        # Pad if window is short (beginning of data)
        if len(window) < ENCODER_LENGTH:
            pad = np.zeros((ENCODER_LENGTH - len(window), window.shape[1]), dtype=np.float32)
            window = np.vstack([pad, window])

        # Normalize per feature across the window
        mean = window.mean(axis=0)
        std = window.std(axis=0) + 1e-8
        window = (window - mean) / std

        drawdown = (self._peak_value - self._portfolio_value) / self._peak_value
        obs = np.concatenate([window.flatten(), [self._position_pct / 100.0, drawdown]])
        return obs.astype(np.float32)

    def step(self, action: np.ndarray):
        score = float(np.clip(action[0], 0.0, 10.0))
        new_position_pct = (score - 5.0) / 5.0 * 100.0
        delta_position = (new_position_pct - self._position_pct) / 100.0

        # Advance one trading day
        data_idx = self._episode_start + self._step_idx + ENCODER_LENGTH
        if data_idx >= len(self.ticker_data):
            terminated = True
            reward = compute_reward(self._returns, self._current_drawdown(), delta_position, is_terminal=True)
            return self._get_obs(), reward, terminated, False, {}

        row = self.ticker_data.iloc[data_idx]
        daily_return = float(row.get("close", 0) / self.ticker_data.iloc[data_idx - 1].get("close", 1) - 1)
        position_return = daily_return * (self._position_pct / 100.0)
        self._portfolio_value *= (1 + position_return)
        self._peak_value = max(self._peak_value, self._portfolio_value)
        self._returns.append(position_return)

        self._position_pct = new_position_pct
        self._step_idx += 1
        terminated = self._step_idx >= self.episode_length

        reward = compute_reward(
            self._returns,
            self._current_drawdown(),
            delta_position,
            is_terminal=terminated,
        )
        return self._get_obs(), reward, terminated, False, {
            "portfolio_value": self._portfolio_value,
            "score": score,
            "drawdown": self._current_drawdown(),
        }

    def _current_drawdown(self) -> float:
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - self._portfolio_value) / self._peak_value
