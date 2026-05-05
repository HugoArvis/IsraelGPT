import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
from loguru import logger


def build_callbacks(make_env_fn, features_df, ticker: str):
    eval_env = DummyVecEnv([make_env_fn])

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=10.0, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/checkpoints/",
        log_path="models/eval_logs/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=stop_callback,
        verbose=1,
    )
    return [eval_callback]
