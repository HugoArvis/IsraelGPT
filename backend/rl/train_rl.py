import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlflow
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from config import (
    PPO_LR, PPO_N_STEPS, PPO_BATCH_SIZE, PPO_N_EPOCHS,
    PPO_GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_RANGE, PPO_ENT_COEF,
    PPO_TOTAL_TIMESTEPS, MODEL_DIR, MLFLOW_TRACKING_URI,
)
from rl.trading_env import TradingEnv
from rl.callbacks import build_callbacks
from models.custom_policy import TFTPPOPolicy


def train_rl(features_df, ticker: str = "AAPL"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def make_env():
        return TradingEnv(features_df, ticker=ticker)

    env = DummyVecEnv([make_env])

    model = PPO(
        policy=TFTPPOPolicy,
        env=env,
        learning_rate=PPO_LR,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        verbose=1,
    )

    callbacks = build_callbacks(make_env, features_df, ticker)

    with mlflow.start_run(run_name=f"ppo_{ticker}"):
        mlflow.log_params({
            "lr": PPO_LR,
            "n_steps": PPO_N_STEPS,
            "batch_size": PPO_BATCH_SIZE,
            "total_timesteps": PPO_TOTAL_TIMESTEPS,
            "ticker": ticker,
        })
        model.learn(total_timesteps=PPO_TOTAL_TIMESTEPS, callback=callbacks)
        save_path = os.path.join(MODEL_DIR, f"ppo_{ticker}")
        model.save(save_path)
        mlflow.log_artifact(save_path + ".zip")
        logger.info(f"PPO model saved to {save_path}.zip")

    return model


if __name__ == "__main__":
    from data.feature_engineering import build_features
    features = build_features()
    train_rl(features)
