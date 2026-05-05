import numpy as np
from config import DRAWDOWN_PENALTY_THRESHOLD


def compute_reward(
    returns_history: list[float],
    current_drawdown: float,
    delta_position: float,
    is_terminal: bool = False,
) -> float:
    """
    r = sharpe_incremental - 2.0 * drawdown_penalty - 0.5 * transaction_cost
    terminal bonus = sharpe_ratio_episode * 10
    """
    sharpe_inc = _sharpe_incremental(returns_history)
    drawdown_penalty = max(0.0, current_drawdown - DRAWDOWN_PENALTY_THRESHOLD)
    transaction_cost = abs(delta_position) * 0.001

    reward = sharpe_inc - 2.0 * drawdown_penalty - 0.5 * transaction_cost

    if is_terminal and len(returns_history) > 1:
        ep_sharpe = _episode_sharpe(returns_history)
        reward += ep_sharpe * 10.0

    return float(reward)


def _sharpe_incremental(returns: list[float], window: int = 20) -> float:
    if len(returns) < 2:
        return 0.0
    window_returns = returns[-window:]
    mu = np.mean(window_returns)
    sigma = np.std(window_returns) + 1e-8  # epsilon avoids division by zero on first days
    return float(mu / sigma)


def _episode_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns) + 1e-8
    annualized = mu / sigma * np.sqrt(252)
    return float(annualized)
