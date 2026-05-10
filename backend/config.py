import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Trading universe
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "JNJ",
    "XOM", "UNH", "PG", "HD", "MA",
]
PRIMARY_TICKER = "AAPL"

# Macro indicators fetched alongside price data (market-wide context)
MACRO_TICKERS = ["^VIX", "^TNX", "DX-Y.NYB", "GC=F", "^GSPC"]

# Data parameters
PRICE_HISTORY_YEARS = 10
ENCODER_LENGTH = 60       # days of lookback for TFT
PREDICTION_HORIZON = 5    # days forward for label

# Labelling thresholds (relative to SP500)
SELL_THRESHOLD = -0.015   # -1.5%
BUY_THRESHOLD = 0.015     # +1.5%
LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
CLASS_WEIGHTS = {0: 3.0, 1: 1.0, 2: 3.0}  # SELL, HOLD, BUY

# TFT hyperparameters
TFT_HIDDEN_SIZE = 32          # kept small — dataset is ~38k rows, not millions
TFT_ATTENTION_HEAD_SIZE = 2   # reduced: fewer params, less overfitting risk
TFT_DROPOUT = 0.3             # raised from 0.1 — primary regularizer
TFT_LSTM_LAYERS = 1           # single layer sufficient; 2 layers overfit on small data
TFT_LEARNING_RATE = 1e-3
TFT_WEIGHT_DECAY = 1e-3       # L2 penalty on all weights via AdamW
TFT_MAX_EPOCHS = 50
TFT_BATCH_SIZE = 64
TFT_EARLY_STOPPING_PATIENCE = 3   # stop fast — patience 5 let models memorize noise
TFT_FEATURE_NOISE_STD = 0.01      # Gaussian noise injected during training batches

# PPO hyperparameters
PPO_LR = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.01
PPO_TOTAL_TIMESTEPS = 1_000_000

# Risk rules
MAX_PORTFOLIO_DRAWDOWN = 0.20   # 20% stop-loss
MAX_POSITION_PCT = 0.20         # 20% per ticker
CONFIDENCE_THRESHOLD = 0.55     # below → no trade
MIN_AVG_VOLUME = 1_000_000      # shares/day
EARNINGS_BLACKOUT_HOURS = 24    # hours before earnings
DRAWDOWN_PENALTY_THRESHOLD = 0.15

# Score / position formula constants
SCORE_NEUTRAL = 5.0

# Base multi-horizon weights for calm-market conditions (regime logic in strategy_scorer.py adjusts dynamically)
# crisis: 5d=0.45 / 21d=0.55 (trend dominates)   calm: 5d=0.65 / 21d=0.35 (tactical dominates)
HORIZON_WEIGHTS = {"weekly": 0.65, "monthly": 0.35}

# Temporal sample weighting — exponential decay half-life in days.
# Recent data gets weight ~1.0; data from HALF_LIFE days ago gets weight ~0.5.
# Set to None to disable weighting entirely.
TEMPORAL_WEIGHT_HALF_LIFE = 252   # ~1 trading year — prioritise recent regimes over distant history

# Crisis periods for training oversampling and evaluation segmentation
CRISIS_PERIODS = [
    ("2008-09-01", "2009-03-31"),   # Global Financial Crisis
    ("2011-08-01", "2011-10-31"),   # EU Debt Crisis
    ("2018-10-01", "2018-12-31"),   # Q4 2018 selloff
    ("2020-02-01", "2020-04-30"),   # COVID crash
    ("2022-01-01", "2022-10-15"),   # Rate-hike bear market
]
RECOVERY_PERIODS = [
    ("2009-04-01", "2009-12-31"),
    ("2012-01-01", "2012-06-30"),
    ("2019-01-01", "2019-06-30"),
    ("2020-05-01", "2020-12-31"),
    ("2022-10-16", "2023-06-30"),
]
CRISIS_SAMPLE_WEIGHT = 5.0   # multiplier applied to crisis-period rows during training

# Scheduler (NY time)
SCHEDULE_HOUR = 15
SCHEDULE_MINUTE = 30
SCHEDULE_TIMEZONE = "America/New_York"

# Paths
DATA_DIR = "data/cache"
MODEL_DIR = "models/checkpoints"
MLFLOW_TRACKING_URI = "mlruns"

# WebSocket push intervals (seconds)
WS_MARKET_HOURS_INTERVAL = 1
WS_OFF_HOURS_INTERVAL = 60
