import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ── Trading universe ───────────────────────────────────────────────────────────
# Expanded from 15 to 78 tickers: diverse sectors + market caps + sector ETFs.
# Rationale: more cross-sectional diversity → less survivorship bias, better
# generalisation across market regimes. ETFs act as clean regime anchors.
TICKERS = [
    # ── Large-cap US equities (original universe) ──────────────────
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "JNJ",
    "XOM", "UNH", "PG", "HD", "MA",
    # ── Technology ─────────────────────────────────────────────────
    "AMD", "AVGO", "QCOM", "CRM", "ADBE",
    "ORCL", "CSCO", "TXN", "NOW", "INTC",
    # ── Healthcare ─────────────────────────────────────────────────
    "LLY", "ABBV", "MRK", "PFE", "ABT", "CVS", "MDT",
    # ── Financials ─────────────────────────────────────────────────
    "BAC", "GS", "WFC", "BRK-B", "AXP", "MS", "BLK",
    # ── Consumer Discretionary ─────────────────────────────────────
    "MCD", "NKE", "SBUX", "BKNG", "TJX",
    # ── Consumer Staples ───────────────────────────────────────────
    "KO", "PEP", "WMT", "COST", "CL",
    # ── Energy ─────────────────────────────────────────────────────
    "CVX", "COP", "EOG",
    # ── Industrials ────────────────────────────────────────────────
    "BA", "HON", "CAT", "RTX", "UPS",
    # ── Communications ─────────────────────────────────────────────
    "DIS", "NFLX", "CMCSA", "VZ",
    # ── Materials / Utilities ──────────────────────────────────────
    "LIN", "NEE",
    # ── Sector & factor ETFs ───────────────────────────────────────
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
    "GLD", "TLT", "QQQ", "IWM", "HYG",
]

# Tickers for which we actually generate BUY/SELL/HOLD signals — equities only.
# ETFs are in TICKERS (fetched, used as cross-sectional features) but excluded here
# because their low volatility skews labels toward HOLD and dilutes the training signal.
EQUITY_TICKERS = [t for t in TICKERS if t not in {
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
    "GLD", "TLT", "QQQ", "IWM", "HYG",
}]

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
CLASS_WEIGHTS = {0: 2.0, 1: 1.0, 2: 2.0}  # SELL, HOLD, BUY

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

# TFT hyperparameters
TFT_HIDDEN_SIZE = 32
TFT_ATTENTION_HEAD_SIZE = 2
TFT_DROPOUT = 0.3
TFT_LSTM_LAYERS = 1
TFT_LEARNING_RATE = 1e-3
TFT_WEIGHT_DECAY = 1e-3
TFT_MAX_EPOCHS = 50
TFT_BATCH_SIZE = 64
TFT_EARLY_STOPPING_PATIENCE = 3
TFT_FEATURE_NOISE_STD = 0.01

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

# Multi-horizon weights (regime-adaptive; strategy_scorer.py adjusts dynamically)
HORIZON_WEIGHTS = {"weekly": 0.10, "monthly": 0.90}  # 5d near-random on recent data → 21d dominates

# Regression scoring: predicted alpha at which position reaches ±100%
# e.g. 0.03 → a +3% predicted 21d alpha vs SP500 = full long position
MAX_PREDICTED_RETURN = 0.03

# Temporal sample weighting — exponential decay half-life in days
# Recent data weighted ~1.0; data HALF_LIFE days ago weighted ~0.5
TEMPORAL_WEIGHT_HALF_LIFE = 252   # ~1 trading year

# Rolling training window — only train on the last N years of data.
# Older data describes market regimes that no longer apply and hurts out-of-sample
# performance on recent data. 3 years captures 1+ full market cycle.
ROLLING_WINDOW_YEARS = 5

# Scheduler (NY time) — daily trading decision
SCHEDULE_HOUR = 15
SCHEDULE_MINUTE = 30
SCHEDULE_TIMEZONE = "America/New_York"

# Weekly retraining schedule (Sunday evening — before Monday market open)
RETRAIN_DAY_OF_WEEK = "sun"
RETRAIN_HOUR = 18
RETRAIN_MINUTE = 0

# Paths
DATA_DIR = "data/cache"
MODEL_DIR = "models/checkpoints"
MLFLOW_TRACKING_URI = "mlruns"

# WebSocket push intervals (seconds)
WS_MARKET_HOURS_INTERVAL = 1
WS_OFF_HOURS_INTERVAL = 60
