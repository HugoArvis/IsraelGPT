import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from loguru import logger
from config import TICKERS, EQUITY_TICKERS, DATA_DIR, ENCODER_LENGTH, PREDICTION_HORIZON, SELL_THRESHOLD, BUY_THRESHOLD, MACRO_TICKERS

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False
    logger.warning("ta library not installed — technical indicators will be zeros")


def _load_prices(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ticker}_prices.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No price data for {ticker} — run fetch_prices.py first")
    df = pd.read_parquet(path)
    # yfinance 1.x returns MultiIndex columns like ('Close', 'AAPL') — flatten to 'close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    return df


def _load_macro() -> pd.DataFrame:
    """
    Load market-wide macro indicators (VIX, 10Y yield, DXY, Gold, SP500).
    Returns a daily DataFrame aligned to trading days.
    Falls back gracefully if files are missing — run fetch_prices.py first.
    """
    # ticker → column name mapping
    macro_map = {
        "^VIX": "vix",
        "^TNX": "tnx",
        "DX-Y.NYB": "dxy",
        "GC=F": "gold",
        "^GSPC": "sp500",
    }
    frames = {}
    for ticker, col in macro_map.items():
        path = os.path.join(DATA_DIR, f"{ticker}_prices.parquet")
        if not os.path.exists(path):
            logger.debug(f"Macro file missing for {ticker} — run fetch_prices.py")
            continue
        try:
            df = pd.read_parquet(path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            frames[col] = df["close"].rename(col)
        except Exception as e:
            logger.warning(f"Could not load macro {ticker}: {e}")

    if not frames:
        return pd.DataFrame()
    macro = pd.concat(frames.values(), axis=1)
    macro.index = pd.to_datetime(macro.index)
    return macro


def _load_sentiment(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ticker}_sentiment.parquet")
    if not os.path.exists(path):
        logger.warning(f"No sentiment data for {ticker} — using zeros")
        return pd.DataFrame()
    return pd.read_parquet(path)


def _compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market-state features that tell the model WHAT REGIME it is in.
    All use only data available at time t (no look-ahead).
    """
    # VIX stress level: z-score and ordinal bucket (0=calm,1=normal,2=elevated,3=panic)
    if "vix" in df.columns and df["vix"].notna().any():
        vix = df["vix"]
        vix_mean = vix.rolling(252, min_periods=63).mean()
        vix_std  = vix.rolling(252, min_periods=63).std().replace(0, np.nan)
        df["vix_z"]      = ((vix - vix_mean) / vix_std).fillna(0.0)
        df["vix_regime"] = pd.cut(
            vix, bins=[0, 15, 25, 35, 9999], labels=[0, 1, 2, 3]
        ).astype(float).fillna(1.0)
    else:
        df["vix_z"]      = 0.0
        df["vix_regime"] = 1.0

    # SP500 position relative to 200-day MA and depth of drawdown from 252-day high
    if "sp500" in df.columns and df["sp500"].notna().any():
        sp = df["sp500"]
        sp_200d = sp.rolling(200, min_periods=50).mean()
        df["sp500_200d_ratio"] = (sp / sp_200d.replace(0, np.nan) - 1).fillna(0.0)

        sp_high = sp.rolling(252, min_periods=63).max()
        df["market_drawdown"] = ((sp - sp_high) / sp_high.replace(0, np.nan)).fillna(0.0)

        # 3-month SP500 momentum — positive = uptrend, negative = downtrend
        df["sp500_momentum"] = sp.pct_change(63).fillna(0.0)
    else:
        df["sp500_200d_ratio"] = 0.0
        df["market_drawdown"]  = 0.0
        df["sp500_momentum"]   = 0.0

    # Stock's own drawdown from its 252-day high (how deep in the hole is this ticker)
    stock_high = df["close"].rolling(252, min_periods=63).max()
    df["stock_drawdown"] = ((df["close"] - stock_high) / stock_high.replace(0, np.nan)).fillna(0.0)

    return df


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if not _TA_AVAILABLE:
        df["rsi_14"] = 50.0
        df["macd"] = 0.0
        df["macd_signal"] = 0.0
        df["bb_upper"] = df["close"]
        df["bb_lower"] = df["close"]
        df["atr_14"] = 0.0
        df["sma_20"] = df["close"]
        df["sma_50"] = df["close"]
        df["ema_12"] = df["close"]
        df["ema_26"] = df["close"]
        df["volume_ratio"] = 1.0
        return df

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["rsi_14"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    df["sma_20"] = ta.trend.sma_indicator(close, window=20)
    df["sma_50"] = ta.trend.sma_indicator(close, window=50)
    df["ema_12"] = ta.trend.ema_indicator(close, window=12)
    df["ema_26"] = ta.trend.ema_indicator(close, window=26)
    vol_ma = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_ma.replace(0, np.nan)
    return df


def _atr_thresholds(relative_return: pd.Series, atr: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Compute buy/sell thresholds that target ~25/50/25 SELL/HOLD/BUY distribution.
    Uses rolling 252-day percentiles of relative_return, scaled by ATR regime ratio
    so thresholds widen in high-volatility periods and compress in low-volatility ones.
    """
    window = 252
    min_p  = 63

    # Rolling 25th and 75th percentiles of relative_return (target the 25/50/25 split)
    buy_pct  = relative_return.rolling(window, min_periods=min_p).quantile(0.75)
    sell_pct = relative_return.rolling(window, min_periods=min_p).quantile(0.25)

    # ATR regime ratio — scale thresholds with current vs average volatility
    atr_aligned = atr.reindex(relative_return.index).ffill()
    atr_mean    = atr_aligned.rolling(window, min_periods=min_p).mean()
    atr_ratio   = (atr_aligned / atr_mean.replace(0, np.nan)).clip(0.5, 2.0).fillna(1.0)

    buy_thresh  = (buy_pct  * atr_ratio).fillna(abs(BUY_THRESHOLD))
    sell_thresh = (sell_pct * atr_ratio).fillna(SELL_THRESHOLD)  # sell_pct is negative

    # Guard: buy_thresh must be positive, sell_thresh must be negative
    buy_thresh  = buy_thresh.clip(lower=abs(BUY_THRESHOLD))
    sell_thresh = sell_thresh.clip(upper=-abs(SELL_THRESHOLD))

    return buy_thresh, sell_thresh


def _compute_labels_for_horizon(df: pd.DataFrame, sp500_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Return a label array (0=SELL, 1=HOLD, 2=BUY) for a given forward horizon."""
    ticker_fwd = df["close"].shift(-horizon) / df["close"] - 1
    if sp500_df is not None and not sp500_df.empty:
        sp500_aligned = sp500_df["close"].reindex(df.index).ffill()
        sp500_fwd = sp500_aligned.shift(-horizon) / sp500_aligned - 1
        relative_return = ticker_fwd - sp500_fwd
    else:
        relative_return = ticker_fwd

    buy_thresh, sell_thresh = _atr_thresholds(relative_return, df["atr_14"])

    return np.where(
        relative_return > buy_thresh, 2,
        np.where(relative_return < sell_thresh, 0, 1)
    )


def _compute_labels(df: pd.DataFrame, sp500_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each day: SELL=0, HOLD=1, BUY=2
    ATR-based dynamic thresholds targeting ~25/50/25 class balance.
    Data leakage safe: labels derived from future prices, never used as input features.
    """
    ticker_fwd = df["close"].shift(-PREDICTION_HORIZON) / df["close"] - 1
    if sp500_df is not None and not sp500_df.empty:
        sp500_aligned = sp500_df["close"].reindex(df.index).ffill()
        sp500_fwd = sp500_aligned.shift(-PREDICTION_HORIZON) / sp500_aligned - 1
        relative_return = ticker_fwd - sp500_fwd
    else:
        relative_return = ticker_fwd

    buy_thresh, sell_thresh = _atr_thresholds(relative_return, df["atr_14"])

    labels = np.where(
        relative_return > buy_thresh, 2,
        np.where(relative_return < sell_thresh, 0, 1)
    )
    df["label"] = labels
    df["relative_return"] = relative_return
    return df


def build_features(ticker: str | None = None) -> pd.DataFrame:
    # EQUITY_TICKERS: stocks only — ETFs excluded from label generation
    # TICKERS: everything fetched (equities + ETFs used for cross-sectional ranks)
    tickers = [ticker] if ticker else EQUITY_TICKERS
    all_frames = []

    # Load SP500 for relative return calculation
    try:
        sp500_df = _load_prices("^GSPC") if "^GSPC" not in tickers else None
    except Exception:
        sp500_df = None

    # Load macro features once — same for all tickers
    macro = _load_macro()

    for tkr in tickers:
        try:
            df = _load_prices(tkr)
            df = _compute_indicators(df)

            sentiment = _load_sentiment(tkr)
            if not sentiment.empty:
                df = df.join(
                    sentiment[["compound", "sentiment_pos", "sentiment_neg", "n_articles"]],
                    how="left",
                )
            else:
                df["compound"] = 0.0
                df["sentiment_pos"] = 0.333
                df["sentiment_neg"] = 0.333
                df["n_articles"] = 0

            # Join macro features (forward-fill to handle non-overlapping market days)
            if not macro.empty:
                df = df.join(macro, how="left")
                df[macro.columns] = df[macro.columns].ffill()
            else:
                for col in ["vix", "tnx", "dxy", "gold", "sp500"]:
                    df[col] = 0.0

            df = _compute_regime_features(df)

            df["day_of_week"] = df.index.dayofweek
            df["quarter_end"] = ((df.index.month % 3 == 0) & (df.index.day >= 25)).astype(int)

            # Lag returns — explicit temporal context for tree-based models
            df["return_1d"]  = df["close"].pct_change(1)
            df["return_5d"]  = df["close"].pct_change(5)
            df["return_10d"] = df["close"].pct_change(10)
            df["return_21d"] = df["close"].pct_change(21)

            # Extended momentum factors (Jegadeesh-Titman / Fama-French)
            # 3/6/12-month returns are among the strongest documented return predictors
            df["return_63d"]  = df["close"].pct_change(63)
            df["return_126d"] = df["close"].pct_change(126)
            df["return_252d"] = df["close"].pct_change(252)

            # Price-relative features — scale-invariant, better for trees than raw prices
            df["close_sma20_ratio"] = df["close"] / df["sma_20"].replace(0, np.nan) - 1
            df["close_sma50_ratio"] = df["close"] / df["sma_50"].replace(0, np.nan) - 1
            bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
            df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range

            df = _compute_labels(df, sp500_df)
            df["label_1d"]  = _compute_labels_for_horizon(df, sp500_df, 1)
            df["label_21d"] = _compute_labels_for_horizon(df, sp500_df, 21)

            # Regression targets — raw relative alpha for each horizon
            # relative_return (5d) is already added by _compute_labels()
            if sp500_df is not None and not sp500_df.empty:
                sp500_a = sp500_df["close"].reindex(df.index).ffill()
                df["relative_return_21d"] = (
                    (df["close"].shift(-21) / df["close"] - 1)
                    - (sp500_a.shift(-21) / sp500_a - 1)
                )
            else:
                df["relative_return_21d"] = df["close"].shift(-21) / df["close"] - 1
            df["ticker"] = tkr
            df.dropna(subset=["rsi_14", "macd", "close"], inplace=True)
            all_frames.append(df)
        except Exception as e:
            logger.error(f"Feature engineering failed for {tkr}: {e}")

    if not all_frames:
        raise RuntimeError("No feature data could be built")
    combined = pd.concat(all_frames).sort_index()

    # Cross-sectional return rank — where does this ticker rank among all tickers on the same day?
    # Percentile 1.0 = top performer that day, 0.0 = worst. Strong empirical predictor (momentum factor).
    for col, out in [
        ("return_1d",  "cs_rank_1d"),
        ("return_5d",  "cs_rank_5d"),
        ("return_21d", "cs_rank_21d"),
        ("return_63d", "cs_rank_63d"),
        ("return_252d","cs_rank_252d"),
    ]:
        combined[out] = combined.groupby(level=0)[col].rank(pct=True).fillna(0.5)

    # Encode ticker identity so LightGBM can learn ticker-specific patterns
    ticker_map = {t: i for i, t in enumerate(sorted(combined["ticker"].unique()))}
    combined["ticker_id"] = combined["ticker"].map(ticker_map).astype(int)

    logger.info(f"Features built: {len(combined)} rows across {len(all_frames)} tickers")
    return combined


FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower",
    "atr_14", "sma_20", "sma_50", "ema_12", "ema_26", "volume_ratio",
    "compound", "sentiment_pos", "sentiment_neg", "n_articles",
    "vix", "tnx", "dxy", "gold", "sp500",
    "day_of_week", "quarter_end",
    "return_1d", "return_5d", "return_10d", "return_21d",
    "return_63d", "return_126d", "return_252d",
    "close_sma20_ratio", "close_sma50_ratio", "bb_position",
    # Regime features
    "vix_z", "vix_regime", "sp500_200d_ratio", "market_drawdown", "stock_drawdown", "sp500_momentum",
    # Cross-sectional momentum rank (percentile vs all tickers on the same day)
    "cs_rank_1d", "cs_rank_5d", "cs_rank_21d", "cs_rank_63d", "cs_rank_252d",
    "ticker_id",
]

CATEGORICAL_FEATURES = ["ticker_id"]


if __name__ == "__main__":
    df = build_features()
    print(df.tail())
    print(f"Label distribution:\n{df['label'].value_counts()}")
