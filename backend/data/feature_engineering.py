import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from loguru import logger
from config import TICKERS, DATA_DIR, ENCODER_LENGTH, PREDICTION_HORIZON, SELL_THRESHOLD, BUY_THRESHOLD, MACRO_TICKERS

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


def _compute_labels_for_horizon(df: pd.DataFrame, sp500_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Return a label array (0=SELL, 1=HOLD, 2=BUY) for a given forward horizon."""
    ticker_fwd = df["close"].shift(-horizon) / df["close"] - 1
    if sp500_df is not None and not sp500_df.empty:
        sp500_aligned = sp500_df["close"].reindex(df.index).ffill()
        sp500_fwd = sp500_aligned.shift(-horizon) / sp500_aligned - 1
        relative_return = ticker_fwd - sp500_fwd
    else:
        relative_return = ticker_fwd

    rolling_vol = relative_return.rolling(20).std().fillna(abs(BUY_THRESHOLD))
    buy_thresh  = rolling_vol.clip(lower=BUY_THRESHOLD)
    sell_thresh = rolling_vol.clip(lower=abs(SELL_THRESHOLD))

    return np.where(
        relative_return > buy_thresh, 2,
        np.where(relative_return < -sell_thresh, 0, 1)
    )


def _compute_labels(df: pd.DataFrame, sp500_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each day: SELL=0, HOLD=1, BUY=2
    Dynamic thresholds: 1× rolling 20-day volatility of relative returns,
    floored at the base thresholds from config.
    This prevents tiny thresholds in low-vol regimes and adapts to high-vol periods.
    Data leakage safe: labels derived from future prices, never used as input features.
    """
    ticker_fwd = df["close"].shift(-PREDICTION_HORIZON) / df["close"] - 1
    if sp500_df is not None and not sp500_df.empty:
        sp500_aligned = sp500_df["close"].reindex(df.index).ffill()
        sp500_fwd = sp500_aligned.shift(-PREDICTION_HORIZON) / sp500_aligned - 1
        relative_return = ticker_fwd - sp500_fwd
    else:
        relative_return = ticker_fwd

    # Dynamic thresholds: rolling volatility, floored at base config values
    rolling_vol = relative_return.rolling(20).std().fillna(abs(BUY_THRESHOLD))
    buy_thresh = rolling_vol.clip(lower=BUY_THRESHOLD)
    sell_thresh = rolling_vol.clip(lower=abs(SELL_THRESHOLD))

    labels = np.where(
        relative_return > buy_thresh, 2,
        np.where(relative_return < -sell_thresh, 0, 1)
    )
    df["label"] = labels
    df["relative_return"] = relative_return
    return df


def build_features(ticker: str | None = None) -> pd.DataFrame:
    tickers = [ticker] if ticker else TICKERS
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

            df["day_of_week"] = df.index.dayofweek
            df["quarter_end"] = ((df.index.month % 3 == 0) & (df.index.day >= 25)).astype(int)

            df = _compute_labels(df, sp500_df)
            df["label_1d"]  = _compute_labels_for_horizon(df, sp500_df, 1)
            df["label_21d"] = _compute_labels_for_horizon(df, sp500_df, 21)
            df["ticker"] = tkr
            df.dropna(subset=["rsi_14", "macd", "close"], inplace=True)
            all_frames.append(df)
        except Exception as e:
            logger.error(f"Feature engineering failed for {tkr}: {e}")

    if not all_frames:
        raise RuntimeError("No feature data could be built")
    combined = pd.concat(all_frames).sort_index()
    logger.info(f"Features built: {len(combined)} rows across {len(all_frames)} tickers")
    return combined


FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower",
    "atr_14", "sma_20", "sma_50", "ema_12", "ema_26", "volume_ratio",
    "compound", "sentiment_pos", "sentiment_neg", "n_articles",
    "vix", "tnx", "dxy", "gold", "sp500",
    "day_of_week", "quarter_end",
]


if __name__ == "__main__":
    df = build_features()
    print(df.tail())
    print(f"Label distribution:\n{df['label'].value_counts()}")
