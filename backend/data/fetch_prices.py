import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import yfinance as yf
from loguru import logger
from datetime import datetime, timedelta
from config import TICKERS, MACRO_TICKERS, PRICE_HISTORY_YEARS, DATA_DIR


def _parquet_path(ticker: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{ticker}_prices.parquet")


def fetch_prices(ticker: str, force: bool = False) -> pd.DataFrame:
    path = _parquet_path(ticker)
    if os.path.exists(path) and not force:
        df = pd.read_parquet(path)
        # Incremental update: fetch only missing recent days
        last_date = df.index.max()
        today = pd.Timestamp.now().normalize()
        if last_date >= today - pd.Timedelta(days=2):
            logger.debug(f"{ticker}: cache up-to-date (last={last_date.date()})")
            return df
        start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(f"{ticker}: incremental fetch from {start}")
        new_data = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if not new_data.empty:
            new_data.index = pd.to_datetime(new_data.index)
            df = pd.concat([df, new_data[~new_data.index.isin(df.index)]])
            df.sort_index(inplace=True)
            df.to_parquet(path)
        return df

    start_date = (datetime.now() - timedelta(days=365 * PRICE_HISTORY_YEARS)).strftime("%Y-%m-%d")
    logger.info(f"{ticker}: full download from {start_date}")
    df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")
    # Flatten MultiIndex columns produced by yfinance 1.x
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.to_parquet(path)
    logger.info(f"{ticker}: saved {len(df)} rows to {path}")
    return df


def fetch_all_prices(force: bool = False) -> dict[str, pd.DataFrame]:
    prices = {}
    all_tickers = list(TICKERS) + list(MACRO_TICKERS)
    for ticker in all_tickers:
        try:
            prices[ticker] = fetch_prices(ticker, force=force)
        except Exception as e:
            logger.error(f"Failed to fetch prices for {ticker}: {e}")
    return prices


def fetch_latest_prices() -> dict[str, pd.DataFrame]:
    """Called daily by scheduler — incremental update only."""
    return fetch_all_prices(force=False)


if __name__ == "__main__":
    fetch_all_prices()
    print("All prices fetched successfully.")
