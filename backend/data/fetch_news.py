import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import finnhub
import pandas as pd
from datetime import datetime, timedelta, timezone
from loguru import logger
from config import FINNHUB_API_KEY, TICKERS, DATA_DIR

try:
    from models.finbert_embedder import FinBERTEmbedder
    _FINBERT_AVAILABLE = True
except Exception:
    _FINBERT_AVAILABLE = False


def _news_path(ticker: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{ticker}_sentiment.parquet")


def _get_finnhub_client() -> finnhub.Client | None:
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "your_key_here":
        logger.warning("FINNHUB_API_KEY not set — skipping news fetch")
        return None
    return finnhub.Client(api_key=FINNHUB_API_KEY)


def fetch_news_for_ticker(
    ticker: str,
    client: finnhub.Client,
    embedder,
    days_back: int = 7,
) -> pd.DataFrame:
    """
    Fetch recent news, run FinBERT, return daily aggregated sentiment.
    Cutoff: only articles before 09:30 NY of each day (data leakage rule).
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)
    from_str = start_dt.strftime("%Y-%m-%d")
    to_str = end_dt.strftime("%Y-%m-%d")

    try:
        articles = client.company_news(ticker, _from=from_str, to=to_str)
    except Exception as e:
        logger.error(f"Finnhub news fetch failed for {ticker}: {e}")
        return pd.DataFrame()

    records = []
    for art in articles:
        pub_ts = art.get("datetime", 0)
        if not pub_ts:
            continue
        pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
        headline = art.get("headline", "")
        summary = art.get("summary", "")
        text = f"{headline}. {summary}".strip()
        if not text or text == ".":
            continue
        records.append({"datetime": pub_dt, "text": text})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("datetime")

    if embedder is not None:
        texts = df["text"].tolist()
        scores = embedder.score_batch(texts)  # list of dicts {positive, negative, neutral}
        df["sentiment_pos"] = [s["positive"] for s in scores]
        df["sentiment_neg"] = [s["negative"] for s in scores]
        df["sentiment_neu"] = [s["neutral"] for s in scores]
        df["compound"] = df["sentiment_pos"] - df["sentiment_neg"]
    else:
        df["sentiment_pos"] = 0.333
        df["sentiment_neg"] = 0.333
        df["sentiment_neu"] = 0.334
        df["compound"] = 0.0

    # Aggregate to daily — only articles before 09:30 NY count for that day
    from zoneinfo import ZoneInfo
    ny = ZoneInfo("America/New_York")
    df["date"] = df["datetime"].apply(
        lambda dt: (dt.astimezone(ny).date() if dt.astimezone(ny).hour < 9
                    or (dt.astimezone(ny).hour == 9 and dt.astimezone(ny).minute < 30)
                    else (dt.astimezone(ny) + timedelta(days=1)).date())
    )
    daily = df.groupby("date").agg(
        sentiment_pos=("sentiment_pos", "mean"),
        sentiment_neg=("sentiment_neg", "mean"),
        sentiment_neu=("sentiment_neu", "mean"),
        compound=("compound", "mean"),
        n_articles=("text", "count"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily.set_index("date", inplace=True)
    return daily


def fetch_and_embed_news(days_back: int = 7) -> dict[str, pd.DataFrame]:
    client = _get_finnhub_client()
    if client is None:
        return {}

    embedder = None
    if _FINBERT_AVAILABLE:
        from models.finbert_embedder import FinBERTEmbedder
        embedder = FinBERTEmbedder()

    results = {}
    for ticker in TICKERS:
        path = _news_path(ticker)
        existing = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
        daily = fetch_news_for_ticker(ticker, client, embedder, days_back=days_back)
        if daily.empty:
            results[ticker] = existing
            continue
        if not existing.empty:
            combined = pd.concat([existing, daily[~daily.index.isin(existing.index)]])
            combined.sort_index(inplace=True)
        else:
            combined = daily
        combined.to_parquet(path)
        results[ticker] = combined
        logger.info(f"{ticker}: {len(daily)} new sentiment rows saved")
    return results


if __name__ == "__main__":
    fetch_and_embed_news(days_back=30)
    print("News fetched and embedded.")
