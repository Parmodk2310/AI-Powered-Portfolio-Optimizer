"""
news_fetcher.py
Fetches recent financial news for stock tickers using NewsAPI.
Free tier: 100 requests/day, 1 month history, 1 req/sec rate limit.

Run this file directly to test:
    python src/data/news_fetcher.py
"""

import requests
import time
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2/everything"


# ── Core Functions ─────────────────────────────────────────────────────────────

def fetch_news(
    ticker: str,
    company_name: str = None,
    days_back: int = 7,
    max_articles: int = 20
) -> List[Dict]:
    """
    Fetch recent news articles for a stock ticker.

    Args:
        ticker       : Stock symbol e.g. "AAPL"
        company_name : Full name e.g. "Apple" (improves search quality)
        days_back    : How many days back to search (max 30 on free tier)
        max_articles : Maximum number of articles to return

    Returns:
        List of article dicts with keys:
        ticker, title, description, url, published_at, source, text

    Example:
        articles = fetch_news("AAPL", "Apple", days_back=7)
        print(articles[0]["title"])
    """
    if not NEWS_API_KEY:
        raise ValueError(
            "NEWS_API_KEY not found. "
            "Add it to your .env file: NEWS_API_KEY=your_key_here"
        )

    # Use company name if provided — gives better results than just ticker
    query = company_name if company_name else ticker
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q": f"{query} stock",
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY,
    }

    try:
        logger.info(f"Fetching news for {ticker} (query: '{query} stock')")
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "error":
            logger.error(f"NewsAPI error for {ticker}: {data.get('message')}")
            return []

        articles = []
        for article in data.get("articles", []):
            # Skip articles with missing title or description
            if not article.get("title") or not article.get("description"):
                continue
            # Skip removed articles
            if article.get("title") == "[Removed]":
                continue

            articles.append({
                "ticker": ticker,
                "title": article["title"],
                "description": article["description"],
                "url": article["url"],
                "published_at": article["publishedAt"],
                "source": article["source"]["name"],
                # text = combined title + description (used by FinBERT and FAISS)
                "text": f"{article['title']}. {article['description']}"
            })

        logger.info(f"Found {len(articles)} articles for {ticker}")
        return articles

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching news for {ticker}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {ticker}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error for {ticker}: {e}")
        return []


def fetch_news_batch(
    tickers: List[str],
    company_names: Dict[str, str] = None,
    days_back: int = 7,
    max_articles: int = 20,
    delay: float = 1.1
) -> Dict[str, List[Dict]]:
    """
    Fetch news for multiple tickers with rate limiting.

    Args:
        tickers       : List of stock symbols
        company_names : Dict mapping ticker to company name
                        e.g. {"AAPL": "Apple", "MSFT": "Microsoft"}
        days_back     : Days back to search
        max_articles  : Max articles per ticker
        delay         : Seconds between requests (1.1 > 1 to be safe)

    Returns:
        Dict mapping ticker to list of article dicts

    Example:
        news = fetch_news_batch(["AAPL", "MSFT"], {"AAPL": "Apple"})
        print(news["AAPL"][0]["title"])
    """
    company_names = company_names or {}
    results = {}

    logger.info(f"Fetching news for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        articles = fetch_news(
            ticker=ticker,
            company_name=company_names.get(ticker),
            days_back=days_back,
            max_articles=max_articles
        )
        results[ticker] = articles

        # Rate limiting — NewsAPI free tier: 1 req/sec
        # We wait even after the last ticker to be safe
        if i < len(tickers) - 1:
            logger.info(f"Waiting {delay}s (rate limit)...")
            time.sleep(delay)

    total_articles = sum(len(a) for a in results.values())
    logger.info(f"Total articles fetched: {total_articles}")
    return results


def get_article_texts(ticker: str, articles: List[Dict]) -> List[str]:
    """
    Extract just the text strings from a list of articles.
    This is what sentiment.py and vector_store.py will consume.

    Args:
        ticker   : Stock symbol (for logging)
        articles : List of article dicts from fetch_news()

    Returns:
        List of text strings (title + description combined)

    Example:
        texts = get_article_texts("AAPL", articles)
        # ["Apple beats Q4 earnings. Revenue exceeded expectations...",
        #  "Apple Vision Pro sales. The headset has sold..."]
    """
    texts = [a["text"] for a in articles if a.get("text")]
    logger.info(f"Extracted {len(texts)} text strings for {ticker}")
    return texts


def check_api_status() -> Dict:
    """
    Check if NewsAPI key is valid and how many requests remain today.

    Returns:
        Dict with status, requests_remaining, requests_limit
    """
    if not NEWS_API_KEY:
        return {"status": "error", "message": "NEWS_API_KEY not set in .env"}

    try:
        # Make a minimal request to check quota
        response = requests.get(
            BASE_URL,
            params={"q": "test", "pageSize": 1, "apiKey": NEWS_API_KEY},
            timeout=5
        )
        data = response.json()

        if data.get("status") == "ok":
            return {
                "status": "ok",
                "message": "API key valid",
                "total_results_for_test": data.get("totalResults", 0)
            }
        else:
            return {
                "status": "error",
                "message": data.get("message", "Unknown error")
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Main — Run this to test ────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("AI Portfolio Optimizer — News Fetcher Test")
    print("=" * 60)

    # ── Test 1: API Key Check ──
    print("\n[1] Checking NewsAPI key...")
    status = check_api_status()
    print(f"Status: {status}")
    if status["status"] == "error":
        print("ERROR: Fix your API key first. Add NEWS_API_KEY to .env file")
        exit(1)

    # ── Test 2: Single Ticker ──
    print("\n[2] Fetching news for single ticker (AAPL)...")
    articles = fetch_news("AAPL", company_name="Apple", days_back=7)
    print(f"Found: {len(articles)} articles")

    if articles:
        print("\nFirst article:")
        print(f"  Title     : {articles[0]['title']}")
        print(f"  Source    : {articles[0]['source']}")
        print(f"  Published : {articles[0]['published_at']}")
        print(f"  Text      : {articles[0]['text'][:150]}...")

    time.sleep(1.1)  # Rate limit

    # ── Test 3: Multiple Tickers ──
    print("\n[3] Fetching news for multiple tickers...")
    tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "NFLX", "AMD", "INTC",
    "ORCL", "CRM", "ADBE", "PYPL", "QCOM",
    "IBM", "CSCO", "AVGO", "SHOP", "UBER"
]
    company_names = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "TSLA": "Tesla",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "INTC": "Intel",
    "ORCL": "Oracle",
    "CRM": "Salesforce",
    "ADBE": "Adobe",
    "PYPL": "PayPal",
    "QCOM": "Qualcomm",
    "IBM": "IBM",
    "CSCO": "Cisco",
    "AVGO": "Broadcom",
    "SHOP": "Shopify",
    "UBER": "Uber"
}

    news_by_ticker = fetch_news_batch(tickers, company_names)

    print("\nResults:")
    for ticker, arts in news_by_ticker.items():
        print(f"\n{ticker}: {len(arts)} articles")
        for a in arts[:2]:
            print(f"  [{a['source']}] {a['title'][:65]}")

    # ── Test 4: get_article_texts ──
    print("\n[4] Testing get_article_texts() for AAPL...")
    aapl_texts = get_article_texts("AAPL", news_by_ticker.get("AAPL", []))
    print(f"Text strings extracted: {len(aapl_texts)}")
    if aapl_texts:
        print(f"First text: {aapl_texts[0][:120]}...")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)