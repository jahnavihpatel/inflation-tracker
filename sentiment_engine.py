"""
Sentiment Engine
Pulls news headlines via NewsAPI and scores them with VADER.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# VADER — no model download needed, pure rule-based
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except ImportError:
    VADER_OK = False

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

# Map each item to search keywords
ITEM_KEYWORDS = {
    "Eggs (dozen)":                  "egg prices shortage",
    "Milk (whole, gallon)":          "milk dairy prices",
    "Ground beef (lb)":              "beef cattle prices",
    "Chicken breast (lb)":           "chicken poultry prices",
    "Bread, white (loaf)":           "wheat bread flour prices",
    "Butter (lb)":                   "butter dairy prices",
    "Sugar (5 lb bag)":              "sugar prices",
    "Coffee, ground (13 oz)":        "coffee prices",
    "Gasoline, regular (gallon)":    "gasoline oil prices",
    "Electricity (kWh)":             "electricity energy prices",
    "Natural gas (therm)":           "natural gas prices",
    "Heating oil (gallon)":          "heating oil prices",
    "Rent of primary residence":     "rent housing prices inflation",
    "New cars":                      "new car prices auto",
    "Used cars & trucks":            "used car prices",
    "Airline fares":                 "airline ticket prices",
    "Medical care (overall)":        "healthcare medical costs",
    "Prescription drugs":            "drug pharmaceutical prices",
    "College tuition":               "tuition college costs",
    "Apparel (overall)":             "clothing apparel prices tariff",
}

CATEGORY_KEYWORDS = {
    "Food & Beverages":     "food grocery prices inflation",
    "Energy":               "oil gas energy prices geopolitical",
    "Housing":              "rent housing real estate prices",
    "Transportation":       "auto airline transportation prices",
    "Healthcare":           "healthcare medical insurance costs",
    "Education":            "tuition education costs",
    "Apparel":              "clothing apparel tariff prices",
    "Recreation":           "consumer goods prices",
    "Other":                "consumer prices inflation",
}


def fetch_headlines(query: str, api_key: str, days_back: int = 7) -> list[str]:
    """Fetch recent headlines from NewsAPI for a query."""
    if not api_key:
        return []
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": 20,
                "language": "en",
                "apiKey": api_key,
            },
            timeout=10,
        )
        data = resp.json()
        articles = data.get("articles", [])
        return [
            f"{a.get('title', '')} {a.get('description', '')}"
            for a in articles if a.get("title")
        ]
    except Exception:
        return []


def score_sentiment(texts: list[str]) -> dict:
    """Score a list of texts with VADER. Returns compound score -1 to +1."""
    if not VADER_OK or not texts:
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0, "headline_count": 0}

    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t) for t in texts]
    return {
        "compound":  round(sum(s["compound"] for s in scores) / len(scores), 3),
        "positive":  round(sum(s["pos"] for s in scores) / len(scores), 3),
        "negative":  round(sum(s["neg"] for s in scores) / len(scores), 3),
        "neutral":   round(sum(s["neu"] for s in scores) / len(scores), 3),
        "headline_count": len(texts),
    }


def sentiment_to_risk(compound: float) -> tuple[str, str]:
    """Convert compound score to risk label and emoji."""
    if compound < -0.2:
        return "HIGH", "🔴"
    if compound < -0.05:
        return "MEDIUM", "🟡"
    return "LOW", "🟢"


def shopping_advice(compound: float, forecast_pct: float) -> tuple[str, str]:
    """
    Combine sentiment + forecast direction into a shopping recommendation.
    forecast_pct: expected % change in next 3 months from ARIMA forecast.
    """
    if forecast_pct > 3 and compound < -0.1:
        return "Buy now", "🔴 Price rising + negative news sentiment. Stock up."
    if forecast_pct > 3:
        return "Buy soon", "🟡 Price trending up. Consider buying ahead."
    if forecast_pct < -2 and compound > 0:
        return "Wait", "🟢 Price expected to drop. Hold off for now."
    if compound < -0.2:
        return "Watch", "🟡 Negative news sentiment. Monitor before buying."
    return "Neutral", "⚪ No strong signal either way."


def get_item_sentiment(item: str, api_key: str) -> dict:
    """Full pipeline: fetch headlines → score → label for one item."""
    query    = ITEM_KEYWORDS.get(item, item)
    headlines = fetch_headlines(query, api_key)
    scores   = score_sentiment(headlines)
    risk, emoji = sentiment_to_risk(scores["compound"])
    return {
        "item": item,
        "risk_level": risk,
        "risk_emoji": emoji,
        "compound_score": scores["compound"],
        "headline_count": scores["headline_count"],
        "headlines": headlines[:5],  # keep top 5 for display
        **scores,
    }


def get_all_sentiment(api_key: str, items: list[str]) -> pd.DataFrame:
    """Fetch sentiment for all items. Cached by caller."""
    rows = []
    for item in items:
        row = get_item_sentiment(item, api_key)
        rows.append(row)
    return pd.DataFrame(rows)
