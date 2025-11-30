"""
Sentiment Validator

Provides simple sentiment analysis utilities for review text.

Functions:
- compute_sentiment(text) -> float
    Returns a compound sentiment score in [-1.0, 1.0] (negative -> positive).
- sentiment_label(score, thresholds=( -0.05, 0.05)) -> str
    Converts a compound score into "negative", "neutral", "positive".
- sentiment_vs_rating_flag(text, rating, tolerance=0.4) -> dict
    Compares model text sentiment to numeric rating and flags mismatches.
- batch_sentiment_metrics(texts, ratings=None) -> dict
    Compute distribution metrics across many samples.
"""

from typing import List, Optional, Tuple, Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create a single shared analyzer instance (cheap, thread-safe for our use)
_ANALYZER = SentimentIntensityAnalyzer()


def compute_sentiment(text: str) -> float:
    """
    Compute sentiment compound score for a single text.

    Args:
        text: review text (English recommended).

    Returns:
        compound score (float) in range [-1.0, 1.0], where:
          -1.0 strongly negative, 0 neutral, +1.0 strongly positive.

    Notes:
    - VADER is rule-based and fast. It's a good default for short reviews.
    - For other languages or domain-specific sentiment, replace with a different model.
    """
    if not text:
        return 0.0
    scores = _ANALYZER.polarity_scores(text)
    return float(scores.get("compound", 0.0))


def sentiment_label(score: float, neutral_interval: Tuple[float, float] = (-0.05, 0.05)) -> str:
    """
    Convert a compound sentiment score into categorical label.

    Args:
        score: compound sentiment score from compute_sentiment.
        neutral_interval: (low, high) bounds considered "neutral".

    Returns:
        One of: "negative", "neutral", "positive".
    """
    low, high = neutral_interval
    if score <= low:
        return "negative"
    if score >= high:
        return "positive"
    return "neutral"


def sentiment_vs_rating_flag(
    text: str,
    rating: int,
    rating_scale: Tuple[int, int] = (1, 5),
    tolerance: float = 0.4,
) -> Dict[str, Any]:
    """
    Compare textual sentiment with numeric rating and return a diagnostic dict.

    Purpose:
      - Detect cases where text sentiment contradicts rating (possible model hallucination
        or prompt-template mismatch), e.g., rating=1 but text is strongly positive.

    Args:
        text: generated review text.
        rating: integer rating (e.g., 1..5).
        rating_scale: tuple (min_rating, max_rating).
        tolerance: allowed difference between normalized rating and sentiment before flagging.

    Returns:
        Dict with keys:
          - sentiment_score: float (-1..1)
          - sentiment_label: "negative"/"neutral"/"positive"
          - normalized_rating: float in [-1..1] mapped from rating scale
          - rating_label: "negative"/"neutral"/"positive"
          - mismatch: bool (True if labels differ AND distance > tolerance)
          - distance: float absolute difference between sentiment_score and normalized_rating

    Implementation details:
      - We map rating to [-1, 1] so it is comparable to VADER compound scores.
      - A mismatch is reported if the absolute distance exceeds `tolerance`.
    """
    sentiment_score = compute_sentiment(text)
    s_label = sentiment_label(sentiment_score)

    # Normalize rating to [-1, 1]
    min_r, max_r = rating_scale
    if min_r >= max_r:
        raise ValueError("rating_scale must be (min, max) with min < max")
    # linear map rating to [-1,1]
    normalized = ((rating - min_r) / (max_r - min_r)) * 2.0 - 1.0

    r_label = sentiment_label(normalized)

    distance = abs(sentiment_score - normalized)
    mismatch = distance > tolerance and s_label != r_label

    return {
        "sentiment_score": sentiment_score,
        "sentiment_label": s_label,
        "normalized_rating": normalized,
        "rating_label": r_label,
        "distance": distance,
        "mismatch": mismatch,
    }


def batch_sentiment_metrics(texts: List[str], ratings: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Compute sentiment metrics across a corpus of generated texts.

    Args:
        texts: list of review texts.
        ratings: optional list of numeric ratings aligned with texts.

    Returns:
        Dict with:
          - avg_sentiment: mean compound score
          - pct_positive: fraction labeled positive
          - pct_neutral: fraction labeled neutral
          - pct_negative: fraction labeled negative
          - rating_mismatch_rate: fraction of samples flagged as mismatch (if ratings provided)
    """
    n = len(texts)
    if n == 0:
        return {
            "avg_sentiment": 0.0,
            "pct_positive": 0.0,
            "pct_neutral": 0.0,
            "pct_negative": 0.0,
            "rating_mismatch_rate": None,
        }

    scores = [compute_sentiment(t) for t in texts]
    labels = [sentiment_label(s) for s in scores]

    avg_sent = sum(scores) / n
    pct_positive = sum(1 for l in labels if l == "positive") / n
    pct_neutral = sum(1 for l in labels if l == "neutral") / n
    pct_negative = sum(1 for l in labels if l == "negative") / n

    mismatch_rate = None
    if ratings is not None:
        if len(ratings) != n:
            raise ValueError("ratings length must match texts length")
        mismatches = 0
        for t, r in zip(texts, ratings):
            info = sentiment_vs_rating_flag(t, r)
            if info["mismatch"]:
                mismatches += 1
        mismatch_rate = mismatches / n

    return {
        "avg_sentiment": avg_sent,
        "pct_positive": pct_positive,
        "pct_neutral": pct_neutral,
        "pct_negative": pct_negative,
        "rating_mismatch_rate": mismatch_rate,
    }