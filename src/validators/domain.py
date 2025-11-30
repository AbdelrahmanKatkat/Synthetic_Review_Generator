"""
Domain Validator

Provides lightweight checks to determine whether a generated review is plausible
for a chosen product domain (e.g., project-management SaaS).

Functions:
- feature_mention_score(text, features) -> float: fraction of lexicon items mentioned
- contains_blacklisted_phrase(text, blacklist) -> Optional[str]: returns offending phrase or None
- domain_validator(text, features, blacklist, feature_threshold) -> dict: combined pass/fail + scores
- batch_domain_metrics(texts, features, blacklist, feature_threshold) -> dict: corpus-level metrics
"""

from typing import List, Dict, Optional
import re
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for robust substring matching."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _fuzzy_contains(text: str, phrase: str, threshold: float = 0.85) -> bool:
    """
    Check whether `phrase` is approximately contained in `text`.
    
    - Try exact substring match first (fast)
    - Otherwise, scan text windows of similar length using SequenceMatcher
    """
    text_n = _normalize(text)
    phrase_n = _normalize(phrase)

    # Exact substring
    if phrase_n in text_n:
        return True

    # Sliding-window fuzzy match
    p_len = len(phrase_n.split())
    words = text_n.split()
    if p_len == 0 or len(words) == 0:
        return False

    # Window sizes: p_len-1 to p_len+1 to allow small drift
    window_sizes = [max(1, p_len - 1), p_len, p_len + 1]
    for w in window_sizes:
        for i in range(0, max(1, len(words) - w + 1)):
            window = " ".join(words[i : i + w])
            ratio = SequenceMatcher(None, phrase_n, window).ratio()
            if ratio >= threshold:
                return True
    return False


def feature_mention_score(text: str, features: List[str]) -> float:
    """
    Compute the fraction of `features` that are mentioned in `text`.
    
    Returns:
      - score in [0.0, 1.0] = (#features_found) / (len(features))
      - If features list is empty, returns 0.0
    """
    if not features:
        return 0.0

    found = sum(1 for feat in features if _fuzzy_contains(text, feat))
    return found / len(features)


def contains_blacklisted_phrase(text: str, blacklist: List[str]) -> Optional[str]:
    """
    Detect if any blacklisted phrase appears (fuzzily) in `text`.
    
    Returns:
      - the matched blacklist phrase (original form) if found
      - None otherwise
    """
    for bad in blacklist:
        if _fuzzy_contains(text, bad, threshold=0.80):
            return bad
    return None


def domain_validator(
    text: str,
    features: List[str],
    blacklist: List[str],
    feature_threshold: float = 0.05,
) -> Dict[str, object]:
    """
    Combined domain check for a single text.
    
    Args:
      - text: generated review text
      - features: list of expected domain phrases
      - blacklist: list of impossible phrases
      - feature_threshold: minimum fraction of features mentioned to consider "domain-y"
    
    Returns:
      dict with keys:
        - domain_score: feature_mention_score (float)
        - blacklisted: matched blacklist phrase or None
        - passed: bool (True if domain_score >= feature_threshold and no blacklist)
        - reasons: list[str] explaining failures (empty if passed)
    """
    reasons: List[str] = []
    score = feature_mention_score(text, features)
    black = contains_blacklisted_phrase(text, blacklist)

    if black:
        reasons.append(f"blacklist_match:{black}")

    if score < feature_threshold:
        reasons.append(f"low_domain_score:{score:.3f}")

    passed = (black is None) and (score >= feature_threshold)

    return {
        "domain_score": score,
        "blacklisted": black,
        "passed": passed,
        "reasons": reasons,
    }


def batch_domain_metrics(
    texts: List[str], features: List[str], blacklist: List[str], feature_threshold: float = 0.05
) -> Dict[str, object]:
    """
    Corpus-level domain metrics.
    
    Returns:
      - avg_domain_score: mean of feature_mention_score across texts
      - pct_passing: fraction of texts that pass domain_validator
      - blacklist_hits: list of tuples (index, matched_phrase) where blacklists were found
      - examples_low_score: up to 5 sample indices with lowest domain scores
    """
    n = len(texts)
    if n == 0:
        return {
            "avg_domain_score": 0.0,
            "pct_passing": 0.0,
            "blacklist_hits": [],
            "examples_low_score": [],
        }

    scores = []
    passes = 0
    blacklist_hits = []

    for i, t in enumerate(texts):
        res = domain_validator(t, features, blacklist, feature_threshold)
        scores.append(res["domain_score"])
        if res["blacklisted"]:
            blacklist_hits.append((i, res["blacklisted"]))
        if res["passed"]:
            passes += 1

    avg_score = sum(scores) / n
    pct_passing = passes / n

    # Indices of lowest domain scores (for quick manual inspection)
    sorted_idx = sorted(range(n), key=lambda i: scores[i])
    examples_low = sorted_idx[: min(5, n)]

    return {
        "avg_domain_score": avg_score,
        "pct_passing": pct_passing,
        "blacklist_hits": blacklist_hits,
        "examples_low_score": examples_low,
    }
