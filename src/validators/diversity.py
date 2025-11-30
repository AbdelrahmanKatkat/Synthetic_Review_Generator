"""
Diversity Validator

This module provides functions to measure the diversity of a list of generated texts.

Metrics:
- Vocabulary overlap (word repetition across samples)
- Semantic similarity (cosine similarity between TF-IDF vectors)

These metrics help detect low-diversity outputs, e.g., when a model
generates repetitive or overly similar reviews.
"""


from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def vocabulary_overlap(texts: List[str]) -> float:
    """
    Compute the fraction of repeated words across all samples.

    Args:
        texts (List[str]): List of generated text samples.

    Returns:
        float: fraction of words that appear in more than one text (0 = fully unique, 1 = all words repeated).

    Implementation:
    - Split texts into words.
    - Count frequency of each word across all samples.
    - Compute fraction of words appearing >1 times.
    """

    word_counts: Dict[str, int] = {}
    for text in texts:
        for word in text.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1

    if not word_counts:
        return 0.0

    repeated = sum(1 for count in word_counts.values() if count > 1)
    total_words = len(word_counts)

    return repeated / total_words

def semantic_similarity(texts: List[str]) -> float:
    """
    Compute the average pairwise cosine similarity between text samples using TF-IDF vectors.

    Args:
        texts (List[str]): List of generated text samples.

    Returns:
        float: average similarity (0 = very diverse, 1 = very similar).

    Implementation:
    - Convert texts to TF-IDF vectors.
    - Compute cosine similarity matrix.
    - Return average of the upper triangle (pairwise comparisons) to avoid self-similarity.
    """

    if len(texts) < 2:
        return 0.0  # only one text, diversity undefined

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarity_matrix = cosine_similarity(tfidf_matrix) # The result is a square matrix (similarity_matrix) where the element at position (i,j) represents the cosine similarity between document i and document j. The values range from -1 to 1, where 1 indicates identical documents, 0 indicates orthogonal (no similarity), and negative values indicate opposite directions (not typical for text data).
    # extract upper triangle values excluding diagonal
    n = len(texts)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += similarity_matrix[i, j]
            count += 1

    return total / count if count > 0 else 0.0 # average of the upper triangle

def diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Compute multiple diversity metrics at once.

    Args:
        texts (List[str]): List of generated samples.

    Returns:
        Dict[str, float]: {
            "vocab_overlap": float,
            "semantic_similarity": float
        }

    Usage:
        >>> metrics = diversity_metrics(generated_texts)
        >>> if metrics["semantic_similarity"] > 0.8:
        >>>     print("Samples are too similar!")
    """
    return {
        "vocab_overlap": vocabulary_overlap(texts),
        "semantic_similarity": semantic_similarity(texts),
    }