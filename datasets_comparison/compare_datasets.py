import json
import statistics
from pathlib import Path
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

def load_jsonl(filepath):
    """Load JSONL file and return list of reviews."""
    reviews = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))
    return reviews

def calculate_avg_length(reviews):
    """Calculate average review length in words."""
    lengths = [len(review['text'].split()) for review in reviews]
    return {
        'mean': statistics.mean(lengths),
        'median': statistics.median(lengths),
        'min': min(lengths),
        'max': max(lengths),
        'std_dev': statistics.stdev(lengths) if len(lengths) > 1 else 0
    }

def analyze_sentiment(reviews):
    """Analyze sentiment using TextBlob."""
    sentiments = []
    for review in reviews:
        blob = TextBlob(review['text'])
        sentiments.append(blob.sentiment.polarity)
    
    return {
        'mean_polarity': statistics.mean(sentiments),
        'median_polarity': statistics.median(sentiments),
        'min_polarity': min(sentiments),
        'max_polarity': max(sentiments),
        'std_dev': statistics.stdev(sentiments) if len(sentiments) > 1 else 0,
        'positive_count': sum(1 for s in sentiments if s > 0.1),
        'neutral_count': sum(1 for s in sentiments if -0.1 <= s <= 0.1),
        'negative_count': sum(1 for s in sentiments if s < -0.1)
    }

def check_unrealistic_positivity(reviews):
    """Check if synthetic reviews are unrealistically positive."""
    # Count how many reviews have sentiment > 0.5 (very positive)
    very_positive = sum(1 for r in reviews if TextBlob(r['text']).sentiment.polarity > 0.5)
    percentage = (very_positive / len(reviews)) * 100
    
    # Check for repetitive patterns
    texts = [r['text'].lower() for r in reviews]
    common_phrases = {}
    
    for text in texts:
        words = text.split()
        # Look for repeated 3-word phrases
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if phrase in common_phrases:
                common_phrases[phrase] += 1
            else:
                common_phrases[phrase] = 1
    
    # Find most common repetitive phrases
    repetitive_phrases = {k: v for k, v in common_phrases.items() if v > len(reviews) * 0.1}
    
    return {
        'very_positive_percentage': percentage,
        'very_positive_count': very_positive,
        'repetitive_phrases': dict(sorted(repetitive_phrases.items(), key=lambda x: x[1], reverse=True)[:10])
    }

def compare_rating_distributions(real_reviews, synthetic_reviews):
    """Compare rating distributions."""
    real_ratings = [r['rating'] for r in real_reviews]
    synthetic_ratings = [r['rating'] for r in synthetic_reviews]
    
    return {
        'real': {
            'mean': statistics.mean(real_ratings),
            'median': statistics.median(real_ratings),
            'mode': statistics.mode(real_ratings) if real_ratings else None,
            'distribution': {i: real_ratings.count(i) for i in range(1, 6)}
        },
        'synthetic': {
            'mean': statistics.mean(synthetic_ratings),
            'median': statistics.median(synthetic_ratings),
            'mode': statistics.mode(synthetic_ratings) if synthetic_ratings else None,
            'distribution': {i: synthetic_ratings.count(i) for i in range(1, 6)}
        }
    }

def visualize_comparison(real_reviews, synthetic_reviews, output_dir):
    """Create visualization comparing real vs synthetic."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Review Length Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    real_lengths = [len(r['text'].split()) for r in real_reviews]
    synthetic_lengths = [len(r['text'].split()) for r in synthetic_reviews]
    
    axes[0].hist(real_lengths, bins=30, alpha=0.7, label='Real', color='blue')
    axes[0].set_title('Real Review Lengths')
    axes[0].set_xlabel('Words')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(synthetic_lengths, bins=30, alpha=0.7, label='Synthetic', color='orange')
    axes[1].set_title('Synthetic Review Lengths')
    axes[1].set_xlabel('Words')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sentiment Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    real_sentiments = [TextBlob(r['text']).sentiment.polarity for r in real_reviews]
    synthetic_sentiments = [TextBlob(r['text']).sentiment.polarity for r in synthetic_reviews]
    
    axes[0].hist(real_sentiments, bins=30, alpha=0.7, label='Real', color='blue')
    axes[0].set_title('Real Review Sentiments')
    axes[0].set_xlabel('Sentiment Polarity')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    axes[1].hist(synthetic_sentiments, bins=30, alpha=0.7, label='Synthetic', color='orange')
    axes[1].set_title('Synthetic Review Sentiments')
    axes[1].set_xlabel('Sentiment Polarity')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Rating Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    real_ratings = [r['rating'] for r in real_reviews]
    synthetic_ratings = [r['rating'] for r in synthetic_reviews]
    
    x = [1, 2, 3, 4, 5]
    real_counts = [real_ratings.count(i) for i in x]
    synthetic_counts = [synthetic_ratings.count(i) for i in x]
    
    width = 0.35
    x_pos = range(len(x))
    
    ax.bar([p - width/2 for p in x_pos], real_counts, width, label='Real', color='blue', alpha=0.7)
    ax.bar([p + width/2 for p in x_pos], synthetic_counts, width, label='Synthetic', color='orange', alpha=0.7)
    
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title('Rating Distribution: Real vs Synthetic')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # File paths
    real_dataset = Path('e:/On Going Projects/Capital Growth/Job Growth/Easygenerator/Synthetic_Reviews/Synthetic_Review_Generator/datasets_comparison/real_dataset.jsonl')
    synthetic_dataset = Path('e:/On Going Projects/Capital Growth/Job Growth/Easygenerator/Synthetic_Reviews/Synthetic_Review_Generator/datasets_comparison/dataset.jsonl')
    output_dir = Path('e:/On Going Projects/Capital Growth/Job Growth/Easygenerator/Synthetic_Reviews/Synthetic_Review_Generator/datasets_comparison/comparison_results')
    
    # Load datasets
    print("Loading datasets...")
    real_reviews = load_jsonl(real_dataset)
    synthetic_reviews = load_jsonl(synthetic_dataset)
    
    print(f"Real reviews: {len(real_reviews)}")
    print(f"Synthetic reviews: {len(synthetic_reviews)}")
    print("\n" + "="*80 + "\n")
    
    # 1. Average review length
    print("1. REVIEW LENGTH ANALYSIS")
    print("-" * 80)
    real_length = calculate_avg_length(real_reviews)
    synthetic_length = calculate_avg_length(synthetic_reviews)
    
    print(f"Real Reviews:")
    print(f"  Mean length: {real_length['mean']:.2f} words")
    print(f"  Median length: {real_length['median']:.2f} words")
    print(f"  Range: {real_length['min']} - {real_length['max']} words")
    print(f"  Std Dev: {real_length['std_dev']:.2f}")
    print()
    print(f"Synthetic Reviews:")
    print(f"  Mean length: {synthetic_length['mean']:.2f} words")
    print(f"  Median length: {synthetic_length['median']:.2f} words")
    print(f"  Range: {synthetic_length['min']} - {synthetic_length['max']} words")
    print(f"  Std Dev: {synthetic_length['std_dev']:.2f}")
    print()
    print(f"Difference: {abs(real_length['mean'] - synthetic_length['mean']):.2f} words")
    print("\n" + "="*80 + "\n")
    
    # 2. Sentiment analysis
    print("2. SENTIMENT ANALYSIS")
    print("-" * 80)
    real_sentiment = analyze_sentiment(real_reviews)
    synthetic_sentiment = analyze_sentiment(synthetic_reviews)
    
    print(f"Real Reviews:")
    print(f"  Mean polarity: {real_sentiment['mean_polarity']:.4f}")
    print(f"  Median polarity: {real_sentiment['median_polarity']:.4f}")
    print(f"  Range: {real_sentiment['min_polarity']:.4f} - {real_sentiment['max_polarity']:.4f}")
    print(f"  Positive: {real_sentiment['positive_count']} ({real_sentiment['positive_count']/len(real_reviews)*100:.1f}%)")
    print(f"  Neutral: {real_sentiment['neutral_count']} ({real_sentiment['neutral_count']/len(real_reviews)*100:.1f}%)")
    print(f"  Negative: {real_sentiment['negative_count']} ({real_sentiment['negative_count']/len(real_reviews)*100:.1f}%)")
    print()
    print(f"Synthetic Reviews:")
    print(f"  Mean polarity: {synthetic_sentiment['mean_polarity']:.4f}")
    print(f"  Median polarity: {synthetic_sentiment['median_polarity']:.4f}")
    print(f"  Range: {synthetic_sentiment['min_polarity']:.4f} - {synthetic_sentiment['max_polarity']:.4f}")
    print(f"  Positive: {synthetic_sentiment['positive_count']} ({synthetic_sentiment['positive_count']/len(synthetic_reviews)*100:.1f}%)")
    print(f"  Neutral: {synthetic_sentiment['neutral_count']} ({synthetic_sentiment['neutral_count']/len(synthetic_reviews)*100:.1f}%)")
    print(f"  Negative: {synthetic_sentiment['negative_count']} ({synthetic_sentiment['negative_count']/len(synthetic_reviews)*100:.1f}%)")
    print()
    print(f"Sentiment Difference: {abs(real_sentiment['mean_polarity'] - synthetic_sentiment['mean_polarity']):.4f}")
    print("\n" + "="*80 + "\n")
    
    # 3. Unrealistic positivity check
    print("3. UNREALISTIC POSITIVITY CHECK")
    print("-" * 80)
    real_positivity = check_unrealistic_positivity(real_reviews)
    synthetic_positivity = check_unrealistic_positivity(synthetic_reviews)
    
    print(f"Real Reviews:")
    print(f"  Very positive (>0.5): {real_positivity['very_positive_count']} ({real_positivity['very_positive_percentage']:.1f}%)")
    print(f"  Most repetitive phrases: {len(real_positivity['repetitive_phrases'])}")
    if real_positivity['repetitive_phrases']:
        print("  Top 3 repeated phrases:")
        for phrase, count in list(real_positivity['repetitive_phrases'].items())[:3]:
            print(f"    '{phrase}': {count} times")
    print()
    print(f"Synthetic Reviews:")
    print(f"  Very positive (>0.5): {synthetic_positivity['very_positive_count']} ({synthetic_positivity['very_positive_percentage']:.1f}%)")
    print(f"  Most repetitive phrases: {len(synthetic_positivity['repetitive_phrases'])}")
    if synthetic_positivity['repetitive_phrases']:
        print("  Top 3 repeated phrases:")
        for phrase, count in list(synthetic_positivity['repetitive_phrases'].items())[:3]:
            print(f"    '{phrase}': {count} times")
    print("\n" + "="*80 + "\n")
    
    # 4. Rating distribution
    print("4. RATING DISTRIBUTION")
    print("-" * 80)
    rating_comparison = compare_rating_distributions(real_reviews, synthetic_reviews)
    
    print(f"Real Reviews:")
    print(f"  Mean rating: {rating_comparison['real']['mean']:.2f}")
    print(f"  Median rating: {rating_comparison['real']['median']:.1f}")
    print(f"  Mode rating: {rating_comparison['real']['mode']}")
    print(f"  Distribution:")
    for rating, count in rating_comparison['real']['distribution'].items():
        print(f"    {rating} stars: {count} ({count/len(real_reviews)*100:.1f}%)")
    print()
    print(f"Synthetic Reviews:")
    print(f"  Mean rating: {rating_comparison['synthetic']['mean']:.2f}")
    print(f"  Median rating: {rating_comparison['synthetic']['median']:.1f}")
    print(f"  Mode rating: {rating_comparison['synthetic']['mode']}")
    print(f"  Distribution:")
    for rating, count in rating_comparison['synthetic']['distribution'].items():
        print(f"    {rating} stars: {count} ({count/len(synthetic_reviews)*100:.1f}%)")
    print("\n" + "="*80 + "\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    visualize_comparison(real_reviews, synthetic_reviews, output_dir)
    print(f"Visualizations saved to: {output_dir}")
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
