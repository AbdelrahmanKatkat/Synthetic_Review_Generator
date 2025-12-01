# Dataset Comparison Summary

## Files Created

### 1. convert_markdown_to_jsonl.py
**Purpose**: Converts the collected_dataset.md file to JSONL format

**Output**: Creates `real_dataset.jsonl` with the following structure:
```json
{"id": "real_00001", "rating": 5, "text": "Review text here..."}
```

**Results**: Successfully converted 90 real reviews from the markdown file

### 2. compare_datasets.py
**Purpose**: Comprehensive comparison of real vs synthetic reviews

**Metrics Analyzed**:

#### 1. Average Review Length
- Calculates mean, median, min, max, and standard deviation of review lengths (in words)
- Compares real vs synthetic to identify if synthetic reviews are too short/long

#### 2. Sentiment Analysis
- Uses TextBlob to analyze sentiment polarity (-1 to +1)
- Counts positive, neutral, and negative reviews
- Compares mean sentiment between datasets

#### 3. Unrealistic Positivity Check
- Identifies reviews with very high positivity (>0.5 polarity)
- Detects repetitive phrases that appear too frequently
- Flags synthetic reviews that may be unrealistically positive

#### 4. Rating Distribution
- Compares distribution of 1-5 star ratings
- Identifies if synthetic reviews are skewed toward higher ratings

**Visualizations Generated**:
- `length_distribution.png`: Histograms comparing review lengths
- `sentiment_distribution.png`: Sentiment polarity distributions
- `rating_distribution.png`: Side-by-side bar charts of rating counts

All visualizations saved to: `comparison_results/`

## Usage

### Step 1: Convert Markdown to JSONL
```bash
python convert_markdown_to_jsonl.py
```

### Step 2: Run Comparison Analysis
```bash
python compare_datasets.py
```

## Input Files
- `collected_dataset.md`: Real reviews in markdown format
- `dataset.jsonl`: Synthetic reviews (copied from datasets folder)

## Output Files
- `real_dataset.jsonl`: Converted real reviews
- `comparison_results/`: Directory containing visualizations
  - `length_distribution.png`
  - `sentiment_distribution.png`
  - `rating_distribution.png`

## Key Findings
The comparison script provides detailed metrics on:
- Whether synthetic reviews match real review lengths
- If synthetic reviews show different sentiment patterns
- Whether synthetic text contains unrealistic repetition or excessive positivity
- How rating distributions compare between real and synthetic datasets

## Dependencies
```bash
pip install textblob matplotlib seaborn
```

## Note
The comparison script automatically generates both numerical statistics (printed to console) and visual comparisons (saved as PNG files) for easy analysis.
