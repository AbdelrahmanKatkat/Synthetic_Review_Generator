# Synthetic Review Generator

Generate realistic product reviews using Gemini Flash 1.5 and Mistral-7B with quality guardrails.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
GEMINI_API_KEY=your_actual_gemini_key
HUGGINGFACE_TOKEN=your_hf_token  # Optional, for Mistral
```

**Get API Keys:**
- **Gemini**: https://aistudio.google.com/app/apikey
- **HuggingFace**: https://huggingface.co/settings/tokens (free account)

### 3. Generate Reviews

```bash
python main.py generate
```

That's it! Reviews will be generated in `data/runs/[timestamp]/`

## What You Get

```
data/runs/2024-11-27_21-10/
├── reviews.jsonl          # Your dataset (400 reviews)
├── quality_report.md      # Quality metrics
└── config.yaml            # Config snapshot
```

## Configuration

Edit `config.yaml` to customize:

- **Domain**: Change product type
- **Personas**: Add/modify reviewer personas
- **Sample count**: Number of reviews to generate
- **Quality thresholds**: Similarity, domain score, etc.

## CLI Commands

```bash
# Generate reviews
python main.py generate

# Custom config
python main.py generate --config my_config.yaml

# Validate existing dataset
python main.py validate data/runs/xxx/reviews.jsonl
```

## Architecture

**Files (only 6!):**
- `models.py` - LLM adapters (Gemini, Mistral)
- `validators.py` - Quality checks (diversity, sentiment, domain)
- `generator.py` - Review generation logic
- `main.py` - CLI interface
- `config.yaml` - Configuration
- `.env` - API keys

**Flow:**
1. Load config
2. Initialize models (Gemini + Mistral)
3. For each review:
   - Sample persona and rating
   - Generate with LLM
   - Validate quality
   - Retry if failed (up to 3 times)
4. Save to JSONL
5. Generate quality report

## Quality Guardrails

✅ **Diversity**: Checks embedding similarity (max 0.92)  
✅ **Sentiment**: Rating must match text sentiment  
✅ **Domain**: Must mention valid product features  

Failed reviews are regenerated automatically.

## Models

### Gemini Flash 1.5
- Fast, reliable, strong JSON output
- Used for ~50% of reviews
- Cost: ~$0.50 for 400 reviews

### Mistral-7B (Optional)
- Open-source, different writing style
- Free via HuggingFace Inference API
- Used for ~50% of reviews

## Example Output

```json
{
  "title": "Great for Team Collaboration",
  "body": "We've been using this for 3 months and the kanban board...",
  "tags": ["kanban", "integration"],
  "rating": 4,
  "persona": "marketing_manager",
  "model": "gemini-1.5-flash",
  "date": "2024-08-15",
  "quality": {
    "passed": true,
    "diversity": {"max_similarity": 0.43},
    "sentiment": {"sentiment_score": 0.68},
    "domain": {"score": 0.85, "features_mentioned": ["kanban", "slack integration"]}
  }
}
```

## Troubleshooting

**No HuggingFace token?**
- Generator will use Gemini only (still works!)
- Add token later to enable Mistral

**API rate limits?**
- Reduce `sample_count` in config
- Built-in retry logic handles transient errors

**Low quality scores?**
- Adjust thresholds in `config.yaml`
- Check that domain features match your use case

## System Design

See [Problem Statement/System_Design.md](Problem%20Statement/System_Design.md) for detailed architecture.

## License

MIT
