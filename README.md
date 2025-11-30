# Synthetic Review Generator

A production-ready system for generating high-quality synthetic product reviews using multiple LLM providers with built-in quality validation.

## 🚀 Quick Start

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
```env
GEMINI_API_KEY=your_actual_gemini_key
HUGGINGFACE_TOKEN=your_hf_token
HF_USE_API=true
```

**Get API Keys:**
- **Gemini**: https://aistudio.google.com/app/apikey
- **HuggingFace**: https://huggingface.co/settings/tokens (free account)

### 3. Generate Reviews

```bash
python run.py
```

Or with a custom config:

```bash
python run.py --config configs/pmtool.yaml
```

Reviews will be generated in `runs/run_[timestamp]/`

## 📁 Project Structure

```
Synthetic_Review_Generator/
├── src/
│   ├── adapters/           # LLM provider adapters
│   │   ├── base_adapter.py    # Abstract base interface
│   │   ├── gemini_adapter.py  # Google Gemini implementation
│   │   └── mistral_adapter.py # Mistral (HuggingFace) implementation
│   ├── validators/         # Quality validation modules
│   │   ├── diversity.py       # Vocabulary & semantic diversity checks
│   │   ├── domain.py          # Domain-specific feature validation
│   │   └── sentiment.py       # Rating-sentiment alignment
│   ├── cli/
│   │   └── main.py           # CLI entry point
│   ├── orchestrator.py     # Core generation pipeline
│   └── models.py           # (Reserved for future data models)
├── configs/
│   └── pmtool.yaml         # Configuration for PM SaaS reviews
├── datasets/               # Generated datasets storage
├── docs/                   # Documentation & assignment PDFs
├── reports/                # Quality reports (auto-generated)
├── .env.example            # Environment variables template
├── .gitignore
├── requirements.txt
└── README.md
```

## 🏗️ Architecture

### Core Components

**1. Orchestrator (`orchestrator.py`)**
- Coordinates the entire generation pipeline
- Manages adapter selection (weighted random)
- Handles persona sampling and rating distribution
- Executes validation and regeneration logic
- Saves results to JSONL and generates quality reports

**2. Adapters (`src/adapters/`)**
- **BaseAdapter**: Abstract interface defining `generate()` method
- **GeminiAdapter**: Google Gemini Flash 1.5 integration
- **MistralAdapter**: Mistral-7B via HuggingFace Inference API
- All adapters return `GenerationResult` with text + metadata (latency, tokens, provider)

**3. Validators (`src/validators/`)**
- **Diversity**: TF-IDF semantic similarity + vocabulary overlap
- **Sentiment**: VADER sentiment score vs. rating alignment
- **Domain**: Feature lexicon matching + blacklist filtering

### Generation Flow

```
1. Load YAML config (personas, rating distribution, models, thresholds)
2. For each sample (up to sample_count):
   a. Choose adapter (weighted: Gemini 60%, Mistral 40%)
   b. Sample persona and rating
   c. Build prompt
   d. Generate review
   e. Validate (domain + sentiment)
   f. If validation fails → regenerate (up to max_attempts)
   g. Accept or reject sample
3. Save accepted samples to datasets/
4. Generate quality report in reports/
```

## 🎯 Configuration

Edit `configs/pmtool.yaml` to customize:

```yaml
domain: "Project Management SaaS"
sample_count: 100

models:
  - name: "gemini"
    weight: 0.6
  - name: "mistral"
    weight: 0.4

personas:
  - name: "marketing_manager"
    background: "Works in a small agency"
    voice_style: "friendly"
  - name: "devops_engineer"
    background: "Linux sysadmin"
    voice_style: "concise"

rating_distribution: {1: 10, 2: 10, 3: 20, 4: 40, 5: 20}

feature_lexicon:
  features: ["timeline", "board view", "integration", "automation"]
  blacklist: ["quantum compiler", "teleportation"]

quality_thresholds:
  domain_score_min: 0.05
  sentiment_tolerance: 0.6

regeneration:
  max_attempts: 2
```

## 📊 Output Format

Generated reviews are saved as JSONL in `datasets/`:

```json
{
  "title": "Great for Team Collaboration",
  "body": "We've been using this PM tool for 3 months...",
  "tags": ["kanban", "integration"],
  "rating": 4,
  "persona": "marketing_manager",
  "model": "gemini-1.5-flash-latest",
  "date": "2024-08-15",
  "quality": {
    "passed": true,
    "diversity": {"semantic_similarity": 0.43},
    "sentiment": {"score": 0.68, "expected_range": [0.2, 1.0]},
    "domain": {"score": 0.85, "features_found": ["kanban", "integration"]}
  }
}
```

## 🛡️ Quality Guardrails

### Diversity Validator
- **Metric**: TF-IDF cosine similarity between reviews
- **Threshold**: < 0.92 (configurable)
- **Purpose**: Prevent repetitive or overly similar outputs

### Sentiment Validator
- **Metric**: VADER sentiment score
- **Logic**: Rating 1-2 → negative, 3 → neutral, 4-5 → positive
- **Tolerance**: 0.6 (allows some variance)
- **Purpose**: Ensure rating matches review sentiment

### Domain Validator
- **Metric**: Feature mention count / total features
- **Blacklist**: Reject reviews mentioning impossible features
- **Min Score**: 0.05 (at least 1 feature mentioned)
- **Purpose**: Keep reviews realistic and domain-relevant

Failed validations trigger automatic regeneration (up to `max_attempts`).

## 🤖 Supported Models

### Google Gemini Flash Latest
- **Pros**: Fast, reliable, excellent JSON adherence, **completely FREE**
- **Cost**: $0 (free tier covers all usage for this project)
- **Use Case**: Primary generation model (60% weight)
- **Model**: `gemini-flash-latest` (no version pinning needed)

### Mistral-7B (HuggingFace API)
- **Pros**: Free, open-source, diverse writing style
- **API**: HuggingFace Inference API (requires token)
- **Use Case**: Style diversity (40% weight)

**Note**: You can run with Gemini only if no HuggingFace token is provided.

## 📈 Quality Reports

After generation, check `reports/quality_report.md` for:
- Total samples generated vs. accepted
- Rejection breakdown by validator
- Average latency per model

**Import Errors?**
- Make sure you're running from the project root
- All packages now have `__init__.py` files

**API Errors?**
- Verify `.env` has correct API keys
- Check API quotas/rate limits
- HuggingFace API requires free account registration

**Low Quality Scores?**
- Reduce `domain_score_min` threshold
- Increase `sentiment_tolerance`
- Expand `feature_lexicon` for your domain

**Empty datasets/ folder?**
- Normal before first run
- Generated files appear after running orchestrator successfully

## 📚 Documentation

- **System Design**: `docs/System_Design.md` (architecture deep-dive)
- **Assignment Brief**: `docs/AI_Engineer_Assignment.pdf`

## 🧪 Development

### Adding a New Adapter

1. Create `src/adapters/my_adapter.py`
2. Inherit from `ModelAdapter`
3. Implement `generate()` method
4. Return `GenerationResult` with metadata
5. Update orchestrator config to include new model

Example:
```python
from .base_adapter import ModelAdapter, GenerationResult, GenerationMetadata

class MyAdapter(ModelAdapter):
    def generate(self, prompt, max_tokens=256, temperature=1.0, **kwargs):
        # Your implementation
        return GenerationResult(text="...", metadata=GenerationMetadata(...))
```

### Adding a New Validator

1. Create `src/validators/my_validator.py`
2. Implement validation logic
3. Integrate in `orchestrator.py`'s validation pipeline
