# 🔬 Synthetic Review Generator

A production-ready system for generating high-quality synthetic product reviews using **multiple LLM providers** with **built-in quality validation**.

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

**Entry Point:** `run.py`

```bash
python run.py --config configs/pmtool.yaml
```

All outputs (reviews, reports, datasets) are saved to `datasets/run_[timestamp]/`

## 📁 Project Structure

```
Synthetic_Review_Generator/
├── run.py                   # 🚀 Main entry point
├── src/
│   ├── adapters/            # 🤖 LLM provider adapters
│   │   ├── base_adapter.py     # Abstract base class for all adapters
│   │   ├── gemini_adapter.py   # Google Gemini Flash 1.5 (60% weight)
│   │   ├── flan_t5_adapter.py  # Google Flan-T5 (20% weight)
│   │   └── bloomz_adapter.py   # BLOOMZ (20% weight)
│   ├── validators/          # ✅ Quality validation modules
│   │   ├── domain.py           # Domain-specific feature + blacklist validation
│   │   ├── sentiment.py        # Rating-sentiment alignment (VADER)
│   │   └── diversity.py        # TF-IDF similarity + vocabulary overlap
│   ├── orchestrator.py      # 🎛️ Core generation pipeline coordinator
│   └── models.py            # (Reserved for future data models)
├── configs/
│   └── pmtool.yaml          # ⚙️ Domain configuration (PM SaaS reviews)
├── datasets/                # 💾 Generated datasets + reports storage
│   └── run_[timestamp]/
│       ├── reviews.jsonl       # Accepted reviews with metadata
│       ├── summary.json        # Generation statistics
│       └── quality_report.md   # Human-readable quality report
├── docs/                    # 📚 Documentation
│   └── System_Design.md        # Complete architecture documentation
├── .env.example             # Environment variables template
├── .gitignore
├── requirements.txt
└── README.md
```

## 🏗️ Architecture

### Entry Point: `run.py`

The **`run.py`** script is the main entry point that:
1. Loads YAML configuration
2. Initializes all model adapters
3. Creates the orchestrator
4. Starts the generation process
5. Saves everything to `datasets/run_[timestamp]/`

### Core Components

**1. Model Adapters (`src/adapters/`)** - Inherit from `BaseAdapter`

All adapters inherit from the **`BaseAdapter`** abstract class for consistency:

- **`GeminiAdapter`** - Google Gemini Flash 1.5 (60% weight)
  - Fast, reliable, excellent JSON adherence
  - Free tier API
  
- **`FlanT5Adapter`** - Google Flan-T5 via HuggingFace (20% weight)
  - Instruction-tuned T5 model
  - Adds writing style diversity
  
- **`BLOOMZAdapter`** - BLOOMZ via HuggingFace (20% weight)
  - Multilingual capability
  - Different generation patterns

**Why multiple models?**
- ✅ Diversity in writing styles
- ✅ Benchmark and compare performance
- ✅ Fallback if one model fails

**2. Validation System (`src/validators/`)** - 3 Independent Scripts

- **`domain.py`** - Domain-specific validation
  - Checks for **feature mentions** from lexicon (e.g., "timeline", "automation")
  - Detects **blacklisted terms** (e.g., "quantum", "magic")
  - Rejects reviews mentioning impossible features
  
- **`sentiment.py`** - Rating-sentiment alignment
  - Uses **VADER sentiment analysis**
  - Ensures rating matches review sentiment (1-2 = negative, 3 = neutral, 4-5 = positive)
  - Prevents mismatches (e.g., rating 1 with positive text)
  
- **`diversity.py`** - Similarity detection
  - Computes **TF-IDF vectors** for each review
  - Calculates **cosine similarity** against existing reviews
  - Rejects if similarity > 0.92 (too repetitive)

**3. Orchestrator (`orchestrator.py`)** - Central Control System

- Manages the generation loop
- Handles weighted adapter selection
- Coordinates all three validators
- Implements retry logic (regenerates failures up to max_attempts)
- Tracks metrics (latency, acceptance rates, model performance)
- Saves outputs to `datasets/`

### Generation Flow

```
run.py starts
  ↓
Load config from configs/pmtool.yaml
  ↓
Orchestrator initialization
  ↓
For each sample (up to sample_count):
  1. Select model by weight (Gemini 60%, Flan-T5 20%, BLOOMZ 20%)
  2. Sample persona + rating from distribution
  3. Build prompt
  4. Call adapter.generate() (inherits from BaseAdapter)
  5. Validate review:
     → domain.py (feature + blacklist check)
     → sentiment.py (rating alignment)
     → diversity.py (similarity check)
  6. If all pass → accept and save
  7. If any fail → regenerate (up to 2 retries)
  ↓
Save to datasets/run_[timestamp]/
  - reviews.jsonl       (all accepted reviews)
  - summary.json        (statistics)
  - quality_report.md   (human-readable report)
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

## 🛡️ Quality Guardrails - Three Validation Scripts

### 1️⃣ Domain Validator (`validators/domain.py`)
**Purpose:** Ensure reviews mention realistic features and avoid impossible claims

- **Feature Matching**: Checks for mentions from `feature_lexicon` (e.g., "kanban", "sprint planning", "timeline")
- **Blacklist Detection**: Rejects reviews with impossible terms (e.g., "quantum", "teleportation", "magic")
- **Scoring**: `features_found / total_features`
- **Threshold**: Min score 0.05 (at least 1 feature mentioned)

**Example Rejection:**
```json
{"title": "Amazing Quantum Integration", "body": "Uses quantum computing..."}
```
❌ Rejected - Contains blacklisted term "quantum"

---

### 2️⃣ Sentiment Validator (`validators/sentiment.py`)
**Purpose:** Ensure sentiment matches the rating

- **Method**: VADER sentiment analysis (-1 to +1)
- **Logic**: 
  - Rating 1-2 → Negative sentiment expected
  - Rating 3 → Neutral sentiment expected  
  - Rating 4-5 → Positive sentiment expected
- **Tolerance**: 0.6 (allows natural variance)

**Example Rejection:**
```json
{"rating": 1, "body": "Absolutely love this tool! It's perfect!"}
```
❌ Rejected - Positive sentiment but rating is 1/5

---

### 3️⃣ Diversity Validator (`validators/diversity.py`)
**Purpose:** Prevent repetitive or overly similar reviews

- **Method**: TF-IDF vectors + cosine similarity
- **Checks**: Compares each review against all existing reviews
- **Threshold**: < 0.92 (configurable)
- **Purpose**: Ensure natural variation in word choice, phrasing, and style

**Why it matters:** Real reviews have natural diversity. If all reviews sound identical, the dataset is unrealistic.

---

**Failed validations trigger automatic regeneration** (up to `max_attempts` = 2)

## 🎯 Model Selection Rationale

### Why Instruction-Tuned Local Models?

During development, I explored different approaches for model integration:

**🔄 Approaches Tried:**
1. **API-based models** (e.g., OpenAI, Anthropic)
   - ❌ **Issue**: Quota limits exhausted quickly with 500+ samples
   - ❌ **Issue**: Rate limiting caused generation delays
   - ❌ **Issue**: Requires paid API keys for production use

2. **Base URL endpoints** (custom model endpoints)
   - ❌ **Issue**: Base URLs can change or become unavailable
   - ❌ **Issue**: Unreliable for production deployment
   - ❌ **Issue**: Still subject to rate limits

**✅ Final Decision: Instruction-Tuned Models**

I chose **Flan-T5** and **BLOOMZ** as the primary models alongside Gemini because:

- ✅ **Instruction-tuned**: Pre-trained to follow prompts effectively
- ✅ **Local deployment**: Can run locally without API dependencies
- ✅ **No rate limits**: Generate unlimited reviews without quotas
- ✅ **Free**: No API costs for production
- ✅ **HuggingFace API option**: Can use free inference API during development
- ✅ **Reproducible**: Consistent results without API changes

**Model Strategy:**
- **Flan-T5 (50%)**: Instruction-tuned, can run locally if needed
- **BLOOMZ (50%)**: Instruction-tuned, adds diversity, local-ready

This hybrid approach provides **reliability** (local models) with **speed** (Gemini API) while avoiding quota and rate limit issues.

---

## 🤖 Model Adapters - Base Inheritance Pattern

### Base Adapter (`BaseAdapter`)

All adapters inherit from the abstract `BaseAdapter` class:

```python
class BaseAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        """Generate text from the model"""
        pass
```

### Implemented Adapters

#### 1. **Flan-T5 Adapter** (`flan_t5_adapter.py`)
- **Model**: Google Flan-T5 (via HuggingFace)
- **Weight**: 50% (configurable)
- **Pros**: Instruction-tuned, adds writing diversity
- **Cost**: Free HuggingFace Inference API
- **Use Case**: Style variation

#### 2. **BLOOMZ Adapter** (`bloomz_adapter.py`)
- **Model**: BLOOMZ (via HuggingFace)
- **Weight**: 50% (configurable)
- **Pros**: Multilingual, different generation patterns
- **Cost**: Free HuggingFace Inference API
- **Use Case**: Additional diversity

### Model Selection

Models are selected using **weighted random sampling** based on configuration:

```yaml
models:
  - name: "flan-t5"
    weight: 0.5    # 50% of reviews
  - name: "bloomz"
    weight: 0.5    # 50% of reviews
```

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
