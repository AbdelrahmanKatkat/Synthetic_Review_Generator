"""
Orchestrator for Synthetic Review Generation

Coordinates generation, validation, regeneration, and persistence of synthetic reviews.
Loads YAML config, picks personas/ratings, builds prompts, calls adapters, validates
samples using domain & sentiment validators, and produces quality reports.
"""

from __future__ import annotations
import yaml
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from adapters.base_adapter import GenerationResult
from validators.domain import domain_validator
from validators.sentiment import sentiment_vs_rating_flag, batch_sentiment_metrics
from validators.diversity import diversity_metrics


class Orchestrator:
    """
    Orchestrates review generation with quality guardrails.
    
    Usage:
      orch = Orchestrator(config_path="configs/pmtool.yaml", adapters={"gemini": adapter_instance})
      orch.run()
    """

    def __init__(self, config_path: str, adapters: Dict[str, Any]):
        """
        Args:
            config_path: path to YAML configuration file
            adapters: dict mapping adapter_name -> adapter_instance
        """
        self.config_path = Path(config_path)
        self.adapters = adapters
        self.config = self._load_config(self.config_path)

        # Prepare output folders - changed from runs to datasets
        self.datasets_dir = Path("datasets")
        self.datasets_dir.mkdir(exist_ok=True)
        self.run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.dataset_dir = self.datasets_dir / f"run_{self.run_id}"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Internal storage
        self.accepted_samples: List[Dict[str, Any]] = []
        self.rejected_samples: List[Dict[str, Any]] = []
        
        # Per-model stats
        self.model_stats: Dict[str, Dict[str, Any]] = {
            name: {"requested": 0, "accepted": 0, "total_latency_ms": 0.0} for name in adapters.keys()
        }

    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load YAML config with default values."""
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        cfg.setdefault("sample_count", 100)
        cfg.setdefault("personas", [])
        cfg.setdefault("rating_distribution", {1: 10, 2: 10, 3: 20, 4: 40, 5: 20})
        cfg.setdefault("regeneration", {"max_attempts": 2})
        cfg.setdefault("quality_thresholds", {"domain_score_min": 0.05, "sentiment_tolerance": 0.6})
        return cfg

    def _choose_adapter(self) -> Tuple[str, Any]:
        """Choose an adapter randomly weighted by config models weights."""
        models_cfg = self.config.get("models", [])
        
        # Filter to only available adapters
        available_models = [m for m in models_cfg if m["name"] in self.adapters]
        
        if available_models:
            names = [m["name"] for m in available_models]
            weights = [m.get("weight", 1.0) for m in available_models]
            chosen = random.choices(names, weights=weights, k=1)[0]
            return chosen, self.adapters[chosen]
        
        # Fallback: pick uniformly
        if not self.adapters:
            raise RuntimeError("No adapters available to choose from.")

        k = random.choice(list(self.adapters.keys()))
        return k, self.adapters[k]

    def _sample_persona_and_rating(self) -> Tuple[Optional[Dict[str, Any]], int]:
        """Sample persona and rating based on config."""
        personas = self.config.get("personas", [])
        rating_distribution = self.config.get("rating_distribution", {})
        
        # Sample rating
        if rating_distribution:
            choices = list(rating_distribution.keys())
            weights = [rating_distribution[k] for k in choices]
            rating = int(random.choices(choices, weights=weights, k=1)[0])
        else:
            rating = random.randint(1, 5)

        persona = random.choice(personas) if personas else None
        return persona, rating

    def _build_prompt(self, persona: Optional[Dict[str, Any]], rating: int, config: Dict[str, Any]) -> str:
        """Build a prompt for adapters."""
        domain = config.get("domain", "Product")
        persona_desc = ""
        if persona:
            persona_desc = f"Persona: {persona.get('name','user')}. Role: {persona.get('background','')}. Voice: {persona.get('voice_style','')}"
        
        prompt = (
            f"Write a {rating}-star review for a {domain}.\n"
            f"{persona_desc}\n"
            "Include 1-2 short sentences describing a product feature and one improvement.\n"
            "Output only the review body (no JSON or metadata)."
        )
        return prompt

    def generate_one(self, adapter_name: str, adapter: Any, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        """Call adapter.generate and capture latency. Retry on rate limits."""
        self.model_stats[adapter_name]["requested"] += 1

        # Retry logic for rate limits
        max_retries = 3
        base_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                print(f"  [DEBUG] Generating with {adapter_name} (attempt {attempt+1})...")
                gen_result: GenerationResult = adapter.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                print("  [DEBUG] Generation complete.")
                
                latency = getattr(gen_result.metadata, "latency_ms", None)
                if latency is not None:
                    self.model_stats[adapter_name]["total_latency_ms"] += float(latency)

                return gen_result
            
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit error (429 or quota exceeded)
                if "429" in error_str or "rate" in error_str or "quota" in error_str or "resource" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"  Rate limit hit, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                # Re-raise if not rate limit or max retries exceeded
                raise

    def validate_and_maybe_regenerate(
        self,
        text: str,
        rating: int,
        features: List[str],
        blacklist: List[str],
        tolerance: float,
        max_attempts: int,
        adapter_name: str,
        adapter: Any,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Validate generated text with domain and sentiment validators.
        If validation fails, regenerate up to max_attempts.
        
        Returns:
            (accepted_sample_dict, rejection_reason)
        """
        attempts = 1  # Start at 1 since we already have initial text
        
        # FIXED: Changed loop condition to prevent infinite loop
        while attempts <= max_attempts:
            print(f"  [DEBUG] Validation attempt {attempts}/{max_attempts}...")
            # Domain check
            dom = domain_validator(
                text, 
                features=features, 
                blacklist=blacklist, 
                feature_threshold=self.config["quality_thresholds"].get("domain_score_min", 0.05)
            )
            
            # Sentiment vs rating check
            sent = sentiment_vs_rating_flag(text, rating, tolerance=tolerance)

            # Build quality profile
            quality = {
                "domain_score": dom["domain_score"],
                "domain_passed": dom["passed"],
                "domain_reasons": dom["reasons"],
                "sentiment_score": sent["sentiment_score"],
                "sentiment_label": sent["sentiment_label"],
                "rating_mismatch": sent["mismatch"],
                "sentiment_distance": sent["distance"],
                "attempt": attempts,
            }

            # Check if sample passes validation
            if dom["passed"] and not sent["mismatch"]:
                print("  [DEBUG] Sample ACCEPTED.")
                # Accepted!
                sample = {
                    "id": f"{self.run_id}_{len(self.accepted_samples)+1:05d}",
                    "rating": rating,
                    "text": text,
                    "adapter": adapter_name,
                    "model": getattr(adapter, "model", None),
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "quality": quality,
                }
                self.model_stats[adapter_name]["accepted"] += 1
                return sample, None

            # Validation failed - check if we can retry
            if attempts >= max_attempts:
                # Exhausted attempts
                return None, f"rejected_after_{attempts}_attempts"
            
            # Regenerate with possibly different adapter
            print(f"  [DEBUG] Sample REJECTED. Reasons: {dom['reasons']} Mismatch: {sent['mismatch']}. Regenerating...")
            attempts += 1
            adapter_name, adapter = self._choose_adapter()
            print(f"  [DEBUG] Switching to adapter: {adapter_name}")
            gen_res = self.generate_one(adapter_name, adapter, prompt, max_tokens, temperature)
            text = gen_res.text

        # Should not reach here
        return None, "unknown_error"

    def _write_jsonl(self, samples: List[Dict[str, Any]]):
        """Write accepted samples to JSONL file."""
        out_path = self.dataset_dir / "dataset.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _write_report(self, samples: List[Dict[str, Any]]):
        """Write markdown report summarizing quality and model stats."""
        report_path = self.dataset_dir / "report.md"
        total = len(samples)
        
        # Calculate average latencies
        avg_latency = {}
        for name, s in self.model_stats.items():
            req = s["requested"] or 1
            avg = (s["total_latency_ms"] / req) if req else None
            avg_latency[name] = avg

        # Diversity metrics (corpus-level)
        texts = [s["text"] for s in samples]
        div = diversity_metrics(texts) if texts else {"vocab_overlap": 0.0, "semantic_similarity": 0.0}
        
        # Sentiment corpus metrics
        ratings = [s["rating"] for s in samples]
        sent = batch_sentiment_metrics(texts, ratings)

        lines = [
            f"# Run report: {self.run_id}",
            f"Date (UTC): {datetime.utcnow().isoformat()}Z",
            "",
            f"Total accepted samples: {total}",
            "",
            "## Model stats",
        ]
        
        for name, s in self.model_stats.items():
            lines.append(f"- {name}: requested={s['requested']}, accepted={s['accepted']}, avg_latency_ms={avg_latency.get(name)}")
        
        lines.extend([
            "",
            "## Diversity",
            f"- vocab_overlap: {div['vocab_overlap']:.4f}",
            f"- semantic_similarity: {div['semantic_similarity']:.4f}",
            "",
            "## Sentiment",
            f"- avg_sentiment: {sent['avg_sentiment']:.4f}",
            f"- pct_positive: {sent['pct_positive']:.2%}",
            f"- pct_neutral: {sent['pct_neutral']:.2%}",
            f"- pct_negative: {sent['pct_negative']:.2%}",
            f"- rating_mismatch_rate: {sent['rating_mismatch_rate']}",
            "",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def run(self):
        """
        Execute the generation pipeline:
        - Loop until sample_count accepted
        - Pick adapter, persona, prompt, generate, validate, maybe regenerate
        - Save accepted samples and produce report
        """
        cfg = self.config
        count_target = int(cfg.get("sample_count", 100))
        max_attempts = cfg.get("regeneration", {}).get("max_attempts", 2)
        tolerance = cfg.get("quality_thresholds", {}).get("sentiment_tolerance", 0.6)

        features = cfg.get("feature_lexicon", {}).get("features", [])
        blacklist = cfg.get("feature_lexicon", {}).get("blacklist", [])

        max_rounds = count_target * 10  # Safety cap to avoid infinite loops
        rounds = 0
        
        print(f"Target: {count_target} samples")
        print(f"Max attempts per sample: {max_attempts}")

        while len(self.accepted_samples) < count_target and rounds < max_rounds:
            rounds += 1
            print(f"\n[DEBUG] Starting round {rounds}. Accepted so far: {len(self.accepted_samples)}")

            # Pick adapter and persona/rating
            adapter_name, adapter = self._choose_adapter()
            persona, rating = self._sample_persona_and_rating()
            prompt = self._build_prompt(persona, rating, cfg)
            max_tokens = cfg.get("review_characteristics", {}).get("target_length_tokens", 120)
            temperature = cfg.get("review_characteristics", {}).get("temperature", 1.0)

            # Generate first attempt
            gen_res = self.generate_one(adapter_name, adapter, prompt, max_tokens, temperature)
            text = gen_res.text
            print(f"  [DEBUG] Generated text: {text!r}")

            # Validate and possibly regenerate
            accepted, reason = self.validate_and_maybe_regenerate(
                text=text,
                rating=rating,
                features=features,
                blacklist=blacklist,
                tolerance=tolerance,
                max_attempts=max_attempts,
                adapter_name=adapter_name,
                adapter=adapter,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if accepted:
                self.accepted_samples.append(accepted)
                # Progress logging every 10 samples
                if len(self.accepted_samples) % 10 == 0:
                    acceptance_rate = len(self.accepted_samples) / rounds * 100
                    print(f"Progress: {len(self.accepted_samples)}/{count_target} accepted (acceptance rate: {acceptance_rate:.1f}%)")
            else:
                self.rejected_samples.append({
                    "attempt_text": text,
                    "reason": reason,
                    "adapter": adapter_name,
                    "model": getattr(adapter, "model", None),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
            
            # Delay to respect free tier rate limits (Gemini free tier: 15 req/min = ~4s/req)
            print("  [DEBUG] Sleeping for 5s...")
            time.sleep(5.0)

        # Write outputs
        self._write_jsonl(self.accepted_samples)
        self._write_report(self.accepted_samples)

        # Save summary JSON
        summary = {
            "run_id": self.run_id,
            "accepted": len(self.accepted_samples),
            "rejected": len(self.rejected_samples),
            "model_stats": self.model_stats,
            "dataset_dir": str(self.dataset_dir),
        }
        with open(self.dataset_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nRun complete: accepted={len(self.accepted_samples)} rejected={len(self.rejected_samples)}")
        print(f"Artifacts saved in: {self.dataset_dir}")
