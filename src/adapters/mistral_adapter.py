"""
Mistral Adapter (local transformers)

This adapter uses the Hugging Face `transformers` text-generation pipeline
to run a model locally (no HTTP calls). It is intentionally minimal.

Behavior:
- By default, attempts to load the specified model locally via `transformers`.
- If loading fails, it raises a clear error explaining options:
    * install the model or
    * set HF_USE_API=true to use the hosted API (not implemented here)
- Returns GenerationResult (text + metadata) matching base adapter.

Requirements:
    pip install transformers torch accelerate
(Or an environment with a suitable model already available.)
"""

import os
import time
from typing import Optional

from dotenv import load_dotenv

# transformer imports (may raise if not installed)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except Exception as _err:
    # We delay raising until adapter is initialized so run.py can show clearer message
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

from .base_adapter import ModelAdapter, GenerationResult, GenerationMetadata

load_dotenv()


class MistralAdapter(ModelAdapter):
    """
    Local Mistral adapter using transformers pipeline.

    Args:
      model: local model id or path (string). Example: "mistralai/mistral-7b-instruct"
      device: -1 for CPU, 0..N for GPU device id (int or None to auto-select)
    """

    def __init__(self, model: str = "mistralai/mistral-7b-instruct", device: Optional[int] = None):
        self.model = model
        self.provider = "local-transformers"

        # Determine device: default to GPU 0 if available, else CPU
        self.device = device if device is not None else -1

        # Quick check: transformers must be installed
        if pipeline is None:
            raise RuntimeError(
                "transformers (and optionally torch) not available. "
                "Install with: pip install transformers torch accelerate"
            )

        # Try to create a local generation pipeline.
        # This will attempt to download the model if not present (requires internet)
        # and may fail if model is too large for your hardware.
        try:
            # Use text-generation pipeline which handles tokenizer + model under the hood.
            # device = -1 indicates CPU, 0 indicates CUDA:0 when available.
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                device=self.device,
                trust_remote_code=True,  # some community models require this
            )
        except Exception as exc:
            # Provide explicit guidance rather than a noisy stack trace.
            raise RuntimeError(
                f"Failed to load model '{self.model}' locally: {exc}\n\n"
                "Options:\n"
                "  * Ensure the model id/path is correct and accessible.\n"
                "  * If the model is large, ensure you have enough GPU RAM or use a smaller model.\n"
                "  * Install transformers & torch: pip install transformers torch accelerate\n"
                "  * If you prefer using the Hugging Face Inference API instead, set HF_USE_API=true\n"
            ) from exc

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 1.0, **kwargs) -> GenerationResult:
        """
        Generate text locally using the transformers pipeline.

        Args:
            prompt: input prompt string
            max_tokens: maximum new tokens to generate
            temperature: sampling temperature
        Returns:
            GenerationResult(text, metadata)
        """
        # Prepare generation kwargs compatible with transformers pipeline
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "temperature": float(temperature),
            # do_sample True to use temperature; can adjust top_p/top_k if desired
            "do_sample": True if temperature > 0.0 else False,
        }

        start = time.time()
        outputs = self.generator(prompt, **gen_kwargs)
        latency_ms = (time.time() - start) * 1000.0

        # The transformers pipeline returns a list of dicts with 'generated_text'
        # Example: [{"generated_text": " ... "}]
        text = ""
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict) and "generated_text" in first:
                text = first["generated_text"]
            elif isinstance(first, dict) and "text" in first:
                text = first["text"]
            else:
                # Fallback: stringify first output
                text = str(first)
        else:
            # Fallback: stringify outputs
            text = str(outputs)

        # Build minimal metadata
        metadata = GenerationMetadata(
            provider=self.provider,
            model=self.model,
            tokens_in=None,
            tokens_out=None,
            latency_ms=latency_ms,
        )

        return GenerationResult(text=text, metadata=metadata)
