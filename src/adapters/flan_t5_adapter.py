"""
Flan-T5 Adapter (local transformers)

This adapter uses the Hugging Face `transformers` text2text-generation pipeline
to run Google's Flan-T5 model locally.

Requirements:
    pip install transformers torch accelerate
"""

import time
from typing import Optional
from dotenv import load_dotenv

# transformer imports
try:
    from transformers import pipeline
except Exception:
    pipeline = None

from .base_adapter import ModelAdapter, GenerationResult, GenerationMetadata

load_dotenv()

class FlanT5Adapter(ModelAdapter):
    """
    Local Flan-T5 adapter using transformers pipeline.
    
    Args:
        model: local model id or path. Default: "google/flan-t5-small"
        device: -1 for CPU, 0..N for GPU device id
    """

    def __init__(self, model: str = "google/flan-t5-small", device: Optional[int] = None):
        self.model = model
        self.provider = "local-flan-t5"
        self.device = device if device is not None else -1

        if pipeline is None:
            raise RuntimeError(
                "transformers not available. Install with: pip install transformers torch accelerate"
            )

        try:
            # Flan-T5 is a seq2seq model, so we use text2text-generation
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                device=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{self.model}' locally: {exc}"
            ) from exc

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 1.0, **kwargs) -> GenerationResult:
        """
        Generate text locally using Flan-T5.
        """
        # Flan-T5/T5 generation kwargs
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "temperature": float(temperature),
            "do_sample": True if temperature > 0.0 else False,
        }

        start = time.time()
        outputs = self.generator(prompt, **gen_kwargs)
        latency_ms = (time.time() - start) * 1000.0

        # Output format for text2text-generation is usually [{"generated_text": "..."}]
        text = ""
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict) and "generated_text" in first:
                text = first["generated_text"]
            else:
                text = str(first)
        else:
            text = str(outputs)

        metadata = GenerationMetadata(
            provider=self.provider,
            model=self.model,
            latency_ms=latency_ms,
        )

        return GenerationResult(text=text, metadata=metadata)
