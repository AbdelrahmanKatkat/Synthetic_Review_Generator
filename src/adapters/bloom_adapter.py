"""
BLOOMZ Adapter (local transformers)

This adapter uses Hugging Face `transformers` to run BLOOMZ-560M locally.
BLOOMZ is a cross-lingual language model fine-tuned on xP3.

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

class BloomAdapter(ModelAdapter):
    """
    Local BLOOMZ adapter using transformers pipeline.
    
    Args:
        model: local model id or path. Default: "bigscience/bloomz-560m"
        device: -1 for CPU, 0..N for GPU device id
    """

    def __init__(self, model: str = "bigscience/bloomz-560m", device: Optional[int] = None):
        self.model = model
        self.provider = "local-bloom"
        self.device = device if device is not None else -1

        if pipeline is None:
            raise RuntimeError(
                "transformers not available. Install with: pip install transformers torch accelerate"
            )

        try:
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                device=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{self.model}' locally: {exc}"
            ) from exc

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 1.0, **kwargs) -> GenerationResult:
        """
        Generate text locally using BLOOMZ.
        """
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "temperature": float(temperature),
            "do_sample": True if temperature > 0.0 else False,
        }

        start = time.time()
        outputs = self.generator(prompt, **gen_kwargs)
        latency_ms = (time.time() - start) * 1000.0

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
