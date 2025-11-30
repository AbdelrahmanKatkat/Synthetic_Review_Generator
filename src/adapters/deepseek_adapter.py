"""
DeepSeek Adapter (local transformers)

This adapter uses the Hugging Face `transformers` library to run DeepSeek 
text generation models locally (no HTTP calls).

Behavior:
- Loads DeepSeek-Coder or DeepSeek-Chat locally via transformers
- Uses text-generation pipeline for efficient inference
- Returns GenerationResult matching base adapter interface

Requirements:
    pip install transformers torch accelerate
"""

import os
import time
from typing import Optional

from dotenv import load_dotenv

# Transformer imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except Exception as _err:
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

from .base_adapter import ModelAdapter, GenerationResult, GenerationMetadata

load_dotenv()


class DeepSeekAdapter(ModelAdapter):
    """
    Local DeepSeek adapter using transformers pipeline.
    
    Args:
      model: DeepSeek model id. Options:
        - "deepseek-ai/deepseek-coder-6.7b-instruct" (6.7B, best for code/text)
        - "deepseek-ai/deepseek-llm-7b-chat" (7B, chat optimized)
      device: -1 for CPU, 0+ for GPU device id (None = auto-select)
    """

    def __init__(
        self, 
        model: str = "deepseek-ai/deepseek-coder-6.7b-instruct",
        device: Optional[int] = None
    ):
        self.model = model
        self.provider = "DeepSeek"
        
        # Determine device: default to GPU 0 if available, else CPU
        self.device = device if device is not None else 0
        
        # Check transformers availability
        if pipeline is None:
            raise RuntimeError(
                "transformers not available. "
                "Install with: pip install transformers torch accelerate"
            )
        
        # Create text-generation pipeline
        try:
            print(f"Loading DeepSeek model: {self.model}...")
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                device=self.device,
                trust_remote_code=True,
                torch_dtype="auto",  # Automatically choose best dtype
            )
            print(f"DeepSeek model loaded successfully on device {self.device}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load DeepSeek model '{self.model}': {exc}\n\n"
                "Options:\n"
                "  * Ensure model id is correct: deepseek-ai/deepseek-coder-6.7b-instruct\n"
                "  * Ensure sufficient GPU/CPU RAM available\n"
                "  * Install dependencies: pip install transformers torch accelerate\n"
                "  * For smaller model, try: deepseek-ai/deepseek-coder-1.3b-instruct\n"
            ) from exc

    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 256, 
        temperature: float = 1.0, 
        **kwargs
    ) -> GenerationResult:
        """
        Generate text locally using DeepSeek model.
        
        Args:
            prompt: input prompt string
            max_tokens: maximum new tokens to generate
            temperature: sampling temperature
        Returns:
            GenerationResult(text, metadata)
        """
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "temperature": float(temperature),
            "do_sample": True if temperature > 0.0 else False,
            "top_p": 0.95,
            "top_k": 50,
        }
        
        start = time.time()
        outputs = self.generator(prompt, **gen_kwargs)
        latency_ms = (time.time() - start) * 1000.0
        
        # Extract generated text (pipeline returns list of dicts)
        text = ""
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict) and "generated_text" in first:
                text = first["generated_text"]
                # Remove the prompt from output (pipeline includes it)
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()
            elif isinstance(first, dict) and "text" in first:
                text = first["text"]
            else:
                text = str(first)
        else:
            text = str(outputs)
        
        # Build metadata
        metadata = GenerationMetadata(
            provider=self.provider,
            model=self.model,
            tokens_in=None,
            tokens_out=None,
            latency_ms=latency_ms,
        )
        
        return GenerationResult(text=text, metadata=metadata)
