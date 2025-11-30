"""
Llama Adapter (local transformers)

This adapter uses the Hugging Face `transformers` text-generation pipeline
to run Llama 3.2 models locally (no HTTP calls).

Behavior:
- Loads Llama 3.2-3B-Instruct locally via transformers
- Uses chat template for proper instruction formatting
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


class LlamaAdapter(ModelAdapter):
    """
    Local Llama adapter using transformers pipeline.
    
    Args:
      model: Llama model id. Options:
        - "meta-llama/Llama-3.2-3B-Instruct" (3B, lightweight)
        - "meta-llama/Llama-3.2-1B-Instruct" (1B, very lightweight)
      device: -1 for CPU, 0+ for GPU device id (None = auto-select)
    """

    def __init__(
        self, 
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: Optional[int] = None
    ):
        self.model = model
        self.provider = "Llama"
        
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
            print(f"Loading Llama model: {self.model}...")
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                device=self.device,
                trust_remote_code=True,
                torch_dtype="auto",
            )
            print(f"Llama model loaded successfully on device {self.device}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Llama model '{self.model}': {exc}\n\n"
                "Options:\n"
                "  * Ensure model id is correct: meta-llama/Llama-3.2-3B-Instruct\n"
                "  * Ensure sufficient GPU/CPU RAM available\n"
                "  * Install dependencies: pip install transformers torch accelerate\n"
                "  * For smaller model, try: meta-llama/Llama-3.2-1B-Instruct\n"
                "  * You may need to accept license on HuggingFace: hf auth login\n"
            ) from exc

    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 256, 
        temperature: float = 1.0, 
        **kwargs
    ) -> GenerationResult:
        """
        Generate text locally using Llama model with chat template.
        
        Args:
            prompt: input prompt string
            max_tokens: maximum new tokens to generate
            temperature: sampling temperature
        Returns:
            GenerationResult(text, metadata)
        """
        # Format prompt as chat message for instruction model
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "temperature": float(temperature),
            "do_sample": True if temperature > 0.0 else False,
            "top_p": 0.9,
            "top_k": 50,
        }
        
        start = time.time()
        outputs = self.generator(messages, **gen_kwargs)
        latency_ms = (time.time() - start) * 1000.0
        
        # Extract generated text
        text = ""
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict) and "generated_text" in first:
                # Get the assistant's response from chat format
                generated = first["generated_text"]
                if isinstance(generated, list) and len(generated) > 1:
                    # Chat template returns list of messages
                    assistant_msg = generated[-1]
                    if isinstance(assistant_msg, dict) and "content" in assistant_msg:
                        text = assistant_msg["content"]
                    else:
                        text = str(assistant_msg)
                elif isinstance(generated, str):
                    text = generated
                else:
                    text = str(generated)
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
