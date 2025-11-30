"""
LLaMA-2 Adapter (local transformers + 4-bit quantization)

This adapter uses Hugging Face `transformers` to run LLaMA-2 locally,
applying 4-bit quantization via `bitsandbytes` to reduce memory usage.

Requirements:
    pip install transformers torch accelerate bitsandbytes scipy
"""

import time
import torch
from typing import Optional
from dotenv import load_dotenv

# transformer imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None

from .base_adapter import ModelAdapter, GenerationResult, GenerationMetadata

load_dotenv()

class Llama2Adapter(ModelAdapter):
    """
    Local LLaMA-2 adapter with 4-bit quantization.
    
    Args:
        model: local model id or path. Default: "meta-llama/Llama-2-7b-chat-hf"
    """

    def __init__(self, model: str = "meta-llama/Llama-2-7b-chat-hf", device: Optional[int] = None):
        self.model = model
        self.provider = "local-llama2-4bit"
        
        # For 4-bit quantization, we typically let accelerate handle device placement (device_map="auto")
        # Explicit device assignment might conflict with device_map="auto"
        self.device_map = "auto" 

        if pipeline is None or BitsAndBytesConfig is None:
            raise RuntimeError(
                "transformers or bitsandbytes not available. "
                "Install with: pip install transformers torch accelerate bitsandbytes scipy"
            )

        try:
            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
            
            # Load model with quantization
            self.model_obj = AutoModelForCausalLM.from_pretrained(
                self.model,
                quantization_config=bnb_config,
                device_map=self.device_map,
                trust_remote_code=True
            )

            # Create pipeline using the loaded model and tokenizer
            self.generator = pipeline(
                "text-generation",
                model=self.model_obj,
                tokenizer=self.tokenizer,
                # device_map is handled by model loading
            )
            
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{self.model}' with quantization: {exc}\n"
                "Ensure you have a GPU, installed bitsandbytes, and have access to the model (huggingface-cli login)."
            ) from exc

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 1.0, **kwargs) -> GenerationResult:
        """
        Generate text locally using LLaMA-2.
        """
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "temperature": float(temperature),
            "do_sample": True if temperature > 0.0 else False,
            "top_p": 0.95,
        }

        start = time.time()
        # The pipeline might return different structures depending on version/task
        outputs = self.generator(prompt, **gen_kwargs)
        latency_ms = (time.time() - start) * 1000.0

        text = ""
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict) and "generated_text" in first:
                # LLaMA generation often includes the prompt, we might want to strip it if needed
                # But for now, we return as is or let the orchestrator handle it.
                # Often text-generation pipeline returns the full text (prompt + completion).
                full_text = first["generated_text"]
                # Optional: strip prompt if it's just a completion task, 
                # but base adapter contract doesn't strictly enforce this.
                # We'll return full text for now.
                text = full_text
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
