"""
Base Adapter Interface for Synthetic Review Generator

This module defines the minimal, provider-agnostic interface for any model
(OpenAI, Gemini, Mistral, Local LLM, etc.) used in the synthetic data pipeline.

Every provider-specific adapter will:
    - Inherit from ModelAdapter
    - Implement the `.generate()` method
    - Return a GenerationResult containing:
          * generated text
          * metadata (latency, token usage, provider, model, raw response)

Keeping a clean interface allows the orchestrator to use any model
interchangeably without caring about provider-specific details.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


# -----------------------------
# Metadata about each generation
# -----------------------------
@dataclass
class GenerationMetadata:
    """
    Holds useful information about a generation call.

    These fields help you later with:
        - Quality reporting
        - Provider comparison
        - Benchmarking latency and cost
        - Inspecting raw model output (debugging)

    All fields are optional because not every provider returns all metrics.
    """
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    latency_ms: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None


# -----------------------------
# Wrapper containing text + metadata
# -----------------------------
@dataclass
class GenerationResult:
    """
    Represents the outcome of a text generation call.

    Why not return just a string?
    --------------------------------
    Because the project requires:
        - Comparing models
        - Tracking latency and quality per provider
        - Generating quality reports
        - Debugging low-quality outliers

    This container gives you both the text and the metadata together.
    """
    text: str
    metadata: GenerationMetadata


# -----------------------------
# Abstract base adapter
# -----------------------------
class ModelAdapter(ABC):
    """
    Base class for all model adapters.

    Each adapter must implement:
        - generate(): main sync generation method

    Optional:
        - agenerate(): async version for libraries that support async HTTP
          (OpenAI, Anthropic, Google Gemini, Mistral, etc.)

    The orchestrator **only relies on this interface**, so any provider that
    follows it will work seamlessly.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        **kwargs,
    ) -> GenerationResult:
        """
        Main synchronous generation method.

        Implementations MUST:
            - Send the prompt to the model provider
            - Build and return a GenerationResult object
            - Measure latency
            - (Optional) extract token usage if supported
            - Handle provider-specific exceptions and wrap them cleanly

        Providers: OpenAI, Gemini, Mistral, Ollama...
        """
        raise NotImplementedError

    # ----------------------------------------------------
    # Optional: async generation wrapper
    # ----------------------------------------------------
    async def agenerate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        **kwargs,
    ) -> GenerationResult:
        """
        Optional async version of `generate`.

        Default behavior:
            - Runs the sync method in a background thread.
            - Useful for keeping code simple while still allowing async execution.

        Providers that support async HTTP clients can override this for efficiency.
        """
        from concurrent.futures import ThreadPoolExecutor
        import asyncio

        loop = asyncio.get_running_loop()

        # Run sync generate() in a background thread to avoid blocking event loop
        with ThreadPoolExecutor(1) as ex:
            return await loop.run_in_executor(
                ex,
                lambda: self.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                ),
            )
