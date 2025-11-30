"""
Gemini Adapter (Sync)
=====================

This module provides a minimal, production-oriented adapter for the Google
Gemini Generative AI models using the official `google-genai` SDK.

Purpose
-------
The adapter wraps the Gemini client behind a unified, model-agnostic interface
(`ModelAdapter`). This allows your Orchestrator to interact with multiple LLM
providers (Gemini, Mistral, OpenAI, etc.) using the **same call pattern**.

Key Features
------------
- Loads API key from environment (.env supported via python-dotenv).
- Initializes a Google GenAI client safely.
- Provides a `generate()` method that:
  - Sends prompts to Gemini
  - Wraps output in `GenerationResult`
  - Measures latency for observability and debugging
- Uses `GenerateContentConfig` for clean configuration of temperature,
  max tokens, etc.
- Raises helpful errors if configuration or environment variables are missing.

Dependencies
------------
Requires the modern SDK:
    pip install google-genai

NOTES:
- The legacy package `google-generativeai` is NOT compatible with this adapter.
- This adapter expects:     from google import genai
"""

import os
import time
from typing import Any, Optional

from google import genai
from google.genai import types

from .base_adapter import ModelAdapter, GenerationResult, GenerationMetadata

from dotenv import load_dotenv
load_dotenv()

class GeminiAdapter(ModelAdapter):
    """
    Adapter class for Google's Gemini models using the official `google-genai` SDK.

    This adapter follows the internal ModelAdapter interface, providing a clean,
    predictable API for your orchestrator.

    Parameters
    ----------
    model : str, default "gemini-flash-latest"
        The model version to use. Common choices include:
        - "gemini-flash-latest"  (fast, inexpensive, good for general tasks)
        - "gemini-pro"           (more capable)
        - "gemini-1.5-flash"
        - "gemini-1.5-pro"

    Attributes
    ----------
    api_key : str
        Google Gemini API key loaded from environment variable: GEMINI_API_KEY.
    client : genai.Client
        The instantiated Google GenAI SDK client.
    provider : str
        Identifier returned in metadata ("GoogleGemini").
    model : str
        The selected model name used for inference.
    """

    def __init__(self, model: str = "gemini-1.5-flash-latest"):
        """
        Initialize the Gemini client and validate environment configuration.

        Raises
        ------
        ValueError
            If GEMINI_API_KEY is not present in environment variables.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in your .env or system environment."
            )

        # Initialize Google GenAI SDK client
        self.client = genai.Client(api_key=self.api_key)

        self.model = model
        self.provider = "GoogleGemini"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate text from a Gemini model.

        This function handles:
        - Latency measurement
        - Response parsing
        - SDK call parameterization
        - Wrapping output in unified GenerationResult objects

        Parameters
        ----------
        prompt : str
            The user's input text.
        max_tokens : int, default 256
            Maximum number of tokens the model can generate.
        temperature : float, default 1.0
            Controls randomness. Lower = more deterministic.
        **kwargs : Any
            Ignored extra arguments for compatibility with orchestrator.

        Returns
        -------
        GenerationResult
            Contains:
            - text: str (the generated output)
            - metadata: GenerationMetadata (latency, provider, model, etc.)

        Raises
        ------
        Exception
            Any internal SDK exceptions will bubble up unless caught by caller.
        """
        # ---- Start latency timer ----
        start = time.time()

        # ---- Build the generation configuration ----
        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        # ---- Perform the model inference ----
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=gen_config,
        )

        # ---- Extract the model's text output ----
        # According to SDK docs, `response.text` is the standard text accessor.
        text = getattr(response, "text", "") or ""

        # ---- Compute latency ----
        latency_ms = (time.time() - start) * 1000.0

        # ---- Wrap metadata ----
        metadata = GenerationMetadata(
            provider=self.provider,
            model=self.model,
            tokens_in=None,   # SDK currently does not expose token counts
            tokens_out=None,
            latency_ms=latency_ms,
        )

        # ---- Return unified result ----
        return GenerationResult(text=text, metadata=metadata)
