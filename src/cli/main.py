"""
main.py

Entry point for running the Orchestrator with real model adapters.

Behavior:
- Imports real adapters (Gemini, Mistral)
- No fallback: if imports fail, the script raises an error
- If all adapters fail during generation, an exception is raised
- Validators are executed automatically by Orchestrator

Usage:
    python main.py "your prompt here"
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import Orchestrator


def run(prompt: str):
    """
    Executes a full LLM orchestration pipeline:
    - Loads adapters (strict: must exist)
    - Passes the prompt through validators
    - Runs generation via orchestrator
    - Prints final output and validation data
    """

    # Import adapters (strict mode)
    try:
        from adapters.gemini_adapter import GeminiAdapter
        from adapters.mistral_adapter import MistralAdapter
    except Exception as e:
        raise ImportError(
            f"Failed to import one or more adapters: {e}\n"
            "Make sure GeminiAdapter and MistralAdapter are implemented."
        )

    # Instantiate adapters
    model_adapters = [
        GeminiAdapter(),
        MistralAdapter(),
    ]

    # No fallback — strict behavior
    orchestrator = Orchestrator(
        model_adapters=model_adapters,
        fallback_adapter=None,   # explicitly no fallback
    )

    # This may raise if all adapters fail (by design)
    result = orchestrator.generate(prompt)

    print("\n=== ORCHESTRATION RESULT ===")
    print("Model Used:", result["provider"])
    print("Latency:", f"{result['latency_ms']:.2f} ms")
    print("Text Output:\n", result["text"])

    print("\n--- Validators ---")
    for name, value in result["validators"].items():
        print(f"{name}: {value}")

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        user_prompt = "Explain the importance of good API design."

    run(user_prompt)
