"""
Simple test script to check imports
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print(f"Python path: {sys.path[0]}")

try:
    from orchestrator import Orchestrator
    print("✅ Orchestrator import: SUCCESS")
except Exception as e:
    print(f"❌ Orchestrator import: FAILED - {e}")

try:
    from adapters.base_adapter import ModelAdapter
    print("✅ Base adapter import: SUCCESS")
except Exception as e:
    print(f"❌ Base adapter import: FAILED - {e}")

try:
    from adapters.gemini_adapter import GeminiAdapter
    print("✅ Gemini adapter import: SUCCESS")
    
    # Try to actually instantiate it
    adapter = GeminiAdapter()
    print("✅ Gemini adapter instantiation: SUCCESS")
except Exception as e:
    print(f"❌ Gemini adapter import/init: FAILED - {e}")

try:
    from adapters.mistral_adapter import MistralAdapter
    print("✅ Mistral adapter import: SUCCESS")
    
    # Try to actually instantiate it
    adapter = MistralAdapter()
    print("✅ Mistral adapter instantiation: SUCCESS")
except Exception as e:
    print(f"❌ Mistral adapter import/init: FAILED - {e}")

print("\n✅ All tests complete!")
