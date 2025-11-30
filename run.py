"""
Simple run script to start the orchestrator.

- Strict mode: no fallback. If adapters/imports fail, exit with clear error.
- Prints basic progress and final location of artifacts.

Usage:
    python run.py
"""

import os
import sys
from pathlib import Path

# ensure project src is importable when running this script directly
ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main():
    # Import here so sys.path is already set
    # try:
    #     from adapters.deepseek_adapter import DeepSeekAdapter
    # except Exception as e:
    #     print("Error importing DeepSeekAdapter:", e)
    #     DeepSeekAdapter = None

    try:
        from adapters.flan_t5_adapter import FlanT5Adapter
    except Exception as e:
        print("Error importing FlanT5Adapter:", e)
        FlanT5Adapter = None

    try:
        from adapters.bloom_adapter import BloomAdapter
    except Exception as e:
        print("Error importing BloomAdapter:", e)
        BloomAdapter = None

    adapters = {}
    
    # # Initialize DeepSeek adapter
    # if DeepSeekAdapter is not None:
    #     try:
    #         print("Initializing DeepSeek adapter...")
    #         adapters["deepseek"] = DeepSeekAdapter()
    #     except Exception as e:
    #         print(f"Failed to initialize DeepSeekAdapter: {e}")
    #         print("  Consider using a smaller model or ensure GPU/RAM available")

    # # Initialize Llama adapter
    # if LlamaAdapter is not None:
    #     try:
    #         print("Initializing Llama adapter...")
    #         adapters["llama"] = LlamaAdapter()
    #     except Exception as e:
    #         print(f"Failed to initialize LlamaAdapter: {e}")
    #         print("  You may need to accept license: huggingface-cli login")

    # Initialize Flan-T5 adapter
    if FlanT5Adapter is not None:
        try:
            print("Initializing Flan-T5 adapter...")
            adapters["flan-t5"] = FlanT5Adapter()
        except Exception as e:
            print(f"Failed to initialize FlanT5Adapter: {e}")

    # Initialize BLOOM adapter
    if BloomAdapter is not None:
        try:
            print("Initializing BLOOM adapter...")
            adapters["bloom"] = BloomAdapter()
        except Exception as e:
            print(f"Failed to initialize BloomAdapter: {e}")

    if not adapters:
        print("\nNo adapters available or initialization failed. Exiting.")
        print("Ensure transformers, torch, and accelerate are installed:")
        print("  pip install transformers torch accelerate")
        sys.exit(1)

    print(f"\n{len(adapters)} adapter(s) initialized: {list(adapters.keys())}")

    # Import orchestrator now
    try:
        from orchestrator import Orchestrator
    except Exception as e:
        print("Failed to import Orchestrator:", e)
        sys.exit(1)

    config_path = "configs/pmtool.yaml"
    print(f"Loading configuration from: {config_path}")
    orch = Orchestrator(config_path=config_path, adapters=adapters)

    print("\nStarting review generation...\n")
    try:
        orch.run()
    except Exception as e:
        print("Generation failed with error:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nRun finished. Artifacts in:", orch.dataset_dir)


if __name__ == "__main__":
    main()
