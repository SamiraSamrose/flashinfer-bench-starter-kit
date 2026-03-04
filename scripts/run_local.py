"""
Run local benchmark against the FlashInfer-Trace dataset.
Usage: python scripts/run_local.py

Requires:
  - Local CUDA-capable GPU
  - FIB_DATASET_PATH environment variable set to your flashinfer-trace clone
  - pip install flashinfer-bench
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    dataset_path = os.environ.get("FIB_DATASET_PATH")
    if not dataset_path:
        print("Error: FIB_DATASET_PATH environment variable not set.")
        print("  git lfs install")
        print("  git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest")
        print("  export FIB_DATASET_PATH=/path/to/mlsys26-contest")
        sys.exit(1)

    solution_json = ROOT / "solution.json"
    if not solution_json.exists():
        print("Error: solution.json not found. Run first:")
        print("  python scripts/pack_solution.py")
        sys.exit(1)

    try:
        from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet, Solution
    except ImportError:
        print("Error: flashinfer-bench not installed.")
        print("  pip install flashinfer-bench")
        sys.exit(1)

    # Load solution
    with open(solution_json) as f:
        solution_data = json.load(f)
    solution = Solution(**solution_data)

    # Load dataset traces
    traces = TraceSet.from_path(dataset_path)

    # Filter workloads matching our definition
    definition_name = solution_data["definition"]
    matching = [t for t in traces if t.definition == definition_name]
    if not matching:
        print(f"Warning: No workloads found for definition '{definition_name}' in dataset.")
        print(f"  Available definitions: {sorted(set(t.definition for t in traces))}")

    print(f"\nRunning benchmark for '{definition_name}' on {len(matching)} workload(s)...")

    config = BenchmarkConfig(warmup_runs=10, iterations=50)
    benchmark = Benchmark(traces, config)

    results = benchmark.run_solution(solution)

    print("\n=== Results ===")
    for r in results:
        status = "✓ PASS" if r.correct else "✗ FAIL"
        throughput = f"{r.throughput:.2f} GB/s" if hasattr(r, "throughput") else ""
        latency    = f"{r.latency_ms:.3f} ms" if hasattr(r, "latency_ms") else ""
        print(f"  {status}  {r.workload_id}  {latency}  {throughput}")

    print(f"\nTotal: {sum(1 for r in results if r.correct)}/{len(results)} passed")


if __name__ == "__main__":
    main()
