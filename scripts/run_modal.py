"""
Run benchmark on NVIDIA B200 GPUs via Modal.
Usage: modal run scripts/run_modal.py

One-time setup:
  modal setup
  modal volume create flashinfer-trace
  modal volume put flashinfer-trace /path/to/flashinfer-trace
"""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

# Modal configuration
APP_NAME  = "flashinfer-bench"
VOLUME    = modal.Volume.from_name("flashinfer-trace")
IMAGE     = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "triton>=3.0",
        "flashinfer-bench",
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=IMAGE,
    gpu="B200",                        # NVIDIA Blackwell B200
    volumes={"/flashinfer-trace": VOLUME},
    timeout=3600,
)
def run_benchmark(solution_json_str: str) -> list[dict]:
    import json
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet, Solution

    solution_data = json.loads(solution_json_str)
    solution = Solution(**solution_data)

    traces = TraceSet.from_path("/flashinfer-trace")
    config = BenchmarkConfig(warmup_runs=10, iterations=50)
    benchmark = Benchmark(traces, config)

    results = benchmark.run_solution(solution)

    return [
        {
            "workload_id": r.workload_id,
            "correct":     r.correct,
            "latency_ms":  getattr(r, "latency_ms", None),
            "throughput":  getattr(r, "throughput", None),
            "speedup":     getattr(r, "speedup_vs_baseline", None),
        }
        for r in results
    ]


@app.local_entrypoint()
def main():
    solution_json = ROOT / "solution.json"
    if not solution_json.exists():
        print("Error: solution.json not found. Run first:")
        print("  python scripts/pack_solution.py")
        raise SystemExit(1)

    solution_json_str = solution_json.read_text()
    solution_data     = json.loads(solution_json_str)
    print(f"Submitting solution '{solution_data['name']}' "
          f"(definition={solution_data['definition']}) to B200...")

    results = run_benchmark.remote(solution_json_str)

    print("\n=== B200 Benchmark Results ===")
    for r in results:
        status   = "✓ PASS" if r["correct"] else "✗ FAIL"
        latency  = f"{r['latency_ms']:.3f} ms" if r["latency_ms"] else "N/A"
        speedup  = f"{r['speedup']:.2f}x" if r["speedup"] else ""
        print(f"  {status}  {r['workload_id']}  {latency}  {speedup}")

    passed = sum(1 for r in results if r["correct"])
    print(f"\nTotal: {passed}/{len(results)} passed")
    print("\nNote: Modal scores are for reference. Official eval runs on bare metal.")
