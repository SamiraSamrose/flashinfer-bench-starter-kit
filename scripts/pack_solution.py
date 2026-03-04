"""
Pack solution source files into solution.json for submission.
Usage: python scripts/pack_solution.py
"""

import json
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    # Load config
    with open(ROOT / "config.toml", "rb") as f:
        config = tomllib.load(f)

    solution_cfg = config["solution"]
    build_cfg    = config["build"]

    language    = build_cfg["language"]
    entry_point_raw = build_cfg["entry_point"]  # e.g. "kernel.py::kernel"

    # Collect source files
    sources = []

    if language == "triton":
        src_dir = ROOT / "solution" / "triton"
        for path in sorted(src_dir.rglob("*.py")):
            rel = path.relative_to(src_dir)
            sources.append({
                "path": str(rel),
                "content": path.read_text(),
            })

    elif language == "cuda":
        src_dir = ROOT / "solution" / "cuda"
        for ext in ("*.cu", "*.cuh", "*.cpp", "*.h", "*.py"):
            for path in sorted(src_dir.rglob(ext)):
                rel = path.relative_to(src_dir)
                sources.append({
                    "path": str(rel),
                    "content": path.read_text(),
                })

    elif language == "python":
        src_dir = ROOT / "solution"
        for path in sorted(src_dir.rglob("*.py")):
            rel = path.relative_to(src_dir)
            sources.append({
                "path": str(rel),
                "content": path.read_text(),
            })

    if not sources:
        raise RuntimeError(f"No source files found for language='{language}'")

    solution = {
        "name":       solution_cfg["name"],
        "definition": solution_cfg["definition"],
        "author":     solution_cfg["author"],
        "spec": {
            "language":                language,
            "target_hardware":         ["NVIDIA_B200"],
            "entry_point":             entry_point_raw,
            "dependencies":            ["torch", "triton >= 3.0"] if language == "triton"
                                       else ["torch"],
            "destination_passing_style": False,
        },
        "sources": sources,
    }

    out_path = ROOT / "solution.json"
    with open(out_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Packed {len(sources)} source file(s) into {out_path}")
    print(f"  name:       {solution['name']}")
    print(f"  definition: {solution['definition']}")
    print(f"  language:   {language}")
    print(f"  entry_point:{entry_point_raw}")


if __name__ == "__main__":
    main()
