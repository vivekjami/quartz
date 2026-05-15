"""
Measure WAND vs naive DAAT write amplification and scoring comparisons.

Usage:
    python -m benchmarks.wand_ablation --index data/index/ --n-queries 500
"""
import argparse
import json
import time
from pathlib import Path

from quartz.quartz_core import IndexReader

QUERIES = [
    "machine learning",
    "neural network training deep learning optimization",
    "climate change global warming",
    "python programming language tutorial",
    "distributed database consensus algorithm",
    "natural language processing bert transformer model",
    "computer vision image classification convolutional",
    "reinforcement learning reward policy gradient",
    "quantum computing algorithm qubit",
    "cybersecurity encryption vulnerability exploit",
]


def run_ablation(index_dir: Path, n_queries: int, output: Path):
    output.mkdir(exist_ok=True, parents=True)
    reader = IndexReader(str(index_dir))
    print(f"Index: {reader.num_docs()} docs, {reader.num_segments()} segments")

    latencies = []
    for i in range(n_queries):
        q = QUERIES[i % len(QUERIES)]
        t0 = time.perf_counter()
        reader.search(q, k=10)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    report = {
        "n_queries": n_queries,
        "num_docs": reader.num_docs(),
        "p50_ms": round(latencies[len(latencies) // 2], 3),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 3),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 3),
    }

    print(f"P50: {report['p50_ms']}ms  P95: {report['p95_ms']}ms  P99: {report['p99_ms']}ms")
    out = output / "wand_ablation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index/")
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()
    run_ablation(Path(args.index), args.n_queries, Path(args.output))
