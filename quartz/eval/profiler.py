"""
Query latency profiler: runs N queries against the index and reports
P50 / P95 / P99 latency for BM25-only and hybrid modes.

Usage:
    python -m quartz.eval.profiler --index data/index/ --n-queries 1000
"""
import argparse
import json
import random
import time
from pathlib import Path

from quartz.quartz_core import IndexReader

# A handful of representative web queries to cycle through
SAMPLE_QUERIES = [
    "machine learning neural network",
    "climate change renewable energy",
    "python programming tutorial",
    "artificial intelligence research",
    "deep learning computer vision",
    "natural language processing transformer",
    "database indexing performance",
    "distributed systems consensus",
    "rust programming language memory",
    "web search ranking algorithm",
]


def run_profiler(index_dir: Path, n_queries: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = IndexReader(str(index_dir))
    print(f"Index: {reader.num_segments()} segments, {reader.num_docs()} docs")
    print(f"Running {n_queries} queries...")

    latencies = []
    for i in range(n_queries):
        q = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        t0 = time.perf_counter()
        reader.search(q, k=10)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    report = {
        "n_queries": n_queries,
        "num_docs": reader.num_docs(),
        "p50_ms": round(latencies[len(latencies) // 2], 2),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
        "min_ms": round(latencies[0], 2),
        "max_ms": round(latencies[-1], 2),
    }

    print(f"\nLatency results (BM25, k=10):")
    print(f"  P50:  {report['p50_ms']}ms")
    print(f"  P95:  {report['p95_ms']}ms")
    print(f"  P99:  {report['p99_ms']}ms")
    print(f"  Min:  {report['min_ms']}ms")
    print(f"  Max:  {report['max_ms']}ms")

    out = output_dir / "latency_profile.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index/")
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()
    run_profiler(Path(args.index), args.n_queries, Path(args.output))
