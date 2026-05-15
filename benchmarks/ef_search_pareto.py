"""
Plot HNSW ef_search vs recall@10 and P95 latency (Pareto frontier).
This curve is what production IR engineers think about every day.

Usage:
    python -m benchmarks.ef_search_pareto --hnsw-dir data/hnsw/ --n-queries 200
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from quartz.hnsw.graph import HNSWGraph


def run_pareto(hnsw_dir: Path, n_queries: int = 200, output: Path = Path("results/")):
    graph = HNSWGraph.load(hnsw_dir)
    output.mkdir(exist_ok=True, parents=True)

    all_ids = list(graph.vectors.keys())
    if len(all_ids) == 0:
        print("HNSW graph is empty — run `python -m quartz.hnsw.build` first.")
        return

    query_ids = random.sample(all_ids, min(n_queries, len(all_ids)))
    queries = [graph.vectors[i].astype(np.float32) for i in query_ids]

    ef_values = [10, 20, 50, 100, 200, 400]
    data = []

    # Ground truth: brute-force exact nearest neighbors
    print(f"Computing ground truth for {len(queries)} queries (brute force)...")
    all_vecs = np.stack([graph.vectors[i].astype(np.float32) for i in all_ids])
    gt = []
    for q in queries:
        norm_q = np.linalg.norm(q)
        if norm_q == 0:
            gt.append(set())
            continue
        sims = np.dot(all_vecs, q) / (np.linalg.norm(all_vecs, axis=1) * norm_q + 1e-9)
        top10 = set(all_ids[i] for i in np.argsort(-sims)[:10])
        gt.append(top10)

    for ef in ef_values:
        latencies, recalls = [], []
        for q, true_top10 in zip(queries, gt):
            t0 = time.perf_counter()
            results = graph.search(q, k=10, ef=ef)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            retrieved = set(doc_id for _, doc_id in results)
            r = len(retrieved & true_top10) / max(len(true_top10), 1)
            recalls.append(r)

        latencies.sort()
        entry = {
            "ef": ef,
            "recall_at_10": round(sum(recalls) / len(recalls), 4),
            "p50_ms": round(latencies[len(latencies) // 2], 3),
            "p95_ms": round(latencies[int(len(latencies) * 0.95)], 3),
        }
        data.append(entry)
        print(f"ef={ef:3d}: recall@10={entry['recall_at_10']:.4f}, P95={entry['p95_ms']:.2f}ms")

    out = output / "pareto.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"\nSaved Pareto data to {out}")

    # Try to plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        p95s = [d["p95_ms"] for d in data]
        recalls = [d["recall_at_10"] for d in data]
        efs = [d["ef"] for d in data]

        ax1.plot(p95s, recalls, "o-", linewidth=2, markersize=8, color="#4F8EF7")
        for d in data:
            ax1.annotate(f"ef={d['ef']}", (d["p95_ms"], d["recall_at_10"]),
                         textcoords="offset points", xytext=(5, -12), fontsize=9)
        ax1.set_xlabel("P95 Latency (ms)")
        ax1.set_ylabel("Recall@10")
        ax1.set_title("HNSW: Recall vs Latency Pareto Frontier")
        ax1.grid(True, alpha=0.3)

        ax2.bar([str(e) for e in efs], p95s, color="#4F8EF7")
        ax2.set_xlabel("ef_search")
        ax2.set_ylabel("P95 Latency (ms)")
        ax2.set_title("P95 Latency vs ef_search")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = output / "pareto.png"
        plt.savefig(str(plot_path), dpi=150)
        print(f"Plot saved: {plot_path}")
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hnsw-dir", default="data/hnsw/")
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()
    run_pareto(Path(args.hnsw_dir), args.n_queries, Path(args.output))
