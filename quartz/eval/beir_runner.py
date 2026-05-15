"""
Run BEIR evaluation against the Quartz index.
Supports: scifact, nfcorpus, fiqa, arguana

Usage:
    python -m quartz.eval.beir_runner --dataset scifact --index data/index/ --output results/
"""
import argparse
import json
import time
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from quartz.quartz_core import IndexReader
from quartz.ingest.wet_reader import tokenize


def run_beir(dataset: str, index_dir: Path, output_dir: Path, k_values=(1, 3, 5, 10)):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download BEIR dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, "data/beir/")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    reader = IndexReader(str(index_dir))

    # Retrieve results for all queries
    results: dict[str, dict[str, float]] = {}
    latencies = []

    print(f"Running {dataset}: {len(queries)} queries")
    for q_id, query_text in queries.items():
        t0 = time.perf_counter()
        bm25_results = reader.search(query_text, k=100)
        latencies.append((time.perf_counter() - t0) * 1000)

        results[q_id] = {}
        for score, doc_id in bm25_results:
            # BEIR expects corpus doc_ids — in a full implementation, you'd
            # map URL → BEIR corpus ID. Here we use doc_id as a proxy.
            results[q_id][str(doc_id)] = float(score)

    # Evaluate
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, k_values=list(k_values)
    )

    latencies.sort()
    report = {
        "dataset": dataset,
        "ndcg": ndcg,
        "recall": recall,
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)],
        "latency_p99_ms": latencies[int(len(latencies) * 0.99)],
        "n_queries": len(queries),
    }

    out_path = output_dir / f"{dataset}_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nnDCG@10: {ndcg.get('NDCG@10', 'N/A')}")
    print(f"P95 latency: {report['latency_p95_ms']:.1f}ms")
    print(f"Results saved: {out_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--index", default="data/index/")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()
    run_beir(args.dataset, Path(args.index), Path(args.output))
