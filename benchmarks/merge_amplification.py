"""
Measure write amplification caused by the tiered merge policy.

Usage:
    python -m benchmarks.merge_amplification --index data/index/
"""
import argparse
import json
from pathlib import Path

from quartz.quartz_core import IndexReader


def measure_amplification(index_dir: Path, output: Path):
    output.mkdir(exist_ok=True, parents=True)
    reader = IndexReader(str(index_dir))

    # Read segment meta from the actual directory, because IndexReader
    # might only expose aggregate stats. We can read meta.json directly.
    total_docs = 0
    total_postings_written = 0
    
    segments = 0
    for seg_dir in index_dir.glob("seg_*"):
        if not seg_dir.is_dir():
            continue
        meta_file = seg_dir / "meta.json"
        if not meta_file.exists():
            continue
            
        with open(meta_file, "r") as f:
            meta = json.load(f)
            total_docs += meta.get("num_docs", 0)
            # A segment with merge_gen > 0 means its postings were rewritten
            merge_gen = meta.get("merge_gen", 0)
            postings = meta.get("total_postings", 0)
            
            # Rough estimate: each posting was written (merge_gen + 1) times
            total_postings_written += postings * (merge_gen + 1)
            segments += 1

    if total_docs == 0:
        print("Index is empty. Run ingestion first.")
        return

    # Calculate write amplification factor
    # Amplification = (Total postings written to disk) / (Unique postings currently in index)
    unique_postings = sum(
        json.load(open(d / "meta.json"))["total_postings"]
        for d in index_dir.glob("seg_*")
        if (d / "meta.json").exists()
    )
    
    amp_factor = total_postings_written / max(unique_postings, 1)

    report = {
        "num_segments": segments,
        "num_docs": total_docs,
        "unique_postings": unique_postings,
        "total_postings_written": total_postings_written,
        "write_amplification_factor": round(amp_factor, 2)
    }

    print(f"Write Amplification Benchmark:")
    print(f"  Segments:       {segments}")
    print(f"  Documents:      {total_docs:,}")
    print(f"  Write Amp:      {amp_factor:.2f}x")
    
    out = output / "merge_amplification.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index/")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()
    measure_amplification(Path(args.index), Path(args.output))
