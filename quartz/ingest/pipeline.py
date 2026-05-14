import time
import xxhash  # uv pip install xxhash
from pathlib import Path
from tqdm import tqdm

from quartz.quartz_core import IndexWriter   # Rust extension
from quartz.ingest.wet_reader import iter_wet_file, tokenize
from quartz.ingest.simhash import SimHashDeduplicator

def run_ingestion(
    wet_dir: Path,
    index_dir: Path,
    flush_threshold_mb: int = 50,
    max_docs: int | None = None,
):
    index_dir.mkdir(parents=True, exist_ok=True)
    writer = IndexWriter(str(index_dir), flush_threshold_mb * 1024 * 1024)
    deduper = SimHashDeduplicator()

    wet_files = sorted(wet_dir.glob("*.warc.wet.gz"))
    if not wet_files:
        wet_files = sorted(wet_dir.glob("*.wet.gz"))
    print(f"Found {len(wet_files)} WET files")

    total_docs = 0
    t0 = time.perf_counter()

    with tqdm(wet_files, desc="Files") as file_bar:
        for wet_file in file_bar:
            for doc in iter_wet_file(wet_file):
                if max_docs and total_docs >= max_docs:
                    break

                # Dedup check
                if deduper.check_and_add(doc.text[:1000]):
                    continue

                # Content hash for change detection
                content_hash = xxhash.xxh64(doc.text.encode()).intdigest()

                # Tokenize
                tokens = tokenize(doc.text)
                if len(tokens) < 10:
                    continue

                # Write to index (Rust)
                writer.add_doc(
                    url=doc.url,
                    tokens=tokens,
                    crawl_ts=doc.crawl_ts,
                    content_hash=content_hash,
                )
                total_docs += 1

                if total_docs % 100_000 == 0:
                    elapsed = time.perf_counter() - t0
                    throughput = total_docs / elapsed
                    file_bar.set_postfix({
                        "docs": f"{total_docs:,}",
                        "docs/s": f"{throughput:.0f}",
                        "dupe_rate": f"{deduper.dupe_rate:.2%}",
                    })

            if max_docs and total_docs >= max_docs:
                break

    # Final flush
    writer.flush()
    writer.run_merge()

    elapsed = time.perf_counter() - t0
    print(f"\nIngestion complete:")
    print(f"  Documents indexed:  {total_docs:,}")
    print(f"  Dupe rate:          {deduper.dupe_rate:.2%}")
    print(f"  Total time:         {elapsed:.1f}s")
    print(f"  Throughput:         {total_docs/elapsed:.0f} docs/sec")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/wet/")
    parser.add_argument("--index-dir", default="data/index/")
    parser.add_argument("--flush-mb", type=int, default=50)
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    run_ingestion(
        wet_dir=Path(args.data_dir),
        index_dir=Path(args.index_dir),
        flush_threshold_mb=args.flush_mb,
        max_docs=args.max_docs,
    )
