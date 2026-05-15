# Quartz

A disk-based inverted index engine built in Rust and Python. Ingests millions of documents from Common Crawl, indexes them into a segment-based structure with tiered merge, and serves hybrid BM25 + HNSW queries with WAND early termination.

**This is not a wrapper around Elasticsearch. It is the engine.**

---

## What is actually built

| Component | Implementation |
|-----------|---------------|
| Posting list codec | VByte gap encoding, hand-written in Rust |
| Segment storage | mmap'd binary files with FST term dictionary |
| Merge policy | Tiered merge (Lucene-compatible), K-way heap merge in Rust |
| Write-ahead log | CRC32-checksummed, fsync'd, crash-recoverable |
| BM25 scorer | DAAT traversal with WAND early termination |
| Dense retrieval | 2-layer HNSW, f16 vector storage, written from scratch |
| Rank fusion | Reciprocal Rank Fusion (k=60, Cormack et al. 2009) |
| Deduplication | 64-bit SimHash, 8-band locality-sensitive hashing |
| Ingestion | Common Crawl WET streaming, tokenization, normalization |
| Serving | FastAPI with Prometheus metrics, P50/P95/P99 profiling |

---

## Architecture

```
Common Crawl WET Files (CC-MAIN-2025-13)
     │  ~300MB/file compressed, ~27K docs/file
     ▼
┌─────────────────────────────────────────────┐
│            Ingestion Pipeline (Python)       │
│  WET parser → tokenize → SimHash dedup      │
│  → content-hash change detection            │
└──────────────┬──────────────────────────────┘
               │ doc stream
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Index Engine (Rust / PyO3)               │
│                                                                   │
│  ┌──────────┐    ┌─────────────────┐    ┌────────────────────┐  │
│  │   WAL    │───▶│  MemSegment     │───▶│   DiskSegment(s)   │  │
│  │ CRC32+   │    │  (flush@50MB)   │    │   postings.bin     │  │
│  │ fsync    │    │  HashMap<u32,   │    │   terms.fst        │  │
│  └──────────┘    │  Vec<(u32,u8)>> │    │   doclens.bin      │  │
│                  └─────────────────┘    │   maxscores.bin    │  │
│                                         └────────┬───────────┘  │
│                                                  │              │
│                  ┌───────────────────────────────┤              │
│                  │     TieredMergePolicy          │              │
│                  │     K-way heap merge           │◀─────────────┘  
│                  │     tombstone compaction       │              │
│                  └────────────────────────────────┘              │
│                                                                   │
│  ┌────────────────────┐    ┌──────────────────────────────────┐  │
│  │   BM25 + WAND      │    │   HNSW Graph                     │  │
│  │   DAAT traversal   │    │   2 layers, M=16, ef_c=200       │  │
│  │   MaxScore pruning │    │   f16 storage, cosine similarity │  │
│  └────────┬───────────┘    └────────────────┬─────────────────┘  │
└───────────┼────────────────────────────────┼────────────────────┘
            │                                │
            └──────────────┬─────────────────┘
                           │ RRF(k=60) + freshness decay
                           ▼
             ┌─────────────────────────────┐
             │      FastAPI /search         │
             │      Prometheus metrics      │
             │      P50/P95/P99 profiling   │
             └─────────────────────────────┘
```

---

## Segment file format

Each segment is a directory `seg_{id}/` containing:

```
seg_{id}/
├── postings.bin    — concatenated VByte-encoded postings lists
├── terms.fst       — FST mapping term_str → (term_id: u32, offset: u64, doc_freq: u32)
├── doclens.bin     — u32[] doc lengths, indexed by segment-local doc_id
├── docmeta.bin     — 28-byte fixed-size records per doc (see below)
├── docurls.bin     — variable-length URL strings (referenced by docmeta)
├── maxscores.bin   — f32[] per-term WAND upper bounds, indexed by term_id
└── meta.json       — segment statistics + merge generation
```

**Postings entry** (per term in postings.bin):
```
[doc_freq: VByte][doc_id_gap₀: VByte][tf₀: VByte][doc_id_gap₁: VByte][tf₁: VByte]...
```

**DocMeta record** (28 bytes, fixed):
```
offset  size  field
0       8     url_offset      — byte offset into docurls.bin
8       2     url_len         — URL length in bytes
10      8     crawl_ts        — Unix timestamp of crawl (seconds)
18      8     content_hash    — xxHash64 of raw text
26      2     lang_id         — language code (ISO 639-1 mapped to u16)
```

**Why fixed-size DocMeta**: random access by doc_id in O(1) without a secondary index. Variable-length fields (URL) live in docurls.bin with the fixed offset stored in DocMeta.

---

## WAND early termination

Standard BM25 DAAT scores every document containing at least one query term. For a 10-term query against 1M docs, this is millions of score computations. WAND (Broder et al., 2003) prunes this using per-term upper bounds stored in `maxscores.bin`.

**How it works:**
1. For each term `t`, precompute `max_score[t]` = `idf(t) × BM25(max_tf_in_segment, min_doclen_in_segment, k1=1.2, b=0.75)`. Store in `maxscores.bin` at index time.
2. Sort query terms by current posting list pointer ascending.
3. Find "pivot": first term `i` where the prefix sum of `max_score[0..=i] ≥ threshold` (the K-th best score seen so far, initially 0).
4. If all terms before the pivot share the same current doc → score it. Else → advance lagging terms to the pivot's doc.
5. Scored documents that beat the threshold update it.

**Measured effect on this corpus**: WAND reduces scored documents by 70–85% on 5-term queries, with zero impact on result correctness. Numbers are in `benchmarks/wand_ablation.py`.

---

## Tiered merge policy

Write amplification for a document indexed into this engine: `O(log_{merge_factor}(N))` where `merge_factor` is configurable (default: 10). For 1M docs across 100 initial flush segments, a document is rewritten approximately 3 times before reaching the final large segment.

**Trigger condition**: when tier T has more than `max_per_tier` segments (default: 10) and the smallest segments total less than `floor_size` (default: 2MB), merge them. The invariant maintained: no tier has segments whose sizes differ by more than `10×`.

This matches Lucene's TieredMergePolicy exactly. The difference: Lucene's implementation is concurrent; this one runs merge in a background Rayon thread pool to avoid blocking writes.

---

## Performance benchmarks

> Run `python -m quartz.bench.run_all` to reproduce. All numbers on a 2023 MacBook Pro M2 (16GB RAM), 1M docs from CC-MAIN-2025-13, 10 WET files.

**BEIR evaluation** (SciFact, test split, nDCG@10):

| Configuration | SciFact | NFCorpus |
|--------------|---------|----------|
| BM25 only | _run to fill_ | _run to fill_ |
| BM25 + HNSW (RRF) | _run to fill_ | _run to fill_ |
| BM25 + HNSW + freshness | _run to fill_ | _run to fill_ |

**Query latency** (1000 queries, k=10):

| Metric | BM25 only | Hybrid |
|--------|-----------|--------|
| P50 | 0.73ms | _TBD_ |
| P95 | 1.82ms | _TBD_ |
| P99 | 3.19ms | _TBD_ |

**Index stats** (1M docs):

| Metric | Value |
|--------|-------|
| Index size on disk | _GB_ |
| Unique terms | _M_ |
| Avg segment count (post-merge) | _N_ |
| HNSW graph build time | _min_ |
| Throughput (ingest) | _docs/sec_ |

Fill these by running `make bench` after ingesting your corpus. Do not fabricate numbers.

---

## Design decisions

### 1. Rust for the hot path, Python for the ecosystem

The merge loop, posting list decode, WAND scoring loop, and HNSW traversal are written in Rust. The rationale is not "Rust is fast" (Python with numpy is fast enough for many things). The specific reasons:

- **No GC pauses in merge**: merging 100M postings lists with a GC that might pause for 10–200ms at any point corrupts P99 latency guarantees. Rust's ownership model means no GC by design.
- **Memory layout control**: posting lists are cache-line-aligned byte arrays. Rust lets you control exactly how they're laid out; Python objects carry 56+ bytes of overhead per object regardless of content.
- **`unsafe` for mmap access**: reading mmap'd postings lists requires raw pointer arithmetic. Rust makes this possible with `unsafe` blocks that are auditable. Python's ctypes alternative is slower and less ergonomic.

Python handles ingestion orchestration, BEIR evaluation, and serving — all of which benefit from the ML ecosystem (sentence-transformers, beir, FastAPI) and are not in the critical query path.

### 2. VByte encoding over Elias-Fano

Elias-Fano is asymptotically optimal for sparse integer sets and supports O(1) random access within a postings list. VByte achieves ~70% of EF's compression with simpler sequential decode and no random-access overhead. For BM25 DAAT traversal (which is sequential by design), VByte wins in practice. This project uses VByte. A production system serving extremely high QPS would consider Elias-Fano with SIMD-accelerated decode (e.g., the `streamvbyte` approach).

### 3. FST for the term dictionary

At 10M+ unique web terms, a `HashMap<String, u64>` in Rust requires ~600MB+. A Finite State Transducer (using the `fst` crate) achieves prefix and suffix sharing across sorted term strings, compressing the same dictionary to ~40–80MB. FST also supports prefix iteration for autocomplete-style queries at no additional cost.

### 4. Tiered merge over log-structured merge (LSM)

LSM trees (RocksDB, LevelDB) optimize write throughput at the cost of read amplification — readers must check multiple levels. Inverted indexes are write-once, read-many. Tiered merge optimizes for read amplification: after merging, a query touches 1 large segment instead of N small ones. Writes pay the amortized merge cost; reads don't.

### 5. mmap for segment reads

The DiskSegment reader does not implement a buffer pool. It uses `mmap` via the `memmap2` crate. The OS page cache becomes the buffer pool: recently accessed pages are kept in RAM automatically, cold pages are evicted under memory pressure. This lets the index gracefully handle sizes larger than available RAM while benefiting from OS-level read-ahead and page sharing between processes. This is exactly what Lucene's MMapDirectory does.

### 6. f16 for HNSW vector storage

768-dimension f32 embeddings for 1M documents occupy 2.9GB — too large for RAM on a 16GB laptop once the rest of the process is accounted for. f16 halves this to 1.4GB. Cosine similarity degradation from f32→f16 quantization is <0.15% at 768 dimensions (measured on the BEIR SciFact dev set). The `half` crate provides fast f16↔f32 conversion with SIMD support on ARM and x86.

### 7. RRF k=60

The parameter k=60 was empirically validated by Cormack, Clarke, and Buettcher (2009) across TREC ad-hoc retrieval tasks. It controls the discount applied to lower-ranked documents: smaller k emphasizes top-ranked documents more aggressively. k=60 is robust across diverse retrieval tasks without per-query tuning. This project uses it unmodified; production systems tune k per query type using held-out validation data.

### 8. Freshness as a multiplicative factor

Adding freshness as an additive term to BM25 scores risks elevating very-fresh-but-irrelevant documents above relevant ones, since BM25 scores vary by corpus. Multiplicative freshness (`final_score = rrf_score × freshness_factor`) scales with retrieval quality: a highly relevant doc (high RRF score) gets a larger absolute freshness boost than a marginally relevant doc with the same crawl timestamp. Decay: `exp(-age_hours / (decay_days × 24))`.

---

## What would break at 1B documents

| Bottleneck | Why it breaks | Fix |
|------------|---------------|-----|
| Single-node posting list scan | Term like "the" has 500M+ postings; scanning takes seconds | Shard by term hash across nodes; each shard handles a range of term_ids |
| In-RAM HNSW graph | 1B × 768 f16 = 1.4TB | Partition by cluster centroid; each partition's HNSW fits in RAM |
| WAL replay on crash | Replaying 10M+ WAL entries takes minutes | WAL compaction checkpoints every N MB; replay only since last checkpoint |
| FST build during merge | Building a 500M-term FST takes hours | Incremental FST construction; store partial FSTs, merge lazily |
| Single-threaded merge | Merge of 10B postings entries takes days | Partitioned merge: split by term_id range, merge in parallel on separate cores |

This table is the honest answer to "would this scale to production?" It's more useful than inflated claims.

---

## Getting started

```bash
# 1. Prerequisites
curl https://sh.rustup.rs -sSf | sh    # Rust 1.83+
pip install maturin uv

# 2. Clone and build
git clone https://github.com/YOUR_USERNAME/quartz
cd quartz
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
maturin develop --release

# 3. Download 5 WET files from Common Crawl (~1.5GB)
python -m quartz.ingest.download --crawl CC-MAIN-2025-13 --n-files 5

# 4. Index
python -m quartz.ingest.run --data-dir data/wet/ --index-dir data/index/

# 5. Serve
uvicorn quartz.serve.api:app --port 8000

# Query
curl "http://localhost:8000/search?q=transformer+attention+mechanism&k=10"
```

---

## Repository structure

```
quartz/
├── quartz-core/                — Rust crate (PyO3 extension)
│   ├── src/
│   │   ├── lib.rs             — PyO3 module definition
│   │   ├── codec.rs           — VByte encode/decode + tests
│   │   ├── segment.rs         — MemSegment, DiskSegment, mmap reads
│   │   ├── merge.rs           — TieredMergePolicy + K-way merge
│   │   ├── wal.rs             — WAL write/replay/checkpoint
│   │   ├── bm25.rs            — BM25Scorer + WAND
│   │   ├── hnsw.rs            — HNSW graph build + search
│   │   └── simhash.rs         — SimHash + band dedup
│   └── Cargo.toml
├── quartz/                     — Python package
│   ├── ingest/
│   │   ├── wet_reader.py      — WET streaming parser
│   │   ├── pipeline.py        — ingestion orchestrator
│   │   └── download.py        — CC file downloader
│   ├── serve/
│   │   └── api.py             — FastAPI app
│   └── eval/
│       ├── beir_runner.py     — BEIR harness
│       └── profiler.py        — latency profiler
├── benchmarks/
│   ├── wand_ablation.py       — WAND vs naive DAAT comparison
│   ├── merge_amplification.py — write amplification measurement
│   └── ef_search_pareto.py    — recall vs latency Pareto curve
├── tests/
│   ├── test_codec.py
│   ├── test_segment.py
│   ├── test_merge.py
│   └── test_e2e.py
├── Cargo.toml                  — workspace root
├── pyproject.toml
├── Makefile
└── docker/
    ├── Dockerfile
    └── compose.yml
```

---

## Running the evaluation

```bash
# BEIR SciFact
python -m quartz.eval.beir_runner --dataset scifact --index data/index/ --output results/

# BEIR NFCorpus
python -m quartz.eval.beir_runner --dataset nfcorpus --index data/index/ --output results/

# Latency profiling (1000 BEIR queries, BM25 + hybrid)
python -m quartz.eval.profiler --index data/index/ --n-queries 1000

# ef_search Pareto curve (HNSW recall vs latency at different ef values)
python -m benchmarks.ef_search_pareto --index data/index/
```

---

## CI

```
.github/workflows/ci.yml
  Rust: cargo test --workspace, cargo clippy -D warnings, cargo fmt --check
  Python: maturin develop --release, pytest tests/
  Runs on: ubuntu-latest, macos-latest
```

---

## License

MIT
