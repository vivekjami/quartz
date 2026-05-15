import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from quartz.quartz_core import IndexReader
from quartz.hnsw.graph import HNSWGraph
from quartz.serve.fusion import reciprocal_rank_fusion, freshness_factor

INDEX_DIR = Path("data/index/")
HNSW_DIR = Path("data/hnsw/")

# Prometheus metrics
QUERY_LATENCY = Histogram(
    "quartz_query_latency_seconds",
    "Query latency",
    buckets=[0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0],
)
QUERY_COUNT = Counter("quartz_queries_total", "Total queries")

reader: Optional[IndexReader] = None
hnsw: Optional[HNSWGraph] = None
embedder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global reader, hnsw, embedder
    reader = IndexReader(str(INDEX_DIR))
    if HNSW_DIR.exists():
        try:
            hnsw = HNSWGraph.load(HNSW_DIR)
            print(f"Loaded HNSW graph ({len(hnsw.vectors)} vectors)")
        except Exception as e:
            print(f"Warning: could not load HNSW graph: {e}")
            hnsw = None
    print(f"Index loaded: {reader.num_segments()} segment(s), {reader.num_docs()} docs")
    yield
    del reader


app = FastAPI(
    title="Quartz Search",
    description="Disk-based hybrid BM25+HNSW search engine built in Rust + Python",
    version="0.1.0",
    lifespan=lifespan,
)


def get_embedder():
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(10, ge=1, le=100, description="Number of results"),
    mode: str = Query("hybrid", regex="^(bm25|dense|hybrid)$", description="Search mode"),
    ef: int = Query(50, ge=10, le=500, description="HNSW ef_search parameter"),
):
    QUERY_COUNT.inc()
    t0 = time.perf_counter()

    bm25_results = []
    hnsw_results = []

    if mode in ("bm25", "hybrid"):
        bm25_results = reader.search(q, k=k * 2)  # over-fetch for fusion

    if mode in ("dense", "hybrid") and hnsw is not None:
        model = get_embedder()
        vec = model.encode(q, normalize_embeddings=True)
        hnsw_results = hnsw.search(vec, k=k * 2, ef=ef)
    
    if mode == "bm25":
        results = [(score, doc_id) for score, doc_id in bm25_results[:k]]
    elif mode == "dense":
        results = [(1.0 - dist, doc_id) for dist, doc_id in hnsw_results[:k]]
    else:
        # hybrid: merge BM25 and HNSW with Reciprocal Rank Fusion
        results = reciprocal_rank_fusion(bm25_results, hnsw_results)[:k]

    # Apply freshness decay and fetch doc metadata
    response_docs = []
    for score, doc_id in results:
        try:
            meta = reader.get_doc_meta(doc_id)
            fresh = freshness_factor(meta["crawl_ts"])
            final_score = score * (0.85 + 0.15 * fresh)  # 15% freshness weight
            response_docs.append({
                "url": meta["url"],
                "score": round(final_score, 6),
                "crawl_ts": meta["crawl_ts"],
                "freshness_factor": round(fresh, 4),
            })
        except Exception:
            continue

    latency_ms = (time.perf_counter() - t0) * 1000
    QUERY_LATENCY.observe(latency_ms / 1000)

    return JSONResponse({
        "query": q,
        "mode": mode,
        "results": response_docs,
        "latency_ms": round(latency_ms, 2),
        "num_segments": reader.num_segments(),
        "num_docs": reader.num_docs(),
    })


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "num_segments": reader.num_segments() if reader else 0,
        "num_docs": reader.num_docs() if reader else 0,
        "hnsw_loaded": hnsw is not None,
    }
