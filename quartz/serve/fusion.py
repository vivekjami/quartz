import math
from datetime import datetime, timezone


def reciprocal_rank_fusion(
    bm25_results: list[tuple[float, int]],   # (score, doc_id)
    hnsw_results: list[tuple[float, int]],   # (distance, doc_id)
    k: int = 60,
) -> list[tuple[float, int]]:
    """
    RRF (Cormack et al., 2009). k=60 is empirically validated across TREC tasks.
    score(d) = Σ 1/(k + rank(d))
    """
    scores: dict[int, float] = {}
    for rank, (_, doc_id) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (_, doc_id) in enumerate(hnsw_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def freshness_factor(crawl_ts: int, decay_days: float = 30.0) -> float:
    """
    Multiplicative freshness boost. Exponential decay.
    Fresh docs (crawled today) → factor ≈ 1.0
    30-day-old docs → factor ≈ 0.37 (1/e)
    90-day-old docs → factor ≈ 0.05

    Multiplicative (not additive) because BM25 scores are corpus-dependent;
    additive freshness would give uniform absolute boosts across very different
    score scales.
    """
    age_hours = (datetime.now(timezone.utc).timestamp() - crawl_ts) / 3600
    return math.exp(-age_hours / (decay_days * 24))
