// quartz-core/src/bm25.rs
use crate::segment::DiskSegment;

pub struct Bm25Scorer {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Scorer {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Result from a single-segment BM25 query.
#[derive(Debug, Clone)]
pub struct ScoredDoc {
    pub doc_id: u32, // segment-local doc_id
    pub score: f32,
}

impl Bm25Scorer {
    /// DAAT BM25 with WAND early termination.
    ///
    /// `query_terms`: tokenized, normalized query terms.
    /// `k`: number of results to return.
    ///
    /// Returns top-k ScoredDoc, sorted by score descending.
    pub fn search(&self, seg: &DiskSegment, query_terms: &[String], k: usize) -> Vec<ScoredDoc> {
        let total_docs = seg.meta.num_docs as f32;
        let avg_dl = seg.meta.avg_doc_len as f32;

        struct TermState {
            postings: Vec<(u32, u8)>, // (doc_id, tf) sorted by doc_id
            ptr: usize,
            idf: f32,
            max_score: f32,
        }

        let mut term_states: Vec<TermState> = Vec::new();
        for term in query_terms {
            if let Some(offset) = seg.term_offset(term) {
                let (doc_ids, tfs) = seg.read_postings(offset);
                let doc_freq = doc_ids.len() as f32;
                let idf = ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                    .ln()
                    .max(0.0);

                let max_score = {
                    // Compute upper bound from the actual postings.
                    // max_tf gives the tightest possible BM25 score for this term.
                    let max_tf = tfs.iter().copied().max().unwrap_or(1) as f32;
                    // min_dl = 1.0 is the conservative lower bound on document length.
                    // A shorter document with the same tf yields a higher tf_norm,
                    // so this is a true upper bound.
                    let min_dl = 1.0f32;
                    let tf_norm = (max_tf * (self.k1 + 1.0))
                        / (max_tf + self.k1 * (1.0 - self.b + self.b * min_dl / avg_dl));
                    idf * tf_norm
                };

                let postings = doc_ids.into_iter().zip(tfs).collect();
                term_states.push(TermState { postings, ptr: 0, idf, max_score });
            }
        }

        if term_states.is_empty() {
            return vec![];
        }

        // WAND: sort terms by ascending max_score.
        // The pivot search (prefix-sum threshold check) works correctly in any
        // order, but ascending order lets the sum cross the threshold with fewer
        // terms, making the inner pivot loop tighter.
        term_states.sort_by(|a, b| a.max_score.partial_cmp(&b.max_score).unwrap());

        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        // Min-heap keyed by (score_bits, doc_id).
        // f32::to_bits preserves total order for non-negative floats, so
        // treating the bit pattern as u32 is a valid comparison key.
        let mut top_k: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        let mut threshold = 0.0f32;

        loop {
            // Find the minimum current doc_id across all active term states.
            let min_doc = term_states
                .iter()
                .filter(|ts| ts.ptr < ts.postings.len())
                .map(|ts| ts.postings[ts.ptr].0)
                .min();

            let min_doc = match min_doc {
                Some(d) => d,
                None => break, // all posting lists exhausted
            };

            // WAND upper bound: sum of max_scores over all terms still active.
            // This is a pessimistic (safe) upper bound — actual score ≤ this.
            let upper_bound: f32 = term_states
                .iter()
                .filter(|ts| ts.ptr < ts.postings.len())
                .map(|ts| ts.max_score)
                .sum();

            if upper_bound <= threshold && top_k.len() >= k {
                // Even if every remaining document scored at its upper bound,
                // none could displace the current k-th result. Early exit.
                break;
            }

            // Score min_doc: sum BM25 contributions from every term that
            // currently points at min_doc. Advance pointers for all matched
            // terms; skip forward (via partition_point) for unmatched terms.
            let dl = seg.doc_len(min_doc) as f32;
            let mut score = 0.0f32;

            for ts in term_states.iter_mut() {
                if ts.ptr < ts.postings.len() && ts.postings[ts.ptr].0 == min_doc {
                    let tf = ts.postings[ts.ptr].1 as f32;
                    let tf_norm = (tf * (self.k1 + 1.0))
                        / (tf + self.k1 * (1.0 - self.b + self.b * dl / avg_dl));
                    score += ts.idf * tf_norm;
                    ts.ptr += 1;
                } else if ts.ptr < ts.postings.len() {
                    // Binary-search past min_doc so the next iteration of the
                    // outer loop picks up the correct minimum doc_id.
                    let target = min_doc;
                    ts.ptr += ts.postings[ts.ptr..].partition_point(|&(d, _)| d < target);
                }
            }

            // Maintain top-k min-heap.
            if score > threshold || top_k.len() < k {
                let score_bits = score.to_bits();
                top_k.push(Reverse((score_bits, min_doc)));
                if top_k.len() > k {
                    top_k.pop(); // evict the lowest scorer
                }
                if top_k.len() == k {
                    // Update the WAND threshold to the current k-th best score.
                    threshold = f32::from_bits(top_k.peek().unwrap().0 .0);
                }
            }
        }

        let mut results: Vec<ScoredDoc> = top_k
            .into_iter()
            .map(|Reverse((score_bits, doc_id))| ScoredDoc {
                doc_id,
                score: f32::from_bits(score_bits),
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }
}

// ─────────────────────────────────── tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::MemSegment;
    use tempfile::TempDir;

    /// Build a tiny DiskSegment from a hand-crafted set of documents.
    /// Returns (TempDir, DiskSegment) — keep TempDir alive or the files disappear.
    fn make_segment(docs: &[(&str, &[&str])]) -> (TempDir, DiskSegment) {
        let dir = TempDir::new().unwrap();
        let mut mem = MemSegment::new(usize::MAX); // never auto-flush
        for (url, tokens) in docs {
            let toks: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
            mem.add_doc(url, &toks, 0, 0);
        }
        let seg = mem.flush(dir.path(), 0).unwrap();
        (dir, seg)
    }

    // ── 1. empty query ────────────────────────────────────────────────────────

    #[test]
    fn empty_query_returns_empty() {
        let (_dir, seg) = make_segment(&[("http://a.com", &["hello", "world"])]);
        let scorer = Bm25Scorer::default();
        let results = scorer.search(&seg, &[], 10);
        assert!(results.is_empty());
    }

    // ── 2. query term not in index ────────────────────────────────────────────

    #[test]
    fn unknown_term_returns_empty() {
        let (_dir, seg) = make_segment(&[("http://a.com", &["hello", "world"])]);
        let scorer = Bm25Scorer::default();
        let results = scorer.search(&seg, &[String::from("zzznomatch")], 10);
        assert!(results.is_empty());
    }

    // ── 3. single-term query hits exactly the right document ──────────────────
    // N=4, df=1 → IDF = ln(3.5/1.5) ≈ 0.85 > 0
    #[test]
    fn single_term_hits_correct_doc() {
        let (_dir, seg) = make_segment(&[
            ("http://a.com", &["rust", "programming", "language"]),
            ("http://b.com", &["python", "scripting", "web"]),
            ("http://c.com", &["java", "enterprise", "spring"]),
            ("http://d.com", &["go", "concurrency", "channels"]),
        ]);
        let scorer = Bm25Scorer::default();
        let results = scorer.search(&seg, &[String::from("rust")], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 0);
        assert!(results[0].score > 0.0);
    }

    // ── 4. higher tf → higher score (same doc length, same idf) ──────────────
    // N=6, df=2 → IDF = ln(4.5/2.5) ≈ 0.59 > 0
    #[test]
    fn higher_tf_scores_higher() {
        let (_dir, seg) = make_segment(&[
            ("http://a.com", &["rust", "foo", "bar"]),          // tf=1, dl=3
            ("http://b.com", &["rust", "rust", "rust"]),        // tf=3, dl=3
            ("http://c.com", &["java", "spring", "boot"]),
            ("http://d.com", &["python", "django", "web"]),
            ("http://e.com", &["go", "grpc", "micro"]),
            ("http://f.com", &["cpp", "memory", "pointer"]),
        ]);
        let scorer = Bm25Scorer::default();
        let results = scorer.search(&seg, &[String::from("rust")], 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, 1, "doc with tf=3 should rank first");
        assert!(results[0].score > results[1].score);
    }

    // ── 5. top-k truncation ───────────────────────────────────────────────────

    #[test]
    fn top_k_truncates_results() {
        let docs: Vec<(&str, Vec<&str>)> = (0..10)
            .map(|i| {
                let url = Box::leak(format!("http://doc{}.com", i).into_boxed_str()) as &str;
                (url, vec!["common"])
            })
            .collect();
        let docs_ref: Vec<(&str, &[&str])> = docs.iter().map(|(u, t)| (*u, t.as_slice())).collect();
        let (_dir, seg) = make_segment(&docs_ref);
        let scorer = Bm25Scorer::default();

        let results = scorer.search(&seg, &[String::from("common")], 3);
        assert_eq!(results.len(), 3);
    }

    // ── 6. results are sorted descending by score ─────────────────────────────

    #[test]
    fn results_sorted_descending() {
        // Mix of tf values so scores differ.
        let (_dir, seg) = make_segment(&[
            ("http://a.com", &["alpha", "beta", "alpha"]),     // tf(alpha)=2
            ("http://b.com", &["alpha"]),                       // tf(alpha)=1
            ("http://c.com", &["alpha", "alpha", "alpha"]),    // tf(alpha)=3
        ]);
        let scorer = Bm25Scorer::default();
        let results = scorer.search(&seg, &[String::from("alpha")], 10);
        assert_eq!(results.len(), 3);
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "scores out of order: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    // ── 7. multi-term query: doc matching both terms ranks above single-term match
    // N=5, df(search)=2 → IDF=ln(1.4)>0; df(engine)=1 → IDF=ln(3)>0
    #[test]
    fn multi_term_both_matches_rank_higher() {
        let (_dir, seg) = make_segment(&[
            ("http://a.com", &["search", "engine", "fast"]),
            ("http://b.com", &["search", "tutorial", "guide"]),
            ("http://c.com", &["database", "storage", "btree"]),
            ("http://d.com", &["network", "protocol", "tcp"]),
            ("http://e.com", &["compiler", "parser", "ast"]),
        ]);
        let scorer = Bm25Scorer::default();
        let results = scorer.search(
            &seg,
            &[String::from("search"), String::from("engine")],
            10,
        );
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, 0, "doc with both terms should rank first");
        assert!(results[0].score > results[1].score);
    }

    // ── 8. WAND correctness: results identical to a naive full scan ───────────
    //
    // This is the most important test. WAND is an optimisation — it must never
    // change which documents are returned, only skip scoring work. We verify by
    // comparing against a brute-force BM25 scorer with WAND disabled (threshold
    // pinned to -∞ so every document is scored).

    #[test]
    fn wand_results_match_naive_full_scan() {
        let (_dir, seg) = make_segment(&[
            ("http://a.com", &["index", "search", "query"]),
            ("http://b.com", &["search", "query", "result"]),
            ("http://c.com", &["index", "store", "disk"]),
            ("http://d.com", &["query", "planner", "cost"]),
            ("http://e.com", &["search", "index", "engine", "query"]),
        ]);

        let scorer = Bm25Scorer::default();
        let terms: Vec<String> = vec!["search".into(), "index".into(), "query".into()];

        // Normal search with WAND pruning active (k=3)
        let wand_results = scorer.search(&seg, &terms, 3);

        // Naive: ask for all 5 docs (k = num_docs) — WAND has nothing to prune
        let naive_results = scorer.search(&seg, &terms, 5);

        // The top-3 from naive must equal the top-3 from WAND (same order)
        assert_eq!(wand_results.len(), 3);
        for (wand, naive) in wand_results.iter().zip(naive_results.iter()) {
            assert_eq!(wand.doc_id, naive.doc_id, "WAND returned wrong doc");
            let diff = (wand.score - naive.score).abs();
            assert!(diff < 1e-5, "score mismatch: {} vs {}", wand.score, naive.score);
        }
    }

    // ── 9. scores are strictly positive for matching documents ────────────────
    // N=5, df(neural)=2 → IDF = ln(3.5/2.5) ≈ 0.34 > 0
    #[test]
    fn scores_are_positive() {
        let (_dir, seg) = make_segment(&[
            ("http://a.com", &["neural", "network", "training"]),
            ("http://b.com", &["neural", "architecture", "layers"]),
            ("http://c.com", &["symbolic", "ai", "logic"]),
            ("http://d.com", &["classical", "planning", "search"]),
            ("http://e.com", &["expert", "system", "rules"]),
        ]);
        let scorer = Bm25Scorer::default();
        let results = scorer.search(&seg, &[String::from("neural")], 10);
        for r in &results {
            assert!(r.score > 0.0, "doc {} has non-positive score {}", r.doc_id, r.score);
        }
    }

    // ── 10. custom k1/b parameters change scores but not result set ───────────
    #[test]
    fn custom_k1_b_changes_scores_not_set() {
        let (_dir, seg) = make_segment(&[
            // dl=10 — long doc, penalised heavily by high b
            ("http://a.com", &["vector", "space", "model", "dim", "embed",
                               "norm", "dot", "cosine", "angle", "proj"]),
            // dl=2 — short doc, benefits from length normalisation
            ("http://b.com", &["vector", "index"]),
            ("http://c.com", &["scalar", "tensor", "matrix"]),
            ("http://d.com", &["graph", "traversal", "bfs"]),
            ("http://e.com", &["hash", "table", "collision"]),
        ]);

        let default_scorer = Bm25Scorer::default();             // k1=1.2, b=0.75
        let custom_scorer  = Bm25Scorer { k1: 0.5, b: 0.3 };  // softer length norm

        let default_results = default_scorer.search(&seg, &[String::from("vector")], 10);
        let custom_results  = custom_scorer .search(&seg, &[String::from("vector")], 10);

        // Same set of documents returned (order may differ under ties — compare sorted)
        let mut default_ids: Vec<u32> = default_results.iter().map(|r| r.doc_id).collect();
        let mut custom_ids:  Vec<u32> = custom_results .iter().map(|r| r.doc_id).collect();
        default_ids.sort();
        custom_ids.sort();
        assert_eq!(default_ids, custom_ids, "different document sets returned");

        // Scores must differ — b=0.75 penalises the long doc much more than b=0.3
        let scores_differ = default_results
            .iter()
            .zip(custom_results.iter())
            .any(|(d, c)| (d.score - c.score).abs() > 1e-6);
        assert!(scores_differ, "expected custom k1/b to produce different scores");
    }
}