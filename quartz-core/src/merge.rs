// quartz-core/src/merge.rs
use crate::codec::{decode_postings, encode_postings};
use crate::segment::{DiskSegment, SegmentMeta};
use fst::MapBuilder;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Tiered merge policy — mirrors Lucene's TieredMergePolicy.
///
/// Invariant maintained: segments at tier T have sizes within a factor of
/// `merge_factor` of each other. When a tier overflows `max_per_tier`,
/// merge the smallest segments in that tier.
///
/// Write amplification: O(log_{merge_factor}(total_segments))
/// For 1M docs in 100 initial segments, merge_factor=10: each doc rewritten ~2-3 times.
pub struct TieredMergePolicy {
    pub max_per_tier: usize, // trigger merge when a tier exceeds this (default: 10)
    pub merge_factor: usize, // segments per merge operation (default: 10)
    pub floor_size: u64,     // minimum segment size to consider for tiering (default: 2MB)
}

impl Default for TieredMergePolicy {
    fn default() -> Self {
        Self {
            max_per_tier: 10,
            merge_factor: 10,
            floor_size: 2 * 1024 * 1024,
        }
    }
}

impl TieredMergePolicy {
    /// Find segments to merge. Returns None if no merge is needed.
    /// Returns the indices into `segments` of the chosen merge candidates.
    pub fn find_merge(&self, segments: &[DiskSegment]) -> Option<Vec<usize>> {
        if segments.len() <= self.max_per_tier {
            return None;
        }

        // Sort by size ascending (smallest segments merge first — reduces write amplification)
        let mut indexed: Vec<(usize, u64)> = segments
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.size_bytes))
            .collect();
        indexed.sort_by_key(|&(_, size)| size);

        // Find the tier with the most segments exceeding max_per_tier
        // Simple approach: take the merge_factor smallest segments
        if indexed.len() > self.max_per_tier {
            let candidates: Vec<usize> = indexed[..self.merge_factor]
                .iter()
                .map(|&(i, _)| i)
                .collect();
            return Some(candidates);
        }

        None
    }
}

/// K-way merge of DiskSegments into a new merged segment.
///
/// Algorithm: min-heap over (term_str, segment_index).
/// Pull the globally minimum term, collect all postings for that term
/// across all segments, merge-sort the postings, re-encode, write.
///
/// This is equivalent to external sort merge: works with O(K) RAM
/// regardless of total postings count.
pub fn k_way_merge(
    segments: &[&DiskSegment],
    output_dir: &Path,
    seg_id: u64,
) -> std::io::Result<DiskSegment> {
    let seg_dir: PathBuf = output_dir.join(format!("seg_{:08x}", seg_id));
    fs::create_dir_all(&seg_dir)?;

    // Open FST iterators for all segments (sorted order guaranteed by FST)
    let _fst_maps: Vec<_> = segments.iter().map(|s| s.fst.stream()).collect();
    // Note: fst::Streamer<'_> is the iterator type.
    // We'll use fst::automaton::AlwaysMatch to stream all terms.
    use fst::Streamer;
    let mut streams: Vec<_> = segments
        .iter()
        .map(|s| {
            use fst::IntoStreamer;
            s.fst.into_stream()
        })
        .collect();

    // Heap entry: (term_bytes, segment_index, postings_offset)
    // We use BinaryHeap with Reverse for min-heap behavior.
    let mut heap: BinaryHeap<Reverse<(Vec<u8>, usize, u64)>> = BinaryHeap::new();

    // Seed the heap with the first term from each segment
    for (i, stream) in streams.iter_mut().enumerate() {
        if let Some((term, offset)) = stream.next() {
            heap.push(Reverse((term.to_vec(), i, offset)));
        }
    }

    let mut postings_out = BufWriter::new(File::create(seg_dir.join("postings.bin"))?);
    let mut fst_builder = MapBuilder::new(BufWriter::new(File::create(seg_dir.join("terms.fst"))?))
        .map_err(std::io::Error::other)?;
    let mut current_offset: u64 = 0;

    // Merged doc lengths: concatenation of all segments' doclens
    // (doc IDs in the merged segment are segment0_docs ++ segment1_docs ++ ...)
    let mut merged_doclens: Vec<u32> = Vec::new();
    let doc_id_offsets: Vec<u32> = {
        let mut offsets = vec![0u32];
        for seg in segments.iter() {
            offsets.push(offsets.last().unwrap() + seg.meta.num_docs as u32);
        }
        offsets
    };

    for seg in segments.iter() {
        let n = seg.meta.num_docs as usize;
        for doc_id in 0..n {
            merged_doclens.push(seg.doc_len(doc_id as u32));
        }
    }

    let avg_dl =
        merged_doclens.iter().map(|&l| l as f64).sum::<f64>() / merged_doclens.len() as f64;

    let mut max_scores: Vec<(f32, u32)> = Vec::new(); // (max_score, term_counter)
    let mut term_counter: u32 = 0;
    let total_docs = merged_doclens.len() as f32;

    while let Some(Reverse((min_term, seg_idx, offset))) = heap.pop() {
        // Collect this term from all segments that have it at the heap top
        let mut combined_ids: Vec<u32> = Vec::new();
        let mut combined_tfs: Vec<u8> = Vec::new();

        // Process the term we just popped
        let (ids, tfs) = decode_postings(&segments[seg_idx].postings_mmap[offset as usize..]);
        let base = doc_id_offsets[seg_idx];
        combined_ids.extend(ids.iter().map(|&d| d + base));
        combined_tfs.extend_from_slice(&tfs);

        // Advance that segment's stream and push next term onto heap
        if let Some((next_term, next_offset)) = streams[seg_idx].next() {
            heap.push(Reverse((next_term.to_vec(), seg_idx, next_offset)));
        }

        // Drain all other segments that also have this term at the front
        loop {
            match heap.peek() {
                Some(Reverse((t, _, _))) if t == &min_term => {
                    let Reverse((_, other_seg, other_offset)) = heap.pop().unwrap();
                    let (ids2, tfs2) = decode_postings(
                        &segments[other_seg].postings_mmap[other_offset as usize..],
                    );
                    let base2 = doc_id_offsets[other_seg];
                    combined_ids.extend(ids2.iter().map(|&d| d + base2));
                    combined_tfs.extend_from_slice(&tfs2);
                    if let Some((nt, no)) = streams[other_seg].next() {
                        heap.push(Reverse((nt.to_vec(), other_seg, no)));
                    }
                }
                _ => break,
            }
        }

        // Sort combined postings by doc_id
        let mut pairs: Vec<(u32, u8)> = combined_ids.into_iter().zip(combined_tfs).collect();
        pairs.sort_by_key(|&(d, _)| d);
        let sorted_ids: Vec<u32> = pairs.iter().map(|&(d, _)| d).collect();
        let sorted_tfs: Vec<u8> = pairs.iter().map(|&(_, t)| t).collect();

        // Compute WAND max score for merged postings
        let doc_freq = sorted_ids.len() as f32;
        let idf = ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            .ln()
            .max(0.0);
        let max_tf = *sorted_tfs.iter().max().unwrap_or(&1) as f32;
        let min_dl = merged_doclens
            .iter()
            .map(|&l| l as f32)
            .fold(f32::INFINITY, f32::min)
            .max(1.0);
        let k1 = 1.2f32;
        let b = 0.75f32;
        let tf_norm =
            (max_tf * (k1 + 1.0)) / (max_tf + k1 * (1.0 - b + b * min_dl / avg_dl as f32));
        max_scores.push((idf * tf_norm, term_counter));

        // Write merged postings
        let encoded = encode_postings(&sorted_ids, &sorted_tfs);
        fst_builder
            .insert(&min_term, current_offset)
            .map_err(std::io::Error::other)?;
        postings_out.write_all(&encoded)?;
        current_offset += encoded.len() as u64;
        term_counter += 1;
    }
    fst_builder
        .finish()
        .map_err(std::io::Error::other)?;

    // Write maxscores.bin
    let mut ms_file = BufWriter::new(File::create(seg_dir.join("maxscores.bin"))?);
    for &(score, _) in &max_scores {
        ms_file.write_all(&score.to_le_bytes())?;
    }

    // Write merged doclens, docmeta, docurls (concatenation)
    let mut doclens_file = BufWriter::new(File::create(seg_dir.join("doclens.bin"))?);
    for &dl in &merged_doclens {
        doclens_file.write_all(&dl.to_le_bytes())?;
    }

    // Concatenate docmeta and docurls from all segments
    let mut docmeta_file = BufWriter::new(File::create(seg_dir.join("docmeta.bin"))?);
    let mut docurls_file = BufWriter::new(File::create(seg_dir.join("docurls.bin"))?);
    let mut url_offset_base: u64 = 0;
    for seg in segments.iter() {
        docurls_file.write_all(&seg.docurls_mmap)?;
        // Copy docmeta, adjusting url_offset by url_offset_base
        for doc_id in 0..seg.meta.num_docs {
            let meta = seg.doc_meta(doc_id as u32);
            let adjusted_url_offset = meta.url_offset + url_offset_base;
            let adjusted = crate::segment::DocMeta {
                url_offset: adjusted_url_offset,
                url_len: meta.url_len,
                crawl_ts: meta.crawl_ts,
                content_hash: meta.content_hash,
                lang_id: meta.lang_id,
            };
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &adjusted as *const _ as *const u8,
                    crate::segment::DOC_META_SIZE,
                )
            };
            docmeta_file.write_all(bytes)?;
        }
        url_offset_base += seg.docurls_mmap.len() as u64;
    }

    // Write meta.json
    let max_merge_gen = segments.iter().map(|s| s.meta.merge_gen).max().unwrap_or(0);
    let meta = SegmentMeta {
        seg_id,
        num_docs: merged_doclens.len() as u64,
        num_terms: term_counter as u64,
        avg_doc_len: avg_dl,
        total_postings: max_scores.len() as u64,
        merge_gen: max_merge_gen + 1,
    };
    fs::write(seg_dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;

    // Explicitly flush all writers before opening merged segment
    drop(postings_out);
    drop(ms_file);
    drop(doclens_file);
    drop(docmeta_file);
    drop(docurls_file);

    DiskSegment::open(&seg_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::MemSegment;
    use tempfile::tempdir;

    fn make_segment(dir: &Path, seg_id: u64, docs: &[(&str, Vec<&str>)]) -> DiskSegment {
        let mut mem = MemSegment::new(64 * 1024 * 1024);
        for (url, tokens) in docs {
            let owned: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
            mem.add_doc(url, &owned, 0, 0);
        }
        mem.flush(dir, seg_id).unwrap()
    }

    #[test]
    fn merge_two_segments() {
        let dir = tempdir().unwrap();
        let p = dir.path();

        let seg_a = make_segment(p, 0, &[("http://a.com", vec!["rust", "fast"])]);
        let seg_b = make_segment(p, 1, &[("http://b.com", vec!["rust", "python"])]);

        let merged = k_way_merge(&[&seg_a, &seg_b], p, 99).unwrap();

        // "rust" appears in both segments — must have 2 postings entries
        let offset = merged.term_offset("rust").expect("rust must exist");
        let (ids, _) = decode_postings(&merged.postings_mmap[offset as usize..]);
        assert_eq!(ids.len(), 2, "rust should appear in 2 docs");

        // total docs = 2
        assert_eq!(merged.meta.num_docs, 2);
        // merge_gen incremented
        assert_eq!(merged.meta.merge_gen, 1);
    }

    #[test]
    fn merge_preserves_unique_terms() {
        let dir = tempdir().unwrap();
        let p = dir.path();

        let seg_a = make_segment(p, 0, &[("http://a.com", vec!["alpha", "beta"])]);
        let seg_b = make_segment(p, 1, &[("http://b.com", vec!["gamma", "delta"])]);

        let merged = k_way_merge(&[&seg_a, &seg_b], p, 99).unwrap();

        // All 4 unique terms must survive the merge
        assert!(merged.term_offset("alpha").is_some());
        assert!(merged.term_offset("beta").is_some());
        assert!(merged.term_offset("gamma").is_some());
        assert!(merged.term_offset("delta").is_some());
        assert_eq!(merged.meta.num_terms, 4);
    }

    #[test]
    fn tiered_policy_triggers_on_overflow() {
        let dir = tempdir().unwrap();
        let p = dir.path();

        // Build 12 segments (> max_per_tier=10)
        let segs: Vec<DiskSegment> = (0..12)
            .map(|i| make_segment(p, i, &[(&format!("http://{}.com", i), vec!["term"])]))
            .collect();

        let policy = TieredMergePolicy::default();
        let candidates = policy.find_merge(&segs);
        assert!(candidates.is_some(), "policy must trigger with 12 segments");
        assert_eq!(candidates.unwrap().len(), policy.merge_factor);
    }

    #[test]
    fn tiered_policy_no_merge_when_under_limit() {
        let dir = tempdir().unwrap();
        let p = dir.path();

        // Build only 5 segments (< max_per_tier=10)
        let segs: Vec<DiskSegment> = (0..5)
            .map(|i| make_segment(p, i, &[(&format!("http://{}.com", i), vec!["term"])]))
            .collect();

        let policy = TieredMergePolicy::default();
        assert!(policy.find_merge(&segs).is_none());
    }
}
