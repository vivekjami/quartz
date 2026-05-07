// quartz-core/src/segment.rs
use crate::codec::{decode_postings, encode_postings};
use fst::{Map, MapBuilder};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

// ──────────────────────────────── DocMeta ────────────────────────────────────

/// Fixed-size 28-byte record per document. Random access by doc_id = O(1).
#[repr(C, packed)]
pub struct DocMeta {
    pub url_offset: u64,   // byte offset into docurls.bin
    pub url_len: u16,      // URL length in bytes
    pub crawl_ts: u64,     // Unix timestamp (seconds)
    pub content_hash: u64, // xxHash64 of raw text
    pub lang_id: u16,      // ISO 639-1 mapped to u16 (0=unknown, 1=en, ...)
}

pub const DOC_META_SIZE: usize = std::mem::size_of::<DocMeta>(); // must be 28

// ──────────────────────────────── MemSegment ─────────────────────────────────

/// In-memory segment. Accepts document writes until flush threshold is reached.
pub struct MemSegment {
    /// term_id → sorted list of (doc_id, term_freq)
    pub postings: BTreeMap<u32, Vec<(u32, u8)>>,
    /// term_string → term_id
    pub term_dict: BTreeMap<String, u32>,
    pub doc_metas: Vec<DocMeta>,
    pub doc_urls: Vec<u8>,          // packed URL bytes
    pub doc_lengths: Vec<u32>,      // number of tokens per doc
    pub next_term_id: u32,
    pub next_doc_id: u32,
    pub byte_size: usize,           // approximate RAM usage
    pub flush_threshold: usize,     // default: 50 * 1024 * 1024 (50MB)
}

impl MemSegment {
    pub fn new(flush_threshold: usize) -> Self {
        Self {
            postings: BTreeMap::new(),
            term_dict: BTreeMap::new(),
            doc_metas: Vec::new(),
            doc_urls: Vec::new(),
            doc_lengths: Vec::new(),
            next_term_id: 0,
            next_doc_id: 0,
            byte_size: 0,
            flush_threshold,
        }
    }

    /// Index a single document. Returns (doc_id, should_flush).
    pub fn add_doc(
        &mut self,
        url: &str,
        tokens: &[String],
        crawl_ts: u64,
        content_hash: u64,
    ) -> (u32, bool) {
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        // Term frequency map for this document
        let mut tf_map: BTreeMap<u32, u8> = BTreeMap::new();
        for token in tokens {
            let term_id = *self.term_dict.entry(token.clone()).or_insert_with(|| {
                let id = self.next_term_id;
                self.next_term_id += 1;
                id
            });
            let count = tf_map.entry(term_id).or_insert(0);
            *count = count.saturating_add(1);
        }

        // Update postings lists
        for (term_id, tf) in &tf_map {
            self.postings
                .entry(*term_id)
                .or_default()
                .push((doc_id, *tf));
        }

        // Store URL
        let url_offset = self.doc_urls.len() as u64;
        let url_bytes = url.as_bytes();
        self.doc_urls.extend_from_slice(url_bytes);

        // Store DocMeta
        self.doc_metas.push(DocMeta {
            url_offset,
            url_len: url_bytes.len() as u16,
            crawl_ts,
            content_hash,
            lang_id: 1, // simplified: assume English
        });
        self.doc_lengths.push(tokens.len() as u32);

        // Approximate byte tracking: 24 bytes overhead per postings entry
        self.byte_size += tokens.len() * 24;

        (doc_id, self.byte_size >= self.flush_threshold)
    }

    /// Flush to disk. Returns a DiskSegment handle.
    /// After flushing, this MemSegment should be discarded.
    pub fn flush(&self, dir: &Path, seg_id: u64) -> std::io::Result<DiskSegment> {
        let seg_dir = dir.join(format!("seg_{:08x}", seg_id));
        fs::create_dir_all(&seg_dir)?;

        // 1. Write postings.bin + build FST term dictionary
        let postings_path = seg_dir.join("postings.bin");
        let terms_path = seg_dir.join("terms.fst");
        let maxscores_path = seg_dir.join("maxscores.bin");

        let mut postings_file = BufWriter::new(File::create(&postings_path)?);

        // FST requires keys in sorted order. BTreeMap<String, u32> gives us
        // terms in lexicographic order. We need to map term_id → offset in postings.bin.
        // Build a reverse map: term_string (sorted) → (term_id, offset, doc_freq)
        let mut term_to_offset: BTreeMap<&str, (u32, u64, u32)> = BTreeMap::new();
        let mut max_scores: Vec<f32> = vec![0.0f32; self.next_term_id as usize];
        let total_docs = self.doc_lengths.len() as f32;
        let avg_dl = self.doc_lengths.iter().map(|&l| l as f64).sum::<f64>()
            / total_docs.max(1.0) as f64;

        let mut current_offset: u64 = 0;

        // Iterate term_dict in sorted-by-name order (BTreeMap guarantees this)
        // but we need sorted by term_string for FST. term_dict IS sorted by string.
        for (term_str, &term_id) in &self.term_dict {
            if let Some(posting_list) = self.postings.get(&term_id) {
                let doc_freq = posting_list.len() as u32;
                let mut doc_ids: Vec<u32> = posting_list.iter().map(|&(d, _)| d).collect();
                let mut tfs: Vec<u8> = posting_list.iter().map(|&(_, t)| t).collect();

                // Sort by doc_id (required for delta encoding)
                let mut pairs: Vec<(u32, u8)> = doc_ids.into_iter().zip(tfs).collect();
                pairs.sort_by_key(|&(d, _)| d);
                doc_ids = pairs.iter().map(|&(d, _)| d).collect();
                tfs = pairs.iter().map(|&(_, t)| t).collect();

                let encoded = encode_postings(&doc_ids, &tfs);
                postings_file.write_all(&encoded)?;

                // Compute WAND max score for this term
                let idf = ((total_docs - doc_freq as f32 + 0.5)
                    / (doc_freq as f32 + 0.5))
                    .ln()
                    .max(0.0);
                let max_tf = *tfs.iter().max().unwrap_or(&1) as f32;
                let min_dl = self
                    .doc_lengths
                    .iter()
                    .map(|&l| l as f32)
                    .fold(f32::INFINITY, f32::min)
                    .max(1.0);
                let k1 = 1.2f32;
                let b = 0.75f32;
                let tf_norm =
                    (max_tf * (k1 + 1.0)) / (max_tf + k1 * (1.0 - b + b * min_dl / avg_dl as f32));
                max_scores[term_id as usize] = idf * tf_norm;

                term_to_offset.insert(term_str.as_str(), (term_id, current_offset, doc_freq));
                current_offset += encoded.len() as u64;
            }
        }

        // 2. Build FST: key = term_str, value = packed u64 (term_id:32 | offset:32 for simplicity;
        //    for large indexes use a separate offset file)
        // FST value: pack term_id (u32) and doc_freq (u32) into u64; offset stored separately.
        // For this implementation: value = offset into postings.bin (u64 fits fst's u64 value).
        // We store term_id and doc_freq in a sidecar .terminfo file.
        let mut fst_builder = MapBuilder::new(BufWriter::new(File::create(&terms_path)?)).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut terminfo_file = BufWriter::new(File::create(seg_dir.join("terminfo.bin"))?);

        for (term_str, &(term_id, offset, doc_freq)) in &term_to_offset {
            // FST maps term_str → postings.bin byte offset
            fst_builder.insert(term_str.as_bytes(), offset).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            // terminfo.bin: indexed by term_id, stores doc_freq (4 bytes)
            // Note: term_ids are dense (0..next_term_id), so indexing directly works.
            let _ = term_id; // used below for max_scores
            terminfo_file.write_all(&doc_freq.to_le_bytes())?;
        }
        fst_builder.finish().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // 3. Write maxscores.bin
        let mut maxscores_file = BufWriter::new(File::create(&maxscores_path)?);
        for &score in &max_scores {
            maxscores_file.write_all(&score.to_le_bytes())?;
        }

        // 4. Write doclens.bin
        let mut doclens_file =
            BufWriter::new(File::create(seg_dir.join("doclens.bin"))?);
        for &dl in &self.doc_lengths {
            doclens_file.write_all(&dl.to_le_bytes())?;
        }

        // 5. Write docmeta.bin (fixed-size records)
        let mut docmeta_file =
            BufWriter::new(File::create(seg_dir.join("docmeta.bin"))?);
        for meta in &self.doc_metas {
            // Safety: DocMeta is repr(C, packed), all fields are integers. Safe to cast to bytes.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    meta as *const DocMeta as *const u8,
                    DOC_META_SIZE,
                )
            };
            docmeta_file.write_all(bytes)?;
        }

        // 6. Write docurls.bin
        fs::write(seg_dir.join("docurls.bin"), &self.doc_urls)?;

        // 7. Write meta.json
        let meta = SegmentMeta {
            seg_id,
            num_docs: self.doc_lengths.len() as u64,
            num_terms: self.next_term_id as u64,
            avg_doc_len: avg_dl,
            total_postings: self
                .postings
                .values()
                .map(|v| v.len() as u64)
                .sum(),
            merge_gen: 0,
        };
        let meta_json = serde_json::to_vec_pretty(&meta)?;
        fs::write(seg_dir.join("meta.json"), meta_json)?;

        
        // Explicitly flush and drop all writers before opening as DiskSegment.
        // BufWriters are dropped AFTER the return value is computed in Rust,
        // so without this, DiskSegment::open mmaps empty files.
        drop(postings_file);
        drop(maxscores_file);
        drop(doclens_file);
        drop(docmeta_file);
        // terminfo_file and fst_builder are already consumed/finished above


        DiskSegment::open(&seg_dir)
    }
}

// ──────────────────────────────── DiskSegment ────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub seg_id: u64,
    pub num_docs: u64,
    pub num_terms: u64,
    pub avg_doc_len: f64,
    pub total_postings: u64,
    pub merge_gen: u32, // number of merges this segment has been through
}

/// Read-only, mmap-backed segment. Safe to Clone (increments mmap refcount).
pub struct DiskSegment {
    pub dir: PathBuf,
    pub meta: SegmentMeta,
    pub postings_mmap: Mmap,     // postings.bin
    pub doclens_mmap: Mmap,      // doclens.bin (u32[])
    pub docmeta_mmap: Mmap,      // docmeta.bin (DocMeta[])
    pub docurls_mmap: Mmap,      // docurls.bin
    pub maxscores_mmap: Mmap,    // maxscores.bin (f32[])
    pub fst: Map<Vec<u8>>,       // FST term dictionary (fully loaded)
    pub size_bytes: u64,         // total size of all files
}

impl DiskSegment {
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        let meta: SegmentMeta =
            serde_json::from_slice(&fs::read(dir.join("meta.json"))?)?;

        let mmap = |name: &str| -> std::io::Result<Mmap> {
            let f = File::open(dir.join(name))?;
            // Safety: the file is read-only after flush. No concurrent writers.
            unsafe { Mmap::map(&f) }
        };

        let postings_mmap = mmap("postings.bin")?;
        let doclens_mmap = mmap("doclens.bin")?;
        let docmeta_mmap = mmap("docmeta.bin")?;
        let docurls_mmap = mmap("docurls.bin")?;
        let maxscores_mmap = mmap("maxscores.bin")?;

        let fst_bytes = fs::read(dir.join("terms.fst"))?;
        let fst = Map::new(fst_bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let size_bytes = ["postings.bin", "terms.fst", "doclens.bin",
                          "docmeta.bin", "docurls.bin", "maxscores.bin", "terminfo.bin"]
            .iter()
            .filter_map(|name| fs::metadata(dir.join(name)).ok())
            .map(|m| m.len())
            .sum();

        Ok(Self {
            dir: dir.to_path_buf(),
            meta,
            postings_mmap,
            doclens_mmap,
            docmeta_mmap,
            docurls_mmap,
            maxscores_mmap,
            fst,
            size_bytes,
        })
    }

    /// Look up a term's postings list offset in postings.bin.
    pub fn term_offset(&self, term: &str) -> Option<u64> {
        self.fst.get(term.as_bytes())
    }

    /// Decode the postings list for a term at a given byte offset.
    pub fn read_postings(&self, offset: u64) -> (Vec<u32>, Vec<u8>) {
        decode_postings(&self.postings_mmap[offset as usize..])
    }

    /// Get doc length by segment-local doc_id.
    pub fn doc_len(&self, doc_id: u32) -> u32 {
        let offset = (doc_id as usize) * 4;
        u32::from_le_bytes(
            self.doclens_mmap[offset..offset + 4].try_into().unwrap(),
        )
    }

    /// Get DocMeta by segment-local doc_id.
    pub fn doc_meta(&self, doc_id: u32) -> &DocMeta {
        let offset = (doc_id as usize) * DOC_META_SIZE;
        let bytes = &self.docmeta_mmap[offset..offset + DOC_META_SIZE];
        // Safety: DocMeta is repr(C, packed). Bytes are written by flush() above.
        unsafe { &*(bytes.as_ptr() as *const DocMeta) }
    }

    /// Get max WAND score for a term_id.
    pub fn max_score(&self, term_id: u32) -> f32 {
        let offset = (term_id as usize) * 4;
        if offset + 4 > self.maxscores_mmap.len() {
            return 0.0;
        }
        f32::from_le_bytes(
            self.maxscores_mmap[offset..offset + 4].try_into().unwrap(),
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn flush_and_reopen() {
        let dir = tempdir().unwrap();
        let mut mem = MemSegment::new(64 * 1024 * 1024);

        let tokens: Vec<String> = vec!["rust".into(), "fast".into(), "rust".into()];
        mem.add_doc("http://example.com", &tokens, 1234567890, 999);

        let seg = mem.flush(dir.path(), 0).unwrap();

        // doc 0 has 3 tokens
        assert_eq!(seg.doc_len(0), 3);

        // "rust" appears with tf=2
        let offset = seg.term_offset("rust").expect("rust must be indexed");
        let (ids, tfs) = seg.read_postings(offset);
        assert_eq!(ids, vec![0]);
        assert_eq!(tfs, vec![2]);

        // "fast" appears with tf=1
        let offset = seg.term_offset("fast").expect("fast must be indexed");
        let (ids, tfs) = seg.read_postings(offset);
        assert_eq!(ids, vec![0]);
        assert_eq!(tfs, vec![1]);

        // segment metadata
        assert_eq!(seg.meta.num_docs, 1);
        assert_eq!(seg.meta.num_terms, 2); // rust, fast
    }

    #[test]
    fn multi_doc_postings() {
        let dir = tempdir().unwrap();
        let mut mem = MemSegment::new(64 * 1024 * 1024);

        mem.add_doc("http://a.com", &["rust".into(), "search".into()], 0, 0);
        mem.add_doc("http://b.com", &["rust".into(), "python".into()], 1, 1);
        mem.add_doc("http://c.com", &["python".into(), "search".into()], 2, 2);

        let seg = mem.flush(dir.path(), 1).unwrap();

        // "rust" in docs 0 and 1
        let offset = seg.term_offset("rust").unwrap();
        let (ids, _) = seg.read_postings(offset);
        assert_eq!(ids, vec![0, 1]);

        // "python" in docs 1 and 2
        let offset = seg.term_offset("python").unwrap();
        let (ids, _) = seg.read_postings(offset);
        assert_eq!(ids, vec![1, 2]);

        assert_eq!(seg.meta.num_docs, 3);
    }
}