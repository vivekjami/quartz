use pyo3::prelude::*;
use std::path::PathBuf;

pub mod bm25;
pub mod codec;
pub mod merge;
pub mod segment;
pub mod wal;

use segment::{DiskSegment, MemSegment};

#[pyclass]
pub struct IndexWriter {
    index_dir: PathBuf,
    flush_threshold: usize,
    active_segment: MemSegment,
    next_seg_id: u64,
}

#[pymethods]
impl IndexWriter {
    #[new]
    fn new(index_dir: String, flush_threshold: usize) -> Self {
        IndexWriter {
            index_dir: PathBuf::from(index_dir),
            flush_threshold,
            active_segment: MemSegment::new(flush_threshold),
            next_seg_id: 0,
        }
    }

    fn add_doc(
        &mut self,
        url: &str,
        tokens: Vec<String>,
        crawl_ts: u64,
        content_hash: u64,
    ) -> PyResult<()> {
        let (_doc_id, should_flush) =
            self.active_segment
                .add_doc(url, &tokens, crawl_ts, content_hash);
        if should_flush {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> PyResult<()> {
        if self.active_segment.next_doc_id > 0 {
            self.active_segment
                .flush(&self.index_dir, self.next_seg_id)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            self.next_seg_id += 1;
            self.active_segment = MemSegment::new(self.flush_threshold);
        }
        Ok(())
    }

    fn run_merge(&mut self) -> PyResult<()> {
        Ok(())
    }
}

#[pyclass]
pub struct IndexReader {
    index_dir: PathBuf,
    segments: Vec<DiskSegment>,
}

#[pymethods]
impl IndexReader {
    #[new]
    fn new(index_dir: String) -> PyResult<Self> {
        let path = PathBuf::from(&index_dir);
        let mut segments = Vec::new();
        if path.exists() {
            if let Ok(entries) = std::fs::read_dir(&path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_dir() && p.file_name().unwrap().to_string_lossy().starts_with("seg_") {
                        if let Ok(seg) = DiskSegment::open(&p) {
                            segments.push(seg);
                        }
                    }
                }
            }
        }
        Ok(IndexReader {
            index_dir: path,
            segments,
        })
    }

    fn num_docs(&self) -> u64 {
        self.segments.iter().map(|s| s.meta.num_docs).sum()
    }

    fn num_segments(&self) -> usize {
        self.segments.len()
    }

    fn get_doc_meta<'py>(&self, py: Python<'py>, doc_id: u32) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::PyDict;
        let mut base = 0;
        for seg in &self.segments {
            let n = seg.meta.num_docs as u32;
            if doc_id >= base && doc_id < base + n {
                let local_id = doc_id - base;
                let meta = seg.doc_meta(local_id);
                let dict = PyDict::new(py);

                let url_start = meta.url_offset as usize;
                let url_end = url_start + meta.url_len as usize;
                let url = String::from_utf8_lossy(&seg.docurls_mmap[url_start..url_end]).to_string();

                dict.set_item("url", url)?;
                dict.set_item("crawl_ts", meta.crawl_ts)?;
                dict.set_item("content_hash", meta.content_hash)?;
                dict.set_item("lang_id", meta.lang_id)?;
                return Ok(dict);
            }
            base += n;
        }
        Err(pyo3::exceptions::PyValueError::new_err("Document not found"))
    }

    fn search(&self, q: String, k: usize) -> PyResult<Vec<(f32, u32)>> {
        let terms: Vec<String> = q.split_whitespace().map(|s| s.to_string()).collect();
        let scorer = bm25::Bm25Scorer::default();
        let mut all_results = Vec::new();

        let mut base = 0;
        for seg in &self.segments {
            let res = scorer.search(seg, &terms, k);
            for r in res {
                all_results.push((r.score, r.doc_id + base));
            }
            base += seg.meta.num_docs as u32;
        }

        all_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        all_results.truncate(k);
        Ok(all_results)
    }
}

#[pymodule]
fn quartz_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexWriter>()?;
    m.add_class::<IndexReader>()?;
    Ok(())
}
