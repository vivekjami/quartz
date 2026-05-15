import pytest
from quartz.quartz_core import IndexWriter, IndexReader
import tempfile, os


def test_indexwriter_creates_segment():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = IndexWriter(tmpdir, 64 * 1024 * 1024)
        writer.add_doc(
            "http://example.com",
            ["rust", "python", "search", "engine"],
            1_700_000_000,
            12345,
        )
        writer.flush()

        # A segment directory should now exist
        segments = [d for d in os.listdir(tmpdir) if d.startswith("seg_")]
        assert len(segments) == 1


def test_indexreader_searches():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = IndexWriter(tmpdir, 64 * 1024 * 1024)
        writer.add_doc(
            "http://rust-lang.org",
            ["rust", "systems", "programming", "memory", "safety"],
            1_700_000_000,
            1,
        )
        writer.add_doc(
            "http://python.org",
            ["python", "scripting", "programming", "dynamic"],
            1_700_000_001,
            2,
        )
        writer.flush()

        reader = IndexReader(tmpdir)
        assert reader.num_docs() == 2
        assert reader.num_segments() == 1

        results = reader.search("rust programming", k=5)
        assert len(results) > 0
        # First result should be the rust doc (doc_id=0)
        assert results[0][1] == 0


def test_indexreader_get_doc_meta():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = IndexWriter(tmpdir, 64 * 1024 * 1024)
        writer.add_doc("http://example.com/page", ["test", "document"], 9999, 42)
        writer.flush()

        reader = IndexReader(tmpdir)
        meta = reader.get_doc_meta(0)
        assert meta["url"] == "http://example.com/page"
        assert meta["crawl_ts"] == 9999
