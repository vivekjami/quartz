import tempfile
from pathlib import Path

from quartz.ingest.simhash import SimHashDeduplicator


def test_fingerprint_deterministic():
    deduper = SimHashDeduplicator()
    fp1 = deduper.fingerprint("hello world this is a test")
    fp2 = deduper.fingerprint("hello world this is a test")
    assert fp1 == fp2


def test_exact_duplicate_detected():
    deduper = SimHashDeduplicator()
    text = "the quick brown fox jumps over the lazy dog"
    assert deduper.check_and_add(text) is False  # first time: not a dupe
    assert deduper.check_and_add(text) is True   # second time: duplicate


def test_different_text_not_duplicate():
    deduper = SimHashDeduplicator()
    text_a = "machine learning and neural networks"
    text_b = "climate change and renewable energy sources in the future"
    deduper.check_and_add(text_a)
    assert deduper.check_and_add(text_b) is False


def test_dupe_rate():
    deduper = SimHashDeduplicator()
    text = "duplicate content here"
    deduper.check_and_add(text)
    deduper.check_and_add(text)
    deduper.check_and_add(text)
    # 2 out of 3 are dupes
    assert deduper.dupe_rate > 0.5


def test_hamming():
    assert SimHashDeduplicator.hamming(0b1010, 0b1010) == 0
    assert SimHashDeduplicator.hamming(0b1010, 0b0101) == 4
