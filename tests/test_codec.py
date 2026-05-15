import pytest
from quartz.ingest.wet_reader import tokenize


def test_tokenize_basic():
    tokens = tokenize("Hello World! This is a test.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens


def test_tokenize_strips_punctuation():
    tokens = tokenize("rust-lang, python3.11: great!")
    # punctuation stripped, short tokens filtered
    for t in tokens:
        assert t.isalnum() or "_" in t


def test_tokenize_min_length():
    tokens = tokenize("a an to be or not")
    # all single-char / two-char tokens filtered (len < 2)
    for t in tokens:
        assert len(t) >= 2


def test_tokenize_max_length():
    long_word = "a" * 60
    tokens = tokenize(long_word)
    for t in tokens:
        assert len(t) <= 50


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []
