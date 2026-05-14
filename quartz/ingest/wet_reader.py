import gzip
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

@dataclass
class WETDocument:
    url: str
    text: str
    crawl_ts: int   # Unix timestamp
    lang: str       # detected language (simplified)

# Minimal WARC/WET record parser — no dependencies beyond stdlib
def _parse_wet(f) -> Iterator[dict]:
    """Parse a WET file, yielding raw record dicts."""
    headers: dict[str, str] = {}
    in_header = True
    body_lines: list[str] = []
    content_length = 0

    for raw_line in f:
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace")
        line = line.rstrip("\r\n")

        if in_header:
            if line == "":
                in_header = False
                content_length = int(headers.get("content-length", 0))
                body_lines = []
            elif ":" in line:
                key, _, val = line.partition(":")
                headers[key.strip().lower()] = val.strip()
        else:
            body_lines.append(line)
            if sum(len(l) + 1 for l in body_lines) >= content_length:
                record_type = headers.get("warc-type", "")
                if record_type == "conversion":
                    yield {
                        "url": headers.get("warc-target-uri", ""),
                        "ts": headers.get("warc-date", ""),
                        "body": "\n".join(body_lines),
                    }
                headers = {}
                body_lines = []
                in_header = True

def _ts_to_unix(ts_str: str) -> int:
    """Parse WARC-Date '2025-01-15T14:22:01Z' → Unix timestamp."""
    import calendar
    from datetime import datetime, timezone
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return int(time.time())

_WHITESPACE = re.compile(r"\s+")

def tokenize(text: str) -> list[str]:
    """
    Normalize and tokenize text to a list of lowercase tokens.
    Strips punctuation, folds whitespace, limits to 50 chars per token.
    Not a linguistic tokenizer — fast and sufficient for BM25.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # strip punctuation
    tokens = _WHITESPACE.split(text.strip())
    return [t for t in tokens if 2 <= len(t) <= 50]

def iter_wet_file(path: Path) -> Iterator[WETDocument]:
    """Stream documents from a single WET .gz file."""
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8", errors="replace") as f:
        for record in _parse_wet(f):
            url = record["url"]
            if not url.startswith("http"):
                continue
            text = record["body"].strip()
            if len(text) < 100:  # skip boilerplate stubs
                continue
            yield WETDocument(
                url=url,
                text=text,
                crawl_ts=_ts_to_unix(record["ts"]),
                lang="en",  # production: use langdetect or fastText
            )
