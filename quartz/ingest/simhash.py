import hashlib
import struct
from array import array

BITS = 64
BANDS = 8
BAND_SIZE = BITS // BANDS  # 8 bits per band

class SimHashDeduplicator:
    """
    64-bit SimHash with LSH band deduplication.
    Near-duplicate threshold: Hamming distance ≤ 3.

    Memory at 1M docs: ~30MB for band tables.
    Time complexity: O(1) per lookup (hash table).
    """

    def __init__(self):
        # 8 band tables, each maps band_value (u64) → list of fingerprints
        self.band_tables: list[dict[int, list[int]]] = [{} for _ in range(BANDS)]
        self.total_seen = 0
        self.total_dupes = 0

    @staticmethod
    def fingerprint(text: str) -> int:
        """Compute 64-bit SimHash of text."""
        v = array("q", [0] * BITS)  # signed 64-bit ints for accumulation
        tokens = text.lower().split()
        # Use only first 200 tokens for speed; sufficient for near-dup detection
        for token in tokens[:200]:
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big")
            for i in range(BITS):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        return sum(1 << i for i in range(BITS) if v[i] > 0)

    @staticmethod
    def hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    def is_duplicate(self, fp: int) -> bool:
        """Check if fp is a near-duplicate of any seen fingerprint (Hamming ≤ 3)."""
        for b in range(BANDS):
            band_val = (fp >> (b * BAND_SIZE)) & ((1 << BAND_SIZE) - 1)
            candidates = self.band_tables[b].get(band_val, [])
            for candidate in candidates:
                if self.hamming(fp, candidate) <= 3:
                    return True
        return False

    def add(self, fp: int):
        """Register a fingerprint as seen."""
        for b in range(BANDS):
            band_val = (fp >> (b * BAND_SIZE)) & ((1 << BAND_SIZE) - 1)
            self.band_tables[b].setdefault(band_val, []).append(fp)

    def check_and_add(self, text: str) -> bool:
        """Returns True if duplicate (should skip), else adds and returns False."""
        self.total_seen += 1
        fp = self.fingerprint(text)
        if self.is_duplicate(fp):
            self.total_dupes += 1
            return True
        self.add(fp)
        return False

    @property
    def dupe_rate(self) -> float:
        return self.total_dupes / max(self.total_seen, 1)
