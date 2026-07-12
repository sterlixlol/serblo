"""MinHash + LSH near-duplicate detection (PRD M2 stage 4).

Built from scratch (numpy + xxhash) because datasketch can't be pip-installed under
Arch's PEP 668. Runs AFTER transliteration, pooled across all sources, so it collapses
cross-script Serbian twins (same article in Cyrillic and Latin) and cross-source overlap.

Math note: signatures use the Mersenne prime P = 2**31 - 1 so that a*h + b stays below
2**64 and numpy uint64 arithmetic doesn't silently overflow (a 2**61 prime would).
"""
import numpy as np
import xxhash

_P = np.uint64(2147483647)      # 2**31 - 1


def shingles(text, k):
    """Word k-gram shingles; short texts collapse to a single whole-text shingle."""
    w = text.split()
    if len(w) < k:
        return [" ".join(w)] if w else []
    return [" ".join(w[i:i + k]) for i in range(len(w) - k + 1)]


class MinHasher:
    def __init__(self, num_perm=128, k=5, seed=0):
        self.num_perm, self.k = num_perm, k
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, int(_P), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_P), size=num_perm, dtype=np.uint64)

    def signature(self, text):
        sh = shingles(text, self.k)
        if not sh:
            return None
        base = np.fromiter((xxhash.xxh32(s).intdigest() % int(_P) for s in sh),
                           dtype=np.uint64, count=len(sh))
        # (num_perm, n_shingles) permuted hashes, then min over shingles -> signature
        h = (self.a[:, None] * base[None, :] + self.b[:, None]) % _P
        return h.min(axis=1).astype(np.uint64)


def jaccard(sig1, sig2):
    """Estimated Jaccard similarity from two MinHash signatures."""
    return float(np.mean(sig1 == sig2))


def band_keys(sig, bands, rows):
    """Stable per-band bucket keys for a signature (xxhash of each band's bytes)."""
    return [(band, xxhash.xxh64(sig[band * rows:(band + 1) * rows].tobytes()).intdigest())
            for band in range(bands)]


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb
