"""Tests for serblo.dedup.minhash: MinHash/LSH near-dup core (PRD M2).

Runs under pytest, or standalone: `python tests/test_minhash.py`.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.dedup.minhash import MinHasher, jaccard, band_keys, shingles, UnionFind

MH = MinHasher(num_perm=128, k=5, seed=0)
LONG = "Marko Kraljević jaše konja preko polja i pije vino sa majkom Jevrosimom u hladu " * 5


def test_identical_signature_jaccard_one():
    t = "ovo je jedan duži tekst na srpskom jeziku koji ima dovoljno reci za shingle"
    assert jaccard(MH.signature(t), MH.signature(t)) == 1.0


def test_near_dup_high_jaccard():
    edited = LONG.replace("vino", "rakiju", 1)         # one word changed in a long doc
    j = jaccard(MH.signature(LONG), MH.signature(edited))
    assert j > 0.7, f"near-dup jaccard too low: {j}"


def test_different_low_jaccard():
    a = "fudbalska utakmica bila je veoma uzbudljiva sinoc na stadionu partizana pred punom publikom"
    b = "recept za sarmu zahteva kiseli kupus mleveno meso pirinac i mnogo strpljenja tokom kuvanja"
    j = jaccard(MH.signature(a), MH.signature(b))
    assert j < 0.2, f"unrelated jaccard too high: {j}"


def test_near_dup_shares_band():
    edited = LONG.replace("vino", "rakiju", 1)
    k1 = set(band_keys(MH.signature(LONG), 16, 8))
    k2 = set(band_keys(MH.signature(edited), 16, 8))
    assert k1 & k2, "near-dup did not collide in any LSH band"


def test_shingles():
    assert shingles("", 5) == []
    assert shingles("dve reci", 5) == ["dve reci"]      # < k -> whole text
    assert len(shingles("a b c d e f", 5)) == 2         # 6 words -> two 5-grams


def test_empty_signature_none():
    assert MH.signature("") is None
    assert MH.signature("   ") is None


def test_unionfind_transitive():
    uf = UnionFind(6)
    uf.union(0, 1); uf.union(1, 2); uf.union(4, 5)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(4)
    assert uf.find(3) == 3


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in fns:
        try:
            fn(); print(f"PASS {name}")
        except AssertionError as e:
            fails += 1; print(f"FAIL {name}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    raise SystemExit(1 if fails else 0)
