"""Tests for scripts/distribution_gate.py: unigram-distribution gate (the 'ej detector').

Runs under pytest, or standalone: `python tests/test_distribution_gate.py`.
"""
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from distribution_gate import js_divergence, kl_divergence, skew_table


def test_js_identical_is_zero():
    c = Counter({"a": 10, "b": 5, "c": 1})
    assert js_divergence(c, c) == 0.0


def test_js_disjoint_is_one():
    a = Counter({"x": 5, "y": 5})
    b = Counter({"p": 5, "q": 5})
    assert abs(js_divergence(a, b) - 1.0) < 1e-9      # disjoint supports -> 1 bit


def test_js_symmetric():
    a = Counter({"a": 8, "b": 2, "c": 1})
    b = Counter({"a": 3, "b": 6, "d": 2})
    assert abs(js_divergence(a, b) - js_divergence(b, a)) < 1e-12


def test_kl_identical_near_zero():
    c = Counter({"a": 10, "b": 5, "c": 3})
    assert abs(kl_divergence(c, c)) < 1e-9


def test_over_table_surfaces_the_tic():
    # synth over-uses 'ej'; real barely has it -> 'ej' must top the OVER table
    synth = Counter({"ej": 40, "i": 30, "je": 30, "na": 20})
    real = Counter({"i": 40, "je": 40, "na": 30, "ej": 1})
    top = skew_table(synth, real, "over", top=3, min_count=2)
    assert top[0][1] == "ej"


def test_under_table_surfaces_the_gap():
    # 'hej' is common in real, absent in synth -> tops the UNDER table
    synth = Counter({"i": 40, "je": 40, "na": 30})
    real = Counter({"hej": 40, "i": 20, "je": 20, "na": 20})
    top = skew_table(synth, real, "under", top=3, min_count=2)
    assert top[0][1] == "hej"


def test_min_count_filters_noise():
    # a token appearing once must not dominate the ranking
    synth = Counter({"rare": 1, "common": 100, "ej": 30})
    real = Counter({"common": 100, "ej": 1})
    top = skew_table(synth, real, "over", top=5, min_count=4)
    assert all(row[1] != "rare" for row in top)


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
