"""Tests for serblo.generate.refpool: grounding anchor excerpts (PRD §7.4).

Runs under pytest, or standalone: `python tests/test_refpool.py`.
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.generate.refpool import anchor_ok, excerpt, make_anchor

SENT = ("Kulturno nasleđe je važan deo identiteta jer se kroz njega prenose vrednosti "
        "i priče koje su oblikovale zajednicu. ")
LONG = SENT * 12                                          # ~200+ words of clean Serbian


def test_excerpt_within_bounds_and_sentence_trimmed():
    rng = random.Random(0)
    for _ in range(20):
        ex = excerpt(LONG, rng)
        assert ex is not None
        n = len(ex.split())
        assert 45 <= n <= 140                             # min minus trim allowance
        assert ex.endswith(".")                           # trimmed to a sentence end


def test_excerpt_too_short_none():
    assert excerpt("samo par reči ovde", random.Random(0)) is None


def test_anchor_ok_gates():
    assert anchor_ok(LONG)
    assert not anchor_ok(LONG.replace("nasleđe", "uredi", 1))      # wiki chrome
    assert not anchor_ok(LONG[: len(SENT)] + " | kolona | tabela " + LONG)  # table pipes
    assert not anchor_ok("The quick brown fox jumps over the lazy dog. " * 20)  # not Serbian
    digity = " ".join(f"{i}. stavka {i*3}" for i in range(60))
    assert not anchor_ok(digity)                          # digit-heavy list


def test_make_anchor_record_shape():
    a = make_anchor(LONG, random.Random(0), "essay", "fineweb2", 7)
    assert a is not None
    assert a["id"] == "ref-essay-000007"
    assert a["register"] == "essay" and a["source"] == "fineweb2"
    assert len(a["text"].split()) >= 45


def test_make_anchor_rejects_junk():
    assert make_anchor("uredi | " * 60, random.Random(0), "essay", "fineweb2", 0) is None


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
