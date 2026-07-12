"""Tests for serblo.pipeline.normalize: the load-bearing correctness gotchas (PRD §7.3).

Invariants encoded here (a violation quietly poisons the whole corpus):
  - Cyrillic -> Latin gajica is exact
  - diacritics (č ć š ž đ) are NEVER stripped
  - already-Latin text passes through unchanged (idempotent)
  - no Cyrillic survives normalization

Runs under pytest, or standalone: `python tests/test_normalize.py`.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.pipeline.normalize import normalize_text, has_cyrillic, diacritic_ratio

# (Cyrillic input, expected Latin gajica): verified against cyrtranslit directly.
CYR_LATIN = [
    ("Ђорђе", "Đorđe"), ("људи", "ljudi"), ("њива", "njiva"), ("џез", "džez"),
    ("Чачак", "Čačak"), ("шума", "šuma"), ("жаба", "žaba"), ("ћирилица", "ćirilica"),
    ("Ово је тест: ђ љ њ џ ћ ч ш ж", "Ovo je test: đ lj nj dž ć č š ž"),
]


def test_cyrillic_to_latin_exact():
    for src, exp in CYR_LATIN:
        assert normalize_text(src) == exp, f"{src!r} -> {normalize_text(src)!r}, exp {exp!r}"


def test_diacritics_preserved():
    for ch in "čćšžđ":
        assert ch in normalize_text(f"reč {ch} kraj"), f"lost diacritic {ch!r}"
    assert normalize_text("Ниш") == "Niš"          # Cyrillic -> keeps š
    assert normalize_text("već") == "već"           # Latin -> keeps ć


def test_latin_passthrough_idempotent():
    latin = "Čačak već ljudi u Nišu, ovo je već latinica."
    assert normalize_text(latin) == latin


def test_double_normalize_idempotent():
    for src, _ in CYR_LATIN:
        once = normalize_text(src)
        assert normalize_text(once) == once, f"not idempotent: {once!r}"


def test_no_residual_cyrillic():
    for src, _ in CYR_LATIN:
        assert not has_cyrillic(normalize_text(src)), f"residual Cyrillic from {src!r}"


def test_has_cyrillic():
    assert has_cyrillic("Ђорђе")
    assert not has_cyrillic("Djordje")


def test_nfc_composes_decomposed():
    dec = "č"                                  # c + combining caron == č
    out = normalize_text(dec)
    assert out == "č" and len(out) == 1, f"NFC failed: {out!r} len {len(out)}"


def test_empty_and_none():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""
    assert normalize_text("   \n ") == "   \n "      # whitespace preserved


def test_diacritic_ratio():
    assert diacritic_ratio("ččćšž") > 0.9
    assert diacritic_ratio("hello world") == 0.0
    assert diacritic_ratio("") == 0.0


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
