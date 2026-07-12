"""Tests for serblo.pipeline.encoding: the OpenSubtitles CP1250-as-CP1251 mojibake fix.

Runs under pytest, or standalone: `python tests/test_encoding.py`.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.pipeline.encoding import fix_mojibake, has_mixed_script_token

# (corrupted OpenSubtitles line, correct Serbian): real samples from sr v2024 mono.
CORRUPT = [
    ("Ћeliљ da nazoveљ tatu?", "Želiš da nazoveš tatu?"),
    ("Vaћi.", "Važi."),
    ("иuvaj se svog muћa.", "čuvaj se svog muža."),
    ("Sebe naziva Dћolin Parker.", "Sebe naziva Džolin Parker."),
    ("Љta nije u redu?", "Šta nije u redu?"),
    ("Poћurite!", "Požurite!"),
]


def test_repairs_cp1250_as_cp1251():
    for src, exp in CORRUPT:
        assert fix_mojibake(src) == exp, f"{src!r} -> {fix_mojibake(src)!r}, exp {exp!r}"


def test_leaves_clean_latin_untouched():
    for s in ["- Da.", "Zdravo, kako si?", "Idemo!", "Brzo!"]:
        assert fix_mojibake(s) == s, f"clobbered clean latin: {s!r}"


def test_leaves_genuine_cyrillic_untouched():
    s = "Здраво, како си данас?"          # real Cyrillic, must NOT be round-tripped
    assert fix_mojibake(s) == s


def test_codeswitch_line_not_flagged():
    # Cyrillic word + Latin word, but no single token mixes scripts -> not mojibake
    assert not has_mixed_script_token("Здраво Peter")
    assert fix_mojibake("Здраво Peter") == "Здраво Peter"


def test_mixed_token_detected():
    assert has_mixed_script_token("Ћeliљ")
    assert not has_mixed_script_token("Zdravo")
    assert not has_mixed_script_token("Здраво")


def test_empty():
    assert fix_mojibake("") == ""
    assert fix_mojibake(None) is None


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
