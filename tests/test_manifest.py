"""Tests for serblo.generate.manifest: M3 combinatorial prompt manifest (PRD §7.4, §7.9).

Runs under pytest, or standalone: `python tests/test_manifest.py`.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.generate.manifest import (
    DOC_CLOSE, DOC_OPEN, axes_for, conversational_axes, decode, essay_axes,
    iter_records, load_config, make_record, sample_indices, space_size,
)

CFG = load_config(pathlib.Path(__file__).resolve().parent.parent / "configs" / "generation-config.yaml")
CONV = conversational_axes(CFG)
ESSAY = essay_axes(CFG)


def test_space_sizes():
    # 5 x 6 x 12 x 7 x 14 x 4 x 4(output_shape)
    assert space_size(CONV) == 5 * 6 * 12 * 7 * 14 * 4 * 4 == 564480
    # 7 x 6 x 12 x 3 x 2 x 3
    assert space_size(ESSAY) == 7 * 6 * 12 * 3 * 2 * 3 == 9072


def test_decode_bounds_and_canonical_order():
    first = decode(0, CONV)
    last = decode(space_size(CONV) - 1, CONV)
    assert list(first) == [name for name, _ in CONV]          # canonical order preserved
    assert first["voice"] == CONV[0][1][0]                    # index 0 -> all first values
    assert last["output_shape"] == CONV[-1][1][-1]            # index max -> all last values


def test_decode_is_bijective_over_space():
    n = space_size(ESSAY)
    seen = {tuple(decode(i, ESSAY).values()) for i in range(n)}
    assert len(seen) == n                                     # every index -> a distinct combo


def test_sample_distinct_and_reproducible():
    a = sample_indices(CONV, 5000, seed=0)
    b = sample_indices(CONV, 5000, seed=0)
    c = sample_indices(CONV, 5000, seed=1)
    assert a == b                                             # same seed -> same draw
    assert a != c                                             # different seed -> different draw
    assert len(a) == len(set(a)) == 5000                     # distinct while n <= space


def test_sample_repeats_only_past_space():
    n = space_size(ESSAY) + 500
    idx = sample_indices(ESSAY, n, seed=0)
    assert len(idx) == n
    assert set(idx) == set(range(space_size(ESSAY)))         # full space covered, remainder = repeats


def test_conversational_prompt_fully_filled():
    rec = next(iter_records(CFG, "conversational", 1, seed=0))
    p = rec["prompt"]
    assert "{" not in p and "}" not in p                     # no leftover format slots
    assert p.rstrip().endswith("nothing outside them.")      # §7.9 wrapper is the tail
    assert DOC_OPEN in p and DOC_CLOSE in p


def test_essay_prompt_fully_filled():
    rec = next(iter_records(CFG, "essay", 1, seed=0))
    p = rec["prompt"]
    assert "{" not in p and "}" not in p
    assert "Serbian" in p and "Latin" in p                   # register instruction survived
    assert DOC_OPEN in p and DOC_CLOSE in p


def test_record_schema_and_ids():
    recs = list(iter_records(CFG, "conversational", 3, seed=0))
    assert [r["id"] for r in recs] == ["conv-0000000", "conv-0000001", "conv-0000002"]
    for r in recs:
        assert set(r) == {"id", "register", "axes", "prompt"}
        assert r["register"] == "conversational"
        # conversational axes = the sampled combo + the rotated {particles} provenance
        assert set(r["axes"]) == {name for name, _ in CONV} | {"particles"}


def test_iter_reproducible_ids_and_axes():
    a = [(r["id"], tuple(r["axes"].values())) for r in iter_records(CFG, "essay", 200, seed=0)]
    b = [(r["id"], tuple(r["axes"].values())) for r in iter_records(CFG, "essay", 200, seed=0)]
    assert a == b


def test_axes_for_dispatch():
    assert axes_for(CFG, "conversational") == CONV
    assert axes_for(CFG, "essay") == ESSAY


FAKE_POOL = [{"id": f"ref-essay-{i:06d}", "register": "essay", "source": "t",
              "text": f"Primer stila broj {i} koji pokazuje kako se piše prirodno i tečno."}
             for i in range(50)]


def test_grounding_rate_and_provenance():
    recs = list(iter_records(CFG, "essay", 400, seed=0, refpool=FAKE_POOL))
    grounded = [r for r in recs if "anchor" in r["axes"]]
    rate = len(grounded) / len(recs)
    assert 0.65 <= rate <= 0.85, f"grounding rate {rate} not ~0.75"
    g = grounded[0]
    anchor_text = next(a["text"] for a in FAKE_POOL if a["id"] == g["axes"]["anchor"])
    assert anchor_text in g["prompt"]                     # the anchor really is in the prompt
    assert g["prompt"].rstrip().endswith("nothing outside them.")  # transport still last
    ungrounded = next(r for r in recs if "anchor" not in r["axes"])
    assert "voice/register to match" not in ungrounded["prompt"]


def test_grounding_deterministic_and_off_when_no_pool():
    a = [(r["id"], r["axes"].get("anchor")) for r in iter_records(CFG, "essay", 100, 0, FAKE_POOL)]
    b = [(r["id"], r["axes"].get("anchor")) for r in iter_records(CFG, "essay", 100, 0, FAKE_POOL)]
    assert a == b                                         # same seed -> same anchors
    bare = list(iter_records(CFG, "essay", 50, seed=0, refpool=None))
    assert all("anchor" not in r["axes"] for r in bare)
    # conversational rate is 0.0 -> never grounded even WITH a pool
    conv = list(iter_records(CFG, "conversational", 50, seed=0, refpool=FAKE_POOL))
    assert all("anchor" not in r["axes"] for r in conv)


def test_grounding_brace_safe():
    pool = [{"id": "ref-essay-000000", "register": "essay", "source": "t",
             "text": "Tekst sa {viticama} i {jos} zagrada koji mora ostati doslovan."}]
    recs = [r for r in iter_records(CFG, "essay", 60, seed=0, refpool=pool) if "anchor" in r["axes"]]
    assert recs, "no grounded record in 60 draws at rate 0.75?"
    assert "{viticama}" in recs[0]["prompt"]              # .replace, not .format


def test_particles_rotate_and_are_reproducible():
    recs = list(iter_records(CFG, "conversational", 200, seed=0))
    prompts_have = all(r["axes"]["particles"] in r["prompt"] for r in recs)
    assert prompts_have                                        # {particles} filled into prompt
    combos = {r["axes"]["particles"] for r in recs}
    assert len(combos) > 20                                    # genuinely rotating, not fixed
    # no single particle dominates: the most common appears in < 60% of prompts
    from collections import Counter
    c = Counter(p for r in recs for p in r["axes"]["particles"].split(", "))
    assert c.most_common(1)[0][1] < 0.6 * len(recs)
    # reproducible: same seed -> same particle draw per id
    again = {r["id"]: r["axes"]["particles"] for r in iter_records(CFG, "conversational", 200, seed=0)}
    assert {r["id"]: r["axes"]["particles"] for r in recs} == again


def test_make_record_uses_thesis_and_shape_lookups():
    # essay thesis slot must resolve to the instruction text, not the key
    combo = decode(0, ESSAY)
    rec = make_record(CFG, "essay", combo, 0)
    thesis_instr = CFG["templates"]["thesis_instructions"][combo["thesis"]]
    assert thesis_instr in rec["prompt"]
    # conversational output_shape slot likewise
    ccombo = decode(0, CONV)
    crec = make_record(CFG, "conversational", ccombo, 0)
    shape_instr = CFG["templates"]["output_shape_instructions"][ccombo["output_shape"]]
    assert shape_instr in crec["prompt"]


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
