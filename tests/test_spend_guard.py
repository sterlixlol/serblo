"""Tests for serblo.generate.spend_guard: client-side spend hard-cap (real money).

The invariant under test: committed spend can NEVER exceed the budget, because we only
ever reserve worst-case and worst-case is bounded by maxOutputTokens.

Runs under pytest, or standalone: `python tests/test_spend_guard.py`.
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.generate.spend_guard import (
    SpendGuard, actual_cost, batch_worst_cost, per_prompt_worst_cost,
)

MODEL = "gemini-2.5-flash-lite"                              # $0.05/M in, $0.20/M out


def test_worst_cost_math():
    # 20k prompts, 130 in, 640 max out: in 20k*130/1e6*0.05 + out 20k*640/1e6*0.20
    c = batch_worst_cost(20000, 130, 640, MODEL)
    assert abs(c - (20000 * 130 / 1e6 * 0.05 + 20000 * 640 / 1e6 * 0.20)) < 1e-9
    assert abs(c - 2.69) < 0.001                             # $0.13 in + $2.56 out


def test_actual_never_exceeds_worst():
    # actual output <= maxOutputTokens -> actual cost <= reserved worst, always
    worst = batch_worst_cost(1000, 130, 640, MODEL)
    for out_per in (10, 200, 639, 640):                     # any real output up to the cap
        act = actual_cost(1000 * 130, 1000 * out_per, MODEL)
        assert act <= worst + 1e-9


def test_hard_cap_refuses_and_trims():
    with tempfile.TemporaryDirectory() as d:
        g = SpendGuard(pathlib.Path(d) / "spend-x.json", budget=5.0)
        per = per_prompt_worst_cost(130, 640, MODEL)        # ~$0.0001286/prompt
        fit = g.max_prompts(130, 640, MODEL)
        assert fit == int(5.0 / per)                        # exactly what the budget covers
        # reserving the full fit consumes ~all budget; one more prompt must not fit
        g.reserve("job1", batch_worst_cost(fit, 130, 640, MODEL))
        assert g.remaining() >= -1e-9                       # never negative
        assert g.max_prompts(130, 640, MODEL) == 0          # nothing left


def test_settle_frees_headroom():
    with tempfile.TemporaryDirectory() as d:
        g = SpendGuard(pathlib.Path(d) / "spend-y.json", budget=10.0)
        worst = batch_worst_cost(10000, 130, 640, MODEL)
        g.reserve("job1", worst)
        before = g.remaining()
        # actual: only 200 out-tok/gen used (not the 640 cap) -> real cost far below worst
        act = actual_cost(10000 * 130, 10000 * 200, MODEL)
        g.settle("job1", act)
        assert act < worst
        assert g.remaining() > before                       # headroom freed
        assert abs(g.committed() - act) < 1e-9


def test_invariant_across_many_batches():
    # fire batches until refused; committed must never cross budget even at worst case
    with tempfile.TemporaryDirectory() as d:
        g = SpendGuard(pathlib.Path(d) / "spend-z.json", budget=3.7)
        i = 0
        while True:
            fit = g.max_prompts(130, 640, MODEL)
            if fit <= 0:
                break
            n = min(fit, 5000)                              # batch size cap
            g.reserve(f"job{i}", batch_worst_cost(n, 130, 640, MODEL))
            assert g.committed() <= 3.7 + 1e-9              # THE invariant, checked every step
            i += 1
        assert g.committed() <= 3.7 + 1e-9
        assert g.remaining() < per_prompt_worst_cost(130, 640, MODEL)   # squeezed to the last prompt


def test_shared_ledger_across_out_dirs():
    # BUG regression: a per-out-dir ledger would grant the full budget once per dir.
    # The guard must share one file per account (the out dirs' parent), so a second
    # dir's submit sees the first dir's committed spend.
    with tempfile.TemporaryDirectory() as d:
        parent = pathlib.Path(d)
        (parent / "vertex-a").mkdir(); (parent / "vertex-b").mkdir()
        g1 = SpendGuard(parent / "spend-acct.json", budget=100.0)   # dir A submit
        g1.reserve("jobA", 80.0)
        g2 = SpendGuard(parent / "spend-acct.json", budget=100.0)   # dir B submit, same account
        assert abs(g2.committed() - 80.0) < 1e-9                    # sees A's reservation
        assert g2.remaining() < 21.0                                # NOT a fresh $100


def test_ledger_persists_and_recaps():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "spend-p.json"
        g = SpendGuard(p, budget=5.0)
        g.reserve("j", 2.0)
        g2 = SpendGuard(p, budget=5.0)                       # reload
        assert abs(g2.committed() - 2.0) < 1e-9
        g3 = SpendGuard(p, budget=1.0)                       # re-cap lower (never widen silently)
        assert g3.data["budget"] == 1.0
        assert g3.remaining() < 0                            # already over the new cap -> refuses more
        assert g3.max_prompts(130, 640, MODEL) == 0


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
