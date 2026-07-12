"""Tests for serblo.generate.run_api: OpenAI-compatible open-weight driver (pure logic).

Runs under pytest, or standalone: `python tests/test_api.py`.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.generate import run_api

SR = ("ma bre ljudi ova današnja muzika je nešto strašno, ne znam šta bih rekao, "
      "sve neki žestok bit i vrištanje a pesme nigde, ne sluša to niko normalan više")


def test_lenient_recovers_unwrapped_serbian():
    # model ignored the <<<DOC>>> wrapper but wrote fine Serbian -> recovered + flagged
    recs, issues = run_api.doc_records("x-1", "conversational", {}, SR, "nim", "glm", lenient=True)
    assert len(recs) == 1
    assert recs[0]["text"] == SR
    assert "lenient_no_sentinel" in recs[0]["issues"]


def test_lenient_still_junk_gates_english():
    # unwrapped but NON-Serbian (leaked reasoning) -> lenient takes it, junk gate drops it
    junk = "Okay, let me write an informal Serbian text about music for this forum thread task."
    recs, issues = run_api.doc_records("x-2", "conversational", {}, junk, "nim", "glm", lenient=True)
    assert recs == []
    assert "dropped_junk_doc" in issues


def test_strict_mode_drops_unwrapped():
    recs, issues = run_api.doc_records("x-3", "conversational", {}, SR, "nim", "glm", lenient=False)
    assert recs == [] and "no_sentinels" in issues


def test_wrapped_output_unaffected_by_lenient():
    recs, _ = run_api.doc_records("x-4", "essay", {}, f"<<<DOC>>>\n{SR}\n<<<END>>>", "nim", "k", lenient=True)
    assert recs[0]["text"] == SR
    assert not recs[0]["issues"]                          # clean path, no lenient flag


def test_concurrent_records_inflight_on_budget_stop(tmp_path):
    """A budget-stop must DRAIN the in-flight generations (record them),
    not discard them. Old code fut.cancel()'d running futures and dropped up to K finished docs."""
    import time
    st = run_api.Stats(tmp_path, "nim", "m", "nim")
    doc = ("<<<DOC>>>\n" + "ovo je jedan dovoljno dugačak srpski tekst koji treba da prođe "
           "kroz sve provere i gejtove bez ikakvih problema danas " * 3 + "\n<<<END>>>")

    def do_one(rec):
        i = int(rec["id"].split("-")[1])
        time.sleep(0.02 if i == 0 else 0.3)                # id 0 lands first -> trips the budget stop
        return rec, doc, 0.02, 1, True, False

    manifest = ({"id": f"c-{i}", "register": "conversational", "axes": {}} for i in range(6))
    run_api.run_concurrent(manifest, do_one, st, concurrency=6, max_requests=1)
    st.close()
    assert st.stop and "budget" in st.stop
    assert st.ok == 6, f"all 6 in-flight generations must be recorded (drained), got {st.ok}"


def test_rate_limiter_unpaced_round_robins():
    rl = run_api.RateLimiter(["a", "b", "c"], rpm_per_key=0)
    assert [rl.get_key() for _ in range(6)] == ["a", "b", "c", "a", "b", "c"]


def test_rate_limiter_paces_each_key():
    import time
    rl = run_api.RateLimiter(["k"], rpm_per_key=120)        # one key @ 120/min = 1 per 0.5s
    t0 = time.time()
    rl.get_key()                                            # first is free
    rl.get_key()                                            # second must wait ~0.5s
    assert time.time() - t0 >= 0.4, "a single key must be paced to its interval, not fired instantly"


def _rec():
    return {"id": "x", "register": "conversational", "axes": {}}


def test_budget_honoured_on_rate_limited_path(tmp_path):
    """A persistently 429'd run must still stop at --max-requests, not churn."""
    st = run_api.Stats(tmp_path, "nim", "m", "nim")
    rl = (_rec(), "throttled", 1.0, 1, False, True)          # good=False, rate_limited=True
    for _ in range(20):
        if st.stop:
            break
        st.record(rl, max_requests=5)
    st.close()
    assert st.stop and "budget" in st.stop
    assert st.requests <= 6, "should stop at the budget, not run all 20"


def test_rate_limit_does_not_trip_failstreak(tmp_path):
    """A throttle is not a systemic fault; many 429s in a row must NOT stop the run."""
    st = run_api.Stats(tmp_path, "nim", "m", "nim")
    rl = (_rec(), "throttled", 1.0, 1, False, True)
    for _ in range(10):
        st.record(rl, max_requests=10_000)                  # high budget so only the streak logic is tested
    st.close()
    assert st.stop is None and st.fail_streak == 0


def test_auth_failure_trips_failstreak(tmp_path):
    """A non-rate-limit failure (bad auth / dead model) IS systemic and must stop."""
    st = run_api.Stats(tmp_path, "nim", "m", "nim")
    authfail = (_rec(), "HTTP 401", 1.0, 1, False, False)   # good=False, rate_limited=False
    for _ in range(10):
        if st.stop:
            break
        st.record(authfail, max_requests=10_000)
    st.close()
    assert st.stop and "systemic" in st.stop


def test_timeout_flagged_transient_not_systemic(monkeypatch):
    """A read timeout is TRANSIENT (slow endpoint), like a 429: generate() must flag rate_limited so a
    temporarily-slow model (NIM's GLM-5.2 degrading to >200s) trickles instead of the
    fail-streak hard-killing the whole lineage."""
    monkeypatch.setattr(run_api, "BACKOFF", [])              # no retry sleeps in the test

    def timed_out(*a, **k):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(run_api, "call", timed_out)
    limiter = run_api.RateLimiter(["k"], rpm_per_key=0)
    _text, _wall, _calls, ok, rate_limited = run_api.generate("u", limiter, "m", "p", 0.9, 1)
    assert ok is False and rate_limited is True             # transient flag set -> record() won't trip streak


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
