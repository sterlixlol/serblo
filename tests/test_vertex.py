"""Tests for serblo.generate.run_vertex: Vertex Gemini batch driver (PRD §7b).

Pure logic only: request-payload construction, cost math, output-line parsing, the
retrieve->convert hookup, resume-id loading, and the offline DRY-RUN submit path.
NO network, NO API, NO spend: the live submit/poll/retrieve calls are never touched.

Runs under pytest, or standalone: `python tests/test_vertex.py`.
"""
import io
import json
import pathlib
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.generate import run_vertex as rv


# ---- request payload construction -------------------------------------------

def test_build_request_shape():
    rec = {"id": "essay-0000042", "register": "essay", "axes": {"topic_domain": "history"},
           "prompt": "Napiši kratak esej.\n\n<<<DOC>>> … <<<END>>>"}
    req = rv.build_request(rec, "gemini-2.5-flash-lite", 0.8, 640)
    assert req["key"] == "essay-0000042"                       # id round-trips as the batch key
    contents = req["request"]["contents"]
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == rec["prompt"]    # exact prompt, unmodified
    assert req["request"]["generationConfig"]["temperature"] == 0.8
    assert req["request"]["generationConfig"]["maxOutputTokens"] == 640   # the cost cap


def test_build_request_serializes_and_only_text_is_prompt():
    rec = {"id": "essay-1", "register": "essay", "axes": {}, "prompt": "Đačko: šć ž č."}
    line = json.dumps(rv.build_request(rec, rv.MODEL_DEFAULT, 0.8, 640), ensure_ascii=False)
    back = json.loads(line)
    assert back["request"]["contents"][0]["parts"][0]["text"] == "Đačko: šć ž č."


def test_build_request_on_real_manifest_line():
    p = pathlib.Path(rv.ROOT / "data" / "manifests" / "essay.jsonl")
    if not p.exists():
        print("SKIP test_build_request_on_real_manifest_line (no essay.jsonl)")
        return
    with open(p, encoding="utf-8") as f:
        rec = json.loads(f.readline())
    req = rv.build_request(rec, rv.MODEL_DEFAULT, 0.8, 640)
    assert req["key"] == rec["id"]
    assert req["request"]["contents"][0]["parts"][0]["text"] == rec["prompt"]


# ---- cost math (the multiplication) -----------------------------------------

def test_estimate_cost_multiplication():
    prompts = ["a" * 400, "a" * 600]                           # 1000 chars total -> 250 in-tok
    est = rv.estimate_cost(prompts, est_output_tokens=700, model="gemini-2.5-flash-lite")
    assert est["n"] == 2
    assert est["in_tok"] == 250.0                              # 1000 / 4.0
    assert est["out_tok"] == 1400                              # 2 * 700
    assert abs(est["in_cost"] - (250 / 1e6 * 0.05)) < 1e-12    # = 1.25e-5
    assert abs(est["out_cost"] - (1400 / 1e6 * 0.20)) < 1e-12  # = 2.8e-4
    assert abs(est["total"] - (est["in_cost"] + est["out_cost"])) < 1e-12


def test_estimate_cost_scales_3k_and_5k():
    # avg essay prompt ~518 chars -> ~130 in-tok; 700 out-tok assumed.
    for n, want_total in ((3000, 0.44), (5000, 0.73)):
        prompts = ["x" * 518] * n
        est = rv.estimate_cost(prompts, 700, "gemini-2.5-flash-lite")
        assert round(est["total"], 2) == want_total, f"n={n}: {est['total']}"


def test_estimate_cost_unknown_model_no_crash():
    est = rv.estimate_cost(["hello"], 700, "some-unlisted-model")
    assert est["price"] is None and est["total"] is None       # cost unknown, but no exception


# ---- Vertex output line parsing ---------------------------------------------

def test_parse_output_line_happy():
    obj = {"key": "essay-7",
           "response": {"candidates": [{"content": {"parts": [{"text": "<<<DOC>>>Tekst.<<<END>>>"}]}}]}}
    key, text = rv.parse_output_line(obj)
    assert key == "essay-7"
    assert text == "<<<DOC>>>Tekst.<<<END>>>"


def test_parse_output_line_multipart_joined_and_str_input():
    obj = {"key": "essay-8",
           "response": {"candidates": [{"content": {"parts": [{"text": "aa"}, {"text": "bb"}]}}]}}
    key, text = rv.parse_output_line(json.dumps(obj))           # accepts a raw JSON string too
    assert (key, text) == ("essay-8", "aabb")


def test_parse_output_line_blocked_returns_none_text():
    for resp in ({"candidates": []}, {}, {"candidates": [{"content": {"parts": []}}]}):
        key, text = rv.parse_output_line({"key": "essay-9", "response": resp})
        assert key == "essay-9" and text is None                # blocked/empty -> caller logs + skips


def test_usage_of_line_billing_tokens():
    # real Vertex output-line shape: usageMetadata carries GOOGLE's billing
    # counts, the only numbers settle may trust (our word-ratio undercounts ~25-40%)
    obj = {"response": {"candidates": [], "usageMetadata":
           {"promptTokenCount": 184, "candidatesTokenCount": 190, "totalTokenCount": 374}}}
    assert rv.usage_of_line(obj) == (184, 190)
    assert rv.usage_of_line({"response": {}}) == (0, 0)       # absent -> caller pads
    assert rv.usage_of_line({}) == (0, 0)


def test_is_succeeded_enum_prefix_and_partial():
    # str(enum) is class-prefixed: the live bug that skipped a SUCCEEDED job
    assert rv.is_succeeded("JobState.JOB_STATE_SUCCEEDED")
    assert rv.is_succeeded("JOB_STATE_SUCCEEDED")
    assert not rv.is_succeeded("JobState.JOB_STATE_PARTIALLY_SUCCEEDED")
    assert not rv.is_succeeded("JobState.JOB_STATE_FAILED")


# ---- retrieve -> convert hookup (offline, no GCS) ----------------------------

def test_doc_records_from_vertex_text():
    body = ("Ovo je esej o istoriji koji ima dovoljno reči da pređe prag od petnaest "
            "reči jer je politika takva da se kratki fragmenti odbacuju.")
    raw = f"<<<DOC>>>\n{body}\n<<<END>>>"
    recs, issues = rv.doc_records("essay-5", "essay", {"topic_domain": "history"},
                                  raw, "gemini-2.5-flash-lite")
    assert len(recs) == 1
    r = recs[0]
    assert r["id"] == "essay-5-d0"
    assert r["text"] == body
    assert r["source"] == "vertex"
    assert r["register"] == "essay"
    assert r["model"] == "gemini-2.5-flash-lite"
    assert r["axes"] == {"topic_domain": "history"}
    assert r["words"] == len(body.split())
    assert r["issues"] is None
    assert issues == []


def test_output_line_to_docs_endtoend():
    # the whole phase-3 transform, offline: a synthetic Vertex output line -> clean doc.
    body = ("Prvi pasus koji je dovoljno dug da pređe prag od petnaest reči jer se "
            "kratki fragmenti odbacuju.\n\nDrugi pasus koji nastavlja tekst.")
    obj = {"key": "essay-11",
           "response": {"candidates": [{"content": {"parts": [
               {"text": f"Evo teksta:\n<<<DOC>>>\n{body}\n<<<END>>>"}]}}]}}
    key, text = rv.parse_output_line(obj)
    recs, _ = rv.doc_records(key, "essay", {}, text, rv.MODEL_DEFAULT)
    assert [r["text"] for r in recs] == [body]        # chatter dropped, sentinels stripped
    assert recs[0]["source"] == "vertex"


# ---- resume / ledger loading ------------------------------------------------

def test_load_done_reads_ids_and_tolerates_corruption():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "raw.jsonl"
        p.write_text(
            json.dumps({"id": "essay-0", "raw": "x"}) + "\n"
            + json.dumps({"id": "essay-1", "raw": "y"}) + "\n"
            + '{"id": "essay-2", "raw": "trun',             # crash-truncated last line
            encoding="utf-8")
        assert rv.load_done(d) == {"essay-0", "essay-1"}
    assert rv.load_done(pathlib.Path(d)) == set()             # missing dir/file -> empty, no crash


def test_load_inflight_excludes_retrieved_jobs():
    with tempfile.TemporaryDirectory() as d:
        dd = pathlib.Path(d)
        # ids now live in each job's .map.jsonl sidecar (streamed at submit), not inline
        (dd / "A.map.jsonl").write_text(
            json.dumps({"key": "a1"}) + "\n" + json.dumps({"key": "a2"}) + "\n", encoding="utf-8")
        (dd / "B.map.jsonl").write_text(json.dumps({"key": "b1"}) + "\n", encoding="utf-8")
        (dd / "jobs.jsonl").write_text(
            json.dumps({"job_name": "A", "map_local": str(dd / "A.map.jsonl"), "retrieved": False}) + "\n"
            + json.dumps({"job_name": "B", "map_local": str(dd / "B.map.jsonl"), "retrieved": False}) + "\n"
            + json.dumps({"job_name": "B", "retrieved": True, "ts": 1}) + "\n",  # B later retrieved
            encoding="utf-8")
        # A still in flight -> its ids block resubmission; B retrieved -> its ids drop out
        assert rv.load_inflight(d) == {"a1", "a2"}


# ---- DRY-RUN submit: builds payload, prints cost, spends nothing -------------

def _essay_manifest(dir_path, n):
    with open(pathlib.Path(dir_path) / "essay.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "id": f"essay-{i:07d}", "register": "essay", "axes": {"k": i},
                "prompt": f"Napiši esej broj {i}. " + "x" * 500,
            }, ensure_ascii=False) + "\n")


def _input_file(out_dir):
    """The single batch-input file, excluding the .map.jsonl sidecar (*.jsonl matches both)."""
    files = [p for p in pathlib.Path(out_dir).glob("input-essay-*.jsonl")
             if not p.name.endswith(".map.jsonl")]
    assert len(files) == 1, files
    return files[0]


def _args(manifest_dir, out_dir, **over):
    base = dict(manifest_dir=str(manifest_dir), out=str(out_dir), register="essay",
                model=rv.MODEL_DEFAULT, temperature=0.8, limit=0, est_output_tokens=700,
                account="default", budget=None, max_output_tokens=768,
                submit=False, poll=False, retrieve=False, rebuild=False,
                gcs_bucket=None, project=None, location="us-central1")
    base.update(over)
    return SimpleNamespace(**base)


def test_dryrun_writes_input_no_job_no_spend():
    with tempfile.TemporaryDirectory() as md, tempfile.TemporaryDirectory() as od:
        _essay_manifest(md, 3)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rv.submit(_args(md, od), ["essay"])
        out = buf.getvalue()

        maps = list(pathlib.Path(od).glob("input-essay-*.map.jsonl"))
        assert len(maps) == 1
        lines = [json.loads(x) for x in _input_file(od).read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 3
        assert lines[0]["key"] == "essay-0000000"
        assert lines[0]["request"]["generationConfig"]["temperature"] == 0.8
        # NOTHING submitted:
        assert not (pathlib.Path(od) / "jobs.jsonl").exists()
        assert not (pathlib.Path(od) / "raw.jsonl").exists()
        # dry-run printed the cost (incl. worst-case) + the exact would-run live steps:
        assert "DRY RUN" in out and "WORST-CASE (reserved)" in out
        assert "gcloud storage cp" in out and "batches.create(" in out


def test_dryrun_resume_skips_spent_ids():
    with tempfile.TemporaryDirectory() as md, tempfile.TemporaryDirectory() as od:
        _essay_manifest(md, 3)
        # pretend essay-0000001 was already retrieved in a prior run
        (pathlib.Path(od) / "raw.jsonl").write_text(
            json.dumps({"id": "essay-0000001", "model": rv.MODEL_DEFAULT, "raw": "x"}) + "\n",
            encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            rv.submit(_args(md, od), ["essay"])
        keys = [json.loads(x)["key"] for x in _input_file(od).read_text(encoding="utf-8").splitlines()]
        assert keys == ["essay-0000000", "essay-0000002"]      # spent id filtered out


def test_dryrun_limit_caps_batch():
    with tempfile.TemporaryDirectory() as md, tempfile.TemporaryDirectory() as od:
        _essay_manifest(md, 10)
        with redirect_stdout(io.StringIO()):
            rv.submit(_args(md, od, limit=4), ["essay"])
        assert len(_input_file(od).read_text(encoding="utf-8").splitlines()) == 4


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
