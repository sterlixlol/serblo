"""Vertex AI Gemini batch generation driver: manifest -> batch job -> §7.9 converter (PRD §7.3/§7b).

The essay register's PRIMARY generation channel: Gemini 2.5 Flash-Lite via Vertex AI
**batch** mode. Credit note: the $300 GCP trial EXCLUDES Gemini in AI Studio but COVERS
first-party Gemini on VERTEX (STATUS.md §7b), so this routes through Vertex, where
gemini-2.5-flash-lite is credit-eligible (partner models in Model Garden are NOT).

Why this differs from run_mimo.py: MiMo spends one call per prompt (raw-first per id).
A Vertex batch spends ONE async job covering many prompts (~24h turnaround, 50% off).
So the pipeline is three phases, resumable at each:

  1. submit  (DEFAULT = DRY RUN; real submit only with --submit + --gcs-bucket)
             Build one JSONL line per prompt -> {"key": <id>, "request": {GenerateContentRequest}}.
             `key` round-trips into the output so responses match back to ids regardless of
             order. Dry run writes the input JSONL locally, prints a sample + the projected
             cost, and STOPS: it constructs no client and makes no network/subprocess call.
             Real submit uploads the JSONL to GCS (`gcloud storage cp`) and calls
             client.batches.create(); the job is recorded in jobs.jsonl the instant create()
             returns (so its ids never re-submit = no double spend).
  2. poll    client.batches.get(name) until a terminal JobState. (written, not exercised)
  3. retrieve For each SUCCEEDED job: read the predictions JSONL from its GCS dest, match
             each line's `key` -> id, save RAW append-only, convert via extract_docs, append
             register-tagged docs. (written, not exercised)

Vertex requires GCS (or BigQuery) input: **inline requests are Gemini-Developer-API only**
(verified in the installed SDK: google/genai/batches.py `_BatchJobSource_to_vertex` and
`_transformers.t_batch_job_source` both raise for inline on Vertex). gsutil is absent on this
box; `gcloud storage` is present and is what the upload/download steps use.

Layout under --out (append-only JSONL, all incrementally flushed):
  input-{reg}-{stamp}.jsonl      the batch input (local; uploaded to GCS on --submit)
  input-{reg}-{stamp}.map.jsonl  {key, register, axes} per line, in input order (id<->response
                                 fallback if `key` does not echo back, + carries axes to docs)
  jobs.jsonl   {job_name, state, model, register, input_gcs, output_gcs, ids, n, stamp, ts,
                retrieved}: the submit ledger; a retrieve appends {job_name, retrieved:true}
  raw.jsonl    {id, model, raw, ts}   ground truth, written at RETRIEVE time, docs re-derivable
  docs.jsonl   {id, text, source, register, axes, model, words, issues}  same shape as run_mimo
  errors.jsonl {id, reason, ts}       blocked/empty responses (no raw line -> naturally retried)

Resume: an id in raw.jsonl = RETRIEVED; an id inside a not-yet-retrieved jobs.jsonl job =
IN FLIGHT. submit skips their union, so nothing is generated twice.

Usage:
  # dry run (default: prints payload + cost, spends NOTHING):
  PYTHONPATH=src python -m serblo.generate.run_vertex --register essay --limit 3000
  # real submit (spends credits, needs a bucket):
  PYTHONPATH=src python -m serblo.generate.run_vertex --register essay --limit 3000 \
      --submit --gcs-bucket my-bucket --project kazi-training-26230 --location us-central1
  # later:
  PYTHONPATH=src python -m serblo.generate.run_vertex --poll     --project ... --location ...
  PYTHONPATH=src python -m serblo.generate.run_vertex --retrieve --project ... --location ...
  # rebuild docs.jsonl from raw.jsonl (no network):
  PYTHONPATH=src python -m serblo.generate.run_vertex --rebuild
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.generate.convert import extract_docs, pick_payload_doc  # noqa: E402

SOURCE = "vertex"
MODEL_DEFAULT = "gemini-2.5-flash-lite"
ESSAY_TEMPERATURE = 0.8                                   # configs/generation-config.yaml essay temp

# Google's own rule of thumb for Gemini: ~4 characters per token (ai.google.dev/gemini-api/docs/tokens).
# Only the prompt TEXT is billed; the JSON wrapper/key are not tokenized. Used for the estimate
# ONLY: the real bill comes from usage metadata after a run.
CHARS_PER_TOKEN = 4.0
EST_OUTPUT_TOKENS_DEFAULT = 700                           # assumed essay length (real FineWeb-2 essays avg ~821 tok)

# BATCH prices (already the 50%-off tier), USD per 1M tokens.
PRICING = {
    "gemini-2.5-flash-lite": {"in": 0.05, "out": 0.20},
    "gemini-2.5-flash":      {"in": 0.15, "out": 1.25},   # for completeness if --model overridden
}

# Terminal JobState values (poll/retrieve stop here). Mirrors the SDK sample's completed set
# plus EXPIRED. Compared as plain strings so this module never needs the genai import.
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED", "JOB_STATE_EXPIRED",
}
SUCCEEDED = "JOB_STATE_SUCCEEDED"


def is_succeeded(state):
    """str(job.state) is enum-prefixed ('JobState.JOB_STATE_SUCCEEDED'): match by suffix.

    Suffix (not substring) so JOB_STATE_PARTIALLY_SUCCEEDED can never pass: it ends in
    'PARTIALLY_SUCCEEDED', which does not end with 'JOB_STATE_SUCCEEDED'."""
    return str(state).endswith(SUCCEEDED)


# ============================ pure logic (unit-tested) ============================

def build_request(record, model, temperature, max_output_tokens):
    """One Vertex batch input line as a dict: {"key", "request"}.

    `key` = the manifest id; Vertex echoes it into the output line, so responses match back
    to ids robustly (order is not guaranteed). `request` is a GenerateContentRequest, a
    single user turn carrying the manifest prompt (which already embeds the §7.9 wrapper).
    maxOutputTokens is the LOAD-BEARING cost cap: it bounds each generation's worst-case
    cost, which is what makes the client-side spend guard a hard guarantee (spend_guard.py).
    """
    return {
        "key": record["id"],
        "request": {
            "contents": [{"role": "user", "parts": [{"text": record["prompt"]}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
        },
    }


def estimate_cost(prompts, est_output_tokens, model, chars_per_token=CHARS_PER_TOKEN):
    """Projected batch cost for a list of prompt strings. Pure; shows every factor so the
    caller can print the multiplication. input tokens estimated from real chars / 4."""
    n = len(prompts)
    in_tok = sum(len(p) for p in prompts) / chars_per_token
    out_tok = n * est_output_tokens
    price = PRICING.get(model)
    in_cost = out_cost = total = None
    if price is not None:
        in_cost = in_tok / 1e6 * price["in"]
        out_cost = out_tok / 1e6 * price["out"]
        total = in_cost + out_cost
    return {
        "n": n, "in_tok": in_tok, "out_tok": out_tok,
        "avg_in_tok": (in_tok / n) if n else 0.0,
        "est_out_per": est_output_tokens, "price": price,
        "in_cost": in_cost, "out_cost": out_cost, "total": total,
    }


def usage_of_line(obj):
    """(prompt_tokens, output_tokens) from a Vertex output line's usageMetadata, GOOGLE'S
    billing token counts, the only numbers the spend guard may settle with. Our own
    tokenizer ratio (1.577 tok/word) under-counts Gemini's multilingual tokenizer on
    Serbian by ~25-40% (measured: promptTokenCount 184 vs chars/4 estimate ~130), which
    would leak phantom headroom into the budget. Returns (0, 0) when absent; caller
    falls back to a PADDED estimate (over-counting actual only makes us safer)."""
    um = (obj.get("response") or {}).get("usageMetadata") or {}
    return um.get("promptTokenCount", 0) or 0, um.get("candidatesTokenCount", 0) or 0


def parse_output_line(obj, model=None):
    """(key, text) from one Vertex batch prediction JSONL record.

    Output line = {"key", "request", "response": {"candidates": [{"content": {"parts":
    [{"text": ...}]}}]}}. Returns (key, None) when a prompt was blocked / produced no text
    (empty candidates or empty parts) so the caller can log + skip it.
    """
    if isinstance(obj, str):
        obj = json.loads(obj)
    key = obj.get("key")
    resp = obj.get("response") or {}
    cands = resp.get("candidates") or []
    if not cands:
        return key, None
    parts = ((cands[0] or {}).get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return key, (text or None)


def doc_records(rid, register, axes, raw, model):
    """Convert one raw payload -> AT MOST ONE docs.jsonl record (shared by retrieve + rebuild).

    salvage_unclosed=True mirrors run_mimo: recover a doc whose closer was mangled or whose
    tail was truncated at max_output_tokens. pick_payload_doc enforces 1-prompt=1-doc
    (leaked-reasoning junk gated, stuttered copies collapsed, last survivor kept).
    """
    docs, issues = extract_docs(raw, salvage_unclosed=True)
    doc, pol_issues = pick_payload_doc(docs)
    issues = issues + pol_issues
    if doc is None:
        return [], issues
    return [
        {"id": f"{rid}-d0", "text": doc, "source": SOURCE, "register": register,
         "axes": axes, "model": model, "words": len(doc.split()),
         "issues": issues if issues else None}
    ], issues


# ============================ resume / ledger I/O ============================

def load_done(out_dir):
    """ids already RETRIEVED (present in raw.jsonl). Tolerates a corrupt last line."""
    done, path = set(), Path(out_dir) / "raw.jsonl"
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _read_jobs(out_dir):
    """job_name -> merged record. submit writes the full record; retrieve appends a thin
    {job_name, retrieved:true} status line; later lines shallow-merge onto earlier ones."""
    jobs, path = {}, Path(out_dir) / "jobs.jsonl"
    if not path.exists():
        return jobs
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            jn = r.get("job_name")
            if not jn:
                continue
            jobs.setdefault(jn, {}).update(r)
    return jobs


_DEAD_STATES = ("FAILED", "CANCELLED", "EXPIRED")


def load_inflight(out_dir):
    """ids inside a submitted-but-not-yet-retrieved LIVE job (must not be re-submitted).

    A job whose last polled state is terminal-dead releases its ids; they were never
    generated, so a fresh submit may re-spend them (e.g. the first job on a new project
    failed INTERNAL because the Vertex service agent wasn't provisioned yet)."""
    inflight = set()
    for rec in _read_jobs(out_dir).values():
        if rec.get("retrieved"):
            continue
        state = str(rec.get("state") or "")
        if any(d in state for d in _DEAD_STATES):
            continue
        # ids come from the batch's .map.jsonl sidecar (submit streams it), not an inline
        # list, which would be 500k entries per job on jumbo batches.
        mp = rec.get("map_local")
        if mp and Path(mp).exists():
            with open(mp, encoding="utf-8") as f:
                for line in f:
                    try:
                        inflight.add(json.loads(line)["key"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return inflight


def iter_manifest(path):
    path = Path(path)
    if not path.exists():
        print(f"note: no manifest at {path}", flush=True)
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def manifest_meta(manifest_dir, registers):
    """id -> (register, axes), for tagging retrieved docs (same idea as run_mimo.rebuild)."""
    meta = {}
    for reg in registers:
        for r in iter_manifest(Path(manifest_dir) / f"{reg}.jsonl"):
            meta[r["id"]] = (r["register"], r["axes"])
    return meta


def rebuild(out_dir, manifest_dir, registers):
    """Re-derive docs.jsonl from raw.jsonl + manifest. Zero requests, zero spend.

    Run after any converter improvement, recovering payloads older parsing stranded.
    Atomic replace; first occurrence per id wins (crash-retry dups).
    """
    out_dir = Path(out_dir)
    meta = manifest_meta(manifest_dir, registers)
    raw_path = out_dir / "raw.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"nothing to rebuild: {raw_path} missing")
    tmp = out_dir / "docs.jsonl.tmp"
    seen, n_raw, n_docs, n_words, missing = set(), 0, 0, 0, 0
    with open(raw_path, encoding="utf-8") as rf, open(tmp, "w", encoding="utf-8") as w:
        for line in rf:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            n_raw += 1
            if "axes" in r and "register" in r:          # generation-time provenance wins
                reg, axes = r["register"], r["axes"]
            elif r["id"] in meta:
                reg, axes = meta[r["id"]]
            else:
                missing += 1
                print(f"warn: {r['id']} not in manifest — skipped", flush=True)
                continue
            recs, _ = doc_records(r["id"], reg, axes, r["raw"], r.get("model", MODEL_DEFAULT))
            for rec in recs:
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_docs += 1
                n_words += rec["words"]
    tmp.replace(out_dir / "docs.jsonl")
    print(f"rebuilt docs.jsonl: {n_raw:,} raw -> {n_docs:,} docs, {n_words:,} words"
          f"{f', {missing} missing from manifest' if missing else ''}", flush=True)


# ============================ phase 1: submit (dry run by default) ============================

def _gcs_paths(bucket, register, stamp):
    base = f"gs://{bucket}/serblo/vertex/{register}"
    return f"{base}/input-{stamp}.jsonl", f"{base}/out-{stamp}/"


def submit(a, registers):
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    spent = load_done(out_dir) | load_inflight(out_dir)

    def eligible():
        """Stream un-spent manifest records (never materialize the whole batch: the VM
        is a 1GB micro; holding 500k prompt dicts swap-thrashes it)."""
        for reg in registers:
            for rec in iter_manifest(Path(a.manifest_dir) / f"{reg}.jsonl"):
                if rec["id"] not in spent:
                    yield rec

    # Pass 1: light sample -> avg input tokens (prompts within a register are uniform length,
    # so a few hundred pin the mean). Also grabs the first rec for the display sample.
    first, sample_chars, n_sample = None, 0, 0
    for rec in eligible():
        if first is None:
            first = rec
        sample_chars += len(rec["prompt"]); n_sample += 1
        if n_sample >= 300:
            break
    if n_sample == 0 or first is None:
        print(f"nothing to submit: {len(spent):,} ids already spent, "
              f"manifest exhausted or empty.", flush=True)
        return
    avg_in = sample_chars / n_sample / CHARS_PER_TOKEN

    # ---- SPEND GUARD: how many prompts the budget covers at WORST case (maxOutputTokens
    # makes worst-case exact -> the cap is a guarantee). target = min(limit, budget-fit).
    guard, target = None, (a.limit or None)
    if a.budget is not None:
        from serblo.generate.spend_guard import SpendGuard, per_prompt_worst_cost
        # ledger is per-ACCOUNT and shared across out dirs (parent of the out dirs):
        # a per-dir ledger would grant the full budget once per directory.
        guard = SpendGuard(out_dir.parent / f"spend-{a.account}.json", a.budget)
        per = per_prompt_worst_cost(avg_in, a.max_output_tokens, a.model)
        fit = guard.max_prompts(avg_in, a.max_output_tokens, a.model)
        if fit <= 0:
            raise SystemExit(f"spend guard [{a.account}]: ${guard.remaining():.2f} of "
                             f"${a.budget:.2f} left — can't afford one prompt (worst ${per:.5f}). Stop.")
        target = fit if target is None else min(target, fit)
        print(f"spend guard [{a.account}]: ${guard.remaining():.2f} left of ${a.budget:.2f} "
              f"-> up to {fit:,} prompts (worst ${per:.6f}/prompt)", flush=True)

    reg_tag = registers[0] if len(registers) == 1 else "mixed"
    stamp = time.strftime("%Y%m%dT%H%M%S")
    input_local = out_dir / f"input-{reg_tag}-{stamp}.jsonl"
    map_local = out_dir / f"input-{reg_tag}-{stamp}.map.jsonl"

    # Pass 2: STREAM-write up to `target` prompts (constant memory, resumable/crash-safe)
    written = 0
    with open(input_local, "w", encoding="utf-8") as f, open(map_local, "w", encoding="utf-8") as m:
        for rec in eligible():
            if target is not None and written >= target:
                break
            f.write(json.dumps(build_request(rec, a.model, a.temperature, a.max_output_tokens),
                               ensure_ascii=False) + "\n")
            m.write(json.dumps({"key": rec["id"], "register": rec["register"],
                                "axes": rec["axes"]}, ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        print("nothing to submit after guard.", flush=True)
        return

    price = PRICING.get(a.model)
    gcs_input, gcs_output = _gcs_paths(a.gcs_bucket or "<BUCKET>", reg_tag, stamp)

    print(f"\n{'='*68}\nSERBLO VERTEX BATCH — {'SUBMIT' if a.submit else 'DRY RUN (no spend)'}\n{'='*68}")
    print(f"register(s): {','.join(registers)}   model: {a.model}   temperature: {a.temperature}")
    print(f"location: {a.location}   spent-so-far: {len(spent):,}")
    print(f"prompts this batch: {written:,}   input written: {input_local}")
    print(f"\n--- sample request line (prompt truncated) ---")
    _s = build_request(first, a.model, a.temperature, a.max_output_tokens)
    _s["request"]["contents"][0]["parts"][0]["text"] = first["prompt"][:200] + " …"
    print(json.dumps(_s, ensure_ascii=False, indent=2))

    print(f"\n--- projected cost (Gemini BATCH pricing, 50% off) ---")
    if price is None:
        print(f"  model {a.model} not in PRICING table — cost unknown")
    else:
        in_tok = written * avg_in
        exp_out = written * a.est_output_tokens                # expected (assumed) output
        max_out = written * a.max_output_tokens                # WORST case (the guard's number)
        print(f"  {a.model}:  ${price['in']:.2f}/M in   ${price['out']:.2f}/M out   (cap {a.max_output_tokens} out-tok/gen)")
        print(f"  input : {written:,} x ~{avg_in:.0f} tok = {in_tok/1e6:.3f}M x ${price['in']:.2f} = ${in_tok/1e6*price['in']:.4f}")
        print(f"  output: expected ~{a.est_output_tokens}/gen = ${exp_out/1e6*price['out']:.4f}   |   "
              f"WORST {a.max_output_tokens}/gen = ${max_out/1e6*price['out']:.4f}")
        print(f"  TOTAL: expected ~${(in_tok+exp_out)/1e6*price['out'] if False else in_tok/1e6*price['in']+exp_out/1e6*price['out']:.4f}   "
              f"WORST-CASE (reserved) ${in_tok/1e6*price['in']+max_out/1e6*price['out']:.4f}")

    print(f"\n--- the two live steps a real --submit would run (NOT run now) ---")
    print(f"  1. gcloud storage cp {input_local} {gcs_input}")
    print(f"  2. client.batches.create(model={a.model!r}, src={gcs_input!r},")
    print(f"         config=CreateBatchJobConfig(dest={gcs_output!r}, display_name='serblo-{reg_tag}-{stamp}'))")

    if not a.submit:
        print(f"\nDRY RUN complete — nothing uploaded, no job created, $0 spent.")
        print(f"To submit for real: add --submit --gcs-bucket <bucket> --project <id> [--location {a.location}]\n")
        return

    # -------- REAL SUBMIT (spends credits): guarded, intentionally left un-exercised --------
    if not a.gcs_bucket:
        raise SystemExit("--submit requires --gcs-bucket (where the input JSONL is uploaded)")
    import google.genai as genai                              # lazy: dry run / tests never need it
    from google.genai.types import CreateBatchJobConfig

    print(f"\n>>> uploading input to {gcs_input} …", flush=True)
    up = subprocess.run(["gcloud", "storage", "cp", str(input_local), gcs_input],
                        capture_output=True, text=True)
    if up.returncode != 0:
        raise SystemExit(f"gcloud storage cp failed:\n{up.stderr}")

    client = genai.Client(vertexai=True, project=a.project, location=a.location)
    job = client.batches.create(
        model=a.model,
        src=gcs_input,
        config=CreateBatchJobConfig(dest=gcs_output, display_name=f"serblo-{reg_tag}-{stamp}"),
    )
    # ledger FIRST: the job is now spent even if the interpreter dies on the next line
    with open(out_dir / "jobs.jsonl", "a", encoding="utf-8") as jf:
        jf.write(json.dumps({
            "job_name": job.name, "state": str(job.state), "model": a.model,
            "register": reg_tag, "input_gcs": gcs_input, "output_gcs": gcs_output,
            "input_local": str(input_local), "map_local": str(map_local),
            "n": written, "stamp": stamp,                    # ids live in the .map sidecar
            "avg_in_tok": round(avg_in, 1), "max_out_tok": a.max_output_tokens,
            "account": a.account, "ts": int(time.time()), "retrieved": False,
        }, ensure_ascii=False) + "\n")
        jf.flush()
    if guard is not None:                                    # RESERVE worst-case now; retrieve settles to actual
        from serblo.generate.spend_guard import batch_worst_cost
        worst = batch_worst_cost(written, avg_in, a.max_output_tokens, a.model)
        guard.reserve(job.name, worst)
        print(f"spend guard [{a.account}]: reserved worst-case ${worst:.4f} | committed "
              f"${guard.committed():.2f}/${a.budget:.2f} | ${guard.remaining():.2f} left", flush=True)
    print(f"submitted: {job.name}  state={job.state}\n"
          f"poll with:  --poll --project {a.project} --location {a.location}", flush=True)


# ============================ phase 2: poll (written, not exercised) ============================

def poll_jobs(a):
    import google.genai as genai
    out_dir = Path(a.out)
    jobs = _read_jobs(out_dir)
    pending = [jn for jn, r in jobs.items() if not r.get("retrieved")]
    if not pending:
        print("no in-flight jobs in jobs.jsonl", flush=True)
        return
    client = genai.Client(vertexai=True, project=a.project, location=a.location)
    with open(out_dir / "jobs.jsonl", "a", encoding="utf-8") as jf:
        for jn in pending:
            job = client.batches.get(name=jn)
            state = str(job.state)
            print(f"{jn}: {state}", flush=True)
            jf.write(json.dumps({"job_name": jn, "state": state,
                                 "ts": int(time.time())}, ensure_ascii=False) + "\n")
            jf.flush()


# ============================ phase 3: retrieve + convert (written, not exercised) ============================

def _load_map(map_local):
    """input-order [(key, register, axes)]: id<->response fallback when `key` doesn't echo."""
    rows = []
    p = Path(map_local)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    rows.append((r["key"], r["register"], r["axes"]))
                except (json.JSONDecodeError, KeyError):
                    rows.append((None, None, None))
    return rows


def retrieve_jobs(a, registers):
    import google.genai as genai
    out_dir = Path(a.out)
    jobs = _read_jobs(out_dir)
    todo = [(jn, r) for jn, r in jobs.items() if not r.get("retrieved")]
    if not todo:
        print("no un-retrieved jobs in jobs.jsonl", flush=True)
        return
    client = genai.Client(vertexai=True, project=a.project, location=a.location)
    meta = manifest_meta(a.manifest_dir, registers)             # id -> (register, axes)
    done = load_done(out_dir)
    guard = None
    if a.budget is not None:
        from serblo.generate.spend_guard import SpendGuard
        guard = SpendGuard(out_dir.parent / f"spend-{a.account}.json", a.budget)  # shared per-account

    raw_f = open(out_dir / "raw.jsonl", "a", encoding="utf-8")
    docs_f = open(out_dir / "docs.jsonl", "a", encoding="utf-8")
    err_f = open(out_dir / "errors.jsonl", "a", encoding="utf-8")
    try:
        for jn, rec in todo:
            job = client.batches.get(name=jn)
            state = str(job.state)
            if not is_succeeded(state):
                print(f"{jn}: {state} — not SUCCEEDED, skipping", flush=True)
                continue
            out_gcs = rec.get("output_gcs")
            gmap = _load_map(rec.get("map_local", ""))          # order fallback
            gm = {k: (r_, a_) for k, r_, a_ in gmap if k}       # submit-time axes by key
            tmp = out_dir / "_retrieve_tmp"
            if tmp.exists():                                    # clean per job: stale shards from a
                shutil.rmtree(tmp)                              # prior job would pollute rglob + the
            tmp.mkdir()                                         # line-order fallback
            # results are (possibly sharded/nested) JSONL under the dest prefix; mirror the whole
            # prefix locally with gcloud storage (gsutil absent), then rglob every *.jsonl below it.
            cp = subprocess.run(["gcloud", "storage", "cp", "--recursive",
                                 out_gcs.rstrip("/"), str(tmp)],
                                capture_output=True, text=True)
            if cp.returncode != 0:
                print(f"{jn}: download failed:\n{cp.stderr}", flush=True)
                continue

            n_docs = 0
            n_out_words = 0
            use_in = use_out = 0                             # BILLING tokens (usageMetadata)
            line_i = -1
            for shard in sorted(tmp.rglob("*.jsonl")):
                with open(shard, encoding="utf-8") as sf:
                    for line in sf:
                        line = line.strip()
                        if not line:
                            continue
                        line_i += 1
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        ui, uo = usage_of_line(obj)          # count EVERY line: Google billed it
                        use_in += ui; use_out += uo
                        key, text = parse_output_line(obj, a.model)
                        if key is None and 0 <= line_i < len(gmap):   # `key` didn't echo -> order
                            key = gmap[line_i][0]
                        if key is None:
                            continue
                        if text is None:
                            err_f.write(json.dumps({"id": key, "reason": "no_text",
                                                    "ts": int(time.time())}) + "\n"); err_f.flush()
                            continue
                        if key in done:
                            continue
                        done.add(key)
                        # provenance preference: the .map.jsonl sidecar froze register+axes
                        # AT SUBMIT TIME: the manifest may have been regenerated since
                        # (particles, grounding, ...) and would lie about what was prompted.
                        if key in gm:
                            reg, axes = gm[key]
                        elif key in meta:
                            reg, axes = meta[key]
                        else:
                            reg, axes = registers[0], {}
                        # register+axes embedded at retrieve time: provenance never
                        # depends on a later manifest lookup (see run_mimo)
                        raw_f.write(json.dumps({"id": key, "model": a.model,
                                                "register": reg, "axes": axes, "raw": text,
                                                "ts": int(time.time())}, ensure_ascii=False) + "\n")
                        raw_f.flush()
                        n_out_words += len(text.split())
                        drecs, _ = doc_records(key, reg, axes, text, a.model)
                        for d in drecs:
                            docs_f.write(json.dumps(d, ensure_ascii=False) + "\n"); n_docs += 1
                        docs_f.flush()
            # SPEND GUARD settle: swap the worst-case reservation for ACTUAL cost, from
            # usageMetadata (Google's billing tokens, the only trustworthy numbers).
            # If usage was absent, fall back to a PADDED estimate (2.6 tok/word, over-
            # counting actual keeps the invariant safe; never settle below truth).
            if guard is not None:
                from serblo.generate.spend_guard import actual_cost
                if use_in + use_out > 0:
                    act = actual_cost(use_in, use_out, rec.get("model", a.model))
                    src = f"usage in={use_in:,} out={use_out:,}"
                else:
                    in_tok = rec.get("n", 0) * rec.get("avg_in_tok", 0)
                    act = actual_cost(in_tok, n_out_words * 2.6, rec.get("model", a.model))
                    src = "PADDED word-estimate (no usageMetadata)"
                guard.settle(jn, act)
                print(f"spend guard [{rec.get('account')}]: settled {jn} actual ${act:.4f} "
                      f"({src}) | committed ${guard.committed():.2f}/${guard.data['budget']:.2f} "
                      f"| ${guard.remaining():.2f} left", flush=True)
            # mark retrieved (append-only status line)
            with open(out_dir / "jobs.jsonl", "a", encoding="utf-8") as jf:
                jf.write(json.dumps({"job_name": jn, "retrieved": True,
                                     "ts": int(time.time())}, ensure_ascii=False) + "\n")
            print(f"{jn}: retrieved -> {n_docs} docs", flush=True)
    finally:
        raw_f.close(); docs_f.close(); err_f.close()


# ============================ CLI ============================

def main():
    ap = argparse.ArgumentParser(description="Gemini batch generation via Vertex AI (google-genai).")
    ap.add_argument("--manifest-dir", default=str(ROOT / "data" / "manifests"))
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "vertex"))
    ap.add_argument("--register", choices=["essay", "conversational", "both", "sft"], default="essay")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--temperature", type=float, default=ESSAY_TEMPERATURE)
    ap.add_argument("--limit", type=int, default=0, help="max prompts in this batch (0 = all remaining)")
    ap.add_argument("--est-output-tokens", type=int, default=EST_OUTPUT_TOKENS_DEFAULT,
                    help="assumed output tokens/essay for the cost estimate")
    # SPEND GUARD (paid accounts): client-side hard cap, no Google budget API
    ap.add_argument("--account", default="default",
                    help="account label — its own spend ledger (spend-<account>.json)")
    ap.add_argument("--budget", type=float, default=None,
                    help="hard USD cap for this account; batches trim/refuse to never exceed it")
    ap.add_argument("--max-output-tokens", type=int, default=768,
                    help="per-gen output cap — bounds worst-case cost, making --budget a guarantee")
    # live-only:
    ap.add_argument("--submit", action="store_true", help="REALLY submit (spends credits); needs --gcs-bucket")
    ap.add_argument("--poll", action="store_true", help="poll in-flight jobs (live get)")
    ap.add_argument("--retrieve", action="store_true", help="download + convert SUCCEEDED jobs (live)")
    ap.add_argument("--rebuild", action="store_true", help="re-derive docs.jsonl from raw.jsonl (no network)")
    ap.add_argument("--gcs-bucket", default=None, help="bucket name for input/output (no gs:// prefix)")
    ap.add_argument("--project", default=None, help="GCP project (default: SDK/ADC auto-detect)")
    ap.add_argument("--location", default="us-central1")
    a = ap.parse_args()

    registers = ["essay", "conversational"] if a.register == "both" else [a.register]

    if a.rebuild:
        rebuild(a.out, a.manifest_dir, registers)
    elif a.poll:
        poll_jobs(a)
    elif a.retrieve:
        retrieve_jobs(a, registers)
    else:
        submit(a, registers)


if __name__ == "__main__":
    main()
