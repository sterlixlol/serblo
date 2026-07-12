"""MiMo free-channel generation driver: manifest → opencode → §7.9 converter (PRD §7.4).

Walks the prompt manifest and, per spec, calls `opencode run --pure --agent
serblo-{conv,essay}` (the no-tools project agents pinned to opencode/mimo-v2.5-free),
saves the RAW response to disk immediately, converts it (extract_docs), and appends the
clean docs, one item at a time, append-only, resumable.

Layout under --out (append-only JSONL):
  raw.jsonl    {id, agent, raw, secs, ts}   ground truth; written BEFORE conversion,
                                              so docs are re-derivable forever
  docs.jsonl   {id, text, source, register, axes, model, words, issues}
  errors.jsonl {id, rc, calls, stderr_tail, ts}  failed ids (no raw line -> retried)

Resume: an id in raw.jsonl = request SPENT -> never re-sent. Failures get no raw line ->
retried next run. A crash between raw- and docs-append can double-convert one id on the
next rebuild; pooled exact-dedup (M2) collapses it.

Stops: --max-requests budget (REAL opencode calls, retries included; may overshoot by up
to len(BACKOFF) because an id's retries are not split across runs) · 3 consecutive failed
ids (systemic: quota / network / OC down) · 5 consecutive empty conversions (format broke,
don't burn quota on garbage). --rebuild refuses while a generation run is live (its
docs.jsonl replace would strand the live writer's handle).

Usage:
  PYTHONPATH=src python -m serblo.generate.run_mimo --max-requests 100
  PYTHONPATH=src python -m serblo.generate.run_mimo --register essay --max-requests 50
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.generate.convert import extract_docs, pick_payload_doc  # noqa: E402

AGENT = {"conversational": "serblo-conv", "essay": "serblo-essay"}
# Two independent FREE channels to the same MiMo v2.5 family, separate quota pools:
# OpenCode's free tier and Xiaomi's MiMo Code (an OpenCode fork; no registration).
# Same agent names; each CLI reads its own project agent dir.
CLI = {
    "opencode": {"bin": "opencode", "agent_dir": ".opencode/agent", "model": "opencode/mimo-v2.5-free"},
    "mimo":     {"bin": "mimo",     "agent_dir": ".mimocode/agent", "model": "mimo/mimo-auto"},
}
MAX_FAIL_STREAK = 3
MAX_EMPTY_STREAK = 5
# MiMo free tier burst-throttles: rapid calls fast-fail (empty, <5s) or hang (timeout),
# then clear after a cooldown. Retry the same id with backoff before giving up: a
# throttled request costs quota but the backoff lets the window reset. Verified:
# reasoningEffort/enable_thinking are NOT honored on mimo-v2.5-free, so this (not a
# thinking knob) is the actual efficiency lever.
BACKOFF = [25, 55]                                        # sleeps before retry 1, retry 2


def doc_records(rid, register, axes, raw, model):
    """Convert one raw payload -> AT MOST ONE docs.jsonl record (shared by loop + rebuild).

    salvage_unclosed=True: MiMo often mangles the closer (bare brackets / glued END);
    a salvaged doc with a stripped tail is still good prose. pick_payload_doc then
    enforces 1-prompt=1-doc: junk fragments (leaked reasoning) gated out, stuttered
    copies collapsed, last survivor kept. All markers survive in `issues`.
    """
    docs, issues = extract_docs(raw, salvage_unclosed=True)
    doc, pol_issues = pick_payload_doc(docs)
    issues = issues + pol_issues
    if doc is None:
        return [], issues
    return [
        {"id": f"{rid}-d0", "text": doc, "source": "mimo", "register": register,
         "axes": axes, "model": model, "words": len(doc.split()),
         "issues": issues if issues else None}
    ], issues


def _live_generation_run():
    """pid of a live run_mimo generation process (not a --rebuild), else None.

    Guards --rebuild: its tmp+replace of docs.jsonl would strand a live run's open
    file handle on the unlinked inode; the live run's subsequent appends silently
    vanish from the visible file. Refuse instead."""
    me = os.getpid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "run_mimo" in cmd and "--rebuild" not in cmd:
            return int(p.name)
    return None


def rebuild(out_dir, manifest_dir):
    """Re-derive docs.jsonl from raw.jsonl + manifest. Zero requests spent.

    Run after any converter improvement, recovering payloads that older parsing
    stranded. Atomic replace; keeps first occurrence per id (crash-retry dups).
    """
    pid = _live_generation_run()
    if pid:
        raise SystemExit(f"live run_mimo generation (pid {pid}) is appending docs.jsonl — "
                         f"rebuild would strand its writes. Wait for it to finish.")
    meta = {}
    for reg in AGENT:
        path = manifest_dir / f"{reg}.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    meta[r["id"]] = (r["register"], r["axes"])
    raw_path = out_dir / "raw.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"nothing to rebuild: {raw_path} missing")
    tmp = out_dir / "docs.jsonl.tmp"
    seen, n_raw, n_docs, n_words, missing = set(), 0, 0, 0, 0
    from collections import Counter
    issue_tally = Counter()
    with open(raw_path, encoding="utf-8") as rf, open(tmp, "w", encoding="utf-8") as w:
        for line in rf:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                                # crash-truncated tail
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            n_raw += 1
            if "axes" in r and "register" in r:          # generation-time provenance wins
                reg, axes = r["register"], r["axes"]
            elif r["id"] in meta:                        # legacy lines: manifest fallback
                reg, axes = meta[r["id"]]
            else:
                missing += 1
                print(f"warn: {r['id']} not in manifest — skipped", flush=True)
                continue
            model = r.get("model") or CLI.get(r.get("cli", "opencode"), CLI["opencode"])["model"]  # run_cli raw carries its own model
            recs, issues = doc_records(r["id"], reg, axes, r["raw"], model)
            issue_tally.update(issues)
            for rec in recs:
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_docs += 1
                n_words += rec["words"]
    tmp.replace(out_dir / "docs.jsonl")
    print(f"rebuilt docs.jsonl: {n_raw:,} raw -> {n_docs:,} docs, {n_words:,} words"
          f"{f', {missing} missing from manifest' if missing else ''}", flush=True)
    if issue_tally:                                     # silent failures must leave a visible trail
        print("issue tally:", ", ".join(f"{k}={v}" for k, v in issue_tally.most_common()), flush=True)


def load_done(out_dir):
    """ids already SPENT (present in raw.jsonl). Tolerates a corrupt last line."""
    done, path = set(), out_dir / "raw.jsonl"
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass                                    # crash-truncated tail -> retried
    return done


def iter_manifest(manifest_dir, registers):
    for reg in registers:
        path = manifest_dir / f"{reg}.jsonl"
        if not path.exists():
            print(f"note: no manifest for {reg} ({path})", flush=True)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def generate_one(prompt, agent, timeout, bin_="opencode"):
    t = time.time()
    try:
        p = subprocess.run(
            [bin_, "run", "--pure", "--agent", agent, prompt],
            capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
        return p.returncode, p.stdout, p.stderr, time.time() - t
    except subprocess.TimeoutExpired as e:
        err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return -1, "", f"TIMEOUT after {timeout}s\n{err}", time.time() - t


def generate(prompt, agent, timeout, bin_="opencode"):
    """generate_one + throttle-aware retry. Returns (rc, raw, stderr, wall_secs, calls).

    Retries a throttle-shaped failure (empty output or timeout) after a backoff; each
    retry is a real CLI call (spends quota), so tries are capped at len(BACKOFF)+1.
    """
    wall = 0.0
    last = (-1, "", "", 0.0)
    for attempt in range(len(BACKOFF) + 1):
        rc, raw, stderr, secs = generate_one(prompt, agent, timeout, bin_)
        wall += secs
        if rc == 0 and raw.strip():
            return rc, raw, stderr, wall, attempt + 1
        last = (rc, raw, stderr, secs)
        if attempt < len(BACKOFF):
            time.sleep(BACKOFF[attempt])                 # let the burst window reset
    return last[0], last[1], last[2], wall, len(BACKOFF) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default=str(ROOT / "data" / "manifests"))
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "mimo"))
    ap.add_argument("--register", choices=["conversational", "essay", "both"], default="both")
    ap.add_argument("--max-requests", type=int, default=100,
                    help="quota budget for THIS run = real opencode calls incl. retries")
    ap.add_argument("--timeout", type=int, default=120,
                    help="seconds per call (max clean success observed ~103s; past ~120s = throttle-hang)")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-derive docs.jsonl from raw.jsonl + manifest (no requests)")
    ap.add_argument("--cli", choices=sorted(CLI), default="opencode",
                    help="which free MiMo channel to spend: opencode or mimo (MiMo Code)")
    a = ap.parse_args()

    if a.rebuild:
        rebuild(Path(a.out), Path(a.manifest_dir))
        return

    # preflight: fail before spending anything
    cli = CLI[a.cli]
    if not shutil.which(cli["bin"]):
        raise SystemExit(f"{cli['bin']} binary not found on PATH")
    registers = ["conversational", "essay"] if a.register == "both" else [a.register]
    for reg in registers:
        agent_file = ROOT / cli["agent_dir"] / f"{AGENT[reg]}.md"
        if not agent_file.exists():
            raise SystemExit(f"missing agent config: {agent_file}")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(out_dir)
    print(f"resume: {len(done):,} ids already spent", flush=True)

    attempted = ok = failed = empty = n_docs = n_words = requests = 0
    fail_streak = empty_streak = 0
    stop = None
    t0 = time.time()

    raw_f = open(out_dir / "raw.jsonl", "a", encoding="utf-8")
    docs_f = open(out_dir / "docs.jsonl", "a", encoding="utf-8")
    err_f = open(out_dir / "errors.jsonl", "a", encoding="utf-8")
    try:
        for rec in iter_manifest(Path(a.manifest_dir), registers):
            if rec["id"] in done:
                continue
            if requests >= a.max_requests:                # budget = real opencode calls (retries included)
                stop = f"budget: {requests}/{a.max_requests} requests spent"
                break
            attempted += 1
            agent = AGENT[rec["register"]]
            rc, raw, stderr, secs, calls = generate(rec["prompt"], agent, a.timeout, cli["bin"])
            requests += calls

            if rc != 0 or not raw.strip():
                failed += 1
                fail_streak += 1
                err_f.write(json.dumps(
                    {"id": rec["id"], "rc": rc, "calls": calls, "stderr_tail": stderr[-500:],
                     "ts": int(time.time())}, ensure_ascii=False) + "\n")
                err_f.flush()
                print(f"{rec['id']}: FAIL rc={rc} {secs:5.1f}s x{calls} (streak {fail_streak})", flush=True)
                if fail_streak >= MAX_FAIL_STREAK:
                    stop = f"{MAX_FAIL_STREAK} consecutive failures — systemic, stopping"
                    break
                continue
            fail_streak = 0

            # raw first: id becomes "spent" even if conversion yields nothing.
            # register+axes embedded HERE (generation time): provenance must never
            # depend on a later manifest lookup (manifests get regenerated; the axes
            # that primed THIS generation are frozen in this record).
            raw_f.write(json.dumps(
                {"id": rec["id"], "agent": agent, "cli": a.cli,
                 "register": rec["register"], "axes": rec["axes"], "raw": raw,
                 "secs": round(secs, 1), "ts": int(time.time())}, ensure_ascii=False) + "\n")
            raw_f.flush()
            ok += 1

            recs, issues = doc_records(rec["id"], rec["register"], rec["axes"], raw, cli["model"])
            for drec in recs:
                n_docs += 1
                n_words += drec["words"]
                docs_f.write(json.dumps(drec, ensure_ascii=False) + "\n")
            docs_f.flush()

            if recs:
                empty_streak = 0
            else:
                empty += 1
                empty_streak += 1
                if empty_streak >= MAX_EMPTY_STREAK:
                    stop = f"{MAX_EMPTY_STREAK} consecutive empty conversions — format broke, stopping"
                    break
            words = sum(r["words"] for r in recs)
            retry_tag = f" x{calls}" if calls > 1 else ""
            print(f"{rec['id']}: {secs:5.1f}s{retry_tag} docs={len(recs)} words={words}"
                  f"{' issues=' + ','.join(issues) if issues else ''}"
                  f"  [run: {ok}/{attempted} ok, {requests} req, {n_words:,} words]", flush=True)
    except KeyboardInterrupt:
        stop = "interrupted — everything before this item is saved"
    finally:
        raw_f.close(); docs_f.close(); err_f.close()

    mins = (time.time() - t0) / 60
    print(f"\n=== run summary ===")
    print(f"ids attempted {attempted} | ok {ok} | failed {failed} | empty-convert {empty}")
    print(f"requests spent {requests} (incl. retries) | docs {n_docs} | words {n_words:,} | {mins:.1f} min")
    if stop:
        print(f"stopped: {stop}")
    print(f"cumulative spent ids: {len(done) + ok:,}", flush=True)


if __name__ == "__main__":
    main()
