#!/usr/bin/env python3
"""Generalized CLI generation driver: drives opencode / mimo / cline as text-gen backends.

Each (cli, model) is a distinct LINEAGE for the multi-lineage pool. Mirrors run_mimo's
append-only / resume / throttle-retry machinery; adds a --model dimension and per-CLI
invocation + output parsing:

  opencode  `opencode run --pure --dir ROOT --agent A --model M <prompt>`   -> clean stdout
  mimo      `mimo run --pure --agent A <prompt>`  (agent pins the model)    -> clean stdout
  cline     `cline --provider cline --model M --auto-approve false --json
             --system <sys> <prompt>`  -> JSONL events; final text lives in the
             content_end / contentType="text" events; per-call $ in usage.totalCost

SAFETY: cline METERS. Measured: ~$0.047 cold / ~$0.005 warm per call (~6.8k tokens
of agent scaffolding injected per call at frontier rates). opencode/mimo are $0 (verified:
`opencode stats` = $0.00 over 730 sessions). So --max-cost defaults LOW and halts a metered run
before it can run away; free CLIs never trip it (parsed cost stays 0). auto-approve=false makes
cline deny any tool/skill attempt and fall back to plain text (no hang, nothing executed).

Usage:
  PYTHONPATH=src python -m serblo.generate.run_cli --cli opencode \
      --model opencode/deepseek-v4-flash-free --out data/generated/oc-dsflash --max-requests 30
  PYTHONPATH=src python -m serblo.generate.run_cli --cli cline \
      --model anthropic/claude-opus-4.8 --out data/generated/cline-opus \
      --max-requests 20 --max-cost 0.50
"""
import argparse
import glob
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.generate.convert import extract_docs, pick_payload_doc      # noqa: E402
from serblo.generate.run_mimo import iter_manifest, load_done           # noqa: E402  (shared, tested)

AGENT = {"conversational": "serblo-conv", "essay": "serblo-essay"}
AGENT_DIR = {"opencode": ".opencode/agent", "mimo": ".mimocode/agent", "cline": ".opencode/agent"}
DEFAULT_MODEL = {"opencode": "opencode/mimo-v2.5-free", "mimo": "mimo/mimo-auto",
                 "cline": "anthropic/claude-opus-4.8", "agy": "claude-opus-4.6"}
CLIS = ["opencode", "mimo", "cline", "agy"]                 # agy: no agent dir, prompt-only --print
METERED = {"cline"}                                         # cost PARSED for visibility; ENFORCED only if
CWD_ISOLATED = {"cline", "agy"}                             # --max-cost>0. agentic CLIs run in an empty dir
BACKOFF = [10, 30]                                           # sleeps before retry 1, retry 2
MAX_FAIL_STREAK = 3
MAX_EMPTY_STREAK = 5


def resolve_bin(cli):
    if cli == "cline":
        b = shutil.which("cline")
        if b:
            return b
        hits = glob.glob("/run/user/*/fnm_multishells/*/bin/cline")
        return hits[0] if hits else "cline"
    known = {"opencode": Path.home() / ".opencode" / "bin" / "opencode",
             "mimo": Path.home() / ".mimocode" / "bin" / "mimo",
             "agy": Path.home() / ".local" / "bin" / "agy"}
    b = shutil.which(cli)
    if b:
        return b
    kb = known.get(cli)
    return str(kb) if kb and kb.exists() else cli


def build_argv(cli, bin_, model, agent, system, prompt):
    if cli == "opencode":
        return [bin_, "run", "--pure", "--dir", str(ROOT), "--agent", agent, "--model", model, prompt]
    if cli == "mimo":
        return [bin_, "run", "--pure", "--agent", agent, prompt]        # agent pins mimo's model
    if cli == "cline":
        return [bin_, "--provider", "cline", "--model", model, "--auto-approve", "false",
                "--json", "--system", system, prompt]
    if cli == "agy":
        return [bin_, "--print", prompt]        # --model is ignored in --print; agy uses the pinned default
    raise ValueError(f"unknown cli {cli}")


def parse_output(cli, stdout):
    """-> (text, cost). opencode/mimo return clean stdout ($0). cline returns JSONL events:
    final answer = concatenated content_end/contentType=text; per-call $ = last usage.totalCost."""
    if cli not in METERED:
        return stdout.strip(), 0.0
    parts, cost = [], 0.0
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue                                          # non-JSON log line, skip
        if ev.get("type") != "agent_event":
            continue
        e = ev.get("event", {}) or {}
        if e.get("type") == "content_end" and e.get("contentType") == "text":
            parts.append(e.get("text", "") or "")
        elif e.get("type") == "usage" and "totalCost" in e:
            cost = e["totalCost"]        # last-wins: totalCost is CUMULATIVE (sample 0.047->0.052=0.047+0.005)
    return "\n".join(p for p in parts if p).strip(), cost


def system_prompt(cli, agent):
    """Agent .md body (after YAML frontmatter), passed to cline via --system so all CLIs share
    the same text-only instruction. opencode/mimo read it themselves via --agent."""
    f = ROOT / AGENT_DIR[cli] / f"{agent}.md"
    txt = f.read_text(encoding="utf-8")
    if txt.startswith("---"):
        seg = txt.split("---", 2)
        return seg[2].strip() if len(seg) >= 3 else txt.strip()
    return txt.strip()


def doc_records(rid, register, axes, raw, source, model):
    """One raw payload -> at most one docs.jsonl record. lenient: if the model dropped the
    §7.9 wrapper entirely, take the whole output (the junk gate in pick_payload_doc still
    filters non-Serbian/short garbage)."""
    docs, issues = extract_docs(raw, salvage_unclosed=True)
    if not docs and "no_sentinels" in issues:
        docs = [raw.strip()]
        issues = issues + ["lenient_no_sentinel"]
    doc, pol = pick_payload_doc(docs)
    issues = issues + pol
    if doc is None:
        return [], issues
    return [{"id": f"{rid}-d0", "text": doc, "source": source, "register": register, "axes": axes,
             "model": model, "words": len(doc.split()), "issues": issues if issues else None}], issues


def spent_cost(out_dir):
    """Sum of every billed $ already ledgered for this out-dir (raw.jsonl successes + errors.jsonl
    failures both carry `cost`). Seeds total_cost so --max-cost is a PERSISTENT lifetime cap, not
    per-run: a self-restarting loop can't reset it to $0 and blow past the cap."""
    total = 0.0
    for name in ("raw.jsonl", "errors.jsonl"):
        p = out_dir / name
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    total += float(json.loads(line).get("cost", 0) or 0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
    return total


def failed_ids(out_dir, threshold):
    """ids that have failed >= threshold times (from errors.jsonl). For METERED CLIs, skip these so
    a permanently-broken id isn't re-billed up to 3 calls every relaunch."""
    from collections import Counter
    c = Counter()
    p = out_dir / "errors.jsonl"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    c[json.loads(line)["id"]] += 1
                except (json.JSONDecodeError, KeyError):
                    pass
    return {i for i, n in c.items() if n >= threshold}


def generate(cli, bin_, model, agent, system, prompt, timeout, cwd):
    """generate + throttle-aware retry. Returns (rc, text, cost, stderr_tail, wall, calls).
    cwd: repo ROOT for opencode/mimo (they read their agent dirs); an isolated empty dir for the
    agentic CLIs (cline/agy) so they answer the prompt instead of wandering the repo."""
    wall, cost = 0.0, 0.0
    last = (-1, "", "", 0.0)
    for attempt in range(len(BACKOFF) + 1):
        argv = build_argv(cli, bin_, model, agent, system, prompt)
        t = time.time()
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:                # a timed-out call STILL billed usage: the
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            rc, err = -1, f"TIMEOUT after {timeout}s"          # partial stream carries usage/text events
        wall += time.time() - t
        text, c = parse_output(cli, out)                      # counts cost even from a timed-out partial
        cost += c
        if rc == 0 and text:
            return rc, text, cost, err[-400:], wall, attempt + 1
        last = (rc, text, err[-400:], c)
        if attempt < len(BACKOFF):
            time.sleep(BACKOFF[attempt])
    return last[0], last[1], cost, last[2], wall, len(BACKOFF) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", choices=CLIS, required=True)
    ap.add_argument("--model", help="provider/model (default = the CLI's default model)")
    ap.add_argument("--register", choices=["conversational", "essay", "both"], default="conversational")
    ap.add_argument("--manifest-dir", default=str(ROOT / "data" / "manifests"))
    ap.add_argument("--out", help="OWN dir per (cli,model); default data/generated/<cli>_<model>")
    ap.add_argument("--max-requests", type=int, default=30, help="budget = real CLI calls incl. retries")
    ap.add_argument("--max-cost", type=float, default=0.25,
                    help="LIFETIME $ cap for this out-dir (metered CLIs only): persisted via the cost "
                         "ledger in raw/errors.jsonl so relaunches DON'T reset it; free CLIs never hit it. "
                         "0 = uncapped (cost still tracked+logged — for CLIs confirmed free, e.g. cline). "
                         "A run may overshoot by <=1 generate() batch (<=3 billed calls) past the cap.")
    ap.add_argument("--max-id-fails", type=int, default=3,
                    help="metered CLIs: skip a manifest id already failed this many times, so a broken "
                         "id isn't re-billed every relaunch")
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()

    model = a.model or DEFAULT_MODEL[a.cli]
    bin_ = resolve_bin(a.cli)
    if not shutil.which(bin_) and not Path(bin_).exists():
        raise SystemExit(f"{a.cli} binary not found ({bin_})")
    registers = ["conversational", "essay"] if a.register == "both" else [a.register]
    system = {}
    if a.cli in AGENT_DIR:                                    # agy has no agent config: the manifest
        for reg in registers:                                # prompt (§7.9) carries all the instruction
            af = ROOT / AGENT_DIR[a.cli] / f"{AGENT[reg]}.md"
            if not af.exists():
                raise SystemExit(f"missing agent config: {af}")
        system = {reg: system_prompt(a.cli, AGENT[reg]) for reg in registers}

    out_dir = Path(a.out) if a.out else ROOT / "data" / "generated" / f"{a.cli}_{model.replace('/', '-')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(out_dir)
    metered = a.cli in METERED
    total_cost = spent_cost(out_dir) if metered else 0.0     # PERSISTENT ledger: --max-cost is a
    if metered and a.max_id_fails > 0:                        # LIFETIME cap for this out-dir;
        done = done | failed_ids(out_dir, a.max_id_fails)     # + skip ids that keep failing
    run_cwd = tempfile.mkdtemp(prefix="serblo_cli_") if a.cli in CWD_ISOLATED else str(ROOT)
    if metered and a.max_cost <= 0:
        cap_str = f" | ${total_cost:.4f} nominal, UNCAPPED (logged only)"
    elif metered:
        cap_str = f" | ${total_cost:.4f}/${a.max_cost:.2f} lifetime cap"
    else:
        cap_str = " | FREE"
    print(f"{a.cli}/{model} -> {out_dir}", flush=True)
    print(f"resume {len(done):,} spent | budget {a.max_requests} calls{cap_str}", flush=True)

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
            if requests >= a.max_requests:
                stop = f"budget: {requests}/{a.max_requests} calls spent"
                break
            if metered and a.max_cost > 0 and total_cost >= a.max_cost:
                stop = f"cost cap: ${total_cost:.4f}/${a.max_cost:.2f} spent"
                break
            attempted += 1
            agent = AGENT[rec["register"]]
            rc, text, cost, err, secs, calls = generate(
                a.cli, bin_, model, agent, system.get(rec["register"], ""), rec["prompt"], a.timeout, run_cwd)
            requests += calls
            total_cost += cost

            if rc != 0 or not text:
                failed += 1
                fail_streak += 1
                err_f.write(json.dumps({"id": rec["id"], "rc": rc, "calls": calls, "cost": round(cost, 5),
                                        "stderr_tail": err, "ts": int(time.time())}, ensure_ascii=False) + "\n")
                err_f.flush()
                print(f"{rec['id']}: FAIL rc={rc} {secs:5.1f}s x{calls} (streak {fail_streak})", flush=True)
                if fail_streak >= MAX_FAIL_STREAK:
                    stop = f"{MAX_FAIL_STREAK} consecutive failures — systemic, stopping"
                    break
                continue
            fail_streak = 0

            raw_f.write(json.dumps({"id": rec["id"], "cli": a.cli, "model": model, "agent": agent,
                                    "register": rec["register"], "axes": rec["axes"], "raw": text,
                                    "cost": round(cost, 5), "secs": round(secs, 1),
                                    "ts": int(time.time())}, ensure_ascii=False) + "\n")
            raw_f.flush()
            ok += 1

            recs, issues = doc_records(rec["id"], rec["register"], rec["axes"], text, f"{a.cli}:{model}", model)
            for d in recs:
                n_docs += 1
                n_words += d["words"]
                docs_f.write(json.dumps(d, ensure_ascii=False) + "\n")
            docs_f.flush()
            if recs:
                empty_streak = 0
            else:
                empty += 1
                empty_streak += 1
                if empty_streak >= MAX_EMPTY_STREAK:
                    stop = f"{MAX_EMPTY_STREAK} consecutive empty conversions — format broke, stopping"
                    break
            cost_tag = f" ${cost:.4f}" if metered else ""
            retry_tag = f" x{calls}" if calls > 1 else ""
            print(f"{rec['id']}: {secs:5.1f}s{retry_tag}{cost_tag} docs={len(recs)} "
                  f"words={sum(r['words'] for r in recs)}"
                  f"{' issues=' + ','.join(issues) if issues else ''} "
                  f"[{ok}/{attempted} ok, {requests} req, {n_words:,} w"
                  f"{f', ${total_cost:.3f}' if metered else ''}]", flush=True)
    except KeyboardInterrupt:
        stop = "interrupted — everything before this item is saved"
    finally:
        raw_f.close(); docs_f.close(); err_f.close()
        if a.cli in CWD_ISOLATED:
            shutil.rmtree(run_cwd, ignore_errors=True)

    print(f"\n=== {a.cli}/{model} ===")
    print(f"attempted {attempted} | ok {ok} | failed {failed} | empty {empty} | req {requests} | "
          f"docs {n_docs} | words {n_words:,}"
          f"{f' | COST ${total_cost:.4f}' if metered else ' | $0 (free)'} | {(time.time() - t0) / 60:.1f} min")
    if stop:
        print(f"stopped: {stop}", flush=True)


if __name__ == "__main__":
    main()
