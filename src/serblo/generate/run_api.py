"""OpenAI-compatible HTTP generation driver: NIM/Groq open-weight channels (PRD roster).

Mirrors run_mimo's contract (raw-first append-only, resume-by-spent-id, embedded axes,
1-prompt=1-doc via pick_payload_doc) but talks to an OpenAI-compatible chat endpoint
instead of a CLI. Round-robins a pool of free-tier keys from .env and rotates on 429.

Roster (native-ear selected): GLM-5.2 (z-ai, fast open-weight volume)
+ DeepSeek-v4-pro (quality seasoning, slow: high timeout, low budget). gpt-oss/Llama
(forced-informal), Nemotron (inconsistent), Mistral (broken) all rejected.

CONCURRENCY: these hosted models are LATENCY-bound (a call sits 27-48s waiting), not
rate-bound (measured 0 429s at 144 docs/hr on GLM), so a serial loop wastes the free-tier
ceiling. `--concurrency K` fires K requests in parallel from a thread pool (blocking urllib
releases the GIL during socket I/O, so K threads = K concurrent network waits) and lifts
throughput toward the rate limit. Only the MAIN thread writes to disk (workers just return
results), so appends stay serialized and incremental. Push K until 429s
appear to find the real ceiling.

A system message enforces text-only output; the manifest prompt already carries the §7.9
wrapper. Raw open-weight models comply loosely; issue rates are watched live like MiMo.

Usage (per (host,model,register) into its OWN out dir so ledgers don't mix):
  PYTHONPATH=src python -m serblo.generate.run_api --host nim --model z-ai/glm-5.2 \
      --register conversational --out data/generated/glm --max-requests 300 --concurrency 24
"""
import argparse
import itertools
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from serblo.generate.convert import extract_docs, pick_payload_doc  # noqa: E402

HOSTS = {
    "nim": ("https://integrate.api.nvidia.com/v1/chat/completions", "NIM_KEYS"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_KEYS"),
    "grok": ("https://api.x.ai/v1/chat/completions", "XAI_KEYS"),   # xAI, PAID (~$4 credit), seasoning only
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_KEYS"),   # DROPPED (429 wall)
    "sambanova": ("https://api.sambanova.ai/v1/chat/completions", "SAMBANOVA_KEYS"),   # 8 free keys, Llama-3.3-70B
    "deepseek": ("https://api.deepseek.com/v1/chat/completions", "DEEPSEEK_KEYS"),   # PAID, deepseek-v4-pro
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_KEYS"),  # 10 free keys, gemini-2.5-flash
    "openrouter2": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER2_KEYS"),  # 2nd OR account, deepseek-v4-flash
    "mimo": ("https://api.xiaomimimo.com/v1/chat/completions", "MIMO_KEYS"),  # Xiaomi, PAID, mimo-v2.5-pro ($0.435/$0.87)
}                                                          # groq: 8 free keys, llama-3.3-70b-versatile (both per-key)
SYSTEM = ("You are a text generator, not an assistant. Output ONLY the requested text in "
          "the exact delimiter format asked for. No preamble, no analysis, no translation, "
          "no meta-commentary. Write natural Serbian, never English.")
MAX_FAIL_STREAK = 4
MAX_EMPTY_STREAK = 6
BACKOFF = [3, 10, 30]                                     # 429 / error backoff, rotating keys between


class RateLimiter:
    """Hands out keys while PACING each one to <= rpm_per_key requests/minute, so we stay under
    the provider's limit by construction instead of hammering it and eating 429s + backoff. A key
    is 'used' (reserved for the next 60/rpm seconds) at the moment it's handed out; the request
    itself can take much longer; providers cap requests *started* per minute, not concurrent ones.
    rpm_per_key <= 0 disables pacing (plain thread-safe round-robin)."""

    def __init__(self, keys, rpm_per_key):
        self.keys = list(keys)
        self.interval = 60.0 / rpm_per_key if rpm_per_key and rpm_per_key > 0 else 0.0
        self.next_ok = {k: 0.0 for k in self.keys}
        self.lock = threading.Lock()
        self.ctr = itertools.count()

    def get_key(self):
        if self.interval <= 0:                            # no pacing: atomic round-robin
            return self.keys[next(self.ctr) % len(self.keys)]
        while True:                                       # pick the soonest-free key; wait if none ready
            with self.lock:
                k = min(self.keys, key=lambda x: self.next_ok[x])
                now = time.monotonic()                    # monotonic: NTP steps can't push past the cap
                if self.next_ok[k] <= now:
                    self.next_ok[k] = now + self.interval
                    return k
                wait = self.next_ok[k] - now
            time.sleep(min(wait, 0.25))                   # release the lock while we wait


def load_keys(host):
    env_key = HOSTS[host][1]
    val = os.environ.get(env_key)
    if not val:
        with open(ROOT / ".env", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(env_key + "="):
                    val = line.split("=", 1)[1]
                    break
    if not val:
        raise SystemExit(f"no {env_key} in env or {ROOT/'.env'}")
    return val.split(",")


def call(url, key, model, prompt, temperature, timeout):
    body = json.dumps({
        "model": model, "temperature": temperature,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "serblo/1.0"})                      # non-default UA (Cloudflare 403s urllib)
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = (d.get("choices") or [{}])[0].get("message", {})
    text = msg.get("content") or ""                      # ignore reasoning_content (thinking)
    return text, time.time() - t


def generate(url, limiter, model, prompt, temperature, timeout):
    """Key-rotating call with backoff. Returns (text, secs, calls, ok, rate_limited).
    limiter hands out rate-paced keys so parallel workers spread across keys without over-driving.
    On exhaustion returns the LAST error text (not "") so errors.jsonl isn't blank."""
    wall, rl, last_err = 0.0, False, "empty response"
    for attempt in range(len(BACKOFF) + 1):
        key = limiter.get_key()                           # rate-paced (or round-robin if unpaced)
        try:
            text, secs = call(url, key, model, prompt, temperature, timeout)
            wall += secs
            if text.strip():
                return text, wall, attempt + 1, True, rl
            last_err = "empty response"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                rl = True
                last_err = "HTTP 429"
            elif e.code not in (500, 502, 503):           # 4xx auth/bad-request won't fix on retry
                return f"HTTP {e.code}: {e.read()[:200]!r}", wall, attempt + 1, False, rl
            else:
                last_err = f"HTTP {e.code}"
        except TimeoutError:                              # slow endpoint = TRANSIENT (like a 429): back
            rl = True                                     # off + retry, DON'T trip the systemic fail-streak
            last_err = "read timeout"                     # (a temporarily-slow model must trickle, not die)
        except Exception as e:
            if isinstance(getattr(e, "reason", None), TimeoutError):   # connect-timeout wrapped in URLError
                rl = True
                last_err = "connect timeout"
            else:
                last_err = f"{type(e).__name__}: {e}"[:200]   # capture the real reason, not silence
        if attempt < len(BACKOFF):
            time.sleep(BACKOFF[attempt])
    return last_err, wall, len(BACKOFF) + 1, False, rl


def doc_records(rid, register, axes, raw, source, model, lenient=True):
    """One record per payload. lenient (default for raw open-weight): when the model
    ignored the §7.9 wrapper entirely, take the whole output as the doc; the junk gate
    in pick_payload_doc (looks_serbian + length) still filters garbage, so this recovers
    good-but-unwrapped text (Kimi drops the wrapper ~50%) without admitting English/noise."""
    docs, issues = extract_docs(raw, salvage_unclosed=True)
    if not docs and lenient and "no_sentinels" in issues:
        docs = [raw.strip()]
        issues = issues + ["lenient_no_sentinel"]
    doc, pol = pick_payload_doc(docs)
    issues = issues + pol
    if doc is None:
        return [], issues
    return [{"id": f"{rid}-d0", "text": doc, "source": source, "register": register,
             "axes": axes, "model": model, "words": len(doc.split()),
             "issues": issues if issues else None}], issues


def load_done(out_dir):
    done, path = set(), out_dir / "raw.jsonl"
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
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


class Stats:
    """Shared mutable counters + append-only file handles. record() is called ONLY from the
    main thread (serial loop or the concurrent wait-loop), so writes never race."""
    def __init__(self, out_dir, host, model, source):
        self.host, self.model, self.source = host, model, source
        self.attempted = self.ok = self.failed = self.empty = 0
        self.n_docs = self.n_words = self.requests = self.rate_limited = 0
        self.fail_streak = self.empty_streak = 0
        self.stop = None
        self.t0 = self.t_report = time.time()
        self.raw_f = open(out_dir / "raw.jsonl", "a", encoding="utf-8")
        self.docs_f = open(out_dir / "docs.jsonl", "a", encoding="utf-8")
        self.err_f = open(out_dir / "errors.jsonl", "a", encoding="utf-8")

    def close(self):
        self.raw_f.close(); self.docs_f.close(); self.err_f.close()

    def record(self, result, max_requests):
        """Write one completed request; update counters; set self.stop if a limit is hit.
        `result` is do_one's tuple (rec, text, secs, calls, good, rate_limited)."""
        rec, text, secs, calls, good, rl = result
        self.attempted += 1
        self.requests += calls
        if self.requests >= max_requests and not self.stop:   # budget on EVERY path: a persistently
            self.stop = f"budget: {self.requests}/{max_requests} calls spent"   # rate-limited run must
        if rl:                                                # still honour --max-requests, not churn
            self.rate_limited += 1                            # the whole manifest
        if not good:
            self.failed += 1
            self.err_f.write(json.dumps({"id": rec.get("id", "?"), "calls": calls, "rate_limited": rl,
                                         "detail": text[:200], "ts": int(time.time())},
                                        ensure_ascii=False) + "\n")
            self.err_f.flush()
            if rl:                                        # 429-exhaustion = throttle, not a systemic
                return                                    # fault: back off (already slept) and keep going
            self.fail_streak += 1
            if self.fail_streak >= MAX_FAIL_STREAK:
                self.stop = f"{MAX_FAIL_STREAK} consecutive non-rate-limit failures — systemic, stopping"
            return
        self.fail_streak = 0
        rid, reg, axes = rec.get("id", "?"), rec.get("register", ""), rec.get("axes", {})
        self.raw_f.write(json.dumps({"id": rid, "host": self.host, "model": self.model,
                                     "register": reg, "axes": axes,
                                     "raw": text, "secs": round(secs, 1),
                                     "ts": int(time.time())}, ensure_ascii=False) + "\n")
        self.raw_f.flush()
        self.ok += 1
        recs, _ = doc_records(rid, reg, axes, text, self.source, self.model)
        for d in recs:
            self.n_docs += 1
            self.n_words += d["words"]
            self.docs_f.write(json.dumps(d, ensure_ascii=False) + "\n")
        self.docs_f.flush()
        if recs:
            self.empty_streak = 0
        else:
            self.empty += 1
            self.empty_streak += 1
            if self.empty_streak >= MAX_EMPTY_STREAK:
                self.stop = f"{MAX_EMPTY_STREAK} consecutive empty conversions — format broke, stopping"

    def maybe_report(self, inflight=0):
        now = time.time()
        if now - self.t_report < 30:
            return
        self.t_report = now
        rate = self.n_docs / max((now - self.t0) / 3600, 1e-9)
        print(f"  [{self.n_docs} docs, {self.requests} req, {rate:.0f} docs/hr, "
              f"{self.rate_limited} rate-limited, {inflight} in-flight, {self.n_words:,} words]", flush=True)


def run_concurrent(manifest, do_one, st, concurrency, max_requests):
    """Keep `concurrency` requests in flight, record each as it lands, refill, stop on st.stop.
    On stop OR manifest-exhaustion, DRAIN the still-in-flight futures (record them) rather than
    discarding: the ThreadPoolExecutor blocks on them at exit anyway, so those generations are
    already paid for; dropping them threw away up to K docs per capped run."""
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = set()
        while len(futures) < concurrency:                 # prime the pool
            rec = next(manifest, None)
            if rec is None:
                break
            futures.add(ex.submit(do_one, rec))
        while futures:
            ready, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in ready:
                st.record(fut.result(), max_requests)
            st.maybe_report(len(futures))
            if st.stop:
                break
            while len(futures) < concurrency:             # refill to keep K in flight
                rec = next(manifest, None)
                if rec is None:
                    break
                futures.add(ex.submit(do_one, rec))
        for fut in futures:                               # drain the in-flight (already running)
            try:
                st.record(fut.result(), max_requests)     # record, don't discard (shutdown waits anyway)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", choices=sorted(HOSTS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--register", choices=["conversational", "essay", "both"], default="conversational")
    ap.add_argument("--manifest-dir", default=str(ROOT / "data" / "manifests"))
    ap.add_argument("--out", required=True, help="OWN dir per (host,model) — ledgers must not mix")
    ap.add_argument("--max-requests", type=int, default=200, help="budget = real API calls incl. retries")
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--timeout", type=int, default=120, help="raise for slow reasoning models (DeepSeek)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel in-flight requests (thread pool); 1 = serial. With --rpm-per-key "
                         "set it high (~ keys * rpm * avg_secs / 60) so the rate limiter, not the "
                         "pool, is what throttles")
    ap.add_argument("--rpm-per-key", type=int, default=0,
                    help="pace each key to this many requests/minute so we never exceed the provider "
                         "limit (0 = unpaced round-robin). OpenRouter free = 20")
    a = ap.parse_args()

    url, _ = HOSTS[a.host]
    keys = load_keys(a.host)
    limiter = RateLimiter(keys, a.rpm_per_key)            # paces keys to --rpm-per-key (or round-robin)
    registers = ["conversational", "essay"] if a.register == "both" else [a.register]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(out_dir)
    print(f"{a.host}/{a.model}: resume {len(done):,} spent | {len(keys)} keys | "
          f"concurrency {a.concurrency}", flush=True)

    st = Stats(out_dir, a.host, a.model, a.host)
    manifest = (r for r in iter_manifest(Path(a.manifest_dir), registers) if r["id"] not in done)

    def do_one(rec):
        try:
            text, secs, calls, good, rl = generate(url, limiter, a.model, rec["prompt"],
                                                    a.temperature, a.timeout)
            return rec, text, secs, calls, good, rl
        except Exception as e:                            # malformed record etc.: log as a failure,
            return rec, f"worker error: {e}", 0.0, 1, False, False   # never crash the whole loop

    try:
        if a.concurrency <= 1:
            for rec in manifest:
                if st.stop:
                    break
                st.record(do_one(rec), a.max_requests)
                st.maybe_report()
        else:
            run_concurrent(manifest, do_one, st, a.concurrency, a.max_requests)
    except KeyboardInterrupt:
        st.stop = "interrupted — everything before this item is saved"
    finally:
        st.close()
    print(f"\n=== {a.host}/{a.model} ===")
    print(f"attempted {st.attempted} | ok {st.ok} | failed {st.failed} | empty {st.empty} | "
          f"req {st.requests} | rate-limited {st.rate_limited} | docs {st.n_docs} | "
          f"words {st.n_words:,} | {(time.time()-st.t0)/60:.1f} min")
    if st.stop:
        print(f"stopped: {st.stop}", flush=True)


if __name__ == "__main__":
    main()
