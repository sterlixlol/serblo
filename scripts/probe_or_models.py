#!/usr/bin/env python3
"""One-turn probe of every FREE OpenRouter model: "is OR even worth using?"

Fetches the LIVE :free model list (money guard: only models whose prompt+completion+request
pricing is $0 are ever called; paid models are never touched), then sends exactly ONE chat
turn to each. Models are assigned round-robin to the OPENROUTER_KEYS pool (model i -> key
i % N) and each key is driven by its OWN thread, strictly sequentially, so load is perfectly
even and no key is ever hit concurrently. Every result (status, latency, Serbian snippet,
error) is appended + fsynced after each probe, so it's resumable: a rerun skips done models.

Self-contained (stdlib only) so it can't fail on a missing repo module on the VM.

  python3 scripts/probe_or_models.py --dry-run   # show keys + free-model list + assignment, send nothing
  python3 scripts/probe_or_models.py             # actually probe
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

OR = "https://openrouter.ai/api/v1"
ENV_PATH = Path.home() / "serblo" / ".env"
SYSTEM = ("Ti si generator teksta. Pišeš isključivo prirodan, neformalan srpski. "
          "Bez uvoda, bez prevoda, bez engleskog, bez meta-komentara — samo traženi tekst.")
USER = ("Napiši kratak, opušten komentar kakav bi neko ostavio na društvenoj mreži ili forumu "
        "o tome kako mu je prošao dan. Dve-tri rečenice, svakodnevni govor.")


def load_keys():
    val = os.environ.get("OPENROUTER_KEYS")
    if not val and ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_KEYS="):
                val = line.split("=", 1)[1]
                break
    if not val:
        raise SystemExit(f"no OPENROUTER_KEYS in env or {ENV_PATH}")
    return [k.strip() for k in val.split(",") if k.strip()]


def http(method, path, key, body=None, timeout=90):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(OR + path, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "serblo/1.0", "HTTP-Referer": "https://serblo.local",
        "X-Title": "serblo-probe"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _zero(x):
    """A pricing field counts as free only if it parses to exactly 0; anything unknown -> paid."""
    try:
        return float(x if x not in (None, "") else 0) == 0.0
    except (TypeError, ValueError):
        return False


def free_models(keys):
    last = None
    for k in keys:
        try:
            d = http("GET", "/models", k)
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        out = []
        for m in d.get("data", []):
            p = m.get("pricing", {}) or {}
            if _zero(p.get("prompt")) and _zero(p.get("completion")) and _zero(p.get("request")):
                out.append(m["id"])
        return sorted(set(out))
    raise SystemExit(f"/models failed on all keys: {last}")


def probe(model, key, timeout):
    body = {"model": model, "temperature": 0.9,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER}]}
    t = time.time()
    try:
        d = http("POST", "/chat/completions", key, body, timeout)
        secs = round(time.time() - t, 1)
        msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
        text = (msg.get("content") or "").strip()
        if text:
            return {"model": model, "ok": True, "status": 200, "secs": secs, "chars": len(text),
                    "text": text[:800], "usage": d.get("usage", {}), "used_model": d.get("model", "")}
        return {"model": model, "ok": False, "status": 200, "secs": secs,
                "error": "empty content", "raw": json.dumps(d)[:300]}
    except urllib.error.HTTPError as e:
        secs = round(time.time() - t, 1)
        try:
            bod = e.read()[:400].decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            bod = ""
        return {"model": model, "ok": False, "status": e.code, "secs": secs,
                "error": f"HTTP {e.code}", "body": bod}
    except Exception as e:  # noqa: BLE001
        secs = round(time.time() - t, 1)
        return {"model": model, "ok": False, "status": None, "secs": secs,
                "error": f"{type(e).__name__}: {e}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "serblo" / "or_probe_results.jsonl"))
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    keys = load_keys()
    models = free_models(keys)
    outp = Path(a.out)

    done = set()
    if outp.exists():
        for line in outp.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["model"])
            except Exception:  # noqa: BLE001
                pass
    todo = [m for m in models if m not in done]

    idx = {m: i for i, m in enumerate(models)}          # stable round-robin index over the FULL list
    buckets = {ki: [] for ki in range(len(keys))}
    for m in todo:
        buckets[idx[m] % len(keys)].append(m)

    print(f"keys: {len(keys)} | free models: {len(models)} | done: {len(done)} | to probe: {len(todo)}")
    for ki in range(len(keys)):
        print(f"  key[{ki}] <- {len(buckets[ki])}: {buckets[ki]}")
    if a.dry_run:
        print("DRY RUN — nothing sent.")
        return

    lock = threading.Lock()
    fh = open(outp, "a", encoding="utf-8")
    t0 = time.time()

    def worker(ki):
        for m in buckets[ki]:
            res = probe(m, keys[ki], a.timeout)
            res["key"] = ki
            with lock:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                tag = "OK" if res["ok"] else "XX"
                extra = "" if res["ok"] else f"  {res.get('error', '')}"
                print(f"{tag} [k{ki}] {str(res.get('status')):>4} {res.get('secs')!s:>5}s  {m}{extra}",
                      flush=True)

    threads = [threading.Thread(target=worker, args=(ki,)) for ki in range(len(keys)) if buckets[ki]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fh.close()
    okn = sum(1 for line in outp.read_text(encoding="utf-8").splitlines()
              if line.strip() and json.loads(line).get("ok"))
    total = len(outp.read_text(encoding="utf-8").splitlines())
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min | {okn}/{total} models returned text | -> {outp}")


if __name__ == "__main__":
    main()
