#!/usr/bin/env python3
"""Gentle health ping of the NIM key pool: ONE request per key, to confirm each is cooled off
(HTTP 200) rather than still rate-limited (429). One call per key is the opposite of hammering.

NIM lists models in /v1/models that are NOT necessarily deployed (a call 404s "Function not found"),
so we first DISCOVER a model that actually serves: try known-good staples (then any listed instruct
model), skipping 404s, on one key, and only then ping the whole pool with it, so a 429 genuinely
means rate-limited. Reusable as the pool grows: it pings whatever is in NIM_KEYS.

  python3 scripts/ping_nim_keys.py
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

NIM = "https://integrate.api.nvidia.com/v1"
ENV = Path.home() / "serblo" / ".env"
KNOWN_GOOD = [                                   # NIM catalog staples, small first (fast, uncapped)
    "meta/llama-3.2-3b-instruct",
    "meta/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "microsoft/phi-3-mini-4k-instruct",
    "google/gemma-2-9b-it",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-70b-instruct",
]


def load_keys():
    val = os.environ.get("NIM_KEYS")
    if not val and ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("NIM_KEYS="):
                val = line.split("=", 1)[1]
                break
    if not val:
        raise SystemExit(f"no NIM_KEYS in env or {ENV}")
    return [k.strip() for k in val.split(",") if k.strip()]


def http(method, path, key, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(NIM + path, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.load(r)


def ping(key, model, timeout=60):
    body = {"model": model, "temperature": 0.2,
            "messages": [{"role": "user", "content": "Reci 'zdravo'."}]}
    t = time.time()
    try:
        st, d = http("POST", "/chat/completions", key, body, timeout)
        secs = round(time.time() - t, 1)
        txt = ((d.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "") or ""
        return {"status": st, "secs": secs, "text": txt.strip()[:60]}
    except urllib.error.HTTPError as e:
        secs = round(time.time() - t, 1)
        try:
            bod = e.read()[:160].decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            bod = ""
        return {"status": e.code, "secs": secs, "error": bod}
    except Exception as e:  # noqa: BLE001
        return {"status": None, "secs": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}"[:150]}


def discover_servable(key):
    """Find a model that is actually deployed (any non-404 response): try staples, then listed
    instruct models. A 429 here still means 'served' (and that this key is hot)."""
    listed = []
    try:
        _, d = http("GET", "/models", key)
        listed = [m["id"] for m in d.get("data", [])]
    except Exception:  # noqa: BLE001
        pass
    in_catalog = set(listed)
    extra = [m for m in listed
             if any(t in m.lower() for t in ("instruct", "-it", "chat"))
             and not any(t in m.lower() for t in ("reasoning", "embed", "rerank", "vl", "vision"))]
    candidates = [m for m in KNOWN_GOOD if m in in_catalog] + [m for m in extra if m not in KNOWN_GOOD]
    for m in candidates[:10]:
        r = ping(key, m)
        if r["status"] != 404:                  # deployed (200 / 429 / 5xx): use it
            return m, len(listed), r
    return None, len(listed), None


def main():
    keys = load_keys()
    print(f"NIM keys: {len(keys)}")
    model, n_listed, disc = discover_servable(keys[0])
    if not model:
        raise SystemExit(f"no servable model found (tried staples + instruct models; {n_listed} listed)")
    print(f"probe model: {model}  ({n_listed} models listed; key[0] discovery -> {disc['status']})\n")

    results = {}

    def work(i):
        results[i] = ping(keys[i], model)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(keys))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    cool = hot = 0
    for i in range(len(keys)):
        r = results[i]
        if r["status"] == 200:
            tag, cool = "COOL ", cool + 1
        elif r["status"] == 429:
            tag, hot = "HOT-429", hot + 1
        else:
            tag = "ERR  "
        note = r.get("text") if r["status"] == 200 else r.get("error", "")
        print(f"  key[{i}]  {tag}  {str(r['status']):>4}  {r['secs']!s:>5}s  {(note or '')[:60]}")
    verdict = " — pool fully cooled, ready" if hot == 0 and cool == len(keys) else ""
    print(f"\n{cool}/{len(keys)} cooled (200)" + (f", {hot} still 429" if hot else "") + verdict)


if __name__ == "__main__":
    main()
