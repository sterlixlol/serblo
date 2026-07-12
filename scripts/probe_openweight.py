#!/usr/bin/env python3
"""Serbian-quality probe across free open-weight hosts (Groq + NIM): pick who to wire.

PRD flags free-tier open-weight sources as risky ("weak Serbian"). Before building a
channel for any of them, this hits one strong model per DISTINCT LINEAGE with real
manifest prompts and scores the output: Serbian-ness (looks_serbian), diacritic ratio
(native BCS carries č/ć/š/ž/đ; translationese drops them), length, and a snippet to read.
Distinct lineages are the diversity prize, not distinct hosts. Cheap: ~2 calls/model.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from serblo.generate.convert import extract_docs, looks_serbian  # noqa: E402
from serblo.pipeline.normalize import diacritic_ratio            # noqa: E402

HOSTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "nim": "https://integrate.api.nvidia.com/v1/chat/completions",
}
# (host, model_id, lineage)
MODELS = [
    ("groq", "llama-3.3-70b-versatile", "Meta-Llama"),
    ("groq", "qwen/qwen3.6-27b", "Alibaba-Qwen"),
    ("groq", "openai/gpt-oss-120b", "OpenAI-oss"),
    ("nim", "nvidia/llama-3.3-nemotron-super-49b-v1.5", "NVIDIA-Nemotron"),
    ("nim", "mistralai/ministral-14b-instruct-2512", "Mistral"),
    ("nim", "z-ai/glm-5.2", "Zhipu-GLM"),
    ("nim", "deepseek-ai/deepseek-v4-pro", "DeepSeek"),
]


def load_env():
    env = {}
    for line in open(ROOT / ".env"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v
    return env


def call(url, key, model, prompt, timeout=120):
    body = json.dumps({"model": model, "temperature": 0.9,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "serblo/1.0"})   # Groq is behind Cloudflare: default urllib UA -> 403
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], time.time() - t


def text_of(raw):
    """The §7.9 doc if the model wrapped it, else the raw stripped output."""
    docs, _ = extract_docs(raw, salvage_unclosed=True)
    return docs[0] if docs else raw.strip()


def main():
    env = load_env()
    keys = {"groq": env["GROQ_KEYS"].split(",")[0], "nim": env["NIM_KEYS"].split(",")[0]}
    prompts = {}
    for reg in ("conversational", "essay"):
        with open(ROOT / "data" / "manifests" / f"{reg}.jsonl") as f:
            prompts[reg] = json.loads(f.readline())["prompt"]

    save = Path("results/openweight_probe.json")
    print(f"{'lineage':17}{'model':42}{'reg':5}{'secs':>6}{'words':>7}{'SR?':>5}{'diac%':>7}")
    results = []
    for host, model, lineage in MODELS:
        for reg in ("conversational", "essay"):
            try:
                raw, secs = call(HOSTS[host], keys[host], model, prompts[reg])
                txt = text_of(raw)
                w = len(txt.split())
                sr = looks_serbian(txt)
                diac = diacritic_ratio(txt) * 100
                results.append({"lineage": lineage, "model": model, "reg": reg,
                                "words": w, "sr": sr, "diac": diac, "text": txt})
                print(f"{lineage:17}{model[:40]:42}{reg[:4]:5}{secs:>6.0f}{w:>7}{'Y' if sr else 'N':>5}{diac:>7.2f}")
            except Exception as e:
                print(f"{lineage:17}{model[:40]:42}{reg[:4]:5}  ERROR {str(e)[:40]}")
    save.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nfull texts -> {save}")


if __name__ == "__main__":
    main()
