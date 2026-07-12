#!/usr/bin/env python3
"""Build a Vertex-batch manifest of SFT prompts, so run_vertex --register sft can generate
the instruction pairs on credit-covered Gemini batch (no rate limits).

Each manifest line = {id, prompt, register, axes} in run_vertex's expected shape. Vertex batch
takes ONE user turn, so SFT_SYSTEM is folded into the prompt. Ids are prefixed `vtx-` so they
never collide with the live lanes when everything is pooled. Reuses run_sft's builders verbatim
(register injection, ekavica, dash ban, grounded real-text) so the batch output is identical in
style to the live lanes; only the transport (async batch vs live HTTP) differs.

  python3 scripts/sft_vertex_manifest.py --target-pairs 52000
  then dry-run:  PYTHONPATH=src python3 -m serblo.generate.run_vertex --register sft \
                     --manifest-dir data/manifests-sft --model gemini-2.5-flash-lite --max-output-tokens 1200
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serblo.generate import run_sft as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--target-pairs", type=int, default=52000)
ap.add_argument("--per-call", type=int, default=4)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default=str(ROOT / "data" / "manifests-sft" / "sft.jsonl"))
a = ap.parse_args()

pool = R.build_realtext_pool(500, a.seed)
jobs = R.build_jobs(a.target_pairs, a.per_call, pool, a.seed)      # all 8 tasks, quota-weighted
outp = pathlib.Path(a.out)
outp.parent.mkdir(parents=True, exist_ok=True)

n = 0
with open(outp, "w", encoding="utf-8") as f:
    for j in jobs:
        prompt = R.SFT_SYSTEM + "\n\n" + R.build_prompt(j)        # fold system in (one user turn)
        f.write(json.dumps({
            "id": f"vtx-{j['id']}",
            "prompt": prompt,
            "register": j.get("register", "neutralni"),
            "axes": {"task": j["task"], "topic": j.get("topic")},
        }, ensure_ascii=False) + "\n")
        n += 1

print(f"wrote {n} SFT prompts -> {outp}   (~{n * a.per_call} pairs at {a.per_call}/prompt)")
