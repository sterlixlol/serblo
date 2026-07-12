#!/usr/bin/env python3
"""Assemble the final Serblo SFT dataset from every data/sft/**/pairs.jsonl lane.

Two OpenAI-messages JSONL outputs (nothing generated is thrown away):
  serblo-sft-full.jsonl   RELEASED set: every clean, deduped pair (~95k). The HF artifact.
  serblo-sft-train.jsonl  TRAINING set: full minus over-long answers, with the dominant
                          lineage (DeepSeek, ~37%) capped for diversity (the multi-lineage /
                          anti-swamp lesson). The capped-out DeepSeek pairs still live in FULL.

Cleanup (BOTH outputs): strip em/en-dashes (the AI tell; SFT_SYSTEM banned them),
normalize Cyrillic->Latin (gajica, identical to the pretrain corpus), strip stray <<<...>>>
sentinels + <think> blocks, drop refusals / English-bleed / empties, exact (Q,A) dedup.

Identity system-variation (§30h): a minimal "Ti si Serblo..." system line on most pairs,
bare on a slice (so the model works system-less too). Identity-task pairs are mostly BARE so
the model learns to declare its own name unprompted (identity is baked into those answers).

  PYTHONPATH=src python3 scripts/assemble_sft.py \
      --glob 'data/sft/**/pairs.jsonl' --out data/sft/assembled \
      --deepseek-cap 0.20 --sys-rate 0.85 --identity-sys-rate 0.40 --seed 7
"""
import argparse
import collections
import glob
import json
import pathlib
import random
import re
import sys

import xxhash

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serblo.pipeline.normalize import normalize_text  # noqa: E402  (NFC + Cyrillic->Latin)

# --- cleanup ---
_DASH = str.maketrans({"—": "-", "–": "-", "―": "-", "‒": "-", "−": "-"})
_SENT = re.compile(r"<{2,}\s*/?\s*(?:Q|A|END|DOC)\b\s*>{0,3}|<{2,}|>{2,}", re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_MD = re.compile(r"\*\*|__|^#{1,6}\s+", re.M)          # markdown emphasis / heading markers
_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")

REFUSAL = re.compile(r"\b(as an ai|kao (veštačka|vjestacka) inteligencij|ne mogu da (ti )?pomognem|"
                     r"žao mi je,? ali|izvinjavam se,? ali|i'?m sorry|i cannot|i can'?t help)\b", re.I)
# after normalize_text, modern Serbian Cyrillic is already Latin; anything left is archaic
# (yat ѣ, omega ѡ, hard-sign ъ) or foreign Cyrillic (Russian я/ю/щ) = off-distribution junk.
_CYR = re.compile(r"[Ѐ-ӿ]")
SR_DIA = set("čćšžđČĆŠŽĐ")
SR_WORDS = {"i", "je", "da", "na", "se", "u", "za", "ne", "su", "od", "to", "što", "li", "sam",
            "ali", "ili", "kao", "sa", "po", "me", "te", "mi", "ga", "smo", "ste", "će", "bih"}
EN_COMMON = {"the", "and", "you", "for", "with", "this", "that", "your", "have", "are", "is", "to", "of"}

SYS_POOL = [
    "Ti si Serblo, srpski jezički asistent.",
    "Ti si Serblo, koristan asistent koji govori srpski.",
    "Ti si Serblo, domaći srpski jezički model.",
    "Ti si Serblo.",
]


def looks_english(text):
    toks = re.findall(r"[a-zA-Z]+", text.lower())
    if len(toks) < 4:
        return False
    en = sum(t in EN_COMMON for t in toks)
    sr = sum(t in SR_WORDS for t in toks)
    return en >= 2 and en > sr and not (set(text) & SR_DIA)


def clean_text(s):
    if not s:
        return ""
    s = _THINK.sub("", s)
    s = s.translate(_DASH)
    s = _SENT.sub("", s)
    s = _MD.sub("", s)             # strip markdown ** __ and # headers
    s = normalize_text(s)          # NFC + Cyrillic->Latin (no-op on clean Latin)
    s = _WS.sub(" ", s)
    s = _NL.sub("\n\n", s)
    return "\n".join(ln.strip() for ln in s.splitlines()).strip()


def lineage_of(p):
    blob = ((p.get("model") or "") + " " + (p.get("source") or "")).lower()
    for needle, lin in (("deepseek", "deepseek"), ("vertex", "gemini"), ("gemini", "gemini"),
                        ("hy3", "hy3"), ("tencent", "hy3"), ("mimo", "mimo"), ("opus", "opus"),
                        ("claude", "opus"), ("kimi", "kimi"), ("moonshot", "kimi"), ("llama", "llama")):
        if needle in blob:
            return lin
    return (p.get("model") or p.get("source") or "unknown").lower()


GROUNDED = {"summarize", "rewrite", "extract"}   # instruction must embed a real passage
_MANGLED = re.compile(r"<<+\s*([0-9A-Za-zČĆŠŽĐ]{1,6})")   # <<<Aalet>>>, <<<E>>> = broken sentinel


def has_mangled_sentinel(text):
    return any(m.group(1).upper() not in ("Q", "A", "END", "DOC") for m in _MANGLED.finditer(text))


# identity poison OUTSIDE the identity task: an answer claiming to be another model/company, or
# (in chitchat, where "ko si ti" lives) claiming a specific non-Serblo name. Scoped so it can't
# fire on a story character in a write/qa task.
_MK = r"(google|gugl\w*|open\s?ai|anthropic|meta|microsoft|majkrosoft)"
_MAKE = r"(obučen|obučena|kreiran|kreirana|napravljen|napravljena|razvijen|razvijena|treniran|trenirana|stvoren|stvorena|programiran|dizajniran)"
# first-person-anchored so it fires on "Ja sam ... model ... obučen od strane Google" but NOT on a
# legit third-person fact like "Android je razvijen od strane Google-a".
_WRONG_MAKER = re.compile(
    r"\b(ja sam|zovem se|ime mi je)\s+(chatgpt|čet ?gpt|gpt|gemini|claude|klod|bard|deepseek|kimi|mimo|llama|copilot|kopilot|bing|grok|palm)\b"
    + r"|\bja sam\b[^.]{0,80}\b" + _MAKE + r"\w*\b[^.]{0,30}" + _MK
    + r"|(razvi(o|la|jen[ao]?)|napravi(o|la)|kreira(o|la)|trenira(o|la)|stvori(o|la)|obuči(o|la))\s+me\b[^.]{0,20}" + _MK
    + r"|\bja sam\b[^.]{0,40}(veštačk[ai]|vještačk[ai])\s+inteligencij\w*[^.]{0,30}" + _MK
    + r"|bukvalno\s+(chatgpt|čet ?gpt)", re.I)
_WRONG_NAME = re.compile(r"\b(zovem se|ime mi je|ja sam)\s+(?!Serblo\b)([A-ZČĆŠŽĐ][a-zčćšžđ]{2,})\b")


def is_identity_poison(task, resp):
    if task == "identity":
        return False
    if _WRONG_MAKER.search(resp):
        return True
    return task == "chitchat" and bool(_WRONG_NAME.search(resp))


def is_sr_to_en(instr):
    return bool(re.search(r"\bna\s+engleski\b|to\s+english", instr, re.I))


# conservative truncation: answer ends on a BCS function word with NO terminal punctuation (a
# clear mid-sentence cut). Casual no-period endings on content words are left alone.
_FUNC = {"i", "ili", "a", "ali", "pa", "te", "da", "je", "se", "u", "na", "za", "od", "do", "sa", "s",
         "o", "po", "iz", "kroz", "ka", "uz", "niz", "pod", "nad", "pred", "među", "bez", "oko", "kod",
         "li", "ne", "će", "bi", "su", "sam", "si", "smo", "ste", "jer", "ako", "kad", "dok", "nego",
         "već", "koji", "koja", "koje", "što", "kao"}
_TERM = set(".!?…\"')»:")


def is_truncated(ans):
    a = ans.rstrip()
    if not a or a[-1] in _TERM:
        return False
    toks = a.split()
    last = re.sub(r"[^0-9A-Za-zČĆŠŽĐčćšžđ]+$", "", toks[-1]).lower() if toks else ""
    return last in _FUNC


def to_record(p):
    msgs = []
    if p["_sys"]:
        msgs.append({"role": "system", "content": p["_sys"]})
    msgs.append({"role": "user", "content": p["instruction"]})
    msgs.append({"role": "assistant", "content": p["response"]})
    return {"messages": msgs, "id": p["_uid"], "task": p["task"],
            "register": p.get("register", ""), "source": p["_lin"]}


def write_jsonl(path, pairs):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as w:
        for p in pairs:
            w.write(json.dumps(to_record(p), ensure_ascii=False) + "\n")
    tmp.rename(path)


def report(name, pairs):
    lin = collections.Counter(p["_lin"] for p in pairs)
    task = collections.Counter(p["task"] for p in pairs)
    withsys = sum(1 for p in pairs if p["_sys"])
    n = len(pairs)
    print(f"\n=== {name}: {n:,} pairs ===")
    print("  lineage:  " + "  ".join(f"{k} {v} ({100*v//n}%)" for k, v in lin.most_common()))
    print("  task:     " + "  ".join(f"{k} {v}" for k, v in task.most_common()))
    print(f"  system:   {withsys} with a Serblo system line, {n - withsys} bare "
          f"({100*withsys//n}% / {100*(n-withsys)//n}%)")
    return {"pairs": n, "lineage": dict(lin), "task": dict(task), "with_system": withsys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/sft/**/pairs.jsonl")
    ap.add_argument("--out", default="data/sft/assembled")
    ap.add_argument("--deepseek-cap", type=float, default=0.20, help="max DeepSeek share of TRAIN")
    ap.add_argument("--sys-rate", type=float, default=0.85, help="share of non-identity pairs given a system line")
    ap.add_argument("--identity-sys-rate", type=float, default=0.40, help="share of identity pairs given a system line")
    ap.add_argument("--too-long", type=int, default=120, help="drop answers longer than this (words) from TRAIN")
    ap.add_argument("--sr-en", choices=["drop", "keep"], default="drop", help="sr->en translate pairs (English answers)")
    ap.add_argument("--grounded-min-words", type=int, default=40,
                    help="a grounded task (summarize/rewrite/extract) needs >= this many words in the "
                         "instruction (the embedded passage); below it the source text is missing -> drop")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(a.glob, recursive=True))
    print(f"loading {len(files)} pairs.jsonl files ...")
    raw = []
    n_badjson = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError:
                    n_badjson += 1                 # count, don't silently vanish a mangled lane
    print(f"  {len(raw):,} raw pairs" + (f"  ({n_badjson} malformed lines skipped)" if n_badjson else ""))

    seen = set()
    drops = collections.Counter()
    kept = []
    for p in raw:
        raw_i, raw_r = p.get("instruction", ""), p.get("response", "")
        if has_mangled_sentinel(raw_i) or has_mangled_sentinel(raw_r):
            drops["mangled_sentinel"] += 1
            continue
        instr = clean_text(raw_i)
        resp = clean_text(raw_r)
        task = p.get("task", "?")
        if not instr or not resp:
            drops["empty"] += 1
            continue
        lin = lineage_of(p)
        if lin == "llama":                              # rotten lane (no-op/Croatian/mistranslation)
            drops["llama_lane"] += 1
            continue
        if REFUSAL.search(resp) or REFUSAL.search(instr):
            drops["refusal"] += 1
            continue
        if _CYR.search(instr) or _CYR.search(resp):     # archaic/foreign Cyrillic survived translit
            drops["cyrillic"] += 1
            continue
        if task != "translate" and looks_english(resp):
            drops["en_bleed"] += 1
            continue
        if is_identity_poison(task, resp):              # claims another model/company/name
            drops["identity_poison"] += 1
            continue
        if task == "translate" and a.sr_en == "drop" and is_sr_to_en(instr):
            drops["sr_to_en"] += 1
            continue
        if task in GROUNDED and len(instr.split()) < a.grounded_min_words:   # source text missing
            drops["grounded_no_text"] += 1
            continue
        if is_truncated(resp):                          # clear mid-sentence cut (function-word end)
            drops["truncated"] += 1
            continue
        key = xxhash.xxh64((instr + "\x00" + resp).encode("utf-8")).intdigest()
        if key in seen:
            drops["dup"] += 1
            continue
        seen.add(key)
        p["instruction"], p["response"] = instr, resp
        p["_key"] = key
        p["_lin"] = lin
        p["_words"] = len(resp.split())
        kept.append(p)

    # unique id + deterministic (per-pair, order-independent) system-line assignment
    for i, p in enumerate(kept):
        p["_uid"] = f"serblo-sft-{i:06d}"
        prng = random.Random(a.seed ^ p["_key"])
        rate = a.identity_sys_rate if p["task"] == "identity" else a.sys_rate
        p["_sys"] = prng.choice(SYS_POOL) if prng.random() < rate else None

    print(f"\n  kept {len(kept):,}  |  dropped: " + ", ".join(f"{k} {v}" for k, v in drops.most_common()))

    # --- TRAIN: drop over-long, cap DeepSeek (identity task always protected) ---
    pool = [p for p in kept if p["_words"] <= a.too_long]
    protected = [p for p in pool if p["task"] == "identity"]
    ds = [p for p in pool if p["task"] != "identity" and p["_lin"] == "deepseek"]
    non_ds = [p for p in pool if p["task"] != "identity" and p["_lin"] != "deepseek"]
    cap = a.deepseek_cap
    d_keep = min(len(ds), int(cap * (len(non_ds) + len(protected)) / max(1e-9, 1 - cap)))
    rng.shuffle(ds)
    train = protected + non_ds + ds[:d_keep]
    rng.shuffle(train)

    write_jsonl(outdir / "serblo-sft-full.jsonl", kept)
    write_jsonl(outdir / "serblo-sft-train.jsonl", train)

    rep = {
        "params": vars(a),
        "raw": len(raw), "kept": len(kept), "dropped": dict(drops),
        "full": report("FULL (released)", kept),
        "train": report(f"TRAIN (DeepSeek capped {int(cap*100)}%, dropped {len(ds)-d_keep} over-cap + "
                        f"{sum(1 for p in kept if p['_words'] > a.too_long)} over-long)", train),
    }
    (outdir / "assembly-report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n-> {outdir}/serblo-sft-full.jsonl  ({len(kept):,})")
    print(f"-> {outdir}/serblo-sft-train.jsonl  ({len(train):,})")
    print(f"-> {outdir}/assembly-report.json")


if __name__ == "__main__":
    main()
