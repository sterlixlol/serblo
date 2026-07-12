#!/usr/bin/env python3
"""run_sft.py: generate BCS instruction (Q/A) pairs for SFT of the 125M base.

Each call asks a source for N varied ``<<<Q>>>/<<<A>>>/<<<END>>>`` pairs of ONE task type.
8 task types (TASK_PROMPTS); the grounded ones (summarize/rewrite/extract) inject a REAL
Serbian passage sampled from the deduped backbone + subtitles so the model learns on authentic
text. Reuses run_api's key/rate primitives; own call() so the SFT system prompt (not the prose
one) is used.

Incremental append-only save: every parsed pair is written the moment its
call returns; resume skips job ids already in raw.jsonl. Cost-capped for paid hosts via --max-cost.

  PYTHONPATH=src python3 -m serblo.generate.run_sft --host sambanova --model DeepSeek-V3.2 \
      --target-pairs 500 --out data/sft/pilot --concurrency 8
"""
import argparse
import glob
import gzip
import json
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from serblo.generate.run_api import BACKOFF, HOSTS, RateLimiter, load_keys  # shared primitives

SFT_SYSTEM = (
    "You generate supervised fine-tuning data for a small (125M) Bosnian/Croatian/Serbian "
    "instruction model. Write DIRECTLY in natural native BCS. Never translate from English, no "
    "translationese, no stiff constructions. Latin script (gajica), correct diacritics (c-caron etc: "
    "č ć š ž đ). Use EKAVICA (the Serbian standard): vreme not vrijeme, leto not ljeto, lepo not "
    "lijepo, mleko, dete, hteo, razumem, video, sneg; do NOT use ijekavica forms unless a Bosnian or "
    "Montenegrin register is explicitly requested. Keep ASSISTANT answers SHORT and plain, 1 to 4 "
    "sentences, except explicit 'write a "
    "text' tasks (8 sentences max). NEVER use em-dashes or en-dashes (the '—' and '-' characters): "
    "real Serbian text uses commas, periods, or parentheses, and a dash like that is a dead giveaway "
    "of AI-written text. The assistant is helpful and direct: it NEVER refuses a benign task, never "
    "adds disclaimers, never says 'as an AI / kao veštačka inteligencija'. Vary the phrasing, topic and "
    "structure of every pair, including how the USER asks. Output ONLY pairs, nothing else, each "
    "EXACTLY:\n<<<Q>>>\n{instruction}\n<<<A>>>\n{response}\n<<<END>>>"
)

# {n} = pairs per call, {register}/{topic} rotated per call, {real_text} injected for grounded tasks
TASK_PROMPTS = {
    "write": "Produce {n} pairs where the user asks the assistant to WRITE something short and it does. "
             "Vary the artifact: esej, email, poruka, čestitka, pesma, kratka priča, opis, oglas, recept, "
             "uputstvo, objava za društvene mreže, motivaciono pismo. Topic area: {topic}. Register: "
             "{register}. Each written piece ~3-8 sentences.",
    "qa": "Produce {n} pairs: the user asks a simple question, the assistant answers in 1-3 sentences. "
          "Bias to common-knowledge / definitional / everyday questions about {topic}. NOT obscure facts "
          "(you would hallucinate them); if a question needs deep expertise keep the answer general and "
          "honest, never invent specifics.",
    "translate": "Produce {n} SHORT everyday translation pairs between Serbian and English, both directions. "
                 "User: 'Prevedi na engleski: <srpska rečenica>' or 'Prevedi na srpski: <english sentence>', "
                 "assistant returns ONLY the translation. IMPORTANT: the source sentence and the target "
                 "language must DIFFER (never Serbian to Serbian). Topic area: {topic}. No technical edge cases.",
    "format": "Produce {n} pairs that teach OBEYING a format constraint: 'odgovori jednom rečju', 'nabroji "
              "tačno tri...', 'samo da ili ne', 'u tri crtice, bez uvoda'. The assistant follows it exactly. "
              "Underlying topic: {topic}.",
    "chitchat": "Produce {n} short conversational pairs: pozdravi, 'ko si ti', 'šta možeš', 'hvala', 'kako "
                "si', sitni razgovor. The assistant answers warmly, briefly, in character as a helpful "
                "Serbian assistant. Register/persona: {register}. Natural spoken Serbian, not stiff.",
    "summarize": "Below is a real Serbian text. Produce {n} pairs where the user asks to summarize it (vary "
                 "the phrasing: 'Sažmi', 'O čemu je ovo', 'Ukratko', 'Daj suštinu') and the assistant gives a "
                 "1-3 sentence summary.\n\nTEXT:\n{real_text}",
    "rewrite": "Below is a real Serbian text. Produce {n} pairs where the user asks to REWRITE it, i.e. vary the "
               "transform: formalnije, ležernije, kraće, jednostavnije, ispraviti gramatiku, u prvom licu, kao "
               "poruku prijatelju. The assistant returns ONLY the rewritten text.\n\nTEXT:\n{real_text}",
    "extract": "Below is a real Serbian text. Produce {n} SIMPLE extraction/classification pairs on it: izvuci "
               "imena/mesta/datume, koja je tema, ton (pozitivan/negativan/neutralan), ključne reči, koliko "
               "rečenica. The assistant gives the short direct answer.\n\nTEXT:\n{real_text}",
    "identity": "Produce {n} varied pairs that teach the assistant its IDENTITY. The user asks who/what the "
                "assistant is (vary widely: 'Ko si ti?', 'Koji si ti model?', 'Kako se zoveš?', 'Šta si ti?', "
                "'Jesi li ti ChatGPT?', 'Da li si Gemini?', 'Je l' ti Klod?', 'Ko te je napravio?', 'Na kom "
                "jeziku radiš najbolje?', 'Odakle si?'). The assistant ALWAYS identifies as Serblo, an open "
                "srpski (BCS) jezički model, and NEVER claims to be ChatGPT, Gemini, Claude, DeepSeek, Kimi, "
                "MiMo, or any other model or company. Vary the answer wording (Ja sam Serblo. / Zovem se Serblo, "
                "jezički model za srpski. / Nisam, ja sam Serblo. / Ja sam Serblo, pravljen kao otvoreni srpski "
                "model.). Warm, natural, ekavica, 1-2 sentences.",
}
GROUNDED = {"summarize", "rewrite", "extract"}
QUOTA = {"write": 20000, "qa": 17000, "summarize": 11000, "rewrite": 10000,
         "translate": 8000, "extract": 7000, "format": 6000, "chitchat": 6000, "identity": 300}

# serbian-voice register injection: per-register handle blocks (ekavica default; diacritics
# register-conditional: ošišana only where authentic). Injected into write/chitchat prompts.
REGISTER_BLOCKS = {
    "neutralni": "everyday neutral tone, ekavica, full diacritics, address as 'ti', no swearing, no emoji.",
    "formalni": "strictly formal, ekavica, FULL diacritics, address as 'Vi', nominalized register, no swearing, "
                "no anglicisms, period at sentence end.",
    "poslovni": "professional/business tone, ekavica, full diacritics, 'Vi', polite and concrete, minimal "
                "anglicisms (business terms ok), no emoji.",
    "knjizevni": "literary/poetic, ekavica, full diacritics, rich imagery, longer flowing sentences, no anglicisms.",
    "prijateljski": "relaxed, like to a friend, ekavica, diacritics mostly kept, 'ti', colloquial, a mild swear as "
                    "filler is ok, an occasional anglicism.",
    "genz": "Gen Z slang, ekavica but OSISANA latinica (drop the diacritics), 'ti', heavy slang + anglicisms, "
            "desemanticized swearing, short fragments, NO final period, a 💀 at the very end only if it fits.",
    "gejmerski": "gaming/Discord voice, osisana latinica, 'ti', gamer jargon + English terms, desemanticized "
                 "swearing, short and fast.",
    "navijacki": "football-terrace fan voice, ekavica, 'ti', passionate, exclamations, an occasional ALL-CAPS word, "
                 "no anglicisms, can be rough but pure organic Serbian.",
    "zenski": "warm women's-forum tone, ekavica, diacritics kept, diminutives, NEVER vulgar even when frustrated, "
              "an emoji at the end is ok.",
    "kafanski": "tavern-toast voice, like an older Serb over rakija, ekavica, 'ti', warm and toasting, pure Serbian "
                "with NO anglicisms, a mild oath is ok.",
    "crkveni": "Orthodox/church register, ekavica, full diacritics, 'Vi', calm and lightly archaic, no swearing, "
               "no anglicisms, no emoji.",
    "sms": "SMS/chat abbreviations, osisana latinica, 'ti', short forms (npr, nzm, vrv, kd), very short, no final period.",
}
# sampling weights: leaned toward authentic-casual registers (no human slice now; the models
# carry the casual texture that hand-written pairs would have) while keeping formal + niche present
REGISTER_POOL = (["neutralni"] * 4 + ["prijateljski"] * 6 + ["formalni"] * 2 + ["poslovni"] * 2
                 + ["genz"] * 3 + ["zenski"] * 3 + ["knjizevni"] + ["gejmerski"] * 2 + ["navijacki"] * 2
                 + ["kafanski"] * 2 + ["crkveni"] + ["sms"] * 2)
TOPICS = ["svakodnevni život", "hrana i kuvanje", "tehnologija i telefoni", "sport i fudbal", "muzika",
          "škola i fakultet", "posao", "putovanja", "porodica i prijatelji", "zdravlje",
          "priroda i životinje", "filmovi i serije", "istorija i kultura", "novac i kupovina"]

# per-1M (input, output) USD for paid hosts; free hosts absent -> cost 0
PRICES = {"deepseek": (0.435, 0.87), "grok": (5.0, 15.0)}

PAIR_RE = re.compile(r"<<<Q>>>(.*?)<<<A>>>(.*?)(?=<<<END>>>|<<<Q>>>|\Z)", re.S)


def parse_pairs(raw):
    """Extract (instruction, response) pairs from a batch output. Robust to a missing final
    <<<END>>> (the lookahead also terminates on the next <<<Q>>> or end-of-string)."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)     # DeepSeek/Qwen thinking
    out = []
    for m in PAIR_RE.finditer(raw):
        q = m.group(1).strip().strip("`").strip()
        a = m.group(2).split("<<<END>>>")[0].strip().strip("`").strip()
        if q and a and len(a) >= 2:
            out.append((q, a))
    return out


def attach_passage(q, passage):
    """Grounded tasks: append the real passage to the question so the pair is SELF-CONTAINED (the
    model summarizes/extracts text the trainee can actually see). Guards against double-embed if the
    model already pasted the text into Q. This is the fix for the 'grounded task, no ground' bug."""
    if not passage:
        return q
    probe = " ".join(passage.split()[:6])
    if probe and probe in q:                                     # model already pasted it
        return q
    return q.rstrip() + "\n\n" + passage


def build_realtext_pool(k, seed):
    """Sample ~k real Serbian passages (~110 words ≈ 150 tok) for grounded tasks. Source = fineweb2
    web prose ONLY (clean UTF-8 ekavica/ijekavica) -- deliberately NOT opensubtitles (CP1250 mojibake
    'Èokoladni/æe' + dubbed-dialogue register) or literature (archaic Cyrillic yat), which made poor
    grounded passages in the first pass. Filter by the doc's `source` field."""
    rng = random.Random(seed)
    files = glob.glob(str(ROOT / "data/processed/dedup/*.jsonl.gz"))
    rng.shuffle(files)
    pool = []
    for f in files:
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("source") != "fineweb2":            # skip subtitles / literature / reddit
                        continue
                    t = d.get("text", "").strip()
                    w = t.split()
                    if 40 <= len(w) <= 500:
                        pool.append(" ".join(w[:110]))
                    if len(pool) >= k:
                        return pool
        except OSError:
            continue
    return pool


def sft_call(url, key, model, prompt, temperature, timeout, reasoning=""):
    payload = {"model": model, "temperature": temperature,
               "messages": [{"role": "system", "content": SFT_SYSTEM},
                            {"role": "user", "content": prompt}]}
    if reasoning:                                            # e.g. "none" to kill Gemini-3.5 thinking
        payload["reasoning_effort"] = reasoning
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "serblo/1.0"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = (d.get("choices") or [{}])[0].get("message", {})
    return (msg.get("content") or ""), time.time() - t, (d.get("usage") or {})


def sft_generate(url, limiter, model, prompt, temperature, timeout, reasoning=""):
    """Key-rotating call with backoff. Returns (text, secs, ok, rate_limited, usage)."""
    wall, rl, last_err, usage = 0.0, False, "empty", {}
    for attempt in range(len(BACKOFF) + 1):
        key = limiter.get_key()
        try:
            text, secs, usage = sft_call(url, key, model, prompt, temperature, timeout, reasoning)
            wall += secs
            if text.strip():
                return text, wall, True, rl, usage
            last_err = "empty"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                rl, last_err = True, "HTTP 429"
            elif e.code in (500, 502, 503):
                last_err = f"HTTP {e.code}"
            else:
                body = e.read()[:200]
                if e.code == 400 and b"Provider returned error" in body:   # transient upstream flake -> retry
                    last_err = "HTTP 400 (provider flake)"
                else:
                    return f"HTTP {e.code}: {body!r}", wall, False, rl, {}
        except TimeoutError:
            rl, last_err = True, "read timeout"
        except Exception as e:                              # noqa: BLE001, capture the real reason
            if isinstance(getattr(e, "reason", None), TimeoutError):
                rl, last_err = True, "connect timeout"
            else:
                last_err = f"{type(e).__name__}: {e}"[:160]
        if attempt < len(BACKOFF):
            time.sleep(BACKOFF[attempt])
    return last_err, wall, False, rl, usage


def run_claude(model, prompt, timeout):
    """Generate via the local `claude -p` CLI (flat-rate Max plan, cost 0). Model is pinned so we
    never hit the default model's over-broad safeguard. SFT_SYSTEM is prepended as the instruction. (text, secs, ok)."""
    t = time.time()
    try:
        r = subprocess.run(["claude", "-p", "--model", model, SFT_SYSTEM + "\n\n" + prompt],
                           capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        out = (r.stdout or "").strip()
        return out, time.time() - t, bool(out) and r.returncode == 0
    except subprocess.TimeoutExpired:
        return "", time.time() - t, False


def build_jobs(target_pairs, per_call, pool, seed, tasks=None):
    rng = random.Random(seed)
    quota = {k: v for k, v in QUOTA.items() if not tasks or k in tasks}
    total = sum(quota.values())
    jobs = []
    for task, q in quota.items():
        pairs = max(per_call, round(target_pairs * q / total))
        ncalls = max(1, -(-pairs // per_call))              # ceil
        for i in range(ncalls):
            job = {"id": f"{task}-{i}", "task": task, "n": per_call,
                   "register": rng.choice(REGISTER_POOL), "topic": rng.choice(TOPICS)}
            if task in GROUNDED:
                job["real_text"] = rng.choice(pool) if pool else ""
            jobs.append(job)
    rng.shuffle(jobs)
    return jobs


def build_prompt(job):
    reg_block = REGISTER_BLOCKS.get(job.get("register"), REGISTER_BLOCKS["neutralni"])
    p = TASK_PROMPTS[job["task"]].format(
        n=job["n"], register=reg_block,
        topic=job.get("topic", "svakodnevni život"), real_text=job.get("real_text", ""))
    if job["task"] in GROUNDED:                                  # Q = instruction stem only; we attach the text
        p += ("\n\nIMPORTANT: each <<<Q>>> must be ONLY the short instruction (e.g. 'Sažmi ovaj tekst.', "
              "'O čemu se radi?', 'Izvuci imena.') and must NOT repeat or paste the TEXT above -- the text "
              "is attached to the question automatically. <<<A>>> is the answer about that text.")
    return p


def cost_of(price, usage):
    """price = (in_per_1M, out_per_1M) USD, or None for free."""
    if not price:
        return 0.0
    pin, pout = price
    return (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, choices=list(HOSTS) + ["claude"],
                    help="HTTP host (nim/groq/sambanova/openrouter/...) or 'claude' for the local claude -p CLI")
    ap.add_argument("--model", required=True)
    ap.add_argument("--target-pairs", type=int, default=500)
    ap.add_argument("--per-call", type=int, default=4)
    ap.add_argument("--pool-size", type=int, default=400, help="real passages sampled for grounded tasks (raise for content variety)")
    ap.add_argument("--tasks", default="", help="comma-sep task subset, e.g. write,rewrite,chitchat (default: all 8)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rpm-per-key", type=float, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--reasoning", default="", help="reasoning_effort passthrough (e.g. 'none' to kill Gemini-3.5 thinking)")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--max-cost", type=float, default=0.0, help="USD hard cap (0 = uncapped/free host)")
    ap.add_argument("--in-price", type=float, default=0.0, help="USD per 1M input tokens (overrides PRICES)")
    ap.add_argument("--out-price", type=float, default=0.0, help="USD per 1M output tokens (overrides PRICES)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    is_cli = a.host == "claude"
    url = None if is_cli else HOSTS[a.host][0]
    price = None if is_cli else ((a.in_price, a.out_price) if a.out_price else PRICES.get(a.host))
    limiter = None if is_cli else RateLimiter(load_keys(a.host), a.rpm_per_key)

    tasks = set(a.tasks.split(",")) if a.tasks else None
    pool = build_realtext_pool(a.pool_size, a.seed) if any(t in GROUNDED for t in (tasks or QUOTA)) else []
    if pool:
        print(f"real-text pool: {len(pool)} passages")
    jobs = build_jobs(a.target_pairs, a.per_call, pool, a.seed, tasks)

    done = set()
    raw_path = out / "raw.jsonl"
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{len(jobs)} jobs, {len(done)} done, {len(todo)} to run  (~{len(todo) * a.per_call} pairs)")

    lock = threading.Lock()
    state = {"pairs": 0, "calls": 0, "err": 0, "cost": 0.0, "stop": False}
    fp = open(out / "pairs.jsonl", "a", encoding="utf-8")
    fr = open(raw_path, "a", encoding="utf-8")
    fe = open(out / "errors.jsonl", "a", encoding="utf-8")

    def do_one(job):
        if state["stop"]:
            return
        if is_cli:
            text, secs, ok = run_claude(a.model, build_prompt(job), a.timeout)
            usage = {}
        else:
            text, secs, ok, _rl, usage = sft_generate(url, limiter, a.model, build_prompt(job),
                                                       a.temperature, a.timeout, a.reasoning)
        c = cost_of(price, usage)
        pairs = parse_pairs(text) if ok else []
        with lock:
            state["calls"] += 1
            state["cost"] += c
            if ok and pairs:
                grounded = job["task"] in GROUNDED
                for j, (q, ans) in enumerate(pairs):
                    instr = attach_passage(q, job.get("real_text", "")) if grounded else q
                    fp.write(json.dumps({"id": f"{job['id']}-{j}", "task": job["task"],
                                         "instruction": instr, "response": ans, "register": job.get("register"),
                                         "topic": job.get("topic"), "source": a.host, "model": a.model,
                                         "ts": int(time.time())}, ensure_ascii=False) + "\n")
                state["pairs"] += len(pairs)
            else:
                state["err"] += 1
                fe.write(json.dumps({"id": job["id"], "task": job["task"], "ok": ok,
                                     "err": text[:200] if not ok else "no_pairs_parsed"},
                                    ensure_ascii=False) + "\n")
            fr.write(json.dumps({"id": job["id"], "task": job["task"], "npairs": len(pairs),
                                 "secs": round(secs, 1), "cost": round(c, 6),
                                 "real_text": job.get("real_text", ""), "raw": text},  # persist ground
                                ensure_ascii=False) + "\n")
            fp.flush(); fr.flush(); fe.flush()
            if state["calls"] % 10 == 0 or state["stop"]:
                print(f"  {state['calls']}/{len(todo)} calls · {state['pairs']} pairs · "
                      f"{state['err']} err · ${state['cost']:.4f}")
            if a.max_cost and state["cost"] >= a.max_cost:
                state["stop"] = True

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(do_one, todo))

    fp.close(); fr.close(); fe.close()
    print(f"\nDONE: {state['pairs']} pairs from {state['calls']} calls · {state['err']} failed calls · "
          f"${state['cost']:.4f}\n  -> {out}/pairs.jsonl")


if __name__ == "__main__":
    main()
