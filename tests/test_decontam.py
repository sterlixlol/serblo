"""Decontam: 13-gram overlap between train and the eval set drops the train doc; eval is untouched.

Also pins the BCS-specific rule that matching keeps diacritics (č ć š ž đ are load-bearing), so a
diacritic-stripped near-copy does NOT count as contamination.
"""
import gzip
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from serblo.eval.decontam import build_eval_index, contamination, ngrams, normalize_for_match  # noqa: E402


def test_normalize_keeps_diacritics_and_casefolds():
    assert normalize_for_match("Četiri, PET! šest.") == ["četiri", "pet", "šest"]


def test_ngrams_short_text_collapses_to_one_gram():
    assert ngrams(["a", "b"], 13) == {"a b"}          # < n -> whole text, never empty
    assert ngrams([], 13) == set()
    assert len(ngrams([f"w{i}" for i in range(20)], 13)) == 8   # 20 distinct words -> 8 windows


def test_contamination_flags_shared_span_only():
    probe = "ovo je jedan prilično dugačak primer recenice koji sluzi kao test za proveru preklapanja teksta danas"
    index = build_eval_index([probe], 13)
    # a training doc that embeds the probe verbatim -> contaminated
    dirty = "Uvod. " + probe + " I onda nastavak price."
    assert contamination(dirty, index, 13) is not None
    # unrelated doc -> clean
    assert contamination("potpuno druga recenica bez ikakvog preklapanja sa test skupom ovde", index, 13) is None


def test_diacritics_are_load_bearing():
    probe = "mačka je sela na ćošak i gledala kroz žuti prozor dok je đak učio"
    index = build_eval_index([probe], 13)
    stripped = probe.replace("č", "c").replace("ć", "c").replace("ž", "z").replace("đ", "dj").replace("š", "s")
    assert contamination(probe, index, 13) is not None      # exact -> caught
    assert contamination(stripped, index, 13) is None        # diacritics differ -> NOT the same text


def test_runner_removes_contaminated_docs(tmp_path):
    probe = "held out probe sentence that is comfortably longer than thirteen words so it forms real ngrams here"
    (tmp_path / "eval.jsonl").write_text(json.dumps({"prompt": probe}) + "\n")

    train = tmp_path / "train"
    train.mkdir()
    with gzip.open(train / "shard-00000.jsonl.gz", "wt", encoding="utf-8") as w:
        w.write(json.dumps({"text": "a clean unrelated training document about something else entirely",
                            "source": "fineweb2"}) + "\n")
        w.write(json.dumps({"text": "leak: " + probe + " and more", "source": "fineweb2"}) + "\n")  # contaminated
        w.write(json.dumps({"text": "another totally clean document with no overlap at all",
                            "source": "literature"}) + "\n")

    out = tmp_path / "out"
    env = {"PYTHONPATH": "src", "PATH": os.environ["PATH"]}
    r = subprocess.run([sys.executable, "-m", "serblo.eval.run_decontam",
                        "--train", str(train), "--eval", str(tmp_path / "eval.jsonl"),
                        "--out", str(out), "--write"],
                       cwd=str(ROOT), env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"

    kept = []
    for shard in sorted(out.glob("shard-*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as z:
            kept += [json.loads(l)["text"] for l in z if l.strip()]
    assert len(kept) == 2                              # the contaminated doc dropped
    assert all(probe not in t for t in kept)
    assert "removed 1" in r.stdout
