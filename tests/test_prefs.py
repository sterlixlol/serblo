"""Tests for the preference-pair scripts (gen_prefs / judge_prefs): im_end truncation, batched
K-sampling, judge-prompt construction (length-bias guard), verdict parsing, shuffle mapping."""
import importlib.util
import json
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serblo.train.model import GPT, GPTConfig                                  # noqa: E402

TOK_PATH = ROOT / "data/tokenizer/tokenizer.json"
needs_tok = pytest.mark.skipif(not TOK_PATH.exists(), reason="tokenizer.json not present")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen_prefs = _load("gen_prefs")
judge_prefs = _load("judge_prefs")


def test_truncate_at():
    assert gen_prefs.truncate_at([5, 6, 3, 7, 3], 3) == [5, 6]
    assert gen_prefs.truncate_at([5, 6, 7], 3) == [5, 6, 7]      # no stop token -> untouched
    assert gen_prefs.truncate_at([3, 9], 3) == []                # stop-first -> empty


@needs_tok
def test_sample_k_shapes_and_stop():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOK_PATH))
    torch.manual_seed(0)
    m = GPT(GPTConfig(n_layer=2, d_model=64, n_head=4, n_kv_head=2, mlp_hidden=128,
                      ctx=128, vocab_size=tok.get_vocab_size())).eval()
    out = gen_prefs.sample_k(m, tok, [{"role": "user", "content": "Zdravo?"}],
                             k=3, max_new=12, temp=1.0, top_k=50, rep_penalty=1.3, device="cpu")
    assert len(out) == 3 and all(isinstance(s, str) for s in out)
    assert all("<|" not in s for s in out)                       # specials never leak into text


@needs_tok
def test_sample_batch_multi_prompt():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOK_PATH))
    torch.manual_seed(0)
    m = GPT(GPTConfig(n_layer=2, d_model=64, n_head=4, n_kv_head=2, mlp_hidden=128,
                      ctx=128, vocab_size=tok.get_vocab_size())).eval()
    p1 = gen_prefs.build_chat_prompt(tok, [{"role": "user", "content": "Zdravo svima?"}])
    p2 = p1[:-1] + [p1[-1]]                                      # same length, distinct list
    out = gen_prefs.sample_batch(m, tok, [p1, p2], k=2, max_new=10,
                                 temp=1.0, top_k=50, rep_penalty=1.3, device="cpu")
    assert len(out) == 2 and all(len(rows) == 2 for rows in out)
    assert all(isinstance(s, str) for rows in out for s in rows)
    with pytest.raises(AssertionError):                          # unequal lengths must refuse
        gen_prefs.sample_batch(m, tok, [p1, p1 + p1[:2]], 2, 4, 1.0, 50, 1.3, "cpu")


def test_bucket_by_length():
    items = [("a", [], [1, 2, 3]), ("b", [], [4, 5]), ("c", [], [6, 7, 8]), ("d", [], [9])]
    b = gen_prefs.bucket_by_length(items)
    assert sorted(b) == [1, 2, 3]
    assert [i[0] for i in b[3]] == ["a", "c"] and [i[0] for i in b[2]] == ["b"]


def test_judge_prompt_contains_everything():
    p = judge_prefs.build_judge_prompt(
        [{"role": "user", "content": "Koliko je 2+2?"}], ["Četiri.", "Pet, mislim.", "4"])
    assert "Koliko je 2+2?" in p
    assert "REPLY 1:" in p and "REPLY 3:" in p and "Pet, mislim." in p
    assert '"best"' in p and "1-3" in p
    assert "length is not quality" in judge_prefs.JUDGE_SYSTEM   # the anti-length-bias guard


def test_parse_verdict():
    pv = judge_prefs.parse_verdict
    assert pv('{"best": 2, "worst": 1}', 3) == (1, 0)
    assert pv('Sure! ```json\n{"best": 3, "worst": 2}\n```', 3) == (2, 1)
    assert pv('{"best": 1, "worst": 1}', 3) is None              # best == worst
    assert pv('{"best": 4, "worst": 1}', 3) is None              # out of range
    assert pv('{"best": "who knows"}', 3) is None                # malformed
    assert pv("no json here", 3) is None
    # must scan past a preamble object, and reject booleans
    assert pv('{"note": "ok"} then {"best": 2, "worst": 3}', 3) == (1, 2)
    assert pv('{"best": true, "worst": 2}', 3) is None           # int(True)==1 must not sneak in


def test_iter_jsonl_torn_lines(tmp_path):
    torn_final = tmp_path / "final.jsonl"
    torn_final.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c", "tor')   # crash mid-append
    assert [r["id"] for r in judge_prefs.iter_jsonl(torn_final)] == ["a", "b"]
    assert [r["id"] for r in gen_prefs.iter_jsonl(torn_final)] == ["a", "b"]   # mirror helper

    torn_middle = tmp_path / "middle.jsonl"                      # real corruption -> loud
    torn_middle.write_text('{"id": "a"}\n{"tor\n{"id": "c"}\n')
    with pytest.raises(Exception):
        list(judge_prefs.iter_jsonl(torn_middle))
    with pytest.raises(Exception):
        list(gen_prefs.iter_jsonl(torn_middle))


def test_build_prompt_pools(tmp_path):
    """Preference prompts must be UNSEEN by SFT; first N eligible become the eval-only set."""
    full, train = tmp_path / "full.jsonl", tmp_path / "train.jsonl"
    with open(full, "w") as f:
        for i in range(10):
            f.write(json.dumps({"id": f"p{i}", "messages": [
                {"role": "user", "content": f"Pitanje {i}?"},
                {"role": "assistant", "content": f"Odgovor {i}."}]}) + "\n")
    with open(train, "w") as f:                              # SFT saw p0..p4
        for i in range(5):
            f.write(json.dumps({"id": f"p{i}"}) + "\n")
    rp = tmp_path / "eval_prompts.jsonl"
    excluded, reserved = gen_prefs.build_prompt_pools(full, train, rp, n_reserve=2)
    assert excluded == {f"p{i}" for i in range(5)}
    assert reserved == {"p5", "p6"}                          # first eligible AFTER exclusion
    kept = [r["id"] for r in gen_prefs.iter_jsonl(rp)]
    assert kept == ["p5", "p6"] and rp.exists()
    # eligible for sampling = full - excluded - reserved = p7, p8, p9
    remaining = {f"p{i}" for i in range(10)} - excluded - reserved
    assert remaining == {"p7", "p8", "p9"}
    # second call: reserve file already exists -> honored, not rebuilt
    _, reserved2 = gen_prefs.build_prompt_pools(full, train, rp, n_reserve=99)
    assert reserved2 == {"p5", "p6"}


def test_judge_one_skips_degenerate():
    rec = {"id": "x1", "prompt_messages": [{"role": "user", "content": "?"}],
           "samples": ["isto", "isto", ""]}
    res = judge_prefs.judge_one(rec, "http://x", None, "m", 0.1, 5)
    assert res == {"skip": "fewer than 2 distinct non-empty samples"}


def test_judge_one_shuffle_roundtrip(monkeypatch):
    """The verdict must map back through the per-id shuffle to the ORIGINAL sample texts."""
    rec = {"id": "abc", "prompt_messages": [{"role": "user", "content": "?"}],
           "samples": ["prvi odgovor", "drugi odgovor", "treci odgovor"]}
    seen = {}

    def fake_call(url, key, model, prompt, temperature, timeout):
        seen["prompt"] = prompt
        return '{"best": 1, "worst": 2}'                        # judge prefers REPLY 1 as shown

    class FakeLimiter:
        def get_key(self):
            return "k"

    monkeypatch.setattr(judge_prefs, "judge_call", fake_call)
    res = judge_prefs.judge_one(rec, "http://x", FakeLimiter(), "m", 0.1, 5)
    assert "chosen" in res and res["chosen"] != res["rejected"]
    assert res["chosen"] in rec["samples"] and res["rejected"] in rec["samples"]
    # and the chosen text really is the one shown as REPLY 1 in the shuffled prompt
    reply1 = seen["prompt"].split("REPLY 1:\n")[1].split("\n\nREPLY")[0]
    assert res["chosen"] == reply1
