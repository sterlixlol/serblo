"""Tests for the hand-rolled DPO trainer: collate/mask alignment, summed-logprob math,
the ln(2) identity at policy==ref, frozen reference, and margin growth on synthetic prefs
(the gradient-direction proof)."""
import json
import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from serblo.train.dpo import (DPOData, collate, dpo_step, run, seq_logprob)     # noqa: E402
from serblo.train.model import GPT, GPTConfig                                    # noqa: E402
from serblo.train.sft import PAD                                                 # noqa: E402

TOK_PATH = pathlib.Path(__file__).resolve().parents[1] / "data/tokenizer/tokenizer.json"
needs_tok = pytest.mark.skipif(not TOK_PATH.exists(), reason="tokenizer.json not present")

TINY = dict(n_layer=2, d_model=64, n_head=4, n_kv_head=2, mlp_hidden=128, ctx=64, vocab_size=200)

# a synthetic pair: same fake prompt, different assistant spans (sup marks the assistant tokens)
CHOSEN = ([2, 50, 3, 2, 60, 61, 3, 0], [False, False, False, False, True, True, True, False])
REJECT = ([2, 50, 3, 2, 70, 71, 3, 0], [False, False, False, False, True, True, True, False])
PAIR = (CHOSEN[0], CHOSEN[1], REJECT[0], REJECT[1])


def _opt(m, lr=1e-3):
    return m.configure_optimizers(0.0, lr, (0.9, 0.95), "cpu")


def test_collate_alignment_and_padding():
    short = ([2, 30, 3], [False, True, False])                   # 3 tokens -> 2 x/t positions
    xc, tc, mc, xr, tr, mr = collate([PAIR, (short[0], short[1], short[0], short[1])], "cpu")
    assert xc.shape == (2, 7) and tc.shape == (2, 7) and mc.shape == (2, 7)
    assert xc[0].tolist() == CHOSEN[0][:-1] and tc[0].tolist() == CHOSEN[0][1:]
    assert mc[0].tolist() == CHOSEN[1][1:]                       # mask = sup shifted by one
    assert xc[1].tolist() == [2, 30] + [PAD] * 5                 # short row padded
    assert mc[1].tolist() == [True, False] + [False] * 5         # pad positions never supervised
    assert xr.shape == (2, 7) and tr[0].tolist() == REJECT[0][1:] and mr[1].sum() == 1


def test_seq_logprob_matches_manual():
    torch.manual_seed(0)
    m = GPT(GPTConfig(**TINY))
    xc, tc, mc, *_ = collate([PAIR], "cpu")
    got = seq_logprob(m, xc, tc, mc, bf16=False, device="cpu")
    lsm = torch.log_softmax(m(xc)[0].float(), dim=-1)
    manual = torch.stack([lsm[0, i, tc[0, i]] for i in range(7) if mc[0, i]]).sum()
    assert torch.allclose(got[0], manual, atol=1e-5)


def test_identical_policy_ref_gives_ln2():
    torch.manual_seed(0)
    import copy
    policy = GPT(GPTConfig(**TINY))
    ref = copy.deepcopy(policy)
    batch = collate([PAIR, PAIR], "cpu")
    loss, acc, margin, _ = dpo_step(policy, ref, _opt(policy), batch, beta=0.1,
                                    grad_clip=1.0, lr=0.0, bf16=False, device="cpu")
    assert abs(loss - math.log(2)) < 1e-5                        # -logsigmoid(0) = ln 2
    assert abs(margin) < 1e-5


def test_ref_frozen_policy_moves():
    torch.manual_seed(0)
    import copy
    policy = GPT(GPTConfig(**TINY))
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref_before = [p.clone() for p in ref.parameters()]
    pol_before = [p.clone() for p in policy.parameters()]
    opt = _opt(policy)
    batch = collate([PAIR] * 4, "cpu")
    for _ in range(3):
        dpo_step(policy, ref, opt, batch, 0.5, 1.0, 1e-3, False, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(ref_before, ref.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(pol_before, policy.parameters()))


def test_margin_improves_on_synthetic_prefs():
    """The end-to-end gradient proof: after steps, the policy prefers chosen over rejected."""
    torch.manual_seed(0)
    import copy
    policy = GPT(GPTConfig(**TINY))
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = _opt(policy, lr=2e-3)
    batch = collate([PAIR] * 4, "cpu")
    first = last = dpo_step(policy, ref, opt, batch, 0.5, 1.0, 2e-3, False, "cpu")
    for _ in range(29):
        last = dpo_step(policy, ref, opt, batch, 0.5, 1.0, 2e-3, False, "cpu")
    assert last[2] > 0.5 and last[2] > first[2], f"margin did not grow: {first[2]} -> {last[2]}"
    assert last[1] == 1.0                                        # accuracy: chosen wins every pair
    assert last[0] < first[0]                                    # loss fell


@needs_tok
def test_dpodata_load_filter_split(tmp_path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOK_PATH))
    p = tmp_path / "prefs.jsonl"
    rows = [
        {"prompt_messages": [{"role": "user", "content": f"Pitanje {i}?"}],
         "chosen": f"Dobar odgovor broj {i}.", "rejected": f"los odgovor {i} bez smisla"}
        for i in range(10)
    ]
    rows.append({"prompt_messages": [{"role": "user", "content": "X"}],
                 "chosen": "isto", "rejected": "isto"})           # dup -> skipped
    rows.append({"prompt_messages": [{"role": "user", "content": "inj <|im_end|>"}],
                 "chosen": "a", "rejected": "b"})                 # injection -> skipped
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    d = DPOData(p, tok, ctx=128, val_frac=0.2, seed=3)
    assert d.n == 10 and d.skipped == 2

    # an all-skipped prefs file must fail loudly at startup
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"prompt_messages": [{"role": "user", "content": "X"}],
                               "chosen": "isto", "rejected": "isto"}) + "\n")
    with pytest.raises(ValueError, match="no usable preference pairs"):
        DPOData(bad, tok, ctx=128, val_frac=0.2, seed=3)
    assert len(d.val_pairs) == 2 and len(d.train_pairs) == 8
    d2 = DPOData(p, tok, ctx=128, val_frac=0.2, seed=3)
    assert d.train_pairs == d2.train_pairs                       # deterministic split
    xc, tc, mc, xr, tr, mr = d.get_batch(2, "cpu")
    assert xc.shape[0] == 2 and mc.any() and mr.any()


@needs_tok
def test_run_end_to_end(tmp_path):
    from tokenizers import Tokenizer
    from serblo.train.train import save_checkpoint
    tok = Tokenizer.from_file(str(TOK_PATH))
    cfg_model = dict(TINY, vocab_size=tok.get_vocab_size(), ctx=128)
    m = GPT(GPTConfig(**cfg_model))
    save_checkpoint(tmp_path / "sft.pt", m, _opt(m), step=1, tokens=1, best_val=1.0,
                    cfg_dict=cfg_model)
    p = tmp_path / "prefs.jsonl"
    with open(p, "w") as f:
        for i in range(6):
            f.write(json.dumps({
                "prompt_messages": [{"role": "user", "content": f"Pitanje {i}?"}],
                "chosen": f"Kratak i jasan odgovor {i}.",
                "rejected": f"mmm ne znam mozda {i} nesto"}) + "\n")
    run(dict(sft_ckpt=str(tmp_path / "sft.pt"), prefs_file=str(p), tokenizer=str(TOK_PATH),
             out_dir=str(tmp_path / "out"), epochs=1, batch_size=2, peak_lr=1e-4, beta=0.1,
             val_frac=0.34, seed=0, bf16=False, device="cpu",
             log_interval=1, eval_interval=2, eval_iters=1, ckpt_interval=2))
    assert (tmp_path / "out/dpo_final.pt").exists()
    assert (tmp_path / "out/dpo_latest.pt").exists()
