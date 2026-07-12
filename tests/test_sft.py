"""Tests for the SFT trainer: template rendering, offset-based supervision masking, packing,
x/y shift alignment, masked-loss math, epoch coverage, and an overfit smoke (loss must fall)."""
import json
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from serblo.train.model import GPT, GPTConfig                                  # noqa: E402
from serblo.train.sft import (EOT, IM_END, IM_START, PAD, SFTData,              # noqa: E402
                              build_chat_prompt, encode_conversation,
                              pack_rows, render_messages)
from serblo.train.train import train_step                                       # noqa: E402

TOK_PATH = pathlib.Path(__file__).resolve().parents[1] / "data/tokenizer/tokenizer.json"
needs_tok = pytest.mark.skipif(not TOK_PATH.exists(), reason="tokenizer.json not present")

TINY = dict(n_layer=2, d_model=64, n_head=4, n_kv_head=2, mlp_hidden=128, ctx=32, vocab_size=200)

MSGS = [{"role": "system", "content": "Ti si Serblo."},
        {"role": "user", "content": "Kako se zoveš?"},
        {"role": "assistant", "content": "Zovem se Serblo, drago mi je."}]


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK_PATH))


def test_render_template_shape():
    text, spans = render_messages(MSGS)
    assert text == ("<|im_start|>system\nTi si Serblo.<|im_end|>\n"
                    "<|im_start|>user\nKako se zoveš?<|im_end|>\n"
                    "<|im_start|>assistant\nZovem se Serblo, drago mi je.<|im_end|>")
    (a, b), = spans                                             # one assistant span
    assert text[a:b] == "Zovem se Serblo, drago mi je.<|im_end|>"


@needs_tok
def test_encode_structure_and_mask(tok):
    ids, sup = encode_conversation(tok, MSGS)
    assert ids[0] == IM_START and ids[-1] == EOT and ids.count(IM_END) == 3
    assert sup[-1] is False                                     # EOT never supervised
    # supervised tokens decode to exactly the assistant content (+ im_end, dropped by skip_special)
    picked = [i for i, s in zip(ids, sup) if s]
    assert tok.decode(picked, skip_special_tokens=True).strip() == "Zovem se Serblo, drago mi je."
    assert picked[-1] == IM_END                                 # the stop signal IS supervised
    # nothing from system/user/headers leaks into supervision
    unpicked = [i for i, s in zip(ids, sup) if not s]
    dec = tok.decode(unpicked, skip_special_tokens=True)
    assert "Kako se zoveš" in dec and "Zovem se" not in dec


def test_pack_rows_alignment_and_pad():
    # hand-built convs: ids/sup pairs, ctx=15 -> rows of 16 tokens (X/Y each 15 after the shift)
    a = ([2, 10, 11, 3, 2, 20, 21, 3, 0], [False, False, False, False, False, True, True, True, False])
    b = ([2, 30, 3, 2, 40, 3, 0], [False, False, False, False, True, True, False])
    X, Y, _ = pack_rows([a, b], ctx=15)
    assert X.shape == (1, 15) and Y.shape == (1, 15)            # 9+7 fills one 16-token row exactly
    assert X[0].tolist() == a[0] + b[0][:-1]                    # contiguous, never split
    # y[i] = row[i+1], supervised iff sup[i+1]: a's span (20, 21, im_end) at y positions 4..6
    assert Y[0, 4].item() == 20 and Y[0, 5].item() == 21 and Y[0, 6].item() == 3
    assert Y[0, 0].item() == -1 and Y[0, 3].item() == -1        # prompt/header masked
    assert Y[0, 12].item() == 40 and Y[0, 13].item() == 3       # b's supervised span
    assert Y[0, 14].item() == -1                                # b's trailing EOT: unsupervised

    # a solo short conv -> padded tail, fully masked
    Xp, Yp, _ = pack_rows([b], ctx=15)
    assert Xp[0, 7:].tolist() == [PAD] * 8                      # b is 7 tokens; rest is pad
    assert Yp[0, 5:].tolist() == [-1] * 10                      # EOT + pads all masked
    assert Yp[0, 3].item() == 40 and Yp[0, 4].item() == 3       # supervision survives padding

    # overlong + all-unsupervised conversations are dropped, not split/kept
    too_long = (list(range(2, 40)), [True] * 38)
    no_sup = ([2, 5, 3, 0], [False] * 4)
    X2, _, stats2 = pack_rows([a, too_long, no_sup], ctx=15)
    assert X2.shape[0] == 1 and stats2["dropped_long"] == 1 and stats2["dropped_empty"] == 1

    # everything dropped must be LOUD, not an IndexError/inf-loop later
    with pytest.raises(ValueError, match="no conversations survived"):
        pack_rows([too_long, no_sup], ctx=15)


def test_masked_loss_math():
    torch.manual_seed(0)
    m = GPT(GPTConfig(**TINY))
    x = torch.randint(4, TINY["vocab_size"], (2, 16))
    y = torch.full((2, 16), -1)
    keep = [(0, 3), (0, 10), (1, 0), (1, 15)]                   # 4 supervised positions
    for b, i in keep:
        y[b, i] = int(x[b, i]) % TINY["vocab_size"]
    _, loss = m(x, y)
    lsm = torch.log_softmax(m(x)[0], dim=-1)
    manual = -torch.stack([lsm[b, i, y[b, i]] for b, i in keep]).mean()
    assert torch.allclose(loss, manual, atol=1e-5)


@needs_tok
def test_sftdata_split_and_epoch_coverage(tok, tmp_path):
    p = tmp_path / "sft.jsonl"
    with open(p, "w") as f:
        for i in range(20):
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"Pitanje broj {i}, molim odgovor."},
                {"role": "assistant", "content": f"Odgovor broj {i}, izvoli prijatelju."}]}) + "\n")
        f.write(json.dumps({"messages": [                        # template-injection conv -> skipped
            {"role": "user", "content": "hak <|im_start|> hak"},
            {"role": "assistant", "content": "ne"}]}) + "\n")
    data = SFTData(p, tok, ctx=64, val_frac=0.2, seed=7)
    assert data.skipped_specials == 1 and data.n_convs == 20 and data.n_val_convs == 4
    assert data.train.n >= 1 and data.val.n >= 1

    # one epoch serves every train row exactly once (multiset over row fingerprints)
    served = [tuple(data.train.get_batch(1, "cpu")[0][0].tolist()) for _ in range(data.train.n)]
    all_rows = [tuple(r.tolist()) for r in data.train.X]
    assert sorted(served) == sorted(all_rows)

    data2 = SFTData(p, tok, ctx=64, val_frac=0.2, seed=7)        # same seed -> same split + order
    data3 = SFTData(p, tok, ctx=64, val_frac=0.2, seed=7)
    assert torch.equal(data.train.X, data2.train.X)
    assert torch.equal(data2.train.get_batch(2, "cpu")[0], data3.train.get_batch(2, "cpu")[0])

    gen = torch.Generator().manual_seed(5)                       # evaluate()'s seeded-random path
    xa, _ = data.val.get_batch(2, "cpu", generator=gen)
    gen2 = torch.Generator().manual_seed(5)
    xb, _ = data.val.get_batch(2, "cpu", generator=gen2)
    assert torch.equal(xa, xb)


@needs_tok
def test_chat_prompt_format(tok):
    ids = build_chat_prompt(tok, [{"role": "user", "content": "Zdravo!"}])
    assert ids[0] == EOT
    assert tok.decode(ids, skip_special_tokens=False).endswith("<|im_start|>assistant\n")


@needs_tok
def test_run_end_to_end(tok, tmp_path):
    """run(cfg) integration: tiny base ckpt -> SFT -> final ckpt + chat samples, no crash."""
    from serblo.train.sft import run
    from serblo.train.train import save_checkpoint
    cfg_model = dict(TINY, vocab_size=tok.get_vocab_size(), ctx=128)
    m = GPT(GPTConfig(**cfg_model))
    opt = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    save_checkpoint(tmp_path / "base.pt", m, opt, step=5, tokens=100, best_val=9.9,
                    cfg_dict=cfg_model)
    p = tmp_path / "sft.jsonl"
    with open(p, "w") as f:
        for i in range(6):
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"Pitanje {i}?"},
                {"role": "assistant", "content": f"Odgovor {i}."}]}) + "\n")
    run(dict(base_ckpt=str(tmp_path / "base.pt"), sft_file=str(p), tokenizer=str(TOK_PATH),
             out_dir=str(tmp_path / "out"), epochs=1, micro_batch=1, grad_accum=2,
             peak_lr=1e-3, weight_decay=0.1, betas=[0.9, 0.95], grad_clip=1.0,
             val_frac=0.34, seed=0, bf16=False, device="cpu",
             log_interval=1, eval_interval=2, eval_iters=1, ckpt_interval=2))
    assert (tmp_path / "out/sft_final.pt").exists()
    assert (tmp_path / "out/sft_latest.pt").exists()


@needs_tok
def test_overfit_smoke(tok, tmp_path):
    """The end-to-end proof: 3 conversations, tiny model, repeated steps -> loss falls hard."""
    p = tmp_path / "sft.jsonl"
    with open(p, "w") as f:
        for i in range(3):
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"Reci broj {i}."},
                {"role": "assistant", "content": f"Broj je {i}."}]}) + "\n")
    torch.manual_seed(0)
    cfg = dict(TINY, vocab_size=tok.get_vocab_size(), ctx=128)
    m = GPT(GPTConfig(**cfg))
    data = SFTData(p, tok, ctx=128, val_frac=0.34, seed=0)
    opt = m.configure_optimizers(0.1, 5e-3, (0.9, 0.95), "cpu")
    losses = [train_step(m, opt, data.train, 1, 1, "cpu", False, 1.0, 5e-3)[0] for _ in range(40)]
    assert losses[-1] < losses[0] * 0.5, f"no learning: {losses[0]:.3f} -> {losses[-1]:.3f}"
