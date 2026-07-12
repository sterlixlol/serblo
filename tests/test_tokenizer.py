"""Tests for the 32k byte-level BPE tokenizer (M4). Trains a tiny vocab on a fixture so the
unit tests are fast + hermetic; the real 32k train + token/word-ratio check happens on the GCS
sample at build time (scripts/train_tokenizer.py), asserted there."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from serblo.tokenizer import load_tokenizer                     # noqa: E402
from serblo.tokenizer.train_bpe import SPECIAL_TOKENS, train_bpe  # noqa: E402

# BCS text with the full load-bearing diacritic set (č ć š ž đ) + ekavica
FIXTURE = [
    "Danas je lepo vreme, idemo na fudbal sa ekipom.",
    "Veštačka inteligencija menja svet, ali polako i oprezno.",
    "Đorđe je kupio ćevape, pljeskavice i šećer za ceo tim.",
    "Njegoš je napisao Gorski vijenac, remek-delo naše književnosti.",
    "Reč po reč, čovek nauči da čita između redova.",
] * 400  # enough repeats to train a small merge table


def test_train_and_roundtrip(tmp_path):
    out = tmp_path / "tok.json"
    train_bpe(iter(FIXTURE), vocab_size=500, out_path=str(out))
    tok = load_tokenizer(str(out))
    # byte-level BPE is LOSSLESS: decode(encode(s)) == s exactly, diacritics included
    for s in ["Danas je lepo vreme.", "Đorđe voli ćevape, čaj, šećer i žohare."]:
        assert tok.decode(tok.encode(s).ids) == s


def test_diacritics_preserved(tmp_path):
    out = tmp_path / "tok.json"
    train_bpe(iter(FIXTURE), vocab_size=500, out_path=str(out))
    tok = load_tokenizer(str(out))
    s = "č ć š ž đ Č Ć Š Ž Đ"
    assert tok.decode(tok.encode(s).ids) == s


def test_special_tokens_reserved(tmp_path):
    """Chat/SFT special tokens must exist with stable ids."""
    out = tmp_path / "tok.json"
    train_bpe(iter(FIXTURE), vocab_size=500, out_path=str(out))
    tok = load_tokenizer(str(out))
    vocab = tok.get_vocab()
    for st in SPECIAL_TOKENS:
        assert st in vocab, f"special token {st} not in vocab"
        assert tok.token_to_id(st) is not None
    # a special token encodes to exactly ONE id (not split into pieces)
    for st in SPECIAL_TOKENS:
        assert len(tok.encode(st, add_special_tokens=False).ids) == 1, f"{st} was split"


def test_no_unk(tmp_path):
    """Byte-level → every byte is representable, so arbitrary input never yields UNK / never crashes."""
    out = tmp_path / "tok.json"
    train_bpe(iter(FIXTURE), vocab_size=500, out_path=str(out))
    tok = load_tokenizer(str(out))
    weird = "emoji 🦅 卂 mixed ẞ tab\tand newline\n"
    assert tok.decode(tok.encode(weird).ids) == weird
