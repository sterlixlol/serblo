"""Train the final 32k byte-level BPE for Serblo (M4).

Byte-level BPE = lossless round-trip + no UNK (every byte is in the base alphabet). Special
tokens are reserved up front so the later chat/SFT run can use ChatML without retraining the
tokenizer. The artifact is written atomically (.tmp -> os.replace).
"""
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Reserved for the SFT/chat phase (ChatML-style). <|endoftext|> doubles as the pretrain document
# separator / EOS. Kept minimal + stable; these ids must never move once training starts.
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]


def train_bpe(corpus_iter, vocab_size=32000, out_path="data/tokenizer/tokenizer.json",
              special_tokens=None):
    """Train a byte-level BPE over an iterable of strings and save tokenizer.json atomically.

    corpus_iter : iterable of str (documents).
    Returns the trained Tokenizer.
    """
    special = list(special_tokens) if special_tokens is not None else SPECIAL_TOKENS
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # all 256 bytes -> never UNK
        show_progress=False,
    )
    tok.train_from_iterator(corpus_iter, trainer=trainer)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = out_path + ".tmp"
    tok.save(tmp)
    os.replace(tmp, out_path)                                   # atomic write
    return tok
