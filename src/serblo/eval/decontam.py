"""Decontaminate the training corpus against the eval/probe set (PRD M5 stage).

Removes any training document that shares an n-gram with a held-out eval/probe item, so a
"good" perplexity or pass-rate can't come from the model having memorised the test. Standard
GPT-3 / The Pile approach: word-level 13-grams, drop the whole training doc on any overlap
(simpler and safer for pretraining than span-masking). Direction is fixed: we filter TRAIN,
never the eval set (the eval set is ground truth).

Memory: the eval index is a set of n-grams from a few hundred short prompts, kilobytes. The
training corpus is STREAMED one doc at a time, so this scales to the full 30M-doc backbone with
a flat footprint. Resumable per shard: a crash re-does at most one shard.

Matching normalisation is for COMPARISON only (it never touches stored text): casefold +
collapse whitespace + strip surrounding punctuation from each word. Diacritics are KEPT: in
BCS č ć š ž đ are load-bearing, so "cetiri" must not match "četiri".

Short eval items (< n words) collapse to a single whole-item gram, so a 5-word probe prompt is
still matched exactly rather than silently ignored.
"""
import re
import unicodedata

DEFAULT_N = 13

# A "word" for matching: keep letters (incl. BCS diacritics) and digits, drop surrounding punctuation.
_WORD = re.compile(r"\w+", re.UNICODE)


def normalize_for_match(text):
    """Lowercase word tokens for overlap comparison (NOT for storage, diacritics preserved)."""
    text = unicodedata.normalize("NFC", text)
    return _WORD.findall(text.casefold())


def ngrams(words, n):
    """Set of word n-grams. Texts shorter than n collapse to one whole-text gram (never empty)."""
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def build_eval_index(eval_texts, n=DEFAULT_N):
    """Union of n-grams across every eval/probe item -> the contamination fingerprint set."""
    index = set()
    for t in eval_texts:
        index |= ngrams(normalize_for_match(t), n)
    return index


def contamination(text, eval_index, n=DEFAULT_N):
    """Return the first eval n-gram this doc collides with, or None if clean."""
    for g in ngrams(normalize_for_match(text), n):
        if g in eval_index:
            return g
    return None
