"""Reference-anchor pool for grounding: real-text style anchors (PRD §7.4, config §grounding).

Both essay generators converge on one essay skeleton without grounding (gate:
Flash-Lite self-BLEU-4 1.83× real, MiMo 1.51×). The treatment is injecting a REAL excerpt
per prompt as a style anchor (config's ``style_transfer`` mode): match the voice, write
different content. This module is the pure logic: excerpting and quality gates; the
runner (run_refpool) streams interim shards through it into a versioned pool file that
the manifest generator samples from.

Anchors are sentence-trimmed windows of ~60 to 140 words: long enough to carry voice/structure,
short enough to keep input tokens trivial and content-leak low (dedup catches echoes).
"""
import re

from serblo.generate.convert import looks_serbian

# wiki chrome / markup / links have no place in a style anchor (the real FineWeb-2 slice
# carries `uredi` buttons and table pipes)
_BAD = re.compile(r"uredi|http|www\.|\||\[|\]|==|<<|>>|\{|\}", re.IGNORECASE)
_SENT_END = (".", "!", "?")

MIN_WORDS = 60
MAX_WORDS = 140


def anchor_ok(text, min_words=MIN_WORDS):
    """Quality gate for a candidate excerpt: Serbian, clean, prose-like."""
    words = text.split()
    if len(words) < min_words:
        return False
    if _BAD.search(text):
        return False
    if not looks_serbian(text):
        return False
    digity = sum(1 for w in words if any(c.isdigit() for c in w))
    return digity / len(words) < 0.15                   # tables/lists masquerading as prose


def excerpt(text, rng, min_words=MIN_WORDS, max_words=MAX_WORDS):
    """A random contiguous window, trimmed toward sentence boundaries. None if unusable."""
    words = text.split()
    if len(words) < min_words:
        return None
    span = min(rng.randint(min_words, max_words), len(words))
    start = 0 if len(words) == span else rng.randrange(0, len(words) - span + 1)
    chunk = words[start:start + span]
    if start > 0:                                       # drop a leading partial sentence
        for i, w in enumerate(chunk[:20]):
            if w.endswith(_SENT_END):
                chunk = chunk[i + 1:]
                break
    for j in range(len(chunk) - 1, max(len(chunk) - 25, 0), -1):
        if chunk[j].endswith(_SENT_END):                # end on a sentence when possible
            chunk = chunk[:j + 1]
            break
    if len(chunk) < min_words - 15:
        return None
    return " ".join(chunk)


def make_anchor(doc_text, rng, register, source, seq):
    """excerpt + gate -> pool record, or None."""
    ex = excerpt(doc_text, rng)
    if ex is None or not anchor_ok(ex):
        return None
    return {"id": f"ref-{register[:5]}-{seq:06d}", "register": register,
            "source": source, "text": ex}
