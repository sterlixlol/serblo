"""Text normalization. PRD §7.3 stage 2.

The ordering here is load-bearing for the whole corpus:
  NFC  ->  transliterate Cyrillic -> Latin (Serbian gajica)  ->  (dedup happens later, in M2)

Transliterating to a single script BEFORE dedup is what collapses the same article
appearing once in Cyrillic and once in Latin. Diacritics (č ć š ž đ) are load-bearing
Serbian orthography and are NEVER stripped.
"""
import unicodedata

import cyrtranslit

# Serbian Latin diacritics: must survive normalization intact.
BCS_DIACRITICS = frozenset("čćšžđČĆŠŽĐ")


def normalize_text(text):
    """NFC-normalize, then Cyrillic -> Latin (gajica). Idempotent on Latin input.

    Returns "" for None/empty. Diacritics preserved; no Cyrillic should remain
    for well-formed Serbian/Croatian/Bosnian input.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = cyrtranslit.to_latin(text, "sr")
    # cyrtranslit can, in edge cases, emit decomposed sequences, so re-compose.
    return unicodedata.normalize("NFC", text)


def has_cyrillic(text):
    """True if any Cyrillic codepoint remains (blocks U+0400-052F). Post-normalize
    residual Cyrillic flags non-Serbian text (e.g. Russian/Macedonian misfiled)."""
    return any(0x0400 <= ord(ch) <= 0x052F for ch in text)


def diacritic_ratio(text):
    """Fraction of alphabetic chars that are BCS diacritics: a corpus-health signal.
    A normalized BCS corpus with ~0 here means something stripped the diacritics."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c in BCS_DIACRITICS) / len(alpha)
