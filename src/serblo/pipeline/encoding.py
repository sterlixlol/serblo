"""Mojibake repair for raw sources, runs BEFORE normalize (PRD §7.3).

OpenSubtitles Serbian is riddled with one specific corruption: CP1250 (Central-
European Latin) bytes decoded as CP1251 (Cyrillic), which turns š ž č ć đ into
Cyrillic lookalikes (љ ћ и ж р). Deterministic and reversible, but it must be
applied ONLY to corrupted text: round-tripping a genuine Cyrillic line through
cp1251→cp1250 destroys it.

Detection fingerprint: a whitespace token containing BOTH Latin and Cyrillic
letters. Organic Serbian never mixes scripts *within a word*, so a mixed-script
token is the mojibake signature. (Code-switching like "Здраво Peter" mixes at the
line level but not the token level, so it is correctly left alone.)
"""


def _is_latin(c):
    return "a" <= c <= "z" or "A" <= c <= "Z"


def _is_cyrillic(c):
    return 0x0400 <= ord(c) <= 0x052F


def has_mixed_script_token(text):
    """True if any token mixes Latin and Cyrillic letters: the mojibake signature."""
    for tok in text.split():
        has_lat = has_cyr = False
        for c in tok:
            if _is_latin(c):
                has_lat = True
            elif _is_cyrillic(c):
                has_cyr = True
            if has_lat and has_cyr:
                return True
    return False


def fix_mojibake(text):
    """Repair CP1250-as-CP1251 corruption if detected; else return text unchanged.
    Safe on clean Latin and genuine Cyrillic (both lack mixed-script tokens)."""
    if not text or not has_mixed_script_token(text):
        return text
    try:
        return text.encode("cp1251").decode("cp1250")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text          # un-repairable line, leave for downstream filtering


def fix_latin1_mojibake(text):
    """Repair the *other* OpenSubtitles corruption: CP1250 bytes decoded as Latin-1, which turns
    č ć đ into è æ ð (the `bs` subtitles' "senèenjem" for "senčenjem"). Self-gating: text carrying
    genuine (non-Latin-1) diacritics like š/č/ž fails the latin-1 encode and is returned unchanged,
    so clean text is never corrupted; plain ASCII round-trips identically. Applied per line."""
    if not text:
        return text
    try:
        return text.encode("latin-1").decode("cp1250")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def repair_text(text):
    """Apply both OpenSubtitles mojibake repairs. Cyrillic-variant is line-level (gated by a
    mixed-script token). Latin-1-variant is PER WORD: a single line often mixes a correctly-encoded
    word (real š/č/ž → latin-1 encode fails → kept) with a corrupted one ("èinova" → all latin-1 →
    fixed to "činova"), so whole-line repair would leave the corrupted words untouched."""
    out = []
    for line in text.split("\n"):
        line = fix_mojibake(line)
        line = " ".join(fix_latin1_mojibake(w) for w in line.split(" "))
        out.append(line)
    return "\n".join(out)
