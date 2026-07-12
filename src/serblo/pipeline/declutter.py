"""Chrome / boilerplate stripper for real scraped text. M2 cleaning (PRD §8).

A FineWeb-2 slice carries wiki/forum/UI chrome: TOC dumps ("Sadržaj",
"[uredi]"), maintenance-template boxes (pipe-prefixed lines), citation markers ("[1]"),
edit-logs, nav, spam URLs. The chrome is SHARED boilerplate, so it homogenizes the corpus
(any two chromed docs look alike through it) and teaches the model UI structure, not
Serbian. But most flagged docs are GOOD prose lightly contaminated, so this is SURGICAL:
strip chrome lines + inline markers, keep the prose; the caller drops a doc only if
cleaning removed most of it.

Pure text->text. Runs before dedup so near-dup detection sees clean prose (order matters,
like transliteration). Diacritics and ordinary punctuation are preserved.
"""
import re

# Inline markers removed in place (surrounding prose kept). Order matters: bracketed
# keyword tags before the bare [N] citation rule.
_INLINE = [
    re.compile(r"\[\s*(?:uredi|уреди|edit|izvor|citation needed|traži se izvor)[^\]]*\]", re.I),
    re.compile(r"https?://\S+"),
    re.compile(r"\bwww\.\S+"),
    re.compile(r"\[\d+\]"),                                    # [1] [12] citations
]

# Whole-line chrome: the line is dropped.
_CHROME_LINE = [
    re.compile(r"^\s*\|"),                                     # wiki template / table rows
    re.compile(r"^\s*Sadržaj\s*:?\s*$", re.I),                 # TOC header
    re.compile(r"^\s*\[[^\]]*(?:menjao|mijenjao|obrisao|izmenio|uredio|edited|edited by)[^\]]*\]\s*$", re.I),
    re.compile(r"^[\W_]{3,}$"),                                # separators / ascii art
    re.compile(r"^\s*(?:Preuzeto sa|Kategorij[ae]?\s*:|Navigacij|Glavni meni|Lични алати|Pretraga|Alatke|Izvori|Reference)\s*:?\s*$", re.I),
    re.compile(r"^\s*(?:Upvote|Downvote|Reply|Share|Report|Award|Give Award)\s*$", re.I),  # reddit
    re.compile(r"^\s*u/\S+\s*$"),                              # reddit usernames
    re.compile(r"^\s*•?\s*\d+\s*(?:points?|comments?|poena|komentara)\s*$", re.I),
    re.compile(r"^\s*•?\s*\d+\s*[hmdgh]\w*\s+(?:ago|ranije)\s*$", re.I),
]

# High-confidence boilerplate phrases (wiki maintenance templates, forum edit-logs) that
# appear mid-line as normal-looking Serbian but only ever occur as chrome; drop the line.
_CHROME_PHRASE = re.compile(
    r"Pomozite (?:Wikipediji|Vikipediji)"
    r"|nije dovoljno (?:potkrijepljen|potkrepljen)"
    r"|Ovaj članak (?:je (?:patrljak|klica)|ili neka od njegovih)"
    r"|možete pomoći Wikipediji"
    r"|Ovu poruku je (?:menjao|mijenjao|uredio)"
    r"|\b[Ii]zmenjeno \d+ puta\b", re.I)


def declutter(text):
    """Strip chrome lines + inline markers; return cleaned prose."""
    kept = []
    for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _CHROME_PHRASE.search(ln) or any(p.search(ln) for p in _CHROME_LINE):
            continue
        for p in _INLINE:
            ln = p.sub("", ln)
        kept.append(ln)
    out = "\n".join(kept)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" +([,.!?;:])", r"\1", out)                 # heal " ." left by a stripped [1]
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def survival(original, cleaned):
    """Fraction of the original word count that survived cleaning. Low = doc was mostly chrome."""
    ow = len(original.split())
    return len(cleaned.split()) / ow if ow else 0.0
