"""OpenSubtitles BCS acquisition. PRD M1, conversational backbone.

Streams OPUS OpenSubtitles v2024 monolingual plain text (one subtitle fragment per
line) directly from CSC object storage, gunzipping on the fly (no full download).
Raw subtitle text carries the CP1250-as-CP1251 mojibake (see pipeline.encoding),
so the runner repairs before normalize.

Caveat (PRD §7.6): subtitles are *translated* dialogue -> some translationese. This
is volume for the conversational register, balanced downstream by original sources
(forum, human pairs, casual synthetic).
"""
import gzip
import urllib.request

BASE = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono"
BCS_LANGS = ["sr", "hr", "bs"]
_UA = "Mozilla/5.0 (X11; Linux x86_64) Serblo/0.1"


def stream_lines(lang, skip=0):
    """Yield decoded text lines from `<lang>.txt.gz`, skipping the first `skip`.

    Streaming order is the file's byte order (deterministic), which is what makes
    skip-based resume correct. skip is O(skip), fine at slice scale.
    """
    url = f"{BASE}/{lang}.txt.gz"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    resp = urllib.request.urlopen(req, timeout=60)
    with gzip.GzipFile(fileobj=resp) as gz:
        i = 0
        for raw in gz:
            if i < skip:
                i += 1
                continue
            i += 1
            yield raw.decode("utf-8", errors="replace").rstrip("\n")
