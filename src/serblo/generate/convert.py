"""§7.9 transport converter: sentinel-wrapped model output → clean text (PRD §7.9).

The generation side wraps every payload in rare ASCII sentinels (``<<<DOC>>> … <<<END>>>``
for pretrain prose; ``<<<Q>>> / <<<A>>> / <<<END>>>`` for instruction pairs). This is the
receiving half: a deterministic, forgiving parser that slices payloads out of whatever a
model actually returned: preamble chatter, markdown fences, sentinels glued to text,
stuttered openers, forgotten closers, truncated tails, and never crashes on malformed
output: it skips and reports issue codes. Sentinels are stripped HERE, before anything
reaches the tokenizer (PRD §7.2).

Pure text→text, no I/O. Each backend's collect runner feeds raw outputs through
``extract_docs()`` / ``extract_pairs()``, then ``normalize_text()``, then shards, the
same shape as the M1 runners. A payload may carry SEVERAL docs (batching multiple docs
per request is a free-tier quota lever), so both extractors return lists.
"""
import re

# Tolerant sentinel match, two tiers (matching-only looseness; doc text never mutated):
# 1. bracket-wrapped anywhere: >=2 brackets per side incl. CONFUSABLES; live MiMo
#    emitted "<<<DOC>>" (lost bracket) and "◄◄END►►" (arrow glyphs).
# 2. decorated solo line: a line that is ONLY punct-decoration + DOC/END; live MiMo
#    closed docs with ">>>END", ">>>END>>>", "***END>>>". Requires a >=2-char punct run
#    on at least one side, so organic "END" alone, chat speaker-labels ("A:"), or "---"
#    dividers can't false-positive. DOC/END only (Q/A too collision-prone with chat).
_MARK = re.compile(
    r"[<‹«◄❮]{2,}\s*(DOC|END|Q|A)\s*[>›»►❯]{2,}"
    r"|^[ \t]*[^\w\s]{2,}[ \t]*(DOC|END)[ \t]*[^\w\s]*[ \t]*$"
    r"|^[ \t]*[^\w\s]*[ \t]*(DOC|END)[ \t]*[^\w\s]{2,}[ \t]*$",
    re.IGNORECASE | re.MULTILINE)
_FENCE_LINE = re.compile(r"^[ \t]*```[\w-]*[ \t]*$", re.MULTILINE)   # fence-only lines


def _events(raw):
    """Split raw into an ordered stream of ("text", chunk) / ("mark", KIND) events."""
    pos = 0
    for m in _MARK.finditer(raw):
        if m.start() > pos:
            yield "text", raw[pos:m.start()]
        kind = next(g for g in m.groups() if g)      # whichever tier matched
        yield "mark", kind.upper()
        pos = m.end()
    if pos < len(raw):
        yield "text", raw[pos:]


# MiMo wraps names/emphasis in markdown (**Ime**, ### Naslov), leaking '*'/'#' tokens into
# the corpus (measured ~2.9% of tokens in one batch). Strip the markup, keep inner text.
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_HEADER = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
_MD_BULLET = re.compile(r"^[ \t]*\*[ \t]+", re.MULTILINE)     # "* item" list bullets -> plain


def _strip_markdown(text):
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BULLET.sub("", text)
    return text


# Dash normalization: MiMo renders clause dashes as em-dashes at 72x the real rate,
# while its TOTAL dash rate matches real (12.15 vs 12.58 per 1k): an orthography swap,
# not a style excess. Real Serbian web text writes "-", so normalize em- and en-dashes
# to it: tight between word chars (year ranges get a hyphen: 1918-1941), else spaced.
_DASH_TIGHT = re.compile(r"(?<=\w)[—–](?=\w)")
_DASH_LOOSE = re.compile(r"[ \t]*[—–][ \t]*")
_DASH_LINESTART = re.compile(r"(?m)^ - ")                 # dialogue-marker dash keeps line start


def _normalize_dashes(text):
    text = _DASH_TIGHT.sub("-", text)
    text = _DASH_LOOSE.sub(" - ", text)
    return _DASH_LINESTART.sub("- ", text)


def _clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_markdown(text)
    text = _normalize_dashes(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Trailing mangled-closer stripper for SALVAGE only: when a doc has no recognized
# close marker, MiMo often still left a corrupted one glued to the tail: "jbg>>>",
# "lolpeace>>>END", "lol<END>", "drugo.>>>END>>>", ".]]". Bracket class is tight
# (brackets + '*' only, no '.'/':'/'/') so sentence periods and ":/" emoticons survive.
_BRK = r"<>»«›‹◄►❯❮\[\]*"
_TAIL_JUNK = re.compile(
    rf"\s*(?:[{_BRK}]+\s*(?:END|DOC)\b\s*[{_BRK}]*"     # brackets + END/DOC (+ maybe brackets)
    rf"|[{_BRK}]*\s*(?:END|DOC)\b\s*[{_BRK}]+"          # END/DOC + brackets
    rf"|[{_BRK}]{{2,}})\s*$",                            # or a bare bracket run (>=2)
    re.IGNORECASE)


def _strip_tail_junk(text):
    return _TAIL_JUNK.sub("", text).rstrip()


def extract_docs(raw, salvage_unclosed=False):
    """All ``<<<DOC>>>…<<<END>>>`` payloads in raw -> (docs, issues).

    Forgiving: chatter outside blocks is dropped; markdown fences stripped; a DOC opener
    while a block is open implicitly closes the previous one (stutter / forgotten END);
    stray Q/A marks inside prose are stripped and logged. EOF inside an open block is
    skipped unless salvage_unclosed=True; truncated generations usually end mid-sentence,
    so salvage only where volume matters more than a clean tail.
    """
    raw = _FENCE_LINE.sub("", raw or "")
    docs, issues, buf = [], [], None            # buf None = outside a block
    saw_mark = False

    def close(reason=None, salvage=False):
        nonlocal buf
        doc = _clean("".join(buf or []))         # buf never None here; keeps pyright quiet
        if salvage:
            doc = _strip_tail_junk(doc)          # drop a corrupted closer glued to the tail
        if doc:
            docs.append(doc)
        else:
            issues.append("empty_doc")
        if reason:
            issues.append(reason)
        buf = None

    for kind, val in _events(raw):
        if kind == "text":
            if buf is not None:
                buf.append(val)
            continue
        saw_mark = True
        if val == "DOC":
            if buf is not None:
                close("implicit_close")
            buf = []
        elif val == "END":
            if buf is not None:
                close()
            else:
                issues.append("stray_end")
        else:                                    # Q/A sentinel in prose mode
            issues.append(f"stray_{val.lower()}")
    if buf is not None:
        if salvage_unclosed:
            close("salvaged_unclosed", salvage=True)
        else:
            issues.append("unclosed_doc")
    if not saw_mark:
        issues.append("no_sentinels")
    return docs, issues


# --- payload policy: 1 prompt = 1 doc (current generation mode) -----------------------
# Live failure (essay-0000273): MiMo's reasoning leaked into the output and QUOTED the
# sentinels, so extraction yielded junk English fragments + the essay three times (draft,
# final, final). Conv payloads also stuttered identical copies (conv-0000060 ×4). Until we
# deliberately batch N docs per request, every multi-doc payload is a malfunction; this
# policy reduces it to the one intended text. Junk is caught by a Serbian-ness gate
# (leaked English reasoning has ~zero Serbian function words; dedup would never catch it).
_SR_STOP = frozenset(
    "je i u se da na za su od sa o a ali ili pa ni kao kad što sto koji koja koje bi će ce "
    "ne to ovo ono mi ti on ona smo ste nije ima nema bio bila bilo biti sam si mu ga ih "
    "im joj te me nas vas njih jer već vec još jos samo sve svi može moze".split())
# 'i'/'a'/'to'/'me'/'on' are English words too, so a pure SR-rate gate passes leaked English
# ("this is an instruction to…" hits i+to). So the gate is DISCRIMINATIVE: Serbian function
# words must also outnumber English ones.
_EN_STOP = frozenset(
    "the is are was were and of in that it for with as this at be or not have has will "
    "would should let check start end format text follow write requirements instruction "
    "generate exactly nothing outside".split())
_MIN_DOC_WORDS = 15
_MIN_SR_RATE = 0.08


def looks_serbian(text, min_rate=_MIN_SR_RATE):
    """Cheap BCS gate: Serbian function-word rate >= min_rate AND Serbian hits > English hits.

    Real BCS (even very_sloppy, diacritic-free, English-code-switched chat) passes easily;
    leaked English reasoning / markup junk fails one or both conditions."""
    words = re.findall(r"[a-zA-ZčćšžđČĆŠŽĐ]+", text.lower())
    if not words:
        return False
    sr = sum(1 for w in words if w in _SR_STOP)
    en = sum(1 for w in words if w in _EN_STOP)
    return sr / len(words) >= min_rate and sr > en


def pick_payload_doc(docs, min_words=_MIN_DOC_WORDS):
    """(final_doc | None, issues): collapse a payload's doc list to the ONE intended text.

    Order: junk gate (length + Serbian-ness) -> exact-dedup (order-preserving) -> keep the
    LAST survivor (generation order is draft -> critique -> final; last = the revision).
    """
    issues = []
    kept, seen = [], set()
    for d in docs:
        if len(d.split()) < min_words or not looks_serbian(d):
            issues.append("dropped_junk_doc")
            continue
        if d in seen:
            issues.append("dropped_dup_doc")
            continue
        seen.add(d)
        kept.append(d)
    if not kept:
        return None, issues
    if len(kept) > 1:
        issues.append(f"kept_last_of_{len(kept)}")
    return kept[-1], issues


def extract_pairs(raw):
    """All ``<<<Q>>>q<<<A>>>a<<<END>>>`` records in raw -> (pairs, issues).

    pairs = [(question, answer), ...]. A new Q while an answer is open salvages the
    finished-looking pair (models forget END between records); a Q without an A, or a
    pair truncated at EOF, is dropped and logged.
    """
    raw = _FENCE_LINE.sub("", raw or "")
    pairs, issues = [], []
    state, q_buf, a_buf = "out", [], []
    saw_mark = False

    def flush(ok):
        nonlocal state, q_buf, a_buf
        q, a = _clean("".join(q_buf)), _clean("".join(a_buf))
        if ok and q and a:
            pairs.append((q, a))
        elif ok:
            issues.append("empty_q_or_a")
        state, q_buf, a_buf = "out", [], []

    for kind, val in _events(raw):
        if kind == "text":
            if state == "q":
                q_buf.append(val)
            elif state == "a":
                a_buf.append(val)
            continue
        saw_mark = True
        if val == "Q":
            if state != "out":                   # forgot END between records
                issues.append("q_reopens_pair")
                flush(state == "a")              # salvage a complete pair, drop a half one
            state = "q"
        elif val == "A":
            if state == "q":
                state = "a"
            else:
                issues.append("stray_a")
        elif val == "END":
            if state == "a":
                flush(True)
            elif state == "q":
                issues.append("q_without_a")
                flush(False)
            else:
                issues.append("stray_end")
        else:                                    # DOC sentinel in pairs mode
            issues.append("stray_doc")
    if state != "out":
        issues.append("unclosed_pair")           # truncated at EOF -> drop
    if not saw_mark:
        issues.append("no_sentinels")
    return pairs, issues
