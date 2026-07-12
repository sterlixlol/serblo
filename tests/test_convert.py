"""Tests for serblo.generate.convert: §7.9 transport converter (PRD §7.9).

Runs under pytest, or standalone: `python tests/test_convert.py`.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.generate.convert import extract_docs, extract_pairs, looks_serbian, pick_payload_doc

DOC = "Ej ljudi jel neko gledao utakmicu sinoć? Kakav promašaj u 89. bre."
ESSAY = ("Kulturno nasleđe je temeljni stub identiteta jer se kroz njega prenose "
         "vrednosti i priče koje su oblikovale zajednicu tokom mnogo generacija.")


def test_happy_single_doc():
    docs, issues = extract_docs(f"<<<DOC>>>\n{DOC}\n<<<END>>>")
    assert docs == [DOC]
    assert issues == []


def test_chatter_outside_dropped():
    raw = f"Evo teksta koji ste tražili:\n<<<DOC>>>\n{DOC}\n<<<END>>>\nNadam se da pomaže!"
    docs, issues = extract_docs(raw)
    assert docs == [DOC]
    assert issues == []


def test_multiple_docs_per_payload():
    raw = "<<<DOC>>>prvi tekst ovde<<<END>>>\n<<<DOC>>>drugi tekst ovde<<<END>>>"
    docs, issues = extract_docs(raw)
    assert docs == ["prvi tekst ovde", "drugi tekst ovde"]
    assert issues == []


def test_markdown_fences_stripped():
    raw = f"```text\n<<<DOC>>>\n{DOC}\n<<<END>>>\n```"
    docs, issues = extract_docs(raw)
    assert docs == [DOC]
    assert issues == []


def test_inline_sentinels_glued_to_text():
    docs, _ = extract_docs("<<<DOC>>>Prvi red\ndrugi red<<<END>>>")
    assert docs == ["Prvi red\ndrugi red"]


def test_tolerant_sentinel_spelling():
    docs, _ = extract_docs(f"<<< doc >>>{DOC}<<<end>>>")   # inner spaces + lowercase
    assert docs == [DOC]


def test_dropped_bracket_sentinel_live_mimo():
    # regression from live MiMo v2.5 output (smoke test): opener came back as
    # "<<<DOC>>" (one closing bracket lost). Strict >>> would drop the whole doc.
    docs, issues = extract_docs(f"<<<DOC>>{DOC}<<<END>>>")
    assert docs == [DOC]
    assert issues == []


def test_confusable_bracket_sentinel_live_mimo():
    # regression from live MiMo v2.5 output (driver run): closer came back
    # as "◄◄END►►" (angle brackets rendered as arrow glyphs); the doc was complete
    # but strict matching stranded it as unclosed_doc.
    docs, issues = extract_docs(f"<<<DOC>>>\n{DOC}\n◄◄END►►")
    assert docs == [DOC]
    assert issues == []


def test_decorated_solo_line_closers_live_mimo():
    # regression from live MiMo v2.5 output (350-batch): closers came back
    # as decorated solo lines with mangled/absent bracket runs.
    for closer in (">>>END", ">>>END>>>", "***END>>>"):
        docs, issues = extract_docs(f"<<<DOC>>>\n{DOC}\n{closer}\n")
        assert docs == [DOC], f"closer {closer!r} not recognized"
        assert issues == []


def test_solo_line_needs_decoration_and_keyword():
    # organic content must NOT trigger tier-2: bare END line, chat speaker-label,
    # divider line, keyword-less bracket run.
    docs, issues = extract_docs("<<<DOC>>>prvi deo\nEND\nA:\n---\ndrugi deo<<<END>>>")
    assert docs == ["prvi deo\nEND\nA:\n---\ndrugi deo"]
    assert issues == []
    docs2, issues2 = extract_docs("<<<DOC>>>tekst pa samo zagrade\n>>>\n")
    assert docs2 == [] and "unclosed_doc" in issues2   # bare ">>>" stays unrecoverable


def test_stuttered_open_implicitly_closes():
    docs, issues = extract_docs("<<<DOC>>>prvi deo<<<DOC>>>drugi deo<<<END>>>")
    assert docs == ["prvi deo", "drugi deo"]
    assert "implicit_close" in issues


def test_unclosed_skipped_by_default_salvaged_on_flag():
    raw = "<<<DOC>>>tekst koji je presečen na pola rečeni"
    docs, issues = extract_docs(raw)
    assert docs == [] and "unclosed_doc" in issues
    docs2, issues2 = extract_docs(raw, salvage_unclosed=True)
    assert docs2 == ["tekst koji je presečen na pola rečeni"]
    assert "salvaged_unclosed" in issues2


def test_salvage_strips_mangled_closers_live_mimo():
    # regressions from the live 350-batch: unclosed docs whose closer MiMo corrupted.
    # salvage=True must recover the prose and strip the junk; without it, they stay dropped.
    cases = {
        f"<<<DOC>>>{DOC} jbg>>>":            f"{DOC} jbg",          # bare bracket run
        f"<<<DOC>>{DOC} lolpeace>>>END":     f"{DOC} lolpeace",     # END glued after text
        f"<<<DOC>>{DOC} lol<END>":           f"{DOC} lol",          # single-bracket END
        f"<<<DOC>>>{DOC} drugo.>>>END>>>":   f"{DOC} drugo.",       # keeps sentence period
        f"<<<DOC>>{DOC} ne laze.]]":         f"{DOC} ne laze.",     # square-bracket closer, keeps '.'
    }
    for raw, want in cases.items():
        got, _ = extract_docs(raw, salvage_unclosed=True)
        assert got == [want], f"{raw!r} -> {got!r}, wanted [{want!r}]"
        assert extract_docs(raw)[0] == [], f"{raw!r} should stay dropped without salvage"


def test_salvage_keeps_emoticon_before_marker():
    # ":/" emoticon must survive: the tail stripper's bracket class excludes ':' and '/'
    got, _ = extract_docs(f"<<<DOC>>>{DOC} dugo:/END>>>", salvage_unclosed=True)
    assert got == [f"{DOC} dugo:/"]


def test_salvage_no_closer_keeps_full_text():
    got, issues = extract_docs(f"<<<DOC>>>{DOC} kraj recenice.", salvage_unclosed=True)
    assert got == [f"{DOC} kraj recenice."]
    assert "salvaged_unclosed" in issues


def test_salvage_does_not_touch_clean_doc():
    raw = f"<<<DOC>>>\n{DOC}\n<<<END>>>"
    assert extract_docs(raw) == extract_docs(raw, salvage_unclosed=True)


def test_markdown_bold_and_headers_stripped():
    # MiMo leaks markdown into prose (measured ~2.9% '*' tokens); strip markup, keep text.
    docs, _ = extract_docs("<<<DOC>>>**Marko**: ćao ljudi\n### Naslov\n* prva stavka<<<END>>>")
    assert docs == ["Marko: ćao ljudi\nNaslov\nprva stavka"]


def test_dash_normalization():
    # MiMo's em-dash fingerprint (72x real) is an orthography swap -> normalize to "-"
    docs, _ = extract_docs("<<<DOC>>>Nasleđe je važno — ono čuva identitet — i traje kroz vekove, "
                           "od 1918—1941 pa sve do danas, u svim krajevima naše zemlje<<<END>>>")
    assert docs == ["Nasleđe je važno - ono čuva identitet - i traje kroz vekove, "
                    "od 1918-1941 pa sve do danas, u svim krajevima naše zemlje"]


def test_dash_dialogue_marker_keeps_line_start():
    docs, _ = extract_docs("<<<DOC>>>Prva rečenica koja ima dovoljno reči za prag da prođe je ovde.\n"
                           "— Rekao je da dolazi sutra ujutru kod nas<<<END>>>")
    assert docs[0].split("\n")[1].startswith("- Rekao je")


def test_plain_hyphen_untouched():
    docs, _ = extract_docs("<<<DOC>>>ovo je neki normalan tekst sa crticom - i to sasvim običnom - "
                           "koja mora da ostane potpuno ista kao što je bila<<<END>>>")
    assert " - i to sasvim običnom - " in docs[0]


def test_markdown_strip_keeps_plain_asterisk():
    # a lone '*' that isn't markup (e.g. "2 * 3") must survive
    docs, _ = extract_docs("<<<DOC>>>ma bre 2 * 3 je 6<<<END>>>")
    assert docs == ["ma bre 2 * 3 je 6"]


def test_empty_doc_dropped_with_issue():
    docs, issues = extract_docs("<<<DOC>>>\n   \n<<<END>>>")
    assert docs == [] and "empty_doc" in issues


def test_no_sentinels_skipped():
    docs, issues = extract_docs("Model je ignorisao format i samo napisao tekst.")
    assert docs == [] and issues == ["no_sentinels"]


def test_stray_qa_marks_stripped_from_prose():
    docs, issues = extract_docs("<<<DOC>>>deo jedan <<<Q>>>deo dva<<<END>>>")
    assert docs == ["deo jedan deo dva"]        # sentinel never reaches the corpus
    assert "stray_q" in issues


def test_no_sentinel_survives_into_docs():
    raw = "<<<DOC>>>a<<<Q>>>b<<<A>>>c<<<END>>><<<DOC>>>d<<<END>>>"
    docs, _ = extract_docs(raw)
    assert all("<<<" not in d and ">>>" not in d for d in docs)


def test_crlf_and_blankline_collapse():
    docs, _ = extract_docs("<<<DOC>>>red jedan\r\n\r\n\r\n\r\nred dva<<<END>>>")
    assert docs == ["red jedan\n\nred dva"]


def test_looks_serbian_gate():
    assert looks_serbian(ESSAY)                                        # formal Serbian
    assert looks_serbian("ma bre jbg ne znam sta da ti kazem ali to je to i nema dalje brate")  # sloppy, no diacritics
    assert looks_serbian("jel neko igrao onaj novi lol update? meni je laggy ali ok je i dalje se igra")  # English code-switch
    assert not looks_serbian("This is an instruction to generate text. I'll follow the format requirements exactly:")
    assert not looks_serbian("at the start,")
    assert not looks_serbian("...")


def test_pick_payload_doc_reasoning_leak_live_shape():
    # regression: live essay-0000273, reasoning leaked + quoted sentinels ->
    # extraction yields [draft, junk, junk, final, junk, final]; policy must return final.
    draft = ESSAY + " Ovo je prva verzija teksta koja govori o nasleđu i njegovom značaju za sve nas."
    final = ESSAY + " Ovo je konačna, doterana verzija teksta o nasleđu i njegovom značaju za buduće generacije."
    docs = [draft, "at the start,", "...", final, "at start / 2.", final]
    doc, issues = pick_payload_doc(docs)
    assert doc == final
    assert issues.count("dropped_junk_doc") == 3
    assert issues.count("dropped_dup_doc") == 1
    assert "kept_last_of_2" in issues


def test_pick_payload_doc_stuttered_copies():
    # regression: live conv-0000060, the same chat 4x -> exactly one survives, no marker
    chat = DOC + " Ma pusti to, bilo je i gorih utakmica ali ova je bila najgora do sada."
    doc, issues = pick_payload_doc([chat, chat, chat, chat])
    assert doc == chat
    assert issues.count("dropped_dup_doc") == 3
    assert not any(i.startswith("kept_last") for i in issues)          # single survivor = clean


def test_pick_payload_doc_single_clean_untouched():
    doc, issues = pick_payload_doc([DOC + " I još jedna rečenica da pređemo prag od petnaest reči, bre."])
    assert doc is not None and issues == []


def test_pick_payload_doc_all_junk_none():
    doc, issues = pick_payload_doc(["Let me check the requirements:", "..."])
    assert doc is None and issues.count("dropped_junk_doc") == 2


def test_pairs_happy_and_multiple():
    raw = ("<<<Q>>>Koliko je 2+2?<<<A>>>Četiri.<<<END>>>\n"
           "<<<Q>>>Glavni grad Srbije?<<<A>>>Beograd.<<<END>>>")
    pairs, issues = extract_pairs(raw)
    assert pairs == [("Koliko je 2+2?", "Četiri."), ("Glavni grad Srbije?", "Beograd.")]
    assert issues == []


def test_pairs_forgotten_end_salvaged():
    raw = "<<<Q>>>Prvo pitanje?<<<A>>>Prvi odgovor.<<<Q>>>Drugo pitanje?<<<A>>>Drugi odgovor.<<<END>>>"
    pairs, issues = extract_pairs(raw)
    assert pairs == [("Prvo pitanje?", "Prvi odgovor."), ("Drugo pitanje?", "Drugi odgovor.")]
    assert "q_reopens_pair" in issues


def test_pairs_q_without_a_dropped():
    pairs, issues = extract_pairs("<<<Q>>>Pitanje bez odgovora?<<<END>>>")
    assert pairs == [] and "q_without_a" in issues


def test_pairs_truncated_at_eof_dropped():
    pairs, issues = extract_pairs("<<<Q>>>Pitanje?<<<A>>>Odgovor presečen na po")
    assert pairs == [] and "unclosed_pair" in issues


def test_pairs_stray_marks_logged():
    pairs, issues = extract_pairs("<<<A>>>odgovor niotkuda<<<END>>>")
    assert pairs == []
    assert "stray_a" in issues and "stray_end" in issues


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in fns:
        try:
            fn(); print(f"PASS {name}")
        except AssertionError as e:
            fails += 1; print(f"FAIL {name}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    raise SystemExit(1 if fails else 0)
