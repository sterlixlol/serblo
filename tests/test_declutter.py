"""Tests for serblo.pipeline.declutter: chrome/boilerplate stripper (PRD §8 M2).

Runs under pytest, or standalone: `python tests/test_declutter.py`.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.pipeline.declutter import declutter, survival


def test_strips_citation_markers_keeps_prose():
    t = "Nuklearni holokaust je izraz koji se odnosi na uništenje[1] civilizacije[12] usled rata."
    out = declutter(t)
    assert "[1]" not in out and "[12]" not in out
    assert out == "Nuklearni holokaust je izraz koji se odnosi na uništenje civilizacije usled rata."


def test_drops_wiki_edit_and_toc():
    t = "Sadržaj\nPrincip rada [uredi]\nOvo je stvarni tekst o principu rada uređaja koji radi."
    out = declutter(t)
    assert "Sadržaj" not in out.split("\n")
    assert "[uredi]" not in out
    assert "Princip rada" in out                       # section title kept, marker stripped
    assert "Ovo je stvarni tekst" in out


def test_drops_pipe_template_and_maintenance_phrase():
    t = ("Centralni komitet\n"
         "|Ovaj članak ili neka od njegovih sekcija nije dovoljno potkrijepljena izvorima.\n"
         "| Pomozite Wikipediji tako što ćete dodati izvore.\n"
         "Ovo je pravi sadržaj članka o komitetu i njegovom radu tokom istorije.")
    out = declutter(t)
    assert "|" not in out
    assert "Pomozite Wikipediji" not in out
    assert "nije dovoljno" not in out
    assert "Ovo je pravi sadržaj" in out
    assert out.startswith("Centralni komitet")


def test_strips_urls():
    t = "Pogledajte na http://sharecash.org/index.php?ref=1 ili www.europa.rs za više informacija."
    out = declutter(t)
    assert "http" not in out and "www." not in out
    assert "Pogledajte na" in out and "za više informacija" in out


def test_drops_reddit_chrome_lines():
    t = "u/jelena_99\n3h ago\nma bre najviše me živcira ovo danas ljudi ozbiljno\nUpvote\n4\nReply\nShare"
    out = declutter(t)
    lines = out.split("\n")
    assert any("živcira" in ln for ln in lines)
    assert not any(ln.strip() in ("Upvote", "Reply", "Share", "u/jelena_99") for ln in lines)


def test_drops_forum_editlog_line():
    t = "Ovo je pravi tekst poruke koju je korisnik napisao na forumu o nekoj temi.\n[Ovu poruku je menjao tdjokic dana 03.05.2006. u 04:20]"
    out = declutter(t)
    assert "menjao" not in out
    assert "Ovo je pravi tekst" in out


def test_clean_prose_untouched():
    t = ("Ma bre ljudi ova današnja muzika je nešto strašno, ne znam šta bih rekao, sve neki "
         "žestok bit a pesme nigde. Ne sluša to niko normalan više, ozbiljno vam kažem.")
    assert declutter(t) == t                            # no chrome -> byte-identical
    assert survival(t, declutter(t)) == 1.0


def test_survival_ratio():
    orig = "jedan dva tri četiri pet šest sedam osam devet deset"
    assert survival(orig, "jedan dva tri četiri pet") == 0.5
    assert survival("", "") == 0.0


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
