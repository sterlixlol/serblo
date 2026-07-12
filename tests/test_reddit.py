"""Tests for serblo.acquire.reddit.parse_thread: extracting Serbian text from a
WebGrab reddit-scrape markdown blob. Fixture is a real (trimmed) r/srbija scrape.

Runs under pytest, or standalone: `python tests/test_reddit.py`.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.acquire.reddit import parse_thread

FIXTURE = """# Subota 13h. Eh, moj Beograde...

**r/srbija** | u/Nemanja_Torbica | 218 points (94% upvoted) | 2026-02-28

Gužva u Beogradu.

**Link:** https://i.redd.it/icj4lenax9mg1.jpeg

---

## Comments

- **u/ThatMaintenance8366** (35 pts, 2026-02-28): Izadjosmo I mi gušteri na sunce 😅

Jeste bila gužva, ja se jedva parkirah u Zemunu.
  - **u/Nemanja_Torbica** (8 pts, 2026-02-28): Pozdrav za Zemun!
- **u/AutoModerator** (1 pts, 2026-02-28): Molimo vas da pratite pravila subreddita.
- **u/notoriousbgone** (134 pts, 2026-02-28): Neće ljudi metroom da se voze nego svojim kolima.
- **u/x** (2 pts, 2026-02-28): [deleted]
"""

OUT = parse_thread(FIXTURE)


def test_title_and_body_present():
    assert "Subota 13h. Eh, moj Beograde..." in OUT
    assert "Gužva u Beogradu." in OUT


def test_multiline_comment_joined():
    assert "Izadjosmo I mi gušteri na sunce 😅 Jeste bila gužva, ja se jedva parkirah u Zemunu." in OUT


def test_nested_comment_kept():
    assert "Pozdrav za Zemun!" in OUT
    assert "Neće ljudi metroom da se voze nego svojim kolima." in OUT


def test_scaffolding_stripped():
    for junk in ["u/Nemanja_Torbica", "pts", "points", "**r/srbija**", "Link:", "## Comments", "i.redd.it"]:
        assert junk not in OUT, f"leaked scaffolding: {junk!r}"


def test_bot_and_deleted_dropped():
    assert "AutoModerator" not in OUT
    assert "Molimo vas da pratite pravila" not in OUT      # bot body dropped too
    assert "[deleted]" not in OUT


def test_empty():
    assert parse_thread("") == ""
    assert parse_thread(None) == ""


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in fns:
        try:
            fn(); print(f"PASS {name}")
        except AssertionError as e:
            fails += 1; print(f"FAIL {name}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    print("--- parsed output ---")
    print(OUT)
    raise SystemExit(1 if fails else 0)
