"""Parse WebGrab reddit-scrape markdown into human-written Serbian text (PRD M1).

The scraper saves each post's raw scrape markdown verbatim. This extracts the
actual text (post title, body, and comment texts) and drops the scaffolding
(usernames, points, dates, links, AutoModerator, [deleted]/[removed]).
The result is one casual dialogue "thread document" per post.
"""
import re

_META = re.compile(r"^\*\*r/\S+\*\*\s*\|")                       # "**r/sub** | u/.. | N points | date"
_COMMENT = re.compile(r"^\s*-\s+\*\*u/[^*]+\*\*\s*\([^)]*\):\s?(.*)$")
_DROP = {"[deleted]", "[removed]", "", "...", "_"}


def _keep_continuation(s):
    return s and s != "---" and not s.startswith("**Link:") and not s.startswith("## ")


def parse_thread(raw_md):
    """Return post title + body + comment texts, one piece per line."""
    if not raw_md:
        return ""
    lines = raw_md.split("\n")

    title = lines[0][2:].strip() if lines and lines[0].startswith("# ") else ""

    meta_idx = comments_idx = None
    for i, ln in enumerate(lines):
        if meta_idx is None and _META.match(ln):
            meta_idx = i
        if ln.strip() == "## Comments":
            comments_idx = i
            break

    body = []
    if meta_idx is not None:
        end = comments_idx if comments_idx is not None else len(lines)
        for ln in lines[meta_idx + 1:end]:
            s = ln.strip()
            if _keep_continuation(s):
                body.append(s)

    comments, cur = [], None
    if comments_idx is not None:
        for ln in lines[comments_idx + 1:]:
            m = _COMMENT.match(ln)
            if m:
                if cur is not None:
                    comments.append(" ".join(cur).strip())
                if "**u/AutoModerator**" in ln:
                    cur = None                                   # skip bot + its continuations
                else:
                    t = m.group(1).strip()
                    cur = [t] if t else []
            elif cur is not None and _keep_continuation(ln.strip()):
                cur.append(ln.strip())
        if cur is not None:
            comments.append(" ".join(cur).strip())

    pieces = []
    if title and title not in _DROP:
        pieces.append(title)
    if body:
        pieces.append(" ".join(body))
    pieces.extend(c for c in comments if c not in _DROP)
    return "\n".join(pieces)
