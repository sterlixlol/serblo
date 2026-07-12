#!/usr/bin/env python3
"""Live terminal monitor for the synthetic-generation pillar.

One ssh call per refresh counts the lines of every generator's docs.jsonl on
the generation VM (tiny remote for-loop over `wc -l`), then redraws a bordered
table: lines now, lines/min since the monitor started, and a total row.
Missing files show as `pending`; if the VM is unreachable the last good
numbers stay on screen with a calm retry note. Stdlib only.

Usage:
  python scripts/gen_monitor.py           # live, refreshes every 25 s, Ctrl+C quits
  python scripts/gen_monitor.py --once    # single snapshot, then exit
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

# ── connection: edit these when the VM changes ─────────────────────────────
HOST = "104.154.207.146"       # GCE external IP (changes over time)
USER = "sterlix"
SSH_KEY = "~/.ssh/google_compute_engine"
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=15",
    "-o", "BatchMode=yes",                  # fail fast instead of prompting
    "-o", "UserKnownHostsFile=/dev/null",   # IP gets reused across VMs; avoid key-mismatch refusals
    "-o", "LogLevel=ERROR",
]
SSH_HARD_TIMEOUT = 45          # s; ConnectTimeout only covers the handshake

# ── what to watch ────────────────────────────────────────────────────────────
REMOTE_BASE = "$HOME/serblo/data/generated"
GENERATORS = [                 # (display name, file under REMOTE_BASE): one row per model lineage
    ("gemini",   "vertex-conv-pillar/docs.jsonl"),   # anchor, static (500k pillar, done)
    ("glm",      "glm/docs.jsonl"),
    ("deepseek", "deepseek/docs.jsonl"),
    ("kimi",     "kimi/docs.jsonl"),
    ("mimo",     "mimo/docs.jsonl"),
    ("grok",     "grok/docs.jsonl"),
]
REFRESH_SECONDS = 25

IS_TTY = sys.stdout.isatty()


def sgr(code, text):
    """Wrap text in an ANSI style when on a real terminal; plain otherwise."""
    return f"\x1b[{code}m{text}\x1b[0m" if IS_TTY else text


def fetch_counts():
    """One ssh round-trip. Returns {name: lines, or None if that file is missing},
    or None if the whole call failed (unreachable / timeout / garbled output)."""
    files = " ".join(f'"{REMOTE_BASE}/{path}"' for _, path in GENERATORS)
    remote = (
        f"for f in {files}; do "
        'if [ -f "$f" ]; then wc -l < "$f"; else echo MISSING; fi; '
        "done"
    )
    cmd = ["ssh", "-i", os.path.expanduser(SSH_KEY), *SSH_OPTS,
           f"{USER}@{HOST}", remote]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=SSH_HARD_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    tokens = [t for t in (ln.strip() for ln in proc.stdout.splitlines())
              if t == "MISSING" or t.isdigit()]
    if len(tokens) != len(GENERATORS):
        return None                          # partial/garbled: treat cycle as failed
    return {name: (None if tok == "MISSING" else int(tok))
            for (name, _), tok in zip(GENERATORS, tokens)}


def fmt_rate(delta, elapsed_s):
    if elapsed_s < 5:                        # too early for a meaningful rate
        return "—"
    return f"{delta / (elapsed_s / 60):,.1f}"


def render(counts, baseline, t0, started, note):
    """Redraw the whole screen from the latest state."""
    now = datetime.now().strftime("%H:%M:%S")
    up = timedelta(seconds=int(time.monotonic() - started))
    elapsed = (time.monotonic() - t0) if t0 is not None else 0.0

    rows = []                                # (name, lines_str, rate_str, dim)
    total = 0
    for name, _ in GENERATORS:
        if counts is None:
            rows.append((name, "—", "—", False))
        elif counts[name] is None:
            rows.append((name, "pending", "—", True))
        else:
            c = counts[name]
            total += c
            rows.append((name, f"{c:,}",
                         fmt_rate(c - baseline[name], elapsed), False))
    if counts is None:
        total_row = ("TOTAL", "—", "—")
    else:
        total_row = ("TOTAL", f"{total:,}",
                     fmt_rate(total - sum(baseline.values()), elapsed))

    name_w = max(len("generator"), len("TOTAL"), *(len(r[0]) for r in rows))
    lines_w = max(len("lines"), len(total_row[1]), *(len(r[1]) for r in rows))
    rate_w = max(len("lines/min"), len(total_row[2]), *(len(r[2]) for r in rows))

    def hline(left, mid, right):
        return ("  " + left + "─" * (name_w + 2) + mid + "─" * (lines_w + 2)
                + mid + "─" * (rate_w + 2) + right)

    def row(name, lines, rate, dim=False, bold=False):
        cells = [f" {name:<{name_w}} ", f" {lines:>{lines_w}} ",
                 f" {rate:>{rate_w}} "]
        if dim:
            cells = [sgr("2", c) for c in cells]
        if bold:
            cells = [sgr("1", c) for c in cells]
        return "  │" + "│".join(cells) + "│"

    out = [
        "",
        "  " + sgr("1", "serblo · generator monitor"),
        "  " + sgr("2", f"{USER}@{HOST} · every {REFRESH_SECONDS} s · {now} · up {up}"),
        "",
        hline("┌", "┬", "┐"),
        row("generator", "lines", "lines/min"),
        hline("├", "┼", "┤"),
        *[row(n, l, r, dim=d) for n, l, r, d in rows],
        hline("├", "┼", "┤"),
        row(*total_row, bold=True),
        hline("└", "┴", "┘"),
        "",
        "  " + sgr("2", note),
        "",
    ]
    if IS_TTY:
        sys.stdout.write("\x1b[2J\x1b[H")    # clear screen, cursor home
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="Watch generator progress on the VM.")
    ap.add_argument("--once", action="store_true", help="one snapshot, then exit")
    args = ap.parse_args()

    started = time.monotonic()
    counts = None        # last good fetch
    baseline = None      # counts at first good fetch (missing file → 0)
    t0 = None            # monotonic time of first good fetch
    last_good = None

    if IS_TTY:
        sys.stdout.write("\x1b[?25l")        # hide cursor
    try:
        while True:
            tick = time.monotonic()
            fresh = fetch_counts()
            stamp = datetime.now().strftime("%H:%M:%S")
            if fresh is not None:
                counts, last_good = fresh, stamp
                if baseline is None:
                    baseline = {n: (c or 0) for n, c in fresh.items()}
                    t0 = time.monotonic()
                note = f"updated {stamp} · Ctrl+C to quit"
            elif last_good:
                note = f"unreachable — retrying (showing {last_good} data)"
            else:
                note = "unreachable — retrying (no data yet)"
            render(counts, baseline, t0, started, note)
            if args.once:
                break
            time.sleep(max(0.0, REFRESH_SECONDS - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        if IS_TTY:
            sys.stdout.write("\x1b[?25h\n")  # restore cursor
            sys.stdout.flush()


if __name__ == "__main__":
    main()
