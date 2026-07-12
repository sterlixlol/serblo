#!/usr/bin/env python3
"""Serblo pretrain watch: a read-only instrument panel for the Vertex AI run.

Pulls the current serblo-full training leg via gcloud (list → describe →
train logs → eval logs → checkpoint listing, strictly one call at a time),
computes the numbers that matter (ppl, epoch, spend, ETAs), and draws one
calm panel. Never writes anything anywhere: this script only looks.

Usage:
  python3 scripts/watch_training.py             # one snapshot, then exit
  python3 scripts/watch_training.py --watch     # live panel, refresh every 30 s
  python3 scripts/watch_training.py --watch 60  # ... every 60 s (min 5)

If a gcloud call fails or a log window is empty, the affected fields show
an em dash and everything else keeps rendering. Stdlib only.
"""
import argparse
import collections
import io
import math
import os
import pickle
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

# ── account / job: edit these two first when hopping cloud accounts ─────────
ACCOUNT          = "banda776drugovi@gmail.com"
PROJECT          = "project-a9a36e4a-9735-4042-803"
REGION           = "us-central1"
CKPT_PREFIX      = "gs://serblo-vertex-a9a36e4a/serblo/checkpoints/"
JOB_FILTER       = "serblo-full"        # displayName substring

# ── run geometry ─────────────────────────────────────────────────────────────
TOK_PER_STEP     = 524_288              # micro-batch 32 × grad-accum 8 × 2048 ctx
EFFECTIVE_TOKENS = 23_647_957_288       # one epoch = a full pass over this
MAX_STEPS        = 135_000            # 2.99 epochs, annealed finish
TOK_PER_BYTE     = 0.2424               # for bits-per-byte
RANDOM_LOSS      = 10.4                 # ln(32000): where loss started
PATIENCE         = 15                   # val-plateau auto-stop threshold

# ── money ────────────────────────────────────────────────────────────────────
HOURLY_COST      = 4.0                  # Spot H100, USD/hr (rough)
BUDGET           = 252.0                # this account's credit
MONEY_HOP        = 230.0                # switch accounts when spend crosses this

GCLOUD_TIMEOUT   = 30                   # s per call; a refresh is ~5 calls
TRAIN_LOG_LIMIT  = 30                   # sparkline window
LIVE_HEAD_BYTES  = 262_143              # ckpt head-read window (data.pkl is ~68KB)
FALLBACK_STEPS_PER_HR = 2_482           # measured H100 cadence; ETA fallback when logs are dark

AUTH = [f"--account={ACCOUNT}", f"--project={PROJECT}"]


# ── palette ──────────────────────────────────────────────────────────────────
# chrome faint, labels gray, values bright; accent for data art; g/a/r = health
COLOR_ON = ("NO_COLOR" not in os.environ) and (
    sys.stdout.isatty()
    or os.environ.get("CLICOLOR_FORCE") not in (None, "0")
    or os.environ.get("FORCE_COLOR") not in (None, "0")
)

def _c(code):
    return f"\x1b[{code}m" if COLOR_ON else ""

RESET  = _c("0")
BOLD   = _c("1")
FAINT  = _c("38;5;240")    # box lines, footer
LABEL  = _c("38;5;246")    # field labels, quiet notes
VALUE  = _c("38;5;231")    # hero numbers
ACCENT = _c("38;5;110")    # bars, sparkline, wordmark
GREEN  = _c("38;5;114")
AMBER  = _c("38;5;215")
RED    = _c("38;5;203")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DASH    = "—"


def st(text, *codes):
    """Style a fragment (no-op when color is off)."""
    return "".join(codes) + str(text) + RESET if COLOR_ON else str(text)


def vlen(s):
    """Visible length: what the terminal shows, escapes stripped."""
    return len(ANSI_RE.sub("", s))


def vtrunc(s, width):
    """Truncate to a visible width without cutting an escape sequence."""
    if vlen(s) <= width:
        return s
    out, budget, saw_esc = [], width - 1, False
    for tok in re.split(r"(\x1b\[[0-9;]*m)", s):
        if tok.startswith("\x1b"):
            out.append(tok)
            saw_esc = True
        else:
            take = tok[:max(0, budget)]
            out.append(take)
            budget -= len(take)
    return "".join(out) + ("…" if width > 0 else "") + (RESET if saw_esc else "")


# ── little formatters ────────────────────────────────────────────────────────
def fmt_int(n):
    return f"{n:,}" if n is not None else DASH


def fmt_tokens(n):
    if n is None:
        return DASH
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            return f"{n / div:.1f}{suffix}"
    return str(n)


def fmt_dur(seconds):
    """'17h 29m', '2d 3h', '45m', '<1m': two units, no seconds noise."""
    if seconds is None or seconds < 0:
        return DASH
    m = int(seconds // 60)
    d, h, m = m // 1440, (m % 1440) // 60, m % 60
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m" if m else "<1m"


def fmt_eta(hours):
    """'~40.1h · Jul 11 08:05' in local time; '—' when unknowable."""
    if hours is None or hours < 0 or not math.isfinite(hours):
        return DASH
    at = datetime.now().astimezone() + timedelta(hours=hours)
    return f"~{hours:.1f}h · {at:%b %d %H:%M}"


def fmt_money(x):
    return f"${x:,.2f}" if x is not None else DASH


def fmt_gib(nbytes):
    return f"{nbytes / 2**30:.2f} GiB" if nbytes else DASH


# ── one gcloud call ──────────────────────────────────────────────────────────
def sh(args):
    """Run one read-only gcloud command. stdout on success (may be empty),
    None when the call failed or timed out; callers show a dash and move on."""
    try:
        p = subprocess.run(
            args, capture_output=True, text=True,
            timeout=GCLOUD_TIMEOUT, stdin=subprocess.DEVNULL,
        )
        return p.stdout.strip() if p.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def parse_ts(s):
    try:
        return datetime.fromisoformat(s.strip())
    except (ValueError, AttributeError):
        return None


# ── fetchers (each wraps exactly one verified gcloud command) ────────────────
def fetch_legs():
    """All serblo-full legs, newest API order. None = gcloud unreachable."""
    out = sh([
        "gcloud", "ai", "custom-jobs", "list", f"--region={REGION}", *AUTH,
        f"--filter=displayName~{JOB_FILTER}",
        "--format=value(name.basename(),displayName,state,createTime)",
        "--limit=10",
    ])
    if out is None:
        return None
    legs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) >= 4 and parts[0].isdigit():
            legs.append({
                "id": parts[0], "name": parts[1], "state": parts[2],
                "created": parse_ts(parts[3]), "ended": None,
            })
    return legs


def enrich_leg(leg):
    """Refresh one leg's state and pick up endTime once it has finished."""
    out = sh([
        "gcloud", "ai", "custom-jobs", "describe", leg["id"],
        f"--region={REGION}", *AUTH,
        "--format=value(state,createTime,endTime)",
    ])
    if not out:
        return
    parts = [p.strip() for p in out.split("\t")]
    if len(parts) >= 1 and parts[0]:
        leg["state"] = parts[0]
    if len(parts) >= 2 and parts[1]:
        leg["created"] = parse_ts(parts[1]) or leg["created"]
    if len(parts) >= 3 and parts[2]:
        leg["ended"] = parse_ts(parts[2])


def fetch_log_lines(job_id, needle, limit, freshness):
    out = sh([
        "gcloud", "logging", "read",
        f'resource.type="ml_job" resource.labels.job_id="{job_id}" '
        f'textPayload:"{needle}"',
        f"--limit={limit}", "--order=desc", f"--freshness={freshness}",
        *AUTH, "--format=value(textPayload)",
    ])
    return out.splitlines() if out else []


TRAIN_RE = re.compile(
    r"step\s+([\d,]+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*lr\s+(\S+)"
    r"\s*\|\s*gnorm\s+([\d.]+)\s*\|\s*([\d,]+)\s*tok/s")
EVAL_RE = re.compile(
    r"\[eval\]\s*step\s+([\d,]+)\s*\|\s*val_loss\s+([\d.]+)"
    r"\s*\|\s*best\s+([\d.]+)\s*\|\s*bad_evals\s+(\d+)")
CKPT_ROW_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(gs://\S+)\s*$")


def fetch_train(job_id):
    """Parsed train points, oldest → newest (for the sparkline)."""
    pts = []
    for line in fetch_log_lines(job_id, "tok/s", TRAIN_LOG_LIMIT, "6h"):
        m = TRAIN_RE.search(line)
        if m:
            pts.append({
                "step": int(m.group(1).replace(",", "")),
                "loss": float(m.group(2)),
                "lr": m.group(3),
                "gnorm": float(m.group(4)),
                "tok_s": int(m.group(5).replace(",", "")),
            })
    pts.reverse()
    return pts


def fetch_eval(job_id):
    """Parsed eval points, newest first."""
    pts = []
    for line in fetch_log_lines(job_id, "val_loss", 4, "12h"):
        m = EVAL_RE.search(line)
        if m:
            pts.append({
                "step": int(m.group(1).replace(",", "")),
                "val_loss": float(m.group(2)),
                "best": float(m.group(3)),
                "bad": int(m.group(4)),
            })
    return pts


def fetch_ckpts():
    """[{name, size, ts}] for ckpt_* objects; None = listing failed entirely."""
    out = sh(["gcloud", "storage", "ls", "-l", CKPT_PREFIX, *AUTH])
    if out is None:                       # -l someday unsupported? plain names still reassure
        out = sh(["gcloud", "storage", "ls", CKPT_PREFIX, *AUTH])
        if out is None:
            return None
    items = []
    for line in out.splitlines():
        m = CKPT_ROW_RE.match(line)
        url = m.group(3) if m else (line.strip() if line.strip().startswith("gs://") else None)
        if not url:
            continue
        name = url.rstrip("/").rsplit("/", 1)[-1]
        if not name.startswith("ckpt"):
            continue
        items.append({
            "name": name,
            "size": int(m.group(1)) if m else None,
            "ts": parse_ts(m.group(2)) if m else None,
        })
    return items


def _sh_bytes(args):
    """Like sh() but binary stdout (for the checkpoint head-read)."""
    try:
        p = subprocess.run(args, capture_output=True,
                           timeout=GCLOUD_TIMEOUT, stdin=subprocess.DEVNULL)
        return p.stdout if p.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


class _AnyStub:
    """Inert stand-in for every torch class during metadata unpickling."""
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _AnyStub()
    def __setitem__(self, k, v): pass
    def __setstate__(self, s): pass


class _MetaUnpickler(pickle.Unpickler):
    """Only OrderedDict is real; everything else becomes _AnyStub, so no torch
    import is needed and no arbitrary class can load (safer than a full unpickle)."""
    def find_class(self, mod, name):
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        return _AnyStub

    def persistent_load(self, pid):
        return None


def fetch_ckpt_live():
    """Exact {step, tokens, best_val, bad_evals} straight from ckpt_latest.pt via a
    256KB head-read: the log-proof channel. torch.save writes a zip whose FIRST
    entry is data.pkl (streamed: local-header sizes are 0, payload ends at the
    PK\\x07\\x08 data descriptor). None on any failure; logs then carry the panel."""
    raw = _sh_bytes(["gcloud", "storage", "cat", "-r", f"0-{LIVE_HEAD_BYTES}",
                     CKPT_PREFIX + "ckpt_latest.pt", *AUTH])
    if not raw or raw[:4] != b"PK\x03\x04":
        return None
    try:
        nlen, elen = struct.unpack("<HH", raw[26:30])
        start = 30 + nlen + elen
        end = raw.find(b"PK\x07\x08", start)
        if end < 0:
            return None
        d = _MetaUnpickler(io.BytesIO(raw[start:end])).load()
        return {"step": int(d["step"]), "tokens": int(d["tokens"]),
                "best_val": float(d["best_val"]), "bad_evals": int(d.get("bad_evals", 0))}
    except Exception:
        return None


# ── gather one reading ───────────────────────────────────────────────────────
def gather():
    now = datetime.now(timezone.utc)
    snap = {"now": now, "legs": None, "cur": None, "active": False,
            "train": [], "eval": [], "ckpts": None, "spend": None}

    legs = fetch_legs()
    snap["legs"] = legs
    if not legs:
        return snap                        # unreachable (None) or nothing matched ([])

    for leg in legs:                       # one describe per leg, sequentially
        enrich_leg(leg)

    epoch0 = datetime.min.replace(tzinfo=timezone.utc)
    running = [l for l in legs if l["state"] == "JOB_STATE_RUNNING"]
    snap["cur"] = max(running or legs, key=lambda l: l["created"] or epoch0)
    snap["active"] = bool(running)

    hours = 0.0
    for leg in legs:                       # a leg bills from create to end (or now)
        if leg["created"]:
            hours += max(0.0, ((leg["ended"] or now) - leg["created"]).total_seconds()) / 3600
    snap["spend"] = hours * HOURLY_COST

    job_id = snap["cur"]["id"]
    snap["train"] = fetch_train(job_id)
    snap["eval"] = fetch_eval(job_id)
    snap["ckpts"] = fetch_ckpts()
    snap["live"] = fetch_ckpt_live()       # log-proof truth channel
    return snap


# ── drawing ──────────────────────────────────────────────────────────────────
BLOCKS   = "▁▂▃▄▅▆▇█"
PARTIALS = " ▏▎▍▌▋▊▉"


def sparkline(vals):
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span < 1e-12:
        return "▄" * len(vals)
    return "".join(BLOCKS[min(7, int((v - lo) / span * 8))] for v in vals)


def bar(frac, width, color):
    frac = min(max(frac, 0.0), 1.0)
    cells = frac * width
    full = int(cells)
    eighth = int((cells - full) * 8)
    body = "█" * full + (PARTIALS[eighth] if full < width and eighth else "")
    return st(body, color) + st("░" * (width - vlen(body)), FAINT)


class Panel:
    """Accumulates rows, then boxes them at a fixed inner width."""

    def __init__(self, width):
        self.w = width                     # total panel width incl. borders
        self.iw = width - 2                # interior between the │ borders
        self.lines = []

    def row(self, left="", right=""):
        left = "  " + left
        gap = self.iw - vlen(left) - vlen(right) - 2
        if gap < 1 and right:              # crowded: keep the left cell, drop right
            left, right, gap = vtrunc(left, self.iw - 2), "", 0
        body = left + " " * max(0, gap) + right + ("  " if right else "")
        self.lines.append(vtrunc(body, self.iw) if vlen(body) > self.iw else body)

    def rule(self, title=None):
        if title:
            fill = "─" * max(0, self.iw - len(title) - 3)   # "─ " + title + " " + fill
            self.lines.append(("RULE", st("├─ ", FAINT) + st(title, BOLD, LABEL)
                               + st(" " + fill + "┤", FAINT)))
        else:
            self.lines.append(("RULE", st("├" + "─" * self.iw + "┤", FAINT)))

    def render(self):
        top = st("╭" + "─" * self.iw + "╮", FAINT)
        bot = st("╰" + "─" * self.iw + "╯", FAINT)
        out = [top]
        for line in self.lines:
            if isinstance(line, tuple):    # a pre-built rule, already full width
                out.append(line[1])
            else:
                pad = " " * max(0, self.iw - vlen(line))
                out.append(st("│", FAINT) + line + pad + st("│", FAINT))
        out.append(bot)
        return "\n".join(out)


STATE_WORD = {
    "JOB_STATE_RUNNING":   ("RUNNING", GREEN),
    "JOB_STATE_PENDING":   ("PENDING", AMBER),
    "JOB_STATE_QUEUED":    ("QUEUED", AMBER),
    "JOB_STATE_PAUSED":    ("PAUSED", AMBER),
    "JOB_STATE_SUCCEEDED": ("SUCCEEDED", GREEN),
    "JOB_STATE_FAILED":    ("FAILED", RED),
    "JOB_STATE_CANCELLED": ("CANCELLED", RED),
}


def state_badge(state):
    word, color = STATE_WORD.get(state, (state.replace("JOB_STATE_", "") or DASH, AMBER))
    return st("● ", color) + st(word, BOLD, color)


def bad_evals_color(bad):
    return GREEN if bad < 5 else (AMBER if bad < 10 else RED)


def compress_steps(steps):
    """[10000, 20000, ...] → '10k · 20k · … · 190k · 200k (20)'."""
    tags = [f"{s // 1000}k" for s in steps]
    if len(tags) > 6:
        tags = tags[:2] + ["…"] + tags[-3:]
        return " · ".join(tags) + f"  ({len(steps)} total)"
    return " · ".join(tags) if tags else DASH


def render(snap, watch_every=None):
    term_w = shutil.get_terminal_size((80, 24)).columns
    p = Panel(max(66, min(80, term_w)))
    now = snap["now"]
    cur, legs = snap["cur"], snap["legs"]
    train, evals, ckpts = snap["train"], snap["eval"], snap["ckpts"]
    last = train[-1] if train else None

    # derived numbers
    live = snap.get("live")                # exact, log-proof (from ckpt_latest's head)
    log_step = last["step"] if last else (evals[0]["step"] if evals else None)
    step = log_step
    if live and (step is None or live["step"] > step):
        step = live["step"]
    stale_logs = bool(live) and (log_step is None or live["step"] - log_step >= 1500)
    loss = last["loss"] if last else None
    tok_s = round(sum(t["tok_s"] for t in train) / len(train)) if train else None
    tokens_seen = (live["tokens"] if (live and step == live["step"])
                   else step * TOK_PER_STEP if step is not None else None)
    epoch = tokens_seen / EFFECTIVE_TOKENS if tokens_seen is not None else None
    planned_epochs = MAX_STEPS * TOK_PER_STEP / EFFECTIVE_TOKENS
    spend = snap["spend"]
    left_to_hop = max(0.0, MONEY_HOP - spend) if spend is not None else None
    steps_per_hr = tok_s * 3600 / TOK_PER_STEP if tok_s else 0.0
    if not steps_per_hr and stale_logs and snap["active"]:
        steps_per_hr = FALLBACK_STEPS_PER_HR   # measured cadence: ETA survives log blackouts
    eta_hop_h = (left_to_hop / HOURLY_COST) if (left_to_hop is not None and snap["active"]) else None
    eta_max_h = ((MAX_STEPS - step) / steps_per_hr
                 if (step is not None and steps_per_hr and snap["active"]) else None)

    # ── header ──
    p.row(st("SERBLO", BOLD, ACCENT) + st(" · 125M BCS base model", LABEL),
          st(f"Vertex AI · {REGION}", FAINT))
    if cur:
        age = ((cur["ended"] or now) - cur["created"]).total_seconds() if cur["created"] else None
        when = f"up {fmt_dur(age)}" if not cur["ended"] else \
               f"ran {fmt_dur(age)} · ended {fmt_dur((now - cur['ended']).total_seconds())} ago"
        extra = f" · {len(legs)} legs" if len(legs) > 1 else ""
        p.row(st(cur["name"], VALUE, BOLD) + "   " + state_badge(cur["state"]),
              st(when + extra, LABEL))
        if not snap["active"]:
            p.row(st("no active leg — supervisor should be resubmitting", AMBER))
    elif legs is None:
        p.row(st("● ", RED) + st("gcloud unreachable — check network/auth, will show what it can", RED))
    else:
        p.row(st("● ", RED) + st(f"no jobs matching '{JOB_FILTER}' found", RED))

    # ── learning ──
    p.rule("LEARNING")
    if loss is not None:
        p.row(st("train loss ", LABEL) + st(f"{loss:.4f}", VALUE, BOLD)
              + st("  ppl ", LABEL) + st(f"{math.exp(loss):.2f}", VALUE),
              st(f"{RANDOM_LOSS} ", LABEL) + st("→", GREEN)
              + st(f" {loss:.2f}", VALUE) + st(f" · -{(1 - loss / RANDOM_LOSS) * 100:.0f}% vs random", LABEL))
    else:
        p.row(st("train loss ", LABEL) + st(DASH, VALUE)
              + st("  (no train lines in the last 6h)", LABEL))
    if evals:
        ev = evals[0]
        is_best = ev["val_loss"] <= ev["best"]
        p.row(st("val loss   ", LABEL) + st(f"{ev['val_loss']:.4f}", VALUE, BOLD)
              + st("  ppl ", LABEL) + st(f"{math.exp(ev['val_loss']):.2f}", VALUE)
              + st(f"  @ step {ev['step'] // 1000}k", FAINT),
              st("best ", LABEL) + st(f"{ev['best']:.4f}", GREEN if is_best else VALUE)
              + st("  bad evals ", LABEL)
              + st(f"{ev['bad']}/{PATIENCE}", BOLD, bad_evals_color(ev["bad"])))
    else:
        p.row(st("val loss   ", LABEL) + st(DASH, VALUE)
              + st("  (no eval lines in the last 12h)", LABEL))
    if live and (not evals or evals[0]["step"] < live["step"] - 1500):
        p.row(st("live ckpt  ", LABEL) + st(f"best {live['best_val']:.4f}", VALUE, BOLD)
              + st("  ppl ", LABEL) + st(f"{math.exp(live['best_val']):.2f}", VALUE)
              + st(f"  @ step {live['step'] // 1000}k", FAINT),
              st("bad evals ", LABEL)
              + st(f"{live['bad_evals']}/{PATIENCE}", BOLD, bad_evals_color(live["bad_evals"])))
    if stale_logs:
        p.row(st("logs stale — live numbers read from ckpt_latest.pt (Cloud Logging lag)", AMBER))
    p.row(st("bits/byte  ", LABEL)
          + st(f"{loss / math.log(2) * TOK_PER_BYTE:.3f}" if loss is not None else DASH, VALUE),
          st(f"lr {last['lr']} · gnorm {last['gnorm']}", LABEL) if last else "")
    if len(train) >= 2:
        losses = [t["loss"] for t in train]
        p.row("")
        p.row(st(sparkline(losses), ACCENT)
              + st(f"  last {len(losses)} logs · {min(losses):.3f}–{max(losses):.3f}"
                   f" · avg {sum(losses) / len(losses):.3f}", FAINT))

    # ── progress ──
    p.rule("PROGRESS")
    bw = p.iw - 46                          # bar shares geometry with the money bar
    if step is not None:
        frac = step / MAX_STEPS
        p.row(st("step   ", LABEL) + st(f"{fmt_int(step):>7}", VALUE, BOLD)
              + st(f" / {fmt_int(MAX_STEPS)}  ", LABEL)
              + bar(frac, bw, ACCENT) + st(f"  {frac * 100:.1f}%", VALUE))
        p.row(st("epoch  ", LABEL) + st(f"{epoch:.2f}", VALUE, BOLD)
              + st(f" of ~{planned_epochs:.1f} at {MAX_STEPS // 1000}k steps", LABEL),
              st("tokens seen ", LABEL) + st(fmt_tokens(tokens_seen), VALUE, BOLD))
        p.row(st("ETA    ", LABEL)
              + st(f"step {MAX_STEPS // 1000}k in {fmt_eta(eta_max_h)}"
                   if eta_max_h is not None else f"step {MAX_STEPS // 1000}k {DASH}", VALUE))
    else:
        p.row(st("step   ", LABEL) + st(DASH, VALUE) + st(f" / {fmt_int(MAX_STEPS)}  ", LABEL)
              + bar(0, bw, ACCENT))

    # ── money ──
    p.rule("MONEY")
    if spend is not None:
        frac = spend / MONEY_HOP
        money_color = GREEN if frac < 0.8 else (AMBER if frac < 0.95 else RED)
        p.row(st("spend  ", LABEL) + st(f"{fmt_money(spend):>7}", VALUE, BOLD)
              + st(f" / ${MONEY_HOP:.0f} hop ", LABEL)      # same 28-col prefix as the step bar
              + bar(frac, bw, money_color) + st(f"  {frac * 100:.1f}%", VALUE))
        hop_note = ""
        if eta_hop_h is not None and eta_max_h is not None:
            hop_note = "hop comes first" if eta_hop_h < eta_max_h else "200k comes first"
        p.row(st("hop    ", LABEL) + st(fmt_money(left_to_hop), VALUE)
              + st(" left → ", LABEL)
              + st(fmt_eta(eta_hop_h) if snap["active"] else "paused (no active leg)", VALUE),
              st((hop_note + " · " if hop_note else "") + f"credit ${BUDGET:.0f}", FAINT))
    else:
        p.row(st("spend  ", LABEL) + st(DASH, VALUE))
    p.row(st("rate   ", LABEL)
          + (st(f"{fmt_int(tok_s)}", VALUE, BOLD) + st(" tok/s · ", LABEL)
             + st(f"{fmt_int(round(steps_per_hr))}", VALUE) + st(" steps/hr", LABEL)
             if tok_s else st(DASH, VALUE)),
          st(f"${HOURLY_COST:.2f}/hr spot H100", FAINT))

    # ── checkpoints ──
    p.rule("CHECKPOINTS")
    if ckpts:
        latest = next((c for c in ckpts if c["name"] == "ckpt_latest.pt"), None)
        if latest:
            if latest["ts"]:
                age_s = (now - latest["ts"]).total_seconds()
                age_color = GREEN if age_s < 45 * 60 else (AMBER if age_s < 2 * 3600 else RED)
                saved = st("saved ", LABEL) + st(f"{fmt_dur(age_s)} ago", BOLD, age_color)
            else:
                saved = st("present", GREEN)
            p.row(st("latest     ", LABEL) + st("ckpt_latest.pt", VALUE)
                  + st(f"  {fmt_gib(latest['size'])}  ", LABEL) + saved,
                  st("updates every 1k steps", FAINT))
        else:
            p.row(st("latest     ", LABEL) + st("ckpt_latest.pt missing", AMBER))
        steps = sorted(int(m.group(1)) for c in ckpts
                       if (m := re.match(r"ckpt_step(\d+)\.pt$", c["name"])))
        p.row(st("snapshots  ", LABEL) + st(compress_steps(steps), VALUE),
              st("permanent, every 10k", FAINT))
    elif ckpts is None:
        p.row(st("checkpoint listing unavailable (gcloud storage ls failed)", LABEL))
    else:
        p.row(st("no ckpt_* objects under " + CKPT_PREFIX, AMBER))

    # ── footer ──
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    tail = (f"refreshing every {watch_every:.0f}s · Ctrl-C to quit"
            if watch_every else "one snapshot · --watch for a live panel")
    return p.render() + "\n" + st(f"  read {stamp} · {tail}", FAINT)


# ── entry ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Read-only dashboard for the Serblo Vertex AI pretraining run.",
        epilog="examples:  watch_training.py           one snapshot\n"
               "           watch_training.py --watch    live, every 30 s\n"
               "           watch_training.py --watch 60 live, every 60 s",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", nargs="?", const=30.0, type=float, default=None,
                    metavar="SECONDS", help="refresh in a loop (default 30 s, min 5)")
    args = ap.parse_args()

    if args.watch is None:
        print(render(gather()))
        return

    every = max(5.0, args.watch)
    if COLOR_ON:
        sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")    # hide cursor, clear once
        sys.stdout.flush()
    try:
        while True:
            frame = render(gather(), watch_every=every)
            if COLOR_ON:
                sys.stdout.write("\x1b[H" + frame + "\x1b[J\n")   # repaint in place
            else:
                sys.stdout.write(frame + "\n")
            sys.stdout.flush()
            time.sleep(every)
    except KeyboardInterrupt:
        pass
    finally:
        if COLOR_ON:
            sys.stdout.write("\x1b[?25h")             # cursor back
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    main()
