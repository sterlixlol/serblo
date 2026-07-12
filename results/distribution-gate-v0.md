# Distribution gate v0 — the unigram companion to self-BLEU (2026-07-03)

`scripts/distribution_gate.py`. Built because self-BLEU/distinct-n are blind to unigram-
frequency skew: the "ej" prime passed the diversity gate untouched and was caught only by
eye. This tool compares synthetic vs real unigram distributions and names the skewed tokens.

- **Jensen-Shannon divergence** (bits, symmetric, no smoothing) — the headline number.
- **KL(synth‖real) / KL(real‖synth)** — asymmetric, add-k smoothed.
- **OVER / UNDER token tables** — ranked by KL contribution; the actionable part.
- **`--control`** splits real in half for the sampling-noise JS floor to judge against.

## Validation (conv MiMo, PRE particle-fix, vs real subtitles, n=300)

| | JS divergence | read |
|---|---|---|
| control (real vs real) | **0.1135** | noise floor |
| MiMo synth vs real | **0.3645** | **3.2× the floor → clearly skewed** |

The OVER table reproduced the eyeball catch automatically and found more. **Crucially, a
high ratio is NOT automatically a defect** — the gate surfaces skews; classifying them is a
human/classifier call:

| token | synth/1k | real/1k | ratio | verdict |
|---|---|---|---|---|
| `*` | 28.7 | 0.0 | ∞ | **DEFECT — markdown `**bold**` leak. Fixed:** converter strips it. |
| `bre` `ma` `ej` `jbg` `pa` | — | — | 5–856× | **DEFECT — the particle prime. Fixed:** rotating 2-of-18 pool. |
| `lol` `reply` `brate` (English/slang) | — | — | high | **AUTHENTIC — keep.** Code-switching / English bleed is normal in casual BCS (gaming, tech, net-slang; "lol" ≈ League of Legends). NOT filtered. |
| `@` `:` (handles, speaker labels) | — | — | high | borderline — structural, plausibly real in forum/chat. Watch, don't filter. |

UNDER table = mostly subtitle-register artifacts (`.` `?` `!` `gospodine` `vas`) — dubbed-
movie formality that authentic online casual legitimately lacks. Not a synth defect.

## Takeaways
- Gate is a **diagnostic, not an auto-reject.** JS-vs-control-floor flags "something's skewed";
  the token tables say what; a human decides defect vs authentic.
- English/code-switching is **in-scope and authentic** for casual BCS — never filter it.
- Re-run after the essay batch finishes + `--rebuild` (particle-fix + markdown-strip applied):
  the `*` row should vanish and the particle ratios should collapse toward the floor. Content
  words and English will still differ from subtitles — that's real register difference, fine.
