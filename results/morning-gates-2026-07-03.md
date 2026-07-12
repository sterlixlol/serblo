# Morning gates (2026-07-03): particle-fix validation + Gemini vs MiMo essay

All at n=300, same-session pools (real conv = subtitles, real essay = FineWeb-2), same
scripts as prior gates. Corpus at gate time: MiMo 955 docs / 356.5k words (raw ledger 973),
Vertex/Gemini 4,991 docs / 1.588M words.

## 1. Particle fix: EMPIRICALLY CONFIRMED (night-1 conv, rotated particles)

| metric | pre-fix conv | night-1 conv | real subtitles |
|---|---|---|---|
| self-BLEU-4 | 0.307 | **0.239** | 0.334 |
| distinct-1 | 0.086 | **0.105** | 0.135 |
| JS divergence | 0.365 | **0.334** | floor 0.114 |

- **Primed-token skew eliminated:** bre (was 856×), ej (50×), jbg (∞), pa — all GONE from
  the top skew rows. `ma` fell 54×→26× (still pool-primed ~11% of prompts + MiMo just
  likes it).
- **Phrase diversity IMPROVED** (0.307→0.239 self-BLEU-4): varied priming = varied openings.
- distinct-1 gap to real closed ~40%.
- **What remains is NOT the prompt's doing:**
  (a) MiMo's intrinsic casual lexicon — `al` 620×, `brate` 189×, `ono` 13× — model
  fingerprint, addressable by grounding/classifier, not by prompt edits;
  (b) `:` 32× and `@` (handles/speaker labels) — artifacts of the REQUESTED platforms
  (discord/comment threads). Real *online chat* has these; the subtitle reference doesn't.
  (c) UNDER rows (`.` `!` `?`) = subtitles are punctuation-dense dubbed dialogue.
  → The remaining conv JS gap is substantially REFERENCE-POOL MISMATCH (subtitles ≠ online
  chat). Long-term fix: a genuine forum/chat reference pool (the thin Reddit slice was the
  plan; only 38 docs exist). Log, don't chase with prompt hacks.

## 2. Gemini Flash-Lite essay: WORSE structural diversity than MiMo (surprise)

| metric (n=300) | Gemini Flash-Lite | MiMo | real |
|---|---|---|---|
| self-BLEU-4 | **0.263 (1.83× real)** | 0.218 (1.51×) | 0.144 |
| self-BLEU-3 | 0.427 | 0.380 | 0.283 |
| distinct-1 | 0.155 | 0.166 | 0.176 |
| JS divergence | **0.385** | 0.344 | floor 0.207 |
| avg tokens/doc | 367 | 434 | ~770 |

Gemini's shorter docs should *depress* self-BLEU, yet it scores highest → true structural
sameness, close to the 2× kill-line. This matches the PRD's original "Lite mode-collapses
at volume" fear — measured on essays. Both models share the same handicaps here (one
template, no grounding, temp 0.8), and both converge on essay skeletons; Lite converges
hardest. **Allocation consequence (per the locked gate-reweighting mechanism):** essay
synthetic should be MULTI-SOURCE garnish (Lite + MiMo + later GPT/Opus), and **grounding
(style_transfer anchors) is now the #1 essay lever** — it attacks the shared failure mode.

Cheap A/Bs worth running before scaling essay volume:
- **Flash (non-Lite) 500-essay batch** ≈ 500×130 in-tok ×$0.15/M + 500×700 out ×$1.25/M ≈
  **$0.45** — is the pricier tier meaningfully more diverse?
- **Temp 0.8 → 1.0** on Lite, 500 essays ≈ **$0.08** — cheapest possible diversity dial.

## 3. Cost correction (actual vs projected)
Gemini essays came back SHORT (367 tok avg vs 700 assumed): actual output ≈ 1.9M tokens →
**actual job cost ≈ $0.41**, not $0.73. Credit burn so far: ~$0.41 of $300.

## 4. Provenance repair (same morning)
The 03:10 dash-norm rebuild had stamped new-manifest axes (incl. `particles`) onto 312
pre-fix conv docs (runbook §10 step-5 caveat, self-inflicted). Fixed structurally:
- raw records now embed register+axes AT GENERATION TIME (run_mimo + run_vertex);
- rebuild prefers embedded axes, manifest is legacy fallback;
- `scripts/backfill_raw_axes.py` repaired all 973+5,000 existing raws from the archived
  v0 manifest (old conv) / current manifests (rest). Verified: 0/313 old conv raws carry
  particles, 343/343 night-1 do.
