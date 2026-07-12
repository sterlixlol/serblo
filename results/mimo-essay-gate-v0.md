# MiMo essay — diversity + distribution gate v0 (2026-07-03)

**Source:** 308 essay docs / ~118k words, `mimo-v2.5` free channels, post payload-policy rebuild.
Gates at n=300, same-session pools (real = FineWeb-2 slice), `diversity_baseline.py` +
`distribution_gate.py`. Conv gate for comparison: `results/mimo-conv-gate-v0.md`.

## Verdict: CONDITIONAL PASS — essay-light only (which was the plan)

| metric (n=300) | MiMo essay | real essays | ratio | read |
|---|---|---|---|---|
| self-BLEU-2 | 0.624 | 0.520 | 1.20× | elevated |
| self-BLEU-3 | 0.380 | 0.283 | 1.34× | elevated |
| **self-BLEU-4** | **0.218** | **0.144** | **1.51×** | **amber: real homogenization, below the 2× kill-rule** |
| distinct-1/2/3 | .166/.647/.899 | .176/.645/.886 | ~1× | lexical variety at PARITY |
| JS divergence | 0.351 | floor 0.207 | 1.69× | skewed, but ⅓ of it is real-side dirt (below) |

Interpretation: **"same skeleton, different words."** Word-level diversity matches real
(distinct-n parity — the conv-gate story in reverse), but phrase/structure repetition is
1.5× real: single model + single template + **no grounding** converges on one essay
architecture. Shorter synth docs (434 vs ~770 tok) should bias self-BLEU DOWN, so the gap
is not a length artifact — if anything understated. Contrast conv: MiMo conv PASSED
below-real (0.307 vs 0.334); the essay register is where MiMo homogenizes.

## MiMo's essay fingerprint (distribution gate OVER table)

- **em-dash `—` 72× real rate** (10.2/1k vs 0.14) — MiMo's LLM-ese tell, in Serbian too.
- Academic-connective inflation: `poput` 12×, `predstavlja` 7×, `često` 7×, `već` 7×,
  `kroz` 6×, `između` 6× — the essay-glue vocabulary, all pumped.
- Commas +27% (long subordinate-clause style).
Not auto-rejects (per gate doctrine) — but they're the exact features the XLM-R classifier
and grounding should push against at scale.

## Incidental finding (real-side): FineWeb-2 slice carries wiki chrome
UNDER table shows real essays full of `|`, `()`, `[]`, `/`, and literally **`uredi`**
(sr-wiki "edit" buttons) at ~1/1k tokens → **M2/M5 TODO: boilerplate stripping** of wiki
chrome from the real corpus. The gate accidentally audited the *real* data too.

## Actions
1. MiMo essay allocation stays **essay-light** (PRD already said so; gate confirms).
2. Do NOT scale MiMo essay volume before wiring **grounding** (style_transfer anchors from
   real essays — in generation-config, not yet in the harness) and/or temp bump; re-gate after.
3. Gemini Flash-Lite (Vertex batch, in flight) is the essay PRIMARY — head-to-head on the
   330 shared prompt ids when it lands.
4. Real-corpus wiki-chrome strip goes on the M2 backlog.
