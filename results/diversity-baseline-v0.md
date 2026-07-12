# Diversity baselines — v0 (real BCS, two registers)

**Date:** 2026-07-01
**Purpose:** the real-Serbian target numbers the diversity gate (PRD §7.4) measures synthetic against — **per register**, because the two registers are ~2× apart.
**Tool:** `scripts/diversity_baseline.py`, **n = 300**, seed 0, identical preprocessing (NFC + Cyrillic→Latin gajica).

| register | source | n docs staged | avg tokens/doc |
|---|---|---|---|
| general/essay | FineWeb-2 BCS (M1 slice) | 80k | 821 |
| conversational | OpenSubtitles BCS (M1 slice) | 60k | 323 |

## Targets (compare synthetic at n=300, same register, matched doc length)

| metric | **essay** pooled / srp | **conversational** pooled / sr |
|---|---|---|
| self-BLEU-2 | 0.529 / 0.564 | 0.658 / 0.682 |
| self-BLEU-3 | 0.281 / 0.311 | 0.429 / 0.459 |
| **self-BLEU-4** | **0.126 / 0.146** | **0.252 / 0.279** |
| distinct-2 | 0.647 / 0.630 | 0.545 / 0.521 |
| distinct-3 | 0.897 / 0.904 | 0.843 / 0.826 |
| entropy-norm 2/3 | 0.951 / 0.991 | 0.919 / 0.980 |

```json
{"version":"v0","n":300,
 "essay":{"source":"fineweb2-bcs","avg_tokens":821,
   "pooled":{"self_bleu":{"2":0.5288,"3":0.2807,"4":0.1264},"distinct_n":{"1":0.176,"2":0.6472,"3":0.8971},"entropy_norm":{"1":0.734,"2":0.951,"3":0.991}},
   "srp":{"self_bleu":{"2":0.5637,"3":0.3113,"4":0.1461},"distinct_n":{"1":0.1663,"2":0.6298,"3":0.9037},"entropy_norm":{"1":0.721,"2":0.948,"3":0.992}}},
 "conversational":{"source":"opensubtitles-bcs","avg_tokens":323,
   "pooled":{"self_bleu":{"2":0.6583,"3":0.4286,"4":0.252},"distinct_n":{"1":0.1536,"2":0.5454,"3":0.8432},"entropy_norm":{"1":0.675,"2":0.919,"3":0.98}},
   "sr":{"self_bleu":{"2":0.6822,"3":0.4586,"4":0.2792},"distinct_n":{"1":0.1444,"2":0.5212,"3":0.826},"entropy_norm":{"1":0.67,"2":0.914,"3":0.978}}}}
```

## The key finding: registers are ~2× apart
**Conversational self-BLEU-4 ≈ 0.25–0.28 vs essay ≈ 0.13–0.15** — dialogue is roughly twice as repetitive (subtitles recycle "Da / Ne znam / Hvala" endlessly). **Consequence: the gate must use per-register targets.** Gating conversational synthetic against the essay number (~0.13) would wrongly reject good dialogue; gating essay synthetic against the conversational number (~0.25) would wave through slop. Use ~0.25 for casual synthetic, ~0.13 for essay synthetic.

## How the gate uses this
A synthetic source **passes** its register if self-BLEU lands *near* that register's real value (not far above) and distinct-n/entropy land *near* (not far below), at n=300. Rough fail: self-BLEU-4 ≳ 2× the register target. Gemini + MiMo are the primary gate targets (single-lineage, high volume).

## Caveats
- **self-BLEU is length-sensitive.** Conversational docs are shorter (323 vs 821 tok), which inflates conversational self-BLEU somewhat independent of register. So gate conversational synthetic **at matched doc length** (~300–350 tok / ~50 dialogue lines) and n=300, or the comparison drifts.
- **self-BLEU is reference-set-size sensitive** — n=300 is locked; compare synthetic at exactly n=300.
- **Conversational v0 = subtitles = *translated* dialogue** → translationese-skewed. It anchors the conversational *distribution*, but the authentic-original-Serbian target (forum/human) may sit differently. Refine when forum/comment data is staged.
- Reproduce: `python scripts/diversity_baseline.py --input <one-doc-per-line>.txt --n 300`.
