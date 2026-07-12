# MiMo conversational — diversity gate v0 (2026-07-02)

First real synthetic batch through the gate. **Source:** `mimo-v2.5-free` via OpenCode,
315 conv docs / 118,779 words (`data/generated/mimo/docs.jsonl`), from the 20k-conv
manifest (seed 0). Gate = `scripts/diversity_baseline.py`, n=300, same script both sides,
same session (avoids the sampling drift vs the v0 baseline doc).

## Verdict: PASS (phrase-level). Not mode-collapsed.

| metric (n=300) | MiMo | real subtitles | read |
|---|---|---|---|
| self-BLEU-2 | 0.725 | 0.709 | even |
| self-BLEU-3 | 0.496 | 0.500 | even |
| **self-BLEU-4** | **0.307** | 0.334 | MiMo *slightly more diverse* (lower=better) |
| **distinct-1** | **0.086** | 0.135 | MiMo LOWER — the particle skew (below) |
| distinct-2 | 0.435 | 0.505 | ~even |
| distinct-3 | 0.729 | 0.799 | ~even |
| entropy-1 norm | 0.70 | 0.669 | even |

MiMo docs are *longer* (438 vs 327 tok/doc), which biases self-BLEU up, yet it still
comes in below real — the diversity is real, not a length artifact. MiMo ships as the
conversational co-anchor as planned.

## The particle prime (found by eyeball, then measured)

The fixed example list in `conversational_base` — "particles (pa, ma, bre, ej, jbg)" —
was copied verbatim. Rate per 1,000 words, MiMo vs real subtitles:

| particle | in prompt? | MiMo/1k | real/1k | ratio |
|---|---|---|---|---|
| bre | primed | 10.34 | 0.02 | 540x |
| ma | primed | 17.36 | 0.32 | 54x |
| jbg | primed | 7.67 | 0.00 | inf |
| ej | primed | 5.85 | 0.07 | 79x |
| pa | primed | 16.59 | 2.68 | 6x |
| **hej** | NOT primed | **0.01** | 0.61 | 0x |
| joj | NOT primed | 0.34 | 0.67 | 1x |

Smoking gun: "ej" (primed) 79x too high, but "hej" (its unprimed synonym) ~0. Copying the
example list, not a voice. **self-BLEU is blind to this** (phrase overlap barely moves on a
unigram tic); it surfaced only in **distinct-1** (0.086 vs 0.135) and by manual reading.

**Corpus impact:** these 315 docs = 0.00055% of the 30B corpus. Even if MiMo-casual grew
to 1% of corpus: ej-rate 0.20→0.26/1k (+28%); at the realistic garnish share, +8%.
Negligible — kept as-is, not regenerated.

## Fixes

1. **Done** — particles are now a rotating 2-of-18 sample per prompt (`conversational_particles`
   pool + `manifest.particles_for`). Each particle now primes ~10-14% of prompts (was 100%).
   Applies to all future (bulk) generation; deterministic/reproducible per id.
2. **TODO gate upgrade** — add a unigram / function-word frequency check (KL vs real) as a
   companion to self-BLEU, so a token-distribution tic is caught automatically next time
   instead of by eyeball.
