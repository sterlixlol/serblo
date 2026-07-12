# Real-data declutter (chrome stripping) — M2 cleaning, 2026-07-03

## Origin
Stefan's observation: clean synthetic conversational ("@handle ...") is better training
data than raw scraped Reddit with its `Upvote/Downvote/Reply/3h ago/avatar` chrome. This
prompted checking how much chrome sits in our REAL backbone.

## Two hypotheses, both measured, both flipped from intuition
1. First crude detector (any `|`/`[` = polluted) → "43% of FineWeb-2 polluted." **Inflated** —
   it flagged legit `[1]` citations, inline URLs, and the *word* "kategorije".
2. Intuition: chrome inflates real's apparent diversity. **Backwards** — chrome is SHARED
   boilerplate, so it HOMOGENIZES (any two chromed docs look alike through it). Cleaning made
   real MORE diverse, not less.

## What declutter actually is (`serblo.pipeline.declutter`)
Surgical, because the flagged docs are ~95% good prose lightly contaminated (the wiki
"uredi"/"cite" fixtures are solid articles with `[uredi]`/`[1]` markers). Strips:
- inline: `[uredi]`/`[уреди]` tags, `[N]` citations, URLs (`http`, `www.`);
- chrome lines: pipe-prefixed wiki template/table rows, `Sadržaj` TOC, forum edit-logs,
  separator/ascii-art lines, nav, Reddit chrome (`u/…`, `Upvote`, `3h ago`);
- boilerplate phrases: "Pomozite Wikipediji", "nije dovoljno potkrijepljena", edit-logs.
Keeps prose + diacritics + section titles. Caller drops a doc only if survival < 0.5 or
< 20 words. Runs BEFORE dedup (order matters, like transliteration).

## At-scale result (the whole real backbone)
| source | docs | kept | prose words retained | dropped |
|---|---|---|---|---|
| fineweb2 | 80,000 | 99.9% | 99.5% | 58 (50 mostly-chrome, 8 short) |
| opensubtitles | 60,000 | 100% | 99.9% | 0 |
| literature | 24,390 | 100% | 98.9% | 6 short |
| forum | 38 | 100% | 99.5% | 0 |

FineWeb-2 was ~97.4% pristine (77,921/80,000 had ZERO chrome). The corpus was cleaner than
the crude number implied — but it is now *precisely* chrome-free, and the clean-source pass
proves declutter doesn't over-strip (subtitles untouched at 99.9%).

## Gate re-baseline (essay, n=300)
| reference | self-BLEU-4 | distinct-1 |
|---|---|---|
| raw real | 0.144 | 0.176 |
| **decluttered real (surgical)** | **0.137** | **0.191** |
| grounded synthetic | 0.243 | 0.171 |

Cleaning made real slightly MORE diverse (chrome was mildly homogenizing — confirmed, small
because contamination was ~2.6% of docs). The synth↔real structural gap is genuine and if
anything wider against clean real. **Essay diversity conclusion stands: stop tuning it.**

## Pipeline consequence
Canonical flow is now acquire → **declutter** → dedup → ... . Point dedup + gate baselines
at `data/interim-clean/` going forward (`run_dedup --interim data/interim-clean`). The value
prop of the deliverable — a CLEAN released BCS dataset — is now real, not aspirational.
