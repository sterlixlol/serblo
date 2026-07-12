# Serblo-125M 🇷🇸

[![Base model](https://img.shields.io/badge/🤗-serblo--125m-ffd21e)](https://huggingface.co/sterlixlol/serblo-125m)
[![Instruct](https://img.shields.io/badge/🤗-serblo--125m--instruct-ffd21e)](https://huggingface.co/sterlixlol/serblo-125m-instruct)
[![SFT data](https://img.shields.io/badge/🤗-serblo--sft-blue)](https://huggingface.co/datasets/sterlixlol/serblo-sft)
[![Preference data](https://img.shields.io/badge/🤗-serblo--prefs-blue)](https://huggingface.co/datasets/sterlixlol/serblo-prefs)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

This is the full pipeline and training code behind Serblo, a 125M-parameter Bosnian-Croatian-Serbian language model built from scratch: corpus acquisition, cleaning, dedup, tokenizer, pretraining, SFT, preference data and DPO. Everything the released artifacts came from.

```
>>> Kako se zoveš?
Zovem se Serblo. To je srpski jezički model...

>>> Gde si bio juče, brate?
Pa dobro sam sad, kod tebe se živi! Evo me tu gde jesam...
```

## Layout

| path | what lives there |
|---|---|
| `src/serblo/acquire/` | corpus acquisition: FineWeb-2 BCS, OpenSubtitles, srWaC, literature |
| `src/serblo/pipeline/` | normalization, transliteration, mojibake repair, decluttering |
| `src/serblo/dedup/` | MinHash dedup, including a streaming version for 28M+ docs |
| `src/serblo/generate/` | multi-model synthetic generation: API + CLI drivers, diversity gates |
| `src/serblo/tokenizer/` | 32k byte-level BPE training |
| `src/serblo/train/` | model, pretraining loop, SFT trainer, DPO trainer (pure PyTorch) |
| `src/serblo/eval/` | decontamination |
| `scripts/` | runnable stages, sampling, chat REPL, eval tooling |
| `configs/` | tokenizer, debug run, H100 pretraining, SFT, DPO |
| `docs/samples.md` | what the released models actually generate |
| `results/` | diversity-gate and eval reports produced during the build |

Everything runs with `PYTHONPATH=src`. Data directories are gitignored, the pipeline recreates them.

## Reproducing the stages

Each stage is a script. In build order:

```bash
# corpus: acquire, clean, dedup
python3 -m serblo.acquire.run_fineweb_scale --target-docs 0
python3 -m serblo.dedup.run_dedup_scale --interim

# tokenizer (32k BPE, reserves chat special tokens)
python3 scripts/train_tokenizer.py

# pack to uint16 shards, then pretrain
python3 scripts/pack_corpus_fast.py
python3 -m serblo.train.train --config configs/train_h100.yaml

# post-training: SFT, preference pairs, judge, DPO
python3 -m serblo.train.sft --config configs/sft.yaml
python3 scripts/gen_prefs.py --ckpt data/sft/ckpts/sft_final.pt
python3 scripts/judge_prefs.py --samples data/dpo/samples.jsonl
python3 -m serblo.train.dpo --config configs/dpo.yaml

# blind eval (SFT vs DPO, three judges)
python3 scripts/gen_eval_answers.py && python3 scripts/judge_eval.py
```

The synthetic generation stage (`src/serblo/generate/`) drives multiple model APIs and CLIs with per-lineage output dirs and diversity gating. It needs API keys in `.env` (gitignored), see the host table in `run_api.py`.

Pretraining ran on one Spot H100 through Vertex AI with checkpoint-to-GCS every 1,000 steps and an auto-resubmitting supervisor (`scripts/vertex_supervisor.sh`). The training story, including the incidents, is in the model card. `scripts/watch_training.py` is the live dashboard that watched it.

## Chatting with the model

`scripts/chat.py` is a small streaming REPL for checkpoints produced by this pipeline. For the released Hugging Face weights, use the loading snippet from the model card (safetensors with tied embeddings) or convert them to a checkpoint once and point chat.py at it.

## Tests

```bash
PYTHONPATH=src python3 -m pytest tests/
```

201 tests cover the mask logic, packing, trainers, dedup parity, converters and the preference scripts.

## Notes

This is working code from a real 10-day build, not a polished library. Some scripts assume GCS paths or specific local dirs, all of them are short enough to read before running. The corpus itself is not in this repo: FineWeb-2 is public, the pipeline rebuilds the rest.

Apache-2.0. Built by Stefan Selakov ([@sterlixlol](https://huggingface.co/sterlixlol)).
