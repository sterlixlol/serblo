"""Training loop (P2 debug / P3 full): AdamW + linear-warmup cosine LR + grad-accum + bf16 +
ATOMIC RESUMABLE checkpoints (model+optim+step+tokens+full RNG) to local (+ optional GCS) every N
steps and on SIGTERM (spot preemption) + val-plateau stop.

Helpers (lr_at / save_checkpoint / load_checkpoint / train_step) are pure enough to unit-test;
the debug run integration-tests the whole thing. See docs plan §P2/§P3.
"""
import argparse
import json
import math
import os
import pathlib
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from serblo.train.data import ShardLoader          # noqa: E402
from serblo.train.model import GPT, GPTConfig       # noqa: E402


def lr_at(step, warmup, total, peak, floor_frac=0.1):
    """Linear warmup then cosine decay from peak to peak*floor_frac; clamped to the floor past total."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step >= total:
        return peak * floor_frac
    prog = (step - warmup) / max(1, total - warmup)          # 0 -> 1 across the decay
    return peak * floor_frac + 0.5 * (1 - floor_frac) * peak * (1 + math.cos(math.pi * prog))


def _device_type(device):
    return "cuda" if str(device).startswith("cuda") else "cpu"


def save_checkpoint(path, model, opt, step, tokens, best_val, cfg_dict, bad_evals=0):
    """Atomically save everything needed to resume bit-exactly (incl. all RNG state)."""
    ck = {
        "model": model.state_dict(), "optim": opt.state_dict(),
        "step": step, "tokens": tokens, "best_val": best_val, "bad_evals": bad_evals,
        "config": cfg_dict,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(), "py_rng": random.getstate(),
    }
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(ck, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)                                    # atomic
    return path


def load_checkpoint(path, model, opt=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"])
    if opt is not None:
        opt.load_state_dict(ck["optim"])
    torch.set_rng_state(ck["torch_rng"].cpu())                  # RNG state MUST be a CPU ByteTensor
    if ck.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
    np.random.set_state(ck["numpy_rng"])
    random.setstate(ck["py_rng"])
    return ck


def train_step(model, opt, loader, micro_bs, accum, device, bf16, grad_clip, lr):
    """One optimizer step over `accum` micro-batches. Returns (mean loss, grad norm)."""
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(accum):
        x, y = loader.get_batch(micro_bs, device)
        with torch.autocast(device_type=_device_type(device), dtype=torch.bfloat16, enabled=bf16):
            _, loss = model(x, y)
        (loss / accum).backward()                           # scale so grad = mean over accum
        total += loss.item() / accum
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return total, float(gn)


@torch.no_grad()
def evaluate(model, loader, micro_bs, device, bf16, iters=20, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)                   # fixed windows: comparable + resume-invariant
    losses = []
    for _ in range(iters):
        x, y = loader.get_batch(micro_bs, device, generator=gen)
        with torch.autocast(device_type=_device_type(device), dtype=torch.bfloat16, enabled=bf16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _gcs_push(local, gcs_prefix):
    """Copy a file to GCS. Returns True on success; loud WARN on failure (spot boxes rely on this
    being the real persistence; a silent failure = nothing uploaded after preemption)."""
    if not gcs_prefix:
        return True
    r = subprocess.run(["gcloud", "storage", "cp", str(local), gcs_prefix.rstrip("/") + "/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [WARN] GCS push of {local} failed: {r.stderr.strip()[:200]}", flush=True)
    return r.returncode == 0


def run(cfg):
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))
    bf16 = cfg.get("bf16", True) and _device_type(device) == "cuda"
    out = pathlib.Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    gcs_prefix = cfg.get("gcs_prefix")

    gcfg = GPTConfig(**cfg["model"])
    meta = json.loads((pathlib.Path(cfg["data_dir"]) / "meta.json").read_text())
    assert meta.get("vocab_size") == gcfg.vocab_size, (          # guard the stale-tokenizer foot-gun
        f"shard vocab {meta.get('vocab_size')} != model vocab {gcfg.vocab_size} — repack with the current tokenizer")

    if gcs_prefix:                                               # fail fast if the spot-box persistence is unreachable
        marker = out / ".gcs_ok"
        marker.write_text("ok")
        if not _gcs_push(marker, gcs_prefix):
            raise SystemExit("GCS unreachable -> refusing to start (checkpoints wouldn't survive preemption)")

    model = GPT(gcfg).to(device)
    opt = model.configure_optimizers(cfg["weight_decay"], cfg["peak_lr"], tuple(cfg["betas"]), device)
    train_loader = ShardLoader(cfg["data_dir"], gcfg.ctx, "train")
    val_loader = ShardLoader(cfg["data_dir"], gcfg.ctx, "val")

    micro_bs, accum = cfg["micro_batch"], cfg["grad_accum"]
    tokens_per_step = micro_bs * accum * gcfg.ctx
    max_steps, warmup = cfg["max_steps"], cfg["warmup_steps"]
    ckpt_int = cfg["ckpt_interval"]
    snap_int = cfg.get("snapshot_interval", ckpt_int * 10)       # also keep periodic immortal snapshots
    step, tokens, best_val, bad_evals = 0, 0, float("inf"), 0

    latest = out / "ckpt_latest.pt"
    if latest.exists():                                         # resume: load on CPU (RNG restore needs CPU tensors)
        ck = load_checkpoint(latest, model, opt, map_location="cpu")
        step, tokens, best_val = ck["step"], ck["tokens"], ck["best_val"]
        bad_evals = ck.get("bad_evals", 0)                       # older checkpoints lack this key
        print(f"resumed from step {step} ({tokens:,} tokens)", flush=True)

    compiled = torch.compile(model) if cfg.get("compile") else model   # wrap forward only; ckpt/opt use `model`

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))   # spot preemption -> ckpt+exit
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    def checkpoint():
        save_checkpoint(latest, model, opt, step, tokens, best_val, cfg["model"], bad_evals)
        _gcs_push(latest, gcs_prefix)
        if step > 0 and step % snap_int == 0:                   # immortal snapshot (NaN-proof: latest could be bad)
            snap = out / f"ckpt_step{step}.pt"
            save_checkpoint(snap, model, opt, step, tokens, best_val, cfg["model"], bad_evals)
            _gcs_push(snap, gcs_prefix)

    print(f"training: {model.num_params():,} params, {tokens_per_step:,} tok/step, "
          f"target {max_steps} steps, device {device}, bf16 {bf16}", flush=True)
    model.train()
    t0, tok0 = time.time(), tokens
    while step < max_steps and not stop["flag"]:
        lr = lr_at(step, warmup, max_steps, cfg["peak_lr"], cfg.get("lr_floor_frac", 0.1))
        loss, gnorm = train_step(compiled, opt, train_loader, micro_bs, accum, device, bf16,
                                 cfg["grad_clip"], lr)
        if not math.isfinite(loss):                             # stop WITHOUT saving -> last good ckpt survives
            print(f"[FATAL] non-finite loss {loss} at step {step + 1} — stopping, NOT overwriting the checkpoint",
                  flush=True)
            sys.exit(1)
        step += 1
        tokens += tokens_per_step

        if step % cfg["log_interval"] == 0:
            dt = time.time() - t0
            tps = (tokens - tok0) / dt if dt > 0 else 0
            print(f"step {step:6d} | loss {loss:.4f} | lr {lr:.2e} | gnorm {gnorm:.2f} | "
                  f"{tps:,.0f} tok/s", flush=True)
            t0, tok0 = time.time(), tokens

        if step % cfg["eval_interval"] == 0 and not stop["flag"]:   # skip eval on the SIGTERM exit path
            vloss = evaluate(compiled, val_loader, micro_bs, device, bf16, cfg.get("eval_iters", 20))
            improved = vloss < best_val - cfg.get("plateau_min_delta", 1e-3)
            best_val = min(best_val, vloss)
            bad_evals = 0 if improved else bad_evals + 1
            print(f"  [eval] step {step} | val_loss {vloss:.4f} | best {best_val:.4f} | "
                  f"bad_evals {bad_evals}", flush=True)
            if cfg.get("patience") and bad_evals >= cfg["patience"]:
                print(f"  val plateau ({bad_evals} evals w/o improvement) -> stopping", flush=True)
                break

        if step % ckpt_int == 0:
            checkpoint()

    checkpoint()                                                # final / on stop
    print(f"done: step {step}, {tokens:,} tokens, best_val {best_val:.4f}"
          f"{' (SIGTERM)' if stop['flag'] else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    run(cfg)


if __name__ == "__main__":
    main()
