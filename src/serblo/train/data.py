"""Training data loader: random (x, y) context windows over memmapped uint16 token shards.

Per-batch: pick shards weighted by size, then a random offset in each -> a [ctx] window; y is x
shifted by one. Memmap => never loads a whole shard into RAM.
"""
import json
import pathlib

import numpy as np
import torch


class ShardLoader:
    def __init__(self, tokenized_dir, ctx, split="train"):
        d = pathlib.Path(tokenized_dir)
        meta = json.loads((d / "meta.json").read_text())
        self.ctx = ctx
        self.mmaps, sizes = [], []
        for name in meta[f"{split}_shards"]:
            m = np.memmap(d / name, dtype=np.uint16, mode="r")
            if len(m) >= ctx + 1:                       # a shard must fit at least one window
                self.mmaps.append(m)
                sizes.append(len(m))
        if not self.mmaps:
            raise ValueError(f"no {split} shard with >= {ctx + 1} tokens in {d}")
        self.sizes = np.asarray(sizes, dtype=np.float64)
        self.weights = torch.as_tensor(self.sizes / self.sizes.sum())
        self.total_tokens = int(self.sizes.sum())

    def get_batch(self, batch_size, device, generator=None):
        shard_ids = torch.multinomial(self.weights, batch_size, replacement=True, generator=generator)
        x = torch.empty(batch_size, self.ctx, dtype=torch.long)
        y = torch.empty(batch_size, self.ctx, dtype=torch.long)
        for b, si in enumerate(shard_ids.tolist()):
            m = self.mmaps[si]
            hi = len(m) - self.ctx - 1                  # inclusive max start offset
            off = int(torch.randint(0, hi + 1, (1,), generator=generator).item())
            chunk = torch.from_numpy(np.asarray(m[off:off + self.ctx + 1], dtype=np.int64))
            x[b] = chunk[:-1]
            y[b] = chunk[1:]
        if str(device).startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
