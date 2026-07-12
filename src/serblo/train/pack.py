"""Pack tokenized documents into flat uint16 training shards + a val split.

Docs are token-separated by ``eos_id``. Writes ``train_NNNNN.bin`` / ``val_NNNNN.bin`` (raw
little-endian uint16) + ``meta.json``. Every write is atomic (.part -> fsync -> os.replace).

Resumable: ``pack_progress.json`` records ``docs_done``, advanced ONLY at a
completed shard flush, so a crash mid-shard re-processes the in-flight docs on resume with no loss
and no duplication. Vocab must be < 65536 for uint16 (asserted).
"""
import json
import os
import pathlib

import numpy as np


def _atomic_write_u16(path, ids):
    arr = np.asarray(ids, dtype=np.uint16)
    tmp = str(path) + ".part"
    with open(tmp, "wb") as f:
        f.write(arr.tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path, obj):
    tmp = str(path) + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def pack_docs(docs, encode_fn, out_dir, eos_id, shard_tokens=100_000_000, val_every=200,
              tokenizer_sha256="", vocab_size=None, resume=True):
    """Stream ``docs`` (iterable of str) through ``encode_fn(str)->list[int]`` into uint16 shards.

    Every ``val_every``-th doc (by global index) goes to the val split. Returns the meta dict.
    """
    assert 0 <= eos_id < 65536, "eos_id must fit in uint16"
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prog = out / "pack_progress.json"

    st = {"docs_done": 0, "shards": 0, "train_tokens": 0, "val_tokens": 0}
    if resume and prog.exists():
        st = json.loads(prog.read_text())
    committed = st["docs_done"]
    train_buf, val_buf = [], []

    def flush(upto_doc):
        # a shard "slot" always advances together for train+val so their file indices stay aligned
        if train_buf:
            _atomic_write_u16(out / f"train_{st['shards']:05d}.bin", train_buf)
            st["train_tokens"] += len(train_buf)
        if val_buf:
            _atomic_write_u16(out / f"val_{st['shards']:05d}.bin", val_buf)
            st["val_tokens"] += len(val_buf)
        st["shards"] += 1
        st["docs_done"] = upto_doc
        _atomic_write_json(prog, st)          # progress AFTER the data is durably on disk
        train_buf.clear()
        val_buf.clear()

    processed_upto = committed
    for i, doc in enumerate(docs):
        if i < committed:                     # resume: skip docs already in flushed shards
            continue
        ids = encode_fn(doc)
        if ids:
            assert max(ids) < 65536, "token id exceeds uint16 range"
        ids = list(ids) + [eos_id]
        # every val_every-th doc -> val; use (val_every-1) so doc 0 is NOT special-cased to val
        (val_buf if i % val_every == val_every - 1 else train_buf).extend(ids)
        processed_upto = i + 1
        if len(train_buf) >= shard_tokens:
            flush(processed_upto)
    if train_buf or val_buf:
        flush(processed_upto)

    meta = {
        "vocab_size": vocab_size, "eos_id": eos_id, "shard_dtype": "uint16",
        "train_tokens": st["train_tokens"], "val_tokens": st["val_tokens"],
        "shards": st["shards"], "tokenizer_sha256": tokenizer_sha256,
        "train_shards": sorted(p.name for p in out.glob("train_*.bin")),
        "val_shards": sorted(p.name for p in out.glob("val_*.bin")),
    }
    _atomic_write_json(out / "meta.json", meta)
    return meta
