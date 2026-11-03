#!/usr/bin/env python3
"""Prepare a bounded C4 (+WikiText-2) QAT dataset with the *upstream* preprocessing.

The upstream `get_c4_wiki_train` tokenizes the whole `c4-train.00000-of-01024.json.gz`
shard (~350 M tokens, 1-2 h on CPU). Here we stream the same shard and materialise only
`--max_texts` documents, then call the upstream function unchanged, so the tokenization /
grouping (2048-token blocks, no separator between documents) is byte-for-byte the same.

Usage:
  python scripts/prepare_c4.py --model_id models/opt-1.3b --max_texts 80000 \
      --out data/c4_wiki_80k
"""
import argparse
import itertools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.join(ROOT, "repo")
sys.path.insert(0, REPO)

import datasets  # noqa: E402
from utils import datautils  # noqa: E402

_orig_load_dataset = datasets.load_dataset


def make_patched_load(max_texts):
    def patched_load(path, *args, **kwargs):
        if path == "allenai/c4":
            print(f"INFO: streaming c4, taking {max_texts} documents", flush=True)
            stream_kwargs = {k: v for k, v in kwargs.items() if k in ("data_files", "cache_dir", "token", "revision")}
            stream_kwargs["streaming"] = True
            stream = _orig_load_dataset(path, *args, **stream_kwargs)
            split = "train" if "train" in stream else list(stream.keys())[0]
            rows = []
            t0 = time.time()
            for i, ex in enumerate(itertools.islice(stream[split], max_texts)):
                rows.append(ex["text"])
                if (i + 1) % 10000 == 0:
                    print(f"  {i+1}/{max_texts} docs ({time.time()-t0:.0f}s, "
                          f"{sum(len(r) for r in rows)/1e6:.0f} Mchars)", flush=True)
            print(f"INFO: materialised {len(rows)} docs, {sum(len(r) for r in rows)/1e6:.1f} Mchars", flush=True)
            return datasets.DatasetDict({"train": datasets.Dataset.from_dict({"text": rows})})
        return _orig_load_dataset(path, *args, **kwargs)
    return patched_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=os.path.join(ROOT, "models/opt-1.3b"))
    ap.add_argument("--max_texts", type=int, default=80000)
    ap.add_argument("--out", default=os.path.join(ROOT, "data/c4_wiki_80k"))
    ns = ap.parse_args()

    datautils.load_dataset = make_patched_load(ns.max_texts)
    tokenizer = datautils.load_tokenizer(ns.model_id)

    t0 = time.time()
    print("INFO: tokenizing + grouping with upstream get_c4_wiki_train ...", flush=True)
    ds = datautils.get_c4_wiki_train(tokenizer)
    train_ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    print(f"INFO: done in {time.time()-t0:.0f}s | train rows={len(train_ds)} "
          f"| columns={train_ds.column_names}", flush=True)

    os.makedirs(os.path.dirname(ns.out), exist_ok=True)
    if os.path.isdir(ns.out):
        import shutil
        shutil.rmtree(ns.out)
    datasets.DatasetDict({"train": train_ds}).save_to_disk(ns.out)
    n_tokens = len(train_ds) * 2048
    print(f"INFO: saved {ns.out} | {len(train_ds)} blocks x 2048 = {n_tokens/1e6:.1f} M tokens", flush=True)
    print("PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
