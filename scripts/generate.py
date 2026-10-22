#!/usr/bin/env python3
"""Load a LittleBit-compressed model locally and generate text (real inference path).

Uses the official loader (packed binary weights -> binarized inference mode).

Example:
  python scripts/generate.py --model_dir out/opt1.3b-0.3-qat --prompt "The capital of France is" \
      --max_new_tokens 48
"""
import argparse
import json
import os
import resource
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "repo")
sys.path.insert(0, REPO)

from quantization.hub import LittleBitModel  # noqa: E402
from quantization.modules import LittleBitLinear  # noqa: E402
from utils.datautils import load_tokenizer  # noqa: E402


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "y", "1")


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def dir_size_mb(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                tot += os.path.getsize(fp)
    return tot / 1024 ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--base_model", default=None, help="tokenizer source (defaults to model config)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--report_json", default=None)
    ns = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if ns.device == "auto" else ns.device
    dtype = getattr(torch, ns.dtype)
    size_mb = dir_size_mb(ns.model_dir)
    print(f"INFO: model_dir={ns.model_dir} on-disk={size_mb:.1f} MB device={device} dtype={dtype}", flush=True)

    t0 = time.time()
    wrapper = LittleBitModel.from_pretrained(ns.model_dir, torch_dtype=dtype, device=device)
    load_s = time.time() - t0
    model = wrapper.model
    model.eval()
    n_binarized = sum(1 for m in model.modules() if isinstance(m, LittleBitLinear) and m._binarized)
    n_lb = sum(1 for m in model.modules() if isinstance(m, LittleBitLinear))
    print(f"INFO: loaded in {load_s:.1f}s | LittleBitLinear modules={n_lb} binarized={n_binarized}", flush=True)
    print(f"INFO: peak RSS={rss_gb():.2f} GB", flush=True)

    tok_path = ns.base_model or ns.model_dir
    try:
        tokenizer = load_tokenizer(tok_path)
    except Exception:
        cfg = json.load(open(os.path.join(ns.model_dir, "config.json")))
        tokenizer = load_tokenizer(cfg.get("_name_or_path", ns.base_model))

    prompts = [ns.prompt]
    if ns.prompts_file:
        prompts = [l.strip() for l in open(ns.prompts_file) if l.strip()]

    results = []
    for p in prompts:
        text = p if p.endswith((" ", "\n")) else p + " "
        enc = tokenizer(text, return_tensors="pt").to(device)
        gen_kw = dict(max_new_tokens=ns.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        t0 = time.time()
        out = model.generate(**enc, **gen_kw)
        dt = time.time() - t0
        n_new = out.shape[1] - enc["input_ids"].shape[1]
        full = tokenizer.decode(out[0], skip_special_tokens=True)
        completed = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        tps = n_new / dt
        print(f"\n>>> PROMPT: {p!r}\n>>> COMPLETION: {completed!r}\n"
              f">>> {n_new} tokens in {dt:.2f}s = {tps:.2f} tok/s", flush=True)
        results.append({"prompt": p, "completion": completed, "tokens": int(n_new),
                        "seconds": round(dt, 3), "tokens_per_s": round(tps, 3)})

    # warm timing repeats for a stable tok/s estimate
    if ns.repeats and prompts:
        enc = tokenizer(prompts[0] if prompts[0].endswith(" ") else prompts[0] + " ", return_tensors="pt").to(device)
        times = []
        for _ in range(ns.repeats):
            t0 = time.time()
            out = model.generate(**enc, max_new_tokens=ns.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            times.append(time.time() - t0)
            n_new = out.shape[1] - enc["input_ids"].shape[1]
        best = min(times)
        print(f"INFO: best of {ns.repeats}: {n_new} tokens in {best:.2f}s = {n_new/best:.2f} tok/s", flush=True)

    peak = rss_gb()
    print(f"INFO: peak RSS={peak:.2f} GB | on-disk={size_mb:.1f} MB", flush=True)
    if ns.report_json:
        with open(ns.report_json, "w") as f:
            json.dump({"model_dir": ns.model_dir, "on_disk_mb": round(size_mb, 2), "load_seconds": round(load_s, 2),
                       "peak_rss_gb": round(peak, 3), "binarized_modules": n_binarized,
                       "littlebit_modules": n_lb, "generations": results}, f, indent=2)
        print(f"INFO: report -> {ns.report_json}", flush=True)


if __name__ == "__main__":
    main()
