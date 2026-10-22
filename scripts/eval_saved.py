#!/usr/bin/env python3
"""Evaluate a saved LittleBit checkpoint (packed binary weights) on PPL + a generation smoke test."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.join(ROOT, "repo")
sys.path.insert(0, HERE)
from lbcommon import decide_device_env, preimport_model_module  # noqa: E402

decide_device_env()  # before `import torch`

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, REPO)

from quantization.hub import LittleBitModel  # noqa: E402
from quantization.modules import LittleBitLinear  # noqa: E402
from utils.datautils import get_eval_loaders, load_tokenizer  # noqa: E402


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "y", "1")


def dir_size_mb(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            tot += os.path.getsize(os.path.join(root, f))
    return tot / 1024 ** 2


@torch.no_grad()
def ppl(model, tokenizer, dataset, seqlen, device, max_chunks=0):
    testenc = get_eval_loaders(dataset, tokenizer).input_ids
    nsamples = testenc.numel() // seqlen
    if max_chunks and max_chunks < nsamples:
        nsamples = max_chunks
    nlls = []
    model.eval()
    t0 = time.time()
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)
        logits = model(batch, use_cache=False).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * (seqlen - 1))
        dt = time.time() - t0
        print(f"  chunk {i}/{nsamples} ({dt:.0f}s, {dt/(i+1):.1f}s/chunk, "
              f"{seqlen*(i+1)/dt:.1f} tok/s)", flush=True)
    return torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1))).item(), nsamples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_from", default=None)
    ap.add_argument("--ppl_task", default="wikitext2")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--gen_prompt", default="The capital of France is")
    ap.add_argument("--gen_tokens", type=int, default=24)
    ap.add_argument("--max_chunks", type=int, default=0, help="0 = full test set; else first N chunks of 2048 tokens")
    ap.add_argument("--report_json", default=None)
    ns = ap.parse_args()

    device = ns.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dtype = getattr(torch, ns.dtype)
    size_mb = dir_size_mb(ns.model_dir)
    print(f"INFO: {ns.model_dir} on-disk={size_mb:.1f} MB device={device} dtype={ns.dtype}", flush=True)

    t0 = time.time()
    preimport_model_module(ns.model_dir)
    wrapper = LittleBitModel.from_pretrained(ns.model_dir, torch_dtype=dtype, device=device)
    print(f"INFO: loaded in {time.time()-t0:.1f}s", flush=True)
    model = wrapper.model
    model.eval()
    if ns.dtype != "auto":
        model = model.to(dtype)  # also casts the BF16 buffers/params saved in the checkpoint
        print(f"INFO: cast model to {ns.dtype}", flush=True)

    n_lb = sum(1 for m in model.modules() if isinstance(m, LittleBitLinear))
    n_bin = sum(1 for m in model.modules() if isinstance(m, LittleBitLinear) and m._binarized)
    packed_params = 0
    scale_params = 0
    for m in model.modules():
        if isinstance(m, LittleBitLinear):
            for k, v in m.state_dict().items():
                if k.endswith("_packed"):
                    packed_params += v.numel() * v.element_size()
                else:
                    scale_params += v.numel() * v.element_size()
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    print(f"INFO: LittleBitLinear={n_lb} binarized={n_bin} | binary(stored)={packed_params/1e6:.2f} MB "
          f"scales={scale_params/1e6:.2f} MB | param dtypes={dtypes}", flush=True)

    tokenizer = load_tokenizer(ns.tokenizer_from or ns.model_dir)
    result = {"model_dir": ns.model_dir, "size_mb": round(size_mb, 2), "binarized": n_bin,
              "packed_mb": round(packed_params / 1e6, 3), "scales_mb": round(scale_params / 1e6, 3),
              "device": device, "dtype": ns.dtype}

    for ds in [d for d in ns.ppl_task.split(",") if d]:
        t0 = time.time()
        val, n_used = ppl(model, tokenizer, ds, ns.seqlen, device, ns.max_chunks)
        print(f"[{ds}] PPL = {val:.4f} ({time.time()-t0:.0f}s, {n_used} chunks)", flush=True)
        result[f"ppl_{ds}"] = val
        result[f"ppl_{ds}_chunks"] = n_used

    if ns.gen_tokens > 0:
        enc = tokenizer(ns.gen_prompt, return_tensors="pt").to(device)
        t0 = time.time()
        out = model.generate(**enc, max_new_tokens=ns.gen_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        dt = time.time() - t0
        n_new = out.shape[1] - enc["input_ids"].shape[1]
        comp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n>>> PROMPT: {ns.gen_prompt!r}\n>>> COMPLETION: {comp!r}\n>>> {n_new} tok in {dt:.2f}s "
              f"= {n_new/max(dt,1e-6):.2f} tok/s", flush=True)
        result["generation"] = {"prompt": ns.gen_prompt, "completion": comp,
                               "tokens_per_s": round(n_new / max(dt, 1e-6), 3)}

    if ns.report_json:
        with open(ns.report_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"INFO: report -> {ns.report_json}", flush=True)


if __name__ == "__main__":
    main()
