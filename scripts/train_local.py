#!/usr/bin/env python3
"""Local (no DeepSpeed) LittleBit QAT driver: Dual-SVID init + KD with SmoothSign.

Adapts the official main.py so it runs on a single GPU or on CPU without DeepSpeed/GPUtil.

Example:
  CUDA_VISIBLE_DEVICES="" python scripts/train_local.py \
      --model_id models/opt-1.3b --eff_bit 0.3 --residual True \
      --dataset wikitext2 --max_steps 200 --per_device_train_batch_size 2 \
      --lr 1.2e-4 --save_dir out/opt1.3b-0.3-qat
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.join(ROOT, "repo")
sys.path.insert(0, HERE)
from lbcommon import decide_device_env  # noqa: E402

decide_device_env()  # must run before `import torch` so CUDA_VISIBLE_DEVICES takes effect

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, REPO)

from quantization.utils import apply_littlebit_patch  # noqa: E402
from quantization.modules import LittleBitLinear  # noqa: E402
from utils.datautils import get_wikitext2_train, load_tokenizer, prepare_dataset  # noqa: E402
from utils.kd_utils import KDTrainer  # noqa: E402
from utils.utils import prepare_model_for_training, print_trainable_parameters  # noqa: E402
from transformers import AutoModelForCausalLM, TrainingArguments, default_data_collator, set_seed  # noqa: E402


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "y", "1")


def pick_device(requested, min_free_gb=6.0):
    """Avoid OOM when the GPU is busy with other workloads (e.g. LM Studio)."""
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception as e:  # GPU saturated / busy (e.g. another workload grabbed the device)
        print(f"WARN: CUDA probe failed ({e}) -> falling back to CPU", flush=True)
        return "cpu"
    if free < min_free_gb * 1024 ** 3:
        print(f"WARN: CUDA free {free/1024**3:.2f}GB < {min_free_gb}GB needed -> falling back to CPU", flush=True)
        return "cpu"
    return "cuda"


def rechunk(ds, seqlen):
    """Split the (2048-token) grouped dataset rows into shorter blocks.

    Shorter blocks give ~(2048/seqlen)x more optimizer steps for the same token budget and the
    same wall-clock cost per token (the KD teacher forward dominates), which matters a lot for
    short local QAT schedules.
    """
    import numpy as np
    keep = ["input_ids", "attention_mask", "labels"]
    cols = {k: [] for k in keep if k in ds.column_names}
    src = {k: ds[k] for k in cols}
    for i in range(len(ds)):
        n = len(src[list(cols)[0]][i])
        for start in range(0, n - seqlen + 1, seqlen):
            for k in cols:
                cols[k].append(list(src[k][i][start:start + seqlen]))
    import datasets as hfds
    return hfds.Dataset.from_dict(cols)


def build_quant_args(ns):
    return argparse.Namespace(
        quant_func=ns.quant_func, quant_mod="LittleBitLinear", residual=ns.residual, split_dim=ns.split_dim,
        eff_bit=ns.eff_bit, kv_factor=ns.kv_factor, min_split_dim=ns.min_split_dim, use_itq=ns.use_itq,
        itq_n_iter=ns.itq_n_iter, model_id=ns.model_id,
    )


def save_artifacts(model, tokenizer, save_dir, ns):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    model.config.use_cache = True
    quant_params = {
        "quant_func": ns.quant_func, "eff_bit": ns.eff_bit, "split_dim": ns.split_dim,
        "residual": bool(ns.residual), "kv_factor": ns.kv_factor, "min_split_dim": ns.min_split_dim,
        "quant_mod": "LittleBitLinear", "use_itq": bool(ns.use_itq), "itq_n_iter": ns.itq_n_iter,
    }
    if "save_dtype" in quant_params and False:
        pass
    with open(os.path.join(save_dir, "littlebit_config.json"), "w") as f:
        json.dump(quant_params, f, indent=2)
    for k, v in quant_params.items():
        setattr(model.config, k, v)
    sd = model.state_dict()
    if getattr(ns, "save_dtype", "bfloat16") == "bfloat16":
        for k, v in list(sd.items()):
            if "packed" not in k and "shape" not in k and v.dtype == torch.float32:
                sd[k] = v.to(torch.bfloat16)
    model.save_pretrained(save_dir, state_dict=sd, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    # stored size of the binary factors + fp scales (the compressed part)
    tot = 0
    for k, v in sd.items():
        if "packed" in k or "shape" in k or k.endswith(("u1", "u2", "v1", "v2", "u1_R", "u2_R", "v1_R", "v2_R", "bias")):
            tot += v.numel() * v.element_size()
    print(f"INFO: saved to {save_dir} (quantized part ~{tot/1e6:.2f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="models/opt-1.3b")
    ap.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4", "c4_wiki"])
    ap.add_argument("--dataset_path", default=None,
                    help="pre-built dataset dir (save_to_disk), e.g. data/c4_wiki_80k from prepare_c4.py")
    ap.add_argument("--data_root", default="./")
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--train_seqlen", type=int, default=0,
                    help="if >0, rechunk training blocks to this length (faster steps, more updates)")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--lr", type=float, default=1.2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.02)
    ap.add_argument("--l2l_loss_scale", type=float, default=10.0)
    ap.add_argument("--quant_func", default="SmoothSign")
    ap.add_argument("--eff_bit", type=float, default=0.3)
    ap.add_argument("--residual", type=str2bool, default=True)
    ap.add_argument("--split_dim", type=int, default=1024)
    ap.add_argument("--kv_factor", type=float, default=1.0)
    ap.add_argument("--min_split_dim", type=int, default=8)
    ap.add_argument("--use_itq", type=str2bool, default=False)
    ap.add_argument("--itq_n_iter", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    ap.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    ap.add_argument("--optim", default="adamw_torch",
                    choices=["adamw_torch", "paged_adamw_8bit", "adamw_bnb_8bit", "adafactor"],
                    help="optimizer (8-bit Adam keeps the FP32 Adam states off the 12 GB budget)")
    ap.add_argument("--save_dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="dtype of the non-binary tensors in the saved checkpoint (upstream uses bfloat16)")
    ap.add_argument("--report_json", default=None)
    ns = ap.parse_args()

    set_seed(ns.seed)
    device = pick_device(ns.device)
    if device == "cpu":
        # the upstream LittleBitLinear moves its SVD to cuda whenever torch.cuda.is_available(),
        # so hide the (fully occupied) GPU from the process when we run on CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if ns.dtype == "auto":
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = getattr(torch, ns.dtype)

    print(f"INFO: device={device} dtype={dtype} eff_bit={ns.eff_bit} residual={ns.residual}", flush=True)
    tokenizer = load_tokenizer(ns.model_id)

    if ns.dataset_path:
        import datasets as hfds
        ds = hfds.load_from_disk(ns.dataset_path)
        train_ds = ds["train"] if isinstance(ds, hfds.DatasetDict) else ds
    elif ns.dataset == "wikitext2":
        train_ds = get_wikitext2_train(tokenizer)
    else:
        train_ds = prepare_dataset(argparse.Namespace(data_root=ns.data_root, dataset=ns.dataset), tokenizer)
    if ns.max_samples and len(train_ds) > ns.max_samples:
        train_ds = train_ds.select(range(ns.max_samples))
    if ns.train_seqlen and ns.train_seqlen != ns.seqlen:
        train_ds = rechunk(train_ds, ns.train_seqlen)
    print(f"INFO: dataset={ns.dataset} samples={len(train_ds)} seqlen={ns.seqlen}", flush=True)

    student = AutoModelForCausalLM.from_pretrained(
        ns.model_id, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    student.config.use_cache = False
    if ns.gradient_checkpointing:
        prepare_model_for_training(student)
    student = apply_littlebit_patch(student, build_quant_args(ns), do_train=True)
    student.to(device)
    print_trainable_parameters(student)

    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        ns.model_id, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.config.use_cache = False
    teacher.to(device)
    print(f"INFO: teacher loaded in {time.time()-t0:.0f}s", flush=True)

    targs = TrainingArguments(
        output_dir=ns.save_dir or "out/tmp",
        per_device_train_batch_size=ns.per_device_train_batch_size,
        gradient_accumulation_steps=ns.gradient_accumulation_steps,
        warmup_ratio=ns.warmup_ratio,
        num_train_epochs=ns.num_train_epochs,
        max_steps=ns.max_steps,
        bf16=(device == "cuda" and dtype == torch.bfloat16),
        fp16=False,
        logging_steps=1,
        save_strategy="no",
        learning_rate=ns.lr,
        lr_scheduler_type="cosine",
        optim=ns.optim,
        adam_beta1=0.9,
        adam_beta2=0.999,
        report_to=[],
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        use_cpu=(device == "cpu"),
    )

    trainer = KDTrainer(
        model=student, teacher_model=teacher, l2l_loss_scale=ns.l2l_loss_scale,
        processing_class=tokenizer, train_dataset=train_ds, args=targs,
        data_collator=default_data_collator,
    )
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    print(f"INFO: training done in {train_secs:.0f}s", flush=True)

    if ns.save_dir:
        save_artifacts(student, tokenizer, ns.save_dir, ns)
    if ns.report_json:
        with open(ns.report_json, "w") as f:
            json.dump({"model_id": ns.model_id, "eff_bit": ns.eff_bit, "residual": bool(ns.residual),
                       "max_steps": ns.max_steps, "lr": ns.lr, "train_seconds": train_secs,
                       "train_samples": len(train_ds), "save_dir": ns.save_dir}, f, indent=2)


if __name__ == "__main__":
    main()
