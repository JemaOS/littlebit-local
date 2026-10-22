#!/usr/bin/env python3
"""LittleBit: Dual-SVID initialization + WikiText-2/C4 perplexity (no QAT).

Usage:
  python scripts/init_ppl.py --model_id models/opt-1.3b --eff_bit 0.1 --residual True
  python scripts/init_ppl.py --model_id models/opt-1.3b --eff_bit 1.0 --residual True --save_dir out/opt1.3b-1.0
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "repo")
sys.path.insert(0, HERE)
from lbcommon import decide_device_env  # noqa: E402

decide_device_env()  # before `import torch`

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, REPO)

# NOTE: import quantization.utils BEFORE quantization.modules (this order avoids a
# circular import in the upstream repo: modules/littlebit -> utils.binary_packer -> utils/__init__ -> modules).
from quantization.utils import apply_littlebit_patch, get_quant_func_and_mod  # noqa: E402
from quantization.modules import LittleBitLinear  # noqa: E402
from utils.datautils import get_eval_loaders, load_tokenizer  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "y", "1")


def build_args(ns):
    return argparse.Namespace(
        quant_func=ns.quant_func,
        quant_mod="LittleBitLinear",
        residual=ns.residual,
        split_dim=ns.split_dim,
        eff_bit=ns.eff_bit,
        kv_factor=ns.kv_factor,
        min_split_dim=ns.min_split_dim,
        use_itq=ns.use_itq,
        itq_n_iter=ns.itq_n_iter,
        model_id=ns.model_id,
    )


@torch.no_grad()
def ppl(model, tokenizer, dataset, seqlen, device):
    testenc = get_eval_loaders(dataset, tokenizer).input_ids
    nsamples = testenc.numel() // seqlen
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
        if i % 25 == 0:
            print(f"  chunk {i}/{nsamples} ({time.time()-t0:.0f}s)", flush=True)
    ppl_val = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1))).item()
    return ppl_val


def effective_weight(mod):
    """Reconstruct the implicit effective weight from a LittleBitLinear module (Eq. 5 + Eq. 9)."""
    def blk(U, V, u1, u2, v1, v2):
        q = mod.quant_func
        Us = q(U.float())
        Vs = q(V.float())
        h = u1.float().reshape(-1, 1)
        g = v2.float().reshape(1, -1)
        lat = (u2.float() * v1.float()).reshape(1, -1)
        return ((Us * lat) * h) @ Vs * g

    W = blk(mod.U, mod.V, mod.u1, mod.u2, mod.v1, mod.v2)
    if mod.residual:
        W = W + blk(mod.U_R, mod.V_R, mod.u1_R, mod.u2_R, mod.v1_R, mod.v2_R)
    return W


@torch.no_grad()
def reconstruction_diagnostics(model, orig_weights, max_detail=4):
    """Compare the Dual-SVID initialized effective weights with the original FP weights."""
    details = []
    num = 0.0
    den = 0.0
    svd_num = 0.0
    for name, mod in model.named_modules():
        if not isinstance(mod, LittleBitLinear):
            continue
        W = orig_weights.get(name)
        if W is None:
            continue
        W = W.float()
        W_hat = effective_weight(mod)
        err = torch.linalg.norm(W_hat - W) / torch.linalg.norm(W)
        # rank-r truncated SVD reference error at the same rank (isolates the cost of binarization)
        q = min(mod.split_dim, min(W.shape) - 1)
        try:
            U_t, S_t, V_t = torch.svd_lowrank(W, q=q)
            W_svd = (U_t * S_t) @ V_t.t()
            svd_err = (torch.linalg.norm(W_svd - W) / torch.linalg.norm(W)).item()
        except Exception as e:  # noqa
            svd_err = float("nan")
            print(f"WARN svd_lowrank failed for {name}: {e}", flush=True)
        num += (torch.linalg.norm(W_hat - W) ** 2).item()
        den += (torch.linalg.norm(W) ** 2).item()
        svd_num += (torch.linalg.norm(W_svd - W) ** 2).item()
        if len(details) < max_detail:
            details.append({"name": name, "rank": mod.split_dim, "eff_bit_actual": round(mod.eff_bit_actual, 4),
                            "rel_err_dualsvid": round(err.item(), 4), "rel_err_trunc_svd": round(svd_err, 4)})
        del W, W_hat
    agg = {
        "rel_err_dualsvid_global": (num / den) ** 0.5,
        "rel_err_trunc_svd_global": (svd_num / den) ** 0.5,
        "per_layer_sample": details,
    }
    return agg


def storage_report(model):
    """Real stored bytes for the quantized representation + reference counts."""
    q_bytes = 0
    q_params = 0
    orig_bytes = 0
    other_bytes = 0
    for name, mod in model.named_modules():
        if isinstance(mod, LittleBitLinear):
            n, m = mod.out_features, mod.in_features
            orig_bytes += n * m * 2  # fp16 reference
            q_params += n * m
            sd = mod.state_dict()
            for k, v in sd.items():
                if k.endswith("_packed") or k in ("u1", "u2", "v1", "v2", "u1_R", "u2_R", "v1_R", "v2_R", "bias"):
                    q_bytes += v.numel() * v.element_size()
    for p in model.parameters():
        if not any(True for _ in []):
            pass
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    other_bytes = total - sum(
        v.numel() * v.element_size()
        for mod in model.modules() if isinstance(mod, LittleBitLinear)
        for k, v in mod.state_dict().items()
        if not k.endswith("_packed")
    )
    return {
        "quantized_linear_params": q_params,
        "quantized_linear_bytes": q_bytes,
        "real_bpw_on_linears": q_bytes * 8 / max(q_params, 1),
        "fp16_reference_linears_bytes": orig_bytes,
        "compression_x_on_linears": orig_bytes / max(q_bytes, 1),
        "whole_model_bytes": total,
        "non_linear_bytes": other_bytes,
    }


def layer_table(model):
    rows = []
    for name, mod in model.named_modules():
        if isinstance(mod, LittleBitLinear):
            rows.append((name, mod.in_features, mod.out_features, int(mod._split_dim_final),
                         round(mod.eff_bit_target, 4) if mod.eff_bit_target is not None else None,
                         round(mod.eff_bit_actual, 4)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="models/opt-1.3b")
    ap.add_argument("--eff_bit", type=float, default=1.0)
    ap.add_argument("--residual", type=str2bool, default=True)
    ap.add_argument("--split_dim", type=int, default=1024)
    ap.add_argument("--kv_factor", type=float, default=1.0)
    ap.add_argument("--min_split_dim", type=int, default=8)
    ap.add_argument("--quant_func", default="SmoothSign")
    ap.add_argument("--use_itq", type=str2bool, default=False)
    ap.add_argument("--itq_n_iter", type=int, default=50)
    ap.add_argument("--ppl_task", default="wikitext2")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    ap.add_argument("--fp_eval", type=str2bool, default=False, help="evaluate unquantized baseline")
    ap.add_argument("--diagnostics", type=str2bool, default=True, help="report Dual-SVID reconstruction error")
    ap.add_argument("--diag_detail", type=int, default=4)
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--report_json", default=None)
    ns = ap.parse_args()

    if ns.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = ns.device
    if device == "cpu":
        # upstream code (LittleBitLinear._decompose_matrix) forces cuda whenever it is visible
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if ns.dtype == "auto":
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = getattr(torch, ns.dtype)

    print(f"INFO: device={device} dtype={dtype} eff_bit={ns.eff_bit} residual={ns.residual}", flush=True)
    tokenizer = load_tokenizer(ns.model_id)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        ns.model_id, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    )
    print(f"INFO: loaded FP model in {time.time()-t0:.1f}s", flush=True)

    result = {"model_id": ns.model_id, "eff_bit_target": ns.eff_bit, "residual": bool(ns.residual)}
    if not ns.fp_eval:
        orig_weights = {name: mod.weight.detach().clone()
                        for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
        t0 = time.time()
        model = apply_littlebit_patch(model, build_args(ns), do_train=True)
        print(f"INFO: Dual-SVID init done in {time.time()-t0:.1f}s", flush=True)
        if ns.diagnostics:
            t0 = time.time()
            diag = reconstruction_diagnostics(model, orig_weights, max_detail=ns.diag_detail)
            print(f"INFO: dual-svid global rel err = {diag['rel_err_dualsvid_global']:.4f} | "
                  f"rank-only SVD rel err = {diag['rel_err_trunc_svd_global']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            for d in diag["per_layer_sample"]:
                print(f"INFO:   {d['name']}: rank={d['rank']} eb={d['eff_bit_actual']} "
                      f"err_dualsvid={d['rel_err_dualsvid']} err_svd={d['rel_err_trunc_svd']}", flush=True)
            result["init_diagnostics"] = diag
        del orig_weights
        rows = layer_table(model)
        result["layers"] = rows[:6]
        result["eff_bit_actual_mean"] = sum(r[5] for r in rows) / len(rows)
        result["eff_bit_actual_min"] = min(r[5] for r in rows)
        result["eff_bit_actual_max"] = max(r[5] for r in rows)
        print(f"INFO: split_dim range {min(r[3] for r in rows)}..{max(r[3] for r in rows)}", flush=True)
        print(f"INFO: eff_bit_actual mean={result['eff_bit_actual_mean']:.4f} "
              f"min={result['eff_bit_actual_min']:.4f} max={result['eff_bit_actual_max']:.4f}", flush=True)

    model.to(device)
    if device == "cpu":
        model = model.float()

    if ns.ppl_task:
        for ds in ns.ppl_task.split(","):
            t0 = time.time()
            val = ppl(model, tokenizer, ds, ns.seqlen, device)
            print(f"[{ds}] PPL = {val:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            result[f"ppl_{ds}"] = val

    if not ns.fp_eval:
        rep = storage_report(model)
        for k, v in rep.items():
            print(f"INFO: {k} = {v:.4f}" if isinstance(v, float) else f"INFO: {k} = {v}", flush=True)
        result["storage"] = rep

    if ns.save_dir:
        os.makedirs(ns.save_dir, exist_ok=True)
        model.eval()
        model.config.use_cache = True
        quant_params = {
            "quant_func": ns.quant_func,
            "eff_bit": ns.eff_bit,
            "split_dim": ns.split_dim,
            "residual": bool(ns.residual),
            "kv_factor": ns.kv_factor,
            "min_split_dim": ns.min_split_dim,
            "quant_mod": "LittleBitLinear",
            "use_itq": bool(ns.use_itq),
            "itq_n_iter": ns.itq_n_iter,
        }
        with open(os.path.join(ns.save_dir, "littlebit_config.json"), "w") as f:
            json.dump(quant_params, f, indent=2)
        for k, v in quant_params.items():
            setattr(model.config, k, v)
        sd = model.state_dict()
        for k, v in list(sd.items()):
            if "packed" not in k and "shape" not in k and v.dtype == torch.float32:
                sd[k] = v.to(torch.bfloat16)
        model.save_pretrained(ns.save_dir, state_dict=sd, safe_serialization=True)
        tokenizer.save_pretrained(ns.save_dir)
        print(f"INFO: saved to {ns.save_dir}", flush=True)
        result["save_dir"] = ns.save_dir

    if ns.report_json:
        with open(ns.report_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"INFO: report -> {ns.report_json}", flush=True)


if __name__ == "__main__":
    main()
