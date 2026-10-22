#!/usr/bin/env python3
"""End-to-end LittleBit pipeline for one local model: FP baseline -> Dual-SVID init ->
QAT (KD/SmoothSign) -> load the packed checkpoint -> PPL + generation.

Fully resumable: every step writes logs/<tag>_*.json and is skipped when that file exists
(use --force to redo). Device is chosen automatically (GPU only if enough free VRAM, else CPU).

Example:
  venv/bin/python scripts/run_full_pipeline.py --model_id models/opt-1.3b --tag opt1.3b \
      --bpws 0.1 0.3 --qat_steps 300 --bs 4 --train_seqlen 512
"""
import argparse
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, "venv/bin/python")
LOGS = os.path.join(ROOT, "logs")

# Paper learning rates (Appendix: OPT-1.3B, QAT)
PAPER_LR = {1.0: 8e-5, 0.8: 1.2e-4, 0.7: 1.2e-4, 0.55: 1.2e-4, 0.3: 2.0e-4, 0.1: 2.0e-4}


def run(cmd, log_path, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    dt = time.time() - t0
    with open(log_path, "a") as lf:
        lf.write(f"exit={p.returncode}\n")
    print(f"  -> exit={p.returncode} in {dt:.0f}s ({os.path.relpath(log_path, ROOT)})", flush=True)
    return p.returncode


def load(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="models/opt-1.3b")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bpws", nargs="+", type=float, default=[0.3, 0.1])
    ap.add_argument("--residual", default="True")
    ap.add_argument("--qat_steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--train_seqlen", type=int, default=512)
    ap.add_argument("--ppl_seqlen", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dataset", default="wikitext2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min_free_mb", type=int, default=5000,
                    help="with --device auto: use the GPU only if at least this much VRAM is free")
    ap.add_argument("--threads", type=int, default=0, help="OMP_NUM_THREADS for children (0 = leave default)")
    ap.add_argument("--gen_tokens", type=int, default=32)
    ap.add_argument("--save_dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args()

    tag = ns.tag or os.path.basename(ns.model_id.rstrip("/"))
    os.makedirs(LOGS, exist_ok=True)
    env_extra = {}
    if ns.threads:
        env_extra = {"OMP_NUM_THREADS": str(ns.threads), "TOKENIZERS_PARALLELISM": "false"}
    if ns.device == "cpu":
        env_extra["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env_extra.setdefault("LITTLEBIT_MIN_FREE_MB", str(ns.min_free_mb))
    if ns.device == "cuda":
        env_extra["LITTLEBIT_DEVICE"] = "cuda"
    print(f"=== pipeline tag={tag} model={ns.model_id} bpws={ns.bpws} device={ns.device} ===", flush=True)

    # ---- 1. FP baseline ----
    fp_json = os.path.join(LOGS, f"{tag}_fp.json")
    if ns.force or not os.path.exists(fp_json):
        run([PY, "scripts/init_ppl.py", "--model_id", ns.model_id, "--device", ns.device,
             "--ppl_task", ns.dataset, "--seqlen", str(ns.ppl_seqlen), "--fp_eval", "True",
             "--report_json", fp_json], os.path.join(LOGS, f"{tag}_fp.log"), env_extra)

    for eb in ns.bpws:
        print(f"\n--- BPW {eb} ---", flush=True)

        # ---- 2. Dual-SVID init + PPL ----
        init_json = os.path.join(LOGS, f"{tag}_init_{eb}.json")
        if ns.force or not os.path.exists(init_json):
            run([PY, "scripts/init_ppl.py", "--model_id", ns.model_id, "--device", ns.device,
                 "--eff_bit", str(eb), "--residual", ns.residual, "--ppl_task", ns.dataset,
                 "--seqlen", str(ns.ppl_seqlen), "--report_json", init_json],
                os.path.join(LOGS, f"{tag}_init_{eb}.log"), env_extra)

        # ---- 3. QAT ----
        qat_dir = os.path.join(ROOT, "out", f"{tag}-{eb}-qat")
        qat_json = os.path.join(LOGS, f"{tag}_qat_{eb}.json")
        lr = ns.lr if ns.lr is not None else PAPER_LR.get(eb, 1.2e-4)
        if ns.force or not os.path.exists(qat_json):
            run([PY, "scripts/train_local.py", "--model_id", ns.model_id, "--device", ns.device,
                 "--eff_bit", str(eb), "--residual", ns.residual, "--dataset", ns.dataset,
                 "--train_seqlen", str(ns.train_seqlen), "--max_steps", str(ns.qat_steps),
                 "--per_device_train_batch_size", str(ns.bs), "--lr", str(lr),
                 "--save_dtype", ns.save_dtype, "--save_dir", qat_dir, "--report_json", qat_json],
                os.path.join(LOGS, f"{tag}_qat_{eb}.log"), env_extra)

        # ---- 4. Evaluate the packed checkpoint locally (+ generation) ----
        eval_json = os.path.join(LOGS, f"{tag}_eval_{eb}.json")
        if os.path.isdir(qat_dir) and (ns.force or not os.path.exists(eval_json)):
            run([PY, "scripts/eval_saved.py", "--model_dir", qat_dir, "--device", ns.device,
                 "--dtype", ns.save_dtype, "--seqlen", str(ns.ppl_seqlen), "--gen_tokens", str(ns.gen_tokens),
                 "--report_json", eval_json], os.path.join(LOGS, f"{tag}_eval_{eb}.log"), env_extra)

    # ---- Summary ----
    fp = load(fp_json) or {}
    rows = []
    for eb in ns.bpws:
        rows.append({
            "bpw": eb,
            "init_ppl": (load(os.path.join(LOGS, f"{tag}_init_{eb}.json")) or {}).get(f"ppl_{ns.dataset}"),
            "qat_ppl": (load(os.path.join(LOGS, f"{tag}_eval_{eb}.json")) or {}).get(f"ppl_{ns.dataset}"),
            "size_mb": (load(os.path.join(LOGS, f"{tag}_eval_{eb}.json")) or {}).get("size_mb"),
            "tok_s": ((load(os.path.join(LOGS, f"{tag}_eval_{eb}.json")) or {}).get("generation") or {}).get("tokens_per_s"),
            "completion": ((load(os.path.join(LOGS, f"{tag}_eval_{eb}.json")) or {}).get("generation") or {}).get("completion"),
        })
    summary = {"tag": tag, "model_id": ns.model_id, "fp_ppl": fp.get(f"ppl_{ns.dataset}"),
               "qat_steps": ns.qat_steps, "bs": ns.bs, "train_seqlen": ns.train_seqlen, "rows": rows}
    out_json = os.path.join(LOGS, f"{tag}_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ PIPELINE SUMMARY ================", flush=True)
    print(f"model={ns.model_id}  FP PPL = {summary['fp_ppl']}")
    print(f"{'BPW':>6}{'init PPL':>14}{'QAT PPL':>14}{'size MB':>10}{'tok/s':>9}")
    for r in rows:
        print(f"{r['bpw']:>6}{str(r['init_ppl']):>14}{str(r['qat_ppl']):>14}"
              f"{str(r['size_mb']):>10}{str(r['tok_s']):>9}")
        if r["completion"]:
            print(f"       generation: {r['completion'][:100]!r}")
    print(f"summary -> {out_json}", flush=True)
    print("PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
