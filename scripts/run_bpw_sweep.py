#!/usr/bin/env python3
"""BPW sweep for LittleBit: FP baseline + Dual-SVID init PPL at several BPW.

Resumable: a run whose report json already exists is skipped (delete it to redo).
Writes logs/<name>.log and logs/<name>.json, then prints a summary table.

Usage: venv/bin/python scripts/run_bpw_sweep.py [--force] [--device cpu] [--model_id models/opt-1.3b]
       [--bpws 1.0 0.55 0.3 0.1] [--residual True] [--seqlen 2048] [--tag suffix]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=os.path.join(ROOT, "models/opt-1.3b"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bpws", nargs="+", type=float, default=[1.0, 0.55, 0.3, 0.1])
    ap.add_argument("--residual", default="True")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--ppl_task", default="wikitext2")
    ap.add_argument("--fp", default="True")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args()

    py = os.path.join(ROOT, "venv/bin/python")
    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    script = os.path.join(HERE, "init_ppl.py")

    runs = []
    if ns.fp.lower() in ("true", "1", "yes"):
        runs.append((f"fp{ns.tag}", ["--fp_eval", "True"]))
    for eb in ns.bpws:
        runs.append((f"init_{eb}{ns.tag}",
                     ["--eff_bit", str(eb), "--residual", ns.residual, "--diag_detail", "2"]))

    summary = []
    for name, extra in runs:
        log_path = os.path.join(logs, f"{name}.log")
        json_path = os.path.join(logs, f"{name}.json")
        if os.path.exists(json_path) and not ns.force:
            print(f"SKIP {name} (report exists)", flush=True)
        else:
            cmd = [py, script, "--model_id", ns.model_id, "--device", ns.device,
                   "--seqlen", str(ns.seqlen), "--ppl_task", ns.ppl_task,
                   "--report_json", json_path] + extra
            print(f"RUN  {name}: {' '.join(cmd)}", flush=True)
            t0 = time.time()
            with open(log_path, "w") as lf:
                p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
            print(f"  {name} exit={p.returncode} in {time.time()-t0:.0f}s", flush=True)
            with open(log_path, "a") as lf:
                lf.write(f"exit={p.returncode}\n")
        if os.path.exists(json_path):
            with open(json_path) as f:
                summary.append((name, json.load(f)))
        else:
            summary.append((name, {}))

    print("\n================ SUMMARY ================", flush=True)
    print(f"{'run':<16}{'eff_bit':>9}{'split(min..max)':>18}{'PPL':>10}{'real BPW':>10}{'compr.':>9}")
    for name, rep in summary:
        ppl = rep.get(f"ppl_{ns.ppl_task}", float("nan"))
        st = rep.get("storage", {})
        print(f"{name:<16}{rep.get('eff_bit_target', float('nan')):>9}"
              f"{rep.get('eff_bit_actual_min', float('nan')):>9.3f}..{rep.get('eff_bit_actual_max', float('nan')):<8.3f}"
              f"{ppl:>10.4f}{st.get('real_bpw_on_linears', float('nan')):>10.4f}"
              f"{st.get('compression_x_on_linears', float('nan')):>9.1f}x")
    if len(summary) > 1 and summary[0][1].get(f"ppl_{ns.ppl_task}"):
        fp = summary[0][1][f"ppl_{ns.ppl_task}"]
        print(f"\nFP reference PPL = {fp:.4f}")
        for name, rep in summary[1:]:
            p = rep.get(f"ppl_{ns.ppl_task}")
            if p:
                print(f"  {name}: PPL {p:.4f} (+{p-fp:.2f} vs FP, x{p/fp:.2f})")
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
