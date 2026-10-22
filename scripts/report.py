#!/usr/bin/env python3
"""Aggregate every logs/*.json report into RESULTS.md (reproducible reporting).

Usage: venv/bin/python scripts/report.py [--out RESULTS.md]
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGS = os.path.join(ROOT, "logs")


def rows():
    out = []
    for name in sorted(os.listdir(LOGS)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(LOGS, name)
        try:
            with open(path) as f:
                rep = json.load(f)
        except Exception:
            continue
        ppl = {k[4:]: v for k, v in rep.items() if k.startswith("ppl_") and not k.endswith("_chunks")}
        gen = rep.get("generation") or {}
        out.append({
            "file": name,
            "model": os.path.basename(str(rep.get("model_id") or rep.get("model_dir") or "?")),
            "eff_bit": rep.get("eff_bit_target"),
            "residual": rep.get("residual"),
            "ppl": ppl,
            "size_mb": rep.get("size_mb"),
            "on_disk_mb": rep.get("on_disk_mb"),
            "packed_mb": rep.get("packed_mb"),
            "real_bpw": (rep.get("storage") or {}).get("real_bpw_on_linears"),
            "compression": (rep.get("storage") or {}).get("compression_x_on_linears"),
            "completion": gen.get("completion"),
            "tok_s": gen.get("tokens_per_s"),
            "binarized": rep.get("binarized"),
        })
    return out


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "RESULTS.md"))
    ns = ap.parse_args()
    rs = rows()
    lines = ["# LittleBit — résultats agrégés", "",
             f"Généré par `scripts/report.py` le {time.strftime('%Y-%m-%d %H:%M')}. "
             "Toutes les valeurs proviennent des rapports `logs/*.json` (aucune saisie manuelle).", ""]

    ppl_models = [r for r in rs if r["ppl"]]
    if ppl_models:
        keys = sorted({k for r in ppl_models for k in r["ppl"]})
        lines += ["## Perplexité", "",
                  "| rapport | modèle | BPW cible | PPL " + " | PPL ".join(keys) + " | taille (Mo) | BPW réel | compression |",
                  "|---|---|---|" + "---|" * (len(keys) + 3)]
        for r in ppl_models:
            cells = " | ".join(fmt(r["ppl"].get(k)) for k in keys)
            lines.append(f"| `{r['file']}` | {r['model']} | {fmt(r['eff_bit'], 2)} | {cells} | "
                         f"{fmt(r['size_mb'], 1) if r['size_mb'] else fmt(r['on_disk_mb'], 1)} | "
                         f"{fmt(r['real_bpw'], 3)} | {fmt(r['compression'], 1)}x |")
        lines.append("")

    gens = [r for r in rs if r["completion"]]
    if gens:
        lines += ["## Générations locales", "", "| rapport | modèle | tok/s | complétion |", "|---|---|---|---|"]
        for r in gens:
            comp = (r["completion"] or "").replace("\n", " ")[:120]
            lines.append(f"| `{r['file']}` | {r['model']} | {fmt(r['tok_s'], 2)} | `{comp}` |")
        lines.append("")

    lines += ["## Références papier (PPL WikiText-2, après QAT complet : 5 époques, ~1B tokens, 4×H100)", "",
              "| Modèle | FP16 | 1.0 BPW | 0.55 BPW | 0.3 BPW | 0.1 BPW | Empreinte @0.1 BPW |", "|---|---|---|---|---|---|---|",
              "| OPT-1.3B | 14.62 | 20.33 | 23.35 | 28.30 | 53.76 | — |",
              "| Llama2-7B | 5.47 | 9.08 | 10.47 | 12.00 | 15.92 | 0.63 Go (21.4×) |",
              "| Llama2-13B | 4.88 | 8.18 | 9.24 | 10.48 | 15.09 | 0.84 Go (31.0×) |",
              "| Llama3-8B | 6.10 | 16.30 | 18.47 | 20.34 | 26.11 | — |", "",
              "Mémoire (tab. 5 du papier) : Llama2-13B passe de 26,06 Go (FP16) à 0,84 Go à 0,1 BPW = **31×**.",
              "Les embeddings et `lm_head` restent en FP16, d'où l'écart entre BPW effectif et facteur de compression.", "",
              "Checkpoint communautaire entraîné (0.5 époque, 1×H200) : Qwen3-8B @0.55 BPW → PPL 24.48 (BF16 9.73).", ""]
    with open(ns.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {ns.out} ({len(rs)} reports, {len(ppl_models)} with PPL)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
