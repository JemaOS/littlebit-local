# LittleBit — résultats agrégés

Généré par `scripts/report.py` le 2026-10-22 17:31. Toutes les valeurs proviennent des rapports `logs/*.json` (aucune saisie manuelle).

## Perplexité

| rapport | modèle | BPW cible | PPL wikitext2 | taille (Mo) | BPW réel | compression |
|---|---|---|---|---|---|---|
| `125m_e0.1.json` | opt-125m | 0.10 | 61495.6055 | — | 0.235 | 68.0x |
| `125m_e0.3.json` | opt-125m | 0.30 | 206151.8281 | — | 0.434 | 36.8x |
| `125m_e1.0.json` | opt-125m | 1.00 | 11236.9365 | — | 1.116 | 14.3x |
| `125m_fp.json` | opt-125m | 1.00 | 27.6553 | — | — | —x |
| `eval_125m_qat30.json` | opt125m-0.3-qat | — | 2289.1541 | 158.7 | — | —x |
| `fp.json` | opt-1.3b | 1.00 | 14.6237 | — | — | —x |
| `pipe125m_fp.json` | opt-125m | 1.00 | 27.6553 | — | — | —x |
| `pipe125m_init_0.1.json` | opt-125m | 0.10 | 61495.6055 | — | 0.235 | 68.0x |

## Générations locales

| rapport | modèle | tok/s | complétion |
|---|---|---|---|
| `eval_125m_qat30.json` | opt125m-0.3-qat | 105.21 | ` the the the the the the the the the the the the the the the the` |

## Références papier (PPL WikiText-2, après QAT complet : 5 époques, ~1B tokens, 4×H100)

| Modèle | FP16 | 1.0 BPW | 0.55 BPW | 0.3 BPW | 0.1 BPW | Empreinte @0.1 BPW |
|---|---|---|---|---|---|---|
| OPT-1.3B | 14.62 | 20.33 | 23.35 | 28.30 | 53.76 | — |
| Llama2-7B | 5.47 | 9.08 | 10.47 | 12.00 | 15.92 | 0.63 Go (21.4×) |
| Llama2-13B | 4.88 | 8.18 | 9.24 | 10.48 | 15.09 | 0.84 Go (31.0×) |
| Llama3-8B | 6.10 | 16.30 | 18.47 | 20.34 | 26.11 | — |

Mémoire (tab. 5 du papier) : Llama2-13B passe de 26,06 Go (FP16) à 0,84 Go à 0,1 BPW = **31×**.
Les embeddings et `lm_head` restent en FP16, d'où l'écart entre BPW effectif et facteur de compression.

Checkpoint communautaire entraîné (0.5 époque, 1×H200) : Qwen3-8B @0.55 BPW → PPL 24.48 (BF16 9.73).
