# LittleBit — résultats agrégés

Généré par `scripts/report.py` le 2026-11-04 04:08. Toutes les valeurs proviennent des rapports `logs/*.json` (aucune saisie manuelle).

## Perplexité

| rapport | modèle | BPW cible | PPL c4 | PPL wikitext2 | taille (Mo) | BPW réel | compression |
|---|---|---|---|---|---|---|---|
| `125m_e0.1.json` | opt-125m | 0.10 | — | 61495.6055 | — | 0.235 | 68.0x |
| `125m_e0.3.json` | opt-125m | 0.30 | — | 206151.8281 | — | 0.434 | 36.8x |
| `125m_e0.55.json` | opt-125m | 0.55 | — | 39106.3984 | — | 0.666 | 24.0x |
| `125m_e1.0.json` | opt-125m | 1.00 | — | 11236.9365 | — | 1.116 | 14.3x |
| `125m_fp.json` | opt-125m | 1.00 | — | 27.6553 | — | — | —x |
| `125m_itq_e0.55.json` | opt-125m | 0.55 | — | 6683.3140 | — | 0.666 | 24.0x |
| `125m_itq_e1.0.json` | opt-125m | 1.00 | — | 8768.1260 | — | 1.116 | 14.3x |
| `eval_125m_qat30.json` | opt125m-0.3-qat | — | — | 2289.1541 | 158.7 | — | —x |
| `fp.json` | opt-1.3b | 1.00 | — | 14.6237 | — | — | —x |
| `fp_c4.json` | opt-1.3b | 1.00 | 14.7279 | — | — | — | —x |
| `init_0.1.json` | opt-1.3b | 0.10 | — | 14501.8047 | — | 0.132 | 121.1x |
| `init_0.3.json` | opt-1.3b | 0.30 | — | 13092.6826 | — | 0.340 | 47.1x |
| `init_0.55.json` | opt-1.3b | 0.55 | — | 18975.4199 | — | 0.601 | 26.6x |
| `init_1.0.json` | opt-1.3b | 1.00 | — | 18211.4336 | — | 1.044 | 15.3x |
| `lb_qwen3_8b_eval_fp32.json` | lb-qwen3-8b-0.55bpw | — | — | 24.4770 | 2846.4 | — | —x |
| `opt1.3b_c4_itq_eval.json` | opt1.3b-0.55-c4-itq | — | — | 71.8000 | 291.7 | — | —x |
| `opt1.3b_c4_itq_eval_c4.json` | opt1.3b-0.55-c4-itq | — | 70.3013 | — | 291.7 | — | —x |
| `opt1.3b_itq6k_eval_c4.json` | opt1.3b-0.55-qat-itq-6k | — | 187.1935 | — | 291.7 | — | —x |
| `opt1.3b_qat_0.1_eval.json` | opt1.3b-0.1-qat | — | — | 775.3759 | 420.8 | — | —x |
| `opt1.3b_qat_0.1_long_eval.json` | opt1.3b-0.1-qat-long | — | — | 536.4883 | 224.4 | — | —x |
| `opt1.3b_qat_0.3_eval.json` | opt1.3b-0.3-qat | — | — | 465.6932 | 254.2 | — | —x |
| `opt1.3b_qat_0.3_long_eval.json` | opt1.3b-0.3-qat-long | — | — | 146.4551 | 254.2 | — | —x |
| `opt1.3b_qat_0.55_itq_6k_eval.json` | opt1.3b-0.55-qat-itq-6k | — | — | 68.1079 | 291.7 | — | —x |
| `opt1.3b_qat_0.55_itq_eval.json` | opt1.3b-0.55-qat-itq | — | — | 77.9294 | 291.7 | — | —x |
| `opt1.3b_qat_0.55_long_eval.json` | opt1.3b-0.55-qat-long | — | — | 109.5307 | 291.7 | — | —x |
| `pipe125m_fp.json` | opt-125m | 1.00 | — | 27.6553 | — | — | —x |
| `pipe125m_init_0.1.json` | opt-125m | 0.10 | — | 61495.6055 | — | 0.235 | 68.0x |

## Générations locales

| rapport | modèle | tok/s | complétion |
|---|---|---|---|
| `eval_125m_qat30.json` | opt125m-0.3-qat | 105.21 | ` the the the the the the the the the the the the the the the the` |
| `lb_qwen3_8b_eval_fp32.json` | lb-qwen3-8b-0.55bpw | 1.26 | ` Paris, and the city is the capital of the French Republic. The capital is located in the French Riviera, which` |
| `opt1.3b_c4_itq_eval.json` | opt1.3b-0.55-c4-itq | 72.02 | ` the capital of the United States, and the capital of the United Kingdom. The capital of France is the capital of the Un` |
| `opt1.3b_qat_0.1_eval.json` | opt1.3b-0.1-qat | 8.60 | ` the first first first first first first first first the first first first first the first first first first the first f` |
| `opt1.3b_qat_0.1_long_eval.json` | opt1.3b-0.1-qat-long | 63.64 | ` a few of the first-@.@.@.@.@.@.@ @.@ @.` |
| `opt1.3b_qat_0.3_eval.json` | opt1.3b-0.3-qat | 35.28 | ` a new-@-@-@-@-@-@-@-@-@-@-@` |
| `opt1.3b_qat_0.3_long_eval.json` | opt1.3b-0.3-qat-long | 69.17 | ` the capital of the United Kingdom of France. The United Kingdom of France is the capital of the United Kingdom of Franc` |
| `opt1.3b_qat_0.55_itq_6k_eval.json` | opt1.3b-0.55-qat-itq-6k | 69.90 | ` the French capital of the Republic of France. The French capital is located in the city of Paris, in the north-west of ` |
| `opt1.3b_qat_0.55_itq_eval.json` | opt1.3b-0.55-qat-itq | 66.51 | ` the capital of the French Republic of France. The French Republic of France is the capital of the French Republic of Fr` |
| `opt1.3b_qat_0.55_long_eval.json` | opt1.3b-0.55-qat-long | 42.98 | ` the largest city in France. The city is the largest city in France.  The city of Paris is the largest city in France.  ` |

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
