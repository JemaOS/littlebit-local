# LittleBit local — sub-1-bit quantization of local LLMs (reproduction & working setup)

Reproduction **locale** de *LittleBit: Ultra Low-Bit Quantization via Latent Factorization*
(Banseok Lee, Dongkyu Kim, Youngcheon You, Youngmin Kim — NeurIPS 2025, arXiv:2506.13771),
à partir du code officiel [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit),
avec pour objectif de **faire tourner un modèle compressé en local** (init → QAT → export → inférence).

Papier local : `/home/john/Téléchargements/arXiv-2506.13771v5/main.tex`.

---

## 1. Matériel & contraintes

| Ressource | Valeur |
|---|---|
| GPU | RTX 4070 Laptop 12 Go — **occupée par LM Studio (~8.5–9 Go)** → ~0.4 Go libres |
| CPU | 32 threads (utilisé pour tout le pipeline) |
| RAM | 188 Go |
| Disque | 1.8 To libres |
| Python | 3.12.3 (venv local, pas de conda) |

Conséquence : tout le pipeline tourne **sur CPU**. Le code détecte la VRAM libre et retombe
automatiquement sur CPU (`pick_device`, seuil 6 Go) pour ne jamais OOM.

## 2. Installation (reproductible)

```bash
python3 -m venv venv
venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
venv/bin/pip install -r requirements-local.txt   # transformers==4.51.3 (pin papier), datasets, accelerate, ...
git clone https://github.com/SamsungLabs/LittleBit repo
python3 fetch_model.py facebook/opt-1.3b models/opt-1.3b tf_model.h5 flax_model.msgpack
python3 fetch_model.py facebook/opt-125m models/opt-125m tf_model.h5 flax_model.msgpack
```

`fetch_model.py` : téléchargeur HF sans dépendance (urllib), utilisé car le repo `opt-1.3b`
expose `pytorch_model.bin` fp16 (2.6 Go) et des formats inutiles (`tf_model.h5`, `flax_model.msgpack`).

## 3. Vérification de la fidélité code ↔ papier

| Papier | Code officiel | Vérifié |
|---|---|---|
| `W ≈ UVᵀ`, facteurs binarisés | `LittleBitLinear._decompose_matrix` (`svd_lowrank(q=split_dim)`, `U = U_t√S`, `V = √S Vᵀ`) | ✅ |
| Échelles `h`,`g`,`ℓ` (Éq. 3) | `u1`=h, `v2`=g, `u2`=ℓ_u, `v1`=ℓ_v, `ℓ = ℓ_u ⊙ ℓ_v` | ✅ |
| `Ŵ_pri = diag(h)U_sign diag(ℓ)V_signᵀ diag(g)` (Éq. 5) | forward `(((x*v2) @ Vqᵀ) * (v1*u2)) @ Uqᵀ * u1` | ✅ (écart max 5e-7, `scripts/test_module_math.py`) |
| Dual-SVID init : SVD tronquée + rank-1 SVD sur `|U|`,`|V|` | `_rank_one_decompose` sur `abs(U)`,`abs(V)` | ✅ |
| Residual Compensation (Éq. 8–10) | 2ᵉ chemin identique, init sur `W − Ŵ_pri,0` | ✅ |
| SmoothSign (gradient `tanh(100x)`) | `quantization/functions/binary.py` (`alpha=100`) | ✅ |
| `L = KL + 10·MSE_hidden` | `utils/kd_utils.py` (`l2l_loss_scale=10`) | ✅ |
| BPW effectif `s(a+b+16)+16(a+b)` / `ab` (et `×2` si residual) | `_compute_eff_bits` | ✅ (exact, testé) |

**Bugs upstream rencontrés (contournés sans modifier le repo) :**

1. **Import circulaire** : `quantization.modules` doit être importé **après** `quantization.utils`
   (comme dans `main.py`), sinon `ImportError` sur `LittleBitLinear`.
2. **`LittleBitLinear.split_dim_used`** lit `self._split_dim` qui n'existe pas (le buffer est
   `_split_dim_final`) → `AttributeError` dès qu'on l'appelle (`extra_repr`).
3. **Chargement d'un checkpoint (`quantization/hub.py`)** : le squelette est construit sous
   `with torch.device("meta")`, or l'import de `transformers.models.opt.modeling_opt` y échoue
   (`Tensor.item() cannot be called on meta tensors`). Contournement : pré-importer le module
   `modeling_<model_type>` avant (`scripts/lbcommon.py::preimport_model_module`).
4. **`main.py` importe `deepspeed`/`GPUtil`/`wandb` inconditionnellement** → `scripts/train_local.py`
   réimplémente le même entraînement (KD + SmoothSign + save au format packé) sans DeepSpeed.

## 4. Scripts

| Script | Rôle |
|---|---|
| `scripts/test_module_math.py` | tests unitaires : forward == poids effectif explicite, formules split_dim/BPW |
| `scripts/init_ppl.py` | chargement FP → **init Dual-SVID** → PPL WikiText-2 + diagnostic de reconstruction + rapport de taille réelle ; `--save_dir` pour exporter |
| `scripts/run_bpw_sweep.py` | balayage FP + 1.0/0.55/0.3/0.1 BPW (reprise automatique, écrit `logs/*.json`) |
| `scripts/train_local.py` | QAT local (KD, SmoothSign, sans DeepSpeed), sauvegarde au format packé `*_packed`/`*_shape` |
| `scripts/eval_saved.py` | PPL + génération d'un checkpoint compressé via le loader officiel (`LittleBitModel`) |
| `scripts/generate.py` | inférence locale + mesure mémoire/tok-s |
| `scripts/lbcommon.py` | helpers (pré-import `modeling_*`, garde VRAM, lecture `model_type`) |
| `scripts/run_full_pipeline.py` | **pipeline complet reprenable** : FP → init Dual-SVID → QAT → éval du checkpoint + génération, un seul commandement pour un modèle donné |

### Exécution

```bash
# tout le pipeline pour OPT-1.3B aux taux 0.1 et 0.3 BPW (reprend là où il s'est arrêté)
venv/bin/python scripts/run_full_pipeline.py --model_id models/opt-1.3b --tag opt1.3b \
    --bpws 0.1 0.3 --qat_steps 300 --bs 4 --train_seqlen 512 --threads 12
```

### Pièges d'exécution rencontrés (importants pour reproduire)

- **GPU saturé** : `LittleBitLinear._decompose_matrix` déplace le SVD sur `cuda` dès que
  `torch.cuda.is_available()` est vrai → OOM quand un autre workload occupe la carte.
  Contournement : `lbcommon.decide_device_env()` (appelé **avant** `import torch`) cache le GPU
  (`CUDA_VISIBLE_DEVICES=""`) si la VRAM libre est < seuil (`LITTLEBIT_MIN_FREE_MB`, 5 Go par défaut).
- **Commandes longues** : lancer via un processus *persistent* (les commandes shell longues sont
  interrompues), et les scripts sont reprenables (`logs/*.json` sert de point de reprise).
- **Distillation** : la perte KD du code officiel est `batchmean` sur la dimension *batch* (et non
  tokens) → ordre de grandeur ≈ `seq_len × KL` nats ; à titre indicatif seulement.

## 5. Résultats

### 5.1 OPT-125M (validation rapide du pipeline)

PPL WikiText-2, `seqlen=2048`, CPU.

| Config | PPL | Taille disque | Notes |
|---|---|---|---|
| FP32 (référence) | **27.66** | 241 Mo | |
| Init Dual-SVID 1.0 BPW (residual) | 11 237 | — | modèle cassé à l'init (attendu : la binarisation détruit le modèle) |
| Init Dual-SVID 0.3 BPW (residual) | 206 152 | — | |
| Init Dual-SVID 0.1 BPW (residual) | 61 496 | — | |
| **QAT 30 steps @0.3 BPW** (bs2, lr1.2e-4, WikiText-2) | **2 289** | 20 Mo | 90× mieux que l'init → l'entraînement fonctionne |
| QAT 400 steps @1.0 BPW | en cours | — | |

Erreur de reconstruction globale de l'init (Frobenius relatif) :
1.0 BPW : **0.75** (SVD tronquée seule : 0.53) — 0.3 BPW : 0.89 (0.82) — 0.1 BPW : 0.96 (0.94).

### 5.1.1 Effet de la rotation Joint-ITQ (`--use_itq True`, LittleBit-2)

| Config (OPT-125M) | erreur init | PPL init |
|---|---|---|
| 1.0 BPW, sans ITQ | 0.7500 | 11 237 |
| 1.0 BPW, **avec ITQ (50 iter)** | **0.6335** | **8 768** |
| 0.55 BPW, sans ITQ | 0.8257 | 39 106 |
| 0.55 BPW, **avec ITQ (50 iter)** | **0.7602** | **6 683** |

L'ITQ est une rotation **orthogonale** du couple de facteurs : elle ne change pas l'erreur de la SVD
tronquée mais aligne les facteurs sur l'hypercube binaire, ce qui réduit directement l'erreur de
binarisation. Le gain **croît quand le débit baisse** (−16 % d'erreur à 1,0 BPW, −8 % mais **5,85×**
de PPL en moins à 0,55 BPW). C'est le mécanisme activé (`use_itq: true`) dans la recette
communautaire. Coût : `itq_n_iter` SVD par couche à l'init.

Génération après 30 steps @0.3 BPW (dégénérée, attendu pour 30 steps) :
`"The capital of France is"` → `" the the the the the the the the the the the the the the the the"`,
105 tok/s (CPU, opt-125m).

### 5.2 OPT-1.3B (modèle du papier)

Références papier (PPL WikiText-2, après QAT complet : 5 époques, ~1B tokens, 4×H100) :
FP 14.62 → 1.0 BPW 20.33 → 0.55 BPW 23.35 → 0.3 BPW 28.30 → 0.1 BPW 53.76.

| Config (local, CPU/GPU) | PPL WikiText-2 | Détails |
|---|---|---|
| FP32 (référence) | **14,6237** | ✅ identique au papier (14,62) → protocole validé |
| Init Dual-SVID @1,0 BPW | 18 211 | BPW réel 1,044 — compression **15,3×** (linéaires) |
| Init Dual-SVID @0,55 BPW | 18 975 | BPW réel 0,601 — **26,6×** |
| Init Dual-SVID @0,3 BPW | 13 093 | BPW réel 0,340 — **47,1×** |
| Init Dual-SVID @0,1 BPW | 14 502 | BPW réel 0,132 — **121,1×** |
| **QAT 19734 steps @0,55 BPW + ITQ (C4, 40,4 M tokens)** | **71,8** | 1 époque complète C4+WikiText-2 (aucune répétition) ; 292 Mo dont **81 Mo binaires** ; 72 tok/s ; texte répétitif |
| **QAT 6000 steps @0,55 BPW + ITQ** | **68,1** | ✅ meilleur sur WikiText-2 : 18 975 → 68,1 (**×279**) ; 292 Mo dont **81 Mo binaires** ; **69,9 tok/s** ; texte cohérent |
| QAT 3000 steps @0,55 BPW + ITQ | 77,9 | même budget que ci-dessous : **−29 %** grâce à l'ITQ |
| QAT 3000 steps @0,55 BPW | 109,5 | 18 975 → 109,5 (×173) ; 292 Mo dont 81 Mo binaires ; 43 tok/s ; texte cohérent |
| **QAT 3000 steps @0,3 BPW** (GPU) | **146,5** | 13 093 → 146,5 (**×89**) ; 254 Mo dont 42 Mo binaires ; 69 tok/s ; grammatical mais faux |
| QAT 600 steps @0,1 BPW (GPU) | 536,5 | 14 502 → 536,5 (**×27,0**) ; 224 Mo dont 11 Mo binaires ; 63,6 tok/s |
| QAT 300 steps @0,3 BPW (GPU) | 465,7 | 13 093 → 465,7 (**×28,1**) ; 254 Mo dont 42 Mo binaires ; 35,3 tok/s |
| QAT 200 steps @0,1 BPW (CPU) | 775,4 | 14 502 → 775,4 (×18,7) ; 421 Mo dont 11 Mo binaires ; 8,6 tok/s |

> **Modèle final** — OPT-1.3B @0,55 BPW + Joint-ITQ, 6000 steps (10 M tokens vus, 5 époques) :
> `"The capital of France is"` → **`" the French capital of the Republic of France. The French capital is located in the city of Paris, in the north-west of the city."`**
> (69,9 tok/s, 292 Mo sur disque dont 81 Mo de facteurs binaires, ≈26× moins de poids linéaires qu'en FP16.)

### 5.2.3 C4 vs WikiText-2 : le classement s'inverse selon le domaine mesuré

Deux modèles identiques (OPT-1.3B @0,55 BPW + ITQ, même init, même protocole), seule la **donnée
d'entraînement** change. Évaluation croisée sur les deux domaines (140 chunks de 2048 tokens chacun,
GPU) :

| Entraînement | tokens vus | PPL WikiText-2 | PPL C4 |
|---|---|---|---|
| FP32 (référence) | — | **14,62** | **14,73** |
| WikiText-2 train (5,06 époques) | 10 M (2 M uniques) | **68,1** | 187,2 |
| C4 + WikiText-2 (1 époque) | 40,4 M (uniques) | 71,8 | **70,3** |

Lecture :
- Le modèle **WikiText-2** gagne sur le domaine qu'il a vu 5 fois (68,1) mais s'effondre hors domaine
  (187,2 sur C4) — il a mémorisé le corpus.
- Le modèle **C4** est **stable sur les deux domaines** (70,3 / 71,8) : il généralise. Sur C4 il est
  **2,7× meilleur** que le modèle WikiText-2 (70,3 vs 187,2).
- Conclusion méthodologique : la PPL WikiText-2 seule **récompense le sur-apprentissage du domaine**.
  Pour juger un modèle réellement utilisable, il faut la moyenne des deux domaines — et là, C4
  (≈71,0) bat WikiText-2 (≈127,7) d'un facteur 1,8.
- Écart restant avec le papier (23,35) : 40,4 M tokens contre ~1 Md, soit 25× moins de données ;
  les 3,0× de PPL restants sont cohérents avec cet écart.

### 5.2.2 Limite identifiée : passage de « compute-limited » à « data-limited »

Doubler le calcul (3000 → 6000 steps) ne rend que **12,6 %** de PPL (77,9 → 68,1), alors que les
3000 premiers steps en rendaient 29 % (109,5 → 77,9). Raison : l'entraînement tourne sur
**WikiText-2 train** (~2 M tokens uniques) et atteint **5,06 époques** — les données se répètent.
Le papier utilise C4 + WikiText-2 (~1 milliard de tokens). La prochaine marche n'est donc pas un
réglage d'optimiseur mais **plus de données** (`--dataset c4_wiki`, déjà supporté par le dépôt),
ce qui devrait rapprocher des 23,35 du papier sans changer le reste du pipeline.

### 5.2.1 ITQ de bout en bout — même budget, même protocole

| OPT-1.3B @0,55 BPW, 3000 steps | PPL WikiText-2 | Δ |
|---|---|---|
| sans ITQ | 109,53 | référence |
| **avec Joint-ITQ (`itq_n_iter=20`)** | **77,93** | **−29 % de PPL** |

Le gain observé à l'initialisation (erreur 0,826 → 0,760) se **transmet donc jusqu'au modèle final** :
la rotation orthogonale aligne les facteurs sur l'hypercube binaire, ce qui rend le seuil de
binarisation plus facile à traverser pendant le QAT. Génération ITQ : `"the capital of the French
Republic of France…"` (grammaticale, 66,5 tok/s). C'est le réglage retenu pour les runs longs.

Erreurs de reconstruction de l'init (Frobenius global) : 0,735 / 0,815 / 0,876 / 0,941 à
1,0 / 0,55 / 0,3 / 0,1 BPW (contre 0,546 / 0,703 / 0,808 / 0,913 pour la SVD tronquée seule).

> **Interprétation** : à l'initialisation, le modèle est détruit à *tous* les débits sub-1-bit
> (PPL ≈ 1,3–1,9·10⁴) — c'est attendu et cohérent avec le papier : c'est le QAT qui reconstruit la
> qualité. Le run 0,1 BPW le démontre (18,7× de gain en 200 steps, soit ~0,4 M tokens), mais reste
> très loin des 53,76 du papier, obtenus avec **~1 milliard de tokens sur 4×H100** (facteur ≈ 2 500×).

## 6. Référence communauté « H200 » — étalon réaliste (et validation de nos choix)

Le projet communautaire [LittleBitLLM/littlebit](https://huggingface.co/LittleBitLLM/littlebit)
publie la recette exacte et des mesures matérielles qui confirment nos analyses :

| Modèle / BPW | GPU | s/step | VRAM pic | Params entraînables | Init SVD+ITQ |
|---|---|---|---|---|---|
| Qwen3-0.6B @0.55 | 1×H100 80 Go | 1.03 | — | — | ~2,3 min |
| Qwen3-8B @0.55 | 1×H200 141 Go | 2.84 | ~107 Go | **3,7 B** | ~14 min |

Résultat publié (checkpoint *early*, **0,5 époque ≈ 10 350 steps ≈ 8,5 h sur H200**) :
Qwen3-8B @0.55 BPW → **PPL WikiText-2 24,48** (BF16 : 9,73), C4 32,10 (13,30), ARC-C 19,5 (à chance).

Enseignements retenus :
- Le nombre de paramètres entraînables est **énorme** (3,7 B pour un 8B à 0,55 BPW) → un QAT
  complet est hors de portée sur CPU ; même un opérateur « full GPU » se limite à 0,5 époque.
- `use_itq=True` (rotation **Joint-ITQ** de LittleBit-2) est **activé** dans leur recette
  (`--use_itq True --itq_n_iter 50`) : piste pour améliorer notre init (à tester localement).
- Leur `eval.py` souffre du même import circulaire DeepSpeed↔transformers (leur patch
  `eval-import-fix.patch` fait exactement ce que `scripts/lbcommon.py` fait ici).
- Confirmé : **le BPW annoncé porte sur les couches linéaires**, embeddings et `lm_head` restent BF16.

### 6.1 Checkpoint entraîné exécuté localement — **validation indépendante**

Le dépôt publié `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` (Qwen3-8B compressé à 0,55 BPW,
0,5 époque de distillation sur 1×H200) a été téléchargé (`models/lb-qwen3-8b-0.55bpw/`, 2,85 Go)
et exécuté **ici, sur CPU**, avec le loader officiel : 252 couches `LittleBitLinear` passées en mode
binarisé, **466 Mo de facteurs binaires + 24 Mo d'échelles** (soit ~0,49 BPW sur les linéaires).

| Métrique | Valeur publiée | **Notre mesure locale** |
|---|---|---|
| PPL WikiText-2 (146 chunks, seqlen 2048) | 24,48 | **24,4770** ✅ |
| Génération | — | `"The capital of France is"` → `" Paris, and the city is the capital of the French Republic. The capital is located in the French Riviera, which"` |
| Débit | — | 1,26 tok/s (CPU, fp32, 3 Go de RAM) |

C'est la double preuve attendue : (1) notre harnais de chargement/évaluation est **exact** (PPL
identique à 4 chiffres significatifs au chiffre publié), (2) un modèle LittleBit **entraîné**
génère du texte cohérent en local sur cette machine, sans GPU.

## 7. Limites assumées

- Le papier utilise **4×H100 et ~1B tokens** de QAT par configuration ; nous sommes sur CPU avec un
  budget de quelques millions de tokens. Les PPL finaux locaux seront donc **loin** des valeurs du
  papier : l'objet est de montrer un pipeline **correct et reproductible** (init → QAT → export →
  inférence locale), avec compression mémoire mesurée.
- Les embeddings et `lm_head` ne sont pas quantifiés (comme dans le code officiel : `lm_head`
  est exclu du patch), le BPW annoncé porte donc sur les couches linéaires internes.

## 9. Licences

- **Code upstream** (`SamsungLabs/LittleBit`) : Apache-2.0 — cloné dans `repo/` et **non vendu** ici
  (voir `UPSTREAM_NOTES.md` pour le commit exact et les correctifs nécessaires).
- **Ce dépôt** (scripts, rapports, README/NOTES) : travail original, à placer sous la licence de
  votre choix.
- **Checkpoint tiers évalué** `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` : CC BY-NC 4.0 (non
  commercial), plus la licence de `Qwen/Qwen3-8B`. Aucun poids n'est redistribué ici.
- **Modèles de base** `facebook/opt-125m` / `facebook/opt-1.3b` : licence Meta OPT (non redistribués ici).

## 10. Journal

- 2026-10-22 : env + repo + modèles ; tests unitaires OK ; sweep OPT-1.3B lancé ; premier
  checkpoint compressé OPT-125M (0.3 BPW) sauvegardé, rechargé et exécuté localement (PPL 2289).
- 2026-10-22 : pipeline complet reprenable (`run_full_pipeline.py`), garde VRAM (GPU caché si
  occupé), cron de surveillance GPU (30 min) ; découverte du checkpoint communautaire entraîné
  Qwen3-8B @0.55 BPW en cours de récupération.
- 2026-11-02 : meilleur modele : OPT-1.3B @0,55 BPW + ITQ, 6000 steps -> PPL 68,1 (x279 vs init),
  81 Mo de facteurs binaires, texte coherent, 69,9 tok/s ; ITQ = -29 % de PPL a budget egal ;
  limite identifiee : data-limited (5 epoques sur WikiText-2 train, 2 M tokens uniques).
- 2026-10-22 : reproduction exacte du FP du papier (OPT-1.3B **14,6237** vs 14,62) ; gain ITQ
  quantifié ; `RESULTS.md` généré ; QAT OPT-1.3B @0,1 BPW lancé ; publication de ce dépôt sur
  l'organisation GitHub JemaOS.
