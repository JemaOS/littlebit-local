# Notes sur le code officiel (bugs rencontrés et contournements)

Repo upstream : [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit),
commit utilisé : `42d658b0c79f76450b34b6a3547462c7cdc3e1a0` (2026-10-09, *Fix bit extraction in
binary_unpacker #22*) — c'est aussi le commit que la recette communautaire « LittleBitLLM » épingle.

Ces problèmes ont été rencontrés en exécutant le code **tel quel**, sans y toucher : les
contournements vivent dans nos scripts (`scripts/lbcommon.py`, ordre d'import, garde VRAM).

## 1. Import circulaire `quantization.modules` ↔ `quantization.utils`

```
ImportError: cannot import name 'LittleBitLinear' from partially initialized module
'quantization.modules' (most likely due to a circular import)
```

`quantization/modules/littlebit.py` fait `from quantization.utils.binary_packer import binary_packer`,
ce qui déclenche `quantization/utils/__init__.py` → `quant_util` → `from quantization.modules import
LittleBitLinear` sur un paquet en cours d'initialisation.

*Contournement* : importer `quantization.utils` **avant** `quantization.modules` (c'est ce que fait
`main.py`). Correctif propre côté upstream : rendre `utils/__init__.py` paresseux (`__getattr__`).

## 2. `LittleBitLinear.split_dim_used` → `AttributeError`

```python
@property
def split_dim_used(self):          # lit self._split_dim
    ...
```
or le buffer réellement enregistré est `self._split_dim_final` :
```
AttributeError: 'LittleBitLinear' object has no attribute 'split_dim_used'
```
Conséquences : toute inspection du modèle (`print(model)`, `extra_repr`, scripts d'analyse) casse.
*Contournement* : utiliser `int(m._split_dim_final)`. Correctif : corriger la propriété (1 ligne).

## 3. Construction sous `torch.device("meta")` et import des modeling modules

`quantization/hub.py::load_quantized_model` construit le squelette dans
`with torch.device("meta")`. L'import de `transformers.models.opt.modeling_opt` (et d'autres)
échoue alors à l'intérieur de ce contexte :

```
RuntimeError: Failed to import transformers.models.opt.modeling_opt because of the following error:
Tensor.item() cannot be called on meta tensors
```

*Contournement* : pré-importer `transformers.models.<model_type>.modeling_<model_type>` avant
d'entrer dans le contexte (`scripts/lbcommon.py::preimport_model_module`).

## 4. `deepspeed` doit être importé avant `lm_eval`/`transformers` (eval.py)

Sans cela, `transformers.modeling_utils` charge `deepspeed` en plein init et on retombe sur le
problème circulaire n° 1. La recette communautaire publie le même correctif sous
`recipe/patches/eval-import-fix.patch`.

## 5. `_decompose_matrix` force CUDA dès qu'elle est visible

```python
calc_device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else original_device
```
Sur une machine où la carte est **déjà occupée par un autre workload** (ici LM Studio, ~8,5 Go/12 Go),
l'initialisation SVD part sur le GPU et échoue en `CUDA error: out of memory` — même quand on a
demandé le CPU.

*Contournement* : `scripts/lbcommon.py::decide_device_env()` (appelé **avant** `import torch`)
positionne `CUDA_VISIBLE_DEVICES=""` si la VRAM libre est inférieure à un seuil, ce qui rend
`torch.cuda.is_available()` faux dans le code upstream.

## 6. Divers observés (pas des bugs bloquants)

- `main.py` importe `deepspeed`, `GPUtil`, `wandb` **inconditionnellement** → notre entraînement
  local (`scripts/train_local.py`) reprend `KDTrainer` + `apply_littlebit_patch` sans ces dépendances.
- `min_split_dim: 8` (défaut) : sur de petites matrices à très bas BPW, le rang est « collé » au
  plancher, donc le BPW effectif réel est supérieur à la cible (ex. OPT-125M à 0,1 BPW → 0,12–0,24
  selon la couche). À garder en tête lors de la comparaison taille/qualité.
- La perte KD du dépôt (`utils/kd_utils.py`) fait un `batchmean` sur la dimension **batch** et non
  « tokens × vocab » : l'ordre de grandeur est ≈ `seq_len × KL` nats, ce qui n'est pas la convention
  habituelle (utile à savoir avant de comparer des courbes de perte).
- `get_eval_loaders` force `tokenizer.bos_token_id = 1` et `eos_token_id = 2` (convention Llama/OPT) ;
  inoffensif pour GPT-2/OPT/Qwen (aucun token spécial ajouté) mais à ne pas réutiliser ailleurs.
