"""Small helpers shared by the local LittleBit scripts."""
import importlib
import json
import os
import subprocess


def decide_device_env(min_free_mb=None):
    """Hide the GPU (CUDA_VISIBLE_DEVICES="") when it is too occupied for this workload.

    MUST be called BEFORE importing torch: the upstream LittleBitLinear moves its SVD work to
    cuda whenever torch.cuda.is_available() is true, which OOMs when another workload (here
    LM Studio's llama-server, ~8.5 GB) already holds the 12 GB device.

    Explicit hints: LITTLEBIT_DEVICE=cpu|cuda forces the choice; LITTLEBIT_MIN_FREE_MB sets the
    free-memory threshold (default 5000 MB).
    """
    if os.environ.get("LITTLEBIT_DEVICE", "").lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if os.environ.get("LITTLEBIT_DEVICE", "").lower() == "cuda":
        return "cuda"
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return "cpu"
    if min_free_mb is None:
        min_free_mb = int(os.environ.get("LITTLEBIT_MIN_FREE_MB", "5000"))
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        free_mb = max(int(x.strip()) for x in out.splitlines() if x.strip().isdigit())
    except Exception:
        free_mb = 0
    if free_mb < min_free_mb:
        print(f"INFO: free VRAM {free_mb} MB < {min_free_mb} MB -> running on CPU (hiding CUDA)")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    print(f"INFO: free VRAM {free_mb} MB -> CUDA available")
    return "cuda"


def model_type_of(model_dir_or_id):
    cfg = os.path.join(model_dir_or_id, "config.json")
    if os.path.isfile(cfg):
        with open(cfg) as f:
            return json.load(f).get("model_type", "")
    return ""


def preimport_model_module(model_dir_or_id):
    """Import the transformers modeling module before any `torch.device('meta')` context.

    The official loader (quantization/hub.py -> load_quantized_model) builds the skeleton model
    inside `with torch.device("meta")`. Importing a modeling module there can fail because some
    modules create tensors and call .item() at import time (e.g. transformers 4.51 / OPT:
    "Tensor.item() cannot be called on meta tensors"). Importing it beforehand is enough.
    """
    mt = model_type_of(model_dir_or_id)
    if not mt:
        return None
    try:
        return importlib.import_module(f"transformers.models.{mt}.modeling_{mt}")
    except Exception as e:  # pragma: no cover
        print(f"WARN: could not pre-import modeling_{mt}: {e}")
        return None
