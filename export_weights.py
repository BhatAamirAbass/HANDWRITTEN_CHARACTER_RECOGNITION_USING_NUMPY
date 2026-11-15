"""One-time export: .keras -> small NumPy .npz (run locally; needs h5py)."""
import io
import json
import zipfile

import h5py
import numpy as np

from config import MODEL_PATH, MODEL_DIR

OUT = MODEL_DIR / "weights.npz"
LAYERS = ["conv2d", "conv2d_1", "conv2d_2", "dense", "dense_1"]

with zipfile.ZipFile(MODEL_PATH) as z:
    h5_bytes = z.read("model.weights.h5")

arrays = {}
with h5py.File(io.BytesIO(h5_bytes)) as f:
    for name in LAYERS:
        arrays[f"{name}_w"] = f[f"layers/{name}/vars/0"][()]
        arrays[f"{name}_b"] = f[f"layers/{name}/vars/1"][()]

np.savez_compressed(OUT, **arrays)
print("Saved", OUT, {k: v.shape for k, v in arrays.items()})
