# src/plots.py
import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_field_png(field2d, title: str, out_png: str):
    arr = np.array(field2d, dtype=np.float32)

    # robust vmin/vmax ignoring NaN
    finite = np.isfinite(arr)
    if finite.any():
        vmin = float(np.nanpercentile(arr, 2))
        vmax = float(np.nanpercentile(arr, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
    else:
        vmin, vmax = 0.0, 1.0

    plt.figure(figsize=(7, 4))
    plt.title(title)
    im = plt.imshow(arr, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
