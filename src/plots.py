# src/plots.py
from __future__ import annotations

import os
from typing import Optional, Literal

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


ScaleMode = Literal["obs", "percentile", "minmax", "fixed"]


def save_field_png(
    field2d,
    title: str,
    out_png: str,
    mask: np.ndarray | None = None,
    figsize=(7, 4),
    dpi: int = 150,
    interpolation: str = "nearest",
    scale: ScaleMode = "minmax",   # <-- CHANGE HERE
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    obs_max: Optional[float] = None,
    pad_mm: float = 10.0,
    delete_existing: bool = True,
) -> None:
    """
    Debug plot.

    scale modes:
      - "obs":      vmin=0, vmax=obs_max+pad_mm   (REQUIRED: obs_max)
      - "percentile": vmin/vmax from 2..98 percentiles of finite
      - "minmax":   vmin/vmax from finite min/max
      - "fixed":    use provided vmin/vmax (REQUIRED: vmin and vmax)

    mask:
      - if provided: outside mask => NaN
      - NaN are masked in plotting (not rendered)
    """
    arr = np.array(field2d, dtype=np.float32, copy=True)

    if mask is not None:
        m = np.array(mask, dtype=bool)
        arr[~m] = np.nan

    finite = np.isfinite(arr)
    if not finite.any():
        # no data
        vmin2, vmax2 = 0.0, 1.0
        plot_arr = np.ma.masked_invalid(arr)
    else:
        plot_arr = np.ma.masked_invalid(arr)

        if scale == "obs":
            if obs_max is None:
                raise ValueError('save_field_png(scale="obs") requires obs_max=...')
            vmin2 = 0.0 if vmin is None else float(vmin)
            vmax2 = float(obs_max) + float(pad_mm) if vmax is None else float(vmax)

        elif scale == "percentile":
            pvmin = float(np.nanpercentile(arr, 2))
            pvmax = float(np.nanpercentile(arr, 98))
            # fallback if degenerate
            if (not np.isfinite(pvmin)) or (not np.isfinite(pvmax)) or (pvmin == pvmax):
                pvmin = float(np.nanmin(arr))
                pvmax = float(np.nanmax(arr))
            vmin2 = pvmin if vmin is None else float(vmin)
            vmax2 = pvmax if vmax is None else float(vmax)

        elif scale == "minmax":
            vmin2 = float(np.nanmin(arr)) if vmin is None else float(vmin)
            vmax2 = float(np.nanmax(arr)) if vmax is None else float(vmax)

        elif scale == "fixed":
            if vmin is None or vmax is None:
                raise ValueError('save_field_png(scale="fixed") requires vmin=... and vmax=...')
            vmin2, vmax2 = float(vmin), float(vmax)

        else:
            raise ValueError(f"Unknown scale mode: {scale}")

    # Safety
    if (not np.isfinite(vmin2)) or (not np.isfinite(vmax2)) or (vmin2 == vmax2):
        vmin2, vmax2 = 0.0, 1.0

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    if delete_existing and os.path.exists(out_png):
        try:
            os.remove(out_png)
        except OSError:
            pass

    plt.figure(figsize=figsize)
    plt.title(title)
    im = plt.imshow(plot_arr, origin="lower", vmin=vmin2, vmax=vmax2, interpolation=interpolation)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
