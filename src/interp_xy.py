# src/interp_xy.py
from __future__ import annotations
import numpy as np

def bilinear_sample_regular_xy(field, xs, ys, px, py, fill_value=np.nan):
    field = np.asarray(field, dtype=np.float32)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)

    ny, nx = field.shape
    ix = np.searchsorted(xs, px) - 1
    iy = np.searchsorted(ys, py) - 1

    ok = (ix >= 0) & (ix < nx - 1) & (iy >= 0) & (iy < ny - 1)
    out = np.full(px.shape, fill_value, dtype=np.float32)
    if not np.any(ok):
        return out

    ix0 = ix[ok]; iy0 = iy[ok]
    x0 = xs[ix0]; x1 = xs[ix0 + 1]
    y0 = ys[iy0]; y1 = ys[iy0 + 1]

    tx = (px[ok] - x0) / (x1 - x0)
    ty = (py[ok] - y0) / (y1 - y0)
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)

    f00 = field[iy0, ix0]
    f10 = field[iy0, ix0 + 1]
    f01 = field[iy0 + 1, ix0]
    f11 = field[iy0 + 1, ix0 + 1]

    out_ok = (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11
    out[ok] = out_ok.astype(np.float32)
    return out
