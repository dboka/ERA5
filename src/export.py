# src/export.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import rasterio
from rasterio.transform import from_origin


@dataclass
class GeoTiffWriteOptions:
    nodata: float = -9999.0
    epsg: int = 3059
    compress: str = "deflate"
    predictor: int = 2
    tiled: bool = True
    blocksize: int = 512
    origin_lower: bool = True  # True if array row0=south (imshow origin="lower")


def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={a.shape}")
    return a


def _regular_step(vals: np.ndarray, name: str) -> float:
    d = np.diff(vals.astype(float))
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError(f"{name}: cannot compute step (diff empty)")
    step = float(np.median(d))
    if not np.isfinite(step) or abs(step) <= 0:
        raise ValueError(f"{name}: invalid step={step}")
    return step


def write_geotiff_lks92(
    out_tif: str,
    field2d: np.ndarray,   # (ny,nx)
    xs: np.ndarray,        # 1D x (nx)
    ys: np.ndarray,        # 1D y (ny)
    opts: GeoTiffWriteOptions = GeoTiffWriteOptions(),
) -> None:
    arr = np.asarray(field2d, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"field2d must be 2D, got shape={arr.shape}")

    xs = _as_1d(xs).astype(float)
    ys = _as_1d(ys).astype(float)

    ny, nx = arr.shape
    if xs.size != nx:
        raise ValueError(f"xs length mismatch: len(xs)={xs.size} vs nx={nx}")
    if ys.size != ny:
        raise ValueError(f"ys length mismatch: len(ys)={ys.size} vs ny={ny}")

    # normalize axes to ascending
    x_asc = xs[0] < xs[-1]
    y_asc = ys[0] < ys[-1]

    xs2 = xs if x_asc else xs[::-1]
    ys2 = ys if y_asc else ys[::-1]
    arr2 = arr.copy()
    if not x_asc:
        arr2 = arr2[:, ::-1]
    if not y_asc:
        arr2 = arr2[::-1, :]

    arr_out = np.array(arr2, dtype=np.float32)
    arr_out[~np.isfinite(arr_out)] = opts.nodata

    dx = abs(_regular_step(xs2, "xs"))
    dy = abs(_regular_step(ys2, "ys"))

    if opts.origin_lower:
        arr_out = np.flipud(arr_out)

    west = float(xs2.min() - dx / 2.0)
    north = float(ys2.max() + dy / 2.0)
    transform = from_origin(west, north, dx, dy)

    profile = dict(
        driver="GTiff",
        height=arr_out.shape[0],
        width=arr_out.shape[1],
        count=1,
        dtype="float32",
        crs=f"EPSG:{opts.epsg}",
        transform=transform,
        nodata=opts.nodata,
        compress=opts.compress,
        predictor=opts.predictor,
        tiled=opts.tiled,
    )
    if opts.tiled:
        profile["blockxsize"] = int(opts.blocksize)
        profile["blockysize"] = int(opts.blocksize)

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr_out, 1)
