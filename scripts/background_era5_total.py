# src/background_era5_total.py
from __future__ import annotations
import numpy as np
import xarray as xr

def load_era5_monthly_total_mm_to_lvgrid(monthly_total_nc: str,
                                        lon2d: np.ndarray,
                                        lat2d: np.ndarray,
                                        mask: np.ndarray) -> np.ndarray:
    """
    Input netcdf: variable tp_mm_month (mm), dims (latitude, longitude) OR (time, lat, lon).
    Output: 2D float32 array on LV grid (ny,nx), with 0 outside mask and no NaN.
    ERA5 is global -> should not have coastal NaN holes like ERA5-Land.
    """
    ds = xr.open_dataset(monthly_total_nc, engine="netcdf4")

    if "tp_mm_month" in ds.data_vars:
        v = ds["tp_mm_month"]
    elif "tp" in ds.data_vars:
        v = ds["tp"]
    else:
        raise ValueError(f"Expected tp_mm_month (or tp). Found: {list(ds.data_vars)}")

    if "time" in v.dims and v.sizes.get("time", 0) == 1:
        v = v.isel(time=0)

    # sort coords for safe interp
    if "latitude" in v.coords:
        v = v.sortby("latitude")
    if "longitude" in v.coords:
        v = v.sortby("longitude")

    bg = v.interp(
        latitude=(("y", "x"), lat2d),
        longitude=(("y", "x"), lon2d),
        method="linear",
    ).values.astype(np.float32)

    # Safety
    bg[~mask] = 0.0
    bg[np.isnan(bg)] = 0.0
    return bg
