# src/background_era5_total.py
from __future__ import annotations

import numpy as np
import xarray as xr


def load_era5_monthly_total_mm_to_lvgrid(
    monthly_total_nc: str,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    FULL ERA5 background loader (no ERA5-Land land/sea mask holes).

    Input netcdf: variable tp_mm_month (mm) OR tp (could be mm already OR meters),
                 dims typically (latitude, longitude) or (time, latitude, longitude).
    Output: 2D float32 array on LV grid (ny,nx), with 0 outside mask and no NaN.

    Notes:
    - ERA5 usually provides tp in meters if you download hourly and don't convert.
      This loader does NOT force unit conversion; main.py already has heuristics.
    - Uses linear interpolation to LV 2D lon/lat grid.
    """
    ds = xr.open_dataset(monthly_total_nc, engine="netcdf4")

    # Pick variable
    if "tp_mm_month" in ds.data_vars:
        v = ds["tp_mm_month"]
    elif "tp" in ds.data_vars:
        v = ds["tp"]
    else:
        raise ValueError(f"Expected tp_mm_month (or tp). Found: {list(ds.data_vars)}")

    # Drop singleton time dim if present
    for tdim in ("time", "valid_time"):
        if tdim in v.dims and v.sizes.get(tdim, 0) == 1:
            v = v.isel({tdim: 0})

    # Ensure coordinate names are ERA-style
    # If file uses lat/lon, normalize to latitude/longitude for interp()
    if "lat" in v.coords and "latitude" not in v.coords:
        v = v.rename({"lat": "latitude"})
    if "lon" in v.coords and "longitude" not in v.coords:
        v = v.rename({"lon": "longitude"})

    if "latitude" not in v.coords or "longitude" not in v.coords:
        raise ValueError(
            f"Cannot find latitude/longitude coords. Coords: {list(v.coords)}"
        )

    # Sort coords (important if latitude is descending)
    v = v.sortby("latitude")
    v = v.sortby("longitude")

    # Interpolate to LV grid 2D coords
    bg = v.interp(
        latitude=(("y", "x"), lat2d),
        longitude=(("y", "x"), lon2d),
        method="linear",
    ).values.astype(np.float32)

    # Safety for GridPP: no NaN
    bg[~mask] = 0.0
    bg[np.isnan(bg)] = 0.0
    return bg
