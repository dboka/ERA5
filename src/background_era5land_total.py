# src/background_era5land_total.py
from __future__ import annotations
import numpy as np
import xarray as xr

def load_era5land_monthly_total_mm_to_lvgrid(monthly_total_nc: str,
                                            lon2d: np.ndarray,
                                            lat2d: np.ndarray,
                                            mask: np.ndarray) -> np.ndarray:
    """
    Input netcdf: variable tp_mm_month or tp (in m or mm), dims (latitude, longitude) OR (time, lat, lon).
    Automatically converts meters -> mm if needed.
    Output: 2D float32 array on LV grid (ny,nx) in mm, with 0 outside mask and no NaN.
    """
    ds = xr.open_dataset(monthly_total_nc, engine="netcdf4")

    if "tp_mm_month" in ds.data_vars:
        v = ds["tp_mm_month"]
    elif "tp" in ds.data_vars:
        v = ds["tp"]
    else:
        raise ValueError(f"Expected tp_mm_month (or tp). Found: {list(ds.data_vars)}")

    # Check units and convert if needed
    units = (v.attrs.get("units") or "").strip().lower()
    print(f"[loader] Variable: {v.name}, units: '{units}'")
    
    # noņem time dim ja ir
    if "time" in v.dims and v.sizes.get("time", 0) == 1:
        v = v.isel(time=0)

    # ERA lat var būt dilstošs; interp labāk ar sakārtotu
    if "latitude" in v.coords:
        v = v.sortby("latitude")
    if "longitude" in v.coords:
        v = v.sortby("longitude")

    # Debug ranges BEFORE interpolation
    v_min_before = float(np.nanmin(v.values))
    v_max_before = float(np.nanmax(v.values))
    print(f"[loader] raw values before interp: min={v_min_before:.6f}, max={v_max_before:.6f}")

    # Interp uz 2D koordinātēm
    bg = v.interp(
        latitude=(("y", "x"), lat2d),
        longitude=(("y", "x"), lon2d),
        method="linear",
    ).values.astype(np.float32)

    # Convert units: m -> mm
    if units in ["m", "meters", "meter"]:
        print(f"[loader] Converting from meters to mm (×1000)")
        bg = bg * 1000.0
    elif units in ["mm", "millimeters", "millimeter", ""]:
        print(f"[loader] Already in mm or units not specified, using as-is")
    else:
        print(f"[loader] WARNING: Unknown units '{units}', assuming mm")

    print(f"[loader] After interp: min={float(np.nanmin(bg)):.6f}, max={float(np.nanmax(bg)):.6f}")

    # Fill for GridPP: no NaN
    bg[~mask] = 0.0
    bg[np.isnan(bg)] = 0.0
    return bg
