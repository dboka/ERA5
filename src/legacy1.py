import numpy as np
import xarray as xr

def load_era5land_monthly_tp_mm_to_lvgrid(
    era_nc_path: str,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Reads ERA5-Land monthly total precipitation from NetCDF and interpolates to LV grid.
    Handles both:
      - tp_mm_month (units mm)  [processed file]
      - tp (units m)            [raw file]
    Returns background_filled float32 (mm/month) with 0 outside mask.
    """
    ds = xr.open_dataset(era_nc_path, engine="netcdf4")

    # Try new format first (tp_mm_month in mm), then fall back to old format (tp in m)
    if "tp_mm_month" in ds.data_vars:
        tp_mm = ds["tp_mm_month"]
        units = (tp_mm.attrs.get("units") or "").lower()
        print(f"[background] Using tp_mm_month variable with units: {units}")

        # drop time-like dims if present
        if "valid_time" in tp_mm.dims:
            tp_mm = tp_mm.squeeze("valid_time", drop=True)
        elif "time" in tp_mm.dims:
            tp_mm = tp_mm.squeeze("time", drop=True)

        # optional sanity (don’t hard-fail if units missing, but you can)
        if units and units != "mm":
            raise ValueError(f"Expected tp_mm_month units 'mm', got '{units}'")

    elif "tp" in ds.data_vars:
        tp = ds["tp"]
        units = (tp.attrs.get("units") or "").lower()
        print(f"[background] Using tp variable with units: {units}")

        # drop time-like dims if present
        if "valid_time" in tp.dims:
            tp = tp.squeeze("valid_time", drop=True)
        elif "time" in tp.dims:
            tp = tp.squeeze("time", drop=True)

        if units != "m":
            raise ValueError(f"Expected tp units 'm', got '{units}'")

        tp_mm = tp * 1000.0  # m -> mm/month

    else:
        raise ValueError(
            f"Expected 'tp_mm_month' or 'tp' in {era_nc_path}. Found: {list(ds.data_vars)}"
        )

    # ensure latitude increasing for interp
    tp_mm = tp_mm.sortby("latitude")

    bg = tp_mm.interp(
        latitude=(("y", "x"), lat2d),
        longitude=(("y", "x"), lon2d),
        method="linear",
    ).values.astype(np.float32)

    # GridPP: no NaN
    bg[~mask] = 0.0
    bg[np.isnan(bg)] = 0.0
    return bg
