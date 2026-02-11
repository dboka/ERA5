# scripts/make_era5_monthly_total.py
from __future__ import annotations

import argparse
import numpy as np
import xarray as xr


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert ERA5/ERA5-Land hourly tp to monthly total precipitation (mm)"
    )
    p.add_argument("--in_nc", required=True, help="Input hourly NetCDF with variable 'tp'")
    p.add_argument("--out_nc", required=True, help="Output NetCDF with variable 'tp_mm_month' (mm)")
    p.add_argument(
        "--mode",
        choices=["auto", "sum", "diff"],
        default="auto",
        help="auto=detect (recommended), sum=sum(tp) for hourly step-accum, diff=diff+sum for running accumulation",
    )
    return p.parse_args()


def main():
    a = parse_args()

    ds = xr.open_dataset(a.in_nc, engine="netcdf4")

    if "tp" not in ds.data_vars:
        raise ValueError(f"'tp' not found. Vars: {list(ds.data_vars)}")

    tp = ds["tp"]

    # Detect time dim name
    tdim = "valid_time" if "valid_time" in tp.dims else ("time" if "time" in tp.dims else None)
    if tdim is None:
        raise ValueError(f"Cannot find time dim in {tp.dims}")

    print("[tp] units:", tp.attrs.get("units"))
    print("[tp] dims :", tp.dims, "shape:", tp.shape)
    print("[tp] time dim:", tdim, "time size:", int(tp.sizes[tdim]))

    # ERA5 tp is in meters -> convert to mm
    tp_mm = (tp * 1000.0).astype("float32")

    # --- detect whether tp is running accumulation or hourly-step accumulation ---
    mode = a.mode
    if mode == "auto":
        # If running accumulation, tp should be mostly non-decreasing in time.
        # Compute fraction of negative diffs on a small sample window.
        ntime = int(tp.sizes[tdim])
        if ntime < 2:
            # cannot diff -> fallback to sum (at least gives something)
            mode = "sum"
            frac_neg = 0.0
        else:
            k = min(48, ntime)  # first 48 hours if available
            tp_s = tp_mm.isel({tdim: slice(0, k)})
            d = tp_s.diff(tdim)
            frac_neg = float((d < -1e-6).mean().values)
            # heuristic threshold
            mode = "diff" if frac_neg < 0.01 else "sum"

        print(f"[detect] frac negative diffs (sample) = {frac_neg:.4f} -> mode='{mode}'")

    # --- compute monthly total (mm) ---
    if mode == "sum":
        # tp already represents precipitation over the hour (step accumulation)
        tp_mm_month = tp_mm.sum(tdim)
        long_name = "Monthly total precipitation (sum of hourly tp)"
    elif mode == "diff":
        # tp is running accumulation -> use increments
        inc = tp_mm.diff(tdim)
        inc = inc.where(inc >= 0, 0.0)  # reset handling
        tp_mm_month = inc.sum(tdim)
        long_name = "Monthly total precipitation (from hourly increments of running accumulation)"
    else:
        raise ValueError(f"Unexpected mode: {mode}")

    tp_mm_month = tp_mm_month.astype("float32")
    tp_mm_month.name = "tp_mm_month"
    tp_mm_month.attrs["units"] = "mm"
    tp_mm_month.attrs["long_name"] = long_name

    # Normalize coord names
    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot find lat/lon coords. Coords: {list(ds.coords)}")

    out = xr.Dataset(
        {"tp_mm_month": tp_mm_month},
        coords={"latitude": ds[lat_name], "longitude": ds[lon_name]},
    )

    out.to_netcdf(a.out_nc)
    print("Wrote:", a.out_nc)
    print("Monthly total mm min/max:", float(tp_mm_month.min()), float(tp_mm_month.max()))
    print("NaN count:", int(np.isnan(tp_mm_month.values).sum()))


if __name__ == "__main__":
    main()
