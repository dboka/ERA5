#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
import xarray as xr

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_nc", required=True)
    p.add_argument("--out_nc", required=True)
    return p.parse_args()

def main():
    a = parse_args()
    ds = xr.open_dataset(a.in_nc, engine="netcdf4")

    if "tp" not in ds:
        raise ValueError(f"'tp' not found. Vars: {list(ds.data_vars)}")

    tp = ds["tp"]
    print("[tp] units:", tp.attrs.get("units"))
    print("[tp] dims:", tp.dims, "shape:", tp.shape)

    # Detect time dim name
    tdim = "valid_time" if "valid_time" in tp.dims else ("time" if "time" in tp.dims else None)
    if tdim is None:
        raise ValueError(f"Cannot find time dim in {tp.dims}")

    # 1) increments
    inc = tp.diff(tdim)

    # 2) reset handling: negative increments -> 0
    inc = inc.where(inc >= 0, 0.0)

    # 3) sum to monthly total (mm)
    tp_mm_month = inc.sum(tdim) * 1000.0
    tp_mm_month = tp_mm_month.astype("float32")
    tp_mm_month.name = "tp_mm_month"
    tp_mm_month.attrs["units"] = "mm"
    tp_mm_month.attrs["long_name"] = "Monthly total precipitation (from hourly increments)"

    out = xr.Dataset(
        {"tp_mm_month": tp_mm_month},
        coords={"latitude": ds["latitude"], "longitude": ds["longitude"]},
    )

    out.to_netcdf(a.out_nc)
    print("Wrote:", a.out_nc)
    print("Monthly total mm min/max:", float(tp_mm_month.min()), float(tp_mm_month.max()))

if __name__ == "__main__":
    main()
