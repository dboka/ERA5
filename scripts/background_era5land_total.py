# scripts/make_era5_monthly_total.py
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

    # time dim
    tdim = "valid_time" if "valid_time" in tp.dims else ("time" if "time" in tp.dims else None)
    if tdim is None:
        raise ValueError(f"Cannot find time dim in {tp.dims}")

    # 1) increments
    inc = tp.diff(tdim)

    # 2) negative increments -> 0 (reset handling)
    inc = inc.where(inc >= 0, 0.0)

    # 3) monthly total (mm)
    tp_mm_month = (inc.sum(tdim) * 1000.0).astype("float32")
    tp_mm_month.name = "tp_mm_month"
    tp_mm_month.attrs["units"] = "mm"
    tp_mm_month.attrs["long_name"] = "Monthly total precipitation (from hourly increments)"

    # coords: keep whatever ds provides
    # Most ERA5 files have latitude/longitude coords
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

if __name__ == "__main__":
    main()
