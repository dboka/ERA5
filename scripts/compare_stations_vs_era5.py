#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import xarray as xr

def parse_args():
    p = argparse.ArgumentParser(description="Compare station monthly precip vs ERA5-Land monthly total")
    p.add_argument("--stations", required=True, help="Stations CSV with lon,lat,month_sum,Gads,Menesis")
    p.add_argument("--era_nc", required=True, help="ERA5-Land monthly total nc (tp_mm_month)")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--method", default="linear", choices=["linear", "nearest"])
    return p.parse_args()

def main():
    a = parse_args()

    # --- Stations ---
    df = pd.read_csv(a.stations, sep=";", decimal=".", encoding="utf-8", low_memory=False)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    need = ["gh_id", "Gads", "Menesis", "month_sum", "lon", "lat"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in stations file: {missing}\nHave: {df.columns.tolist()}")

    sub = df[(df["Gads"] == a.year) & (df["Menesis"] == a.month)].copy()
    if len(sub) == 0:
        raise RuntimeError(f"No stations for {a.year}-{a.month:02d}")

    sub = sub.dropna(subset=["lon", "lat", "month_sum"]).copy()
    sub = sub.sort_values("gh_id").reset_index(drop=True)

    lon_s = sub["lon"].to_numpy(np.float32)
    lat_s = sub["lat"].to_numpy(np.float32)
    obs = sub["month_sum"].to_numpy(np.float32)

    # --- ERA5 ---
    ds = xr.open_dataset(a.era_nc, engine="netcdf4")

    if "tp_mm_month" not in ds:
        raise ValueError(f"'tp_mm_month' not found in {a.era_nc}. Vars: {list(ds.data_vars)}")

    v = ds["tp_mm_month"]

    # xarray interp wants latitude increasing
    v = v.sortby("latitude").sortby("longitude")

    # Interpolate to station points
    era_at = v.interp(
        latitude=xr.DataArray(lat_s, dims="points"),
        longitude=xr.DataArray(lon_s, dims="points"),
        method=a.method,
    ).values.astype(np.float32)

    # --- Errors ---
    err = era_at - obs
    abs_err = np.abs(err)

    # avoid divide-by-zero explosions
    rel_err = np.where(obs != 0, (err / obs) * 100.0, np.nan).astype(np.float32)

    mae = float(np.nanmean(abs_err))
    rmse = float(np.sqrt(np.nanmean(err**2)))
    bias = float(np.nanmean(err))

    out = pd.DataFrame({
        "gh_id": sub["gh_id"].values,
        "year": a.year,
        "month": a.month,
        "lon": lon_s,
        "lat": lat_s,
        "obs_mm": obs,
        "era5_mm": era_at,
        "err_mm": err,
        "abs_err_mm": abs_err,
        "rel_err_pct": rel_err,
    })

    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    out.to_csv(a.out_csv, index=False)

    print("Wrote:", a.out_csv)
    print(f"Stations: {len(out)}  MAE={mae:.2f} mm  RMSE={rmse:.2f} mm  BIAS={bias:.2f} mm")
    print("First rows:\n", out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
