# scripts/download_era5_hourly.py
from __future__ import annotations
import argparse
import os
import calendar
import cdsapi

def parse_args():
    p = argparse.ArgumentParser("Download ERA5 hourly total_precipitation (tp) for bbox/month")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--out", required=True, help="Output netcdf path, e.g. data/era5/era5_tp_hourly_2025_09.nc")
    # Latvia bbox (north, west, south, east)
    p.add_argument("--north", type=float, default=58.2)
    p.add_argument("--west", type=float, default=20.5)
    p.add_argument("--south", type=float, default=55.5)
    p.add_argument("--east", type=float, default=28.5)
    return p.parse_args()

def main():
    a = parse_args()
    ndays = calendar.monthrange(a.year, a.month)[1]

    req = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": "total_precipitation",
        "year": f"{a.year}",
        "month": f"{a.month:02d}",
        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
        "time": [f"{h:02d}:00" for h in range(0, 24)],
        "area": [a.north, a.west, a.south, a.east],
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    c = cdsapi.Client()
    c.retrieve("reanalysis-era5-single-levels", req, a.out)
    print("Saved:", os.path.abspath(a.out))

if __name__ == "__main__":
    main()