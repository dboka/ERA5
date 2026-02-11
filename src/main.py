# src/main.py
from __future__ import annotations

import argparse
import os
import numpy as np
import gridpp

from config import Paths, RunConfig
from io_data import read_grid_csv, read_stations_csv
from grid_build import build_lv_grid_cartesian, build_points_cartesian
from oi_precip import log_ratio_oi, loocv_log_ratio
from plots import ensure_dir, save_field_png, save_text

# FULL ERA5 monthly-total loader (no ERA5-Land coastline NaN holes)
from background_era5_total import load_era5_monthly_total_mm_to_lvgrid


def parse_args():
    p = argparse.ArgumentParser(
        description="LV precip interpolation with GridPP (log-ratio OI + LOOCV)"
    )
    p.add_argument("--grid", required=True, help="LV grid CSV (1x1_LV_grid_2024_xy2.csv)")
    p.add_argument("--stations", required=True, help="Stations monthly CSV")
    p.add_argument("--year", type=int, default=2013)
    p.add_argument("--month", type=int, default=1)
    p.add_argument("--out", default="out")

    # ERA5 background (monthly total precipitation in mm)
    # Expect output from scripts/make_era5_monthly_total.py
    p.add_argument(
        "--era_nc",
        default="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/data/era5/era5_tp_mm_month_2013_01.nc",
        help="ERA5 monthly total precipitation NetCDF (mm), e.g. era5_tp_mm_month_2025_09.nc",
    )

    # OI params
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--L", type=float, default=50000.0)
    p.add_argument("--pobs_d", type=float, default=0.3)
    p.add_argument("--max_points", type=int, default=30)

    # Cross-validation exclusion radius (m)
    # Use small value (e.g. 1..50) for strict LOOCV; larger (e.g. 2000) for spatial CV
    p.add_argument("--cv_radius", type=float, default=2000.0)

    p.add_argument("--threads", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()

    paths = Paths(grid_csv=args.grid, stations_csv=args.stations, out_dir=args.out)
    cfg = RunConfig(
        year=args.year,
        month=args.month,
        eps=args.eps,
        L=args.L,
        pobs_d=args.pobs_d,
        max_points=args.max_points,
        cv_radius_m=args.cv_radius,
        omp_threads=args.threads,
    )

    ensure_dir(paths.out_dir)
    gridpp.set_omp_threads(cfg.omp_threads)

    # 1) grid
    g = read_grid_csv(paths.grid_csv)
    griddata = build_lv_grid_cartesian(g)
    griddata.xs2d, griddata.ys2d = np.meshgrid(griddata.xs, griddata.ys)

    # 2) stations
    df = read_stations_csv(paths.stations_csv)
    sub = df[(df["Gads"] == cfg.year) & (df["Menesis"] == cfg.month)].copy()
    if len(sub) == 0:
        raise RuntimeError(f"No station rows found for {cfg.year}-{cfg.month:02d}")

    # stable order for debug
    if "gh_id" in sub.columns:
        sub = sub.sort_values("gh_id")

    points, obs, px, py = build_points_cartesian(sub)

    # Drop stations outside grid bounding box (in projected x/y)
    xmin, xmax = float(griddata.xs.min()), float(griddata.xs.max())
    ymin, ymax = float(griddata.ys.min()), float(griddata.ys.max())
    inside = (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
    n_out = int((~inside).sum())
    if n_out > 0:
        print(f"[stations] WARNING: {n_out} stations outside grid bbox. Dropping them.")
        sub = sub.loc[inside].copy()
        points, obs, px, py = build_points_cartesian(sub)

    if len(obs) < 3:
        raise RuntimeError("[stations] Too few stations left after bbox filter.")

    # 3) background (ERA5 monthly total in mm) interpolated to LV lon/lat grid
    era_nc = args.era_nc
    if not os.path.exists(era_nc):
        raise FileNotFoundError(f"ERA5 background file not found: {era_nc}")

    background = load_era5_monthly_total_mm_to_lvgrid(
        era_nc, griddata.lon2d, griddata.lat2d, griddata.mask
    )

    # Safety: verify/convert units (heuristics)
    bg_max = float(np.nanmax(background))
    bg_min_valid = float(np.nanmin(background[background > 0])) if (background > 0).any() else 0.0
    print(f"[background] Loaded: min_valid={bg_min_valid:.6f}, max={bg_max:.6f}")

    # If max is suspiciously small (< 5 mm) but >0 -> likely meters, convert to mm
    if 0 < bg_max < 5.0:
        print("[background] Max < 5, likely meters. Converting to mm (×1000)")
        background = background * 1000.0
        bg_max2 = float(np.nanmax(background))
        print(f"[background] After conversion: max={bg_max2:.6f} mm")

    # Enforce GridPP-safe: float32, no NaN, 0 outside mask
    background = background.astype(np.float32)
    background[~griddata.mask] = 0.0
    background[np.isnan(background)] = 0.0

    # Plot-friendly version
    background_plot = background.copy()
    background_plot[~griddata.mask] = np.nan

    save_field_png(
        background_plot,
        f"Background (ERA5) {cfg.year}-{cfg.month:02d}",
        os.path.join(paths.out_dir, "01_background.png"),
    )

    # 4) OI (log-ratio)
    analysis, analysis_plot, d_grid, bg_at_stations, structure = log_ratio_oi(
        ogrid=griddata.ogrid,
        points=points,
        obs=obs,
        background_filled=background,
        mask=griddata.mask,
        eps=cfg.eps,
        L=cfg.L,
        pobs_d=cfg.pobs_d,
        max_points=cfg.max_points,
    )

    # 5) LOOCV / spatial CV
    pred_cv, err, mae, rmse = loocv_log_ratio(
        points=points,
        obs=obs,
        bg_at_stations=bg_at_stations,
        structure=structure,
        eps=cfg.eps,
        pobs_d=cfg.pobs_d,
        max_points=cfg.max_points,
        cv_radius_m=cfg.cv_radius_m,
    )

    # 6) Save outputs
    correction_plot = (analysis_plot - background_plot).astype(np.float32)

    save_field_png(
        analysis_plot,
        f"Analysis (log-ratio OI) {cfg.year}-{cfg.month:02d}",
        os.path.join(paths.out_dir, "02_analysis.png"),
    )
    save_field_png(
        correction_plot,
        "Correction (analysis - background)",
        os.path.join(paths.out_dir, "03_correction.png"),
    )

    report = []
    report.append(f"Run: {cfg.year}-{cfg.month:02d}")
    report.append(f"Stations used: {len(obs)}")
    report.append(f"ERA5 file: {era_nc}")
    report.append(f"OI params: eps={cfg.eps} L={cfg.L} pobs_d={cfg.pobs_d} max_points={cfg.max_points}")
    report.append(f"CV radius (m): {cfg.cv_radius_m}")
    report.append(f"LOOCV/CV: MAE={mae:.3f} RMSE={rmse:.3f}")
    report.append(f"Obs min/max: {float(obs.min()):.3f} / {float(obs.max()):.3f}")
    report.append(f"PredCV min/max: {float(pred_cv.min()):.3f} / {float(pred_cv.max()):.3f}")
    report.append(f"Background min/max: {float(background.min()):.3f} / {float(background.max()):.3f}")

    report_text = "\n".join(report)
    print(report_text)
    save_text(os.path.join(paths.out_dir, "report.txt"), report_text)


if __name__ == "__main__":
    main()
