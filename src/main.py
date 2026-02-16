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
from export import write_geotiff_lks92
from background_era5_total import load_era5_monthly_total_mm_to_lvgrid
from interp_xy import bilinear_sample_regular_xy


def parse_args(defaults: RunConfig) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LV precip interpolation with GridPP (log-ratio OI + LOOCV)")

    p.add_argument("--grid", required=True, help="LV grid CSV")
    p.add_argument("--stations", required=True, help="Stations monthly CSV")
    p.add_argument("--era_nc", required=True, help="ERA5 monthly total precipitation NetCDF (mm or m)")
    p.add_argument("--out", default="out")

    p.add_argument("--year", type=int, default=defaults.year)
    p.add_argument("--month", type=int, default=defaults.month)

    p.add_argument("--eps", type=float, default=defaults.eps)
    p.add_argument("--L", type=float, default=defaults.L)
    p.add_argument("--pobs_d", type=float, default=defaults.pobs_d)
    p.add_argument("--max_points", type=int, default=defaults.max_points)

    p.add_argument("--cv_radius", type=float, default=defaults.cv_radius_m)
    p.add_argument("--threads", type=int, default=defaults.omp_threads)

    return p.parse_args()


def main():
    defaults = RunConfig()
    args = parse_args(defaults)

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
        max_cells=defaults.max_cells,
    )

    ensure_dir(paths.out_dir)
    gridpp.set_omp_threads(cfg.omp_threads)

    print(
        f"[RUN] year={cfg.year} month={cfg.month} eps={cfg.eps} L={cfg.L} "
        f"pobs_d={cfg.pobs_d} max_points={cfg.max_points} cv_radius={cfg.cv_radius_m} threads={cfg.omp_threads}"
    )

    # 1) grid
    g = read_grid_csv(paths.grid_csv)
    griddata = build_lv_grid_cartesian(g)
    griddata.xs2d, griddata.ys2d = np.meshgrid(griddata.xs, griddata.ys)

    # 2) stations
    df = read_stations_csv(paths.stations_csv)
    sub = df[(df["Gads"] == cfg.year) & (df["Menesis"] == cfg.month)].copy()
    if len(sub) == 0:
        raise RuntimeError(f"No station rows found for {cfg.year}-{cfg.month:02d}")

    if "gh_id" in sub.columns:
        sub = sub.sort_values("gh_id")

    points, obs, px, py = build_points_cartesian(sub)

    # Drop stations outside grid bbox
    xmin, xmax = float(griddata.xs.min()), float(griddata.xs.max())
    ymin, ymax = float(griddata.ys.min()), float(griddata.ys.max())
    inside = (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
    if (~inside).any():
        sub = sub.loc[inside].copy()
        points, obs, px, py = build_points_cartesian(sub)

    if len(obs) < 3:
        raise RuntimeError("Too few stations after bbox filter.")

    obs = np.asarray(obs, dtype=np.float32)
    obs_max = float(np.nanmax(obs))
    vmin_scale = 0.0
    vmax_scale = obs_max + 10.0

    # 3) background to LV grid
    if not os.path.exists(args.era_nc):
        raise FileNotFoundError(f"ERA5 background file not found: {args.era_nc}")

    background = load_era5_monthly_total_mm_to_lvgrid(
        args.era_nc, griddata.lon2d, griddata.lat2d, griddata.mask
    ).astype(np.float32)

    # unit heuristic: meters -> mm
    bg_max = float(np.nanmax(background))
    if 0 < bg_max < 5.0:
        print("[background] looks like meters -> converting to mm (×1000)")
        background *= 1000.0

    # plot background (SAME SCALE AS ANALYSIS)
    background_plot = background.copy()
    background_plot[~griddata.mask] = np.nan
    save_field_png(
        background_plot,
        f"Background (ERA5) {cfg.year}-{cfg.month:02d}",
        os.path.join(paths.out_dir, "01_background.png"),
        mask=None,
        scale="obs",
        obs_max=obs_max,
        pad_mm=10.0,
        interpolation="nearest",
    )

    # OI needs background everywhere -> fill outside mask with interior mean
    bg_for_oi = background.copy()
    inside_vals = bg_for_oi[griddata.mask]
    bg_mean = float(np.nanmean(inside_vals)) if inside_vals.size else 0.0
    bg_for_oi[~griddata.mask] = bg_mean
    bg_for_oi[~np.isfinite(bg_for_oi)] = bg_mean

    # background at stations via SAFE bilinear in x/y
    fillv = float(np.nanmean(bg_for_oi[griddata.mask])) if np.any(griddata.mask) else bg_mean
    bg_at_stations = bilinear_sample_regular_xy(
        bg_for_oi, griddata.xs, griddata.ys, px, py, fill_value=fillv
    ).astype(np.float32)
    bg_at_stations = np.maximum(bg_at_stations, 0.0).astype(np.float32)

    # 4) OI (log-ratio)
    analysis, analysis_plot, d_grid, structure = log_ratio_oi(
        ogrid=griddata.ogrid,
        points=points,
        obs=obs,
        background_filled=bg_for_oi,
        bg_at_stations=bg_at_stations,   # <-- consistent now
        mask=griddata.mask,
        eps=cfg.eps,
        L=cfg.L,
        pobs_d=cfg.pobs_d,
        max_points=cfg.max_points,
    )

    # 5) CV
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

    # 6) plots
    analysis_plot2 = analysis.copy()
    analysis_plot2[~griddata.mask] = np.nan

    background_plot2 = background.copy()
    background_plot2[~griddata.mask] = np.nan

    correction_plot = (analysis_plot2 - background_plot2).astype(np.float32)

    save_field_png(
        analysis_plot2,
        f"Analysis (log-ratio OI) {cfg.year}-{cfg.month:02d}",
        os.path.join(paths.out_dir, "02_analysis.png"),
        mask=None,
        scale="obs",
        obs_max=obs_max,
        pad_mm=10.0,
        interpolation="nearest",
    )

    # correction: symmetric around 0
    finite_corr = correction_plot[np.isfinite(correction_plot)]
    corr_abs = float(np.nanmax(np.abs(finite_corr))) if finite_corr.size else 1.0

    save_field_png(
        correction_plot,
        "Correction (analysis - background)",
        os.path.join(paths.out_dir, "03_correction.png"),
        mask=None,
        scale="obs",
        obs_max=obs_max,
        interpolation="nearest",
        
    )

    # GeoTIFF export
    analysis_tif = analysis.copy()
    analysis_tif[~griddata.mask] = np.nan

    write_geotiff_lks92(
        os.path.join(paths.out_dir, "02_analysis.tif"),
        analysis_tif,
        griddata.xs,
        griddata.ys,
    )

    report = []
    report.append(f"Run: {cfg.year}-{cfg.month:02d}")
    report.append(f"Stations used: {len(obs)}")
    report.append(f"ERA5 file: {args.era_nc}")
    report.append(f"OI params: eps={cfg.eps} L={cfg.L} pobs_d={cfg.pobs_d} max_points={cfg.max_points}")
    report.append(f"CV radius (m): {cfg.cv_radius_m}")
    report.append(f"CV: MAE={mae:.3f} RMSE={rmse:.3f}")
    report.append(f"Obs min/max: {float(np.nanmin(obs)):.3f} / {float(np.nanmax(obs)):.3f}")
    report.append(f"Plot scale: vmin={vmin_scale:.1f} vmax={vmax_scale:.1f} (0..obs_max+10)")
    report.append(f"PredCV min/max: {float(np.nanmin(pred_cv)):.3f} / {float(np.nanmax(pred_cv)):.3f}")
    report.append(f"Background min/max: {float(np.nanmin(background_plot2)):.3f} / {float(np.nanmax(background_plot2)):.3f}")

    report_text = "\n".join(report)
    print(report_text)
    save_text(os.path.join(paths.out_dir, "report.txt"), report_text)


if __name__ == "__main__":
    main()
