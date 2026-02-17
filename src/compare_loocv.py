#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

from pyproj import Transformer
from pykrige.ok import OrdinaryKriging

# ---- import your GridPP LOOCV pieces ----
from config import RunConfig, Paths
from io_data import read_grid_csv, read_stations_csv
from grid_build import build_lv_grid_cartesian, build_points_cartesian
from background_era5_total import load_era5_monthly_total_mm_to_lvgrid
from interp_xy import bilinear_sample_regular_xy
from oi_precip import log_ratio_oi, loocv_log_ratio
from plots import ensure_dir, save_text


def parse_args():
    p = argparse.ArgumentParser("Compare LOOCV: GridPP log-ratio OI vs UK (Python drift+OK)")
    p.add_argument("--grid_csv", required=True, help="LV grid CSV (has x,y,h5,cont_pr etc)")
    p.add_argument("--stations_monthly_csv", required=True, help="Monthly stations CSV (WITH_COORDS)")
    p.add_argument("--stations_meta_csv", required=True, help="Station meta CSV (has cont_pr + elevation)")
    p.add_argument("--era_nc", required=True, help="ERA5 monthly total precipitation NetCDF")
    p.add_argument("--out", default="out_compare")

    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)

    # columns in meta
    p.add_argument("--meta_id_col", default="gh_id")
    p.add_argument("--meta_cont_col", default="cont_pr")      # station cont_pr
    p.add_argument("--meta_h_col", default="elevation")       # station elevation (try this first)

    # if meta uses other names, you can pass them
    p.add_argument("--meta_h_col_alt", default="ELEVATION")   # fallback

    # UK params (to mimic your R defaults)
    p.add_argument("--uk_range_m", type=float, default=41500.0)
    p.add_argument("--uk_nugget", type=float, default=0.0)

    # GridPP params (same as your main defaults, override if needed)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--L", type=float, default=30000.0)
    p.add_argument("--pobs_d", type=float, default=2.0)
    p.add_argument("--max_points", type=int, default=80)
    p.add_argument("--cv_radius", type=float, default=60000.0)
    p.add_argument("--threads", type=int, default=8)

    return p.parse_args()


def _design_matrix(px: np.ndarray, py: np.ndarray, cont: np.ndarray, h: np.ndarray) -> np.ndarray:
    x = px.astype(np.float64)
    y = py.astype(np.float64)
    c = cont.astype(np.float64)
    z = h.astype(np.float64)
    # [1, x, y, x^2, y^2, x*y, cont, h]
    return np.column_stack([
        np.ones_like(x),
        x, y,
        x * x, y * y,
        x * y,
        c, z
    ])


def _ols_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    # robust-ish OLS via lstsq
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return X_pred @ beta


def uk_loocv(px: np.ndarray, py: np.ndarray, obs: np.ndarray, cont: np.ndarray, h: np.ndarray,
            rng: float, nugget: float) -> np.ndarray:
    """
    UK approximation: fit drift by OLS, krige residuals by OK (Exp variogram), LOOCV.
    Returns pred array length N.
    """
    n = len(obs)
    pred = np.full(n, np.nan, dtype=np.float64)

    # prebuild full design matrix
    X = _design_matrix(px, py, cont, h)

    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False

        Xtr = X[m]
        ytr = obs[m].astype(np.float64)

        # 1) drift
        drift_tr = _ols_fit_predict(Xtr, ytr, Xtr)
        resid_tr = ytr - drift_tr

        # 2) OK on residuals
        # sill ~ var(resid) (simple)
        sill = float(np.nanvar(resid_tr))
        if not np.isfinite(sill) or sill <= 0:
            # fallback: no spatial correction
            drift_i = float(_ols_fit_predict(Xtr, ytr, X[i:i+1])[0])
            pred[i] = drift_i
            continue

        ok = OrdinaryKriging(
            px[m], py[m], resid_tr,
            variogram_model="exponential",
            variogram_parameters={"sill": sill, "range": float(rng), "nugget": float(nugget)},
            enable_plotting=False,
            coordinates_type="euclidean",
        )

        # predict residual at left-out point
        r_i, _ = ok.execute("points", np.array([px[i]]), np.array([py[i]]))
        r_i = float(np.asarray(r_i).ravel()[0])

        drift_i = float(_ols_fit_predict(Xtr, ytr, X[i:i+1])[0])
        pred_i = drift_i + r_i

        # mimic R “clamp to obs min/max”
        omin = float(np.nanmin(ytr))
        omax = float(np.nanmax(ytr))
        pred_i = max(omin, min(omax, pred_i))

        pred[i] = pred_i

    return pred.astype(np.float64)


def metrics(err: np.ndarray) -> dict:
    e = err[np.isfinite(err)]
    if e.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan}
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
    }


def main():
    a = parse_args()
    ensure_dir(a.out)

    # ---- load monthly obs ----
    df = read_stations_csv(a.stations_monthly_csv)
    sub = df[(df["Gads"] == a.year) & (df["Menesis"] == a.month)].copy()
    if len(sub) == 0:
        raise RuntimeError(f"No station rows for {a.year}-{a.month:02d}")

    if "gh_id" in sub.columns:
        sub = sub.sort_values("gh_id")

    # ---- load station meta (cont_pr + elevation) ----
    meta = pd.read_csv(a.stations_meta_csv, sep=";", decimal=".", encoding="utf-8", low_memory=False)
    meta.columns = [c.strip().replace("\ufeff", "") for c in meta.columns]

    if a.meta_id_col not in meta.columns:
        raise ValueError(f"meta_id_col not found in meta csv: {a.meta_id_col}")

    # choose elevation column
    hcol = a.meta_h_col if a.meta_h_col in meta.columns else None
    if hcol is None and a.meta_h_col_alt in meta.columns:
        hcol = a.meta_h_col_alt
    if hcol is None:
        raise ValueError(f"Could not find elevation column in meta. Tried: {a.meta_h_col}, {a.meta_h_col_alt}")

    if a.meta_cont_col not in meta.columns:
        raise ValueError(f"meta_cont_col not found in meta csv: {a.meta_cont_col}")

    meta_small = meta[[a.meta_id_col, a.meta_cont_col, hcol]].copy()
    meta_small = meta_small.rename(columns={a.meta_id_col: "gh_id", a.meta_cont_col: "cont_pr", hcol: "elev"})

    # merge (keep only stations present in monthly)
    sub = sub.merge(meta_small, how="left", on="gh_id")

    # drop if no cont/elev
    sub = sub.dropna(subset=["cont_pr", "elev"]).copy()
    if len(sub) < 3:
        raise RuntimeError("Too few stations after merging meta (need cont_pr + elev).")

    # ---- build grid + background (for GridPP LOOCV) ----
    g = read_grid_csv(a.grid_csv)
    griddata = build_lv_grid_cartesian(g)
    griddata.xs2d, griddata.ys2d = np.meshgrid(griddata.xs, griddata.ys)

    # build GridPP points/obs in your existing way
    points, obs, px, py = build_points_cartesian(sub)
    obs = np.asarray(obs, dtype=np.float32)

    # background ERA5 to LV grid
    background = load_era5_monthly_total_mm_to_lvgrid(
        a.era_nc, griddata.lon2d, griddata.lat2d, griddata.mask
    ).astype(np.float32)
    bg_max = float(np.nanmax(background))
    if 0 < bg_max < 5.0:
        background *= 1000.0

    bg_for_oi = background.copy()
    inside_vals = bg_for_oi[griddata.mask]
    bg_mean = float(np.nanmean(inside_vals)) if inside_vals.size else 0.0
    bg_for_oi[~griddata.mask] = bg_mean
    bg_for_oi[~np.isfinite(bg_for_oi)] = bg_mean

    fillv = float(np.nanmean(bg_for_oi[griddata.mask])) if np.any(griddata.mask) else bg_mean
    bg_at_stations = bilinear_sample_regular_xy(
        bg_for_oi, griddata.xs, griddata.ys, px, py, fill_value=fillv
    ).astype(np.float32)
    bg_at_stations = np.maximum(bg_at_stations, 0.0).astype(np.float32)

    # GridPP OI structure
    analysis, analysis_plot, d_grid, structure = log_ratio_oi(
        ogrid=griddata.ogrid,
        points=points,
        obs=obs,
        background_filled=bg_for_oi,
        bg_at_stations=bg_at_stations,
        mask=griddata.mask,
        eps=a.eps,
        L=a.L,
        pobs_d=a.pobs_d,
        max_points=a.max_points,
    )

    pred_gridpp, err_g, mae_g, rmse_g = loocv_log_ratio(
        points=points,
        obs=obs,
        bg_at_stations=bg_at_stations,
        structure=structure,
        eps=a.eps,
        pobs_d=a.pobs_d,
        max_points=a.max_points,
        cv_radius_m=a.cv_radius,
    )

    # ---- UK LOOCV (Python drift+OK) ----
    cont = sub["cont_pr"].to_numpy(dtype=np.float64)
    elev = sub["elev"].to_numpy(dtype=np.float64)
    pred_uk = uk_loocv(px, py, obs.astype(np.float64), cont, elev, rng=a.uk_range_m, nugget=a.uk_nugget)

    err_uk = pred_uk - obs.astype(np.float64)
    err_gridpp = pred_gridpp.astype(np.float64) - obs.astype(np.float64)

    m_g = metrics(err_gridpp)
    m_u = metrics(err_uk)

    out_df = pd.DataFrame({
        "gh_id": sub["gh_id"].astype(str).to_numpy(),
        "obs": obs.astype(np.float64),
        "pred_gridpp": pred_gridpp.astype(np.float64),
        "err_gridpp": err_gridpp,
        "pred_uk": pred_uk,
        "err_uk": err_uk,
        "cont_pr": cont,
        "elev": elev,
        "px": px.astype(np.float64),
        "py": py.astype(np.float64),
    })

    out_csv = os.path.join(a.out, "loocv_compare.csv")
    out_df.to_csv(out_csv, index=False)

    report = []
    report.append(f"Run: {a.year}-{a.month:02d}")
    report.append(f"Stations used: {len(out_df)}")
    report.append("")
    report.append("GridPP log-ratio OI LOOCV:")
    report.append(f"  MAE={m_g['mae']:.3f} RMSE={m_g['rmse']:.3f} BIAS={m_g['bias']:.3f}")
    report.append("")
    report.append("UK (drift+OK residuals) LOOCV:")
    report.append(f"  MAE={m_u['mae']:.3f} RMSE={m_u['rmse']:.3f} BIAS={m_u['bias']:.3f}")
    report.append("")
    report.append(f"UK params: range_m={a.uk_range_m} nugget={a.uk_nugget}")
    report.append(f"Drift: 1, x, y, x^2, y^2, x*y, cont_pr, elev")

    report_text = "\n".join(report)
    print(report_text)
    save_text(os.path.join(a.out, "loocv_summary.txt"), report_text)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
