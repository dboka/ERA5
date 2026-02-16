# src/oi_precip.py
from __future__ import annotations

import numpy as np
import gridpp


def log_ratio_oi(
    ogrid: gridpp.Grid,
    points: gridpp.Points,
    obs: np.ndarray,
    background_filled: np.ndarray,
    bg_at_stations: np.ndarray,   # <--- NEW (obligāts)
    mask: np.ndarray,
    eps: float,
    L: float,
    pobs_d: float,
    max_points: int,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),  # maigāks nekā min/max
):
    eps = np.float32(eps)

    bg_at_stations = np.asarray(bg_at_stations, dtype=np.float32)
    bg_at_stations = np.maximum(bg_at_stations, 0.0).astype(np.float32)

    obs = np.asarray(obs, dtype=np.float32)
    obs = np.maximum(obs, 0.0).astype(np.float32)

    # log-ratio anomalies at stations
    d_obs = (np.log(obs + eps) - np.log(bg_at_stations + eps)).astype(np.float32)

    structure = gridpp.BarnesStructure(float(L))
    d0_pts = np.zeros_like(d_obs, dtype=np.float32)
    pvec = np.full(d_obs.shape, float(pobs_d), dtype=np.float32)

    # OI of anomalies on grid
    d_grid = gridpp.optimal_interpolation(
        ogrid,
        np.zeros_like(background_filled, dtype=np.float32),  # first-guess anomaly = 0
        points,
        d_obs,
        d0_pts,
        pvec,
        structure,
        int(max_points),
    ).astype(np.float32)

    # soft clip (optional) – percentiles, not hard min/max
    lo_p, hi_p = clip_percentiles
    lo = float(np.nanpercentile(d_obs, lo_p))
    hi = float(np.nanpercentile(d_obs, hi_p))
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        d_grid = np.clip(d_grid, lo, hi).astype(np.float32)

    # analysis in mm
    analysis = (background_filled + eps) * np.exp(d_grid) - eps
    analysis = np.maximum(analysis, 0.0).astype(np.float32)

    analysis_plot = analysis.copy()
    analysis_plot[~mask] = np.nan

    return analysis, analysis_plot, d_grid, structure


def loocv_log_ratio(
    points: gridpp.Points,
    obs: np.ndarray,
    bg_at_stations: np.ndarray,
    structure: gridpp.StructureFunction,
    eps: float,
    pobs_d: float,
    max_points: int,
    cv_radius_m: float,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
):
    eps = np.float32(eps)

    bg_at_stations = np.asarray(bg_at_stations, dtype=np.float32)
    bg_at_stations = np.maximum(bg_at_stations, 0.0).astype(np.float32)

    obs = np.asarray(obs, dtype=np.float32)
    obs = np.maximum(obs, 0.0).astype(np.float32)

    d_obs = (np.log(obs + eps) - np.log(bg_at_stations + eps)).astype(np.float32)

    d0_pts = np.zeros_like(d_obs, dtype=np.float32)
    pvec = np.full(d_obs.shape, float(pobs_d), dtype=np.float32)

    structure_cv = gridpp.CrossValidation(structure, float(cv_radius_m))

    d_cv = gridpp.optimal_interpolation(
        points, d0_pts, points, d_obs, d0_pts, pvec, structure_cv, int(max_points)
    ).astype(np.float32)

    lo_p, hi_p = clip_percentiles
    lo = float(np.nanpercentile(d_obs, lo_p))
    hi = float(np.nanpercentile(d_obs, hi_p))
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        d_cv = np.clip(d_cv, lo, hi).astype(np.float32)

    pred = (bg_at_stations + eps) * np.exp(d_cv) - eps
    pred = np.maximum(pred, 0.0).astype(np.float32)

    err = pred - obs
    mae = float(np.nanmean(np.abs(err)))
    rmse = float(np.sqrt(np.nanmean(err**2)))
    return pred, err, mae, rmse
