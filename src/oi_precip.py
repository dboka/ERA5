from __future__ import annotations
import numpy as np
import gridpp

def make_background_constant(shape: tuple[int, int], value: float, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      background_filled: 2D float32 without NaN (safe for gridpp)
      background_plot:   2D float32 with NaN outside mask (nice for plots)
    """
    bg = np.full(shape, float(value), dtype=np.float32)
    bg_plot = bg.copy()
    bg_plot[~mask] = np.nan

    bg_filled = bg.copy()
    bg_filled[~mask] = 0.0
    return bg_filled, bg_plot

def log_ratio_oi(
    ogrid: gridpp.Grid,
    points: gridpp.Points,
    obs: np.ndarray,
    background_filled: np.ndarray,
    mask: np.ndarray,
    eps: float,
    L: float,
    pobs_d: float,
    max_points: int,
):
    eps = np.float32(eps)

    bg_at_stations = gridpp.nearest(ogrid, points, background_filled).astype(np.float32)

    # Guard: background at stations must be >=0
    bg_at_stations = np.maximum(bg_at_stations, 0.0).astype(np.float32)

    print(f"[OI] bg_at_stations range: {float(bg_at_stations.min()):.3f}...{float(bg_at_stations.max()):.3f} mm")
    print(f"[OI] obs range: {float(obs.min()):.3f}...{float(obs.max()):.3f} mm")
    print(f"[OI] eps: {float(eps):.3f} mm")

    d_obs = (np.log(obs + eps) - np.log(bg_at_stations + eps)).astype(np.float32)
    print(f"[OI] d_obs range: {float(d_obs.min()):.6f}...{float(d_obs.max()):.6f}")

    structure = gridpp.BarnesStructure(float(L))
    d0_pts = np.zeros_like(d_obs, dtype=np.float32)
    pvec = np.full(d_obs.shape, float(pobs_d), dtype=np.float32)

    d_grid = gridpp.optimal_interpolation(
        ogrid,
        np.zeros_like(background_filled, dtype=np.float32),  # d-background = 0
        points,
        d_obs,
        d0_pts,
        pvec,
        structure,
        int(max_points),
    ).astype(np.float32)

    # Safety: clip ONLY d-field to observed station range
    d_min, d_max = float(d_obs.min()), float(d_obs.max())
    d_grid = np.clip(d_grid, d_min, d_max).astype(np.float32)
    print(f"[OI] d_grid clipped to: {float(d_grid.min()):.6f}...{float(d_grid.max()):.6f}")

    analysis = (background_filled + eps) * np.exp(d_grid) - eps
    analysis = np.maximum(analysis, 0.0).astype(np.float32)

    analysis_plot = analysis.copy()
    analysis_plot[~mask] = np.nan

    return analysis, analysis_plot, d_grid, bg_at_stations, structure

def loocv_log_ratio(
    points: gridpp.Points,
    obs: np.ndarray,
    bg_at_stations: np.ndarray,
    structure: gridpp.StructureFunction,
    eps: float,
    pobs_d: float,
    max_points: int,
    cv_radius_m: float,
):
    eps = np.float32(eps)
    bg_at_stations = np.maximum(bg_at_stations, 0.0).astype(np.float32)

    d_obs = (np.log(obs + eps) - np.log(bg_at_stations + eps)).astype(np.float32)
    print(f"[LOOCV] d_obs range: {float(d_obs.min()):.6f}...{float(d_obs.max()):.6f}")

    d0_pts = np.zeros_like(d_obs, dtype=np.float32)
    pvec = np.full(d_obs.shape, float(pobs_d), dtype=np.float32)

    structure_cv = gridpp.CrossValidation(structure, float(cv_radius_m))

    d_cv = gridpp.optimal_interpolation(
        points,
        d0_pts,      # background at points in d-space
        points,
        d_obs,
        d0_pts,
        pvec,
        structure_cv,
        int(max_points),
    ).astype(np.float32)

    # Safety: clip ONLY d
    d_min, d_max = float(d_obs.min()), float(d_obs.max())
    d_cv = np.clip(d_cv, d_min, d_max).astype(np.float32)
    print(f"[LOOCV] d_cv clipped to: {float(d_cv.min()):.6f}...{float(d_cv.max()):.6f}")

    pred = (bg_at_stations + eps) * np.exp(d_cv) - eps
    pred = np.maximum(pred, 0.0).astype(np.float32)

    err = pred - obs
    mae = float(np.nanmean(np.abs(err)))
    rmse = float(np.sqrt(np.nanmean(err**2)))
    return pred, err, mae, rmse

