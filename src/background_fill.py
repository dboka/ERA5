# src/background_fill.py
from __future__ import annotations
import numpy as np
import gridpp

def fill_background_holes_barnes_xy(
    xs2d: np.ndarray,
    ys2d: np.ndarray,
    mask: np.ndarray,
    bg: np.ndarray,
    min_bg: float,
    L_fill: float = 80000.0,
    max_points: int = 200,
) -> np.ndarray:
    """
    Fill bg holes using GridPP OI/Barnes in CARTESIAN coordinates,
    but compute only at hole locations (fast).

    Points constructor for Cartesian expects: Points(y, x, alt, laf, gridpp.Cartesian)
    """
    bg = bg.astype(np.float32).copy()
    m = mask.astype(bool)

    valid = m & np.isfinite(bg) & (bg > float(min_bg))
    hole  = m & (~valid)

    n_valid = int(valid.sum())
    n_hole  = int(hole.sum())

    if n_hole == 0:
        bg[~m] = 0.0
        bg[~np.isfinite(bg)] = 0.0
        return bg

    if n_valid < 10:
        raise RuntimeError(
            f"Par maz valid bg šūnu priekš Barnes fill: valid={n_valid} (check min_bg/units)."
        )

    # --- build Points for valid "observations" ---
    vx = xs2d[valid].ravel().astype(np.float32)  # x
    vy = ys2d[valid].ravel().astype(np.float32)  # y
    vval = bg[valid].ravel().astype(np.float32)

    velev = np.zeros(vval.shape, dtype=np.float32)
    vlaf  = np.zeros(vval.shape, dtype=np.float32)
    obs_pts = gridpp.Points(vy, vx, velev, vlaf, gridpp.Cartesian)

    # --- build Points for holes (output locations) ---
    hx = xs2d[hole].ravel().astype(np.float32)
    hy = ys2d[hole].ravel().astype(np.float32)

    helev = np.zeros(hx.shape, dtype=np.float32)
    hlaf  = np.zeros(hx.shape, dtype=np.float32)
    hole_pts = gridpp.Points(hy, hx, helev, hlaf, gridpp.Cartesian)

    structure = gridpp.BarnesStructure(float(L_fill))

    # OI at hole points only
    smooth_hole = gridpp.optimal_interpolation(
        hole_pts,
        np.zeros(n_hole, dtype=np.float32),     # background at holes
        obs_pts,
        vval,                                   # "obs values"
        np.zeros(n_valid, dtype=np.float32),    # obs background
        np.full(n_valid, 1.0, dtype=np.float32),
        structure,
        int(max_points),
    ).astype(np.float32)

    out = bg.copy()
    out[hole] = smooth_hole
    out[~m] = 0.0
    out[~np.isfinite(out)] = 0.0
    return out


def fill_background_holes_barnes_cartesian_only_holes(
    xs: np.ndarray,
    ys: np.ndarray,
    mask: np.ndarray,
    bg_mm: np.ndarray,
    min_bg_mm: float,
    L_fill_m: float = 80000.0,
    max_points: int = 200,
) -> np.ndarray:
    """
    Fill background holes using GridPP Barnes OI in Cartesian coordinates.
    
    Args:
        xs: 1D array of x coordinates (m)
        ys: 1D array of y coordinates (m)
        mask: 2D boolean mask of valid LV cells
        bg_mm: 2D background field in mm
        min_bg_mm: minimum valid background value (mm)
        L_fill_m: length scale for Barnes interpolation (m)
        max_points: max points for OI
    
    Returns:
        2D filled background (mm/month) with no NaN, all constraints enforced.
    """
    # Build 2D grids from 1D coordinates
    xs2d, ys2d = np.meshgrid(xs, ys)
    
    bg_out = bg_mm.astype(np.float32).copy()
    m = mask.astype(bool)
    
    # Identify valid cells and holes
    valid = m & np.isfinite(bg_out) & (bg_out > float(min_bg_mm))
    hole = m & (~valid)
    
    n_valid = int(valid.sum())
    n_hole = int(hole.sum())
    
    print(f"[fill] valid cells: {n_valid}, holes: {n_hole}")
    
    # If no holes, just enforce constraints
    if n_hole == 0:
        bg_out[~m] = 0.0
        bg_out[~np.isfinite(bg_out)] = 0.0
        return bg_out
    
    # Check we have enough valid points for interpolation
    if n_valid < 10:
        raise RuntimeError(
            f"Too few valid bg cells for Barnes fill: {n_valid} (check min_bg_mm and units)"
        )
    
    # Extract valid points as observations
    y_valid = ys2d[valid].ravel().astype(np.float32)
    x_valid = xs2d[valid].ravel().astype(np.float32)
    obs_vals = bg_mm[valid].ravel().astype(np.float32)
    
    elev_valid = np.zeros(n_valid, dtype=np.float32)
    laf_valid = np.zeros(n_valid, dtype=np.float32)
    obs_pts = gridpp.Points(y_valid, x_valid, elev_valid, laf_valid, gridpp.Cartesian)
    
    # Extract hole points as output locations
    y_hole = ys2d[hole].ravel().astype(np.float32)
    x_hole = xs2d[hole].ravel().astype(np.float32)
    
    elev_hole = np.zeros(n_hole, dtype=np.float32)
    laf_hole = np.zeros(n_hole, dtype=np.float32)
    hole_pts = gridpp.Points(y_hole, x_hole, elev_hole, laf_hole, gridpp.Cartesian)
    
    # Run optimal interpolation at hole locations only
    structure = gridpp.BarnesStructure(float(L_fill_m))
    
    smooth_hole = gridpp.optimal_interpolation(
        hole_pts,
        np.zeros(n_hole, dtype=np.float32),          # bg at holes
        obs_pts,
        obs_vals,                                     # obs values
        np.zeros(n_valid, dtype=np.float32),         # obs background
        np.ones(n_valid, dtype=np.float32),          # obs errors
        structure,
        int(max_points),
    ).astype(np.float32)
    
    # Constrain filled values: cap at max of valid cells to prevent local overshoot
    valid_max = float(np.max(obs_vals))
    valid_min = float(np.min(obs_vals[obs_vals > 0])) if (obs_vals > 0).any() else float(min_bg_mm)
    
    smooth_hole = np.clip(smooth_hole, valid_min, valid_max)
    
    # Fill holes with constrained Barnes values, then clamp to minimum
    bg_out[hole] = np.maximum(smooth_hole, float(min_bg_mm))
    
    # Enforce physical constraints: no negatives, enforce min everywhere in mask
    bg_out[mask] = np.maximum(bg_out[mask], 0.0)
    bg_out[~mask] = 0.0
    bg_out[~np.isfinite(bg_out)] = 0.0
    
    print(f"[fill] valid range: {valid_min:.3f}..{valid_max:.3f} mm")
    print(f"[fill] filled values clamped to: {valid_min:.3f}..{valid_max:.3f} mm")
    
    return bg_out
