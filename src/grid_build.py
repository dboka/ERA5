from __future__ import annotations
import numpy as np
import pandas as pd
import gridpp

class GridData:
    def __init__(self, xs, ys, X, Y, elev, elev_filled, mask, ogrid, lon2d, lat2d):
        self.xs = xs
        self.ys = ys
        self.X = X
        self.Y = Y
        self.elev = elev
        self.elev_filled = elev_filled
        self.mask = mask
        self.ogrid = ogrid
        self.lon2d = lon2d
        self.lat2d = lat2d

def build_lv_grid_cartesian(g: pd.DataFrame, max_cells: int = 2_000_000) -> GridData:
    for c in ["x", "y", "h5", "lon", "lat"]:
        if c not in g.columns:
            raise ValueError(f"LV grid missing column '{c}'. Available: {g.columns.tolist()}")

    g = g.copy()
    for c in ["x", "y", "h5", "lon", "lat"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["x", "y"]).copy()

    # --- IMPORTANT: snap to 1km index to avoid float pivot mismatch ---
    dx = 1000.0
    dy = 1000.0
    xmin = float(g["x"].min())
    ymin = float(g["y"].min())

    g["xi"] = np.rint((g["x"].to_numpy(np.float64) - xmin) / dx).astype(np.int32)
    g["yi"] = np.rint((g["y"].to_numpy(np.float64) - ymin) / dy).astype(np.int32)

    nx = int(g["xi"].max()) + 1
    ny = int(g["yi"].max()) + 1
    ncell = nx * ny

    print(f"[grid] rows={len(g)} nx={nx} ny={ny} nx*ny={ncell}")

    if ncell > max_cells:
        raise RuntimeError(f"[grid] STOP: nx*ny={ncell} too large (safety max={max_cells}).")

    # rebuild exact x/y coordinates from integer indices (stable!)
    xs = (xmin + np.arange(nx, dtype=np.float32) * dx).astype(np.float32)
    ys = (ymin + np.arange(ny, dtype=np.float32) * dy).astype(np.float32)

    X, Y = np.meshgrid(xs, ys)  # (ny,nx)

    # fill 2D arrays via indexing, not pivot
    elev = np.full((ny, nx), np.nan, dtype=np.float32)
    lon2d = np.full((ny, nx), np.nan, dtype=np.float32)
    lat2d = np.full((ny, nx), np.nan, dtype=np.float32)

    xi = g["xi"].to_numpy(np.int32)
    yi = g["yi"].to_numpy(np.int32)

    elev[yi, xi] = g["h5"].to_numpy(np.float32)
    lon2d[yi, xi] = g["lon"].to_numpy(np.float32)
    lat2d[yi, xi] = g["lat"].to_numpy(np.float32)

    mask = ~np.isnan(elev)
    missing = int((~mask).sum())
    print(f"[grid] missing cells in bounding rectangle: {missing} / {elev.size} (coverage={mask.mean():.3f})")

    elev_filled = np.nan_to_num(elev, nan=0.0).astype(np.float32)
    laf = mask.astype(np.float32)  # 1 inside LV, 0 outside

    ogrid = gridpp.Grid(Y.astype(np.float32), X.astype(np.float32), elev_filled, laf, gridpp.Cartesian)

    # If some masked cells have missing lon/lat, drop them from mask
    bad_geo = int((mask & (np.isnan(lon2d) | np.isnan(lat2d))).sum())
    if bad_geo > 0:
        print(f"[grid] WARNING: {bad_geo} masked cells have missing lon/lat. Dropping from mask.")
        mask = mask & ~np.isnan(lon2d) & ~np.isnan(lat2d)

    return GridData(xs, ys, X.astype(np.float32), Y.astype(np.float32),
                    elev, elev_filled, mask, ogrid, lon2d, lat2d)

def build_points_cartesian(st_sub: pd.DataFrame) -> tuple[gridpp.Points, np.ndarray, np.ndarray, np.ndarray]:
    need = ["x", "y", "elevation", "month_sum"]
    for c in need:
        if c not in st_sub.columns:
            raise ValueError(f"Stations subset missing '{c}'. Available: {st_sub.columns.tolist()}")

    st_sub = st_sub.dropna(subset=need).copy()

    px = st_sub["x"].to_numpy(np.float32)
    py = st_sub["y"].to_numpy(np.float32)
    pelev = st_sub["elevation"].to_numpy(np.float32)
    obs = st_sub["month_sum"].to_numpy(np.float32)

    # Points(y, x, elev, laf, Cartesian)
    points = gridpp.Points(py, px, pelev, np.ones_like(pelev, np.float32), gridpp.Cartesian)
    return points, obs, px, py
