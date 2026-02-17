# src/makemap.py
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import rasterio
from rasterio import features

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe

# Logo helper
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Optional borders + clipping
try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from pyproj import CRS, Transformer
except Exception:
    CRS = None
    Transformer = None


def parse_args():
    p = argparse.ArgumentParser("Make LVGMC-style production map (nice scale + 5mm quantization)")

    p.add_argument("--analysis_tif", required=True, help="02_analysis.tif (EPSG:3059 recommended)")
    p.add_argument("--grid_csv", required=True, help="1x1 grid centers CSV with x/y columns (LKS92).")
    p.add_argument("--grid_sep", default=",", help="Grid CSV separator (default ,)")
    p.add_argument("--grid_decimal", default=".", help="Grid CSV decimal mark (default .)")
    p.add_argument("--grid_x_col", default="x", help="Grid x column name (default x)")
    p.add_argument("--grid_y_col", default="y", help="Grid y column name (default y)")
    p.add_argument("--grid_crs", default="EPSG:3059", help="CRS of grid x/y (default EPSG:3059)")

    p.add_argument("--stations_csv", required=True, help="Stations CSV (monthly)")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--out_png", required=True)

    # vectors
    p.add_argument("--lv_border", default=None, help="Latvia border vector (gpkg/shp). Optional but recommended.")
    p.add_argument("--robeza_shp", default=None, help="Border shapefile. If set, used as lv_border.")
    p.add_argument("--lv_muni", default=None, help="Latvia municipalities/admin borders (gpkg/shp). Optional.")
    p.add_argument("--neighbors", default=None, help="Neighbor countries vector (gpkg/shp). Optional.")
    

    # clip raster to LV border polygon (makes coastline crisp)
    p.add_argument("--clip_to_lv", action="store_true", help="Clip raster to lv_border polygon")

    # Stations CSV columns
    p.add_argument("--csv_sep", default=";", help="Stations CSV separator (default ;)")
    p.add_argument("--csv_decimal", default=".", help="Stations CSV decimal mark (default .)")
    p.add_argument("--station_value_col", default="month_sum", help="Value column (default month_sum)")
    p.add_argument("--station_lon_col", default="lon")
    p.add_argument("--station_lat_col", default="lat")
    p.add_argument("--station_name_col", default=None, help="Optional name column in monthly CSV. If missing -> use meta file, else gh_id.")
    p.add_argument("--station_id_col", default="gh_id", help="Station id column (default gh_id)")
    p.add_argument("--stations_crs", default="EPSG:4326", help="CRS of station lon/lat (default EPSG:4326)")

    # Station meta (your staciju_dati_norma3.csv)
    p.add_argument(
        "--stations_meta_csv",
        default="/home/denissbokadenissboka/projects/gridpp_lab/staciju_dati_norma3.csv",
        help="Stations meta CSV containing station id -> station name mapping.",
    )
    p.add_argument("--meta_sep", default=";", help="Meta CSV separator (default ;)")
    p.add_argument("--meta_decimal", default=".", help="Meta CSV decimal mark (default .)")
    p.add_argument("--meta_id_col", default=None, help="Meta CSV id column (optional; auto-detect if None).")
    p.add_argument("--meta_name_col", default=None, help="Meta CSV name column (optional; auto-detect if None).")

    # styling
    p.add_argument("--lang", default="LV", choices=["LV", "EN"])
    p.add_argument("--bg_color", default="#EFF3F6")
    p.add_argument("--neighbors_fill", default="#E6EAEE")
    p.add_argument("--neighbors_edge", default="#C9CED3")
    p.add_argument("--border_col", default="#48525B")
    p.add_argument("--muni_col", default="#7C8893")

    # display-only smoothing (optional)
    p.add_argument("--smooth_sigma", type=float, default=0.0, help="Display-only NaN-safe smoothing sigma (0=off)")

    # labels
    p.add_argument("--label_stations", action="store_true")
    p.add_argument("--max_labels", type=int, default=60)
    p.add_argument("--label_top_n", type=int, default=9999)
    p.add_argument("--label_dx_frac", type=float, default=0.008)
    p.add_argument("--label_dy_frac", type=float, default=0.008)

    # requested behavior
    p.add_argument("--quant_step", type=float, default=5.0, help="Quantization step in mm (default 5mm)")
    p.add_argument("--tick_step", type=float, default=25.0, help="Colorbar ticks step in mm (default 25mm)")
    p.add_argument("--vmax_round", type=float, default=25.0, help="Round vmax up to nearest N (default 25mm)")

    # logo
    p.add_argument(
        "--logo_png",
        default="/home/denissbokadenissboka/projects/gridpp_lab/images/LVGMC_1300-AM.png",
        help="Logo PNG path to draw at very left corner (figure).",
    )
    p.add_argument("--logo_zoom", type=float, default=0.12, help="Logo zoom (default 0.12).")
    p.add_argument("--logo_pad", type=float, default=0.01, help="Padding from figure corner (0-0.05 typical).")

    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


LV_MONTHS = {
    1: "janvāris", 2: "februāris", 3: "marts", 4: "aprīlis", 5: "maijs", 6: "jūnijs",
    7: "jūlijs", 8: "augusts", 9: "septembris", 10: "oktobris", 11: "novembris", 12: "decembris"
}
EN_MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}


def lvgmc_precip_cmap() -> LinearSegmentedColormap:
    colors = [
        "#F4F6C6",
        "#D8EEB5",
        "#A8DEB8",
        "#6CCFD0",
        "#2F95C8",
        "#1F5FA8",
        "#1A2C7A",
    ]
    return LinearSegmentedColormap.from_list("lvgmc_precip", colors, N=256)


def comma_formatter(lang: str, ndp: int = 0):
    def _fmt(x, _pos=None):
        s = f"{x:.{ndp}f}"
        if lang == "LV":
            s = s.replace(".", ",")
        return s
    return FuncFormatter(_fmt)


def transform_points_to_crs(xs: np.ndarray, ys: np.ndarray, src_crs: str, dst_crs) -> Tuple[np.ndarray, np.ndarray]:
    if dst_crs is None:
        return xs, ys
    if CRS is None or Transformer is None:
        raise ImportError("pyproj is required for CRS transforms: pip install pyproj")
    st = CRS.from_user_input(src_crs)
    rr = CRS.from_user_input(dst_crs)
    if st == rr:
        return xs, ys
    tr = Transformer.from_crs(st, rr, always_xy=True)
    x2, y2 = tr.transform(xs, ys)
    return np.asarray(x2, dtype=float), np.asarray(y2, dtype=float)


def gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)


def gaussian_smooth_nan(a: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-safe smoothing: smooth(values*mask)/smooth(mask)."""
    if sigma <= 0:
        return a

    a = a.astype(np.float32, copy=False)
    valid = np.isfinite(a).astype(np.float32)
    a0 = np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    k = gaussian_kernel1d(sigma)

    def conv1d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
        pad = len(kernel) // 2
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (pad, pad)
        arrp = np.pad(arr, pad_width, mode="reflect")

        out = np.zeros_like(arr, dtype=np.float32)
        for i, w in enumerate(kernel):
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(i, i + arr.shape[axis])
            out += w * arrp[tuple(sl)]
        return out

    num = conv1d(conv1d(a0, k, axis=1), k, axis=0)
    den = conv1d(conv1d(valid, k, axis=1), k, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan).astype(np.float32)
    return out


def edges_from_centers_midpoint(v: np.ndarray) -> np.ndarray:
    """Robust edges for (almost) regular grid using midpoints."""
    v = np.asarray(v, dtype=float)
    if v.size < 2:
        raise ValueError("Need at least 2 centers.")
    v = np.sort(v)

    mids = (v[:-1] + v[1:]) * 0.5
    edges = np.empty(v.size + 1, dtype=float)
    edges[1:-1] = mids
    edges[0] = v[0] - (mids[0] - v[0])
    edges[-1] = v[-1] + (v[-1] - mids[-1])
    return edges


def sample_tif_at_centers_vectorized(tif_path: str, xs: np.ndarray, ys: np.ndarray, grid_crs: str) -> Tuple[np.ndarray, object]:
    """Fast sampling: compute raster indices for all centers using inverse affine transform (nearest)."""
    xs = np.sort(xs.astype(float))
    ys = np.sort(ys.astype(float))

    with rasterio.open(tif_path) as src:
        r_crs = src.crs
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        X, Y = np.meshgrid(xs, ys)
        xq = X.ravel()
        yq = Y.ravel()
        xq, yq = transform_points_to_crs(xq, yq, grid_crs, r_crs)

        inv = ~src.transform
        cols, rows = inv * (xq, yq)
        rows = np.floor(rows).astype(np.int32)
        cols = np.floor(cols).astype(np.int32)

        Z = np.full(rows.shape, np.nan, dtype=np.float32)
        ok = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
        if np.any(ok):
            Z[ok] = arr[rows[ok], cols[ok]]
        Z = Z.reshape((ys.size, xs.size))
        return Z, r_crs


def clip_grid_to_polygon(Z: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, polygon_gdf) -> np.ndarray:
    """Clip raster-like grid (cell edges) to polygon: outside -> NaN."""
    if polygon_gdf is None or len(polygon_gdf) == 0:
        return Z

    dx = float(np.median(np.diff(x_edges)))
    dy = float(np.median(np.diff(y_edges)))
    west = float(x_edges[0])
    north = float(y_edges[-1])
    transform = rasterio.transform.from_origin(west, north, dx, dy)

    Ztop = np.flipud(Z)

    geoms = [geom for geom in polygon_gdf.geometry if geom is not None]
    mask = features.geometry_mask(
        geoms,
        out_shape=Ztop.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )

    Ztop2 = np.where(mask, Ztop, np.nan).astype(np.float32)
    return np.flipud(Ztop2)


def _pick_station_name_column(df: pd.DataFrame, explicit: Optional[str], fallback_id_col: str) -> str:
    if explicit and explicit in df.columns:
        return explicit
    for c in ["Nosaukums", "nosaukums", "Stacija", "stacija", "Station", "station", "Name", "name", "STATION", "NAME"]:
        if c in df.columns:
            return c
    return fallback_id_col


def _ceil_to(x: float, step: float) -> float:
    if not np.isfinite(x):
        return step
    if step <= 0:
        return float(x)
    return float(np.ceil(x / step) * step)


def _quantize_mm(Z: np.ndarray, step: float) -> np.ndarray:
    """Quantize values to nearest step (mm)."""
    if step <= 0:
        return Z
    Z = Z.astype(np.float32, copy=False)
    out = Z.copy()
    m = np.isfinite(out)
    out[m] = np.round(out[m] / step) * step
    out[m] = np.maximum(out[m], 0.0)
    return out.astype(np.float32)


def _autodetect_meta_cols(meta: pd.DataFrame, id_hint: Optional[str], name_hint: Optional[str]) -> Tuple[str, str]:
    """
    Try to find (id_col, name_col) in your staciju_dati_norma3.csv robustly.
    """
    cols = list(meta.columns)

    def norm(s: str) -> str:
        return str(s).strip().lower()

    ncols = {norm(c): c for c in cols}

    # explicit hints
    if id_hint and id_hint in cols and name_hint and name_hint in cols:
        return id_hint, name_hint
    if id_hint and id_hint in cols and name_hint is None:
        # pick name later
        pass
    if name_hint and name_hint in cols and id_hint is None:
        # pick id later
        pass

    # likely id columns
    id_candidates = []
    for key in ["gh_id", "ghid", "id", "station_id", "st_id", "kods", "code"]:
        if key in ncols:
            id_candidates.append(ncols[key])
    # also any column containing "gh" and "id"
    for c in cols:
        lc = norm(c)
        if "gh" in lc and "id" in lc and c not in id_candidates:
            id_candidates.append(c)

    # likely name columns
    name_candidates = []
    for key in ["nosaukums", "stacija", "station", "name", "nosauk"]:
        if key in ncols:
            name_candidates.append(ncols[key])
    for c in cols:
        lc = norm(c)
        if ("nosauk" in lc or "stacij" in lc or "station" in lc or lc == "name") and c not in name_candidates:
            name_candidates.append(c)

    # apply explicit hint preference
    if id_hint and id_hint in cols:
        id_col = id_hint
    else:
        id_col = id_candidates[0] if id_candidates else cols[0]  # fallback

    if name_hint and name_hint in cols:
        name_col = name_hint
    else:
        # avoid choosing same as id
        name_col = None
        for c in name_candidates:
            if c != id_col:
                name_col = c
                break
        if name_col is None:
            # fallback: first different column
            for c in cols:
                if c != id_col:
                    name_col = c
                    break
        if name_col is None:
            name_col = id_col

    return id_col, name_col


def _apply_station_names_from_meta(
    sub: pd.DataFrame,
    station_id_col: str,
    stations_meta_csv: str,
    meta_sep: str,
    meta_decimal: str,
    meta_id_col: Optional[str],
    meta_name_col: Optional[str],
) -> pd.DataFrame:
    """
    Adds/overwrites 'station_name' column in sub using mapping from stations_meta_csv.
    Keeps fallback to id if mapping missing.
    """
    sub = sub.copy()
    sub["station_name"] = sub[station_id_col].astype(str)

    if not stations_meta_csv:
        return sub
    if not os.path.exists(stations_meta_csv):
        print(f"[warn] stations_meta_csv not found: {stations_meta_csv} (will use ids as names)")
        return sub

    meta = pd.read_csv(stations_meta_csv, sep=meta_sep, decimal=meta_decimal, encoding="utf-8", low_memory=False)
    meta.columns = [c.strip().replace("\ufeff", "") for c in meta.columns]

    id_col, name_col = _autodetect_meta_cols(meta, meta_id_col, meta_name_col)

    # mapping
    m = meta.dropna(subset=[id_col]).copy()
    m[id_col] = m[id_col].astype(str)
    if name_col in m.columns:
        m[name_col] = m[name_col].astype(str)
    else:
        m[name_col] = m[id_col].astype(str)

    mapping = dict(zip(m[id_col].values, m[name_col].values))

    sub["station_name"] = sub[station_id_col].astype(str).map(mapping).fillna(sub[station_id_col].astype(str))
    return sub


def _draw_logo(fig: plt.Figure, logo_path: str, zoom: float = 0.12, pad: float = 0.01):
    """
    Draw logo at top-left corner of the *figure* (very left corner).
    """
    if not logo_path:
        return
    if not os.path.exists(logo_path):
        print(f"[warn] logo not found: {logo_path}")
        return

    try:
        img = plt.imread(logo_path)
        oi = OffsetImage(img, zoom=zoom)
        # Figure coordinates: (0,0)=bottom-left, (1,1)=top-right
        ab = AnnotationBbox(
            oi,
            (pad, 1.0 - pad),
            xycoords="figure fraction",
            frameon=False,
            box_alignment=(0, 1),  # align left/top
            zorder=200,
        )
        fig.add_artist(ab)
    except Exception as e:
        print(f"[warn] failed to draw logo: {e}")


def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.out_png) or ".", exist_ok=True)

    # If user provided --robeza_shp, use it as lv_border
    if getattr(a, "robeza_shp", None):
        a.lv_border = a.robeza_shp

    cmap = lvgmc_precip_cmap()

    # --- load grid centers ---
    g = pd.read_csv(a.grid_csv, sep=a.grid_sep, decimal=a.grid_decimal, low_memory=False)
    g.columns = [c.strip().replace("\ufeff", "") for c in g.columns]
    if a.grid_x_col not in g.columns or a.grid_y_col not in g.columns:
        raise ValueError(f"Grid CSV missing columns: {a.grid_x_col}, {a.grid_y_col}")

    xs = np.sort(g[a.grid_x_col].astype(float).unique())
    ys = np.sort(g[a.grid_y_col].astype(float).unique())
    if xs.size < 2 or ys.size < 2:
        raise RuntimeError("Grid CSV must contain full 2D grid centers (>=2 unique x and y).")

    # --- sample tif at centers ---
    Z, r_crs = sample_tif_at_centers_vectorized(a.analysis_tif, xs, ys, a.grid_crs)

    # optional smoothing (display-only)
    Zp = gaussian_smooth_nan(Z, a.smooth_sigma) if (a.smooth_sigma and a.smooth_sigma > 0) else Z

    # optional clip to LV border (crisp coastline)
    x_edges = edges_from_centers_midpoint(xs)
    y_edges = edges_from_centers_midpoint(ys)
    west, east = float(x_edges[0]), float(x_edges[-1])
    south, north = float(y_edges[0]), float(y_edges[-1])

    # --- vectors ---
    lv_border = None
    lv_muni = None
    nbr = None
    if gpd is not None:
        if a.neighbors:
            nbr = gpd.read_file(a.neighbors)
            if r_crs is not None:
                nbr = nbr.to_crs(r_crs)
        if a.lv_muni:
            lv_muni = gpd.read_file(a.lv_muni)
            if r_crs is not None:
                lv_muni = lv_muni.to_crs(r_crs)
        if a.lv_border:
            lv_border = gpd.read_file(a.lv_border)
            if r_crs is not None:
                lv_border = lv_border.to_crs(r_crs)

    if a.clip_to_lv:
        if lv_border is None:
            raise RuntimeError("--clip_to_lv requires --lv_border/--robeza_shp")
        Zp = clip_grid_to_polygon(Zp, x_edges, y_edges, lv_border)

    # --- stations (monthly) ---
    df = pd.read_csv(a.stations_csv, sep=a.csv_sep, decimal=a.csv_decimal, encoding="utf-8", low_memory=False)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    need = [a.station_id_col, "Gads", "Menesis", a.station_value_col, a.station_lon_col, a.station_lat_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Stations CSV missing columns: {missing}")

    sub = df[(df["Gads"] == a.year) & (df["Menesis"] == a.month)].copy()
    sub = sub.dropna(subset=[a.station_lon_col, a.station_lat_col, a.station_value_col]).copy()
    if len(sub) == 0:
        raise RuntimeError(f"No station rows for {a.year}-{a.month:02d}")

    # CRS transform coords
    sx = sub[a.station_lon_col].to_numpy(dtype=float)
    sy = sub[a.station_lat_col].to_numpy(dtype=float)
    sx, sy = transform_points_to_crs(sx, sy, a.stations_crs, r_crs)

    ok = (sx >= west) & (sx <= east) & (sy >= south) & (sy <= north)
    sub = sub.loc[ok].copy()
    sx = sx[ok]
    sy = sy[ok]
    if len(sub) == 0:
        raise RuntimeError("No stations inside map bounds after CRS transform.")

    # ---- station names: prefer monthly CSV name col if present; otherwise use meta mapping ----
    picked = _pick_station_name_column(sub, a.station_name_col, a.station_id_col)
    if picked == a.station_id_col:
        # no name in monthly -> use your staciju_dati_norma3.csv
        sub = _apply_station_names_from_meta(
            sub=sub,
            station_id_col=a.station_id_col,
            stations_meta_csv=a.stations_meta_csv,
            meta_sep=a.meta_sep,
            meta_decimal=a.meta_decimal,
            meta_id_col=a.meta_id_col,
            meta_name_col=a.meta_name_col,
        )
        name_col = "station_name"
    else:
        name_col = picked

    names = sub[name_col].astype(str).tolist()
    vals = sub[a.station_value_col].to_numpy(dtype=float)

    # ---- requested scaling ----
    Zq = _quantize_mm(Zp, float(a.quant_step))

    vmin = 0.0
    max_field = float(np.nanmax(Zq)) if np.isfinite(np.nanmax(Zq)) else 0.0
    max_st = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 0.0
    vmax_raw = max(max_field, max_st, 0.0)

    vmax = _ceil_to(vmax_raw, float(a.vmax_round))
    if vmax <= 0:
        vmax = float(a.vmax_round)

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    tick_step = float(a.tick_step)
    ticks = np.arange(0.0, vmax + 0.5 * tick_step, tick_step, dtype=float)
    if ticks.size == 0 or ticks[-1] != vmax:
        if ticks.size == 0:
            ticks = np.array([0.0, vmax], dtype=float)
        else:
            ticks[-1] = vmax

    # --- FIGURE ---
    fig = plt.figure(figsize=(10.8, 6.2), dpi=a.dpi)
    fig.patch.set_facecolor(a.bg_color)

    ax = fig.add_axes([0.00, 0.16, 0.96, 0.80])
    ax.set_facecolor(a.bg_color)
    ax.set_axis_off()

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    # neighbors
    if nbr is not None:
        try:
            nbr.plot(ax=ax, facecolor=a.neighbors_fill, edgecolor=a.neighbors_edge, linewidth=0.7, zorder=0)
        except Exception:
            nbr.boundary.plot(ax=ax, color=a.neighbors_edge, linewidth=0.7, zorder=0)

    # raster (quantized)
    Zm = np.ma.masked_invalid(Zq)
    pm = ax.pcolormesh(
        x_edges,
        y_edges,
        Zm,
        shading="flat",
        cmap=cmap,
        norm=norm,
        zorder=1,
        antialiased=False,
        linewidth=0,
    )

    # admin boundaries
    if lv_muni is not None:
        lv_muni.boundary.plot(ax=ax, color=a.muni_col, linewidth=0.35, zorder=2, alpha=0.85)

    # LV border
    if lv_border is not None:
        lv_border.boundary.plot(ax=ax, color=a.border_col, linewidth=1.2, zorder=3)

    # stations: white halo + black dot
    ax.scatter(sx, sy, s=42, c="white", edgecolors="white", linewidths=0.0, zorder=5)
    ax.scatter(sx, sy, s=18, c="black", edgecolors="black", linewidths=0.3, zorder=6)

    # labels: NAME + value (0 decimals)
    if a.label_stations:
        order = np.argsort(vals)[::-1]
        order = order[: min(len(order), a.label_top_n, a.max_labels)]

        dx = (east - west) * (a.label_dx_frac * 0.55)
        dy = (north - south) * (a.label_dy_frac * 0.55)

        for i in order:
            name = str(names[i])
            v = float(vals[i])
            vtxt = f"{v:.0f}"
            if a.lang == "LV":
                vtxt = vtxt.replace(".", ",")

            txt = f"{name}\n{vtxt}"

            t = ax.text(
                float(sx[i]) + dx,
                float(sy[i]) + dy,
                txt,
                fontsize=7.5,
                linespacing=0.9,
                ha="left",
                va="bottom",
                color="black",
                zorder=10,
            )
            t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9)])

    # title
    if a.lang == "LV":
        month_name = LV_MONTHS.get(a.month, str(a.month))
        line1 = f"{a.year}. gada {month_name}"
        line2 = "Nokrišņu daudzums, mm"
    else:
        month_name = EN_MONTHS.get(a.month, str(a.month))
        line1 = f"{month_name} {a.year}"
        line2 = "Precipitation, mm"

    ax.text(0.03, 0.08, line1, transform=ax.transAxes, fontsize=22, color="black",
            ha="left", va="bottom", zorder=40)
    ax.text(0.03, 0.01, line2, transform=ax.transAxes, fontsize=22, fontweight="bold",
            color="black", ha="left", va="bottom", zorder=40)

    # colorbar bottom (nice ticks)
    cax = fig.add_axes([0.03, 0.07, 0.84, 0.028])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=14, length=0)
    cb.ax.xaxis.set_major_formatter(comma_formatter(a.lang, ndp=0))
    cb.set_label("")

    # logo at very left corner (figure top-left)
    _draw_logo(fig, a.logo_png, zoom=float(a.logo_zoom), pad=float(a.logo_pad))

    plt.savefig(a.out_png, dpi=a.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Wrote: {a.out_png}")
    print(f"[scale] vmin={vmin:.0f} vmax_raw={vmax_raw:.1f} -> vmax={vmax:.0f} (rounded by {a.vmax_round})")
    print(f"[quant] step={a.quant_step}mm | [ticks] step={a.tick_step}mm")
    if name_col == "station_name":
        print(f"[names] used meta mapping from: {a.stations_meta_csv}")
    else:
        print(f"[names] used column in stations_csv: {name_col}")


if __name__ == "__main__":
    main()
