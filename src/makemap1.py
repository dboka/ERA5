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
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe

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
    p = argparse.ArgumentParser("Make LVGMC-style production map (R-like geom_raster via pcolormesh)")

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
    p.add_argument("--lv_muni", default=None, help="Latvia municipalities/admin borders (gpkg/shp). Optional.")
    p.add_argument("--neighbors", default=None, help="Neighbor countries vector (gpkg/shp). Optional.")

    # clip raster to LV border polygon (makes coastline crisp)
    p.add_argument("--clip_to_lv", action="store_true", help="Clip raster to lv_border polygon")

    # logo
    p.add_argument("--logo_png", default=None, help="LVGMC logo PNG (transparent recommended). Optional.")
    p.add_argument("--logo_xy", default="0.03,0.94", help="Logo anchor in axes fraction (x,y). default 0.03,0.94")
    p.add_argument("--logo_zoom", type=float, default=0.22, help="Logo scale factor (rough).")

    # Stations CSV columns
    p.add_argument("--csv_sep", default=";", help="Stations CSV separator (default ;)")
    p.add_argument("--csv_decimal", default=".", help="Stations CSV decimal mark (default .)")
    p.add_argument("--station_value_col", default="month_sum", help="Value column (default month_sum)")
    p.add_argument("--station_lon_col", default="lon")
    p.add_argument("--station_lat_col", default="lat")
    p.add_argument("--station_name_col", default=None, help="Optional name column. If missing -> auto-detect, else gh_id.")
    p.add_argument("--station_id_col", default="gh_id", help="Fallback label id column (default gh_id)")
    p.add_argument("--stations_crs", default="EPSG:4326", help="CRS of station lon/lat (default EPSG:4326)")

    # styling
    p.add_argument("--lang", default="LV", choices=["LV", "EN"])
    p.add_argument("--bg_color", default="#EFF3F6")
    p.add_argument("--neighbors_fill", default="#E6EAEE")
    p.add_argument("--neighbors_edge", default="#C9CED3")
    p.add_argument("--border_col", default="#48525B")
    p.add_argument("--muni_col", default="#7C8893")

    # classification controls (your request: 0..100 by 0.5)
    p.add_argument("--class_min", type=float, default=0.0, help="Classification minimum (default 0)")
    p.add_argument("--class_max", type=float, default=100.0, help="Classification maximum (default 100)")
    p.add_argument("--class_step", type=float, default=0.5, help="Class step (default 0.5)")

    # "class boundary" lines like in the example (NOT black isolines):
    # these are subtle white-ish lines drawn at class boundaries using contour,
    # but styled to be light (like LVGMC map).
    p.add_argument("--class_lines", action="store_true", help="Draw subtle class boundary lines (like LVGMC)")
    p.add_argument("--class_line_step", type=float, default=5.0, help="Boundary line step in mm (e.g. 5 or 10)")
    p.add_argument("--class_line_color", default="#FFFFFF", help="Boundary line color (default white)")
    p.add_argument("--class_line_lw", type=float, default=0.55, help="Boundary line width")
    p.add_argument("--class_line_alpha", type=float, default=0.35, help="Boundary line alpha")

    # colorbar ticks (keep simple like LVGMC: 0,50,100)
    p.add_argument("--cb_tick_step", type=float, default=50.0, help="Colorbar tick step (default 50 => 0,50,100)")

    p.add_argument("--cmap", default="lvgmc_precip", help="lvgmc_precip OR any matplotlib cmap name")

    # display smoothing (display-only)
    p.add_argument("--smooth_sigma", type=float, default=0.0, help="Display-only Gaussian smoothing sigma in grid cells (0=off)")

    p.add_argument("--label_stations", action="store_true")
    p.add_argument("--max_labels", type=int, default=80)
    p.add_argument("--label_top_n", type=int, default=9999, help="Label only top-N by value (default all)")
    p.add_argument("--label_dx_frac", type=float, default=0.006, help="Label x offset as fraction of map width")
    p.add_argument("--label_dy_frac", type=float, default=0.006, help="Label y offset as fraction of map height")
    p.add_argument("--label_fontsize", type=float, default=8.5, help="Label font size (default 8.5)")
    p.add_argument("--label_value_decimals", type=int, default=1, help="Decimals for month_sum labels (default 1)")

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


def comma_formatter(lang: str, ndp: int = 1):
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


def sample_tif_at_centers_vectorized(
    tif_path: str, xs: np.ndarray, ys: np.ndarray, grid_crs: str
) -> Tuple[np.ndarray, object]:
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

    candidates = [
        "Nosaukums", "nosaukums",
        "Stacija", "stacija",
        "Station", "station",
        "Name", "name",
        "STATION", "NAME",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    return fallback_id_col


def make_class_breaks(vmin: float = 0.0, vmax: float = 100.0, step: float = 0.5) -> np.ndarray:
    if step <= 0:
        raise ValueError("class_step must be > 0")
    if vmax <= vmin:
        raise ValueError("class_max must be > class_min")

    n = int(round((vmax - vmin) / step))
    edges = vmin + step * np.arange(n + 1, dtype=float)
    edges[0] = vmin
    edges[-1] = vmax
    return edges


def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.out_png) or ".", exist_ok=True)

    base_cmap = lvgmc_precip_cmap() if a.cmap == "lvgmc_precip" else plt.get_cmap(a.cmap)

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

    # --- optional display smoothing ---
    Zp = gaussian_smooth_nan(Z, a.smooth_sigma) if (a.smooth_sigma and a.smooth_sigma > 0) else Z
    if not np.isfinite(Zp).any():
        raise RuntimeError("No finite values after sampling. Check CRS/extents.")

    # edges from centers
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
            if not os.path.exists(a.neighbors):
                raise FileNotFoundError(f"neighbors file not found: {a.neighbors}")
            nbr = gpd.read_file(a.neighbors)
            if r_crs is not None:
                nbr = nbr.to_crs(r_crs)

        if a.lv_muni:
            if not os.path.exists(a.lv_muni):
                raise FileNotFoundError(f"lv_muni file not found: {a.lv_muni}")
            lv_muni = gpd.read_file(a.lv_muni)
            if r_crs is not None:
                lv_muni = lv_muni.to_crs(r_crs)

        if a.lv_border:
            if not os.path.exists(a.lv_border):
                raise FileNotFoundError(f"lv_border file not found: {a.lv_border}")
            lv_border = gpd.read_file(a.lv_border)
            if r_crs is not None:
                lv_border = lv_border.to_crs(r_crs)

    # --- optional crisp clip ---
    if a.clip_to_lv:
        if lv_border is None:
            raise RuntimeError("--clip_to_lv requires --lv_border")
        Zp = clip_grid_to_polygon(Zp, x_edges, y_edges, lv_border)

    # --- stations ---
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

    sx = sub[a.station_lon_col].to_numpy(dtype=float)
    sy = sub[a.station_lat_col].to_numpy(dtype=float)
    sx, sy = transform_points_to_crs(sx, sy, a.stations_crs, r_crs)

    ok = (sx >= west) & (sx <= east) & (sy >= south) & (sy <= north)
    sub = sub.loc[ok].copy()
    sx = sx[ok]
    sy = sy[ok]
    if len(sub) == 0:
        raise RuntimeError("No stations inside map bounds after CRS transform.")

    name_col = _pick_station_name_column(sub, a.station_name_col, a.station_id_col)
    names = sub[name_col].astype(str).tolist()
    vals = sub[a.station_value_col].to_numpy(dtype=float)

    # --- DISCRETE classification ---
    class_min = float(a.class_min)
    class_max = float(a.class_max)
    class_step = float(a.class_step)

    # Clip display field to class range
    Zp = np.clip(Zp, class_min, class_max).astype(np.float32)

    bounds = make_class_breaks(class_min, class_max, class_step)
    n_classes = len(bounds) - 1

    cmap_disc = base_cmap.resampled(n_classes)
    norm = BoundaryNorm(bounds, cmap_disc.N, clip=True)

    # --- FIGURE ---
    fig = plt.figure(figsize=(10.8, 6.2), dpi=a.dpi)
    fig.patch.set_facecolor(a.bg_color)

    ax = fig.add_axes([0.02, 0.16, 0.96, 0.80])
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

    # raster
    Zm = np.ma.masked_invalid(Zp)
    pm = ax.pcolormesh(
        x_edges,
        y_edges,
        Zm,
        shading="flat",
        cmap=cmap_disc,
        norm=norm,
        zorder=1,
        antialiased=False,
        linewidth=0,
    )

    # subtle class boundary lines (like example) - not black isolines
    if a.class_lines:
        Xc, Yc = np.meshgrid(xs, ys)
        step = float(a.class_line_step)
        if step > 0:
            lev = np.arange(class_min, class_max + 1e-9, step, dtype=float)
            Zc = np.asarray(Zp, dtype=float)
            Zc = np.where(np.isfinite(Zc), Zc, np.nan)

            ax.contour(
                Xc, Yc, Zc,
                levels=lev,
                colors=str(a.class_line_color),
                linewidths=float(a.class_line_lw),
                alpha=float(a.class_line_alpha),
                zorder=2.1,
            )

    # admin boundaries
    if lv_muni is not None:
        lv_muni.boundary.plot(ax=ax, color=a.muni_col, linewidth=0.35, zorder=2, alpha=0.85)

    # LV border
    if lv_border is not None:
        lv_border.boundary.plot(ax=ax, color=a.border_col, linewidth=1.2, zorder=3)

    # stations: white halo + black dot
    ax.scatter(sx, sy, s=46, c="white", edgecolors="white", linewidths=0.0, zorder=5)
    ax.scatter(sx, sy, s=18, c="black", edgecolors="black", linewidths=0.3, zorder=6)

    # labels: NAME + month_sum
    if a.label_stations:
        order = np.argsort(vals)[::-1]
        order = order[: min(len(order), a.label_top_n, a.max_labels)]

        dx = (east - west) * float(a.label_dx_frac)
        dy = (north - south) * float(a.label_dy_frac)

        for i in order:
            name = str(names[i])
            v = float(vals[i])

            fmt = f"{{:.{int(a.label_value_decimals)}f}}"
            vtxt = fmt.format(v)
            if a.lang == "LV":
                vtxt = vtxt.replace(".", ",")

            txt = f"{name}\n{vtxt}"
            t = ax.text(
                float(sx[i]) + dx,
                float(sy[i]) + dy,
                txt,
                fontsize=float(a.label_fontsize),
                linespacing=0.9,
                ha="left",
                va="bottom",
                color="black",
                zorder=10,
            )
            t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white", alpha=0.92)])

    # logo
    if a.logo_png:
        import matplotlib.image as mpimg
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        lx, ly = [float(x) for x in a.logo_xy.split(",")]
        img = mpimg.imread(a.logo_png)
        oi = OffsetImage(img, zoom=a.logo_zoom)
        ab = AnnotationBbox(
            oi,
            (lx, ly),
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(0, 1),
            zorder=30,
        )
        ax.add_artist(ab)

    # title
    if a.lang == "LV":
        month_name = LV_MONTHS.get(a.month, str(a.month))
        line1 = f"{a.year}. gada {month_name}"
        line2 = "Nokrišņu daudzums, mm"
    else:
        month_name = EN_MONTHS.get(a.month, str(a.month))
        line1 = f"{month_name} {a.year}"
        line2 = "Precipitation, mm"

    ax.text(0.12, 0.08, line1, transform=ax.transAxes, fontsize=22, color="black",
            ha="left", va="bottom", zorder=40)
    ax.text(0.12, 0.01, line2, transform=ax.transAxes, fontsize=22, fontweight="bold",
            color="black", ha="left", va="bottom", zorder=40)

    # colorbar bottom: show only coarse ticks (e.g. 0,50,100)
    cax = fig.add_axes([0.10, 0.07, 0.84, 0.028])
    tick_step = float(a.cb_tick_step)
    if tick_step <= 0:
        tick_step = 50.0
    ticks = np.arange(class_min, class_max + 1e-9, tick_step, dtype=float)

    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=14, length=0)
    cb.ax.xaxis.set_major_formatter(comma_formatter(a.lang, ndp=1))
    cb.set_label("")

    plt.savefig(a.out_png, dpi=a.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Wrote:", a.out_png)


if __name__ == "__main__":
    main()
