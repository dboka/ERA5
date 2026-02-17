#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run compare_loocv.py and generate an HTML report (GridPP vs UK)")
    p.add_argument("--python", required=True, help="Python executable (venv), e.g. .../.venv/bin/python")
    p.add_argument("--compare_py", required=True, help="Path to compare_loocv.py")
    p.add_argument("--grid_csv", required=True)
    p.add_argument("--stations_monthly_csv", required=True)
    p.add_argument("--stations_meta_csv", required=True)
    p.add_argument("--era_nc", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)

    # output directory for compare_loocv + report
    p.add_argument("--out", required=True, help="Output directory (will contain loocv_compare.csv + report.html)")

    # meta mapping
    p.add_argument("--meta_id_col", default="gh_id")
    p.add_argument("--meta_cont_col", default="cont_pr")
    p.add_argument("--meta_h_col", default="elevation")

    # UK params
    p.add_argument("--uk_range_m", type=float, default=41500.0)
    p.add_argument("--uk_nugget", type=float, default=0.0)

    # GridPP params
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--L", type=float, default=41500.0)
    p.add_argument("--pobs_d", type=float, default=0.1)
    p.add_argument("--max_points", type=int, default=29)
    p.add_argument("--cv_radius", type=float, default=2000.0)

    p.add_argument("--title", default=None, help="Optional HTML title")
    return p.parse_args()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def run_compare(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Runs compare_loocv.py and returns (stdout, csv_path).
    """
    ensure_dir(args.out)
    cmd = [
        args.python,
        args.compare_py,
        "--grid_csv", args.grid_csv,
        "--stations_monthly_csv", args.stations_monthly_csv,
        "--stations_meta_csv", args.stations_meta_csv,
        "--era_nc", args.era_nc,
        "--year", str(args.year),
        "--month", str(args.month),
        "--out", args.out,
        "--meta_id_col", args.meta_id_col,
        "--meta_cont_col", args.meta_cont_col,
        "--meta_h_col", args.meta_h_col,
        "--uk_range_m", str(args.uk_range_m),
        "--uk_nugget", str(args.uk_nugget),
        "--eps", str(args.eps),
        "--L", str(args.L),
        "--pobs_d", str(args.pobs_d),
        "--max_points", str(args.max_points),
        "--cv_radius", str(args.cv_radius),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "compare_loocv.py failed\n\n"
            f"CMD:\n{' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n"
        )

    csv_path = os.path.join(args.out, "loocv_compare.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

    return proc.stdout, csv_path


def metrics(err: np.ndarray) -> Dict[str, float]:
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "BIAS": np.nan}
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    return {"MAE": mae, "RMSE": rmse, "BIAS": bias}


def fig_to_base64_png(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def scatter_obs_pred(obs: np.ndarray, pred: np.ndarray, title: str) -> str:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    m = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[m], pred[m]

    fig = plt.figure(figsize=(5.6, 4.4))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(obs, pred, s=22)
    if obs.size:
        mn = float(min(obs.min(), pred.min()))
        mx = float(max(obs.max(), pred.max()))
        ax.plot([mn, mx], [mn, mx])
    ax.set_xlabel("Obs (mm)")
    ax.set_ylabel("Pred (mm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    return fig_to_base64_png(fig)


def hist_errors(err: np.ndarray, title: str) -> str:
    err = np.asarray(err, float)
    err = err[np.isfinite(err)]
    fig = plt.figure(figsize=(5.6, 4.4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(err, bins=14)
    ax.set_xlabel("Error (Pred - Obs), mm")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    return fig_to_base64_png(fig)


def build_html(
    title: str,
    stdout_text: str,
    df: pd.DataFrame,
    out_html: str,
) -> None:
    # expected columns (best-effort)
    # We'll try common names; adjust if your csv differs.
    col_obs = "obs"
    col_pred_g = "pred_gridpp"
    col_pred_u = "pred_uk"

    missing = [c for c in [col_obs, col_pred_g, col_pred_u] if c not in df.columns]
    if missing:
        raise ValueError(f"loocv_compare.csv is missing required columns: {missing}\nColumns: {list(df.columns)}")

    obs = df[col_obs].to_numpy(float)
    pred_g = df[col_pred_g].to_numpy(float)
    pred_u = df[col_pred_u].to_numpy(float)
    err_g = pred_g - obs
    err_u = pred_u - obs

    mg = metrics(err_g)
    mu = metrics(err_u)

    # Figures
    img_sc_g = scatter_obs_pred(obs, pred_g, "GridPP LOOCV: Obs vs Pred")
    img_sc_u = scatter_obs_pred(obs, pred_u, "UK (R drift+OK): Obs vs Pred")
    img_hi_g = hist_errors(err_g, "GridPP LOOCV errors")
    img_hi_u = hist_errors(err_u, "UK LOOCV errors")

    # Top errors table
    df2 = df.copy()
    df2["err_gridpp"] = err_g
    df2["err_uk"] = err_u
    df2["abs_err_gridpp"] = np.abs(err_g)
    df2["abs_err_uk"] = np.abs(err_u)

    # Try to show station id/name if present
    station_cols = [c for c in ["gh_id", "Stations", "station", "name", "Nosaukums"] if c in df2.columns]
    show_cols = station_cols + [col_obs, col_pred_g, "err_gridpp", col_pred_u, "err_uk"]
    show_cols = [c for c in show_cols if c in df2.columns]

    top_g = df2.sort_values("abs_err_gridpp", ascending=False).head(10)[show_cols]
    top_u = df2.sort_values("abs_err_uk", ascending=False).head(10)[show_cols]

    def df_to_html_table(d: pd.DataFrame) -> str:
        return d.to_html(index=False, float_format=lambda x: f"{x:.2f}")

    html = f"""<!doctype html>
<html lang="lv">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{title}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color: #111; }}
h1 {{ margin: 0 0 6px 0; }}
.small {{ color:#555; font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }}
.card {{ border: 1px solid #e6e6e6; border-radius: 14px; padding: 14px 14px; background: #fff; }}
.kpi {{ display:flex; gap:12px; flex-wrap:wrap; }}
.kpi .item {{ border:1px solid #eee; border-radius: 12px; padding:10px 12px; }}
code, pre {{ background:#f6f7f8; border:1px solid #eee; border-radius:10px; padding: 10px; overflow:auto; }}
img {{ max-width: 100%; height: auto; border-radius: 10px; border:1px solid #eee; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #eee; padding: 6px 8px; font-size: 13px; }}
th {{ background: #fafafa; text-align:left; }}
</style>
</head>
<body>

<h1>{title}</h1>
<div class="small">LOOCV salīdzinājums: <b>GridPP log-ratio OI</b> vs <b>UK (R drift + OK reziduāli)</b></div>

<div style="height:14px"></div>

<div class="grid">
  <div class="card">
    <h3 style="margin:0 0 10px 0;">Kopsavilkums (GridPP)</h3>
    <div class="kpi">
      <div class="item"><b>MAE</b><div>{mg["MAE"]:.3f}</div></div>
      <div class="item"><b>RMSE</b><div>{mg["RMSE"]:.3f}</div></div>
      <div class="item"><b>BIAS</b><div>{mg["BIAS"]:.3f}</div></div>
      <div class="item"><b>N</b><div>{int(np.isfinite(obs).sum())}</div></div>
    </div>
  </div>

  <div class="card">
    <h3 style="margin:0 0 10px 0;">Kopsavilkums (UK)</h3>
    <div class="kpi">
      <div class="item"><b>MAE</b><div>{mu["MAE"]:.3f}</div></div>
      <div class="item"><b>RMSE</b><div>{mu["RMSE"]:.3f}</div></div>
      <div class="item"><b>BIAS</b><div>{mu["BIAS"]:.3f}</div></div>
      <div class="item"><b>N</b><div>{int(np.isfinite(obs).sum())}</div></div>
    </div>
  </div>
</div>

<div style="height:18px"></div>

<div class="grid">
  <div class="card">
    <h3 style="margin:0 0 10px 0;">Obs vs Pred</h3>
    <div class="small">GridPP</div>
    <img src="data:image/png;base64,{img_sc_g}" />
    <div style="height:10px"></div>
    <div class="small">UK</div>
    <img src="data:image/png;base64,{img_sc_u}" />
  </div>

  <div class="card">
    <h3 style="margin:0 0 10px 0;">Kļūdu sadalījums</h3>
    <div class="small">GridPP</div>
    <img src="data:image/png;base64,{img_hi_g}" />
    <div style="height:10px"></div>
    <div class="small">UK</div>
    <img src="data:image/png;base64,{img_hi_u}" />
  </div>
</div>

<div style="height:18px"></div>

<div class="grid">
  <div class="card">
    <h3 style="margin:0 0 10px 0;">Top 10 kļūdas (GridPP)</h3>
    {df_to_html_table(top_g)}
  </div>
  <div class="card">
    <h3 style="margin:0 0 10px 0;">Top 10 kļūdas (UK)</h3>
    {df_to_html_table(top_u)}
  </div>
</div>

<div style="height:18px"></div>

<div class="card">
  <h3 style="margin:0 0 10px 0;">compare_loocv.py output (stdout)</h3>
  <pre>{stdout_text}</pre>
</div>

</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    title = args.title or f"LOOCV salīdzinājums {args.year}-{args.month:02d}: GridPP vs UK"

    stdout_text, csv_path = run_compare(args)
    df = pd.read_csv(csv_path)

    out_html = os.path.join(args.out, "report.html")
    build_html(title=title, stdout_text=stdout_text, df=df, out_html=out_html)
    print(f"[OK] Wrote HTML report: {out_html}")


if __name__ == "__main__":
    main()
