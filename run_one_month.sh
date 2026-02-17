#!/usr/bin/env bash
set -euo pipefail

YEAR=2025
MONTH=10   # <-- maini te

PY="/home/denissbokadenissboka/projects/gridpp_lab/.venv/bin/python"

MAIN_PY="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/src/main.py"
COMPARE_PY="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/src/compare_loocv.py"

GRID_CSV="/home/denissbokadenissboka/projects/gridpp_lab/1x1_LV_grid_2024_xy2.csv"
STATIONS_MONTHLY="/home/denissbokadenissboka/projects/gridpp_lab/HPRAB_monthly_1990_2024_15plusStations_WITH_COORDS.csv"
META_CSV="/home/denissbokadenissboka/projects/gridpp_lab/staciju_dati_norma3.csv"

ERA_NC="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/data/era5/era5_tp_mm_month_${YEAR}_$(printf "%02d" $MONTH).nc"
OUT_DIR="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/out_${YEAR}_$(printf "%02d" $MONTH)_test"

# --------- GridPP params (main.py + loocv) ----------
THREADS=4
EPS=1
LVAL=41500
POBS_D=0.1 
MAX_POINTS=1 
CV_RADIUS=0 

# --------- UK params (salīdzināšanai) ---------------
UK_RANGE_M=41500
UK_NUGGET=0

META_ID_COL="gh_id"
META_CONT_COL="cont_pr"
META_H_COL="elevation"

mkdir -p "$OUT_DIR"

echo "[RUN] ${YEAR}-$(printf "%02d" $MONTH) -> $OUT_DIR"
echo "[GridPP] eps=$EPS L=$LVAL pobs_d=$POBS_D max_points=$MAX_POINTS cv_radius=$CV_RADIUS threads=$THREADS"
echo "[UK] range=$UK_RANGE_M nugget=$UK_NUGGET"

# 1) Analysis field (tif + report.txt)
"$PY" -u "$MAIN_PY" \
  --grid "$GRID_CSV" \
  --stations "$STATIONS_MONTHLY" \
  --year "$YEAR" --month "$MONTH" \
  --era_nc "$ERA_NC" \
  --out "$OUT_DIR" \
  --threads "$THREADS" \
  --eps "$EPS" \
  --L "$LVAL" \
  --pobs_d "$POBS_D" \
  --max_points "$MAX_POINTS"

# 2) LOOCV (GridPP + UK) -> loocv_compare.csv
"$PY" -u "$COMPARE_PY" \
  --grid_csv "$GRID_CSV" \
  --stations_monthly_csv "$STATIONS_MONTHLY" \
  --stations_meta_csv "$META_CSV" \
  --era_nc "$ERA_NC" \
  --year "$YEAR" --month "$MONTH" \
  --out "$OUT_DIR" \
  --meta_id_col "$META_ID_COL" \
  --meta_cont_col "$META_CONT_COL" \
  --meta_h_col "$META_H_COL" \
  --uk_range_m "$UK_RANGE_M" \
  --uk_nugget "$UK_NUGGET" \
  --eps "$EPS" \
  --L "$LVAL" \
  --pobs_d "$POBS_D" \
  --max_points "$MAX_POINTS" \
  --cv_radius "$CV_RADIUS"

# 3) Pretty print MAE/RMSE/BIAS from loocv_compare.csv
CSV="$OUT_DIR/loocv_compare.csv"
"$PY" - <<PY
import pandas as pd, numpy as np

YEAR = int("$YEAR")
MONTH = int("$MONTH")

df=pd.read_csv("$CSV")
obs=df["obs"].to_numpy(float)
pg=df["pred_gridpp"].to_numpy(float)
pu=df["pred_uk"].to_numpy(float)

m=np.isfinite(obs)&np.isfinite(pg)&np.isfinite(pu)
obs,pg,pu=obs[m],pg[m],pu[m]

def met(pred):
    e=pred-obs
    mae=float(np.mean(np.abs(e)))
    rmse=float(np.sqrt(np.mean(e**2)))
    bias=float(np.mean(e))
    return mae,rmse,bias

g=met(pg); u=met(pu)

print(f"[LOOCV {YEAR}-{MONTH:02d}] N={len(obs)}")
print(f"  GridPP: MAE={g[0]:.3f} RMSE={g[1]:.3f} BIAS={g[2]:.3f}")
print(f"  UK    : MAE={u[0]:.3f} RMSE={u[1]:.3f} BIAS={u[2]:.3f}")
PY