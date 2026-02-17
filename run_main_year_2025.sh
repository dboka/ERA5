#!/usr/bin/env bash
set -euo pipefail

PY="/home/denissbokadenissboka/projects/gridpp_lab/.venv/bin/python"
MAIN="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/src/main.py"

GRID="/home/denissbokadenissboka/projects/gridpp_lab/1x1_LV_grid_2024_xy2.csv"
STATIONS="/home/denissbokadenissboka/projects/gridpp_lab/HPRAB_monthly_1990_2024_15plusStations_WITH_COORDS.csv"
ERA_DIR="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/data/era5"

YEAR=2025
OUT_BASE="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/out_${YEAR}_main_runs"

THREADS=4
LVAL=41500
POBS_D=0.1
MAX_POINTS=29

mkdir -p "$OUT_BASE"

echo "[RUN] year=$YEAR -> $OUT_BASE"
echo "[CFG] L=$LVAL pobs_d=$POBS_D max_points=$MAX_POINTS threads=$THREADS"

for m in $(seq -w 1 12); do
  ERA_NC="${ERA_DIR}/era5_tp_mm_month_${YEAR}_${m}.nc"
  if [[ ! -f "$ERA_NC" ]]; then
    echo "[SKIP] missing ERA: $ERA_NC"
    continue
  fi

  OUT_DIR="${OUT_BASE}/${YEAR}_${m}"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "==================== ${YEAR}-${m} ===================="

  "$PY" -u "$MAIN" \
    --grid "$GRID" \
    --stations "$STATIONS" \
    --year "$YEAR" --month "$((10#$m))" \
    --era_nc "$ERA_NC" \
    --out "$OUT_DIR" \
    --threads "$THREADS" \
    --L "$LVAL" \
    --pobs_d "$POBS_D" \
    --max_points "$MAX_POINTS"
done

echo ""
echo "[DONE] Outputs in: $OUT_BASE"
