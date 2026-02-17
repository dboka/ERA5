#!/usr/bin/env bash
set -euo pipefail

PY="/home/denissbokadenissboka/projects/gridpp_lab/.venv/bin/python"
MAKEMAP="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/src/makemap.py"

GRID_CSV="/home/denissbokadenissboka/projects/gridpp_lab/1x1_LV_grid_2024_xy2.csv"
STATIONS_CSV="/home/denissbokadenissboka/projects/gridpp_lab/HPRAB_monthly_1990_2024_15plusStations_WITH_COORDS.csv"
META_CSV="/home/denissbokadenissboka/projects/gridpp_lab/staciju_dati_norma3.csv"

ROBEZA_SHP="/home/denissbokadenissboka/projects/gridpp_lab/robeza/Robeza.shp"
LOGO_PNG="/home/denissbokadenissboka/projects/gridpp_lab/images/LVGMC_1300-AM.png"

YEAR=2025
# kur ir mēnešu rezultātu mapes ar 02_analysis.tif
OUT_BASE="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/out_${YEAR}_main_runs"
# ja tev mapes saucas out_2025_01, out_2025_02, tad nomaini OUT_BASE uz "..../gridpp_precip" un zemāk OUT_DIR konstrukciju

for m in $(seq -w 1 12); do
  OUT_DIR="${OUT_BASE}/${YEAR}_${m}"
  TIF="${OUT_DIR}/02_analysis.tif"
  PNG="${OUT_DIR}/map3.png"

  if [[ ! -f "$TIF" ]]; then
    echo "[SKIP] missing tif: $TIF"
    continue
  fi

  echo ""
  echo "==================== ${YEAR}-${m} ===================="
  echo "[IN ] $TIF"
  echo "[OUT] $PNG"

  "$PY" "$MAKEMAP" \
    --analysis_tif "$TIF" \
    --grid_csv "$GRID_CSV" \
    --stations_csv "$STATIONS_CSV" \
    --stations_meta_csv "$META_CSV" \
    --year "$YEAR" --month "$((10#$m))" \
    --out_png "$PNG" \
    --lang LV \
    --label_stations \
    --smooth_sigma 0.6 \
    --quant_step 5 \
    --tick_step 25 \
    --vmax_round 25 \
    --robeza_shp "$ROBEZA_SHP" \
    --clip_to_lv \
    --logo_png "$LOGO_PNG" \
    --dpi 300
done

echo ""
echo "[DONE] all maps attempted."
