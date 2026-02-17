#!/usr/bin/env bash
set -euo pipefail

YEAR=2025

VENV_PY="/home/denissbokadenissboka/projects/gridpp_lab/.venv/bin/python"
REPORT_PY="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/src/loocv_html_report.py"
COMPARE_PY="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/src/compare_loocv.py"

GRID_CSV="/home/denissbokadenissboka/projects/gridpp_lab/1x1_LV_grid_2024_xy2.csv"
MONTHLY_CSV="/home/denissbokadenissboka/projects/gridpp_lab/HPRAB_monthly_1990_2024_15plusStations_WITH_COORDS.csv"
META_CSV="/home/denissbokadenissboka/projects/gridpp_lab/staciju_dati_norma3.csv"

ERA_DIR="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/data/era5"
OUT_BASE="/home/denissbokadenissboka/projects/gridpp_lab/gridpp_precip/out_${YEAR}_html_compare"

# --- config (tavs salīdzinājuma setups) ---
META_ID_COL="gh_id"
META_CONT_COL="cont_pr"
META_H_COL="elevation"

UK_RANGE_M="41500"
UK_NUGGET="0"

EPS="1"
LVAL="41500"
POBS_D="0.1"
MAX_POINTS="29"
CV_RADIUS="2000"     # <-- ja gribi stingru LOOCV, liec 60000

mkdir -p "$OUT_BASE"

echo "[RUN] Year=$YEAR OUT=$OUT_BASE"
echo "[CFG] eps=$EPS L=$LVAL pobs_d=$POBS_D max_points=$MAX_POINTS cv_radius=$CV_RADIUS | UK range=$UK_RANGE_M nugget=$UK_NUGGET"

# header for summary.csv
SUMMARY="$OUT_BASE/summary_${YEAR}.csv"
echo "year,month,stations,gridpp_mae,gridpp_rmse,gridpp_bias,uk_mae,uk_rmse,uk_bias,out_dir" > "$SUMMARY"

for m in $(seq -w 1 12); do
  ERA_NC="${ERA_DIR}/era5_tp_mm_month_${YEAR}_${m}.nc"
  if [[ ! -f "$ERA_NC" ]]; then
    echo "[SKIP] missing ERA file: $ERA_NC"
    continue
  fi

  OUT_DIR="${OUT_BASE}/${YEAR}_${m}"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "==================== ${YEAR}-${m} ===================="

  # Run report generator (it runs compare_loocv.py internally)
  "$VENV_PY" "$REPORT_PY" \
    --python "$VENV_PY" \
    --compare_py "$COMPARE_PY" \
    --grid_csv "$GRID_CSV" \
    --stations_monthly_csv "$MONTHLY_CSV" \
    --stations_meta_csv "$META_CSV" \
    --era_nc "$ERA_NC" \
    --year "$YEAR" --month "$((10#$m))" \
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
    --cv_radius "$CV_RADIUS" \
    --title "LOOCV ${YEAR}-${m}: GridPP vs UK"

  # Extract metrics from loocv_compare.csv (expects columns: obs,pred_gridpp,pred_uk)
  CSV="${OUT_DIR}/loocv_compare.csv"
  if [[ ! -f "$CSV" ]]; then
    echo "[WARN] missing $CSV"
    continue
  fi

  # python one-liner to compute metrics + station count
  line="$("$VENV_PY" - <<PY
import pandas as pd, numpy as np
df=pd.read_csv("$CSV")
obs=df["obs"].to_numpy(float)
pg=df["pred_gridpp"].to_numpy(float)
pu=df["pred_uk"].to_numpy(float)
m=np.isfinite(obs)&np.isfinite(pg)&np.isfinite(pu)
obs,pg,pu=obs[m],pg[m],pu[m]
eg=pg-obs
eu=pu-obs
def met(e):
    e=e[np.isfinite(e)]
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2))), float(np.mean(e))
g_mae,g_rmse,g_bias=met(eg)
u_mae,u_rmse,u_bias=met(eu)
print(f"{len(obs)},{g_mae:.3f},{g_rmse:.3f},{g_bias:.3f},{u_mae:.3f},{u_rmse:.3f},{u_bias:.3f}")
PY
)"

  stations="$(echo "$line" | cut -d, -f1)"
  g_mae="$(echo "$line" | cut -d, -f2)"
  g_rmse="$(echo "$line" | cut -d, -f3)"
  g_bias="$(echo "$line" | cut -d, -f4)"
  u_mae="$(echo "$line" | cut -d, -f5)"
  u_rmse="$(echo "$line" | cut -d, -f6)"
  u_bias="$(echo "$line" | cut -d, -f7)"

  echo "${YEAR},$((10#$m)),$stations,$g_mae,$g_rmse,$g_bias,$u_mae,$u_rmse,$u_bias,$OUT_DIR" >> "$SUMMARY"
done

# Build index.html from summary CSV
INDEX_HTML="$OUT_BASE/index.html"
"$VENV_PY" - <<PY
import pandas as pd
import html

summary_path="$SUMMARY"
out_html="$INDEX_HTML"

df=pd.read_csv(summary_path)
df=df.sort_values(["year","month"])

def row(r):
    m=int(r["month"])
    mm=f"{m:02d}"
    link=f"{r['out_dir']}/report.html".replace("$OUT_BASE/","")  # relative
    return f"<tr>" \
           f"<td>{r['year']}-{mm}</td>" \
           f"<td>{int(r['stations'])}</td>" \
           f"<td>{r['gridpp_mae']:.3f}</td><td>{r['gridpp_rmse']:.3f}</td><td>{r['gridpp_bias']:.3f}</td>" \
           f"<td>{r['uk_mae']:.3f}</td><td>{r['uk_rmse']:.3f}</td><td>{r['uk_bias']:.3f}</td>" \
           f"<td><a href='{html.escape(link)}'>report</a></td>" \
           f"</tr>"

rows="\n".join(row(r) for _,r in df.iterrows())

html_doc=f"""<!doctype html>
<html lang="lv">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>LOOCV year summary</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color:#111; }}
h1 {{ margin:0 0 8px 0; }}
.small {{ color:#555; font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #eee; padding: 8px 10px; font-size: 13px; }}
th {{ background: #fafafa; text-align:left; }}
</style>
</head>
<body>
<h1>LOOCV salīdzinājums {df['year'].iloc[0]}</h1>
<div class="small">GridPP log-ratio OI vs UK (R drift + OK reziduāli). Links uz katra mēneša report.html.</div>
<div style="height:14px"></div>
<table>
<thead>
<tr>
  <th>Mēnesis</th>
  <th>N</th>
  <th>GridPP MAE</th><th>GridPP RMSE</th><th>GridPP BIAS</th>
  <th>UK MAE</th><th>UK RMSE</th><th>UK BIAS</th>
  <th>HTML</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
"""
open(out_html,"w",encoding="utf-8").write(html_doc)
print("[OK] wrote:", out_html)
PY

echo ""
echo "[DONE]"
echo "Summary CSV: $SUMMARY"
echo "Index HTML : $INDEX_HTML"
echo "Open: file://$INDEX_HTML"
