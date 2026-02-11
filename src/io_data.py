from __future__ import annotations
import pandas as pd

def read_grid_csv(path: str) -> pd.DataFrame:
    # 1) pamēģinām normāli ar komatu
    df = pd.read_csv(path, sep=",", encoding="utf-8", low_memory=False)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # 2) ja ielasījās kā 1 kolonna ar komatiem headerī -> pār-lasam
    if len(df.columns) == 1 and "," in df.columns[0]:
        df = pd.read_csv(path, sep=",", encoding="utf-8", low_memory=False, engine="python")
        df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # 3) ja joprojām 1 kolonna, mēģinam semikolu (dažreiz LV faili tādi)
    if len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False, engine="python")
        df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # 4) ja kolonnas nosaukums ir "ID,x,y,...", sadalam kolonnas
    if len(df.columns) == 1 and ("ID,x,y" in df.columns[0] or "lon,lat" in df.columns[0]):
        # tā ir tipiska pazīme, ka parsing noticis kā 1 kolonna
        df = pd.read_csv(path, sep=",", encoding="utf-8", low_memory=False, engine="python")
        df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # sanity
    print(f"[read_grid_csv] cols={len(df.columns)} first_cols={df.columns.tolist()[:8]}")
    return df

def read_stations_csv(path: str) -> pd.DataFrame:
    # Your stations file is ; separated.
    # If encoding ever breaks, change to cp1257.
    df = pd.read_csv(path, sep=";", decimal=".", encoding="utf-8", low_memory=False)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df