"""
Analýza FWHM distribúcie pre výber optimálneho k-faktora.
Spusti: python scripts/_fwhm_analysis.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000288")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Načítaj OBS_FILES z CSV alebo DB
candidates = [
    DRAFT / "photometry" / "obs_files.csv",
    DRAFT / "obs_files.csv",
    DRAFT / "platesolve" / "obs_files.csv",
]

df = None
for p in candidates:
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        print(f"Načítané: {p} ({len(df)} riadkov)")
        break

if df is None:
    db_path = DRAFT / "vyvar.db"
    if db_path.is_file():
        m = re.search(r"draft_0*(\d+)$", DRAFT.name, flags=re.I)
        draft_id = int(m.group(1)) if m else None
        con = sqlite3.connect(str(db_path))
        try:
            cols = {str(r[1]).upper() for r in con.execute("PRAGMA table_info('OBS_FILES')").fetchall()}
            if draft_id is not None and "DRAFT_ID" in cols:
                df = pd.read_sql_query(
                    "SELECT * FROM OBS_FILES WHERE DRAFT_ID = ?",
                    con,
                    params=(draft_id,),
                )
                print(f"Načítané z DB: {db_path} (DRAFT_ID={draft_id}, {len(df)} riadkov)")
            else:
                df = pd.read_sql_query("SELECT * FROM OBS_FILES", con)
                print(f"Načítané z DB: {db_path} ({len(df)} riadkov, bez filtra DRAFT_ID)")
        finally:
            con.close()

if df is None:
    cfg_path = _ROOT / "config.json"
    if cfg_path.is_file():
        import json as _json

        cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
        db_main = Path(str(cfg.get("database_path") or "")).expanduser()
        m = re.search(r"draft_0*(\d+)$", DRAFT.name, flags=re.I)
        draft_id = int(m.group(1)) if m else None
        if db_main.is_file():
            con = sqlite3.connect(str(db_main))
            try:
                cols = {str(r[1]).upper() for r in con.execute("PRAGMA table_info('OBS_FILES')").fetchall()}
                if draft_id is not None and "DRAFT_ID" in cols:
                    df = pd.read_sql_query(
                        "SELECT * FROM OBS_FILES WHERE DRAFT_ID = ?",
                        con,
                        params=(draft_id,),
                    )
                    print(
                        f"Načítané z hlavnej DB (fallback): {db_main} "
                        f"(DRAFT_ID={draft_id}, {len(df)} riadkov)"
                    )
                else:
                    df = pd.read_sql_query("SELECT * FROM OBS_FILES", con)
                    print(f"Načítané z hlavnej DB (fallback): {db_main} ({len(df)} riadkov)")
            finally:
                con.close()

if df is None:
    print("ERROR: Nenašiel som zdroj FWHM dát")
    sys.exit(1)

# Nájdi FWHM stĺpec
fwhm_col = None
for c in ["fwhm_mean", "fwhm", "FWHM", "fwhm_median"]:
    if c in df.columns:
        fwhm_col = c
        break

if fwhm_col is None:
    print(f"Dostupné stĺpce: {list(df.columns)}")
    sys.exit(1)

fwhm = pd.to_numeric(df[fwhm_col], errors="coerce").dropna().values.astype(float)
fwhm = fwhm[np.isfinite(fwhm) & (fwhm > 0)]
print(f"\nFWHM stĺpec: '{fwhm_col}', N={len(fwhm)}")

if len(fwhm) == 0:
    print("ERROR: Po filtrácii nie sú žiadne kladné FWHM hodnoty")
    sys.exit(1)

# Základná štatistika
median_f = float(np.median(fwhm))
mad = float(np.median(np.abs(fwhm - median_f)))
sigma_mad = mad * 1.4826
std_f = float(np.std(fwhm))
p10 = float(np.percentile(fwhm, 10))
p90 = float(np.percentile(fwhm, 90))
p95 = float(np.percentile(fwhm, 95))
p99 = float(np.percentile(fwhm, 99))

print("\n--- Štatistika ---")
print(f"Min:    {fwhm.min():.3f} px")
print(f"P10:    {p10:.3f} px")
print(f"Median: {median_f:.3f} px")
print(f"P90:    {p90:.3f} px")
print(f"P95:    {p95:.3f} px")
print(f"P99:    {p99:.3f} px")
print(f"Max:    {fwhm.max():.3f} px")
print(f"Std:    {std_f:.3f} px")
print(f"MAD:    {mad:.3f} px  (sigma_MAD={sigma_mad:.3f})")

print("\n--- Auto-limit pre rôzne k ---")
print(f"{'k':>6}  {'limit':>8}  {'zachovaných':>12}  {'odrezaných':>10}")
print("-" * 45)
for k in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    limit = median_f + k * sigma_mad
    kept = int(np.sum(fwhm <= limit))
    cut = len(fwhm) - kept
    pct = 100.0 * kept / len(fwhm)
    print(f"{k:>6.1f}  {limit:>8.3f}  {kept:>6} ({pct:5.1f}%)  {cut:>10}")

print("\n--- Je_rejected stĺpec? ---")
rej_col = next((c for c in ["is_rejected", "IS_REJECTED"] if c in df.columns), None)
if rej_col:
    s = pd.to_numeric(df[rej_col], errors="coerce").fillna(0)
    n_rej = int((s != 0).sum())
    print(f"Manuálne/auto rejected: {n_rej}")
else:
    print("Stĺpec IS_REJECTED nenájdený")
