# -*- coding: ascii -*-
"""C7 pre-C6 verification (read-only)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))
from masterstar_gaia_accounting import _norm_cid  # noqa: E402

OUTDIR = Path(__file__).resolve().parent
B3 = ROOT / "dev" / "results" / "session_20260825_sel_ghost_01_b3"
LIVE = ROOT / "Archive" / "Drafts" / "draft_000516"
SETUP = "NoFilter_60_2"
R1 = B3 / "t3_r1"
R2 = B3 / "t3_r2"
WT_R1 = ROOT / ".worktrees" / "b1b_c592ecf"
HEADLINE_A = "1498000793739050368"
HEADLINE_B = "1485987151737107200"
LOST = "1500387696044768384"
ONLY_R0 = [
    "1496997386298488832",
    "1497063283984301696",
    "1498903316693061248",
    "1500387696044768384",
]
ONLY_R1 = ["1504304603139151872"]


def cid(v: object) -> str:
    return _norm_cid(v)


def load_ms(root: Path) -> pd.DataFrame:
    p = root / "platesolve" / SETUP / "masterstars_full_match.csv"
    df = pd.read_csv(p, dtype=str)
    df["_cid"] = df["catalog_id"].map(cid)
    return df


def row_fields(df: pd.DataFrame, star: str) -> dict:
    hit = df.loc[df["_cid"] == star]
    if hit.empty:
        return {"present": False}
    r = hit.iloc[0]
    out = {"present": True}
    for c in ("x", "y", "source_state", "vy_match_mode", "name", "vy_dao_pass", "peak_dao", "phot_g_mean_mag", "mag"):
        if c in hit.columns:
            out[c] = r[c]
    return out


def main() -> int:
    rec: dict = {}
    # C7-1: T3 R1 path contamination
    run_t3 = (B3 / "run_t3.py").read_text(encoding="utf-8", errors="replace")
    rec["c7_1"] = {
        "harness": "dev/results/session_20260825_sel_ghost_01_b3/run_t3.py",
        "r1_sys_path": "sys.path.insert(0, WT_R1/src_py) but does NOT clear ROOT/src_py",
        "worktree": str(WT_R1),
        "worktree_exists": WT_R1.is_dir(),
        "iter4_in_worktree": (WT_R1 / "src_py" / "dao_gaia_stage_01_iter4.py").is_file() if WT_R1.is_dir() else False,
        "iter4_in_head": (ROOT / "src_py" / "dao_gaia_stage_01_iter4.py").is_file(),
        "run_t3_insert_snippet": "sys.path.insert(0, str(src_py))" if "sys.path.insert(0, str(src_py))" in run_t3 else "missing",
    }
    # Isolated import probe
    import subprocess

    probe = (
        "import sys, json\n"
        f"wt = r'{WT_R1 / 'src_py'}'\n"
        f"head = r'{ROOT / 'src_py'}'\n"
        "sys.path = [wt]\n"
        "out = {'path0': sys.path[0], 'head_on_path': head in sys.path}\n"
        "try:\n"
        "    import dao_gaia_stage_01_iter4 as m\n"
        "    out['iter4_file'] = getattr(m, '__file__', None)\n"
        "    out['iter4_ok'] = True\n"
        "except Exception as e:\n"
        "    out['iter4_ok'] = False\n"
        "    out['iter4_err'] = type(e).__name__ + ': ' + str(e)[:200]\n"
        "try:\n"
        "    import pipeline as p\n"
        "    out['pipeline_file'] = getattr(p, '__file__', None)\n"
        "except Exception as e:\n"
        "    out['pipeline_err'] = type(e).__name__ + ': ' + str(e)[:200]\n"
        "print(json.dumps(out))\n"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(ROOT))
    rec["c7_1"]["isolated_probe_rc"] = r.returncode
    rec["c7_1"]["isolated_probe"] = r.stdout.strip() or r.stderr[-400:]

    probe2 = (
        "import sys, json\n"
        f"wt = r'{WT_R1 / 'src_py'}'\n"
        f"head = r'{ROOT / 'src_py'}'\n"
        "sys.path.insert(0, wt)\n"
        "if str(head) not in sys.path:\n"
        "    sys.path.append(head)\n"
        "out = {'head_on_path': any(str(head).replace('\\\\','/') in p.replace('\\\\','/') for p in sys.path)}\n"
        "try:\n"
        "    import dao_gaia_stage_01_iter4 as m\n"
        "    out['iter4_file'] = m.__file__\n"
        "    out['iter4_ok'] = True\n"
        "except Exception as e:\n"
        "    out['iter4_ok'] = False\n"
        "    out['iter4_err'] = str(e)[:200]\n"
        "print(json.dumps(out))\n"
    )
    r2 = subprocess.run([sys.executable, "-c", probe2], capture_output=True, text=True, cwd=str(ROOT))
    rec["c7_1"]["contaminated_probe"] = r2.stdout.strip() or r2.stderr[-400:]

    r1_meta = R1 / "platesolve" / SETUP / "photometry" / "pipeline_meta.json"
    if r1_meta.is_file():
        meta = json.loads(r1_meta.read_text(encoding="utf-8"))
        rec["c7_1"]["r1_n_ms"] = meta.get("n_masterstars") or meta.get("n_rows_out")
        rec["c7_1"]["r1_git"] = (meta.get("provenance") or {}).get("git_hash") if isinstance(meta.get("provenance"), dict) else meta.get("git_hash")

    # C7-2 --full copy list from source
    rec["c7_2"] = {
        "fn": "session_baseline_check._copy_frozen_anchor_inputs",
        "file": "dev/scripts/session_baseline_check.py",
        "lines": "578-608",
        "copies": [
            "platesolve/NoFilter_60_2/ entire tree EXCEPT photometry/, _hrd_cache/, *.pdf (includes MASTERSTAR.fits, masterstars_full_match.csv, WCS in FITS headers, field_catalog, ePSF products)",
            "detrended_aligned/lights/NoFilter_60_2/ (aligned lights FITS; these are preprocessed, not raw)",
            "cal_diag.json, draft_manifest.json, sat_diag.json from snapshot root -> work_root and platesolve setup",
        ],
        "then_deletes": "destination photometry/ is rmtree'd and recreated empty",
        "not_copied_as_inputs": "live photometry/ (proc CSVs, LCs) - rebuilt; raw lights; calibration masters from CalibrationLibrary are NOT in this copy (pre_calibrated/aligned path)",
        "blind_zone": "MASTERSTAR stack+CSV+ePSF, per-frame aligned FITS (WCS in headers), cal_diag, sat_diag, draft_manifest, field catalogs under platesolve",
    }

    r0 = load_ms(LIVE)
    r1 = load_ms(R1)
    r2 = load_ms(R2) if (R2 / "platesolve" / SETUP / "masterstars_full_match.csv").is_file() else pd.DataFrame()

    def same_ms(a: dict, b: dict, keys: tuple[str, ...]) -> bool:
        if not a.get("present") or not b.get("present"):
            return False
        for k in keys:
            if str(a.get(k)) != str(b.get(k)):
                return False
        return True

    rec["c7_3"] = {}
    rec["c7_3"][HEADLINE_A] = {
        "r0": row_fields(r0, HEADLINE_A),
        "r1": row_fields(r1, HEADLINE_A),
    }
    rec["c7_3"][HEADLINE_A]["ms_identical_xy_state_mode_name"] = same_ms(
        rec["c7_3"][HEADLINE_A]["r0"],
        rec["c7_3"][HEADLINE_A]["r1"],
        ("x", "y", "source_state", "vy_match_mode", "name"),
    )

    rec["c7_3"][HEADLINE_B] = {
        "r0": row_fields(r0, HEADLINE_B),
        "r1": row_fields(r1, HEADLINE_B),
        "ms_identical_xy_state_mode_name": False,
    }
    rec["c7_3"][HEADLINE_B]["ms_identical_xy_state_mode_name"] = same_ms(
        rec["c7_3"][HEADLINE_B]["r0"],
        rec["c7_3"][HEADLINE_B]["r1"],
        ("x", "y", "source_state", "vy_match_mode", "name"),
    )

    def phot(root: Path) -> Path:
        return root / "platesolve" / SETUP / "photometry"

    lc_a0 = pd.read_csv(phot(LIVE) / "lightcurves" / f"lightcurve_{HEADLINE_A}.csv")
    lc_a1 = pd.read_csv(phot(R1) / "lightcurves" / f"lightcurve_{HEADLINE_A}.csv")
    rec["c7_3"][HEADLINE_A]["row25_r0"] = {
        "flag": str(lc_a0["flag"].iloc[25]) if "flag" in lc_a0.columns else None,
        "mag": float(pd.to_numeric(lc_a0["mag_calib"], errors="coerce").iloc[25]),
        "bjd": float(pd.to_numeric(lc_a0["bjd"], errors="coerce").iloc[25]),
        "mag_inst": float(pd.to_numeric(lc_a0["mag_inst"], errors="coerce").iloc[25]),
    }
    rec["c7_3"][HEADLINE_A]["row25_r1"] = {
        "flag": str(lc_a1["flag"].iloc[25]) if "flag" in lc_a1.columns else None,
        "mag": str(lc_a1["mag_calib"].iloc[25]),
        "bjd": str(lc_a1["bjd"].iloc[25]),
        "mag_inst": str(lc_a1["mag_inst"].iloc[25]),
    }

    # Find which proc file corresponds to row 25 via filename if present
    for col in lc_a0.columns:
        if "file" in col.lower() or col in ("frame", "image", "csv"):
            rec["c7_3"][HEADLINE_A]["row25_r0"][col] = str(lc_a0[col].iloc[25])
            rec["c7_3"][HEADLINE_A]["row25_r1"][col] = str(lc_a1[col].iloc[25]) if col in lc_a1.columns else None

    lc_b0 = pd.read_csv(phot(LIVE) / "lightcurves" / f"lightcurve_{HEADLINE_B}.csv")
    lc_b1 = pd.read_csv(phot(R1) / "lightcurves" / f"lightcurve_{HEADLINE_B}.csv")
    m0 = pd.to_numeric(lc_b0["mag_calib"], errors="coerce")
    m1 = pd.to_numeric(lc_b1["mag_calib"], errors="coerce")
    rec["c7_3"][HEADLINE_B]["nfin_r0"] = int(m0.notna().sum())
    rec["c7_3"][HEADLINE_B]["nfin_r1"] = int(m1.notna().sum())
    rec["c7_3"][HEADLINE_B]["n_r0_only_finite_vs_r1_nan"] = int((m0.notna() & m1.isna()).sum()) if len(m0) == len(m1) else None

    # C7-4 lost target
    rec["c7_4"] = {
        "id": LOST,
        "r0": row_fields(r0, LOST),
        "r1": row_fields(r1, LOST),
        "r2": row_fields(r2, LOST) if not r2.empty else {"present": None},
    }
    # VSX / variable_targets
    vt = LIVE / "platesolve" / SETUP / "variable_targets.csv"
    if not vt.is_file():
        vt = phot(LIVE) / "variable_targets.csv"
    if vt.is_file():
        vtd = pd.read_csv(vt, dtype=str)
        idc = "catalog_id" if "catalog_id" in vtd.columns else "source_id"
        hit = vtd[vtd[idc].map(cid) == LOST] if idc in vtd.columns else pd.DataFrame()
        rec["c7_4"]["variable_targets"] = hit.head(1).to_dict(orient="records") if not hit.empty else []
        if "name" in vtd.columns:
            rec["c7_4"]["vt_name_cols"] = [c for c in vtd.columns if "vsx" in c.lower() or c in ("name", "star", "auid")][:12]

    census_r1 = R1 / "platesolve" / SETUP / "gaia_source_state_census.csv"
    census_r2 = R2 / "platesolve" / SETUP / "gaia_source_state_census.csv"
    for lab, p in (("r1", census_r1), ("r2", census_r2)):
        if p.is_file():
            cdf = pd.read_csv(p, dtype=str)
            idc = "catalog_id" if "catalog_id" in cdf.columns else cdf.columns[0]
            hit = cdf[cdf[idc].map(cid) == LOST]
            rec["c7_4"][f"census_{lab}"] = hit.head(1).to_dict(orient="records") if not hit.empty else []
            rec["c7_4"][f"census_{lab}_n"] = int(len(cdf))

    rec["c7_5"] = {"only_r0": {}, "only_r1": {}, "r2_vs_r1_set": {}}
    for star in ONLY_R0:
        rec["c7_5"]["only_r0"][star] = {
            "r0": row_fields(r0, star),
            "r1": row_fields(r1, star),
            "r2": row_fields(r2, star) if not r2.empty else {},
        }
    for star in ONLY_R1:
        rec["c7_5"]["only_r1"][star] = {
            "r0": row_fields(r0, star),
            "r1": row_fields(r1, star),
            "r2": row_fields(r2, star) if not r2.empty else {},
        }
    if not r2.empty:
        s0, s1, s2 = set(r0["_cid"]), set(r1["_cid"]), set(r2["_cid"])
        rec["c7_5"]["r2_vs_r1_set"] = {
            "n_r0": len(s0),
            "n_r1": len(s1),
            "n_r2": len(s2),
            "r2_minus_r1": sorted(s2 - s1)[:20],
            "r1_minus_r2": sorted(s1 - s2)[:20],
            "n_r2_minus_r1": len(s2 - s1),
            "n_r1_minus_r2": len(s1 - s2),
            "lost_in_r2": LOST in s2,
            "lost_in_r1": LOST in s1,
            "lost_in_r0": LOST in s0,
        }

    outp = OUTDIR / "c7_measure.json"
    outp.write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
    print("wrote", outp)
    print("C7-1 isolated", rec["c7_1"].get("isolated_probe"))
    print("C7-1 contaminated", rec["c7_1"].get("contaminated_probe"))
    print("C7-3 A identical", rec["c7_3"][HEADLINE_A]["ms_identical_xy_state_mode_name"])
    print("C7-4 r0 present", rec["c7_4"]["r0"].get("present"), "r1", rec["c7_4"]["r1"].get("present"), "r2", rec["c7_4"]["r2"].get("present"))
    print("C7-5 r2 vs r1", rec["c7_5"]["r2_vs_r1_set"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
