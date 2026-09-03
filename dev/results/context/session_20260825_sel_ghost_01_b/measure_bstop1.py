# -*- coding: ascii -*-
"""SEL-GHOST-01 B-STOP-1 predictions from the sandbox MASTERSTAR products."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))

from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from wcs_invertibility import post_match_pixel_sep  # noqa: E402

SESSION = Path(__file__).resolve().parent
LIVE520 = ROOT / "Archive" / "Drafts" / "draft_000520" / "platesolve" / "g_60_4"
LIVE516 = ROOT / "Archive" / "Drafts" / "draft_000516" / "platesolve" / "NoFilter_60_2"
SB520 = SESSION / "sandbox_520" / "g_60_4"
SB516 = SESSION / "sandbox_516" / "NoFilter_60_2"

GHOST_IDS = [
    "1112112413285008896",
    "1112115024625070720",
    "1111930718988511616",
    "1112119250872867200",
    "1112110042463052928",
    "1111931371821079552",
    "1111737823417422464",
    "1111922300852743808",
]
G12_IDS = [
    "1112113680298377344",
    "1111931371823539456",
    "1112113066119992064",
    "1111920204908702336",
    "1112110695298081664",
    "1112130898824233216",
    "1112121862213003648",
    "1111749157833870208",
    "1111754659689117952",
    "1112121067641532160",
    "1111955148762490496",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nonempty_cid(s: pd.Series) -> pd.Series:
    return s.map(normalize_gaia_source_id).astype(str).str.strip().replace({"nan": "", "None": ""})


def cid_set(df: pd.DataFrame) -> set[str]:
    c = nonempty_cid(df["catalog_id"])
    return {x for x in c.tolist() if x}


def latest_infolog(folder: Path) -> Path | None:
    logs = sorted(folder.glob("infolog_*.txt"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def grep_log(text: str, pat: str) -> list[str]:
    rx = re.compile(pat)
    return [ln for ln in text.splitlines() if rx.search(ln)]


def load_meta(platesolve: Path) -> dict:
    p = platesolve / "photometry" / "pipeline_meta.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


DETECTED_STATES = {"DETECTED_P1", "DETECTED_P2"}


def dao_mask(ms: pd.DataFrame) -> pd.Series:
    if "source_state" not in ms.columns:
        return pd.Series(True, index=ms.index)
    st = ms["source_state"].fillna("").astype(str)
    return ~st.str.contains("CATALOG", case=False, regex=False) & ~st.eq("CATALOG_ONLY")


def ghost_rows(ms: pd.DataFrame) -> list[dict]:
    cid = nonempty_cid(ms["catalog_id"])
    name = ms["name"].fillna("").astype(str) if "name" in ms.columns else pd.Series([""] * len(ms))
    out = []
    for gid in GHOST_IDS:
        hit = ms.loc[cid.eq(gid)]
        name_hit = ms.loc[name.eq(gid)]
        rec = hit.iloc[0] if len(hit) else (name_hit.iloc[0] if len(name_hit) else None)
        if rec is None:
            out.append({"catalog_id": gid, "has_catalog_id": False, "present": False})
            continue
        out.append(
            {
                "catalog_id": gid,
                "has_catalog_id": bool(len(hit)),
                "present": True,
                "name": rec.get("name"),
                "source_state": rec.get("source_state"),
                "source_type": rec.get("source_type"),
                "vy_match_mode": rec.get("vy_match_mode"),
                "vy_identity_gate": rec.get("vy_identity_gate"),
                "gaia_dao_resid_px": rec.get("gaia_dao_resid_px"),
                "x": rec.get("x"),
                "y": rec.get("y"),
            }
        )
    return out


def g12_verdicts(ms_csv: Path, ms_fits: Path, cone: Path, fwhm_px: float) -> dict:
    ms = pd.read_csv(ms_csv, dtype={"catalog_id": str, "name": str})
    cone_df = pd.read_csv(cone, dtype={"catalog_id": str, "source_id": str}, low_memory=False)
    with fits.open(ms_fits, memmap=False) as hdul:
        w = WCS(hdul[0].header)
    gmap: dict[str, tuple[float, float]] = {}
    cid_col = "catalog_id" if "catalog_id" in cone_df.columns else "source_id"
    ra_col = "ra_deg" if "ra_deg" in cone_df.columns else "ra"
    de_col = "dec_deg" if "dec_deg" in cone_df.columns else "dec"
    for _, r in cone_df.iterrows():
        k = normalize_gaia_source_id(str(r.get(cid_col, "")))
        if not k:
            continue
        try:
            ra = float(r.get(ra_col))
            de = float(r.get(de_col))
        except (TypeError, ValueError):
            continue
        if math.isfinite(ra) and math.isfinite(de):
            gmap[k] = (ra, de)
    ms = ms.copy()
    ms["_cid"] = nonempty_cid(ms["catalog_id"])
    by_cid = {str(r["_cid"]): r for _, r in ms.iterrows() if r["_cid"]}
    rows = []
    n_ok_warn = 0
    for gid in G12_IDS:
        rec = by_cid.get(gid)
        g = gmap.get(gid)
        if rec is None or g is None:
            rows.append({"catalog_id": gid, "present": False, "verdict": "missing"})
            continue
        x = float(pd.to_numeric(rec.get("x"), errors="coerce"))
        y = float(pd.to_numeric(rec.get("y"), errors="coerce"))
        verdict, dpx = post_match_pixel_sep(x, y, g[0], g[1], w, fwhm_px=float(fwhm_px))
        if verdict in ("ok", "warn"):
            n_ok_warn += 1
        st = str(rec.get("source_state") or "")
        detected = st in DETECTED_STATES
        rows.append(
            {
                "catalog_id": gid,
                "present": True,
                "verdict": verdict,
                "d_px": dpx,
                "source_state": st,
                "detected": detected,
                "vy_identity_gate": rec.get("vy_identity_gate"),
                "gaia_dao_resid_px": rec.get("gaia_dao_resid_px"),
                "name": rec.get("name"),
            }
        )
    det_rows = [r for r in rows if r.get("detected")]
    det_ok_warn = sum(1 for r in det_rows if r.get("verdict") in ("ok", "warn"))
    return {
        "n_ok_warn": n_ok_warn,
        "n": len(G12_IDS),
        "n_detected_with_cid": len(det_rows),
        "n_detected_ok_warn": det_ok_warn,
        "rows": rows,
    }


def honest_reported(ms: pd.DataFrame, n_dao: int, fwhm_px: float) -> dict:
    cid = nonempty_cid(ms["catalog_id"])
    n_rep = int((cid != "").sum())
    if "vy_identity_gate" in ms.columns:
        gate = ms["vy_identity_gate"].fillna("").astype(str).str.strip().str.lower()
        n_honest = int(((gate == "ok") | (gate == "warn")).sum())
    elif "gaia_dao_resid_px" in ms.columns:
        d = pd.to_numeric(ms["gaia_dao_resid_px"], errors="coerce")
        n_honest = int(((cid != "") & (d <= 3.0 * float(fwhm_px))).sum())
    else:
        n_honest = None
    return {
        "n_dao": int(n_dao),
        "n_reported": n_rep,
        "reported_rate": (n_rep / n_dao) if n_dao else None,
        "n_honest": n_honest,
        "honest_rate": (n_honest / n_dao) if (n_dao and n_honest is not None) else None,
    }


def main() -> int:
    infolog = latest_infolog(SESSION)
    text = infolog.read_text(encoding="utf-8", errors="replace") if infolog else ""
    ms520 = pd.read_csv(SB520 / "masterstars_full_match.csv", dtype={"catalog_id": str, "name": str})
    live520 = pd.read_csv(LIVE520 / "masterstars_full_match.csv", dtype={"catalog_id": str, "name": str})
    meta520 = load_meta(SB520)
    idg520 = meta520.get("identity_gate") or {}
    opt520 = meta520.get("optimizer_refit") or {}

    opt_entry_lines = grep_log(text, r"optimizer input columns")
    opt_entry_n = []
    for ln in opt_entry_lines:
        m = re.search(r"matched_nonempty=(\d+)/", ln)
        if m:
            opt_entry_n.append(int(m.group(1)))
    gate_lines = grep_log(text, r"post_match_identity_gate")
    widen_lines = grep_log(text, r"0\.95 widen iter|zhoda .* < 70")

    n_dao_snr = 685
    n_dao_520 = n_dao_snr
    fwhm_520 = float((idg520.get("fwhm_px") if isinstance(idg520, dict) else None) or 1.25)
    st520 = ms520["source_state"].fillna("").astype(str) if "source_state" in ms520.columns else pd.Series([""] * len(ms520))
    dao_rows = ms520.loc[st520.isin({"DETECTED_P1", "DETECTED_P2", "DAO_ONLY"})]
    rates_520_full = honest_reported(ms520, int(len(ms520)), fwhm_520)
    rates_520_dao = honest_reported(dao_rows, n_dao_520, fwhm_520)

    ghosts = ghost_rows(ms520)
    n_ghost_ids = int(sum(1 for g in ghosts if g.get("has_catalog_id")))

    g12 = g12_verdicts(SB520 / "masterstars_full_match.csv", SB520 / "MASTERSTAR.fits", SB520 / "field_catalog_cone.csv", fwhm_520)

    out_516: dict = {}
    if (SB516 / "masterstars_full_match.csv").is_file():
        ms516 = pd.read_csv(SB516 / "masterstars_full_match.csv", dtype={"catalog_id": str, "name": str})
        live516 = pd.read_csv(LIVE516 / "masterstars_full_match.csv", dtype={"catalog_id": str, "name": str})
        meta516 = load_meta(SB516)
        idg516 = meta516.get("identity_gate") or {}
        set_sb = cid_set(ms516)
        set_live = cid_set(live516)
        st_sb = ms516["source_state"].fillna("").astype(str) if "source_state" in ms516.columns else pd.Series([""] * len(ms516))
        st_lv = live516["source_state"].fillna("").astype(str) if "source_state" in live516.columns else pd.Series([""] * len(live516))
        det_sb = cid_set(ms516.loc[st_sb.isin(DETECTED_STATES)])
        det_lv = cid_set(live516.loc[st_lv.isin(DETECTED_STATES)])
        out_516 = {
            "identity_gate": idg516,
            "n_lock_geometry_reject": (idg516.get("n_lock_geometry_reject") if isinstance(idg516, dict) else None),
            "fail_count": (idg516.get("fail") if isinstance(idg516, dict) else None),
            "catalog_id_set_equal": set_sb == set_live,
            "n_sandbox": len(set_sb),
            "n_live": len(set_live),
            "only_sandbox": sorted(set_sb - set_live)[:20],
            "only_live": sorted(set_live - set_sb)[:20],
            "detected_catalog_id_set_equal": det_sb == det_lv,
            "n_detected_sandbox": len(det_sb),
            "n_detected_live": len(det_lv),
            "detected_only_sandbox": sorted(det_sb - det_lv)[:20],
            "detected_only_live": sorted(det_lv - det_sb)[:20],
            "match_sep_requested": meta516.get("match_sep_arcsec_requested"),
            "match_sep_effective": meta516.get("match_sep_arcsec_effective"),
            "wcs_gaia_pixel_refine_iters": meta516.get("wcs_gaia_pixel_refine_iters"),
            "optimizer_refit": meta516.get("optimizer_refit"),
            "widen_loop_fired": float(meta516.get("match_sep_arcsec_effective") or 0) > 12.0,
        }

    opt520_first = opt_entry_n[0] if opt_entry_n else None
    gate_out = idg520.get("n_matched_out") if isinstance(idg520, dict) else None
    pb1 = (opt520_first == gate_out) and (opt520_first != 347)
    pb4_all = int(g12.get("n_ok_warn") or 0) >= 10
    pb4_det = int(g12.get("n_detected_ok_warn") or 0) >= 10
    pb5_fail_quiet = (out_516.get("fail_count") == 0) if out_516 else False
    pb5_cid = bool(out_516.get("catalog_id_set_equal")) if out_516 else False
    pb5_widen = (out_516.get("widen_loop_fired") is False) if out_516 else False

    pred = {
        "verdicts": {
            "P-B1": bool(pb1),
            "P-B2": n_ghost_ids == 0,
            "P-B3_note": "optimizer skipped, n_pairs=7 < 50; no Grip; B4 not wired",
            "P-B4_all_rows_with_cid": pb4_all,
            "P-B4_detected_only": pb4_det,
            "P-B5_fail_quiet": pb5_fail_quiet,
            "P-B5_catalog_id_set": pb5_cid,
            "P-B5_widen_not_fired": pb5_widen,
            "B1e_516_lock_reject_zero": (out_516.get("n_lock_geometry_reject") == 0) if out_516 else False,
        },
        "P-B1_opt_entry_eq_gate_out": {
            "opt_entry_matched_nonempty_all": opt_entry_n,
            "opt_entry_520_first": opt520_first,
            "gate_n_matched_out": gate_out,
            "identity_gate": idg520,
            "match_sep_requested": meta520.get("match_sep_arcsec_requested"),
            "match_sep_effective": meta520.get("match_sep_arcsec_effective"),
            "wcs_gaia_pixel_refine_iters": meta520.get("wcs_gaia_pixel_refine_iters"),
        },
        "P-B2_ghosts_without_catalog_id": {
            "n_ghosts_with_id": n_ghost_ids,
            "pass": n_ghost_ids == 0,
            "rows": ghosts,
        },
        "P-B3_grip_rms": opt520,
        "P-B4_g12_ok_warn": g12,
        "P-B5_516": out_516,
        "honest_vs_reported_520_full_table": rates_520_full,
        "honest_vs_reported_520_dao": rates_520_dao,
        "widen_log_lines": widen_lines[:40],
        "gate_log_lines": gate_lines[:20],
        "opt_entry_log_lines": opt_entry_lines[:10],
        "infolog": str(infolog) if infolog else None,
        "live_sha_520_csv": sha256_file(LIVE520 / "masterstars_full_match.csv"),
        "live_sha_516_csv": sha256_file(LIVE516 / "masterstars_full_match.csv"),
        "live_sha_516_epsf": sha256_file(LIVE516 / "masterstar_epsf.fits"),
    }
    (SESSION / "bstop1_measure.json").write_text(json.dumps(pred, indent=2, default=str), encoding="ascii")
    print(json.dumps(pred, indent=2, default=str)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
