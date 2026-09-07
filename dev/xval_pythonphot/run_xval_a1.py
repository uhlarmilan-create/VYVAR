# -*- coding: ascii -*-
"""EPSF-XVAL-A1: PythonPhot vs VYVAR ePSF measurement harness (dev-only).

Positions: snapshot proc x,y. PSF stars: same pool as the gated ePSF
(`psf_photometry._epsf_prepare_stars` on live 516, read-only). Gain / RN:
`param_resolver.resolve_gain` / `resolve_read_noise` on the MASTERSTAR
header (same call as `psf_photometry.py:770-775`). Never hardcoded.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "dev"
SRC = REPO / "src_py"
for _p in (str(SRC), str(DEV)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.session_baseline_check import (  # noqa: E402
    SETUP,
    SNAPSHOT_NAME,
    _copy_frozen_anchor_inputs,
)
from xval_pythonphot.vendor import aper, getpsf, pkfit, pkfit_norecenter  # noqa: E402

TARGET_CID = "1498613634033133184"
CHECK_CID = "1497613731286514432"
ZEROPOINT = 25.0
G4_EXPECT = {
    "csv": "bfa24039",
    "fits": "13e77cf8",
    "epsf": "172f9540",
}
WORK = REPO / "tmp" / "session_20260907_epsfxval"
OUT = WORK / "out"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def g4_live_516() -> dict[str, Any]:
    live = REPO / "Archive" / "Drafts" / "draft_000516" / "platesolve" / SETUP
    rows = [
        ("csv", live / "masterstars_full_match.csv", G4_EXPECT["csv"]),
        ("fits", live / "MASTERSTAR.fits", G4_EXPECT["fits"]),
        ("epsf", live / "masterstar_epsf.fits", G4_EXPECT["epsf"]),
    ]
    out: dict[str, Any] = {}
    ok = True
    for key, path, prefix in rows:
        digest = _sha256_file(path) if path.is_file() else ""
        verdict = "PASS" if digest.startswith(prefix) else "FAIL"
        if verdict != "PASS":
            ok = False
        out[key] = {"path": str(path), "sha256": digest, "prefix": prefix, "verdict": verdict}
    out["pass"] = ok
    return out


def _flux_to_inst_mag(flux: np.ndarray) -> np.ndarray:
    f = np.asarray(flux, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def _readonly_equipment_intrinsics(equipment_id: int) -> tuple[float, float]:
    """Read EQUIPMENTS GAIN_ADU / READNOISE_E without opening VyvarDatabase (it writes)."""
    import sqlite3

    dbp = REPO / "vyvar.sqlite3"
    uri = f"file:{dbp.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        row = con.execute(
            "SELECT GAIN_ADU, READNOISE_E FROM EQUIPMENTS WHERE id = ?",
            (int(equipment_id),),
        ).fetchone()
    finally:
        con.close()
    if not row:
        raise RuntimeError(f"EQUIPMENTS id={equipment_id} missing")
    return float(row[0]), float(row[1])


def resolve_gain_rn(masterstar_fits: Path, photometry_dir: Path) -> dict[str, Any]:
    """Gain / RN from the same authorities the ePSF / PSF-LC chain uses. Never hardcoded.

    Gain: ``psf_internal_lc._load_gain_authority`` (live 516 sidecar
    ``gain_photon_transfer.json``, stamped on the PSF LC as
    ``gain_authority=g_pt=...``). RN: EQUIPMENTS.READNOISE_E via
    ``param_resolver.resolve_read_noise`` (header binning scale), same
    function as ``psf_photometry.py:770-775``. DB opened sqlite ``mode=ro``
    only; ``VyvarDatabase`` is not instantiated (it creates tables).
    """
    from param_resolver import resolve_read_noise
    from psf_internal_lc import _load_gain_authority

    with fits.open(masterstar_fits, memmap=True) as hd:
        hdr = hd[0].header
    gain, gsrc = _load_gain_authority(photometry_dir)
    eq_id = 1
    try:
        man = json.loads((photometry_dir.parent.parent.parent / "draft_manifest.json").read_text(encoding="utf-8"))
        eq_id = int(man.get("equipment_id") or 1)
    except Exception:  # noqa: BLE001
        eq_id = 1
    _g_db, rn_db = _readonly_equipment_intrinsics(eq_id)
    r = resolve_read_noise(hdr, db=None, equipment_id=eq_id, cfg=None, db_value=rn_db)
    rn = float(r.value) if r.value is not None else float("nan")
    if not math.isfinite(gain) or gain <= 0 or not math.isfinite(rn) or rn < 0:
        raise RuntimeError(
            f"gain/RN unresolved: gain={gain} src={gsrc} rn={rn} src={r.source}"
        )
    return {
        "gain": float(gain),
        "gain_source": str(gsrc),
        "read_noise": rn,
        "read_noise_source": str(r.source),
        "equipment_id": eq_id,
        "cite": (
            "gain: psf_internal_lc._load_gain_authority (PSF LC g_pt sidecar); "
            "RN: param_resolver.resolve_read_noise + EQUIPMENTS.READNOISE_E "
            "(psf_photometry.py:770-775)"
        ),
    }


def reconstruct_psf_star_ids(live_ps: Path) -> tuple[list[str], dict[str, Any]]:
    """PSF-star pool from the G-EPSF gate census of the same 67-star model.

    Live ``build_epsf_science_set`` is no longer that 2026-08-22 set
    (today n_after_science_scope=3). Meta does not list IDs. The retained
    gate artifact ``r3_build_stars_516.csv`` is the 67-star list from
    EPSF-VALID-02 R1/R4 (same n_stars_used / created_utc as live meta).
    """
    census = (
        REPO / "dev" / "results" / "context" / "session_20260822_epsf_valid_02_r1r4"
        / "r3_build_stars_516.csv"
    )
    meta_path = live_ps / "masterstar_epsf_meta.json"
    epsf_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    fwhm_px = float(epsf_meta.get("fwhm_px") or 3.3014)
    df = pd.read_csv(census, dtype={"catalog_id": str})
    ids = [str(x).strip() for x in df["catalog_id"].tolist() if str(x).strip()]
    meta = {
        "n_ids": len(ids),
        "fwhm_px": float(fwhm_px),
        "source": str(census.relative_to(REPO)).replace("\\", "/"),
        "expected_n_stars_used": int(epsf_meta.get("n_stars_used") or 0),
        "created_utc": str(epsf_meta.get("created_utc") or ""),
    }
    return ids, meta


def list_psf_lc_ids(lc_dir: Path) -> list[str]:
    ids: list[str] = []
    for p in sorted(lc_dir.glob("lightcurve_*_psf.csv")):
        stem = p.stem
        cid = stem[len("lightcurve_") : -len("_psf")]
        if cid:
            ids.append(cid)
    return ids


def parse_lc_header(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if "=" in line:
                k, v = line[1:].split("=", 1)
                out[k.strip()] = v.strip()
    return out


def load_vyvar_psf_lc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df["source_file"] = df["source_file"].astype(str).str.strip()
    return df


def _aper1d(vals: Any, n: int) -> np.ndarray:
    a = np.asarray(vals, dtype=np.float64).reshape(-1)
    if a.size < n:
        pad = np.full(n - a.size, np.nan)
        a = np.concatenate([a, pad])
    return a[:n]


def measure_frame(
    image: np.ndarray,
    *,
    psf_xy: list[tuple[str, float, float]],
    star_xy: list[tuple[str, float, float]],
    gain: float,
    rn: float,
    fitrad: float,
    psfrad: float,
    psf_fits: Path,
    recenter: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """getpsf + pkfit. Returns {cid: meas} and a status dict (never silent)."""
    status: dict[str, Any] = {"ok": False, "reason": "", "n_psf_used": 0}
    meas: dict[str, dict[str, float]] = {}
    import contextlib
    import io

    if not psf_xy:
        status["reason"] = "no_psf_stars_in_frame"
        return meas, status
    if not star_xy:
        status["reason"] = "no_lc_stars_in_frame"
        return meas, status

    px = np.array([p[1] for p in psf_xy], dtype=np.float64)
    py = np.array([p[2] for p in psf_xy], dtype=np.float64)
    _devnull = io.StringIO()
    try:
        with contextlib.redirect_stdout(_devnull):
            mag, _me, _fl, _fe, skyv, _se, _bf, _out = aper.aper(
            image,
            px,
            py,
            phpadu=gain,
            apr=float(max(fitrad, 3.0)),
            zeropoint=ZEROPOINT,
            skyrad=[40.0, 50.0],
            badpix=[-12000.0, 60000.0],
            exact=True,
        )
    except Exception as exc:  # noqa: BLE001
        status["reason"] = f"aper_psf:{type(exc).__name__}:{exc}"
        return meas, status
    npsf = len(psf_xy)
    mag1 = _aper1d(mag, npsf)
    sky1 = _aper1d(skyv, npsf)
    good = np.isfinite(mag1) & np.isfinite(sky1) & np.isfinite(px) & np.isfinite(py)
    if int(good.sum()) < 5:
        status["reason"] = f"aper_psf_too_few:{int(good.sum())}"
        return meas, status
    idpsf = np.where(good)[0]
    psf_fits.parent.mkdir(parents=True, exist_ok=True)
    if psf_fits.exists():
        psf_fits.unlink()
    try:
        with contextlib.redirect_stdout(_devnull):
            gauss, psf, psfmag = getpsf.getpsf(
                image,
                px,
                py,
                mag1,
                sky1,
                rn,
                gain,
                idpsf,
                float(psfrad),
                float(fitrad),
                str(psf_fits),
                zeropoint=ZEROPOINT,
                verbose=False,
            )
    except Exception as exc:  # noqa: BLE001
        status["reason"] = f"getpsf:{type(exc).__name__}:{exc}"
        return meas, status
    if gauss is None or psf is None or not np.all(np.isfinite(np.asarray(gauss, dtype=np.float64))):
        status["reason"] = "getpsf_nonfinite_gauss"
        return meas, status
    status["n_psf_used"] = int(getattr(fits.getheader(psf_fits), "get", lambda *_: 0)("NSTARS") or 0)
    try:
        status["n_psf_used"] = int(fits.getheader(psf_fits).get("NSTARS") or 0)
    except Exception:  # noqa: BLE001
        pass

    sx = np.array([p[1] for p in star_xy], dtype=np.float64)
    sy = np.array([p[2] for p in star_xy], dtype=np.float64)
    try:
        with contextlib.redirect_stdout(_devnull):
            smag, _me, _fl, _fe, ssky, _se, _bf, _out = aper.aper(
            image,
            sx,
            sy,
            phpadu=gain,
            apr=float(max(fitrad, 3.0)),
            zeropoint=ZEROPOINT,
            skyrad=[40.0, 50.0],
            badpix=[-12000.0, 60000.0],
            exact=True,
        )
    except Exception as exc:  # noqa: BLE001
        status["reason"] = f"aper_lc:{type(exc).__name__}:{exc}"
        return meas, status
    nstar = len(star_xy)
    smag1 = _aper1d(smag, nstar)
    ssky1 = _aper1d(ssky, nstar)
    if recenter:
        pk = pkfit.pkfit_class(image, gauss, psf, rn, gain)
    else:
        pk = pkfit_norecenter.pkfit_class(image, gauss, psf, rn, gain)

    for i, (cid, x, y) in enumerate(star_xy):
        rec: dict[str, float] = {
            "x": float(x),
            "y": float(y),
            "x_fit": float(x),
            "y_fit": float(y),
            "sky": float(ssky1[i]) if i < len(ssky1) else float("nan"),
            "ap_mag": float(smag1[i]) if i < len(smag1) else float("nan"),
            "scale": float("nan"),
            "flux": float("nan"),
            "errmag": float("nan"),
            "chi": float("nan"),
            "sharp": float("nan"),
            "niter": float("nan"),
            "fail": 1.0,
        }
        sky_i = rec["sky"]
        apm = rec["ap_mag"]
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(sky_i) and math.isfinite(apm)):
            rec["fail_reason"] = 1.0  # marker; string stored separately
            meas[cid] = rec
            meas[cid]["_reason"] = "aper_nan"  # type: ignore[assignment]
            continue
        scale0 = float(10.0 ** (-0.4 * (apm - float(psfmag))))
        if not math.isfinite(scale0) or scale0 <= 0:
            scale0 = 1.0
        try:
            if recenter:
                errmag, chi, sharp, niter, scale, xnew, ynew = pk.pkfit(
                    scale0, x, y, sky_i, float(fitrad), xyout=True, recenter=True
                )
                rec["x_fit"] = float(xnew)
                rec["y_fit"] = float(ynew)
            else:
                errmag, chi, sharp, niter, scale = pk.pkfit_norecenter(
                    scale0, x, y, sky_i, float(fitrad)
                )
        except Exception as exc:  # noqa: BLE001
            rec["_reason"] = f"pkfit:{type(exc).__name__}:{exc}"  # type: ignore[assignment]
            meas[cid] = rec
            continue
        rec["scale"] = float(scale) if np.isfinite(scale) else float("nan")
        rec["errmag"] = float(errmag) if np.isfinite(errmag) else float("nan")
        rec["chi"] = float(chi) if np.isfinite(chi) else float("nan")
        rec["sharp"] = float(sharp) if np.isfinite(sharp) else float("nan")
        rec["niter"] = float(niter)
        if int(niter) == -1:
            rec["_reason"] = "pkfit_singular"  # type: ignore[assignment]
        elif not math.isfinite(rec["scale"]):
            rec["_reason"] = "pkfit_nan_scale"  # type: ignore[assignment]
        else:
            rec["flux"] = float(rec["scale"] * 10.0 ** (0.4 * (ZEROPOINT - float(psfmag))))
            rec["fail"] = 0.0
            rec["_reason"] = ""  # type: ignore[assignment]
        meas[cid] = rec
    status["ok"] = True
    status["reason"] = ""
    status["psfmag"] = float(psfmag)
    return meas, status


def _proc_rows(proc: Path, wanted: set[str]) -> dict[str, dict[str, float]]:
    if not proc.is_file():
        return {}
    df = pd.read_csv(proc, usecols=lambda c: c in {"catalog_id", "x", "y", "psf_flux", "psf_fit_ok", "phot_g_mean_mag", "catalog_mag", "mag"})
    if "catalog_id" not in df.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        cid = str(row.get("catalog_id") or "").strip()
        if cid not in wanted:
            continue
        rec = {
            "x": float(pd.to_numeric(row.get("x"), errors="coerce")),
            "y": float(pd.to_numeric(row.get("y"), errors="coerce")),
        }
        if "psf_flux" in df.columns:
            rec["psf_flux"] = float(pd.to_numeric(row.get("psf_flux"), errors="coerce"))
        mag = float("nan")
        for col in ("phot_g_mean_mag", "catalog_mag", "mag"):
            if col in df.columns:
                mag = float(pd.to_numeric(row.get(col), errors="coerce"))
                if math.isfinite(mag):
                    break
        rec["catalog_mag"] = mag
        out[cid] = rec
    return out


def rms_after_median(a: np.ndarray) -> float:
    v = np.asarray(a, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    v = v - float(np.median(v))
    return float(np.sqrt(np.mean(v * v)))


def apply_reading(rms_mmag: float) -> str:
    if not math.isfinite(rms_mmag):
        return "R-X3: RMS(diff) > 10.0 mmag -> STOP, root-cause before any validation claim. (RMS non-finite)"
    if rms_mmag <= 3.0:
        return "R-X1: both RMS(diff) <= 3.0 mmag -> AIJ-class implementation agreement of the ePSF chain."
    if rms_mmag <= 10.0:
        return "R-X2: 3.0 < RMS(diff) <= 10.0 mmag -> agreement with caveats; report the top per-epoch contributors and their frames."
    return "R-X3: RMS(diff) > 10.0 mmag -> STOP, root-cause before any validation claim."


def rebuild_delta(
    target_flux: np.ndarray,
    comp_flux: dict[str, np.ndarray],
    weights: dict[str, float],
    comp_ids: list[str],
) -> np.ndarray:
    """AIJ tot_C_cnts ensemble: delta = m_t - (-2.5 log10(sum F_c)).

    Same combination as ``photometry_lightcurve.ensemble_normalize``
    (flux sum; weights do not enter ens_med). Full pinned membership or
    NaN (INV-PSF-LC-PIN-01). Implemented here to avoid the
    photometry_lightcurve / photometry_core circular import from a
    standalone harness.
    """
    _ = weights
    ft = np.asarray(target_flux, dtype=np.float64)
    n = len(ft)
    delta = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not (math.isfinite(float(ft[i])) and float(ft[i]) > 0):
            continue
        csum = 0.0
        missing = False
        for cid in comp_ids:
            arr = comp_flux.get(cid)
            if arr is None or i >= len(arr):
                missing = True
                break
            fv = float(arr[i])
            if not (math.isfinite(fv) and fv > 0):
                missing = True
                break
            csum += fv
        if missing or csum <= 0:
            continue
        delta[i] = -2.5 * math.log10(float(ft[i]) / csum)
    return delta


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="EPSF-XVAL-A1 PythonPhot harness")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all snapshot frames")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = REPO / "Archive" / "Drafts" / SNAPSHOT_NAME
    live_ps = REPO / "Archive" / "Drafts" / "draft_000516" / "platesolve" / SETUP
    live_lc = live_ps / "photometry" / "lightcurves"
    live_phot = live_ps / "photometry"
    g4_before = g4_live_516()

    print("[xval] copy frozen era04 snapshot into tmp sandbox")
    ps_dst, lights_dst = _copy_frozen_anchor_inputs(snapshot, WORK / "sandbox")
    frames = sorted(p for p in lights_dst.glob("*.fits") if p.name.upper() != "MASTERSTAR.FITS")
    if int(args.max_frames) > 0:
        frames = frames[: int(args.max_frames)]
    print(f"[xval] sandbox frames={len(frames)} lights={lights_dst}")

    gain_info = resolve_gain_rn(live_ps / "MASTERSTAR.fits", live_phot)
    print(f"[xval] gain={gain_info['gain']} src={gain_info['gain_source']} "
          f"rn={gain_info['read_noise']} src={gain_info['read_noise_source']}")

    psf_ids, psf_meta = reconstruct_psf_star_ids(live_ps)
    print(f"[xval] reconstructed ePSF pool n={len(psf_ids)} meta={psf_meta}")

    lc_ids = list_psf_lc_ids(live_lc)
    measure_ids = sorted(set(lc_ids) | {TARGET_CID, CHECK_CID})
    from psf_internal_lc import resolve_ensemble_ids

    ens_ids, ens_w, ens_src = resolve_ensemble_ids(TARGET_CID, live_phot)
    measure_ids = sorted(set(measure_ids) | set(ens_ids))
    wanted = set(measure_ids) | set(psf_ids)
    print(f"[xval] LC stars={len(lc_ids)} measure={len(measure_ids)} ens={ens_ids} src={ens_src}")

    vyvar_lcs: dict[str, pd.DataFrame] = {}
    catmag: dict[str, float] = {}
    for cid in lc_ids:
        p = live_lc / f"lightcurve_{cid}_psf.csv"
        vyvar_lcs[cid] = load_vyvar_psf_lc(p)

    tgt_lc = vyvar_lcs[TARGET_CID]
    epoch_files = [str(s).strip() for s in tgt_lc["source_file"].tolist()]
    fwhm_px = float(psf_meta.get("fwhm_px") or 3.3)
    fitrad = float(fwhm_px)
    psfrad = float(max(fitrad + 1.0, 8.0))

    # Live proc (read-only) supplies VYVAR psf_flux for the check (no check PSF LC file).
    live_lights = REPO / "Archive" / "Drafts" / "draft_000516" / "detrended_aligned" / "lights" / SETUP

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    recenter_rows: list[dict[str, Any]] = []
    recenter_names = []
    if frames:
        picks = [0, len(frames) // 4, len(frames) // 2, (3 * len(frames)) // 4, len(frames) - 1]
        recenter_names = [frames[i].name for i in picks]

    for i_fr, fp in enumerate(frames):
        t_fr = time.perf_counter()
        stem = fp.stem  # BO_CVn_Light_001
        proc_name = f"proc_{stem}.csv"
        snap_proc = lights_dst / proc_name
        if not snap_proc.is_file():
            failures.append({"frame": fp.name, "stage": "proc", "reason": "snapshot_proc_missing", "catalog_id": ""})
            print(f"[xval] FAIL {fp.name} snapshot_proc_missing")
            continue
        xy = _proc_rows(snap_proc, wanted)
        psf_xy = [(cid, xy[cid]["x"], xy[cid]["y"]) for cid in psf_ids if cid in xy
                  and math.isfinite(xy[cid]["x"]) and math.isfinite(xy[cid]["y"])]
        star_xy = [(cid, xy[cid]["x"], xy[cid]["y"]) for cid in measure_ids if cid in xy
                   and math.isfinite(xy[cid]["x"]) and math.isfinite(xy[cid]["y"])]
        for cid in measure_ids:
            if cid not in xy:
                failures.append({"frame": fp.name, "stage": "position", "reason": "missing_in_proc", "catalog_id": cid})
            if cid in xy:
                cm = xy[cid].get("catalog_mag", float("nan"))
                if cid not in catmag and math.isfinite(cm):
                    catmag[cid] = cm
        try:
            image = np.asarray(fits.getdata(fp), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            failures.append({"frame": fp.name, "stage": "fits", "reason": f"{type(exc).__name__}:{exc}", "catalog_id": ""})
            continue
        psf_fits = WORK / "psf" / f"{stem}_pp.fits"
        meas, status = measure_frame(
            image,
            psf_xy=psf_xy,
            star_xy=star_xy,
            gain=float(gain_info["gain"]),
            rn=float(gain_info["read_noise"]),
            fitrad=fitrad,
            psfrad=psfrad,
            psf_fits=psf_fits,
            recenter=False,
        )
        if not status["ok"]:
            failures.append({"frame": fp.name, "stage": "getpsf", "reason": status["reason"], "catalog_id": ""})
            print(f"[xval] FAIL {fp.name} {status['reason']}")
            continue
        live_proc = live_lights / proc_name
        live_xy = _proc_rows(live_proc, wanted) if live_proc.is_file() else {}
        for cid, rec in meas.items():
            reason = str(rec.get("_reason") or "")
            if rec.get("fail", 1.0) >= 0.5:
                failures.append({"frame": fp.name, "stage": "pkfit", "reason": reason or "pkfit_fail", "catalog_id": cid})
            rows.append({
                "source_file": proc_name,
                "fits": fp.name,
                "catalog_id": cid,
                "pp_flux": rec.get("flux"),
                "pp_scale": rec.get("scale"),
                "pp_chi": rec.get("chi"),
                "pp_niter": rec.get("niter"),
                "pp_x": rec.get("x"),
                "pp_y": rec.get("y"),
                "vyvar_psf_flux_live_proc": live_xy.get(cid, {}).get("psf_flux", float("nan")),
                "fail": rec.get("fail"),
                "fail_reason": reason,
            })
        if fp.name in recenter_names:
            meas_r, st_r = measure_frame(
                image,
                psf_xy=psf_xy,
                star_xy=star_xy,
                gain=float(gain_info["gain"]),
                rn=float(gain_info["read_noise"]),
                fitrad=fitrad,
                psfrad=psfrad,
                psf_fits=WORK / "psf" / f"{stem}_pp_recenter.fits",
                recenter=True,
            )
            if not st_r["ok"]:
                failures.append({"frame": fp.name, "stage": "pkfit_recenter", "reason": st_r["reason"], "catalog_id": ""})
            else:
                for cid, rec in meas_r.items():
                    base = meas.get(cid, {})
                    recenter_rows.append({
                        "fits": fp.name,
                        "catalog_id": cid,
                        "dx": float(rec.get("x_fit", np.nan)) - float(base.get("x", np.nan)),
                        "dy": float(rec.get("y_fit", np.nan)) - float(base.get("y", np.nan)),
                        "flux_norecenter": base.get("flux"),
                        "flux_recenter": rec.get("flux"),
                        "fail_re": rec.get("fail"),
                    })
        dt = time.perf_counter() - t_fr
        print(f"[xval] {i_fr+1}/{len(frames)} {fp.name} psf_stars={len(psf_xy)} "
              f"n_psf_used={status.get('n_psf_used')} dt={dt:.1f}s")

    meas_df = pd.DataFrame(rows)
    if meas_df.empty:
        meas_df = pd.DataFrame(columns=["source_file", "fits", "catalog_id", "pp_flux", "pp_scale",
                                        "pp_chi", "pp_niter", "pp_x", "pp_y",
                                        "vyvar_psf_flux_live_proc", "fail", "fail_reason"])
    meas_df.to_csv(OUT / "pp_fluxes.csv", index=False)
    fail_df = pd.DataFrame(failures)
    if fail_df.empty:
        fail_df = pd.DataFrame(columns=["frame", "stage", "reason", "catalog_id"])
    fail_df.to_csv(OUT / "failures.csv", index=False)

    # M1: LC star set, VYVAR from PSF LC, PP from harness
    m1_rows: list[dict[str, Any]] = []
    for cid in lc_ids:
        lc = vyvar_lcs[cid]
        sub = meas_df[(meas_df["catalog_id"] == cid) & (meas_df["fail"] < 0.5)]
        merged = lc.merge(sub[["source_file", "pp_flux"]], on="source_file", how="left")
        m_v = _flux_to_inst_mag(pd.to_numeric(merged["psf_flux"], errors="coerce").to_numpy())
        m_p = _flux_to_inst_mag(pd.to_numeric(merged["pp_flux"], errors="coerce").to_numpy())
        d = m_v - m_p
        ok = np.isfinite(d)
        n_ok = int(ok.sum())
        n_fail = int((~ok).sum())
        rms = rms_after_median(d)
        m1_rows.append({
            "catalog_id": cid,
            "catalog_mag": catmag.get(cid, float("nan")),
            "n_epochs_lc": int(len(lc)),
            "n_ok": n_ok,
            "n_nan_or_fail": n_fail,
            "rms_mag": rms,
            "rms_mmag": rms * 1000.0 if math.isfinite(rms) else float("nan"),
            "median_d_mag": float(np.nanmedian(d)) if n_ok else float("nan"),
        })
    m1 = pd.DataFrame(m1_rows).sort_values(["catalog_mag", "catalog_id"], na_position="last")
    m1.to_csv(OUT / "m1_per_star.csv", index=False)

    # M2 target
    tgt_sub = meas_df[(meas_df["catalog_id"] == TARGET_CID)]
    tgt_m = tgt_lc.merge(tgt_sub[["source_file", "pp_flux"]], on="source_file", how="left")
    n_ep = len(tgt_m)
    pp_comp = {}
    vy_comp = {}
    for cid in ens_ids:
        if cid not in vyvar_lcs:
            # ensemble member may not have its own PSF LC; use live proc
            arr_pp = np.full(n_ep, np.nan)
            arr_vy = np.full(n_ep, np.nan)
            for i, sf in enumerate(tgt_m["source_file"].tolist()):
                hit = meas_df[(meas_df["catalog_id"] == cid) & (meas_df["source_file"] == sf)]
                if not hit.empty and float(hit.iloc[0]["fail"]) < 0.5:
                    arr_pp[i] = float(hit.iloc[0]["pp_flux"])
                arr_vy[i] = float(hit.iloc[0]["vyvar_psf_flux_live_proc"]) if not hit.empty else float("nan")
            pp_comp[cid] = arr_pp
            vy_comp[cid] = arr_vy
            continue
        c_lc = vyvar_lcs[cid]
        c_sub = meas_df[meas_df["catalog_id"] == cid]
        cm = tgt_m[["source_file"]].merge(c_lc[["source_file", "psf_flux"]], on="source_file", how="left")
        cm = cm.merge(c_sub[["source_file", "pp_flux"]], on="source_file", how="left")
        pp_comp[cid] = pd.to_numeric(cm["pp_flux"], errors="coerce").to_numpy()
        vy_comp[cid] = pd.to_numeric(cm["psf_flux"], errors="coerce").to_numpy()

    pp_tgt = pd.to_numeric(tgt_m["pp_flux"], errors="coerce").to_numpy()
    pp_delta = rebuild_delta(pp_tgt, pp_comp, ens_w, ens_ids)
    vy_delta = pd.to_numeric(tgt_m["psf_delta_mag"], errors="coerce").to_numpy()
    d_tgt = vy_delta - pp_delta
    rms_tgt = rms_after_median(d_tgt)
    # top contributors
    resid = np.asarray(d_tgt, dtype=np.float64)
    if np.isfinite(resid).any():
        resid = resid - float(np.nanmedian(resid[np.isfinite(resid)]))
    top_idx = np.argsort(-np.abs(np.where(np.isfinite(resid), resid, 0.0)))[:8]

    # M2 check: no dedicated PSF LC; rebuild both sides from fluxes + target ensemble
    chk_pp = np.full(n_ep, np.nan)
    chk_vy = np.full(n_ep, np.nan)
    for i, sf in enumerate(tgt_m["source_file"].tolist()):
        hit = meas_df[(meas_df["catalog_id"] == CHECK_CID) & (meas_df["source_file"] == sf)]
        if hit.empty:
            continue
        if float(hit.iloc[0]["fail"]) < 0.5:
            chk_pp[i] = float(hit.iloc[0]["pp_flux"])
        chk_vy[i] = float(hit.iloc[0]["vyvar_psf_flux_live_proc"])
    pp_chk_delta = rebuild_delta(chk_pp, pp_comp, ens_w, ens_ids)
    vy_chk_delta = rebuild_delta(chk_vy, vy_comp, ens_w, ens_ids)
    d_chk = vy_chk_delta - pp_chk_delta
    rms_chk = rms_after_median(d_chk)
    resid_c = np.asarray(d_chk, dtype=np.float64)
    if np.isfinite(resid_c).any():
        resid_c = resid_c - float(np.nanmedian(resid_c[np.isfinite(resid_c)]))
    top_chk = np.argsort(-np.abs(np.where(np.isfinite(resid_c), resid_c, 0.0)))[:8]

    def _top_table(idx: np.ndarray, resid_arr: np.ndarray, src: pd.Series) -> list[dict[str, Any]]:
        out = []
        for j in idx:
            if j >= len(resid_arr) or not math.isfinite(float(resid_arr[j])):
                continue
            out.append({
                "source_file": str(src.iloc[j]),
                "resid_mag": float(resid_arr[j]),
                "resid_mmag": float(resid_arr[j]) * 1000.0,
            })
        return out

    rec_note: dict[str, Any] = {}
    if recenter_rows:
        rdf = pd.DataFrame(recenter_rows)
        rdf.to_csv(OUT / "recenter_sensitivity.csv", index=False)
        ok = rdf[np.isfinite(rdf["dx"]) & np.isfinite(rdf["dy"])]
        flux_ok = ok[np.isfinite(ok["flux_norecenter"]) & np.isfinite(ok["flux_recenter"])]
        dflux = flux_ok["flux_recenter"] - flux_ok["flux_norecenter"]
        dmag = -2.5 * np.log10(np.clip(flux_ok["flux_recenter"].to_numpy(), 1e-12, None) /
                               np.clip(flux_ok["flux_norecenter"].to_numpy(), 1e-12, None))
        rec_note = {
            "n_frames": int(rdf["fits"].nunique()),
            "frames": sorted(rdf["fits"].unique().tolist()),
            "median_abs_dx_px": float(np.median(np.abs(ok["dx"]))) if not ok.empty else float("nan"),
            "median_abs_dy_px": float(np.median(np.abs(ok["dy"]))) if not ok.empty else float("nan"),
            "median_dflux": float(np.median(dflux)) if not flux_ok.empty else float("nan"),
            "median_dmag_mmag": float(np.median(dmag) * 1000.0) if not flux_ok.empty else float("nan"),
            "n_pairs": int(len(ok)),
        }

    both_ok = math.isfinite(rms_tgt) and math.isfinite(rms_chk)
    rms_tgt_mm = rms_tgt * 1000.0 if math.isfinite(rms_tgt) else float("nan")
    rms_chk_mm = rms_chk * 1000.0 if math.isfinite(rms_chk) else float("nan")
    worst = max([x for x in (rms_tgt_mm, rms_chk_mm) if math.isfinite(x)], default=float("nan"))
    if both_ok:
        reading = apply_reading(worst)
    else:
        reading = apply_reading(float("inf"))

    g4_after = g4_live_516()
    payload = {
        "task": "EPSF-XVAL-A1-PYTHONPHOT-01",
        "route": (
            "sandbox copy of draft_000516_snapshot_era04_20260826 aligned lights; "
            "G-EPSF products (PSF LCs + epsf meta/model) read from live 516; "
            "no sandbox ePSF rebuild"
        ),
        "runtime_s": time.perf_counter() - t0,
        "n_frames_sandbox": len(frames),
        "n_lc_stars": len(lc_ids),
        "n_psf_pool": len(psf_ids),
        "psf_pool_meta": psf_meta,
        "gain": gain_info,
        "fitrad_px": fitrad,
        "psfrad_px": psfrad,
        "ensemble": {"ids": ens_ids, "weights": ens_w, "source": ens_src},
        "check_cid": CHECK_CID,
        "check_has_psf_lc": (live_lc / f"lightcurve_{CHECK_CID}_psf.csv").is_file(),
        "m1": m1.to_dict(orient="records"),
        "m2": {
            "target_cid": TARGET_CID,
            "target_rms_mag": rms_tgt,
            "target_rms_mmag": rms_tgt_mm,
            "target_n_finite": int(np.isfinite(d_tgt).sum()),
            "target_top_epochs": _top_table(top_idx, resid, tgt_m["source_file"]),
            "check_cid": CHECK_CID,
            "check_rms_mag": rms_chk,
            "check_rms_mmag": rms_chk_mm,
            "check_n_finite": int(np.isfinite(d_chk).sum()),
            "check_top_epochs": _top_table(top_chk, resid_c, tgt_m["source_file"]),
            "reading_applied": reading,
            "reading_target": apply_reading(rms_tgt_mm),
            "reading_check": apply_reading(rms_chk_mm),
        },
        "recenter": rec_note,
        "failures_n": int(len(fail_df)),
        "failures_by_stage": fail_df["stage"].value_counts().to_dict() if not fail_df.empty else {},
        "g4_before": g4_before,
        "g4_after": g4_after,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "runtime_s": payload["runtime_s"],
        "m2_target_mmag": rms_tgt_mm,
        "m2_check_mmag": rms_chk_mm,
        "reading": reading,
        "failures_n": payload["failures_n"],
        "g4": g4_after["pass"],
    }, indent=2))
    return 0 if g4_after["pass"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception:
        traceback.print_exc()
        raise
