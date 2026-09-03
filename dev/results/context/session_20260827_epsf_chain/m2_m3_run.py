# -*- coding: ascii -*-
"""M2/M3: time ePSF chain on an era04 WORK COPY; compute G3. Never writes live 516."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src_py"))
sys.path.insert(0, str(REPO / "dev"))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from epsf_frame_accounting import list_epsf_science_light_fits  # noqa: E402
from epsf_psf_merge import merge_psf_into_sidecar  # noqa: E402
from epsf_science_set import build_epsf_science_set  # noqa: E402
from pipeline import _epsf_fit_catalog_ids, _export_catalog_psf_st_fields  # noqa: E402
from proc_frame_store import proc_csv_path_for_aligned_fits  # noqa: E402
from psf_internal_lc import write_internal_psf_lightcurves  # noqa: E402
from psf_photometry import build_epsf_model  # noqa: E402

SNAP = REPO / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
LIVE = REPO / "Archive" / "Drafts" / "draft_000516"
SETUP = "NoFilter_60_2"
WORK = REPO / "tmp" / "epsf_chain_m2_era04"
OUT = Path(__file__).resolve().parent
BO = "1498613634033133184"
FW = "1497343732462852864"
DRAFT_ID = 516
APERTURE_FULL_S = 1402.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dump(name: str, obj: dict) -> None:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="ascii")
    print("wrote", p.name, flush=True)


def _copy_work() -> tuple[Path, Path]:
    ps_src = SNAP / "platesolve" / SETUP
    li_src = SNAP / "detrended_aligned" / "lights" / SETUP
    ps_dst = WORK / "platesolve" / SETUP
    li_dst = WORK / "detrended_aligned" / "lights" / SETUP
    if (ps_dst / "masterstar_epsf.fits").is_file() and li_dst.is_dir():
        print("resume: reusing existing work copy", flush=True)
        return ps_dst, li_dst
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ps_src, ps_dst)
    shutil.copytree(li_src, li_dst)
    for name in ("cal_diag.json", "draft_manifest.json", "sat_diag.json"):
        src = SNAP / name
        if src.is_file():
            shutil.copy2(src, WORK / name)
            shutil.copy2(src, ps_dst / name)
    qc_src = SNAP / "calibrated" / "lights" / "qc_metrics.csv"
    if qc_src.is_file():
        qc_dst = WORK / "calibrated" / "lights"
        qc_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(qc_src, qc_dst / "qc_metrics.csv")
    return ps_dst, li_dst


def _g3(lc_dir: Path, tid: str) -> dict:
    path = lc_dir / f"lightcurve_{tid}_psf.csv"
    if not path.is_file():
        return {"missing": True, "path": str(path)}
    df = pd.read_csv(path, comment="#", low_memory=False)
    if "psf_delta_mag" not in df.columns:
        return {"missing_col": True, "path": str(path)}
    x = pd.to_numeric(df["psf_delta_mag"], errors="coerce").to_numpy(dtype=float)
    fin = x[np.isfinite(x)]
    n_fin = int(fin.size)
    if n_fin == 0:
        dem = float("nan")
    else:
        med = float(np.median(fin))
        dem = float(np.sqrt(np.mean((fin - med) ** 2))) * 1000.0
    drop = None
    if "drop_reason" in df.columns:
        vals = df["drop_reason"].dropna().astype(str)
        drop = vals.iloc[0] if not vals.empty else None
    return {
        "missing": False,
        "catalog_id": tid,
        "n_finite": n_fin,
        "n_rows": int(x.size),
        "demeaned_rms_mmag": dem,
        "drop_reason_sample": drop,
    }


def _meta_subset(meta: dict) -> dict:
    funnel = meta.get("build_funnel") or {}
    return {
        "n_stars_used": meta.get("n_stars_used"),
        "fwhm_px": meta.get("fwhm_px"),
        "cutout_size": meta.get("cutout_size"),
        "oversampling": meta.get("oversampling"),
        "spatial_order": meta.get("spatial_order"),
        "smoothing_kernel": meta.get("smoothing_kernel"),
        "created_utc": meta.get("created_utc"),
        "draft_id": meta.get("draft_id"),
        "n_after_science_scope": funnel.get("n_after_science_scope"),
        "n_after_isolation": funnel.get("n_after_isolation"),
        "n_final": funnel.get("n_final"),
        "epsf_qc": meta.get("epsf_qc"),
    }


def main() -> None:
    t_all = time.perf_counter()
    rec: dict = {"started_utc": _utc(), "work": str(WORK), "snapshot": str(SNAP)}

    live_epsf = LIVE / "platesolve" / SETUP / "masterstar_epsf.fits"
    live_meta_p = LIVE / "platesolve" / SETUP / "masterstar_epsf_meta.json"
    rec["live_516"] = {
        "epsf_exists": live_epsf.is_file(),
        "epsf_sha256": _sha256(live_epsf) if live_epsf.is_file() else None,
        "meta": _meta_subset(json.loads(live_meta_p.read_text(encoding="utf-8")))
        if live_meta_p.is_file()
        else None,
    }
    print("live epsf sha", rec["live_516"]["epsf_sha256"], flush=True)

    print("copying freeze to work copy...", flush=True)
    t0 = time.perf_counter()
    ps, lights = _copy_work()
    rec["copy_s"] = round(time.perf_counter() - t0, 3)
    print(f"copy {rec['copy_s']:.1f}s", flush=True)

    sci = build_epsf_science_set(ps)
    rec["science_set"] = sci.to_meta_dict()
    print("science set n_total", sci.n_total, flush=True)

    cfg = AppConfig()
    cfg.psf_photometry_enabled = True
    cfg.photometry_mode = "both"
    if (ps / "masterstar_epsf.fits").is_file():
        print("resume: skip rebuild, using existing ePSF", flush=True)
        rec["build_s"] = 0.0
        rec["epsf_path"] = str(ps / "masterstar_epsf.fits")
        rec["work_epsf_sha256"] = _sha256(ps / "masterstar_epsf.fits")
        meta_p = ps / "masterstar_epsf_meta.json"
        work_meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
        rec["work_meta"] = _meta_subset(work_meta)
        rec["build_n_stars"] = work_meta.get("n_stars_used")
    else:
        db = VyvarDatabase(cfg.database_path)
        try:
            print("M2 step1 build_epsf_model...", flush=True)
            t0 = time.perf_counter()
            epsf_path = build_epsf_model(
                masterstar_fits_path=ps / "MASTERSTAR.fits",
                masterstars_csv_path=ps / "masterstars_full_match.csv",
                db=db,
                draft_id=DRAFT_ID,
            )
            rec["build_s"] = round(time.perf_counter() - t0, 3)
            rec["epsf_path"] = str(epsf_path)
            rec["work_epsf_sha256"] = _sha256(Path(epsf_path))
            meta_p = ps / "masterstar_epsf_meta.json"
            work_meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
            rec["work_meta"] = _meta_subset(work_meta)
            rec["build_n_stars"] = work_meta.get("n_stars_used")
            print(
                f"build {rec['build_s']:.1f}s n_stars={rec['build_n_stars']} "
                f"sha={rec['work_epsf_sha256'][:16]}",
                flush=True,
            )
        finally:
            try:
                db.conn.close()
            except Exception:
                pass

    files = list_epsf_science_light_fits(lights)
    rec["n_frames"] = len(files)
    target_ids = _epsf_fit_catalog_ids(ps, psf_photometry_enabled=True)
    rec["n_fit_ids"] = len(target_ids) if target_ids else 0
    st_base = _export_catalog_psf_st_fields(cfg, ps)
    st_base["platesolve_dir"] = str(ps.resolve())
    st_base["draft_id"] = DRAFT_ID
    st_base["_run_epsf"] = True
    idx_by_name = {p.name: i for i, p in enumerate(files)}

    print(
        f"M2 step2/3 fit+merge {len(files)} frames x {rec['n_fit_ids']} ids...",
        flush=True,
    )
    per_frame: list[dict] = []
    t_fit = time.perf_counter()
    for i, fp in enumerate(files, start=1):
        t1 = time.perf_counter()
        st = dict(st_base)
        st["epsf_frame_index_by_name"] = idx_by_name
        st["epsf_frame_index"] = idx_by_name.get(fp.name)
        sidecar = proc_csv_path_for_aligned_fits(fp)
        skip = False
        if sidecar.is_file():
            try:
                prev = pd.read_csv(sidecar, usecols=lambda c: c in ("psf_fit_ok",), low_memory=False)
                if "psf_fit_ok" in prev.columns and bool(prev["psf_fit_ok"].astype(bool).any()):
                    skip = True
            except Exception:
                skip = False
        if skip:
            item = {
                "i": i,
                "file": fp.name,
                "s": 0.0,
                "n_fit": None,
                "n_ok": None,
                "status": "skipped_resume",
            }
            per_frame.append(item)
            print(f"  frame {i}/{len(files)} {fp.name} skipped_resume", flush=True)
            continue
        row = merge_psf_into_sidecar(
            fits_path=fp,
            sidecar_path=sidecar,
            st=st,
            target_ids=target_ids,
        )
        rec_f = row.get("psf_frame_record") or {}
        elapsed = time.perf_counter() - t1
        item = {
            "i": i,
            "file": fp.name,
            "s": round(elapsed, 3),
            "n_fit": rec_f.get("n_fit"),
            "n_ok": rec_f.get("n_ok"),
            "status": row.get("status"),
        }
        per_frame.append(item)
        if i <= 5 or i % 5 == 0 or i == len(files):
            print(
                f"  frame {i}/{len(files)} {fp.name} "
                f"n_fit={item['n_fit']} n_ok={item['n_ok']} {elapsed:.2f}s",
                flush=True,
            )
        if i == 1:
            eta = elapsed * len(files)
            print(f"  first-frame ETA remaining ~{eta:.0f}s", flush=True)
            _dump("m2_progress.json", {"first_s": elapsed, "eta_s": eta, "n_frames": len(files)})
    rec["fit_merge_s"] = round(time.perf_counter() - t_fit, 3)
    rec["per_frame"] = per_frame
    rec["n_fit_median"] = float(np.median([p["n_fit"] or 0 for p in per_frame]))
    rec["n_ok_median"] = float(np.median([p["n_ok"] or 0 for p in per_frame]))
    rec["n_ok_min"] = int(min(p["n_ok"] or 0 for p in per_frame))
    rec["n_ok_max"] = int(max(p["n_ok"] or 0 for p in per_frame))
    rec["fit_vs_aperture"] = rec["fit_merge_s"] / APERTURE_FULL_S
    print(
        f"fit+merge {rec['fit_merge_s']:.1f}s "
        f"({rec['fit_vs_aperture']:.2f} x aperture {APERTURE_FULL_S:.0f}s) "
        f"n_ok median={rec['n_ok_median']}",
        flush=True,
    )

    print("M2 step4 write_internal_psf_lightcurves...", flush=True)
    t0 = time.perf_counter()
    lc_out = write_internal_psf_lightcurves(
        platesolve_dir=ps,
        frames_root=lights,
        photometry_dir=ps / "photometry",
        cfg=cfg,
    )
    rec["lc_s"] = round(time.perf_counter() - t0, 3)
    rec["lc"] = {
        "n_written": lc_out.get("n_written"),
        "n_skipped": lc_out.get("n_skipped"),
    }
    print(f"LC {rec['lc_s']:.1f}s written={rec['lc']['n_written']}", flush=True)

    lc_dir = ps / "photometry" / "lightcurves"
    rec["m3"] = {"bo": _g3(lc_dir, BO), "fw": _g3(lc_dir, FW)}
    rec["m3"]["ref"] = {"bo_mmag": 8.495, "fw_mmag": 5.218, "n_full": 134}
    live_sha = rec["live_516"].get("epsf_sha256") or ""
    rec["m3"]["epsf_sha_match_live"] = rec["work_epsf_sha256"] == live_sha
    rec["m3"]["n_stars_match_live"] = rec["work_meta"].get("n_stars_used") == (
        rec["live_516"].get("meta") or {}
    ).get("n_stars_used")
    rec["total_s"] = round(time.perf_counter() - t_all, 3)
    rec["finished_utc"] = _utc()
    rec["cost_stop"] = rec["fit_merge_s"] > 0.5 * APERTURE_FULL_S
    _dump("m2_m3.json", rec)
    print("M3 BO", rec["m3"]["bo"], flush=True)
    print("M3 FW", rec["m3"]["fw"], flush=True)
    print("cost_stop", rec["cost_stop"], "total_s", rec["total_s"], flush=True)


if __name__ == "__main__":
    main()
