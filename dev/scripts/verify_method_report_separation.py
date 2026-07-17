#!/usr/bin/env python3
"""Part B verification: method-keyed report separation (362 PSF OFF, 364 PSF ON)."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from export_reports import export_all_method_lightcurve_reports  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from method_lc_output import MethodLcWriteContext, save_method_variant_lightcurve  # noqa: E402
from photometry_core import (  # noqa: E402
    _build_csv_lookup,
    _load_adaptive_blend_map,
    _normalize_gaia_id,
    compute_lc_flux_method,
    parse_comp_quality_json_map,
    read_flux_from_csv,
)
from citations import build_run_citation_context, load_pipeline_meta  # noqa: E402

CONFIG_PATH = _ROOT / "config.json"
DRAFT_362 = 362
SETUP_362 = "NoFilter_60_2"
DRAFT_364 = 364
SETUP_364 = "Luminance_180_2"


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _hash_dir(
    d: Path,
    *,
    exclude_suffixes: tuple[str, ...] = ("_psf", "_adaptive"),
) -> dict[str, str]:
    out: dict[str, str] = {}
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(s in name for s in exclude_suffixes):
            continue
        out[str(p.relative_to(d))] = _sha256(p)
    return out


def _set_config_flags(*, psf: bool, adaptive: bool) -> dict[str, Any]:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    backup = dict(raw)
    raw["psf_photometry_enabled"] = bool(psf)
    raw["psf_adaptive_enabled"] = bool(adaptive)
    CONFIG_PATH.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return backup


def _restore_config(backup: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(backup, indent=2) + "\n", encoding="utf-8")


@dataclass
class _MiniState:
    gaia_db_path: str | None
    plate_scale_arcsec: float
    obs_group: str


def _build_all_frames(
    proc_dir: Path,
    star_ids: list[str],
    cfg: AppConfig,
    apertures: dict[str, float],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(proc_dir.glob("proc_*.csv")):
        csv_df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
        lookup = _build_csv_lookup(csv_df, "catalog_id")
        df_frame = read_flux_from_csv(
            csv_path,
            star_ids,
            apertures,
            csv_df=csv_df,
            lookup=lookup,
            gain=float(cfg.gain),
            read_noise=float(cfg.read_noise),
        )
        if not df_frame.empty:
            frames.append(df_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _backfill_method_lcs(draft: int, setup: str, cfg: AppConfig) -> list[str]:
    """Generate PSF/adaptive LC CSVs from proc frames + existing aperture LC metadata."""
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft:06d}"
    phot = draft_dir / "platesolve" / setup / "photometry"
    lc_dir = phot / "lightcurves"
    proc_dir = draft_dir / "detrended_aligned" / "lights" / setup
    ps_dir = draft_dir / "platesolve" / setup
    comp_all = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    at_df = pd.read_csv(phot / "active_targets.csv", dtype={"catalog_id": str})
    at_by = {_norm_id(r["catalog_id"]): r for _, r in at_df.iterrows()}

    comp_index: dict[str, pd.DataFrame] = {}
    for tid, sub in comp_all.groupby("target_catalog_id"):
        comp_index[_norm_id(tid)] = sub.copy()

    ms_fits = ps_dir / "MASTERSTAR.fits"
    blend_map = _load_adaptive_blend_map(ms_fits if ms_fits.is_file() else None)

    state = _MiniState(
        gaia_db_path=str(cfg.gaia_db_path or "").strip() or None,
        plate_scale_arcsec=float(getattr(cfg, "export_arcsec_per_px", 1.3) or 1.3),
        obs_group=setup,
    )

    written: list[str] = []
    alt_methods: list[str] = []
    if bool(cfg.psf_photometry_enabled):
        alt_methods.append("psf")
    if bool(cfg.psf_adaptive_enabled):
        alt_methods.append("adaptive")
    if not alt_methods:
        return written

    for lc_path in sorted(lc_dir.glob("lightcurve_*.csv")):
        name = lc_path.stem
        if name.endswith("_psf") or name.endswith("_adaptive"):
            continue
        target_cid = name.replace("lightcurve_", "", 1)
        if not target_cid:
            continue
        lc_df = pd.read_csv(lc_path, low_memory=False)
        if lc_df.empty:
            continue
        target_row = at_by.get(_norm_id(target_cid))
        if target_row is None:
            continue
        target_comps = comp_index.get(_norm_id(target_cid), pd.DataFrame())
        if target_comps.empty:
            continue
        comp_ids = [_norm_id(c) for c in target_comps["catalog_id"].tolist() if _norm_id(c)]
        star_ids = [target_cid] + [c for c in comp_ids if c != target_cid]
        apertures = {cid: 3.0 * float(cfg.aperture_fwhm_factor) for cid in star_ids}
        all_frames = _build_all_frames(proc_dir, star_ids, cfg, apertures)
        if all_frames.empty:
            continue
        if "psf_flux" in all_frames.columns and (
            bool(cfg.psf_photometry_enabled) or bool(cfg.psf_adaptive_enabled)
        ):
            all_frames["lc_flux_method"] = compute_lc_flux_method(
                all_frames,
                blend_map,
                resolve_fwhm=float(cfg.psf_adaptive_resolve_fwhm),
                snr_lo=float(cfg.psf_adaptive_snr_lo),
            )

        comp_catalog_mag = {
            _norm_id(r["catalog_id"]): float(pd.to_numeric(r.get("mag"), errors="coerce"))
            for _, r in target_comps.iterrows()
            if _norm_id(r["catalog_id"])
        }
        tier_weights = {
            1: float(cfg.comp_tier1_weight),
            2: float(cfg.comp_tier2_weight),
            3: float(cfg.comp_tier3_weight),
            4: float(cfg.comp_tier4_weight),
        }
        comp_tier_map: dict[str, int] = {}
        comp_rms_map: dict[str, float] = {}
        for _, r in target_comps.iterrows():
            cid0 = _normalize_gaia_id(r["catalog_id"])
            tier = int(pd.to_numeric(r.get("comp_tier", 4), errors="coerce") or 4)
            comp_tier_map[cid0] = max(1, min(4, tier))
            rms_raw = float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
            tw = float(tier_weights.get(comp_tier_map[cid0], 0.25))
            if np.isfinite(rms_raw) and rms_raw > 1e-6 and np.isfinite(tw) and tw > 0:
                comp_rms_map[cid0] = rms_raw / np.sqrt(tw)
            else:
                comp_rms_map[cid0] = rms_raw

        bjd = pd.to_numeric(lc_df["bjd"], errors="coerce").to_numpy(dtype=float)
        hjd = pd.to_numeric(lc_df.get("hjd", bjd), errors="coerce").to_numpy(dtype=float)
        jd = pd.to_numeric(lc_df.get("jd", bjd), errors="coerce").to_numpy(dtype=float)
        airmass_arr = pd.to_numeric(lc_df.get("airmass"), errors="coerce").to_numpy(dtype=float)
        flip_arr = lc_df.get("is_flipped", pd.Series(False, index=lc_df.index)).astype(bool).to_numpy()
        err = pd.to_numeric(lc_df.get("err"), errors="coerce").to_numpy(dtype=float)
        ap_arr = pd.to_numeric(lc_df.get("aperture_r_px"), errors="coerce").to_numpy(dtype=float)
        src_files = lc_df.get("source_file", pd.Series([""] * len(lc_df))).astype(str).tolist()
        flags = lc_df.get("flag", pd.Series([""] * len(lc_df))).astype(str).tolist()
        sat_flags = np.array([f.strip().lower() == "saturated" for f in flags], dtype=bool)
        target_frames = all_frames[all_frames["catalog_id"].astype(str).map(_norm_id) == _norm_id(target_cid)].copy()

        ac_ok = bool(lc_df.get("ac_ok", pd.Series([False])).iloc[0]) if "ac_ok" in lc_df.columns else False
        ac_result = {
            "ok": ac_ok,
            "delta_m_corr": float(pd.to_numeric(lc_df.get("ac_correction", pd.Series([np.nan])).iloc[0], errors="coerce")),
            "scatter_mag": float(pd.to_numeric(lc_df.get("ac_scatter", pd.Series([np.nan])).iloc[0], errors="coerce")),
            "n_ref_stars": int(pd.to_numeric(lc_df.get("ac_n_ref", pd.Series([0])).iloc[0], errors="coerce") or 0),
        }
        comp_bp_rp: dict[str, float] = {}
        for _, r in target_comps.iterrows():
            cid0 = _normalize_gaia_id(r["catalog_id"])
            comp_bp_rp[cid0] = float(pd.to_numeric(r.get("bp_rp"), errors="coerce"))
        target_bp_rp = float(pd.to_numeric(target_row.get("bp_rp"), errors="coerce"))

        base_ctx = MethodLcWriteContext(
            method="psf",
            target_cid=target_cid,
            comp_ids=comp_ids,
            all_frames=all_frames,
            lc_dir=lc_dir,
            cfg=cfg,
            stability_sigma=3.0,
            outlier_sigma=3.0,
            comp_catalog_mag=comp_catalog_mag,
            comp_rms_map=comp_rms_map,
            comp_tier_map=comp_tier_map,
            tier_weights=tier_weights,
            target_row=target_row,
            state=state,
            apertures_px=apertures,
            ac_result=ac_result,
            comp_bp_rp=comp_bp_rp,
            target_bp_rp=target_bp_rp,
            bjd=bjd,
            hjd=hjd,
            jd=jd,
            airmass_arr=airmass_arr,
            flip_arr=flip_arr,
            err=err,
            ap_arr=ap_arr,
            src_files=src_files,
            sat_flags=sat_flags,
            target_frames=target_frames,
            lunar_phase_pct=float(pd.to_numeric(lc_df.get("lunar_phase_pct", pd.Series([np.nan])).iloc[0], errors="coerce")),
            lunar_separation_deg=float(
                pd.to_numeric(lc_df.get("lunar_separation_deg", pd.Series([np.nan])).iloc[0], errors="coerce")
            ),
            lunar_risk=str(lc_df.get("lunar_risk", pd.Series(["UNKNOWN"])).iloc[0] or "UNKNOWN"),
        )
        for m in alt_methods:
            ctx = MethodLcWriteContext(**{**base_ctx.__dict__, "method": m})
            out = save_method_variant_lightcurve(ctx)
            if out is not None:
                written.append(out.name)
    return written


def _reexport_draft(draft: int, setup: str, cfg: AppConfig) -> int:
    phot = Path(cfg.archive_root) / "Drafts" / f"draft_{draft:06d}" / "platesolve" / setup / "photometry"
    lc_dir = phot / "lightcurves"
    reports_dir = phot / "lightcurves_reports"
    at_df = pd.read_csv(phot / "active_targets.csv", dtype={"catalog_id": str})
    sum_df = pd.read_csv(phot / "photometry_summary.csv", dtype={"catalog_id": str})
    comp_all = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    sum_by = {str(r["catalog_id"]).strip(): r for _, r in sum_df.iterrows() if str(r.get("catalog_id", "")).strip()}
    comp_index = {_norm_id(tid): sub.copy() for tid, sub in comp_all.groupby("target_catalog_id")}
    run_cite = build_run_citation_context(cfg, pipeline_meta=load_pipeline_meta(phot), targets_df=at_df)
    n_ok = 0
    for _, trow in at_df.iterrows():
        target_cid = _norm_id(trow.get("catalog_id", ""))
        if not target_cid or not (lc_dir / f"lightcurve_{target_cid}.csv").is_file():
            continue
        cq_path = lc_dir / f"comp_quality_{target_cid}.json"
        comp_qmap: dict[str, str] = {}
        if cq_path.is_file():
            raw = json.loads(cq_path.read_text(encoding="utf-8"))
            for qk, qv in parse_comp_quality_json_map(raw).items():
                nk = _norm_id(qk)
                q2 = str(qv.get("quality", "")).strip().lower()
                if q2 != "excluded":
                    comp_qmap[nk] = q2
        paths = export_all_method_lightcurve_reports(
            reports_dir,
            trow,
            lc_dir=lc_dir,
            target_cid=target_cid,
            comp_df=comp_index.get(target_cid, pd.DataFrame()).copy(),
            summary_row=sum_by.get(target_cid, pd.Series(dtype=object)),
            observer_code=str(cfg.observer_code or ""),
            observer_name=str(cfg.observer_name or "Unknown Observer"),
            comp_quality_map=comp_qmap or None,
            arcsec_per_px=float(cfg.export_arcsec_per_px),
            software_version="VYVAR 1.0",
            cfg=cfg,
            obs_group=setup,
            targets_df=at_df,
            run_citation_ctx=run_cite,
        )
        if paths:
            n_ok += 1
    return n_ok


def _read_export_label(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for ln in text.splitlines():
        if ln.startswith("#SOFTWARE="):
            out["software"] = ln.split("=", 1)[1].strip()
        if "meth=" in ln and ln.startswith("#"):
            out["notes_meth"] = ln
    for ln in text.splitlines():
        if "," in ln and not ln.startswith("#"):
            parts = ln.split(",")
            if len(parts) >= 15:
                out["notes_field"] = parts[14]
            break
    return out


def verify_362_aperture_only(cfg: AppConfig) -> dict[str, Any]:
    phot = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_362:06d}" / "platesolve" / SETUP_362 / "photometry"
    aavso = phot / "lightcurves_reports" / "aavso"
    varastro = phot / "lightcurves_reports" / "varastro"
    before = _hash_dir(aavso)
    n = _reexport_draft(DRAFT_362, SETUP_362, cfg)
    after = _hash_dir(aavso)
    psf_files = list(aavso.glob("*_psf.txt")) + list(aavso.glob("*_adaptive.txt"))
    sample = next(iter(sorted(aavso.glob("*.txt"))), None)
    labels = _read_export_label(sample) if sample else {}
    return {
        "targets_reexported": n,
        "aavso_files_before": len(before),
        "aavso_files_after": len(after),
        "byte_stable": before == after,
        "hash_mismatches": sorted(set(before) ^ set(after))[:5]
        + [k for k in before if k in after and before[k] != after[k]][:5],
        "psf_or_adaptive_aavso": len(psf_files),
        "varastro_psf_adaptive": len(list(varastro.glob("*_psf.txt"))) + len(list(varastro.glob("*_adaptive.txt"))),
        "sample_software": labels.get("software", ""),
        "sample_notes_meth": labels.get("notes_field", ""),
    }


def verify_364_psf_on(cfg: AppConfig, *, adaptive: bool) -> dict[str, Any]:
    phot = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_364:06d}" / "platesolve" / SETUP_364 / "photometry"
    lc_dir = phot / "lightcurves"
    aavso = phot / "lightcurves_reports" / "aavso"

    written = _backfill_method_lcs(DRAFT_364, SETUP_364, cfg)
    n = _reexport_draft(DRAFT_364, SETUP_364, cfg)

    aperture_files = [p for p in aavso.glob("*.txt") if "_psf" not in p.name and "_adaptive" not in p.name]
    psf_files = list(aavso.glob("*_psf.txt"))
    adapt_files = list(aavso.glob("*_adaptive.txt"))

    flux_check: dict[str, Any] = {}
    for lc_ap in sorted(lc_dir.glob("lightcurve_*.csv")):
        if "_psf" in lc_ap.stem or "_adaptive" in lc_ap.stem:
            continue
        cid = lc_ap.stem.replace("lightcurve_", "", 1)
        lc_psf = lc_dir / f"lightcurve_{cid}_psf.csv"
        if not lc_psf.is_file():
            continue
        df_a = pd.read_csv(lc_ap, usecols=["mag_inst", "method"], low_memory=False)
        df_p = pd.read_csv(lc_psf, usecols=["mag_inst", "method"], low_memory=False)
        same_inst = bool(np.allclose(df_a["mag_inst"], df_p["mag_inst"], equal_nan=True))
        flux_check[cid] = {
            "aperture_method_col": str(df_a["method"].iloc[0]) if "method" in df_a.columns else "",
            "psf_method_col": str(df_p["method"].iloc[0]) if "method" in df_p.columns else "",
            "mag_inst_identical_to_aperture": same_inst,
            "psf_na_frames": int(df_p["mag_inst"].isna().sum()),
        }

    labels: dict[str, str] = {}
    for tag, p in (
        ("aperture", next(iter(aperture_files), None)),
        ("psf", next(iter(psf_files), None)),
        ("adaptive", next(iter(adapt_files), None)),
    ):
        if p is not None:
            labels[tag] = _read_export_label(p).get("software", "")

    return {
        "adaptive_flag": adaptive,
        "method_lcs_written": written,
        "targets_reexported": n,
        "aperture_aavso": len(aperture_files),
        "psf_aavso": len(psf_files),
        "adaptive_aavso": len(adapt_files),
        "export_labels": labels,
        "flux_check": flux_check,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = AppConfig()
    orig = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    print("=== Part B1: draft 362, PSF OFF (default) ===")
    _restore_config({**orig, "psf_photometry_enabled": False, "psf_adaptive_enabled": False})
    cfg = AppConfig()
    r362 = verify_362_aperture_only(cfg)
    for k, v in r362.items():
        print(f"  {k}: {v}")

    print("\n=== Part B2a: draft 364, psf_photometry_enabled=true ===")
    backup = _set_config_flags(psf=True, adaptive=False)
    cfg = AppConfig()
    r364_psf = verify_364_psf_on(cfg, adaptive=False)
    for k, v in r364_psf.items():
        print(f"  {k}: {v}")

    print("\n=== Part B2b: draft 364, psf + adaptive enabled ===")
    _set_config_flags(psf=True, adaptive=True)
    cfg = AppConfig()
    r364_ad = verify_364_psf_on(cfg, adaptive=True)
    for k, v in r364_ad.items():
        print(f"  {k}: {v}")

    print("\n=== Restore config flags ===")
    _restore_config(backup)
    restored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    print(
        f"  psf_photometry_enabled={restored.get('psf_photometry_enabled')} "
        f"psf_adaptive_enabled={restored.get('psf_adaptive_enabled')}"
    )

    ok362 = (
        r362["byte_stable"]
        and r362["psf_or_adaptive_aavso"] == 0
        and r362["varastro_psf_adaptive"] == 0
        and "aperture" in r362.get("sample_software", "").lower()
    )
    ok364 = (
        r364_ad["psf_aavso"] > 0
        and r364_ad["aperture_aavso"] > 0
        and "PSF" in r364_ad["export_labels"].get("psf", "")
        and any(not v.get("mag_inst_identical_to_aperture", True) for v in r364_ad["flux_check"].values())
    )
    ok_ad = r364_ad["adaptive_aavso"] > 0 and "adaptive" in r364_ad["export_labels"].get("adaptive", "").lower()

    print(f"\nOVERALL: 362_ok={ok362} 364_psf_ok={ok364} 364_adaptive_ok={ok_ad}")
    return 0 if (ok362 and ok364 and ok_ad) else 1


if __name__ == "__main__":
    raise SystemExit(main())
