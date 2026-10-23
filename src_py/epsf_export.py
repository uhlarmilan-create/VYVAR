"""EPSF-EXPORT-2026-10 (Milan 2026-10-19): ePSF light curves for the user.

RUN ePSF writes, next to the internal ``lightcurve_<id>_psf.csv``, one plain file per measured target in
``photometry/lightcurves_reports/epsf/``: BJD_TDB, differential magnitude, 1-sigma error, with a commented
header describing the star, the night, the method, the comparison ensemble and the epoch accounting. It is a
user product, not an AAVSO / VarAstro submission (INV-PSF-SUBMIT-01 keeps those aperture-only).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from epsf_frame import OVERSAMPLING
from epsf_lc_compare import p2p_scatter
from export_reports import _bjd_to_datestr_yyyymmdd, _guess_setup_info_from_obs_group, _safe_filename
from gaia_catalog_id import read_vyvar_csv
from report_methods import lc_csv_path
from star_page_m3 import _psf_lc_header
from vyvar_version import VYVAR_SOFTWARE_VERSION

EPSF_EXPORT_DIRNAME = "epsf"
COLUMNS = ("bjd_tdb", "delta_mag", "err")


def epsf_export_path(photometry_dir: Path | str, name: str, bjd: float, setup: str) -> Path:
    """``<photometry>/lightcurves_reports/epsf/<name>_<YYYYMMDD>_<setup>_epsf.txt``."""
    return (Path(photometry_dir) / "lightcurves_reports" / EPSF_EXPORT_DIRNAME
            / f"{_safe_filename(name)}_{_bjd_to_datestr_yyyymmdd(bjd)}_{_safe_filename(setup)}_epsf.txt")


def find_epsf_export(photometry_dir: Path | str, cid: str, setup: str) -> Path | None:
    """The export file of one target and setup (matched by its ``# catalog_id=`` header line)."""
    for p in sorted((Path(photometry_dir) / "lightcurves_reports" / EPSF_EXPORT_DIRNAME).glob(
            f"*_{_safe_filename(setup)}_epsf.txt")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                if line.startswith("# catalog_id="):
                    if line.split("=", 1)[1].strip() == str(cid):
                        return p
                    break
    return None


def _epsf_rows(psf_csv: Path) -> pd.DataFrame:
    """Epochs the ePSF light curve kept (``psf_fit_ok`` and a finite magnitude / error), sorted by time."""
    lc = pd.read_csv(psf_csv, comment="#", low_memory=False)
    ok = lc["psf_fit_ok"].astype(str).str.lower().eq("true")
    out = pd.DataFrame({
        "bjd_tdb": pd.to_numeric(lc["bjd"], errors="coerce"),
        "delta_mag": pd.to_numeric(lc["psf_delta_mag"], errors="coerce"),
        "err": pd.to_numeric(lc["psf_delta_mag_err"], errors="coerce"),
    })
    out = out.loc[ok & np.isfinite(out).all(axis=1)]
    return out.sort_values("bjd_tdb").reset_index(drop=True)


def _target_info(photometry_dir: Path) -> dict[str, dict[str, Any]]:
    from epsf_science_set import _norm_id_gaia_or_raw  # noqa: PLC0415

    at_p = photometry_dir / "active_targets.csv"
    if not at_p.is_file():
        return {}
    at = read_vyvar_csv(at_p, low_memory=False)
    out: dict[str, dict[str, Any]] = {}
    for _, r in at.iterrows():
        cid = _norm_id_gaia_or_raw(r.get("catalog_id"))
        if cid and cid not in out:
            out[cid] = r.to_dict()
    return out


def _txt(v: Any) -> str:
    s = "" if v is None else str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def _num(v: Any, nd: int) -> str:
    x = pd.to_numeric(v, errors="coerce")
    return f"{float(x):.{nd}f}" if pd.notna(x) else ""


def epsf_export_lines(cid: str, info: dict[str, Any], psf_csv: Path, rows: pd.DataFrame, *,
                      setup: str, observer_code: str, manifest: dict[str, Any]) -> list[str]:
    """Header + data lines of one export file."""
    hdr = _psf_lc_header(psf_csv)
    name = _txt(info.get("vsx_name")) or _txt(info.get("name")) or cid
    filt, exptime, binning = _guess_setup_info_from_obs_group(setup)
    t0, t1 = float(rows["bjd_tdb"].iloc[0]), float(rows["bjd_tdb"].iloc[-1])
    frames = manifest.get("frames") or []
    fw = np.array([float(f.get("fwhm_px", np.nan)) for f in frames], dtype=np.float64)
    merged = [m for m in ((manifest.get("fit_sources") or {}).get("merged_neighbours") or [])
              if str(m.get("catalog_id")) == cid]
    drops = [f"{hdr[k]} ({lbl})" for k, lbl in (("psf_lc_n_epochs_target_saturated", "target saturated"),
                                               ("psf_lc_n_epochs_target_outside_footprint", "target off the frame's data"),
                                               ("psf_lc_n_epochs_dropped_pin", "frame not fitted or a comparison star missing"))
             if int(float(hdr.get(k, "0") or 0))]
    grid = "x".join(str(v) for v in ((manifest.get("spatial") or {}).get("grid") or [1, 1]))
    p2p = 1e3 * p2p_scatter(rows["delta_mag"])
    L = [
        "# VYVAR ePSF light curve - user product, NOT an AAVSO / VarAstro submission file",
        f"# star={name}",
        f"# catalog_id={cid}",
        f"# vsx_type={_txt(info.get('vsx_type'))}",
        f"# vsx_period_d={_txt(info.get('vsx_period'))}",
        f"# ra_deg={_num(info.get('ra_deg'), 6)}",
        f"# dec_deg={_num(info.get('dec_deg'), 6)}",
        f"# gaia_g_mag={_num(info.get('mag'), 3)}",
        f"# observer_code={observer_code}",
        f"# setup={setup}",
        f"# filter={filt}",
        f"# exptime_s={exptime or ''}",
        f"# binning={binning or ''}",
        f"# night_utc={_bjd_to_datestr_yyyymmdd(t0)}",
        f"# time_span_bjd_tdb={t0:.6f}-{t1:.6f}",
        "# time=BJD_TDB, mid-exposure",
        "# delta_mag=differential instrumental magnitude of the target against the weighted ensemble of the "
        "comparison stars below (not transformed to a standard system; aperture and ePSF levels differ by a constant)",
        "# second_order_extinction=" + ("k'' x (target - comparison colour) x airmass subtracted, the same term as the "
                                        "aperture light curve" if hdr.get("psf_lc_k2_applied") == "true" else "not applied"),
        "# err=1-sigma magnitude error: PSF fit (Fisher, model variance) + unresolved-neighbour term + ensemble "
        "+ scintillation + night systematic floor",
        f"# method=per-frame ePSF (photutils EPSFBuilder, oversampling {OVERSAMPLING}, {grid} spatial perturbation), "
        "joint fit of the target with its Gaia neighbours at fixed positions, local sky annulus",
        f"# epsf_frames={len(frames)}",
        f"# epsf_fwhm_px_median={np.nanmedian(fw):.2f}" if fw.size and np.isfinite(fw).any() else "# epsf_fwhm_px_median=",
        f"# epsf_model_sha256={hdr.get('epsf_model_sha256', '')}",
        f"# ensemble_n_comp={hdr.get('ensemble_n_comp', '')}",
        f"# ensemble_comp_ids={hdr.get('ensemble_pinned_ids', '')}",
    ]
    for m in merged:
        L.append(f"# unresolved_neighbour=Gaia DR3 {m['neighbour_id']} at {float(m['sep_px']):.1f} px, "
                 f"{float(m['dmag']):+.2f} mag (light included in delta_mag)")
    L += [
        f"# n_epochs={len(rows)}",
        f"# epochs_dropped={'; '.join(drops) if drops else 'none'}",
        "# p2p_scatter_mmag=" + (f"{p2p:.1f} (robust epoch-to-epoch scatter, includes the star's own variation)"
                                 if math.isfinite(p2p) else ""),
        f"# epsf_targets_selection={(manifest.get('fit_set') or {}).get('selection', '')}",
        f"# software={VYVAR_SOFTWARE_VERSION}",
        ",".join(COLUMNS),
    ]
    L += [f"{b:.6f},{m:.4f},{e:.4f}" for b, m, e in rows.itertuples(index=False)]
    return L


def write_epsf_user_lightcurves(platesolve_dir: Path | str, target_ids: Iterable[str], *, setup: str,
                                observer_code: str = "", photometry_dir: Path | str | None = None) -> dict[str, Any]:
    """One export file per target whose ePSF light curve has at least one kept epoch. Returns
    {"n_written", "paths", "skipped": {catalog_id: reason}}."""
    import json  # noqa: PLC0415

    ps = Path(platesolve_dir)
    phot = Path(photometry_dir) if photometry_dir is not None else ps / "photometry"
    lc_dir = phot / "lightcurves"
    man_p = ps / "epsf_frames" / "epsf_frames_manifest.json"
    manifest = json.loads(man_p.read_text(encoding="utf-8")) if man_p.is_file() else {}
    info = _target_info(phot)
    paths: list[Path] = []
    skipped: dict[str, str] = {}
    for cid in sorted(str(t) for t in target_ids):
        old = find_epsf_export(phot, cid, setup)
        if old is not None:  # this run replaces the target's file (its name may change, or it may have no epoch)
            old.unlink()
        psf_csv = lc_csv_path(lc_dir, cid, "psf")
        if not psf_csv.is_file():
            skipped[cid] = "no ePSF light curve"
            continue
        rows = _epsf_rows(psf_csv)
        if rows.empty:
            skipped[cid] = "no epoch kept"
            continue
        tinfo = info.get(cid, {})
        name = _txt(tinfo.get("vsx_name")) or _txt(tinfo.get("name")) or cid
        out = epsf_export_path(phot, name, float(rows["bjd_tdb"].iloc[0]), setup)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(epsf_export_lines(cid, tinfo, psf_csv, rows, setup=setup,
                                                   observer_code=observer_code, manifest=manifest)) + "\n",
                       encoding="utf-8")
        paths.append(out)
    return {"n_written": len(paths), "paths": [str(p) for p in paths], "skipped": skipped}
