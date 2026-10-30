"""Internal diagnostic per-target PSF light curves (EPSF-LC-LOG-01).

These files are NEVER an AAVSO/VarAstro submission product (INV-PSF-SUBMIT-01).
They are written additively beside aperture LCs; catalogs, aperture LCs, and
export bytes are not rewritten by this module.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from gaia_catalog_id import normalize_gaia_source_id
from report_methods import lc_csv_path
from stats_core import _coerce_bool

LOGGER = logging.getLogger(__name__)

INTERNAL_BANNER = "# INTERNAL DIAGNOSTIC PRODUCT - NOT FOR AAVSO/VARASTRO SUBMISSION"
# G09-14: what the file is now (EPSF-SHAPE-01 is closed; the PSF is per frame)
TRUST_NOTE_LINE1 = ("# differential ePSF magnitudes against the target's aperture ensemble (its comparison stars "
                    "with an ePSF measurement on the night);")
TRUST_NOTE_LINE2 = "# relative photometry only - internal comparison with the aperture light curve, not submitted"
PRODUCT_NAME = "internal_psf_diagnostic"
# EPSF-SPATIAL-2026-10: one fit for every star of a frame (photutils PSFPhotometry, frame ePSF +
# night spatial perturbation, local annulus background); errors = Fisher, model variance.
PSF_FIT_METHOD = "frame_psfphotometry_epsf_spatial_local_bkg"
PSF_ERR_MODE = "fisher_model_variance"

_PROC_USECOLS = (
    "catalog_id",
    "bjd_tdb_mid",
    "hjd_mid",
    "jd_mid",
    "psf_flux",
    "psf_flux_err",
    "psf_chi2",
    "psf_fit_ok",
    "psf_sat_masked",
    "psf_edge_masked",
    "dao_flux",
)


def _time_base_of(aperture_lc: pd.DataFrame) -> str:
    """G09-39: the time base of the aperture light curve the PSF light curve copies its times from."""
    from photometry_lightcurve import resolve_lc_time_base  # noqa: PLC0415

    try:
        return resolve_lc_time_base(aperture_lc)
    except ValueError as exc:
        return f"unknown ({exc})"


def psf_usable_mask(fit_ok: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Epochs that enter the ensemble: ``psf_fit_ok`` (night chi2 locus, FIT-OK-ADMISSION-01) and a positive flux."""
    fl = np.asarray(flux, dtype=np.float64)
    return np.asarray(fit_ok, dtype=bool) & np.isfinite(fl) & (fl > 0)


def _flux_to_inst_mag(flux: np.ndarray) -> np.ndarray:
    f = np.asarray(flux, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def _load_gain_authority(photometry_dir: Path) -> tuple[float, str]:
    """The gain aperture photometry used (GAIN-ONE-2026-10, aperture_per_target.json)."""
    from aperture_pertarget import load_photometric_gain  # noqa: PLC0415

    g, _rn = load_photometric_gain(photometry_dir)
    return (g, "aperture_photometry") if math.isfinite(g) and g > 0 else (float("nan"), "unresolved")


def load_epsf_build_meta(platesolve_dir: Path) -> dict[str, Any]:
    """Provenance of the ePSF models behind the PSF columns.

    EPSF-FRAME-2026-10: RUN ePSF fits every frame with its own ePSF; the manifest
    ``epsf_frames/epsf_frames_manifest.json`` lists them (sha256, FWHM, build stars).
    :func:`psf_photometry.epsf_models_sha256` of its content identifies the model set. The models
    themselves are not written (OUTPUT-REVIEW-2026-10). Without a manifest every field is empty.
    """
    from psf_photometry import EPSF_FRAMES_DIR, EPSF_FRAMES_MANIFEST, epsf_models_sha256  # noqa: PLC0415

    manifest_path = Path(platesolve_dir) / EPSF_FRAMES_DIR / EPSF_FRAMES_MANIFEST
    if not manifest_path.is_file():
        return {"model_path": None, "model_name": "", "model_sha256": "", "n_stars": -1, "build_timestamp": "",
                "oversampling": "", "smoothing_kernel": "", "cutout_size": "", "raw_meta": {}}
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = man.get("frames") or []
    smooth = sorted({str(f.get("smoothing_kernel") or "") for f in frames})
    return {
        "model_path": manifest_path,
        "model_name": f"{EPSF_FRAMES_DIR}/{EPSF_FRAMES_MANIFEST} (per-frame ePSF, {len(frames)} frames)",
        "model_sha256": epsf_models_sha256(man),
        "n_stars": int(man.get("n_build_stars", -1)),
        "build_timestamp": str(man.get("created_utc") or ""),
        "oversampling": man.get("oversampling", ""),
        "smoothing_kernel": ",".join(smooth),
        "cutout_size": man.get("cutout_size", ""),
        "raw_meta": man,
    }


def _git_provenance() -> tuple[str, str]:
    from photometry_provenance import _resolve_git_provenance  # noqa: PLC0415

    git_hash, git_dirty, _files = _resolve_git_provenance()
    gh = str(git_hash) if git_hash else ""
    if git_dirty is True:
        dirty = "True"
    elif git_dirty is False:
        dirty = "False"
    else:
        dirty = "null"
    return gh, dirty


def resolve_ensemble_ids(target_cid: str, photometry_dir: Path) -> tuple[list[str], str]:
    """Aperture ``comparison_stars_per_target`` membership for ``target_cid`` (G09-16: ids only - the
    PSF ensemble computes its own Broeg weights in ensemble_normalize)."""
    tid = normalize_gaia_source_id(target_cid)
    comp_pt = Path(photometry_dir) / "comparison_stars_per_target.csv"
    if not comp_pt.is_file():
        return [], "none"
    from comp_table import load_comp_table, used  # noqa: PLC0415

    df = used(load_comp_table(comp_pt))
    if "target_catalog_id" not in df.columns or "catalog_id" not in df.columns:
        return [], "none"
    tcol = df["target_catalog_id"].map(normalize_gaia_source_id)
    sub = df.loc[tcol == tid]
    ids = [normalize_gaia_source_id(v) for v in sub["catalog_id"].tolist()]
    ids = [c for c in ids if c]
    return ids, "comparison_stars_per_target"


def comps_with_psf_measurement_on_night(stack: pd.DataFrame) -> set[str]:
    """Catalog IDs with at least one finite psf_flux > 0 across the night stack.

    Structural presence check (EPSF-CHI2-LOCUS-01 Phase 1b Defect 2): a pinned
    comp that was never fitted (e.g. outside the ePSF science set) cannot be a
    PSF-LC ensemble member. This is absence, not a quality filter.
    """
    if stack is None or stack.empty or "catalog_id" not in stack.columns:
        return set()
    if "psf_flux" not in stack.columns:
        return set()
    flux = pd.to_numeric(stack["psf_flux"], errors="coerce")
    ok = np.isfinite(flux.to_numpy(dtype=np.float64)) & (flux.to_numpy(dtype=np.float64) > 0)
    ids = stack.loc[ok, "catalog_id"].map(normalize_gaia_source_id)
    return {c for c in ids.tolist() if c}


def filter_ensemble_to_psf_measured(
    comp_ids: Sequence[str],
    measured: set[str],
) -> tuple[list[str], list[str]]:
    """Drop comps with no PSF measurement on the night; returns (kept, dropped)."""
    kept: list[str] = []
    dropped: list[str] = []
    for cid in comp_ids:
        c = normalize_gaia_source_id(cid)
        if not c:
            continue
        if c in measured:
            kept.append(c)
        else:
            dropped.append(c)
    return kept, dropped


def build_provenance_header(
    *,
    epsf_meta: dict[str, Any],
    gain_value: float,
    gain_source: str,
    n_comp: int,
    pinned_ids: Sequence[str],
    ensemble_source: str,
    git_hash: str,
    git_dirty: str,
    n_epochs_full: int = 0,
    n_epochs_dropped_pin: int = 0,
    psf_ap_level_offset_mag: float = float("nan"),
    extra_header_lines: Sequence[str] = (),
    ensemble_dropped_no_psf: Sequence[str] = (),
) -> list[str]:
    gtxt = f"{gain_value:.6g}" if math.isfinite(float(gain_value)) else "nan"
    ids = ",".join(str(c) for c in pinned_ids)
    dropped = ",".join(str(c) for c in ensemble_dropped_no_psf)
    lines = [
        INTERNAL_BANNER,
        f"# epsf_model_file={epsf_meta.get('model_name') or ''}",
        f"# epsf_model_sha256={epsf_meta.get('model_sha256') or ''}",
        f"# epsf_n_stars={epsf_meta.get('n_stars')}",
        f"# epsf_build_timestamp={epsf_meta.get('build_timestamp') or ''}",
        f"# epsf_oversampling={epsf_meta.get('oversampling')}",
        f"# epsf_smoothing_kernel={epsf_meta.get('smoothing_kernel') or ''}",
        f"# epsf_cutout_size={epsf_meta.get('cutout_size')}",
        f"# psf_fit={PSF_FIT_METHOD}",
        f"# psf_err_mode={PSF_ERR_MODE}",
        f"# gain_authority=gain={gtxt} source={gain_source}",
        f"# ensemble_n_comp={int(n_comp)}",
        f"# ensemble_pinned_ids={ids}",
        f"# ensemble_source={ensemble_source}",
        f"# ensemble_dropped_no_psf_measurement={dropped}",
        "# ensemble_drop_reason=structural_absence_no_psf_on_night"
        if dropped
        else "# ensemble_drop_reason=",
        f"# git_hash={git_hash}",
        f"# git_dirty={git_dirty}",
        TRUST_NOTE_LINE1,
        TRUST_NOTE_LINE2,
        f"# product={PRODUCT_NAME}",
        f"# psf_lc_n_epochs_full={int(n_epochs_full)}",
        f"# psf_lc_n_epochs_dropped_pin={int(n_epochs_dropped_pin)}",
        f"# psf_ap_level_offset_mag={psf_ap_level_offset_mag:.6g}"
        if math.isfinite(float(psf_ap_level_offset_mag))
        else "# psf_ap_level_offset_mag=nan",
    ]
    for extra in extra_header_lines:
        s = str(extra).rstrip()
        if s:
            lines.append(s)
    return lines


def _read_proc_stack(frames_root: Path) -> pd.DataFrame:
    root = Path(frames_root)
    files = sorted(root.glob("proc_*.csv"))
    if not files:
        raise FileNotFoundError(f"No proc_*.csv under {root}")
    chunks: list[pd.DataFrame] = []
    for p in files:
        try:
            from proc_frame_store import read_proc_csv  # noqa: PLC0415

            df = read_proc_csv(p, low_memory=False, dtype={"catalog_id": str})
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PSF-LC] skip unreadable %s: %s", p.name, exc)
            continue
        keep = [c for c in _PROC_USECOLS if c in df.columns]
        if "catalog_id" not in keep:
            continue
        df = df[keep].copy()
        df["catalog_id"] = df["catalog_id"].map(normalize_gaia_source_id)
        df["proc_csv"] = p.name
        df["epoch_key"] = p.name  # = aperture LC source_file (proc_*.csv name)
        chunks.append(df)
    if not chunks:
        raise FileNotFoundError(f"No readable proc CSVs under {root}")
    return pd.concat(chunks, ignore_index=True)


def _star_frame_table(stack: pd.DataFrame, cid: str, epoch_keys: Sequence[str]) -> pd.DataFrame:
    sub = stack.loc[stack["catalog_id"] == cid]
    by_key: dict[str, pd.Series] = {}
    if not sub.empty:
        for _, row in sub.iterrows():
            by_key[str(row.get("epoch_key") or "")] = row
    rows = []
    for key in epoch_keys:
        rec = by_key.get(str(key))
        if rec is None:
            rows.append(
                {
                    "epoch_key": key,
                    "psf_flux": np.nan,
                    "psf_flux_err": np.nan,
                    "psf_chi2": np.nan,
                    "psf_fit_ok": False,
                    "sat_masked": False,
                    "edge_masked": False,
                    "aperture_flux": np.nan,
                }
            )
            continue
        flux = float(pd.to_numeric(rec.get("dao_flux"), errors="coerce"))
        # EPSF-BLEND-2026-10: a fit whose box lost pixels to the saturation mask is not used in the
        # light curve (wing-only fits on draft 535 injections: +0.04..+0.11 mag), as the aperture
        # light curve drops saturated epochs.
        sat = _coerce_bool(rec.get("psf_sat_masked"))
        # EPSF-FOOTPRINT-2026-10: fit box or sky annulus off the aligned frame's data (constant fill).
        edge = _coerce_bool(rec.get("psf_edge_masked"))
        rows.append(
            {
                "epoch_key": key,
                "psf_flux": np.nan if (sat or edge) else float(pd.to_numeric(rec.get("psf_flux"), errors="coerce")),
                "psf_flux_err": float(pd.to_numeric(rec.get("psf_flux_err"), errors="coerce")),
                "psf_chi2": float(pd.to_numeric(rec.get("psf_chi2"), errors="coerce")),
                "psf_fit_ok": _coerce_bool(rec.get("psf_fit_ok")) and not (sat or edge),
                "sat_masked": sat,
                "edge_masked": edge,
                "aperture_flux": flux,
            }
        )
    return pd.DataFrame(rows)


def _write_csv_with_header(path: Path, header_lines: Sequence[str], df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        for line in header_lines:
            fh.write(str(line).rstrip() + "\n")
        df.to_csv(fh, index=False)


def write_one_internal_psf_lc(
    *,
    target_cid: str,
    stack: pd.DataFrame,
    aperture_lc: pd.DataFrame,
    lc_dir: Path,
    photometry_dir: Path,
    epsf_meta: dict[str, Any],
    gain_value: float,
    gain_source: str,
    git_hash: str,
    git_dirty: str,
) -> Path | None:
    from photometry_lightcurve import ensemble_normalize  # noqa: PLC0415
    from sigma_floor_core import combine_production_err_mag, rel_sigma_to_mag  # noqa: PLC0415

    tid = normalize_gaia_source_id(target_cid)
    if not tid or aperture_lc is None or aperture_lc.empty:
        return None
    ap = aperture_lc.copy()
    if "source_file" not in ap.columns:
        LOGGER.warning("[PSF-LC] %s aperture LC missing source_file", tid)
        return None
    epoch_keys = [str(s).strip() for s in ap["source_file"].tolist()]
    n = len(epoch_keys)
    tgt = _star_frame_table(stack, tid, epoch_keys)
    comp_ids, ens_source = resolve_ensemble_ids(tid, photometry_dir)
    if not comp_ids:
        LOGGER.warning("[PSF-LC] %s has no ensemble membership; skip", tid)
        return None
    measured = comps_with_psf_measurement_on_night(stack)
    comp_ids, dropped_no_psf = filter_ensemble_to_psf_measured(comp_ids, measured)
    if dropped_no_psf:
        LOGGER.info(
            "[PSF-LC] %s dropped %d pinned comp(s) with no PSF measurement on night: %s",
            tid,
            len(dropped_no_psf),
            ",".join(dropped_no_psf),
        )
    if not comp_ids:
        LOGGER.warning(
            "[PSF-LC] %s has no ensemble members with PSF measurement; skip", tid
        )
        return None

    psf_ok = tgt["psf_fit_ok"].to_numpy(dtype=bool)
    psf_flux = tgt["psf_flux"].to_numpy(dtype=np.float64)
    usable = psf_usable_mask(psf_ok, psf_flux)
    target_mag = np.full(n, np.nan, dtype=np.float64)
    target_mag[usable] = _flux_to_inst_mag(psf_flux)[usable]

    comp_mag: dict[str, np.ndarray] = {}
    comp_quality: dict[str, dict] = {}
    for cid in comp_ids:
        ctab = _star_frame_table(stack, cid, epoch_keys)
        cf = ctab["psf_flux"].to_numpy(dtype=np.float64)
        cok = psf_usable_mask(ctab["psf_fit_ok"].to_numpy(dtype=bool), cf)
        mag = np.full(n, np.nan, dtype=np.float64)
        mag[cok] = _flux_to_inst_mag(cf)[cok]
        comp_mag[cid] = mag
        comp_quality[cid] = {"quality": "good"}

    # differential only: catalogue magnitudes are irrelevant (zeros), the calibrated level is not used
    _, psf_delta, ensemble_scatter = ensemble_normalize(
        target_mag,
        comp_mag,
        {cid: 0.0 for cid in comp_ids},
        comp_quality,
    )

    # INV-PSF-LC-PIN-01: full pinned ensemble membership or NaN. No partial ZP.
    drop_reason = [""] * n
    psf_delta = np.asarray(psf_delta, dtype=np.float64)
    for i in range(n):
        missing: list[str] = []
        for cid in comp_ids:
            mag_i = float(comp_mag[cid][i]) if i < len(comp_mag[cid]) else float("nan")
            if not math.isfinite(mag_i):
                missing.append(cid)
        if bool(tgt["sat_masked"].iloc[i]):
            psf_delta[i] = float("nan")
            drop_reason[i] = "target_saturated"
        elif bool(tgt["edge_masked"].iloc[i]):
            psf_delta[i] = float("nan")
            drop_reason[i] = "target_outside_footprint"
        elif not usable[i]:
            # G09-04: the target's own fit failed (psf_fit_ok False / no flux) - an epoch with a reason,
            # not a silent NaN counted as complete
            psf_delta[i] = float("nan")
            drop_reason[i] = "target_fit_not_ok"
        elif missing:
            psf_delta[i] = float("nan")
            drop_reason[i] = f"comp_psf_fail:{missing[0]}"

    # EPSF-K2-2026-10: the aperture light curve's per-epoch second-order extinction term
    # (k'' x (target - comparison colour) x airmass) - it belongs to the star's colour and the epoch,
    # not to the measuring method; without it the PSF light curve keeps a colour x airmass trend.
    k2_term = pd.to_numeric(ap.get("k2_delta_mag"), errors="coerce").to_numpy(dtype=np.float64) \
        if "k2_delta_mag" in ap.columns else np.full(n, np.nan)
    k2_applied = bool(len(k2_term) == n and np.isfinite(k2_term).any())
    if k2_applied:
        psf_delta = np.where(np.isfinite(k2_term), psf_delta - k2_term, psf_delta)
    for i in range(n):
        if not drop_reason[i] and not math.isfinite(float(psf_delta[i])):
            drop_reason[i] = "ensemble_zero_point_nan"  # every NaN epoch has a reason (G09-04)
    n_dropped_pin = int(sum(1 for r in drop_reason if r.startswith("comp_psf_fail")))
    n_target_sat = int(sum(1 for r in drop_reason if r == "target_saturated"))
    n_target_edge = int(sum(1 for r in drop_reason if r == "target_outside_footprint"))
    n_target_fit = int(sum(1 for r in drop_reason if r == "target_fit_not_ok"))
    # G09-04: complete epochs are the finite ones; every NaN epoch carries its reason
    n_full = int(np.count_nonzero(np.isfinite(psf_delta)))
    ap_delta = pd.to_numeric(ap.get("delta_mag"), errors="coerce").to_numpy(dtype=np.float64)
    if len(ap_delta) != n:
        ap_delta = np.full(n, np.nan)
    level_vals = []
    for i in range(n):
        if drop_reason[i]:
            continue
        pdlt = float(psf_delta[i])
        adlt = float(ap_delta[i])
        if math.isfinite(pdlt) and math.isfinite(adlt):
            level_vals.append(pdlt - adlt)
    level_off = float(np.median(level_vals)) if level_vals else float("nan")

    pfe = tgt["psf_flux_err"].to_numpy(dtype=np.float64)
    err_rel = np.full(n, np.nan, dtype=np.float64)
    good_err = usable & np.isfinite(pfe) & (pfe > 0)
    err_rel[good_err] = pfe[good_err] / psf_flux[good_err]
    psf_delta_err = np.full(n, np.nan, dtype=np.float64)
    sc = np.asarray(ensemble_scatter, dtype=np.float64)
    # EPSF-ERR-2026-10: scintillation belongs to the star and the epoch, not to the measuring
    # method - the aperture light curve's per-epoch term is added to the PSF fit error (draft 535:
    # bright comparison stars carry ~4 mmag beyond the PSF fit error, the scintillation level).
    # EPSF-SYS-2026-10: the same holds for the night's systematic floor (flat field / drift of the stars
    # over the chip, SIGMA-SYS-NIGHT measured on the comparison stars): Brno 533 g comparison stars, PSF
    # scatter over the PSF error 5.1 / 2.2 / 1.7 / 1.3 at G 10-12 / 12-13 / 13-14 / > 14 without it.
    def _ap_rel(col: str) -> np.ndarray:
        if col in ap.columns:
            return pd.to_numeric(ap[col], errors="coerce").to_numpy(dtype=np.float64)
        return np.zeros(n, dtype=np.float64)

    scint_rel = _ap_rel("err_scint_rel")
    sys_rel = _ap_rel("err_sigma_sys_rel")
    for i in range(n):
        if not usable[i] or not math.isfinite(float(err_rel[i])):
            continue
        if drop_reason[i]:
            continue
        sem = float(sc[i]) if i < len(sc) and math.isfinite(float(sc[i])) else 0.0
        scint_mag = rel_sigma_to_mag(float(scint_rel[i])) if math.isfinite(float(scint_rel[i])) else 0.0
        sys_mag = rel_sigma_to_mag(float(sys_rel[i])) if math.isfinite(float(sys_rel[i])) else 0.0
        psf_delta_err[i] = combine_production_err_mag(
            float(err_rel[i]), sem, sigma_sys_mag=sys_mag, sigma_scint_mag=scint_mag
        )

    ap_flux = tgt["aperture_flux"].to_numpy(dtype=np.float64)
    ratio = np.full(n, np.nan, dtype=np.float64)
    both = usable & np.isfinite(ap_flux) & (ap_flux > 0)
    ratio[both] = psf_flux[both] / ap_flux[both]

    header = build_provenance_header(
        epsf_meta=epsf_meta,
        gain_value=gain_value,
        gain_source=gain_source,
        n_comp=len(comp_ids),
        pinned_ids=comp_ids,
        ensemble_source=ens_source,
        git_hash=git_hash,
        git_dirty=git_dirty,
        n_epochs_full=n_full,
        n_epochs_dropped_pin=n_dropped_pin,
        psf_ap_level_offset_mag=level_off,
        extra_header_lines=[f"# psf_lc_k2_applied={str(k2_applied).lower()}",
                             f"# psf_lc_time_base={_time_base_of(ap)}",
                             f"# psf_lc_n_epochs_target_saturated={n_target_sat}",
                             f"# psf_lc_n_epochs_target_outside_footprint={n_target_edge}",
                             f"# psf_lc_n_epochs_target_fit_not_ok={n_target_fit}"],
        ensemble_dropped_no_psf=dropped_no_psf,
    )

    def _num(arr: np.ndarray, nd: int) -> list[Any]:
        out: list[Any] = []
        for v in np.asarray(arr, dtype=np.float64):
            if math.isfinite(float(v)):
                out.append(round(float(v), nd))
            else:
                out.append(float("nan"))
        return out

    bjd = pd.to_numeric(ap.get("bjd"), errors="coerce") if "bjd" in ap.columns else pd.Series([np.nan] * n)
    hjd = pd.to_numeric(ap.get("hjd"), errors="coerce") if "hjd" in ap.columns else pd.Series([np.nan] * n)
    jd = pd.to_numeric(ap.get("jd"), errors="coerce") if "jd" in ap.columns else pd.Series([np.nan] * n)

    out = pd.DataFrame(
        {
            "bjd": _num(np.asarray(bjd, dtype=np.float64), 10),
            "hjd": _num(np.asarray(hjd, dtype=np.float64), 10),
            "jd": _num(np.asarray(jd, dtype=np.float64), 10),
            "source_file": epoch_keys,
            "psf_fit_ok": [bool(v) for v in psf_ok],
            "psf_flux": _num(psf_flux, 6),
            "psf_flux_err": _num(pfe, 6),
            "psf_chi2": _num(tgt["psf_chi2"].to_numpy(dtype=np.float64), 6),
            "psf_delta_mag": _num(psf_delta, 6),
            "psf_delta_mag_err": _num(psf_delta_err, 6),
            "psf_ap_ratio": _num(ratio, 6),
            "psf_epoch_drop_reason": drop_reason,
        }
    )
    out_path = lc_csv_path(lc_dir, tid, "psf")
    _write_csv_with_header(out_path, header, out)
    return out_path


def write_internal_psf_lightcurves(
    *,
    platesolve_dir: Path | str,
    frames_root: Path | str,
    photometry_dir: Path | str | None = None,
    target_ids: Iterable[str],
    output_directory: Path | str | None = None,
) -> dict[str, Any]:
    """Write ``lightcurve_<id>_psf.csv`` for the given aperture LC targets. Additive only."""
    ps = Path(platesolve_dir)
    root = Path(frames_root)
    phot = Path(photometry_dir) if photometry_dir is not None else ps / "photometry"
    lc_dir = phot / "lightcurves"
    if not lc_dir.is_dir():
        raise FileNotFoundError(f"Aperture LC directory missing: {lc_dir}")
    out_dir = Path(output_directory) if output_directory is not None else lc_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    epsf_meta = load_epsf_build_meta(ps)
    gain_value, gain_source = _load_gain_authority(phot)
    git_hash, git_dirty = _git_provenance()
    stack = _read_proc_stack(root)

    target_ids = [normalize_gaia_source_id(t) for t in target_ids if normalize_gaia_source_id(t)]

    written: list[str] = []
    skipped: list[str] = []
    for tid in target_ids:
        ap_path = lc_csv_path(lc_dir, tid, "aperture")
        if not ap_path.is_file():
            skipped.append(tid)
            continue
        try:
            ap_lc = pd.read_csv(ap_path, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PSF-LC] cannot read aperture LC %s: %s", ap_path.name, exc)
            skipped.append(tid)
            continue
        path = write_one_internal_psf_lc(
            target_cid=tid,
            stack=stack,
            aperture_lc=ap_lc,
            lc_dir=out_dir,
            photometry_dir=phot,
            epsf_meta=epsf_meta,
            gain_value=gain_value,
            gain_source=gain_source,
            git_hash=git_hash,
            git_dirty=git_dirty,
        )
        if path is not None:
            written.append(str(path))
        else:
            skipped.append(tid)

    LOGGER.info("[PSF-LC] wrote %d internal PSF LC file(s); skipped %d", len(written), len(skipped))
    return {
        "written": written,
        "skipped": skipped,
        "n_written": len(written),
        "n_skipped": len(skipped),
        "lc_dir": str(lc_dir),
        "output_directory": str(out_dir),
    }
