"""Moved from pipeline.py (CONSOLIDATE-01E6a). Facade re-exports these names.

Alignment orchestration, plate solve, masterstar build, VSX/exo merge.
The four giants stay in pipeline.py this wave (E6b).
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Sequence
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import pandas as pd
from config import AppConfig
from database import DraftTechnicalMetadataError, VyvarDatabase, _db_header_pixel_native_um_mean, query_local_gaia_by_source_ids, query_local_vsx
from photometry import common_field_intersection_bbox_px
from gaia_catalog_id import catalog_id_series_for_masterstars_export, normalize_gaia_source_id, read_vyvar_csv
from infolog import log_event
from optics_selection import resolve_optics_ids_for_platesolve
from utils import fits_binning_xy_from_header, iter_fits_paths_recursive as _iter_fits_recursive, maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel, normalize_telescope_focal_mm_for_plate_scale, plate_scale_arcsec_per_pixel, plate_solve_fov_deg_diagonal_from_scale, strip_celestial_wcs_keys, strip_vendor_platesolve_metadata
import itertools
import fits_meta
from fits_meta import _focal_mm_plausible
from masterstar_gaia_accounting import _dao_xy_binned_to_full
from masterstar_gaia_accounting import _dao_full_to_binned_xy
from pipeline_constants import _EXO_HOST_ANNOTATION_COLUMNS
from pipeline_calibrate import (
    _archive_preprocess_lights_root,
    _filter_light_paths_maybe,
    _vyvar_parallel_worker_count,
)
from pipeline_preprocess import (
    _partition_detrended_by_subfolder,
    filter_files_by_qc_metrics_allowlist,
)

# Same named logger as pipeline.LOGGER (logging.getLogger singleton).
# Avoids pipeline -> pipeline_astrometry -> pipeline at module load.
LOGGER = logging.getLogger("pipeline")

# Aperturna fotometria Faz 0-2A (active_targets.csv, zone_flag, skip_photometry) je v ``photometry_core``
# - ``run_phase0_and_phase1`` / ``run_phase2a``, nie v tomto subore.

_EPSF_SKIP_LOGGED: set[str] = set()


def _photometry_mode_run_flags(
    cfg: AppConfig | None = None,
    *,
    platesolve_dir: Path | str | None = None,
) -> tuple[bool, bool]:
    """Return ``(_run_aperture, _run_epsf)`` from ``photometry_mode`` (and ePSF file presence)."""
    _cfg = cfg if cfg is not None else AppConfig()
    _phot_mode = str(getattr(_cfg, "photometry_mode", "both")).lower().strip()
    if _phot_mode not in ("aperture", "epsf", "both"):
        _phot_mode = "both"
    _run_aperture = _phot_mode in ("aperture", "both")
    _run_epsf = _phot_mode in ("epsf", "both")
    if _run_epsf and platesolve_dir is not None:
        _epsf_fits = Path(platesolve_dir) / "masterstar_epsf.fits"
        if not _epsf_fits.is_file():
            _ps_key = str(Path(platesolve_dir).resolve())
            if _ps_key not in _EPSF_SKIP_LOGGED:
                _EPSF_SKIP_LOGGED.add(_ps_key)
                logging.getLogger("pipeline").info(
                    "photometry_mode includes epsf but no ePSF model found at %s - skipping PSF photometry",
                    _epsf_fits,
                )
            _run_epsf = False
    return _run_aperture, _run_epsf


def _export_catalog_psf_st_fields(cfg: AppConfig, platesolve_dir: Path) -> dict[str, Any]:
    """ePSF model path and PSF toggles for per-frame catalog workers (TODO-8 Phase 2B)."""
    _epsf_fits = Path(platesolve_dir) / "masterstar_epsf.fits"
    _run_aperture, _run_epsf = _photometry_mode_run_flags(cfg, platesolve_dir=platesolve_dir)
    return {
        "epsf_model_path": str(_epsf_fits) if _epsf_fits.is_file() else "",
        "psf_photometry_enabled": bool(cfg.psf_photometry_enabled),
        "photometry_mode": str(getattr(cfg, "photometry_mode", "both")),
        "_run_aperture": bool(_run_aperture),
        "_run_epsf": bool(_run_epsf),
        "gain": float(cfg.gain),
        "read_noise": float(cfg.read_noise),
        "psf_ac_policy": "p4_none",
    }


def _pipeline_ui_info(msg: str) -> None:
    """Log always; during a long job route to the bottom footer instead of ``st.info``."""
    log_event(msg)
    try:
        import streamlit as st

        fs = st.session_state.get("vyvar_footer_state")
        if isinstance(fs, dict) and fs.get("running"):
            st.session_state["vyvar_footer_state"] = {**fs, "status_detail": str(msg)[:800]}
            _fn = st.session_state.get("vyvar_ui_rerender_footer")
            if callable(_fn):
                _fn()
                return
        st.info(msg)
    except Exception:  # noqa: BLE001
        pass


def _ensure_parent_dirs_for_aligned_fits(out_path: Path) -> None:
    """Ensure parent folders exist for nested outputs under ``detrended_aligned/lights/...``."""
    os.makedirs(str(out_path.parent), exist_ok=True)


def _assert_alignment_produced_fits(aligned_root: Path) -> None:
    """Fail fast if alignment wrote no FITS under the group folder (recursive; not only top-level)."""
    n = len(_iter_fits_recursive(aligned_root))
    if n == 0:
        raise RuntimeError(
            "Alignment zlyhal - nenasli sa ziadne vystupne subory! "
            f"(ziadne FITS v {aligned_root.resolve()} vratane podadresarov.)"
        )


def draft_obs_group_count(archive_path: Path | str) -> int:
    """Number of obs_group subdirs under draft lights root that contain FITS."""
    from draft_provenance import draft_archive_root, resolve_draft_lights_root

    ap = draft_archive_root(Path(archive_path).expanduser())
    pl = resolve_draft_lights_root(ap)
    if not pl.is_dir():
        pl = ap / "processed" / "lights"
    if not pl.is_dir():
        return 0
    return sum(
        1
        for d in pl.iterdir()
        if d.is_dir() and any(_iter_fits_recursive(d))
    )


def draft_is_multi_group_obs(archive_path: Path | str) -> bool:
    return int(draft_obs_group_count(archive_path)) > 1


def resolve_masterstar_input_root(
    archive_path: Path | str,
    setup_name: str | None = None,
    *,
    app_config: AppConfig | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> Path | None:
    """Pick MASTERSTAR input root under draft lights[/setup_name] with robust fallback.

    Uses ``calibrated/lights`` or ``non_calibrated/lights`` (via :func:`resolve_draft_lights_root`).

    When ``setup_name`` is explicitly provided, only that subdir is considered - no cross-group scan.
    Returns ``None`` when the requested setup is missing/empty (multi-group safe).
    """
    from draft_provenance import draft_archive_root, resolve_draft_lights_root

    ap = draft_archive_root(Path(archive_path).expanduser())
    lights_root = resolve_draft_lights_root(ap, draft_id=draft_id, db=db)
    setup_key = str(setup_name or "").strip()

    def _pick_under(base: Path) -> Path | None:
        if setup_key:
            cand = base / setup_key
            if cand.is_dir() and any(_iter_fits_recursive(cand)):
                return cand
            log_event(
                f"WARN: MASTERSTAR input root for setup {setup_key!r} not found under {base}; "
                "refusing cross-group fallback"
            )
            return None
        if base.is_dir():
            subdirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.name.casefold())
            for sd in subdirs:
                if any(_iter_fits_recursive(sd)):
                    return sd
            if any(_iter_fits_recursive(base)):
                return base
        return None

    hit = _pick_under(lights_root)
    if hit is not None:
        return hit
    return lights_root if not setup_key else None


def _path_segments_forbidden_for_masterstar_physical_source(
    p: Path,
    *,
    pre_calibrated: bool = False,
) -> bool:
    """``True`` if resolved path must not be used as MASTERSTAR physical source.

    In VYVAR-calibrated mode, ``non_calibrated/`` and ``raw/`` are forbidden (use processed/calibrated).
    In pre-calibrated mode, ``non_calibrated/lights/`` **is** the valid source tree; only ``raw/`` is forbidden.
    """
    try:
        parts = Path(p).resolve().parts
    except OSError:
        parts = Path(p).parts
    bad = {"raw"} if pre_calibrated else {"raw", "non_calibrated"}
    return any(seg.casefold() in bad for seg in parts)


def _path_is_under_tree(root: Path, p: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _pick_preferred_masterstar_basename_hit(
    hits: list[Path],
    *,
    pre_calibrated: bool = False,
) -> Path | None:
    """Pri viacerych zhodach basename zvol radsej ``proc_*.fits`` (spracovany snimok)."""
    if not hits:
        return None
    clean = [
        h
        for h in hits
        if not _path_segments_forbidden_for_masterstar_physical_source(h, pre_calibrated=pre_calibrated)
    ]
    if not clean:
        return None
    proc_first = [h for h in clean if h.name.casefold().startswith("proc_")]
    use = proc_first if proc_first else clean
    return sorted(use, key=lambda x: str(x).casefold())[0]


def _header_vy_fwhm_px(hdr: fits.Header | None) -> float | None:
    """Measured QC FWHM from ``VY_FWHM`` if present and sane."""
    if hdr is None or "VY_FWHM" not in hdr:
        return None
    try:
        v = float(hdr["VY_FWHM"])
        if math.isfinite(v) and 0.5 < v < 80.0:
            return float(v)
    except (TypeError, ValueError):
        return None
    return None


def _sort_masterstar_paths_by_fwhm(
    files: list[Path],
    *,
    fwhm_by_basename: dict[str, float] | None = None,
) -> list[Path]:
    """Najlepsie prve: najnizsi VY_FWHM (hlavicka alebo DB mapa), nezname na koniec."""
    _fb = fwhm_by_basename or {}
    scored: list[tuple[float, str, Path]] = []
    for fp in files:
        v: float | None = None
        try:
            with fits.open(fp, memmap=False) as h:
                v = _header_vy_fwhm_px(h[0].header)
        except Exception:  # noqa: BLE001
            pass
        if v is None:
            _n = fp.name.casefold()
            vv = _fb.get(_n)
            if vv is None and _n.startswith("proc_"):
                vv = _fb.get(_n[5:])
            if vv is None and not _n.startswith("proc_"):
                vv = _fb.get(f"proc_{_n}")
            if vv is not None and math.isfinite(vv) and vv > 0:
                v = float(vv)
        score = float(v) if v is not None and math.isfinite(v) and v > 0 else float("inf")
        scored.append((score, str(fp).casefold(), fp))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in scored]


def _strip_external_platesolve_header(hdr: fits.Header) -> None:
    """Drop celestial WCS and common third-party plate-solve keywords (ASTAP, astrometry.net, ...).

    VYVAR must establish astrometry via :func:`vyvar_platesolver.solve_wcs_with_local_gaia` only.
    """
    strip_celestial_wcs_keys(hdr)
    strip_vendor_platesolve_metadata(hdr)
    for _k in (
        "WCSAXES",
        "WCSDIM",
        "CROTA1",
        "CROTA2",
        "WCSNAME",
        "VY_PSOLV",
        "VY_SIPRF",
    ):
        try:
            del hdr[_k]
        except KeyError:
            pass


def build_masterstar_from_detrended(
    *,
    detrended_root: Path,
    output_fits: Path,
    only_paths: "Sequence[Path | str] | None" = None,
    fwhm_fallback_px: float | None = None,
    app_config: AppConfig | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
    pre_calibrated: bool = False,
) -> dict[str, Any]:
    """Build MASTERSTAR by copying the single best processed FITS (lowest VY_FWHM)."""
    import shutil

    import numpy as np

    root = Path(detrended_root).resolve()
    _forbid_kw = {"pre_calibrated": bool(pre_calibrated)}
    all_fits = [
        fp
        for fp in _iter_fits_recursive(root)
        if _path_is_under_tree(root, fp)
        and not _path_segments_forbidden_for_masterstar_physical_source(fp, **_forbid_kw)
    ]
    files = _filter_light_paths_maybe(all_fits, only_paths)
    if not files and only_paths is not None:
        remapped: list[Path] = []
        seen_r: set[str] = set()
        for op in only_paths:
            hit = _resolve_best_effort_path_under(root, str(op), pre_calibrated=pre_calibrated)
            if hit is None or not _path_is_under_tree(root, hit):
                continue
            if _path_segments_forbidden_for_masterstar_physical_source(hit, **_forbid_kw):
                log_event(f"MASTERSTAR: mapovanie zahodilo RAW cestu -> {hit}")
                continue
            try:
                rk = str(hit.resolve()).casefold()
            except OSError:
                rk = str(hit).casefold()
            if rk in seen_r:
                continue
            seen_r.add(rk)
            remapped.append(hit)
        if remapped:
            files = remapped
            log_event(
                f"MASTERSTAR: vyber zluceny cez best-effort mapovanie ({len(files)} FITS; path filter bol prazdny)."
            )
    if files:
        if only_paths is None:
            _pipeline_ui_info(
                f"Najdenych {len(files)} suborov v {root} (vratane podadresarov)."
            )
        else:
            _pipeline_ui_info(
                f"Najdenych {len(files)} suborov pre MASTERSTAR v {root} (vyber kandidatov; "
                f"v strome {len(all_fits)} FITS celkom)."
            )
    if not files:
        if only_paths is not None:
            try:
                _want = ", ".join(Path(str(x)).name for x in only_paths[:8])
            except Exception:  # noqa: BLE001
                _want = str(only_paths)
            msg = (
                f"MASTERSTAR: explicitny vyber sa nezhoduje so subormi pod {root}: {_want}"
                + (" ..." if len(list(only_paths)) > 8 else "")
            )
            log_event(msg)
            raise FileNotFoundError(msg)
        # Cely strom (bez filtra kandidatov): deterministicky maly batch z disku.
        log_event("MASTERSTAR: bez filtra kandidatov - beriem prvych N FITS z priecinka.")
        batch = sorted(
            (
                fp
                for fp in _iter_fits_recursive(root)
                if _path_is_under_tree(root, fp)
                and not _path_segments_forbidden_for_masterstar_physical_source(fp, **_forbid_kw)
            ),
            key=lambda p: str(p).casefold(),
        )
        if batch:
            n_take = max(1, min(8, len(batch)))
            files = batch[:n_take]
            all_fits = batch

    if not files:
        if not all_fits:
            msg = (
                f"Nenasli sa ziadne FITS subory v {root} (prehladavana cesta, vratane podadresarov)."
            )
        else:
            msg = (
                f"Ziadne FITS pre MASTERSTAR po vybere kandidatov: {root} obsahuje {len(all_fits)} subor(ov), "
                "ale ziadna cesta nezodpoveda vyberu z databazy."
            )
        _pipeline_ui_info(msg)
        raise FileNotFoundError(msg)
    try:
        _comp_names = [Path(p).name for p in files]
        log_event(f"[folder] MASTERSTAR COMPOSITION: Using [{', '.join(_comp_names)}]")
    except Exception:  # noqa: BLE001
        pass

    sorted_files = _sort_masterstar_paths_by_fwhm(files, fwhm_by_basename=None)

    output_fits.parent.mkdir(parents=True, exist_ok=True)
    _cfg_ms = app_config or AppConfig()

    reference_path = Path(sorted_files[0])
    if _path_segments_forbidden_for_masterstar_physical_source(
        reference_path, **_forbid_kw
    ) or not _path_is_under_tree(root, reference_path):
        raise FileNotFoundError(
            f"MASTERSTAR: odmietnuty zdroj mimo lights stromu alebo z RAW: {reference_path}"
        )
    if not reference_path.exists():
        log_event(f"[X] MASTERSTAR FAIL: Reference file {reference_path} not found.")
        fallback_hits = [
            x
            for x in _iter_fits_recursive(root)
            if x.name == reference_path.name
            and _path_is_under_tree(root, x)
            and not _path_segments_forbidden_for_masterstar_physical_source(x, **_forbid_kw)
        ]
        if fallback_hits:
            reference_path = _pick_preferred_masterstar_basename_hit(fallback_hits) or fallback_hits[0]
            log_event(f"[OK] MASTERSTAR fallback reference found: {reference_path}")
        else:
            raise FileNotFoundError(f"MASTERSTAR reference file not found: {reference_path}")

    best_frame_fwhm_px: float | None = None
    try:
        from astropy.io.fits import getheader

        _ref_hdr = getheader(str(reference_path))
        _bfw = float(_ref_hdr.get("VY_FWHM", 0))
        if 0.5 < _bfw < 80.0:
            best_frame_fwhm_px = float(_bfw)
    except Exception:  # noqa: BLE001
        best_frame_fwhm_px = None

    try:
        with fits.open(reference_path, memmap=False) as hdul0:
            _d0 = hdul0[0].data
            _sh = getattr(_d0, "shape", None)
            if _d0 is None or _sh is None or len(_sh) != 2:
                raise ValueError("MASTERSTAR: referencny FITS nie je platny 2D primary.")
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"MASTERSTAR: neviem nacitat referencny FITS: {exc}") from exc

    shutil.copy2(reference_path, output_fits)
    if len(files) <= 1:
        _ms_pick_msg = "jediny kandidat"
    else:
        _ms_pick_msg = f"kandidatov {len(files)}; najlepsi podla VY_FWHM"
    log_event(f"MASTERSTAR: cista kopia -> {output_fits} (zdroj {reference_path.name}, {_ms_pick_msg}).")

    # Auto FWHM: median VY_FWHM zo vsetkych processed FITS v root sade
    _all_processed = list(_iter_fits_recursive(root))
    _fwhm_auto_values: list[float] = []
    for _fp_fw in _all_processed:
        try:
            with fits.open(_fp_fw, memmap=False) as _hf_fw:
                _v_fw = _header_vy_fwhm_px(_hf_fw[0].header)
                if _v_fw is not None and 1.0 < _v_fw < 15.0:
                    _fwhm_auto_values.append(float(_v_fw))
        except Exception:  # noqa: BLE001
            pass
    if _fwhm_auto_values:
        _fwhm_auto = float(np.median(np.asarray(_fwhm_auto_values, dtype=np.float64)))
        log_event(
            f"MASTERSTAR: auto FWHM z {len(_fwhm_auto_values)} processed FITS "
            f"= {_fwhm_auto:.3f} px (median VY_FWHM)"
        )
    else:
        _fwhm_auto = float(fwhm_fallback_px) if fwhm_fallback_px is not None else 4.5
        log_event(f"MASTERSTAR: VY_FWHM nedostupne - fallback FWHM = {_fwhm_auto:.1f} px")

    try:
        with fits.open(output_fits, mode="update", memmap=False) as h:
            hdr = h[0].header
            # Vzdy prepis VY_FWHM hodnotou z auto vypoctu (median sady)
            vy_fwhm = float(_fwhm_auto)
            hdr["VY_FWHM"] = (vy_fwhm, "FWHM [pix] auto z medianu processed FITS")
            log_event(f"MASTERSTAR: VY_FWHM = {vy_fwhm:.3f} px zapisany do FITS.")
            # VY_FWHM_GAUSS sa doplni po plate-solve (2D Gaussian fit na MASTERSTAR).

            _strip_external_platesolve_header(hdr)
            log_event(
                "MASTERSTAR: z MASTERSTAR kopie odstraneny externy WCS/plate-solve metadata "
                "(ASTAP, astrometry.net, ...) - astrometriu nastavi vyhradne VYVAR Gaia solver."
            )

            h.flush()
    except Exception as _exc:  # noqa: BLE001
        log_event(f"MASTERSTAR: VY_FWHM zapis zlyhal: {_exc!s}")

    return {
        "masterstar_path": str(output_fits),
        "frames_used": int(len(files)),
        "reference_path": str(reference_path),
        "reference_index": 0,
        "stacked": False,
        "frames_combined": 1,
        "copied_from": str(reference_path),
        "best_frame_fwhm_px": best_frame_fwhm_px,
    }


def _update_masterstar_obs_file_status(
    *,
    cfg: AppConfig | None,
    draft_id: int | None,
    selected_ref_path: Path | None,
    wcs_ok: bool,
    n_stars: int,
) -> None:
    if draft_id is None or selected_ref_path is None:
        return
    try:
        from draft_provenance import load_or_init_manifest, patch_draft_manifest, resolve_draft_dir_for_id

        app_cfg = cfg or AppConfig()
        db = VyvarDatabase(Path(app_cfg.database_path))
        ar = getattr(app_cfg, "archive_root", None)
        if ar:
            db._archive_root_override = Path(str(ar)).expanduser().resolve()
        try:
            root = resolve_draft_dir_for_id(db, int(draft_id))
            if root is None:
                return
            manifest = load_or_init_manifest(root, int(draft_id))
            files = manifest.get("files")
            if not isinstance(files, list):
                return
            ref_name = selected_ref_path.name
            ref_l = ref_name.casefold()
            raw_l = ref_l[5:] if ref_l.startswith("proc_") else ref_l
            proc_l = raw_l if raw_l.startswith("proc_") else f"proc_{raw_l}"
            updated = 0
            for i, entry in enumerate(files):
                if not isinstance(entry, dict):
                    continue
                fp = str(entry.get("file_path") or "")
                fp_l = Path(fp).name.casefold()
                if not (
                    fp_l == raw_l
                    or fp_l == proc_l
                    or fp_l == ref_l
                    or raw_l in fp_l
                    or proc_l in fp_l
                    or ref_l in fp_l
                ):
                    continue
                patched = dict(entry)
                insp = dict(patched.get("inspection") or {})
                insp["wcs"] = 1 if bool(wcs_ok) else 0
                insp["stars"] = int(max(0, int(n_stars)))
                patched["inspection"] = insp
                files[i] = patched
                updated += 1
            if updated:
                patch_draft_manifest(root, int(draft_id), files=files)
            log_event(
                f"MASTERSTAR manifest update: DRAFT_ID={int(draft_id)}, WCS={1 if wcs_ok else 0}, "
                f"Stars={int(n_stars)}, rows={int(updated)}"
            )
        finally:
            db.conn.close()
    except Exception as exc:  # noqa: BLE001
        log_event(f"MASTERSTAR manifest update skipped: {exc!s}")


def _resolve_best_effort_path_under(
    root: Path,
    raw_path: str,
    *,
    pre_calibrated: bool = False,
) -> Path | None:
    """Map an manifest files[].FILE_PATH (often archived raw path) to an existing file under ``root``.

    Strategy: exact relative join, else basename match (first hit). This is intentionally heuristic because
    imports may store different path bases (absolute vs relative, calibrated vs processed).
    """
    _forbid_kw = {"pre_calibrated": bool(pre_calibrated)}
    rp = str(raw_path or "").strip()
    if not rp:
        return None
    p = Path(rp)
    # If already absolute and exists under root tree, accept.
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(root.resolve())
            cand = (root / rel).resolve()
            if (
                cand.is_file()
                and _path_is_under_tree(root, cand)
                and not _path_segments_forbidden_for_masterstar_physical_source(cand, **_forbid_kw)
            ):
                return cand
        except Exception:  # noqa: BLE001
            pass
    # If relative, try directly.
    cand2 = (root / p).resolve()
    if (
        cand2.is_file()
        and _path_is_under_tree(root, cand2)
        and not _path_segments_forbidden_for_masterstar_physical_source(cand2, **_forbid_kw)
    ):
        return cand2
    # Explicit processed-name fallback in the same folder (e.g. Light_066.fits -> proc_Light_066.fits).
    if not pre_calibrated:
        try:
            cand2_proc = cand2.with_name(_safe_proc_name(cand2.name)).resolve()
            if (
                cand2_proc.is_file()
                and _path_is_under_tree(root, cand2_proc)
                and not _path_segments_forbidden_for_masterstar_physical_source(cand2_proc, **_forbid_kw)
            ):
                return cand2_proc
        except Exception:  # noqa: BLE001
            pass
    # Basename / fuzzy suffix fallback (handles prefixes like ``proc_``).
    name = p.name
    if not name:
        return None
    _name_cf = name.casefold()
    _name_noproc = _name_cf[5:] if _name_cf.startswith("proc_") else _name_cf
    hits = []
    for x in _iter_fits_recursive(root):
        xn = x.name.casefold()
        xn_noproc = xn[5:] if xn.startswith("proc_") else xn
        if (
            xn == _name_cf
            or xn_noproc == _name_noproc
            or xn.endswith(_name_cf)
            or xn_noproc.endswith(_name_noproc)
        ):
            if not _path_segments_forbidden_for_masterstar_physical_source(x, **_forbid_kw):
                hits.append(x)
    return _pick_preferred_masterstar_basename_hit(hits, pre_calibrated=pre_calibrated)


def get_masterstar_candidate_rows(
    draft_id: int,
    percentage: float,
    *,
    fwhm_max_px: float | None = None,
    db: VyvarDatabase,
) -> "pd.DataFrame":
    """Rank draft light frames by quality metrics for MASTERSTAR selection.

    Strict filter: include only rows with ``IS_REJECTED`` 0/NULL.
    Score (higher is better) is a normalized version of:

    Score = (1 / fwhm) * (1 / sky_level) * snr_estimate
    where ``snr_estimate`` is approximated from ``STAR_COUNT`` and ``SKY_LEVEL`` when no explicit SNR exists.
    """
    import numpy as np
    import pandas as pd

    did = int(draft_id)
    pct = float(percentage)
    pct = float(max(0.1, min(100.0, pct)))

    rows = db.fetch_draft_light_rows_for_quality(did)
    if not rows:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    # Strict: IS_REJECTED == 0 (or NULL)
    if "IS_REJECTED" in df.columns:
        df["IS_REJECTED"] = pd.to_numeric(df["IS_REJECTED"], errors="coerce").fillna(0).astype(int)
        df = df[df["IS_REJECTED"] == 0].copy()
    lim_active = bool(fwhm_max_px is not None and float(fwhm_max_px) > 0)
    lim_v = float(fwhm_max_px) if lim_active else 0.0
    if lim_active:
        _f = pd.to_numeric(df.get("FWHM"), errors="coerce")
        df = df[_f.notna() & (_f <= lim_v)].copy()
    if df.empty:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    df["FWHM"] = pd.to_numeric(df.get("FWHM"), errors="coerce")
    df["SKY_LEVEL"] = pd.to_numeric(df.get("SKY_LEVEL"), errors="coerce")
    df["STAR_COUNT"] = pd.to_numeric(df.get("STAR_COUNT"), errors="coerce").fillna(0).astype(int)

    # Robust fallbacks: median scales for normalization.
    f_med = float(np.nanmedian(df["FWHM"].values)) if np.isfinite(np.nanmedian(df["FWHM"].values)) else 0.0
    s_med = (
        float(np.nanmedian(df["SKY_LEVEL"].values))
        if np.isfinite(np.nanmedian(df["SKY_LEVEL"].values))
        else 0.0
    )
    if not (math.isfinite(f_med) and f_med > 0):
        f_med = 1.0
    if not (math.isfinite(s_med) and s_med > 0):
        s_med = 1.0

    eps = 1e-9
    f = df["FWHM"].astype(float)
    sky = df["SKY_LEVEL"].astype(float)
    stars = df["STAR_COUNT"].astype(float)

    # snr_estimate: more stars and lower sky -> better SNR (proxy; true SNR not stored in manifest files[]).
    snr_est = (stars + 1.0) / np.sqrt(np.maximum(sky, 0.0) + 1.0)

    f_norm_inv = f_med / (np.maximum(f, 0.0) + eps)
    sky_norm_inv = s_med / (np.maximum(sky, 0.0) + eps)

    score = f_norm_inv * sky_norm_inv * snr_est
    score = np.where(np.isfinite(score), score, 0.0)

    df["SNR_EST"] = snr_est
    df["SCORE"] = score

    df = df.sort_values(["SCORE", "STAR_COUNT"], ascending=[False, False], kind="mergesort").reset_index(
        drop=True
    )

    k = int(max(1, math.ceil(len(df) * (pct / 100.0))))
    return df.head(k).loc[:, ["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"]]


def get_masterstar_candidates(draft_id: int, percentage: float, *, db: VyvarDatabase) -> list[str]:
    """Return FILE_PATH list of top-ranked MASTERSTAR candidates (temporary list; manifest files[] is not modified)."""
    df = get_masterstar_candidate_rows(int(draft_id), float(percentage), db=db)
    if df.empty:
        return []
    return [str(x) for x in df["FILE_PATH"].tolist() if str(x).strip()]


def _vyvar_open_database(cfg: AppConfig) -> VyvarDatabase | None:
    try:
        return VyvarDatabase(Path(cfg.database_path))
    except Exception:  # noqa: BLE001
        return None


def _header_focal_length_mm(header: fits.Header) -> float | None:
    """Focal length [mm] from common FITS keys (``FOCALLEN`` / ``FOCLEN`` often in **metres**)."""
    for key in ("FOCALLEN", "FOCLEN", "TELFOCA", "FOCAL_LEN", "FOCALL", "FOC_LEN"):
        if key not in header or header[key] in (None, "", " ", "0", 0):
            continue
        try:
            v = float(header[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v <= 0:
            continue
        mm = v * 1000.0 if v < 25.0 else v
        if 40.0 <= mm <= 120_000.0:
            return float(mm)
    return None


def resolve_plate_solve_fov_deg_hint(
    hdr: fits.Header,
    h: int,
    w: int,
    *,
    database_path: Path | str | None = None,
    equipment_id: int | None = None,
    draft_id: int | None = None,
) -> float | None:
    """Estimate plate-solve FOV diameter [deg] along chip diagonal (optics from header, else DB scale + NAXIS)."""
    if h <= 0 or w <= 0:
        return None
    f_mm = _header_focal_length_mm(hdr)
    p_um = _db_header_pixel_native_um_mean(hdr)
    if f_mm is not None and p_um is not None and f_mm > 0 and p_um > 0:
        diag_mm = math.hypot(float(w) * float(p_um) * 0.001, float(h) * float(p_um) * 0.001)
        rad = 2.0 * math.atan2(0.5 * diag_mm, float(f_mm))
        if math.isfinite(rad) and rad > 0:
            return float(rad * 180.0 / math.pi)

    dbp = str(database_path or "").strip()
    if not dbp:
        return None
    try:
        db = VyvarDatabase(Path(dbp))
    except Exception:  # noqa: BLE001
        return None
    try:
        eq = int(equipment_id) if equipment_id is not None else None
        tel: int | None = None
        if draft_id is not None:
            dr = db.fetch_obs_draft_by_id(int(draft_id))
            if dr is not None:
                if eq is None and dr.get("ID_EQUIPMENTS") is not None:
                    eq = int(dr["ID_EQUIPMENTS"])
                if dr.get("ID_TELESCOPE") is not None:
                    tel = int(dr["ID_TELESCOPE"])
        xb, yb = fits_binning_xy_from_header(hdr)
        bin_b = max(1, int(xb), int(yb))
        sc = compute_plate_scale_from_db(eq, tel, db.conn, binning=bin_b)
        if sc is None or not math.isfinite(float(sc)) or float(sc) <= 0:
            return None
        nx = int(hdr.get("NAXIS1", w) or w)
        ny = int(hdr.get("NAXIS2", h) or h)
        return plate_solve_fov_deg_diagonal_from_scale(nx, ny, float(sc))
    except Exception:  # noqa: BLE001
        # EXC-0309: T4 -- db.conn.close failure in FOV hint finally block is ignored after the hint is computed o... (EXCEPT-BULK 2026-07-08)
        return None
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass


def _resolve_focal_mm_for_plate_scale(
    header: fits.Header | None,
    db: VyvarDatabase | None,
    *,
    telescope_id: int | None = None,
) -> tuple[float | None, str]:
    """Plausible FITS focal first; else ``TELESCOPE.FOCAL``."""
    if header is not None:
        hdr_mm = _header_focal_length_mm(header)
        if hdr_mm is not None:
            hdr_n, hfixed = normalize_telescope_focal_mm_for_plate_scale(hdr_mm)
            if hfixed:
                log_event(
                    f"FOCAL: v hlavicke FITS ohnisko vyzeralo ako 10x preklep ({hdr_mm:g} mm -> {hdr_n:g} mm)."
                )
            hdr_mm = hdr_n
        if hdr_mm is not None and _focal_mm_plausible(hdr_mm):
            return float(hdr_mm), "fits_header"
    if db is not None and telescope_id is not None:
        try:
            tel_raw = db.get_telescope_focal_mm(int(telescope_id))
        except Exception:  # noqa: BLE001
            tel_raw = None
        if tel_raw is not None:
            tel_n, tel_fixed = normalize_telescope_focal_mm_for_plate_scale(float(tel_raw))
            if tel_fixed:
                log_event(
                    f"FOCAL: TELESCOPE.FOCAL (ID={int(telescope_id)}) vyzeralo ako 10x preklep "
                    f"({tel_raw:g} mm -> {tel_n:g} mm) - pouzite pre mierku / solver."
                )
            if _focal_mm_plausible(tel_n):
                return float(tel_n), "database_telescope"
    if db is not None:
        raw = db.get_telescope_focal_mm(None)
        if raw is not None:
            norm, fixed = normalize_telescope_focal_mm_for_plate_scale(float(raw))
            if fixed:
                log_event(
                    f"FOCAL: TELESCOPE.FOCAL v DB vyzeralo ako 10x preklep ({raw:g} mm -> {norm:g} mm) - "
                    "pouzite pre mierku / solver."
                )
            if _focal_mm_plausible(norm):
                return float(norm), "database_telescope"
    return None, "none"


def _plate_solve_input_bundle(
    fits_path: Path,
    *,
    app_config: AppConfig | None,
    equipment_id: int | None,
    draft_id: int | None = None,
    telescope_id: int | None = None,
) -> dict[str, Any]:
    """Open DB once: metadata, effective pixel, focal length, expected plate scale [arcsec/pixel]."""
    cfg_u = app_config or AppConfig()
    db_u = _vyvar_open_database(cfg_u)
    _eq_use, _tel_use = resolve_optics_ids_for_platesolve(
        db_u, draft_id, equipment_id=equipment_id, telescope_id=telescope_id
    )
    out: dict[str, Any] = {
        "meta": {},
        "header": None,
        "eff_um": None,
        "focal_mm": None,
        "expected_arcsec_per_px": None,
    }
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            out["header"] = hdul[0].header.copy()
        out["meta"] = fits_meta.extract_fits_metadata(
            fits_path,
            db=db_u,
            app_config=cfg_u,
            id_equipment=_eq_use,
            draft_id=draft_id,
        )
        if draft_id is not None:
            _m = out["meta"]
            if _m.get("focal_length") is None or _m.get("effective_pixel_um_plate_scale") is None:
                raise DraftTechnicalMetadataError(int(draft_id))
        foc: float | None = None
        _mf = out["meta"].get("focal_length")
        if _mf is not None:
            try:
                _fx = float(_mf)
                if math.isfinite(_fx) and _focal_mm_plausible(_fx):
                    foc = _fx
            except (TypeError, ValueError):
                foc = None
        if foc is None:
            foc, _ = _resolve_focal_mm_for_plate_scale(
                out["header"], db_u, telescope_id=_tel_use
            )
        out["focal_mm"] = foc
        eff_um: float | None = None
        x_bin = max(1, int(out["meta"].get("binning", 1) or 1))
        if db_u is not None and _eq_use is not None:
            try:
                _nat = db_u.get_equipment_pixel_size_um(int(_eq_use))
            except Exception:  # noqa: BLE001
                _nat = None
            if _nat is not None:
                try:
                    _nv = float(_nat)
                    if math.isfinite(_nv) and 0.5 < _nv <= 300.0:
                        eff_um = float(_nv) * float(x_bin)
                except (TypeError, ValueError):
                    pass
        if eff_um is None:
            _ev = out["meta"].get("effective_pixel_um_plate_scale")
            if _ev is not None:
                try:
                    _ef = float(_ev)
                    if math.isfinite(_ef) and _ef > 0:
                        eff_um = _ef
                except (TypeError, ValueError):
                    eff_um = None
        out["eff_um"] = eff_um
        if eff_um is not None and foc is not None:
            out["expected_arcsec_per_px"] = plate_scale_arcsec_per_pixel(
                pixel_pitch_um=float(eff_um),
                focal_length_mm=float(foc),
            )
            calculated_scale = out["expected_arcsec_per_px"]
            if calculated_scale is not None:
                log_event(f"MATH CHECK: ({eff_um} / {foc}) * 206.265 = {calculated_scale}")
    except Exception as exc:  # noqa: BLE001
        # EXC-0313: T4 -- db_u.conn.close failure in plate-solve bundle finally is ignored after bundle assembly. (EXCEPT-BULK 2026-07-08)
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().plate_solve_bundle_fail += 1
        LOGGER.error("[PLATE-SOLVE] _plate_solve_input_bundle failed: %s", exc)
        out["bundle_error"] = str(exc)
    finally:
        if db_u is not None:
            try:
                db_u.conn.close()
            except Exception:  # noqa: BLE001
                pass
    return out


def compute_plate_scale_from_db(
    equipment_id: int | None,
    telescope_id: int | None,
    db_conn: Any,
    *,
    binning: int = 1,
) -> float | None:
    """Vypocita plate scale [arcsec/px] z EQUIPMENTS a TELESCOPE tabuliek.

    Formula: plate_scale = (pixel_um * binning) / focal_mm * 206.265
    """
    try:
        pixel_um = None
        focal_mm = None
        bx = max(1, int(binning))
        if equipment_id is not None:
            row = db_conn.execute(
                "SELECT PIXELSIZE FROM EQUIPMENTS WHERE ID = ?",
                (int(equipment_id),),
            ).fetchone()
            if row is not None and row[0] is not None:
                pixel_um = float(row[0])

        if telescope_id is not None:
            row = db_conn.execute(
                "SELECT FOCAL FROM TELESCOPE WHERE ID = ?",
                (int(telescope_id),),
            ).fetchone()
            if row is not None and row[0] is not None and float(row[0]) > 0:
                focal_mm = float(row[0])

        if pixel_um and focal_mm:
            scale = (pixel_um * bx) / focal_mm * 206.265
            return round(scale, 4)
    except Exception:  # noqa: BLE001
        pass
    return None


def _try_rescale_masterstar_linear_wcs_to_expected_plate_scale(
    fits_path: Path,
    *,
    app_config: AppConfig | None,
    equipment_id: int | None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """If DB/optics yield expected arcsec/pixel and the on-disk WCS is linear (no SIP), rescale CD when mismatch is large."""
    from pipeline import (  # noqa: PLC0415
        _all_pix2world_icrs_deg,
        maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel,
    )

    out: dict[str, Any] = {"rescaled": False}
    try:
        b = _plate_solve_input_bundle(
            fits_path,
            app_config=app_config or AppConfig(),
            equipment_id=equipment_id,
            draft_id=draft_id,
        )
        exp = b.get("expected_arcsec_per_px")
        if exp is None:
            return out
        exp_f = float(exp)
        if not math.isfinite(exp_f) or exp_f <= 0:
            return out
        fp = Path(fits_path)
        with fits.open(fp, memmap=False) as hdul:
            hdr0 = hdul[0].header.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w0 = WCS(hdr0)
        if not w0.has_celestial or w0.sip is not None:
            return out
        try:
            _psm0 = np.asarray(w0.pixel_scale_matrix, dtype=np.float64)
            _old_scale = float(np.sqrt(np.abs(np.linalg.det(_psm0))) * 3600.0)
        except Exception:  # noqa: BLE001
            _old_scale = float("nan")
        w2, changed = maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel(w0, exp_f)
        if not changed:
            return out
        _target_scale = exp_f
        try:
            _psm2 = np.asarray(w2.pixel_scale_matrix, dtype=np.float64)
            _new_scale = float(np.sqrt(np.abs(np.linalg.det(_psm2))) * 3600.0)
        except Exception:  # noqa: BLE001
            _new_scale = float("nan")
        try:
            wh = w2.to_header(relax=True)
        except Exception:  # noqa: BLE001
            return out
        with fits.open(fp, mode="update", memmap=False) as hdul:
            h = hdul[0].header
            strip_celestial_wcs_keys(h)
            for k in wh:
                if k in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                    continue
                if k.startswith("NAXIS") and k != "NAXIS":
                    continue
                try:
                    h[k] = wh[k]
                except Exception:  # noqa: BLE001
                    pass
            h.add_history("VYVAR: CD scaled to expected arcsec/pixel from equipment optics")
            hdul.flush()
        if math.isfinite(_old_scale) and math.isfinite(_new_scale):
            log_event(
                f"WCS rescale: {_old_scale:.4f} -> {_new_scale:.4f} arcsec/px "
                f"(target {exp_f:.3f})"
            )
        else:
            log_event(
                f"WCS PLATE SCALE: linearny TAN prisposobeny optickej mierke {exp_f:.3f} arcsec/px."
            )
        _ms_csv = fp.parent / "masterstars_full_match.csv"
        if _ms_csv.is_file():
            try:
                _ms_df = read_vyvar_csv(_ms_csv)
                if _ms_df is not None and not _ms_df.empty and "x" in _ms_df.columns and "y" in _ms_df.columns:
                    _mx = pd.to_numeric(_ms_df["x"], errors="coerce").to_numpy(dtype=np.float64)
                    _my = pd.to_numeric(_ms_df["y"], errors="coerce").to_numpy(dtype=np.float64)
                    _mra, _mdec = _all_pix2world_icrs_deg(w2, _mx, _my)
                    _ms_df["ra_deg"] = _mra
                    _ms_df["dec_deg"] = _mdec
                    _vyvar_df_to_csv(_ms_df, _ms_csv)
                    log_event(
                        "masterstars_full_match.csv ra_deg/dec_deg recomputed after WCS rescale"
                    )
            except Exception as _csv_exc:  # noqa: BLE001
                from except_fix_counters import get_except_fix_counters

                get_except_fix_counters().masterstars_rescale_coords_fail += 1
                LOGGER.error(
                    "masterstars_full_match.csv ra/dec recompute after WCS rescale failed: %s",
                    _csv_exc,
                )
        out["rescaled"] = True
        out["expected_arcsec_per_px"] = exp_f
        if math.isfinite(_old_scale):
            out["old_scale_arcsec_per_px"] = _old_scale
        if math.isfinite(_new_scale):
            out["new_scale_arcsec_per_px"] = _new_scale
    except Exception as exc:  # noqa: BLE001
        log_event(f"WCS PLATE SCALE: uprava CD preskocena - {exc!s}")
        out["error"] = str(exc)
    return out


def _apply_wcs_header_to_fits(fits_path: Path, wcs_hdr: fits.Header) -> None:
    """Merge celestial WCS keywords from ``wcs_hdr`` into the image FITS (primary HDU)."""
    with fits.open(fits_path, mode="update", memmap=False) as hdul:
        h = hdul[0].header
        strip_celestial_wcs_keys(h)
        for k in wcs_hdr:
            if k in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                continue
            if k.startswith("NAXIS") and k != "NAXIS":
                continue
            try:
                h[k] = wcs_hdr[k]
            except Exception:  # noqa: BLE001
                pass
        hdul.flush()


def _solve_wcs_external(
    fits_path: Path,
    *,
    backend: str = "vyvar",
    astrometry_api_key: str | None = None,
    plate_solve_fov_deg: float = 1.0,
    hint_ra_deg: float | None = None,
    hint_dec_deg: float | None = None,
    app_config: AppConfig | None = None,
    equipment_id: int | None = None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """Plate solve: vyhradne VYVAR (lokalna Gaia DB). Backend ``auto`` / Astrometry.net aliasy sa mapuju na VYVAR."""
    be = (backend or "vyvar").strip().lower()
    if be in {"astap", "astap_cli", "local"}:
        LOGGER.warning("platesolve backend %r is no longer supported; using VYVAR.", be)
        be = "vyvar"

    fp_solve = Path(fits_path)

    def _finalize_plate_solve_result(r: dict[str, Any]) -> dict[str, Any]:
        # Post-refine removed; Gaia-based pipeline keeps the WCS as-solved.
        try:
            sm = r.get("sip_meta") if isinstance(r, dict) else None
            if isinstance(sm, dict) and sm.get("initial_wcs_offset_px") is not None:
                _off = float(sm.get("initial_wcs_offset_px"))
                if math.isfinite(_off) and _off > 0:
                    log_event(
                        f"DEBUG: Initial WCS offset detected: {_off:.2f} pixels. Applying coarse correction (Pass 0)..."
                    )
        except Exception:  # noqa: BLE001
            pass
        if isinstance(r, dict) and not bool(r.get("solved", False)):
            try:
                mr = r.get("match_rate")
                mrf = float(mr) if mr is not None else float("nan")
                if math.isfinite(mrf) and mrf < 0.02:
                    log_event(
                        f"WARNING: Plate solve final match rate too low ({mrf * 100.0:.1f}%). "
                        "Returning solved=False to prevent downstream matching on invalid WCS."
                    )
            except Exception:  # noqa: BLE001
                pass
        return r

    def _try_vyvar(*, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg_u = app_config or AppConfig()
        gaia_db = (cfg_u.gaia_db_path or "").strip()
        if not gaia_db:
            return {
                "solved": False,
                "reason": "VYVAR solver: v Settings nastav cestu k lokalnej Gaia DR3 SQLite DB (gaia_db_path).",
            }
        from vyvar_platesolver import solve_wcs_with_local_gaia, _get_masterstar_wcs_parity

        b = bundle or _plate_solve_input_bundle(
            fits_path,
            app_config=cfg_u,
            equipment_id=equipment_id,
            draft_id=draft_id,
        )
        _is_masterstar = fp_solve.name.strip().upper() == "MASTERSTAR.FITS"
        hint_ra = hint_ra_deg
        hint_dec = hint_dec_deg
        _em = b.get("meta") or {}
        eff_um = b.get("eff_um")
        exp_scale = b.get("expected_arcsec_per_px")

        # Per-frame solve MUST NOT invoke blind solver.
        # VYVAR platesolver takes RA/Dec from FITS header (VYTARG*/RA/DEC/WCS); caller hint args are not used.
        # Therefore, for non-MASTERSTAR frames we inject pointing hint from MASTERSTAR WCS (CRVAL1/2).
        # Mirror orientation hint:
        # - MASTERSTAR: can be hinted from MASTERSTAR header (VY_MIRR) to speed/robustify the sweep.
        # - per-frame: do NOT inherit from MASTERSTAR; frames may have different orientation.
        preferred_mirror: str | None = None
        if (not _is_masterstar) and (hint_ra is None or hint_dec is None):
            try:
                # Nacitaj center z MASTERSTAR.fits (ma platny WCS po plate solve)
                masterstar_path: Path | None = None
                _fp_here = Path(fits_path)
                # Guess setup name from parent folder (e.g. processed/lights/R_60_1/*.fits or detrended_aligned/lights/V_60_1/*.fits)
                _setup_guess = ""
                try:
                    _setup_guess = str(_fp_here.parent.name or "").strip()
                except Exception:  # noqa: BLE001
                    _setup_guess = ""
                _candidate = _fp_here
                for _lvl in range(6):  # max 6 urovni hore
                    _candidate = _candidate.parent
                    # New (multi-filter): prefer per-setup MASTERSTAR in platesolve/<setup>/MASTERSTAR.fits
                    _ms_path_setup = (
                        (_candidate / "platesolve" / _setup_guess / "MASTERSTAR.fits")
                        if _setup_guess
                        else None
                    )
                    # Back-compat: old location platesolve/MASTERSTAR.fits (single-setup drafts)
                    _ms_path_root = _candidate / "platesolve" / "MASTERSTAR.fits"
                    _ms_path = _ms_path_setup if _ms_path_setup is not None else _ms_path_root
                    try:
                        if bool(cfg_u.debug_platesolver):
                            log_event(
                                "DEBUG: MASTERSTAR search "
                                f"lvl={_lvl} setup={_setup_guess!r} "
                                f"setup_path={_ms_path_setup} exists_setup={_ms_path_setup.is_file() if _ms_path_setup is not None else False} "
                                f"root_path={_ms_path_root} exists_root={_ms_path_root.is_file()}"
                            )
                    except Exception:  # noqa: BLE001
                        pass
                    # Prefer setup MASTERSTAR if it exists; in multi-group drafts never use root MASTERSTAR.
                    _multi_obs = draft_is_multi_group_obs(_candidate)
                    if _ms_path_setup is not None and _ms_path_setup.is_file():
                        masterstar_path = _ms_path_setup
                        break
                    if _multi_obs and _setup_guess:
                        continue
                    if _ms_path_root.is_file():
                        masterstar_path = _ms_path_root
                        break
                if masterstar_path is not None and masterstar_path.is_file():
                    from astropy.io import fits as _fits

                    with _fits.open(masterstar_path, memmap=False) as _hdul:
                        _mhdr = _hdul[0].header
                        _ms_ra = _mhdr.get("CRVAL1")
                        _ms_dec = _mhdr.get("CRVAL2")
                        if _ms_ra is not None and _ms_dec is not None:
                            hint_ra = float(_ms_ra)
                            hint_dec = float(_ms_dec)
                            log_event(
                                f"INFO: Per-frame hint z MASTERSTAR CRVAL: RA={hint_ra:.4f} Dec={hint_dec:.4f}"
                            )
                    # Do not set preferred_mirror for per-frame solves here.
            except Exception as _e:  # noqa: BLE001
                log_event(f"WARNING: Per-frame hint z MASTERSTAR zlyhal: {_e}")

        # Fallback: if MASTERSTAR hint is missing, try draft manifest center (can be 0/0).
        if (not _is_masterstar) and (hint_ra is None or hint_dec is None) and draft_id is not None:
            try:
                _db_hint = _vyvar_open_database(cfg_u)
                if _db_hint is not None:
                    try:
                        drow = _db_hint.fetch_obs_draft_by_id(int(draft_id)) or {}
                        ra_db = drow.get("CENTEROFFIELDRA")
                        de_db = drow.get("CENTEROFFIELDDE")
                        if ra_db is not None and de_db is not None:
                            ra_f = float(ra_db)
                            de_f = float(de_db)
                            if math.isfinite(ra_f) and math.isfinite(de_f) and not (
                                abs(ra_f) < 1e-9 and abs(de_f) < 1e-9
                            ):
                                hint_ra = float(ra_f)
                                hint_dec = float(de_f)
                                log_event(
                                    f"INFO: Per-frame hint z draft manifest center: RA={hint_ra:.4f} Dec={hint_dec:.4f}"
                                )
                    finally:
                        try:
                            _db_hint.conn.close()
                        except Exception:  # noqa: BLE001
                            pass
            except Exception:  # noqa: BLE001
                pass

        _db_ps = _vyvar_open_database(cfg_u)
        _auto_ps: float | None = None
        if _db_ps is not None:
            try:
                _eq_ps, _tel_ps = resolve_optics_ids_for_platesolve(
                    _db_ps, draft_id, equipment_id=equipment_id
                )
                _bx = max(1, int(_em.get("binning", 1) or 1))
                _auto_ps = compute_plate_scale_from_db(
                    int(_eq_ps) if _eq_ps is not None else None,
                    _tel_ps,
                    _db_ps.conn,
                    binning=_bx,
                )
            except Exception:  # noqa: BLE001
                _auto_ps = None
            finally:
                try:
                    _db_ps.conn.close()
                except Exception:  # noqa: BLE001
                    pass
        exp_scale = _auto_ps or exp_scale or None
        if _auto_ps is not None:
            log_event(
                f"INFO: Plate scale z DB (Equipment+Telescope): {_auto_ps:.4f} arcsec/px"
            )
        elif _is_masterstar:
            log_event(
                "WARNING: Plate scale z DB nedostupna - solver odvodi mierku z FITS alebo None"
            )
        _bx = max(1, int(_em.get("binning", 1) or 1))
        _pix = _em.get("pixel_size_um_physical")
        _foc_mm = b.get("focal_mm")
        _pix_s = f"{float(_pix):.4g}" if _pix is not None else "n/a"
        _foc_s = f"{float(_foc_mm):.4g}" if _foc_mm is not None else "n/a"
        _eff_s = f"{float(eff_um):.4g}" if eff_um is not None else "n/a"
        log_event(
            f"SOLVER INPUT: Focal={_foc_s}mm, Pixel={_pix_s}um, Bin={_bx}x -> Effective Pixel={_eff_s}um"
        )
        if exp_scale is not None and _foc_mm is not None and eff_um is not None:
            log_event(
                f"PLATE SOLVING: Mierka nastavena na {float(exp_scale):.3f} arcsec/px "
                f"(vypocitane z {float(_foc_mm)}mm a {float(eff_um)}um)"
            )
        elif exp_scale is not None:
            log_event(f"PLATE SOLVING: Mierka nastavena na {float(exp_scale):.3f} arcsec/px")

        try:
            if bool(cfg_u.debug_platesolver):
                log_event(
                    f"DEBUG: _try_vyvar hint_ra={hint_ra} hint_dec={hint_dec} is_masterstar={_is_masterstar}"
                )
        except Exception:  # noqa: BLE001
            pass

        # Ensure per-frame images have VYTARGRA/VYTARGDE in FITS header so vyvar_platesolver
        # can avoid blind solving (it reads hints from the header, not from caller args).
        if (not _is_masterstar) and hint_ra is not None and hint_dec is not None:
            try:
                with fits.open(fp_solve, mode="update", memmap=False) as hdul:
                    h0 = hdul[0].header
                    if "VYTARGRA" not in h0 or "VYTARGDE" not in h0:
                        h0["VYTARGRA"] = (float(hint_ra), "VYVAR plate-solve hint RA [deg] ICRS")
                        h0["VYTARGDE"] = (float(hint_dec), "VYVAR plate-solve hint Dec [deg] ICRS")
                        hdul.flush()
                log_event(f"INFO: Per-frame VYTARG zapisany: RA={float(hint_ra):.4f} Dec={float(hint_dec):.4f}")
            except Exception as e:  # noqa: BLE001
                from except_fix_counters import get_except_fix_counters

                get_except_fix_counters().vytarg_header_write_fail += 1
                LOGGER.error("Per-frame VYTARG header write failed: %s", e)

        _no_sip = os.environ.get("VYVAR_PLATE_SOLVE_NO_SIP", "").strip().lower() in {"1", "true", "yes", "on"}
        # For MASTERSTAR only: hint mirror orientation from its own header (VY_MIRR) / parity.
        if _is_masterstar:
            try:
                preferred_mirror = _get_masterstar_wcs_parity(Path(fp_solve))
            except Exception:  # noqa: BLE001
                preferred_mirror = None

        return _finalize_plate_solve_result(
            solve_wcs_with_local_gaia(
                fp_solve,
                hint_ra_deg=float(hint_ra) if hint_ra is not None else None,
                hint_dec_deg=float(hint_dec) if hint_dec is not None else None,
                fov_diameter_deg=float(plate_solve_fov_deg),
                gaia_db_path=Path(gaia_db),
                enable_sip=not _no_sip,
                effective_pixel_um=eff_um,
                focal_length_mm=float(_foc_mm) if _foc_mm is not None else None,
                expected_plate_scale_arcsec_per_px=exp_scale,
                preferred_mirror=preferred_mirror,
                max_catalog_rows=100000 if _is_masterstar else None,
                faintest_mag_limit=18.0 if _is_masterstar else None,
            )
        )

    if be == "auto":
        cfg_a = app_config or AppConfig()
        b_auto = _plate_solve_input_bundle(
            fits_path, app_config=cfg_a, equipment_id=equipment_id, draft_id=draft_id
        )
        log_event('Plate-solve backend "auto": len VYVAR lokalny solver (bez Astrometry.net).')
        return _try_vyvar(bundle=b_auto)

    # Accept legacy backend aliases for backward compatibility.
    if be in {"vyvar", "vyvar_platesolver", "vyvar_gaia"}:
        return _try_vyvar(bundle=None)

    if be in {"astrometry.net", "astrometry_net", "online", "net"}:
        LOGGER.warning(
            "Plate-solve backend %r uz nie je podporovany - pouzivam VYVAR (lokalna Gaia).", be
        )
        cfg_n = app_config or AppConfig()
        b_net = _plate_solve_input_bundle(
            fits_path, app_config=cfg_n, equipment_id=equipment_id, draft_id=draft_id
        )
        return _try_vyvar(bundle=b_net)

    LOGGER.warning("Unknown platesolve backend %r; using VYVAR.", be)
    return _try_vyvar(bundle=None)


def _wcs_field_center_radec_deg(fits_path: Path) -> tuple[float, float] | None:
    """RA/Dec (deg) of image center from existing celestial WCS."""

    try:
        with fits.open(fits_path, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
        h = int(hdr.get("NAXIS2", 0) or 0)
        wpx = int(hdr.get("NAXIS1", 0) or 0)
        if h <= 0 or wpx <= 0:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr)
        if not w.has_celestial:
            return None
        c = SkyCoord.from_pixel((wpx - 1) / 2.0, (h - 1) / 2.0, wcs=w, origin=0)
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:  # noqa: BLE001
        return None


def _merge_vsx_exoplanet_variable_targets(
    vsx_df: pd.DataFrame,
    exo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge VSX + exoplanet promotion rows; same Gaia id -> one row with VSX labels + exo_*."""
    from gaia_catalog_id import normalize_gaia_source_id_series

    vsx = vsx_df.copy() if vsx_df is not None else pd.DataFrame()
    exo = exo_df.copy() if exo_df is not None else pd.DataFrame()
    if vsx.empty and exo.empty:
        return vsx
    if "catalog_id" in vsx.columns:
        vsx["catalog_id"] = normalize_gaia_source_id_series(vsx["catalog_id"])
    if not exo.empty and "catalog_id" in exo.columns:
        exo["catalog_id"] = normalize_gaia_source_id_series(exo["catalog_id"])

    exo_extra = list(_EXO_HOST_ANNOTATION_COLUMNS) + ["target_origin"]
    for col in exo_extra:
        if col not in vsx.columns:
            vsx[col] = ""
    if exo.empty:
        return vsx

    if vsx.empty:
        return exo

    exo_by_cid: dict[str, pd.Series] = {}
    for _, er in exo.iterrows():
        cid = str(er.get("catalog_id", "") or "").strip()
        if cid:
            exo_by_cid[cid] = er

    vsx_cids: set[str] = set()
    for i in vsx.index:
        cid = str(vsx.at[i, "catalog_id"] or "").strip()
        if not cid:
            continue
        vsx_cids.add(cid)
        er = exo_by_cid.get(cid)
        if er is not None:
            for col in exo_extra:
                if col in er.index:
                    vsx.at[i, col] = er[col]

    exo_only = exo.loc[~exo["catalog_id"].astype(str).str.strip().isin(vsx_cids)].copy()
    if exo_only.empty:
        return vsx
    merged = pd.concat([vsx, exo_only], ignore_index=True)
    log_event(
        f"[EXO TARGET] variable_targets merge: VSX={len(vsx)} exo-only added={len(exo_only)} "
        f"total={len(merged)}"
    )
    return merged


def _query_vsx_local_frame_bbox(
    *,
    wcs: Any,
    width_px: int,
    height_px: int,
    vsx_db_path: Path | None,
    margin_px: float = 50.0,
    center: SkyCoord | None = None,
    require_db: bool = False,
) -> pd.DataFrame:
    """Query **local VSX** within the FRAME footprint (frame bbox + ``margin_px``), spatial-first.

    Completeness must not depend on row order: the frame bbox is tiny (sub-degree), so the SQL
    result is small and the global ``catalog_query_max_rows`` cap never truncates it (unlike the
    3.5 deg cone in ``_query_vsx_local``, which hit the 15000-row cap and dropped a contiguous Dec
    slice). Returns raw VSX rows over the bbox; the caller applies the precise in-frame pixel
    filter (matching ``margin_px``). RA wrap is handled via centre-relative offsets.
    """
    from database import count_vsx_local_rows, require_vsx_local_db_path

    if require_db:
        vp = require_vsx_local_db_path(vsx_db_path)
        n_total = count_vsx_local_rows(vp)
    elif vsx_db_path is None:
        return pd.DataFrame()
    else:
        vp = Path(vsx_db_path).expanduser().resolve()
        if not vp.is_file():
            return pd.DataFrame()
        n_total = 0
    m = float(margin_px)
    w = float(width_px)
    h = float(height_px)
    # Pixel-space samples covering the frame border + margin (corners + edge midpoints).
    xs_px = np.asarray([-m, w * 0.5, w + m, -m, w + m, -m, w * 0.5, w + m], dtype=np.float64)
    ys_px = np.asarray([-m, -m, -m, h * 0.5, h * 0.5, h + m, h + m, h + m], dtype=np.float64)
    try:
        world = wcs.all_pix2world(xs_px, ys_px, 0)
        ras = np.asarray(world[0], dtype=np.float64)
        decs = np.asarray(world[1], dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().vsx_frame_bbox_wcs_fail += 1
        LOGGER.error("[VSX] frame-bbox WCS all_pix2world failed: %s", exc)
        return pd.DataFrame()
    ok = np.isfinite(ras) & np.isfinite(decs)
    if not bool(ok.any()):
        return pd.DataFrame()
    ras = ras[ok]
    decs = decs[ok]
    de_min = float(np.min(decs))
    de_max = float(np.max(decs))
    if center is not None:
        try:
            ra0 = float(center.icrs.ra.deg)
        except Exception:  # noqa: BLE001
            ra0 = float(np.median(ras))
    else:
        ra0 = float(np.median(ras))
    dra = ((ras - ra0 + 180.0) % 360.0) - 180.0
    ra_min = ra0 + float(np.min(dra))
    ra_max = ra0 + float(np.max(dra))
    pad = 1.0 / 3600.0  # 1 arcsec rounding pad
    rows = query_local_vsx(
        vp,
        ra_min=ra_min - pad,
        ra_max=ra_max + pad,
        dec_min=de_min - pad,
        dec_max=de_max + pad,
        max_rows=None,  # frame bbox is tiny -> no cap needed (spatial-first completeness)
    )
    if not rows:
        if require_db:
            log_event(
                f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
                f"(frame bbox+{int(m)}px, RA=[{ra_min:.4f},{ra_max:.4f}], Dec=[{de_min:.4f},{de_max:.4f}])"
            )
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        if require_db:
            from database import VSXCatalogError

            raise VSXCatalogError(f"VSX query returned unexpected columns from {vp}.")
        return pd.DataFrame()
    try:
        log_event(
            f"CATALOG SEARCH (VSX local, frame bbox+{int(m)}px): {len(df)} zdrojov "
            f"(RA=[{ra_min:.4f},{ra_max:.4f}], Dec=[{de_min:.4f},{de_max:.4f}], bez cap)"
        )
    except Exception:  # noqa: BLE001
        pass
    return df


def _finite_positive_adu(value: Any) -> float | None:
    """Return a finite positive ADU, else None (NaN/inf/<=0/unparsable)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(v) and v > 0:
        return v
    return None


def _sat_adu_from_draft_sat_diag(archive_path: Path | None) -> float | None:
    """Read ``sat_diag.json`` ``sat_adu`` from a draft archive root when present."""
    if archive_path is None:
        return None
    try:
        from sat_diag import load_sat_diag_json  # noqa: PLC0415

        ctx = load_sat_diag_json(Path(archive_path) / "sat_diag.json")
    except Exception:  # noqa: BLE001
        return None
    if ctx is None:
        return None
    return _finite_positive_adu(getattr(ctx, "sat_adu", None))


def _equipment_saturate_adu_from_db(equipment_id: int | None) -> float | None:
    """Read ``EQUIPMENTS.SATURATE_ADU`` when a valid equipment id is given."""
    if equipment_id is None:
        return None
    try:
        eid = int(equipment_id)
    except (TypeError, ValueError):
        return None
    if eid <= 0:
        return None
    try:
        cfg = AppConfig()
        db = VyvarDatabase(cfg.database_path)
        return db.get_equipment_saturation_adu(eid)
    except Exception:  # noqa: BLE001
        return None


_VYVAR_TIME_JD_CSV_COLS = ("jd_mid", "hjd_mid", "bjd_tdb_mid")


def _vyvar_df_round_time_jd_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Round geocentric/heliocentric/barycentric JD columns to six decimals for stable CSV / spreadsheet display."""
    cols = [c for c in _VYVAR_TIME_JD_CSV_COLS if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
    return out


def _vyvar_df_to_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Write sidecar / per-frame proc CSV.

    When ``mag`` is present it holds the Gaia catalog G magnitude (``phot_g_mean_mag`` at
    cross-match time) - constant per star for the night, not the frame instrumental magnitude.
    Science paths must use ``dao_flux`` (see ``docs/VYVAR_PROCESS.md``).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    export_df = df
    if "catalog_id" in df.columns:
        export_df = df.copy()
        export_df["catalog_id"] = catalog_id_series_for_masterstars_export(df)
    export_df = _vyvar_df_round_time_jd_for_csv(export_df)
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.csv as pacsv  # type: ignore[import-not-found]

        pacsv.write_csv(pa.Table.from_pandas(export_df, preserve_index=False), str(p))
    except Exception:  # noqa: BLE001
        export_df.to_csv(p, index=False, lineterminator="\n", na_rep="")


def select_comparison_stars_spatial_grid(
    df: pd.DataFrame,
    *,
    width_px: float,
    height_px: float,
    n_comp: int = 0,
    require_catalog_match: bool = True,
    require_photometry_ok: bool = True,
    require_non_variable: bool = True,
    variable_targets_df: pd.DataFrame | None = None,
    safe_bbox: tuple[float, float, float, float] | None = None,
    exclude_nonlinear_badcolumn: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose comparison stars for ensemble photometry: ~one brightest candidate per spatial grid cell.

    Uses catalog-matched, photometry-safe detections, stratified on a nxxny grid sized for ``n_comp`` and
    image aspect ratio so stars are spread across the full detector (typical 100-200 comps for APASS-like work).

    When ``require_non_variable`` is True and column ``catalog_known_variable`` exists, stars flagged as
    known variables (VSX and/or Gaia ``phot_variable_flag``) are excluded from the ensemble.
    """
    import math

    import numpy as np

    if df.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "empty"}

    work = df.copy()
    if "is_usable" in work.columns:
        work = work[work["is_usable"].fillna(False).astype(bool)]
    if work.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "no_rows_after_usable_filter"}
    try:
        from config import AppConfig  # noqa: PLC0415

        _seed_pool = bool(getattr(AppConfig(), "masterstar_forced_seed_comp_pool_enabled", False))
    except Exception:  # noqa: BLE001
        _seed_pool = False
    if not _seed_pool:
        if "source_state" in work.columns:
            work = work[work["source_state"].astype(str).str.strip() != "FORCED_SEED"]
        if "forced_photometry" in work.columns:
            _fp = work["forced_photometry"].astype(str).str.strip().str.lower()
            work = work[~_fp.isin(["true", "1", "yes"])]
        if work.empty:
            return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "no_rows_after_seed_pool_filter"}
    if require_catalog_match and "catalog" in work.columns:
        work = work[work["catalog"].fillna("").astype(str).str.strip() != ""]
    if require_photometry_ok and "photometry_ok" in work.columns:
        work = work[work["photometry_ok"].fillna(True).astype(bool)]
    if require_non_variable and "catalog_known_variable" in work.columns:
        work = work[~work["catalog_known_variable"].fillna(False).astype(bool)]
    # Proximity veto: exclude comp candidates within 10 arcsec of any VSX variable target.
    # This is intentionally coordinate-based (not catalog_id-based), because VSX->Gaia cross-id can resolve
    # to a different nearby Gaia source_id than the comp candidate, yet still represent the same physical star.
    if (
        variable_targets_df is not None
        and not getattr(variable_targets_df, "empty", True)
        and "ra_deg" in work.columns
        and "dec_deg" in work.columns
        and "ra_deg" in variable_targets_df.columns
        and "dec_deg" in variable_targets_df.columns
    ):
        try:
            from astropy.coordinates import SkyCoord  # noqa: PLC0415
            import astropy.units as u  # noqa: PLC0415

            w_ra = pd.to_numeric(work["ra_deg"], errors="coerce")
            w_de = pd.to_numeric(work["dec_deg"], errors="coerce")
            ok_w = w_ra.notna() & w_de.notna()
            v_ra = pd.to_numeric(variable_targets_df["ra_deg"], errors="coerce")
            v_de = pd.to_numeric(variable_targets_df["dec_deg"], errors="coerce")
            ok_v = v_ra.notna() & v_de.notna()
            if bool(ok_w.any()) and bool(ok_v.any()):
                comp_coo = SkyCoord(
                    ra=w_ra.loc[ok_w].astype(float).to_numpy() * u.deg,
                    dec=w_de.loc[ok_w].astype(float).to_numpy() * u.deg,
                    frame="icrs",
                )
                vsx_coo = SkyCoord(
                    ra=v_ra.loc[ok_v].astype(float).to_numpy() * u.deg,
                    dec=v_de.loc[ok_v].astype(float).to_numpy() * u.deg,
                    frame="icrs",
                )
                _idx, sep2d, _ = comp_coo.match_to_catalog_sky(vsx_coo)
                veto_arcsec = 10.0
                near = sep2d <= (float(veto_arcsec) * u.arcsec)
                n_removed = int(getattr(near, "sum", lambda: 0)())
                if n_removed > 0:
                    drop_idx = w_ra.loc[ok_w].index[np.asarray(near, dtype=bool)]
                    work = work.drop(index=drop_idx)
                log_event(
                    f"[COMP SELECT] Proximity veto: removed {n_removed} comp candidates within {float(veto_arcsec):.0f} arcsec of VSX targets"
                )
        except Exception:  # noqa: BLE001
            pass
    # Annulus-aware border filter: keep only stars whose centroid is within the safe bbox
    # (intersection of aligned frames shrunk by sky-annulus outer radius).
    if safe_bbox is not None and "x" in work.columns and "y" in work.columns:
        try:
            from aperture_policy import stars_fit_on_chip  # noqa: PLC0415

            x0b, y0b, x1b, y1b = safe_bbox
            before = int(len(work))
            # 4-tuple naxis = precomputed safe bbox (already shrunk by resolver r_out).
            _on = stars_fit_on_chip(
                work["x"], work["y"], (0.0, 0.0, 0.0), (float(x0b), float(y0b), float(x1b), float(y1b))
            )
            work = work[np.asarray(_on, dtype=bool)]
            removed = before - int(len(work))
            if removed > 0:
                logging.info(
                    f"[BORDER] Comp selection: removed {removed} candidates outside safe bbox "
                    f"(annulus-aware intersection)"
                )
        except Exception:  # noqa: BLE001
            pass
    if exclude_nonlinear_badcolumn and "likely_nonlinear" in work.columns:
        _ln = pd.to_numeric(work["likely_nonlinear"], errors="coerce").fillna(0).astype(int)
        work = work[_ln == 0]
    if exclude_nonlinear_badcolumn and "on_bad_column" in work.columns:
        _ob = pd.to_numeric(work["on_bad_column"], errors="coerce").fillna(0).astype(int)
        work = work[_ob == 0]

    if work.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "no_rows_after_filter"}

    w = float(width_px)
    h = float(height_px)
    if w <= 0 or h <= 0:
        w = float(work["x"].max()) + 1.0
        h = float(work["y"].max()) + 1.0

    if "flux" not in work.columns:
        work["flux"] = 0.0
    work["_flux_key"] = pd.to_numeric(work["flux"], errors="coerce").fillna(0.0)

    # COMP-POOL-01: n_comp <= 0 means uncapped pool (no grid truncate).
    if int(n_comp) <= 0:
        picked = work.sort_values("_flux_key", ascending=False).drop(
            columns=["_flux_key"], errors="ignore"
        )
        picked.insert(0, "comp_id", [f"COMP_{i+1:04d}" for i in range(len(picked))])
        picked.insert(1, "role", "comparison")
        meta = {
            "n_selected": int(len(picked)),
            "grid_nx": 0,
            "grid_ny": 0,
            "n_comp_requested": 0,
            "pool_uncapped": True,
            "width_px": float(w),
            "height_px": float(h),
            "require_non_variable": bool(require_non_variable),
            "exclude_nonlinear_badcolumn": bool(exclude_nonlinear_badcolumn),
        }
        return picked, meta

    ar = w / h
    nc = max(1, int(n_comp))
    ny = max(1, int(round(math.sqrt(nc / ar))))
    nx = max(1, int(math.ceil(nc / ny)))
    while nx * ny < nc:
        nx += 1

    cw = w / float(nx)
    ch = h / float(ny)

    ix = np.floor(np.clip(work["x"].to_numpy(dtype=float), 0.0, w - 1e-6) / cw).astype(int)
    iy = np.floor(np.clip(work["y"].to_numpy(dtype=float), 0.0, h - 1e-6) / ch).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    work["_cell"] = ix + iy * nx

    # Brightest per cell
    picked = (
        work.sort_values("_flux_key", ascending=False)
        .groupby("_cell", as_index=False, sort=False)
        .head(1)
    )

    if len(picked) > nc:
        picked = picked.nlargest(nc, "_flux_key")

    picked = picked.drop(columns=["_cell", "_flux_key"], errors="ignore")
    picked.insert(0, "comp_id", [f"COMP_{i+1:04d}" for i in range(len(picked))])
    picked.insert(1, "role", "comparison")

    meta = {
        "n_selected": int(len(picked)),
        "grid_nx": int(nx),
        "grid_ny": int(ny),
        "n_comp_requested": int(nc),
        "pool_uncapped": False,
        "width_px": float(w),
        "height_px": float(h),
        "require_non_variable": bool(require_non_variable),
        "exclude_nonlinear_badcolumn": bool(exclude_nonlinear_badcolumn),
    }
    return picked, meta


def write_photometry_plan_files(
    *,
    platesolve_dir: Path,
    masterstar_fits: Path,
    masterstars_csv: Path,
    n_comparison_stars: int = 0,
    require_non_variable: bool = True,
    draft_id: int | None = None,
    database_path: Path | str | None = None,
    aligned_files: list[Path] | None = None,
    aligned_ram_frames: list[tuple[str, Any, Any]] | None = None,
    require_safe_bbox: bool = False,
) -> dict[str, Any]:
    """Write ``comparison_stars.csv`` from ``masterstars.csv`` + image size; stub ``variable_targets.csv``."""
    from pipeline import (  # noqa: PLC0415
        _build_exoplanet_promotion_rows_from_masterstars,
        _effective_field_catalog_cone_radius_deg,
    )

    import numpy as np
    import json


    ps = Path(platesolve_dir)
    ps.mkdir(parents=True, exist_ok=True)

    if not masterstars_csv.is_file():
        return {"comparison_stars_csv": "", "variable_targets_csv": "", "error": "missing masterstars.csv"}

    # Read IDs as strings (avoid float64 precision loss for 19-digit Gaia IDs).
    df = pd.read_csv(masterstars_csv, low_memory=False, dtype={"catalog_id": str, "name": str})
    try:
        from gaia_catalog_id import catalog_id_series_for_masterstars_export  # noqa: PLC0415

        if "catalog_id" in df.columns:
            df = df.copy()
            df["catalog_id"] = catalog_id_series_for_masterstars_export(df)
    except Exception:  # noqa: BLE001
        pass

    _enrichment_provenance: dict[str, Any] = {"source": "masterstars_full_match.csv"}
    try:
        from masterstars_enrichment import missing_comp_selection_enrichment_columns  # noqa: PLC0415

        _missing_enrich = missing_comp_selection_enrichment_columns(df)
        if _missing_enrich:
            _msg = (
                "comp-selection enrichment unavailable for this draft: columns "
                f"{_missing_enrich} absent from masterstars_full_match.csv - "
                "selection proceeds without nonlinearity/bad-column exclusion"
            )
            logging.error("[COMP-SELECTION] %s", _msg)
            log_event(f"ERROR: {_msg}")
            _enrichment_provenance = {
                "source": "masterstars_full_match.csv",
                "enrichment_missing_columns": list(_missing_enrich),
                "nonlinearity_badcolumn_exclusion": False,
                "message": _msg,
            }
        else:
            _enrichment_provenance["nonlinearity_badcolumn_exclusion"] = True
    except Exception as _enrich_exc:  # noqa: BLE001
        logging.error("[COMP-SELECTION] enrichment column check failed: %s", _enrich_exc)
    try:
        with fits.open(masterstar_fits, memmap=False) as hdul:
            hdr = hdul[0].header
    except Exception as _ms_open_exc:  # noqa: BLE001
        log_event(f"write_photometry_plan_files: nepodarilo sa otvorit MASTERSTAR.fits ({masterstar_fits}): {_ms_open_exc!s}")
        return {"comparison_stars_csv": "", "variable_targets_csv": "", "error": "MASTERSTAR open failed"}
    wpx = int(hdr.get("NAXIS1", 0) or 0)
    h = int(hdr.get("NAXIS2", 0) or 0)
    if wpx <= 0 or h <= 0:
        return {"comparison_stars_csv": "", "error": "MASTERSTAR has no data"}

    _cfg_plan = AppConfig()

    comp_path = ps / "comparison_stars.csv"
    var_path = ps / "variable_targets.csv"

    # --- Annulus-aware intersection bbox (Variant A2) ---
    # Compute a "safe" bbox from the intersection of aligned frames, shrunk by the sky-annulus outer radius.
    # CONSOLIDATE-01B A2: FWHM source stays MASTERSTAR VY_FWHM (qc vs header differ on all
    # 134 era04 frames at FITS-card rounding; night-median qc vs MASTERSTAR is also not
    # last-ulp identical). Do not invent 3.5 / 10.5.
    _safe_bbox: tuple[float, float, float, float] | None = None
    _r_out: float | None = None
    try:
        from aperture_policy import fwhm_from_header_vy_fwhm, resolve_aperture_geometry  # noqa: PLC0415

        _fwhm_px = fwhm_from_header_vy_fwhm(hdr)
        if _fwhm_px is None:
            raise ValueError("MASTERSTAR VY_FWHM missing; cannot resolve annulus r_out")
        _r_ap_b, _r_in_b, _r_out = resolve_aperture_geometry(
            f=float(_cfg_plan.aperture_fwhm_factor),
            fwhm_px=float(_fwhm_px),
            annulus_inner_fwhm=float(_cfg_plan.annulus_inner_fwhm),
            annulus_outer_fwhm=float(_cfg_plan.annulus_outer_fwhm),
        )
        _ = (_r_ap_b, _r_in_b)

        draft_root = ps.parent.parent
        aligned_dir = draft_root / "detrended_aligned" / "lights" / ps.name
        if aligned_files is not None:
            all_aligned = [Path(p) for p in aligned_files if Path(p).exists()]
            LOGGER.info(f"[BORDER] Using {len(all_aligned)} pre-supplied aligned frame paths")
        else:
            all_aligned = sorted(aligned_dir.glob("proc_*.fits"))
            LOGGER.info(f"[BORDER] Glob found {len(all_aligned)} aligned frames in {aligned_dir}")

        ram_arrays: list[Any] = []
        if aligned_ram_frames:
            for _name, _hdr, _arr in aligned_ram_frames:
                if _arr is not None:
                    ram_arrays.append(_arr)
            if ram_arrays:
                LOGGER.info(f"[BORDER] Using {len(ram_arrays)} aligned RAM frames for intersection bbox")

        n_frame_sources = len(all_aligned) + (len(ram_arrays) if not all_aligned else 0)
        if n_frame_sources == 0:
            if require_safe_bbox:
                raise RuntimeError(
                    "[BORDER] Post-alignment border filter requires aligned frames but none "
                    "were found on disk or in RAM handoff"
                )
            log_event(
                "[BORDER] Deferred: no aligned proc_*.fits on disk yet "
                "(pre-alignment or RAM-handoff); border filter skipped"
            )
        elif n_frame_sources < 2:
            if require_safe_bbox:
                raise RuntimeError(
                    f"[BORDER] Post-alignment border filter requires >=2 aligned frames, got {n_frame_sources}"
                )
            log_event(f"[BORDER] Not enough aligned frames for intersection bbox: {n_frame_sources}")
        else:
            frames_for_bbox = all_aligned if all_aligned else []
            ram_for_bbox = [] if all_aligned else ram_arrays
            try:
                if draft_id is not None and database_path:
                    dbp2 = Path(str(database_path))
                    if dbp2.is_file():
                        jr = detect_field_jumps(
                            db=VyvarDatabase(dbp2), draft_id=int(draft_id)
                        )
                        if bool(jr.get("has_jump")) and int(jr.get("n_groups") or 0) > 0:
                            dom = None
                            for g in (jr.get("groups") or []):
                                if bool(g.get("is_dominant")):
                                    dom = g
                                    break
                            if dom is not None:
                                fs = int(dom.get("frame_start") or 0)
                                fe = int(dom.get("frame_end") or 0)
                                if all_aligned:
                                    frames_for_bbox = all_aligned[fs : fe + 1]
                                elif aligned_ram_frames:
                                    ram_for_bbox = [
                                        arr for _n, _h, arr in aligned_ram_frames[fs : fe + 1] if arr is not None
                                    ]
                                log_event(
                                    f"[BORDER] Field jump detected - using dominant group frames {fs}-{fe} "
                                    f"({len(frames_for_bbox) or len(ram_for_bbox)} frames) for intersection bbox"
                                )
                            else:
                                log_event(
                                    f"[BORDER] Field jump detected - no dominant group found; using all "
                                    f"{len(frames_for_bbox) or len(ram_for_bbox)} frames"
                                )
                        else:
                            log_event(
                                f"[BORDER] Field stable - using all "
                                f"{len(frames_for_bbox) or len(ram_for_bbox)} aligned frames for intersection bbox"
                            )
            except Exception as _jr_exc:  # noqa: BLE001
                log_event(f"[BORDER] detect_field_jumps failed: {_jr_exc!s} - using all aligned frames")

            from photometry_core import (  # noqa: PLC0415
                common_field_intersection_bbox_px,
                common_field_intersection_bbox_px_from_arrays,
            )

            raw_bbox = None
            if len(frames_for_bbox) >= 2:
                raw_bbox = common_field_intersection_bbox_px(frame_paths=frames_for_bbox, finite_stride=16)
            elif len(ram_for_bbox) >= 2:
                raw_bbox = common_field_intersection_bbox_px_from_arrays(
                    frame_arrays=ram_for_bbox, finite_stride=16
                )

            if raw_bbox is None:
                if require_safe_bbox:
                    raise RuntimeError(
                        "[BORDER] Post-alignment border filter failed: intersection bbox returned None "
                        f"(disk={len(frames_for_bbox)}, ram={len(ram_for_bbox)})"
                    )
                log_event("[BORDER] intersection bbox returned None - no border filter applied")
            else:
                x0, y0, x1, y1 = raw_bbox
                sb = (float(x0) + _r_out, float(y0) + _r_out, float(x1) - _r_out, float(y1) - _r_out)
                if sb[2] > sb[0] and sb[3] > sb[1]:
                    _safe_bbox = sb
                    log_event(
                        f"[BORDER] Safe bbox (shrunk by r_out={_r_out:.1f}px): "
                        f"x=[{_safe_bbox[0]:.0f},{_safe_bbox[2]:.0f}] y=[{_safe_bbox[1]:.0f},{_safe_bbox[3]:.0f}]"
                    )
                elif require_safe_bbox:
                    raise RuntimeError(
                        f"[BORDER] Post-alignment safe bbox invalid after shrink (r_out={_r_out:.1f}px)"
                    )
                else:
                    log_event(
                        f"[BORDER] Safe bbox invalid after shrink (r_out={_r_out:.1f}px); skipping border filter"
                    )
                    _safe_bbox = None
    except RuntimeError:
        raise
    except Exception as _bbox_exc:  # noqa: BLE001
        if require_safe_bbox:
            raise RuntimeError(f"[BORDER] safe_bbox computation failed: {_bbox_exc!s}") from _bbox_exc
        log_event(f"[BORDER] safe_bbox computation failed: {_bbox_exc!s} - skipping border filter")
        _safe_bbox = None

    # VSX variable targets for the field (frame bbox), with pixel coords from MASTERSTAR WCS.
    var_cols = [
        "name",
        "catalog_id",
        "catalog",
        "ra_deg",
        "dec_deg",
        "priority",
        "notes",
        "vsx_name",
        "vsx_type",
        "vsx_period",
        "x",
        "y",
        "mag",
        "zone",
        "gaia_match_arcsec",
        "gaia_match_quality",
        "gaia_match_source",
        "vsx_mag_max",
        "exo_host_obj_id",
        "exo_host_name",
        "exo_cat_source",
        "exo_disposition",
        "exo_match_sep_arcsec",
        "target_origin",
    ]
    vsx_out = pd.DataFrame(columns=var_cols)
    _vsx_n_cone = 0
    _vsx_diag: dict[str, Any] = {}
    try:
        from database import VSXCatalogError, require_vsx_local_db_path

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w0 = WCS(hdr)
        if not bool(getattr(w0, "has_celestial", False)):
            raise RuntimeError("MASTERSTAR WCS has no celestial axes.")
        if w0.has_celestial:
            center, radius_deg = _effective_field_catalog_cone_radius_deg(
                w0, int(h), int(wpx), plate_solve_fov_deg=None, fits_header=hdr
            )
            _vsx_p3 = require_vsx_local_db_path(_cfg_plan.vsx_local_db_path)
            # B-cap fix: variable_targets are driven by the FRAME footprint (bbox + the same
            # 50px margin used by the in-frame pixel filter below), not the 3.5 deg cone box. The
            # cone box hit the 15000-row catalog_query_max_rows cap (no ORDER BY) and silently
            # dropped a contiguous Dec slice (northern half of the field, incl. bright named
            # variables). The frame bbox is tiny -> no cap -> spatial-first completeness. Note this
            # list also drives the global comparison-pool veto, so the now-complete variable set
            # correctly purges newly-recognised variables from the comp ensemble (Milan-approved,
            # CURSOR_RESULT_round1).
            vsx_df = _query_vsx_local_frame_bbox(
                wcs=w0,
                width_px=int(wpx),
                height_px=int(h),
                vsx_db_path=_vsx_p3,
                margin_px=50.0,
                center=center,
                require_db=True,
            )
            n_vsx_in_cone = int(len(vsx_df)) if vsx_df is not None else 0
            _vsx_n_cone = n_vsx_in_cone

            # Masterstars catalogue (Phase 0 DAO membership only; not the VSX->Gaia match RHS).
            ga = df.copy()
            if "catalog_id" in ga.columns and "ra_deg" in ga.columns and "dec_deg" in ga.columns:
                ga["catalog_id"] = ga["catalog_id"].fillna("").astype(str).str.strip()
                ga = ga[ga["catalog_id"].ne("")].copy()
                ga["ra_deg"] = pd.to_numeric(ga["ra_deg"], errors="coerce")
                ga["dec_deg"] = pd.to_numeric(ga["dec_deg"], errors="coerce")
                ga = ga[ga["ra_deg"].notna() & ga["dec_deg"].notna()].copy()
            else:
                ga = ga.iloc[0:0].copy()
            from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

            _ms_ids: set[str] = set()
            if not ga.empty:
                _ms_ids = {
                    str(normalize_gaia_source_id(x)).strip()
                    for x in ga["catalog_id"].astype(str).tolist()
                    if str(x).strip()
                }

            if vsx_df is not None and not vsx_df.empty and "ra_deg" in vsx_df.columns and "dec_deg" in vsx_df.columns:
                v = vsx_df.copy()
                v["ra_deg"] = pd.to_numeric(v["ra_deg"], errors="coerce")
                v["dec_deg"] = pd.to_numeric(v["dec_deg"], errors="coerce")
                v = v[v["ra_deg"].notna() & v["dec_deg"].notna()].copy()
                if not v.empty:
                    # Pixel coords from WCS (origin=0).
                    xy = w0.all_world2pix(v["ra_deg"].astype(float).to_numpy(), v["dec_deg"].astype(float).to_numpy(), 0)
                    x = np.asarray(xy[0], dtype=float)
                    y = np.asarray(xy[1], dtype=float)
                    v["x"] = x
                    v["y"] = y
                    in_frame = (v["x"] >= -50.0) & (v["y"] >= -50.0) & (v["x"] <= float(wpx) + 50.0) & (v["y"] <= float(h) + 50.0)
                    v = v.loc[in_frame].copy()

                n_vsx_in_frame = int(len(v))

                # Density-aware VSX -> Gaia DR3 cross-match (Marrese 2017/2019; Sutherland & Saunders 1992).
                # RHS is the deep local Gaia catalogue over the frame bbox; masterstars membership is derived after.
                cat_id_out = [""] * int(len(v))
                mag_out: list[float] = [float("nan")] * int(len(v))
                zone_out: list[str] = [""] * int(len(v))
                sep_out: list[float] = [float("nan")] * int(len(v))
                quality_out: list[str] = [""] * int(len(v))
                source_out: list[str] = [""] * int(len(v))
                _cm_diag: dict[str, Any] = {}
                if len(v) > 0:
                    from database import get_gaia_db_max_g_mag  # noqa: PLC0415
                    from vsx_gaia_crossmatch import (  # noqa: PLC0415
                        VsxGaiaCrossmatchError,
                        field_area_deg2_from_wcs,
                        match_vsx_to_gaia_density_aware,
                        query_gaia_for_frame_bbox,
                    )

                    try:
                        gaia_db = str(_cfg_plan.gaia_db_path or "").strip()
                    except Exception:  # noqa: BLE001
                        gaia_db = ""
                    if not gaia_db:
                        raise RuntimeError("VSX->Gaia cross-match refused: gaia_db_path not configured")

                    _field_area = field_area_deg2_from_wcs(w0, int(wpx), int(h))
                    _gaia_db_max_g = float(get_gaia_db_max_g_mag(gaia_db))
                    _gaia_rows = query_gaia_for_frame_bbox(
                        gaia_db,
                        w0,
                        int(wpx),
                        int(h),
                        margin_px=50.0,
                        center=center,
                    )
                    _gdf = pd.DataFrame(_gaia_rows) if _gaia_rows else pd.DataFrame()
                    if _gdf.empty:
                        raise RuntimeError(
                            "VSX->Gaia cross-match refused: no Gaia DR3 sources in frame bbox"
                        )
                    _sid_col = "source_id" if "source_id" in _gdf.columns else "catalog_id"
                    _ra_col = "ra" if "ra" in _gdf.columns else "ra_deg"
                    _dec_col = "dec" if "dec" in _gdf.columns else "dec_deg"
                    _gdf = _gdf.copy()
                    _gdf["catalog_id"] = _gdf[_sid_col].apply(
                        lambda x: str(normalize_gaia_source_id(x)).strip()
                    )
                    _gdf["ra_deg"] = pd.to_numeric(_gdf[_ra_col], errors="coerce")
                    _gdf["dec_deg"] = pd.to_numeric(_gdf[_dec_col], errors="coerce")
                    _gdf = _gdf[
                        _gdf["catalog_id"].ne("")
                        & _gdf["ra_deg"].notna()
                        & _gdf["dec_deg"].notna()
                    ].copy()
                    _pmra = (
                        pd.to_numeric(_gdf["pmra"], errors="coerce").to_numpy(dtype=float)
                        if "pmra" in _gdf.columns
                        else None
                    )
                    _pmdec = (
                        pd.to_numeric(_gdf["pmdec"], errors="coerce").to_numpy(dtype=float)
                        if "pmdec" in _gdf.columns
                        else None
                    )
                    _vsx_mag_max = (
                        pd.to_numeric(v.get("mag_max"), errors="coerce").to_numpy(dtype=float)
                        if "mag_max" in v.columns
                        else None
                    )
                    try:
                        _match_rows, _cm_diag_obj = match_vsx_to_gaia_density_aware(
                            v["ra_deg"].astype(float).to_numpy(),
                            v["dec_deg"].astype(float).to_numpy(),
                            _gdf["catalog_id"].astype(str).to_numpy(),
                            _gdf["ra_deg"].astype(float).to_numpy(),
                            _gdf["dec_deg"].astype(float).to_numpy(),
                            field_area_deg2=float(_field_area),
                            pmra=_pmra,
                            pmdec=_pmdec,
                            gaia_db_max_g=_gaia_db_max_g,
                            vsx_mag_max=_vsx_mag_max,
                            masterstars_ids=_ms_ids,
                        )
                        _cm_diag = {
                            "q_fit": _cm_diag_obj.q_fit,
                            "w_fit": _cm_diag_obj.w_fit,
                            "sigma_narrow_arcsec": _cm_diag_obj.sigma_narrow_arcsec,
                            "sigma_broad_arcsec": _cm_diag_obj.sigma_broad_arcsec,
                            "sigma_arcsec": _cm_diag_obj.sigma_narrow_arcsec,
                            "rho_per_deg2": _cm_diag_obj.rho_per_deg2,
                            "mean_nn_arcsec": _cm_diag_obj.mean_nn_arcsec,
                            "pm_path": _cm_diag_obj.pm_path,
                            "pm_columns_present": _cm_diag_obj.pm_columns_present,
                            "n_pm_finite": _cm_diag_obj.n_pm_finite,
                            "n_vsx": _cm_diag_obj.n_vsx,
                            "n_gaia": _cm_diag_obj.n_gaia,
                            "n_accepted": _cm_diag_obj.n_accepted,
                            "expected_contamination_fraction": _cm_diag_obj.expected_contamination_fraction,
                            "r_max_arcsec": _cm_diag_obj.r_max_arcsec,
                            "candidate_multiplicity": dict(_cm_diag_obj.candidate_multiplicity),
                            "multi_candidate_fraction": _cm_diag_obj.multi_candidate_fraction,
                            "fit_degenerate_warn": _cm_diag_obj.fit_degenerate_warn,
                            "gaia_db_max_g": _cm_diag_obj.gaia_db_max_g,
                            "vsx_fainter_than_gaia_db": _cm_diag_obj.vsx_fainter_than_gaia_db,
                            "sep_quantiles_before_pm": _cm_diag_obj.sep_quantiles_before_pm,
                            "sep_quantiles_after_pm": _cm_diag_obj.sep_quantiles_after_pm,
                            "sep_quantiles_accepted": _cm_diag_obj.sep_quantiles_accepted,
                            "masterstars_in_frame": _cm_diag_obj.masterstars_in_frame,
                            "masterstars_eligible": _cm_diag_obj.masterstars_eligible,
                            "masterstars_accepted": _cm_diag_obj.masterstars_accepted,
                            "outcome_check": _cm_diag_obj.outcome_check,
                        }
                        _gaia_by_cid = {
                            str(r["catalog_id"]): r for _, r in _gdf.iterrows()
                        }
                        for i, mr in enumerate(_match_rows):
                            if mr.accepted and mr.catalog_id:
                                cat_id_out[i] = mr.catalog_id
                                sep_out[i] = float(mr.sep_arcsec)
                                quality_out[i] = mr.quality
                                if mr.catalog_id in _ms_ids:
                                    source_out[i] = "masterstars"
                                    gro = ga.loc[ga["catalog_id"].astype(str) == mr.catalog_id]
                                    if not gro.empty:
                                        gro = gro.iloc[0]
                                        try:
                                            _mg = gro.get("mag")
                                            mag_out[i] = (
                                                float(_mg)
                                                if _mg is not None and str(_mg).strip() != ""
                                                else float("nan")
                                            )
                                        except (TypeError, ValueError):
                                            mag_out[i] = float("nan")
                                        try:
                                            zr = gro.get("zone")
                                            zone_out[i] = str(zr).strip().lower() if zr is not None else ""
                                        except Exception:  # noqa: BLE001
                                            zone_out[i] = ""
                                else:
                                    source_out[i] = "gaia_dr3_direct"
                                    grow = _gaia_by_cid.get(mr.catalog_id)
                                    if grow is not None:
                                        try:
                                            _gm = grow.get("g_mag")
                                            mag_out[i] = (
                                                float(_gm)
                                                if _gm is not None and str(_gm).strip() != ""
                                                else float("nan")
                                            )
                                        except (TypeError, ValueError):
                                            mag_out[i] = float("nan")
                            else:
                                source_out[i] = "no_match"
                                try:
                                    vsx_name0 = str(v.iloc[i].get("name", "") or "").strip()
                                    vsx_ra = float(v.iloc[i]["ra_deg"])
                                    vsx_dec = float(v.iloc[i]["dec_deg"])
                                    log_event(
                                        f"VSX no Gaia match: {vsx_name0} ra={vsx_ra:.4f} dec={vsx_dec:.4f} "
                                        "- hviezda nebude sledovana"
                                    )
                                except Exception:  # noqa: BLE001
                                    pass
                    except VsxGaiaCrossmatchError as _cm_exc:
                        log_event(f"VSX->Gaia DR3 cross-match refused: {_cm_exc!s}")
                        raise RuntimeError(str(_cm_exc)) from _cm_exc

                # Period column varies by VSX schema.
                _pcol = None
                for c in ("period", "varperiod", "var_period", "Period", "VarPeriod", "Var_Period"):
                    if c in v.columns:
                        _pcol = c
                        break
                if _pcol is None:
                    v["vsx_period"] = np.nan
                else:
                    v["vsx_period"] = pd.to_numeric(v[_pcol], errors="coerce")

                vname = v.get("name", pd.Series([""] * len(v))).fillna("").astype(str).str.strip()
                vtype = v.get("var_type", pd.Series([""] * len(v))).fillna("").astype(str).str.strip()
                notes = []
                for t, p in zip(vtype.tolist(), v["vsx_period"].tolist(), strict=False):
                    t0 = str(t or "").strip()
                    p0 = float(p) if p is not None and pd.notna(p) else None
                    if p0 is not None:
                        notes.append(f"{t0} P={p0}")
                    else:
                        notes.append(t0)

                _mmx = pd.to_numeric(v.get("mag_max"), errors="coerce") if "mag_max" in v.columns else pd.Series(
                    [float("nan")] * len(v), dtype=float
                )
                vsx_out = pd.DataFrame(
                    {
                        "name": vname,
                        "catalog_id": cat_id_out,
                        "catalog": "VSX",
                        "ra_deg": v["ra_deg"].astype(float).to_numpy(),
                        "dec_deg": v["dec_deg"].astype(float).to_numpy(),
                        "priority": 1,
                        "notes": notes,
                        "vsx_name": vname,
                        "vsx_type": vtype,
                        "vsx_period": v["vsx_period"].to_numpy(),
                        "x": pd.to_numeric(v.get("x"), errors="coerce"),
                        "y": pd.to_numeric(v.get("y"), errors="coerce"),
                        "mag": np.asarray(mag_out, dtype=float),
                        "zone": zone_out,
                        "gaia_match_arcsec": np.asarray(sep_out, dtype=float),
                        "gaia_match_quality": quality_out,
                        "gaia_match_source": source_out,
                        "vsx_mag_max": _mmx.to_numpy(dtype=float),
                    }
                )
                n_gaia_ok = int(sum(1 for c in cat_id_out if str(c).strip()))
                _zone_hist: dict[str, int] = {}
                n_phase0_hint = int(n_gaia_ok)
                for i in range(len(zone_out)):
                    if not str(cat_id_out[i]).strip():
                        continue
                    z = str(zone_out[i]).strip().lower()
                    _zone_hist[z] = int(_zone_hist.get(z, 0)) + 1
                _vsx_diag = {
                    "vsx_rows_in_frame_bbox": int(n_vsx_in_cone),
                    "vsx_rows_after_in_frame_margin": int(n_vsx_in_frame),
                    "n_with_masterstar_match": int(n_gaia_ok),
                    "vsx_rows_written_csv": int(len(vsx_out)),
                    "gaia_matches_within_10arcsec": int(n_gaia_ok),
                    "masterstars_zone_counts_among_gaia_matched": _zone_hist,
                    "phase0_active_targets_hint_count": int(n_phase0_hint),
                    "vsx_gaia_crossmatch": dict(_cm_diag),
                    "phase0_note_sk": (
                        "Faza 0 (`select_active_targets` v photometry_core.py): vsetky zony z masterstars "
                        "prejdu do active_targets.csv s `zone_flag`; saturovane maju `skip_photometry=True` "
                        "a Faza 2A ich nefotometruje (pozri `run_phase2a`). Vylucene su len prazdny "
                        "`catalog_id` a ciele mimo snimky (okrajovy filter)."
                    ),
                }
                _vsx_n_cone = int(n_vsx_in_cone)
    except VSXCatalogError:
        raise
    except Exception as _vsx_exc:  # noqa: BLE001
        log_event(f"variable_targets.csv (VSX export) preskoceny: {_vsx_exc!s}")
        vsx_out = pd.DataFrame(columns=var_cols)

    exo_promo = _build_exoplanet_promotion_rows_from_masterstars(
        df,
        hdr,
        _cfg_plan,
        frame_w_px=int(wpx),
        frame_h_px=int(h),
        margin_px=50.0,
    )
    merged_var = _merge_vsx_exoplanet_variable_targets(vsx_out, exo_promo)
    _proximity_veto_df: pd.DataFrame | None = None
    if merged_var is not None and not merged_var.empty:
        _ra_v = pd.to_numeric(merged_var.get("ra_deg"), errors="coerce")
        _de_v = pd.to_numeric(merged_var.get("dec_deg"), errors="coerce")
        _proximity_veto_df = merged_var.loc[_ra_v.notna() & _de_v.notna()].copy()

    comp_df, cmeta = select_comparison_stars_spatial_grid(
        df,
        width_px=float(wpx),
        height_px=float(h),
        n_comp=int(n_comparison_stars),
        require_non_variable=bool(require_non_variable),
        variable_targets_df=_proximity_veto_df,
        safe_bbox=_safe_bbox,
    )
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in comp_df.columns:
            comp_df = comp_df.copy()
            comp_df["catalog_id"] = normalize_gaia_source_id_series(comp_df["catalog_id"])
    except Exception:  # noqa: BLE001
        pass
    _vyvar_df_to_csv(comp_df, comp_path)

    # Always overwrite (even if it exists) so UI sees current field cone + exo promotions.
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in merged_var.columns:
            merged_var = merged_var.copy()
            merged_var["catalog_id"] = normalize_gaia_source_id_series(merged_var["catalog_id"])
    except Exception:  # noqa: BLE001
        pass
    _vyvar_df_to_csv(merged_var, var_path)
    if _vsx_diag:
        log_event(
            "variable_targets.csv (VSX): "
            f"VSX in frame bbox={_vsx_diag.get('vsx_rows_in_frame_bbox')} -> "
            f"in-frame margin={_vsx_diag.get('vsx_rows_after_in_frame_margin')} -> "
            f"detection-limited (DAO+Gaia match)={_vsx_diag.get('n_with_masterstar_match')} -> "
            f"CSV={int(len(merged_var))}. "
            f"Odhad VSX s Gaia ID (Faza 0 potom cross-match na masterstars): {_vsx_diag.get('phase0_active_targets_hint_count')}."
        )
    else:
        log_event(
            f"variable_targets.csv (VSX export): cone={_vsx_n_cone} -> zapisane={int(len(merged_var))} "
            f"({var_path.name})"
        )

    plan_path = ps / "photometry_plan.json"
    plan = {
        "purpose": "VYVAR photometry: comparison ensemble + variable targets",
        "comparison_stars": str(comp_path),
        "variable_targets_template": str(var_path),
        "masterstars": str(masterstars_csv),
        "masterstar_fits": str(masterstar_fits),
        "comparison_selection": cmeta,
        "comp_selection_enrichment": _enrichment_provenance,
        "variable_targets_diagnostics": _vsx_diag,
        "safe_bbox_px": list(_safe_bbox) if _safe_bbox is not None else None,
        "safe_bbox_r_out_px": float(_r_out) if _r_out is not None else None,
        "next_steps": [
            "Fill variable_targets.csv with programme stars (catalog IDs / coordinates).",
            "Use comparison_stars.csv comp_id / catalog_id to tie ensemble photometry on detrended_aligned frames.",
            "Run validate_comparison_ensemble_flatness on detrended_aligned frames to check comp stars stay flat vs ensemble.",
            "Light curves: aperture photometry per frame vs time (JD), then search for new variables vs field behavior.",
        ],
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    return {
        "comparison_stars_csv": str(comp_path),
        "variable_targets_csv": str(var_path),
        "photometry_plan_json": str(plan_path),
        "comparison_selection": cmeta,
        "comp_selection_enrichment": _enrichment_provenance,
    }


def _sync_comparison_stars_across_setups(platesolve_root: Path) -> None:
    """Copy comparison_stars.csv from a reference setup to all other setup dirs under platesolve/.

    Reference preference:
    - first setup containing 'R_' in its name with comparison_stars.csv
    - else first setup (sorted) that has comparison_stars.csv
    """
    try:
        root = Path(platesolve_root)
        if not root.is_dir():
            return
        setup_dirs = [
            d
            for d in root.iterdir()
            if d.is_dir() and (d / "per_frame_catalog_index.csv").exists()
        ]
        if len(setup_dirs) < 2:
            return
        ref_dir: Path | None = None
        for d in setup_dirs:
            if "R_" in d.name and (d / "comparison_stars.csv").exists():
                ref_dir = d
                break
        if ref_dir is None:
            for d in sorted(setup_dirs, key=lambda p: p.name.casefold()):
                if (d / "comparison_stars.csv").exists():
                    ref_dir = d
                    break
        if ref_dir is None:
            return
        ref_comp = ref_dir / "comparison_stars.csv"
        if not ref_comp.is_file():
            return
        import shutil

        for d in setup_dirs:
            if d == ref_dir:
                continue
            target_comp = d / "comparison_stars.csv"
            try:
                shutil.copy2(ref_comp, target_comp)
                log_event(
                    f"INFO: Comparison stars skopirovane z {ref_dir.name} -> {d.name}"
                )
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PIPE] comparison-star sync across setups skipped: %s", exc)
        return


def _catalog_match_radius_px(
    wcs_obj: Any,
    *,
    match_sep_arcsec: float,
    wpx: int,
    h: int,
) -> float:
    """Pixel matching radius for Gaia<->DAO (floor 10 px)."""
    import numpy as np
    from astropy.wcs.utils import proj_plane_pixel_scales

    floor_px = 10.0
    try:
        scales_deg = proj_plane_pixel_scales(wcs_obj)
        arcsec_per_px = float(np.mean(scales_deg)) * 3600.0
        if math.isfinite(arcsec_per_px) and arcsec_per_px > 0:
            return max(floor_px, float(match_sep_arcsec) / arcsec_per_px)
    except Exception:  # noqa: BLE001
        pass
    return floor_px


def _merge_dao_pass1_pass2_tables(
    tbl_pass1: Any,
    pass2_rows: list[dict[str, float]],
    *,
    bfac: int,
    dedup_px: float = 3.0,
) -> Any:
    """Append pass-2 detections; drop pass-2 if within ``dedup_px`` of pass-1 (full image coords).

    Adds integer column ``vy_dao_pass`` (1=pass1, 2=pass2) so the prematch peak gate can
    exempt pass-2 recoveries (SNR-GATE-01).
    """
    import numpy as np
    from astropy.table import Table, vstack

    if tbl_pass1 is not None and len(tbl_pass1) > 0 and "vy_dao_pass" not in tbl_pass1.colnames:
        tbl_pass1 = Table(tbl_pass1, copy=True)
        tbl_pass1["vy_dao_pass"] = np.ones(len(tbl_pass1), dtype=np.int16)

    if not pass2_rows:
        return tbl_pass1
    p1x = np.asarray([], dtype=np.float64)
    p1y = np.asarray([], dtype=np.float64)
    if tbl_pass1 is not None and len(tbl_pass1) > 0:
        xb = np.asarray(tbl_pass1["x_centroid"], dtype=np.float64)
        yb = np.asarray(tbl_pass1["y_centroid"], dtype=np.float64)
        p1x, p1y = _dao_xy_binned_to_full(xb, yb, int(bfac))
    kept: list[dict[str, float]] = []
    for row in pass2_rows:
        xf = float(row["x_full"])
        yf = float(row["y_full"])
        if p1x.size:
            d = np.hypot(p1x - xf, p1y - yf)
            if float(np.min(d)) < float(dedup_px):
                continue
        xb, yb = _dao_full_to_binned_xy(xf, yf, int(bfac))
        kept.append(
            {
                "x_centroid": xb,
                "y_centroid": yb,
                "flux": float(row["flux"]),
                "peak": float(row.get("peak", row["flux"])),
                "vy_dao_pass": 2,
            }
        )
    if not kept:
        return tbl_pass1
    t2 = Table(kept)
    if tbl_pass1 is None or len(tbl_pass1) == 0:
        return t2
    return vstack([tbl_pass1, t2])


def _dao_targeted_pass2_unmatched_gaia(
    data0: "np.ndarray",
    tbl_pass1: Any,
    *,
    cat_df: pd.DataFrame,
    wcs_obj: Any,
    bfac: int,
    fwhm_px: float,
    pass2_sigma: float,
    pass2_center_tol_px: float = 5.0,
    match_sep_arcsec: float,
    wpx: int,
    h: int,
) -> tuple[Any, int, int]:
    """Run local DAO on Gaia positions with no pass-1 DAO neighbor. Returns (merged_tbl, n_unmatched, n_pass2)."""
    import numpy as np
    from photutils.detection import DAOStarFinder  # type: ignore

    from masterstar_gaia_accounting import Pass2AcceptParams, dao_pass2_try_at_position

    if cat_df is None or cat_df.empty or "ra_deg" not in cat_df.columns or "dec_deg" not in cat_df.columns:
        return tbl_pass1, 0, 0

    ra = pd.to_numeric(cat_df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de = pd.to_numeric(cat_df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(ra) & np.isfinite(de)
    if not bool(ok.any()):
        return tbl_pass1, 0, 0

    gx, gy = wcs_obj.world_to_pixel_values(ra[ok], de[ok])
    inb = (gx >= 0) & (gx < float(wpx)) & (gy >= 0) & (gy < float(h))
    gx = gx[inb]
    gy = gy[inb]
    n_gaia_in = int(gx.size)
    if n_gaia_in == 0:
        return tbl_pass1, 0, 0

    match_r_px = _catalog_match_radius_px(wcs_obj, match_sep_arcsec=float(match_sep_arcsec), wpx=wpx, h=h)
    dao_x = np.asarray([], dtype=np.float64)
    dao_y = np.asarray([], dtype=np.float64)
    if tbl_pass1 is not None and len(tbl_pass1) > 0:
        xb = np.asarray(tbl_pass1["x_centroid"], dtype=np.float64)
        yb = np.asarray(tbl_pass1["y_centroid"], dtype=np.float64)
        dao_x, dao_y = _dao_xy_binned_to_full(xb, yb, int(bfac))

    unmatched_idx: list[int] = []
    if dao_x.size == 0:
        unmatched_idx = list(range(n_gaia_in))
    else:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(np.column_stack([dao_x, dao_y]))
        dist, _ = tree.query(np.column_stack([gx, gy]), distance_upper_bound=float(match_r_px))
        unmatched_idx = [i for i, d in enumerate(dist) if not np.isfinite(d) or float(d) > float(match_r_px)]

    n_unmatched = len(unmatched_idx)
    if n_unmatched == 0:
        return tbl_pass1, 0, 0

    sigma_p2 = max(1.5, min(20.0, float(pass2_sigma)))
    fwhm_cut = max(1.2, min(20.0, float(fwhm_px)))
    center_tol = max(0.5, min(10.0, float(pass2_center_tol_px)))
    p2_params = Pass2AcceptParams(
        sigma=sigma_p2,
        center_tol_px=center_tol,
        fwhm_px=float(fwhm_cut),
    )
    pass2_rows: list[dict[str, float]] = []
    n_empty_cutouts = 0

    for i in unmatched_idx:
        x0 = float(gx[i])
        y0 = float(gy[i])
        hit = dao_pass2_try_at_position(
            data0, x0, y0, wpx=int(wpx), h=int(h), params=p2_params
        )
        if not hit.get("accepted"):
            if str(hit.get("reason", "")) == "no_detection":
                n_empty_cutouts += 1
            continue
        pass2_rows.append(
            {
                "x_full": float(hit["x_det"]),
                "y_full": float(hit["y_det"]),
                "flux": float(hit["flux"]),
                "peak": float(hit.get("peak", hit["flux"])),
            }
        )

    if n_empty_cutouts > 0:
        LOGGER.info(
            "[DAO pass 2] %d/%d targeted cutouts had no detection "
            "(NoDetectionsWarning suppressed)",
            int(n_empty_cutouts),
            int(n_unmatched),
        )
    merged = _merge_dao_pass1_pass2_tables(tbl_pass1, pass2_rows, bfac=int(bfac), dedup_px=3.0)
    return merged, n_unmatched, len(pass2_rows)


def _merge_platesolve_gaia_pairs_into_masterstars_df(
    df: pd.DataFrame,
    *,
    pairs_x: list[float],
    pairs_y: list[float],
    pairs_ra: list[float],
    pairs_de: list[float],
    pairs_catalog_id: list[str],
    max_pair_px: float = 12.0,
) -> pd.DataFrame:
    """Map VYVAR plate-solve Gaia pairs onto DAO rows so ``astrometry_optimizer`` sees catalog_id + small sep."""
    import numpy as np

    if df is None or df.empty or not pairs_x:
        return df
    n = len(pairs_x)
    if len(pairs_y) != n or len(pairs_ra) != n or len(pairs_de) != n or len(pairs_catalog_id) != n:
        return df
    if "x" not in df.columns or "y" not in df.columns:
        return df
    out = df.copy()
    x = pd.to_numeric(out["x"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(out["y"], errors="coerce").to_numpy(dtype=np.float64)
    max2 = float(max_pair_px) ** 2
    used: set[int] = set()
    for i in range(n):
        cid = str(pairs_catalog_id[i] or "").strip()
        if not cid:
            continue
        d2 = (x - float(pairs_x[i])) ** 2 + (y - float(pairs_y[i])) ** 2
        d2[~np.isfinite(d2)] = np.inf
        j = int(np.argmin(d2))
        if j in used or float(d2[j]) > max2:
            continue
        used.add(j)
        if "catalog_id" in out.columns:
            out.iat[j, out.columns.get_loc("catalog_id")] = cid
        if "ra_deg" in out.columns:
            out.iat[j, out.columns.get_loc("ra_deg")] = float(pairs_ra[i])
        if "dec_deg" in out.columns:
            out.iat[j, out.columns.get_loc("dec_deg")] = float(pairs_de[i])
        if "match_sep_arcsec" in out.columns:
            out.iat[j, out.columns.get_loc("match_sep_arcsec")] = 0.25
    return out


def _fill_masterstars_gaia_matched_bp_rp_from_local_db(
    df: pd.DataFrame,
    *,
    gaia_db_path: str,
) -> tuple[pd.DataFrame, int, int]:
    """Dopln ``bp_rp`` / ``b_v`` z lokalnej Gaia SQLite pre ``GAIA_MATCHED`` bez farby v masterstars CSV.

    Frame-wide Gaia dotaz (``ORDER BY g_mag LIMIT``) casto vynecha uz omatchovane hviezdy na snimke;
    tato davka ide priamo podla ``source_id``.
    """
    if df is None or getattr(df, "empty", True):
        return df, 0, 0
    need = {"catalog_id", "bp_rp", "source_type"}
    if not need.issubset(df.columns):
        return df, 0, 0

    st_ok = df["source_type"].astype(str).str.strip().eq("GAIA_MATCHED")
    bpr = pd.to_numeric(df["bp_rp"], errors="coerce")
    gid = df["catalog_id"].map(normalize_gaia_source_id)
    gid_ok = gid.ne("")
    mask = st_ok & bpr.isna() & gid_ok
    n_missing = int(mask.sum())
    if n_missing <= 0:
        return df, 0, 0

    gdb = str(gaia_db_path or "").strip()
    try:
        gdb_ok = bool(gdb) and Path(gdb).is_file()
    except OSError:
        gdb_ok = False
    if not gdb_ok:
        return df, 0, n_missing

    out = df.copy()

    sub_idx = out.index[mask]
    keys_series = gid.loc[sub_idx]
    uniq_keys = sorted({k for k in keys_series.tolist() if k})
    if not uniq_keys:
        return out, 0, n_missing

    gaia_map = query_local_gaia_by_source_ids(gdb, uniq_keys)
    bprp_raw = keys_series.map(lambda k: (gaia_map.get(k) or {}).get("bp_rp"))
    bprp_num = pd.to_numeric(bprp_raw, errors="coerce")
    fill_ok = bprp_num.notna()
    to_fill = bprp_num.index[fill_ok]
    if len(to_fill) > 0:
        out.loc[to_fill, "bp_rp"] = bprp_num.loc[to_fill].astype(float)
    n_filled = int(fill_ok.sum())
    return out, n_filled, n_missing


def _pass2_sibling_wcs_recovery(
    *,
    reports: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    job_list: list[dict[str, Any]],
    align_kw: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pass 2: recover failed filter sub-groups via verified sibling WCS + bulk-shift."""
    from astrometry_align import _astrometry_align_impl_body  # noqa: PLC0415
    from pipeline import generate_masterstar_and_catalog  # noqa: PLC0415

    cfg = align_kw.get("app_config") or AppConfig()
    if not bool(getattr(cfg, "masterstar_sibling_recovery_enabled", True)):
        return reports, skipped
    if len(job_list) <= 1 or not skipped or not reports:
        return reports, skipped

    from vyvar_platesolver import (
        filter_code_from_setup_name,
        pick_sibling_donor_filter,
        try_recover_masterstar_sibling_wcs,
    )

    job_by_gkey = {str(j.get("gkey") or ""): j for j in job_list}
    report_by_gkey: dict[str, dict[str, Any]] = {}
    for rep in reports:
        gk = str(rep.get("observation_group_key") or "")
        if gk:
            report_by_gkey[gk] = rep

    verified_filters: set[str] = set()
    for gk in report_by_gkey:
        setup = Path(gk).name if gk else "(root)"
        flt = filter_code_from_setup_name(setup)
        if flt:
            verified_filters.add(flt)

    still_skipped: list[dict[str, Any]] = []
    archive_path = Path(align_kw["archive_path"])

    for sk in skipped:
        gkey = str(sk.get("gkey") or "")
        setup = str(sk.get("setup") or (Path(gkey).name if gkey else "(root)"))
        recipient_filter = filter_code_from_setup_name(setup)
        job = job_by_gkey.get(gkey)
        if not recipient_filter or job is None:
            still_skipped.append(sk)
            continue

        donor_filter = pick_sibling_donor_filter(recipient_filter, verified_filters)
        if donor_filter is None:
            still_skipped.append(sk)
            continue

        donor_gkey: str | None = None
        donor_report: dict[str, Any] | None = None
        for gk, rep in report_by_gkey.items():
            d_setup = Path(gk).name if gk else "(root)"
            if filter_code_from_setup_name(d_setup) == donor_filter:
                donor_gkey = gk
                donor_report = rep
                break
        donor_job = job_by_gkey.get(donor_gkey or "") if donor_gkey else None
        if donor_report is None or donor_job is None:
            still_skipped.append(sk)
            continue

        platesolve_dir = Path(job["platesolve_dir"])
        platesolve_dir.mkdir(parents=True, exist_ok=True)
        recipient_ms = platesolve_dir / "MASTERSTAR.fits"
        donor_ms_str = str(donor_report.get("masterstar_fits") or "").strip()
        donor_ms = (
            Path(donor_ms_str)
            if donor_ms_str
            else Path(donor_job["platesolve_dir"]) / "MASTERSTAR.fits"
        )

        try:
            if not recipient_ms.is_file():
                generate_masterstar_and_catalog(
                    archive_path=archive_path,
                    source_root=Path(job["detrended_root"]),
                    platesolve_dir=platesolve_dir,
                    setup_name=gkey or None,
                    masterstar_fits_only=True,
                    app_config=cfg,
                    equipment_id=align_kw.get("id_equipment"),
                    draft_id=align_kw.get("draft_id"),
                    master_dark_path=align_kw.get("master_dark_path"),
                    platesolve_backend=str(align_kw.get("platesolve_backend") or "vyvar"),
                    plate_solve_fov_deg=float(
                        align_kw.get("plate_solve_fov_deg") or cfg.plate_solve_fov_deg
                    ),
                )

            _bundle = _plate_solve_input_bundle(
                recipient_ms if recipient_ms.is_file() else donor_ms,
                app_config=cfg,
                equipment_id=align_kw.get("id_equipment"),
                draft_id=align_kw.get("draft_id"),
            )
            with fits.open(recipient_ms, memmap=False) as _hd_ms:
                _ms_hdr = _hd_ms[0].header
                _ms_data = _hd_ms[0].data
            _fov = resolve_plate_solve_fov_deg_hint(
                _ms_hdr,
                int(_ms_data.shape[0]),
                int(_ms_data.shape[1]),
                database_path=cfg.database_path,
                equipment_id=align_kw.get("id_equipment"),
                draft_id=align_kw.get("draft_id"),
            )
            if _fov is None:
                _fov = float(align_kw.get("plate_solve_fov_deg") or cfg.plate_solve_fov_deg)

            rec_result = try_recover_masterstar_sibling_wcs(
                recipient_masterstar_fits=recipient_ms,
                donor_masterstar_fits=donor_ms,
                recipient_filter=recipient_filter,
                donor_filter=donor_filter,
                frame_paths=[Path(f) for f in (job.get("files") or [])],
                app_config=cfg,
                plate_solve_fov_deg=float(_fov),
                expected_plate_scale_arcsec_per_px=_bundle.get("expected_arcsec_per_px"),
                effective_pixel_um=_bundle.get("eff_um"),
                focal_length_mm=_bundle.get("focal_mm"),
            )
            if not rec_result.get("confirmed"):
                still_skipped.append(sk)
                continue

            generate_masterstar_and_catalog(
                archive_path=archive_path,
                platesolve_dir=platesolve_dir,
                setup_name=gkey or None,
                masterstar_skip_build=True,
                masterstar_platesolve_skip_solve=True,
                app_config=cfg,
                equipment_id=align_kw.get("id_equipment"),
                draft_id=align_kw.get("draft_id"),
                platesolve_backend=str(align_kw.get("platesolve_backend") or "vyvar"),
                plate_solve_fov_deg=float(_fov),
                catalog_match_max_sep_arcsec=float(
                    align_kw.get("catalog_match_max_sep_arcsec") or 25.0
                ),
                max_catalog_rows=int(align_kw.get("max_catalog_rows") or 12000),
                dao_threshold_sigma=float(
                    align_kw.get("dao_threshold_sigma") or cfg.masterstar_dao_threshold_sigma
                ),
                catalog_local_gaia_only=align_kw.get("catalog_local_gaia_only"),
            )

            rep = _astrometry_align_impl_body(
                job=job,
                sibling_recovery_use_masterstar=True,
                build_masterstar_and_catalogs=False,
                **{
                    k: v
                    for k, v in align_kw.items()
                    if k not in {"build_masterstar_and_catalogs"}
                },
            )
            rep["sibling_recovery"] = rec_result
            reports.append(rep)
            report_by_gkey[gkey] = rep
            verified_filters.add(recipient_filter)
            log_event(
                f"SIBLING-WCS Pass2: recovered {setup} via donor {donor_filter} - "
                f"n_tight={((rec_result.get('after') or {}).get('n_matched_tight'))}"
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Sibling-WCS Pass2 failed for %s: %s", setup, exc)
            log_event(f"! Sibling-WCS Pass2: set {setup} zostava preskoceny - {exc}")
            still_skipped.append(sk)

    return reports, still_skipped


def _merge_astrometry_group_reports(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    if len(rows) == 1:
        return rows[0]
    tot_aligned = sum(int(x.get("aligned_frames") or 0) for x in rows)
    tot_in = sum(int(x.get("input_frames") or 0) for x in rows)
    merged = dict(rows[-1])
    merged["aligned_frames"] = int(tot_aligned)
    merged["input_frames"] = int(tot_in)
    merged["observation_subgroup_reports"] = rows
    merged["masterstar_fits"] = "; ".join(
        str(x.get("masterstar_fits") or "").strip()
        for x in rows
        if str(x.get("masterstar_fits") or "").strip()
    )
    merged["masterstars_csv"] = "; ".join(
        str(x.get("masterstars_csv") or "").strip()
        for x in rows
        if str(x.get("masterstars_csv") or "").strip()
    )
    log_event(
        f"Astrometria: dokoncene {len(rows)} pod-skupin - zarovnanych snimok spolu {tot_aligned} / {tot_in}."
    )
    return merged


def _run_osc_multi_group_alignment(
    *,
    job_list: list[dict[str, Any]],
    osc_bundles: dict[str, dict[str, dict[str, Any]]],
    align_kw: dict[str, Any],
) -> dict[str, Any]:
    """OSC-2: oneRGGB full solve+align first; R/G/B reuse WCS + registration handoff."""
    from astrometry_align import _astrometry_align_impl_body  # noqa: PLC0415
    from pipeline import generate_masterstar_and_catalog  # noqa: PLC0415

    from osc_align import (
        OSC_REGISTRATION_HANDOFF,
        load_registration_handoff,
        log_channel_match_rate_verification,
        propagate_wcs_between_fits,
        require_osc_donor_products,
        write_wcs_propagation_meta,
    )

    build_ms = bool(align_kw.get("build_masterstar_and_catalogs"))
    reports: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    archive_path = Path(align_kw["archive_path"])
    cfg = align_kw.get("app_config") or AppConfig()

    for base in sorted(osc_bundles.keys(), key=str.casefold):
        bundle = osc_bundles[base]
        one_job = dict(bundle["oneRGGB"])
        one_job["osc_write_registration_handoff"] = True
        _setup = Path(str(one_job.get("gkey") or "")).name or base
        one_rggb_failed = False
        try:
            one_rep = _astrometry_align_impl_body(job=one_job, **align_kw)
            reports.append(one_rep)
        except Exception as exc:  # noqa: BLE001
            one_rggb_failed = True
            LOGGER.error("OSC oneRGGB alignment failed for %s: %s", _setup, exc)
            skipped.append({"gkey": one_job.get("gkey"), "setup": _setup, "skipped_reason": str(exc)})
            continue

        donor_ps = Path(one_job["platesolve_dir"])
        try:
            require_osc_donor_products(donor_ps)
            handoff = load_registration_handoff(donor_ps / OSC_REGISTRATION_HANDOFF)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("OSC donor products missing for %s: %s", base, exc)
            skipped.append({"gkey": one_job.get("gkey"), "setup": _setup, "skipped_reason": str(exc)})
            continue

        for ch in ("R", "G", "B"):
            ch_job = dict(bundle[ch])
            ch_setup = Path(str(ch_job.get("gkey") or "")).name or f"{base}_{ch}"
            ch_ps = Path(ch_job["platesolve_dir"])
            ch_ps.mkdir(parents=True, exist_ok=True)
            try:
                if build_ms:
                    generate_masterstar_and_catalog(
                        archive_path=archive_path,
                        source_root=Path(ch_job["detrended_root"]),
                        platesolve_dir=ch_ps,
                        setup_name=str(ch_job.get("gkey") or "") or None,
                        masterstar_fits_only=True,
                        app_config=cfg,
                        equipment_id=align_kw.get("id_equipment"),
                        draft_id=align_kw.get("draft_id"),
                        master_dark_path=align_kw.get("master_dark_path"),
                        platesolve_backend=str(align_kw.get("platesolve_backend") or "vyvar"),
                        plate_solve_fov_deg=float(
                            align_kw.get("plate_solve_fov_deg") or cfg.plate_solve_fov_deg
                        ),
                    )
                    ch_ms = ch_ps / "MASTERSTAR.fits"
                    propagate_wcs_between_fits(donor_ps / "MASTERSTAR.fits", ch_ms)
                    cat_info = generate_masterstar_and_catalog(
                        archive_path=archive_path,
                        platesolve_dir=ch_ps,
                        setup_name=str(ch_job.get("gkey") or "") or None,
                        masterstar_skip_build=True,
                        masterstar_platesolve_skip_solve=True,
                        app_config=cfg,
                        equipment_id=align_kw.get("id_equipment"),
                        draft_id=align_kw.get("draft_id"),
                        platesolve_backend=str(align_kw.get("platesolve_backend") or "vyvar"),
                        plate_solve_fov_deg=float(
                            align_kw.get("plate_solve_fov_deg") or cfg.plate_solve_fov_deg
                        ),
                        catalog_match_max_sep_arcsec=float(
                            align_kw.get("catalog_match_max_sep_arcsec") or 25.0
                        ),
                        max_catalog_rows=int(align_kw.get("max_catalog_rows") or 12000),
                        dao_threshold_sigma=float(
                            align_kw.get("dao_threshold_sigma") or cfg.masterstar_dao_threshold_sigma
                        ),
                        catalog_local_gaia_only=align_kw.get("catalog_local_gaia_only"),
                    )
                    mr = None
                    if isinstance(cat_info, dict):
                        sm = cat_info.get("solve_meta") or cat_info
                        if isinstance(sm, dict):
                            mr = sm.get("match_rate")
                    write_wcs_propagation_meta(
                        ch_ps,
                        donor_dir=donor_ps,
                        channel=str(ch),
                        match_rate=float(mr) if mr is not None else None,
                    )
                    log_channel_match_rate_verification(
                        channel=str(ch),
                        match_rate=float(mr) if mr is not None else None,
                        one_rggb_failed=one_rggb_failed,
                        log_event=log_event,
                    )
                ch_kw = {
                    k: v
                    for k, v in align_kw.items()
                    if k not in {"build_masterstar_and_catalogs"}
                }
                ch_rep = _astrometry_align_impl_body(
                    job=ch_job,
                    osc_registration_handoff=handoff,
                    sibling_recovery_use_masterstar=True,
                    build_masterstar_and_catalogs=False,
                    **ch_kw,
                )
                ch_rep["osc_channel"] = ch
                ch_rep["osc_donor"] = str(donor_ps)
                reports.append(ch_rep)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("OSC channel %s alignment failed for %s: %s", ch, ch_setup, exc)
                skipped.append({"gkey": ch_job.get("gkey"), "setup": ch_setup, "skipped_reason": str(exc)})

    from osc_align import parse_osc_channel_from_setup

    mono_jobs: list[dict[str, Any]] = []
    for j in job_list:
        gkey = str(j.get("gkey") or "")
        setup = Path(gkey).name if gkey else ""
        _, ch = parse_osc_channel_from_setup(setup)
        if ch is None:
            mono_jobs.append(j)

    for j in mono_jobs:
        try:
            reports.append(_astrometry_align_impl_body(job=j, **align_kw))
        except Exception as exc:  # noqa: BLE001
            setup = Path(str(j.get("gkey") or "")).name if j.get("gkey") else "(root)"
            skipped.append({"gkey": j.get("gkey"), "setup": setup, "skipped_reason": str(exc)})

    if not reports:
        raise RuntimeError(
            "OSC astrometry: no group succeeded. "
            + "; ".join(f"{s.get('setup')}: {s.get('skipped_reason')}" for s in skipped)
        )
    merged = _merge_astrometry_group_reports(reports)
    if skipped:
        merged["skipped_subgroups"] = skipped
    merged["osc_orchestration"] = True
    return merged


def astrometry_align_and_build_masterstar(
    *,
    archive_path: Path,
    astrometry_api_key: str | None = None,
    max_control_points: int = 80,
    min_detected_stars: int = 100,
    max_detected_stars: int = 500,
    platesolve_backend: str = "vyvar",
    plate_solve_fov_deg: float = 1.0,
    max_extra_platesolve: int = 0,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    max_catalog_rows: int = 12000,
    n_comparison_stars: int = 0,
    require_non_variable_comparisons: bool = True,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    id_equipment: int | None = None,
    draft_id: int | None = None,
    catalog_local_gaia_only: bool | None = None,
    build_masterstar_and_catalogs: bool = False,
    progress_cb: "callable | None" = None,
    ram_align_and_catalog: bool = False,
    app_config: AppConfig | None = None,
    masterstar_candidate_paths: "Sequence[str] | None" = None,
    masterstar_selection_pct: float | None = None,
    master_dark_path: Path | str | None = None,
) -> dict[str, Any]:
    """Astrometry + alignment + per-frame catalog CSV (mandatory outputs).

    Preprocessed frames under ``<archive>/processed/lights`` (or legacy ``detrended/lights``) are grouped
    by full relative parent folder (multi-observation / FILTER+EXP+BIN layout). Each group gets its own
    alignment, ``MASTERSTAR``, and ``platesolve/<group>/`` outputs. Frames stored directly in ``lights/``
    form a single group.
    """
    from astrometry_align import _astrometry_align_impl_body  # noqa: PLC0415
    from pipeline import find_qc_metrics_csv  # noqa: PLC0415

    ap = Path(archive_path)
    _cfg_align_root = app_config or AppConfig()
    input_root = _archive_preprocess_lights_root(
        ap,
        app_config=_cfg_align_root,
        draft_id=draft_id,
        db=None,
    )
    if not input_root.exists():
        log_event(f"[X] ERROR: Input path {input_root} does not exist! Trying fallback...")
        processed_lights = ap / "processed" / "lights"
        subdirs: list[Path] = []
        try:
            if processed_lights.is_dir():
                subdirs = [d for d in processed_lights.iterdir() if d.is_dir()]
        except Exception:  # noqa: BLE001
            subdirs = []
        if subdirs:
            subdirs = sorted(subdirs, key=lambda p: p.name.casefold())
            input_root = subdirs[0]
            log_event(f"[OK] Fallback found: {input_root}")
    det_top = input_root
    ali_top = ap / "detrended_aligned" / "lights"
    os.makedirs(str(ali_top), exist_ok=True)
    ps_top = ap / "platesolve"
    os.makedirs(str(ps_top), exist_ok=True)
    files_all = _iter_fits_recursive(det_top)
    _n_before_qc = len(files_all)
    _qc_csv = find_qc_metrics_csv(
        ap,
        app_config=_cfg_align_root,
        draft_id=draft_id,
        db=None,
    )
    if _qc_csv is None or not _qc_csv.is_file():
        raise FileNotFoundError(
            "Preprocess QC step required; run Analyze/preprocess first to produce qc_metrics.csv"
        )
    files_all, _qc_status_map = filter_files_by_qc_metrics_allowlist(files_all, _qc_csv)
    log_event(
        f"QC allowlist: using {len(files_all)}/{_n_before_qc} lights FITS "
        f"(qc_metrics.csv) for alignment from {det_top}"
    )
    from invariants_runtime import check_qc01_skipproc_alignment  # noqa: PLC0415

    check_qc01_skipproc_alignment(files_all, _qc_csv, meta={"invariants": []})
    _bayermask = None
    if id_equipment is not None and _cfg_align_root is not None:
        try:
            from database import VyvarDatabase

            _osc_db = VyvarDatabase(str(_cfg_align_root.database_path))
            _bayermask = _osc_db.get_equipment_bayermask(int(id_equipment))
        except Exception:  # noqa: BLE001
            _bayermask = None
    from invariants_runtime import check_osc01_channel_extraction_required  # noqa: PLC0415

    check_osc01_channel_extraction_required(
        files_all,
        equipment_bayermask=_bayermask,
        meta={"invariants": []},
    )
    _n_root_only = len(list(det_top.glob("*.fits"))) if det_top.exists() else 0
    _n_all = len(files_all)
    if _n_root_only != _n_all:
        log_event(
            f"FITS celkom: {_n_all} pod {det_top} ({_n_root_only} priamo v koreni; "
            f"ostatne v podpriecinkoch napr. filter/exp/binning)."
        )
    else:
        log_event(f"FITS celkom: {_n_all} v {det_top}")
    log_event(f"Alignment input root: {det_top}")
    if not files_all:
        raise FileNotFoundError(
            f"Chybaju FITS v {det_top}. Plate solve cita len **spracovane** snimky. "
            "Najprv spusti **MAKE MASTERSTAR** po kroku **Analyze** (zapis z "
            f"`{ap / 'calibrated' / 'lights'}` -> `{ap / 'processed' / 'lights'}` alebo starsie `{ap / 'detrended' / 'lights'}`)."
        )
    job_list: list[dict[str, Any]] = []
    # Group strictly by real folder structure under processed/detrended lights.
    # DO NOT create scan_<id> subfolders: users expect platesolve/ + detrended_aligned/ to mirror
    # processed/lights/<setup_name>/ layout (as in older drafts like draft_000231).
    groups = _partition_detrended_by_subfolder(files_all, det_top)
    for gkey in sorted(groups.keys()):
        gfs = groups[gkey]
        if not gfs:
            continue
        _setup_name = Path(gkey).name if gkey else ""
        if _setup_name:
            log_event(
                f"Alignment subgroup detected: setup_name={_setup_name} "
                f"(input={det_top / gkey}, files={len(gfs)})"
            )
        if gkey:
            job_list.append(
                {
                    "gkey": gkey,
                    "scanning_id": None,
                    "detrended_root": det_top / gkey,
                    "aligned_root": ali_top / gkey,
                    "platesolve_dir": ps_top / gkey,
                    "files": gfs,
                }
            )
        else:
            job_list.append(
                {
                    "gkey": "",
                    "scanning_id": None,
                    "detrended_root": det_top,
                    "aligned_root": ali_top,
                    "platesolve_dir": ps_top,
                    "files": gfs,
                }
            )
    if len(job_list) > 1:
        log_event(
            "Astrometria: viacero pod-pozorovani (podpriecinky v processed|detrended/lights) - "
            "samostatne zarovnanie, MASTERSTAR a katalogy pre kazdu skupinu."
        )
    from osc_align import partition_jobs_for_osc_alignment

    job_list, osc_meta = partition_jobs_for_osc_alignment(job_list)
    if osc_meta.get("has_osc_bundles"):
        from invariants_runtime import check_osc02_unified_frame_sets

        check_osc02_unified_frame_sets(osc_meta["bundles"], meta={"invariants": []})
    _ms_paths = [str(x).strip() for x in (masterstar_candidate_paths or []) if str(x).strip()]
    try:
        _ms_pct = float(masterstar_selection_pct) if masterstar_selection_pct is not None else None
    except (TypeError, ValueError):
        _ms_pct = None
    if _ms_pct is not None and (not math.isfinite(_ms_pct) or _ms_pct <= 0):
        _ms_pct = None
    _md_bpm_job: str | None = None
    if master_dark_path is not None and str(master_dark_path).strip():
        _mdp = Path(str(master_dark_path))
        if _mdp.is_file():
            _md_bpm_job = str(_mdp.resolve())
    _multi_obs = len(job_list) > 1
    for _j in job_list:
        _j["masterstar_candidate_paths"] = [] if _multi_obs else _ms_paths
        _j["masterstar_selection_pct"] = _ms_pct
        if _md_bpm_job:
            _j["master_dark_path"] = _md_bpm_job

    _kw: dict[str, Any] = dict(
        archive_path=archive_path,
        astrometry_api_key=astrometry_api_key,
        max_control_points=max_control_points,
        min_detected_stars=min_detected_stars,
        max_detected_stars=max_detected_stars,
        platesolve_backend=platesolve_backend,
        plate_solve_fov_deg=plate_solve_fov_deg,
        max_extra_platesolve=max_extra_platesolve,
        catalog_match_max_sep_arcsec=catalog_match_max_sep_arcsec,
        saturate_level_fraction=saturate_level_fraction,
        max_catalog_rows=max_catalog_rows,
        n_comparison_stars=n_comparison_stars,
        require_non_variable_comparisons=require_non_variable_comparisons,
        faintest_mag_limit=faintest_mag_limit,
        dao_threshold_sigma=dao_threshold_sigma,
        id_equipment=id_equipment,
        draft_id=draft_id,
        catalog_local_gaia_only=catalog_local_gaia_only,
        build_masterstar_and_catalogs=build_masterstar_and_catalogs,
        progress_cb=progress_cb,
        ram_align_and_catalog=ram_align_and_catalog,
        app_config=app_config,
    )
    if osc_meta.get("has_osc_bundles"):
        return _run_osc_multi_group_alignment(
            job_list=job_list,
            osc_bundles=osc_meta["bundles"],
            align_kw=_kw,
        )
    if len(job_list) == 1:
        return _astrometry_align_impl_body(job=job_list[0], **_kw)

    reports: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for j in job_list:
        _gkey = str(j.get("gkey") or "")
        _setup = Path(_gkey).name if _gkey else "(root)"
        try:
            reports.append(_astrometry_align_impl_body(job=j, **_kw))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Astrometria/MASTERSTAR failed for set %s: %s", _setup, exc)
            log_event(
                f"! Set {_setup}: plate-solve/MASTERSTAR zlyhal - set sa preskakuje, "
                f"pokracujem dalsim setom. Dovod: {exc}"
            )
            skipped.append(
                {"gkey": _gkey, "setup": _setup, "solved": False, "skipped_reason": str(exc)}
            )
            continue

    reports, skipped = _pass2_sibling_wcs_recovery(
        reports=reports,
        skipped=skipped,
        job_list=job_list,
        align_kw=_kw,
    )

    if not reports:
        raise RuntimeError(
            "Astrometria: ziadny set nepresiel plate-solve/MASTERSTAR. "
            + "; ".join(f"{s['setup']}: {s['skipped_reason']}" for s in skipped)
        )

    merged = _merge_astrometry_group_reports(reports)
    if skipped:
        merged["skipped_subgroups"] = skipped
        log_event(
            f"Astrometria: {len(reports)} setov OK, {len(skipped)} preskocenych: "
            + ", ".join(s["setup"] for s in skipped)
        )
    return merged


def _safe_proc_name(original_name: str) -> str:
    name = (original_name or "").strip()
    if not name:
        return "proc.fits"
    if name.lower().startswith("proc_"):
        return name
    return "proc_" + name


def _vyvar_per_frame_csv_workers(app_config: AppConfig | None = None) -> int:
    """Process workers for plate-solve step 3 per-frame catalog (DAO + match + CSV).

    Rovnaky zaklad ako QC/alignment (see :func:`_vyvar_parallel_worker_count`); dalsi strop podla skutocnej velkosti
    snimku riesi ``_vyvar_cap_mp_workers_for_catalog``.
    """
    return _vyvar_parallel_worker_count(app_config)


def _field_jump_empty_result() -> dict[str, Any]:
    """Return empty/unknown result when data is insufficient."""
    return {
        "has_jump": False,
        "n_groups": 0,
        "jump_frames": [],
        "groups": [],
        "dominant_group_frac": 1.0,
        "recommended_min_frames_frac": 0.3,
        "warning_text": None,
        "total_frames": 0,
    }


def detect_field_jumps(
    db: "VyvarDatabase",
    draft_id: int,
    jump_threshold_arcmin: float = 3.0,
    min_frames_in_group: int = 5,
) -> dict[str, Any]:
    """
    Detect sudden field position jumps from per-frame DRIFT values.

    Uses DRIFT_DRA / DRIFT_DDE from manifest files[] (already computed by
    sync_obs_files_drift_arcmin_for_draft).

    Returns dict with keys:
      has_jump: bool
      n_groups: int
      jump_frames: list of {frame_index, delta_arcmin, file}
      groups: list of {group_id, frame_start, frame_end,
                       n_frames, frac, is_dominant}
      dominant_group_frac: float
      recommended_min_frames_frac: float
      warning_text: str | None
      total_frames: int
    """
    import math

    import numpy as np  # noqa: F401
    import pandas as pd

    _ = (min_frames_in_group,)  # reserved for future grouping guards

    try:
        rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"[FIELD JUMP] DB fetch failed: {exc}")
        return _field_jump_empty_result()

    if not isinstance(rows, pd.DataFrame):
        rows = pd.DataFrame(rows)
    if rows.empty:
        return _field_jump_empty_result()

    if "FILE_PATH" not in rows.columns:
        return _field_jump_empty_result()

    # Filter: light frames if IMAGETYP is present (defensive; fetch_* already does this).
    if "IMAGETYP" in rows.columns:
        rows = rows[rows["IMAGETYP"].astype(str).str.contains("light", case=False, na=False)]

    # Sort by FILE_PATH (typically chronological on acquisition naming).
    rows = rows.sort_values("FILE_PATH").reset_index(drop=True)

    # Require DRIFT plane offsets.
    for col in ("DRIFT_DRA", "DRIFT_DDE", "DRIFT"):
        if col not in rows.columns:
            return _field_jump_empty_result()
        rows[col] = pd.to_numeric(rows[col], errors="coerce")

    # Keep only frames with finite DRIFT_DRA/DRIFT_DDE.
    valid = rows.dropna(subset=["DRIFT_DRA", "DRIFT_DDE"]).copy().reset_index(drop=True)
    if len(valid) < 3:
        return _field_jump_empty_result()

    total = int(len(valid))

    # Frame-to-frame delta (arcmin) in the DRIFT plane (degrees).
    dra = valid["DRIFT_DRA"].to_numpy(dtype=float)
    dde = valid["DRIFT_DDE"].to_numpy(dtype=float)

    deltas: list[float] = []
    for i in range(1, total):
        d = math.sqrt((dra[i] - dra[i - 1]) ** 2 + (dde[i] - dde[i - 1]) ** 2) * 60.0
        deltas.append(float(d))

    # Identify jumps.
    jump_indices: list[dict[str, Any]] = []
    thr = float(jump_threshold_arcmin)
    if not (math.isfinite(thr) and thr > 0):
        thr = 3.0
    for i, d in enumerate(deltas):
        if math.isfinite(float(d)) and float(d) > thr:
            jump_indices.append(
                {
                    "frame_index": int(i + 1),  # boundary at i+1 (0-based index in `valid`)
                    "delta_arcmin": round(float(d), 2),
                    "file": str(valid.iloc[i + 1].get("FILE_PATH", "")),
                }
            )

    if not jump_indices:
        return {
            "has_jump": False,
            "n_groups": 1,
            "jump_frames": [],
            "groups": [
                {
                    "group_id": 1,
                    "frame_start": 0,
                    "frame_end": total - 1,
                    "n_frames": total,
                    "frac": 1.0,
                    "is_dominant": True,
                }
            ],
            "dominant_group_frac": 1.0,
            "recommended_min_frames_frac": 0.3,
            "warning_text": None,
            "total_frames": total,
        }

    # Split into groups using jump boundaries.
    boundaries = [0] + [int(j["frame_index"]) for j in jump_indices] + [total]
    groups: list[dict[str, Any]] = []
    for gid, (start, end) in enumerate(itertools.pairwise(boundaries), 1):
        n = int(end - start)
        groups.append(
            {
                "group_id": int(gid),
                "frame_start": int(start),
                "frame_end": int(end - 1),
                "n_frames": n,
                "frac": round(float(n) / float(total), 3),
                "is_dominant": False,
            }
        )

    dominant = max(groups, key=lambda g: int(g.get("n_frames", 0) or 0))
    dominant["is_dominant"] = True
    dom_frac = float(dominant.get("frac", 0.0) or 0.0)

    recommended = round(dom_frac * 0.95, 2)
    recommended = max(0.3, min(0.95, float(recommended)))

    jump_frame_nos = [int(j["frame_index"]) for j in jump_indices]
    warning = (
        "Field jump detected at frame(s) "
        f"{', '.join(str(f) for f in jump_frame_nos)} "
        f"(threshold {thr:.1f}'). "
        f"{len(groups)} position groups found. "
        f"Dominant group: {int(dominant['n_frames'])}/{total} frames "
        f"({dom_frac * 100:.0f}%). "
        f"Recommended min_frames_frac >= {recommended:.2f}."
    )

    return {
        "has_jump": True,
        "n_groups": int(len(groups)),
        "jump_frames": jump_indices,
        "groups": groups,
        "dominant_group_frac": float(dom_frac),
        "recommended_min_frames_frac": float(recommended),
        "warning_text": warning,
        "total_frames": total,
    }


