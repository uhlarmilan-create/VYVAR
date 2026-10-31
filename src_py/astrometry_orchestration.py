"""Astrometry stage orchestration: alignment + MASTERSTAR per setup, OSC groups, sibling WCS recovery.

LAYER-2026-10 (code review 2 T13): moved out of ``pipeline_astrometry`` so the helper module no longer
calls back into ``astrometry_align`` / ``masterstar_build`` (which import its helpers).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from astropy.io import fits

from config import AppConfig
from infolog import log_event
from pipeline_calibrate import _archive_preprocess_lights_root
from pipeline_preprocess import _partition_detrended_by_subfolder, filter_files_by_qc_metrics_allowlist
from fits_suffixes import list_fits_files
from utils import iter_fits_paths_recursive as _iter_fits_recursive
from pipeline_astrometry import (
    LOGGER,
    _solve_wcs_external,
    _wcs_field_center_radec_deg,
    filter_code_from_setup_name,
    pick_sibling_donor_setup,
)


# --- moved from pipeline_astrometry (LAYER-2026-10) ---


def _pass2_sibling_wcs_recovery(
    *,
    reports: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    job_list: list[dict[str, Any]],
    align_kw: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pass 2: a filter sub-group whose MASTERSTAR did not solve is solved again by astrosolve with
    the field centre and plate scale of a solved sibling filter as the hint (BLIND-SOLVE-2026-10,
    Milan 2026-10-20: one plate-solve code; the old bulk-shift WCS copy with its own matching and
    odds is removed). The result is verified and refined like any solve."""
    from astropy.wcs import WCS  # noqa: PLC0415
    from astropy.wcs.utils import proj_plane_pixel_scales  # noqa: PLC0415

    from astrometry_align import _astrometry_align_impl_body  # noqa: PLC0415
    from masterstar_build import generate_masterstar_and_catalog  # noqa: PLC0415

    cfg = align_kw["app_config"]
    if len(job_list) <= 1 or not skipped or not reports:
        return reports, skipped

    job_by_gkey = {str(j.get("gkey") or ""): j for j in job_list}
    report_by_gkey: dict[str, dict[str, Any]] = {}
    for rep in reports:
        gk = str(rep.get("observation_group_key") or "")
        if gk:
            report_by_gkey[gk] = rep

    # G03-35: donors keyed by setup (same filter first, then the nearest wavelength)
    solved_setup_gkey: dict[str, str] = {}
    for gk in report_by_gkey:
        setup = Path(gk).name if gk else "(root)"
        if filter_code_from_setup_name(setup):
            solved_setup_gkey.setdefault(setup, gk)

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

        donor_setup = pick_sibling_donor_setup(setup, list(solved_setup_gkey))
        if donor_setup is None:
            still_skipped.append(sk)
            continue
        donor_gkey: str | None = solved_setup_gkey[donor_setup]
        donor_report: dict[str, Any] | None = report_by_gkey.get(donor_gkey)
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
                )

            donor_centre = _wcs_field_center_radec_deg(donor_ms)
            with fits.open(donor_ms, memmap=False) as _hd_d:
                _wd = WCS(_hd_d[0].header).celestial
            donor_scale = float(np.mean(proj_plane_pixel_scales(_wd))) * 3600.0
            if donor_centre is None or not math.isfinite(donor_scale) or donor_scale <= 0:
                still_skipped.append(sk)
                continue
            rec_result = _solve_wcs_external(
                recipient_ms,
                hint_ra_deg=float(donor_centre[0]),
                hint_dec_deg=float(donor_centre[1]),
                scale_hint_arcsec=donor_scale,
                app_config=cfg,
            )
            rec_result["donor_setup"] = donor_setup
            rec_result["donor_filter"] = filter_code_from_setup_name(donor_setup)
            if not rec_result.get("solved"):
                log_event(
                    f"SIBLING Pass2: {setup} not solved with the {donor_setup} hint - "
                    f"{rec_result.get('message')}"
                )
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
            solved_setup_gkey.setdefault(setup, gkey)
            log_event(
                f"SIBLING Pass2: {setup} solved with the {donor_setup} hint - "
                f"{rec_result.get('n_match')} matches, rms {rec_result.get('rms_px')} px"
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Sibling Pass2 failed for %s: %s", setup, exc)
            log_event(f"! Sibling Pass2: set {setup} remains skipped - {exc}")
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
        f"Astrometry: finished {len(rows)} subgroups - aligned frames in total {tot_aligned} / {tot_in}."
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
    from masterstar_build import generate_masterstar_and_catalog  # noqa: PLC0415

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
    cfg = align_kw["app_config"]

    for base in sorted(osc_bundles.keys(), key=str.casefold):
        bundle = osc_bundles[base]
        one_job = dict(bundle["oneRGGB"])
        one_job["osc_write_registration_handoff"] = True
        _setup = Path(str(one_job.get("gkey") or "")).name or base
        try:
            one_rep = _astrometry_align_impl_body(job=one_job, **align_kw)
            reports.append(one_rep)
        except Exception as exc:  # noqa: BLE001
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
                    )
                    mr = None
                    if isinstance(cat_info, dict):
                        sm = cat_info.get("solve") or cat_info
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
    n_comparison_stars: int = 0,
    require_non_variable_comparisons: bool = True,
    id_equipment: int | None = None,
    draft_id: int | None = None,
    build_masterstar_and_catalogs: bool = False,
    progress_cb: "callable | None" = None,
    ram_align_and_catalog: bool = False,
    app_config: AppConfig,
    masterstar_candidate_paths: "Sequence[str] | None" = None,
    masterstar_selection_pct: float | None = None,
) -> dict[str, Any]:
    """Astrometry + alignment + per-frame catalog CSV (mandatory outputs).

    Preprocessed frames under ``<archive>/processed/lights`` (or legacy ``detrended/lights``) are grouped
    by full relative parent folder (multi-observation / FILTER+EXP+BIN layout). Each group gets its own
    alignment, ``MASTERSTAR``, and ``platesolve/<group>/`` outputs. Frames stored directly in ``lights/``
    form a single group.
    """
    from astrometry_align import _astrometry_align_impl_body  # noqa: PLC0415
    from pipeline_catalog import find_qc_metrics_csv  # noqa: PLC0415

    ap = Path(archive_path)
    _cfg_align_root = app_config
    input_root = _archive_preprocess_lights_root(
        ap,
        draft_id=draft_id,
        db=None,
    )
    if not input_root.exists():
        # FAIL-LOUD-2026-10 (code review 2 G03-37): the former fallback took the first subfolder of
        # processed/lights and silently dropped every other setup of the night
        raise FileNotFoundError(f"preprocessed lights folder missing: {input_root}")
    det_top = input_root
    ali_top = ap / "detrended_aligned" / "lights"
    os.makedirs(str(ali_top), exist_ok=True)
    ps_top = ap / "platesolve"
    os.makedirs(str(ps_top), exist_ok=True)
    files_all = _iter_fits_recursive(det_top)
    _n_before_qc = len(files_all)
    _qc_csv = find_qc_metrics_csv(
        ap,
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

            _osc_db = VyvarDatabase.for_config(_cfg_align_root)
            _bayermask = _osc_db.get_equipment_bayermask(int(id_equipment))
        except Exception:  # noqa: BLE001
            _bayermask = None
    from invariants_runtime import check_osc01_channel_extraction_required  # noqa: PLC0415

    check_osc01_channel_extraction_required(
        files_all,
        equipment_bayermask=_bayermask,
        meta={"invariants": []},
    )
    _n_root_only = len(list_fits_files(det_top))
    _n_all = len(files_all)
    if _n_root_only != _n_all:
        log_event(
            f"FITS total: {_n_all} under {det_top} ({_n_root_only} directly in the root; "
            f"the rest in subfolders, e.g. filter/exp/binning)."
        )
    else:
        log_event(f"FITS total: {_n_all} in {det_top}")
    log_event(f"Alignment input root: {det_top}")
    if not files_all:
        raise FileNotFoundError(
            f"No FITS in {det_top}. Plate solve reads only **processed** frames. "
            "Run **MAKE MASTERSTAR** after the **Analyze** step first (writes from "
            f"`{ap / 'calibrated' / 'lights'}` -> `{ap / 'processed' / 'lights'}` or older `{ap / 'detrended' / 'lights'}`)."
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
            "Astrometry: several sub-observations (subfolders in processed|detrended/lights) - "
            "separate alignment, MASTERSTAR and catalogs for each group."
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
    _multi_obs = len(job_list) > 1
    for _j in job_list:
        _j["masterstar_candidate_paths"] = [] if _multi_obs else _ms_paths
        _j["masterstar_selection_pct"] = _ms_pct

    _kw: dict[str, Any] = dict(
        archive_path=archive_path,
        n_comparison_stars=n_comparison_stars,
        require_non_variable_comparisons=require_non_variable_comparisons,
        id_equipment=id_equipment,
        draft_id=draft_id,
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
            LOGGER.error("Astrometry/MASTERSTAR failed for set %s: %s", _setup, exc)
            log_event(
                f"! Set {_setup}: plate-solve/MASTERSTAR failed - skipping the set, "
                f"continuing with the next set. Reason: {exc}"
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
            "Astrometry: no set passed plate-solve/MASTERSTAR. "
            + "; ".join(f"{s['setup']}: {s['skipped_reason']}" for s in skipped)
        )

    merged = _merge_astrometry_group_reports(reports)
    if skipped:
        merged["skipped_subgroups"] = skipped
        log_event(
            f"Astrometry: {len(reports)} sets OK, {len(skipped)} skipped: "
            + ", ".join(s["setup"] for s in skipped)
        )
    return merged
