from __future__ import annotations

import json
import math
import logging
import re
import shutil
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import numpy as np
from astropy.time import Time

from config import AppConfig
from citations import (
    build_run_citation_context,
    emit_export_citation_lines,
    emit_varastro_method_summary_lines,
    load_pipeline_meta,
)
from gaia_catalog_id import normalize_gaia_source_id
from invariants_runtime import InvariantViolation
from report_methods import aavso_export_path, lc_csv_path, varastro_export_path
from check_star_kmag import (
    check_kmag_sidecar_path,
)
from photometry_core import (
    _resolve_plate_scale_arcsec_per_px,
    parse_comp_quality_json_map,
)
from vyvar_version import VYVAR_SOFTWARE_VERSION

# Gaia ID must be str - float64 loses digits
_GAIA_ID_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}


INV_PSF_SUBMIT_01 = "INV-PSF-SUBMIT-01"


class ExportFailure(TypedDict):
    target_id: str
    method: str
    reason: str


def _accumulate_export_stat(export_stats: dict[str, int] | None, key: str, n: int = 1) -> None:
    if export_stats is None:
        return
    export_stats[key] = int(export_stats.get(key, 0)) + int(n)


def _maybe_set_airmass_citation(run_citation_ctx: Any, lc_df: pd.DataFrame) -> None:
    if run_citation_ctx is None:
        return
    from photometry_core import lc_has_finite_airmass  # noqa: PLC0415

    if lc_has_finite_airmass(lc_df):
        run_citation_ctx.use_airmass = True

def record_export_failure(
    failures: list[ExportFailure] | None,
    target_id: str,
    method: str,
    reason: str,
) -> None:
    """Append one export failure and log at ERROR (batch callers collect + summarize)."""
    if failures is None:
        logging.error(
            "[EXPORT] failed %s method=%s: %s",
            target_id or "?",
            method or "-",
            reason,
        )
        return
    failures.append(
        {
            "target_id": str(target_id or "").strip() or "?",
            "method": str(method or "").strip(),
            "reason": str(reason),
        }
    )
    logging.error(
        "[EXPORT] failed %s method=%s: %s",
        failures[-1]["target_id"],
        failures[-1]["method"] or "-",
        failures[-1]["reason"],
    )


def log_export_batch_summary(
    failures: list[ExportFailure],
    export_stats: dict[str, int] | None = None,
) -> None:
    """Emit operator-visible batch summary when any per-target exports failed or were empty."""
    if export_stats:
        _tb_ref = int(export_stats.get("time_base_refused") or 0)
        if _tb_ref:
            logging.error(
                "[EXPORT] time_base refused (non-BJD_TDB or unknown/mixed): %d target-method(s)",
                _tb_ref,
            )
    if not failures:
        return
    ids = sorted({f["target_id"] for f in failures})
    logging.error(
        "[EXPORT] batch finished with %d export failure(s) across %d target(s)",
        len(failures),
        len(ids),
    )
    if len(ids) <= 25:
        logging.error("[EXPORT] failed target ids: %s", ",".join(ids))
    else:
        logging.error(
            "[EXPORT] failed target ids (first 25): %s ...",
            ",".join(ids[:25]),
        )
    for f in failures:
        logging.error(
            "[EXPORT]   %s | method=%s | %s",
            f["target_id"],
            f.get("method") or "-",
            f["reason"],
        )


def _aavso_software_header_line(software_version: str) -> str:
    """Format AAVSO ``#SOFTWARE`` from the canonical version string."""
    ver = str(software_version or VYVAR_SOFTWARE_VERSION).strip() or VYVAR_SOFTWARE_VERSION
    soft_id = ver.replace(" ", "/") if "/" not in ver else ver
    return f"#SOFTWARE={soft_id} (aperture photometry; Broeg 2005 ensemble)\n"


def _resolve_export_arcsec_per_px(
    photometry_dir: Path,
    cfg: AppConfig,
) -> float | None:
    """Derive plate scale (arcsec/px) for export headers - derive-or-None (no magic 1.3).

    Priority: ``pipeline_meta.json`` ``plate_scale_arcsec_px`` (aperture photometry session);
    then WCS/CD from sibling ``MASTERSTAR.fits`` via ``_resolve_plate_scale_arcsec_per_px``.
    """
    meta = load_pipeline_meta(photometry_dir)
    ps = meta.get("plate_scale_arcsec_px")
    if ps is not None:
        try:
            v = float(ps)
            if math.isfinite(v) and v > 0:
                return float(v)
        except (TypeError, ValueError):
            pass
    masterstar = photometry_dir.parent / "MASTERSTAR.fits"
    if masterstar.is_file():
        try:
            v = _resolve_plate_scale_arcsec_per_px(cfg, masterstar)
            if v is not None and math.isfinite(float(v)) and float(v) > 0:
                return float(v)
        except Exception:  # noqa: BLE001
            # EXC-0078: T2 -- report/export may omit or misstate (if v is not None and math.isfinite(float(v)) and fl... (EXCEPT-BULK 2026-07-08)
            pass
    return None


def _resolved_site_from_meta(output_dir: Path | str | None) -> dict[str, Any] | None:
    """Read the per-draft resolved observer site from ``pipeline_meta.json``.

    Aperture photometry persists ``observer_location`` (param_resolver: draft ID_LOCATION ->
    header SITELAT -> flagged config) with lat/lon/alt/source. Exported coordinates
    must match the site used for BJD/airmass, so we prefer this over ``cfg`` (which
    tracks the *last* session and can drift when reprocessing an old draft).
    """
    if output_dir is None:
        return None
    try:
        pm = Path(output_dir) / "pipeline_meta.json"
        if not pm.is_file():
            return None
        meta = json.loads(pm.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().export_observer_location_read_fail += 1
        logging.warning(
            "[EXPORT] observer_location read failed from %s: %s",
            pm,
            exc,
        )
        return None
    loc = meta.get("observer_location") if isinstance(meta, dict) else None
    if not isinstance(loc, dict):
        return None
    try:
        lat = float(loc.get("lat"))
        lon = float(loc.get("lon"))
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)) or (lat == 0.0 and lon == 0.0):
        return None
    try:
        alt = float(loc.get("alt_m", 0.0) or 0.0)
    except (TypeError, ValueError):
        alt = 0.0
    return {
        "lat": lat,
        "lon": lon,
        "alt_m": alt,
        "name": str(loc.get("name", "") or "").strip(),
        "source": str(loc.get("source", "resolved") or "resolved"),
    }


def _site_coords(
    cfg: AppConfig | None, site: dict[str, Any] | None
) -> tuple[float, float, float, str] | None:
    """(lat, lon, alt_m, name) from the resolved per-draft site, else config."""
    if site is not None:
        return (
            float(site["lat"]),
            float(site["lon"]),
            float(site.get("alt_m", 0.0) or 0.0),
            str(site.get("name", "") or "").strip(),
        )
    if cfg is None:
        return None
    try:
        lat = float(getattr(cfg, "observer_lat", 0.0) or 0.0)
        lon = float(getattr(cfg, "observer_lon", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)) or (lat == 0.0 and lon == 0.0):
        return None
    alt = float(getattr(cfg, "observer_alt_m", 0.0) or 0.0)
    name = str(getattr(cfg, "observer_location_name", "") or "").strip()
    return lat, lon, alt, name


def _vyvar_export_citation_lines(
    cfg: AppConfig | None = None,
    *,
    run_ctx: Any = None,
    photometry_dir: Path | None = None,
    target_row: pd.Series | None = None,
    targets_df: pd.DataFrame | None = None,
    obs_group: str = "",
) -> list[str]:
    """Comment header block for AAVSO / VAR.ASTRO text exports (CITATIONS.bib)."""
    if run_ctx is None:
        run_ctx = _osc_export_citation_context(
            cfg,
            photometry_dir=photometry_dir,
            obs_group=obs_group,
            targets_df=targets_df,
            target_row=target_row,
        )
    return emit_export_citation_lines(run_ctx)


def _safe_filename(name: str) -> str:
    """Sanitize a VSX name for the filesystem. 'BO CVn' -> 'BO_CVn'."""
    s = str(name).strip()
    s = re.sub(r"[^\w\s\-\+]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s or "unknown"


def _bjd_to_datestr_yyyymmdd(bjd_tdb: float) -> str:
    """BJD(TDB) (JD-like float) -> calendar date string YYYYMMDD (UTC-like display)."""
    try:
        t = Time(float(bjd_tdb), format="jd", scale="tdb")
        # For file naming we want just a stable date; use UTC conversion for readability.
        dt = t.utc.datetime
        return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"
    except Exception:  # noqa: BLE001
        # EXC-0080: T3 -- BJD->datestr fail -> 'unknown' token (EXCEPT-BULK-2 2026-07-08)
        return "unknown"


def _fmt_opt_num_na(v: Any, fmt: str, *, na: str = "na") -> str:
    """Format a number via pandas to_numeric; missing -> ``na``. Not the UI empty='-' helper."""
    try:
        f = float(pd.to_numeric(v, errors="coerce"))
    except (TypeError, ValueError):  # noqa: BLE001
        return na
    return (format(f, fmt) if math.isfinite(f) else na)


# Default AAVSO observer placeholder when config is unset (must trigger export/validator warning).


def _filter_lookup_key(name: str) -> str:
    s = str(name or "").strip().upper()
    return re.sub(r"[\s\-]+", "_", s)


def _osc_export_citation_context(
    cfg: AppConfig | None,
    *,
    photometry_dir: Path | None,
    obs_group: str,
    targets_df: pd.DataFrame | None = None,
    target_row: pd.Series | None = None,
    run_ctx: Any = None,
) -> Any:
    """Build citation context with OSC export flags when obs-group is a channel folder."""
    from citations import RunCitationContext, build_run_citation_context
    from gaia_johnson import TRANSFORM_CITATION
    from osc_align import is_osc_export_eligible_obs_group, parse_osc_channel

    if run_ctx is not None and isinstance(run_ctx, RunCitationContext):
        base = run_ctx
    else:
        meta = load_pipeline_meta(photometry_dir)
        targets = targets_df
        if targets is None and target_row is not None:
            targets = pd.DataFrame([target_row])
        base = build_run_citation_context(
            cfg,
            pipeline_meta=meta,
            targets_df=targets,
            obs_group=obs_group,
        )
    ch = parse_osc_channel(obs_group)
    if ch is None:
        return base
    osc_bin = int(getattr(cfg, "osc_channel_binning", 2) or 2) if cfg else 2
    return RunCitationContext(
        **{
            **base.__dict__,
            "osc_channel_export": is_osc_export_eligible_obs_group(obs_group),
            "osc_channel_binning": osc_bin,
            "osc_transform_citation": TRANSFORM_CITATION if is_osc_export_eligible_obs_group(obs_group) else "",
        }
    )


def _guess_setup_info_from_obs_group(obs_group: str) -> tuple[str, str | None, str | None]:
    """Best-effort: parse 'NoFilter_60_2' or 'NoFilter_60_2_G' -> (filter, exptime, binning)."""
    from osc_align import parse_osc_channel_from_setup

    raw = str(obs_group or "").strip().split("|")[0].strip()
    base, _ch = parse_osc_channel_from_setup(raw)
    parts = [p for p in base.split("_") if p]
    if not parts:
        return "", None, None
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), parts[-1], None
    return base or raw, None, None


# Eclipsing types per the VSX documentation
_ECLIPSING_TOKENS = frozenset(
    {
        "E",  # Generic eclipsing
        "EA",  # Algol (Beta Persei)
        "EB",  # Beta Lyrae
        "EW",  # W Ursae Majoris
        "EP",  # Planetary transits
        "E-DO",  # Disk occultation
        "ELL",  # Ellipsoidal (no eclipse but of interest to VAR.ASTRO)
        "EC",  # Contact binaries - ASAS survey type
        "ED",  # Detached eclipsing - ASAS survey type
        "ESD",  # Semi-detached eclipsing - ASAS survey type
    }
)


def _token_is_eclipsing(token: str) -> bool:
    """Check whether a single VSX token (without special characters) is eclipsing."""
    t = str(token or "").strip().rstrip(":").upper()
    if not t:
        return False
    if t in _ECLIPSING_TOKENS:
        return True
    if t.startswith("E-"):
        return True
    return False


def _is_eclipsing(vsx_type: str) -> bool:
    """Return True if the VSX type contains an eclipsing component (VSX conventions)."""
    if not vsx_type or not isinstance(vsx_type, str):
        return False

    vsx_type = vsx_type.strip()
    if not vsx_type:
        return False

    # Pipe | -> OR/uncertainty: only if ALL alternatives are eclipsing
    if "|" in vsx_type:
        alternatives = [a.strip() for a in vsx_type.split("|") if a.strip()]
        if not alternatives:
            return False
        for alt in alternatives:
            main = alt.split("/")[0].split("+")[0].strip().rstrip(":")
            if not _token_is_eclipsing(main):
                return False
        return True

    # Plus + -> AND: if ANY component is eclipsing -> True
    plus_parts = vsx_type.split("+")
    for part in plus_parts:
        part = part.strip()
        if not part:
            continue
        main_token = part.split("/")[0].strip()
        if _token_is_eclipsing(main_token):
            return True

    return False


def _copy_field_image(
    lc_dir: Path,
    catalog_id: str,
    dst_dir: Path,
    fname: str,
    obs_date: str,
) -> str | None:
    """Copy the target field map (JPEG; PNG in older drafts) next to the VarAstro file."""
    from field_map import field_map_path  # noqa: PLC0415

    candidates = [p for p in (field_map_path(lc_dir, catalog_id), field_map_path(lc_dir.parent)) if p is not None]
    for src in candidates:
        try:
            if src.exists():
                dst_name = f"{fname}_{obs_date}_field{src.suffix}"
                dst = Path(dst_dir) / dst_name
                try:
                    shutil.copy2(src, dst)
                    return dst_name
                except Exception as exc:  # noqa: BLE001
                    # EXC-0084: T4 -- field image copy fail already warns (EXCEPT-BULK-2 2026-07-08)
                    logging.warning("[EXPORT] Field image copy failed: %s", exc)
                    return None
        except Exception:  # noqa: BLE001
            # EXC-0085: T2 -- report/export may omit or misstate (logging.warning('[EXPORT] Field image copy failed: ... (EXCEPT-BULK-2 2026-07-08)
            continue
    return None


def _select_export_lc_rows(lc_df: pd.DataFrame, *, differential: bool = False, mag_column: str | None = None) -> pd.DataFrame:
    """Rows suitable for AAVSO/VAR.ASTRO export; the exported magnitude is put in ``mag_calib``
    for the writers.

    Default magnitude: the published one (``mag_calib_ct`` with a colour term, else ``mag_calib``;
    ``photometry_lightcurve.published_mag_column``). ``differential`` (UNCAL-EXPORT-2026-10): the
    target has no standard zero point and ``delta_mag`` is exported."""
    if lc_df is None or lc_df.empty:
        return pd.DataFrame()
    from photometry_lightcurve import published_mag_column  # noqa: PLC0415

    work = lc_df.copy()
    if mag_column:
        mag_col = str(mag_column)
    elif differential:
        mag_col = "delta_mag"
    else:
        mag_col = published_mag_column(work) or "mag_calib"
    if mag_col not in work.columns:
        return pd.DataFrame()
    mag = pd.to_numeric(work[mag_col], errors="coerce")
    # A row needs a time: UTC JD (AAVSO) or BJD_TDB (VarAstro); each writer checks its own.
    _jd = pd.to_numeric(work["jd"], errors="coerce") if "jd" in work.columns else None
    _bj = pd.to_numeric(work["bjd"], errors="coerce") if "bjd" in work.columns else None
    _t_ok = pd.Series(False, index=work.index)
    for _tc in (_jd, _bj):
        if _tc is not None:
            _t_ok = _t_ok | (_tc.notna() & np.isfinite(_tc.to_numpy(dtype=float)))
    finite = mag.notna() & np.isfinite(mag.to_numpy(dtype=float)) & _t_ok
    if "flag" in work.columns:
        from lc_outlier import EXPORT_EXCLUDE_FLAGS  # noqa: PLC0415

        fl = work["flag"].astype(str).str.strip().str.lower()
        # FRAME-REJECT-2026-10: only the physical reasons drop an epoch (saturated, low_snr, no_data,
        # edge_fail, nondetection); frame_qc / artifact / high_err / spike stay, named in NOTES FLAG=.
        bad = fl.isin(list(EXPORT_EXCLUDE_FLAGS))
        mask = finite & ~bad
        n_excl = int((finite & bad).sum())
        if n_excl:
            logging.info(
                "[EXPORT] LC-OUTLIER excluded %d epochs (flags in %s)",
                n_excl,
                sorted(EXPORT_EXCLUDE_FLAGS),
            )
        # FIX-EXPORTALLBAD-2026-10: when every epoch carries an exclusion flag nothing is
        # exported (the old fallback exported all of them, flags ignored).
        out = work.loc[mask].copy()
    else:
        out = work.loc[finite].copy()
    if mag_col != "mag_calib":
        out["mag_calib"] = pd.to_numeric(out[mag_col], errors="coerce")
    return out


def _lc_first_str(lc: pd.DataFrame, col: str) -> str:
    if col not in lc.columns:
        return ""
    for v in lc[col].tolist():
        sv = str(v if v is not None else "").strip()
        if sv and sv.lower() not in ("nan", "none"):
            return sv
    return ""


def _lc_first_float(lc: pd.DataFrame, col: str) -> float:
    if col not in lc.columns:
        return float("nan")
    v = pd.to_numeric(lc[col], errors="coerce").dropna()
    return float(v.iloc[0]) if len(v) else float("nan")


def _summary_str(summary_row: Any, key: str) -> str:
    try:
        v = summary_row.get(key, "")
    except AttributeError:
        return ""
    sv = str(v if v is not None else "").strip()
    return "" if sv.lower() in ("nan", "none") else sv


def _summary_float(summary_row: Any, key: str, lc: pd.DataFrame, lc_col: str) -> float:
    """A per-target value of the summary; the LC column of drafts before OUTPUT-REVIEW step 6."""
    try:
        has = key in summary_row
    except TypeError:
        has = False
    if has:
        v = pd.to_numeric(pd.Series([summary_row.get(key)]), errors="coerce").iloc[0]
        return float(v) if pd.notna(v) else float("nan")
    return _lc_first_float(lc, lc_col)


def _ct_order_from_mode(ct_mode: str) -> int:
    """'band:g:o1' -> 1; anything else -> 0."""
    m = re.match(r"^band:[^:]+:o(\d+)$", str(ct_mode or "").strip())
    return int(m.group(1)) if m else 0


def _load_comp_quality_json(lc_dir: Path, target_cid: str) -> dict[str, Any]:
    p = Path(lc_dir) / f"comp_quality_{target_cid}.json"
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def _check_sidecar_by_file(lc_dir: Path, target_cid: str) -> tuple[str, dict[str, tuple[float, float]]]:
    """(check catalog id, source_file -> (kmag, kmag_err)) from the aperture-photometry sidecar."""
    p = check_kmag_sidecar_path(lc_dir, target_cid)
    if not p.is_file():
        return "", {}
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:  # noqa: BLE001
        return "", {}
    if df.empty:
        return "", {}
    from check_star_kmag import load_check_star  # noqa: PLC0415

    cid = str(load_check_star(lc_dir, target_cid).get("catalog_id", "") or "").strip()
    out: dict[str, tuple[float, float]] = {}
    if "source_file" in df.columns and "kmag" in df.columns:
        km = pd.to_numeric(df["kmag"], errors="coerce").to_numpy(dtype=float)
        ke = (
            pd.to_numeric(df["kmag_err"], errors="coerce").to_numpy(dtype=float)
            if "kmag_err" in df.columns
            else np.full(len(df), np.nan)
        )
        for sf, a, b in zip(df["source_file"].astype(str).str.strip(), km, ke, strict=False):
            if sf:
                out[sf] = (float(a), float(b))
    return cid, out


def _notes_value(v: Any) -> str:
    """NOTES sub-field value: no delimiter / pipe / equals characters."""
    return re.sub(r"[,|=\s]+", "_", str(v)).strip("_")


#: subfolder of aavso/ and varastro/ for files that must not be submitted (TRUST RED)
DONTSUBMIT_DIRNAME = "_dontsubmit"


def export_lightcurve_reports(
    output_dir: Path,
    target_row: pd.Series,
    lc_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    summary_row: pd.Series,
    *,
    observer_code: str = "",
    observer_name: str = "Unknown Observer",
    comp_quality_map: dict[str, str] | None = None,
    software_version: str | None = None,
    cfg: AppConfig | None = None,
    lc_dir: Path | None = None,
    obs_group: str = "",
    targets_df: pd.DataFrame | None = None,
    run_citation_ctx: Any = None,
    export_method: str = "aperture",
    export_failures: list[ExportFailure] | None = None,
    export_stats: dict[str, int] | None = None,
) -> dict[str, Path]:
    """AAVSO Extended Format + VarAstro files for one target (EXPORT-2026-10).

    AAVSO (Extended File Format spec): ``#DATE=JD`` with the UTC Julian date at
    mid-exposure (BJD is not an accepted AAVSO date type), FILTER from the single band
    authority (:func:`gaia_bands.filter_band_for_obs_group`), TRANS=YES only when a colour
    term of order >= 1 was applied in a standard band, ensemble with CNAME=ENSEMBLE,
    CMAG=na, KNAME/KMAG = check star and its measured magnitude, CHART = 'Gaia DR3'
    (comparison magnitudes are not from an AAVSO chart), NOTES on every row as
    ``KEY=value`` sub-fields separated by ``|``. Run description and citations are
    ``#`` comment lines after the required parameters.

    VarAstro: plain text, column 1 BJD_TDB, column 2 magnitude, column 3 error, with a
    ``#`` header (time, method, zero point, comparison stars, check star, trust - as in the PDF
    report). Written for eclipsing types. A differential target (no zero point) is written to
    VarAstro only; a RED-trust target into ``_dontsubmit/`` (EXPORT-AUDIT-2026-10).

    A target whose magnitudes are not on the band of the filter (``band_system`` other
    than the zero-point band - no measured comparison star with a valid band magnitude) is
    not exported; the reason is recorded.
    """
    from gaia_bands import filter_band_for_obs_group, zp_mag_from_gaia  # noqa: PLC0415

    fresh_cfg = cfg or AppConfig()
    if not str(observer_code).strip():
        observer_code = str(getattr(fresh_cfg, "observer_code", "") or "")
    if observer_name == "Unknown Observer":
        _on = str(getattr(fresh_cfg, "observer_name", "") or "").strip()
        if _on:
            observer_name = _on

    out_base = Path(output_dir)
    (out_base / "aavso").mkdir(parents=True, exist_ok=True)
    (out_base / "varastro").mkdir(parents=True, exist_ok=True)
    _phot_dir = out_base.parent
    _lc_dir = Path(lc_dir) if lc_dir is not None else (_phot_dir / "lightcurves")
    _sw_ver = str(software_version or VYVAR_SOFTWARE_VERSION).strip() or VYVAR_SOFTWARE_VERSION
    _resolved_site = _resolved_site_from_meta(out_base)

    # STAR-NAME-2026-10: VSX name, else exoplanet host (TOI-xxxx ...), else Gaia DR3 id
    from star_page_m3 import target_display_name  # noqa: PLC0415

    vsx_name = target_display_name(target_row)
    safe = _safe_filename(vsx_name)
    _export_method = str(export_method or "aperture").strip().lower()
    if _export_method != "aperture":
        raise InvariantViolation(
            INV_PSF_SUBMIT_01,
            "AAVSO/VarAstro submission writers refuse lc_method="
            f"{_export_method}; science exports are aperture-only",
        )
    target_cid = str(normalize_gaia_source_id(target_row.get("catalog_id", "")) or "").strip()
    _target_id = target_cid or vsx_name

    # UNCAL-EXPORT-2026-10 (Milan 2026-10-11): a target without a standard zero point
    # (band_system NONE - no comparison star with a valid band magnitude) is exported as
    # differential magnitudes for VarAstro only (EXPORT-AUDIT-2026-10: no AAVSO file).
    # GSPC-STD-2026-10 (Milan 2026-10-11, option a): when the GSPC zero point exists for such a
    # target (>= gspc_zp.MIN_COMPS accepted comps, mag_gspc in the LC) it is exported as STD on
    # the GSPC scale (NOTES ZP=GaiaDR3-GSPC); without it (e.g. archive unreachable) as differential.
    _cq0 = _load_comp_quality_json(_lc_dir, target_cid)
    _bs0 = str(summary_row.get("band_system", "") or _cq0.get("band_system", "") or "").strip()
    _gz0 = _cq0.get("gspc_zero_point") if isinstance(_cq0.get("gspc_zero_point"), dict) else {}
    from gspc_zp import gspc_standard_available  # noqa: PLC0415

    gspc_std = bool(
        _bs0.upper() == "NONE" and lc_df is not None and "mag_gspc" in lc_df.columns
        and gspc_standard_available(_gz0, pd.to_numeric(lc_df["mag_gspc"], errors="coerce").to_numpy(dtype=float))
    )
    differential = _bs0.upper() == "NONE" and not gspc_std
    lc_normal = _select_export_lc_rows(lc_df.copy() if lc_df is not None else pd.DataFrame(),
                                       differential=differential, mag_column="mag_gspc" if gspc_std else None)
    if lc_normal.empty:
        record_export_failure(export_failures, _target_id, _export_method, "no exportable LC points")
        return {}
    # TRUST-COVERAGE-2026-10: too few usable epochs -> not exported (same rule as trust RED)
    from trust_flag_core import coverage_insufficient  # noqa: PLC0415

    _n_lc_all = int(len(lc_df)) if lc_df is not None else 0
    if coverage_insufficient(len(lc_normal), _n_lc_all):
        record_export_failure(export_failures, _target_id, _export_method,
                              f"insufficient data: {len(lc_normal)} of {_n_lc_all} epochs usable")
        return {}
    _maybe_set_airmass_citation(run_citation_ctx, lc_normal)

    # --- observation group / band authority --------------------------------------------
    obs_group_resolved = str(obs_group or "").strip()
    if not obs_group_resolved:
        try:
            _sd = out_base.parent.parent
            if _sd.name and _sd.name.lower() not in ("platesolve", "photometry"):
                obs_group_resolved = _sd.name
        except Exception:  # noqa: BLE001
            obs_group_resolved = ""
    from invariants_runtime import check_osc03_export_eligibility  # noqa: PLC0415
    from osc_align import is_onerggb_internal_obs_group  # noqa: PLC0415

    if is_onerggb_internal_obs_group(obs_group_resolved):
        record_export_failure(export_failures, _target_id, _export_method, "oneRGGB internal-only (not exported)")
        return {}
    fb = filter_band_for_obs_group(obs_group_resolved)
    aavso_filter = fb.aavso_code
    _umap = getattr(fresh_cfg, "aavso_filter_map", None) or {}
    if fb.filter_token and _umap:
        for uk, uv in _umap.items():
            if _filter_lookup_key(str(uk)) == _filter_lookup_key(fb.filter_token) and str(uv).strip():
                aavso_filter = str(uv).strip()
    check_osc03_export_eligibility(
        obs_group_resolved, aavso_filter, meta={"target": _target_id, "method": _export_method}
    )
    setup_filter_raw, exptime_s, binning = _guess_setup_info_from_obs_group(obs_group_resolved)

    cq = _load_comp_quality_json(_lc_dir, target_cid)
    # GSPC-ZP-2026-10: calibration note B (red comps / GSPC zero point differs) in the header
    # comments and as an AAVSO NOTES sub-field. Note A targets are not exported (band_system).
    from gaia_bands import zero_point_note_text  # noqa: PLC0415

    _gz_exp = cq.get("gspc_zero_point") if isinstance(cq.get("gspc_zero_point"), dict) else {}
    _zpn = str(_gz_exp.get("note") or "")
    _zp_note_txt = zero_point_note_text(_zpn, fb.zp_band, _gz_exp) if _zpn in ("B", "B-heuristic") else ""
    _zp_note_short = (zero_point_note_text(_zpn, fb.zp_band, _gz_exp, short=True)
                      if _zpn in ("B", "B-heuristic") else "")
    band_system = str(summary_row.get("band_system", "") or cq.get("band_system", "") or "").strip()
    on_band = bool(band_system) and band_system == fb.zp_band
    # EXPORT-AUDIT-2026-10 (Milan 2026-10-21): a differential target (no zero point) goes to VarAstro
    # only - AAVSO MTYPE=DIF needs one named comparison star (CNAME/CMAG), not an ensemble, and DIF
    # data are neither validated nor plotted there.
    aavso_ok = (on_band or gspc_std) and bool(aavso_filter)
    if differential:
        _accumulate_export_stat(export_stats, "aavso_skipped_differential")
    if (on_band or gspc_std) and not aavso_filter:
        record_export_failure(
            export_failures, _target_id, _export_method, f"filter '{fb.filter_token}' has no AAVSO code - AAVSO file not written"
        )
    if gspc_std:
        _zp_note_txt = zero_point_note_text("A", fb.zp_band, _gz_exp)
        _zp_note_short = ""
        _accumulate_export_stat(export_stats, "gspc_std")
    elif differential:
        # the note must not claim the GSPC standard export when the export is differential
        _gz_dif = dict(_gz_exp, C=None)
        _zp_note_txt = zero_point_note_text("A", fb.zp_band, _gz_dif)
        _zp_note_short = zero_point_note_text("A", fb.zp_band, _gz_dif, short=True)
        _accumulate_export_stat(export_stats, "differential")
    elif not on_band:
        record_export_failure(
            export_failures,
            _target_id,
            _export_method,
            f"magnitudes not on the {fb.label} scale (band_system={band_system or 'unknown'}) - not exported",
        )
        _accumulate_export_stat(export_stats, "band_refused")
        if not (fb.zp_band == "G" and band_system == "G"):
            return {}

    # --- per-target facts ------------------------------------------------------------------
    # colour term of the target: photometry_summary.csv (OUTPUT-REVIEW step 6); the LC of older
    # drafts repeats it on every row (ct_correction there = ct_corr of the summary)
    ct_mode = _summary_str(summary_row, "ct_mode") or _lc_first_str(lc_normal, "ct_mode")
    ct_order = _ct_order_from_mode(ct_mode)
    ct_ok = (_summary_str(summary_row, "ct_ok") or _lc_first_str(lc_normal, "ct_ok")).lower() in ("true", "1", "yes")
    ct_c1 = _summary_float(summary_row, "ct_c1", lc_normal, "ct_c1")
    ct_corr = _summary_float(summary_row, "ct_corr", lc_normal, "ct_correction")
    trans_yes = bool(fb.transformable and ct_ok and ct_order >= 1) and not differential and not gspc_std

    weights = cq.get("broeg_weights") if isinstance(cq.get("broeg_weights"), dict) else {}
    band_mags = cq.get("comp_band_mag") if isinstance(cq.get("comp_band_mag"), dict) else {}
    comp_ids: list[str] = []
    if comp_df is not None and not comp_df.empty and "catalog_id" in comp_df.columns:
        for v in comp_df["catalog_id"].tolist():
            ck = str(normalize_gaia_source_id(v) or "").strip()
            if ck and ck not in comp_ids:
                if comp_quality_map and str(comp_quality_map.get(ck, "")).strip().lower() == "excluded":
                    continue
                comp_ids.append(ck)
    if weights:
        comp_ids = [c for c in comp_ids if c in weights] or comp_ids
    n_comp = len(comp_ids)

    check_cid, kmag_by_file = _check_sidecar_by_file(_lc_dir, target_cid)
    k_ref = float("nan")
    if check_cid:
        _kr = band_mags.get(check_cid)
        if _kr is not None:
            k_ref = float(_kr)
        elif comp_df is not None and "catalog_id" in comp_df.columns:
            _hit = comp_df[comp_df["catalog_id"].map(lambda x: str(normalize_gaia_source_id(x) or "")) == check_cid]
            if not _hit.empty:
                k_ref = zp_mag_from_gaia(
                    float(pd.to_numeric(_hit.iloc[0].get("mag"), errors="coerce")),
                    float(pd.to_numeric(_hit.iloc[0].get("bp_rp"), errors="coerce")),
                    fb,
                )

    # check star on the GSPC scale: relative KMAG + the same constant shift as the target
    _gspc_shift = float("nan")
    # REPORT-AUDIT-2026-10: why a GSPC check star has no catalogue magnitude (KREFMAG absent)
    _k_ref_why = ""
    if gspc_std:
        _sh = (pd.to_numeric(lc_normal.get("mag_gspc"), errors="coerce")
               - pd.to_numeric(lc_normal.get("delta_mag"), errors="coerce")).dropna()
        _gspc_shift = float(_sh.median()) if len(_sh) else float("nan")
        # catalogue magnitude of the check star on the same (GSPC) scale, when GSPC accepts it
        k_ref = float("nan")
        if check_cid:
            _k_ref_why = "no Gaia XP synthetic photometry"
            try:
                from gspc_zp import CACHE_NAME, gspc_mag, read_cache  # noqa: PLC0415

                _kr_row = read_cache(_phot_dir / CACHE_NAME).get(check_cid)
                if _kr_row is not None:
                    _km, _ks, _why = gspc_mag(_kr_row, fb.zp_band)
                    if not _why:
                        k_ref, _k_ref_why = float(_km), ""
                    else:
                        _k_ref_why = str(_why).replace(",", ";").replace("|", "/")
            except Exception:  # noqa: BLE001
                k_ref = float("nan")

    _trust = str(summary_row.get("trust", "") or "").strip().upper()
    # GS11-2026-10: blend light of Gaia neighbours removed from the magnitudes -> say so per epoch
    from dilution import gs11_dilution_from_summary  # noqa: PLC0415

    _gs11 = gs11_dilution_from_summary(summary_row, cfg)
    _trust_reason = str(summary_row.get("trust_reason", "") or "").strip()
    # TRUST-ALWAYS-2026-10: fail closed - a target without a trust evaluation is not for submission
    if _trust not in ("GREEN", "YELLOW", "RED"):
        _trust, _trust_reason = "RED", "trust not evaluated"

    jd0 = _lc_first_float(lc_normal, "jd")
    bjd0 = _lc_first_float(lc_normal, "bjd")
    _t_tag = jd0 if math.isfinite(jd0) else bjd0
    date_tag = _bjd_to_datestr_yyyymmdd(float(_t_tag)) if math.isfinite(_t_tag) else "unknown"
    aavso_path = aavso_export_path(out_base, safe, date_tag)
    var_path = varastro_export_path(out_base, safe, date_tag)
    # EXPORT-AUDIT-2026-10 (Milan 2026-10-21): a RED-trust target is written for inspection into
    # <aavso|varastro>/_dontsubmit/, not next to the files to submit
    if _trust == "RED":
        aavso_path = aavso_path.parent / DONTSUBMIT_DIRNAME / aavso_path.name
        var_path = var_path.parent / DONTSUBMIT_DIRNAME / var_path.name
        _accumulate_export_stat(export_stats, "dontsubmit_red")

    _cite_kw: dict[str, Any] = {
        "photometry_dir": _phot_dir,
        "target_row": target_row,
        "obs_group": obs_group_resolved,
    }
    if run_citation_ctx is not None:
        _cite_kw["run_ctx"] = run_citation_ctx
    elif targets_df is not None:
        _cite_kw["targets_df"] = targets_df
    methods_block = _vyvar_export_citation_lines(fresh_cfg, **_cite_kw)

    _sc = _site_coords(fresh_cfg, _resolved_site)
    site = (
        f"{_sc[3] or 'observer site'} {_sc[0]:.4f}N {_sc[1]:.4f}E {_sc[2]:.0f} m" if _sc is not None else ""
    )

    # Aperture actually used: r = k x FWHM(frame), per epoch in the LC (aperture_r_px).
    _r_ap = (
        pd.to_numeric(lc_normal["aperture_r_px"], errors="coerce").dropna()
        if "aperture_r_px" in lc_normal.columns
        else pd.Series(dtype=float)
    )
    # k actually used for this target: aperture_per_target.json (f_star); the LC column
    # aperture_f of drafts before EXPORT-2026-10 holds the group default instead.
    _k_ap = float("nan")
    _apt: dict[str, Any] = {}
    try:
        _apt = json.loads((_phot_dir / "aperture_per_target.json").read_text(encoding="utf-8"))
        _ent = (_apt.get("targets") or {}).get(target_cid) or {}
        _k_ap = float(_ent.get("f_star", float("nan")))
    except (OSError, ValueError, TypeError, AttributeError):
        _k_ap = float("nan")
    if not math.isfinite(_k_ap):
        _k_ap = _lc_first_float(lc_normal, "aperture_f")
    _scale = _resolve_export_arcsec_per_px(_phot_dir, fresh_cfg)
    ap_line = ""
    if len(_r_ap):
        _rm = float(_r_ap.median())
        _as = (
            f" = {_rm * float(_scale):.2f} arcsec"
            if _scale is not None and math.isfinite(float(_scale)) and float(_scale) > 0
            else ""
        )
        _a_in, _a_out = (
            float(getattr(fresh_cfg, "annulus_inner_fwhm", 2.7)),
            float(getattr(fresh_cfg, "annulus_outer_fwhm", 5.2)),
        )
        ap_line = (
            f"# Aperture: radius r = k x FWHM(frame), k = {_k_ap:.2f}; used r median {_rm:.2f} px{_as}"
            f" (range {float(_r_ap.min()):.2f}-{float(_r_ap.max()):.2f} px; per epoch: AAVSO NOTES APER);"
            f" sky annulus {_a_in:.1f}-{_a_out:.1f} FWHM\n"
        )

    # --- facts of the run for the VarAstro header (EXPORT-AUDIT-2026-10: as in the PDF report) ------
    _flag_ser = (lc_normal["flag"].fillna("").astype(str).str.strip().str.lower()
                 if "flag" in lc_normal.columns else pd.Series("", index=lc_normal.index))
    _flag_counts = {k: int(v) for k, v in _flag_ser[~_flag_ser.isin(("", "normal"))].value_counts().items()}
    _kser = pd.Series([kmag_by_file.get(str(sf).strip(), (float("nan"), float("nan")))[0]
                       for sf in lc_normal.get("source_file", pd.Series(dtype=str)).astype(str)], dtype=float)
    _keser = pd.Series([kmag_by_file.get(str(sf).strip(), (float("nan"), float("nan")))[1]
                        for sf in lc_normal.get("source_file", pd.Series(dtype=str)).astype(str)], dtype=float)
    if gspc_std:
        _kser = _kser + _gspc_shift
    _comp_by_id: dict[str, Any] = {}
    if comp_df is not None and not comp_df.empty and "catalog_id" in comp_df.columns:
        for _, _cr in comp_df.iterrows():
            _comp_by_id.setdefault(str(normalize_gaia_source_id(_cr.get("catalog_id")) or "").strip(), _cr)
    _zp = cq.get("zero_point") if isinstance(cq.get("zero_point"), dict) else {}
    _fwhm_night = float("nan")
    try:
        _fwhm_night = float(_apt.get("fwhm_night_px", float("nan")))
    except (AttributeError, TypeError, ValueError):
        _fwhm_night = float("nan")

    def _detail_lines() -> list[str]:
        d: list[str] = []
        _jd = pd.to_numeric(lc_normal.get("jd", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(_jd):
            t0, t1 = Time(float(_jd.min()), format="jd", scale="utc"), Time(float(_jd.max()), format="jd", scale="utc")
            d.append(f"# Series: {t0.iso[:19]} - {t1.iso[:19]} UTC ({24.0 * (float(_jd.max()) - float(_jd.min())):.2f} h); "
                     f"{len(lc_df) if lc_df is not None else 0} epochs measured, {len(lc_normal)} exported\n")
        _am = pd.to_numeric(lc_normal.get("airmass", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(_am):
            d.append(f"# Airmass: {float(_am.min()):.3f} - {float(_am.max()):.3f} (at the target, Kasten & Young 1989)"
                     + (f"; FWHM night {_fwhm_night:.2f} px" if math.isfinite(_fwhm_night) else "") + "\n")
        if _flag_counts:
            d.append("# Flagged epochs kept in the export (statistical flags only, no sigma clipping): "
                     + ", ".join(f"{k} {v}" for k, v in sorted(_flag_counts.items())) + "\n")
        if _zp:
            _sd = _zp.get("sigma_zp_data_mag")
            d.append(f"# Zero point: {_zp.get('band', fb.zp_band)} from {_zp.get('n_comp', n_comp)} comparison stars "
                     f"(effective {_fmt_opt_num_na(_zp.get('n_eff'), '.1f')}); sigma_ZP model "
                     f"{_fmt_opt_num_na(_zp.get('sigma_zp_model_mag'), '.4f')} mag"
                     + (f", from their mutual scatter {_fmt_opt_num_na(_sd, '.4f')} mag" if _sd is not None else "") + "\n")
        if comp_ids:
            d.append(f"# Comparison stars: Gaia DR3 id | G | BP-RP | {fb.zp_band} | distance arcmin | rms mmag | weight\n")
            for c in comp_ids:
                _cr = _comp_by_id.get(c)
                _g = _bp = _dist = _rms = float("nan")
                if _cr is not None:
                    _g = float(pd.to_numeric(_cr.get("mag"), errors="coerce"))
                    _bp = float(pd.to_numeric(_cr.get("bp_rp"), errors="coerce"))
                    _dist = 60.0 * float(pd.to_numeric(_cr.get("_dist_deg"), errors="coerce"))
                    _rms = 1085.7362 * float(pd.to_numeric(_cr.get("comp_rms"), errors="coerce"))
                d.append(f"#   {c} | {_fmt_opt_num_na(_g, '.3f')} | {_fmt_opt_num_na(_bp, '.3f')} | "
                         f"{_fmt_opt_num_na(band_mags.get(c), '.3f')} | {_fmt_opt_num_na(_dist, '.1f')} | "
                         f"{_fmt_opt_num_na(_rms, '.1f')} | {_fmt_opt_num_na(weights.get(c), '.3f')}\n")
        if check_cid:
            _km = _kser.dropna()
            _kmed = float(_km.median()) if len(_km) else float("nan")
            _ksc = float(_km.std(ddof=1)) if len(_km) > 1 else float("nan")
            _kem = float(_keser.dropna().median()) if _keser.notna().any() else float("nan")
            d.append(f"# Check star Gaia DR3 {check_cid}: catalogue {_fmt_opt_num_na(k_ref, '.3f')}, measured median "
                     f"{_fmt_opt_num_na(_kmed, '.3f')}"
                     + (f" (difference {_kmed - k_ref:+.3f})" if math.isfinite(_kmed) and math.isfinite(k_ref) else "")
                     + f", scatter {_fmt_opt_num_na(_ksc, '.4f')} mag, median error {_fmt_opt_num_na(_kem, '.4f')} mag, "
                     f"{len(_km)} epochs\n")
        return d

    def _desc_lines(time_line: str, *, ensemble_list: bool = True) -> list[str]:
        d: list[str] = []
        d.append("# --- VYVAR run description (comment lines) ---\n")
        if site:
            d.append(f"# Site: {site}\n")
        d.append(
            f"# Setup: {obs_group_resolved or 'na'} | Filter: {fb.filter_token or 'na'} | "
            f"Exp: {exptime_s or 'na'} s | Bin: {binning or 'na'}\n"
        )
        d.append(time_line)
        if differential:
            d.append(f"# Magnitudes: DIFFERENTIAL, not calibrated (delta_mag = target - ensemble, {fb.filter_token or 'na'} filter)\n")
        elif gspc_std:
            d.append(f"# Magnitudes: {fb.label}; zero point from Gaia DR3 GSPC {fb.zp_band} magnitudes of the comparison stars\n")
        else:
            d.append(f"# Magnitudes: {fb.label}; comparison magnitudes Gaia DR3 G, BP-RP -> {fb.zp_band} (Gaia DR3 doc 5.5.1)\n")
        if _zp_note_txt:
            d.append(f"# {_zp_note_txt}\n")
        if differential or gspc_std:
            d.append("# Colour term: none -> TRANS=NO\n")
        elif trans_yes:
            d.append(
                f"# Colour term: polynomial of order {ct_order} in Gaia BP-RP, fitted on the comparison stars of the "
                f"field against their Gaia DR3 {fb.zp_band} magnitudes (Gaia DR3 doc 5.5.1), applied as a constant "
                f"(correction {ct_corr:+.4f} mag for this target; c1={ct_c1:+.4f}) -> TRANS=YES\n"
            )
        elif fb.transformable:
            d.append("# Colour term: not significant / not applied for this target -> TRANS=NO\n")
        else:
            d.append("# Colour term: none (filter reported untransformed) -> TRANS=NO\n")
        if ap_line:
            d.append(ap_line)
        d.append("# Error: 1-sigma total (CCD equation + ensemble + scintillation + systematic floor)\n")
        d.append(f"# Ensemble: {n_comp} comparison stars, Broeg (2005) weights" + (":\n" if ensemble_list else "\n"))
        for c in (comp_ids if ensemble_list else []):
            _bm = band_mags.get(c)
            _w = weights.get(c)
            d.append(
                f"#   {c}  {fb.zp_band}={_fmt_opt_num_na(_bm, '.3f')}  w={_fmt_opt_num_na(_w, '.3f')}\n"
            )
        if check_cid:
            d.append(f"# Check star: {check_cid}  {fb.zp_band}={_fmt_opt_num_na(k_ref, '.3f')} (catalogue)\n")
        if _trust:
            d.append(f"# Trust: {_trust} - {_trust_reason}\n")
        return d

    paths: dict[str, Path] = {}

    # --- AAVSO Extended ----------------------------------------------------------------------
    if aavso_ok:
        a_lines: list[str] = ["#TYPE=Extended\n"]
        _obc = str(observer_code).strip()
        if not _obc:
            logging.warning("[EXPORT] observer code not set - set observer_code in config")
        a_lines.append(f"#OBSCODE={_obc}\n")
        a_lines.append(_aavso_software_header_line(_sw_ver))
        a_lines.append("#DELIM=,\n")
        a_lines.append("#DATE=JD\n")
        a_lines.append("#OBSTYPE=CCD\n")
        a_lines.extend(_desc_lines("# Time: JD (UTC) at mid-exposure (DATE-OBS + EXPTIME/2)\n"))
        a_lines.extend(methods_block)
        a_lines.append("#STARID,DATE,MAG,MAGERR,FILTER,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AIRMASS,GROUP,CHART,NOTES\n")
        starid = vsx_name.replace(",", " ")[:30]
        # NOTES free text (AAVSO: describe non-AAVSO comparison stars in NOTES; no ',' or '|')
        _comp_txt = "; ".join(
            f"{c} {fb.zp_band}{_fmt_opt_num_na(band_mags.get(c), '.3f')} w{_fmt_opt_num_na(weights.get(c), '.3f')}"
            for c in comp_ids
        )
        _notes_free = (f"Ensemble of {n_comp} Gaia DR3 comparison stars: {_comp_txt}"
                       + (f"; check star Gaia DR3 {check_cid}" if check_cid else "")
                       + (f" without GSPC catalogue magnitude ({_k_ref_why})" if check_cid and _k_ref_why else ""))
        _n_rows = 0
        for _, row in lc_normal.iterrows():
            jd = float(pd.to_numeric(row.get("jd", float("nan")), errors="coerce"))
            mag = float(pd.to_numeric(row.get("mag_calib", float("nan")), errors="coerce"))
            err = float(pd.to_numeric(row.get("err", float("nan")), errors="coerce"))
            if not (math.isfinite(jd) and math.isfinite(mag) and math.isfinite(err) and err > 0):
                continue
            am = float(pd.to_numeric(row.get("airmass", float("nan")), errors="coerce"))
            sf = str(row.get("source_file", "") or "").strip()
            km, _ke = kmag_by_file.get(sf, (float("nan"), float("nan")))
            if gspc_std:
                km = km + _gspc_shift
            notes = [_notes_free]
            _mi = float(pd.to_numeric(row.get("mag_inst", float("nan")), errors="coerce"))
            if math.isfinite(_mi):
                notes.append(f"VMAGINS={_mi:.4f}")
            if math.isfinite(k_ref):
                notes.append(f"KREFMAG={k_ref:.3f}")
            _rap = float(pd.to_numeric(row.get("aperture_r_px", float("nan")), errors="coerce"))
            if math.isfinite(_rap):
                notes.append(f"APER={_rap:.2f}px")
            notes.append(f"NCOMP={n_comp}")
            notes.append("ZP=GaiaDR3-GSPC" if gspc_std else f"ZP=GaiaDR3-{_notes_value(fb.zp_band)}")
            if gspc_std:
                _sz = _gz_exp.get("sigma_data") if _gz_exp.get("sigma_data") is not None else _gz_exp.get("sigma_model")
                if _sz is not None:
                    notes.append(f"ZPERR={float(_sz):.3f}")
            if _zp_note_short:
                notes.append("ZPNOTE=" + _notes_value(_zp_note_short.split("=", 1)[1]))
            if trans_yes:
                notes.append(f"TRANSMETH=colour_term_order{ct_order}_in_BP-RP_fitted_on_Gaia_DR3_comparison_stars")
                notes.append(f"CT={ct_corr:+.4f}")
            # EXPORT-AUDIT-2026-10: statistical flags are kept (no sigma clipping) and named per epoch
            _fl = str(row.get("flag", "") or "").strip().lower()
            if _fl and _fl not in ("normal", "nan"):
                notes.append(f"FLAG={_notes_value(_fl)}")
            if _trust:
                notes.append(f"TRUST={_notes_value(_trust)}")
            if _gs11 is not None:
                notes.append(f"GS11D={_gs11[0]:.3f}")
            a_lines.append(
                ",".join(
                    [
                        starid,
                        f"{jd:.6f}",
                        f"{mag:.3f}",
                        f"{min(1.0857362 * err, 9.999):.3f}",
                        aavso_filter,
                        "YES" if trans_yes else "NO",
                        "STD",
                        "ENSEMBLE",
                        "na",
                        check_cid[:20] if check_cid and math.isfinite(km) else "na",
                        f"{km:.3f}" if check_cid and math.isfinite(km) else "na",
                        f"{am:.4f}" if math.isfinite(am) else "na",
                        "na",
                        "Gaia DR3",
                        "|".join(notes) if notes else "na",
                    ]
                )
                + "\n"
            )
            _n_rows += 1
        if _n_rows == 0:
            record_export_failure(export_failures, _target_id, _export_method, "AAVSO: no rows with JD/mag/err")
        else:
            try:
                aavso_path.parent.mkdir(parents=True, exist_ok=True)
                aavso_path.write_text("".join(a_lines), encoding="ascii", errors="replace")
                logging.info("[EXPORT] AAVSO: %s (%d rows)", aavso_path.name, _n_rows)
                paths["aavso"] = aavso_path
            except OSError as exc:
                record_export_failure(export_failures, _target_id, _export_method, f"AAVSO write error: {exc}")

    # --- VarAstro -------------------------------------------------------------------------------
    vsx_type = str(target_row.get("vsx_type", "") or "").strip()
    if not _is_eclipsing(vsx_type):
        logging.info("[EXPORT] Skip varastro %s - not an eclipsing type (%s)", vsx_name, vsx_type)
        return paths
    from photometry_core import TIME_BASE_BJD_TDB, resolve_lc_time_base  # noqa: PLC0415

    try:
        _tb = resolve_lc_time_base(lc_normal)
    except ValueError:
        _tb = None
    if _tb != TIME_BASE_BJD_TDB:
        record_export_failure(
            export_failures, _target_id, _export_method, f"VarAstro: time_base={_tb} (BJD_TDB required)"
        )
        _accumulate_export_stat(export_stats, "time_base_refused")
        return paths
    v_lines: list[str] = [
        f"# {_sw_ver} - differential ensemble aperture photometry | Observer: {observer_name}"
        + (f" ({str(observer_code).strip()})" if str(observer_code).strip() else "")
        + "\n",
        f"# Star: {vsx_name} | Type: {vsx_type or 'na'} | Gaia DR3 {target_cid or 'na'}"
        f" | G={_fmt_opt_num_na(target_row.get('mag'), '.3f')}"
        f" BP-RP={_fmt_opt_num_na(target_row.get('bp_rp'), '.3f')}\n",
    ]
    v_lines.extend(
        _desc_lines("# Time: BJD_TDB at mid-exposure (select BJD at upload; Eastman et al. 2010)\n",
                    ensemble_list=False)
    )
    v_lines.extend(_detail_lines())
    v_lines.extend(emit_varastro_method_summary_lines(run_citation_ctx) if run_citation_ctx is not None else [])
    v_lines.extend(methods_block)
    # FIX-VARASTRO-MAP-2026-10: VarAstro needs the field chart (variable + comps) next to the
    # data file; the copy was lost in the EXPORT-2026-10 rewrite (08d6a98).
    try:
        var_path.parent.mkdir(parents=True, exist_ok=True)
        field_img = _copy_field_image(_lc_dir, str(target_cid or ""), var_path.parent, str(safe), str(date_tag))
    except Exception:  # noqa: BLE001
        field_img = None
    if field_img is None:
        record_export_failure(export_failures, _target_id, _export_method, "VarAstro: field map not found")
    v_lines.append(f"# FIELD IMAGE: {field_img or 'not available'}\n")
    v_lines.append("# BJD_TDB          dif_mag  err\n" if differential
                   else "# BJD_TDB          mag      err\n")
    _nv = 0
    for _, row in lc_normal.iterrows():
        bjd = float(pd.to_numeric(row.get("bjd", float("nan")), errors="coerce"))
        mag = float(pd.to_numeric(row.get("mag_calib", float("nan")), errors="coerce"))
        err = float(pd.to_numeric(row.get("err", float("nan")), errors="coerce"))
        if not (math.isfinite(bjd) and math.isfinite(mag) and math.isfinite(err)):
            continue
        v_lines.append(f"{bjd:.6f}   {mag:.4f}   {1.0857362 * err:.4f}\n")
        _nv += 1
    if _nv:
        try:
            var_path.parent.mkdir(parents=True, exist_ok=True)
            var_path.write_text("".join(v_lines), encoding="ascii", errors="replace")
            logging.info("[EXPORT] VAR.ASTRO: %s (%d rows)", var_path.name, _nv)
            paths["varastro"] = var_path
        except OSError as exc:
            record_export_failure(export_failures, _target_id, _export_method, f"VarAstro write error: {exc}")
    return paths


def export_all_method_lightcurve_reports(
    output_dir: Path,
    target_row: pd.Series,
    *,
    lc_dir: Path,
    target_cid: str,
    comp_df: pd.DataFrame,
    summary_row: pd.Series,
    cfg: AppConfig | None = None,
    export_failures: list[ExportFailure] | None = None,
    export_stats: dict[str, int] | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Path]]:
    """Export AAVSO + VarAstro for the aperture submission product only.

    Internal PSF LC files (``lightcurve_*_psf.csv``) are never routed
    through these writers (INV-PSF-SUBMIT-01). The glob remains as a presence
    signal for reports; submission methods stay aperture-only.
    """
    fresh_cfg = cfg or AppConfig()
    _lc_dir = Path(lc_dir)
    _tid = str(target_cid or target_row.get("catalog_id", "") or "").strip()
    _have_psf_files = any(_lc_dir.glob("lightcurve_*_psf.csv"))
    if _have_psf_files:
        logging.info(
            "[EXPORT] internal PSF LC files present; "
            "INV-PSF-SUBMIT-01 keeps AAVSO/VarAstro aperture-only"
        )
    _methods = ["aperture"]
    out: dict[str, dict[str, Path]] = {}
    for method in _methods:
        lc_path = lc_csv_path(_lc_dir, target_cid, method)
        if not lc_path.is_file():
            # F-435-EXPORT-GHOSTS: active_targets may lack an LC (no comps / dropped).
            # Skip with INFO - do not record as export failure.
            logging.info(
                "[EXPORT] skip %s %s: no LC CSV (not a photometry product)",
                _tid or target_cid,
                method,
            )
            continue
        try:
            lc_df = pd.read_csv(lc_path, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            record_export_failure(
                export_failures,
                _tid,
                method,
                f"LC CSV read error: {exc}",
            )
            continue
        if lc_df.empty:
            record_export_failure(export_failures, _tid, method, "LC CSV empty")
            continue
        try:
            paths = export_lightcurve_reports(
                output_dir,
                target_row,
                lc_df,
                comp_df,
                summary_row,
                cfg=fresh_cfg,
                lc_dir=_lc_dir,
                export_method=method,
                export_failures=export_failures,
                export_stats=export_stats,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            record_export_failure(
                export_failures,
                _tid,
                method,
                f"export error: {exc}",
            )
            continue
        if paths:
            out[method] = paths
    return out


def export_group_reports(
    photometry_dir: Path | str,
    *,
    obs_group: str,
    cfg: AppConfig | None = None,
    at_df: pd.DataFrame | None = None,
    summary_df: pd.DataFrame | None = None,
    comp_index: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Write AAVSO / VarAstro files for every light curve of one observation group.

    Callable after aperture photometry and on its own (re-export of an existing draft without
    re-running photometry). Inputs default to the files in ``photometry_dir``.
    Returns {"n_ok", "n_skip", "failures", "stats"}.
    """
    from report_methods import lc_csv_path  # noqa: PLC0415

    phot = Path(photometry_dir)
    _cfg = cfg or AppConfig()
    lc_dir = phot / "lightcurves"
    if at_df is None:
        at_df = pd.read_csv(phot / "active_targets.csv", low_memory=False, dtype=_GAIA_ID_DTYPE)
    if summary_df is None:
        summary_df = pd.read_csv(phot / "photometry_summary.csv", low_memory=False, dtype={"catalog_id": str})
    if comp_index is None:
        from comp_table import load_comp_table, used  # noqa: PLC0415

        _cdf = used(load_comp_table(phot / "comparison_stars_per_target.csv"))
        _cdf["_tid"] = _cdf["target_catalog_id"].map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
        comp_index = {k: g.drop(columns=["_tid"]) for k, g in _cdf.groupby("_tid")}
    reports_dir = phot / "lightcurves_reports"
    # OUTPUT-REVIEW-2026-10 step 7: the folders hold this export only (no file of an earlier export)
    from run_outputs import clear_export_dirs  # noqa: PLC0415

    clear_export_dirs(phot, ("aavso", "varastro"))
    (reports_dir / "aavso").mkdir(parents=True, exist_ok=True)
    (reports_dir / "varastro").mkdir(parents=True, exist_ok=True)
    run_cite = build_run_citation_context(
        _cfg,
        pipeline_meta=load_pipeline_meta(phot),
        targets_df=at_df,
        obs_group=obs_group,
    )
    sum_by: dict[str, pd.Series] = {}
    if "catalog_id" in summary_df.columns:
        for _, r in summary_df.iterrows():
            cid = str(normalize_gaia_source_id(r.get("catalog_id")) or "").strip()
            if cid:
                sum_by[cid] = r
    failures: list[ExportFailure] = []
    stats: dict[str, int] = {}
    n_ok = n_skip = 0
    for _, trow in at_df.iterrows():
        tcid = str(normalize_gaia_source_id(trow.get("catalog_id", "")) or "").strip()
        if not tcid:
            continue
        # a target measured in this run: summary row with frames (skipped targets have none)
        _srow = sum_by.get(tcid)
        _nfr = pd.to_numeric(pd.Series([_srow.get("n_frames") if _srow is not None else None]), errors="coerce").iloc[0]
        if not (pd.notna(_nfr) and _nfr > 0) or not lc_csv_path(lc_dir, tcid, "aperture").is_file():
            n_skip += 1
            continue
        qmap: dict[str, str] = {}
        cq = lc_dir / f"comp_quality_{tcid}.json"
        if cq.is_file():
            try:
                for qk, qv in parse_comp_quality_json_map(json.loads(cq.read_text(encoding="utf-8"))).items():
                    q2 = str(qv.get("quality", "")).strip().lower()
                    if q2 != "excluded":
                        qmap[str(normalize_gaia_source_id(qk) or "")] = q2
            except (OSError, ValueError):
                qmap = {}
        try:
            out = export_all_method_lightcurve_reports(
                reports_dir,
                trow,
                lc_dir=lc_dir,
                target_cid=tcid,
                comp_df=(comp_index.get(tcid, pd.DataFrame())).copy(),
                summary_row=sum_by.get(tcid, pd.Series(dtype=object)),
                observer_code=str(getattr(_cfg, "observer_code", "") or ""),
                observer_name=str(getattr(_cfg, "observer_name", "") or "Unknown Observer"),
                comp_quality_map=qmap or None,
                cfg=_cfg,
                obs_group=obs_group,
                targets_df=at_df,
                run_citation_ctx=run_cite,
                export_failures=failures,
                export_stats=stats,
            )
            if out:
                n_ok += 1
            else:
                n_skip += 1
        except Exception as exc:  # noqa: BLE001
            record_export_failure(failures, tcid, "all", f"export batch error: {exc}")
            n_skip += 1
    log_export_batch_summary(failures, stats)
    logging.info("[EXPORT] %s: %d targets exported, %d skipped", obs_group, n_ok, n_skip)
    return {"n_ok": n_ok, "n_skip": n_skip, "failures": failures, "stats": stats}
