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
from report_methods import (
    aavso_export_path,
    active_report_methods,
    lc_csv_path,
    software_method_label,
    varastro_export_path,
)
from check_star_kmag import (
    build_aligned_comp_inst,
    check_catalog_id_from_sidecar,
    check_kmag_sidecar_path,
    comp_ensemble_maps,
    kmag_from_sidecar,
    kmag_values_for_export,
    resolve_proc_csv_dir,
    select_check_star,
)
from photometry_core import (
    _resolve_plate_scale_arcsec_per_px,
    check_comparison_stability,
    parse_comp_quality_json_map,
    pytics_iterative_weights,
)

# Gaia ID musi byt str - float64 straca cifry
_GAIA_ID_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}

# Gaia DR3 source_id is 19 decimal digits. A shorter numeric prefix in an export
# is a different star (EXPORT-HDR-01).
_GAIA_ID_FULL_DIGITS = 19


def find_truncated_gaia_ids(text: str, full_ids: list[str] | tuple[str, ...]) -> list[str]:
    """Return catalog IDs that appear in ``text`` only as a proper prefix.

    Truncated Gaia IDs are a worse defect than omitting them: the stub names a
    different star. Full IDs present in the text are allowed.
    """
    blob = str(text or "")
    out: list[str] = []
    for raw in full_ids:
        fid = str(raw or "").strip()
        if len(fid) < 10:
            continue
        if fid in blob:
            continue
        for n in range(len(fid) - 1, 9, -1):
            if fid[:n] in blob:
                out.append(fid)
                break
    return out


def format_aavso_notes_ensemble(*, n_comp: int, lc_method: str) -> str:
    """AAVSO NOTES: ensemble size, not a truncated Gaia list (EXPORT-HDR-01)."""
    meth = str(lc_method or "aperture").strip() or "aperture"
    n = max(0, int(n_comp))
    return f"meth={meth}|n_comp={n} GaiaDR3 ensemble"

# Single source for export headers (AAVSO #SOFTWARE + VarAstro Software line).
VYVAR_SOFTWARE_VERSION = "VYVAR 1.0"


class ExportFailure(TypedDict):
    target_id: str
    method: str
    reason: str


def _accumulate_export_stat(export_stats: dict[str, int] | None, key: str, n: int = 1) -> None:
    if export_stats is None:
        return
    export_stats[key] = int(export_stats.get(key, 0)) + int(n)


def _count_err_scatter_unmatched_epochs(lc_df: pd.DataFrame) -> int:
    if "err_scatter_unmatched" not in lc_df.columns:
        return 0
    col = lc_df["err_scatter_unmatched"]
    if col.dtype == bool:
        return int(col.sum())
    return int(col.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "t")).sum())


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
        _sc_un = int(export_stats.get("err_scatter_unmatched_epochs") or 0)
        if _tb_ref:
            logging.error(
                "[EXPORT] time_base refused (non-BJD_TDB or unknown/mixed): %d target-method(s)",
                _tb_ref,
            )
        if _sc_un:
            logging.info(
                "[EXPORT] err_scatter_unmatched epochs (err unchanged): %d total",
                _sc_un,
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


def _aavso_software_header_line(software_version: str, export_method: str) -> str:
    """Format AAVSO ``#SOFTWARE`` from the canonical version string."""
    ver = str(software_version or VYVAR_SOFTWARE_VERSION).strip() or VYVAR_SOFTWARE_VERSION
    soft_id = ver.replace(" ", "/") if "/" not in ver else ver
    return (
        f"#SOFTWARE={soft_id} ({software_method_label(export_method)} photometry; "
        "Broeg 2005 ensemble)\n"
    )


def _resolve_export_arcsec_per_px(
    photometry_dir: Path,
    cfg: AppConfig,
) -> float | None:
    """Derive plate scale (arcsec/px) for export headers - derive-or-None (no magic 1.3).

    Priority: ``pipeline_meta.json`` ``plate_scale_arcsec_px`` (Phase 2A session);
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

    Phase 2A persists ``observer_location`` (param_resolver: draft ID_LOCATION ->
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


def _append_aavso_observer_location_lines(
    lines: list[str], cfg: AppConfig | None, site: dict[str, Any] | None = None
) -> None:
    coords = _site_coords(cfg, site)
    if coords is None:
        return
    lat, lon, alt, _name = coords
    lines.append(f"#LATITUDE={lat:.4f}\n")
    lines.append(f"#LONGITUDE={lon:.4f}\n")
    lines.append(f"#ELEVATION={alt:.0f}\n")


def _append_varastro_site_line(
    lines: list[str], cfg: AppConfig | None, site: dict[str, Any] | None = None
) -> None:
    coords = _site_coords(cfg, site)
    if coords is None:
        return
    lat, lon, alt, name = coords
    if not name:
        name = "Observer site"
    lines.append(f"# Site: {name} ({lat:.4f} degN, {lon:.4f} degE, {alt:.0f} m)\n")


def _vyvar_export_citation_lines(
    cfg: AppConfig | None = None,
    *,
    run_ctx: Any = None,
    photometry_dir: Path | None = None,
    target_row: pd.Series | None = None,
    targets_df: pd.DataFrame | None = None,
    lc_method: str | None = None,
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
            lc_method=lc_method,
        )
    return emit_export_citation_lines(run_ctx)


def _varastro_alg_lines(
    fresh_cfg: AppConfig | None,
    *,
    run_ctx: Any = None,
    photometry_dir: Path | None = None,
    lc_method: str | None = None,
) -> list[str]:
    """VAR.ASTRO per-run algorithm summary (CITATIONS.bib)."""
    if run_ctx is None:
        run_ctx = build_run_citation_context(
            fresh_cfg,
            pipeline_meta=load_pipeline_meta(photometry_dir),
            lc_method=lc_method,
        )
    return emit_varastro_method_summary_lines(run_ctx)


def _safe_filename(name: str) -> str:
    """Sanitizuj VSX meno pre filesystem. 'BO CVn' -> 'BO_CVn'."""
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


def _fmt_opt_num(v: Any, fmt: str, *, na: str = "na") -> str:
    try:
        f = float(pd.to_numeric(v, errors="coerce"))
    except (TypeError, ValueError) as exc:  # noqa: BLE001
        return na
    return (format(f, fmt) if math.isfinite(f) else na)


def _aavso_gs11_notes_suffix(summary_row: pd.Series, cfg: AppConfig | None) -> str:
    """Append GS11 dilution tag to AAVSO NOTES when blend is significant."""
    if cfg is None or not bool(getattr(cfg, "gs11_dilution_enabled", False)):
        return ""
    try:
        d = float(pd.to_numeric(summary_row.get("dilution_factor", 1.0), errors="coerce"))
    except (TypeError, ValueError):
        return ""
    if math.isfinite(d) and d < 0.99:
        return f"|GS11:D={d:.3f}"
    return ""


def _fmt_opt_int(v: Any, *, na: str = "na") -> str:
    try:
        f = float(pd.to_numeric(v, errors="coerce"))
    except (TypeError, ValueError) as exc:  # noqa: BLE001
        return na
    if not math.isfinite(f):
        return na
    try:
        return str(int(f))
    except Exception:  # noqa: BLE001
        return na


# Default AAVSO observer placeholder when config is unset (must trigger export/validator warning).

# Built-in filter/setup name -> AAVSO Extended FILT code (uppercase keys).
# User overrides via ``AppConfig.aavso_filter_map`` (merged on lookup).
_AAVSO_FILTER_BUILTIN: dict[str, str] = {
    "U": "U",
    "B": "B",
    "V": "V",
    "R": "R",
    "I": "I",
    "RJ": "RJ",
    "IJ": "IJ",
    "RC": "Rc",
    "IC": "Ic",
    "SU": "SU",
    "SG": "SG",
    "SR": "SR",
    "SI": "SI",
    "SZ": "SZ",
    "CV": "CV",
    "CR": "CR",
    "TB": "TB",
    "TG": "TG",
    "TR": "TR",
    "J": "J",
    "H": "H",
    "K": "K",
    "Y": "Y",
    "L": "CV",
    "LUM": "CV",
    "LUMINANCE": "CV",
    "NOFILTER": "CV",
    "NO_FILTER": "CV",
    "CLEAR": "CV",
}


def _resolve_aavso_filter(
    filter_name: str,
    cfg: AppConfig | None = None,
) -> tuple[str, str | None]:
    """Map setup/filter name to AAVSO FILT code.

    Returns ``(code, warning_message)``. Unrecognized filters emit ``UNKN`` and a
    non-empty warning - never silently default to CV.
    """
    raw = str(filter_name or "").strip()
    if not raw:
        return "UNKN", "FILT empty - review before AAVSO submit"

    key = _filter_lookup_key(raw)
    merged: dict[str, str] = dict(_AAVSO_FILTER_BUILTIN)
    if cfg is not None:
        umap = getattr(cfg, "aavso_filter_map", None) or {}
        for uk, uv in umap.items():
            nk = _filter_lookup_key(str(uk))
            if nk and str(uv).strip():
                merged[nk] = str(uv).strip()

    if key in merged:
        return merged[key], None

    key_compact = key.replace("_", "")
    if "NOFILTER" in key_compact or key_compact == "CLEAR" or key.startswith("LUM"):
        return "CV", None

    return (
        "UNKN",
        f"FILT unrecognized: '{raw}' - map via aavso_filter_map or review before AAVSO submit",
    )


def _filter_lookup_key(name: str) -> str:
    s = str(name or "").strip().upper()
    return re.sub(r"[\s\-]+", "_", s)


def resolve_aavso_filt_from_obs_group(
    obs_group: str,
    cfg: AppConfig | None = None,
) -> tuple[str, str | None]:
    """End-to-end AAVSO FILT: OSC channel tokens TR/TG/TB; mono via setup parse."""
    from osc_align import is_onerggb_internal_obs_group, obs_group_band_token

    og = str(obs_group or "").strip()
    if is_onerggb_internal_obs_group(og):
        return "", "oneRGGB internal-only (E1 - not exported)"
    tok = obs_group_band_token(og)
    if tok in ("TR", "TG", "TB"):
        return tok, None
    setup_filter_raw, _, _ = _guess_setup_info_from_obs_group(og)
    return _resolve_aavso_filter(setup_filter_raw, cfg)


def _prepare_osc_comp_df_for_export(
    comp_df: pd.DataFrame | None,
    aavso_band_token: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach Johnson comp mags for OSC band exports; return notes for excluded comps."""
    from gaia_johnson import TRANSFORM_CITATION, transform_comp_row_for_osc_band

    notes: list[str] = []
    if comp_df is None or comp_df.empty:
        return pd.DataFrame() if comp_df is None else comp_df.copy(), notes
    if aavso_band_token not in ("TR", "TG", "TB"):
        return comp_df.copy(), notes

    out = comp_df.copy()
    j_mags: list[float] = []
    j_errs: list[float] = []
    j_ok: list[bool] = []
    for _, row in out.iterrows():
        res = transform_comp_row_for_osc_band(row, aavso_band_token, log_exclusions=True)
        j_mags.append(float(res.johnson_mag) if res.ok else float("nan"))
        j_errs.append(float(res.johnson_mag_err) if res.ok else float("nan"))
        j_ok.append(bool(res.ok))
        if not res.ok:
            cid = str(row.get("catalog_id", "") or "")[:18]
            notes.append(f"comp {cid} excluded: {res.reason}")
    out["johnson_mag"] = j_mags
    out["johnson_mag_err"] = j_errs
    out["johnson_ok"] = j_ok
    out.attrs["osc_transform_citation"] = TRANSFORM_CITATION
    return out, notes


def _osc_export_citation_context(
    cfg: AppConfig | None,
    *,
    photometry_dir: Path | None,
    obs_group: str,
    targets_df: pd.DataFrame | None = None,
    target_row: pd.Series | None = None,
    lc_method: str | None = None,
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
            lc_method=lc_method,
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


# Zakrytove typy podla VSX dokumentacie
_ECLIPSING_TOKENS = frozenset(
    {
        "E",  # Generic eclipsing
        "EA",  # Algol (Beta Persei)
        "EB",  # Beta Lyrae
        "EW",  # W Ursae Majoris
        "EP",  # Planetary transits
        "E-DO",  # Disk occultation
        "ELL",  # Ellipsoidal (no eclipse but VAR.ASTRO ma zaujem)
        "EC",  # Contact binaries - ASAS survey typ
        "ED",  # Detached eclipsing - ASAS survey typ
        "ESD",  # Semi-detached eclipsing - ASAS survey typ
    }
)


def _token_is_eclipsing(token: str) -> bool:
    """Skontroluj ci jeden VSX token (bez specialnych znakov) je zakrytovy."""
    t = str(token or "").strip().rstrip(":").upper()
    if not t:
        return False
    if t in _ECLIPSING_TOKENS:
        return True
    if t.startswith("E-"):
        return True
    return False


def _is_eclipsing(vsx_type: str) -> bool:
    """Vrati True ak VSX typ obsahuje zakrytovu komponentu (VSX konvencie)."""
    if not vsx_type or not isinstance(vsx_type, str):
        return False

    vsx_type = vsx_type.strip()
    if not vsx_type:
        return False

    # Pipe | -> OR/neistota: len ak VSETKY alternativy su zakrytove
    if "|" in vsx_type:
        alternatives = [a.strip() for a in vsx_type.split("|") if a.strip()]
        if not alternatives:
            return False
        for alt in alternatives:
            main = alt.split("/")[0].split("+")[0].strip().rstrip(":")
            if not _token_is_eclipsing(main):
                return False
        return True

    # Plus + -> AND: ak KTORYKOLVEK komponent je zakrytovy -> True
    plus_parts = vsx_type.split("+")
    for part in plus_parts:
        part = part.strip()
        if not part:
            continue
        main_token = part.split("/")[0].strip()
        if _token_is_eclipsing(main_token):
            return True

    return False


def _select_check_star(
    comp_df: pd.DataFrame,
    *,
    ensemble_ids: set[str] | None = None,
    n_comp_min: int = 3,
    cfg: AppConfig | None = None,
) -> pd.Series | None:
    return select_check_star(
        comp_df,
        ensemble_ids=ensemble_ids or set(),
        n_comp_min=n_comp_min,
        cfg=cfg,
    )


def _copy_field_image(
    lc_dir: Path,
    catalog_id: str,
    dst_dir: Path,
    fname: str,
    obs_date: str,
) -> str | None:
    """Skopiruj field map PNG do varastro adresara."""
    candidates = [
        lc_dir / f"field_map_{catalog_id}.png",
        lc_dir / "field_map.png",
        lc_dir.parent / "field_map.png",
    ]
    for src in candidates:
        try:
            if src.exists():
                dst_name = f"{fname}_{obs_date}_field.png"
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


def _normalize_comp_df_export_columns(
    comp_df: pd.DataFrame,
    comp_quality_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Map Phase-1 CSV names (``comp_rms``, ``comp_weight``) to export columns (``p2p_rms``, ``w_rel``)."""
    df = comp_df.copy()
    if df.empty:
        return df
    if "p2p_rms" not in df.columns and "comp_rms" in df.columns:
        df["p2p_rms"] = df["comp_rms"]
    if "w_rel" not in df.columns and "comp_weight" in df.columns:
        from photometry_core import apply_comp_w_rel_for_display

        df = apply_comp_w_rel_for_display(df, comp_quality_map)
    return df


def _export_comp_status_label(row: pd.Series, quality_map: dict[str, str] | None) -> str:
    """COMP TABLE status: good / suspect / rejected; fallback ``comp_quality`` map alebo ``unknown``."""
    for col in ("status", "stav", "quality"):
        if col not in row.index:
            continue
        v = row.get(col)
        try:
            if v is None or (isinstance(v, float) and not math.isfinite(float(v))):
                continue
        except Exception:  # noqa: BLE001
            # EXC-0086: T2 -- report/export may omit or misstate (if v is None or (isinstance(v, float) and not math.... (EXCEPT-BULK-2 2026-07-08)
            continue
        s = str(v).strip().lower()
        if not s or s in ("?", "nan", "none"):
            continue
        if s == "excluded":
            return "rejected"
        if s in ("good", "suspect", "rejected"):
            return s
    if quality_map:
        k = str(normalize_gaia_source_id(row.get("catalog_id")) or "").strip()
        q = str(quality_map.get(k, "") or "").strip().lower()
        if q == "excluded":
            return "rejected"
        if q in ("good", "suspect"):
            return q
    return "unknown"


def _format_varastro_comp_table(
    comp_df: pd.DataFrame,
    *,
    comp_quality_map: dict[str, str] | None = None,
    use_johnson_mag: bool = False,
    post_weight_rel_map: dict[str, float] | None = None,
) -> str:
    header = (
        "# Nr  CatalogId             Mag    BP-RP  dBPRP  tier_color  "
        "p2p_RMS  w_pre  w_post tier status\n"
    )
    if comp_df is None or comp_df.empty:
        return header
    lines = [header]
    df = _normalize_comp_df_export_columns(comp_df, comp_quality_map)
    export_row_n = 0
    for _, row in df.iterrows():
        cid_key = str(normalize_gaia_source_id(row.get("catalog_id")) or "").strip()
        if comp_quality_map and cid_key:
            q_skip = str(comp_quality_map.get(cid_key, "") or "").strip().lower()
            if q_skip == "excluded":
                continue
        if use_johnson_mag and "johnson_ok" in row.index and not bool(row.get("johnson_ok")):
            continue
        export_row_n += 1
        cid = str(row.get("catalog_id", "") or "")[:19].ljust(19)
        if use_johnson_mag and "johnson_mag" in row.index:
            mag = pd.to_numeric(row.get("johnson_mag", float("nan")), errors="coerce")
        else:
            mag = pd.to_numeric(row.get("mag", float("nan")), errors="coerce")
        bprp = pd.to_numeric(row.get("bp_rp", float("nan")), errors="coerce")
        dbprp = pd.to_numeric(row.get("delta_bprp_abs", float("nan")), errors="coerce")
        p2p = pd.to_numeric(row.get("p2p_rms", float("nan")), errors="coerce")
        w_rel = pd.to_numeric(row.get("w_rel", float("nan")), errors="coerce")
        tier = pd.to_numeric(row.get("tier", row.get("comp_tier", 4)), errors="coerce")
        st = _export_comp_status_label(row, comp_quality_map)

        mag_s = f"{float(mag):.3f}" if math.isfinite(float(mag)) else "  -  "
        bprp_s = f"{float(bprp):.3f}" if math.isfinite(float(bprp)) else "  -  "
        dbprp_s = f"{float(dbprp):.3f}" if math.isfinite(float(dbprp)) else "  -  "
        p2p_s = f"{float(p2p):.4f}" if math.isfinite(float(p2p)) else "  -  "
        wrel_s = f"{float(w_rel):.3f}" if math.isfinite(float(w_rel)) else "  -  "
        w_post = float("nan")
        if post_weight_rel_map:
            try:
                w_post = float(post_weight_rel_map.get(cid_key, float("nan")))
            except Exception:  # noqa: BLE001
                w_post = float("nan")
        wpost_s = f"{float(w_post):.3f}" if math.isfinite(float(w_post)) else "  -  "
        try:
            tier_i = int(tier) if math.isfinite(float(tier)) else 4
        except Exception:  # noqa: BLE001
            tier_i = 4
        st_s = str(st)[:10].ljust(10)

        tier_cs = str(row.get("color_tier_src", "") or "")[:12].ljust(12)
        lines.append(
            f"# C{export_row_n:02d} {cid} {mag_s}  {bprp_s}  {dbprp_s}  {tier_cs} {p2p_s}  "
            f"{wrel_s}  {wpost_s}  {tier_i}  {st_s}\n"
        )
    return "".join(lines)


def _export_post_weight_rel_map(
    *,
    comp_df: pd.DataFrame,
    lc_normal: pd.DataFrame,
    proc_dir: Path | None,
    comp_quality_map: dict[str, str] | None,
    cfg: AppConfig,
    export_method: str,
    proc_csv_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, float]:
    """Recompute export-time post-PyTICS relative weights for the real ensemble."""
    if proc_dir is None or not proc_dir.is_dir() or comp_df is None or comp_df.empty or "catalog_id" not in comp_df.columns:
        return {}
    source_files = lc_normal.get("source_file", pd.Series([""] * len(lc_normal))).astype(str).tolist()
    if not source_files:
        return {}
    comp_ids: list[str] = []
    for _, crow in comp_df.iterrows():
        cid = str(normalize_gaia_source_id(crow.get("catalog_id", "")) or "").strip()
        if not cid:
            continue
        if comp_quality_map and str(comp_quality_map.get(cid, "") or "").strip().lower() == "excluded":
            continue
        if cid not in comp_ids:
            comp_ids.append(cid)
    if len(comp_ids) < 2:
        return {}
    comp_lc = build_aligned_comp_inst(
        proc_dir,
        comp_ids,
        source_files,
        cfg,
        export_method,
        csv_cache=proc_csv_cache,
    )
    _comp_catalog_mag, _comp_tier_map, comp_rms_map, _tier_weights = comp_ensemble_maps(comp_df, cfg)
    comp_quality = check_comparison_stability(
        {c: comp_lc[c] for c in comp_ids if c in comp_lc},
        comp_rms_map=comp_rms_map,
        n_comp_min=2,
        outlier_sigma=3.0,
        common_mode_detrend=True,
    )
    if comp_quality_map:
        for cid, q in comp_quality_map.items():
            q2 = str(q or "").strip().lower()
            if cid in comp_quality:
                if q2 == "excluded":
                    comp_quality[cid]["quality"] = "excluded"
                elif q2 in ("good", "suspect"):
                    comp_quality[cid]["quality"] = q2
    post_rms = pytics_iterative_weights(
        comp_lc={c: comp_lc[c] for c in comp_ids if c in comp_lc},
        comp_quality=comp_quality,
        comp_rms_map=comp_rms_map,
        n_iter=int(cfg.pytics_n_iter),
        enabled=bool(cfg.pytics_enabled),
    )
    weights: dict[str, float] = {}
    for cid in comp_ids:
        r = float(post_rms.get(cid, float("nan")))
        if math.isfinite(r) and r > 1e-9:
            weights[cid] = 1.0 / (r * r)
    if not weights:
        return {}
    w_max = max(weights.values())
    return {cid: (w / w_max) for cid, w in weights.items() if math.isfinite(w) and w > 0}


def _select_export_lc_rows(lc_df: pd.DataFrame) -> pd.DataFrame:
    """Rows suitable for AAVSO/VAR.ASTRO export (canonical ``mag_calib_final`` when present)."""
    if lc_df is None or lc_df.empty:
        return pd.DataFrame()
    work = lc_df.copy()
    mag_col = "mag_calib"
    if "mag_calib_final" in work.columns:
        mag_col = "mag_calib_final"
    elif "mag_calib_ac" in work.columns:
        ac_ok = (
            work["ac_ok"].astype(bool)
            if "ac_ok" in work.columns
            else pd.Series(False, index=work.index)
        )
        mac = pd.to_numeric(work["mag_calib_ac"], errors="coerce")
        use_ac = ac_ok & mac.notna() & np.isfinite(mac.to_numpy(dtype=float))
        if use_ac.any():
            work = work.copy()
            work["_export_mag"] = pd.to_numeric(work["mag_calib"], errors="coerce")
            work.loc[use_ac, "_export_mag"] = mac.loc[use_ac]
            mag_col = "_export_mag"
    mag = pd.to_numeric(work.get(mag_col), errors="coerce")
    bjd = pd.to_numeric(work.get("bjd"), errors="coerce")
    finite = mag.notna() & np.isfinite(mag.to_numpy(dtype=float)) & bjd.notna() & np.isfinite(
        bjd.to_numpy(dtype=float)
    )
    if "flag" in work.columns:
        fl = work["flag"].astype(str).str.strip().str.lower()
        good = fl.isin(("normal", "")) | fl.isna()
        bad = fl.isin(("no_data", "saturated", "edge_fail", "nondetection"))
        mask = finite & (good | ~bad)
        out = work.loc[mask].copy()
        if out.empty and finite.any():
            out = work.loc[finite].copy()
    else:
        out = work.loc[finite].copy()
    if mag_col == "_export_mag" and "mag_calib" in out.columns:
        out = out.copy()
        out["mag_calib"] = out["_export_mag"]
    elif mag_col == "mag_calib_final" and "mag_calib" in out.columns:
        out = out.copy()
        out["mag_calib"] = pd.to_numeric(out["mag_calib_final"], errors="coerce")
    return out


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
    arcsec_per_px: float | None = None,
    software_version: str | None = None,
    cfg: AppConfig | None = None,
    lc_dir: Path | None = None,
    obs_group: str = "",
    targets_df: pd.DataFrame | None = None,
    run_citation_ctx: Any = None,
    export_method: str = "aperture",
    active_methods: list[str] | None = None,
    proc_csv_cache: dict[str, pd.DataFrame] | None = None,
    export_failures: list[ExportFailure] | None = None,
    export_stats: dict[str, int] | None = None,
) -> dict[str, Path]:
    """Generuje AAVSO a VAR.ASTRO subory pre jeden target."""
    fresh_cfg = cfg or AppConfig()
    if not str(observer_code).strip() and fresh_cfg is not None:
        observer_code = str(getattr(fresh_cfg, "observer_code", "") or "")
    if observer_name == "Unknown Observer" and fresh_cfg is not None:
        _on = str(getattr(fresh_cfg, "observer_name", "") or "").strip()
        if _on:
            observer_name = _on

    out_base = Path(output_dir)
    (out_base / "aavso").mkdir(parents=True, exist_ok=True)
    (out_base / "varastro").mkdir(parents=True, exist_ok=True)

    _phot_dir = out_base.parent
    _sw_ver = str(software_version or VYVAR_SOFTWARE_VERSION).strip() or VYVAR_SOFTWARE_VERSION
    arcsec_per_px = _resolve_export_arcsec_per_px(_phot_dir, fresh_cfg)

    # Per-draft resolved observer site (param_resolver, persisted by Phase 2A) so the
    # exported #LATITUDE / VAR.ASTRO site matches the location used for BJD/airmass.
    _resolved_site = _resolved_site_from_meta(out_base)

    vsx_name = str(target_row.get("vsx_name", "") or "").strip() or "unknown"
    safe = _safe_filename(vsx_name)
    _export_method = str(export_method or "aperture").strip().lower()
    _target_id = str(target_row.get("catalog_id", "") or vsx_name).strip()

    # Use exportable LC points (normal frames; canonical mag_calib_final when present).
    lc0 = lc_df.copy() if lc_df is not None else pd.DataFrame()
    lc_normal = _select_export_lc_rows(lc0)

    if lc_normal.empty:
        record_export_failure(
            export_failures,
            _target_id,
            _export_method,
            "no exportable LC points (flags/mag empty)",
        )
        return {}

    from photometry_core import (  # noqa: PLC0415
        TIME_BASE_BJD_TDB,
        resolve_lc_time_base,
    )

    try:
        _time_base = resolve_lc_time_base(lc_normal)
    except ValueError as exc:
        record_export_failure(
            export_failures,
            _target_id,
            _export_method,
            f"time_base invalid: {exc}",
        )
        _accumulate_export_stat(export_stats, "time_base_refused")
        return {}

    if _time_base != TIME_BASE_BJD_TDB:
        record_export_failure(
            export_failures,
            _target_id,
            _export_method,
            f"time_base={_time_base}: export refused (AAVSO requires BJD_TDB)",
        )
        _accumulate_export_stat(export_stats, "time_base_refused")
        return {}

    _n_scatter_unmatched = _count_err_scatter_unmatched_epochs(lc_normal)
    if _n_scatter_unmatched:
        _accumulate_export_stat(
            export_stats,
            "err_scatter_unmatched_epochs",
            _n_scatter_unmatched,
        )
        logging.info(
            "[EXPORT] %s method=%s: %d epoch(s) err_scatter_unmatched (excluded from LC)",
            str(vsx_name),
            _export_method,
            _n_scatter_unmatched,
        )

    _maybe_set_airmass_citation(run_citation_ctx, lc_normal)

    bjd_first = pd.to_numeric(lc_normal.iloc[0].get("bjd", float("nan")), errors="coerce")
    date_tag = _bjd_to_datestr_yyyymmdd(float(bjd_first)) if math.isfinite(float(bjd_first)) else "unknown"

    _active_methods = list(active_methods or active_report_methods(fresh_cfg))
    aavso_path = aavso_export_path(out_base, safe, date_tag, _export_method, active_methods=_active_methods)
    var_path = varastro_export_path(out_base, safe, date_tag, _export_method, active_methods=_active_methods)

    # Setup/filter info (best effort).
    obs_group_resolved = str(obs_group or "").strip()
    if not obs_group_resolved:
        obs_group_resolved = str(
            summary_row.get("obs_group", "") or summary_row.get("setup", "") or ""
        ).strip()
    if not obs_group_resolved:
        obs_group_resolved = str(summary_row.get("obs_group_name", "") or "").strip()
    if not obs_group_resolved:
        try:
            setup_dir = out_base.parent.parent
            if setup_dir.name and setup_dir.name.lower() not in ("platesolve", "photometry"):
                obs_group_resolved = setup_dir.name
        except Exception:  # noqa: BLE001
            # EXC-0087: T2 -- report/export may omit or misstate (if setup_dir.name and setup_dir.name.lower() not in... (EXCEPT-BULK-2 2026-07-08)
            pass

    from invariants_runtime import check_osc03_export_eligibility
    from osc_align import is_onerggb_internal_obs_group

    if is_onerggb_internal_obs_group(obs_group_resolved):
        logging.info("[EXPORT] skip oneRGGB internal obs-group %s", obs_group_resolved)
        record_export_failure(
            export_failures,
            _target_id,
            _export_method,
            "oneRGGB internal-only (OSC E1 - not exported)",
        )
        return {}

    setup_filter_raw, exptime_s, binning = _guess_setup_info_from_obs_group(obs_group_resolved)
    aavso_filter, filt_warn = resolve_aavso_filt_from_obs_group(obs_group_resolved, fresh_cfg)
    _osc_export = aavso_filter in ("TR", "TG", "TB")
    comp_export_df, _osc_comp_notes = _prepare_osc_comp_df_for_export(
        comp_df,
        aavso_filter,
    )
    check_osc03_export_eligibility(
        obs_group_resolved,
        aavso_filter,
        meta={"target": _target_id, "method": _export_method},
    )

    # Target catalog info.
    target_cid = str(target_row.get("catalog_id", "") or "").strip()
    t_mag = pd.to_numeric(target_row.get("mag", float("nan")), errors="coerce")
    t_bprp = pd.to_numeric(target_row.get("bp_rp", float("nan")), errors="coerce")

    # Check star (exclude Phase-2A ensemble members when comp_quality sidecar exists).
    from check_star_kmag import resolve_ensemble_ids_for_check  # noqa: PLC0415

    _lc_dir_pre = Path(lc_dir) if lc_dir is not None else (out_base.parent / "lightcurves")
    _ens_ids = resolve_ensemble_ids_for_check(
        target_cid,
        comp_df,
        lc_dir=_lc_dir_pre,
        comp_quality_map=comp_quality_map,
        cfg=fresh_cfg,
    )
    check_row = _select_check_star(
        comp_df,
        ensemble_ids=_ens_ids,
        n_comp_min=3,
        cfg=fresh_cfg,
    )
    _lc_dir = Path(lc_dir) if lc_dir is not None else (out_base.parent / "lightcurves")
    _side_cid = check_catalog_id_from_sidecar(_lc_dir, target_cid)
    if _side_cid:
        check_cid = _side_cid
        if check_row is None or str(normalize_gaia_source_id(check_row.get("catalog_id")) or "").strip() != _side_cid:
            if comp_df is not None and not comp_df.empty and "catalog_id" in comp_df.columns:
                _hit = comp_df[
                    comp_df["catalog_id"].astype(str).map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
                    == _side_cid
                ]
                if not _hit.empty:
                    check_row = _hit.iloc[0]
    else:
        check_cid = str(check_row.get("catalog_id")) if check_row is not None else "na"
    _phot_dir = out_base.parent
    _proc_dir = resolve_proc_csv_dir(_phot_dir, obs_group_resolved)
    kmag_values, _kmag_mode = kmag_values_for_export(
        check_row,
        comp_df,
        lc_normal,
        target_cid=target_cid,
        lc_dir=_lc_dir,
        proc_dir=_proc_dir,
        comp_quality_map=comp_quality_map,
        cfg=fresh_cfg,
        export_method=_export_method,
        proc_csv_cache=proc_csv_cache,
    )
    logging.debug("[EXPORT] KMAG mode=%s for %s", _kmag_mode, str(vsx_name))
    _post_weight_rel = _export_post_weight_rel_map(
        comp_df=comp_df,
        lc_normal=lc_normal,
        proc_dir=_proc_dir,
        comp_quality_map=comp_quality_map,
        cfg=fresh_cfg,
        export_method=_export_method,
        proc_csv_cache=proc_csv_cache,
    )

    _lc_method = _export_method
    if "method" in lc_normal.columns and _export_method == "aperture":
        _mvals = lc_normal["method"].astype(str).str.strip().str.lower()
        _mvals = _mvals[_mvals != ""]
        if not _mvals.empty:
            try:
                _lc_method = str(_mvals.mode().iloc[0])
            except Exception:  # noqa: BLE001
                _lc_method = str(_mvals.iloc[0])

    # NOTES: ensemble size, never a truncated Gaia list (EXPORT-HDR-01).
    _n_notes_comp = 0
    _full_note_ids: list[str] = []
    if comp_df is not None and (not comp_df.empty) and "catalog_id" in comp_df.columns:
        for _, crow in comp_df.iterrows():
            v2 = str(crow.get("catalog_id", "") or "").strip()
            if not v2:
                continue
            ck = str(normalize_gaia_source_id(v2) or "").strip()
            if comp_quality_map and ck:
                if str(comp_quality_map.get(ck, "") or "").strip().lower() == "excluded":
                    continue
            _n_notes_comp += 1
            _full_note_ids.append(v2)
    notes_first = format_aavso_notes_ensemble(n_comp=_n_notes_comp, lc_method=_lc_method)
    _gs11_note = _aavso_gs11_notes_suffix(summary_row, fresh_cfg)
    if _gs11_note:
        notes_first = (notes_first + _gs11_note)[:100]
    _trunc = find_truncated_gaia_ids(notes_first, _full_note_ids)
    if _trunc:
        raise ValueError(f"AAVSO NOTES would emit truncated Gaia IDs: {_trunc}")

    _trust = str(summary_row.get("trust", "") or "").strip().upper()
    _trust_reason = str(summary_row.get("trust_reason", "") or "").strip()
    if _trust:
        from trust_flag_core import format_export_trust_note  # noqa: PLC0415

        _remain = max(0, 100 - len(notes_first) - 1)
        if _remain >= 8:
            _tnote = format_export_trust_note(_trust, _trust_reason, max_len=_remain)
            if _tnote:
                notes_first = (notes_first + "|" + _tnote)[:100]

    # --- AAVSO Extended ---
    _cite_kw: dict[str, Any] = {
        "photometry_dir": _phot_dir,
        "target_row": target_row,
        "lc_method": _export_method,
        "obs_group": obs_group_resolved,
    }
    if run_citation_ctx is not None:
        _cite_kw["run_ctx"] = _osc_export_citation_context(
            fresh_cfg,
            photometry_dir=_phot_dir,
            obs_group=obs_group_resolved,
            run_ctx=run_citation_ctx,
            lc_method=_export_method,
        )
    elif targets_df is not None:
        _cite_kw["targets_df"] = targets_df
    if _export_method in ("psf", "adaptive"):
        _cite_kw["run_ctx"] = _osc_export_citation_context(
            fresh_cfg,
            photometry_dir=_phot_dir,
            obs_group=obs_group_resolved,
            targets_df=targets_df,
            target_row=target_row,
            lc_method=_export_method,
        )
    a_lines: list[str] = []
    a_lines.extend(_vyvar_export_citation_lines(fresh_cfg, **_cite_kw))
    if _osc_export:
        from gaia_johnson import TRANSFORM_CITATION

        a_lines.append(f"# OSC comp/check mags: Gaia G+BP-RP -> Johnson ({TRANSFORM_CITATION})\n")
        for note in _osc_comp_notes[:5]:
            a_lines.append(f"#WARNING={note}\n")
    if filt_warn:
        a_lines.append(f"#WARNING={filt_warn}\n")
        logging.warning("[EXPORT] %s (%s)", filt_warn, str(vsx_name))
    a_lines.append("#TYPE=Extended\n")
    _obc = str(observer_code).strip()
    if not _obc:
        # Fail-open for local files: still write OBSCODE= empty; warn only when unset.
        a_lines.append(
            "#WARNING=observer code not set - AAVSO submission requires an "
            "observer code (config: aavso_observer_code)\n"
        )
        logging.warning(
            "[EXPORT] observer code not set for %s - set aavso_observer_code in config",
            str(vsx_name),
        )
    a_lines.append(f"#OBSCODE={_obc}\n")
    _append_aavso_observer_location_lines(a_lines, fresh_cfg, _resolved_site)
    a_lines.append(_aavso_software_header_line(_sw_ver, _export_method))
    # WIDE-ERR-03: one comment line naming err mode, gain authority, calibration ranges.
    try:
        from err_calibration import ERR_CALIB_SIDECAR, load_sidecar  # noqa: PLC0415

        _eem = str(getattr(fresh_cfg, "export_err_mode", "calibrated") or "calibrated")
        _g_note = "g_pt/container"
        _cal_note = "none"
        if _phot_dir is not None:
            _cal = load_sidecar(Path(_phot_dir) / ERR_CALIB_SIDECAR)
            _gpt = None
            try:
                import json as _json  # noqa: PLC0415

                _gp = Path(_phot_dir) / "gain_photon_transfer.json"
                if _gp.is_file():
                    _gpt = _json.loads(_gp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                _gpt = None
            if isinstance(_gpt, dict):
                _auth = (_gpt.get("authority") or {})
                if _auth.get("value_e_per_adu_container") is not None:
                    _g_note = (
                        f"{_auth.get('source')}={float(_auth['value_e_per_adu_container']):.4g} "
                        "e-/ADU_container"
                    )
            if _cal and (_cal.get("bins") or []):
                _ss = [float(b.get("s", 1)) for b in _cal["bins"]]
                _rr = [float(b.get("sigma_r_rel", 0)) for b in _cal["bins"]]
                _cal_note = (
                    f"s=[{min(_ss):.3g},{max(_ss):.3g}] "
                    f"sigma_r_rel=[{min(_rr):.3g},{max(_rr):.3g}]"
                )
        a_lines.append(
            f"#ERR_MODEL=mode={_eem}; gain={_g_note}; calib={_cal_note}\n"
        )
    except Exception as _eem_exc:  # noqa: BLE001
        logging.debug("[EXPORT] ERR_MODEL comment skipped: %s", _eem_exc)
    a_lines.append("#DELIM=,\n")
    a_lines.append("#DATE=BJD\n")
    a_lines.append("#OBSTYPE=CCD\n")
    a_lines.append("#\n")
    _ct_ok_export = False
    _ct_c1_e = float("nan")
    _ct_c1_se_e = float("nan")
    _ct_corr_e = float("nan")
    _ct_mode_e = ""
    if "ct_ok" in lc_normal.columns:
        try:
            _ct_ok_export = False
            _ct_series = lc_normal["ct_ok"]
            for _v in _ct_series.tolist():
                if isinstance(_v, bool) and _v:
                    _ct_ok_export = True
                    break
                if str(_v).strip().lower() in ("true", "1", "yes"):
                    _ct_ok_export = True
                    break
        except Exception:  # noqa: BLE001
            _ct_ok_export = False
    if _ct_ok_export:
        try:
            _ct_c1_e = float(pd.to_numeric(lc_normal["ct_c1"], errors="coerce").dropna().iloc[0])
        except Exception:  # noqa: BLE001
            _ct_c1_e = float("nan")
        try:
            if "ct_c1_stderr" in lc_normal.columns:
                _ct_c1_se_e = float(
                    pd.to_numeric(lc_normal["ct_c1_stderr"], errors="coerce").dropna().iloc[0]
                )
        except Exception:  # noqa: BLE001
            _ct_c1_se_e = float("nan")
        try:
            _ct_corr_e = float(
                pd.to_numeric(lc_normal["ct_correction"], errors="coerce").dropna().iloc[0]
            )
        except Exception:  # noqa: BLE001
            _ct_corr_e = float("nan")
        try:
            if "ct_mode" in lc_normal.columns:
                _ct_mode_e = str(lc_normal["ct_mode"].iloc[0] or "").strip()
        except Exception:  # noqa: BLE001
            _ct_mode_e = ""
        a_lines.append(
            f"#COLOR_LEVEL: applied mode={_ct_mode_e or 'ct'} "
            f"k={_ct_c1_e:+.4f}"
            + (f"+/-{_ct_c1_se_e:.4f}" if math.isfinite(_ct_c1_se_e) else "")
            + " mag/BP-RP "
            f"correction={_ct_corr_e:+.4f} mag (constant; export-only)\n"
        )
        a_lines.append(
            "#COLOR_LEVEL: magnitudes are colour-level corrected toward ensemble "
            "weighted BP-RP; not a filter transformation to Johnson/Cousins\n"
        )
    _trans_flag = "YES" if _ct_ok_export else "NO"

    starid = vsx_name[:30]
    for row_pos, (_, row) in enumerate(lc_normal.iterrows()):
        bjd = pd.to_numeric(row.get("bjd", float("nan")), errors="coerce")
        mag_calib = pd.to_numeric(row.get("mag_calib", float("nan")), errors="coerce")
        err = pd.to_numeric(row.get("err", float("nan")), errors="coerce")
        am = pd.to_numeric(row.get("airmass", float("nan")), errors="coerce")
        if not (math.isfinite(float(bjd)) and math.isfinite(float(mag_calib))):
            continue

        date_s = f"{float(bjd):.6f}"
        mag_s = f"{float(mag_calib):.3f}"
        err_s = f"{float(err):.3f}" if math.isfinite(float(err)) else "na"
        am_s = f"{float(am):.3f}" if math.isfinite(float(am)) else "na"
        notes_s = notes_first if row_pos == 0 else "na"
        kmag_str = kmag_values[row_pos] if row_pos < len(kmag_values) else "na"

        # STARID,DATE,MAG,MAGERR,FILTER,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES
        a_lines.append(
            ",".join(
                [
                    starid,
                    date_s,
                    mag_s,
                    err_s,
                    aavso_filter,
                    _trans_flag,
                    "STD",
                    "ENSEMBLE",
                    "na",
                    (check_cid if check_cid else "na"),
                    kmag_str,
                    am_s,
                    "na",
                    "na",
                    notes_s,
                ]
            )
            + "\n"
        )
    paths: dict[str, Path] = {}
    try:
        aavso_path.write_text("".join(a_lines), encoding="utf-8")
        logging.info("[EXPORT] AAVSO: %s", str(aavso_path.name))
        paths["aavso"] = aavso_path
    except OSError as exc:
        record_export_failure(
            export_failures,
            _target_id,
            _export_method,
            f"AAVSO write error: {exc}",
        )
        return paths

    # --- VAR.ASTRO.CZ ---
    vsx_type = str(target_row.get("vsx_type", "") or "").strip()
    if not _is_eclipsing(vsx_type):
        logging.info("[EXPORT] Skip varastro %s - nie zakrytova (%s)", str(vsx_name), str(vsx_type))
        return {"aavso": aavso_path}

    v_lines: list[str] = []
    v_lines.extend(_vyvar_export_citation_lines(fresh_cfg, **_cite_kw))
    v_lines.append("# VYVAR - Differential Ensemble Photometry\n")
    v_lines.append(f"# Software: {_sw_ver} | Observer: {observer_name}\n")
    _append_varastro_site_line(v_lines, fresh_cfg, _resolved_site)
    if obs_group_resolved:
        v_lines.append(f"# Setup: {obs_group_resolved}\n")
    v_lines.append("#\n")
    v_lines.append("# TIME SYSTEM: BJD(TDB)\n")

    bjd_hjd_median = float("nan")
    if "hjd" in lc_normal.columns and "bjd" in lc_normal.columns:
        try:
            off = (pd.to_numeric(lc_normal["bjd"], errors="coerce") - pd.to_numeric(lc_normal["hjd"], errors="coerce")).dropna()
            if len(off) > 0:
                bjd_hjd_median = float(off.median())
        except Exception:  # noqa: BLE001
            bjd_hjd_median = float("nan")
    if math.isfinite(bjd_hjd_median):
        v_lines.append(f"#   BJD-HJD approx (median): {bjd_hjd_median:.6f}\n")
    v_lines.append("#\n")
    v_lines.extend(_varastro_alg_lines(fresh_cfg, photometry_dir=_phot_dir, lc_method=_export_method))
    _cfg_tier = fresh_cfg
    _tier_lims = (
        _cfg_tier.comp_tier_bprp_limits()
        if hasattr(_cfg_tier, "comp_tier_bprp_limits")
        else []
    )

    def _tier_lim(idx: int, dflt: float) -> float:
        return float((_tier_lims[idx] if idx < len(_tier_lims) else 0.0) or dflt)

    _t1 = _tier_lim(0, 0.25)
    _t2 = _tier_lim(1, 0.48)
    _t3 = _tier_lim(2, 0.79)
    v_lines.append(
        f"#   Tier system (|DeltaBP-RP| Gaia): "
        f"T1<={_t1:.2f}(w=1.00) T2<={_t2:.2f}(w=0.85) T3<={_t3:.2f}(w=0.50) T4>{_t3:.2f}(w=0.25)\n"
    )
    if _osc_export:
        from gaia_johnson import TRANSFORM_CITATION, johnson_band_for_osc_aavso_token

        _jb = johnson_band_for_osc_aavso_token(aavso_filter) or "?"
        v_lines.append(f"# Color system: Gaia BP-RP -> Johnson {_jb} ({TRANSFORM_CITATION})\n")
    else:
        v_lines.append("# Color system: Gaia BP-RP\n")
    if setup_filter_raw or exptime_s or binning or aavso_filter:
        v_lines.append(
            f"#   Filter: {aavso_filter or setup_filter_raw or 'CV'} | Exp: {exptime_s or 'na'}s | Bin: {binning or 'na'}\n"
        )
    ap_px = pd.to_numeric(summary_row.get("aperture_px", float("nan")), errors="coerce")
    fwhm_px = pd.to_numeric(summary_row.get("fwhm_px", float("nan")), errors="coerce")
    n_frames = pd.to_numeric(summary_row.get("n_frames", float("nan")), errors="coerce")
    n_good_comp = pd.to_numeric(summary_row.get("n_good_comp", float("nan")), errors="coerce")
    lc_rms = pd.to_numeric(summary_row.get("lc_rms", float("nan")), errors="coerce")
    if math.isfinite(float(ap_px)) and arcsec_per_px is not None and math.isfinite(float(arcsec_per_px)):
        v_lines.append(
            f"#   Aperture: {float(ap_px):.2f}px ({float(ap_px) * float(arcsec_per_px):.2f}arcsec)\n"
        )
    if math.isfinite(float(fwhm_px)):
        v_lines.append(f"#   FWHM: {float(fwhm_px):.2f}px\n")
    if math.isfinite(float(n_frames)) or math.isfinite(float(n_good_comp)) or math.isfinite(float(lc_rms)):
        v_lines.append(
            f"#   n_frames: {_fmt_opt_int(n_frames)} | "
            f"n_ensemble_comp: {_fmt_opt_int(n_good_comp)} (stability good+suspect; not comp_qa n_clean) | "
            f"lc_rms: {_fmt_opt_num(lc_rms, '.4f')}\n"
        )
    _trust = str(summary_row.get("trust", "") or "").strip().upper()
    _trust_reason = str(summary_row.get("trust_reason", "") or "").strip()
    if _trust:
        from trust_flag_core import format_varastro_trust_comment  # noqa: PLC0415

        v_lines.append(format_varastro_trust_comment(_trust, _trust_reason))
    v_lines.append("#\n")

    v_lines.append(f"# VAR Name: {vsx_name} | Type: {vsx_type}\n")
    v_lines.append(f"#   Catalog: GaiaDR3 | CatalogId: {target_cid or 'na'}\n")
    v_lines.append(
        f"#   CatalogMag: {_fmt_opt_num(t_mag, '.3f')} | "
        f"BP-RP: {_fmt_opt_num(t_bprp, '.3f')}\n"
    )
    v_lines.append("#\n")
    v_lines.append("# COMP TABLE:\n")
    v_lines.append(
        _format_varastro_comp_table(
            comp_export_df if _osc_export else comp_df,
            comp_quality_map=comp_quality_map,
            use_johnson_mag=_osc_export,
            post_weight_rel_map=_post_weight_rel,
        )
    )
    v_lines.append("#\n")
    if check_row is not None:
        if _osc_export:
            from gaia_johnson import transform_comp_row_for_osc_band

            chk_res = transform_comp_row_for_osc_band(check_row, aavso_filter, log_exclusions=False)
            chk_mag = float(chk_res.johnson_mag) if chk_res.ok else float("nan")
        else:
            chk_mag = pd.to_numeric(check_row.get("mag", float("nan")), errors="coerce")
        chk_p2p = pd.to_numeric(check_row.get("p2p_rms", float("nan")), errors="coerce")
        v_lines.append(
            f"# CHK CatalogId: {check_cid} | Mag: {_fmt_opt_num(chk_mag, '.3f')} | "
            f"p2p_rms: {_fmt_opt_num(chk_p2p, '.4f')}\n"
        )
    else:
        v_lines.append("# CHK CatalogId: na\n")
    v_lines.append("#\n")

    # Field image copy (if available).
    try:
        # output_dir points to .../photometry/lightcurves_reports
        # field maps live next to lightcurves, i.e. .../photometry/lightcurves/
        _lc_dir = out_base.parent / "lightcurves"
        field_img = _copy_field_image(_lc_dir, str(target_cid or ""), (out_base / "varastro"), str(safe), str(date_tag))
    except Exception:  # noqa: BLE001
        field_img = None
    if field_img:
        v_lines.append(f"# FIELD IMAGE: {field_img}\n")
    else:
        v_lines.append("# FIELD IMAGE: not available\n")
    try:
        _eem_v = str(getattr(fresh_cfg, "export_err_mode", "calibrated") or "calibrated")
        v_lines.append(f"# ERR_MODEL: mode={_eem_v} (see AAVSO #ERR_MODEL line)\n")
    except Exception:  # noqa: BLE001
        pass
    v_lines.append("#\n")

    v_lines.append("# BJD(TDB)       delta_mag  err    mag_calib\n")
    for _, row in lc_normal.iterrows():
        bjd = pd.to_numeric(row.get("bjd", float("nan")), errors="coerce")
        dmag = pd.to_numeric(row.get("delta_mag", float("nan")), errors="coerce")
        err = pd.to_numeric(row.get("err", float("nan")), errors="coerce")
        mag_cal = pd.to_numeric(row.get("mag_calib", float("nan")), errors="coerce")
        if not math.isfinite(float(bjd)):
            continue
        v_lines.append(f"{float(bjd):.6f}   {_fmt_opt_num(dmag, '.4f'):>7}   {_fmt_opt_num(err, '.4f'):>7}   {_fmt_opt_num(mag_cal, '.4f'):>7}\n")

    try:
        var_path.write_text("".join(v_lines), encoding="utf-8")
        logging.info("[EXPORT] VAR.ASTRO: %s", str(var_path.name))
        paths["varastro"] = var_path
    except OSError as exc:
        record_export_failure(
            export_failures,
            _target_id,
            _export_method,
            f"VarAstro write error: {exc}",
        )

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
    proc_csv_cache: dict[str, pd.DataFrame] | None = None,
    export_failures: list[ExportFailure] | None = None,
    export_stats: dict[str, int] | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Path]]:
    """Export AAVSO + VarAstro for each active photometry method."""
    fresh_cfg = cfg or AppConfig()
    _lc_dir = Path(lc_dir)
    _tid = str(target_cid or target_row.get("catalog_id", "") or "").strip()
    _have_psf_files = any(_lc_dir.glob("lightcurve_*_psf.csv")) or any(
        _lc_dir.glob("lightcurve_*_adaptive.csv")
    )
    _methods = active_report_methods(
        fresh_cfg,
        have_psf_cols=_have_psf_files
        or bool(getattr(fresh_cfg, "psf_photometry_enabled", False))
        or bool(getattr(fresh_cfg, "psf_adaptive_enabled", False)),
    )
    out: dict[str, dict[str, Path]] = {}
    _proc_cache = proc_csv_cache if proc_csv_cache is not None else {}
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
                active_methods=_methods,
                proc_csv_cache=_proc_cache,
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

