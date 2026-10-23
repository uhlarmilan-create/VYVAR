from __future__ import annotations

import glob
import json
import logging
import math
import re
import textwrap
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from citations import build_run_citation_context, emit_pdf_methods_sections
from target_page import TargetPages, _norm_cid_decimal, load_frames, star_fields
from report_methods import pdf_report_path

if TYPE_CHECKING:
    from reportlab.pdfgen import canvas

# Gaia ID must be str - float64 loses digits
_GAIA_ID_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}

# Time + airmass columns only (PDF cover airmass summary from lightcurve_*.csv).
_AIRMASS_COLS = {
    "bjd",
    "bjd_tdb_mid",
    "bjd_tdb",
    "hjd",
    "jd",
    "airmass",
    "air_mass",
    "AIRMASS",
    "am",
}


def _register_pdf_unicode_fonts() -> tuple[str, str, str]:
    """DejaVu Sans for Slovak / UTF-8; fallback to Helvetica if registration fails."""
    try:
        import reportlab

        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        fonts_dir = Path(reportlab.__file__).resolve().parent / "fonts"
        faces = [
            ("VYVARSans", "DejaVuSans.ttf"),
            ("VYVARSansBd", "DejaVuSans-Bold.ttf"),
            ("VYVARSansOb", "DejaVuSans-Oblique.ttf"),
        ]
        for reg_name, fname in faces:
            fp = fonts_dir / fname
            if not fp.is_file():
                raise FileNotFoundError(str(fp))
            pdfmetrics.registerFont(TTFont(reg_name, str(fp)))
        return faces[0][0], faces[1][0], faces[2][0]
    except Exception:  # noqa: BLE001
        H = "Hel"
        return (H + "vetica", H + "vetica-Bold", H + "vetica-Oblique")


def config_deviation_model(pipeline_meta: dict[str, Any] | None) -> dict[str, Any]:
    """Build the PDF Configuration-page model from a run provenance snapshot.

    Deviations only: keys whose run/snapshot value differs from the current dataclass
    default. Sourced from ``provenance.config_snapshot`` when present; otherwise falls
    back to the live AppConfig with a visible warning (legacy drafts). Unknown/legacy
    snapshot keys are collected separately (never crash the page).
    """
    import params_registry as pr  # noqa: PLC0415

    snapshot: dict[str, Any] | None = None
    prov: dict[str, Any] = {}
    meta = pipeline_meta if isinstance(pipeline_meta, dict) else {}
    raw_prov = meta.get("provenance")
    if isinstance(raw_prov, dict):
        prov = raw_prov
        raw = prov.get("config_snapshot")
        if isinstance(raw, dict) and raw:
            snapshot = raw

    if snapshot is not None:
        cfg_dict = dict(snapshot)
        source_label = "run snapshot"
        fallback = False
    else:
        try:
            from config import AppConfig

            cfg_dict = AppConfig().to_dict()
        except Exception:  # noqa: BLE001
            cfg_dict = {}
        source_label = "live (no run snapshot)"
        fallback = True
        prov = {}

    dev = pr.compute_deviations(cfg_dict)
    header = {
        "git_hash": prov.get("git_hash"),
        "git_dirty_code": prov.get("git_dirty_code"),
        "stamped_at_utc": prov.get("stamped_at_utc"),
        "entry_point": prov.get("entry_point"),
        "n_modified": len(dev["modified"]),
    }
    fingerprint = {
        "preprocess_sky_surface_order": cfg_dict.get("preprocess_sky_surface_order"),
        "seed_policy": prov.get("labbe_rng_seed_policy"),
        "density_profile": (meta.get("field_density") or {}).get("density_class")
        or (meta.get("dynamic_params") or {}).get("density_class"),
    }
    return {
        "source_label": source_label,
        "fallback": fallback,
        "rows": dev["modified"],
        "unknown_keys": dev["unknown"],
        "header": header,
        "fingerprint": fingerprint,
    }


def full_config_snapshot_model(pipeline_meta: dict[str, Any] | None) -> dict[str, Any]:
    """Full as-run config snapshot (every key) grouped by phase, for the report appendix.

    This is the honest "copy of the config file" taken from the run (``provenance.config_snapshot``),
    not from disk. Legacy drafts with no snapshot fall back to the live AppConfig with a visible
    warning; never crashes. Keys not present in the registry are grouped under ``other``.
    """
    import params_registry as pr  # noqa: PLC0415

    snapshot: dict[str, Any] | None = None
    meta = pipeline_meta if isinstance(pipeline_meta, dict) else {}
    raw_prov = meta.get("provenance")
    if isinstance(raw_prov, dict):
        raw = raw_prov.get("config_snapshot")
        if isinstance(raw, dict) and raw:
            snapshot = raw

    if snapshot is not None:
        cfg_dict = dict(snapshot)
        source_label = "run snapshot"
        fallback = False
    else:
        try:
            from config import AppConfig  # noqa: PLC0415

            cfg_dict = AppConfig().to_dict()
        except Exception:  # noqa: BLE001
            cfg_dict = {}
        source_label = "live (no run snapshot)"
        fallback = True

    try:
        registry = pr.load_registry()
    except Exception:  # noqa: BLE001
        registry = {}
    phase_of = {k: e.get("phase") for k, e in registry.items()}

    by_phase: dict[str, list[tuple[str, Any]]] = {}
    for key in sorted(cfg_dict.keys()):
        phase = phase_of.get(key) or "other"
        by_phase.setdefault(phase, []).append((key, cfg_dict[key]))

    ordered = [p for p in pr.PHASES if p in by_phase]
    if "other" in by_phase:
        ordered.append("other")

    reg_keys = set(registry)
    omitted_keys = sorted(reg_keys - set(cfg_dict))

    return {
        "source_label": source_label,
        "fallback": fallback,
        "n_keys": len(cfg_dict),
        "registry_count": len(reg_keys),
        "omitted_keys": omitted_keys,
        "phases": ordered,
        "by_phase": by_phase,
    }


def resolved_facts_model(
    pipeline_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run-effective resolved facts block for the report.

    Reports the values the run ACTUALLY used: site (id/name/lat/lon/alt + source), gain and
    read noise (+source), saturation, plate scale, frame dims, binning, filter, exposure.
    Sourced from ``pipeline_meta.resolved_facts`` (written by the aperture-photometry resolver capture),
    with graceful fallback to ``observer_location`` / ``dynamic_params`` for legacy drafts and a
    visible warning. Never crashes.
    """
    meta = pipeline_meta if isinstance(pipeline_meta, dict) else {}
    rf = meta.get("resolved_facts") if isinstance(meta.get("resolved_facts"), dict) else {}
    obs = meta.get("observer_location") if isinstance(meta.get("observer_location"), dict) else {}
    dyn = meta.get("dynamic_params") if isinstance(meta.get("dynamic_params"), dict) else {}

    warnings: list[str] = []
    fallback = not bool(rf)
    if fallback:
        warnings.append(
            "no resolved_facts block in provenance (legacy draft) - values below are recovered "
            "from observer_location / dynamic_params where available; some may be blank."
        )

    def _fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    def _src(res: dict[str, Any], extra: str | None = None) -> str:
        s = res.get("source") if isinstance(res, dict) else None
        key = res.get("key") if isinstance(res, dict) else None
        parts = [str(s)] if s else []
        if key:
            parts.append(f"[{key}]")
        if extra:
            parts.append(extra)
        return " ".join(parts) if parts else "-"

    site = rf.get("site") if isinstance(rf.get("site"), dict) else {}
    site_id = site.get("location_id", obs.get("location_id"))
    site_name = site.get("name", obs.get("name"))
    site_lat = site.get("lat", obs.get("lat"))
    site_lon = site.get("lon", obs.get("lon"))
    site_alt = site.get("alt_m", obs.get("alt_m"))
    site_source = site.get("source") or obs.get("source") or "-"

    gain = rf.get("gain") if isinstance(rf.get("gain"), dict) else {}
    rn = rf.get("read_noise") if isinstance(rf.get("read_noise"), dict) else {}
    gain_val = gain.get("value", dyn.get("gain"))
    rn_val = rn.get("value", dyn.get("read_noise"))
    plate = rf.get("plate_scale_arcsec_per_px", dyn.get("plate_scale_arcsec_px"))
    _fd = meta.get("frame_depth") if isinstance(meta.get("frame_depth"), dict) else {}

    rows: list[dict[str, str]] = [
        {
            "label": "Site (LOCATION)",
            "value": f"id={_fmt(site_id)} {_fmt(site_name)} | lat={_fmt(site_lat)} lon={_fmt(site_lon)} alt={_fmt(site_alt)} m",
            "source": str(site_source),
        },
        {"label": "Gain (e-/ADU)", "value": _fmt(gain_val), "source": _src(gain) if gain else "-"},
        {"label": "Read noise (e-)", "value": _fmt(rn_val), "source": _src(rn) if rn else "-"},
        {"label": "Saturation (ADU)", "value": _fmt(rf.get("saturation_adu")), "source": "-"},
        {"label": "Plate scale (arcsec/px)", "value": _fmt(plate), "source": "-"},
        {
            "label": "Frame (px)",
            "value": f"{_fmt(rf.get('frame_width_px'))} x {_fmt(rf.get('frame_height_px'))}",
            "source": "-",
        },
        {"label": "Binning", "value": _fmt(rf.get("binning")), "source": "-"},
        {"label": "Filter", "value": _fmt(rf.get("filter")), "source": "FITS / obs_group"},
        {"label": "Exposure (s)", "value": _fmt(rf.get("exptime_s")), "source": "FITS"},
    ]

    # VSX auto-target scope (report-only): detection-limited adoption.
    if _fd:
        # FRAME-DEPTH-2026-10: faint limit measured on the MASTERSTAR and the Gaia depth used.
        rows.extend([
            {
                "label": "Frame faint limit (Gaia G)",
                "value": _fmt(_fd.get("g_lim_frame")),
                "source": (
                    f"MASTERSTAR {_fd.get('method')}: G50={_fmt(_fd.get('g50'))}, sigma={_fmt(_fd.get('sigma_mag'))}"
                ),
            },
            {
                "label": "Gaia matching depth (G)",
                "value": _fmt(_fd.get("g_lim")),
                "source": (
                    f"catalogue depth G={_fmt(_fd.get('db_depth_g'))}"
                    + (" - LIMITED BY THE CATALOGUE" if _fd.get("limited_by_catalogue") else "")
                ),
            },
        ])
        _tu = meta.get("gaia_topup") if isinstance(meta.get("gaia_topup"), dict) else {}
        _tu_status = str(_tu.get("status") or "")
        if _tu_status and _tu_status not in ("not_needed", "covered"):
            if _tu_status == "ok":
                _tu_val = (
                    f"fetched online to G {_fmt(_tu.get('g_to'))}: {_fmt(_tu.get('n_new'))} new stars"
                )
            elif _tu_status == "no_coverage":
                _tu_val = "field outside the Dec range of the main Gaia DB - not extended"
            else:
                _tu_val = (
                    f"NOT POSSIBLE ({_fmt(_tu.get('error'))}) - matching limited to the local DB "
                    f"G {_fmt(_tu.get('depth_before'))}"
                )
            rows.append({"label": "Gaia DB top-up (field)", "value": _tu_val, "source": "gaia_coverage"})
    rows.append(
        {
            "label": "VSX auto-target scope",
            "value": "detection-limited (DAO+Gaia match)",
            "source": "pipeline policy",
        }
    )

    if meta.get("n_dao_unmatched") is not None or meta.get("dao_only_n_artifact_negative") is not None:
        n_dao = int(meta.get("n_dao_unmatched") or 0)
        depth = meta.get("confirmable_depth_g")
        winner = meta.get("confirmable_depth_winner") or "?"
        depth_s = f"{float(depth):.2f}" if depth is not None else "unresolved"
        parts = []
        for k, short in (
            ("dao_only_n_artifact_negative", "artifact_negative"),
            ("dao_only_n_unmatched_in_range", "unmatched_in_range"),
            ("dao_only_n_ambiguous_depth", "ambiguous_depth"),
            ("dao_only_n_beyond_catalogue", "beyond_catalogue"),
            ("dao_only_n_indeterminate", "indeterminate"),
        ):
            if meta.get(k) is not None:
                parts.append(f"{short}={int(meta[k])}")
        val = (
            f"{n_dao} total [{', '.join(parts)}] | confirmable_depth G={depth_s} (from {winner}); "
            "counts installation-specific"
        )
        sg = meta.get("dao_only_sigma_g_row_median")
        if sg is not None:
            val += f" | sigma_g median={float(sg):.3f} mag"
        rms = meta.get("dao_only_flux_fit_residual_rms")
        if rms is not None and math.isfinite(float(rms)):
            val += f" | flux-to-G RMS={float(rms):.3f} mag"
        un_frac = meta.get("dao_only_sigma_g_unmeasurable_fraction")
        if un_frac is not None:
            thr = meta.get("dao_only_sigma_g_unmeasurable_threshold_mag", 1.0)
            val += f" | unmeasurable(sigma_g>{float(thr):.1f})={float(un_frac):.3f}"
        rows.append(
            {
                "label": "DAO_ONLY census (informational)",
                "value": val,
                "source": "masterstars magnitude class",
            }
        )

    return {
        "fallback": fallback,
        "warnings": warnings,
        "rows": rows,
        "vsx_depth": {"status": "n/a", "warn": False, "line": "detection-limited (DAO+Gaia match)"},
    }


def gs11_report_lines(pipeline_meta: dict[str, Any] | None, cfg: Any) -> list[str]:
    """Text lines for PDF / tests - Flux Dilution (Gaia) subsection."""
    enabled = bool(getattr(cfg, "gs11_dilution_enabled", False)) if cfg is not None else False
    gs11: dict[str, Any] = {}
    if isinstance(pipeline_meta, dict):
        raw = pipeline_meta.get("gs11_summary")
        if isinstance(raw, dict):
            gs11 = raw
    if not enabled and not bool(gs11.get("enabled", False)):
        return ["  Flux dilution correction: disabled"]
    if not enabled:
        return ["  Flux dilution correction: disabled"]
    ap = float(gs11.get("aperture_arcsec", float("nan")))
    ap_s = f"{ap:.1f} arcsec" if math.isfinite(ap) else "-"
    try:
        med = float(gs11.get("median_correction_mmag", 0.0))
    except (TypeError, ValueError):
        med = float("nan")
    med_s = f"{med:.1f} mmag" if math.isfinite(med) else "-"
    return [
        "  Flux Dilution Assessment (Gaia DR3)",
        "  Status:                 enabled",
        f"  Aperture used:          {ap_s}",
        f"  Targets corrected:      {int(gs11.get('targets_corrected', 0) or 0)}  "
        f"(median Deltam = {med_s})",
        f"  Targets skipped (D<0.50): {int(gs11.get('targets_skipped_low_d', 0) or 0)}",
    ]


def _publication_lc_mag_column(columns: Any) -> str | None:
    """Y-axis / export magnitude column for publication-facing LC plots and exports."""
    from photometry_lightcurve import LC_PUBLISHED_MAG_COLUMNS  # noqa: PLC0415

    cols = set(columns)
    return next((c for c in LC_PUBLISHED_MAG_COLUMNS if c in cols), None)


class _PhotometryReportBuilder:
    """Internal PDF builder; section renderers extracted from generate_photometry_report."""

    _norm_cid_decimal = staticmethod(_norm_cid_decimal)

    def __init__(
        self,
        draft_dir: Path,
        obs_group: str,
        output_pdf: Path | None,
        var_results: dict[str, Any] | None,
        candidates: list[str] | None,
        crossmatch_bullets: dict[str, str] | None,
        accepted_periods: dict[str, float] | None,
        variability_timestamp: str | None,
        report_draft_label: str | None,
        tess_results: dict | None,
        report_title: str,
        font_reg: str,
        font_bold: str,
        font_obl: str,
        colors_mod: Any,
        cm_mod: Any,
        mm_mod: Any,
        landscape_fn: Any,
        a4_size: Any,
        canvas_mod: Any,
        image_reader_mod: Any,
        table_mod: Any,
        table_style_mod: Any,
        paragraph_mod: Any,
        paragraph_style_mod: Any,
        ta_left_mod: Any,
    ) -> None:

        self._colors = colors_mod
        self.cm = cm_mod
        self.mm = mm_mod
        self.landscape = landscape_fn
        self.A4 = a4_size
        self.canvas = canvas_mod
        self.ImageReader = image_reader_mod
        self.Table = table_mod
        self.TableStyle = table_style_mod
        self.Paragraph = paragraph_mod
        self.ParagraphStyle = paragraph_style_mod
        self.TA_LEFT = ta_left_mod
        self.draft_dir = Path(draft_dir)
        self.obs_group = str(obs_group)
        self.output_pdf = Path(output_pdf) if output_pdf is not None else None
        self.colors = self._colors
        self.FONT_REG = font_reg
        self.FONT_BOLD = font_bold
        self.FONT_OBL = font_obl
        self.candidates = candidates
        self._var_results = var_results
        self._crossmatch_bullets = dict(crossmatch_bullets or {})
        self._accepted_periods = dict(accepted_periods or {})
        self._variability_ts = str(variability_timestamp or '').strip()
        self._report_draft_lbl = str(report_draft_label or '').strip() or str(Path(draft_dir).name)
        self._tess_results = dict(tess_results or {})
        self._report_title = str(report_title or 'VYVAR \u2014 Summary Measure Report')
        self._candidates_set = {str(x).strip() for x in (candidates or []) if str(x).strip()}
        self.platesolve_dir = self.draft_dir / "platesolve" / self.obs_group
        self.photometry_dir = self.platesolve_dir / "photometry"
        self.lc_dir = self.photometry_dir / "lightcurves"
        self.cache_dir = self.photometry_dir / "_report_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # AP-TAB-2026-10: tables prepared once in target_page (same data as the UI tab).
        _fr = load_frames(self.platesolve_dir)
        self._pipeline_meta = _fr["pipeline_meta"]
        if "calibration_mode" not in (self._pipeline_meta or {}):
            try:
                from draft_provenance import resolve_calibration_mode

                _mode = resolve_calibration_mode(archive_path=self.draft_dir)
                if _mode:
                    self._pipeline_meta = dict(self._pipeline_meta or {})
                    self._pipeline_meta["calibration_mode"] = _mode
            except Exception:  # noqa: BLE001
                # EXC-0226: T2 -- report/export may omit or misstate (calibration mode line)
                pass
        self.summary_df = _fr["summary_df"]
        self.comp_df = _fr["comp_df"]
        self.at_df = _fr["at_df"]
        self._cfg, self._cfg_source_label = _fr["cfg"], _fr["cfg_label"]
        self._candidates_norm = {_norm_cid_decimal(str(x)) for x in self._candidates_set if str(x).strip()}
        self._pages = TargetPages(self.platesolve_dir, summary_df=self.summary_df, at_df=self.at_df,
                                  comp_df=self.comp_df, cfg=self._cfg)
        self._use_bprp_primary = True

        if (self._var_results is None) or (not self._candidates_set):
            vr_csv, cnorm_csv = self._try_load_variability_from_csv()
            if vr_csv is not None and self._var_results is None:
                self._var_results = vr_csv
            if cnorm_csv and not self._candidates_set:
                self._candidates_set = set(str(x).strip() for x in cnorm_csv if str(x).strip())
            if self._candidates_set:
                self._candidates_norm = {_norm_cid_decimal(str(x)) for x in self._candidates_set if str(x).strip()}

        self.obs_date_human = self._obs_date_str()
        self.date_token = datetime.today().strftime("%Y%m%d")
        try:
            # If obs_date_human is in YYYY-MM-DD, take it.
            if "-" in self.obs_date_human and len(self.obs_date_human) >= 10:
                self.date_token = self.obs_date_human.split("T", 1)[0].replace("-", "")
        except Exception:  # noqa: BLE001
            # EXC-0228: T2 -- report/export may omit or misstate (if '-' in self.obs_date_human and len(self.obs_date... (EXCEPT-BULK 2026-07-08)
            pass

        if self.output_pdf is None:
            self.output_pdf = pdf_report_path(self.draft_dir, self.obs_group, date_str=self.date_token)
        self.output_pdf = Path(self.output_pdf)
        self.output_pdf.parent.mkdir(parents=True, exist_ok=True)

        # Styles / self.colors
        self.C_TITLE = self.colors.HexColor("#1a1a2e")
        self.C_GOOD = self.colors.HexColor("#2ecc71")
        self.C_MID = self.colors.HexColor("#f39c12")
        self.C_BAD = self.colors.HexColor("#e74c3c")

        # Aggregate metrics (cover page): measured targets only (FIX-NLC-2026-10: rows of skipped /
        # no-comparison targets have n_frames 0 and no light curve)
        _measured = (pd.to_numeric(self.summary_df["n_frames"], errors="coerce").fillna(0) > 0
                     if "n_frames" in self.summary_df.columns
                     else pd.Series(True, index=self.summary_df.index))
        self.measured_summary_df = self.summary_df[_measured]
        self.n_lc = int(len(self.measured_summary_df))
        self.lc_count_display = self._format_lc_count_display()
        self.med_rms = float(np.nanmedian(pd.to_numeric(self.summary_df.get("lc_rms"), errors="coerce"))) if self.n_lc else float("nan")
        self.rms_lt_005 = int((pd.to_numeric(self.summary_df.get("lc_rms"), errors="coerce") < 0.05).sum()) if self.n_lc else 0
        self.avg_good_comp = float(np.nanmean(pd.to_numeric(self.measured_summary_df.get("n_good_comp"), errors="coerce"))) if self.n_lc else float("nan")
        self.best_rms = float(np.nanmin(pd.to_numeric(self.summary_df.get("lc_rms"), errors="coerce"))) if self.n_lc else float("nan")
        self.worst_rms = float(np.nanmax(pd.to_numeric(self.summary_df.get("lc_rms"), errors="coerce"))) if self.n_lc else float("nan")
        self.avg_bp_rp = float(np.nanmean(pd.to_numeric(self.summary_df.get("bp_rp"), errors="coerce"))) if self.n_lc else float("nan")
        self.setups = 1
        self.fwhm_px = float("nan")
        self.fwhm_label = "FWHM (MASTERSTAR)"
        # REPORT-AUDIT-2026-10: the cover shows the FWHM the apertures were scaled to (night median
        # of the measured frames, aperture_per_target.json); MASTERSTAR (a single best frame) only
        # when that file is missing.
        try:
            _apj = json.loads((self.photometry_dir / "aperture_per_target.json").read_text(encoding="utf-8"))
            _fn = float(_apj.get("fwhm_night_px", float("nan")))
            if np.isfinite(_fn) and _fn > 0:
                self.fwhm_px, self.fwhm_label = _fn, "FWHM (night median)"
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        if not np.isfinite(self.fwhm_px) and (self.platesolve_dir / "MASTERSTAR.fits").exists():
            try:
                from platesolve_ui_paths import masterstar_fwhm_px  # noqa: PLC0415

                self.fwhm_px = float(masterstar_fwhm_px(self.platesolve_dir / "MASTERSTAR.fits"))
            except Exception:  # noqa: BLE001
                # EXC-0229: T2 -- report/export may omit or misstate (self.fwhm_px = float(_load_fwhm(self.platesolve_dir... (EXCEPT-BULK 2026-07-08)
                pass
        self.aperture_px = float(np.nanmedian(pd.to_numeric(self.summary_df.get("aperture_px"), errors="coerce"))) if self.n_lc else float("nan")

        self.aavso_dir = self.photometry_dir / "lightcurves_reports" / "aavso"
        self.varastro_dir = self.photometry_dir / "lightcurves_reports" / "varastro"
        self.n_aavso = len(list(self.aavso_dir.glob("*.txt"))) if self.aavso_dir.is_dir() else 0
        self.n_varastro = len(list(self.varastro_dir.glob("*.txt"))) if self.varastro_dir.is_dir() else 0
        # REPORT-AUDIT-2026-10: RED trust targets are exported to <sub>/_dontsubmit (not for submission)
        from export_reports import DONTSUBMIT_DIRNAME  # noqa: PLC0415

        self.n_aavso_ds = len(list((self.aavso_dir / DONTSUBMIT_DIRNAME).glob("*.txt")))
        self.n_varastro_ds = len(list((self.varastro_dir / DONTSUBMIT_DIRNAME).glob("*.txt")))

        self.comp_pool_cover_rows = self._build_comp_pool_cover_rows()
        self._var_cand_by_cid = self._load_variability_candidates_by_cid()
        self._lc_stats_by_cid = self._pages.lc_stats_by_cid
        self._night_qc = self._build_night_qc_summary()

        # ---------------------------------------------------------------------
        # Canvas-based layout (precise positioning; one star per page)
        # ---------------------------------------------------------------------

        self.PAGE_W, self.PAGE_H = self.landscape(self.A4)
        self.M_LEFT = 1.0 * self.cm
        self.M_RIGHT = 1.0 * self.cm
        self.M_TOP = 1.0 * self.cm
        self.M_BOTTOM = 0.8 * self.cm
        self.USE_W = self.PAGE_W - self.M_LEFT - self.M_RIGHT
        self.USE_H = self.PAGE_H - self.M_TOP - self.M_BOTTOM

        # Overflow verify mode (layout QA; set via generate_photometry_report).
        self._verify_overflow = False
        self._overflow_violations: list[str] = []
        self._para_styles: dict[str, Any] = {}

        # Star page geometry (self.cm -> pt)
        self.TITLE_H = 0.8 * self.cm
        self.METRICS_H = 0.5 * self.cm
        self.SEP_H = 0.1 * self.cm
        # GRAPH_H is dynamic per star (see draw_star_page)
        self.LC_W = 16.5 * self.cm
        self.GAP_W = 0.5 * self.cm
        self.FI_W = self.USE_W - self.LC_W - self.GAP_W

        # Colour is raw Gaia BP-RP (catalog); B-V transform is not applied (deprecated).
        self.NOTE_TXT = (
            "Gaia BP-RP colour | COMP weights: w = 1/sigma^2 - Broeg et al., Astron. Nachr. 326, 134 (2005)"
        )

        # Per Phase 4: max pixel width (long edge cap for PDF embed) + JPEG quality
        self._IMAGE_PDF_SETTINGS: dict[str, tuple[int, int]] = {
            "default": (1200, 72),
            "lc": (1000, 75),
            "field": (900, 72),
            "hockey": (1400, 78),
            "field_map": (1800, 80),
            "tess_phased": (900, 72),
            "tess_blend": (1200, 72),
            "logo": (400, 85),
            "hrd": (1000, 75),
            "hrd_field": (900, 72),
        }

        # ---------------------------------------------------------------------
        # QA page (FWHM + sky + masterstar candidate table)
        # ---------------------------------------------------------------------
        self.bullets_by_cid = {_norm_cid_decimal(str(k)): str(v) for k, v in self._crossmatch_bullets.items()}

    def _try_load_variability_from_csv(self, ) -> tuple[dict[str, Any] | None, set[str]]:
        """Fallback when UI did not pass var_results/candidates: read exported pipeline/UI CSV."""
        from variability_field import load_variability_candidates  # noqa: PLC0415

        try:
            vdf = load_variability_candidates(self.photometry_dir)
        except Exception:  # noqa: BLE001
            # EXC-0230: T2 -- report/export may omit or misstate (EXCEPT-BULK 2026-07-08)
            return None, set()
        if vdf.empty or "catalog_id" not in vdf.columns:
            return None, set()
        cands_raw = {str(x).strip() for x in vdf["catalog_id"].tolist() if str(x).strip()}
        rms_like = vdf.copy()
        try:
            from gaia_catalog_id import normalize_gaia_source_id_series

            rms_like["catalog_id"] = normalize_gaia_source_id_series(rms_like["catalog_id"])
        except Exception:  # noqa: BLE001
            rms_like["catalog_id"] = rms_like["catalog_id"].map(_norm_cid_decimal)
        vsx_known: set[str] = set()
        if "_cid" in self.summary_df.columns and "vsx_name" in self.summary_df.columns:
            m = self.summary_df["vsx_name"].fillna("").astype(str).str.strip().ne("")
            vsx_known = set(self.summary_df.loc[m, "_cid"].astype(str).map(_norm_cid_decimal))
        rms_like["vsx_known_variable"] = rms_like["catalog_id"].astype(str).map(_norm_cid_decimal).isin(vsx_known)
        rms_like["edge_ok"] = True
        if "edge_ok" in rms_like.columns:
            try:
                rms_like["edge_ok"] = pd.to_numeric(rms_like["edge_ok"], errors="coerce").fillna(1).astype(bool)
            except Exception:  # noqa: BLE001
                rms_like["edge_ok"] = True
        dm = (
            rms_like["detection_method"].fillna("RMS").astype(str)
            if "detection_method" in rms_like.columns
            else pd.Series(["RMS+VDI"] * len(rms_like), index=rms_like.index)
        )
        cr_rms = dm.str.contains("RMS", case=False, na=False)
        if not bool(cr_rms.any()):
            cr_rms = pd.Series(True, index=rms_like.index)
        cr_vdi = dm.str.contains("VDI", case=False, na=False)
        rms_like["is_variable_candidate"] = cr_rms
        if "vdi_score" not in rms_like.columns:
            rms_like["vdi_score"] = np.nan
        if "vdi_z_score" not in rms_like.columns:
            rms_like["vdi_z_score"] = np.nan
        vdi_df = rms_like[["catalog_id", "vdi_score", "vdi_z_score"]].copy()
        vdi_df["is_variable_candidate"] = cr_vdi.astype(bool).values
        return {"rms_df": rms_like, "vdi_df": vdi_df}, {_norm_cid_decimal(str(x)) for x in cands_raw}
    def _lc_files(self) -> list[Path]:
        """This run's aperture light curves (lightcurves/lightcurve_<id>.csv). The summary's
        absolute lc_csv paths break when a draft is moved (REPORT-AUDIT-2026-10)."""
        if not self.lc_dir.is_dir():
            return []
        return [p for p in sorted(self.lc_dir.glob("lightcurve_*.csv")) if not p.name.endswith("_psf.csv")]

    def _measured_jd_range(self) -> tuple[float, float]:
        """(first, last) UTC JD of the epochs measured in this run (the light curves)."""
        if not hasattr(self, "_jd_range_cache"):
            lo, hi = float("inf"), float("-inf")
            for p in self._lc_files():
                try:
                    jd = pd.to_numeric(pd.read_csv(p, usecols=["jd"])["jd"], errors="coerce").dropna()
                except (OSError, ValueError):
                    continue
                if len(jd):
                    lo, hi = min(lo, float(jd.min())), max(hi, float(jd.max()))
            self._jd_range_cache = (lo, hi) if math.isfinite(lo) else (float("nan"), float("nan"))
        return self._jd_range_cache

    def _obs_date_str(self, ) -> str:
        """REPORT-AUDIT-2026-10: the UTC date of the first measured epoch (as the export file names);
        the MASTERSTAR DATE-OBS (best frame, can be after midnight) only when no light curve exists."""
        jd0, _ = self._measured_jd_range()
        if math.isfinite(jd0):
            from astropy.time import Time  # noqa: PLC0415

            return Time(jd0, format="jd", scale="utc").iso[:10].replace("-", ".")
        ms_fits = self.platesolve_dir / "MASTERSTAR.fits"
        if ms_fits.exists():
            try:
                from astropy.io import fits

                with fits.open(ms_fits, memmap=False) as hdul:
                    hdr = hdul[0].header
                for key in ("DATE-OBS", "DATEOBS", "DATE"):
                    v = str(hdr.get(key, "") or "").strip()
                    if v:
                        return v.split("T", 1)[0].replace("-", ".")
            except Exception:  # noqa: BLE001
                # EXC-0231: T2 -- report/export may omit or misstate (# Keep only date part. / return v.split('T', 1)[0].... (EXCEPT-BULK 2026-07-08)
                pass
        return datetime.today().strftime("%d.%m.%Y")
    def _metric_color(self, v: float) -> Any:
        if not np.isfinite(v):
            return self.colors.black
        if v < 0.05:
            return self.C_GOOD
        if v < 0.1:
            return self.C_MID
        return self.C_BAD

    def _format_lc_count_display(self) -> str:
        """Light-curve count line for summary metrics (total + per-flag breakdown)."""
        n = int(self.n_lc)
        if "lc_quality_flag" not in self.summary_df.columns:
            return f"{n:d}"
        vc = self.measured_summary_df["lc_quality_flag"].astype(str).str.strip().str.lower().value_counts()
        parts: list[str] = []
        for key, label in (
            ("good", "good"),
            ("noisy", "noisy"),
            ("noisy_moon", "noisy_moon"),
            ("short_baseline", "short_baseline"),
            ("no_data", "no_data"),
            ("saturated", "saturated"),
        ):
            c = int(vc.get(key, 0))
            if c > 0:
                parts.append(f"{c} {label}")
        # LC-OUTLIER-01: epoch flag class totals of this run's light curves (REPORT-FLAGS-2026-10:
        # read self.output_dir, which does not exist - the error was swallowed, no count was shown)
        from collections import Counter  # noqa: PLC0415

        epoch_flags: Counter[str] = Counter()
        for p in self._lc_files():
            try:
                fl = pd.read_csv(p, usecols=["flag"])["flag"]
            except (OSError, ValueError, KeyError):
                continue
            epoch_flags.update(fl.astype(str).str.strip().str.lower().tolist())
        for key in ("artifact", "frame_qc", "high_err", "spike_unconfirmed", "saturated"):
            c = int(epoch_flags.get(key, 0))
            if c > 0:
                parts.append(f"{c} {key} epochs")
        if parts:
            return f"{n:d} total  ({', '.join(parts)})"
        return f"{n:d}"

    def _draw_observing_conditions_section(self, c: "canvas.Canvas", y: float) -> float:
        """Observer site (config) + lunar context (pipeline_meta); skip if meta file missing."""
        if self._pipeline_meta is None:
            return y

        y -= 0.45 * self.cm
        c.setFont(self.FONT_BOLD, 12)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y, "Observing Conditions")
        c.setFillColor(self.colors.black)
        y -= 0.55 * self.cm

        c.setFont(self.FONT_BOLD, 10)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y, "Observer Location")
        c.setFillColor(self.colors.black)
        y -= 0.45 * self.cm
        c.setFont(self.FONT_REG, 9)
        _cfg_o = self._cfg
        # Prefer the per-draft resolved observer site (param_resolver, persisted by
        # aperture photometry in pipeline_meta) so the report matches the BJD/airmass location.
        _meta_loc = self._pipeline_meta.get("observer_location") if isinstance(self._pipeline_meta, dict) else None
        if isinstance(_meta_loc, dict) and _meta_loc.get("lat") is not None:
            _loc_name = str(_meta_loc.get("name", "") or "").strip()
            _lat = float(_meta_loc.get("lat", 0.0) or 0.0)
            _lon = float(_meta_loc.get("lon", 0.0) or 0.0)
            _alt = float(_meta_loc.get("alt_m", 0.0) or 0.0)
        else:
            _loc_name = str(getattr(_cfg_o, "observer_location_name", "") or "").strip() if _cfg_o else ""
            _lat = float(getattr(_cfg_o, "observer_lat", 0.0) or 0.0) if _cfg_o else 0.0
            _lon = float(getattr(_cfg_o, "observer_lon", 0.0) or 0.0) if _cfg_o else 0.0
            _alt = float(getattr(_cfg_o, "observer_alt_m", 0.0) or 0.0) if _cfg_o else 0.0
        _has_coords = math.isfinite(_lat) and math.isfinite(_lon) and (_lat != 0.0 or _lon != 0.0)
        if not _loc_name and not _has_coords:
            y = self._draw_flow_lines(c, y, [("Location: not configured", self.FONT_REG, 9.0)], paginate=True)
        else:
            if _loc_name:
                y = self._draw_flow_lines(c, y, [(f"Site: {_loc_name}", self.FONT_REG, 9.0)], paginate=True)
            if _has_coords:
                y = self._draw_flow_lines(
                    c,
                    y,
                    [
                        (f"Latitude:   {_lat:.4f}\u00b0N", self.FONT_REG, 9.0),
                        (f"Longitude:  {_lon:.4f}\u00b0E", self.FONT_REG, 9.0),
                        (f"Altitude:   {_alt:.0f} m", self.FONT_REG, 9.0),
                    ],
                    paginate=True,
                )
            elif not _loc_name:
                y = self._draw_flow_lines(c, y, [("Location: not configured", self.FONT_REG, 9.0)], paginate=True)

        y -= 0.55 * self.cm  # flowed lines end at their bottom edge (REPORT-AUDIT-2026-10)
        c.setFont(self.FONT_BOLD, 10)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y, "Lunar Conditions")
        c.setFillColor(self.colors.black)
        y -= 0.45 * self.cm
        c.setFont(self.FONT_REG, 9)
        lunar = self._pipeline_meta.get("lunar_context")
        if not isinstance(lunar, dict):
            y = self._draw_flow_lines(
                c,
                y,
                [("Lunar conditions: not available (observer location not configured)", self.FONT_REG, 9.0)],
                paginate=True,
            )
        else:
            try:
                phase = float(lunar.get("lunar_phase_pct", float("nan")))
            except (TypeError, ValueError):
                phase = float("nan")
            try:
                age = float(lunar.get("lunar_age_days", float("nan")))
            except (TypeError, ValueError):
                age = float("nan")
            try:
                sep = float(lunar.get("lunar_separation_deg", float("nan")))
            except (TypeError, ValueError):
                sep = float("nan")
            try:
                alt_moon = float(lunar.get("lunar_altitude_deg", float("nan")))
            except (TypeError, ValueError):
                alt_moon = float("nan")
            risk = str(lunar.get("lunar_risk", "") or "").strip().upper() or "-"
            reason = str(lunar.get("lunar_risk_reason", "") or "").strip()

            lunar_lines: list[tuple[str, str, float]] = []
            if math.isfinite(phase) and math.isfinite(age):
                lunar_lines.append(
                    (f"Lunar phase:       {phase:.1f}%  ({age:.1f} days since new moon)", self.FONT_REG, 9.0)
                )
            else:
                lunar_lines.append(("Lunar phase:       -", self.FONT_REG, 9.0))
            lunar_lines.append(
                (
                    f"Lunar separation:  {sep:.1f}\u00b0 from field center"
                    if math.isfinite(sep)
                    else "Lunar separation:  -",
                    self.FONT_REG,
                    9.0,
                )
            )
            lunar_lines.append(
                (
                    f"Lunar altitude:    {alt_moon:.1f}\u00b0 at session midpoint"
                    if math.isfinite(alt_moon)
                    else "Lunar altitude:    -",
                    self.FONT_REG,
                    9.0,
                )
            )
            lunar_lines.append((f"Contamination risk: {risk}", self.FONT_REG, 9.0))
            y = self._draw_flow_lines(c, y, lunar_lines, paginate=True)
            if reason:
                reason_style = self._get_para_style("lunar_reason", fontName=self.FONT_OBL, fontSize=7.5)
                y = self._draw_paragraph_block(
                    c,
                    self.M_LEFT + 0.15 * self.cm,
                    y,
                    self.USE_W - 0.15 * self.cm,
                    self._pdf_escape(f'"{reason}"'),
                    reason_style,
                    paginate=True,
                )

        y -= 0.55 * self.cm  # flowed lines end at their bottom edge (REPORT-AUDIT-2026-10)
        c.setFont(self.FONT_BOLD, 10)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y, "LC Quality Assessment")
        c.setFillColor(self.colors.black)
        y -= 0.45 * self.cm
        c.setFont(self.FONT_REG, 9)
        qs = self._pipeline_meta.get("lc_quality_summary")
        if not isinstance(qs, dict) or not qs.get("available", True):
            if "lc_quality_flag" in self.summary_df.columns:
                vc = (
                    self.summary_df["lc_quality_flag"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .value_counts()
                )
                total = int(len(self.summary_df))
                good = int(vc.get("good", 0))
                pct = 100.0 * good / total if total else 0.0
                lines = [
                    f"  Good LCs:        {good}  ({pct:.1f}%)",
                    f"  Noisy:           {int(vc.get('noisy', 0))}",
                    f"  Noisy (moon):    {int(vc.get('noisy_moon', 0))}",
                    f"  Short baseline:  {int(vc.get('short_baseline', 0))}",
                    f"  No data:         {int(vc.get('no_data', 0))}",
                    f"  Saturated:       {int(vc.get('saturated', 0))}",
                ]
            else:
                lines = ["  LC quality: not available (re-run aperture photometry)"]
        else:
            total = int(qs.get("total", 0) or 0)
            good = int(qs.get("good", 0) or 0)
            pct = 100.0 * good / total if total else 0.0
            lines = [
                f"  Good LCs:        {good}  ({pct:.1f}%)",
                f"  Noisy:           {int(qs.get('noisy', 0) or 0)}",
                f"  Noisy (moon):    {int(qs.get('noisy_moon', 0) or 0)}",
                f"  Short baseline:  {int(qs.get('short_baseline', 0) or 0)}",
                f"  No data:         {int(qs.get('no_data', 0) or 0)}",
                f"  Saturated:       {int(qs.get('saturated', 0) or 0)}",
            ]
            try:
                slope = float(qs.get("rms_model_slope", float("nan")))
                intercept = float(qs.get("rms_model_intercept", float("nan")))
                n_cal = int(qs.get("rms_model_n_stars", 0) or 0)
            except (TypeError, ValueError):
                slope = intercept = float("nan")
                n_cal = 0
            if math.isfinite(slope) and math.isfinite(intercept):
                cal_txt = f"  ({n_cal} calibration stars)" if n_cal > 0 else ""
                lines.append(
                    f"  RMS model:       slope={slope:.3f}, intercept={intercept:.3f}{cal_txt}"
                )
        for ln in lines:
            y = self._draw_flow_lines(c, y, [(ln, self.FONT_REG, 9.0)], paginate=True)

        y = self._draw_gs11_dilution_section(c, y)

        y -= 0.25 * self.cm
        return y

    def _draw_gs11_dilution_section(self, c: "canvas.Canvas", y: float) -> float:
        """Flux dilution (GS11) block from pipeline_meta.gs11_summary."""
        y -= 0.55 * self.cm  # flowed lines end at their bottom edge (REPORT-AUDIT-2026-10)
        c.setFont(self.FONT_BOLD, 10)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y, "Flux Dilution (Gaia)")
        c.setFillColor(self.colors.black)
        y -= 0.45 * self.cm
        c.setFont(self.FONT_REG, 9)
        for ln in gs11_report_lines(self._pipeline_meta, self._cfg):
            y = self._draw_flow_lines(c, y, [(ln, self.FONT_REG, 9.0)], paginate=True)
        return y

    def _build_comp_pool_cover_rows(self, ) -> list[tuple[str, str]]:
        rows_out: list[tuple[str, str]] = []
        if getattr(self.comp_df, "empty", True):
            rows_out.append(("Comparison pool", "- (no comparison_stars_per_target.csv)"))
            return rows_out
        if "catalog_id" not in self.comp_df.columns:
            rows_out.append(("Comparison pool", "-"))
            return rows_out
        # REPORT-AUDIT-2026-10: comp_df holds the comparison stars the ensembles used (reserves
        # dropped on load). The tier is relative to each target's colour, so it is counted per
        # target ensemble (selected_tier), not per star.
        cdf = self.comp_df
        ids = cdf["catalog_id"].map(_norm_cid_decimal).astype(str)
        rms = pd.to_numeric(cdf.get("comp_rms"), errors="coerce")
        magv = pd.to_numeric(cdf.get("mag"), errors="coerce")
        gdf = pd.DataFrame({"_id": ids, "comp_rms": rms, "mag": magv})
        gdf = gdf[gdf["_id"].astype(str).str.len() > 0].copy()
        if gdf.empty:
            rows_out.append(("Comparison pool", "-"))
            return rows_out
        per_id_rms_med = gdf.groupby("_id", sort=False)["comp_rms"].median()
        med_rms_pool = float(np.nanmedian(per_id_rms_med.to_numpy(dtype=float)))
        mag_per_id_min = gdf.groupby("_id", sort=False)["mag"].min()
        mag_per_id_max = gdf.groupby("_id", sort=False)["mag"].max()
        mn_mag = float(np.nanmin(mag_per_id_min.to_numpy(dtype=float)))
        mx_mag = float(np.nanmax(mag_per_id_max.to_numpy(dtype=float)))
        n_unique = int(gdf["_id"].nunique())
        rows_out.append(("Comparison stars used (unique)", f"{n_unique:d}"))
        rows_out.append(("Median comp rms (per star)",
                         f"{1085.7 * med_rms_pool:.1f} mmag" if np.isfinite(med_rms_pool) else "-"))
        if np.isfinite(mn_mag) and np.isfinite(mx_mag):
            rows_out.append(("Comp mag range", f"{mn_mag:.3f} - {mx_mag:.3f}"))
        else:
            rows_out.append(("Comp mag range", "-"))
        if "selected_tier" in self.comp_df.columns and "target_catalog_id" in self.comp_df.columns:
            _st = (self.comp_df.groupby("target_catalog_id", sort=False)["selected_tier"].first()
                   .astype(str).str.upper().str.replace("TIER", "", regex=False))
            _tc = " / ".join(f"{int((_st == str(t)).sum()):d}" for t in (1, 2, 3, 4))
            rows_out.append(("Target ensembles by colour tier (T1 / T2 / T3 / T4)", _tc))
        n_good = n_susp = n_rej = 0
        try:
            from photometry_core import parse_comp_quality_json_map

            for qpath in sorted(self.lc_dir.glob("comp_quality_*.json")):
                if not qpath.is_file():
                    continue
                qraw = json.loads(qpath.read_text(encoding="utf-8"))
                if not isinstance(qraw, dict):
                    continue
                for _cid_q, st in parse_comp_quality_json_map(qraw).items():
                    s = str(st.get("quality", "")).strip().lower()
                    if s == "good":
                        n_good += 1
                    elif s == "suspect":
                        n_susp += 1
                    elif s in ("excluded", "rejected"):
                        n_rej += 1
        except Exception:  # noqa: BLE001
            # EXC-0232: T2 -- report/export may omit or misstate (elif s in ('excluded', 'rejected'): / n_rej += 1 / ... (EXCEPT-BULK 2026-07-08)
            pass
        if n_good or n_susp or n_rej:
            rows_out.append(("Good / suspect / excluded (target-comp pairs)", f"{n_good:d} / {n_susp:d} / {n_rej:d}"))
        else:
            rows_out.append(("Good / suspect / excluded (target-comp pairs)", "-"))
        return rows_out

    def _load_variability_candidates_by_cid(self) -> dict[str, dict[str, Any]]:
        from variability_field import load_variability_candidates  # noqa: PLC0415

        out: dict[str, dict[str, Any]] = {}
        try:
            vdf = load_variability_candidates(self.photometry_dir)
        except Exception:  # noqa: BLE001
            # EXC-0233: T2 -- report/export may omit or misstate (EXCEPT-BULK 2026-07-08)
            return out
        for _, r in vdf.iterrows():
            nk = self._norm_cid_decimal(r.get("catalog_id"))
            if nk:
                out[nk] = r.to_dict()
        return out

    def _resolve_observer_identity(self) -> tuple[str, str]:
        name = ""
        code = ""
        if isinstance(self._pipeline_meta, dict):
            obs = self._pipeline_meta.get("observer")
            if isinstance(obs, dict):
                name = str(obs.get("name", obs.get("observer_name", "")) or "").strip()
                code = str(obs.get("code", obs.get("obscode", obs.get("observer_code", ""))) or "").strip()
        cfg = self._cfg
        if cfg is None:
            try:
                from config import AppConfig

                cfg = AppConfig()
            except Exception:  # noqa: BLE001
                cfg = None
        if cfg is not None:
            if not name:
                name = str(getattr(cfg, "observer_name", "") or "").strip()
            if not code:
                code = str(getattr(cfg, "observer_code", "") or "").strip()
        if not name:
            name = "-"
        if not code:
            code = "-"
        return name, code

    def _resolve_plate_scale_arcsec(self) -> float:
        if isinstance(self._pipeline_meta, dict):
            dp = self._pipeline_meta.get("dynamic_params")
            if isinstance(dp, dict):
                ps = pd.to_numeric(dp.get("plate_scale_arcsec_px"), errors="coerce")
                if np.isfinite(float(ps)) and float(ps) > 0:
                    return float(ps)
            ps2 = pd.to_numeric(self._pipeline_meta.get("plate_scale_arcsec_px"), errors="coerce")
            if np.isfinite(float(ps2)) and float(ps2) > 0:
                return float(ps2)
        ms = self.platesolve_dir / "MASTERSTAR.fits"
        if ms.is_file():
            try:
                from astropy.io import fits
                from astropy.wcs import WCS

                with fits.open(ms) as hdul:
                    w = WCS(hdul[0].header)
                    cd = w.pixel_scale_matrix
                    scale = float(np.sqrt(np.abs(np.linalg.det(cd)))) * 3600.0
                    if np.isfinite(scale) and scale > 0:
                        return scale
            except Exception:  # noqa: BLE001
                # EXC-0234: T2 -- report/export may omit or misstate (if np.isfinite(scale) and scale > 0: / return scale... (EXCEPT-BULK 2026-07-08)
                pass
        return float("nan")

    def _resolve_equipment_summary(self) -> str:
        tel = cam = ""
        draft_id = self._draft_id_from_dirname()
        if draft_id is not None:
            try:
                from config import AppConfig
                from database import VyvarDatabase

                row = VyvarDatabase(AppConfig().database_path).fetch_obs_draft_telescope_equipment(int(draft_id))
                if isinstance(row, dict):
                    tel = str(row.get("telescope_name", "") or "").strip()
                    cam = str(row.get("equipment_name", "") or "").strip()
            except Exception:  # noqa: BLE001
                # EXC-0235: T2 -- report/export may omit or misstate (tel = str(row.get('telescope_name', '') or '').stri... (EXCEPT-BULK 2026-07-08)
                pass
        if isinstance(self._pipeline_meta, dict):
            eq = self._pipeline_meta.get("equipment")
            if isinstance(eq, dict):
                tel = tel or str(eq.get("telescope", eq.get("telescope_name", "")) or "").strip()
                cam = cam or str(eq.get("camera", eq.get("equipment_name", "")) or "").strip()
        parts = [p for p in (tel, cam) if p]
        if parts:
            return " . ".join(parts)
        return str(self.obs_group or "-")

    def _build_night_qc_summary(self) -> dict[str, Any]:
        qc = self._load_obs_files_for_obs()
        if qc.empty:
            qc = self._load_qc_metrics_for_obs()
        out: dict[str, Any] = {
            "n_total": 0,
            "n_used": 0,
            "n_rejected": 0,
            "fwhm_min": float("nan"),
            "fwhm_med": float("nan"),
            "fwhm_max": float("nan"),
            "bjd_min": float("nan"),
            "bjd_max": float("nan"),
        }
        if qc.empty:
            return self._add_lc_bjd_span(out)
        out["n_total"] = int(len(qc))
        rej = 0
        if "IS_REJECTED" in qc.columns:  # frame QC decision of the run
            rej = int(pd.to_numeric(qc["IS_REJECTED"], errors="coerce").fillna(0).astype(bool).sum())
        out["n_rejected"] = int(rej)
        out["n_used"] = int(max(0, out["n_total"] - out["n_rejected"]))
        # REPORT-AUDIT-2026-10: used = frames with light-curve points; FWHM over those frames
        measured = self._measured_frame_stems()
        qc_used = qc
        if measured and "_dst_name" in qc.columns:
            m_used = qc["_dst_name"].map(lambda n: self._frame_stem(n) in measured)
            if bool(m_used.any()):
                out["n_used"] = int(m_used.sum())
                qc_used = qc.loc[m_used]
        if "FWHM_PX" in qc_used.columns:
            f = pd.to_numeric(qc_used["FWHM_PX"], errors="coerce")
            if f.notna().any():
                out["fwhm_min"] = float(np.nanmin(f))
                out["fwhm_med"] = float(np.nanmedian(f))
                out["fwhm_max"] = float(np.nanmax(f))
        return self._add_lc_bjd_span(out)

    def _add_lc_bjd_span(self, out: dict[str, Any]) -> dict[str, Any]:
        # BJD span from the light curves (the QC table carries a UTC inspection JD, not BJD)
        bjd_vals: list[float] = []
        for lp in self._lc_files():
            try:
                dfl = pd.read_csv(lp, usecols=lambda c: c in {"bjd", "bjd_tdb", "hjd", "jd"}, low_memory=False)
                xcol = next((xc for xc in ("bjd_tdb", "bjd", "hjd", "jd") if xc in dfl.columns), None)
                if xcol:
                    xv = pd.to_numeric(dfl[xcol], errors="coerce").dropna()
                    bjd_vals.extend(float(x) for x in xv.to_numpy(dtype=float) if np.isfinite(x))
            except Exception:  # noqa: BLE001
                # EXC-0236: T2 -- report/export may omit or misstate (xv = pd.to_numeric(dfl[xcol], errors='coerce').drop... (EXCEPT-BULK 2026-07-08)
                continue
        if bjd_vals:
            arr = np.asarray(bjd_vals, dtype=float)
            out["bjd_min"] = float(np.nanmin(arr))
            out["bjd_max"] = float(np.nanmax(arr))
        return out

    def _variability_cover_rows(self) -> list[tuple[str, str]]:
        vm = self._variability_cover_metrics()
        if not vm:
            return []
        rows = [
            ("Stars analysed (variability)", f"{int(vm.get('n_all', 0)):d}"),
            ("New RMS candidates", f"{int(vm.get('n_rms', 0)):d}"),
            ("New VDI candidates", f"{int(vm.get('n_vdi', 0)):d}"),
            ("Combined edge-safe candidates", f"{int(vm.get('n_combined', 0)):d}"),
            ("Known VSX variables", f"{int(vm.get('n_vsx', 0)):d}"),
        ]
        return rows

    def _compress_image_for_pdf(self, 
        src_path: str | Path,
        max_width_px: int = 1200,
        jpeg_quality: int = 72,
    ) -> tuple[BytesIO, str]:
        """
        Open image, resize if wider than max_width_px (preserve aspect ratio),
        convert to JPEG, return (BytesIO, "JPEG"). RGBA is flattened onto white.
        ``force_jpeg`` reserved for API compatibility with the Phase 4 spec.
        """
        src = Path(src_path)
        if not src.is_file():
            raise FileNotFoundError(str(src))
        try:
            from PIL import Image as PILImage
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Pillow required for PDF image compression") from exc

        # OUTPUT-REVIEW-2026-10 step 7: in memory (a disk cache keyed by the path never hit for the
        # images the report draws again every run)

        def _to_rgb_white_bg(im: Any) -> Any:
            if im.mode == "RGBA":
                bg = PILImage.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[3])
                return bg
            if im.mode == "LA":
                bg = PILImage.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[1])
                return bg
            if im.mode == "P":
                if "transparency" in im.info:
                    im = im.convert("RGBA")
                    return _to_rgb_white_bg(im)
                return im.convert("RGB")
            return im.convert("RGB")

        with PILImage.open(src) as im0:
            if im0.format == "JPEG" and im0.size[0] <= int(max_width_px):
                # already a JPEG at or below the embed width: no second lossy encode
                return BytesIO(src.read_bytes()), "JPEG"
            im = _to_rgb_white_bg(im0)
            w, h = im.size
            mw = int(max_width_px)
            if w > mw and w > 0:
                scale = float(mw) / float(w)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                im = im.resize((nw, nh), resample=PILImage.Resampling.LANCZOS)

            buf = BytesIO()
            im.save(
                buf,
                format="JPEG",
                quality=int(jpeg_quality),
                optimize=True,
                progressive=True,
            )
            buf.seek(0)
            return buf, "JPEG"
    def _compress_png_bytes_for_pdf(self, 
        png_bytes: bytes,
        max_width_px: int = 1200,
        jpeg_quality: int = 72,
    ) -> tuple[BytesIO, str]:
        """Same as _compress_image_for_pdf but from an in-memory PNG (e.g. matplotlib buffer)."""
        try:
            from PIL import Image as PILImage
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Pillow required for PDF image compression") from exc

        def _to_rgb_white_bg(im: Any) -> Any:
            if im.mode == "RGBA":
                bg = PILImage.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[3])
                return bg
            if im.mode == "LA":
                bg = PILImage.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[1])
                return bg
            if im.mode == "P":
                if "transparency" in im.info:
                    im = im.convert("RGBA")
                    return _to_rgb_white_bg(im)
                return im.convert("RGB")
            return im.convert("RGB")

        with PILImage.open(BytesIO(png_bytes)) as im0:
            im = _to_rgb_white_bg(im0)
            w, h = im.size
            mw = int(max_width_px)
            if w > mw and w > 0:
                scale = float(mw) / float(w)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                im = im.resize((nw, nh), resample=PILImage.Resampling.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True, progressive=True)
            buf.seek(0)
            return buf, "JPEG"
    def _page_footer(self, c: "canvas.Canvas") -> None:
        try:
            c.setFont(self.FONT_REG, 9)
            c.setFillColor(self.colors.HexColor("#1a1a2e"))
            left_txt = f"VYVAR - {self._report_draft_lbl} - {self.obs_group}"
            c.drawString(self.M_LEFT, 0.45 * self.cm, left_txt)
            if getattr(self, "_cfg_source_label", "") == "live (no run snapshot)":
                c.setFont(self.FONT_REG, 7)
                c.drawString(self.M_LEFT, 0.28 * self.cm, "config: live (no run snapshot)")
                c.setFont(self.FONT_REG, 9)
            c.drawRightString(self.PAGE_W - self.M_RIGHT, 0.45 * self.cm, f"Page {c.getPageNumber()}")
            c.setFillColor(self.colors.black)
        except Exception:  # noqa: BLE001
            # EXC-0248: T2 -- report/export may omit or misstate (c.drawRightString(self.PAGE_W - self.M_RIGHT, 0.45 ... (EXCEPT-BULK 2026-07-08)
            pass

    def _layout_y_floor(self) -> float:
        """Minimum y (pt) for body content - leave room for footer."""
        return float(self.M_BOTTOM + 0.55 * self.cm)

    def _record_overflow(self, kind: str, detail: str) -> None:
        if self._verify_overflow:
            self._overflow_violations.append(f"{kind}: {detail}")

    @property
    def overflow_violation_count(self) -> int:
        return len(self._overflow_violations)

    def _bounds_check(self, x: float, y_bottom: float, width: float, height: float = 0.0) -> None:
        if not self._verify_overflow:
            return
        if x + width > self.M_LEFT + self.USE_W + 0.5:
            self._record_overflow("width", f"x={x:.1f} w={width:.1f} exceeds USE_W")
        if y_bottom < self._layout_y_floor():
            self._record_overflow(
                "bottom",
                f"y_bottom={y_bottom:.1f} below floor={self._layout_y_floor():.1f} (h={height:.1f})",
            )

    def _layout_page_break(self, c: "canvas.Canvas") -> float:
        self._page_footer(c)
        c.showPage()
        c.setPageSize(self.landscape(self.A4))
        return float(self.PAGE_H - self.M_TOP)

    def _layout_ensure_space(self, c: "canvas.Canvas", y: float, need_pt: float) -> float:
        y = float(y)
        if y - float(need_pt) < self._layout_y_floor():
            return self._layout_page_break(c)
        return y

    def _get_para_style(
        self,
        name: str,
        *,
        fontName: str | None = None,
        fontSize: float = 8,
        leading: float | None = None,
        textColor: Any = None,
        alignment: int | None = None,
    ) -> Any:
        cache_key = f"{name}|{fontName or self.FONT_REG}|{fontSize}|{leading}|{textColor}|{alignment}"
        if cache_key in self._para_styles:
            return self._para_styles[cache_key]
        sty = self.ParagraphStyle(
            name=f"vyvar_{name}_{len(self._para_styles)}",
            fontName=fontName or self.FONT_REG,
            fontSize=float(fontSize),
            leading=float(leading if leading is not None else fontSize * 1.25),
            alignment=self.TA_LEFT if alignment is None else int(alignment),
            textColor=textColor or self.colors.black,
            wordWrap="CJK",
        )
        self._para_styles[cache_key] = sty
        return sty

    def _pdf_escape(self, text: Any) -> str:
        return escape(str(text if text is not None else ""))

    def _pdf_break_long(self, text: str, chunk: int = 10) -> str:
        s = str(text or "")
        if len(s) <= chunk:
            return self._pdf_escape(s)
        return "<br/>".join(self._pdf_escape(s[i : i + chunk]) for i in range(0, len(s), chunk))

    def _pdf_id_display(self, text: Any, *, break_digits: bool = True) -> str:
        s = str(text if text is not None else "").strip()
        if not s:
            return "-"
        if break_digits and s.isdigit() and len(s) > 12:
            return self._pdf_break_long(s, 10)
        if len(s) > 48:
            return self._pdf_escape(f"{s[:22]}\u2026{s[-10:]}")
        return self._pdf_escape(s)

    def _para_row_height(self, para: Any, width: float, min_h: float = 12.0) -> float:
        try:
            _pw, ph = para.wrap(max(12.0, width - 4.0), 9999.0)
        except Exception:  # noqa: BLE001
            ph = min_h
        return float(max(min_h, ph + 4.0))

    def _draw_paragraph_block(
        self,
        c: "canvas.Canvas",
        x: float,
        y_top: float,
        width: float,
        html: str,
        style: Any,
        *,
        paginate: bool = True,
        gap_pt: float = 2.0,
        check_bounds: bool = True,
    ) -> float:
        para = self.Paragraph(html, style)
        _pw, ph = para.wrap(max(12.0, width), 9999.0)
        if paginate:
            y_top = self._layout_ensure_space(c, y_top, ph + gap_pt)
        y_bottom = y_top - ph
        if check_bounds:
            self._bounds_check(x, y_bottom, _pw, ph)
        para.drawOn(c, x, y_bottom)
        return y_bottom - gap_pt

    def _draw_flow_lines(
        self,
        c: "canvas.Canvas",
        y: float,
        lines: list[tuple[str, str, float]],
        *,
        width: float | None = None,
        line_step: float = 9.0,
        paginate: bool = True,
    ) -> float:
        """Draw wrapped lines. Each item is (text, fontName, fontSize)."""
        use_w = float(width if width is not None else self.USE_W)
        for text, font_name, fs in lines:
            sty = self._get_para_style(
                f"flow_{hash((font_name, fs)) & 0xFFFF}",
                fontName=font_name,
                fontSize=fs,
            )
            approx_chars = max(24, int(use_w / max(4.0, fs * 0.52)))
            for sub in textwrap.wrap(str(text), width=approx_chars) or [str(text)]:
                para = self.Paragraph(self._pdf_escape(sub), sty)
                _pw, ph = para.wrap(use_w, 9999.0)
                step = max(line_step, ph + 1.0)
                if paginate:
                    y = self._layout_ensure_space(c, y, step)
                y_bottom = y - ph
                self._bounds_check(self.M_LEFT, y_bottom, _pw, ph)
                para.drawOn(c, self.M_LEFT, y_bottom)
                y = y_bottom - max(0.0, step - ph - 1.0)
        return y

    def _variability_cover_metrics(self, ) -> dict[str, Any] | None:
        """Build counts from ``var_results`` for the cover page (same merge logic as the variability UI)."""
        vr = self._var_results
        if not vr:
            return None
        rms_df = vr.get("rms_df")
        if not isinstance(rms_df, pd.DataFrame) or rms_df.empty:
            return None
        work = rms_df.copy()
        vdi_df = vr.get("vdi_df")
        if isinstance(vdi_df, pd.DataFrame) and not vdi_df.empty:
            work = work.merge(
                vdi_df[["catalog_id", "vdi_score", "vdi_z_score", "is_variable_candidate"]],
                on="catalog_id",
                how="left",
                suffixes=("_rms", "_vdi"),
            )
            work = work.rename(columns={"is_variable_candidate": "is_variable_candidate_vdi"})
        else:
            work["vdi_score"] = np.nan
            work["vdi_z_score"] = np.nan
            work["is_variable_candidate_vdi"] = False
        if "is_variable_candidate" in work.columns and "is_variable_candidate_rms" not in work.columns:
            work = work.rename(columns={"is_variable_candidate": "is_variable_candidate_rms"})
        work["is_variable_candidate_rms"] = work["is_variable_candidate_rms"].fillna(False).astype(bool)
        work["is_variable_candidate_vdi"] = work["is_variable_candidate_vdi"].fillna(False).astype(bool)
        work["is_candidate_combined"] = work["is_variable_candidate_rms"] | work["is_variable_candidate_vdi"]
        work["vsx_known_variable"] = work["vsx_known_variable"].fillna(False).astype(bool)
        n_combined_edge = int((work["is_candidate_combined"] & ~work["vsx_known_variable"]).sum())
        n_all = int(len(work))
        n_rms = int((work["is_variable_candidate_rms"] & ~work["vsx_known_variable"]).sum())
        n_vdi = int((work["is_variable_candidate_vdi"] & ~work["vsx_known_variable"]).sum())
        n_vsx = int(work["vsx_known_variable"].sum())
        return {
            "n_all": n_all,
            "n_rms": n_rms,
            "n_vdi": n_vdi,
            "n_combined": n_combined_edge,
            "n_vsx": n_vsx,
        }
    def _draw_image_fit(self, 
        c: "canvas.Canvas",
        img_path: Path,
        x: float,
        y_top: float,
        w: float,
        h: float,
        *,
        image_type: str = "default",
        force_jpeg: bool = False,
    ) -> None:
        """Draw image fitted into (w,h) box, top-left. Raster sources are JPEG-compressed before ImageReader."""
        try:
            if not img_path or not Path(img_path).exists():
                return
            ip = Path(img_path)
            mw, jq = self._IMAGE_PDF_SETTINGS.get(image_type, self._IMAGE_PDF_SETTINGS["default"])
            try:
                jbuf, _fmt = self._compress_image_for_pdf(ip, mw, jq)
                ir = self.ImageReader(jbuf)
            except Exception:  # noqa: BLE001
                ir = self.ImageReader(str(ip))
            iw, ih = ir.getSize()
            if not iw or not ih:
                return
            sx = w / float(iw)
            sy = h / float(ih)
            s = min(sx, sy)
            dw = float(iw) * s
            dh = float(ih) * s
            c.drawImage(ir, x, y_top - dh, width=dw, height=dh, mask="auto")
        except Exception:  # noqa: BLE001
            # EXC-0249: T2 -- report/export may omit or misstate (dh = float(ih) * s / c.drawImage(ir, x, y_top - dh,... (EXCEPT-BULK 2026-07-08)
            return
    def _draw_kv_table_section(self, 
        c: "canvas.Canvas",
        y: float,
        *,
        title: str,
        rows_kv: list[tuple[str, str]],
        paginate: bool = True,
        continued_title: str | None = None,
    ) -> float:
        if not rows_kv:
            return y
        metric_w = 72 * self.mm
        value_w = 93 * self.mm
        val_style = self._get_para_style("kv_value", fontSize=8)
        hdr_h = 0.48 * self.cm
        y_floor = self._layout_y_floor()
        idx = 0
        page_idx = 0
        while idx < len(rows_kv):
            if paginate:
                y = self._layout_ensure_space(c, y, hdr_h + 0.55 * self.cm)
            sec_title = title if page_idx == 0 else (continued_title or f"{title} (continued)")
            c.setFont(self.FONT_BOLD, 12)
            c.setFillColor(self.C_TITLE)
            c.drawString(self.M_LEFT, y, sec_title)
            c.setFillColor(self.colors.black)
            y -= 0.55 * self.cm

            chunk_rows: list[list[Any]] = [["Metric", "Value"]]
            chunk_heights: list[float] = [hdr_h]
            avail = y - y_floor
            while idx < len(rows_kv):
                a, b = rows_kv[idx]
                val_html = self._pdf_break_long(str(b), 36) if len(str(b)) > 36 else self._pdf_escape(b)
                val_para = self.Paragraph(val_html, val_style)
                rh = self._para_row_height(val_para, value_w, min_h=0.44 * self.cm)
                need = float(sum(chunk_heights) + rh)
                if len(chunk_rows) > 1 and need > avail:
                    break
                chunk_rows.append(
                    [
                        self.Paragraph(f"<b>{self._pdf_escape(a)}</b>", val_style),
                        val_para,
                    ]
                )
                chunk_heights.append(rh)
                idx += 1

            t = self.Table(
                chunk_rows,
                colWidths=[metric_w, value_w],
                rowHeights=chunk_heights,
            )
            sty = self.TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), self.C_TITLE),
                    ("TEXTCOLOR", (0, 0), (-1, 0), self.colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), self.FONT_BOLD),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.25, self.colors.HexColor("#cccccc")),
                ]
            )
            for i in range(1, len(chunk_rows)):
                bg = self.colors.HexColor("#f5f5f5") if (i % 2 == 0) else self.colors.white
                sty.add("BACKGROUND", (0, i), (-1, i), bg)
            t.setStyle(sty)
            th = float(sum(chunk_heights))
            if paginate:
                y = self._layout_ensure_space(c, y, th)
            table_bottom = y - th
            self._bounds_check(self.M_LEFT, table_bottom, metric_w + value_w, th)
            t.wrap(metric_w + value_w, th)
            t.drawOn(c, self.M_LEFT, table_bottom)
            y = table_bottom - 0.35 * self.cm
            page_idx += 1
            if idx < len(rows_kv) and paginate:
                self._page_footer(c)
                c.showPage()
                c.setPageSize(self.landscape(self.A4))
                y = float(self.PAGE_H - self.M_TOP)
        return y

    def _report_cover_page(self, c: "canvas.Canvas") -> None:
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 22)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.6 * self.cm, self._report_title)
        c.setFillColor(self.colors.black)

        y -= 1.25 * self.cm
        logo_path = None
        try:
            logo_path = (Path(__file__).resolve().parent.parent / "img" / "VYVAR_logo.png").resolve()
        except Exception:  # noqa: BLE001
            logo_path = None
        if logo_path and Path(logo_path).exists():
            logo_w = float(self.PAGE_W) * 0.40
            logo_h = 5.2 * self.cm
            x0 = (float(self.PAGE_W) - logo_w) / 2.0
            self._draw_image_fit(
                c,
                logo_path,
                x=x0,
                y_top=y,
                w=logo_w,
                h=logo_h,
                image_type="logo",
            )
            y -= logo_h + 0.35 * self.cm

        c.setFont(self.FONT_REG, 11)
        c.drawString(self.M_LEFT, y, f"Draft:     {self.draft_dir.name}")
        y -= 0.6 * self.cm
        c.drawString(self.M_LEFT, y, f"Setup:     {self.obs_group}")
        y -= 0.6 * self.cm
        c.drawString(self.M_LEFT, y, f"Observation date:     {self.obs_date_human}")
        y -= 0.6 * self.cm
        obs_name, obs_code = self._resolve_observer_identity()
        c.drawString(self.M_LEFT, y, f"Observer:     {obs_name}  (OBSCODE {obs_code})")
        y -= 0.6 * self.cm
        eq_txt = self._resolve_equipment_summary()
        eq_style = self._get_para_style("cover_eq", fontSize=11)
        y = self._draw_paragraph_block(
            c, self.M_LEFT, y + 0.15 * self.cm, self.USE_W, f"Equipment: {self._pdf_escape(eq_txt)}", eq_style, paginate=False, gap_pt=0.1 * self.cm
        )
        y -= 0.35 * self.cm
        ps = self._resolve_plate_scale_arcsec()
        ps_txt = f"{ps:.3f} arcsec/px" if np.isfinite(ps) else "-"
        c.drawString(self.M_LEFT, y, f"Plate scale:     {ps_txt}")
        y -= 0.6 * self.cm
        try:
            from draft_provenance import calibration_mode_report_line

            _cal_mode = None
            if isinstance(self._pipeline_meta, dict):
                _cal_mode = self._pipeline_meta.get("calibration_mode")
            c.drawString(self.M_LEFT, y, calibration_mode_report_line(_cal_mode))
            y -= 0.6 * self.cm
        except Exception:  # noqa: BLE001
            # EXC-0252: T2 -- report/export may omit or misstate (c.drawString(self.M_LEFT, y, calibration_mode_repor... (EXCEPT-BULK 2026-07-08)
            pass
        c.drawString(self.M_LEFT, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # VSX scope note on cover (informational).
        try:
            y -= 0.55 * self.cm
            c.setFont(self.FONT_REG, 10)
            c.drawString(
                self.M_LEFT,
                y,
                "VSX auto-target scope: detection-limited (DAO+Gaia match)",
            )
        except Exception as exc:  # noqa: BLE001
            logging.debug("[VSX-SCOPE] cover line skipped: %s", exc)

        var_rows = self._variability_cover_rows()
        if var_rows:
            y -= 0.45 * self.cm
            c.setFont(self.FONT_BOLD, 10)
            c.setFillColor(self.C_TITLE)
            c.drawString(self.M_LEFT, y, "Variability detection (this field)")
            c.setFillColor(self.colors.black)
            y -= 0.45 * self.cm
            c.setFont(self.FONT_REG, 10)
            for k, v in var_rows:
                c.drawString(self.M_LEFT, y, f"{k}: {v}")
                y -= 0.42 * self.cm

        y -= 0.55 * self.cm
        c.setStrokeColor(self.colors.HexColor("#cccccc"))
        c.setLineWidth(0.5)
        c.line(self.M_LEFT, y, self.PAGE_W - self.M_RIGHT, y)
        self._page_footer(c)
        c.showPage()
    def _report_observation_summary(self, c: "canvas.Canvas") -> None:
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 12)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.5 * self.cm, "Observation summary")
        c.setFillColor(self.colors.black)
        y -= 1.0 * self.cm

        c.setFont(self.FONT_BOLD, 12)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y, "Summary metrics")
        c.setFillColor(self.colors.black)
        y -= 0.6 * self.cm

        rows = [
            ("Light curves", self.lc_count_display),
            # REPORT-AUDIT-2026-10: targets of the run without a light curve get no target page
            ("Targets without light curve (no page)", f"{max(0, len(self.summary_df) - self.n_lc):d}"),
            ("Median lc_rms", f"{self.med_rms:.4f}" if np.isfinite(self.med_rms) else "-"),
            ("RMS < 0.05 mag", f"{self.rms_lt_005:d}"),
            ("Avg good comp", f"{self.avg_good_comp:.2f}" if np.isfinite(self.avg_good_comp) else "-"),
            ("Best lc_rms", f"{self.best_rms:.4f}" if np.isfinite(self.best_rms) else "-"),
            ("Worst lc_rms", f"{self.worst_rms:.4f}" if np.isfinite(self.worst_rms) else "-"),
            ("Avg BP-RP", f"{self.avg_bp_rp:.3f}" if np.isfinite(self.avg_bp_rp) else "-"),
            ("Setups", f"{self.setups:d}"),
            ("Exports (+ _dontsubmit)", f"AAVSO {self.n_aavso:d} (+{self.n_aavso_ds:d}) . "
                                        f"VAR.ASTRO {self.n_varastro:d} (+{self.n_varastro_ds:d})"),
        ]
        c.setFont(self.FONT_REG, 12)
        for k, v in rows:
            c.drawString(self.M_LEFT, y, f"{k}:")
            if k in ("Median lc_rms", "Best lc_rms", "Worst lc_rms"):
                c.setFillColor(self._metric_color(float(pd.to_numeric(v, errors="coerce"))))
            c.drawString(self.M_LEFT + 7.5 * self.cm, y, str(v))
            c.setFillColor(self.colors.black)
            y -= 0.55 * self.cm

        y -= 0.4 * self.cm
        fwhm_txt = f"{self.fwhm_px:.3f}px" if np.isfinite(self.fwhm_px) else "-"
        ap_txt = f"{self.aperture_px:.2f}px" if np.isfinite(self.aperture_px) else "-"
        c.setFont(self.FONT_REG, 11)
        c.drawString(self.M_LEFT, y, f"{self.fwhm_label}: {fwhm_txt}   |   Aperture (median): {ap_txt}")
        y -= 0.35 * self.cm
        y = self._draw_observing_conditions_section(c, y)

        def _cover_obs_condition_rows() -> list[tuple[str, str]]:
            kv: list[tuple[str, str]] = []
            qc_df0 = self._load_obs_files_for_obs()
            if qc_df0.empty:
                qc_df0 = self._load_qc_metrics_for_obs()
            am_vals: list[float] = []
            if not qc_df0.empty:
                for cnm in qc_df0.columns:
                    if "airmass" in str(cnm).lower():
                        v = pd.to_numeric(qc_df0[cnm], errors="coerce").dropna()
                        am_vals.extend(float(x) for x in v.to_numpy(dtype=float) if np.isfinite(x))
            if not am_vals:
                for lp in self._lc_files():
                    try:
                        dfl = pd.read_csv(
                            lp,
                            usecols=lambda c: c in _AIRMASS_COLS,
                            low_memory=False,
                        )
                        for acol in ("airmass", "AIRMASS", "air_mass"):
                            if acol in dfl.columns:
                                v = pd.to_numeric(dfl[acol], errors="coerce").dropna()
                                am_vals.extend(float(x) for x in v.to_numpy(dtype=float) if np.isfinite(x))
                                break
                    except Exception:  # noqa: BLE001
                        # EXC-0253: T2 -- report/export may omit or misstate (am_vals.extend(float(x) for x in v.to_numpy(dtype=f... (EXCEPT-BULK 2026-07-08)
                        continue
            if am_vals:
                arr = np.asarray(am_vals, dtype=float)
                kv.append(
                    (
                        "Airmass (min / max / median)",
                        f"{float(np.nanmin(arr)):.4f} / {float(np.nanmax(arr)):.4f} / {float(np.nanmedian(arr)):.4f}",
                    )
                )
            else:
                kv.append(("Airmass (min / max / median)", "-"))
            sky_m = float("nan")
            if not qc_df0.empty and "SKY_LEVEL" in qc_df0.columns:
                _sky_df = qc_df0
                _meas = self._measured_frame_stems()
                if _meas and "_dst_name" in qc_df0.columns:
                    _mm = qc_df0["_dst_name"].map(lambda n: self._frame_stem(n) in _meas)
                    if bool(_mm.any()):
                        _sky_df = qc_df0.loc[_mm]
                sky_m = float(np.nanmedian(pd.to_numeric(_sky_df["SKY_LEVEL"], errors="coerce")))
            kv.append(("Sky background (median, used frames)", f"{sky_m:.2f}" if np.isfinite(sky_m) else "-"))
            nq = self._night_qc
            n_used = int(nq.get("n_used", 0) or 0)
            n_rej = int(nq.get("n_rejected", 0) or 0)
            n_tot = int(nq.get("n_total", 0) or 0)
            if n_tot > 0:
                kv.append(("Frames (used / rejected / total)", f"{n_used:d} / {n_rej:d} / {n_tot:d}"))
            else:
                kv.append(("Frames (used / rejected / total)", "-"))
            bjd_min = float(nq.get("bjd_min", float("nan")))
            bjd_max = float(nq.get("bjd_max", float("nan")))
            if np.isfinite(bjd_min) and np.isfinite(bjd_max):
                kv.append(("BJD(TDB) span (session)", f"{bjd_min:.5f} -> {bjd_max:.5f}"))
            else:
                kv.append(("BJD(TDB) span (session)", "-"))
            fmin = float(nq.get("fwhm_min", float("nan")))
            fmed = float(nq.get("fwhm_med", float("nan")))
            fmax = float(nq.get("fwhm_max", float("nan")))
            if np.isfinite(fmed):
                kv.append(
                    (
                        "FWHM px used frames (min / med / max)",
                        f"{fmin:.3f} / {fmed:.3f} / {fmax:.3f}",
                    )
                )
            else:
                kv.append(("FWHM px used frames (min / med / max)", "-"))
            nf_med = float("nan")
            if "n_frames" in self.summary_df.columns:
                _nf = pd.to_numeric(self.summary_df["n_frames"], errors="coerce")
                _nf = _nf[_nf > 0]
                nf_med = float(np.nanmedian(_nf)) if len(_nf) else float("nan")
            kv.append(("LC points (median per measured target)", f"{int(round(nf_med)):d}" if np.isfinite(nf_med) else "-"))
            ms_fits = self.platesolve_dir / "MASTERSTAR.fits"
            if ms_fits.is_file():
                try:
                    from astropy.io import fits

                    with fits.open(ms_fits, memmap=False) as hdul:
                        _hsw = hdul[0].header.get("VY_HSWN")
                        _hsep = hdul[0].header.get("VY_HSEP")
                    if _hsw is not None and bool(_hsw):
                        _sep_s = "-"
                        try:
                            _sep_s = f"{float(_hsep[0] if isinstance(_hsep, tuple) else _hsep):.3f} deg"
                        except (TypeError, ValueError):
                            pass
                        kv.append(
                            (
                                "Plate-solve hint offset (warning)",
                                f"Stale pointing hint vs solved WCS ({_sep_s}); solve verified via catalog recovery.",
                            )
                        )
                except Exception:  # noqa: BLE001
                    # EXC-0254: T2 -- report/export may omit or misstate () / ) / except Exception:  # noqa: BLE001 / pass / ... (EXCEPT-BULK 2026-07-08)
                    pass
            return kv

        y = self._draw_kv_table_section(
            c,
            y - 0.45 * self.cm,
            title="Observational conditions",
            rows_kv=_cover_obs_condition_rows(),
            paginate=True,
            continued_title="Observational conditions (continued)",
        )

        y -= 0.45 * self.cm
        if y < self._layout_y_floor() + 1.0 * self.cm:
            y = self._layout_page_break(c)
        c.setStrokeColor(self.colors.HexColor("#cccccc"))
        c.setLineWidth(0.5)
        c.line(self.M_LEFT, y, self.PAGE_W - self.M_RIGHT, y)
        y -= 0.55 * self.cm
        c.setFont(self.FONT_BOLD, 12)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y, "Methods")
        c.setFillColor(self.colors.black)
        y -= 0.6 * self.cm
        c.setFont(self.FONT_REG, 9)
        c.drawString(self.M_LEFT, y, "Ensemble photometry: weighted ZP w_i = 1/sigma_i^2")
        y -= 0.45 * self.cm
        _cfg_m = self._cfg
        if _cfg_m is None:
            try:
                from config import AppConfig

                _cfg_m = AppConfig()
            except Exception:  # noqa: BLE001
                _cfg_m = None
        _ap_med = float(self.aperture_px) if np.isfinite(float(self.aperture_px)) else float("nan")
        _pu_ver = "?"
        _ap_ver = "?"
        try:
            import photutils

            _pu_ver = str(getattr(photutils, "__version__", "?"))
        except Exception:  # noqa: BLE001
            # EXC-0255: T2 -- report/export may omit or misstate (_pu_ver = str(getattr(photutils, '__version__', '?'... (EXCEPT-BULK 2026-07-08)
            pass
        try:
            import astropy

            _ap_ver = str(getattr(astropy, "__version__", "?"))
        except Exception:  # noqa: BLE001
            # EXC-0256: T2 -- report/export may omit or misstate (_ap_ver = str(getattr(astropy, '__version__', '?'))... (EXCEPT-BULK 2026-07-08)
            pass

        _period_used = bool(self._accepted_periods) or bool(
            isinstance(self._var_results, dict)
            and any(
                isinstance(v, dict) and v.get("period") not in (None, "", float("nan"))
                for v in self._var_results.values()
            )
        )
        _var_det_used = bool(self._candidates_set) or (
            (self.photometry_dir / "variability_field.csv").is_file()
            or (self.photometry_dir / "variability_candidates.csv").is_file()
        )
        _cite_ctx = build_run_citation_context(
            _cfg_m,
            pipeline_meta=self._pipeline_meta,
            targets_df=self.at_df,
            obs_group=str(self.obs_group or ""),
            period_analysis=_period_used,
            variability_detection=_var_det_used,
            tess_used=bool(self._tess_results),
            psf_used=any(self.lc_dir.glob("lightcurve_*_psf.csv")) if self.lc_dir.is_dir() else False,
        )
        methods_sections = emit_pdf_methods_sections(_cite_ctx)
        methods_lines: list[str] = []
        for sec_title, sec_items in methods_sections:
            methods_lines.append(sec_title)
            methods_lines.extend(sec_items)
        if np.isfinite(_ap_med):
            insert_at = 1 if len(methods_lines) > 1 else 0
            methods_lines.insert(
                insert_at,
                f"  Aperture: median target r={_ap_med:.2f}px (same radius for target and comps)",
            )
        _sf_meta = (self._pipeline_meta or {}).get("sigma_floor")
        _sn_meta = (self._pipeline_meta or {}).get("sigma_sys_night")
        if isinstance(_sn_meta, dict) and _sn_meta.get("applied") == "night":
            # SIGMA-SYS-NIGHT-2026-10
            methods_lines.insert(
                2 if len(methods_lines) > 2 else len(methods_lines),
                f"  Per-point uncertainties include this night's systematic floor "
                f"{1000 * float(_sn_meta.get('sigma_sys_mag', 0.0)):.1f} mmag, measured on "
                f"{int(_sn_meta.get('n_comps', 0))} bright comparison stars (scatter minus photon noise); "
                "correlated (red) noise is reported separately.",
            )
        elif isinstance(_sf_meta, dict):
            methods_lines.insert(
                2 if len(methods_lines) > 2 else len(methods_lines),
                f"  Per-point uncertainties include a per-rig systematic floor "
                f"({1000 * float(_sf_meta.get('sigma_sys_mag', 0.0) or 0.0):.1f} mmag, EQUIPMENTS); correlated "
                "(red) noise is reported separately and is not included in per-point bars.",
            )
        if _pu_ver != "?" or _ap_ver != "?":
            for i, ln in enumerate(methods_lines):
                if ln.strip().startswith("Bradley et al."):
                    methods_lines[i] = f"  {ln.strip()} (v{_pu_ver})"
                elif ln.strip().startswith("Astropy Collaboration"):
                    methods_lines[i] = f"  {ln.strip()} (v{_ap_ver})"
        c.setFont(self.FONT_BOLD, 8)
        c.setFont(self.FONT_REG, 7.5)
        line_step = 9.0
        _section_titles = {sec[0] for sec in methods_sections}
        flow_lines: list[tuple[str, str, float]] = []
        for block in methods_lines:
            font_name = self.FONT_BOLD if block in _section_titles else self.FONT_REG
            font_size = 8.0 if block in _section_titles else 7.5
            approx_chars = max(24, int(self.USE_W / (font_size * 0.52)))
            for ln in textwrap.wrap(block, width=approx_chars):
                flow_lines.append((ln, font_name, font_size))
        y = self._draw_flow_lines(c, y, flow_lines, line_step=line_step, paginate=True)

        y -= 0.35 * self.cm
        if y < self._layout_y_floor() + 0.55 * self.cm:
            y = self._layout_page_break(c)
        c.setFont(self.FONT_REG, 9)
        c.setStrokeColor(self.colors.HexColor("#cccccc"))
        c.setLineWidth(0.5)
        c.line(self.M_LEFT, y, self.PAGE_W - self.M_RIGHT, y)
        y -= 0.55 * self.cm
        y = self._draw_kv_table_section(
            c, y, title="Comparison Star Pool", rows_kv=self.comp_pool_cover_rows, paginate=True
        )

        self._page_footer(c)
        c.showPage()


    def _draft_id_from_dirname(self, ) -> int | None:
        """Best-effort draft_id parse from ``draft_000123`` directory name."""
        nm = str(getattr(self.draft_dir, "name", "") or "")
        if "draft_" not in nm:
            return None
        try:
            tail = nm.split("draft_", 1)[1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            return int(digits) if digits else None
        except Exception:  # noqa: BLE001
            # EXC-0257: T2 -- report/export may omit or misstate (digits = ''.join(ch for ch in tail if ch.isdigit())... (EXCEPT-BULK 2026-07-08)
            return None
    def _load_obs_files_for_obs(self, ) -> pd.DataFrame:
        """Load QA metrics from DB (manifest files[]) for this draft + setup."""
        draft_id = self._draft_id_from_dirname()
        if draft_id is None:
            return pd.DataFrame()
        try:
            from config import AppConfig
            from database import VyvarDatabase
        except Exception:  # noqa: BLE001
            # EXC-0258: T2 -- report/export may omit or misstate (from config import AppConfig / from database import... (EXCEPT-BULK 2026-07-08)
            return pd.DataFrame()
        try:
            cfg = AppConfig()
            db = VyvarDatabase(cfg.database_path)
            rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
        except Exception:  # noqa: BLE001
            # EXC-0259: T2 -- report/export may omit or misstate (db = VyvarDatabase(cfg.database_path) / rows = db.f... (EXCEPT-BULK 2026-07-08)
            return pd.DataFrame()
        if not rows:
            return pd.DataFrame()
        dfo = pd.DataFrame(rows)
        if dfo.empty:
            return pd.DataFrame()

        # Filter to current observation group (best-effort).
        try:
            p = str(self.obs_group).split("_")
            flt = str(p[0]) if len(p) >= 1 else ""
            exp = float(p[1]) if len(p) >= 2 else float("nan")
            binv = str(int(float(p[2]))) if len(p) >= 3 else ""
            group_key = f"{flt}|{int(exp)}|{binv}" if (flt and np.isfinite(exp) and binv) else ""
        except Exception:  # noqa: BLE001
            flt, exp, group_key = "", float("nan"), ""

        if group_key and "OBSERVATION_GROUP_KEY" in dfo.columns:
            m = dfo["OBSERVATION_GROUP_KEY"].astype(str).eq(str(group_key))
            if m.any():
                dfo = dfo.loc[m].copy()

        if (not group_key or dfo.empty) and ("FILTER" in dfo.columns):
            m2 = dfo["FILTER"].astype(str).str.strip().eq(str(flt))
            if np.isfinite(exp) and "EXPTIME" in dfo.columns:
                ex = pd.to_numeric(dfo["EXPTIME"], errors="coerce")
                m2 = m2 & (np.abs(ex - float(exp)) < 1e-3)
            if m2.any():
                dfo = dfo.loc[m2].copy()

        if dfo.empty:
            return pd.DataFrame()

        # Normalize columns to the QC shape used below.
        dfo["_dst_name"] = dfo["FILE_PATH"].astype(str).apply(lambda s: Path(s).name) if "FILE_PATH" in dfo.columns else ""
        dfo["FWHM_PX"] = pd.to_numeric(dfo.get("FWHM"), errors="coerce")
        dfo["SKY_LEVEL"] = pd.to_numeric(dfo.get("SKY_LEVEL"), errors="coerce")
        dfo["ELONGATION"] = pd.to_numeric(dfo.get("ELONGATION_MEAN"), errors="coerce")
        dfo["STAR_COUNT"] = pd.to_numeric(dfo.get("STAR_COUNT"), errors="coerce")
        dfo["REJECTED_AUTO"] = pd.to_numeric(dfo.get("REJECTED_AUTO"), errors="coerce")
        dfo = dfo.reset_index(drop=True)
        dfo["frame_index"] = np.arange(1, len(dfo) + 1, dtype=int)
        dfo["_qa_source"] = "db"
        return self._mark_qc_rejected(dfo)
    def _load_qc_metrics_for_obs(self, ) -> pd.DataFrame:
        from pipeline_catalog import find_qc_metrics_csv

        qc_csv = find_qc_metrics_csv(self.draft_dir)
        if qc_csv is None:
            return pd.DataFrame()
        try:
            dfq = pd.read_csv(qc_csv, low_memory=False)
        except Exception:  # noqa: BLE001
            # EXC-0260: T2 -- report/export may omit or misstate (try: / dfq = pd.read_csv(qc_csv, low_memory=False) ... (EXCEPT-BULK 2026-07-08)
            return pd.DataFrame()
        if dfq.empty:
            return pd.DataFrame()
        if "dst" in dfq.columns:
            m = dfq["dst"].astype(str).str.contains(str(self.obs_group), regex=False)
            dfq = dfq.loc[m].copy()
        if dfq.empty:
            return pd.DataFrame()
        dfq["_dst_name"] = dfq["dst"].astype(str).apply(lambda s: Path(s).name) if "dst" in dfq.columns else ""
        dfq["FWHM_PX"] = pd.to_numeric(dfq.get("fwhm_px"), errors="coerce")
        dfq["SKY_LEVEL"] = pd.to_numeric(dfq.get("bg_median"), errors="coerce")
        dfq["ELONGATION"] = pd.to_numeric(dfq.get("elongation"), errors="coerce")
        dfq["STAR_COUNT"] = pd.to_numeric(dfq.get("n_stars_detected"), errors="coerce")
        dfq = dfq.reset_index(drop=True)
        dfq["frame_index"] = np.arange(1, len(dfq) + 1, dtype=int)
        return self._mark_qc_rejected(dfq)
    def _frame_qc_group(self) -> dict[str, Any]:
        """This setup's entry of ``frame_qc.json`` (QC-PER-GROUP-2026-10), or {}."""
        try:
            from frame_qc import load_frame_qc  # noqa: PLC0415

            groups = (load_frame_qc(self.draft_dir).get("groups") or {})
        except (OSError, ValueError, TypeError):
            return {}
        for g in groups.values():
            if str(g.get("setup") or "") == str(self.obs_group):
                return dict(g)
        return {}
    @staticmethod
    def _frame_stem(name: Any) -> str:
        """Frame identity across raw / processed / per-frame CSV names: 'proc_X_001.fits' -> 'x_001'."""
        st = Path(str(name or "").replace("\\", "/")).stem.lower()
        return st[5:] if st.startswith("proc_") else st

    def _mark_qc_rejected(self, df: pd.DataFrame) -> pd.DataFrame:
        """REPORT-AUDIT-2026-10: IS_REJECTED from the run's frame QC (frame_qc.json 'rejected' of
        this setup); the loaders never set it, so rejected frames were shown as kept."""
        if df.empty or "_dst_name" not in df.columns:
            return df
        rej = {self._frame_stem(k) for k in (self._frame_qc_group().get("rejected") or {})}
        df["IS_REJECTED"] = df["_dst_name"].map(lambda n: int(self._frame_stem(n) in rej))
        return df

    def _measured_frame_stems(self) -> set[str]:
        """Frames that gave at least one light-curve point (LC ``source_file``), cached."""
        if getattr(self, "_measured_stems_cache", None) is not None:
            return self._measured_stems_cache
        out: set[str] = set()
        for lp in self._lc_files():
            try:
                sf = pd.read_csv(lp, usecols=["source_file"])["source_file"].dropna().astype(str)
            except (OSError, ValueError, KeyError):
                continue
            out.update(self._frame_stem(x) for x in sf.unique())
        self._measured_stems_cache = out
        return out

    def _qa_fwhm_limit_px(self, ) -> float:
        """FWHM limit applied by the run's frame QC for this setup (frame_qc.json); NaN when
        the run applied none. The report never computes its own limit."""
        lim = self._frame_qc_group().get("fwhm_limit")
        try:
            lf = float(lim)
        except (TypeError, ValueError):
            return float("nan")
        return lf if np.isfinite(lf) and lf > 0.0 else float("nan")
    def _qc_row_by_frame_index(self, qc_df: pd.DataFrame, fi: int) -> dict[str, Any] | None:
        if fi < 1 or qc_df.empty or "frame_index" not in qc_df.columns:
            return None
        m = pd.to_numeric(qc_df["frame_index"], errors="coerce") == int(fi)
        if bool(m.any()):
            return qc_df.loc[m].iloc[0].to_dict()
        return None
    def _qc_row_by_file_hint(self, qc_df: pd.DataFrame, hint: str) -> dict[str, Any] | None:
        s = str(hint or "").strip()
        if not s or qc_df.empty or "_dst_name" not in qc_df.columns:
            return None
        if s.lower() in ("none", "nan", "0"):
            return None
        p = Path(s)
        candidates = {p.name.strip(), s.strip()}
        dn = qc_df["_dst_name"].astype(str).str.strip()
        for nm in candidates:
            if not nm:
                continue
            want = Path(nm).name.lower()
            m = dn.str.lower() == want
            if bool(m.any()):
                return qc_df.loc[m].iloc[0].to_dict()
        ref_l = p.name.lower()
        if len(ref_l) < 3:
            return None
        best: dict[str, Any] | None = None
        best_len = -1
        for _, row in qc_df.iterrows():
            dnl = str(row.get("_dst_name") or "").strip().lower()
            if not dnl:
                continue
            if dnl == ref_l or dnl.startswith(ref_l) or ref_l.startswith(dnl):
                L = min(len(dnl), len(ref_l))
                if best_len < L:
                    best_len = L
                    best = row.to_dict()
        return best
    def _masterstar_from_candidates_csv(self, qc_df: pd.DataFrame) -> dict[str, Any] | None:
        csvp = self.platesolve_dir / "masterstar_candidates.csv"
        if not csvp.is_file() or qc_df.empty:
            return None
        try:
            cdf = pd.read_csv(csvp, low_memory=False)
        except Exception:  # noqa: BLE001
            # EXC-0262: T2 -- report/export may omit or misstate (try: / cdf = pd.read_csv(csvp, low_memory=False) / ... (EXCEPT-BULK 2026-07-08)
            return None
        if cdf.empty:
            return None
        for sc in ("rank", "RANK", "qc_rank", "overall_rank", "score", "SCORE", "quality_score"):
            if sc in cdf.columns:
                low = str(sc).lower()
                asc = low in ("rank", "qc_rank", "overall_rank")
                try:
                    cdf = cdf.sort_values(sc, ascending=asc).reset_index(drop=True)
                except Exception:  # noqa: BLE001
                    cdf = cdf.reset_index(drop=True)
                break
        path_cols = [
            c
            for c in (
                "FILE_PATH",
                "file_path",
                "path",
                "dst",
                "FILE",
                "filename",
                "basename",
                "processed_path",
            )
            if c in cdf.columns
        ]
        if not path_cols:
            return None
        top = cdf.iloc[0]
        hit = self._qc_row_by_file_hint(qc_df, str(top.get(path_cols[0]) or ""))
        if hit:
            hit["_ms_resolve"] = "masterstar_candidates.csv"
        return hit
    def _match_qc_row_by_vy_header_metrics(self, qc_df: pd.DataFrame, hdr: Any) -> dict[str, Any] | None:
        if hdr is None or qc_df is None or qc_df.empty:
            return None
        try:
            fwhm = float(hdr.get("VY_FWHM", float("nan")))
            elong = float(hdr.get("VY_ELONG", float("nan")))
            nstar = float(hdr.get("VY_NSTAR", float("nan")))
            sky = float(hdr.get("VY_QCBG", float("nan")))
        except Exception:  # noqa: BLE001
            # EXC-0263: T2 -- report/export may omit or misstate (nstar = float(hdr.get('VY_NSTAR', float('nan'))) / ... (EXCEPT-BULK 2026-07-08)
            return None
        d = np.zeros(len(qc_df), dtype=float)
        wsum = 0.0
        _q = pd.to_numeric(qc_df["FWHM_PX"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(fwhm) and np.isfinite(_q).any():  # FIX-FITSQA-NAN: skip an all-NaN QC column
            d += np.square(_q - float(fwhm))
            wsum += 1.0
        _q = pd.to_numeric(qc_df["ELONGATION"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(elong) and np.isfinite(_q).any():  # FIX-FITSQA-NAN: skip an all-NaN QC column
            d += np.square(_q - float(elong))
            wsum += 1.0
        _q = pd.to_numeric(qc_df["STAR_COUNT"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(nstar) and np.isfinite(_q).any():  # FIX-FITSQA-NAN: skip an all-NaN QC column
            d += np.square(_q - float(nstar))
            wsum += 1.0
        _q = pd.to_numeric(qc_df["SKY_LEVEL"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(sky) and np.isfinite(_q).any():  # FIX-FITSQA-NAN: skip an all-NaN QC column
            d += np.square(_q - float(sky))
            wsum += 1.0
        if wsum <= 0.0 or not np.isfinite(d).any():
            # FIX-FITSQA-NAN: QC columns all NaN (e.g. missing SKY_LEVEL) -> no match, never crash the PDF
            return None
        i = int(np.nanargmin(d))
        r = qc_df.iloc[i].to_dict()
        r["_match_dist"] = float(d[i])
        r["_ms_resolve"] = "VY_FWHM/header-metrics"
        return r
    def _resolve_masterstar_used_frame(self, qc_df: pd.DataFrame) -> dict[str, Any] | None:
        """Resolve MASTERSTAR source row: FITS header frame/path keywords, CSV rank, else VY_* metric match."""
        if qc_df is None or qc_df.empty:
            return None
        ms_fits = self.platesolve_dir / "MASTERSTAR.fits"
        hdr = None
        if ms_fits.is_file():
            try:
                from astropy.io import fits

                with fits.open(ms_fits, memmap=False) as hdul:
                    hdr = hdul[0].header
            except Exception:  # noqa: BLE001
                hdr = None
        if hdr is not None:
            for key in ("MSFRAME", "MASTERFRM", "MASTERFRAME", "VY_MSFRAME", "VY_SRCFRAME", "SRCFRAME", "VY_FRAME"):
                raw = hdr.get(key)
                if raw is None:
                    continue
                if isinstance(raw, tuple):
                    raw = raw[0]
                try:
                    fi = int(float(str(raw).strip()))
                    if fi > 0:
                        hit = self._qc_row_by_frame_index(qc_df, fi)
                        if hit:
                            hit["_ms_resolve"] = f"header:{key}"
                            return hit
                except Exception:  # noqa: BLE001
                    # EXC-0264: T2 -- report/export may omit or misstate (hit['_ms_resolve'] = f'header:{key}' / return hit /... (EXCEPT-BULK 2026-07-08)
                    continue
            for key in ("VY_REFFILE", "VY_SRCPATH", "VY_SRCFILE", "VY_REF", "ORIGNAME", "ORIGFILE"):
                raw = hdr.get(key)
                if raw is None:
                    continue
                if isinstance(raw, tuple):
                    raw = raw[0]
                s = str(raw).strip()
                hit = self._qc_row_by_file_hint(qc_df, s)
                if hit:
                    hit["_ms_resolve"] = f"header:{key}"
                    return hit
        hit_csv = self._masterstar_from_candidates_csv(qc_df)
        if hit_csv:
            return hit_csv
        if hdr is not None:
            return self._match_qc_row_by_vy_header_metrics(qc_df, hdr)
        return None
    def _report_fits_qa(self, c: "canvas.Canvas") -> None:
        qc_df = self._load_obs_files_for_obs()
        if qc_df.empty:
            qc_df = self._load_qc_metrics_for_obs()
        if qc_df.empty:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001
            # EXC-0265: T2 -- report/export may omit or misstate (try: / import matplotlib.pyplot as plt / except Exc... (EXCEPT-BULK 2026-07-08)
            return

        # Charts use per-frame QC from manifest files[] (FWHM_PX vs frame_index), same source as FITS QA dashboard.
        used = self._resolve_masterstar_used_frame(qc_df)

        # FWHM limit line: the limit the run's frame QC applied to this setup (frame_qc.json).
        fwhm_limit = float(self._qa_fwhm_limit_px())
        lim_active = bool(np.isfinite(fwhm_limit) and float(fwhm_limit) > 0.0)

        # Title
        c.setFont(self.FONT_BOLD, 16)
        c.setFillColor(self.colors.black)
        c.drawString(self.M_LEFT, self.PAGE_H - self.M_TOP - 0.4 * self.cm, "FITS Quality Assessment")
        c.setFont(self.FONT_REG, 9)
        c.setFillColor(self.colors.HexColor("#555555"))
        c.drawString(self.M_LEFT, self.PAGE_H - self.M_TOP - 1.0 * self.cm, f"Setup: {self.obs_group}")
        c.setFillColor(self.colors.black)

        x = qc_df["frame_index"].to_numpy(dtype=int)

        png_payload = b""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.0, 6.2), constrained_layout=True)
        try:
            # FWHM plot
            f = pd.to_numeric(qc_df["FWHM_PX"], errors="coerce").to_numpy(dtype=float)
            if "IS_REJECTED" in qc_df.columns:  # frame QC decision of the run
                rej_mask = pd.to_numeric(qc_df["IS_REJECTED"], errors="coerce").fillna(0).to_numpy() > 0
            else:
                rej_mask = np.zeros(len(qc_df), dtype=bool)
            col = np.where(rej_mask, "#c0392b", np.where(np.isfinite(f), "#27ae60", "#95a5a6"))
            ax1.plot(x, f, color="#6c5ce7", linewidth=1.1, alpha=0.65)
            ax1.scatter(x, f, c=col, s=20)
            if lim_active:
                ax1.axhline(y=float(fwhm_limit), color="#c0392b", linestyle="--", linewidth=1.3)
            ax1.set_xlabel("Frame Index")
            ax1.set_ylabel("FWHM (px)")
            ax1.set_title(
                "FWHM - green kept, red rejected by frame QC"
                + (f"; dashed = channel limit {float(fwhm_limit):.2f} px" if lim_active else "")
            )
            ax1.grid(True, alpha=0.25)

            sky = pd.to_numeric(qc_df["SKY_LEVEL"], errors="coerce").to_numpy(dtype=float)
            col2 = np.where(rej_mask, "#c0392b", "#27ae60")
            ax2.plot(x, sky, color="#6c5ce7", linewidth=1.1, alpha=0.65)
            ax2.scatter(x, sky, c=col2, s=20)
            ax2.set_xlabel("Frame Index")
            ax2.set_ylabel("Sky level (ADU)")
            ax2.set_title("Background sky level - green kept, red rejected by frame QC")
            ax2.grid(True, alpha=0.25)

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            png_payload = buf.getvalue()
        finally:
            plt.close(fig)

        if not png_payload:
            return

        qa_mw, qa_jq = self._IMAGE_PDF_SETTINGS["default"]
        try:
            jbuf, _fmt_q = self._compress_png_bytes_for_pdf(png_payload, qa_mw, qa_jq)
            img = self.ImageReader(jbuf)
        except Exception:  # noqa: BLE001
            img = self.ImageReader(BytesIO(png_payload))
        plots_h = 11.0 * self.cm
        c.drawImage(
            img,
            self.M_LEFT,
            self.PAGE_H - self.M_TOP - 1.6 * self.cm - plots_h,
            width=self.USE_W,
            height=plots_h,
            preserveAspectRatio=True,
            mask="auto",
        )

        # Table section under plots
        y0 = self.PAGE_H - self.M_TOP - 1.6 * self.cm - plots_h - 0.6 * self.cm
        c.setFont(self.FONT_BOLD, 11)
        c.setFillColor(self.colors.black)
        c.drawString(self.M_LEFT, y0, "Masterstar reference frame")
        y0 -= 0.45 * self.cm
        c.setFont(self.FONT_REG, 9)
        c.setFillColor(self.colors.HexColor("#333333"))
        if used:
            used_fn = str(used.get("_dst_name") or "")
            used_fr = int(pd.to_numeric(used.get("frame_index"), errors="coerce") or 0)
            used_fwhm = pd.to_numeric(used.get("FWHM_PX"), errors="coerce")
            if np.isfinite(used_fwhm):
                c.drawString(
                    self.M_LEFT,
                    y0,
                    f"Used frame: {used_fn}  (Frame {used_fr}, FWHM={float(used_fwhm):.2f} px)",
                )
            else:
                c.drawString(self.M_LEFT, y0, f"Used frame: {used_fn}  (Frame {used_fr})")
        else:
            c.drawString(self.M_LEFT, y0, "Used frame: - (could not infer from available metadata)")
        gq = self._frame_qc_group()
        if gq:
            y0 -= 0.7 * self.cm
            c.setFont(self.FONT_BOLD, 11)
            c.setFillColor(self.colors.black)
            c.drawString(self.M_LEFT, y0, "Frame QC (this setup)")
            c.setFont(self.FONT_REG, 9)
            c.setFillColor(self.colors.HexColor("#333333"))
            el = gq.get("elong_limit")
            dz = gq.get("dzp_limit")
            pa = gq.get("patchy_limit")
            bt = gq.get("best_transparency")

            def _f(v: Any, fmt: str) -> str:
                return format(float(v), fmt) if v is not None else "none"

            lines = [
                f"Frames: {int(gq.get('n_frames') or 0)}, rejected: {int(gq.get('n_rejected') or 0)}"
                f"; FWHM median {_f(gq.get('fwhm_median'), '.2f')} px (not a rejection criterion)",
                "Reject (limits = median + k*1.4826*MAD of this setup): no FWHM; elongation > "
                f"{_f(el, '.3f')}; transparency loss dZP > {_f(dz, '.3f')} mag; "
                f"non-uniform (quadrant spread) > {_f(pa, '.3f')} mag; < 5 common stars",
                "MASTERSTAR source: max T/(FWHM*sigma_pix)"
                + (f" -> {Path(str(gq.get('best_path')).replace(chr(92), '/')).name}" if gq.get("best_path") else "")
                + (f" (T={float(bt):.3f})" if bt is not None else ""),
            ]
            for ln in lines:
                y0 -= 0.42 * self.cm
                c.drawString(self.M_LEFT, y0, ln)
        c.setFillColor(self.colors.black)

        self._page_footer(c)
        c.showPage()
    def _format_comp_catalog_id(self, row: Any) -> str:
        bv_src = str(row.get("bv_source", "") or "").strip().lower()
        cid = self._norm_cid_decimal(row.get("catalog_id", ""))
        if not cid:
            return "-"
        if bv_src in ("gaia_bprp", "gaia_teff", "unknown", ""):
            return cid
        if bv_src == "tycho2":
            tycho_id = str(row.get("tycho_id", row.get("tycho2_id", "")) or "").strip()
            if tycho_id and tycho_id.lower() not in ("nan", "none"):
                return f"TYC {tycho_id}"
            return cid
        if bv_src == "apass":
            apass_id = str(row.get("apass_id", "") or "").strip()
            if apass_id and apass_id.lower() not in ("nan", "none"):
                return f"AP {apass_id}"
            return cid
        return cid

    def _proc_csv_dir(self) -> Path | None:
        proc_dir = self.draft_dir / "detrended_aligned" / "lights" / self.obs_group
        if proc_dir.is_dir() and any(proc_dir.glob("proc_*.csv")):
            return proc_dir
        for p in sorted(self.draft_dir.glob("detrended_aligned/lights/*/")):
            if p.is_dir() and any(p.glob("proc_*.csv")):
                return p
        return None

    def _rms_p2p_from_quality_note(self, note: str) -> float:
        m = re.search(r"p2p=([0-9.]+(?:e[-+]?\d+)?)", str(note or ""), flags=re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return float("nan")

    def _proc_flux_rows(self) -> list[tuple[float, list[tuple[str, float]]]]:
        """Per proc CSV (sorted): (median frame time, [(catalog_id, flux) of every comparison-pool row]).

        SPEED-2026-10: read once per report (was once per target: 62 x 97 reads on 534 g). Rows keep the
        CSV order; only ids of the report's comparison table are kept (bounded memory)."""
        cached = getattr(self, "_proc_flux_rows_cache", None)
        if cached is not None:
            return cached
        rows_out: list[tuple[float, list[tuple[str, float]]]] = []
        proc_dir = self._proc_csv_dir()
        pool: set[str] = set()
        if not self.comp_df.empty and "catalog_id" in self.comp_df.columns:
            pool = {c for c in (_norm_cid_decimal(x) for x in self.comp_df["catalog_id"].tolist()) if c}
        if proc_dir is not None and pool:
            for fp in sorted(glob.glob(str(proc_dir / "proc_*.csv"))):
                try:
                    from proc_frame_store import read_proc_csv  # noqa: PLC0415

                    df = read_proc_csv(
                        fp,
                        dtype={"catalog_id": str},
                        usecols=lambda c: c in {"catalog_id", "dao_flux", "flux", "bjd_tdb_mid", "jd_mid", "hjd_mid"},
                    )
                except Exception:  # noqa: BLE001
                    # EXC-0266: T2 -- report/export may omit or misstate (EXCEPT-BULK 2026-07-08)
                    continue
                if df.empty or "catalog_id" not in df.columns:
                    continue
                tcol = next((c for c in ("bjd_tdb_mid", "jd_mid", "hjd_mid") if c in df.columns), None)
                tval = float(pd.to_numeric(df[tcol], errors="coerce").median()) if tcol else float("nan")
                flux_col = "dao_flux" if "dao_flux" in df.columns else "flux"
                vals = df[flux_col].tolist() if flux_col in df.columns else [None] * len(df)
                kept: list[tuple[str, float]] = []
                for raw_id, raw_fx in zip(df["catalog_id"].tolist(), vals, strict=True):
                    cid = _norm_cid_decimal(raw_id)
                    if cid not in pool:
                        continue
                    kept.append((cid, float(pd.to_numeric(raw_fx, errors="coerce"))))
                rows_out.append((tval, kept))
        self._proc_flux_rows_cache = rows_out
        return rows_out

    def _comp_rms_p2p_map_from_proc(self, comp_ids: list[str]) -> dict[str, float]:
        """Report-only Abbe p2p from per-frame dao_flux (same definition as the aperture-photometry stability)."""
        proc_dir = self._proc_csv_dir()
        if proc_dir is None or not comp_ids:
            return {}
        want = {_norm_cid_decimal(x) for x in comp_ids if _norm_cid_decimal(x)}
        flux_by_cid: dict[str, list[tuple[float, float]]] = {c: [] for c in want}
        for tval, kept in self._proc_flux_rows():
            for cid, fx in kept:
                if cid in want and math.isfinite(fx) and fx > 0 and math.isfinite(tval):
                    flux_by_cid.setdefault(cid, []).append((tval, fx))
        out: dict[str, float] = {}
        for cid, pairs in flux_by_cid.items():
            if len(pairs) < 3:
                out[cid] = float("nan")
                continue
            pairs.sort(key=lambda x: x[0])
            mags = np.array([-2.5 * math.log10(fx) for _, fx in pairs if fx > 0], dtype=float)
            if mags.size < 3:
                out[cid] = float("nan")
                continue
            diff = np.diff(mags)
            out[cid] = float(np.std(diff) / math.sqrt(2.0)) if diff.size > 1 else float("nan")
        return out

    def _comp_rms_p2p_map_for_target(self, target_cid: str, comp_ids: list[str]) -> dict[str, float]:
        qpath = self.lc_dir / f"comp_quality_{target_cid}.json"
        parsed: dict[str, float] = {}
        if qpath.is_file():
            try:
                raw = json.loads(qpath.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    from photometry_core import parse_comp_quality_json_map

                    qmap = parse_comp_quality_json_map(raw)
                    for ck, info in qmap.items():
                        nk = _norm_cid_decimal(ck)
                        if not nk:
                            continue
                        v = self._rms_p2p_from_quality_note(info.get("note", ""))
                        if math.isfinite(v):
                            parsed[nk] = v
            except Exception:  # noqa: BLE001
                parsed = {}
        proc_map = self._comp_rms_p2p_map_from_proc(comp_ids)
        for cid in comp_ids:
            nk = _norm_cid_decimal(cid)
            if not nk:
                continue
            if nk not in parsed or not math.isfinite(parsed.get(nk, float("nan"))):
                if nk in proc_map and math.isfinite(proc_map[nk]):
                    parsed[nk] = proc_map[nk]
        return parsed

    def _comp_rows_for_target(self, cid: str) -> tuple[list[list[str]], str]:
        if self.comp_df.empty or not cid:
            return [], ""
        if "_tcid" not in self.comp_df.columns:
            return [], ""
        sub = self.comp_df[self.comp_df["_tcid"] == cid].copy()
        if sub.empty:
            return [], ""
        # Normalize dist column name
        if "_dist_deg" in sub.columns and "dist_deg" not in sub.columns:
            sub["dist_deg"] = sub["_dist_deg"]

        # Optional per-target comp quality from AperturePhotometry (saved as JSON).
        qmap: dict[str, dict[str, str]] = {}
        try:
            qpath = self.lc_dir / f"comp_quality_{cid}.json"
            if qpath.exists():
                import json

                qraw = json.loads(qpath.read_text(encoding="utf-8"))
                if isinstance(qraw, dict):
                    from photometry_core import parse_comp_quality_json_map

                    qmap = parse_comp_quality_json_map(qraw)
        except Exception:  # noqa: BLE001
            qmap = {}
        qmap_str = (
            {k: v["quality"] for k, v in qmap.items()}
            if qmap
            else {}
        )
        excluded_for_note: list[tuple[str, str]] = []
        comp_id_list = [_norm_cid_decimal(r.get("catalog_id", "")) for _, r in sub.iterrows()]
        comp_id_list = [x for x in comp_id_list if x]
        rms_p2p_map = self._comp_rms_p2p_map_for_target(cid, comp_id_list)

        def _fmt(v: Any, nd: int) -> str:
            x = pd.to_numeric(v, errors="coerce")
            return f"{float(x):.{nd}f}" if np.isfinite(x) else "-"

        # Relative weights via apply_comp_w_rel_for_display (excluded -> w_rel=0).
        w_rel_by_row: dict[int, float] = {}
        try:
            if "comp_weight" in sub.columns:
                from photometry_core import apply_comp_w_rel_for_display

                sub_w = apply_comp_w_rel_for_display(sub, qmap_str)
                if "w_rel" in sub_w.columns:
                    for i in range(len(sub_w)):
                        wr = pd.to_numeric(sub_w.iloc[i].get("w_rel"), errors="coerce")
                        w_rel_by_row[i] = float(wr) if math.isfinite(float(wr)) else float("nan")
        except Exception:  # noqa: BLE001
            w_rel_by_row = {}

        def _cid_short(catalog_id: str) -> str:
            digits = "".join(ch for ch in str(catalog_id) if ch.isdigit())
            return digits[-6:] if len(digits) >= 6 else digits or str(catalog_id)[-6:]

        out: list[list[str]] = []
        sub = sub.reset_index(drop=True)
        for i in range(len(sub)):
            r = sub.iloc[i]
            ccid = ""
            try:
                ccid = self._norm_cid_decimal(r.get("catalog_id", ""))
            except Exception:  # noqa: BLE001
                ccid = ""
            q_entry = qmap.get(ccid, {}) if ccid else {}
            q_quality = str(q_entry.get("quality", "") or "").strip().lower()
            q_note = str(q_entry.get("note", "") or "").strip()
            if q_quality == "excluded":
                excluded_for_note.append((_cid_short(ccid), q_note or "excluded"))
                continue
            # 'stav' may be stored under different column names; fallback to comp_quality json.
            stav = ""
            for col in ("stav", "quality", "comp_quality"):
                if col in sub.columns:
                    stav = str(r.get(col, "") or "").strip()
                    break
            if not stav and q_quality:
                if q_quality == "suspect":
                    stav = f"suspect - {q_note}" if q_note else "suspect"
                else:
                    stav = q_quality
            _cts0 = str(r.get("color_tier_src", "") or "").strip().lower()
            _cts_lbl = (
                "BP-RP"
                if _cts0 == "bprp"
                else ("BV->" if _cts0 == "bv_converted" else ("?" if _cts0 == "unknown" else ("-" if not _cts0 else _cts0[:5])))
            )
            bv_src_cell = (
                "G-bp"
                if str(r.get("bv_source", "") or "").strip().lower() == "gaia_bprp"
                else (
                    "G-T"
                    if str(r.get("bv_source", "") or "").strip().lower() == "gaia_teff"
                    else (
                        "AP"
                        if str(r.get("bv_source", "") or "").strip().lower() == "apass"
                        else (
                            "TY"
                            if str(r.get("bv_source", "") or "").strip().lower() == "tycho2"
                            else "?"
                        )
                    )
                )
            )
            tail = [
                _fmt(r.get("bp_rp"), 3),
                _fmt(r.get("delta_bprp_abs"), 3) if "delta_bprp_abs" in sub.columns else "-",
                _cts_lbl,
                _fmt(r.get("dist_deg"), 4),
                str(int(pd.to_numeric(r.get("comp_n_frames"), errors="coerce")))
                if np.isfinite(pd.to_numeric(r.get("comp_n_frames"), errors="coerce"))
                else "-",
                _fmt(r.get("comp_rms"), 4),
                _fmt(rms_p2p_map.get(ccid, float("nan")), 4),
                _fmt(
                    w_rel_by_row.get(i, float("nan")),
                    3,
                ),
                str(r.get("comp_tier", "") or ""),
                stav,
            ]
            row0 = [str(len(out) + 1), self._format_comp_catalog_id(r), _fmt(r.get("mag"), 3)]
            if self._use_bprp_primary:
                out.append(row0 + tail)
            else:
                out.append(row0 + [_fmt(r.get("b_v"), 3), bv_src_cell] + tail)
        excluded_note = ""
        if excluded_for_note:
            parts = ", ".join(f"{short} ({note})" for short, note in excluded_for_note)
            n_ex = len(excluded_for_note)
            excluded_note = (
                f"{n_ex} comp star(s) excluded from ensemble: {parts}"
            )
        return out, excluded_note
    def _report_per_star_page(self, c: "canvas.Canvas", star_data: dict[str, Any]) -> None:
        """STAR-PAGE-M3-2026-10 (layout approved by Milan 2026-10-11): header with the object
        name by source (VSX / exoplanet host / Gaia), trust line; left: light curve with the
        check-star strip (UT top axis) and the comparison-star table; right: facts, target
        cutout (aperture + sky annulus) and the field map; calibration note box at the bottom.
        Content from target_page.TargetPages.page (AP-TAB-2026-10: same facts as the UI tab)."""
        import star_page_m3 as m3  # noqa: PLC0415

        c.setPageSize(self.landscape(self.A4))
        pg = self._pages.page(star_data, obs_group=self.obs_group)
        cid, name, band, rows = pg["cid"], pg["name"], pg["band"], pg["comp_rows"]

        # --- header -------------------------------------------------------------------------
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 15)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 14, name)
        _tw = c.stringWidth(name, self.FONT_BOLD, 15)
        c.setFont(self.FONT_REG, 9)
        c.setFillColor(self.colors.HexColor("#57606a"))
        c.drawString(self.M_LEFT + _tw + 0.6 * self.cm, y - 13, "  |  ".join(pg["subline"]))
        y -= 0.85 * self.cm
        trust = pg["trust"]
        if trust:
            col = {"GREEN": "#1b5e20", "YELLOW": "#9a6700", "RED": "#b71c1c"}.get(trust, "#37474f")
            _ts = self._get_para_style("m3_trust", fontSize=8.5, leading=10, textColor=self.colors.HexColor(col))
            _tp = self.Paragraph(f"<b>TRUST: {self._pdf_escape(trust)}</b>&nbsp;&nbsp; "
                                 f"{self._pdf_escape(pg['trust_reason'][:260])}", _ts)
            _, _th = _tp.wrap(self.USE_W, 3 * self.cm)
            _tp.drawOn(c, self.M_LEFT, y - _th + 8.5)
            y -= _th - 8.5
        c.setFillColor(self.colors.black)
        y -= 0.35 * self.cm
        y_body = y

        # --- geometry -----------------------------------------------------------------------
        LW = 0.655 * self.USE_W
        GAP = 0.4 * self.cm
        RX = self.M_LEFT + LW + GAP
        RW = self.USE_W - LW - GAP
        y_floor = self._layout_y_floor()

        # --- calibration / excluded note box (height first) -----------------------------------
        note_txt = pg["note_text"]
        note_style = self._get_para_style("m3_note", fontSize=7, leading=8.6)
        note_h = 0.0
        note_para = None
        if note_txt:
            note_para = self.Paragraph(self._pdf_escape(note_txt), note_style)
            _, _nh = note_para.wrap(self.USE_W - 0.4 * self.cm, 10 * self.PAGE_H)
            note_h = _nh + 0.55 * self.cm

        # --- comparison table ------------------------------------------------------------------
        hdr = [h.replace("{band}", band or "g") for h in m3.COMP_HEADERS]
        cell = self._get_para_style("m3_cell", fontSize=6.6, leading=7.6, alignment=1)
        tdata: list[list[Any]] = [hdr] + [[self.Paragraph(self._pdf_escape(v), cell) for v in r] for r in rows]
        base = [14, 78, 26, 28, 30, 26, 32, 22, 50, 62]
        cw = [w * LW / sum(base) for w in base]
        t = None
        t_h = 0.0
        if rows:
            t = self.Table(tdata, colWidths=cw, rowHeights=[0.48 * self.cm] + [0.40 * self.cm] * len(rows))
            st = self.TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), self.C_TITLE), ("TEXTCOLOR", (0, 0), (-1, 0), self.colors.white),
                ("FONTNAME", (0, 0), (-1, 0), self.FONT_BOLD), ("FONTSIZE", (0, 0), (-1, 0), 7),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, self.colors.HexColor("#d0d7de")),
            ])
            for i, r in enumerate(rows, start=1):
                bg = "#fff8c5" if r[-1].startswith("suspect") else ("#ffffff" if i % 2 else "#f6f8fa")
                st.add("BACKGROUND", (0, i), (-1, i), self.colors.HexColor(bg))
            t.setStyle(st)
            _, t_h = t.wrap(LW, 10 * self.PAGE_H)
        foot_h = 0.38 * self.cm if rows else 0.0

        # --- light curve height: what is left on the left column -------------------------------
        avail = y_body - y_floor - note_h - t_h - foot_h - 0.35 * self.cm
        lc_h = float(np.clip(avail, 6.0 * self.cm, 11.5 * self.cm))
        table_on_next_page = avail < 6.0 * self.cm

        lc_csv = pg["lc_csv"]
        lc_img = None
        try:
            if lc_csv.is_file():
                _png = self.cache_dir / f"m3_lc_{cid}.png"
                lc_img = m3.plot_lc_check(lc_csv, _png, check_csv=pg["check_csv"], uncalibrated=pg["uncalibrated"],
                                          gspc_band=band, show_gspc=pg["show_gspc"],
                                          figsize=(10.0, max(4.2, 10.0 * lc_h / LW)))
        except Exception as exc:  # noqa: BLE001
            logging.warning("[PDF] M3 light curve %s: %s", cid, exc)
            lc_img = None
        if lc_img is not None and Path(lc_img).exists():
            self._draw_image_fit(c, Path(lc_img), self.M_LEFT, y_body, LW, lc_h, image_type="lc")
        else:
            c.setFont(self.FONT_REG, 9)
            c.drawString(self.M_LEFT + 0.5 * self.cm, y_body - lc_h / 2, "Light curve not available")

        # --- right column: facts | cutout, then field map ---------------------------------------
        fx_w = 0.62 * RW
        lab_style = self._get_para_style("m3_flab", fontSize=6.6, leading=7.8, textColor=self.colors.HexColor("#57606a"))
        val_style = self._get_para_style("m3_fval", fontSize=6.6, leading=7.8)
        warn_style = self._get_para_style("m3_fwarn", fontSize=6.6, leading=7.8, textColor=self.colors.HexColor("#bc4c00"))
        ftab = self.Table(
            [[self.Paragraph(self._pdf_escape(a), lab_style),
              self.Paragraph(self._pdf_escape(b), warn_style if a in pg["warn"] else val_style)]
             for a, b in pg["facts"]],
            colWidths=[0.34 * fx_w, 0.66 * fx_w],
        )
        ftab.setStyle(self.TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 1),
                                       ("RIGHTPADDING", (0, 0), (-1, -1), 1), ("TOPPADDING", (0, 0), (-1, -1), 0.6),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6)]))
        _, f_h = ftab.wrap(fx_w, 10 * self.PAGE_H)
        ftab.drawOn(c, RX, y_body - f_h)
        cut_w = RW - fx_w - 0.15 * self.cm
        try:
            _ms = self._masterstar_data()
            if (_ms is not None and np.isfinite(pg["x"]) and np.isfinite(pg["r_ap"])
                    and np.isfinite(pg["fwhm_night"])):
                cut_png = m3.render_cutout(_ms, pg["x"], pg["y"], pg["r_ap"], pg["r_in"], pg["r_out"],
                                           self.cache_dir / f"m3_cut_{cid}.png",
                                           neighbours=self._pages.field_neighbours(pg["x"], pg["y"], pg["r_out"], exclude_id=cid))
                if cut_png is not None:
                    c.setFont(self.FONT_REG, 6)
                    c.setFillColor(self.colors.HexColor("#57606a"))
                    c.drawString(RX + fx_w + 0.15 * self.cm, y_body - 0.05 * self.cm, "aperture / sky annulus")
                    self._draw_image_fit(c, cut_png, RX + fx_w + 0.15 * self.cm, y_body - 0.2 * self.cm, cut_w, cut_w,
                                         image_type="field")
        except Exception as exc:  # noqa: BLE001
            logging.warning("[PDF] M3 cutout %s: %s", cid, exc)
        y_fm = y_body - max(f_h, cut_w + 0.2 * self.cm) - 0.25 * self.cm
        fi = star_data.get("field_img")
        fm_h = y_fm - y_floor - note_h - 0.2 * self.cm
        if fi and Path(str(fi)).exists() and fm_h > 2.0 * self.cm:
            self._draw_image_fit(c, Path(str(fi)), RX, y_fm, RW, fm_h, image_type="field")

        # --- comparison table under the light curve --------------------------------------------
        y_t = y_body - lc_h - 0.25 * self.cm
        if t is not None and table_on_next_page:
            self._draw_m3_note_box(c, note_para, note_h)
            self._page_footer(c)
            c.showPage()
            c.setPageSize(self.landscape(self.A4))
            c.setFont(self.FONT_BOLD, 12)
            c.drawString(self.M_LEFT, self.PAGE_H - self.M_TOP - 0.5 * self.cm, f"{name} (continued)")
            y_t = self.PAGE_H - self.M_TOP - 0.9 * self.cm
            note_para, note_h = None, 0.0
        if t is not None:
            t.drawOn(c, self.M_LEFT, y_t - t_h)
            c.setFont(self.FONT_OBL, 5.4)
            c.setFillColor(self.colors.HexColor("#57606a"))
            c.drawString(self.M_LEFT, y_t - t_h - 0.25 * self.cm, m3.COMP_FOOT)
            c.setFillColor(self.colors.black)
        else:
            c.setFont(self.FONT_REG, 9)
            c.drawString(self.M_LEFT, y_t - 0.4 * self.cm, "Comparison star table not available")
        self._draw_m3_note_box(c, note_para, note_h)
        self._page_footer(c)
        c.showPage()

    def _report_epsf_star_page(self, c: "canvas.Canvas", star_data: dict[str, Any]) -> None:
        """EPSF-PDF-2026-10 (Milan 2026-10-19): the page right after a target's light-curve page when the
        target has an ePSF light curve - aperture and ePSF curves on one axis, their difference per epoch,
        and the ePSF facts. Nothing when the target has no usable ePSF curve."""
        import star_page_m3 as m3  # noqa: PLC0415

        cid = str(star_data.get("catalog_id") or "")
        try:
            data = m3.epsf_page_data(self.lc_dir, self.platesolve_dir, cid)
        except Exception as exc:  # noqa: BLE001
            logging.warning("[PDF] ePSF page %s: %s", cid, exc)
            return
        if data is None:
            return
        c.setPageSize(self.landscape(self.A4))
        pg = self._pages.page(star_data, obs_group=self.obs_group)
        name = pg["name"]
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 15)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 14, name)
        _tw = c.stringWidth(name, self.FONT_BOLD, 15)
        c.setFont(self.FONT_REG, 9)
        c.setFillColor(self.colors.HexColor("#57606a"))
        c.drawString(self.M_LEFT + _tw + 0.6 * self.cm, y - 13, "  |  ".join(pg["subline"]))
        y -= 0.85 * self.cm
        c.setFont(self.FONT_BOLD, 9)
        c.setFillColor(self.colors.HexColor("#bc4c00"))
        c.drawString(self.M_LEFT, y + 2, "ePSF PHOTOMETRY (internal)")
        c.setFillColor(self.colors.black)
        y -= 0.45 * self.cm

        LW = 0.655 * self.USE_W
        GAP = 0.4 * self.cm
        RX = self.M_LEFT + LW + GAP
        RW = self.USE_W - LW - GAP
        note = ("Internal ePSF light curve, for comparison with the aperture light curve of the previous page; the "
                "AAVSO / VarAstro exports use the aperture photometry. Both curves are differential; the ePSF ensemble is "
                "the aperture ensemble without the comparison stars that have no ePSF measurement (count above); each "
                "ensemble weights its stars by its own method's scatter, so the ePSF curve is "
                "drawn at the aperture level (median over the common epochs). Errors: aperture from the CCD equation, "
                "ePSF from the fit (+ blend term) - both with the ensemble, scintillation and the night's systematic "
                "floor.")
        note_para = self.Paragraph(self._pdf_escape(note), self._get_para_style("m3_note", fontSize=7, leading=8.6))
        _, _nh = note_para.wrap(self.USE_W - 0.4 * self.cm, 10 * self.PAGE_H)
        note_h = _nh + 0.55 * self.cm
        lc_h = float(np.clip(y - self._layout_y_floor() - note_h - 0.4 * self.cm, 6.0 * self.cm, 15.5 * self.cm))
        img = None
        try:
            img = m3.plot_lc_epsf(data["df"], self.cache_dir / f"m3_epsf_{cid}.png",
                                  figsize=(10.0, max(4.2, 10.0 * lc_h / LW)))
        except Exception as exc:  # noqa: BLE001
            logging.warning("[PDF] ePSF light curve %s: %s", cid, exc)
        if img is not None and Path(img).exists():
            self._draw_image_fit(c, Path(img), self.M_LEFT, y, LW, lc_h, image_type="lc")
        lab = self._get_para_style("m3_flab", fontSize=6.6, leading=7.8, textColor=self.colors.HexColor("#57606a"))
        val = self._get_para_style("m3_fval", fontSize=6.6, leading=7.8)
        ftab = self.Table([[self.Paragraph(self._pdf_escape(a), lab), self.Paragraph(self._pdf_escape(b), val)]
                           for a, b in data["facts"]], colWidths=[0.30 * RW, 0.70 * RW])
        ftab.setStyle(self.TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 1),
                                       ("RIGHTPADDING", (0, 0), (-1, -1), 1), ("TOPPADDING", (0, 0), (-1, -1), 0.8),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8)]))
        _, f_h = ftab.wrap(RW, 10 * self.PAGE_H)
        ftab.drawOn(c, RX, y - f_h)
        self._draw_m3_note_box(c, note_para, note_h)
        self._page_footer(c)
        c.showPage()

    def _draw_m3_note_box(self, c: "canvas.Canvas", para: Any, h: float) -> None:
        """Calibration / exclusion note in an orange box above the footer (M3)."""
        if para is None or h <= 0:
            return
        y0 = self._layout_y_floor()
        c.setFillColor(self.colors.HexColor("#fff1e5"))
        c.setStrokeColor(self.colors.HexColor("#bc4c00"))
        c.setLineWidth(0.6)
        c.rect(self.M_LEFT, y0, self.USE_W, h - 0.15 * self.cm, fill=1, stroke=1)
        c.setFillColor(self.colors.black)
        _, ph = para.wrap(self.USE_W - 0.4 * self.cm, 10 * self.PAGE_H)
        para.drawOn(c, self.M_LEFT + 0.2 * self.cm, y0 + (h - 0.15 * self.cm - ph) / 2)

    def _report_summary_table(self, c: "canvas.Canvas") -> None:
        def _cell_txt(val: Any) -> str:
            return str(val if val is not None else "").strip()

        def _zone_row_fill(zf: str) -> Any:
            z = str(zf or "").strip().lower()
            if z == "linear":
                return self.colors.HexColor("#eaf7eb")
            if "catalog" in z:
                return self.colors.HexColor("#fff6e0")
            return self.colors.white

        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 14)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.5 * self.cm, "Summary of all stars (vsx_type order, then lc_rms)")
        c.setFillColor(self.colors.black)
        y -= 1.0 * self.cm

        id_key = "catalog_id" if "catalog_id" in self.summary_df.columns else ("_cid" if "_cid" in self.summary_df.columns else None)
        mag_key = next((k for k in ("lc_median_mag", "mag") if k in self.summary_df.columns), None)
        tail_want = ("vsx_name", "vsx_type", "lc_rms", "n_good_comp", "n_points", "merr_med", "lc_quality_flag")
        if not id_key:
            self._page_footer(c)
            c.showPage()
            return

        cols: list[str] = [str(id_key)]
        if mag_key:
            cols.append(mag_key)
        for cn2 in tail_want:
            if cn2 not in cols:
                cols.append(cn2)

        base_cols = [c for c in cols if c in self.summary_df.columns]
        work = self.summary_df[base_cols].copy()
        if "n_points" in cols and "n_points" not in work.columns:
            work["n_points"] = 0
        if "merr_med" in cols and "merr_med" not in work.columns:
            work["merr_med"] = float("nan")
        if "lc_quality_flag" in cols and "lc_quality_flag" in self.summary_df.columns:
            work["lc_quality_flag"] = self.summary_df["lc_quality_flag"].values
        elif "lc_quality_flag" in cols:
            work["lc_quality_flag"] = ""
        work["lc_rms"] = pd.to_numeric(work.get("lc_rms"), errors="coerce")
        if mag_key and mag_key in work.columns:
            work[mag_key] = pd.to_numeric(work.get(mag_key), errors="coerce")
        id_series = work[id_key].map(_norm_cid_decimal)
        for i, cid_v in enumerate(id_series.astype(str)):
            st = self._lc_stats_by_cid.get(str(cid_v).strip(), {})
            if "n_points" in work.columns:
                work.at[work.index[i], "n_points"] = int(st.get("n_points", 0) or 0)
            if "merr_med" in work.columns:
                work.at[work.index[i], "merr_med"] = st.get("merr_med", float("nan"))

        hdr = [("catalog_id" if cn_h == "_cid" else str(cn_h)) for cn_h in cols]
        id_style = self._get_para_style("sum_id", fontSize=6.5)
        name_style = self._get_para_style("sum_name", fontSize=7)
        cell_style = self._get_para_style("sum_cell", fontSize=7)
        data: list[list[Any]] = [hdr]
        for _, r in work.iterrows():
            row_out: list[Any] = []
            for cn in cols:
                if cn == id_key:
                    row_out.append(self.Paragraph(self._pdf_id_display(r.get(cn)), id_style))
                elif cn == "vsx_name":
                    row_out.append(
                        self.Paragraph(self._pdf_id_display(r.get(cn), break_digits=False), name_style)
                    )
                elif mag_key and cn == mag_key:
                    x = pd.to_numeric(r.get(cn), errors="coerce")
                    row_out.append(
                        self.Paragraph(
                            self._pdf_escape(f"{float(x):.3f}" if np.isfinite(float(x)) else "-"),
                            cell_style,
                        )
                    )
                elif cn == "lc_rms" or cn == "merr_med":
                    x = pd.to_numeric(r.get(cn), errors="coerce")
                    row_out.append(
                        self.Paragraph(
                            self._pdf_escape(f"{float(x):.4f}" if np.isfinite(float(x)) else "-"),
                            cell_style,
                        )
                    )
                elif cn == "n_points":
                    x = int(pd.to_numeric(r.get(cn), errors="coerce") or 0)
                    row_out.append(self.Paragraph(self._pdf_escape(str(x)), cell_style))
                elif cn == "lc_quality_flag":
                    row_out.append(self.Paragraph(self._pdf_escape(_cell_txt(r.get(cn)) or "-"), cell_style))
                else:
                    row_out.append(self.Paragraph(self._pdf_escape(_cell_txt(r.get(cn))), cell_style))
            data.append(row_out)

        _width_pct: dict[str, float] = {
            "catalog_id": 0.20,
            "_cid": 0.20,
            "lc_median_mag": 0.07,
            "mag": 0.07,
            "vsx_name": 0.17,
            "vsx_type": 0.06,
            "lc_rms": 0.06,
            "n_good_comp": 0.06,
            "n_points": 0.05,
            "merr_med": 0.05,
            "lc_quality_flag": 0.07,
            "zone_flag": 0.08,
            "bp_rp": 0.06,
        }
        fracs = [float(_width_pct.get(cn_u, 0.08)) for cn_u in cols]
        fsum = float(sum(fracs)) or 1.0
        col_widths = [self.USE_W * (f / fsum) for f in fracs]

        base_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), self.C_TITLE),
            ("TEXTCOLOR", (0, 0), (-1, 0), self.colors.white),
            ("FONTNAME", (0, 0), (-1, 0), self.FONT_BOLD),
            ("FONTSIZE", (0, 0), (-1, 0), 7.5),
            ("FONTNAME", (0, 1), (-1, -1), self.FONT_REG),
            ("FONTSIZE", (0, 1), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.25, self.colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]

        HEADER_H = 0.60 * self.cm
        MIN_ROW_H = 0.44 * self.cm
        table_bottom_y = self.M_BOTTOM + 1.0 * self.cm

        def _row_height(row_cells: list[Any]) -> float:
            rh = MIN_ROW_H
            for j, cell in enumerate(row_cells):
                if isinstance(cell, self.Paragraph):
                    rh = max(rh, self._para_row_height(cell, col_widths[j], min_h=MIN_ROW_H))
            return rh

        header_row = data[0]
        data_rows = data[1:]
        chunk_start = 0
        page_idx = 0
        while chunk_start < len(data_rows):
            if page_idx > 0:
                y = self.PAGE_H - self.M_TOP
                c.setFont(self.FONT_BOLD, 14)
                c.setFillColor(self.C_TITLE)
                c.drawString(
                    self.M_LEFT,
                    y - 0.5 * self.cm,
                    "Summary of all stars (vsx_type order, then lc_rms) - continued",
                )
                c.setFillColor(self.colors.black)
                y -= 1.0 * self.cm
            else:
                y = self.PAGE_H - self.M_TOP - 1.0 * self.cm

            avail_h = y - table_bottom_y
            chunk_rows: list[list[Any]] = [header_row]
            chunk_heights: list[float] = [HEADER_H]
            i0 = chunk_start
            while chunk_start < len(data_rows):
                rh = _row_height(data_rows[chunk_start])
                if len(chunk_rows) > 1 and sum(chunk_heights) + rh > avail_h:
                    break
                chunk_rows.append(data_rows[chunk_start])
                chunk_heights.append(rh)
                chunk_start += 1

            sty = self.TableStyle(list(base_cmds))
            if "zone_flag" in work.columns:
                for j in range(1, len(chunk_rows)):
                    abs_i = i0 + j - 1
                    if 0 <= abs_i < len(work):
                        zv = str(work.iloc[abs_i].get("zone_flag", "") or "")
                        sty.add("BACKGROUND", (0, j), (-1, j), _zone_row_fill(zv))
            elif "lc_quality_flag" in work.columns:
                for j in range(1, len(chunk_rows)):
                    abs_i = i0 + j - 1
                    if 0 <= abs_i < len(work):
                        qv = str(work.iloc[abs_i].get("lc_quality_flag", "") or "").strip().lower()
                        if qv == "good":
                            sty.add("BACKGROUND", (0, j), (-1, j), self.colors.HexColor("#eaf7eb"))
                        elif qv in ("noisy", "noisy_moon", "short_baseline", "saturated"):
                            sty.add("BACKGROUND", (0, j), (-1, j), self.colors.HexColor("#fff6e0"))
            t = self.Table(chunk_rows, colWidths=col_widths, rowHeights=chunk_heights)
            t.setStyle(sty)
            th = float(sum(chunk_heights))
            table_y = table_bottom_y
            self._bounds_check(self.M_LEFT, table_y, self.USE_W, th)
            t.wrap(self.USE_W, th)
            t.drawOn(c, self.M_LEFT, table_y)
            self._page_footer(c)
            if chunk_start < len(data_rows):
                c.showPage()
            page_idx += 1
        c.showPage()

    def _report_hrd_unavailable_page(self, c: "canvas.Canvas", reason: str) -> None:
        """Emit a visible placeholder when the HRD section cannot be rendered."""
        if getattr(c, "_code", None):  # FIX-BLANKPAGE-2026-10: only when the current page has content
            c.showPage()
        y0 = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 14)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y0 - 0.5 * self.cm, "Field astrophysics (Hertzsprung-Russell diagram)")
        c.setFillColor(self.colors.black)
        c.setFont(self.FONT_REG, 11)
        c.drawString(self.M_LEFT, y0 - 1.1 * self.cm, f"HRD panel unavailable: {reason}")
        self._page_footer(c)
        c.showPage()

    def _report_hrd_page(self, c: "canvas.Canvas") -> None:
        """HRD + highlighted stars table (best effort)."""
        import sqlite3

        try:
            from config import AppConfig

            from hrd_analysis import (
                HRD_CAPTION,
                HRD_EMPTY_FIELD_MSG,
                annotate_field_image,
                build_hrd_dataframe,
                ensure_clean_field_background_png,
                get_top_interesting_stars,
                HRD_PARALLAX_MIN_MAS,
                HRD_PARALLAX_SNR_MIN,
                plot_hrd_matplotlib,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logging.error("PDF HRD: import failed (%s)", exc)
            self._report_hrd_unavailable_page(c, f"import failed ({exc})")
            return

        gdb = Path(str(getattr(AppConfig(), "gaia_db_path", "") or "").strip())
        cfg = AppConfig()
        ms_csv = self.platesolve_dir / "masterstars_full_match.csv"
        if not ms_csv.is_file():
            logging.error("PDF HRD: missing %s", ms_csv.name)
            self._report_hrd_unavailable_page(c, f"missing {ms_csv.name}")
            return
        if not gdb.is_file():
            logging.error("PDF HRD: gaia_db_path not configured or file missing")
            self._report_hrd_unavailable_page(c, "Gaia SQLite database not available")
            return

        hrd_png = self.cache_dir / "hrd_field_summary.png"
        top = pd.DataFrame()
        hrd_df = pd.DataFrame()
        _hrd_build_errors = (
            OSError,
            sqlite3.Error,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            RuntimeError,
            ArithmeticError,
        )
        try:
            pmin, psnr = HRD_PARALLAX_MIN_MAS, HRD_PARALLAX_SNR_MIN
            hrd_df = build_hrd_dataframe(ms_csv, gdb, parallax_min_mas=pmin, parallax_snr_min=psnr)
            if hrd_df.empty:
                self._report_hrd_unavailable_page(c, "no field stars in masterstars catalog")
                return
            enrich_cache = self.platesolve_dir / "_hrd_cache" / "hrd_enrich.json"
            top = get_top_interesting_stars(hrd_df, cfg=cfg, cache_path=enrich_cache)
            plot_hrd_matplotlib(hrd_df, top, output_path=hrd_png, obs_group=self.obs_group)
        except _hrd_build_errors as exc:
            logging.exception("PDF HRD: build/plot failed")
            self._report_hrd_unavailable_page(c, str(exc))
            return

        if getattr(c, "_code", None):  # FIX-BLANKPAGE-2026-10: the previous section already ended its page
            c.showPage()
        y0 = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 14)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y0 - 0.5 * self.cm, "Field astrophysics (Hertzsprung-Russell diagram)")
        c.setFillColor(self.colors.black)
        y = y0 - 1.1 * self.cm
        hrd_plot_bottom = y

        if hrd_png.is_file():
            try:
                hmw, hjq = self._IMAGE_PDF_SETTINGS["hrd"]
                hbuf, _hf = self._compress_image_for_pdf(hrd_png, hmw, hjq)
                ir = self.ImageReader(hbuf)
            except Exception:  # noqa: BLE001
                ir = self.ImageReader(str(hrd_png))
            iw, ih = ir.getSize()
            max_w = self.USE_W * 0.58
            max_h = (self.PAGE_H - self.M_TOP - self.M_BOTTOM - 3.0 * self.cm) * 0.62
            sc = min(max_w / float(iw), max_h / float(ih))
            dw, dh = iw * sc, ih * sc
            c.drawImage(ir, self.M_LEFT, y - dh, width=dw, height=dh, mask="auto")
            hrd_plot_bottom = y - dh
            y -= dh + 0.25 * self.cm

            ann_png: Path | None = None
            is_empty = bool(top.get("_empty_field", pd.Series([False])).any()) if not top.empty else True
            plot_stars = (
                top[~top.get("_empty_field", False).astype(bool)] if not is_empty and not top.empty else pd.DataFrame()
            )
            if not plot_stars.empty:
                try:
                    bg_png, from_fits = ensure_clean_field_background_png(
                        self.platesolve_dir,
                        self.photometry_dir,
                        cache_dir=self.cache_dir,
                    )
                    if bg_png is not None:
                        ann_out = self.cache_dir / "hrd_field_annotated_report.png"
                        nss_on = bool(getattr(cfg, "hrd_nss_category_enabled", False))
                        ann_result = annotate_field_image(
                            bg_png,
                            plot_stars,
                            hrd_df,
                            platesolve_dir=self.platesolve_dir,
                            output_path=ann_out,
                            nss_category_enabled=nss_on,
                            png_from_fits=from_fits,
                            invert=True,
                        )
                        if ann_result is not None and ann_result.is_file():
                            ann_png = ann_result
                except Exception:  # noqa: BLE001
                    logging.exception("PDF HRD: annotated field image failed")

            if ann_png is not None:
                try:
                    fmw, fjq = self._IMAGE_PDF_SETTINGS["hrd_field"]
                    fbuf, _ff = self._compress_image_for_pdf(ann_png, fmw, fjq)
                    air = self.ImageReader(fbuf)
                except Exception:  # noqa: BLE001
                    air = self.ImageReader(str(ann_png))
                aiw, aih = air.getSize()
                gap = 0.35 * self.cm
                max_w_ann = max(1.0, self.USE_W - dw - gap)
                max_h_ann = y0 - 1.1 * self.cm - hrd_plot_bottom
                if max_h_ann < 2.0 * self.cm:
                    max_h_ann = (self.PAGE_H - self.M_TOP - self.M_BOTTOM - 3.0 * self.cm) * 0.62
                asc = min(max_w_ann / float(aiw), max_h_ann / float(aih))
                adw, adh = aiw * asc, aih * asc
                ax = self.M_LEFT + dw + gap
                ay = y0 - 1.1 * self.cm - adh
                c.drawImage(air, ax, ay, width=adw, height=adh, mask="auto")
                c.setFont(self.FONT_REG, 7)
                c.setFillColor(self.colors.HexColor("#555555"))
                cap = "Extreme objects marked on the MASTERSTAR field"
                c.drawString(ax, ay - 0.28 * self.cm, cap)
                c.setFillColor(self.colors.black)
                # HRD-PAGE-2026-10: the HRD caption goes below the lower of the two images
                y = min(y, ay - 0.28 * self.cm - 0.45 * self.cm)

        c.setFont(self.FONT_REG, 7)
        c.setFillColor(self.colors.HexColor("#555555"))
        approx_chars = max(40, int(self.USE_W / (7 * 0.45)))
        for line in textwrap.wrap(HRD_CAPTION, width=approx_chars) or [HRD_CAPTION]:
            c.drawString(self.M_LEFT, y, line)
            y -= 0.32 * self.cm
        y -= 0.15 * self.cm
        c.setFillColor(self.colors.black)

        _enrich_skip_note = ""
        from hrd_enrich import load_last_enrich  # noqa: PLC0415

        _sum = load_last_enrich(self.platesolve_dir / "_hrd_cache" / "hrd_enrich.json")
        _sr = _sum.get("enrich_skip_reason") or _sum.get("skip_reason")
        if _sr:
            _enrich_skip_note = f"enrichment unavailable: {_sr}"
        if _enrich_skip_note:
            c.setFont(self.FONT_REG, 8)
            c.setFillColor(self.colors.HexColor("#884400"))
            c.drawString(self.M_LEFT, y, _enrich_skip_note)
            y -= 0.45 * self.cm
            c.setFillColor(self.colors.black)

        is_empty = bool(top.get("_empty_field", pd.Series([False])).any()) if not top.empty else True
        if is_empty:
            tab_cols = ["category"]
            rows_t: list[list[str]] = [tab_cols, [HRD_EMPTY_FIELD_MSG]]
            col_ws = [self.USE_W]
            rt = self.Table(rows_t, colWidths=col_ws, repeatRows=0)
            rt.setStyle(
                self.TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), self.FONT_REG),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ("GRID", (0, 0), (-1, -1), 0.2, self.colors.HexColor("#cccccc")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            _, th = rt.wrap(self.USE_W, y - self.M_BOTTOM)
            rt.drawOn(c, self.M_LEFT, y - th)
            y -= th + 0.3 * self.cm
        elif top is not None and not top.empty:
            # HRD-PAGE-2026-10 (Milan 2026-10-11: text overlapped): wrapped Paragraph cells,
            # widths as fractions of USE_W so the table always fits the page.
            rows_t, col_ws = self._hrd_table_rows(top)
            rt = self.Table(rows_t, colWidths=col_ws, repeatRows=1)
            rt.setStyle(
                self.TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), self.C_TITLE),
                        ("GRID", (0, 0), (-1, -1), 0.2, self.colors.HexColor("#cccccc")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 2),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                        ("TOPPADDING", (0, 0), (-1, -1), 1),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ]
                )
            )
            _, th = rt.wrap(self.USE_W, y - self.M_BOTTOM)
            if th > y - self.M_BOTTOM:
                # too tall for the rest of the page: split over pages
                parts = rt.split(self.USE_W, y - self.M_BOTTOM) or [rt]
                for i_p, part in enumerate(parts):
                    if i_p > 0:
                        self._page_footer(c)
                        c.showPage()
                        y = self.PAGE_H - self.M_TOP
                    _, th_p = part.wrap(self.USE_W, y - self.M_BOTTOM)
                    part.drawOn(c, self.M_LEFT, y - th_p)
                    y -= th_p + 0.3 * self.cm
            else:
                rt.drawOn(c, self.M_LEFT, y - th)
                y -= th + 0.3 * self.cm

        self._page_footer(c)
        c.showPage()
        if not is_empty and top is not None and not top.empty:
            self._report_hrd_extreme_details(c, top)

    #: HRD-PAGE-2026-10: (header, key(s), fraction of USE_W); fractions sum to 1.0
    _HRD_TABLE_SPEC: tuple[tuple[str, tuple[str, ...], float], ...] = (
        ("Category", ("category",), 0.17),
        ("Gaia DR3", ("catalog_id",), 0.15),
        ("SIMBAD (type)", ("simbad_id",), 0.19),
        ("SpT", ("sp_type_raw",), 0.07),
        ("G", ("mag_g",), 0.05),
        ("M_G", ("abs_mag_g",), 0.05),
        ("BP-RP", ("bp_rp",), 0.06),
        ("Teff [K]", ("teff",), 0.06),
        ("logg", ("logg",), 0.05),
        ("d [pc]", ("dist_pc", "dist_gspphot"), 0.08),
        ("src", ("src",), 0.07),
    )

    def _hrd_table_rows(self, top: pd.DataFrame) -> tuple[list[list[Any]], list[float]]:
        """Header + rows of wrapped Paragraph cells and column widths summing to USE_W.

        d [pc]: 1/parallax when the parallax is reliable, otherwise Gaia GSP-Phot distance
        marked "(GSP)"."""
        hdr_sty = self._get_para_style("hrd_th", fontName=self.FONT_BOLD, fontSize=6.5, textColor=self.colors.white)
        cell_sty = self._get_para_style("hrd_td", fontSize=6.5, leading=7.6)
        na = {"", "nan", "none", "n/a", "\u2014", "-"}

        def _val(rr: pd.Series, keys: tuple[str, ...]) -> str:
            for i_k, k in enumerate(keys):
                s = str(rr.get(k, "") or "").strip()
                if s.lower() not in na:
                    return f"{s} (GSP)" if i_k > 0 and k == "dist_gspphot" else s
            return "\u2014"

        rows: list[list[Any]] = [[self.Paragraph(self._pdf_escape(h), hdr_sty) for h, _k, _f in self._HRD_TABLE_SPEC]]
        work = top.loc[~top.get("_empty_field", pd.Series(False, index=top.index)).astype(bool)]
        for _, rr in work.iterrows():
            rows.append([
                self.Paragraph(self._pdf_escape(_val(rr, keys)), cell_sty)  # CJK wrap breaks long ids
                for _h, keys, _f in self._HRD_TABLE_SPEC
            ])
        widths = [float(self.USE_W) * f for _h, _k, f in self._HRD_TABLE_SPEC]
        return rows, widths

    def _report_hrd_extreme_details(self, c: "canvas.Canvas", top: pd.DataFrame) -> None:
        """Follow-on page(s): per-object catalog/astrophysics detail blocks."""
        from hrd_analysis import (
            HRD_DETAILS_DISTANCE_CAVEAT,
            build_hrd_detail_line,
            hrd_detail_header_name,
        )

        work = top.loc[~top.get("_empty_field", False).astype(bool)].head(17)
        if work.empty:
            return

        y = float(self.PAGE_H - self.M_TOP)
        c.setFont(self.FONT_BOLD, 14)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.5 * self.cm, "Extreme objects -- details")
        c.setFillColor(self.colors.black)
        y -= 1.0 * self.cm

        y = self._draw_flow_lines(
            c,
            y,
            [(HRD_DETAILS_DISTANCE_CAVEAT, self.FONT_REG, 7.0)],
            line_step=8.0,
            paginate=True,
        )
        y -= 0.15 * self.cm

        for _, rr in work.iterrows():
            row_dict = rr.to_dict()
            header = (
                f"{hrd_detail_header_name(row_dict)} -- "
                f"{row_dict.get('category', '')} -- "
                f"{row_dict.get('ident', 'candidate')}"
            )
            detail = build_hrd_detail_line(row_dict)
            y = self._draw_flow_lines(
                c,
                y,
                [(header, self.FONT_BOLD, 8.0)],
                line_step=10.0,
                paginate=True,
            )
            y = self._draw_flow_lines(
                c,
                y,
                [(detail, self.FONT_REG, 7.0)],
                line_step=8.5,
                paginate=True,
            )
            y -= 0.12 * self.cm

        self._page_footer(c)
        c.showPage()

    def _report_field_map(self, c: "canvas.Canvas") -> None:
        fmp: Path | None = None
        from field_map import field_map_path  # noqa: PLC0415

        fmp = field_map_path(self.photometry_dir)
        if fmp is None:
            logging.warning("PDF field map: field_map.jpg not found under %s - skipping page", self.photometry_dir)
            return
        c.setPageSize(self.landscape(self.A4))
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 16)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.35 * self.cm, "Field Map")
        c.setFont(self.FONT_REG, 9)
        c.setFillColor(self.colors.HexColor("#555555"))
        c.drawString(self.M_LEFT, y - 0.85 * self.cm, f"Observation group: {self.obs_group}")
        c.setFillColor(self.colors.black)
        try:
            fmw, fjq = self._IMAGE_PDF_SETTINGS["field_map"]
            fbuf, _ff = self._compress_image_for_pdf(fmp, fmw, fjq)
            ir = self.ImageReader(fbuf)
        except Exception:  # noqa: BLE001
            ir = self.ImageReader(str(fmp))
        iw, ih = ir.getSize()
        if not iw or not ih:
            self._page_footer(c)
            c.showPage()
            return
        header_band = 1.15 * self.cm
        avail_w = self.USE_W
        avail_h = self.PAGE_H - self.M_TOP - self.M_BOTTOM - header_band
        sc = min(avail_w / float(iw), avail_h / float(ih))
        dw, dh = float(iw) * sc, float(ih) * sc
        x0 = self.M_LEFT + (avail_w - dw) / 2.0
        y_bottom = self.M_BOTTOM + max(0.0, (avail_h - dh) / 2.0)
        c.drawImage(ir, x0, y_bottom, width=dw, height=dh, mask="auto")
        self._page_footer(c)
        c.showPage()
    def _report_variability_v2(self, c: "canvas.Canvas") -> None:
        """Variability pages from variability_field (VAR-2026-10).

        Page 1: method + hockey-stick. Page 2: candidates table, field with candidates.
        Per candidate: night light curve; TESS block only when ``tess_enabled`` is on (or
        the user added the star from the UI); a period is printed only when RELIABLE.
        """
        import json as _json  # noqa: PLC0415

        from variability_plots import candidate_lc_png, field_png, hockey_png  # noqa: PLC0415

        phot = self.photometry_dir
        try:
            meta = _json.loads((phot / "variability_meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        fcsv = phot / "variability_field.csv"
        c.setPageSize(self.landscape(self.A4))
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 14)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.45 * self.cm, "Variability - new variable-star candidates")
        c.setFillColor(self.colors.black)
        y -= 1.0 * self.cm
        if meta.get("status") != "ok" or not fcsv.is_file():
            c.setFont(self.FONT_REG, 10)
            c.drawString(self.M_LEFT, y, f"Variability search not available ({meta.get('status', 'not run')}).")
            self._page_footer(c)
            c.showPage()
            return
        field = pd.read_csv(fcsv, dtype={"catalog_id": str}, low_memory=False)
        nm = meta.get("noise_model") or {}
        sty = self._get_para_style("var_v2_body", fontName=self.FONT_REG, fontSize=8.5, leading=11)
        txt = (
            f"Every star measured on at least 90 % of the {meta.get('n_frames')} frames ({meta.get('n_stars')} stars) is put on a "
            "common frame zero point (Honeycutt 1992). Its robust scatter (1.4826 MAD) is compared with the scatter expected from "
            f"its CCD error, scaled and floored by the field itself (a = {float(nm.get('a') or 0):.2f}, floor = "
            f"{1000 * float(nm.get('floor_mag') or 0):.1f} mmag); the second index is 1/eta (von Neumann ratio; smooth = variable; tail from a skew-normal fitted to the field) against stars of similar "
            f"brightness (Sokolovsky et al. 2017). A star is a candidate when one index exceeds z = "
            f"{float(meta.get('z_threshold') or 0):.2f} (field false-alarm probability 1 %) and it passes the comparison-star "
            f"gates: unsaturated, |neighbour light| <= 1 %, sky annulus inside every frame, SNR >= snr_min "
            f"({meta.get('n_gates_ok')} stars pass). A signal that a linear dependence on FWHM and airmass explains is marked "
            "as systematic and not listed. No sigma clipping, no detrending."
        )
        y = self._draw_paragraph_block(c, self.M_LEFT, y, self.USE_W, self._pdf_escape(txt), sty, paginate=False)
        hp = self.cache_dir / "variability_hockey_report.png"
        hockey_png(field, meta, hp, title=f"{self.obs_group} - new candidates: {meta.get('n_candidates', 0)}, "
                                          f"known variables recovered: {meta.get('n_known_detected', 0)}")
        self._draw_image_fit(c, hp, self.M_LEFT, y - 0.2 * self.cm, self.USE_W, y - self.M_BOTTOM - 0.6 * self.cm, image_type="hockey")
        self._page_footer(c)
        c.showPage()

        from variability_field import load_variability_candidates  # noqa: PLC0415

        cand = load_variability_candidates(phot)
        c.setPageSize(self.landscape(self.A4))
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 13)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.45 * self.cm, "Candidates")
        c.setFillColor(self.colors.black)
        y -= 0.9 * self.cm
        from tess_runner import tess_dir_for  # noqa: PLC0415
        from variability_store import load_adopted_periods, load_report_include, tess_cell, tess_summary  # noqa: PLC0415

        tess_on = bool(getattr(self._cfg, "tess_enabled", False)) if self._cfg is not None else False
        include = set(load_report_include(phot))
        adopted = load_adopted_periods(phot)
        if cand.empty:
            c.setFont(self.FONT_REG, 10)
            kd = int(meta.get("n_known_detected") or 0)
            c.drawString(self.M_LEFT, y, f"No new candidate in this group ({kd} known variable(s) in the field recovered by the search).")
            y -= 0.6 * self.cm
        else:
            hdr = ["#", "Gaia DR3", "RA", "Dec", "G", "BP-RP", "s [mag]", "z", "index", "catalogues", "TESS"]
            xs = [0.0, 0.6, 4.4, 6.2, 8.0, 9.1, 10.3, 11.6, 12.6, 14.7, 22.0]

            def _table_head(yh: float) -> float:
                c.setFont(self.FONT_BOLD, 8)
                for h, x in zip(hdr, xs):
                    c.drawString(self.M_LEFT + x * self.cm, yh, h)
                c.setFont(self.FONT_REG, 7.5)
                return yh - 0.45 * self.cm

            y = _table_head(y)
            for _, r in cand.iterrows():
                if y < self.M_BOTTOM + 0.8 * self.cm:
                    # every candidate gets a row (each also gets its own page below)
                    self._page_footer(c)
                    c.showPage()
                    c.setPageSize(self.landscape(self.A4))
                    y = _table_head(self.PAGE_H - self.M_TOP - 0.45 * self.cm)
                tr = tess_summary(phot, r["catalog_id"])
                tcell = tess_cell(tr) if (tess_on or r["catalog_id"] in include) else "-"
                zz = max(float(r["z_s"]) if pd.notna(r["z_s"]) else -9, float(r["z_eta"]) if pd.notna(r["z_eta"]) else -9)
                vals = [r["label"], r["catalog_id"], f"{float(r['ra_deg']):.5f}", f"{float(r['dec_deg']):+.5f}",
                        f"{float(r['mag']):.2f}", f"{float(r['bp_rp']):.2f}" if pd.notna(r.get("bp_rp")) else "-",
                        f"{float(r['s']):.4f}", f"{zz:.1f}", str(r["index"]) if pd.notna(r.get("index")) and str(r.get("index")) else "-",
                        str(r.get("catalogs", "not checked"))[:60], tcell[:40]]
                for v, x in zip(vals, xs):
                    c.drawString(self.M_LEFT + x * self.cm, y, self._pdf_escape(str(v)))
                y -= 0.42 * self.cm
            if y < self.M_BOTTOM + 8 * self.cm:
                self._page_footer(c)
                c.showPage()
                c.setPageSize(self.landscape(self.A4))
                y = self.PAGE_H - self.M_TOP
                c.setFont(self.FONT_BOLD, 13)
                c.setFillColor(self.C_TITLE)
                c.drawString(self.M_LEFT, y - 0.45 * self.cm, "Candidates in the field")
                c.setFillColor(self.colors.black)
                y -= 0.9 * self.cm
            fp = self.cache_dir / "variability_field_report.png"
            if field_png(self.platesolve_dir / "MASTERSTAR.fits", cand, fp) is not None:
                self._draw_image_fit(c, fp, self.M_LEFT, y - 0.3 * self.cm, self.USE_W * 0.55,
                                     y - self.M_BOTTOM - 0.6 * self.cm, image_type="hockey")
        self._page_footer(c)
        c.showPage()

        for _, r in cand.iterrows():
            cid = str(r["catalog_id"])
            c.setPageSize(self.landscape(self.A4))
            y = self.PAGE_H - self.M_TOP
            c.setFont(self.FONT_BOLD, 12)
            c.setFillColor(self.C_TITLE)
            c.drawString(self.M_LEFT, y - 0.45 * self.cm, f"{r['label']}  Gaia DR3 {cid}  G = {float(r['mag']):.2f}")
            c.setFillColor(self.colors.black)
            y -= 0.9 * self.cm
            lc = phot / "lightcurves" / f"varcand_{cid}.csv"
            lp = self.cache_dir / f"varcand_{cid}.png"
            half = (self.USE_W - 0.5 * self.cm) / 2.0
            if candidate_lc_png(lc, lp, f"{r['label']} night light curve", time_label=str(meta.get("time_column") or "JD")) is not None:
                self._draw_image_fit(c, lp, self.M_LEFT, y, half, 7.0 * self.cm, image_type="tess_phased")
            tr = tess_summary(phot, cid)
            if tr and (tess_on or cid in include):
                x2 = self.M_LEFT + half + 0.5 * self.cm
                c.setFont(self.FONT_BOLD, 10)
                c.drawString(x2, y - 0.3 * self.cm, "TESS")
                c.setFont(self.FONT_REG, 8.5)
                b = tr.get("best") or {}
                lines = [f"Verdict: {tr.get('verdict')}"]
                if tr.get("verdict") == "RELIABLE" and b.get("period_d"):
                    lines.append(f"P = {b['period_d']:.6f} +- {float(b.get('period_err_d') or 0):.6f} d  ({b.get('method')}), FAP {float(b.get('fap') or 0):.1e}")
                    if "equal minima" in str(b.get("note", "")):
                        lines.append(f"equal minima and maxima: for an EW/ELL binary the orbital period is 2P = {2 * float(b['period_d']):.6f} d")
                    lines.append(f"amplitude {float(b.get('amplitude_mag') or 0):.3f} mag, minimum BJD {float(b.get('epoch_min_bjd') or 0):.5f}")
                for rs in tr.get("reasons") or []:
                    lines.append(f"- {rs}")
                secs = tr.get("sectors") or []
                lines.append(f"sectors: {', '.join(str(sx.get('sector')) for sx in secs)}; neighbour / target light in TESS aperture "
                             + (f"{100 * float(tr['contamination']):.0f} %;" if tr.get("contamination") is not None else "unknown;")
                             + f" target likelihood {float(tr.get('target_likelihood') or 0):.2f}; amplitude S/N "
                             + (f"{float(tr['snr_amplitude']):.1f}" if tr.get("snr_amplitude") is not None else "-"))
                if b.get("note"):
                    lines.append(str(b.get("note"))[:110])
                if cid in adopted:
                    lines.append(f"period adopted by the observer in VYVAR: P = {adopted[cid]['period_d']:.7f} d, "
                                 f"T0 = {adopted[cid]['t0_bjd']:.5f} BJD_TDB")
                yy = y - 0.8 * self.cm
                for ln in lines:
                    c.drawString(x2, yy, self._pdf_escape(ln)[:120])
                    yy -= 0.42 * self.cm
                tdir = tess_dir_for(phot, cid)
                yimg = min(yy, y - 7.3 * self.cm) - 0.2 * self.cm
                w3 = (self.USE_W - 1.0 * self.cm) / 3.0
                for j, key in enumerate(("periodogram", "phased", "pixel_map")):
                    fpath = tdir / f"{key}.png"
                    if fpath.is_file():
                        self._draw_image_fit(c, Path(fpath), self.M_LEFT + j * (w3 + 0.5 * self.cm), yimg, w3,
                                             yimg - self.M_BOTTOM - 0.5 * self.cm, image_type="tess_phased")
            self._page_footer(c)
            c.showPage()

    def _report_osc_multiband_summary(self, c: "canvas.Canvas") -> None:
        from osc_align import obs_group_band_token, osc_multiband_summary_rows

        rows = osc_multiband_summary_rows(self.draft_dir, self.obs_group)
        if not rows:
            return
        c.setPageSize(self.landscape(self.A4))
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 14)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.35 * self.cm, "OSC multi-band summary (unified frame set)")
        c.setFillColor(self.colors.black)
        y -= 1.1 * self.cm
        c.setFont(self.FONT_BOLD, 9)
        c.drawString(self.M_LEFT, y, "Channel")
        c.drawString(self.M_LEFT + 2.2 * self.cm, y, "Band")
        c.drawString(self.M_LEFT + 3.8 * self.cm, y, "Export")
        c.drawString(self.M_LEFT + 6.5 * self.cm, y, "n_LCs")
        c.drawString(self.M_LEFT + 8.5 * self.cm, y, "obs-group")
        y -= 0.55 * self.cm
        c.setFont(self.FONT_REG, 9)
        for row in rows:
            ch = str(row.get("channel", ""))
            tok = str(row.get("band_token", "") or obs_group_band_token(str(row.get("obs_group", ""))))
            elig = "yes" if row.get("export_eligible") else "internal"
            n_lc = int(row.get("n_lightcurves", 0) or 0)
            og = str(row.get("obs_group", ""))
            c.drawString(self.M_LEFT, y, ch)
            c.drawString(self.M_LEFT + 2.2 * self.cm, y, tok or "-")
            c.drawString(self.M_LEFT + 3.8 * self.cm, y, elig)
            c.drawString(self.M_LEFT + 6.5 * self.cm, y, str(n_lc))
            c.drawString(self.M_LEFT + 8.5 * self.cm, y, og[:48])
            y -= 0.45 * self.cm
        self._page_footer(c)
        c.showPage()

    def _report_abbreviations(self, c: "canvas.Canvas") -> None:
        c.setPageSize(self.landscape(self.A4))
        y0 = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 16)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y0 - 0.45 * self.cm, "Abbreviations & Notes")
        c.setFillColor(self.colors.black)
        pairs = [
            ("AAVSO", "American Association of Variable Star Observers"),
            ("BJD(TDB)", "Barycentric Julian Date (TDB time standard; LC column ``bjd``)"),
            ("BP-RP", "Gaia colour index (Blue Photometer minus Red Photometer)"),
            ("comp", "Comparison star"),
            ("DAO", "DAOPHOT star detection algorithm (Stetson 1987)"),
            ("dBP-RP", "Colour difference between comparison star and target (BP-RP)"),
            ("dist '", "Distance of a comparison star from the target (arcminutes)"),
            ("EA", "Algol-type eclipsing binary"),
            ("EB", "Beta Lyrae-type eclipsing binary"),
            ("EW", "W UMa-type eclipsing binary (contact)"),
            ("ELL", "Ellipsoidal variable"),
            ("lc", "Light curve"),
            ("lc_rms", "Light curve RMS scatter (magnitudes)"),
            ("lc_median_mag", "Median calibrated magnitude"),
            ("n_good_comp", "Aperture-photometry ensemble pool: comps with stability quality good or suspect (summary column)"),
            ("n_clean", "Comparison stars in the target's ensemble (all passed the selection gates; trust gate metric)"),
            ("setup", "Observation group (filter _ exposure _ binning, e.g. NoFilter_60_2)"),
            ("comp_rms", "Comparison-star selection detrended relative-flux scatter (comparison star stability)"),
            ("rms mmag", "Comparison-star stability rms (comparison-star selection, 1.0857 x relative-flux scatter)"),
            ("period_reliability", "reliable / uncertain / noise - quality flag for TESS period"),
            ("RMS candidate", "Star with RMS significantly above the noise envelope"),
            ("ROT", "Rotational variable"),
            ("SNR", "Signal-to-noise ratio"),
            ("TESS", "Transiting Exoplanet Survey Satellite (NASA) - used for period analysis"),
            ("tier", "Comparison star colour tier (1=closest BP-RP match, 4=most distant)"),
            ("VAR", "Variable (type unconfirmed)"),
            ("VAR.ASTRO", "var.astro.cz (Czech Astronomical Society) light-curve / minimum database; 3-column BJD export"),
            ("CV", "AAVSO filter code: unfiltered (clear) data on a V zero point, untransformed"),
            ("check star", "Star measured like the target with the same ensemble; KMAG measured vs catalogue"),
            ("_dontsubmit", "Export subfolder for RED-trust targets (written, not for submission)"),
            ("GSPC", "Gaia DR3 synthetic photometry (XP spectra) - zero point for g/r/i from synthetic standards"),
            ("ePSF", "Effective PSF (Anderson & King 2000) - internal PSF light curve, not exported"),
            ("k''", "Second-order (colour) extinction coefficient, mag per airmass per BP-RP"),
            ("MERR", "Median per-point magnitude error (1.0857 x relative flux error)"),
            ("mmag", "Millimagnitude (0.001 mag)"),
            ("n_eff", "Effective number of comparison stars of a weighted ensemble"),
            ("OOE", "Out-of-eclipse: rms over the epochs outside eclipses"),
            ("p2p", "Point-to-point scatter, 1.4826 x MAD(diff) / sqrt(2)"),
            ("sigma_ZP", "Uncertainty of the zero point"),
            ("TRUST", "GREEN / YELLOW / RED user information from data-quality warnings (check star, ZP scatter, ...)"),
            ("VDI", "Variability Detection Index (phase-space metric)"),
            ("VDI candidate", "Star with high VDI score (non-random phase distribution)"),
            ("w(rel)", "Relative weight of comparison star in ensemble"),
            ("ZP", "Zero point (photometric calibration offset)"),
            ("zone_flag", "linear = DAO-detected; saturated = likely saturated on MASTERSTAR"),
            ("FWHM", "Full Width at Half Maximum (PSF size in pixels)"),
        ]
        pairs_sorted = sorted(pairs, key=lambda p: str(p[0]).lower())
        left_x = self.M_LEFT
        mid = self.M_LEFT + self.USE_W / 2.0 + 0.3 * self.cm
        yl = yr = y0 - 1.1 * self.cm
        line_skip = 9.0
        c.setFont(self.FONT_REG, 7.5)
        half = (len(pairs_sorted) + 1) // 2
        for i, (abbr, desc) in enumerate(pairs_sorted):
            block = f"{abbr}: {desc}"
            lines = textwrap.wrap(block, width=55) or [block]
            if i < half:
                for ln in lines:
                    c.drawString(left_x, yl, ln)
                    yl -= line_skip
            else:
                for ln in lines:
                    c.drawString(mid, yr, ln)
                    yr -= line_skip
        self._page_footer(c)
        c.showPage()

    def _report_configuration_page(self, c: "canvas.Canvas") -> None:
        """Configuration page: run-vs-default deviations from the provenance snapshot."""
        model = config_deviation_model(self._pipeline_meta)

        def _fmt(v: Any) -> str:
            if isinstance(v, bool):
                return "True" if v else "False"
            if isinstance(v, float):
                return f"{v:g}"
            if isinstance(v, dict):
                return "{}" if not v else json.dumps(v, sort_keys=True)
            if v is None:
                return "None"
            s = str(v)
            return s if len(s) <= 48 else s[:45] + "..."

        c.setPageSize(self.landscape(self.A4))
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 16)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.45 * self.cm, "Configuration")
        c.setFillColor(self.colors.black)
        y -= 1.2 * self.cm

        if model["fallback"]:
            c.setFont(self.FONT_BOLD, 10)
            c.setFillColor(self.colors.HexColor("#b00000"))
            c.drawString(
                self.M_LEFT, y,
                "provenance snapshot missing - showing live config, not run config",
            )
            c.setFillColor(self.colors.black)
            y -= 0.6 * self.cm

        hdr = model["header"]
        fp = model["fingerprint"]
        c.setFont(self.FONT_REG, 9)
        header_lines = [
            f"git_hash: {_fmt(hdr['git_hash'])}    git_dirty_code: {_fmt(hdr['git_dirty_code'])}"
            f"    stamped_at_utc: {_fmt(hdr['stamped_at_utc'])}",
            f"entry_point: {_fmt(hdr['entry_point'])}    parameters modified: {hdr['n_modified']}",
            f"fingerprint: preprocess_sky_surface_order={_fmt(fp['preprocess_sky_surface_order'])}"
            f", seed_policy={_fmt(fp['seed_policy'])}, density_profile={_fmt(fp['density_profile'])}",
        ]
        for ln in header_lines:
            c.drawString(self.M_LEFT, y, ln)
            y -= 0.42 * self.cm
        y -= 0.2 * self.cm

        # VSX scope one-liner on the config page.
        try:
            c.setFont(self.FONT_REG, 9)
            c.drawString(
                self.M_LEFT,
                y,
                "VSX auto-target scope: detection-limited (DAO+Gaia match)",
            )
            y -= 0.42 * self.cm
            y -= 0.1 * self.cm
        except Exception as exc:  # noqa: BLE001
            logging.debug("[VSX-SCOPE] config page line skipped: %s", exc)

        c.setFont(self.FONT_OBL, 8)
        c.drawString(
            self.M_LEFT, y,
            "Defaults are evaluated against current code; for exact run-time semantics "
            "consult the git hash above.",
        )
        y -= 0.6 * self.cm

        col_key = self.M_LEFT
        col_run = self.M_LEFT + 12.0 * self.cm
        col_def = self.M_LEFT + 20.0 * self.cm

        def _draw_table_header(yy: float) -> float:
            c.setFont(self.FONT_BOLD, 8.5)
            c.setFillColor(self.C_TITLE)
            c.drawString(col_key, yy, "key")
            c.drawString(col_run, yy, "run value")
            c.drawString(col_def, yy, "default")
            c.setFillColor(self.colors.black)
            return yy - 0.4 * self.cm

        rows = model["rows"]
        c.setFont(self.FONT_BOLD, 10)
        c.drawString(self.M_LEFT, y, f"Deviations from defaults ({len(rows)})")
        y -= 0.55 * self.cm
        y = _draw_table_header(y)
        c.setFont(self.FONT_REG, 8)
        if not rows:
            c.drawString(col_key, y, "(none - run config matches code defaults)")
            y -= 0.4 * self.cm
        for row in rows:
            if y < self._layout_y_floor() + 1.0 * self.cm:
                y = self._layout_page_break(c)
                c.setFont(self.FONT_BOLD, 16)
                c.setFillColor(self.C_TITLE)
                c.drawString(self.M_LEFT, y - 0.45 * self.cm, "Configuration (cont.)")
                c.setFillColor(self.colors.black)
                y -= 1.2 * self.cm
                y = _draw_table_header(y)
                c.setFont(self.FONT_REG, 8)
            c.drawString(col_key, y, _fmt(row["key"]))
            c.drawString(col_run, y, _fmt(row["value"]))
            c.drawString(col_def, y, _fmt(row["default"]))
            y -= 0.36 * self.cm

        unknown = model["unknown_keys"]
        if unknown:
            y -= 0.3 * self.cm
            if y < self._layout_y_floor() + 1.0 * self.cm:
                y = self._layout_page_break(c)
                y -= 0.5 * self.cm
            c.setFont(self.FONT_BOLD, 8.5)
            c.drawString(self.M_LEFT, y, f"Unknown / legacy snapshot keys ({len(unknown)}):")
            y -= 0.4 * self.cm
            c.setFont(self.FONT_REG, 7.5)
            for ln in textwrap.wrap(", ".join(unknown), width=140) or ["(none)"]:
                c.drawString(self.M_LEFT, y, ln)
                y -= 0.32 * self.cm

        y = max(y, self._layout_y_floor() + 0.4 * self.cm)
        c.setFont(self.FONT_OBL, 7.5)
        c.drawString(
            self.M_LEFT, self._layout_y_floor() + 0.05 * self.cm,
            "Full run configuration snapshot: photometry/pipeline_meta.json (provenance.config_snapshot).",
        )
        self._page_footer(c)
        c.showPage()

    def _report_resolved_facts_page(self, c: "canvas.Canvas") -> None:
        """Resolved-facts block: values the run actually used (site/gain/RN/plate/frame/etc.)."""
        model = resolved_facts_model(
            self._pipeline_meta,
        )

        c.setPageSize(self.landscape(self.A4))
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 16)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.45 * self.cm, "Configuration - resolved facts (as run)")
        c.setFillColor(self.colors.black)
        y -= 1.2 * self.cm

        c.setFont(self.FONT_OBL, 8)
        c.drawString(
            self.M_LEFT, y,
            "Values the run actually used, resolved from FITS / WCS / DB at run time "
            "(config values are only fallbacks).",
        )
        y -= 0.6 * self.cm

        if model["fallback"]:
            c.setFont(self.FONT_BOLD, 9)
            c.setFillColor(self.colors.HexColor("#b00000"))
            for w in model["warnings"]:
                for ln in textwrap.wrap(w, width=150):
                    c.drawString(self.M_LEFT, y, ln)
                    y -= 0.4 * self.cm
            c.setFillColor(self.colors.black)
            y -= 0.2 * self.cm

        _vsx = model.get("vsx_depth") if isinstance(model.get("vsx_depth"), dict) else {}
        if bool(_vsx.get("warn")) and _vsx.get("message"):
            c.setFont(self.FONT_BOLD, 9)
            c.setFillColor(self.colors.HexColor("#b00000"))
            c.drawString(self.M_LEFT, y, f"WARN: {_vsx['message']}")
            c.setFillColor(self.colors.black)
            y -= 0.5 * self.cm

        col_lbl = self.M_LEFT
        col_val = self.M_LEFT + 7.0 * self.cm
        col_src = self.M_LEFT + 20.0 * self.cm
        c.setFont(self.FONT_BOLD, 8.5)
        c.setFillColor(self.C_TITLE)
        c.drawString(col_lbl, y, "fact")
        c.drawString(col_val, y, "run value")
        c.drawString(col_src, y, "source")
        c.setFillColor(self.colors.black)
        y -= 0.45 * self.cm

        c.setFont(self.FONT_REG, 9)
        for row in model["rows"]:
            if y < self._layout_y_floor() + 1.0 * self.cm:
                y = self._layout_page_break(c)
            c.drawString(col_lbl, y, str(row["label"]))
            c.drawString(col_val, y, str(row["value"])[:90])
            c.drawString(col_src, y, str(row["source"])[:42])
            y -= 0.45 * self.cm

        self._page_footer(c)
        c.showPage()

    def _report_full_config_snapshot_pages(self, c: "canvas.Canvas") -> None:
        """Appendix: the complete as-run config snapshot, grouped by phase, multi-column."""
        model = full_config_snapshot_model(self._pipeline_meta)

        def _fmt(v: Any) -> str:
            if isinstance(v, bool):
                return "True" if v else "False"
            if isinstance(v, float):
                return f"{v:g}"
            if isinstance(v, dict):
                s = "{}" if not v else json.dumps(v, sort_keys=True)
            elif v is None:
                s = "None"
            else:
                s = str(v)
            return s if len(s) <= 34 else s[:31] + "..."

        n_cols = 3
        col_gap = 0.4 * self.cm
        col_w = (self.USE_W - (n_cols - 1) * col_gap) / n_cols
        col_x = [self.M_LEFT + i * (col_w + col_gap) for i in range(n_cols)]
        row_h = 0.34 * self.cm

        c.setPageSize(self.landscape(self.A4))

        def _new_page(title: str) -> tuple[float, int]:
            c.setPageSize(self.landscape(self.A4))
            yy = self.PAGE_H - self.M_TOP
            c.setFont(self.FONT_BOLD, 16)
            c.setFillColor(self.C_TITLE)
            c.drawString(self.M_LEFT, yy - 0.45 * self.cm, title)
            c.setFillColor(self.colors.black)
            return yy - 1.2 * self.cm, 0

        y, col = _new_page("Full configuration snapshot (as run)")
        c.setFont(self.FONT_OBL, 8)
        c.drawString(
            self.M_LEFT, y,
            f"Complete as-run configuration ({model['n_keys']} keys) from "
            "provenance.config_snapshot, grouped by phase. This is the copy of the config "
            "taken from the run, not from disk.",
        )
        y -= 0.55 * self.cm
        if model["fallback"]:
            c.setFont(self.FONT_BOLD, 9)
            c.setFillColor(self.colors.HexColor("#b00000"))
            c.drawString(
                self.M_LEFT, y,
                "provenance snapshot missing - showing live config, not run config",
            )
            c.setFillColor(self.colors.black)
            y -= 0.5 * self.cm
        # Honesty note: snapshot may cover fewer keys than the 304-entry registry (legacy runs
        # written before the complete-snapshot writer; fresh runs record the full field set).
        omitted = model.get("omitted_keys") or []
        if omitted:
            c.setFont(self.FONT_OBL, 8)
            c.setFillColor(self.colors.HexColor("#b00000"))
            note = (
                f"{len(omitted)} keys omitted from snapshot: not recorded by the run that wrote "
                f"it (fresh runs record all {model.get('registry_count', 0)}). Omitted: "
                + ", ".join(omitted)
            )
            for ln in textwrap.wrap(note, width=170)[:3]:
                c.drawString(self.M_LEFT, y, ln)
                y -= 0.4 * self.cm
            c.setFillColor(self.colors.black)
            y -= 0.1 * self.cm
        content_top = y
        floor = self._layout_y_floor() + 0.8 * self.cm

        def _advance(yy: float, cc: int) -> tuple[float, int]:
            """Move down one row; wrap to the next column / page when the column is full."""
            yy -= row_h
            if yy < floor:
                cc += 1
                if cc >= n_cols:
                    self._page_footer(c)
                    c.showPage()
                    ny, _ = _new_page("Full configuration snapshot (cont.)")
                    return ny, 0
                return content_top, cc
            return yy, cc

        for phase in model["phases"]:
            entries = model["by_phase"].get(phase, [])
            # phase header
            if y < floor:
                y, col = _advance(y, col)
            c.setFont(self.FONT_BOLD, 8.5)
            c.setFillColor(self.C_TITLE)
            c.drawString(col_x[col], y, f"[{phase}] ({len(entries)})")
            c.setFillColor(self.colors.black)
            y, col = _advance(y, col)
            c.setFont(self.FONT_REG, 7)
            for key, val in entries:
                c.drawString(col_x[col], y, f"{key} = {_fmt(val)}")
                y, col = _advance(y, col)
            # blank line between phases
            y, col = _advance(y, col)

        self._page_footer(c)
        c.showPage()

    def _masterstar_data(self) -> np.ndarray | None:
        """MASTERSTAR image, read once per report (the per-target cutouts use it)."""
        if not hasattr(self, "_ms_data_cache"):
            from astropy.io import fits  # noqa: PLC0415

            _ms = self.platesolve_dir / "MASTERSTAR.fits"
            try:
                self._ms_data_cache = np.asarray(fits.getdata(_ms), dtype=float) if _ms.is_file() else None
            except (OSError, ValueError, TypeError) as exc:
                logging.warning("[PDF] MASTERSTAR not readable for the cutouts: %s", exc)
                self._ms_data_cache = None
        return self._ms_data_cache

    def build_pdf(self) -> Path:

        self._overflow_violations.clear()
        # OUTPUT-REVIEW-2026-10 step 7: every image of the cache is drawn again by this report
        from run_outputs import clear_report_cache  # noqa: PLC0415

        clear_report_cache(self.photometry_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build PDF
        c = self.canvas.Canvas(str(self.output_pdf), pagesize=self.landscape(self.A4))

        # 1-2) Cover + observation summary
        self._report_cover_page(c)
        self._report_observation_summary(c)

        # 3) FITS Quality Assessment (optional)
        self._report_fits_qa(c)

        # 4) Summary table + 5) HRD
        self._report_summary_table(c)
        self._report_hrd_page(c)

        # 6) Field map (full landscape page)
        self._report_field_map(c)

        # 7+) Per-star pages

        for _, row in self.summary_df.iterrows():
            try:
                cid = str(row.get("_cid", "") or self._norm_cid_decimal(row.get("catalog_id", "")))
                vsx_name = str(row.get("vsx_name", cid) or cid)
                _lc_rms = pd.to_numeric(row.get("lc_rms"), errors="coerce")
                _n_frames = int(pd.to_numeric(row.get("n_frames"), errors="coerce") or 0)
                if _n_frames <= 0 or (not np.isfinite(float(_lc_rms))):
                    logging.debug(
                        "[PDF] Skip LC page - no data: %s (n_frames=%d, lc_rms=%s)",
                        vsx_name,
                        _n_frames,
                        _lc_rms,
                    )
                    continue

                # target field map of aperture photometry (embedded once, at the PDF size)
                from field_map import field_map_path  # noqa: PLC0415

                field_img = field_map_path(self.lc_dir, cid)

                comp_rows_pdf, comp_excluded_note = self._comp_rows_for_target(cid)
                star_data = {
                    **star_fields(row, self._lc_stats_by_cid.get(cid, {})),
                    "field_img": str(field_img) if field_img is not None else "",
                    "comp_rows": comp_rows_pdf,
                    "comp_excluded_note": comp_excluded_note,
                }
                # TRUST-COVERAGE-2026-10 / PDF-NODATA-2026-10 (Milan 2026-10-12): a target with too
                # few usable epochs gets no page (it stays in the summary table of all targets).
                from trust_flag_core import coverage_insufficient, lc_epoch_counts  # noqa: PLC0415

                _ne, _nl = lc_epoch_counts(self.photometry_dir, cid)
                if bool(_ne is not None and _nl and coverage_insufficient(_ne, _nl)):
                    logging.info("PDF: no page for %s - insufficient data (%s of %s epochs usable)", cid, _ne, _nl)
                    continue
                self._report_per_star_page(c, star_data)
                self._report_epsf_star_page(c, star_data)
            except Exception as exc_star:  # noqa: BLE001
                # EXC-0273: T2 -- report/export may omit or misstate (sparse_buf = [] / self._report_per_star_page(c, sta... (EXCEPT-BULK 2026-07-08)
                logging.warning("PDF: skip star (%s): %s", row.get("vsx_name", ""), exc_star)
                continue

        # VAR-2026-10: variability search (hockey-stick, candidates, field, TESS when on)
        try:
            self._report_variability_v2(c)
        except Exception as exc:  # noqa: BLE001
            logging.warning("PDF variability section skipped (non-fatal): %s", exc)
        try:
            self._report_osc_multiband_summary(c)
        except Exception as exc:  # noqa: BLE001
            logging.warning("PDF OSC multiband summary skipped (non-fatal): %s", exc)
        self._report_abbreviations(c)
        try:
            self._report_configuration_page(c)
        except Exception as exc:  # noqa: BLE001
            logging.warning("PDF Configuration page skipped (non-fatal): %s", exc)
        try:
            self._report_resolved_facts_page(c)
        except Exception as exc:  # noqa: BLE001
            logging.warning("PDF resolved-facts page skipped (non-fatal): %s", exc)
        try:
            self._report_full_config_snapshot_pages(c)
        except Exception as exc:  # noqa: BLE001
            logging.warning("PDF full-config snapshot appendix skipped (non-fatal): %s", exc)

        c.save()
        if self._verify_overflow and self._overflow_violations:
            logging.warning(
                "PDF layout verify: %d overflow violation(s): %s",
                len(self._overflow_violations),
                "; ".join(self._overflow_violations[:5]),
            )
        return self.output_pdf
def generate_photometry_report(
    draft_dir: Path,
    obs_group: str,
    output_pdf: Path | None,
    *,
    var_results: dict[str, Any] | None = None,
    candidates: list[str] | None = None,
    crossmatch_bullets: dict[str, str] | None = None,
    accepted_periods: dict[str, float] | None = None,
    variability_timestamp: str | None = None,
    report_draft_label: str | None = None,
    tess_results: dict | None = None,
    report_title: str = "VYVAR \u2014 Summary Measure Report",
    verify_overflow: bool = False,
) -> Path | None:
    """
    Build a PDF photometry report for one observation night.

    Returns the path to the written PDF, or None if reportlab is not installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import cm, mm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph, Table, TableStyle
    except Exception as exc:  # noqa: BLE001
        # EXC-0274: T2 -- report/export may omit or misstate (from reportlab.lib.styles import ParagraphStyle / f... (EXCEPT-BULK 2026-07-08)
        logging.warning("reportlab is not installed, skipping PDF (%s)", exc)
        return None

    FONT_REG, FONT_BOLD, FONT_OBL = _register_pdf_unicode_fonts()

    _title = str(report_title or "VYVAR \u2014 Summary Measure Report")

    builder = _PhotometryReportBuilder(
        draft_dir=Path(draft_dir),
        obs_group=str(obs_group),
        output_pdf=output_pdf,
        var_results=var_results,
        candidates=candidates,
        crossmatch_bullets=crossmatch_bullets,
        accepted_periods=accepted_periods,
        variability_timestamp=variability_timestamp,
        report_draft_label=report_draft_label,
        tess_results=tess_results,
        report_title=_title,
        font_reg=FONT_REG,
        font_bold=FONT_BOLD,
        font_obl=FONT_OBL,
        colors_mod=colors,
        cm_mod=cm,
        mm_mod=mm,
        landscape_fn=landscape,
        a4_size=A4,
        canvas_mod=canvas,
        image_reader_mod=ImageReader,
        table_mod=Table,
        table_style_mod=TableStyle,
        paragraph_mod=Paragraph,
        paragraph_style_mod=ParagraphStyle,
        ta_left_mod=TA_LEFT,
    )
    builder._verify_overflow = bool(verify_overflow)
    result = builder.build_pdf()
    if verify_overflow:
        generate_photometry_report.last_overflow_violations = builder.overflow_violation_count  # type: ignore[attr-defined]
    return result


generate_photometry_report.last_overflow_violations = 0  # type: ignore[attr-defined]
