from __future__ import annotations

import glob
import hashlib
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
from decimal import Decimal, InvalidOperation

from citations import build_run_citation_context, emit_pdf_methods_sections
from report_methods import (
    active_report_methods,
    lc_csv_path,
    pdf_report_path,
    report_title as method_report_title,
)

if TYPE_CHECKING:
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph

# Gaia ID musí byť str — float64 stráca cifry
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


def _norm_cid(x: Any) -> str:
    """Normalize Gaia ``catalog_id`` to canonical digit string (report / joins)."""
    s = str(x or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    try:
        return str(int(Decimal(s)))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        try:
            return str(int(s))
        except Exception:  # noqa: BLE001
            return s


def _is_catalog_only(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: True = catalog_only row (no real DAO detection)."""
    mask = pd.Series(False, index=df.index)
    if "zone_flag" in df.columns:
        mask |= df["zone_flag"].astype(str).str.strip().str.lower() == "catalog_only"
    if "zone" in df.columns:
        mask |= df["zone"].astype(str).str.strip().str.lower() == "catalog_only"
    return mask


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


_GS11_SUMMARY_KEYS: tuple[str, ...] = (
    "enabled",
    "aperture_arcsec",
    "comps_gs11_rejected",
    "targets_corrected",
    "targets_skipped_low_d",
    "median_correction_mmag",
    "max_correction_mmag",
)


def gs11_report_lines(pipeline_meta: dict[str, Any] | None, cfg: Any) -> list[str]:
    """Text lines for PDF / tests — Flux Dilution (Gaia) subsection."""
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
    ap_s = f"{ap:.1f} arcsec" if math.isfinite(ap) else "—"
    try:
        med = float(gs11.get("median_correction_mmag", 0.0))
    except (TypeError, ValueError):
        med = float("nan")
    med_s = f"{med:.1f} mmag" if math.isfinite(med) else "—"
    return [
        "  Flux Dilution Assessment (Gaia DR3)",
        "  Status:                 enabled",
        f"  Aperture used:          {ap_s}",
        f"  Comps GS11-rejected:    {int(gs11.get('comps_gs11_rejected', 0) or 0)}",
        f"  Targets corrected:      {int(gs11.get('targets_corrected', 0) or 0)}  "
        f"(median Δm = {med_s})",
        f"  Targets skipped (D<0.50): {int(gs11.get('targets_skipped_low_d', 0) or 0)}",
    ]


class _PhotometryReportBuilder:
    """Internal PDF builder; section renderers extracted from generate_photometry_report."""

    _norm_cid = staticmethod(_norm_cid)

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
        photometry_method: str = "aperture",
        active_methods: list[str] | None = None,
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
        self._photometry_method = str(photometry_method or "aperture").strip().lower()
        self._active_methods = list(active_methods or ["aperture"])
        self._report_title = str(report_title or 'VYVAR \u2014 Summary Measure Report')
        self._candidates_set = {str(x).strip() for x in (candidates or []) if str(x).strip()}
        self.platesolve_dir = self.draft_dir / "platesolve" / self.obs_group
        self.photometry_dir = self.platesolve_dir / "photometry"
        self.lc_dir = self.photometry_dir / "lightcurves"
        self.cache_dir = self.photometry_dir / "_report_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.summary_csv = self.photometry_dir / "photometry_summary.csv"
        self._pipeline_meta: dict[str, Any] | None = None
        _pm_path = self.photometry_dir / "pipeline_meta.json"
        if _pm_path.is_file():
            try:
                self._pipeline_meta = json.loads(_pm_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._pipeline_meta = {}
        else:
            self._pipeline_meta = {}
        if "calibration_mode" not in (self._pipeline_meta or {}):
            try:
                from draft_provenance import resolve_calibration_mode

                _mode = resolve_calibration_mode(archive_path=self.draft_dir)
                if _mode:
                    self._pipeline_meta = dict(self._pipeline_meta or {})
                    self._pipeline_meta["calibration_mode"] = _mode
            except Exception:  # noqa: BLE001
                pass
        self.comp_csv = self.photometry_dir / "comparison_stars_per_target.csv"
        self.at_csv_primary = self.platesolve_dir / "active_targets.csv"
        self.at_csv_alt = self.photometry_dir / "active_targets.csv"

        self.active_targets_csv = self.at_csv_primary if self.at_csv_primary.exists() else self.at_csv_alt

        if not self.summary_csv.exists():
            raise FileNotFoundError(f"Missing photometry_summary.csv: {self.summary_csv}")

        self.summary_df = pd.read_csv(self.summary_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
        self.comp_df = (
            pd.read_csv(self.comp_csv, low_memory=False, dtype={**_GAIA_ID_DTYPE, "target_catalog_id": str})
            if self.comp_csv.exists()
            else pd.DataFrame()
        )
        self.at_df = (
            pd.read_csv(self.active_targets_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
            if self.active_targets_csv.exists()
            else pd.DataFrame()
        )

        if "catalog_id" in self.summary_df.columns:
            self.summary_df["_cid"] = self.summary_df["catalog_id"].map(_norm_cid)
        if "catalog_id" in self.at_df.columns:
            self.at_df["_cid"] = self.at_df["catalog_id"].map(_norm_cid)
        if "target_catalog_id" in self.comp_df.columns:
            self.comp_df["_tcid"] = self.comp_df["target_catalog_id"].map(_norm_cid)
        elif "catalog_id" in self.comp_df.columns:
            self.comp_df["_tcid"] = self.comp_df["catalog_id"].map(_norm_cid)

        self._candidates_norm = {_norm_cid(str(x)) for x in self._candidates_set if str(x).strip()}

        # Join metadata (vsx_type, bp_rp) into summary for report ordering/labels.
        if not self.at_df.empty and "_cid" in self.at_df.columns and "_cid" in self.summary_df.columns:
            meta_cols = [c for c in ("vsx_type", "zone_flag", "bp_rp", "b_v", "vsx_name") if c in self.at_df.columns]
            meta = self.at_df[["_cid"] + meta_cols].drop_duplicates("_cid")
            self.summary_df = self.summary_df.merge(meta, how="left", on="_cid", suffixes=("", "_at"))

        if "vsx_type" in self.summary_df.columns:
            self.summary_df["_vsx_rank"] = self.summary_df["vsx_type"].map(self._vsx_type_sort_rank)
        else:
            self.summary_df["_vsx_rank"] = 99
        if "lc_rms" in self.summary_df.columns:
            self.summary_df["_lc_rms"] = pd.to_numeric(self.summary_df["lc_rms"], errors="coerce")
            self.summary_df = self.summary_df.sort_values(["_vsx_rank", "_lc_rms"], ascending=[True, True], na_position="last")
        else:
            self.summary_df = self.summary_df.sort_values("_vsx_rank", ascending=True, na_position="last")

        try:
            from config import AppConfig

            self._cfg = AppConfig()
            self._use_bprp_primary = True
        except Exception:  # noqa: BLE001
            self._cfg = None
            self._use_bprp_primary = True

        if (self._var_results is None) or (not self._candidates_set):
            vr_csv, cnorm_csv = self._try_load_variability_from_csv()
            if vr_csv is not None and self._var_results is None:
                self._var_results = vr_csv
            if cnorm_csv and not self._candidates_set:
                self._candidates_set = set(str(x).strip() for x in cnorm_csv if str(x).strip())
            if self._candidates_set:
                self._candidates_norm = {_norm_cid(str(x)) for x in self._candidates_set if str(x).strip()}

        self.obs_date_human = self._obs_date_str()
        self.date_token = datetime.today().strftime("%Y%m%d")
        try:
            # If obs_date_human is in YYYY-MM-DD, take it.
            if "-" in self.obs_date_human and len(self.obs_date_human) >= 10:
                self.date_token = self.obs_date_human.split("T", 1)[0].replace("-", "")
        except Exception:  # noqa: BLE001
            pass

        if self.output_pdf is None:
            self.output_pdf = pdf_report_path(
                self.draft_dir,
                self.obs_group,
                self._photometry_method,
                active_methods=self._active_methods,
                date_str=self.date_token,
            )
        self.output_pdf = Path(self.output_pdf)
        self.output_pdf.parent.mkdir(parents=True, exist_ok=True)

        # Styles / self.colors
        self.C_TITLE = self.colors.HexColor("#1a1a2e")
        self.C_GOOD = self.colors.HexColor("#2ecc71")
        self.C_MID = self.colors.HexColor("#f39c12")
        self.C_BAD = self.colors.HexColor("#e74c3c")

        # Aggregate metrics (cover page)
        self.n_lc = int(len(self.summary_df))
        self.lc_count_display = self._format_lc_count_display()
        self.med_rms = float(np.nanmedian(pd.to_numeric(self.summary_df.get("lc_rms"), errors="coerce"))) if self.n_lc else float("nan")
        self.rms_lt_005 = int((pd.to_numeric(self.summary_df.get("lc_rms"), errors="coerce") < 0.05).sum()) if self.n_lc else 0
        self.avg_good_comp = float(np.nanmean(pd.to_numeric(self.summary_df.get("n_good_comp"), errors="coerce"))) if self.n_lc else float("nan")
        self.best_rms = float(np.nanmin(pd.to_numeric(self.summary_df.get("lc_rms"), errors="coerce"))) if self.n_lc else float("nan")
        self.worst_rms = float(np.nanmax(pd.to_numeric(self.summary_df.get("lc_rms"), errors="coerce"))) if self.n_lc else float("nan")
        self.avg_bp_rp = float(np.nanmean(pd.to_numeric(self.summary_df.get("bp_rp"), errors="coerce"))) if self.n_lc else float("nan")
        self.setups = 1
        self.fwhm_px = float("nan")
        if (self.platesolve_dir / "MASTERSTAR.fits").exists():
            try:
                from ui_aperture_photometry import _load_fwhm  # local import

                self.fwhm_px = float(_load_fwhm(self.platesolve_dir / "MASTERSTAR.fits"))
            except Exception:  # noqa: BLE001
                pass
        self.aperture_px = float(np.nanmedian(pd.to_numeric(self.summary_df.get("aperture_px"), errors="coerce"))) if self.n_lc else float("nan")

        self.aavso_dir = self.photometry_dir / "lightcurves_reports" / "aavso"
        self.varastro_dir = self.photometry_dir / "lightcurves_reports" / "varastro"
        self.n_aavso = len(list(self.aavso_dir.glob("*.txt"))) if self.aavso_dir.is_dir() else 0
        self.n_varastro = len(list(self.varastro_dir.glob("*.txt"))) if self.varastro_dir.is_dir() else 0

        self.comp_pool_cover_rows = self._build_comp_pool_cover_rows()
        self._var_cand_by_cid = self._load_variability_candidates_by_cid()
        self._lc_stats_by_cid = self._build_target_lc_stats()
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

        self.NOTE_TXT = (
            "Gaia BP-RP colour | COMP weights: w = 1/sigma^2 - Broeg et al., Astron. Nachr. 326, 134 (2005)"
            if self._use_bprp_primary
            else (
                "B-V from Gaia BP-RP (Riello et al. 2021) | "
                "COMP weights: w = 1/sigma^2 - Broeg et al., Astron. Nachr. 326, 134 (2005)"
            )
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
        }

        # ---------------------------------------------------------------------
        # QA page (FWHM + sky + masterstar candidate table)
        # ---------------------------------------------------------------------
        self.bullets_by_cid = {_norm_cid(str(k)): str(v) for k, v in self._crossmatch_bullets.items()}

    def _vsx_type_sort_rank(self, val: Any) -> int:
        """Sort key: EA, EB, EW, ROT, VAR, then other non-empty types, then unknown/nan."""
        s0 = str(val if val is not None else "").strip().upper()
        if not s0 or s0 in ("NAN", "NONE", "—", "-"):
            return 99
        s = s0
        for sep in ("/", ",", ";", " ", "("):
            if sep in s:
                s = s.split(sep, 1)[0].strip()
                break
        order = ("EA", "EB", "EW", "ROT", "VAR")
        if s in order:
            return int(order.index(s))
        if s == "ELL":
            return 5
        return 50
    def _try_load_variability_from_csv(self, ) -> tuple[dict[str, Any] | None, set[str]]:
        """Fallback when UI did not pass var_results/candidates: read exported pipeline/UI CSV."""
        vpaths = [
            self.photometry_dir / "variability_candidates.csv",
            self.platesolve_dir / "variability_candidates.csv",
        ]
        vp = next((p for p in vpaths if p.is_file()), None)
        if vp is None:
            return None, set()
        try:
            vdf = pd.read_csv(vp, low_memory=False, dtype=_GAIA_ID_DTYPE)
        except Exception:  # noqa: BLE001
            return None, set()
        if vdf.empty or "catalog_id" not in vdf.columns:
            return None, set()
        cands_raw = {str(x).strip() for x in vdf["catalog_id"].tolist() if str(x).strip()}
        rms_like = vdf.copy()
        try:
            from gaia_catalog_id import normalize_gaia_source_id_series

            rms_like["catalog_id"] = normalize_gaia_source_id_series(rms_like["catalog_id"])
        except Exception:  # noqa: BLE001
            rms_like["catalog_id"] = rms_like["catalog_id"].map(_norm_cid)
        vsx_known: set[str] = set()
        if "_cid" in self.summary_df.columns and "vsx_name" in self.summary_df.columns:
            m = self.summary_df["vsx_name"].fillna("").astype(str).str.strip().ne("")
            vsx_known = set(self.summary_df.loc[m, "_cid"].astype(str).map(_norm_cid))
        rms_like["vsx_known_variable"] = rms_like["catalog_id"].astype(str).map(_norm_cid).isin(vsx_known)
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
        return {"rms_df": rms_like, "vdi_df": vdi_df}, {_norm_cid(str(x)) for x in cands_raw}
    def _obs_date_str(self, ) -> str:
        # Prefer MASTERSTAR DATE-OBS if present.
        ms_fits = self.platesolve_dir / "MASTERSTAR.fits"
        if ms_fits.exists():
            try:
                from astropy.io import fits

                with fits.open(ms_fits, memmap=False) as hdul:
                    hdr = hdul[0].header
                for key in ("DATE-OBS", "DATEOBS", "DATE"):
                    v = str(hdr.get(key, "") or "").strip()
                    if v:
                        # Keep only date part.
                        return v.split("T", 1)[0].replace("-", ".")
            except Exception:  # noqa: BLE001
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
        vc = self.summary_df["lc_quality_flag"].astype(str).str.strip().str.lower().value_counts()
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
        if parts:
            return f"{n:d} total  ({', '.join(parts)})"
        return f"{n:d}"

    def _lunar_risk_fill_color(self, risk: str) -> Any:
        r = str(risk or "").strip().upper()
        if r == "LOW":
            return self.C_GOOD
        if r == "MEDIUM":
            return self.C_MID
        if r == "HIGH":
            return self.C_BAD
        return self.colors.black

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
        # Phase 2A in pipeline_meta) so the report matches the BJD/airmass location.
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

        y -= 0.2 * self.cm
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
            risk = str(lunar.get("lunar_risk", "") or "").strip().upper() or "—"
            reason = str(lunar.get("lunar_risk_reason", "") or "").strip()

            lunar_lines: list[tuple[str, str, float]] = []
            if math.isfinite(phase) and math.isfinite(age):
                lunar_lines.append(
                    (f"Lunar phase:       {phase:.1f}%  ({age:.1f} days since new moon)", self.FONT_REG, 9.0)
                )
            else:
                lunar_lines.append(("Lunar phase:       —", self.FONT_REG, 9.0))
            lunar_lines.append(
                (
                    f"Lunar separation:  {sep:.1f}\u00b0 from field center"
                    if math.isfinite(sep)
                    else "Lunar separation:  —",
                    self.FONT_REG,
                    9.0,
                )
            )
            lunar_lines.append(
                (
                    f"Lunar altitude:    {alt_moon:.1f}\u00b0 at session midpoint"
                    if math.isfinite(alt_moon)
                    else "Lunar altitude:    —",
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

        y -= 0.2 * self.cm
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
                lines = ["  LC quality: not available (re-run Phase 2A)"]
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
        y -= 0.2 * self.cm
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
            rows_out.append(("Comparison pool", "— (no comparison_stars_per_target.csv)"))
            return rows_out
        if "catalog_id" not in self.comp_df.columns:
            rows_out.append(("Comparison pool", "—"))
            return rows_out
        ids = self.comp_df["catalog_id"].map(_norm_cid).astype(str)
        tcol = pd.to_numeric(self.comp_df.get("comp_tier"), errors="coerce")
        rms = pd.to_numeric(self.comp_df.get("comp_rms"), errors="coerce")
        magv = pd.to_numeric(self.comp_df.get("mag"), errors="coerce")
        gdf = pd.DataFrame({"_id": ids, "_t": tcol, "comp_rms": rms, "mag": magv})
        gdf = gdf[gdf["_id"].astype(str).str.len() > 0].copy()
        if gdf.empty:
            rows_out.append(("Comparison pool", "—"))
            return rows_out
        gdf["_t"] = gdf["_t"].fillna(99).astype(int)
        best_tier = gdf.groupby("_id", sort=False)["_t"].min()
        per_id_rms_med = gdf.groupby("_id", sort=False)["comp_rms"].median()
        med_rms_pool = float(np.nanmedian(per_id_rms_med.to_numpy(dtype=float)))
        mag_per_id_min = gdf.groupby("_id", sort=False)["mag"].min()
        mag_per_id_max = gdf.groupby("_id", sort=False)["mag"].max()
        mn_mag = float(np.nanmin(mag_per_id_min.to_numpy(dtype=float)))
        mx_mag = float(np.nanmax(mag_per_id_max.to_numpy(dtype=float)))
        n_unique = int(best_tier.shape[0])
        n_t1 = int((best_tier == 1).sum())
        n_t2 = int((best_tier == 2).sum())
        n_t3p = int((best_tier >= 3).sum())
        rows_out.append(("Unique comparison stars", f"{n_unique:d}"))
        rows_out.append(("Median comp_rms (per star)", f"{med_rms_pool:.6f}" if np.isfinite(med_rms_pool) else "—"))
        if np.isfinite(mn_mag) and np.isfinite(mx_mag):
            rows_out.append(("Comp mag range", f"{mn_mag:.3f} – {mx_mag:.3f}"))
        else:
            rows_out.append(("Comp mag range", "—"))
        rows_out.append(("Tier 1 comp stars", f"{n_t1:d}"))
        rows_out.append(("Tier 2 comp stars", f"{n_t2:d}"))
        rows_out.append(("Tier 3+ comp stars", f"{n_t3p:d}"))
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
            pass
        if n_good or n_susp or n_rej:
            rows_out.append(("Good / suspect / rejected (comp_quality)", f"{n_good:d} / {n_susp:d} / {n_rej:d}"))
        else:
            rows_out.append(("Good / suspect / rejected (comp_quality)", "—"))
        return rows_out

    def _load_variability_candidates_by_cid(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for vp in (
            self.photometry_dir / "variability_candidates.csv",
            self.platesolve_dir / "variability_candidates.csv",
        ):
            if not vp.is_file():
                continue
            try:
                vdf = pd.read_csv(vp, low_memory=False, dtype=_GAIA_ID_DTYPE)
            except Exception:  # noqa: BLE001
                continue
            if vdf.empty or "catalog_id" not in vdf.columns:
                continue
            for _, r in vdf.iterrows():
                nk = self._norm_cid(r.get("catalog_id"))
                if nk:
                    out[nk] = r.to_dict()
            break
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
                code = str(getattr(cfg, "observer_code", "") or getattr(cfg, "aavso_observer_code", "") or "").strip()
        if not name:
            name = "—"
        if not code:
            code = "—"
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
        if self._cfg is not None:
            ps3 = pd.to_numeric(getattr(self._cfg, "phase01_plate_scale_arcsec_per_px", float("nan")), errors="coerce")
            if np.isfinite(float(ps3)) and float(ps3) > 0:
                return float(ps3)
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
                pass
        if isinstance(self._pipeline_meta, dict):
            eq = self._pipeline_meta.get("equipment")
            if isinstance(eq, dict):
                tel = tel or str(eq.get("telescope", eq.get("telescope_name", "")) or "").strip()
                cam = cam or str(eq.get("camera", eq.get("equipment_name", "")) or "").strip()
        parts = [p for p in (tel, cam) if p]
        if parts:
            return " · ".join(parts)
        return str(self.obs_group or "—")

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
            return out
        out["n_total"] = int(len(qc))
        rej = 0
        if "REJECTED_AUTO" in qc.columns:
            rej = int((pd.to_numeric(qc["REJECTED_AUTO"], errors="coerce").fillna(0) > 0).sum())
        elif "IS_REJECTED" in qc.columns:
            rej = int(pd.to_numeric(qc["IS_REJECTED"], errors="coerce").fillna(0).astype(bool).sum())
        out["n_rejected"] = int(rej)
        out["n_used"] = int(max(0, out["n_total"] - out["n_rejected"]))
        if "FWHM_PX" in qc.columns:
            f = pd.to_numeric(qc["FWHM_PX"], errors="coerce")
            if f.notna().any():
                out["fwhm_min"] = float(np.nanmin(f))
                out["fwhm_med"] = float(np.nanmedian(f))
                out["fwhm_max"] = float(np.nanmax(f))
        jd_col = next((c for c in ("INSPECTION_JD", "bjd", "BJD", "jd_mid") if c in qc.columns), None)
        if jd_col:
            jd = pd.to_numeric(qc[jd_col], errors="coerce").dropna()
            if len(jd):
                out["bjd_min"] = float(jd.min())
                out["bjd_max"] = float(jd.max())
        if not np.isfinite(float(out["bjd_min"])) and "lc_csv" in self.summary_df.columns:
            bjd_vals: list[float] = []
            for _, sr in self.summary_df.iterrows():
                lp = Path(str(sr.get("lc_csv") or "").strip())
                if not lp.is_file():
                    continue
                try:
                    dfl = pd.read_csv(lp, usecols=lambda c: c in {"bjd", "bjd_tdb", "hjd", "jd"}, low_memory=False)
                    xcol = next((xc for xc in ("bjd_tdb", "bjd", "hjd", "jd") if xc in dfl.columns), None)
                    if xcol:
                        xv = pd.to_numeric(dfl[xcol], errors="coerce").dropna()
                        bjd_vals.extend(float(x) for x in xv.to_numpy(dtype=float) if np.isfinite(x))
                except Exception:  # noqa: BLE001
                    continue
            if bjd_vals:
                arr = np.asarray(bjd_vals, dtype=float)
                out["bjd_min"] = float(np.nanmin(arr))
                out["bjd_max"] = float(np.nanmax(arr))
        return out

    def _build_target_lc_stats(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        lc_col = "lc_csv" if "lc_csv" in self.summary_df.columns else None
        if not lc_col:
            return stats
        for _, row in self.summary_df.iterrows():
            cid = self._norm_cid(row.get("catalog_id", row.get("_cid", "")))
            if not cid:
                continue
            pth = Path(str(row.get(lc_col) or "").strip())
            if not pth.is_file():
                pth = self.lc_dir / pth.name if pth.name else pth
            if not pth.is_file():
                n_fr = int(pd.to_numeric(row.get("n_frames"), errors="coerce") or 0)
                stats[cid] = {"n_points": n_fr if n_fr > 0 else 0, "merr_med": float("nan")}
                continue
            try:
                df = pd.read_csv(
                    pth,
                    usecols=lambda c: c in {"flag", "err", "magerr", "MAGERR"},
                    low_memory=False,
                )
            except Exception:  # noqa: BLE001
                stats[cid] = {"n_points": 0, "merr_med": float("nan")}
                continue
            if "flag" in df.columns:
                dfn = df[df["flag"].astype(str).eq("normal")]
                if not dfn.empty:
                    df = dfn
            n_pt = int(len(df))
            err_col = "err" if "err" in df.columns else ("magerr" if "magerr" in df.columns else "MAGERR")
            merr = float("nan")
            if err_col in df.columns:
                merr = float(np.nanmedian(pd.to_numeric(df[err_col], errors="coerce")))
            stats[cid] = {"n_points": n_pt, "merr_med": merr}
        return stats

    def _resolve_check_kname(self, check_cid: str, target_cid: str) -> str:
        cc = self._norm_cid(check_cid)
        if not cc:
            return "—"
        if not self.comp_df.empty and "_tcid" in self.comp_df.columns:
            sub = self.comp_df[self.comp_df["_tcid"].astype(str).eq(self._norm_cid(target_cid))]
            if not sub.empty and "catalog_id" in sub.columns:
                m = sub["catalog_id"].map(_norm_cid).astype(str).eq(cc)
                if bool(m.any()) and "vsx_name" in sub.columns:
                    vn = str(sub.loc[m, "vsx_name"].iloc[0] or "").strip()
                    if vn and vn.lower() not in ("nan", "none"):
                        return vn
        if not self.at_df.empty and "_cid" in self.at_df.columns and "vsx_name" in self.at_df.columns:
            m2 = self.at_df["_cid"].astype(str).eq(cc)
            if bool(m2.any()):
                vn2 = str(self.at_df.loc[m2, "vsx_name"].iloc[0] or "").strip()
                if vn2 and vn2.lower() not in ("nan", "none"):
                    return vn2
        return cc

    def _check_star_report_for(self, target_cid: str) -> dict[str, str]:
        out = {"kname": "—", "kmag": "—", "scatter": "—"}
        cid = self._norm_cid(target_cid)
        if not cid:
            return out
        try:
            from check_star_kmag import check_kmag_sidecar_path
        except Exception:  # noqa: BLE001
            check_kmag_sidecar_path = lambda lc_dir, tc: Path(lc_dir) / f"check_kmag_{tc}.csv"  # noqa: E731
        side = check_kmag_sidecar_path(self.lc_dir, cid)
        if side.is_file():
            try:
                sdf = pd.read_csv(side, low_memory=False)
                if not sdf.empty and "kmag" in sdf.columns:
                    km = pd.to_numeric(sdf["kmag"], errors="coerce")
                    med = float(np.nanmedian(km))
                    sc = float(np.nanstd(km)) if int(km.notna().sum()) > 1 else float("nan")
                    out["kmag"] = f"{med:.3f}" if np.isfinite(med) else "na"
                    out["scatter"] = f"{sc:.4f} mag" if np.isfinite(sc) else "—"
                    chk = str(sdf["check_catalog_id"].iloc[0] if "check_catalog_id" in sdf.columns else "")
                    out["kname"] = self._resolve_check_kname(chk, cid)
                    return out
            except Exception:  # noqa: BLE001
                pass
        try:
            from check_star_kmag import resolve_ensemble_ids_for_check, select_check_star

            if not self.comp_df.empty and "_tcid" in self.comp_df.columns:
                sub = self.comp_df[self.comp_df["_tcid"].astype(str).eq(cid)]
                if not sub.empty:
                    from config import AppConfig  # noqa: PLC0415

                    _chk_cfg = AppConfig()
                    ens = resolve_ensemble_ids_for_check(
                        cid,
                        sub,
                        lc_dir=self.lc_dir,
                        comp_quality_map=None,
                        cfg=_chk_cfg,
                    )
                    chk_row = select_check_star(sub, ensemble_ids=ens, cfg=_chk_cfg)
                    if chk_row is not None:
                        cc = self._norm_cid(chk_row.get("catalog_id", ""))
                        out["kname"] = self._resolve_check_kname(cc, cid)
        except Exception:  # noqa: BLE001
            pass
        return out

    def _ground_variability_line(self, target_cid: str, vsx_type: str = "") -> str:
        nk = self._norm_cid(target_cid)
        if nk and nk in self._accepted_periods:
            p = pd.to_numeric(self._accepted_periods.get(nk), errors="coerce")
            if np.isfinite(float(p)) and float(p) > 0:
                amp = ""
                vc = self._var_cand_by_cid.get(nk, {})
                rms = pd.to_numeric(vc.get("rms_pct"), errors="coerce") if vc else float("nan")
                if np.isfinite(float(rms)):
                    amp = f", RMS%={float(rms):.2f}"
                vt = str(vsx_type or vc.get("vsx_type", "") or "").strip()
                vt_txt = f", type={vt}" if vt else ""
                return f"Period: {float(p):.6f} d (accepted{amp}{vt_txt})"
        vc2 = self._var_cand_by_cid.get(nk, {})
        if vc2:
            rms2 = pd.to_numeric(vc2.get("rms_pct"), errors="coerce")
            dm = str(vc2.get("detection_method", "") or "").strip()
            vs = pd.to_numeric(vc2.get("variability_score"), errors="coerce")
            parts = []
            if dm:
                parts.append(dm)
            if np.isfinite(float(rms2)):
                parts.append(f"RMS%={float(rms2):.2f}")
            if np.isfinite(float(vs)):
                parts.append(f"score={float(vs):.3f}")
            vt2 = str(vsx_type or vc2.get("vsx_type", "") or "").strip()
            if vt2:
                parts.append(f"type={vt2}")
            if parts:
                return "Variability: " + ", ".join(parts)
        return "Period: no period"

    def _variability_edge_filter_note(self) -> str:
        """Read edge_filter_failed from exported variability_candidates.csv (pipeline ground truth)."""
        for vp in (
            self.photometry_dir / "variability_candidates.csv",
            self.platesolve_dir / "variability_candidates.csv",
        ):
            if not vp.is_file():
                continue
            try:
                vdf = pd.read_csv(vp, low_memory=False, nrows=1)
            except Exception:  # noqa: BLE001
                continue
            if vdf.empty:
                continue
            if "edge_filter_note" in vdf.columns:
                note = str(vdf["edge_filter_note"].iloc[0] or "").strip()
                if note:
                    return note
            if "edge_filter_failed" in vdf.columns:
                try:
                    if bool(vdf["edge_filter_failed"].iloc[0]):
                        return "EDGE-UNFILTERED: edge safety check failed"
                except Exception:  # noqa: BLE001
                    pass
            break
        return ""

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
        edge_note = self._variability_edge_filter_note()
        if edge_note:
            rows.append(("Edge filter status", edge_note))
        return rows

    def _compress_image_for_pdf(self, 
        src_path: str | Path,
        max_width_px: int = 1200,
        jpeg_quality: int = 72,
        force_jpeg: bool = False,
    ) -> tuple[BytesIO, str]:
        """
        Open image, resize if wider than max_width_px (preserve aspect ratio),
        convert to JPEG, return (BytesIO, "JPEG"). RGBA is flattened onto white.
        ``force_jpeg`` reserved for API compatibility with the Phase 4 spec.
        """
        _ = force_jpeg
        src = Path(src_path)
        if not src.is_file():
            raise FileNotFoundError(str(src))
        try:
            from PIL import Image as PILImage
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Pillow required for PDF image compression") from exc

        digest = hashlib.md5(str(src.resolve()).encode("utf-8"), usedforsecurity=False).hexdigest()[:24]
        cdir = self.cache_dir / "pdf_embed"
        cdir.mkdir(parents=True, exist_ok=True)
        cpath = cdir / f"{digest}_w{int(max_width_px)}_q{int(jpeg_quality)}.jpg"
        if cpath.is_file() and cpath.stat().st_mtime >= src.stat().st_mtime:
            out = BytesIO(cpath.read_bytes())
            out.seek(0)
            return out, "JPEG"

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
            cpath.write_bytes(buf.getvalue())
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
    def _prepare_jpeg(self, src: Path, dst: Path, *, max_side_px: int = 1600, quality: int = 76) -> Path | None:
        """Convert/resize image to JPEG for smaller PDF + better compatibility."""
        try:
            from PIL import Image as PILImage  # pillow
        except Exception as exc:  # noqa: BLE001
            logging.warning("PIL (pillow) nie je dostupný, kompresiu obrázkov preskakujem (%s)", exc)
            return src if src.exists() else None

        try:
            src = Path(src)
            dst = Path(dst)
            if not src.exists():
                return None
            # cache validity: reuse if newer than source
            if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
                return dst

            with PILImage.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size
                m = max(w, h)
                if m > int(max_side_px) and m > 0:
                    scale = float(max_side_px) / float(m)
                    nw = max(1, int(round(w * scale)))
                    nh = max(1, int(round(h * scale)))
                    im = im.resize((nw, nh), resample=PILImage.Resampling.LANCZOS)

                dst.parent.mkdir(parents=True, exist_ok=True)
                im.save(dst, format="JPEG", quality=int(quality), optimize=True, progressive=True)
            return dst
        except Exception as exc:  # noqa: BLE001
            logging.warning("JPEG príprava zlyhala pre %s: %s", src, exc)
            return src if src.exists() else None
    def _plot_lightcurve_to_jpeg(self, lc_csv: Path, out_jpg: Path) -> Path | None:
        """Fallback: generate lightcurve plot from CSV when PNG is missing."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            logging.warning("matplotlib nie je dostupný, krivku z CSV nevygenerujem (%s)", exc)
            return None

        try:
            lc_csv = Path(lc_csv)
            out_jpg = Path(out_jpg)
            if not lc_csv.exists():
                return None
            # cache validity
            if out_jpg.exists() and out_jpg.stat().st_mtime >= lc_csv.stat().st_mtime:
                return out_jpg

            df = pd.read_csv(lc_csv, low_memory=False)
            if df.empty:
                return None
            # Prefer normal points
            if "flag" in df.columns:
                dfn = df[df["flag"].astype(str).eq("normal")].copy()
            else:
                dfn = df.copy()
            if dfn.empty:
                dfn = df

            xcol = None
            for c in ("bjd_tdb", "bjd", "hjd", "jd"):
                if c in dfn.columns:
                    xcol = c
                    break
            ycol = "mag_calib_ct" if "mag_calib_ct" in dfn.columns else ("mag_calib" if "mag_calib" in dfn.columns else None)
            if xcol is None or ycol is None:
                return None

            x = pd.to_numeric(dfn[xcol], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(dfn[ycol], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            x = x[ok]
            y = y[ok]
            if len(x) < 2:
                return None

            fig = plt.figure(figsize=(10.5, 4.2), dpi=150)
            try:
                ax = fig.gca()
                ax.scatter(x, y, s=6, c="#1f77b4", alpha=0.9, linewidths=0)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.25)
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                fig.tight_layout()
                out_png = out_jpg.with_suffix(".png")
                out_png.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_png, dpi=150)
            finally:
                plt.close(fig)
            # Convert to JPEG for size (match per-star LC embed limits)
            return self._prepare_jpeg(out_png, out_jpg, max_side_px=1000, quality=75)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Plot lightcurve zlyhal (%s): %s", lc_csv, exc)
            return None

    @staticmethod
    def _robust_rms_148mad(y: np.ndarray) -> float:
        arr = np.asarray(y, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return float("nan")
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return float(1.48 * mad)

    def _load_lc_xy_from_csv(self, lc_csv: Path) -> tuple[np.ndarray, np.ndarray, str, str] | None:
        lc_csv = Path(lc_csv)
        if not lc_csv.is_file():
            return None
        try:
            df = pd.read_csv(lc_csv, low_memory=False)
        except Exception:  # noqa: BLE001
            return None
        if df.empty:
            return None
        if "flag" in df.columns:
            dfn = df[df["flag"].astype(str).eq("normal")].copy()
        else:
            dfn = df.copy()
        if dfn.empty:
            dfn = df
        xcol = next((c for c in ("bjd_tdb", "bjd", "hjd", "jd") if c in dfn.columns), None)
        ycol = "mag_calib_ct" if "mag_calib_ct" in dfn.columns else ("mag_calib" if "mag_calib" in dfn.columns else None)
        if xcol is None or ycol is None:
            return None
        x = pd.to_numeric(dfn[xcol], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(dfn[ycol], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if len(x) < 2:
            return None
        return x, y, str(xcol), str(ycol)

    def _overlay_lc_cache_fresh(self, out_jpg: Path, sources: list[Path]) -> bool:
        out_jpg = Path(out_jpg)
        if not out_jpg.is_file():
            return False
        mt = float(out_jpg.stat().st_mtime)
        for src in sources:
            sp = Path(src)
            if sp.is_file() and sp.stat().st_mtime > mt:
                return False
        return True

    def _plot_lightcurve_overlay_to_jpeg(
        self,
        aperture_csv: Path,
        overlay_specs: list[tuple[str, Path]],
        out_jpg: Path,
    ) -> Path | None:
        """Primary-report only: aperture + optional PSF/adaptive on one axes."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            logging.warning("matplotlib nie je dostupný, overlay LC nevygenerujem (%s)", exc)
            return None

        aperture_csv = Path(aperture_csv)
        out_jpg = Path(out_jpg)
        src_paths = [aperture_csv] + [Path(p) for _lbl, p in overlay_specs]
        if self._overlay_lc_cache_fresh(out_jpg, src_paths):
            return out_jpg

        ap_loaded = self._load_lc_xy_from_csv(aperture_csv)
        if ap_loaded is None:
            return None
        x_ap, y_ap, xcol, ycol = ap_loaded

        series: list[tuple[str, np.ndarray, np.ndarray, str, str]] = [
            ("aperture", x_ap, y_ap, "#1f77b4", "o"),
        ]
        style_extra = {
            "PSF": ("#d62728", "^"),
            "Adaptive": ("#2ca02c", "s"),
        }
        for label, csv_path in overlay_specs:
            loaded = self._load_lc_xy_from_csv(Path(csv_path))
            if loaded is None:
                continue
            x_o, y_o, _, _ = loaded
            col, mk = style_extra.get(label, ("#9467bd", "D"))
            series.append((label, x_o, y_o, col, mk))

        if len(series) < 2:
            return self._plot_lightcurve_to_jpeg(aperture_csv, out_jpg)

        stats: dict[str, dict[str, float]] = {}
        for name, _x, y_s, _c, _m in series:
            stats[name] = {
                "rms": self._robust_rms_148mad(y_s),
                "med": float(np.median(y_s)) if len(y_s) else float("nan"),
            }

        try:
            fig = plt.figure(figsize=(10.5, 4.2), dpi=150)
            try:
                ax = fig.gca()
                for name, x_s, y_s, col, mk in series:
                    rms_v = stats[name]["rms"]
                    leg = f"{name}  RMS={rms_v:.4f}" if np.isfinite(rms_v) else name
                    ax.scatter(x_s, y_s, s=6, c=col, alpha=0.85, linewidths=0, marker=mk, label=leg)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.25)
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.legend(loc="best", fontsize=7, framealpha=0.9)

                note_lines: list[str] = []
                ap_rms = stats.get("aperture", {}).get("rms", float("nan"))
                ap_med = stats.get("aperture", {}).get("med", float("nan"))
                for alt in ("PSF", "Adaptive"):
                    if alt not in stats:
                        continue
                    alt_rms = stats[alt]["rms"]
                    alt_med = stats[alt]["med"]
                    if np.isfinite(ap_rms) and np.isfinite(alt_rms) and ap_rms > 0:
                        note_lines.append(f"RMS ratio {alt}/aperture: {alt_rms / ap_rms:.3f}")
                    if np.isfinite(ap_med) and np.isfinite(alt_med):
                        dmed = float(alt_med - ap_med)
                        if abs(dmed) >= 0.010:
                            note_lines.append(f"Median offset {alt}\u2212aperture: {dmed:+.4f} mag")
                if note_lines:
                    ax.text(
                        0.02,
                        0.02,
                        "\n".join(note_lines),
                        transform=ax.transAxes,
                        fontsize=6.5,
                        va="bottom",
                        ha="left",
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
                    )

                fig.tight_layout()
                out_png = out_jpg.with_suffix(".png")
                out_png.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_png, dpi=150)
            finally:
                plt.close(fig)
            return self._prepare_jpeg(out_png, out_jpg, max_side_px=1000, quality=75)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Plot LC overlay zlyhal (%s): %s", aperture_csv, exc)
            return None

    def _resolve_primary_lc_image(
        self,
        cid: str,
        lc_png: Path,
        lc_csv: Path,
    ) -> Path | None:
        """LC JPEG for embed: overlay on primary report when PSF-family CSVs exist."""
        if self._photometry_method != "aperture":
            _method_lc = lc_csv_path(self.lc_dir, cid, self._photometry_method)
            if _method_lc.is_file():
                lc_csv = _method_lc
            _lc_cache = self.cache_dir / f"lc_{cid}_{self._photometry_method}.jpg"
            return self._plot_lightcurve_to_jpeg(lc_csv, _lc_cache)

        overlay_specs: list[tuple[str, Path]] = []
        for label, method in (("PSF", "psf"), ("Adaptive", "adaptive")):
            op = lc_csv_path(self.lc_dir, cid, method)
            if op.is_file():
                overlay_specs.append((label, op))
        if overlay_specs:
            _lc_cache = self.cache_dir / f"lc_{cid}_overlay.jpg"
            img = self._plot_lightcurve_overlay_to_jpeg(lc_csv, overlay_specs, _lc_cache)
            if img is not None:
                return img

        _lc_cache = self.cache_dir / f"lc_{cid}.jpg"
        if lc_png.exists():
            return self._prepare_jpeg(lc_png, _lc_cache, max_side_px=1000, quality=75)
        return self._plot_lightcurve_to_jpeg(lc_csv, _lc_cache)

    def _page_footer(self, c: "canvas.Canvas") -> None:
        try:
            c.setFont(self.FONT_REG, 9)
            c.setFillColor(self.colors.HexColor("#1a1a2e"))
            left_txt = f"VYVAR — {self._report_draft_lbl} — {self.obs_group}"
            c.drawString(self.M_LEFT, 0.45 * self.cm, left_txt)
            c.drawRightString(self.PAGE_W - self.M_RIGHT, 0.45 * self.cm, f"Page {c.getPageNumber()}")
            c.setFillColor(self.colors.black)
        except Exception:  # noqa: BLE001
            pass

    def _layout_y_floor(self) -> float:
        """Minimum y (pt) for body content — leave room for footer."""
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
    ) -> Any:
        cache_key = f"{name}|{fontName or self.FONT_REG}|{fontSize}|{leading}|{textColor}"
        if cache_key in self._para_styles:
            return self._para_styles[cache_key]
        sty = self.ParagraphStyle(
            name=f"vyvar_{name}_{len(self._para_styles)}",
            fontName=fontName or self.FONT_REG,
            fontSize=float(fontSize),
            leading=float(leading if leading is not None else fontSize * 1.25),
            alignment=self.TA_LEFT,
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
            return "—"
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
        n_combined_edge = 0
        try:
            from config import AppConfig
            from ui_variability import count_edge_safe_combined_candidates

            cfg_d = AppConfig().to_dict()
            n_combined_edge = int(
                count_edge_safe_combined_candidates(
                    rms_df,
                    vdi_df if isinstance(vdi_df, pd.DataFrame) else pd.DataFrame(),
                    self.platesolve_dir,
                    cfg_d,
                )
            )
        except Exception:  # noqa: BLE001
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
        compress_raster: bool = True,
        force_jpeg: bool = False,
    ) -> None:
        """Draw image fitted into (w,h) box, top-left. Raster sources are JPEG-compressed before ImageReader."""
        del compress_raster  # kept for API compatibility; compression is always applied when Pillow works
        try:
            if not img_path or not Path(img_path).exists():
                return
            ip = Path(img_path)
            mw, jq = self._IMAGE_PDF_SETTINGS.get(image_type, self._IMAGE_PDF_SETTINGS["default"])
            try:
                jbuf, _fmt = self._compress_image_for_pdf(ip, mw, jq, force_jpeg=force_jpeg)
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

    def _sanitize_katalogy_pdf_line(self, line: str) -> str:
        s = str(line or "").strip()
        if not s:
            return ""
        if "žiadny záznam" in s.lower():
            return "—"
        return s.replace("ďalších", "more").replace("Ďalších", "more")
    def _katalogy_positive_lines(self, text: Any) -> list[str]:
        out: list[str] = []
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("🔭"):
                continue
            if "žiadny záznam" in line:
                continue
            if "no match" in line.lower():
                continue
            out.append(line)
        return out
    def _katalogy_cell_for_pdf(self, text: Any) -> str:
        pos = [self._sanitize_katalogy_pdf_line(x) for x in self._katalogy_positive_lines(text)]
        pos = [x for x in pos if x and x != "—"]
        if not pos:
            return "—"
        head = pos[:3]
        extra = len(pos) - 3
        body = "\n".join(head)
        if extra > 0:
            return body + f"\n(+{extra} more)"
        return body
    def _katalogy_row_has_positive(self, text: Any) -> bool:
        return bool(self._katalogy_positive_lines(text))
    def _draw_hockey_stick_png(self, 
        *,
        photometry_dir_hs: Path,
        platesolve_dir_hs: Path,
        cache_dir_hs: Path,
    ) -> Path | None:
        """Build RMS hockey-stick PNG; prefer color-coded report at hockey_stick_report.png."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.lines import Line2D  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        def _legacy_simple_plot() -> Path | None:
            df0 = pd.DataFrame()
            ps_csv = photometry_dir_hs / "photometry_summary.csv"
            ps_xlsx = photometry_dir_hs / "photometry_summary.xlsx"
            try:
                if ps_csv.exists():
                    df0 = pd.read_csv(ps_csv, low_memory=False)
                elif ps_xlsx.exists():
                    df0 = pd.read_excel(ps_xlsx)
            except Exception:  # noqa: BLE001
                return None
            if df0.empty:
                return None
            rename_map: dict[str, str] = {}
            for c0 in list(df0.columns):
                cl = str(c0).strip().lower()
                if cl in ("gaia_g_mag", "g_mag", "gaia_mag", "catalog_mag") or cl in ("mag_median", "lc_median_mag"):
                    rename_map[c0] = "mag"
                elif cl in ("rms", "lc_rms"):
                    rename_map[c0] = "lc_rms"
                elif cl in ("rms_pct", "rms_percent", "rms_%"):
                    rename_map[c0] = "rms_pct"
                elif cl in ("id", "gaia_id"):
                    rename_map[c0] = "catalog_id"
            if rename_map:
                df0 = df0.rename(columns=rename_map)
            _co_mask0 = _is_catalog_only(df0)
            if _co_mask0.any():
                logging.info(
                    "[HOCKEY STICK] Excluding %d catalog_only stars "
                    "(no DAO detection — sky noise only)",
                    int(_co_mask0.sum()),
                )
                df0 = df0.loc[~_co_mask0].copy()
            mag_c = "mag" if "mag" in df0.columns else next(
                (c for c in df0.columns if "mag" in str(c).lower() and "calib" not in str(c).lower()),
                None,
            )
            id_c = "catalog_id" if "catalog_id" in df0.columns else next(
                (c for c in df0.columns if "catalog_id" in str(c).lower()), None
            )
            if "rms_pct" in df0.columns:
                rms_c = "rms_pct"
            elif "lc_rms" in df0.columns:
                rms_c = "lc_rms"
            else:
                rms_c = next(
                    (c for c in df0.columns if "lc_rms" in str(c).lower() or "rms_pct" in str(c).lower()),
                    None,
                )
            if not mag_c or not id_c or not rms_c:
                return None
            df0[mag_c] = pd.to_numeric(df0[mag_c], errors="coerce")
            df0[rms_c] = pd.to_numeric(df0[rms_c], errors="coerce")
            df0 = df0.dropna(subset=[mag_c, rms_c]).copy()
            if df0.empty:
                return None
            if rms_c == "lc_rms":
                df0["_rms_plot"] = df0[rms_c].astype(float) * 100.0
            else:
                df0["_rms_plot"] = df0[rms_c].astype(float)
            cands0 = {self._norm_cid(str(x)) for x in self._candidates_set if str(x).strip()}
            fig0, ax0 = plt.subplots(figsize=(10, 4.5))
            try:
                fig0.patch.set_facecolor("white")
                if id_c:
                    m0 = ~df0[id_c].astype(str).map(lambda s: self._norm_cid(str(s))).isin(cands0)
                else:
                    m0 = np.ones(len(df0), dtype=bool)
                ax0.scatter(df0.loc[m0, mag_c], df0.loc[m0, "_rms_plot"], s=15, alpha=0.5, color="#4A90D9", label="Stars")
                if id_c and cands0:
                    m1 = df0[id_c].astype(str).map(lambda s: self._norm_cid(str(s))).isin(cands0)
                    if bool(m1.any()):
                        ax0.scatter(
                            df0.loc[m1, mag_c],
                            df0.loc[m1, "_rms_plot"],
                            s=60,
                            alpha=0.9,
                            color="#E24B4A",
                            zorder=5,
                            label="Candidates",
                        )
                ax0.set_xlabel("Magnitude")
                ax0.set_ylabel("RMS (%)")
                ax0.set_title("RMS Hockey Stick — variability candidates highlighted")
                ax0.legend(framealpha=0.8)
                ax0.grid(True, alpha=0.3)
                outp = cache_dir_hs / "hockey_stick.png"
                fig0.savefig(outp, dpi=96, bbox_inches="tight")
            finally:
                plt.close(fig0)
            return outp if outp.exists() else None

        df = pd.DataFrame()
        ps_csv = photometry_dir_hs / "photometry_summary.csv"
        ps_xlsx = photometry_dir_hs / "photometry_summary.xlsx"
        try:
            if ps_csv.exists():
                df = pd.read_csv(ps_csv, low_memory=False)
            elif ps_xlsx.exists():
                df = pd.read_excel(ps_xlsx)
        except Exception:  # noqa: BLE001
            return _legacy_simple_plot()
        if df.empty:
            return _legacy_simple_plot()

        rename_map: dict[str, str] = {}
        for c0 in list(df.columns):
            cl = str(c0).strip().lower()
            if cl in ("gaia_g_mag", "g_mag", "gaia_mag", "catalog_mag") or cl in ("mag_median", "lc_median_mag"):
                rename_map[c0] = "mag"
            elif cl in ("rms", "lc_rms"):
                rename_map[c0] = "lc_rms"
            elif cl in ("rms_pct", "rms_percent", "rms_%"):
                rename_map[c0] = "rms_pct"
            elif cl in ("expected_rms_pct", "expected_rms", "expected_noise_pct"):
                rename_map[c0] = "expected_rms_pct"
            elif cl in ("id", "gaia_id"):
                rename_map[c0] = "catalog_id"
        if rename_map:
            df = df.rename(columns=rename_map)

        _co_mask = _is_catalog_only(df)
        if _co_mask.any():
            logging.info(
                "[HOCKEY STICK] Excluding %d catalog_only stars "
                "(no DAO detection — sky noise only)",
                int(_co_mask.sum()),
            )
            df = df.loc[~_co_mask].copy()

        mag_col = "mag" if "mag" in df.columns else next(
            (c for c in df.columns if "mag" in str(c).lower() and "calib" not in str(c).lower()),
            None,
        )
        id_col = "catalog_id" if "catalog_id" in df.columns else next(
            (c for c in df.columns if "catalog_id" in str(c).lower()), None
        )
        if "rms_pct" in df.columns:
            rms_col = "rms_pct"
        elif "lc_rms" in df.columns:
            rms_col = "lc_rms"
        else:
            rms_col = next(
                (c for c in df.columns if "lc_rms" in str(c).lower() or "rms_pct" in str(c).lower()),
                None,
            )
        if not mag_col or not rms_col or not id_col:
            return _legacy_simple_plot()

        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce")
        df[rms_col] = pd.to_numeric(df[rms_col], errors="coerce")
        df = df.dropna(subset=[mag_col, rms_col]).copy()
        if df.empty:
            return _legacy_simple_plot()

        if rms_col == "lc_rms":
            df["_rms_plot"] = df[rms_col].astype(float) * 100.0
        else:
            df["_rms_plot"] = df[rms_col].astype(float)

        # Match UI hockey stick: mag 8–15, RMS 0.8–10 %, log Y-axis.
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce")
        df["_rms_plot"] = pd.to_numeric(df["_rms_plot"], errors="coerce")
        df = df[
            (df[mag_col] >= 8.0)
            & (df[mag_col] <= 15.0)
            & (df["_rms_plot"] >= 0.8)
            & (df["_rms_plot"] <= 10.0)
        ].copy()
        if df.empty:
            return _legacy_simple_plot()

        df["_nid"] = df[id_col].map(lambda s: self._norm_cid(str(s)))
        df = df.reset_index(drop=True)

        vpaths = [photometry_dir_hs / "variability_candidates.csv", platesolve_dir_hs / "variability_candidates.csv"]
        vp = next((p for p in vpaths if p.is_file()), None)
        cand_ids: set[str] = set()
        katalogy_by_id: dict[str, str] = {}
        if vp is not None:
            try:
                vdf = pd.read_csv(vp, low_memory=False, dtype=_GAIA_ID_DTYPE)
            except Exception:  # noqa: BLE001
                vdf = pd.DataFrame()
            id_v = self._col_pick(vdf, ("catalog_id", "Catalog_ID", "gaia_id")) if not vdf.empty else None
            kat_v = self._col_pick(vdf, ("katalogy", "katalógy", "katalogy", "catalog_match")) if not vdf.empty else None
            if id_v and id_v in vdf.columns:
                for _, vr in vdf.iterrows():
                    nk = self._norm_cid(str(vr.get(id_v, "") or ""))
                    if nk:
                        cand_ids.add(nk)
                        if kat_v and kat_v in vdf.columns:
                            katalogy_by_id[nk] = str(vr.get(kat_v, "") or "")

        vsx_n_col = next((c for c in df.columns if str(c).lower() == "vsx_name"), None)
        vsx_t_col = next((c for c in df.columns if str(c).lower() == "vsx_type"), None)

        def _known_vsx_row(i: int) -> bool:
            """Known VSX: real catalog name + VSX type (not Gaia ROT-only labels)."""
            if not vsx_n_col or vsx_n_col not in df.columns:
                return False
            row = df.iloc[i]
            vn = str(row.get(vsx_n_col, "") or "").strip()
            if not vn or vn in ("nan", "—", "-", "None"):
                return False
            vt = ""
            if vsx_t_col and vsx_t_col in df.columns:
                vt = str(row.get(vsx_t_col, "") or "").strip()
            if vt in ("", "nan", "—", "-", "None", "ROT"):
                return False
            return True

        nrows = len(df)
        is_cand = np.array([str(df.iloc[i]["_nid"]) in cand_ids for i in range(nrows)], dtype=bool)
        is_known = np.array([_known_vsx_row(i) for i in range(nrows)], dtype=bool)
        pos_kat = np.array(
            [self._katalogy_row_has_positive(katalogy_by_id.get(str(df.iloc[i]["_nid"]), "")) for i in range(nrows)],
            dtype=bool,
        )

        mask_stable = ~is_cand & ~is_known
        mask_known = is_known
        mask_orange = is_cand & ~is_known & pos_kat
        mask_red = is_cand & ~is_known & ~pos_kat

        fig, ax = plt.subplots(figsize=(10, 4.5))
        try:
            fig.patch.set_facecolor("white")

            if bool(mask_stable.any()):
                ax.scatter(
                    df.loc[mask_stable, mag_col],
                    df.loc[mask_stable, "_rms_plot"],
                    s=10,
                    alpha=0.45,
                    color="#16a34a",
                    zorder=1,
                )
            if bool(mask_known.any()):
                ax.scatter(
                    df.loc[mask_known, mag_col],
                    df.loc[mask_known, "_rms_plot"],
                    s=40,
                    alpha=0.95,
                    marker="x",
                    linewidths=1.0,
                    color="#f59e0b",
                    zorder=5,
                )
            if bool(mask_orange.any()):
                ax.scatter(
                    df.loc[mask_orange, mag_col],
                    df.loc[mask_orange, "_rms_plot"],
                    s=64,
                    alpha=0.9,
                    marker="o",
                    color="#f97316",
                    zorder=5,
                )
            if bool(mask_red.any()):
                ax.scatter(
                    df.loc[mask_red, mag_col],
                    df.loc[mask_red, "_rms_plot"],
                    s=64,
                    alpha=0.9,
                    marker="o",
                    color="#ef4444",
                    zorder=5,
                )

            # Expected noise curve (same mag window as scatter points).
            if "expected_rms_pct" in df.columns:
                curve = df[[mag_col, "expected_rms_pct"]].dropna().sort_values(mag_col)
                curve = curve[
                    (curve[mag_col] >= 8.0)
                    & (curve[mag_col] <= 15.0)
                    & (curve["expected_rms_pct"] >= 0.8)
                    & (curve["expected_rms_pct"] <= 10.0)
                ]
                if not curve.empty:
                    ax.plot(
                        curve[mag_col],
                        curve["expected_rms_pct"],
                        color="#888888",
                        linewidth=2,
                        zorder=2,
                        label="Expected noise",
                    )

            import matplotlib.ticker as ticker  # noqa: PLC0415

            ax.set_yscale("log")
            ax.set_ylim(0.8, 10.0)
            ax.set_xlim(8.0, 15.0)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))
            ax.set_xlabel("Magnitude")
            ax.set_ylabel("RMS (%)")
            ax.set_title("RMS Hockey Stick — variability candidates highlighted")
            ax.grid(True, alpha=0.3, color="#9ca3af", which="both")

            legend_handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#16a34a", markersize=8, linestyle="None", label="Stable stars"),
                Line2D([0], [0], marker="x", color="#f59e0b", linestyle="None", markersize=10, markeredgewidth=1.5, label="Known VSX match"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#f97316", markersize=8, linestyle="None", label="Candidate (catalog match)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#ef4444", markersize=8, linestyle="None", label="Candidate (no catalog match)"),
            ]
            ax.legend(handles=legend_handles, framealpha=0.9, loc="best")

            report_png = photometry_dir_hs / "hockey_stick_report.png"
            fig.savefig(report_png, dpi=96, bbox_inches="tight")
        finally:
            plt.close(fig)
        return report_png if report_png.is_file() else _legacy_simple_plot()
    def _report_cover_page(self, c: "canvas.Canvas") -> None:
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 22)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.6 * self.cm, self._report_title)
        c.setFillColor(self.colors.black)

        y -= 1.25 * self.cm
        logo_path = None
        try:
            logo_path = (Path(__file__).resolve().parent / "img" / "VYVAR_logo.png").resolve()
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
        ps_txt = f"{ps:.3f} arcsec/px" if np.isfinite(ps) else "—"
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
            pass
        c.drawString(self.M_LEFT, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
            ("Median lc_rms", f"{self.med_rms:.4f}" if np.isfinite(self.med_rms) else "—"),
            ("RMS < 0.05 mag", f"{self.rms_lt_005:d}"),
            ("Avg good comp", f"{self.avg_good_comp:.2f}" if np.isfinite(self.avg_good_comp) else "—"),
            ("Best lc_rms", f"{self.best_rms:.4f}" if np.isfinite(self.best_rms) else "—"),
            ("Worst lc_rms", f"{self.worst_rms:.4f}" if np.isfinite(self.worst_rms) else "—"),
            ("Avg BP-RP", f"{self.avg_bp_rp:.3f}" if np.isfinite(self.avg_bp_rp) else "—"),
            ("Setups", f"{self.setups:d}"),
            ("Exports", f"AAVSO {self.n_aavso:d} · VAR.ASTRO {self.n_varastro:d}"),
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
        fwhm_txt = f"{self.fwhm_px:.3f}px" if np.isfinite(self.fwhm_px) else "—"
        ap_txt = f"{self.aperture_px:.2f}px" if np.isfinite(self.aperture_px) else "—"
        c.setFont(self.FONT_REG, 11)
        c.drawString(self.M_LEFT, y, f"FWHM: {fwhm_txt}   |   Aperture: {ap_txt}")
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
            lc_col = "lc_csv" if "lc_csv" in self.summary_df.columns else None
            if not am_vals and lc_col:
                for _, sr in self.summary_df.iterrows():
                    pth = str(sr.get(lc_col) or "").strip()
                    if not pth:
                        continue
                    lp = Path(pth)
                    if not lp.is_file():
                        continue
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
                kv.append(("Airmass (min / max / median)", "—"))
            sky_m = float("nan")
            if not qc_df0.empty and "SKY_LEVEL" in qc_df0.columns:
                sky_m = float(np.nanmedian(pd.to_numeric(qc_df0["SKY_LEVEL"], errors="coerce")))
            kv.append(("Sky background (median)", f"{sky_m:.2f}" if np.isfinite(sky_m) else "—"))
            nq = self._night_qc
            n_used = int(nq.get("n_used", 0) or 0)
            n_rej = int(nq.get("n_rejected", 0) or 0)
            n_tot = int(nq.get("n_total", 0) or 0)
            if n_tot > 0:
                kv.append(("Frames (used / rejected / total)", f"{n_used:d} / {n_rej:d} / {n_tot:d}"))
            else:
                kv.append(("Frames (used / rejected / total)", "—"))
            bjd_min = float(nq.get("bjd_min", float("nan")))
            bjd_max = float(nq.get("bjd_max", float("nan")))
            if np.isfinite(bjd_min) and np.isfinite(bjd_max):
                kv.append(("BJD span (session)", f"{bjd_min:.5f} → {bjd_max:.5f}"))
            else:
                kv.append(("BJD span (session)", "—"))
            fmin = float(nq.get("fwhm_min", float("nan")))
            fmed = float(nq.get("fwhm_med", float("nan")))
            fmax = float(nq.get("fwhm_max", float("nan")))
            if np.isfinite(fmed):
                kv.append(
                    (
                        "FWHM px (min / med / max)",
                        f"{fmin:.3f} / {fmed:.3f} / {fmax:.3f}",
                    )
                )
            else:
                kv.append(("FWHM px (min / med / max)", "—"))
            nf_med = float("nan")
            if "n_frames" in self.summary_df.columns:
                nf_med = float(np.nanmedian(pd.to_numeric(self.summary_df["n_frames"], errors="coerce")))
            kv.append(("LC points (median per target)", f"{int(round(nf_med)):d}" if np.isfinite(nf_med) else "—"))
            ms_fits = self.platesolve_dir / "MASTERSTAR.fits"
            if ms_fits.is_file():
                try:
                    from astropy.io import fits

                    with fits.open(ms_fits, memmap=False) as hdul:
                        _hsw = hdul[0].header.get("VY_HSWN")
                        _hsep = hdul[0].header.get("VY_HSEP")
                    if _hsw is not None and bool(_hsw):
                        _sep_s = "—"
                        try:
                            _sep_s = f"{float(_hsep[0] if isinstance(_hsep, tuple) else _hsep):.3f}°"
                        except (TypeError, ValueError):
                            pass
                        kv.append(
                            (
                                "Plate-solve hint offset (warning)",
                                f"Stale pointing hint vs solved WCS ({_sep_s}); solve verified via catalog recovery.",
                            )
                        )
                except Exception:  # noqa: BLE001
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
        _comp_factor = float(getattr(_cfg_m, "aperture_comp_factor", 1.1) or 1.1) if _cfg_m else 1.1
        _var_factor = float(getattr(_cfg_m, "aperture_variable_factor", 1.0) or 1.0) if _cfg_m else 1.0
        _pu_ver = "?"
        _ap_ver = "?"
        try:
            import photutils

            _pu_ver = str(getattr(photutils, "__version__", "?"))
        except Exception:  # noqa: BLE001
            pass
        try:
            import astropy

            _ap_ver = str(getattr(astropy, "__version__", "?"))
        except Exception:  # noqa: BLE001
            pass

        _period_used = bool(self._accepted_periods) or bool(
            isinstance(self._var_results, dict)
            and any(
                isinstance(v, dict) and v.get("period") not in (None, "", float("nan"))
                for v in self._var_results.values()
            )
        )
        _var_det_used = bool(self._candidates_set) or (
            (self.photometry_dir / "variability_candidates.csv").is_file()
        )
        _cite_ctx = build_run_citation_context(
            _cfg_m,
            pipeline_meta=self._pipeline_meta,
            targets_df=self.at_df,
            period_analysis=_period_used,
            variability_detection=_var_det_used,
            tess_used=bool(self._tess_results),
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
                f"  SNR-optimal aperture: median target r={_ap_med:.2f}px "
                f"(var x{_var_factor:.2f}); comp/check r x{_comp_factor:.2f}",
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
            return None
    def _load_obs_files_for_obs(self, ) -> pd.DataFrame:
        """Load QA metrics from DB (OBS_FILES) for this draft + setup."""
        draft_id = self._draft_id_from_dirname()
        if draft_id is None:
            return pd.DataFrame()
        try:
            from config import AppConfig
            from database import VyvarDatabase
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
        try:
            cfg = AppConfig()
            db = VyvarDatabase(cfg.database_path)
            rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
        except Exception:  # noqa: BLE001
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
        return dfo
    def _load_qc_metrics_for_obs(self, ) -> pd.DataFrame:
        from pipeline import find_qc_metrics_csv

        qc_csv = find_qc_metrics_csv(self.draft_dir, app_config=None)
        if qc_csv is None:
            return pd.DataFrame()
        try:
            dfq = pd.read_csv(qc_csv, low_memory=False)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
        if dfq.empty:
            return pd.DataFrame()
        if "dst" in dfq.columns:
            m = dfq["dst"].astype(str).str.contains(str(self.obs_group), regex=False)
            dfq = dfq.loc[m].copy()
        if dfq.empty:
            return pd.DataFrame()
        dfq["_dst_name"] = dfq["dst"].astype(str).apply(lambda s: Path(s).name) if "dst" in dfq.columns else ""
        dfq["FWHM_PX"] = pd.to_numeric(dfq.get("self.fwhm_px"), errors="coerce")
        dfq["SKY_LEVEL"] = pd.to_numeric(dfq.get("bg_median"), errors="coerce")
        dfq["ELONGATION"] = pd.to_numeric(dfq.get("elongation"), errors="coerce")
        if "n_stars_detected" in dfq.columns:
            dfq["STAR_COUNT"] = pd.to_numeric(dfq.get("n_stars_detected"), errors="coerce")
        else:
            dfq["STAR_COUNT"] = pd.to_numeric(dfq.get("n_sources"), errors="coerce")
        dfq = dfq.reset_index(drop=True)
        dfq["frame_index"] = np.arange(1, len(dfq) + 1, dtype=int)
        return dfq
    def _compute_masterstar_score(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)
        if df.empty:
            return score

        def _norm_inverse(s: pd.Series) -> pd.Series:
            mn, mx = float(s.min()), float(s.max())
            if not (np.isfinite(mn) and np.isfinite(mx) and mx > mn):
                return pd.Series(1.0, index=s.index)
            return 1.0 - (s - mn) / (mx - mn)

        def _norm_direct(s: pd.Series) -> pd.Series:
            mn, mx = float(s.min()), float(s.max())
            if not (np.isfinite(mn) and np.isfinite(mx) and mx > mn):
                return pd.Series(1.0, index=s.index)
            return (s - mn) / (mx - mn)

        fwhm = pd.to_numeric(df.get("FWHM_PX"), errors="coerce")
        elong = pd.to_numeric(df.get("ELONGATION"), errors="coerce")
        stars = pd.to_numeric(df.get("STAR_COUNT"), errors="coerce")
        sky = pd.to_numeric(df.get("SKY_LEVEL"), errors="coerce")

        if fwhm.notna().sum() >= 2:
            score += 0.45 * _norm_inverse(fwhm.fillna(fwhm.max()))
        if elong.notna().sum() >= 2:
            score += 0.30 * _norm_inverse(elong.fillna(elong.max()))
        if stars.notna().sum() >= 2:
            score += 0.15 * _norm_direct(stars.fillna(stars.min()))
        if sky.notna().sum() >= 2:
            score += 0.10 * _norm_inverse(sky.fillna(sky.max()))
        return score
    def _qa_fwhm_limit_px(self, qc_df: pd.DataFrame) -> float:
        """Default FWHM threshold matching ``ui_quality_dashboard`` first-load seed (not session slider tweaks)."""
        if qc_df is None or qc_df.empty:
            return float("nan")
        for col in (
            "fwhm_limit_px",
            "FWHM_LIMIT_PX",
            "FWHM_LIMIT",
            "fwhm_limit",
            "reject_self.fwhm_px",
            "VYVAR_REJECT_FWHM",
        ):
            if col not in qc_df.columns:
                continue
            v0 = pd.to_numeric(qc_df[col].iloc[0], errors="coerce")
            fv = float(v0)
            if np.isfinite(fv) and fv > 0.0:
                return float(round(fv, 4))
        try:
            from config import AppConfig
            from photometry_core import compute_auto_fwhm_limit

            cfg = AppConfig()
        except Exception:  # noqa: BLE001
            return float("nan")
        arr = pd.to_numeric(qc_df.get("FWHM_PX"), errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        n = int(arr.size)
        if n < 2:
            return float("nan")
        auto_on = bool(getattr(cfg, "auto_fwhm_enabled", True))
        k = float(getattr(cfg, "auto_fwhm_k_factor", 1.5))
        if n >= 3 and auto_on:
            ar = compute_auto_fwhm_limit(arr, k=k)
            lim = ar.get("auto_limit")
            if lim is not None:
                lf = float(lim)
                if np.isfinite(lf) and lf > 0.0:
                    return lf
            return float("nan")
        if n >= 5 and not auto_on:
            return float(round(float(np.median(arr) * 1.05), 2))
        return float("nan")
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
            return None
        d = np.zeros(len(qc_df), dtype=float)
        wsum = 0.0
        if np.isfinite(fwhm):
            d += np.square(pd.to_numeric(qc_df["FWHM_PX"], errors="coerce").to_numpy(dtype=float) - float(fwhm))
            wsum += 1.0
        if np.isfinite(elong):
            d += np.square(
                pd.to_numeric(qc_df["ELONGATION"], errors="coerce").to_numpy(dtype=float) - float(elong)
            )
            wsum += 1.0
        if np.isfinite(nstar):
            d += np.square(
                pd.to_numeric(qc_df["STAR_COUNT"], errors="coerce").to_numpy(dtype=float) - float(nstar)
            )
            wsum += 1.0
        if np.isfinite(sky):
            d += np.square(pd.to_numeric(qc_df["SKY_LEVEL"], errors="coerce").to_numpy(dtype=float) - float(sky))
            wsum += 1.0
        if wsum <= 0.0:
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
            return

        # Charts use per-frame QC from OBS_FILES (FWHM_PX vs frame_index), same source as FITS QA dashboard.
        used = self._resolve_masterstar_used_frame(qc_df)

        # FWHM limit line: same default as FITS QA dashboard (auto MAD when enabled, else median×1.05 for n≥5).
        fwhm_limit = float(self._qa_fwhm_limit_px(qc_df))
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
            if lim_active:
                col = np.where(np.isfinite(f) & (f <= float(fwhm_limit)), "#27ae60", "#c0392b")
            else:
                col = np.where(np.isfinite(f), "#27ae60", "#95a5a6")
            ax1.plot(x, f, color="#6c5ce7", linewidth=1.1, alpha=0.65)
            ax1.scatter(x, f, c=col, s=20)
            if lim_active:
                ax1.axhline(y=float(fwhm_limit), color="#c0392b", linestyle="--", linewidth=1.3)
            ax1.set_xlabel("Frame Index")
            ax1.set_ylabel("FWHM (px)")
            ax1.set_title("FWHM — green ≤ limit, red > limit")
            ax1.grid(True, alpha=0.25)

            # Sky plot: prefer DB flag (REJECTED_AUTO) if available, else robust outlier visualization.
            sky = pd.to_numeric(qc_df["SKY_LEVEL"], errors="coerce").to_numpy(dtype=float)
            sky_out = None
            if "REJECTED_AUTO" in qc_df.columns:
                ra = pd.to_numeric(qc_df["REJECTED_AUTO"], errors="coerce").fillna(0).to_numpy(dtype=float)
                sky_out = ra.astype(int) > 0
            if sky_out is None:
                sky_med = float(np.nanmedian(sky)) if np.isfinite(np.nanmedian(sky)) else float("nan")
                sky_mad = float(np.nanmedian(np.abs(sky - sky_med))) if np.isfinite(sky_med) else float("nan")
                sky_sigma = float(sky_mad / 0.6745) if np.isfinite(sky_mad) and sky_mad > 0 else float("nan")
                sky_out = np.isfinite(sky) & np.isfinite(sky_med) & np.isfinite(sky_sigma) & (
                    np.abs(sky - sky_med) > 5.0 * sky_sigma
                )
            col2 = np.where(sky_out, "#c0392b", "#27ae60")
            ax2.plot(x, sky, color="#6c5ce7", linewidth=1.1, alpha=0.65)
            ax2.scatter(x, sky, c=col2, s=20)
            ax2.set_xlabel("Frame Index")
            ax2.set_ylabel("Sky level (ADU)")
            ax2.set_title("Background sky level (red = auto-outlier)")
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
            c.drawString(self.M_LEFT, y0, "Used frame: — (could not infer from available metadata)")
        c.setFillColor(self.colors.black)

        self._page_footer(c)
        c.showPage()
    def _format_comp_catalog_id(self, row: Any) -> str:
        bv_src = str(row.get("bv_source", "") or "").strip().lower()
        cid = self._norm_cid(row.get("catalog_id", ""))
        if not cid:
            return "—"
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

    def _comp_rms_p2p_map_from_proc(self, comp_ids: list[str]) -> dict[str, float]:
        """Report-only Abbe p2p from per-frame dao_flux (same definition as Phase-2A stability)."""
        proc_dir = self._proc_csv_dir()
        if proc_dir is None or not comp_ids:
            return {}
        want = {_norm_cid(x) for x in comp_ids if _norm_cid(x)}
        flux_by_cid: dict[str, list[tuple[float, float]]] = {c: [] for c in want}
        for fp in sorted(glob.glob(str(proc_dir / "proc_*.csv"))):
            try:
                df = pd.read_csv(
                    fp,
                    dtype={"catalog_id": str},
                    usecols=lambda c: c in {"catalog_id", "dao_flux", "flux", "bjd_tdb_mid", "jd_mid", "hjd_mid"},
                )
            except Exception:  # noqa: BLE001
                continue
            if df.empty or "catalog_id" not in df.columns:
                continue
            df["catalog_id"] = df["catalog_id"].map(_norm_cid)
            tcol = next((c for c in ("bjd_tdb_mid", "jd_mid", "hjd_mid") if c in df.columns), None)
            tval = float(pd.to_numeric(df[tcol], errors="coerce").median()) if tcol else float("nan")
            flux_col = "dao_flux" if "dao_flux" in df.columns else "flux"
            sub = df[df["catalog_id"].isin(want)]
            for _, row in sub.iterrows():
                cid = str(row["catalog_id"])
                fx = float(pd.to_numeric(row.get(flux_col), errors="coerce"))
                if math.isfinite(fx) and fx > 0 and math.isfinite(tval):
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
                        nk = _norm_cid(ck)
                        if not nk:
                            continue
                        ent = raw.get(ck, raw.get(nk))
                        if isinstance(ent, dict) and math.isfinite(float(ent.get("rms_p2p", float("nan")))):
                            parsed[nk] = float(ent["rms_p2p"])
                        else:
                            v = self._rms_p2p_from_quality_note(info.get("note", ""))
                            if math.isfinite(v):
                                parsed[nk] = v
            except Exception:  # noqa: BLE001
                parsed = {}
        proc_map = self._comp_rms_p2p_map_from_proc(comp_ids)
        for cid in comp_ids:
            nk = _norm_cid(cid)
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

        # Optional per-target comp quality from Phase2A (saved as JSON).
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
        comp_id_list = [_norm_cid(r.get("catalog_id", "")) for _, r in sub.iterrows()]
        comp_id_list = [x for x in comp_id_list if x]
        rms_p2p_map = self._comp_rms_p2p_map_for_target(cid, comp_id_list)

        def _fmt(v: Any, nd: int) -> str:
            x = pd.to_numeric(v, errors="coerce")
            return f"{float(x):.{nd}f}" if np.isfinite(x) else "—"

        # Relative weights via apply_comp_w_rel_for_display (excluded → w_rel=0).
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
                ccid = self._norm_cid(r.get("catalog_id", ""))
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
                    stav = f"suspect — {q_note}" if q_note else "suspect"
                else:
                    stav = q_quality
            _cts0 = str(r.get("color_tier_src", "") or "").strip().lower()
            _cts_lbl = (
                "BP-RP"
                if _cts0 == "bprp"
                else ("BV→" if _cts0 == "bv_converted" else ("?" if _cts0 == "unknown" else ("—" if not _cts0 else _cts0[:5])))
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
                _fmt(r.get("delta_bprp_abs"), 3) if "delta_bprp_abs" in sub.columns else "—",
                _cts_lbl,
                _fmt(r.get("dist_deg"), 4),
                str(int(pd.to_numeric(r.get("comp_n_frames"), errors="coerce")))
                if np.isfinite(pd.to_numeric(r.get("comp_n_frames"), errors="coerce"))
                else "—",
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
    def _should_trigger_tess_report(self, bullets: str) -> bool:
        if not bullets or str(bullets).strip() in ("", "—"):
            return True
        VAR_CATS = [
            "VSX",
            "ASAS-SN",
            "ZTF",
            "Gaia",
            "ATLAS",
            "CSS",
            "KELT",
            "VSBS",
            "TESS-EB",
        ]
        for line in str(bullets).split("\n"):
            line0 = str(line).strip()
            for cat in VAR_CATS:
                if (
                    line0.startswith(cat)
                    and "žiadny záznam" not in line0
                    and "no match" not in line0.lower()
                ):
                    return False
        return True
    def _get_candidate_row_pdf(self, var_results_in: Any, cid: str) -> dict[str, Any]:
        """Extract RA, Dec, mag from var_results for the PDF."""
        try:
            import pandas as pd  # noqa: PLC0415

            if var_results_in is None:
                return {"ra": 0.0, "dec": 0.0, "mag": None}
            if isinstance(var_results_in, pd.DataFrame):
                df0 = var_results_in
            elif isinstance(var_results_in, dict):
                df0 = pd.DataFrame(var_results_in)
            else:
                return {"ra": 0.0, "dec": 0.0, "mag": None}
            id_col = next((c for c in df0.columns if "catalog_id" in str(c).lower() or str(c) == "id"), None)
            if id_col is None:
                return {"ra": 0.0, "dec": 0.0, "mag": None}
            sub = df0[df0[id_col].astype(str) == str(cid)]
            if sub.empty:
                return {"ra": 0.0, "dec": 0.0, "mag": None}
            r = sub.iloc[0]
            ra = dec = None
            for rc in ("ra", "RAJ2000", "RA", "ra_deg"):
                if rc in r.index and pd.notna(r[rc]):
                    ra = float(r[rc])
                    break
            for dc in ("dec", "DEJ2000", "DE", "dec_deg"):
                if dc in r.index and pd.notna(r[dc]):
                    dec = float(r[dc])
                    break
            mag = None
            for mc in ("mag", "Vmag", "mag_median"):
                if mc in r.index and pd.notna(r[mc]):
                    mag = float(r[mc])
                    break
            if ra is None or dec is None:
                return {"ra": 0.0, "dec": 0.0, "mag": mag}
            return {"ra": float(ra), "dec": float(dec), "mag": mag}
        except Exception:  # noqa: BLE001
            return {"ra": 0.0, "dec": 0.0, "mag": None}
    def _generate_candidate_lc_png(self, cid: str, photometry_dir_in: Path, cache_dir_in: Path) -> Path | None:
        """Load light-curve CSV for cid and save as PNG."""
        import glob
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        pattern = str(photometry_dir_in / "lightcurves" / f"*{str(cid)[:18]}*.csv")
        files = glob.glob(pattern)
        if not files:
            return None
        try:
            df = pd.read_csv(files[0])
            time_col = next((c for c in df.columns if "bjd" in str(c).lower() or "time" in str(c).lower()), None)
            mag_col = next((c for c in df.columns if "mag" in str(c).lower()), None)
            if not time_col or not mag_col:
                return None
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
            df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce")
            df = df.dropna(subset=[time_col, mag_col])
            if df.empty:
                return None
            t0 = float(df[time_col].min())
            fig, ax = plt.subplots(figsize=(10, 3))
            try:
                fig.patch.set_facecolor("white")
                ax.scatter(df[time_col] - t0, df[mag_col], s=8, alpha=0.7, color="#378ADD")
                ax.set_xlabel("BJD - BJD0")
                ax.set_ylabel("mag_inst")
                ax.set_title(f"Light curve — {str(cid)[:20]}")
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                png_path = cache_dir_in / f"lc_{str(cid)[:18]}.png"
                fig.savefig(png_path, dpi=120, bbox_inches="tight")
            finally:
                plt.close(fig)
            return png_path if png_path.exists() else None
        except Exception as exc:  # noqa: BLE001
            logging.warning("[PDF] _generate_candidate_lc_png failed: %s", exc)
            return None
    def _draw_candidate_detail_page(self, 
        c: "canvas.Canvas",
        *,
        cid: str,
        candidate_row: dict[str, Any],
        bullets: str,
    ) -> None:
        """Detail page for a new variable candidate (reportlab)."""
        c.setPageSize(self.landscape(self.A4))
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 13)
        c.setFillColor(self.colors.black)
        c.drawString(self.M_LEFT, y - 0.6 * self.cm, f"NEW VARIABLE CANDIDATE: {str(cid)[:24]}")
        y -= 1.1 * self.cm
        c.setFont(self.FONT_REG, 9)
        try:
            mag0 = candidate_row.get("mag")
            mag_txt = f"{float(mag0):.3f}" if mag0 is not None and np.isfinite(float(mag0)) else "—"
        except Exception:  # noqa: BLE001
            mag_txt = "—"
        c.drawString(
            self.M_LEFT,
            y,
            f"mag={mag_txt}  RA={float(candidate_row.get('ra', 0.0)):.4f}  Dec={float(candidate_row.get('dec', 0.0)):.4f}",
        )
        y -= 0.6 * self.cm

        c.setFont(self.FONT_BOLD, 10)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y, "Catalog crossmatch:")
        c.setFillColor(self.colors.black)
        y -= 0.5 * self.cm
        c.setFont(self.FONT_REG, 8)
        for ln in (bullets or "—").split("\n")[:12]:
            c.drawString(self.M_LEFT, y, str(ln)[:110])
            y -= 0.35 * self.cm
        y -= 0.2 * self.cm

        lc_png = self._generate_candidate_lc_png(str(cid), self.photometry_dir, self.cache_dir)
        if lc_png is not None:
            c.setFont(self.FONT_BOLD, 10)
            c.setFillColor(self.C_TITLE)
            c.drawString(self.M_LEFT, y, "Light curve (ground-based):")
            c.setFillColor(self.colors.black)
            y -= 0.25 * self.cm
            self._draw_image_fit(c, lc_png, self.M_LEFT, y, self.USE_W, 55.0 * self.mm, image_type="lc")
            y -= 55.0 * self.mm + 0.3 * self.cm

        self._page_footer(c)
        c.showPage()

    def _is_sparse_star_data(self, sd: dict[str, Any]) -> bool:
        lc_rms = float(pd.to_numeric(sd.get("lc_rms"), errors="coerce"))
        lc_path = str(sd.get("lc_img") or "")
        lc_ok = bool(lc_path) and Path(lc_path).is_file()
        comp_ok = bool(sd.get("comp_rows"))
        return (not np.isfinite(lc_rms)) and (not lc_ok) and (not comp_ok)
    def _draw_compact_star_block(self, c: "canvas.Canvas", sd: dict[str, Any], y_top: float, block_h: float) -> None:
        bottom = y_top - block_h
        yy = y_top - 0.35 * self.cm
        vsx_name = str(sd.get("vsx_name", "") or "")
        vsx_type = str(sd.get("vsx_type", "") or "")
        zone_flag = str(sd.get("zone_flag", "") or "")
        c.setFont(self.FONT_BOLD, 11)
        c.setFillColor(self.colors.black)
        if self._use_bprp_primary:
            bp_disp = float(pd.to_numeric(sd.get("bp_rp"), errors="coerce"))
            bpt = f"{bp_disp:.3f}" if np.isfinite(bp_disp) else "—"
            c.drawString(self.M_LEFT, yy, f"{vsx_name}  |  {vsx_type}  |  {zone_flag}  |  BP-RP: {bpt}")
        else:
            c.drawString(self.M_LEFT, yy, f"{vsx_name}  |  {vsx_type}  |  {zone_flag}")
        yy -= 0.55 * self.cm
        lc_rms = float(pd.to_numeric(sd.get("lc_rms_ooe", sd.get("lc_rms")), errors="coerce"))
        good_comp = int(
            pd.to_numeric(sd.get("n_stability_good", sd.get("good_comp")), errors="coerce") or 0
        )
        ap_px = float(pd.to_numeric(sd.get("aperture_px"), errors="coerce"))
        rms_txt = f"{lc_rms:.4f}" if np.isfinite(lc_rms) else "—"
        ap_txt = f"{ap_px:.1f}px" if np.isfinite(ap_px) else "—"
        c.setFont(self.FONT_REG, 8)
        c.setFillColor(self.colors.HexColor("#444444"))
        c.drawString(
            self.M_LEFT,
            yy,
            f"lc_rms (OOE): {rms_txt}  |  stable comp: {good_comp:d}  |  aperture: {ap_txt}  |   "
            "Vizier: https://vizier.cds.unistra.fr/viz-bin/Vsx",
        )
        yy -= 0.45 * self.cm
        n_sat = int(pd.to_numeric(sd.get("n_saturated"), errors="coerce") or 0)
        reason = ""
        if n_sat > 0:
            reason = " — saturated / outlier-dominated"
        elif str(sd.get("skip_reason", "") or "").strip():
            reason = f" — {str(sd.get('skip_reason')).strip()}"
        c.setFont(self.FONT_REG, 8)
        c.drawString(self.M_LEFT, yy, f"Light curve not available{reason}.")
        c.setStrokeColor(self.colors.HexColor("#cccccc"))
        c.setLineWidth(0.6)
        c.line(self.M_LEFT, bottom + 0.15 * self.cm, self.PAGE_W - self.M_RIGHT, bottom + 0.15 * self.cm)
    def _report_per_star_compact_page(self, c: "canvas.Canvas", stars: list[dict[str, Any]]) -> None:
        c.setPageSize(self.landscape(self.A4))
        ns = max(1, len(stars))
        gap = 0.35 * self.cm
        avail = self.PAGE_H - self.M_TOP - self.M_BOTTOM - 0.5 * self.cm
        per = min(120.0 * self.mm, (avail - gap * (ns - 1)) / float(ns))
        y_cur = self.PAGE_H - self.M_TOP
        for sd in stars:
            self._draw_compact_star_block(c, sd, y_cur, float(per))
            y_cur -= float(per) + gap
        self._page_footer(c)
        c.showPage()
    def _draw_catalog_crossmatch_block(self, 
        c: "canvas.Canvas", *, cid_key: str, is_cand: bool, y_top: float, max_lines: int | None = None
    ) -> float:
        if not is_cand or cid_key not in self.bullets_by_cid:
            return y_top
        y = y_top
        c.setFont(self.FONT_BOLD, 11)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.35 * self.cm, "Catalog crossmatch")
        c.setFillColor(self.colors.black)
        y -= 0.62 * self.cm
        txt = str(self.bullets_by_cid.get(cid_key, "") or "").strip()
        c.setFont(self.FONT_REG, 8)
        if txt:
            raw_lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            extra = 0
            if max_lines is not None and len(raw_lines) > int(max_lines):
                extra = len(raw_lines) - int(max_lines)
                raw_lines = raw_lines[: int(max_lines)]
            for ln in raw_lines:
                c.drawString(self.M_LEFT, y - 0.26 * self.cm, ln[:200])
                y -= 0.30 * self.cm
            if extra > 0:
                c.setFont(self.FONT_OBL, 7)
                c.setFillColor(self.colors.HexColor("#666666"))
                c.drawString(self.M_LEFT, y - 0.26 * self.cm, f"(+{extra} more lines not shown)")
                c.setFillColor(self.colors.black)
                y -= 0.30 * self.cm
        else:
            c.drawString(
                self.M_LEFT,
                y - 0.28 * self.cm,
                "No catalog match found — potential new variable",
            )
            y -= 0.38 * self.cm
        return y
    def _draw_aperture_correction_block(self, c: "canvas.Canvas", cid_key: str, y_top: float) -> float:
        """Draw compact aperture correction info (from comp_quality_{cid}.json) if space allows."""
        # Need space; do not paginate here (draw_star_page owns showPage()).
        _MIN_H_PT = 50.0
        try:
            if float(y_top) < float(self.M_BOTTOM) + _MIN_H_PT:
                return y_top
        except Exception:  # noqa: BLE001
            return y_top

        qpath = self.lc_dir / f"comp_quality_{cid_key}.json"
        if not qpath.is_file():
            return y_top
        try:
            qraw = json.loads(qpath.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return y_top
        if not isinstance(qraw, dict):
            return y_top
        ac = qraw.get("aperture_correction", None)
        if not isinstance(ac, dict):
            return y_top

        # Top separator line
        y = float(y_top)
        c.setStrokeColor(self.colors.HexColor("#bbbbbb"))
        c.setLineWidth(0.3)
        c.line(self.M_LEFT, y - 2.0, self.PAGE_W - self.M_RIGHT, y - 2.0)
        y -= 12.0

        ac_ok = bool(ac.get("ok", False))
        reason = str(ac.get("reason", "disabled") or "disabled").strip()
        if not ac_ok:
            c.setFont(self.FONT_REG, 7)
            c.setFillColor(self.colors.HexColor("#777777"))
            c.drawString(self.M_LEFT, y, f"Aperture correction: not applied ({reason})")
            c.setFillColor(self.colors.black)
            return y - 14.0

        dm = pd.to_numeric(ac.get("delta_m_corr", None), errors="coerce")
        sc = pd.to_numeric(ac.get("scatter_mag", None), errors="coerce")
        n_ref = int(pd.to_numeric(ac.get("n_ref_stars", 0), errors="coerce") or 0)
        if not np.isfinite(dm):
            c.setFont(self.FONT_REG, 7)
            c.setFillColor(self.colors.HexColor("#777777"))
            c.drawString(self.M_LEFT, y, "Aperture correction: not applied (missing ΔM)")
            c.setFillColor(self.colors.black)
            return y - 14.0

        # Row 1
        c.setFont(self.FONT_BOLD, 8)
        c.setFillColor(self.colors.HexColor("#333333"))
        c.drawString(self.M_LEFT, y, "Aperture Correction (Method B)")
        y -= 12.0

        # Row 2
        c.setFont(self.FONT_REG, 7)
        c.setFillColor(self.colors.HexColor("#444444"))
        sc_txt = f"{float(sc):.4f}" if np.isfinite(sc) else "—"
        c.drawString(
            self.M_LEFT,
            y,
            f"ΔM = {float(dm):+.4f} mag  |  scatter = {sc_txt} mag  |  n_ref = {int(n_ref)}",
        )
        y -= 11.0

        # Row 3
        ref_ids = ac.get("ref_star_ids", []) if isinstance(ac.get("ref_star_ids", []), list) else []
        ref_s = ", ".join(str(x)[:16] for x in ref_ids[:3] if str(x).strip())
        if ref_s:
            c.setFont(self.FONT_REG, 7)
            c.setFillColor(self.colors.HexColor("#888888"))
            c.drawString(self.M_LEFT, y, f"Ref: {ref_s}")
            y -= 11.0

        c.setFont(self.FONT_REG, 6)
        c.setFillColor(self.colors.HexColor("#888888"))
        c.drawString(
            self.M_LEFT,
            y,
            "Method B: weighted mean offset between instrumental magnitude and Gaia catalog "
            "magnitude for n_ref reference stars (comp stars with good Phase-2A stability). "
            "ΔM is applied to all target magnitudes. Scatter = 1σ of ref star residuals.",
        )
        y -= 10.0

        c.setFillColor(self.colors.black)
        return y
    def _report_per_star_page(self, c: "canvas.Canvas", star_data: dict[str, Any]) -> None:
        c.setPageSize(self.landscape(self.A4))

        y_cursor = self.PAGE_H - self.M_TOP
        cid_key = self._norm_cid(str(star_data.get("catalog_id", "") or ""))
        is_cand = cid_key in self._candidates_norm

        vsx_name = str(star_data.get("vsx_name", "") or "")
        vsx_type = str(star_data.get("vsx_type", "") or "")
        zone_flag = str(star_data.get("zone_flag", "") or "")
        bp_rp_val = star_data.get("bp_rp", float("nan"))
        try:
            bp_rp_f = float(bp_rp_val)
        except Exception:  # noqa: BLE001
            bp_rp_f = float("nan")
        bp_rp_txt = f"{bp_rp_f:.3f}" if np.isfinite(bp_rp_f) else "—"

        b_v_val = star_data.get("b_v", float("nan"))
        try:
            b_v_f = float(b_v_val)
        except Exception:  # noqa: BLE001
            b_v_f = float("nan")
        b_v_txt = f"{b_v_f:.2f}" if np.isfinite(b_v_f) else "—"

        vtype_short = vsx_type if len(vsx_type) <= 64 else (vsx_type[:61] + "...")
        if self._use_bprp_primary:
            title = f"{vsx_name}  |  {vtype_short}  |  {zone_flag}  |  BP-RP: {bp_rp_txt}"
        else:
            title = f"{vsx_name}  |  {vtype_short}  |  {zone_flag}  |  BP-RP: {bp_rp_txt}  |  B-V: {b_v_txt}"
        title_style = self._get_para_style("star_title", fontName=self.FONT_BOLD, fontSize=14, leading=16)
        y_cursor = self._draw_paragraph_block(
            c, self.M_LEFT, y_cursor, self.USE_W, self._pdf_escape(title), title_style, paginate=False, gap_pt=0.06 * self.cm
        )

        trust = str(star_data.get("trust", "") or "").strip().upper()
        trust_reason = str(star_data.get("trust_reason", "") or "").strip()

        lc_rms = float(pd.to_numeric(star_data.get("lc_rms_ooe", star_data.get("lc_rms")), errors="coerce"))
        good_comp = int(
            pd.to_numeric(
                star_data.get("n_stability_good", star_data.get("good_comp")),
                errors="coerce",
            )
            or 0
        )
        ap_px = float(pd.to_numeric(star_data.get("aperture_px"), errors="coerce"))
        rms_txt = f"{lc_rms:.4f}" if np.isfinite(lc_rms) else "—"
        ap_txt = f"{ap_px:.1f}px" if np.isfinite(ap_px) else "—"
        metrics = (
            f"lc_rms (OOE): {rms_txt}   |   stable comp: {good_comp:d}   |   aperture: {ap_txt}   |   "
            "Vizier: https://vizier.cds.unistra.fr/viz-bin/Vsx"
        )
        metrics_style = self._get_para_style("star_metrics", fontSize=9, textColor=self.colors.HexColor("#666666"))
        y_cursor = self._draw_paragraph_block(
            c,
            self.M_LEFT,
            y_cursor,
            self.USE_W,
            self._pdf_escape(metrics),
            metrics_style,
            paginate=False,
            gap_pt=0.12 * self.cm,
        )
        n_pt = int(star_data.get("n_points", 0) or 0)
        merr_v = star_data.get("merr_med", float("nan"))
        lqf = str(star_data.get("lc_quality_flag", "") or "").strip()
        merr_txt = f"{float(merr_v):.4f}" if np.isfinite(float(pd.to_numeric(merr_v, errors="coerce"))) else "—"
        extra1 = f"n_points: {n_pt:d}  |  MERR (median): {merr_txt}  |  LC quality: {lqf or '—'}"
        extra_style = self._get_para_style("star_extra", fontSize=8, textColor=self.colors.HexColor("#555555"))
        y_cursor = self._draw_paragraph_block(
            c, self.M_LEFT, y_cursor, self.USE_W, self._pdf_escape(extra1), extra_style, paginate=False, gap_pt=0.08 * self.cm
        )
        if trust:
            reason_short = trust_reason.split(" — ")[0].strip()
            if len(reason_short) > 100:
                reason_short = reason_short[:97] + "..."
            trust_line = f"TRUST: {trust} — {reason_short}" if reason_short else f"TRUST: {trust}"
            trust_color = {
                "GREEN": "#1b5e20",
                "YELLOW": "#e65100",
                "RED": "#b71c1c",
            }.get(trust, "#37474f")
            trust_style = self._get_para_style(
                "star_trust",
                fontName=self.FONT_BOLD,
                fontSize=9,
                textColor=self.colors.HexColor(trust_color),
                leading=11,
            )
            y_cursor = self._draw_paragraph_block(
                c,
                self.M_LEFT,
                y_cursor,
                self.USE_W,
                self._pdf_escape(trust_line),
                trust_style,
                paginate=False,
                gap_pt=0.08 * self.cm,
            )
        chk = star_data.get("check_star") if isinstance(star_data.get("check_star"), dict) else {}
        chk_line = (
            f"Check star: {chk.get('kname', '—')}  |  KMAG: {chk.get('kmag', '—')}  |  scatter: {chk.get('scatter', '—')}"
        )
        var_line = str(star_data.get("variability_line", "") or "")
        y_cursor = self._draw_paragraph_block(
            c, self.M_LEFT, y_cursor, self.USE_W, self._pdf_escape(chk_line), extra_style, paginate=False, gap_pt=0.06 * self.cm
        )
        y_cursor = self._draw_paragraph_block(
            c, self.M_LEFT, y_cursor, self.USE_W, self._pdf_escape(var_line), extra_style, paginate=False, gap_pt=0.12 * self.cm
        )
        c.setFillColor(self.colors.black)

        c.setStrokeColor(self.colors.HexColor("#cccccc"))
        c.setLineWidth(0.8)
        c.line(self.M_LEFT, y_cursor, self.PAGE_W - self.M_RIGHT, y_cursor)
        y_cursor -= self.SEP_H + 0.1 * self.cm

        comp_rows_all = list(star_data.get("comp_rows") or [])
        comp_excluded_note = str(star_data.get("comp_excluded_note", "") or "").strip()
        MAX_COMP_ROWS = 12
        comp_rows = comp_rows_all[:MAX_COMP_ROWS]
        n_comp_hidden = max(0, len(comp_rows_all) - len(comp_rows))
        excluded_note_h = 0.38 * self.cm if comp_excluded_note else 0.0

        cm_extra_h = 0.0
        if is_cand and cid_key in self.bullets_by_cid:
            cm_extra_h = 0.35 * self.cm + 0.62 * self.cm + 6 * 0.30 * self.cm + 0.2 * self.cm
        cm_extra_h += 0.95 * self.cm

        n_comp_rows = int(len(comp_rows))
        ROW_H = 0.55 * self.cm
        HEADER_H = 0.60 * self.cm
        TABLE_MARGIN = 0.25 * self.cm
        NOTE_H = 0.45 * self.cm
        trunc_note_h = 0.38 * self.cm if n_comp_hidden > 0 else 0.0
        table_h_needed = (
            HEADER_H + n_comp_rows * ROW_H + NOTE_H + TABLE_MARGIN + trunc_note_h + excluded_note_h
        )
        AP_RESERVE = 0.85 * self.cm
        BOTTOM_RESERVE = self.M_BOTTOM + 0.75 * self.cm
        graph_cap = 92.0 * self.mm
        y_avail = y_cursor - table_h_needed - AP_RESERVE - BOTTOM_RESERVE - cm_extra_h
        graph_h = float(np.clip(y_avail, 48.0 * self.mm, graph_cap))

        lc_x = self.M_LEFT
        fi_x = self.M_LEFT + self.LC_W + self.GAP_W
        y_top = y_cursor

        lc_img = star_data.get("lc_img")
        fi_img = star_data.get("field_img")
        if lc_img and Path(lc_img).exists():
            self._draw_image_fit(c, Path(lc_img), lc_x, y_top, self.LC_W, graph_h, image_type="lc")
        else:
            c.setFont(self.FONT_REG, 9)
            c.setFillColor(self.colors.HexColor("#333333"))
            c.drawString(lc_x + 0.5 * self.cm, y_top - graph_h * 0.5, "Light curve not available")
            c.setFillColor(self.colors.black)

        if fi_img and Path(fi_img).exists():
            self._draw_image_fit(c, Path(fi_img), fi_x, y_top, self.FI_W, graph_h, image_type="field")

        y_cursor = y_top - graph_h
        y_cursor = self._draw_catalog_crossmatch_block(
            c, cid_key=cid_key, is_cand=is_cand, y_top=y_cursor, max_lines=6
        )

        c.setStrokeColor(self.colors.HexColor("#cccccc"))
        c.setLineWidth(0.8)
        c.line(self.M_LEFT, y_cursor, self.PAGE_W - self.M_RIGHT, y_cursor)
        y_cursor -= self.SEP_H + 0.05 * self.cm

        if comp_rows:
            if self._use_bprp_primary:
                headers = [
                    "#",
                    "catalog_id",
                    "mag",
                    "BP-RP",
                    "dBPRP",
                    "colΔ",
                    "dist_deg",
                    "n_frames",
                    "comp_rms",
                    "rms_p2p",
                    "w (rel)",
                    "tier",
                    "status",
                ]
                base_widths = [16, 98, 32, 32, 32, 26, 32, 30, 30, 30, 26, 16, 30]
                d_bprp_col = 4
                foot = (
                    "status = Phase-2A LC stability; comp_rms = Phase-1 flux scatter; "
                    "rms_p2p = Abbe p2p on inst. mag (matches exclusion footnote); "
                    "tier = Phase-1 colour tier; colΔ = tier colour source; catalog_id = Gaia/AP/TY."
                )
            else:
                headers = [
                    "#",
                    "catalog_id",
                    "mag",
                    "B-V",
                    "B-V src",
                    "BP-RP",
                    "dBPRP",
                    "colΔ",
                    "dist_deg",
                    "n_frames",
                    "comp_rms",
                    "rms_p2p",
                    "w (rel)",
                    "tier",
                    "status",
                ]
                base_widths = [16, 94, 30, 26, 26, 30, 30, 24, 30, 28, 28, 26, 16, 28]
                d_bprp_col = 6
                foot = (
                    "status = Phase-2A LC stability; comp_rms = Phase-1 flux scatter; "
                    "rms_p2p = Abbe p2p on inst. mag (matches exclusion footnote); "
                    "tier = Phase-1 colour tier; B-V informational; catalog_id = Gaia/AP/TY."
                )
            table_data: list[list[Any]] = [headers]
            comp_id_style = self._get_para_style("comp_catalog_id", fontSize=6.5)
            comp_cell_style = self._get_para_style("comp_cell", fontSize=7)
            for row in comp_rows:
                cells: list[Any] = []
                for j, cell in enumerate(row):
                    if j == 1:
                        cells.append(self.Paragraph(self._pdf_id_display(cell), comp_id_style))
                    else:
                        cells.append(self.Paragraph(self._pdf_escape(str(cell)), comp_cell_style))
                table_data.append(cells)
            _scale = float(self.USE_W) / float(sum(base_widths)) if sum(base_widths) > 0 else 1.0
            col_widths = [w * _scale for w in base_widths]
            row_heights: list[float] = [0.55 * self.cm]
            for row_cells in table_data[1:]:
                rh = 0.44 * self.cm
                for j, cell in enumerate(row_cells):
                    if isinstance(cell, self.Paragraph):
                        rh = max(rh, self._para_row_height(cell, col_widths[j], min_h=0.44 * self.cm))
                row_heights.append(rh)

            t = self.Table(table_data, colWidths=col_widths, rowHeights=row_heights)
            style = self.TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), self.C_TITLE),
                    ("TEXTCOLOR", (0, 0), (-1, 0), self.colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), self.FONT_BOLD),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("FONTNAME", (0, 1), (-1, -1), self.FONT_REG),
                    ("FONTSIZE", (0, 1), (-1, -1), 7),
                    ("FONTSIZE", (1, 1), (1, -1), 6.5),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.3, self.colors.HexColor("#cccccc")),
                    ("FONTNAME", (d_bprp_col, 1), (d_bprp_col, -1), self.FONT_BOLD),
                    ("FONTSIZE", (d_bprp_col, 1), (d_bprp_col, -1), 7),
                ]
            )
            for i in range(1, len(table_data)):
                stav = str(table_data[i][-1] or "").strip().lower()
                if stav == "good" or stav.startswith("good"):
                    bg = self.colors.HexColor("#d4edda")
                elif stav.startswith("suspect"):
                    bg = self.colors.HexColor("#fff3cd")
                else:
                    bg = self.colors.HexColor("#f8d7da") if stav else self.colors.white
                style.add("BACKGROUND", (0, i), (-1, i), bg)
            t.setStyle(style)

            _, t_h = t.wrap(self.USE_W, y_cursor - self.M_BOTTOM)
            table_y = y_cursor - TABLE_MARGIN
            t.drawOn(c, self.M_LEFT, table_y - t_h)
            c.setFont(self.FONT_OBL, 5.5)
            c.setFillColor(self.colors.HexColor("#666666"))
            c.drawString(self.M_LEFT, table_y - t_h - 0.35 * self.cm, foot)
            c.setFillColor(self.colors.black)
            y_after_foot = table_y - t_h - 0.45 * self.cm
            if comp_excluded_note:
                note_style = self._get_para_style("comp_excluded", fontSize=7, textColor=self.colors.HexColor("#888888"))
                y_after_foot = self._draw_paragraph_block(
                    c,
                    self.M_LEFT,
                    y_after_foot,
                    self.USE_W,
                    self._pdf_escape(comp_excluded_note),
                    note_style,
                    paginate=False,
                    gap_pt=0.12 * self.cm,
                )
            if n_comp_hidden > 0:
                c.setFont(self.FONT_OBL, 7)
                c.setFillColor(self.colors.HexColor("#555555"))
                c.drawString(
                    self.M_LEFT,
                    y_after_foot - 0.07 * self.cm,
                    f"(+{n_comp_hidden} more comparison stars not shown)",
                )
                c.setFillColor(self.colors.black)
                y_cursor = y_after_foot - 0.27 * self.cm
            else:
                y_cursor = y_after_foot
        else:
            c.setFont(self.FONT_REG, 9)
            c.setFillColor(self.colors.HexColor("#333333"))
            c.drawString(self.M_LEFT, y_cursor - 0.4 * self.cm, "Comparison star table not available")
            c.setFillColor(self.colors.black)

        # Aperture correction block (optional; best-effort, no pagination here)
        y_cursor = self._draw_aperture_correction_block(c, cid_key, y_cursor)

        c.setFont(self.FONT_OBL, 6)
        note_style = self._get_para_style("star_note", fontName=self.FONT_OBL, fontSize=6, textColor=self.colors.HexColor("#666666"))
        note_y_top = self._layout_y_floor() + 0.25 * self.cm
        self._draw_paragraph_block(
            c,
            self.M_LEFT,
            note_y_top,
            self.USE_W,
            self._pdf_escape(self.NOTE_TXT),
            note_style,
            paginate=False,
            gap_pt=0.0,
            check_bounds=False,
        )
        c.setFillColor(self.colors.black)

        self._page_footer(c)
        c.showPage()
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
        id_series = work[id_key].map(_norm_cid)
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
                            self._pdf_escape(f"{float(x):.3f}" if np.isfinite(float(x)) else "—"),
                            cell_style,
                        )
                    )
                elif cn == "lc_rms" or cn == "merr_med":
                    x = pd.to_numeric(r.get(cn), errors="coerce")
                    row_out.append(
                        self.Paragraph(
                            self._pdf_escape(f"{float(x):.4f}" if np.isfinite(float(x)) else "—"),
                            cell_style,
                        )
                    )
                elif cn == "n_points":
                    x = int(pd.to_numeric(r.get(cn), errors="coerce") or 0)
                    row_out.append(self.Paragraph(self._pdf_escape(str(x)), cell_style))
                elif cn == "lc_quality_flag":
                    row_out.append(self.Paragraph(self._pdf_escape(_cell_txt(r.get(cn)) or "—"), cell_style))
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
                    "Summary of all stars (vsx_type order, then lc_rms) — continued",
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

    def _report_psf_summary_section(self, c: "canvas.Canvas") -> None:
        """ePSF summary table (top stars by PSF OK %) — non-fatal if data missing."""
        try:
            from photometry_core import load_epsf_metrics_for_draft
        except Exception as exc:  # noqa: BLE001
            logging.warning("PDF PSF section: import failed (%s)", exc)
            return

        proc_dir = self.draft_dir / "detrended_aligned" / "lights" / self.obs_group
        if not proc_dir.is_dir() or not any(proc_dir.glob("proc_*.csv")):
            proc_glob = sorted(self.draft_dir.glob("detrended_aligned/lights/*/"))
            proc_dir = next(
                (p for p in proc_glob if p.is_dir() and any(p.glob("proc_*.csv"))),
                proc_dir,
            )
        if not proc_dir.is_dir():
            return

        at_for_merge = self.at_df if isinstance(self.at_df, pd.DataFrame) else pd.DataFrame()
        epsf_df = load_epsf_metrics_for_draft(proc_dir, at_for_merge)
        if epsf_df.empty:
            return

        top = epsf_df.head(20).copy()
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 14)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.5 * self.cm, "ePSF photometry summary (top 20 by PSF OK %)")
        c.setFillColor(self.colors.black)
        y -= 1.0 * self.cm

        chi_thr = float(getattr(self._cfg, "psf_chi2_threshold", 50.0) or 50.0) if self._cfg else 50.0
        c.setFont(self.FONT_REG, 9)
        c.drawString(
            self.M_LEFT,
            y - 0.35 * self.cm,
            f"χ² acceptance threshold: {chi_thr:.0f}  |  stars in table: {len(epsf_df)}",
        )
        y -= 0.75 * self.cm

        id_col = "vsx_name" if "vsx_name" in top.columns else "catalog_id"
        hdr = ["Star", "PSF OK", "Frames", "PSF %", "mean χ²", "min χ²"]
        data: list[list[str]] = [hdr]
        for _, r in top.iterrows():
            n_ok = int(pd.to_numeric(r.get("n_psf_ok"), errors="coerce") or 0)
            n_fr = int(pd.to_numeric(r.get("n_frames"), errors="coerce") or 0)
            pct = pd.to_numeric(r.get("pct_psf_ok"), errors="coerce")
            mchi = pd.to_numeric(r.get("mean_chi2"), errors="coerce")
            mnchi = pd.to_numeric(r.get("min_chi2"), errors="coerce")
            data.append(
                [
                    self.Paragraph(
                        self._pdf_id_display(str(r.get(id_col, r.get("catalog_id", "")) or "").strip()),
                        self._get_para_style("psf_star", fontSize=7),
                    ),
                    f"{n_ok}/{n_fr}",
                    str(n_fr),
                    f"{float(pct):.1f}" if pd.notna(pct) else "—",
                    f"{float(mchi):.1f}" if pd.notna(mchi) else "—",
                    f"{float(mnchi):.1f}" if pd.notna(mnchi) else "—",
                ]
            )

        col_widths = [
            self.USE_W * 0.28,
            self.USE_W * 0.12,
            self.USE_W * 0.10,
            self.USE_W * 0.12,
            self.USE_W * 0.14,
            self.USE_W * 0.14,
        ]
        row_h = 0.50 * self.cm
        psf_row_heights: list[float] = [0.55 * self.cm]
        for row in data[1:]:
            rh = row_h
            cell0 = row[0]
            if isinstance(cell0, self.Paragraph):
                rh = max(rh, self._para_row_height(cell0, col_widths[0], min_h=row_h))
            psf_row_heights.append(rh)
        t = self.Table(data, colWidths=col_widths, rowHeights=psf_row_heights)
        t.setStyle(
            self.TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), self.C_TITLE),
                    ("TEXTCOLOR", (0, 0), (-1, 0), self.colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), self.FONT_BOLD),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("FONTNAME", (0, 1), (-1, -1), self.FONT_REG),
                    ("FONTSIZE", (0, 1), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.25, self.colors.HexColor("#cccccc")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        t.wrapOn(c, self.USE_W, self.USE_H)
        table_bottom = self.M_BOTTOM + 1.2 * self.cm
        t.drawOn(c, self.M_LEFT, table_bottom)
        self._page_footer(c)
        c.showPage()

    def _report_hrd_page(self, c: "canvas.Canvas") -> None:
        """HRD + highlighted stars table (best effort)."""
        try:
            from config import AppConfig

            from hrd_analysis import (
                build_hrd_dataframe,
                get_top_interesting_stars,
                plot_hrd_matplotlib,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("PDF HRD: import failed (%s)", exc)
            return

        gdb = Path(str(getattr(AppConfig(), "gaia_db_path", "") or "").strip())
        ms_csv = self.platesolve_dir / "masterstars_full_match.csv"
        if not ms_csv.is_file():
            logging.info("PDF HRD: skip — missing %s", ms_csv.name)
            return
        if not gdb.is_file():
            logging.info("PDF HRD: skip — gaia_db_path not configured or file missing")
            return

        hrd_png = self.cache_dir / "hrd_field_summary.png"
        top = pd.DataFrame()
        hrd_df = pd.DataFrame()
        try:
            hrd_df = build_hrd_dataframe(ms_csv, gdb)
            if hrd_df.empty:
                return
            top = get_top_interesting_stars(hrd_df)
            plot_hrd_matplotlib(hrd_df, top, output_path=hrd_png)
        except Exception as exc:  # noqa: BLE001
            logging.warning("PDF HRD: build/plot failed (%s)", exc)
            return

        c.showPage()
        y0 = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 14)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y0 - 0.5 * self.cm, "Field astrophysics (Hertzsprung–Russell diagram)")
        c.setFillColor(self.colors.black)
        y = y0 - 1.1 * self.cm

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
            y -= dh + 0.35 * self.cm

        if top is not None and not top.empty:
            tab_cols = [
                c for c in ("category", "catalog_id", "mag_g", "abs_mag_g", "bp_rp", "teff", "logg") if c in top.columns
            ]
            if tab_cols:
                rows_t: list[list[str]] = [tab_cols]
                for _, rr in top.iterrows():
                    rows_t.append([str(rr.get(c, "") or "") for c in tab_cols])
                base_w = [3.0 * self.cm, 5.0 * self.cm, 2.0 * self.cm, 2.0 * self.cm, 2.0 * self.cm, 2.2 * self.cm, 2.0 * self.cm]
                col_ws = base_w[: len(tab_cols)]
                rt = self.Table(rows_t, colWidths=col_ws, repeatRows=1)
                rt.setStyle(
                    self.TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), self.C_TITLE),
                            ("TEXTCOLOR", (0, 0), (-1, 0), self.colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), self.FONT_BOLD),
                            ("FONTSIZE", (0, 0), (-1, 0), 7),
                            ("FONTNAME", (0, 1), (-1, -1), self.FONT_REG),
                            ("FONTSIZE", (0, 1), (-1, -1), 6.5),
                            ("GRID", (0, 0), (-1, -1), 0.2, self.colors.HexColor("#cccccc")),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    )
                )
                _, th = rt.wrap(self.USE_W, y - self.M_BOTTOM)
                rt.drawOn(c, self.M_LEFT, y - th)
                y -= th + 0.3 * self.cm

        self._page_footer(c)
        c.showPage()
    def _report_field_map(self, c: "canvas.Canvas") -> None:
        fmp: Path | None = None
        for cand in (self.platesolve_dir / "field_map.png", self.photometry_dir / "field_map.png"):
            if cand.is_file():
                fmp = cand
                break
        if fmp is None:
            logging.warning(
                "PDF field map: field_map.png not found under %s or %s — skipping page",
                self.platesolve_dir,
                self.photometry_dir,
            )
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
    def _find_hockey_stick_disk_png(self, ) -> Path | None:
        for pat in ("hockey_stick*.png", "rms_hockey*.png"):
            hits = sorted(self.photometry_dir.glob(pat))
            for p in hits:
                if p.is_file():
                    return p
        hp = self.cache_dir / "hockey_stick.png"
        return hp if hp.is_file() else None
    def _report_hockey_stick(self, c: "canvas.Canvas") -> None:
        """Landscape page; restore default landscape for following pages."""
        c.setPageSize(self.landscape(self.A4))
        y = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 14)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y - 0.45 * self.cm, "Variability Analysis \u2014 RMS Hockey Stick")
        c.setFillColor(self.colors.black)
        y -= 1.05 * self.cm
        gen = self._draw_hockey_stick_png(
            photometry_dir_hs=self.photometry_dir,
            platesolve_dir_hs=self.platesolve_dir,
            cache_dir_hs=self.cache_dir,
        )
        p = Path(gen) if gen is not None and Path(gen).is_file() else None
        if p is None:
            p2 = self._find_hockey_stick_disk_png()
            p = Path(p2) if p2 is not None and Path(p2).is_file() else None
        box_w = self.USE_W * 0.9
        box_h = (self.PAGE_H - self.M_TOP - self.M_BOTTOM - 1.5 * self.cm) * 0.8
        if p is not None and p.is_file():
            try:
                hmw, hjq = self._IMAGE_PDF_SETTINGS["hockey"]
                hb, _hk = self._compress_image_for_pdf(p, hmw, hjq)
                ir = self.ImageReader(hb)
            except Exception:  # noqa: BLE001
                ir = self.ImageReader(str(p))
            iw, ih = ir.getSize()
            if iw and ih:
                sc = min(box_w / float(iw), box_h / float(ih))
                dw, dh = float(iw) * sc, float(ih) * sc
                x0 = self.M_LEFT + (self.USE_W - dw) / 2.0
                y_top = y
                c.drawImage(ir, x0, y_top - dh, width=dw, height=dh, mask="auto")
        else:
            c.setFont(self.FONT_REG, 11)
            c.drawString(self.M_LEFT, y - 0.5 * self.cm, "Hockey stick diagram not available")
        self._page_footer(c)
        c.showPage()
        c.setPageSize(self.landscape(self.A4))
    def _col_pick(self, df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None
    def _report_candidates_table(self, c: "canvas.Canvas") -> None:
        vpaths = [self.photometry_dir / "variability_candidates.csv", self.platesolve_dir / "variability_candidates.csv"]
        vp = next((p for p in vpaths if p.is_file()), None)

        def _empty_candidates_page(title: str, msg: str) -> None:
            y0 = self.PAGE_H - self.M_TOP
            c.setFont(self.FONT_BOLD, 14)
            c.setFillColor(self.C_TITLE)
            c.drawString(self.M_LEFT, y0 - 0.5 * self.cm, title)
            c.setFillColor(self.colors.black)
            c.setFont(self.FONT_REG, 11)
            c.drawString(self.M_LEFT, y0 - 1.05 * self.cm, msg)
            self._page_footer(c)
            c.showPage()

        if vp is None:
            _empty_candidates_page("Variability Candidates", "No variability candidates detected")
            return
        try:
            vdf = pd.read_csv(vp, low_memory=False, dtype=_GAIA_ID_DTYPE)
        except Exception:  # noqa: BLE001
            _empty_candidates_page("Variability Candidates", "No variability candidates detected")
            return
        if vdf.empty:
            _empty_candidates_page("Variability Candidates", "No variability candidates detected")
            return

        def _short(val: Any, n: int) -> str:
            s = str(val if val is not None else "").strip()
            return s if len(s) <= n else s[: n - 1] + "\u2026"

        idc = self._col_pick(vdf, ("catalog_id", "Catalog_ID", "gaia_id"))
        if not idc:
            _empty_candidates_page("Variability Candidates", "No variability candidates detected")
            return

        mag_c = self._col_pick(vdf, ("mag", "lc_median_mag"))
        bprp_c = self._col_pick(vdf, ("bp_rp", "BP_RP"))
        rms_c = self._col_pick(vdf, ("rms_pct", "rms_pct_lc", "RMS_PCT"))
        dm_c = self._col_pick(vdf, ("detection_method", "DETECTION_METHOD"))
        vs_c = self._col_pick(vdf, ("variability_score", "VARIABILITY_SCORE"))
        kat_c = self._col_pick(vdf, ("katalogy", "katalógy", "katalogy", "catalog_match"))
        zn_c = self._col_pick(vdf, ("zone", "zone_flag", "ZONE"))

        # catalog_id ~0.28 self.USE_W; katalogy column width drives Paragraph.wrap row heights.
        frac = [0.28, 0.052, 0.052, 0.062, 0.088, 0.068, 0.323, 0.075]
        col_widths = [self.USE_W * f for f in frac]
        kat_col_w = float(col_widths[6])

        kat_style = self.ParagraphStyle(
            name="varkat",
            fontName=self.FONT_REG,
            fontSize=7,
            leading=10,
            alignment=self.TA_LEFT,
        )

        def _katalogy_paragraph_source_lines(raw: Any) -> list[str]:
            out: list[str] = []
            for raw_line in str(raw or "").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("🔭"):
                    continue
                if "žiadny záznam" in line:
                    continue
                if "no match" in line.lower():
                    continue
                out.append(self._sanitize_katalogy_pdf_line(line))
            return [x for x in out if x and x != "—"]

        def _kat_cell(raw: Any) -> Paragraph:
            pos = _katalogy_paragraph_source_lines(raw)
            if not pos:
                return self.Paragraph("—", kat_style)
            lines = [escape(str(x)[:500]) for x in pos[:4]]
            extra = len(pos) - 4
            if extra > 0:
                lines.append(f"(+{extra} more)")
            return self.Paragraph("<br/>".join(lines), kat_style)

        def _kat_row_h_pts(raw: Any) -> float:
            para = _kat_cell(raw)
            try:
                _pw, ph = para.wrap(max(12.0, kat_col_w - 6.0), 9999.0)
            except Exception:  # noqa: BLE001
                ph = float(3 * kat_style.leading)
            return float(max(3.0 * float(kat_style.leading), ph + 4.0))

        hdr = [
            "catalog_id",
            "mag",
            "bp_rp",
            "rms_pct",
            "detection_method",
            "variability_score",
            "katalogy",
            "zone",
        ]
        data_rows: list[list[Any]] = []
        pos_flags: list[bool] = []
        row_h_pts: list[float] = []
        for _, rr in vdf.iterrows():
            mag_v = pd.to_numeric(rr.get(mag_c), errors="coerce") if mag_c else float("nan")
            bp_v = pd.to_numeric(rr.get(bprp_c), errors="coerce") if bprp_c else float("nan")
            rms_v = pd.to_numeric(rr.get(rms_c), errors="coerce") if rms_c else float("nan")
            vs_v = pd.to_numeric(rr.get(vs_c), errors="coerce") if vs_c else float("nan")
            raw_kat = rr.get(kat_c, "") if kat_c else ""
            pos_flags.append(self._katalogy_row_has_positive(raw_kat))
            row_h_pts.append(_kat_row_h_pts(raw_kat))
            cid_full = self._norm_cid(rr.get(idc)) if idc else ""
            data_rows.append(
                [
                    cid_full if cid_full else "—",
                    f"{float(mag_v):.3f}" if mag_c and np.isfinite(mag_v) else "—",
                    f"{float(bp_v):.3f}" if bprp_c and np.isfinite(bp_v) else "—",
                    f"{float(rms_v):.2f}" if rms_c and np.isfinite(rms_v) else "—",
                    _short(rr.get(dm_c), 9) if dm_c else "—",
                    f"{float(vs_v):.4f}" if vs_c and np.isfinite(vs_v) else "—",
                    _kat_cell(raw_kat) if kat_c else self.Paragraph("—", kat_style),
                    _short(rr.get(zn_c), 8) if zn_c else "—",
                ]
            )

        hdr_h = 0.58 * self.cm
        table_bottom_y = self.M_BOTTOM + 1.0 * self.cm
        title_gap = 1.05 * self.cm
        avail_h = float(self.PAGE_H - self.M_TOP - title_gap - table_bottom_y)

        page_idx = 0
        i0 = 0
        while i0 < len(data_rows):
            cum = float(hdr_h)
            ntake = 0
            while i0 + ntake < len(data_rows):
                rh = row_h_pts[i0 + ntake]
                if cum + rh > avail_h:
                    if ntake == 0:
                        ntake = 1
                    break
                cum += rh
                ntake += 1
            chunk = data_rows[i0 : i0 + ntake]
            chunk_pos = pos_flags[i0 : i0 + ntake]
            chunk_heights = row_h_pts[i0 : i0 + ntake]
            title = "Variability Candidates" if page_idx == 0 else "Variability Candidates (continued)"
            y_top = self.PAGE_H - self.M_TOP
            c.setFont(self.FONT_BOLD, 14)
            c.setFillColor(self.C_TITLE)
            c.drawString(self.M_LEFT, y_top - 0.5 * self.cm, title)
            c.setFillColor(self.colors.black)
            y_tab = y_top - title_gap
            tbl = [hdr] + chunk
            row_heights = [hdr_h] + chunk_heights
            t = self.Table(tbl, colWidths=col_widths, rowHeights=row_heights)
            sty = self.TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), self.C_TITLE),
                    ("TEXTCOLOR", (0, 0), (-1, 0), self.colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), self.FONT_BOLD),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("FONTNAME", (0, 1), (-1, -1), self.FONT_REG),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("FONTSIZE", (6, 1), (6, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.25, self.colors.HexColor("#cccccc")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("VALIGN", (6, 1), (6, -1), "TOP"),
                    ("LEFTPADDING", (6, 1), (6, -1), 3),
                    ("RIGHTPADDING", (6, 1), (6, -1), 3),
                ]
            )
            g1 = self.colors.HexColor("#d1fae5")
            g2 = self.colors.HexColor("#ecfdf5")
            r1 = self.colors.HexColor("#fee2e2")
            r2 = self.colors.HexColor("#fef2f2")
            for r in range(1, len(tbl)):
                pos = chunk_pos[r - 1]
                alt = ((i0 + r - 1) % 2 == 0)
                bg = (g1 if alt else g2) if pos else (r1 if alt else r2)
                sty.add("BACKGROUND", (0, r), (-1, r), bg)
            t.setStyle(sty)
            tw, th = t.wrap(self.USE_W, y_tab - table_bottom_y)
            t.drawOn(c, self.M_LEFT, y_tab - th)
            self._page_footer(c)
            c.showPage()
            i0 += ntake
            page_idx += 1
    def _report_tess_section(self, c: "canvas.Canvas") -> None:
        tess_root = self.photometry_dir / "_tess"
        if not tess_root.is_dir():
            return

        def _vsx_display_for_cid(cid_raw: str) -> str:
            nk = self._norm_cid(str(cid_raw))
            if "_cid" in self.summary_df.columns and "vsx_name" in self.summary_df.columns:
                m = self.summary_df["_cid"].astype(str).eq(nk)
                if bool(m.any()):
                    vn = str(self.summary_df.loc[m, "vsx_name"].iloc[0] or "").strip()
                    if vn and vn.lower() not in ("nan", "none"):
                        return vn
            return str(cid_raw)

        def _sector_sort_key(sec: dict[str, Any]) -> float:
            sn = sec.get("sector")
            try:
                return float(sn)
            except (TypeError, ValueError):
                return float("inf")

        def _rel_color(rel: str) -> Any:
            r = str(rel or "").strip().lower()
            if r == "reliable":
                return self.colors.HexColor("#16a34a")
            if r == "uncertain":
                return self.colors.HexColor("#f59e0b")
            if r == "noise":
                return self.colors.HexColor("#ef4444")
            return self.colors.HexColor("#555555")

        def _fmt_metric(v: Any) -> str:
            x = pd.to_numeric(v, errors="coerce")
            return f"{float(x):.4g}" if np.isfinite(x) else "—"

        cand_dirs = sorted([p for p in tess_root.iterdir() if p.is_dir() and not p.name.startswith(".")])
        if not cand_dirs:
            return

        c.setPageSize(self.landscape(self.A4))
        gap_x = 0.35 * self.cm
        phased_h = (160.0 / 96.0) * 25.4 * self.mm
        blend_h = (180.0 / 96.0) * 25.4 * self.mm
        gap_phased_blend = 4.0 * self.mm

        for tdir in cand_dirs:
            cid = tdir.name
            jpath = tdir / "result.json"
            jd: dict[str, Any] | None = None
            if jpath.is_file():
                try:
                    raw = json.loads(jpath.read_text(encoding="utf-8"))
                    jd = raw if isinstance(raw, dict) else None
                except Exception:  # noqa: BLE001
                    jd = None

            y = self.PAGE_H - self.M_TOP
            disp = _vsx_display_for_cid(cid)
            c.setFont(self.FONT_BOLD, 14)
            c.setFillColor(self.C_TITLE)
            c.drawString(self.M_LEFT, y - 0.45 * self.cm, f"TESS Analysis — {disp}")
            c.setFillColor(self.colors.black)
            y -= 0.95 * self.cm

            if jd is None:
                c.setFont(self.FONT_REG, 10)
                c.drawString(self.M_LEFT, y, f"Catalog ID: {cid}")
                y -= 0.55 * self.cm
                c.setFont(self.FONT_REG, 10)
                c.setFillColor(self.colors.HexColor("#b45309"))
                c.drawString(self.M_LEFT, y, "TESS result data not available")
                c.setFillColor(self.colors.black)
                self._page_footer(c)
                c.showPage()
                continue

            pc = float(pd.to_numeric(jd.get("period_consensus"), errors="coerce"))
            p2c = float(pd.to_numeric(jd.get("period_2p_consensus"), errors="coerce"))
            ptxt = f"{pc:.6f}" if np.isfinite(pc) else "—"
            p2txt = f"{p2c:.6f}" if np.isfinite(p2c) else "—"
            rel = str(jd.get("period_reliability", "") or "").strip() or "—"
            nsec = int(pd.to_numeric(jd.get("total_sectors_found"), errors="coerce") or 0)

            cid_disp = str(jd.get("catalog_id", cid))
            rel_color = _rel_color(rel)
            try:
                rel_hex = str(rel_color.hexval()).replace("0x", "#")
            except Exception:  # noqa: BLE001
                rel_hex = "#555555"
            header_html = (
                f"Catalog ID: {self._pdf_id_display(cid_disp)} | "
                f"Period: {self._pdf_escape(ptxt)} d | 2P: {self._pdf_escape(p2txt)} d | "
                f'Reliability: <font color="{rel_hex}">{self._pdf_escape(rel)}</font>'
            )
            hdr_style = self._get_para_style("tess_hdr", fontSize=9)
            y = self._draw_paragraph_block(c, self.M_LEFT, y, self.USE_W, header_html, hdr_style, paginate=True)
            y -= 0.12 * self.cm
            y = self._draw_flow_lines(
                c,
                y,
                [(f"Sectors found: {nsec}", self.FONT_REG, 9.0)],
                paginate=True,
            )
            y -= 0.12 * self.cm

            sectors_raw = jd.get("sectors") or []
            sectors = [s for s in sectors_raw if isinstance(s, dict)]
            sectors.sort(key=_sector_sort_key)
            amp = snr = fstd = None
            for s in sectors:
                if amp is None and s.get("amplitude_ppt") is not None:
                    amp = s.get("amplitude_ppt")
                if snr is None and s.get("snr") is not None:
                    snr = s.get("snr")
                if fstd is None and s.get("flux_std") is not None:
                    fstd = s.get("flux_std")
            c.setFont(self.FONT_REG, 8)
            c.setFillColor(self.colors.HexColor("#333333"))
            c.drawString(self.M_LEFT, y, f"Amplitude (ppt): {_fmt_metric(amp)}")
            c.drawString(self.M_LEFT + 5.5 * self.cm, y, f"SNR: {_fmt_metric(snr)}")
            c.drawString(self.M_LEFT + 9.5 * self.cm, y, f"Flux std: {_fmt_metric(fstd)}")
            c.setFillColor(self.colors.black)
            y -= 0.55 * self.cm

            def _fmt_period_cell(v: Any) -> str:
                x = pd.to_numeric(v, errors="coerce")
                return f"{float(x):.6f}" if np.isfinite(x) and float(x) > 0.0 else "—"

            def _tess_blend_tail_h(sec: dict[str, Any]) -> float:
                blend_raw = sec.get("blend_check_path")
                bpath = Path(str(blend_raw or "")) if blend_raw else None
                if bpath is not None and bpath.is_file():
                    return float(blend_h + 0.25 * self.cm)
                return float(0.45 * self.cm)

            def _sector_block_h(sec: dict[str, Any]) -> float:
                section_title_h = 0.42 * self.cm
                small_gap = 0.02 * self.cm
                return section_title_h + phased_h + gap_phased_blend + _tess_blend_tail_h(sec) + small_gap

            def _tess_period_analysis_table(jd_in: dict[str, Any]) -> tuple[Any | None, float]:
                secs_t = [s for s in (jd_in.get("sectors") or []) if isinstance(s, dict)]
                if not secs_t:
                    return None, 0.0
                rows_t: list[list[str]] = [
                    ["Sector", "N pts", "P (d)", "Method", "P_anova (d)", "P_consensus (d)"]
                ]
                for s in sorted(secs_t, key=_sector_sort_key):
                    sn = s.get("sector")
                    try:
                        sec_lab = str(int(float(sn)))
                    except (TypeError, ValueError):
                        sec_lab = str(sn) if sn is not None else "?"
                    n_pt = int(pd.to_numeric(s.get("n_points"), errors="coerce") or 0)
                    rows_t.append(
                        [
                            sec_lab,
                            str(n_pt) if n_pt else "—",
                            _fmt_period_cell(s.get("period_ls")),
                            str(s.get("period_method_used") or "").strip() or "—",
                            _fmt_period_cell(s.get("period_anova")),
                            _fmt_period_cell(s.get("period_consensus")),
                        ]
                    )
                if len(rows_t) <= 1:
                    return None, 0.0
                col_tw = [self.USE_W * f for f in (0.08, 0.09, 0.14, 0.26, 0.14, 0.18)]
                rt = self.Table(rows_t, colWidths=col_tw, repeatRows=1)
                st = self.TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), self.C_TITLE),
                        ("TEXTCOLOR", (0, 0), (-1, 0), self.colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), self.FONT_BOLD),
                        ("FONTSIZE", (0, 0), (-1, 0), 7.5),
                        ("FONTNAME", (0, 1), (-1, -1), self.FONT_REG),
                        ("FONTSIZE", (0, 1), (-1, -1), 7.5),
                        ("GRID", (0, 0), (-1, -1), 0.3, self.colors.HexColor("#cccccc")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ]
                )
                rt.setStyle(st)
                _tw, th = rt.wrap(self.USE_W, 9999.0)
                return rt, float(th)

            period_tbl, period_tbl_h = _tess_period_analysis_table(jd)
            if period_tbl is not None and period_tbl_h > 0.0:
                c.setFont(self.FONT_BOLD, 9)
                c.drawString(self.M_LEFT, y - 0.12 * self.cm, "Period analysis")
                y -= 0.38 * self.cm
                period_tbl.drawOn(c, self.M_LEFT, y - period_tbl_h)
                y -= period_tbl_h + 0.35 * self.cm

            if not sectors:
                c.setFont(self.FONT_REG, 9)
                c.drawString(self.M_LEFT, y, "No sector entries in TESS result.")
                self._page_footer(c)
                c.showPage()
                continue

            half_w = (self.USE_W - gap_x) / 2.0
            i_sec = 0
            page_sub = 0
            while i_sec < len(sectors):
                if page_sub > 0:
                    y = self.PAGE_H - self.M_TOP
                    c.setFont(self.FONT_BOLD, 12)
                    c.setFillColor(self.C_TITLE)
                    c.drawString(self.M_LEFT, y - 0.4 * self.cm, f"TESS Analysis — {disp} (continued)")
                    c.setFillColor(self.colors.black)
                    y -= 0.85 * self.cm

                available_h = float(y - self.M_BOTTOM)
                h0 = _sector_block_h(sectors[i_sec])
                h1 = _sector_block_h(sectors[i_sec + 1]) if (i_sec + 1) < len(sectors) else None

                if h1 is not None and (h0 + h1) <= available_h:
                    n_pack = 2
                elif h0 <= available_h:
                    n_pack = 1
                else:
                    if page_sub == 0:
                        self._page_footer(c)
                        c.showPage()
                        page_sub += 1
                        continue
                    n_pack = 1

                for j in range(n_pack):
                    sec = sectors[i_sec + j]
                    sn = sec.get("sector")
                    try:
                        sec_lab = str(int(float(sn)))
                    except (TypeError, ValueError):
                        sec_lab = str(sn) if sn is not None else "?"
                    c.setFont(self.FONT_BOLD, 10)
                    c.setFillColor(self.C_TITLE)
                    c.drawString(self.M_LEFT, y - 0.32 * self.cm, f"Sector {sec_lab}")
                    c.setFillColor(self.colors.black)
                    y -= 0.42 * self.cm
                    pp = sec.get("plot_phased_p_path")
                    p2p = sec.get("plot_phased_2p_path")
                    p_path = Path(str(pp)) if pp else None
                    p2_path = Path(str(p2p)) if p2p else None
                    if p_path is not None and p_path.is_file():
                        self._draw_image_fit(c, p_path, self.M_LEFT, y, half_w, phased_h, image_type="tess_phased")
                    if p2_path is not None and p2_path.is_file():
                        self._draw_image_fit(
                            c, p2_path, self.M_LEFT + half_w + gap_x, y, half_w, phased_h, image_type="tess_phased"
                        )
                    y -= phased_h + gap_phased_blend
                    blend_raw = sec.get("blend_check_path")
                    bpath = Path(str(blend_raw or "")) if blend_raw else None
                    if bpath is not None and bpath.is_file():
                        self._draw_image_fit(c, bpath, self.M_LEFT, y, self.USE_W, blend_h, image_type="tess_blend")
                        y -= blend_h + 0.25 * self.cm
                    else:
                        c.setFont(self.FONT_REG, 8)
                        c.setFillColor(self.colors.HexColor("#666666"))
                        c.drawString(self.M_LEFT, y - 0.2 * self.cm, "Blend check image not available for this sector.")
                        c.setFillColor(self.colors.black)
                        y -= 0.45 * self.cm

                i_sec += n_pack
                self._page_footer(c)
                c.showPage()
                page_sub += 1
    def _report_abbreviations(self, c: "canvas.Canvas") -> None:
        c.setPageSize(self.landscape(self.A4))
        y0 = self.PAGE_H - self.M_TOP
        c.setFont(self.FONT_BOLD, 16)
        c.setFillColor(self.C_TITLE)
        c.drawString(self.M_LEFT, y0 - 0.45 * self.cm, "Abbreviations & Notes")
        c.setFillColor(self.colors.black)
        pairs = [
            ("AAVSO", "American Association of Variable Star Observers"),
            ("BJD", "Barycentric Julian Date"),
            ("BP-RP", "Gaia colour index (Blue Photometer minus Red Photometer)"),
            ("comp", "Comparison star"),
            ("DAO", "DAOPHOT star detection algorithm (Stetson 1987)"),
            ("dBPRP", "Colour difference between comparison star and target (BP-RP)"),
            ("dr", "Angular distance from catalog position (arcseconds)"),
            ("EA", "Algol-type eclipsing binary"),
            ("EB", "Beta Lyrae-type eclipsing binary"),
            ("EW", "W UMa-type eclipsing binary (contact)"),
            ("ELL", "Ellipsoidal variable"),
            ("lc", "Light curve"),
            ("lc_rms", "Light curve RMS scatter (magnitudes)"),
            ("lc_median_mag", "Median calibrated magnitude"),
            ("n_good_comp", "Number of comparison stars used in final ensemble"),
            ("self.obs_group", "Observation group (filter + exposure combination)"),
            ("comp_rms", "Phase-1 detrended relative-flux scatter (comparison star stability)"),
            ("rms_p2p", "Abbe point-to-point RMS on comparison-star inst. magnitudes (Phase-2A)"),
            ("period_reliability", "reliable / uncertain / noise — quality flag for TESS period"),
            ("RMS candidate", "Star with RMS significantly above the noise envelope"),
            ("ROT", "Rotational variable"),
            ("SNR", "Signal-to-noise ratio"),
            ("TESS", "Transiting Exoplanet Survey Satellite (NASA) — used for period analysis"),
            ("tier", "Comparison star colour tier (1=closest BP-RP match, 4=most distant)"),
            ("VAR", "Variable (type unconfirmed)"),
            ("VAR.ASTRO", "Slovak variable star reporting format"),
            ("VDI", "Variability Detection Index (phase-space metric)"),
            ("VDI candidate", "Star with high VDI score (non-random phase distribution)"),
            ("w(rel)", "Relative weight of comparison star in ensemble"),
            ("ZP", "Zero point (photometric calibration offset)"),
            ("zone_flag", "linear = DAO-detected; catalog_only = position from VSX catalog only"),
            (
                "Aperture correction Method B",
                "Ensemble-based aperture correction using stable reference stars; ΔM = offset applied, "
                "scatter = stability of correction",
            ),
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

    def build_pdf(self) -> Path:

        self._overflow_violations.clear()

        # Build PDF
        c = self.canvas.Canvas(str(self.output_pdf), pagesize=self.landscape(self.A4))

        # 1–2) Cover + observation summary
        self._report_cover_page(c)
        self._report_observation_summary(c)

        # 3) FITS Quality Assessment (optional)
        self._report_fits_qa(c)

        # 4) Summary table + optional ePSF summary + 5) HRD
        self._report_summary_table(c)
        try:
            if self._cfg and bool(getattr(self._cfg, "psf_photometry_enabled", False)):
                _epsf_fits = next(
                    iter(self.draft_dir.glob("platesolve/*/masterstar_epsf.fits")),
                    None,
                )
                if _epsf_fits is not None and Path(_epsf_fits).is_file():
                    self._report_psf_summary_section(c)
        except Exception as exc:  # noqa: BLE001
            logging.warning("PDF PSF section skipped (non-fatal): %s", exc)
        self._report_hrd_page(c)

        # 6) Field map (full landscape page)
        self._report_field_map(c)

        # 7+) Per-star pages
        sparse_buf: list[dict[str, Any]] = []

        for _, row in self.summary_df.iterrows():
            try:
                cid = str(row.get("_cid", "") or self._norm_cid(row.get("catalog_id", "")))
                vsx_name = str(row.get("vsx_name", cid) or cid)
                _zf = str(row.get("zone_flag", row.get("zone", "")) or "").strip().lower()
                if _zf == "catalog_only":
                    logging.debug("[PDF] Skip LC page — catalog_only: %s", vsx_name)
                    continue
                _lc_rms = pd.to_numeric(row.get("lc_rms"), errors="coerce")
                _n_frames = int(pd.to_numeric(row.get("n_frames"), errors="coerce") or 0)
                if _n_frames <= 0 or (not np.isfinite(float(_lc_rms))):
                    logging.debug(
                        "[PDF] Skip LC page — no data: %s (n_frames=%d, lc_rms=%s)",
                        vsx_name,
                        _n_frames,
                        _lc_rms,
                    )
                    continue

                # Lightcurve image (prefer existing PNG -> jpeg; else generate from CSV)
                lc_png = Path(str(row.get("lc_png", "") or "")).expanduser()
                if not lc_png.is_absolute():
                    lc_png = (self.lc_dir / lc_png.name) if lc_png.name else lc_png
                lc_csv = Path(str(row.get("lc_csv", "") or "")).expanduser()
                if not lc_csv.is_absolute():
                    lc_csv = (self.lc_dir / lc_csv.name) if lc_csv.name else lc_csv

                lc_img = self._resolve_primary_lc_image(cid, lc_png, lc_csv)

                # Field image (field_map per target preferred)
                field_img = None
                fm = self.lc_dir / f"field_map_{cid}.png"
                if fm.exists():
                    field_img = fm
                else:
                    for cand in (
                        self.platesolve_dir / "masterstar_field.png",
                        self.photometry_dir / f"field_{cid}.png",
                        self.photometry_dir / f"field_{vsx_name}.png",
                    ):
                        if cand.exists():
                            field_img = cand
                            break
                field_img_jpg = self._prepare_jpeg(Path(field_img), self.cache_dir / f"field_{cid}.jpg", max_side_px=900, quality=72) if field_img else None

                n_sat = int(pd.to_numeric(row.get("n_saturated"), errors="coerce") or 0)
                comp_rows_pdf, comp_excluded_note = self._comp_rows_for_target(cid)
                lc_st = self._lc_stats_by_cid.get(cid, {})
                chk_rep = self._check_star_report_for(cid)
                star_data = {
                    "catalog_id": cid,
                    "vsx_name": vsx_name,
                    "vsx_type": row.get("vsx_type", ""),
                    "zone_flag": row.get("zone_flag", ""),
                    "bp_rp": row.get("bp_rp", float("nan")),
                    "b_v": row.get("b_v", float("nan")),
                    "lc_rms": row.get("lc_rms_ooe", row.get("lc_rms", float("nan"))),
                    "lc_rms_full": row.get("lc_rms", float("nan")),
                    "good_comp": row.get("n_stability_good", row.get("n_good_comp", 0)),
                    "n_stability_good": row.get("n_stability_good", row.get("n_good_comp", 0)),
                    "aperture_px": row.get("aperture_px", float("nan")),
                    "n_saturated": n_sat,
                    "n_points": int(lc_st.get("n_points", 0) or 0),
                    "merr_med": lc_st.get("merr_med", float("nan")),
                    "lc_quality_flag": row.get("lc_quality_flag", ""),
                    "trust": row.get("trust", ""),
                    "trust_reason": row.get("trust_reason", ""),
                    "check_star": chk_rep,
                    "variability_line": self._ground_variability_line(cid, str(row.get("vsx_type", "") or "")),
                    "lc_img": str(lc_img) if lc_img is not None else "",
                    "field_img": str(field_img_jpg) if field_img_jpg is not None else "",
                    "comp_rows": comp_rows_pdf,
                    "comp_excluded_note": comp_excluded_note,
                }
                if self._is_sparse_star_data(star_data):
                    sparse_buf.append(star_data)
                    if len(sparse_buf) >= 2:
                        self._report_per_star_compact_page(c, list(sparse_buf[:2]))
                        sparse_buf = sparse_buf[2:]
                else:
                    if sparse_buf:
                        self._report_per_star_compact_page(c, list(sparse_buf))
                        sparse_buf = []
                    self._report_per_star_page(c, star_data)
            except Exception as exc_star:  # noqa: BLE001
                logging.warning("PDF: skip star (%s): %s", row.get("vsx_name", ""), exc_star)
                continue

        if sparse_buf:
            self._report_per_star_compact_page(c, list(sparse_buf))

        # 3b) Detail pages for "new" variability self.candidates (no variability catalog matches)
        if self._candidates_set and self._crossmatch_bullets:
            for cid0 in [str(x).strip() for x in (self.candidates or []) if str(x).strip()]:
                cid_key = self._norm_cid(cid0)
                bullets = str(self._crossmatch_bullets.get(cid0, self._crossmatch_bullets.get(cid_key, "")) or "")
                if self._should_trigger_tess_report(bullets):
                    row_pdf = self._get_candidate_row_pdf(self._var_results, str(cid0))
                    self._draw_candidate_detail_page(
                        c,
                        cid=str(cid0),
                        candidate_row=row_pdf,
                        bullets=bullets,
                    )

        self._report_hockey_stick(c)
        self._report_candidates_table(c)
        self._report_tess_section(c)
        self._report_abbreviations(c)

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
    photometry_method: str = "aperture",
    active_methods: list[str] | None = None,
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
        logging.warning("reportlab is not installed, skipping PDF (%s)", exc)
        return None

    FONT_REG, FONT_BOLD, FONT_OBL = _register_pdf_unicode_fonts()

    _pm = str(photometry_method or "aperture").strip().lower()
    _methods = list(active_methods) if active_methods else None
    if _methods is None:
        try:
            from config import AppConfig

            _cfg_pdf = AppConfig()
            _lc_dir_pdf = Path(draft_dir) / "platesolve" / str(obs_group) / "photometry" / "lightcurves"
            _have_psf_lc = any(_lc_dir_pdf.glob("lightcurve_*_psf.csv")) or any(
                _lc_dir_pdf.glob("lightcurve_*_adaptive.csv")
            )
            _methods = active_report_methods(
                _cfg_pdf,
                have_psf_cols=_have_psf_lc
                or bool(getattr(_cfg_pdf, "psf_photometry_enabled", False))
                or bool(getattr(_cfg_pdf, "psf_adaptive_enabled", False)),
            )
        except Exception:  # noqa: BLE001
            _methods = ["aperture"]
    _title = method_report_title(
        str(report_title or "VYVAR \u2014 Summary Measure Report"),
        _pm,
        active_methods=_methods,
    )

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
        photometry_method=_pm,
        active_methods=_methods,
    )
    builder._verify_overflow = bool(verify_overflow)
    result = builder.build_pdf()
    if verify_overflow:
        generate_photometry_report.last_overflow_violations = builder.overflow_violation_count  # type: ignore[attr-defined]
    return result


generate_photometry_report.last_overflow_violations = 0  # type: ignore[attr-defined]


def generate_all_method_photometry_reports(
    draft_dir: Path,
    obs_group: str,
    *,
    output_pdf: Path | None = None,
    base_report_title: str = "VYVAR \u2014 Summary Measure Report",
    **kwargs: Any,
) -> list[Path]:
    """Build one PDF per active photometry method (aperture-only keeps legacy path)."""
    try:
        from config import AppConfig
    except Exception:  # noqa: BLE001
        p = generate_photometry_report(
            draft_dir,
            obs_group,
            output_pdf,
            report_title=base_report_title,
            **kwargs,
        )
        return [p] if p is not None else []

    cfg = AppConfig()
    lc_dir = Path(draft_dir) / "platesolve" / str(obs_group) / "photometry" / "lightcurves"
    have_psf_lc = any(lc_dir.glob("lightcurve_*_psf.csv")) or any(
        lc_dir.glob("lightcurve_*_adaptive.csv")
    )
    methods = active_report_methods(
        cfg,
        have_psf_cols=have_psf_lc
        or bool(getattr(cfg, "psf_photometry_enabled", False))
        or bool(getattr(cfg, "psf_adaptive_enabled", False)),
    )
    paths: list[Path] = []
    for method in methods:
        out = (
            Path(output_pdf)
            if (output_pdf is not None and method == "aperture" and len(methods) == 1)
            else pdf_report_path(
                draft_dir,
                obs_group,
                method,
                active_methods=methods,
            )
        )
        p = generate_photometry_report(
            draft_dir,
            obs_group,
            out,
            report_title=base_report_title,
            photometry_method=method,
            active_methods=methods,
            **kwargs,
        )
        if p is not None:
            paths.append(Path(p))
    return paths

