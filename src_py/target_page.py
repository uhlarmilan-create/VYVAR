"""Per-target page data (AP-TAB-2026-10): one source for the PDF star page and the
Aperture Photometry tab.

``load_frames`` reads and prepares the photometry tables of one observation group exactly
as the PDF report uses them; ``TargetPages`` holds them in memory together with the
per-target files (light curve, check star, comp quality) and builds the facts of a
target page (header, trust line, facts list, comparison rows, zero-point note).
No reportlab and no Streamlit here.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gaia_catalog_id import read_vyvar_csv

from mag_constants import MAG_ERR_SCALE

_GAIA_ID_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}


def _norm_cid_decimal(x: Any) -> str:
    """Normalize Gaia ``catalog_id`` to canonical digit string (report / joins)."""
    s = str(x or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    try:
        return str(int(Decimal(s)))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        try:
            return str(int(s))
        except (ValueError, TypeError):
            return s


class _ReportConfigView:
    """Cfg-like view: provenance ``config_snapshot`` first, live ``AppConfig`` for missing keys."""

    def __init__(self, snapshot: dict[str, Any], live: Any | None) -> None:
        self._snapshot = dict(snapshot)
        self._live = live

    def __getattr__(self, name: str) -> Any:
        if name in self._snapshot:
            return self._snapshot[name]
        if self._live is not None:
            return getattr(self._live, name)
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        if self._snapshot:
            out = dict(self._snapshot)
            if self._live is not None and hasattr(self._live, "to_dict"):
                for k, v in self._live.to_dict().items():
                    out.setdefault(k, v)
            return out
        if self._live is not None and hasattr(self._live, "to_dict"):
            return dict(self._live.to_dict())
        return {}


def resolve_report_config(
    pipeline_meta: dict[str, Any] | None,
    live: Any,
) -> tuple[Any | None, str]:
    """PDF cfg source: run ``provenance.config_snapshot`` when present (``live`` fills keys it lacks), else
    ``live`` - the caller's config (the run's, or the GUI's), never a fresh read (CONFIG-ONE-2026-10)."""
    snapshot: dict[str, Any] | None = None
    if isinstance(pipeline_meta, dict):
        prov = pipeline_meta.get("provenance")
        if isinstance(prov, dict):
            raw = prov.get("config_snapshot")
            if isinstance(raw, dict) and raw:
                snapshot = raw
    if snapshot is not None:
        return _ReportConfigView(snapshot, live), "run snapshot"
    return live, "live (no run snapshot)"


def vsx_type_sort_rank(val: Any) -> int:
    """Sort key: EA, EB, EW, ROT, VAR, then other non-empty types, then unknown/nan."""
    s0 = str(val if val is not None else "").strip().upper()
    if not s0 or s0 in ("NAN", "NONE", "-"):
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


def load_frames(platesolve_dir: Path, live_cfg: Any) -> dict[str, Any]:
    """Summary, active targets and comparison stars of one group, prepared as the report uses
    them: ``_cid`` keys, target metadata and exoplanet-host columns merged into the summary,
    rows sorted by VSX type then lc_rms; plus pipeline_meta and the run config."""
    platesolve_dir = Path(platesolve_dir)
    photometry_dir = platesolve_dir / "photometry"
    summary_csv = photometry_dir / "photometry_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing photometry_summary.csv: {summary_csv}")
    pipeline_meta: dict[str, Any] = {}
    _pm_path = photometry_dir / "pipeline_meta.json"
    if _pm_path.is_file():
        try:
            pipeline_meta = json.loads(_pm_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pipeline_meta = {}
    comp_csv = photometry_dir / "comparison_stars_per_target.csv"
    at_csv = photometry_dir / "active_targets.csv"

    summary_df = pd.read_csv(summary_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    from comp_table import load_comp_table, used  # noqa: PLC0415

    comp_df = used(load_comp_table(comp_csv)) if comp_csv.exists() else pd.DataFrame()
    at_df = pd.read_csv(at_csv, low_memory=False, dtype=_GAIA_ID_DTYPE) if at_csv.exists() else pd.DataFrame()

    if "catalog_id" in summary_df.columns:
        summary_df["_cid"] = summary_df["catalog_id"].map(_norm_cid_decimal)
    if "catalog_id" in at_df.columns:
        at_df["_cid"] = at_df["catalog_id"].map(_norm_cid_decimal)
    if "target_catalog_id" in comp_df.columns:
        comp_df["_tcid"] = comp_df["target_catalog_id"].map(_norm_cid_decimal)
    elif "catalog_id" in comp_df.columns:
        comp_df["_tcid"] = comp_df["catalog_id"].map(_norm_cid_decimal)

    if not at_df.empty and "_cid" in at_df.columns and "_cid" in summary_df.columns:
        meta_cols = [c for c in ("vsx_type", "zone_flag", "bp_rp", "vsx_name") if c in at_df.columns]
        meta = at_df[["_cid"] + meta_cols].drop_duplicates("_cid")
        summary_df = summary_df.merge(meta, how="left", on="_cid", suffixes=("", "_at"))

    from pipeline_ui_helpers import resolve_masterstars_metadata_csv  # noqa: PLC0415

    ms_csv = resolve_masterstars_metadata_csv(platesolve_dir)
    if ms_csv is not None and "_cid" in summary_df.columns:
        try:
            ms_df = pd.read_csv(ms_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
            if "catalog_id" in ms_df.columns:
                ms_df["_cid"] = ms_df["catalog_id"].map(_norm_cid_decimal)
                exo_meta = [
                    c
                    for c in ("exo_host_obj_id", "exo_host_name", "exo_cat_source", "exo_disposition",
                              "exo_match_sep_arcsec")
                    if c in ms_df.columns
                ]
                if exo_meta:
                    summary_df = summary_df.merge(
                        ms_df[["_cid"] + exo_meta].drop_duplicates("_cid"), how="left", on="_cid", suffixes=("", "_ms")
                    )
        except (OSError, ValueError):
            pass

    if "vsx_type" in summary_df.columns:
        summary_df["_vsx_rank"] = summary_df["vsx_type"].map(vsx_type_sort_rank)
    else:
        summary_df["_vsx_rank"] = 99
    if "lc_rms" in summary_df.columns:
        summary_df["_lc_rms"] = pd.to_numeric(summary_df["lc_rms"], errors="coerce")
        summary_df = summary_df.sort_values(["_vsx_rank", "_lc_rms"], ascending=[True, True], na_position="last")
    else:
        summary_df = summary_df.sort_values("_vsx_rank", ascending=True, na_position="last")

    cfg, cfg_label = resolve_report_config(pipeline_meta, live_cfg)
    return {
        "summary_df": summary_df, "at_df": at_df, "comp_df": comp_df,
        "pipeline_meta": pipeline_meta, "cfg": cfg, "cfg_label": cfg_label,
    }


def _int0(v: Any) -> int:
    """Integer value of a summary cell; 0 when missing or not a number."""
    x = pd.to_numeric(v, errors="coerce")
    return int(x) if pd.notna(x) else 0


def star_fields(row: Mapping[str, Any], lc_stats: Mapping[str, Any]) -> dict[str, Any]:
    """Summary-row facts of one target used on its page (no image paths)."""
    cid = str(row.get("_cid", "") or _norm_cid_decimal(row.get("catalog_id", "")))
    return {
        "catalog_id": cid,
        "vsx_name": str(row.get("vsx_name", cid) or cid),
        "vsx_type": row.get("vsx_type", ""),
        "zone_flag": row.get("zone_flag", ""),
        "bp_rp": row.get("bp_rp", float("nan")),
        "lc_rms": row.get("lc_rms_ooe", row.get("lc_rms", float("nan"))),
        "lc_rms_full": row.get("lc_rms", float("nan")),
        "good_comp": row.get("n_stability_good", row.get("n_good_comp", 0)),
        "n_stability_good": row.get("n_stability_good", row.get("n_good_comp", 0)),
        "aperture_px": row.get("aperture_px", float("nan")),
        "n_saturated": _int0(row.get("n_saturated")),
        "n_points": int(lc_stats.get("n_points", 0) or 0),
        "merr_med": lc_stats.get("merr_med", float("nan")),
        "lc_quality_flag": row.get("lc_quality_flag", ""),
        "trust": row.get("trust", ""),
        "trust_reason": row.get("trust_reason", ""),
        "exo_host_name": row.get("exo_host_name", ""),
        "exo_cat_source": row.get("exo_cat_source", ""),
        "exo_disposition": row.get("exo_disposition", ""),
        "psf_ratio": row.get("psf_ratio", float("nan")),
        "psf_scale": row.get("psf_scale", float("nan")),
        "target_blend_frac": row.get("target_blend_frac", float("nan")),
        "dilution_factor": row.get("dilution_factor", float("nan")),
        "dilution_delta_mag": row.get("dilution_delta_mag", float("nan")),
        "gs11_dilution_skipped": row.get("gs11_dilution_skipped", False),
    }


class TargetPages:
    """In-memory data of one observation group for target pages (PDF and UI)."""

    def __init__(
        self,
        platesolve_dir: Path,
        *,
        summary_df: pd.DataFrame,
        at_df: pd.DataFrame,
        comp_df: pd.DataFrame,
        cfg: Any | None,
    ) -> None:
        self.platesolve_dir = Path(platesolve_dir)
        self.photometry_dir = self.platesolve_dir / "photometry"
        self.lc_dir = self.photometry_dir / "lightcurves"
        self.summary_df = summary_df
        self.at_df = at_df
        self.comp_df = comp_df
        self.cfg = cfg
        self._cq: dict[str, dict[str, Any]] = {}
        self._lc: dict[str, pd.DataFrame | None] = {}
        self._chk: dict[str, pd.DataFrame | None] = {}
        self._gspc_rows_cache: dict[str, Any] | None = None
        self._ms_g_bprp_map: dict[str, tuple[float, float]] | None = None
        self._ms_xyg: np.ndarray | None = None
        self._ms_ids: np.ndarray = np.array([], dtype=str)
        self._manifest: dict[str, Any] | None = None
        self._manifest_read = False
        self.lc_stats_by_cid = self._build_target_lc_stats()

    # --- per-target files (read once) ---------------------------------------------------------
    def comp_quality(self, cid: str) -> dict[str, Any]:
        if cid not in self._cq:
            import star_page_m3 as m3  # noqa: PLC0415

            self._cq[cid] = m3.load_comp_quality(self.lc_dir, cid)
        return self._cq[cid]

    def lc_df(self, cid: str) -> pd.DataFrame | None:
        if cid not in self._lc:
            p = self.lc_dir / f"lightcurve_{cid}.csv"
            try:
                self._lc[cid] = pd.read_csv(p, low_memory=False) if p.is_file() else None
            except (OSError, ValueError):
                self._lc[cid] = None
        return self._lc[cid]

    def check_df(self, cid: str) -> pd.DataFrame | None:
        if cid not in self._chk:
            from check_star_kmag import check_kmag_sidecar_path  # noqa: PLC0415

            p = check_kmag_sidecar_path(self.lc_dir, cid)
            try:
                self._chk[cid] = pd.read_csv(p, low_memory=False) if p.is_file() else None
            except (OSError, ValueError):
                self._chk[cid] = None
        return self._chk[cid]

    def export_record(self, cid: str) -> dict[str, Any] | None:
        """G11-03: what the exporter wrote for this target (``export_manifest.json``); None when not exported."""
        if not self._manifest_read:
            from export_reports import load_export_manifest  # noqa: PLC0415

            self._manifest, self._manifest_read = load_export_manifest(self.photometry_dir), True
        return (self._manifest or {}).get(_norm_cid_decimal(cid))

    def export_mag_column(self, cid: str) -> str:
        """Magnitude column of the target's export (differential targets: delta_mag)."""
        from photometry_lightcurve import export_mag_column  # noqa: PLC0415

        df = self.lc_df(cid)
        return export_mag_column(df if df is not None else pd.DataFrame(), differential=self.uncalibrated(cid))

    def preload(self, cids: list[str]) -> None:
        """Read every target's files now so switching targets never touches the disk."""
        for cid in cids:
            self.comp_quality(cid)
            self.lc_df(cid)
            self.check_df(cid)
        self.gspc_rows()
        self._ms_g_bprp("")
        self.field_neighbours(0.0, 0.0, 0.0)

    # --- group-level -------------------------------------------------------------------------
    def _build_target_lc_stats(self) -> dict[str, dict[str, Any]]:
        """Points used and median error (mag) per target, from its light curve in ``lc_dir``.
        The canonical ``lightcurve_<cid>.csv`` is used, not the absolute path stored in the
        summary (that path breaks when a draft is moved or opened on another computer)."""
        stats: dict[str, dict[str, Any]] = {}
        if "catalog_id" not in self.summary_df.columns and "_cid" not in self.summary_df.columns:
            return stats
        for _, row in self.summary_df.iterrows():
            cid = _norm_cid_decimal(row.get("catalog_id", row.get("_cid", "")))
            if not cid:
                continue
            df = self.lc_df(cid)
            if df is None:
                n_fr = _int0(row.get("n_frames"))
                stats[cid] = {"n_points": n_fr if n_fr > 0 else 0, "merr_med": float("nan")}
                continue
            # REPORT-AUDIT-2026-10 / G11-12: points = the epochs the exports carry (export_row_mask),
            # the same count as the target page's "Frames" line
            from photometry_lightcurve import export_row_mask  # noqa: PLC0415

            df = df[export_row_mask(df, mag_column=self.export_mag_column(cid))]
            n_pt = int(len(df))
            err_col = "err" if "err" in df.columns else ("magerr" if "magerr" in df.columns else "MAGERR")
            merr = float("nan")
            if err_col in df.columns:
                merr = float(np.nanmedian(pd.to_numeric(df[err_col], errors="coerce")))
                if err_col == "err":
                    # FIX-ERRUNIT-2026-10: LC 'err' is the relative flux error; MERR is shown in
                    # mag like the AAVSO / VarAstro exports (x 2.5/ln10).
                    merr *= MAG_ERR_SCALE
            stats[cid] = {"n_points": n_pt, "merr_med": merr}
        return stats

    def gspc_rows(self) -> dict[str, Any]:
        """GSPC cache rows of this group (empty when the module did not run)."""
        if self._gspc_rows_cache is None:
            try:
                from gspc_zp import CACHE_NAME, read_cache  # noqa: PLC0415

                self._gspc_rows_cache = read_cache(self.photometry_dir / CACHE_NAME)
            except Exception:  # noqa: BLE001 - optional module output
                self._gspc_rows_cache = {}
        return self._gspc_rows_cache

    def _ms_g_bprp(self, cid: str) -> tuple[float, float]:
        """(G, BP-RP) of a Gaia source from masterstars_full_match.csv (nan when absent)."""
        if self._ms_g_bprp_map is None:
            mp: dict[str, tuple[float, float]] = {}
            try:
                _d = read_vyvar_csv(self.platesolve_dir / "masterstars_full_match.csv",
                                 usecols=lambda c: c in ("catalog_id", "mag", "phot_g_mean_mag", "bp_rp"),
                                 dtype={"catalog_id": str}, low_memory=False)
                # G: phot_g_mean_mag when filled, else the catalogue mag (Gaia G in masterstars)
                _gm = pd.to_numeric(_d.get("mag"), errors="coerce")
                if "phot_g_mean_mag" in _d.columns:
                    _gm = pd.to_numeric(_d["phot_g_mean_mag"], errors="coerce").fillna(_gm)
                for cc, g, b in zip(_d["catalog_id"], _gm, pd.to_numeric(_d.get("bp_rp"), errors="coerce")):
                    k = _norm_cid_decimal(cc)
                    if k:
                        mp[k] = (float(g), float(b))
            except (OSError, ValueError, KeyError):
                mp = {}
            self._ms_g_bprp_map = mp
        return self._ms_g_bprp_map.get(_norm_cid_decimal(cid), (float("nan"), float("nan")))

    def field_neighbours(self, x: float, y: float, r: float, *, exclude_id: str = "") -> list[tuple[float, float, float]]:
        """(x, y, G) of the Gaia stars within r px of (x, y), for the cutout.

        BLEND-CUTOUT-2026-10: the Gaia cone of the field (``field_catalog_cone.csv``, the catalogue of
        the neighbour-light estimate) on the MASTERSTAR WCS, not only the DAO-detected stars; the
        target itself (``exclude_id``) is left out. (The masterstar read by name order put magnitudes
        into x - no neighbour was ever found.)"""
        if self._ms_xyg is None:
            self._ms_xyg, self._ms_ids = np.zeros((0, 3)), np.array([], dtype=str)
            try:
                from astropy.io import fits  # noqa: PLC0415
                from astropy.wcs import WCS  # noqa: PLC0415

                cat = read_vyvar_csv(self.platesolve_dir / "field_catalog_cone.csv", low_memory=False,
                                  dtype={"catalog_id": str})
                hdr = fits.getheader(self.platesolve_dir / "MASTERSTAR.fits")
                cx, cy = WCS(hdr).all_world2pix(pd.to_numeric(cat["ra_deg"], errors="coerce").to_numpy(),
                                                pd.to_numeric(cat["dec_deg"], errors="coerce").to_numpy(), 0)
                g = pd.to_numeric(cat["mag"], errors="coerce").to_numpy(float)
                ok = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(g)
                self._ms_xyg = np.c_[cx[ok], cy[ok], g[ok]]
                self._ms_ids = cat["catalog_id"].astype(str).map(_norm_cid_decimal).to_numpy()[ok]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logging.debug("[TARGET-PAGE] Gaia neighbours not available: %s", exc)
        a = self._ms_xyg
        if a.size == 0:
            return []
        m = (np.abs(a[:, 0] - x) <= r) & (np.abs(a[:, 1] - y) <= r)
        if exclude_id:
            m &= self._ms_ids != _norm_cid_decimal(exclude_id)
        return [(float(p[0]), float(p[1]), float(p[2])) for p in a[m]]

    # --- per-target facts --------------------------------------------------------------------
    def uncalibrated(self, cid: str) -> bool:
        """UNCAL-LC-2026-10: True when the target has no band zero point (band NONE)."""
        from gaia_bands import lc_uncalibrated  # noqa: PLC0415

        if not cid:
            return False
        _sb = None
        if "band_system" in self.summary_df.columns:
            _m = self.summary_df["catalog_id"].astype(str).map(_norm_cid_decimal) == _norm_cid_decimal(cid)
            if bool(_m.any()):
                _sb = str(self.summary_df.loc[_m, "band_system"].iloc[0])
        return lc_uncalibrated(self.lc_dir, str(cid), _sb)

    def gspc_info(self, cid: str) -> dict[str, Any]:
        """GSPC-ZP-2026-10: the gspc_zero_point record of the target (empty when absent)."""
        if not cid:
            return {}
        gz = self.comp_quality(cid).get("gspc_zero_point")
        return dict(gz) if isinstance(gz, dict) else {}

    def zero_point_note(self, cid: str) -> str:
        """ZP-ERR-2026-10: comps that set the absolute level and its uncertainty.
        GSPC-ZP-2026-10: plus calibration note A / B (gaia_bands.zero_point_note_text)."""
        from gaia_bands import zero_point_note_text  # noqa: PLC0415

        _gz = self.gspc_info(cid)
        _note = str(_gz.get("note") or "")
        raw = self.comp_quality(cid)
        if not raw:
            return ""
        zp = raw.get("zero_point")
        if not isinstance(zp, dict):
            return ""
        band = str(zp.get("band") or "")
        if band in ("", "NONE"):
            if _note == "A":
                _b = str(_gz.get("band") or "").strip() or "band"
                from gspc_zp import gspc_standard_available  # noqa: PLC0415

                if not gspc_standard_available(_gz, self._mag_gspc(cid)):
                    _gz = dict(_gz, C=None)  # same rule as the export: differential
                return zero_point_note_text("A", _b, _gz)
            return "Zero point: no comparison star with a valid band magnitude - magnitudes not calibrated."
        n = int(zp.get("n_comp") or 0)
        parts = [f"Zero point ({band}): {n} comparison star{'s' if n != 1 else ''} with a valid Gaia DR3 -> {band} magnitude"]
        if zp.get("n_eff") is not None:
            parts[0] += f" (n_eff {float(zp['n_eff']):.1f})"
        if zp.get("sigma_zp_model_mag") is not None:
            parts.append(f"sigma_ZP {1000 * float(zp['sigma_zp_model_mag']):.0f} mmag from the relation scatter "
                         f"{1000 * float(zp['sigma_rel_mag']):.0f} mmag / sqrt(n_eff)")
        if zp.get("sigma_zp_data_mag") is not None:
            parts.append(f"{1000 * float(zp['sigma_zp_data_mag']):.0f} mmag from their mutual scatter")
        _txt = "; ".join(parts) + "."
        if _note in ("B", "B-heuristic"):
            _txt += " " + zero_point_note_text(_note, band, _gz)
        return _txt

    def _mag_gspc(self, cid: str) -> np.ndarray:
        df = self.lc_df(cid)
        if df is None or "mag_gspc" not in df.columns:
            return np.array([np.nan])
        return pd.to_numeric(df["mag_gspc"], errors="coerce").to_numpy(dtype=float)

    def _resolve_check_kname(self, check_cid: str) -> str:
        cc = _norm_cid_decimal(check_cid)
        if not cc:
            return "-"
        # G11-40: the comp table has no vsx_name; the name comes from the active targets
        if not self.at_df.empty and "_cid" in self.at_df.columns and "vsx_name" in self.at_df.columns:
            m2 = self.at_df["_cid"].astype(str).eq(cc)
            if bool(m2.any()):
                vn2 = str(self.at_df.loc[m2, "vsx_name"].iloc[0] or "").strip()
                if vn2 and vn2.lower() not in ("nan", "none"):
                    return vn2
        return cc

    def check_star(self, target_cid: str) -> dict[str, str]:
        """Check star actually used (sidecar) - CHECK-FACTS-2026-10: name, full Gaia id, G, BP-RP,
        measured KMAG vs catalogue KMAG, scatter vs the expected scatter (median of the per-point
        KMAG errors). The page never re-selects a check star (CHECK-RULES-2026-10)."""
        out = {"kname": "-", "kmag": "-", "scatter": "-", "kcid": "", "g": "", "bp_rp": "",
               "kref": "", "kdelta": "", "err_med": "", "scatter_ratio": ""}
        cid = _norm_cid_decimal(target_cid)
        sdf = self.check_df(cid) if cid else None
        if sdf is None or sdf.empty or "kmag" not in sdf.columns:
            return out
        # G11-11: the check-star numbers of the export header and the trust (exported epochs, ddof=1)
        st = self.check_stats(cid)
        med, sc = st.median, st.scatter
        out["kmag"] = f"{med:.3f}" if np.isfinite(med) else "na"
        uncal = self.uncalibrated(cid)
        if uncal:
            # UNCAL-LC-2026-10: the check is verified on its differential magnitude;
            # its level is relative to the ensemble, not a catalogue magnitude.
            out["kmag"] = "- (not calibrated)"
        out["scatter"] = f"{sc:.4f} mag" if np.isfinite(sc) else "-"
        from check_star_kmag import load_check_star  # noqa: PLC0415

        chk = str(load_check_star(self.lc_dir, cid).get("catalog_id", "") or "")
        out["kname"] = self._resolve_check_kname(chk)
        kc = _norm_cid_decimal(chk)
        out["kcid"] = kc
        g_k, b_k = self._ms_g_bprp(kc) if kc else (float("nan"), float("nan"))
        out["g"] = f"{g_k:.2f}" if np.isfinite(g_k) else ""
        out["bp_rp"] = f"{b_k:.2f}" if np.isfinite(b_k) else ""
        if "kmag_err" in sdf.columns:
            em = st.err_med
            if np.isfinite(em) and em > 0:
                out["err_med"] = f"{em:.4f}"
                if np.isfinite(sc):
                    out["scatter_ratio"] = f"{sc / em:.1f}"
        if kc and not uncal and np.isfinite(med):
            _bm = self.comp_quality(cid).get("comp_band_mag")
            _kr = pd.to_numeric(_bm.get(kc), errors="coerce") if isinstance(_bm, dict) else float("nan")
            if np.isfinite(float(_kr)):
                out["kref"] = f"{float(_kr):.3f}"
                out["kdelta"] = f"{med - float(_kr):+.3f}"
        return out

    def check_stats(self, cid: str) -> Any:
        """``check_star_kmag.CheckStarStats`` of the target over its exported epochs (G11-11)."""
        from check_star_kmag import exported_check_stats  # noqa: PLC0415

        return exported_check_stats(self.lc_dir, cid, lc_df=self.lc_df(cid), sidecar=self.check_df(cid),
                                    mag_column=self.export_mag_column(cid))

    def summary_row(self, cid: str) -> dict[str, Any]:
        sd = self.summary_df
        if "_cid" not in sd.columns:
            return {}
        m = sd[sd["_cid"] == cid]
        return m.iloc[0].to_dict() if not m.empty else {}

    def at_row(self, cid: str, star_data: Mapping[str, Any]) -> dict[str, Any]:
        at_row: dict[str, Any] = {}
        if not self.at_df.empty and "_cid" in self.at_df.columns:
            _m = self.at_df[self.at_df["_cid"] == cid]
            if not _m.empty:
                at_row = _m.iloc[0].to_dict()
        for _k in ("vsx_name", "vsx_type", "exo_host_name", "exo_cat_source", "exo_disposition"):
            if not str(at_row.get(_k, "") or "").strip() and str(star_data.get(_k, "") or "").strip():
                at_row[_k] = star_data.get(_k)
        return at_row

    def page(self, star_data: Mapping[str, Any], *, obs_group: str) -> dict[str, Any]:
        """Content of one target page (STAR-PAGE-M3-2026-10, layout approved by Milan 2026-10-11)."""
        import star_page_m3 as m3  # noqa: PLC0415
        from dilution import gs11_dilution_from_summary  # noqa: PLC0415
        from gspc_zp import gspc_standard_available  # noqa: PLC0415

        cid = _norm_cid_decimal(str(star_data.get("catalog_id", "") or ""))
        at_row = self.at_row(cid, star_data)
        cq = self.comp_quality(cid)
        gz = cq.get("gspc_zero_point") if isinstance(cq.get("gspc_zero_point"), dict) else {}
        uncal = self.uncalibrated(cid)
        zp = cq.get("zero_point") if isinstance(cq.get("zero_point"), dict) else {}
        band = str(gz.get("band") or zp.get("band") or "").strip()
        if band.upper() == "NONE":
            band = str(gz.get("band") or "").strip()
        lc_csv = self.lc_dir / f"lightcurve_{cid}.csv"
        facts_lc = m3.lc_facts(lc_csv, mag_column=self.export_mag_column(cid)) if lc_csv.is_file() else {}

        # header
        _bp = float(pd.to_numeric(at_row.get("bp_rp", star_data.get("bp_rp")), errors="coerce"))
        _g = float(pd.to_numeric(at_row.get("mag"), errors="coerce"))
        sub = [m3.target_source_line(at_row)]
        if np.isfinite(_g):
            sub.append(f"G {_g:.2f}" + (f"  BP-RP {_bp:.2f}" if np.isfinite(_bp) else ""))
        sub.append(str(obs_group))
        if facts_lc.get("date_ut"):
            sub.append(str(facts_lc["date_ut"]))
        trust = str(star_data.get("trust", "") or "").strip().upper()
        trust_reason = str(star_data.get("trust_reason", "") or "").split(" - ")[0].strip()

        # note box + comparison rows
        note_parts = []
        zp_txt = self.zero_point_note(cid).strip()
        if zp_txt:
            note_parts.append(zp_txt)
        comp_sub = (self.comp_df[self.comp_df["_tcid"] == cid]
                    if (not self.comp_df.empty and "_tcid" in self.comp_df.columns) else pd.DataFrame())
        rows, excluded, hidden = m3.comp_rows(comp_sub, cq, band=band or "g", gspc_rows=self.gspc_rows(),
                                              target_bp_rp=_bp)
        if excluded:
            note_parts.append(f"{len(excluded)} comparison star(s) excluded from the ensemble: " + ", ".join(excluded) + ".")
        _sm = cq.get("comp_smart") if isinstance(cq.get("comp_smart"), dict) else {}
        # COMP-CATOFF-2026-10: exclusions for the catalogue offset are named separately
        _co = (_sm.get("catalog_offset") or {}).get("excluded") or {}
        _repl_sus = [c for c in (_sm.get("replaced") or []) if c not in _co]
        if _sm and (_sm.get("replaced") or _sm.get("kept_suspect") or _co):
            _t = []
            if _co:
                _t.append(f"{len(_co)} excluded for catalogue offset > "
                          f"{float((_sm.get('catalog_offset') or {}).get('max_mag') or 0):.2f} mag ("
                          + ", ".join(f"{c} {float(v):+.2f}" if v is not None else str(c) for c, v in _co.items()) + ")")
            if _repl_sus:
                _t.append(f"{len(_repl_sus)} suspect comp(s) dropped")
            if _sm.get("added"):
                _t.append(f"{len(_sm.get('added') or [])} replaced from the reserve")
            if _sm.get("kept_suspect"):
                _t.append(f"{len(_sm['kept_suspect'])} suspect comp(s) kept - fewer than 3 clean candidates")
            note_parts.append("Comparison set (SMART): " + "; ".join(_t) + ".")
        if hidden:
            note_parts.append(f"+{hidden} more comparison stars not shown.")
        tier_by_cid: dict[str, str] = {}
        if not comp_sub.empty and "catalog_id" in comp_sub.columns and "comp_tier" in comp_sub.columns:
            for c_id, t_v in zip(comp_sub["catalog_id"].map(_norm_cid_decimal), comp_sub["comp_tier"]):
                t_s = str(t_v or "").strip()
                tier_by_cid[c_id] = "" if t_s.lower() in ("nan", "none") else t_s

        # facts
        w_br = cq.get("broeg_weights") if isinstance(cq.get("broeg_weights"), dict) else {}
        _ra = float(pd.to_numeric(at_row.get("ra_deg"), errors="coerce"))
        _de = float(pd.to_numeric(at_row.get("dec_deg"), errors="coerce"))
        _am = facts_lc.get("airmass", (float("nan"), float("nan")))
        _r = facts_lc.get("r_ap_med", float("nan"))
        _k = facts_lc.get("k", float("nan"))
        _fw = facts_lc.get("fwhm_night", float("nan"))
        chk = self.check_star(cid)
        _ain = float(getattr(self.cfg, "annulus_inner_fwhm", 2.7) or 2.7) if self.cfg else 2.7
        _aout = float(getattr(self.cfg, "annulus_outer_fwhm", 5.2) or 5.2) if self.cfg else 5.2
        _lc_rms = float(pd.to_numeric(star_data.get("lc_rms"), errors="coerce"))
        _merr = float(pd.to_numeric(star_data.get("merr_med"), errors="coerce"))
        _gz_note = str(gz.get("note") or "")
        _gspc_std = bool(uncal and gspc_standard_available(gz, self._mag_gspc(cid)))
        if _gspc_std:
            zp_val = f"Gaia DR3 GSPC {band}, {int(gz.get('n_comp') or 0)} comps"
        elif uncal:
            zp_val = "NOT CALIBRATED (differential)"
        elif band:
            zp_val = f"Gaia DR3 -> {band}, {int(zp.get('n_comp') or 0)} comps"
            # REPORT-AUDIT-2026-10: a clear / tri-colour filter is on the band's zero point but
            # reported under its own AAVSO code (CV, TG, ...), untransformed
            from gaia_bands import filter_band_for_obs_group  # noqa: PLC0415

            _fbp = filter_band_for_obs_group(obs_group)
            if not _fbp.transformable and _fbp.aavso_code and _fbp.aavso_code != band:
                zp_val += f"; reported as {_fbp.aavso_code} (untransformed)"
        else:
            zp_val = "-"
        # G11-03: the files the exporter actually wrote (export_manifest.json), not a re-derived routing
        from export_reports import export_fact_text  # noqa: PLC0415

        exp_val = export_fact_text(self.export_record(cid))
        # PSF-LOCAL-2026-10: broader own PSF -> aperture was scaled
        _pr = float(pd.to_numeric(star_data.get("psf_ratio"), errors="coerce"))
        _ps = float(pd.to_numeric(star_data.get("psf_scale"), errors="coerce"))
        _psf_txt = ""
        if np.isfinite(_pr) and _pr > 1.25:
            _psf_txt = (f"; own PSF x{_pr:.2f} field, radius scaled" if np.isfinite(_ps) and _ps > 1.0
                        else f"; own PSF x{_pr:.2f} field (neighbour, not scaled)")
        # REPORT-AUDIT-2026-10: the photometry scales the target's annulus with its own PSF too
        _fw_t = _fw * (_ps if np.isfinite(_ps) and _ps > 0 else 1.0)
        _bf = float(pd.to_numeric(star_data.get("target_blend_frac"), errors="coerce"))
        # CHECK-FACTS-2026-10: full check-star facts
        _kn, _kc = str(chk.get("kname", "-") or "-"), str(chk.get("kcid", "") or "")
        _chk_facts: list[tuple[str, str]] = [
            ("Check star", _kn if (not _kc or _kn == _kc) else f"{_kn} (Gaia DR3 {_kc})"),
        ]
        if _kc and _kn == _kc:
            _chk_facts[0] = ("Check star", f"Gaia DR3 {_kc}")
        _gb = "  ".join(x for x in (f"G {chk['g']}" if chk.get("g") else "",
                                     f"BP-RP {chk['bp_rp']}" if chk.get("bp_rp") else "") if x)
        if _gb:
            _chk_facts.append(("Check G / colour", _gb))
        if chk.get("kref"):
            _chk_facts.append(("Check KMAG", f"{chk.get('kmag')} meas. / {chk['kref']} cat. ({band or 'g'}), "
                                             f"diff {chk.get('kdelta')}"))
        else:
            _chk_facts.append(("Check KMAG", str(chk.get("kmag", "-"))))
        _sc_t = str(chk.get("scatter", "-")).replace(" mag", "")
        if chk.get("err_med"):
            _sc_t += f" (expected {chk['err_med']}, x{chk.get('scatter_ratio') or '-'})"
        _chk_facts.append(("Check scatter", _sc_t))
        warn_rows: set[str] = set()
        try:
            if float(chk.get("scatter_ratio") or "nan") > 2.0:
                warn_rows.add("Check scatter")
        except ValueError:
            pass
        if _psf_txt:
            warn_rows.add("Aperture")
        if np.isfinite(_bf) and _bf > 0.01:
            warn_rows.add("Blend")
            _chk_facts.append(("Blend", f"neighbours add {100.0 * _bf:.1f} % light in the aperture"))
        # GS11-2026-10: Gaia neighbour light removed from the magnitudes (flux, frame by frame)
        _gs11 = gs11_dilution_from_summary(star_data, self.cfg)
        if _gs11 is not None:
            warn_rows.add("Flux dilution")
            _dm_t = f", magnitudes corrected by {_gs11[1]:+.3f} mag" if np.isfinite(_gs11[1]) else ""
            _chk_facts.append(("Flux dilution", f"Gaia D = {_gs11[0]:.3f}{_dm_t} (AAVSO NOTES GS11D)"))
        facts = [
            ("Gaia DR3", cid),
            ("RA / Dec", m3.ra_dec_text(_ra, _de)),
            ("Frames", f"{int(facts_lc.get('n_exported', 0))} exported / "
                       f"{int(facts_lc.get('n_total', 0))}  ({int(facts_lc.get('n_excluded', 0))} excluded, "
                       f"{int(facts_lc.get('n_noted', 0))} flagged in NOTES)"),
            ("Span / airmass", f"{facts_lc.get('span_h', float('nan')):.1f} h / {_am[0]:.2f}-{_am[1]:.2f}"),
            ("Aperture", (f"{_r:.1f} px ({_k:.2f} x FWHM)" if np.isfinite(_r) else "-") + _psf_txt),
            ("Sky annulus", f"{max(_ain * _fw_t, _r if np.isfinite(_r) else 0.0):.1f}-{_aout * _fw_t:.1f} px"
             if np.isfinite(_fw_t) else "-"),
            ("rms OOE / MERR", f"{_lc_rms:.3f} / {_merr:.4f} mag" if np.isfinite(_lc_rms) else "-"),
            *_chk_facts,
            ("Comps", f"{len(rows)}  n_eff {m3.n_eff(w_br):.1f}" if w_br else f"{len(rows)}"),
            ("Zero point", zp_val),
        ]
        show_gspc = _gz_note in ("A", "B") and gz.get("C") is not None
        if show_gspc:
            _sg = gz.get("sigma_data") if gz.get("sigma_data") is not None else gz.get("sigma_model")
            _mm = gz.get("median_mag")
            facts.append((f"{band} GSPC", (f"{float(_mm):.3f}" if _mm is not None else "-")
                          + (f" +- {float(_sg):.3f}" if _sg is not None else "") + f" ({int(gz.get('n_comp') or 0)} comps)"))
        facts.append(("Export", exp_val))
        warn = {a for a, b in facts if "NOT CAL" in b or a in warn_rows}
        return {
            "cid": cid,
            "name": m3.target_display_name(at_row, cid),
            "subline": [x for x in sub if x],
            "trust": trust,
            "trust_reason": trust_reason,
            "facts": facts,
            "warn": warn,
            "comp_rows": rows,
            "comp_tier": tier_by_cid,
            "note_text": "  ".join(note_parts),
            "band": band,
            "uncalibrated": uncal,
            "show_gspc": show_gspc,
            "lc_csv": lc_csv,
            "check_csv": self.lc_dir / f"check_kmag_{cid}.csv",
            "at_row": at_row,
            "ra_deg": _ra,
            "dec_deg": _de,
            "x": float(pd.to_numeric(at_row.get("x"), errors="coerce")),
            "y": float(pd.to_numeric(at_row.get("y"), errors="coerce")),
            "r_ap": _r,
            "r_in": max(_ain * _fw_t, _r) if np.isfinite(_fw_t) else float("nan"),
            "r_out": _aout * _fw_t if np.isfinite(_fw_t) else float("nan"),
            "fwhm_night": _fw,
        }


def load_target_pages(platesolve_dir: Path, live_cfg: Any) -> TargetPages:
    """TargetPages for one observation group, tables prepared as for the PDF report."""
    fr = load_frames(platesolve_dir, live_cfg)
    return TargetPages(platesolve_dir, summary_df=fr["summary_df"], at_df=fr["at_df"], comp_df=fr["comp_df"],
                       cfg=fr["cfg"])
